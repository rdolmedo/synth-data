"""tbsched as the admission client of a replay of recorded traffic (#451 validation step 2; design arms 2a-2d).

A `SchedClient` decides which new attempts start, on which engine, and whether a turn may go now. A replay driver needs
five calls: `scrape()` every ~15 s, `decide()` every ~60 s, `next_start(task, attempt)` until it returns None,
`await before_turn(...)` / `after_turn(...)` around every request, and `ended(...)` when an attempt ends. `replay()`
below is such a driver over synth.bench's Replay (its corpus, targets and `call`). synth.bench's own --adapt-hours
loop can call the same five, so the bench.py hook is a few lines.

Policies (`policy`):
* `sched`: tbsched: per-engine aimd limits (control.aimd), credit per engine (place.allocate / withdraw), routing to
  the engine with the most unspent credit (hash tie-break with the namespace), optional collapse gate (#477);
* `hash`: the 11-15 pi_eval rule: engine = sha256(task/attempt) over the targets, one run limit, a start burst per 60 s
  (32, as the M5 ladder launched its runs; pi_eval's own default is 8);
* `proto`: `hash` + the M5 autoscale (legacy.m5_verdict / m5_limit every 120 s on the run limit).
In a replay the "task" of an attempt is its recorded trajectory: two attempts replaying one trajectory hash to the same
engine, as parallel rounds of one task did on 11-15.

  python -m synth.bench --adapt-hours 0.5 --sched sched|hash|proto [--sched-gate] [--sched-limit 96] \\
      [--sched-aimd JSON (e.g. {"policy": "guard"})] [--sched-stationary] [--gap-scale X] \\
      --corpus GLOB --corpus-cache FILE --bases URL,URL --served-name NAME --out ROWS.jsonl
"""
import asyncio
import collections
import json
import random
import time
import urllib.request

from tbsched import control, legacy, place
from tbsched.gate import CollapseGate

RUN = 'replay'


def fetch(url, timeout=5.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


class SchedClient:
    def __init__(self, targets, policy='sched', *, aimd=None, placep=None, gate=False, namespace=RUN, limit=96,
                 burst=32, sibling_hold=True, demand=10 ** 9, scrape=fetch, now=time.time, max_wait=1800.0):
        """targets: [{'metrics': url of its /metrics, 'match': labels of its series or None}] in the replay's order."""
        assert policy in ('sched', 'hash', 'proto'), policy
        self.targets, self.policy, self.now, self.fetch = targets, policy, now, scrape
        self.aimd, self.placep, self.ns, self.demand = aimd or {}, placep or {}, namespace, demand
        n = len(targets)
        self.windows = [control.Window(120.0) for _ in range(n)]
        self.live, self.started, self.credit = [0] * n, [0] * n, [0] * n
        self.fresh = [collections.deque() for _ in range(n)]
        self.st = [{} for _ in range(n)]
        self.gates = [CollapseGate() for _ in range(n)] if gate else None
        self.max_wait = max_wait
        self.limit, self.burst, self.sibling_hold, self.calm, self.next_scale = limit, burst, sibling_hold, 0, 0.0
        self.starts60 = collections.deque()
        self.grants = collections.deque()   # (time, credits granted): the rolling per-minute burst
        self.decisions = 0

    # ------------------------------------------------------------------------------------------------- observe
    def fetch_all(self):
        """The targets' /metrics texts (None where a fetch failed). Safe in a thread: touches no client state."""
        out = []
        for tg in self.targets:
            try:
                out.append(self.fetch(tg['metrics']))
            except Exception:   # noqa: BLE001 -- a missing sample is a stale window, not a crash
                out.append(None)
        return out

    def scrape(self, texts=None):
        """Apply a scrape (texts from fetch_all, else fetched now). Run it on the thread that calls before_turn /
        after_turn / ended: it ticks the gates."""
        texts = self.fetch_all() if texts is None else texts
        t = self.now()
        for i, (tg, text) in enumerate(zip(self.targets, texts)):
            if text is None:
                continue
            try:
                self.windows[i].add(t, control.parse_metrics(text, tg.get('match')))
            except Exception:   # noqa: BLE001
                continue
        if self.gates:
            for i, g in enumerate(self.gates):
                g.tick(t, loaded=control.loaded(self.obs(i), {**control.AIMD, **self.aimd}))

    def obs(self, i):
        now = self.now()
        while self.fresh[i] and now - self.fresh[i][0] > 120:
            self.fresh[i].popleft()
        g = self.gates[i] if self.gates else None
        return self.windows[i].obs(sessions=self.live[i], fresh=len(self.fresh[i]),
                                   reuse=g.g.eff if g is not None else None)

    # ------------------------------------------------------------------------------------------------- decide
    def decide(self):
        """One scheduler tick; returns a row for the run log."""
        now, self.decisions = self.now(), self.decisions + 1
        obs = [self.obs(i) for i in range(len(self.targets))]
        row = {'t': round(now), 'policy': self.policy, 'live': list(self.live),
               'running': [round(o.running, 1) for o in obs], 'waiting': [o.waiting for o in obs],
               'kv': [round(o.kv, 3) for o in obs], 'hit': [None if o.hit is None else round(o.hit, 3) for o in obs],
               'gen_tok_s': [round(o.gen_tok_s) for o in obs]}
        if self.policy == 'sched':
            engines, limits, states, phases = {}, [], [], []
            for i, o in enumerate(obs):
                res = control.limit_step(self.st[i], o, self.aimd, o.sessions, now,
                                   committed=o.sessions + min(max(0, self.credit[i] - self.started[i]), o.fresh))
                limits.append(res['limit'])
                states.append(res['state'])
                phases.append(res.get('phase'))
                g = self.gates[i] if self.gates else None
                if g is not None:
                    ev = control.hit_evidence(o, {**control.AIMD, **self.aimd})
                    if ev is not None and ev < g.collapse_eff:
                        g.trigger(now, 'hit_collapse')
                gs = g.g.stats() if g is not None else {}
                engines[str(i)] = place.Engine(str(i), 'g', res['limit'],
                                               open=not o.stale and not (gs.get('ctx_waiting') or gs.get('ctx_parked')))
            self.engines = engines
            self.refill()
            row.update(limit=limits, state=states, phase=phases, unspent=[c - s for c, s in zip(self.credit, self.started)],
                       gate=[g.mode for g in self.gates] if self.gates else None)
        elif self.policy == 'proto' and now >= self.next_scale:
            self.next_scale = now + 120
            verdict = legacy.m5_verdict(obs, mixed=False, sibling_hold=self.sibling_hold)
            self.calm = self.calm + 1 if verdict == 'calm' else 0
            self.limit = legacy.m5_limit(self.limit, verdict, self.calm, sum(self.live))
            row.update(verdict=verdict)
        if self.policy != 'sched':
            row.update(limit=self.limit)
        return row

    def refill(self):
        """Withdraw and allocate credit at the current limits (no limit change): every scrape, so slots that sessions
        free between decisions are refilled within seconds, not at the next decision."""
        if self.policy != 'sched' or not getattr(self, 'engines', None):
            return
        for e, eng in self.engines.items():
            g = self.gates[int(e)] if self.gates else None
            gs = g.g.stats() if g is not None else {}
            eng.open = not self.obs(int(e)).stale and not (gs.get('ctx_waiting') or gs.get('ctx_parked'))
        for e, n in place.withdraw([self.run_view()], self.engines).get(RUN, {}).items():
            self.credit[int(e)] -= n
        now = self.now()
        while self.grants and now - self.grants[0][0] >= 60:     # the burst is per rolling minute, decisions and
            self.grants.popleft()                                 # refills together (it staggers cold prefills)
        left = {'g': place.burst_budget(self.engines, self.placep)['g'] - sum(n for _, n in self.grants)}
        add = place.allocate([self.run_view()], self.engines, self.placep, burst_left=left).get(RUN, {})
        for e, n in add.items():
            self.credit[int(e)] += n
        if add:
            self.grants.append((now, sum(add.values())))

    def run_view(self):
        es = [str(i) for i in range(len(self.targets))]
        return place.Run(RUN, 'agentic', es, self.demand, credit={str(i): c for i, c in enumerate(self.credit)},
                         starts={str(i): s for i, s in enumerate(self.started)},
                         live={str(i): n for i, n in enumerate(self.live)})

    def room_now(self):
        """Attempts that may start now: unspent credit (sched) or the run limit's free room (hash / proto)."""
        if self.policy == 'sched':
            return sum(max(0, c - s) for c, s in zip(self.credit, self.started))
        return max(0, min(self.limit - sum(self.live), self.burst))

    def next_start(self, task, attempt):
        """The target index of a new attempt now, or None (no room)."""
        now = self.now()
        if self.policy == 'sched':
            unspent = {str(i): c - s for i, (c, s) in enumerate(zip(self.credit, self.started))}
            e = place.route(unspent, self.ns, task, attempt)
            if e is None:
                return None
            i = int(e)
        else:
            while self.starts60 and now - self.starts60[0] > 60:
                self.starts60.popleft()
            if sum(self.live) >= self.limit or len(self.starts60) >= self.burst:
                return None
            self.starts60.append(now)
            i = legacy.hashed(len(self.targets), task, attempt)
        self.started[i] += 1
        self.live[i] += 1
        self.fresh[i].append(now)
        return i

    # ------------------------------------------------------------------------------------------------- turns
    async def before_turn(self, i, key, need):
        """Seconds the collapse gate held this turn (0 without a gate); fails open after max_wait."""
        if not self.gates:
            return 0.0
        g, t0 = self.gates[i], time.monotonic()
        while not g.ask(key, need, self.now()):
            if time.monotonic() - t0 >= self.max_wait:
                break
            await asyncio.sleep(0.5)
        return time.monotonic() - t0

    def after_turn(self, i, key, prompt, completion, cached):
        if self.gates:
            self.gates[i].finished(key, prompt, completion, cached, self.now())

    def ended(self, i, key):
        self.live[i] = max(0, self.live[i] - 1)
        if self.gates:
            self.gates[i].end(key)


# ================================================================================================= replay driver
async def attempt(replay, client, sched, i, ti, j, recorded_body, httpx_errors, k0=0):
    """Replay trajectory j as attempt i on target ti (synth.bench Replay.attempt, with the gate around each turn),
    from turn k0 (> 0: a stationary start, the attempt joins mid-trajectory with its recorded history)."""
    trajectory = replay.corpus[j]
    key, loaded, need = f'{replay.conversation_prefix}/attempt/{i}', [], 0
    first = True
    try:
        for k in range(k0, len(trajectory['requests'])):
            if replay.stop.is_set():
                return
            try:
                body = await recorded_body(trajectory, k, loaded)
            except (OSError, ValueError, KeyError):
                return
            body.update({'model': replay.served, **replay.stream_args(), **replay.overrides})
            if first and k0:   # a mid-trajectory join: its first prompt is the whole recorded history (~4 chars/token)
                need = len(json.dumps(body.get('messages') or [])) // 4
            first = False
            held = await sched.before_turn(ti, key, need)
            try:
                turn = await replay.call(client, replay.targets[ti], body, conversation=key)
            except httpx_errors as exc:
                replay.errors.append({'t': time.time(), 'agent': i, 'error': repr(exc)[:300]})
                return
            turn['gate_s'], turn['target'] = round(held, 2), ti
            replay.turns.append(turn)
            sched.after_turn(ti, key, turn['prompt'], turn['completion'], turn['cached'])
            need = turn['prompt'] + turn['completion']
            gap = trajectory['gaps'][k]
            if gap:
                try:
                    await asyncio.wait_for(replay.stop.wait(), timeout=min(gap, replay.gap_cap) * replay.gap_scale)
                except asyncio.TimeoutError:
                    pass
    finally:
        sched.ended(ti, key)


async def replay(rp, sched, *, hours, scrape_every=15.0, decide_every=60.0, out=None, tag=None, seed=0,
                 stationary=False):
    """Admission-driven replay with a SchedClient (rp: synth.bench.Replay). stationary: the initial fill (the attempts
    the first decision admits) joins its trajectories at a random turn, so the population has a mix of context ages
    from the start instead of one launch cohort that ages together (#6: the cohort effect on a 2-h DeepSeek run);
    later attempts, refills of ended sessions, start at turn 0 as in a steady state (synth.bench Replay.agent does the
    same for each agent's first trajectory). Turns are drawn uniformly, so the joined population is somewhat younger
    than a time-weighted steady state."""
    import httpx
    from synth.bench import recorded_body
    tasks, started = {}, 0
    fill = None                    # stationary: how many attempts the first decision admits (they join mid-trajectory)
    end, next_decide = time.time() + hours * 3600, 0.0
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30), limits=limits, trust_env=False) as client:
        try:
            while time.time() < end:
                sched.scrape(await asyncio.to_thread(sched.fetch_all))   # fetch off-loop, apply on the loop thread
                for i in [i for i, t in tasks.items() if t.done()]:
                    del tasks[i]
                if time.time() < next_decide:
                    sched.refill()
                else:
                    next_decide = time.time() + decide_every
                    row = sched.decide()
                    row.update(tag=tag, attempts=len(tasks), turns=len(rp.turns), errors=len(rp.errors))
                    if out:
                        with open(out, 'a') as f:
                            f.write(json.dumps(row) + '\n')
                    if fill is None:
                        fill = sched.room_now() if stationary else 0
                while True:
                    j = random.Random(f'{seed}/attempt/{started + 1}').randrange(len(rp.corpus))
                    ti = sched.next_start(f'traj{j}', 0)
                    if ti is None:
                        break
                    started += 1
                    n = len(rp.corpus[j]['requests'])
                    k0 = random.Random(f'{seed}/k0/{started}').randrange(n) if started <= (fill or 0) and n > 1 else 0
                    tasks[started] = asyncio.create_task(attempt(rp, client, sched, started, ti, j, recorded_body,
                                                                 (httpx.HTTPError, RuntimeError, ValueError), k0=k0))
                await asyncio.sleep(scrape_every)
        finally:
            rp.stop.set()
            for t in tasks.values():
                t.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            if out:
                with open(str(out) + '.errors.json', 'w') as f:
                    json.dump(rp.errors, f)
    return rp

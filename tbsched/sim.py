"""Discrete-time simulator for scheduler policies (#451 workstream 1, validation step 1). Stdlib only, CPU, seconds.

It is a model, not a replay: each test runs the OLD rule and the NEW rule on the same simulated load and compares
them. The parts that matter for the lessons are mechanistic, not fitted curves:
* an engine has a KV pool (`kv_tokens`). A running request holds its prompt, its decoded tokens and a fixed state block
  (`state_tokens`: ~20k for Nemotron Super's Mamba/MTP state). Idle sessions' prompts stay cached in what is left, in
  LRU order, and are evicted when admissions need room. So more open sessions -> more evictions -> lower prefix hit ->
  more prefill -> fewer decode tokens: the Super thrash, with no thrash curve put in by hand;
* each step splits the engine's time between prefill (up to 90%, at `prefill_rate`) and decode (an aggregate rate
  rising with the batch, `gen_max x n / (n + n_half)`, at most `per_seq` per sequence);
* the KV pool overflowing preempts the youngest request (full recompute later);
* agentic sessions alternate model requests and tool calls (suite shapes below), replay sessions start with a cold
  ~100k prefill, single-turn sessions are one long decode. A share of sessions can hang (no more requests).
Metrics use vLLM's names (control.parse_metrics), so policies read the simulator the way they read a live engine.

Calibration (tests/test_sim.py): a CritPt-like mix keeps a high hit rate around 100 sessions/engine and collapses
past ~200, the shape measured on Super on 11-15 (CritPt 84 running 78% hit -> 173 running 27%).
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import heapq
import math
import random

from tbsched import control, legacy, lifecycle, place
from tbsched.gate import CollapseGate

PROFILES = {
    # Nemotron 3.5 Super, 1 engine per B200 (19.5M-token GPU KV pool; ~7.4k gen tok/s at capacity on Pi traffic)
    'super': dict(kv_tokens=19_500_000, state_tokens=20_000, max_seqs=512, gen_max=10_000, n_half=40, per_seq=120.0,
                  prefill_rate=60_000),
}
SUITES = {
    'ab': dict(turns=(8, 25), tool=(3, 20), out=(200, 1200), tool_tokens=(300, 1500), prompt0=6000),
    'critpt': dict(turns=(15, 40), tool=(5, 60), out=(1000, 4000), tool_tokens=(1500, 5000), prompt0=5000),
    'tbsci': dict(turns=(30, 70), tool=(20, 200), out=(400, 1500), tool_tokens=(800, 2000), prompt0=8000),
    'replay': dict(turns=(1, 2), tool=(5, 30), out=(2000, 6000), tool_tokens=(500, 1500), prompt0=(80_000, 120_000)),
    'hle': dict(single=True, turns=(1, 1), out=(3000, 40000), prompt0=2000),
}


def u(rng, lo_hi):
    if isinstance(lo_hi, (int, float)):
        return lo_hi
    lo, hi = lo_hi
    return rng.uniform(lo, hi)


class Req:
    __slots__ = ('sess', 'prompt', 'out', 'prefill', 'decoded', 'held', 'single')

    def __init__(self, sess, prompt, out, single=False):
        self.sess, self.prompt, self.out, self.single = sess, int(prompt), int(out), single
        self.prefill = self.decoded = self.held = 0


class Engine:
    def __init__(self, name, group, prof):
        self.name, self.group = name, group
        self.__dict__.update(prof)
        self.waiting, self.running, self.idle = deque(), [], OrderedDict()
        self.idle_tokens = self.held = 0.0
        self.c = {'vllm:generation_tokens_total': 0.0, 'vllm:prefix_cache_hits_total': 0.0,
                  'vllm:prefix_cache_queries_total': 0.0, 'vllm:num_preemptions_total': 0.0}
        self.kinds = {}            # session kind -> last time it had a request here
        self.released = False

    def metrics(self):
        return {**self.c, 'vllm:num_requests_running': float(len(self.running)),
                'vllm:num_requests_waiting': float(len(self.waiting)),
                'vllm:kv_cache_usage_perc': min(1.0, self.held / self.kv_tokens)}

    def _evict(self, need):
        while self.idle and self.held + self.idle_tokens + need > self.kv_tokens:
            _, v = self.idle.popitem(last=False)
            self.idle_tokens -= v

    def step(self, now, dt, finished):
        while self.waiting and len(self.running) < self.max_seqs:
            r = self.waiting[0]
            sid = None if r.single else id(r.sess)
            own = self.idle.pop(sid, 0) if sid is not None else 0
            self.idle_tokens -= own
            need = r.prompt + self.state_tokens
            self._evict(need)
            if self.held + need > self.kv_tokens:
                if own:
                    self.idle[sid] = own
                    self.idle_tokens += own
                break
            self.waiting.popleft()
            cached = min(own, r.prompt)
            self.c['vllm:prefix_cache_queries_total'] += r.prompt
            self.c['vllm:prefix_cache_hits_total'] += cached
            r.prefill, r.held = r.prompt - cached, need
            r.sess.last_cached = cached
            self.held += need
            self.running.append(r)
        pending = sum(r.prefill for r in self.running)
        share = min(0.9, pending / (self.prefill_rate * dt)) if pending else 0.0
        budget = share * self.prefill_rate * dt
        for r in self.running:
            if budget <= 0:
                break
            if r.prefill > 0:
                take = min(r.prefill, budget)
                r.prefill -= take
                budget -= take
        dec = [r for r in self.running if r.prefill <= 0]
        if dec:
            n = len(dec)
            per = min(self.per_seq, (1 - share) * self.gen_max * n / (n + self.n_half) / n) * dt
            for r in dec:
                got = min(per, r.out - r.decoded)
                r.decoded += got
                r.held += got
                self.held += got
                self.c['vllm:generation_tokens_total'] += got
        self._evict(0)
        while self.held > self.kv_tokens and self.running:   # preempt the youngest: recompute later
            r = self.running.pop()
            self.held -= r.held
            r.prompt, r.out, r.decoded, r.held, r.prefill = r.prompt + int(r.decoded), r.out - int(r.decoded), 0, 0, 0
            self.waiting.appendleft(r)
            self.c['vllm:num_preemptions_total'] += 1
        done = [r for r in self.running if r.decoded >= r.out]
        if done:
            self.running = [r for r in self.running if r.decoded < r.out]
            for r in done:
                self.held -= r.held
                if not r.single:
                    sid = id(r.sess)
                    self.idle[sid] = r.prompt   # the prompt is what the next turn can reuse (its prefix)
                    self.idle_tokens += r.prompt
                finished(r, now)
            self._evict(0)


@dataclass
class RunSpec:
    name: str
    suite: str
    items: int
    groups: list                    # engine groups it starts on
    namespace: str = ''             # seed namespace (differs per round)
    tasks: list | None = None       # task names (default: <name>-t<i>); parallel rounds share them
    weight: float = 1.0
    model: str = 'super'
    limit: int = 96                 # old rule: max_in_flight
    heavy: float = 0.0              # share of tasks with 2.5x tool output (long contexts concentrate under hashing)
    hang: float = 0.0               # share of sessions that hang (no more model requests)

    @property
    def kind(self):
        return 'single' if SUITES[self.suite].get('single') else 'replay' if self.suite == 'replay' else 'agentic'


class Sess:
    __slots__ = ('run', 'task', 'attempt', 'retry', 'engine', 'ctx', 'turns', 'turn', 'last', 'hang_at', 'start',
                 'last_cached', 'key', 'heavy', 'waiting_gate', 'done', 'inflight')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class World:
    def __init__(self, groups, runs, policy, profile='super', seed=0, dt=5.0, wall_cap=7200.0, window=131072):
        self.rng, self.dt, self.now, self.wall_cap, self.window = random.Random(seed), dt, 0.0, wall_cap, window
        self.engines = {e: Engine(e, g, PROFILES[profile]) for g, es in groups.items() for e in es}
        self.groups = {g: list(es) for g, es in groups.items()}
        self.runs = {r.name: r for r in runs}
        self.run_groups = {r.name: list(r.groups) for r in runs}
        self.queue = {r.name: deque((t, 0, 0) for t in (r.tasks or [f'{r.name}-t{i}' for i in range(r.items)]))
                      for r in runs}
        self.retries = {r.name: deque() for r in runs}
        self.live = {r.name: [] for r in runs}
        self.done = {r.name: 0 for r in runs}
        self.infra = {r.name: 0 for r in runs}
        self.gen = {r.name: 0.0 for r in runs}
        self.wake = []                        # (time, seq, session): tool calls ending
        self.seq = 0
        self.gates = {}                       # engine -> CollapseGate (when the policy installs them)
        self.trace = []                       # per-tick samples (policy.tick appends)
        self.events = []
        self.policy = policy
        policy.attach(self)

    # ------------------------------------------------------------------------------------------- queue / sessions
    def claimable(self, run):
        return len(self.queue[run]) + len(self.retries[run])

    def peek(self, run):
        q = self.retries[run] or self.queue[run]
        return q[0] if q else None

    def claim(self, run):
        q = self.retries[run] or self.queue[run]
        return q.popleft() if q else None

    def engines_of(self, run):
        return [e for g in self.run_groups[run] for e in self.groups.get(g, []) if not self.engines[e].released]

    def start(self, run, item, engine):
        spec, rng = self.runs[run], self.rng
        s = SUITES[spec.suite]
        task, attempt, retry = item
        h = int(hashlib.sha256(task.encode()).hexdigest(), 16)
        heavy = (h % 1000) < spec.heavy * 1000
        sess = Sess(run=run, task=task, attempt=attempt, retry=retry, engine=engine, ctx=int(u(rng, s['prompt0'])),
                    turns=max(1, int(u(rng, s['turns']))), turn=0, last=self.now, start=self.now, last_cached=0,
                    key=f'{spec.namespace}/{task}/{attempt}/{retry}', heavy=heavy, waiting_gate=False, done=False,
                    inflight=False,
                    hang_at=(int(u(rng, s['turns']) * rng.random()) if rng.random() < spec.hang else None))
        self.live[run].append(sess)
        self.request(sess)
        return sess

    def request(self, sess):
        e = self.engines[sess.engine]
        if e.released:
            return self.end(sess, infra='engine_released')
        g = self.gates.get(sess.engine)
        if g is not None and not g.ask(sess.key, sess.ctx, self.now):
            sess.waiting_gate = True
            self.push(self.now + 1.0, sess)
            return None
        sess.waiting_gate = False
        s = SUITES[self.runs[sess.run].suite]
        e.kinds[self.runs[sess.run].kind] = self.now
        sess.inflight = True
        e.waiting.append(Req(sess, sess.ctx, u(self.rng, s['out']), single=bool(s.get('single'))))
        return None

    def push(self, t, sess):
        self.seq += 1
        heapq.heappush(self.wake, (t, self.seq, sess))

    def finished(self, r, now):
        sess = r.sess
        sess.inflight = False
        if sess.done:
            return
        self.gen[sess.run] += r.out
        g = self.gates.get(sess.engine)
        if g is not None:
            g.finished(sess.key, r.prompt, r.out, sess.last_cached, now)
        sess.turn += 1
        sess.last = now
        s = SUITES[self.runs[sess.run].suite]
        if sess.turn >= sess.turns:
            return self.end(sess)
        mult = 2.5 if sess.heavy else 1.0
        sess.ctx += r.out + int(u(self.rng, s['tool_tokens']) * mult)
        if sess.ctx >= self.window:
            return self.end(sess)   # context_limit: a natural end for scheduling purposes
        if sess.hang_at is not None and sess.turn >= sess.hang_at:
            return None   # hung: no more requests (pi_backend_timeout); ended by the watchdog or the wall cap
        self.push(now + u(self.rng, s['tool']), sess)
        return None

    def end(self, sess, infra=None):
        if sess.done:
            return
        sess.done = True
        self.live[sess.run].remove(sess)
        g = self.gates.get(sess.engine)
        if g is not None:
            g.g.drop(sess.key)
        if infra:
            self.infra[sess.run] += 1
            self.retries[sess.run].append((sess.task, sess.attempt, sess.retry + 1))
            self.events.append((round(self.now), 'infra', sess.run, sess.task, infra))
        else:
            self.done[sess.run] += 1

    # ------------------------------------------------------------------------------------------------ the clock
    def run(self, hours=None, until=None):
        end = self.now + hours * 3600 if hours is not None else math.inf
        while self.now < end:
            if until is not None and until(self):
                break
            self.policy.step(self, self.now)
            while self.wake and self.wake[0][0] <= self.now:
                _, _, sess = heapq.heappop(self.wake)
                if not sess.done:
                    self.request(sess)
            for e in self.engines.values():
                if not e.released:
                    e.step(self.now, self.dt, self.finished)
            self.watch()
            self.now += self.dt
        return self

    def watch(self):
        """Hung sessions: the policy's watchdog (gap since the last model request ended) or the wall cap."""
        gap = getattr(self.policy, 'watchdog_sec', None)
        for run, ss in self.live.items():
            for sess in list(ss):
                if self.now - sess.start > self.wall_cap:
                    self.end(sess, infra='wall_cap')
                elif gap and not sess.waiting_gate and not sess.inflight and self.now - sess.last > gap:
                    self.end(sess, infra='hung_session')   # a request in flight or a gate wait is activity

    # ------------------------------------------------------------------------------------------------ views
    def sessions_on(self, engine, kind=None):
        return sum(1 for run, ss in self.live.items() for s in ss
                   if s.engine == engine and (kind is None or self.runs[run].kind == kind))

    def mixed(self, engine, window=120.0):
        k = self.engines[engine].kinds
        return any(self.now - k.get(x, -math.inf) <= window for x in ('single', 'replay'))

    def total_gen(self):
        return sum(e.c['vllm:generation_tokens_total'] for e in self.engines.values())


# ================================================================================================== policies
class OldPolicy:
    """What ran on 11-15: pi_eval's hash pin (sha256(task/attempt) over the run's engines), a fixed max_in_flight and a
    start burst per 60 s per run, and optionally the M5 autoscale (per registry, every 120 s: calm on every engine ->
    raise binding runs x1.15; pressure on either engine -> cut every run x0.85; `sibling_hold`: the 05:08 patch)."""

    def __init__(self, burst=32, autoscale=False, sibling_hold=True, tick=120.0):
        self.burst, self.autoscale, self.sibling_hold, self.tick_sec = burst, autoscale, sibling_hold, tick
        self.limits, self.starts, self.windows, self.calm, self.next_tick = {}, {}, {}, {}, 0.0

    def attach(self, w):
        self.limits = {r: spec.limit for r, spec in w.runs.items()}
        self.starts = {r: deque() for r in w.runs}
        self.windows = {e: control.Window(120.0) for e in w.engines}

    def step(self, w, now):
        for e in w.engines:
            self.windows[e].add(now, w.engines[e].metrics())
        if self.autoscale and now >= self.next_tick:
            self.next_tick = now + self.tick_sec
            self.scale(w, now)
        for run in w.runs:
            st = self.starts[run]
            while st and now - st[0] > 60:
                st.popleft()
            es = w.engines_of(run)
            while es and len(w.live[run]) < self.limits[run] and len(st) < self.burst and w.claimable(run):
                task, attempt, retry = w.claim(run)
                w.start(run, (task, attempt, retry), es[legacy.hashed(len(es), task, attempt)])
                st.append(now)

    def scale(self, w, now):
        for g, es in w.groups.items():
            verdict = legacy.m5_verdict([self.windows[e].obs() for e in es], any(w.mixed(e) for e in es),
                                        self.sibling_hold)
            self.calm[g] = self.calm.get(g, 0) + 1 if verdict == 'calm' else 0
            for r in [r for r in w.runs if g in w.run_groups[r]]:
                self.limits[r] = legacy.m5_limit(self.limits[r], verdict, self.calm[g], len(w.live[r]))


class SchedPolicy:
    """tbsched: per-engine limits (control.aimd), credits (place.allocate/withdraw), routing at claim (place.route),
    optionally engine-group decisions (lifecycle.plan_groups), the collapse gate and the liveness watchdog."""

    def __init__(self, aimd=None, placep=None, life=None, groups=False, gate=False, watchdog_sec=None, tick=60.0,
                 start=None):
        self.aimd, self.placep, self.life = aimd or {}, placep or {}, life or {}
        self.groups_on, self.gate_on, self.watchdog_sec, self.tick_sec = groups, gate, watchdog_sec, tick
        self.start_limit = start
        self.st, self.single, self.credit, self.starts, self.windows = {}, {}, {}, {}, {}
        self.gst, self.status, self.next_tick, self.limits = {}, {}, 0.0, {}

    def attach(self, w):
        self.windows = {e: control.Window(120.0) for e in w.engines}
        for e in w.engines:
            self.st[e] = {'limit': self.start_limit or {**control.AIMD, **self.aimd}['start']}
            self.single[e] = {'limit': control.SINGLE['start']}
            if self.gate_on:
                w.gates[e] = CollapseGate()
        self.credit = {r: {} for r in w.runs}
        self.starts = {r: {} for r in w.runs}
        self.status = {g: 'active' for g in w.groups}

    def obs(self, w, e):
        ss = [s for ss in w.live.values() for s in ss if s.engine == e and w.runs[s.run].kind != 'single']
        g = w.gates.get(e)
        return self.windows[e].obs(sessions=len(ss), singles=w.sessions_on(e, 'single'), mixed=w.mixed(e),
                                   ctx_tokens=float(sum(s.ctx for s in ss)), kv_tokens=w.engines[e].kv_tokens,
                                   fresh=sum(1 for s in ss if w.now - s.start <= 120),
                                   reuse=g.g.eff if g is not None else None)

    def tick(self, w, now):
        engines = {}
        for e, eng in w.engines.items():
            if eng.released:
                continue
            o = self.obs(w, e)
            unspent = sum(max(0, self.credit[r].get(e, 0) - self.starts[r].get(e, 0)) for r in w.runs
                          if w.runs[r].kind != 'single')
            fresh = sum(1 for r, ss in w.live.items() if w.runs[r].kind != 'single' for x in ss
                        if x.engine == e and now - x.start <= 120)
            res = control.limit_step(self.st[e], o, self.aimd, o.sessions, now, committed=o.sessions + min(unspent, fresh))
            sc = control.single_cap(self.single[e], o, {}, now)
            self.limits[e] = res['limit']
            g = w.gates.get(e)
            if g is not None:
                ev = control.hit_evidence(o, {**control.AIMD, **self.aimd})
                if ev is not None and ev < g.collapse_eff:
                    g.trigger(now, 'hit_collapse', capacity=o.kv_tokens)
            gs = g.g.stats() if g is not None else {}
            hold = bool(gs.get('ctx_waiting') or gs.get('ctx_parked'))   # sessions held at the gate: no new ones
            engines[e] = place.Engine(e, eng.group, res['limit'], sc['limit'],
                                      open=self.status.get(eng.group) == 'active' and not hold)
            w.trace.append((round(now), e, round(o.running), o.sessions, None if o.hit is None else round(o.hit, 3),
                            round(o.gen_tok_s), res['limit'], res['state']))
        runs = self.views(w)
        for name, take in place.withdraw(runs, engines).items():
            for e, n in take.items():
                self.credit[name][e] -= n
        runs = self.views(w)
        for name, add in place.allocate(runs, engines, self.placep).items():
            for e, n in add.items():
                self.credit[name][e] = self.credit[name].get(e, 0) + n
        if self.groups_on:
            self.plan_groups(w, now)

    def views(self, w):
        out = []
        for r, spec in w.runs.items():
            es = w.engines_of(r)
            live = {}
            for s in w.live[r]:
                live[s.engine] = live.get(s.engine, 0) + 1
            out.append(place.Run(r, spec.kind, es, w.claimable(r), credit=dict(self.credit[r]),
                                 starts=dict(self.starts[r]), live=live, weight=spec.weight))
        return out

    def plan_groups(self, w, now):
        gv = {}
        for g, es in w.groups.items():
            live = [e for e in es if not w.engines[e].released]
            gv[g] = lifecycle.GroupView(g, 'super', live, 'released' if not live else self.status[g],
                                        sessions=sum(w.sessions_on(e) for e in live),
                                        running=(sum(len(w.engines[e].running) for e in live) / len(live)) if live else 0,
                                        limit=sum(self.limits.get(e, 0) for e in live))
        rv = {r: lifecycle.RunView(r, 'super', spec.kind, {g: 1.0 for g in w.run_groups[r]}, w.claimable(r),
                                   live=len(w.live[r]), session_sec=600.0) for r, spec in w.runs.items()}
        for act in lifecycle.plan_groups(gv, rv, self.gst, now, self.life):
            w.events.append((round(now), *act))
            if act[0] == 'join':
                w.run_groups[act[2]].append(act[1])
            elif act[0] == 'drain':
                self.status[act[1]] = 'draining'
            elif act[0] == 'release':
                self.status[act[1]] = 'released'
                for e in w.groups[act[1]]:
                    w.engines[e].released = True
                for ss in w.live.values():
                    for s in list(ss):
                        if s.engine in w.groups[act[1]]:
                            w.end(s, infra='tail_cut')

    def step(self, w, now):
        for e in w.engines:
            self.windows[e].add(now, w.engines[e].metrics())
        for e, g in w.gates.items():
            g.sync({s.key: s.start for ss in w.live.values() for s in ss if s.engine == e}, now)
            g.tick(now, loaded=True)
        if now >= self.next_tick:
            self.next_tick = now + self.tick_sec
            self.tick(w, now)
        for r, spec in w.runs.items():
            while True:
                item = w.peek(r)
                if item is None:
                    break
                unspent = {e: self.credit[r].get(e, 0) - self.starts[r].get(e, 0) for e in w.engines_of(r)}
                e = place.route(unspent, spec.namespace, item[0], item[1])
                if e is None:
                    break
                w.start(r, w.claim(r), e)
                self.starts[r][e] = self.starts[r].get(e, 0) + 1

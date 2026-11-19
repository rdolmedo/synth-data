"""The campaign scheduler process (#451 workstream 1): one per campaign, `python -m tbsched.loop CAMPAIGN [--once]`.

CAMPAIGN/sched/config.json:
  {"mode": "shadow",                       # shadow (default): decide and record, act on nothing | on
   "tick_sec": 60, "scrape_sec": 15, "gate": "off",
   "aimd": {...}, "place": {...}, "life": {...},          # overrides of control.AIMD, place.PLACE, lifecycle.LIFE
   "groups": {"m5l-ab": {"registry": "/data/.../serving/m5l-ab", "model": "super"}},   # "shared": true = never
                                                       # released here (another campaign uses it and releases it);
                                                       # "mixed": true = it also serves traffic not listed as runs
   "runs": {"ab-t2": {"dir": "/data/.../runs/m5l_automationbench_t2/t2", "kind": "agentic",
                      "groups": {"m5l-ab": 1.0}, "weight": 1.0, "slots": 32, "hooked": false}}}

Workers of a run launched with pi_eval `run.py --sched-state CAMPAIGN/sched/runs/<run>.json` read `credit_urls`
there (aa_evals/common/pi_eval/sched_router.py) while `mode` is "on" and `updated` is fresh.

Each scrape: every engine's /metrics into a 2-min window. Each tick:
  1. the lease (exit if another instance took it over);
  2. per run: work.pending (claims of workers condor no longer lists are dead, i.e. claimable) and its pins;
  3. per engine: Obs (sessions from the pins, fresh starts, mixed traffic) -> control.aimd limit and single cap;
  4. place.withdraw + place.allocate -> credit per (run, engine), written to sched/runs/<run>.json;
  5. lifecycle.plan_groups (join / drain / release / stop_run) and lifecycle.plan_workers per run;
  6. sched/engines/*.json, sched/groups/*.json, events.jsonl.
In shadow mode steps 4-5 only write sched/ files and events (marked shadow): no run file, no condor command. `on`
applies them and is refused for a run unless "hooked": true (its workers read tbsched credits; the pipeline hooks
land in their own flag-gated PRs). Every condor write is by exact id, after an owner check.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from tbsched import control, lifecycle, place, work
from tbsched.state import Lease, StateDir, atomic, read

USER = os.environ.get('USER', 'user')


class LiveIO:
    """Everything the loop reads from or does to the outside world (tests pass a fake)."""

    def scrape(self, url, timeout=5.0):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as r:
            return r.read().decode('utf-8', 'replace')

    def condor_jobs(self, clusters):
        """{'cluster.proc': (status, owner)} for those clusters (status: idle | running | held | other)."""
        if not clusters:
            return {}
        out = subprocess.run(['condor_q', *sorted(clusters), '-af', 'ClusterId', 'ProcId', 'JobStatus', 'Owner'],
                             capture_output=True, text=True, timeout=120)
        if out.returncode:
            raise RuntimeError(out.stderr[-300:])
        names = {'1': 'idle', '2': 'running', '5': 'held'}
        jobs = {}
        for line in out.stdout.splitlines():
            p = line.split()
            if len(p) == 4:
                jobs[f'{p[0]}.{p[1]}'] = (names.get(p[2], 'other'), p[3])
        return jobs

    def condor_rm(self, ids):
        if ids:
            subprocess.run(['condor_rm', *ids], capture_output=True, text=True, timeout=120)

    def submit_workers(self, run_dir, n):
        """pi_eval's own submitter from the run's frozen code (as the M5 scale_run.py did)."""
        cfg = read(Path(run_dir) / 'config.json', {})
        code = (
            'import json, sys; from pathlib import Path; '
            'from aa_evals.common.pi_eval import run as R; '
            f'r = Path({str(run_dir)!r}); print(R.submit_workers(r, json.loads((r / "config.json").read_text()), {int(n)}))')
        subprocess.run([sys.executable, '-c', code], cwd=cfg.get('code') or None, capture_output=True, text=True,
                       timeout=600, env={**os.environ, 'PYTHONPATH': cfg.get('code', '')})

    def touch(self, path, text):
        Path(path).write_text(text)

    def remove(self, path):
        Path(path).unlink(missing_ok=True)


def metrics_url(engine_url):
    base = engine_url.rstrip('/')
    return (base[:-3] if base.endswith('/v1') else base) + '/metrics'


class Scheduler:
    def __init__(self, campaign, io=None, now=time.time):
        self.root = Path(campaign)
        self.state = StateDir(self.root / 'sched')
        self.io, self.now = io or LiveIO(), now
        self.lease = Lease(self.root / 'sched')
        self.windows, self.last_ok, self.pin_cache = {}, {}, {}
        self.cfg = {}
        self.started = self.now()

    # ---------------------------------------------------------------------------------------------------- config
    def load(self):
        self.cfg = read(self.root / 'sched' / 'config.json', {}) or {}
        self.mode = self.cfg.get('mode', 'shadow')
        self.p_aimd = control.params(self.cfg.get('aimd', {}))   # GUARD's settings too for policy "guard"
        self.p_place = {**place.PLACE, **self.cfg.get('place', {})}
        self.p_life = {**lifecycle.LIFE, **self.cfg.get('life', {})}
        self.engines, self.client_url = {}, {}   # engine url -> group; client url (pins) -> engine url
        self.engine_client = {}                  # engine url -> the client url workers pin (sched_router.py)
        self.pool = {}                           # engine url -> engines behind it (a pooled gate)
        self.group_client_urls = {}
        for g, spec in self.cfg.get('groups', {}).items():
            st = self.state.get('groups', g, {}) or {}
            if st.get('status') == 'released':
                continue
            ep = read(Path(spec['registry']) / 'endpoints.json', {}) or {}
            urls = [v1(u) for u in ep.get('urls') or []]     # as pi_eval's registry.resolve writes them (pins)
            engine_urls = [v1(u) for u in ep.get('engine_urls') or []] or urls
            if len(engine_urls) != len(urls):   # one gate over a pool: the gate picks the engine, so each client url
                for u in urls:                 # is one engine here (scraped through the gate's /metrics), with the
                    self.pool[u] = max(1, len(engine_urls) // max(1, len(urls)))   # pool's limits
                engine_urls = urls
            self.group_client_urls[g] = list(urls)
            for u, e in zip(urls, engine_urls):
                self.client_url[u] = e
                self.engine_client[e] = u
            for e in engine_urls:
                self.client_url.setdefault(e, e)
                self.engine_client.setdefault(e, e)
                self.engines[e] = g

    # ---------------------------------------------------------------------------------------------------- observe
    def scrape(self):
        t = self.now()
        for e in self.engines:
            try:
                m = control.parse_metrics(self.io.scrape(metrics_url(e)))
            except Exception:   # noqa: BLE001 -- a dead engine is stale, not a crash
                continue
            self.windows.setdefault(e, control.Window(120.0)).add(t, m)
            self.last_ok[e] = t

    def observe_runs(self):
        runs = {}
        clusters = {name: worker_clusters(Path(spec['dir'])) for name, spec in self.cfg.get('runs', {}).items()}
        try:
            self.jobs = self.io.condor_jobs(set().union(*clusters.values()) if clusters else set())
            known = True
        except Exception:   # noqa: BLE001 -- unknown: trust heartbeats only, act on nothing
            self.jobs, known = {}, False
        now = self.now()
        for name, spec in self.cfg.get('runs', {}).items():
            d = Path(spec['dir'])
            # a run whose worker clusters we cannot see (none recorded, condor down) has unknown liveness: heartbeats only
            run_known = known and bool(clusters[name])
            alive = (lambda w: (str(w).split('-')[0] in self.jobs) if w else None) if run_known else None
            pend = work.pending(d, worker_alive=alive, now=now)
            claims = d / 'queue' / 'claims'
            cache = self.pin_cache.setdefault(name, {})
            keys = work._names(claims)
            # a pin is written at claim time: a claim that is no longer live and has none never gets one
            pins = work.pins(d, keys, cache, settled=keys - pend.live_keys)
            urls = self.group_urls(spec)
            starts, live, fresh = {}, {}, {}
            for k in keys:
                url = pins.get(k) or work.hashed(urls, k)    # unpinned (unhooked pi_eval): its hash, as the worker does
                if url is None:
                    continue
                e = self.client_url.get(url, url)
                starts[e] = starts.get(e, 0) + 1
                if k in pend.live_keys:                    # dead claims are claimable work, not engine load
                    live[e] = live.get(e, 0) + 1
                    rec = read(claims / k / 'claim.json', {}) or {}
                    if now - _epoch(rec.get('claimed_at'), now - 1e9) <= 120:
                        fresh[e] = fresh.get(e, 0) + 1
            for k in [k for k in cache if k not in keys]:
                cache.pop(k)
            # engines whose serving gate holds new conversations for this run (pi_eval admission.json): its workers
            # skip them, so no credit may wait there
            adm = read(d / 'queue' / 'admission.json', {}) or {}
            held = {self.client_url.get(urls[i], urls[i]) for i in adm.get('gate_held_replicas') or []
                    if isinstance(i, int) and 0 <= i < len(urls)}
            runs[name] = {'spec': spec, 'pending': pend, 'starts': starts, 'live': live, 'fresh': fresh,
                          'state': self.state.get('runs', name, {}) or {}, 'jobs_known': run_known, 'held': held}
        return runs

    def group_urls(self, spec):
        """Client urls of the run's first group (the order pi_eval hashes over)."""
        for g in spec.get('groups', {}):
            return self.group_client_urls.get(g, [])
        return []

    # ---------------------------------------------------------------------------------------------------- decide
    def tick(self):
        if not self.lease.acquire(self.now()) or not self.lease.renew(self.now()):
            raise SystemExit('another scheduler holds the lease for this campaign')
        self.load()
        now = self.now()
        runs = self.observe_runs()
        kinds_on = {}
        for r in runs.values():
            for e, n in r['live'].items():
                if n:
                    kinds_on.setdefault(e, set()).add(r['spec'].get('kind', 'agentic'))
        engines = {}
        for e, g in self.engines.items():
            st = self.state.get('engines', e, {}) or {}
            ov = self.state.override('engines', e)
            sessions = sum(r['live'].get(e, 0) for r in runs.values() if r['spec'].get('kind', 'agentic') != 'single')
            singles = sum(r['live'].get(e, 0) for r in runs.values() if r['spec'].get('kind') == 'single')
            win = self.windows.get(e)
            stale = win is None or now - self.last_ok.get(e, -math.inf) > 60
            gspec = self.cfg.get('groups', {}).get(g, {})
            o = (win.obs if win else control.Window().obs)(
                sessions=sessions, singles=singles,
                # "mixed": true for a group whose engines also serve traffic tbsched does not see as runs (e.g. the
                # aa_evals dispatcher's single-turn lanes): their hit rate is no thrash evidence
                mixed=bool(kinds_on.get(e, set()) & {'single', 'replay'}) or bool(gspec.get('mixed')),
                fresh=sum(r['fresh'].get(e, 0) for r in runs.values()))
            if stale:
                o.stale = True
            p = scale_aimd({**self.p_aimd, **ov.get('aimd', {})}, self.pool.get(e, 1))
            fresh_pi = sum(r['fresh'].get(e, 0) for r in runs.values() if r['spec'].get('kind', 'agentic') != 'single')
            unspent = sum(max(0, (r['state'].get('credit') or {}).get(e, 0) - r['starts'].get(e, 0))
                          for r in runs.values() if r['spec'].get('kind', 'agentic') != 'single')
            res = control.limit_step(st.setdefault('aimd', {}), o, p, sessions, now,
                                     committed=sessions + min(unspent, fresh_pi))
            sc = control.single_cap(st.setdefault('single', {}), o, ov.get('single', {}), now)
            limit = int(ov.get('limit', res['limit']))
            gstat = (self.state.get('groups', g, {}) or {}).get('status', 'active')
            engines[e] = place.Engine(e, g, limit, sc['limit'], open=gstat == 'active' and not o.stale)
            self.state.put('engines', e, {**st, 'name': e, 'group': g, 'limit': limit, 'single_cap': sc['limit'],
                                          'state': res['state'], 'causes': res['causes'], 'sessions': sessions,
                                          'singles': singles, 'obs': {k: getattr(o, k) for k in (
                                              'running', 'waiting', 'kv', 'hit', 'gen_tok_s', 'mixed', 'stale')},
                                          'updated': now})
            if res['changed']:
                self.state.event(kind='limit', engine=e, limit=limit, state=res['state'], causes=res['causes'],
                                 shadow=self.mode != 'on')
        views = []
        for name, r in runs.items():
            ov = self.state.override('runs', name)
            groups = {**r['spec'].get('groups', {}), **r['state'].get('joined', {})}
            es = [e for e, g in self.engines.items() if g in groups and e not in r['held']]
            views.append(place.Run(name, r['spec'].get('kind', 'agentic'), es, r['pending'].claimable,
                                   credit=dict(r['state'].get('credit', {})), starts=r['starts'], live=r['live'],
                                   weight=float(ov.get('weight', r['spec'].get('weight', 1.0)))))
        take = place.withdraw(views, engines)
        for v in views:
            for e, n in take.get(v.name, {}).items():
                v.credit[e] -= n
        add = place.allocate(views, engines, self.p_place)
        for v in views:
            for e, n in add.get(v.name, {}).items():
                v.credit[e] = v.credit.get(e, 0) + n
        decisions = {'credit': {v.name: v.credit for v in views}, 'workers': {}, 'groups': []}
        for v in views:
            r = runs[v.name]
            slots = int(r['spec'].get('slots', 16))
            unspent = sum(v.unspent(e) for e in v.engines)
            plan = lifecycle.plan_workers(
                r['pending'].live, unspent, r['pending'].claimable, slots, self.workers(r) if r['jobs_known'] else [],
                self.p_life, known=r['pending'].known and r['jobs_known'])
            if not v.engines and r['pending'].known:   # every engine group released: nothing to run on, keep it stopped
                plan = {**plan, 'add': 0, 'stop': True, 'no_engines': True}
            decisions['workers'][v.name] = plan
        decisions['groups'] = self.plan_groups(runs, engines, now)
        for v in views:
            r = runs[v.name]
            pend = r['pending']
            credit_urls = {self.engine_client.get(e, e): c for e, c in v.credit.items() if e in v.engines}
            self.state.put('runs', v.name, {**r['state'], 'name': v.name, 'credit': v.credit, 'engines': v.engines,
                                            'credit_urls': credit_urls,
                                            'namespace': r['spec'].get('namespace', ''),
                                            'pending': {'unclaimed': pend.unclaimed, 'retries': pend.retries,
                                                        'live': pend.live, 'dead': pend.dead, 'known': pend.known},
                                            'updated': now,
                                            'workers': decisions['workers'][v.name],
                                            # 'on' only for a hooked run: workers launched with --sched-state follow
                                            # credit only then, so "hooked": false is an off switch
                                            'mode': 'on' if self.mode == 'on' and r['spec'].get('hooked') else 'shadow'})
        if not self.lease.held():   # a takeover during this tick: the new holder decides from here
            raise SystemExit('lease lost during the tick')
        self.apply(decisions, runs)
        return decisions

    def workers(self, r):
        """The run's worker jobs as condor lists them now (ours only), with heartbeat, idle time and worker-local
        failures. Idle-since times live in the run's state file, so a restart does not reset them."""
        d, now = Path(r['spec']['dir']), self.now()
        clusters = worker_clusters(d)
        hbs = {}
        for hb in (d / 'workers').glob('*.json') if (d / 'workers').exists() else []:
            jid = hb.name.split('-')[0]
            if jid not in hbs or hb.stat().st_mtime > hbs[jid].stat().st_mtime:
                hbs[jid] = hb    # the newest heartbeat of a job (a restarted job writes a new file)
        idle_since = r['state'].setdefault('idle_since', {})
        fails, oks = local_failures(d, now, self.p_life['recycle_window_sec'])
        out = []
        for jid, (status, owner) in sorted(self.jobs.items()):
            if owner != USER or jid.split('.')[0] not in clusters:
                continue
            hb = hbs.get(jid)
            rec = read(hb, {}) if hb else {}
            running = len(rec.get('running') or [])
            if running or status != 'running':
                idle_since.pop(jid, None)
            else:
                idle_since.setdefault(jid, now)
            out.append(lifecycle.Worker(jid, status, running=running,
                                        hb_age=(now - hb.stat().st_mtime) if hb else None,
                                        idle_for=now - idle_since.get(jid, now),
                                        local_failures=fails.get(jid, 0), successes=oks.get(jid, 0)))
        for jid in [j for j in idle_since if j not in self.jobs]:
            idle_since.pop(jid)
        return out

    def plan_groups(self, runs, engines, now):
        gv = {}
        for g, spec in self.cfg.get('groups', {}).items():
            st = self.state.get('groups', g, {}) or {}
            es = [e for e, gg in self.engines.items() if gg == g]
            obs = [self.windows[e].obs() if e in self.windows else None for e in es]
            sessions = sum(r['live'].get(e, 0) for r in runs.values() for e in es)
            # busy (never released): a registry shared with other campaigns (their sessions in tool calls are
            # invisible here: the owning campaign releases it); no engine read; an engine unreadable for less than
            # dead_engine_sec or with live claims on it; more requests than our sessions explain (another client);
            # live claims we could not place on an engine
            dead_after = self.p_life.get('dead_engine_sec', 600)
            unread = [e for e, o in zip(es, obs) if o is None or now - self.last_ok.get(e, -math.inf) > 60]
            why = ('shared' if spec.get('shared') else 'no_engines' if not es else
                   'engine_unreadable' if any(now - self.last_ok.get(e, self.started) < dead_after
                                              or any(r['live'].get(e) for r in runs.values()) for e in unread) else
                   'other_clients' if sum(o.running + o.waiting for o in obs if o) > sessions else
                   'unplaced_claims' if any(r['pending'].live and not r['live'] for r in runs.values()
                                            if g in r['spec'].get('groups', {})) else None)
            busy = why is not None
            if why != st.get('busy_why'):
                self.state.event(kind='group_busy', group=g, why=why)
                st = {**st, 'busy_why': why}
                self.state.put('groups', g, {**st, 'name': g})
            gv[g] = lifecycle.GroupView(g, spec.get('model', ''), es, st.get('status', 'active'), sessions=sessions,
                                        running=(sum(o.running for o in obs if o) / len(es)) if es else 0.0,
                                        limit=sum(engines[e].limit for e in es if e in engines), busy=busy)
        # spreading a run onto another group is opt-in ("steal": true): pi_eval workers (sched_router.py) use their own
        # registry only, so credit on a joined group would strand; and never for an unhooked run in on mode
        rv = {n: lifecycle.RunView(n, r['spec'].get('model', ''), r['spec'].get('kind', 'agentic'),
                                   {**r['spec'].get('groups', {}), **r['state'].get('joined', {})},
                                   r['pending'].claimable, r['pending'].live,
                                   steal=bool(r['spec'].get('steal', False)) and (self.mode != 'on' or bool(
                                       r['spec'].get('hooked')))) for n, r in runs.items()}
        gst = {g: (self.state.get('groups', g, {}) or {}).get('counters', {}) for g in gv}
        keep = {g for g in gv if self.state.override('groups', g).get('keep')}
        acts = lifecycle.plan_groups(gv, rv, gst, now, self.p_life, keep=keep)
        for g in gv:
            st = self.state.get('groups', g, {}) or {}
            self.state.put('groups', g, {**st, 'name': g, 'status': st.get('status', 'active'), 'counters': gst[g]})
        return acts

    # ---------------------------------------------------------------------------------------------------- act
    def apply(self, decisions, runs):
        shadow = self.mode != 'on'
        for name, plan in decisions['workers'].items():
            st = self.state.get('runs', name, {}) or {}
            sig = json.dumps(plan, sort_keys=True)
            active = plan['add'] or plan['remove'] or plan['recycle'] or plan['recycle_wait'] or plan['stop']
            if active and sig != st.get('last_plan'):     # one event per change, not one per tick
                self.state.event(kind='workers', run=name, shadow=shadow, **plan)
            if sig != st.get('last_plan'):
                self.state.put('runs', name, {**st, 'last_plan': sig})
        last = read(self.state.root / 'last_group_actions.json', {}) or {}
        sig = json.dumps(decisions['groups'])
        if sig != last.get('acts'):
            for act in decisions['groups']:
                self.state.event(kind='group', action=list(act), shadow=shadow)
            atomic(self.state.root / 'last_group_actions.json', {'acts': sig})
        if shadow:
            return
        for name, plan in decisions['workers'].items():
            r = runs[name]
            if not r['spec'].get('hooked') or not r['jobs_known']:
                continue   # its workers do not read tbsched credits yet, or condor's view is unknown: no action
            d = Path(r['spec']['dir'])
            mine = {j for j, (_, owner) in self.jobs.items() if owner == USER}
            self.io.condor_rm([j for j in plan['recycle'] + plan['remove'] if j in mine])
            stop = d / 'STOP_WORKERS'
            if plan['stop'] and not stop.exists():
                self.write_stop(name, stop, 'tbsched: its engines were released\n' if plan.get('no_engines')
                                else 'tbsched: nothing left to claim\n')
            elif r['pending'].claimable > 0 and stop.exists() and not plan.get('no_engines'):
                if not self.our_stop(name, stop):
                    continue         # a person or another tool stopped this run: leave it stopped, add nothing
                self.io.remove(stop)  # work came back (retries, a new round): workers must claim again
                self.state.event(kind='unstop', run=name)
            if plan['add']:
                self.io.submit_workers(d, plan['add'])
        for act in decisions['groups']:
            if act[0] == 'join':
                if not runs.get(act[2], {}).get('spec', {}).get('hooked'):
                    continue
                st = self.state.get('runs', act[2], {}) or {}
                self.state.put('runs', act[2], {**st, 'joined': {**st.get('joined', {}), act[1]: 1.0}})
            elif act[0] in ('drain', 'release'):
                users = [r for r in runs.values() if act[1] in r['spec'].get('groups', {})
                         or act[1] in r['state'].get('joined', {})]
                if not users or not all(r['spec'].get('hooked') for r in users):
                    continue   # no run here, or one that still schedules itself: the old rules own this group
                st = self.state.get('groups', act[1], {}) or {}
                if act[0] == 'release' and not self.release(act[1]):
                    continue   # not removed (no job record, not ours, condor down): stays active, retried next tick
                self.state.put('groups', act[1], {**st, 'status': 'draining' if act[0] == 'drain' else 'released',
                                                  'since': self.now()})
            elif act[0] == 'stop_run':
                r = runs.get(act[1])
                stop = Path(r['spec']['dir']) / 'STOP_WORKERS' if r else None
                if r and r['spec'].get('hooked') and not stop.exists():
                    self.write_stop(act[1], stop, 'tbsched: its engines were released\n')

    def write_stop(self, run, path, text):
        """STOP_WORKERS written by tbsched, recorded with its mtime: only that file is tbsched's to clear later."""
        self.io.touch(path, text)
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = None
        st = self.state.get('runs', run, {}) or {}
        self.state.put('runs', run, {**st, 'stop_written': mtime})

    def our_stop(self, run, path):
        try:
            text, mtime = Path(path).read_text(), Path(path).stat().st_mtime
        except OSError:
            return False
        rec = (self.state.get('runs', run, {}) or {}).get('stop_written')
        return text.startswith('tbsched:') and rec is not None and abs(mtime - rec) < 1e-3

    def release(self, group) -> bool:
        """condor_rm the group's serving job after an owner check (as M5's release_idle and pi_eval's release).
        True when it was removed or is already gone; False when nothing could be done (no record, not ours, condor
        unreadable): the group then stays active and the release is retried."""
        spec = self.cfg['groups'][group]
        job = (read(Path(spec['registry']) / 'job.json', {}) or {}).get('job_id')
        if not job:
            self.state.event(kind='release_blocked', group=group, why='no job.json job_id')
            return False
        try:
            jobs = self.io.condor_jobs({job.split('.')[0]})
        except Exception as e:   # noqa: BLE001
            self.state.event(kind='release_blocked', group=group, why=f'condor_q: {e!r}'[:300])
            return False
        if job not in jobs:
            self.state.event(kind='released', group=group, job=job, already_gone=True)
            return True
        if jobs[job][1] != USER:
            self.state.event(kind='release_blocked', group=group, why=f'job {job} is owned by {jobs[job][1]}')
            return False
        self.io.condor_rm([job])
        self.state.event(kind='released', group=group, job=job)
        return True


def v1(url):
    """pi_eval's form of a registry url (harbor_mswea.endpoint._v1): workers pin and look up credit by it."""
    url = str(url).rstrip('/')
    return url if url.endswith('/v1') else url + '/v1'


POOL_KEYS = ('start', 'floor', 'ceil', 'min_step', 'load_min', 'run_max', 'wait_calm', 'wait_max')


def scale_aimd(p, n):
    """The per-engine aimd counts for a pooled gate of n engines (one scheduler engine)."""
    return p if n <= 1 else {**p, **{k: p[k] * n for k in POOL_KEYS if k in p}}


def worker_clusters(run_dir):
    """Worker clusters a run recorded: pi_eval's jobs.json (a list) or eval_q35's ({"jobs": [...]}, worker roles)."""
    rec = read(Path(run_dir) / 'jobs.json', []) or []
    if isinstance(rec, dict):
        rec = rec.get('jobs') or []
    return {str(j['cluster']) for j in rec if isinstance(j, dict) and 'cluster' in j
            and str(j.get('role', 'worker')).startswith('worker')}


def local_failures(run_dir, now, window):
    """({worker job id: worker-local infra failures}, {worker job id: successes}) over attempts that ended within the
    window (attempts/<key>/outcome.json + attempt.json's worker, as the M5 recycler read them)."""
    bad, ok = {}, {}
    try:
        entries = list(os.scandir(Path(run_dir) / 'attempts'))
    except FileNotFoundError:
        return bad, ok
    for ent in entries:
        f = Path(ent.path) / 'outcome.json'
        try:
            if now - f.stat().st_mtime > window:
                continue
        except OSError:
            continue
        o = read(f, {}) or {}
        w = str((read(Path(ent.path) / 'attempt.json', {}) or {}).get('worker', '')).split('-')[0]
        if not w:
            continue
        if any(x in str(o.get('infrastructure_reason', '')) for x in lifecycle.LOCAL_FAILURES):
            bad[w] = bad.get(w, 0) + 1
        elif not o.get('infrastructure_failure'):
            ok[w] = ok.get(w, 0) + 1
    return bad, ok


def _epoch(iso, default):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(iso).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('campaign')
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # run()'s finally releases the lease
    run(Scheduler(a.campaign), once=a.once)


def run(s, once=False, sleep=time.sleep):
    s.load()
    tick_sec, scrape_sec = s.cfg.get('tick_sec', 60), s.cfg.get('scrape_sec', 15)
    next_tick = 0.0
    try:
        while True:
            s.scrape()
            if time.time() >= next_tick:
                next_tick = time.time() + tick_sec
                try:
                    s.tick()
                except SystemExit:
                    raise
                except Exception as e:   # noqa: BLE001 -- keep scheduling; the event says what failed
                    s.state.event(kind='tick_failed', error=repr(e)[:500])
                if once:
                    return
            if not s.lease.held():
                raise SystemExit('lease lost: another scheduler took over')
            sleep(scrape_sec)
    finally:
        s.lease.release()   # only if still ours: a restart need not wait stale_sec


if __name__ == '__main__':
    main()

"""Watching and changing a live production run (#237, #238; m1-hero debrief #172, R13 / R6).

  python -m task_synthesis.pipeline.production status <run> [--json]
  python -m task_synthesis.pipeline.production replan <run> --vocab v2h0 --seed N [--mix easy-heavy] [--apply]

`status` is the run's one health source: every component's job id, its condor state ("unknown" when condor_q fails,
never "gone"), its heartbeat age from its own files, its counters, and the alerts as explicit fields, so a monitor
only prints `alerts`. Components: driver, DeepSeek servers (running / waiting from /metrics), bind job, round shards
(claimed / admitted / in flight / accepted / cut, from their heartbeats), stream producer, Super controller and
engines, decontam.

`replan` re-plans the specs nobody has started (e.g. m1-hero's v2h -> v2h0 switch, #165) without touching a running
session; see `replan()`.
"""
import json
import os
import random
import re
import socket
import subprocess
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import production as P

CONDOR_STATES = {1: 'idle', 2: 'running', 3: 'removed', 4: 'completed', 5: 'held', 6: 'transferring', 7: 'suspended'}
LIVE = ('idle', 'running', 'held', 'transferring', 'suspended')

# heartbeat ages (s) past which a component that should be live is stale
STALE = {'driver': 900, 'server': 180, 'bind': 300, 'shard': 600, 'producer': 1200, 'controller': 120,
         'decontam': 3600}
TICK_SLOW_S = 60          # a Super controller tick longer than this is an alert
DRIVER_ALERT_WINDOW_S = 3600


def now():
    return time.time()


def age(t):
    return None if not t else round(now() - t, 1)


def mtime(*paths):
    """Newest mtime of the paths that exist (None if none does)."""
    best = None
    for p in paths:
        try:
            m = os.stat(p).st_mtime
        except OSError:
            continue
        best = m if best is None else max(best, m)
    return best


# ---------------------------------------------------------------------------------------------------------- condor
def condor_states(job_ids, timeout=60):
    """-> ({'123.4': 'running', ...} for the given ids, None) or (None, error) when condor_q fails. A job id without
    a proc ("123") stands for its whole cluster: every proc of it is listed. Absent from the result: not in the queue."""
    clusters = sorted({str(j).split('.')[0] for j in job_ids if j})
    if not clusters:
        return {}, None
    try:
        r = subprocess.run(['condor_q', *clusters, '-af:t', 'ClusterId', 'ProcId', 'JobStatus'],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f'{type(e).__name__}: {e}'[:300]
    err = (r.stderr or '').strip()
    if r.returncode != 0 or 'Error' in err or 'SCHEDD' in err:
        return None, (err or f'condor_q rc {r.returncode}')[:300]
    out = {}
    for line in r.stdout.splitlines():
        f = line.split('\t')
        if len(f) >= 3 and f[0].strip().isdigit() and f[1].strip().isdigit():
            try:
                out[f'{f[0].strip()}.{f[1].strip()}'] = CONDOR_STATES.get(int(f[2]), f[2].strip())
            except ValueError:
                continue
    return out, None


def job_state(states, job_id):
    """A job's state from condor_states' result: 'unknown' when condor_q failed, 'gone' when it is not in the queue."""
    if not job_id:
        return None
    if states is None:
        return 'unknown'
    job_id = str(job_id)
    if '.' not in job_id:
        procs = {k: v for k, v in states.items() if k.split('.')[0] == job_id}
        return procs or 'gone'
    return states.get(job_id, 'gone')


# --------------------------------------------------------------------------------------------------------- servers
def pending_ids(rd):
    """Spec ids of the round's specs.jsonl that go to binding (bind=pending)."""
    try:
        with open(rd / 'specs.jsonl') as f:
            return {json.loads(l)['spec_id'] for l in f if '"bind": "pending"' in l}
    except OSError:
        return set()


def read_beats(rd):
    """Every round job's heartbeat (<round>/shards/<owner>.json, pipeline.claims; one per job incarnation)."""
    out = []
    try:
        names = os.listdir(rd / 'shards')
    except OSError:
        return out
    for n in names:
        if n.endswith('.json') and not n.startswith('.'):
            b = P.read_json(rd / 'shards' / n)
            if isinstance(b, dict) and b.get('owner'):
                out.append(b)
    return out


def server_metrics(urls, timeout=5):
    """(running, waiting) summed over a server's URLs from vLLM's /metrics, or None if any read fails."""
    run = wait = 0.0
    for u in urls:
        base = u.rstrip('/').removesuffix('/v1')
        try:
            t = urllib.request.urlopen(base + '/metrics', timeout=timeout).read().decode()
        except (OSError, ValueError):
            return None
        for line in t.splitlines():
            if line.startswith('vllm:num_requests_running{'):
                run += float(line.split()[-1])
            elif line.startswith('vllm:num_requests_waiting{'):
                wait += float(line.split()[-1])
    return run, wait


# ---------------------------------------------------------------------------------------------------------- status
class Status:
    def __init__(self, name, metrics=True):
        self.d = P.Driver(name)
        self.name, self.metrics = name, metrics
        self.alerts = []
        self.t = now()

    def alert(self, component, kind, message):
        self.alerts.append({'component': component, 'kind': kind, 'message': message})

    def check_job(self, component, job, state, expected):
        """Alerts for a job that should be live: held, gone (only when condor answered), unknown is not an alert."""
        states = state.values() if isinstance(state, dict) else [state]
        for k, s in (state.items() if isinstance(state, dict) else [(job, state)]):
            if s == 'held':
                self.alert(component, 'held', f'job {k} is held')
        if expected and state == 'gone':
            self.alert(component, 'gone', f'job {job} is not in the queue')
        elif expected and isinstance(state, dict) and not any(s in LIVE for s in states):
            self.alert(component, 'gone', f'no proc of job {job} is in the queue')

    def check_beat(self, component, beat_age, limit, expected, what='heartbeat'):
        if expected and beat_age is not None and beat_age > limit:
            self.alert(component, 'stale_heartbeat', f'{what} {beat_age:.0f} s old (> {limit} s)')
        elif expected and beat_age is None:
            self.alert(component, 'no_heartbeat', f'no {what} yet')

    # ---- components
    def jobs(self):
        """Every job id the status reads, for one condor_q call."""
        d, ids = self.d, []
        ids.append((P.read_json(d.run / 'driver.json') or {}).get('job_id'))
        for s in d.server_names():
            ids.append((P.read_json(P.serving_dir() / s / 'job.json') or {}).get('job_id'))
        ids.append((P.read_json(P.BINDS / d.info['bind'] / 'job.json') or {}).get('job_id'))
        ids.append((P.read_json(d.rd / 'job.json') or {}).get('job_id'))
        ids.append(d.state.get('producer_job'))
        ids += [r['job_id'] for r in self.super_receipts()]
        hold = P.read_json(d.run / 'decontam.HOLD') or {}
        ids.append(hold.get('job'))
        return [j for j in ids if j]

    def super_receipts(self):
        """The campaign's latest controller and serving receipts (one per role and index: the highest generation)."""
        o = P.read_json(self.d.camp / 'ops' / 'owned-jobs.json') or {}
        best = {}
        for r in o.get('jobs') or []:
            if r.get('role') not in ('controller', 'serving'):
                continue
            k = (r['role'], r.get('index', 0))
            if k not in best or (r.get('generation', 0), r.get('time', 0)) >= (best[k].get('generation', 0),
                                                                               best[k].get('time', 0)):
                best[k] = r
        return sorted(best.values(), key=lambda r: (r['role'], r.get('index', 0)))

    def driver(self, states):
        d = self.d
        job = (P.read_json(d.run / 'driver.json') or {}).get('job_id')
        beat = P.read_json(d.run / 'heartbeat.json') or {}
        t = beat.get('time') or mtime(d.run / 'driver.log', d.run / 'state.json')
        expected = not d.state.get('complete')
        st = job_state(states, job)
        self.check_job('driver', job, st, expected)
        self.check_beat('driver', age(t), STALE['driver'], expected and st not in ('gone', 'unknown', None))
        alerts = []
        try:
            with open(d.run / 'driver.log', 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 200_000))
                lines = f.read().decode(errors='replace').splitlines()
        except OSError:
            lines = []
        for line in lines:
            m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (ALERT .*)', line)
            if m:
                try:
                    ts = time.mktime(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'))
                except ValueError:
                    continue
                if self.t - ts < DRIVER_ALERT_WINDOW_S:
                    alerts.append(line[:400])
        for a in alerts[-10:]:
            self.alert('driver', 'driver_alert', a)
        return {'job': job, 'condor': st, 'heartbeat_age': age(t), 'expected': expected,
                'pass': beat.get('pass'), 'recent_alerts': alerts[-10:]}

    def servers(self, states):
        d = self.d
        expected_all = bool(d.info['own_servers']) and not d.state.get('servers_released')
        active = set((d.state.get('server_active') or {}).values()) | {
            s for s in d.info['servers'] if s not in (d.state.get('server_active') or {})}
        swaps = {sw['new'] for sw in (d.state.get('server_swaps') or {}).values()}
        rows, todo = [], []
        for s in d.server_names():
            job = (P.read_json(P.serving_dir() / s / 'job.json') or {}).get('job_id')
            ep = P.read_json(P.serving_dir() / s / 'endpoints.json') or {}
            row = {'name': s, 'job': job, 'condor': job_state(states, job), 'heartbeat_age': age(ep.get('updated_at')),
                   'healthy': P.healthy(s), 'active': s in active, 'swap_target': s in swaps,
                   'running': None, 'waiting': None, 'metrics': None}
            rows.append(row)
            expected = expected_all and s in active
            if job:
                self.check_job(f'server {s}', job, row['condor'], expected)
            if expected and row['condor'] == 'running':
                self.check_beat(f'server {s}', row['heartbeat_age'], STALE['server'], True, 'endpoints.json')
            if self.metrics and ep.get('urls') and row['condor'] in ('running', 'unknown'):
                todo.append((row, ep['urls']))
        if todo:
            with ThreadPoolExecutor(len(todo)) as ex:
                for (row, _), m in zip(todo, ex.map(lambda x: server_metrics(x[1]), todo)):
                    row['metrics'] = 'ok' if m else 'unknown'
                    if m:
                        row['running'], row['waiting'] = m
        return rows

    def bind(self, states):
        d, info = self.d, self.d.info
        jobinfo = P.read_json(P.BINDS / info['bind'] / 'job.json') or {}
        job = jobinfo.get('job_id') if (jobinfo.get('args') or {}).get('round') == info['round'] else None
        beat = P.read_json(d.rd / 'bind' / 'HEARTBEAT') or {}
        try:
            have = {n[:-5] for n in os.listdir(d.rd / 'bind') if n.endswith('.json')}
        except OSError:
            have = set()
        # the current specs' binding share (a re-plan replaces pending specs: older results stay in bind/)
        pending = pending_ids(d.rd)
        selected, results = len(pending), len(pending & have)
        expected = bool(selected) and not d.state.get('bind_done') and not (d.rd / 'STOP').exists()
        st = job_state(states, job)
        if job:
            self.check_job('bind', job, st, expected)
        if expected and st == 'running':
            self.check_beat('bind', age(beat.get('time')), STALE['bind'], True)
        return {'job': job, 'condor': st, 'heartbeat_age': age(beat.get('time')), 'expected': expected,
                'selected': selected, 'results': results, 'pending': selected - results, 'done_this_job': beat.get('done'),
                'todo_this_job': beat.get('todo'), 'rate_per_min': beat.get('rate_per_min'),
                'slots': beat.get('slots'), 'concurrency': beat.get('concurrency'), 'done': bool(d.state.get('bind_done'))}

    def round(self, states):
        d = self.d
        job = (P.read_json(d.rd / 'job.json') or {}).get('job_id')
        st = job_state(states, job and str(job).split('.')[0])   # one cluster, a proc per shard (job.json: <cluster>.0)
        expected = not d.state.get('round_finished')
        if job:
            self.check_job('round', job, st, expected)
        beats = read_beats(d.rd)
        sum_keys = ('claimed', 'admitted', 'accepted', 'cut', 'errors', 'done')
        live_keys = ('open', 'sessions', 'inflight')
        totals = {k: 0 for k in sum_keys + live_keys}
        shards = {}
        for b in sorted(beats, key=lambda b: b.get('time', 0)):
            for k in sum_keys:                                   # since each job (incarnation) started: summed
                totals[k] += b.get(k) or 0
            shards[b.get('shard', 0)] = b                       # the latest incarnation of each shard
        rows = []
        for i, b in sorted(shards.items()):
            a = age(b.get('time'))
            live = b.get('state') != 'exited' and a is not None and a < STALE['shard']
            if live:
                for k in live_keys:
                    totals[k] += b.get(k) or 0
            rows.append({'shard': i, 'owner': b['owner'], 'host': b.get('host'), 'state': b.get('state'),
                         'heartbeat_age': a, 'live': live, **{k: b.get(k) for k in sum_keys + live_keys},
                         'pool_load': b.get('pool_load'), 'target_inflight': b.get('target_inflight'),
                         'specs_version': b.get('specs_version')})
            proc_state = st.get(f'{job.split(".")[0]}.{i}') if isinstance(st, dict) and job else None
            if expected and b.get('state') != 'exited' and a is not None and a > STALE['shard']:
                self.alert(f'round shard {i}', 'stale_heartbeat',
                           f'{b["owner"]}: heartbeat {a:.0f} s old (> {STALE["shard"]} s; claims are taken over at the '
                           f'stale limit){f", condor {proc_state}" if proc_state else ""}')
        stop = (d.rd / 'STOP').read_text().strip()[:300] if (d.rd / 'STOP').exists() else None
        return {'job': job, 'condor': st, 'expected': expected, 'specs': d.info['specs'], 'stop': stop,
                'finished': bool(d.state.get('round_finished')), 'admission_override': P.read_json(d.rd / 'admission.json'),
                'counters': totals, 'shards': rows,
                'note': 'counters claimed/admitted/accepted/cut/errors/done are summed over every job since its start; '
                        'open/sessions/inflight over live jobs now. A claim precedes admission.'}

    def stream(self, states):
        d = self.d
        job = d.state.get('producer_job')
        st = job_state(states, job)
        end = (d.stream / 'END.json').exists()
        beat = P.read_json(d.stream / 'producer' / 'HEARTBEAT') or {}
        t = beat.get('time') or mtime(d.stream / 'manifest.json')
        expected = bool(job) and not end
        if job:
            self.check_job('stream', job, st, expected)
        if expected and st == 'running' and beat:
            self.check_beat('stream', age(t), max(STALE['producer'], 3 * (beat.get('interval') or 300)), True)
        try:
            chunks = sum(1 for n in os.listdir(d.stream / 'chunks') if n.endswith('.tar'))
        except OSError:
            chunks = 0
        man = P.read_json(d.stream / 'manifest.json', {}) or {}
        return {'job': job, 'condor': st, 'heartbeat_age': age(t), 'expected': expected, 'chunks': chunks,
                'tasks': len(man), 'end': end}

    def super(self, states):
        d = self.d
        ops = d.camp / 'ops'
        p = P.read_json(ops / 'progress.json')
        if p is None:
            return {'started': (ops / 'campaign.json').exists(), 'progress': None}
        expected = p.get('phase') != 'complete'
        receipts = self.super_receipts()
        ctl = next((r for r in receipts if r['role'] == 'controller'), None)
        ctl_state = job_state(states, ctl and ctl['job_id'])
        if ctl:
            self.check_job('super controller', ctl['job_id'], ctl_state, expected)
        tick_age = age(p.get('updated_at'))
        if expected and tick_age is not None and tick_age > TICK_SLOW_S:
            self.alert('super controller', 'tick_late', f'progress.json {tick_age:.0f} s old (> {TICK_SLOW_S} s)')
        last_slow = None
        try:
            with open(ops / 'controller' / 'slow-ticks.jsonl', 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode(errors='replace').splitlines()
            last_slow = json.loads(tail[-1]) if tail else None
        except (OSError, ValueError):
            pass
        if (expected and last_slow and self.t - last_slow.get('time', 0) < 600
                and (last_slow.get('seconds') or 0) > TICK_SLOW_S):
            self.alert('super controller', 'tick_slow', f'a {last_slow["seconds"]:.0f} s tick {age(last_slow["time"]):.0f} '
                                                       f's ago (> {TICK_SLOW_S} s): {last_slow}')
        if p.get('needs_attention') or p.get('controller_error'):
            self.alert('super controller', 'needs_attention', str(p.get('controller_error'))[:300])
        engines = []
        for r in receipts:
            if r['role'] != 'serving':
                continue
            ep = P.read_json(ops / 'serving' / f'job-{r["index"]}' / 'endpoints.json') or {}
            s = job_state(states, r['job_id'])
            engines.append({'index': r['index'], 'job': r['job_id'], 'condor': s, 'heartbeat_age': age(ep.get('updated_at')),
                            'phases': [x.get('phase') for x in ep.get('replicas') or []]})
            if s == 'held':
                self.alert(f'super engine {r["index"]}', 'held', f'job {r["job_id"]} is held')
        camp = P.read_json(ops / 'campaign.json') or {}
        ceiling = (camp.get('serving') or {}).get('jobs')
        policy = P.read_json(ops / 'admission-policy.json') or {}
        cap = min(ceiling, int(policy['serving_jobs'])) if ceiling and policy.get('serving_jobs') else ceiling
        pt = p.get('per_task') or {}
        active = p.get('active_train') or 0
        return {'started': True, 'phase': p.get('phase'), 'stage': p.get('controller_stage'), 'expected': expected,
                'controller': {'job': ctl and ctl['job_id'], 'condor': ctl_state, 'tick_age': tick_age,
                               'last_slow_tick': last_slow},
                'engines': engines, 'engines_live': sum(1 for e in engines if e['condor'] == 'running'),
                'engines_ready': len(p.get('backends_ready') or []), 'engines_cap': cap, 'engines_ceiling': ceiling,
                'active_attempts': active,
                'queued_attempts': max(0, (pt.get('missing_trajectories') or 0) - active) if pt else None,
                'unopened_tasks': max(0, (pt.get('available_tasks') or 0) - (pt.get('tasks') or 0)) if pt else None,
                'saved_tokens': p.get('train_tokens_saved'), 'saved_tokens_per_s': p.get('train_saved_tokens_per_second'),
                'trajectories': p.get('train_trajectories'), 'train_stop': p.get('train_stop'),
                'projected_tokens': pt.get('projected_tokens'), 'tokens_budget': pt.get('tokens_budget'),
                'last_tick': p.get('updated_at')}

    def decontam(self, states):
        d = self.d
        hold = P.read_json(d.run / 'decontam.HOLD')
        work = (hold or {}).get('work') or d.state.get('decontam_work')
        job = (hold or {}).get('job')
        st = job_state(states, job)
        expected = hold is not None
        if job:
            self.check_job('decontam', job, st, expected)
        if not work:
            return {'hold': hold is not None, 'job': job, 'condor': st, 'work': None}
        W = Path(work)
        last = {}
        try:
            with open(W / 'events.log', 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 100_000))
                lines = f.read().decode(errors='replace').splitlines()
        except OSError:
            lines = []
        phase, snapshot, pass_no, pairs = None, None, None, None
        for line in lines:
            if (m := re.search(r'snapshot (\d+) pass (\d+)', line)):
                snapshot, pass_no, phase = int(m.group(1)), int(m.group(2)), 'judging'
            elif (m := re.search(r'snapshot (\d+) tasks: signals', line)):
                snapshot, phase = int(m.group(1)), 'signals'
            elif (m := re.search(r'snapshot (\d+): list .* pairs judged (\d+)', line)):
                snapshot, pairs, phase = int(m.group(1)), int(m.group(2)), 'listed'
            elif 'PUBLISHED' in line:
                phase = 'published'
            elif 'run: done' in line:
                phase = 'done'
            elif re.search(r'update: \d+ units', line):
                phase = phase if phase == 'judging' else 'updating'
        try:
            passes = sorted(os.listdir(W / 'passes'))
        except OSError:
            passes = []
        cur = W / 'passes' / passes[-1] if passes else None
        files = [W / 'events.log', *(W.glob('verdicts-*.jsonl'))]
        if cur is not None:
            try:
                files += [cur / n for n in os.listdir(cur)]
            except OSError:
                pass
        t = mtime(*files)
        if expected and st == 'running':   # events.log stays quiet while a pass judges: the verdicts and pass files beat
            self.check_beat('decontam', age(t), STALE['decontam'], True, 'newest work file')
        last = lines[-1][:300] if lines else None
        return {'hold': hold is not None, 'job': job, 'condor': st, 'work': str(W), 'phase': phase,
                'snapshot': snapshot, 'pass': pass_no, 'pairs_judged_last_list': pairs,
                'current_pass_dir': cur.name if cur else None, 'heartbeat_age': age(t), 'last_event': last}

    def quota(self):
        f = self.d.state.get('fast_files')
        if not f:
            return None
        q = P.quota_cfg(self.d.cfg)
        free = f['limit'] - f['used']
        if free < q['alert_free_files']:
            self.alert('quota', 'low_quota', f'/data files: {free:,} free of {f["limit"]:,} (< {q["alert_free_files"]:,})')
        return {'used': f['used'], 'limit': f['limit'], 'free': free, 'age': age(f.get('time'))}

    def collect(self):
        states, err = condor_states(self.jobs())
        if states is None:
            self.alert('condor', 'condor_unknown', f'condor_q failed: job states unknown ({err})')
        out = {'run': self.name, 'time': self.t, 'host': socket.gethostname(),
               'condor': 'ok' if states is not None else 'unknown', 'condor_error': err}
        comps = {}
        for k, fn in (('driver', self.driver), ('servers', self.servers), ('bind', self.bind), ('round', self.round),
                      ('stream', self.stream), ('super', self.super), ('decontam', self.decontam)):
            try:
                comps[k] = fn(states)
            except Exception as e:  # noqa: BLE001  (one unreadable component must not hide the others)
                comps[k] = {'error': f'{type(e).__name__}: {e}'[:300]}
                self.alert(k, 'status_error', comps[k]['error'])
        comps['quota'] = self.quota()
        out['components'] = comps
        out['alerts'] = self.alerts
        return out


def fmt(x, unit=''):
    if x is None:
        return '?'
    if isinstance(x, float):
        x = round(x)
    return f'{x:,}{unit}' if isinstance(x, int) else str(x)


def render(s):
    """The status as text (the same fields as --json)."""
    c, out = s['components'], []
    out.append(f'run {s["run"]} at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["time"]))}; condor {s["condor"]}'
               + (f' ({s["condor_error"]})' if s['condor_error'] else ''))
    dr = c.get('driver') or {}
    out.append(f'  driver: job {dr.get("job")} {dr.get("condor")}, heartbeat {fmt(dr.get("heartbeat_age"), " s")} ago')
    for v in c.get('servers') or []:
        if not v['job']:
            continue
        out.append(f'  server {v["name"]}: job {v["job"]} {v["condor"]}, {"healthy" if v["healthy"] else "not healthy"}'
                   f', endpoints {fmt(v["heartbeat_age"], " s")} ago, running {fmt(v["running"])} waiting '
                   f'{fmt(v["waiting"])}' + ('' if v['active'] else ' (inactive)'))
    b = c.get('bind') or {}
    if 'error' not in b:
        out.append(f'  bind: job {b.get("job")} {b.get("condor")}, {b.get("results")}/{b.get("selected")} results '
                   f'({b.get("pending")} pending), {fmt(b.get("rate_per_min"))}/min, slots {fmt(b.get("slots"))}/'
                   f'{fmt(b.get("concurrency"))}, heartbeat {fmt(b.get("heartbeat_age"), " s")} ago')
    r = c.get('round') or {}
    if 'error' not in r:
        k = r['counters']
        out.append(f'  round: job {r["job"]}, {r["specs"]} specs; claimed {k["claimed"]:,} admitted {k["admitted"]:,} '
                   f'accepted {k["accepted"]:,} cut {k["cut"]:,} done {k["done"]:,}; now open {k["open"]:,} sessions '
                   f'{k["sessions"]:,} in flight {k["inflight"]:,}; STOP {r["stop"] or "-"}')
        for x in r['shards']:
            out.append(f'    shard {x["shard"]} {x["host"]} {x["state"]} beat {fmt(x["heartbeat_age"], " s")} ago: '
                       f'open {fmt(x["open"])} sessions {fmt(x["sessions"])} in flight {fmt(x["inflight"])} '
                       f'accepted {fmt(x["accepted"])} pool {fmt(x["pool_load"])}/{fmt(x["target_inflight"])}')
    st = c.get('stream') or {}
    if 'error' not in st:
        out.append(f'  stream: job {st.get("job")} {st.get("condor")}, {st.get("chunks")} chunks, {st.get("tasks"):,} '
                   f'tasks, END {"yes" if st.get("end") else "no"}')
    su = c.get('super') or {}
    if su.get('progress', True) is not None and 'error' not in su and su.get('started'):
        out.append(f'  super: {su["phase"]}, engines {su["engines_live"]} live / cap {fmt(su["engines_cap"])}, active '
                   f'{su["active_attempts"]:,}, queued {fmt(su["queued_attempts"])}, saved {fmt(su["saved_tokens"])} '
                   f'({fmt(su["saved_tokens_per_s"])}/s), tick {fmt(su["controller"]["tick_age"], " s")} ago')
    de = c.get('decontam') or {}
    if de.get('work'):
        out.append(f'  decontam: job {de.get("job")} {de.get("condor")}, {de.get("phase")} snapshot {de.get("snapshot")} '
                   f'pass {de.get("pass")}, files {fmt(de.get("heartbeat_age"), " s")} ago')
    out.append(f'  alerts ({len(s["alerts"])}):' + ('' if s['alerts'] else ' none'))
    for a in s['alerts']:
        out.append(f'    {a["component"]}: {a["kind"]}: {a["message"]}')
    return '\n'.join(out)


def status(a):
    s = Status(a.name, metrics=not a.no_metrics).collect()
    print(json.dumps(s, indent=1) if a.json else render(s), flush=True)
    return s


# ---------------------------------------------------------------------------------------------------------- replan
# R6 (#172 section 3C): m1-hero's v2h -> v2h0 switch (#165) by hand, from m1-pilot-coord/scripts/v2h0_switch.py and
# old_shards.py: hold the bind job, pre-claim the specs the shards could still start, rewrite specs.jsonl (started specs
# kept + new draws on an unused seed), start new shards and a new bind job, then drain the old shards for ~3 h and remove
# each at 0 open specs. Since #238 every round job re-reads specs.jsonl when it changes (Round.reload_specs): the shards
# drop the specs that left the file where they wait, start the new ones, and keep their started sessions. So the old
# shards' drain is gone; shards started before that code are refused (or, --restart-old-shards, removed: their open
# claims go stale and a new round job takes them over and re-runs them from scratch, #171).
REPLAN_OWNER = 'replan'
CONFIRM_POLL_S = 10


def draw(n, seed, mix, vocab):
    from . import specs as specs_mod
    return specs_mod.sample(n, seed, None, specs_mod.parse_mix(mix), vocab)


def condor_cmd(argv, timeout=120):
    """-> (ok, output) of a condor command (patched in tests)."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f'{type(e).__name__}: {e}'
    err = (r.stderr or '').strip()
    return r.returncode == 0 and 'Error' not in err, (r.stdout + err).strip()


def job_owner_and_args(job_id):
    """(Owner, Args, JobStatus) of a job, None when it is not in the queue, or 'unknown' when condor_q fails."""
    ok, out = condor_cmd(['condor_q', str(job_id), '-af:t', 'Owner', 'JobStatus', 'Args'])
    if not ok:
        return 'unknown'
    line = next((l for l in out.splitlines() if l.strip()), None)
    if not line:
        return None
    f = line.split('\t') + ['', '']
    return f[0], int(f[1]) if f[1].strip().isdigit() else None, f[2]


def submit_bind(d):
    """Submit the run's bind job (Driver.bind_args; `bind submit` writes its job.json) -> its job id or None."""
    out = d.sh(d.bind_args(), 'bind submit (replan)')
    return out.strip().splitlines()[-1] if out else None


def submit_round(d):
    out = d.sh(d.round_args(), 'round submit (replan)')
    return out.strip().splitlines()[-1] if out else None


def seeds_used(name):
    """Seeds of every round's first draw (production.seed_in_use) and of every run's re-plans."""
    used = {}
    for f in P.RUNS.glob('*/run.json'):
        info = P.read_json(f) or {}
        for r in info.get('replans') or []:
            used[r.get('seed')] = f'{f.parent.name} (replan {r.get("index")})'
    return used


class Replan:
    def __init__(self, a, log=print):
        self.a, self.say = a, log
        self.d = P.Driver(a.name)
        self.rd, self.info, self.cfg = self.d.rd, self.d.info, self.d.cfg
        self.rec = self.cfg['recipe']
        self.mix = a.mix or self.rec['minutes_mix']
        self.share = (a.asset_share if a.asset_share is not None else self.rec['asset_share']) \
            if self.rec['real_assets'] != 'off' else 0.0
        self.index = len(self.info.get('replans') or []) + 1
        self.owner = f'{REPLAN_OWNER}-{self.index}-s{a.seed}'

    # ---- checks
    def preconditions(self):
        from . import specs as specs_mod
        problems = []
        if self.a.vocab not in specs_mod.VOCAB_NAMES:
            problems.append(f'--vocab {self.a.vocab!r}: one of {specs_mod.VOCAB_NAMES}')
        try:
            specs_mod.parse_mix(self.mix)
        except ValueError as e:
            problems.append(f'--mix: {e}')
        if (self.rd / 'STOP').exists():
            problems.append(f'{self.rd}/STOP exists: the round is stopping, nothing to re-plan')
        if self.d.state.get('round_finished'):
            problems.append('the round has finished')
        if (other := P.seed_in_use(self.a.seed)) or (other := seeds_used(self.a.name).get(self.a.seed)):
            problems.append(f'seed {self.a.seed} already drew {other}: pass another --seed')
        if any(s['spec_id'].startswith(f's{self.a.seed}-') for s in self.specs):
            problems.append(f'seed {self.a.seed}: specs.jsonl already holds its draws')
        return problems

    def shards(self):
        """(live heartbeats, live ones that cannot reload specs.jsonl). A live round job without heartbeats (before
        #171) counts as one that cannot."""
        stale = 60 * float(self.cfg['round'].get('claim_stale_minutes', 10))
        live = [b for b in read_beats(self.rd) if b.get('state') != 'exited' and now() - b.get('time', 0) < stale]
        old = [b for b in live if not b.get('reload_specs')]
        job = (P.read_json(self.rd / 'job.json') or {}).get('job_id')
        if not live and job:
            q = job_owner_and_args(str(job).split('.')[0])
            if q == 'unknown':
                raise SystemExit(f'round job {job}: condor_q failed; cannot tell whether it runs (try again)')
            if q is not None and q[1] in (1, 2, 5):
                old = [{'owner': f'round job {job} (no heartbeats: started before #171)'}]
        return live, old

    # ---- the plan
    def load(self):
        from .roundpack import RoundFiles
        if RoundFiles(self.rd).is_packed():   # #245: a packed round is finished; re-plans write into work/
            raise SystemExit(f'round {self.rd.name} is packed (roundpack): nothing to re-plan')
        self.lines = [l for l in (self.rd / 'specs.jsonl').read_text().splitlines() if l.strip()]
        self.specs = [json.loads(l) for l in self.lines]
        try:
            self.results = {n[:-5] for n in os.listdir(self.rd / 'bind') if n.endswith('.json')}
        except OSError:
            self.results = set()
        try:
            self.dirs = set(os.listdir(self.rd / 'work'))
        except OSError:
            self.dirs = set()
        self._started = {}
        self.scan()

    def started(self, sid):
        """A spec is started once a round job claimed it (or it has a record); this re-plan's own pre-claims are not.
        One read for a claimed spec (records are written under a claim): m1-hero's 78,860 specs at ~3 stats each took
        5.5 min on Lustre."""
        if sid not in self.dirs:
            return False
        if sid in self._started:
            return self._started[sid]
        wd = self.rd / 'work' / sid
        try:
            v = f'"{self.owner}"' not in (wd / '.claim' / 'owner').read_text()
        except FileNotFoundError:
            v = ((wd / '.claim').exists()      # claimed, owner file not written yet: a round job's
                 or (wd / 'record.json').exists() or (wd / 'error.json').exists())
        except OSError:
            v = True
        self._started[sid] = v
        return v

    def scan(self, sids=None):
        """started() of every spec with a work dir (or of `sids`), 32 at a time."""
        todo = [sid for sid in (sids if sids is not None else (s['spec_id'] for s in self.specs)) if sid in self.dirs]
        for sid in todo:
            self._started.pop(sid, None)
        with ThreadPoolExecutor(32) as ex:
            list(ex.map(self.started, todo))

    def window(self):
        over = P.read_json(self.rd / 'admission.json') or {}
        v = over.get('bind_window')
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            return int(v)
        return int(self.cfg['round'].get('bind_window', 300))

    def startable(self):
        """Unstarted specs a round job could start now (round.bind_ready): bound or not to binding, and at most `window`
        unresolved pending specs before it in the queue. With the bind job held no further result arrives, so no other
        spec can become startable until the shards have the new file."""
        w = self.window()
        unres = [i for i, s in enumerate(self.specs) if s.get('bind') == 'pending' and s['spec_id'] not in self.results]
        limit = unres[w] if w and len(unres) > w else len(self.specs)
        return [s['spec_id'] for i, s in enumerate(self.specs[:limit + 1])
                if not self.started(s['spec_id']) and (s.get('bind') != 'pending' or s['spec_id'] in self.results)]

    def plan(self):
        started = {s['spec_id'] for s in self.specs if self.started(s['spec_id'])}
        kept = [s for s in self.specs if s['spec_id'] in started]
        self.replaced_vocab = dict(Counter(str(s.get('vocab')) for s in self.specs if s['spec_id'] not in started))
        bad = [s['spec_id'] for s in kept if s.get('bind') == 'pending' and s['spec_id'] not in self.results
               and not (self.rd / 'work' / s['spec_id'] / 'record.json').exists()]
        n = len(self.specs) - len(kept)
        drawn = draw(n, self.a.seed, self.mix, self.a.vocab) if n else []
        excluded = set(self.rec.get('bind_excluded_classes', []))
        elig = [s for s in drawn if P.eligible(s, excluded)]
        chosen = set(s['spec_id'] for s in random.Random(f'{self.a.seed}-bind').sample(elig, round(self.share * len(elig))))
        for s in drawn:
            if s['spec_id'] in chosen:
                s['bind'] = 'pending'
        ids = [s['spec_id'] for s in kept + drawn]
        if len(ids) != len(set(ids)):
            raise SystemExit('spec ids collide between the kept specs and the new draws')
        return kept, drawn, elig, chosen, bad

    # ---- side effects
    def bind_job(self):
        j = P.read_json(P.BINDS / self.info['bind'] / 'job.json') or {}
        return j.get('job_id') if (j.get('args') or {}).get('round') == self.info['round'] else None

    def hold_bind(self, job):
        """condor_hold the run's bind job (owner and --round verified) and wait until it is held or gone. A held job is
        alive to the driver, which therefore does not resubmit it meanwhile."""
        q = job_owner_and_args(job)
        if q is None:
            return 'gone'
        if q == 'unknown':
            raise SystemExit(f'bind job {job}: condor_q failed; not re-planning blind (try again)')
        owner, st, args = q
        if owner != os.environ.get('USER', 'user') or f'--round {self.info["round"]}' not in args:
            raise SystemExit(f'bind job {job}: owner {owner}, args {args[:200]!r}: not this run\'s bind job')
        if st != 5:
            ok, out = condor_cmd(['condor_hold', str(job)])
            if not ok:
                raise SystemExit(f'condor_hold {job} failed: {out[:300]}')
        for _ in range(60):
            q = job_owner_and_args(job)
            if q is None or (q != 'unknown' and q[1] == 5):
                return 'held' if q else 'gone'
            time.sleep(2)
        raise SystemExit(f'bind job {job} not held after 2 min; nothing changed except the hold (release it by hand: '
                         f'condor_release {job})')

    def preclaim(self, sids):
        """mkdir work/<spec>/.claim with this re-plan as owner: from here on no round job can claim them (a job that
        claimed one first keeps it: it is kept as started)."""
        lost = 0
        for sid in sids:
            wd = self.rd / 'work' / sid
            wd.mkdir(parents=True, exist_ok=True)
            try:
                os.mkdir(wd / '.claim')
            except FileExistsError:
                lost += 1
                continue
            P.write_json(wd / '.claim' / 'owner', {'owner': self.owner, 'replan': self.index, 'seed': self.a.seed,
                                                   'host': socket.gethostname(), 'pid': os.getpid(), 'time': now()})
            self.dirs.add(sid)
        return lost

    def run(self):
        a, say = self.a, self.say
        self.load()
        problems = self.preconditions()
        live, old = self.shards()
        bind = self.bind_job()
        start = self.startable()
        say(f'run {a.name}, round {self.info["round"]}: {len(self.specs):,} specs; bind results {len(self.results):,}; '
            f'window {self.window()}; unstarted specs the round jobs could start now: {len(start):,}')
        say(f'live round jobs: {len(live)} ({len(old)} without specs reload: '
            f'{sorted(b["owner"] for b in old)[:12]}); bind job {bind}')
        if old and not a.restart_old_shards:
            problems.append(f'{len(old)} live round jobs cannot reload specs.jsonl (started before #238): pass '
                            f'--restart-old-shards to remove them (their open specs are taken over and re-run), or '
                            f'wait for them to exit')
        if not live and not old:
            say('no live round job: the new specs run once a round job starts (the driver alerts; `round submit`)')
        if problems:
            for p in problems:
                say(f'REFUSED: {p}')
            if a.apply:
                raise SystemExit(1)
        if a.apply and not problems:
            if bind:
                say(f'bind job {bind}: {self.hold_bind(bind)}')
            self.results = {n[:-5] for n in os.listdir(self.rd / 'bind') if n.endswith('.json')} \
                if (self.rd / 'bind').exists() else set()
            start = self.startable()
            lost = self.preclaim(start)
            say(f'pre-claimed {len(start) - lost:,} unstarted specs; {lost} were claimed by round jobs meanwhile (kept)')
            self.dirs = set(os.listdir(self.rd / 'work')) if (self.rd / 'work').exists() else set()
            self.scan(start)   # the pre-claimed specs (a lost race leaves one started); nothing else can change
        kept, drawn, elig, chosen, bad = self.plan()
        if bad:
            msg = f'{len(bad)} started specs wait for a bind result: {bad[:10]}'
            if a.apply and not problems:
                say(f'WARNING {msg} (the new bind job binds them: they stay pending in the new file)')
            else:
                say(f'note: {msg}')
        kept_pending = sum(1 for s in kept if s.get('bind') == 'pending')
        summary = {'index': self.index, 'time': now(), 'seed': a.seed, 'vocab': a.vocab, 'mix': self.mix,
                   'asset_share': self.share, 'specs_before': len(self.specs), 'kept': len(kept), 'new': len(drawn),
                   'eligible_new': len(elig), 'bind_selected_new': len(chosen), 'bind_selected_kept': kept_pending,
                   'replaced_vocab': self.replaced_vocab,
                   'live_jobs': len(live), 'old_jobs': len(old), 'bind_job': bind, 'why': a.why, 'owner': self.owner}
        say(f'kept (started) specs: {len(kept):,}; new {a.vocab} specs: {len(drawn):,} (seed {a.seed}, mix {self.mix}); '
            f'eligible {len(elig):,}; to binding {len(chosen):,} (+{kept_pending:,} kept pending)')
        if not a.apply or problems:
            say('dry run: nothing written (--apply to re-plan)')
            return summary
        # ---- write: specs.jsonl (the round jobs reload it), run.json (the driver's config and overrides_log)
        new = kept + drawn
        bak = self.rd / f'specs.before-replan-{self.index}.jsonl'
        if not bak.exists():
            bak.write_text(''.join(l + '\n' for l in self.lines))
        tmp = self.rd / '.specs.jsonl.replan.tmp'
        tmp.write_text(''.join(json.dumps(s) + '\n' for s in new))
        os.replace(tmp, self.rd / 'specs.jsonl')
        st = os.stat(self.rd / 'specs.jsonl')
        summary['specs_version'] = [st.st_ino, st.st_size, st.st_mtime_ns]
        P.write_json(self.rd / f'replan-{self.index}.json', summary)
        say(f'wrote {self.rd}/specs.jsonl: {len(kept):,} kept + {len(drawn):,} new = {len(new):,} (before: {bak.name})')
        info = P.read_json(self.d.run / 'run.json')
        self.info = dict(self.info)       # the run.json the driver started from (kept for the checks below)
        old_recipe = dict(info['config']['recipe'])
        info['config']['recipe'].update(vocab=a.vocab, minutes_mix=self.mix,
                                        **({'asset_share': self.share} if a.asset_share is not None else {}))
        info['specs'] = len(new)
        info['bind_selected'] = len(chosen) + kept_pending
        info.setdefault('replans', []).append(summary)
        info.setdefault('overrides_log', []).append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'by': f'production replan {self.index}',
            'change': f'recipe.vocab {old_recipe.get("vocab")} -> {a.vocab}, minutes_mix {old_recipe.get("minutes_mix")} '
                      f'-> {self.mix}: {len(kept)} started specs kept, {len(drawn)} unstarted replaced (seed {a.seed}, '
                      f'{len(chosen)} to binding); {a.why}'.rstrip('; ')})
        P.write_json(self.d.run / 'run.json', info)
        # ---- bind job: the held one goes, a new one binds the new pending specs (its slots ramp up, bind --ramp-minutes)
        self.d.info = info
        if bind:
            q = job_owner_and_args(bind)
            if q not in (None, 'unknown'):
                ok, out = condor_cmd(['condor_rm', str(bind)])
                say(f'old bind job {bind} removed' if ok else f'WARNING condor_rm {bind} failed: {out[:200]}')
        if info['bind_selected'] and not self.d.state.get('bind_done'):
            job = submit_bind(self.d)
            say(f'new bind job {job}' if job else 'WARNING bind submit failed: the driver resubmits it on its next pass')
        elif info['bind_selected']:
            say('WARNING the driver marked binding done before this re-plan: it will not supervise the new bind job; '
                'restart the driver (launch) after submitting `bind submit` by hand')
        if info['bind_selected'] and not self.info.get('bind_selected'):
            say('WARNING the run was planned without binding: the running driver read run.json at its start and does '
                'not supervise binding; restart it (launch) to pick up the new run.json')
        # ---- old round jobs (no reload): remove them; a new round job takes their claims over at the stale limit
        if old:
            job = (P.read_json(self.rd / 'job.json') or {}).get('job_id')
            if job:
                q = job_owner_and_args(job.split('.')[0])
                if q not in (None, 'unknown') and f'--round {self.info["round"]}' in q[2]:
                    ok, out = condor_cmd(['condor_rm', job.split('.')[0]])
                    say(f'old round jobs {job.split(".")[0]} removed' if ok else f'WARNING condor_rm failed: {out[:200]}')
            new_job = submit_round(self.d)
            say(f'new round job {new_job}: it takes over the old jobs\' open claims once their heartbeats are '
                f'{self.cfg["round"].get("claim_stale_minutes", 10)} min old' if new_job else
                'WARNING round submit failed: submit it by hand')
        else:
            self.confirm(summary['specs_version'])
        return summary

    def confirm(self, version):
        """Wait until every live round job reports the new specs.jsonl in its heartbeat."""
        deadline = now() + 60 * self.a.confirm_minutes
        while True:
            live, _ = self.shards()
            behind = [b['owner'] for b in live if b.get('specs_version') != version]
            if not behind:
                self.say(f'every live round job ({len(live)}) runs the new specs.jsonl')
                return True
            if now() > deadline:
                self.say(f'WARNING {len(behind)} live round jobs have not reloaded specs.jsonl after '
                         f'{self.a.confirm_minutes} min: {behind[:12]} (check their round logs)')
                return False
            time.sleep(CONFIRM_POLL_S)


def replan(a):
    return Replan(a).run()

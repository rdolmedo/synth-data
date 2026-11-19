"""Super traces of the synthesized tasks with the TB-Sci-lite eval's own harness (#353, #348): one trace per task.

The eval is `aa_evals.tbsci_lite.pi --pool v2.1` = aa_evals.common.pi_eval (Pi 1.2.0 pi-cont16k, pi180) with
131,072-token context and submit at the limit, 16,384 output tokens per request, the registry's request block
(pi_thinking high, inline_reasoning_history, T 1.0 / top_p 0.95), 2 h agent cap on the environment clock, offline
agent, harbor_ns multi-uid, NsSidecarEnvironment, HARBOR_NS_MEMCAP 1.5 and CPU view. This driver runs exactly that,
`bench = tbsci_lite`, on the coordinator's Super engines (serving:m5-super-<k>), over chunks of accepted tasks:

  - accepted = round record status valid and judge pass (task_synthesis round, m5/tbsci_lite/round_sci.py);
  - a chunk becomes one pi_eval run on one engine (runs live under ROOT/traces/runs/tbsci_lite/<run>);
  - per engine, the runs' live max_in_flight (RUN/control.json) sum to at most alloc.json's per-engine cap (re-read
    every 60 s); a new chunk starts on an engine once its queued (unclaimed) attempts run out;
  - finished attempts are exported (export.py) as they land; at the token target no new chunk starts and the open
    rounds get STOP; --cut-tail stops the runs' workers (running attempts are kept as truncated).

  python -m m5.tbsci_lite.traces submit --name pilot --rounds p1 [--target-tokens N] [--chunk 100]   # CPU job
  python -m m5.tbsci_lite.traces run ...       # the driver loop (inside the job; restartable, state in ROOT/traces/<name>)
  python -m m5.tbsci_lite.traces status --name pilot
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common.pi_eval import control
from .common import PY, ROOT, bad_nodes, durable_write, read_strict, frozen_checkout, requirements_excluding

TRACES = ROOT / 'traces'
RUNS_ROOT = TRACES / 'runs'
os.environ.setdefault('AA_RUNS_ROOT', str(RUNS_ROOT))
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
BENCH = 'tbsci_lite'
SEED_NAMESPACE = 'm5-tbsci-lite-synth-v1'
ENV_CLASS = 'aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment'
# The lite eval's pi_eval defaults (aa_evals/tbsci/pi.py DEFAULTS + aa_evals/tbsci_lite/pi.py), minus the lite pool's
# own items (tasks tar, oracle runs, validation, excluded tasks, per-task agent classes for one lite task).
DEFAULTS = {'bench': BENCH, 'seed_namespace': SEED_NAMESPACE, 'attempts': 1, 'agent_cap_sec': 7200.0,
            'per_task_cap': True, 'multiuid': True, 'env_class': ENV_CLASS,
            # HARBOR_NS_EVICT_IDLE_H: one image per task, used once; evict after 15 min idle (see RunHealth)
            'harbor_env': {'HARBOR_NS_MEMCAP': '1.5', 'HARBOR_NS_CPU_VIEW': '1', 'HARBOR_NS_EVICT_IDLE_H': '0.25'},
            'aa_metric_key': None,
            'notes': 'm5 #353: Super traces of synthesized TB-Sci-lite-style tasks; the lite eval protocol '
                     '(aa_evals/tbsci_lite/pi.py: Pi pi180, 131,072 context, 2 h agent cap, offline agent)'}
# Worker shape: aa_evals/pipeline.json `tbsci_lite_synth` (#451: shapes come from pipeline.json, not launchers) -- the
# measured synth workload (pilot ~0.2-0.5 CPU per attempt; prod3 16-slot workers at 96 GB peaked 8.7-38 GB, 0 holds),
# not the eval's tbsci_lite shape. Slots, disk and bid stay here.
SHAPE_KEY = 'tbsci_lite_synth'
WORKER = ['--slots', '16', '--shape-key', SHAPE_KEY, '--worker-disk-gb', '200', '--cpu-bid', '36']
# 24 slots / 64 GB went over memory on v2a (64.2 GB peak, held); 16 slots at 6 GB each, ~0.2-0.5 CPU each


def log(d, msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(d / 'driver.log', 'a') as f:
        f.write(line + '\n')


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def write_json(p, obj):
    durable_write(p, json.dumps(obj, indent=1), check=json.loads)


def alloc():
    a = read_json(ALLOC, {}) or {}
    s = a.get('super', {})
    return list(s.get('serving') or []), int((s.get('max_inflight_per_engine') or {}).get(BENCH, 0))


def engine_ok(name):
    info = read_json(f'/data/aai/tbench/tasksynth/serving/{name}/endpoints.json')
    return bool(info and info.get('healthy_at') and time.time() - info.get('updated_at', 0) < 120)


class Accepted:
    """Accepted tasks of the rounds, in record order (scan of new work/<spec>/record.json; decided specs cached)."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.decided = {}   # spec -> (round, accepted?)
        self.order = []

    def scan(self):
        for r in self.rounds:
            work = ROOT / 'rounds' / r / 'work'
            if not work.is_dir():
                continue
            for e in os.scandir(work):
                if e.name in self.decided:
                    continue
                rec = read_json(Path(e.path) / 'record.json')
                if rec is None:
                    continue
                ok = rec.get('status') == 'valid' and bool((rec.get('judge') or {}).get('pass'))
                self.decided[e.name] = (r, ok)
                if ok and (ROOT / 'rounds' / r / 'tasks' / e.name / 'task.toml').exists():
                    self.order.append(e.name)
        return self.order

    def round_of(self, spec):
        return self.decided[spec][0]

    def rounds_finished(self):
        """Every round's specs are decided (records for all of specs.jsonl) or the round was stopped and drained."""
        for r in self.rounds:
            d = ROOT / 'rounds' / r
            n = sum(1 for _ in open(d / 'specs.jsonl')) if (d / 'specs.jsonl').exists() else 0
            if sum(1 for s, (rr, _) in self.decided.items() if rr == r) < n and not (d / 'summary.json').exists():
                return False
        return True


REQUEUE_REASONS = {'worker_exception', 'no_harbor_result'}
MAX_REQUEUES = 2
ENOSPC_WINDOW, ENOSPC_N, REPLACE_GAP = 900, 3, 1200


def _ts(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except ValueError:
        return 0.0


def _true(v):
    return v is True or str(v).lower() == 'true'


class RunHealth:
    """Worker scratch filling up (prod1/prod2, 2026-11-12): about 3 h into a worker's life every attempt fails in
    seconds with OSError(28, 'No space left on device') or NoHarborResult, and pi_eval's 6 retries all land on the
    same worker, so 257 prod1 and 128+ prod2 tasks ended as infrastructure failures within minutes. Per run:
      - ENOSPC_N worker ENOSPC failures within ENOSPC_WINDOW s -> the run's workers are removed and replaced (fresh
        scratch; the claim reaper retries their in-flight attempts), at most once per REPLACE_GAP s;
      - a task whose latest attempt is an infrastructure failure in REQUEUE_REASONS goes back to pending (a new chunk),
        at most MAX_REQUEUES times."""

    def __init__(self):
        self.seen = {}        # run -> done file names read
        self.cand = {}        # run -> tasks with an infra failure in REQUEUE_REASONS
        self.enospc = {}      # run -> finished_at times of ENOSPC failures
        self.replaced = {}    # run -> time of the last replacement

    def scan(self, run):
        d = Path(run) / 'queue' / 'done'
        if not d.is_dir():
            return
        seen = self.seen.setdefault(str(run), set())
        for n in os.listdir(d):
            if n in seen:
                continue
            rec = read_json(d / n)
            if rec is None:
                continue   # being written: read it next time
            seen.add(n)
            if not _true(rec.get('infrastructure_failure')):
                continue
            if 'No space left' in str(rec.get('exception_message')):
                self.enospc.setdefault(str(run), []).append(_ts(rec.get('finished_at')))
            if rec.get('infrastructure_reason') in REQUEUE_REASONS and rec.get('task'):
                self.cand.setdefault(str(run), set()).add(rec['task'])

    def needs_replacement(self, run, now):
        recent = [t for t in self.enospc.get(str(run), []) if now - t < ENOSPC_WINDOW]
        return len(recent) >= ENOSPC_N and now - self.replaced.get(str(run), 0) > REPLACE_GAP

    def final_failures(self, run):
        """Candidates whose latest attempt is done and is an infrastructure failure in REQUEUE_REASONS."""
        cand = self.cand.get(str(run))
        if not cand:
            return []
        q = Path(run) / 'queue'
        latest = {}
        for sub in ('claims', 'done'):
            for k in (os.listdir(q / sub) if (q / sub).is_dir() else []):
                base, _, r = k.removesuffix('.json').rpartition('__r')
                task = base.split('__a')[0]
                if task in cand:
                    latest[task] = max(latest.get(task, ('', -1)), (k.removesuffix('.json'), int(r) if r.isdigit() else 0),
                                       key=lambda x: x[1])
        out = []
        for task, (key, _) in latest.items():
            rec = read_json(q / 'done' / f'{key}.json')
            if rec and _true(rec.get('infrastructure_failure')) and rec.get('infrastructure_reason') in REQUEUE_REASONS:
                out.append(task)
        return out


def replace_workers(run, d):
    """condor_rm the run's live workers (jobs.json) and submit as many fresh ones (at least 1)."""
    cfg = read_json(Path(run) / 'config.json')
    jobs = read_json(Path(run) / 'jobs.json', []) or []
    clusters = [str(j['cluster']) for j in jobs]
    live = []
    if clusters:
        q = subprocess.run(['condor_q', *clusters, '-af', 'ClusterId'], capture_output=True, text=True)
        live = sorted(set(q.stdout.split()))
    if live:
        subprocess.run(['condor_rm', *live], capture_output=True, text=True)
    from aa_evals.common.harbor_mswea import run as hm_run
    from aa_evals.common.pi_eval import run as pe_run
    hm_run.EXCLUDE_NODES = tuple(sorted(set(hm_run.EXCLUDE_NODES) | set(bad_nodes())))
    pe_run.EXCLUDE_NODES = hm_run.EXCLUDE_NODES
    out = pe_run.submit_workers(Path(run), cfg, max(1, len(live)))
    log(d, f'{Path(run).name}: worker scratch full (ENOSPC): removed {",".join(live) or "-"}, submitted {out}')


SLOTS = int(dict(zip(WORKER[::2], WORKER[1::2]))['--slots'])
ORPHAN_SEC, TOPUP_GAP = 600, 600


def live_workers(run):
    """The run's workers (jobs.json clusters) that are idle or running in condor."""
    cl = [str(j['cluster']) for j in (read_json(Path(run) / 'jobs.json', []) or [])]
    if not cl:
        return []
    q = subprocess.run(['condor_q', *cl, '-af', 'ClusterId', 'JobStatus'], capture_output=True, text=True)
    return [x.split()[0] for x in q.stdout.splitlines() if len(x.split()) == 2 and x.split()[1] in ('1', '2')]


def add_workers(run, n, d, why):
    cfg = read_json(Path(run) / 'config.json')
    from aa_evals.common.harbor_mswea import run as hm_run
    from aa_evals.common.pi_eval import run as pe_run
    hm_run.EXCLUDE_NODES = tuple(sorted(set(hm_run.EXCLUDE_NODES) | set(bad_nodes())))
    pe_run.EXCLUDE_NODES = hm_run.EXCLUDE_NODES
    out = pe_run.submit_workers(Path(run), cfg, n)
    log(d, f'{Path(run).name}: {why}: submitted {n} worker(s) {out}')


def unfinished_tasks(run):
    """Items of a run whose latest attempt has no done record."""
    q = Path(run) / 'queue'
    items = [x.split()[0] for x in open(q / 'items.txt') if x.strip()] if (q / 'items.txt').exists() else []
    latest = {}
    for sub in ('claims', 'done'):
        for k in (os.listdir(q / sub) if (q / sub).is_dir() else []):
            base, _, r = k.removesuffix('.json').rpartition('__r')
            t = base.split('__a')[0]
            latest[t] = max(latest.get(t, (-1, '')), (int(r) if r.isdigit() else 0, k.removesuffix('.json')))
    done = set(os.path.splitext(k)[0] for k in (os.listdir(q / 'done') if (q / 'done').is_dir() else []))
    return [t for t in items if t not in latest or latest[t][1] not in done]


def run_counts(run):
    """(items, done, claimed) of a pi_eval run from its queue directories."""
    items = sum(1 for _ in open(run / 'queue' / 'items.txt')) if (run / 'queue' / 'items.txt').exists() else 0
    done_keys = os.listdir(run / 'queue' / 'done') if (run / 'queue' / 'done').is_dir() else []
    claim_keys = os.listdir(run / 'queue' / 'claims') if (run / 'queue' / 'claims').is_dir() else []
    # a retried attempt leaves a done file for each retry (<task>__a<k>__r<n>.json): count an item done only when
    # its latest retry is done, and claimed once per item; a stale r0 next to a running r1 is not done (v2a stopped
    # early when 21 retried r0s made one chunk's open count negative and cancelled the others' open attempts)
    def split(k):
        base, _, r = k.removesuffix('.json').rpartition('__r')
        return base, int(r) if r.isdigit() else 0
    latest = {}
    for k in claim_keys + done_keys:
        b, r = split(k)
        latest[b] = max(latest.get(b, -1), r)
    done = sum(1 for k in done_keys if split(k)[1] == latest[split(k)[0]])
    claimed = len(latest)
    return items, min(done, items) if items else done, min(claimed, items) if items else claimed


def create_run(d, name, chunk_id, tasks, acc, engine, cap):
    """Copy the chunk's task dirs, create the pi_eval run (it builds the six-hour bundle and submits its workers),
    then drop the copies (the bundle tar holds them)."""
    from aa_evals.common.harbor_mswea import run as hm_run
    from aa_evals.common.pi_eval import run as pe_run
    hm_run.EXCLUDE_NODES = tuple(sorted(set(hm_run.EXCLUDE_NODES) | set(bad_nodes())))
    pe_run.EXCLUDE_NODES = hm_run.EXCLUDE_NODES   # pi_eval/run.py imported the tuple by name: patch its copy too
    src = TRACES / name / 'chunks' / chunk_id
    if src.exists():
        shutil.rmtree(src)
    (src / 'tasks').mkdir(parents=True)
    for t in tasks:
        shutil.copytree(ROOT / 'rounds' / acc.round_of(t) / 'tasks' / t, src / 'tasks' / t, symlinks=True)
    run_id = chunk_id
    argv = ['--run-id', run_id, '--model', f'serving:{engine}', '--tasks-tar', str(src), '--max-in-flight', str(cap),
            '--attempts', '1', *WORKER,
            # 2 h wall clock = a hang backstop here (Super-speed generation; the 131,072-token context
            # and its finish notice end attempts). The eval's model-time caps: 2 h without the capped
            # protocol (part of that protocol, #344); under it 12 h by default, a hang backstop
            # (CAPPED_WALL_CAP_SEC in aa_evals/campaign/launch.py, #531 B6)
            '--agent-wall-cap-sec', '7200']
    pe_run.main(argv, DEFAULTS)
    shutil.rmtree(src, ignore_errors=True)
    run = RUNS_ROOT / BENCH / run_id
    names = set((read_json(run / 'config.json', {}) or {}).get('task_names') or [])
    return run, [t for t in tasks if t in names]


def fresh_chunk_id(name, k):
    """A chunk id with no run dir and no bundle yet: pi_eval reuses an existing pi_six_hour_<id>.tar by name, so a
    chunk id left by a driver that died mid-creation would run the stale bundle's tasks (the pilot lost 15 that way)."""
    base = f'{name}-c{k:03d}'
    for suffix in [''] + [chr(c) for c in range(ord('b'), ord('z') + 1)]:
        cid = base + suffix
        if not (RUNS_ROOT / BENCH / cid).exists() and not (RUNS_ROOT / BENCH / 'bundles' / f'pi_six_hour_{cid}.tar').exists():
            return cid
    raise RuntimeError(f'no free chunk id for {base}')


def drive(a):
    d = TRACES / a.name
    d.mkdir(parents=True, exist_ok=True)
    state = read_strict(d / 'state.json') or {'chunks': [], 'assigned': [], 'stopped': False}   # EIO must not reset it
    write_json(d / 'config.json', vars(a))
    acc = Accepted(a.rounds)
    from . import export as ex
    exporter = ex.Exporter(d / 'export')
    last_scan = 0
    health = RunHealth()
    c_topup_at = [0.0, 0.0]
    last_requeue = 0
    state.setdefault('requeued', {})   # task -> chunk ids it was handed back from
    while True:
        now = time.time()
        engines, cap = alloc()
        if now - last_scan > a.scan_sec:
            acc.scan()
            last_scan = now
        assigned = set(state['assigned'])
        pending = [t for t in acc.order if t not in assigned]
        # export what finished
        for c in state['chunks']:
            run = Path(c['run'])
            if run.exists():
                exporter.collect(run)
        tokens = exporter.tokens()
        open_attempts = 0
        per_engine = {}
        for c in state['chunks']:
            run = Path(c['run'])
            items, done, claimed = run_counts(run)
            c['items'], c['done'], c['claimed'] = items, done, claimed
            c['open'] = 0 if c.get('orphaned') else max(0, items - done)
            open_attempts += c['open']
            if c['open'] > 0:
                per_engine.setdefault(c['engine'], []).append(c)
        mean_tok = exporter.mean_tokens() or a.tokens_prior
        projected = tokens + open_attempts * mean_tok
        if a.target_tokens and not state['stopped'] and projected + len(pending) * mean_tok >= a.target_tokens * (1 + a.margin):
            for r in a.rounds:
                stop = ROOT / 'rounds' / r / 'STOP'
                if not stop.exists():
                    stop.write_text(f'traces {a.name}: projected {projected:.0f} + pending >= target\n')
            log(d, f'supply covers the target: STOP written to the rounds (exported {tokens}, projected {projected:.0f})')
            state['stopped'] = True
        # other drivers of this worker (TRACES/*/state.json) share the per-engine cap: their open attempts count
        external = {}
        others = {}   # engine -> other drivers' chunks with open attempts
        for other in TRACES.glob('*/state.json'):
            if other.parent.name == a.name:
                continue
            for c in (read_json(other, {}) or {}).get('chunks', []):
                if max(0, c.get('open', 0)) > 0:
                    external[c['engine']] = external.get(c['engine'], 0) + c['open']
                    others.setdefault(c['engine'], []).append(c)
        # live caps: every driver applies the same rule to all drivers' chunks on an engine (oldest chunk first,
        # each its open count), and writes only its own chunks' share, so the shares sum to at most the cap.
        # (Subtracting the other drivers' open counts starved an old chunk behind a newer driver's 190-item queue.)
        for eng, cs in per_engine.items():
            left = cap if eng in engines else 0
            mine = {id(c) for c in cs}
            for c in sorted(cs + others.get(eng, []), key=lambda c: c.get('created', 0)):
                give = max(0, min(c['open'], left))
                left -= give
                if id(c) not in mine:
                    continue
                c['share'] = give
                if control.read(c['run']).get('max_in_flight') != give:   # read-merge-write under its lock (#451)
                    control.update(c['run'], {'max_in_flight': give})
        if now - c_topup_at[1] > a.scan_sec:   # a finished run's workers idle on (6 held 48 CPU for 12 h, 11-13)
            c_topup_at[1] = now
            for c in state['chunks']:
                if c['open'] == 0 and not c.get('reaped') and Path(c['run']).exists():
                    live = live_workers(c['run'])
                    if live:
                        subprocess.run(['condor_rm', *live], capture_output=True, text=True)
                        log(d, f'{c["id"]}: finished: removed idle workers {",".join(live)}')
                    c['reaped'] = True
        for c in state['chunks']:   # a finished chunk keeps no share (v2a-c000 was left at 21)
            if c['open'] == 0 and Path(c['run']).exists():
                if control.read(c['run']).get('max_in_flight', 0) > 1:
                    control.update(c['run'], {'max_in_flight': 0})
        # new chunks: an engine whose runs have no unclaimed attempts left takes the next chunk
        if not (a.target_tokens and projected >= a.target_tokens):
            # pack (coordinator, 11-13): the most-loaded engine that still has room under the cap first, so the
            # tail concentrates on few engines; load = open attempts on it (all drivers), room = cap - min(cap, load)
            load = {e: sum(c['open'] for c in per_engine.get(e, [])) + external.get(e, 0) for e in engines}
            for eng in sorted(engines, key=lambda e: -min(cap, load[e])):
                if not pending or cap - min(cap, load[eng]) <= 0 or state.get('tail_stopped') or not engine_ok(eng):
                    continue
                cs = per_engine.get(eng, [])
                unclaimed = sum(max(0, c['items'] - c['claimed']) for c in cs)
                if unclaimed > 0:
                    continue
                if len(pending) < a.min_chunk and not acc.rounds_finished():
                    continue
                take = pending[:a.chunk]
                k = len(state['chunks'])
                chunk_id = fresh_chunk_id(a.name, k)
                try:
                    run, took = create_run(d, a.name, chunk_id, take, acc, eng, max(1, cap - external.get(eng, 0)))
                except (SystemExit, Exception) as e:  # noqa: BLE001
                    log(d, f'chunk {chunk_id} on {eng} failed to start: {type(e).__name__}: {e}')
                    continue
                state['chunks'].append({'id': chunk_id, 'engine': eng, 'run': str(run), 'tasks': len(took),
                                        'created': time.time(), 'items': len(took), 'done': 0, 'claimed': 0,
                                        'open': len(took)})
                state['assigned'] += took
                pending = [t for t in pending if t not in set(took)]
                lost = [t for t in take if t not in set(took)]
                log(d, f'chunk {chunk_id}: {len(took)} tasks on {eng} (cap {cap})'
                       + (f'; {len(lost)} not in the run, left pending: {",".join(lost)}' if lost else ''))
                if lost:
                    break   # retry them in a later pass (a new chunk id)
        # an engine the coordinator stops serving (m5-super-5..7, 2026-11-12 19:21 UTC) strands its chunks: after
        # ORPHAN_SEC their workers are removed and the tasks without a final attempt go back to pending
        unserved = state.setdefault('unserved_since', {})
        for eng in {c['engine'] for c in state['chunks']}:
            if eng in engines:
                unserved.pop(eng, None)
            else:
                unserved.setdefault(eng, now)
        for c in state['chunks']:
            since = unserved.get(c['engine'])
            if c.get('orphaned') or c['open'] <= 0 or not since or now - since < ORPHAN_SEC:
                continue
            back = set(unfinished_tasks(c['run']))
            live = live_workers(c['run'])
            if live:
                subprocess.run(['condor_rm', *live], capture_output=True, text=True)
            c['orphaned'], c['open'] = True, 0
            state['assigned'] = [t for t in state['assigned'] if t not in back]
            log(d, f'{c["id"]}: {c["engine"]} not served for {(now - since) / 60:.0f} min: removed workers '
                   f'{",".join(live) or "-"}, {len(back)} unfinished tasks back to pending')
        # cap above one worker's slots (alloc 24 per engine, workers 16 slots): add workers to cover the share
        if now - c_topup_at[0] > a.scan_sec:
            c_topup_at[0] = now
            for c in state['chunks']:
                if c.get('orphaned') or c['open'] <= 0 or now - c.get('topped', 0) < TOPUP_GAP:
                    continue
                need = math.ceil(min(c.get('share', 0), c['open']) / SLOTS)
                live = live_workers(c['run'])
                if need > len(live):
                    c['topped'] = now
                    try:
                        add_workers(c['run'], need - len(live), d, f'share {c.get("share")} > {len(live)} x {SLOTS} slots')
                    except Exception as e:  # noqa: BLE001
                        log(d, f'{c["id"]}: adding workers failed: {type(e).__name__}: {e}')
        # worker scratch full: replace the run's workers; tasks that ended as infrastructure failures go back
        requeued_now = []
        for c in state['chunks']:
            run = Path(c['run'])
            if not run.exists():
                continue
            health.scan(run)
            if c['open'] > 0 and health.needs_replacement(run, now):
                health.replaced[str(run)] = now
                try:
                    replace_workers(run, d)
                except Exception as e:  # noqa: BLE001
                    log(d, f'{c["id"]}: worker replacement failed: {type(e).__name__}: {e}')
        if now - last_requeue > a.scan_sec:
            last_requeue = now
            assigned_set = set(state['assigned'])
            for c in state['chunks']:
                for t in health.final_failures(Path(c['run'])):
                    froms = state['requeued'].setdefault(t, [])
                    if c['id'] in froms or len(froms) >= MAX_REQUEUES or t not in assigned_set:
                        continue
                    froms.append(c['id'])
                    assigned_set.discard(t)
                    requeued_now.append(t)
            if requeued_now:
                state['assigned'] = [t for t in state['assigned'] if t in assigned_set]
                log(d, f'requeued {len(requeued_now)} tasks whose last attempt was an infrastructure failure '
                       f'({",".join(requeued_now[:5])}{"..." if len(requeued_now) > 5 else ""})')
        # tail rule (coordinator, 11-13): stop at --stop-total exported tokens over all drivers, or when nothing is
        # queued and fewer than --tail-inflight attempts run; the live workers are removed (stragglers are dropped,
        # not exported truncated) and no chunk starts again
        if (a.stop_total or a.tail_inflight) and not state.get('tail_stopped') and state['chunks']:
            others = sum((read_json(h, {}) or {}).get('exported_tokens', 0) for h in TRACES.glob('*/HEARTBEAT')
                         if h.parent.name != a.name)
            inflight = unclaimed = 0
            for c in state['chunks']:
                if c['open'] > 0:
                    inflight += c['claimed'] - (c['items'] - c['open'])
                    unclaimed += c['items'] - c['claimed']
            why = None
            if a.stop_total and tokens + others >= a.stop_total:
                why = f'total {tokens + others:.0f} >= {a.stop_total:.0f}'
            elif a.tail_inflight and not pending and unclaimed == 0 and 0 < inflight < a.tail_inflight:
                why = f'tail: nothing queued, {inflight} in flight < {a.tail_inflight}'
            if why:
                state['tail_stopped'], state['tail_stopped_at'] = True, now
                removed = []
                for c in state['chunks']:
                    if c['open'] > 0 and not c.get('orphaned'):
                        write_json(Path(c['run']) / 'control.json',
                                   {**(read_json(Path(c['run']) / 'control.json', {}) or {}), 'max_in_flight': 0})
                        live = live_workers(c['run'])
                        if live:
                            subprocess.run(['condor_rm', *live], capture_output=True, text=True)
                            removed += live
                        c['orphaned'], c['open'] = True, 0
                log(d, f'TAIL STOP ({why}): no more launches; removed workers {",".join(removed) or "-"} '
                       f'({inflight} in flight dropped); exported {tokens} here, {tokens + others} over all drivers')
                open_attempts = 0
        # tail cut
        if a.cut_tail and state['stopped'] and a.target_tokens and tokens >= a.target_tokens:
            for c in state['chunks']:
                p = Path(c['run']) / 'STOP_WORKERS'
                if not p.exists():
                    p.write_text('target reached\n')
        write_json(d / 'state.json', state)
        write_json(d / 'HEARTBEAT', {'time': time.time(), 'exported_tokens': tokens, 'exported': exporter.count(),
                                     'projected_tokens': projected, 'pending_tasks': len(pending),
                                     'accepted': len(acc.order), 'open_attempts': open_attempts,
                                     'engines': engines, 'cap': cap,
                                     'per_engine_open': {e: sum(c['open'] for c in cs) for e, cs in per_engine.items()}})
        # done: nothing open, and nothing pending unless the token target is covered. STOP only ends generation: a
        # stopped driver used to exit with 949 accepted prod3 tasks untraced (STOP counted them as supply).
        reached = bool(a.target_tokens) and projected >= a.target_tokens
        done_all = (not requeued_now and bool(state['chunks']) and open_attempts == 0
                    and (acc.rounds_finished() or state['stopped']) and (not pending or reached)) or \
            (state.get('tail_stopped') and now - state.get('tail_stopped_at', now) > 3 * a.tick_sec)  # last exports in
        if done_all:
            exporter.flush()
            log(d, f'all traces done: {exporter.count()} exported, {tokens} tokens')
            return
        time.sleep(a.tick_sec)


def submit(a):
    code = frozen_checkout()
    d = TRACES / a.name
    d.mkdir(parents=True, exist_ok=True)
    args = ' '.join(f'--{k.replace("_", "-")} {v if not isinstance(v, list) else " ".join(v)}'
                    for k, v in vars(a).items() if k not in ('op', 'bid') and v not in (None, False))
    args = args.replace('--cut-tail True', '--cut-tail')
    env = {'USER': 'user', 'HOME': '/home/user', 'PYTHONNOUSERSITE': '1', 'PATH': '/usr/local/bin:/usr/bin:/bin',
           'PYTHONPATH': str(code), 'AA_RUNS_ROOT': str(RUNS_ROOT)}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = d / 'driver.sub'
    sub.write_text(f'''universe = vanilla
executable = {PY}
arguments = "-u -m m5.tbsci_lite.traces run {args}"
initialdir = {code}
output = {d}/driver.$(Cluster).out
error = {d}/driver.$(Cluster).err
log = {d}/driver.condor.log
request_cpus = 2
request_memory = 8192
request_disk = 20G
+JobBatchName = "m5-tbsci-traces-{a.name}"
+MaxRuntime = 259200
getenv = false
environment = "{environment}"
requirements = {requirements_excluding(bad_nodes())}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip())


def status(a):
    d = TRACES / a.name
    hb = read_json(d / 'HEARTBEAT', {})
    st = read_json(d / 'state.json', {}) or {}
    print(json.dumps({'heartbeat_age_s': round(time.time() - hb.get('time', 0)), **hb,
                      'chunks': [{k: c.get(k) for k in ('id', 'engine', 'items', 'done', 'claimed')}
                                 for c in st.get('chunks', [])]}, indent=1))


def resubmit_workers(a):
    """Replace a run's workers (a node went bad, or a worker was held): the run's stored worker shape is set to the
    measured tbsci_lite_synth shape at WORKER's slots (its frozen pi_eval's worker_shape; a snapshot older than #451
    keeps the stored shape) and N workers are submitted with the current exclusions. Their claim reaper retries the dead
    worker's attempts."""
    run = RUNS_ROOT / BENCH / a.run
    cfg = read_json(run / 'config.json')
    sys.path.insert(0, cfg['code'])
    from aa_evals.common.harbor_mswea import run as hm_run
    from aa_evals.common.pi_eval import run as pe_run
    pe_run.EXCLUDE_NODES = tuple(sorted(set(hm_run.EXCLUDE_NODES) | set(bad_nodes())))
    w = dict(zip(WORKER[::2], WORKER[1::2]))
    try:
        slots = int(w['--slots'])
        if pe_run.bench_shape(SHAPE_KEY) is None:
            raise KeyError(f'{SHAPE_KEY} not in this snapshot')
        cpus, mem_gb = pe_run.worker_shape(SHAPE_KEY, slots, None, None)
        cfg['worker'].update(slots=slots, cpus=cpus, mem_gb=mem_gb, shape_check={
            'shape_key': SHAPE_KEY, 'measured': {'cpus': cpus, 'mem_gb': mem_gb}, 'warnings': []})
    except Exception as e:   # noqa: BLE001 -- a frozen copy without worker_shape, the trunk's pipeline.json or the key
        print(f'keeping the stored worker shape ({e!r})', flush=True)
    write_json(run / 'config.json', cfg)
    print(pe_run.submit_workers(run, cfg, a.n))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    for n in ('run', 'submit'):
        p = sp.add_parser(n)
        p.add_argument('--name', required=True)
        p.add_argument('--rounds', nargs='+', required=True)
        p.add_argument('--target-tokens', type=int, default=0, help='saved tokens (0: trace every accepted task)')
        p.add_argument('--margin', type=float, default=0.03)
        p.add_argument('--tokens-prior', type=int, default=60000)
        p.add_argument('--chunk', type=int, default=200)
        p.add_argument('--min-chunk', type=int, default=40)
        p.add_argument('--scan-sec', type=int, default=180)
        p.add_argument('--tick-sec', type=int, default=60)
        p.add_argument('--cut-tail', action='store_true')
        p.add_argument('--stop-total', type=float, default=0, help='tail rule: exported tokens over all drivers')
        p.add_argument('--tail-inflight', type=int, default=0, help='tail rule: stop below this many in flight')
        if n == 'submit':
            p.add_argument('--bid', type=int, default=36)
    r = sp.add_parser('resubmit-workers')
    r.add_argument('--run', required=True)
    r.add_argument('--n', type=int, default=1)
    s = sp.add_parser('status')
    s.add_argument('--name', required=True)
    a = ap.parse_args(argv)
    if a.op == 'run':
        drive(a)
    elif a.op == 'submit':
        submit(a)
    elif a.op == 'resubmit-workers':
        resubmit_workers(a)
    else:
        status(a)


if __name__ == '__main__':
    main()

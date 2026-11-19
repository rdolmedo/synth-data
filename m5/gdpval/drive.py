"""The gpu123 driver (light): launch a pi_eval Super run on every READY batch of the producers, keep the sum of live
sessions per Super engine within this worker's cap in alloc.json (re-read every 60 s, live via each run's
control.json), and stop launching once the token target is in sight.

  python -m m5.gdpval.drive --roots PROD_ROOT [...] --phase prod [--target-tokens 330e6] [--once]

State: /data/aai/m5/gdpval/drive-<phase>.json (batch -> run id, engine). Runs: /data/aai/m5/gdpval/runs/gdpval_m5/<phase>-<root>-<bNNN>.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from m5.gdpval import traces as TR

ALLOC = Path('/opt/aai/m5-coord/alloc.json')
SERVING = Path('/data/aai/tbench/tasksynth/serving')
RUNS = Path(TR.RUNS_ROOT) / TR.BENCH
PY = '/home/user/python-3.11/bin/python3'


def alloc():
    a = json.loads(ALLOC.read_text())
    engines = [e for e in a['super']['serving']]
    return engines, int(a['super']['max_inflight_per_engine']['gdpval'])


def healthy(engine: str) -> bool:
    try:
        d = json.loads((SERVING / engine / 'endpoints.json').read_text())
        return bool(d.get('healthy_at')) and time.time() - d['updated_at'] < 120
    except (OSError, ValueError, KeyError):
        return False


def run_state(run: Path) -> dict:
    """items, done (physical non-infra done records counted once per task), claims in flight."""
    q = run / 'queue'
    try:
        items = [x for x in (q / 'items.txt').read_text().split('\n') if x]
    except OSError:
        return {'items': 0, 'resolved': 0, 'active': 0}
    done = {p.name.split('__')[0] for p in (q / 'done').glob('*.json')} if (q / 'done').exists() else set()
    claims = {p.name for p in (q / 'claims').glob('*')} if (q / 'claims').exists() else set()
    donek = {p.stem for p in (q / 'done').glob('*.json')} if (q / 'done').exists() else set()
    return {'items': len(items), 'resolved': len(done), 'active': len(claims - donek)}


def set_control(run: Path, n: int):
    from aa_evals.common.pi_eval import control   # read-merge-write (#451): other writers' keys and hand edits stay
    if control.read(run).get('max_in_flight') != n:
        control.update(run, {'max_in_flight': n, 'burst': 8})


def exported_tokens(export_root: Path) -> int:
    m = export_root / 'manifest.json'
    try:
        return int(json.loads(m.read_text()).get('usable_tokens') or 0)
    except (OSError, ValueError):
        return 0


def fix_holds(state):
    """Pi workers held (e.g. over the memory limit: the cgroup counts sandbox tmpfs and page cache) are removed and
    replaced with cpus x 16 GB; the held worker's claims go back to the queue through pi_eval's stale-claim reaper."""
    out = subprocess.run(['condor_q', '-hold', '-af', 'ClusterId', 'JobBatchName', 'HoldReason'], capture_output=True,
                         text=True).stdout
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 2 or not parts[1].startswith(f'aa-pi-{TR.BENCH}-'):
            continue
        cluster, run_id = parts[0], parts[1][len(f'aa-pi-{TR.BENCH}-'):]
        cfgp = RUNS / run_id / 'config.json'
        try:
            cfg = json.loads(cfgp.read_text())
        except (OSError, ValueError):
            continue
        w = cfg['worker']
        w['mem_gb'] = max(w['mem_gb'], w['cpus'] * 16)
        tmp = cfgp.with_suffix('.partial')
        tmp.write_text(json.dumps(cfg, indent=1, default=str))
        os.replace(tmp, cfgp)
        subprocess.run(['condor_rm', cluster], capture_output=True)
        r = subprocess.run([PY, '-m', 'm5.gdpval.traces', 'resubmit', '--run-id', run_id], capture_output=True,
                           text=True, cwd=str(Path(__file__).resolve().parents[2]))
        print(time.strftime('%H:%M:%S'), 'held worker', cluster, run_id, (parts[2] if len(parts) > 2 else '')[:120],
              '-> resubmitted', r.returncode, r.stdout[-200:], r.stderr[-200:], flush=True)
        state.setdefault('hold_fixes', []).append({'cluster': cluster, 'run': run_id, 'at': time.time()})


def step(a, state):
    fix_holds(state)
    engines, cap = alloc()
    engines = [e for e in engines if healthy(e)]
    runs = state.setdefault('runs', {})
    # live runs per engine and their remaining work
    load = {e: [] for e in engines}
    for b, r in runs.items():
        st = run_state(RUNS / r['run_id'])
        r['state'] = st
        if st['resolved'] < st['items'] and r['engine'] in load:
            load[r['engine']].append(r)
    # allot each engine's cap to its runs, oldest first, by remaining items
    for e, rs in load.items():
        left = cap
        for r in sorted(rs, key=lambda r: r['launched']):
            rem = r['state']['items'] - r['state']['resolved']
            if 'slots' not in r:   # a run can hold at most its workers' slots in flight
                try:
                    w = json.loads((RUNS / r['run_id'] / 'config.json').read_text())['worker']
                    r['slots'] = int(w['slots']) * int(w['n'])
                except (OSError, ValueError, KeyError):
                    r['slots'] = cap
            n = max(0, min(left, rem, r['slots']))
            set_control(RUNS / r['run_id'], n)
            left -= n
        state.setdefault('spare', {})[e] = left
    tokens = exported_tokens(Path(a.export)) if a.export else 0
    planned = sum(r.get('n', 0) for r in runs.values() if r['run_id'].startswith(a.phase + '-'))
    state['tokens_exported'] = tokens
    state['planned_traces'] = planned
    state['projected_tokens'] = planned * a.tokens_per_trace
    stop = bool(a.target_tokens) and (tokens >= a.target_tokens or planned * a.tokens_per_trace >= a.target_tokens
                                      or (a.stop_after_tasks and planned >= a.stop_after_tasks))
    state['stopped_launching'] = stop
    # periodic incremental export (CPU job): pack resolved attempts, drop their archives, refresh the manifest
    if a.export and time.time() - state.get('last_export', 0) > a.export_every * 60:
        job = state.get('export_job')
        alive = job and subprocess.run(['condor_q', str(job), '-af', 'JobStatus'], capture_output=True,
                                       text=True).stdout.strip()
        if not alive:
            rr = ' '.join(str(RUNS / r['run_id']) for r in runs.values() if r['run_id'].startswith(a.phase + '-'))
            if rr:
                cmd = (f'python3 -m m5.gdpval.export pack --runs {rr} --out {a.export} --delete-archives && '
                       f'python3 -m m5.gdpval.export manifest --out {a.export}')
                r = subprocess.run([PY, '-m', 'm5.gdpval.jobs', 'submit', '--name', f'export-{a.phase}', '--cpus', '2',
                                    '--mem-gb', '8', '--hours', '6', '--', 'bash', '-c', cmd], capture_output=True,
                                   text=True, cwd=str(Path(__file__).resolve().parents[2]))
                out = (r.stdout or '').strip().split()
                state['export_job'] = out[-1].rstrip('.') if r.returncode == 0 and out else None
                state['last_export'] = time.time()
                print(time.strftime('%H:%M:%S'), 'export job', state['export_job'], r.stderr[-300:], flush=True)
    # launch ready batches on the engine with the most spare capacity
    if not stop:
        ready = []
        for root in a.roots:
            for d in sorted((Path(root) / 'batches').glob('b*')):
                key = f'{Path(root).name}/{d.name}'
                if (d / 'READY').exists() and key not in runs:
                    ready.append((key, d))
        for key, d in ready:
            spare = state.get('spare', {})
            e = max(spare, key=spare.get) if spare else None
            if e is None or spare[e] < a.min_spare:
                break
            run_id = f'{a.phase}-{key.replace("/", "-")}'
            n = json.loads((d / 'bundle.json').read_text())['n']
            m = min(spare[e], n)
            cmd = [PY, '-m', 'm5.gdpval.traces', 'launch', '--bundle', str(d / 'bundle.tar'), '--run-id', run_id,
                   '--model', f'serving:{e}', '--max-in-flight', str(m), '--slots', str(max(cap, 1)),
                   *(['--worker-cpus', str(a.worker_cpus)] if a.worker_cpus else []),
                   *(['--worker-mem-gb', str(a.worker_mem_gb)] if a.worker_mem_gb else []),
                   '--worker-disk-gb', str(a.worker_disk_gb)]
            print(time.strftime('%H:%M:%S'), 'launch', run_id, 'on', e, 'n', n, 'in flight', m, flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
            print(r.stdout[-800:], r.stderr[-800:], flush=True)
            if r.returncode == 0:
                try:   # the run must queue exactly this bundle's tasks (pi_eval reuses bundles by name)
                    cfg = json.loads((RUNS / run_id / 'config.json').read_text())
                    if cfg['tasks'] != n or Path(cfg['bundle']) != d / 'bundle.tar':
                        print('BUNDLE MISMATCH', run_id, cfg['tasks'], n, cfg['bundle'], flush=True)
                except (OSError, ValueError, KeyError) as ex:
                    print('bundle check failed', run_id, ex, flush=True)
                runs[key] = {'run_id': run_id, 'engine': e, 'n': n, 'launched': time.time()}
                spare[e] -= m
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--phase', required=True)
    ap.add_argument('--export', default=None, help='export root whose manifest.json counts usable tokens')
    ap.add_argument('--target-tokens', type=float, default=0)
    ap.add_argument('--stop-after-tasks', type=int, default=0)
    ap.add_argument('--tokens-per-trace', type=float, default=62000, help='pilot: usable tokens per launched trace')
    ap.add_argument('--export-every', type=float, default=90, help='minutes between export jobs')
    ap.add_argument('--min-spare', type=int, default=4)
    # pilot: ~0.11 cores, ~0.3 GB per session; default: pi_eval's measured shape (aa_evals/pipeline.json, #451)
    ap.add_argument('--worker-cpus', type=int, default=None)
    ap.add_argument('--worker-mem-gb', type=int, default=None)
    ap.add_argument('--worker-disk-gb', type=int, default=150)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    sp = Path('/data/aai/m5/gdpval') / f'drive-{a.phase}.json'   # small; atomically replaced
    state = json.loads(sp.read_text()) if sp.exists() else {}
    while True:
        try:
            state = step(a, state)
        except Exception as e:  # noqa: BLE001
            print('drive error', type(e).__name__, str(e)[:300], flush=True)
        tmp = sp.with_suffix('.partial')
        tmp.write_text(json.dumps(state, indent=1, default=str))
        os.replace(tmp, sp)
        if a.once:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()

"""Stream production: bundle newly generated problems in batches, run them on every Super engine at the alloc cap, and
stop at the token target.

  python -m m5.critpt.stream --gen /data/aai/m5/critpt/gen/prod1 --target-tokens 330e6 [--once] [--dry-run]

Every 60 s:
- caps: alloc.json super.max_inflight_per_engine.critpt (C) over the engines in super.serving that have a run.
  Per engine the runs (oldest batch first) share C: run r gets min(budget, active_r + pending_r) through its live
  control.json, so a new batch only takes the slots the draining one frees, and a lowered cap blocks new starts.
- batches: when the engines' total pending (queued, unclaimed) falls below one cap's worth and >= --min-batch new
  problems exist (or generation ended), the complete lines of GEN/problems.jsonl not yet in any bundle become bundle
  prod-bN (write-once) and are launched as m5c-prod-bN-e<k>, hash-sharded over the serving engines.
- tokens: kept = sum of max_context_tokens over submitted traces (pilot: 0.97 of prepare's token count), plus
  --base-tokens (the pilot export). When kept + in-flight projection >= target: generation gets {"abort": true},
  no new batches start, and every run gets STOP_WORKERS (running attempts finish; the tail is cut later by hand).
State: /data/aai/m5/critpt/stream/state.json (verified writes).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import safeio
from aa_evals.common.pi_eval import control

ALLOC = Path('/opt/aai/m5-coord/alloc.json')
RUNS = Path('/data/aai/m5/critpt/runs/m5_critpt')
BUNDLES = Path('/data/aai/m5/critpt/bundles')
STATE = Path('/data/aai/m5/critpt/stream/state.json')
SERVING = Path('/data/aai/tbench/tasksynth/serving')
# trace runs are launched from the worktree, not from this process's code copy: pi_eval snapshots the code with git
# (rev-parse HEAD), and a condor job runs a `git archive` copy without .git
REPO = Path(os.environ.get('M5_CRITPT_LAUNCH_REPO', '/opt/aai/worktrees/m5-critpt'))
BUNDLE_PY = '/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python'
ENGINES = [f'm5-super-{i}' for i in range(1, 65)]   # any m5-super-N that alloc lists


def log(*a):
    print(time.strftime('%H:%M:%SZ', time.gmtime()), *a, flush=True)


def names(d: Path, suf=''):
    return {p.name.removesuffix(suf) for p in d.glob('*')} if d.exists() else set()


def worker_counts() -> dict[str, dict[int, int]]:
    """{run id: {condor JobStatus: count}} for this bench's Pi workers (one condor_q per tick)."""
    out = subprocess.run(['condor_q', '-constraint', 'regexp("^aa-pi-m5_critpt-", JobBatchName)', '-af',
                          'JobBatchName', 'JobStatus'], capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise OSError(f'condor_q failed: {out.stderr[-200:]}')
    c: dict[str, dict[int, int]] = {}
    for line in out.stdout.split('\n'):
        if line.strip():
            name, status = line.split()
            d = c.setdefault(name.removeprefix('aa-pi-m5_critpt-'), {})
            d[int(status)] = d.get(int(status), 0) + 1
    return c


def run_state(run: Path, workers: dict | None = None) -> dict:
    q = run / 'queue'
    # items.txt: "task attempt"; claims/done are named task__a<attempt>__r0
    items = [f'{t}__a{k}__r0' for t, k in (x.split() for x in (q / 'items.txt').read_text().split('\n') if x.strip())]
    done, claims = names(q / 'done', '.json'), names(q / 'claims')
    cfg = json.loads((run / 'config.json').read_text())
    w = cfg.get('worker') or {}
    wc = (workers or {}).get(run.name, {})
    n_live = wc.get(2, 0) + wc.get(1, 0)   # running + idle (queued) workers
    return {'run': run.name, 'engine': cfg['model'].split(':', 1)[1], 'slots_per_worker': int(w.get('slots', 24)),
            'slots': int(w.get('slots', 24)) * max(1, n_live), 'workers_idle': wc.get(1, 0),
            'workers_held': wc.get(5, 0),
            'items': len(items), 'done': len(done),
            'active': len(claims - done), 'pending': len(set(items) - claims), 'stopped': (run / 'STOP_WORKERS').exists()}


_SEEN: dict[str, tuple[int, int]] = {}   # outcome path -> (tokens if submitted else 0, submitted)


def kept_tokens(runs: list[Path]) -> tuple[int, int, int]:
    """-> (sum of max_context_tokens over submitted traces, n submitted, n finished). Outcomes are read once."""
    for run in runs:
        for o in (run / 'attempts').glob('*/outcome.json'):
            k = str(o)
            if k in _SEEN:
                continue
            try:
                r = json.loads(o.read_text())
            except (OSError, ValueError):
                continue
            ok = not r.get('infrastructure_failure') and (r.get('score') or 0) >= 1
            _SEEN[k] = (((r.get('stats') or {}).get('max_context_tokens') or 0) if ok else 0, int(ok))
    return sum(t for t, _ in _SEEN.values()), sum(s for _, s in _SEEN.values()), len(_SEEN)


def bundled_ids() -> set:
    ids = set()
    for m in BUNDLES.glob('*.json'):
        ids |= set(json.loads(m.read_text())['tasks'])
    return ids


def serving_engines(alloc) -> list[str]:
    out = []
    for e in alloc['super']['serving']:
        if e not in ENGINES:
            continue
        try:
            info = json.loads((SERVING / e / 'endpoints.json').read_text())
            if time.time() - info['updated_at'] < 120 and info.get('healthy_at'):
                out.append(e)
        except (OSError, ValueError, KeyError):
            pass
    return out


def water_fill(n: int, backlog: dict[str, int]) -> dict[str, int]:
    """Split n new tasks over engines so that backlog + share is as even as possible (fill the shortest first)."""
    share = {e: 0 for e in backlog}
    for _ in range(n):
        e = min(backlog, key=lambda x: (backlog[x] + share[x], x))
        share[e] += 1
    return share


def launch_split(bundle: Path, rid_prefix: str, todo: list[str], engines: list[str], backlog: dict[str, int],
                 cap: int, dry: bool):
    """Launch `todo` (tasks of `bundle`) as one run per engine, sized by water_fill over the engines' backlogs;
    `backlog` is updated in place so later launches in the same tick see these. Runs start at max_in_flight 0 (the
    next tick gives them their share of the engine's cap after the older runs)."""
    share = water_fill(len(todo), {e: backlog.get(e, 0) for e in engines})
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    k = 0
    for e in engines:
        part, k = todo[k:k + share[e]], k + share[e]
        if not part:
            continue
        rid = f'{rid_prefix}e{e.rsplit("-", 1)[1]}'
        log(f'{rid}: {len(part)} tasks on {e} (backlog {backlog.get(e, 0)})')
        if dry:
            continue
        tf = STATE.parent / 'tasks' / f'{rid}.txt'
        tf.parent.mkdir(parents=True, exist_ok=True)
        safeio.write_verified(tf, ('\n'.join(part) + '\n').encode())
        r = subprocess.run([BUNDLE_PY, '-m', 'm5.critpt.traces', 'launch', '--run-id', rid, '--bundle', str(bundle),
                            '--engine', e, '--tasks-file', str(tf), '--max-in-flight', str(cap), '--workers', '1',
                            '--slots', '24'],   # CPUs/RAM: pi_eval's measured shape (#451)
                           env=env, cwd=str(REPO), capture_output=True, text=True)
        if r.returncode == 0:
            safeio.write_json(RUNS / rid / 'control.json', {'max_in_flight': 0, 'burst': 8, 'why': 'stream: new'})
            backlog[e] = backlog.get(e, 0) + len(part)
        log(rid, 'launched' if r.returncode == 0 else f'LAUNCH FAILED {r.returncode}', r.stdout[-300:], r.stderr[-300:])


def new_batch(gen: Path, n_batch: int, engines: list[str], backlog: dict[str, int], cap: int, dry: bool):
    data = (gen / 'problems.jsonl').read_bytes()
    data = data[:data.rfind(b'\n') + 1]
    have = bundled_ids()
    rows = [json.loads(x) for x in data.splitlines()]
    new = [r for r in rows if r['problem_id'] not in have]
    if not new:
        return None
    name = f'prod-b{n_batch}'
    snap = gen.parent / f'{gen.name}-{name}.jsonl'
    log(f'batch {name}: {len(new)} new problems -> {len(engines)} engines')
    if not dry:
        safeio.write_verified(snap, b''.join(json.dumps(r).encode() + b'\n' for r in new),
                              check=lambda p: [json.loads(x) for x in Path(p).read_text().splitlines()])
        env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
        subprocess.run([BUNDLE_PY, '-m', 'm5.critpt.traces', 'bundle', '--problems', str(snap), '--out',
                        str(BUNDLES / f'{name}.tar')], env=env, cwd=str(REPO), check=True)
    launch_split(BUNDLES / f'{name}.tar', f'm5c-{name}-', [r['problem_id'] for r in new], engines, backlog, cap, dry)
    return name


def pending_tasks(run: Path) -> list[str]:
    q = run / 'queue'
    claims = names(q / 'claims')
    return sorted({t for t, k in (x.split() for x in (q / 'items.txt').read_text().split('\n') if x.strip())
                   if f'{t}__a{k}__r0' not in claims})


def move_orphans(r: dict, engines: list[str], backlog: dict[str, int], cap: int, dry: bool, why: str = ''):
    """Stop run r's workers claiming (running attempts finish) and relaunch its unstarted tasks from the same bundle,
    water-filled over the serving engines. Used when r's engine left alloc's serving list, and to rebalance (an idle
    run on an overloaded engine, e.g. after a new engine joins)."""
    run = RUNS / r['run']
    if not dry:
        (run / 'STOP_WORKERS').write_text('stop\n')
    time.sleep(2)
    todo = pending_tasks(run)
    if not todo or not engines:
        return
    cfg = json.loads((run / 'config.json').read_text())
    n = 1
    while any(RUNS.glob(f'{r["run"]}-m{n}*')):
        n += 1
    if r['engine'] in backlog:
        backlog[r['engine']] -= len(todo)
    log(f'{r["run"]}: {why or "engine " + r["engine"] + " not serving"}; moving {len(todo)} unstarted tasks')
    launch_split(Path(cfg['bundle']), f'{r["run"]}-m{n}', todo, engines, backlog, cap, dry)


def batch_no(run_name: str) -> int:
    return int(run_name.split('-b', 1)[1].split('-', 1)[0])


def tick(a, st: dict):
    alloc = json.loads(ALLOC.read_text())
    cap = int(alloc['super']['max_inflight_per_engine']['critpt'])
    engines = serving_engines(alloc)
    runs = sorted((p for p in RUNS.glob('m5c-prod-b*-e*') if (p / 'queue' / 'items.txt').exists()),
                  key=lambda p: (batch_no(p.name), p.name))
    wcounts = worker_counts()
    rs = [run_state(p, wcounts) for p in runs]
    # caps per engine, oldest batch first
    for e in {r['engine'] for r in rs}:
        budget = cap if e in alloc['super']['serving'] else 0
        for r in (x for x in rs if x['engine'] == e):
            # a stopped run claims nothing new, but its traces still running count against the engine's cap
            want = r['active'] if r['stopped'] else r['active'] + r['pending']
            give = min(budget, want, r['slots'])   # a run can't use more than its workers' slots; the rest goes on
            budget -= give
            new = {'max_in_flight': give, 'burst': 8, 'why': f'stream: alloc critpt {cap}/engine'}
            if control.read(RUNS / r['run']).get('max_in_flight') != give and not a.dry_run:
                control.update(RUNS / r['run'], new)   # read-merge-write (#451): other writers' keys stay
            r['cap'] = give
        # held workers (e.g. over the memory limit): their claims go stale and are retried by the other workers;
        # remove them so a replacement can be added
        for x in (y for y in rs if y['engine'] == e and y['workers_held']):
            log(f'{x["run"]}: {x["workers_held"]} held worker(s); condor_rm')
            if not a.dry_run:
                subprocess.run(['condor_rm', '-constraint', f'JobBatchName == "aa-pi-m5_critpt-{x["run"]}" && '
                                'JobStatus == 5'], capture_output=True, text=True)
        # budget left over because the runs' workers have too few slots: add a worker to the run with the most queued
        # work (one per engine per tick, none while that run already has a worker waiting to start)
        live = [x for x in rs if x['engine'] == e and not x['stopped'] and x['pending'] > x['slots_per_worker']]
        if budget > 0 and e in alloc['super']['serving'] and live and not st.get('target_reached'):
            r = max(live, key=lambda x: x['pending'])
            if not r['workers_idle']:
                log(f'{e}: {budget} of {cap} unused (worker slots); adding a worker to {r["run"]}')
                if not a.dry_run:
                    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
                    p = subprocess.run([BUNDLE_PY, '-m', 'm5.critpt.traces', 'resubmit', '--run-id', r['run'],
                                        '--n', '1'], env=env, cwd=str(REPO), capture_output=True, text=True)
                    log(r['run'], 'worker added' if p.returncode == 0 else f'RESUBMIT FAILED {p.returncode}',
                        p.stderr[-300:])
    backlog = {e: sum(x['active'] + x['pending'] for x in rs if x['engine'] == e and not x['stopped'])
               for e in engines}
    if not st.get('target_reached'):
        for r in rs:
            if r['engine'] not in alloc['super']['serving'] and not r['stopped'] and r['pending']:
                move_orphans(r, engines, backlog, cap, a.dry_run)
        # rebalance: only when an engine is close to running dry (backlog < 2 caps, e.g. a new engine) and another
        # holds > 2 caps more: move one idle run (nothing in flight) off the heaviest engine
        if engines:
            lo = min(backlog.values())
            # a run moves when an engine is close to running dry (< 2 caps, e.g. a new engine) and the run's engine
            # holds > 2 caps more, or for any gap > 4 caps; idle runs first. A run with traces in flight stops
            # claiming, its running traces finish, and its unstarted tasks are water-filled over all engines (its own
            # engine included, so that engine keeps its share). One move per tick.
            gap = lambda r: backlog[r['engine']] - lo
            cand = [r for r in rs if not r['stopped'] and r['pending'] >= cap and r['engine'] in backlog
                    and ((lo < 2 * cap and gap(r) > 2 * cap) or gap(r) > 4 * cap)]
            if cand:
                r = max(cand, key=lambda x: (x['active'] == 0, backlog[x['engine']], x['pending']))
                move_orphans(r, engines, backlog, cap, a.dry_run,
                             why=f'rebalance ({r["engine"]} backlog {backlog[r["engine"]]} vs min {lo})')
    tok, sub, fin = kept_tokens(runs)
    active = sum(r['active'] for r in rs)
    rate = tok / sub if sub else 37000
    share = sub / fin if fin else 0.87
    projected = a.base_tokens + tok + active * share * rate
    gen_alive = subprocess.run(['condor_q', '-constraint', f'JobBatchName == "{a.gen_batch}"', '-af', 'JobStatus'],
                               capture_output=True, text=True).stdout.strip()
    st.update({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cap': cap, 'engines': engines,
               'kept_tokens': a.base_tokens + tok, 'submitted': sub, 'finished': fin, 'active': active,
               'projected_tokens': int(projected), 'target': a.target_tokens, 'gen_alive': bool(gen_alive),
               'pending_live': sum(r['pending'] for r in rs if not r['stopped']),
               'runs': {r['run']: {k: r[k] for k in ('items', 'done', 'active', 'pending', 'cap')} for r in rs}})
    kept = a.base_tokens + tok
    hit = kept >= a.stop_kept_tokens if a.stop_kept_tokens else projected >= a.target_tokens
    if hit and not st.get('target_reached'):
        st['target_reached'] = st['at']
        what = (f'kept {kept / 1e6:.1f}M >= {a.stop_kept_tokens / 1e6:.0f}M' if a.stop_kept_tokens else
                f'projected {projected / 1e6:.1f}M >= {a.target_tokens / 1e6:.0f}M')
        log(f'TARGET: {what}: abort gen, stop all runs' + (', kill in-flight traces' if a.kill_tail else ''))
        if not a.dry_run:
            safeio.write_json(Path(a.gen) / 'control.json', {'n': 0, 'abort': True})
            for r in rs:
                (RUNS / r['run'] / 'STOP_WORKERS').write_text('stop\n')
            if a.kill_tail:   # tail rule: don't wait for stragglers (unfinished attempts have no outcome; export skips them)
                k = subprocess.run(['condor_rm', '-constraint', 'regexp("^aa-pi-m5_critpt-", JobBatchName)'],
                                   capture_output=True, text=True)
                log(f'TARGET: condor_rm trace workers: {k.stdout.strip()[-200:]} {k.stderr.strip()[-200:]}')
    pending = sum(r['pending'] for r in rs if not r['stopped'])   # a stopped run's leftovers were moved
    # a new batch when the queues run low, or at once for the last problems once generation has ended
    if not st.get('target_reached') and engines and (pending < cap * len(engines) or not gen_alive):
        nb = max([batch_no(r['run']) for r in rs] + [0]) + 1
        data = (Path(a.gen) / 'problems.jsonl').read_bytes()
        ids = {json.loads(x)['problem_id'] for x in data[:data.rfind(b'\n') + 1].splitlines()}
        unbundled = len(ids - bundled_ids())
        if unbundled >= a.min_batch or (unbundled and not gen_alive):
            b = new_batch(Path(a.gen), nb, engines, backlog, cap, a.dry_run)
            if b:
                st.setdefault('batches', []).append({'name': b, 'at': st['at'], 'n': unbundled})
    if not a.dry_run:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        safeio.write_json(STATE, st, indent=1)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', required=True)
    ap.add_argument('--target-tokens', type=float, default=330e6)
    ap.add_argument('--stop-kept-tokens', type=float, default=0,
                    help='stop when KEPT tokens (finished, submitted traces + --base-tokens) reach this (overrides '
                         '--target-tokens)')
    ap.add_argument('--kill-tail', action='store_true', help='at the stop, condor_rm all trace workers at once')
    ap.add_argument('--base-tokens', type=float, default=8173699, help='tokens already exported (pilot)')
    ap.add_argument('--min-batch', type=int, default=500)
    ap.add_argument('--gen-batch', default='m5-critpt-gen-prod1b', help='condor JobBatchName of the generation job')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    last = None
    while True:
        try:
            st = tick(a, st)
            line = (f"cap {st['cap']} engines {len(st['engines'])} kept {st['kept_tokens'] / 1e6:.1f}M "
                    f"projected {st['projected_tokens'] / 1e6:.1f}M sub/fin {st['submitted']}/{st['finished']} "
                    f"active {st['active']} pending {st.get('pending_live', 0)}")
            if line != last:
                log(line)
                last = line
        except Exception as e:   # keep streaming through transient /data or condor errors
            log(f'tick failed: {e!r}')
        if a.once:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

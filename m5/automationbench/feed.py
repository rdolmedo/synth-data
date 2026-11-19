"""Stream accepted tasks into Super (#359 production): bundles of accepted tasks become pi_eval runs on the Super
engines of alloc.json; each engine's in-flight sessions stay within the worker's cap; stops at the token target.

  PYTHONPATH=. venvs/harbor/bin/python -m m5.automationbench.feed --accepted 'GLOB' --state DIR --prefix prod \
      [--batch 600] [--target-tokens 330e6] [--slots 32]

Every 60 s:
- caps: alloc.json `super.serving` (engines) and `max_inflight_per_engine.automationbench`;
- launch: when an engine's queued-but-unclaimed attempts fall below its cap and >= --batch accepted tasks (or all
  that are left once generation has ended: DIR/GEN_DONE) are not yet bundled, a bundle (bundle.py, the eval's own
  builder) and a pi_eval run on that engine (traces.py) are made from them;
- control: each run's control.json `max_in_flight` = the engine's cap shared in launch order (older runs first), so
  the engine never holds more than the cap;
- stop: when the resolved attempts' tokens (pi_eval's records: each attempt's largest request context, the full
  sequence as the server counted it) reach --target-tokens, no new runs start and every
  run gets STOP_WORKERS (running attempts finish), which cuts the tails;
- state: DIR/feed.json (runs, bundled tasks), DIR/feed.log.
Bundles, run dirs and shards live on /data (/data/aai/m5/automationbench); run the feeder as a CPU job (jobs.py).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path('/data/aai/m5/automationbench')
AB_PY = '/opt/aai/venvs/automationbench/bin/python'
HARBOR_PY = '/opt/aai/venvs/harbor/bin/python'


def log(state_dir: Path, *msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%S ') + ' '.join(str(m) for m in msg)
    print(line, flush=True)
    with (state_dir / 'feed.log').open('a') as f:
        f.write(line + '\n')


def alloc():
    a = json.loads(Path('/opt/aai/m5-coord/alloc.json').read_text())
    return a['super']['serving'], int(a['super']['max_inflight_per_engine']['automationbench'])


def engine_ok(name: str) -> bool:
    try:
        e = json.loads(Path(f'/data/aai/tbench/tasksynth/serving/{name}/endpoints.json').read_text())
        return bool(e.get('healthy_at')) and time.time() - e.get('updated_at', 0) < 120
    except (OSError, ValueError):
        return False


def accepted_names(pattern: str) -> list[tuple[str, str, int]]:
    """(task name, file, line number) of every accepted task, in file order."""
    out = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        out.append((json.loads(line)['task']['info']['task_name'], p, i))
                    except (ValueError, KeyError):
                        pass
    return out


def resolved_tokens(runs: list[str]) -> tuple[int, int]:
    """(tokens, attempts) over the runs' resolved ok attempts that finished by themselves: each attempt's largest request context (the server's
    count with Super's tokenizer: the full sequence of a linear session). Read from pi_eval's attempt records, so
    the stop rule does not wait for collect.py."""
    from aa_evals.common.pi_eval import port
    n = k = 0
    for r in runs:
        d = run_dir(r)
        if not (d / 'queue').exists():
            continue
        for row in port.resolved(d):
            if row['status'] == 'ok' and port.end_of(row) == 'finish':   # export drops no-submit endings
                n += int(row.get('max_context_tokens') or 0)
                k += 1
    return n, k


def make_bundle(state_dir: Path, run_id: str, items: list[tuple[str, str, int]]) -> Path:
    sel = DATA / 'bundles' / f'{run_id}.jsonl'
    sel.parent.mkdir(parents=True, exist_ok=True)
    want = {}
    for name, p, i in items:
        want.setdefault(p, set()).add(i)
    with sel.open('w') as o:
        for p, lines in want.items():
            with open(p) as f:
                for i, line in enumerate(f):
                    if i in lines:
                        o.write(line)
    tar = DATA / 'bundles' / f'{run_id}.tar'
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    subprocess.run([AB_PY, '-m', 'm5.automationbench.bundle', str(sel), str(tar), f'm5ab-{run_id}'], check=True,
                   env=env, cwd=str(REPO))
    return tar


def launch(run_id: str, tar: Path, engine: str, cap: int, slots: int):
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    workers = max(1, -(-cap // slots))
    # CPUs/RAM: pi_eval's measured shape for automationbench_pi (aa_evals/pipeline.json, #378), not a launcher's (#451)
    subprocess.run([HARBOR_PY, '-m', 'm5.automationbench.traces', 'launch', '--bundle', str(tar), '--run-id', run_id,
                    '--engine', engine, '--slots', str(slots), '--workers', str(workers), '--max-in-flight', str(cap),
                    '--worker-disk-gb', str(max(10, slots * 5 // 8))], check=True, env=env, cwd=str(REPO))


def write_control(run: str, cap: int, why: str, burst: int = 32) -> None:
    """adaptive.write with a burst that can fill the cap: its fixed 4 starts/min held each 64-session run to ~4
    traces/min (production start, 2026-11-12). Read-merge-write (#451): other writers' keys and hand edits stay."""
    from aa_evals.common.pi_eval import control
    control.update(run, {'max_in_flight': int(cap), 'burst': burst, 'why': why})


def run_dir(run_id: str) -> Path:
    return DATA / 'pi_runs' / 'm5_automationbench' / run_id


def main():
    """loop() with transient /data I/O errors logged and retried (an EIO on one bundle write killed a feeder)."""
    while True:
        try:
            return loop()
        except OSError as e:
            print(time.strftime('%Y-%m-%dT%H:%M:%S'), f'I/O error, retrying in 120 s: {e!r}', flush=True)
            time.sleep(120)


def loop():
    from aa_evals.common.pi_eval.adaptive import state as run_state
    ap = argparse.ArgumentParser()
    ap.add_argument('--accepted', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--batch', type=int, default=600)
    ap.add_argument('--target-tokens', type=float, default=330e6)
    ap.add_argument('--shards', default='', help='unused (kept for old command lines)')
    ap.add_argument('--slots', type=int, default=64)
    ap.add_argument('--max-tasks', type=int, default=0, help='bundle at most this many tasks in total (pilot)')
    a = ap.parse_args()
    sd = Path(a.state)
    sd.mkdir(parents=True, exist_ok=True)
    st_p = sd / 'feed.json'
    st = json.loads(st_p.read_text()) if st_p.exists() else {'runs': [], 'bundled': [], 'stopped': False}
    while True:
        engines, cap = alloc()
        live = [e for e in engines if engine_ok(e)]
        # control: share each engine's cap over its runs, oldest first
        for e in engines:
            left = cap if e in live else 0
            for r in [r for r in st['runs'] if r['engine'] == e]:
                d = run_dir(r['id'])
                if not (d / 'queue').exists():
                    continue
                inflight, todo = run_state(str(d))
                give = min(left, inflight + todo)
                write_control(str(d), max(give, 0), f'feed: engine {e} cap {cap}')
                left -= give
            st.setdefault('free', {})[e] = left
        if time.time() - st.get('tokens_t', 0) > 300:
            st['tokens'], st['attempts'] = resolved_tokens([r['id'] for r in st['runs']])
            st['tokens_t'] = time.time()
        tokens = st.get('tokens', 0)
        if not st['stopped'] and tokens >= a.target_tokens:
            st['stopped'] = True
            for r in st['runs']:
                (run_dir(r['id']) / 'STOP_WORKERS').touch()
            log(sd, f'token target reached ({tokens:,} >= {a.target_tokens:,.0f}): all runs stop claiming')
        bundled = set(st['bundled'])
        pending = [x for x in accepted_names(a.accepted) if x[0] not in bundled]
        if a.max_tasks:
            pending = pending[:max(0, a.max_tasks - len(bundled))]
        gen_done = (sd / 'GEN_DONE').exists()
        if not st['stopped']:
            for e in live:
                if st['free'].get(e, 0) <= 0 or not pending:
                    continue
                if len(pending) < a.batch and not gen_done and not a.max_tasks:
                    continue
                take, pending = pending[:a.batch], pending[a.batch:]
                k = len(st['runs'])
                rid = f'{a.prefix}-{k:03d}'
                while run_dir(rid).exists() or (DATA / 'bundles' / f'{rid}.tar').exists():   # never reuse an id
                    k += 1
                    rid = f'{a.prefix}-{k:03d}'
                tar = make_bundle(sd, rid, take)
                launch(rid, tar, e, cap, a.slots)
                queued = sorted({ln.split()[0] for ln in (run_dir(rid) / 'queue' / 'items.txt').read_text().splitlines() if ln.strip()})
                if queued != sorted(x[0] for x in take):
                    log(sd, f'ERROR {rid}: queued {len(queued)} items != the {len(take)} bundled; stopping it')
                    (run_dir(rid) / 'STOP_WORKERS').touch()
                    raise SystemExit(2)
                st['runs'].append({'id': rid, 'engine': e, 'n': len(take), 'at': time.time()})
                st['bundled'] += [x[0] for x in take]
                log(sd, f'launched {rid}: {len(take)} tasks on {e}')
        tmp = st_p.with_suffix('.tmp')
        tmp.write_text(json.dumps(st))
        os.replace(tmp, st_p)
        if (st['stopped'] or (gen_done and not pending)) and all(
                run_state(str(run_dir(r['id'])))[0] == 0 and (st['stopped'] or run_state(str(run_dir(r['id'])))[1] == 0)
                for r in st['runs'] if (run_dir(r['id']) / 'queue').exists()):
            log(sd, 'all runs drained; feeder exits')
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

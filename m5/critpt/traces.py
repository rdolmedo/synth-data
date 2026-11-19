"""Super traces for generated CritPt problems under the eval's own Pi harness (#348), one trace per problem.

  python -m m5.critpt.traces bundle --problems P.jsonl [P2.jsonl ...] --out BUNDLE.tar [--skip-in RUN_DIRS] [--limit N]
  python -m m5.critpt.traces launch --run-id ID --bundle BUNDLE.tar --engine m5-super-1 [--max-in-flight 16] [--shard 0/2]
  python -m m5.critpt.traces status --run-id ID

`bundle` turns each problem into a Harbor task exactly as aa_evals.critpt.pi.tasks builds the 70 test tasks
(`build_task`: the authors' one-step system prompt + the Stirrup user prompt with the answer file, the CritPt SIF
as the image, 4 CPUs / 16 GB, offline), so the trace's prompt, sandbox and limits are the eval's.
`launch` runs aa_evals.common.pi_eval on it with the port's shared arguments (`port.base_args`: Pi 1.2.0
pi-cont16k, 131,072 window without compaction, 16,384 output tokens per request, 2 h environment clock, the
registry's request block, text-only), 1 attempt per problem, seed namespace m5-critpt-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# run dirs (and pi_eval's code snapshots) on /data, never under /home (coordinator rule, 2026-11-12)
os.environ.setdefault('AA_RUNS_ROOT', '/data/aai/m5/critpt/runs')

from aa_evals.common.pi_eval import port  # noqa: E402

EXCLUDED_NODES_FILE = Path('/opt/aai/m5-coord/excluded-nodes.txt')
BENCH = 'm5_critpt'
SEED_NAMESPACE = 'm5-critpt-v1'


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def bundle(a):
    from aa_evals.critpt import data
    from aa_evals.critpt.pi import tasks as PT
    probs = []
    seen = set()
    for p in a.problems:
        for r in read_jsonl(p):
            if r['problem_id'] in seen:
                continue
            seen.add(r['problem_id'])
            probs.append(r)
    skip = set()
    for d in (a.skip_in or []):
        q = Path(d) / 'queue' / 'items.txt'
        if q.exists():
            skip |= {x.split()[0] for x in q.read_text().split('\n') if x.strip()}
    probs = [r for r in probs if r['problem_id'] not in skip]
    if a.limit:
        probs = probs[:a.limit]
    out = Path(a.out)
    if out.exists() or out.with_suffix('.json').exists():   # bundles are write-once: a run never sees a changed one
        raise SystemExit(f'{out} exists: bundles are immutable, write a new path')
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or None) as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        for r in probs:
            p = data.Problem(r['problem_id'], r['description'], r['code_template'])
            meta[r['problem_id']] = PT.build_task(p, root)
        part = out.with_suffix('.tar.partial')
        with tarfile.open(part, 'w') as tf:
            for f in sorted(Path(tmp).rglob('*')):
                tf.add(f, arcname=f.relative_to(tmp).as_posix(), recursive=False, filter=PT._norm)
        with tarfile.open(part) as tf:   # read back before publishing (enforced /data quota: truncated writes)
            n_toml = sum(1 for m in tf.getmembers() if m.name.endswith('/task.toml'))
        if n_toml != len(meta):
            raise SystemExit(f'{part}: {n_toml} task.toml read back, {len(meta)} written')
        part.replace(out)
    from aa_evals.common.pi_eval.task_inputs import sha256
    manifest = {'bundle_version': f'{SEED_NAMESPACE}-bundle', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'base_ref': PT.BASE_REF, 'sif_sha256': PT.SIF_SHA256, 'answer_file': PT.ANSWER,
                'system_prompt_sha': PT.prompts.sha(PT.prompts.system_prompt()), 'tasks': meta,
                'problems_from': [str(p) for p in a.problems], 'sha256': sha256(out)}
    from . import safeio
    safeio.write_json(out.with_suffix('.json'), manifest, indent=1)
    print(f'{out}: {len(meta)} tasks')


def excluded_nodes() -> list[str]:
    """The coordinator's shared exclusions (read at submit time), on top of pi_eval's EXCLUDE_NODES."""
    try:
        out = []
        for l in EXCLUDED_NODES_FILE.read_text().splitlines():
            if l.startswith('# BAD_NODES') and ':' in l:   # the aa-serving BAD_NODES list, quoted in the header
                out += l.split(':', 1)[1].split()
            elif l.split('#')[0].strip():
                out.append(l.split('#')[0].strip())
        return list(dict.fromkeys(out))
    except OSError:
        return []


def launch(a):
    bundle_path = Path(a.bundle)
    tasks = sorted(json.loads(bundle_path.with_suffix('.json').read_text())['tasks'])
    if a.tasks_file:   # an explicit subset (the streamer moves a dropped engine's unstarted tasks to another engine)
        want = set(Path(a.tasks_file).read_text().split())
        if not want <= set(tasks):
            raise SystemExit(f'{len(want - set(tasks))} tasks of --tasks-file are not in the bundle')
        tasks = sorted(want)
    if a.shard:   # i/k: this run takes the tasks whose hash falls in shard i (one run per Super engine)
        i, k = map(int, a.shard.split('/'))
        tasks = [t for t in tasks if int(hashlib.sha256(t.encode()).hexdigest(), 16) % k == i]
        print(f'shard {a.shard}: {len(tasks)} tasks', flush=True)
    run_dir = port.runs_root() / BENCH / a.run_id
    if run_dir.exists():   # fresh run ids only (a reused id can silently run stale state; m5-tbsci_lite lost tasks)
        raise SystemExit(f'{run_dir} exists: use a fresh --run-id')
    args = port.base_args(BENCH, a.run_id, f'serving:{a.engine}', bundle_path, tasks, 1, SEED_NAMESPACE)
    args += ['--max-in-flight', str(a.max_in_flight), '--burst', str(a.burst), '--slots', str(a.slots),
             *(['--worker-cpus', str(a.worker_cpus)] if a.worker_cpus else []),
             *(['--worker-mem-gb', str(a.worker_mem_gb)] if a.worker_mem_gb else []),
             '--worker-disk-gb', str(a.worker_disk_gb), '--cpu-bid', str(a.cpu_bid),
             # 2 h wall clock = a hang backstop here (Super-speed generation; the 131,072-token context
             # and its finish notice end attempts). The eval's model-time caps: 2 h without the capped
             # protocol (part of that protocol, #344); under it 12 h by default, a hang backstop
             # (CAPPED_WALL_CAP_SEC in aa_evals/campaign/launch.py, #531 B6)
             '--agent-wall-cap-sec', '7200']
    if a.workers:
        args += ['--workers', str(a.workers)]
    # pi_eval.run with the shared exclusions added to its module-level EXCLUDE_NODES (read when it writes the
    # worker submit file), in the harbor venv as port.launch runs it
    import subprocess
    extra = excluded_nodes()
    code = ('import sys, aa_evals.common.pi_eval.run as R; '
            f'R.EXCLUDE_NODES = tuple(dict.fromkeys(R.EXCLUDE_NODES + tuple({extra!r}))); R.main(sys.argv[1:])')
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    print('excluding', extra, flush=True)
    r = subprocess.run([port.HARBOR_PYTHON, '-c', code, *args], env=env, cwd=str(REPO))
    if r.returncode:
        raise SystemExit(f'pi_eval.run failed ({r.returncode})')
    q = run_dir / 'queue' / 'items.txt'
    queued = {x.split()[0].split('__')[0] for x in q.read_text().split('\n') if x.strip()} if q.exists() else set()
    if queued != set(tasks):
        raise SystemExit(f'queued {len(queued)} tasks != selected {len(tasks)} '
                         f'(missing {sorted(set(tasks) - queued)[:5]}, extra {sorted(queued - set(tasks))[:5]})')
    print(f'{a.run_id}: queued {len(queued)} tasks = selection', flush=True)


def resubmit(a):
    """Submit N more workers to an existing run, with the run's own stored arguments (pi_eval checks they match)."""
    import subprocess
    run_dir = port.runs_root() / BENCH / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    w = cfg['worker']
    args = port.base_args(BENCH, a.run_id, cfg['model'], Path(cfg['bundle']), sorted(cfg['task_names']), 1,
                          SEED_NAMESPACE)
    args += ['--max-in-flight', str(cfg['max_in_flight']), '--burst', str(cfg['burst']), '--slots', str(w['slots']),
             '--worker-cpus', str(w['cpus']), '--worker-mem-gb', str(w['mem_gb']),
             '--worker-disk-gb', str(w['disk_gb']), '--cpu-bid', str(w['bid']), '--resubmit', str(a.n)]
    extra = excluded_nodes()
    code = ('import sys, aa_evals.common.pi_eval.run as R; '
            f'R.EXCLUDE_NODES = tuple(dict.fromkeys(R.EXCLUDE_NODES + tuple({extra!r}))); R.main(sys.argv[1:])')
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    r = subprocess.run([port.HARBOR_PYTHON, '-c', code, *args], env=env, cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if r.returncode:
        raise SystemExit(f'pi_eval.run --resubmit failed ({r.returncode})')


def status(a):
    print(json.dumps(port.status(port.runs_root() / BENCH / a.run_id), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('bundle')
    b.add_argument('--problems', nargs='+', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--skip-in', nargs='*')
    b.add_argument('--limit', type=int, default=0)
    l = sp.add_parser('launch')
    l.add_argument('--run-id', required=True)
    l.add_argument('--bundle', required=True)
    l.add_argument('--engine', default='m5-super-1')
    l.add_argument('--shard', default='', help='i/k: only the tasks in hash shard i of k (one run per engine)')
    l.add_argument('--tasks-file', default='', help='only these task ids (whitespace-separated)')
    l.add_argument('--max-in-flight', type=int, default=16)
    l.add_argument('--burst', type=int, default=8)
    l.add_argument('--slots', type=int, default=16)
    l.add_argument('--workers', type=int, default=0)
    # pilot: 0.3-1.8 CPU per 16-slot worker, peak 7.3 GB; default: pi_eval's measured shape (pipeline.json, #451)
    l.add_argument('--worker-cpus', type=int, default=None)
    l.add_argument('--worker-mem-gb', type=int, default=None)
    l.add_argument('--worker-disk-gb', type=int, default=30)
    l.add_argument('--cpu-bid', type=int, default=36)
    # sizing (coordinator CPU rule, 2026-11-12): the 1-slot plumbing worker averaged 0.29 CPU and ~0.3 GB RSS per
    # session (549 CPU-s over 1924 s); 16 slots -> 6 CPU / 48 GB. Re-measure CpusUsage/RSS on the pilot (>=50% target).
    s = sp.add_parser('status')
    s.add_argument('--run-id', required=True)
    r = sp.add_parser('resubmit')
    r.add_argument('--run-id', required=True)
    r.add_argument('--n', type=int, default=1)
    a = ap.parse_args()
    {'bundle': bundle, 'launch': launch, 'status': status, 'resubmit': resubmit}[a.cmd](a)


if __name__ == '__main__':
    main()

"""Super traces for generated GDPval problems under the eval's own harness: aa_evals.common.pi_eval with the GDPval
port's arguments (aa_evals/common/pi_eval/port.py base_args: Pi 1.2.0 pi-cont16k, 131,072 window, no compaction,
16,384 output tokens per request, the serving registry's request block, 2 h environment clock, offline agent,
single-uid, staged task inputs, 16 GB sandbox cap, text-only model input), one attempt per problem.

  python -m m5.gdpval.traces launch --bundle B.tar --run-id ID --model serving:m5-super-1 [--max-in-flight 16]
  python -m m5.gdpval.traces status --run-id ID

Runs live under AA_RUNS_ROOT=/data/aai/m5/gdpval/runs (bench `gdpval_m5`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from aa_evals.common.pi_eval import port

RUNS_ROOT = '/data/aai/m5/gdpval/runs'
BENCH = 'gdpval_m5'
SEED_NAMESPACE = 'gdpval-m5-v1'


def bundle_tasks(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        return sorted({Path(m.name).parts[1] for m in tf.getmembers()
                       if len(Path(m.name).parts) == 3 and Path(m.name).parts[2] == 'task.toml'})


def env():
    return {**os.environ, 'AA_RUNS_ROOT': RUNS_ROOT, 'PYTHONPATH': str(port.REPO), 'PYTHONNOUSERSITE': '1'}


def launch(a):
    names = bundle_tasks(Path(a.bundle))
    args = port.base_args(BENCH, a.run_id, a.model, Path(a.bundle), names, 1, SEED_NAMESPACE, None)
    args += ['--max-in-flight', str(a.max_in_flight), '--burst', str(a.burst), '--slots', str(a.slots),
             *(['--worker-cpus', str(a.worker_cpus)] if a.worker_cpus else []),
             *(['--worker-mem-gb', str(a.worker_mem_gb)] if a.worker_mem_gb else []),
             '--worker-disk-gb', str(a.worker_disk_gb), '--cpu-bid', str(a.cpu_bid),
             # 2 h wall clock = a hang backstop here (Super-speed generation; the 131,072-token context
             # and its finish notice end attempts). The eval's model-time caps: 2 h without the capped
             # protocol (part of that protocol, #344); under it 12 h by default, a hang backstop
             # (CAPPED_WALL_CAP_SEC in aa_evals/campaign/launch.py, #531 B6)
             '--agent-wall-cap-sec', '7200']
    # Pi finish notice: pi_eval's default (v4 sticky, #375 sftnotice); the exporter ships each trial's notice map
    if a.workers:
        args += ['--workers', str(a.workers)]
    if a.no_submit:
        args.append('--no-submit')
    print(' '.join(args))
    # pi_eval submits its workers with harbor_mswea's EXCLUDE_NODES: add BAD_NODES + the coordinator's list
    from m5.gdpval.jobs import excluded_nodes
    code = ('import sys; import aa_evals.common.pi_eval.run as P; '
            f'P.EXCLUDE_NODES = tuple(sorted(set(P.EXCLUDE_NODES) | set({excluded_nodes()!r}))); '
            'P.main(sys.argv[1:])')
    r = subprocess.run([port.HARBOR_PYTHON, '-c', code, *args], env=env(), cwd=str(port.REPO))
    sys.exit(r.returncode)


def resubmit(a):
    """One more worker for an existing run (pi_eval.run --resubmit 1), with the coordinator's node exclusions."""
    from m5.gdpval.jobs import excluded_nodes
    code = ('import sys; import aa_evals.common.pi_eval.run as P; '
            f'P.EXCLUDE_NODES = tuple(sorted(set(P.EXCLUDE_NODES) | set({excluded_nodes()!r}))); '
            'P.main(sys.argv[1:])')
    r = subprocess.run([port.HARBOR_PYTHON, '-c', code, '--bench', BENCH, '--run-id', a.run_id, '--resubmit', '1'],
                       env=env(), cwd=str(port.REPO))
    sys.exit(r.returncode)


def status(a):
    r = subprocess.run([port.HARBOR_PYTHON, '-m', 'aa_evals.common.pi_eval.run', '--bench', BENCH, '--run-id',
                        a.run_id, '--status'], env=env(), cwd=str(port.REPO), capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['launch', 'status', 'resubmit'])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--bundle')
    ap.add_argument('--model', default='serving:m5-super-1')
    ap.add_argument('--max-in-flight', type=int, default=16)
    ap.add_argument('--burst', type=int, default=8)
    ap.add_argument('--slots', type=int, default=16)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--worker-cpus', type=int, default=None)  # CPUs/RAM: pi_eval's measured shape (aa_evals/pipeline.json) unless given (#451)
    ap.add_argument('--worker-mem-gb', type=int, default=None)
    ap.add_argument('--worker-disk-gb', type=int, default=150)
    ap.add_argument('--cpu-bid', type=int, default=36)
    ap.add_argument('--no-submit', action='store_true')
    a = ap.parse_args()
    {'launch': launch, 'status': status, 'resubmit': resubmit}[a.cmd](a)


if __name__ == '__main__':
    main()

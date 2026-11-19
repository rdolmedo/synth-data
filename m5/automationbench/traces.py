"""Super traces of synthetic tasks under the eval's Pi harness (#348 port; aa_evals/automationbench/pi/).

  python -m m5.automationbench.traces launch --bundle B.tar --run-id ID --engine m5-super-1 [--slots 16] [--max-in-flight 16]

Protocol = the eval's (`port.base_args`: Pi 1.2.0 pi-cont16k, 131,072-token window, no compaction, 16,384 output
tokens per request, the registry's request settings for Super, 2 h environment clock plus the eval's 2 h wall cap,
offline agent, single-uid sandboxes, text-only model input) on `serving:<engine>`; one attempt per task (1 trace per
problem). Only scheduling differs (worker size, bid, in-flight cap).
Runs live under $AA_RUNS_ROOT (default here: /data/aai/m5/automationbench/pi_runs; launch from a CPU job); collect.py packs the attempts
into shards and prunes the trial archives (/data budget).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

BENCH = 'm5_automationbench'
SEED_NAMESPACE = 'm5-automationbench-v1'
RUNS_ROOT = '/data/aai/m5/automationbench/pi_runs'   # launch from a CPU job (jobs.py): gpu123's /data writes are slow


def bundle_names(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        return sorted({Path(m.name).parts[1] for m in tf.getmembers()
                       if len(Path(m.name).parts) == 3 and Path(m.name).parts[2] == 'task.toml'})


def launch(a):
    os.environ.setdefault('AA_RUNS_ROOT', RUNS_ROOT)
    from aa_evals.common.pi_eval import port
    bundle = Path(a.bundle)
    names = bundle_names(bundle)
    args = port.base_args(BENCH, a.run_id, f'serving:{a.engine}', bundle, names, 1, SEED_NAMESPACE)
    args += ['--agent-wall-cap-sec', '7200', '--slots', str(a.slots), '--workers', str(a.workers),
             '--max-in-flight', str(a.max_in_flight), *(['--worker-cpus', str(a.worker_cpus)] if a.worker_cpus else []),
             *(['--worker-mem-gb', str(a.worker_mem_gb)] if a.worker_mem_gb else []),
             '--worker-disk-gb', str(a.worker_disk_gb),
             '--cpu-bid', str(a.bid), '--burst', str(a.burst)]
    # Pi finish notice: pi_eval's default (v4 sticky, #375 sftnotice); the exporter ships each trial's notice map
    print(' '.join(args), flush=True)
    import subprocess
    env = {**os.environ, 'PYTHONPATH': str(port.REPO), 'PYTHONNOUSERSITE': '1'}
    r = subprocess.run([port.HARBOR_PYTHON, '-m', 'm5.automationbench.pi_launch', *args], env=env, cwd=str(port.REPO))
    if r.returncode:   # port.launch, with the M5 node exclusions added to pi_eval's (pi_launch.py)
        raise SystemExit(f'pi_eval.run failed ({r.returncode})')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    l = sub.add_parser('launch')
    l.add_argument('--bundle', required=True)
    l.add_argument('--run-id', required=True)
    l.add_argument('--engine', required=True)
    l.add_argument('--slots', type=int, default=16)
    l.add_argument('--workers', type=int, default=1)
    l.add_argument('--max-in-flight', type=int, default=16)
    l.add_argument('--worker-cpus', type=int, default=None)  # CPUs/RAM: pi_eval's measured shape (aa_evals/pipeline.json) unless given (#451)
    l.add_argument('--worker-mem-gb', type=int, default=None)
    l.add_argument('--worker-disk-gb', type=int, default=100)
    l.add_argument('--bid', type=int, default=36)
    l.add_argument('--burst', type=int, default=16)
    a = ap.parse_args()
    if a.cmd == 'launch':
        launch(a)


if __name__ == '__main__':
    main()

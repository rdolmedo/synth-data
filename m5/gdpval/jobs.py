"""CPU condor jobs for the GDPval pipeline, run from a frozen detached worktree of HEAD (jobs on Lustre read scripts
live, so never from the tree being edited).

  python -m m5.gdpval.jobs submit --name NAME --cpus 8 --mem-gb 32 -- python -m m5.gdpval.stage check ...
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKTREES = Path('/data/aai/m5/gdpval/code')   # frozen checkouts on /data (home quota is small)
PY = '/home/user/python-3.11/bin/python3'
LOGS = Path('/data/aai/m5/gdpval/jobs')
BAD_NODES = ['cpu203', 'cpu205', 'cpu207', 'cpu208', 'gpu110', 'gpu132', 'gpu123', 'gpu143', 'gpu166', 'gpu172', 'gpu106', 'cpu305']
BID = 36
EXTRA_EXCLUDED = Path('/opt/aai/m5-coord/excluded-nodes.txt')   # coordinator's list


def excluded_nodes() -> list[str]:
    """BAD_NODES plus the coordinator's excluded-nodes.txt, read at submit time."""
    extra = []
    try:
        for ln in EXTRA_EXCLUDED.read_text().splitlines():
            ln = ln.split('#', 1)[0].strip()
            extra += ln.split()
    except OSError:
        pass
    return sorted(set(BAD_NODES) | set(extra))


def frozen() -> Path:
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit('commit first: jobs run from a frozen worktree of HEAD')
    path = WORKTREES / f'm5-gdpval-run-{sha[:10]}'
    if not path.exists():
        subprocess.run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(path), sha], check=True,
                       capture_output=True)
    return path


def submit(name: str, argv: list[str], cpus: int = 4, mem_gb: int = 16, disk_gb: int = 40, hours: float = 12) -> str:
    code = frozen()
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    if argv and argv[0] == 'python':
        argv = [PY, *argv[1:]]
    elif argv and argv[0] == 'decontam-python':   # sentence-transformers venv of decontam/ (data-synthesis)
        argv = ['/data/aai/tbenchtrain/decontam_calibforge/venv/bin/python', *argv[1:]]
    script = LOGS / f'{name}-{stamp}.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'
                      f'cd {code}\nexport PYTHONPATH={code} PYTHONNOUSERSITE=1 HOME=/home/user\n'
                      'export TMPDIR=${_CONDOR_SCRATCH_DIR:-/tmp} HF_HOME=${_CONDOR_SCRATCH_DIR:-/tmp}/hf\n'
                      'export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy=localnet,.internal.cluster.internal\n'
                      + ' '.join(f"'{x}'" for x in argv) + '\n')
    script.chmod(0o755)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{x}")' for x in excluded_nodes())
    sub = LOGS / f'{name}-{stamp}.sub'
    sub.write_text(f'''universe = vanilla
executable = {script}
request_cpus = {cpus}
request_memory = {mem_gb * 1024}
request_disk = {disk_gb}G
+BypassLXCfs = "true"
+JobBatchName = "m5-gdpval-{name}"
periodic_remove = (JobStatus == 2) && ((time() - EnteredCurrentStatus) > {int(hours * 3600)})
getenv = false
requirements = {req}
output = {LOGS}/{name}-{stamp}.$(Cluster).out
error = {LOGS}/{name}-{stamp}.$(Cluster).err
log = {LOGS}/{name}-{stamp}.log
queue
''')
    r = subprocess.run(['condor_submit_bid', str(BID), str(sub)], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    if r.returncode:
        raise SystemExit('condor_submit_bid failed')
    return r.stdout.strip().split()[-1].rstrip('.')


def main():
    import sys
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: jobs submit --name N [--cpus C ...] -- COMMAND...')
    i = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['submit'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--cpus', type=int, default=4)
    ap.add_argument('--mem-gb', type=int, default=16)
    ap.add_argument('--disk-gb', type=int, default=40)
    ap.add_argument('--hours', type=float, default=12)
    a = ap.parse_args(argv[:i])
    submit(a.name, argv[i + 1:], a.cpus, a.mem_gb, a.disk_gb, a.hours)


if __name__ == '__main__':
    main()

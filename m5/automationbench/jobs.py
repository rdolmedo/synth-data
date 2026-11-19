"""CPU jobs for this pipeline, run from a frozen detached worktree of the current commit (never the live tree).

  python3 m5/automationbench/jobs.py NAME --cpus 2 --mem-gb 8 [--venv automationbench|harbor|m5-hle-decon] -- MODULE ARGS...

The job itself (on a node other than gpu123, whose /data writes are slow) checks out HEAD as a detached worktree under
/data/aai/m5/automationbench/code/<sha> if it is not there yet, then runs `python -m MODULE ARGS` from it.
Files: /data/aai/m5/automationbench/jobs/NAME.{sub,sh,out,err,log}; bid 36; nodes.excluded() (BAD_NODES + m5-coord/excluded-nodes.txt) excluded.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/data/aai/m5/automationbench')
JOBS = ROOT / 'jobs'
CODE = ROOT / 'code'
VENVS = {'automationbench': '/opt/aai/venvs/automationbench/bin/python',
         'harbor': '/opt/aai/venvs/harbor/bin/python',
         'm5-hle-decon': '/data/aai/m5/hle/venv-decon/bin/python'}   # m5-hle's venv (sentence-transformers, sklearn), read-only


def head() -> str:
    if subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'm5', 'aa_evals'],
                      capture_output=True, text=True).stdout.strip():
        sys.exit('commit m5/ and aa_evals/ first: jobs run from a frozen checkout of HEAD')
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def main():
    argv = sys.argv[1:]
    i = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--cpus', type=int, default=2)
    ap.add_argument('--mem-gb', type=int, default=8)
    ap.add_argument('--disk-gb', type=int, default=10)
    ap.add_argument('--venv', default='automationbench')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--bid', type=int, default=36)
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    sha = head()
    code = CODE / sha[:12]
    JOBS.mkdir(parents=True, exist_ok=True)
    sh = JOBS / f'{a.name}.sh'
    envs = ' '.join(a.env)
    sh.write_text(f'''#!/bin/bash
set -uo pipefail
CODE={code}
if [ ! -e $CODE/.complete ]; then
  if mkdir {CODE}/.lock-{sha[:12]} 2>/dev/null; then
    git -C {REPO} worktree add --detach $CODE {sha} && touch $CODE/.complete
    rmdir {CODE}/.lock-{sha[:12]}
  else
    for i in $(seq 120); do [ -e $CODE/.complete ] && break; sleep 5; done
  fi
fi
[ -e $CODE/.complete ] || {{ echo "no frozen checkout at $CODE"; exit 2; }}
cd $CODE
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 OMP_NUM_THREADS={a.cpus} HF_HUB_OFFLINE=1 AA_RUNS_ROOT={ROOT}/pi_runs {envs}
exec {VENVS[a.venv]} -m {' '.join(shlex.quote(c) for c in cmd)}
''')
    sh.chmod(0o755)
    sys.path.insert(0, str(REPO))
    from m5.automationbench.nodes import excluded
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{x}")' for x in excluded())
    sub = JOBS / f'{a.name}.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = {sh}
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = {a.disk_gb}G
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin"
requirements = {req}
+JobBatchName = "m5-automationbench-{a.name}"
output = {JOBS}/{a.name}.out
error = {JOBS}/{a.name}.err
log = {JOBS}/{a.name}.log
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip(), f'code {code}')


if __name__ == '__main__':
    main()

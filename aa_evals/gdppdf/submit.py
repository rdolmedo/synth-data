"""Submit GDP.pdf work as HTCondor CPU jobs, each from a frozen copy of the code.

  python -m aa_evals.gdppdf.submit prepare [--cpus 16]
  python -m aa_evals.gdppdf.submit run --run-id ID -- --model serving:aa-super --limit 5 ...

`run` passes everything after `--` to `python -m aa_evals.gdppdf.run --run-id ID`. The code is copied to
<out>/code/ first (never edit a script on Lustre while a job runs it). API keys are not put in the submit
file: run.py reads them from the environment or from ~/.bashrc inside the job. PI_RESPONSE_BUDGET (the capped
protocol), when set here, is forwarded into the `run` job (aa_evals/common/response_budget_env.py).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common import aai_open, response_budget_env, serving
from aa_evals.gdppdf import data

REPO = Path(__file__).resolve().parents[2]
PYTHON = '/data/aai/aa_evals/envs/gdppdf/bin/python'
# AA_GDPPDF_RUNS overrides the runs root (keep it on /data; never /home).
RUNS = Path(os.environ.get('AA_GDPPDF_RUNS') or '/data/aai/aa_evals/runs/gdppdf')
PROXY_ENV = ('http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 '
             'HTTPS_PROXY=http://proxy.internal:8080 no_proxy=localhost,127.0.0.1,localnet,.internal.cluster.internal '
             'NO_PROXY=localhost,127.0.0.1,localnet,.internal.cluster.internal')


def git_sha():
    r = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', 'aa_evals/gdppdf', 'aa_evals/judges', 'aa_evals/tracejudge',
                            'aa_evals/common/protocol.py', 'aa_evals/common/serving.py', 'aa_evals/common/aai_open.py',
                            'tb_agents'],
                           capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ('-dirty' if dirty else '')


def freeze(dst):
    """Copy aa_evals/{__init__,gdppdf/*.py,judges/*.py,tracejudge/*.py,common/{__init__,protocol,serving,aai_open}.py} to dst/aa_evals
    and tb_agents/*.py (run.py imports its response budget for every request) to dst/tb_agents; return dst."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ('gdppdf', 'judges', 'tracejudge', 'common'):
        (dst / 'aa_evals' / sub).mkdir(parents=True)
    shutil.copy2(REPO / 'aa_evals' / '__init__.py', dst / 'aa_evals' / '__init__.py')
    for sub in ('gdppdf', 'judges', 'tracejudge'):
        for f in (REPO / 'aa_evals' / sub).glob('*.py'):
            shutil.copy2(f, dst / 'aa_evals' / sub / f.name)
    for name in ('__init__.py', 'protocol.py', 'serving.py', 'aai_open.py'):   # not the agentic harnesses under common/
        shutil.copy2(REPO / 'aa_evals' / 'common' / name, dst / 'aa_evals' / 'common' / name)
    disp = REPO / 'aa_evals' / 'common' / 'dispatch'   # --dispatch (#344): the lane and its duration table
    (dst / 'aa_evals' / 'common' / 'dispatch').mkdir()
    for f in [*disp.glob('*.py'), *disp.glob('*.json')]:
        if not f.name.startswith('test_'):
            shutil.copy2(f, dst / 'aa_evals' / 'common' / 'dispatch' / f.name)
    (dst / 'tb_agents').mkdir()
    for f in (REPO / 'tb_agents').glob('*.py'):
        if not f.name.startswith('test_'):
            shutil.copy2(f, dst / 'tb_agents' / f.name)
    (dst / 'GIT_SHA').write_text(git_sha() + '\n')
    return dst


def submit(sub_path, text, bid=None):
    """bid None: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py, #378)."""
    from aa_evals.common import condor_cost
    sub_path.write_text(text)
    condor_cost.submit(sub_path, bid)


def job(name, code, args, out_dir, cpus, mem_gb, disk_gb, env_extra=''):
    from aa_evals.common.condor_cost import bad_nodes   # the shared list, read now (tbshared; at submission)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    return f'''universe = vanilla
executable = {PYTHON}
arguments = "{args}"
initialdir = {code}
output = {out_dir}/{name}.$(Cluster).out
error = {out_dir}/{name}.$(Cluster).err
log = {out_dir}/{name}.condor.log
request_gpus = 0
request_cpus = {cpus}
request_memory = {mem_gb * 1024}
request_disk = {disk_gb}G
+MaxRuntime = 172800
+JobBatchName = "aa-gdppdf-{name}"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={code} PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 {env_extra}"
requirements = {req}
queue
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prepare')
    p.add_argument('--cpus', type=int, default=16)
    r = sp.add_parser('run')
    r.add_argument('--run-id', required=True)
    # c30 (#378): text-only runs used ~0.01 cores of 8 and 0.1 GB; pass --cpus 8 for --vision auto (page rendering)
    r.add_argument('--cpus', type=int, default=None, help='default 1 (text-only); 8 with --vision auto')
    r.add_argument('--mem-gb', type=int, default=None, help='default 4 (text-only); 32 with --vision auto')
    r.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if a.cmd == 'prepare':
        from aa_evals.gdppdf import prepare
        out = prepare.PREP_DIR
        out.mkdir(parents=True, exist_ok=True)
        code = freeze(out / f'code_{stamp}')
        submit(out / 'prepare.sub', job('prepare', code, f'-u -m aa_evals.gdppdf.prepare --workers {a.cpus}',
                                        out, a.cpus, 32, 20))
    else:
        out = RUNS / a.run_id
        out.mkdir(parents=True, exist_ok=True)
        # one frozen copy per submission: re-freezing a shared code/ dir would replace the code under a job that
        # is still running in it (its spawned workers then fail with FileNotFoundError)
        code = freeze(out / f'code_{stamp}_{os.getpid()}')
        rest = [x for x in a.rest if x != '--']
        vision = any(x == '--vision=auto' or (x == 'auto' and i and rest[i - 1] == '--vision') for i, x in enumerate(rest))
        a.cpus = a.cpus or (8 if vision else 1)
        a.mem_gb = a.mem_gb or (32 if vision else 4)
        args = ' '.join(['-u', '-m', 'aa_evals.gdppdf.run', '--run-id', a.run_id] + rest)
        env = PROXY_ENV + (f' AA_GDPPDF_RUNS={RUNS}' if os.environ.get('AA_GDPPDF_RUNS') else '')
        budget = response_budget_env.forwarded(PYTHON, code)   # PI_RESPONSE_BUDGET, only when set here
        env += (' ' + response_budget_env.condor_env(budget)) if budget else ''
        env += serving.condor_env()   # TASKSYNTH_SERVING_ROOT, only when set here
        env += aai_open.condor_env()  # AAI_OPEN_VERSION (#481), only when set here
        submit(out / 'run.sub', job('run', code, args, out, a.cpus, a.mem_gb, 20, env))
    (out / 'submitted.json').write_text(json.dumps({'cmd': a.cmd, 'at': stamp, 'code': str(code),
                                                    'git_sha': git_sha()}, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""Run an HLE eval as an HTCondor CPU job, from a frozen copy of the committed code.

    python -m aa_evals.hle.submit --run-id <id> [--hours 12] -- --model serving:aa-super --limit 30 ...

Everything after `--` goes to aa_evals.hle.run (without --run-id). The job is a plain async HTTP client
(2 CPUs); the model and judge are remote endpoints. aa_evals/ and tb_agents/ must be committed: the job runs from
`git archive HEAD` unpacked under /data/aai/aa_evals/code/hle/<sha>/, so later edits never reach it.
Job files and the receipt are in <run dir>/job/. API keys are read in the job from the environment or
~/.bashrc (client.secret), never written to the submit file. PI_RESPONSE_BUDGET (the capped protocol), when set
here, is forwarded into the job (aa_evals/common/response_budget_env.py).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common import aai_open, response_budget_env, serving
from .run import RUNS_ROOT, main as run_main

REPO = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(os.environ.get('AA_HLE_CODE_ROOT') or '/data/aai/aa_evals/code/hle')
PYTHON = '/home/user/python-3.11/bin/python3'
from aa_evals.common.condor_cost import bad_nodes  # noqa: E402  (the shared list, read at submit time)
# run.py imports tb_agents.response_budget(_single) for every item (budget or not): the frozen copy needs it
FROZEN = ('aa_evals', 'tb_agents')
PROXY_ENV = {'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
             'no_proxy': 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'}


def frozen_copy():
    git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, check=True).stdout
    if git('status', '--porcelain', '--', *FROZEN).strip():
        raise SystemExit('aa_evals/ or tb_agents/ has uncommitted changes: commit first (jobs run from a frozen copy '
                         'of HEAD)')
    sha = git('rev-parse', 'HEAD').strip()
    dest = CODE_ROOT / sha[:12]
    if not (dest / 'FROZEN_SHA').exists():
        tmp = CODE_ROOT / f'.{sha[:12]}.tmp'
        subprocess.run(['rm', '-rf', str(tmp)], check=True)
        tmp.mkdir(parents=True)
        archive = subprocess.run(['git', '-C', str(REPO), 'archive', 'HEAD', *FROZEN], capture_output=True,
                                 check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=archive, check=True)
        (tmp / 'FROZEN_SHA').write_text(sha + '\n')
        if dest.exists():
            subprocess.run(['rm', '-rf', str(dest)], check=True)
        tmp.rename(dest)
    return dest, sha


def submit(a, run_args):
    if any(x == '--run-id' or x.startswith('--run-id=') for x in run_args):
        raise SystemExit('pass --run-id to submit, not after --')
    code, sha = frozen_copy()
    from aa_evals.common.pi_eval.port import job_missing   # run.py imports these for every item, budget or not
    missing = job_missing(PYTHON, ('tb_agents.response_budget', 'tb_agents.response_budget_single'), code)
    if missing:
        raise SystemExit(f'the job interpreter {PYTHON} cannot import {missing} from {code} (python -s, #451)')
    job = RUNS_ROOT / a.run_id / 'job'
    job.mkdir(parents=True, exist_ok=True)
    receipt = job / 'job.json'
    if receipt.exists() and not a.force:
        raise SystemExit(f'{receipt} exists ({json.loads(receipt.read_text())["job_id"]}); condor_rm it or --force')
    argv = ['--run-id', a.run_id, *run_args]
    (job / 'argv.json').write_text(json.dumps(argv) + '\n')
    env = {'PYTHONPATH': str(code), 'PYTHONNOUSERSITE': '1', 'HLE_JOB_ID': '$(Cluster).$(Process)',
           'HOME': str(Path.home()), 'AA_HLE_RUNS_ROOT': str(RUNS_ROOT), **PROXY_ENV,
           **response_budget_env.forwarded(PYTHON, code),   # PI_RESPONSE_BUDGET, only when set here
           **serving.forwarded(),                           # TASKSYNTH_SERVING_ROOT, only when set here
           **aai_open.forwarded()}                          # AAI_OPEN_VERSION (#481), only when set here
    reqs = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    sub = job / 'job.sub'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m aa_evals.hle.submit exec {job / 'argv.json'}
initialdir = {code}
output = {job}/job.$(Cluster).out
error = {job}/job.$(Cluster).err
log = {job}/job.condor.log
request_cpus = {a.cpus}
request_memory = {a.memory_gib * 1024}
request_disk = {a.disk_gib}G
+JobBatchName = "aa-hle-{a.run_id}"
+MaxRuntime = {int(a.hours * 3600)}
getenv = false
environment = "{response_budget_env.condor_env(env)}"
requirements = {reqs}
queue
''')
    from aa_evals.common import condor_cost
    recs = condor_cost.submit(sub, a.bid)   # default: the CPU bid ladder (#378)
    job_id = recs[0]['cluster'] + '.0'
    receipt.write_text(json.dumps({'job_id': job_id, 'run_id': a.run_id, 'argv': argv, 'code': str(code), 'sha': sha,
                                   'bid': a.bid, 'submitted_at': time.time()}, indent=2) + '\n')
    print(f'{job_id}  logs: {job}/job.{job_id.split(".")[0]}.out')


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'exec':
        sys.exit(run_main(json.loads(Path(sys.argv[2]).read_text())))
    argv = sys.argv[1:]
    run_args = []
    if '--' in argv:
        i = argv.index('--')
        argv, run_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--bid', type=int, default=None, help='fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)')
    ap.add_argument('--hours', type=float, default=24)
    ap.add_argument('--cpus', type=int, default=1)          # c30 (#378): the client used ~0 cores, 0.4 GB
    ap.add_argument('--memory-gib', type=int, default=4)
    ap.add_argument('--disk-gib', type=int, default=5)
    ap.add_argument('--force', action='store_true')
    submit(ap.parse_args(argv), run_args)


if __name__ == '__main__':
    main()

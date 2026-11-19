"""Submit GDPval-AA work as HTCondor CPU jobs, each from a frozen copy of the code (never edit Lustre code under a
running job). Jobs run job.sh, which unpacks the venv tar into node scratch.

  python -m aa_evals.gdpval.submit run --run-id ID [--workers 4 --concurrency 24] [--judge-arm ARM ...] \
      -- --model serving:NAME --reasoning-effort medium [--limit N --repeats K]
      (init runs here, on the login node: it reads the live endpoint; then workers + one following judge per arm)
  python -m aa_evals.gdpval.submit judge --run-id ID --judge-arm ARM [--judge-endpoint SPEC] [--judge-images 0]
  python -m aa_evals.gdpval.submit prepare            # refs + expert deliverable renders (once per render version)
  python -m aa_evals.gdpval.submit workers --run-id ID --workers N    # more workers for an existing run
  python -m aa_evals.gdpval.submit tracejudge --run-id ID [--tracejudge serving:aa-judge-gemma4-31b]   # M3.4 watch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_TAR = '/data/aai/aa_evals/envs/gdpval-venv.tar'   # relocatable venv (Python 3.12, Stirrup 0.2.1); job.sh unpacks it
LOCAL_VENV = Path('/tmp/gdpval-venv')                       # the login node's copy, for `init`
PROXY = 'http://proxy.internal:8080'
NO_PROXY = 'localhost,127.0.0.1,localnet,.internal.cluster.internal'
CODE_ROOT = Path('/data/aai/aa_evals/code/gdpval')


def git_sha():
    r = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(REPO), 'diff', 'HEAD', '--', 'aa_evals/gdpval', 'aa_evals/common',
                           'aa_evals/judges', 'aa_evals/tracejudge', 'aa_evals/hle/client.py'],
                          capture_output=True, text=True).stdout
    untracked = subprocess.run(['git', '-C', str(REPO), 'ls-files', '--others', '--exclude-standard',
                                'aa_evals/gdpval'], capture_output=True, text=True).stdout
    if diff or untracked:
        h = hashlib.sha256((diff + untracked + ''.join(
            (REPO / f).read_text(errors='ignore') for f in untracked.split())).encode()).hexdigest()[:10]
        return f'{r}-dirty-{h}'
    return r


def freeze():
    sha = git_sha()
    dst = CODE_ROOT / sha
    if (dst / 'GIT_SHA').exists():
        return dst
    tmp = dst.with_name(dst.name + '.partial')
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'aa_evals').mkdir(parents=True)
    shutil.copy2(REPO / 'aa_evals' / '__init__.py', tmp / 'aa_evals' / '__init__.py')
    for sub in ('gdpval', 'judges', 'tracejudge'):
        shutil.copytree(REPO / 'aa_evals' / sub, tmp / 'aa_evals' / sub,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'sandbox'))
    # single modules other packages import: common/protocol (summary block), hle/client (the trace judge's LLM client)
    # common/jsonl: run.py imports it at module level (#451 frozen-copy test: the copy without it could not start)
    for sub, files in (('common', ('__init__.py', 'protocol.py', 'serving.py', 'jsonl.py')), ('hle', ('__init__.py', 'client.py'))):
        (tmp / 'aa_evals' / sub).mkdir()
        for f in files:
            shutil.copy2(REPO / 'aa_evals' / sub / f, tmp / 'aa_evals' / sub / f)
    (tmp / 'GIT_SHA').write_text(sha + '\n')
    tmp.rename(dst)
    return dst


def job(name, code, args, out_dir, cpus, mem_gb, disk_gb, batch):
    from aa_evals.common.condor_cost import bad_nodes   # the shared list, read now (tbshared; at submission)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    env = (f'HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={code} PYTHONNOUSERSITE=1 '
           f'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 http_proxy={PROXY} https_proxy={PROXY} HTTP_PROXY={PROXY} '
           f'HTTPS_PROXY={PROXY} no_proxy={NO_PROXY} NO_PROXY={NO_PROXY}')
    from aa_evals.common import serving
    env += serving.condor_env()   # TASKSYNTH_SERVING_ROOT, only when set here
    return f'''universe = vanilla
executable = {code}/aa_evals/gdpval/job.sh
arguments = "{args}"
initialdir = {code}
output = {out_dir}/{name}.$(Cluster).$(Process).out
error = {out_dir}/{name}.$(Cluster).$(Process).err
log = {out_dir}/{name}.condor.log
request_gpus = 0
request_cpus = {cpus}
request_memory = {mem_gb * 1024}
request_disk = {disk_gb}G
+BypassLXCfs = "true"
+MaxRuntime = 172800
+JobBatchName = "{batch}"
getenv = false
environment = "{env}"
requirements = {req}
'''


def submit(sub_path, text, n=1, bid=None):
    """bid None: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py, #378)."""
    from aa_evals.common import condor_cost
    sub_path.write_text(text + f'queue {n}\n')
    condor_cost.submit(sub_path, bid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['run', 'judge', 'prepare', 'workers', 'tracejudge'])
    ap.add_argument('--run-id')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--concurrency', type=int, default=24)
    ap.add_argument('--judge-arm', action='append', default=None)
    ap.add_argument('--judge-endpoint', default=None)
    ap.add_argument('--judge-images', type=int, default=1)
    ap.add_argument('--judge-concurrency', type=int, default=64)
    ap.add_argument('--tracejudge', default='serving:aa-judge-gemma4-31b')
    argv = sys.argv[1:]
    cut = argv.index('--') if '--' in argv else len(argv)   # everything after `--` goes to run.py
    a = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    code = freeze()
    from aa_evals.gdpval.run import RUNS
    if a.what == 'prepare':
        out = Path('/data/aai/aa_evals/data/gdpval/render_v1/jobs')
        out.mkdir(parents=True, exist_ok=True)
        submit(out / 'prepare.sub', job('prepare', code, ' '.join(['--stage prepare --concurrency 16', *rest]), out, 16, 48, 200,
                                        'aa-gdpval-prepare'))
        return
    run_dir = RUNS / a.run_id
    if a.what == 'run':
        if not (LOCAL_VENV / 'bin' / 'python').exists():
            LOCAL_VENV.mkdir(parents=True, exist_ok=True)
            subprocess.run(['tar', '-C', str(LOCAL_VENV), '-xf', VENV_TAR], check=True)
        r = subprocess.run([str(LOCAL_VENV / 'bin' / 'python'), '-B', '-m', 'aa_evals.gdpval.run', '--run-id', a.run_id, '--stage', 'init', *rest],
                           cwd=code, env={'PYTHONPATH': str(code), 'PATH': '/usr/bin:/bin', 'HOME': '/home/user'})
        if r.returncode:
            sys.exit(r.returncode)
    jobs = run_dir / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f'code.{code.name}').write_text(str(code) + '\n')
    if a.what in ('run', 'workers'):
        c = a.concurrency
        submit(jobs / 'work.sub', job('work', code, f'--run-id {a.run_id} --stage work --concurrency {c}', jobs,
                                      max(8, c), max(64, 6 * c), 60 + 8 * c, f'aa-gdpval-{a.run_id}'), n=a.workers)
    if a.what == 'tracejudge':   # follows the run and judges each attempt as it finishes; fold happens in `score`
        submit(jobs / 'tracejudge.sub', job('tracejudge', code, f'--module aa_evals.tracejudge watch --run-dir {run_dir} '
                                            f'--judge {a.tracejudge}', jobs, 2, 8, 10, f'aa-gdpval-tj-{a.run_id}'))
        return
    if a.what in ('run', 'judge'):
        for arm in a.judge_arm or (['qwen38-27b-medium-pin'] if a.what == 'judge' else []):
            extra = f' --judge-endpoint {a.judge_endpoint}' if a.judge_endpoint else ''
            submit(jobs / f'judge-{arm}.sub', job(f'judge-{arm}', code,
                   f'--run-id {a.run_id} --stage judge --follow --judge-arm {arm} --judge-images {a.judge_images} '
                   f'--concurrency {a.judge_concurrency}{extra}', jobs, 4, 16, 20, f'aa-gdpval-judge-{a.run_id}'))


if __name__ == '__main__':
    main()

"""Submit m5-gdppdf CPU jobs to HTCondor (bid 36, BAD_NODES excluded, frozen code checkout).

  python -m m5.gdppdf.condor --name cc-meta --cpus 4 --mem 32 --disk 20 -- python -u -m m5.gdppdf.docs.cc_meta ...

Code runs from a frozen detached worktree (`freeze`), never from the tree being edited: jobs on Lustre read
scripts live. Logs go to /data/aai/m5/gdppdf/logs/<name>.<cluster>.{out,err}.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data/aai/m5/gdppdf')
LOGS = ROOT / 'logs'
PYTHON = '/data/aai/aa_evals/envs/gdppdf/bin/python'   # the GDP.pdf eval venv (LiteParse 2.5.0, pandas, httpx)
# + zstandard, scikit-learn, tokenizers (setup_pylib.sh), as one tarball unpacked into job scratch (274 MB of small
# files: copying them onto Lustre took over an hour)
PYLIB_TAR = '/data/aai/m5/gdppdf/code/pylib-v1.tar.gz'
PYLIB = '$S/pylib'
REPO = Path('/opt/aai/repo')
FROZEN = Path('/opt/aai/worktrees/frozen')
# jobs import tb_agents lazily (aa_evals/{scicode,gdppdf}/run.py: the response budget, #446/#449; #451 test)
FROZEN_DIRS = ('m5', 'aa_evals', 'tb_agents')
# aa-serving task_synthesis/serve/serve_teacher.py BAD_NODES (2026-11-11)
BAD_NODES = ['cpu203', 'cpu205', 'cpu207', 'cpu208', 'gpu110', 'gpu132', 'gpu123', 'gpu143', 'gpu166', 'gpu172', 'gpu106', 'cpu305', 'gpu137']   # + gpu137: EIO on a /data append (m5-gdppdf, 11-12)
BID = 36


CODE = ROOT / 'code'
RUN_SH_TEXT = """#!/bin/bash
# m5-gdppdf job wrapper: unpack the frozen code tarball into job scratch, then run the command from there.
set -e
TAR=$1; shift
S=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}
mkdir -p "$S/code"
tar -xzf "$TAR" -C "$S/code"
basename "$TAR" .tar.gz > "$S/code/GIT_SHA"
mkdir -p "$S/pylib"
tar -xzf /data/aai/m5/gdppdf/code/pylib-v1.tar.gz -C "$S/pylib"
cd "$S/code"
export PYTHONPATH="$S/code:$S/pylib"
exec "$@"
"""


def run_sh():
    """The wrapper, versioned by content (queued jobs read it live, so it is never rewritten in place)."""
    import hashlib
    p = CODE / f'run-{hashlib.sha1(RUN_SH_TEXT.encode()).hexdigest()[:8]}.sh'
    if not p.exists():
        p.write_text(RUN_SH_TEXT)
        p.chmod(0o755)
    return p


def freeze(ref='HEAD'):
    """A frozen copy of FROZEN_DIRS at `ref`'s commit: a ~2 MB git-archive tarball on /data that each job
    unpacks into its own scratch (home is a small NFS quota; jobs on Lustre read scripts live). Returns its path."""
    here = Path(__file__).resolve().parents[2]
    sha = subprocess.run(['git', '-C', str(here), 'rev-parse', ref], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(here), 'status', '--porcelain', '--', 'm5/gdppdf'],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f'uncommitted changes under m5/gdppdf; commit first:\n{dirty}')
    CODE.mkdir(parents=True, exist_ok=True)
    run_sh()
    path = CODE / f'm5-gdppdf-{sha[:10]}.tar.gz'
    if not path.exists():
        data = subprocess.run(f'git -C {here} archive {sha} {" ".join(FROZEN_DIRS)} | gzip -n', shell=True, capture_output=True,
                              check=True).stdout
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(data)
        if tmp.read_bytes() != data:   # gpu123 has lost /data writes before: check what landed
            sys.exit(f'{tmp}: write check failed')
        tmp.rename(path)
    return path


def submit(name, cmd, cpus=2, mem_gb=8, disk_gb=10, hours=24, code=None, env=None, queue=1, args_list=None,
           bid=BID, max_running=None):
    """Submit `cmd` (a list; argv[0] 'python' is the m5 env's python). With args_list, one process per item
    with $(item) substituted in cmd. Returns the cluster id."""
    code = code or freeze()
    if str(code).endswith('.tar.gz'):   # run through the wrapper from job scratch
        cmd = [str(run_sh()), str(code)] + ([PYTHON] + cmd[1:] if cmd[0] == 'python' else cmd)
        pypath = ''   # set by run.sh
    else:
        pypath = str(code)
    LOGS.mkdir(parents=True, exist_ok=True)
    if cmd[0] == 'python':
        cmd = [PYTHON] + cmd[1:]
    exe, args = cmd[0], cmd[1:]
    envs = {'HOME': '/home/user', 'USER': 'user', 'PATH': '/usr/local/bin:/usr/bin:/bin',
            'PYTHONPATH': pypath, 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1',
            'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
            'HTTP_PROXY': 'http://proxy.internal:8080', 'HTTPS_PROXY': 'http://proxy.internal:8080',
            'no_proxy': 'localhost,127.0.0.1,.internal,.internal', 'NO_PROXY': 'localhost,127.0.0.1,.internal,.internal',
            **(env or {})}
    shared = Path('/opt/aai/m5-coord/excluded-nodes.txt')   # the coordinator's list
    extra = re.findall(r'\b[gi]\d{3}\b', shared.read_text()) if shared.exists() else []
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in sorted(set(BAD_NODES) | set(extra)))
    def q(a):   # HTCondor "new" argument syntax: "" is a literal double quote; '...' groups, '' inside is a quote
        a = a.replace('"', '""')
        return "'" + a.replace("'", "''") + "'" if (' ' in a or "'" in a or not a) else a
    arg_str = ' '.join(q(a) for a in args)
    sub = f"""universe = vanilla
executable = {exe}
arguments = "{arg_str}"
initialdir = {LOGS}
output = {LOGS}/{name}.$(Cluster).$(Process).out
error = {LOGS}/{name}.$(Cluster).$(Process).err
log = {LOGS}/{name}.condor.log
request_gpus = 0
request_cpus = {cpus}
request_memory = {int(mem_gb * 1024)}
request_disk = {disk_gb}G
+MaxRuntime = {int(hours * 3600)}
+JobBatchName = "m5-gdppdf-{name}"
getenv = false
environment = "{' '.join(f'{k}={v}' for k, v in envs.items())}"
requirements = {req}
"""
    if max_running:
        sub += f'max_materialize = {max_running}\n'
    if args_list:
        sub += 'queue item from (\n' + '\n'.join(str(x) for x in args_list) + '\n)\n'
    else:
        sub += f'queue {queue}\n'
    path = LOGS / f'{name}.{time.strftime("%Y%m%dT%H%M%S")}.sub'
    path.write_text(sub)
    r = subprocess.run(['condor_submit_bid', str(bid), str(path)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'condor_submit_bid failed: {r.stdout}\n{r.stderr}')
    cluster = int(r.stdout.strip().split('cluster')[-1].strip(' .\n'))
    rec = {'name': name, 'cluster': cluster, 'code': str(code), 'cmd': cmd, 'cpus': cpus, 'mem_gb': mem_gb,
           'n': len(args_list) if args_list else queue, 'submitted': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(LOGS / 'jobs.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'submitted {name}: cluster {cluster} ({rec["n"]} procs) from {code}', flush=True)
    return cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cpus', type=int, default=2)
    ap.add_argument('--mem', type=float, default=8)
    ap.add_argument('--disk', type=float, default=10)
    ap.add_argument('--hours', type=float, default=24)
    ap.add_argument('--items', default=None, help='file with one item per line ($(item) in the command)')
    ap.add_argument('--max-running', type=int, default=None)
    ap.add_argument('--code', default=None, help='code dir (default: freeze HEAD)')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    items = [l.strip() for l in open(a.items) if l.strip()] if a.items else None
    submit(a.name, cmd, a.cpus, a.mem, a.disk, a.hours, code=Path(a.code) if a.code else None, args_list=items,
           max_running=a.max_running)


if __name__ == '__main__':
    main()

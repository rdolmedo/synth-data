"""Submit an m5/hle module as an HTCondor CPU job from a frozen copy of HEAD.

    python -m m5.hle.submit --name <batch> [--cpus 2 --mem 8 --disk 20 --hours 12] -- <module> [args...]

The copy is /data/aai/m5/hle/code/m5-hle-<sha12> (jobs read scripts live, so never the
editing tree). Logs go to /data/aai/m5/hle/logs/<name>.<cluster>.{out,err}.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FROZEN = Path('/data/aai/m5/hle/code')   # home holds only the worktree (coordinator, 11-12)
# what the copy holds: aa_evals' HLE client imports tb_agents lazily (#449); the frozen-copy test reads this (#451)
FROZEN_DIRS = ('m5', 'aa_evals', 'tb_agents')
PYTHON = '/home/user/python-3.11/bin/python3'
VENV_PYTHON = '/data/aai/m5/hle/venv-decon/bin/python'
LOGS = Path('/data/aai/m5/hle/logs')
BAD_NODES = ['cpu203', 'cpu205', 'cpu207', 'cpu208', 'gpu110', 'gpu132', 'gpu123', 'gpu143', 'gpu166', 'gpu172', 'gpu106', 'cpu305']
PROXY_ENV = {'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
             'no_proxy': 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'}


def frozen():
    git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, check=True).stdout
    if git('status', '--porcelain', '--', 'm5', ':!m5/hle/data/leaves').strip():   # leaves land while the brainstorm runs
        raise SystemExit('m5/ has uncommitted changes: commit first')
    sha = git('rev-parse', 'HEAD').strip()
    dest = FROZEN / f'm5-hle-{sha[:12]}'
    if not (dest / 'FROZEN_SHA').exists():   # a `git archive` copy of FROZEN_DIRS (a full checkout is slow)
        tmp = FROZEN / f'.m5-hle-{sha[:12]}.tmp'
        subprocess.run(['rm', '-rf', str(tmp)], check=True)
        tmp.mkdir(parents=True)
        archive = subprocess.run(['git', '-C', str(REPO), 'archive', sha, *FROZEN_DIRS], capture_output=True,
                                 check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=archive, check=True)
        (tmp / 'FROZEN_SHA').write_text(sha + '\n')
        tmp.rename(dest)
    return dest, sha


def main():
    argv = sys.argv[1:]
    i = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cpus', type=int, default=2)
    ap.add_argument('--mem', type=int, default=8, help='GiB')
    ap.add_argument('--disk', type=int, default=20, help='GiB')
    ap.add_argument('--hours', type=float, default=12)
    ap.add_argument('--bid', type=int, default=36)
    ap.add_argument('--env', action='append', default=[], help='K=V')
    ap.add_argument('--venv', action='store_true', help='run with the decon venv (sklearn, sentence-transformers)')
    a = ap.parse_args(argv[:i])
    mod, *args = argv[i + 1:]
    code, sha = frozen()
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {'PYTHONPATH': str(code), 'PYTHONNOUSERSITE': '1', 'HOME': str(Path.home()), 'M5_SHA': sha, **PROXY_ENV,
           **dict(e.split('=', 1) for e in a.env)}
    extra = []
    try:   # the coordinator's shared list, read at submit time (one node per line, '#' comments)
        for line in Path('/opt/aai/m5-coord/excluded-nodes.txt').read_text().splitlines():
            n = line.split('#')[0].strip()
            if n:
                extra.append(n.split()[0])
    except OSError:
        pass
    reqs = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in dict.fromkeys(BAD_NODES + extra))
    sub = LOGS / f'{a.name}.sub'
    q = lambda s: "'" + s.replace("'", "''") + "'"
    sub.write_text(f'''universe = vanilla
executable = {VENV_PYTHON if a.venv else PYTHON}
arguments = "-u -m {mod} {' '.join(q(x) for x in args)}"
initialdir = {code}
output = {LOGS}/{a.name}.$(Cluster).out
error = {LOGS}/{a.name}.$(Cluster).err
log = {LOGS}/{a.name}.condor.log
request_cpus = {a.cpus}
request_memory = {a.mem * 1024}
request_disk = {a.disk}G
+JobBatchName = "m5-hle-{a.name}"
+MaxRuntime = {int(a.hours * 3600)}
getenv = false
environment = "{' '.join(f"{k}='{v}'" for k, v in env.items())}"
requirements = {reqs}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    m = re.search(r'cluster (\d+)', r.stdout)
    if m:
        print(f'job {m.group(1)}  sha {sha[:12]}  logs {LOGS}/{a.name}.{m.group(1)}.out')


if __name__ == '__main__':
    main()

"""Run one command of this package in a small CPU job (gpu123 must not write bulk data to /data).

  python -m m5.tbsci_lite.job --name NAME [--cpus 2 --mem-gb 8 --ts] -- <module> <args...>
e.g. python -m m5.tbsci_lite.job --name smoke -- m5.tbsci_lite.smoke --run-id x
--ts puts the task-synthesis checkout first on PYTHONPATH and runs from it (generation-side code).
"""
import argparse
import shlex
import subprocess

from .common import PY, ROOT, TS_CHECKOUT, bad_nodes, frozen_checkout, job_pythonpath, requirements_excluding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cpus', type=int, default=2)
    ap.add_argument('--mem-gb', type=int, default=8)
    ap.add_argument('--disk-gb', type=int, default=20)
    ap.add_argument('--bid', type=int, default=36)
    ap.add_argument('--ts', action='store_true')
    ap.add_argument('--py', default=PY)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    code = frozen_checkout()
    d = ROOT / 'logs' / 'jobs'
    d.mkdir(parents=True, exist_ok=True)
    pp = job_pythonpath(code) if a.ts else str(code)
    env = {'USER': 'user', 'HOME': '/home/user', 'PYTHONNOUSERSITE': '1', 'PATH': '/usr/local/bin:/usr/bin:/bin',
           'PYTHONPATH': pp, 'AA_RUNS_ROOT': str(ROOT / 'traces' / 'runs'),
           'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
           'no_proxy': 'localhost,127.0.0.1,.internal.cluster.internal,localnet'}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    args = ' '.join(shlex.quote(x) for x in cmd[1:])
    sub = d / f'{a.name}.sub'
    sub.write_text(f'''universe = vanilla
executable = {a.py}
arguments = "-u -m {cmd[0]} {args}"
initialdir = {TS_CHECKOUT if a.ts else code}
output = {d}/{a.name}.$(Cluster).out
error = {d}/{a.name}.$(Cluster).err
log = {d}/condor.log
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = {a.disk_gb}G
+JobBatchName = "m5-tbsci-{a.name}"
+MaxRuntime = 86400
getenv = false
environment = "{environment}"
requirements = {requirements_excluding(bad_nodes())}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip(), f'-> {d}/{a.name}.<cluster>.out')


if __name__ == '__main__':
    main()

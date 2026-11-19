"""(#353, from task_synthesis/base_image/build_base.py) Build m5/tbsci_lite/base_image/Dockerfile with harbor_ns and publish it as a shared base (local/tasksynth-base:N).

  python -m task_synthesis.base_image.build_base submit        # CPU job (needs the proxy for apt/pip)
  python -m m5.tbsci_lite.base_image.build_base run           # inside the job

Task Dockerfiles then start `FROM local/tasksynth-base:N`; harbor_ns reads <shared>/bases/<name>.sqfs instead of a
registry. The job also writes <shared>/bases/<name>.manifest.json (dpkg packages, pip freeze, tool versions), which the
generator prompt shows so instructions only promise what the image has. Bump VERSION whenever the Dockerfile changes.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from synth.common import PROXY_ENV, PYTHON, atomic
from m5.tbsci_lite.common import TS_CHECKOUT, bad_nodes, frozen_checkout, job_pythonpath, requirements_excluding
from synth.prefix_bases import SHARED, dpkg_installed, safe_name, shared_paths, squash, unsquash_ok
ROOT = Path('/data/aai/m5/tbsci_lite')

VERSION = 1
REF = f'local/tasksynth-sci:{VERSION}'
HERE = Path(__file__).resolve().parent
JOBS = ROOT / 'base_image'
TOOLS = ['python3', 'python', 'pip3', 'git', 'gcc', 'gfortran', 'make', 'cmake', 'Rscript', 'ngspice', 'jq', 'curl']


def manifest_path():
    return SHARED / 'bases' / f'{safe_name(REF)}.manifest.json'


def inventory(rootfs):
    """What the image offers, read from its rootfs (no sandbox needed)."""
    rootfs = Path(rootfs)
    pkgs = sorted(p for p in (dpkg_installed(rootfs) or set()) if p)
    py = []
    for site in rootfs.glob('usr/local/lib/python3*/dist-packages'):
        for d in site.glob('*.dist-info'):
            m = re.match(r'(.+?)-(\d[^-]*)\.dist-info$', d.name)
            if m:
                py.append(f'{m.group(1).lower().replace("_", "-")}=={m.group(2)}')
    tools = {t: any((rootfs / p / t).exists() for p in ('usr/bin', 'bin', 'usr/local/bin')) for t in TOOLS}
    pyver = sorted(p.name for p in rootfs.glob('usr/bin/python3.*') if re.fullmatch(r'python3\.\d+', p.name))
    return {'ref': REF, 'python': pyver, 'pip': sorted(py), 'apt_count': len(pkgs), 'apt': pkgs, 'tools': tools}


async def build():
    from harbor_ns.cli import default_store
    sqfs, meta_path = shared_paths(REF)
    if sqfs.exists() and meta_path.exists():
        print(f'{REF} already published at {sqfs}; bump VERSION to rebuild')
        return
    store = default_store()
    t0 = time.time()
    key, rootfs, meta = await store.build(HERE, label=f'tasksynth-sci-{VERSION}')
    inv = inventory(rootfs)
    missing = [t for t, ok in inv['tools'].items() if not ok]
    if missing:
        raise SystemExit(f'base image lacks {missing}; build log: {store.image_dir(key)}/build.log')
    work = Path(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')) / 'squash'
    work.mkdir(parents=True, exist_ok=True)
    local = work / 'base.sqfs'
    squash(rootfs, local, True)
    ok, err = unsquash_ok(local, work)
    xattrs = True
    if not ok:
        xattrs = False
        local.unlink(missing_ok=True)
        squash(rootfs, local, False)
        ok, err = unsquash_ok(local, work)
    if not ok:
        raise SystemExit(f'squashfs does not unpack: {err}')
    tmp = sqfs.with_name(f'.{sqfs.name}.{os.getpid()}')
    subprocess.run(['dd', f'if={local}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'conv=fsync', 'status=none'],
                   check=True)
    tmp.replace(sqfs)
    atomic(meta_path, {**meta.to_json(), 'base': REF, 'oci': True})
    atomic(manifest_path(), {**inv, 'bytes': local.stat().st_size, 'xattrs': xattrs, 'seconds': round(time.time() - t0),
                             'host': socket.gethostname(), 'time': time.time(),
                             'dockerfile': (HERE / 'Dockerfile').read_text()})
    print(json.dumps({'ref': REF, 'sqfs': str(sqfs), 'bytes': local.stat().st_size, 'seconds': round(time.time() - t0)}))


def submit(a):
    JOBS.mkdir(parents=True, exist_ok=True)
    repo = frozen_checkout()
    env = {'USER': os.environ.get('USER', 'user'), 'HOME': os.environ.get('HOME', '/home/user'),
           'PYTHONNOUSERSITE': '1', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONPATH': job_pythonpath(repo),
           **PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = JOBS / f'build-v{VERSION}.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export HARBOR_NS_LOCAL=$_CONDOR_SCRATCH_DIR/harbor_ns TMPDIR=$_CONDOR_SCRATCH_DIR; exec {PYTHON} -u -m m5.tbsci_lite.base_image.build_base run'"
initialdir = {TS_CHECKOUT}
output = {JOBS}/build-v{VERSION}.$(Cluster).out
error = {JOBS}/build-v{VERSION}.$(Cluster).err
log = {JOBS}/build.condor.log
request_cpus = 8
request_memory = 32768
request_disk = 100G
+JobBatchName = "tasksynth-sci-v{VERSION}"
+MaxRuntime = 14400
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {requirements_excluding(bad_nodes())}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    sub.add_parser('run')
    s = sub.add_parser('submit')
    s.add_argument('--bid', type=int, default=36)
    a = ap.parse_args()
    if a.op == 'run':
        asyncio.run(build())
    else:
        submit(a)


if __name__ == '__main__':
    main()

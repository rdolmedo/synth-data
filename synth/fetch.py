"""Pinned checkpoint download for a new teacher: Hub revision -> node scratch -> /data, verified.

    python -m synth.fetch <repo_id> <revision> <dest>            # inside a CPU job
    python -m synth.fetch <repo_id> <revision> <dest> --submit   # submit that CPU job

Files download to node-local /tmp (or the condor scratch dir when /tmp is too small), never
straight to /data: the Hub client's small writes run at a few MB/s on Lustre, while one
`dd bs=64M oflag=direct` per file runs at ~1 GB/s. Each copy lands under a temporary name and
is checked against the Hub's metadata (SHA-256 for LFS files, the git blob hash otherwise)
before the rename. <dest>/checkpoint-ready.json is
written last, so its presence means every file of the revision is present and verified.
A rerun skips files already verified in <dest>.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import condor
from .common import PROXY_ENV, PYTHON, REPO, atomic, read, sha


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def verified(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    return (sha(path) == entry['sha256']) if entry['sha256'] else git_blob(path) == entry['blob_id']


def staging(size):
    """Node-local download directory with room for the whole download."""
    for base in ('/tmp', os.environ.get('_CONDOR_SCRATCH_DIR')):
        if base and shutil.disk_usage(base).free > size * 1.05 + (20 << 30):
            path = Path(base) / 'synth-fetch'
            path.mkdir(exist_ok=True)
            print(f'staging in {path}', flush=True)
            return path
    raise RuntimeError(f'no node-local directory has {size / 1e9:.0f} GB free; raise request_disk')


def fetch(repo, revision, dest):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f'{repo}@{revision} resolved to {info.sha}; pin a full commit hash')
    files = {s.rfilename: {'size': s.size, 'sha256': s.lfs.sha256 if s.lfs else None, 'blob_id': s.blob_id}
             for s in info.siblings}
    dest.mkdir(parents=True, exist_ok=True)
    done = read(dest / '.fetch-verified.json', {})
    todo = [n for n in files if done.get(n) != files[n]]
    print(f'{repo}@{revision}: {len(files)} files, {sum(f["size"] for f in files.values()) / 1e9:.1f} GB, '
          f'{len(todo)} to fetch', flush=True)
    scratch = staging(sum(files[n]['size'] for n in todo))
    if todo:
        start = time.time()
        snapshot_download(repo, revision=revision, local_dir=scratch, allow_patterns=todo, max_workers=16)
        print(f'downloaded in {time.time() - start:.0f} s', flush=True)

    def place(name):
        out = dest / name
        if not verified(out, files[name]):
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f'.{out.name}.fetch')
            subprocess.run(['dd', f'if={scratch / name}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'conv=fsync',
                            'status=none'], check=True)
            if not verified(tmp, files[name]):
                tmp.unlink()
                raise ValueError(f'{name}: content does not match the Hub metadata')
            tmp.replace(out)
        (scratch / name).unlink(missing_ok=True)
        return name

    with ThreadPoolExecutor(8) as pool:
        for name in pool.map(place, todo):
            done[name] = files[name]
            atomic(dest / '.fetch-verified.json', done)
            print(f'verified {name}', flush=True)
    missing = [n for n in files if done.get(n) != files[n]]
    if missing:
        raise RuntimeError(f'unverified files: {missing[:5]}')
    atomic(dest / 'checkpoint-ready.json', {'repository': repo, 'revision': revision, 'files': files,
                                            'verified_at': time.time()})
    print(f'{dest} ready', flush=True)


def submit(repo, revision, dest, bid):
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest.parent / f'fetch-{dest.name}'
    env = {'USER': 'user', 'HOME': os.environ.get('HOME', '/home/user'), 'PYTHONNOUSERSITE': '1',
           'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': str(REPO), 'HF_XET_HIGH_PERFORMANCE': '1',
           **PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = stem.with_suffix('.sub')
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m synth.fetch {repo} {revision} {dest}
initialdir = {REPO}
output = {stem}.$(Cluster).out
error = {stem}.$(Cluster).err
log = {stem}.condor.log
request_cpus = 16
request_memory = 32768
request_disk = 1100G
+JobBatchName = "synth-fetch-{dest.parent.name}"
+MaxRuntime = 172800
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {condor.requirements({"gpu132"} | condor.bad_nodes(), False)}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('revision')
    ap.add_argument('dest', type=Path)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--bid', type=int, default=56)
    args = ap.parse_args()
    if args.submit:
        submit(args.repo, args.revision, args.dest.resolve(), args.bid)
    else:
        fetch(args.repo, args.revision, args.dest.resolve())


if __name__ == '__main__':
    main()

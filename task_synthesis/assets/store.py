"""The asset store: one tar.zst per asset (the /data file quota counts files) and JSONL records with provenance.

  <ROOT>/assets/store/<id[-2:]>/<id>.tar.zst
  <ROOT>/assets/index/<shard>.jsonl          one record per line; one shard per writing process (Lustre has no
                                             kernel locks, so writers never share a file)

Record: asset_id, source, key, url, page_url, version, sha256 (of what was downloaded), license, license_raw,
license_verdict, attribution, title, description, cls (asset class), formats, language, target (class x domain x
formats the sourcing aimed at), query, strategy, size_bytes, n_files, manifest, scrub, tar, tar_bytes, tar_sha256,
fetched_at.
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from ..common import ROOT
from . import manifest as manifest_mod
from .scrub import scrub_tree

ASSETS = ROOT / 'assets'
STORE, INDEX = ASSETS / 'store', ASSETS / 'index'


def asset_id(source, key):
    return f'{source}-{hashlib.sha256(str(key).encode()).hexdigest()[:14]}'


def tar_path(aid):
    return STORE / aid[-2:] / f'{aid}.tar.zst'


def copy_to_fast(src, dst):
    """Node-local file -> /data: dd with direct I/O above 1 MB (plain cp through the page cache is ~20 MB/s)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    if src.stat().st_size > 1 << 20:
        subprocess.run(['dd', f'if={src}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    else:
        shutil.copyfile(src, tmp)
    tmp.replace(dst)


def make_tar(src_dir, out, paths=None):
    """tar.zst of src_dir (or of the listed relative paths in it), owner-neutral and sorted."""
    members = paths if paths else ['.']
    subprocess.run(['tar', '--zstd', '-cf', str(out), '--sort=name', '--owner=0', '--group=0', '--numeric-owner',
                    '--mtime=@1700000000',
                    '-C', str(src_dir), *members], check=True, capture_output=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


class Writer:
    def __init__(self, shard=None):
        self.shard = shard or f'{socket.gethostname()}-{os.getpid()}-{int(time.time())}'
        INDEX.mkdir(parents=True, exist_ok=True)
        self.path = INDEX / f'{self.shard}.jsonl'

    def put(self, staging, record, scrub=True):
        """staging: node-local directory holding the asset's files (already fetched and unpacked). Scrubs it,
        builds the manifest, writes the tar to the store and appends the record. -> record."""
        staging = Path(staging)
        record = dict(record)
        if scrub:
            record['scrub'] = scrub_tree(staging)
        m = manifest_mod.build(staging)
        if record.get('git'):
            m['stats']['(git history)'] = record['git']
        record.update(manifest=m, n_files=m['n_files'], size_bytes=m['total_bytes'])
        aid = record['asset_id']
        with tempfile.TemporaryDirectory(prefix='asset-tar-') as tmp:
            out = Path(tmp) / f'{aid}.tar.zst'
            make_tar(staging, out)
            record.update(tar=str(tar_path(aid).relative_to(ASSETS)), tar_bytes=out.stat().st_size,
                          tar_sha256=sha256_file(out))
            copy_to_fast(out, tar_path(aid))
        record.setdefault('fetched_at', time.time())
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        return record

    def reject(self, record, reason):
        """Candidates that failed (license, size, scrub, fetch) are logged too: cost and pass-rate accounting."""
        with open(INDEX / f'{self.shard}.rejected.jsonl', 'a') as f:
            f.write(json.dumps({**{k: record.get(k) for k in ('asset_id', 'source', 'key', 'url', 'license_raw',
                                                              'cls', 'target', 'query', 'strategy', 'size')},
                                'reason': str(reason)[:500], 'time': time.time()}) + '\n')


def load_index(with_manifest=True):
    """asset_id -> record (the last record of an id wins)."""
    out = {}
    for p in sorted(INDEX.glob('*.jsonl')):
        if p.name.endswith('.rejected.jsonl'):
            continue
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not with_manifest:
                    r.pop('manifest', None)
                out[r['asset_id']] = r
    return out


def load_rejected():
    out = []
    for p in sorted(INDEX.glob('*.rejected.jsonl')):
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def members_of(record):
    r = subprocess.run(['tar', '--zstd', '-tf', str(ASSETS / record['tar'])], capture_output=True, text=True, check=True)
    return [m.rstrip('/') for m in r.stdout.splitlines()]


def extract(record, dest, paths=None):
    """Extract the asset (or the listed relative paths: files or directories) into dest. Paths the tar does not hold
    are skipped (a fit check may name one loosely); a selection matching nothing extracts the whole asset."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    members = []
    if paths:
        want = {'./' + p.strip().removeprefix('./').strip('/') for p in paths if p.strip() not in ('', '.', './')}
        have = members_of(record)
        members = sorted({m for m in have if m in want or any(m.startswith(w + '/') for w in want)})
        members = [m for m in members if not any(m.startswith(o + '/') for o in members if o != m)]
    subprocess.run(['tar', '--zstd', '-xf', str(ASSETS / record['tar']), '-C', str(dest), *members],
                   check=True, capture_output=True)
    return dest


def task_tar(record, out, paths=None):
    """The asset (or a selection of it) as one tar.zst for a task's environment."""
    with tempfile.TemporaryDirectory(prefix='asset-sel-') as tmp:
        root = extract(record, Path(tmp) / 'a', paths)
        make_tar(root, out)
    return out

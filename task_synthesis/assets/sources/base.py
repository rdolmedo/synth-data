"""Adapter interface. A source searches a catalog (metadata only, cheap) and fetches one candidate into a node-local
staging directory (size-capped). Sourcing is driven by a target, never by the agent action (BLUEPRINT.md: keep the
action out of sourcing, so one asset can serve several tasks):

  target = {'cls': 'tabular-data', 'formats': ['csv'], 'language': None, 'services': [], 'domain': 'life-sciences',
            'scale': 'none' | 'snippet' | 'mini-package' | 'real-project'}

A candidate is a dict: source, key (unique in the source), url, page_url, title, description, license_raw, size
(bytes, may be None), formats, meta (source-specific). fetch() returns the provenance fields of the record (version,
sha256, url, attribution, license_raw if refined, ...) and leaves the asset's files in `staging`.
"""
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

MB = 1 << 20
# Download caps per class: tasks stay lean (10 GB disk), bundles stay shippable.
MAX_BYTES = {'ml-model': 400 * MB, 'media': 60 * MB, 'tabular-data': 80 * MB, 'database': 80 * MB,
             'scientific-data': 60 * MB, 'document': 40 * MB, 'git-repo': 80 * MB, 'binary': 60 * MB}
DEFAULT_MAX = 60 * MB
MAX_FILES = 4000


class Reject(Exception):
    """The candidate is unusable (license, size, content); logged with its reason."""


class Source:
    name = ''
    classes = ()

    def __init__(self, http):
        self.http = http

    async def search(self, target, query, n):
        raise NotImplementedError

    async def fetch(self, cand, staging, target):
        raise NotImplementedError


def cap(target):
    return MAX_BYTES.get(target['cls'], DEFAULT_MAX)


def unpack(archive, dest, strip_single_top=True):
    """tar/zip -> dest (path traversal and links refused); a single top-level directory is stripped."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.unpack')
    tmp.mkdir()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if len(names) > MAX_FILES * 2:
                    raise Reject(f'archive has {len(names)} entries')
                for n in names:
                    if n.startswith('/') or '..' in Path(n).parts:
                        raise Reject(f'unsafe path in archive: {n}')
                z.extractall(tmp)
        else:
            with tarfile.open(archive) as t:
                members = t.getmembers()
                if len(members) > MAX_FILES * 2:
                    raise Reject(f'archive has {len(members)} entries')
                safe = [m for m in members if (m.isfile() or m.isdir()) and not m.name.startswith('/')
                        and '..' not in Path(m.name).parts]
                t.extractall(tmp, members=safe, filter='data')
        tops = list(tmp.iterdir())
        src = tops[0] if strip_single_top and len(tops) == 1 and tops[0].is_dir() else tmp
        for x in src.iterdir():
            shutil.move(str(x), str(dest / x.name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def count_files(d):
    return sum(1 for p in Path(d).rglob('*') if p.is_file())


def git(*args, cwd=None, timeout=900):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise Reject(f'git {args[0]} failed: {r.stderr[-300:]}')
    return r.stdout

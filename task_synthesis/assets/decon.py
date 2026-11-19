"""Asset-level decontamination against the eval sets (TB2.1, TB-Lite). Instruction decon (pipeline/textsim.py) misses
a task built on the same upstream project or data file as an eval task, so assets are checked too:

  - keys:  GitHub owner/repo and Hugging Face repo ids referenced anywhere in the eval tasks -> candidates blocked
  - urls:  every other URL in the eval tasks (query stripped) -> an asset fetched from one is blocked
  - files: SHA-256 of every eval file of 512 bytes or more (license texts and VCS boilerplate excluded) -> an asset
           containing any of them is rejected

The blocklist is built once from the eval tars into <ROOT>/assets/decon.json.
"""
import hashlib
import json
import re
import tarfile
from pathlib import Path

from ..common import ROOT
from ..pipeline.textsim import EVAL_TARS

CACHE = ROOT / 'assets' / 'decon.json'
# license texts, VCS/CI boilerplate and build-tool wrappers identical across thousands of repositories (prefix
# match), and autotools helper scripts (exact names)
BOILERPLATE = re.compile(r'(^|/)((license|licence|copying|notice|authors|\.gitignore|\.gitattributes|\.editorconfig|'
                         r'code_of_conduct|contributing|gradlew|gradle-wrapper|mvnw|maven-wrapper|\.travis|'
                         r'\.pre-commit-config)([.\-][^/]*)?|install-sh|config\.guess|config\.sub|ltmain\.sh|missing|depcomp|'
                         r'compile|ar-lib|test-driver|py\.typed)$', re.I)
GH = re.compile(r'github\.com[/:]([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)')
HF = re.compile(r'huggingface\.co/(?:datasets/)?([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)')
URL = re.compile(r'https?://[^\s"\'<>)\]}]+')
GENERIC_HOSTS = ('pypi.org', 'files.pythonhosted.org', 'download.pytorch.org', 'registry.npmjs.org', 'deb.debian.org',
                 'archive.ubuntu.com', 'security.ubuntu.com', 'docs.python.org', 'www.w3.org', 'json-schema.org',
                 'schemas.', 'example.com', 'localhost', '127.0.0.1')


def build():
    keys, urls, files = set(), set(), {}
    for name, path in EVAL_TARS.items():
        with tarfile.open(path) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                task = m.name.split('/')[1] if '/' in m.name else m.name
                data = tar.extractfile(m).read()
                if m.size >= 512 and not BOILERPLATE.search(m.name):
                    files.setdefault(hashlib.sha256(data).hexdigest(), f'{name}/{task}')
                if m.size > 4_000_000:
                    continue
                text = data.decode(errors='replace')
                for o, r in GH.findall(text):
                    if o.lower() not in ('repos', 'search', 'orgs', 'users', 'gists'):
                        keys.add(f'{o}/{r.removesuffix(".git")}'.lower())
                for o, r in HF.findall(text):
                    keys.add(f'{o}/{r}'.lower())
                for u in URL.findall(text):
                    u = u.split('?')[0].split('#')[0].rstrip('.,;:/').lower()
                    if not any(h in u for h in GENERIC_HOSTS) and u.count('/') >= 3:
                        urls.add(u)
    out = {'keys': sorted(keys), 'urls': sorted(urls), 'files': files}
    CACHE.write_text(json.dumps(out))
    return out


_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = json.loads(CACHE.read_text()) if CACHE.exists() else build()
        _DB['keys'], _DB['urls'] = set(_DB['keys']), set(_DB['urls'])
    return _DB


def blocked_candidate(cand):
    """-> reason or None. Checks the catalog key (GitHub repo, HF id) and URLs before anything is fetched."""
    d = db()
    key = str(cand.get('key') or '').lower()
    repo = key.split(':')[0]
    if repo in d['keys'] or key in d['keys']:
        return f'eval tasks reference {repo}'
    for u in (cand.get('url'), cand.get('page_url')):
        if u and u.split('?')[0].rstrip('/').lower() in d['urls']:
            return f'eval tasks reference {u}'
    return None


def shared_files(root, limit=5):
    """Files of a fetched asset whose SHA-256 equals an eval file's -> [(relative path, eval task)]."""
    d = db()['files']
    hits = []
    for p in Path(root).rglob('*'):
        if p.is_symlink() or not p.is_file() or p.stat().st_size < 512 or p.stat().st_size > 200 << 20:
            continue
        rel = str(p.relative_to(root))
        if '.git/' in rel or BOILERPLATE.search(rel):
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in d:
            hits.append((rel, d[h]))
            if len(hits) >= limit:
                break
    return hits


if __name__ == '__main__':
    b = build()
    print(f'{len(b["keys"])} repo keys, {len(b["urls"])} URLs, {len(b["files"])} file hashes -> {CACHE}')

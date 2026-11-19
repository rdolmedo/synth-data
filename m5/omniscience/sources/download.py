"""One-off source downloads for m5-omniscience (runs in an HTCondor CPU job; never on gpu123).

Everything is fetched into local /tmp, filtered, packed as jsonl.zst and copied to /data with dd oflag=direct.
Each output gets a manifest row (source, licence, provenance, SHA-256, n docs). Parts:
  docs        sparse shallow clones of official language/library doc and changelog repos (docs_repos.json)
  cstd        WG14 C standard drafts (C11 N1570 html, C23 N3220 pdf -> pdftotext)
  changelogs  CHANGELOG/NEWS/HISTORY/RELEASES files from the local swezero bare-repo cache (read in place)
  pmc         Common Pile PubMed Central open-access full text (CC BY/BY-SA/CC0), shard prefixes
  cap         Common Pile's Caselaw Access Project shards (public domain): state and federal opinions
  edgar       EDGAR-CORPUS 10-K sections (Apache-2.0), a few years
Usage: python download.py OUT_DIR part [part ...]
"""
import gzip
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import zstandard

HERE = Path(__file__).parent
TMP = Path(os.environ.get('TMPDIR_OMNI', '/tmp/omni_dl'))
PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({'http': 'http://proxy.internal:8080', 'https': 'http://proxy.internal:8080'}))
TEXT_EXT = {'.md', '.mdx', '.rst', '.txt', '.html', '.htm', '.adoc', '.asciidoc', '.texi', '.rd', '.xml', '.markdown', ''}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fetch(url, dest, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'research-data-fetch/1.0'})
            with PROXY.open(req, timeout=600) as r, open(dest, 'wb') as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            return dest
        except Exception as e:
            log('fetch retry', url, e)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f'fetch failed: {url}')


class Pack:
    """jsonl.zst writer in /tmp; close() copies to OUT with dd and returns a manifest row."""

    def __init__(self, out_dir, name):
        self.name, self.out_dir = name, Path(out_dir)
        self.tmp = TMP / f'{name}.jsonl.zst'
        self.f = open(self.tmp, 'wb')
        self.w = zstandard.ZstdCompressor(level=10, threads=2).stream_writer(self.f)
        self.n = 0

    def add(self, rec):
        self.w.write((json.dumps(rec, ensure_ascii=False) + '\n').encode())
        self.n += 1

    def close(self, **meta):
        self.w.close()
        sha = hashlib.sha256(self.tmp.read_bytes()).hexdigest()
        dst = self.out_dir / self.tmp.name
        subprocess.run(['dd', f'if={self.tmp}', f'of={dst}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f'{dst}: sha mismatch after copy')
        row = {'file': dst.name, 'n_docs': self.n, 'sha256': sha, 'bytes': dst.stat().st_size,
               'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **meta}
        with open(self.out_dir / 'MANIFEST.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        log('wrote', dst, self.n)
        self.tmp.unlink()
        return row


def part_docs(out):
    repos = json.loads((HERE / 'docs_repos.json').read_text())
    pk = Pack(out, 'docs')
    prov = []
    for repo, paths, lic, sub in repos:
        d = TMP / 'git' / repo.replace('/', '__')
        try:
            if not d.exists():
                env = dict(os.environ, https_proxy='http://proxy.internal:8080', http_proxy='http://proxy.internal:8080', GIT_TERMINAL_PROMPT='0')
                subprocess.run(['git', 'clone', '-q', '--depth', '1', '--filter=blob:none', '--sparse',
                                f'https://github.com/{repo}.git', str(d)], check=True, env=env, timeout=3600)
                subprocess.run(['git', '-C', str(d), 'sparse-checkout', 'set', '--no-cone', *[('/' + p) for p in paths]],
                               check=True, env=env, timeout=3600)
            sha = subprocess.check_output(['git', '-C', str(d), 'rev-parse', 'HEAD'], text=True).strip()
        except Exception as e:
            log('clone failed', repo, e)
            prov.append({'repo': repo, 'error': str(e)[:200]})
            continue
        n = 0
        for f in sorted(d.rglob('*')):
            if not f.is_file() or '.git' in f.parts or f.suffix.lower() not in TEXT_EXT:
                continue
            if f.stat().st_size > 8_000_000 or f.stat().st_size < 200:
                continue
            try:
                text = f.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            rel = str(f.relative_to(d))
            pk.add({'doc_id': f'docs:{repo}:{rel}', 'source': 'software_docs', 'repo': repo, 'path': rel, 'commit': sha,
                    'licence': lic, 'subtopic_hint': sub, 'title': f'{repo} {rel}', 'text': text})
            n += 1
        prov.append({'repo': repo, 'commit': sha, 'licence': lic, 'paths': paths, 'n_files': n})
        log(repo, n)
        subprocess.run(['rm', '-rf', str(d)])
    pk.close(source='software_docs', licence='per repo (see repos)', provenance='github sparse shallow clones',
             repos=prov)


def part_cstd(out):
    pk = Pack(out, 'cstd')
    items = [('https://port70.net/~nsz/c/c11/n1570.html', 'n1570.html', 'C11 draft N1570'),
             ('https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3220.pdf', 'n3220.pdf', 'C2y/C23 working draft N3220'),
             ('https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf', 'n3096.pdf', 'C23 draft N3096')]
    prov = []
    for url, name, title in items:
        p = TMP / name
        try:
            fetch(url, p)
        except RuntimeError as e:
            prov.append({'url': url, 'error': str(e)})
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if name.endswith('.pdf'):
            text = subprocess.check_output(['pdftotext', '-layout', str(p), '-'], text=True, errors='replace')
        else:
            text = re.sub(r'<[^>]+>', '', p.read_text(errors='replace'))
        # one record per ~section-sized chunk keeps records small; the pool chunks further
        for i in range(0, len(text), 200_000):
            pk.add({'doc_id': f'cstd:{name}:{i // 200_000}', 'source': 'software_docs', 'repo': 'WG14', 'path': name,
                    'licence': 'ISO/IEC JTC1 SC22 WG14 public working draft', 'subtopic_hint': 'C programming language',
                    'title': title, 'text': text[i:i + 200_000]})
        prov.append({'url': url, 'sha256': sha})
    pk.close(source='software_docs', licence='WG14 public drafts', provenance=prov)


CHANGE_RE = re.compile(r'(^|/)(CHANGELOG|CHANGES|HISTORY|NEWS|RELEASES|RELEASE[-_]NOTES|RELEASENOTES|UPGRADING|MIGRATION)[^/]*$', re.I)


def part_changelogs(out):
    root = Path('/data/aai/swezero-gen/repo-cache')
    pk = Pack(out, 'changelogs')
    nrepo = 0
    for repo in sorted(root.iterdir()):
        try:
            ls = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-l', 'HEAD'], text=True,
                                         timeout=300, stderr=subprocess.DEVNULL)
            sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
                                          stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
        lic = ''
        got = 0
        for line in ls.splitlines():
            meta, path = line.split('\t', 1)
            parts = meta.split()
            if len(parts) < 4 or parts[1] != 'blob' or parts[3] == '-':
                continue
            size = int(parts[3])
            if re.match(r'^(LICENSE|LICENCE|COPYING)', path, re.I) and not lic:
                try:
                    lic = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', parts[2]], text=True,
                                                  errors='replace')[:300].strip().splitlines()[0][:120]
                except Exception:
                    pass
            if path.count('/') > 2 or not CHANGE_RE.search(path) or not (500 < size < 5_000_000):
                continue
            if '/vendor/' in path or 'node_modules' in path or '/testdata/' in path:
                continue
            try:
                text = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', parts[2]], text=True,
                                               errors='replace')
            except Exception:
                continue
            pk.add({'doc_id': f'chg:{repo.name}:{path}', 'source': 'changelog', 'repo': repo.name, 'path': path,
                    'commit': sha, 'licence_line': lic, 'title': f'{repo.name} {path}', 'text': text})
            got += 1
        nrepo += bool(got)
    log('changelog repos with files', nrepo)
    pk.close(source='changelog', licence='per repo (licence_line)', provenance=f'{root} (swezero bare repos, HEAD)')


def stream_lines(url, max_bytes, gz=True):
    """Lines of a remote (gzipped) jsonl, reading at most max_bytes of the compressed stream (a prefix of the shard).
    The proxy gives ~0.5 MB/s per stream, so shards are sampled by prefix and in parallel."""
    import zlib
    req = urllib.request.Request(url, headers={'User-Agent': 'research-data-fetch/1.0'})
    d = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None
    got, buf = 0, b''
    with PROXY.open(req, timeout=600) as r:
        while got < max_bytes:
            b = r.read(1 << 20)
            if not b:
                break
            got += len(b)
            buf += d.decompress(b) if d else b
            *lines, buf = buf.split(b'\n')
            for l in lines:
                if l.strip():
                    yield l
    # the last partial line is dropped


def parallel_shards(urls, max_bytes, handle, gz=True, workers=8):
    """Run handle(url, line_iter) over shard prefixes in parallel threads; returns per-url byte budgets used."""
    import concurrent.futures as cf

    def one(u):
        for i in range(4):
            try:
                return handle(u, stream_lines(u, max_bytes, gz))
            except Exception as e:
                log('shard retry', u, e)
                time.sleep(30)
        return None
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, urls))


def part_pmc(out, n_shards=8, max_bytes=200 << 20):
    """Common Pile PubMed Central (CC BY / CC BY-SA / CC0 open-access articles, markdown full text)."""
    import threading
    base = 'https://huggingface.co/datasets/common-pile/pubmed/resolve/main/data/'
    urls = [f'{base}{i:05d}_pubmedcentral.jsonl.gz' for i in range(0, 17, max(1, 17 // n_shards))][:n_shards]
    pk, lock = Pack(out, 'pmc'), threading.Lock()

    def handle(u, lines):
        n = 0
        for l in lines:
            d = json.loads(l)
            t = d.get('text') or ''
            if not (3000 <= len(t) <= 400_000):
                continue
            md = d.get('metadata') or {}
            title = t.strip().split('\n', 1)[0].lstrip('# ').strip()[:300]
            with lock:
                pk.add({'doc_id': f"pmc:{d.get('id')}", 'source': 'pubmed', 'title': title,
                        'licence': md.get('license', ''), 'url': md.get('url', ''), 'text': t[:60000]})
            n += 1
        log('pmc shard', u.rsplit('/', 1)[1], n)
        return n
    got = parallel_shards(urls, max_bytes, handle)
    pk.close(source='pubmed', licence='CC BY / CC BY-SA / CC0 per document (Common Pile PMC OA)',
             provenance={'repo': 'common-pile/pubmed', 'shards': urls, 'prefix_bytes': max_bytes, 'docs': got})


def part_cap(out, n_shards=24, max_bytes=120 << 20, keep=0.6):
    import threading
    rng = random.Random(354)
    shards = sorted(rng.sample(range(173), n_shards))
    urls = [f'https://huggingface.co/datasets/common-pile/caselaw_access_project/resolve/main/cap_{s:05d}.jsonl.gz'
            for s in shards]
    pk, lock = Pack(out, 'cap'), threading.Lock()

    def handle(u, lines):
        r = random.Random(u)
        n = 0
        for l in lines:
            d = json.loads(l)
            t = d.get('text') or ''
            if len(t.split()) < 800 or r.random() > keep:
                continue
            with lock:
                pk.add({'doc_id': f"cap:{d['id']}", 'source': 'case_law', 'title': t.strip().split('\n', 1)[0][:200],
                        'licence': (d.get('metadata') or {}).get('license', 'Public Domain'), 'text': t[:60000]})
            n += 1
        log('cap shard', u.rsplit('/', 1)[1], n)
        return n
    got = parallel_shards(urls, max_bytes, handle)
    pk.close(source='case_law', licence='Public Domain (Common Pile CAP raw)',
             provenance={'repo': 'common-pile/caselaw_access_project', 'shards': urls, 'prefix_bytes': max_bytes,
                         'keep': keep, 'docs': got})


def part_edgar(out, years=('2008', '2012', '2016', '2019', '2020'), max_bytes=250 << 20):
    import threading
    secs = {'section_1': 'Item 1. Business', 'section_1A': 'Item 1A. Risk Factors', 'section_7': "Item 7. MD&A",
            'section_7A': 'Item 7A. Market Risk', 'section_8': 'Item 8. Financial Statements'}
    urls = [f'https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{y}/train.jsonl' for y in years]
    pk, lock = Pack(out, 'edgar'), threading.Lock()

    def handle(u, lines):
        y = u.split('/')[-2]
        n = 0
        for l in lines:
            d = json.loads(l)
            for k, label in secs.items():
                t = d.get(k) or ''
                if len(t) < 3000:
                    continue
                with lock:
                    pk.add({'doc_id': f"edgar:{d.get('filename')}:{k}", 'source': 'sec_filings',
                            'title': f"{d.get('filename')} ({y}) {label}", 'cik': d.get('cik'), 'year': y,
                            'text': t[:60000]})
                n += 1
        log('edgar', y, n)
        return n
    got = parallel_shards(urls, max_bytes, handle, gz=False)
    pk.close(source='sec_filings', licence='Apache-2.0 (EDGAR-CORPUS); filings are US public records',
             provenance={'repo': 'eloukas/edgar-corpus', 'years': years, 'prefix_bytes': max_bytes, 'docs': got})


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for part in sys.argv[2:]:
        log('== part', part)
        try:
            globals()[f'part_{part}'](out)
        except Exception as e:
            log('PART FAILED', part, repr(e))


if __name__ == '__main__':
    main()

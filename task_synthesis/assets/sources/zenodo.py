"""Zenodo records (datasets and publications): tabular, structured, database, scientific, document and text files.

Search is the records API (anonymous: page size <= 25, ~30 requests/min, paced here). A candidate is one record and the
files in it that match the target formats (plain or .gz/.bz2/.xz compressed); when the record ships no matching file
but exactly one archive, the archive is unpacked and filtered at fetch time. Licenses come from metadata.license.id;
access must be open. Records pin by id (each Zenodo version has its own id) and DOI.
"""
import bz2
import gzip
import hashlib
import lzma
import re
import shutil
from pathlib import Path

from ..http import TooLarge
from .base import MAX_FILES, Reject, Source, cap, unpack

API = 'https://zenodo.org/api/records'
PACE_S = 2.1        # the API answers 30 requests per minute to anonymous clients (x-ratelimit-limit: 30)

FORMAT_EXT = {
    'csv': ['.csv'], 'tsv': ['.tsv', '.tab'], 'excel': ['.xlsx', '.xls', '.xlsm'], 'parquet': ['.parquet', '.pq'],
    'feather': ['.feather', '.arrow'], 'json': ['.json', '.geojson'], 'jsonl': ['.jsonl', '.ndjson'],
    'xml': ['.xml'], 'yaml': ['.yaml', '.yml'], 'toml': ['.toml'], 'rdf': ['.ttl', '.rdf', '.nt', '.n3', '.jsonld', '.owl'],
    'sqlite': ['.sqlite', '.sqlite3', '.db'], 'sql-dump': ['.sql'], 'server-db': ['.sql', '.dump'],
    'wal-journal': ['.sqlite', '.sqlite3', '.db'], 'numpy': ['.npy', '.npz'], 'hdf5': ['.h5', '.hdf5'],
    # spectra often ship as 2-column text or CSV
    'spectrum': ['.jdx', '.dx', '.spc', '.msp', '.mgf', '.mzml', '.spe', '.xy', '.txt', '.csv'], 'cif': ['.cif', '.mmcif'],
    'fasta': ['.fasta', '.fa', '.fna', '.faa', '.ffn'], 'genbank': ['.gb', '.gbk', '.genbank', '.gbff'],
    'pdb': ['.pdb', '.ent'], 'molecule': ['.sdf', '.mol', '.mol2', '.smi', '.xyz'], 'ontology': ['.obo', '.owl'],
    'pdf': ['.pdf'], 'office': ['.docx', '.xlsx', '.pptx', '.odt', '.ods', '.doc'], 'markdown': ['.md'],
    'latex': ['.tex'], 'html': ['.html', '.htm'], 'plain-text': ['.txt'], 'log': ['.log', '.txt'],
    'word-list': ['.txt', '.dic'], 'netcdf': ['.nc'],
    # spec vocabulary v2 network-capture formats (#118)
    'pcap': ['.pcap', '.cap'], 'pcapng': ['.pcapng'], 'wifi-80211-capture': ['.pcap', '.pcapng', '.cap'],
    'usbmon-capture': ['.pcap', '.pcapng'], 'har': ['.har'], 'warc': ['.warc'], 'zeek-logs': ['.log'],
    'candump-log': ['.log', '.asc'], 'btsnoop-log': ['.log', '.btsnoop', '.cfa'],
    'netflow-v5': ['.nfcapd', '.csv'], 'ipfix': ['.ipfix', '.csv'], 'nfcapd-files': ['.nfcapd'],
}
CLASS_DEFAULT = {
    'tabular-data': ['csv', 'tsv', 'excel', 'parquet'], 'structured-data': ['json', 'jsonl', 'xml', 'yaml'],
    'database': ['sqlite', 'sql-dump'], 'scientific-data': ['csv', 'numpy', 'fasta', 'cif', 'spectrum', 'pdb'],
    'document': ['pdf', 'office', 'markdown'], 'text-data': ['plain-text', 'log'],
    'network-capture': ['pcap', 'pcapng'],
}
# file_type filter values the API matches well (by extension); rare formats go into the query text instead
FILE_TYPE = {'csv': 'csv', 'excel': 'xlsx', 'json': 'json', 'pdf': 'pdf', 'xml': 'xml', 'tsv': 'tsv',
             'plain-text': 'txt', 'office': 'docx'}
COMPRESSED = ('.gz', '.bz2', '.xz')
ARCHIVES = ('.zip', '.tar.gz', '.tgz', '.tar', '.tar.bz2', '.tar.xz')
SPAM = re.compile(r'\bmarket\b.*\b(19|20)\d\d\b|\bmarket (size|share|report|research|forecast)\b', re.I)
MIN_BYTES = {'tabular-data': 4000, 'database': 4000, 'structured-data': 2000, 'scientific-data': 1000}


def target_exts(target):
    fmts = [f for f in (target.get('formats') or []) if f in FORMAT_EXT] or CLASS_DEFAULT.get(target['cls'], [])
    return fmts, sorted({e for f in fmts for e in FORMAT_EXT[f]})


def base_ext(name, exts):
    """The matching data extension of a file name (compressed variants included), or None."""
    low = name.lower()
    for c in COMPRESSED:
        if low.endswith(c) and not low.endswith('.tar' + c):
            low = low[: -len(c)]
            break
    return next((e for e in exts if low.endswith(e)), None)


def is_archive(name):
    low = name.lower()
    return low.endswith(ARCHIVES)


def select(files, exts, max_bytes, min_bytes):
    """Up to 10 matching files within max_bytes, in record order, skipping single files over the cap and stubs."""
    match = [f for f in files if base_ext(f['key'], exts) and f.get('size', 0) >= min_bytes]
    out, total = [], 0
    for f in match:
        if f['size'] > max_bytes or total + f['size'] > max_bytes:
            continue
        out.append(f)
        total += f['size']
        if len(out) == 10:
            break
    return out, total


def creators(meta, k=3):
    names = [c.get('name', '') for c in meta.get('creators', []) if c.get('name')]
    return ', '.join(names[:k]) + (' et al.' if len(names) > k else '')


def decompress(path, limit):
    """foo.csv.gz -> foo.csv in place, bounded by limit bytes of output."""
    low = path.name.lower()
    opener = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}[next(c for c in COMPRESSED if low.endswith(c))]
    out = path.with_name(path.name[: path.name.rfind('.')])
    n = 0
    with opener(path, 'rb') as src, open(out, 'wb') as dst:
        while chunk := src.read(1 << 20):
            n += len(chunk)
            if n > limit:
                dst.close()
                out.unlink(missing_ok=True)
                raise Reject(f'{path.name} decompresses past {limit} bytes')
            dst.write(chunk)
    path.unlink()
    return out


class Zenodo(Source):
    name = 'zenodo'
    classes = ('tabular-data', 'structured-data', 'database', 'scientific-data', 'document', 'text-data')

    async def _api(self, params):
        await self.http.gate.wait('zenodo-api', PACE_S)
        return await self.http.json(API, params=params)

    async def search(self, target, query, n):
        fmts, exts = target_exts(target)
        if not exts:
            raise Reject(f'zenodo: no file extensions for {target}')
        params = {'q': query, 'size': 25, 'sort': 'bestmatch', 'access_right': 'open'}
        params['type'] = 'publication' if target['cls'] == 'document' and set(fmts) & {'pdf', 'latex'} else 'dataset'
        ft = next((FILE_TYPE[f] for f in fmts if f in FILE_TYPE), None)
        if ft and len(fmts) == 1:
            params['file_type'] = ft
        j = await self._api(params)
        limit, min_b = cap(target), MIN_BYTES.get(target['cls'], 500)
        out = []
        for h in j.get('hits', {}).get('hits', []):
            m = h.get('metadata', {})
            title = m.get('title') or ''
            if m.get('access_right') != 'open' or SPAM.search(title):
                continue
            files = [{'key': f['key'], 'size': f.get('size', 0), 'checksum': f.get('checksum'),
                      'url': f.get('links', {}).get('self')} for f in h.get('files', [])]
            chosen, total = select(files, exts, limit, min_b)
            archive = None
            if not chosen:
                arcs = [f for f in files if is_archive(f['key']) and f['size'] <= limit]
                if len(arcs) == 1 and len([f for f in files if is_archive(f['key'])]) == 1:
                    archive, total = arcs[0], arcs[0]['size']
                else:
                    continue
            desc = re.sub(r'<[^>]+>', ' ', m.get('description') or '')
            out.append({'source': self.name, 'key': str(h['id']), 'url': h.get('links', {}).get('self_html')
                        or f'https://zenodo.org/records/{h["id"]}', 'page_url': f'https://zenodo.org/records/{h["id"]}',
                        'title': title[:200], 'description': re.sub(r'\s+', ' ', desc).strip()[:400],
                        'license_raw': (m.get('license') or {}).get('id'), 'size': total,
                        'formats': sorted({base_ext(f['key'], exts).lstrip('.') for f in chosen}) or ['archive'],
                        'meta': {'files': chosen, 'archive': archive, 'doi': h.get('doi') or m.get('doi'),
                                 'version': m.get('version'), 'creators': creators(m),
                                 'keywords': (m.get('keywords') or [])[:10],
                                 'publication_date': m.get('publication_date'), 'n_files': len(files)},
                        'query': query})
        return out[: n * 3]

    async def fetch(self, cand, staging, target):
        staging = Path(staging)
        meta, limit = cand['meta'], cap(target)
        _, exts = target_exts(target)
        digests, total = [], 0
        try:
            if meta.get('archive'):
                arc = meta['archive']
                tmp = staging.with_name(staging.name + '-' + Path(arc['key']).name)
                d, nb = await self.http.download(arc['url'], tmp, limit)
                digests.append(d)
                total += nb
                raw = staging.with_name(staging.name + '.raw')
                try:
                    unpack(tmp, raw)
                finally:
                    tmp.unlink(missing_ok=True)
                kept = 0
                for p in sorted(raw.rglob('*')):
                    if p.is_file() and base_ext(p.name, exts) and kept < MAX_FILES:
                        rel = p.relative_to(raw)
                        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(staging / rel))
                        kept += 1
                shutil.rmtree(raw, ignore_errors=True)
                if not kept:
                    raise Reject('the archive holds no file of the target formats')
                for p in list(staging.rglob('*')):
                    if p.is_file() and p.name.lower().endswith(COMPRESSED):
                        decompress(p, limit * 4)
            else:
                for f in meta['files']:
                    dest = staging / Path(f['key']).name
                    d, nb = await self.http.download(f['url'], dest, limit - total)
                    digests.append(d)
                    total += nb
                    if dest.name.lower().endswith(COMPRESSED):
                        decompress(dest, limit * 4)
        except TooLarge as e:
            raise Reject(str(e))
        if sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) > limit * 4:
            raise Reject('decompressed asset over the size cap')
        sha = digests[0] if len(digests) == 1 else hashlib.sha256(''.join(digests).encode()).hexdigest()
        rid = cand['key']
        version = f'zenodo:{rid}' + (f' v{str(meta["version"]).lstrip("vV")}' if meta.get('version') else '')
        who = meta.get('creators') or 'Zenodo depositors'
        return {'version': version, 'sha256': sha, 'download_bytes': total,
                'url': f'https://zenodo.org/records/{rid}', 'doi': meta.get('doi'),
                'attribution': f'{who}, "{cand["title"]}", Zenodo, doi:{meta.get("doi")}, {cand["license_raw"]}'}

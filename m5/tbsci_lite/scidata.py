"""The shared real-data cache for science tasks (#353): /data/aai/m5/shared/scidata/ (owner: m5-tbsci_lite).

One-off curation, internet only here (a CPU job through the cluster proxy); agents never fetch anything.

  search    Zenodo records API, per field: each subtopic name of vocab.json is a query (type=dataset, open access).
            A candidate keeps the record's science-format files up to 40 MB in all (single files <= 30 MB).
  fetch     download (pinned: record id, file checksums), unpack single-file .gz/.bz2/.xz, SHA-256 per file.
  block     reject candidates matching lite's assets: any file whose SHA-256 equals a file of a lite task, a record
            whose DOI, Zenodo id or URL a lite task cites, or a title naming a lite task's dataset (lite_refs).
  render    M1's manifest (task_synthesis.assets.manifest) plus readers for NetCDF/HDF5/FITS/NPY: what the task
            author is shown (gen.txt) and the judge (judge.txt).
  fit       DeepSeek (high effort): can this dataset anchor a 2-6 h computational task in the field? modality, format.

Layout: datasets/<id>/{asset.tar.zst, gen.txt, judge.txt, record.json}; index.jsonl (one record per accepted
dataset, with license, URL, DOI, per-file SHA-256, tar SHA-256); rejected.jsonl; lite_refs.json; PROVENANCE.md.

  python -m m5.tbsci_lite.scidata submit [--per-field 40]      # CPU job
  python -m m5.tbsci_lite.scidata run ...                      # inside it (resumable)
  python -m m5.tbsci_lite.scidata stats
"""
import argparse
import asyncio
import bz2
import gzip
import hashlib
import json
import lzma
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from .common import (LITE_TASKS, PROXY_ENV, SHARED_SCIDATA, TS_CHECKOUT, bad_nodes, frozen_checkout, job_pythonpath,
                     requirements_excluding)

ROOT = SHARED_SCIDATA
APY = '/data/aai/venvs/tasksynth-assets/bin/python'
API = 'https://zenodo.org/api/records'
PACE_S = 2.2
MAX_TOTAL, MAX_FILE = 20 << 20, 15 << 20
DATA_EXT = {'.csv', '.tsv', '.tab', '.txt', '.dat', '.json', '.xlsx', '.xls', '.parquet', '.npy', '.npz', '.nc',
            '.nc4', '.h5', '.hdf5', '.hdf', '.fits', '.fit', '.fts', '.fasta', '.fa', '.fna', '.faa', '.fastq', '.gb',
            '.gbk', '.pdb', '.cif', '.mmcif', '.sdf', '.mol', '.mol2', '.xyz', '.vcf', '.gff', '.gff3', '.gtf', '.bed',
            '.nwk', '.newick', '.tre', '.phy', '.aln', '.sto', '.mzml', '.mgf', '.jdx', '.dx', '.spc', '.xy', '.edf',
            '.wav', '.tif', '.tiff', '.png', '.mat', '.sac', '.mseed', '.miniseed', '.las', '.geojson', '.shp',
            '.dbf', '.shx', '.prj', '.gpx', '.kml', '.asc', '.grd', '.obj', '.stl', '.vtk', '.vtu', '.graphml',
            '.gml', '.edgelist', '.mtx', '.lp', '.mps', '.sp', '.cir', '.net', '.inp', '.md', '.readme'}
COMPRESSED = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}
# open licenses (Zenodo metadata.license.id, lowercased); no-derivatives licenses are excluded (tasks subset and convert)
LICENSES = {'cc-by-4.0', 'cc-by-3.0', 'cc-by-2.0', 'cc-by-1.0', 'cc0-1.0', 'cc-zero', 'cc-by-sa-4.0', 'cc-by-sa-3.0',
            'cc-by-nc-4.0', 'cc-by-nc-sa-4.0', 'cc-by-nc-3.0', 'odc-by-1.0', 'odc-by', 'odc-pddl-1.0', 'pddl-1.0',
            'odc-odbl-1.0', 'odbl-1.0', 'mit', 'bsd-3-clause', 'bsd-2-clause', 'apache-2.0', 'gpl-3.0', 'lgpl-3.0',
            'cc-pdm-1.0', 'etalab-2.0', 'ogl-uk-3.0', 'other-open', 'other-at', 'cc-by-sa'}
UA = 'tasksynth-scidata/1.0 (research data curation)'


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def ext_of(name):
    n = name.lower()
    for c in COMPRESSED:
        if n.endswith(c):
            n = n[:-len(c)]
    return Path(n).suffix


# ------------------------------------------------------------------------------------------------ lite references
def lite_refs():
    """SHA-256 of every file in the lite pool tar, plus the DOIs, Zenodo ids and URLs its tasks cite."""
    hashes, dois, zen, urls = set(), set(), set(), set()
    with tarfile.open(LITE_TASKS) as tf:
        for m in tf:
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            h = hashlib.sha256()
            head = b''
            while chunk := f.read(1 << 22):
                h.update(chunk)
                if len(head) < 400_000:
                    head += chunk[:400_000]
            hashes.add(h.hexdigest())
            if m.size < 2_000_000:
                text = head.decode(errors='replace')
                dois |= {d.lower().rstrip('.,);') for d in re.findall(r'10\.\d{4,9}/[^\s"\'<>)\]]+', text)}
                zen |= set(re.findall(r'zenodo\.org/(?:record|records|doi/10\.5281/zenodo\.)/?(\d+)', text))
                zen |= set(re.findall(r'10\.5281/zenodo\.(\d+)', text))
                urls |= {u.rstrip('.,);') for u in re.findall(r'https?://[^\s"\'<>)\]]+', text)}
    return {'file_sha256': sorted(hashes), 'dois': sorted(dois), 'zenodo_ids': sorted(zen),
            'urls': sorted(urls), 'built': time.time()}


# ------------------------------------------------------------------------------------------------ search / fetch
class Zenodo:
    def __init__(self):
        import httpx
        self.c = httpx.Client(timeout=120, follow_redirects=True, headers={'User-Agent': UA},
                              proxy=os.environ.get('https_proxy'))
        self.last = 0.0

    def get(self, url, **params):
        for attempt in range(6):
            wait = PACE_S - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.c.get(url, params=params or None)
            except Exception as e:  # noqa: BLE001
                log(f'GET {url} failed: {type(e).__name__}: {e}')
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(60)
                continue
            if r.status_code >= 500:
                time.sleep(15 * (attempt + 1))
                continue
            return r
        return None

    def search(self, q, size=25):
        r = self.get(API, q=q, type='dataset', size=size, sort='bestmatch', access_right='open')
        if r is None or r.status_code != 200:
            return []
        return r.json().get('hits', {}).get('hits', [])

    def download(self, url, dest, max_bytes):
        for attempt in range(4):
            wait = PACE_S - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                with self.c.stream('GET', url) as r:
                    if r.status_code == 429:
                        time.sleep(60)
                        continue
                    r.raise_for_status()
                    n = 0
                    with open(dest, 'wb') as f:
                        for chunk in r.iter_bytes(1 << 20):
                            n += len(chunk)
                            if n > max_bytes:
                                raise ValueError('too large')
                            f.write(chunk)
                return True
            except ValueError:
                return False
            except Exception as e:  # noqa: BLE001
                log(f'download {url} failed: {type(e).__name__}: {e}')
                time.sleep(10 * (attempt + 1))
        return False


def candidate(hit):
    """A Zenodo hit -> candidate dict, or (None, reason)."""
    md = hit.get('metadata', {})
    lic = ((md.get('license') or {}).get('id') or '').lower()
    if md.get('access_right') != 'open':
        return None, 'not open'
    if lic not in LICENSES:
        return None, f'license {lic or "none"}'
    files = [f for f in hit.get('files') or [] if f.get('size')]
    keep = [f for f in files if ext_of(f['key']) in DATA_EXT and f['size'] <= MAX_FILE]
    data = [f for f in keep if ext_of(f['key']) not in {'.md', '.readme', '.txt'} or f['size'] > 20_000]
    if not data:
        return None, 'no science-format data file <= 30 MB'
    keep.sort(key=lambda f: (ext_of(f['key']) in {'.md', '.readme'}, f['size']))
    chosen, total = [], 0
    for f in keep:
        if total + f["size"] <= MAX_TOTAL and len(chosen) < 15:
            chosen.append(f)
            total += f['size']
    if total < 5_000:
        return None, 'too small'
    desc = re.sub(r'<[^>]+>', ' ', md.get('description') or '')
    desc = re.sub(r'\s+', ' ', desc).strip()
    return {'zenodo_id': str(hit['id']), 'concept': str(hit.get('conceptrecid') or hit['id']),
            'doi': (md.get('doi') or hit.get('doi') or '').lower(), 'title': md.get('title', ''),
            'description': desc[:1500], 'license': lic, 'url': hit.get('links', {}).get('self_html') or
            f'https://zenodo.org/records/{hit["id"]}', 'version': md.get('version'),
            'publication_date': md.get('publication_date'), 'creators': [c.get('name') for c in md.get('creators', [])][:6],
            'keywords': md.get('keywords', [])[:12],
            'files': [{'key': f['key'], 'size': f['size'], 'checksum': f.get('checksum'),
                       'url': (f.get('links') or {}).get('self')} for f in chosen],
            'n_files_record': len(files), 'bytes': total}, None


# ------------------------------------------------------------------------------------------------ render helpers
def describe_binary(p):
    """Short structural summary of NetCDF/HDF5/FITS/NPY/MAT files (what M1's manifest cannot read)."""
    e = ext_of(p.name)
    try:
        if e in ('.nc', '.nc4'):
            import netCDF4
            with netCDF4.Dataset(p) as ds:
                dims = {k: len(v) for k, v in ds.dimensions.items()}
                vs = [f'{k}{tuple(v.dimensions)} {v.dtype} units={getattr(v, "units", "?")}' for k, v in
                      list(ds.variables.items())[:30]]
                attrs = {k: str(ds.getncattr(k))[:120] for k in list(ds.ncattrs())[:12]}
            return f'NetCDF dims={dims}; variables: ' + '; '.join(vs) + f'; global attrs: {attrs}'
        if e in ('.h5', '.hdf5', '.hdf'):
            import h5py
            out = []
            with h5py.File(p, 'r') as f:
                f.visititems(lambda n, o: out.append(f'{n} {o.shape} {o.dtype}') if hasattr(o, 'shape') and
                             len(out) < 40 else None)
            return 'HDF5 datasets: ' + '; '.join(out)
        if e in ('.fits', '.fit', '.fts'):
            from astropy.io import fits
            with fits.open(p, memmap=True) as h:
                parts = []
                for i, hdu in enumerate(h[:8]):
                    hdr = hdu.header
                    keys = {k: str(hdr[k])[:40] for k in list(hdr.keys())[:25] if k and k not in ('COMMENT', 'HISTORY')}
                    shape = getattr(hdu.data, 'shape', None) if hdu.data is not None else None
                    cols = hdu.columns.names[:30] if hasattr(hdu, 'columns') else None
                    parts.append(f'HDU{i} {type(hdu).__name__} shape={shape} cols={cols} header={keys}')
            return 'FITS: ' + ' | '.join(parts)
        if e in ('.npy',):
            import numpy as np
            a = np.load(p, mmap_mode='r', allow_pickle=False)
            return f'NPY shape={a.shape} dtype={a.dtype}'
        if e == '.npz':
            import numpy as np
            with np.load(p, allow_pickle=False) as z:
                return 'NPZ ' + '; '.join(f'{k} {z[k].shape} {z[k].dtype}' for k in list(z.keys())[:30])
    except Exception as ex:  # noqa: BLE001
        return f'(unreadable: {type(ex).__name__}: {str(ex)[:120]})'
    return None


def render_dataset(root, rec):
    sys.path.insert(0, str(TS_CHECKOUT))
    from task_synthesis.assets import manifest as M
    m = M.build(root)
    extra = []
    for p, rel in M.walk(root):
        d = describe_binary(p)
        if d:
            extra.append(f'- {rel}: {d[:1500]}')
    gen = M.render(m, 16000)
    if extra:
        gen += '\n\n## Structure of binary files\n' + '\n'.join(extra[:30])
    head = (f'Source: Zenodo record {rec["zenodo_id"]} (DOI {rec["doi"]}); "{rec["title"]}"; license {rec["license"]}; '
            f'{rec["url"]}\nDescription: {rec["description"][:500]}\n\n')
    judge = head + M.render(m, 9000) + ('\n\n## Structure of binary files\n' + '\n'.join(extra[:15]) if extra else '')
    return m, gen, judge


FIT_SYSTEM = 'You curate real scientific datasets for synthetic computational-science tasks. Answer with JSON only.'
FIT_PROMPT = """A task author will build one realistic computational task (2-6 expert hours: analyse the data, fit or build a model, reconstruct, predict, infer or design, and deliver precisely specified result files) in the field of {field} around this dataset. The task runs offline on 2 CPUs; the author sees the file listing and contents below and writes a setup script that copies or subsets the data.

Judge the dataset:
- fit: true if it is real scientific or engineering measurement/observation/simulation data with enough content and documentation (column meanings, units, or a self-explaining structure) to anchor such a task in this field; false for paper PDFs, code-only records, slide decks, tiny tables, opaque files without any explanation, personal or patient-identifying data, or data unrelated to the field.
- modality: one of {mods}.
- format: a short name of the main data format (e.g. "CSV time series", "NetCDF-4 grid", "FITS image", "FASTA").
- subtopic: a few words naming the scientific setting.

Reply {{"fit": true|false, "reason": "...", "modality": "...", "format": "...", "subtopic": "..."}}.

# Dataset: {title}
{description}

{render}"""


async def fit_check(llm, field_desc, rec, gen, mods):
    from task_synthesis.vocab.opus import parse_json
    text = FIT_PROMPT.format(field=field_desc, mods=', '.join(mods), title=rec['title'],
                             description=rec['description'][:1200], render=gen[:14000])
    for _ in range(3):
        content, _, usage = await llm.chat([{'role': 'system', 'content': FIT_SYSTEM}, {'role': 'user', 'content': text}])
        try:
            j = parse_json(content)
            return {**j, 'usage': usage}
        except ValueError:
            continue
    return {'fit': False, 'reason': 'unparseable fit reply'}


# ------------------------------------------------------------------------------------------------ run
def make_tar(src, out):
    subprocess.run(['tar', '--zstd', '-cf', str(out), '--sort=name', '--owner=0', '--group=0', '--numeric-owner',
                    '--mtime=@1700000000', '-C', str(src), '.'], check=True, capture_output=True)


def to_fast(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    if src.stat().st_size > 1 << 20:
        subprocess.run(['dd', f'if={src}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    else:
        shutil.copyfile(src, tmp)
    tmp.replace(dst)


def done_ids():
    seen = set()
    for name in ('index.jsonl', 'rejected.jsonl'):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                    seen.add(r.get('concept') or r.get('zenodo_id'))
                except ValueError:
                    pass
    return seen


async def run(a):
    sys.path.insert(0, str(TS_CHECKOUT))
    from task_synthesis.pipeline.llm import Endpoint, wait_healthy
    from .vocab import FIELD_DESC, MODALITIES
    vocab = json.loads((Path(__file__).resolve().parent / 'vocab.json').read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    refs_path = ROOT / 'lite_refs.json'
    if not refs_path.exists():
        log('hashing the lite pool for blocking')
        tmp = Path(os.environ.get('TMPDIR', '/tmp')) / 'lite_refs.json'
        tmp.write_text(json.dumps(lite_refs()))
        to_fast(tmp, refs_path)
    refs = json.loads(refs_path.read_text())
    block_hash, block_doi, block_zen = set(refs['file_sha256']), set(refs['dois']), set(refs['zenodo_ids'])
    block_url = set(refs['urls'])
    log(f'lite refs: {len(block_hash)} file hashes, {len(block_doi)} DOIs, {len(block_zen)} Zenodo ids')
    llm_box = []
    llm_lock = asyncio.Lock()

    async def get_llm():
        async with llm_lock:
            if not llm_box:
                await wait_healthy(a.gen)
                llm = Endpoint(a.gen, max_in_flight=a.llm_concurrency)
                if a.effort:   # as round.with_effort
                    kw = {**llm.extra.get('chat_template_kwargs', {}), 'reasoning_effort': a.effort}
                    llm.extra = {**{k: v for k, v in llm.extra.items() if k != 'reasoning_effort'},
                                 'chat_template_kwargs': kw, 'reasoning_effort': a.effort}
                llm_box.append(llm)
        return llm_box[0]
    z = Zenodo()
    seen = done_ids()
    shard = f'{socket.gethostname()}-{os.getpid()}'
    index_tmp = Path(os.environ.get('TMPDIR', '/tmp'))
    rng = random.Random(a.seed)
    fields = a.fields or sorted(FIELD_DESC)
    # datasets per field ~ the field's task share (floor 15), scaled to --total
    share = {f: w * vocab['domains'][d]['weight'] for d, x in vocab['domains'].items() for f, w in x['fields'].items()}
    want = {f: max(a.floor, round(a.total * share.get(f, 0))) for f in fields}
    have = {}
    for line in (ROOT / 'index.jsonl').read_text().splitlines() if (ROOT / 'index.jsonl').exists() else []:
        r = json.loads(line)
        have[r['field']] = have.get(r['field'], 0) + 1
    sem = asyncio.Semaphore(a.llm_concurrency)

    def append(name, rec):
        with open(ROOT / f'{name}.{shard}.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')

    async def judge_and_store(field, sub, cand, stage):
        try:
            m, gen, judge = await asyncio.to_thread(render_dataset, stage, cand)
            llm = await get_llm()
            async with sem:
                fit = await fit_check(llm, FIELD_DESC[field], cand, gen, list(MODALITIES))
            if not fit.get('fit') or fit.get('modality') not in MODALITIES:
                append('rejected', {**cand, 'field': field, 'query': sub, 'reason': f'fit: {fit.get("reason")}'})
                return False
            did = f'zen{cand["zenodo_id"]}'
            out = Path(os.environ.get('TMPDIR', '/tmp')) / f'{did}.tar.zst'
            make_tar(stage, out)
            d = ROOT / 'datasets' / did
            to_fast(out, d / 'asset.tar.zst')
            (d / 'gen.txt').write_text(gen)
            (d / 'judge.txt').write_text(judge)
            rec = {**cand, 'id': did, 'field': field, 'query': sub, 'source': 'zenodo', 'dir': str(d),
                   'modality': fit['modality'], 'format': fit.get('format'), 'subtopic': fit.get('subtopic'),
                   'fit_reason': fit.get('reason'), 'sha256': sha256(out), 'tar_bytes': out.stat().st_size,
                   'n_files': m['n_files'], 'size_bytes': m['total_bytes'], 'fetched_at': time.time()}
            (d / 'record.json').write_text(json.dumps(rec, indent=1))
            append('index', rec)
            out.unlink()
            return True
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    pending = []
    for field in fields:
        subs = [s['name'] for s in vocab['subtopics'][field]]
        rng.shuffle(subs)
        got = have.get(field, 0)
        for sub in subs:
            if got >= want[field]:
                break
            q = re.sub(r'[^A-Za-z0-9 \-]', ' ', sub)
            hits = await asyncio.to_thread(z.search, q)
            log(f'{field} / {sub}: {len(hits)} hits')
            took = 0
            for hit in hits:
                if took >= a.per_query or got >= want[field]:
                    break
                cand, why = candidate(hit)
                key = str(hit.get('conceptrecid') or hit.get('id'))
                if key in seen:
                    continue
                seen.add(key)
                if cand is None:
                    continue
                if (cand['doi'] in block_doi or cand['zenodo_id'] in block_zen or cand['concept'] in block_zen
                        or any(cand['zenodo_id'] in u for u in block_url)):
                    append('rejected', {**cand, 'field': field, 'query': sub, 'reason': 'cited by a lite task'})
                    continue
                stage = Path(tempfile.mkdtemp(prefix='sci-', dir=os.environ.get('TMPDIR')))
                ok, files = True, []
                for f in cand['files']:
                    dest = stage / Path(f['key']).name
                    if not f.get('url') or not await asyncio.to_thread(z.download, f['url'], dest, MAX_FILE + 1):
                        ok = False
                        break
                    if f.get('checksum', '').startswith('md5:'):
                        if hashlib.md5(dest.read_bytes()).hexdigest() != f['checksum'][4:]:
                            ok = False
                            break
                    for c, opener in COMPRESSED.items():
                        if dest.name.lower().endswith(c):
                            plain = dest.with_name(dest.name[:-len(c)])
                            try:
                                with opener(dest, 'rb') as src, open(plain, 'wb') as dst:
                                    n = 0
                                    while chunk := src.read(1 << 20):
                                        n += len(chunk)
                                        if n > 50 << 20:
                                            raise ValueError('too large unpacked')
                                        dst.write(chunk)
                                dest.unlink()
                                dest = plain
                            except (OSError, ValueError, EOFError):
                                ok = False
                            break
                    if not ok:
                        break
                    files.append({'key': f['key'], 'path': dest.name, 'sha256': sha256(dest),
                                  'bytes': dest.stat().st_size})
                if not ok:
                    shutil.rmtree(stage, ignore_errors=True)
                    append('rejected', {**cand, 'field': field, 'query': sub, 'reason': 'download failed'})
                    continue
                if any(x['sha256'] in block_hash for x in files):
                    shutil.rmtree(stage, ignore_errors=True)
                    append('rejected', {**cand, 'field': field, 'query': sub, 'reason': 'file identical to a lite asset'})
                    continue
                cand['files'] = files
                took += 1
                got += 1   # optimistic; the fit check may still reject it
                pending.append(asyncio.create_task(judge_and_store(field, sub, cand, stage)))
                pending = [t for t in pending if not t.done()]
        log(f'{field}: {got} candidates queued (want {want[field]})')
    results = await asyncio.gather(*pending, return_exceptions=True)
    log(f'done; last batch: {sum(1 for r in results if r is True)} stored')


def merge():
    """Concatenate the per-process index/rejected shards into index.jsonl / rejected.jsonl (dedup by id)."""
    for name in ('index', 'rejected'):
        rows, keys = [], set()
        for p in sorted(ROOT.glob(f'{name}.jsonl')) + sorted(ROOT.glob(f'{name}.*.jsonl')):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                k = r.get('id') or (r.get('concept'), r.get('reason'))
                k = json.dumps(k)
                if k not in keys:
                    keys.add(k)
                    rows.append(r)
        tmp = ROOT / f'{name}.jsonl.partial'
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        tmp.replace(ROOT / f'{name}.jsonl')
        for p in ROOT.glob(f'{name}.*.jsonl'):
            p.unlink()
    print('merged')


def usable():
    """{dataset_id: record} of accepted datasets whose files are in place."""
    out = {}
    for p in [ROOT / 'index.jsonl', *sorted(ROOT.glob('index.*.jsonl'))]:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            d = Path(r['dir'])
            if (d / 'asset.tar.zst').exists() and (d / 'gen.txt').exists():
                out[r['id']] = r
    return out


def stats():
    import collections
    u = usable()
    rej = []
    for p in [ROOT / 'rejected.jsonl', *sorted(ROOT.glob('rejected.*.jsonl'))]:
        if p.exists():
            rej += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(json.dumps({'accepted': len(u), 'by_field': collections.Counter(r['field'] for r in u.values()).most_common(),
                      'by_modality': collections.Counter(r['modality'] for r in u.values()).most_common(),
                      'licenses': collections.Counter(r['license'] for r in u.values()).most_common(),
                      'mb': round(sum(r['tar_bytes'] for r in u.values()) / 2**20, 1),
                      'rejected': collections.Counter(re.sub(r':.*', '', r['reason']) for r in rej).most_common()},
                     indent=1))


def submit(a):
    code = frozen_checkout()
    jobs = ROOT / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    args = ' '.join(f'--{k.replace("_", "-")} {v}' for k, v in vars(a).items()
                    if k not in ('op', 'bid', 'fields') and v is not None)
    if a.fields:
        args += ' --fields ' + ' '.join(a.fields)
    env = {'USER': 'user', 'HOME': '/home/user', 'PYTHONNOUSERSITE': '1',
           'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': job_pythonpath(code), **PROXY_ENV,
           'no_proxy': 'localhost,127.0.0.1,.internal.cluster.internal,localnet',
           'NO_PROXY': 'localhost,127.0.0.1,.internal.cluster.internal,localnet'}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = jobs / 'scidata.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $TMPDIR; /home/user/.local/bin/uv pip install -q --python {APY} --target $_CONDOR_SCRATCH_DIR/pylib netCDF4 h5py astropy && export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR/pylib && exec {APY} -u -m m5.tbsci_lite.scidata run {args}'"
initialdir = {TS_CHECKOUT}
output = {jobs}/scidata.$(Cluster).out
error = {jobs}/scidata.$(Cluster).err
log = {jobs}/condor.log
request_cpus = 2
request_memory = 8192
request_disk = 30G
+JobBatchName = "m5-tbsci-scidata"
+MaxRuntime = 86400
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {requirements_excluding(bad_nodes())}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    for name in ('run', 'submit'):
        p = sp.add_parser(name)
        p.add_argument('--gen', default='m5-ds')
        p.add_argument('--total', type=int, default=600, help='datasets to curate over all fields')
        p.add_argument('--floor', type=int, default=15, help='at least this many per field')
        p.add_argument('--per-query', type=int, default=2, help='candidates kept per subtopic query')
        p.add_argument('--llm-concurrency', type=int, default=16)
        p.add_argument('--effort', default='high')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--fields', nargs='*')
        if name == 'submit':
            p.add_argument('--bid', type=int, default=36)
    sp.add_parser('stats')
    sp.add_parser('merge')
    a = ap.parse_args(argv)
    if a.op == 'run':
        asyncio.run(run(a))
    elif a.op == 'submit':
        submit(a)
    elif a.op == 'merge':
        merge()
    else:
        stats()


if __name__ == '__main__':
    main()

"""The Common Crawl PDF lane via pixparse/pdfa-eng-wds: the SafeDocs CC-MAIN-2021-31-PDF-UNTRUNCATED corpus,
filtered to English (XLM-R on page 1), files <100 MB, as webdataset tars on Hugging Face (~840 MB, ~1.2-2.2k PDFs per
shard; each sample is <id>.pdf + <id>.json with per-page words/lines). Sample ids are SafeDocs file ids, so the
SafeDocs provenance table (cc_meta) gives each PDF's source URL.

  python -m m5.gdppdf.docs.pdfa triage --shard 12
      -> /data/aai/m5/shared/docs/cc2021/triage/pdfa-<shard>.jsonl.zst: id, sha256, bytes, n_pages, words,
         words per page, snippet (pages 1-3, collapsed), mid page text. No PDFs kept.
  python -m m5.gdppdf.docs.pdfa extract --shard 12 --selection sel.jsonl --workers 8
      -> texts/pdfa-<shard>.jsonl.zst (LiteParse 2.5.0 + English OCR, exactly the eval's prepare.py config) and
         pdfs/pdfa-<shard>.tar (the selected PDFs, named <doc_id>.pdf)
HF dataset licence: see the dataset card (PDF Association / SafeDocs; documents keep their own copyright; Common
Crawl terms of use). Used for research training data only; not redistributed.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

import zstandard

REPO = 'pixparse/pdfa-eng-wds'
HF = f'https://huggingface.co/datasets/{REPO}/resolve/main'
SHARED = Path('/data/aai/m5/shared/docs/cc2021')
TESSDATA = '/data/aai/aa_evals/data/gdppdf/tessdata_best_e12c65a9'
TESSDATA_SHA256 = '8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba'
MOJIBAKE_RE = re.compile('[ÂÃ][\u0080-¿]|â\u0080[\u0080-¿]|â€')
WS = re.compile(r'[ \t]+')


def shard_name(i):
    return f'pdfa-eng-train-{int(i):04d}.tar'


def download_url(url, dst, parts=8, tries=6):
    """Parallel ranged download (HF serves ~3.5 MB/s per stream through the proxy, S3 ~0.35 MB/s)."""
    sizes = []
    for t in range(tries):
        r = subprocess.run(['curl', '-sSIL', url], capture_output=True, text=True)
        sizes = re.findall(r'(?im)^content-length:\s*(\d+)', r.stdout)
        if sizes:
            break
        time.sleep(20 * (t + 1))
    size = int(sizes[-1])
    step = -(-size // parts)

    def part(k):
        p = Path(f'{dst}.part{k}')
        lo, hi = k * step, min(size, (k + 1) * step) - 1
        for t in range(tries):
            r = subprocess.run(['curl', '-sSfL', '--retry', '3', '-r', f'{lo}-{hi}', '-o', str(p), url],
                               capture_output=True, text=True)
            if r.returncode == 0 and p.stat().st_size == hi - lo + 1:
                return p
            time.sleep(20 * (t + 1))
        raise RuntimeError(f'part {k} of {url} failed: {r.stderr[:200]}')

    t0 = time.time()
    with cf.ThreadPoolExecutor(parts) as ex:
        ps = list(ex.map(part, range(parts)))
    with open(dst, 'wb') as out:
        for p in ps:
            with open(p, 'rb') as f:
                while b := f.read(1 << 24):
                    out.write(b)
            p.unlink()
    print(f'downloaded {url} {size / 1e6:.0f} MB in {time.time() - t0:.0f}s', flush=True)


def download(i, dst, parts=8):
    download_url(f'{HF}/{shard_name(i)}', dst, parts)


def page_text(pg):
    lines = pg.get('lines') or {}
    txt = lines.get('text') if isinstance(lines, dict) else None
    if txt:
        return '\n'.join(txt)
    words = (pg.get('words') or {}).get('text') or []
    return ' '.join(words)


def collapse(t):
    return '\n'.join(WS.sub(' ', l).strip() for l in t.splitlines() if l.strip())


def iter_samples(tar_path):
    """(id, json_bytes, pdf_bytes) per sample (members come in pairs with the same basename)."""
    cur, parts = None, {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            base, _, ext = m.name.rpartition('.')
            if base != cur:
                if cur is not None:
                    yield cur, parts.get('json'), parts.get('pdf')
                cur, parts = base, {}
            parts[ext] = tf.extractfile(m).read()
        if cur is not None:
            yield cur, parts.get('json'), parts.get('pdf')


def write_zst(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()))
    os.rename(tmp, path)


def triage(a):
    out = SHARED / 'triage' / f'pdfa-{int(a.shard):04d}.jsonl.zst'
    if out.exists():
        print(f'{out} exists')
        return
    tar = Path(a.scratch) / shard_name(a.shard)
    download(a.shard, tar)
    rows = []
    for sid, js, pdf in iter_samples(tar):
        rec = {'id': sid, 'shard': int(a.shard), 'bytes': len(pdf or b''),
               'sha256': hashlib.sha256(pdf).hexdigest() if pdf else None}
        try:
            pages = json.loads(js)['pages']
        except (TypeError, ValueError, KeyError):
            rec['error'] = 'bad json'
            rows.append(rec)
            continue
        texts = [page_text(p) for p in pages]
        n = len(texts)
        words = [len(t.split()) for t in texts]
        mid = max(4, (n + 1) // 2) if n > 4 else None
        rec.update(n_pages=n, words=sum(words), empty_pages=sum(w < 5 for w in words),
                   snippet='\n'.join(f'[page {k + 1}]\n{collapse(texts[k])[:2500]}' for k in range(min(3, n)))[:6000],
                   mid_page=mid, mid_text=collapse(texts[mid - 1])[:1500] if mid else '')
        rows.append(rec)
    tar.unlink()
    write_zst(out, rows)
    print(f'wrote {len(rows)} rows to {out}', flush=True)


def extract(a):
    import liteparse
    from importlib.metadata import version
    assert version('liteparse') == '2.5.0', version('liteparse')
    assert hashlib.sha256(open(f'{TESSDATA}/eng.traineddata', 'rb').read()).hexdigest() == TESSDATA_SHA256
    tag = f'pdfa-{int(a.shard):04d}-{Path(a.selection).stem}'   # one shard file per (PDFA shard, selection)
    out = SHARED / 'texts' / f'{tag}.jsonl.zst'
    if out.exists():
        print(f'{out} exists')
        return
    sel = {}
    for l in open(a.selection):
        if l.strip():
            r = json.loads(l)
            if int(r['shard']) == int(a.shard):
                sel[r['id']] = r
    print(f'{tag}: {len(sel)} selected', flush=True)
    tar = Path(a.scratch) / shard_name(a.shard)
    download(a.shard, tar)
    cfg = dict(ocr_enabled=True, ocr_language='eng', tessdata_path=TESSDATA, dpi=150.0, quiet=True,
               num_workers=a.workers, ocr_failure_fatal=True)
    lp = liteparse.LiteParse(**cfg)
    rows, pdfs = [], []
    t0 = time.time()
    for sid, _, pdf in iter_samples(tar):
        if sid not in sel or not pdf:
            continue
        r = sel[sid]
        t1 = time.time()
        try:
            res = lp.parse(pdf)
        except Exception as e:
            rows.append({'doc_id': r['doc_id'], 'id': sid, 'error': f'{type(e).__name__}: {e}'[:300]})
            print(f'{sid}: parse failed {rows[-1]["error"]}', flush=True)
            continue
        pages = sorted(({'page': p.page_num, 'width': p.width, 'height': p.height, 'text': p.text}
                        for p in res.pages), key=lambda p: p['page'])
        rows.append({'doc_id': r['doc_id'], 'id': sid, 'shard': int(a.shard), 'sha256': hashlib.sha256(pdf).hexdigest(),
                     'n_pages': len(pages), 'chars': sum(len(p['text']) for p in pages),
                     'mojibake': sum(len(MOJIBAKE_RE.findall(p['text'])) for p in pages),
                     'parse_seconds': round(time.time() - t1, 2), 'liteparse_config': {k: v for k, v in cfg.items()
                                                                                      if k != 'num_workers'},
                     'pages': pages})
        pdfs.append((r['doc_id'] + '.pdf', pdf))
        print(f'{sid} {len(pages)}p {rows[-1]["chars"]} chars {rows[-1]["parse_seconds"]}s', flush=True)
    tar.unlink()
    tpath = SHARED / 'pdfs' / f'{tag}.tar'
    tpath.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tpath) + '.tmp', 'w') as tf:
        for name, data in pdfs:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    os.rename(str(tpath) + '.tmp', tpath)
    write_zst(out, rows)
    print(f'done {len(rows)} docs in {time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['triage', 'extract'])
    ap.add_argument('--shard', required=True)
    ap.add_argument('--selection', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--scratch', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    triage(a) if a.cmd == 'triage' else extract(a)


if __name__ == '__main__':
    main()

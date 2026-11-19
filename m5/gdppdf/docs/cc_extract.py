"""Full LiteParse 2.5.0 + English OCR extraction of the selected CC PDFs of one zip, exactly as the GDP.pdf eval
prepares its documents (aa_evals/gdppdf/prepare.py: ocr_enabled, eng, pinned tessdata_best e12c65a9, dpi 150).

  python -m m5.gdppdf.docs.cc_extract --zip 0123 --selection <selection.jsonl> --workers 4

Writes /data/aai/m5/shared/docs/cc2021/texts/<zip>.jsonl.zst (one line per document: doc_id, file_name,
sha256, n_pages, chars, mojibake, pages [{page, width, height, text}]) and pdfs/<zip>.tar (the PDF files).
"""
import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from pathlib import Path

import zstandard

from m5.gdppdf.docs.cc_triage import SHARED, TESSDATA, fetch, zip_url

TESSDATA_SHA256 = '8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba'
MOJIBAKE_RE = re.compile('[ÂÃ][\u0080-¿]|â\u0080[\u0080-¿]|â€')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--selection', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--scratch', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    import liteparse
    from importlib.metadata import version
    assert version('liteparse') == '2.5.0', version('liteparse')
    assert hashlib.sha256(open(f'{TESSDATA}/eng.traineddata', 'rb').read()).hexdigest() == TESSDATA_SHA256
    z = f'{int(a.zip):04d}'
    out = SHARED / 'texts' / f'{z}.jsonl.zst'
    if out.exists():
        print(f'{out} exists', flush=True)
        return
    sel = [json.loads(l) for l in open(a.selection) if l.strip()]
    sel = {r['file_name']: r for r in sel if r['zip'] == z}
    print(f'zip {z}: {len(sel)} selected', flush=True)
    zp = Path(a.scratch) / f'{z}.zip'
    t0 = time.time()
    fetch(zip_url(z), zp)
    cfg = dict(ocr_enabled=True, ocr_language='eng', tessdata_path=TESSDATA, dpi=150.0, quiet=True,
               num_workers=a.workers, ocr_failure_fatal=True)
    lp = liteparse.LiteParse(**cfg)
    rows, pdfs = [], []
    with zipfile.ZipFile(zp) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for fn, r in sel.items():
            data = zf.read(names[fn])
            t1 = time.time()
            try:
                res = lp.parse(data)
            except Exception as e:
                print(f'{fn}: parse failed {type(e).__name__}: {e}'[:300], flush=True)
                rows.append({'doc_id': r['doc_id'], 'file_name': fn, 'error': f'{type(e).__name__}: {e}'[:300]})
                continue
            pages = sorted(({'page': p.page_num, 'width': p.width, 'height': p.height, 'text': p.text}
                            for p in res.pages), key=lambda p: p['page'])
            rows.append({'doc_id': r['doc_id'], 'file_name': fn, 'zip': z, 'sha256': hashlib.sha256(data).hexdigest(),
                         'n_pages': len(pages), 'chars': sum(len(p['text']) for p in pages),
                         'mojibake': sum(len(MOJIBAKE_RE.findall(p['text'])) for p in pages),
                         'parse_seconds': round(time.time() - t1, 2), 'pages': pages})
            pdfs.append((r['doc_id'] + '.pdf', data))
            print(f'{fn} {len(pages)}p {rows[-1]["chars"]} chars {rows[-1]["parse_seconds"]}s', flush=True)
    zp.unlink()
    (SHARED / 'pdfs').mkdir(parents=True, exist_ok=True)
    tpath = SHARED / 'pdfs' / f'{z}.tar'
    with tarfile.open(str(tpath) + '.tmp', 'w') as tf:
        for name, data in pdfs:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    os.rename(str(tpath) + '.tmp', tpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + '.tmp', 'wb') as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()))
    os.rename(str(out) + '.tmp', out)
    print(f'done {len(rows)} docs in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

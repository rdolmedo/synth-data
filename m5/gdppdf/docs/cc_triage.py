"""Triage one CC-MAIN-2021-31-PDF zip: a short text snippet of every candidate PDF, for domain classification.

  python -m m5.gdppdf.docs.cc_triage --zip 0123

Downloads the zip to job scratch, and for each candidate in candidates.parquet (cc_meta) extracts pages 1-3 and the
middle page with LiteParse 2.5.0 (no OCR; textless candidates get OCR on pages 1-2). Writes
/data/aai/m5/shared/docs/cc2021/triage/<zip>.jsonl.zst (one line per candidate: file_name, sha256, n_pages,
snippet, mid_page, mid) and deletes the zip. No PDFs are kept here; the selected ones are fetched again by
cc_extract.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd
import zstandard

BASE = 'https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles'
SHARED = Path('/data/aai/m5/shared/docs/cc2021')
TESSDATA = '/data/aai/aa_evals/data/gdppdf/tessdata_best_e12c65a9'
WS = re.compile(r'[ \t]+')


def zip_url(z):
    lo = int(z) // 1000 * 1000
    return f'{BASE}/{lo:04d}-{lo + 999:04d}/{z}.zip'


def collapse(t):
    return '\n'.join(WS.sub(' ', l).strip() for l in t.splitlines() if l.strip())


def fetch(url, dst, tries=6):
    for i in range(tries):
        r = subprocess.run(['curl', '-sSfL', '--retry', '3', '-o', str(dst), url], capture_output=True, text=True)
        if r.returncode == 0:
            return
        print(f'fetch {url} failed ({r.stderr.strip()[:200]}), retry {i + 1}', flush=True)
        time.sleep(30 * (i + 1))
    raise RuntimeError(f'cannot fetch {url}')


def snippet(data, n_pages, textless):
    from liteparse import LiteParse
    mid = max(4, (n_pages + 1) // 2) if n_pages > 4 else None
    target = '1-3' + (f',{mid}' if mid else '')
    if textless:
        lp = LiteParse(ocr_enabled=True, ocr_language='eng', tessdata_path=TESSDATA, target_pages='1-2', quiet=True,
                       num_workers=1, ocr_failure_fatal=False)
    else:
        lp = LiteParse(ocr_enabled=False, target_pages=target, quiet=True, num_workers=1)
    res = lp.parse(data)
    pages = {p.page_num: p.text for p in res.pages}
    head = '\n'.join(f'[page {k}]\n{collapse(pages[k])[:2500]}' for k in sorted(pages) if k <= 3)
    return head[:6000], (collapse(pages.get(mid, ''))[:1500] if mid else ''), mid, len(res.pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--scratch', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    z = f'{int(a.zip):04d}'
    out = SHARED / 'triage' / f'{z}.jsonl.zst'
    if out.exists():
        print(f'{out} exists', flush=True)
        return
    cand = pd.read_parquet(SHARED / 'meta' / 'candidates.parquet', filters=[('zip', '==', z)])
    print(f'zip {z}: {len(cand)} candidates', flush=True)
    zp = Path(a.scratch) / f'{z}.zip'
    t0 = time.time()
    fetch(zip_url(z), zp)
    print(f'downloaded {zp.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s', flush=True)
    rows = []
    with zipfile.ZipFile(zp) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for r in cand.to_dict('records'):
            n = names.get(r['file_name'])
            if not n:
                continue
            data = zf.read(n)
            rec = {'file_name': r['file_name'], 'zip': z, 'sha256': hashlib.sha256(data).hexdigest(),
                   'bytes': len(data), 'pages_meta': r['pages'], 'url': r['url'], 'host': r['host'],
                   'textless': bool(r['textless'])}
            t1 = time.time()
            try:
                rec['snippet'], rec['mid_text'], rec['mid_page'], rec['n_parsed'] = snippet(
                    data, int(r['pages']), bool(r['textless']))
            except Exception as e:   # broken PDFs: recorded, never selected
                rec['error'] = f'{type(e).__name__}: {e}'[:300]
            rec['secs'] = round(time.time() - t1, 2)
            rows.append(rec)
    zp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()))
    tmp.rename(out)
    print(f'wrote {len(rows)} rows ({sum("error" in r for r in rows)} errors) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

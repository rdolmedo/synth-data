"""Extract every PDF's per-page text with LiteParse 2.5.0 and English OCR, as AA does.

AA: "Source PDFs are prepared with LiteParse 2.5.0, with English OCR enabled. Every model receives the full
extracted text of every page." LiteParse's Python binding runs Tesseract (tesseract-rs) with
tessdata_best `eng`; we pin that file instead of letting LiteParse download it at run time.

Output (one file for all 100 documents; the /data file quota is tight):
  <DATA_ROOT>/prepared/<PREP_TAG>/texts.jsonl   one line per PDF: task_id, pdf sha256, pages [{page, width,
                                                height, text}], mojibake counts
  <DATA_ROOT>/prepared/<PREP_TAG>/manifest.json versions, config, totals

Heavy (OCR over 4,592 pages): run it as a condor job (`python -m aa_evals.gdppdf.submit prepare`).
Resumable: documents already in texts.jsonl are skipped.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from aa_evals.gdppdf import data

LITEPARSE_VERSION = '2.5.0'
TESSDATA = data.DATA_ROOT / 'tessdata_best_e12c65a9'   # tesseract-ocr/tessdata_best@e12c65a9, eng.traineddata
TESSDATA_SHA256 = '8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba'
PREP_TAG = f'liteparse-{LITEPARSE_VERSION}_eng_tessbest-e12c65a9'
PREP_DIR = data.DATA_ROOT / 'prepared' / PREP_TAG
TEXTS = PREP_DIR / 'texts.jsonl'

# UTF-8 bytes decoded as Latin-1/cp1252 ("Â£", "â€”", "Ã©"): counted per document so the README can say how
# much of the extracted text carries it. We do not repair it; AA's text presumably has the same artifacts.
MOJIBAKE_RE = re.compile('[ÂÃ][\u0080-¿]|â\u0080[\u0080-¿]|â€')


def load_texts(path=TEXTS):
    """{task_id: record} for every prepared document."""
    out = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec['task_id']] = rec
    return out


def parse_one(lp, task):
    t0 = time.time()
    res = lp.parse(str(task.pdf_path))
    pages = [{'page': p.page_num, 'width': p.width, 'height': p.height, 'text': p.text} for p in res.pages]
    pages.sort(key=lambda p: p['page'])
    return {'task_id': task.task_id, 'pdf': task.pdf_path.name, 'pdf_sha256': data.sha256(task.pdf_path),
            'n_pages': len(pages), 'chars': sum(len(p['text']) for p in pages),
            'mojibake': sum(len(MOJIBAKE_RE.findall(p['text'])) for p in pages),
            'parse_seconds': round(time.time() - t0, 2), 'pages': pages}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='LiteParse OCR workers')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)

    import liteparse
    from importlib.metadata import version
    if version('liteparse') != LITEPARSE_VERSION:
        sys.exit(f'liteparse {version("liteparse")} installed, need {LITEPARSE_VERSION}')
    if data.sha256(TESSDATA / 'eng.traineddata') != TESSDATA_SHA256:
        sys.exit(f'{TESSDATA}/eng.traineddata does not match the pinned sha256')

    tasks = data.load_tasks()
    if a.limit:
        tasks = tasks[:a.limit]
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    done = load_texts()
    cfg = dict(ocr_enabled=True, ocr_language='eng', tessdata_path=str(TESSDATA), dpi=150.0, quiet=True,
               num_workers=a.workers, ocr_failure_fatal=True)
    lp = liteparse.LiteParse(**cfg)
    print(f'[prepare] {len(tasks)} tasks, {len(done)} done, config {lp.get_config()}', flush=True)
    t0 = time.time()
    with open(TEXTS, 'a') as out:
        for i, task in enumerate(tasks):
            if task.task_id in done:
                continue
            rec = parse_one(lp, task)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            done[task.task_id] = rec
            print(f'[prepare] {i + 1}/{len(tasks)} {task.task_id} pages={rec["n_pages"]} chars={rec["chars"]} '
                  f'mojibake={rec["mojibake"]} {rec["parse_seconds"]}s (elapsed {time.time() - t0:.0f}s)', flush=True)

    recs = [done[t.task_id] for t in tasks if t.task_id in done]
    manifest = {'prep_tag': PREP_TAG, 'liteparse': LITEPARSE_VERSION, 'tessdata': str(TESSDATA),
                'tessdata_sha256': TESSDATA_SHA256, 'config': {k: v for k, v in cfg.items() if k != 'num_workers'},
                'dataset': data.dataset_info(), 'documents': len(recs),
                'pages': sum(r['n_pages'] for r in recs), 'chars': sum(r['chars'] for r in recs),
                'mojibake': sum(r['mojibake'] for r in recs),
                'docs_with_mojibake': sum(1 for r in recs if r['mojibake']),
                'empty_pages': sum(1 for r in recs for p in r['pages'] if not p['text'].strip()),
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (PREP_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('[prepare] manifest', json.dumps(manifest), flush=True)
    if not a.limit and manifest['pages'] != data.EXPECTED['pages']:
        sys.exit(f'page total {manifest["pages"]} != AA\'s {data.EXPECTED["pages"]}')


if __name__ == '__main__':
    main()

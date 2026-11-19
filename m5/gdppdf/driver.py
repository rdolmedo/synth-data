"""Production driver: keep the generator fed as document slices finish extracting.

  python -m m5.gdppdf.driver --selections sel1.jsonl sel2.jsonl [sel3.jsonl] --exclude <docs files> \
      --out /data/aai/m5/gdppdf/production --concurrency 100

Loop (every --every s): rebuild the shared index over all selections and extracted texts; run document-level
decontamination on documents not checked before (appending to doc_flags.jsonl); write docs_all.jsonl (usable,
unflagged, not too long, not used by prototypes/pilots, in a fixed shuffled order); run the generator over it (it
skips documents already done). Stops when <out>/STOP exists (the trace runner reached its token target, or by hand)
or when every selected document is extracted and generated.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

DOCS = Path('/data/aai/m5/shared/docs')


def run(cmd):
    print('[driver] $', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print(f'[driver] exit {r.returncode}', flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selections', nargs='+', required=True)
    ap.add_argument('--exclude', nargs='*', default=[], help='docs files whose documents are never used')
    ap.add_argument('--out', required=True)
    ap.add_argument('--concurrency', type=int, default=100)
    ap.add_argument('--every', type=int, default=1200)
    ap.add_argument('--tag', default='drv')
    a = ap.parse_args()
    out = Path(a.out)
    py = sys.executable
    checked_path = DOCS / 'doc_flags_checked.txt'
    excl = {json.loads(l)['doc_id'] for f in a.exclude for l in open(f) if l.strip()}
    while not (out / 'STOP').exists():
        t0 = time.time()
        run([py, '-u', '-m', 'm5.gdppdf.docs.build_index', '--selection', *a.selections,
             '--texts', str(DOCS / 'cc2021' / 'texts' / 'pdfa-*-sel*.jsonl.zst')])
        rows = [json.loads(l) for l in open(DOCS / 'index.jsonl')]
        checked = set(checked_path.read_text().split()) if checked_path.exists() else set()
        new = [r for r in rows if r.get('texts') and r['doc_id'] not in checked]
        if new:
            tmp_idx = out / 'index_new.jsonl'
            tmp_idx.write_text(''.join(json.dumps(r) + '\n' for r in new))
            tmp_flags = out / 'doc_flags_new.jsonl'
            if run([py, '-u', '-m', 'm5.gdppdf.decontam.docs', '--index', str(tmp_idx), '--out', str(tmp_flags)]) == 0:
                with open(DOCS / 'doc_flags.jsonl', 'a') as f:
                    f.write(tmp_flags.read_text())
                with open(checked_path, 'a') as f:
                    f.write(''.join(r['doc_id'] + '\n' for r in new))
                checked |= {r['doc_id'] for r in new}
        flagged = {json.loads(l)['doc_id'] for l in open(DOCS / 'doc_flags.jsonl') if json.loads(l).get('flag')}
        ok = [r for r in rows if r.get('texts') and not r.get('too_long') and not r.get('canary')
              and r['doc_id'] in checked and r['doc_id'] not in flagged and r['doc_id'] not in excl]
        ok.sort(key=lambda r: r['doc_id'])
        random.Random(357).shuffle(ok)
        (out / 'docs_all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ok))
        print(f'[driver] {len(rows)} indexed, {len(ok)} usable for generation', flush=True)
        run([py, '-u', '-m', 'm5.gdppdf.gen.generate', '--docs', str(out / 'docs_all.jsonl'), '--out', str(out),
             '--concurrency', str(a.concurrency), '--tag', a.tag])
        selected = sum(1 for s in a.selections for l in open(s) if l.strip())
        extracted = sum(1 for r in rows if r.get('texts') or r.get('extract_error'))
        if extracted >= selected * 0.995 and not new:
            print('[driver] all selected documents extracted and generated', flush=True)
            break
        time.sleep(max(0, a.every - (time.time() - t0)))
    print('[driver] done', flush=True)


if __name__ == '__main__':
    main()

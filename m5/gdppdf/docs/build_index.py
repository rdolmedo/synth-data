"""Build /data/aai/m5/shared/docs/index.jsonl from selections and extracted text shards.

  python -m m5.gdppdf.docs.build_index --selection sel1.jsonl [sel2.jsonl ...]

For each extracted document: Super tokens of its rendered page text (the eval's render_text with an empty task, so
prompt tokens = these + the task's + 512 margin), chars, mojibake, canary check, and the selection metadata.
Documents whose text exceeds MAX_TOKENS are kept in the index with too_long=true (never used for problems).
"""
import argparse
import glob
import json
import os
from pathlib import Path

from aa_evals.gdppdf import inputs
from m5.gdppdf.docs.store import DOCS, read_zst_jsonl

SUPER_TOKENIZER = '/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tokenizer.json'
CANARY = '329d96c2-986f-486e-81db-4b6bd8539ead'
MAX_TOKENS = 92000   # 131072 window - 32768 reserve - 512 margin - ~5.8k for the longest task prompts
LICENCE = {'pdfa-eng-wds': 'Common Crawl CC-MAIN-2021-31 PDF (SafeDocs, via pixparse/pdfa-eng-wds); publisher '
                           'copyright, Common Crawl terms of use; research training input, not redistributed'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', nargs='+', required=True)
    ap.add_argument('--texts', default=str(DOCS / 'cc2021' / 'texts' / '*.jsonl.zst'))
    ap.add_argument('--out', default=str(DOCS / 'index.jsonl'))
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(SUPER_TOKENIZER)
    sel = {}
    for p in a.selection:
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                sel[r['doc_id']] = r
    rows = []
    for shard in sorted(glob.glob(a.texts)):
        for d in read_zst_jsonl(shard):
            if d['doc_id'] not in sel:
                continue
            m = dict(sel[d['doc_id']])
            if 'error' in d:
                m.update(texts=None, extract_error=d['error'])
                rows.append(m)
                continue
            pages = [p['text'] for p in d['pages']]
            text = inputs.render_text('', pages, 'x.pdf', None)
            n_tok = len(tok.encode(text, add_special_tokens=False).ids)
            m.update(texts=shard, pdf_tar=shard.replace('/texts/', '/pdfs/').replace('.jsonl.zst', '.tar'),
                     n_pages=d['n_pages'], chars=d['chars'], mojibake=d['mojibake'], super_tokens=n_tok,
                     too_long=n_tok > MAX_TOKENS, canary=CANARY in text, empty_pages=sum(not t.strip() for t in pages),
                     licence=LICENCE.get(m.get('source'), m.get('licence')))
            rows.append(m)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, a.out)
    ok = [r for r in rows if r.get('texts') and not r['too_long'] and not r['canary']]
    print(json.dumps({'indexed': len(rows), 'usable': len(ok), 'too_long': sum(bool(r.get('too_long')) for r in rows),
                      'errors': sum('extract_error' in r for r in rows), 'canary': sum(bool(r.get('canary')) for r in rows),
                      'super_tokens_mean': round(sum(r['super_tokens'] for r in ok) / max(1, len(ok)))}), flush=True)


if __name__ == '__main__':
    main()

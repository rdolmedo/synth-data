"""The real lane's document pool, selected from m5-gdppdf's shared cache (/data/aai/m5/shared/docs, read in
place; only the selected docs' text is kept here, the PDFs stay in the cache's tars).

  python -m m5.gdpval.docpool build --out POOL.jsonl     (CPU job: decompresses the text shards with zstd)

Kept: public reference material (doc types below; no research articles, no business records: private-style
material is the synthetic lane's), 2-40 pages, 1.5k-25k Super tokens, text present, not too long, no canary, not
flagged by gdppdf's GDP.pdf decontamination. GDPval hash and canary exclusions are applied again by DocPicker.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DOCS = Path('/data/aai/m5/shared/docs')
TYPES = {'guideline_standard', 'form_instructions', 'product_manual', 'financial_report', 'technical_report',
         'handbook_policy', 'contract_policy', 'data_tables'}


def read_zst_jsonl(path):
    data = subprocess.run(['zstd', '-dc', str(path)], capture_output=True, check=True).stdout
    return [json.loads(x) for x in data.decode().split('\n') if x.strip()]


def build(a):
    flagged = set()
    fp = DOCS / 'doc_flags.jsonl'
    if fp.exists():
        for ln in fp.read_text().splitlines():
            x = json.loads(ln)
            if x.get('flag'):
                flagged.add(x['doc_id'])
    rows = [json.loads(x) for x in (DOCS / 'index.jsonl').read_text().splitlines() if x.strip()]
    keep = [r for r in rows if r.get('texts') and not r.get('too_long') and not r.get('canary')
            and r['doc_id'] not in flagged and r.get('doc_type') in TYPES and 2 <= (r.get('n_pages') or 0) <= 40
            and 1500 <= (r.get('super_tokens') or 0) <= 25000 and r.get('pdf_tar')]
    by_shard = {}
    for r in keep:
        by_shard.setdefault(r['texts'], []).append(r)
    n = 0
    with open(a.out + '.partial', 'w') as f:
        for shard, rs in by_shard.items():
            texts = {x['doc_id']: x for x in read_zst_jsonl(shard) if 'pages' in x}
            for r in rs:
                t = texts.get(r['doc_id'])
                if not t:
                    continue
                text = '\n\n'.join(f'[page {p["page"]}]\n{p["text"]}' for p in t['pages'])
                f.write(json.dumps({k: r.get(k) for k in ('doc_id', 'source', 'id', 'domain', 'doc_type', 'title',
                                                          'issuer', 'series', 'year', 'summary', 'n_pages',
                                                          'super_tokens', 'sha256', 'pdf_tar', 'licence')}
                                   | {'pdf_member': f'{r["doc_id"]}.pdf', 'text': text}) + '\n')
                n += 1
    Path(a.out + '.partial').replace(a.out)
    print(f'{len(rows)} indexed, {len(keep)} selected, {n} written -> {a.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build'])
    ap.add_argument('--out', required=True)
    build(ap.parse_args())


if __name__ == '__main__':
    main()

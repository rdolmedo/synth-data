"""Document-level decontamination of the cache against the 100 GDP.pdf test documents (before any generation).

  python -m m5.gdppdf.decontam.docs --index <index.jsonl> --out <doc_flags.jsonl>

Flags a cache document when:
- its PDF sha256 equals a test PDF's;
- the canary GUID appears in its text;
- whole-document MinHash Jaccard (5-word shingles) with a test document >= 0.2;
- a page of >= 50 words has Jaccard >= 0.5 with a test page (LSH candidates, exact Jaccard on shingle sets), on two
  or more pages, or on one page of >= 200 words (one short shared boilerplate page, e.g. a standard federal
  disclaimer, is reported but not flagged).
Writes one line per flagged or reported document: doc_id, reasons, best matches (test task id, page, Jaccard).
"""
import argparse
import collections
import hashlib
import json
import re

import numpy as np

from m5.gdppdf.docs.store import DocStore

CANARY = '329d96c2-986f-486e-81db-4b6bd8539ead'
NPERM = 64
BANDS, ROWS = 16, 4
_P = (1 << 61) - 1
_rs = np.random.RandomState(35713)
_A = _rs.randint(1, 1 << 30, size=NPERM).astype('uint64')
_B = _rs.randint(0, 1 << 30, size=NPERM).astype('uint64')


def shingles(t, k=5):
    ws = re.findall(r'[a-z0-9]+', t.lower())
    return {' '.join(ws[i:i + k]) for i in range(max(0, len(ws) - k + 1))}, len(ws)


def sig(sh):
    h = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), 'little') & ((1 << 30) - 1)
                     for s in sh), dtype='uint64', count=len(sh))
    return ((np.outer(h, _A) + _B) % np.uint64(_P)).min(axis=0)


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from aa_evals.gdppdf import prepare
    test = prepare.load_texts()
    test_sha = {r['pdf_sha256']: tid for tid, r in test.items()}
    tdoc, tpage, bands = {}, {}, collections.defaultdict(list)
    for tid, r in test.items():
        sh, _ = shingles(' '.join(p['text'] for p in r['pages']))
        tdoc[tid] = sh
        for p in r['pages']:
            s, n = shingles(p['text'])
            if n >= 50 and s:
                tpage[(tid, p['page'])] = s
                g = sig(s)
                for b in range(BANDS):
                    bands[(b, g[b * ROWS:(b + 1) * ROWS].tobytes())].append((tid, p['page']))
    tdoc_sig = {tid: sig(s) for tid, s in tdoc.items() if s}
    print(f'test: {len(tdoc)} docs, {len(tpage)} pages indexed', flush=True)
    store = DocStore(a.index)
    out = open(a.out, 'w')
    n = flagged = 0
    for doc_id, m in store.meta.items():
        if not m.get('texts'):
            continue
        d = store.get(doc_id)
        n += 1
        reasons, matches = [], []
        if d.get('sha256') in test_sha:
            reasons.append(f'sha256={test_sha[d["sha256"]]}')
        text = ' '.join(p['text'] for p in d['pages'])
        if CANARY in text:
            reasons.append('canary')
        dsh, _ = shingles(text)
        if dsh:
            g = sig(dsh)
            best = max(((float((g == s).mean()), tid) for tid, s in tdoc_sig.items()), default=(0, None))
            if best[0] >= 0.1:
                j = jac(dsh, tdoc[best[1]])
                matches.append({'kind': 'doc', 'test': best[1], 'jaccard': round(j, 3)})
                if j >= 0.2:
                    reasons.append(f'doc_jaccard={j:.2f}:{best[1]}')
        page_hits = []
        for p in d['pages']:
            s, nw = shingles(p['text'])
            if nw < 50 or not s:
                continue
            g = sig(s)
            cands = {c for b in range(BANDS) for c in bands.get((b, g[b * ROWS:(b + 1) * ROWS].tobytes()), [])}
            for c in cands:
                j = jac(s, tpage[c])
                if j >= 0.5:
                    page_hits.append({'page': p['page'], 'words': nw, 'test': c[0], 'test_page': c[1],
                                      'jaccard': round(j, 3)})
        if page_hits:
            matches += page_hits[:10]
            pages = {h['page'] for h in page_hits}
            if len(pages) >= 2 or any(h['words'] >= 200 for h in page_hits):
                reasons.append(f'pages={len(pages)}')
        if reasons or matches:
            out.write(json.dumps({'doc_id': doc_id, 'flag': bool(reasons), 'reasons': reasons, 'matches': matches}) + '\n')
            flagged += bool(reasons)
    out.close()
    print(json.dumps({'docs_checked': n, 'flagged': flagged}), flush=True)


if __name__ == '__main__':
    main()

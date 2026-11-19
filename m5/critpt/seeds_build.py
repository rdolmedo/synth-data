"""Seed papers per subtopic, with test-derived seed blocking (a CPU job; never on gpu123).

  python -m m5.critpt.seeds_build --taxonomy taxonomy.json --arxiv /data/aai/m5/shared/arxiv --out DIR [--top 200]

1. Physics papers of the shared proof-pile-2 arXiv index whose categories meet a field's categories, 15k-250k chars.
2. TF-IDF (title + abstract, 1-2 grams); for every subtopic, the top-N papers within its field's categories for the
   query name + scope + keywords -> candidates.json {subtopic_id: [arxiv_id, ...]} (ranked).
3. Blocking (decontamination of seeds): papers a CritPt test challenge names by reference (an explicit list kept with
   the data, NAMED_FILE, not in the repo), and for every test challenge its 25 nearest physics papers by TF-IDF of the
   challenge's description against title + abstract (the papers it is most plausibly built on) -> blocked.json.
   Blocked papers never enter candidates.
4. The LaTeX bodies of all candidates, streamed from the shards -> seeds.jsonl.zst (arxiv_id, title, categories,
   abstract, n_chars, n_math, text).
"""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zstandard

CRITPT_PARQUET = '/data/aai/aa_evals/data/critpt/hf_5b60585/data/train-00000-of-00001.parquet'
NAMED_FILE = Path('/data/aai/m5/critpt/seeds/blocked_named.json')   # {arxiv_id: why}; test-derived, kept off the repo
MATH = re.compile(r'\\begin\{(equation|align|eqnarray|gather|multline)|\$\$|\\\[')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taxonomy', required=True)
    ap.add_argument('--arxiv', default='/data/aai/m5/shared/arxiv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--block-k', type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tax = json.loads(Path(a.taxonomy).read_text())
    arx = Path(a.arxiv)

    cats_all = sorted({c for f in tax['fields'].values() for c in f['categories']})
    idx = pq.read_table(arx / 'index.parquet', columns=['arxiv_id', 'split', 'shard', 'line', 'n_chars', 'title',
                                                        'categories', 'abstract', 'matched']).to_pandas()
    idx = idx[idx.matched & idx.n_chars.between(15000, 250000)].drop_duplicates('arxiv_id')
    cset = idx.categories.fillna('').str.split()
    keep = cset.apply(lambda cs: any(c in cats_all or c.split('.')[0] in cats_all for c in cs))
    idx = idx[keep].reset_index(drop=True)
    cset = idx.categories.str.split()
    print('physics papers', len(idx), flush=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(stop_words='english', sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_df=0.3,
                          max_features=600000, dtype=np.float32)
    docs = (idx.title.fillna('') + '. ' + idx.title.fillna('') + '. ' + idx.abstract.fillna('')).str.replace('\n', ' ')
    X = normalize(vec.fit_transform(docs))
    print('tfidf', X.shape, flush=True)

    # blocking
    test = pq.read_table(CRITPT_PARQUET, columns=['problem_id', 'problem_description']).to_pylist()
    T = normalize(vec.transform([t['problem_description'] for t in test]))
    S = (T @ X.T).toarray()
    blocked = json.loads(NAMED_FILE.read_text())
    for i, t in enumerate(test):
        for j in np.argsort(-S[i])[:a.block_k]:
            blocked.setdefault(idx.arxiv_id[j], f'nearest-{a.block_k} to {t["problem_id"]} (cos {S[i, j]:.3f})')
    (out / 'blocked.json').write_text(json.dumps(blocked, indent=0))
    print('blocked', len(blocked), flush=True)
    is_blocked = idx.arxiv_id.isin(set(blocked)).to_numpy()

    # candidates per subtopic
    cand = {}
    for f, fd in tax['fields'].items():
        fc = set(fd['categories'])
        infield = cset.apply(lambda cs: any(c in fc or c.split('.')[0] in fc for c in cs)).to_numpy() & ~is_blocked
        sub_idx = np.where(infield)[0]
        Xf = X[sub_idx]
        for s in fd['subtopics']:
            q = normalize(vec.transform([f'{s["name"]}. {s["name"]}. {s["scope"]} ' + ' '.join(s.get('keywords', []))]))
            sims = (Xf @ q.T).toarray().ravel()
            top = np.argsort(-sims)[:a.top]
            cand[s['id']] = [[idx.arxiv_id[sub_idx[j]], round(float(sims[j]), 4)] for j in top]
        # open draws: any in-field paper (a random sample, fixed)
        rng = np.random.default_rng(17)
        cand[f'{f}::open'] = [[idx.arxiv_id[j], 0.0] for j in rng.choice(sub_idx, size=min(2000, len(sub_idx)),
                                                                         replace=False)]
        print(f, len(sub_idx), flush=True)
    (out / 'candidates.json').write_text(json.dumps(cand))

    # extract texts
    need = {x for v in cand.values() for x, _ in v}
    rows = idx[idx.arxiv_id.isin(need)]
    by_shard = defaultdict(dict)
    for r in rows.itertuples():
        by_shard[(r.split, r.shard)][r.line] = r
    cctx = zstandard.ZstdCompressor(level=10, threads=4)
    n = 0
    with open(out / 'seeds.jsonl.zst', 'wb') as fo, cctx.stream_writer(fo) as w:
        for (split, shard), lines in sorted(by_shard.items()):
            with open(arx / 'proof-pile-2-arxiv' / split / shard, 'rb') as f:
                rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding='utf-8')
                for i, line in enumerate(rd):
                    r = lines.get(i)
                    if r is None:
                        continue
                    text = json.loads(line)['text']
                    w.write((json.dumps({'arxiv_id': r.arxiv_id, 'title': r.title, 'categories': r.categories,
                                         'abstract': r.abstract, 'n_chars': len(text),
                                         'n_math': len(MATH.findall(text)) + text.count('$') // 2,
                                         'text': text}) + '\n').encode())
                    n += 1
    print('seeds', n, 'of', len(need), flush=True)
    (out / 'seeds_stats.json').write_text(json.dumps({'physics_papers': len(idx), 'blocked': len(blocked),
                                                      'subtopics': len(cand), 'seed_papers': n}, indent=1))


if __name__ == '__main__':
    main()

"""Leaf check (#359 rule: no leaf may come from a single test item): each brainstormed scenario leaf against the
600 scored tasks' requests, by TF-IDF (word 1-2 grams) and bge-small-en-v1.5 cosine.

  # test side first (AutomationBench venv): python -m m5.automationbench.decon prepare --accepted /dev/null --work W
  HF_HOME=/data/aai/m5/hle/hf_home HF_HUB_OFFLINE=1 /data/aai/m5/hle/venv-decon/bin/python -m m5.automationbench.leafcheck \
      --leaves DIR --test W/test.jsonl --out W/leafcheck.json

A leaf is flagged when its nearest test task is close (embed >= 0.80 or tfidf >= 0.30) and clearly closer than the
second nearest (margin >= 0.05 embed / 0.08 tfidf): it then resembles one test task rather than a class of them.
Flagged leaves are listed for review; the sampler skips leaves in DIR/blocked.txt.
"""
import argparse
import json
from pathlib import Path

BGE = '/data/aai/m5/automationbench/sources/bge-small-en-v1.5'   # BAAI/bge-small-en-v1.5 @5c38ec7, MIT


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-embed', action='store_true')
    a = ap.parse_args()
    test = [json.loads(x) for x in open(a.test)]
    leaves = []
    for f in sorted(Path(a.leaves).glob('*.jsonl')):
        d, w = f.stem.split('.')
        for x in map(json.loads, f.read_text().splitlines()):
            leaves.append({'key': f'{d}.{w}.{x["slug"]}', 'domain': d,
                           'text': f'{x["scenario"]} Rules: {x.get("rule_hooks", "")}'})
    T, L = [t['trigger'] for t in test], [l['text'] for l in leaves]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english').fit(T + L)
    St = (vec.transform(L) @ vec.transform(T).T).toarray()
    if a.no_embed:
        Se = np.zeros_like(St)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(BGE, device='cpu')
        Se = m.encode(L, batch_size=64, normalize_embeddings=True) @ m.encode(T, batch_size=64, normalize_embeddings=True).T
    flagged = []
    for i, l in enumerate(leaves):
        row = {}
        for name, S, thr, margin in (('embed', Se, 0.80, 0.05), ('tfidf', St, 0.30, 0.08))[a.no_embed:]:
            o = np.argsort(-S[i])[:2]
            top, second = float(S[i, o[0]]), float(S[i, o[1]])
            row[name] = {'top': round(top, 3), 'second': round(second, 3), 'test': test[o[0]]['id']}
            if top >= thr and top - second >= margin:
                row.setdefault('why', []).append(name)
        if row.get('why'):
            flagged.append({'leaf': l['key'], 'text': l['text'], **row})
    res = {'n_leaves': len(leaves), 'n_flagged': len(flagged),
           'embed_top_quantiles': [float(np.quantile(Se.max(axis=1), q)) for q in (0.5, 0.9, 0.99)],
           'tfidf_top_quantiles': [float(np.quantile(St.max(axis=1), q)) for q in (0.5, 0.9, 0.99)],
           'flagged': sorted(flagged, key=lambda r: -r[('tfidf' if a.no_embed else 'embed')]['top'])}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'flagged'}))
    for r in res['flagged'][:60]:
        print(r['leaf'], r['why'], r.get('embed'), r['tfidf'])


if __name__ == '__main__':
    main()

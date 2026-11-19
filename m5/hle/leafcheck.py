"""Flag leaves that point at a single HLE test item (#19 lesson: a leaf with one source item drove most flags).

    python -m m5.hle.leafcheck [--out /data/aai/m5/hle/private/leafcheck.json]

The brainstorm never saw HLE questions, but a niche leaf can still coincide with one. For each leaf (name + scope +
queries/techniques) this finds its nearest test questions (all of cais/hle + Diamond) by TF-IDF and bge embeddings.
A leaf is flagged when its best match is close and clearly closer than its second (one item, not a family). Flagged
leaves are reviewed by hand and broadened or merged (m5/hle/data/leaf_edits.json); the report with test text stays in
the private dir.
"""
import argparse
import json
from pathlib import Path

from .decon import EMBED_MODEL, load_test

HERE = Path(__file__).resolve().parent


def leaf_text(d, l):
    extra = l.get('queries') or l.get('techniques') or []
    return f'{d["subject"]}: {l["leaf"]}. {l.get("scope") or ""} {" ".join(extra)}'


def main():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/aai/m5/hle/private/leafcheck.json')
    ap.add_argument('--embed', action='store_true')
    a = ap.parse_args()
    test = load_test()
    T = [t['question'] for t in test]
    leaves = []
    for f in sorted((HERE / 'data' / 'leaves').glob('*.json')):
        d = json.loads(f.read_text())
        for l in d['leaves']:
            leaves.append((f.stem, l['leaf'], leaf_text(d, l)))
    L = [x[2] for x in leaves]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words='english').fit(T + L)
    S = (vec.transform(L) @ vec.transform(T).T).toarray()
    sigs = {'tfidf': S}
    if a.embed:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBED_MODEL, device='cpu')
        sigs['embed'] = m.encode(L, normalize_embeddings=True) @ m.encode(T, normalize_embeddings=True).T
    flags = []
    for name, M in sigs.items():
        o = np.sort(M, axis=1)
        top1, top2 = o[:, -1], o[:, -2]
        thr = {'tfidf': (0.25, 1.6), 'embed': (0.80, 1.06)}[name]
        for i in np.where((top1 >= thr[0]) & (top1 >= thr[1] * top2))[0]:
            j = int(M[i].argmax())
            flags.append({'signal': name, 'node': leaves[i][0], 'leaf': leaves[i][1], 'top1': float(top1[i]),
                          'top2': float(top2[i]), 'test_id': test[j]['id'], 'test_question': test[j]['question'][:600]})
    flags.sort(key=lambda x: -x['top1'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({'n_leaves': len(leaves), 'flags': flags}, indent=1) + '\n')
    print(len(leaves), 'leaves', len(flags), 'flags')
    for fl in flags[:60]:
        print(f'{fl["signal"]} {fl["top1"]:.2f}/{fl["top2"]:.2f} {fl["node"]} :: {fl["leaf"]}')


if __name__ == '__main__':
    main()

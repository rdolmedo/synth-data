"""leaves_raw.json -> taxonomy/vocab.json: near-duplicate leaves dropped (TF-IDF cosine over title + gist, within a
subfield), leaves blocked that name the same core computation as a test problem (TF-IDF candidates reviewed by
hand: taxonomy/leaf_block.json), ids assigned, cell weights and step-count distribution attached.

    python -m m5.scicode.taxonomy build            # writes vocab.json and WORK/leaf_test_candidates.json
"""
import collections
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: I001

from . import taxonomy as T
from .common import TEST_PROBLEMS, append_line, read_jsonl, safe_write

DUP_COS = 0.33   # calibrated by reading pairs: 0.3-0.4 is mostly the same subject reworded
TEST_CAND_COS = 0.25


def text(x):
    return f"{x['title']}. {x['gist']}"


def test_texts():
    out = {}
    for p in read_jsonl(TEST_PROBLEMS):
        out[p['problem_id']] = ' '.join([p['problem_name'].replace('_', ' '), p['problem_description_main']]
                                        + [s['step_description_prompt'] for s in p['sub_steps']])
    return out


def main():
    fitd = T.fit()
    areas = json.loads((T.WORK / 'areas.json').read_text())
    w = T.cell_weights(fitd, areas)
    rows = T.all_leaf_rows()
    tt = test_texts()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words='english')
    X = vec.fit_transform([text(x) for _, x in rows] + list(tt.values()))
    L, Tm = X[:len(rows)], X[len(rows):]
    # near-duplicates anywhere in the vocabulary (greedy, keep the first): leaves of one subject in two cells or
    # reworded within a cell would give near-duplicate problems
    keep = np.ones(len(rows), bool)
    S = (L @ L.T).toarray()
    np.fill_diagonal(S, 0)
    ndup = 0
    order = np.random.default_rng(0).permutation(len(rows))
    for a in order:
        if not keep[a]:
            continue
        for b in np.nonzero(S[a] > DUP_COS)[0]:
            if keep[b]:
                keep[b] = False
                ndup += 1
    del S
    # candidates against the test problems, for the hand review
    St = (L @ Tm.T).toarray()
    tids = list(tt)
    cands = []
    for i in np.nonzero(St.max(1) > TEST_CAND_COS)[0]:
        j = int(St[i].argmax())
        cands.append({'i': int(i), 'cell': rows[i][0], 'title': rows[i][1]['title'], 'gist': rows[i][1]['gist'],
                      'test': tids[j], 'cos': round(float(St[i, j]), 3)})
    cands.sort(key=lambda r: -r['cos'])
    safe_write(T.WORK / 'leaf_test_candidates.json', json.dumps(cands, indent=1))
    blk_path = T.HERE / 'leaf_block.json'
    block = set(json.loads(blk_path.read_text())['block_titles']) if blk_path.exists() else set()
    mrg_path = T.HERE / 'leaf_merge.json'
    merged = set(json.loads(mrg_path.read_text())['drop_titles']) if mrg_path.exists() else set()
    for i, (_, x) in enumerate(rows):   # same-subject duplicates (leaf_merge) replace the TF-IDF dedup
        if x['title'] in merged:
            keep[i] = False
            ndup += 1
    leaves = collections.defaultdict(list)
    nblock = 0
    for i, (c, x) in enumerate(rows):
        if not keep[i]:
            continue
        if x['title'] in block:
            nblock += 1
            continue
        s, m = c.split('|')
        leaves[c].append({'id': f"{s.split('/')[1]}.{m}.{len(leaves[c]):04d}", **x})
    vocab = {'cells': {c: p for c, p in w.items() if leaves.get(c)},
             'leaves': dict(leaves), 'p_steps': {str(k): v for k, v in fitd['p_steps'].items()},
             'stats': {'raw': len(rows), 'near_dups_dropped': ndup, 'blocked': nblock,
                       'kept': sum(len(v) for v in leaves.values()), 'test_candidates': len(cands)}}
    z = sum(vocab['cells'].values())
    vocab['cells'] = {c: p / z for c, p in vocab['cells'].items()}
    safe_write(T.HERE / 'vocab.json', json.dumps(vocab, indent=0, ensure_ascii=False))
    print(json.dumps(vocab['stats']))

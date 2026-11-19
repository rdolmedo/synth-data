"""Merge brainstormed leaves into leaves.json and apply the single-test-item rule.

Leaves are brainstormed from labels only, so none is derived from a test item. As a guard against a leaf that
nevertheless centres on one test item (it would concentrate ~20 questions next to it), each leaf's text (name, scope,
queries) is compared with every v1.1 test question by TF-IDF cosine; a leaf whose best match is >= DROP_COS and
whose second-best match is far below (one item, not a class) is dropped, since its siblings cover the subtopic.
Only the drop decision and counts are printed; test text is never written into leaves.json.
"""
import csv
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent
TEST = Path('/data/aai/aa_evals/data/omniscience/public-v1.1@1f9cbce/AA-Omniscience_dataset_public.csv')
DROP_COS = 0.35


def main(out=HERE / 'leaves.json'):
    sub = json.loads((HERE / 'subtopics.json').read_text())
    leaves = []
    for f in sorted((HERE / 'leaves').glob('*.json')):
        d = json.loads(f.read_text())
        w = sub['domains'][d['domain']]['weight'] * sub['domains'][d['domain']]['subtopics'][d['subtopic']]['weight']
        slug = f.stem
        for i, L in enumerate(d['leaves']):
            leaves.append({'leaf_id': f'{slug}#{i:03d}', 'domain': d['domain'], 'subtopic': d['subtopic'],
                           'subtopic_weight': w, 'leaf': L['leaf'].strip(), 'scope': L.get('scope', ''),
                           'queries': [q for q in L['queries'] if isinstance(q, str)][:4],
                           'sources': L.get('sources', [])})
    test = [r['question'] for r in csv.DictReader(open(TEST, encoding='utf-8'))]
    texts = [f"{L['leaf']} {L['scope']} {' '.join(L['queries'])}" for L in leaves]
    vec = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=1).fit(texts + test)
    S = cosine_similarity(vec.transform(texts), vec.transform(test))
    keep, dropped = [], 0
    for L, row in zip(leaves, S):
        top = sorted(row, reverse=True)[:2]
        if top[0] >= DROP_COS and top[1] < 0.6 * top[0]:
            dropped += 1
            continue
        keep.append(L)
    subs = {(L['domain'], L['subtopic']) for L in keep}
    print(f'{len(leaves)} leaves, {dropped} dropped by the single-test-item guard, {len(keep)} kept, '
          f'{len(subs)} subtopics covered')
    Path(out).write_text(json.dumps({'n': len(keep), 'n_dropped_single_item': dropped, 'leaves': keep}, indent=0,
                                    ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:])

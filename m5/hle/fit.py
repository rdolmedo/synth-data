"""Fit HLE-Diamond's class-level mix (text-only questions) into taxonomy weights. Only aggregate counts leave here.

    python -m m5.hle.fit > m5/hle/data/fit.json

Levels: partition (the benchmark's 50/50) > category (fitted per partition, no flattening) > answer type (MC share
shrunk toward the partition's share) > subject node (merged raw subjects with >= 2 text items in the category, weights
per partition shrunk toward the category's subject mix; singleton subjects fold into a category-wide `broad` node).
"""
import collections
import json
import sys

sys.path.insert(0, '.')
from aa_evals.hle import data as hledata

MERGE = {'Puzzle': 'Puzzles', 'Game Of Go': 'Go', 'Cultural Studies (General)': 'Cultural Studies',
         'Advanced Applied Math': 'Applied Mathematics', 'Sport': 'Sports', 'Music': 'Musicology',
         'Classical Physics': 'Physics', 'Literature': 'Comparative Literature',
         'Combined Chemistry And Trivia': None}   # None: folded into the category's broad node
K_MC = 10      # pseudo-counts pulling a cell's MC share to its partition's
K_SUBJ = 4     # pseudo-counts pulling a partition's subject mix to the category's


def main():
    rows = [r for r in hledata.load('cais/hle-diamond') if not r['image']]
    out = {'source': 'cais/hle-diamond text-only-v1 (728 questions)', 'partition': {'reasoning': 0.5, 'knowledge': 0.5},
           'category': {}, 'mc_share': {}, 'subjects': {}, 'raw_counts': {}}
    for part in ('reasoning', 'knowledge'):
        P = [r for r in rows if r['partition'] == part]
        cat = collections.Counter(r['category'] for r in P)
        out['category'][part] = {c: n / len(P) for c, n in sorted(cat.items())}
        mc_p = sum(r['answer_type'] == 'multipleChoice' for r in P) / len(P)
        out['mc_share'][part] = {}
        for c in cat:
            C = [r for r in P if r['category'] == c]
            m = sum(r['answer_type'] == 'multipleChoice' for r in C)
            out['mc_share'][part][c] = (m + K_MC * mc_p) / (len(C) + K_MC)
        out['raw_counts'][part] = {c: {'n': n, 'mc': sum(r['answer_type'] == 'multipleChoice' for r in P if r['category'] == c)}
                                   for c, n in cat.items()}
    # question length (words) quantiles per partition x answer type, and the MC option-count mix
    import re
    out['length_words'] = {}
    for part in ('reasoning', 'knowledge'):
        for at in ('exactMatch', 'multipleChoice'):
            w = sorted(len(r['question'].split()) for r in rows if r['partition'] == part and r['answer_type'] == at)
            out['length_words'][f'{part}/{at}'] = [w[round(q / 20 * (len(w) - 1))] for q in range(21)]
    nopt = collections.Counter(len(re.findall(r'^\s*([A-Z])[\.\)]\s', r['question'], re.M))
                               for r in rows if r['answer_type'] == 'multipleChoice')
    out['mc_options'] = {str(k): v for k, v in sorted(nopt.items()) if k >= 3}
    subj = lambda r: MERGE.get(r['raw_subject'], r['raw_subject'])
    for c in sorted({r['category'] for r in rows}):
        C = [r for r in rows if r['category'] == c]
        cc = collections.Counter(subj(r) for r in C)
        nodes = sorted(s for s, n in cc.items() if n >= 2 and s is not None)
        broad = sum(n for s, n in cc.items() if n < 2 or s is None)
        share = {s: cc[s] / len(C) for s in nodes}
        share['broad'] = broad / len(C)
        out['subjects'][c] = {}
        for part in ('reasoning', 'knowledge'):
            P = [r for r in C if r['partition'] == part]
            pc = collections.Counter(subj(r) if subj(r) in share else 'broad' for r in P)
            w = {s: (pc.get(s, 0) + K_SUBJ * share[s]) / (len(P) + K_SUBJ) for s in share}
            # a subject seen only in the other partition keeps a small share; drop it when it is below 2%
            w = {s: v for s, v in w.items() if v >= 0.02 or pc.get(s, 0)}
            z = sum(w.values())
            out['subjects'][c][part] = {s: round(v / z, 4) for s, v in sorted(w.items(), key=lambda x: -x[1])}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()

"""Spirit check and pilot metrics: generated problems vs the GDPval test set (class-level comparison).

  python -m m5.gdpval.spirit --final F1.jsonl [F2 ...] --out DIR [--label] [--n-prod 11000]

- labels the generated problems with the same Opus labeler as the test set (label_test.py) when --label;
- compares task type, deliverable formats, input presence and file types, prompt length, needs-web share,
  number of deliverables, occupation spread, task-type spread per sector against the sampler targets;
- repeat meter: share of problems with a TF-IDF near-duplicate (cosine >= 0.5) among the generated set, and its
  linear projection to the production count (#53: repeats grow ~linearly with N);
- 8-gram overlap of each problem prompt with the generator's own instructions and spec text (its "examples") and
  with the real document it was given (seeds).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from m5.gdpval import prompts as P
from m5.gdpval.opus import WORK, ask_json
from m5.gdpval.textsim import ngrams, sim_matrix, tfidf_matrix


def load(paths):
    out = {}
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            r = json.loads(ln)
            out[r['idx']] = r
    return [out[k] for k in sorted(out)]


def label(recs):
    from m5.gdpval.label_test import SYSTEM, batch_prompt
    rows = [{'task_id': f'g{r["idx"]:07d}', 'occupation': r['spec']['occupation'], 'prompt': r['problem']['task_prompt'],
             'reference_files': [n for n, _ in r['problem']['input_files']] + ([r['doc']['name']] if r.get('doc') else []),
             'deliverable_files': []} for r in recs]
    batches = [rows[i:i + 10] for i in range(0, len(rows), 10)]
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda b: ask_json(SYSTEM, batch_prompt(b), 'label-gen'), batches))
    lab = {}
    for b, o in zip(batches, outs):
        by = {x['id']: x for x in o}
        for r in b:
            x = by.get(r['task_id'][:8])
            if x:
                lab[int(r['task_id'][1:])] = x
    return lab


def dist(c: collections.Counter, n=None):
    n = n or sum(c.values())
    return {k: round(v / n, 3) for k, v in c.most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', action='store_true')
    ap.add_argument('--n-prod', type=int, default=11000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load(a.final)
    test = json.loads((WORK / 'test_labels.json').read_text())
    test = {k: v for k, v in test.items() if k[:8] not in __import__('m5.gdpval.sampler', fromlist=['x']).TEXT_ONLY_EXCLUDED}
    rep = {'n_generated': len(recs), 'n_test': len(test)}
    gw = [len(r['problem']['task_prompt'].split()) for r in recs]
    tw = [v['n_words'] for v in test.values()]
    rep['words'] = {'gen': np.percentile(gw, [10, 25, 50, 75, 90]).round().tolist(),
                    'test': np.percentile(tw, [10, 25, 50, 75, 90]).round().tolist()}
    rep['lane'] = dist(collections.Counter(r['spec']['lane'] for r in recs))
    rep['has_inputs'] = {'gen': round(np.mean([r['spec']['lane'] != 'none' for r in recs]), 3),
                         'test': round(np.mean([bool(v['ref_ext']) for v in test.values()]), 3)}
    ge = collections.Counter(Path(n).suffix.lower() for r in recs for n, _ in r['problem']['input_files'])
    ge.update(Path(r['doc']['name']).suffix.lower() for r in recs if r.get('doc'))
    rep['input_ext'] = {'gen': dist(ge), 'test': dist(collections.Counter(e for v in test.values() for e in v['ref_ext']))}
    rep['n_inputs'] = {'gen': dist(collections.Counter(min(len(r['problem']['input_files']) + bool(r.get('doc')), 5)
                                                       for r in recs if r['spec']['lane'] != 'none')),
                       'test': dist(collections.Counter(min(len(v['ref_ext']), 5) for v in test.values() if v['ref_ext']))}
    rep['occupations'] = {'n': len({r['spec']['occupation'] for r in recs}),
                          'max_share': max(collections.Counter(r['spec']['occupation'] for r in recs).values()) / len(recs)}
    rep['open_share'] = round(np.mean([r['spec']['open'] for r in recs]), 3)
    rep['spec_task_type'] = dist(collections.Counter(r['spec']['task_type'] for r in recs))
    rep['spec_formats'] = dist(collections.Counter(f for r in recs for f in r['spec']['formats']))
    rep['test_task_type'] = dist(collections.Counter(v['task_type'] for v in test.values()))
    rep['test_formats'] = dist(collections.Counter(f for v in test.values() for f in v['deliverable_formats']))
    if a.label:
        lab = label(recs)
        (out / 'gen_labels.json').write_text(json.dumps(lab, indent=1))
        rep['label_task_type'] = dist(collections.Counter(x['task_type'] for x in lab.values()))
        rep['label_formats'] = dist(collections.Counter(f for x in lab.values() for f in x['deliverable_formats']))
        rep['label_needs_web'] = {'gen': round(np.mean([bool(x.get('needs_web')) for x in lab.values()]), 3),
                                  'test': round(np.mean([bool(v.get('needs_web')) for v in test.values()]), 3)}
        rep['label_n_deliverables'] = {'gen': dist(collections.Counter(min(int(x.get('n_deliverables') or 1), 4)
                                                                       for x in lab.values())),
                                       'test': dist(collections.Counter(min(int(v.get('n_deliverables') or 1), 4)
                                                                        for v in test.values()))}
        rep['label_provenance'] = {'gen': dist(collections.Counter(i['provenance'] for x in lab.values()
                                                                   for i in x.get('inputs') or [])),
                                   'test': dist(collections.Counter(i['provenance'] for v in test.values()
                                                                    for i in v.get('inputs') or []))}
        rep['label_input_role'] = {'gen': dist(collections.Counter(i['role'] for x in lab.values()
                                                                   for i in x.get('inputs') or [])),
                                   'test': dist(collections.Counter(i['role'] for v in test.values()
                                                                    for i in v.get('inputs') or []))}
        agree = [lab[r['idx']]['task_type'] == r['spec']['task_type'] for r in recs if r['idx'] in lab]
        rep['label_matches_spec_task_type'] = round(np.mean(agree), 3) if agree else None
    # repeat meter
    texts = [r['problem']['task_prompt'] for r in recs]
    V = tfidf_matrix(texts)
    S = sim_matrix(V, V)
    np.fill_diagonal(S, 0)
    for thr in (0.4, 0.5, 0.6):
        share = float(np.mean(S.max(1) >= thr))
        rep[f'repeat_share_cos{thr}'] = {'pilot': round(share, 4),
                                         'projected_prod': round(min(1.0, share * a.n_prod / max(len(recs), 1)), 4)}
    top = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:20:2], S.shape))[0]
    rep['top_pairs'] = [{'a': recs[i]['idx'], 'b': recs[j]['idx'], 'cos': round(float(S[i, j]), 3),
                         'occ': [recs[i]['spec']['occupation'], recs[j]['spec']['occupation']]} for i, j in top]
    # 8-gram overlap with the generator's instructions/spec ("examples") and with the seed document
    inst = ngrams(P.SYSTEM + '\n' + P.SYNTH_RULES + '\n' + P.REAL_RULES + '\n' + P.OUTPUT_FORMAT, 8)
    ex, seed = [], []
    for r in recs:
        g = ngrams(r['problem']['task_prompt'], 8)
        if not g:
            continue
        spec = ngrams(P.spec_text(r['spec']), 8)
        ex.append(len(g & (inst | spec)) / len(g))
        if r.get('doc') and r['doc'].get('text_path') and os.path.exists(r['doc']['text_path']):
            seed.append(len(g & ngrams(Path(r['doc']['text_path']).read_text()[:400000], 8)) / len(g))
    rep['ngram8_overlap_examples'] = {'mean': round(float(np.mean(ex)), 4), 'p95': round(float(np.percentile(ex, 95)), 4),
                                      'max': round(float(np.max(ex)), 4)}
    if seed:
        rep['ngram8_overlap_seed_doc'] = {'n': len(seed), 'mean': round(float(np.mean(seed)), 4),
                                          'p95': round(float(np.percentile(seed, 95)), 4)}
    (out / 'spirit.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

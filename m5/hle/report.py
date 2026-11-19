"""Pilot / production metrics for the HLE pipeline (the M5 checklist).

    python -m m5.hle.report --problems 'P/problems*.jsonl' [--traces T/traces.jsonl] [--project 14000]

- problems: status mix, check failures, length vs HLE quantiles, answer forms, MC letter balance;
- spread over the taxonomy (partition x category x answer type) against the fitted targets; leaf coverage;
- repeat meter: TF-IDF cosine nearest neighbour within the set; near-duplicate pair rate projected to --project;
- copying: share of question 8-grams found in the style guide and in the problem's own seed excerpt;
- traces: submit share, context-limit share, tokens per trace (prompt + completion), by partition.
"""
import argparse
import collections
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def grams(t, n=8):
    w = re.findall(r'\w+', (t or '').lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load(pattern):
    rows = {}
    for p in sorted(glob.glob(pattern)):
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            rows[r['id']] = r
    return list(rows.values())


def repeat_meter(texts, project, thr=(0.6, 0.7, 0.8)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    if len(texts) < 3:
        return {}
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words='english').fit_transform(texts)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    n = len(texts)
    out = {'n': n, 'nn_cos_median': float(np.median(S.max(axis=1))), 'nn_cos_p95': float(np.quantile(S.max(axis=1), 0.95))}
    for t in thr:
        pairs = int((S >= t).sum() // 2)
        rate = pairs / (n * (n - 1) / 2)
        share = float((S.max(axis=1) >= t).mean())
        # each problem's chance of having a near-duplicate among N others ~ 1 - (1 - rate)^(N-1)
        out[f'cos>={t}'] = {'pairs': pairs, 'pair_rate': rate, 'share_with_dup': share,
                            'projected_share_with_dup': 1 - (1 - rate) ** (project - 1)}
    i, j = np.unravel_index(np.argmax(S), S.shape)
    out['top_pair'] = [int(i), int(j), float(S[i, j])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True)
    ap.add_argument('--traces')
    ap.add_argument('--project', type=int, default=14000)
    ap.add_argument('--out')
    a = ap.parse_args()
    fit = json.loads((HERE / 'data' / 'fit.json').read_text())
    P = load(a.problems)
    rep = {'n_problems': len(P), 'status': dict(collections.Counter(p.get('status') for p in P)),
           'check_failures': dict(collections.Counter(c for p in P for c in p.get('checks') or []))}
    ok = [p for p in P if p.get('status') == 'ok']
    rep['finish'] = dict(collections.Counter(p.get('finish') for p in P))
    rep['gen_tokens'] = {'completion_mean': st.mean([p['usage']['completion'] or 0 for p in P]) if P else None,
                         'prompt_mean': st.mean([p['usage']['prompt'] or 0 for p in P]) if P else None}
    # spread vs targets
    cell = collections.Counter((p['spec']['partition'], p['spec']['category'], p['spec']['answer_type']) for p in ok)
    tgt = {}
    for part, cw in fit['category'].items():
        for c, w in cw.items():
            mc = fit['mc_share'][part][c]
            tgt[(part, c, 'multipleChoice')] = fit['partition'][part] * w * mc
            tgt[(part, c, 'exactMatch')] = fit['partition'][part] * w * (1 - mc)
    n = max(1, len(ok))
    rep['spread'] = sorted([{'cell': '/'.join(k), 'target': round(v, 4), 'got': round(cell.get(k, 0) / n, 4)}
                            for k, v in tgt.items()], key=lambda x: -x['target'])
    rep['spread_tv_distance'] = 0.5 * sum(abs(cell.get(k, 0) / n - v) for k, v in tgt.items())
    rep['open_draw_share'] = sum(1 for p in ok if not p['spec'].get('leaf')) / n
    leaves = collections.Counter(p['spec'].get('leaf') for p in ok if p['spec'].get('leaf'))
    rep['leaves'] = {'distinct': len(leaves), 'max_uses': max(leaves.values()) if leaves else 0}
    # lengths vs HLE quantiles
    rep['length_words'] = {}
    for key in fit['length_words']:
        part, at = key.split('/')
        w = sorted(len(p['question'].split()) for p in ok if p['spec']['partition'] == part and p['spec']['answer_type'] == at)
        if w:
            q = fit['length_words'][key]
            rep['length_words'][key] = {'n': len(w), 'gen_q25_50_75': [w[len(w) // 4], w[len(w) // 2], w[3 * len(w) // 4]],
                                        'hle_q25_50_75': [q[5], q[10], q[15]]}
    rep['mc_letters'] = dict(collections.Counter(p['answer'].strip()[:1] for p in ok if p['spec']['answer_type'] == 'multipleChoice'))
    rep['answer_words_median'] = st.median([len(p['answer'].split()) for p in ok if p['spec']['answer_type'] == 'exactMatch']) if ok else None
    rep['latex_share'] = {c: round(sum('$' in p['question'] for p in ok if p['spec']['category'] == c) /
                                   max(1, sum(p['spec']['category'] == c for p in ok)), 2) for c in fit['category']['reasoning']}
    # copying
    sg = grams((HERE / 'prompts' / 'style_guide.md').read_text())
    rep['copy_style_guide_8gram_share'] = st.mean([len(grams(p['question']) & sg) / max(1, len(grams(p['question']))) for p in ok]) if ok else None
    sd = [p['seed_8gram_share'] for p in ok if 'seed_8gram_share' in p]
    rep['copy_seed_8gram_share'] = {'mean': st.mean(sd), 'p90': sorted(sd)[int(0.9 * (len(sd) - 1))],
                                    'max': max(sd)} if sd else None
    rep['repeat_meter'] = repeat_meter([p['question'] for p in ok], a.project)
    if a.traces:
        T = load(a.traces)
        T = [t for t in T if t.get('status') == 'ok']
        rep['traces'] = {'n': len(T), 'submit_share': sum(t['has_answer'] for t in T) / max(1, len(T)),
                         'context_limit_share': sum(t['hit_context'] for t in T) / max(1, len(T)),
                         'no_answer_stop_share': sum((not t['has_answer']) and not t['hit_context'] for t in T) / max(1, len(T)),
                         'tokens_mean': st.mean([t['tokens_total'] for t in T]) if T else None,
                         'tokens_median': st.median([t['tokens_total'] for t in T]) if T else None,
                         'kept_tokens': sum(t['tokens_total'] for t in T if t['kept'])}
        for part in ('reasoning', 'knowledge'):
            tp = [t for t in T if t['spec']['partition'] == part]
            if tp:
                rep['traces'][part] = {'n': len(tp), 'tokens_mean': st.mean([t['tokens_total'] for t in tp]),
                                       'tokens_median': st.median([t['tokens_total'] for t in tp]),
                                       'submit_share': sum(t['has_answer'] for t in tp) / len(tp),
                                       'context_limit_share': sum(t['hit_context'] for t in tp) / len(tp)}
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s + '\n')
    print(s)


if __name__ == '__main__':
    main()

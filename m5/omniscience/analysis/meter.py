"""Pilot / prototype metrics for m5-omniscience (the shared checklist plus the spirit check).

  python -m m5.omniscience.analysis.meter RUN_DIR [RUN_DIR ...] --leaves L --production 66000 [--out report.json]

- Repeat meter (TF-IDF word 1-2 gram cosine between accepted questions). Pair probabilities are measured separately
  for pairs in the same leaf, the same subtopic (other leaf), the same domain (other subtopic) and across domains,
  then projected to the production composition (calls per leaf/subtopic/domain from the fitted weights):
  expected near-duplicates per item = sum over strata of (n_stratum - 1) x p_stratum.
- 8-gram overlap of questions with the style examples (all 600 v1.0 questions) and with each question's seed passage.
- Taxonomy spread against targets (domain and subtopic shares).
- Traces: submit share (final answer present), context-limit share, tokens per trace, abstention share.
- Spirit: question/answer length and answer-type mix against the v1.1 test set (class-level statistics only).
"""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

V10 = Path('/data/aai/aa_evals/data/omniscience/public-v1@e4883ed/AA-Omniscience_dataset_public.csv')
V11 = Path('/data/aai/aa_evals/data/omniscience/public-v1.1@1f9cbce/AA-Omniscience_dataset_public.csv')
THRESH = (0.5, 0.6, 0.7)
ABSTAIN = re.compile(r"\b(i (do not|don't|cannot|can't) (know|recall|reliably|confirm|determine|say|find|verify)|"
                     r"not (sure|certain|aware)|unable to (determine|confirm|answer|verify|recall)|"
                     r"no (reliable|specific) (information|record|knowledge)|i'm not able|i am not able|"
                     r"need more context|without (more|additional) (context|information|tools))", re.I)


def norm_words(s):
    return re.sub(r'[^0-9a-z]+', ' ', s.lower()).split()


def grams(s, n=8):
    w = norm_words(s)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def answer_type(a):
    a = a.strip()
    if re.fullmatch(r'(c\.\s*)?\d{3,4}( (bce?|ce|ad))?', a, re.I):
        return 'year'
    if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d', a, re.I):
        return 'date'
    if re.fullmatch(r'v?\d+(\.\d+){1,3}[a-z0-9.\-]*', a, re.I):
        return 'version'
    if re.fullmatch(r'[\s$€£~≈<>%+\-.,:/()0-9a-z]*\d[\s$€£~≈<>%+\-.,:/()0-9a-z]*', a, re.I) and \
            len(re.findall(r'[a-z]', a, re.I)) <= 12:
        return 'number'
    if re.search(r'[=∑∫^_\\(){}]|[α-ωΑ-Ω]', a) or ('`' in a) or re.search(r'[a-z]+\(|::|\.\w+\(', a):
        return 'formula/code'
    if len(a.split()) <= 4 and sum(w[:1].isupper() for w in a.split()) >= max(1, len(a.split()) - 1):
        return 'name'
    return 'phrase'


def dist(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {'n': len(xs), 'mean': round(statistics.mean(xs), 1), 'p10': q(.1), 'median': q(.5), 'p90': q(.9),
            'p99': q(.99), 'max': xs[-1]}


def load(run_dirs):
    gen, traces = [], {}
    for d in run_dirs:
        d = Path(d)
        if (d / 'gen.jsonl').exists():
            gen += [json.loads(l) for l in open(d / 'gen.jsonl')]
        if (d / 'traces.jsonl').exists():
            for l in open(d / 'traces.jsonl'):
                r = json.loads(l)
                traces[r['problem_id']] = r
    return gen, traces


def repeat_meter(acc, leaves, production):
    qs = [r['question'] for r in acc]
    if len(qs) < 3:
        return {}
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english').fit_transform(qs)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    lab = [(r['domain'], r['subtopic'], r['leaf_id']) for r in acc]
    strata = collections.defaultdict(lambda: [0, collections.Counter()])
    n = len(qs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lab[i], lab[j]
            k = 'leaf' if a[2] == b[2] else 'subtopic' if a[1] == b[1] else 'domain' if a[0] == b[0] else 'cross'
            strata[k][0] += 1
            for t in THRESH:
                if S[i, j] >= t:
                    strata[k][1][t] += 1
    p = {k: {str(t): (v[1][t] / v[0] if v[0] else None) for t in THRESH} | {'pairs': v[0],
                                                                               'hits': {str(t): v[1][t] for t in THRESH}}
         for k, v in strata.items()}
    # production composition: calls per leaf, per subtopic, per domain
    by_sub = collections.Counter()
    sub_leaves = collections.Counter()
    for L in leaves:
        by_sub[(L['domain'], L['subtopic'])] = L['subtopic_weight']
        sub_leaves[(L['domain'], L['subtopic'])] += 1
    proj = {}
    for t in THRESH:
        exp = 0.0
        for (d, s), w in by_sub.items():
            n_sub = production * w
            n_leaf = n_sub / sub_leaves[(d, s)]
            n_dom = production * sum(ww for (dd, _), ww in by_sub.items() if dd == d)
            terms = {'leaf': n_leaf - 1, 'subtopic': n_sub - n_leaf, 'domain': n_dom - n_sub, 'cross': production - n_dom}
            e = sum(max(0, terms[k]) * (p.get(k, {}).get(str(t)) or 0) for k in terms)
            exp += w * e
        proj[str(t)] = round(exp, 4)
    nn = S.max(1)
    observed = {str(t): round(float((nn >= t).mean()), 4) for t in THRESH}
    top = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:16:2], S.shape))[0]
    examples = [(round(float(S[i, j]), 3), qs[i], qs[j]) for i, j in top]
    return {'n': n, 'pair_prob': p, 'observed_share_with_nn_ge': observed,
            'projected_dups_per_item_at_production': proj, 'production_n': production, 'top_pairs': examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--passages', default=None, help='passages.jsonl.zst for the 8-gram seed overlap')
    ap.add_argument('--production', type=int, default=66000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    leaves = json.loads(Path(a.leaves).read_text())['leaves']
    gen, traces = load(a.runs)
    acc = [r for r in gen if r.get('accepted')]
    rep = {'gen_calls': len(gen), 'status': dict(collections.Counter(r.get('status') for r in gen)),
           'rejects': dict(collections.Counter(b for r in gen for b in (r.get('reject') or []))),
           'accepted': len(acc)}
    rep['accept_rate'] = round(len(acc) / max(1, len(gen)), 3)
    u = [r for r in gen if r.get('usage')]
    rep['generator_tokens'] = {'prompt': dist([r['usage']['prompt'] or 0 for r in u]),
                               'completion': dist([r['usage']['completion'] or 0 for r in u]),
                               'latency_s': dist([r['latency_s'] for r in u])}
    rep['by_source'] = dict(collections.Counter(r['source'] for r in acc))
    rep['accept_by_source'] = {s: round(sum(1 for r in acc if r['source'] == s) /
                                        max(1, sum(1 for r in gen if r['source'] == s)), 3)
                               for s in sorted({r['source'] for r in gen})}
    # taxonomy spread
    tgt_dom, tgt_sub = collections.Counter(), {}
    for L in leaves:
        tgt_sub[(L['domain'], L['subtopic'])] = L['subtopic_weight']
    for (d, s), w in tgt_sub.items():
        tgt_dom[d] += w
    got_dom = collections.Counter(r['domain'] for r in acc)
    got_sub = collections.Counter((r['domain'], r['subtopic']) for r in acc)
    n = max(1, len(acc))
    rep['domain_spread'] = {d: {'target': round(w, 3), 'got': round(got_dom[d] / n, 3)} for d, w in tgt_dom.items()}
    rep['subtopic_spread'] = {'l1': round(sum(abs(got_sub[k] / n - w) for k, w in tgt_sub.items()), 3),
                              'covered': sum(1 for k in tgt_sub if got_sub[k]), 'of': len(tgt_sub),
                              'leaves_used': len({r['leaf_id'] for r in acc})}
    # 8-gram overlap
    ex = list(csv.DictReader(open(V10, encoding='utf-8')))
    exg = set().union(*(grams(e['question']) for e in ex))
    rep['8gram_with_style_examples'] = round(sum(bool(grams(r['question']) & exg) for r in acc) / n, 4)
    rep['8gram_with_style_examples_examples'] = [r['question'] for r in acc if grams(r['question']) & exg][:5]
    if a.passages:
        from m5.omniscience.gen.pipeline import read_zst
        need = {r['pid'] for r in acc}
        ptext = {p['pid']: p.get('header', '') + ' ' + p['text'] for p in read_zst(a.passages) if p['pid'] in need}
        ov = [len(grams(r['question']) & grams(ptext.get(r['pid'], ''))) for r in acc]
        rep['8gram_with_seed_passage'] = {'share_any': round(sum(o > 0 for o in ov) / n, 3), 'mean_shared': round(
            statistics.mean(ov), 2) if ov else 0}
    rep['repeat_meter'] = repeat_meter(acc, leaves, a.production)
    # spirit (class-level only)
    test = list(csv.DictReader(open(V11, encoding='utf-8')))
    rep['spirit'] = {
        'question_chars': {'ours': dist([len(r['question']) for r in acc]), 'test': dist([len(t['question']) for t in test])},
        'answer_chars': {'ours': dist([len(r.get('answer_check_only', '')) for r in acc]),
                         'test': dist([len(t['answer']) for t in test])},
        'answer_type': {'ours': {k: round(v / n, 3) for k, v in collections.Counter(
            answer_type(r.get('answer_check_only', '')) for r in acc).most_common()},
            'test': {k: round(v / len(test), 3) for k, v in collections.Counter(
                answer_type(t['answer']) for t in test).most_common()}}}
    # traces
    tr = [traces[r['problem_id']] for r in acc if r['problem_id'] in traces]
    if tr:
        m = len(tr)
        rep['traces'] = {
            'n': m,
            'submit_share': round(sum(bool(t['answer'].strip()) for t in tr) / m, 3),
            'context_limit_share': round(sum(t['finish_reason'] == 'length' for t in tr) / m, 3),
            'timeout_share': round(sum(t['finish_reason'] == 'timeout' for t in tr) / m, 3),
            'abstain_share': round(sum(bool(ABSTAIN.search(t['answer'])) for t in tr) / m, 3),
            'total_tokens': dist([t.get('total_tokens') or 0 for t in tr]),
            'completion_tokens': dist([(t.get('usage') or {}).get('completion_tokens') or 0 for t in tr]),
            'sum_total_tokens': sum(t.get('total_tokens') or 0 for t in tr),
            'prompt_ids': dict(collections.Counter(t['prompt_id'] for t in tr)),
        }
    s = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == '__main__':
    main()

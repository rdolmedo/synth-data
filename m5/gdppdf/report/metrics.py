"""Pilot / production metrics for #357's checklist.

  python -m m5.gdppdf.report.metrics --problems '<dir>/problems-*.jsonl' --traces '<dir>/traces/traces-*.jsonl' \
      --index <index.jsonl> --taxonomy m5/gdppdf/taxonomy/taxonomy.json [--labels <labels.jsonl>] --n-prod 12600

- traces: submit share (non-empty final answer), context-limit share (finish_reason length), no-answer kinds,
  tokens per trace (prompt + completion = the whole exported sequence), output tokens;
- repeat meter: TF-IDF (word 1-2 gram) nearest neighbours over all accepted problems; pairs above 0.5 / 0.6 / 0.7,
  split into same-document and cross-document; cross-document pair rate projected to --n-prod problems
  (expected pairs = p * N(N-1)/2, share of problems with a near-duplicate ~ 1 - exp(-p (N-1)));
- copying: share of each problem's 8-grams found in the generator's own prompt text (instructions, slot text,
  leaves) and in its seed document;
- taxonomy spread: domains, document types, page bins, task types (spec and, with --labels, as labelled), asks,
  pitfalls, inline material, length, open draws, leaf concentration; against taxonomy.json.
"""
import argparse
import collections
import glob
import json
import math
import statistics as st

from m5.gdppdf.gen import checks, prompt as P


def load(globpat):
    out = []
    for p in sorted(glob.glob(globpat)):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                out.append(json.loads(l))
    return out


def q(v, ps=(0.1, 0.5, 0.9)):
    v = sorted(v)
    return {f'p{int(p * 100)}': v[min(len(v) - 1, int(p * len(v)))] for p in ps} if v else {}


def dist(c):
    n = sum(c.values())
    return {k: round(v / n, 3) for k, v in sorted(c.items(), key=lambda x: -x[1])} if n else {}


def repeat_meter(probs, n_prod):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [p['prompt'] for p in probs]
    n = len(texts)
    if n < 3:
        return {}
    m = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True, min_df=1).fit_transform(texts)
    docs = np.array([hash(p['doc_id']) for p in probs])
    out = {'n': n}
    counts = {th: [0, 0] for th in (0.5, 0.6, 0.7)}
    nn = np.zeros(n, dtype=np.float32)
    examples = []
    for a in range(0, n, 2000):   # blocks of rows: memory stays ~2000 x n
        sim = (m[a:a + 2000] @ m.T).toarray()
        rows = np.arange(a, min(n, a + 2000))
        same = docs[rows][:, None] == docs[None, :]
        sim[np.arange(len(rows)), rows] = 0
        nn[rows] = np.where(same, 0, sim).max(axis=1)
        upper = np.arange(n)[None, :] > rows[:, None]
        for th in counts:
            hit = (sim > th) & upper
            counts[th][0] += int((hit & same).sum())
            counts[th][1] += int((hit & ~same).sum())
        ii, jj = np.nonzero((sim > 0.6) & upper & ~same)
        examples += [(float(sim[x, y]), probs[rows[x]]['problem_id'], probs[y]['problem_id']) for x, y in zip(ii, jj)][:20]
    n_cross_pairs = n * (n - 1) / 2 - sum(math.comb(c, 2) for c in collections.Counter(docs.tolist()).values())
    for th, (same_c, cross_c) in counts.items():
        pr = cross_c / max(1, n_cross_pairs)
        out[f'>{th}'] = {'same_doc_pairs': same_c, 'cross_doc_pairs': cross_c, 'cross_pair_rate': pr,
                         'projected_pairs_at_N': round(pr * n_prod * (n_prod - 1) / 2),
                         'projected_share_with_dup_at_N': round(1 - math.exp(-pr * (n_prod - 1)), 4)}
    out['nn_cross_doc_cos'] = q(nn.tolist(), (0.5, 0.9, 0.99, 0.999))
    out['examples_over_0.6'] = sorted(examples, reverse=True)[:10]
    return out


def copying(probs, store, taxonomy):
    leaves = {l['leaf'] for v in taxonomy['leaves'].values() for l in v}
    template = ' '.join([P.INTRO, P.SLOT, *P.TASK_TYPE_TEXT.values(), *P.PITFALL_KINDS, *P.INLINE_TEXT.values(),
                         *leaves])
    tg = checks.ngrams(checks.words(template))
    ex, seed = [], []
    for p in probs:
        g = checks.ngrams(checks.words(p['prompt']))
        if not g:
            continue
        ex.append(len(g & tg) / len(g))
        if store is not None:
            d = store.get(p['doc_id'])
            dg = checks.ngrams(checks.words(' '.join(x['text'] for x in d['pages'])))
            seed.append(len(g & dg) / len(g))
    r = {'examples_8gram_share': {'mean': round(st.mean(ex), 4), **q(ex, (0.5, 0.9, 0.99))} if ex else None}
    if seed:
        r['seed_doc_8gram_share'] = {'mean': round(st.mean(seed), 4), **q(seed, (0.5, 0.9, 0.99))}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True)
    ap.add_argument('--traces', default=None)
    ap.add_argument('--index', default=None)
    ap.add_argument('--taxonomy', default='m5/gdppdf/taxonomy/taxonomy.json')
    ap.add_argument('--labels', default=None, help='label_requests.py output on the problems')
    ap.add_argument('--n-prod', type=int, default=12600)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tax = json.load(open(a.taxonomy))
    allp = load(a.problems)
    acc = [p for p in allp if p['status'] == 'accepted']
    rep = {'problems': {'written': len(allp), 'accepted': len(acc),
                        'rejected': dist(collections.Counter(p['status'] for p in allp if p['status'] != 'accepted')),
                        'docs': len({p['doc_id'] for p in acc}),
                        'prompt_words': q([p['flags'].get('words', 0) for p in acc]),
                        'open_draws': round(sum(p['spec'].get('open_draw', False) for p in acc) / max(1, len(acc)), 4),
                        'page_span': q([p['flags'].get('page_span', 0) for p in acc])}}
    if a.traces:
        tr = load(a.traces)
        n = len(tr)
        ok = [t for t in tr if t['status'] == 'ok']
        tok = [(t.get('usage') or {}).get('prompt_tokens', 0) + (t.get('usage') or {}).get('completion_tokens', 0)
               for t in tr if t.get('usage')]
        comp = [(t.get('usage') or {}).get('completion_tokens', 0) for t in tr if t.get('usage')]
        rep['traces'] = {'n': n, 'submit_share': round(len(ok) / max(1, n), 4),
                         'context_limit_share': round(sum(t.get('finish_reason') == 'length' for t in tr) / max(1, n), 4),
                         'status': dist(collections.Counter(t['status'] for t in tr)),
                         'tokens_per_trace': {'mean': round(st.mean(tok)) if tok else None, **q(tok)},
                         'completion_tokens': {'mean': round(st.mean(comp)) if comp else None, **q(comp)},
                         'total_tokens': sum(tok), 'seconds': q([t.get('seconds', 0) for t in tr])}
    store = None
    if a.index:
        from m5.gdppdf.docs.store import DocStore
        store = DocStore(a.index)
    rep['repeats'] = repeat_meter(acc, a.n_prod)
    rep['copying'] = copying(acc, store, tax)
    spec = lambda k: dist(collections.Counter(str(p['spec'].get(k)) for p in acc))
    leaf_c = collections.Counter(p['spec']['leaf'] for p in acc)
    from m5.gdppdf.docs.select import BINS, bin_of
    docs = {p['doc_id']: p for p in acc}
    rep['spread'] = {
        'domain': {'got': dist(collections.Counter(p['domain'] for p in acc)), 'target': tax['domains']},
        'doc_type': dist(collections.Counter(p.get('doc_type') for p in acc)),
        'page_bins_docs': {'got': dist(collections.Counter(f'{BINS[bin_of(p["n_pages"])][0]}-{BINS[bin_of(p["n_pages"])][1]}'
                                                       for p in docs.values())),
                           'target': {f'{lo}-{hi}': s for lo, hi, s in BINS}},
        'task_type_spec': {'got': spec('task_type'), 'target': tax['task_types_global']},
        'n_asks_spec': spec('n_asks'), 'pitfalls_spec': {'got': spec('n_pitfalls'), 'target': tax['pitfalls']},
        'inline_spec': spec('inline'), 'format_spec': spec('format'),
        'leaves': {'distinct': len(leaf_c), 'max_share': round(max(leaf_c.values()) / max(1, len(acc)), 4) if leaf_c else 0}}
    if a.labels:
        lab = {r['id']: r['label'] for r in load(a.labels)}
        L = [lab[p['problem_id']] for p in acc if p['problem_id'] in lab]
        rep['spread']['labelled'] = {
            'task_type': dist(collections.Counter(l.get('task_type') for l in L)),
            'n_asks': q([l.get('n_asks') or 0 for l in L]), 'inline': round(st.mean(bool(l.get('inline_material')) for l in L), 3),
            'persona': round(st.mean(bool(l.get('persona')) for l in L), 3),
            'points_to_location': round(st.mean(bool(l.get('points_to_location')) for l in L), 3),
            'format': dist(collections.Counter(l.get('output_format') for l in L))}
    s = json.dumps(rep, indent=1)
    print(s)
    if a.out:
        open(a.out, 'w').write(s + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Leave-one-author-out cross-validation of the sampler's smoothing strengths.

For each smoothed conditional in sample_assets.fit (anchor | bucket, action | works_on, domain | anchor,
origin | bucket, and the none-bucket action/domain), hold out every task by one author, refit the counts on
the rest, and score log P(held-out value | condition) under the same shrink-to-prior estimate the sampler uses,
for a grid of pseudo-count strengths k. Then check whether flattening (temperature) or capping prolific
authors' weight improves held-out likelihood.

  python3 smoothing_cv.py [--sampler-dir DIR] [--authors task_authors.json] [--lam 0.5]
"""
import argparse, collections, json, math, os, sys

K_GRID = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
TAU_GRID = [1.0, 1.25, 1.5, 2.0, 3.0]
CAPS = [None, 8, 4, 2]
EPS = 0.2  # pseudo-counts per vocabulary value in every backoff prior (the sampler uses 0.2 for anchor classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sampler-dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--authors', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_authors.json'))
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--a', default='TB2.1')
    ap.add_argument('--b', default='TB3.0')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.sampler_dir)
    import sample_assets as sa

    d = args.sampler_dir
    anchors = json.load(open(os.path.join(d, 'anchor_labels.json')))
    rows = sa.load(os.path.join(d, 'asset_labels.json'), os.path.join(d, 'anchor_buckets.json'),
                   os.path.join(d, 'anchor_labels.json'))
    who = {(x['set'], x['task']): x['names'][0] for x in json.load(open(args.authors))}
    for r, a in zip(rows, anchors):
        r['task'], r['author'] = a['task'], who[(a['set'], a['task'])]
    na, nb = (sum(r['set'] == s for r in rows) for s in (args.a, args.b))
    nbar = (1 - args.lam) * na + args.lam * nb
    set_w = {args.a: (1 - args.lam) * nbar / na, args.b: args.lam * nbar / nb}

    # (name, rows it applies to, condition, value, rows the backoff prior is tallied over, vocabulary)
    A = lambda r: r['bucket'] != 'none'
    N = lambda r: r['bucket'] == 'none'
    ALL = lambda r: True
    vocab = lambda f, keep=ALL: sorted({f(r) for r in rows if keep(r) and f(r) is not None})
    specs = [
        ('anchor | bucket', A, lambda r: r['bucket'], lambda r: r['anchor'], A, sa.CLASSES, 3, 'k_back'),
        ('action | works_on', A, lambda r: r['works_on'], lambda r: r['action'], ALL, vocab(lambda r: r['action']), 2, '2 (fixed)'),
        ('action | none', N, lambda r: 0, lambda r: r['action'], ALL, vocab(lambda r: r['action']), 2, '2 (fixed)'),
        ('domain | anchor', A, lambda r: r['anchor'], lambda r: r['domain'], ALL, vocab(lambda r: r['domain']), 2, '2 (fixed)'),
        ('domain | none', N, lambda r: 0, lambda r: r['domain'], ALL, vocab(lambda r: r['domain']), 2, '2 (fixed)'),
        ('origin | bucket', A, lambda r: r['bucket'], lambda r: r['origin'] or None, A,
         vocab(lambda r: r['origin'] or None, A), 3, 'k_back'),
    ]
    authors = sorted({r['author'] for r in rows})

    def weights(train, cap):
        n = collections.Counter(r['author'] for r in train)
        return [(set_w[r['set']] * (min(1.0, cap / n[r['author']]) if cap else 1.0), r) for r in train]

    def model(train_w, keep, cond, val, prior_keep, voc, k):
        prior = collections.Counter({v: EPS for v in voc})
        for w, r in train_w:
            if prior_keep(r) and val(r) is not None:
                prior[val(r)] += w
        s = sum(prior.values())
        prior = {v: c / s for v, c in prior.items()}
        by = collections.defaultdict(collections.Counter)
        for w, r in train_w:
            if keep(r) and val(r) is not None:
                by[cond(r)][val(r)] += w
        def p(x, tau=1.0):
            cnt = by.get(x, {})
            q = {v: cnt.get(v, 0) + k * prior[v] for v in voc}
            if tau != 1.0:
                q = {v: c ** (1 / tau) for v, c in q.items()}
            z = sum(q.values())
            return {v: c / z for v, c in q.items()}
        return p, by

    def cv(spec, k, tau=1.0, cap=None):
        name, keep, cond, val, prior_keep, voc, _, _ = spec
        per_task, per_author, novel_cond, novel_any, n = 0.0, [], 0, 0, 0
        for a in authors:
            test = [r for r in rows if r['author'] == a and keep(r) and val(r) is not None]
            if not test:
                continue
            train = [r for r in rows if r['author'] != a]
            p, by = model(weights(train, cap), keep, cond, val, prior_keep, voc, k)
            seen_any = {val(r) for r in train if keep(r)}
            lls = []
            for r in test:
                w = set_w[r['set']]
                ll = math.log(p(cond(r), tau)[val(r)])
                per_task += w * ll
                n += w
                lls.append(ll)
                novel_cond += w * (by.get(cond(r), {}).get(val(r), 0) == 0)
                novel_any += w * (val(r) not in seen_any)
            per_author.append(sum(lls) / len(lls))
        return per_task / n, sum(per_author) / len(per_author), novel_cond / n, novel_any / n

    results = {}
    for spec in specs:
        name, *_, current, current_name = spec
        grid = {k: cv(spec, k) for k in K_GRID}
        best_k = max(grid, key=lambda k: grid[k][0])
        best_k_author = max(grid, key=lambda k: grid[k][1])
        cur = cv(spec, current)
        taus = {t: cv(spec, best_k, tau=t)[0] for t in TAU_GRID}
        caps = {}
        for cap in CAPS:
            g = {k: cv(spec, k, cap=cap) for k in K_GRID}
            kb = max(g, key=lambda k: g[k][1])
            caps[str(cap)] = {'best_k': kb, 'per_author_ll': g[kb][1], 'per_task_ll': g[kb][0]}
        results[name] = {'n_values': len(spec[5]), 'current_k': current, 'current_name': current_name,
                         'current_ll': cur[0], 'best_k': best_k, 'best_ll': grid[best_k][0],
                         'best_k_per_author': best_k_author, 'novel_given_condition': cur[2],
                         'novel_overall': cur[3], 'grid': {str(k): v[0] for k, v in grid.items()},
                         'tau': {str(t): v for t, v in taus.items()}, 'author_cap': caps}

    print(f'Leave-one-author-out CV over {len(authors)} authors, {len(rows)} tasks ({args.a} + {args.b}, lam={args.lam}).')
    print('Mean held-out log-likelihood per task (higher is better).\n')
    print(f'{"conditional":<20} {"values":>6} {"current k":>10} {"ll":>7} {"best k":>7} {"ll":>7} {"gain":>6} '
          f'{"k (per-author)":>14} {"unseen|cond":>11} {"unseen":>7}')
    for name, r in results.items():
        print(f'{name:<20} {r["n_values"]:>6} {r["current_k"]:>10} {r["current_ll"]:>7.3f} {r["best_k"]:>7} '
              f'{r["best_ll"]:>7.3f} {r["best_ll"] - r["current_ll"]:>6.3f} {r["best_k_per_author"]:>14} '
              f'{r["novel_given_condition"]:>11.0%} {r["novel_overall"]:>7.0%}')
    print('\nTemperature at the best k (ll per task; tau=1 means no flattening):')
    for name, r in results.items():
        print(f'  {name:<20} ' + '  '.join(f'tau={t}: {v:.3f}' for t, v in r['tau'].items()))
    print('\nCapping prolific authors (training weight x min(1, cap/n_author)); per-author mean ll at its best k:')
    for name, r in results.items():
        print(f'  {name:<20} ' + '  '.join(f'cap={c}: {v["per_author_ll"]:.3f} (k={v["best_k"]})' for c, v in r['author_cap'].items()))
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

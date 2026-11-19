"""Spirit check (#359): generated tasks against the 600 scored tasks, class-level, per domain.

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.spirit --accepted 'GLOB' --out REPORT.json \
      [--label N]   # also label N generated tasks with the test labeller (Opus) for the workflow / trap mix

Compares: request length (closing sentences stripped), state size and record count before and after noise, number
of apps, app frequencies, workflow and trap mix (labels), targets and rule sources; plus the spec-vs-output
agreement (apps, size, length) and repeats (TF-IDF nearest neighbour of request + skeleton within the set).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path


def q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {'p10': xs[len(xs) // 10], 'median': statistics.median(xs), 'p90': xs[(9 * len(xs)) // 10], 'n': len(xs)}


def side_stats(items):
    """items: [{'domain', 'trigger', 'own_state', 'state'}] -> per-domain stats."""
    from m5.automationbench.check import count_records
    from m5.automationbench.taxonomy import strip_suffixes
    by = collections.defaultdict(list)
    for x in items:
        by[x['domain']].append(x)
    out = {}
    for d, xs in sorted(by.items()):
        apps = collections.Counter(a for x in xs for a in x['own_state'] if a != 'meta')
        out[d] = {'n': len(xs),
                  'trigger_words': q([len(strip_suffixes(x['trigger'])[0].split()) for x in xs]),
                  'bytes_pre_noise': q([len(json.dumps({k: v for k, v in x['own_state'].items() if k != 'meta'})) for x in xs]),
                  'bytes_post_noise': q([len(json.dumps({k: v for k, v in x['state'].items() if k != 'meta'})) for x in xs]),
                  'records_pre_noise': q([count_records(x['own_state']) for x in xs]),
                  'n_apps': q([len([a for a in x['own_state'] if a != 'meta']) for x in xs]),
                  'apps_top': {a: round(c / len(xs), 2) for a, c in apps.most_common(8)}}
    return out


def repeat_meter(items, thr=0.8):
    import math
    import re
    import numpy as np
    from m5.automationbench.decon import skeleton
    docs = [x['trigger'] + '\n' + skeleton(x['own_state']) for x in items]
    if len(docs) < 3:
        return None
    toks = []
    for d in docs:   # word 1-2 grams, sublinear tf, idf, l2 (TfidfVectorizer's defaults as used in decon)
        w = re.findall(r'\b\w\w+\b', d.lower())
        toks.append(collections.Counter(w + [a + ' ' + b for a, b in zip(w, w[1:])]))
    df = collections.Counter(t for c in toks for t in c)
    vocab = {t: i for i, t in enumerate(df)}
    n = len(docs)
    X = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, c in enumerate(toks):
        for t, k in c.items():
            X[i, vocab[t]] = (1 + math.log(k)) * (math.log((1 + n) / (1 + df[t])) + 1)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    S = X @ X.T
    np.fill_diagonal(S, 0)
    nn = S.max(axis=1)
    return {'n': len(docs), 'threshold': thr, 'share_nn_over': float((nn >= thr).mean()),
            'nn_median': float(np.median(nn)), 'nn_p90': float(np.quantile(nn, 0.9))}


def label_generated(items, n, out_dir):
    """The test labeller (label_test.py) on n generated tasks (no assertions shown)."""
    import random
    from m5.automationbench import label_test as L
    rows = random.Random(359).sample(items, min(n, len(items)))
    shaped = [{'task_name': x['name'], 'trigger': x['trigger'],
               'info': {'initial_state': x['own_state'], 'assertions': []}} for x in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    import concurrent.futures as cf
    batches = [shaped[i:i + 10] for i in range(0, len(shaped), 10)]
    with cf.ThreadPoolExecutor(4) as ex:
        files = list(ex.map(lambda ib: L.run_batch(ib[1], out_dir, ib[0]), enumerate(batches)))
    labs = [json.loads(l) for f in files if f for l in Path(f).read_text().splitlines()]
    return labs


def mix(labels, domain_of):
    from m5.automationbench.taxonomy import merge_workflow
    by = collections.defaultdict(lambda: {'workflow': collections.Counter(), 'traps': collections.Counter(), 'n': 0,
                                         'targets': [], 'docs': []})
    for l in labels:
        d = domain_of(l['task'])
        b = by[d]
        b['n'] += 1
        b['workflow'][merge_workflow(l['workflow'])] += 1
        b['traps'].update(t for t in set(l['traps']) if not t.startswith('other'))
        b['targets'].append(int(l.get('n_targets') or 0))
        b['docs'].append(int(l.get('n_policy_docs') or 0))
    return {d: {'n': b['n'], 'workflow': {k: round(v / b['n'], 2) for k, v in b['workflow'].most_common()},
                'traps_per_task': round(sum(b['traps'].values()) / b['n'], 2),
                'traps': {k: round(v / b['n'], 2) for k, v in b['traps'].most_common()},
                'targets_median': statistics.median(b['targets']), 'docs_median': statistics.median(b['docs'])}
            for d, b in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--accepted', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', type=int, default=0)
    ap.add_argument('--test-labels', default='/data/aai/m5/automationbench/fit/test_labels.jsonl')
    a = ap.parse_args()
    from m5.automationbench.upstream import DOMAINS, load
    test = []
    for d in DOMAINS:
        noisy = {r['task_name']: r for r in load(d, noise=True)}
        for r in load(d, noise=False):
            test.append({'name': r['task_name'], 'domain': d, 'trigger': r['trigger'], 'own_state': r['info']['initial_state'],
                         'state': noisy[r['task_name']]['info']['initial_state']})
    gen = []
    for p in sorted(glob.glob(a.accepted)):
        for line in open(p):
            r = json.loads(line)
            t = r['task']
            gen.append({'name': t['info']['task_name'], 'domain': r['spec']['domain'], 'trigger': t['prompt'][1]['content'],
                        'own_state': r['own_state'], 'state': t['info']['initial_state'], 'spec': r['spec'],
                        'rounds': r['rounds'], 'warnings': r.get('warnings') or []})
    rep = {'test': side_stats(test), 'generated': side_stats(gen), 'repeat_meter_generated': repeat_meter(gen),
           'rounds': dict(collections.Counter(g['rounds'] for g in gen)),
           'warnings': dict(collections.Counter(w.split(' ')[0] + ' ' + w.split(' ')[1] for g in gen for w in g['warnings']))}
    test_labels = [json.loads(x) for x in open(a.test_labels)]
    rep['labels_test'] = mix(test_labels, lambda n: n.split('.')[0])
    if a.label:
        dom = {g['name']: g['domain'] for g in gen}
        labs = label_generated(gen, a.label, Path(a.out).with_suffix('.labels'))
        rep['labels_generated'] = mix(labs, lambda n: dom.get(n, n.split('.')[0]))
        # how often the labeller finds the traps the spec asked for
        spec_traps = {g['name']: set(g['spec']['traps']) for g in gen}
        hit = [len(spec_traps[l['task']] & set(l['traps'])) / max(1, len(spec_traps[l['task']])) for l in labs if l['task'] in spec_traps]
        rep['spec_traps_found_by_labeller'] = round(statistics.mean(hit), 3) if hit else None
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for d in DOMAINS:
        t, g = rep['test'].get(d), rep['generated'].get(d)
        if not g:
            continue
        f = lambda s, k: (s[k] or {}).get('median') if s else None  # noqa: E731
        print(f'{d:11s} words {f(t, "trigger_words")}/{f(g, "trigger_words")}  pre-bytes {f(t, "bytes_pre_noise")}/{f(g, "bytes_pre_noise")}'
              f'  post-bytes {f(t, "bytes_post_noise")}/{f(g, "bytes_post_noise")}  records {f(t, "records_pre_noise")}/{f(g, "records_pre_noise")}'
              f'  apps {f(t, "n_apps")}/{f(g, "n_apps")}  (test/generated medians, n={g["n"]})')
    print('repeat meter:', rep['repeat_meter_generated'], 'rounds:', rep['rounds'])


if __name__ == '__main__':
    main()

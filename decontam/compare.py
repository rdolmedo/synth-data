"""TB2.1 pass@1 of the decontaminated CalibForge run, its controls and the reference runs, with paired task-bootstrap CIs.

  python -m decontam.compare --exclusions DIR/decontamination-rewrites.json [--benchmark tblite] [--out FILE]

Subsets of TB2.1 tasks: all 89; without look-alikes (every TB2.1 task with a CalibForge task judged its duplicate or
rewrite: judge or propagated); without the lexical-flag references (the preliminary analysis's 27-task cut; many
lexical flags are shared Dockerfiles, and the ones that are rewrites were judged too).
"""
import argparse
import json
from pathlib import Path

import numpy as np

EVAL = Path('/data/aai/tbenchtrain/eval_q35')
RUNS = {'decon': 'tb21p180-super-subset-4b-calibforge-decon-1b', 'base': 'tb21p180-base-4b',
        'tmax': 'tb21p180-super-subset-4b-tmax-1b', 'cf': 'tb21p180-super-subset-4b-calibforge-1b',
        'mix3b': 'tb21p180-super-scaling-4b-3b', 'rts': 'tb21p180-super-rts-4b-1b',
        # Controls (same fast path as decon): original CF-1B data retrained; TMax + every record of the excluded tasks;
        # the same TMax prefix + per-category matched clean CalibForge tasks (same records and tokens).
        'cf_fast': 'tb21p180-super-subset-4b-calibforge-1b-fastpath',
        'tmax_contam': 'tb21p180-super-subset-4b-tmax-cfcontam-1b', 'tmax_clean': 'tb21p180-super-subset-4b-tmax-cfclean-1b'}
CONTRASTS = [('cf', 'base'), ('cf', 'tmax'), ('mix3b', 'base'), ('cf_fast', 'cf'), ('cf_fast', 'base'),
             ('tmax_contam', 'tmax_clean'), ('tmax_contam', 'tmax'), ('tmax_clean', 'tmax'), ('tmax_contam', 'base'),
             ('tmax_clean', 'base')]


def tblite(runs):
    """TB-Lite names of the same runs (tblp180-*); no look-alike split (CalibForge did not target TB-Lite)."""
    return {k: r.replace('tb21p180-', 'tblp180-', 1) for k, r in runs.items()}


def per_task(run):
    return {t: (v or 0.0) for t, v in json.loads((EVAL / run / 'scores.json').read_text())['per_task'].items()}


def boot(diff, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(a):
    global RUNS
    if a.benchmark == 'tblite':
        RUNS = tblite(RUNS)
    ex = json.loads(Path(a.exclusions).read_text())['excluded']
    judged = {v['tb'] for e in ex.values() for v in e['verdicts'] if v['source'] in ('judge', 'propagated')}
    lexical = {v['tb'] for e in ex.values() for v in e['verdicts'] if v['source'] == 'lexical'}
    P = {k: per_task(r) for k, r in RUNS.items() if (EVAL / r / 'scores.json').exists()}
    tasks = sorted(P['base'])
    subsets = {'all': tasks, 'no_lookalikes': [t for t in tasks if t not in judged],
               'lookalikes': [t for t in tasks if t in judged], 'no_lexical_refs': [t for t in tasks if t not in lexical]}
    if a.benchmark == 'tblite':
        subsets = {'all': tasks}
    out = {'n_tasks': {k: len(v) for k, v in subsets.items()},
           'lookalike_tasks': sorted(judged & set(tasks)), 'pass_at_1': {}, 'paired': {}}
    contrasts = [('decon', k) for k in P if k != 'decon'] + CONTRASTS
    for name, ts in subsets.items():
        out['pass_at_1'][name] = {k: round(100 * np.mean([P[k][t] for t in ts]), 1) for k in P}
        for x, y in contrasts:
            if x in P and y in P:
                d = np.array([P[x][t] - P[y][t] for t in ts])
                lo, hi = boot(d)
                out['paired'].setdefault(name, {})[f'{x}-{y}'] = [round(100 * d.mean(), 1), round(100 * lo, 1), round(100 * hi, 1)]
        # Removed tasks vs clean replacements, same stack, both swaps averaged per task: (cf_fast - decon) and
        # (tmax_contam - tmax_clean).
        if all(k in P for k in ('cf_fast', 'decon', 'tmax_contam', 'tmax_clean')):
            d = np.array([(P['cf_fast'][t] - P['decon'][t] + P['tmax_contam'][t] - P['tmax_clean'][t]) / 2 for t in ts])
            lo, hi = boot(d)
            out['paired'][name]['pooled_removed-clean'] = [round(100 * d.mean(), 1), round(100 * lo, 1), round(100 * hi, 1)]
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclusions', required=True)
    ap.add_argument('--out')
    ap.add_argument('--benchmark', choices=['tb2.1', 'tblite'], default='tb2.1')
    main(ap.parse_args())

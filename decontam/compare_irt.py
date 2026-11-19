"""Paired contrasts of the CalibForge decontamination runs on the IRT-subset k=9 evals (user, 2026-10-28).

  python -m decontam.compare_irt --exclusions DIR/decontamination-rewrites.json [--benchmark tblite] [--old-prefix P] [--out FILE]

The subset is the ~1/3 of tasks with the most 2PL-IRT information at these runs' abilities
(/data/aai/tbenchtrain/decontam_calibforge/irt_k9/irt_select.py, fitted on every Pi-agent eval of the benchmark);
the k=9 attempts use a fresh seed namespace, so they are new samples.
- primary: the subset, fresh k=9 attempts only (the selection data never enter the estimate);
- full: the whole benchmark, subset tasks with fresh + k=3 attempts pooled, other tasks with their k=3 attempts.
Each contrast: pass@1 difference, 20k-resample task bootstrap 95% CI, and p under eval-sampling noise alone (per task, the
two runs' attempts are exchangeable: hypergeometric reassignment, 20k draws).
"""
import argparse
import json
from pathlib import Path

import numpy as np

EVAL = Path('/data/aai/tbenchtrain/eval_q35')
RUNS = {'decon': 'super-subset-4b-calibforge-decon-1b', 'cf_fast': 'super-subset-4b-calibforge-1b-fastpath',
        'cf': 'super-subset-4b-calibforge-1b', 'tmax_contam': 'super-subset-4b-tmax-cfcontam-1b',
        'tmax_clean': 'super-subset-4b-tmax-cfclean-1b'}
CONTRASTS = [('cf_fast', 'decon'), ('cf', 'decon'), ('cf_fast', 'cf'), ('tmax_contam', 'tmax_clean')]
PREFIX = {'tb2.1': 'tb21p180', 'tblite': 'tblp180'}


def counts(run):
    """task -> [passes, attempts] (score >= 1 passes; tail cuts and unresolved-as-failure as score.py)."""
    out = {}
    for k, r in json.loads((EVAL / run / 'scores.json').read_text())['per_attempt_records'].items():
        c = out.setdefault(k.rsplit('__a', 1)[0], [0, 0])
        c[0] += (r.get('score') or 0) >= 1
        c[1] += 1
    return out


def contrast(C, x, y, tasks, rng, n=20000):
    """C[run][task] = (passes, attempts). Per-task rate differences, averaged over tasks."""
    sx, nx = np.array([C[x][t][0] for t in tasks], float), np.array([C[x][t][1] for t in tasks], float)
    sy, ny = np.array([C[y][t][0] for t in tasks], float), np.array([C[y][t][1] for t in tasks], float)
    d = sx / nx - sy / ny
    idx = rng.integers(0, len(d), (n, len(d)))
    lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
    tot = (sx + sy).astype(int)
    draw = rng.hypergeometric(np.broadcast_to(tot, (n, len(d))), np.broadcast_to((nx + ny - tot).astype(int), (n, len(d))),
                              np.broadcast_to(nx.astype(int), (n, len(d))))
    null = (draw / nx - (tot - draw) / ny).mean(1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())
    return [round(100 * d.mean(), 1), round(100 * lo, 1), round(100 * hi, 1), round(p, 3)], d


def main(a):
    pre = PREFIX[a.benchmark]
    new = {k: counts(f'{pre}irt-{r}') for k, r in RUNS.items()}
    old = {k: counts(f'{a.old_prefix or pre}-{r}') for k, r in RUNS.items()}
    sub = sorted(new['decon'])
    assert all(sorted(new[k]) == sub for k in new), 'runs disagree on the subset'
    full = sorted(old['decon'])
    pooled = {k: {t: ([new[k][t][0] + old[k][t][0], new[k][t][1] + old[k][t][1]] if t in new[k] else old[k][t]) for t in full}
              for k in RUNS}
    sets = {'subset_fresh': (new, sub), 'full_pooled': (pooled, full)}
    if a.benchmark == 'tb2.1':
        ex = json.loads(Path(a.exclusions).read_text())['excluded']
        judged = {v['tb'] for e in ex.values() for v in e['verdicts'] if v['source'] in ('judge', 'propagated')}
        sets['subset_fresh_lookalikes'] = (new, [t for t in sub if t in judged])
        sets['subset_fresh_others'] = (new, [t for t in sub if t not in judged])
    out = {'benchmark': a.benchmark, 'subset_tasks': len(sub), 'full_tasks': len(full),
           'attempts_per_task_fresh': {k: sorted({c[1] for c in v.values()}) for k, v in new.items()},
           'n_tasks': {}, 'pass_at_1': {}, 'paired': {}}
    for name, (C, ts) in sets.items():
        rng = np.random.default_rng(0)
        out['n_tasks'][name] = len(ts)
        out['pass_at_1'][name] = {k: round(100 * float(np.mean([C[k][t][0] / C[k][t][1] for t in ts])), 1) for k in RUNS}
        res, diffs = {}, {}
        for x, y in CONTRASTS:
            res[f'{x}-{y}'], diffs[(x, y)] = contrast(C, x, y, ts, rng)
        # removed tasks vs clean replacements, both same-stack swaps averaged per task (as decontam.compare)
        d = (diffs[('cf_fast', 'decon')] + diffs[('tmax_contam', 'tmax_clean')]) / 2
        lo, hi = np.quantile(d[rng.integers(0, len(d), (20000, len(d)))].mean(1), [0.025, 0.975])
        res['pooled_removed-clean'] = [round(100 * d.mean(), 1), round(100 * lo, 1), round(100 * hi, 1), None]
        out['paired'][name] = res
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclusions', required=True)
    ap.add_argument('--benchmark', choices=sorted(PREFIX), default='tb2.1')
    ap.add_argument('--old-prefix', help='prefix of the k=3 evals (default: the benchmark prefix)')
    ap.add_argument('--out')
    main(ap.parse_args())

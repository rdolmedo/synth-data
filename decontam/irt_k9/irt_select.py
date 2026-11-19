"""2PL IRT on the eval_q35 Pi-agent runs; pick the ~1/3 of tasks most informative at the abilities of the target runs.

  python irt_select.py BENCH OUT_JSON [--harness pi,pi180]

P(pass | run m, task i) = sigmoid(a_i (theta_m - b_i)), binomial counts per (run, task) from per_attempt_records
(score >= 1 passes, tail cuts fail, as score.py). MAP with theta ~ N(0,1), b ~ N(0, 3^2), log a ~ N(0, 0.5^2), Adam.
Selection: mean Fisher information a^2 P (1-P) over the target runs' thetas (symmetric in the runs: it never looks at
which target run passed what).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/aai/tbenchtrain/eval_q35')
TARGETS = ['super-subset-4b-calibforge-decon-1b', 'super-subset-4b-calibforge-1b-fastpath', 'super-subset-4b-calibforge-1b',
           'super-subset-4b-tmax-cfcontam-1b', 'super-subset-4b-tmax-cfclean-1b']
PREFIX = {'tb2.1': 'tb21p180-', 'tblite': 'tblp180-'}


def load(bench, harnesses):
    runs, data = [], {}
    for f in sorted(glob.glob(str(ROOT / '*/config.json'))):
        run = Path(f).parent.name
        cfg = json.load(open(f))
        if run.startswith('debug') or cfg.get('benchmark', 'tblite') != bench or cfg.get('harness', 'pi') not in harnesses:
            continue
        if not (ROOT / run / 'scores.json').exists():
            continue
        s = json.load(open(ROOT / run / 'scores.json'))
        if s.get('coverage', 0) < 0.99:
            continue
        cnt = {}
        for k, r in s['per_attempt_records'].items():
            t = k.rsplit('__a', 1)[0]
            p, n = cnt.get(t, (0, 0))
            cnt[t] = (p + ((r.get('score') or 0) >= 1), n + 1)
        runs.append(run)
        data[run] = cnt
    tasks = sorted(set.intersection(*(set(c) for c in data.values())))
    S = np.array([[data[r][t][0] for t in tasks] for r in runs], float)
    N = np.array([[data[r][t][1] for t in tasks] for r in runs], float)
    return runs, tasks, S, N


def fit(S, N, iters=20000, lr=0.02, seed=0):
    M, I = S.shape
    rng = np.random.default_rng(seed)
    th, b, al = rng.normal(0, 0.1, M), np.zeros(I), np.zeros(I)
    params = [th, b, al]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    for it in range(1, iters + 1):
        a = np.exp(al)
        z = a[None, :] * (th[:, None] - b[None, :])
        P = 1 / (1 + np.exp(-z))
        r = S - N * P
        g = [(r * a[None, :]).sum(1) - th,
             -(r * a[None, :]).sum(0) - b / 9.0,
             (r * a[None, :] * (th[:, None] - b[None, :])).sum(0) - al / 0.25]
        for j, p in enumerate(params):   # Adam ascent on the log posterior
            m1[j] = 0.9 * m1[j] + 0.1 * g[j]
            m2[j] = 0.999 * m2[j] + 0.001 * g[j] ** 2
            p += lr * (m1[j] / (1 - 0.9 ** it)) / (np.sqrt(m2[j] / (1 - 0.999 ** it)) + 1e-8)
    a = np.exp(al)
    P = 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    ll = float((S * np.log(P + 1e-12) + (N - S) * np.log(1 - P + 1e-12)).sum())
    return th, b, a, ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', choices=['tb2.1', 'tblite'])
    ap.add_argument('out')
    ap.add_argument('--harness', default='pi,pi180')
    ap.add_argument('--frac', type=float, default=1 / 3)
    a = ap.parse_args()
    runs, tasks, S, N = load(a.bench, set(a.harness.split(',')))
    th, b, disc, ll = fit(S, N)
    tgt = [runs.index(PREFIX[a.bench] + r) for r in TARGETS]
    Pt = 1 / (1 + np.exp(-disc[None, :] * (th[tgt][:, None] - b[None, :])))
    info = (disc[None, :] ** 2 * Pt * (1 - Pt)).mean(0)
    order = np.argsort(-info)
    k = int(round(a.frac * len(tasks)))
    pick = [tasks[i] for i in order[:k]]
    # Share of the target runs' k=3 attempt variance (sum over tasks of p(1-p)) on the picked tasks.
    pt = S[tgt] / N[tgt]
    var = (pt * (1 - pt)).sum(0)
    out = {'bench': a.bench, 'harnesses': a.harness, 'n_runs': len(runs), 'n_tasks': len(tasks), 'k': k, 'loglik': round(ll, 1),
           'targets': {r: round(float(th[i]), 3) for r, i in zip(TARGETS, tgt)},
           'theta_range_all': [round(float(th.min()), 2), round(float(th.max()), 2)],
           'picked': pick,
           'info_share_picked': round(float(info[order[:k]].sum() / info.sum()), 3),
           'target_attempt_var_share_picked': round(float(var[order[:k]].sum() / var.sum()), 3),
           'target_pass_rate_picked': round(float(pt[:, order[:k]].mean()), 3),
           'target_pass_rate_all': round(float(pt.mean()), 3),
           'items': {tasks[i]: {'a': round(float(disc[i]), 3), 'b': round(float(b[i]), 3), 'info': round(float(info[i]), 4),
                                'target_pass': round(float(pt[:, i].mean()), 3), 'all_pass': round(float((S[:, i] / N[:, i]).mean()), 3)}
                     for i in order},
           'runs': {r: round(float(t), 3) for r, t in zip(runs, th)}}
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('items', 'runs', 'picked')}, indent=1))


if __name__ == '__main__':
    main()

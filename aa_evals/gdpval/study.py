"""GDPval-AA judge study (#183): how the judge arms that judged a run compare, on the same stored matches.

  python -m aa_evals.gdpval.study --run-id RUN [--arms A,B,...]    # default: every matches/<arm>.jsonl in the run

Per arm: n, the model's win rate vs the expert (anchored tasks only, i.e. not the 35 without an expert deliverable),
ties, position (the model's score as A and as B, order-flip rate), judge errors, and the implied Elo with the expert
at 968. Per pair of arms, on the (attempt, order) judgments both made: raw agreement and Cohen's kappa on
{model wins, tie, expert wins}, plus per-attempt agreement on the two-order mean. Writes study.json next to summary.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

from aa_evals.gdpval import elo
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.run import RUNS, key, load_outcomes, read_jsonl


def load(run_dir, arm, ok_keys, anchored):
    recs = {}
    for r in read_jsonl(run_dir / 'matches' / f'{arm}.jsonl'):
        if r.get('winner') and r['key'] in ok_keys and r['task_id'] in anchored:
            recs[(r['key'], r['order'])] = r['model_score']
    errors = sum(1 for r in read_jsonl(run_dir / 'matches' / f'{arm}.jsonl') if r.get('error') and not r.get('winner'))
    return recs, errors


def kappa(a: list, b: list) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def arm_stats(recs, errors):
    if not recs:
        return {'n': 0, 'judge_errors': errors}
    s = list(recs.values())
    ma = [v for (k, o), v in recs.items() if o == 'MA']
    mb = [v for (k, o), v in recs.items() if o == 'MB']
    flips = [recs[(k, 'MA')] != recs[(k, 'MB')] for k in {k for k, _ in recs} if (k, 'MA') in recs and (k, 'MB') in recs]
    w = sum(s) / len(s)
    return {'n': len(s), 'winrate_vs_expert': round(w, 4), 'implied_elo': round(elo.elo_from_winrate(w), 1),
            'ties': sum(v == 0.5 for v in s), 'model_as_A': round(sum(ma) / len(ma), 4) if ma else None,
            'model_as_B': round(sum(mb) / len(mb), 4) if mb else None,
            'order_flip_rate': round(sum(flips) / len(flips), 4) if flips else None, 'judge_errors': errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--arms', default=None)
    a = ap.parse_args()
    run_dir = RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
    outs = load_outcomes(run_dir, keys)
    ok_keys = {k for k in keys if outs.get(k, {}).get('status') == 'ok'}
    anchored = {t.task_id for t in T.load_tasks() if t.deliverable_files}
    arms = a.arms.split(',') if a.arms else sorted(p.stem for p in (run_dir / 'matches').glob('*.jsonl'))
    data = {arm: load(run_dir, arm, ok_keys, anchored) for arm in arms}
    out = {'run': a.run_id, 'arms': {arm: arm_stats(*data[arm]) for arm in arms}, 'pairs': {}}
    for x, y in itertools.combinations(arms, 2):
        rx, ry = data[x][0], data[y][0]
        common = sorted(set(rx) & set(ry))
        att = sorted({k for k, _ in common if (k, 'MA') in rx and (k, 'MB') in rx and (k, 'MA') in ry and (k, 'MB') in ry})
        mx = [(rx[(k, 'MA')] + rx[(k, 'MB')]) / 2 for k in att]
        my = [(ry[(k, 'MA')] + ry[(k, 'MB')]) / 2 for k in att]
        out['pairs'][f'{x} vs {y}'] = {
            'n_judgments': len(common),
            'agreement': round(sum(rx[c] == ry[c] for c in common) / len(common), 4) if common else None,
            'kappa': kappa([rx[c] for c in common], [ry[c] for c in common]),
            'n_attempts_both_orders': len(att),
            'attempt_agreement': round(sum(p == q for p, q in zip(mx, my)) / len(att), 4) if att else None}
    (run_dir / 'study.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

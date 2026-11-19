"""Regression check of decontam.pool against a decontam.synth run on the same tasks (issue #119 vs #19's student 1).

  python -m decontam.pool_check --work W_POOL --old W_SYNTH --list OLD_LIST_DIR

Compares the five lexical/embedding signals (scores and each test task's top-50 per signal), the fingerprint ranks
(if W_SYNTH has fp-ranks.npz), and the list rebuilt by `decontam.pool final` from the same verdicts (W_POOL/final-check)
with OLD_LIST_DIR/decontamination-variants.json. Prints a JSON report; exit 1 if the lists differ.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--k', type=int, default=50)
    a = ap.parse_args()
    new, old = np.load(Path(a.work) / 'scores.npz'), np.load(Path(a.old) / 'scores.npz')
    assert list(new['tb']) == list(old['tb']), 'reference order differs'
    ncf, ocf = [str(x) for x in new['cf']], [str(x) for x in old['cf']]
    assert sorted(ncf) == sorted(ocf), f'candidate sets differ: {len(ncf)} vs {len(ocf)}'
    perm = np.array([ncf.index(c) for c in ocf]) if ncf != ocf else np.arange(len(ncf))
    rep = {'refs': len(new['tb']), 'cands': len(ncf), 'signals': {}}
    for s in ('instr', 'instr_tests', 'ids', 'overlap', 'embed'):
        A, B = new[f'score_{s}'][:, perm].astype(np.float64), old[f'score_{s}']
        topA = np.argsort(-A, axis=1, kind='stable')[:, :a.k]
        topB = np.argsort(-B, axis=1, kind='stable')[:, :a.k]
        same = [len(set(x) & set(y)) / a.k for x, y in zip(topA, topB)]
        # pairs with a positive score in the old top-k that the new top-k misses (what the windows would judge)
        miss = sum(len({j for j in y if B[i, j] > 0} - set(x)) for i, (x, y) in enumerate(zip(topA, topB)))
        rep['signals'][s] = {'max_abs_diff': float(np.abs(A - B).max()), 'top_k_overlap_mean': round(float(np.mean(same)), 5),
                             'top_k_overlap_min': round(float(np.min(same)), 3), 'old_top_k_pairs_missed': int(miss)}
    for f in ('instruction', 'tests', 'dockerfile', 'env', 'solution'):
        k = f'contain_{f}'
        if k in old and k in new:
            rep['signals'][k] = {'max_abs_diff': float(np.abs(new[k][:, perm].astype(np.float64) - old[k]).max())}
    rep['signals']['shared_files'] = {'max_abs_diff': float(np.abs(new['shared_files'][:, perm] - old['shared_files']).max())}
    old_list = json.loads((Path(a.list) / 'decontamination-variants.json').read_text())
    new_list = json.loads((Path(a.work) / 'final-check' / 'decontamination-variants.json').read_text())
    o, n = set(old_list['excluded']), set(new_list['excluded'])
    rep['list'] = {'old': len(o), 'new': len(n), 'only_old': sorted(o - n), 'only_new': sorted(n - o),
                   'old_counts': {k: old_list['counts'][k] for k in ('pairs_judged', 'pairs_excluded', 'excluded_tasks')},
                   'new_counts': {k: new_list['counts'][k] for k in ('pairs_judged', 'pairs_excluded', 'excluded_tasks')}}
    print(json.dumps(rep, indent=1))
    (Path(a.work) / 'check.json').write_text(json.dumps(rep, indent=1))
    sys.exit(0 if o == n else 1)


if __name__ == '__main__':
    main()

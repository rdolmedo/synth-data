"""Cost model of the thorough decontamination on a pool (issue #119): judge pairs, DeepSeek server-hours and the wall
time left after generation ends, for one pass at the end vs the main pass near the end (default 0.9 of the planned
pool, or the round's STOP) plus a last pass at END (the plan), or any other snapshot schedule.

  python -m decontam.pool_cost --tasks 146000 [--snapshots 0.9,1] [--servers 2] [--measured W]

Every number below is an assumption with a default measured on student 1 (#19, 7,296 tasks) or the #125 pilot; pass --measured W (a
decontam.pool work dir after a real snapshot, e.g. #125's pilot) to replace the pair counts with that run's.

Pair model per snapshot of n tasks:
  window pairs   = W50 (test tasks x signals x top-50, after overlap between signals; ~flat in n: the windows are per
                   test task, not per candidate) + LITE (TB-Lite top-10 arm)
  forced pairs   = min(F_RATE * n, F_CAP) (spec-detail pairs, windowed: <= 50 per TB2.1 task)
  extra windows  = flagged pairs replaced by the next window: FLAG * n * REFILL
  new at snapshot i (incremental) = window pairs * (1 - RETAIN * n_{i-1} / n_i)   (pairs already judged are skipped;
                   RETAIN = share of an earlier window still in the window of a pool g x larger, times 1/g)
Samples = 2 per pair + 6 more for each pair with a non-NO label (NONNO share). Server throughput: SAMPLES_PER_H per
4xB200 DeepSeek server at --concurrency 512 (#19 fp pass: 79,542 samples in 7,452 s).
CPU (one 32-core job): READ_S per task to read a chunk (features, bundles, embeddings), SIGNAL_S per task per snapshot.
"""
import argparse
import json
from pathlib import Path

A = {
    'W50': (27803, 'pairs in the top-50 windows of the five windowed signals, forced-only pairs excluded (student 1,\n'
            '           2026-11-04: 28,346 first-pass fp + nofp pairs minus 543 forced-only; six signals were 37,812)'),
    'LITE': (3119, 'TB-Lite no-fingerprint arm, top-10 of the five windowed signals (student 1: 3,119; six: 4,580)'),
    'F_RATE': (0.374, 'forced pairs per task before windowing (student 1: 2,730 / 7,296)'),
    'F_CAP': (4450, 'forced pairs after windowing: 89 TB2.1 tasks x 50'),
    'FLAG': (0.00315, 'flagged share of tasks (#125 pilot: 11 / 3,496; student 1, v1 vocabulary: 36 / 7,296)'),
    'REFILL': (5.0, 'pairs judged per flagged task to refill the windows (one per windowed signal it sat in)'),
    'RETAIN': (1.0, 'share of an earlier snapshot window kept in a pool g x larger, times 1/g (1 = random order)'),
    'NONNO': (0.00066, 'pairs with a non-NO label among their first two samples (#125 pilot; student 1: 0.0016)'),
    'SAMPLES_PER_H': (46400, 'judge samples per hour per 4xB200 DeepSeek server (#125, all arms; #19 fp pass: 38,426)'),
    'B200_PER_SERVER': (4, 'B200s per DeepSeek server'),
    'READ_S': (2.2, 'CPU seconds per task to read it, mostly the bge-small embedding (#125 pilot: <= 2.2; student-1\n'
               '            regression and the real-asset mock: ~0.7-1.5)'),
    'SIGNAL_S': (0.0025, 'CPU seconds per task per snapshot for the signals (student-1 regression: tfidf + ids + overlap\n'
                 '              16 s at 7,296 tasks, + ~150 s fixed for loading torch/the model and ranks)'),
}


def model(n_list, v):
    rows, prev = [], 0
    for n in n_list:
        windows = v['W50'] + v['LITE'] + min(v['F_RATE'] * n, v['F_CAP'])
        new = windows * (1 - v['RETAIN'] * prev / n) + v['FLAG'] * (n - prev) * v['REFILL']
        samples = new * (2 + 6 * v['NONNO'])
        rows.append({'tasks': n, 'new_pairs': round(new), 'samples': round(samples),
                     'server_h': round(samples / v['SAMPLES_PER_H'], 2)})
        prev = n
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', type=int, required=True, help='final pool size')
    ap.add_argument('--snapshots', default='0.9,1', help='fractions of the pool at which snapshots run')
    ap.add_argument('--servers', type=int, default=2, help='DeepSeek servers the judge shares')
    ap.add_argument('--measured', help='decontam.pool work dir: take W50/LITE/F_CAP from its first snapshot')
    for k, (d, h) in A.items():
        ap.add_argument(f'--{k.lower().replace("_", "-")}', type=float, default=d, help=h)
    a = ap.parse_args()
    v = {k: getattr(a, k.lower()) for k in A}
    if a.measured:
        first = sorted((Path(a.measured) / 'passes').glob('*-00'))[0]
        n = {arm: sum(1 for _ in open(p)) for arm in ('fp', 'nofp', 'lite')
             if (p := first / f'pairs-{arm}.jsonl').exists()}
        v['W50'], v['LITE'] = n.get('fp', 0) + n.get('nofp', 0), n.get('lite', 0)
    one = model([a.tasks], v)
    inc = model([round(float(f) * a.tasks) for f in a.snapshots.split(',')], v)
    out = {'assumptions': {k: v[k] for k in A},
           'one_pass_at_end': {**one[0], 'wall_h_after_generation': round(one[0]['server_h'] / a.servers, 2)},
           'scheduled': {'snapshots': inc, 'server_h_total': round(sum(r['server_h'] for r in inc), 2),
                           'wall_h_after_generation': round(inc[-1]['server_h'] / a.servers, 2)},
           'b200_h': {'one_pass': round(one[0]['server_h'] * v['B200_PER_SERVER'], 1),
                      'scheduled': round(sum(r['server_h'] for r in inc) * v['B200_PER_SERVER'], 1)},
           'cpu_h': {'read_once': round(v['READ_S'] * a.tasks / 3600, 2),
                     'signals_one_pass': round(v['SIGNAL_S'] * a.tasks / 3600, 2),
                     'signals_scheduled': round(sum(v['SIGNAL_S'] * r['tasks'] for r in inc) / 3600, 2)}}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

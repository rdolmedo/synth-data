"""Gold self-test: grade every scored step with the dataset's own ground_truth_code (SciCode-Verified ships it; the
upstream test split does not). A sound step passes with its gold; a patch must keep that true.

  python -m aa_evals.scicode.goldcheck --dataset verified-v2 [--envs np126=local:/path/python,...] [--steps 72.6,..]
          [--jobs 16] [--out file.json]

Gold code is trusted, so local: envs are fine here.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from . import data, grader


def gold_codes(ds: data.Dataset, prob: dict) -> list[str]:
    out = []
    for s in prob['sub_steps']:
        if s['step_number'] in data.SKIP_STEPS:
            out.append(data.skip_gold(s['step_number']))
        else:
            out.append(s.get('ground_truth_code') or '')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='verified-v2')
    ap.add_argument('--envs', default=None, help='label=env,... (default: the two SIFs)')
    ap.add_argument('--steps', default='')
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    envs = dict(x.split('=', 1) for x in a.envs.split(',')) if a.envs else dict(grader.DEFAULT_ENVS)
    ds = data.load(a.dataset)
    want = set(a.steps.split(',')) if a.steps else None
    jobs = []
    for prob in ds.problems:
        codes = gold_codes(ds, prob)
        for k, s in enumerate(prob['sub_steps']):
            sid = s['step_number']
            if sid in data.SKIP_STEPS or sid in ds.excluded_steps or (want and sid not in want):
                continue
            if not s.get('ground_truth_code'):
                print(f'{sid}: no ground_truth_code', flush=True)
                continue
            jobs.append((prob, k, codes))

    def one(job):
        prob, k, codes = job
        s = prob['sub_steps'][k]
        fn = lambda vendor, h5: grader.build_script(prob, k, codes, len(s['test_cases']), s['test_cases'],
                                                    vendor=vendor, h5=h5)
        r = grader.grade_step(fn, envs, ds.h5_for(s['step_number']), a.timeout, scratch=grader.scratch_dir())
        flag = 'PASS' if r['pass'] else 'FAIL'
        envs_s = ' '.join(f"{k}:{'P' if v['pass'] else 'F'}{v['secs']:.0f}s" for k, v in r['per_env'].items())
        print(f"{s['step_number']:>6} {flag} {envs_s} n_tests={len(s['test_cases'])}", flush=True)
        if not r['pass']:
            for lab, v in r['per_env'].items():
                print(f'   [{lab}] ' + v['tail'][-400:].replace('\n', '\n   '), flush=True)
        return s['step_number'], r

    with ThreadPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(one, jobs))
    n_pass = sum(r['pass'] for r in res.values())
    print(f'gold pass {n_pass}/{len(res)}; fails: {sorted(k for k, r in res.items() if not r["pass"])}')
    zero = [s['step_number'] for p in ds.problems for s in p['sub_steps']
            if s['step_number'] not in data.SKIP_STEPS and s['step_number'] not in ds.excluded_steps
            and not s['test_cases']]
    print(f'steps with no test cases: {zero}')
    if a.out:
        json.dump({'dataset': ds.pins, 'envs': envs, 'results': res, 'zero_test_steps': zero}, open(a.out, 'w'),
                  indent=1)


if __name__ == '__main__':
    main()

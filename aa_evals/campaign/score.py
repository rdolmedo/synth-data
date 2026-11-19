#!/usr/bin/env python3
"""AAI-open scores of a campaign's models (#481): each slot's score from the runs' own summaries, the aggregate
(aa_evals/common/aai_open.py: AA v5.0 slot weights, raw %, GDPval Elo through the Briefcase clamp) and the AAI-open
version every run recorded. A run on another version, a '+custom' run or an 'unrecorded' one (before #481) is named
next to its score, and the model's line says which versions it mixes.

  python -m aa_evals.campaign.score CONFIG [MODEL ...] [--stage full] [--scale raw|universe] [--json OUT]

Sources (the same as the v0 scorers, m5/prereg_aai_open/scores.py and the M5 ladder's sb2_score.py):
- TB2.1, TB-Sci-lite: pi_eval summary `score_resolved_only` (<AA_RUNS_ROOT>/<bench>/<run_id>);
- AutomationBench, CritPt: the collected run's summary `score`;
- HLE, Omniscience (+ non-hallucination rate), GDP.pdf (all-pass; mean pass reported): the `<run_id>-graded` summary;
- SciCode: main-problem / sub-step accuracy from the summary, or from items.jsonl while steps are still grading;
- GDPval: Bradley-Terry Elo of the run in its reference pool (<runs>/gdpval/_pools/<campaign>-refs-<model>), judge arm
  aai_open.JUDGE_ARM, expert anchored at 968 (aa_evals.gdpval.elo / pool).
A slot without a finished summary is missing: the aggregate is then out of the weights present (`points`), as the
v0 lean evals' 65-point score. `--scale universe` gives the c30 leaderboard convention (retained / universe).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

from aa_evals.campaign import launch as L
from aa_evals.common import aai_open

RUNS = Path('/data/aai/aa_evals/runs')   # the default; a campaign's grading.runs_root wins (as in grade.py)
PI_RUNS = Path(os.environ.get('AA_RUNS_ROOT', '/data/aai/aa_evals/runs_fallback'))
PI_BENCH = {'tb21': 'tb21', 'tbsci_lite': 'tbsci_lite'}
GRADED = ('hle', 'omniscience', 'gdppdf')


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def scicode_scores(run_dir: Path) -> tuple[float, float] | None:
    s = _load(run_dir / 'summary.json')
    if s and s.get('main_problem_accuracy') is not None:
        return s['main_problem_accuracy'], s['sub_step_accuracy']
    try:
        rows = [json.loads(x) for x in (run_dir / 'items.jsonl').read_text().splitlines() if x.strip()]
    except OSError:
        return None
    if not rows:
        return None
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['problem_id'], r['repeat'])].append(r['score'])
    return 100 * sum(all(v) for v in by.values()) / len(by), 100 * sum(r['score'] for r in rows) / len(rows)


def gdpval_elo(pool: Path) -> tuple[float, list[float]] | None:
    meta = _load(pool / 'pool.json')
    if not meta:
        return None
    from aa_evals.gdpval import elo as E, pool as P   # pandas / pyarrow: only when a pool exists
    runs = meta['runs']
    obs = P.observations(pool, aai_open.JUDGE_ARM, runs)
    r = E.fit([o for v in obs.values() for o in v])
    ci = E.bootstrap(obs, n=300)
    return float(r[runs[0]]), [float(ci[runs[0]][0]), float(ci[runs[0]][1])]


def model_scores(C: L.Campaign, model: str, stage: str, runs: Path = RUNS, pi_runs: Path = PI_RUNS) -> dict:
    """{'scores': slot -> value, 'versions': suite -> recorded label, 'extra': reported but unweighted values}."""
    scores, versions, extra = {}, {}, {}
    for suite in L.model_suites(C, model):
        rid = C.run_id(model, suite, stage)
        if suite in PI_BENCH:
            d = pi_runs / PI_BENCH[suite] / rid
            s = _load(d / 'summary.json')
            if s and s.get('score_resolved_only') is not None:
                scores[suite] = s['score_resolved_only']
        elif suite in ('automationbench', 'critpt'):
            d = runs / suite / rid
            s = _load(d / 'summary.json')
            if s and s.get('score') is not None:
                scores[suite] = s['score']
        elif suite in GRADED:
            d = runs / suite / rid
            s = _load(runs / suite / f'{rid}-graded' / 'summary.json')
            if s and s.get('score') is not None:
                key = {'hle': 'hle', 'omniscience': 'omniscience', 'gdppdf': 'gdppdf_allpass'}[suite]
                scores[key] = s['score']
                nhr = ((s.get('metrics') or {}).get('non_hallucination_rate') or {}).get('score')
                if suite == 'omniscience' and nhr is not None:
                    scores['omni_nhr'] = nhr
                if suite == 'gdppdf':
                    extra['gdppdf_meanpass'] = s.get('mean_pass')
        elif suite == 'scicode':
            d = runs / suite / rid
            x = scicode_scores(d)
            if x:
                scores['scicode_main'], extra['scicode_sub'] = x
        elif suite == 'gdpval':
            d = runs / suite / rid
            x = gdpval_elo(runs / 'gdpval' / '_pools' / (f'{C.name}-refs-{model}' + ('-canary' if stage == 'canary' else '')))
            if x:
                scores['gdpval'], extra['gdpval_ci95'] = x
        if suite in GRADED and (runs / suite / f'{rid}-graded' / 'config.json').exists():
            d = runs / suite / f'{rid}-graded'   # the graded copy adds grade-stage deviations (judge arm)
        v = aai_open.of_run(d)
        if v == 'unrecorded' and suite in ('automationbench', 'critpt', 'gdpval'):   # collected dirs: their Pi run
            v = aai_open.of_run(pi_runs / f'{suite}_pi' / rid)
        versions[suite] = v
    return {'scores': scores, 'versions': versions, 'extra': extra}


def report(C: L.Campaign, models: list[str], stage: str, scale: str, **kw) -> dict:
    out = {}
    for m in models:
        r = model_scores(C, m, stage, **kw)
        agg = aai_open.aggregate(r['scores'], scale)
        labels = sorted(set(r['versions'].values()))
        out[m] = {**r, 'aggregate': agg, 'aai_open_versions': labels}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('models', nargs='*')
    ap.add_argument('--stage', choices=['full', 'canary'], default='full')
    ap.add_argument('--scale', choices=['raw', 'universe'], default='raw')
    ap.add_argument('--json', type=Path)
    a = ap.parse_args(argv)
    C = L.Campaign(a.config)
    models = a.models or list(C.models)
    runs = Path(((C.cfg.get('grading') or {}).get('runs_root')) or RUNS)
    res = report(C, models, a.stage, a.scale, runs=runs)
    for m, r in res.items():
        agg = r['aggregate']
        mix = r['aai_open_versions']
        print(f"=== {m}: AAI-open {agg['score']:.2f} of {agg['points']} pts ({agg['scale']}); "
              f"versions {', '.join(mix)}" + (f"; missing {', '.join(agg['missing'])}" if agg['missing'] else ''))
        for k, w in aai_open.SLOTS:
            if k in r['scores']:
                suite = {'omni_nhr': 'omniscience', 'gdppdf_allpass': 'gdppdf', 'scicode_main': 'scicode'}.get(k, k)
                print(f"   {k:16s} {w:3d}  {r['scores'][k]:8.2f}   {r['versions'].get(suite, '')}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()

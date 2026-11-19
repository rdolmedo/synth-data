"""GDPval-AA model-vs-model pairs and the pooled Elo (#183; coordinator 2026-11-05, AA's own design).

AA's scale comes from model-vs-model pairs. Ours adds those to the per-run matches against the human expert's
deliverable, and fits one BT/Elo over everything with the expert pinned at 968 (elo.py):

  python -m aa_evals.gdpval.pool plan  --pool m36 --runs RUN1,RUN2,... [--k 2 --seed 731]
  python -m aa_evals.gdpval.pool judge --pool m36 --judge-arm ARM [--judge-endpoint SPEC] [--concurrency 64]
  python -m aa_evals.gdpval.pool fit   --pool m36 [--judge-arm ARM] [--bootstrap 1000]

plan: for every finished attempt (status ok, render done) of every run, on every task (the 35 tasks without an
  expert deliverable included), `k` opponents are drawn from the other runs' finished attempts on the same task. Each
  is a distinct other run where possible, with a uniformly drawn repeat, from a seed hashed from (seed, run, task,
  repeat). Unordered duplicates are merged. Each pair is judged in both A/B orders (half weight each). The plan is
  frozen in `pairs.jsonl`: plan once every run in the pool is complete, so the pairs don't depend on timing.
  Budget: about runs x attempts x k / 2 x 2 orders judge calls (6 models x 1,100 attempts, k=2: about 13k calls).
judge: the same judge prompt, budget and arms as the vs-expert matches (judge.py); matches/<arm>.jsonl.
fit: players are the run ids plus 'expert'. Observations are the pool matches plus every run's vs-expert matches
  for the same arm, minus the no-expert tasks. CIs come from a bootstrap over tasks. Output: summary.json, with each
  run's Elo, CI, AA's Elo for it, and the rank correlation with AA's.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

from aa_evals.gdpval import elo
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.run import PREP, RUNS, aa_reference, key, load_outcomes, now, read_jsonl

POOLS = RUNS / '_pools'


def finished(run_id: str) -> dict[str, list[int]]:
    """task_id -> repeats whose attempt finished ok and has a render."""
    run_dir = RUNS / run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    keys = [key(t, r) for t in cfg['selection']['task_ids'] for r in range(cfg['repeats'])]
    outs = load_outcomes(run_dir, keys)
    by_task: dict[str, list[int]] = {}
    for k in keys:
        if outs.get(k, {}).get('status') == 'ok' and (run_dir / 'attempts' / k / 'render' / 'manifest.json').exists():
            tid, rep = k.rsplit('__r', 1)
            by_task.setdefault(tid, []).append(int(rep))
    return by_task


def _rng(seed: int, *parts) -> random.Random:
    h = hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def plan(runs: list[str], k: int, seed: int) -> list[dict]:
    fin = {r: finished(r) for r in runs}
    pairs: dict[tuple, dict] = {}
    for r in runs:
        for tid, reps in sorted(fin[r].items()):
            for rep in sorted(reps):
                others = [o for o in runs if o != r and fin[o].get(tid)]
                if not others:
                    continue
                g = _rng(seed, r, tid, rep)
                picks = g.sample(others, min(k, len(others)))
                while len(picks) < k:          # fewer other runs than k: reuse runs, other repeats
                    picks.append(g.choice(others))
                for o in picks:
                    orep = g.choice(sorted(fin[o][tid]))
                    a, b = sorted([(r, rep), (o, orep)])
                    pid = f'{tid}|{a[0]}|{a[1]}|{b[0]}|{b[1]}'
                    pairs.setdefault(pid, {'pair': pid, 'task_id': tid, 'a': {'run': a[0], 'repeat': a[1]},
                                           'b': {'run': b[0], 'repeat': b[1]}})
    return list(pairs.values())


def _render(side: dict, tid: str) -> Path:
    return RUNS / side['run'] / 'attempts' / f'{tid}__r{side["repeat"]}' / 'render'


async def judge_pool(pool: Path, arm_name: str, endpoint: str | None, concurrency: int, images: int):
    from aa_evals.gdpval import judge as J
    from aa_evals.judges.arms import ARMS
    arm = ARMS[arm_name]
    spec = endpoint or f'serving:{arm.server}'
    pairs = list(read_jsonl(pool / 'pairs.jsonl'))
    out = pool / 'matches' / f'{arm_name}.jsonl'
    out.parent.mkdir(exist_ok=True)
    budget = J.Budget()
    (pool / 'matches' / f'{arm_name}.meta.json').write_text(json.dumps(
        {'arm': arm_name, 'spec': spec, 'extra': dict(arm.extra), 'prompt_version': J.JUDGE_PROMPT_VERSION,
         'budget': budget.describe(), 'images': images, 'orders': ['AB', 'BA']}, indent=1))
    done = {(r['pair'], r['order']) for r in read_jsonl(out) if r.get('winner') or r.get('error')}
    tasks = {t.task_id: t for t in T.load_tasks()}
    judge = J.Judge(spec, dict(arm.extra), budget=budget)
    sem = asyncio.Semaphore(concurrency)
    fh = open(out, 'a')

    async def one(p, order):
        t = tasks[p['task_id']]
        first, second = (p['a'], p['b']) if order == 'AB' else (p['b'], p['a'])
        stats = {}
        async with sem:
            # built inside the semaphore: a prompt holds its page images as base64 (~1.5 MB each), so building all
            # of them up front needed ~55 GB at 35k calls (c30's two-reference pools)
            msgs = J.build_messages(t, PREP / t.task_id / 'refs', _render(first, t.task_id),
                                    _render(second, t.task_id), budget, bool(images), stats)
            res = await judge.ask(msgs, stats=stats)
        rec = {'pair': p['pair'], 'task_id': p['task_id'], 'a': p['a'], 'b': p['b'], 'order': order,
               'winner': res.get('winner'), 'error': res.get('error'), 'tries': res.get('tries'),
               'usage': res.get('usage'), 'prompt_stats': stats, 'text_tail': (res.get('text') or '')[-1500:],
               'at': now()}
        if rec['winner']:
            # score of pair side `a`: in order AB, a is shown as A; in BA, a is shown as B
            rec['score_a'] = J.score_for('MA' if order == 'AB' else 'MB', rec['winner'])
        fh.write(json.dumps(rec) + '\n')
        fh.flush()

    todo = [(p, o) for p in pairs for o in ('AB', 'BA') if (p['pair'], o) not in done]
    print(now(), f'{len(pairs)} pairs, {len(todo)} judgments to do', flush=True)
    res = await asyncio.gather(*(one(p, o) for p, o in todo), return_exceptions=True)
    bad = [r for r in res if isinstance(r, Exception)]
    fh.close()
    await judge.aclose()
    print(now(), f'judged {len(todo) - len(bad)}; infra failures {len(bad)} (rerun to retry)', flush=True)


def observations(pool: Path, arm: str, runs: list[str]) -> dict[str, list]:
    """task_id -> [(a, b, score_a, weight)]: pool pairs plus each run's vs-expert matches (anchored tasks only)."""
    no_expert = {t.task_id for t in T.load_tasks() if not t.deliverable_files}
    by_task: dict[str, list] = {}
    for r in read_jsonl(pool / 'matches' / f'{arm}.jsonl'):
        if r.get('winner'):
            by_task.setdefault(r['task_id'], []).append((r['a']['run'], r['b']['run'], r['score_a'], 0.5))
    for run in runs:
        run_dir = RUNS / run
        cfg = json.loads((run_dir / 'config.json').read_text())
        keys = [key(t, rr) for t in cfg['selection']['task_ids'] for rr in range(cfg['repeats'])]
        outs = load_outcomes(run_dir, keys)
        for r in read_jsonl(run_dir / 'matches' / f'{arm}.jsonl'):
            if r.get('winner') and r['task_id'] not in no_expert and outs.get(r['key'], {}).get('status') == 'ok':
                by_task.setdefault(r['task_id'], []).append((run, elo.EXPERT, r['model_score'], 0.5))
    return by_task


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rank = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda j: v[j]))}  # noqa: E731
    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def fit_pool(pool: Path, arm: str, n_boot: int) -> dict:
    meta = json.loads((pool / 'pool.json').read_text())
    runs = meta['runs']
    by_task = observations(pool, arm, runs)
    obs = [o for v in by_task.values() for o in v]
    ratings = elo.fit(obs)
    ci = elo.bootstrap(by_task, n=n_boot)
    rows = []
    for run in runs:
        cfg = json.loads((RUNS / run / 'config.json').read_text())
        ref = aa_reference(cfg['model'].get('aa_slug'))
        n = sum(1 for a, b, _, _ in obs if run in (a, b))
        rows.append({'run': run, 'model': cfg['model'].get('aa_name') or cfg['model'].get('served_name'),
                     'elo': round(ratings.get(run, float('nan')), 1),
                     'ci95': [round(x, 1) for x in ci.get(run, (float('nan'), float('nan')))],
                     'n_observations': n, 'aa_elo': (ref or {}).get('elo')})
    paired = [(r['elo'], r['aa_elo']) for r in rows if r['aa_elo'] is not None]
    summary = {'benchmark': 'GDPval-AA (pooled)', 'pool': pool.name, 'judge_arm': arm, 'anchor': {elo.EXPERT: elo.EXPERT_ELO},
               'n_pairs': len(list(read_jsonl(pool / 'pairs.jsonl'))), 'n_observations': len(obs),
               'n_tasks': len(by_task), 'k': meta['k'], 'seed': meta['seed'],
               'models': sorted(rows, key=lambda r: -r['elo']),
               'spearman_vs_aa': spearman([p[0] for p in paired], [p[1] for p in paired])}
    (pool / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['plan', 'judge', 'fit'])
    ap.add_argument('--pool', required=True)
    ap.add_argument('--runs', help='comma-separated run ids (plan)')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--seed', type=int, default=731)
    ap.add_argument('--judge-arm', default='qwen38-27b-medium-pin')
    ap.add_argument('--judge-endpoint', default=None)
    ap.add_argument('--judge-images', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--force', action='store_true', help='plan: replace an existing plan')
    a = ap.parse_args()
    pool = POOLS / a.pool
    if a.what == 'plan':
        runs = [r for r in (a.runs or '').split(',') if r]
        if len(runs) < 2:
            sys.exit('--runs needs at least two runs')
        if (pool / 'pairs.jsonl').exists() and not a.force:
            sys.exit(f'{pool}/pairs.jsonl exists (frozen plan); --force to replace')
        pool.mkdir(parents=True, exist_ok=True)
        pairs = plan(runs, a.k, a.seed)
        with open(pool / 'pairs.jsonl.partial', 'w') as fh:
            for p in pairs:
                fh.write(json.dumps(p) + '\n')
        (pool / 'pairs.jsonl.partial').rename(pool / 'pairs.jsonl')
        (pool / 'pool.json').write_text(json.dumps({'runs': runs, 'k': a.k, 'seed': a.seed, 'planned_at': now(),
                                                    'n_pairs': len(pairs)}, indent=1))
        print(f'{len(pairs)} pairs -> {2 * len(pairs)} judgments ({pool})')
    elif a.what == 'judge':
        asyncio.run(judge_pool(pool, a.judge_arm, a.judge_endpoint, a.concurrency, a.judge_images))
    else:
        s = fit_pool(pool, a.judge_arm, a.bootstrap)
        print(json.dumps({k: s[k] for k in ('n_pairs', 'n_observations', 'spearman_vs_aa')}),
              *[f"\n  {m['model']}: {m['elo']} {m['ci95']} (AA {m['aa_elo']})" for m in s['models']])


if __name__ == '__main__':
    main()

"""items.jsonl and summary.json of a run (aa_evals/README.md contract).

pass@1 = mean over tasks of the mean of that task's scored repeats (AA: pass@1 averaged over repeats), on 0-100.
Infrastructure failures that survived their retries are excluded (status infra_error) and counted; a run with more
than 2% of its items lost to infrastructure, or with items still pending, is not a valid result.
ci95: 10,000-resample bootstrap over tasks (each task keeps its repeats), seed 731.

  python -m aa_evals.common.harbor_mswea.score RUN      (re-score; also run by the last worker)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .queue import Queue, atomic, now, read
from aa_evals.common import jsonl

LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
ROW_KEYS = ('key', 'exit_status', 'api_calls', 'format_errors', 'finish_reasons', 'context_overflow',
            'context_cuts', 'empty_tool_args', 'max_context_tokens',
            'env_time_exceeded', 'unscored', 'exception_type', 'reward', 'agent_wall_sec', 'model_sec', 'env_sec',
            'trial_archive', 'infrastructure_reason', 'wall_sec', 'usage')


def aa_reference(slug: str | None, metric_key: str | None) -> float | None:
    if not slug or not metric_key:
        return None
    for m in (read(LEADERBOARD, {}) or {}).get('models', []):
        if m.get('slug') == slug:
            v = (m.get('scores') or {}).get(metric_key)
            return round(100 * v, 2) if isinstance(v, (int, float)) else None
    return None


def bootstrap(task_means: list[float], n: int = 10000, seed: int = 731) -> list[float] | None:
    if len(task_means) < 2:
        return None
    rng = random.Random(seed)
    k = len(task_means)
    stats = sorted(sum(rng.choice(task_means) for _ in range(k)) / k for _ in range(n))
    return [round(100 * stats[int(0.025 * n)], 2), round(100 * stats[int(0.975 * n) - 1], 2)]


def rows(run: Path) -> list[dict]:
    q = Queue(run)
    resolved = q.resolved()
    out = []
    for task, repeat in q.items():
        rec = resolved.get((task, repeat))
        if rec is None:
            status, score_ = 'pending', None
        elif rec.get('infrastructure_failure'):
            status, score_ = 'infra_error', None
        else:
            status, score_ = rec.get('status') or 'ok', rec.get('score')
        row = {'item_id': task, 'repeat': repeat, 'score': score_, 'status': status,
               'transcript': (rec or {}).get('transcript')}
        row.update({x: (rec or {}).get(x) for x in ROW_KEYS if (rec or {}).get(x) is not None})
        out.append(row)
    return out


def summarize(run: Path, items: list[dict]) -> dict:
    cfg = read(run / 'config.json', {}) or {}
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in items:
        if r['status'] in ('ok', 'model_error'):
            by_task[r['item_id']].append(float(r['score']))
    means = {t: sum(v) / len(v) for t, v in by_task.items()}
    status = Counter(r['status'] for r in items)
    n_infra = status.get('infra_error', 0)
    physical = Queue(run).done()
    score_ = round(100 * sum(means.values()) / len(means), 2) if means else None
    s = {
        'benchmark': cfg.get('benchmark'),
        'metric': cfg.get('metric', 'pass@1'),
        'score': score_,
        'n_items': len(means),
        'n_repeats': cfg.get('repeats'),
        'ci95': bootstrap(list(means.values())),
        'n_infra_errors': n_infra,
        'aa_reference': aa_reference(cfg.get('aa_slug'), cfg.get('aa_metric_key')),
        'aa_slug': cfg.get('aa_slug'),
        'model': cfg.get('model'), 'served': cfg.get('served'), 'run_id': cfg.get('run_id'),
        'n_scored_attempts': status.get('ok', 0) + status.get('model_error', 0),
        'n_pending': status.get('pending', 0),
        'infra_error_rate': round(n_infra / len(items), 4) if items else None,
        'n_infra_attempts_retried': sum(1 for r in physical.values() if r.get('infrastructure_failure')),
        'infra_reasons': dict(Counter(r.get('infrastructure_reason') for r in physical.values()
                                      if r.get('infrastructure_failure'))),
        'valid': bool(items) and status.get('pending', 0) == 0 and n_infra <= 0.02 * len(items),
        'per_repeat': {},
        'exit_status': dict(Counter(r.get('exit_status') or 'none' for r in items
                                    if r['status'] in ('ok', 'model_error'))),
        'n_unscored_model_errors': status.get('model_error', 0),
        'n_context_overflow': sum(1 for r in items if r.get('context_overflow')),
        'n_attempts_with_empty_tool_args': sum(1 for r in items if r.get('empty_tool_args')),
        'n_env_time_exceeded': sum(1 for r in items if r.get('env_time_exceeded')),
        'mean_steps': None,
        'updated_at': now(),
    }
    used = [r for r in items if r.get('usage') and r.get('wall_sec')]
    if used:
        s['resources'] = {
            'attempt_wall_h_mean': round(sum(r['wall_sec'] for r in used) / len(used) / 3600, 2),
            'attempt_wall_h_max': round(max(r['wall_sec'] for r in used) / 3600, 2),
            'peak_rss_gb_max': max(r['usage']['peak_rss_gb'] for r in used),
            'cpu_core_h_used': round(sum(r['usage']['cpu_sec'] for r in used) / 3600, 1),
            'cpu_core_h_reserved': round(sum(r['usage']['cores_reserved'] * r['wall_sec'] for r in used) / 3600, 1),
            'note': 'resolving attempts only; process-tree samples every 60 s (agent, sandboxes, verifier)'}
    steps = [r['api_calls'] for r in items if isinstance(r.get('api_calls'), int)]
    s['mean_steps'] = round(sum(steps) / len(steps), 1) if steps else None
    reps = defaultdict(list)
    for r in items:
        if r['status'] in ('ok', 'model_error'):
            reps[r['repeat']].append(float(r['score']))
    s['per_repeat'] = {str(k): {'n': len(v), 'pass_rate': round(100 * sum(v) / len(v), 2)} for k, v in sorted(reps.items())}
    # tasks of the benchmark not run (e.g. TB4's GPU tasks): a lower bound that counts them as failures
    n_total = cfg.get('n_tasks_benchmark')
    if n_total and means and not cfg.get('subset'):
        s['score_all_tasks_lower_bound'] = round(100 * sum(means.values()) / n_total, 2)
        s['n_tasks_benchmark'] = n_total
        s['excluded_tasks'] = cfg.get('excluded_tasks')
    # tasks whose reference solution fails in our runtime (an oracle run of the same tasks): reported separately
    # tasks whose reference solution fails in our runtime: oracle runs of the same tasks, comma-separated; for each
    # task the last run that has it decides (re-validations after a runtime fix override earlier runs)
    oracle = cfg.get('oracle_run')
    if oracle and means:
        verdict: dict[str, bool] = {}
        for od in [x for x in str(oracle).split(',') if x]:
            o_rows = jsonl.read(Path(od) / 'items.jsonl')
            per: dict[str, list[bool]] = defaultdict(list)
            for r in o_rows:
                if r['status'] in ('ok', 'model_error'):
                    per[r['item_id']].append(r.get('score') == 1.0)
            verdict.update({t: all(v) for t, v in per.items() if v})
        failing = sorted(t for t, ok in verdict.items() if not ok)
        kept = [m for t, m in means.items() if t not in failing]
        s.update(oracle_run=oracle, oracle_failing_tasks=failing,
                 score_excl_oracle_failures=round(100 * sum(kept) / len(kept), 2) if kept else None,
                 n_items_excl_oracle_failures=len(kept))
    cats = cfg.get('categories') or {}
    if cats and means:
        per = defaultdict(list)
        for t, m in means.items():
            per[cats.get(t, '?')].append(m)
        s['per_category'] = {c: {'n_tasks': len(v), 'score': round(100 * sum(v) / len(v), 2)}
                             for c, v in sorted(per.items())}
    return s


def finalize(run) -> dict:
    run = Path(run)
    items = rows(run)
    atomic(run / 'items.jsonl', ''.join(json.dumps(r) + '\n' for r in items))
    s = summarize(run, items)
    atomic(run / 'summary.json', s)
    return s


if __name__ == '__main__':
    print(json.dumps(finalize(sys.argv[1]), indent=1))

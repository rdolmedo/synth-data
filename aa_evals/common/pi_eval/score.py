"""items.jsonl and summary.json of a pi_eval run (aa_evals/README.md contract), from eval_q35's queue.

pass@1 = mean over tasks of the mean of that task's resolved attempts (reward >= 1 counts; eval_q35 score.py), on 0-100.
An attempt resolves at its first non-infrastructure outcome (eval_q35 queue_state.logical); attempts whose
infrastructure retries ran out are status infra_error, excluded and counted (> 2% of items: not a valid result).
Cutoff rule (no tails; overnight.md CUTOFF rules): at the deadline the unresolved attempts are cutoffs (`n_cutoff`);
`score` and `score_excl_oracle_failures` count them as 0 (over all attempts); `*_resolved_only` are secondary.
`valid`: infrastructure errors <= 2% of attempts; `complete`: no cutoffs.
Typed non-regenerating grading-pending outcomes remain unscored and block primary score, CI, valid and complete.
ci95: 10,000-resample task bootstrap, seed 731.

  python -m aa_evals.common.pi_eval.score RUN
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from aa_evals.common import jsonl
from aa_evals.common.harbor_mswea.queue import atomic, now, read
from aa_evals.common.harbor_mswea.score import aa_reference, bootstrap


def _jsonl(p: Path, bad: list) -> list[dict]:
    """The records of a proxy log ('\n'-separated, aa_evals.common.jsonl); undecodable lines go to `bad` (the
    attempt's `stats_bad_lines`, counted in the summary) instead of emptying the whole file."""
    try:
        return jsonl.read(p, bad=bad)
    except OSError as e:
        bad.append((str(p), 0, repr(e)))
        return []


USAGE = re.compile(r'"usage":\{"prompt_tokens":(\d+),"total_tokens":\d+,"completion_tokens":(\d+)')   # vLLM SSE


def attempt_stats(rec: dict) -> dict:
    """Per-attempt Pi accounting from the trial directory (no transcript content)."""
    trial = Path(rec['trial_dir']) if rec.get('trial_dir') else None
    if trial is None or not trial.exists():
        return {}
    out = {}
    clock = read(trial / 'agent' / 'env_clock.json', {}) or {}
    if clock:
        out.update(end_state=clock.get('state'), model_requests=clock.get('model_requests'),
                   wall_sec=clock.get('wall_sec'), model_sec=clock.get('model_sec'), env_sec=clock.get('env_sec'))
    pi = trial.parent / 'pi-cont16k'
    bad: list = []
    limit = _jsonl(pi / 'context-limit.jsonl', bad)
    out['context_limit'] = any(x.get('termination_reason') == 'context_limit' for x in limit)
    out['output_cap_hits'] = len(_jsonl(pi / 'turns.jsonl', bad))
    if bad:
        out['stats_bad_lines'] = len(bad)
    # largest context of any model request (prompt + completion tokens, from the usage in the proxy's wire responses)
    best_p = best_c = None
    for f in sorted((pi / 'wire-responses').glob('*.response')) if (pi / 'wire-responses').is_dir() else []:
        try:
            with f.open('rb') as fh:
                fh.seek(max(0, f.stat().st_size - 8192))
                tail = fh.read().decode('utf-8', 'replace')
        except OSError:
            continue
        m = None
        for m in USAGE.finditer(tail):
            pass
        if m:
            pt, ct = int(m.group(1)), int(m.group(2))
            best_p = max(best_p or 0, pt)
            best_c = max(best_c or 0, pt + ct)
    if best_c is not None:
        out.update(max_prompt_tokens=best_p, max_context_tokens=best_c)
    result = read(trial / 'result.json', {}) or {}
    ar = result.get('agent_result') or {}
    out.update(n_input_tokens=ar.get('n_input_tokens'), n_output_tokens=ar.get('n_output_tokens'))
    return {k: v for k, v in out.items() if v is not None}


def rows(run: Path) -> list[dict]:
    sys.path.insert(0, str(Path(read(run / 'config.json')['code']) / 'eval_q35'))
    from queue_state import State   # eval_q35's queue (the run's frozen code)
    st = State(run)
    logical = st.logical()
    out = []
    for (task, attempt), rec in sorted(logical.items()):
        if rec is None:
            status, sc = 'pending', None
        elif rec.get('infrastructure_failure'):
            status, sc = 'infra_error', None
        elif rec.get('grading_review_required') is True and rec.get('grading_pending') is True:
            status, sc = 'grading_pending', None
        else:
            status, sc = 'ok', 1.0 if (rec.get('score') or 0) >= 1 else 0.0
        row = {'item_id': task, 'repeat': attempt, 'score': sc, 'status': status}
        if rec:
            row.update({k: rec.get(k) for k in ('key', 'reward', 'exception_type', 'infrastructure_reason', 'unscored',
                                                 'pi_error', 'retry', 'trial_dir', 'grading_review_required',
                                                 'grading_pending', 'grading_pending_evidence') if rec.get(k) is not None})
            row['transcript'] = rec.get('trial_archive') or rec.get('trial_dir')
            row.update(rec.get('stats') or attempt_stats(rec))
        out.append(row)
    return out


def _rate(v: list) -> float | None:
    return round(100 * sum(v) / len(v), 1) if v else None


def summarize(run: Path, items: list[dict]) -> dict:
    cfg = read(run / 'config.json', {}) or {}
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in items:
        if r['status'] == 'ok':
            by_task[r['item_id']].append(r['score'])
    means = {t: sum(v) / len(v) for t, v in by_task.items()}
    status = Counter(r['status'] for r in items)
    n_infra = status.get('infra_error', 0)
    n_pending_grading = status.get('grading_pending', 0)
    physical = [json.loads(p.read_text()) for p in (run / 'queue' / 'done').glob('*.json')]
    # an attempt stopped at the run's cutoff (record marked 'cutoff' when the run was frozen) is a cutoff, not infra
    stopped_at_cutoff = [r for r in physical if r.get('cutoff')]
    physical = [r for r in physical if not r.get('cutoff')]
    ok_rows = [r for r in items if r['status'] == 'ok']
    tasks_all = sorted({r['item_id'] for r in items})
    pend0 = defaultdict(list)
    for r in items:
        if r['status'] in ('ok', 'pending'):
            pend0[r['item_id']].append(r['score'] or 0.0)
    scored_or_grading_tasks = set(pend0) | {r['item_id'] for r in items if r['status'] == 'grading_pending'}
    s = {
        'benchmark': cfg.get('benchmark'), 'metric': 'pass@1', 'harness': cfg.get('harness_label'),
        # CUTOFF rules (overnight.md, audit 2026-11-03): the score is over all attempts, cutoffs (unresolved at the
        # deadline) counted 0; infrastructure errors are never scored. The resolved-only score is secondary.
        'score': round(100 * sum(sum(v) / len(v) for v in pend0.values()) / len(pend0), 2) if pend0 else None,
        'score_resolved_only': round(100 * sum(means.values()) / len(means), 2) if means else None,
        'n_items': len(scored_or_grading_tasks), 'n_items_resolved': len(means), 'n_tasks_run': len(tasks_all), 'n_repeats': cfg.get('attempts'),
        'ci95': bootstrap([sum(v) / len(v) for v in pend0.values()]),
        'n_infra_errors': n_infra, 'n_pending': status.get('pending', 0),
        # attempts whose proxy logs (context-limit / turns) had undecodable lines: end and cap counts may be low
        'n_attempts_stats_bad_lines': sum(1 for r in items if r.get('stats_bad_lines')),
        'n_pending_grading': n_pending_grading,
        # at a deadline cutoff (no tails) the pending attempts are the cutoffs: 0 in score
        'n_cutoff': status.get('pending', 0),
        'infra_error_rate': round(n_infra / len(items), 4) if items else None,
        'valid': bool(items) and not n_pending_grading and n_infra <= 0.02 * len(items),   # infra <= 2% (cutoffs do not invalidate)
        'complete': status.get('pending', 0) == 0 and n_pending_grading == 0,
        'n_infra_attempts_retried': sum(1 for r in physical if r.get('infrastructure_failure')),
        'infra_reasons': dict(Counter(r.get('infrastructure_reason') for r in physical if r.get('infrastructure_failure'))),
        'n_stopped_at_cutoff': len(stopped_at_cutoff),
        'aa_reference': aa_reference(cfg.get('aa_slug'), cfg.get('aa_metric_key')), 'aa_slug': cfg.get('aa_slug'),
        'model': cfg.get('model_spec'), 'served': cfg.get('served'), 'reasoning_effort': cfg.get('reasoning_effort'),
        'run_id': cfg.get('run_id'), 'agent': cfg.get('agent', 'pi'), 'code_commit': cfg.get('code_commit'),
        'end_states': dict(Counter(r.get('end_state') or 'none' for r in ok_rows)),
        'context_window': (cfg.get('protocol') or {}).get('context_window'),
        'n_context_limit': sum(1 for r in ok_rows if r.get('context_limit')),
        'context_limit_pass_rate': _rate([r['score'] for r in ok_rows if r.get('context_limit')]),
        # attempts whose context (prompt + completion of one request) passed 132k: the protocol default, where an
        # episode would have ended (r3); comparable across runs with a larger window
        'n_past_132k': sum(1 for r in ok_rows if (r.get('max_context_tokens') or 0) >= 132000),
        'past_132k_pass_rate': _rate([r['score'] for r in ok_rows if (r.get('max_context_tokens') or 0) >= 132000]),
        'n_with_context_stats': sum(1 for r in ok_rows if r.get('max_context_tokens') is not None),
        'n_unscored': sum(1 for r in ok_rows if r.get('unscored')),
        'n_pi_errors': sum(1 for r in ok_rows if r.get('pi_error')),
        'output_cap_hits': sum(r.get('output_cap_hits') or 0 for r in ok_rows),
        'mean_model_requests': round(sum(r.get('model_requests') or 0 for r in ok_rows) / len(ok_rows), 1)
        if ok_rows else None,
        'per_repeat': {}, 'updated_at': now(),
        # M3.0 / M3.2 protocol record (#173, #175)
        'n_infra': n_infra, 'n_repeats_planned': cfg.get('attempts'),
        'context_cap': (cfg.get('protocol') or {}).get('context_window'),
        'judge': None,   # not judged: the task's own verifier decides pass/fail
        'request_overrides': (cfg.get('request') or {}).get('overrides'),
        'request_source': (cfg.get('request') or {}).get('source'),
        'request_adapters': (cfg.get('request') or {}).get('adapters'),
        # #481: the run's AAI-open version ('unrecorded' for runs before the field existed)
        'aai_open_version': (cfg.get('aai_open') or {}).get('version', 'unrecorded'),
    }
    reps = defaultdict(list)
    for r in ok_rows:
        reps[r['repeat']].append(r['score'])
    s['per_repeat'] = {str(k): {'n': len(v), 'pass_rate': round(100 * sum(v) / len(v), 2)} for k, v in sorted(reps.items())}
    # tasks whose reference solution fails in our runtime (oracle runs of this benchmark, comma-separated run dirs;
    # for each task the last run that has it decides): the score without them is reported alongside
    oracle = cfg.get('oracle_run')
    if oracle and pend0:
        verdict: dict[str, bool] = {}
        for od in [x for x in str(oracle).split(',') if x]:
            per: dict[str, list[bool]] = defaultdict(list)
            for r in jsonl.read(Path(od) / 'items.jsonl'):   # a finalized run: a bad line raises
                if r['status'] == 'ok':
                    per[r['item_id']].append(r.get('score') == 1.0)
            verdict.update({t: all(v) for t, v in per.items() if v})
        failing = sorted(t for t, ok in verdict.items() if not ok)
        kept = [m for t, m in means.items() if t not in failing]
        kept0 = [sum(v) / len(v) for t, v in pend0.items() if t not in failing]
        s.update(oracle_run=oracle, oracle_failing_tasks=failing,
                 score_excl_oracle_failures=round(100 * sum(kept0) / len(kept0), 2) if kept0 else None,
                 score_excl_oracle_failures_resolved_only=round(100 * sum(kept) / len(kept), 2) if kept else None,
                 n_items_excl_oracle_failures=len(kept0))
    cats = cfg.get('categories') or {}
    if cats and pend0:
        per = defaultdict(list)
        for t, m in ((t, sum(v) / len(v)) for t, v in pend0.items()):
            per[cats.get(t, '?')].append(m)
        s['per_category'] = {c: {'n_tasks': len(v), 'score': round(100 * sum(v) / len(v), 2)} for c, v in sorted(per.items())}
    walls = [r['wall_sec'] for r in ok_rows if r.get('wall_sec')]
    if walls:
        s['attempt_wall_h'] = {'mean': round(sum(walls) / len(walls) / 3600, 2), 'max': round(max(walls) / 3600, 2)}
    if n_pending_grading:
        # Grader adapter review is neither a model zero nor a generation cutoff.
        # Resolved-only diagnostics remain labeled as such; no primary result or
        # confidence interval may be published until every grading row resolves.
        s['score'] = None
        s['ci95'] = None
        if 'score_excl_oracle_failures' in s:
            s['score_excl_oracle_failures'] = None
    return s


def finalize(run) -> dict:
    run = Path(run)
    items = rows(run)
    atomic(run / 'items.jsonl', ''.join(json.dumps(r) + '\n' for r in items))
    s = summarize(run, items)
    from aa_evals.tracejudge import fold_summary   # n_flagged from the trace judge's flags.jsonl (M3.4, #177)
    s = fold_summary(run, s)
    atomic(run / 'summary.json', s)
    return s


if __name__ == '__main__':
    print(json.dumps(finalize(sys.argv[1]), indent=1))

#!/usr/bin/env python3
"""Score an eval run (campaign 20261015T215325Z convention) and summarize its trajectories.

  score.py RUN_DIR [RUN_DIR ...]          -> RUN_DIR/scores.json, summary on stdout

pass@1 = mean over tasks of the mean of its 3 single-attempt outcomes (reward >= 1 passes), with a
10,000-resample task bootstrap 95% interval (seed 731). Attempts ended by the eval tail rule count as scored ("tail_cut"
records: by the verifier, or as failures); tail_cuts and pass_resolved_only (item-level, cut items dropped) report them. The resolving physical attempt of a logical
attempt is its first non-infrastructure outcome (infrastructure failures get same-seed retries; an
attempt whose retries are exhausted counts as a failure). Partial runs: pass over finished attempts.

Per-trajectory statistics (from the saved Pi sessions, proxy logs and env_clock.json; for terminus-2 runs from the
agent's env_clock.json: episodes ended at the context limit, 16,384-token output-cap hits, requests):
  timeouts (env-time budget vs wall-clock backstop; attempts that ran past 21,600 s of wall clock),
  context-limit submissions (policy 'submit'), Pi compactions (policy 'compact'), length-capped (16,384-token) turns and continuations, requests whose max_tokens the
  proxy clamped to the context window, output tokens, and serving decode throughput (vLLM metrics).
serving_numerics: resolving attempts per effective serving-numerics hash (serve_numerics.py; "unrecorded" before
2026-11-14), and attempts whose run changed serving while they ran.
"""
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read  # noqa: E402
from queue_state import State  # noqa: E402
from rawpack import locate, raw_root  # noqa: E402

CONTINUE = 'Thinking exceeded the maximum thinking budget, please continue.'


def trial_stats(run, rec):
    """Trajectory statistics of one physical attempt."""
    s = {'compactions': 0, 'length_stops': 0, 'continuations': 0, 'assistant_messages': 0,
         'error_stops': 0, 'clamped_requests': 0, 'requests': 0, 'output_tokens': None,
         'agent_wall_sec': None, 'env_clock': None, 'context_400': 0, 'context_limit_submits': 0}
    raw = raw_root(run / 'attempts' / rec['key']) / rec['key']   # loose raw/ or packed raw.zip (rawpack.py)
    trial = locate(rec['trial_dir']) if rec.get('trial_dir') else None
    if trial and trial.exists():
        result = read(trial / 'result.json', {})
        ar = result.get('agent_result') or {}
        s['output_tokens'] = ar.get('n_output_tokens')
        s['input_tokens'] = ar.get('n_input_tokens')
        ex = result.get('agent_execution') or {}
        if ex.get('started_at') and ex.get('finished_at'):
            from datetime import datetime
            s['agent_wall_sec'] = (datetime.fromisoformat(ex['finished_at']) -
                                   datetime.fromisoformat(ex['started_at'])).total_seconds()
        s['env_clock'] = read(trial / 'agent' / 'env_clock.json')
        if (s['env_clock'] or {}).get('harness') == 'terminus-2':   # tb_agents/terminus_eval.py
            c = s['env_clock']
            s.update(context_limit_submits=int(c.get('state') == 'context_limit'), length_stops=c.get('output_cap_hits') or 0,
                     requests=c.get('requests') or 0, assistant_messages=c.get('episodes') or 0)
            return s
        for path in sorted(trial.glob('agent/pi/sessions/**/*.jsonl')):
            for line in path.read_bytes().decode(errors='replace').split('\n'):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('type') == 'compaction':
                    s['compactions'] += 1
                m = e.get('message') or {}
                if e.get('type') == 'message' and m.get('role') == 'assistant':
                    s['assistant_messages'] += 1
                    s['length_stops'] += m.get('stopReason') == 'length'
                    s['error_stops'] += m.get('stopReason') == 'error'
                    if 'context window' in str(m.get('errorMessage', '')):
                        s['context_400'] += 1
                if e.get('type') == 'message' and m.get('role') == 'user' and CONTINUE in json.dumps(m.get('content')):
                    s['continuations'] += 1
    limit_log = raw / 'pi-cont16k' / 'context-limit.jsonl'
    if limit_log.exists():   # submit-at-limit policy: the guard ended the episode at the window
        s['context_limit_submits'] = sum(1 for line in limit_log.read_text().splitlines() if line.strip())
    for meta in sorted((raw / 'pi-cont16k' / 'wire-responses').glob('*.meta.json')) if raw.exists() else []:
        m = read(meta, {})
        if m.get('agent_request'):
            s['requests'] += 1
            s['clamped_requests'] += (m.get('max_tokens') or 16384) < 16384
    return s


def throughput(run, since=0.0):
    rows = []
    for line in (run / 'serve' / 'metrics.jsonl').read_text().splitlines() if (run / 'serve' / 'metrics.jsonl').exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    rows = [r for r in rows if r.get('t', 0) >= since]   # this eval only (the server outlives queue resets)
    if len(rows) < 2:
        return None
    windows, gen, busy = [], 0.0, 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b['t'] - a['t']
        dg = b.get('generation_tokens_total', 0) - a.get('generation_tokens_total', 0)
        if 0 < dt < 120 and dg >= 0 and b.get('num_requests_running', 0) > 0:
            gen += dg
            busy += dt
            windows.append((dg / dt, b.get('num_requests_running', 0), b.get('kv_cache_usage_perc', 0)))
    pre = rows[-1].get('num_preemptions_total', 0) - rows[0].get('num_preemptions_total', 0)
    hits = rows[-1].get('prefix_cache_hits_total', 0) - rows[0].get('prefix_cache_hits_total', 0)
    queries = rows[-1].get('prefix_cache_queries_total', 0) - rows[0].get('prefix_cache_queries_total', 0)
    return {'generated_tokens': gen, 'busy_sec': round(busy), 'mean_output_tok_s': round(gen / busy, 1) if busy else None,
            'median_output_tok_s': round(statistics.median(w[0] for w in windows), 1) if windows else None,
            'mean_running_requests': round(statistics.mean(w[1] for w in windows), 1) if windows else None,
            # output tokens/s per running request (a request in prefill counts as running)
            'per_request_tok_s': round(statistics.mean(w[0] / w[1] for w in windows if w[1]), 1) if windows else None,
            'max_kv': round(max(r.get('kv_cache_usage_perc', 0) for r in rows), 3),
            'preemptions': pre, 'prefix_cache_hit_rate': round(hits / queries, 3) if queries else None}


def score(run):
    run = Path(run)
    state = State(run)
    config = read(run / 'config.json', {})
    logical = state.logical()
    tasks = sorted({t for t, _ in logical})
    attempts = sorted({a for _, a in logical})
    resolved = {k: v for k, v in logical.items() if v is not None}
    passed = {k: (v.get('score') or 0) >= 1 for k, v in resolved.items()}
    n_expected = len(logical)
    out = {'run': str(run), 'model': config.get('model'), 'tasks': len(tasks), 'attempts_per_task': len(attempts),
           'expected': n_expected, 'resolved': len(resolved), 'coverage': len(resolved) / n_expected if n_expected else 0,
           'passed': sum(passed.values()),
           'pass_over_finished': sum(passed.values()) / len(resolved) if resolved else None}
    out['per_attempt'] = [{'attempt': a, 'finished': sum(1 for (t, x) in resolved if x == a),
                           'passed': sum(p for (t, x), p in passed.items() if x == a),
                           'pass_rate': (sum(p for (t, x), p in passed.items() if x == a) / len(tasks)) if
                           all((t, a) in resolved for t in tasks) else None} for a in attempts]
    if len(resolved) == n_expected and n_expected:
        per_task = [sum(passed[t, a] for a in attempts) / len(attempts) for t in tasks]
        out['pass_at_1'] = sum(per_task) / len(tasks)
        rng = random.Random(731)
        samples = sorted(sum(rng.choices(per_task, k=len(tasks))) / len(tasks) for _ in range(10000))
        out['task_bootstrap_95'] = [samples[250], samples[9749]]
        out['per_task'] = {t: per_task[i] for i, t in enumerate(tasks)}
        if config.get('irt_items'):   # IRT-selected mix (irt_mix.py): ability, sub-scores, full-benchmark projections
            import irt_mix
            bench_of = {k[1]: k[0] for k in irt_mix.load_items(config['irt_items'])[1]}
            cnt = {(bench_of[t], t): [sum(passed[t, a] for a in attempts), len(attempts)] for t in tasks}
            out['irt'] = irt_mix.summary(cnt, config['irt_items'])
    # physical attempts: infrastructure failures and trajectory statistics
    records = [read(p) for p in sorted((run / 'queue' / 'done').glob('*.json'))]
    records = [r for r in records if r]
    out['physical_attempts'] = len(records)
    out['infra_failures'] = dict(Counter(r.get('infrastructure_reason') for r in records if r.get('infrastructure_failure')))
    out['infra_exhausted'] = sum(1 for v in resolved.values() if v.get('infrastructure_failure'))
    chosen = list(resolved.values())
    stats = {r['key']: trial_stats(run, r) for r in chosen}
    states = Counter((s['env_clock'] or {}).get('state') for s in stats.values())
    out['timeouts'] = {
        # tail cuts end with AgentTimeoutError too; they are counted under tail_cuts, not here
        'agent_timeout_error': sum(1 for r in chosen if r.get('exception_type') == 'AgentTimeoutError' and not r.get('tail_cut')),
        'env_budget_spent': states.get('env_budget_spent', 0), 'wall_clock_backstop': states.get('wall_clock_backstop', 0),
        'agent_wall_over_21600s': sum(1 for s in stats.values() if (s['agent_wall_sec'] or 0) > 21600),
        'passed_with_agent_wall_over_21600s': sum(1 for r in chosen if (stats[r['key']]['agent_wall_sec'] or 0) > 21600
                                                  and (r.get('score') or 0) >= 1)}
    # Tail rule (EVAL_REQUESTS.md 10-18 ~09:55 UTC): attempts ended by the eval's tail cut, and pass over the rest
    cuts = {k: v for k, v in resolved.items() if v.get('tail_cut')}
    kept = [p for k, p in passed.items() if k not in cuts]
    out['tail_cuts'] = {'total': len(cuts), **dict(Counter(f"reason_{v.get('tail_cut_reason')}" for v in cuts.values())),
                        **dict(Counter(f"scored_{v.get('tail_cut_scored')}" for v in cuts.values())),
                        'passed': sum(passed[k] for k in cuts)}
    out['pass_resolved_only'] = sum(kept) / len(kept) if kept else None   # item-level, cut items dropped
    out['exceptions'] = dict(Counter(r.get('exception_type') for r in chosen))
    out['unscored_model_failures'] = sum(1 for r in chosen if r.get('unscored'))
    agg = lambda f: sum(f(s) for s in stats.values())  # noqa: E731
    out['events'] = {
        'context_limit_submissions': agg(lambda s: s['context_limit_submits'] > 0),
        'trajectories_with_compaction': agg(lambda s: s['compactions'] > 0), 'compactions': agg(lambda s: s['compactions']),
        'length_capped_turns': agg(lambda s: s['length_stops']), 'continuation_messages': agg(lambda s: s['continuations']),
        'trajectories_with_length_cap': agg(lambda s: s['length_stops'] > 0),
        'context_clamped_requests': agg(lambda s: s['clamped_requests']),
        'context_window_400': agg(lambda s: s['context_400']), 'pi_error_stops': agg(lambda s: s['error_stops']),
        'assistant_messages': agg(lambda s: s['assistant_messages']), 'model_requests': agg(lambda s: s['requests'])}
    tok = [s['output_tokens'] for s in stats.values() if s['output_tokens'] is not None]
    wall = [s['agent_wall_sec'] for s in stats.values() if s['agent_wall_sec'] is not None]
    out['trajectory'] = {'mean_output_tokens': round(statistics.mean(tok)) if tok else None,
                         'median_output_tokens': round(statistics.median(tok)) if tok else None,
                         'median_agent_wall_sec': round(statistics.median(wall)) if wall else None,
                         'max_agent_wall_sec': round(max(wall)) if wall else None,
                         'median_env_sec': round(statistics.median([s['env_clock']['env_sec'] for s in stats.values()
                                                                    if s['env_clock']])) if any(s['env_clock'] for s in stats.values()) else None}
    out['serving'] = throughput(run, config.get('reset_epoch', 0.0))
    # Effective serving numerics of the resolving attempts (worker.serving_numerics, serve_numerics.py): more than one
    # hash means the run's numerics changed mid-run (e.g. a move from 1x H100 FP8 to 2x H100 BF16 under one variant)
    out['serving_numerics'] = {
        'by_sha256': dict(Counter(v.get('serving_numerics_sha256') or 'unrecorded' for v in chosen)),
        'changed_during_attempt': sum(1 for v in chosen if v.get('serving_numerics_changed'))}
    out['protocol'] = config.get('protocol')
    out['per_attempt_records'] = {f'{t}__a{a}': {'key': v['key'], 'score': v.get('score'),
                                                 'exception': v.get('exception_type'), 'tail_cut': bool(v.get('tail_cut')),
                                                 'serving_numerics_sha256': v.get('serving_numerics_sha256'),
                                                 **stats[v['key']]}
                                  for (t, a), v in sorted(resolved.items())}
    (run / 'scores.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    for run in sys.argv[1:]:
        s = score(run)
        brief = {k: v for k, v in s.items() if k not in ('per_attempt_records', 'per_task')}
        print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()

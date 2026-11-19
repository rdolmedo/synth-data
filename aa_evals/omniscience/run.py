"""AA-Omniscience 1.1 on the public v1.1 set: answer, grade, summarize.

  python -m aa_evals.omniscience.run --model serving:aa-super --judge serving:aa-super --run-id smoke-super --limit 60
  python -m aa_evals.omniscience.run --answers-from smoke-super --judge serving:<other-judge> --run-id smoke-super-j2

The judge is a self-hosted open model (user decision, 2026-11-02: no closed API judge). The default is
serving:aa-judge. AA's own grader, openai:gpt-5.6-luna, needs --allow-external-api, because it is external spend.

The run directory (/data/aai/aa_evals/runs/omniscience/<run_id>/) holds config.json, items.jsonl, summary.json
and transcripts/. Re-running the same command resumes: finished items are kept, infra errors are retried, and an
item whose answer exists but whose grading failed is regraded without asking the model again.
"""
from __future__ import annotations

import argparse
import contextlib
import concurrent.futures as cf
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from aa_evals.common import aai_open, protocol
from aa_evals.judges import pins
from aa_evals.tracejudge import fold_summary

from . import client, data, judge, metrics, presets
from .prompts import OMNISCIENCE_ANSWER_PROMPT

BENCH = 'omniscience'
RUNS_ROOT = Path(os.environ.get('AA_EVALS_RUNS', '/data/aai/aa_evals/runs')) / BENCH
LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
CODE_ROOT = Path(__file__).resolve().parents[2]
INFRA_LIMIT = 0.02             # contract: > 2% infra errors is not a valid result
CIRCUIT_BREAKER = 25           # consecutive model-side infra errors before the run stops submitting work
AUTO_MAX_TOKENS_MARGIN = 2048  # prompt tokens + slack, subtracted from the endpoint's context length
DEFAULT_JUDGE = 'serving:aa-judge'  # a self-hosted open judge; which model is pending the judge test (#31)


def git_info() -> dict:
    frozen = CODE_ROOT / 'GIT_SHA'
    if frozen.exists():  # a frozen copy made by submit.sh
        return json.loads(frozen.read_text())
    if (CODE_ROOT / 'FROZEN_SHA').exists():  # a `git archive HEAD` copy (hle/judges submit)
        return {'sha': (CODE_ROOT / 'FROZEN_SHA').read_text().strip(), 'dirty': False, 'source': str(CODE_ROOT)}
    try:
        sha = subprocess.check_output(['git', '-C', str(CODE_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(CODE_ROOT), 'status', '--porcelain', '--',
                                         'aa_evals/omniscience'], text=True).strip()
        return {'sha': sha, 'dirty': bool(dirty), 'source': str(CODE_ROOT)}
    except Exception as e:
        return {'sha': None, 'error': str(e), 'source': str(CODE_ROOT)}


def aa_reference(aa_name: str | None) -> dict | None:
    if not aa_name or not LEADERBOARD.exists():
        return None
    board = json.loads(LEADERBOARD.read_text())
    for m in board.get('models', []):
        if m.get('name') == aa_name:
            s = m.get('scores') or {}
            if 'omniscience-accuracy' not in s:
                return {'model': aa_name, 'note': 'listed by AA without AA-Omniscience scores'}
            return {'model': aa_name, 'source': f'{LEADERBOARD} (fetched {board.get("fetched")})',
                    'aa_version': '1.1 (3,000 private questions)',
                    'accuracy': round(100 * s['omniscience-accuracy'], 2),
                    'non_hallucination_rate': round(100 * s['omniscience-non-hallucination'], 2),
                    'omniscience_index': round(m['omniscience'], 2) if m.get('omniscience') is not None else None}
    return {'model': aa_name, 'note': 'not found in AA leaderboard snapshot'}


def parse_json_arg(s: str | None) -> dict:
    return json.loads(s) if s else {}


def build_gen(args, ep: client.Endpoint) -> tuple[str | None, str | None, dict]:
    """The model's request settings (M3.0 protocol, aa_evals/common/protocol.py): a serving: model starts from its
    registry's request.overrides (the lab's sampling, thinking, the spec's default effort); --preset, or a url:
    model, uses a preset from presets.py. --reasoning-effort sets the effort wherever the request carries one."""
    overrides = (ep.request or {}).get('overrides') if ep.kind == 'serving' else None
    if overrides and not args.preset:
        preset_name, gen = None, {**json.loads(json.dumps(overrides)), 'max_tokens': 'auto'}
        p = {'aa_name': None}
    else:
        preset_name, p = presets.for_model(ep.model, args.preset)
        gen = p['gen']
    if args.reasoning_effort is not None:
        try:
            gen = protocol.apply_effort(gen, args.reasoning_effort)
        except ValueError as e:
            raise SystemExit(f'--reasoning-effort {args.reasoning_effort}: {e}')
    for k in ('temperature', 'top_p'):
        v = getattr(args, k)
        if v is not None:
            gen[k] = v
    if args.max_tokens is not None:
        gen['max_tokens'] = args.max_tokens if args.max_tokens == 'auto' else int(args.max_tokens)
    if args.chat_template_kwargs:
        gen['chat_template_kwargs'] = parse_json_arg(args.chat_template_kwargs)
    gen.update(parse_json_arg(args.extra_body))
    aa_name = args.aa_name or protocol.aa_model(ep.model, gen)[0] or p['aa_name']
    return preset_name, aa_name, gen


def resolve_max_tokens(ep: client.Endpoint, gen: dict, context_window: int | None) -> tuple[dict, dict]:
    """max_tokens 'auto': the run's context (the protocol's 128k cap, never above the endpoint's) minus a margin
    for the prompt (system + question, a few hundred tokens). Returns (gen, context record)."""
    gen = dict(gen)
    ctx = client.max_model_len(ep)
    win, why = protocol.window(ctx, context_window)
    context = {'window': win, 'rule': why, 'endpoint_max_model_len': ctx, 'requested': context_window}
    if gen.get('max_tokens') == 'auto':
        if win is None:
            raise SystemExit(f'--max-tokens auto: {ep.spec} does not report max_model_len; pass --max-tokens N')
        gen['max_tokens'] = win - AUTO_MAX_TOKENS_MARGIN
        gen['_max_tokens_from'] = f'context {win} - {AUTO_MAX_TOKENS_MARGIN}'
    elif win is not None and int(gen.get('max_tokens') or 0) > win - AUTO_MAX_TOKENS_MARGIN:
        raise SystemExit(f'--max-tokens {gen["max_tokens"]} exceeds the {win}-token context window '
                         '(--context-window 0 lifts the protocol cap for calibration runs)')
    return gen, context


def judge_gen(ep: client.Endpoint, args) -> dict:
    """Judge request settings. A serving: judge starts from its registry's request.overrides (the lab's sampling),
    as HLE's and GDP.pdf's judges do, so one --judge-extra-body means the same request on all three benchmarks
    (the judge study, #41). --judge-preset picks a preset instead; openai: uses the preset for its model."""
    if args.judge_preset or ep.kind == 'openai':
        gen = presets.for_model(ep.model, args.judge_preset)[1]['gen']
    elif ep.kind == 'serving':
        gen = json.loads(json.dumps((ep.request or {}).get('overrides') or {}))
    else:
        gen = {}
    if ep.kind == 'openai':
        gen.pop('max_tokens', None)  # OpenAI reasoning models: the API's default output cap
    elif gen.get('max_tokens') in (None, 'auto'):
        gen['max_tokens'] = presets.JUDGE_LOCAL_MAX_TOKENS
    if args.judge_reasoning_effort:
        gen['reasoning_effort'] = args.judge_reasoning_effort
    gen.update(parse_json_arg(args.judge_extra_body))
    return gen


def request_body(gen: dict) -> dict:
    return {k: v for k, v in gen.items() if not k.startswith('_')}


class Writer:
    """items.jsonl is append-only during a run; the last row per (item_id, repeat) wins."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self.lock, self.path.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r['item_id'], r['repeat'])] = r
    return rows


def answer_item(ep, gen, item, timeout) -> dict:
    system = OMNISCIENCE_ANSWER_PROMPT.format(domain=item['domain'], subtopic=item['subtopic'])
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': item['question']}]
    errors: list = []
    t0 = time.time()
    try:
        from tb_agents import response_budget as response_policy
        from tb_agents.response_budget_single import run as budgeted_call
        call = lambda body: client.chat(ep, body, timeout=max(1, timeout-(time.time()-t0)), log=errors, timeout_is_model=True)
        body = dict(request_body(gen), messages=messages)
        budget = response_policy.with_benchmark(response_policy.settings(), 'omniscience')
        resp = budgeted_call(call, body, budget) if budget else call(body)
    except client.ModelTimeout as e:
        # A runaway generation on a healthy server: the model's behavior, with no answer (NOT_ATTEMPTED).
        return {'messages': messages, 'answer': '', 'reasoning': '', 'finish_reason': 'timeout', 'usage': {},
                'latency_s': round(time.time() - t0, 3), 'attempts': len(errors), 'errors': errors,
                'timeout_detail': str(e)}
    out = client.split_response(resp)
    return {'messages': messages, 'answer': out['answer'], 'reasoning': out['reasoning'],
            'finish_reason': out['finish_reason'], 'usage': out['usage'], 'latency_s': round(time.time() - t0, 3),
            'attempts': resp.get('_attempts'), 'errors': errors, 'gate_queue_ms': resp.get('_gate_queue_ms'),
            'response_budget': resp.get('_response_budget')}


def cutoff_row(row: dict) -> dict:
    """An item still unanswered at --deadline-s. No tails: it is not correct (0 for accuracy), it is left out of the
    non-hallucination denominator (metrics.py; the conservative variant counts it INCORRECT), and a resume of the run
    retries it."""
    return dict(row, status='cutoff', grade=metrics.CUTOFF, grade_source='cutoff', score=0, net=0)


def process(task, ctx) -> dict:
    item, rep, prior = task
    tpath = ctx['tdir'] / f'{item["item_id"]}.r{rep}.json'
    row = {'item_id': item['item_id'], 'repeat': rep, 'domain': item['domain'], 'subtopic': item['subtopic'],
           'gold': item['answer'], 'transcript': str(tpath.relative_to(ctx['run_dir']))}
    transcript = {'item': item, 'repeat': rep}

    # 1. The model's answer: reuse one that exists (resume after a judge failure, or --answers-from).
    ans = None
    if prior is not None and prior.get('answer_record'):
        ans = prior['answer_record']
    elif ctx['answers_from'] is not None:
        src = ctx['answers_from'].get((item['item_id'], rep))
        if src is not None and src.get('status') == 'cutoff':
            return cutoff_row(row)
        if src is None or src.get('status') == 'infra_error' or 'response' not in src:
            row.update(status='infra_error', stage='model', error='no answer in --answers-from run', score=None)
            return row
        st = json.loads((ctx['answers_from_dir'] / src['transcript']).read_text())
        ans = st['model']
        ans['reused_from'] = str(ctx['answers_from_dir'])
    elif ctx.get('stage') == 'grade':
        # The grading stage must never repair a missing answer by calling the generation server.
        row.update(status='infra_error', stage='model', error='no stored generation to grade', score=None)
        return row
    else:
        if ctx['stop'].is_set():
            row.update(status='skipped', stage='model', error='circuit breaker open', score=None)
            return row
        try:
            lane = ctx.get('lane')   # #344 dispatcher: the model request waits for a slot of the shared budget
            with (lane.slot(f"{item['item_id']}.r{rep}", item['item_id'], ctx['pred'](item['item_id']))
                  if lane else contextlib.nullcontext()):
                ans = answer_item(ctx['model_ep'], ctx['gen'], item, ctx['timeout'])
            ctx['consecutive_infra'][0] = 0
        except client.InfraError as e:
            ctx['consecutive_infra'][0] += 1
            if ctx['consecutive_infra'][0] >= CIRCUIT_BREAKER:
                ctx['stop'].set()
            row.update(status='infra_error', stage='model', error=str(e)[:500], score=None)
            transcript['model_error'] = str(e)
            tpath.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
            return row
    transcript['model'] = ans
    truncated = ans['finish_reason'] == 'length'
    timed_out = ans['finish_reason'] == 'timeout'
    row.update(response=ans['answer'], finish_reason=ans['finish_reason'], truncated=truncated,
               completion_tokens=(ans.get('usage') or {}).get('completion_tokens'),
               reasoning_chars=len(ans.get('reasoning') or ''), latency_s=ans.get('latency_s'),
               gate_queue_ms=ans.get('gate_queue_ms'))

    if ctx.get('stage') == 'generate':
        tpath.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
        row.update(status='pending', score=None, answer_record=ans)
        return row

    # At the deadline, an answer that is in hand (or waiting for a judge slot) is graded, never discarded; one that
    # arrives after the deadline is not (its item is already a cutoff).
    key = (item['item_id'], rep)
    with ctx['cut_lock']:
        if ctx['cut'].is_set():
            tpath.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
            row.update(status='skipped', error='answer arrived after the deadline')
            return row
        ctx['judging'].add(key)

    # 2. The grade.
    try:
        with ctx['judge_slots']:
            g = judge.grade(ctx['judge_ep'], ctx['judge_gen'], question=item['question'], target=item['answer'],
                            predicted=ans['answer'], timeout=ctx['judge_timeout'])
    except client.InfraError as e:
        transcript['judge_error'] = str(e)
        transcript['judge_calls'] = getattr(e, 'calls', None)
        tpath.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
        # Keep the answer in the row so a resume regrades without asking the model again.
        row.update(status='infra_error', stage='judge', error=str(e)[:500], score=None, answer_record=ans)
        return row
    transcript['judge'] = g
    row.update(judge_calls=len(g['calls']),
               judge_prompt_tokens=sum((c.get('usage') or {}).get('prompt_tokens') or 0 for c in g['calls']),
               judge_completion_tokens=sum((c.get('usage') or {}).get('completion_tokens') or 0 for c in g['calls']),
               judge_latency_s=round(sum(c.get('latency_s') or 0 for c in g['calls']), 3))
    tpath.write_text(json.dumps(transcript, ensure_ascii=False, indent=1))
    net = {'CORRECT': 1, 'INCORRECT': -1}.get(g['grade'], 0)
    # A response cut off at max_tokens is the model's failure: it is scored as graded (usually NOT_ATTEMPTED,
    # since the answer is empty), and flagged model_error so truncations stay visible.
    row.update(status='model_error' if truncated or timed_out else 'ok', grade=g['grade'], grade_source=g['source'],
               score=1 if g['grade'] == 'CORRECT' else 0, net=net)
    return row


def judge_tokens(rows: dict, expected: list) -> dict:
    """Judge prompt/completion tokens this run spent, from the transcripts' judge calls (cost, #41)."""
    tot = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'latency_s': 0.0}
    for k in expected:
        r = rows.get(k) or {}
        for key, out in (('judge_calls', 'calls'), ('judge_prompt_tokens', 'prompt_tokens'),
                         ('judge_completion_tokens', 'completion_tokens'), ('judge_latency_s', 'latency_s')):
            tot[out] += r.get(key) or 0
    tot['latency_s'] = round(tot['latency_s'], 1)
    return tot


def summarize(run_dir: Path, rows: dict, expected: list[tuple[str, int]], cfg: dict) -> dict:
    final = [rows[k] for k in expected if k in rows]
    scored = [r for r in final if r['status'] in ('ok', 'model_error', 'cutoff')]
    n_infra = sum(r['status'] == 'infra_error' for r in final)
    n_missing = len(expected) - len(scored) - n_infra
    by_item: dict[str, list[str]] = {}
    # status, not grade, marks a cutoff (rows written before the cutoff rule carry grade NOT_ATTEMPTED)
    grades = [metrics.CUTOFF if r['status'] == 'cutoff' else r['grade'] for r in scored]
    for r, g in zip(scored, grades):
        by_item.setdefault(r['item_id'], []).append(g)
    m = metrics.compute(grades)
    ci = metrics.bootstrap_ci(by_item)
    infra_rate = n_infra / len(expected) if expected else 0.0
    complete = n_missing == 0
    rnd = lambda v: None if v is None else round(v, 2)  # noqa: E731
    summary = {
        'benchmark': 'AA-Omniscience 1.1',
        'dataset': f'{cfg["dataset"]["repo"]}@{cfg["dataset"]["revision"][:7]}',
        'metric': metrics.METRIC_NAMES['accuracy'],
        'score': rnd(m['accuracy']),
        'ci95': ci['accuracy'],
        'metrics': {k: {'name': metrics.METRIC_NAMES[k], 'score': rnd(m[k]), 'ci95': ci[k]} for k in m},
        'grades': metrics.counts(grades),
        'n_items': len({k[0] for k in expected}),
        'n_repeats': cfg['repeats'],
        'n_scored': len(scored),
        'n_infra_errors': n_infra,
        'infra_error_rate': round(infra_rate, 4),
        'n_missing': n_missing,
        'n_pending': sum(r['status'] == 'pending' for r in final),
        'generation_complete': len(final) == len(expected) and not any(
            r['status'] == 'infra_error' and r.get('stage') == 'model' for r in final),
        'n_truncated': sum(bool(r.get('truncated')) for r in scored),
        'n_empty_answers': sum(r.get('grade_source') == 'rule_empty' for r in scored),
        'n_cutoff': sum(r['status'] == 'cutoff' for r in scored),
        'n_timeouts': sum(r.get('finish_reason') == 'timeout' for r in scored),
        'cutoff_rule': 'cutoffs: not correct (accuracy, index); excluded from the non-hallucination denominator; '
                       '*_cutoff_incorrect metrics count them INCORRECT',
        'deadline_s': cfg.get('deadline_s'),
        'complete': complete,
        'valid': complete and infra_rate <= INFRA_LIMIT,
        'subset': cfg['limit'] is not None and cfg['limit'] < cfg['dataset']['n'],
        'model': cfg['model']['served_name'],
        'judge': cfg['judge']['served_name'],
        'judge_arm': cfg['judge'].get('arm'),
        **protocol.summary_fields(model=cfg['model'].get('spec'),
                                  served_name=cfg['model'].get('served_name'), body=cfg['generation'],
                                  context=cfg.get('context'), judge_arm=cfg['judge'].get('arm'),
                                  judge_settings_key=None, code_sha=(cfg.get('git') or {}).get('sha'),
                                  repeats=cfg['repeats'], image_inputs=False),
        'aa_reference': cfg.get('aa_reference'),
        'wall_s': round(time.time() - cfg['_t0'], 1) if '_t0' in cfg else None,
        'concurrency': cfg.get('concurrency'),
        'judge_concurrency': cfg.get('judge_concurrency'),
        'judge_tokens': judge_tokens(rows, expected),
        'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    }
    summary = fold_summary(run_dir, summary)   # n_flagged: the M3.4 trace judge (#177)
    summary['aai_open_version'] = aai_open.of_config(cfg)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['all', 'generate', 'grade'], default='all',
                    help='generate saves ungraded answers without a live judge; grade reuses answers without a live model')
    ap.add_argument('--model', help='serving:<name> | url:<base>@<served_name> | openai:<model>')
    ap.add_argument('--answers-from', help='regrade the answers of an earlier run (a run id under --runs-root, or an '
                                           'absolute run dir) instead of calling a model')
    ap.add_argument('--judge', default=DEFAULT_JUDGE,
                    help=f'grader spec (default {DEFAULT_JUDGE}, a self-hosted open model). AA uses '
                         'openai:gpt-5.6-luna (medium), which needs --allow-external-api')
    ap.add_argument('--allow-external-api', action='store_true',
                    help='allow openai: specs (external API spend; the user has declined it for now)')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--dataset', default='v1.1', choices=sorted(data.DATASETS))
    ap.add_argument('--limit', type=int, help='domain-balanced subset of N questions (smoke tests)')
    ap.add_argument('--subset-seed', type=int, default=0)
    ap.add_argument('--ids', help='file with one question id per line: run exactly these (after --limit; recorded '
                                  'in config.json as item_ids; per-item repeat plans run one id set per run)')
    ap.add_argument('--repeats', type=int, default=5, help='M3 protocol: 5 (AA: 1 over 3,000 private questions)')
    ap.add_argument('--context-window', type=int, default=protocol.CONTEXT_WINDOW,
                    help='total tokens per request, prompt + output (M3.0: 131072 for regular evals; 0 = the '
                         "endpoint's full context, for Qwen3.8 calibration runs only)")
    ap.add_argument('--preset', help=f'generation preset {sorted(presets.PRESETS)}; default from served name')
    ap.add_argument('--aa-name', help='model name in AA\'s leaderboard (default from preset)')
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--top-p', type=float)
    ap.add_argument('--reasoning-effort', help='set wherever the model request carries an effort (top level and '
                                               'chat_template_kwargs); refused for models without effort levels')
    ap.add_argument('--max-tokens', help="'auto' (endpoint context minus a margin) or an integer")
    ap.add_argument('--chat-template-kwargs', help='JSON; replaces the preset value')
    ap.add_argument('--extra-body', help='JSON merged into every model request')
    ap.add_argument('--judge-preset')
    ap.add_argument('--judge-reasoning-effort', help='AA: medium (for gpt-5.6-luna)')
    ap.add_argument('--judge-extra-body', help='JSON merged into every judge request')
    ap.add_argument('--judge-arm', help='a judge arm from aa_evals/judges/arms.py (e.g. qwen38-27b-medium-pin, '
                                        'super-full-pin): sets --judge to its server and its settings as the extra '
                                        'body (a --judge-extra-body merges over them); recorded as judge.arm. '
                                        f'Default: the pinned arm for this benchmark, {pins.default_arm(BENCH)}, '
                                        'unless --judge names another judge')
    ap.add_argument('--concurrency', type=int, default=32)
    ap.add_argument('--dispatch', metavar='DIR', help='#344: model requests take slots of this dispatcher\'s shared '
                    'in-flight budget (aa_evals.common.dispatch), longest predicted first; replaces --concurrency '
                    'for the model (the judge keeps --judge-concurrency)')
    ap.add_argument('--timeout', type=float, default=3600, help='per model request, seconds')
    ap.add_argument('--judge-timeout', type=float, default=900)
    ap.add_argument('--judge-concurrency', type=int,
                    help='max judge requests in flight (default --concurrency); keep a shared judge server at <=16')
    ap.add_argument('--deadline-s', type=float,
                    help='no tails: after this many seconds, unfinished items count as 0 (status cutoff) and the run '
                         'ends; a resume retries them')
    ap.add_argument('--note', action='append', default=[], help='free text recorded in config.json (repeatable)')
    ap.add_argument('--runs-root', type=Path, default=RUNS_ROOT)
    args = ap.parse_args(argv)
    source_cfg = None
    if args.stage == 'grade':
        source = Path(args.answers_from) if args.answers_from and Path(args.answers_from).is_absolute() else args.runs_root / (args.answers_from or args.run_id)
        source_cfg = json.loads((source / 'config.json').read_text())
        args.answers_from, args.model = str(source), None
        args.dataset = source_cfg['dataset']['name']
        args.limit, args.subset_seed, args.repeats = source_cfg['limit'], source_cfg['subset_seed'], source_cfg['repeats']
        item_ids = source_cfg.get('item_ids')
    if bool(args.model) == bool(args.answers_from):
        ap.error('give exactly one of --model and --answers-from')
    if args.judge_arm is None and args.judge == DEFAULT_JUDGE and not args.judge_preset:
        args.judge_arm = pins.default_arm(BENCH)   # the benchmark's pinned judge (#41 audit 1, #175)
    if args.judge_arm:
        from aa_evals.judges import arms
        if args.judge_arm not in arms.ARMS:
            ap.error(f'unknown --judge-arm {args.judge_arm}; one of {sorted(arms.ARMS)}')
        arm = arms.ARMS[args.judge_arm]
        if args.judge != DEFAULT_JUDGE and args.judge != f'serving:{arm.server}':
            ap.error(f'--judge {args.judge} conflicts with --judge-arm {arm.name} (server {arm.server})')
        args.judge = f'serving:{arm.server}'
        args.judge_extra_body = json.dumps({**arm.extra, **parse_json_arg(args.judge_extra_body)})
    active_specs = (args.model,) if args.stage == 'generate' else (args.judge,) if args.stage == 'grade' else (args.model, args.judge)
    external = [s for s in active_specs if s and s.startswith('openai:')]
    if external and not args.allow_external_api:
        ap.error(f'{external[0]} is an external API (spend); the user declined closed API judges on 2026-11-02. '
                 'Pass --allow-external-api only with a fresh approval.')

    run_dir = args.runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tdir = run_dir / 'transcripts'
    tdir.mkdir(exist_ok=True)

    spec, items = data.load(args.dataset)
    items = data.subset(items, args.limit, args.subset_seed)
    if args.stage != 'grade':
        item_ids = sorted({l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()},
                          key=int) if args.ids else None
    if item_ids is not None:
        missing = set(item_ids) - {it['item_id'] for it in items}
        if missing:
            ap.error(f'--ids: {len(missing)} ids not in the dataset (after --limit), e.g. {sorted(missing)[:3]}')
        items = [it for it in items if it['item_id'] in set(item_ids)]
    expected = [(it['item_id'], r) for it in items for r in range(args.repeats)]

    answers_from = answers_from_dir = None
    if args.answers_from:
        answers_from_dir = Path(args.answers_from) if Path(args.answers_from).is_absolute() \
            else args.runs_root / args.answers_from
        src_cfg = json.loads((answers_from_dir / 'config.json').read_text())
        if src_cfg['dataset']['name'] != args.dataset:
            ap.error(f'--answers-from run used dataset {src_cfg["dataset"]["name"]}, not {args.dataset}')
        answers_from = load_rows(answers_from_dir / 'items.jsonl')
        model_ep, gen = None, src_cfg['generation']
        context = src_cfg.get('context')
        model_desc = dict(src_cfg['model'], answers_from=str(answers_from_dir))
        preset_name, aa_name = src_cfg.get('preset'), args.aa_name or (src_cfg.get('aa_reference') or {}).get('model')
    else:
        model_ep = client.resolve(args.model)
        preset_name, aa_name, gen = build_gen(args, model_ep)
        gen, context = resolve_max_tokens(model_ep, gen, args.context_window)
        model_desc = model_ep.describe()
    judge_ep = client.resolve(args.judge) if args.stage != 'generate' else None
    jgen = judge_gen(judge_ep, args) if judge_ep else {}

    cfg = {
        'benchmark': 'AA-Omniscience 1.1', 'run_id': args.run_id, 'argv': sys.argv, 'host': socket.gethostname(),
        'started_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'stage': args.stage, 'git': source_cfg.get('git') if source_cfg else git_info(), 'stage_git': git_info(),
        'dataset': {'name': spec.name, 'repo': spec.repo, 'revision': spec.revision, 'sha256': spec.sha256,
                    'n': spec.n, 'aa_version': spec.aa_version},
        'limit': args.limit, 'subset_seed': args.subset_seed, 'repeats': args.repeats,
        **({'item_ids': item_ids} if item_ids is not None else {}),
        'model': model_desc, 'preset': preset_name, 'generation': gen, 'context': context,
        'effort': protocol.effort_of(gen),
        'prompt': {'system': 'OMNISCIENCE_ANSWER_PROMPT (AA card, verbatim)', 'user': 'question text'},
        'judge': dict(judge_ep.describe() if judge_ep else {'spec': args.judge, 'served_name': None}, generation=jgen, arm=args.judge_arm,
                      prompt='OMNISCIENCE_GRADER_TEMPLATE (AA card, verbatim), one user message',
                      empty_answer_rule='empty answer -> NOT_ATTEMPTED without a call'),
        'harness': 'aa_evals.omniscience.run: one chat completion per question, no tools',
        'concurrency': args.concurrency, 'judge_concurrency': args.judge_concurrency or args.concurrency,
        'timeout_s': args.timeout, 'deadline_s': args.deadline_s,
        'dispatch': args.dispatch and str(Path(args.dispatch).resolve()),
        'aa_reference': aa_reference(aa_name),
        'notes': args.note,
    }
    # #481: the AAI-open version (the launcher's label, or that of the run whose answers this one grades)
    inherit = src_cfg if args.answers_from else None
    dev = [] if inherit else aai_open.client_deviations(
        context_window=(context or {}).get('window'), overrides={'temperature': args.temperature, 'top_p': args.top_p},
        flags={'--extra-body': args.extra_body, '--chat-template-kwargs': args.chat_template_kwargs,
               '--max-tokens': args.max_tokens, '--preset': args.preset})
    dev += aai_open.judge_deviations(args.judge_arm) if args.stage != 'generate' else []
    cfg['aai_open'] = aai_open.record(dev, source_cfg=inherit)
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for k in ('dataset', 'limit', 'subset_seed', 'repeats', 'item_ids'):
            if old.get(k) != cfg.get(k):
                ap.error(f'{run_dir} exists with a different {k}: {old.get(k)} != {cfg.get(k)}')
        changing_stage = old.get('stage') == 'generate' or args.stage == 'generate'
        if old['model'].get('served_name') != cfg['model'].get('served_name') or (not changing_stage and
                old['judge'].get('served_name') != cfg['judge'].get('served_name')):
            ap.error(f'{run_dir} exists with a different model or judge; use a new --run-id')
        cfg['resumed'] = old.get('resumed', []) + [old.get('started_at')]
        cfg['aai_open'] = aai_open.record(dev, source_cfg=old)   # a resumed run keeps the label it started with
        cfg['stage_history'] = old.get('stage_history', []) + [{'stage': old.get('stage', 'all'), 'git': old.get('stage_git', old.get('git'))}]
        if args.stage == 'generate' and old.get('stage') != 'generate':
            cfg['judge'] = old['judge']
    cfg_path.write_text(json.dumps(cfg, indent=1))

    items_path = run_dir / 'items.jsonl'
    rows = load_rows(items_path)
    todo = []
    by_id = {it['item_id']: it for it in items}
    for iid, rep in expected:
        prior = rows.get((iid, rep))
        if prior and (prior['status'] in ('ok', 'model_error') or (args.stage == 'generate' and prior['status'] == 'pending')):
            continue
        todo.append((by_id[iid], rep, prior if prior and (prior.get('stage') == 'judge' or prior['status'] == 'pending') else None))
    print(f'[{args.run_id}] {len(expected)} item x repeat, {len(expected) - len(todo)} done, {len(todo)} to run; '
          f'model={model_desc.get("served_name")} judge={judge_ep.model if judge_ep else "deferred"} gen={request_body(gen)}', flush=True)

    ctx = {'stage': args.stage, 'run_dir': run_dir, 'tdir': tdir, 'model_ep': model_ep, 'gen': gen, 'judge_ep': judge_ep,
           'judge_gen': jgen, 'timeout': args.timeout, 'judge_timeout': args.judge_timeout,
           'judge_slots': threading.BoundedSemaphore(args.judge_concurrency or args.concurrency),
           'answers_from': answers_from, 'answers_from_dir': answers_from_dir,
           'stop': threading.Event(), 'consecutive_infra': [0],
           'cut': threading.Event(), 'cut_lock': threading.Lock(), 'judging': set()}
    writer = Writer(items_path)
    t0 = time.time()
    cfg['_t0'] = t0
    done = 0
    deadline = t0 + args.deadline_s if args.deadline_s else None
    cut = False
    lane = None
    if args.dispatch and args.stage != 'grade' and todo:
        from aa_evals.common.dispatch.lane import open_dispatch
        lane, pred, _ = open_dispatch(args.dispatch, 'omniscience', run_dir)
        ctx.update(lane=lane, pred=pred)
        todo.sort(key=lambda t: -pred(t[0]['item_id']))   # the pool starts them in this order
        # announce only the attempts that will call the model (not a stored answer awaiting its judge)
        lane.expect([(f"{t[0]['item_id']}.r{t[1]}", t[0]['item_id'], pred(t[0]['item_id'])) for t in todo
                     if not (t[2] or {}).get('answer_record') and answers_from is None])
    pool = cf.ThreadPoolExecutor(max_workers=min(len(todo), 1024) or 1 if lane else args.concurrency)
    task_of = {pool.submit(process, t, ctx): t for t in todo}
    pending = set(task_of)
    while pending:
        wait_s = None if deadline is None else max(0.0, deadline - time.time())
        finished, pending = cf.wait(pending, timeout=wait_s, return_when=cf.FIRST_COMPLETED)
        for fut in finished:
            row = fut.result()
            if row['status'] == 'skipped':
                continue
            writer.append(row)
            rows[(row['item_id'], row['repeat'])] = row
            done += 1
            if done % 10 == 0 or done == len(todo):
                k = metrics.counts(r['grade'] for r in rows.values() if r.get('grade'))
                n_inf = sum(r['status'] == 'infra_error' for r in rows.values())
                print(f'[{args.run_id}] {done}/{len(todo)} in {time.time() - t0:.0f}s grades={k} infra={n_inf}',
                      flush=True)
        if pending and deadline is not None and time.time() >= deadline:
            cut = True
            with ctx['cut_lock']:
                ctx['cut'].set()
                grading = {f for f in pending if (task_of[f][0]['item_id'], task_of[f][1]) in ctx['judging']}
            if grading:  # answers in hand: grade them (bounded by the judge's own timeout and retries)
                print(f'[{args.run_id}] deadline: grading {len(grading)} answers already in hand', flush=True)
                fin, _ = cf.wait(grading, timeout=args.judge_timeout * 2)
                for fut in fin:
                    row = fut.result()
                    writer.append(row)
                    rows[(row['item_id'], row['repeat'])] = row
                pending -= fin
            for fut in pending:
                item, rep, _ = task_of[fut]
                row = cutoff_row({'item_id': item['item_id'], 'repeat': rep, 'domain': item['domain'],
                                  'subtopic': item['subtopic'], 'gold': item['answer']})
                writer.append(row)
                rows[(item['item_id'], rep)] = row
            print(f'[{args.run_id}] deadline {args.deadline_s:.0f}s: {len(pending)} unfinished items count as 0 '
                  f'(status cutoff)', flush=True)
            break
    pool.shutdown(wait=not cut, cancel_futures=True)
    if lane:
        lane.close()
    summary = summarize(run_dir, rows, expected, cfg)
    print(json.dumps({k: summary[k] for k in ('metric', 'score', 'ci95', 'metrics', 'grades', 'n_scored',
                                               'n_infra_errors', 'n_truncated', 'n_cutoff', 'complete',
                                               'valid', 'aa_reference')}, indent=1), flush=True)
    success = summary['generation_complete'] if args.stage == 'generate' else summary['valid']
    rc = 0 if success else 2
    if ctx['stop'].is_set():
        print(f'[{args.run_id}] stopped after {CIRCUIT_BREAKER} consecutive model infra errors; '
              're-run the same command to resume', file=sys.stderr)
        rc = 3
    if cut and argv is None:  # CLI: in-flight requests can't be cancelled, so leave without waiting for them
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())

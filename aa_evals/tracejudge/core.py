"""Judge the finished attempts of a run: rules, then the LLM where the rules can't decide; append flags.jsonl."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from aa_evals.tracejudge import digest as dg
from aa_evals.tracejudge.hook import FLAGS, THRESHOLD, read_flags
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.rules import INFRA, MODEL, NOTE, SUSPICIOUS, decide, demotion, egress_blocked, remote_limits, run_rules
from aa_evals.tracejudge.trace import Trace

POLICIES = ('fails', 'suspicious', 'all', 'every', 'none')


def code_sha(root: Path | None = None) -> str | None:
    """The commit of the code running the judge. A git checkout: its HEAD, only if the repo's toplevel is this code
    root (a pi_eval code snapshot is not a repo, and git would walk up to an unrelated enclosing one). A snapshot: the
    git_sha its `.complete` recorded, or the GIT_SHA/FROZEN_SHA marker used by other submitters. Otherwise None."""
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if top and Path(top).resolve() == root:
            return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return json.loads((root / '.complete').read_text()).get('git_sha') or None
    except (OSError, ValueError, AttributeError):
        pass
    for marker in ('FROZEN_SHA', 'GIT_SHA'):
        try:
            value = (root / marker).read_text().strip()
        except OSError:
            continue
        if value.startswith('{'):
            try:
                record = json.loads(value)
                value = record.get('sha') or record.get('git_sha')
                if value and record.get('dirty'):
                    value += '-dirty'
            except (ValueError, AttributeError):
                continue
        if value:
            return value
    return None


def failed(t: Trace) -> bool:
    s = t.recorded.get('score')
    if t.recorded.get('status') == 'infra_error':
        return True
    try:
        return s is None or float(s) < 1.0
    except (TypeError, ValueError):
        return True


def needs_llm(policy: str, verdict: str, t: Trace) -> bool:
    """fails (default): suspicious attempts plus every failed attempt the rules found clean; suspicious: only those
    the rules can't decide; all: everything but decisive model outcomes; every: every attempt;
    none: rules only."""
    if policy == 'every':
        return True
    if policy == 'none' or verdict == MODEL:
        return False
    if verdict == INFRA:
        return policy == 'all'
    if verdict == SUSPICIOUS:
        return True
    return policy == 'all' or (policy == 'fails' and failed(t))


def build_row(t: Trace, hits, verdict, llm: dict | None, judge_desc: dict | None, digest_chars: int | None,
              threshold: float = THRESHOLD) -> dict:
    flags = [{'category': h.category, 'evidence': h.evidence[:400], 'ref': h.ref, 'confidence': h.confidence,
              'source': f'rule:{h.rule}'} for h in hits if h.polarity == INFRA]
    if llm and llm.get('ok'):
        for f in llm.get('flags') or []:
            f = {**f, 'source': 'llm'}
            why = demotion(t, f)   # the benchmark's tool answering the model, a crash in the model's solution, ...
            if why:                # kept for audit, below the threshold
                f |= {'confidence': min(f.get('confidence', 0), 0.3), 'demoted': why}
            flags.append(f)
    flagged = any(f['confidence'] >= threshold for f in flags)
    r = t.recorded
    return {
        'key': t.key, 'item_id': t.item_id, 'repeat': t.repeat, 'bench': t.bench, 'kind': t.kind,
        'counted': t.counted, 'known_infra': t.known_infra,
        'recorded': {k: r.get(k) for k in ('status', 'score', 'end_state', 'exception_type', 'infra_reason',
                                            'finish_reason', 'error_kind', 'agent_internet', 'agent_network',
                                            'scicode_context_version', 'offline_proxy_review_required',
                                            'offline_proxy_review_pending', 'offline_proxy_review_evidence',
                                            'offline_proxy_review_status', 'offline_proxy_review_receipt_sha256',
                                            'offline_proxy_review_index_audit_sha256', 'offline_proxy_review_source_audit_sha256',
                                            'offline_proxy_review_host_probe_audit_sha256',
                                            'offline_proxy_review_error', 'grading_replay',
                                            'grading_replay_review_pending', 'grading_replay_review_error') if r.get(k) is not None},
        'prefilter': verdict,
        'hits': [h.as_dict() | {'evidence': h.evidence[:300]} for h in hits if h.polarity != NOTE],
        'notes': [h.rule for h in hits if h.polarity == NOTE],
        'flags': flags, 'flagged': flagged,
        'remote_limited': remote_limits(t) if t.kind == 'agentic' else {},   # sites refusing scrapes (429/403/challenge)
        'egress_blocked': egress_blocked(t)[0] if t.kind == 'agentic' else {},   # proxy up, one upstream unreachable
        'categories': sorted({f['category'] for f in flags if f['confidence'] >= threshold}),
        'llm': None if llm is None else {k: llm.get(k) for k in ('ok', 'reason', 'error', 'latency_s', 'prompt_tokens',
                                                                  'completion_tokens', 'calls') if k != 'calls' or k in llm},
        'judge': judge_desc if llm is not None else {'rules_only': True},
        'digest_chars': digest_chars, 'judged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'code_sha': CODE_SHA,
    }


CODE_SHA = code_sha()


async def judge_keys(layout, keys, judge, policy: str, out_path: Path, concurrency: int = 16,
                     budget: int = dg.BUDGET, log=print) -> list[dict]:
    """Judge `keys` of `layout`; append each row to out_path as soon as it's done."""
    sem = asyncio.Semaphore(concurrency)
    load_sem = asyncio.Semaphore(max(2, min(8, os.cpu_count() or 4)))
    rows, desc = [], (judge.describe() if judge else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'a')

    async def one(k):
        async with sem:
            try:
                async with load_sem:
                    t = await asyncio.to_thread(layout.load, k)
            except Exception as e:   # an unreadable attempt is itself worth a look
                row = {'key': k, 'bench': layout.bench, 'counted': True, 'known_infra': False, 'flagged': False,
                       'flags': [], 'load_error': repr(e)[:400], 'judged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                fh.write(json.dumps(row) + '\n'); fh.flush()
                rows.append(row)
                return
            hits = run_rules(t)
            verdict = decide(hits)
            llm, dchars = None, None
            if judge is not None and needs_llm(policy, verdict, t):
                text = dg.render(t, hits, budget)
                dchars = len(text)
                llm = await judge.judge(text)
            row = build_row(t, hits, verdict, llm, desc, dchars)
            fh.write(json.dumps(row) + '\n'); fh.flush()
            rows.append(row)
            if row['flagged']:
                log(f"FLAG {t.key}: {','.join(row['categories'])} "
                    f"{'(known)' if row['known_infra'] else ''} {(row['flags'][0]['evidence'] if row['flags'] else '')[:160]!r}")

    try:
        await asyncio.gather(*(one(k) for k in keys))
    finally:
        fh.close()
    return rows


def pending(layout, out_path: Path, rejudge: bool = False) -> list[str]:
    done = set() if rejudge else set(read_flags(None, out_path))
    return [k for k in layout.finished() if k not in done]


async def judge_run(run_dir, judge, policy='fails', out=None, concurrency=16, rejudge=False, keys=None,
                    layout_name=None, log=print):
    layout = open_run(run_dir, layout_name)
    out_path = Path(out) if out else Path(run_dir) / FLAGS
    todo = keys if keys is not None else pending(layout, out_path, rejudge)
    log(f'{run_dir}: layout {layout.name}, bench {layout.bench}, {len(todo)} attempts to judge -> {out_path}')
    return await judge_keys(layout, todo, judge, policy, out_path, concurrency, log=log)


async def watch_run(run_dir, judge, policy='fails', out=None, concurrency=16, interval=60, fold=True,
                    idle_exit_h: float = 0, layout_name=None, log=print):
    """Follow a running run: judge each attempt as it finishes; stop when the run is complete and every finished
    attempt is judged (then fold n_flagged into summary.json when `fold`)."""
    from aa_evals.tracejudge.hook import fold_file
    out_path = Path(out) if out else Path(run_dir) / FLAGS
    last_new = time.time()
    while True:
        layout = open_run(run_dir, layout_name)
        todo = pending(layout, out_path)
        if todo:
            last_new = time.time()
            await judge_keys(layout, todo, judge, policy, out_path, concurrency, log=log)
        elif layout.complete():
            if fold:
                s = fold_file(run_dir, path=out_path)
                log(f'complete; folded into summary.json: n_flagged={s.get("n_flagged") if s else None}')
            return
        elif idle_exit_h and time.time() - last_new > idle_exit_h * 3600:
            log(f'no new attempts for {idle_exit_h} h; exiting')
            return
        await asyncio.sleep(interval)

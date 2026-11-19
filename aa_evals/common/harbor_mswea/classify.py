"""Outcome of one physical attempt (one Harbor trial of mini-swe-agent): infrastructure failure or model outcome.

Infrastructure (retried, then excluded from the score; rules of issue #18, imported from eval_q35/worker.py):
  - no Harbor result, worker stopped/lost
  - environment build/start, agent setup and sandbox errors without a verifier reward (eval_q35 INFRA), and any
    error raised before the agent was set up (the model never ran)
  - the model backend unreachable after mini's retries: mini's exit status is a litellm connection/server error
    (eval_q35 LLM_INFRA), an endpoint abort (model not served, auth, unsupported params), or an HTTP 400 that is
    not a context-length error (a request we built wrong)
  - mini never started (agent exited non-zero without writing a trajectory)
  - the cluster proxy refused/reset connections in the agent's tool outputs, or in a verifier whose tests never
    started (eval_q35 PROXY_FAIL)
  - a verifier timeout inside the tests' network bootstrap (eval_q35 verifier_bootstrap_timeout)

Model outcomes (scored): the verifier's reward.  A full context (no compaction) is `context_overflow`, whether it
ends as an error or as three responses cut at the window's end (RepeatedFormatError). Without a reward (verifier timeout, AA: "a verifier that exceeds its
timeout counts as a failure"), the attempt scores 0. Mini's exit status is kept: Submitted, LimitsExceeded (step
limit), RepeatedFormatError (3 responses in a row without a valid bash tool call), ContextWindowExceededError
(no compaction: the episode ends, the verifier scores the final state), or none (Harbor's wall-clock backstop).

Time limit (user decision, issue #28): the task's agent timeout counts environment time only, i.e. wall time minus
model time. Model time = for every model response, the time since the previous timestamped message (mini stamps
each response and each batch of tool outputs; format-error retries carry no stamp, so their model time is
included). An attempt whose environment time exceeds the task's limit scores 0 (`env_time_exceeded`); Harbor's
wall-clock limit is a backstop at `backstop_multiplier` x the task's limit.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from eval_q35.worker import INFRA, LLM_INFRA, PROXY_FAIL, verifier_bootstrap_timeout

TRAJ = 'mini-swe-agent.trajectory.json'
# mini's abort exceptions that say nothing about the model: the endpoint does not serve the model (a restart),
# refuses our credentials, or rejects our parameters
ENDPOINT_ABORTS = ('NotFoundError', 'PermissionDeniedError', 'AuthenticationError', 'UnsupportedParamsError')
CONTEXT = re.compile(r'context length|context window|maximum context|max_model_len|too many tokens|'
                     r'prompt is too long|longer than the model', re.I)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def find_trial(raw: Path) -> Path | None:
    trials = [p.parent for p in sorted(Path(raw).glob('*/*/result.json'))]
    return trials[0] if trials else None


def trajectory(trial: Path) -> dict | None:
    return _read(Path(trial) / 'agent' / TRAJ)


def finish_reason(msg: dict) -> str | None:
    resp = (msg.get('extra') or {}).get('response')
    if isinstance(resp, dict):
        ch = resp.get('choices') or [{}]
        return (ch[0] or {}).get('finish_reason')
    return None


def stats(traj: dict | None, result: dict, max_model_len: int | None = None) -> dict:
    """Steps, exit status, finish reasons, model/environment time of one mini trajectory.

    context_cuts: responses cut by the context window (finish_reason length with prompt + completion at the
    server's max_model_len; with max_tokens unset the server allows exactly the rest of the window).
    empty_tool_args: tool calls the server's parser returned without arguments (e.g. Qwen3-Coder XML whose
    command contains `</function>`: the parser ends the call there)."""
    ae = result.get('agent_execution') or {}
    t0, t1 = _ts(ae.get('started_at')), _ts(ae.get('finished_at'))
    wall = (t1 - t0) if t0 and t1 else None
    out = {'agent_wall_sec': round(wall, 1) if wall is not None else None}
    if not traj:
        return out
    info = traj.get('info') or {}
    msgs = traj.get('messages') or []
    reasons: Counter = Counter()
    fmt_errors = 0
    context_cuts = 0
    empty_args = 0
    max_total = 0
    model_sec = 0.0
    prev = t0
    last_exit = None
    for m in msgs:
        ex = m.get('extra') or {}
        role = m.get('role')
        if role == 'assistant' or (role == 'user' and ex.get('interrupt_type') == 'FormatError'):
            r = finish_reason(m)
            if r:
                reasons[r] += 1
            resp = ex.get('response') if isinstance(ex.get('response'), dict) else {}
            u = resp.get('usage') or {}
            total = int(u.get('prompt_tokens') or 0) + int(u.get('completion_tokens') or 0)
            max_total = max(max_total, total)
            if r == 'length' and max_model_len and total >= max_model_len - 16:
                context_cuts += 1
            for tc in (((resp.get('choices') or [{}])[0] or {}).get('message') or {}).get('tool_calls') or []:
                if str((tc.get('function') or {}).get('arguments') or '').strip() in ('', '{}'):
                    empty_args += 1
        if role == 'user' and ex.get('interrupt_type') == 'FormatError':
            fmt_errors += 1
        ts = ex.get('timestamp')
        if isinstance(ts, (int, float)):
            if role == 'assistant' and prev is not None:
                model_sec += max(0.0, ts - prev)
            prev = ts
        if role == 'exit':
            last_exit = ex
    out.update(
        exit_status=info.get('exit_status') or (last_exit or {}).get('exit_status') or None,
        exit_exception=((last_exit or {}).get('exception_str') or '')[:1000] or None,
        api_calls=(info.get('model_stats') or {}).get('api_calls'),
        n_messages=len(msgs), format_errors=fmt_errors, finish_reasons=dict(reasons),
        context_cuts=context_cuts, empty_tool_args=empty_args, max_context_tokens=max_total or None,
        mini_version=info.get('mini_version'),
        model_sec=round(model_sec, 1),
        env_sec=round(wall - model_sec, 1) if wall is not None else None)
    return out


def proxy_failure(trial: Path, traj: dict | None) -> str | None:
    """A cluster-proxy connection failure in the agent's tool outputs, or in a verifier whose tests never started."""
    texts = []
    for m in (traj or {}).get('messages') or []:
        if m.get('role') == 'tool' or ((m.get('extra') or {}).get('raw_output') is not None):
            texts.append(('agent', str((m.get('extra') or {}).get('raw_output') or m.get('content') or '')))
    v = Path(trial) / 'verifier'
    if not (v / 'ctrf.json').exists():
        try:
            texts.append(('verifier', (v / 'test-stdout.txt').read_text(errors='replace')))
        except OSError:
            pass
    for where, text in texts:
        m = PROXY_FAIL.search(text)
        if m:
            return f'{where}: {m.group(0)[:200]}'
    return None


def classify(raw: Path, returncode: int | None, stopped: bool | str = False, *, time_limit_sec: float | None = None,
             max_model_len: int | None = None) -> dict:
    trial = find_trial(raw)
    out: dict = {'returncode': returncode, 'trial_dir': str(trial) if trial else None}
    if stopped:
        reason = stopped if isinstance(stopped, str) else 'worker_stopped'
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': reason, 'score': None,
                'exception_type': 'WorkerStopped'}
    if trial is None:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'no_harbor_result', 'score': None,
                'exception_type': 'NoHarborResult'}
    result = _read(trial / 'result.json', {}) or {}
    exc = result.get('exception_info') or {}
    etype = exc.get('exception_type') or ''
    message = exc.get('exception_message') or ''
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    traj = trajectory(trial)
    st = stats(traj, result, max_model_len)
    out.update(exception_type=etype or None, exception_message=message[:2000] or None, reward=reward, **st)
    exit_status = st.get('exit_status') or ''

    def infra(reason: str, **kw) -> dict:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': reason, 'score': None, **kw}

    proxy = proxy_failure(trial, traj)
    if proxy:
        return infra('proxy_outage', proxy_failure=proxy)
    stall = verifier_bootstrap_timeout(trial, etype)
    if stall:
        return infra('verifier_bootstrap_timeout', verifier_stall=stall)
    if reward is None and any(t in etype for t in INFRA):
        return infra(etype)
    if etype and reward is None and not result.get('agent_setup') and not result.get('agent_execution'):
        # the trial failed before the agent was set up (e.g. a compose service never became healthy: harbor_ns
        # raises a plain RuntimeError); the model never ran
        return infra('environment_not_started')
    if exit_status in LLM_INFRA:
        return infra('llm_backend_unreachable')
    if exit_status in ENDPOINT_ABORTS:
        return infra('endpoint_rejected')
    if exit_status == 'BadRequestError' and not CONTEXT.search(st.get('exit_exception') or ''):
        return infra('bad_request')
    if traj is None and etype == 'NonZeroAgentExitCodeError':
        return infra('agent_did_not_start')

    out['infrastructure_failure'] = False
    # no compaction: the episode ends when the context is full, either as an error or as mini's three format
    # errors in a row when every response is cut at the window's end
    out['context_overflow'] = exit_status == 'ContextWindowExceededError' or (
        exit_status == 'BadRequestError' and bool(CONTEXT.search(st.get('exit_exception') or ''))) or (
        exit_status == 'RepeatedFormatError' and st.get('context_cuts', 0) >= 1)
    if isinstance(reward, (int, float)):
        out['score'] = 1.0 if float(reward) >= 1.0 else 0.0
        out['status'] = 'ok'
    else:
        out['score'] = 0.0
        out['status'] = 'model_error'
        out['unscored'] = True
    env = st.get('env_sec')
    if time_limit_sec and env is not None and env > time_limit_sec:
        out.update(score=0.0, env_time_exceeded=True)
    return out

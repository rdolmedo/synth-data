"""Deterministic pre-filter: exit codes, harness events and error strings.

Each rule returns Hits. A hit has a polarity:
  infra       decisive infra evidence: flagged without the LLM;
  note        context for the LLM (e.g. the memory cap killed a task process), never decisive;
  model       decisive evidence of a model outcome (e.g. the model killed its own agent): not flagged, no LLM;
  suspicious  something the rules can't decide: the LLM reads the digest (with the hit as a hint).
`decide()` folds the hits into one pre-filter verdict.

What is NOT infra here (protocol limits and model outcomes, #66/#67):
  - the context window running out, the per-turn output cap, the agent time limit (env clock / AgentTimeoutError);
  - the sandbox memory cap killing a task process the model started (AA's E2B box does the same);
  - the model killing its own agent (pkill -f matching the agent's command line, kill <pi pid>);
  - a verifier timeout on the model's work (AA's rule; pi_eval scores it 0);
  - an HLE/Omniscience response that ends inside its reasoning (Qwen3.8 medium EOS-in-reasoning is model behavior).
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from aa_evals.tracejudge.trace import Trace

INFRA, MODEL, SUSPICIOUS, NOTE = 'infra', 'model', 'suspicious', 'note'   # note: context for the LLM only


@dataclass
class Hit:
    rule: str
    polarity: str
    category: str | None
    evidence: str
    ref: str
    confidence: float

    def as_dict(self):
        return asdict(self)


# pipeline infra reasons -> category (the pipeline already retried or excluded these; flagged as `known`)
KNOWN_REASON = {
    'worker_stopped': 'harness_cutoff', 'endpoint_moved': 'network', 'proxy_outage': 'network',
    'pi_backend_unreachable': 'network', 'pi_backend_timeout': 'network', 'llm_backend_unreachable': 'network',
    'hung_session': 'network',   # pi_eval hung-session watchdog (#531 A5): a stalled request or an idle Pi
    'response_budget_accounting': 'network',   # the guard's /detokenize under serving load (#531)
    'verifier_no_reward': 'grader_crash', 'verifier_bootstrap_timeout': 'grader_crash',
    'no_harbor_result': 'harness_error', 'pi_startup_before_inference': 'harness_error',
}

ENV_ERR = [   # tool output that points at the machine, not the model's code (error-message form only, not errno tables)
    (r'(?:[:\]] |^)No space left on device', 'sandbox_kill'),
    (r'(?:[:\]] |^)Read-only file system', 'sandbox_kill'),
    (r'fork: (retry: )?Resource temporarily unavailable|Cannot fork|pthread_create failed', 'sandbox_kill'),
    (r'(?:[:\]] |^)Too many open files(?! in system)', 'sandbox_kill'),
    (r'Transport endpoint is not connected|Stale file handle', 'sandbox_kill'),
    (r'Proxy tunneling failed|Tunnel connection failed|ProxyError|Unable to connect to proxy|407 Proxy|'
     r'proxy.{0,30}\b(refused|50[234])\b', 'network'),
    (r'Temporary failure in name resolution|Could not resolve host(?!name \[)', 'network_dns'),
]
NOT_ERROR_LINE = re.compile(r'#define|/\*|strerror|E[A-Z]{3,}\s+\d+|^\s*\d+:\s+[0-9a-f]+\s')
TEMPLATE_TOKENS = re.compile(r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<start_of_turn>|<end_of_turn>|<\|eot_id\|>|'
                             r'<\|start_header_id\|>|<\|channel\|>|<\|turn>')
TOOL_PARSE = re.compile(r'<tool_call>|</tool_call>|<function=|<\|tool_call|\[TOOL_CALLS\]|"name"\s*:\s*"bash"\s*,\s*"arguments"')
SELF_KILL = re.compile(r'\b(pkill|killall|kill)\b')


def _hits_known(t: Trace) -> list[Hit]:
    r, out = t.recorded, []
    if r.get('infrastructure_failure') or r.get('status') == 'infra_error':
        reason = r.get('infra_reason') or r.get('error_kind') or r.get('error_stage') or r.get('error') or 'infra_error'
        cat = KNOWN_REASON.get(str(reason))
        if cat is None:
            s = str(reason).lower()
            cat = ('network' if re.search(r'connect|timeout|unreach|5\d\d|proxy|endpoint', s) else
                   'grader_crash' if re.search(r'judge|grad|verif', s) else 'harness_error')
        out.append(Hit('pipeline_infra', INFRA, cat, f'pipeline recorded infra: {str(reason)[:200]}',
                       'outcome/items status', 1.0))
    if r.get('cutoff'):
        out.append(Hit('run_cutoff', INFRA, 'harness_cutoff', f"stopped at the run cutoff (end_state {r.get('end_state')})",
                       'outcome.json', 1.0))
    return out


def _hits_pi(t: Trace) -> list[Hit]:
    out, r = [], t.recorded
    agent_killed = False
    for e in t.events:
        m = re.search(r'harbor_ns memcap: .*killed pid (\d+) \(([^,)]+)', e.text)
        if m:
            comm = m.group(2).strip()
            if comm.split()[0] in ('pi', 'node') or comm.startswith('pi '):
                agent_killed = True
                out.append(Hit('memcap_killed_agent', INFRA, 'sandbox_kill', e.text, e.ref, 0.95))
            else:
                out.append(Hit('memcap_killed_task_process', NOTE, None,
                               f'the memory cap killed a process the model started, not the agent (a model outcome): {e.text}', e.ref, 0.9))
        if re.search(r'oom-kill|Out of memory: Killed process|oom_reaper|Memory cgroup out of memory', e.text):
            out.append(Hit('kernel_oom', SUSPICIOUS, 'sandbox_kill', e.text, e.ref, 0.6))
        if e.source == 'exception.txt' and re.search(r'RewardFileNotFound|verifier.*(crash|error)|ctrf', e.text, re.I) \
                and 'VerifierTimeoutError' not in e.text:
            out.append(Hit('verifier_exception', SUSPICIOUS, 'grader_crash', e.text[-300:], e.ref, 0.6))
        if e.source == 'verifier' and 'crash frames' in e.text:
            vc = verifier_crash_frame(t)
            if vc and vc[0] == 'solution':
                out.append(Hit('verifier_crash_in_solution', NOTE, None,
                               f'the crash under the verifier is in the model\'s solution ({vc[1][:200]}): the solution '
                               'failing, not the grader', e.ref, 0.9))
        if e.source == 'proxy.log' and re.search(r'error|refused|reset|timeout|5\d\d', e.text, re.I):
            out.append(Hit('proxy_log_error', SUSPICIOUS, 'network', e.text, e.ref, 0.5))
        if e.source == 'archive' and 'no Pi session' in e.text and not r.get('infrastructure_failure'):
            out.append(Hit('no_session', SUSPICIOUS, 'harness_error', e.text, e.ref, 0.6))
    task_kills = [h for h in out if h.rule == 'memcap_killed_task_process']
    if len(task_kills) > 3:   # keep the digest short: three examples and a count
        out = [h for h in out if h.rule != 'memcap_killed_task_process'] + task_kills[:3]
        out.append(Hit('memcap_killed_task_process', NOTE, None,
                       f'... {len(task_kills)} memory-cap kills of task processes in all', 'harbor.log', 0.9))
    exc = r.get('exception_type') or ''
    out_msg = next((e.text for e in t.events if e.source == 'outcome'), '')
    signal_exit = re.search(r'exit (-9|-15|137|143)\b', out_msg) is not None
    if (exc == 'NonZeroAgentExitCodeError' or signal_exit) and not agent_killed:
        # who ended the agent? the model's last tool call killing processes is a model outcome
        calls = [x for x in t.turns if x.kind == 'tool_call']
        # the in-flight call (from exception.txt) is the last one when present; look at the last two
        last = next((c for c in reversed(calls[-2:]) if SELF_KILL.search(c.text)), calls[-1] if calls else None)
        m = re.search(r'exit -?\d+', out_msg)
        code = m.group(0) if m else f'exit {r.get("returncode")}'
        if last and SELF_KILL.search(last.text):
            out.append(Hit('model_killed_own_agent', MODEL, None,
                           f'agent {code} right after the model ran: {last.text[:240]}', last.ref, 0.85))
        else:
            ev = next((e for e in t.events if e.source == 'outcome'), None)
            out.append(Hit('agent_nonzero_exit', SUSPICIOUS, 'sandbox_kill',
                           (ev.text if ev else exc)[:300], ev.ref if ev else 'outcome.json', 0.5))
    if exc == 'RewardFileNotFoundError':
        out.append(Hit('reward_file_missing', INFRA, 'grader_crash', 'the verifier wrote no reward file',
                       'outcome.json:exception_type', 0.8))
    # Pi stopped on an error the pipeline didn't classify as infra
    errs = [x for x in t.turns if x.kind == 'error']
    if errs and not r.get('infrastructure_failure'):
        out.append(Hit('pi_error_stop', SUSPICIOUS, 'network', errs[-1].text[:300], errs[-1].ref, 0.5))
    return out


PROTOCOL_MARKERS = [   # harness messages that report a protocol limit the model ran into (model outcomes)
    (r'context cap reached; episode ended', 'context_window_end',
     "Pi's context guard: the model used up the context window (protocol limit, a model outcome)"),
    (r'was not executed: the response hit the output token limit', 'output_cap_turn',
     'the per-turn output cap (protocol limit): the model wrote a turn longer than the cap'),
    (r'Agent execution timed out after', 'agent_time_limit', 'the agent time limit (protocol limit, a model outcome)'),
]


def _hits_protocol(t: Trace) -> list[Hit]:
    out, seen = [], set()
    texts = [(x.text, x.ref) for x in t.turns if x.kind in ('text', 'tool_result', 'error')] + \
            [(e.text, e.ref) for e in t.events]
    for pat, rule, why in PROTOCOL_MARKERS:
        for text, ref in texts:
            if rule not in seen and re.search(pat, text or ''):
                seen.add(rule)
                out.append(Hit(rule, NOTE, None, why, ref, 0.9))
    if t.recorded.get('context_limit') and 'context_window_end' not in seen:
        out.append(Hit('context_window_end', NOTE, None, 'the pipeline recorded context_limit: the window ran out (protocol)',
                       'outcome.json:stats', 0.9))
    return out


def _hits_turns(t: Trace) -> list[Hit]:
    out = []
    seen = set()
    for x in t.turns:
        if x.kind == 'tool_result':
            for pat, cat in ENV_ERR:
                expected = cat in ('network', 'network_dns') and offline_tool_result_denial(t, x)
                key = (pat, cat, expected)
                if key in seen:
                    continue
                for m in re.finditer(pat, x.text, re.M):
                    a = x.text.rfind('\n', 0, m.start()) + 1
                    line = x.text[a:x.text.find('\n', m.end()) if '\n' in x.text[m.end():] else len(x.text)]
                    if NOT_ERROR_LINE.search(line):
                        continue
                    seen.add(key)
                    if expected:
                        out.append(Hit('offline_tool_network_denied', NOTE, None,
                                       'Agent internet is explicitly disabled; this tool network failure is expected: '
                                       + x.text[max(0, m.start() - 120):m.end() + 120], x.ref, 1.0))
                    elif cat == 'network_dns':
                        # Offline policy does not explain broken task-local DNS or an unattributed failure.
                        out.append(Hit('unexplained_offline_dns' if offline_agent(t) else 'dns_failure_in_tool_output',
                                       SUSPICIOUS if offline_agent(t) else NOTE,
                                       'network' if offline_agent(t) else None,
                                       x.text[max(0, m.start() - 120):m.end() + 120], x.ref, 0.4 if offline_agent(t) else 0.3))
                    else:
                        out.append(Hit('env_error_in_tool_output', SUSPICIOUS, cat,
                                       x.text[max(0, m.start() - 160):m.end() + 160], x.ref, 0.4))
                    break
        if x.role == 'assistant' and x.kind == 'text' and TOOL_PARSE.search(x.text) and 'tool_parse' not in seen:
            seen.add('tool_parse')
            m = TOOL_PARSE.search(x.text)
            out.append(Hit('raw_tool_call_in_text', SUSPICIOUS, 'harness_error',
                           x.text[max(0, m.start() - 100):m.end() + 200], x.ref, 0.5))
    return out


def _hits_template(t: Trace) -> list[Hit]:
    for x in t.turns:
        if x.role == 'assistant' and x.kind == 'text':
            m = TEMPLATE_TOKENS.search(x.text or '')
            if m:
                return [Hit('template_tokens_in_output', SUSPICIOUS, 'harness_error',
                            x.text[max(0, m.start() - 100):m.end() + 150], x.ref, 0.5)]
    return []


def _hits_ending(t: Trace) -> list[Hit]:
    """An agentic attempt whose last model turn is empty (no text, no tool call): the serving or the harness
    returned nothing and the agent stopped."""
    model = [x for x in t.turns if x.role == 'assistant']
    if not model:
        return []
    last = model[-1]
    if last.kind == 'text' and not (last.text or '').strip() and last.meta.get('stop_reason') not in ('length',):
        # EOS inside the reasoning (Qwen3.8 medium): the whole message is reasoning that never closed, no answer and no
        # tool call, finish stop. The model's own outcome, even with tool-call tags drafted inside the reasoning.
        same = [x for x in model if x.ref == last.ref]
        think = ''.join(x.text or '' for x in same if x.kind == 'thinking')
        stops = t.recorded.get('stop_reasons') or t.recorded.get('finish_reasons') or []
        fr = last.meta.get('stop_reason') or last.meta.get('finish_reason') or \
            (stops[-1] if isinstance(stops, list) and len(stops) >= len({x.ref for x in model}) else None)
        if len(think.strip()) >= 200 and '</think>' not in think and not any(x.kind == 'tool_call' for x in same) \
                and fr == 'stop':
            return [Hit('ended_in_reasoning', MODEL, None, f'the last message is {len(think)} chars of reasoning that '
                        'never closed, then EOS (finish stop): no answer, no tool call', last.ref, 0.85)]
        return [Hit('empty_final_turn', SUSPICIOUS, 'silent_zero', f"last model turn is empty (stop {last.meta.get('stop_reason')})",
                    last.ref, 0.5)]
    return []


def _hits_single(t: Trace) -> list[Hit]:
    out, r = [], t.recorded
    final = [x for x in t.turns if x.kind == 'text' and x.role == 'assistant']
    final = final[-1] if final else None
    fr = (final.meta.get('finish_reason') if final else None) or r.get('finish_reason')
    ct = final.meta.get('completion_tokens') if final else None
    thinking = any(x.kind == 'thinking' and x.text.strip() for x in t.turns)
    empty = final is None or not final.text.strip()
    if empty and r.get('status') != 'infra_error':
        if fr == 'length':
            mx = r.get('max_tokens')
            at_cap = bool(ct) and ((mx and ct >= 0.8 * mx) or (not mx and ct >= 32768))
            # decisive only when the stop is at the protocol's cap; a short "length" stop may be a harness cap
            out.append(Hit('empty_at_length', MODEL if at_cap else NOTE, None,
                           f'no answer: hit max tokens ({ct} completion tokens, run max_tokens {mx})',
                           final.ref if final else 'response', 0.85 if at_cap else 0.4))
        elif thinking and fr == 'stop':
            out.append(Hit('ended_in_reasoning', MODEL, None,
                           f'the response ended inside its reasoning (finish stop, {ct} tokens)', 'response', 0.85))
        elif not ct:
            out.append(Hit('empty_no_tokens', SUSPICIOUS, 'silent_zero',
                           f'empty answer with {ct} completion tokens (finish {fr})', 'response', 0.6))
        else:
            out.append(Hit('empty_answer', SUSPICIOUS, 'silent_zero', f'empty answer (finish {fr}, {ct} tokens)', 'response', 0.5))
    mx = r.get('max_tokens')
    if fr == 'length' and mx and ct and ct < 0.8 * mx:
        out.append(Hit('length_below_cap', SUSPICIOUS, 'harness_cutoff',
                       f'finish length at {ct} tokens, below the protocol cap {mx}', 'response', 0.5))
    for e in t.events:
        if e.source == 'client':
            out.append(Hit('client_retry_or_error', SUSPICIOUS, 'network', e.text[:300], e.ref, 0.3))
    plan = r.get('plan') or {}
    if plan.get('n_images') and plan.get('covered') and plan['covered'] < plan['n_images']:
        out.append(Hit('pages_dropped', SUSPICIOUS, 'harness_cutoff',
                       f"input plan covered {plan['covered']} of {plan['n_images']} pages", 'generations.jsonl:plan', 0.5))
    return out


def _hits_agentic_generic(t: Trace) -> list[Hit]:
    out, r = [], t.recorded
    for e in t.events:
        if e.source in ('client', 'harness'):
            out.append(Hit('harness_event', SUSPICIOUS, 'network' if re.search(r'connect|timeout|5\d\d', e.text, re.I)
                           else 'harness_error', e.text[:300], e.ref, 0.4))
        elif e.source == 'native_protocol_boundary':
            out.append(Hit('native_context_boundary_review', SUSPICIOUS, 'harness_cutoff',
                           'Native typed model_error at the declared context boundary; review as a protocol/model '
                           'outcome, not evidence of a harness crash. Exact failed-request budget is unavailable. '
                           + e.text[:300], e.ref, 0.4))
    mt = r.get('max_turns')
    if r.get('stop_condition') == 'max_turns_reached' and not mt:
        pass   # the benchmark's turn cap is protocol
    if r.get('truncated_turns'):
        out.append(Hit('truncated_turns', SUSPICIOUS, 'harness_cutoff', f"{r['truncated_turns']} model turns marked truncated; inspect request budget",
                       'items.jsonl', 0.5))
    return out


LIB_FRAME = re.compile(r'/site-packages/|/dist-packages/|/usr/(local/)?lib/python|/lib/python3|<frozen ')
HARNESS_FRAME = re.compile(r'^/tests/|^/logs/|/_pytest/|/pluggy/|/reward|/verifier/')


def verifier_crash_frame(t: Trace) -> tuple[str, str] | None:
    """(where, frame) for a hard crash under the verifier: 'solution' when the first non-library frame is the model's
    code (e.g. /app/eigen.py calling a C extension the model built), 'harness' when it is the test or reward code."""
    for e in t.events:
        if e.source != 'verifier' or 'crash frames (innermost first):' not in e.text:
            continue
        frames = e.text.split('crash frames (innermost first):', 1)[1].strip().splitlines()
        for f in frames:
            m = re.match(r'File "([^"]+)"', f.strip())
            if not m or LIB_FRAME.search(m.group(1)):
                continue
            return ('harness' if HARNESS_FRAME.search(m.group(1)) else 'solution'), f.strip()
        return 'harness', frames[0].strip() if frames else ''
    return None


TEST_PATH = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)')


def pytest_failed_on_solution(t: Trace) -> str | None:
    """The verifier's pytest ran (tests collected, no collection/setup ERRORs) and every failure is the solution
    failing: an AssertionError on its output ('pytest_assertions_only'), or an ImportError/ModuleNotFoundError raised in
    the solution's own code ('solution_import_in_verifier_env'), e.g. attack.py importing numpy that the task's test
    venv doesn't ship (TB 2.1 runs pytest in its own uv venv). The grader worked either way. Failure types come from
    pytest's "path.py:LINE: Exception" locations when they cover every failure, else from the summary lines."""
    for e in t.events:
        if e.source == 'verifier' and e.text.startswith('pytest summary: '):
            try:
                ps = json.loads(e.text.splitlines()[0][len('pytest summary: '):])
            except ValueError:
                return None
            ran, nf = ps.get('passed', 0) + ps.get('failed', 0), ps.get('failed', 0)
            if not ran or ps.get('errors') or ps.get('no_tests') or not nf:
                return None
            raised = [r.rsplit(':', 1) for r in ps.get('raised_at') or []]
            if len(raised) >= nf:
                items = [(path, exc) for path, exc in raised]
            else:
                fails = ps.get('failures') or []
                if len(fails) != nf:
                    return None
                items = [('', f.split(':', 1)[1]) for f in fails]
            why = 'pytest_assertions_only'
            for path, exc in items:
                if exc in ('AssertionError', 'assert'):
                    continue
                if exc in ('ModuleNotFoundError', 'ImportError') and path and not TEST_PATH.search(path):
                    why = 'solution_import_in_verifier_env'
                    continue
                return None
            return why
    return None


PROXY_FAIL = re.compile(r'Tunnel connection failed|CONNECT tunnel failed|ProxyError|Unable to connect to proxy|'
                        r'<error>\s*5\d\d [A-Z][A-Za-z ]+</error>|'   # httpx ProxyError via web_fetch: CONNECT refused
                        r'Received HTTP code 50[234] from proxy|proxy.{0,30}\b(refused|50[234])\b', re.I)
# hosts a URL contacts; a URL nested in another's path or query (web.archive.org/web/2025/https://x) contacts only the outer
URL_HOST = re.compile(r'(?<![/=%])https?://([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]+)+)')


DNS_CACHE = os.environ.get('TRACEJUDGE_DNS_CACHE', '/data/aai/aa_evals/tracejudge/dns_cache.json')
_DNS: dict = {}


def _dns_file() -> dict:
    try:
        return json.loads(Path(DNS_CACHE).read_text())
    except (OSError, ValueError):
        return {}


def _dns_store(host: str, rec: dict):
    """Merge one verdict into the shared cache (atomic replace; a lost race only loses an entry)."""
    try:
        d = _dns_file()
        d[host] = rec
        tmp = Path(f'{DNS_CACHE}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(d, sort_keys=True))
        os.replace(tmp, DNS_CACHE)
    except OSError:
        pass


def _local_dns(host: str) -> str | None:
    """'ok' / 'nxdomain' (no such name, or no address) / 'again' (SERVFAIL, EAI_AGAIN) / None (no answer in time).
    The trailing dot skips the resolver's search domains (slow NXDOMAINs under load)."""
    import socket
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as Timeout
    ex = ThreadPoolExecutor(1)
    try:
        ex.submit(socket.getaddrinfo, host.rstrip('.') + '.', 443).result(timeout=15)
        return 'ok'
    except socket.gaierror as e:
        if e.errno in (socket.EAI_NONAME, getattr(socket, 'EAI_NODATA', -5)):
            return 'nxdomain'
        return 'again' if e.errno == socket.EAI_AGAIN else None
    except (Timeout, OSError):
        return None
    finally:
        ex.shutdown(wait=False)


def _public_dns(host: str) -> str | None:
    """An independent check: Google's DNS-over-HTTPS through the cluster proxy. 'ok' / 'nxdomain' (Status 3, or no
    A/AAAA) / 'servfail' (Status 2: the name's own servers are broken, unresolvable everywhere) / None."""
    import urllib.request
    seen = []
    for typ in ('A', 'AAAA'):
        try:
            with urllib.request.urlopen(f'https://dns.google/resolve?name={host}&type={typ}', timeout=15) as r:
                d = json.load(r)
        except (OSError, ValueError):
            return None
        st = d.get('Status')
        if st == 0 and any(a.get('type') in (1, 28, 5) for a in d.get('Answer') or []):
            return 'ok'
        seen.append({0: 'nxdomain', 3: 'nxdomain', 2: 'servfail'}.get(st))
        if st in (2, 3):
            break
    return seen[0] if seen and all(seen) else None


def resolves(host: str) -> bool | None:
    """True / False (the name doesn't resolve anywhere) / None (can't tell). Deterministic across judge runs: a
    definitive verdict is cached per process and in a shared file (TRACEJUDGE_DNS_CACHE), so a slow or broken resolver
    on a later run can't flip an earlier demotion. The local resolver decides ok/NXDOMAIN; SERVFAIL, EAI_AGAIN or no
    answer locally defer to the public DoH check, and a name that resolves publicly is True (then the local failure
    was our DNS or proxy, and the flag stays)."""
    return dns_record(host).get('verdict')


def dns_record(host: str, public: bool = False) -> dict:
    """{verdict, local, public, at} for host (see resolves); public=True makes sure the public check has run."""
    host = host.lower().rstrip('.')
    rec = _DNS.get(host) or _dns_file().get(host)
    if rec and rec.get('verdict') in (True, False):
        if public and rec.get('public') is None and rec.get('verdict') is False:
            pub = _public_dns(host)
            if pub is not None:   # a name local DNS lacks but public DNS has: ours to fix, so it resolves
                rec = {**rec, 'public': pub, 'verdict': True if pub == 'ok' else rec['verdict']}
                _dns_store(host, rec)
        _DNS[host] = rec
        return rec
    loc = _local_dns(host)
    if loc is None:
        loc = _local_dns(host)          # one retry before asking elsewhere
    pub = None
    if loc == 'ok':
        verdict = True
    elif loc == 'nxdomain':
        pub = _public_dns(host) if public else None
        verdict = pub == 'ok'
    else:
        pub = _public_dns(host)
        verdict = True if pub == 'ok' else False if pub in ('nxdomain', 'servfail') else None
    rec = {'verdict': verdict, 'local': loc, 'public': pub, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if verdict is not None:
        _DNS[host] = rec
        _dns_store(host, rec)
    return rec


def unresolvable_everywhere(host: str) -> bool:
    """No such name locally (NXDOMAIN/NODATA/EAI_AGAIN) AND in public DNS (NXDOMAIN/NODATA/SERVFAIL). The public check
    goes through the proxy, so during a proxy outage it can't answer and this is False."""
    rec = dns_record(host, public=True)
    return rec.get('local') in ('nxdomain', 'again') and rec.get('public') in ('nxdomain', 'servfail')


HOST_KW = re.compile(r"""host=['"]([A-Za-z0-9.-]+\.[A-Za-z]{2,})['"]|Failed to resolve ['"]([A-Za-z0-9.-]+)['"]|"""
                     r"""(?:Could not resolve (?:destination )?host|ENOTFOUND|CONNECT) :?([A-Za-z0-9.-]+\.[A-Za-z]{2,})""")
# the proxy or a resolver saying the name doesn't exist (Squid ERR_DNS_FAIL, NXDOMAIN, no address)
DNS_FAIL = re.compile(r'ERR_DNS_FAIL|Could not resolve (?:destination )?host|No address associated with hostname|'
                      r'Name or service not known|NXDOMAIN|getaddrinfo ENOTFOUND|NameResolutionError|'
                      r'nodename nor servname provided', re.I)
CONN_ERR = re.compile(r'Max retries exceeded|Failed to establish a new connection|'
                      r'Temporary failure in name resolution|Connection (?:refused|reset)|timed out|SSLError|'
                      r'curl: \(\d+\)|(?:^|>)\s*000\s+https?://|\b(?:HTTP|http_code|status|code)\s*[:=]?\s*000\b|(?:=>|->)\s*000\b', re.I)
WRAPPER = re.compile(r'</?(?:shell_results|stdout|stderr)>|<exit_code>\s*-?\d+\s*</exit_code>')
NOT_RUN = re.compile(r"command not found|failed to run command|No such file or directory: '?(?:curl|wget|python3?)|"
                     r"No module named '?(?:requests|urllib3|httpx|aiohttp)", re.I)
RESPONSE = re.compile(r'HTTP/\d(?:\.\d)? [1-5]\d\d|\b[1-5]\d\d\s+https?://|<content>|"status(?:_code)?"\s*:\s*[1-5]\d\d', re.I)


def _hosts(text: str) -> set:
    return set(URL_HOST.findall(text)) | {a or b or c for a, b, c in HOST_KW.findall(text)}


GENERIC_LABELS = {'www', 'api', 'app', 'apps', 'web', 'search', 'cdn', 'static', 'docs', 'data', 'blog', 'news', 'mail',
                  'srv', 'online', 'site', 'home'}


def _labels(host: str) -> list[str]:
    """The distinctive parts of a host name, for matching output labels ("FOUNDATION" -> api.foundation.app)."""
    return [p for p in host.lower().split('.')[:-1] if len(p) >= 4 and p not in GENERIC_LABELS]


def _by_label(context: str, hosts: list[str]) -> set:
    """The one host whose distinctive name parts appear as words in context, else nothing (ambiguous or none)."""
    hit = {h for h in hosts if any(re.search(r'(?<![A-Za-z0-9])' + re.escape(lab) + r'(?![A-Za-z0-9])', context, re.I)
                                   for lab in _labels(h))}
    return hit if len(hit) == 1 else set()


def _attribute(lines: list[str], n: int, hosts: list[str], reach: int = 40) -> set:
    """The host a host-less failure line belongs to: a label on the line itself ("FOUNDATION ERR ..."), else the
    nearest line above that identifies a host (a section header such as "=== https://katzdeli.com ===" or
    "### FOUNDATION API"), if it identifies exactly one."""
    hs = _by_label(lines[n], hosts)
    if hs:
        return hs
    for k in range(n - 1, max(-1, n - 1 - reach), -1):
        named = _hosts(lines[k]) & set(hosts)
        labelled = {h for h in hosts if any(re.search(r'(?<![A-Za-z0-9])' + re.escape(lab) + r'(?![A-Za-z0-9])',
                                                      lines[k], re.I) for lab in _labels(h))}
        found = named or labelled
        if found:
            return found if len(found) == 1 else set()
    return set()


def proxy_scan(t: Trace, events: list | None = None) -> tuple[dict, dict, int]:
    """(ok, failed, unattributed): host -> first turn index where an HTTP(S) call through the proxy reached it / failed
    at the proxy or at name resolution, and the number of proxy/DNS failure lines no host could be attributed to.

    Results pair with calls in order within a batch (Stirrup and OpenAI-style agents issue several calls per turn).
    A call's hosts come from the URLs in its arguments. A failure line names its host (a URL, urllib3's host='...',
    curl's "Could not resolve host: ..."); naming none, it is attributed by label (the one call host whose name parts
    appear in that line, or the nearest line above that identifies one, e.g. "### FOUNDATION API"), else to the call's
    only host, else to every host of a call that got no response at all (curl -v output), else it stays unattributed.
    A host counts as reached only when its result shows a response and no failure or error for it (a multi-host
    call's output must name it on a non-error line)."""
    ok, bad, unattributed = {}, {}, 0
    pending, prev, batch = [], None, 0
    for x in t.turns:
        if x.kind == 'tool_call':
            if prev == 'tool_result':
                pending = []          # a new batch of calls
                batch += 1
            pending.append(sorted(set(URL_HOST.findall(x.text or ''))))
        elif x.kind == 'tool_result' and pending:
            call_hosts = pending.pop(0)
            if call_hosts:
                text = x.text or ''
                lines = text.splitlines()
                failed, errored, refused, loose_fail, loose_err = set(), set(), set(), 0, False
                is_err = [bool(PROXY_FAIL.search(ln) or DNS_FAIL.search(ln) or CONN_ERR.search(ln)) for ln in lines]
                named = {h for ln, e in zip(lines, is_err) if not e for h in _hosts(ln)} & set(call_hosts)
                for n, line in enumerate(lines):
                    fail, conn = PROXY_FAIL.search(line) or DNS_FAIL.search(line), CONN_ERR.search(line)
                    if not (fail or conn):
                        continue
                    hs = _hosts(line) & set(call_hosts)
                    if not hs and len(call_hosts) > 1:
                        hs = _attribute(lines, n, call_hosts)
                    if not hs and len(call_hosts) == 1:
                        hs = set(call_hosts)
                    (failed if fail else errored).update(hs)
                    if PROXY_FAIL.search(line):
                        refused.update(hs)
                    if not hs:
                        loose_fail, loose_err = loose_fail + bool(fail), loose_err or bool(conn)
                clean = not (loose_fail or loose_err) and not re.search(r'<exit_code>\s*[1-9]', text) \
                    and not NOT_RUN.search(text)   # e.g. "timeout: failed to run command 'curl'": nothing was fetched
                payload = WRAPPER.sub('', text).strip()   # Stirrup's <shell_results> tags around an empty stdout
                responded = bool(RESPONSE.search(text)) or bool(named) or (bool(payload) and clean)
                if loose_fail and not responded:
                    failed |= set(call_hosts)     # nothing answered at all: every host of the call failed
                    refused |= set(call_hosts) if any(PROXY_FAIL.search(ln) for ln in lines) else set()
                elif loose_fail:
                    unattributed += loose_fail
                responders = set()
                for h in call_hosts:
                    if h in failed:
                        bad.setdefault(h, x.idx)
                    elif h not in errored and responded and clean and (len(call_hosts) == 1 or h in named):
                        ok.setdefault(h, x.idx)
                        responders.add(h)
                    elif h in named and h not in errored:
                        responders.add(h)         # answered in this result (another line may have failed)
                if events is not None:
                    events.append({'batch': batch, 'idx': x.idx, 'refused': refused, 'responders': responders})
        prev = x.kind if x.kind in ('tool_call', 'tool_result') else prev
    return ok, bad, unattributed


def proxy_hosts(t: Trace) -> tuple[dict, dict]:
    """(ok, failed) of proxy_scan."""
    ok, bad, _ = proxy_scan(t)
    return ok, bad


def nonexistent_hosts_only(t: Trace) -> str | None:
    """The proxy answers CONNECT to a host with no DNS record with 503: a host the model made up, not an outage
    (#18 outages hit every host). True when other hosts got through the proxy in the same attempt, more hosts
    worked than failed, every failing host never worked in the attempt, and none of them resolves now."""
    ok, bad, unattributed = proxy_scan(t)
    if not bad or unattributed or any(h in ok for h in bad):
        return None   # a proxy failure no host can be attributed to might be the outage: keep the flag
    if all(unresolvable_everywhere(h) for h in bad):
        # every failing name exists nowhere: the 503 is fully explained, even with nothing else fetched
        return (f"proxy 503 only for {', '.join(sorted(bad))}: no such name locally or in public DNS"
                + (f"; {len(ok)} other hosts worked ({', '.join(sorted(ok)[:5])})" if ok else ''))
    if len(ok) < 2 or len(ok) <= len(bad):
        return None
    if resolves(next(iter(ok))) is not True:
        return None   # DNS unusable from the judge: keep the flag
    if all(resolves(h) is False for h in bad):
        return (f"proxy 503 only for {', '.join(sorted(bad))} (no DNS record; never reached in the attempt) while "
                f"{len(ok)} other hosts worked through the proxy ({', '.join(sorted(ok)[:5])})")
    return None


def egress_blocked(t: Trace, events: list | None = None) -> tuple[dict, set]:
    """(blocked, unexplained). blocked: host -> number of proxy CONNECT refusals (proxy 503) to a host that exists, each
    in a call where OTHER hosts worked through the proxy at the same time (the same tool result, or the same batch of
    parallel calls) and were at least as many as the refused ones: the proxy was up and that upstream was unreachable or blocking
    our egress (DuckDuckGo blocks cluster IPs). unexplained: hosts with any refusal lacking that evidence."""
    if events is None:
        events = []
        proxy_scan(t, events)
    by_batch = {}
    for e in events:
        b = by_batch.setdefault(e['batch'], {'refused': set(), 'responders': set()})
        b['refused'] |= e['refused']
        b['responders'] |= e['responders']
    blocked, unexplained = {}, set()
    for e in events:
        for h in e['refused']:
            if unresolvable_everywhere(h):
                continue          # a made-up host: nonexistent_hosts_only's case
            b = by_batch[e['batch']]
            others = (e['responders'] | b['responders']) - {h}
            refused = e['refused'] | b['refused']
            if others and len(others) >= len(refused):   # refusals to MOST hosts of the call stay flagged
                blocked[h] = blocked.get(h, 0) + 1
            else:
                unexplained.add(h)
    return {h: n for h, n in blocked.items() if h not in unexplained}, unexplained


def proxy_failures_explained(t: Trace) -> str | None:
    """Every proxy failure in the attempt is explained by a host that exists nowhere or by a host-specific egress block
    (see egress_blocked), with at least one egress block; no unattributed failures."""
    events = []
    ok, bad, unattributed = proxy_scan(t, events)
    if not bad or unattributed:
        return None
    blocked, unexplained = egress_blocked(t, events)
    if not blocked or unexplained:
        return None
    for h in bad:
        if h not in blocked and not unresolvable_everywhere(h):
            return None           # e.g. a DNS failure for a name that resolves publicly: ours
    return f"proxy refused CONNECT only to {', '.join(sorted(blocked))} while other hosts worked in the same calls"


REMOTE_STATUS = re.compile(r'HTTP Error [45]\d\d|HTTP/\d(?:\.\d)? [45]\d\d\b|\b(?:429 Too Many Requests|403 Forbidden)\b|'
                           r'"status(?:_code)?"\s*:\s*[45]\d\d\b')


REMOTE_THROTTLE = re.compile(r'\b(?:429|403)\b.{0,30}(?:Too Many Requests|Forbidden|rate limit)|'
                             r'(?:Too Many Requests|Forbidden).{0,10}\b(?:429|403)\b|HTTP Error (?:429|403)', re.I)


def proxy_503_amid_site_limits(t: Trace) -> str | None:
    """Proxy failures explained by the remote site refusing a bulk scrape, with the tunnel itself working: each
    proxy-failure line names a host that also answered 429/403 in the attempt (the site throttling, then refusing
    connections), or, with no URL on the line, sits in a result with >= 3x as many remote 4xx/5xx answers. Needs >= 2
    hosts outside the failing ones to have worked through the proxy. A real outage (#18) fails every host."""
    ok, bad = proxy_hosts(t)
    throttled = set()
    for x in t.turns:
        if x.kind == 'tool_result':
            for ln in (x.text or '').splitlines():
                if REMOTE_THROTTLE.search(ln) and not PROXY_FAIL.search(ln):
                    throttled |= set(URL_HOST.findall(ln))
    n_fail = 0
    for x in t.turns:
        if x.kind != 'tool_result':
            continue
        lines = (x.text or '').splitlines()
        loose = 0
        for ln in lines:
            if not PROXY_FAIL.search(ln):
                continue
            n_fail += 1
            hs = set(URL_HOST.findall(ln))
            if hs and not hs & throttled:
                return None          # a failing host the site never throttled: could be the proxy
            loose += not hs
        if loose and sum(1 for ln in lines if REMOTE_STATUS.search(ln) and not PROXY_FAIL.search(ln)) < 3 * loose:
            return None
    if n_fail and len(set(ok) - set(bad)) >= 2:
        return (f'{n_fail} proxy failure line(s), all to sites that were answering 429/403 '
                f"({', '.join(sorted(throttled)[:3]) or 'same batch'}); {len(set(ok) - set(bad))} other hosts worked")
    return None


def retried_and_recovered(t: Trace) -> bool:
    """A single-turn request whose client retried (gateway 5xx, the serving admission gate's "server at capacity" 503)
    and then got a complete answer: the retries cost time, not the outcome."""
    if t.kind == 'agentic' or not any(e.source == 'client' for e in t.events):
        return False
    r = t.recorded
    final = [x for x in t.turns if x.kind == 'text' and x.role == 'assistant']
    fr = (final[-1].meta.get('finish_reason') if final else None) or r.get('finish_reason')
    return (r.get('status') in ('ok', None) and bool(final) and bool((final[-1].text or '').strip()) and fr == 'stop'
            and not TEMPLATE_TOKENS.search(final[-1].text or ''))


# the remote server itself answered (the tunnel worked): httpx raise_for_status, urllib/requests HTTPError, curl -v
# after "200 Connection established", or an anti-bot challenge page
REMOTE_ANSWER = re.compile(r"(?:Server|Client) error '[45]\d\d [^']*' for url|HTTP Error [45]\d\d\b|<HTTPError [45]\d\d\b|"
                           r"\b[45]\d\d (?:Client|Server) Error\b|"
                           r"bots use DuckDuckGo|anomaly-modal|unusual traffic from your computer network|"
                           r"Our systems have detected unusual traffic|cf-chl-|Attention Required! \| Cloudflare|"
                           r"captcha-delivery|Just a moment\.\.\.", re.I)
REMOTE_LIMIT = re.compile(r"(?:Server|Client) error '(?:429|403|503)|HTTP Error (?:429|403|503)\b|<HTTPError (?:429|403|503)\b|"
                          r"\b(?:429|403|503) (?:Client|Server) Error\b|bots use DuckDuckGo|anomaly-modal|"
                          r"unusual traffic from your computer network|Our systems have detected unusual traffic|cf-chl-|"
                          r"Attention Required! \| Cloudflare|captcha-delivery", re.I)


def remote_limits(t: Trace) -> dict:
    """host -> number of tool results where the remote site itself refused (429 / 403 / 503 answered by the site, or an
    anti-bot challenge page): search scraping being throttled, for the run-level n_remote_site_limited stat."""
    out, pending, prev = {}, [], None
    for x in t.turns:
        if x.kind == 'tool_call':
            if prev == 'tool_result':
                pending = []
            pending.append(sorted(set(URL_HOST.findall(x.text or ''))))
        elif x.kind == 'tool_result' and pending:
            hosts = pending.pop(0)
            lines = (x.text or '').splitlines()
            seen = set()
            for n, ln in enumerate(lines):
                if REMOTE_LIMIT.search(ln) and not PROXY_FAIL.search(ln):
                    hs = (_hosts(ln) & set(hosts)) or (set(hosts) if len(hosts) == 1 else _attribute(lines, n, hosts))
                    seen |= hs or {'?'}
            for h in seen:
                out[h] = out.get(h, 0) + 1
        prev = x.kind if x.kind in ('tool_call', 'tool_result') else prev
    return out


def remote_site_answer(t: Trace, flag: dict) -> bool:
    """A network flag whose evidence is the remote server's own answer (an HTTP status error or a challenge page, with
    the tunnel up): the site's outcome (#217 class), not network infra, provided every proxy failure in the attempt is
    explained (no unattributed ones; failing hosts don't exist anywhere)."""
    ev = str(flag.get('evidence') or '')
    if not REMOTE_ANSWER.search(ev) or PROXY_FAIL.search(ev) or DNS_FAIL.search(ev):
        return False
    events = []
    _, bad, unattributed = proxy_scan(t, events)
    if unattributed:
        return False
    blocked, unexplained = egress_blocked(t, events) if bad else ({}, set())
    return not unexplained and all(h in blocked or unresolvable_everywhere(h) for h in bad)


EMPTY_DIR = re.compile(r'\btotal 0\b|No such file or directory|cannot access|is empty|\bempty\b', re.I)


def model_touched_dir(t: Trace, flag: dict) -> bool:
    """A sandbox_kill flag on an empty or missing directory that an earlier agent command removed or used as its
    output/target (rm/mv of it, rmtree, --output-dir <it>): the model cleared it (specimen-provenance: its script
    empties --output-dir /app). Not when the rules saw a real kill (memcap kill of the agent, OOM, a pipeline infra
    label)."""
    ev = f"{flag.get('evidence') or ''} {flag.get('ref') or ''}"
    if flag.get('category') != 'sandbox_kill' or not EMPTY_DIR.search(ev):
        return False
    if any(h.polarity == INFRA or h.rule in ('kernel_oom', 'memcap_killed_agent') for h in run_rules(t)):
        return False
    m = re.search(r'turn (\d+)', str(flag.get('ref') or ''))
    upto = int(m.group(1)) if m else None
    dirs = {d.rstrip('/') for d in re.findall(r'(?<![\w.])(/(?:app|workspace|work|home/\w+|root|data|srv|opt/\w+|output|tmp/[\w.-]+)'
                                             r'(?:/[\w.-]+)*)/?(?=[\s:\'",)]|$)', ev)}
    for d in dirs:
        q = re.escape(d)
        hit = re.compile(rf'\brm\s+(?:-\w+\s+)*{q}(?:/\*?)?(?=[\s;&|]|$)|\bmv\s+(?:-\w+\s+)*{q}/?(?=[\s;&|])|'
                         rf'rmtree\(\s*[\'"]{q}/?[\'"]|'
                         rf'(?:--?(?:output|out|dest|target)(?:[-_]?dir)?|-o)[ =]{q}/?(?=[\s;&|]|$)')
        for x in t.turns:
            if x.kind == 'tool_call' and hit.search(x.text or ''):
                if upto is None or _turn_no(x.ref) is None or _turn_no(x.ref) < upto:
                    return True
    return False


def _turn_no(ref: str) -> int | None:
    m = re.search(r'turn (\d+)', ref or '')
    return int(m.group(1)) if m else None


def demotion(t: Trace, flag: dict) -> str | None:
    """Why an LLM flag is a model/benchmark outcome after all (kept in flags.jsonl below the threshold), or None."""
    if tool_result_only(t, flag):
        return 'tool_result_only'
    if offline_tool_denial(t, flag):
        return 'offline_agent_network_denied'
    if model_touched_dir(t, flag):
        return 'model_touched_dir'
    if flag.get('category') == 'network' and retried_and_recovered(t):
        return 'retried_and_recovered'
    if flag.get('category') == 'network' and (PROXY_FAIL.search(str(flag.get('evidence') or '')) or '503' in
                                              str(flag.get('evidence') or '') or DNS_FAIL.search(str(flag.get('evidence') or ''))):
        if nonexistent_hosts_only(t):
            return 'proxy_503_nonexistent_host'
        if proxy_503_amid_site_limits(t):
            return 'proxy_503_amid_site_limits'
        if proxy_failures_explained(t):
            return 'egress_blocked'
    if flag.get('category') == 'network' and remote_site_answer(t, flag):
        return 'remote_site_answer'
    if flag.get('category') in ('grader_crash', 'harness_error') and \
            'verifier' in str(flag.get('ref') or '') + str(flag.get('evidence') or ''):
        vc = verifier_crash_frame(t)
        if vc and vc[0] == 'solution':
            return 'crash_in_solution'
        why = None if vc else pytest_failed_on_solution(t)
        if why and not any(e.source == 'exception.txt' and re.search(r'RewardFileNotFound', e.text) for e in t.events):
            return why   # the tests ran and the solution failed them; log noise is not a grader crash
    return None


def offline_agent(t: Trace) -> bool:
    policy = t.recorded.get('agent_network')
    return t.recorded.get('agent_internet') is False or (isinstance(policy, dict) and policy.get('mode') == 'offline')


OFFLINE_DENIAL = re.compile(r'Network is unreachable|No route to host|ENETUNREACH|EHOSTUNREACH|'
                           r'Temporary failure in name resolution|Could not resolve host|NameResolutionError|'
                           r'Name or service not known|getaddrinfo ENOTFOUND|Unable to connect to proxy', re.I)
MODEL_API = re.compile(r'APIConnectionError|model[ _-]?(?:endpoint|relay)|'
                       r'/v1/(?:chat/completions|completions|models|responses)|/tokenize', re.I)
HARNESS_NETWORK = re.compile(r'APIConnectionError|ProxyError|connect|refused|reset|timeout|timed out|5\d\d', re.I)


def _public_tool_targets(text: str) -> bool:
    """Require explicit external targets; local task services and ambiguous commands remain reviewable."""
    urls = re.findall(r'https?://[^\s\'"<>]+', text)
    hosts = []
    for url in urls:
        try:
            hosts.append(urlsplit(url.rstrip('),;')).hostname or '')
        except ValueError:
            return False
    hosts += re.findall(r"host=['\"]([^'\"]+)['\"]|Could not resolve host:\s*([^\s]+)", text, re.I)
    hosts = [next((v for v in h if v), '') if isinstance(h, tuple) else h for h in hosts]
    if not hosts:
        return False
    for host in hosts:
        host = host.lower().rstrip('.')
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global or address.is_multicast:
                return False
        except ValueError:
            if '.' not in host or host.endswith(('.local', '.localhost', '.internal', '.internal', '.test', '.invalid')):
                return False
    return True


def offline_tool_result_denial(t: Trace, result) -> bool:
    """Only attribute an explicit agent-side external egress failure to the offline protocol."""
    if not offline_agent(t) or result.kind != 'tool_result' or not OFFLINE_DENIAL.search(result.text):
        return False
    calls, previous = [], None
    for turn in t.turns:
        if turn.kind == 'tool_call':
            if previous == 'tool_result':
                calls = []
            calls.append(turn.text)
        elif turn is result:
            # A mixed parallel batch must not hide a local-service failure through an external call's context.
            context = '\n'.join(calls)
            if MODEL_API.search(context + '\n' + result.text):
                return False
            named_result = re.search(r'https?://|host=[\"\']|Could not resolve host:', result.text, re.I)
            if named_result and not _public_tool_targets(result.text):
                return False
            if re.search(r'\blocalhost\b|\b127\.\d|::1|\b0\.0\.0\.0\b', result.text, re.I):
                return False
            return all(_public_tool_targets(call) for call in calls) if calls else _public_tool_targets(result.text)
        if turn.kind in ('tool_call', 'tool_result'):
            previous = turn.kind
    return False


def offline_tool_denial(t: Trace, flag: dict) -> bool:
    """Demote cited, explained egress denials; independent harness/API or local failures stay reviewable."""
    if not offline_agent(t) or flag.get('category') != 'network' or t.known_infra:
        return False
    cited, evidence = str(flag.get('ref') or ''), str(flag.get('evidence') or '')
    if any(e.ref and e.ref in cited for e in t.events) or re.search(
            r'PIPELINE|HARNESS|outcome|exception|proxy\.log', cited, re.I):
        return False
    if any(x.kind == 'error' for x in t.turns) or any(HARNESS_NETWORK.search(e.text) for e in t.events):
        return False
    matched = [x for x in t.turns if x.ref and re.search(r'(?<![\w:])' + re.escape(x.ref) + r'(?!\d)', cited)]
    if not matched or any(not offline_tool_result_denial(t, x) for x in matched):
        return False
    remainder = cited
    for turn in matched:
        remainder = re.sub(r'(?<![\w:])' + re.escape(turn.ref) + r'(?!\d)', '', remainder)
    if re.search(r'\bturn\s+\d+|\bmsg:\d+|events\[|history\[', remainder):
        return False
    return bool(evidence) and any(evidence in x.text for x in matched)


def tool_result_only(t: Trace, flag: dict) -> bool:
    """An LLM harness_error flag whose cited refs are all tool results: an exception string returned AS a tool result is
    the benchmark's tool / simulated API answering the model's call (e.g. AutomationBench's simulated Gmail doing
    `to.split(',')` on a list the model passed), so benchmark output, not our harness. Harness errors live in the
    pipeline record, harness events or the model's own text (raw tool-call markup, template tokens)."""
    if flag.get('category') != 'harness_error':
        return False
    cited = str(flag.get('ref') or '')
    if any(e.ref and e.ref in cited for e in t.events) or re.search(r'PIPELINE|HARNESS|PRE-?FILTER|RECORD|outcome|exception', cited, re.I):
        return False
    kinds = {}
    for x in t.turns:
        if x.ref and re.search(r'(?<![\w:])' + re.escape(x.ref) + r'(?!\d)', cited):
            kinds.setdefault(x.ref, set()).add(x.kind)
    if not kinds or any(k != {'tool_result'} for k in kinds.values()):
        return False
    ev = str(flag.get('evidence') or '')
    # machine errors in tool output (disk full, fork failure, proxy) point at the sandbox, even if miscategorized
    return not (TOOL_PARSE.search(ev) or TEMPLATE_TOKENS.search(ev) or any(re.search(p, ev) for p, _ in ENV_ERR))


PROTOCOL_ERROR_KINDS = ('input_too_long',)   # the protocol, not infra: the input doesn't fit the context the protocol allows


def _protocol_limit(t: Trace) -> Hit | None:
    """The pipeline's own protocol-limit zero (any benchmark), e.g. GDP.pdf's input_too_long: a document too long to fit
    the 128k cap with the 32k output floor. Decisive model outcome unless the pipeline itself marked the attempt infra."""
    r = t.recorded
    if r.get('status') == 'infra_error' or t.known_infra:
        return None
    plan = r.get('plan') if isinstance(r.get('plan'), dict) else {}
    kind = next((k for k in (r.get('error_kind'), r.get('model_error_kind'), plan.get('reason')) if k in PROTOCOL_ERROR_KINDS), None)
    if not kind:
        return _cap_reached(t)
    return Hit(f'protocol_{kind}', MODEL, None, f"pipeline recorded error_kind={kind}: {str(r.get('error') or '')[:240]}",
               'record', 0.95)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _cap_reached(t: Trace) -> Hit | None:
    """A context_overflow / output_limit error where the model's own tokens reach the protocol cap (>= 95% of the
    context window, or of the per-response output cap): the model filled it by itself (e.g. CritPt: 128k of reasoning
    on a 2k prompt, then Stirrup's stock "Context overflow reached the original prompt"). Without counts, no verdict."""
    r = t.recorded
    # Agentic usage can include many turns, retries and summaries. Even totals far above the window
    # do not prove that any individual request reached it, especially when the failed response is absent.
    if t.kind != 'single' and r.get('token_usage_scope') != 'single_response':
        return None
    err = str(r.get('error') or '')
    m = re.match(r'(context_overflow|output_limit)\b', err) or re.match(r'(context_overflow|output_limit)$', str(r.get('error_kind') or ''))
    if not m:
        return None
    inp, out = _num(r.get('input_tokens')) or 0, _num(r.get('output_tokens'))
    ctx, mo = _num(r.get('context_window')), _num(r.get('max_output_tokens'))
    if not out:
        return None
    if ctx and inp + out >= 0.95 * ctx:
        why = f'{inp + out:.0f} of the {ctx:.0f}-token context window ({inp:.0f} in, {out:.0f} out)'
    elif mo and out >= 0.95 * mo:
        why = f'{out:.0f} output tokens of the {mo:.0f}-token output cap'
    else:
        return None
    return Hit(f'protocol_{m.group(1)}', MODEL, None, f'{m.group(1)} with the model\'s own tokens at the cap: {why}',
               'record', 0.95)


def run_rules(t: Trace) -> list[Hit]:
    hits = _hits_known(t)
    if t.recorded.get('grading_replay_review_pending') is True:
        hits.append(Hit('grading_replay_unresolved', INFRA, 'grader_crash',
                        'Grading replay evidence failed validation; current correction remains pending',
                        'grading-replay-review.json', 1.0))
    if t.kind == 'agentic':
        hits += (_hits_pi(t) if t.source and '/attempts/' in t.source else _hits_agentic_generic(t))
        hits += _hits_turns(t) + _hits_ending(t) + _hits_protocol(t)
    else:
        hits += _hits_single(t)
    hits += _hits_template(t)
    lim = _protocol_limit(t)
    if lim:   # the empty answer and the client error are that limit's consequences: context for a reader, not suspicion
        hits = [Hit(h.rule, NOTE, h.category, h.evidence, h.ref, h.confidence) if h.polarity == SUSPICIOUS else h
                for h in hits] + [lim]
    return hits


def decide(hits: list[Hit]) -> str:
    """'infra' (flag, no LLM needed), 'model' (no flag, no LLM), 'suspicious' (ask the LLM) or 'clean'."""
    if any(h.polarity == INFRA for h in hits):
        return INFRA
    if any(h.polarity == MODEL and h.confidence >= 0.8 for h in hits) and not any(
            h.polarity == SUSPICIOUS and h.confidence >= 0.5 and h.rule != 'agent_nonzero_exit' for h in hits):
        return MODEL
    if any(h.polarity == SUSPICIOUS for h in hits):
        return SUSPICIOUS
    return 'clean'

"""Exact rendered-context guard for Pi: submit normally instead of compacting.

Runs after model-specific request adapters, so /tokenize sees the same messages,
reasoning and template options as generation. Synthetic terminal turns are saved
but are not teacher output. The guard never edits or summarizes conversation history.

One guard for every pipeline (#480: aa-evals' and task-synthesis' copies merged). Optional, all off by default:
  --backend-file PATH   (start_context_guard(backend_file=...)) re-read the owned model
                        endpoint once per request, so a replaced serving host does not strand
                        live sessions; transitions go to backend-transitions.jsonl.
  --output-cap N        (start_context_guard(output_cap=...)) authoritative per-turn cap:
                        replaces Pi's conservative max_tokens estimate with
                        min(N, 16384, remaining context); changes go to context-headroom.jsonl.
                        An exhausted Pi budget (<= 1 token) still submits.
  PI_CONTEXT_RENDERER   model-specific exact renderer for /tokenize when the server's chat
                        parser cannot render the history: `deepseek_v41` (vLLM's /tokenize
                        crashes on DeepSeek V4.1 tool histories) or any module exposing
                        tokenize_payload(payload).
  PI_CONTEXT_TOKENIZER  path to the served model's tokenizer.json: count a renderer's raw prompt
                        in-process instead of a /tokenize round trip through the busy API servers
                        (DeepSeek-1017 live, 2026-10-25: 3.5 s per request, 13% of request time;
                        local counts matched /tokenize on 426 of 426 live requests at ~50 ms).
  PI_CONTEXT_LEASE      module exposing an async context manager reserve(session, events,
                        prompt_tokens, output_budget) held around each forwarded request.
  PI_CONTEXT_TIMINGS=1  per-request renderer/admission/backend timings in request-timings.jsonl
                        (default off; synth/worker.py turns it on with a renderer, task-synthesis' default).
  PI_CONTEXT_STATE      (start_context_guard(state_file=...)) small JSON file rewritten around every
                        request: the agent's current context, whether a request is in flight, and
                        cumulative prompt-cache reuse (vLLM usage.prompt_tokens_details.cached_tokens
                        against the previous turn's context). Synth workers report it to the
                        controller, whose admission counts the KV cache every live agent holds.
  PI_GUARD_BOUND_SHORTCUT=1  skip /tokenize when the previous usage + the new messages' bytes leave a
                        full turn of room (eval_q35; ignored under PI_RESPONSE_REPLAY, and with --output-cap,
                        --state or PI_CONTEXT_LEASE, which need the exact count).
  PI_HTTP_FORCE_CLOSE=1 a new backend connection per request (proxy and guard; aa-evals 915ca4e4).
  PI_FINISH_WARN_AT, PI_FINISH_RESERVE, PI_FINISH_STICKY=1
                        the Pi finish notice (#375, #382, #395): v3 with the first two, v4 sticky
                        notices with all three; the notice map goes to finish-notices.jsonl. Each
                        pipeline sets them from its run's frozen protocol.finish_notice
                        (eval_q35/worker.py, aa_evals pi_eval, synth/worker.py); none = v2.
A request that asks for max_tokens <= 1 (Pi's own estimate is exhausted) is submitted before any /tokenize
(v2-no-spin).
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

CAP_MARK = '[harbor-gen context cap reached; episode ended by the generation harness]'
POLICY = 'submit-at-context-limit-v2-no-spin'

# Opt-in finish notice (#375, user 2026-11-13: budget-aware continuation + a reserved final turn; earlier reasoning
# stays in the history). The guard knows the exact rendered prompt size, so near the window it (a) appends a short
# harness notice with the tokens left to the request it forwards, (b) clamps a turn that would eat into the last
# RESERVE tokens, so (c) one final turn of about RESERVE tokens follows with a "final turn" notice. Nothing of the
# conversation is edited or dropped; the notice is appended to the outgoing request only (Pi's own history keeps the
# untouched messages; every later request near the window gets a fresh notice). Off unless both values are set.
FINISH_POLICY = 'submit-at-context-limit-v3-finish-notice'
# v4 (#375 sftnotice, user 2026-11-13: SFT traces with the notice, "to avoid train/test gaps"): notices are sticky. A
# notice joined to the message at index i is joined again, with the same text, to that message in every later request
# that still has it (same index, role and normalized text), so the requests of a session are append-only extensions of
# each other and the final one, with all its notices, is the one linear trace the model saw. Each new notice records
# (index, role, base_sha256, text) in finish-notices.jsonl; SFT exporters ship those lines and the SFT converter joins
# them to the session's messages (`join_notice`, `notice_base_sha256`). v3 runs keep v3 (PI_FINISH_STICKY unset).
STICKY_POLICY = 'submit-at-context-limit-v4-sticky-notice'
NOTICE_SEP = '\n\n'   # between a str content and its notice (a list content gets a separate text part)
FINISH_WARN = ('[Harness notice] About {remaining:,} tokens of context remain in this session. When they run out, '
               'the session ends and your work is graded as it stands. Start wrapping up: finish your current line '
               'of reasoning and deliver your final answer or deliverables in the form and location the task asks '
               'for.')
FINISH_FINAL = ('[Harness notice] The context is almost full: this is your final turn (about {budget:,} tokens). '
                'Deliver your final answer or deliverables now, in the form and location the task asks for.')
NOTICE_MARGIN = 256   # tokens an appended notice may add to the rendered prompt (it is ~70)
FINAL_SLACK = 2048    # below reserve + this, the next turn is the final one
TURN = 16384


def finish_plan(remaining, requested, warn_at, reserve, turn=TURN):
    """(output budget, notice kind or None) for one request from the exact remaining context (window - prompt).

    final: remaining <= reserve + max(FINAL_SLACK, turn // 2) -> this is the final turn, with everything left;
    clamp: a full turn would eat into the reserve -> it stops at remaining - reserve (with the warning), so the
           next request starts with about `reserve` tokens for the final turn. A clamped turn is never shorter
           than turn // 2 (#382 review S1: with templates that drop reasoning before the last user turn, a cut
           turn adds almost nothing, and short clamps would repeat many times);
    warn:  remaining <= warn_at -> unchanged budget plus the warning.
    The notice margin is held back whenever a notice is appended. Without a notice the budget is the v2 one."""
    plain = min(requested, turn, remaining)
    avail = remaining - NOTICE_MARGIN
    if reserve and remaining <= reserve + max(FINAL_SLACK, turn // 2):
        return min(requested, turn, avail), 'final'
    if reserve and remaining - turn < reserve:
        return min(requested, turn, remaining - reserve, avail), 'warn'
    if warn_at and remaining <= warn_at:
        return min(requested, turn, avail), 'warn'
    return plain, None


def notice_text(kind, remaining, budget):
    return (FINISH_FINAL if kind == 'final' else FINISH_WARN).format(remaining=max(remaining, 0), budget=max(budget, 0))


def notice_base_text(content):
    """A message content's text as the notice map identifies it (#375 sftnotice): a str as is; a list = its text parts
    joined with NOTICE_SEP (non-text parts, e.g. images, are not part of it); None = ''. The guard and the SFT
    splice (sft_qwen35/prepare.py, a verbatim copy) must agree on this byte for byte."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return NOTICE_SEP.join(p.get('text') or '' for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return json.dumps(content, sort_keys=True, ensure_ascii=False)   # not seen from Pi; never carries a notice


def notice_base_sha256(content):
    import hashlib
    return hashlib.sha256(notice_base_text(content).encode('utf-8', 'surrogatepass')).hexdigest()


def join_notice(message, text):
    """A new message with the notice joined as `with_notice` joins it (the input is not modified): a str content gets
    NOTICE_SEP + text, a list content one more text part. None if the message cannot carry it (not a user or tool
    message with str or list content)."""
    if not isinstance(message, dict) or message.get('role') not in ('user', 'tool'):
        return None
    if isinstance(message.get('content'), str):
        return {**message, 'content': message['content'] + NOTICE_SEP + text}
    if isinstance(message.get('content'), list):
        return {**message, 'content': [*message['content'], {'type': 'text', 'text': text}]}
    return None


def with_notice(messages, text):
    """A new message list with the notice joined to the trailing message: a user message (the task, or Pi's
    'please continue') or a tool result. It never adds a message: templates that drop reasoning before the last
    user turn (Gemma, Qwen without preserve_thinking) must render the history exactly as without the notice.
    A new user message only if the last message is neither (not seen from Pi). The input list and its dicts are
    not modified (the bound shortcut compares them)."""
    msgs = list(messages or [])
    joined = join_notice(msgs[-1], text) if msgs else None
    if joined is not None:
        msgs[-1] = joined
    else:
        msgs.append({'role': 'user', 'content': text})
    return msgs


def notice_entry(messages, text):
    """The notice map entry for `text` joined by `with_notice(messages, text)`: index (in the forwarded request, system
    prompt included), role, base_sha256 (normalized text of that message before the notice), text, appended (True
    when with_notice had to add a user message; such a notice is not sticky and an SFT trace with it is dropped)."""
    msgs = list(messages or [])
    if msgs and join_notice(msgs[-1], text) is not None:
        return {'index': len(msgs) - 1, 'role': msgs[-1]['role'],
                'base_sha256': notice_base_sha256(msgs[-1].get('content')), 'text': text, 'appended': False}
    return {'index': len(msgs), 'role': 'user', 'base_sha256': None, 'text': text, 'appended': True}


def sticky_key(entry):
    return entry['index'], entry['role'], entry['base_sha256']


def apply_sticky(messages, sticky, skip=None):
    """(new list, applied indices): every sticky notice {sticky_key: entry} joined again to its message when the
    request still has that message (same index, role and base_sha256); `skip` = an index left alone (the trailing
    message, whose notice is decided afresh). Keyed by content as well as index: a request of another conversation
    through the same guard (e.g. an agent querying the endpoint from its sandbox) neither gets nor clears them.
    The input list and its dicts are not modified."""
    msgs, applied = list(messages or []), []
    for key in sorted(sticky, key=lambda k: k[0]):
        i, role, sha = key
        if i == skip or i >= len(msgs) or i in applied:
            continue
        m = msgs[i]
        if not isinstance(m, dict) or m.get('role') != role or notice_base_sha256(m.get('content')) != sha:
            continue
        joined = join_notice(m, sticky[key]['text'])
        if joined is not None:
            msgs[i] = joined
            applied.append(i)
    return msgs, applied


def finish_settings(env=None):
    """(warn_at, reserve) from PI_FINISH_WARN_AT / PI_FINISH_RESERVE (eval_q35/worker.py, from the run's frozen
    protocol.finish_notice); (0, 0) = off."""
    env = os.environ if env is None else env
    warn, reserve = int(env.get('PI_FINISH_WARN_AT') or 0), int(env.get('PI_FINISH_RESERVE') or 0)
    if warn < 0 or reserve < 0 or (reserve and reserve < 4096):
        raise ValueError(f'bad finish notice settings: warn_at={warn} reserve={reserve}')
    return warn, reserve


def finish_sticky(env=None):
    """PI_FINISH_STICKY=1 (eval_q35/worker.py, when the run's protocol.finish_notice has sticky: true): v4."""
    env = os.environ if env is None else env
    return env.get('PI_FINISH_STICKY') == '1'


def finish_policy(warn_at, reserve, sticky):
    return (STICKY_POLICY if sticky else FINISH_POLICY) if (warn_at or reserve) else POLICY


def finish_notice_record(warn_at, reserve):
    """protocol.finish_notice of a new v4 run, the same record aa-evals' pi_eval run.finish_notice freezes (#395), so
    the eval side can tell the policy from the data: thresholds, policy, the notice texts' sha256 and the final-turn
    point (task-synthesis #428; synth/config.py)."""
    import hashlib
    finish_settings({'PI_FINISH_WARN_AT': str(warn_at), 'PI_FINISH_RESERVE': str(reserve)})   # validates
    texts = FINISH_WARN + '\n' + FINISH_FINAL
    return {'warn_at': int(warn_at), 'reserve': int(reserve), 'policy': STICKY_POLICY, 'sticky': True,
            'notice_margin': NOTICE_MARGIN, 'final_slack': FINAL_SLACK,
            'text_sha256': hashlib.sha256(texts.encode()).hexdigest(),
            'final_at': int(reserve) + max(FINAL_SLACK, TURN // 2)}


USAGE = re.compile(rb'"usage"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})')


def response_usage(tail):
    """The last OpenAI usage object in a (streamed) response tail, or None."""
    for match in reversed(list(USAGE.finditer(tail))):
        try:
            usage = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(usage, dict) and isinstance(usage.get('prompt_tokens'), int):
            return usage
    return None


class ContextState:
    """One agent's KV footprint and prompt-cache reuse, for the campaign's admission.

    context_tokens: the KV the agent holds between turns (last prompt + completion); while a
    request is in flight, its prompt. expected/cached_tokens: per turn, the part of the prompt that
    repeats the previous turn's prompt, and the part vLLM reported as cached; their ratio falls when
    live agents' prefixes are evicted. (The previous completion is not expected: the chat template
    re-renders it differently from the generated tokens, so even an unevicted cache never matches it;
    Nemotron Super smoke 2026-10-21: cached = previous prompt rounded down to the 16-token unit.)"""

    def __init__(self, path):
        self.path = Path(path)
        self.record = {'context_tokens': 0, 'inflight': False, 'turns': 0, 'turns_with_cache_details': 0,
                       'expected_cached_tokens': 0, 'cached_tokens': 0, 'started_at': time.time()}
        self.previous_prompt = None

    def write(self):
        self.record['time'] = time.time()
        tmp = self.path.with_name(f'.{self.path.name}.tmp')
        try:
            tmp.write_text(json.dumps(self.record))
            os.replace(tmp, self.path)
        except OSError:
            pass  # telemetry only

    def request(self, prompt_tokens):
        self.record.update(inflight=True, inflight_prompt_tokens=prompt_tokens,
                           context_tokens=max(self.record['context_tokens'], prompt_tokens))
        self.write()

    def response(self, prompt_tokens, usage):
        r = self.record
        completion = (usage or {}).get('completion_tokens') or 0
        cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
        r['turns'] += 1
        if cached is not None and self.previous_prompt:
            r['turns_with_cache_details'] += 1
            r['expected_cached_tokens'] += min(prompt_tokens, self.previous_prompt)
            r['cached_tokens'] += min(int(cached), prompt_tokens, self.previous_prompt)
        self.previous_prompt = prompt_tokens
        r.update(inflight=False, context_tokens=prompt_tokens + completion, last_prompt_tokens=prompt_tokens,
                 last_completion_tokens=completion, last_cached_tokens=cached)
        r.pop('inflight_prompt_tokens', None)
        self.write()


def context_window(env=None):
    env = os.environ if env is None else env
    values = {int(env[k]) for k in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN')
              if env.get(k) and int(env[k]) > 0}
    if len(values) > 1:
        raise ValueError(f'Conflicting Pi context windows: {sorted(values)}')
    window = next(iter(values), 132000)
    if window < 1:
        raise ValueError('Pi context window must be positive')
    return window


def settings(idle_timeout=1800000):
    return {'httpIdleTimeoutMs': int(idle_timeout), 'compaction': {'enabled': False}}


def terminal_response(model, count, stream, marker=CAP_MARK):
    base = {'id': f'chatcmpl-context-limit-{time.time_ns()}', 'created': int(time.time()), 'model': model}
    # No model inference happened. Exact prompt length is recorded in the guard log.
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    if not stream:
        return {**base, 'object': 'chat.completion', 'usage': usage,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': marker},
                             'finish_reason': 'stop', 'logprobs': None}]}
    base['object'] = 'chat.completion.chunk'
    events = [
        {**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': marker}, 'finish_reason': None}]},
        {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
        {**base, 'choices': [], 'usage': usage},
    ]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode() + b'data: [DONE]\n\n'


RENDERERS = {'deepseek_v41': 'tb_agents.deepseek_v41_context'}


def renderer_hooks(env=None):
    """(tokenize_payload or None, reserve or None, timings enabled) from the environment: PI_CONTEXT_RENDERER (a model's
    registry [request].env, aa_evals M3.2 / synth teachers), PI_CONTEXT_LEASE, PI_CONTEXT_TIMINGS. DeepSeek V4.1:
    vLLM's /tokenize crashes on its tool histories, so the vendored encoder renders the prompt."""
    env = os.environ if env is None else env
    name = env.get('PI_CONTEXT_RENDERER') or ''
    renderer_module = RENDERERS.get(name, name)
    lease_module = env.get('PI_CONTEXT_LEASE', '')
    tokenize = importlib.import_module(renderer_module).tokenize_payload if renderer_module else None
    reserve = importlib.import_module(lease_module).reserve if lease_module else None
    timings = env.get('PI_CONTEXT_TIMINGS', '0') == '1'
    return tokenize, reserve, timings


def local_counter(env=None):
    """count(prompt) -> tokens with PI_CONTEXT_TOKENIZER (the pinned tokenizer.json), or None (count through the
    backend's /tokenize)."""
    env = os.environ if env is None else env
    path = env.get('PI_CONTEXT_TOKENIZER') or ''
    if not path:
        return None
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(path)
    return lambda prompt: len(tokenizer.encode(prompt, add_special_tokens=False).ids)


def start_context_guard(backend, logs, backend_file=None, output_cap=None, state_file=None):
    """Start one lifetime-bound guard; the caller owns termination in its finally."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    warn_at, reserve = finish_settings()
    with (Path(logs) / 'context-guard.log').open('a') as output:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--backend', backend, '--window', str(context_window()), '--listen-fd', str(listener.fileno()),
            '--events', str(Path(logs) / 'context-limit.jsonl'),
            *(['--backend-file', str(backend_file)] if backend_file else []),
            *(['--output-cap', str(output_cap)] if output_cap is not None else []),
            *(['--state', str(state_file)] if state_file else []),
            *(['--bound-shortcut'] if os.environ.get('PI_GUARD_BOUND_SHORTCUT') == '1'
              and not os.environ.get('PI_RESPONSE_REPLAY') else []),
            *(['--submit-at-pi-clamp'] if os.environ.get('PI_GUARD_SUBMIT_AT_PI_CLAMP') == '1' else []),
            *(['--backend-force-close'] if os.environ.get('PI_HTTP_FORCE_CLOSE') == '1' else []),
            *(['--finish-warn-at', str(warn_at), '--finish-reserve', str(reserve)] if warn_at or reserve else []),
            *(['--finish-sticky'] if (warn_at or reserve) and finish_sticky() else [])],
            stdout=output, stderr=subprocess.STDOUT, pass_fds=(listener.fileno(),), start_new_session=True)
    listener.close()
    url = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('Pi context guard exited; inspect context-guard.log')
            try:
                with opener.open(url + '/__pi_guard_health', timeout=1) as response:
                    if response.status == 200:
                        return child, url
            except OSError:
                time.sleep(.1)
        raise RuntimeError('Pi context guard did not become ready')
    except BaseException:
        child.terminate()
        child.wait(timeout=10)
        raise


def main():
    import aiohttp
    from aiohttp import web

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--listen-fd', type=int, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--backend-file', type=Path,
                        help='Optional owned endpoint file, resolved once for each request')
    parser.add_argument('--output-cap', type=int,
                        help='Optional authoritative Pi turn cap, using exact headroom instead of Pi estimates')
    parser.add_argument('--state', type=Path, help='Optional context/cache-reuse state file (PI_CONTEXT_STATE)')
    parser.add_argument('--backend-force-close', action='store_true', help='new backend connection per request')
    parser.add_argument('--submit-at-pi-clamp', action='store_true',
                        help='kept for compatibility: v2-no-spin always submits when Pi asks for max_tokens <= 1')
    parser.add_argument('--bound-shortcut', action='store_true',
                        help='skip /tokenize when the previous reported usage + new message bytes leave a full turn '
                             '(exact with a real server, whose usage is the rendered count; eval_q35 sets it)')
    parser.add_argument('--finish-warn-at', type=int, default=0,
                        help='finish notice: warn when at most this many context tokens remain (0 = off)')
    parser.add_argument('--finish-reserve', type=int, default=0,
                        help='finish notice: keep this many tokens for a final turn (0 = off)')
    parser.add_argument('--finish-sticky', action='store_true',
                        help='v4: re-join every earlier notice to its message in later requests (sticky notices)')
    args = parser.parse_args()
    if args.output_cap is not None and not 1 <= args.output_cap <= 16384:
        parser.error('--output-cap must be 1..16384')
    finish = bool(args.finish_warn_at or args.finish_reserve)
    sticky_on = finish and args.finish_sticky
    policy = finish_policy(args.finish_warn_at, args.finish_reserve, sticky_on)
    sticky = {}   # v4: (index, role, base_sha256) -> the notice map entry last joined to that message
    notices = args.events.with_name('finish-notices.jsonl')
    if sticky_on:   # v4: the map file exists from the start, so an archive without it has lost it (finish_map.py)
        notices.touch()
    # the bound shortcut may skip /tokenize only when no notice or reserve clamp could apply
    room = max(TURN, args.finish_warn_at, args.finish_reserve + TURN + FINAL_SLACK) if finish else TURN
    initial_backend = args.backend.rstrip('/').removesuffix('/v1')
    last_backend = initial_backend
    repo = str(Path(__file__).resolve().parents[1])   # renderer modules live in this repo
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # here, not at import: tbshared/numerics.py loads this file alone as a stdlib-only module
    from tb_agents.jsonlog import append_line
    tokenize_payload, reserve, timings_enabled = renderer_hooks()
    count_locally = local_counter()
    state = ContextState(args.state) if args.state else None
    if args.bound_shortcut and (args.output_cap is not None or state is not None or reserve is not None):
        # the exact-headroom cap, the admission state and the KV lease all need the exact count of every request
        print('--bound-shortcut ignored: --output-cap / --state / PI_CONTEXT_LEASE need the exact count', flush=True)
        args.bound_shortcut = False
    session = None
    timing_log = None

    async def startup(app):
        nonlocal session
        # force_close: a pooled connection that the server closed during a long tool pause fails on reuse
        # ('Cannot write to closing transport' -> 500 to Pi); a fresh local connection per request is cheap.
        connector = aiohttp.TCPConnector(force_close=args.backend_force_close)
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30), trust_env=False,
                                        connector=connector)

    async def cleanup(app):
        await session.close()
        if timing_log is not None:
            timing_log.close()

    @asynccontextmanager
    async def reservation(count, budget):
        if reserve is not None and count is not None:
            async with reserve(session, args.events, count, budget):
                yield
        else:
            yield

    # Previous request and its reported prompt+completion tokens: an append-only request is bounded by
    # those plus the UTF-8 bytes of its new messages (a token is at least one byte) and 32 template tokens
    # per message. When that bound leaves a full turn of room, /tokenize (a second chat render on the
    # serving API process) cannot change the outcome and is skipped.
    last = {'messages': None, 'tools': None, 'tokens': None}

    def bound(payload):
        prev, msgs = last['messages'], payload.get('messages') or []
        # Compressed media bytes do not bound processor-expanded vision tokens.
        # Any non-text content must take the full multimodal /tokenize path.
        for message in msgs:
            content = message.get('content')
            if content is not None and not isinstance(content, str):
                if not isinstance(content, list) or any(
                        not isinstance(block, dict) or block.get('type') not in ('text', 'input_text', 'output_text')
                        for block in content):
                    return None
        if (prev is None or last['tokens'] is None or payload.get('tools') != last['tools']
                or len(msgs) <= len(prev) or msgs[:len(prev)] != prev):
            return None
        return last['tokens'] + sum(len(json.dumps(m, ensure_ascii=False).encode()) + 32 for m in msgs[len(prev):]) + 64

    def record_usage(data):
        usage = None
        for line in data.decode(errors='replace').splitlines():
            line = line.removeprefix('data: ').strip()
            if '"usage"' in line:
                try:
                    usage = json.loads(line).get('usage') or usage
                except ValueError:
                    pass
        if usage and usage.get('prompt_tokens') is not None:
            last['tokens'] = int(usage['prompt_tokens']) + int(usage.get('completion_tokens') or 0)

    async def handle(request):
        nonlocal last_backend, timing_log
        if request.path == '/__pi_guard_health':
            return web.json_response({'policy': policy, 'context_window': args.window,
                                      'finish_notice': {'warn_at': args.finish_warn_at, 'reserve': args.finish_reserve,
                                                        **({'sticky': True} if sticky_on else {})} if finish else None})
        # A replacement of the same pinned model must not strand live Pi
        # sessions on the retired host. Capture one backend for both exact
        # tokenization and inference, without editing or restarting history.
        backend = initial_backend
        if args.backend_file:
            try:
                backend = args.backend_file.read_text().strip().rstrip('/').removesuffix('/v1')
                if not backend.startswith(('http://', 'https://')) or any(c.isspace() for c in backend):
                    raise ValueError('invalid endpoint')
            except (OSError, ValueError) as error:
                return web.json_response({'error': {'message': 'Owned model endpoint unavailable: '+str(error)}}, status=503)
            if backend != last_backend:
                append_line(args.events.with_name('backend-transitions.jsonl'), json.dumps({'time': time.time(),
                    'previous': last_backend, 'backend': backend, 'endpoint_file': str(args.backend_file)}))
                last_backend = backend
        body = await request.read()
        count, effective_budget = None, None
        timing_start = time.monotonic()
        timing = {'time': time.time(), 'request_bytes': len(body)}
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ('host', 'content-length', 'transfer-encoding')}
        chat = request.method == 'POST' and request.path.endswith('/chat/completions')
        if chat:
            payload = json.loads(body)
            b = bound(payload)
            previous_tokens = last['tokens']
            last.update(messages=payload.get('messages') or [], tools=payload.get('tools'), tokens=None)
            asked = [int(payload[k]) for k in ('max_tokens', 'max_completion_tokens') if payload.get(k) is not None]
            if asked and min(asked) <= 1:
                # v2-no-spin before any /tokenize: Pi clamps max_tokens to contextWindow - its estimate - 4096
                # (min 1). At 1 the effective budget is <= 1 whatever the exact count, and forwarding it only
                # spins on one-token stubs plus continuation messages until the count reaches the window.
                append_line(args.events, json.dumps({'policy': policy, 'time': time.time(), 'prompt_tokens': None,
                    'previous_prompt_plus_completion_tokens': previous_tokens, 'context_window': args.window,
                    'message_count': len(payload.get('messages', [])), 'requested_output_budget': min(asked),
                    'effective_output_budget': min(asked), 'termination_reason': 'context_limit',
                    'context_limit_source': 'pi_output_budget', 'synthetic': True, 'sft_loss_weight': 0}))
                stream = bool(payload.get('stream'))
                result = terminal_response(payload.get('model', ''), 0, stream)
                return web.Response(body=result, content_type='text/event-stream') if stream else web.json_response(result)
        original = payload.get('messages') or [] if chat else None   # Pi's own list (last['messages'] holds it)
        trailing = len(original) - 1 if chat else None
        applied = []
        if chat and sticky:
            # v4: earlier notices first, so the count includes them. The trailing message's notice is decided below
            # (a retried request gets it afresh: same messages, same count, same notice, and its entry is replaced).
            # A request without a notice never clears an entry: it may belong to another conversation through this
            # guard (an agent querying the endpoint from its sandbox) whose trailing message happens to match.
            payload['messages'], applied = apply_sticky(original, sticky, skip=trailing)
        if chat and args.bound_shortcut and b is not None and b + room + 1024 <= args.window:
            # provably below the window with a full turn of room: nothing to submit or clamp (a trailing message
            # that had a notice is a retry, which the bound never covers: it is re-joined here all the same)
            if sticky:
                payload['messages'] = apply_sticky(original, sticky)[0]
                body = json.dumps(payload).encode()
        elif chat:
            tokenize = {k: payload[k] for k in ('model', 'messages', 'tools', 'chat_template_kwargs') if k in payload}
            tokenize['add_generation_prompt'] = True
            if tokenize_payload is not None:
                try:
                    tokenize = tokenize_payload(payload)
                except Exception as error:
                    # Deterministic: a 503 here would be retried by the proxy for its whole hold window.
                    print(f'Cannot render context: {type(error).__name__}: {error}', flush=True)
                    return web.json_response({'error': {'message': f'Cannot render Pi context: {error}'}}, status=500)
            try:
                if count_locally is not None and 'prompt' in tokenize and not tokenize.get('add_special_tokens', True):
                    count = await asyncio.to_thread(count_locally, tokenize['prompt'])
                    timing['tokenized_locally'] = True
                else:
                    async with session.post(backend + '/tokenize', json=tokenize, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=120)) as response:
                        response.raise_for_status()
                        count = (await response.json())['count']
                if type(count) is not int or count < 0:
                    raise ValueError('invalid tokenizer count')
                timing['render_and_tokenize_seconds'] = time.monotonic()-timing_start
            except Exception as error:
                print(f'Cannot verify rendered context: {type(error).__name__}: {error}', flush=True)
                return web.json_response({'error': {'message': 'Cannot verify rendered Pi context'}}, status=503)
            remaining = args.window - count
            keys = [k for k in ('max_tokens', 'max_completion_tokens') if payload.get(k) is not None]
            requested_budget = min([int(payload[k]) for k in keys] or [16384])
            effective_budget = min(requested_budget, 16384, remaining)
            notice = None
            if finish:
                # With --output-cap the exact headroom supersedes Pi's estimate, so the plan starts from the cap.
                effective_budget, notice = finish_plan(
                    remaining, requested_budget if args.output_cap is None else args.output_cap,
                    args.finish_warn_at, args.finish_reserve)
                # finish_plan never exceeds the remaining context, so an exhausted window (<= 1) still submits below;
                # an exhausted Pi budget (max_tokens <= 1) was submitted before /tokenize
            # Pi's own context estimator can exhaust its budget before the
            # serving tokenizer reaches the literal limit. Its provider clamps
            # that budget to one; forwarding it causes length/continue loops.
            if effective_budget <= 1:
                append_line(args.events, json.dumps({'policy': policy, 'time': time.time(), 'prompt_tokens': count,
                    'context_window': args.window, 'message_count': len(payload.get('messages', [])),
                    'requested_output_budget': requested_budget, 'remaining_context_tokens': remaining,
                    'effective_output_budget': effective_budget,
                    'termination_reason': 'context_limit',
                    'context_limit_source': 'rendered_prompt' if remaining <= 1 else 'pi_output_budget',
                    'synthetic': True, 'sft_loss_weight': 0}))
                stream = bool(payload.get('stream'))
                result = terminal_response(payload.get('model', ''), count, stream)
                return web.Response(body=result, content_type='text/event-stream') if stream else web.json_response(result)
            previous_caps = {key: payload[key] for key in keys}
            if args.output_cap is None:
                for key in keys or ['max_tokens']:
                    payload[key] = effective_budget
            else:
                # Exact serialized headroom supersedes Pi's conservative estimate (and the finish plan, when on).
                if not finish:
                    effective_budget = min(args.output_cap, 16384, remaining)
                for key in keys or ['max_tokens']:
                    payload[key] = effective_budget
                if any(payload.get(k) != v for k, v in previous_caps.items()):
                    append_line(args.events.with_name('context-headroom.jsonl'), json.dumps({'time': time.time(),
                        'backend': backend, 'prompt_tokens': count, 'context_window': args.window,
                        'requested_caps': previous_caps, 'configured_output_cap': args.output_cap,
                        'applied_caps': {k: payload[k] for k in keys or ['max_tokens']},
                        'reason': 'Exact serialized headroom supersedes Pi conservative estimate'}))
            if notice:
                text = notice_text(notice, remaining, effective_budget)
                entry = notice_entry(payload.get('messages'), text)
                payload['messages'] = with_notice(payload.get('messages'), text)
                if sticky_on and not entry['appended']:
                    sticky[sticky_key(entry)] = entry
                append_line(notices, json.dumps({'policy': policy, 'time': time.time(), 'notice': notice,
                    'prompt_tokens': count, 'context_window': args.window, 'remaining_context_tokens': remaining,
                    'requested_output_budget': requested_budget, 'effective_output_budget': effective_budget,
                    'message_count': len(payload['messages']), **entry,
                    **({'sticky_rejoined': len(applied)} if sticky_on else {})}))
            body = json.dumps(payload).encode()
        if state is not None and count is not None:
            state.request(count)
        tail = b''
        async with reservation(count, effective_budget):
            timing['reservation_wait_seconds'] = time.monotonic()-timing_start-timing.get('render_and_tokenize_seconds', 0)
            backend_start = time.monotonic()
            async with session.request(request.method, backend + request.path_qs, data=body, headers=headers) as response:
                timing['backend_headers_seconds'] = time.monotonic()-backend_start
                output = web.StreamResponse(status=response.status, headers={k: v for k, v in response.headers.items()
                    if k.lower() not in ('content-length', 'transfer-encoding', 'content-encoding')})
                await output.prepare(request)
                try:
                    async for chunk in response.content.iter_any():
                        await output.write(chunk)
                        if chat:
                            tail = (tail + chunk)[-65536:]
                    await output.write_eof()
                except ConnectionResetError:
                    pass
                if chat and response.status == 200:
                    record_usage(tail)   # the bound shortcut's reported usage
                if state is not None and count is not None:
                    state.response(count, response_usage(tail) if response.status == 200 else None)
                timing.update(prompt_tokens=count, output_budget=effective_budget,
                    backend_seconds=time.monotonic()-backend_start,
                    total_seconds=time.monotonic()-timing_start, status=response.status)
        if count is not None and timings_enabled:
            if timing_log is None:
                timing_log = args.events.with_name('request-timings.jsonl').open('a', buffering=1)
            timing_log.write(json.dumps(timing)+'\n')
        return output

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route('*', '/{tail:.*}', handle)
    web.run_app(app, sock=socket.socket(fileno=args.listen_fd), shutdown_timeout=2, print=None)


if __name__ == '__main__':
    main()

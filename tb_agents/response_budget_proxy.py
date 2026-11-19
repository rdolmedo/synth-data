"""Per-rollout response budget proxy, with optional recorded teacher prefix replay.

Buffers one upstream turn to enforce the budget in Snowball tokens even when the
teacher uses a different tokenizer. Retries receive the same cached response.
All forwarded histories, notices, real cut tokens and responses are saved for SFT.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tb_agents import response_budget as B
from tb_agents.jsonlog import append_line
from tb_agents.pi_context_limit import (apply_sticky, notice_entry, sticky_key,
    terminal_response, with_notice)
from tb_agents.response_budget_output import (ACCOUNTING_POLICY, canonical_text, is_datakit, parse_tool_text,
    process as process_output, unfinished_tool)

# At the warning threshold a response is cut. When the cut lands inside an open tool call, the call is completed
# instead (same response, same sampling; bounded by the 16k per-response cap), executed, and the notice follows its
# tool result. The completion is not charged: the finishing allowance starts after the call. The total cap stays hard.
WARNING_TOOL_COMPLETION = 'complete-open-tool-call-at-warning-v1'
SAMPLING = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty',
            'seed')
PER_RESPONSE_CAP = 16384
# The v4 context guard clamps a turn that would eat into its final reserve (16,384): Pi's request then carries a
# max_tokens below the budget cap. When that clamp cuts an open tool call, the call is completed too (charged
# normally), within this much of the reserve, so the final turn keeps room (FINAL_SLACK 2048 + notices).
CLAMP_COMPLETION_ROOM = 12288
# The guard's own accounting failure (#531, 11-15 hangs): it is sticky for the session (a transport retry must not
# repeat paid inference), so the sticky proxy must not hold-and-retry it. GUARD_ERROR_HEADER marks it; the sticky
# proxy (inference/sticky_proxy.py, same header name) passes such a 502 straight to Pi, which then ends the attempt
# as infrastructure (eval_q35 worker: 'response_budget_accounting'). Upstream/server 502s carry no header and keep
# their hold. /detokenize, the usual cause (30 s timeout under load), gets one bounded retry first.
GUARD_ERROR_HEADER = 'X-Response-Budget-Error'
ACCOUNTING_ERROR = 'response budget accounting error'
DETOKENIZE_TIMEOUT = 30
DETOKENIZE_TRIES = 2


def start(backend, logs):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    with (Path(logs) / 'response-budget.log').open('a') as output:
        p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--backend', backend,
            '--listen-fd', str(listener.fileno()), '--logs', str(logs)], stdout=output, stderr=subprocess.STDOUT,
            pass_fds=(listener.fileno(),), start_new_session=True)
    listener.close()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f'http://127.0.0.1:{port}'
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError('response budget proxy exited')
        try:
            with opener.open(url + '/__budget_health', timeout=1) as r:
                if r.status == 200:
                    return p, url
        except OSError:
            time.sleep(.1)
    p.terminate()
    raise RuntimeError('response budget proxy did not become ready')


def wire_message(m):
    m = copy.deepcopy(m)
    for tc in m.get('tool_calls') or []:
        if not isinstance(tc['function'].get('arguments'), str):
            tc['function']['arguments'] = json.dumps(tc['function']['arguments'], ensure_ascii=False)
    return m


def sse(result):
    base = {k: result[k] for k in ('id', 'created', 'model')}
    base['object'] = 'chat.completion.chunk'
    c = result['choices'][0]
    delta = copy.deepcopy(c['message'])
    for i, tc in enumerate(delta.get('tool_calls') or []):
        tc['index'] = i
    events = [{**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
              {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': c['finish_reason']}]},
              {**base, 'choices': [], 'usage': result.get('usage', {})}]
    return ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode() + b'data: [DONE]\n\n'


def main():
    import aiohttp
    from aiohttp import web
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True)
    ap.add_argument('--listen-fd', type=int, required=True)
    ap.add_argument('--logs', type=Path, required=True)
    a = ap.parse_args()
    p = B.settings()
    assert p is not None
    backend = a.backend.rstrip('/').removesuffix('/v1')
    replay = json.loads(Path(os.environ['PI_RESPONSE_REPLAY']).read_text()) if os.environ.get('PI_RESPONSE_REPLAY') else None
    turns = replay['replay'] if replay else []
    used, native_generated, turn, sticky, cache = 0, 0, 0, {}, {}
    notice_path = a.logs / 'finish-notices.jsonl'
    notice_path.touch()
    session = None
    lock = asyncio.Lock()
    idmap = {}
    checked_tools = 0
    accounting_failure = None

    def log(x):
        append_line(a.logs / 'response-budget-wire.jsonl', json.dumps({'time': time.time(), 'policy': B.POLICY,
            'accounting_policy': ACCOUNTING_POLICY, **x}, ensure_ascii=False))

    async def startup(app):
        nonlocal session
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30), trust_env=False)

    async def cleanup(app):
        await session.close()

    async def detokenize(model, tokens, headers):
        """The text of raw token IDs from the backend, with one bounded retry (DETOKENIZE_TRIES): a single slow or
        failed /detokenize under load must not end the session's accounting for good."""
        error = None
        for attempt in range(DETOKENIZE_TRIES):
            if attempt:
                log({'detokenize_retry': attempt, 'after': f'{type(error).__name__}: {error}'})
                await asyncio.sleep(1)
            try:
                async with session.post(backend + '/detokenize', json={'model': model, 'tokens': tokens},
                                        headers=headers, timeout=aiohttp.ClientTimeout(total=DETOKENIZE_TIMEOUT)) as decoded:
                    if decoded.status != 200:
                        raise ValueError(f'Backend detokenization failed with status {decoded.status}')
                    text = (await decoded.json()).get('prompt')
                if not isinstance(text, str):
                    raise ValueError('Backend detokenization did not return text')
                return text
            except (ValueError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = e
        raise error

    async def complete_open_call(payload, tokens, text, headers, limit=None):
        """Continue the cut response token-exactly (/tokenize prompt + its generated IDs -> /v1/completions) until
        the open call closes. {'result', 'text', 'ids', ...} when it closes into clean calls, else {'result': None}."""
        out = {'policy': WARNING_TOOL_COMPLETION, 'result': None}
        room = PER_RESPONSE_CAP - len(tokens)
        if limit is not None:
            room = min(room, limit)
        if room <= 0:
            return {**out, 'skipped': 'per-response cap reached'}
        try:
            body = {'model': payload.get('model'), 'messages': payload['messages'], 'add_generation_prompt': True}
            for k in ('tools', 'chat_template_kwargs'):
                if payload.get(k):
                    body[k] = payload[k]
            async with session.post(backend + '/tokenize', json=body, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    raise ValueError(f'tokenize status {r.status}')
                prompt = (await r.json()).get('tokens')
            if not isinstance(prompt, list):
                raise ValueError('tokenize returned no tokens')
            body = {'model': payload.get('model'), 'prompt': prompt + tokens, 'max_tokens': room,
                    'stop': ['</tool_call>'], 'include_stop_str_in_output': True, 'return_token_ids': True,
                    'stream': False, **{k: payload[k] for k in SAMPLING if k in payload}}
            if is_datakit(text):   # Datakit's </tool_call> is a special token: the stop string needs it kept in text
                body['skip_special_tokens'] = False
            async with session.post(backend + '/v1/completions', json=body, headers=headers) as r:
                raw = await r.read()
                if r.status != 200:
                    raise ValueError(f'completions status {r.status}: {raw[:300]!r}')
                res = json.loads(raw)
            c = res['choices'][0]
            new_text, new_ids = c.get('text') or '', c.get('token_ids')
            if not isinstance(new_ids, list) or any(type(i) is not int for i in new_ids):
                raise ValueError('completions returned no raw token IDs')
        except (ValueError, KeyError, TypeError, aiohttp.ClientError, asyncio.TimeoutError) as error:
            return {**out, 'error': str(error)}
        out.update(extra_native_tokens=len(new_ids), finish_reason=c.get('finish_reason'),
                   stop_reason=c.get('stop_reason'), continuation_text=new_text)
        message = parse_tool_text(text + new_text, payload.get('tools')) if new_text.endswith('</tool_call>') else None
        if message is None:
            return out
        ids = tokens + new_ids
        out.update(text=text + new_text, ids=ids, result={
            'id': f'chatcmpl-warncall-{turn}', 'object': 'chat.completion', 'created': int(time.time()),
            'model': payload.get('model', ''),
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls', 'token_ids': ids}],
            'usage': {'completion_tokens': len(ids)}})
        return out

    async def handle(request):
        nonlocal used, native_generated, turn, checked_tools, accounting_failure
        if request.path == '/__budget_health':
            return web.json_response({'policy': B.POLICY, 'accounting_policy': ACCOUNTING_POLICY,
                'budget': p, 'used': used, 'native_generated_tokens': native_generated})
        raw = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length', 'transfer-encoding')}
        if not (request.method == 'POST' and request.path.endswith('/chat/completions')):
            if request.method == 'POST' and request.path.endswith('/tokenize'):
                tokenize = json.loads(raw)
                if tokenize.get('messages'):
                    tokenize['messages'] = apply_sticky(tokenize['messages'], sticky, skip=len(tokenize['messages'])-1)[0]
                    if p['warn_after_response_tokens'] <= used < B.hard_limit(p):
                        left=max(0, p['total_response_tokens']-used)
                        tokenize['messages'] = with_notice(tokenize['messages'], B.notice(left, final=left<=B.reserve(p), extra=p.get('notice_extra')))
                    tokenize['messages'] = B.replace_continue(tokenize['messages'])
                    raw = json.dumps(tokenize).encode()
            async with session.request(request.method, backend + request.path_qs, data=raw, headers=headers) as r:
                return web.Response(body=await r.read(), status=r.status,
                    headers={k: v for k, v in r.headers.items() if k.lower() not in ('content-length', 'transfer-encoding', 'content-encoding')})
        async with lock:
            payload = json.loads(raw)
            stream = payload.get('stream', False)
            if accounting_failure is not None:
                # A broken accounting backend is infrastructure failure, not a
                # model answer. Never let a transport retry repeat paid inference.
                return web.json_response(accounting_failure, status=502, headers={GUARD_ERROR_HEADER: 'accounting'})
            identity=copy.deepcopy(payload)
            if replay and sum(m.get('role')=='assistant'for m in payload.get('messages',[]))<len(turns):
                # Provider retries can change their seed while retaining exactly
                # the same history. Recorded prefix responses must stay idempotent.
                identity.pop('seed',None)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if key in cache:
                result = cache[key]
                return web.Response(body=sse(result), content_type='text/event-stream') if stream else web.json_response(result)
            before = used
            if used >= B.hard_limit(p):
                result = terminal_response(payload.get('model', ''), 0, False, marker=B.CAP_MARK)
                log({'synthetic': True, 'termination_reason': 'response_budget', 'used': used,
                     'native_generated_tokens': native_generated, 'sft_loss_weight': 0,
                     'messages': payload['messages'], 'response': result})
                cache[key] = result
                return web.Response(body=sse(result), content_type='text/event-stream') if stream else web.json_response(result)
            cap = B.output_cap(used,p)
            payload['messages'] = apply_sticky(payload['messages'], sticky, skip=len(payload['messages'])-1)[0]
            if used >= p['warn_after_response_tokens']:
                left=max(0, p['total_response_tokens']-used)
                text = B.notice(left, final=left<=B.reserve(p), extra=p.get('notice_extra'))
                entry = notice_entry(payload['messages'], text)
                assert not entry['appended'], 'Pi should have supplied a continuation or tool result'
                sticky[sticky_key(entry)] = entry
                payload['messages'] = with_notice(payload['messages'], text)
                append_line(notice_path, json.dumps({'policy': B.POLICY, 'notice': 'response_budget', 'time': time.time(),
                                                     **entry}))
            payload['messages'] = B.replace_continue(payload['messages'])
            # Replay executes the prefix's real tools in a fresh sandbox. Remap volatile IDs.
            if replay and turn < len(turns):
                from aa_evals.automationbench.pi.replay import align, remap
                actual = [m for m in payload['messages'] if m['role'] == 'tool']
                expected = [m for m in replay['messages'] if m['role'] == 'tool']
                for n in range(checked_tools, min(len(actual), len(expected))):
                    got, stored = actual[n].get('content') or '', expected[n].get('content') or ''
                    if isinstance(got, list):
                        got = '\n\n'.join(x.get('text', '') for x in got if x.get('type') == 'text')
                    if isinstance(stored, list):
                        stored = '\n\n'.join(x.get('text', '') for x in stored if x.get('type') == 'text')
                    equal = got == stored or align(got, stored, idmap)
                    log({'replay_tool': n, 'matched_up_to_volatile': equal,
                         **({} if equal else {'got': got, 'stored': stored})})
                checked_tools = min(len(actual), len(expected))
                saved = turns[turn]
                msg = wire_message(saved['message'])
                # Only tool arguments carry remapped runtime IDs. Preserve reasoning/prefix text.
                for tc in msg.get('tool_calls') or []:
                    tc['function']['arguments'] = remap(tc['function']['arguments'], idmap)
                ids = saved['ids']
                result = {'id': f'chatcmpl-replay-{turn}', 'created': int(time.time()), 'model': payload.get('model', ''),
                    'object': 'chat.completion', 'choices': [{'index': 0, 'message': msg, 'finish_reason': saved['finish_reason']}],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': len(ids), 'total_tokens': len(ids)}}
                clipped = saved['finish_reason'] == 'length'
                upstream = None
                # 'charge': a recorded turn whose open tool call crossed the warning point is charged only up to it
                accounting = {'charged_tokens': int(saved.get('charge', len(ids))), 'native_tokens': 0, 'replayed': True}
            else:
                asked=[int(payload[k])for k in ('max_tokens','max_completion_tokens')if payload.get(k)is not None]
                requested = max(1, min(asked+[cap,16384]))
                payload['max_tokens'] = requested
                payload.pop('max_completion_tokens', None)
                payload['stream'] = False
                payload.pop('stream_options', None)
                # Tool parsers can drop unfinished arguments and replace a length
                # stop with tool_calls. Raw IDs are the accounting source of truth.
                payload['return_token_ids'] = True
                if used >= p['warn_after_response_tokens'] and os.environ.get('PI_RESPONSE_FINISH_THINKING') == '0':
                    payload['chat_template_kwargs'] = {**(payload.get('chat_template_kwargs') or {}), 'enable_thinking': False}
                async with session.post(backend + request.path_qs, json=payload, headers=headers) as r:
                    raw_result = await r.read()
                    if r.status != 200:
                        log({'upstream_error': r.status, 'detail': raw_result.decode(errors='replace')})
                        return web.Response(body=raw_result, status=r.status, content_type='application/json')
                    result = json.loads(raw_result)
                upstream = copy.deepcopy(result)
                try:
                    tokens = result['choices'][0].get('token_ids')
                    if not isinstance(tokens, list):
                        raise ValueError('Backend did not return requested raw token IDs')
                    text = await detokenize(payload.get('model'), tokens, headers)
                    completion, at_clamp = None, False
                    open_call = len(tokens) >= requested and unfinished_tool(canonical_text(text))
                    if (os.environ.get('PI_RESPONSE_COMPLETE_TOOL_AT_WARNING', '1') != '0' and open_call
                            and used < p['warn_after_response_tokens'] and requested == cap):
                        completion = await complete_open_call(payload, tokens, text, headers)
                    elif (os.environ.get('PI_RESPONSE_COMPLETE_TOOL_AT_CLAMP', '1') != '0' and open_call
                            and requested < min(cap, PER_RESPONSE_CAP)):
                        # the requester's own limit (the v4 context clamp) cut the call: complete it, charged
                        completion, at_clamp = await complete_open_call(payload, tokens, text, headers,
                                                                        CLAMP_COMPLETION_ROOM), True
                    if at_clamp and completion.get('result') is not None:
                        result = completion['result']
                        accounting = process_output(result, completion['text'], cap, len(completion['ids']), p,
                                                    payload.get('tools'))
                        accounting['clamp_tool_completion'] = {
                            **{k: v for k, v in completion.items() if k not in ('result', 'ids', 'text')},
                            'executed': not accounting['clipped'], 'charged': True}
                        if accounting['clipped']:   # over the budget cap: keep the original cut instead
                            result = copy.deepcopy(upstream)
                            accounting = {**process_output(result, text, cap, requested, p, payload.get('tools')),
                                          'clamp_tool_completion': accounting['clamp_tool_completion']}
                            accounting['charged_tokens'] += completion.get('extra_native_tokens', 0)   # never refund
                    elif completion is not None and completion.get('result') is not None:
                        result = completion['result']
                        accounting = process_output(result, completion['text'], 10 ** 9, len(completion['ids']), p,
                                                    payload.get('tools'))
                        # Charge only up to the warning threshold; the completed call is free.
                        accounting['warning_tool_completion'] = {
                            **{k: v for k, v in completion.items() if k not in ('result', 'ids', 'text')},
                            'executed': True, 'charged': False, 'generated_tokens': accounting['charged_tokens']}
                        accounting['charged_tokens'] = cap
                    else:
                        accounting = process_output(result, text, cap, requested, p, payload.get('tools'))
                        if completion is not None:
                            accounting['clamp_tool_completion' if at_clamp else 'warning_tool_completion'] = {
                                **{k: v for k, v in completion.items() if k not in ('result', 'ids', 'text')},
                                'executed': False, 'charged': at_clamp}
                            if at_clamp:   # generation at a clamp is charged even when the call did not close
                                accounting['charged_tokens'] += completion.get('extra_native_tokens', 0)
                except (ValueError, KeyError, TypeError, aiohttp.ClientError, asyncio.TimeoutError) as error:
                    native_generated += (upstream.get('usage') or {}).get('completion_tokens') or 0
                    detail = f'{type(error).__name__}: {error}'   # a timeout's str() is empty
                    accounting_failure = {'error': {'message': f'{ACCOUNTING_ERROR}: {detail}',
                        'type': 'response_budget_accounting_error'}}
                    log({'accounting_error': detail, 'upstream': upstream})
                    return web.json_response(accounting_failure, status=502, headers={GUARD_ERROR_HEADER: 'accounting'})
                msg, ids, clipped = accounting['message'], accounting['ids'], accounting['clipped']
                result['choices'][0]['message'] = wire_message(msg)
                native_generated += accounting['native_tokens']
            # Charge every generated token, including text discarded by a parser
            # or cropped at a Snowball boundary. Never refund hidden generation.
            used += accounting['charged_tokens']
            log({'turn': turn, 'used_before': before, 'used_after': used, 'cap': cap, 'replayed': turn < len(turns),
                 'messages': payload['messages'], 'tools': payload.get('tools'), 'response': result,
                 'real_body_ids': ids, 'clipped': clipped, 'upstream': upstream,
                 'generation_accounting': {k: v for k, v in accounting.items() if k not in ('message', 'ids')},
                 'native_generated_tokens': native_generated})
            turn += 1
            cache[key] = result
            return web.Response(body=sse(result), content_type='text/event-stream') if stream else web.json_response(result)

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route('*', '/{path:.*}', handle)
    web.run_app(app, sock=socket.socket(fileno=a.listen_fd), shutdown_timeout=2, print=None)


if __name__ == '__main__':
    main()

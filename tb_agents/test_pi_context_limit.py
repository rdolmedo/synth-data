"""Exercise the real Pi CLI and the exact-wire context guard without a GPU."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tb_agents.pi_context_limit import CAP_MARK, context_window, settings, start_context_guard
from tb_agents.pi_continuation import CONTINUE, annotate_sessions
from tb_agents.test_pi_reasoning_history import runtime


@contextlib.contextmanager
def backend(count_override=None, reported_prompt_tokens=1000, tokenize_status=200, chat_status=200):
    requests, tokenizations = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                tokenizations.append(payload)
                count = count_override(payload) if callable(count_override) else count_override if count_override is not None else sum(
                    len(str(m.get('content') or '')) + len(m.get('reasoning_content', ''))
                    for m in payload['messages']) // 4
                self.send_response(tokenize_status(payload) if callable(tokenize_status) else tokenize_status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'count': count}).encode())
                return
            requests.append(payload)
            cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
            self.send_response(chat_status)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in [
                {'choices': [{'index': 0, 'delta': {'reasoning_content': 'r' * (cap * 4)}, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
                {'choices': [], 'usage': {'prompt_tokens': reported_prompt_tokens, 'completion_tokens': cap}},
            ]:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests, tokenizations
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextlib.contextmanager
def guard(url, tmp_path, monkeypatch, window=None, **kwargs):
    for key in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(key, raising=False)
    if window is not None:
        monkeypatch.setenv('PI_CONTEXT_WINDOW', str(window))
    process, url = start_context_guard(url, tmp_path, **kwargs)
    try:
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)



@pytest.fixture(autouse=True)
def _no_finish_env(monkeypatch):
    for key in ('PI_FINISH_WARN_AT', 'PI_FINISH_RESERVE', 'PI_FINISH_STICKY'):
        monkeypatch.delenv(key, raising=False)


def post(url, payload):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=10) as response:
        return response.read()


@pytest.mark.parametrize('count', [132000, 200000])
@pytest.mark.parametrize('stream', [False, True])
def test_overflow_finishes_normally_without_generation(count, stream, tmp_path, monkeypatch):
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        payload = {'model': 'test', 'messages': [{'role': 'user', 'content': 'tool output'}],
                   'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {}}}],
                   'chat_template_kwargs': {'clear_thinking': False}, 'max_tokens': 16384, 'stream': stream}
        result = post(proxy, payload).decode()
    assert '"finish_reason": "stop"' in result
    assert CAP_MARK in result
    assert not requests
    assert tokenizations[0]['tools'] == payload['tools']
    assert tokenizations[0]['chat_template_kwargs'] == payload['chat_template_kwargs']
    event = json.loads((tmp_path / 'context-limit.jsonl').read_text())
    assert event['prompt_tokens'] == count and event['sft_loss_weight'] == 0


def test_final_turn_uses_remaining_context(tmp_path, monkeypatch):
    with backend(131900) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        post(proxy, {'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}],
                     'max_tokens': 16384, 'stream': True})
    assert requests[0]['max_tokens'] == 100
    assert not (tmp_path / 'context-limit.jsonl').exists()


def test_real_cli_retains_length_history_then_submits_without_compaction(runtime, tmp_path, monkeypatch):
    node, bundle = runtime
    home = tmp_path / 'home'
    config = home / '.pi/agent'
    config.mkdir(parents=True)
    (config / 'settings.json').write_text(json.dumps(settings()))
    session = tmp_path / 'pi/sessions/context.jsonl'
    session.parent.mkdir(parents=True)
    with backend() as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
            'baseUrl': proxy + '/v1', 'api': 'openai-completions', 'apiKey': 'mock',
            'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'],
                        'contextWindow': 132000, 'maxTokens': 16384,
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
            'compat': {'supportsDeveloperRole': False}}}}))
        process = subprocess.run([str(node), str(bundle / 'cli.js'), '--print', '--mode', 'json',
            '--provider', 'mock', '--model', 'test', '--thinking', 'high', '--no-extensions',
            '--no-skills', '--no-prompt-templates', '--session', str(session), '--', 'Solve this mock task.'],
            env={**{k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
                 'HOME': str(home), 'PI_CODING_AGENT_DIR': str(config), 'NO_PROXY': 'localhost,127.0.0.1'},
            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert 2 < len(requests) < 20
    for i, payload in enumerate(requests):
        assistants = [m for m in payload['messages'] if m['role'] == 'assistant']
        assert len(assistants) == i
        for previous, message in zip(requests[:i], assistants):
            assert message['reasoning_content'] == 'r' * (previous.get('max_tokens', previous.get('max_completion_tokens')) * 4)
        assert not any('You are a context summarization assistant.' in str(m.get('content', '')) for m in payload['messages'])
        assert 0 < payload.get('max_tokens', payload.get('max_completion_tokens')) <= 16384
    events = [json.loads(line) for line in session.read_text().splitlines()]
    assert not any(e['type'] == 'compaction' for e in events)
    messages = [e['message'] for e in events if e['type'] == 'message']
    assert messages[-1]['stopReason'] == 'stop'
    assert any(b.get('text') == CAP_MARK for b in messages[-1]['content'])
    assert any(CONTINUE in str(m.get('content')) for m in messages)
    annotations = annotate_sessions(tmp_path)['sessions'][0]['entries']
    assert annotations[-1]['sft_loss_weight'] == 0
    assert (tmp_path / 'context-limit.jsonl').exists()


def test_context_settings_and_conflicts():
    assert settings()['compaction']['enabled'] is False
    assert context_window({}) == 132000
    with pytest.raises(ValueError, match='Conflicting'):
        context_window({'PI_CONTEXT': '132000', 'MAX_LEN': '1048576'})


def test_harbor_adapter_writes_disabled_compaction(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, patch
    from harbor.agents.installed.pi import Pi
    from tb_agents.harness_agents import PiCont16kCached
    monkeypatch.setenv('PI_SETTINGS_JSON', '{"compaction":{"enabled":true}}')
    agent = PiCont16kCached.__new__(PiCont16kCached)
    agent._upload_config_text = AsyncMock()
    with patch.object(Pi, '_write_custom_models_json', new_callable=AsyncMock):
        asyncio.run(agent._write_custom_models_json(None, {}))
    written = json.loads(agent._upload_config_text.call_args.kwargs['content'])
    assert written['compaction']['enabled'] is False


@pytest.mark.parametrize('field', ['max_tokens', 'max_completion_tokens'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('count,budget', [(131999, 16384), (131999, 1), (131000, 1), (1000, 1)])
def test_exhausted_output_budget_submits_without_one_token_inference(field, stream, count, budget, tmp_path, monkeypatch):
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        result = post(proxy, {'model': 'test', 'messages': [{'role': 'user', 'content': CONTINUE}],
                             field: budget, 'stream': stream}).decode()
    assert '"finish_reason": "stop"' in result
    assert CAP_MARK in result and not requests
    event = json.loads((tmp_path / 'context-limit.jsonl').read_text())
    assert event['effective_output_budget'] <= 1
    assert event['requested_output_budget'] == budget
    assert event['sft_loss_weight'] == 0


def test_real_pi_estimator_exhaustion_submits_on_first_followup(runtime, tmp_path, monkeypatch):
    node, bundle = runtime
    config = tmp_path / 'home/.pi/agent'
    config.mkdir(parents=True)
    (config / 'settings.json').write_text(json.dumps(settings()))
    session = tmp_path / 'session.jsonl'
    with backend(1000, reported_prompt_tokens=132000) as (url, requests, _), guard(url, tmp_path, monkeypatch) as proxy:
        (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
            'baseUrl': proxy + '/v1', 'api': 'openai-completions', 'apiKey': 'mock',
            'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'],
                        'contextWindow': 132000, 'maxTokens': 16384,
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
            'compat': {'supportsDeveloperRole': False}}}}))
        process = subprocess.run([str(node), str(bundle / 'cli.js'), '--print', '--mode', 'json',
            '--provider', 'mock', '--model', 'test', '--thinking', 'high', '--no-extensions',
            '--no-skills', '--no-prompt-templates', '--session', str(session), '--', 'Solve this mock task.'],
            env={**{k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
                 'HOME': str(config.parents[1]), 'PI_CODING_AGENT_DIR': str(config), 'NO_PROXY': 'localhost,127.0.0.1'},
            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert len(requests) == 1, 'Repeated inference after the Pi estimator exhausted its output budget'
    events = [json.loads(line) for line in session.read_text().splitlines()]
    assert not any(e['type'] == 'compaction' for e in events)
    messages = [e['message'] for e in events if e['type'] == 'message']
    assert sum(m['role'] == 'user' and CONTINUE in str(m.get('content')) for m in messages) == 1
    assert messages[-1]['stopReason'] == 'stop'
    assert CAP_MARK in str(messages[-1]['content'])
    stopped = json.loads((tmp_path / 'context-limit.jsonl').read_text())
    assert stopped['context_limit_source'] == 'pi_output_budget'


# ---------------------------------------------------------------- #375 finish notice (opt-in)
from tb_agents.pi_context_limit import (FINISH_POLICY, NOTICE_MARGIN, finish_plan, finish_settings,  # noqa: E402
                                        with_notice)


@pytest.mark.parametrize('remaining,requested,expect', [
    (100000, 16384, (16384, None)),                 # far from the window: v2 budget, no notice
    (39000, 16384, (16384, 'warn')),                # inside warn_at, a full turn still leaves the reserve
    (30000, 16384, (30000 - 16384, 'warn')),        # a full turn would eat into the reserve: clamp to keep it
    (18000, 16384, (16384, 'final')),               # reserve + slack: the final turn
    (24000, 16384, (16384, 'final')),               # #382 S1: below reserve + turn // 2 -> final, not a 7.6k clamp
    (5000, 16384, (5000 - NOTICE_MARGIN, 'final')), # final turn with whatever is left
    (39000, 4000, (4000, 'warn')),                  # Pi's own smaller request is honoured
])
def test_finish_plan(remaining, requested, expect):
    assert finish_plan(remaining, requested, 40000, 16384) == expect


def test_finish_plan_off_is_v2():
    for remaining in (100000, 30000, 18000, 5000, 100):
        assert finish_plan(remaining, 16384, 0, 0) == (min(16384, remaining), None)


def test_finish_settings_validate():
    assert finish_settings({}) == (0, 0)
    assert finish_settings({'PI_FINISH_WARN_AT': '40000', 'PI_FINISH_RESERVE': '16384'}) == (40000, 16384)
    with pytest.raises(ValueError):
        finish_settings({'PI_FINISH_RESERVE': '100'})


def test_with_notice_appends_without_mutating():
    user = [{'role': 'user', 'content': 'please continue.'}]
    tool = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a'}]}, {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'}]
    blocks = [{'role': 'user', 'content': [{'type': 'text', 'text': 'x'}]}]
    out = with_notice(user, 'N')
    assert out[-1]['content'] == 'please continue.\n\nN' and user[0]['content'] == 'please continue.'
    out = with_notice(tool, 'N')   # joined to the tool result: no new user turn (templates keep the same history)
    assert len(out) == 2 and out[0] == tool[0] and out[1]['content'] == 'ok\n\nN' and tool[1]['content'] == 'ok'
    out = with_notice(blocks, 'N')
    assert out[-1]['content'][-1] == {'type': 'text', 'text': 'N'} and len(blocks[0]['content']) == 1


@pytest.mark.parametrize('count,budget,kind', [(131072 - 30000, 30000 - 16384, 'warn'), (131072 - 18000, 16384, 'final')])
def test_guard_finish_notice_end_to_end(count, budget, kind, tmp_path, monkeypatch):
    monkeypatch.setenv('PI_FINISH_WARN_AT', '40000')
    monkeypatch.setenv('PI_FINISH_RESERVE', '16384')
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(proxy + '/__pi_guard_health', timeout=5) as r:
            health = json.loads(r.read())
        msgs = [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': '', 'reasoning_content': 'r' * 40},
                {'role': 'user', 'content': 'Thinking exceeded the maximum thinking budget, please continue.'}]
        post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert health['policy'] == FINISH_POLICY and health['finish_notice'] == {'warn_at': 40000, 'reserve': 16384}
    sent = requests[0]
    assert sent['max_tokens'] == budget
    assert sent['messages'][:2] == msgs[:2]                       # history (reasoning included) untouched
    assert sent['messages'][-1]['content'].startswith(msgs[-1]['content'] + '\n\n[Harness notice]')
    assert ('final turn' in sent['messages'][-1]['content']) == (kind == 'final')
    ev = json.loads((tmp_path / 'finish-notices.jsonl').read_text())
    assert ev['notice'] == kind and ev['effective_output_budget'] == budget
    assert not (tmp_path / 'context-limit.jsonl').exists()        # a notice is not a context ending


def test_guard_without_finish_env_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv('PI_FINISH_WARN_AT', raising=False)
    monkeypatch.delenv('PI_FINISH_RESERVE', raising=False)
    with backend(131072 - 30000) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        msgs = [{'role': 'user', 'content': 'task'}]
        post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert requests[0]['max_tokens'] == 16384 and requests[0]['messages'] == msgs
    assert not (tmp_path / 'finish-notices.jsonl').exists()


def test_finish_plan_never_clamps_below_half_a_turn():
    for remaining in range(20000, 40001, 97):
        budget, kind = finish_plan(remaining, 16384, 40000, 16384)
        if kind == 'warn' and budget < 16384:
            assert budget >= 16384 // 2, (remaining, budget)


# ---------------------------------------------------------------- #375 sftnotice: v4 sticky notices + notice map
from tb_agents.pi_context_limit import (STICKY_POLICY, apply_sticky, join_notice, notice_base_sha256,  # noqa: E402
                                        notice_base_text, notice_entry)

CONT = 'Thinking exceeded the maximum thinking budget, please continue.'


def test_notice_base_text_and_join():
    assert notice_base_text('a') == 'a' and notice_base_text(None) == ''
    blocks = [{'type': 'text', 'text': 'a'}, {'type': 'image_url', 'image_url': {'url': 'x'}}, {'type': 'text', 'text': 'b'}]
    assert notice_base_text(blocks) == 'a\n\nb'
    joined = join_notice({'role': 'user', 'content': blocks}, 'N')
    assert joined['content'][-1] == {'type': 'text', 'text': 'N'} and len(blocks) == 3
    assert notice_base_text(joined['content']) == 'a\n\nb\n\nN'   # list join == str join
    assert join_notice({'role': 'tool', 'content': 'ok'}, 'N')['content'] == 'ok\n\nN'
    assert join_notice({'role': 'assistant', 'content': 'x'}, 'N') is None
    assert join_notice({'role': 'user', 'content': None}, 'N') is None
    e = notice_entry([{'role': 'system', 'content': 's'}, {'role': 'tool', 'content': 'ok'}], 'N')
    assert e == {'index': 1, 'role': 'tool', 'base_sha256': notice_base_sha256('ok'), 'text': 'N', 'appended': False}
    assert notice_entry([{'role': 'assistant', 'content': 'x'}], 'N')['appended'] is True


def _conv(n_turns):
    """system, task, then n_turns x (assistant, 'please continue') - identical continuation messages."""
    msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'task'}]}]
    for k in range(n_turns):
        msgs += [{'role': 'assistant', 'content': None, 'reasoning_content': f'r{k}'},
                 {'role': 'user', 'content': [{'type': 'text', 'text': CONT}]}]
    return msgs


def test_apply_sticky_identical_continuations_only_at_their_index():
    msgs = _conv(3)                                     # 'please continue' at 3, 5 and 7, identical
    e = notice_entry(msgs[:6], 'N5')                    # notice joined at index 5
    sticky = {(e['index'], e['role'], e['base_sha256']): e}
    out, applied = apply_sticky(msgs, sticky)
    assert applied == [5] and out[5]['content'][-1]['text'] == 'N5'
    assert out[3] == msgs[3] and out[7] == msgs[7] and msgs[5]['content'][-1]['text'] == CONT   # input untouched
    assert apply_sticky(msgs, sticky, skip=5)[1] == []
    assert apply_sticky(msgs[:5], sticky)[1] == []      # a shorter request does not have it
    other = [{'role': 'user', 'content': 'x'}] * 6        # another conversation: different text at index 5
    assert apply_sticky(other, sticky)[1] == []


def _splice(msgs, notices):
    """The SFT splice (sft_qwen35/prepare.py `_splice_finish_notices`): latest entry per index, checked, joined."""
    last = {}
    for x in notices:
        last[x['index']] = x
    out = list(msgs)
    for i, x in last.items():
        assert out[i]['role'] == x['role'] and notice_base_sha256(out[i].get('content')) == x['base_sha256']
        out[i] = join_notice(out[i], x['text'])
    return out


def _sticky_env(monkeypatch, sticky=True, shortcut=False):
    monkeypatch.setenv('PI_FINISH_WARN_AT', '40000')
    monkeypatch.setenv('PI_FINISH_RESERVE', '16384')
    monkeypatch.setenv('PI_FINISH_STICKY', '1') if sticky else monkeypatch.delenv('PI_FINISH_STICKY', raising=False)
    monkeypatch.setenv('PI_GUARD_BOUND_SHORTCUT', '1') if shortcut else monkeypatch.delenv('PI_GUARD_BOUND_SHORTCUT', raising=False)


def test_guard_sticky_notices_round_trip(tmp_path, monkeypatch):
    _sticky_env(monkeypatch)
    count = lambda p: 131072 - 39000 + len(p['messages'])   # noqa: E731 - always inside warn_at (a warn notice)
    reqs = [_conv(1), _conv(2), _conv(2), _conv(3)]          # the third request retries the second
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(proxy + '/__pi_guard_health', timeout=5) as r:
            health = json.loads(r.read())
        for msgs in reqs:
            post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert health['policy'] == STICKY_POLICY and health['finish_notice']['sticky'] is True
    notices = [json.loads(x) for x in (tmp_path / 'finish-notices.jsonl').read_text().splitlines()]
    assert [x['index'] for x in notices] == [3, 5, 5, 7] and all(x['policy'] == STICKY_POLICY for x in notices)
    assert [x['sticky_rejoined'] for x in notices] == [0, 1, 1, 2]
    assert all(x['role'] == 'user' and x['base_sha256'] == notice_base_sha256([{'type': 'text', 'text': CONT}])
               for x in notices)                          # identical continuation messages, told apart by index
    for k, (sent, tok) in enumerate(zip(requests, tokenizations)):
        assert len(sent['messages']) == len(reqs[k])
        # every earlier notice is in the forwarded request and in what /tokenize counted
        for x in notices[:k]:
            if x['index'] < len(sent['messages']) - 1:
                assert sent['messages'][x['index']]['content'][-1]['text'] == x['text']
                assert tok['messages'][x['index']]['content'][-1]['text'] == x['text']
        # the trailing message has exactly one notice (a retry gets it afresh, not twice)
        assert sum(p['text'].startswith('[Harness notice]') for p in sent['messages'][-1]['content']) == 1
        assert tok['messages'][-1] == reqs[k][-1]          # counted before its own notice (margin 256)
    # the forwarded requests are append-only extensions of each other ...
    assert requests[1]['messages'] == requests[2]['messages'] and requests[3]['messages'][:6] == requests[2]['messages']
    # ... and splicing the notice map into Pi's own list gives the final forwarded request
    assert _splice(reqs[-1], notices) == requests[-1]['messages']
    assert requests[-1]['messages'][3]['content'][-1]['text'] == notices[0]['text']   # not moved to a later 'continue'


def test_guard_v3_notices_are_not_sticky(tmp_path, monkeypatch):
    _sticky_env(monkeypatch, sticky=False)
    count = lambda p: 131072 - 39000   # noqa: E731
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        for msgs in (_conv(1), _conv(2)):
            post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert requests[1]['messages'][:4] == _conv(1)        # v3: the earlier notice is gone
    lines = [json.loads(x) for x in (tmp_path / 'finish-notices.jsonl').read_text().splitlines()]
    assert {x['policy'] for x in lines} == {FINISH_POLICY} and 'sticky_rejoined' not in lines[0]


def test_guard_sticky_with_bound_shortcut(tmp_path, monkeypatch):
    """After a notice, requests the bound shortcut proves far from the window skip /tokenize but still carry it; the
    shortcut's previous list stays Pi's own (without notices), so it keeps applying."""
    _sticky_env(monkeypatch, shortcut=True)
    calls = []

    def count(p):
        calls.append(len(p['messages']))
        return 131072 - 39000 if len(calls) == 1 else 1000
    base = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'}]
    turn = lambda k: [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': f't{k}', 'type': 'function',   # noqa: E731
                      'function': {'name': 'bash', 'arguments': '{}'}}]}, {'role': 'tool', 'tool_call_id': f't{k}', 'content': f'out{k}'}]
    reqs = [base, base + turn(0), base + turn(0) + turn(1)]
    with backend(count, reported_prompt_tokens=1000) as (url, requests, tokenizations), \
            guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        for msgs in reqs:
            post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert len(tokenizations) == 1                       # requests 2 and 3 took the shortcut
    notice = json.loads((tmp_path / 'finish-notices.jsonl').read_text())
    assert notice['index'] == 1
    for sent in requests[1:]:
        assert sent['messages'][1]['content'] == 'task\n\n' + notice['text']
        assert sent['messages'][2:] == reqs[len(sent['messages']) // 2 - 1][2:]


def test_guard_sticky_survives_another_conversation(tmp_path, monkeypatch):
    """An agent querying the endpoint from its sandbox goes through the same guard: its requests neither get the main
    conversation's notices nor clear them."""
    _sticky_env(monkeypatch)
    count = lambda p: 131072 - 39000 if p['messages'][0].get('content') == 'sys' else 500   # noqa: E731
    side = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a'}]
    side = side + [{'role': 'user', 'content': [{'type': 'text', 'text': CONT}]}]   # main's message, at index 4
    side3 = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'q'},
             {'role': 'user', 'content': [{'type': 'text', 'text': CONT}]}]        # trailing = main's notice message
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        post(proxy, {'model': 'test', 'messages': _conv(1), 'max_tokens': 16384, 'stream': True})
        post(proxy, {'model': 'test', 'messages': side3, 'max_tokens': 16384, 'stream': True})
        post(proxy, {'model': 'test', 'messages': side, 'max_tokens': 16384, 'stream': True})
        post(proxy, {'model': 'test', 'messages': _conv(2), 'max_tokens': 16384, 'stream': True})
    first = json.loads((tmp_path / 'finish-notices.jsonl').read_text().splitlines()[0])
    assert requests[1]['messages'] == side3 and requests[2]['messages'] == side   # far from the window: no notice
    assert len((tmp_path / 'finish-notices.jsonl').read_text().splitlines()) == 2
    assert requests[3]['messages'][3]['content'][-1]['text'] == first['text']   # main kept it


def test_guard_off_is_v2_with_sticky_env(tmp_path, monkeypatch):
    monkeypatch.delenv('PI_FINISH_WARN_AT', raising=False)
    monkeypatch.delenv('PI_FINISH_RESERVE', raising=False)
    monkeypatch.setenv('PI_FINISH_STICKY', '1')           # ignored without warn_at/reserve
    with backend(131072 - 30000) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(proxy + '/__pi_guard_health', timeout=5) as r:
            assert json.loads(r.read())['policy'] == 'submit-at-context-limit-v2-no-spin'
        post(proxy, {'model': 'test', 'messages': _conv(1), 'max_tokens': 16384, 'stream': True})
    assert requests[0]['messages'] == _conv(1) and requests[0]['max_tokens'] == 16384
    assert not (tmp_path / 'finish-notices.jsonl').exists()


@pytest.mark.parametrize('sticky', [True, False])
def test_guard_v4_creates_the_notice_map_at_startup(sticky, tmp_path, monkeypatch):
    """#375 sftnotice review SHOULD-1: in v4 the (empty) map file exists from the start, so a trial archive without it
    has lost it. v3 keeps its old behaviour (the file only appears with a first notice)."""
    _sticky_env(monkeypatch, sticky=sticky)
    with backend(1000) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        post(proxy, {'model': 'test', 'messages': _conv(1), 'max_tokens': 16384, 'stream': True})   # far: no notice
    path = tmp_path / 'finish-notices.jsonl'
    assert path.exists() == sticky
    if sticky:
        assert path.read_text() == ''


# ---------------------------------------------------------------- task-synthesis' guard options (#480 merge)
# backend file, exact-headroom output cap, state file (synth admission), in-process renderer counts, the v4 record
# synth/config.py freezes, atomic annotation publication. From task-synthesis' tb_agents/test_pi_context_limit.py.
from tb_agents.pi_context_limit import finish_notice_record  # noqa: E402


def test_owned_endpoint_rotation_keeps_exact_guard_and_history(tmp_path, monkeypatch):
    for key in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(key, raising=False)
    with backend(131900) as (first, first_requests, first_counts), backend(132000) as (second, second_requests, second_counts):
        endpoint = tmp_path/'endpoint.url'
        endpoint.write_text(first+'\n')
        process, proxy = start_context_guard(first, tmp_path, backend_file=endpoint)
        payload = {'model': 'test', 'messages': [{'role': 'user', 'content': 'task'},
                   {'role': 'assistant', 'content': '', 'reasoning_content': 'retained reasoning'},
                   {'role': 'user', 'content': CONTINUE}],
                   'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {}}}],
                   'chat_template_kwargs': {'clear_thinking': False}, 'max_tokens': 16384, 'stream': True}
        try:
            post(proxy, payload)
            endpoint.write_text(second+'\n')
            result = post(proxy, payload).decode()
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert first_requests[0]['max_tokens'] == 100
    assert first_counts[0] == second_counts[0]
    assert first_counts[0]['messages'] == payload['messages']
    assert not second_requests
    assert CAP_MARK in result and '"finish_reason": "stop"' in result
    transition = json.loads((tmp_path/'backend-transitions.jsonl').read_text())
    assert transition['previous'] == first.removesuffix('/v1')
    assert transition['backend'] == second.removesuffix('/v1')
    assert json.loads((tmp_path/'context-limit.jsonl').read_text())['sft_loss_weight'] == 0


@pytest.mark.parametrize('count,cap', [(128001, 16384), (131999, 16384), (130000, 512)])
def test_exhausted_pi_budget_submits_even_with_configured_turn_cap(count, cap, tmp_path, monkeypatch):
    for key in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(key, raising=False)
    with backend(count) as (url, requests, tokenizations):
        process, proxy = start_context_guard(url, tmp_path, output_cap=cap)
        try:
            result = post(proxy, {'model': 'test', 'messages': [{'role': 'user', 'content': 'same history'}],
                         'max_completion_tokens': 1, 'stream': True}).decode()
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert not requests
    assert CAP_MARK in result and '"finish_reason": "stop"' in result
    event = json.loads((tmp_path/'context-limit.jsonl').read_text())
    assert event['effective_output_budget'] <= 1


def test_interrupted_annotation_publication_preserves_previous_sidecar(tmp_path, monkeypatch):
    sessions = tmp_path/'pi/sessions'
    sessions.mkdir(parents=True)
    (sessions/'one.jsonl').write_text(json.dumps({'type': 'message', 'id': 'a', 'message': {
        'role': 'assistant', 'content': [{'type': 'text', 'text': CAP_MARK}], 'stopReason': 'stop'}})+'\n')
    annotate_sessions(tmp_path)
    target = tmp_path/'pi/sft-events.json'
    previous = target.read_bytes()
    original = Path.write_text

    def interrupted(path, data, *args, **kwargs):
        if path.name.startswith('sft-events.json.'):
            original(path, data[:20], *args, **kwargs)
            raise OSError('interrupted annotation write')
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        annotate_sessions(tmp_path)
    assert target.read_bytes() == previous
    assert json.loads(previous)['sessions'][0]['entries'][0]['sft_loss_weight'] == 0


def test_state_file_reports_context_and_cache_reuse(tmp_path, monkeypatch):
    """PI_CONTEXT_STATE: the agent's held context and vLLM's cached-prompt counts, for synth admission."""
    for key in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(key, raising=False)
    state = tmp_path / 'agent-state.json'
    with backend(count_override=5000, reported_prompt_tokens=5000) as (url, requests, _):
        process, proxy = start_context_guard(url, tmp_path, state_file=state)
        try:
            for _ in range(2):
                post(proxy, {'model': 'm', 'stream': True, 'max_tokens': 300,
                             'messages': [{'role': 'user', 'content': 'x'}]})
        finally:
            process.terminate()
            process.wait(timeout=10)
    record = json.loads(state.read_text())
    assert (record['context_tokens'], record['turns'], record['inflight']) == (5300, 2, False)
    # The fake backend reports no prompt_tokens_details, so reuse stays unmeasured.
    assert (record['turns_with_cache_details'], record['expected_cached_tokens']) == (0, 0)


def test_renderer_prompt_counted_in_process_without_tokenize(tmp_path, monkeypatch):
    """PI_CONTEXT_TOKENIZER: a renderer's raw prompt is counted with the local tokenizer, never via /tokenize."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    words.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    words.save(str(tmp_path / 'tokenizer.json'))
    monkeypatch.setenv('PI_CONTEXT_RENDERER', 'deepseek_v41')
    monkeypatch.setenv('PI_CONTEXT_TOKENIZER', str(tmp_path / 'tokenizer.json'))
    monkeypatch.setenv('PI_CONTEXT_WINDOW', '132000')
    monkeypatch.setenv('PI_CONTEXT_TIMINGS', '1')   # synth/worker.py sets it with a renderer (shared default: off)
    payload = {'model': 'deepseek-v41-flash-1017', 'stream': False, 'max_tokens': 16384,
               'messages': [{'role': 'user', 'content': 'word ' * 131000}]}
    with backend() as (url, requests, tokenizations):
        process, proxy = start_context_guard(url, tmp_path)
        try:
            post(proxy, payload)
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert tokenizations == []                                   # no /tokenize round trip
    assert requests and requests[0]['max_tokens'] < 1100          # capped by the local count (~131k words)
    timings = [json.loads(line) for line in (tmp_path / 'request-timings.jsonl').read_text().splitlines()]
    assert timings and timings[0]['tokenized_locally'] is True


def test_finish_notice_record_matches_pi_eval():
    """The frozen record is aa-evals pi_eval run.finish_notice's for 40000:16384 (the eval side derives v4 from it)."""
    import hashlib
    from tb_agents import pi_context_limit as G
    r = finish_notice_record(40000, 16384)
    assert r == {'warn_at': 40000, 'reserve': 16384, 'policy': STICKY_POLICY, 'sticky': True, 'notice_margin': 256,
                 'final_slack': 2048, 'final_at': 16384 + 8192,
                 'text_sha256': hashlib.sha256((G.FINISH_WARN + '\n' + G.FINISH_FINAL).encode()).hexdigest()}
    with pytest.raises(ValueError):
        finish_notice_record(40000, 100)


def test_v4_with_exact_headroom_cap_plans_from_the_cap(tmp_path, monkeypatch):
    """--output-cap (PI_CONTEXT_EXACT_HEADROOM): the finish plan starts from the cap, not Pi's estimate; the headroom
    log records the change and the notice is joined as without the cap."""
    _sticky_env(monkeypatch)
    with backend(131072 - 30000) as (url, requests, tokenizations), \
            guard(url, tmp_path, monkeypatch, window=131072, output_cap=16384) as proxy:
        post(proxy, {'model': 'test', 'messages': _conv(1), 'max_tokens': 4000, 'stream': True})
    assert requests[0]['max_tokens'] == 30000 - 16384           # clamped to keep the reserve, from the 16,384 cap
    assert requests[0]['messages'][-1]['content'][-1]['text'].startswith('[Harness notice]')
    notice = json.loads((tmp_path / 'finish-notices.jsonl').read_text())
    assert notice['notice'] == 'warn' and notice['effective_output_budget'] == 30000 - 16384
    assert json.loads((tmp_path / 'context-headroom.jsonl').read_text())['applied_caps'] == {'max_tokens': 30000 - 16384}


@pytest.mark.parametrize('count,asked', [(131072 - 200, 16384), (131072 - 30000, 1)])
def test_v4_submits_when_nothing_is_left(count, asked, tmp_path, monkeypatch):
    """Within the notice margin of the window (or with Pi's budget exhausted) the attempt submits, no notice."""
    _sticky_env(monkeypatch)
    with backend(count) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch, window=131072) as proxy:
        result = post(proxy, {'model': 'test', 'messages': _conv(1), 'max_tokens': asked, 'stream': True}).decode()
    assert not requests and CAP_MARK in result
    assert json.loads((tmp_path / 'context-limit.jsonl').read_text())['policy'] == STICKY_POLICY
    assert (tmp_path / 'finish-notices.jsonl').read_text() == ''


def test_v4_state_file_counts_the_notices(tmp_path, monkeypatch):
    """PI_CONTEXT_STATE (synth admission) sees the prompt as counted, sticky notices included."""
    _sticky_env(monkeypatch)
    state = tmp_path / 'agent-state.json'
    count = lambda p: 131072 - 39000 + sum(len(json.dumps(m)) for m in p['messages']) // 1000   # noqa: E731
    with backend(count) as (url, requests, tokenizations), \
            guard(url, tmp_path, monkeypatch, window=131072, state_file=state) as proxy:
        for msgs in (_conv(1), _conv(2)):
            post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
    assert tokenizations[1]['messages'][3]['content'][-1]['text'].startswith('[Harness notice]')
    assert json.loads(state.read_text())['last_prompt_tokens'] == count(tokenizations[1])


def test_timings_off_by_default_even_with_a_renderer(monkeypatch):
    """aa-evals' guard wrote no request-timings.jsonl (AA DeepSeek V4.1 runs set PI_CONTEXT_RENDERER); synthesis turns
    them on explicitly (synth/worker.py)."""
    from tb_agents.pi_context_limit import renderer_hooks
    assert renderer_hooks({'PI_CONTEXT_RENDERER': 'deepseek_v41'})[2] is False
    assert renderer_hooks({'PI_CONTEXT_RENDERER': 'deepseek_v41', 'PI_CONTEXT_TIMINGS': '1'})[2] is True


def test_bound_shortcut_yields_to_exact_count_options(tmp_path, monkeypatch):
    """PI_GUARD_BOUND_SHORTCUT with --state: every request is counted exactly (the admission state needs it)."""
    monkeypatch.setenv('PI_GUARD_BOUND_SHORTCUT', '1')
    monkeypatch.delenv('PI_RESPONSE_REPLAY', raising=False)
    state = tmp_path / 'agent-state.json'
    msgs = [{'role': 'user', 'content': 'task'}]
    with backend(1000) as (url, requests, tokenizations), \
            guard(url, tmp_path, monkeypatch, window=131072, state_file=state) as proxy:
        post(proxy, {'model': 'test', 'messages': msgs, 'max_tokens': 16384, 'stream': True})
        post(proxy, {'model': 'test', 'messages': msgs + [{'role': 'assistant', 'content': 'x'},
                                                          {'role': 'user', 'content': 'y'}],
                     'max_tokens': 16384, 'stream': True})
    assert len(tokenizations) == 2 and len(requests) == 2
    assert json.loads(state.read_text())['turns'] == 2

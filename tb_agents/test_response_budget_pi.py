"""Exercise cost accounting and submit-as-is with a local backend and real Pi."""
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request

from tb_agents import response_budget as B
from tb_agents.pi_context_limit import settings
from tb_agents.pi_continuation import annotate_sessions
from tb_agents.response_budget_proxy import start
from tb_agents.test_pi_reasoning_history import runtime  # noqa: F401
from tb_agents.test_response_budget import mock_generation


@contextlib.contextmanager
def backend(reply):
    chats = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/detokenize'):
                result = {'prompt': B.tokenizer().decode(payload['tokens'], skip_special_tokens=False)}
            else:
                assert self.path.endswith('/chat/completions')
                chats.append(copy.deepcopy(payload))
                result = reply(payload, len(chats))
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', chats
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextlib.contextmanager
def guard(url, logs, budget, monkeypatch):
    monkeypatch.setenv('PI_RESPONSE_BUDGET', json.dumps(budget))
    child, endpoint = start(url, logs)
    try:
        yield endpoint
    finally:
        child.terminate()
        child.wait(timeout=10)


def post(endpoint, payload):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(endpoint + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=20) as response:
        return json.loads(response.read())


def test_truncated_tool_cost_is_not_refunded_and_total_cap_prevents_more_generation(tmp_path, monkeypatch):
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 100,
              'total_response_tokens': 200, 'token_counting': 'backend-native'}

    def reply(payload, index):
        assert payload['return_token_ids'] is True
        if index == 1:
            return mock_generation({'role': 'assistant', 'reasoning_content': 'Think. ' * 300, 'content': ''},
                                   'length', payload['max_tokens'])
        raw = '<tool_call>\n<function=write>\n<parameter=path>artifact.txt</parameter>\n<parameter=content>' + ' word' * 500
        ids = B.encode(raw)[:payload['max_tokens']]
        return {'id': str(index), 'created': 1, 'model': 'test', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'bad', 'type': 'function',
                'function': {'name': 'write', 'arguments': '{"path":"artifact.txt"}'}}]},
            'finish_reason': 'tool_calls', 'token_ids': ids}], 'usage': {'completion_tokens': len(ids)}}

    with backend(reply) as (url, chats), guard(url, tmp_path, budget, monkeypatch) as endpoint:
        messages = [{'role': 'user', 'content': 'Produce the artifact.'}]
        tools = [{'type': 'function', 'function': {'name': 'write', 'parameters': {'required': ['path', 'content']}}}]
        for _ in range(3):
            result = post(endpoint, {'model': 'test', 'messages': messages, 'tools': tools, 'max_tokens': 16384})
            choice = result['choices'][0]
            assert choice['finish_reason'] == 'length'
            assert not choice['message'].get('tool_calls')
            messages += [choice['message'], {'role': 'user', 'content': B.CONTINUE}]
        terminal_payload = {'model': 'test', 'messages': messages, 'tools': tools, 'max_tokens': 16384}
        terminal = post(endpoint, terminal_payload)
        assert terminal['choices'][0]['finish_reason'] == 'stop'
        assert terminal['choices'][0]['message']['content'] == B.CAP_MARK
        assert post(endpoint, terminal_payload) == terminal
        assert [p['max_tokens'] for p in chats] == [100, 90, 10]
    events = [json.loads(line) for line in (tmp_path / 'response-budget-wire.jsonl').read_text().splitlines()]
    assert [r['used_after'] for r in events if 'turn' in r] == [100, 190, 200]
    assert events[-1]['synthetic'] and events[-1]['sft_loss_weight'] == 0
    assert events[-1]['native_generated_tokens'] == 200


def test_missing_raw_accounting_is_infra_failure_without_repeated_inference(tmp_path, monkeypatch):
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 100, 'total_response_tokens': 200}

    def reply(payload, index):
        return {'choices': [{'message': {'role': 'assistant', 'content': '42'}, 'finish_reason': 'stop'}],
                'usage': {'completion_tokens': 100}}

    with backend(reply) as (url, chats), guard(url, tmp_path, budget, monkeypatch) as endpoint:
        for text in ('First request', 'Transport retry with changed context'):
            try:
                post(endpoint, {'model': 'test', 'messages': [{'role': 'user', 'content': text}]})
                raise AssertionError('Unmeasurable inference was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 502
                assert json.loads(error.read())['error']['type'] == 'response_budget_accounting_error'
        assert len(chats) == 1


def test_real_pi_finishes_last_valid_tool_then_submits_workspace_at_total_cap(runtime, tmp_path, monkeypatch):
    node, bundle = runtime
    artifact = tmp_path / 'submission.txt'
    reserve = 100
    final = None
    for n in range(reserve * 2):
        message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'save-final', 'type': 'function',
            'function': {'name': 'write', 'arguments': {'path': str(artifact), 'content': 'saved' + ' x' * n}}}]}
        if B.length(message) == reserve:
            final = message
            break
    assert final is not None, 'Cannot construct the exact-budget fixture'
    budget = {'warn_after_response_tokens': 10000, 'finish_response_tokens': 1000,
              'total_response_tokens': 11000, 'token_counting': 'backend-native'}
    decoded = 0

    def reply(payload, index):
        nonlocal decoded
        cap = payload['max_tokens']
        if decoded < budget['total_response_tokens'] - reserve:
            result = mock_generation({'role': 'assistant', 'reasoning_content': 'Think. ' * 15000, 'content': ''},
                                     'length', cap)
        else:
            assert cap == reserve and decoded + cap == budget['total_response_tokens']
            assert payload.get('tool_choice') != 'none', 'GDPval-style finishing tools were disabled'
            result = mock_generation(final, 'tool_calls', cap)
        decoded += result['usage']['completion_tokens']
        return result

    config = tmp_path / 'agent-config'
    config.mkdir()
    (config / 'settings.json').write_text(json.dumps(settings()))
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    monkeypatch.setenv('PI_RESPONSE_FINISH_THINKING', '0')
    with backend(reply) as (url, chats), guard(url, tmp_path, budget, monkeypatch) as endpoint:
        (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
            'baseUrl': endpoint + '/v1', 'api': 'openai-completions', 'apiKey': 'mock',
            'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'],
                        'contextWindow': 132000, 'maxTokens': 16384,
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
            'compat': {'supportsDeveloperRole': False}}}}))
        result = subprocess.run([str(node), str(bundle / 'cli.js'), '--print', '--mode', 'json',
            '--provider', 'mock', '--model', 'test', '--thinking', 'high', '--no-extensions',
            '--no-skills', '--no-prompt-templates', '--session', str(session), '--', 'Save the requested submission.'],
            cwd=tmp_path, env={**{k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
                              'PI_CODING_AGENT_DIR': str(config), 'NO_PROXY': 'localhost,127.0.0.1'},
            capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        assert len(chats) == 3 and decoded == budget['total_response_tokens']
    assert artifact.read_text() == final['tool_calls'][0]['function']['arguments']['content']
    events = [json.loads(line) for line in session.read_text().splitlines()]
    messages = [r['message'] for r in events if r['type'] == 'message']
    assert messages[-1]['stopReason'] == 'stop'
    assert any(b.get('text') == B.CAP_MARK for b in messages[-1]['content'])
    assert sum(m['role'] == 'toolResult' and m.get('toolName') == 'write' for m in messages) == 1
    assert not any(e['type'] == 'compaction' for e in events)
    assert annotate_sessions(tmp_path)['sessions'][0]['entries'][-1]['sft_loss_weight'] == 0


@contextlib.contextmanager
def flaky_detokenize_backend(fail_first):
    """A backend whose first `fail_first` /detokenize calls answer 500 (#531: a stalled /detokenize under load)."""
    state = {'detok': 0, 'chats': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            status = 200
            if self.path.endswith('/detokenize'):
                state['detok'] += 1
                if state['detok'] <= fail_first:
                    status, result = 500, {'error': 'busy'}
                else:
                    result = {'prompt': B.tokenizer().decode(payload['tokens'], skip_special_tokens=False)}
            else:
                state['chats'] += 1
                ids = B.encode('42')
                result = {'id': 'x', 'created': 1, 'model': 'test', 'choices': [{'index': 0, 'message': {
                    'role': 'assistant', 'content': '42'}, 'finish_reason': 'stop', 'token_ids': ids}],
                    'usage': {'completion_tokens': len(ids)}}
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_detokenize_is_retried_once_before_accounting_fails(tmp_path, monkeypatch):
    """#531: one failed /detokenize no longer ends the session's accounting (11-15 hang root cause)."""
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 100, 'total_response_tokens': 200}
    with flaky_detokenize_backend(1) as (url, state), guard(url, tmp_path, budget, monkeypatch) as endpoint:
        result = post(endpoint, {'model': 'test', 'messages': [{'role': 'user', 'content': 'Answer.'}]})
        assert result['choices'][0]['message']['content'] == '42'
        assert state == {'detok': 2, 'chats': 1}
    log = (tmp_path / 'response-budget-wire.jsonl').read_text()
    assert '"detokenize_retry": 1' in log


def test_accounting_failure_502_is_marked_for_the_sticky_proxy(tmp_path, monkeypatch):
    """#531: after the bounded retry, the guard's own 502 carries X-Response-Budget-Error (the sticky proxy passes it
    to Pi instead of holding it) and a message the worker classifies as infrastructure; it stays sticky."""
    from tb_agents.response_budget_proxy import ACCOUNTING_ERROR, GUARD_ERROR_HEADER
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 100, 'total_response_tokens': 200}
    with flaky_detokenize_backend(10 ** 6) as (url, state), guard(url, tmp_path, budget, monkeypatch) as endpoint:
        for text in ('First request', 'A later turn'):
            try:
                post(endpoint, {'model': 'test', 'messages': [{'role': 'user', 'content': text}]})
                raise AssertionError('accounting failure was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 502 and error.headers.get(GUARD_ERROR_HEADER) == 'accounting'
                body = json.loads(error.read())['error']
                assert body['type'] == 'response_budget_accounting_error'
                assert body['message'].startswith(ACCOUNTING_ERROR)
        assert state == {'detok': 2, 'chats': 1}   # two tries, then sticky: no repeated inference

"""Model clock of inference/sticky_proxy.py, PiCont16kEnvClock and the 'compact' context policy (no GPU)."""
import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextlib.contextmanager
def slow_backend(delay):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "test"}]}')

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            time.sleep(delay)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'index': 0, 'finish_reason': 'stop',
                                                      'message': {'role': 'assistant', 'content': 'ok'}}],
                                         'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def proxy(backend, *extra):
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    p = subprocess.Popen([sys.executable, str(REPO / 'inference/sticky_proxy.py'), '--port', str(port),
                          '--backends', backend, '--turn-max-tokens', '16384', '--turn-continue', '0', *extra],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:
                OPENER.open(url + '/proxy/model-clock', timeout=1)
                break
            except OSError:
                time.sleep(.1)
        yield url
    finally:
        p.terminate()
        p.wait(timeout=10)


def clock(url):
    return json.loads(OPENER.open(url + '/proxy/model-clock', timeout=5).read())


def chat(url):
    req = urllib.request.Request(url + '/v1/chat/completions', headers={'Content-Type': 'application/json'},
                                 data=json.dumps({'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}]}).encode())
    return json.loads(OPENER.open(req, timeout=180).read())


def test_proxy_model_clock_counts_overlapping_requests_once():
    with slow_backend(1.0) as backend, proxy(backend) as url:
        assert clock(url) == {'model_sec': 0.0, 'in_flight': 0, 'requests': 0, 'gate_holds': 0, 'gate_held_sec': 0.0}
        threads = [threading.Thread(target=chat, args=(url,)) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(.5)
        mid = clock(url)
        assert mid['in_flight'] == 2 and mid['model_sec'] > 0.3
        for t in threads:
            t.join()
        c = clock(url)
    assert c['in_flight'] == 0 and c['requests'] == 2
    assert 0.9 < c['model_sec'] < 1.8    # two overlapping 1 s requests count once


class FakeRun:
    """Stands in for Pi.run: sleeps; the fake proxy clock says whether a model request is in flight."""
    def __init__(self, seconds):
        self.seconds = seconds

    async def __call__(self, *args, **kwargs):
        await asyncio.sleep(self.seconds)


def envclock_agent(tmp_path, monkeypatch, timeout_sec, model_busy):
    from tb_agents.harness_agents import PiCont16kEnvClock
    (tmp_path / 'environment').mkdir()
    (tmp_path / 'task.toml').write_text(f'version = "1.0"\n[agent]\ntimeout_sec = {timeout_sec}\n')
    monkeypatch.setenv('PI_MODEL_CLOCK_URL', 'http://unused')
    agent = PiCont16kEnvClock.__new__(PiCont16kEnvClock)
    agent.logs_dir = tmp_path / 'logs'
    t0 = time.monotonic()

    async def model_sec(url):
        return {'model_sec': (time.monotonic() - t0) if model_busy else 0.0,
                'in_flight': int(model_busy), 'requests': 1}

    agent._model_sec = model_sec
    return agent, SimpleNamespace(environment_dir=tmp_path / 'environment')


def test_envclock_spends_budget_on_environment_time(tmp_path, monkeypatch):
    from harbor.trial.errors import AgentTimeoutError
    from tb_agents.harness_agents import PiCont16kCached
    agent, env = envclock_agent(tmp_path, monkeypatch, 3, model_busy=False)
    with patch.object(PiCont16kCached, 'run', FakeRun(60)):
        t0 = time.monotonic()
        with pytest.raises(AgentTimeoutError):
            asyncio.run(agent.run('task', env, None))
    assert 3 <= time.monotonic() - t0 < 15
    record = json.loads((tmp_path / 'logs/env_clock.json').read_text())
    assert record['state'] == 'env_budget_spent' and record['env_sec'] >= 3


def test_envclock_does_not_charge_model_time(tmp_path, monkeypatch):
    from tb_agents.harness_agents import PiCont16kCached
    agent, env = envclock_agent(tmp_path, monkeypatch, 3, model_busy=True)
    with patch.object(PiCont16kCached, 'run', FakeRun(12)):
        asyncio.run(agent.run('task', env, None))
    record = json.loads((tmp_path / 'logs/env_clock.json').read_text())
    assert record['state'] == 'finished' and record['wall_sec'] >= 12 and record['env_sec'] < 3


def test_compact_policy_keeps_pi_compaction(monkeypatch):
    from harbor.agents.installed.pi import Pi
    from tb_agents.harness_agents import PiCont16kCached
    monkeypatch.setenv('PI_CONTEXT_POLICY', 'compact')
    agent = PiCont16kCached.__new__(PiCont16kCached)
    agent._upload_config_text = AsyncMock()
    with patch.object(Pi, '_write_custom_models_json', new_callable=AsyncMock):
        asyncio.run(agent._write_custom_models_json(None, {}))
    written = json.loads(agent._upload_config_text.call_args.kwargs['content'])
    assert 'compaction' not in written and written['httpIdleTimeoutMs'] == 1800000


def test_kv_gate_holds_requests_until_kv_drops():
    kv = {'value': 0.9}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (f'vllm:kv_cache_usage_perc{{engine="0"}} {kv["value"]}\n' if self.path == '/metrics'
                    else '{"data": [{"id": "test"}]}').encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'index': 0, 'finish_reason': 'stop',
                                                      'message': {'role': 'assistant', 'content': 'ok'}}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with proxy(f'http://127.0.0.1:{server.server_port}', '--kv-gate', '0.8,0.7') as url:
            done = threading.Event()
            t = threading.Thread(target=lambda: (chat(url), done.set()))
            t.start()
            assert not done.wait(8)            # KV 0.9 >= 0.8: held
            assert clock(url)['in_flight'] == 1
            kv['value'] = 0.75                 # between LOW and HIGH: still closed
            assert not done.wait(12)
            kv['value'] = 0.5                  # <= LOW: released (after <= 10 s jitter)
            assert done.wait(70)
            c = clock(url)
            assert c['gate_holds'] == 1 and c['gate_held_sec'] >= 20
            t.join()
            # open gate: no hold
            t0 = time.monotonic()
            chat(url)
            assert time.monotonic() - t0 < 5 and clock(url)['gate_holds'] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_kv_gate_file_and_staleness(tmp_path):
    gate_file = tmp_path / 'kv.json'
    gate_file.write_text(json.dumps({'t': time.time(), 'kv': 0.95}))
    with slow_backend(0.0) as backend, proxy(backend, '--kv-gate', '0.8,0.7', '--kv-gate-file', str(gate_file)) as url:
        done = threading.Event()
        t = threading.Thread(target=lambda: (chat(url), done.set()))
        t.start()
        assert not done.wait(8)                        # file says 0.95: held
        gate_file.write_text(json.dumps({'t': time.time(), 'kv': 0.5}))
        assert done.wait(70)                           # released (<= 30 s jitter)
        t.join()
        gate_file.write_text(json.dumps({'t': time.time() - 600, 'kv': 0.99}))   # stale reading: open
        time.sleep(11)                                 # past the 10 s cache
        t0 = time.monotonic()
        chat(url)
        assert time.monotonic() - t0 < 5


def test_context_clamp_skips_tokenize_only_when_provably_safe():
    """Append-only requests far from the window skip /tokenize; near the window the exact count clamps."""
    calls = {'tokenize': 0, 'count': 1000}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "test"}]}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                calls['tokenize'] += 1
                out = {'count': calls['count']}
            else:
                seen.append(payload)
                out = {'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
                       'usage': {'prompt_tokens': calls['count'], 'completion_tokens': 10}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def send(url, messages):
        req = urllib.request.Request(url + '/v1/chat/completions', headers={'Content-Type': 'application/json'},
                                     data=json.dumps({'model': 'test', 'messages': messages, 'max_tokens': 16384}).encode())
        return json.loads(OPENER.open(req, timeout=30).read())

    try:
        with proxy(f'http://127.0.0.1:{server.server_port}', '--context-window', '132000') as url:
            msgs = [{'role': 'user', 'content': 'task'}]
            send(url, msgs)                                    # first request: no bound, exact count
            assert calls['tokenize'] == 1
            msgs = msgs + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'x' * 1000}]
            send(url, msgs)                                    # append-only, far from the window: skipped
            assert calls['tokenize'] == 1 and seen[-1]['max_tokens'] == 16384
            calls['count'] = 120000                            # near the window: bound fails -> exact count clamps
            msgs = msgs + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'y' * 120000}]
            send(url, msgs)
            assert calls['tokenize'] == 2 and seen[-1]['max_tokens'] == 12000
            send(url, [{'role': 'user', 'content': 'rewritten history'}])   # not append-only: exact count
            assert calls['tokenize'] == 3
    finally:
        server.shutdown()
        server.server_close()


def test_real_pi_ends_episode_on_context_overflow(tmp_path):
    """A prompt beyond the proxy's window: 400 with vLLM-like wording; Pi ends the episode (no compaction)."""
    from tb_agents.test_pi_reasoning_history import CACHE, BUNDLE
    import tarfile
    if not (CACHE / BUNDLE).is_file():
        pytest.skip('patched Pi bundle not available')
    root = tmp_path / 'pi'
    with tarfile.open(CACHE / BUNDLE) as archive:
        archive.extractall(root, members=[m for m in archive if m.isfile() or m.isdir()], filter='data')
    node = root / 'opt/harbor-pi/node/bin/node'
    cli = root / 'opt/harbor-pi/pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
    state = {'summarized': False, 'chat': []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "test"}]}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            text = json.dumps(payload.get('messages'))
            if self.path == '/tokenize':
                big = 'Thinking exceeded' in text and not state['summarized'] and 'summar' not in text.lower()
                body = {'count': 140000 if big else 2000}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
                return
            state['chat'].append(payload)
            summarizing = 'summar' in text.lower() and 'Thinking exceeded' in text and len(state['chat']) > 1
            if summarizing:
                state['summarized'] = True
                delta, finish, usage = {'content': 'Summary of the work so far.'}, 'stop', {'prompt_tokens': 2000, 'completion_tokens': 8}
            elif len(state['chat']) == 1:
                delta, finish, usage = {'reasoning_content': 'r' * 400}, 'length', {'prompt_tokens': 1000, 'completion_tokens': 16384}
            else:
                delta, finish, usage = {'content': 'done'}, 'stop', {'prompt_tokens': 2000, 'completion_tokens': 2}
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                          {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                          {'choices': [], 'usage': usage}]:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / 'home'
    config = home / '.pi/agent'
    config.mkdir(parents=True)
    (config / 'settings.json').write_text(json.dumps({'httpIdleTimeoutMs': 60000}))
    session = tmp_path / 'sessions/s.jsonl'
    session.parent.mkdir(parents=True)
    try:
        # The proxy's window (20000) is below Pi's (132000): Pi does not compact on its own estimate, so only
        # the proxy's overflow error can trigger the compaction.
        with proxy(f'http://127.0.0.1:{server.server_port}', '--context-window', '20000') as url:
            (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
                'baseUrl': url + '/v1', 'api': 'openai-completions', 'apiKey': 'mock',
                'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'],
                            'contextWindow': 132000, 'maxTokens': 16384,
                            'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
                'compat': {'supportsDeveloperRole': False}}}}))
            import os as _os
            p = subprocess.run([str(node), str(cli), '--print', '--mode', 'json', '--provider', 'mock', '--model', 'test',
                                '--thinking', 'high', '--no-extensions', '--no-skills', '--no-prompt-templates',
                                '--session', str(session), '--', 'Solve this mock task.'],
                               env={**{k: v for k, v in _os.environ.items() if not k.lower().endswith('_proxy')},
                                    'HOME': str(home), 'PI_CODING_AGENT_DIR': str(config), 'NO_PROXY': '127.0.0.1'},
                               cwd=tmp_path, capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
        server.server_close()
    events = [json.loads(line) for line in session.read_text().splitlines()]
    # Pi 1.2.0 drops an error turn from context (context_edit, replacement null) and then does not treat it as
    # recoverable: the episode ends, exactly as with vLLM's own overflow 400 in campaign 20261015T215325Z
    # (4 teacher eval sessions end that way). The proxy's wording therefore changes nothing; it only mirrors vLLM's.
    assert not any(e['type'] == 'compaction' for e in events)
    last = [e['message'] for e in events if e['type'] == 'message' and e['message']['role'] == 'assistant'][-1]
    assert last['stopReason'] == 'error' and 'maximum context length is 20000 tokens' in last['errorMessage']
    assert events[-1]['type'] == 'context_edit' and events[-1]['replacement'] is None


def test_context_guard_skips_tokenize_when_bound_leaves_a_turn(tmp_path, monkeypatch):
    from tb_agents.pi_context_limit import CAP_MARK, start_context_guard
    calls = {'tokenize': 0, 'count': 1000, 'prompt_usage': 1000}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                calls['tokenize'] += 1
                body, ctype = json.dumps({'count': calls['count']}).encode(), 'application/json'
            else:
                seen.append(payload)
                chunks = [{'choices': [{'index': 0, 'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]},
                          {'choices': [], 'usage': {'prompt_tokens': calls['prompt_usage'], 'completion_tokens': 5}}]
                body = (''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n').encode()
                ctype = 'text/event-stream'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for k in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('PI_GUARD_BOUND_SHORTCUT', '1')
    guard, url = start_context_guard(f'http://127.0.0.1:{server.server_port}/v1', tmp_path)

    def send(messages):
        req = urllib.request.Request(url + '/v1/chat/completions', headers={'Content-Type': 'application/json'},
                                     data=json.dumps({'model': 't', 'messages': messages, 'stream': True,
                                                      'max_tokens': 16384}).encode())
        return OPENER.open(req, timeout=30).read().decode()

    try:
        msgs = [{'role': 'user', 'content': 'task'}]
        send(msgs)
        assert calls['tokenize'] == 1                     # no previous request: exact count
        msgs += [{'role': 'assistant', 'content': 'ok'}, {'role': 'tool', 'content': 'x' * 2000}]
        send(msgs)
        assert calls['tokenize'] == 1 and seen[-1]['max_tokens'] == 16384   # bound ~3.2k: skipped
        calls['prompt_usage'] = 120000                     # the server reports a long prompt for this turn
        msgs += [{'role': 'assistant', 'content': 'ok'}, {'role': 'tool', 'content': 'y'}]
        send(msgs)
        assert calls['tokenize'] == 1                      # bound used the previous usage (3.2k): skipped
        calls['count'] = 125000
        msgs += [{'role': 'assistant', 'content': 'ok'}, {'role': 'tool', 'content': 'z'}]
        send(msgs)                                         # bound 120k + a turn > 132k: exact count, clamp
        assert calls['tokenize'] == 2 and seen[-1]['max_tokens'] == 7000
        calls['count'] = 132000
        msgs += [{'role': 'assistant', 'content': 'ok'}, {'role': 'tool', 'content': 'w'}]
        out = send(msgs)
        assert calls['tokenize'] == 3 and CAP_MARK in out  # at the window: submitted, not generated
    finally:
        guard.terminate()
        guard.wait(timeout=10)
        server.shutdown()
        server.server_close()


def test_real_pi_submits_when_its_clamp_leaves_one_token(tmp_path, monkeypatch):
    """Usage near the window makes Pi ask for max_tokens=1; with --submit-at-pi-clamp the guard submits at once
    instead of looping on one-token stubs until the exact count reaches the window."""
    from tb_agents.pi_context_limit import CAP_MARK, start_context_guard
    from tb_agents.test_pi_reasoning_history import CACHE, BUNDLE
    import tarfile
    if not (CACHE / BUNDLE).is_file():
        pytest.skip('patched Pi bundle not available')
    root = tmp_path / 'pi'
    with tarfile.open(CACHE / BUNDLE) as archive:
        archive.extractall(root, members=[m for m in archive if m.isfile() or m.isdir()], filter='data')
    node = root / 'opt/harbor-pi/node/bin/node'
    cli = root / 'opt/harbor-pi/pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
    chats, toks = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                toks.append(payload)
                body, ctype = json.dumps({'count': 120000}).encode(), 'application/json'   # true count < window
            else:
                chats.append(payload)
                cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
                chunks = [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'r' * min(cap, 400)}, 'finish_reason': None}]},
                          {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
                          # the server reports a prompt that leaves Pi (window - 4096 safety) no room
                          {'choices': [], 'usage': {'prompt_tokens': 129000, 'completion_tokens': cap}}]
                body = (''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n').encode()
                ctype = 'text/event-stream'
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for k in ('PI_CONTEXT_WINDOW', 'PI_CONTEXT', 'PI_CONTEXT_PROXY', 'MAX_LEN'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('PI_GUARD_SUBMIT_AT_PI_CLAMP', '1')
    guard, url = start_context_guard(f'http://127.0.0.1:{server.server_port}/v1', tmp_path)
    home = tmp_path / 'home'
    config = home / '.pi/agent'
    config.mkdir(parents=True)
    (config / 'settings.json').write_text(json.dumps({'httpIdleTimeoutMs': 60000, 'compaction': {'enabled': False}}))
    (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
        'baseUrl': url + '/v1', 'api': 'openai-completions', 'apiKey': 'mock',
        'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'], 'contextWindow': 132000,
                    'maxTokens': 16384, 'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
        'compat': {'supportsDeveloperRole': False}}}}))
    session = tmp_path / 'sessions/s.jsonl'
    session.parent.mkdir(parents=True)
    import os as _os
    try:
        p = subprocess.run([str(node), str(cli), '--print', '--mode', 'json', '--provider', 'mock', '--model', 'test',
                            '--thinking', 'high', '--no-extensions', '--no-skills', '--no-prompt-templates',
                            '--session', str(session), '--', 'Solve this mock task.'],
                           env={**{k: v for k, v in _os.environ.items() if not k.lower().endswith('_proxy')},
                                'HOME': str(home), 'PI_CODING_AGENT_DIR': str(config), 'NO_PROXY': '127.0.0.1'},
                           cwd=tmp_path, capture_output=True, text=True, timeout=120)
    finally:
        guard.terminate()
        guard.wait(timeout=10)
        server.shutdown()
        server.server_close()
    assert p.returncode == 0, p.stderr[-2000:]
    assert len(chats) == 1                       # turn 1 generated; turn 2 (Pi asks for 1 token) was submitted
    events = [json.loads(line) for line in session.read_text().splitlines()]
    last = [e['message'] for e in events if e['type'] == 'message' and e['message']['role'] == 'assistant'][-1]
    assert last['stopReason'] == 'stop' and any(b.get('text') == CAP_MARK for b in last['content'])
    log = [json.loads(line) for line in (tmp_path / 'context-limit.jsonl').read_text().splitlines()]
    assert log[-1]['termination_reason'] == 'context_limit' and log[-1]['context_limit_source'] == 'pi_output_budget'
    assert log[-1]['requested_output_budget'] == 1

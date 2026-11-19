"""Exercise real proxy HTTP/SSE behavior without spending model tokens."""
import asyncio
import json
from pathlib import Path
import socket
import sys

import aiohttp
from aiohttp import web
import pytest

from sticky_proxy import sse, clamp_max_tokens, assistant_from_response
from synth.wire import load as load_wire


def test_readiness_accepts_a_busy_backends_delayed_models_response():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from types import SimpleNamespace
    import time
    from gen.pi_cont16k import wait_for_proxy

    class DelayedModels(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == '/v1/models'
            time.sleep(2.2)  # Production /models took 2.4s: the old 2s probe always failed.
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), DelayedModels)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wait_for_proxy(SimpleNamespace(poll=lambda: None),
                       f'http://127.0.0.1:{server.server_port}/v1', 'delayed-backend', 20)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_sse_unicode_line_separators_preserve_content_and_reasoning():
    delta = {'content': 'NEL\x85LS\u2028PS\u2029end', 'reasoning_content': 'thought\x85next'}
    body = ('data: ' + json.dumps({'choices': [{'delta': delta}]}, ensure_ascii=False)
            + '\n\ndata: [DONE]\n\n').encode()
    message = assistant_from_response(body, stream=True, anthropic=False)
    assert message['content'] == delta['content']
    assert message['reasoning_content'] == delta['reasoning_content']


async def exercise(tmp_path, stops, stream, *, anthropic=False, retries=2,
                   disable_load_polling=False, no_poll=False, rank=None, no_load_poll=False):
    requests = []
    metric_requests = []
    expected_rank = str(rank) if rank is not None else None

    async def generate(req):
        assert req.headers.get('X-data-parallel-rank') == expected_rank
        payload = await req.json()
        requests.append(payload)
        index = len(requests) - 1
        reason = stops[index]
        if anthropic:
            result = {'type': 'message', 'content': [{'type': 'text', 'text': f'sample-{index}'}],
                      'stop_reason': reason, 'usage': {'output_tokens': 512}}
            return web.json_response(result)
        message = {'role': 'assistant', 'content': f'sample-{index}', 'reasoning_content': 'thought'}
        usage = {'prompt_tokens': 100, 'completion_tokens': 512}
        if stream:
            body = sse({'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]})
            body += sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}]})
            body += sse({'choices': [], 'usage': usage}) + b'data: [DONE]\n\n'
            return web.Response(body=body, content_type='text/event-stream')
        return web.json_response({'choices': [{'index': 0, 'message': message, 'finish_reason': reason}],
                                  'usage': usage})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_post('/v1/messages', generate)
    async def metrics(req):
        metric_requests.append(req.path)
        return web.Response(text='vllm:kv_cache_usage_perc{engine="0"} 0.25\n')
    app.router.add_get('/metrics', metrics)
    async def models(req):
        return web.json_response({'data': [{'id': 'test'}]})
    app.router.add_get('/v1/models', models)
    runner = web.AppRunner(app)
    await runner.setup()
    backend = socket.socket()
    backend.bind(('127.0.0.1', 0))
    backend.listen(128)
    site = web.SockSite(runner, backend)
    await site.start()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    log = tmp_path / 'turns.jsonl'
    wire = tmp_path / 'logs' / 'wire.jsonl'
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
        '--port', str(port), '--listen-fd', str(listener.fileno()),
        '--backends', f'http://127.0.0.1:{backend.getsockname()[1]}',
        '--turn-max-tokens', '512', '--turn-continue', '0',
        *([] if retries is None else ['--turn-retries', str(retries)]),
        '--hold', '0', '--turn-log', str(log), '--shutdown-timeout', '0',
        '--wire-log', str(wire),
        *(['--disable-load-polling'] if disable_load_polling else []),
        *(['--single-backend-no-poll'] if no_poll else []),
        *(['--no-load-poll'] if no_load_poll else []),
        *(['--backend-headers', json.dumps({'X-data-parallel-rank': str(rank)})] if rank is not None else []),
        pass_fds=(listener.fileno(),), stdout=asyncio.subprocess.DEVNULL)
    listener.close()
    ready = '/v1/models' if disable_load_polling else '/metrics'
    try:
        async with aiohttp.ClientSession() as client:
            for _ in range(100):
                try:
                    async with client.get(f'http://127.0.0.1:{port}{ready}') as r:
                        await r.read()
                    break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(.05)
            payload = {'messages': [{'role': 'user', 'content': 'solve'}], 'seed': 42,
                       'stream': stream, 'max_tokens': 65536, 'model': 'test'}
            if not anthropic:
                payload['max_completion_tokens'] = 32768
            path = '/v1/messages' if anthropic else '/v1/chat/completions'
            async with client.post(f'http://127.0.0.1:{port}{path}', json=payload) as r:
                assert r.status == 200
                output = await r.text()
            if disable_load_polling:
                assert metric_requests == [], 'a single backend needs no background routing polls'
                async with client.get(f'http://127.0.0.1:{port}/metrics') as r:
                    assert '0.25' in await r.text(), 'explicit metrics must still reach the backend'
                assert metric_requests == ['/metrics']
            elif no_poll or no_load_poll:
                # Exactly the explicit readiness GET /metrics above. Startup must
                # not also poll a backend that has no alternative routing choice.
                assert len(metric_requests) == 1, 'Only the explicit readiness GET should fetch metrics'
            else:
                assert metric_requests, 'default background polling must remain enabled'
        assert len(requests) == len(stops)
        assert [r['seed'] for r in requests] == list(range(42, 42 + len(stops)))
        assert all(r['max_tokens'] == 512 for r in requests)
        if not anthropic:
            assert all(r['max_completion_tokens'] == 512 for r in requests)
        assert all(r['messages'] == payload['messages'] for r in requests), 'proxy must not continue privately'
        calls = load_wire(wire)
        assert len(calls) == len(stops), 'one wire record per backend request'
        assert all(c['complete'] and c['status'] == 200 for c in calls)
        assert [c['body'] for c in calls] == requests, 'the wire log holds exactly what the backend received'
        assert 'sample-0' in bytes(calls[0]['response']).decode()
        if not anthropic:
            assert 'thought' in bytes(calls[0]['response']).decode()
        assert f'sample-{len(stops)-1}' in output
        assert all(f'sample-{i}' not in output for i in range(len(stops)-1))
        if any(s in ('length', 'max_tokens') for s in stops):
            record = json.loads(log.read_text())
            assert record['attempts'] == len(stops)
            assert record['continued'] == 0
            assert len(record['sample_usage']) == len(stops)
        return output
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), 5)
        await runner.cleanup()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('stops', [['stop'], ['length', 'stop'], ['length', 'length', 'length']])
def test_resample_and_return_only_selected_response(tmp_path, stops, stream):
    asyncio.run(exercise(tmp_path, stops, stream))


def test_anthropic_cap(tmp_path):
    asyncio.run(exercise(tmp_path, ['max_tokens', 'end_turn'], False, anthropic=True))


@pytest.mark.parametrize('stream', [False, True])
def test_default_returns_first_truncation_without_resampling(tmp_path, stream):
    output = asyncio.run(exercise(tmp_path, ['length'], stream, retries=None))
    assert '"length"' in output


def test_lower_client_budget_is_preserved():
    payload = {'max_tokens': 128}
    clamp_max_tokens(payload, 16384, False)
    assert payload['max_tokens'] == 128


@pytest.mark.parametrize('stream', [False, True])
def test_single_backend_can_disable_routing_polls_and_keep_live_metrics(tmp_path, stream):
    asyncio.run(exercise(tmp_path, ['length'], stream, retries=0, disable_load_polling=True))


def test_disabling_routing_polls_rejects_multiple_backends():
    import subprocess
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
        '--port', '0', '--backends', 'http://127.0.0.1:1,http://127.0.0.1:2',
        '--disable-load-polling'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'requires exactly one backend' in result.stderr


@pytest.mark.parametrize('stream', [False, True])
def test_single_backend_skips_background_poll_but_preserves_requests(tmp_path, stream):
    asyncio.run(exercise(tmp_path, ['length'], stream, retries=0, no_poll=True))


def test_disabling_routing_poll_rejects_multiple_backends():
    import subprocess
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
        '--port', '0', '--backends', 'http://one,http://two', '--single-backend-no-poll'],
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'requires exactly one backend' in result.stderr


@pytest.mark.parametrize('stream', [False, True])
def test_explicit_sticky_dp_rank_reaches_backend(tmp_path, stream):
    asyncio.run(exercise(tmp_path, ['length'], stream, retries=0, rank=3))


def test_single_backend_does_not_poll_metrics_when_externally_admitted(tmp_path):
    asyncio.run(exercise(tmp_path, ['stop'], False, no_load_poll=True))

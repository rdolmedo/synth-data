"""inference/sticky_proxy.py --wire-log: one file holds the served request and raw stream."""
import asyncio
import json
from pathlib import Path
import socket
import sys

import aiohttp
from aiohttp import web

from synth.wire import load, sse_events

PROXY = Path(__file__).resolve().parents[2] / 'inference' / 'sticky_proxy.py'


def sse(obj):
    return b'data: ' + json.dumps(obj).encode() + b'\n\n'


async def exercise(tmp_path, fail_first=False):
    seen = []

    async def generate(req):
        payload = await req.json()
        seen.append(payload)
        if fail_first and len(seen) == 1:
            return web.json_response({'error': 'busy'}, status=500)
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await resp.prepare(req)
        await resp.write(sse({'id': 'r1', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'think '}}]}))
        await resp.write(sse({'id': 'r1', 'choices': [{'index': 0, 'delta': {'content': 'done'}, 'finish_reason': 'stop'}]}))
        await resp.write(b'data: [DONE]\n\n')
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_get('/v1/models', lambda req: web.json_response({'data': [{'id': 'm'}]}))
    runner = web.AppRunner(app)
    await runner.setup()
    backend = socket.socket()
    backend.bind(('127.0.0.1', 0))
    backend.listen(64)
    await web.SockSite(runner, backend).start()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(64)
    port = listener.getsockname()[1]
    wire = tmp_path / 'logs' / 'wire.jsonl'
    process = await asyncio.create_subprocess_exec(sys.executable, str(PROXY), '--port', str(port),
        '--listen-fd', str(listener.fileno()), '--backends', f'http://127.0.0.1:{backend.getsockname()[1]}',
        '--turn-max-tokens', '16384', '--turn-retries', '0', '--turn-continue', '0', '--hold', '0',
        '--no-load-poll', '--shutdown-timeout', '0', '--wire-log', str(wire),
        '--request-overrides', json.dumps({'temperature': 1.0, 'seed': 7}),
        pass_fds=(listener.fileno(),), stdout=asyncio.subprocess.DEVNULL)
    listener.close()
    try:
        async with aiohttp.ClientSession() as client:
            for _ in range(100):
                try:
                    async with client.get(f'http://127.0.0.1:{port}/v1/models') as r:
                        await r.read()
                    break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(.05)
            payload = {'model': 'm', 'stream': True, 'max_tokens': 99999,
                       'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'go'}],
                       'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {}}}]}
            async with client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=payload) as r:
                status, text = r.status, await r.text()
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), 5)
        await runner.cleanup()
    return status, text, seen, load(wire)


def test_wire_log_keeps_served_request_and_stream(tmp_path):
    status, text, seen, calls = asyncio.run(exercise(tmp_path))
    assert status == 200 and 'done' in text
    assert len(calls) == 1 and calls[0]['complete'] and calls[0]['status'] == 200
    body = calls[0]['body']
    assert body == seen[0], 'the log holds exactly what the backend received'
    assert body['temperature'] == 1.0 and body['seed'] == 7 and body['max_tokens'] == 16384
    assert body['tools'] and body['messages'][0]['content'] == 'sys'
    deltas = [c['delta'] for e in sse_events(calls[0]['response']) for c in e['choices']]
    assert deltas[0]['reasoning_content'] == 'think ' and deltas[1]['content'] == 'done'


def test_wire_log_records_backend_errors(tmp_path):
    status, _, seen, calls = asyncio.run(exercise(tmp_path, fail_first=True))
    assert status == 500 and len(seen) == 1
    assert calls[0]['status'] == 500 and calls[0]['complete'] and b'busy' in calls[0]['error']

"""#531: the sticky proxy passes the response-budget guard's own 502 (X-Response-Budget-Error) straight to Pi instead
of holding it for --hold seconds (the 11-15 hang); an upstream/server 502 without the header keeps its hold."""
import asyncio
from pathlib import Path
import socket
import sys
import time

import aiohttp
from aiohttp import web

GUARD_HEADER = 'X-Response-Budget-Error'
GUARD = {GUARD_HEADER: 'accounting'}
ERROR = {'error': {'message': 'response budget accounting error: x', 'type': 'response_budget_accounting_error'}}


async def through_proxy(replies, extra, hold='30'):
    """POST one chat request through a sticky proxy whose backend answers `replies` in order (status, headers).
    Returns (status, seconds, backend requests, error body)."""
    seen = []

    async def generate(req):
        await req.read()
        seen.append(time.monotonic())
        status, headers = replies[min(len(seen), len(replies)) - 1]
        if status != 200:
            return web.json_response(ERROR, status=status, headers=headers)
        return web.json_response({'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'},
                                               'finish_reason': 'stop'}]})

    async def models(req):
        return web.json_response({'data': []})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_get('/v1/models', models)
    runner = web.AppRunner(app)
    await runner.setup()
    backend = socket.socket()
    backend.bind(('127.0.0.1', 0))
    backend.listen(128)
    await web.SockSite(runner, backend).start()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
        '--port', str(port), '--listen-fd', str(listener.fileno()),
        '--backends', f'http://127.0.0.1:{backend.getsockname()[1]}', '--hold', hold,
        '--shutdown-timeout', '0', '--no-load-poll', *extra,
        pass_fds=(listener.fileno(),), stdout=asyncio.subprocess.DEVNULL)
    listener.close()
    try:
        async with aiohttp.ClientSession() as client:
            for _ in range(400):   # up to ~20 s (a cold Lustre import); fail fast if the proxy died
                assert process.returncode is None, 'sticky proxy exited at start'
                try:
                    async with client.get(f'http://127.0.0.1:{port}/v1/models') as r:
                        await r.read()
                    break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(.05)
            t0 = time.monotonic()
            async with client.post(f'http://127.0.0.1:{port}/v1/chat/completions',
                                   json={'model': 'm', 'messages': [{'role': 'user', 'content': 'q'}],
                                         'stream': False}) as r:
                body = await r.read()
                return r.status, time.monotonic() - t0, len(seen), body
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), 5)
        await runner.cleanup()


CAPPED = ['--turn-max-tokens', '512', '--turn-retries', '0', '--turn-continue', '0']


def test_guard_502_reaches_the_client_at_once_on_the_capped_turn_path():
    status, sec, n, body = asyncio.run(through_proxy([(502, GUARD), (200, {})], CAPPED))
    assert status == 502 and n == 1 and sec < 5
    assert b'response budget accounting error' in body


def test_guard_502_reaches_the_client_at_once_on_the_plain_path():
    status, sec, n, body = asyncio.run(through_proxy([(502, GUARD), (200, {})], []))
    assert status == 502 and n == 1 and sec < 5


def test_upstream_502_keeps_its_hold():
    """A server 502 (no guard header) is still waited out and retried within --hold (here: answered on try 2)."""
    status, sec, n, body = asyncio.run(through_proxy([(502, {}), (200, {})], CAPPED))
    assert status == 200 and n == 2 and sec >= 9


def test_header_name_matches_the_guard():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import sticky_proxy
    from tb_agents import response_budget_proxy
    assert sticky_proxy.GUARD_ERROR_HEADER == response_budget_proxy.GUARD_ERROR_HEADER == GUARD_HEADER

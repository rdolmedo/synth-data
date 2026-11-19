"""Dedicated rollout proxies retain admission identity across identical model histories."""
import asyncio
from pathlib import Path
import socket
import subprocess
import sys

import aiohttp
from aiohttp import web


def test_rollout_proxy_forwards_distinct_stable_identity():
    async def exercise():
        seen = []

        async def reply(req):
            if req.method == 'POST':
                await req.json()
                seen.append((req.headers.get('X-Conversation-Id'), req.headers.get('X-Gate-Client')))
            return web.json_response({'choices': [{'message': {'role': 'assistant', 'content': 'same'},
                                                   'finish_reason': 'stop'}]})

        app = web.Application()
        app.router.add_route('*', '/{path:.*}', reply)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        backend = site._server.sockets[0].getsockname()[1]
        children = []
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
                for identity in ('repeat-0', 'repeat-1'):
                    with socket.socket() as listener:
                        listener.bind(('127.0.0.1', 0))
                        port = listener.getsockname()[1]
                        child = subprocess.Popen([
                            sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
                            '--port', str(port), '--listen-fd', str(listener.fileno()),
                            '--backends', f'http://127.0.0.1:{backend}',
                            '--conversation-id', identity, '--gate-client', 'tb21',
                            '--shutdown-timeout', '0'], pass_fds=(listener.fileno(),),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    children.append(child)
                    base = f'http://127.0.0.1:{port}'
                    for _ in range(100):
                        try:
                            async with client.get(base + '/v1/models') as response:
                                assert response.status == 200
                            break
                        except aiohttp.ClientConnectionError:
                            assert child.poll() is None
                            await asyncio.sleep(0.02)
                    else:
                        raise AssertionError('rollout proxy did not start')
                    # Even byte-identical answers in two repeats keep separate IDs on all later turns.
                    for messages in ([{'role': 'user', 'content': 'task'}],
                                     [{'role': 'user', 'content': 'task'},
                                      {'role': 'assistant', 'content': 'same'},
                                      {'role': 'user', 'content': 'continue'}]):
                        async with client.post(base + '/v1/chat/completions',
                                               json={'model': 'm', 'messages': messages}) as response:
                            assert response.status == 200
                            await response.read()
            assert seen == [('repeat-0', 'tb21')] * 2 + [('repeat-1', 'tb21')] * 2
        finally:
            for child in children:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            await runner.cleanup()

    asyncio.run(exercise())

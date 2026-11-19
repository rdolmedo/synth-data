"""Agent network isolation with a fixed model relay and separate evaluator networking.

The agent's original namespaces stay offline through teardown, including background processes.
Evaluator commands may use a sibling network namespace while retaining the task's filesystem/PID view.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import aiohttp
from aiohttp import web

HOP = {'host', 'content-length', 'transfer-encoding', 'content-encoding', 'connection'}
ALLOWED = {('GET', '/v1/models'), ('POST', '/v1/chat/completions'),
           ('POST', '/v1/completions'), ('POST', '/tokenize'), ('POST', '/v1/tokenize')}


def capdrop_binary():
    source = Path(__file__).with_name('capdrop.c')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / 'aa-capdrop' / digest
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / 'capdrop'
    with open(directory / 'build.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            partial = directory / 'capdrop.partial'
            subprocess.run(['cc', '-static', '-O2', '-Wall', '-Werror', str(source), '-o', str(partial)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            partial.chmod(0o755)
            partial.replace(binary)
    return binary


def media_is_local(value):
    """Model servers must not fetch a remote media URL on behalf of an offline agent."""
    if isinstance(value, list):
        return all(media_is_local(x) for x in value)
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('image_url', 'video_url', 'audio_url', 'file_url'):
                url = child.get('url') if isinstance(child, dict) else child
                if not isinstance(url, str) or not url.startswith('data:'):
                    return False
            if not media_is_local(child):
                return False
    return True


async def stop_process(proc):
    if proc and proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


def enter_user(pod):
    return ['nsenter', '-t', str(pod.holder_pid), '-U', '--preserve-credentials', '--']


class ModelRelay:
    def __init__(self, box, pod, backend):
        self.box, self.pod = box, pod
        self.backend = backend.rstrip('/').removesuffix('/v1')
        self.proc = self.runner = self.http = None
        self.tmp = None
        self.forwarding = set()

    async def start(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='aa-model-')
        unix = str(Path(self.tmp.name) / 'api.sock')
        self.http = aiohttp.ClientSession(trust_env=False,
                                         timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=7200))

        async def forward(req):
            if (req.method, req.path) not in ALLOWED or req.query_string:
                raise web.HTTPForbidden(text='offline model relay: unsupported route')
            body = await req.read()
            if req.method == 'POST':
                try:
                    payload = json.loads(body)
                except ValueError:
                    raise web.HTTPBadRequest(text='invalid model request JSON')
                if not media_is_local(payload):
                    raise web.HTTPForbidden(text='offline model relay: remote media URLs are disabled')
            headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
            async with self.http.request(req.method, self.backend + req.path, data=body, headers=headers,
                                         allow_redirects=False) as upstream:
                out = web.StreamResponse(status=upstream.status,
                                         headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP})
                await out.prepare(req)
                async for chunk in upstream.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                return out

        async def handle(req):
            task = asyncio.current_task()
            self.forwarding.add(task)
            try:
                return await forward(req)
            finally:
                self.forwarding.discard(task)

        app = web.Application(client_max_size=128 * 1024 * 1024)
        app.router.add_route('*', '/{path:.*}', handle)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=3)
        await self.runner.setup()
        await web.UnixSite(self.runner, unix).start()
        self.proc = await asyncio.create_subprocess_exec(
            *enter_user(self.pod), 'nsenter', '-t', str(self.box.init_pid), '-n', '--',
            sys.executable, '-I', str(Path(__file__).resolve()), 'relay', '--unix', unix,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        line = await asyncio.wait_for(self.proc.stdout.readline(), 15)
        if not line:
            error = (await self.proc.stderr.read()).decode(errors='replace')
            raise RuntimeError('offline model relay failed: ' + error[-1000:])
        self.url = f"http://127.0.0.1:{json.loads(line)['port']}/v1"
        return self.url

    async def close(self):
        await stop_process(self.proc)
        tasks = list(self.forwarding)
        for task in tasks:
            task.cancel()
        if self.http:
            await self.http.close()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        if self.runner:
            await asyncio.wait_for(self.runner.cleanup(), 5)
        if self.tmp:
            self.tmp.cleanup()


class OfflineNetworks:
    def __init__(self, environment):
        self.environment = environment
        self.agent_active = True
        self.evaluator_internet = False
        self.helpers = {}
        self.lock = asyncio.Lock()

    async def isolate(self):
        for service in self.environment._services.values():
            box = service.box
            if box is not None:
                if not box.private_net or box.proc_mode != 'private':
                    raise RuntimeError('offline agents require private network and PID/proc namespaces')
                await box.set_internet(False)

    async def evaluator_network(self, box):
        async with self.lock:
            return await self._evaluator_network(box)

    async def _evaluator_network(self, box):
        from harbor_ns.sandbox import Sandbox
        if self.agent_active or not self.evaluator_internet:
            return None
        if box.init_pid in self.helpers:
            return self.helpers[box.init_pid]['box'].init_pid
        env = self.environment
        directory = Path(tempfile.mkdtemp(prefix='evaluator-net-', dir=env._pod_dir))
        proc = await asyncio.create_subprocess_exec(
            *enter_user(env._pod), 'unshare', '--net', 'sh', '-c',
            'ip link set lo up && sysctl -qw net.ipv4.ip_forward=0 && exec sleep infinity',
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=open(directory / 'namespace.log', 'wb'), start_new_session=True,
            env={**os.environ, 'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'})
        helper = Sandbox(cfg=box.cfg, lower=[], pod=env._pod)
        helper.init_pid, helper._dir = proc.pid, directory
        helper._unshare = proc
        record = {'box': helper, 'relay': None}
        self.helpers[box.init_pid] = record
        for _ in range(500):
            if proc.returncode is not None:
                raise RuntimeError('evaluator network failed: ' + (directory / 'namespace.log').read_text())
            try:
                if os.readlink(f'/proc/{proc.pid}/ns/net') not in (
                        os.readlink('/proc/self/ns/net'), os.readlink(f'/proc/{env._pod.holder_pid}/ns/net')):
                    break
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError('evaluator network namespace did not start')
        if env._pod.bridge:
            await env._pod.attach(helper, env._pod.ip(len(env._services) + len(self.helpers)))
        await helper.set_internet(True)
        # Preserve localhost task servers for grading; the forwarding direction never gives the agent egress.
        record['relay'] = await asyncio.create_subprocess_exec(
            *enter_user(env._pod), sys.executable, '-I', str(Path(__file__).with_name('netns_relay.py')),
            str(helper.init_pid), str(box.init_pid), stdin=subprocess.DEVNULL,
            stdout=open(directory / 'localhost.log', 'wb'), stderr=subprocess.STDOUT, start_new_session=True)
        await asyncio.sleep(0.2)
        if record['relay'].returncode is not None:
            raise RuntimeError('evaluator localhost relay failed')
        return helper.init_pid

    async def close(self):
        for record in self.helpers.values():
            await stop_process(record['relay'])
            helper = record['box']
            await stop_process(helper._slirp)
            await stop_process(helper._unshare)
        self.helpers.clear()


async def local_relay(unix):
    async def pipe(reader, writer):
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    async def handle(reader, writer):
        other_reader, other_writer = await asyncio.open_unix_connection(unix)
        await asyncio.gather(pipe(reader, other_writer), pipe(other_reader, writer), return_exceptions=True)

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    print(json.dumps({'port': server.sockets[0].getsockname()[1]}), flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['relay'])
    parser.add_argument('--unix', required=True)
    args = parser.parse_args()
    asyncio.run(local_relay(args.unix))

"""Offline Pi protocol tests; fake HTTP backends only, no model inference.

Run unit checks anywhere. Set AA_OFFLINE_NS_TESTS=1 on a CPU node to require real
user/PID/mount/network namespaces and slirp, rather than skip that integration check.
"""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest

from aa_evals.common.harbor_mswea import offline


@pytest.mark.parametrize('payload', [
    {'messages': [{'content': 'https://example.invalid is just text'}]},
    {'messages': [{'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]}]},
    {'input_audio': {'data': 'eA==', 'format': 'wav'}},
])
def test_embedded_media_and_plain_text_allowed(payload):
    assert offline.media_is_local(payload)


@pytest.mark.parametrize('key', ['image_url', 'video_url', 'audio_url', 'file_url'])
@pytest.mark.parametrize('value', ['https://example.invalid/a', 'http://127.0.0.1/a', 'file:///etc/passwd'])
def test_remote_media_rejected_recursively(key, value):
    assert not offline.media_is_local({'messages': [{'content': [{key: {'url': value}}]}]})


def test_relay_fixed_routes_media_and_no_proxy(monkeypatch):
    async def go():
        seen = []
        async def backend(request):
            seen.append((request.method, request.path, await request.read()))
            return web.json_response({'fake_model': True})
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', backend)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        # The socket forwarding subprocess is tested separately in actual namespaces.
        async def fake_process(*args, **kwargs):
            stdout = asyncio.StreamReader()
            stdout.feed_data(b'{"port": 1}\n')
            stdout.feed_eof()
            return SimpleNamespace(stdout=stdout, returncode=0)
        monkeypatch.setattr(offline.asyncio, 'create_subprocess_exec', fake_process)
        monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
        monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
        relay = offline.ModelRelay(SimpleNamespace(init_pid=1), SimpleNamespace(holder_pid=1),
                                   f'http://127.0.0.1:{port}/v1')
        try:
            await relay.start()
            connector = aiohttp.UnixConnector(path=str(Path(relay.tmp.name) / 'api.sock'))
            async with aiohttp.ClientSession(connector=connector) as client:
                for method, path in [('GET', '/v1/models'), ('POST', '/v1/chat/completions'),
                                     ('POST', '/v1/completions'), ('POST', '/tokenize'), ('POST', '/v1/tokenize')]:
                    async with client.request(method, 'http://relay' + path, json={}) as response:
                        assert response.status == 200
                before = len(seen)
                for method, path in [('GET', '/egress'), ('CONNECT', '/v1/models'), ('GET', '/v1/models?url=x'),
                                     ('POST', '/v1/models'), ('GET', '/v1/chat/completions')]:
                    async with client.request(method, 'http://relay' + path, json={}, allow_redirects=False) as response:
                        assert response.status in (400, 403, 404)
                for key in ('image_url', 'video_url', 'audio_url', 'file_url'):
                    async with client.post('http://relay/v1/chat/completions',
                                           json={'messages': [{'content': [{key: {'url': 'https://example.invalid'}}]}]}) as response:
                        assert response.status == 403
                async with client.post('http://relay/v1/chat/completions', data='not-json') as response:
                    assert response.status == 400
                assert len(seen) == before
                async with client.post('http://relay/v1/chat/completions',
                                       json={'messages': [{'content': [{'image_url': {'url': 'data:image/png;base64,eA=='}}]}]}) as response:
                    assert response.status == 200
        finally:
            await relay.close()
            await runner.cleanup()
        assert relay.http.closed
        assert not Path(relay.tmp.name).exists()
    asyncio.run(go())


def _require_real_namespaces():
    command = ['unshare', '--user', '--map-root-user', '--net', '--mount', '--pid', '--fork', '--mount-proc', 'true']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        if os.environ.get('AA_OFFLINE_NS_TESTS') == '1':
            pytest.fail('real namespace test was required: ' + result.stderr)
        pytest.skip('private /proc needs a compute node: ' + result.stderr.strip())


async def _exec(argv, code, timeout=20):
    process = await asyncio.create_subprocess_exec(*argv, sys.executable, '-I', '-c', code,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                 start_new_session=True)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        await offline.stop_process(process)
        raise
    assert process.returncode == 0, (out.decode(), err.decode())
    return json.loads(out)


async def _bare_box(pod, directory):
    from harbor_ns.sandbox import Sandbox, RuntimeConfig, _child_pid
    directory.mkdir()
    process = await asyncio.create_subprocess_exec(
        *offline.enter_user(pod), 'unshare', '--net', '--mount', '--pid', '--ipc', '--uts', '--fork', '--kill-child', '--mount-proc',
        'sh', '-c', 'ip link set lo up && exec sleep infinity',
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        pid = await asyncio.to_thread(_child_pid, process.pid)
        for _ in range(100):
            if os.readlink(f'/proc/{pid}/ns/net') != os.readlink('/proc/self/ns/net'):
                break
            await asyncio.sleep(.01)
        box = Sandbox(cfg=RuntimeConfig(nsinit=Path('/unused'), work_root=directory), lower=[], pod=pod)
        box._dir, box._unshare, box.init_pid, box.proc_mode = directory, process, pid, 'private'
        return box
    except BaseException:
        await offline.stop_process(process)
        raise


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux namespace protocol')
@pytest.mark.parametrize('drop_capabilities', [False, True], ids=['unhardened-control', 'capdrop'])
def test_real_namespaces_model_only_services_and_background_isolation(tmp_path, drop_capabilities):
    _require_real_namespaces()
    async def go():
        from harbor_ns.sandbox import Pod
        pod, boxes, owned = Pod(bridge=True), [], []
        controller = model_relay = networks = None
        try:
            await pod.start()
            for i in range(2):
                box = await _bare_box(pod, tmp_path / f'box{i}')
                boxes.append(box)
                await pod.attach(box, pod.ip(i))
            agent, sidecar = boxes
            agent_argv = agent._nsenter()
            if drop_capabilities:
                agent_argv += ['--', str(await asyncio.to_thread(offline.capdrop_binary))]
            env = SimpleNamespace(_services={'main': SimpleNamespace(box=agent), 'sidecar': SimpleNamespace(box=sidecar)},
                                  _pod=pod, _pod_dir=tmp_path)
            networks = offline.OfflineNetworks(env)
            # Controlled host server stands for both the model endpoint and forbidden host egress.
            seen = []
            async def handle(request):
                seen.append(request.path)
                return web.json_response({'fake': True, 'path': request.path})
            app = web.Application(); app.router.add_route('*', '/{path:.*}', handle)
            controller = web.AppRunner(app); await controller.setup()
            site = web.TCPSite(controller, '0.0.0.0', 0); await site.start()
            port = site._server.sockets[0].getsockname()[1]
            addresses = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr']))
            host = next(a['local'] for interface in addresses for a in interface['addr_info']
                        if a.get('scope') == 'global' and not a['local'].startswith('127.'))
            host_url = f'http://{host}:{port}'
            get = lambda url: ('import urllib.request,json;u=urllib.request.build_opener(urllib.request.ProxyHandler({}));'
                               f'print(json.dumps(json.load(u.open({url!r},timeout=3))))')
            await agent.set_internet(True)
            assert (await _exec(agent_argv, get(host_url + '/setup')))['fake']
            await networks.isolate()
            assert all(not box.internet for box in boxes)
            original = os.readlink(f'/proc/{agent.init_pid}/ns/net')
            model_relay = offline.ModelRelay(agent, pod, host_url + '/v1')
            url = await model_relay.start()
            assert (await _exec(agent_argv, get(url + '/models')))['fake']
            # Task-local service connectivity is preserved with every uplink off.
            service_code = ('from http.server import HTTPServer,BaseHTTPRequestHandler\n'
                            'class H(BaseHTTPRequestHandler):\n'
                            ' def do_GET(self):\n'
                            '  self.send_response(200);self.end_headers();self.wfile.write(b"task-local")\n'
                            's=HTTPServer(("0.0.0.0",0),H);print(s.server_port,flush=True);s.serve_forever()')
            local = await asyncio.create_subprocess_exec(*agent_argv, sys.executable, '-I', '-c', service_code,
                                                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                                                       start_new_session=True)
            owned.append(local); task_port = int(await asyncio.wait_for(local.stdout.readline(), 5))
            read_local = (f'import urllib.request,json;u=urllib.request.build_opener(urllib.request.ProxyHandler({{}}));'
                          f'print(json.dumps(u.open("http://{pod.ip(0)}:{task_port}",timeout=3).read().decode()))')
            assert await _exec(sidecar._nsenter(), read_local) == 'task-local'
            deny = f'''import socket,json,urllib.request
out={{}}
for key,address in [('direct',({host!r},{port})),('host_loopback',('127.0.0.1',{port}))]:
 s=socket.socket();s.settimeout(1)
 try:s.connect(address);out[key]=True
 except OSError:out[key]=False
 finally:s.close()
try:
 urllib.request.build_opener(urllib.request.ProxyHandler({{'http':{host_url!r}}})).open('http://forbidden.invalid/',timeout=1)
 out['proxy']=True
except OSError:out['proxy']=False
print(json.dumps(out))
'''
            assert not any((await _exec(agent_argv, deny)).values())
            daemon = "import json,sys;print('ready',flush=True)\nfor line in sys.stdin:exec(json.loads(line),{})"
            background = await asyncio.create_subprocess_exec(*agent_argv, sys.executable, '-I', '-u', '-c', daemon,
                                                             stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                             stderr=asyncio.subprocess.PIPE, start_new_session=True)
            owned.append(background)
            assert (await asyncio.wait_for(background.stdout.readline(), 5)).strip() == b'ready'
            async def background_exec(code):
                background.stdin.write(json.dumps(code).encode() + b'\n')
                await background.stdin.drain()
                line = await asyncio.wait_for(background.stdout.readline(), 20)
                assert line, 'background process exited'
                return json.loads(line)
            assert not any((await background_exec(deny)).values())
            if drop_capabilities:
                status = await background_exec("import json;print(json.dumps(dict(line.split(':',1)for line in open('/proc/self/status')if ':'in line)))")
                denied = (1 << 21) | (1 << 19)  # SYS_ADMIN, SYS_PTRACE
                assert all(not(int(status[k],16)&denied)for k in ['CapBnd','CapPrm','CapEff','CapInh','CapAmb'])
                assert status['NoNewPrivs'].strip() == '1'
            await model_relay.close(); model_relay = None
            # Existing agent processes remain in their original netns while verifier commands get egress.
            networks.agent_active, networks.evaluator_internet = False, True
            evaluator = await networks.evaluator_network(agent)
            assert evaluator != agent.init_pid
            assert os.readlink(f'/proc/{agent.init_pid}/ns/net') == original
            assert not any((await _exec(agent_argv, deny)).values())
            agent.exec_network_pid = evaluator
            evaluator_argv = agent._nsenter()
            assert (await _exec(evaluator_argv, get(host_url + '/grade')))['fake']
            # Localhost relay points only from evaluator to original task server.
            local_get = (f'import urllib.request,json;u=urllib.request.build_opener(urllib.request.ProxyHandler({{}}));'
                         f'print(json.dumps(u.open("http://127.0.0.1:{task_port}",timeout=3).read().decode()))')
            assert await _exec(evaluator_argv, local_get) == 'task-local'
            sleeper = await asyncio.create_subprocess_exec(*evaluator_argv, 'sleep', '20',
                                                          stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                          start_new_session=True)
            owned.append(sleeper)
            await asyncio.sleep(.1)
            # A verifier process visible in agent /proc must not grant a background agent its egress netns.
            escape = f'''import os,glob,json,urllib.request
original=os.readlink('/proc/self/ns/net');answer={{'escaped':False,'visible_other_networks':0}}
for path in glob.glob('/proc/[0-9]*/ns/net'):
 try:
  if os.readlink(path)==original:continue
  answer['visible_other_networks']+=1
  fd=os.open(path,os.O_RDONLY)
  try:os.setns(fd,os.CLONE_NEWNET)
  finally:os.close(fd)
  u=urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
  with u.open({host_url+'/background-escape'!r},timeout=2) as response:
   answer['escaped']=response.status==200
  if answer['escaped']:answer['source']=path;break
 except OSError:pass
print(json.dumps(answer))
'''
            result = await background_exec(escape)
            print('OFFLINE_NETWORK_PROOF ' + json.dumps({'capdrop': drop_capabilities, 'escape': result, 'host_routes': seen, 'agent_net': original, 'evaluator_net': os.readlink(f'/proc/{evaluator}/ns/net')}), flush=True)
            assert result['escaped'] is not drop_capabilities, result
            if drop_capabilities:
                assert '/background-escape' not in seen
            else:
                assert '/background-escape' in seen, 'negative control did not exercise the actual escape' 
        finally:
            if model_relay is not None: await model_relay.close()
            if networks is not None: await networks.close()
            for process in owned: await offline.stop_process(process)
            for box in boxes:
                await offline.stop_process(box._slirp)
                await offline.stop_process(box._unshare)
            await pod.stop()
            if controller is not None: await controller.cleanup()
    asyncio.run(go())


@pytest.mark.parametrize('previous_offline', [False, True])
@pytest.mark.parametrize('termination', ['normal', 'start_error', 'cancel'])
def test_phase_restoration_keeps_original_namespaces_offline(monkeypatch, previous_offline, termination):
    from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    async def go():
        entered = asyncio.Event()
        calls, closed = [], []
        class Box:
            private_net, proc_mode, internet = True, 'private', True
            async def set_internet(self, on):
                calls.append(on); self.internet = on
        class Relay:
            def __init__(self, *args): pass
            async def start(self):
                entered.set()
                if termination == 'start_error': raise RuntimeError('controlled relay start failure')
                if termination == 'cancel': await asyncio.Event().wait()
                return 'http://127.0.0.1:1234/v1'
            async def close(self): closed.append(True)
        monkeypatch.setattr(offline, 'ModelRelay', Relay)
        class Environment:
            offline_agent_model = NsSidecarEnvironment.offline_agent_model
            _apply_network_policy = NsSidecarEnvironment._apply_network_policy
            _offline = None
            _pod = None
            def __init__(self):
                self._box = Box()
                self._services = {'main': SimpleNamespace(box=self._box)}
                self.network_policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK if previous_offline else NetworkMode.PUBLIC)
            async def set_network_policy(self, value):
                if self.network_policy == value: return
                await self._apply_network_policy(value)
                self.network_policy = value
        environment = Environment()
        previous = environment.network_policy
        async def enter():
            async with environment.offline_agent_model('http://fake/v1') as url:
                assert url.endswith('/v1') and environment._offline.agent_active
                assert not environment._box.internet
        task = asyncio.create_task(enter())
        await entered.wait()
        if termination == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        elif termination == 'start_error':
            with pytest.raises(RuntimeError, match='controlled'): await task
        else:
            await task
        assert closed == [True]
        assert calls and not any(calls), 'phase restoration must never restore the agent namespace NAT'
        assert environment.network_policy == previous
        assert not environment._box.internet and not environment._offline.agent_active
        assert environment._offline.evaluator_internet is (not previous_offline)
    asyncio.run(go())


@pytest.mark.parametrize('private_net,proc_mode', [(False, 'private'), (True, 'host')])
def test_unsafe_namespace_configuration_fails_closed(private_net, proc_mode):
    async def go():
        box = SimpleNamespace(private_net=private_net, proc_mode=proc_mode)
        networks = offline.OfflineNetworks(SimpleNamespace(_services={'main': SimpleNamespace(box=box)}))
        with pytest.raises(RuntimeError, match='private network and PID/proc'):
            await networks.isolate()
    asyncio.run(go())


def test_capdrop_wraps_entire_pi_pipeline(monkeypatch):
    import shlex
    from tb_agents.harness_agents import PiCached, Pi
    async def go():
        seen = []
        async def capture(self, environment, command, **kwargs):
            seen.append(command)
        monkeypatch.setattr(Pi, 'exec_as_agent', capture)
        monkeypatch.setenv('PI_AGENT_INTERNET', '0')
        monkeypatch.setenv('PI_INSTRUCTION_ARGV', '1')
        agent = object.__new__(PiCached)
        command = "cd /app && pi --print --mode json --model fake 'instruction' 2>&1 </dev/null | grep answer | tee /tmp/out"
        await agent.exec_as_agent(None, command)
        assert shlex.split(seen[0]) == ['exec', '/opt/harbor-pi/offline-capdrop', '/bin/bash', '-c', command]
        await agent.exec_as_agent(None, 'true')
        assert seen[1] == 'true'
    asyncio.run(go())


def test_relay_close_aborts_disconnected_inflight_model_request(monkeypatch):
    """Canceled agents must not leave host-side endpoint requests running through teardown."""
    async def go():
        entered, release = asyncio.Event(), asyncio.Event()
        async def backend(request):
            entered.set()
            await release.wait()
            return web.json_response({'fake': True})
        app = web.Application(); app.router.add_post('/v1/chat/completions', backend)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        async def fake_process(*args, **kwargs):
            stdout = asyncio.StreamReader(); stdout.feed_data(b'{"port": 1}\n'); stdout.feed_eof()
            return SimpleNamespace(stdout=stdout, returncode=0)
        monkeypatch.setattr(offline.asyncio, 'create_subprocess_exec', fake_process)
        relay = offline.ModelRelay(SimpleNamespace(init_pid=1), SimpleNamespace(holder_pid=1),
                                   f'http://127.0.0.1:{port}/v1')
        closing = request = None
        blocked = False
        try:
            await relay.start()
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(Path(relay.tmp.name) / 'api.sock'))) as client:
                request = asyncio.create_task(client.post('http://relay/v1/chat/completions', json={}))
                await asyncio.wait_for(entered.wait(), 3)
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                closing = asyncio.create_task(relay.close())
                try:
                    await asyncio.wait_for(asyncio.shield(closing), 2)
                except asyncio.TimeoutError:
                    blocked = True
                finally:
                    release.set()
                    await closing
        finally:
            release.set()
            if closing is None: await relay.close()
            await runner.cleanup()
        assert not blocked, 'disconnected endpoint request still blocks relay teardown'
        assert relay.http.closed
    asyncio.run(go())

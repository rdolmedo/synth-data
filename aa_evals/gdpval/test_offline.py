"""Offline protocol and CPU kernel canaries; no model inference or external network requests."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aa_evals.gdpval.network import POLICY, require_offline


SIF = Path(os.environ.get('GDPVAL_OFFLINE_TEST_SIF', '/data/aai/aa_evals/sif/gdpval-v1.sif'))


def test_legacy_protocol_cannot_be_relabelled_or_resumed():
    with pytest.raises(ValueError, match='new run ID'):
        require_offline({})
    with pytest.raises(ValueError, match='new run ID'):
        require_offline({'agent_network': {'mode': 'legacy-online'}})
    require_offline({'agent_network': dict(POLICY)})


def test_offline_default_and_legacy_config_guard(tmp_path, monkeypatch):
    from aa_evals.gdpval import run as R, endpoint as E
    from aa_evals.gdpval.tasks import Task
    monkeypatch.setattr(E, 'resolve', lambda spec: {'served_name': 'synthetic', 'request': {}})
    monkeypatch.setattr(R, 'endpoint_ctx', lambda model: 131072)
    monkeypatch.setattr(R.protocol, 'image_inputs', lambda model: (False, 'synthetic'))
    monkeypatch.setattr(R.T, 'parquet_sha', lambda: 'synthetic')
    monkeypatch.setattr(R, 'select', lambda args: [Task('t', 'test', 'test', 'synthetic task', [], [])])
    monkeypatch.setattr(R, 'RUNS', tmp_path)
    monkeypatch.setenv('BRAVE_API_KEY', 'synthetic-not-a-credential')
    R.main(['--stage', 'init', '--run-id', 'new', '--model', 'url:http://synthetic/v1@m'])
    cfg = json.loads((tmp_path / 'new' / 'config.json').read_text())
    assert cfg['agent_network'] == POLICY
    assert not any('web' in tool for tool in cfg['harness']['tools'])
    old = tmp_path / 'old' / 'config.json'
    old.parent.mkdir()
    old.write_text(json.dumps({'started_at': 'legacy'}))
    before = old.read_bytes()
    with pytest.raises(SystemExit, match='network protocol changed'):
        R.main(['--stage', 'init', '--run-id', 'old', '--model', 'url:http://synthetic/v1@m', '--force'])
    assert old.read_bytes() == before


def test_agent_toolset_has_no_web_provider_and_records_network(tmp_path, monkeypatch):
    from aa_evals.gdpval import agent as A
    from aa_evals.gdpval.smoke_fake import Scripted
    from aa_evals.gdpval.tasks import Task
    fake = Scripted([('abandon_task_finish', {'reason': 'synthetic canary complete'})])
    monkeypatch.setattr(A, 'CappedVLLMClient', lambda **kwargs: fake)
    monkeypatch.setenv('BRAVE_API_KEY', 'synthetic-not-a-credential')
    cfg = {'agent_network': dict(POLICY),
           'model': {'spec': 'url:unused@synthetic', 'served_name': 'synthetic', 'body': {},
                     'max_output': 1024, 'vision': False},
           'context': {'window': 131072}, 'sandbox': {'sif_local': str(SIF)}, 'limits': {'wall_sec': 10}}
    result = asyncio.run(A.run_attempt(Task('synthetic', 'test', 'test', 'A synthetic canary.', [], []), 0,
                                       cfg=cfg, work=tmp_path / 'work', out_dir=tmp_path / 'out'))
    assert result['status'] == 'ok' and result['end'] == 'abandon'
    assert result['agent_network'] == POLICY
    assert fake.seen and not any('web' in name or 'search' in name or 'fetch' in name for name in fake.seen[0][1])
    assert 'code_exec' in fake.seen[0][1]


@pytest.mark.skipif(not SIF.exists() or not shutil.which('apptainer') or not Path('/usr/bin/unshare').exists(),
                    reason='requires cluster Apptainer SIF and unshare for real CPU namespace canary')
def test_kernel_offline_code_exec_keeps_host_model_client_online(tmp_path, monkeypatch):
    from aa_evals.gdpval.sandbox import ApptainerCodeExecToolProvider
    from aa_evals.gdpval.client import CappedVLLMClient
    from stirrup.core.models import UserMessage
    calls = []
    class Model(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            if self.path.endswith('/tokenize'):
                out = {'count': 100}
            else:
                calls.append(self.path)
                out = {'id': 'synthetic', 'object': 'chat.completion', 'created': 0, 'model': 'host-check',
                       'choices': [{'index': 0, 'finish_reason': 'stop',
                                    'message': {'role': 'assistant', 'content': 'host endpoint reachable'}}],
                       'usage': {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104}}
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy'):
        monkeypatch.delenv(name, raising=False)
    host_net = os.readlink('/proc/self/ns/net')
    proxy_ip = socket.gethostbyname('proxy')
    home, scratch = tmp_path / 'home', tmp_path / 'tmp'
    home.mkdir()
    scratch.mkdir()
    probe = '''import json,os,socket,subprocess
from pathlib import Path
proxy=PROXY_IP
port=MODEL_PORT
out={'net':os.readlink('/proc/self/ns/net'),'interfaces':socket.if_nameindex(),'sockets':[]}
for host,p in [('1.1.1.1',80),(proxy,8080),('127.0.0.1',port)]:
 s=socket.socket();s.settimeout(1)
 out['sockets'].append(s.connect_ex((host,p)));s.close()
s=socket.socket(socket.AF_INET6);s.settimeout(1)
out['ipv6']=s.connect_ex(('::1',port));s.close()
os.environ.update(http_proxy='http://'+proxy+':8080',https_proxy='http://'+proxy+':8080',ALL_PROXY='http://'+proxy+':8080')
out['curl_proxy']=subprocess.run(['curl','--proxy','http://'+proxy+':8080','--noproxy','','--max-time','2','http://example.com/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
out['curl_direct']=subprocess.run(['curl','--noproxy','*','--max-time','2','http://1.1.1.1/'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
Path('/tmp/offline-persist').write_text('local execution works')
print(json.dumps(out))
'''.replace('PROXY_IP', repr(proxy_ip)).replace('MODEL_PORT', str(server.server_port))
    (home / 'probe.py').write_text(probe)
    async def exercise():
        stats = {}
        provider = ApptainerCodeExecToolProvider(sif=str(SIF), home=home, tmp=scratch, stats=stats)
        client = CappedVLLMClient(spec=f'url:http://127.0.0.1:{server.server_port}/v1@host-check',
                                 served_name='host-check', body={}, context_cap=131072, max_output=1024, max_tries=1)
        try:
            async with provider:
                result = await provider.run_command('python /home/user/probe.py', timeout=30)
                persisted = await provider.run_command('cat /tmp/offline-persist', timeout=10)
                answer = await client.generate([UserMessage(content='host model client canary')], {})
            return result, persisted, answer, stats
        finally:
            await client.aclose()
    try:
        result, persisted, answer, stats = asyncio.run(exercise())
        assert result.exit_code == 0, result.stderr
        probe_result = json.loads(result.stdout.strip())
        assert probe_result['net'] != host_net and os.readlink('/proc/self/ns/net') == host_net
        assert all(code != 0 for code in probe_result['sockets']) and probe_result['ipv6'] != 0
        assert probe_result['curl_proxy'] != 0 and probe_result['curl_direct'] != 0
        assert [name for _, name in probe_result['interfaces']] == ['lo']
        assert persisted.exit_code == 0 and 'local execution works' in persisted.stdout
        assert any(getattr(block, 'text', '') == 'host endpoint reachable' for block in answer.blocks)
        assert calls == ['/v1/chat/completions']
        assert stats['network_mode'] == 'offline' and 'no-uplink' in stats['network_enforcement']
        print(json.dumps({'kernel_offline_canary': probe_result, 'host_model_requests': calls,
                          'local_files_persist': persisted.exit_code == 0, 'sandbox_stats': stats}))
    finally:
        server.shutdown()
        server.server_close()


def test_namespace_creation_failure_never_runs_command_on_host(tmp_path):
    from aa_evals.gdpval.sandbox import ApptainerCodeExecToolProvider
    fail = tmp_path / 'fail-unshare'
    fail.write_text('#!/bin/sh\necho "unshare: permission denied" >&2\nexit 1\n')
    fail.chmod(0o755)
    marker = tmp_path / 'host-escape'
    async def exercise():
        stats = {}
        provider = ApptainerCodeExecToolProvider(sif=str(SIF), home=tmp_path / 'home', tmp=tmp_path / 'tmp',
                                                stats=stats, unshare=str(fail))
        provider.pid_ns = False  # even after the independent PID fallback, the network wrapper is mandatory
        assert provider.argv('true')[:5] == [str(fail), '--user', '--map-root-user', '--net', '--']
        async with provider:
            result = await provider.run_command(f'touch {marker}', timeout=10)
        return result, stats
    result, stats = asyncio.run(exercise())
    assert result.exit_code != 0 and not marker.exists()
    assert stats['sandbox_errors'] == stats['commands'] == 1

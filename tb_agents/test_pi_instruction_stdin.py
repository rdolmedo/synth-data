"""M3.5: Pi gets the task instruction on stdin, never in a process argv (Pi agent self-kill, #203).

  python -m pytest -q tb_agents/test_pi_instruction_stdin.py
"""
import asyncio
import json
import os
import shlex
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tb_agents.harness_agents import PiCached, PiCont16kCached, instruction_via_stdin

MARK = 'win311-marker-7f3c'
INSTRUCTION = (f"Run Windows 3.11 from /app/isos/{MARK}.img.\n- keep 'quotes' and \"double\" quotes, $HOME, `ticks`;\n"
               "  and a line starting with -- dashes\n")


def harbor_command(instruction, flags='--thinking xhigh --', session='/logs/agent/pi/sessions',
                   out='/logs/agent/pi.txt'):
    """The launch Harbor's Pi.run() builds (harbor/agents/installed/pi.py), with PiCont16kCached's trailing `--`."""
    return (f". ~/.nvm/nvm.sh; mkdir -p {shlex.quote(session)} && "
            f"PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent pi --print --mode json --session-dir {shlex.quote(session)} "
            f"--provider harbor-endpoint --model qwen38-27b {flags} {shlex.quote(instruction)} "
            f"2>&1 </dev/null | grep -v '\"type\":\"message_update\"' | stdbuf -oL tee {shlex.quote(out)}")


def test_rewrite_removes_the_instruction_and_keeps_the_pipeline():
    cmd, instruction = instruction_via_stdin(harbor_command(INSTRUCTION), '/tmp/harbor-pi-agent/instruction.md')
    assert instruction == INSTRUCTION
    assert MARK not in cmd and 'quotes' not in cmd
    assert cmd.endswith("--thinking xhigh -- 2>&1 </tmp/harbor-pi-agent/instruction.md | grep -v "
                        "'\"type\":\"message_update\"' | stdbuf -oL tee /logs/agent/pi.txt")


def test_rewrite_refuses_an_unknown_command_shape():
    with pytest.raises(ValueError):
        instruction_via_stdin('pi --print --mode json hello', '/x')


class FakeEnv:
    default_user = None

    def __init__(self):
        self.uploads, self.commands = {}, []

    async def upload_file(self, src, dst):
        self.uploads[dst] = Path(src).read_text()


def test_agent_uploads_the_instruction_and_never_execs_it(monkeypatch):
    monkeypatch.delenv('PI_INSTRUCTION_ARGV', raising=False)
    seen = []

    async def fake_exec(self, environment, command, env=None, cwd=None, timeout_sec=None):
        seen.append(command)
    monkeypatch.setattr('harbor.agents.installed.base.BaseInstalledAgent.exec_as_agent', fake_exec)
    agent = PiCont16kCached.__new__(PiCont16kCached)
    env = FakeEnv()
    asyncio.run(agent.exec_as_agent(env, harbor_command(INSTRUCTION)))
    assert env.uploads == {PiCached.INSTRUCTION_FILE: INSTRUCTION}
    assert len(seen) == 1 and MARK not in seen[0] and f'<{PiCached.INSTRUCTION_FILE} |' in seen[0]
    asyncio.run(agent.exec_as_agent(env, 'pi --version'))   # other commands pass through unchanged
    assert seen[-1] == 'pi --version'
    monkeypatch.setenv('PI_INSTRUCTION_ARGV', '1')           # the pre-M3.5 launch, kept for reproducing M3.4
    asyncio.run(agent.exec_as_agent(env, harbor_command(INSTRUCTION)))
    assert MARK in seen[-1]


def cmdlines_with(mark):
    hits = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                c = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if mark in c:
                hits.append((p.name, c[:200]))
    return hits


def test_the_cmdline_scan_sees_an_argv():
    probe = subprocess.Popen(['sh', '-c', 'sleep 30', MARK])
    try:
        assert any(pid == str(probe.pid) for pid, _ in cmdlines_with(MARK))
    finally:
        probe.kill()
        probe.wait()


def test_real_pi_reads_the_instruction_from_stdin_and_no_argv_holds_it(tmp_path):
    """The rewritten launch under bash with the pinned Pi bundle: while Pi's request is in flight, no process on the
    machine has the instruction in its cmdline, and the model receives it as the first user message."""
    from tb_agents.test_pi_reasoning_history import BUNDLE, CACHE
    if not (CACHE / BUNDLE).is_file():
        pytest.skip('patched Pi bundle not available')
    root = tmp_path / 'pi'
    with tarfile.open(CACHE / BUNDLE) as archive:
        archive.extractall(root, members=[m for m in archive if m.isfile() or m.isdir()], filter='data')
    node = root / 'opt/harbor-pi/node/bin/node'
    cli = root / 'opt/harbor-pi/pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
    seen = {'argv_hits': None, 'user': None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "test"}]}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if seen['argv_hits'] is None:
                seen['argv_hits'] = cmdlines_with(MARK)
                u = [m for m in body['messages'] if m['role'] == 'user'][0]['content']
                seen['user'] = u if isinstance(u, str) else ''.join(x.get('text', '') for x in u)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in [{'choices': [{'index': 0, 'delta': {'content': 'done'}, 'finish_reason': None}]},
                          {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
                          {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 1}}]:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home, config, bindir = tmp_path / 'home', tmp_path / 'cfg', tmp_path / 'bin'
    for d in (home / '.nvm', config, bindir):
        d.mkdir(parents=True)
    (home / '.nvm/nvm.sh').write_text('')
    (bindir / 'pi').write_text(f'#!/bin/sh\nexec {node} {cli} "$@"\n')
    (bindir / 'pi').chmod(0o755)
    (config / 'settings.json').write_text(json.dumps({'httpIdleTimeoutMs': 60000}))
    (config / 'models.json').write_text(json.dumps({'providers': {'harbor-endpoint': {
        'baseUrl': f'http://127.0.0.1:{server.server_port}/v1', 'api': 'openai-completions', 'apiKey': 'mock',
        'models': [{'id': 'qwen38-27b', 'name': 'q', 'reasoning': True, 'input': ['text'], 'contextWindow': 131072,
                    'maxTokens': 16384, 'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
        'compat': {'supportsDeveloperRole': False}}}}))
    ifile = tmp_path / 'instruction.md'
    ifile.write_text(INSTRUCTION)
    cmd, _ = instruction_via_stdin(harbor_command(INSTRUCTION, session=str(tmp_path / 'sessions'),
                                                  out=str(tmp_path / 'pi.txt')), str(ifile))
    cmd = cmd.replace('PI_CODING_AGENT_DIR=/tmp/harbor-pi-agent', f'PI_CODING_AGENT_DIR={config}')
    assert MARK not in cmd
    try:
        p = subprocess.run(['bash', '-c', cmd], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                           env={**{k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
                                'HOME': str(home), 'PATH': f'{bindir}:{os.environ["PATH"]}', 'NO_PROXY': '127.0.0.1'})
    finally:
        server.shutdown()
        server.server_close()
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert seen['argv_hits'] == [], seen['argv_hits']
    assert seen['user'] == INSTRUCTION.strip()   # Pi trims piped stdin: only the final newline is lost

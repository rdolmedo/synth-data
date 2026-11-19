"""PiCont16kCached options: defaults keep the base behavior; campaign opt-ins compose.

#480 merge: the defaults are the AA-eval/eval_q35 ones (prompt separator on, Harbor's MCP adapter); synthesis turns
on native MCP explicitly (synth/worker.py PI_NATIVE_MCP=1, PI_CLI_PROMPT_SEPARATOR=1)."""
import asyncio
import hashlib
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.pi import Pi

from tb_agents import harness_agents
from tb_agents.harness_agents import PiCached, PiCont16kCached, pi_prompt_separator

ENV = ('PI_CLI_PROMPT_SEPARATOR', 'PI_RUNTIME_COMPAT_ARCHIVE', 'PI_GLIBC_COMPAT_ARCHIVE', 'PI_NATIVE_MCP',
       'PI_MCP_AUTO_CODEMODE', 'PI_SETTINGS_JSON')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)


def agent(tmp_path):
    return PiCont16kCached(logs_dir=tmp_path / 'logs', model_name='openai/test', version='1.2.0', thinking='max')


def test_prompt_separator_rewrite_is_idempotent():
    command = "pi --print --mode json --thinking max '- leading dash' 2>&1 </dev/null | grep -v x | tee o"
    once = pi_prompt_separator(command)
    assert shlex.split(once.split(' 2>&1')[0])[-2:] == ['--', '- leading dash']
    assert pi_prompt_separator(once) == once


@pytest.mark.parametrize('env,expected', [({}, True), ({'PI_CLI_PROMPT_SEPARATOR': '1'}, True),
                                          ({'PI_CLI_PROMPT_SEPARATOR': '0'}, False)])
def test_prompt_separator_is_opt_out(tmp_path, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert shlex.split(agent(tmp_path).build_cli_flags())[-1:] == (['--'] if expected else ['max'])


def servers():
    return [SimpleNamespace(name='jira', transport='stdio', command='python3', args=['/app/s.py']),
            SimpleNamespace(name='remote', transport='http', url='http://task:8123/mcp')]


def test_native_mcp_default_and_options(tmp_path, monkeypatch):
    pi = agent(tmp_path)
    pi.mcp_servers = servers()
    assert pi._build_mcp_config() == Pi._build_mcp_config(pi)   # default: Harbor's adapter config, as AA evals ran
    monkeypatch.setenv('PI_NATIVE_MCP', '1')                    # synthesis
    expected = {'jira': {'command': 'python3', 'args': ['/app/s.py'], 'exposure': 'direct'},
                'remote': {'url': 'http://task:8123/mcp', 'exposure': 'direct'}}
    assert pi._build_mcp_config() == {'mcpServers': expected}
    monkeypatch.setenv('PI_MCP_AUTO_CODEMODE', '0')
    assert pi._build_mcp_config() == {'mcpServers': expected, 'autoEnableCodemode': False}
    monkeypatch.setenv('PI_NATIVE_MCP', '0')
    assert pi._build_mcp_config() == Pi._build_mcp_config(pi)


def test_mcp_legacy_extension_flag_removed_only_with_native_mcp(tmp_path, monkeypatch):
    legacy = '--extension /tmp/harbor-pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts '
    command = 'pi --print --mode json ' + legacy + "--thinking max 'task' 2>&1 </dev/null | grep -v x"
    pi = agent(tmp_path)
    pi.mcp_servers = servers()
    monkeypatch.setattr(PiCached, 'exec_as_agent', AsyncMock())
    asyncio.run(pi.exec_as_agent('env', command))
    assert PiCached.exec_as_agent.call_args.args[1] == command   # default off
    monkeypatch.setenv('PI_NATIVE_MCP', '1')
    asyncio.run(pi.exec_as_agent('env', command))
    assert legacy not in PiCached.exec_as_agent.call_args.args[1]
    monkeypatch.setenv('PI_NATIVE_MCP', '0')
    asyncio.run(pi.exec_as_agent('env', command))
    assert PiCached.exec_as_agent.call_args.args[1] == command


def write_archive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'compat')
    path.with_suffix('.tar.sha256').write_text(hashlib.sha256(b'compat').hexdigest())
    return path


def test_compat_archive_resolution_order(tmp_path, monkeypatch):
    assert harness_agents._compat_archive() is None
    monkeypatch.setenv('PI_GLIBC_COMPAT_ARCHIVE', str(write_archive(tmp_path / 'q.tar')))
    assert harness_agents._compat_archive()[2] == 'PI_GLIBC_COMPAT_ARCHIVE'
    monkeypatch.setenv('PI_RUNTIME_COMPAT_ARCHIVE', str(write_archive(tmp_path / 'g.tar')))
    assert harness_agents._compat_archive()[2] == 'PI_RUNTIME_COMPAT_ARCHIVE'


def test_glibc_failure_without_configured_loader_is_reraised(tmp_path, monkeypatch):
    monkeypatch.setattr(PiCached, 'install', AsyncMock(side_effect=NonZeroAgentExitCodeError('GLIBC_2.28 not found')))
    pi = agent(tmp_path)
    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(pi.install(SimpleNamespace(upload_file=AsyncMock())))


def test_glibc_failure_uses_configured_loader_and_environment_hook(tmp_path, monkeypatch):
    monkeypatch.setenv('PI_RUNTIME_COMPAT_ARCHIVE', str(write_archive(tmp_path / 'g.tar')))
    monkeypatch.setattr(PiCached, 'install', AsyncMock(side_effect=NonZeroAgentExitCodeError('GLIBC_2.28 not found')))
    pi = agent(tmp_path)
    pi.exec_as_root = AsyncMock()
    pi.exec_as_agent = AsyncMock()
    hook = AsyncMock()
    asyncio.run(pi.install(SimpleNamespace(upload_file=AsyncMock(), install_pi_extensions=hook)))
    assert ' -C / && ' in pi.exec_as_root.call_args.kwargs['command']
    assert json.loads((pi.logs_dir / 'pi-runtime-compat.json').read_text())['source'] == 'PI_RUNTIME_COMPAT_ARCHIVE'
    hook.assert_awaited_once_with(pi)


def test_settings_json_merge_keeps_context_policy_last(tmp_path, monkeypatch):
    monkeypatch.setenv('PI_SETTINGS_JSON', json.dumps({'retry': {'enabled': False}, 'compaction': {'enabled': True}}))
    monkeypatch.setattr(Pi, '_write_custom_models_json', AsyncMock())
    pi = agent(tmp_path)
    pi._upload_config_text = AsyncMock()
    asyncio.run(pi._write_custom_models_json(SimpleNamespace(), {'providers': {}}))
    settings = json.loads(pi._upload_config_text.call_args.kwargs['content'])
    assert settings['retry'] == {'enabled': False}
    assert settings['compaction'] == {'enabled': False}

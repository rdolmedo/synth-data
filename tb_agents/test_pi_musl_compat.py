"""Task/platform/checksum guard tests; real cached-image proof is a separate CPU canary."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tb_agents import harness_agents as H
from tb_agents import pi_musl_compat as M


def setup(tmp_path, monkeypatch, task=M.TASK, platform='x86_64-musl'):
    monkeypatch.setattr(H, '_sandbox_platform', AsyncMock(return_value=platform))
    archive = tmp_path / 'compat.tar';archive.write_bytes(b'controlled archive')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(M, 'COMPAT_SHA256', digest)
    archive.with_suffix('.tar.sha256').write_text(digest + ' compat.tar\n')
    monkeypatch.setenv('PI_RUNTIME_COMPAT_ARCHIVE', str(archive))
    env = SimpleNamespace(environment_name=task, upload_file=AsyncMock())
    src = tmp_path / 'pinned.tar';src.write_bytes(b'pinned')
    def bundle(platform):
        assert platform == 'x86_64'
        return src
    agent = SimpleNamespace(logs_dir=tmp_path / 'logs', _cached_bundle=bundle,
        _continuation_sha256='verified-pinned-bundle', ensure_system_dependencies=AsyncMock(),
        exec_as_root=AsyncMock(), exec_as_agent=AsyncMock(return_value=SimpleNamespace(stdout='1.2.0\n')))
    return agent,env,archive


def test_install_uses_verified_private_loader_without_global_environment_change(tmp_path, monkeypatch):
    agent, env, archive = setup(tmp_path, monkeypatch)
    asyncio.run(M.install(agent, env))
    assert env.upload_file.await_count == 2
    command = agent.exec_as_root.call_args.kwargs['command']
    assert '--library-path /opt/harbor-pi/glibc-compat' in command
    assert '/lib64/ld-linux' not in command and 'LD_LIBRARY_PATH=' not in command
    agent.exec_as_agent.assert_awaited_once_with(env, command='pi --version')
    record = json.loads((agent.logs_dir / 'pi-runtime-compat.json').read_text())
    assert record['status'] == 'ready' and record['task'] == M.TASK
    assert record['pi_bundle_sha256'] == 'verified-pinned-bundle'
    assert record['sha256'] == M.COMPAT_SHA256 and record['global_library_env_changed'] is False


@pytest.mark.parametrize('task,platform', [('another-task','x86_64-musl'),(M.TASK,'x86_64'),(M.TASK,'aarch64-musl')])
def test_wrong_task_or_platform_rejected_before_upload(tmp_path, monkeypatch,task,platform):
    agent,env,_ = setup(tmp_path,monkeypatch,task,platform)
    with pytest.raises(ValueError):asyncio.run(M.install(agent,env))
    env.upload_file.assert_not_awaited();agent.exec_as_root.assert_not_awaited()


@pytest.mark.parametrize('change', ['archive','sidecar','missing','empty'])
def test_corrupt_or_missing_compatibility_archive_rejected_before_upload(tmp_path,monkeypatch,change):
    agent,env,archive = setup(tmp_path,monkeypatch)
    if change == 'archive':archive.write_bytes(b'tampered')
    if change == 'sidecar':archive.with_suffix('.tar.sha256').write_text('f'*64)
    if change == 'missing':archive.unlink()
    if change == 'empty':archive.with_suffix('.tar.sha256').write_text('')
    with pytest.raises(ValueError):asyncio.run(M.install(agent,env))
    env.upload_file.assert_not_awaited();agent.exec_as_root.assert_not_awaited()


def test_explicit_subclass_preserves_original_protocol_methods():
    cls=H.PiCont16kEnvClockMuslCompat
    for method in ('run','exec_as_agent','_cached_bundle','build_cli_flags'):
        assert getattr(cls,method) is getattr(H.PiCont16kEnvClock,method)


class ConfigEnvironment:
    environment_name = M.TASK
    def __init__(self, user):self.default_user=user
    def with_default_user(self, user):
        from contextlib import contextmanager
        @contextmanager
        def scoped():
            previous=self.default_user;self.default_user=user
            try:yield
            finally:self.default_user=previous
        return scoped()


def config_agent(tmp_path, stdout='1001\n'):
    record={'task':M.TASK,'platform':'x86_64-musl','status':'ready'}
    (tmp_path/'pi-runtime-compat.json').write_text(json.dumps(record))
    return SimpleNamespace(logs_dir=tmp_path,exec_as_agent=AsyncMock(return_value=SimpleNamespace(stdout=stdout)))


@pytest.mark.parametrize('original',[None,'agent'])
@pytest.mark.parametrize('fails',[False,True])
def test_config_upload_uses_effective_uid_and_restores_default_on_success_or_error(tmp_path,original,fails):
    agent=config_agent(tmp_path);env=ConfigEnvironment(original)
    async def upload(environment,**kwargs):
        assert environment.default_user==1001
        if fails:raise RuntimeError('controlled upload failure')
    coro=M.upload_config(agent,env,upload,content='private',remote_path='/tmp/harbor-pi-agent/models.json',filename='models.json')
    if fails:
        with pytest.raises(RuntimeError,match='controlled'):asyncio.run(coro)
    else:asyncio.run(coro)
    assert env.default_user is original
    row=json.loads((tmp_path/'pi-runtime-compat.json').read_text())['config_ownership'][0]
    assert row['effective_uid']==1001 and row['default_user_before']==row['default_user_after']==original
    assert row['status']==('upload_failed' if fails else 'uploaded')


@pytest.mark.parametrize('uid',['','root','1.5','-1','True','4294967295','1001\n1002','１２３'])
def test_malformed_effective_uid_rejected_before_upload(tmp_path,uid):
    agent=config_agent(tmp_path,uid);env=ConfigEnvironment(None);upload=AsyncMock()
    with pytest.raises(ValueError,match='numeric'):
        asyncio.run(M.upload_config(agent,env,upload,content='',remote_path='/tmp/harbor-pi-agent/models.json',filename='models.json'))
    upload.assert_not_awaited();assert env.default_user is None


def test_owner_override_rejects_foreign_task_and_nonprivate_path(tmp_path):
    for task,path in [('other','/tmp/harbor-pi-agent/models.json'),(M.TASK,'/etc/config')]:
        agent=config_agent(tmp_path);env=ConfigEnvironment(None);env.environment_name=task;upload=AsyncMock()
        with pytest.raises(ValueError,match='restricted'):
            asyncio.run(M.upload_config(agent,env,upload,content='',remote_path=path,filename='models.json'))
        upload.assert_not_awaited();agent.exec_as_agent.assert_not_awaited()

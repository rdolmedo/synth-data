"""CPU tests for harbor_ns scratch hygiene (failed builds, orphaned image directories, the per-build scratch log) and
the offline sandboxes' no_proxy list.

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q harbor_ns/tests
"""
import asyncio
import fcntl
import json
import os
from types import SimpleNamespace

import pytest

from harbor_ns import builder
from harbor_ns.builder import BuildError, ImageMeta, ImageStore, evict
from harbor_ns.sandbox import OFFLINE_DOMAINS, RuntimeConfig, hosts_names, proxy_env


def store(tmp_path, shared=True):
    rt = RuntimeConfig(nsinit=tmp_path / 'nsinit', work_root=tmp_path / 'boxes', nsfake=tmp_path / 'nsfake.so')
    return ImageStore(local_root=tmp_path / 'local', shared_root=tmp_path / 'shared' if shared else None, rt=rt,
                      nsfake=rt.nsfake, network_rebuilds=0)


def context(tmp_path):
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN true\n')
    return ctx


def fake_build(fail=False, hang=False):
    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        (d / 'rootfs' / 'usr').mkdir(parents=True)
        (d / 'rootfs' / 'usr' / 'big').write_bytes(b'x' * 1_000_000)
        with open(log, 'a') as f:
            f.write('step output\n')
        if hang:
            await asyncio.sleep(3600)
        if fail:
            raise BuildError('RUN (line 2) failed with exit code 1')
        return ImageMeta()
    return _build


def builds_log(s):
    return [json.loads(line) for line in (s.local_root / 'logs' / 'builds.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('shared', [True, False])
def test_failed_build_directory_is_deleted_and_its_log_kept(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(ImageStore, '_build', fake_build(fail=True))
    s = store(tmp_path, shared)
    with pytest.raises(BuildError):
        asyncio.run(s.build(context(tmp_path), label='task-x'))
    assert not list((s.local_root / 'images').iterdir()), 'the failed build left its image directory'
    logs = s.shared_root / 'logs' if shared else s.local_root / 'logs'
    failed = list(logs.glob('task-x.*.failed.log'))
    assert len(failed) == 1 and 'step output' in failed[0].read_text()
    (line,) = builds_log(s)
    assert line['ok'] is False and line['label'] == 'task-x' and line['image_gb'] > 0.0009


def test_cancelled_build_directory_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(ImageStore, '_build', fake_build(hang=True))
    s = store(tmp_path)

    async def run():
        await asyncio.wait_for(s.build(context(tmp_path), label='slow'), 0.5)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    assert not list((s.local_root / 'images').iterdir())


def test_successful_build_is_logged_and_low_scratch_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ImageStore, '_build', fake_build())
    s = store(tmp_path)
    key, rootfs, _ = asyncio.run(s.build(context(tmp_path), label='ok-task'))
    assert (rootfs / 'usr' / 'big').exists()
    (line,) = builds_log(s)
    assert line['ok'] is True and line['key'] == key and line['image_gb'] > 0.0009 and line['total_gb'] > 0
    assert 'scratch low' not in capsys.readouterr().err
    monkeypatch.setattr(ImageStore, 'free_gb', lambda self: 5.0)
    monkeypatch.setattr(ImageStore, 'total_gb', lambda self: 200.0)
    (tmp_path / 'ctx' / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN echo other\n')
    asyncio.run(s.build(tmp_path / 'ctx', label='ok-task-2'))
    assert 'scratch low: 5 of 200 GB free' in capsys.readouterr().err


def low_disk(monkeypatch):
    """statvfs of a 200 GB scratch with 5 GB free (below every eviction threshold), however much is deleted."""
    monkeypatch.setattr(builder.os, 'statvfs', lambda p: SimpleNamespace(f_bavail=5, f_frsize=10 ** 9, f_blocks=200))


def test_evict_removes_orphans_whose_lock_is_free_and_keeps_live_builds(tmp_path, monkeypatch):
    root = tmp_path / 'local'
    for name in ('orphan', 'building'):
        (root / 'images' / name / 'rootfs').mkdir(parents=True)
    (root / 'bases' / 'pulled-half' / 'rootfs').mkdir(parents=True)
    fresh = root / 'images' / 'fresh'
    (fresh / 'rootfs').mkdir(parents=True)
    (fresh / 'meta.json').write_text('{}')
    (root / 'locks').mkdir()
    held = open(root / 'locks' / 'img-building.lock', 'w')
    fcntl.flock(held, fcntl.LOCK_EX)  # a builder (another open file description) is working on it
    try:
        low_disk(monkeypatch)
        evict(root)
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    assert not (root / 'images' / 'orphan').exists()
    assert not (root / 'bases' / 'pulled-half').exists()
    assert (root / 'images' / 'building').exists(), 'a directory whose build lock is held must stay'
    assert fresh.exists(), 'an image used minutes ago stays (idle requirement)'


def test_offline_no_proxy_covers_private_names_and_ranges():
    online = proxy_env(['svc'], url='http://127.0.0.1:3128')
    offline = proxy_env(['svc'], url='http://127.0.0.1:3128', offline=True)
    for d in OFFLINE_DOMAINS:
        assert d not in online['no_proxy'].split(',')
        assert d in offline['no_proxy'].split(',') and f'*{d}' in offline['MAVEN_OPTS']
    assert '127.0.0.0/8' in offline['no_proxy'].split(',') and '127.0.0.0/8' not in online['no_proxy']
    assert offline['no_proxy'] == offline['NO_PROXY'] and 'svc' in offline['no_proxy'].split(',')
    assert '\\|*.internal' in offline['GRADLE_OPTS']
    assert '|10.*|' in offline['MAVEN_OPTS'] and '|172.31.*|' in offline['MAVEN_OPTS'] and '10.*' not in online['MAVEN_OPTS']


def test_hosts_names():
    text = '127.0.0.1\tlocalhost buildkitsandbox\n# comment api.x\n10.0.0.2 api.lab.internal db # db.alias\nbad\n'
    assert hosts_names(text) == ['localhost', 'buildkitsandbox', 'api.lab.internal', 'db']


def test_image_hosts_names_follows_a_symlinked_etc_hosts(tmp_path):
    from harbor_ns.environment import image_hosts_names
    (tmp_path / 'etc').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'hosts').write_text('127.0.0.1 localhost legacy.example.org\n')
    os.symlink('/data/hosts', tmp_path / 'etc' / 'hosts')
    assert image_hosts_names(tmp_path) == ['localhost', 'legacy.example.org']
    assert image_hosts_names(tmp_path / 'missing') == []


def test_offline_bypass_leaves_out_the_registries_the_package_proxy_serves(tmp_path, monkeypatch):
    from harbor_ns.environment import offline_bypass_names
    (tmp_path / 'etc').mkdir()
    (tmp_path / 'etc' / 'hosts').write_text('127.0.0.1 localhost pypi.org files.pythonhosted.org api.lab.internal\n')
    monkeypatch.setenv('HARBOR_NS_OFFLINE_PROXY_HOSTS', 'pypi.org,pythonhosted.org')
    assert offline_bypass_names(tmp_path) == ['localhost', 'api.lab.internal']
    monkeypatch.delenv('HARBOR_NS_OFFLINE_PROXY_HOSTS')
    assert 'pypi.org' in offline_bypass_names(tmp_path)

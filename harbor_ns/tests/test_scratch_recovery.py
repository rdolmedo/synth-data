"""#451 lesson 7: one image-eviction rule for every pipeline, and recovery when the scratch runs out of space.

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q harbor_ns/tests/test_scratch_recovery.py
"""
import asyncio
import errno
import json
import os
import time
from pathlib import Path

import pytest

from harbor_ns import builder
from harbor_ns.builder import BuildError, ImageMeta, ImageStore, evict, out_of_space
from harbor_ns.sandbox import RuntimeConfig

GB = 1e9


class _VFS:
    def __init__(self, free, total):
        self.f_bavail, self.f_frsize, self.f_blocks = int(free), 1, int(total)


class Disk:
    """statvfs of a scratch of `total_gb` with `free_gb` free; every deleted image frees `per_image_gb`."""

    def __init__(self, monkeypatch, free_gb, total_gb, per_image_gb=10):
        self.free, self.total, self.deleted = free_gb * GB, total_gb * GB, []
        real_rmtree = builder.shutil.rmtree

        def rmtree(p, ignore_errors=False):
            self.deleted.append(Path(p).name)
            self.free += per_image_gb * GB
            real_rmtree(p, ignore_errors=ignore_errors)
        monkeypatch.setattr(builder.os, 'statvfs', lambda p: _VFS(self.free, self.total))
        monkeypatch.setattr(builder.shutil, 'rmtree', rmtree)


def image(root, name, idle_h, sub='images', meta=True, in_use=False):
    d = root / sub / name
    (d / 'rootfs').mkdir(parents=True)
    if meta:
        (d / 'meta.json').write_text('{}')
        t = time.time() - idle_h * 3600
        os.utime(d / 'meta.json', (t, t))
    if in_use:   # a running trial's marker (ImageStore.mark_in_use: <dir>/.inuse/<token> holding the pid)
        (d / '.inuse').mkdir()
        (d / '.inuse' / 'tok').write_text(str(os.getpid()))
    return d


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv('HARBOR_NS_EVICT_IDLE_H', raising=False)
    r = tmp_path / 'local'
    for sub in ('images', 'bases', 'locks', 'logs'):
        (r / sub).mkdir(parents=True)
    return r


def pool(root):
    image(root, 'orphan', 0, meta=False)
    image(root, 'old', 6)
    image(root, 'mid', 1)
    image(root, 'recent', 5 / 60)
    image(root, 'busy', 9, in_use=True)
    image(root, 'base', 3, sub='bases')


def test_above_the_trigger_nothing_goes(root, monkeypatch):
    pool(root)
    disk = Disk(monkeypatch, 60, 200)        # trigger min(100, 25% of 200) = 50 GB
    assert evict(root)['removed'] == 0 and disk.deleted == []


def test_per_task_pool_frees_lru_down_to_the_10_min_floor(root, monkeypatch):
    """A 200 GB worker below 50 GB free: orphans, then least recently used, until 80 GB (40%) is free; the image used
    5 min ago and the one a running trial uses stay (the old rule kept every image younger than 5 h: ENOSPC)."""
    pool(root)
    disk = Disk(monkeypatch, 30, 200)
    out = evict(root)
    assert disk.deleted == ['orphan', 'old', 'base', 'mid']
    assert (root / 'images' / 'recent').exists() and (root / 'images' / 'busy').exists()
    assert out['removed'] == 3 and out['orphans'] == 1 and out['free_after_gb'] == 70.0
    line = json.loads((root / 'logs' / 'evict.jsonl').read_text().splitlines()[-1])
    assert line['removed'] == 3 and line['free_gb'] == 30.0 and line['idle_h_max'] > 5.9


def test_large_scratch_keeps_the_absolute_thresholds(root, monkeypatch):
    """On >= 400 GB the thresholds are aa-evals' (evict below 100 GB free, up to 160 GB), LRU first."""
    pool(root)
    disk = Disk(monkeypatch, 95, 1000, per_image_gb=40)
    evict(root)
    assert disk.deleted[:2] == ['orphan', 'old'] and (root / 'images' / 'mid').exists()   # 95 + 40 + 40 >= 160


def test_idle_floor_from_env(root, monkeypatch):
    pool(root)
    monkeypatch.setenv('HARBOR_NS_EVICT_IDLE_H', '2')     # m5 traces set 0.25; here: keep anything used within 2 h
    Disk(monkeypatch, 10, 200)
    evict(root)
    assert (root / 'images' / 'mid').exists() and not (root / 'images' / 'old').exists()


def test_force_after_enospc_goes_to_the_target_with_the_same_floor(root, monkeypatch):
    """force: above the trigger too, to the target; the 10 min floor still holds (task_synthesis' check sandbox and
    synth.prefix_bases use rootfs they never mark in use)."""
    pool(root)
    disk = Disk(monkeypatch, 60, 200)        # above the trigger (50), below the target (80)
    out = ImageStore.maybe_evict(type('S', (), {'local_root': root})(), force=True)
    assert out['force'] and disk.deleted[:2] == ['orphan', 'old'] and out['free_after_gb'] >= 80
    Disk(monkeypatch, 5, 200, per_image_gb=1)
    evict(root, force=True)
    assert (root / 'images' / 'recent').exists() and (root / 'images' / 'busy').exists()


def test_image_pinned_during_the_pass_is_kept(root, monkeypatch):
    """A long pass: a trial pins `mid` (ImageStore.claim) while older candidates are being deleted."""
    pool(root)
    disk = Disk(monkeypatch, 30, 200)
    real_rmtree = builder.shutil.rmtree

    def rmtree(p, ignore_errors=False):
        real_rmtree(p, ignore_errors=ignore_errors)
        if Path(p).name == 'old':
            store = type('S', (), {'touch': lambda self, d: None})()
            ImageStore.mark_in_use(store, root / 'images' / 'mid', 'trial')
    monkeypatch.setattr(builder.shutil, 'rmtree', rmtree)
    evict(root)
    assert not (root / 'images' / 'old').exists() and (root / 'images' / 'mid' / 'meta.json').exists()


def test_meta_put_back_when_pinned_between_unlink_and_recheck(root, monkeypatch):
    pool(root)
    Disk(monkeypatch, 30, 200)
    seen = {}
    real = ImageStore._in_use

    def in_use(d):
        if d.name != 'old':
            return real(d)
        seen['n'] = seen.get('n', 0) + 1
        return seen['n'] >= 3        # listing, pre-delete check: free; after meta.json was unlinked: pinned
    monkeypatch.setattr(ImageStore, '_in_use', staticmethod(in_use))
    evict(root)
    assert (root / 'images' / 'old' / 'meta.json').read_text() == '{}' and (root / 'images' / 'old' / 'rootfs').is_dir()


def test_claim_pins_only_an_image_that_is_still_there(tmp_path):
    s = _store(tmp_path)
    d = image(s.local_root, 'img', 1)
    assert s.claim(d, 'trial-1') and ImageStore._in_use(d)
    (d / 'meta.json').unlink()                # evicted after the build returned, before the pin
    assert not s.claim(d, 'trial-2') and not (d / '.inuse' / 'trial-2').exists()
    (d / '.inuse' / '.trial-3.tmp').write_text('')   # a marker being written is skipped, not dropped as dead
    assert ImageStore._in_use(d) and (d / '.inuse' / '.trial-3.tmp').exists()


def test_out_of_space_detection(tmp_path):
    assert out_of_space(OSError(errno.ENOSPC, 'No space left on device'))
    assert out_of_space(OSError(errno.EDQUOT, 'Disk quota exceeded'))
    try:
        try:
            raise OSError(errno.ENOSPC, 'x')
        except OSError as e:
            raise BuildError('RUN (line 4) failed with exit code 1') from e
    except BuildError as chained:
        assert out_of_space(chained)
    log = tmp_path / 'build.log'
    log.write_text('...\ntar: ./usr/lib/x.so: Cannot write: No space left on device\n')
    assert out_of_space(BuildError('RUN (line 4) failed with exit code 2'), log)
    log.write_text('E: Unable to locate package foo\n')
    assert not out_of_space(BuildError('RUN (line 4) failed with exit code 100'), log)


def _store(tmp_path):
    rt = RuntimeConfig(nsinit=tmp_path / 'nsinit', work_root=tmp_path / 'boxes', nsfake=tmp_path / 'nsfake.so')
    return ImageStore(local_root=tmp_path / 'local', shared_root=tmp_path / 'shared', rt=rt, nsfake=rt.nsfake,
                      network_rebuilds=0)


def test_build_that_ran_out_of_space_is_evicted_and_built_again(tmp_path, monkeypatch):
    calls, evictions = [], []

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        calls.append(d)
        (d / 'rootfs').mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            with open(log, 'a') as f:
                f.write('cp: error writing ./big: No space left on device\n')
            raise BuildError('RUN (line 2) failed with exit code 1')
        return ImageMeta()
    monkeypatch.setattr(ImageStore, '_build', _build)
    monkeypatch.setattr(ImageStore, 'maybe_evict',
                        lambda self, *a, force=False, **k: evictions.append(force) or {'removed': 2})
    monkeypatch.setattr(ImageStore, '_persist_squashfs', lambda *a: None)
    s = _store(tmp_path)
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN true\n')
    key, rootfs, _ = asyncio.run(s.build(ctx, label='t'))
    assert len(calls) == 2 and evictions[-1] is True and rootfs.is_dir()
    assert 'ran out of scratch space' in (s.image_dir(key) / 'build.log').read_text()


def test_build_failure_for_another_reason_is_not_retried(tmp_path, monkeypatch):
    calls = []

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        calls.append(d)
        with open(log, 'a') as f:
            f.write('E: Unable to locate package nope\n')
        raise BuildError('RUN (line 2) failed with exit code 1')
    monkeypatch.setattr(ImageStore, '_build', _build)
    s = _store(tmp_path)
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN false\n')
    with pytest.raises(BuildError):
        asyncio.run(s.build(ctx, label='t'))
    assert len(calls) == 1


def test_base_unpack_that_ran_out_of_space_is_retried_never_pulled(tmp_path, monkeypatch):
    """A local/ base whose shared squashfs unpack hits ENOSPC: evict and unpack again (TB-Sci-lite fell through to a
    docker.io pull of local/tasksynth-sci, "denied")."""
    s = _store(tmp_path)
    ref = 'local/tasksynth-sci:1'
    sq = s.shared_root / 'bases' / f'{builder._safe_name(ref)}.sqfs'
    sq.parent.mkdir(parents=True, exist_ok=True)
    sq.write_bytes(b'sqfs')
    sq.with_suffix('.json').write_text(json.dumps(ImageMeta(base=ref, oci={'x': 1}).to_json()))
    runs, evictions = [], []

    async def _run(self, argv, log, env=None, timeout=3600):
        runs.append(argv[0])
        if len(runs) == 1:
            Path(log).write_text('write failed: No space left on device\n')
            raise BuildError('unsquashfs failed (1)')
        Path(argv[argv.index('-d') + 1]).mkdir(parents=True)
    monkeypatch.setattr(ImageStore, '_run', _run)
    monkeypatch.setattr(ImageStore, 'maybe_evict',
                        lambda self, *a, force=False, **k: evictions.append(force) or {'removed': 2})
    rootfs, meta = asyncio.run(s.ensure_base(ref))
    assert runs == ['unsquashfs', 'unsquashfs'] and evictions[-1] is True and rootfs.is_dir() and meta.base == ref

    # a second ENOSPC: a clear BuildError, never a registry pull
    import shutil
    shutil.rmtree(s.local_root / 'bases' / builder._safe_name(ref))
    runs.clear()

    async def _run_full(self, argv, log, env=None, timeout=3600):
        runs.append(argv[0])
        Path(log).write_text('write failed: No space left on device\n')
        raise BuildError('unsquashfs failed (1)')
    monkeypatch.setattr(ImageStore, '_run', _run_full)
    with pytest.raises(BuildError, match='unpacking .* failed'):
        asyncio.run(s.ensure_base(ref))
    assert runs == ['unsquashfs', 'unsquashfs']


def test_no_second_try_when_the_eviction_freed_nothing(tmp_path, monkeypatch):
    calls = []

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        calls.append(d)
        with open(log, 'a') as f:
            f.write('No space left on device\n')
        raise BuildError('RUN (line 2) failed with exit code 1')
    monkeypatch.setattr(ImageStore, '_build', _build)
    monkeypatch.setattr(ImageStore, 'maybe_evict', lambda self, *a, force=False, **k: {'removed': 0, 'orphans': 0})
    s = _store(tmp_path)
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN true\n')
    with pytest.raises(BuildError):
        asyncio.run(s.build(ctx, label='t'))
    assert len(calls) == 1


def test_base_meta_write_that_runs_out_of_space_leaves_no_broken_entry(tmp_path, monkeypatch):
    """A meta.json write that fails (full disk) leaves no 0-byte file: the next call unpacks again."""
    s = _store(tmp_path)
    ref = 'local/x:1'
    sq = s.shared_root / 'bases' / f'{builder._safe_name(ref)}.sqfs'
    sq.parent.mkdir(parents=True, exist_ok=True)
    sq.write_bytes(b'sqfs')
    sq.with_suffix('.json').write_text(json.dumps(ImageMeta(base=ref, oci={'x': 1}).to_json()))

    async def _run(self, argv, log, env=None, timeout=3600):
        Path(argv[argv.index('-d') + 1]).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ImageStore, '_run', _run)
    monkeypatch.setattr(ImageStore, 'maybe_evict', lambda self, *a, force=False, **k: {})
    real_write = Path.write_text
    fail = {'n': 1}

    def write_text(self, data, *a, **kw):
        if self.name.startswith('.meta.json') and fail['n']:
            fail['n'] -= 1
            real_write(self, '', *a, **kw)
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real_write(self, data, *a, **kw)
    monkeypatch.setattr(Path, 'write_text', write_text)
    with pytest.raises(BuildError):
        asyncio.run(s.ensure_base(ref))
    metaf = s.local_root / 'bases' / builder._safe_name(ref) / 'meta.json'
    assert not metaf.exists() and not list(metaf.parent.glob('.meta.json*'))
    rootfs, meta = asyncio.run(s.ensure_base(ref))
    assert meta.base == ref and json.loads(metaf.read_text())['base'] == ref


def test_claim_on_a_full_disk_evicts_and_pins(tmp_path, monkeypatch):
    s = _store(tmp_path)
    d = image(s.local_root, 'img', 1)
    real, calls = ImageStore.mark_in_use, []

    def mark(self, image_dir, token):
        if not calls:
            calls.append('enospc')
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real(self, image_dir, token)
    monkeypatch.setattr(ImageStore, 'mark_in_use', mark)
    monkeypatch.setattr(ImageStore, 'maybe_evict', lambda self, *a, force=False, **k: calls.append(force) or {})
    assert s.claim(d, 'trial') and calls == ['enospc', True] and ImageStore._in_use(d)


def test_recovery_retries_when_another_process_freed_the_space(tmp_path, monkeypatch):
    """Concurrent ENOSPC: the first forced eviction frees the space; the others remove nothing but must still retry."""
    s = _store(tmp_path)
    monkeypatch.setattr(ImageStore, 'maybe_evict', lambda self, *a, force=False, **k: {'removed': 0, 'orphans': 0})
    monkeypatch.setattr(ImageStore, 'free_gb', lambda self: 80.0)
    assert asyncio.run(s._recover_space(0.2))['retry'] is True
    assert asyncio.run(s._recover_space(79.5))['retry'] is False     # nothing changed: no second try


def test_shared_image_unpack_out_of_space_twice_raises_never_rebuilds(tmp_path, monkeypatch):
    s = _store(tmp_path)
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN true\n')
    key = s.context_key(ctx, 'Dockerfile')
    sq = s.shared_root / 'images' / f'{key}.sqfs'
    sq.parent.mkdir(parents=True, exist_ok=True)
    sq.write_bytes(b'sqfs')
    sq.with_suffix('.json').write_text(json.dumps(ImageMeta().to_json()))
    built, runs = [], []

    async def _run(self, argv, log, env=None, timeout=3600):
        runs.append(argv[0])
        Path(log).write_text('write failed: No space left on device\n')
        raise BuildError('unsquashfs failed (1)')

    async def _build(self, *a, **k):
        built.append(1)
        return ImageMeta()
    monkeypatch.setattr(ImageStore, '_run', _run)
    monkeypatch.setattr(ImageStore, '_build', _build)
    monkeypatch.setattr(ImageStore, 'maybe_evict', lambda self, *a, force=False, **k: {'removed': 1})
    with pytest.raises(BuildError, match='ran out of scratch space'):
        asyncio.run(s.build(ctx))
    assert runs == ['unsquashfs', 'unsquashfs'] and built == []


def test_build_pinned_builds_again_when_evicted_before_the_pin_and_pinned_unpins(tmp_path, monkeypatch):
    s = _store(tmp_path)
    d = image(s.local_root, 'img', 0)
    builds = []

    async def build(self, ctx, dockerfile='Dockerfile', label='', **kw):
        builds.append(label)
        if len(builds) == 1:
            (d / 'meta.json').unlink()             # evicted by another process before the caller could pin it
        else:
            (d / 'meta.json').write_text('{}')     # loaded / built again
        return 'img', d / 'rootfs', ImageMeta()
    monkeypatch.setattr(ImageStore, 'build', build)
    key, rootfs, _ = asyncio.run(s.build_pinned(tmp_path, label='t', token='tok'))
    assert builds == ['t', 't'] and (d / '.inuse' / 'tok').exists()

    async def use():
        async with s.pinned(tmp_path, label='u') as (k, r, m):
            assert ImageStore._in_use(r.parent) and len(list((d / '.inuse').iterdir())) == 2
    asyncio.run(use())
    assert sorted(p.name for p in (d / '.inuse').iterdir()) == ['tok']   # the block's own pin is gone

    async def fail():
        async with s.pinned(tmp_path, label='v'):
            raise RuntimeError('boom')
    with pytest.raises(RuntimeError):
        asyncio.run(fail())
    assert sorted(p.name for p in (d / '.inuse').iterdir()) == ['tok']   # also when the block raises


def test_build_pinned_gives_up_after_its_attempts(tmp_path, monkeypatch):
    s = _store(tmp_path)
    d = image(s.local_root, 'img', 0, meta=False)

    builds = []

    async def build(self, ctx, dockerfile='Dockerfile', label='', **kw):
        builds.append(1)
        return 'img', d / 'rootfs', ImageMeta()
    monkeypatch.setattr(ImageStore, 'build', build)
    with pytest.raises(BuildError, match=r'could not be pinned \(evicted each time\)'):
        asyncio.run(s.build_pinned(tmp_path, label='t', token='tok'))
    assert len(builds) == 3 and not ImageStore._in_use(d)
    monkeypatch.setattr(builder, '_free_gb', lambda p: 0.2)
    with pytest.raises(BuildError, match='No space left on device'):    # task_synthesis reads it as infra
        asyncio.run(s.build_pinned(tmp_path, label='t', token='tok'))


def test_build_pinned_cancelled_during_the_claim_leaves_no_pin(tmp_path, monkeypatch):
    import threading
    s = _store(tmp_path)
    d = image(s.local_root, 'img', 0)
    go = threading.Event()
    real = ImageStore.claim

    def slow_claim(self, image_dir, token):
        go.wait(5)                                 # e.g. a forced eviction waiting on evict.lock
        return real(self, image_dir, token)

    async def build(self, ctx, dockerfile='Dockerfile', label='', **kw):
        return 'img', d / 'rootfs', ImageMeta()
    monkeypatch.setattr(ImageStore, 'build', build)
    monkeypatch.setattr(ImageStore, 'claim', slow_claim)

    async def run():
        task = asyncio.ensure_future(s.build_pinned(tmp_path, label='t', token='tok'))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        go.set()
        for _ in range(100):                       # the claim finishes in its thread, then the callback unpins
            await asyncio.sleep(0.02)
            if (d / '.inuse').exists() and not any((d / '.inuse').iterdir()):
                break
    asyncio.run(run())
    assert not ImageStore._in_use(d)

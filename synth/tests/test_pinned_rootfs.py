"""#451 lesson 7 follow-up: synthesis code that uses a harbor_ns image's rootfs outside a harbor trial (task_synthesis'
check and author sandboxes, prefix and task-synthesis base publishing) pins it (ImageStore.build_pinned), so the
eviction at other trials' starts cannot delete it while it is in use.

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q synth/tests/test_pinned_rootfs.py
"""
import ast
import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# files whose rootfs users must pin, and the functions where they build
USERS = {
    'task_synthesis/pipeline/check.py': ['check'],
    'task_synthesis/pipeline/explore.py': ['author_base', '_enter'],
    'synth/prefix_bases.py': ['build_one'],
    'task_synthesis/base_image/build_base.py': ['build'],
}


def _calls(fn):
    return [n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


@pytest.mark.parametrize('path', sorted(USERS))
def test_rootfs_users_build_pinned(path):
    tree = ast.parse((REPO / path).read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in USERS[path]:
        calls = _calls(fns[name])
        assert 'build_pinned' in calls, f'{path}:{name} builds without pinning'
        assert 'build' not in calls, f'{path}:{name} still calls store.build() for a rootfs it uses'
    # pins end with the use (a one-shot job's pin, task_synthesis/base_image, and explore's shared author base last as
    # long as the process)
    if path != 'task_synthesis/base_image/build_base.py':
        assert 'unmark_in_use' in (REPO / path).read_text(), f'{path} pins without ever unpinning'


def test_author_base_is_pinned_for_the_process(monkeypatch):
    from task_synthesis.pipeline import explore
    calls = []

    class Store:
        def context_key(self, ctx, dockerfile):
            return 'k'

        async def build_pinned(self, ctx, dockerfile='Dockerfile', label='', *, token, **kw):
            calls.append((label, token))
            return 'k', Path('/x/rootfs'), object()
    monkeypatch.setattr(explore, '_BASE', {})
    monkeypatch.setattr(explore, '_BASE_LOCK', [])
    monkeypatch.setattr(explore, '_IMAGE_USERS', {})
    store = Store()
    asyncio.run(explore.author_base(store))
    asyncio.run(explore.author_base(store))      # built and pinned once per process
    assert len(calls) == 1 and calls[0][0] == 'explore-base' and calls[0][1].startswith('explore-base-')


def test_a_pin_failure_is_infra_never_the_generators_build(tmp_path, monkeypatch):
    from harbor_ns.builder import BuildError
    from task_synthesis.pipeline import check as chk, round as rnd

    class Store:
        def context_key(self, ctx, dockerfile):
            return 'k'

        def image_dir(self, key):
            return tmp_path / 'img'

        async def build_pinned(self, *a, **kw):
            raise BuildError('label: image could not be pinned (evicted each time)')
    monkeypatch.setattr(chk, '_IMAGE_USERS', {})
    monkeypatch.setattr(chk, 'failed_build_log', lambda store, label: '')
    monkeypatch.setattr(chk, 'drop_dir', lambda d: None)
    task = tmp_path / 'task'
    (task / 'environment').mkdir(parents=True)
    (task / 'environment' / 'Dockerfile').write_text('FROM x\n')
    (task / 'task.toml').write_text('[environment]\n')
    (task / 'tasksynth.json').write_text('{"meta": {}}')
    res = asyncio.run(chk.check(Store(), task))
    assert res['infra'].startswith('image pin failed') and not res['errors']
    assert 'could not be pinned' in rnd.INFRA_MARKS

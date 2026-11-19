"""A canary registry ($TASKSYNTH_SERVING_ROOT) never reaches production readers in the task-synthesis pipeline: every
reader (llm clients of the round, bind and cohort jobs, the driver, live, oneshot_corpus) resolves the variable's
registry; the driver pins its process to the run's registry (run.json, recorded at plan) and refuses another one; every
job it starts (driver, producer, round, bind) gets the root forwarded (condor jobs run with getenv = false). Unset: the
submit files and run.json are as before. Nothing is submitted."""
import ast
import asyncio
import json
import os
import time
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

from task_synthesis.common import REPO
from task_synthesis.pipeline import llm, production as P, round as R
from task_synthesis.tests.test_fill_shards import live as fill_live
from tbshared import serving

NAME = 'canary-x'


def entry(d: Path, served: str):
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (d / 'endpoints.json').write_text(json.dumps({'served_name': served, 'urls': ['http://127.0.0.1:9/v1'],
                                                  'updated_at': now, 'healthy_at': now, 'request': {'overrides': {}}}))
    (d / 'job.json').write_text(json.dumps({'job_id': f'{served}.0'}))


@pytest.fixture
def two_registries(tmp_path, monkeypatch):
    """A 'production' registry (serving.ROOT patched to it) and a canary one, both holding a healthy NAME: a reader
    that ignores the variable gets 'production'."""
    prod, canary = tmp_path / 'prod', tmp_path / 'canary'
    entry(prod / NAME, 'production')
    entry(canary / NAME, 'canary')
    entry(canary / 'canary-y', 'canary')
    monkeypatch.setattr(serving, 'ROOT', prod)
    monkeypatch.setattr(P, 'SERVING', None)
    monkeypatch.setenv(serving.ROOT_ENV, str(canary))
    return prod, canary


def env_of(sub: str) -> str:
    return next(x for x in sub.splitlines() if x.startswith('environment = '))


# ------------------------------------------------------------------------------------------------ readers
def test_every_reader_resolves_the_canary(two_registries, monkeypatch):
    prod, canary = two_registries
    assert llm.Endpoint(NAME).model == 'canary'
    assert llm.Endpoint(f'{NAME}+canary-y').model == 'canary'
    asyncio.run(llm.wait_healthy('missing+canary-y', timeout=5))               # canary-y: only in the canary
    assert P.serving_dir() == canary and P.healthy(NAME)
    assert P.healthy('canary-y') and not (prod / 'canary-y').exists()
    from task_synthesis import common
    monkeypatch.setattr(common, 'METADATA_GLOBS', ())             # not the real /data metadata
    assert any(k.startswith(str(canary)) for k in common.recent_metadata(1))
    monkeypatch.delenv(serving.ROOT_ENV)                         # unset: production, as before
    assert llm.Endpoint(NAME).model == 'production' and P.serving_dir() == prod
    assert not any(k.startswith(str(canary)) for k in common.recent_metadata(1))


def allowed_root_use(rel, tree, node):
    """serving.ROOT is fine as a comparison (is this the production registry?), never as a registry to read."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and any(x is node for x in ast.walk(n)):
            return True
    return False


def test_no_module_pins_the_production_registry():
    """No task_synthesis module names the production registry in code or joins a path onto ROOT / 'serving' or a
    SERVING constant. Allowed: common.METADATA_GLOBS (checkout cleanup lists production's files; recent_metadata adds
    the variable's) and serve_teacher's own default (it honours the variable, #495)."""
    bad = []
    for p in sorted((REPO / 'task_synthesis').rglob('*.py')):
        rel = str(p.relative_to(REPO))
        if '/tests/' in rel or p.name.startswith('test_'):
            continue
        tree = ast.parse(p.read_text())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
                    and str(serving.ROOT) in n.value and rel != 'task_synthesis/common.py'):
                bad.append(f'{rel}:{n.lineno} names the production registry')
            if (isinstance(n, ast.Attribute) and n.attr == 'ROOT' and isinstance(n.value, ast.Name)
                    and n.value.id in ('serving', 'tbserving') and not allowed_root_use(rel, tree, n)):
                bad.append(f'{rel}:{n.lineno} uses serving.ROOT as a registry (use registry_root())')
            if isinstance(n, ast.Call) and any(isinstance(x, ast.Constant) and x.value == 'serving' for x in n.args) \
                    and any(isinstance(x, ast.Name) and x.id in ('ROOT', 'TS_ROOT') for x in n.args):
                bad.append(f'{rel}:{n.lineno} builds ROOT + "serving" (use tbshared.serving)')
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
                left = n.left.id if isinstance(n.left, ast.Name) else n.left.attr if isinstance(n.left, ast.Attribute) else None
                if left == 'SERVING':
                    bad.append(f'{rel}:{n.lineno} joins onto SERVING (use serving_dir())')
                if (isinstance(n.right, ast.Constant) and n.right.value == 'serving' and left in ('ROOT', 'TS_ROOT')
                        and rel != 'task_synthesis/serve/serve_teacher.py'):
                    bad.append(f'{rel}:{n.lineno} joins ROOT / "serving" (use tbshared.serving)')
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ the driver
def test_driver_pins_its_process_to_the_runs_registry(tmp_path, monkeypatch):
    canary = str(tmp_path / 'canary')
    run = {'name': 'c1', 'serving_env': {serving.ROOT_ENV: canary}}
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    assert P.pin_run_root(run) == {serving.ROOT_ENV: canary} and os.environ[serving.ROOT_ENV] == canary
    assert P.pin_run_root(run) == {serving.ROOT_ENV: canary}                     # same root: fine
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'other'))
    with pytest.raises(SystemExit, match='planned on registry'):
        P.pin_run_root(run)                                                     # another canary: refused
    with pytest.raises(SystemExit, match='planned on registry'):
        P.pin_run_root({'name': 'prod'})                                        # a production run under a canary
    monkeypatch.setenv(serving.ROOT_ENV, str(serving.ROOT))
    assert P.pin_run_root({'name': 'prod'}) == {} and serving.ROOT_ENV not in os.environ   # production named
    assert P.pin_run_root({'name': 'prod'}) == {}


def sandbox(monkeypatch, tmp_path, info):
    for k in ('RUNS', 'ROUNDS', 'STREAMS', 'BINDS', 'SYNTH'):
        monkeypatch.setattr(P, k, tmp_path / k.lower())
    (tmp_path / 'runs' / info['name']).mkdir(parents=True)
    (tmp_path / 'runs' / info['name'] / 'run.json').write_text(json.dumps(info))


RUN = {'name': 'c1', 'round': 'c1', 'bind': 'c1', 'stream': '/tmp/none', 'campaign_id': 'x', 'servers': [NAME],
       'config': {'quota': {'files_per_task': 1, 'campaign_loose_files': 1}}, 'tasks_needed': 1}


@pytest.mark.parametrize('canary', [False, True], ids=['off', 'on'])
def test_driver_and_producer_jobs_get_the_runs_root(tmp_path, monkeypatch, canary):
    root = str(tmp_path / 'canary')
    info = {**RUN, **({'serving_env': {serving.ROOT_ENV: root}} if canary else {})}
    sandbox(monkeypatch, tmp_path, info)
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)                         # a shell without the variable
    monkeypatch.setattr(P, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(P, 'fast_files', lambda path='/data': None)
    monkeypatch.setattr(P.subprocess, 'run', lambda *a, **k: Namespace(stdout='1 job(s) submitted to cluster 7.'))
    P.launch(Namespace(name='c1', bid=35))
    drv = env_of((tmp_path / 'runs' / 'c1' / 'driver.sub').read_text())
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)                         # `drive` from a fresh shell
    d = P.Driver('c1')
    assert os.environ.get(serving.ROOT_ENV) == (root if canary else None)       # the driver's own process
    if canary:
        assert P.serving_dir() == Path(root)
        monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'another'))
        with pytest.raises(SystemExit, match='planned on registry'):
            P.Driver('c1')                                                      # a shell naming another registry
        monkeypatch.setenv(serving.ROOT_ENV, root)
    d.sh = lambda argv, what: '1 job(s) submitted to cluster 8.'
    d.stream.mkdir(parents=True, exist_ok=True)
    d.producer_up()
    prod = env_of((tmp_path / 'runs' / 'c1' / 'producer.sub').read_text())
    tail = f' {serving.ROOT_ENV}={root}"' if canary else '"'
    assert drv == f'environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={tmp_path / "code"} PYTHONNOUSERSITE=1{tail}'
    assert prod == f'environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin{tail}'


def test_plan_records_the_root_only_for_another_registry(tmp_path, monkeypatch):
    """run.json carries serving_env only for a registry other than production: unset, or set to the production path,
    the run's files are as before."""
    assert serving.to_record() == {}
    monkeypatch.setenv(serving.ROOT_ENV, str(serving.ROOT))
    assert serving.to_record() == {}
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'canary'))
    assert serving.to_record() == {serving.ROOT_ENV: str(tmp_path / 'canary')}
    with pytest.raises(SystemExit):                                 # recorded values are validated when read back
        P.run_root_env({'serving_env': {serving.ROOT_ENV: '/a b'}})


@pytest.mark.parametrize('module', ['round', 'bind'])
def test_hand_submitted_round_or_bind_of_a_run_gets_the_runs_root(tmp_path, monkeypatch, module):
    """The driver's ALERT says to `round submit` again by hand: from a fresh shell, the job still gets the run's root."""
    root = str(tmp_path / 'canary')
    sandbox(monkeypatch, tmp_path, {**RUN, 'serving_env': {serving.ROOT_ENV: root}})
    fill_live(monkeypatch)
    monkeypatch.setattr(R, 'ROUNDS', tmp_path / 'rounds')
    monkeypatch.setattr(R, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(R, 'cheap_split', lambda n, a, *shape: 0)
    monkeypatch.setattr(R.subprocess, 'run', mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.\n')))
    from task_synthesis.assets import bind as B
    monkeypatch.setattr(B, 'BINDS', tmp_path / 'binds')
    monkeypatch.setattr(B, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(B.subprocess, 'run', mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 78.\n')))
    if module == 'round':
        R.submit(Namespace(round='c1', cpus=32, memory_gib=96, bid=56, cheap_bid=0, jobs=1, disk_gib=200))
        sub = tmp_path / 'rounds' / 'c1' / 'job.sub'
    else:
        B.submit(Namespace(name='c1', round='c1', specs=None, strategy='s', llm=NAME, concurrency=1, http_concurrency=1,
                           tries=1, seed=0, warm_from='none', grow_now=0, ramp_minutes=0, ramp_start=1, gh_share=0,
                           cpus=1, memory_gib=4, bid=26))
        sub = tmp_path / 'binds' / 'c1' / 'job.sub'
    assert f"{serving.ROOT_ENV}='{root}'" in env_of(sub.read_text())
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'another'))         # a shell naming another registry: refused
    with pytest.raises(SystemExit, match='planned on registry'):
        R.submit(Namespace(round='c1', cpus=32, memory_gib=96, bid=56, cheap_bid=0, jobs=1, disk_gib=200)) \
            if module == 'round' else B.submit(Namespace(name='c1'))


# ------------------------------------------------------------------------------------------------ round and bind jobs
@pytest.mark.parametrize('canary', [False, True], ids=['off', 'on'])
def test_round_and_bind_jobs_get_the_root(tmp_path, monkeypatch, canary):
    root = str(tmp_path / 'canary')
    if canary:
        monkeypatch.setenv(serving.ROOT_ENV, root)
    else:
        monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    fill_live(monkeypatch)
    monkeypatch.setattr(P, 'RUNS', tmp_path / 'runs')                          # no production run of these names
    monkeypatch.setattr(R, 'ROUNDS', tmp_path / 'rounds')
    monkeypatch.setattr(R, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(R, 'cheap_split', lambda n, a, *shape: 0)
    run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.\n'))
    monkeypatch.setattr(R.subprocess, 'run', run)
    R.submit(Namespace(round='r1', cpus=32, memory_gib=96, bid=56, cheap_bid=0, jobs=1, disk_gib=200))
    rnd = env_of((tmp_path / 'rounds' / 'r1' / 'job.sub').read_text())
    from task_synthesis.assets import bind as B
    monkeypatch.setattr(B, 'BINDS', tmp_path / 'binds')
    monkeypatch.setattr(B, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(B.subprocess, 'run', mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 78.\n')))
    B.submit(Namespace(name='b1', round='r1', specs=None, strategy='s', llm=f'{NAME}+canary-y', concurrency=1,
                       http_concurrency=1, tries=1, seed=0, warm_from='none', grow_now=0, ramp_minutes=0, ramp_start=1,
                       gh_share=0, cpus=1, memory_gib=4, bid=26))
    bnd = env_of((tmp_path / 'binds' / 'b1' / 'job.sub').read_text())
    for e in (rnd, bnd):
        assert (f"{serving.ROOT_ENV}='{root}'" in e) == canary, e
        assert serving.ROOT_ENV in e or not canary

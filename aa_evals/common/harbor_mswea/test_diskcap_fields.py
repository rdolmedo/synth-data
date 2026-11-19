"""#405: harbor_mswea writes the run's "diskcap" (run.py) and each attempt's cap env and outcome fields (worker.py)."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from aa_evals.common import diskcap
from aa_evals.common.harbor_mswea import run, worker


@pytest.fixture
def new_run(tmp_path, monkeypatch):
    code = tmp_path / 'code'
    code.mkdir()
    (code / '.complete').write_text(json.dumps({'git_sha': 'abc'}))
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'snapshot_code', lambda *a: str(code))
    monkeypatch.setattr(run, 'list_tasks', lambda src: {'rstan-to-pystan': {}, 'sam-cell-seg': {}, 'other': {}})
    monkeypatch.setattr(run, 'bundle_path', lambda v: tmp_path / 'no-bundle.tar')
    # never submit from a test: resuming an existing run submits workers even with --no-submit
    submitted = []
    monkeypatch.setattr(run, 'submit_workers', lambda r, cfg, n: submitted.append((r.name, n)) or [])
    monkeypatch.setattr(run, 'workers_alive', lambda r: False)

    def no_condor(*a, **k):
        raise AssertionError(f'a test ran a subprocess: {a!r}')
    monkeypatch.setattr(run.subprocess, 'run', no_condor)
    monkeypatch.setattr(run.subprocess, 'check_output', no_condor)

    def make(run_id, *args, defaults=None):
        run.main(['--bench', 'b', '--run-id', run_id, '--agent', 'oracle', '--tasks', 'x', '--no-submit', *args],
                 defaults)
        return json.loads((tmp_path / 'runs' / 'b' / run_id / 'config.json').read_text())
    make.submitted = submitted
    return make


def test_run_config_records_the_cap(new_run):
    assert new_run('default')['diskcap'] == diskcap.DEFAULT                     # auto without a family: 3:30720
    assert new_run('off', '--diskcap', 'off')['diskcap'] is None
    assert new_run('two', '--diskcap', '2:1000')['diskcap'] == {'factor': 2.0, 'min_mb': 1000}
    fam = new_run('fam', defaults={'diskcap_family': 'tb2.1'})['diskcap']       # eval's per-task TB 2.1 caps
    assert fam == {'factor': 1.0, 'min_mb': 32768, 'tasks': {'rstan-to-pystan': 92160, 'sam-cell-seg': 40960},
                   'one_per_worker': ['rstan-to-pystan']}
    with pytest.raises(SystemExit):
        new_run('bad', '--diskcap', '3:-1')
    with pytest.raises(SystemExit):
        new_run('smuggled', '--harbor-extra-args', '--ek ns_diskcap=0')
    with pytest.raises(SystemExit):                                              # a stored run keeps its policy
        new_run('off', '--diskcap', '3:30720')
    assert new_run.submitted == []
    new_run('fam', '--diskcap', 'auto', defaults={'diskcap_family': 'tb2.1'})   # same policy: resumes
    assert new_run.submitted == []                                               # --no-submit holds on resume too


class _Q:
    def __init__(self):
        self.done = {}

    def heartbeat(self, k):
        pass

    def finish(self, k, outcome):
        self.done[k] = outcome


def _attempt(tmp_path, monkeypatch, cfg, task='rstan-to-pystan'):
    """One worker.run_item with harbor replaced by a script that records the cap env and writes the box summary."""
    monkeypatch.setattr(worker, 'SCRATCH', tmp_path / 'scratch')
    monkeypatch.setenv('HARBOR_NS_DISKCAP_SPARE', 'python3')   # the worker's own env: never passed on
    monkeypatch.setenv('HARBOR_NS_DISKCAP', '9')
    w = object.__new__(worker.Worker)
    w.cfg = {'agent': 'oracle', 'code': str(tmp_path), 'image_cache': '/x', 'bundle_root': '/x', **cfg}
    w.name, w.q, w.running, w.tasks_dir, w.res = 'w', _Q(), {}, tmp_path / 'tasks', {}
    k = f'{task}__r0__t0'
    w.running[k] = {}
    script = ("import json, os, pathlib, sys; raw = pathlib.Path(sys.argv[1]);"
              "(raw / 'env.json').write_text(json.dumps({x: v for x, v in os.environ.items() if 'DISKCAP' in x}));"
              "log = os.environ.get('HARBOR_NS_DISKCAP_LOG');"
              "log and pathlib.Path(log).write_text(json.dumps({'limit_mb': 92160.0, 'peak_mb': 12.5, "
              "'largest_file_mb': 3.0, 'kills': [{'pid': 7}], 'at_fsize': []}))")
    w.harbor_cmd = lambda k_, task_dir, raw, api_base: [sys.executable, '-c', script, str(raw)]
    seen = {}

    def pack(k_, raw, trial):
        seen['env'] = json.loads((raw / 'env.json').read_text())
        return {}
    w.pack = pack
    asyncio.run(w.run_item(k, sorted(os.sched_getaffinity(0))[:1]))
    return w.q.done[k], seen['env']


def test_worker_env_and_outcome_fields(tmp_path, monkeypatch):
    cfg = {'diskcap': diskcap.auto('tb2.1', ['rstan-to-pystan', 'other'])}
    out, env = _attempt(tmp_path, monkeypatch, cfg)
    assert env['HARBOR_NS_DISKCAP'] == '1' and env['HARBOR_NS_DISKCAP_MIN_MB'] == '92160'
    assert env['HARBOR_NS_DISKCAP_LOG'].endswith('rstan-to-pystan__r0__t0/diskcap.json')
    assert 'HARBOR_NS_DISKCAP_SPARE' not in env
    assert out['diskcap'] == {'factor': 1.0, 'min_mb': 92160}
    assert (out['diskcap_limit_mb'], out['sandbox_disk_peak_mb'], out['sandbox_largest_file_mb'],
            out['diskcap_kills'], out['diskcap_at_fsize']) == (92160.0, 12.5, 3.0, 1, 0)


def test_worker_cap_off(tmp_path, monkeypatch):
    for cfg in ({'diskcap': None}, {}):   # off, and a run from before #405 (no key)
        out, env = _attempt(tmp_path, monkeypatch, cfg)
        assert env == {'HARBOR_NS_DISKCAP': '0'}                       # explicit: the worker's '9' cannot turn it on
        assert out['diskcap'] is None and out['diskcap_limit_mb'] is None

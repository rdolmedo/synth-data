"""CritPt's registry reader and condor job follow $TASKSYNTH_SERVING_ROOT (aa_evals/common/test_serving_root.py covers
the other suites; this file needs the stirrup venv, which critpt.run imports)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import pytest

from aa_evals.common import condor_cost, serving
from aa_evals.common.test_response_budget_env import parse_environment
from aa_evals.critpt import run as critpt


def test_resolve_model_reads_the_variables_registry(tmp_path, monkeypatch):
    d = tmp_path / 'regroot-test-x'
    d.mkdir()
    d.joinpath('endpoints.json').write_text(json.dumps({'served_name': 'from-env', 'urls': ['http://h:1/v1'],
                                                        'updated_at': time.time()}))
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path))
    assert critpt.resolve_model('serving:regroot-test-x')['served_name'] == 'from-env'


@pytest.mark.parametrize('root', [None, '/data/x/serving'], ids=['off', 'on'])
def test_submit_condor_forwards_the_root(tmp_path, monkeypatch, root):
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    if root:
        monkeypatch.setenv(serving.ROOT_ENV, root)
    subs = []
    monkeypatch.setattr(condor_cost, 'submit', lambda path, bid=None: subs.append(Path(path).read_text()))
    sha = 'c' * 40
    monkeypatch.setattr(critpt, 'RUNS', tmp_path / 'critpt')
    monkeypatch.setattr(critpt, 'git_sha', lambda: sha)
    monkeypatch.setattr(critpt.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=''))
    (tmp_path / 'critpt' / '_code' / sha[:12]).mkdir(parents=True)
    critpt.submit_condor(argparse.Namespace(run_id='r1', cpus=1, mem_gb=2, cpu_bid=None), ['--run-id', 'r1', '--condor'])
    env = parse_environment(subs[-1])
    assert env.get(serving.ROOT_ENV) == root and env['CRITPT_CODE_SHA'] == sha

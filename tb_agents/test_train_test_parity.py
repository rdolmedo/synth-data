"""Train/test parity of the Pi finish notice (#480; #375 #395 #428 #429): every pipeline freezes the same v4 record
from the one context guard, and its own attempt environment builder turns that record into the same PI_FINISH_*.

  task synthesis   synth/config.py finish_notice_config('v4')     -> synth/worker.py attempt_env
  AA evals / M5    aa_evals/common/pi_eval/run.py finish_notice()  -> PiEvalWorker.env (eval_q35 Worker.env + pi_eval's)
  eval_q35         eval_q35/common.py finish_notice()              -> eval_q35/worker.py Worker.env
All three default to 40000:16384 (WARN_AT:RESERVE). The response budget (#406, tb_agents/response_budget*.py) is a
separate, opt-in layer (PI_RESPONSE_BUDGET; aa_evals campaign "capped" and M5 finishing data only).
"""
from argparse import Namespace
import hashlib
import importlib.util
from pathlib import Path
import tempfile

import pytest

from tb_agents import pi_context_limit as G

REPO = Path(__file__).resolve().parents[1]
FINISH = ('PI_FINISH_WARN_AT', 'PI_FINISH_RESERVE', 'PI_FINISH_STICKY')
WANT = {'PI_FINISH_WARN_AT': '40000', 'PI_FINISH_RESERVE': '16384', 'PI_FINISH_STICKY': '1'}


def _module(name, path):
    """eval_q35's modules by file, under a unique name (a bare `import common` could pick up another package's)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_finish_env(monkeypatch):
    for key in FINISH:
        monkeypatch.delenv(key, raising=False)


def records():
    from synth.config import finish_notice_config
    from aa_evals.common.pi_eval import run as pi_eval_run
    eval_q35_common = _module('_parity_eval_q35_common', REPO / 'eval_q35' / 'common.py')
    return {'synth': finish_notice_config('v4'),
            'pi_eval': pi_eval_run.finish_notice(Namespace(finish_notice=None, agent='pi')),
            'eval_q35': eval_q35_common.finish_notice(None, 'pi')}


def test_every_pipeline_freezes_the_same_v4_record():
    r = records()
    assert r['synth'] == r['pi_eval'] == r['eval_q35'] == G.finish_notice_record(40000, 16384)
    assert r['synth']['policy'] == G.STICKY_POLICY and r['synth']['sticky'] is True
    assert r['synth']['text_sha256'] == hashlib.sha256((G.FINISH_WARN + '\n' + G.FINISH_FINAL).encode()).hexdigest()


def finish(env):
    return {k: env[k] for k in FINISH if k in env}


def test_synth_attempt_env():
    from synth.tests.test_traintest import campaign, item
    from synth.worker import attempt_env
    with tempfile.TemporaryDirectory() as d:
        env = attempt_env(campaign(d), item(), {}, Path(d) / 'raw' / 'u')
    assert finish(env) == WANT


def test_eval_q35_worker_env(tmp_path):
    from eval_q35 import worker
    cfg = {'served': 'm', 'protocol': {'finish_notice': records()['eval_q35']}}
    me = worker.Worker.__new__(worker.Worker)
    me.config, me.run = cfg, tmp_path
    me.cut_file = lambda k: tmp_path / f'{k}.cut'
    assert finish(me.env('task__a0__r0', 'task', 0)) == WANT


def test_pi_eval_worker_env(tmp_path):
    from aa_evals.common.pi_eval.worker import PiEvalWorker
    cfg = {'served': 'm', 'benchmark': 'tb21', 'protocol': {'finish_notice': records()['pi_eval']}}
    me = PiEvalWorker.__new__(PiEvalWorker)
    me.config, me.run, me.urls = cfg, tmp_path, []
    me.cut_file = lambda k: tmp_path / f'{k}.cut'
    assert finish(me.env('task__a0__r0', 'task', 0)) == WANT


@pytest.mark.parametrize('profile', ['eval', 'synth'])
def test_launcher_records_the_guard_policy_for_both_profiles(profile):
    from gen.pi_cont16k import guard_policy
    env = {'PI_CONT16K_PROFILE': profile, **WANT}
    assert guard_policy(env) == {'context_policy': G.STICKY_POLICY,
                                 'finish_notice': {'warn_at': 40000, 'reserve': 16384, 'sticky': True}}

"""M3.2: per-model request settings for the Pi protocol (run.py resolve_request, the proxy merge).

  python -m pytest -q aa_evals/common/pi_eval/test_request.py
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from aa_evals.common.pi_eval.run import aa_slug_for, resolve_request

REPO = Path(__file__).resolve().parents[3]
QWEN = {'client_model': 'hosted_vllm/qwen38-27b', 'pi_thinking': 'xhigh', 'overrides': {
    'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0,
    'reasoning_effort': 'xhigh', 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True,
                                                          'reasoning_effort': 'xhigh'}}}
SUPER = {'client_model': 'hosted_vllm/nemotron-3.5-super-vl', 'pi_thinking': 'high',
         'adapters': ['inline_reasoning_history'], 'overrides': {
             'temperature': 1.0, 'top_p': 0.95, 'chat_template_kwargs': {
                 'enable_thinking': True, 'medium_effort': False, 'truncate_history_thinking': False}}}


def ep(req):
    return {'request': req, 'registry': '/x/endpoints.json'}


def test_effort_replaces_top_level_and_template_kwarg():
    r = resolve_request(ep(QWEN), '', '', 'medium')
    assert r['overrides']['reasoning_effort'] == 'medium'
    assert r['overrides']['chat_template_kwargs']['reasoning_effort'] == 'medium'
    assert r['overrides']['top_k'] == 20 and r['pi_thinking'] == 'xhigh'
    assert QWEN['overrides']['reasoning_effort'] == 'xhigh'   # the registry dict is not mutated


def test_registry_effort_kept_without_flag():
    assert resolve_request(ep(QWEN), '', '', None)['overrides']['reasoning_effort'] == 'xhigh'


def test_super_has_no_effort_and_keeps_adapter():
    r = resolve_request(ep(SUPER), '', '', None)
    assert 'reasoning_effort' not in r['overrides'] and r['adapters'] == ['inline_reasoning_history']
    # an explicit effort goes on top level only: Super's template kwargs have no reasoning_effort
    r = resolve_request(ep(SUPER), '', '', 'high')
    assert r['overrides']['reasoning_effort'] == 'high'
    assert 'reasoning_effort' not in r['overrides']['chat_template_kwargs']


def test_extra_overrides_merge_template_kwargs():
    r = resolve_request(ep(QWEN), '', '{"temperature": 0.6, "chat_template_kwargs": {"preserve_thinking": false}}', None)
    assert r['overrides']['temperature'] == 0.6
    assert r['overrides']['chat_template_kwargs'] == {'enable_thinking': True, 'preserve_thinking': False,
                                                      'reasoning_effort': 'xhigh'}


def test_no_settings_is_an_error():
    with pytest.raises(SystemExit):
        resolve_request({'request': {}}, '', '', 'medium')


def test_unknown_adapter_is_an_error():
    with pytest.raises(SystemExit):
        resolve_request(ep({**SUPER, 'adapters': ['nope']}), '', '', None)


def test_preset_file(tmp_path):
    f = tmp_path / 't.toml'
    f.write_text('[request]\npi_thinking = "max"\noverrides = { temperature = 1.0, reasoning_effort = "max" }\n')
    r = resolve_request(None, str(f), '', None)
    assert r['overrides'] == {'temperature': 1.0, 'reasoning_effort': 'max'} and r['source'] == str(f)


def test_aa_slugs():
    assert aa_slug_for('qwen38-27b', 'medium') == 'qwen3-8-27b-medium'
    assert aa_slug_for('qwen38-27b', 'xhigh') == 'qwen3-8-27b'
    assert aa_slug_for('deepseek-v41-flash-1017', 'max') == 'deepseek-v4-1-flash-1017'
    assert aa_slug_for('nemotron-3.5-super-vl', '') == 'nemotron-3-5-super-vl'
    assert aa_slug_for('something-else', 'medium') is None


def _proxy():
    spec = importlib.util.spec_from_file_location('sticky_proxy_t', REPO / 'inference' / 'sticky_proxy.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['sticky_proxy_t'] = m
    spec.loader.exec_module(m)
    return m


def test_proxy_merge_and_inline_adapter():
    sp = _proxy()
    payload = {'reasoning_effort': 'max', 'temperature': 0.2,
               'chat_template_kwargs': {'clear_thinking': False, 'x': 1},
               'messages': [{'role': 'user', 'content': 'hi'},
                            {'role': 'assistant', 'content': 'ok', 'reasoning_content': 'because'}]}
    sp.merge_request_overrides(payload, resolve_request(ep(SUPER), '', '', None)['overrides'])
    assert payload['temperature'] == 1.0 and payload['reasoning_effort'] == 'max'   # Pi's own, no override
    assert payload['chat_template_kwargs'] == {'clear_thinking': False, 'x': 1, 'enable_thinking': True,
                                               'medium_effort': False, 'truncate_history_thinking': False}
    sp.inline_reasoning_history(payload)
    a = payload['messages'][1]
    assert a['content'] == '<think>\nbecause</think>ok' and 'reasoning_content' not in a


def test_worker_admission_label_uses_persisted_config_schema(monkeypatch):
    from aa_evals.common.pi_eval.worker import PiEvalWorker, q35
    monkeypatch.setattr(q35.Worker, 'env', lambda *args: {})
    worker = object.__new__(PiEvalWorker)
    # run.py persists `benchmark`; there is no `bench` key in a newly created run.
    worker.config = {'benchmark': 'tbsci_lite', 'agent': 'pi'}
    assert worker.env('task__a0__r0', 'task', 0)['PI_GATE_CLIENT'] == 'tbsci_lite'
    assert worker.env('task__a0__r0', 'task', 0)['PI_AGENT_INTERNET'] == '1'  # historical config
    worker.config['agent_internet'] = False
    assert worker.env('task__a0__r0', 'task', 0)['PI_AGENT_INTERNET'] == '0'
    worker.config['agent'] = 'oracle'
    assert 'PI_GATE_CLIENT' not in worker.env('task__a0__r0', 'task', 0)
    assert 'PI_AGENT_INTERNET' not in worker.env('task__a0__r0', 'task', 0)


def test_worker_disk_cap_from_run_config(monkeypatch):
    """#405: eval_q35's env sets the cap from the run config "diskcap" (run.py writes it: default 3:30720); a config
    without the key (an AA run before #405) stays off (DISKCAP_WITHOUT_KEY), and the worker's own environment never
    changes who the cap spares."""
    from aa_evals.common.pi_eval.worker import PiEvalWorker, q35
    from aa_evals.common import diskcap

    def env(self, k, task, attempt):
        return q35.diskcap_env({'HARBOR_NS_DISKCAP': '7', 'HARBOR_NS_DISKCAP_SPARE': 'python3'},
                               q35.run_diskcap_config(self), f'/x/{k}', task)
    monkeypatch.setattr(q35.Worker, 'env', env)
    worker = object.__new__(PiEvalWorker)
    worker.config = {'benchmark': 'tb21', 'agent': 'pi'}
    e = worker.env('task__a0__r0', 'task', 0)
    assert e['HARBOR_NS_DISKCAP'] == '0' and 'HARBOR_NS_DISKCAP_LOG' not in e and 'HARBOR_NS_DISKCAP_SPARE' not in e
    assert q35.diskcap_config(worker.diskcap_cfg, 'task') is None
    worker.config['diskcap'] = diskcap.arg(diskcap.DEFAULT_ARG)
    e = worker.env('task__a0__r0', 'task', 0)
    assert (e['HARBOR_NS_DISKCAP'], e['HARBOR_NS_DISKCAP_MIN_MB'], e['HARBOR_NS_DISKCAP_LOG']) == \
        ('3', '30720', '/x/task__a0__r0/diskcap.json')
    assert q35.diskcap_config(worker.diskcap_cfg, 'task') == diskcap.DEFAULT == diskcap.config(worker.config, 'task')
    worker.config['diskcap'] = None
    assert worker.env('task__a0__r0', 'task', 0)['HARBOR_NS_DISKCAP'] == '0'
    # with a "diskcap" in the run config, neither the registry's request env nor --harbor-env changes the cap
    worker.config.update(diskcap=diskcap.DEFAULT, request={'overrides': {}, 'env': {'HARBOR_NS_DISKCAP': '0'}},
                         harbor_env={'HARBOR_NS_DISKCAP_MIN_MB': '1', 'HARBOR_NS_DISKCAP_SPARE': 'python3'})
    e = worker.env('task__a0__r0', 'task', 0)
    assert (e['HARBOR_NS_DISKCAP'], e['HARBOR_NS_DISKCAP_MIN_MB']) == ('3', '30720')
    assert 'HARBOR_NS_DISKCAP_SPARE' not in e


def test_worker_one_per_worker_threshold_is_the_runs_worker_disk():
    """#405: a task runs one attempt at a time per worker when its cap is above a quarter of the scratch: pi_eval
    weighs it against its run's worker disk (default 300 GB), so GDPval (storage_mb 20480 -> 60 GiB) is not one."""
    from aa_evals.common.pi_eval.worker import PiEvalWorker, q35
    from aa_evals.common import diskcap
    worker = object.__new__(PiEvalWorker)
    worker.config = {'diskcap': diskcap.DEFAULT, 'worker': {'disk_gb': 300}}
    storage = {'gdpval-task': 20480, 'big': 40000}
    assert q35.diskcap_one_per_worker(worker.diskcap_cfg, storage, worker.diskcap_worker_disk_mb) == {'big'}
    assert q35.diskcap_one_per_worker(worker.diskcap_cfg, storage) == {'gdpval-task', 'big'}   # eval_q35's 200 GiB
    worker.config['worker'] = {}
    assert worker.diskcap_worker_disk_mb == q35.DISKCAP_WORKER_DISK_MB

def test_new_run_offline_default_and_legacy_resume_guard(tmp_path, monkeypatch):
    from aa_evals.common.pi_eval import run
    import json
    args = run.build_parser().parse_args(['--bench', 'tb21', '--run-id', 'new'])
    assert args.agent_internet is False
    assert args.env_class == 'aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment'
    monkeypatch.setattr(run, 'RUNS_ROOT', tmp_path)
    monkeypatch.setattr(run, 'status', lambda path: {})
    stored = tmp_path / 'tb21' / 'legacy'
    stored.mkdir(parents=True)
    (stored / 'config.json').write_text(json.dumps({'agent': 'pi', 'worker': {'n': 1}}))
    with pytest.raises(SystemExit, match='internet policy differs'):
        run.main(['--bench', 'tb21', '--run-id', 'legacy', '--no-submit'])
    run.main(['--bench', 'tb21', '--run-id', 'legacy', '--agent-internet', '--no-submit'])

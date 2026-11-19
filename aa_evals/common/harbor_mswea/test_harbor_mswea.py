"""Tests of the queue, classifier and scorer on synthetic trials (no GPU, no condor).

  PYTHONPATH=. /opt/aai/venvs/harbor/bin/python -m aa_evals.common.harbor_mswea.test_harbor_mswea
(pytest-compatible; the harbor venv has no pytest, so the module runs its tests itself)
"""
import json
from pathlib import Path

from aa_evals.common.harbor_mswea import classify, queue, run, score


def trial(tmp: Path, *, reward=None, exc=None, msg='', exit_status='Submitted', steps=3, exit_exception='',
          wall=(0, 100), tool_output='ok', ctrf=True, traj=True) -> Path:
    raw = tmp / 'raw'
    t = raw / 'job' / 'task__abc'
    (t / 'agent').mkdir(parents=True)
    (t / 'verifier').mkdir()
    res = {'agent_execution': {'started_at': f'2026-11-02T00:00:{wall[0]:02d}+00:00',
                               'finished_at': f'2026-11-02T00:{wall[1] // 60:02d}:{wall[1] % 60:02d}+00:00'}}
    if reward is not None:
        res['verifier_result'] = {'rewards': {'reward': reward}}
    if exc:
        res['exception_info'] = {'exception_type': exc, 'exception_message': msg}
    (t / 'result.json').write_text(json.dumps(res))
    if ctrf:
        (t / 'verifier' / 'ctrf.json').write_text('{}')
    if traj:
        t0 = 1793577600.0   # 2026-11-02T00:00:00Z
        msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]
        for i in range(steps):   # 10 s model, 5 s environment per step
            msgs.append({'role': 'assistant', 'content': '', 'extra': {
                'timestamp': t0 + 15 * i + 10, 'response': {'choices': [{'finish_reason': 'tool_calls'}]}}})
            msgs.append({'role': 'tool', 'content': '', 'extra': {'timestamp': t0 + 15 * i + 15,
                                                                   'raw_output': tool_output}})
        msgs.append({'role': 'exit', 'content': '', 'extra': {'exit_status': exit_status,
                                                               'exception_str': exit_exception}})
        (t / 'agent' / classify.TRAJ).write_text(json.dumps(
            {'info': {'exit_status': exit_status, 'model_stats': {'api_calls': steps}}, 'messages': msgs}))
    return raw


def test_pass_and_times(tmp_path):
    o = classify.classify(trial(tmp_path, reward=1.0, wall=(0, 45)), 0, time_limit_sec=28800)
    assert o['infrastructure_failure'] is False and o['score'] == 1.0 and o['status'] == 'ok'
    assert o['model_sec'] == 30.0 and o['env_sec'] == 15.0 and o['api_calls'] == 3


def test_env_time_limit(tmp_path):
    o = classify.classify(trial(tmp_path, reward=1.0, wall=(0, 45)), 0, time_limit_sec=10)
    assert o['score'] == 0.0 and o['env_time_exceeded']


def test_verifier_timeout_is_model_failure(tmp_path):
    o = classify.classify(trial(tmp_path, exc='VerifierTimeoutError', ctrf=False), 0)
    assert o['infrastructure_failure'] is False and o['score'] == 0.0 and o['status'] == 'model_error'


def test_backend_unreachable_is_infra(tmp_path):
    o = classify.classify(trial(tmp_path, exc='NonZeroAgentExitCodeError', exit_status='APIConnectionError'), 1)
    assert o['infrastructure_failure'] and o['infrastructure_reason'] == 'llm_backend_unreachable'


def test_model_not_served_is_infra(tmp_path):
    o = classify.classify(trial(tmp_path, exc='NonZeroAgentExitCodeError', exit_status='NotFoundError'), 1)
    assert o['infrastructure_reason'] == 'endpoint_rejected'


def test_context_overflow_is_model(tmp_path):
    o = classify.classify(trial(tmp_path, reward=0.0, exit_status='BadRequestError',
                                exit_exception="This model's maximum context length is 131072 tokens"), 1)
    assert o['infrastructure_failure'] is False and o['context_overflow']


def test_context_cut_format_errors_are_overflow(tmp_path):
    raw = trial(tmp_path, reward=0.0, exit_status='RepeatedFormatError')
    t = next(raw.glob('*/*/agent'))
    d = json.loads((t / classify.TRAJ).read_text())
    for _ in range(3):
        d['messages'].insert(-1, {'role': 'user', 'content': 'cut', 'extra': {
            'interrupt_type': 'FormatError', 'response': {
                'choices': [{'finish_reason': 'length', 'message': {'content': ''}}],
                'usage': {'prompt_tokens': 129987, 'completion_tokens': 1085}}}})
    (t / classify.TRAJ).write_text(json.dumps(d))
    o = classify.classify(raw, 0, max_model_len=131072)
    assert o['context_overflow'] and o['context_cuts'] == 3 and o['score'] == 0.0


def test_other_bad_request_is_infra(tmp_path):
    o = classify.classify(trial(tmp_path, exit_status='BadRequestError', exit_exception='unknown field foo'), 1)
    assert o['infrastructure_reason'] == 'bad_request'


def test_env_start_is_infra(tmp_path):
    o = classify.classify(trial(tmp_path, exc='EnvironmentStartTimeoutError', traj=False), 1)
    assert o['infrastructure_failure'] and o['infrastructure_reason'] == 'EnvironmentStartTimeoutError'


def test_error_before_agent_setup_is_infra(tmp_path):
    raw = trial(tmp_path, exc='RuntimeError', msg='service redis never became healthy', traj=False)
    r = next(raw.glob('*/*/result.json'))
    d = json.loads(r.read_text())
    d['agent_execution'] = None
    r.write_text(json.dumps(d))
    o = classify.classify(raw, 1)
    assert o['infrastructure_reason'] == 'environment_not_started'


def test_proxy_in_tool_output_is_infra(tmp_path):
    o = classify.classify(trial(tmp_path, reward=0.0, tool_output='curl: (7) Failed to connect to proxy.internal:8080 port '
                                '8080: Connection refused'), 0)
    assert o['infrastructure_reason'] == 'proxy_outage'


def test_no_result_is_infra(tmp_path):
    (tmp_path / 'raw').mkdir()
    assert classify.classify(tmp_path / 'raw', 1)['infrastructure_reason'] == 'no_harbor_result'


def test_queue_retry_ties_in_queue_order(tmp_path):
    """#531: retries with the same repeat and try come in the queue's order, not a set's per-process order."""
    (tmp_path / 'config.json').write_text(json.dumps({'max_concurrent': 10, 'repeats': 1, 'benchmark': 'x',
                                                      'n_tasks_benchmark': 3}))
    q = queue.Queue(tmp_path)
    q.init([('c', 0), ('a', 0), ('b', 0)])
    (tmp_path / 'queue' / 'items.txt').write_text('c 0\na 0\nb 0\n')   # a claim order that is not the key order
    for t in ('c', 'a', 'b'):
        assert q.claim('w', lambda x, t=t: x == t) == f'{t}__r0__t0'
        q.finish(f'{t}__r0__t0', {'infrastructure_failure': True, 'infrastructure_reason': 'x', 'score': None})
    assert q.candidates()[:3] == ['c__r0__t1', 'a__r0__t1', 'b__r0__t1']   # items.txt order, not the keys'
    # the same under any hash seed (set difference -> iteration order differs per process before #531)
    import os
    import subprocess
    import sys
    code = ('import sys; from aa_evals.common.harbor_mswea import queue; '
            'print(",".join(queue.Queue(sys.argv[1]).candidates()[:3]))')
    root = str(Path(queue.__file__).resolve().parents[3])
    outs = {subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, check=True,
                           env={**os.environ, 'PYTHONHASHSEED': str(h), 'PYTHONPATH': root}).stdout.strip()
            for h in range(8)}
    assert outs == {'c__r0__t1,a__r0__t1,b__r0__t1'}


def test_queue_retries_and_score(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'max_concurrent': 10, 'repeats': 2, 'benchmark': 'x',
                                                      'n_tasks_benchmark': 3}))
    q = queue.Queue(tmp_path)
    q.init([('a', 0), ('a', 1), ('b', 0), ('b', 1)])
    assert q.claim('w', lambda t: t == 'b') == 'b__r0__t0'
    keys = [q.claim('w', lambda t: True) for _ in range(3)]
    assert keys == ['a__r0__t0', 'a__r1__t0', 'b__r1__t0']
    q.finish('a__r0__t0', {'infrastructure_failure': False, 'score': 1.0, 'status': 'ok'})
    q.finish('a__r1__t0', {'infrastructure_failure': False, 'score': 0.0, 'status': 'ok'})
    q.finish('b__r0__t0', {'infrastructure_failure': False, 'score': 1.0, 'status': 'ok'})
    for t in range(queue.MAX_RETRIES + 1):   # b repeat 1 fails on infrastructure every time
        k = f'b__r1__t{t}'
        if t:
            assert q.claim('w', lambda x: True) == k
        q.finish(k, {'infrastructure_failure': True, 'infrastructure_reason': 'x', 'score': None})
    assert q.claim('w', lambda x: True) is None and q.complete()
    s = score.finalize(tmp_path)
    assert s['score'] == 75.0 and s['n_items'] == 2 and s['n_infra_errors'] == 1
    assert s['score_all_tasks_lower_bound'] == 50.0 and not s['valid']   # 1 of 4 items infra > 2%


def test_mini_config_routes_sampling():
    c = run.mini_config(500, {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'high',
                              'chat_template_kwargs': {'enable_thinking': True}})
    kw = c['model']['model_kwargs']
    assert c['agent'] == {'step_limit': 500, 'cost_limit': 0}
    assert kw['temperature'] == 1.0 and kw['top_p'] == 0.95
    assert kw['extra_body'] == {'reasoning_effort': 'high', 'chat_template_kwargs': {'enable_thinking': True}}


if __name__ == '__main__':
    import inspect
    import tempfile
    import traceback
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for name, fn in tests:
        try:
            fn(*([Path(tempfile.mkdtemp())] if 'tmp_path' in inspect.signature(fn).parameters else []))
        except Exception:   # noqa: BLE001
            failed += 1
            print('FAIL', name)
            traceback.print_exc()
    print(f'{len(tests) - failed} passed, {failed} failed')
    raise SystemExit(1 if failed else 0)


def test_url_spec_several_engines(monkeypatch):
    from aa_evals.common.harbor_mswea import endpoint
    one = endpoint.resolve('url:http://h1:1/v1@m')
    assert one['urls'] == ['http://h1:1/v1'] and one['served'] == 'm'
    many = endpoint.resolve('url:http://h1:1/v1,http://h2:2,http://h3:3/v1@m')
    assert many['urls'] == ['http://h1:1/v1', 'http://h2:2/v1', 'http://h3:3/v1'] and many['served'] == 'm'

    class Resp:
        def __init__(self, status):
            self.status = status
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake(url, timeout=10):
        if 'h2' in url:
            raise OSError('refused')
        return Resp(200)
    monkeypatch.setattr(endpoint.urllib.request, 'urlopen', fake)
    assert endpoint.alive('url:http://h1:1/v1,http://h3:3/v1@m')[0]
    ok, _, why = endpoint.alive('url:http://h1:1/v1,http://h2:2/v1@m')
    assert not ok and 'h2' in why

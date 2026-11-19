"""python -m pytest aa_evals/tracejudge/test_tracejudge.py -q  (CPU only, synthetic run dirs)."""
import asyncio
import io
import json
import tarfile
from pathlib import Path

from aa_evals.tracejudge import fold_summary
from aa_evals.tracejudge.core import build_row, judge_run, needs_llm
from aa_evals.tracejudge.digest import render
from aa_evals.tracejudge.judge import parse
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.rules import decide, run_rules


import pytest


@pytest.fixture(autouse=True)
def _private_dns_cache(tmp_path, monkeypatch):
    """Tests never read or write the shared DNS verdict cache."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, 'DNS_CACHE', str(tmp_path / 'dns_cache.json'))
    monkeypatch.setattr(rules, '_DNS', {})
    monkeypatch.setattr(rules, '_local_dns', lambda h: None)      # no real DNS in tests; stub per test
    monkeypatch.setattr(rules, '_public_dns', lambda h: None)


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def _session(*msgs):
    lines = [{'type': 'session'}] + [{'type': 'message', 'message': m} for m in msgs]
    return '\n'.join(json.dumps(x) for x in lines)


def _pi_attempt(run: Path, key: str, outcome: dict, session: str, harbor_log: str = '', exception: str = '',
                verifier: str = ''):
    d = run / 'attempts' / key
    _w(d / 'attempt.json', {'key': key, 'protocol': {'context_window': 131072}})
    _w(d / 'outcome.json', outcome)
    _w(d / 'harbor.log', harbor_log)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, text in ((f'{key}/{key}/T/agent/pi/sessions/s.jsonl', session), (f'{key}/{key}/T/exception.txt', exception),
                           (f'{key}/{key}/T/verifier/test-stdout.txt', verifier)):
            b = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    (d / 'trial.tar.gz').write_bytes(buf.getvalue())


TASK = {'role': 'user', 'content': [{'type': 'text', 'text': 'Solve the task in /app.'}]}


def _call(cmd):
    return {'role': 'assistant', 'content': [{'type': 'toolCall', 'name': 'bash', 'arguments': {'command': cmd}}],
            'stopReason': 'toolUse'}


def _res(text):
    return {'role': 'toolResult', 'toolName': 'bash', 'content': [{'type': 'text', 'text': text}], 'isError': False}


def make_pi_run(tmp: Path) -> Path:
    run = tmp / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    _w(run / 'config.json', {'benchmark': 'tb21'})
    # memcap killed the agent (#145): infra without the LLM
    _pi_attempt(run, 'memkill__a0__r0', {'exception_type': 'NonZeroAgentExitCodeError', 'returncode': 1, 'score': 0.0,
                                         'infrastructure_failure': False, 'exception_message': 'Command failed (exit 137)',
                                         'stats': {'end_state': 'error'}},
                _session(TASK, _call('python sim.py'), _res('running')),
                'harbor_ns memcap: sandbox at 12469 MB > limit 12288 MB; killed pid 39 (pi, 123 MB)\n')
    # the model pkilled its own agent: a model outcome, never flagged
    _pi_attempt(run, 'selfkill__a0__r0', {'exception_type': 'NonZeroAgentExitCodeError', 'returncode': 1, 'score': 0.0,
                                          'infrastructure_failure': False, 'exception_message': 'Command failed (exit -15)',
                                          'stats': {'end_state': 'error'}},
                _session(TASK, _call("pkill -f '[p]redict.py'; sleep 2")))
    # the in-flight call is only in exception.txt
    _pi_attempt(run, 'inflight__a0__r0', {'exception_type': 'NonZeroAgentExitCodeError', 'returncode': 1, 'score': 1.0,
                                          'infrastructure_failure': False, 'exception_message': 'Command failed (exit -15)',
                                          'stats': {'end_state': 'error'}},
                _session(TASK, _call('ls'), _res('a b')),
                exception='{"type":"tool_execution_update","toolName":"bash","args":{"command":"cat x; kill 81 2>/dev/null"}}')
    # a pipeline-known infra retry, then its counted retry
    _pi_attempt(run, 'proxy__a0__r0', {'infrastructure_failure': True, 'infrastructure_reason': 'pi_backend_unreachable',
                                       'score': None, 'stats': {}}, _session(TASK))
    _pi_attempt(run, 'proxy__a0__r1', {'infrastructure_failure': False, 'score': 1.0, 'stats': {'end_state': 'finished'}},
                _session(TASK, _call('make'), _res('ok'), {'role': 'assistant', 'content': [{'type': 'text', 'text': 'done'}],
                                                             'stopReason': 'stop'}))
    # a memcap kill of a task process plus errno tables in tool output: clean
    _pi_attempt(run, 'clean__a0__r0', {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}},
                _session(TASK, _call('cat /usr/include/asm-generic/errno-base.h'),
                         _res('#define ENOSPC 28 /* No space left on device */'),
                         {'role': 'assistant', 'content': [{'type': 'text', 'text': 'I give up'}], 'stopReason': 'stop'}),
                'harbor_ns memcap: sandbox at 9000 MB > limit 8192 MB; killed pid 77 (python3, 8000 MB)\n'
                'ValueError: too many values to unpack (expected 2)\n')
    return run


def test_pi_rules(tmp_path):
    L = open_run(make_pi_run(tmp_path))
    assert L.name == 'pi_eval' and L.bench == 'tb21'
    v = {k: decide(run_rules(L.load(k))) for k in L.finished()}
    assert v['memkill__a0__r0'] == 'infra'
    assert v['selfkill__a0__r0'] == 'model'
    assert v['inflight__a0__r0'] == 'model'
    assert v['proxy__a0__r0'] == 'infra'
    assert v['clean__a0__r0'] == 'clean'
    assert L.load('proxy__a0__r0').counted is False and L.load('proxy__a0__r1').counted is True
    d = render(L.load('clean__a0__r0'), run_rules(L.load('clean__a0__r0')))
    assert 'too many values to unpack' not in d and 'killed pid 77' in d


def test_judge_rules_only_and_fold(tmp_path):
    run = make_pi_run(tmp_path)
    rows = asyncio.run(judge_run(run, None, 'none', log=lambda *a: None))
    assert len(rows) == 6
    s = fold_summary(run, {'score': 50.0})
    assert s['score'] == 50.0
    assert s['n_flagged'] == 1 and s['n_flagged_by_category'] == {'sandbox_kill': 1}
    assert s['tracejudge']['n_judged'] == 5               # the retried physical attempt doesn't count
    assert s['tracejudge']['flagged_keys'] == ['memkill__a0__r0']
    # a second pass judges nothing new
    assert asyncio.run(judge_run(run, None, 'none', log=lambda *a: None)) == []


def test_fold_without_flags(tmp_path):
    s = fold_summary(tmp_path, {'score': 1})
    assert s['n_flagged'] is None and s['tracejudge']['status'] == 'not run'


def test_hle_layout(tmp_path):
    run = tmp_path / 'hle' / 'r'
    _w(run / 'config.json', {'benchmark': 'hle'})
    _w(run / 'transcripts' / 'q1.r0.json', {'generation': {'content': '', 'reasoning': 'hmm ' * 50, 'finish_reason': 'stop',
                                                           'usage': {'completion_tokens': 300}, 'status': 'ok'}})
    _w(run / 'transcripts' / 'q2.r0.json', {'generation': {'content': '', 'finish_reason': 'stop',
                                                           'usage': {'completion_tokens': 0}, 'status': 'ok'}})
    _w(run / 'transcripts' / 'q3.r0.json', {'generation': {'content': '<|im_start|>assistant<|im_end|>', 'finish_reason': 'stop',
                                                           'usage': {'completion_tokens': 5}, 'status': 'ok'}})
    L = open_run(run)
    assert L.name == 'hle'
    assert decide(run_rules(L.load('q1.r0'))) == 'model'        # EOS inside reasoning: model behavior
    assert decide(run_rules(L.load('q2.r0'))) == 'suspicious'   # empty with zero tokens: silent zero?
    assert decide(run_rules(L.load('q3.r0'))) == 'suspicious'   # template tokens in the answer


@pytest.mark.parametrize('saved,generation,expected,source', [
    ({'max_tokens': None}, {'max_tokens': None}, None, 'saved request'),
    ({'max_tokens': 100000}, {'max_tokens': None}, 100000, 'saved request'),
    ({}, {'max_tokens': None}, None, 'generation config'),
    ({}, {'max_tokens': 65536}, 65536, 'generation config'),
    ({}, {}, 32768, 'legacy config'),
])
def test_gdppdf_request_budget_keeps_uncapped_distinct_from_reserve(tmp_path, saved, generation, expected, source):
    run = tmp_path / 'gdppdf' / 'cap'
    _w(run / 'config.json', {'generation': {**generation, 'output_reserve': 32768},
                             'limits': {'max_tokens': 32768}, 'context': {'window': 262144}})
    _w(run / 'generations.jsonl', json.dumps({
        'task_id': 't', 'repeat': 0, 'status': 'ok', 'answer': 'answer', 'finish_reason': 'length',
        'usage': {'completion_tokens': 60000}, **saved}) + '\n')
    trace = open_run(run).load('t/0')
    assert trace.recorded['max_tokens'] == expected
    assert trace.recorded['max_tokens_source'] == source
    assert trace.recorded['output_reserve'] == 32768
    hits = run_rules(trace)
    assert ('length_below_cap' in {h.rule for h in hits}) == (expected == 100000)
    if expected is None:
        digest = render(trace, hits)
        assert 'input-planning floor, not an output-token cap' in digest
        if source == 'saved request':
            assert 'saved request sent no max_tokens cap' in digest
        else:
            assert 'per-request budget was not saved' in digest
            assert 'may have been dynamically clamped' in digest
            assert 'sent no max_tokens cap' not in digest


def test_jsonl_layouts(tmp_path):
    g = tmp_path / 'gdppdf' / 'r'
    _w(g / 'config.json', {'limits': {'max_tokens': 65536}})
    _w(g / 'generations.jsonl', json.dumps({'task_id': 't1', 'repeat': 0, 'answer': 'x', 'finish_reason': 'length',
                                            'usage': {'completion_tokens': 1024}, 'plan': {'covered': 3, 'n_images': 9}}) + '\n')
    _w(g / 'generations.jsonl', (g / 'generations.jsonl').read_text() + json.dumps(
        {'task_id': 't2', 'repeat': 0, 'status': 'model_error', 'error_kind': 'input_too_long', 'score': 0,
         'error': 'protocol limit, not infra: text alone is ~132519 tokens', 'plan': {'reason': 'input_too_long'}}) + '\n')
    L = open_run(g)
    t = L.load('t1/0')
    rules = {h.rule for h in run_rules(t)}
    assert L.name == 'gdppdf' and {'length_below_cap', 'pages_dropped'} <= rules
    assert decide(run_rules(L.load('t2/0'))) == 'model'      # a protocol limit, never a silent zero
    assert not needs_llm('fails', 'model', L.load('t2/0'))
    a = tmp_path / 'automationbench' / 'r'
    _w(a / 'transcripts.jsonl', json.dumps({'item_id': 'x', 'repeat': 0, 'status': 'ok', 'messages': [
        {'role': 'user', 'content': 'do it'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'f', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'ok'}], 'debug': {'errors': ['APIConnectionError: refused']}}) + '\n')
    L = open_run(a)
    t = L.load('x/r0')
    assert L.name == 'automationbench' and t.kind == 'agentic' and decide(run_rules(t)) == 'suspicious'


def test_generic_layout(tmp_path):
    run = tmp_path / 'scicode_verified' / 'r'
    _w(run / 'tr' / 'p1.json', {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}],
                                'events': ['sandbox: exec timeout']})
    _w(run / 'items.jsonl', json.dumps({'item_id': 'p1', 'repeat': 0, 'status': 'ok', 'score': 0,
                                        'transcript': 'tr/p1.json'}) + '\n')
    L = open_run(run)
    t = L.load('p1/r0')
    assert L.name == 'generic' and L.bench == 'scicode_verified'
    assert t.events and t.events[0].text == 'sandbox: exec timeout' and t.turns[-1].text == 'a'


def test_system_prompt_renders():
    from aa_evals.tracejudge.judge import system_prompt
    s = system_prompt()
    assert 'silent_zero' in s and '{"name": ...' in s and '{cats}' not in s


def test_parse_and_policy():
    v = parse('```json\n{"flags": [{"category": "network", "evidence": "x", "ref": "r", "confidence": 2}], "reason": "y"}\n```')
    assert v['flags'][0]['confidence'] == 1.0
    assert parse('{"flags": [{"category": "not_a_category"}], "reason": ""}')['flags'] == []
    assert parse('no json') is None

    class T:
        recorded = {'score': 1.0, 'status': 'ok'}
    assert needs_llm('fails', 'clean', T()) is False
    assert needs_llm('fails', 'suspicious', T()) is True
    assert needs_llm('all', 'model', T()) is False


@pytest.mark.parametrize('verdict', ['clean', 'model', 'infra', 'suspicious'])
@pytest.mark.parametrize('score', [0.0, 1.0, None])
def test_every_policy_includes_decisive_model_outcomes(verdict, score):
    class T:
        recorded = {'score': score, 'status': 'ok'}
    assert needs_llm('every', verdict, T()) is True
    assert needs_llm('none', verdict, T()) is False
    if verdict == 'model':
        assert needs_llm('all', verdict, T()) is False


def test_every_invokes_judge_for_model_outcome_without_changing_all(tmp_path):
    class Judge:
        calls = 0

        def describe(self):
            return {'model': 'fake-only'}

        async def judge(self, text):
            self.calls += 1
            assert text
            return {'ok': True, 'flags': [], 'reason': 'model behavior',
                    'prompt_tokens': 3, 'completion_tokens': 2}

    run = make_pi_run(tmp_path)
    judge = Judge()
    all_rows = asyncio.run(judge_run(run, judge, 'all', keys=['selfkill__a0__r0'],
                                    out=run / 'all.jsonl', log=lambda *a: None))
    assert all_rows[0]['prefilter'] == 'model'
    assert judge.calls == 0 and all_rows[0]['llm'] is None
    every_rows = asyncio.run(judge_run(run, judge, 'every', keys=['selfkill__a0__r0'],
                                      out=run / 'every.jsonl', log=lambda *a: None))
    assert judge.calls == 1
    assert every_rows[0]['prefilter'] == 'model'
    assert every_rows[0]['llm']['ok'] is True
    assert every_rows[0]['llm']['prompt_tokens'] == 3
    assert every_rows[0]['judge'] == {'model': 'fake-only'}


@pytest.mark.parametrize('command', ['judge', 'watch'])
def test_cli_accepts_and_forwards_every_policy(monkeypatch, command):
    from aa_evals.tracejudge import __main__ as cli, core
    seen = []

    async def fake(*args):
        seen.append(args)
        return []

    monkeypatch.setattr(cli, '_judge', lambda args: None)
    monkeypatch.setattr(core, 'judge_run' if command == 'judge' else 'watch_run', fake)
    cli.main([command, '--run-dir', '/unused', '--llm', 'every'])
    assert seen[0][2] == 'every'


def test_tool_result_exception_is_benchmark_output(tmp_path):
    """AutomationBench's simulated Gmail raises on a list `to` and returns the exception as the tool result: the model
    misused the API, so an LLM harness_error flag citing only that result is demoted below the threshold."""
    a = tmp_path / 'automationbench' / 'r'
    _w(a / 'transcripts.jsonl', json.dumps({'item_id': 'sales.create_draft_reply', 'repeat': 0, 'status': 'ok', 'messages': [
        {'role': 'user', 'content': 'reply to the lead'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {
            'name': 'gmail_create_draft', 'arguments': '{"to": [{"email": "a@b.c"}]}'}}]},
        {'role': 'tool', 'content': "Error: 'list' object has no attribute 'split'"},
        {'role': 'assistant', 'content': 'Now I will run the tests. <tool_call>{"name": "bash", "arguments": {}}'}]}) + '\n')
    t = open_run(a).load('sales.create_draft_reply/r0')
    hits = run_rules(t)
    exc = {'category': 'harness_error', 'evidence': "'list' object has no attribute 'split'", 'ref': '[msg:2]', 'confidence': 1.0}
    row = build_row(t, hits, decide(hits), {'ok': True, 'flags': [exc]}, None, None)
    assert row['flagged'] is False and row['flags'][0]['demoted'] == 'tool_result_only'
    # raw tool-call markup in the model's own text stays a harness_error flag
    markup = {'category': 'harness_error', 'evidence': '<tool_call>{"name": "bash"', 'ref': '[msg:3]', 'confidence': 0.9}
    assert build_row(t, hits, decide(hits), {'ok': True, 'flags': [markup]}, None, None)['flagged'] is True
    # a proxy error in a tool result is still infra (only harness_error is demoted)
    net = {'category': 'network', 'evidence': 'Tunnel connection failed: 503', 'ref': '[msg:2]', 'confidence': 0.9}
    assert build_row(t, hits, decide(hits), {'ok': True, 'flags': [net]}, None, None)['flagged'] is True


SEGV = """collected 27 items

../tests/test_outputs.py Fatal Python error: Segmentation fault

Current thread 0x00000fba6beb4240 (most recent call first):
  File "{inner}", line 53 in find_dominant
  File "/tests/test_outputs.py", line 54 in test_eigen_pair
  File "/usr/local/lib/python3.13/site-packages/_pytest/python.py", line 157 in pytest_pyfunc_call
"""


def test_verifier_crash_in_solution(tmp_path):
    """A segfault under the verifier in the model's own /app code (a C extension it built) is the solution failing;
    a crash in the test code stays a grader_crash."""
    run = tmp_path / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    _pi_attempt(run, 'eig__a0__r0', out, _session(TASK, _call('make'), _res('ok')), verifier=SEGV.format(inner='/app/eigen.py'))
    _pi_attempt(run, 'harn__a0__r0', out, _session(TASK, _call('make'), _res('ok')),
                verifier=SEGV.format(inner='/usr/lib/python3/dist-packages/numpy/x.py').replace(
                    'File "/tests/test_outputs.py"', 'File "/tests/conftest.py"'))
    L = open_run(run)
    flag = {'category': 'grader_crash', 'evidence': 'Fatal Python error: Segmentation fault',
            'ref': '[verifier/test-stdout.txt]', 'confidence': 0.9}
    t = L.load('eig__a0__r0')
    hits = run_rules(t)
    assert 'verifier_crash_in_solution' in {h.rule for h in hits} and 'crash frames' in render(t, hits)
    row = build_row(t, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)
    assert row['flagged'] is False and row['flags'][0]['demoted'] == 'crash_in_solution'
    t = L.load('harn__a0__r0')   # the first non-library frame is the test code: the grader crashed
    hits = run_rules(t)
    assert build_row(t, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)['flagged'] is True


def test_code_sha(tmp_path):
    """A code snapshot inside an unrelated repo reports its own .complete sha, never the enclosing repo's HEAD."""
    import subprocess
    from aa_evals.tracejudge.core import code_sha
    g = ['git', '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false']
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'f').write_text('x')
    subprocess.run(g + ['-C', str(tmp_path), 'add', 'f'], check=True)
    subprocess.run(g + ['-C', str(tmp_path), 'commit', '-qm', 'outer'], check=True)
    head = subprocess.run(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    assert code_sha(tmp_path) == head                     # the repo itself
    snap = tmp_path / 'code' / 'abc'
    snap.mkdir(parents=True)
    assert code_sha(snap) is None                         # inside the outer repo, no record: unknown, not the outer HEAD
    (snap / '.complete').write_text(json.dumps({'git_sha': 'deadbeef', 'dirty': False}))
    assert code_sha(snap) == 'deadbeef'
    (snap / '.complete').unlink()
    (snap / 'FROZEN_SHA').write_text('hle-commit\n')
    assert code_sha(snap) == 'hle-commit'
    (snap / 'FROZEN_SHA').unlink()
    (snap / 'GIT_SHA').write_text('gdpval-commit\n')
    assert code_sha(snap) == 'gdpval-commit'
    (snap / 'GIT_SHA').write_text(json.dumps({'sha': 'omni-commit', 'dirty': True}))
    assert code_sha(snap) == 'omni-commit-dirty'


def test_critpt_aggregate_usage_cannot_prove_per_call_cap(tmp_path):
    """Even totals above the window can span accepted turns and failed summaries; retain review."""
    run = tmp_path / 'critpt' / 'r'
    _w(run / 'config.json', {'generation': {'context_window_tokens': 131072, 'max_output_tokens': None}})
    err = 'context_overflow: Context overflow reached the original prompt (configured context_window_tokens=131072)'
    hist = [[{'role': 'user', 'content': 'derive it'}]]
    rows, items = [], []
    for a, (inp, out) in enumerate(((1892, 128411), (1892, 20000), (1333214, 86929))):
        rows.append({'key': f'C/r0/a{a}', 'item_id': 'C', 'repeat': 0, 'attempt': a, 'status': 'model_error', 'error': err,
                     'history': hist, 'answer_code': ''})
        items.append({'item_id': 'C', 'repeat': 0, 'attempt': a, 'status': 'model_error', 'score': 0, 'error': err,
                      'turns': 0, 'input_tokens': inp, 'output_tokens': out})
    _w(run / 'transcripts.jsonl', '\n'.join(json.dumps(r) for r in rows) + '\n')
    _w(run / 'items.jsonl', '\n'.join(json.dumps(r) for r in items) + '\n')
    L = open_run(run)
    assert L.name == 'critpt'
    for a in range(3):
        trace = L.load(f'C/r0/a{a}')
        assert trace.recorded['token_usage_scope'] == 'trajectory_aggregate'
        hits = run_rules(trace)
        assert 'protocol_context_overflow' not in {h.rule for h in hits}
        assert decide(hits) == 'suspicious'
        assert 'exact request budgets cannot be reconstructed' in render(trace, hits)


def _critpt_trace(tmp_path, *, sandbox=None, status='ok', error='', history=None):
    run = tmp_path / 'critpt' / 'source-backed'
    _w(run / 'config.json', {'harness': {'sandbox': sandbox},
                            'generation': {'context_window_tokens': 131072}})
    row = {'key': 'C/r0/a1', 'item_id': 'C', 'repeat': 0, 'attempt': 1, 'status': status, 'error': error}
    _w(run / 'transcripts.jsonl', json.dumps({**row, 'history': history or [], 'answer_code': ''}) + '\n')
    _w(run / 'items.jsonl', json.dumps({**row, 'score': 0, 'input_tokens': 1036956,
                                      'output_tokens': 196062, 'n_summaries': 2}) + '\n')
    return open_run(run).load(row['key'])


@pytest.mark.parametrize('isolation', ['apptainer-net-none', 'unshare-inside'])
def test_critpt_explicit_native_sandbox_policy_reaches_judge(tmp_path, isolation):
    # Native sandbox_argv supports exactly these two network-isolated modes. No task answers in this fixture.
    trace = _critpt_trace(tmp_path, sandbox={'isolation': isolation})
    assert trace.recorded['agent_internet'] is False
    assert trace.recorded['agent_network'] == {
        'mode': 'offline', 'isolation': isolation, 'source': 'config.harness.sandbox.isolation'}
    digest = render(trace, run_rules(trace))
    assert 'Agent tools are offline (kernel-isolated ' + isolation in digest
    assert 'does not describe the host model client or evaluator network' in digest


@pytest.mark.parametrize('sandbox', [None, {}, {'isolation': 'unknown'}, {'isolation': False}, 'offline'])
def test_critpt_unrecorded_or_unknown_isolation_stays_unknown(tmp_path, sandbox):
    trace = _critpt_trace(tmp_path, sandbox=sandbox)
    assert 'agent_internet' not in trace.recorded and 'agent_network' not in trace.recorded
    assert not any(e.source == 'sandbox_policy' for e in trace.events)


@pytest.mark.parametrize('boundary', ['original prompt', 'summarized context'])
def test_critpt_native_context_boundary_is_typed_but_budget_unknown(tmp_path, boundary):
    # Exact stock Stirrup boundary text observed in native model_error; the tiny retained summary is not
    # the failed request. The native aggregate above includes omitted summary/main responses.
    err = f'context_overflow: Context overflow reached the {boundary} (configured context_window_tokens=131072)'
    trace = _critpt_trace(tmp_path, status='model_error', error=err,
                          history=[[{'role': 'user', 'content': 'short retained summary'}]])
    assert trace.recorded['error'] == err
    metadata = trace.recorded['native_protocol_boundary']
    assert metadata['kind'] == 'context_overflow' and metadata['boundary'] == boundary
    assert metadata['request_budget'].startswith('unknown;')
    assert any(e.source == 'native_protocol_boundary' and e.text == err for e in trace.events)
    assert not any(e.source == 'harness' for e in trace.events)
    hits = run_rules(trace)
    assert decide(hits) == 'suspicious'  # source-aware LLM review, no automatic aggregate-based clearing
    assert 'protocol_context_overflow' not in {h.rule for h in hits}
    assert 'native_context_boundary_review' in {h.rule for h in hits}
    assert 'Exact failed-request budget is unavailable' in render(trace, hits)


@pytest.mark.parametrize('isolation,expected', [('apptainer-net-none', 'offline_agent_network_denied'),
                                               ('unknown', None)])
def test_critpt_tool_denial_requires_recorded_policy(tmp_path, isolation, expected):
    from aa_evals.tracejudge.rules import demotion
    # Public-site urllib denial, followed by a submitted answer, as in the audited successful trajectory.
    output = 'urllib.error.URLError: <urlopen error Temporary failure in name resolution>'
    history = [[{'role': 'assistant', 'tool_calls': [{'function': {
        'name': 'code_exec', 'arguments': 'urllib.request.urlopen("https://www.example.org")'}}]},
        {'role': 'tool', 'content': output}]]
    trace = _critpt_trace(tmp_path, sandbox={'isolation': isolation}, history=history)
    result = next(t for t in trace.turns if t.kind == 'tool_result')
    flag = {'category': 'network', 'ref': result.ref, 'evidence': output}
    assert demotion(trace, flag) == expected


@pytest.mark.parametrize('status,error', [
    ('infra_error', 'APIConnectionError: model endpoint connection refused'),
    ('infra_error', 'RuntimeError: sandbox launch failed'),
    ('infra_error', 'context_overflow: Context overflow reached the summarized context'),
    ('model_error', 'context_overflow: unrecognized backend error'),
])
def test_critpt_offline_policy_does_not_hide_real_or_unknown_errors(tmp_path, status, error):
    from aa_evals.tracejudge.rules import demotion
    trace = _critpt_trace(tmp_path, sandbox={'isolation': 'apptainer-net-none'}, status=status, error=error)
    assert 'native_protocol_boundary' not in trace.recorded
    assert trace.recorded['error'] == error
    assert any(e.source == 'harness' and e.text == error for e in trace.events)
    assert decide(run_rules(trace)) in ('infra', 'suspicious')
    flag = {'category': 'network', 'ref': 'transcripts.jsonl:error', 'evidence': error}
    assert demotion(trace, flag) is None


def test_single_response_usage_still_proves_protocol_cap():
    from aa_evals.tracejudge.trace import Trace
    trace = Trace('synthetic', 'C/r0', 'C', 0, 'single', {
        'status': 'model_error', 'error': 'output_limit: length',
        'input_tokens': 1892, 'output_tokens': 128411, 'context_window': 131072})
    assert decide(run_rules(trace)) == 'model'


def test_automation_budget_context_is_factual_and_does_not_auto_clear(tmp_path):
    run = tmp_path / 'automationbench' / 'r'
    _w(run / 'config.json', {'generation': {'context_window': 131072,
        'max_tokens': 'the room left under the context window (estimated per request)',
        'sampling_args': {}}, 'harness': {'max_turns': 50}})
    _w(run / 'items.jsonl', json.dumps({'item_id': 'C', 'repeat': 0, 'status': 'ok',
        'score': 0, 'truncated_turns': 1, 'max_context_tokens': 130202, 'n_cap_overshoot': 0}) + '\n')
    _w(run / 'transcripts.jsonl', json.dumps({'item_id': 'C', 'repeat': 0,
        'debug': {'stop_reasons': ['length']}, 'messages': [
            {'role': 'user', 'content': 'Do the task.'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'Still thinking'}]}) + '\n')
    trace = open_run(run).load('C/r0')
    assert trace.recorded['max_context_tokens'] == 130202
    assert trace.recorded['n_cap_overshoot'] == 0
    assert trace.recorded['context_window'] == 131072 and trace.recorded['max_turns'] == 50
    assert trace.recorded['sampling_max_tokens'] is None
    hits = run_rules(trace)
    assert decide(hits) == 'suspicious'  # saved peak alone cannot prove the last request's exact budget
    digest = render(trace, hits)
    assert 'Exact per-request estimates and requested output budgets are not saved' in digest
    assert 'the room left under the context window' in digest


@pytest.mark.parametrize('sent', [False, True, None])
def test_hle_recorded_image_policy_visible_without_inferring_legacy_policy(tmp_path, sent):
    run = tmp_path / 'hle' / 'r'
    cfg = {'benchmark': 'HLE-Diamond'}
    if sent is not None:
        cfg['image_inputs'] = {'sent': sent, 'why': 'recorded protocol'}
    _w(run / 'config.json', cfg)
    _w(run / 'transcripts/C.r0.json', {'generation': {'status': 'ok', 'content': 'The figure is absent.',
        'finish_reason': 'stop', 'usage': {'completion_tokens': 12}}})
    trace = open_run(run).load('C.r0')
    text = render(trace, run_rules(trace))
    assert ('input_modality_policy' in trace.recorded) is (sent is False)
    if sent is not None:
        assert trace.recorded['image_inputs']['sent'] is sent
        assert f'"sent": {str(sent).lower()}' in text
    else:
        assert 'image_inputs' not in trace.recorded


PYTEST_OUT = """============================= test session starts ==============================
collected 2 items

../tests/test_outputs.py F.                                                [100%]

=================================== FAILURES ===================================
>       assert len(failed_vectors) == 0
E       AssertionError: Failed on some tests
------------------------------ Captured log call -------------------------------
WARNING  urllib3.connectionpool:connectionpool.py:874 Retrying after connection broken by 'NewConnectionError("HTTPConnection(host='localhost', port=36239): Failed to establish a new connection: [Errno 111] Connection refused")'
==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED ../tests/test_outputs.py::test_clean_html_unchanged
FAILED ../tests/test_outputs.py::test_filter_blocks_xss - AssertionError: Fai...
=================== 1 failed, 1 passed in 576.34s (0:09:36) ====================
"""


def test_pytest_assertion_failures_are_not_grader_crash(tmp_path):
    """Selenium teardown retries in "Captured log call" while the tests ran and failed on an assertion: the grader
    worked. A collection error stays a grader_crash candidate."""
    run = tmp_path / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    _pi_attempt(run, 'xss__a0__r0', out, _session(TASK, _call('make'), _res('ok')), verifier=PYTEST_OUT)
    coll = PYTEST_OUT.replace('FAILED ../tests/test_outputs.py::test_filter_blocks_xss - AssertionError: Fai...',
                              'ERROR ../tests/test_outputs.py - ModuleNotFoundError: No module named selenium').replace(
        '1 failed, 1 passed in', '1 passed, 1 error in')
    _pi_attempt(run, 'coll__a0__r0', out, _session(TASK, _call('make'), _res('ok')), verifier=coll)
    L = open_run(run)
    flag = {'category': 'grader_crash', 'evidence': 'NewConnectionError localhost Connection refused',
            'ref': '[verifier/test-stdout.txt]', 'confidence': 0.8}
    t = L.load('xss__a0__r0')
    ev = next(e for e in t.events if e.source == 'verifier')
    assert ev.text.startswith('pytest summary: ') and 'urllib3' not in ev.text
    hits = run_rules(t)
    row = build_row(t, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)
    assert row['flagged'] is False and row['flags'][0]['demoted'] == 'pytest_assertions_only'
    t = L.load('coll__a0__r0')
    hits = run_rules(t)
    assert build_row(t, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)['flagged'] is True


def test_proxy_503_for_made_up_hosts(tmp_path, monkeypatch):
    """The proxy answers CONNECT to a host with no DNS record with 503: hosts the model made up are not an outage when
    other hosts worked. A host that fails then works, an attempt where most hosts fail, or no usable DNS stays flagged."""
    from aa_evals.tracejudge import rules
    real = {'blast.ncbi.nlm.nih.gov', 'data.rcsb.org', 'www.ebi.ac.uk'}
    monkeypatch.setattr(rules, 'resolves', lambda h: h in real)
    e503 = "ERR URLError(OSError('Tunnel connection failed: 503 Service Unavailable'))"
    ok = [_call('curl -s https://data.rcsb.org/rest/v1/core/entry/1ABC'), _res('{"entry": 1}'),
          _call('curl -s https://www.ebi.ac.uk/api/x'), _res('<xml/>'),
          _call('curl -s https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Put'), _res('RID = 7XYZ')]
    run = tmp_path / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    _pi_attempt(run, 'made__a0__r0', out, _session(TASK, *ok,
                _call('python3 try.py https://blast.ccb.jhu.edu/blast/ https://blast.sdsu.edu/'),
                _res(f'https://blast.ccb.jhu.edu/blast/ {e503}\nhttps://blast.sdsu.edu/ {e503}')))
    _pi_attempt(run, 'flaky__a0__r0', out, _session(TASK, *ok,     # same real host fails, then works: a real blip
                _call('curl https://www.ebi.ac.uk/api/y'), _res('curl: (56) CONNECT tunnel failed, response 503')))
    _pi_attempt(run, 'most__a0__r0', out, _session(TASK, _call('curl https://data.rcsb.org/a'), _res('ok'),
                _call('python3 t.py https://x1.example https://x2.example'), _res(f'https://x1.example {e503}\nhttps://x2.example {e503}')))
    L = open_run(run)
    flag = {'category': 'network', 'evidence': f'https://blast.ccb.jhu.edu/blast/ {e503}', 'ref': 'turn 8', 'confidence': 1.0}
    assert rules.demotion(L.load('made__a0__r0'), flag) == 'proxy_503_nonexistent_host'
    assert rules.demotion(L.load('flaky__a0__r0'), flag) is None
    assert rules.demotion(L.load('most__a0__r0'), flag) is None
    monkeypatch.setattr(rules, 'resolves', lambda h: None)          # DNS unusable from the judge: keep the flag
    assert rules.demotion(L.load('made__a0__r0'), flag) is None


def test_proxy_503_amid_site_rate_limits(tmp_path, monkeypatch):
    """Bulk-scraping a site that answers 429/403 and then refuses connections (proxy 503 for that site only), while
    other hosts work: the site's limit, not an outage. A 503 to a host that never throttled stays flagged."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, 'resolves', lambda h: True)          # real hosts: #216's rule does not apply
    e503 = '<urlopen error Tunnel connection failed: 503 Service Unavailable>'
    ok = [_call('curl -s https://pmc.ncbi.nlm.nih.gov/a'), _res('<html>ok</html>'),
          _call('curl -s https://en.wikipedia.org/wiki/X'), _res('<html>ok</html>')]
    scrape = _call('python3 dl.py https://web.archive.org/web/2025id_/https://pubs.acs.org/x.pdf')
    batch = '\n'.join([f'p{i} ERR https://web.archive.org/web/2025id_/https://pubs.acs.org/p{i}.pdf HTTP Error 429: Too Many Requests'
                        for i in range(5)] + [f'p{i} ERR https://web.archive.org/web/2024id_/https://pubs.acs.org/p{i}.pdf {e503}'
                                              for i in range(6)] + ['p9 OK 812573'])
    run = tmp_path / 'tbsci_lite' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    _pi_attempt(run, 'scrape__a0__r0', out, _session(TASK, *ok, scrape, _res(batch)))
    _pi_attempt(run, 'other__a0__r0', out, _session(TASK, *ok, scrape, _res(batch),
                _call('curl https://huggingface.co/api/x'), _res(f'https://huggingface.co/api/x {e503}')))
    L = open_run(run)
    flag = {'category': 'network', 'evidence': e503, 'ref': 'turn 6', 'confidence': 1.0}
    assert rules.demotion(L.load('scrape__a0__r0'), flag) == 'proxy_503_amid_site_limits'
    assert rules.demotion(L.load('other__a0__r0'), flag) is None


def test_single_turn_retried_and_recovered(tmp_path):
    """HLE: the serving gate answered "server at capacity" 503 a few times, then the request finished with an answer:
    a network flag on those retries is demoted. Retries that end with no answer stay flagged."""
    run = tmp_path / 'hle' / 'r'
    _w(run / 'config.json', {'benchmark': 'hle'})
    gate = [{'attempt': j, 'error': "Retryable: HTTP 503: {'error': {'message': 'gate: server at capacity'}}"} for j in (1, 2)]
    _w(run / 'transcripts' / 'q1.r0.json', {'generation': {'content': 'The answer is 42.', 'finish_reason': 'stop',
                                                           'usage': {'completion_tokens': 900}, 'status': 'ok',
                                                           'attempts': gate}})
    _w(run / 'transcripts' / 'q2.r0.json', {'generation': {'content': '', 'finish_reason': None,
                                                           'usage': {'completion_tokens': 0}, 'status': 'ok',
                                                           'attempts': gate}})
    from aa_evals.tracejudge.rules import demotion
    L = open_run(run)
    flag = {'category': 'network', 'evidence': 'Retryable: HTTP 503 gate: server at capacity', 'ref': 'client:attempt 1',
            'confidence': 1.0}
    t1, t2 = L.load('q1.r0'), L.load('q2.r0')
    assert any(e.source == 'client' for e in t1.events), 'the HLE adapter must surface client retries as events'
    assert demotion(t1, flag) == 'retried_and_recovered'
    assert demotion(t2, flag) is None


def test_agentic_ended_in_reasoning(tmp_path):
    """AutomationBench: the last message is unclosed reasoning (with a dangling </tool_call> drafted inside), no content,
    no tool call, finish stop: EOS inside the reasoning, a model outcome. An empty turn without reasoning (the injected
    empty response) or with an unknown stop reason stays suspicious."""
    a = tmp_path / 'automationbench' / 'r'
    think = 'Drafting the email to the team about competitor filings. ' * 10 + '</parameter></function></tool_call>'
    base = [{'role': 'user', 'content': 'monitor patents'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'sheets_get', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': '{"rows": []}'}]
    rows = [{'item_id': 'eos', 'repeat': 0, 'status': 'ok', 'messages': base + [{'role': 'assistant', 'reasoning_content': think}],
             'debug': {'stop_reasons': ['tool_calls', 'stop']}},
            {'item_id': 'empty', 'repeat': 0, 'status': 'ok', 'messages': base + [{'role': 'assistant', 'content': ''}],
             'debug': {'stop_reasons': ['tool_calls', 'stop']}},
            {'item_id': 'unknown', 'repeat': 0, 'status': 'ok', 'messages': base + [{'role': 'assistant', 'reasoning_content': think}],
             'debug': {}}]
    _w(a / 'transcripts.jsonl', '\n'.join(json.dumps(r) for r in rows) + '\n')
    L = open_run(a)
    assert decide(run_rules(L.load('eos/r0'))) == 'model'
    assert decide(run_rules(L.load('empty/r0'))) == 'suspicious'
    assert decide(run_rules(L.load('unknown/r0'))) == 'suspicious'


def test_squid_dns_fail_in_curl_output(tmp_path, monkeypatch):
    """GDPval: curl -v to two made-up hosts prints only Squid's ERR_DNS_FAIL and "CONNECT tunnel failed, response 503"
    (no host on the line); a batch of curls printing "000 <url>" must not count as those hosts working."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, 'resolves', lambda h: h in {'app.leg.wa.gov', 'doh.wa.gov', 'www.ncsbn.org', 'dhp.virginia.gov'})
    a = tmp_path / 'gdpval' / 'r'

    def call(cmd):
        return {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'code_exec', 'arguments': json.dumps({'cmd': cmd})}}]}

    def res(text):
        return {'role': 'tool', 'content': text}
    squid = '< X-Squid-Error: ERR_DNS_FAIL 0\n* CONNECT tunnel failed, response 503\n* closing connection #0'
    msgs = [{'role': 'user', 'content': 'compare nursing boards'},
            call('curl -s https://app.leg.wa.gov/rcw/'), res('HTTP/1.1 200 OK\n<html>RCW 18.79</html>'),
            call('curl -s https://doh.wa.gov/licenses'), res('HTTP/1.1 200 OK\n<html>DOH</html>'),
            call('curl -s https://www.ncsbn.org/nlc'), res('HTTP/1.1 200 OK\n<html>NLC</html>'),
            call('for u in https://ponursing.gov/ https://dhp.virginia.gov/; do curl -s -o /dev/null -w "%{http_code}" $u; done'),
            res('000  https://ponursing.gov/\n000  https://dhp.virginia.gov/'),
            call('curl -sv https://ponursing.gov/ 2>&1 | tail -3; curl -sv https://www.aznursing.gov/ 2>&1 | tail -3'),
            res(squid + '\n====AZ====\n' + squid)]
    _w(a / 'tr' / 'x.json', {'messages': msgs})
    _w(a / 'items.jsonl', json.dumps({'item_id': 'x', 'repeat': 3, 'status': 'ok', 'score': 0, 'transcript': 'tr/x.json'}) + '\n')
    t = open_run(a).load('x/r3')
    ok, bad = rules.proxy_hosts(t)
    assert set(bad) == {'ponursing.gov', 'www.aznursing.gov'} and 'dhp.virginia.gov' not in ok
    # an empty curl -s inside Stirrup's wrapper is no response either
    msgs[-2:-2] = [call('curl -s https://ponursing.gov/graphql'), res('<shell_results>\n<exit_code>0</exit_code>\n<stdout></stdout>\n<stderr></stderr>\n</shell_results>')]
    _w(a / 'tr' / 'x.json', {'messages': msgs})
    t = open_run(a).load('x/r3')
    ok, bad = rules.proxy_hosts(t)
    assert 'ponursing.gov' in bad and 'ponursing.gov' not in ok
    flag = {'category': 'network', 'evidence': 'X-Squid-Error: ERR_DNS_FAIL 0 * CONNECT tunnel failed, response 503',
            'ref': '[msg:12]', 'confidence': 1.0}
    assert rules.demotion(t, flag) == 'proxy_503_nonexistent_host'


class _AlwaysNetwork:
    """A fake LLM judge that always flags the invented host's proxy 503 (as the real one sometimes does)."""
    def describe(self):
        return {'model': 'fake'}

    async def judge(self, text):
        return {'ok': True, 'reason': 'proxy 503', 'prompt_tokens': 1, 'completion_tokens': 1, 'latency_s': 0.0,
                'flags': [{'category': 'network', 'confidence': 0.9, 'ref': 'turn 8',
                           'evidence': "https://blast.ccb.jhu.edu/ ERR Tunnel connection failed: 503 Service Unavailable"}]}


def test_demotion_is_deterministic_across_judge_runs(tmp_path, monkeypatch):
    """The DNS verdict behind a demotion is cached in a shared file: a later judge run whose resolver times out (a new
    process, no in-memory cache) demotes the same flag again instead of keeping it."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, 'DNS_CACHE', str(tmp_path / 'dns.json'))
    monkeypatch.setattr(rules, '_DNS', {})
    real = {'data.rcsb.org', 'www.ebi.ac.uk', 'blast.ncbi.nlm.nih.gov'}
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'ok' if h in real else 'nxdomain')
    monkeypatch.setattr(rules, '_public_dns', lambda h: None)
    e503 = "ERR URLError(OSError('Tunnel connection failed: 503 Service Unavailable'))"
    run = tmp_path / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    _pi_attempt(run, 'made__a0__r0', {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}},
                _session(TASK, _call('curl -s https://data.rcsb.org/x'), _res('{"entry": 1}'),
                         _call('curl -s https://www.ebi.ac.uk/x'), _res('<xml/>'),
                         _call('curl -s https://blast.ncbi.nlm.nih.gov/x'), _res('RID = 7XYZ'),
                         _call('python3 t.py https://blast.ccb.jhu.edu/'), _res(f'https://blast.ccb.jhu.edu/ {e503}')))
    rows = asyncio.run(judge_run(run, _AlwaysNetwork(), 'all', log=lambda *a: None))
    assert rows[0]['flagged'] is False and rows[0]['flags'][0]['demoted'] == 'proxy_503_nonexistent_host'
    monkeypatch.setattr(rules, '_DNS', {})                          # a new judge process
    monkeypatch.setattr(rules, '_local_dns', lambda h: None)        # whose resolver times out
    rows = asyncio.run(judge_run(run, _AlwaysNetwork(), 'all', rejudge=True, log=lambda *a: None))
    assert rows[0]['flagged'] is False and rows[0]['flags'][0]['demoted'] == 'proxy_503_nonexistent_host'
    assert fold_summary(run, {})['n_flagged'] == 0


def test_servfail_everywhere_counts_as_unresolvable(tmp_path, monkeypatch):
    """katzdeli.com: EAI_AGAIN locally and SERVFAIL in public DNS -> doesn't resolve. A name that fails locally but
    resolves publicly is ours to fix (True: the flag stays)."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, 'DNS_CACHE', str(tmp_path / 'dns.json'))
    monkeypatch.setattr(rules, '_DNS', {})
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'again')
    monkeypatch.setattr(rules, '_public_dns', lambda h: {'katzdeli.com': 'servfail', 'katzsdelicatessen.com': 'ok'}[h])
    assert rules.resolves('katzdeli.com') is False
    assert rules.resolves('katzsdelicatessen.com') is True


FEAL_OUT = """============================= test session starts ==============================
collected 1 item

../tests/test_outputs.py F                                                 [100%]

=================================== FAILURES ===================================
>   import numpy as np
E   ModuleNotFoundError: No module named 'numpy'

{where}:39: ModuleNotFoundError
=========================== short test summary info ============================
FAILED ../tests/test_outputs.py::test_feal_differential_cryptanalysis_attack
============================== 1 failed in 1.47s ===============================
"""


def test_solution_import_in_verifier_env(tmp_path):
    """TB 2.1 runs pytest in its own uv venv: the model's attack.py importing numpy from the image's system python
    fails there. That's the task's test environment and the solution, not our harness; the same import failing in
    the test file itself stays flagged."""
    run = tmp_path / 'tb21' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    _pi_attempt(run, 'feal__a0__r0', out, _session(TASK, _call('python3 attack.py'), _res('ok')),
                verifier=FEAL_OUT.format(where='attack.py'))
    _pi_attempt(run, 'tst__a0__r0', out, _session(TASK, _call('python3 attack.py'), _res('ok')),
                verifier=FEAL_OUT.format(where='/tests/test_outputs.py'))
    from aa_evals.tracejudge.rules import demotion
    L = open_run(run)
    flag = {'category': 'harness_error', 'confidence': 1.0, 'ref': '[verifier/test-stdout.txt]',
            'evidence': "verifier: E ModuleNotFoundError: No module named 'numpy' attack.py:39"}
    assert demotion(L.load('feal__a0__r0'), flag) == 'solution_import_in_verifier_env'
    assert demotion(L.load('tst__a0__r0'), flag) is None


def test_lone_invented_host_no_other_fetches(tmp_path, monkeypatch):
    """tautomerization-barrier: the only network use is one probe of crest-online.com (no such name anywhere), after
    "curl: command not found" (which must not count as reaching it). Unresolvable locally and publicly -> demoted
    with no other hosts. A real host failing once with nothing else to compare stays flagged."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'ok' if h == 'pypi.org' else 'nxdomain')
    monkeypatch.setattr(rules, '_public_dns', lambda h: 'ok' if h == 'pypi.org' else 'nxdomain')
    e503 = 'fail <urlopen error Tunnel connection failed: 503 Service Unavailable>'
    run = tmp_path / 'tbsci_lite' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished', 'context_limit': True}}
    _pi_attempt(run, 'crest__a1__r0', out, _session(TASK,
                _call('timeout 15 curl -sI https://crest-online.com'), _res("timeout: failed to run command 'curl': No such file or directory"),
                _call("python3 -c \"import urllib.request; urllib.request.urlopen('https://crest-online.com')\""), _res(e503)))
    _pi_attempt(run, 'pypi__a1__r0', out, _session(TASK,
                _call("python3 -c \"import urllib.request; urllib.request.urlopen('https://pypi.org/simple/')\""), _res(e503)))
    L = open_run(run)
    t = L.load('crest__a1__r0')
    assert rules.proxy_hosts(t) == ({}, {'crest-online.com': 4})
    flag = {'category': 'network', 'evidence': e503, 'ref': 'turn 4', 'confidence': 1.0}
    assert rules.demotion(t, flag) == 'proxy_503_nonexistent_host'
    assert rules.demotion(L.load('pypi__a1__r0'), flag) is None
    monkeypatch.setattr(rules, '_DNS', {})
    monkeypatch.setattr(rules, 'DNS_CACHE', str(tmp_path / 'other.json'))
    monkeypatch.setattr(rules, '_public_dns', lambda h: None)       # public check unreachable (proxy down): keep
    assert rules.demotion(t, flag) is None


def test_urllib_503_attributed_by_label(tmp_path, monkeypatch):
    """gdpval ec2fccc9/r4: one python script fetches opensea.io (200s) and api.foundation.app (no such name); urllib's
    "Tunnel connection failed: 503" names no host, but the script labels it ("FOUNDATION ERR", "### FOUNDATION API").
    A 503 line nothing identifies keeps the flag."""
    from aa_evals.tracejudge import rules
    real = {'opensea.io', 'developer.mozilla.org', 'www.mojeek.com', 'www.bing.com'}
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'ok' if h in real else 'nxdomain')
    monkeypatch.setattr(rules, '_public_dns', lambda h: 'ok' if h in real else 'nxdomain')
    a = tmp_path / 'gdpval' / 'r'

    def call(cmd):
        return {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'code_exec', 'arguments': json.dumps({'cmd': cmd})}}]}

    def res(out):
        return {'role': 'tool', 'content': f'<shell_results>\n<exit_code>0</exit_code>\n<stdout>{out}</stdout>\n<stderr></stderr>\n</shell_results>'}
    e503 = "ERR URLError(OSError('Tunnel connection failed: 503 Service Unavailable'))"
    script = ('python3 - <<EOF\nfor u in ["https://opensea.io/collection/drive"]: print(u, get(u))\n'
              'req = urllib.request.Request("https://api.foundation.app/graphql", data=q)\nEOF')
    base = [{'role': 'user', 'content': 'profile the photographer'},
            call('curl -s https://developer.mozilla.org/x'), res('HTTP/1.1 200 OK'),
            call('curl -s https://www.mojeek.com/search?q=x'), res('HTTP/1.1 200 OK'),
            call(script), res(f'https://opensea.io/collection/drive 200 TITLE: DRIVE | OpenSea\n====\nFOUNDATION {e503}\n'),
            call(script), res(f'### OPENSEA\nhttps://opensea.io/collection/drive 200\n\n### FOUNDATION API retry\n{e503}\n')]
    odd = base[:5] + [call('python3 - <<EOF\nget("https://opensea.io/a"); get("https://www.bing.com/b")\nEOF'),
                      res(f'opensea ok 200 https://opensea.io/a\n{e503}\n')]
    _w(a / 'tr' / 'x.json', {'messages': base})
    _w(a / 'tr' / 'y.json', {'messages': odd})
    _w(a / 'items.jsonl', json.dumps({'item_id': 'x', 'repeat': 4, 'status': 'ok', 'score': 0, 'transcript': 'tr/x.json'}) + '\n'
       + json.dumps({'item_id': 'y', 'repeat': 4, 'status': 'ok', 'score': 0, 'transcript': 'tr/y.json'}) + '\n')
    L = open_run(a)
    t = L.load('x/r4')
    ok, bad, unattributed = rules.proxy_scan(t)
    assert set(bad) == {'api.foundation.app'} and 'opensea.io' in ok and unattributed == 0
    flag = {'category': 'network', 'evidence': f'FOUNDATION {e503}', 'ref': '[msg:6]', 'confidence': 1.0}
    assert rules.demotion(t, flag) == 'proxy_503_nonexistent_host'
    assert rules.demotion(L.load('y/r4'), flag) is None      # the 503 line names nothing: keep the flag


def test_remote_site_answer_vs_proxy_refusal(tmp_path, monkeypatch):
    """The remote server answering 429/503 itself (httpx "Server error '503 ...' for url", or DDG's anti-bot page) is the
    site's outcome; Stirrup web_fetch's bare "<error>503 Service Unavailable</error>" is httpx's ProxyError (the proxy
    refused CONNECT, as in gdpval bbe0a93b/r4 for html.duckduckgo.com) and stays flagged. Throttling is counted."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'ok')
    monkeypatch.setattr(rules, '_public_dns', lambda h: 'ok')
    a = tmp_path / 'gdpval' / 'r'

    def fetch(u):
        return {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'fetch_web_page', 'arguments': json.dumps({'url': u})}}]}

    def res(u, err=None, body='ok'):
        return {'role': 'tool', 'content': f'<web_fetch><url>{u}</url>' + (f'<error>{err}</error>' if err else f'<content>{body}</content>') + '</web_fetch>'}
    ok = [{'role': 'user', 'content': 'find the phone number'},
          fetch('https://www.bing.com/search?q=a'), res('https://www.bing.com/search?q=a'),
          fetch('https://en.wikipedia.org/wiki/A'), res('https://en.wikipedia.org/wiki/A')]
    ddg = 'https://html.duckduckgo.com/html/?q=a'
    remote = "Server error '503 Service Unavailable' for url 'https://html.duckduckgo.com/html/?q=a'"
    rows = {'site': ok + [fetch(ddg), res(ddg, remote)],
            'chal': ok + [fetch(ddg), res(ddg, body='Unfortunately, bots use DuckDuckGo too. Please complete the challenge')],
            'proxy': ok + [fetch(ddg), res(ddg, '503 Service Unavailable')]}
    items = []
    for k, msgs in rows.items():
        _w(a / 'tr' / f'{k}.json', {'messages': msgs})
        items.append(json.dumps({'item_id': k, 'repeat': 4, 'status': 'ok', 'score': 0, 'transcript': f'tr/{k}.json'}))
    _w(a / 'items.jsonl', '\n'.join(items) + '\n')
    L = open_run(a)
    flag = lambda ev: {'category': 'network', 'evidence': ev, 'ref': '[msg:6]', 'confidence': 1.0}
    assert rules.demotion(L.load('site/r4'), flag(remote)) == 'remote_site_answer'
    assert rules.demotion(L.load('chal/r4'), flag('bots use DuckDuckGo too')) == 'remote_site_answer'
    assert rules.demotion(L.load('proxy/r4'), flag('<error>503 Service Unavailable</error>')) is None
    assert rules.remote_limits(L.load('site/r4')) == {'html.duckduckgo.com': 1}
    assert rules.remote_limits(L.load('proxy/r4')) == {}
    rows_ = asyncio.run(judge_run(a, None, 'none', log=lambda *x: None))
    s = fold_summary(a, {})
    assert s['tracejudge']['n_remote_site_limited'] == 2 and s['tracejudge']['remote_limited_hosts'] == {'html.duckduckgo.com': 2}


def test_egress_blocked_host_specific_refusal(tmp_path, monkeypatch):
    """The proxy refused CONNECT to html.duckduckgo.com (web_fetch's bare 503 = httpx ProxyError) while bing worked in
    the same batch of calls: our egress blocked by that upstream, demoted and counted as n_egress_blocked. A refusal
    with no other host working in the same call, or refusals to most hosts of a call, stay flagged."""
    from aa_evals.tracejudge import rules
    monkeypatch.setattr(rules, '_local_dns', lambda h: 'ok')
    monkeypatch.setattr(rules, '_public_dns', lambda h: 'ok')
    a = tmp_path / 'gdpval' / 'r'

    def fetch(*us):
        return {'role': 'assistant', 'content': None, 'tool_calls': [
            {'function': {'name': 'fetch_web_page', 'arguments': json.dumps({'url': u})}} for u in us]}

    def res(u, err=None):
        return {'role': 'tool', 'content': f'<web_fetch><url>{u}</url>' + (f'<error>{err}</error>' if err else '<content>results</content>') + '</web_fetch>'}
    ddg, bing, wiki, ecosia = ('https://html.duckduckgo.com/html/?q=a', 'https://www.bing.com/search?q=a',
                               'https://en.wikipedia.org/wiki/A', 'https://www.ecosia.org/search?q=a')
    refused = '503 Service Unavailable'
    task = {'role': 'user', 'content': 'find the phone number'}
    rows = {'same': [task, fetch(ddg, bing), res(ddg, refused), res(bing)],
            'solo': [task, fetch(bing), res(bing), fetch(ddg), res(ddg, refused)],
            'most': [task, fetch(ddg, wiki, ecosia), res(ddg, refused), res(wiki, refused), res(ecosia)]}
    items = []
    for k, msgs in rows.items():
        _w(a / 'tr' / f'{k}.json', {'messages': msgs})
        items.append(json.dumps({'item_id': k, 'repeat': 4, 'status': 'ok', 'score': 0, 'transcript': f'tr/{k}.json'}))
    _w(a / 'items.jsonl', '\n'.join(items) + '\n')
    L = open_run(a)
    flag = {'category': 'network', 'evidence': f'<error>{refused}</error>', 'ref': '[msg:2]', 'confidence': 1.0}
    assert rules.demotion(L.load('same/r4'), flag) == 'egress_blocked'
    assert rules.egress_blocked(L.load('same/r4'))[0] == {'html.duckduckgo.com': 1}
    assert rules.demotion(L.load('solo/r4'), flag) is None
    assert rules.demotion(L.load('most/r4'), flag) is None
    asyncio.run(judge_run(a, None, 'none', log=lambda *x: None))
    tj = fold_summary(a, {})['tracejudge']
    assert tj['n_egress_blocked'] == 1 and tj['egress_blocked_hosts'] == {'html.duckduckgo.com': 1}


def test_model_cleared_its_own_dir(tmp_path):
    """specimen-provenance: the model's script empties its --output-dir, and it ran it with --output-dir /app; a later
    `ls -la /app` shows total 0. A model outcome. An empty /app nothing in the trace touched stays flagged."""
    from aa_evals.tracejudge.rules import demotion
    run = tmp_path / 'tbsci_lite' / 'r1'
    (run / 'queue').mkdir(parents=True)
    out = {'infrastructure_failure': False, 'score': 0.0, 'stats': {'end_state': 'finished'}}
    empty = 'total 0\ndrwxr-xr-x 1 root root 26 Nov  5 16:13 .\n/app'
    _pi_attempt(run, 'own__a3__r0', out, _session(TASK,
                _call('cd /app && python3 prov.py --input-dir /app/data --output-dir /app'), _res('EXIT=0'),
                _call('ls -la /app; pwd'), _res(empty)))
    _pi_attempt(run, 'none__a3__r0', out, _session(TASK, _call('python3 prov.py --output-dir /tmp/out'), _res('ok'),
                                                   _call('ls -la /app; pwd'), _res(empty)))
    L = open_run(run)
    flag = {'category': 'sandbox_kill', 'confidence': 1.0, 'ref': '[turn 4]',
            'evidence': '[turn 3] CALL bash: ls -la /app; pwd [turn 4] OUTPUT: total 0 ... /app'}
    assert demotion(L.load('own__a3__r0'), flag) == 'model_touched_dir'
    assert demotion(L.load('none__a3__r0'), flag) is None


def test_offline_tool_denial_is_expected_but_model_endpoint_outage_is_infra(tmp_path):
    from aa_evals.tracejudge.trace import Trace, Turn, Event
    from aa_evals.tracejudge.rules import demotion
    call = Turn(6, 'assistant', 'tool_call', 'bash: curl https://pypi.org/simple/', 'turn 6')
    tool = Turn(7, 'tool', 'tool_result', 'ProxyError: Unable to connect to proxy (Network is unreachable)', 'turn 7')
    flag = {'category': 'network', 'confidence': 1.0, 'evidence': 'Unable to connect to proxy', 'ref': '[turn 7]'}
    trace = Trace('tb21', 'x', 'x', 0, 'agentic', {'score': 0, 'agent_internet': False}, turns=[call, tool])
    assert demotion(trace, flag) == 'offline_agent_network_denied'
    trace.recorded = {'score': 0, 'agent_network': {'mode': 'offline'}}
    assert demotion(trace, flag) == 'offline_agent_network_denied'
    trace.recorded = {'score': 0}  # historical runs without an explicit policy retain their rules
    assert demotion(trace, flag) != 'offline_agent_network_denied'
    trace.recorded = {'status': 'infra_error', 'infrastructure_failure': True,
                      'infra_reason': 'pi_backend_unreachable', 'agent_internet': False}
    trace.events = [Event('proxy.log', 'model connection refused', 'proxy.log:1')]
    outage = {**flag, 'evidence': 'model connection refused', 'ref': 'proxy.log:1'}
    assert demotion(trace, outage) is None
    hits = run_rules(trace)
    assert build_row(trace, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)['flagged']


def test_layout_exposes_explicit_offline_policy(tmp_path):
    run = tmp_path / 'tb21' / 'offline'
    (run / 'queue').mkdir(parents=True)
    _w(run / 'config.json', {'benchmark': 'tb21', 'agent_internet': False})
    _pi_attempt(run, 'x__a0__r0', {'score': 0, 'infrastructure_failure': False}, _session(TASK))
    assert open_run(run).load('x__a0__r0').recorded['agent_internet'] is False
    generic = tmp_path / 'gdpval' / 'offline'
    _w(generic / 'config.json', {'benchmark': 'gdpval', 'agent_network': {'mode': 'offline'}})
    _w(generic / 'items.jsonl', json.dumps({'key': 'x/r0', 'item_id': 'x', 'repeat': 0,
                                         'status': 'ok', 'score': 0}) + '\n')
    assert open_run(generic).load('x/r0').recorded['agent_network'] == {'mode': 'offline'}


def _offline_network_trace(call='curl https://pypi.org/simple/',
                           output='ProxyError: Unable to connect to proxy (Network is unreachable)'):
    from aa_evals.tracejudge.trace import Trace, Turn
    return Trace('tb21', 'offline', 'x', 0, 'agentic', {'score': 0, 'agent_internet': False}, turns=[
        Turn(6, 'assistant', 'tool_call', call, 'turn 6'),
        Turn(7, 'tool', 'tool_result', output, 'turn 7')], source='/run/attempts/x')


def _network_flag(ref='[turn 7]', evidence='Unable to connect to proxy'):
    return {'category': 'network', 'confidence': 1.0, 'evidence': evidence, 'ref': ref}


def test_offline_public_denial_stays_auditable_below_threshold():
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace()
    hits = run_rules(trace)
    assert decide(hits) == 'clean'
    row = build_row(trace, hits, decide(hits), {'ok': True, 'flags': [_network_flag()]}, None, None)
    assert not row['flagged']
    assert row['flags'][0]['demoted'] == 'offline_agent_network_denied'
    assert row['recorded']['agent_internet'] is False
    assert 'offline_tool_network_denied' in row['notes']
    assert demotion(trace, _network_flag('[turn 6] [turn 7]')) is None
    assert demotion(trace, _network_flag('[turn 7] [turn 99]')) is None
    assert demotion(trace, _network_flag('[unknown]', 'Unable to connect to proxy')) is None
    assert demotion(trace, _network_flag('[turn 7]', 'unrelated backend failure')) is None


@pytest.mark.parametrize('url', ['http://localhost:8000', 'http://127.0.0.1:8000', 'http://[::1]:8000',
                                 'http://sim:8788', 'http://172.31.0.3:8788', 'http://224.0.0.1:8000', 'http://api.internal:8000',
                                 'https://api.example.org/v1/chat/completions'])
def test_offline_policy_preserves_local_and_model_endpoint_failures(url):
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace('curl ' + url)
    hits = run_rules(trace)
    assert decide(hits) == 'suspicious'
    assert not any(h.rule == 'offline_tool_network_denied' for h in hits)
    assert demotion(trace, _network_flag()) is None
    assert build_row(trace, hits, decide(hits), {'ok': True, 'flags': [_network_flag()]}, None, None)['flagged']


@pytest.mark.parametrize('source,text', [('proxy.log', 'model relay connection refused'),
                                       ('client', 'APIConnectionError: model endpoint unreachable'),
                                       ('harness', 'model request timed out'),
                                       ('verifier', 'ProxyError: grader bootstrap connection refused')])
def test_offline_tool_note_does_not_explain_independent_harness_outage(source, text):
    from aa_evals.tracejudge.trace import Event
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace()
    trace.events.append(Event(source, text, source + ':1'))
    # Even a judge citing the expected tool denial cannot dismiss separate harness evidence.
    assert demotion(trace, _network_flag()) is None
    hits = run_rules(trace)
    assert build_row(trace, hits, decide(hits), {'ok': True, 'flags': [_network_flag()]}, None, None)['flagged']
    assert demotion(trace, _network_flag(source + ':1', text)) is None


def test_offline_public_denial_does_not_deduplicate_later_local_failure():
    from aa_evals.tracejudge.trace import Turn
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace()
    trace.turns += [Turn(8, 'assistant', 'tool_call', 'curl http://warehouse-api:5000', 'turn 8'),
                    Turn(9, 'tool', 'tool_result', 'ProxyError: Unable to connect to proxy (Network is unreachable)', 'turn 9')]
    hits = run_rules(trace)
    assert any(h.rule == 'offline_tool_network_denied' for h in hits)
    assert any(h.rule == 'env_error_in_tool_output' and h.ref == 'turn 9' for h in hits)
    assert decide(hits) == 'suspicious'
    assert demotion(trace, _network_flag('[turn 9]')) is None


def test_offline_local_dns_and_ambiguous_targets_stay_reviewable():
    from aa_evals.tracejudge.rules import demotion
    local = _offline_network_trace('curl http://sim:8788', 'curl: (6) Could not resolve host: sim')
    hits = run_rules(local)
    assert decide(hits) == 'suspicious'
    assert demotion(local, _network_flag(evidence='Could not resolve host: sim')) is None
    ambiguous = _offline_network_trace('python fetch.py')
    assert decide(run_rules(ambiguous)) == 'suspicious'
    assert demotion(ambiguous, _network_flag()) is None
    public = _offline_network_trace('curl https://pypi.org/simple/', 'curl: (6) Could not resolve host: pypi.org')
    assert decide(run_rules(public)) == 'clean'
    assert demotion(public, _network_flag(evidence='Could not resolve host: pypi.org')) == 'offline_agent_network_denied'


@pytest.mark.parametrize('other_call', ['curl http://sim:8788', 'python local_client.py'])
def test_mixed_parallel_offline_batch_is_not_attributed_to_external_call(other_call):
    from aa_evals.tracejudge.trace import Turn
    trace = _offline_network_trace()
    trace.turns.insert(1, Turn(6, 'assistant', 'tool_call', other_call, 'turn 6b'))
    assert decide(run_rules(trace)) == 'suspicious'


@pytest.mark.parametrize('policy', [None, 'offline', False, [], {'mode': 'legacy-online'}])
def test_malformed_or_historical_policy_never_implies_offline(policy):
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace()
    trace.recorded = {'score': 0, 'agent_network': policy}
    assert decide(run_rules(trace)) == 'suspicious'
    assert demotion(trace, _network_flag()) != 'offline_agent_network_denied'


def test_offline_policy_is_visible_in_pi_digest_and_row_override_is_authoritative(tmp_path):
    run = tmp_path / 'tb21' / 'offline'
    (run / 'queue').mkdir(parents=True)
    _w(run / 'config.json', {'benchmark': 'tb21', 'agent_internet': False})
    _pi_attempt(run, 'x__a0__r0', {'score': 0}, _session(TASK))
    trace = open_run(run).load('x__a0__r0')
    assert trace.recorded['agent_network'] == {'mode': 'offline'}
    assert '"agent_network": {"mode": "offline"}' in render(trace, [])
    generic = tmp_path / 'gdpval' / 'offline'
    _w(generic / 'config.json', {'benchmark': 'gdpval', 'agent_network': {'mode': 'offline'}})
    _w(generic / 'items.jsonl', json.dumps({'key': 'x/r0', 'item_id': 'x', 'repeat': 0,
                                         'status': 'ok', 'score': 0, 'agent_network': {}}) + '\n')
    assert open_run(generic).load('x/r0').recorded['agent_network'] == {}


def test_judge_describes_new_policy_prompt_version_without_endpoint_call():
    from types import SimpleNamespace
    from aa_evals.tracejudge.judge import LLMJudge, PROMPT_VERSION, system_prompt
    judge = object.__new__(LLMJudge)
    judge.client = SimpleNamespace(endpoint=SimpleNamespace(served_name='fake'))
    judge.spec, judge.effort, judge.thinking, judge.max_tokens = 'fake', None, None, 8192
    assert judge.describe()['prompt_version'] == PROMPT_VERSION == 'tj-v5'
    prompt = system_prompt()  # formatting the changed prompt must still satisfy the verdict schema
    assert 'agent_internet=false' in prompt and 'agent_network.mode=offline' in prompt
    assert 'task-local servers/services' in prompt and 'endpoint or relay' in prompt


@pytest.mark.parametrize('output', [
    "ProxyError: Unable to connect to proxy; HTTPConnectionPool(host='localhost', port=8000)",
    "ProxyError: Unable to connect to proxy; task server localhost:8000 is unavailable",
    "ProxyError: Unable to connect to proxy; HTTPConnectionPool(host='10.22.0.107', port=8080)",
])
def test_explicit_local_or_private_failure_in_output_is_not_hidden_by_public_call(output):
    from aa_evals.tracejudge.rules import demotion
    trace = _offline_network_trace(output=output)
    assert decide(run_rules(trace)) == 'suspicious'
    assert demotion(trace, _network_flag()) is None


def test_openai_turns_flat_and_nested_tool_calls():
    """AutomationBench stores tool calls flat ({id, name, arguments}); OpenAI nests them under 'function'.
    Both must reach the digest with their real name and arguments, never 'None: None'."""
    from aa_evals.tracejudge.layouts import openai_turns
    msgs = [{'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'a', 'name': 'api_search', 'arguments': '{"query": "contacts"}'},
                {'id': 'b', 'type': 'function', 'function': {'name': 'api_fetch', 'arguments': '{"id": 1}'}}]},
            {'role': 'tool', 'content': 'ok', 'tool_call_id': 'a'}]
    calls = [t.text for t in openai_turns(msgs) if t.kind == 'tool_call']
    assert calls == ['api_search: {"query": "contacts"}', 'api_fetch: {"id": 1}']

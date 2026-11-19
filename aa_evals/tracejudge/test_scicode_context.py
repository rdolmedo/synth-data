"""Synthetic saved SciCode artifacts; CPU only, no generated code execution."""
import gzip
import json

import pytest

from aa_evals.scicode.prompts import extract_code
from aa_evals.tracejudge.core import build_row
from aa_evals.tracejudge.digest import render
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.rules import decide, run_rules


def saved_run(tmp_path, symbol='find_bound_states', raw=True):
    run = tmp_path / 'scicode' / 'r'
    (run / 'transcripts').mkdir(parents=True)
    (run / 'chains').mkdir()
    response = '```python\ndef solve():\n    import heapq\n    return heapq.heappop([])\n```'
    prompt = 'Task\nDEPENDENCIES:\nUse only these dependencies.\nimport numpy as np\nRESPONSE GUIDELINES:\nCode only'
    tail = f'Traceback (most recent call last):\n  File "/work/test.py", line 20, in {symbol}\nTypeError: bad endpoint'
    event = {'source': 'grader:np126', 'text': tail}
    cfg = {'benchmark': 'scicode', 'grading': {'rule': 'pass if any env passes (OR)'}}
    row = {'item_id': '12.8', 'repeat': 4, 'status': 'ok', 'score': 0,
           'transcript': 'transcripts/12.r4.jsonl#12.8/r4', 'chain': 'chains/12.r4.json'}
    transcript = {'key': '12.8/r4', 'generation': {'content': response, 'finish_reason': 'stop'}, 'events': [event]}
    grade = {'pass': False, 'per_env': {'np126': {'pass': False, 'rc': 1, 'timeout': False, 'tail': tail},
                                     'np22': {'pass': False, 'rc': 1, 'timeout': False,
                                              'tail': 'ImportError: unavailable dependency'}}}
    chain = {'steps': {'12.5': {'gen_status': 'ok', 'code': 'def find_bound_states():\n    return bad_call()\n'},
                       '12.8': {'gen_status': 'ok', 'code': extract_code(response), 'grade': grade}}}
    for path, obj in [(run / 'config.json', cfg), (run / 'items.jsonl', row),
                      (run / 'transcripts/12.r4.jsonl', transcript), (run / 'chains/12.r4.json', chain)]:
        path.write_text(json.dumps(obj) + '\n')
    if raw:
        with gzip.open(run / 'transcripts/12.r4.gen.jsonl.gz', 'wt') as f:
            f.write(json.dumps({'step': '12.8', 'prompt': prompt, 'content': response}) + '\n')
    return run, event


def test_protocol_origin_and_or_outcomes_are_visible(tmp_path):
    run, original = saved_run(tmp_path)
    t = open_run(run).load('12.8/r4')
    text = render(t, [])
    assert t.recorded['scicode_context_version'] == 1
    assert build_row(t, [], 'clean', None, None, None)['recorded']['scicode_context_version'] == 1
    assert 'pass if any env passes (OR)' in text
    assert '"np126": {"pass": false, "rc": 1, "timeout": false}' in text
    assert '"np22": {"pass": false, "rc": 1, "timeout": false}' in text
    assert 'Use only these dependencies.\nimport numpy as np' in text
    assert 'import lines removed' in text and 'matches this extractor: True' in text
    assert 'find_bound_states: definition present in generated step12.5' in text
    assert 'if test code shadows names' in text
    assert t.events[0].source == original['source'] and t.events[0].text == original['text']


@pytest.mark.parametrize('symbol', ['find_bound_states', 'verifier_bootstrap'])
def test_context_never_demotes_a_model_or_genuine_grader_flag(tmp_path, symbol):
    run, _ = saved_run(tmp_path, symbol=symbol)
    t = open_run(run).load('12.8/r4')
    text = render(t, [])
    if symbol == 'verifier_bootstrap':
        assert 'no definition found' in text and 'A genuine grader failure remains possible' in text
        assert 'verifier_bootstrap: definition present' not in text
    hits = run_rules(t)
    flag = {'category': 'grader_crash', 'evidence': 'TypeError: bad endpoint',
            'ref': 'events[0]', 'confidence': 1.0}
    row = build_row(t, hits, decide(hits), {'ok': True, 'flags': [flag]}, None, None)
    assert row['flagged'] is True and not row['flags'][0].get('demoted')


@pytest.mark.parametrize('mode', ['absent', 'corrupt', 'unmatched'])
def test_optional_raw_failure_keeps_generic_trace_usable(tmp_path, mode):
    run, original = saved_run(tmp_path, raw=False)
    raw = run / 'transcripts/12.r4.gen.jsonl.gz'
    if mode == 'corrupt':
        raw.write_bytes(b'not a gzip file')
    elif mode == 'unmatched':
        with gzip.open(raw, 'wt') as f:
            f.write(json.dumps({'step': '12.8', 'content': 'different generation', 'prompt': 'different'}) + '\n')
    t = open_run(run).load('12.8/r4')
    assert t.turns[-1].text.startswith('```python') and t.events[0].text == original['text']
    text = render(t, [])
    assert 'dependency/extraction context unavailable' in text
    assert 'generated step12.5' in text and 'matches this extractor: True' not in text


def test_missing_chain_and_raw_do_not_invent_symbol_origin(tmp_path):
    run, _ = saved_run(tmp_path, raw=False)
    (run / 'chains/12.r4.json').unlink()
    t = open_run(run).load('12.8/r4')
    assert 'origin unresolved (source incomplete: True)' in render(t, [])
    assert t.recorded['score'] == 0


def test_context_is_bounded_and_last_raw_generation_wins(tmp_path, monkeypatch):
    from aa_evals.tracejudge import scicode_context
    run, _ = saved_run(tmp_path)
    raw = run / 'transcripts/12.r4.gen.jsonl.gz'
    with gzip.open(raw, 'at') as f:
        f.write(json.dumps({'step': '12.8', 'prompt': 'wrong prompt', 'content': 'new answer'}) + '\n')
    assert 'dependency/extraction context unavailable' in render(open_run(run).load('12.8/r4'), [])
    monkeypatch.setattr(scicode_context, 'LINE_LIMIT', 20)
    t = open_run(run).load('12.8/r4')
    assert 'dependency/extraction context unavailable' in render(t, [])
    assert all(len(e.text) <= scicode_context.CONTEXT_CHARS for e in t.events if e.source == 'scicode-context')


def test_extraction_mismatch_is_exposed_not_repaired(tmp_path):
    run, _ = saved_run(tmp_path)
    cp = run / 'chains/12.r4.json'
    c = json.loads(cp.read_text())
    c['steps']['12.8']['code'] = 'def solve():\n    return 42\n'
    cp.write_text(json.dumps(c))
    assert 'matches this extractor: False' in render(open_run(run).load('12.8/r4'), [])


@pytest.mark.parametrize('error', [ImportError('thin snapshot'), OSError('missing templates')])
def test_optional_extractor_unavailable_keeps_other_context(tmp_path, monkeypatch, error):
    import builtins
    run, original = saved_run(tmp_path)
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == 'aa_evals.scicode.prompts':
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', unavailable)
    t = open_run(run).load('12.8/r4')
    text = render(t, [])
    assert 'extractor unavailable in this judge snapshot' in text
    assert 'matches this extractor: True' not in text
    assert 'Exact saved dependency instruction' in text and 'generated step12.5' in text
    assert t.events[0].text == original['text'] and t.turns[-1].text.startswith('```python')

"""Neutral fixtures only; no endpoint, model, or HLE dataset access."""
import asyncio
import copy
import hashlib
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from aa_evals.hle import prompts
from aa_evals.hle.client import InfraError
from aa_evals.hle.run import Runner
from aa_evals.hle.test_hle import args, completion, item, run_once


VALID = json.dumps({'extracted_final_answer': '6', 'reasoning': 'matches',
                    'correct': 'yes', 'confidence': 90})
SETTINGS = {'max_tokens': 32768, 'temperature': 1.0, 'top_k': 20,
            'response_format': prompts.JUDGE_RESPONSE_FORMAT,
            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}


def grade(tmp_path, plan):
    judge = AsyncMock()
    judge.chat.side_effect = plan
    model = AsyncMock()
    settings = copy.deepcopy(SETTINGS)
    runner = Runner(args(judge_format_attempts=len(plan)), [item(3)], model, judge, {}, settings, tmp_path)
    result = asyncio.run(runner.grade(item(3), 'Answer: 6'))
    assert settings == SETTINGS
    model.chat.assert_not_called()
    expected = {'messages': prompts.judge_messages(item(3), 'Answer: 6'), **SETTINGS}
    for invocation in judge.chat.call_args_list:
        assert invocation.args == (expected,)
    for call in result['calls']:
        assert call['request_max_tokens'] == 32768
        assert call['request_response_format'] == SETTINGS['response_format']
        assert call['request_body_sha256'] == hashlib.sha256(json.dumps(
            expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def test_two_offschema_responses_saved_before_parse(tmp_path):
    raw = [completion('not JSON', finish='length', reasoning='first'),
           completion('{"confidence":600000}', finish='length', reasoning='second')]
    raw[0]['id'], raw[1]['id'] = 'resp-1', 'resp-2'
    out = grade(tmp_path, [(raw[0], [{'status': 200}]), (raw[1], [{'status': 200}])])
    assert out['status'] == 'infra_error'
    assert len(out['attempts']) == 4  # two HTTP rows plus two LOCAL parse markers
    assert [c['transport_attempts_range'] for c in out['calls']] == [[0, 1], [2, 3]]
    assert [c['status'] for c in out['calls']] == ['off_schema', 'off_schema']
    assert [c['raw_response'] for c in out['calls']] == raw
    assert [c['response_id'] for c in out['calls']] == ['resp-1', 'resp-2']
    assert [c['reasoning'] for c in out['calls']] == ['first', 'second']
    assert all(c['finish_reason'] == 'length' for c in out['calls'])
    raw[0]['usage']['prompt_tokens'] = 999
    assert out['calls'][0]['usage']['prompt_tokens'] == 10
    assert out['calls'][0]['raw_response']['usage']['prompt_tokens'] == 10


def test_prior_returned_response_survives_infraerror_with_transport_retries(tmp_path):
    raw = completion('bad')
    attempts = [{'status': 503}, {'error': 'timeout'}, {'gate_rejections': 2}]
    out = grade(tmp_path, [(raw, [{'status': 200}]), InfraError('transport exhausted', attempts)])
    assert out['attempts'] == [{'status': 200}, {'format_error': out['calls'][0]['error']}, *attempts]
    assert out['calls'][0]['raw_response'] == raw
    assert out['calls'][1]['transport_attempts_range'] == [2, 5]
    assert out['calls'][1]['status'] == 'infra_error'
    assert out['calls'][1]['raw_response'] is None
    assert out['calls'][1]['usage'] is None  # not zero; may contain multiple transport attempts


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': 9}])
def test_missing_or_partial_usage_remains_exact(tmp_path, usage):
    raw = completion(VALID)
    if usage is None:
        del raw['usage']
    else:
        raw['usage'] = usage
    out = grade(tmp_path, [(raw, [{'status': 200}])])
    assert out['status'] == 'ok'
    assert out['calls'][0]['usage'] == usage
    assert out['calls'][0]['raw_response'] == raw
    assert out['usage'] == (usage or {})  # existing top-level native contract unchanged


def test_unexpected_client_error_preserves_prior_raw_and_unknown_work(tmp_path):
    raw = completion('bad')
    out = grade(tmp_path, [(raw, [{'status': 200}]), RuntimeError('client failed')])
    assert out['status'] == 'infra_error'
    assert out['calls'][0]['raw_response'] == raw
    assert out['calls'][1]['status'] == 'caller_error'
    assert out['calls'][1]['usage'] is None
    assert out['calls'][1]['transport_attempts_range'] == [2, 2]
    assert len(out['attempts']) == 2  # no invented transport call


def test_malformed_returned_shape_preserves_raw_usage(tmp_path):
    raw = {'id': 'bad-shape', 'choices': [], 'usage': {'completion_tokens': 7}}
    out = grade(tmp_path, [(raw, [{'status': 200}])])
    assert out['calls'][0]['status'] == 'caller_error'
    assert out['calls'][0]['raw_response'] == raw
    assert out['calls'][0]['usage'] == {'completion_tokens': 7}


def test_grade_only_persists_private_journal_with_identical_http_body(tmp_path):
    tdir = tmp_path / 'transcripts'
    tdir.mkdir()
    path = tdir / 'q3.r0.json'
    generation = {'status': 'ok', 'content': 'Answer: 6', 'usage': {'completion_tokens': 25}}
    path.write_text(json.dumps({'item_id': 'q3', 'repeat': 0, 'generation': generation, 'judgment': None}))
    observed = []
    def server(request):
        body = json.loads(request.content)
        observed.append(body)
        assert body == {'model': 'judge', 'messages': prompts.judge_messages(item(3), 'Answer: 6'), **SETTINGS}
        return httpx.Response(200, json=completion('bad', finish='length', reasoning='retained reasoning'))
    rows = run_once(server, [item(3)], tmp_path, args(stage='grade', judge_format_attempts=2), jset=SETTINGS)
    rec = json.loads(path.read_text())
    assert rec['generation'] == generation
    assert rows[0]['status'] == 'infra_error' and rows[0]['score'] is None
    assert len(observed) == 2 and all(b['model'] == 'judge' for b in observed)
    assert len(rec['judgment']['calls']) == 2
    assert rec['judgment']['calls'][1]['content'] == 'bad'
    assert rec['judgment']['calls'][1]['reasoning'] == 'retained reasoning'
    assert rec['judgment']['calls'][1]['usage'] == {'prompt_tokens': 10, 'completion_tokens': 20}


def test_success_after_offschema_keeps_both_responses(tmp_path):
    raw = [completion('bad', finish='length'), completion(VALID)]
    out = grade(tmp_path, [(r, [{'status': 200}]) for r in raw])
    assert out['status'] == 'ok' and out['correct'] == 'yes'
    assert [c['status'] for c in out['calls']] == ['off_schema', 'ok']
    assert [c['raw_response'] for c in out['calls']] == raw
    assert out['usage'] == raw[1]['usage']


def test_cancellation_not_converted_to_retryable_infra(tmp_path):
    with pytest.raises(asyncio.CancelledError):
        grade(tmp_path, [asyncio.CancelledError()])

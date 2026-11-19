"""Regression coverage for dropped tool arguments and actual generation accounting."""
import copy

import pytest

from tb_agents import response_budget as B
from tb_agents.response_budget_output import process


def response(message, ids, finish='tool_calls'):
    return {'choices': [{'message': copy.deepcopy(message), 'token_ids': ids, 'finish_reason': finish}],
            'usage': {'completion_tokens': len(ids)}}


def test_dropped_write_content_is_charged_and_never_executed():
    # Reproduces the observed Super reply: 5,941 generated tokens were parsed
    # into a write(path) call with no content, worth only 37 retained body tokens.
    raw = '<tool_call>\n<function=write>\n<parameter=path>artifact.txt</parameter>\n<parameter=content>' + ' word' * 6000
    ids = B.encode(raw)[:5941]
    raw = B.tokenizer().decode(ids, skip_special_tokens=False)
    message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x', 'type': 'function',
        'function': {'name': 'write', 'arguments': {'path': 'artifact.txt'}}}]}
    result = response(message, ids)
    budget = {'token_counting': 'backend-native'}
    tools = [{'type': 'function', 'function': {'name': 'write', 'parameters': {'required': ['path', 'content']}}}]
    out = process(result, raw, 5941, 5941, budget, tools)
    assert out['charged_tokens'] == out['native_tokens'] == 5941
    assert out['truncated_tool_call'] and out['clipped']
    assert not result['choices'][0]['message'].get('tool_calls')
    assert '<parameter=content>' in result['choices'][0]['message']['content']
    assert result['choices'][0]['finish_reason'] == 'length'


def test_complete_tool_at_exact_limit_remains_executable():
    message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'type': 'function',
        'function': {'name': 'bash', 'arguments': {'command': 'true'}}}]}
    ids = B.encode(B.body(message))
    result = response(message, ids)
    out = process(result, B.tokenizer().decode(ids, skip_special_tokens=False), len(ids), len(ids), {},
                  [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'required': ['command']}}}])
    assert not out['clipped'] and not out['truncated_tool_call']
    assert result['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == {'command': 'true'}
    assert result['choices'][0]['finish_reason'] == 'tool_calls'


def test_full_snowball_generation_is_charged_even_if_boundary_crops_output():
    raw = ' more' * 200
    # A different backend tokenizer can produce this text in 100 native tokens.
    ids = [1] * 100
    result = response({'role': 'assistant', 'content': raw}, ids, finish='length')
    out = process(result, raw, 100, len(ids), {})
    assert out['charged_tokens'] == len(B.encode(raw)) > 100
    assert len(out['ids']) <= 100
    assert out['clipped']


def test_backend_native_units_are_explicit_and_not_retokenized_for_budget():
    result = response({'role': 'assistant', 'content': '42'}, [1, 2, 3], finish='stop')
    out = process(result, '42', 3, 10, {'token_counting': 'backend-native'})
    assert out['charged_tokens'] == 3 and out['raw_snowball_tokens'] == 1


def test_special_eos_is_counted_once_in_common_tokenizer():
    raw = '42<|im_end|>'
    ids = [1, 2]
    result = response({'role': 'assistant', 'content': '42'}, ids, finish='stop')
    out = process(result, raw, 100, 100, {})
    assert out['charged_tokens'] == len(B.encode('42' + B.EOT))


@pytest.mark.parametrize('ids,usage', [(None, 5941), ([1, 2], 5941), (['bad'], 1)])
def test_missing_or_inconsistent_raw_generation_fails_closed(ids, usage):
    result = {'choices': [{'message': {'role': 'assistant', 'content': '42'},
                          'token_ids': ids, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': usage}}
    with pytest.raises(ValueError):
        process(result, '42', 100, 100, {})


def test_backend_cannot_silently_exceed_requested_native_limit():
    result = response({'role': 'assistant', 'content': '42'}, [1, 2, 3], finish='stop')
    with pytest.raises(ValueError, match='requested native limit'):
        process(result, '42', 2, 2, {'token_counting': 'backend-native'})


@pytest.mark.parametrize('asynchronous', [False, True])
def test_single_turn_native_units_use_backend_generation_not_snowball_retained_length(asynchronous):
    import asyncio
    from tb_agents.response_budget_single import run, run_async
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 50,
              'total_response_tokens': 150, 'token_counting': 'backend-native'}
    requests = []

    def reply(payload):
        requests.append(copy.deepcopy(payload))
        first = len(requests) == 1
        return {'content': '' if first else '42', 'reasoning': 'Think carefully. ' * 50 if first else '',
                'finish_reason': 'length' if first else 'stop', 'usage': {'completion_tokens': 100 if first else 3}}

    async def async_call(payload):
        return reply(payload)

    def sync_call(payload):
        out = reply(payload)
        return {'choices': [{'message': {'role': 'assistant', 'reasoning_content': out['reasoning'],
                                         'content': out['content']}, 'finish_reason': out['finish_reason']}],
                'usage': out['usage']}

    payload = {'messages': [{'role': 'user', 'content': 'Answer.'}], 'max_tokens': 1000}
    out = asyncio.run(run_async(async_call, payload, budget)) if asynchronous else run(sync_call, payload, budget)
    metadata = out['response_budget'] if asynchronous else out['_response_budget']
    assert metadata['used'] == 103
    assert [p['max_tokens'] for p in requests] == [100, 45]
    assert len(metadata['calls'][0]['real_body_ids']) > 100

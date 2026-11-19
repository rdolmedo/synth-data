"""CPU-only returned-response fixtures; no endpoint construction or requests."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from aa_evals.hle.client import InfraError
from aa_evals.tracejudge.core import build_row
from aa_evals.tracejudge.judge import LLMJudge
from aa_evals.tracejudge.trace import Trace


def response(content='{"flags": [], "reason": "ok"}', *, usage=True, finish='stop', ident='response-1'):
    data = {'id': ident, 'choices': [{'finish_reason': finish,
                                    'message': {'content': content, 'reasoning_content': 'private reasoning'}}]}
    if usage:
        data['usage'] = {'prompt_tokens': 11, 'completion_tokens': 7}
    return data


def judge_for(client, save_raw=False):
    judge = object.__new__(LLMJudge)
    judge.client, judge.system = client, 'unchanged-system'
    judge.spec, judge.effort, judge.thinking = 'fake', None, None
    judge.max_tokens, judge.structured, judge.save_raw = 8192, True, save_raw
    return judge


class Client:
    endpoint = SimpleNamespace(request={}, served_name='fake', api_key='NEVER-LOG-CREDENTIAL')

    def __init__(self, replies):
        self.replies, self.bodies = iter(replies), []

    def _current(self):
        return self.endpoint

    async def chat(self, body):
        self.bodies.append(copy.deepcopy(body))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result, [{'url': 'not-to-be-logged', 'status': 200}]


@pytest.mark.parametrize('save_raw', [False, True])
def test_success_raw_is_opt_in_and_request_identical(save_raw):
    data = response()
    client = Client([data])
    judge = judge_for(client, save_raw)
    expected_body = judge_for(Client([]), False)._body('digest')
    result = asyncio.run(judge.judge('digest'))
    assert client.bodies == [expected_body]
    call = result['calls'][0]
    assert result['ok'] and result['prompt_tokens'] == 11 and result['completion_tokens'] == 7
    assert call['finish_reason'] == 'stop' and call['response_id'] == 'response-1'
    assert call['usage'] == data['usage'] and call['max_tokens'] == 8192
    assert ('raw_response' in call) is save_raw
    assert ('content' in call) is save_raw and ('reasoning' in call) is save_raw
    if save_raw:
        assert call['raw_response'] == data
        assert call['reasoning'] == 'private reasoning'
    assert 'NEVER-LOG-CREDENTIAL' not in json.dumps(result)
    assert 'not-to-be-logged' not in json.dumps(result)
    trace = Trace('fake', 'key', 'item', 0, 'single', {'score': 0})
    row = build_row(trace, [], 'clean', result, judge.describe(), 6)
    assert row['llm']['calls'] == result['calls']


def test_unparseable_reask_keeps_both_response_bodies_and_caps():
    data = [response('no JSON', finish='length', ident='first'), response(None, finish='length', ident='second')]
    client = Client(data)
    result = asyncio.run(judge_for(client, True).judge('digest'))
    assert result['ok'] is False and len(result['calls']) == 2
    assert [c['response_id'] for c in result['calls']] == ['first', 'second']
    assert [c['finish_reason'] for c in result['calls']] == ['length', 'length']
    assert [c['raw_response'] for c in result['calls']] == data
    assert result['completion_tokens'] == 14
    assert len(client.bodies) == 2 and client.bodies[0] == client.bodies[1]


@pytest.mark.parametrize('first', [True, False])
def test_endpoint_error_preserves_prior_calls_and_unknown_usage(first):
    replies = [InfraError('unavailable')]
    if not first:
        replies.insert(0, response('no JSON', finish='length'))
    result = asyncio.run(judge_for(Client(replies), True).judge('digest'))
    assert result['ok'] is False and len(result['calls']) == len(replies)
    assert result['calls'][-1]['error_kind'] == 'InfraError'
    assert result['calls'][-1]['usage'] is None
    assert 'raw_response' not in result['calls'][-1]
    assert result['prompt_tokens'] is None and result['completion_tokens'] is None
    if not first:
        assert result['calls'][0]['completion_tokens'] == 7
        assert result['calls'][0]['raw_response']['choices'][0]['finish_reason'] == 'length'


def test_missing_usage_is_unknown_not_zero_or_partial_aggregate():
    client = Client([response('bad', usage=False), response()])
    result = asyncio.run(judge_for(client).judge('digest'))
    assert result['ok'] is True
    assert result['calls'][0]['usage'] is None
    assert result['calls'][0]['completion_tokens'] is None
    assert result['calls'][1]['completion_tokens'] == 7
    assert result['completion_tokens'] is None and result['prompt_tokens'] is None


def test_endpoint_resolution_error_before_body_remains_a_saved_error():
    class Unavailable(Client):
        def _current(self):
            raise InfraError('registry unavailable')

    client = Unavailable([])
    result = asyncio.run(judge_for(client, True).judge('digest'))
    assert result['ok'] is False and client.bodies == []
    assert result['calls'][0]['request_body_created'] is False
    assert result['calls'][0]['usage'] is None
    assert 'raw_response' not in result['calls'][0]


def test_shared_judge_concurrent_keys_keep_independent_call_history():
    class Concurrent(Client):
        async def chat(self, body):
            key = body['messages'][-1]['content']
            await asyncio.sleep(0)
            return response(ident=key), []

    judge = judge_for(Concurrent([]), True)

    async def run():
        return await asyncio.gather(*(judge.judge(key) for key in ('one', 'two')))

    one, two = asyncio.run(run())
    assert one['calls'] is not two['calls']
    assert [c['response_id'] for c in one['calls']] == ['one']
    assert [c['response_id'] for c in two['calls']] == ['two']


@pytest.mark.parametrize('command', ['judge', 'watch'])
def test_cli_raw_flag_is_opt_in(monkeypatch, command):
    from aa_evals.tracejudge import __main__ as cli, core, judge as module
    seen = []

    class FakeJudge:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)

        async def close(self):
            pass

    async def fake(*args):
        return []

    monkeypatch.setattr(module, 'LLMJudge', FakeJudge)
    monkeypatch.setattr(core, 'judge_run' if command == 'judge' else 'watch_run', fake)
    cli.main([command, '--run-dir', '/unused'])
    cli.main([command, '--run-dir', '/unused', '--save-raw'])
    assert [a['save_raw'] for a in seen] == [False, True]

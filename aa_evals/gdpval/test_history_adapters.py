"""Real Stirrup serialization and agent-loop tests against in-memory HTTP; no model/SIF execution."""
import asyncio
import copy
import json

import httpx
import pytest
from openai import AsyncOpenAI

from aa_evals.gdpval import client as C


def test_profile_adapters_are_frozen_and_unknown_rejected(tmp_path, monkeypatch):
    from aa_evals.gdpval import run as R, endpoint as E
    from aa_evals.gdpval.tasks import Task
    profile = {'adapters': ['inline_reasoning_history'], 'overrides': {'temperature': 1}}
    monkeypatch.setattr(E, 'resolve', lambda spec: {'served_name': 'synthetic', 'request': profile})
    monkeypatch.setattr(R, 'endpoint_ctx', lambda model: 131072)
    monkeypatch.setattr(R.protocol, 'image_inputs', lambda model: (False, 'synthetic'))
    monkeypatch.setattr(R.T, 'parquet_sha', lambda: 'synthetic')
    monkeypatch.setattr(R, 'select', lambda args: [Task('t', 'test', 'test', 'Synthetic.', [], [])])
    monkeypatch.setattr(R, 'RUNS', tmp_path)
    R.main(['--stage', 'init', '--run-id', 'new', '--model', 'url:http://fake/v1@m'])
    cfg = json.loads((tmp_path / 'new' / 'config.json').read_text())
    assert cfg['model']['adapters'] == ['inline_reasoning_history']
    profile['adapters'].append('unknown')
    assert cfg['model']['adapters'] == ['inline_reasoning_history']
    with pytest.raises(ValueError, match='unknown'):
        R.main(['--stage', 'init', '--run-id', 'bad', '--model', 'url:http://fake/v1@m'])
    assert C.history_adapters({}) == ['mirror_reasoning_content']


def test_served_copy_preserves_multimodal_content_and_source():
    messages = [{'role': 'assistant', 'reasoning_content': 'THINK',
                 'content': [{'type': 'text', 'text': 'ANSWER'}, {'type': 'image_url', 'image_url': {'url': 'data:x'}}],
                 'tool_calls': [{'id': 'same-tool-id'}]}]
    before = copy.deepcopy(messages)
    out = C.adapt_history(messages, ['inline_reasoning_history'])
    assert messages == before
    assert out[0]['content'][0]['text'] == '<think>\nTHINK</think>'
    assert out[0]['content'][1:] == before[0]['content']
    assert out[0]['tool_calls'] == before[0]['tool_calls']
    assert 'reasoning_content' not in out[0]


@pytest.mark.parametrize('model,adapters,body', [
    ('deepseek-v41-flash-1017', ['mirror_reasoning_content'],
     {'reasoning_effort': 'max', 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'max'}}),
    ('qwen38-27b', ['mirror_reasoning_content'],
     {'reasoning_effort': 'medium', 'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}),
    ('nemotron-3.5-super-vl', ['inline_reasoning_history'],
     {'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}}),
])
def test_actual_stirrup_agent_replays_adapted_reasoning(tmp_path, monkeypatch, model, adapters, body):
    from aa_evals.gdpval import agent as A
    from aa_evals.gdpval.network import POLICY
    from aa_evals.gdpval.tasks import Task
    calls = {'count': [], 'chat': []}
    made = []

    def handler(request):
        payload = json.loads(request.content)
        if request.url.path.endswith('/tokenize'):
            calls['count'].append(payload)
            return httpx.Response(200, json={'count': 100})
        calls['chat'].append(payload)
        first = len(calls['chat']) == 1
        tool = 'finish' if first else 'abandon_task_finish'
        arguments = ({'reason': 'synthetic invalid path', 'paths': ['/home/user/does-not-exist']}
                     if first else {'reason': 'synthetic complete'})
        message = {'role': 'assistant', 'content': '', 'reasoning': 'THINK_FIRST' if first else 'THINK_SECOND',
                   'tool_calls': [{'id': f'call-{len(calls["chat"])}', 'type': 'function',
                                   'function': {'name': tool, 'arguments': json.dumps(arguments)}}]}
        return httpx.Response(200, json={'id': 'synthetic', 'object': 'chat.completion', 'created': 0, 'model': model,
                                        'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls'}],
                                        'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}})

    def factory(**kwargs):
        assert kwargs['adapters'] == adapters  # agent must pass the frozen profile, not re-resolve it
        client = C.CappedVLLMClient(**kwargs)
        transport = httpx.MockTransport(handler)
        old_http = client._http
        client._http = httpx.AsyncClient(transport=transport)
        client._clients['http://fake/v1'] = AsyncOpenAI(base_url='http://fake/v1', api_key='synthetic', max_retries=0,
                                                       http_client=httpx.AsyncClient(transport=transport))
        made.append(old_http)
        return client

    monkeypatch.setattr(A, 'CappedVLLMClient', factory)
    cfg = {'agent_network': dict(POLICY),
           'model': {'spec': f'url:http://fake/v1@{model}', 'served_name': model, 'body': body, 'adapters': adapters,
                     'max_output': 8192, 'vision': False},
           'context': {'window': 131072}, 'sandbox': {'sif_local': '/unused-no-execution'}, 'limits': {'wall_sec': 30}}

    async def run():
        try:
            return await A.run_attempt(Task('synthetic', 'test', 'test', 'Synthetic adapter smoke.', [], []), 0,
                                       cfg=cfg, work=tmp_path / 'work', out_dir=tmp_path / 'out')
        finally:
            for old_http in made:
                await old_http.aclose()
    result = asyncio.run(run())
    assert (result['status'], result['end']) == ('ok', 'abandon'), result.get('error')
    assert len(calls['chat']) == len(calls['count']) == 2
    for counted, chat in zip(calls['count'], calls['chat']):
        assert counted['messages'] == chat['messages']
        assert counted['tools'] == chat['tools']
        assert counted['chat_template_kwargs'] == chat['chat_template_kwargs'] == body['chat_template_kwargs']
    assistant = next(m for m in calls['chat'][1]['messages'] if m['role'] == 'assistant')
    if adapters == ['inline_reasoning_history']:
        assert 'reasoning' not in assistant and 'reasoning_content' not in assistant
        content = assistant['content']
        text = content if isinstance(content, str) else ''.join(p.get('text', '') for p in content)
        assert text.count('THINK_FIRST') == 1 and '<think>' in text
    else:
        assert assistant['reasoning'] == assistant['reasoning_content'] == 'THINK_FIRST'
    # Real Stirrup's stored history still contains its original ReasoningBlock, without inline duplication.
    import gzip
    transcript = json.loads(gzip.open(tmp_path / 'out' / 'transcript.json.gz').read())
    assert 'THINK_FIRST' in json.dumps(transcript['history'])
    assert '<think>\\nTHINK_FIRST' not in json.dumps(transcript['history'])

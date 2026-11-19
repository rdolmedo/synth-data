"""Guard regression: render complete tool histories without schema iterators."""
import copy
import json

import pytest

from tb_agents.deepseek_v41_context import tokenize_payload
from tb_agents.test_pi_context_limit import backend, guard, post


def request():
    return {'model': 'deepseek-v41-flash', 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'inspect'},
                                   {'type': 'text', 'text': 'files'}]},
        {'role': 'assistant', 'reasoning': 'retained reasoning', 'content': None,
         'tool_calls': [{'id': 'call1', 'type': 'function', 'function':
                        {'name': 'bash', 'arguments': '{"command":"ls"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call1', 'content': 'file.txt'},
        {'role': 'user', 'content': 'Thinking exceeded the maximum thinking budget, please continue.'}],
        'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'string'}}}}}],
        'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'max', 'drop_thinking': False},
        'max_completion_tokens': 16384, 'stream': False}


def test_render_retains_tool_schema_reasoning_and_continuation():
    payload = request()
    original = copy.deepcopy(payload)
    rendered = tokenize_payload(payload)
    assert payload == original
    assert rendered['add_special_tokens'] is False
    for text in ('inspect\n\nfiles', 'retained reasoning', '<tool_result>file.txt</tool_result>',
                 'Thinking exceeded the maximum thinking budget, please continue.', '"command"'):
        assert text in rendered['prompt']
    payload['reasoning_effort'] = 'low'
    assert 'Reasoning Effort: 50' in tokenize_payload(payload)['prompt']


def test_unverified_multimodal_expansion_fails_closed():
    payload = request()
    payload['messages'][0]['content'].append({'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,example'}})
    with pytest.raises(ValueError, match='cannot count'):
        tokenize_payload(payload)


@pytest.mark.parametrize('stream', [False, True])
def test_guard_tokenizes_rendered_prompt_and_keeps_generation_request(tmp_path, monkeypatch, stream):
    monkeypatch.setenv('PI_CONTEXT_RENDERER', 'deepseek_v41')
    payload = request()
    payload['stream'] = stream
    with backend(131900) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        post(proxy, payload)
    assert tokenizations == [tokenize_payload(payload)]
    assert requests[0]['messages'] == payload['messages']
    assert requests[0]['tools'] == payload['tools']
    assert requests[0]['chat_template_kwargs'] == payload['chat_template_kwargs']
    assert requests[0]['max_completion_tokens'] == 100


def test_guard_still_stops_tool_overflow_with_zero_loss(tmp_path, monkeypatch):
    monkeypatch.setenv('PI_CONTEXT_RENDERER', 'deepseek_v41')
    with backend(132001) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        post(proxy, request())
    assert not requests
    assert tokenizations[0]['add_special_tokens'] is False
    assert json.loads((tmp_path/'context-limit.jsonl').read_text())['sft_loss_weight'] == 0


def test_renderer_serves_the_v41_flash_family_only():
    """deepseek-v41-flash-1017 shares the V4.1 encoder and tokenizer; other models are refused."""
    payload = request()
    base = tokenize_payload(payload)['prompt']
    payload['model'] = 'deepseek-v41-flash-1017'
    assert tokenize_payload(payload)['prompt'] == base
    payload['model'] = 'deepseek-v4-flash-0731'
    with pytest.raises(ValueError):
        tokenize_payload(payload)


def test_guard_fails_closed_without_retry_on_a_render_error(tmp_path, monkeypatch):
    """A renderer that cannot handle the request is a 500 (the proxy retries 502/503 for its hold window)."""
    import urllib.error
    monkeypatch.setenv('PI_CONTEXT_RENDERER', 'deepseek_v41')
    payload = {**request(), 'model': 'deepseek-v4-flash-0731'}
    with backend(1000) as (url, requests, tokenizations), guard(url, tmp_path, monkeypatch) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy, payload)
    assert error.value.code == 500
    assert not requests and not tokenizations, 'nothing reaches the backend'

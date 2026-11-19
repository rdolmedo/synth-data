"""Full image payloads must be counted before budgeting, even in append-only history."""
import base64
import contextlib
import json
import urllib.error

import pytest

from tb_agents.pi_context_limit import CAP_MARK
from tb_agents.test_pi_context_limit import backend, guard, post
from tb_agents.test_pi_envclock import proxy
from tb_agents.test_pi_model_input import PNG


@contextlib.contextmanager
def frontend(kind, url, tmp_path, monkeypatch):
    monkeypatch.setenv('PI_GUARD_BOUND_SHORTCUT', '1')
    monkeypatch.delenv('PI_CONTEXT_RENDERER', raising=False)
    monkeypatch.delenv('PI_CONTEXT_TOKENIZER', raising=False)
    if kind == 'guard':
        with guard(url, tmp_path, monkeypatch, window=131072) as value:
            yield value
    else:
        with proxy(url.removesuffix('/v1'), '--context-window', '131072') as value:
            yield value


def media_messages(kind='image_url'):
    block = ({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}}
             if kind == 'image_url' else {'type': 'input_audio', 'input_audio': {'data': 'AAAA', 'format': 'wav'}})
    initial = [{'role': 'user', 'content': 'task'}]
    appended = initial + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': [block] * 100}]
    return initial, appended


@pytest.mark.parametrize('kind', ['guard', 'proxy'])
@pytest.mark.parametrize('media', ['image_url', 'input_audio'])
def test_append_only_media_uses_expanded_count_and_full_payload(kind, media, tmp_path, monkeypatch):
    initial, messages = media_messages(media)
    expanded = 120605
    with backend(lambda p: 95000 if len(p['messages']) == 1 else expanded,
                 reported_prompt_tokens=95000) as (url, requests, tokenizations):
        with frontend(kind, url, tmp_path, monkeypatch) as endpoint:
            post(endpoint, {'model': 'test', 'messages': initial, 'max_tokens': 16384, 'stream': True})
            post(endpoint, {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True})
            # Even a later text append retains media in history and must count it.
            messages = messages + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'continue'}]
            post(endpoint, {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True})
    assert len(tokenizations) == len(requests) == 3
    assert tokenizations[1]['messages'] == requests[1]['messages'] == messages[:-2]
    assert tokenizations[2]['messages'] == requests[2]['messages'] == messages
    assert requests[1]['max_tokens'] == requests[2]['max_tokens'] == 10467
    assert expanded + requests[1]['max_tokens'] == 131072
    assert not (tmp_path / 'context-limit.jsonl').exists()


@pytest.mark.parametrize('kind', ['guard', 'proxy'])
@pytest.mark.parametrize('content', ['text followup', [{'type': 'text', 'text': 'text followup'}]])
def test_text_only_append_keeps_existing_shortcut(kind, content, tmp_path, monkeypatch):
    initial = [{'role': 'user', 'content': 'task'}]
    with backend(1000) as (url, requests, tokenizations), frontend(kind, url, tmp_path, monkeypatch) as endpoint:
        post(endpoint, {'model': 'test', 'messages': initial, 'max_tokens': 16384, 'stream': True})
        post(endpoint, {'model': 'test', 'messages': initial + [{'role': 'assistant', 'content': 'ok'},
             {'role': 'user', 'content': content}], 'max_tokens': 16384, 'stream': True})
    assert len(tokenizations) == 1 and len(requests) == 2
    assert requests[1]['max_tokens'] == 16384


@pytest.mark.parametrize('kind', ['guard', 'proxy'])
def test_expanded_media_at_window_never_reaches_generation(kind, tmp_path, monkeypatch):
    initial, messages = media_messages()
    with backend(lambda p: 95000 if len(p['messages']) == 1 else 131072,
                 reported_prompt_tokens=95000) as (url, requests, tokenizations):
        with frontend(kind, url, tmp_path, monkeypatch) as endpoint:
            post(endpoint, {'model': 'test', 'messages': initial, 'max_tokens': 16384, 'stream': True})
            payload = {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True}
            if kind == 'guard':
                assert CAP_MARK.encode() in post(endpoint, payload)
            else:
                with pytest.raises(urllib.error.HTTPError) as error:
                    post(endpoint, payload)
                assert error.value.code == 400
    assert len(tokenizations) == 2 and len(requests) == 1
    if kind == 'guard':
        event = json.loads((tmp_path / 'context-limit.jsonl').read_text())
        assert event['prompt_tokens'] == 131072 and event['context_limit_source'] == 'rendered_prompt'


@pytest.mark.parametrize('kind', ['guard', 'proxy'])
def test_media_tokenize_failure_stays_verification_error_without_generation(kind, tmp_path, monkeypatch):
    initial, messages = media_messages()
    status = lambda p: 200 if len(p['messages']) == 1 else 500
    with backend(95000, reported_prompt_tokens=95000, tokenize_status=status) as (url, requests, tokenizations):
        with frontend(kind, url, tmp_path, monkeypatch) as endpoint:
            post(endpoint, {'model': 'test', 'messages': initial, 'max_tokens': 16384, 'stream': True})
            with pytest.raises(urllib.error.HTTPError) as error:
                post(endpoint, {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True})
            assert error.value.code == 503
    assert len(tokenizations) == 2 and len(requests) == 1
    assert not (tmp_path / 'context-limit.jsonl').exists()


@pytest.mark.parametrize('kind', ['guard', 'proxy'])
def test_real_backend_failure_is_not_changed_into_context_limit(kind, tmp_path, monkeypatch):
    _, messages = media_messages()
    with backend(120605, chat_status=500) as (url, requests, tokenizations):
        with frontend(kind, url, tmp_path, monkeypatch) as endpoint:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(endpoint, {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True})
            assert error.value.code == 500
    assert len(tokenizations) == len(requests) == 1
    assert requests[0]['max_tokens'] == 10467
    assert not (tmp_path / 'context-limit.jsonl').exists()


@pytest.mark.parametrize('count', [-1, True, 1.5])
def test_compact_proxy_rejects_invalid_token_count_without_generation(count, tmp_path, monkeypatch):
    _, messages = media_messages()
    with backend(count) as (url, requests, tokenizations):
        with frontend('proxy', url, tmp_path, monkeypatch) as endpoint:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(endpoint, {'model': 'test', 'messages': messages, 'max_tokens': 16384, 'stream': True})
            assert error.value.code == 503
    assert len(tokenizations) == 1 and not requests

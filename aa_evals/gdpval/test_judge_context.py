"""Judge input budgets on synthetic renders and fake endpoints; no real inference."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from aa_evals.gdpval import judge as J



def test_exact_token_admission_reserves_output_and_preserves_task_rubric(tmp_path, monkeypatch):
    task = SimpleNamespace(occupation='tester', sector='synthetic', prompt='Required task is preserved verbatim.',
                           reference_files=[])
    files = [{'name': 'answer.txt', 'kind': 'text', 'size': 10000, 'text': 'long file text ' * 1000,
              'images': ['a.png', 'b.png'], 'notes': []}]
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': files}))
    (tmp_path / 'img').mkdir()
    for name in ('a.png', 'b.png'):
        (tmp_path / 'img' / name).write_bytes(b'fake image')
    messages = J.build_messages(task, None, tmp_path, tmp_path, J.Budget(), True, {})
    original = json.loads(json.dumps(messages))
    calls = []
    def tokens(msgs):
        # Artificial tokenizer: text chars are tokens; each image costs4000 tokens.
        return sum(len(p['text']) if p['type'] == 'text' else 4000 for m in msgs for p in m['content'])
    def handler(req):
        if req.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'max_model_len': 12000}]})
        body = json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            count = tokens(body['messages'])
            if count > 12000:
                return httpx.Response(400, json={'error': 'prompt exceeds context limit'})
            return httpx.Response(200, json={'count': count})
        assert tokens(body['messages']) + body['max_tokens'] + 256 <= 12000
        calls.append(body)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"winner":"A"}'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr(J, 'resolve', lambda spec: {'served_name': 'm', 'urls': ['http://fake/v1']})
    async def exercise():
        judge = J.Judge('url:http://fake/v1@m', {'max_tokens': 2000}, max_tries=1)
        await judge.http.aclose()
        judge.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stats = {}
        result = await judge.ask(messages, stats=stats)
        await judge.aclose()
        return result, stats
    result, stats = asyncio.run(exercise())
    assert result['winner'] == 'A' and len(calls) == 1
    sent = calls[0]['messages'][0]['content']
    assert sent[0] == messages[0]['content'][0] and sent[-1] == messages[0]['content'][-1]
    assert messages == original
    assert stats['context_budget']['endpoint_context_tokens'] == 12000
    assert stats['context_budget']['output_reserve'] == 2000
    assert stats['context_budget']['excerpt_fraction'] < 1
    assert stats['context_budget']['prompt_tokens'] <= stats['context_budget']['input_limit']


@pytest.mark.parametrize('context,count', [(32768, 100), (262144, 180000), (262144, 252000)])
def test_prompt_that_fits_is_unchanged_and_context_limit_is_not_assumed(monkeypatch, context, count):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'task and rubric'}]}]
    sent = []
    def handler(req):
        if req.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'max_model_len': context}]})
        body = json.loads(req.content)
        if req.url.path.endswith('/tokenize'):
            return httpx.Response(200, json={'count': count})
        sent.append(body)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"winner":"tie"}'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr(J, 'resolve', lambda spec: {'served_name': 'm', 'urls': ['http://fake/v1']})
    async def exercise():
        judge = J.Judge('url:http://fake/v1@m', {'max_tokens': 8192}, max_tries=1)
        await judge.http.aclose()
        judge.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stats = {}
        result = await judge.ask(messages, stats=stats)
        await judge.aclose()
        return result, stats
    result, stats = asyncio.run(exercise())
    assert result['winner'] == 'tie' and sent[0]['messages'] == messages
    assert stats['context_budget']['context_tokens'] == context
    assert stats['context_budget']['excerpt_fraction'] == 1


def test_retry_errors_skips_successes_and_preserves_metadata_history(tmp_path, monkeypatch):
    from aa_evals.gdpval import run as R
    root = tmp_path / 'run'
    key = R.key('t', 0)
    mine = root / 'attempts' / key
    (mine / 'render').mkdir(parents=True)
    (mine / 'render' / 'manifest.json').write_text(json.dumps({'files': []}))
    (mine / 'outcome.json').write_text(json.dumps({'status': 'ok', 'end': 'finish'}))
    prep = tmp_path / 'prep' / 't' / 'expert'
    prep.mkdir(parents=True)
    (prep / 'manifest.json').write_text(json.dumps({'files': []}))
    (root / 'matches').mkdir()
    arm = 'qwen38-27b-medium-pin'
    records = [{'key': key, 'order': 'MA', 'winner': 'A', 'model_score': 1},
               {'key': key, 'order': 'MB', 'error': 'context overflow'}]
    out = root / 'matches' / f'{arm}.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in records))
    (root / 'matches' / f'{arm}.meta.json').write_text(json.dumps({'prompt_version': 'v1'}))
    (root / 'config.json').write_text(json.dumps({'selection': {'task_ids': ['t']}, 'repeats': 1}))
    monkeypatch.setattr(R, 'RUNS', tmp_path)
    monkeypatch.setattr(R, 'PREP', tmp_path / 'prep')
    task = SimpleNamespace(task_id='t', occupation='test', sector='synthetic', prompt='task', reference_files=[])
    monkeypatch.setattr(R.T, 'load_tasks', lambda: [task])
    monkeypatch.setattr(R, 'judge_spec', lambda args: (SimpleNamespace(extra={'max_tokens': 100}), 'fake'))
    calls = []
    class FakeJudge:
        def __init__(self, *a, **kw):
            pass
        async def ask(self, messages, **kw):
            calls.append(messages)
            return {'winner': 'B', 'error': None, 'tries': 1}
        async def aclose(self):
            pass
    monkeypatch.setattr(J, 'Judge', FakeJudge)
    args = SimpleNamespace(run_id='run', judge_arm=arm, judge_images=1, concurrency=1, follow=False,
                           retry_judge_errors=True)
    asyncio.run(R.stage_judge(args))
    assert len(calls) == 1
    new = R.read_jsonl(out)
    assert new[:2] == records and new[-1]['order'] == 'MB' and new[-1]['winner'] == 'B'
    meta = json.loads((root / 'matches' / f'{arm}.meta.json').read_text())
    assert meta['history'] == [{'prompt_version': 'v1'}]


def test_mandatory_task_overflow_is_error_without_chat_or_rubric_truncation(monkeypatch):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'mandatory task/rubric ' * 50}]}]
    chats = []
    def handler(req):
        if req.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'max_model_len': 400}]})
        if req.url.path.endswith('/tokenize'):
            body = json.loads(req.content)
            return httpx.Response(200, json={'count': sum(len(p['text']) for p in body['messages'][0]['content'])})
        chats.append(req)
        raise AssertionError('oversized mandatory prompt reached chat')
    monkeypatch.setattr(J, 'resolve', lambda spec: {'served_name': 'm', 'urls': ['http://fake/v1']})
    async def exercise():
        judge = J.Judge('fake', {'max_tokens': 100}, max_tries=2)
        await judge.http.aclose()
        judge.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await judge.ask(messages)
        await judge.aclose()
        return result
    result = asyncio.run(exercise())
    assert result['winner'] is None and 'task/rubric alone' in result['error'] and result['tries'] == 1
    assert not chats and messages[0]['content'][0]['text'] == 'mandatory task/rubric ' * 50


@pytest.mark.parametrize('many_files', [False, True])
def test_252k_input_and_per_file_floor_overflow_respect_output_reserve(tmp_path, monkeypatch, many_files):
    task = SimpleNamespace(occupation='tester', sector='synthetic', prompt='Mandatory task.', reference_files=[])
    files = [{'name': f'file{i}.txt', 'kind': 'text', 'size': 5000, 'text': 'x' * 5000,
              'images': [], 'notes': []} for i in range(150 if many_files else 1)]
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': files}))
    messages = J.build_messages(task, None, tmp_path, None, J.Budget(), False, {})
    size = lambda msgs: sum(len(p['text']) for m in msgs for p in m['content'] if p['type'] == 'text')
    multiplier = 1 if many_files else 252000 / size(messages)
    assert size(messages) > 262144 if many_files else round(size(messages) * multiplier) == 252000
    sent = []
    def handler(req):
        if req.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'max_model_len': 262144}]})
        body = json.loads(req.content)
        count = round(size(body['messages']) * multiplier)
        if req.url.path.endswith('/tokenize'):
            if count > 262144:
                return httpx.Response(400, json={'error': 'context limit exceeded'})
            return httpx.Response(200, json={'count': count})
        assert count + body['max_tokens'] + 256 <= 262144
        sent.append(body)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"winner":"A"}'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr(J, 'resolve', lambda spec: {'served_name': 'm', 'urls': ['http://fake/v1']})
    async def exercise():
        judge = J.Judge('fake', {'max_tokens': 32768}, max_tries=1)
        await judge.http.aclose()
        judge.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stats = {}
        result = await judge.ask(messages, stats=stats)
        await judge.aclose()
        return result, stats
    result, stats = asyncio.run(exercise())
    assert result['winner'] == 'A' and len(sent) == 1
    assert stats['context_budget']['excerpt_fraction'] < 1
    assert sent[0]['messages'][0]['content'][0] == messages[0]['content'][0]

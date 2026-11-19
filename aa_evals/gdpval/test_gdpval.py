"""Offline tests: python -m aa_evals.gdpval.test_gdpval (the gdpval venv; no endpoint, no SIF)."""
import json
import math
import tempfile
from pathlib import Path

from aa_evals.gdpval import elo
from aa_evals.gdpval import judge as J
from aa_evals.gdpval import tasks as T


def test_elo_single_opponent():
    # 92% wins vs the expert -> 968 + 400 log10(0.92/0.08)
    obs = [('m', elo.EXPERT, 1.0, 0.5)] * 92 + [('m', elo.EXPERT, 0.0, 0.5)] * 8
    r = elo.fit(obs)
    assert abs(r[elo.EXPERT] - 968) < 1e-6
    assert abs(r['m'] - elo.elo_from_winrate(0.92)) < 1.0, r
    # ties are half-wins
    r2 = elo.fit([('m', elo.EXPERT, 0.5, 1.0)] * 10)
    assert abs(r2['m'] - 968) < 1.0
    # AA's Qwen3.8 medium 1389 <-> 91.9% vs the expert
    assert abs(elo.winrate_from_elo(1389) - 0.9186) < 1e-3


def test_elo_pool():
    import random
    random.seed(0)
    true = {'a': 1300.0, 'b': 900.0, elo.EXPERT: 968.0}
    obs = []
    for _ in range(4000):
        x, y = random.sample(sorted(true), 2)
        p = elo.winrate_from_elo(true[x], true[y])
        obs.append((x, y, 1.0 if random.random() < p else 0.0, 1.0))
    r = elo.fit(obs)
    assert abs(r['a'] - 1300) < 40 and abs(r['b'] - 900) < 40, r


def test_parse():
    assert J.parse_winner('blah {"winner": "A"}') == 'A'
    assert J.parse_winner('{"winner":"b"} then {"winner": "tie"}') == 'tie'
    assert J.parse_winner('no verdict') is None
    assert J.score_for('MA', 'A') == 1.0 and J.score_for('MB', 'A') == 0.0 and J.score_for('MB', 'tie') == 0.5


def test_prompt_and_messages():
    tasks = T.load_tasks()
    assert len(tasks) == 220
    with_refs = next(t for t in tasks if t.reference_files)
    no_refs = next(t for t in tasks if not t.reference_files)
    p = T.task_prompt(with_refs)
    assert '## Reference Files Location' in p and '[absolute path' not in p and '(This section appears' not in p
    assert all(f'- {dest}' in p for _, dest in with_refs.ref_targets())
    assert with_refs.prompt in p and p.endswith('Please begin working on the task now.')
    q = T.task_prompt(no_refs)
    assert '## Reference Files Location' not in q and '## Completing Your Work' in q
    assert T.system_prompt().startswith('You are an AI agent completing a standalone professional task.')
    # a judge prompt from a fake render: text cut and image caps are applied and counted
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'img').mkdir()
        files = []
        for i in range(3):
            imgs = []
            for k in range(5):
                (d / 'img' / f'f{i}_{k}.png').write_bytes(b'\x89PNG fake')
                imgs.append(f'f{i}_{k}.png')
            files.append({'name': f'doc{i}.docx', 'kind': 'document', 'size': 10, 'n_pages': 5, 'text': 'x' * 50000,
                          'images': imgs, 'notes': []})
        (d / 'manifest.json').write_text(json.dumps({'files': files}))
        stats = {}
        msgs = J.build_messages(no_refs, None, d, None, J.Budget(), True, stats)
        parts = msgs[0]['content']
        n_img = sum(p['type'] == 'image_url' for p in parts)
        assert n_img == J.Budget().side_images, n_img
        assert stats['text_truncated'] == 3 and stats['images_dropped'] == 15 - n_img
        text = ''.join(p.get('text', '') for p in parts)
        assert 'Submission B submitted no files' in text and 'this task has no reference files' in text
        msgs0 = J.build_messages(no_refs, None, d, None, J.Budget(), False, {})
        assert not any(p['type'] == 'image_url' for p in msgs0[0]['content'])


def test_stratified_is_stable():
    a = [t.task_id for t in T.stratified(T.load_tasks())]
    b = [t.task_id for t in T.stratified(list(reversed(T.load_tasks())))]
    assert a == b and len(set(a)) == 220


def test_text_only_task_set():
    tasks = T.load_tasks()
    ids = {t.task_id for t in tasks}
    assert set(T.TEXT_ONLY_EXCLUDED) <= ids and len(T.TEXT_ONLY_EXCLUDED) == 13
    text_only = T.task_set(tasks, 'text-only-v1')
    assert len(text_only) == 207 and not {t.task_id for t in text_only} & set(T.TEXT_ONLY_EXCLUDED)
    assert [t.task_id for t in T.task_set(tasks, 'all')] == [t.task_id for t in tasks]
    no_expert = {t.task_id for t in tasks if not t.deliverable_files}
    assert len(set(T.TEXT_ONLY_EXCLUDED) - no_expert) == 9   # 9 of the 185 anchored tasks
    order = [t.task_id for t in T.stratified(tasks) if t.task_id not in T.TEXT_ONLY_EXCLUDED]
    assert [t.task_id for t in T.task_set(T.stratified(tasks), 'text-only-v1')] == order
    import pytest
    with pytest.raises(ValueError):
        T.task_set(tasks, 'nope')


def test_select_task_set():
    from types import SimpleNamespace
    import pytest
    from aa_evals.gdpval import run as R
    a = SimpleNamespace(task_set='text-only-v1', task_ids=None, limit=None)
    assert len(R.select(a)) == 207
    assert len(R.select(SimpleNamespace(task_set='all', task_ids=None, limit=None))) == 220
    bad = next(iter(T.TEXT_ONLY_EXCLUDED))
    with pytest.raises(SystemExit):
        R.select(SimpleNamespace(task_set='text-only-v1', task_ids=bad, limit=None))
    assert [t.task_id for t in R.select(SimpleNamespace(task_set='all', task_ids=bad, limit=None))] == [bad]


def test_office_fix():
    import io
    import zipfile
    from aa_evals.gdpval.office_fix import fix_ooxml
    ct = ("<?xml version='1.0' encoding='utf-8'?>\n<ns0:Types xmlns:ns0=\"http://schemas.openxmlformats.org/package/"
          "2006/content-types\"><ns0:Default Extension=\"xml\" ContentType=\"application/xml\" /></ns0:Types>")
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="t" Target="docProps/app.xml"/></Relationships>')
    drels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="h" Target="file:///_redacted_/x" TargetMode="External"'
             '<Relationship Id="rId2" Type="s" Target="styles.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('[Content_Types].xml', ct)
        z.writestr('_rels/.rels', rels)
        z.writestr('word/_rels/document.xml.rels', drels)
        z.writestr('word/document.xml', '<w:document xmlns:w="w"/>')
    out = zipfile.ZipFile(io.BytesIO(fix_ooxml(buf.getvalue())))
    c = out.read('[Content_Types].xml').decode()
    assert c.count('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">') == 1 and 'ns0' not in c
    assert 'PartName="/docProps/app.xml"' in c and 'docProps/app.xml' in out.namelist()
    assert 'docProps/core.xml' not in out.namelist()   # only parts the relationships name
    assert 'TargetMode="External"/><Relationship Id="rId2"' in out.read('word/_rels/document.xml.rels').decode()
    assert out.read('word/document.xml') == b'<w:document xmlns:w="w"/>'
    assert fix_ooxml(out.read('word/document.xml')) is None   # not a zip
    # a clean package is left alone
    clean = io.BytesIO()
    with zipfile.ZipFile(clean, 'w') as z:
        z.writestr('[Content_Types].xml', c)
        z.writestr('_rels/.rels', rels.replace('docProps/app.xml', 'word/document.xml'))
    assert fix_ooxml(clean.getvalue()) is None


def test_gate_backpressure_uses_no_try():
    import asyncio
    import httpx
    from openai import AsyncOpenAI
    from aa_evals.gdpval import client as C
    from aa_evals.gdpval import judge as J
    gate = lambda: httpx.Response(503, headers={'Retry-After': '1', 'X-Gate': '1'},   # noqa: E731
                                  json={'error': {'type': 'server_overloaded', 'message': 'busy'}})
    seen = {'chat': 0}

    def handler(req):
        if req.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'max_model_len': 131072}]})
        if req.url.path.endswith('/tokenize'):
            return httpx.Response(200, json={'count': 100})
        seen['chat'] += 1
        if seen['chat'] <= 3:
            return gate()
        msg = {'role': 'assistant', 'content': 'B is better.\n{"winner": "B"}'}
        return httpx.Response(200, json={'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'm',
                                         'choices': [{'index': 0, 'message': msg, 'finish_reason': 'stop'}],
                                         'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'total_tokens': 105}})
    sleeps = []

    async def fast_sleep(x):
        sleeps.append(x)
    real_sleep = asyncio.sleep
    C.asyncio.sleep = J.asyncio.sleep = fast_sleep
    try:
        mock = httpx.MockTransport(handler)
        j = J.Judge('url:http://h/v1@m', {}, max_tries=1)
        j.http = httpx.AsyncClient(transport=mock)
        out = asyncio.run(j.ask([{'role': 'user', 'content': 'x'}]))
        assert out['winner'] == 'B' and out['tries'] == 1, out
        seen['chat'] = 0
        c = C.CappedVLLMClient(spec='url:http://h/v1@m', served_name='m', body={}, context_cap=131072,
                               max_output=8192, max_tries=1)
        c._http = httpx.AsyncClient(transport=mock)
        c._client_for = lambda url: AsyncOpenAI(base_url=url, api_key='x', max_retries=0,   # noqa: E731
                                                http_client=httpx.AsyncClient(transport=mock))
        from stirrup.core.models import UserMessage
        m = asyncio.run(c.generate([UserMessage(content='hi')], {}))
        assert c.stats['gate_rejections'] == 3 and c.stats.get('infra_retries', 0) == 0, c.stats
        assert 'winner' in m.blocks[-1].text
    finally:
        C.asyncio.sleep = J.asyncio.sleep = real_sleep


def test_gate_conversation_identity_survives_identical_answers():
    import asyncio
    import httpx
    from openai import AsyncOpenAI
    from stirrup.core.models import UserMessage
    from aa_evals.gdpval.client import CappedVLLMClient

    async def exercise():
        seen = []

        def handler(req):
            if req.url.path.endswith('/tokenize'):
                return httpx.Response(200, json={'count': 100})
            seen.append((req.headers.get('X-Conversation-Id'), req.headers.get('X-Gate-Client')))
            return httpx.Response(200, json={
                'id': 'same', 'object': 'chat.completion', 'created': 0, 'model': 'm',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'same'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 1, 'total_tokens': 101}})

        for _ in range(2):
            client = CappedVLLMClient(spec='url:http://h/v1@m', served_name='m', body={},
                                      context_cap=131072, max_output=8192)
            await client._http.aclose()
            client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            api = AsyncOpenAI(base_url='http://h/v1', api_key='x', max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            client._clients['http://h/v1'] = api
            try:
                for _ in range(2):
                    await client.generate([UserMessage(content='same task')], {})
            finally:
                await client._http.aclose()
                await api.close()
        assert all(identity and benchmark == 'gdpval' for identity, benchmark in seen)
        assert seen[0] == seen[1] and seen[2] == seen[3] and seen[0] != seen[2]

    asyncio.run(exercise())


def test_pooled_endpoints():
    import json
    import os
    import tempfile
    import time
    from pathlib import Path
    from aa_evals.gdpval import endpoint as E
    old = E.SERVING_ROOT
    with tempfile.TemporaryDirectory(dir='/tmp') as d:
        E.SERVING_ROOT = Path(d)
        for name, url, served in (('a', 'http://a/v1', 'm'), ('b', 'http://b/v1', 'm'), ('c', 'http://c/v1', 'other'),
                                  ('dead', 'http://d/v1', 'm')):
            (Path(d) / name).mkdir()
            (Path(d) / name / 'endpoints.json').write_text(json.dumps({'served_name': served, 'urls': [url]}))
        os.utime(Path(d) / 'dead' / 'endpoints.json', (time.time() - 3600, time.time() - 3600))
        try:
            ep = E.resolve('serving:a*3,serving:b,serving:dead')
            assert ep['served_name'] == 'm' and ep['urls'] == ['http://a/v1'] * 3 + ['http://b/v1'], ep
            assert E.resolve('serving:a')['urls'] == ['http://a/v1']
            try:
                E.resolve('serving:a,serving:c')
                raise AssertionError('mixed models accepted')
            except ValueError:
                pass
            try:
                E.resolve('serving:dead,serving:dead')
                raise AssertionError('all-dead pool accepted')
            except E.InfraFailure:
                pass
        finally:
            E.SERVING_ROOT = old


def test_pool_helpers():
    from aa_evals.gdpval import elo
    from aa_evals.gdpval import pool as P
    assert P._rng(731, 'r', 't', 0).random() == P._rng(731, 'r', 't', 0).random()   # seeded per attempt
    assert P._rng(731, 'r', 't', 0).random() != P._rng(731, 'r', 't', 1).random()
    assert P.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and P.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    # a pool fit recovers the order: x beats y 75%, y meets the expert 50%
    obs = [('y', elo.EXPERT, 0.5, 1.0)] * 40 + [('x', 'y', 1.0, 1.0)] * 30 + [('x', 'y', 0.0, 1.0)] * 10
    r = elo.fit(obs)
    assert abs(r['y'] - elo.EXPERT_ELO) < 1 and abs((r['x'] - r['y']) - 400 * __import__('math').log10(3)) < 2


def test_memcap_scope_is_the_command_tree():
    import os
    import subprocess
    import time
    from aa_evals.gdpval import sandbox as S
    p = subprocess.Popen(['bash', '-c', 'sleep 30 & sleep 30; wait'], start_new_session=True)
    try:
        time.sleep(0.5)
        rss, pids = S._tree_rss(p.pid)
        assert p.pid in pids and len(pids) >= 2, pids          # the command and its children
        assert os.getpid() not in pids and os.getppid() not in pids   # never this process or its parent
        assert rss > 0
    finally:
        os.killpg(p.pid, 9)
        p.wait()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)

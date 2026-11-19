"""synth.bench: recorded Pi requests replay against a mock OpenAI server with Prometheus counters."""
import array
import asyncio
import json
import socket

from aiohttp import web

from synth.bench import Replay, build_corpus, common_prefix, pi_message, with_generated_history


def recorded_run(root, name, turns):
    pi = root / name / 'run-00' / 'harbor' / 'run' / 'pi-cont16k'
    (pi / 'api-records').mkdir(parents=True)
    lines = []
    for i in range(turns):
        body = {'model': 'recorded-name', 'messages': [{'role': 'user', 'content': f'{name} turn {i}'}],
                'stream': True, 'max_completion_tokens': 16384, 'tools': [], 'seed': 7}
        (pi / 'api-records' / f'{i + 1:06d}.request.json').write_text(json.dumps(body))
        (pi / 'api-records' / f'{i + 1:06d}.transport1.meta.json').write_text(json.dumps({'time': 100.0 + 10 * i + 2}))
        lines.append(json.dumps({'t': 100.0 + 10 * i, 'n_messages': 1}))
    (pi / 'requests.jsonl').write_text('\n'.join(lines) + '\n')


async def exercise(tmp_path, port):
    seen, counters = [], {'gen': 0, 'ok': 0}

    async def generate(req):
        seen.append((await req.json(), req.headers.get('X-data-parallel-rank'), req.headers.get('X-Conversation-Id')))
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await resp.prepare(req)
        for delta in ({'reasoning': 'think '}, {'content': 'done'}):
            await resp.write(b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta}]}).encode() + b'\n\n')
            await asyncio.sleep(0.01)
        await resp.write(b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                                 'usage': {'prompt_tokens': 50, 'completion_tokens': 20}}).encode() + b'\n\n')
        await resp.write(b'data: [DONE]\n\n')
        counters['gen'] += 20
        counters['ok'] += 1
        return resp

    async def metrics(req):
        return web.Response(text=f'vllm:generation_tokens_total{{model="m"}} {counters["gen"]}\n'
                                 f'vllm:request_success_total{{model="m"}} {counters["ok"]}\n'
                                 'vllm:kv_cache_usage_perc{model="m"} 0.5\nvllm:num_requests_running{model="m"} 1\n')

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', port).start()
    try:
        corpus = build_corpus(str(tmp_path / 'attempts' / '*' / 'run-*'), 2, tmp_path / 'corpus.json')
        base = f'http://127.0.0.1:{port}'
        replay = Replay(corpus, [{'base': base + '/v1', 'rank': 0}, {'base': base + '/v1', 'rank': 1}],
                        'served-name', gap_cap=0.2)
        rows = await replay.run([(2, 1, 3)], [base], kv_divisor=2, sample_seconds=1)
    finally:
        await runner.cleanup()
    return corpus, rows, seen


def test_replay_sends_recorded_bodies_and_measures(tmp_path):
    for name, turns in (('a', 3), ('b', 4)):
        recorded_run(tmp_path / 'attempts', name, turns)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    corpus, rows, seen = asyncio.run(exercise(tmp_path, port))
    assert sorted(len(t['requests']) for t in corpus) == [3, 4]
    assert all(t['gaps'][:-1] == [8.0] * (len(t['requests']) - 1) and t['gaps'][-1] is None for t in corpus)
    assert seen and all(body['model'] == 'served-name' and body['stream_options'] == {'include_usage': True}
                        and body['seed'] == 7 for body, _, _ in seen)
    assert {rank for _, rank, _ in seen} == {'0', '1'}
    ids = [identity for _, _, identity in seen]
    assert all(ids) and len(set(ids)) >= 2
    assert len({identity.rsplit('/', 1)[0] for identity in ids}) == 2
    row = rows[0]
    assert row['agents'] == 2 and row['turns'] > 0 and row['errors'] == 0
    assert row['srv_gen_tok_s'] > 0 and row['client_gen_tok_s'] > 0 and row['kv_max'] == 0.25


async def exercise_unstreamed(tmp_path, port, corpus_path):
    seen, counters = [], {'gen': 0, 'ok': 0}

    async def generate(req):
        body = await req.json()
        seen.append((body, req.headers.get('X-data-parallel-rank')))
        await asyncio.sleep(0.05)
        counters.update(gen=counters['gen'] + 30, ok=counters['ok'] + 1)
        return web.json_response({'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'x'},
                                               'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 40, 'completion_tokens': 30,
                                            'prompt_tokens_details': {'cached_tokens': 8}}})

    async def metrics(req):
        return web.Response(text=f'vllm:generation_tokens_total{{model="m"}} {counters["gen"]}\n'
                                 f'vllm:request_success_total{{model="m"}} {counters["ok"]}\n'
                                 'vllm:kv_cache_usage_perc{model="m"} 0.5\nvllm:num_requests_running{model="m"} 3\n')

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', port).start()
    try:
        corpus = build_corpus(str(corpus_path), 2, tmp_path / 'cache.json')
        base = f'http://127.0.0.1:{port}'
        replay = Replay(corpus, [{'base': base + '/v1', 'rank': r} for r in range(2)], 'served-name', stream=False)
        rows = await replay.run([(3, 0.5, 2)], [base], kv_divisor=2, sample_seconds=1)
    finally:
        await runner.cleanup()
    return corpus, rows, seen, replay


def test_unstreamed_replay_of_prebuilt_single_request_corpus(tmp_path):
    """A one-shot corpus (task_synthesis/serve/oneshot_corpus.py): single-request trajectories in the cache format,
    sent without streaming, as the task-synthesis round client sends them."""
    trajectories = []
    for i in range(3):
        path = tmp_path / f'req-{i}.json'
        path.write_text(json.dumps({'model': 'recorded', 'messages': [{'role': 'user', 'content': f'spec {i}'}],
                                    'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'max'}))
        trajectories.append({'run': f's{i}', 'requests': [str(path)], 'gaps': [None]})
    corpus_path = tmp_path / 'corpus.json'
    corpus_path.write_text(json.dumps({'pattern': str(corpus_path), 'size': 3, 'seed': 0, 'trajectories': trajectories}))
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    corpus, rows, seen, replay = asyncio.run(exercise_unstreamed(tmp_path, port, corpus_path))
    assert [t['run'] for t in corpus] == ['s0', 's1']
    assert seen and all(b['stream'] is False and 'stream_options' not in b and b['model'] == 'served-name'
                        and b['reasoning_effort'] == 'max' and b['messages'][0]['content'] in ('spec 0', 'spec 1')
                        for b, _ in seen)
    assert {rank for _, rank in seen} == {'0', '1'}
    assert replay.turns[0]['completion'] == 30 and replay.turns[0]['cached'] == 8 and replay.turns[0]['ttft'] is None
    row = rows[0]
    assert row['errors'] == 0 and row['turns'] > 0 and row['srv_gen_tok_s'] > 0 and row['running_mean'] == 3


async def exercise_adaptive(tmp_path, port):
    seen, counters = [], {'gen': 0, 'ok': 0, 'compute': 0, 'hit': 0}

    async def generate(req):
        seen.append(req.headers.get('X-data-parallel-rank'))
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await resp.prepare(req)
        await resp.write(b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': 'ok'}}]}).encode() + b'\n\n')
        await resp.write(b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                                 'usage': {'prompt_tokens': 500, 'completion_tokens': 20}}).encode() + b'\n\n')
        await resp.write(b'data: [DONE]\n\n')
        counters.update(gen=counters['gen'] + 20, ok=counters['ok'] + 1, compute=counters['compute'] + 50,
                        hit=counters['hit'] + 450)
        return resp

    async def metrics(req):
        return web.Response(text=f'vllm:generation_tokens_total{{model="m"}} {counters["gen"]}\n'
                                 f'vllm:request_success_total{{model="m"}} {counters["ok"]}\n'
                                 f'vllm:prompt_tokens_by_source_total{{model="m",source="local_compute"}} {counters["compute"]}\n'
                                 f'vllm:prompt_tokens_by_source_total{{model="m",source="local_cache_hit"}} {counters["hit"]}\n'
                                 'vllm:kv_cache_usage_perc{model="m"} 0.2\nvllm:num_requests_running{model="m"} 1\n')

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', port).start()
    try:
        corpus = build_corpus(str(tmp_path / 'attempts' / '*' / 'run-*'), 2, tmp_path / 'corpus.json')
        base = f'http://127.0.0.1:{port}'
        targets = [{'base': base + '/v1', 'rank': 0}, {'base': base + '/v1', 'rank': 1}]
        replays = [Replay(corpus, targets, 'served-name', gap_cap=2, seed=k) for k in range(2)]
        policy = {'target_search': 'slope', 'new_agents_per_backend_per_period': 4, 'agent_tokens_prior': 1000}
        out = tmp_path / 'adapt.jsonl'
        await asyncio.gather(*[r.adapt([base], policy, 1_000_000, hours=6 / 3600, tick=1, kv_divisor=2, out=out,
                                       clients=2, index=k, coord=tmp_path / 'coord') for k, r in enumerate(replays)])
    finally:
        await runner.cleanup()
    return replays, [json.loads(line) for line in out.read_text().splitlines()], seen


def test_adaptive_replay_starts_attempts_from_admission(tmp_path):
    for name, turns in (('a', 3), ('b', 4)):
        recorded_run(tmp_path / 'attempts', name, turns)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    replays, rows, seen = asyncio.run(exercise_adaptive(tmp_path, port))
    assert rows and all(r['target'] == 0.4 and r['kv'] == 0.1 for r in rows)
    assert max(r['agents'] for r in rows) > 0 and any(r['room'] > 0 for r in rows)
    assert all(r.turns for r in replays)                      # both clients ran attempts
    assert set(seen) == {'0', '1'}


def test_with_generated_history_and_pi_message():
    body = {'messages': [{'role': 'user', 'content': 'task'},
                         {'role': 'assistant', 'content': 'old0', 'tool_calls': [{'id': 'r0', 'type': 'function',
                                                                                    'function': {'name': 'bash', 'arguments': '{}'}}]},
                         {'role': 'tool', 'content': 'out0', 'tool_call_id': 'r0'},
                         {'role': 'assistant', 'content': 'old1'}, {'role': 'user', 'content': 'next'}]}
    new0 = pi_message('why', '\n\n', [{'id': 'g0', 'name': 'bash', 'arguments': '{"command": "ls"}'}])
    assert new0 == {'role': 'assistant', 'content': '<think>\nwhy</think>\n\n',
                    'tool_calls': [{'id': 'g0', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"ls"}'}}]}
    assert pi_message('', 'hi', []) == {'role': 'assistant', 'content': 'hi'}
    out = with_generated_history(body, {0: new0}, 2)
    assert out['messages'][1] == new0 and out['messages'][2]['tool_call_id'] == 'g0'
    assert out['messages'][3]['content'] == 'old1' and body['messages'][2]['tool_call_id'] == 'r0'   # input untouched
    assert with_generated_history(body, {0: new0}, 3) is None      # history does not hold one answer per turn
    assert with_generated_history(body, {}, 2) is None
    assert common_prefix(array.array('i', range(10000)), array.array('i', list(range(5000)) + [-1])) == 5000
    assert common_prefix(array.array('i', [1, 2]), array.array('i', [1, 2, 3])) == 2


def closed_loop_run(root, name, turns):
    pi = root / name / 'run-00' / 'harbor' / 'run' / 'pi-cont16k'
    (pi / 'api-records').mkdir(parents=True)
    messages, lines = [{'role': 'user', 'content': 'task'}], []
    for i in range(turns):
        body = {'model': 'recorded-name', 'messages': list(messages), 'stream': True, 'tools': [], 'seed': 7}
        (pi / 'api-records' / f'{i + 1:06d}.request.json').write_text(json.dumps(body))
        (pi / 'api-records' / f'{i + 1:06d}.transport1.meta.json').write_text(json.dumps({'time': 100.0 + 10 * i + 2}))
        lines.append(json.dumps({'t': 100.0 + 10 * i, 'n_messages': 1}))
        messages += [{'role': 'assistant', 'content': f'rec{i}', 'tool_calls': [
                         {'id': f'rec-{i}', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"x":1}'}}]},
                     {'role': 'tool', 'content': f'out{i}', 'tool_call_id': f'rec-{i}'}]
    (pi / 'requests.jsonl').write_text('\n'.join(lines) + '\n')


def test_closed_loop_replay_sends_own_answers(tmp_path):
    closed_loop_run(tmp_path / 'attempts', 'a', 4)
    seen, n = [], [0]

    def render(messages):   # "tokens": one id per character, assistant turns as content + arguments
        return ''.join(m['content'] + ''.join(c['function']['arguments'] for c in m.get('tool_calls') or [])
                       for m in messages)

    async def generate(req):
        body = await req.json()
        seen.append(body)
        n[0] += 1
        prompt = render(body['messages'])
        answer = pi_message(f'r{n[0]}', 'c', [{'id': f'gen-{n[0]}', 'name': 'bash', 'arguments': '{"y":2}'}])
        output = render([answer])
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await resp.prepare(req)
        chunks = [{'prompt_token_ids': [ord(c) for c in prompt], 'choices': [
                      {'index': 0, 'delta': {'reasoning': f'r{n[0]}'}, 'token_ids': [ord(c) for c in output[:5]]}]},
                  {'choices': [{'index': 0, 'delta': {'content': 'c'}, 'token_ids': [ord(c) for c in output[5:-7]]}]},
                  {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': f'gen-{n[0]}', 'type': 'function',
                                                                       'function': {'name': 'bash', 'arguments': '{"y": 2}'}}]},
                                'token_ids': [ord(c) for c in output[-7:]]}]},
                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                   'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(output),
                             'prompt_tokens_details': {'cached_tokens': 3}}}]
        for chunk in chunks:
            await resp.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
        await resp.write(b'data: [DONE]\n\n')
        return resp

    async def metrics(req):
        return web.Response(text='vllm:generation_tokens_total{model="m"} 1\nvllm:request_success_total{model="m"} 1\n'
                                 'vllm:kv_cache_usage_perc{model="m"} 0.5\nvllm:num_requests_running{model="m"} 1\n')

    async def main(port):
        app = web.Application()
        app.router.add_post('/v1/chat/completions', generate)
        app.router.add_get('/metrics', metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', port).start()
        try:
            corpus = build_corpus(str(tmp_path / 'attempts' / '*' / 'run-*'), 1, tmp_path / 'corpus.json')
            base = f'http://127.0.0.1:{port}'
            replay = Replay(corpus, [{'base': base + '/v1', 'rank': None}], 'served-name', gap_cap=0.05,
                            random_start=False, closed_loop=True)
            await replay.run([(1, 0, 1)], [base], sample_seconds=1, stats_out=tmp_path / 'turns.jsonl')
        finally:
            await runner.cleanup()
        return replay

    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    replay = asyncio.run(main(port))
    assert len(seen) >= 4 and all(b['return_token_ids'] is True for b in seen)
    second = seen[1]['messages']
    assert second[1] == pi_message('r1', 'c', [{'id': 'gen-1', 'name': 'bash', 'arguments': '{"y":2}'}])
    assert second[2] == {'role': 'tool', 'content': 'out0', 'tool_call_id': 'gen-1'}
    stats = [json.loads(line) for line in (tmp_path / 'turns.jsonl').read_text().splitlines()]
    assert stats[0]['k'] == 0 and 'lcp' not in stats[0] and stats[0]['cached'] == 3
    for s in stats[1:4]:
        assert s['closed'] and s['lcp'] == s['prev_prompt'] + s['prev_output'] and s['old_next'] == []
    assert all('message' not in t and 'ids' not in t for t in replay.turns)


def test_wire_delta_corpus_rebuilds_bodies(tmp_path):
    """GLM's delta wire log: prefix copies, tools carried until they change, unfinished streams applied but skipped."""
    from synth.bench import index_run, recorded_body
    import os
    pi = tmp_path / 'attempts' / 'g' / 'raw' / 'harbor' / 'pi-cont16k'
    (pi / 'wire-responses').mkdir(parents=True)
    user = lambda t: {'role': 'user', 'content': t}
    deltas = [(0, [user('sys'), user('q')], [{'name': 't1'}]), (2, [user('a1')], None), (3, [user('a2')], None),
              (2, [user('b3')], [{'name': 't2'}])]
    lines = []
    for i, (copy, append, tools) in enumerate(deltas, 1):
        record = {'request_index': i, 'time': 100.0 + 10 * i,
                  'messages_delta': {'copy_previous_prefix_messages': copy, 'append_messages': append},
                  'parameters': {'model': 'glm', 'seed': i, 'max_completion_tokens': 16384}}
        if tools is not None:
            record['tools'] = tools
        lines.append(json.dumps(record))
        response = pi / 'wire-responses' / f'{i:06d}.response'
        response.write_text('data: {}\n\n' + ('data: [DONE]\n\n' if i != 3 else ''))
        os.utime(response, (100.0 + 10 * i + 4, 100.0 + 10 * i + 4))
    (pi / 'wire-requests.jsonl').write_text('\n'.join(lines) + '\n')
    t = index_run(tmp_path / 'attempts' / 'g')
    assert t['requests'] == [0, 1, 3] and t['gaps'] == [5.5, 15.5, None]
    loaded = []
    bodies = [asyncio.run(recorded_body(t, k, loaded)) for k in range(3)]
    assert [[m['content'] for m in b['messages']] for b in bodies] == [['sys', 'q'], ['sys', 'q', 'a1'], ['sys', 'q', 'b3']]
    assert [b['tools'] for b in bodies] == [[{'name': 't1'}], [{'name': 't1'}], [{'name': 't2'}]]
    assert [b['seed'] for b in bodies] == [1, 2, 4]


def test_api_dir_corpus(tmp_path):
    """Qwen3.8-27B's pi-cont16k attempts: api/<start ns>-<id>/{request.json,response-0.json}, non-200 skipped."""
    from synth.bench import index_run
    api = tmp_path / 'a' / 'raw' / 'trial' / 'pi-cont16k' / 'api'
    for i, (start, end, status) in enumerate([(100, 104, 200), (105, 106, 500), (110, 120, 200), (121, 125, 200)]):
        d = api / f'{start * 10**9}-{i:032x}'
        d.mkdir(parents=True)
        (d / 'request.json').write_text(json.dumps({'messages': [], 'seed': i}))
        (d / 'response-0.json').write_text(json.dumps({'status': status, 'time': float(end)}))
    t = index_run(tmp_path / 'a')
    assert [json.loads(open(r).read())['seed'] for r in t['requests']] == [0, 2, 3]
    assert t['gaps'] == [6.0, 1.0, None]


def test_replay_conversation_ids_are_unique_across_same_seed_instances():
    a = Replay([], [], 'm', seed=0)
    b = Replay([], [], 'm', seed=0)
    assert a.conversation_prefix != b.conversation_prefix

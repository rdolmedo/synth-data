"""Gate tests against a fake vLLM (aiohttp): passthrough, stream relay, queueing and priority, metrics."""
import asyncio
import json
import time

import aiohttp
from aiohttp import web
import pytest

from aa_evals.qwenserve.gate import Client, Gate, Request, conv_key, conversation_key, last_usage, prompt_estimate


def fake_vllm(state):
    async def metrics(req):
        return web.Response(text='vllm:cache_config_info{block_size="1600",kv_cache_size_tokens="800000",num_cpu_blocks="None",num_gpu_blocks="663"} 1.0\n'
                                 f'vllm:kv_cache_usage_perc {state["kv"]}\nvllm:num_requests_running 1\n'
                                 'vllm:num_requests_waiting 0\nvllm:generation_tokens_total 10\n')

    async def chat(req):
        body = await req.json()
        state['seen'].append(body['messages'][-1]['content'])
        while not state['release'].is_set():          # as vLLM: a request whose client left is aborted
            await asyncio.sleep(0.02)
            if req.transport is None or req.transport.is_closing():
                state['aborted'] = state.get('aborted', 0) + 1
                return web.Response(status=499)
        if not body.get('stream'):
            return web.json_response({'choices': [{'message': {'content': 'ok'}}],
                                      'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}})
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await resp.prepare(req)
        for i in range(3):
            await resp.write(f'data: {json.dumps({"choices": [{"delta": {"content": str(i)}}]})}\n\n'.encode())
        await resp.write(b'data: [DONE]\n\n')
        return resp

    async def models(req):
        return web.json_response({'data': [{'id': 'm'}]})
    app = web.Application()
    app.router.add_get('/metrics', metrics)
    app.router.add_post('/v1/chat/completions', chat)
    app.router.add_get('/v1/models', models)
    return app


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, f'http://127.0.0.1:{runner.addresses[0][1]}'


def body(first, last, stream=False, tools=True):
    return {'model': 'm', 'stream': stream, 'tools': [{'type': 'function'}] if tools else None,
            'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': first},
                         {'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': last}]}


def test_gate_end_to_end():
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=2, tick=3600, policy={'agent_tokens_prior': 1024})
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)   # first tick: geometry + initial room
        assert gate.capacity == 800_000 and gate.block == 1600
        async with aiohttp.ClientSession() as s:
            async with s.get(gbase + '/v1/models') as r:     # ungated passthrough
                assert (await r.json())['data'][0]['id'] == 'm'
            gate.room = 1                                   # one new conversation may start
            post = lambda b: asyncio.create_task(s.post(gbase + '/v1/chat/completions', json=b))
            t1 = post(body('A', 'a1'))
            await asyncio.sleep(0.2)
            t2 = post(body('B', 'b1'))                      # new conversation: no room left -> queued
            await asyncio.sleep(0.2)
            assert state['seen'] == ['a1'] and gate.queued('new') == 1
            state['release'].set()
            r1 = await t1
            assert r1.status == 200 and 'X-Gate-Queue-Ms' in r1.headers
            assert (await r1.json())['usage']['total_tokens'] == 10
            await asyncio.sleep(0.2)
            assert gate.convs[conversation_key(body('A', ''))[0]]['tokens'] == 10   # idle context tracked from usage
            # a turn of live conversation A goes before queued new conversation B, and streams relay intact
            r3 = await s.post(gbase + '/v1/chat/completions', json=body('A', 'a2', stream=True))
            text = await r3.text()
            assert text.count('data: ') == 4 and text.endswith('data: [DONE]\n\n')
            assert state['seen'] == ['a1', 'a2'] and gate.queued('new') == 1
            gate.room = 1
            gate.dispatch()
            assert (await t2).status == 200 and state['seen'][-1] == 'b1'
            async with s.get(gbase + '/metrics') as r:
                m = await r.text()
            assert 'vllm:kv_cache_usage_perc' in m and 'gate:queue_seconds_count{class="new"} 2' in m
            assert 'gate:queue_seconds_count{class="cont"} 1' in m
        assert gate.inflight == 0
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_single_turn_requests_are_not_tracked_idle():
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        state['release'].set()
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=4, tick=3600)
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        gate.room = 5
        q = {'model': 'm', 'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'q'}]}
        async with aiohttp.ClientSession() as s:
            async with s.post(gbase + '/v1/chat/completions', json=q) as r:
                assert r.status == 200
        gate.observe({'kv': 0.1, 'wait': 0, 'gen': 0}, 1e12)
        assert not gate.convs
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_last_usage():
    assert last_usage(b'data: {"choices":[],"usage":{"prompt_tokens":5,"total_tokens":9,'
                      b'"completion_tokens_details":{"reasoning_tokens":2}}}\n\ndata: [DONE]')['total_tokens'] == 9
    assert last_usage(b'data: {"choices":[]}') is None


def test_offload_tokens_from_log(tmp_path):
    from aa_evals.qwenserve.gate import log_offload_tokens
    f = tmp_path / 'vllm.log'
    f.write_text('(EngineCore pid=1) INFO [worker.py:213] SimpleCPUOffloadWorker [CPU]: 17 tensors, 5930 CPU blocks (159.98 GB)\n')
    assert log_offload_tokens(f, 832) == 5930 * 832
    assert log_offload_tokens(tmp_path / 'missing', 832) is None


def queue_up(gate, client, n, *, multi=True, tokens=100, cap=None, cont=False):
    loop = asyncio.get_event_loop()
    c = gate.clients.setdefault(client, Client(gate.policy['output_tokens_prior']))
    rs = [Request(loop.create_future(), f'{client}{i}{cont}', multi, client, tokens, cap) for i in range(n)]
    (c.cont if cont else c.new).extend(rs)
    return rs


def admitted(rs):
    return [r.fut.result() if r.fut.done() else None for r in rs]


def test_round_robin_between_clients():
    async def main():
        gate = Gate('http://unused', max_inflight=100, policy={'short_tokens': 0})
        gate.causes, gate.gpu_free = [], 1e9
        agents = queue_up(gate, 'agents', 5)
        live = queue_up(gate, 'agents', 3, cont=True)
        qa = queue_up(gate, 'qa', 2, multi=False, tokens=50_000)
        gate.room = 2
        gate.dispatch()
        # one admission per client per round, a live turn first within a client: the agents' turns use their rounds
        # (no room needed) while the QA client takes the room, instead of waiting behind the agents' new ones
        assert admitted(live) == ['cont'] * 3
        assert admitted(agents) == [None] * 5 and admitted(qa) == ['new', 'new']
        gate.room = 1
        gate.dispatch()
        assert admitted(agents)[:2] == ['new', None]
        # the in-flight cap is shared round-robin too: live turns of one client do not starve another's
        gate2 = Gate('http://unused', max_inflight=2, policy={'short_tokens': 0})
        gate2.causes, gate2.room, gate2.gpu_free = [], 10, 1e9
        a = queue_up(gate2, 'a', 4, cont=True)
        b = queue_up(gate2, 'b', 1, multi=False, tokens=50_000)
        gate2.dispatch()
        assert admitted(a).count('cont') == 1 and admitted(b) == ['new']
    asyncio.run(main())


def test_short_lane_and_fractional_room():
    async def main():
        gate = Gate('http://unused', max_inflight=100, policy={'short_max_inflight': 3})
        gate.causes, gate.room, gate.gpu_free = [], 0, 1e9
        agents = queue_up(gate, 'agents', 3)
        short = queue_up(gate, 'omni', 5, multi=False, tokens=500, cap=1000)
        gate.dispatch()
        assert admitted(agents) == [None] * 3 and admitted(short) == ['short'] * 3 + [None, None]   # lane cap
        gate.short_inflight = 0
        gate.causes = ['kv_inflight_high']             # KV brakes stop the lane; vLLM's queue does not
        gate.dispatch()
        assert admitted(short)[3] is None
        gate.causes, gate.gpu_free = ['inference_queue'], -500_000   # decode-growth budget spent: short still goes
        gate.dispatch()
        assert admitted(short)[3:] == ['short', 'short'] and admitted(agents) == [None] * 3
        # a long single-turn request costs room by its expected KV: 0.25 of an 8k-token agent here
        g = Gate('http://unused', max_inflight=100, policy={'short_tokens': 0})
        g.causes, g.room, g.gpu_free = [], 1, 1e9
        qa = queue_up(g, "hle", 10, multi=False, tokens=1000, cap=1000)
        g.dispatch()
        assert admitted(qa).count("new") == 4
    asyncio.run(main())


def test_learned_completion_sets_the_lane():
    gate = Gate('http://unused', policy={'short_tokens': 16384, 'output_tokens_prior': 32768})
    c = gate.clients.setdefault('omni', Client(32768))
    r = Request(None, 'k', False, 'omni', 2000, None)
    assert gate.expected_tokens(r) == 2000 + 32768        # unknown client: the prior
    for _ in range(10):
        c.learn(700)
    c.estimate(gate.policy)
    assert gate.expected_tokens(r) == 2700                # learned: short
    assert gate.expected_tokens(Request(None, 'k', False, 'omni', 2000, 500)) == 2500   # max_tokens bounds it
    # in-flight requests' output so far is a lower bound on the p90 (the first to complete are the short ones)
    for i in range(3):
        long = Request(None, f'k{i}', False, 'omni', 0, None)
        long.stats['events'] = 50_000
        c.running.add(long)
    assert c.estimate(gate.policy) == 50_000


def snap(kv=0.1, wait=0, preempt=0):
    return {'kv': kv, 'wait': wait, 'run': 1, 'gen': 0, 'preempt': preempt}


def test_decode_growth_budget_and_brakes():
    async def main():
        gate = Gate('http://unused', max_inflight=128, capacity=1_000_000, block=16,
                    policy={'short_tokens': 0, 'new_agents_per_backend_per_period': 100})
        hle = gate.clients.setdefault('hle', Client(gate.policy['output_tokens_prior']))
        for _ in range(5):
            hle.learn(100_000)
        gate.observe(snap(), 0.0)
        gate.observe(snap(), 5.0)
        gate.room = 100
        rs = queue_up(gate, 'hle', 20, multi=False, tokens=1000)
        gate.dispatch()
        # (0.9 - 0.1) x 1M of GPU KV holds 7 requests of 1k prompt + 100k expected output, not the room's 100
        assert admitted(rs).count('new') == 7
        gate.observe(snap(), 10.0)                          # their reservations now count against the budget
        assert gate.reserved == 700_000
        gate.room = 100
        gate.dispatch()
        assert admitted(rs).count('new') == 7
        # an unlearned client without max_tokens: at most probe_inflight uncapped single-turn requests in flight
        g = Gate('http://unused', max_inflight=128, policy={'short_tokens': 0})
        g.causes, g.room, g.gpu_free = [], 1000, 1e12
        probe = queue_up(g, 'new-client', 20, multi=False, tokens=1000)
        g.dispatch()
        assert admitted(probe).count('new') == 16
        # hard brakes: vLLM's waiting queue on two consecutive ticks; a preemption holds and shrinks the GPU budget
        b = Gate('http://unused', capacity=1_000_000, block=16)
        b.observe(snap(), 0.0)
        b.observe(snap(wait=3), 5.0)
        assert b.brakes == []
        b.observe(snap(wait=3), 10.0)
        assert b.brakes == ['vllm_waiting']
        b.observe(snap(preempt=2), 15.0)
        assert b.brakes == ['preemption_hold'] and abs(b.gpu_frac - 0.81) < 1e-9
        r = queue_up(b, 'x', 1, multi=False, tokens=10, cap=10)
        b.room, b.causes, b.inflight = 10, [], 1
        b.dispatch()
        assert admitted(r) == [None]
    asyncio.run(main())


def test_backpressure_503():
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=8, tick=3600, policy={'reject_queue_new': 2, 'reject_after_seconds': 60})
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        gate.room = 0
        async with aiohttp.ClientSession() as s:
            post = lambda b, h=None: asyncio.create_task(s.post(gbase + '/v1/chat/completions', json=b, headers=h))
            held = [post(body(f'Q{i}', 'x'), {'X-Gate-Client': 'tb'}) for i in range(2)]
            await asyncio.sleep(0.2)
            gate.clients['tb'].last_admit -= 120              # a stalled client: nothing admitted for 2 min
            r = await s.post(gbase + '/v1/chat/completions', json=body('Q9', 'x'), headers={'X-Gate-Client': 'tb'})
            assert r.status == 503 and r.headers['Retry-After'] == '60' and r.headers['X-Gate'] == '1'
            assert (await r.json())['error']['type'] == 'server_overloaded'
            other = post(body('O', 'x'), {'X-Gate-Client': 'hle'})       # another client still queues
            await asyncio.sleep(0.2)
            assert gate.queued('new') == 3
            gate.observe(None, time.time() + 61)                         # queued too long: 503, retry later
            for t in held + [other]:
                assert (await t).status == 503
            assert gate.queued('new') == 0 and gate.inflight == 0 and gate.counts['rejected'] == 4
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_prompt_estimate():
    p = {'messages': [{'role': 'system', 'content': 'x' * 300},
                      {'role': 'user', 'content': [{'type': 'text', 'text': 'y' * 3000},
                                                   {'type': 'image_url', 'image_url': {'url': 'data:' + 'z' * 10**6}}]}]}
    assert prompt_estimate(p) == 1100 + 1024 + len(json.dumps('')) // 3
def test_underused_engine_never_holds_new_conversations():
    """aa-qwen-tbsci, 2026-11-03: idle agents' contexts above a target at its floor held room at 0 on an idle engine."""
    gate = Gate('http://unused', capacity=3_887_422, block=832, offload=4_933_760)
    for i in range(22):
        gate.convs[f'c{i}'] = {'tokens': 118_000, 'last': 1000.0, 'inflight': 0, 'multi': True}
    gate.b['target'] = 0.3
    idle = lambda kv=0.0, wait=0, preempt=0: {'kv': kv, 'wait': wait, 'run': 0, 'gen': 0, 'preempt': preempt}
    gate.observe(idle(), 1000.0)
    gate.observe(idle(), 1005.0)
    assert gate.room == 16 and gate.status['underused'] and gate.b['target'] >= 0.6
    # a busy engine keeps the normal rule; so does one with requests waiting inside vLLM or a preemption
    for snap in (idle(kv=0.5), idle(wait=3), idle(preempt=1)):
        g = Gate('http://unused', capacity=3_887_422, block=832, offload=4_933_760)
        for i in range(22):
            g.convs[f'c{i}'] = {'tokens': 118_000, 'last': 1000.0, 'inflight': 0, 'multi': True}
        g.b['target'] = 0.3
        g.observe(idle(), 1000.0)
        g.observe(snap, 1005.0)
        assert g.room == 0 and not g.status['underused']
    # idle conversations are released after the TTL
    gate.observe(idle(), 1000.0 + gate.idle_ttl + 1)
    assert not gate.convs


def test_client_disconnect_drops_queued_and_aborts_inflight():
    """qa1, 2026-11-03: requests of removed client jobs kept running and queued behind the gate."""
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=1, tick=3600)
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        gate.room = 10
        q = lambda i: {'model': 'm', 'messages': [{'role': 'user', 'content': f'q{i}'}], 'max_tokens': 10}

        async def post(i, timeout):
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
                with pytest.raises(asyncio.TimeoutError):
                    await s.post(gbase + '/v1/chat/completions', json=q(i))
        inflight = asyncio.create_task(post(0, 2.0))   # non-streamed: nothing is written to its client until it ends
        await asyncio.sleep(0.1)
        await post(1, 0.3)                              # queued behind max_inflight 1, then its client leaves
        await asyncio.sleep(0.1)
        gate.observe(None, time.time())                # the tick drops it
        await asyncio.sleep(0.1)
        assert gate.queued('new') == 0 and gate.counts['left_queued'] == 1 and gate.inflight == 1
        await inflight                                  # the in-flight request's client leaves at 2 s
        await asyncio.sleep(1.3)                        # the watcher polls every second
        assert state.get('aborted') == 1 and gate.counts['client_gone'] == 1 and gate.inflight == 0
        assert state['seen'] == ['q0']                  # the queued request never reached vLLM
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_backpressure_spares_a_moving_queue():
    async def main():
        gate = Gate('http://unused', max_inflight=1000, policy={'short_tokens': 0})
        gate.causes, gate.gpu_free = [], 1e12
        rs = queue_up(gate, 'tb', 10)
        c = gate.clients['tb']
        for r in rs:
            r.t -= 700                                  # all queued longer than reject_after_seconds
        c.last_admit = time.time() - 30                 # ... but the client was admitted 30 s ago: keep them
        gate.observe(None, time.time())
        assert admitted(rs) == [None] * 10 and len(c.new) == 10
        c.last_admit = time.time() - 700                # stalled: they get a 503
        gate.observe(None, time.time())
        assert admitted(rs) == ['reject'] * 10
    asyncio.run(main())


def test_pool_routes_new_by_free_budget_and_keeps_turns_on_their_engine():
    async def main():
        gate = Gate(['http://e0', 'http://e1'], max_inflight=2, policy={'short_tokens': 0})
        e0, e1 = gate.engines
        for e, free in ((e0, 1e6), (e1, 3e6)):
            e.causes, e.room, e.gpu_free = [], 10, free
        a = queue_up(gate, 'agents', 3)
        gate.dispatch()
        # each new conversation goes to the engine with the most free budget left after the previous one
        assert admitted(a) == ['new'] * 3 and [r.engine.i for r in a] == [1, 1, 0]
        assert gate.convs['agents0False']['engine'] == 1 and gate.inflight == 3
        # a later turn of that conversation goes back to engine 1 (its prefix), even with engine 0 freer
        e0.gpu_free, e1.gpu_free = 9e6, 0
        gate.release(a[1], gate.clients['agents'], 'new', 300)     # frees a slot on engine 1
        turn = Request(asyncio.get_event_loop().create_future(), 'agents0False', True, 'agents', 0, None)
        gate.clients['agents'].cont.append(turn)
        gate.dispatch()
        assert turn.fut.result() == 'cont' and turn.engine is e1 and e1.inflight == 2 == e1.max_inflight
        # a full engine holds its own turns but not the other engine's
        t1 = Request(asyncio.get_event_loop().create_future(), 'agents1False', True, 'agents', 0, None)
        t0 = Request(asyncio.get_event_loop().create_future(), 'agents2False', True, 'agents', 0, None)
        gate.clients['agents'].cont.extend([t1, t0])
        gate.dispatch()
        assert not t1.fut.done() and t0.fut.result() == 'cont' and t0.engine is e0
        # brakes are per engine: a preempting engine takes no new requests while the other still does
        e0.brakes, e0.room, e1.room, e1.gpu_free = ['preemption_hold'], 10, 10, 1e6
        e1.max_inflight = 10
        b = queue_up(gate, 'qa', 2, multi=False, tokens=1000, cap=1000)
        gate.dispatch()
        assert admitted(b) == ['new', 'new'] and {r.engine.i for r in b} == {1}
    asyncio.run(main())


def test_pool_end_to_end_metrics_carry_engine_labels():
    async def main():
        states = [{'kv': 0.1 * (i + 1), 'seen': [], 'release': asyncio.Event()} for i in range(2)]
        for s in states:
            s['release'].set()
        ups = [await serve(fake_vllm(s)) for s in states]
        gate = Gate([b for _, b in ups], max_inflight=4, tick=3600)
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        assert [e.capacity for e in gate.engines] == [800_000, 800_000]
        for e in gate.engines:
            e.room = 5
        async with aiohttp.ClientSession() as s:
            r = await s.post(gbase + '/v1/chat/completions', json=body('A', 'a1'))
            assert r.status == 200
            assert states[0]['seen'] == ['a1'] and states[1]['seen'] == []   # engine 0 has more free KV (0.1 vs 0.2)
            async with s.get(gbase + '/metrics') as r:
                m = await r.text()
            assert 'vllm:kv_cache_usage_perc{replica="0"} 0.1' in m and 'vllm:kv_cache_usage_perc{replica="1"} 0.2' in m
            assert m.count('# TYPE') == len(set(x for x in m.splitlines() if x.startswith('# TYPE')))
            assert 'gate:room{engine="1"}' in m and 'gate:engines 2' in m
            async with s.get(gbase + '/gate/status') as r:
                st = await r.json()
            assert len(st['engines']) == 2 and st['inflight'] == 0
        await g.cleanup()
        for u, _ in ups:
            await u.cleanup()
    asyncio.run(main())


def test_client_share_holds_over_share_clients_for_a_waiting_large_request():
    async def main():
        gate = Gate('http://unused', max_inflight=128, capacity=1_000_000, block=16,
                    policy={'short_tokens': 0, 'client_share': True, 'share_wait_seconds': 0,
                            'new_agents_per_backend_per_period': 100})
        hle = gate.clients.setdefault('hle', Client(gate.policy['output_tokens_prior']))
        for _ in range(5):
            hle.learn(100_000)
        hle.estimate(gate.policy)
        # the agents hold most of the KV with three live conversations
        loop = asyncio.get_event_loop()
        ag = gate.clients.setdefault('agents', Client(gate.policy['output_tokens_prior']))
        for _ in range(5):
            ag.learn(10_000)
        ag.estimate(gate.policy)
        for i in range(3):
            r = Request(loop.create_future(), f'live{i}', True, 'agents', 0, None)
            gate.convs[r.key] = {'tokens': 200_000, 'last': time.time(), 'inflight': 1, 'multi': True, 'engine': 0}
            ag.running.add(r)
        gate.room, gate.causes, gate.gpu_free, gate.inflight = 100, [], 50_000, 3
        new_agents = queue_up(gate, 'agents', 2)
        big = queue_up(gate, 'hle', 20, multi=False, tokens=1000)
        gate.dispatch()
        # both want more than half of the 900k budget: max-min shares are 450k each. HLE (usage 0) waits with a 101k
        # request that does not fit the 50k left: the agents (630k of KV, over their share) start no new
        # conversation, so what they free goes to HLE
        assert gate.hold == {'agents'} and admitted(new_agents) == [None, None] and admitted(big)[0] is None
        assert gate.share['hle'][1] == gate.share['agents'][1] == 450_000
        gate.gpu_free = 200_000
        gate.dispatch()
        assert admitted(big)[:2] == ['new', None] and admitted(new_agents) == [None, None]
        # HLE's queue gone: nobody waits under its share, nobody is held (work-conserving)
        for r in big[1:]:
            gate.clients['hle'].new.remove(r)
        gate.dispatch()
        assert gate.hold == frozenset() and admitted(new_agents) == ['new', 'new']
        # live turns of a held client are never held
        g2 = Gate('http://unused', capacity=1_000_000, policy={'client_share': True, 'share_wait_seconds': 0,
                                                                'short_tokens': 0})
        g2.room, g2.causes, g2.gpu_free = 10, [], 0
        g2.inflight = 1
        big2 = queue_up(g2, 'hle', 5, multi=False, tokens=100_000, cap=1000)
        ag2 = g2.clients.setdefault('agents', Client(1000))
        hog = Request(loop.create_future(), 'hog', True, 'agents', 0, None)
        g2.convs['hog'] = {'tokens': 900_000, 'last': time.time(), 'inflight': 1, 'multi': True, 'engine': 0}
        ag2.running.add(hog)
        turn = queue_up(g2, 'agents', 1, cont=True)
        fresh = queue_up(g2, 'agents', 1)
        g2.room = 10
        g2.gpu_free = 1e9
        g2.dispatch()
        assert g2.hold == {'agents'} and admitted(turn) == ['cont'] and admitted(fresh) == [None]
    asyncio.run(main())


def test_label_engine():
    from aa_evals.qwenserve.gate import label_engine
    t = '# HELP x h\n# TYPE x gauge\nx 1\ny{a="b"} 2.5\n'
    assert label_engine(t, 3) == '# HELP x h\n# TYPE x gauge\nx{replica="3"} 1\ny{replica="3",a="b"} 2.5\n'
    assert label_engine(t, 1, comments=False) == 'x{replica="1"} 1\ny{replica="1",a="b"} 2.5\n'


def test_short_lane_demotes_requests_that_turn_out_long():
    async def main():
        gate = Gate('http://unused', max_inflight=100, capacity=1_000_000, block=16, policy={'short_max_inflight': 2})
        gate.causes, gate.room, gate.gpu_free = [], 0, 1e9
        hle = gate.clients.setdefault('hle', Client(32768))
        for _ in range(5):
            hle.learn(2000)                            # its first completions were short ones
        hle.estimate(gate.policy)
        rs = queue_up(gate, 'hle', 3, multi=False, tokens=500)
        gate.dispatch()
        assert admitted(rs) == ['short', 'short', None] and gate.short_inflight == 2
        rs[0].stats['events'] = 20_000                 # it streamed past short_tokens
        rs[1].t_admit -= 1000                          # it ran past short_seconds
        gate.observe(snap(), time.time())
        assert gate.short_inflight == 0 and gate.counts['short_demoted'] == 2
        gate.dispatch()
        # the long one's output so far also raised HLE's p90 above short_tokens: the next one is not short any more
        assert admitted(rs)[2] == 'new' and gate.short_inflight == 0
        rs[1].cls = 'short'
        gate.short_inflight = 1
        gate.release(rs[0], hle, 'short', 100)         # a demoted request no longer counts against the lane
        assert gate.short_inflight == 1
    asyncio.run(main())


def test_maxmin_share():
    from aa_evals.qwenserve.gate import maxmin
    w = {'a': 1, 'h': 1, 's': 1}
    sh = maxmin(3_000_000, {'a': 5_000_000, 'h': 5_000_000, 's': 100_000}, w)
    assert sh['s'] == 100_000 and sh['a'] == sh['h'] == 1_450_000          # the short client's slack goes to the others
    sh = maxmin(3_000_000, {'a': 5_000_000, 'h': 5_000_000}, {'a': 1, 'h': 2})
    assert sh['a'] == 1_000_000 and sh['h'] == 2_000_000
    assert maxmin(1_000, {'a': 10, 'h': 20}, {'a': 1, 'h': 1}) == {'a': 10, 'h': 20}


def test_protected_client_needs_no_room():
    """ab-long, 2026-11-03: with room 0 (idle agents' contexts over the kv_fit target) protected HLE never started."""
    async def main():
        loop = asyncio.get_event_loop()
        g = Gate('http://unused', capacity=1_000_000, policy={'client_share': True, 'share_wait_seconds': 0,
                                                               'short_tokens': 0})
        g.room, g.causes, g.gpu_free, g.inflight = 0, [], 300_000, 1
        ag = g.clients.setdefault('agents', Client(1000))
        hog = Request(loop.create_future(), 'hog', True, 'agents', 0, None)
        g.convs['hog'] = {'tokens': 800_000, 'last': time.time(), 'inflight': 1, 'multi': True, 'engine': 0}
        ag.running.add(hog)
        fresh = queue_up(g, 'agents', 3)
        hle = queue_up(g, 'hle', 10, multi=False, tokens=100_000, cap=1000)
        g.dispatch()
        assert g.protect == {'hle'} and g.hold == {'agents'}
        assert admitted(hle)[:3] == ['new', 'new', None] and admitted(fresh) == [None] * 3   # budget-bound, not room
    asyncio.run(main())


def test_monitor_reads_pooled_metrics():
    from aa_evals.qwenserve.gate import label_engine
    from aa_evals.qwenserve.monitor import hist, parse, per_engine, quantile
    vllm = '# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{engine="0",model_name="m"} {kv}\n'
    text = label_engine(vllm.replace('{kv}', '0.25'), 0) + label_engine(vllm.replace('{kv}', '0.75'), 1, comments=False)
    assert per_engine(parse(text), 'vllm:kv_cache_usage_perc') == [0.25, 0.75]
    h = lambda counts: parse(''.join(f'gate:queue_seconds_bucket{{class="new",le="{le}"}} {c}\n'
                                     for le, c in zip(('1', '10', '+Inf'), counts)))
    h0, h1 = hist(h((0, 0, 0)), 'new'), hist(h((50, 90, 100)), 'new')
    assert quantile(h0, h1, 0.5) == 1.0 * 50 / 50 and quantile(h0, h1, 0.95) == 10


def test_upstream_reset_before_response_is_resent():
    """A pooled upstream connection the engine closed (reset / disconnect before any response) is retried on a fresh
    connection; an engine that keeps failing still gets a 502."""
    async def main():
        calls = {'n': 0, 'fail': 1}

        async def tokenize(req):
            calls['n'] += 1
            await req.read()
            if calls['n'] <= calls['fail']:
                req.transport.close()          # as an idle keep-alive connection closed under the request
                await asyncio.sleep(0.05)
                return web.Response(text='never sent')
            return web.json_response({'ok': calls['n']})
        app = web.Application()
        app.router.add_post('/tokenize', tokenize)
        async def metrics(req):
            return web.Response(text='')
        app.router.add_get('/metrics', metrics)
        up, base = await serve(app)
        gate = Gate(base, max_inflight=2, tick=3600)
        g, gbase = await serve(gate.app())
        async with aiohttp.ClientSession() as s:
            async with s.post(gbase + '/tokenize', json={'x': 1}) as r:
                assert r.status == 200 and (await r.json())['ok'] == 2
            calls.update(n=0, fail=99)
            async with s.post(gbase + '/tokenize', json={'x': 1}) as r:
                assert r.status == 502 and calls['n'] == 3   # first try + UPSTREAM_RETRIES
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def first_turn(task):
    return {'model': 'm', 'tools': [{'type': 'function'}],
            'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': task}]}


def second_turn(task, answer):
    return {'model': 'm', 'tools': [{'type': 'function'}],
            'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': task},
                         {'role': 'assistant', 'content': answer,
                          'tool_calls': [{'id': 'c0', 'type': 'function',
                                          'function': {'name': 'bash', 'arguments': '{"command": "ls"}'}}]},
                         {'role': 'tool', 'tool_call_id': 'c0', 'content': 'out'}]}


def test_concurrent_repeats_each_go_through_admission():
    """5 repeats of one task (identical system + user messages) start as 5 new conversations: with room for one, one is
    admitted and 4 wait as new (before 2026-11-06 they shared a key and entered as turns of repeat 1). Each repeat's
    second turn then claims a finished first turn and is a live turn (admitted without room); a conversation header
    keys conversations by itself."""
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=8, tick=3600, policy={'agent_tokens_prior': 1024})
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        async with aiohttp.ClientSession() as s:
            post = lambda b, h=None: asyncio.create_task(s.post(gbase + '/v1/chat/completions', json=b, headers=h))  # noqa: E731
            gate.room = 1
            firsts = [post(first_turn('T')) for _ in range(5)]
            await asyncio.sleep(0.3)
            assert len(state['seen']) == 1 and gate.queued('new') == 4 and gate.queued('cont') == 0
            gate.room = 4
            gate.dispatch()
            await asyncio.sleep(0.2)
            assert len(state['seen']) == 5 and gate.counts['new'] == 5 and gate.counts['cont'] == 0
            state['release'].set()
            assert all(r.status == 200 for r in await asyncio.gather(*firsts))
            await asyncio.sleep(0.1)
            assert len(gate.firsts[conv_key(first_turn('T'))]) == 5        # 5 finished first turns await a turn 2
            gate.room = 0                                                   # live turns need no room
            seconds = await asyncio.gather(*[post(second_turn('T', f'answer {i}')) for i in range(5)])
            assert all(r.status == 200 for r in seconds) and gate.counts['cont'] == 5
            keys = {conversation_key(second_turn('T', f'answer {i}'))[0] for i in range(5)}
            assert keys <= set(gate.convs) and len(keys) == 5 and not gate.firsts
            assert all(gate.convs[k]['tokens'] == 10 for k in keys)
            # a third turn of repeat 0 stays on its conversation (its first model turn is unchanged)
            third = second_turn('T', 'answer 0')
            third['messages'] += [{'role': 'assistant', 'content': 'more'}, {'role': 'user', 'content': 'go on'}]
            assert conversation_key(third)[0] == conversation_key(second_turn('T', 'answer 0'))[0]
            assert (await post(third)).status == 200 and gate.counts['cont'] == 6
            # a conversation header keys the conversation by itself: two repeats' second turns with the same messages
            # but different ids are two conversations
            gate.room = 1
            h1 = await post(second_turn('U', 'same'), {'X-Conversation-Id': 'u-1'})
            assert h1.status == 200 and gate.counts['new'] == 6
            gate.room = 0
            h2 = post(second_turn('U', 'same'), {'X-Conversation-Id': 'u-2'})
            await asyncio.sleep(0.3)
            assert gate.queued('new') == 1                                  # a new conversation: waits for room
            gate.room = 1
            gate.dispatch()
            assert (await h2).status == 200 and gate.counts['new'] == 7
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_conversation_identity_keeps_tool_call_ids_and_ignores_reasoning():
    a = second_turn('same task', 'same text')
    b = second_turn('same task', 'same text')
    b['messages'][2]['tool_calls'][0]['id'] = 'another-call'
    assert conversation_key(a)[0] != conversation_key(b)[0]
    a['messages'][2]['reasoning_content'] = 'thinking that clients may drop'
    assert conversation_key(a)[0] == conversation_key(second_turn('same task', 'same text'))[0]


def test_explicit_conversation_does_not_claim_headerless_first_turn():
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        state['release'].set()
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=8, tick=3600, policy={'agent_tokens_prior': 1024})
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        async with aiohttp.ClientSession() as s:
            gate.room = 1
            first = await s.post(gbase + '/v1/chat/completions', json=first_turn('T'))
            assert first.status == 200
            await first.read()
            await asyncio.sleep(0.1)
            gate.room = 0
            explicit = asyncio.create_task(s.post(gbase + '/v1/chat/completions', json=second_turn('T', 'a'),
                                                  headers={'X-Conversation-Id': 'explicit'}))
            await asyncio.sleep(0.2)
            assert gate.queued('new') == 1 and gate.counts['cont'] == 0
            assert len(gate.firsts[conv_key(first_turn('T'))]) == 1
            gate.room = 1
            gate.dispatch()
            response = await explicit
            assert response.status == 200
            await response.read()
            gate.room = 0
            response = await s.post(gbase + '/v1/chat/completions', json=second_turn('T', 'a'),
                                    headers={'X-Conversation-Id': 'explicit'})
            assert response.status == 200 and gate.counts['cont'] == 1
            await response.read()
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())


def test_user_only_prefix_is_stable_after_first_answer():
    initial = first_turn('task')
    initial['messages'].pop(0)
    later = second_turn('task', 'answer')
    later['messages'].pop(0)
    assert conversation_key(initial)[1] == conversation_key(later)[1]


def test_stale_idle_conversations_do_not_count_and_idle_engine_admits():
    """c30-gemma26 (#378): 112 "live" conversations of finished or killed attempts (38M idle tokens against a 3.9M KV)
    held room at 0 while the engine sat at KV ~1%. Idle past idle_count_seconds: not in the working set (still live
    for the TTL); and an under-used engine admits at least underused_room_min per tick whatever the idle set says."""
    snap = lambda kv=0.01, wait=0: {'kv': kv, 'wait': wait, 'run': 2, 'gen': 0, 'preempt': 0}
    gate = Gate('http://unused', capacity=3_894_253, block=16)
    for i in range(112):
        gate.convs[f'c{i}'] = {'tokens': 340_000, 'last': 1000.0, 'inflight': 0, 'multi': True, 'engine': 0}
    gate.observe(snap(), 1000.0)
    gate.observe(snap(), 1005.0)
    # all recent: counted (38M idle), but the engine is idle -> room anyway (old rule: 0, over the tier ceiling)
    assert gate.status['idle_tokens'] > 30_000_000 and gate.status['underused'] and gate.room >= 2
    gate.observe(snap(kv=0.5), 1010.0)                     # busy engine, idle set over target: holds
    assert gate.room == 0 and not gate.status['underused']
    gate.observe(snap(kv=0.5), 1000.0 + 121)               # 2 min later: the stale ones no longer count
    assert gate.status['idle_tokens'] == 0 and gate.status['stale_conversations'] == 112 and gate.room > 0
    assert len(gate.convs) == 112                          # still live (their engine, turn priority) until the TTL
    gate.observe(snap(kv=0.5), 1000.0 + gate.idle_ttl + 1)
    assert not gate.convs
    # real pressure still holds: in-flight KV high, or vLLM's queue non-empty
    for s in (snap(kv=0.95), snap(kv=0.2, wait=5)):
        g = Gate('http://unused', capacity=3_894_253, block=16)
        g.observe(s, 1000.0)
        g.observe(s, 1005.0)
        g.observe(s, 1010.0)
        assert g.engines[0].blocked() or g.room == 0                # a cause/brake, or no room


def test_conversation_of_a_client_that_left_is_dropped_and_cut_turns_use_the_text_estimate():
    """#378: a turn whose client left mid-stream (killed attempt) drops its conversation at once; a turn that ended
    without usage records the prompt estimated from its text, not request bytes / 3, bounded by max_model_len."""
    gate = Gate('http://unused', capacity=1_000_000, block=16)
    e = gate.engines[0]
    e.max_len = 131_072
    client = gate.clients.setdefault('pi', Client())

    def admit(key, est, status, usage=None, events=0):
        r = Request(None, key, True, 'pi', 0, None)
        r.est, r.engine = est, e
        r.stats.update(status=status, usage=usage, events=events)
        conv = gate.convs.setdefault(key, {'tokens': 0, 'last': 0, 'inflight': 0, 'multi': True})
        conv['inflight'] += 1
        e.inflight += 1
        client.inflight += 1
        e.running.add(r)
        client.running.add(r)
        gate.release(r, client, 'cont', 4_500_000)
    admit('ok', 20_000, 200, {'total_tokens': 21_000, 'completion_tokens': 900})
    assert gate.convs['ok']['tokens'] == 21_000
    admit('cut', 20_000, 200, None, events=300)            # ended without usage: the estimate, not 1.5M
    assert gate.convs['cut']['tokens'] == 20_300
    admit('huge', 400_000, 200, None)                      # never above the served context
    assert gate.convs['huge']['tokens'] == 131_072
    for status in ('client_gone', 'client_reset'):
        admit(status, 20_000, status)
        assert gate.convs[status]['gone'] and gate.convs[status]['tokens'] == 20_000   # out of the working set, kept
    assert gate.counts['conversation_dropped'] == 2 and e.inflight == 0 and client.inflight == 0
    for dt in (0, 1):
        gate.observe({'kv': 0.5, 'wait': 0, 'run': 1, 'gen': 0, 'preempt': 0}, time.time() + dt)
    assert abs(gate.status['idle_tokens'] - (21_000 + 20_300 + 131_072)) < 48
    assert gate.status['dropped_conversations'] == 2 and gate.full_status()['live_conversations'] == 3
    # a client that timed out resends the turn: it is a live turn of its own conversation again
    assert conversation_live(gate, 'client_gone') and gate.counts['conversation_revived'] == 1
    assert gate.usage(gate.clients['pi']) >= 20_000          # it runs with its context, not 0
    gate.firsts['p'] = __import__('collections').deque(['client_reset'])
    gate.claim('p', 'other')                                 # another repeat's second turn never takes a tombstone
    assert 'other' not in gate.convs and 'client_reset' in gate.convs


def conversation_live(gate, key):
    c = gate.clients['pi']
    r = Request(asyncio.get_event_loop_policy().new_event_loop().create_future(), key, True, 'pi', 0, None)
    c.cont.append(r)
    gate.engines[0].room = 0
    gate.dispatch()
    return r.fut.done() and r.fut.result() == 'cont' and not gate.convs[key].get('gone')


def test_killed_agent_mid_stream_drops_its_conversation():
    """End to end: an agent's streamed turn whose client is killed mid-request leaves no live conversation behind."""
    async def main():
        state = {'kv': 0.1, 'seen': [], 'release': asyncio.Event()}
        up, base = await serve(fake_vllm(state))
        gate = Gate(base, max_inflight=4, tick=3600)
        g, gbase = await serve(gate.app())
        await asyncio.sleep(0.2)
        gate.room = 5
        async with aiohttp.ClientSession() as s:      # turn 2 of a live conversation completes: it is tracked
            state['release'].set()
            async with s.post(gbase + '/v1/chat/completions', json=body('A', 'a1', stream=True)) as r:
                await r.read()
        assert len(gate.convs) == 1 and gate.counts.get('conversation_dropped', 0) == 0
        state['release'].clear()
        s = aiohttp.ClientSession()
        task = asyncio.create_task(s.post(gbase + '/v1/chat/completions', json=body('A', 'a2', stream=True)))
        await asyncio.sleep(0.3)
        assert gate.inflight == 1
        task.cancel()                                   # the agent is killed mid-turn
        await s.close()
        for _ in range(30):
            await asyncio.sleep(0.1)
            if not gate.inflight:
                break
        assert gate.inflight == 0 and gate.counts['conversation_dropped'] == 1
        assert all(c.get('gone') for c in gate.convs.values())   # tombstone only: nothing counts
        await g.cleanup()
        await up.cleanup()
    asyncio.run(main())

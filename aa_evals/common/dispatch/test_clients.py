"""#344 wiring in the single-turn clients: SciCode chain slot + chain cap, HLE lane order + request cap."""
import asyncio
import json
import time
import types
from pathlib import Path

import pytest

pytest.importorskip('httpx')

from aa_evals.common.dispatch import dispatcher as dsp
from aa_evals.common.dispatch.lane import read_json


def init_dispatch(tmp_path, suites):
    t = tmp_path / 'durations.json'
    t.write_text(json.dumps({'name': 't', 'models': ['a'], 'suites': suites}))
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '2', '--durations', str(t)])
    return d


def grant(d, lane, n):
    p = d / 'lanes' / lane / 'control.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'max_starts': n, 'epoch': time.time()}))


# ------------------------------------------------------------------------------------------- SciCode

def scicode_runner(tmp_path, d, cap, gen_sec, monkeypatch):
    from aa_evals.scicode import run as sc
    R = sc.Runner.__new__(sc.Runner)
    R.a = types.SimpleNamespace(dispatch=str(d) if d else None, chain_cap_sec=cap)
    R.run_dir = tmp_path / 'sc'
    (R.run_dir / 'chains').mkdir(parents=True)
    (R.run_dir / 'transcripts').mkdir()
    R.items_path = R.run_dir / 'items.jsonl'
    R.cfg = {'code_sha': 'x', 'generation': {'context_window': 1000}}
    R.with_bg = False
    R.lock = asyncio.Lock()
    R.final, R.done = {}, 0
    R.ds = types.SimpleNamespace(excluded_steps=set(), name='t')
    R.lane, R.pred = None, None
    if d:
        from aa_evals.common.dispatch.lane import open_dispatch
        R.lane, R.pred, _ = open_dispatch(d, 'scicode', R.run_dir, poll=0.05)
    R.chain_cap = cap
    calls = []

    async def generate(client, http, budget, prompt):
        calls.append(time.time())
        await asyncio.sleep(gen_sec)
        return {'status': 'ok', 'content': '```python\ndef f(): return 1\n```', 'reasoning': '',
                'finish_reason': 'stop', 'usage': {'completion_tokens': 5}, 'max_tokens': 10, 'latency_s': gen_sec}

    async def grade(prob, k, codes):
        return {'pass': True, 'per_env': {}}
    R.generate, R.grade = generate, grade
    monkeypatch.setattr(sc.prompts, 'build_prompt', lambda prob, k, codes, bg: f'step {k}')
    return R, calls


PROB = {'problem_id': '7', 'sub_steps': [{'step_number': '7.1'}, {'step_number': '7.2'}, {'step_number': '7.3'}]}


def test_scicode_chain_cap_scores_produced_steps(tmp_path, monkeypatch):
    R, calls = scicode_runner(tmp_path, None, cap=0.25, gen_sec=0.15, monkeypatch=monkeypatch)
    asyncio.run(R.chain(None, None, None, PROB, 0))
    st = json.loads((R.run_dir / 'chains' / '7.r0.json').read_text())
    # 7.1 generated (0.15 s), 7.2 cut by the 0.25 s cap mid-request, 7.3 never generated
    assert st['steps']['7.1']['gen_status'] == 'ok'
    assert [st['steps'][s]['finish_reason'] for s in ('7.2', '7.3')] == ['chain_cap', 'chain_cap']
    rows = {r['item_id']: r for r in map(json.loads, open(R.items_path))}
    assert rows['7.1']['score'] == 1 and rows['7.2']['score'] == 0 and rows['7.3']['status'] == 'model_error'
    assert len(calls) == 2
    assert not R.to_generate(PROB, 0)   # capped steps are final: a resume does not regenerate them


def test_scicode_chain_holds_one_lane_slot(tmp_path, monkeypatch):
    d = init_dispatch(tmp_path, {'scicode': {'7': {'a': 100}, '8': {'a': 900}}})
    R, calls = scicode_runner(tmp_path, d, cap=None, gen_sec=0.05, monkeypatch=monkeypatch)
    p8 = {**PROB, 'problem_id': '8', 'sub_steps': [{'step_number': '8.1'}]}

    async def main():
        t = asyncio.gather(R.chain(None, None, None, PROB, 0), R.chain(None, None, None, p8, 0))
        await asyncio.sleep(0.1)          # both chains wait: the allowance is 0 until the dispatcher grants
        grant(d, 'scicode.sc', 2)   # two chain starts, longest first; each holds its slot while it generates
        R.lane.refresh()
        await t
    asyncio.run(main())
    R.lane.close()
    ev = [json.loads(x) for x in open(d / 'lanes' / 'scicode.sc' / 'events.jsonl')]
    starts = [e['key'] for e in ev if e['ev'] == 'start']
    assert starts == ['8.r0', '7.r0']                   # longest predicted chain first
    assert max(e['weight'] for e in ev) == 1 and len(calls) == 4


# ------------------------------------------------------------------------------------------- HLE

def hle_runner(tmp_path, d, cap, latency, monkeypatch):
    from aa_evals.hle import run as hr

    class Model:
        def __init__(self):
            self.order = []

        async def chat(self, body):
            self.order.append(body['messages'][-1]['content'])
            await asyncio.sleep(latency)
            return ({'id': 'x', 'choices': [{'message': {'content': 'Answer: 1'}, 'finish_reason': 'stop'}],
                     'usage': {'completion_tokens': 3}}, [])

        async def close(self):
            pass
    a = types.SimpleNamespace(concurrency=4, judge_concurrency=1, dispatch=str(d) if d else None, request_cap_sec=cap,
                              system_prompt=None, text_only=True, system_role='system', stage='generate',
                              progress_every=1000, cutoff_min=None)
    items = [{'id': i, 'question': f'q{i}', 'image': '', 'answer': '1', 'answer_type': 'exactMatch',
              'category': 'x', 'partition': 'p'} for i in ('q1', 'q2', 'q3')]
    monkeypatch.setattr(hr.prompts, 'build_messages', lambda item, **kw: [{'role': 'user', 'content': item['id']}])
    R = hr.Runner(a, items, Model(), None, {}, {}, tmp_path / 'hle')
    (tmp_path / 'hle' / 'transcripts').mkdir(parents=True, exist_ok=True)
    return R


def test_hle_request_cap_is_model_outcome(tmp_path, monkeypatch):
    R = hle_runner(tmp_path, None, cap=0.05, latency=0.3, monkeypatch=monkeypatch)
    out = asyncio.run(R.generate(R.items['q1'], 0))
    assert out['status'] == 'model_error' and out['finish_reason'] == 'request_cap'
    R.a.stage = 'all'   # graded: a capped request is the model's zero, never pending or infra
    row = R.row(R.items['q1'], {'repeat': 0, 'generation': out, 'judgment': None})
    assert row['score'] == 0 and row['status'] == 'model_error'


def test_hle_lane_longest_first(tmp_path, monkeypatch):
    d = init_dispatch(tmp_path, {'hle': {'q1': {'a': 10}, 'q2': {'a': 500}, 'q3': {'a': 50}}})
    R = hle_runner(tmp_path, d, cap=None, latency=0.02, monkeypatch=monkeypatch)

    async def main():
        t = asyncio.create_task(R.run([('q1', 0), ('q2', 0), ('q3', 0)]))
        await asyncio.sleep(0.2)
        R.lane.flush()
        assert read_json(d / 'lanes' / 'hle.hle' / 'lane.json')['n_pending'] == 3   # announced, nothing granted
        grant(d, 'hle.hle', 3)
        R.lane.refresh()
        return await t
    rows = asyncio.run(main())
    assert R.model.order == ['q2', 'q3', 'q1'] and len(rows) == 3
    st = read_json(d / 'lanes' / 'hle.hle' / 'lane.json')
    assert st['closed'] and st['in_flight'] == 0 and st['done'] == 3


def test_scicode_chain_cap_counts_time_across_resumes(tmp_path, monkeypatch):
    R, calls = scicode_runner(tmp_path, None, cap=0.3, gen_sec=0.01, monkeypatch=monkeypatch)
    # a chain resumed after an infra error at 7.2: 0.3 s of its generation time were already spent
    (R.run_dir / 'chains' / '7.r0.json').write_text(json.dumps({'problem_id': '7', 'repeat': 0, 'chain_gen_sec': 0.3,
        'steps': {'7.1': {'gen_status': 'ok', 'code': 'x', 'grade': {'pass': True, 'per_env': {}}},
                  '7.2': {'gen_status': 'infra_error'}}}))
    R.final = {('7.1', 0): 'ok'}
    asyncio.run(R.chain(None, None, None, PROB, 0))
    st = json.loads((R.run_dir / 'chains' / '7.r0.json').read_text())
    assert not calls and [st['steps'][s]['finish_reason'] for s in ('7.2', '7.3')] == ['chain_cap', 'chain_cap']


def test_scicode_lane_closes_before_cpu_grading(tmp_path, monkeypatch):
    """#531 B1: once every chain has generated, the lane is closed (the dispatcher sees it done, nothing in flight)
    while the last steps' CPU tests still run, so the server can be released; the rows still land after grading."""
    from aa_evals.scicode import run as sc
    d = init_dispatch(tmp_path, {'scicode': {'7': {'a': 100}}})
    R, calls = scicode_runner(tmp_path, d, cap=None, gen_sec=0.01, monkeypatch=monkeypatch)
    R.a.request_timeout, R.a.repeats = 10, 1
    R.cfg['model'] = {'spec': 'x'}

    class Client:
        def __init__(self, *a, **k):
            pass

        async def close(self):
            pass
    monkeypatch.setattr(sc, 'ChatClient', Client)
    monkeypatch.setattr(sc, 'resolve_now', lambda spec, wait: None)
    monkeypatch.setattr(sc, 'Budget', lambda ctx, ep: types.SimpleNamespace(mode=None))
    seen = {}

    async def main():
        release = asyncio.Event()

        async def grade(prob, k, codes):
            if k == len(prob['sub_steps']) - 1:   # the last step's tests hang until released
                seen['lane'] = read_json(d / 'lanes' / 'scicode.sc' / 'lane.json')
                await release.wait()
            return {'pass': True, 'per_env': {}}
        R.grade = grade
        t = asyncio.create_task(R.run([PROB]))
        await asyncio.sleep(0.05)
        grant(d, 'scicode.sc', 1)
        R.lane.refresh()
        for _ in range(200):
            if 'lane' in seen and R.lane.closed:
                break
            await asyncio.sleep(0.02)
        seen['closed_while_grading'] = R.lane.closed and not t.done()
        seen['lane_after_close'] = read_json(d / 'lanes' / 'scicode.sc' / 'lane.json')
        seen['rows_at_close'] = [json.loads(x)['item_id'] for x in open(R.items_path)] if R.items_path.exists() else []
        reg = read_json(d / 'lanes' / 'scicode.sc' / 'registration.json')
        seen['dispatcher_view'] = dsp.StLane(reg, d / 'lanes' / 'scicode.sc').snapshot(time.time())
        release.set()
        await t
    asyncio.run(main())
    assert len(calls) == 3
    assert seen['closed_while_grading']
    st = seen['lane_after_close']
    assert st['closed'] and st['in_flight'] == 0 and st['n_pending'] == 0
    rows = [json.loads(x) for x in open(R.items_path)]
    assert sorted(r['item_id'] for r in rows) == ['7.1', '7.2', '7.3']
    assert '7.3' not in seen['rows_at_close']                 # its tests were still running when the lane closed
    v = seen['dispatcher_view']
    assert v['done'] and not v['dead'] and v['in_flight'] == 0 and v['n_pending'] == 0


def test_scicode_grading_only_resume_needs_no_server(tmp_path, monkeypatch):
    """A resume whose chains are all generated (only grading left) never resolves the endpoint (#531 review)."""
    from aa_evals.scicode import run as sc
    R, calls = scicode_runner(tmp_path, None, cap=None, gen_sec=0.0, monkeypatch=monkeypatch)
    R.a.request_timeout, R.a.repeats = 10, 1
    R.cfg['model'] = {'spec': 'x'}
    (R.run_dir / 'chains' / '7.r0.json').write_text(json.dumps({'problem_id': '7', 'repeat': 0, 'steps': {
        s: {'gen_status': 'ok', 'code': 'x', 'finish_reason': 'stop'} for s in ('7.1', '7.2', '7.3')}}))

    class Client:
        def __init__(self, *a, **k):
            pass

        async def close(self):
            pass

    def no_server(spec, wait):
        raise AssertionError('resolve_now called on a grading-only resume')
    monkeypatch.setattr(sc, 'ChatClient', Client)
    monkeypatch.setattr(sc, 'resolve_now', no_server)
    asyncio.run(R.run([PROB]))
    assert not calls and sorted(json.loads(x)['item_id'] for x in open(R.items_path)) == ['7.1', '7.2', '7.3']

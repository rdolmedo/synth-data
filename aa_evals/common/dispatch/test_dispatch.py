"""Unit tests of the #344 dispatcher: predictions, lane allowance and order, allocation, a dispatcher tick over a fake
pi_eval queue and a live single-turn lane, the pi_eval claim order and wall cap."""
import asyncio
import json
import math
import threading
import time
from pathlib import Path

import pytest

from aa_evals.common.dispatch import dispatcher as dsp
from aa_evals.common.dispatch import pilane
from aa_evals.common.dispatch.durations import DEFAULT_TABLE, V1_TABLE, Durations, capped_mean
from aa_evals.common.dispatch.lane import Lane, read_json


def table(tmp_path, suites=None, models=('a', 'b', 'c')):
    p = tmp_path / 'durations.json'
    p.write_text(json.dumps({'name': 't', 'models': list(models), 'suites': suites or {
        'tb21': {'long': {'a': 4000, 'b': 1000}, 'mid': {'a': 900, 'b': 900}, 'short': {'a': 100, 'c': 100}},
        'hle': {'q1': {'a': 50, 'b': 200}, 'q2': {'a': 3000}, 'q3': {'b': 10}}}}))
    return p


# ------------------------------------------------------------------------------------------- durations

def test_predict_geomean_own_blend_and_fallbacks(tmp_path):
    d = Durations(table(tmp_path))
    assert d.predict('tb21', 'long') == pytest.approx(2000, rel=1e-3)        # geometric mean of 4000 and 1000
    own = Durations(table(tmp_path), model='a')
    # others: b=1000; blended with own 4000 -> sqrt(4000 * 1000)
    assert own.predict('tb21', 'long') == pytest.approx(math.sqrt(4000 * 1000), rel=1e-3)
    assert own.predict('hle', 'q2') == 3000                                   # only its own time
    assert d.predict('tb21', 'unknown') == pytest.approx(d.suite_median('tb21'))
    assert d.predict('nosuite', 'x') == 600.0
    with pytest.raises(ValueError):
        Durations(table(tmp_path), model='zzz')


@pytest.mark.parametrize('path', [None, V1_TABLE])
def test_shipped_table_loads(path):
    d = Durations(path)
    assert set(d.table['suites']) >= {'tb21', 'tbsci_lite', 'hle', 'scicode', 'omniscience', 'gdppdf', 'gdpval',
                                      'critpt', 'automationbench'}
    for m in d.table['models']:
        Durations(path, model=m)
    assert d.predict('tb21', next(iter(d.table['suites']['tb21']))) > 0


def test_default_table_is_v2_with_v1_items_and_ds():
    d, v1 = Durations(), Durations(V1_TABLE)
    assert DEFAULT_TABLE.name == 'durations_v2.json' and d.table['name'] == 'durations_v2'
    assert 'ds-max' in d.table['models'] and Durations(model='ds-max-m35').model == 'ds-max'
    for s, items in v1.table['suites'].items():          # v1 is the fallback: no item with a v1 duration lost
        assert {i for i, per in items.items() if max(per.values()) > 1.0} <= set(d.table['suites'][s])
    cap = d.table['lognormal']['cap_sec']
    for s, items in d.table['suites'].items():           # E[min(T, cap)] never exceeds the cap
        assert all(0 < d.predict(s, i) <= cap[s] + 0.1 for i in items)


def lognormal_table(tmp_path, **lg):
    p = tmp_path / 'v2.json'
    p.write_text(json.dumps({'name': 't2', 'models': ['a', 'b', 'c'], 'own_weight': 0.75,
                             'lognormal': {'sigma': {'tb21': 0.5}, 'tau': {'tb21': 0.4}, 'cap_sec': {'tb21': 7200}, **lg},
                             'suites': {'tb21': {'x': {'a': 1000, 'b': 4000}, 'huge': {'a': 50000, 'b': 50000}}}}))
    return p


def test_capped_mean():
    assert capped_mean(math.log(100), 0.0, 7200) == pytest.approx(100)
    assert capped_mean(math.log(100), 0.5, None) == pytest.approx(100 * math.exp(0.125))
    assert capped_mean(math.log(100), 0.5, 1e9) == pytest.approx(100 * math.exp(0.125))
    assert capped_mean(math.log(1e6), 0.5, 7200) == pytest.approx(7200, rel=1e-6)      # always past the cap
    assert capped_mean(math.log(7200), 0.5, 7200) < 7200


def test_lognormal_predict(tmp_path):
    p = lognormal_table(tmp_path)
    d = Durations(p)
    # others only: median 2000 (geomean of 1000, 4000), sd sqrt(0.5^2 + 0.4^2)
    assert d.predict('tb21', 'x') == pytest.approx(capped_mean(math.log(2000), math.sqrt(0.41), 7200), abs=0.1)
    own = Durations(p, model='a')   # others = b 4000; own weight 0.75 on a = 1000; sd sqrt(0.25 + (0.25 * 0.4)^2)
    mu = 0.25 * math.log(4000) + 0.75 * math.log(1000)
    assert own.predict('tb21', 'x') == pytest.approx(capped_mean(mu, math.sqrt(0.25 + 0.01), 7200), abs=0.1)
    assert 7190 < d.predict('tb21', 'huge') <= 7200    # latent 50000 s: capped
    assert d.predict('tb21', 'unknown') == pytest.approx(d.suite_median('tb21'), abs=0.1)
    assert d.predict('nosuite', 'x') == 600.0


# ------------------------------------------------------------------------------------------- lane

def test_lane_credits_longest_first_and_no_refill(tmp_path):
    lane = Lane(tmp_path / 'lane', 'hle', 'hle', start_poller=False)
    order = []

    async def one(key, pred):
        async with lane.aslot(key, key, pred):
            order.append(key)
            await asyncio.sleep(0.02)

    async def main():
        tasks = [asyncio.create_task(one(k, p)) for k, p in (('a', 10), ('b', 500), ('c', 50), ('d', 1000), ('e', 5))]
        await asyncio.sleep(0.01)
        assert not order                                  # credit 0: nothing starts
        lane.set_credit(2)
        await asyncio.sleep(0.2)
        assert order == ['d', 'b'] and lane.in_flight == 0   # finished: the slots do NOT go back to this lane
        lane.set_credit(5)
        await asyncio.gather(*tasks)
    asyncio.run(main())
    assert order == ['d', 'b', 'c', 'a', 'e'] and lane.starts == 5
    lane.flush()
    ev = [json.loads(x) for x in open(tmp_path / 'lane' / 'events.jsonl')]
    assert [e['key'] for e in ev if e['ev'] == 'start'] == order


def test_lane_weight_needs_full_credit(tmp_path):
    lane = Lane(tmp_path / 'lane', 'g', 'gdppdf', start_poller=False)
    started = []

    async def main():
        async def one(k, p, w):
            async with lane.aslot(k, k, p, weight=w):
                started.append((k, lane.in_flight))
                await asyncio.sleep(0.02)
        t = [asyncio.create_task(one('heavy', 100, 3)), asyncio.create_task(one('light', 1, 1))]
        await asyncio.sleep(0.02)
        lane.set_credit(2)                        # heavy (3) does not fit 2, and blocks the lighter one behind it
        await asyncio.sleep(0.05)
        assert not started
        lane.set_credit(4)
        await asyncio.gather(*t)
    asyncio.run(main())
    assert [k for k, _ in started] == ['heavy', 'light'] and lane.starts == 4
    assert [p[3] for p in Lane(tmp_path / 'x', 'x', 'x', start_poller=False).pending()] == []


def test_lane_thread_slot_and_publish(tmp_path):
    lane = Lane(tmp_path / 'lane', 'omni', 'omniscience', start_poller=False)
    lane.expect([('x', 'x', 5.0), ('y', 'y', 50.0, 2)])
    st = lane.state()
    assert st['pending'] == [['y', 'y', 50.0, 2], ['x', 'x', 5.0, 1]] and st['in_flight'] == 0
    lane.set_credit(2)
    seen = []

    def work(k, p):
        with lane.slot(k, k, p):
            seen.append(k)
            time.sleep(0.05)
    ts = [threading.Thread(target=work, args=a) for a in (('y', 50.0), ('x', 5.0))]
    for t in ts:
        t.start()
        time.sleep(0.01)
    for t in ts:
        t.join()
    assert seen == ['y', 'x'] and lane.in_flight == 0
    (tmp_path / 'lane' / 'control.json').write_text(json.dumps({'max_starts': 7, 'epoch': 1}))
    lane.refresh()
    assert lane.credit == 7 and read_json(tmp_path / 'lane' / 'lane.json')['credit'] == 7
    (tmp_path / 'lane' / 'control.json').write_text(json.dumps({'max_starts': 3, 'epoch': 2}))
    lane.refresh()
    assert lane.credit == 7                       # credits never shrink


def test_lane_cancel_releases(tmp_path):
    lane = Lane(tmp_path / 'lane', 'h', 'hle', start_poller=False)

    async def main():
        async def hold():
            async with lane.aslot('a', 'a', 1):
                await asyncio.sleep(10)
        t = asyncio.create_task(hold())
        lane.set_credit(1)
        await asyncio.sleep(0.02)
        assert lane.in_flight == 1
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
    asyncio.run(main())
    assert lane.in_flight == 0 and not lane.running


def test_lane_open_per_run_names_and_owner_check(tmp_path, capsys):
    d = tmp_path / 'disp'
    a = Lane.open(d, 'hle', tmp_path / 'run1', start_poller=False)
    b = Lane.open(d, 'hle', tmp_path / 'run2', start_poller=False)
    assert a.name == 'hle.run1' and b.name == 'hle.run2'
    a.flush()
    st = json.loads((d / 'lanes' / 'hle.run1' / 'lane.json').read_text())
    st['pid'] = 1                                 # another live process on this host owns it
    (d / 'lanes' / 'hle.run1' / 'lane.json').write_text(json.dumps(st))
    with pytest.raises(SystemExit):
        Lane.open(d, 'hle', tmp_path / 'run1', start_poller=False)
    st['closed'] = True
    (d / 'lanes' / 'hle.run1' / 'lane.json').write_text(json.dumps(st))
    (d / 'lanes' / 'hle.run1' / 'control.json').write_text(json.dumps({'max_starts': 9}))
    c = Lane.open(d, 'hle', tmp_path / 'run1', start_poller=False)   # resumed: old credit already spent
    assert c.credit == 9 and c.starts == 9


def test_lane_warns_when_starved(tmp_path, capsys, monkeypatch):
    from aa_evals.common.dispatch import lane as L
    lane = Lane(tmp_path / 'lane', 'h', 'hle', start_poller=False)
    monkeypatch.setattr(L, 'CONTROL_STALE', 0.0)

    async def main():
        t = asyncio.create_task(lane.aslot('a', 'a', 1).__aenter__())
        await asyncio.sleep(0.02)
        lane.refresh()
        t.cancel()
    asyncio.run(main())
    assert 'no new start credit' in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- allocation

def test_allocate_strict_order_weights_capacity():
    q = {'tb': [(900, 1), (800, 1), (700, 1)], 'hle': [(1000, 1), (50, 1)], 'pdf': [(850, 2), (10, 1)]}
    room = {'tb': 2, 'hle': 10, 'pdf': 10}
    assert dsp.allocate(3, room, q) == {'tb': 1, 'hle': 1, 'pdf': 0}    # pdf (850, weight 2) waits for 2 free
    assert dsp.allocate(4, room, q) == {'tb': 1, 'hle': 1, 'pdf': 2}
    assert dsp.allocate(6, room, q) == {'tb': 2, 'hle': 2, 'pdf': 2}    # tb capped at its room; then hle 50
    assert dsp.allocate(0, room, q) == {'tb': 0, 'hle': 0, 'pdf': 0}
    assert dsp.allocate(1, {'pdf': 5}, {'pdf': [(9, 3)]}, idle=True) == {'pdf': 3}   # alone when nothing in use


def test_factors_shrink():
    f = dsp.Factors()
    for _ in range(20):
        f.add('hle', 200, 100)
    x = f.factors()
    assert 1.8 < x['hle'] < 2.0 and x['*'] < x['hle']
    assert f.factors({'tb21': [math.log(3)] * 5})['tb21'] > 1.5


def test_iso_epoch_formats():
    from aa_evals.common.harbor_mswea.queue import now as hm_now
    q35_now = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(timespec='seconds')
    for s in (q35_now, hm_now()):
        assert abs(dsp.iso_epoch(s) - time.time()) < 5
    assert dsp.iso_epoch('garbage') is None and dsp.iso_epoch(None) is None


# ------------------------------------------------------------------------------------------- dispatcher

def q35_now():
    """eval_q35 common.now(): what pi_eval's queue and worker heartbeats really contain."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def fake_pi_run(root: Path, items, slots=4, done=(), claims=(), name='tb21run'):
    run = root / name
    for sub in ('queue/claims', 'queue/done', 'queue/retry', 'workers'):
        (run / sub).mkdir(parents=True)
    (run / 'queue' / 'items.txt').write_text(''.join(f'{t} {k}\n' for t, k in items))
    (run / 'config.json').write_text(json.dumps({'worker': {'slots': slots}}))
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': False}))
    hb = q35_now()
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': hb}))
    for k in claims:
        (run / 'queue' / 'claims' / k).mkdir()
        (run / 'queue' / 'claims' / k / 'claim.json').write_text(json.dumps({'claimed_at': hb}))
    for k, sec in done:
        (run / 'queue' / 'done' / f'{k}.json').write_text(json.dumps(
            {'wall_sec': sec, 'finished_at': hb, 'infrastructure_failure': False, 'score': 1.0}))
    return run


def claim(run, k):
    (run / 'queue' / 'claims' / k).mkdir()
    (run / 'queue' / 'claims' / k / 'claim.json').write_text(json.dumps({'claimed_at': q35_now()}))


def test_dispatcher_tick_end_to_end(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '5', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['short', 'mid', 'long'], 2)
    assert order[:2] == [('long', 0), ('long', 1)] and order[-1] == ('short', 1)
    run = fake_pi_run(tmp_path, order, slots=3, claims=['long__a0__r0'], done=[('long__a0__r0', 1800.0)])
    claim(run, 'long__a1__r0')
    pilane.hold(run)
    assert read_json(run / 'control.json')['max_starts'] == 0
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'hlerun', start_poller=False)
    lane.expect([('q2.r0', 'q2', 3000), ('q1.r0', 'q1', 100), ('q3.r0', 'q3', 10)])
    lane.flush()
    D = dsp.Dispatcher(d)
    st = D.tick()
    # pi: 2 starts, 1 in flight, capacity 3; free = 5 - 1 = 4 -> hle q2 (3000), pi mid x2 (900), hle q1 (100)
    assert st['released'] and st['in_flight'] == 1
    assert st['lanes']['tb21run']['grant'] == 2 and st['lanes']['hle.hlerun']['grant'] == 2
    assert read_json(run / 'control.json')['max_starts'] == 4            # 2 claims so far + 2
    assert read_json(d / 'lanes' / 'hle.hlerun' / 'control.json')['max_starts'] == 2
    ev = [json.loads(x) for x in open(d / 'events.jsonl')]
    assert {(e['key'], e['ev']) for e in ev} == {('long__a0__r0', 'start'), ('long__a0__r0', 'end'),
                                                  ('long__a1__r0', 'start')}
    assert 0.9 < st['factors']['tb21'] < 1.0                             # learned from 1800 s vs table 2000 s
    # the next tick: nothing started yet -> the unspent credit counts as in use, no new grant (no overshoot)
    st2 = D.tick()
    assert st2['used'] == 5 and st2['lanes']['tb21run']['grant'] == 0 and st2['lanes']['hle.hlerun']['grant'] == 0
    assert read_json(run / 'control.json')['max_starts'] == 4
    # a restarted dispatcher keeps the credits and does not duplicate events
    st3 = dsp.Dispatcher(d).tick()
    assert st3['used'] == 5 and len(open(d / 'events.jsonl').read().splitlines()) == 3
    dsp.cmd_status(type('A', (), {'dir': str(d)}))


def test_finished_slot_goes_to_longest_across_lanes(tmp_path):
    """Review of #363 (sticky slots): a lane does not refill its own freed slots; the dispatcher hands them to the
    longest pending attempt of any lane, and a budget cut drains in flight."""
    d = tmp_path / 'disp'
    suites = {'omniscience': {str(i): {'a': 30} for i in range(20)}, 'hle': {'x': {'a': 5000}}}
    dsp.main(['init', str(d), '--budget', '4', '--durations', str(table(tmp_path, suites)), '--expect-lanes', '1'])
    D = dsp.Dispatcher(d)
    A = Lane.open(d, 'omniscience', tmp_path / 'a', start_poller=False)
    evs = {i: threading.Event() for i in range(20)}

    def work(i):
        with A.slot(f'a{i}', str(i), 30.0):
            evs[i].wait()
    ths = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(20)]
    for t in ths:
        t.start()
    time.sleep(0.2)
    A.flush()
    D.tick()
    A.refresh()
    time.sleep(0.1)
    assert A.in_flight == 4
    B = Lane.open(d, 'hle', tmp_path / 'b', start_poller=False)
    B.expect([(f'b{i}', 'x', 5000.0) for i in range(4)])
    B.flush()
    got_b = []
    for rnd in range(4):
        k = next(i for i in sorted(evs) if not evs[i].is_set() and f'a{i}' in A.running)
        evs[k].set()
        time.sleep(0.2)
        A.refresh()
        assert A.in_flight == 3 - rnd                  # A did not refill its freed slot
        B.flush()
        st = D.tick()
        got_b.append(st['lanes']['hle.b']['grant'])
        B.refresh()
        B.starts += st['lanes']['hle.b']['grant']      # B's attempts start (simulated)
        B.in_flight += st['lanes']['hle.b']['grant']
        B.expected = dict(list(B.expected.items())[st['lanes']['hle.b']['grant']:])
        B.flush()
    assert got_b == [1, 1, 1, 1]                       # every freed slot went to the longer lane
    # budget cut to 2 with B's 4 in flight: nobody gets credit until in flight drains below 2
    dsp.cmd_budget(type('A', (), {'dir': str(d), 'n': 2}))
    st = D.tick()
    assert st['used'] == 4 and all(v['grant'] == 0 for v in st['lanes'].values())
    for e in evs.values():
        e.set()
    A.set_credit(10 ** 6)
    for t in ths:
        t.join(2)


def test_dispatcher_holds_until_lanes_ready(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '5', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    lane = Lane.open(d, 'hle', run=tmp_path / 'r', start_poller=False)
    lane.expect([('a', 'q1', 10)])
    lane.flush()
    st = dsp.Dispatcher(d).tick()
    assert not st['released'] and st['lanes']['hle.r']['grant'] == 0


def test_release_waits_for_announced_work(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '5', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    a = Lane.open(d, 'hle', run=tmp_path / 'a', start_poller=False)
    a.expect([('k', 'q1', 10)])
    a.flush()
    b = Lane.open(d, 'scicode', run=tmp_path / 'b', start_poller=False)
    b.flush()                                     # registered, still loading its problems
    D = dsp.Dispatcher(d)
    assert not D.tick()['released']
    b.expect([])                                  # nothing to do is an announcement too
    b.flush()
    assert D.tick()['released']


def test_dead_lane_released_and_reregistered(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    lane = Lane.open(d, 'hle', run=tmp_path / 'r', start_poller=False)
    lane.expect([(f'k{i}', 'q1', 10) for i in range(5)])
    lane.flush()
    D = dsp.Dispatcher(d)
    assert D.tick()['used'] == 3
    st = json.loads((d / 'lanes' / 'hle.r' / 'lane.json').read_text())
    st['epoch'] -= 3600                                # the client died an hour ago
    (d / 'lanes' / 'hle.r' / 'lane.json').write_text(json.dumps(st))
    st2 = D.tick()
    assert st2['used'] == 0 and st2['lanes']['hle.r']['dead'] and 'client gone' in (d / 'log.txt').read_text()
    lane2 = Lane.open(d, 'hle', run=tmp_path / 'r', start_poller=False)   # the client is resumed
    lane2.expect([(f'k{i}', 'q1', 10) for i in range(5)])
    lane2.flush()
    st3 = D.tick()
    assert 're-registered' in (d / 'log.txt').read_text() and st3['lanes']['hle.r']['grant'] == 3


# ------------------------------------------------------------------------------------------- pi credit claim

def test_pi_credit_claim(tmp_path):
    run = fake_pi_run(tmp_path, [('a', 0), ('b', 0), ('c', 0)])
    calls = []

    class State:
        def claim(self, worker, avail):
            k = f'x{len(calls)}'
            calls.append(k)
            claim(run, k)
            return k
    s = State()
    pilane.credit_claim(s, run)
    assert s.claim('w', set()) is None                 # no control.json yet: nothing
    pilane.hold(run)
    assert s.claim('w', set()) is None                 # credit 0
    (run / 'control.json').write_text(json.dumps({'max_starts': 2}))
    assert s.claim('w', set()) == 'x0' and s.claim('w', set()) == 'x1' and s.claim('w', set()) is None
    (run / 'queue' / 'dispatch.lock').mkdir()          # another worker inside the critical section
    (run / 'control.json').write_text(json.dumps({'max_starts': 5}))
    assert s.claim('w', set()) is None
    (run / 'queue' / 'dispatch.lock').rmdir()
    assert s.claim('w', set()) == 'x2'


# ------------------------------------------------------------------------------------------- pi_eval wall cap

def test_worker_wall_cap_multiplier():
    pytest.importorskip('harbor')
    from aa_evals.common.pi_eval import worker as W

    class Fake(W.PiEvalWorker):
        def __init__(self, cfg):
            self.config, self.run, self.urls = cfg, Path('/tmp/none'), ['http://x/v1']
    base = {'agent': 'pi', 'protocol': {'agent_timeout_sec': 21600}, 'envclock': True, 'envclock_mult': 1 / 3,
            'backstop_multiplier': 2.0, 'agent_cap_sec': 7200, 'benchmark': 'tb21'}

    def mult(cfg, e):
        e = dict(e)
        c = cfg
        # only the HARBOR_EXTRA_ARGS rewrite matters here: reproduce the tail of env() on a prepared env
        w = Fake(c)
        orig = W.q35.Worker.env
        W.q35.Worker.env = lambda self, k, t, a: dict(e)
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(W.model_input, 'apply_env', lambda *a, **k: None)
                mp.setattr(W.task_agents, 'apply_env', lambda *a, **k: None)
                mp.setattr(W.task_environment, 'overrides_for', lambda *a, **k: {})
                out = w.env('t__a0__r0', 't', 0)
        finally:
            W.q35.Worker.env = orig
        return float(out['HARBOR_EXTRA_ARGS'].split('--agent-timeout-multiplier ')[1].split()[0])
    e = {'HARBOR_EXTRA_ARGS': '--agent-timeout-multiplier 2'}
    assert mult(base, e) == pytest.approx(2 / 3)                          # unchanged: 2 x 7200 s = 4 h backstop
    assert mult({**base, 'agent_wall_cap_sec': 7200}, e) == pytest.approx(1 / 3)   # 7200 s wall
    # an explicit cap above the 4 h backstop replaces it: model time stays free on a slow endpoint (m5l 11-16)
    assert mult({**base, 'agent_wall_cap_sec': 30000}, e) == pytest.approx(30000 / 21600)
    assert mult({**base, 'agent_wall_cap_sec': 43200}, e) == pytest.approx(2.0)


def test_plan_command(tmp_path, capsys):
    assert dsp.simulate([3, 2, 1, 1], 2) == 4
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'suites': {'tb21': {'items': [{'id': 'long', 'repeats': 2}, {'id': 'new', 'repeats': 1}]},
                                           'hle': {'items': [{'id': 'q2', 'repeats': 1}]}}}))
    dsp.main(['plan', str(plan), '--budget', '2', '--durations', str(table(tmp_path)), '--ids-out', str(tmp_path / 'ids')])
    out = capsys.readouterr().out
    assert 'tb21' in out and 'in table    1' in out and 'total 4 attempts' in out
    assert (tmp_path / 'ids' / 'tb21.ids').read_text() == 'long\nnew\n'


def test_suite_name_and_coverage_warning(tmp_path, capsys):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '5', '--durations', str(table(tmp_path))])
    rec, order = pilane.plan(d, 'tb21_pi', 'tb21', ['short', 'long'], 1)   # the suite, not the bench name
    assert order[0] == ('long', 0) and 'coverage_warning' not in rec
    rec, _ = pilane.plan(d, 'tb21_pi', 'tb21_pi', ['short', 'long'], 1)
    assert 'coverage_warning' in rec and 'WARNING (dispatch)' in capsys.readouterr().err
    run = fake_pi_run(tmp_path, [('short', 0), ('long', 0)])
    pilane.register(rec, run)
    st = dsp.Dispatcher(d).tick()
    assert st['lanes']['tb21_pi']['no_prediction_share'] == 1.0 and st['lanes']['tb21_pi']['warning']
    assert 'not in the duration table' in (d / 'log.txt').read_text()


def test_run_default_dispatch_suite_strips_pi():
    pytest.importorskip('harbor')
    from aa_evals.common.pi_eval import run as R
    a = R.build_parser({'bench': 'gdpval_pi'}).parse_args(['--run-id', 'x', '--dispatch', '/x'])
    assert a.dispatch_suite is None and (a.dispatch_suite or a.bench.removesuffix('_pi')) == 'gdpval'


# ------------------------------------------------------------------------------------------- re-review (#363)

def test_pi_lane_names_unique_per_bench_and_owned_by_run(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '5', '--durations', str(table(tmp_path))])
    r1 = fake_pi_run(tmp_path, [('long', 0)], name='a')
    rec, _ = pilane.plan(d, 'tb21.same-id', 'tb21', ['long'], 1)
    pilane.register(rec, r1)
    pilane.register(rec, r1)                                  # a resume of the same run is fine
    with pytest.raises(SystemExit):
        pilane.register(rec, fake_pi_run(tmp_path, [('long', 0)], name='b'))


def test_pi_lane_dead_without_heartbeat(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '4', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['short', 'mid', 'long'], 2)
    run = fake_pi_run(tmp_path, order, slots=4)
    pilane.register(rec, run)
    D = dsp.Dispatcher(d)
    st = D.tick()
    assert st['lanes']['tb21.r']['grant'] == 4
    claim(run, 'long__a0__r0')
    old = json.loads((run / 'workers' / 'w1.json').read_text())
    import datetime
    old['heartbeat'] = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
    (run / 'workers' / 'w1.json').write_text(json.dumps(old))   # every worker gone for an hour
    st = D.tick()
    assert st['lanes']['tb21.r']['dead'] and st['used'] == 0 and st['in_flight'] == 0
    assert read_json(run / 'control.json')['max_starts'] == 1   # unspent credit withdrawn: claims so far


def test_pi_lane_before_first_worker_writes_state(tmp_path):
    """c30/smk1: a Pi lane whose workers have not written a heartbeat yet (and claimed nothing) made 'dead' the
    empty claims set, and every tick failed writing state.json ('Object of type set is not JSON serializable')."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '4', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['short', 'mid', 'long'], 2)
    run = fake_pi_run(tmp_path, order, slots=4)
    (run / 'workers' / 'w1.json').unlink()   # submitted, no worker up yet
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'hlerun', start_poller=False)
    lane.expect([('q1.r0', 'q1', 100)])
    lane.flush()
    st = dsp.Dispatcher(d).tick()
    assert st['lanes']['tb21.r']['dead'] is False
    assert read_json(d / 'state.json')['lanes']['tb21.r']['dead'] is False


def test_release_survives_restart_and_skips_dead_lanes(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    lane = Lane.open(d, 'hle', run=tmp_path / 'r', start_poller=False)
    lane.expect([('k0', 'q1', 10)])
    lane.flush()
    assert dsp.Dispatcher(d).tick()['released']
    st = json.loads((d / 'lanes' / 'hle.r' / 'lane.json').read_text())
    st['epoch'] -= 3600
    (d / 'lanes' / 'hle.r' / 'lane.json').write_text(json.dumps(st))
    lane2 = Lane.open(d, 'omniscience', run=tmp_path / 'o', start_poller=False)
    lane2.expect([('o0', '1', 10)])
    lane2.flush()
    D = dsp.Dispatcher(d)                                       # restart: released at once, the dead lane ignored
    assert D.released and D.tick()['lanes']['omniscience.o']['grant'] == 1


def test_lane_open_after_killed_client_same_host(tmp_path):
    import subprocess
    d = tmp_path / 'disp'
    p = subprocess.Popen(['sleep', '0'])
    p.wait()                                                    # a pid that is gone
    lane_dir = d / 'lanes' / 'hle.r'
    lane_dir.mkdir(parents=True)
    import socket as _s
    (lane_dir / 'lane.json').write_text(json.dumps({'epoch': time.time(), 'closed': False, 'pid': p.pid,
                                                    'host': _s.gethostname()}))
    t0 = time.time()
    Lane.open(d, 'hle', run=tmp_path / 'r', start_poller=False)  # no 120 s wait, no exit
    assert time.time() - t0 < 5


def test_one_dispatcher_per_dir(tmp_path):
    import socket as _s
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path))])
    (d / 'dispatcher.lock').write_text(json.dumps({'host': _s.gethostname(), 'pid': 1}))   # init: always alive
    with pytest.raises(SystemExit):
        dsp.Dispatcher(d).lock()
    import subprocess
    p = subprocess.Popen(['sleep', '0'])
    p.wait()
    (d / 'dispatcher.lock').write_text(json.dumps({'host': _s.gethostname(), 'pid': p.pid}))   # dead: taken over
    dsp.Dispatcher(d).lock()
    assert read_json(d / 'dispatcher.lock')['pid'] == __import__('os').getpid()


def test_lock_exclusive_lost_lock_and_init_force(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path))])
    D = dsp.Dispatcher(d)
    D.lock()
    assert D.owns_lock()
    (d / 'dispatcher.lock').write_text(json.dumps({'host': 'elsewhere', 'pid': 1}))   # a newer dispatcher took over
    (d / 'STOP').unlink(missing_ok=True)
    D.loop()                                    # returns at once: it no longer owns the lock
    assert 'lost dispatcher.lock' in (d / 'log.txt').read_text()
    (d / 'state.json').write_text(json.dumps({'released': True, 'epoch': 0}))
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--force'])
    assert not (d / 'state.json').exists() and not dsp.Dispatcher(d).released


# ------------------------------------------------------------------------------------------- round 2

def test_drain_keeps_capacity_of_other_snapshot(tmp_path):
    """c30 live run: STOP_WORKERS 'drain <sha>' stops only the oversized workers on that snapshot; the right-sized
    ones on another snapshot keep claiming, so the lane is not paused and its capacity is their slots."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '40', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['short', 'mid', 'long'], 10)
    run = fake_pi_run(tmp_path, order, slots=24)
    (run / 'workers' / 'w1.json').unlink()
    jobs = []
    for cluster, snap, slots in (('100', '191fe99b1267', 24), ('200', 'abcdef123456', 8)):
        sub = run / 'jobs' / f'worker-{cluster}.sub'
        sub.parent.mkdir(exist_ok=True)
        sub.write_text(f'universe = vanilla\narguments = /x/code/{snap}/aa_evals/common/pi_eval/worker_job.sh {run} {slots}\n')
        jobs.append({'cluster': cluster, 'n': 2, 'submit_file': str(sub)})
        for proc in (0, 1):
            (run / 'workers' / f'{cluster}.{proc}-h-1.json').write_text(json.dumps({'heartbeat': q35_now()}))
    (run / 'jobs.json').write_text(json.dumps(jobs))
    pilane.register(rec, run)
    lane = dsp.PiLane(json.loads((d / 'lanes' / 'tb21.r' / 'registration.json').read_text()))
    assert lane.snapshot(time.time())['capacity'] == 2 * 24 + 2 * 8   # per-job slots, not the config's
    (run / 'STOP_WORKERS').write_text('drain 191fe99b1267\n')
    s = lane.snapshot(time.time())
    assert not s['paused'] and s['capacity'] == 16 and s['draining_workers'] == 2
    (run / 'STOP_WORKERS').write_text('stop\n')
    s = lane.snapshot(time.time())
    assert s['paused'] and s['capacity'] == 0


def test_tick_gate_backoff_and_reservation(tmp_path, monkeypatch):
    """Live tick: gate status read from the registry; while it holds, no credit; once clear, the gate-paused Pi lane
    takes its longest-first share (held for its workers) before the single-turn lane."""
    import http.server
    import threading as th
    from aa_evals.common.harbor_mswea import endpoint
    state = {'queued_new': 5}
    other = {'queued_new': 7}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(other if self.path.startswith('/r2') else state).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(('127.0.0.1', 0), H)
    th.Thread(target=srv.serve_forever, daemon=True).start()
    reg = tmp_path / 'serving' / 'm'
    reg.mkdir(parents=True)
    (reg / 'endpoints.json').write_text(json.dumps(
        {'gate_status_urls': [f'http://127.0.0.1:{srv.server_port}/gate/status',
                              f'http://127.0.0.1:{srv.server_port}/r2/gate/status']}))
    monkeypatch.setattr(endpoint, 'SERVING_ROOT', tmp_path / 'serving')
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '6', '--durations', str(table(tmp_path)), '--expect-lanes', '2',
              '--registry', 'm'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['long', 'mid'], 2)
    run = fake_pi_run(tmp_path, order, slots=4)
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': True,
                                                              'reason': 'gate holds 5 new conversation(s)'}))
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'h', start_poller=False)
    lane.expect([(f'q{i}', 'q1', 100) for i in range(10)])
    lane.flush()
    D = dsp.Dispatcher(d)
    st = D.tick()
    assert st['gate_hold'] and st['gate_queued'] == 5 and all(v['grant'] == 0 for v in st['lanes'].values())
    state['queued_new'] = 0   # one replica clear, the other still holding 7: not every replica holds -> grants
    st = D.tick()
    assert not st['gate_hold'] and not st['st_hold'] and st['lanes']['tb21.r']['grant'] == 4
    assert st['lanes']['hle.h']['grant'] == 2
    assert st['lanes']['tb21.r']['gate_paused'] and read_json(run / 'control.json')['max_starts'] == 4
    # #378: its workers claim again (replica 2 holds, new attempts run on replica 1): not gate-paused
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': False, 'gate_held_replicas': [1]}))
    st = D.tick()
    assert st['lanes']['tb21.r']['gate_partial'] and not st['lanes']['tb21.r']['gate_paused']
    srv.shutdown()


def test_no_gate_registry_is_unchanged(tmp_path, monkeypatch):
    from aa_evals.common.harbor_mswea import endpoint
    (tmp_path / 'serving' / 'm').mkdir(parents=True)
    (tmp_path / 'serving' / 'm' / 'endpoints.json').write_text(json.dumps({'urls': ['http://x/v1']}))
    monkeypatch.setattr(endpoint, 'SERVING_ROOT', tmp_path / 'serving')
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--registry', 'm'])
    assert dsp.Dispatcher(d).gate_queued() is None


def test_lane_exact_credit_withdrawal(tmp_path):
    """#378: control.json `exact` lowers the credit (single-turn hold); without it credits only grow (old dispatchers).
    The lane says it honours `exact` (lane.json shrinkable)."""
    lane = Lane(tmp_path / 'lane', 'sc', 'scicode', start_poller=False)
    lane.flush()
    assert read_json(tmp_path / 'lane' / 'lane.json')['shrinkable'] is True
    (tmp_path / 'lane' / 'control.json').write_text(json.dumps({'max_starts': 10, 'epoch': 1}))
    lane.refresh()
    (tmp_path / 'lane' / 'control.json').write_text(json.dumps({'max_starts': 4, 'epoch': 2}))
    lane.refresh()
    assert lane.credit == 10                                     # not exact: the larger stays
    (tmp_path / 'lane' / 'control.json').write_text(json.dumps({'max_starts': 4, 'epoch': 3, 'exact': True}))
    lane.refresh()
    assert lane.credit == 4
    started = []
    with lane.slot('a', 'a', 5):
        started.append('a')
    lane.starts = 5                                              # started more than the withdrawn credit: no new start
    w = lane._enqueue('b', 'b', 1, 1, lambda: started.append('b'))
    assert not w.granted and started == ['a']
    lane.set_credit(6, exact=True)
    assert w.granted and started == ['a', 'b']


def st_views(**pi):
    v = {'pi': {'in_flight': 0, 'starts': 2, 'credit': 2, 'capacity': 0, 'gate_paused': True, 'worker_capacity': 8,
                'dead': False, 'withdrawable': True, 'kind': 'pi', 'unspent_since': None, 'pending': [(900, 1)] * 6},
         'sc': {'in_flight': 30, 'starts': 40, 'credit': 46, 'capacity': 10 ** 6, 'gate_paused': False, 'kind': 'lane',
                'worker_capacity': 10 ** 6, 'dead': False, 'withdrawable': False, 'unspent_since': None,
                'shrinkable': True, 'pending': [(1000, 1)] * 20},
         'old': {'in_flight': 5, 'starts': 5, 'credit': 9, 'capacity': 10 ** 6, 'gate_paused': False, 'kind': 'lane',
                 'worker_capacity': 10 ** 6, 'dead': False, 'withdrawable': False, 'unspent_since': None,
                 'pending': [(2000, 1)] * 5}}
    v['pi'].update(pi)
    return v


def test_single_turn_hold():
    """#378: gate holding and a pi lane gate-paused: unspent credit of shrinkable single-turn lanes goes back at once,
    no single-turn lane gets credit (old clients keep theirs: they would not honour a lower one)."""
    grants, withdraw, info = dsp.decide(st_views(), 60, 3, 0.0)
    assert info['st_hold'] and info['gate_hold'] and withdraw == {'sc': 40} and not any(grants.values())
    assert info['used']['sc'] == 30 and info['used']['old'] == 5 + 4
    # gate clear (the pi lane's workers have not re-read it yet): no hold; grants go longest first again
    grants, withdraw, info = dsp.decide(st_views(), 60, 0, 0.0)
    assert not info['st_hold'] and not withdraw and grants['old'] == 1 and grants['sc'] > 0   # (4 covered)
    # a pi lane that can claim (not gate-paused), one with nothing pending, or a dead one: no hold
    for pi in ({'gate_paused': False, 'capacity': 8}, {'pending': []}, {'dead': True}):
        assert not dsp.decide(st_views(**pi), 60, 3, 0.0)[2]['st_hold']
    # switched off (--single-turn-hold off) or no gate signal: round 2
    assert not dsp.decide(st_views(), 60, 3, 0.0, st_hold_on=False)[2]['st_hold']
    assert not dsp.decide(st_views(), 60, None, 0.0)[2]['st_hold']


def test_single_turn_hold_tick_writes_exact(tmp_path, monkeypatch):
    """Live tick: the withdrawal reaches the client's control.json as an exact (lower) credit and is logged."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '8', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['long', 'mid'], 2)
    run = fake_pi_run(tmp_path, order, slots=4)
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'h', start_poller=False)
    lane.expect([(f'q{i}', 'q1', 100) for i in range(10)])
    lane.flush()
    D = dsp.Dispatcher(d)
    D.tick()
    lane.refresh()
    assert lane.credit == 4                                      # pi 4 (its workers' slots), hle the other 4
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': True, 'reason': 'gate holds 3 new '
                                                                                        'conversation(s)'}))
    monkeypatch.setattr(dsp.Dispatcher, 'gate_queued', lambda self: 3)
    st = D.tick()
    assert st['st_hold'] and read_json(d / 'lanes' / 'hle.h' / 'control.json') == {
        **read_json(d / 'lanes' / 'hle.h' / 'control.json'), 'max_starts': 0, 'exact': True}
    lane.refresh()
    assert lane.credit == 0 and 'single-turn hold on' in (d / 'log.txt').read_text()


def test_single_turn_hold_spares_old_clients(tmp_path, monkeypatch):
    """A client on older lane.py (no `shrinkable` in lane.json) is never sent an exact (lower) credit."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '8', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    rec, order = pilane.plan(d, 'tb21.r', 'tb21', ['long', 'mid'], 2)
    run = fake_pi_run(tmp_path, order, slots=4)
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'h', start_poller=False)
    lane.expect([(f'q{i}', 'q1', 100) for i in range(10)])
    lane.flush()
    st = read_json(d / 'lanes' / 'hle.h' / 'lane.json')
    st.pop('shrinkable')
    (d / 'lanes' / 'hle.h' / 'lane.json').write_text(json.dumps(st))   # as an old client publishes it
    D = dsp.Dispatcher(d)
    D.tick()
    assert read_json(d / 'lanes' / 'hle.h' / 'control.json')['max_starts'] == 4
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': True, 'reason': 'gate holds 3 new '
                                                                                        'conversation(s)'}))
    monkeypatch.setattr(dsp.Dispatcher, 'gate_queued', lambda self: 3)
    s = D.tick()
    ctl = read_json(d / 'lanes' / 'hle.h' / 'control.json')
    assert s['st_hold'] and ctl['max_starts'] == 4 and 'exact' not in ctl and s['lanes']['hle.h']['grant'] == 0


def test_pending_retries_follow_the_claim_order(tmp_path):
    # #545: tied retries (same attempt and retry index) in the queue's own order, as queue_state.claim takes them; the
    # set's iteration order (hash randomization) must not leak into predictions or the reported head
    items = [('zeta', 0), ('alpha', 0), ('mid', 0), ('mid', 1)]
    run = fake_pi_run(tmp_path, items, claims=['zeta__a0__r0', 'alpha__a0__r0', 'mid__a0__r0', 'mid__a1__r0'])
    for k in ('alpha__a0__r1', 'mid__a1__r1', 'zeta__a0__r1', 'mid__a0__r2'):
        (run / 'queue' / 'retry' / f'{k}.json').write_text('{}')
    s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
    assert [k for k, _, _ in s['pending']] == ['zeta__a0__r1', 'alpha__a0__r1', 'mid__a0__r2', 'mid__a1__r1']

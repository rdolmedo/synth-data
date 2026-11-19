"""Lazy first workers (#378, pi_eval --lazy-workers): a run registers with no worker; the dispatcher counts it ready for
the start-up release, gives it one probe slot, and submits its first workers when it first grants it credit."""
import json
import sys
import threading
import types

from aa_evals.common.dispatch import dispatcher as dsp
from aa_evals.common.dispatch import pilane
from aa_evals.common.dispatch.test_autoscale import SyncThread, jobs_json
from aa_evals.common.dispatch.lane import Lane
from aa_evals.common.dispatch.test_dispatch import claim, fake_pi_run, table, q35_now


def lazy_run(tmp_path, d, name='tb21run', workers=2, slots=4):
    rec, order = pilane.plan(d, name, 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=slots, name=name)
    (run / 'workers' / 'w1.json').unlink()                     # no worker yet
    (run / 'config.json').write_text(json.dumps({'worker': {'slots': slots, 'n': workers}, 'lazy_workers': True}))
    pilane.hold(run)
    pilane.register(rec, run)
    return run


def setup(tmp_path, monkeypatch, budget=10, expect=1):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', str(budget), '--durations', str(table(tmp_path)), '--expect-lanes',
              str(expect)])
    added = []

    def submit(run, cfg, n):
        added.append((run.name, n))
        jobs_json(run, age=0)                                    # what pi_eval.submit_workers records
        return '99'
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run', types.SimpleNamespace(submit_workers=submit))
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
    return d, added


def test_lazy_lane_is_ready_reserves_its_first_fleet_and_submits_its_first_workers(tmp_path, monkeypatch):
    d, added = setup(tmp_path, monkeypatch)
    run = lazy_run(tmp_path, d)
    D = dsp.Dispatcher(d)
    st = D.tick()
    assert st['released']                                        # a registered lazy lane counts as ready
    assert st['lanes']['tb21run']['grant'] == 8                  # its first fleet's slots (2 workers x 4) reserved
    assert added == [('tb21run', 2)]                             # the run's worker.n, at the first grant
    assert 'first credit' in (d / 'log.txt').read_text()
    for _ in range(3):
        st = D.tick()
    assert added == [('tb21run', 2)]                             # once
    assert st['lanes']['tb21run']['grant'] == 0                  # no heartbeat yet: no capacity beyond the credit
    assert json.loads((run / 'control.json').read_text())['max_starts'] == 8


def test_a_lazy_lane_reserves_its_first_fleet_in_longest_first_order(tmp_path, monkeypatch):
    # its longer items come first: it takes the budget as eager first workers would have, not just a probe slot
    d, added = setup(tmp_path, monkeypatch, budget=4, expect=2)
    lazy_run(tmp_path, d, name='lazyrun')
    rec, order = pilane.plan(d, 'busyrun', 'tb21', ['short'], 6)
    run = fake_pi_run(tmp_path, order, slots=8, name='busyrun')  # a running lane with a live worker
    jobs_json(run)
    pilane.hold(run)
    pilane.register(rec, run)
    st = dsp.Dispatcher(d).tick()
    assert st['lanes']['lazyrun']['grant'] == 4                  # long and mid items, its 8 reserved slots
    assert st['lanes']['busyrun']['grant'] == 0                  # shorter items wait, as for an eager lane


def test_non_lazy_lane_without_a_worker_still_holds_the_release(tmp_path, monkeypatch):
    d, added = setup(tmp_path, monkeypatch)
    run = lazy_run(tmp_path, d)
    cfg = json.loads((run / 'config.json').read_text())
    cfg['lazy_workers'] = False
    (run / 'config.json').write_text(json.dumps(cfg))
    st = dsp.Dispatcher(d).tick()
    assert not st['released'] and added == []


def test_lazy_by_default_for_new_dispatch_dirs_only(tmp_path):
    from aa_evals.common.pi_eval.run import lazy_workers
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path))])
    assert json.loads((d / 'dispatch.json').read_text())['lazy_workers'] is True
    assert lazy_workers(None, str(d)) is True                    # new dir: lazy by default
    assert lazy_workers(False, str(d)) is False                  # --no-lazy-workers
    old = tmp_path / 'old'
    dsp.main(['init', str(old), '--budget', '10', '--durations', str(table(tmp_path))])
    cfg = json.loads((old / 'dispatch.json').read_text())
    del cfg['lazy_workers']                                      # a dir made before lazy workers (e.g. a live one)
    (old / 'dispatch.json').write_text(json.dumps(cfg))
    assert lazy_workers(None, str(old)) is False                 # stays eager
    assert lazy_workers(True, str(old)) is True                  # unless asked
    assert lazy_workers(None, None) is False                     # no dispatcher: no credit to wait for
    import pytest
    with pytest.raises(SystemExit):
        lazy_workers(True, None)
    off = tmp_path / 'off'
    dsp.main(['init', str(off), '--budget', '10', '--durations', str(table(tmp_path)), '--no-lazy-workers'])
    assert lazy_workers(None, str(off)) is False


def test_runs_without_the_field_are_eager_in_the_dispatcher(tmp_path, monkeypatch):
    d, added = setup(tmp_path, monkeypatch)
    run = lazy_run(tmp_path, d)
    cfg = json.loads((run / 'config.json').read_text())
    del cfg['lazy_workers']                                      # a run made before: never lazy
    (run / 'config.json').write_text(json.dumps(cfg))
    st = dsp.Dispatcher(d).tick()
    assert not st['released'] and added == []


def test_status_shows_lazy_and_a_slow_first_submission_is_not_doubled(tmp_path, monkeypatch):
    d, added = setup(tmp_path, monkeypatch)
    lazy_run(tmp_path, d)
    started = []
    monkeypatch.setattr(threading, 'Thread', lambda target, args, daemon: types.SimpleNamespace(
        start=lambda: started.append(args[0])))               # the submission never returns (a wedged schedd)
    D = dsp.Dispatcher(d)
    st = D.tick()
    assert st['lanes']['tb21run']['lazy'] is True and started == ['tb21run']
    D.lanes['tb21run'].lazy_tried -= dsp.LAZY_RETRY_SEC + 1   # past the retry timer, still in flight
    D.tick()
    assert started == ['tb21run']


# ---- #511 regression (val-sb10b 11-18): the single-turn lanes took the whole budget at launch

def launch(tmp_path, monkeypatch, budget=256):
    """val-sb10b's launch shape: three lazy Pi lanes with launch-small first fleets (LAUNCH_SHARE x budget each:
    4 workers x 16 slots) and two single-turn lanes with hundreds of items; predictions TB2.1 > GDPval > AB > SciCode
    > Omniscience."""
    suites = {'tb21': {'t': {'a': 5000}}, 'gdpval': {'g': {'a': 2000}}, 'ab': {'w': {'a': 900}},
              'scicode': {'s': {'a': 700}}, 'omni': {'o': {'a': 60}}}
    d = tmp_path / 'disp'
    p = tmp_path / 'durations.json'
    p.write_text(json.dumps({'name': 't', 'models': ['a'], 'suites': suites}))
    dsp.main(['init', str(d), '--budget', str(budget), '--durations', str(p), '--expect-lanes', '5'])
    added = []

    def submit(run, cfg, n):
        added.append((run.name, n))
        jobs_json(run, age=0)
        return '99'
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run', types.SimpleNamespace(submit_workers=submit))
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
    runs = {}
    for name, suite, item in (('tb21run', 'tb21', 't'), ('gdprun', 'gdpval', 'g'), ('abrun', 'ab', 'w')):
        rec, order = pilane.plan(d, name, suite, [item], 200)
        run = fake_pi_run(tmp_path, order, slots=16, name=name)
        (run / 'workers' / 'w1.json').unlink()
        (run / 'config.json').write_text(json.dumps({'worker': {'slots': 16, 'n': 4}, 'lazy_workers': True}))
        pilane.hold(run)
        pilane.register(rec, run)
        runs[name] = run
    for suite, item, n in (('scicode', 's', 400), ('omni', 'o', 300)):
        lane = Lane.open(d, suite, run=tmp_path / f'{suite}run', start_poller=False)
        lane.expect([(f'{item}.{i}', item, 1) for i in range(n)])
        lane.flush()
    return d, added, runs


def grants(st):
    return {n.split('.')[0].replace('run', ''): v['grant'] for n, v in st['lanes'].items()}


def test_lazy_pi_lanes_are_not_starved_by_single_turn_lanes_at_launch(tmp_path, monkeypatch):
    d, added, runs = launch(tmp_path, monkeypatch)
    D = dsp.Dispatcher(d)
    st = D.tick()
    # longest first over the reserved first fleets (64 slots each), the single-turn lanes after them; with only the
    # probe slot this was tb21/gdp/ab 1 each and SciCode 253 (val-sb10b: 251/256 in flight after a minute)
    assert grants(st) == {'tb21': 64, 'gdp': 64, 'ab': 64, 'scicode': 64, 'omni': 0}
    assert sorted(added) == [('abrun', 4), ('gdprun', 4), ('tb21run', 4)]   # every first fleet submitted at once
    # while the first workers start (no heartbeat yet) their credit is held: nobody takes those slots
    for _ in range(3):
        st = D.tick()
    assert sum(v['grant'] for v in st['lanes'].values()) == 0 and st['used'] == 256
    assert [json.loads((r / 'control.json').read_text())['max_starts'] for r in runs.values()] == [64, 64, 64]


def lane_of(D, name):
    return next(l for k, l in D.lanes.items() if k.startswith(name))


def test_a_slow_first_start_keeps_its_reservation_through_the_start_up_grace(tmp_path, monkeypatch):
    # reserve_sec (600 s) alone gave the slots back while slow workers were still starting: #511's starvation again
    d, added, runs = launch(tmp_path, monkeypatch)
    D = dsp.Dispatcher(d)
    D.tick()
    D.tick()
    for name, run in runs.items():
        lane_of(D, name).unspent_since -= dsp.RESERVE_SEC + 1   # nothing started for > reserve_sec ...
        jobs_json(run, age=900)                                  # ... but submitted 900 s ago, within ready_timeout
    st = D.tick()
    assert [lane_of(D, n).credit for n in runs] == [64, 64, 64] and grants(st)['scicode'] == 0


def test_a_launch_reservation_whose_workers_never_start_goes_back(tmp_path, monkeypatch):
    d, added, runs = launch(tmp_path, monkeypatch)
    D = dsp.Dispatcher(d)
    D.tick()
    for run in runs.values():
        jobs_json(run, age=dsp.READY_TIMEOUT + 60)               # first workers submitted past the grace, none started
    st = D.tick()
    assert [lane_of(D, n).credit for n in runs] == [0, 0, 0]
    assert grants(st)['scicode'] + grants(st)['omni'] == 192      # the slots go back


def test_a_restarted_dispatcher_trims_a_stale_launch_reservation(tmp_path, monkeypatch):
    # the in-memory reservation state is gone after a restart; the trim reads only files
    d, added, runs = launch(tmp_path, monkeypatch)
    dsp.Dispatcher(d).tick()
    for run in runs.values():
        jobs_json(run, age=dsp.READY_TIMEOUT + 60)
    D2 = dsp.Dispatcher(d)
    D2.tick()
    assert [json.loads((r / 'control.json').read_text())['max_starts'] for r in runs.values()] == [0, 0, 0]


def test_a_partly_started_first_fleet_keeps_only_what_its_live_workers_can_run(tmp_path, monkeypatch):
    d, added, runs = launch(tmp_path, monkeypatch)
    D = dsp.Dispatcher(d)
    D.tick()
    run = runs['tb21run']
    jobs_json(run, age=dsp.READY_TIMEOUT + 60)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': q35_now()}))   # 1 of 4 workers up: 16 slots
    for i in range(5):
        claim(run, f't__a{i}__r0')
    D.tick()
    assert lane_of(D, 'tb21run').credit == 5 + (16 - 5)         # starts + its live workers' free slots


def test_a_lazy_lane_whose_first_submission_keeps_failing_does_not_reserve_again(tmp_path, monkeypatch):
    d, added, runs = launch(tmp_path, monkeypatch)

    def fail(run, cfg, n):
        raise RuntimeError('schedd down')
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run', types.SimpleNamespace(submit_workers=fail))
    D = dsp.Dispatcher(d)
    D.tick()
    D.tick()
    lane = next(l for k, l in D.lanes.items() if k.startswith('tb21run'))
    assert lane.lazy and lane.credit == 64                       # reserved; its submission failed, still lazy
    lane.unspent_since -= dsp.RESERVE_SEC + 1
    D.tick()                                                     # withdrawn
    for _ in range(3):
        st = D.tick()
    assert lane.credit <= dsp.LAZY_PROBE_SLOTS                   # only the probe slot until it starts something

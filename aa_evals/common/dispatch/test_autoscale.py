"""Pi worker growth (#378): decide()'s shortfall, workers_to_add, and the tick that submits workers."""
import json
import sys
import threading
import time
import types

from aa_evals.common.dispatch import dispatcher as dsp
from aa_evals.common.dispatch import pilane
from aa_evals.common.dispatch.test_dispatch import fake_pi_run, table


def view(**kw):
    v = {'in_flight': 0, 'starts': 0, 'credit': 0, 'capacity': 0, 'gate_paused': False, 'worker_capacity': 0,
         'dead': False, 'withdrawable': True, 'unspent_since': None, 'withdrawn': False, 'reserved': False,
         'pending': [], 'scalable': False}
    v.update(kw)
    return v


def test_shortfall_is_what_a_slot_bound_lane_would_also_take():
    views = {'p': view(capacity=2, worker_capacity=2, pending=[(5000, 1)] * 6, scalable=True),
             'h': view(withdrawable=False, capacity=10 ** 6, worker_capacity=10 ** 6, pending=[(100, 1)] * 2)}
    grants, _, info = dsp.decide(views, 10, None, 0.0)
    assert grants == {'p': 2, 'h': 2}                        # p is capped by its slots; h takes its two
    assert info['shortfall'] == {'p': 4, 'h': 0}             # 6 free left: p's 4 other long items
    views['p']['scalable'] = False
    assert dsp.decide(views, 10, None, 0.0)[2]['shortfall'] == {'p': 0, 'h': 0}


def test_no_shortfall_when_budget_binds_or_gate_holds_or_gate_paused():
    p = view(capacity=2, worker_capacity=2, pending=[(5000, 1)] * 6, scalable=True)
    assert dsp.decide({'p': p}, 2, None, 0.0)[2]['shortfall'] == {'p': 0}          # the budget, not slots, binds
    assert dsp.decide({'p': p}, 10, 3, 0.0)[2]['shortfall'] == {'p': 0}            # gate holding: no new credit
    gp = view(capacity=0, worker_capacity=2, gate_paused=True, pending=[(5000, 1)] * 6, scalable=True)
    assert dsp.decide({'p': gp}, 10, None, 0.0)[2]['shortfall'] == {'p': 0}        # the gate limits it, not slots
    longer = view(withdrawable=False, capacity=10 ** 6, worker_capacity=10 ** 6, pending=[(9000, 1)] * 10)
    assert dsp.decide({'p': p, 'h': longer}, 10, None, 0.0)[2]['shortfall']['p'] == 0   # longer work fills the budget


def test_a_full_budget_still_shows_the_longest_lanes_shortfall():
    # #451 11-16 Snowflake lesson / val-sb10b 11-18: TB2.1 at its first fleet (64 of a 256 budget) while 900
    # Omniscience items absorb every freed slot; the shortfall only counted slots left after this tick's grants (0)
    tb = view(in_flight=64, starts=64, credit=64, capacity=64, worker_capacity=64, pending=[(5000, 1)] * 200,
              scalable=True)
    omni = view(withdrawable=False, kind='lane', in_flight=192, starts=192, credit=192, capacity=10 ** 6,
                worker_capacity=10 ** 6, pending=[(60, 1)] * 900)
    grants, _, info = dsp.decide({'tb': tb, 'omni': omni}, 256, None, 0.0)
    assert grants == {'tb': 0, 'omni': 0} and info['free'] == 0                  # the budget is full
    assert info['shortfall']['tb'] == 192                                          # it would hold the whole budget
    assert info['shortfall']['omni'] == 0
    # the shorter lane's turn: it would hold nothing more of the budget than it has
    short = view(in_flight=64, starts=64, credit=64, capacity=64, worker_capacity=64, pending=[(30, 1)] * 200,
                 scalable=True)
    assert dsp.decide({'tb': short, 'omni': omni}, 256, None, 0.0)[2]['shortfall']['tb'] == 0
    # full budget, still 0: a longer single-turn lane holds it; another pi lane's long tail holds it (pi sessions
    # last hours: not counted as freeing); no live worker yet; the gate holding
    longer = view(withdrawable=False, kind='lane', in_flight=192, starts=192, credit=192, capacity=10 ** 6,
                  worker_capacity=10 ** 6, pending=[(9000, 1)] * 900)
    assert dsp.decide({'tb': tb, 'h': longer}, 256, None, 0.0)[2]['shortfall']['tb'] == 0
    tail = view(in_flight=192, starts=192, credit=192, capacity=192, worker_capacity=192, pending=[])
    assert dsp.decide({'tb': tb, 'other': tail}, 256, None, 0.0)[2]['shortfall']['tb'] == 0
    assert dsp.decide({'tb': {**tb, 'worker_capacity': 0, 'capacity': 0}, 'omni': omni}, 256, None,
                      0.0)[2]['shortfall']['tb'] == 0
    assert dsp.decide({'tb': tb, 'omni': omni}, 256, 3, 0.0)[2]['shortfall']['tb'] == 0
    # a lane whose workers all left keeps the gradual path (the slots free now, or a tenth of the budget), not its
    # whole-budget share (256 - 0)
    gone = {**tb, 'in_flight': 0, 'starts': 0, 'credit': 0, 'capacity': 0, 'worker_capacity': 0, 'idle_exited': True}
    assert dsp.decide({'tb': gone, 'omni': omni}, 256, None, 0.0)[2]['shortfall']['tb'] == 64   # 256 - 192 free


def test_workers_to_add():
    w = dsp.workers_to_add
    assert w(4, 2, 0, 10, 2, 2, dsp.SCALE_TICKS, None) == 2       # wants 6 slots, has 2 -> +2 workers of 2
    assert w(4, 2, 0, 10, 2, 2, dsp.SCALE_TICKS - 1, None) == 0   # not sustained yet
    assert w(4, 2, 0, 10, 2, 2, dsp.SCALE_TICKS, 60.0) == 0       # cooldown
    assert w(4, 2, 0, 10, 6, 2, dsp.SCALE_TICKS, None) == 0       # starting workers already cover it
    assert w(40, 2, 0, 3, 2, 2, dsp.SCALE_TICKS, None) == 2       # never more slots than in flight + pending (5)
    assert w(0, 2, 0, 10, 2, 2, dsp.SCALE_TICKS, None) == 0
    assert w(40, 2, 0, 100, 2, 2, dsp.SCALE_TICKS, None, budget=8) == 3   # capped at the budget (8 slots)


class SyncThread:
    """threading.Thread stand-in: runs the target at start(), so ticks are deterministic."""
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def jobs_json(run, age=3600):
    import datetime
    t = datetime.datetime.fromtimestamp(time.time() - age, datetime.timezone.utc).isoformat(timespec='seconds')
    (run / 'jobs.json').write_text(json.dumps([{'cluster': '10', 'n': 1, 'submit_file': '/x.sub', 'submitted_at': t}]))


def test_tick_grows_a_slot_bound_pi_lane_once_per_cooldown(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    assert json.loads((d / 'dispatch.json').read_text())['pi_autoscale'] is True
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2)
    jobs_json(run)
    pilane.hold(run)
    pilane.register(rec, run)
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 2)   # one worker of 2 slots queued or running
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    for _ in range(dsp.SCALE_TICKS + 2):
        st = D.tick()
    assert st['lanes']['tb21run']['grant'] == 0 and added == [3]    # 8 items, 2 slots: wants 8 -> +3 workers of 2
    assert 'slot-bound' in (d / 'log.txt').read_text()
    for _ in range(3):
        D.tick()
    assert added == [3]                                             # cooldown


def test_tick_grows_the_longest_lane_while_a_single_turn_lane_fills_the_budget(tmp_path, monkeypatch):
    # val-sb10b 11-18: the single-turn lane takes every freed slot, the budget stays full; the Pi lane (longer items,
    # its 2 slots busy) must still grow (before: no shortfall, it stayed at its first fleet)
    from aa_evals.common.dispatch.lane import Lane
    from aa_evals.common.dispatch.test_dispatch import claim
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '6', '--durations', str(table(tmp_path)), '--expect-lanes', '2'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2)
    jobs_json(run)
    for k in ('long__a0__r0', 'long__a1__r0'):
        claim(run, k)                                                # its 2 slots busy
    (run / 'control.json').write_text(json.dumps({'max_starts': 2}))
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'hlerun', start_poller=False)
    lane.expect([(f'q3.r{i}', 'q3', 1) for i in range(40)])          # short items, many
    lane.flush()
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 2)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    st = D.tick()
    assert st['lanes']['hle.hlerun']['grant'] == 4                   # the single-turn lane fills the budget
    for _ in range(dsp.SCALE_TICKS + 1):
        st = D.tick()
    assert st['used'] == 6 and added and added[0] >= 2               # the budget stays full and the Pi lane grows


def test_existing_dispatch_dirs_do_not_grow(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    cfg = json.loads((d / 'dispatch.json').read_text())
    del cfg['pi_autoscale']                                          # a dir made before #378 (e.g. c30)
    (d / 'dispatch.json').write_text(json.dumps(cfg))
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2)
    pilane.hold(run)
    pilane.register(rec, run)
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    D = dsp.Dispatcher(d)
    for _ in range(dsp.SCALE_TICKS + 2):
        D.tick()
    assert added == []


def test_alive_worker_slots_counts_queued_running_and_held(tmp_path):
    class L:
        default_slots = 4

        def worker_jobs(self):
            return {'10': ('snap', 4), '11': ('snap', 8)}
    out = types.SimpleNamespace(stdout='10 2\n11 1\n11 5\n11 4\n', stderr='', returncode=0)
    # a held worker (5, e.g. a memory hold the net releases) counts: growth must not replace it at each cooldown;
    # a completed one (4) does not
    assert dsp.alive_worker_slots(L(), runner=lambda *a, **k: out) == 20


def test_no_growth_before_the_runs_own_submission_or_on_condor_errors(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2)       # no jobs.json: launched with --no-submit, or still submitting
    pilane.hold(run)
    pilane.register(rec, run)
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(threading, 'Thread', SyncThread)

    def boom(lane):
        raise RuntimeError('condor_q failed')
    monkeypatch.setattr(dsp, 'alive_worker_slots', boom)
    D = dsp.Dispatcher(d)
    for _ in range(dsp.SCALE_TICKS + 2):
        D.tick()
    assert added == []
    jobs_json(run, age=1)                             # just submitted: still not
    for _ in range(2):
        D.tick()
    assert added == []
    jobs_json(run)                                    # old enough, but condor_q fails: logged, no submission
    for lane in D.lanes.values():
        lane.last_scale = None
    for _ in range(dsp.SCALE_TICKS + 1):
        D.tick()
    assert added == [] and 'worker growth failed' in (d / 'log.txt').read_text()


def test_alive_worker_slots_raises_on_condor_error():
    class L:
        default_slots = 4

        def worker_jobs(self):
            return {'10': ('snap', 4)}
    bad = types.SimpleNamespace(stdout='', stderr='Failed to connect', returncode=1)
    import pytest
    with pytest.raises(RuntimeError):
        dsp.alive_worker_slots(L(), runner=lambda *a, **k: bad)


def test_server_released_once_after_all_lanes_done_and_idle(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1',
              '--registry', 'c30-x', '--release-server-after', '600', '--no-pi-autoscale'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long'], 1)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['long__a0__r0'], done=[('long__a0__r0', 100.0)])
    pilane.hold(run)
    pilane.register(rec, run)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    seen = []
    D = dsp.Dispatcher(d)
    monkeypatch.setattr(D, '_release_server', lambda name: seen.append(name))
    monkeypatch.setattr(D, 'engines_idle', lambda: True)
    D.tick()
    D.released = True
    t0 = time.time()
    D.maybe_release_server(D.snaps, 0, t0)
    assert seen == []                                                           # idle, but not for 600 s yet
    D.maybe_release_server(D.snaps, 0, t0 + 601)
    D.maybe_release_server(D.snaps, 0, t0 + 1200)
    assert seen == ['c30-x']                                                    # once


def release_setup(tmp_path, monkeypatch, st_lane):
    """A finished pi lane plus a single-turn lane whose lane.json is `st_lane` (merged over a fresh one)."""
    from aa_evals.common.dispatch.lane import Lane
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '2',
              '--registry', 'c30-x', '--release-server-after', '600', '--no-pi-autoscale'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long'], 1)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['long__a0__r0'], done=[('long__a0__r0', 100.0)])
    pilane.hold(run)
    pilane.register(rec, run)
    lane = Lane.open(d, 'hle', run=tmp_path / 'hlerun', start_poller=False)
    lane.expect([])
    lane.flush()
    p = d / 'lanes' / 'hle.hlerun' / 'lane.json'
    p.write_text(json.dumps({**json.loads(p.read_text()), 'starts': 5, **st_lane}))
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    seen = []
    D = dsp.Dispatcher(d)
    monkeypatch.setattr(D, '_release_server', lambda name: seen.append(name))
    monkeypatch.setattr(D, 'engines_idle', lambda: True)
    D.tick()
    D.released = True
    return d, D, seen


def idle_for(D, minutes):
    t0 = time.time()
    for m in range(0, minutes + 1):
        D.maybe_release_server(D.snaps, 0, t0 + 60 * m)


def test_release_when_nothing_is_pending_or_in_flight_even_if_a_lane_is_not_done(tmp_path, monkeypatch):
    # B1 (#531; sb2-0p33b 11-18): a SciCode step hung in CPU grading kept its lane open (not `done`) and 2 B200s idle
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': False, 'in_flight': 0, 'n_pending': 0})
    assert not D.snaps['hle.hlerun']['done']
    idle_for(D, 11)
    assert seen == ['c30-x']


def test_a_lane_that_will_send_more_model_work_holds_the_server(tmp_path, monkeypatch):
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': False, 'in_flight': 0, 'n_pending': 0,
                                                       'hold_server': True})
    idle_for(D, 30)
    assert seen == []


def test_a_dead_lane_with_work_left_blocks_the_release_and_says_so(tmp_path, monkeypatch):
    # B2: release_when_idle v1 released sb2-2p2b while its hung Omniscience client had 830 attempts unrun
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': False, 'in_flight': 3, 'n_pending': 830,
                                                       'epoch': time.time() - 3600})
    assert D.snaps['hle.hlerun']['dead']
    idle_for(D, 30)
    assert seen == [] and 'STUCK' in (d / 'log.txt').read_text()


def test_busy_or_unreadable_engines_hold_the_server(tmp_path, monkeypatch):
    for engines in (False, None):
        with __import__('tempfile').TemporaryDirectory() as tmp:
            from pathlib import Path
            d, D, seen = release_setup(Path(tmp), monkeypatch, {'closed': True, 'in_flight': 0, 'n_pending': 0})
            monkeypatch.setattr(D, 'engines_idle', lambda e=engines: e)
            idle_for(D, 30)
            assert seen == [], engines


def pi_release_setup(tmp_path, monkeypatch, claims=(), done=(), items=('long',), stop=None):
    """One pi lane whose workers' heartbeats are an hour old (dead), with the given claims/done; optional STOP_WORKERS."""
    import datetime
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1',
              '--registry', 'c30-x', '--release-server-after', '600', '--no-pi-autoscale'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', list(items), 1)
    run = fake_pi_run(tmp_path, order, slots=2, claims=list(claims), done=list(done))
    old = datetime.datetime.fromtimestamp(time.time() - 3600, datetime.timezone.utc).isoformat(timespec='seconds')
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': old}))
    if stop is not None:
        (run / 'STOP_WORKERS').write_text(stop)
    pilane.hold(run)
    pilane.register(rec, run)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    seen = []
    D = dsp.Dispatcher(d)
    monkeypatch.setattr(D, '_release_server', lambda name: seen.append(name))
    monkeypatch.setattr(D, 'engines_idle', lambda: True)
    D.lanes_first_seen = None
    D.tick()
    for lane in D.lanes.values():
        lane.first_seen = time.time() - 7200
    D.tick()
    D.released = True
    return d, D, seen


def test_a_dead_pi_lane_with_open_claims_is_stuck(tmp_path, monkeypatch):
    # review: its in_flight reads 0 once dead; the workers may come back and re-queue against a removed server
    d, D, seen = pi_release_setup(tmp_path, monkeypatch, claims=['long__a0__r0'], items=('long',))
    s = D.snaps['tb21run']
    assert s['dead'] and s['n_pending'] == 0 and s['work_left'] == 1
    idle_for(D, 30)
    assert seen == [] and 'STUCK' in (d / 'log.txt').read_text()


def test_a_lane_stopped_on_purpose_lets_the_server_go(tmp_path, monkeypatch):
    d, D, seen = pi_release_setup(tmp_path, monkeypatch, items=('long', 'mid'), stop='stop: suite dropped\n')
    assert D.snaps['tb21run']['stopped'] and D.snaps['tb21run']['n_pending'] > 0
    D.lanes['tb21run'].credit = 0
    for lane in D.snaps.values():
        lane['starts'] = max(lane['starts'], 1)
    idle_for(D, 11)
    assert seen == ['c30-x'] and 'STUCK' not in (d / 'log.txt').read_text()


def test_no_hold_lets_an_operator_drop_a_stuck_lane(tmp_path, monkeypatch):
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': False, 'in_flight': 3, 'n_pending': 830,
                                                       'epoch': time.time() - 3600})
    idle_for(D, 15)
    assert seen == []
    (d / 'lanes' / 'hle.hlerun' / 'NO_HOLD').write_text('')
    D.tick()
    idle_for(D, 11)
    assert seen == ['c30-x']


def test_an_unannounced_lane_holds_the_server(tmp_path, monkeypatch):
    # review: SciCode opens its lane, then resolves its inputs (up to 10 min) before lane.expect
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': False, 'in_flight': 0, 'n_pending': 0,
                                                       'announced': False})
    idle_for(D, 30)
    assert seen == []


def test_hold_server_is_written_by_the_lane_client(tmp_path):
    from aa_evals.common.dispatch.lane import Lane
    lane = Lane(tmp_path / 'lane', 'scicode', 'scicode', start_poller=False)
    assert lane.state()['hold_server'] is False
    lane.hold_server = True                      # next chain step after grading: more model work is coming
    assert lane.state()['hold_server'] is True


def test_engines_are_not_probed_after_the_release(tmp_path, monkeypatch):
    d, D, seen = release_setup(tmp_path, monkeypatch, {'closed': True, 'in_flight': 0, 'n_pending': 0})
    calls = []
    monkeypatch.setattr(D, 'engines_idle', lambda: calls.append(1) or True)
    idle_for(D, 11)
    n = len(calls)
    idle_for(D, 11)
    assert seen == ['c30-x'] and len(calls) == n


def test_engines_idle_reads_every_engine(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--registry', 'r'])
    D = dsp.Dispatcher(d)
    reg = tmp_path / 'serving' / 'r'
    reg.mkdir(parents=True)
    (reg / 'endpoints.json').write_text(json.dumps({'engine_urls': ['http://a/v1', 'http://b/v1']}))
    monkeypatch.setattr(D, 'registry_root', lambda: tmp_path / 'serving')
    texts = {'http://a/metrics': 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n',
             'http://b/metrics': 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting 0\n'}

    class Op:
        def open(self, url, timeout=10):
            if url not in texts:
                raise OSError('down')
            return types.SimpleNamespace(read=lambda u=url: texts[u].encode())
    monkeypatch.setattr(dsp.urllib.request, 'build_opener', lambda *a: Op())
    assert D.engines_idle() is True
    texts['http://b/metrics'] = 'vllm:num_requests_running 2\n'
    assert D.engines_idle() is False
    del texts['http://a/metrics']
    assert D.engines_idle() is None
    texts['http://a/metrics'] = 'vllm:num_requests_running garbage\n'   # any error, not only OSError: busy
    assert D.engines_idle() is None


def test_release_server_refuses_foreign_jobs(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--release-server-after', '600'])
    import aa_evals.common.harbor_mswea.endpoint as ep
    monkeypatch.setattr(ep, 'SERVING_ROOT', tmp_path / 'serving')
    (tmp_path / 'serving' / 'c30-x').mkdir(parents=True)
    (tmp_path / 'serving' / 'c30-x' / 'job.json').write_text(json.dumps({'job_id': '123.0'}))
    import getpass
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        out = f'{getpass.getuser()} tasksynth-serve-m5-super-1 2' if argv[0] == 'condor_q' else ''
        return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
    D = dsp.Dispatcher(d)
    D._release_server('c30-x', runner=runner)
    assert not any(c[0] == 'condor_rm' for c in calls) and 'refusing' in (d / 'log.txt').read_text()

    def runner2(argv, **kw):
        calls.append(argv)
        out = f'{getpass.getuser()} tasksynth-serve-c30-x 2' if argv[0] == 'condor_q' else 'removed'
        return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
    D._release_server('c30-x', runner=runner2)
    assert ['condor_rm', '123.0'] in calls


# ------------------------------------------------------------------------------- #378 idle exit (no start credit)

def iso(age):
    import datetime
    return datetime.datetime.fromtimestamp(time.time() - age, datetime.timezone.utc).isoformat(timespec='seconds')


def test_snapshot_idle_exited_lane_is_waiting_not_dead(tmp_path):
    run = fake_pi_run(tmp_path, [('a', 0), ('b', 0), ('c', 0)], slots=2, claims=['a__a0__r0'],
                      done=[('a__a0__r0', 100.0)])
    (run / 'queue' / 'admission.json').write_text(json.dumps({'paused': True, 'reason': 'gate holds 2 new'}))
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(3600), 'running': []}))
    (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(1800), 'exited': 'idle-no-credit'}))
    lane = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)})
    s = lane.snapshot(time.time())
    assert not s['dead'] and s['idle_exited'] and s['capacity'] == 0 and s['ready']
    assert not s['paused'] and not s['gate_paused'] and 'no start credit' in s['reason']   # stale admission ignored
    # an exited worker's fresh heartbeat gives no slots
    (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(5), 'exited': 'idle-no-credit'}))
    assert dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())['worker_capacity'] == 0
    # a lane with an attempt still claimed and no worker left is dead, as before
    (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(1800), 'exited': 'idle-no-credit'}))
    claim_dir = run / 'queue' / 'claims' / 'b__a0__r0'
    claim_dir.mkdir()
    (claim_dir / 'claim.json').write_text(json.dumps({'claimed_at': iso(2000)}))
    assert dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())['dead']
    # the last worker to leave did not say why (killed, or eval_q35's own exit): dead, as before
    claim_dir.joinpath('claim.json').unlink()
    claim_dir.rmdir()
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(900), 'running': []}))
    assert dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())['dead']


def test_idle_exited_lane_gets_shortfall_and_grows_back(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['long__a0__r0'], done=[('long__a0__r0', 100.0)])
    jobs_json(run)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(1800), 'exited': 'idle-no-credit'}))
    (run / 'control.json').write_text(json.dumps({'max_starts': 1}))
    pilane.register(rec, run)
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)   # the workers that left are out of the queue
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    D.released = True
    for _ in range(dsp.SCALE_TICKS + 1):
        st = D.tick()
    lane = st['lanes']['tb21run']
    assert not lane['dead'] and lane['grant'] == 0 and lane['n_pending'] == 7
    assert added == [4]                                             # 7 pending, budget 10: 8 slots -> 4 x 2


def test_idle_exited_lane_keeps_its_longest_first_turn():
    """Its workers left; the slots freed tick by tick would go to the shorter lane after it (capacity-0 lanes pass
    their turn). Its shortfall is what it would win of the next slots to free, so it grows back as the head."""
    views = {'p': view(capacity=0, worker_capacity=0, pending=[(5000, 1)] * 20, scalable=True, idle_exited=True),
             's': view(withdrawable=False, capacity=10 ** 6, worker_capacity=10 ** 6, in_flight=9,
                       pending=[(10, 1)] * 50)}
    grants, _, info = dsp.decide(views, 10, None, 0.0)
    assert grants == {'p': 0, 's': 1}                      # work-conserving: the free slot is still used
    assert info['shortfall']['p'] == 1                     # max(free 1, budget / 10 = 1)
    views['s']['in_flight'] = 10
    assert dsp.decide(views, 10, None, 0.0)[2]['shortfall']['p'] == 1   # also with no slot free this tick
    views['p']['idle_exited'] = False                      # a lane with live idle-able slots: as before
    assert dsp.decide(views, 10, None, 0.0)[2]['shortfall']['p'] == 0
    longer = view(withdrawable=False, capacity=10 ** 6, worker_capacity=10 ** 6, in_flight=10,
                  pending=[(9000, 1)] * 50)
    views = {'p': {**views['p'], 'idle_exited': True}, 's': longer}
    assert dsp.decide(views, 10, None, 0.0)[2]['shortfall']['p'] == 0   # not the head: waits, no workers
    assert dsp.decide({'p': views['p']}, 10, 2, 0.0)[2]['shortfall']['p'] == 0   # gate holding: nothing


def test_idle_exited_lane_credit_is_withdrawn(tmp_path, monkeypatch):
    """Credit granted just as its last worker left has no one to spend it: back to the budget."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2)
    jobs_json(run)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(30), 'exited': 'idle-no-credit'}))
    (run / 'control.json').write_text(json.dumps({'max_starts': 3}))   # 3 unspent: the whole budget
    pilane.register(rec, run)
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    D.released = True
    for _ in range(dsp.SCALE_TICKS + 1):
        st = D.tick()
    assert json.loads((run / 'control.json').read_text())['max_starts'] == 0
    assert st['lanes']['tb21run']['outstanding'] == 0 and added == [2]   # grows back: 3 slots -> 2 workers of 2


def test_old_autoscale_dirs_have_no_idle_exit(tmp_path):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--no-pi-autoscale'])
    assert json.loads((d / 'dispatch.json').read_text())['pi_idle_exit_sec'] == 0


def test_virtual_shortfall_ignores_lanes_that_cannot_take_credit():
    p = view(capacity=0, worker_capacity=0, pending=[(5000, 1)] * 20, scalable=True, idle_exited=True)
    s = view(withdrawable=False, capacity=10 ** 6, worker_capacity=10 ** 6, in_flight=99, pending=[(10, 1)] * 50)
    for other in (view(dead=True, pending=[(9000, 1)] * 50),                      # an abandoned run
                  view(capacity=0, worker_capacity=4, pending=[(9000, 1)] * 50),  # stopped / paused
                  view(capacity=0, worker_capacity=4, gate_paused=True, withdrawn=True, pending=[(9000, 1)] * 50)):
        sh = dsp.decide({'p': p, 's': s, 'x': other}, 100, None, 0.0)[2]['shortfall']
        assert sh['p'] == 10, other                                                # a tenth of the budget
    assert dsp.decide({'p': p, 's': s}, 100, None, 0.0, flowing=False)[2]['shortfall']['p'] == 1   # only free now


def test_a_lost_stragglers_claims_alone_make_a_dead_lane_revivable(tmp_path, monkeypatch):
    # #538 review: claim packing empties all but the fullest worker; if that last straggler is evicted holding claims
    # and nothing is pending, nobody reaps them (revivable needed pending work)
    for mark in ('idle-no-credit', 'no-claimable'):
        root = tmp_path / mark
        root.mkdir()
        d = root / 'disp'
        dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(root)), '--expect-lanes', '1'])
        rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long'], 2)
        run = fake_pi_run(root, order, slots=2, claims=['long__a0__r0', 'long__a1__r0'])
        jobs_json(run)
        (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(3600), 'exited': mark}))
        (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(1800),
                                                             'running': ['long__a0__r0', 'long__a1__r0']}))
        pilane.register(rec, run)
        s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
        assert s['dead'] and s['revivable'], mark
        assert s['n_pending'] == 2 and s['work_left'] == 2   # nothing unclaimed: its 2 lost claims are the work left
        added = []
        monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                            types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
        monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
        monkeypatch.setattr(threading, 'Thread', SyncThread)
        D = dsp.Dispatcher(d)
        D.released = True
        for _ in range(dsp.SCALE_TICKS + 1):
            D.tick()
        assert added, mark                                # a worker comes back to reap the lost claims


def test_lost_claims_come_first_and_no_hold_abandons(tmp_path, monkeypatch):
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 2)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['mid__a0__r0'])
    jobs_json(run)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(3600), 'exited': 'no-claimable'}))
    (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(1800), 'running': ['mid__a0__r0']}))
    pilane.register(rec, run)
    s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
    assert s['revivable'] and s['pending'][0][0] == 'mid__a0__r0'   # reaped, it is a retry: claimed first
    assert s['work_left'] == s['n_pending'] == 4                    # 3 unclaimed + 1 lost claim, counted once
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    D.released = True
    (d / 'lanes' / 'tb21run').mkdir(parents=True, exist_ok=True)
    (d / 'lanes' / 'tb21run' / 'NO_HOLD').write_text('')      # the operator let the lane go: never revived
    for _ in range(dsp.SCALE_TICKS + 1):
        D.tick()
    assert list(D.lanes) == ['tb21run'] and added == []
    (d / 'lanes' / 'tb21run' / 'NO_HOLD').unlink()             # without it, the same lane grows back
    for _ in range(dsp.SCALE_TICKS + 1):
        D.tick()
    assert added


def test_an_abandoned_lane_is_not_revived(tmp_path):
    # no worker left on its own (all evicted or killed): dead, not revived (unchanged)
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long'], 2)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['long__a0__r0'])
    jobs_json(run)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(1800), 'running': ['long__a0__r0']}))
    pilane.register(rec, run)
    s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
    assert s['dead'] and not s['revivable']


def test_dead_lane_after_idle_exits_grows_back(tmp_path, monkeypatch):
    """Some workers left idle; the last one then died holding a claim: dead, but revivable (a new worker reaps it)."""
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '10', '--durations', str(table(tmp_path)), '--expect-lanes', '1'])
    rec, order = pilane.plan(d, 'tb21run', 'tb21', ['long', 'mid'], 4)
    run = fake_pi_run(tmp_path, order, slots=2, claims=['long__a0__r0'])
    jobs_json(run)
    (run / 'workers' / 'w1.json').write_text(json.dumps({'heartbeat': iso(3600), 'exited': 'idle-no-credit'}))
    (run / 'workers' / 'w2.json').write_text(json.dumps({'heartbeat': iso(1800), 'running': ['long__a0__r0']}))
    pilane.register(rec, run)
    s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
    assert s['dead'] and s['revivable']
    added = []
    monkeypatch.setitem(sys.modules, 'aa_evals.common.pi_eval.run',
                        types.SimpleNamespace(submit_workers=lambda r, cfg, n: added.append(n) or '99'))
    monkeypatch.setattr(dsp, 'alive_worker_slots', lambda lane: 0)
    monkeypatch.setattr(threading, 'Thread', SyncThread)
    D = dsp.Dispatcher(d)
    D.released = True
    for _ in range(dsp.SCALE_TICKS + 1):
        D.tick()
    assert added == [4]                                   # 7 pending: 7 slots -> 4 workers of 2
    (run / 'STOP_WORKERS').write_text('')                 # stopped on purpose: never revived
    s = dsp.PiLane({'name': 'p', 'suite': 'tb21', 'run': str(run)}).snapshot(time.time())
    assert s['paused']

"""#531 claim packing (claim_pack.py): scarce work goes to the fullest live workers; only who claims changes."""
import datetime as dt
import json
import time
import types

from aa_evals.common.pi_eval import claim_pack as P
from aa_evals.common.pi_eval import worker as W   # noqa: F401  (puts eval_q35/ on sys.path: queue_state, common)

import common as q35common      # eval_q35/common.py
import queue_state               # eval_q35/queue_state.py


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


def make_run(tmp_path, items, limit=1000, max_starts=None):
    run = tmp_path / 'run'
    q = run / 'queue'
    for sub in ('claims', 'done', 'retry'):
        (q / sub).mkdir(parents=True, exist_ok=True)
    (run / 'workers').mkdir()
    (q / 'items.txt').write_text(''.join(f'{t} {a}\n' for t, a in items))
    (q / 'admission.json').write_text(json.dumps({'epoch': time.time(), 'paused': False, 'limit': limit,
                                                  'burst': 10 ** 6, 'window_sec': 60}))
    if max_starts is not None:
        (run / 'control.json').write_text(json.dumps({'max_starts': max_starts}))
    return run


def heartbeat(run, name, running, t=None, **extra):
    (run / 'workers' / f'{name}.json').write_text(json.dumps({'heartbeat': iso(t or time.time()),
                                                               'running': running, **extra}))


def fake_worker(run, name, running=(), slots=4, dispatched=False, sched=False, pack=True):
    w = types.SimpleNamespace(run=run, name=name, running={k: {} for k in running},
                              state=queue_state.State(run), a=types.SimpleNamespace(slots=slots),
                              config={'pack_claims': pack, 'worker': {'slots': slots},
                                      **({'dispatch': {'dir': 'x'}} if dispatched else {}),
                                      **({'sched': {'file': 'x'}} if sched else {})})
    return w


def test_peers_only_fresh_live_claiming_heartbeats(tmp_path):
    run = make_run(tmp_path, [('t', 0)])
    now = time.time()
    heartbeat(run, 'a', ['k1', 'k2'])                                   # an older heartbeat: no slots/claiming keys
    heartbeat(run, 'b', ['k5'], slots=24, claiming=True)
    heartbeat(run, 'drained', ['k6', 'k7'], slots=32, claiming=False)   # STOP_WORKERS drain, proxy down, gate hold
    heartbeat(run, 'stale', ['k3'], t=now - P.PEER_FRESH - 5)
    heartbeat(run, 'old', ['k8'], t=now - P.PEER_GONE - 5)
    heartbeat(run, 'gone', [], exited='idle-no-credit')
    heartbeat(run, 'me', ['k4'])
    (run / 'workers' / 'one_per_worker.json').write_text(json.dumps({'tasks': ['x']}))   # not a heartbeat
    (run / 'workers' / 'bad.json').write_text('{')
    gone = set()
    assert sorted(P.peers(run / 'workers', 'me', now, 32, gone)) == [('a', 2, 32), ('b', 1, 24)]
    assert gone == {'old', 'gone', 'one_per_worker'}                   # never read again
    (run / 'workers' / 'gone.json').write_text('{')                     # would raise if re-read: skipped
    assert sorted(P.peers(run / 'workers', 'me', now, 32, gone)) == [('a', 2, 32), ('b', 1, 24)]


def test_demand_counts_unclaimed_and_retries_capped_by_credit_and_limit(tmp_path):
    run = make_run(tmp_path, [('t1', 0), ('t2', 0), ('t3', 0), ('t4', 0)], limit=10, max_starts=5)
    st = queue_state.State(run)
    (run / 'queue' / 'claims' / 't1__a0__r0').mkdir()
    (run / 'queue' / 'retry' / 't9__a0__r1.json').write_text('{}')
    assert P.demand(st, run, dispatched=False) == 4          # t2, t3, t4 + the retry
    assert P.demand(st, run, dispatched=True) == 4           # credit 5 - 1 claim
    (run / 'control.json').write_text(json.dumps({'max_starts': 3}))
    assert P.demand(st, run, dispatched=True) == 2
    (run / 'queue' / 'admission.json').write_text(json.dumps({'limit': 2, 'paused': False, 'epoch': time.time()}))
    assert P.demand(st, run, dispatched=False) == 1          # 2 in flight allowed, 1 active


def test_ahead_free_orders_by_running_then_name():
    others = [('b', 3, 4), ('c', 1, 4), ('a', 1, 4), ('full', 4, 4), ('big', 1, 8)]
    assert P.ahead_free('me', 1, others) == 1 + 3 + 3 + 7    # b (more running); a, c, big (as many, earlier names)
    assert P.ahead_free('a', 1, [('me', 1, 4)]) == 0         # 'me' sorts after 'a'
    assert P.ahead_free('a', 3, others) == 0                 # b has as many but sorts later; full has no slot


def test_packer_defers_only_while_fuller_workers_can_take_the_work(tmp_path):
    run = make_run(tmp_path, [(f't{i}', 0) for i in range(3)])
    heartbeat(run, 'full', ['x1', 'x2'])                      # 2 of 4 slots free
    t = [1000.0]
    me = P.Packer(fake_worker(run, 'me'), 4, False, log=lambda m: None, clock=lambda: t[0])
    assert me.defer() is False                                # demand 3 > the fuller worker's 2 free slots
    (run / 'queue' / 'claims' / 't0__a0__r0').mkdir()
    t[0] += P.CACHE_SEC
    assert me.defer() is True                                 # demand 2 fits there: leave it to 'full'
    heartbeat(run, 'full', ['x1', 'x2', 'x3', 'x4'])          # it filled up
    t[0] += P.CACHE_SEC
    assert me.defer() is False


def test_packer_bounded_deferral(tmp_path):
    run = make_run(tmp_path, [('t0', 0), ('t1', 0)])
    t = [1000.0]
    msgs = []
    me = P.Packer(fake_worker(run, 'me'), 4, False, log=msgs.append, clock=lambda: t[0])
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is True
    t[0] += P.DEFER_MAX - 1
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is True and not msgs                     # still within the bound
    t[0] += P.CACHE_SEC + 1
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is False and msgs                        # the fuller worker never claimed: claim freely
    t[0] += P.RELEASE_SEC - 10
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is False                                 # ... for RELEASE_SEC
    t[0] += 20
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is True                                  # then packing again


def test_packer_dropping_demand_keeps_deferring(tmp_path):
    run = make_run(tmp_path, [('t0', 0), ('t1', 0), ('t2', 0)])
    t = [1000.0]
    me = P.Packer(fake_worker(run, 'me'), 4, False, log=lambda m: None, clock=lambda: t[0])
    heartbeat(run, 'full', ['x1'], t=t[0])
    assert me.defer() is True
    for i in range(2):                                          # the fuller worker keeps claiming
        t[0] += P.DEFER_MAX - 1
        (run / 'queue' / 'claims' / f't{i}__a0__r0').mkdir()
        heartbeat(run, 'full', ['x1'], t=t[0])
        assert me.defer() is True


def test_install_flag_and_tbsched(tmp_path):
    run = make_run(tmp_path, [('t0', 0)])
    for kw, on in (({}, True), ({'pack': False}, False), ({'sched': True}, False)):
        w = fake_worker(run, 'me', **kw)
        p = P.install(w)
        assert (p is not None) is on
        assert ('claim' in vars(w.state)) is on               # wrapped (instance attribute) only when on


def simulate(tmp_path, pack, n_items=10, slots=8, names=('w1', 'w2', 'w3')):
    """Three workers claim from one eval_q35 queue (State.claim, the real one), one claim each per step in turn (as
    concurrent workers do, 2 s apart); an attempt runs 12 steps. Returns [(key, worker)] in claim order."""
    run = make_run(tmp_path, [(f't{i}', a) for i in range(n_items // 2) for a in range(2)])
    t = [1000.0]
    ws = []
    for n in names:
        w = fake_worker(run, n, slots=slots, pack=pack)
        w.packer = P.install(w, log=lambda m: None)
        if w.packer:
            w.packer.clock = lambda: t[0]
        ws.append(w)
    order, left = [], {}
    for rnd in range(60):
        t[0] += P.CACHE_SEC + 1
        for w in ws:   # finish attempts that ran 3 rounds; heartbeats like eval_q35's main loop
            for k in [k for k, r in left.items() if r == rnd and k in w.running]:
                del w.running[k]
                (run / 'queue' / 'done' / f'{k}.json').write_text('{}')
            heartbeat(run, w.name, sorted(w.running), t=t[0])
        for w in ws:
            if len(w.running) < slots:
                k = w.state.claim(w.name, {f't{i}' for i in range(n_items)})
                if k is not None:
                    w.running[k] = {}
                    left[k] = rnd + 12
                    order.append((k, w.name))
                    heartbeat(run, w.name, sorted(w.running), t=t[0])
    return order


def test_packing_changes_who_claims_not_what_or_its_seed(tmp_path):
    off = simulate(tmp_path / 'off', pack=False)
    on = simulate(tmp_path / 'on', pack=True)
    keys_off, keys_on = [k for k, _ in off], [k for k, _ in on]
    assert keys_on == keys_off and len(keys_on) == 10          # same attempts, same (queue) order
    seeds = lambda ks: [q35common.seed(*q35common.parse_key(k)[:2], 'ns') for k in ks]
    assert seeds(keys_on) == seeds(keys_off)                   # seeds depend on the key only
    used = lambda o: {w for _, w in o}
    assert used(off) == {'w1', 'w2', 'w3'}
    assert used(on) == {'w1', 'w2'}                            # packed: w3 never claimed, so it can leave


def test_no_claimable_exit_waits_for_a_lost_workers_claim(tmp_path):
    """IDLE_EXIT_S is 5 min now; the reaper needs STALE_SEC: an orphaned claim keeps the worker (#531)."""
    import os
    from aa_evals.common.pi_eval.test_idle_exit import make
    w = make(tmp_path, max_starts=1, claims=['t1__a0__r0'], items=[('t1', 0)])
    assert W.PiEvalWorker.IDLE_EXIT_S == 300
    assert W.PiEvalWorker.claimable(w) is False                # fresh claim of another worker: nothing for us
    d = w.run / 'queue' / 'claims' / 't1__a0__r0'
    old = time.time() - W.ORPHAN_SEC - 10
    os.utime(d, (old, old))
    assert W.PiEvalWorker.claimable(w) is True                 # its worker is lost: stay until it is reaped
    (w.run / 'queue' / 'done' / 't1__a0__r0.json').write_text('{}')
    assert W.PiEvalWorker.claimable(w) is False


def test_own_claims_are_not_progress(tmp_path):
    """Review #1: this worker's own claims (after a release) must not restart its wait, or a fuller worker that never
    claims would hold starts to ~3 per minute."""
    run = make_run(tmp_path, [(f't{i}', 0) for i in range(6)])
    t = [1000.0]
    w = fake_worker(run, 'me', slots=8)
    me = P.Packer(w, 8, False, log=lambda m: None, clock=lambda: t[0])
    claim = me.wrap(w.state.claim)
    heartbeat(run, 'full', ['x1'], t=t[0], slots=8, claiming=True)   # fuller (1 running), 7 free, never claims
    assert claim('me', {f't{i}' for i in range(6)}) is None        # deferred
    t[0] += P.DEFER_MAX + 1
    heartbeat(run, 'full', ['x1'], t=t[0], slots=8, claiming=True)
    got = []
    for _ in range(6):                                             # released: claims everything left, no new wait
        t[0] += P.CACHE_SEC + 1
        k = claim('me', {f't{i}' for i in range(6)})
        if k:
            got.append(k)
            w.running[k] = {}
    assert len(got) == 6 and me.mine == 6


def test_a_peer_that_cannot_claim_is_ignored(tmp_path):
    run = make_run(tmp_path, [('t0', 0), ('t1', 0)])
    t = [1000.0]
    me = P.Packer(fake_worker(run, 'me'), 4, False, log=lambda m: None, clock=lambda: t[0])
    heartbeat(run, 'full', ['x1', 'x2'], t=t[0], slots=4, claiming=False)   # e.g. an old snapshot being drained
    assert me.defer() is False


def test_no_peer_scan_without_demand(tmp_path, monkeypatch):
    run = make_run(tmp_path, [('t0', 0)], max_starts=1)
    (run / 'queue' / 'claims' / 't0__a0__r0').mkdir()
    me = P.Packer(fake_worker(run, 'me', dispatched=True), 4, True, log=lambda m: None)
    monkeypatch.setattr(P, 'peers', lambda *a, **k: (_ for _ in ()).throw(AssertionError('scanned')))
    assert me.defer() is False                                 # no credit left: decided without reading workers/

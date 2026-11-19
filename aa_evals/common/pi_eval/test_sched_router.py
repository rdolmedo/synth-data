"""#451 tbsched routing hook: claims within the scheduler's credit per engine, pinned to the engine with the most unspent
credit; the old rules whenever the scheduler's state is missing, not "on" or stale."""
import json
import random
import time

import pytest

from aa_evals.common.pi_eval import sched_router as R
from aa_evals.common.pi_eval import worker as W
from aa_evals.common.pi_eval.test_idle_exit import make

URLS = ['http://r0.invalid/v1', 'http://r1.invalid/v1']


class ClaimState:
    """eval_q35 State.claim's shape: mkdir the next unclaimed item's claim dir."""

    def __init__(self, q, items):
        self.q, self.items_ = q, items
        self.calls = 0

    def claim(self, worker, tasks_available):
        self.calls += 1
        for t, a in self.items_:
            k = f'{t}__a{a}__r0'
            d = self.q / 'claims' / k
            if not d.exists():
                d.mkdir()
                return k
        return None


def setup(tmp_path, credit, mode='on', updated=None, items=40):
    w = make(tmp_path, items=[(f't{i}', 0) for i in range(items)])
    w.urls, w.held_urls, w.pinned = list(URLS), frozenset(), {}
    w.config = {'seed_namespace': 'ns-r2', 'sched': {'file': str(tmp_path / 'r1.json')}}
    st = ClaimState(w.run / 'queue', [(f't{i}', 0) for i in range(items)])
    w.state = st
    (tmp_path / 'r1.json').write_text(json.dumps({'mode': mode, 'updated': time.time() if updated is None else updated,
                                                  'credit_urls': credit}))
    logs = []
    r = R.install(w, log=logs.append)
    return w, st, r, logs


def pins(w):
    out = {}
    for d in (w.run / 'queue' / 'claims').iterdir():
        p = d / 'replica.json'
        out[d.name] = json.loads(p.read_text())['url'] if p.exists() else None
    return out


def test_claims_follow_credit_to_the_engine_with_most_room(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 2, URLS[1]: 5})
    got = [w.state.claim('w1', None) for _ in range(9)]
    assert got[:7] == [f't{i}__a0__r0' for i in range(7)] and got[7:] == [None, None]   # 7 credits in all
    p = pins(w)
    assert sum(u == URLS[0] for u in p.values()) == 2 and sum(u == URLS[1] for u in p.values()) == 5
    assert all(w.pinned[k] == u for k, u in p.items())
    assert logs == ['tbsched router: credit governs claims']


@pytest.mark.parametrize('mode,updated,why', [('shadow', None, "mode 'shadow'"), ('on', 1.0, 'stale')])
def test_old_rules_when_the_scheduler_does_not_govern(tmp_path, mode, updated, why):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 0, URLS[1]: 0}, mode=mode, updated=updated)
    assert w.state.claim('w1', None) == 't0__a0__r0'          # no credit check: the run's own limits apply
    assert pins(w) == {'t0__a0__r0': None}
    assert logs == [f'tbsched router: old rules ({why})']


def test_a_missing_state_file_is_old_rules(tmp_path):
    w, st, r, logs = setup(tmp_path, {})
    (tmp_path / 'r1.json').unlink()
    assert w.state.claim('w1', None) == 't0__a0__r0' and not r.active()


def test_a_gate_holding_engine_gets_no_new_attempt(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 10, URLS[1]: 10})
    w.held_urls = frozenset([URLS[1]])
    for _ in range(4):
        w.state.claim('w1', None)
    assert set(pins(w).values()) == {URLS[0]}


def test_unpinned_claims_count_on_their_hashed_engine(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 3, URLS[1]: 3})
    old = [f'x{i}__a0__r0' for i in range(4)]
    for k in old:
        (w.run / 'queue' / 'claims' / k).mkdir()                 # claimed before the hook, no replica.json
    hashed = [R.hashed(URLS, k) for k in old]
    left = {u: 3 - hashed.count(u) for u in URLS}
    n = 0
    while w.state.claim('w1', None):
        n += 1
    assert n == sum(max(0, v) for v in left.values())


def test_no_claim_attempt_without_credit(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 0, URLS[1]: 0})
    assert w.state.claim('w1', None) is None and st.calls == 0


def test_another_worker_holding_the_lock_means_no_claim_now(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 5, URLS[1]: 5})
    (w.run / 'queue' / 'sched.lock').mkdir()
    assert w.state.claim('w1', None) is None and st.calls == 0


def test_copies_match_tbsched():
    from tbsched import place, work
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(1, 4)
        urls = [f'http://e{i}/v1' for i in range(n)]
        unspent = {u: rng.randint(-3, 40) for u in urls}
        task, attempt, ns = f't{rng.randint(0, 999)}', rng.randint(0, 4), rng.choice(['a', 'b-r2', ''])
        assert R.route(unspent, ns, task, attempt) == place.route(unspent, ns, task, attempt)
        k = f'{task}__a{attempt}__r{rng.randint(0, 3)}'
        assert R.hashed(urls, k) == work.hashed(urls, k)


def test_refresh_opens_eval_q35_limits_only_while_credit_governs(tmp_path, monkeypatch):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 5, URLS[1]: 5})
    w.config.update({'agent': 'pi', 'model_spec': 'serving:fake', 'served': 'm', 'max_in_flight': 48, 'burst': 8})
    w.sched = r
    monkeypatch.setattr(W.registry, 'alive', lambda *a: (True, {'urls': URLS, 'served': 'm'}, ''))
    monkeypatch.setattr(W, 'alive', lambda *a: True)
    monkeypatch.setattr(W, 'gate_queued_new', lambda u: 0)
    w.refresh()
    adm = json.loads((w.run / 'queue' / 'admission.json').read_text())
    assert (adm['limit'], adm['burst'], adm.get('tbsched')) == (R.OPEN, R.OPEN, True)
    (tmp_path / 'r1.json').write_text(json.dumps({'mode': 'shadow', 'updated': time.time(), 'credit_urls': {}}))
    w.refresh()
    adm = json.loads((w.run / 'queue' / 'admission.json').read_text())
    assert (adm['limit'], adm['burst'], adm.get('tbsched')) == (48, 8, None)


# ---- review fixes: the real eval_q35 State, malformed files, provisional hashes, the #402 pin path

def real_state(tmp_path, n=40, limit=4, burst=2, tbsched=False):
    from queue_state import State      # eval_q35/ is on sys.path once pi_eval's worker module is imported
    run = tmp_path / 'realrun'
    q = run / 'queue'
    for d in ('claims', 'done', 'retry'):
        (q / d).mkdir(parents=True, exist_ok=True)
    (q / 'items.txt').write_text(''.join(f't{i} 0\n' for i in range(n)))
    (q / 'admission.json').write_text(json.dumps({'epoch': time.time(), 'paused': False, 'limit': limit,
                                                  'burst': burst, 'window_sec': 60,
                                                  **({'tbsched': True} if tbsched else {})}))
    return run, State(run)


def router_on(tmp_path, run, st, rec):
    w = make(tmp_path / 'w', items=[])
    w.run, w.urls, w.held_urls, w.pinned = run, list(URLS), frozenset(), {}
    w.config = {'seed_namespace': 'ns', 'sched': {'file': str(tmp_path / 'state.json')}}
    (tmp_path / 'state.json').write_text(json.dumps(rec))
    raw = st.claim
    old = lambda worker, avail: raw(worker, avail)        # stands in for the #402 replica claim
    w.state = st
    st.claim = old
    logs = []
    R.install(w, raw_claim=raw, log=logs.append)
    return w, logs


def test_on_the_way_back_no_claim_while_admission_is_still_open(tmp_path):
    run, st = real_state(tmp_path, limit=10 ** 6, burst=10 ** 6, tbsched=True)
    w, logs = router_on(tmp_path, run, st, {'mode': 'on', 'updated': 1.0, 'credit_urls': {}})   # stale
    assert w.state.claim('w1', {f't{i}' for i in range(40)}) is None
    (run / 'queue' / 'admission.json').write_text(json.dumps({'epoch': time.time(), 'paused': False, 'limit': 4,
                                                              'burst': 2, 'window_sec': 60}))   # the next refresh
    got = [w.state.claim('w1', {f't{i}' for i in range(40)}) for _ in range(10)]
    assert sum(k is not None for k in got) == 2                     # the old limits hold again (burst 2)


def test_credit_mode_with_the_real_state(tmp_path):
    run, st = real_state(tmp_path, limit=10 ** 6, burst=10 ** 6, tbsched=True)
    w, logs = router_on(tmp_path, run, st, {'mode': 'on', 'updated': time.time(),
                                            'credit_urls': {URLS[0]: 3, URLS[1]: 1}})
    got = [w.state.claim('w1', {f't{i}' for i in range(40)}) for _ in range(6)]
    assert sum(k is not None for k in got) == 4
    p = [json.loads((run / 'queue' / 'claims' / k / 'replica.json').read_text()) for k in got if k]
    assert [x['url'] for x in p].count(URLS[0]) == 3 and all(x['why'] == 'tbsched' for x in p)


@pytest.mark.parametrize('content', ['[1, 2]', '{"mode": "on", "updated": "soon", "credit_urls": {}}',
                                     '{"mode": "on", "updated": 1e12, "credit_urls": {"u": 1}}',
                                     '{"mode": "on", "updated": NOW, "credit_urls": {"u": "x"}}',
                                     '{"mode": "on", "updated": NOW, "credit_urls": [1]}'])
def test_a_malformed_or_future_state_file_is_old_rules_never_a_crash(tmp_path, content):
    w, st, r, logs = setup(tmp_path, {})
    (tmp_path / 'r1.json').write_text(content.replace('NOW', str(time.time())))
    assert r.credit() is None
    assert w.state.claim('w1', None) == 't0__a0__r0'


def test_an_unpinned_live_claim_is_read_again(tmp_path):
    w, st, r, logs = setup(tmp_path, {URLS[0]: 5, URLS[1]: 5})
    d = w.run / 'queue' / 'claims' / 'x1__a0__r0'
    d.mkdir()
    first = r.starts(URLS)
    other = URLS[1] if R.hashed(URLS, 'x1__a0__r0') == URLS[0] else URLS[0]
    (d / 'replica.json').write_text(json.dumps({'url': other, 'why': 'gate holds'}))   # an old-rules #402 pin, late
    assert r.starts(URLS) == {other: 1} and first != {other: 1}


def test_credit_mode_does_not_take_the_402_path(tmp_path):
    run, st = real_state(tmp_path, limit=10 ** 6, burst=10 ** 6, tbsched=True)
    calls = []
    w, logs = router_on(tmp_path, run, st, {'mode': 'on', 'updated': time.time(), 'credit_urls': {URLS[0]: 2}})
    raw = type(st).claim.__get__(st)                   # eval_q35's State.claim, unwrapped
    w.state.claim = R.SchedRouter(w, tmp_path / 'state.json', 'ns', log=logs.append).wrap(
        lambda *a: calls.append(a) or None, raw)       # the old rules' claim must not be called
    assert w.state.claim('w1', {f't{i}' for i in range(40)}) is not None and calls == []

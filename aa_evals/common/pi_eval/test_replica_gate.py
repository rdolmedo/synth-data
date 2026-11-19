"""#378 per-replica gate: a pi_eval worker pauses the run (admission.json `paused`, the dispatcher's "gate-paused") only
while every replica's admission gate holds new conversations; until then it claims in the run's order and pins an
attempt whose hashed replica holds to a clear one, for the attempt's whole life."""
import json
import sys

from aa_evals.common.pi_eval import worker as W
from aa_evals.common.pi_eval.test_idle_exit import make

URLS = ['http://r0.invalid/v1', 'http://r1.invalid/v1']
KEYS = [f't{i}__a{a}__r0' for i in range(8) for a in range(2)]


def setup(tmp_path, monkeypatch, queued):
    w = make(tmp_path, max_starts=100, items=[(f't{i}', a) for i in range(8) for a in range(2)])
    w.config = {'agent': 'pi', 'model_spec': 'serving:fake', 'served': 'm', 'max_in_flight': 100}
    w.urls, w.held_urls, w.pinned = [], frozenset(), {}
    monkeypatch.setattr(W.registry, 'alive', lambda *a: (True, {'urls': URLS, 'served': 'm'}, ''))
    monkeypatch.setattr(W, 'alive', lambda *a: True)
    monkeypatch.setattr(W, 'gate_queued_new', lambda u: queued[URLS.index(u)])
    return w


def admission(w):
    return json.loads((w.run / 'queue' / 'admission.json').read_text())


def test_one_replica_holds_not_paused_and_repins(tmp_path, monkeypatch):
    w = setup(tmp_path, monkeypatch, [0, 16])
    w.refresh()
    adm = admission(w)
    assert not adm['paused'] and adm['reason'] is None and adm['gate_held_replicas'] == [1] and adm['replicas'] == 2
    assert w.hold == '' and w.held_urls == {URLS[1]} and 'replica(s) 1 of 2' in w.gate_note
    assert {w.hashed_url(k) for k in KEYS} == set(URLS)          # both replicas get attempts by hash
    on1 = [k for k in KEYS if w.hashed_url(k) == URLS[1]]
    on0 = [k for k in KEYS if w.hashed_url(k) == URLS[0]]
    for k in KEYS:
        (w.run / 'queue' / 'claims' / k).mkdir()
        assert w.pin(k) == (URLS[0] if k in on1 else None)      # re-pinned, or its hashed replica (clear)
        assert w.url_for(k) == URLS[0]                           # every new attempt runs on the clear replica
    # the re-pinned ones say so in their claim dir, for the run's other workers (turn_gate_host)
    assert all(json.loads((w.run / 'queue' / 'claims' / k / 'replica.json').read_text())['url'] == URLS[0]
               for k in on1)
    assert not any((w.run / 'queue' / 'claims' / k / 'replica.json').exists() for k in on0)
    other = make(tmp_path / 'other')
    other.run, other.urls = w.run, URLS
    assert all(other.url_for(k) == URLS[0] for k in on1)
    # a pin lasts the attempt's life: the gate clearing (or the other replica holding) does not move it
    monkeypatch.setattr(W, 'gate_queued_new', lambda u: [5, 0][URLS.index(u)])
    w.refresh()
    assert w.held_urls == {URLS[0]} and all(w.url_for(k) == URLS[0] for k in KEYS)


def test_all_replicas_hold_pauses(tmp_path, monkeypatch):
    w = setup(tmp_path, monkeypatch, [3, 5])
    w.refresh()
    adm = admission(w)
    assert adm['paused'] and adm['reason'] == 'gate holds 8 new conversation(s)' and adm['gate_held_replicas'] == [0, 1]
    assert w.held_urls == frozenset() and w.hold


def test_gate_clear_or_no_gate_keeps_the_hash(tmp_path, monkeypatch):
    for q in ([0, 0], [None, None], [None, 0]):
        w = setup(tmp_path / str(q), monkeypatch, q)
        w.refresh()
        adm = admission(w)
        assert not adm['paused'] and adm['reason'] is None and adm['gate_held_replicas'] == []
        assert all(w.pin(k) is None and w.url_for(k) == w.hashed_url(k) for k in KEYS)


def test_single_replica_unchanged(tmp_path, monkeypatch):
    w = setup(tmp_path, monkeypatch, [2])
    monkeypatch.setattr(W.registry, 'alive', lambda *a: (True, {'urls': URLS[:1], 'served': 'm'}, ''))
    w.refresh()
    assert admission(w)['paused'] and admission(w)['reason'] == 'gate holds 2 new conversation(s)'


def test_claim_wrapper_pins_in_claim_order(tmp_path, monkeypatch):
    """The wrapper PiEvalWorker.__init__ installs around eval_q35 State.claim: claims stay in the run's order."""
    sys.path.insert(0, str(W.REPO / 'eval_q35'))
    import queue_state
    run = tmp_path / 'run'
    q = run / 'queue'
    for sub in ('claims', 'done', 'retry'):
        (q / sub).mkdir(parents=True)
    items = [(f't{i}', 0) for i in range(6)]
    (q / 'items.txt').write_text(''.join(f'{t} {a}\n' for t, a in items))
    (q / 'admission.json').write_text(json.dumps({'epoch': 1e12, 'paused': False, 'limit': 100, 'burst': 100,
                                                  'window_sec': 60}))
    monkeypatch.setattr(queue_state.time, 'time', lambda: 1e12)
    w = make(tmp_path / 'w')
    w.run, w.urls, w.held_urls, w.pinned = run, URLS, frozenset({URLS[1]}), {}
    claim = w.replica_claim(queue_state.State(run).claim)
    got = [claim('w', {t for t, _ in items}) for _ in range(6)]
    assert got == [f'{t}__a0__r0' for t, _ in items]
    assert all(w.url_for(k) == URLS[0] for k in got)


def test_pin_never_raises(tmp_path, monkeypatch):
    """The attempt is already claimed when pin runs: an error leaves it on its hashed replica, the claim stands."""
    w = setup(tmp_path, monkeypatch, [0, 16])
    w.refresh()
    k = next(k for k in KEYS if w.hashed_url(k) == URLS[1])
    w.held_urls = frozenset({'http://gone.invalid/v1'})       # refresh swapped the replica list meanwhile
    assert w.pin(k) is None and w.url_for(k) == URLS[1]
    w.held_urls = frozenset({URLS[1]})

    def boom(*a, **kw):
        raise OSError('lustre')
    monkeypatch.setattr(W.q35, 'atomic', boom)
    claim = w.replica_claim(lambda worker, avail: k)
    assert claim('w', set()) == k                               # pinned in memory, the record failed: logged
    assert w.url_for(k) == URLS[0]

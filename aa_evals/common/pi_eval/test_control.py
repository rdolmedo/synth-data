"""control.update: read-merge-write of a run's live control.json (#451 workstream 2)."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aa_evals.common.pi_eval import control   # noqa: E402


def test_keeps_keys_other_writers_set(tmp_path):
    # a dispatcher's credit, a fence and a hand edit survive a feeder's share update (a whole-file write erased them)
    (tmp_path / 'control.json').write_text(json.dumps({'max_starts': 40, 'fence': 120, 'note': 'by hand'}))
    out = control.update(tmp_path, {'max_in_flight': 64, 'burst': 32, 'why': 'feed'})
    assert json.loads((tmp_path / 'control.json').read_text()) == out == {
        'max_starts': 40, 'fence': 120, 'note': 'by hand', 'max_in_flight': 64, 'burst': 32, 'why': 'feed'}
    assert control.update(tmp_path, {}, remove=['note']) == {k: v for k, v in out.items() if k != 'note'}
    assert not list(tmp_path.glob('.control*'))   # no lock or temp file left


def test_missing_file_starts_empty_and_an_invalid_one_is_kept_aside(tmp_path, capsys):
    assert control.update(tmp_path, {'max_in_flight': 1}) == {'max_in_flight': 1}
    (tmp_path / 'control.json').write_text('{"max_starts": 3,')   # a hand edit saved half-way
    assert control.update(tmp_path, {'burst': 2}) == {'burst': 2}
    bad = list(tmp_path.glob('control.json.bad-*'))
    assert len(bad) == 1 and bad[0].read_text() == '{"max_starts": 3,' and 'does not parse' in capsys.readouterr().err


def writer(run, key, n):
    load = control._load

    def slow_load(path):   # widen the read -> write window: without the lock, updates would be lost
        value = load(path)
        time.sleep(0.01)
        return value
    control._load = slow_load
    for i in range(n):
        control.update(run, {key: i})


def test_concurrent_writers_lose_no_key(tmp_path):
    # rounds.json, 11-16: a writer holding its copy across a launch erased an entry another process had added
    procs = [multiprocessing.Process(target=writer, args=(tmp_path, f'k{j}', 25)) for j in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    assert json.loads((tmp_path / 'control.json').read_text()) == {f'k{j}': 24 for j in range(4)}


def test_a_dead_holders_lock_is_broken(tmp_path):
    lock = tmp_path / '.control.lock'
    lock.mkdir()
    (lock / 'owner').write_text('dead.1')
    old = time.time() - control.LOCK_STALE_SEC - 5
    os.utime(lock, (old, old))
    t = time.monotonic()
    assert control.update(tmp_path, {'a': 1}, wait=5) == {'a': 1}
    assert time.monotonic() - t < 2 and not lock.exists() and not list(tmp_path.glob('.control.lock*'))


def test_a_live_lock_times_out_and_still_merges(tmp_path):
    (tmp_path / 'control.json').write_text('{"max_starts": 9}')
    (tmp_path / '.control.lock').mkdir()
    (tmp_path / '.control.lock' / 'owner').write_text('someone.else')
    assert control.update(tmp_path, {'a': 1}, wait=0.2) == {'max_starts': 9, 'a': 1}
    assert (tmp_path / '.control.lock' / 'owner').read_text() == 'someone.else'   # not ours: left alone


def test_a_holder_never_removes_a_lock_it_does_not_own(tmp_path):
    lock = tmp_path / '.control.lock'
    assert control._lock(lock, 'mine', 1)
    (lock / 'owner').write_text('theirs')   # broken as stale and retaken by another writer meanwhile
    control._unlock(lock, 'mine')
    assert (lock / 'owner').read_text() == 'theirs'
    control._unlock(lock, 'theirs')
    assert not lock.exists()


def test_a_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(control.os, 'replace', lambda a, b: (_ for _ in ()).throw(OSError(122, 'quota')))
    with pytest.raises(OSError):
        control.update(tmp_path, {'a': 1})
    assert not list(tmp_path.glob('.control*'))


def test_adaptive_write_keeps_other_keys(tmp_path):
    from aa_evals.common.pi_eval import adaptive
    (tmp_path / 'control.json').write_text(json.dumps({'max_starts': 12, 'note': 'by hand'}))
    adaptive.write(str(tmp_path), 7, 'adaptive T=8')
    assert json.loads((tmp_path / 'control.json').read_text()) == {
        'max_starts': 12, 'note': 'by hand', 'max_in_flight': 7, 'burst': 4, 'why': 'adaptive T=8'}


def make_queue(run, items, claims=(), done=(), retries=(), stale=()):
    q = run / 'queue'
    for d in ('claims', 'done', 'retry'):
        (q / d).mkdir(parents=True, exist_ok=True)
    (q / 'items.txt').write_text(''.join(f'{t} 0\n' for t in items))
    old = time.time() - 3600
    for k in claims:
        (q / 'claims' / k).mkdir()
        (q / 'claims' / k / 'claim.json').write_text('{"worker": "w1"}')
        if k in stale:
            for p in (q / 'claims' / k / 'claim.json', q / 'claims' / k):
                os.utime(p, (old, old))
    for k in done:
        (q / 'done' / f'{k}.json').write_text('{}')
    for k in retries:
        (q / 'retry' / f'{k}.json').write_text('{}')


def test_adaptive_state_counts_live_claims_and_claimable_work(tmp_path):
    from aa_evals.common.pi_eval import adaptive
    make_queue(tmp_path, ['a', 'b', 'c', 'd'], claims=['a__a0__r0', 'b__a0__r0', 'c__a0__r0', 'c__a0__r1'],
               done=['c__a0__r0'], retries=['c__a0__r1', 'b__a0__r1'], stale=['b__a0__r0'])
    # live: a r0, c r1; dead: b r0 (no heartbeat); unclaimed: d; unclaimed retry: b r1 (c's r1 retry file is claimed)
    assert adaptive.state(str(tmp_path)) == (2, 3)
    (tmp_path / 'STOP_WORKERS').write_text('stop\n')
    assert adaptive.state(str(tmp_path)) == (2, 0)


def test_a_stopped_run_whose_last_worker_died_drains(tmp_path):
    # before: a dead worker's open claim counted as in flight forever, so "all runs drained" never came
    from aa_evals.common.pi_eval import adaptive
    make_queue(tmp_path, ['a'], claims=['a__a0__r0'], stale=['a__a0__r0'])
    (tmp_path / 'STOP_WORKERS').write_text('stop\n')
    assert adaptive.state(str(tmp_path)) == (0, 0)


def test_adaptive_state_raises_on_an_unreadable_queue(tmp_path):
    # never (0, 0): that wrote max_in_flight 0 and ended the limiter as "all runs drained" on a /data EIO
    import pytest
    from aa_evals.common.pi_eval import adaptive
    (tmp_path / 'queue' / 'claims').mkdir(parents=True)
    with pytest.raises(OSError):
        adaptive.state(str(tmp_path))

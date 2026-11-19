"""The one pending-work helper and restart-safe state (#451: drain checks, read-merge-write, overlapping controllers)."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from tbsched import work
from tbsched.state import Lease, StateDir


def make_run(root, items, claims=(), done=(), retry=(), workers=None, old=()):
    q = Path(root) / 'queue'
    for d in ('claims', 'done', 'retry'):
        (q / d).mkdir(parents=True, exist_ok=True)
    (q / 'items.txt').write_text(''.join(f'{t} {a}\n' for t, a in items))
    for k in claims:
        (q / 'claims' / k).mkdir()
        (q / 'claims' / k / 'claim.json').write_text(json.dumps({'worker': (workers or {}).get(k, 'w1')}))
        (q / 'claims' / k / 'heartbeat').write_text('1\n')
    for k in old:
        for p in [q / 'claims' / k, *(q / 'claims' / k).iterdir()]:
            os.utime(p, (time.time() - 3600, time.time() - 3600))
    for k in done:
        (q / 'done' / f'{k}.json').write_text('{}')
    for k in retry:
        (q / 'retry' / f'{k}.json').write_text('{}')
    return Path(root)


class Pending(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_claimed_retry_files_are_history_not_work(self):
        # AB finishing r3 waited ~50 min on 31 retry files that had already been claimed
        run = make_run(self.tmp.name, [('t1', 0), ('t2', 0)], claims=['t1__a0__r0', 't2__a0__r0', 't1__a0__r1'],
                       done=['t1__a0__r0', 't2__a0__r0', 't1__a0__r1'], retry=['t1__a0__r1'])
        p = work.pending(run)
        self.assertEqual((p.claimable, p.live, p.drained), (0, 0, True))

    def test_claims_of_gone_workers_are_claimable_not_live(self):
        # AB finishing r1: 121 retries held by workers that no longer existed; the round driver waited ~2.5 h
        run = make_run(self.tmp.name, [('t1', 0), ('t2', 0), ('t3', 0)], claims=['t1__a0__r0', 't2__a0__r0'],
                       workers={'t1__a0__r0': '17.0-g1', 't2__a0__r0': '18.0-g2'})
        p = work.pending(run, worker_alive=lambda w: w.startswith('17.'))
        self.assertEqual((p.unclaimed, p.live, p.dead, p.claimable), (1, 1, 1, 2))

    def test_silent_claims_are_dead(self):
        run = make_run(self.tmp.name, [('t1', 0)], claims=['t1__a0__r0'], old=['t1__a0__r0'])
        self.assertEqual(work.pending(run).dead, 1)

    def test_unclaimed_retries_and_items(self):
        run = make_run(self.tmp.name, [('t1', 0), ('t2', 0), ('t3', 0)], claims=['t1__a0__r0'], done=['t1__a0__r0'],
                       retry=['t1__a0__r1'])
        p = work.pending(run)
        self.assertEqual((p.unclaimed, p.retries, p.claimable, p.drained), (2, 1, 3, False))

    def test_an_unreadable_queue_is_unknown_never_drained(self):
        p = work.pending(Path(self.tmp.name) / 'nope')
        self.assertEqual((p.known, p.drained), (False, False))

    def test_live_and_dead_keys(self):
        run = make_run(self.tmp.name, [('t1', 0), ('t2', 0)], claims=['t1__a0__r0', 't2__a0__r0'],
                       workers={'t1__a0__r0': '17.0-g1', 't2__a0__r0': '18.0-g2'})
        p = work.pending(run, worker_alive=lambda w: w.startswith('17.'))
        self.assertEqual((p.live_keys, p.dead_keys), ({'t1__a0__r0'}, {'t2__a0__r0'}))

    def test_unpinned_claims_hash_like_pi_eval(self):
        import hashlib
        urls = ['http://a/v1', 'http://b/v1']
        h = int(hashlib.sha256(b't7/0').hexdigest(), 16) % 2
        self.assertEqual(work.hashed(urls, 't7__a0__r3'), urls[h])
        self.assertIsNone(work.hashed([], 't7__a0__r0'))

    def test_pins(self):
        run = make_run(self.tmp.name, [('t1', 0)], claims=['t1__a0__r0'])
        (run / 'queue' / 'claims' / 't1__a0__r0' / 'replica.json').write_text(json.dumps({'url': 'http://e1'}))
        self.assertEqual(work.pins(run), {'t1__a0__r0': 'http://e1'})


class State(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_lease_holder_and_takeover_only_when_stale(self):
        a, b = Lease(self.root, stale_sec=300, owner='a'), Lease(self.root, stale_sec=300, owner='b')
        self.assertTrue(a.acquire(now=1000))
        self.assertFalse(b.acquire(now=1100))
        self.assertTrue(a.held())
        self.assertTrue(b.acquire(now=1400))      # a stopped renewing
        self.assertFalse(a.held())                # a must exit at its next tick
        self.assertFalse(a.renew(now=1401))       # and a late renew cannot take it back
        self.assertTrue(b.held())

    def test_two_newcomers_racing_for_a_stale_lease_one_wins(self):
        old = Lease(self.root, stale_sec=300, owner='old')
        old.acquire(now=0)
        b, c = Lease(self.root, stale_sec=300, owner='b'), Lease(self.root, stale_sec=300, owner='c')
        seen = c._epochs()                        # c read the epochs before b took over ...
        self.assertTrue(b.acquire(now=1000))
        c._epochs = lambda: seen                  # ... and acts on that stale view
        self.assertFalse(c.acquire(now=1000))
        self.assertTrue(b.held())
        self.assertFalse(c.held())

    def test_entries_are_separate_files_and_overrides_are_never_written(self):
        s = StateDir(self.root)
        s.put('runs', 'gdp/t2', {'name': 'gdp/t2', 'credit': {'e0': 4}})
        s.put('runs', 'ab', {'name': 'ab'})
        self.assertEqual(set(s.all('runs')), {'gdp/t2', 'ab'})
        (self.root / 'overrides' / 'runs').mkdir(parents=True)
        (self.root / 'overrides' / 'runs' / 'ab.json').write_text('{"weight": 3}')
        s.put('runs', 'ab', {'name': 'ab', 'credit': {}})
        self.assertEqual(s.override('runs', 'ab'), {'weight': 3})

    def test_events_are_whole_lines(self):
        s = StateDir(self.root)
        for i in range(50):
            s.event(kind='x', i=i)
        lines = (self.root / 'events.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(x)['i'] for x in lines], list(range(50)))


if __name__ == '__main__':
    unittest.main()

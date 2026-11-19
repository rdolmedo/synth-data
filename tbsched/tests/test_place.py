"""Credits per (run, engine) and routing at claim (#451 lessons 2, 3, 4, 8 and the parallel-round hash lesson)."""
import collections
import hashlib
import unittest

from tbsched import place
from tbsched.place import Engine, Run, allocate, route, withdraw


def engines(*specs):
    return {n: Engine(n, g, lim, single) for n, g, lim, single in specs}


class Allocate(unittest.TestCase):
    def test_credit_goes_where_room_is(self):
        es = engines(('e0', 'g', 100, 0), ('e1', 'g', 100, 0))
        r = Run('r', 'agentic', ['e0', 'e1'], demand=500, live={'e0': 90, 'e1': 20})
        got = allocate([r], es, {'burst_min': 1000})['r']
        self.assertEqual(got, {'e0': 10, 'e1': 80})     # both engines end at their limit

    def test_unspent_credit_counts_as_used(self):
        es = engines(('e0', 'g', 100, 0))
        r = Run('r', 'agentic', ['e0'], demand=500, credit={'e0': 100}, starts={'e0': 40}, live={'e0': 40})
        self.assertEqual(allocate([r], es)['r'], {})     # 40 live + 60 unspent = 100

    def test_never_more_than_the_claimable_work(self):
        es = engines(('e0', 'g', 100, 0))
        r = Run('r', 'agentic', ['e0'], demand=7)
        self.assertEqual(allocate([r], es)['r'], {'e0': 7})

    def test_weights_split_shared_room(self):
        es = engines(('e0', 'g', 90, 0))
        a, b = Run('a', 'agentic', ['e0'], 500, weight=2), Run('b', 'agentic', ['e0'], 500, weight=1)
        got = allocate([a, b], es, {'burst_min': 1000})
        self.assertEqual((got['a']['e0'], got['b']['e0']), (60, 30))

    def test_longest_first_fills_in_priority_order(self):
        es = engines(('e0', 'g', 50, 0))
        a, b = Run('a', 'agentic', ['e0'], 40, priority=10), Run('b', 'agentic', ['e0'], 40, priority=99)
        got = allocate([a, b], es, {'order': 'longest_first', 'burst_min': 1000})
        self.assertEqual((got['b'], got['a']), ({'e0': 40}, {'e0': 10}))

    def test_burst_is_per_group_and_shared_by_runs(self):
        # 11-15: three replay rounds launched together put 283 cold prefills in one engine's queue
        es = engines(('e0', 'g', 200, 0), ('e1', 'g', 200, 0))
        rs = [Run(f'r{k}', 'agentic', ['e0', 'e1'], 500) for k in range(3)]
        got = allocate(rs, es, {'burst_frac': 0.25})
        self.assertEqual(sum(n for g in got.values() for n in g.values()), 100)

    def test_replays_have_their_own_cap_and_burst_per_engine(self):
        es = engines(('e0', 'g', 300, 0))
        r = Run('fin', 'replay', ['e0'], 500, live={'e0': 20})
        self.assertEqual(allocate([r], es)['fin'], {'e0': 4})            # replay_per_engine 24
        r2 = Run('fin', 'replay', ['e0'], 500)
        self.assertEqual(allocate([r2], es)['fin'], {'e0': 8})           # replay_burst 8 per tick

    def test_single_turn_uses_the_single_cap_not_the_session_limit(self):
        es = engines(('e0', 'g', 10, 50))
        a = Run('pi', 'agentic', ['e0'], 100, live={'e0': 10})
        s = Run('hle', 'single', ['e0'], 100)
        got = allocate([a, s], es, {'burst_min': 1000})
        self.assertEqual((got['pi'], got['hle']), ({}, {'e0': 50}))

    def test_a_closed_engine_gets_nothing(self):
        es = engines(('e0', 'g', 100, 0), ('e1', 'g', 100, 0))
        es['e0'].open = False
        got = allocate([Run('r', 'agentic', ['e0', 'e1'], 500)], es, {'burst_min': 1000})['r']
        self.assertEqual(got, {'e1': 100})


class Withdraw(unittest.TestCase):
    def test_a_cut_limit_takes_back_unspent_credit_at_once(self):
        es = engines(('e0', 'g', 50, 0))
        r = Run('r', 'agentic', ['e0'], 500, credit={'e0': 120}, starts={'e0': 40}, live={'e0': 40})
        self.assertEqual(withdraw([r], es), {'r': {'e0': 70}})            # 40 live + 10 unspent = 50

    def test_a_closed_engine_loses_all_unspent_credit(self):
        es = engines(('e0', 'g', 100, 0))
        es['e0'].open = False
        r = Run('r', 'agentic', ['e0'], 500, credit={'e0': 30}, starts={'e0': 10}, live={'e0': 10})
        self.assertEqual(withdraw([r], es), {'r': {'e0': 20}})

    def test_credit_beyond_the_work_left_is_taken_back(self):
        es = engines(('e0', 'g', 100, 0), ('e1', 'g', 100, 0))
        r = Run('r', 'agentic', ['e0', 'e1'], 5, credit={'e0': 10, 'e1': 4})
        self.assertEqual(withdraw([r], es), {'r': {'e0': 9}})


class Floors(unittest.TestCase):
    def test_credit_never_below_starts(self):
        # lost state or a run hooked mid-way: 300 starts, no credit -> it must not wait 300/burst ticks
        es = engines(('e0', 'g', 400, 0))
        r = Run('r', 'agentic', ['e0'], 500, credit={}, starts={'e0': 300}, live={'e0': 50})
        self.assertEqual(r.credit, {'e0': 300})
        self.assertEqual(allocate([r], es, {'burst_min': 1000})['r'], {'e0': 350})

    def test_phantom_credit_on_a_dropped_engine_is_withdrawn(self):
        es = engines(('e0', 'g', 100, 0), ('e1', 'h', 100, 0))
        r = Run('r', 'agentic', ['e1'], 500, credit={'e0': 80})            # e0's group was removed from r
        self.assertEqual(withdraw([r], es), {'r': {'e0': 80}})


class Route(unittest.TestCase):
    def test_the_engine_with_most_unspent_credit(self):
        self.assertEqual(route({'e0': 3, 'e1': 40}, 'ns', 't', 0), 'e1')
        self.assertIsNone(route({'e0': 0}, 'ns', 't', 0))

    def test_ties_are_spread_by_the_hash(self):
        picks = collections.Counter(route({'e0': 10, 'e1': 10}, 'ns', f't{i}', 0) for i in range(1000))
        self.assertTrue(400 < picks['e0'] < 600, picks)

    def test_parallel_rounds_of_a_task_do_not_share_an_engine_by_construction(self):
        # 11-15: engine = hash(task, attempt) with attempt 0 in every round -> a task's rounds all on one engine
        old = lambda task: int(hashlib.sha256(f'{task}/0'.encode()).hexdigest(), 16) % 2   # noqa: E731
        same_old = same_new = 0
        for i in range(300):
            same_old += len({old(f't{i}') for _ in range(3)}) == 1
            same_new += len({route({'e0': 5, 'e1': 5}, f'tbsci-r{k}', f't{i}', 0) for k in (2, 3, 4)}) == 1
        self.assertEqual(same_old, 300)
        self.assertLess(same_new, 0.4 * 300)     # ~25% expected by chance with 2 engines and 3 rounds


if __name__ == '__main__':
    unittest.main()

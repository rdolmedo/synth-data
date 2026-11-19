"""Worker and engine-group lifecycle (#451 lessons 6, 7, 8, 9 and the 11-15 backfill / steal / release lessons)."""
import unittest

from tbsched.lifecycle import GroupView, RunView, Worker, plan_groups, plan_workers


def ws(n, running=4, **kw):
    return [Worker(f'{i}.0', 'running', running=running, hb_age=10, **kw) for i in range(n)]


class Workers(unittest.TestCase):
    def test_sized_to_live_sessions_plus_spendable_credit(self):
        self.assertEqual(plan_workers(0, 96, 4000, 16, [])['add'], 6)
        self.assertEqual(plan_workers(40, 30, 4000, 16, ws(3))['add'], 2)

    def test_a_stale_limit_cannot_over_provision(self):
        # TB-Sci-lite p0: limit 207 left from a ramp, 40 live: workers follow the credit (bounded by the per-tick
        # burst), not the old limit (a naive backfill would have taken the registry to ~440 sessions)
        self.assertEqual(plan_workers(40, 16, 4000, 16, ws(3))['add'], 1)

    def test_a_black_hole_worker_is_recycled_once_idle_and_backfilled(self):
        w = ws(4)
        w[1].local_failures, w[1].successes, w[1].running = 40, 14, 0
        w[2].local_failures, w[2].successes = 3, 0            # below the threshold
        w[3].local_failures, w[3].successes = 6, 2            # still runs 4 sessions: removing it would kill them
        plan = plan_workers(48, 16, 4000, 16, w)
        self.assertEqual((plan['recycle'], plan['recycle_wait']), (['1.0'], ['3.0']))
        self.assertEqual(plan['add'], 1)                       # 64 wanted, 3 x 16 kept: one replaces it

    def test_a_drained_run_stops_and_gets_nothing(self):
        plan = plan_workers(0, 0, 0, 16, ws(2, running=0))
        self.assertEqual((plan['add'], plan['stop']), (0, True))
        self.assertEqual(plan_workers(10, 0, 0, 16, ws(2))['add'], 0)   # nothing claimable: no new workers

    def test_idle_surplus_workers_are_trimmed_after_a_grace(self):
        w = ws(5, running=0, idle_for=600)
        w[0].running = 10
        plan = plan_workers(10, 0, 50, 16, w)
        self.assertEqual(len(plan['remove']), 4)
        w2 = ws(5, running=0, idle_for=60)
        w2[0].running = 10
        self.assertEqual(plan_workers(10, 0, 50, 16, w2)['remove'], [])

    def test_claimable_work_with_no_worker_gets_one(self):
        # dead claims are claimable, but only a worker reaps them (AB r1 sat ~2.5 h with no worker left)
        self.assertEqual(plan_workers(0, 0, 121, 16, [])['add'], 1)

    def test_an_unknown_queue_gets_no_action(self):
        plan = plan_workers(0, 0, 0, 16, ws(2, running=0), known=False)
        self.assertEqual((plan['add'], plan['stop'], plan['remove']), (0, False, []))

    def test_trim_keeps_one_worker_while_there_is_work(self):
        # work but no credit (engines full, a gate hold): trimming the last worker would re-add it next tick
        plan = plan_workers(0, 0, 50, 16, ws(1, running=0, idle_for=400))
        self.assertEqual((plan['remove'], plan['add']), ([], 0))
        plan = plan_workers(0, 0, 50, 16, ws(3, running=0, idle_for=400))
        self.assertEqual(len(plan['remove']), 2)

    def test_queued_jobs_count_as_capacity(self):
        w = [Worker('1.0', 'idle'), Worker('2.0', 'idle')]
        self.assertEqual(plan_workers(0, 32, 4000, 16, w)['add'], 0)


def group(name, sessions=0, running=0.0, limit=200, status='active'):
    return GroupView(name, 'super', [f'{name}-e0', f'{name}-e1'], status, sessions, running, limit)


class Groups(unittest.TestCase):
    def test_a_hungry_group_joins_the_largest_backlog_once_per_cooldown(self):
        # m5l-ab sat at 27-37 running/engine for 20+ min with a few items left; critpt2 stole twice in 3 min
        gs = {'a': group('a', 60, 30), 'b': group('b', 200, 90), 'c': group('c', 200, 90)}
        rs = {'small': RunView('small', 'super', 'agentic', {'a': 1}, 30, 60),
              'gdp': RunView('gdp', 'super', 'agentic', {'b': 1}, 4000, 200, session_sec=1800),
              'tb': RunView('tb', 'super', 'agentic', {'c': 1}, 900, 200, session_sec=2400)}
        st = {}
        self.assertEqual(plan_groups(gs, rs, st, 0), [('join', 'a', 'gdp')])
        self.assertEqual([a for a in plan_groups(gs, rs, st, 300) if a[0] == 'join'], [])
        self.assertEqual([a for a in plan_groups(gs, rs, st, 601) if a[0] == 'join'], [('join', 'a', 'gdp')])

    def test_no_steal_from_another_model_or_a_tiny_backlog(self):
        gs = {'a': group('a'), 'b': group('b', 100, 80)}
        rs = {'x': RunView('x', 'qwen', 'agentic', {'b': 1}, 4000), 'y': RunView('y', 'super', 'agentic', {'b': 1}, 20)}
        self.assertFalse([a for a in plan_groups(gs, rs, {}, 0) if a[0] == 'join'])

    def test_end_game_drains_the_least_loaded_group_when_the_others_hold_the_rest(self):
        gs = {'a': group('a', 10, 5), 'b': group('b', 120, 60, limit=300)}
        rs = {'r': RunView('r', 'super', 'agentic', {'a': 1, 'b': 1}, 100, 130)}
        self.assertIn(('drain', 'a'), plan_groups(gs, rs, {}, 0))
        rs['r'].claimable = 400
        self.assertNotIn(('drain', 'a'), plan_groups(gs, rs, {}, 0))

    def test_no_steal_in_the_end_game_and_no_drain_right_after_a_join(self):
        gs = {'a': group('a', 10, 5), 'b': group('b', 120, 60, limit=300)}
        rs = {'r': RunView('r', 'super', 'agentic', {'b': 1}, 100, 120)}
        self.assertFalse([a for a in plan_groups(gs, rs, {}, 0) if a[0] == 'join'])     # end game: consolidate
        rs['r'].claimable = 1000
        st = {}
        self.assertEqual([a for a in plan_groups(gs, rs, st, 0) if a[0] == 'join'], [('join', 'a', 'r')])
        rs['r'].groups['a'] = 1.0
        rs['r'].claimable = 100                                                       # end game one tick later
        self.assertNotIn(('drain', 'a'), plan_groups(gs, rs, st, 60))

    def test_busy_or_unread_groups_are_never_released(self):
        gs = {'a': group('a'), 'b': GroupView('b', 'super', [], 'active')}
        gs['a'].busy = True                     # another client's requests, or metrics we cannot read
        rs = {'r': RunView('r', 'super', 'agentic', {'a': 1, 'b': 1}, 0, 0, steal=False)}
        st = {}
        for t in range(5):
            self.assertFalse([a for a in plan_groups(gs, rs, st, 60 * t) if a[0] in ('release', 'stop_run')])

    def test_never_drain_a_group_a_run_depends_on(self):
        gs = {'a': group('a', 10, 5), 'b': group('b', 120, 60, limit=300)}
        rs = {'only_a': RunView('only_a', 'super', 'agentic', {'a': 1}, 50, 10, steal=False),
              'b': RunView('b', 'super', 'agentic', {'b': 1}, 50, 120, steal=False)}
        self.assertFalse([a for a in plan_groups(gs, rs, {}, 0) if a[0] == 'drain'])

    def test_release_after_idle_ticks_then_stop_the_runs_left_without_engines(self):
        gs = {'a': group('a'), 'b': group('b', 50, 30)}
        rs = {'done': RunView('done', 'super', 'agentic', {'a': 1}, 0, 0, steal=False),
              'busy': RunView('busy', 'super', 'agentic', {'b': 1}, 10, 50, steal=False)}
        st = {}
        self.assertFalse([a for a in plan_groups(gs, rs, st, 0) if a[0] == 'release'])
        acts = plan_groups(gs, rs, st, 60)
        self.assertIn(('release', 'a', False), acts)
        self.assertIn(('stop_run', 'done'), acts)
        self.assertNotIn(('stop_run', 'busy'), acts)

    def test_kept_groups_and_groups_with_work_stay(self):
        gs = {'a': group('a')}
        rs = {'r': RunView('r', 'super', 'agentic', {'a': 1}, 5, 0)}
        st = {}
        for t in range(5):
            self.assertFalse([a for a in plan_groups(gs, rs, st, 60 * t) if a[0] == 'release'])
        rs['r'].claimable = 0
        for t in range(5):
            self.assertFalse([a for a in plan_groups(gs, rs, st, 60 * t, keep={'a'}) if a[0] == 'release'])

    def test_tail_cut_only_where_the_pipeline_allows_it(self):
        gs = {'a': group('a', sessions=3)}
        rs = {'r': RunView('r', 'super', 'agentic', {'a': 1}, 0, 3, steal=False)}
        for tail, want in (('wait', False), ('cut', True)):
            st, got = {}, False
            for t in range(10):
                got |= ('release', 'a', True) in plan_groups(gs, rs, st, 60 * t, {'tail': tail})
            self.assertEqual(got, want, tail)


if __name__ == '__main__':
    unittest.main()

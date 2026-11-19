"""Per-engine limits: load-aware signals (#451 lessons 1, 2, 10 and the 11-15 controller lessons)."""
import unittest

from tbsched import control
from tbsched.control import AIMD, Obs, aimd, judge

P = dict(AIMD)


class Signals(unittest.TestCase):
    def test_a_cold_engine_stale_hit_window_is_no_evidence(self):
        # flat-d after single-turn SciCode: 23-33% hit at 7 running -> not pressure
        self.assertEqual(judge(Obs(running=7, hit=0.25, hit_tokens=5e6), P)[0], 'calm')

    def test_a_mixed_engine_uses_queue_and_kv_only(self):
        # m5l-ab2 with SciCode co-located: 58% hit at 149 running, nothing waiting, no thrash
        o = Obs(running=149, hit=0.58, hit_tokens=5e6, kv=0.6, mixed=True)
        self.assertNotEqual(judge(o, P)[0], 'pressure')
        self.assertEqual(judge(Obs(running=149, waiting=40, mixed=True), P), ('pressure', ['queue']))

    def test_a_loaded_agentic_engine_with_a_low_hit_is_pressure(self):
        self.assertEqual(judge(Obs(running=145, hit=0.04, hit_tokens=5e6), P), ('pressure', ['prefix_hit']))

    def test_a_launch_wave_is_no_evidence(self):
        # first turns are cold by construction: 96 new sessions read 0.17
        o = Obs(running=60, hit=0.17, hit_tokens=5e6, sessions=96, fresh=96)
        self.assertIsNone(control.hit_evidence(o, P))
        self.assertNotEqual(judge(o, P)[0], 'pressure')

    def test_client_side_reuse_counts_even_on_a_mixed_engine(self):
        o = Obs(running=150, hit=0.9, hit_tokens=5e6, mixed=True, reuse=0.3)
        self.assertEqual(judge(o, P), ('pressure', ['prefix_hit']))

    def test_too_few_tokens_in_the_window_is_no_evidence(self):
        self.assertIsNone(control.hit_evidence(Obs(running=100, hit=0.1, hit_tokens=1000), P))

    def test_kv_and_preemption_are_pressure_anywhere(self):
        self.assertEqual(judge(Obs(running=5, kv=0.99), P)[1], ['kv'])
        self.assertEqual(judge(Obs(running=5, preempted=True), P)[1], ['preemption'])


class Aimd(unittest.TestCase):
    def test_pressure_on_one_engine_never_cuts_its_sibling(self):
        # fin-pi 11-15: 145 running at 4% hit on one engine, 27 at 91% on the other; the registry-level rule cut both
        hot, cold = {'limit': 120}, {'limit': 120}
        r1 = aimd(hot, Obs(running=145, hit=0.04, hit_tokens=5e6), {}, 140, now=0)
        r2 = aimd(cold, Obs(running=27, hit=0.91, hit_tokens=5e6), {}, 118, now=0)
        self.assertEqual((r1['state'], r1['limit']), ('pressure', 102))
        self.assertEqual(r2['state'], 'calm')
        self.assertGreaterEqual(cold['limit'], 120)

    def test_raises_only_after_calm_ticks_and_only_when_binding(self):
        st = {'limit': 100}
        calm = Obs(running=40, hit=0.95, hit_tokens=5e6)
        self.assertFalse(aimd(st, calm, {}, 95, now=0)['changed'])
        self.assertTrue(aimd(st, calm, {}, 95, now=60)['changed'])
        self.assertEqual(st['limit'], 115)
        st2 = {'limit': 100}
        for t in range(5):
            aimd(st2, calm, {}, 50, now=60 * t)   # not binding: 50 of 100 used
        self.assertEqual(st2['limit'], 100)

    def test_committed_credit_makes_the_limit_bind(self):
        # replay 11-18: 80 live of 96 with ~15 credits granted and being spent: the limit binds
        calm = Obs(running=40, hit=0.95, hit_tokens=5e6)
        st = {'limit': 96}
        for t in range(3):
            aimd(st, calm, {}, 80, now=60 * t, committed=95)
        self.assertGreater(st['limit'], 96)
        st2 = {'limit': 96}
        for t in range(3):
            aimd(st2, calm, {}, 80, now=60 * t, committed=80)    # nothing started recently: not binding
        self.assertEqual(st2['limit'], 96)

    def test_a_cut_starts_from_the_sessions_and_waits_before_the_next(self):
        st = {'limit': 300}
        aimd(st, Obs(running=150, waiting=50), {}, 100, now=0)
        self.assertEqual(st['limit'], 85)            # from 100 live sessions, not from a stale 300
        aimd(st, Obs(running=150, waiting=50), {}, 85, now=60)
        self.assertEqual(st['limit'], 85)            # cut_hold_sec
        aimd(st, Obs(running=150, waiting=50), {}, 85, now=130)
        self.assertEqual(st['limit'], 72)

    def test_a_thrash_cut_learns_a_ceiling_and_reprobes_later(self):
        st = {'limit': 150}
        aimd(st, Obs(running=150, hit=0.2, hit_tokens=5e6), {}, 150, now=0)
        self.assertEqual(st['hi'], 128)
        calm = Obs(running=40, hit=0.95, hit_tokens=5e6)
        for t in range(1, 40):
            aimd(st, calm, {}, st['limit'], now=60 * t)
        self.assertEqual(st['limit'], 128)           # held below the level that thrashed
        aimd(st, calm, {}, st['limit'], now=2800)    # reprobe_sec later: one step past it
        aimd(st, calm, {}, st['limit'], now=2860)
        self.assertGreater(st['limit'], 128)

    def test_a_young_cohort_does_not_look_calm_past_its_projected_working_set(self):
        p = {'ctx_prior': 70_000}
        o = Obs(running=12, hit=0.95, hit_tokens=5e6, sessions=250, ctx_tokens=250 * 20_000, kv_tokens=19.5e6)
        self.assertEqual(judge(o, {**P, **p})[0], 'hold')      # 258 x 70k > 0.9 x 19.5M
        o.sessions, o.ctx_tokens = 200, 200 * 20_000
        self.assertEqual(judge(o, {**P, **p})[0], 'calm')

    def test_single_turn_cap_ignores_the_hit_rate(self):
        st = {'limit': 100}
        r = control.single_cap(st, Obs(running=150, hit=0.0, hit_tokens=5e6, singles=99, reuse=0.3), {}, now=0)
        self.assertNotEqual(r['state'], 'pressure')
        self.assertNotIn('hi', st)
        self.assertEqual(control.single_start(19.5e6, 86_000) // 1, 204)


class Guard(unittest.TestCase):
    """policy = "guard": raise while a step adds >= 2% generated tok/s, back on a flat step, cut on thrash evidence."""
    P = {'policy': 'guard', 'start': 100, 'settle_sec': 60, 'window_sec': 60, 'measure_sec': 120, 'reprobe_sec': 600}

    def drive(self, curve, minutes=60, hit=0.9, kv=0.99, running=190, st=None):
        st = {} if st is None else st
        for t in range(0, minutes * 60, 30):
            lim = st.get('limit', 100)
            o = Obs(running=running, kv=kv, hit=hit, hit_tokens=5e6, gen_tok_s=curve(lim))
            control.limit_step(st, o, self.P, lim, t, committed=lim)
        return st

    def test_climbs_while_throughput_grows_and_stops_at_the_knee(self):
        knee = lambda n: 10_000 * min(n, 200) / 200            # flat past 200 sessions  # noqa: E731
        st = self.drive(knee, minutes=90)
        self.assertTrue(175 <= st['limit'] <= 232, st)          # at the knee (one step either side), not at ceil
        self.assertEqual(st['guard']['phase'], 'hold')

    def test_ignores_the_gpu_kv_gauge_and_running_count(self):
        # replay 11-18: 99% KV and 190 running while throughput still grew: aimd held, the guard climbs
        grow = lambda n: 50 * n                                 # noqa: E731
        st = self.drive(grow, minutes=30)
        self.assertGreater(st['limit'], 150)
        st2 = {}
        for t in range(0, 1800, 30):
            control.limit_step(st2, Obs(running=190, kv=0.99, hit=0.9, hit_tokens=5e6, gen_tok_s=9000), {'start': 100},
                               100, t, committed=100)
        self.assertLess(st2['limit'], 100)                       # aimd cuts at the same readings (KV > 0.97)

    def test_zero_throughput_is_no_evidence(self):
        st = self.drive(lambda n: 0.0, minutes=180)
        self.assertEqual(st['limit'], 100)

    def test_noise_does_not_ratchet_the_limit_up(self):
        import random
        finals = []
        for seed in range(20):
            rng = random.Random(seed)
            knee = lambda n: 10_000 * min(n, 200) / 200 * (1 + rng.gauss(0, 0.04))   # noqa: E731
            finals.append(self.drive(knee, minutes=8 * 60)['limit'])
        self.assertLessEqual(max(finals), 290, finals)          # one probe step past the knee at most
        self.assertGreaterEqual(min(finals), 150, finals)

    def test_through_the_loops_settings_the_guard_keeps_its_own(self):
        # loop.py merges defaults itself: control.params must give the guard GUARD's caps, not aimd's
        p = control.params({'policy': 'guard'})
        self.assertEqual((p['kv_max'], p['ceil'], p['run_max']), (control.GUARD['kv_max'], 400, 10 ** 9))
        st = {}
        r = control.limit_step(st, Obs(running=190, kv=0.99, hit=0.9, hit_tokens=5e6, gen_tok_s=9000), p, 100, 0,
                               committed=100)
        self.assertNotEqual(r['state'], 'pressure')

    def test_cuts_on_thrash_evidence_and_holds(self):
        st = self.drive(lambda n: 50 * n, minutes=10)
        before = st['limit']
        control.limit_step(st, Obs(running=190, hit=0.2, hit_tokens=5e6, gen_tok_s=2000), self.P, before, 10_000,
                           committed=before)
        self.assertLess(st['limit'], before)
        self.assertEqual(st['guard']['phase'], 'hold')

    def test_a_limit_that_is_not_filled_gives_no_evidence(self):
        st = {}
        for t in range(0, 3600, 30):
            control.limit_step(st, Obs(running=50, kv=0.3, hit=0.9, hit_tokens=5e6, gen_tok_s=5000), self.P, 40, t,
                               committed=40)
        self.assertEqual(st['limit'], 100)


class Metrics(unittest.TestCase):
    TEXT = '''# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="m"} 10
vllm:num_requests_running{engine="1",model_name="m"} 30
vllm:num_requests_waiting{engine="0"} 2
vllm:kv_cache_usage_perc{engine="0"} 0.5
vllm:generation_tokens_total{engine="0"} 1000
vllm:prefix_cache_queries_total{engine="0"} 2e6
vllm:prefix_cache_hits_total{engine="0"} 1.5e6
vllm:num_preemptions_total{engine="0"} 0
'''

    def test_parse_sums_series_and_filters_labels(self):
        self.assertEqual(control.parse_metrics(self.TEXT)['vllm:num_requests_running'], 40)
        two = 'vllm:kv_cache_usage_perc{engine="0"} 0.6\nvllm:kv_cache_usage_perc{engine="1"} 0.7\n'
        self.assertEqual(control.parse_metrics(two)['vllm:kv_cache_usage_perc'], 0.7)   # DP ranks: max, not 1.3
        m = control.parse_metrics(self.TEXT, {'engine': 0})
        self.assertEqual(m['vllm:num_requests_running'], 10)
        self.assertEqual(m['vllm:prefix_cache_hits_total'], 1.5e6)

    def test_a_cpu_kv_tier_counts_as_hits(self):
        # replay 11-18: GPU prefix hit ~0 at 99% KV, but the CPU tier served the prefixes (no thrash)
        def m(compute, gpu, cpu, q, h):
            return control.parse_metrics(
                f'vllm:prompt_tokens_by_source_total{{engine="0",source="local_compute"}} {compute}\n'
                f'vllm:prompt_tokens_by_source_total{{engine="0",source="local_cache_hit"}} {gpu}\n'
                f'vllm:prompt_tokens_by_source_total{{engine="0",source="external_kv_transfer"}} {cpu}\n'
                f'vllm:prefix_cache_queries_total {q}\nvllm:prefix_cache_hits_total {h}\n')
        w = control.Window(120)
        w.add(0, m(0, 0, 0, 0, 0))
        w.add(60, m(1e5, 1e5, 8e5, 1e6, 1e4))
        o = w.obs()
        self.assertAlmostEqual(o.hit, 0.9)
        self.assertEqual(o.hit_tokens, 1e6)
        w2 = control.Window(120)                                      # no by-source series: GPU prefix counters
        w2.add(0, control.parse_metrics('vllm:prefix_cache_queries_total 0\nvllm:prefix_cache_hits_total 0\n'))
        w2.add(60, control.parse_metrics('vllm:prefix_cache_queries_total 1e6\nvllm:prefix_cache_hits_total 5e5\n'))
        self.assertAlmostEqual(w2.obs().hit, 0.5)

    def test_window_counter_deltas(self):
        w = control.Window(120)
        m0 = control.parse_metrics(self.TEXT, {'engine': 0})
        m1 = dict(m0, **{'vllm:generation_tokens_total': 61_000, 'vllm:prefix_cache_queries_total': 3e6,
                         'vllm:prefix_cache_hits_total': 1.6e6, 'vllm:num_preemptions_total': 1})
        w.add(0, m0)
        w.add(60, m1)
        o = w.obs(sessions=12)
        self.assertAlmostEqual(o.gen_tok_s, 1000)
        self.assertAlmostEqual(o.hit, 0.1)
        self.assertEqual((o.hit_tokens, o.preempted, o.sessions), (1e6, True, 12))
        self.assertTrue(control.Window().obs().stale)


if __name__ == '__main__':
    unittest.main()

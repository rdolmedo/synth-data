"""Old rules vs tbsched on the same simulated load, one test per #451 lesson (tbsched/sim.py; CPU, ~1 min total).

OldPolicy is what ran on 11-15: pi_eval's hash pin, a fixed max_in_flight and start burst per run, and optionally
the M5 autoscale (registry-level calm/pressure). SchedPolicy is tbsched: per-engine aimd, credits, routing at claim,
optionally engine-group decisions, the collapse gate and the liveness watchdog.
"""
import unittest

from tbsched.sim import OldPolicy, RunSpec, SchedPolicy, World


def hit(w, e):
    c = w.engines[e].c
    return c['vllm:prefix_cache_hits_total'] / max(1.0, c['vllm:prefix_cache_queries_total'])


def finished(w):
    return all(w.claimable(r) == 0 and not w.live[r] for r in w.runs)


class Sampler:
    """Wraps a policy and samples sessions per engine every minute."""

    def __init__(self, p):
        self.p, self.samples, self.next = p, [], 0.0

    def attach(self, w):
        self.p.attach(w)

    def step(self, w, now):
        self.p.step(w, now)
        if now >= self.next:
            self.next = now + 60
            self.samples.append((now, {e: w.sessions_on(e) for e in w.engines},
                                 {e: w.sessions_on(e, 'replay') for e in w.engines}))

    @property
    def watchdog_sec(self):
        return getattr(self.p, 'watchdog_sec', None)


class Calibration(unittest.TestCase):
    def test_super_like_engine_thrashes_past_its_session_range(self):
        # measured on Super 11-15: CritPt 84 running 78% hit -> 173 running 27%
        out = {}
        for n in (100, 240):
            w = World({'g': ['e0']}, [RunSpec('c', 'critpt', 5000, ['g'], limit=n)], OldPolicy(burst=n), seed=1)
            w.run(hours=1)
            out[n] = (hit(w, 'e0'), w.engines['e0'].c['vllm:generation_tokens_total'], w.done['c'])
        self.assertGreater(out[100][0], 0.85)
        self.assertLess(out[240][0], 0.6)
        self.assertLess(out[240][2], out[100][2])      # more sessions, fewer finished: thrash


class Lessons(unittest.TestCase):
    def test_least_loaded_routing_keeps_engines_even(self):
        s = Sampler(SchedPolicy())
        w = World({'g': ['e0', 'e1']}, [RunSpec('c', 'critpt', 4000, ['g'], heavy=0.4)], s, seed=1)
        w.run(hours=1.5)
        late = [x[1] for x in s.samples if x[0] >= 1800]
        self.assertLessEqual(max(max(v.values()) - min(v.values()) for v in late), 2)

    def test_launch_wave_does_not_cut_the_limit(self):
        p = SchedPolicy()
        w = World({'g': ['e0', 'e1']}, [RunSpec('c', 'critpt', 4000, ['g'])], p, seed=1)
        w.run(hours=0.25)
        self.assertFalse([r for r in w.trace if r[7] == 'pressure'], [r for r in w.trace if r[7] == 'pressure'][:3])
        self.assertGreaterEqual(min(r[6] for r in w.trace), 96)

    def test_no_regression_near_the_knee(self):
        def run(p):
            w = World({'g': ['e0', 'e1']}, [RunSpec('c', 'critpt', 4000, ['g'], limit=320, heavy=0.4)], p, seed=1)
            return w.run(hours=2).done['c']
        self.assertGreaterEqual(run(SchedPolicy()), 0.97 * run(OldPolicy(burst=64)))

    def test_parallel_rounds_of_a_slow_suite(self):
        # TB-Sci-lite finishing r2-4 at once (11-15): three rounds over the same tasks, attempt 0 each
        def run(p):
            tasks = [f't{i}' for i in range(400)]
            runs = [RunSpec(f'r{k}', 'tbsci', 400, ['g'], namespace=f'tbsci-r{k}', tasks=tasks, heavy=0.3)
                    for k in range(3)]
            return sum(World({'g': ['e0', 'e1']}, runs, p, seed=1).run(hours=2).done.values())
        old, new, gated = run(OldPolicy()), run(SchedPolicy()), run(SchedPolicy(aimd={'ceil': 320}, gate=True))
        self.assertGreaterEqual(new, old)
        self.assertGreaterEqual(gated, 1.3 * old)

    def test_co_located_replays_are_capped_per_engine(self):
        # 11-15: finishing replays at 96/engine next to second traces collapsed an engine (hit 93% -> 0%)
        def run(p):
            runs = [RunSpec('t2', 'critpt', 3000, ['g'], namespace='t2', limit=240),
                    RunSpec('fin', 'replay', 400, ['g'], namespace='fin-r1', limit=192)]
            s = Sampler(p)
            w = World({'g': ['e0', 'e1']}, runs, s, seed=3).run(hours=1.5)
            return w.done, max(max(x[2].values()) for x in s.samples)
        old, _ = run(OldPolicy(burst=64))
        new, replays = run(SchedPolicy(start=120))
        self.assertLessEqual(replays, 24)
        self.assertGreaterEqual(new['t2'], 1.1 * old['t2'])
        self.assertEqual(new['fin'], 400)

    def test_single_turn_work_fills_a_light_pi_engine_without_cutting_it(self):
        # 11-15: 160 SciCode workers on the near-idle Pi registry m5l-ab2: 2.5k -> 5.2-7.7k tok/s
        def run(hle):
            runs = [RunSpec('ab', 'ab', 3000, ['g'], namespace='ab')] + ([RunSpec('hle', 'hle', 2000, ['g'])] if hle else [])
            p = SchedPolicy(aimd={'start': 15, 'floor': 15})
            w = World({'g': ['e0', 'e1']}, runs, p, seed=3).run(hours=1)
            return w.total_gen(), [r for r in w.trace if r[7] == 'pressure']
        alone, _ = run(False)
        mixed, pressure = run(True)
        self.assertGreater(mixed, 3 * alone)          # 4.4-4.8x here (3.11 vs 3.12 float order)
        self.assertEqual(pressure, [])

    def test_collapse_gate_recovers_an_over_admitted_engine_in_minutes(self):
        # #477 / #6 10-25: an over-admitted engine whose hit rate collapsed and stayed down for hours
        fixed = {'start': 300, 'floor': 300, 'ceil': 300}
        out = {}
        for gate in (False, True):
            w = World({'g': ['e0']}, [RunSpec('c', 'tbsci', 3000, ['g'], heavy=0.3)], SchedPolicy(aimd=fixed, gate=gate),
                      seed=2).run(hours=2.5)
            low = [r for r in w.trace if r[0] > 900 and r[4] is not None and r[4] < 0.5]
            out[gate] = (len(low), w.done['c'], w.gates.get('e0'))
        self.assertGreaterEqual(out[False][0], 45)        # minutes with hit < 0.5 after the launch
        self.assertLessEqual(out[True][0], 25)
        self.assertGreaterEqual(out[True][1], 2 * out[False][1])
        self.assertTrue(out[True][2].triggers)

    def test_stealing_and_release_instead_of_splits(self):
        # GDPval/TB-Sci-lite sat on one registry each while others emptied (lesson 8); release when idle (lesson 9)
        def run(p):
            runs = [RunSpec('big', 'critpt', 900, ['g1'], namespace='big', limit=240),
                    RunSpec('small', 'critpt', 60, ['g2'], namespace='small', limit=240)]
            w = World({'g1': ['a0', 'a1'], 'g2': ['b0', 'b1']}, runs, p, seed=4)
            return w.run(hours=6, until=finished)
        old, new = run(OldPolicy(burst=64)), run(SchedPolicy(groups=True, gate=True))
        self.assertLessEqual(new.now, 0.85 * old.now)
        self.assertEqual(new.done, {'big': 900, 'small': 60})
        done_at = new.now
        new.run(hours=0.1)                       # idle afterwards: every group goes within release_ticks
        verbs = [e[1] for e in new.events]
        self.assertEqual(verbs.count('join'), 1)
        released = {e[2]: e[0] for e in new.events if e[1] == 'release'}
        self.assertEqual(set(released), {'g1', 'g2'})
        self.assertLessEqual(max(released.values()), done_at + 3 * 60)
        self.assertIn('stop_run', verbs)
        self.assertTrue(all(e.released for e in new.engines.values()))

    def test_watchdog_ends_hung_sessions_long_before_the_wall_cap(self):
        # 11-15: 362 GDPval and 53 AB sessions sat with no model request until the 2 h cap and held the drain
        def run(p):
            w = World({'g': ['e0', 'e1']}, [RunSpec('gdp', 'ab', 800, ['g'], limit=160, hang=0.05)], p, seed=5)
            return w.run(hours=6, until=finished)
        old, new = run(OldPolicy(burst=64)), run(SchedPolicy(watchdog_sec=600))
        self.assertLessEqual(new.now, 0.4 * old.now)          # 0.62-0.73 h vs 2.4-2.5 h
        self.assertLess(new.now, old.now - 3600)
        self.assertEqual(new.done['gdp'], 800)
        self.assertTrue(any(e[1] == 'infra' and e[4] == 'hung_session' for e in new.events))


if __name__ == '__main__':
    unittest.main()

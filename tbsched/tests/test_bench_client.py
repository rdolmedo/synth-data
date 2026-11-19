"""The replay client: tbsched, hash and M5-proto policies on synthetic metrics, and the driver on a fake replay."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tbsched import bench_client, legacy
from tbsched.bench_client import SchedClient


def text(running=0, waiting=0, kv=0.3, gen=0, hits=0, queries=0):
    return (f'vllm:num_requests_running {running}\nvllm:num_requests_waiting {waiting}\n'
            f'vllm:kv_cache_usage_perc {kv}\nvllm:generation_tokens_total {gen}\n'
            f'vllm:prefix_cache_hits_total {hits}\nvllm:prefix_cache_queries_total {queries}\n'
            'vllm:num_preemptions_total 0\n')


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def client(policy, metrics, **kw):
    clock, n = Clock(), [0]

    def fetch(url):
        n[0] += 1
        return metrics[url](n[0])
    c = SchedClient([{'metrics': u} for u in metrics], policy, scrape=fetch, now=clock, **kw)
    return c, clock


class Policies(unittest.TestCase):
    def test_sched_routes_new_attempts_to_the_lighter_engine(self):
        c, clock = client('sched', {'a': lambda k: text(10), 'b': lambda k: text(10)})
        c.live = [80, 10]
        for _ in range(3):
            c.scrape()
            clock.t += 15
        c.decide()
        got = [c.next_start(f't{i}', 0) for i in range(60)]
        self.assertEqual(got.count(1), 60)                     # engine 0: 80 of 96; engine 1 has 86 free
        self.assertEqual(c.live, [80, 70])

    def test_refill_tops_up_credit_between_decisions(self):
        c, clock = client('sched', {'a': lambda k: text(10), 'b': lambda k: text(10)})
        c.scrape()
        clock.t += 15
        c.scrape()
        c.decide()
        while c.next_start('t', 0) is not None:
            pass
        n = sum(c.started)
        c.refill()                                             # within the minute: the burst (96) is spent
        self.assertIsNone(c.next_start('x', 0))
        clock.t += 61
        c.refill()                                             # the next minute: up to the limits (96 + 96)
        got = [c.next_start(f'u{i}', 0) for i in range(120)]
        self.assertEqual(sum(x is not None for x in got), 96)
        c.ended(0, 'k1')
        c.ended(1, 'k2')
        clock.t += 61
        c.refill()                                             # two sessions ended: two new starts, no decide()
        got = [c.next_start(f'v{i}', 0) for i in range(4)]
        self.assertEqual(sum(x is not None for x in got), 2)
        self.assertEqual(sum(c.started), n + 98)

    def test_hash_is_the_11_15_rule(self):
        c, clock = client('hash', {'a': lambda k: text(), 'b': lambda k: text()}, limit=10, burst=6)
        got = [c.next_start(f't{i}', 0) for i in range(8)]
        self.assertEqual(got[:6], [legacy.hashed(2, f't{i}', 0) for i in range(6)])
        self.assertEqual(got[6:], [None, None])                # burst per 60 s
        clock.t += 61
        self.assertEqual(sum(x is not None for x in (c.next_start(f'u{i}', 0) for i in range(8))), 4)   # limit 10

    def test_proto_cuts_the_run_on_pressure_on_either_engine(self):
        c, clock = client('proto', {'a': lambda k: text(100, waiting=50), 'b': lambda k: text(10)}, limit=100,
                          sibling_hold=False)
        c.scrape()
        clock.t += 15
        c.scrape()
        self.assertEqual(c.decide()['verdict'], 'pressure')
        self.assertEqual(c.limit, 85)

    def test_gate_holds_the_youngest_after_a_collapse(self):
        c, clock = client('sched', {'a': lambda k: text(70)}, gate=True, max_wait=0.6)
        for i in range(6):
            c.after_turn(0, f's{i}', 100_000, 100, 0)
        c.gates[0].trigger(clock.t, 'test', capacity=1e6)
        waited = asyncio.run(c.before_turn(0, 's5', 100_100))
        self.assertGreaterEqual(waited, 0.5)                   # parked, then failed open at max_wait
        self.assertLess(asyncio.run(c.before_turn(0, 's0', 100_100)), 0.1)


class FakeReplay:
    def __init__(self, n_targets, trajectories):
        self.corpus = trajectories
        self.targets = [{'base': f'http://e{i}/v1', 'rank': None} for i in range(n_targets)]
        self.served, self.overrides, self.conversation_prefix = 'm', {}, 'cp'
        self.turns, self.errors = [], []
        self.stop = asyncio.Event()
        self.gap_cap, self.gap_scale = 300, 1.0

    def stream_args(self):
        return {'stream': True, 'stream_options': {'include_usage': True}}

    async def call(self, client, target, body, conversation=None):
        await asyncio.sleep(0.01)
        return {'t': 0, 'prompt': 1000, 'completion': 50, 'cached': 900, 'latency': 0.01}


class Driver(unittest.TestCase):
    def test_replay_runs_attempts_on_routed_targets_and_ends_them(self):
        import synth.bench as B
        orig = B.recorded_body

        async def body(trajectory, k, loaded):
            return {'messages': []}
        B.recorded_body = body
        try:
            rp = FakeReplay(2, [{'requests': [1, 2, 3], 'gaps': [0.01, 0.01, None]}] * 5)
            sched = SchedClient([{'metrics': 'a'}, {'metrics': 'b'}], 'sched', scrape=lambda u: text(5),
                                aimd={'start': 4, 'floor': 4, 'ceil': 4}, demand=8)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'rows.jsonl'
                asyncio.run(bench_client.replay(rp, sched, hours=0.5 / 3600, scrape_every=0.05, decide_every=0.1,
                                                out=str(out), tag='t'))
                rows = [json.loads(x) for x in out.read_text().splitlines()]
        finally:
            B.recorded_body = orig
        self.assertTrue(rows and rows[0]['policy'] == 'sched')
        self.assertIn('phase', rows[0])
        self.assertGreater(len(rp.turns), 6)
        self.assertEqual({t['target'] for t in rp.turns}, {0, 1})
        self.assertEqual(sched.live, [0, 0])                   # every attempt ended (finished or cancelled)



class Stationary(unittest.TestCase):
    def test_only_the_initial_fill_joins_mid_trajectory(self):
        import synth.bench as B
        orig_body, orig_attempt = B.recorded_body, bench_client.attempt
        k0s = []

        async def body(trajectory, k, loaded):
            return {'messages': []}

        async def spy(*a, k0=0, **kw):
            k0s.append(k0)
            return await orig_attempt(*a, k0=k0, **kw)
        B.recorded_body, bench_client.attempt = body, spy
        try:
            rp = FakeReplay(2, [{'requests': list(range(10)), 'gaps': [0.01] * 9 + [None]}] * 5)
            sched = SchedClient([{'metrics': 'a'}, {'metrics': 'b'}], 'sched', scrape=lambda u: text(5),
                                aimd={'start': 4, 'floor': 4, 'ceil': 4}, demand=8)
            asyncio.run(bench_client.replay(rp, sched, hours=0.3 / 3600, scrape_every=0.05, decide_every=0.1,
                                            stationary=True))
        finally:
            B.recorded_body, bench_client.attempt = orig_body, orig_attempt
        fill = 8                                               # the first decision's room: both engines' limits of 4
        self.assertGreater(len(k0s), fill)
        self.assertTrue(any(k > 0 for k in k0s[:fill]), k0s)    # the initial fill joins mid-trajectory
        self.assertTrue(all(k == 0 for k in k0s[fill:]), k0s)   # refills start at turn 0

if __name__ == '__main__':
    unittest.main()

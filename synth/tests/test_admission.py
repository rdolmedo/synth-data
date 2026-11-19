"""KV-fit admission (synth/admission.py), its controller wiring, and the guard's context state."""
import json
import math
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest import mock

from synth import admission, condor, controller as controller_module
from synth.common import atomic, read
from synth.config import PROTOCOL_DEFAULTS, Campaign, build, load_teacher
from synth.tests.test_controller_sim import FakeCluster
from tb_agents.pi_context_limit import ContextState, response_usage

CAPACITY, BLOCK = 11_443_384, 4256


def obs(**kw):
    base = {'kv': 0.0, 'wait': 0, 'preempted': False, 'stale': False, 'capacity': CAPACITY,
            'own_idle_tokens': 0, 'own_agents': 0, 'other_idle_tokens': 0, 'other_agents': 0,
            'reuse_expected': 0, 'reuse_cached': 0}
    return {**base, **kw}


A0 = admission.DEFAULTS['kv_target_fraction']


class Step(unittest.TestCase):
    def test_without_a_reuse_signal_the_target_never_exceeds_the_kv_tiers(self):
        policy = {'kv_target_max': 2.0}
        b = {'target': 2.0}
        st = admission.step(b, policy, 0, obs(), 20_000)              # no cached-token reports
        self.assertTrue(st['blind_cap'])
        self.assertEqual(st['limit'], int(CAPACITY / 20_000))
        self.assertEqual(b['target'], 2.0, 'the search state is kept; only the budget is capped')
        off = admission.step({'target': 2.0}, policy, 0, obs(offload=CAPACITY // 2), 20_000)
        self.assertEqual(off['limit'], int(1.5 * CAPACITY / 20_000))  # GPU KV + the CPU tier
        seen = admission.step({'target': 2.0}, policy, 0, obs(reuse_expected=3_000_000, reuse_cached=2_990_000), 20_000)
        self.assertFalse(seen['blind_cap'])
        self.assertEqual(seen['limit'], int(2.0 * CAPACITY / 20_000))

    def test_fills_to_the_target_share_of_kv(self):
        b, mean = {}, 20_000
        st = admission.step(b, {}, 0, obs(), mean)
        self.assertEqual(st['limit'], int(A0 * CAPACITY / mean))       # 228 TMax agents per B200 at 0.4
        self.assertEqual(st['room'], 32)                               # ramp: 32 per backend per tick
        full = admission.step(b, {}, 1, obs(own_idle_tokens=int(A0 * CAPACITY), own_agents=280), mean)
        self.assertEqual(full['room'], 0)
        self.assertFalse(full['paused'])

    def test_young_agents_count_at_the_mean_demand(self):
        """100 agents that hold 2k tokens each still occupy 100 x mean: a ramp is not over-admitted."""
        st = admission.step({}, {}, 0, obs(own_idle_tokens=100 * 2048, own_agents=100), 30_000)
        self.assertEqual(st['working_set_tokens'], 100 * 30_000)
        self.assertEqual(st['free_agents'], int((A0 * CAPACITY - 3_000_000) / 30_000))

    def test_other_campaigns_share_the_budget(self):
        st = admission.step({}, {}, 0, obs(other_idle_tokens=int(A0 * CAPACITY), other_agents=200), 20_000)
        self.assertEqual((st['room'], st['agents']), (0, 200))

    def test_inflight_kv_comes_from_the_gauge(self):
        """Running requests hold more than their prompt (Super: ~20k tokens of Mamba/MTP state blocks each),
        so the in-flight part of the working set is the gauge, not the agents' prompts."""
        st = admission.step({}, {}, 0, obs(kv=0.4, own_idle_tokens=int(0.2 * CAPACITY), own_agents=100), 20_000)
        self.assertEqual(st['room'], 0)
        self.assertAlmostEqual(st['measured_tokens'], 0.6 * CAPACITY, delta=2)
        st = admission.step({}, {}, 0, obs(kv=0.2, own_idle_tokens=int(0.1 * CAPACITY), own_agents=100), 20_000)
        self.assertEqual(st['free_agents'], int((A0 - 0.3) * CAPACITY / 20_000))

    def test_inflight_gauge_brake_resumes_after_hold(self):
        b = {}
        self.assertEqual(admission.step(b, {}, 0, obs(kv=0.93), 20_000)['causes'], ['kv_inflight_high'])
        self.assertTrue(admission.step(b, {}, 10, obs(kv=0.84), 20_000)['paused'])
        self.assertTrue(admission.step(b, {}, 35, obs(kv=0.84), 20_000)['paused'])
        self.assertEqual(admission.step(b, {}, 41, obs(kv=0.84), 20_000)['causes'], [])

    def test_preemption_pauses_and_lowers_the_target(self):
        b = {}
        st = admission.step(b, {}, 0, obs(preempted=True), 20_000)
        self.assertEqual((st['causes'], st['target']), (['preemption'], round(A0 * 0.9, 4)))
        self.assertTrue(admission.step(b, {}, 100, obs(), 20_000)['paused'])
        self.assertFalse(admission.step(b, {}, 130, obs(), 20_000)['paused'])

    def test_reuse_moves_the_target_only_when_operating_near_it(self):
        at_target = int(A0 * CAPACITY)
        reuse_mode = {'target_search': 'reuse'}
        b = {}
        st = admission.step(b, reuse_mode, 0, obs(own_idle_tokens=at_target, own_agents=280, reuse_expected=3e6, reuse_cached=2.4e6), 20_000)
        self.assertEqual(st['target'], round(A0 * 0.9, 4))           # thrash at the target: cut below it
        b = {}
        for t in range(0, 400, 15):
            st = admission.step(b, reuse_mode, t, obs(own_idle_tokens=at_target, own_agents=280, reuse_expected=3e6, reuse_cached=2.95e6), 20_000)
        self.assertGreater(st['target'], A0)                          # clean at the target: grow
        # Draining after an overload (working set far below the target, evicted agents still missing): no cut.
        b = {'target': 0.55}
        st = admission.step(b, reuse_mode, 0, obs(own_idle_tokens=int(0.3 * CAPACITY), own_agents=150, reuse_expected=3e6, reuse_cached=2e6), 20_000)
        self.assertEqual(st['target'], 0.55)

    def run_search(self, gen_of_target, ticks=2000, filled=True, start=None):
        b = {} if start is None else {'target': start}
        targets = []
        for k in range(ticks):
            t = 15 * k
            alpha = b.get('target', A0)
            st = admission.step(b, {}, t, obs(own_idle_tokens=int((alpha if filled else alpha / 2) * CAPACITY), own_agents=200,
                                              reuse_expected=1e6, reuse_cached=0.9e6, gen_tok_s=gen_of_target(alpha)), 20_000)
            targets.append(st['target'])
        return targets

    def test_throughput_search_settles_near_the_peak(self):
        peak = lambda a: 5000 - 40000 * (a - 0.55) ** 2   # single peak at a 0.55 working set
        late = self.run_search(peak)[-600:]
        self.assertTrue(0.42 <= min(late) and max(late) <= 0.70, (min(late), max(late)))
        # From above the peak it comes down.
        late = self.run_search(peak, start=0.85)[-600:]
        self.assertTrue(0.42 <= min(late) and max(late) <= 0.70, (min(late), max(late)))

    def test_throughput_search_holds_when_not_filled_to_the_target(self):
        self.assertEqual(set(self.run_search(lambda a: 4000 + 1000 * a, ticks=200, filled=False)), {A0})

    def test_cache_collapse_cuts_at_once(self):
        b = {}
        st = admission.step(b, {}, 0, obs(own_idle_tokens=int(A0 * CAPACITY), own_agents=200,
                                          reuse_expected=2e6, reuse_cached=0.6e6, gen_tok_s=3000), 20_000)
        self.assertEqual((st['target'], b['search']['direction']), (round(A0 * 0.9, 4), -1))

    def test_live_bounds_apply_to_the_stored_target(self):
        b = {'target': 0.3}
        self.assertEqual(admission.step(b, {'kv_target_min': 0.45}, 0, obs(), 20_000)['target'], 0.45)

    def test_a_cpu_kv_tier_scales_the_target_cap_and_start(self):
        # Without a tier: the GPU-only bounds. With a tier 4x the GPU KV: start at one GPU's worth, cap at 0.9 x 5.
        self.assertEqual(admission.step({}, {}, 0, obs(), 20_000)['target'], A0)
        self.assertEqual(admission.step({}, {}, 0, obs(offload=4 * CAPACITY), 20_000)['target'], 1.0)
        self.assertEqual(admission.step({'target': 9.0}, {}, 0, obs(offload=4 * CAPACITY), 20_000)['target'], 4.5)
        self.assertEqual(admission.step({'target': 9.0}, {}, 0, obs(), 20_000)['target'], 0.9)

    def test_a_cpu_tier_keeps_the_target_at_one_gpu_or_above(self):
        # Preemptions and a falling throughput curve cut a GPU-only target; with a CPU tier the floor is 1.0.
        b = {'target': 1.0}
        for k in range(400):
            admission.step(b, {}, 15 * k, obs(offload=4 * CAPACITY, own_idle_tokens=int(b['target'] * CAPACITY),
                                              own_agents=200, preempted=k % 20 == 0, reuse_expected=1e6,
                                              reuse_cached=0.95e6, gen_tok_s=9000 - 3000 * b['target']), 20_000)
        self.assertGreaterEqual(b['target'], 1.0)
        b = {'target': 0.5}
        for k in range(400):
            admission.step(b, {}, 15 * k, obs(own_idle_tokens=int(b['target'] * CAPACITY), own_agents=200,
                                              preempted=k % 20 == 0, reuse_expected=1e6, reuse_cached=0.95e6,
                                              gen_tok_s=9000 - 3000 * b['target']), 20_000)
        self.assertLess(b['target'], 0.5, 'without a CPU tier the same signals still lower the target')

    def test_throughput_search_climbs_past_the_gpu_kv_with_a_cpu_tier(self):
        b, targets = {}, []
        for k in range(2000):
            alpha = b.get('target', 1.0)
            st = admission.step(b, {}, 15 * k, obs(offload=4 * CAPACITY, own_idle_tokens=int(alpha * CAPACITY), own_agents=400,
                                                   reuse_expected=1e6, reuse_cached=0.95e6,
                                                   gen_tok_s=8000 - 2000 * (alpha - 2.0) ** 2), 20_000)
            targets.append(st['target'])
        late = targets[-600:]
        self.assertTrue(1.5 <= min(late) and max(late) <= 2.6, (min(late), max(late)))

    def fill_run(self, inflight_of_target, ticks=80, offload=4 * CAPACITY, preempt_every=None, policy=None):
        policy = {'inflight_fill_below': 0.55} if policy is None else policy
        """A backend kept filled to its target; the in-flight gauge follows the target; throughput rises with it."""
        b, targets = {}, []
        for k in range(ticks):
            alpha = b.get('target', 1.0 if offload else A0)
            st = admission.step(b, policy, 15 * k, obs(offload=offload, own_idle_tokens=int(alpha * CAPACITY), own_agents=400,
                                                   kv=inflight_of_target(alpha), reuse_expected=1e6, reuse_cached=0.95e6,
                                                   preempted=bool(preempt_every and k % preempt_every == 0),
                                                   gen_tok_s=3000 + 1500 * alpha), 20_000)
            targets.append(st['target'])
        return targets

    def test_a_cpu_tier_backend_short_of_running_requests_grows_fast(self):
        # Long tool time (CalibForge): in-flight KV 0.25 x target. 20 minutes take the target past 2.
        targets = self.fill_run(lambda a: 0.25 * a)
        self.assertGreaterEqual(targets[-1], 2.0, targets[::10])
        self.assertLess(self.fill_run(lambda a: 0.25 * a, policy={})[-1], 1.5, 'off by default')

    def test_no_fast_growth_when_running_requests_fill_the_gpu_or_without_a_cpu_tier(self):
        off = {'inflight_fill_below': 0}
        for kw in ({'inflight_of_target': lambda a: 0.85},                       # TMax: KV-bound
                   {'inflight_of_target': lambda a: 0.25 * a, 'offload': 0},     # GPU-only backend
                   {'inflight_of_target': lambda a: 0.25 * a, 'preempt_every': 8},
                   {'inflight_of_target': lambda a: 0.0}):                       # no gauge: no evidence
            self.assertEqual(self.fill_run(**kw), self.fill_run(**kw, policy=off), kw)

    def test_idle_footprint_whole_blocks_and_none_in_flight(self):
        self.assertEqual(admission.idle_tokens(None, BLOCK), 0)
        self.assertEqual(admission.idle_tokens({'context_tokens': 5000}, BLOCK), 2 * BLOCK)
        self.assertEqual(admission.idle_tokens({'context_tokens': 5000, 'inflight': True}, BLOCK), 0)


def log_interp(points):
    """gen tok/s as a function of the target, linear in log2(target) through (target, gen) points."""
    xs = [(math.log2(a), g) for a, g in points]
    def f(a):
        x = math.log2(a)
        lo, hi = (xs[0], xs[1]) if x <= xs[0][0] else (xs[-2], xs[-1]) if x >= xs[-1][0] else \
            next((xs[i], xs[i + 1]) for i in range(len(xs) - 1) if xs[i][0] <= x <= xs[i + 1][0])
        return lo[1] + (hi[1] - lo[1]) * (x - lo[0]) / (hi[0] - lo[0])
    return f


# DeepSeek-V4.1-Flash-1017, tuned command, 4xB200 (cpu401 replay 2026-10-23), with agents = 4,000 x target:
# 40.6k / 41.4k / 38.3k at 2,000 / 3,000 / 4,000 agents; KV never binds, the engine does.
ENGINE_BOUND = log_interp([(0.25, 33000), (0.5, 40600), (0.75, 41400), (1.0, 38300)])


class SlopeSearch(unittest.TestCase):
    """target_search = "slope" (issue #6): grow while a doubling adds >= 3%, stop at the knee or first thrash."""
    MODE = {'target_search': 'slope'}

    def simulate(self, gen_of, reuse_of=lambda a: 0.95, start=A0, hours=6, filled=True, wait=0, seed=1, demand=1.0):
        """demand < 1: agents use that share of the prior's KV, so the backend fills by agent count (mean x agents)."""
        rng, b, trace = random.Random(seed), {'target': start}, []
        for k in range(int(hours * 3600 / 15)):
            a = b['target']
            agents = int((a if filled else a / 2) * CAPACITY / 20_000)
            st = admission.step(b, self.MODE, 15 * k, obs(
                own_idle_tokens=int((a if filled else a / 2) * CAPACITY * demand), own_agents=agents, wait=wait,
                reuse_expected=1e6, reuse_cached=1e6 * reuse_of(a), gen_tok_s=gen_of(a) * (1 + rng.uniform(-0.02, 0.02))), 20_000)
            trace.append((15 * k, st['target'], st['causes']))
        return b, trace

    def test_settles_at_the_knee_of_an_engine_bound_curve_within_minutes(self):
        best = max(ENGINE_BOUND(a / 100) for a in range(15, 91))
        for seed in range(5):
            b, trace = self.simulate(ENGINE_BOUND, seed=seed)
            final = trace[-1][1]
            self.assertTrue(0.45 <= final <= 0.8, (seed, final, b['slope']['points']))
            self.assertGreaterEqual(ENGINE_BOUND(final), 0.95 * best)
            first = next(t for t, a, _ in trace if ENGINE_BOUND(a) >= 0.95 * best)
            self.assertLessEqual(first, 15 * 60, seed)                 # within ~10 min of start (one window)
            late = {a for t, a, _ in trace if t >= 2 * 3600}
            self.assertLessEqual(len(late), 2, late)                   # holds, with occasional one-step probes

    def test_agents_lighter_than_the_prior_still_count_as_filled(self):
        # DeepSeek agents hold ~12-16k KV tokens against the 20k prior: the target fills by agent count.
        b, trace = self.simulate(ENGINE_BOUND, demand=0.7)
        self.assertTrue(0.45 <= trace[-1][1] <= 0.8, b['slope']['points'])

    def test_stops_growing_at_the_first_sign_of_thrash(self):
        # Super-like: throughput still creeps up past the knee while idle prefixes get evicted (reuse falls).
        gen = log_interp([(0.2, 3000), (0.5, 4200), (0.9, 4400)])
        reuse = lambda a: 0.80 if a <= 0.5 else max(0.55, 0.80 - (a - 0.5) * 0.6)
        b, trace = self.simulate(gen, reuse_of=reuse)
        self.assertGreaterEqual(reuse(trace[-1][1]), 0.9 * 0.80, b['slope'])
        self.assertEqual(b['slope']['last'], 'thrash')
        self.assertLessEqual(sum(1 for t, a, _ in trace if reuse(a) < 0.72) * 15, 3 * 900)   # only brief probes past it

    def test_an_unfillable_target_with_a_queue_is_the_engine_cap(self):
        b, trace = self.simulate(ENGINE_BOUND, hours=1, filled=False, wait=500)
        self.assertEqual({a for _, a, _ in trace}, {A0})
        self.assertIn('engine_cap', trace[-1][2])

    def test_cache_collapse_cuts_at_once(self):
        b = {'target': 0.6}
        st = admission.step(b, self.MODE, 0, obs(own_idle_tokens=int(0.6 * CAPACITY), own_agents=200,
                                                 reuse_expected=2e6, reuse_cached=0.6e6, gen_tok_s=3000), 20_000)
        self.assertEqual((st['target'], b['slope']['phase']), (round(0.6 * 0.9, 4), 'hold'))


class GuardState(unittest.TestCase):
    def test_usage_from_a_streamed_tail_and_cache_reuse(self):
        tail = (b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":9000,"completion_tokens":500,'
                b'"prompt_tokens_details":{"cached_tokens":8000}}}\n\ndata: [DONE]\n\n')
        self.assertEqual(response_usage(tail)['prompt_tokens_details']['cached_tokens'], 8000)
        self.assertIsNone(response_usage(b'data: [DONE]'))
        with tempfile.TemporaryDirectory() as d:
            state = ContextState(Path(d) / 's.json')
            state.request(8000)
            self.assertTrue(read(Path(d) / 's.json')['inflight'])
            state.response(8000, {'prompt_tokens': 8000, 'completion_tokens': 1000,
                                  'prompt_tokens_details': {'cached_tokens': 0}})
            state.request(9500)
            state.response(9500, response_usage(tail.replace(b'9000', b'9500')))
            r = read(Path(d) / 's.json')
            self.assertEqual((r['context_tokens'], r['turns'], r['inflight']), (10000, 2, False))
            # The second turn repeats the first turn's 8000-token prompt (its completion is re-rendered,
            # never matched); vLLM served all 8000 from cache.
            self.assertEqual((r['expected_cached_tokens'], r['cached_tokens']), (8000, 8000))


class KvFitConfig(unittest.TestCase):
    def test_kv_fit_campaigns_freeze_kv_fit_brakes_not_the_limit_controller_values(self):
        config = build(Path(__file__).resolve().parents[1] / 'campaigns' / 'nemotron35-super-tmax-kvfit.toml')
        a = config['admission']
        self.assertEqual((a['mode'], a['emergency_kv_high'], a['resume_kv'], a['max_waiting']), ('kv_fit', 0.92, 0.85, 64))
        self.assertEqual(a['kv_target_fraction'], A0)
        self.assertNotIn('maximum_limit_per_backend', a)
        self.assertEqual(config['teacher']['serving']['tunables']['prompt_tokens_details'], 1)


class KvFitController(unittest.TestCase):
    def campaign(self, d, **workers):
        config = {'id': 'kvfit', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                  'protocol': dict(PROTOCOL_DEFAULTS),
                  'admission': {'eval_share': 0.25, **admission.DEFAULTS, 'mode': 'kv_fit', 'period_seconds': 0,
                                'agent_tokens_prior': 25 * BLOCK},
                  'train': {'mode': 'tokens', 'tokens_target': 10 ** 12, 'task_list': [f't{i}' for i in range(50)]},
                  'eval': {'enabled': False}, 'smoke': {'attempts': []},
                  'serving': {'jobs': 1, 'gpus_per_job': 4},
                  'workers': {'count': 2, 'slots': 8, **workers}}
        campaign = Campaign(config)
        campaign.ops.mkdir(parents=True)
        return campaign

    def run_ticks(self, campaign, ticks, contexts=lambda c: {}):
        cluster = FakeCluster(campaign)
        submitted = lambda role: (lambda c, index, generation, *a, **k: cluster.submit(c, role, index=index, generation=generation))
        with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
                mock.patch.object(condor, 'remove', cluster.remove), \
                mock.patch.object(condor, 'submit_serving', submitted('serving')), \
                mock.patch.object(condor, 'submit_worker', submitted('worker')), \
                mock.patch.object(controller_module.Controller, 'export', lambda self, force=False: None):
            c = controller_module.Controller(campaign)
            c.s['stage'] = 'running'
            saves, per_tick = [], []
            original_save, original_durable = c.save, c.durable
            c.save = lambda: (saves.append(1), original_save())
            c.durable = lambda items: (saves.append(1), original_durable(items))  # the journal record before inboxes
            for _ in range(ticks):
                condor._LISTINGS.clear()
                q = condor.queue(campaign)
                c.ensure_serving(q)
                cluster.publish()
                metrics = campaign.ops / 'metrics' / 'backends' / 'j0r0.json'
                atomic(metrics, {**read(metrics), 'raw': {'kv': 0.0, 'wait': 0, 'preempt': 0, 'run': 0}})
                endpoints = campaign.ops / 'serving' / 'job-0' / 'endpoints.json'
                published = read(endpoints)
                for r in published['replicas']:
                    r.update(kv_capacity_tokens=CAPACITY, kv_block_tokens=BLOCK)
                atomic(endpoints, published)
                for receipt in condor.receipts(campaign):  # heartbeats with each worker's agent contexts
                    if receipt['role'] == 'worker' and receipt['job_id'] in cluster.jobs:
                        mine = [i for i in c.active() if i['worker'] == receipt['index']]
                        atomic(campaign.ops / 'workers' / str(receipt['index']) / 'heartbeat.json',
                               {'updated_at': time.time(), 'job_id': receipt['job_id'], 'active': [i['id'] for i in mine],
                                'contexts': {i['id']: contexts(c) for i in mine}})
                ready = c.backends(q)
                saves.clear()
                before = len(c.active())
                c.running(q, ready)
                c.save()
                per_tick.append((len(c.active()) - before, len(saves)))
            return c, per_tick

    def test_dispatch_fills_the_kv_budget_with_one_save_per_tick(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = self.campaign(d, max_count=40)
            c, saves = self.run_ticks(campaign, 6, contexts=lambda c: {'context_tokens': 25 * BLOCK})
            limit = int(A0 * CAPACITY / (25 * BLOCK))
            self.assertEqual(len(c.active()), limit)  # 43 agents x 106k tokens fit 40% of 11.4M
            # The 2 configured workers heartbeat from the second tick (16 slots); the scaled-up ones a tick later.
            first = next(t for t in saves if t[0])
            self.assertEqual(first, (16, 2), 'a dispatching tick records once (journal) and snapshots once')
            self.assertEqual(sum(n for n, _ in saves), limit)
            # Workers scale to the attempts that hold slots + 25% headroom (8 slots + 2 spare), grown by the slots
            # dispatch ran short of, not to the KV room (#241).
            self.assertEqual(c.worker_count(), -(-limit * 5 // (4 * 8)) + 2)
            load = read(campaign.ops / 'serving' / 'load' / 'kvfit.json')
            self.assertEqual(load['backends']['j0r0']['agents'], limit)
            samples = (campaign.ops / 'metrics' / 'admission-samples.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(samples[-1])['backends']['j0r0']['room'], 0)

    def test_young_agents_do_not_pull_the_mean_below_the_prior(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = self.campaign(d, max_count=40)
            c, _ = self.run_ticks(campaign, 6, contexts=lambda c: {'context_tokens': 2 * BLOCK})
            self.assertEqual(len(c.active()), int(A0 * CAPACITY / (25 * BLOCK)))  # still the prior's 43

    def test_inboxes_follow_the_saved_state(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = self.campaign(d, max_count=4)
            c, _ = self.run_ticks(campaign, 2)
            c.collect_save() if c._save_future is None else c._save_future.result()
            # What a restarted controller recovers: the last snapshot plus the journal (#235).
            saved = controller_module.Controller(campaign).s
            for item in c.active():
                self.assertIn(item['id'], saved['items'])
                self.assertTrue((campaign.ops / 'workers' / str(item['worker']) / 'inbox' / f'{item["id"]}.json').exists())


if __name__ == '__main__':
    unittest.main()


class StalePublisher(unittest.TestCase):
    def test_a_slow_supervisor_pauses_admission_but_interrupts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = KvFitController.campaign(None, d, max_count=4)
            c, _ = KvFitController.run_ticks(None, campaign, 3)
            live = len(c.active())
            self.assertGreater(live, 0)
            endpoints = campaign.ops / 'serving' / 'job-0' / 'endpoints.json'
            atomic(endpoints, {**read(endpoints), 'updated_at': time.time() - 600})  # supervisor stalled
            q = {r['job_id']: {'JobStatus': 2} for r in condor.receipts(campaign)}
            self.assertEqual(c.backends(q), {})
            for _ in range(3):
                c.s['backend_missing_since'] = {k: v - 300 for k, v in c.s.get('backend_missing_since', {}).items()}
                c.check_backends(c.backends(q, include_stale=True))
            self.assertEqual(sum(1 for i in c.active() if i.get('infrastructure_failure')), 0)


class StalledSupervisor(unittest.TestCase):
    def test_stale_files_fall_back_to_the_engine(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = KvFitController.campaign(None, d, max_count=4)
            c, _ = KvFitController.run_ticks(None, campaign, 2)
            endpoints = campaign.ops / 'serving' / 'job-0' / 'endpoints.json'
            atomic(endpoints, {**read(endpoints), 'updated_at': time.time() - 600})
            q = {r['job_id']: {'JobStatus': 2} for r in condor.receipts(campaign)}
            with mock.patch.object(controller_module.Controller, 'engine_alive', lambda self, url: True):
                ready = c.backends(q)
            self.assertTrue(ready['j0r0']['publisher_stale'])
            with mock.patch.object(controller_module.Controller, 'engine_alive', lambda self, url: False):
                self.assertEqual(c.backends(q), {})
            metrics = campaign.ops / 'metrics' / 'backends' / 'j0r0.json'
            atomic(metrics, {**read(metrics), 'timestamp': time.time() - 600})
            raw = {'kv': 0.3, 'run': 10, 'wait': 0, 'preempt': 0, 'gen': 1000.0}
            with mock.patch('synth.probe.scrape', lambda base: raw), \
                    mock.patch('synth.probe.server_summary', lambda a, b, dt: {'srv_gen_tok_s': 99}):
                first = c.backend_sample('j0r0', ready['j0r0'], time.time())
                second = c.backend_sample('j0r0', ready['j0r0'], time.time() + 15)
            self.assertEqual((first['source'], first['raw']['kv']), ('controller_scrape', 0.3))
            self.assertEqual(second['srv_gen_tok_s'], 99)

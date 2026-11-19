"""#234 R1: serving engines follow demand (synth/autoscale.py) and ensure_serving never submits above the live cap."""
import math
import tempfile
import time
import unittest
from unittest import mock

from synth import autoscale, condor, controller as controller_module
from synth.common import atomic
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher
from synth.tests.test_controller_sim import FakeCluster

P = autoscale.params({'autoscale': {'start': 2, 'up_seconds': 300, 'down_seconds': 900, 'up_backlog': 1.0,
                                    'down_util': 0.5}})


def run(state, now, engines, demand, ceiling=8, fit=100, p=P):
    return autoscale.decide(state, now, engines=engines, ceiling=ceiling, demand=demand, fit=fit, p=p)


class Policy(unittest.TestCase):
    def test_params(self):
        self.assertFalse(autoscale.params({})['enabled'])
        self.assertTrue(autoscale.params({'autoscale': True})['enabled'])
        self.assertTrue(P['enabled'], 'a [serving.autoscale] table means on')
        self.assertFalse(autoscale.params({'autoscale': {'enabled': False, 'start': 3}})['enabled'])
        self.assertEqual(P['min_engines'], 2, 'the floor defaults to start')

    def test_starts_at_the_floor_without_waiting(self):
        self.assertEqual(run({}, 0, 0, 0), 2)
        self.assertEqual(run({}, 0, 0, 0, ceiling=1), 1, 'the floor never exceeds the ceiling')

    def test_adds_one_engine_after_a_sustained_backlog(self):
        st = {}
        # capacity 2 x 100; demand 350: backlog 150 > 1 engine's fit
        self.assertEqual(run(st, 0, 2, 350), 2)
        self.assertEqual(run(st, 299, 2, 350), 2, 'not before up_seconds')
        self.assertEqual(run(st, 300, 2, 350), 3)
        self.assertEqual(st['up_since'], None, 'timers restart after a change')
        self.assertEqual(run(st, 301, 3, 450), 3, 'the next engine waits its own window')

    def test_a_backlog_gap_restarts_the_up_window(self):
        st = {}
        run(st, 0, 2, 350)
        run(st, 200, 2, 250)         # backlog 50 < fit: condition broken
        self.assertEqual(run(st, 400, 2, 350), 2)
        self.assertEqual(run(st, 700, 2, 350), 3)

    def test_ceiling_bounds_the_scale_up(self):
        st = {}
        run(st, 0, 8, 10_000)
        self.assertEqual(run(st, 1000, 8, 10_000), 8)
        self.assertEqual(run({}, 0, 9, 10_000), 8, 'above the live cap: follow it at once')

    def test_up_step(self):
        p = {**P, 'up_step': 3}
        st = {}
        run(st, 0, 2, 2000, p=p)
        self.assertEqual(run(st, 300, 2, 2000, p=p), 5, 'up to up_step at once')
        st = {}
        run(st, 0, 2, 350, p=p)
        self.assertEqual(run(st, 300, 2, 350, p=p), 3, 'no more than the backlog needs')
        st = {}
        run(st, 0, 2, math.inf, p=p)
        self.assertEqual(run(st, 300, 2, math.inf, p=p), 5, 'unbounded demand (tokens mode)')

    def test_drains_one_engine_after_sustained_low_demand_not_below_the_floor(self):
        st = {}
        self.assertEqual(run(st, 0, 4, 150), 4)      # 150 < 0.5 x 400
        self.assertEqual(run(st, 899, 4, 150), 4)
        self.assertEqual(run(st, 900, 4, 150), 3)
        st2 = {}
        run(st2, 0, 2, 0)
        self.assertEqual(run(st2, 10_000, 2, 0), 2, 'never below min_engines')

    def test_no_drain_while_demand_recovers(self):
        st = {}
        run(st, 0, 4, 150)
        run(st, 600, 4, 250)        # 250 >= 200: not under
        self.assertEqual(run(st, 1000, 4, 150), 4)


def config(root, jobs=4, autoscale_cfg=None, tasks=400):
    serving = {'jobs': jobs, 'gpus_per_job': 1}
    if autoscale_cfg is not None:
        serving['autoscale'] = autoscale_cfg
    return {'id': 'simscale', 'root': root, 'teacher': load_teacher('nemotron35-super'),
            'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
            'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': [f't{i}' for i in range(tasks)]},
            'eval': {'enabled': False}, 'smoke': {'attempts': []},
            'serving': serving, 'workers': {'count': 8, 'slots': 64}}


class Harness:
    def __init__(self, cfg):
        self.campaign = Campaign(cfg)
        self.campaign.ops.mkdir(parents=True)
        self.cluster = FakeCluster(self.campaign)
        self.submits = []

    def __enter__(self):
        cl = self.cluster

        def sub(role):
            def f(c, index, generation, *a, **k):
                if role == 'serving':
                    self.submits.append(index)
                return cl.submit(c, role, index=index, generation=generation)
            return f
        self.patches = [mock.patch.object(condor, 'queue', cl.queue), mock.patch.object(condor, 'query', cl.query),
                        mock.patch.object(condor, 'remove', cl.remove), mock.patch.object(condor, 'release', cl.remove),
                        mock.patch.object(condor, 'submit_serving', sub('serving')),
                        mock.patch.object(condor, 'submit_worker', sub('worker'))]
        for p in self.patches:
            p.start()
        self.c = controller_module.Controller(self.campaign)
        self.c.s['stage'] = 'running'
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def serving_up(self):
        return sorted(r['index'] for r in condor.receipts(self.campaign)
                      if r['role'] == 'serving' and r['job_id'] in self.cluster.jobs)

    def tick(self):
        condor._LISTINGS.clear()
        q = condor.queue(self.campaign)
        self.c.ensure_serving(q)
        self.cluster.publish()
        self.c._heartbeats = None
        self.c.harvest()
        self.c.running(q, self.c.backends(q))

    def finish_all(self):
        for item in self.c.active():
            atomic(self.c.raw(item) / 'worker-result.json', {'finished_at': time.time(), 'interrupted': False})


class EnsureServingCap(unittest.TestCase):
    def test_restart_with_a_lower_live_cap_submits_only_up_to_the_cap(self):
        """m1-hero entry 7: ceiling 10, ops/admission-policy.json serving_jobs 7, controller (re)start."""
        with tempfile.TemporaryDirectory() as d, Harness(config(d, jobs=10)) as h:
            atomic(h.campaign.ops / 'admission-policy.json', {'serving_jobs': 7})
            h.tick()
            h.tick()
            self.assertEqual(h.serving_up(), list(range(7)))
            self.assertEqual(sorted(set(h.submits)), list(range(7)))
            atomic(h.campaign.ops / 'admission-policy.json', {'serving_jobs': 9})   # the cap raised: 7, 8 start
            h.tick()    # scale_down brings 7 and 8 back ...
            h.tick()    # ... and ensure_serving submits them
            self.assertEqual(h.serving_up(), list(range(9)))

    def test_a_dead_job_above_the_cap_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as d, Harness(config(d, jobs=3)) as h:
            h.tick()
            self.assertEqual(h.serving_up(), [0, 1, 2])
            atomic(h.campaign.ops / 'admission-policy.json', {'serving_jobs': 2})
            job2 = condor.latest(h.campaign, 'serving', 2)['job_id']
            h.cluster.jobs.pop(job2)            # job 2 dies while capped
            h.c.s['released_serving'] = []      # (and was not released by the drain path)
            h.c.s['draining_serving'] = []
            for _ in range(3):
                h.tick()
            self.assertEqual(h.serving_up(), [0, 1])
            self.assertEqual(h.submits.count(2), 1, 'job 2 not resubmitted above the cap')


class ControllerAutoscale(unittest.TestCase):
    FAST = {'start': 1, 'up_seconds': 0, 'down_seconds': 0, 'period_seconds': 0, 'up_backlog': 1.0, 'down_util': 0.5}

    def test_starts_small_scales_up_with_demand_and_never_past_the_cap(self):
        with tempfile.TemporaryDirectory() as d, Harness(config(d, jobs=4, autoscale_cfg=self.FAST)) as h:
            h.tick()
            self.assertEqual(h.serving_up(), [0], 'the campaign starts with `start` engines')
            seen = []
            for _ in range(8):
                h.tick()
                seen.append(len(h.serving_up()))
            self.assertEqual(h.serving_up(), [0, 1, 2, 3], '400 owed attempts vs fit 64: up to the ceiling')
            self.assertEqual(seen, sorted(seen), 'one engine at a time, never down while the backlog lasts')
            self.assertEqual(h.c.s['autoscale']['wanted'], [0, 1, 2, 3])
            atomic(h.campaign.ops / 'admission-policy.json', {'serving_jobs': 2})
            h.tick()
            self.assertEqual(h.c.s['autoscale']['wanted'], [0, 1], 'the live cap trims the engines in use')
            self.assertTrue({2, 3} <= set(h.c.s['draining_serving'] + h.c.s['released_serving']))
            self.assertEqual(max(h.submits), 3)

    def test_drains_and_releases_when_demand_falls_and_comes_back_through_ensure_serving(self):
        cfg = config('', jobs=3, autoscale_cfg=self.FAST, tasks=200)
        with tempfile.TemporaryDirectory() as d:
            cfg['root'] = d
            with Harness(cfg) as h:
                for _ in range(6):
                    h.tick()
                self.assertEqual(h.serving_up(), [0, 1, 2])
                for _ in range(12):            # every attempt finishes; the campaign runs out of work
                    h.finish_all()
                    h.tick()
                    if h.c.s['stage'] != 'running' or h.c.s['train_stop']:
                        break
                # demand fell: engines drained down to the floor, then training stopped and serving was released
                downs = [e for e in h.c.s['autoscale']['events'] if e[1] == 'down']
                self.assertTrue(downs, 'low demand drains engines')
                self.assertGreaterEqual(len(h.c.s['autoscale']['wanted']), 1, 'never below the floor')
                self.assertTrue(set(e[2] for e in downs) <= set(h.c.s['released_serving']), 'drained -> released')

    def test_scale_up_brings_back_a_draining_engine_before_submitting(self):
        with tempfile.TemporaryDirectory() as d, Harness(config(d, jobs=3, autoscale_cfg=self.FAST)) as h:
            for _ in range(5):
                h.tick()
            self.assertEqual(h.serving_up(), [0, 1, 2])
            n = len(h.submits)
            a = h.c.s['autoscale']
            a['wanted'].remove(2)                     # as if the policy had just drained 2 (attempts still pinned)
            h.c.s['draining_serving'].append(2)
            h.tick()                                  # demand is high: 2 comes back, no new job
            self.assertIn(2, a['wanted'])
            self.assertNotIn(2, h.c.s['draining_serving'])
            self.assertEqual(len(h.submits), n)
            self.assertEqual([e[1:3] for e in a['events']][-1], ['up', 2])

    def test_restart_keeps_the_running_engines(self):
        with tempfile.TemporaryDirectory() as d, Harness(config(d, jobs=4, autoscale_cfg={**self.FAST, 'start': 2})) as h:
            for j in (0, 1, 2):
                h.cluster.submit(h.campaign, 'serving', index=j)
            h.tick()
            self.assertEqual(sorted(h.c.s['autoscale']['wanted'])[:3], [0, 1, 2])
            self.assertEqual(h.submits.count(0), 0, 'running jobs are adopted by the policy, not resubmitted')


if __name__ == '__main__':
    unittest.main()

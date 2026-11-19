"""Drive the controller through a whole per-task campaign with fake Condor, serving and workers."""
import collections
import gzip
import json
import math
from pathlib import Path
import time
import unittest
from unittest import mock

from synth import condor, controller as controller_module
from synth.common import atomic, read, sha
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher


class FakeCluster:
    def __init__(self, campaign):
        self.c = campaign
        self.jobs = {}
        self.next_id = 1000

    def submit(self, campaign, role, *, index=0, generation=0, **kwargs):
        condor._LISTINGS.clear()  # as condor.submit does
        job_id = f'{self.next_id}.0'
        self.next_id += 1
        atomic(campaign.ops / 'jobs' / f'{role}-{index}-g{generation}-submission.json',
               {'job_id': job_id, 'role': role, 'index': index, 'generation': generation, 'time': time.time() - 300})
        self.jobs[job_id] = {'ClusterId': int(job_id.split('.')[0]), 'ProcId': 0, 'JobStatus': 2}
        return job_id

    def queue(self, campaign):
        """Like condor.queue: only this campaign's receipted jobs."""
        return self.query(condor.owned_ids(campaign))

    def query(self, job_ids):
        return {j: self.jobs[j] for j in job_ids if j in self.jobs}

    def remove(self, campaign, job_id):
        assert job_id in condor.owned_ids(campaign)
        self.jobs.pop(job_id, None)

    def publish(self):
        """Every running serving job healthy with one replica; every worker heartbeating."""
        now = time.time()
        for index in range(self.c.config['serving']['jobs']):
            serving = condor.latest(self.c, 'serving', index)
            if not serving or serving['job_id'] not in self.jobs:
                continue
            name = f'j{index}r0'
            home = Path(serving.get('serving_ops') or self.c.ops)  # an adopted job publishes in its own campaign
            generation = serving.get('serving_generation', serving['generation'])  # the job's own generation
            atomic(home / 'serving' / f'job-{index}' / 'endpoints.json', {'generation': generation,
                'updated_at': now, 'replicas': [{'name': name, 'url': f'http://fake{index}:1/v1', 'phase': 'healthy',
                                                 'healthy_at': 1.0, 'control_generation': 0, 'dp_size': 1}]})
            atomic(home / 'metrics' / 'backends' / f'{name}.json',
                   {'timestamp': now, 'raw': {'kv': 0.1, 'wait': 0, 'preempt': 0, 'run': 1}})
        for receipt in condor.receipts(self.c):
            if receipt['role'] == 'worker' and receipt['job_id'] in self.jobs:
                atomic(self.c.ops / 'workers' / str(receipt['index']) / 'heartbeat.json',
                       {'updated_at': now, 'job_id': receipt['job_id'], 'active': []})


def simulate(test, campaign, cluster, exported, long_running=(), ticks=60, on_tick=None):
    """Run controller ticks; fake workers finish every attempt except tasks in long_running
    (those finish once nothing else is active); on_tick(tick, controller) runs before each tick.
    Returns (controller, events)."""
    submitted = lambda role: (lambda c, index, generation, *a, **k: cluster.submit(c, role, index=index, generation=generation))

    def fake_export(self, force=False):
        """Every finished attempt exports usable, except the first try of task c."""
        for item in self.s['items'].values():
            if item['status'] == 'finished' and item['id'] not in exported:
                usable = not (item['task'] == 'c' and item['attempt'] == 0)
                record = {'id': item['id'], 'task': item['task'], 'usable': usable,
                          'finalized': True, 'tokens': 100 if usable else 0}
                if usable:
                    path = campaign.train / 'exports' / f'{item["id"]}.json.gz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, 'wt') as f:
                        json.dump({'tokens': 100, 'finalized': True}, f)
                    record.update(path=str(path), sha256=sha(path))
                exported[item['id']] = record
        self.indices['train'] = dict(exported)
        self.s['last_export_covered_after'] = time.time()
        self.manifests()

    events = []
    with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
            mock.patch.object(condor, 'remove', cluster.remove), \
            mock.patch.object(condor, 'submit_serving', submitted('serving')), \
            mock.patch.object(condor, 'submit_worker', submitted('worker')), \
            mock.patch.object(controller_module.Controller, 'export', fake_export):
        c = controller_module.Controller(campaign)
        for tick in range(ticks):
            if on_tick:
                on_tick(tick, c)
            condor._LISTINGS.clear()  # real ticks are 15 s apart; the listing cache lives 1 s
            q = condor.queue(campaign)
            if c.s['stage'] in ('smoke', 'running'):
                c.ensure_serving(q)
            cluster.publish()
            ready = c.backends(q)
            c.check_backends(ready)
            c.harvest()
            c.export()
            if c.s['stage'] in ('smoke', 'running'):
                c.stream_tick()
            if c.s['stage'] == 'smoke':
                c.smoke(q, ready)
            elif c.s['stage'] == 'running':
                c.running(q, ready)
            elif c.s['stage'] == 'finalizing':
                c.finalize(q)
            c.save()
            c.progress(ready)
            events.append({'tick': tick, 'released': list(c.s.get('released_serving', [])),
                           'active': len(c.active()), 'serving_jobs': sorted(j for j, r in
                               ((r['job_id'], r) for r in condor.receipts(campaign) if r['role'] == 'serving') if j in cluster.jobs)})
            others = [i for i in c.active() if i['task'] not in long_running]
            for item in c.active():
                if item['task'] in long_running and others:
                    continue
                if item['task'] in long_running and tick < 10:
                    continue
                folder = campaign.root_for(item['kind']) / 'raw' / item['id']
                atomic(folder / 'worker-result.json', {'finished_at': time.time(), 'interrupted': False})
            if c.s['stage'] == 'complete':
                break
    test.assertEqual(c.s['stage'], 'complete', c.s['stage'])
    return c, events


class ControllerSimulation(unittest.TestCase):
    def test_per_task_top_up_runs_to_completion(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            baseline = Path(d) / 'baseline.json'
            baseline.write_text(json.dumps({'records': [{'task': 'a', 'usable': True}]}))
            config = {'id': 'sim', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'max_extra_attempts_per_task': 2,
                                'baseline_manifests': [str(baseline)], 'task_list': ['a', 'b', 'c']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 2, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            c, _ = simulate(self, campaign, cluster, {})
            items = [i for i in c.s['items'].values() if i['kind'] == 'train']
            per_task = {t: sorted(i['attempt'] for i in items if i['task'] == t) for t in 'abc'}
            self.assertEqual(per_task, {'a': [0], 'b': [0, 1], 'c': [0, 1, 2]})
            self.assertEqual(len({i['seed'] for i in items}), len(items))
            completion = read(campaign.ops / 'completion.json')
            self.assertEqual(completion['per_task_min'], 2)
            self.assertEqual(completion['tasks_below_target'], [])
            packed = read(campaign.train / 'attempts' / 'index.json')          # [archive] pack_on_complete (default)
            self.assertEqual(len(packed['attempts']), len(items))
            self.assertEqual(list((campaign.train / 'raw').iterdir()), [])
            manifest = read(campaign.train / 'manifest.json')
            self.assertEqual(manifest['per_task']['a'], {'baseline': 1, 'new': 1, 'total': 2})
            self.assertEqual(manifest['per_task']['c'], {'baseline': 0, 'new': 2, 'total': 2})
            self.assertTrue((campaign.ops / 'serving' / 'RELEASE').exists())
            self.assertEqual(cluster.jobs, {}, 'every owned job released at completion')


    def test_drains_and_releases_a_serving_job_for_the_tail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim2', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS),
                      'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 4},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'max_extra_attempts_per_task': 0,
                                'baseline_manifests': [], 'task_list': ['a', 'b', 'd', 'e', 'f', 'g']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 2, 'gpus_per_job': 4}, 'workers': {'count': 3, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            c, events = simulate(self, campaign, cluster, {}, long_running=('g',))
            tail = [e for e in events if e['active'] == 1]
            self.assertTrue(tail and any(e['released'] for e in tail),
                            'the job without the long attempt is released while it still runs')
            self.assertEqual(len(tail[-1]['serving_jobs']), 1)
            log = (campaign.ops / 'status.log').read_text()
            self.assertIn('Draining serving job', log)
            self.assertIn('Released drained serving job', log)

    def test_retained_serving_is_never_drained_or_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim3', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS),
                      'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 4},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'max_extra_attempts_per_task': 0,
                                'baseline_manifests': [], 'task_list': ['a', 'b', 'd', 'e', 'f', 'g']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 2, 'gpus_per_job': 4}, 'workers': {'count': 3, 'slots': 4}}
            campaign = Campaign(config)
            (campaign.ops / 'serving').mkdir(parents=True)
            atomic(campaign.ops / 'serving' / 'RETAIN.json', {'reason': 'successor campaign'})
            cluster = FakeCluster(campaign)
            c, events = simulate(self, campaign, cluster, {}, long_running=('g',))
            self.assertFalse(any(e['released'] for e in events))
            serving = sorted(r['job_id'] for r in condor.receipts(campaign) if r['role'] == 'serving')
            self.assertEqual(sorted(j for j in cluster.jobs if j in serving), serving, 'serving jobs still queued')
            self.assertFalse(any(r['role'] != 'serving' and r['job_id'] in cluster.jobs for r in condor.receipts(campaign)))
            self.assertFalse((campaign.ops / 'serving' / 'RELEASE').exists())
            completion = read(campaign.ops / 'completion.json')
            self.assertFalse(completion['owned_resources_released'])
            self.assertEqual(completion['retained_serving']['jobs'], serving)
            self.assertNotIn('Draining serving job', (campaign.ops / 'status.log').read_text())

    def test_token_budget_opens_tasks_in_order_with_full_trajectories(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tasks = [f't{i:02d}' for i in range(20)]
            config = {'id': 'sim4', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'max_extra_attempts_per_task': 2,
                                'tokens_budget': 1000, 'tokens_per_trajectory_prior': 100, 'task_list': tasks,
                                'budget_stop': 'projected'},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 2, 'slots': 4}}
            validation = Path(d) / 'validation.json'
            atomic(validation, {**{t: True for t in tasks[:12]}, 't02': False})
            config['train']['validation'] = str(validation)
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c, _ = simulate(self, campaign, FakeCluster(campaign), {})
            used = sorted({i['task'] for i in c.s['items'].values() if i['kind'] == 'train'})
            expected = ['t00', 't01', 't03', 't04', 't05']
            self.assertEqual(used, expected, 'valid tasks open in list order, only as many as the budget needs')
            completion = read(campaign.ops / 'completion.json')
            self.assertEqual((completion['tasks'], completion['per_task_min']), (5, 2))
            self.assertGreaterEqual(completion['train_tokens_saved'], 1000)
            manifest = read(campaign.train / 'manifest.json')
            self.assertEqual((sorted(manifest['per_task']), manifest['skipped_invalid_tasks']), (expected, 1))

    def test_saved_budget_runs_at_max_concurrency_then_closes_and_releases(self):
        # Default budget policy (user, 2026-10-24): keep opening tasks while saved tokens are under the budget, never
        # drain serving for a tail, close the moment the budget is reached, release serving before the final exports.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tasks = [f't{i:02d}' for i in range(60)]
            config = {'id': 'sim4s', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS),
                      'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 4},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'max_extra_attempts_per_task': 2,
                                'tokens_budget': 1500, 'tokens_per_trajectory_prior': 100, 'task_list': tasks},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 2, 'gpus_per_job': 4}, 'workers': {'count': 3, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c, events = simulate(self, campaign, FakeCluster(campaign), {})
            log = (campaign.ops / 'status.log').read_text()
            self.assertIn('Token budget reached at max concurrency', log)
            self.assertIn('Released serving: training stopped', log)
            self.assertNotIn('Draining serving job', log)
            closed = c.s['budget_closed']
            self.assertGreaterEqual(closed['estimated_saved'], 1500)
            completion = read(campaign.ops / 'completion.json')
            self.assertGreaterEqual(completion['train_tokens_saved'], 1500)
            self.assertLessEqual(completion['train_tokens_saved'], 1500 + 100 * 8, 'overshoot bounded by in-flight work')
            before = [e for e in events if e['tick'] < 3]
            self.assertTrue(all(len(e['serving_jobs']) == 2 for e in events if e['active']), 'both backends kept')

    def test_token_budget_waits_for_validation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            validation = Path(d) / 'validation.json'
            atomic(validation, {'t00': True})
            config = {'id': 'sim5', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'tokens_budget': 1000,
                                'tokens_per_trajectory_prior': 100, 'validation': str(validation),
                                'task_list': ['t00', 't01', 't02']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c = controller_module.Controller(campaign)
            c.s['stage'] = 'running'
            self.assertEqual(c.next_train(), ('t00', 0))
            c.s['opened'], c.s['items'] = ['t00'], {}
            self.assertIsNone(c.open_next_task(True), 't01 and t02 are not validated yet')
            self.assertEqual(c.pending_work(), float('inf'))
            atomic(validation, {'t00': True, 't01': False, 't02': True})
            import os
            os.utime(validation, (time.time() + 5, time.time() + 5))
            self.assertEqual(c.open_next_task(True), 't02')
            self.assertEqual(c.s['skipped_invalid'], 1)

    def test_token_budget_skips_ahead_and_catches_up(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            validation = Path(d) / 'validation.json'
            atomic(validation, {'a': True, 'c': True, 'd': True})
            config = {'id': 'sim6', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'tokens_budget': 1000,
                                'validation': str(validation), 'task_list': ['a', 'b', 'c', 'd']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c = controller_module.Controller(campaign)
            self.assertEqual([c.open_next_task(True), c.open_next_task(True)], ['a', 'c'], 'b is pending: skip ahead')
            self.assertIsNone(c.open_next_task(False), 'budget closed: d stays closed, b still pending')
            self.assertTrue(c.budget_pending())
            self.assertFalse(c.train_target_reached())
            atomic(validation, {'a': True, 'b': True, 'c': True, 'd': True})
            os.utime(validation, (time.time() + 5, time.time() + 5))
            self.assertEqual(c.open_next_task(False), 'b', 'a valid task behind the frontier opens anyway')
            self.assertFalse(c.budget_pending())
            self.assertEqual(c.s['opened'], ['a', 'c', 'b'])

    def test_saved_budget_overflows_one_trajectory_while_nothing_can_open(self):
        import tempfile
        for policy, expected in (('saved', [('a', 0), ('a', 1), ('a', 2), None]), ('projected', [('a', 0), ('a', 1), None, None])):
            with tempfile.TemporaryDirectory() as d:
                validation = Path(d) / 'validation.json'
                atomic(validation, {'a': True})  # b and c are still being validated
                config = {'id': 'sim7', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                          'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                          'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'tokens_budget': 10 ** 9,
                                    'validation': str(validation), 'task_list': ['a', 'b', 'c'], 'budget_stop': policy},
                          'eval': {'enabled': False}, 'smoke': {'attempts': []},
                          'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
                campaign = Campaign(config)
                campaign.ops.mkdir(parents=True)
                c = controller_module.Controller(campaign)
                plan = c.train_plan()
                self.assertEqual([c.next_train(plan) for _ in range(4)], expected, policy)

    def test_adopter_waits_for_a_running_source_instead_of_submitting(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            base = {'teacher': load_teacher('nemotron35-super'), 'protocol': dict(PROTOCOL_DEFAULTS),
                    'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0}, 'eval': {'enabled': False},
                    'smoke': {'attempts': []}, 'workers': {'count': 1, 'slots': 4}}
            source = Campaign({**base, 'id': 'src', 'root': d, 'serving': {'jobs': 1, 'gpus_per_job': 4},
                               'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['x']}})
            source.ops.mkdir(parents=True)
            source.freeze()
            atomic(source.ops / 'serving' / 'RETAIN.json', {'reason': 'test'})
            campaign = Campaign({**base, 'id': 'adopter', 'root': d,
                                 'serving': {'jobs': 1, 'gpus_per_job': 4, 'adopt_ops': str(source.ops)},
                                 'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']}})
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            submitted = []
            with mock.patch.object(condor, 'query', cluster.query), \
                    mock.patch.object(condor, 'submit_serving', lambda c, index, generation: submitted.append(c.id)):
                c = controller_module.Controller(campaign)
                c.ensure_serving(cluster.queue(campaign))
                self.assertEqual(submitted, [], 'the running source has no serving yet: wait, do not submit')
                atomic(source.ops / 'jobs' / 'serving-0-g1-submission.json',
                       {'job_id': '901.0', 'role': 'serving', 'index': 0, 'generation': 1, 'time': time.time()})
                cluster.jobs['901.0'] = {'ClusterId': 901, 'ProcId': 0, 'JobStatus': 1}
                condor._LISTINGS.clear()
                c.ensure_serving(cluster.queue(campaign))
                self.assertEqual(condor.latest(campaign, 'serving', 0)['job_id'], '901.0')
                cluster.jobs.pop('901.0')
                atomic(source.ops / 'completion.json', {'status': 'complete'})
                receipt = campaign.ops / 'jobs' / 'serving-0-g0-submission.json'   # the adopted receipt, aged
                atomic(receipt, {**read(receipt), 'time': time.time() - 300})
                condor._RECEIPTS.clear()
                condor._LISTINGS.clear()
                c.ensure_serving(cluster.queue(campaign))
                self.assertEqual(submitted, ['adopter'], 'the source is complete: submit its own')

    def test_adopts_a_retained_serving_job_instead_of_submitting(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            base = {'teacher': load_teacher('nemotron35-super'), 'protocol': dict(PROTOCOL_DEFAULTS),
                    'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0}, 'eval': {'enabled': False},
                    'smoke': {'attempts': []}, 'workers': {'count': 2, 'slots': 4}}
            source = Campaign({**base, 'id': 'old', 'root': d, 'serving': {'jobs': 1, 'gpus_per_job': 4},
                               'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['x']}})
            source.ops.mkdir(parents=True)
            source.freeze()
            atomic(source.ops / 'jobs' / 'serving-0-g2-submission.json',
                   {'job_id': '900.0', 'role': 'serving', 'index': 0, 'generation': 2, 'time': time.time() - 300})
            config = {**base, 'id': 'new', 'root': d,
                      'serving': {'jobs': 1, 'gpus_per_job': 4, 'adopt_ops': str(source.ops)},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a', 'b']}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            cluster.jobs['900.0'] = {'ClusterId': 900, 'ProcId': 0, 'JobStatus': 2}
            controller = controller_module.Controller(campaign)
            with self.assertRaisesRegex(RuntimeError, 'RETAIN'):
                controller.adopt_serving(cluster.queue(campaign))
            atomic(source.ops / 'serving' / 'RETAIN.json', {'reason': 'test'})
            # An own serving job of an earlier generation that is gone does not block adoption.
            atomic(campaign.ops / 'jobs' / 'serving-0-g0-submission.json',
                   {'job_id': '800.0', 'role': 'serving', 'index': 0, 'generation': 0, 'time': time.time() - 300})
            c, _ = simulate(self, campaign, cluster, {})
            adopted = condor.latest(campaign, 'serving', 0)
            self.assertEqual((adopted['job_id'], adopted['serving_ops'], adopted['generation'], adopted['serving_generation']),
                             ('900.0', str(source.ops), 1, 2))
            self.assertEqual(len([r for r in condor.receipts(campaign) if r['role'] == 'serving']), 2, 'no new submission')
            self.assertTrue(all(i['backend'] == 'j0r0' for i in c.s['items'].values()))
            self.assertIn('900.0', cluster.jobs, 'the source still runs: the adopter detaches, never removes it')
            mark = read(source.ops / 'serving' / 'adopters' / 'new-900.0.json')
            self.assertTrue(mark['detached_at'])
            self.assertEqual(condor.attached(source.ops, '900.0'), [])
            # Once the source has completed, the last adopter to let go removes it.
            atomic(source.ops / 'completion.json', {'status': 'complete'})
            with mock.patch.object(condor, 'remove', cluster.remove):
                condor.release(campaign, '900.0')
            self.assertNotIn('900.0', cluster.jobs)

    def test_retain_holds_serving_only_while_an_adopter_is_attached(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            campaign = Campaign({'id': 'src', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                                 'protocol': dict(PROTOCOL_DEFAULTS), 'admission': dict(ADMISSION_DEFAULTS),
                                 'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']},
                                 'eval': {'enabled': False}, 'smoke': {'attempts': []},
                                 'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}})
            campaign.ops.mkdir(parents=True)
            FakeCluster(campaign).submit(campaign, 'serving')
            job = condor.latest(campaign, 'serving', 0)['job_id']
            retain = campaign.ops / 'serving' / 'RETAIN.json'
            atomic(retain, {'reason': 'successor'})
            c = controller_module.Controller(campaign)
            self.assertTrue(c.retained_serving(), 'fresh: the successor has an hour to adopt')
            os.utime(retain, (time.time() - 7200,) * 2)
            self.assertIsNone(c.retained_serving(), 'stale and nobody adopted: released as usual')
            adopter = Path(d) / 'next' / 'ops'
            atomic(campaign.ops / 'serving' / 'adopters' / f'next-{job}.json',
                   {'campaign': 'next', 'ops': str(adopter), 'job_id': job, 'adopted_at': time.time()})
            self.assertTrue(c.retained_serving(), 'an attached adopter keeps it')
            atomic(adopter / 'completion.json', {'status': 'complete'})
            self.assertIsNone(c.retained_serving(), 'a completed adopter no longer does')

    def test_infrastructure_failures_do_not_use_up_attempts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim7', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'max_extra_attempts_per_task': 1,
                                'max_infrastructure_failures_per_task': 5, 'task_list': ['a']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c = controller_module.Controller(campaign)
            for n in range(4):  # four attempts lost to a full worker scratch
                c.s['items'][f'x{n}'] = {'id': f'x{n}', 'kind': 'train', 'task': 'a', 'attempt': n, 'status': 'finished',
                                         'infrastructure_failure': 'WorkerRunError'}
            self.assertEqual(c.next_train(), ('a', 4), 'the cap (2 + 1) ignores infrastructure failures; numbering does not')
            c.s['items']['x4'] = {'id': 'x4', 'kind': 'train', 'task': 'a', 'attempt': 4, 'status': 'finished',
                                  'infrastructure_failure': 'WorkerRunError'}
            self.assertIsNone(c.next_train(), 'five infrastructure failures stop the task')

    def test_close_train_finalizes_with_what_it_has(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim8', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS),
                      'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 4},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'max_extra_attempts_per_task': 0,
                                'baseline_manifests': [], 'task_list': ['a', 'b', 'g']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            atomic(campaign.ops / 'CLOSE_TRAIN.json', {'reason': 'test'})
            c, _ = simulate(self, campaign, FakeCluster(campaign), {}, long_running=('g',))
            completion = read(campaign.ops / 'completion.json')
            self.assertEqual(completion['train_closed_early'], {'reason': 'test'})
            self.assertIn('Training closed by ops/CLOSE_TRAIN.json', (campaign.ops / 'status.log').read_text())

    def test_smoke_lost_to_infrastructure_is_retried(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim9', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': [{'task': 'a', 'kind': 'train'}]},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
                    mock.patch.object(condor, 'submit_serving', lambda c, index, generation, *a, **k: cluster.submit(c, 'serving', index=index, generation=generation)), \
                    mock.patch.object(condor, 'submit_worker', lambda c, index, generation, *a, **k: cluster.submit(c, 'worker', index=index, generation=generation)):
                c = controller_module.Controller(campaign)
                for _ in range(2):
                    q = condor.queue(campaign)
                    c.ensure_serving(q)
                    c.ensure_workers(q, 1)
                    cluster.publish()
                    ready = c.backends(q)
                    c.smoke(q, ready)
                first = [i for i in c.s['items'].values() if i['kind'] == 'smoke']
                self.assertEqual([i['retry'] for i in first], [0])
                first[0].update(status='finished', infrastructure_failure='OwnedWorkerLost')
                c.smoke(condor.queue(campaign), c.backends(condor.queue(campaign)))
                retries = sorted(i['retry'] for i in c.s['items'].values() if i['kind'] == 'smoke')
                self.assertEqual(retries, [0, 1], 'the lost smoke is reassigned as retry 1, not judged')

    def test_worker_shapes_from_overrides_and_proportional_fill(self):
        overrides = {'0-1': {'slots': 8, 'cpus': 32}, '1': {'memory_gib': 96}}
        self.assertEqual(condor.worker_shape(overrides, 0), {'slots': 8, 'cpus': 32})
        self.assertEqual(condor.worker_shape(overrides, 1), {'slots': 8, 'cpus': 32, 'memory_gib': 96})
        self.assertEqual(condor.worker_shape(overrides, 2), {})
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'simshape', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS),
                      'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 64},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1,
                                'task_list': [f't{i}' for i in range(6)]},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4},
                      'workers': {'count': 2, 'slots': 4, 'cpus': 16, 'memory_gib': 192, 'disk_gib': 320, 'bid': 56}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            atomic(campaign.ops / 'worker-resource-overrides.json', {'1': {'slots': 8, 'cpus': 32}})
            cluster = FakeCluster(campaign)
            shapes = {}

            def submit_worker(c, index, generation, shape=None, avoid=()):
                shapes[index] = shape
                return cluster.submit(c, 'worker', index=index, generation=generation)
            with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
                    mock.patch.object(condor, 'submit_serving', lambda c, index, generation, *a, **k: cluster.submit(c, 'serving', index=index, generation=generation)), \
                    mock.patch.object(condor, 'submit_worker', submit_worker):
                c = controller_module.Controller(campaign)
                c.s['stage'] = 'running'
                for _ in range(3):
                    condor._LISTINGS.clear()
                    q = condor.queue(campaign)
                    c.ensure_serving(q)
                    cluster.publish()
                    for index in (0, 1):  # each worker heartbeats its own --slots as capacity
                        hb = campaign.ops / 'workers' / str(index) / 'heartbeat.json'
                        if hb.exists():
                            atomic(hb, {**read(hb), 'capacity': (4, 8)[index]})
                    c._heartbeats = None
                    ready = c.backends(q)
                    c.running(q, ready)
            self.assertEqual(shapes[1], {'slots': 8, 'cpus': 32})
            self.assertEqual(shapes[0], {})
            per_worker = collections.Counter(i['worker'] for i in c.active())
            self.assertEqual((per_worker[0], per_worker[1]), (2, 4), 'filled in proportion to each worker\'s slots')

    def test_tail_rule_waits_while_an_adopter_shares_the_serving(self):
        import tempfile
        for shared in (False, True):
            with tempfile.TemporaryDirectory() as d:
                campaign = Campaign({'id': 'src', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                                     'protocol': dict(PROTOCOL_DEFAULTS), 'admission': dict(ADMISSION_DEFAULTS),
                                     'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']},
                                     'eval': {'enabled': False}, 'smoke': {'attempts': []},
                                     'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}})
                campaign.ops.mkdir(parents=True)
                FakeCluster(campaign).submit(campaign, 'serving')
                job = condor.latest(campaign, 'serving', 0)['job_id']
                if shared:
                    atomic(campaign.ops / 'serving' / 'adopters' / f'next-{job}.json',
                           {'campaign': 'next', 'ops': str(Path(d) / 'next' / 'ops'), 'job_id': job, 'adopted_at': time.time()})
                c = controller_module.Controller(campaign)
                c.s.update(train_starved_since=time.time() - 3600, opened=['a'])
                with mock.patch.object(c, 'max_concurrency_open', return_value=True):
                    c.tail_check({'j0r0': {'limit': 100}})
                self.assertEqual(bool(c.s.get('tail_closed')), not shared)

    def test_worker_interruptions_are_infrastructure_failures(self):
        self.assertEqual(controller_module.infrastructure_failure({}, {'interrupted': True}), 'WorkerInterrupted')
        self.assertIsNone(controller_module.infrastructure_failure({}, {'interrupted': False}))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim10', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': dict(ADMISSION_DEFAULTS),
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            (campaign.ops / 'controller').mkdir(parents=True)
            atomic(campaign.ops / 'controller' / 'state.json', {'stage': 'running', 'items': {
                'x': {'id': 'x', 'kind': 'train', 'task': 'a', 'status': 'finished', 'interrupted': True},
                'y': {'id': 'y', 'kind': 'train', 'task': 'a', 'status': 'finished', 'interrupted': True,
                      'infrastructure_failure': 'TrainingTargetReached'}}, 'train_cursor': 0, 'workers': {},
                'backends': {}, 'train_stop': False})
            c = controller_module.Controller(campaign)
            self.assertEqual(c.s['items']['x']['infrastructure_failure'], 'WorkerInterrupted')
            self.assertEqual(c.s['items']['y']['infrastructure_failure'], 'TrainingTargetReached')

    def test_tail_rule_cuts_a_trickle_and_reports_unresolved_eval(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim11', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': dict(ADMISSION_DEFAULTS),
                      'train': {'mode': 'tokens', 'tokens_target': 100, 'task_list': ['a']},
                      'eval': {'attempts_per_task': 2, 'task_list': ['e1', 'e2']}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            c = controller_module.Controller(campaign)
            c.s['train_stop'] = True
            item = lambda uid, task, attempt, **kw: {'id': uid, 'kind': 'eval', 'task': task, 'attempt': attempt,
                                                     'retry': 0, 'assigned_at': time.time(), **kw}
            c.s['items'] = {'r': item('r', 'e1', 0, status='finished', reward=1.0),
                            's': item('s', 'e2', 0, status='finished', reward=None, exception_type='VerifierTimeoutError'),
                            'live': item('live', 'e1', 1, status='active')}
            c.indices['eval'] = {'r': {'usable': True, 'finalized': True}}
            self.assertEqual(list(c.eval_candidates()), [('e2', 1, 0)])
            c.s['items']['live2'] = item('live2', 'e2', 1, status='active')
            self.assertEqual(c.remaining_work(), (0, 4))
            atomic(campaign.ops / 'tail-policy.json', {'hold_minutes': 0})
            c.tail_check({'j0r0': {'limit': 64}})
            self.assertTrue(c.s['tail_closed'])
            self.assertEqual(c.s['items']['live']['infrastructure_failure'], 'TailCut')
            self.assertTrue((c.raw(c.s['items']['live']) / 'CANCEL').exists())
            self.assertIsNone(c.next_eval(), 'no evaluation starts after the tail rule')
            self.assertTrue(c.train_target_reached())
            c.score(c.resolved_eval())
            scores = read(campaign.eval / 'scores.json')
            self.assertEqual((scores['pass_at_1'], scores['pass_at_1_unresolved_as_fail'], scores['pass_at_1_resolved_only']),
                             (None, 0.25, 1.0))
            self.assertEqual(scores['unresolved'], {'VerifierTimeoutError': 1, 'TailCut': 2})

    def test_tail_rule_ignores_a_busy_or_open_campaign(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'sim12', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': dict(ADMISSION_DEFAULTS),
                      'train': {'mode': 'tokens', 'tokens_target': 100, 'task_list': ['a']},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            atomic(campaign.ops / 'tail-policy.json', {'hold_minutes': 0})
            c = controller_module.Controller(campaign)
            c.tail_check({'j0r0': {'limit': 64}})
            self.assertIsNone(c.s.get('tail_closed'), 'an open token target is never a tail')


class ImageAffinity(unittest.TestCase):
    def run_ticks(self, d, affinity):
        groups = Path(d) / 'groups.json'
        groups.write_text(json.dumps({'t0': 'g1', 't1': 'g1', 't2': 'g2', 't3': 'g2'}))
        config = {'id': f'simaff{int(affinity)}', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                  'protocol': dict(PROTOCOL_DEFAULTS),
                  # 16-slot workers: slack 0.5 (8 slots) holds two tasks' 4 attempts (production: 48 slots, 0.25)
                  'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0, 'initial_limit_per_backend': 64,
                                'affinity_slack': 0.5},
                  'train': {'mode': 'per_task', 'trajectories_per_task': 4, 'image_groups': str(groups),
                            'task_list': ['t0', 't1', 't2', 't3', 't4']},
                  'eval': {'enabled': False}, 'smoke': {'attempts': []},
                  'serving': {'jobs': 1, 'gpus_per_job': 4},
                  'workers': {'count': 4, 'slots': 16, 'affinity': affinity}}
        campaign = Campaign(config)
        campaign.ops.mkdir(parents=True)
        cluster = FakeCluster(campaign)
        with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
                mock.patch.object(condor, 'submit_serving', lambda c, index, generation, *a, **k: cluster.submit(c, 'serving', index=index, generation=generation)), \
                mock.patch.object(condor, 'submit_worker', lambda c, index, generation, *a, **k: cluster.submit(c, 'worker', index=index, generation=generation)):
            c = controller_module.Controller(campaign)
            c.s['stage'] = 'running'
            for _ in range(5):  # burst admission starts 8 attempts per tick
                condor._LISTINGS.clear()
                q = condor.queue(campaign)
                c.ensure_serving(q)
                cluster.publish()
                c._heartbeats = None
                c.running(q, c.backends(q))
        return c

    def test_a_tasks_attempts_and_its_groups_tasks_share_a_worker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.run_ticks(d, True)
            items = [i for i in c.s['items'].values() if i['kind'] == 'train']
            self.assertEqual(len(items), 20)
            workers = {t: {i['worker'] for i in items if i['task'] == t} for t in ('t0', 't1', 't2', 't3', 't4')}
            self.assertTrue(all(len(w) == 1 for w in workers.values()), workers)
            self.assertEqual(workers['t0'], workers['t1'], 'same shared base, same worker')
            self.assertEqual(workers['t2'], workers['t3'])
            self.assertTrue(all(i['worker_job'] for i in items))
            self.assertEqual(c.s['affinity_stats']['other'], 3, 'only the first attempt of t0, t2, t4 is free')

    def test_without_affinity_attempts_spread_over_workers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.run_ticks(d, False)
            items = [i for i in c.s['items'].values() if i['kind'] == 'train']
            self.assertGreater(len({i['worker'] for i in items if i['task'] == 't0'}), 1)

    def test_affinity_yields_to_load_balance(self):
        c = object.__new__(controller_module.Controller)
        c.s = {}
        c._image_groups = {'a': 'g', 'b': 'g'}
        aff = (collections.defaultdict(set, {'a': {0}}), collections.defaultdict(set, {'g': {0}}))
        slots = {0: 10, 1: 10}
        self.assertEqual(c.affine_worker('a', {0: 3, 1: 1}, slots, 1, aff, 0.25), 0)   # 0.3 <= 0.1 + 0.25
        self.assertEqual(c.affine_worker('b', {0: 5, 1: 1}, slots, 1, aff, 0.25), 1)   # 0.5 > 0.35: least loaded
        self.assertEqual(c.s['affinity_stats'], {'task': 1, 'group': 0, 'other': 1})


class PostAgentAdmission(unittest.TestCase):
    def test_attempts_past_their_agent_do_not_count_against_the_engine(self):
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            c = object.__new__(controller_module.Controller)
            c.c = mock.Mock(config={'workers': {'slots': 48, 'max_count': 10}}, id='t')
            c.ops = Path(d)
            c.pool = ThreadPoolExecutor(2)
            c._reuse_seen = {}
            c.s = {'backends': {}, 'items': {f'u{i}': {'id': f'u{i}', 'status': 'active', 'kind': 'train', 'backend': 'b0',
                                                       'worker': 0} for i in range(14)}}
            # u0-u5 verifying, u6-u9 in their agent (a context), u10-u13 still building their environment
            c._heartbeats = {0: {'job_id': '1.0', 'contexts': {f'u{i}': {'context_tokens': 1000} for i in range(6, 10)},
                                 'post_agent': ['u0', 'u1', 'u2', 'u3', 'u4', 'u5']}}
            c.policy = lambda: {'mode': 'kv_fit', 'period_seconds': 0, 'eval_share': 0}
            c.shared_load = lambda ready, now: {}
            c.backend_sample = lambda name, backend, now: {'timestamp': time.time(), 'raw': {'kv': 0.1, 'wait': 0, 'run': 4}}
            status = c.admission_kv_fit({'b0': {'name': 'b0', 'url': 'http://x', 'kv_capacity_tokens': 10_000_000}})
            self.assertEqual(status['b0']['active'], 8, 'four agents + four builds (weight 1 by default)')
            self.assertEqual((c._post_agent, c._pre_agent), (6, 4))
            c.scale_workers(status)
            self.assertGreaterEqual(c.s['worker_demand'], 14, 'verifying and building attempts still hold worker slots')


class WorkerFleet(unittest.TestCase):
    """#241: the fleet follows the attempts that run, not the KV room; workers above the target drain and retire."""

    def fleet(self, d, items, workers=None, **state):
        from concurrent.futures import ThreadPoolExecutor
        c = object.__new__(controller_module.Controller)
        w = {'count': 2, 'max_count': 100, 'slots': 10, 'spare': 2, 'headroom': 0.25, 'shrink_after_sec': 900,
             **(workers or {})}
        c.c = mock.Mock(config={'workers': w, 'admission': {'mode': 'kv_fit'}}, id='t')
        c.ops = Path(d)
        c.pool = ThreadPoolExecutor(2)
        c._heartbeats = {}
        c.s = {'items': {f'u{n}': {'id': f'u{n}', 'status': 'active', 'kind': 'train', 'backend': 'b0',
                                   'worker': wk} for n, wk in enumerate(items)}, **state}
        return c

    def test_demand_is_live_attempts_plus_slot_shortage_not_kv_room(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.fleet(d, [0] * 20)
            c.scale_workers({'b0': {'active': 20, 'free_agents': 4000, 'limit': 4020}})
            self.assertEqual(c.s['worker_demand'], 20, 'KV room (free_agents) is not demand')
            self.assertEqual(c.s['worker_target'], 5, 'ceil(20 x 1.25 / 10) + 2 spare')
            c.s['worker_slot_short'] = 30
            c.scale_workers({'b0': {'active': 20, 'free_agents': 4000, 'limit': 4020}})
            self.assertEqual(c.s['worker_target'], 9, 'ceil(50 x 1.25 / 10) + 2')
            self.assertEqual(c.s['worker_high_water'], 9)

    def test_idle_workers_above_target_retire_after_the_hysteresis(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.fleet(d, [0, 1], worker_high_water=6, worker_target=3)
            c.scale_worker_fleet(2, math.inf, c.active())
            self.assertEqual(c.s['retired_workers'], [], 'startup and dips wait shrink_after_sec')
            c.s['worker_shrink_since'] = time.time() - 901
            c.scale_worker_fleet(2, math.inf, c.active())
            self.assertEqual(c.s['retired_workers'], [3, 4, 5])
            self.assertEqual((Path(d) / 'workers' / '5' / 'STOP').read_text(), 'synth-retired\n')
            self.assertEqual(c.s['draining_workers'], [])

    def test_busy_workers_drain_after_the_hysteresis_then_retire(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # 6 workers with attempts on each (an even spread), target 3
            items = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5]
            c = self.fleet(d, items, worker_high_water=6, worker_target=3)
            c.scale_worker_fleet(len(items), math.inf, c.active())
            self.assertEqual((c.s['retired_workers'], c.s['draining_workers']), ([], []), 'waits shrink_after_sec')
            c.s['worker_shrink_since'] = time.time() - 901
            c.scale_worker_fleet(len(items), math.inf, c.active())
            # loads 0:2 1:2 2:2 3:1 4:2 5:3 -> least-loaded first (3), ties to the higher index (4, 2); never 0
            self.assertEqual(sorted(c.s['draining_workers']), [2, 3, 4])
            c._heartbeats = {i: {'job_id': f'{i}.0', 'updated_at': time.time(), 'capacity': 10} for i in range(6)}
            with mock.patch.object(condor, 'latest', lambda campaign, role, i: {'job_id': f'{i}.0'}):
                self.assertEqual(c.ready_workers(6), [(2, 0), (2, 1), (3, 5)], 'dispatch skips draining workers')
            for item in c.s['items'].values():  # worker 3's attempt ends
                if item['worker'] == 3:
                    item['status'] = 'finished'
            c.scale_worker_fleet(len(items) - 1, math.inf, c.active())
            self.assertEqual(c.s['retired_workers'], [3])
            self.assertEqual(sorted(c.s['draining_workers']), [2, 4])

    def test_a_higher_target_cancels_drains_then_unretires(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.fleet(d, [0, 1, 2, 3], worker_high_water=6, worker_target=6, retired_workers=[5],
                           draining_workers=[3, 4], worker_shrink_since=time.time() - 5000)
            (Path(d) / 'workers' / '5').mkdir(parents=True)
            (Path(d) / 'workers' / '5' / 'STOP').write_text('synth-retired\n')
            c.scale_worker_fleet(4, math.inf, c.active())
            self.assertEqual((c.s['draining_workers'], c.s['retired_workers']), ([], []))
            self.assertFalse((Path(d) / 'workers' / '5' / 'STOP').exists())
            self.assertNotIn('worker_shrink_since', c.s)

    def test_dispatch_growth_signal_is_bounded_and_needs_work(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = self.fleet(d, [], workers={'grow_step': 2})
            c.proxy_ok = lambda policy: True
            c.ready_workers = lambda count: []          # every worker full (or none ready yet)
            c.worker_count = lambda: 2
            c.eval_candidates = lambda: []
            c.c.enabled = lambda kind: kind == 'train'
            c.s.update(train_stop=False, backends={})
            c.job_of = lambda name: 0
            status = {'b0': {'paused': False, 'room': 3000, 'limit': 3000, 'active': 0}}
            policy = {'mode': 'kv_fit', 'affinity_slack': 0.25, 'eval_share': 0}
            c.dispatch({}, status, policy)
            self.assertEqual(c.s['worker_slot_short'], 20, 'min(room 3000, grow_step 2 x 10 slots)')
            c.s['train_starved_since'] = time.time()        # the last dispatch that looked found no work
            c.dispatch({}, status, policy)
            self.assertEqual(c.s['worker_slot_short'], 0)
            c.s.pop('train_starved_since')
            c.s['train_stop'] = True                           # tail: training closed, KV room left
            c.dispatch({}, status, policy)
            self.assertEqual(c.s['worker_slot_short'], 0, 'no growth once training is closed')


class WorkerCheapBid(unittest.TestCase):
    """#242: workers start at workers.cheap_bid (CPU-only and A100 nodes) and are rebid to workers.bid when idle."""

    def test_worker_bid(self):
        self.assertEqual(condor.worker_bid({'bid': 40}), 40)
        self.assertEqual(condor.worker_bid({'bid': 40, 'cheap_bid': 25}), 25)
        self.assertEqual(condor.worker_bid({'bid': 40, 'cheap_bid': 56}), 40, 'never above workers.bid')

    def test_idle_cheap_worker_is_rebid_after_the_wait(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = object.__new__(controller_module.Controller)
            c.c = mock.Mock(config={'workers': {'bid': 40, 'cheap_bid': 25, 'cheap_bid_wait_sec': 600}}, id='t')
            c.ops = Path(d)
            c.s = {'items': {}, 'retired_workers': []}
            c._heartbeats = {}
            now = time.time()
            receipts = {0: {'job_id': '10.0', 'time': now - 700, 'generation': 0},
                        1: {'job_id': '11.0', 'time': now - 100, 'generation': 0},
                        2: {'job_id': '12.0', 'time': now - 700, 'generation': 0}}
            q = {'10.0': {'JobStatus': 1, 'JobPrio': -975}, '11.0': {'JobStatus': 1, 'JobPrio': -975},
                 '12.0': {'JobStatus': 1, 'JobPrio': -960}}    # 12.0: already at bid 40
            rebids = []
            with mock.patch.object(condor, 'latest', lambda campaign, role, i: receipts[i]), \
                    mock.patch.object(condor, 'rebid', lambda campaign, job, bid: rebids.append((job, bid))):
                c.ensure_workers(q, 3)
            self.assertEqual(rebids, [('10.0', 40)], 'only the cheap worker that waited 600 s')


class CheapCapacity(unittest.TestCase):
    """#242: count the CPU-only jobs that fit on cheap nodes before submitting, and split the bids by it."""

    def test_fits_keeps_the_gpu_reserve(self):
        self.assertEqual(condor.fits(64, 512 * 1024, 0, 12, 96 * 1024), 5, 'CPU-only node: 5 x 12 CPUs')
        # 2 free GPUs: keep min(16, 16) CPUs and min(64000, 64000) MB free -> (64 - 16) // 12 = 4
        self.assertEqual(condor.fits(64, 512 * 1024, 2, 12, 96 * 1024), 4)
        self.assertEqual(condor.fits(20, 512 * 1024, 4, 12, 96 * 1024), 0, '20 - 12 < 16 reserved')

    def test_cheap_capacity_counts_only_cheap_unexcluded_nodes(self):
        status = '\n'.join(['g1\t15\t64\t524288\t0', 'g2\t25\t30\t524288\t0', 'h1\t35\t200\t2000000\t0',
                            'b1\t55\t300\t3000000\t2', 'g9\t15\t64\t524288\t0', 'bad\tundefined\t8\t1\t0'])
        self.assertEqual(condor.cheap_capacity(12, 96, max_floor=25, exclude=['g9'], status=status), 5 + 2)

    def test_workers_beyond_cheap_room_go_out_at_the_normal_bid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            c = object.__new__(controller_module.Controller)
            c.c = mock.Mock(config={'workers': {'bid': 40, 'cheap_bid': 25, 'cpus': 12, 'memory_gib': 96}}, id='t')
            c.ops = Path(d)
            c.s = {'items': {}, 'retired_workers': []}
            c._heartbeats = {}
            submitted = []
            with mock.patch.object(condor, 'latest', lambda campaign, role, i: None), \
                    mock.patch.object(condor, 'receipts', lambda campaign: []), \
                    mock.patch.object(condor, 'cheap_capacity', lambda *a, **k: 2), \
                    mock.patch.object(condor, 'submit_worker',
                                      lambda campaign, index, gen, shape, cheap=True, avoid=(): submitted.append((index, cheap))):
                c.ensure_workers({}, 4)
            self.assertEqual(submitted, [(0, True), (1, True), (2, False), (3, False)])


class ServingOverrides(unittest.TestCase):
    def test_serving_jobs_override_drains_and_releases_then_restores(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = {'id': 'simcap', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                      'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                      'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': [f't{i}' for i in range(40)]},
                      'eval': {'enabled': False}, 'smoke': {'attempts': []},
                      'serving': {'jobs': 3, 'gpus_per_job': 1}, 'workers': {'count': 2, 'slots': 64}}
            campaign = Campaign(config)
            campaign.ops.mkdir(parents=True)
            cluster = FakeCluster(campaign)
            sub = lambda role: (lambda c, index, generation, *a, **k: cluster.submit(c, role, index=index, generation=generation))
            with mock.patch.object(condor, 'queue', cluster.queue), mock.patch.object(condor, 'query', cluster.query), \
                    mock.patch.object(condor, 'remove', cluster.remove), mock.patch.object(condor, 'release', cluster.remove), \
                    mock.patch.object(condor, 'submit_serving', sub('serving')), mock.patch.object(condor, 'submit_worker', sub('worker')):
                c = controller_module.Controller(campaign)
                c.s['stage'] = 'running'

                def tick():
                    condor._LISTINGS.clear()
                    q = condor.queue(campaign)
                    c.ensure_serving(q)
                    cluster.publish()
                    c._heartbeats = None
                    c.running(q, c.backends(q))
                tick()
                atomic(campaign.ops / 'admission-policy.json', {'serving_jobs': 1})
                tick()
                self.assertEqual(sorted(c.s['draining_serving'] + c.s['released_serving']), [1, 2])
                for item in c.active():  # attempts on the capped jobs finish
                    atomic(c.raw(item) / 'worker-result.json', {'finished_at': time.time(), 'interrupted': False})
                c.harvest()
                tick()
                self.assertEqual(sorted(c.s['released_serving']), [1, 2])
                self.assertTrue(all(i['backend'] == 'j0r0' for i in c.active()), 'new attempts only on the kept job')
                atomic(campaign.ops / 'admission-policy.json', {})
                tick()
                self.assertEqual(c.s['released_serving'], [])
                self.assertEqual(c.s['capped_serving'], {})


class ProxyBreaker(unittest.TestCase):
    def test_pauses_dispatch_while_the_proxy_fails_and_resumes_after_it_recovers(self):
        from concurrent.futures import Future
        c = object.__new__(controller_module.Controller)
        c.s = {}
        c.c = mock.Mock()
        c.ops = Path('/nonexistent-synth-test')
        answers = iter([(True, 'ok'), (False, 'x'), (False, 'x'), (False, 'x'), (True, 'ok'), (True, 'ok'), (True, 'ok')])

        def submit(fn, *a):
            f = Future()
            f.set_result(next(answers))
            return f
        c.pool = mock.Mock(submit=submit)
        policy = {'proxy_probe': {'interval_sec': 0, 'fail_after': 3, 'recover_after': 2}}
        with mock.patch.object(controller_module, 'best_effort'):
            seen = [c.proxy_ok(policy) for _ in range(7)]
        # each call harvests the previous probe, then starts the next one
        self.assertEqual(seen, [True, True, True, True, False, False, True])
        self.assertTrue(c.s['proxy']['ok'])
        self.assertEqual(c.proxy_ok({}), True, 'no probe configured: never pauses')


class WorkerCapacity(unittest.TestCase):
    def test_each_worker_fills_to_its_own_heartbeat_capacity(self):
        c = object.__new__(controller_module.Controller)
        c.c = mock.Mock(config={'workers': {'slots': 48}})
        c.s = {}
        now = time.time()
        c._heartbeats = {0: {'job_id': '1.0', 'updated_at': now, 'capacity': 24},   # submitted before slots = 48
                         1: {'job_id': '2.0', 'updated_at': now, 'capacity': 48},
                         2: {'job_id': '3.0', 'updated_at': now}}                     # older worker: config slots
        load = {0: 24, 1: 30, 2: 40}
        c.active = lambda kind=None, backend=None: [{'worker': w} for w, n in load.items() for _ in range(n)]
        latest = {0: '1.0', 1: '2.0', 2: '3.0'}
        with mock.patch.object(controller_module.condor, 'latest', lambda camp, role, i: {'job_id': latest[i]}):
            self.assertEqual(c.ready_workers(3), [(30, 1), (40, 2)])


class StarvedTail(unittest.TestCase):
    """Under an open "saved" budget the tail rule's remaining work is infinite; a resolved list with nothing to
    dispatch for starved_minutes and the load at most starved_fraction of capacity closes the tail anyway."""

    def check(self, ops, *, resolved, starved_minutes_ago, active):
        c = object.__new__(controller_module.Controller)
        c.c = mock.Mock(config={}, tasks=lambda kind: [f't{i}' for i in range(100)])
        c.ops = ops
        c.s = {'opened': [f't{i}' for i in range(resolved - 5)], 'invalid': [f'x{i}' for i in range(5)],
               'train_starved_since': time.time() - 60 * starved_minutes_ago}
        c.active = lambda kind=None, backend=None: [{}] * active
        c.max_concurrency_open = lambda: True
        c.serving_shared = lambda: []
        c.remaining_work = lambda: (float('inf'), 800)
        c.interrupt = mock.Mock()
        c.tail_check({'j0r0': {'limit': 1000}})
        return bool(c.s.get('tail_closed')), c.interrupt.called

    def test_closes_only_when_the_list_is_resolved_starved_and_drained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ops = Path(d)
            self.assertEqual(self.check(ops, resolved=100, starved_minutes_ago=6, active=400), (True, True))
            self.assertEqual(self.check(ops, resolved=100, starved_minutes_ago=6, active=600), (False, False))
            self.assertEqual(self.check(ops, resolved=100, starved_minutes_ago=2, active=400), (False, False))
            self.assertEqual(self.check(ops, resolved=50, starved_minutes_ago=6, active=400), (False, False),
                             'a lagging validation early in a run is not a tail')


if __name__ == '__main__':
    unittest.main()

class AdoptionSpec(unittest.TestCase):
    """adopt_ops compares the engine a serving job runs, not every field of the teacher spec."""

    def test_client_side_and_unused_spec_changes_still_adopt(self):
        from synth.controller import same_engine
        teacher = load_teacher('deepseek-v41-flash-1017')
        frozen = {'teacher': teacher, 'protocol': dict(PROTOCOL_DEFAULTS), 'serving': {'gpus_per_job': 4}}
        changed = json.loads(json.dumps(frozen))
        changed['teacher']['request']['env']['PI_CONTEXT_TOKENIZER'] = '/tok.json'      # client side
        changed['teacher']['serving']['optional_args']['unused_flag'] = ['--never-rendered']  # tunable unset
        self.assertTrue(same_engine(frozen, changed))
        engine = json.loads(json.dumps(frozen))
        engine['teacher']['serving']['tunables']['max_seqs'] = 1024                  # a different engine
        self.assertFalse(same_engine(frozen, engine))
        env = json.loads(json.dumps(frozen))
        env['teacher']['runtime']['env']['NCCL_NVLS_ENABLE'] = '1'
        self.assertFalse(same_engine(frozen, env))

    def test_revision_label_alone_still_adopts(self):
        """The engine serves the staged local checkpoint (no --revision): a corrected revision label (Qwen3.8 FP8,
        482ca0f -> 017b9c7a) must not cost a cold start; a different checkpoint must."""
        from synth.controller import same_engine
        teacher = load_teacher('deepseek-v41-flash-1017')
        frozen = {'teacher': teacher, 'protocol': dict(PROTOCOL_DEFAULTS), 'serving': {'gpus_per_job': 4}}
        rev = json.loads(json.dumps(frozen))
        rev['teacher']['revision'] = 'corrected-label'
        self.assertTrue(same_engine(frozen, rev))
        ckpt = json.loads(json.dumps(frozen))
        ckpt['teacher']['checkpoint'] = '/other/checkpoint'
        self.assertFalse(same_engine(frozen, ckpt))

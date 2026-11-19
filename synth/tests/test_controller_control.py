"""#235 (controller placement, condor_q, state save) and #244 (attempt failures, cap, host exclusion, exactly M,
worker memory) with fake Condor, fake slow saves and fake process tables."""
import asyncio
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest import mock

from synth import condor, controller as controller_module
from synth.common import atomic, read
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher
from synth.endpoint_relay import EndpointRelay
from synth.memory_guard import MemoryGuard
from synth.tests.test_controller_sim import FakeCluster, simulate

GiB = 2 ** 30


def campaign_in(d, cid='ctl', tasks=('a', 'b', 'c'), train=None, workers=None, controller=None):
    config = {'id': cid, 'root': d, 'teacher': load_teacher('nemotron35-super'),
              'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
              'train': {'mode': 'per_task', 'trajectories_per_task': 2, 'max_extra_attempts_per_task': 2,
                        'task_list': list(tasks), **(train or {})},
              'eval': {'enabled': False}, 'smoke': {'attempts': []},
              'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 2, 'slots': 4, **(workers or {})},
              'controller': controller or {}}
    campaign = Campaign(config)
    campaign.ops.mkdir(parents=True)
    return campaign


def finished(c, uid, task, failure=None, host='g1', at=None, **extra):
    c.s['items'][uid] = {'id': uid, 'kind': 'train', 'task': task, 'attempt': 0, 'status': 'finished', 'worker': 0,
                         'finished_at': time.time() if at is None else at, 'host': host,
                         **({'infrastructure_failure': failure} if failure else {}), **extra}
    return c.s['items'][uid]


# ---------------------------------------------------------------------------------------------------- condor_q (#235)
class QueueViewTest(unittest.TestCase):
    def view(self, read_queue, **kw):
        return condor.QueueView(None, read_queue=read_queue, **{'refresh_sec': 0.2, 'max_age_sec': 1.0, **kw})

    def test_a_slow_schedd_does_not_hold_the_tick(self):
        calls = []

        def slow():
            calls.append(time.monotonic())
            time.sleep(0.6)
            return {'1.0': {'JobStatus': 2, 'n': len(calls)}}
        view = self.view(slow)
        t = time.monotonic()
        self.assertEqual(view.get()['1.0']['n'], 1)           # the first read is waited for
        self.assertGreater(time.monotonic() - t, 0.5)
        time.sleep(0.3)                                        # older than refresh: a background read starts
        t = time.monotonic()
        self.assertEqual(view.get()['1.0']['n'], 1)           # ... and the cached snapshot serves at once
        self.assertLess(time.monotonic() - t, 0.1)
        time.sleep(0.7)
        self.assertEqual(view.get()['1.0']['n'], 2)           # the background read landed
        time.sleep(0.1)
        self.assertEqual(len(calls), 3)                        # (the call above started the next one)

    def test_a_failing_condor_q_stays_unknown_once_the_snapshot_is_too_old(self):
        state = {'fail': False}

        def flaky():
            if state['fail']:
                raise RuntimeError('schedd timeout')
            return {'1.0': {'JobStatus': 2}}
        view = self.view(flaky)
        self.assertIn('1.0', view.get())
        state['fail'] = True
        time.sleep(0.3)
        self.assertIn('1.0', view.get())                       # a fresh snapshot still serves while a read fails
        time.sleep(1.1)
        with self.assertRaises(RuntimeError):
            view.get()                                         # nothing fresh, and the read fails: unknown, never {}
        self.assertGreaterEqual(view.failures, 1)

    def test_after_a_submission_only_a_newer_read_is_waited_for(self):
        gate = threading.Event()
        seen = []

        def read_queue():
            seen.append(time.monotonic())
            if len(seen) == 2:
                gate.wait(5)            # the second read is slow: it started before the submission below
            return {'n': len(seen)}
        view = self.view(read_queue, refresh_sec=0.05, max_age_sec=0.1)
        view.get()
        time.sleep(0.06)
        view.get()                      # launches read 2 (slow), returns read 1
        condor._SUBMITTED[0] = time.monotonic()
        time.sleep(0.15)                # read 1 is too old now
        threading.Timer(0.2, gate.set).start()
        self.assertEqual(view.get(), {'n': 3}, 'a read that started after the submission')
        condor._SUBMITTED[0] = float('-inf')


class QueueQueryTest(unittest.TestCase):
    def test_attribute_plus_named_receipts_until_they_are_gone(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            jobs = campaign.ops / 'jobs'
            old = time.time() - 3600
            atomic(jobs / 'worker-0-g0-submission.json', {'job_id': '10.0', 'role': 'worker', 'index': 0,
                                                           'generation': 0, 'time': old})                 # no attribute
            atomic(jobs / 'worker-0-g1-submission.json', {'job_id': '11.0', 'role': 'worker', 'index': 0,
                                                           'generation': 1, 'time': old, 'campaign_attr': True})
            atomic(jobs / 'serving-0-g0-submission.json', {'job_id': '12.0', 'role': 'serving', 'index': 0,
                                                            'generation': 0, 'time': old})
            constraints = []

            def run(argv, **kw):
                constraints.append(argv[2])
                return mock.Mock(stdout=json.dumps([{'ClusterId': 11, 'ProcId': 0, 'JobStatus': 2},
                                                    {'ClusterId': 12, 'ProcId': 0, 'JobStatus': 2},
                                                    {'ClusterId': 99, 'ProcId': 0, 'JobStatus': 2}]))  # 99: no receipt
            condor._LISTINGS.clear()
            with mock.patch.object(condor.subprocess, 'run', run):
                for _ in range(3):
                    q = condor.queue(campaign)
            self.assertEqual(sorted(q), ['11.0', '12.0'], 'only receipted jobs')
            self.assertEqual(constraints[0], 'SynthCampaign == "ctl" || ClusterId == 10 || ClusterId == 12')
            self.assertEqual(constraints[-1], 'SynthCampaign == "ctl" || ClusterId == 12',
                             'a receipt absent from two reads is no longer named')

    def test_submit_sets_the_campaign_attribute(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.'))
            with mock.patch.object(condor.subprocess, 'run', run):
                condor.submit(campaign, 'worker', cpus=1, memory_gib=1, disk_gib=1, bid=1)
            sub = (campaign.ops / 'jobs' / 'worker-0-g0.sub').read_text()
            self.assertIn('+SynthCampaign = "ctl"', sub)
            self.assertTrue(read(campaign.ops / 'jobs' / 'worker-0-g0-submission.json')['campaign_attr'])

    def test_every_role_excludes_tbshareds_bad_nodes_read_at_submit_time(self):
        # uni-canary 11-18: synth controllers landed on gpu103, gpu123 and gpu166 (campaign lists only)
        from tbshared import condor_cost
        # no live additions here (condor_cost.BAD_NODES_FILE; tbshared/test_condor_cost.py covers them)
        live = mock.patch.dict(os.environ, {condor_cost.BAD_NODES_FILE_ENV: '/nonexistent/bad-nodes.txt'})
        live.start()
        self.addCleanup(live.stop)
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.'))
            with mock.patch.object(condor.subprocess, 'run', run):
                for role, index in (('controller', 0), ('worker', 1)):
                    condor.submit(campaign, role, index=index, cpus=1, memory_gib=1, disk_gib=1, bid=1,
                                  exclude=['cpu999'], fill=False)
                    sub = (campaign.ops / 'jobs' / f'{role}-{index}-g0.sub').read_text()
                    for node in [*condor_cost.BAD_NODES, 'cpu999']:
                        self.assertIn(f'(TARGET.UtsnameNodename != "{node}")', sub, (role, node))
                    receipt = read(campaign.ops / 'jobs' / f'{role}-{index}-g0-submission.json')
                    self.assertTrue(set(condor_cost.BAD_NODES) <= set(receipt['exclude']))
            listed = set(condor_cost.BAD_NODES)
            with mock.patch.object(condor_cost, 'BAD_NODES', ()):   # read from the file, not the imported constant
                self.assertEqual(condor.bad_nodes(), listed)
            for text, want in (("BAD_NODES = ('gpu999',  # EIO (11-12)\n             'gpu998')\n", {'gpu999', 'gpu998'}),
                               ("X = 1\nBAD_NODES = ['gpu997']\n", {'gpu997'}),
                               ("BAD_NODES = ('gpu103')\n", set(condor_cost.BAD_NODES)),      # a str, not a tuple
                               ("BAD_NODES = OTHER + ('g1',)\n", set(condor_cost.BAD_NODES))):
                src = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)   # an edited list, no restart
                src.write(text)
                src.close()
                with mock.patch.object(condor_cost, '__file__', src.name):
                    self.assertEqual(condor.bad_nodes(), want, text)
            with mock.patch.object(condor_cost, '__file__', '/nonexistent.py'):
                self.assertEqual(condor.bad_nodes(), set(condor_cost.BAD_NODES))
        import synth.datasets, synth.fetch   # their own submit files build requirements from condor (import check)
        self.assertIs(synth.fetch.condor, condor)
        self.assertIs(synth.datasets.condor, condor)


# ------------------------------------------------------------------------------------------------- placement (#235)
class PlacementTest(unittest.TestCase):
    def receipts(self, campaign, hosts):
        q = {}
        for i, host in enumerate(hosts):
            job = f'{100 + i}.0'
            atomic(campaign.ops / 'jobs' / f'worker-{i}-g0-submission.json',
                   {'job_id': job, 'role': 'worker', 'index': i, 'generation': 0, 'time': time.time()})
            q[job] = {'JobStatus': 2, 'RemoteHost': f'slot1_{i}@{host}.internal.cluster.internal'}
        condor._LISTINGS.clear()
        return q

    def test_workers_avoid_full_hosts_and_the_controller_host(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d, workers={'max_per_host': 3})
            q = self.receipts(campaign, ['g1', 'g1', 'g1', 'g2', 'g2', 'g3'])
            c = controller_module.Controller(campaign)
            c.host = 'g9'
            self.assertEqual(c.worker_avoid(q), ['g1', 'g9'])
            submitted = []
            with mock.patch.object(condor, 'submit_worker', lambda *a, **k: submitted.append(k['avoid'])):
                c.ensure_workers(q, 8)
            self.assertEqual(submitted, [['g1', 'g9']] * 2, 'indices 6 and 7 are submitted away from g1 and g9')

    def test_cheap_capacity_counts_room_per_host_under_max_per_host(self):
        # g1 has two partitionable slots (5 + 5 workers fit) and already runs 2 campaign workers; g2 fits 2, runs none;
        # h1 is too expensive. With max_per_host 3: g1 takes 1 more, g2 takes 2.
        status = '\n'.join(['g1\t15\t64\t524288\t0', 'g1\t15\t64\t524288\t0', 'g2\t25\t30\t524288\t0',
                            'h1\t35\t200\t2000000\t0'])
        self.assertEqual(condor.cheap_capacity(12, 96, status=status), 12, 'no cap: as before')
        self.assertEqual(condor.cheap_capacity(12, 96, status=status, per_host=3, placed={'g1': 2}), 1 + 2)
        self.assertEqual(condor.cheap_capacity(12, 96, status=status, per_host=3, placed={'g1': 4}), 2)

    def test_cheap_placement_composes_with_max_per_host(self):
        """#255 cheap bids x #244 max_per_host: room is counted per host under the cap, and every submission (cheap or
        not) excludes the hosts at the cap."""
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d, workers={'max_per_host': 3, 'cheap_bid': 25, 'bid': 40, 'cpus': 12,
                                               'memory_gib': 96})
            q = self.receipts(campaign, ['g1', 'g1', 'g1', 'g2'])
            status = '\n'.join(['g1\t15\t128\t1048576\t0', 'g2\t15\t128\t1048576\t0'])   # 10 fit on each
            c = controller_module.Controller(campaign)
            c.host = None
            submitted = []
            run = mock.Mock(return_value=mock.Mock(stdout=status))
            with mock.patch.object(condor.subprocess, 'run', run), \
                    mock.patch.object(condor, 'submit_worker',
                                      lambda camp, index, gen, shape, cheap=True, avoid=(): submitted.append((index, cheap, avoid))):
                c.ensure_workers(q, 8)
            # g1 is full (3); g2 has room for 2 more under the cap: workers 4, 5 cheap, 6, 7 at the normal bid
            self.assertEqual(submitted, [(4, True, ['g1']), (5, True, ['g1']), (6, False, ['g1']), (7, False, ['g1'])])

    def test_the_controller_avoids_worker_hosts_and_its_bad_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            q = self.receipts(campaign, ['g1', 'g2'])
            atomic(campaign.ops / 'controller' / 'excluded-nodes.json', ['gpu166'])
            run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 78.'))
            with mock.patch.object(condor.subprocess, 'run', run):
                condor.submit_controller(campaign, 0, q=q)
            sub = (campaign.ops / 'jobs' / 'controller-0-g0.sub').read_text()
            for node in ('g1', 'g2', 'gpu166', 'gpu106'):
                self.assertIn(f'(TARGET.UtsnameNodename != "{node}")', sub)

    def test_slow_ticks_move_the_controller_once(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d, controller={'relocate_slow_ticks': 3, 'relocate_tick_sec': 60})
            atomic(campaign.ops / 'jobs' / 'controller-0-g0-submission.json',
                   {'job_id': '5.0', 'role': 'controller', 'index': 0, 'generation': 0, 'time': time.time()})
            condor._LISTINGS.clear()
            c = controller_module.Controller(campaign)
            c.host = 'gpu166'
            moved = []
            with mock.patch.dict('os.environ', {'SYNTH_JOB_ID': '5.0'}), \
                    mock.patch.object(condor, 'submit_controller', lambda camp, gen, q=None: moved.append(gen) or '6.0'):
                for elapsed in (70, 80, 20, 70, 70):
                    c.placement_check(elapsed)
                self.assertEqual(moved, [], 'a fast tick resets the count')
                c.placement_check(65)
                self.assertEqual(moved, [1])
                self.assertTrue(c.stopping)
                self.assertEqual(read(campaign.ops / 'controller' / 'excluded-nodes.json'), ['gpu166'])
                c.stopping, c.slow_ticks = False, 0
                for _ in range(3):
                    c.placement_check(90)
                self.assertEqual(moved, [1], 'at most one move per 30 min')

    def test_io_errors_move_the_controller(self):
        import errno
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            c = controller_module.Controller(campaign)
            moved = []
            with mock.patch.dict('os.environ', {'SYNTH_JOB_ID': '5.0'}), \
                    mock.patch.object(condor, 'submit_controller', lambda camp, gen, q=None: moved.append(gen) or '6.0'):
                c.note_io_error(OSError(errno.EIO, 'Input/output error'))
                c.placement_check(5)
                self.assertEqual(moved, [])
                c.note_io_error(OSError(errno.ENOENT, 'not an I/O error'))
                c.note_io_error(OSError(errno.EIO, 'Input/output error'))
                c.placement_check(5)
                self.assertEqual(moved, [0])


# ------------------------------------------------------------------------------------------------ state save (#235)
class StateSaveTest(unittest.TestCase):
    def test_a_slow_write_never_holds_the_tick_and_the_journal_keeps_every_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d, controller={'state_save_interval_sec': 0})
            real = controller_module.write_text
            gate = threading.Event()

            def slow_write(path, text, retries=8):
                gate.wait(10)            # a /data stall
                real(path, text, retries)
            with mock.patch.object(controller_module, 'write_text', slow_write):
                c = controller_module.Controller(campaign)
                t = time.monotonic()
                c.save()                                         # snapshot 1 starts and hangs
                backend = {'name': 'j0r0', 'identity': 'x', 'url': 'http://fake/v1'}
                uid = c.assign('train', 'a', 0, 0, backend)      # journaled, inbox published
                c.save()                                         # snapshot 1 still in flight: deferred
                self.assertLess(time.monotonic() - t, 1.0)
                self.assertTrue((campaign.ops / 'workers' / '0' / 'inbox' / f'{uid}.json').exists())
                self.assertFalse((campaign.ops / 'controller' / 'state.json').exists())
                recovered = controller_module.Controller(campaign).s   # a crash now: snapshot missing
                self.assertIn(uid, recovered['items'])
                gate.set()
                c._save_future.result()
                c.save(force=True)
            saved = read(campaign.ops / 'controller' / 'state.json')
            self.assertIn(uid, saved['items'])
            self.assertEqual(sorted(p.name for p in (campaign.ops / 'controller' / 'journal').glob('*.jsonl')), [],
                             'segments a written snapshot covers are deleted')

    def test_snapshots_are_periodic_and_on_stage_changes(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d, controller={'state_save_interval_sec': 3600})
            writes = []
            with mock.patch.object(controller_module, 'write_text', lambda path, text, retries=8: writes.append(1)):
                c = controller_module.Controller(campaign)
                for _ in range(5):
                    c.save()
                    c._save_future and c._save_future.result()
                self.assertEqual(len(writes), 1)
                c.s['stage'] = 'running'
                c.save()
                c._save_future.result()
                self.assertEqual(len(writes), 2)

    def test_dispatch_counters_and_opened_tasks_replay(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = campaign_in(d)
            c = controller_module.Controller(campaign)
            c.save(force=True)
            c.s['opened'] = ['a', 'b']
            c.s['budget_cursor'] = 2
            c.durable([{'id': 'u1', 'kind': 'train', 'task': 'a', 'status': 'active'}])
            c.s['opened'].append('c')
            c.durable([{'id': 'u2', 'kind': 'train', 'task': 'c', 'status': 'active'}])
            line = (campaign.ops / 'controller' / 'journal').glob('*.jsonl').__next__().read_text().splitlines()[-1]
            self.assertEqual(json.loads(line)['extend'], {'opened': ['c']}, 'only the new entries')
            s = controller_module.Controller(campaign).s
            self.assertEqual(s['opened'], ['a', 'b', 'c'])
            self.assertEqual(s['budget_cursor'], 2)
            self.assertEqual(sorted(s['items']), ['u1', 'u2'])


# ---------------------------------------------------------------------------------- classification + caps (#244)
class ClassificationTest(unittest.TestCase):
    def test_no_model_request_is_a_pre_agent_failure_whatever_harbor_says(self):
        f = controller_module.infrastructure_failure
        agent_ran = {'agent_execution': {'started_at': 'x'}}
        timeout = {**agent_ran, 'exception_info': {'exception_type': 'AgentTimeoutError'}}
        self.assertEqual(f(timeout, {}, requests=False), 'PreAgentFailure/AgentTimeoutError')   # model relay start-up
        self.assertEqual(f({}, {'returncode': 1}, requests=False), 'PreAgentFailure/NoModelRequest')  # cont16k proxy
        self.assertIsNone(f(timeout, {}, requests=True))
        self.assertIsNone(f(timeout, {}, requests=None))           # no Pi logs to tell: as before

    def test_an_agent_command_failure_after_requests_is_the_agents_outcome(self):
        f = controller_module.infrastructure_failure
        nce = {'agent_execution': {'started_at': 'x'}, 'exception_info': {'exception_type': 'NetworkConnectionError'}}
        self.assertIsNone(f(nce, {}, requests=True))               # pi killed by the agent's own pkill
        self.assertEqual(f(nce, {}, requests=None), 'NetworkConnectionError')

    def test_model_requests_reads_the_cont16k_logs(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            self.assertIsNone(controller_module.model_requests(raw))
            (raw / '.pi-cont16k' / 'harbor-x').mkdir(parents=True)
            self.assertFalse(controller_module.model_requests(raw))
            (raw / 'harbor' / 'pi-cont16k').mkdir(parents=True)
            (raw / 'harbor' / 'pi-cont16k' / 'requests.jsonl').write_text('{}\n')
            self.assertTrue(controller_module.model_requests(raw))

    def test_cap_of_four_counts_pre_agent_and_mid_run_but_not_interruptions(self):
        with tempfile.TemporaryDirectory() as d:
            c = controller_module.Controller(campaign_in(d, tasks=('a',)))
            for n, failure in enumerate(['PreAgentFailure/BuildError', 'TrainingClosed', 'OwnedBackendRestart',
                                         'PreAgentFailure/BuildError', 'NetworkConnectionError']):
                finished(c, f'x{n}', 'a', failure, attempt=n)
            c.s['items']['x2']['controller_interruption'] = True
            self.assertEqual(c.tried()['a'], 0)
            self.assertEqual(c.next_train()[0], 'a')
            finished(c, 'x5', 'a', 'OwnedWorkerLost')
            self.assertEqual(c.tried()['a'], float('inf'), 'four failures (2 pre-agent, 2 mid-run) stop the task')
            self.assertIsNone(c.next_train())
            self.assertEqual(c.failure_summary(), {'pre_agent': 2, 'mid_run': 2, 'interruption': 2,
                                                   'tasks_stopped': 1, 'cap_per_task': 4})


class ExactlyMTest(unittest.TestCase):
    def usable_per_task(self, d, count_interrupted):
        campaign = campaign_in(d, tasks=('a', 'b', 'c', 'd'), train={'count_interrupted_usable': count_interrupted})
        cluster, exported = FakeCluster(campaign), {}

        def on_tick(tick, c):
            for item in c.active():
                if item['attempt'] % 2 == 1 and not item.get('infrastructure_failure'):
                    item['infrastructure_failure'] = 'OwnedWorkerLost'
        simulate(self, campaign, cluster, exported, on_tick=on_tick)
        usable = {}
        for record in exported.values():
            usable[record['task']] = usable.get(record['task'], 0) + bool(record['usable'])
        return usable

    def test_the_old_rule_gives_extra_traces(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(sum(self.usable_per_task(d, False).values()), 8)

    def test_usable_traces_of_interrupted_attempts_count_toward_m(self):
        """Every second attempt is cut by an infrastructure interruption but exports usable; each task still gets
        exactly M usable trajectories (m1-hero: 382 tasks got 597 beyond M = 2)."""
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.usable_per_task(d, True), {'a': 2, 'b': 2, 'c': 2, 'd': 2})


# --------------------------------------------------------------------------------------------- host exclusion (#244)
class HostExclusionTest(unittest.TestCase):
    def setup(self, d, **workers):
        campaign = campaign_in(d, tasks=tuple(f't{i}' for i in range(30)), workers=workers)
        atomic(campaign.ops / 'jobs' / 'worker-0-g0-submission.json',
               {'job_id': '200.0', 'role': 'worker', 'index': 0, 'generation': 0, 'time': time.time()})
        atomic(campaign.ops / 'jobs' / 'worker-1-g0-submission.json',
               {'job_id': '201.0', 'role': 'worker', 'index': 1, 'generation': 0, 'time': time.time()})
        for index, host in ((0, 'g90'), (1, 'g91')):
            atomic(campaign.ops / 'workers' / str(index) / 'heartbeat.json',
                   {'updated_at': time.time(), 'job_id': f'20{index}.0', 'host': f'{host}.internal', 'active': []})
        condor._LISTINGS.clear()
        return campaign, controller_module.Controller(campaign)

    def test_a_host_failing_every_attempt_is_excluded_and_its_workers_moved(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, c = self.setup(d)
            live = {'id': 'live', 'kind': 'train', 'task': 't0', 'attempt': 9, 'status': 'active', 'worker': 0}
            c.s['items']['live'] = live
            for n in range(40):                           # healthy elsewhere
                finished(c, f'ok{n}', f't{n % 30}', host='g91')
            c.host_health([c.s['items'][f'ok{n}'] for n in range(40)])
            items = [finished(c, f'x{n}', f't{n % 7}', 'PreAgentFailure/EnvironmentStartTimeoutError', host='g90',
                              at=1000 + n) for n in range(19)]
            removed = []
            with mock.patch.object(condor, 'remove', lambda camp, job: removed.append(job)):
                c.host_health(items)
                self.assertEqual(read(campaign.ops / 'excluded-nodes.json', []), [], '19 in a row is not enough')
                c.host_health([finished(c, 'x19', 't3', 'PreAgentFailure/NoModelRequest', host='g90', at=2000)])
            self.assertEqual(read(campaign.ops / 'excluded-nodes.json'), ['g90'])
            self.assertEqual(removed, ['200.0'])
            self.assertEqual(live['infrastructure_failure'], 'HostExcluded')
            self.assertEqual(controller_module.failure_phase(live), 'interruption')
            log = [json.loads(x) for x in (campaign.ops / 'host-exclusions.jsonl').read_text().splitlines()]
            self.assertEqual(log[0]['host'], 'g90')
            self.assertEqual(log[0]['consecutive_pre_agent_failures'], 20)
            self.assertIn('undo', log[0])
            self.assertEqual([i for _, i in c.ready_workers(2)], [1], 'no dispatch to the excluded host meanwhile')

    def test_a_success_resets_the_streak_and_a_cluster_wide_outage_excludes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, c = self.setup(d)
            pre = 'PreAgentFailure/EnvironmentStartTimeoutError'
            batch = [finished(c, f'x{n}', f't{n % 7}', pre, host='g90', at=n) for n in range(15)]
            batch += [finished(c, 'ok', 't1', host='g90', at=16)]
            batch += [finished(c, f'y{n}', f't{n % 7}', pre, host='g90', at=20 + n) for n in range(15)]
            c.host_health(batch)
            self.assertEqual(c.s['host_streaks']['g90']['n'], 15)
            outage = [finished(c, f'z{n}', f't{n % 9}', pre, host=f'g{100 + n % 5}', at=100 + n) for n in range(100)]
            c.host_health(outage)
            self.assertEqual(read(campaign.ops / 'excluded-nodes.json', []), [], 'every host fails: not the host')

    def test_disabled_with_zero(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, c = self.setup(d, exclude_host_after=0)
            c.host_health([finished(c, f'x{n}', f't{n}', 'PreAgentFailure/X', host='g90', at=n) for n in range(30)])
            self.assertEqual(read(campaign.ops / 'excluded-nodes.json', []), [])


# ---------------------------------------------------------------------------------------- worker memory guard (#244)
class MemoryGuardTest(unittest.TestCase):
    def table(self):
        # attempt A (root 10): harbor 10 + sandbox 11, 12 (other mount ns); attempt B (root 20): harbor 20 + child 21
        return {10: {'ppid': 1, 'rss': 1 * GiB, 'mnt': 'own'}, 11: {'ppid': 10, 'rss': 2 * GiB, 'mnt': 'boxA'},
                12: {'ppid': 11, 'rss': 1 * GiB, 'mnt': 'boxA'}, 20: {'ppid': 1, 'rss': 1 * GiB, 'mnt': 'own'},
                21: {'ppid': 20, 'rss': 9 * GiB, 'mnt': 'own'}}

    def guard(self, unreclaimable, shm=0, **kw):
        killed = []
        memory = {'limit': 96 * GiB, 'usage': 96 * GiB, 'anon': unreclaimable, 'shmem': 0, 'unreclaimable': unreclaimable}
        g = MemoryGuard(read_memory=lambda: dict(memory), read_table=self.table, own_mnt='own',
                        read_shm=lambda pid: shm, kill=lambda pid, sig: killed.append((pid, sig)), **kw)
        g.request = None
        return g, killed

    def test_below_the_threshold_nothing_happens_and_admission_stops_first(self):
        g, killed = self.guard(75 * GiB)
        self.assertIsNone(g.check({'A': 10, 'B': 20}))
        self.assertFalse(g.admits(), 'over 75%: no new attempts')
        g, _ = self.guard(60 * GiB)
        g.sample()
        self.assertTrue(g.admits())
        self.assertEqual(killed, [])

    def test_the_largest_attempt_is_stopped_its_sandbox_when_that_holds_the_memory(self):
        g, killed = self.guard(85 * GiB, shm=30 * GiB)        # A's /dev/shm holds 30 GiB: A is the largest
        record = g.check({'A': 10, 'B': 20})
        self.assertEqual(record['attempt'], 'A')
        self.assertEqual(sorted(p for p, _ in killed), [11, 12], 'only the sandbox; harbor wraps up')
        self.assertTrue(all(sig == signal.SIGKILL for _, sig in killed))
        self.assertIsNone(g.check({'A': 10, 'B': 20}), 'cooldown while the memory is freed')

    def test_memory_outside_the_sandbox_stops_the_whole_attempt(self):
        g, killed = self.guard(90 * GiB)
        record = g.check({'A': 10, 'B': 20})
        self.assertEqual(record['attempt'], 'B')
        self.assertEqual(sorted(p for p, _ in killed), [20, 21])

    def test_the_budget_is_the_smaller_of_the_cgroup_limit_and_the_request(self):
        g, _ = self.guard(0)
        g.request = 48 * GiB
        self.assertEqual(g.sample()['budget'], 48 * GiB)


# --------------------------------------------------------------------------------------- relay readiness (#244)
class RelayReadinessTest(unittest.TestCase):
    def test_logs_are_node_local_until_close_and_the_deadline_is_in_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            async def scenario():
                relay = EndpointRelay(Path(d) / 'logs')
                slow = await relay.spawn(['sh', '-c', 'echo first; sleep 1.5; echo "N listening on AF=2 x"; sleep 30'],
                                         'namespace-relay.log')
                self.assertFalse((Path(d) / 'logs' / 'namespace-relay.log').exists())
                t = time.monotonic()
                await relay.wait_for('listening on', 'namespace-relay.log', (slow,), 'relay', timeout=5)
                self.assertGreater(time.monotonic() - t, 1.0)    # longer than 10 polls: the old loop counted polls
                never = await relay.spawn(['sh', '-c', 'echo first; sleep 30'], 'other.log')
                with self.assertRaises(TimeoutError):
                    await relay.wait_for('listening on', 'other.log', (never,), 'relay', timeout=0.5)
                await relay.close()
                self.assertIn('listening on', (Path(d) / 'logs' / 'namespace-relay.log').read_text())
                self.assertTrue((Path(d) / 'logs' / 'other.log').exists())
            asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()

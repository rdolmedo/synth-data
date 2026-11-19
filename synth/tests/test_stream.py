"""Task streams (synth/stream.py, [train.stream]): chunk publishing and admission, a campaign that takes chunks while
it runs, the producer of task_synthesis, and the offline package proxy's refusals."""
import collections
from concurrent.futures import Future
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
import time
import tomllib
import unittest
from unittest import mock

from synth import controller as controller_module
from synth import stream
from synth.common import atomic, read
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher
from synth.tests.test_controller_sim import FakeCluster, simulate


def make_tasks(folder, names, timeout=600):
    folder = Path(folder)
    for name in names:
        (folder / name / 'tests').mkdir(parents=True, exist_ok=True)
        (folder / name / 'task.toml').write_text(f'[agent]\ntimeout_sec = {timeout}\n\n[verifier]\ntimeout_sec = 300\n')
        (folder / name / 'instruction.md').write_text(f'Do {name}.\n')
        (folder / name / 'tests' / 'test.sh').write_text('exit 0\n')
    return folder


class SyncPool:
    """ThreadPoolExecutor stand-in: work runs at submit, so a stream chunk is admitted within one tick."""

    def __init__(self, *a, **k):
        pass

    def map(self, fn, *iterables):
        return list(map(fn, *iterables))

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class Chunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {'SYNTH_ARCHIVE_STAGE': str(self.d)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def campaign(self, **train):
        config = {'id': 'st', 'root': str(self.d / 'campaigns'), 'teacher': load_teacher('nemotron35-super'),
                  'protocol': dict(PROTOCOL_DEFAULTS), 'train': {'mode': 'per_task', 'trajectories_per_task': 1,
                  'stream': {'dir': str(self.d / 'stream')}, **train}}
        campaign = Campaign(config)
        campaign.ops.mkdir(parents=True)
        return campaign

    def test_publish_commit_order_end_and_admit(self):
        src, s = make_tasks(self.d / 'src', ['t1', 't2', 't3']), self.d / 'stream'
        first = stream.publish(s, src, ['t2', 't1'], source={'round': 'r1'}, meta={'t1': 1, 't2': 2})
        self.assertEqual((first['index'], first['tasks']), (0, ['t2', 't1']))
        self.assertEqual(read(s / first['meta']), {'t1': 1, 't2': 2})
        with self.assertRaisesRegex(ValueError, 'already in'):
            stream.publish(s, src, ['t1'])
        second = stream.publish(s, src, ['t3'])
        atomic(s / 'chunks' / '0003.json', {'index': 3, 'tasks': ['x']})  # after a gap: not committed
        self.assertEqual([r['index'] for r in stream.committed(s)], [0, 1])
        self.assertEqual(stream.end(s, 'done')['chunks'], 2)
        with self.assertRaisesRegex(ValueError, 'ended'):
            stream.publish(s, src, ['t9'])
        self.assertFalse((s / 'publish.lock').exists())

        campaign = self.campaign()
        audit = stream.admit(campaign, first, s)
        self.assertEqual(audit['tasks_order'], ['t2', 't1'])
        self.assertEqual(stream.admit(campaign, first, s), audit, 'admission is idempotent')
        with tarfile.open(campaign.ops / 'stream' / 'train-0000.tar') as tar:
            names = {m.name for m in tar}
            toml = tomllib.loads(tar.extractfile('tasks/t1/task.toml').read().decode())
        self.assertIn('tasks/t2/tests/test.sh', names)
        self.assertEqual(toml['agent']['timeout_sec'], PROTOCOL_DEFAULTS['agent_timeout_sec'])
        self.assertEqual(toml['verifier']['timeout_sec'], 300, 'verifier limits are kept')
        self.assertEqual(audit['tasks']['t1']['original_agent'], {'timeout_sec': 600})
        self.assertEqual(stream.campaign_tars(campaign.root), [campaign.ops / 'stream' / 'train-0000.tar'])
        (s / second['tar']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'sha256'):
            stream.admit(campaign, second, s)

    def test_publish_refuses_a_task_campaigns_cannot_admit(self):
        src, s = make_tasks(self.d / 'src', ['ok', 'empty']), self.d / 'stream'
        (src / 'empty' / 'task.toml').write_text('')
        with self.assertRaisesRegex(ValueError, 'cannot admit.*empty'):
            stream.publish(s, src, ['ok', 'empty'])
        self.assertEqual(stream.committed(s), [])
        self.assertFalse((s / 'publish.lock').exists())

    def test_admission_leaves_out_a_bad_task_instead_of_blocking_the_stream(self):
        src, s = make_tasks(self.d / 'src', ['a', 'b']), self.d / 'stream'
        record = stream.publish(s, src, ['a', 'b'])
        # a chunk published around the check (another producer, an older version): b has no [agent].timeout_sec
        import io
        local = self.d / 'bad.tar'
        with tarfile.open(s / record['tar']) as old, tarfile.open(local, 'w') as new:
            for m in old:
                data = old.extractfile(m) if m.isfile() else None
                if m.name == 'tasks/b/task.toml':
                    body = b'[verifier]\ntimeout_sec = 300\n'
                    m.size, data = len(body), io.BytesIO(body)
                new.addfile(m, data)
        local.replace(s / record['tar'])
        from synth.common import sha
        record['sha256'] = sha(s / record['tar'])
        audit = stream.admit(self.campaign(), record, s)
        self.assertEqual((audit['tasks_order'], audit['count']), (['a'], 1))
        self.assertIn('timeout_sec', audit['invalid']['b'])

    def test_a_dead_producers_lock_is_taken_over(self):
        src, s = make_tasks(self.d / 'src', ['a']), self.d / 'stream'
        (s / 'publish.lock').mkdir(parents=True)
        atomic(s / 'publish.lock' / 'owner.json', {'host': __import__('socket').gethostname(), 'pid': 2 ** 22 + 7})
        (s / 'publish.lock' / '.owner.json.1.0.tmp').write_text('{')   # a crash mid-write left this
        self.assertEqual(stream.publish(s, src, ['a'])['index'], 0)
        (s / 'publish.lock').mkdir()
        atomic(s / 'publish.lock' / 'owner.json', {'host': 'elsewhere', 'pid': 1})
        with self.assertRaisesRegex(RuntimeError, 'one producer per stream'):
            stream.end(s)

    def test_worker_skips_the_missing_frozen_bundle_and_extracts_chunks_once(self):
        import asyncio
        from synth.worker import Worker
        src, s = make_tasks(self.d / 'src', ['t1', 't2']), self.d / 'stream'
        record = stream.publish(s, src, ['t1', 't2'])
        campaign = self.campaign()
        campaign.freeze()
        audit = stream.admit(campaign, record, s)
        scratch = self.d / 'scratch'
        scratch.mkdir()
        with mock.patch.dict(os.environ, {'SYNTH_CAMPAIGN': str(campaign.frozen_path),
                                          '_CONDOR_SCRATCH_DIR': str(scratch)}):
            worker = Worker(0, 4)
            worker.stage()   # a stream-only campaign has no ops/train_tasks.tar
            self.assertFalse((scratch / 'train' / 'READY').exists())

            async def two_attempts():
                calls = []
                real = tarfile.open
                with mock.patch.object(tarfile, 'open', lambda *a, **k: calls.append(a) or real(*a, **k)):
                    await asyncio.gather(worker.extract_chunk(audit['path'], 'train'),
                                         worker.extract_chunk(audit['path'], 'train'))
                return calls
            self.assertEqual(len(asyncio.run(two_attempts())), 1, 'two attempts of one chunk extract it once')
        toml = tomllib.loads((scratch / 'train' / 'tasks' / 't2' / 'task.toml').read_text())
        self.assertEqual(toml['agent']['timeout_sec'], PROTOCOL_DEFAULTS['agent_timeout_sec'])

    def test_prepare_accepts_a_stream_only_campaign(self):
        from synth.config import build
        path = self.d / 'c.toml'
        path.write_text(f'id = "x"\nteacher = "nemotron35-super"\nroot = "{self.d}"\n[train]\nmode = "per_task"\n'
                        f'trajectories_per_task = 1\n[train.stream]\ndir = "{self.d / "stream"}"\n')
        config = build(path)
        self.assertEqual(config['train']['task_list'], [])
        path.write_text(path.read_text().replace(str(self.d / 'stream'), 'relative/stream'))
        with self.assertRaisesRegex(ValueError, 'absolute'):
            build(path)


class StreamingCampaign(unittest.TestCase):
    def config(self, d, **train):
        return {'id': 'sim', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'max_extra_attempts_per_task': 2,
                          'stream': {'dir': str(Path(d) / 'stream'), 'poll_sec': 0}, **train},
                'eval': {'enabled': False}, 'smoke': {'attempts': []},
                'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 2, 'slots': 4}}

    def test_takes_chunks_while_running_and_completes_after_the_end(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'SYNTH_ARCHIVE_STAGE': d}), \
                mock.patch.object(controller_module, 'ThreadPoolExecutor', SyncPool):
            src, s = make_tasks(Path(d) / 'src', ['s-a', 's-b', 's-x', 's-y']), Path(d) / 'stream'
            stream.publish(s, src, ['s-b', 's-a'])
            campaign = Campaign(self.config(d))
            campaign.ops.mkdir(parents=True)
            seen = {}

            def on_tick(tick, c):
                if tick == 8:
                    seen['waiting'] = (c.s['stage'], c.stream_open(), c.train_target_reached(), c.pending_work(),
                                       sorted(i['task'] for i in c.s['items'].values()))
                    stream.publish(s, src, ['s-y', 's-x'])
                if tick == 14:
                    stream.end(s, 'test')
            c, _ = simulate(self, campaign, FakeCluster(campaign), {}, ticks=40, on_tick=on_tick)
            self.assertEqual(seen['waiting'], ('running', True, False, math.inf, ['s-a', 's-b']),
                             'an open stream with every listed task done keeps the campaign running')
            self.assertEqual(c.train_list(), ['s-b', 's-a', 's-y', 's-x'])
            items = {i['task']: i for i in c.s['items'].values()}
            self.assertEqual(sorted(items), ['s-a', 's-b', 's-x', 's-y'])
            self.assertTrue(items['s-x']['bundle'].endswith('ops/stream/train-0001.tar'))
            self.assertTrue(items['s-a']['bundle'].endswith('ops/stream/train-0000.tar'))
            chunks = read(campaign.ops / 'stream' / 'chunks.json')
            self.assertEqual([ch['count'] for ch in chunks['chunks']], [2, 2])
            completion = read(campaign.ops / 'completion.json')
            self.assertEqual((completion['stream']['chunks'], completion['stream']['open']), (2, False))
            self.assertEqual(read(campaign.train / 'manifest.json')['stream']['tasks'], 4)

    def controller(self, d, state, **train):
        campaign = Campaign(self.config(d, **train))
        campaign.ops.mkdir(parents=True, exist_ok=True)
        c = controller_module.Controller(campaign)
        c.s.update(state)
        return c

    def test_open_stream_is_more_work_coming(self):
        with tempfile.TemporaryDirectory() as d:
            st = {'stream': {'dir': d, 'chunks': [{'count': 2, 'tar': '/x/train-0000.tar'}], 'tasks': ['a', 'b'],
                             'end': None}, 'opened': ['a', 'b'], 'budget_cursor': 2}
            c = self.controller(d, st, tokens_budget=1000, budget_stop='saved')
            self.assertFalse(c.budget_exhausted())
            self.assertTrue(c.budget_open())
            self.assertEqual(c.pending_work(), math.inf)
            self.assertEqual(c.remaining_work()[0], math.inf)
            self.assertFalse(c.train_target_reached())
            self.assertEqual(c.stream_bundle('b'), '/x/train-0000.tar')
            c.s['stream']['end'] = {'chunks': 1}
            self.assertTrue(c.budget_exhausted(), 'END after the admitted chunks: the list is exhausted for good')
            self.assertFalse(c.budget_open())
            c.s['stream']['end'] = {'chunks': 2}
            self.assertTrue(c.stream_open(), 'END counts a chunk not admitted yet')

    def test_a_projected_budget_that_is_spent_closes_training_with_the_stream_open(self):
        with tempfile.TemporaryDirectory() as d:
            st = {'stream': {'dir': d, 'chunks': [{'count': 2, 'tar': 't'}], 'tasks': ['a', 'b'], 'end': None},
                  'opened': ['a', 'b'], 'budget_cursor': 2, 'frontier': 2}
            c = self.controller(d, st, tokens_budget=1000, budget_stop='projected')
            c.train_counts = lambda finalized_only=False: collections.Counter({'a': 1, 'b': 1})
            c.totals = lambda: (1500, 2, 2)
            c.unexported_finished = lambda: []
            self.assertFalse(c.budget_open())
            self.assertTrue(c.train_target_reached(), 'a spent budget outranks the open stream')
            self.assertEqual(c.pending_work(), 0)
            c.totals = lambda: (400, 2, 2)   # the budget opens tasks again: wait for the stream
            c._saved_cache = None            # (cached per tick)
            self.assertTrue(c.budget_open())
            self.assertFalse(c.train_target_reached())
            self.assertEqual(c.pending_work(), math.inf)

    def test_skip_file_passes_over_a_chunk_that_cannot_be_admitted(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'SYNTH_ARCHIVE_STAGE': d}), \
                mock.patch.object(controller_module, 'ThreadPoolExecutor', SyncPool):
            src, s = make_tasks(Path(d) / 'src', ['a', 'b']), Path(d) / 'stream'
            first = stream.publish(s, src, ['a'])
            stream.publish(s, src, ['b'])
            stream.end(s)
            (s / first['tar']).write_bytes(b'corrupt')
            c = self.controller(d, {'stage': 'running'})
            for _ in range(3):
                c.stream_tick()
            self.assertEqual((c.s['stream']['chunks'], c.stream_open()), ([], True), 'a corrupt chunk holds the stream')
            atomic(c.ops / 'stream' / 'skip.json', {'0': 'tar corrupted on /data'})
            for _ in range(4):
                c.stream_tick()
            self.assertEqual([ch.get('skipped') for ch in c.s['stream']['chunks']], ['tar corrupted on /data', None])
            self.assertEqual([ch.get('skipped') for ch in read(c.ops / 'stream' / 'chunks.json')['chunks']],
                             ['tar corrupted on /data', None])
            self.assertEqual((c.train_list(), c.stream_open()), (['b'], False))
            self.assertTrue(c.stream_bundle('b').endswith('train-0001.tar'))

    def test_per_task_target_waits_for_the_end(self):
        with tempfile.TemporaryDirectory() as d:
            st = {'stream': {'dir': d, 'chunks': [{'count': 1, 'tar': 't'}], 'tasks': ['a'], 'end': None}}
            c = self.controller(d, st)
            c.train_counts = lambda finalized_only=False: collections.Counter({'a': 1})
            self.assertFalse(c.train_target_reached())
            c.s['stream']['end'] = {'chunks': 1}
            self.assertTrue(c.train_target_reached())

    def test_tokens_mode_waits_for_chunks_before_cycling(self):
        with tempfile.TemporaryDirectory() as d:
            st = {'stream': {'dir': d, 'chunks': [{'count': 2, 'tar': 't'}], 'tasks': ['a', 'b'], 'end': None}}
            c = self.controller(d, st, mode='tokens', tokens_target=10 ** 9)
            plan = {'issued': collections.Counter()}
            self.assertEqual([c.next_train(plan), c.next_train(plan), c.next_train(plan)], [('a', 0), ('b', 0), None])
            c.s['stream']['end'] = {'chunks': 1}
            self.assertEqual(c.next_train(plan), ('a', 1), 'after the end, tasks cycle as before')

    def test_starved_tail_rule_holds_while_the_stream_is_open(self):
        c = object.__new__(controller_module.Controller)
        c.c = mock.Mock(config={}, tasks=lambda kind: [f't{i}' for i in range(100)], enabled=lambda kind: True)
        c.ops = Path(tempfile.mkdtemp())
        c.s = {'opened': [f't{i}' for i in range(100)], 'invalid': [], 'train_starved_since': time.time() - 3600}
        c.active = lambda kind=None, backend=None: [{}] * 10
        c.max_concurrency_open = lambda: True
        c.serving_shared = lambda: []
        c.remaining_work = lambda: (math.inf, 800)
        c.interrupt = mock.Mock()
        c.stream_wanted = lambda: True
        c.tail_check({'j0r0': {'limit': 1000}})
        self.assertIsNone(c.s.get('tail_closed'))
        c.stream_wanted = lambda: False
        c.tail_check({'j0r0': {'limit': 1000}})
        self.assertTrue(c.s.get('tail_closed'), 'after the end the starved rule applies as before')

    def test_streamed_smoke_task_waits_for_its_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            campaign = Campaign({**self.config(d), 'smoke': {'attempts': [{'task': 's-a', 'kind': 'train'}]}})
            campaign.ops.mkdir(parents=True)
            c = controller_module.Controller(campaign)
            c.ready_workers = mock.Mock(return_value=[])
            c.smoke({}, {'j0r0': {}})
            c.ready_workers.assert_not_called()
            self.assertIn('Smoke waits for', (campaign.ops / 'status.log').read_text())
            c.s['stream'] = {'dir': d, 'chunks': [], 'tasks': [], 'end': {'chunks': 0}}
            with self.assertRaisesRegex(RuntimeError, 'stream has ended'):
                c.smoke({}, {'j0r0': {}})

    def test_failed_admission_is_logged_and_retried(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(controller_module, 'ThreadPoolExecutor', SyncPool):
            c = self.controller(d, {'stage': 'running'})
            atomic(Path(d) / 'stream' / 'chunks' / '0000.json', {'index': 0, 'tasks': ['a'], 'tar': 'chunks/0000.tar',
                                                                   'sha256': 'f' * 64})
            with mock.patch.object(controller_module.stream_chunks, 'admit', side_effect=ValueError('bad tar')):
                c.stream_tick()   # submits (synchronously fails)
                c.stream_tick()   # collects the failure, polls again
            self.assertEqual(c.s['stream']['chunks'], [])
            self.assertIn('bad tar', c.s['stream']['error']['error'])
            self.assertIn('retrying', (c.ops / 'status.log').read_text())


class PackageProxyRefusals(unittest.TestCase):
    def test_local_hosts_get_a_bypass_hint(self):
        from synth.package_proxy import local_host, refusal
        for host in ('127.0.0.1', '127.0.0.10', '10.42.0.10', '::1', 'localhost', 'api.lab.internal', 'admin.corp',
                     'nas.home.arpa', 'homeserver.local', 'application', ''):
            self.assertTrue(local_host(host), host)
        for host in ('github.com', 'raw.githubusercontent.com', '8.8.8.8', 'download.pytorch.org'):
            self.assertFalse(local_host(host), host)
        self.assertIn('connect to it directly', refusal('127.0.0.1'))
        self.assertEqual(refusal('github.com'), 'Forbidden: only the Ubuntu, PyPI and npm registries are reachable')
        self.assertNotIn('\r', refusal('evil\r\nX: y.internal'))


class NearDuplicatesAgainstPublished(unittest.TestCase):
    def test_matches_the_full_search_on_pairs_with_a_new_text(self):
        from task_synthesis.pipeline.textsim import near_duplicates_fast, near_duplicates_new
        words = [f'w{i}' for i in range(400)]
        texts = {f'old{k}': ' '.join(words[k * 10:k * 10 + 10]) for k in range(20)}
        texts.update({'new0': texts['old3'] + ' extra', 'new1': ' '.join(words[300:310]),
                      'new2': ' '.join(words[300:310]) + ' more', 'new3': ' '.join(words[350:360])})
        new = ['new0', 'new1', 'new2', 'new3']
        full = [p for p in near_duplicates_fast(texts) if p[0] in new or p[1] in new]
        self.assertEqual(near_duplicates_new(texts, new), full)
        self.assertEqual({(a, b) for a, b, _ in full}, {('old3', 'new0'), ('new1', 'new2')})


class Producer(unittest.TestCase):
    """task_synthesis.pipeline.campaign stream: judge-passed round tasks -> chunks, new near-duplicates dropped."""

    def round(self, rounds, name, specs):
        d = rounds / name
        (d / 'work').mkdir(parents=True, exist_ok=True)
        lines = []
        for sid, text, passed in specs:
            lines.append(json.dumps({'spec_id': sid}))
            if passed is None:
                continue
            atomic(d / 'work' / sid / 'record.json', {'spec_id': sid, 'status': 'valid', 'instruction': text,
                   'judge': {'pass': passed}, 'meta': {'slug': sid, 'services': False}, 'spec': {'minutes': 30}})
            make_tasks(d / 'tasks', [sid])
            (d / 'tasks' / sid / 'instruction.md').write_text(text)
        (d / 'specs.jsonl').write_text('\n'.join(lines) + '\n')

    def test_publishes_new_tasks_and_drops_near_duplicates_of_earlier_chunks(self):
        from task_synthesis.pipeline import campaign as producer
        words = [f'w{i}' for i in range(400)]
        text = lambda k: ' '.join(words[k * 10:k * 10 + 10])
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'SYNTH_ARCHIVE_STAGE': d}), \
                mock.patch.object(producer, 'ROUNDS', Path(d) / 'rounds'), \
                mock.patch.object(producer, 'CAMPAIGNS', Path(d) / 'campaigns'), \
                mock.patch('task_synthesis.pipeline.textsim.decon',
                           lambda texts: {k: {'flagged': k == 's1-04'} for k in texts}):
            rounds, s = Path(d) / 'rounds', Path(d) / 'stream'
            self.round(rounds, 'r1', [('s1-01', text(1), True), ('s1-02', text(2), True), ('s1-03', text(3), False),
                                      ('s1-05', '', None)])   # s1-05 still running: no record yet
            p = producer.StreamProducer(['r1'], s)
            self.assertFalse(producer.round_done('r1'))
            first = p.publish()
            self.assertEqual(sorted(first['tasks']), ['tasksynth-s1-01', 'tasksynth-s1-02'])
            self.assertIsNone(p.publish(), 'nothing new')
            self.round(rounds, 'r1', [('s1-01', text(1), True), ('s1-02', text(2), True), ('s1-03', text(3), False),
                                      ('s1-04', text(4), True), ('s1-05', text(1) + ' ' + text(30), True),
                                      ('s1-06', text(6), True), ('s1-07', text(30), True), ('s1-08', text(8), True)])
            (rounds / 'r1' / 'tasks' / 's1-08' / 'task.toml').write_text('')   # emptied by an overlapping job
            self.assertTrue(producer.round_done('r1'))
            p.follow(interval=0, min_tasks=50, end=True)   # the round is done: what is left, then END
            (second,) = stream.committed(s)[1:]
            # s1-05 = s1-01 + s1-07's text: dropped as a near-duplicate of s1-01; s1-07 duplicates only the dropped
            # s1-05, so it is kept
            self.assertEqual(sorted(second['tasks']), ['tasksynth-s1-06', 'tasksynth-s1-07'])
            rejected = read(s / 'producer' / 'rejected.json')
            self.assertEqual(rejected['tasksynth-s1-04']['reason'], 'decon')
            self.assertEqual(rejected['tasksynth-s1-08']['reason'], 'invalid_task')
            self.assertEqual((rejected['tasksynth-s1-05']['reason'], rejected['tasksynth-s1-05']['of']),
                             ('near_duplicate', 's1-01'))
            self.assertNotIn('tasksynth-s1-07', rejected)
            # repaired after the stream ended: a new stream reconsiders it (decon and duplicate verdicts stand)
            make_tasks(rounds / 'r1' / 'tasks', ['s1-08'])
            s2 = Path(d) / 'stream2'
            (s2 / 'producer').mkdir(parents=True)
            atomic(s2 / 'producer' / 'rejected.json', rejected)
            third = producer.StreamProducer(['r1'], s2).publish()
            self.assertIn('tasksynth-s1-08', third['tasks'])
            self.assertNotIn('tasksynth-s1-04', third['tasks'])
            self.assertNotIn('tasksynth-s1-08', read(s2 / 'producer' / 'rejected.json'))
            # still-invalid tasks neither count toward min_tasks nor trigger the near-duplicate pass
            self.round(rounds, 'r2', [('s2-01', text(11), True), ('s2-02', text(12), True), ('s2-03', text(13), True)])
            for sid in ('s2-02', 's2-03'):
                (rounds / 'r2' / 'tasks' / sid / 'task.toml').write_text('')
            p2 = producer.StreamProducer(['r2'], s2)
            with mock.patch('task_synthesis.pipeline.textsim.near_duplicates_new') as pass_:
                self.assertIsNone(p2.publish(min_tasks=2))
                pass_.assert_not_called()
            self.assertEqual(read(s2 / 'producer' / 'rejected.json')['tasksynth-s2-02']['reason'], 'invalid_task')
            self.assertEqual(p2.publish(min_tasks=1)['tasks'], ['tasksynth-s2-01'])
            self.assertEqual(stream.ended(s)['chunks'], 2)
            manifest = read(s / 'manifest.json')
            self.assertEqual(sorted(manifest), ['tasksynth-s1-01', 'tasksynth-s1-02', 'tasksynth-s1-06', 'tasksynth-s1-07'])
            self.assertEqual(manifest['tasksynth-s1-06']['round'], 'r1')
            self.assertEqual(len(read(s / 'smoke.json')), 2)


if __name__ == '__main__':
    unittest.main()

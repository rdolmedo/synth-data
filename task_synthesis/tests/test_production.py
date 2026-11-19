"""Offline tests for the M1 production chain (#118): real-asset presets, effort scope, streamed binding in the round,
the production planner's eligibility and sizing, the campaign template and the driver's stop rule.

  python -m unittest task_synthesis.tests.test_production     (harbor venv)
"""
import asyncio
import json
import math
import tempfile
import tomllib
import unittest
from argparse import Namespace
from pathlib import Path

from task_synthesis.common import REPO
from task_synthesis.pipeline import production, round as R


class Presets(unittest.TestCase):
    def test_wx_sets_the_4th_ab_recipe(self):
        a = R.parse_args(['run', '--round', 'x', '--real-assets', 'wx'])
        self.assertEqual((a.author, a.workflow, a.judge_repairs, a.judge_effort, a.reasoning_effort, a.author_tokens,
                          a.author_commands), ('workspace', 'lean3', 1, 'high', 'high', 128000, 60))
        self.assertEqual(a.effort_scope, 'asset')

    def test_wy_was_removed(self):
        with self.assertRaises(SystemExit):
            R.parse_args(['submit', '--round', 'x', '--real-assets', 'wy'])

    def test_oneshot_and_off_keep_the_one_shot_author(self):
        for preset in ('oneshot', 'off'):
            a = R.parse_args(['run', '--round', 'x', '--real-assets', preset])
            self.assertEqual((a.author, a.reasoning_effort, a.judge_effort), ('oneshot', None, None))

    def test_target_inflight_default_depends_on_the_author(self):
        self.assertEqual(R.parse_args(['run', '--round', 'x', '--real-assets', 'wx']).target_inflight, 1200)
        self.assertEqual(R.parse_args(['run', '--round', 'x', '--real-assets', 'oneshot']).target_inflight, 0)
        self.assertEqual(R.parse_args(['run', '--round', 'x']).target_inflight, 0)       # one-shot without a preset
        self.assertEqual(R.parse_args(['run', '--round', 'x', '--real-assets', 'oneshot',
                                       '--target-inflight', '2000']).target_inflight, 2000)

    def test_conflicting_flag_is_an_error_and_equal_flag_is_fine(self):
        with self.assertRaises(SystemExit):
            R.parse_args(['run', '--round', 'x', '--real-assets', 'wx', '--reasoning-effort', 'low'])
        a = R.parse_args(['run', '--round', 'x', '--real-assets', 'wx', '--author-tokens', '128000'])
        self.assertEqual(a.author_tokens, 128000)

    def test_submit_args_round_trip(self):
        """`round submit` passes the parsed options to the job; the job must parse them back to the same recipe."""
        a = R.parse_args(['submit', '--round', 'x', '--real-assets', 'wx', '--gen', 'a+b', '--judge', 'a+b'])
        args = [w for k, v in vars(a).items()
                if k not in R.SUBMIT_ONLY
                and v is not None and v is not False
                for w in ([f'--{k.replace("_", "-")}'] + ([] if v is True else [str(v)]))]
        b = R.parse_args(['run'] + args)
        for k in ('author', 'workflow', 'judge_repairs', 'judge_effort', 'reasoning_effort', 'author_tokens',
                  'author_commands', 'effort_scope', 'bind_window', 'real_assets', 'vocab', 'minutes_mix'):
            self.assertEqual(str(getattr(a, k)), str(getattr(b, k)), k)


class Admission(unittest.TestCase):
    """round.admit: one-shot specs are gated by the pool's load (no session cap); admission.json overrides live."""

    def round(self, tmp, author, load, sessions_open=0):
        r = R.Round.__new__(R.Round)
        r.a = Namespace(author=author, target_inflight=1000, target_sessions=100, build_parallel=4)
        r.dir, r.shards, r.open_authors = Path(tmp), 2, sessions_open
        r.build_slots = asyncio.Semaphore(4)
        r.logged = []
        r.log = r.logged.append

        async def pool_inflight():
            return load, 2
        r.pool_inflight = pool_inflight
        return r

    def admitted(self, r):
        async def go():
            try:
                await asyncio.wait_for(r.admit(), 0.3)
                return True
            except asyncio.TimeoutError:
                return False
        return asyncio.run(go())

    def test_oneshot_is_gated_by_load_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1,500 in flight on 2 servers, target 1,000 per server: room; 500 open specs would break a session cap of
            # 100 per server, which does not apply to one-shot specs
            self.assertTrue(self.admitted(self.round(tmp, 'oneshot', 1500, sessions_open=500)))
            self.assertFalse(self.admitted(self.round(tmp, 'oneshot', 2500)))
            self.assertFalse(self.admitted(self.round(tmp, 'workspace', 1500, sessions_open=500)))

    def test_unknown_load_holds_and_reads_are_single_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.round(tmp, 'oneshot', 0)
            del r.pool_inflight

            async def none():
                return None
            r.pool_inflight = none
            self.assertFalse(self.admitted(r))          # a server's load could not be read: no admission
            # the real pool_inflight: many concurrent callers share one read (stale cache at a job's start)
            r2 = R.Round.__new__(R.Round)
            r2.gen = Namespace(load={'http://a/v1': 0, 'http://b/v1': 0})
            calls = []

            def fake(url, timeout):
                calls.append(url)
                raise OSError('slow server')
            orig = R.urllib.request.urlopen
            R.urllib.request.urlopen = fake
            try:
                async def go():
                    return await asyncio.gather(*(r2.pool_inflight() for _ in range(50)))
                out = asyncio.run(go())
            finally:
                R.urllib.request.urlopen = orig
            self.assertEqual(out, [None] * 50)
            self.assertEqual(len(calls), 2)            # one read per server, not 50 x 2

    def test_recently_admitted_specs_count_as_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.round(tmp, 'oneshot', 1998)
            self.assertTrue(self.admitted(r))       # 1998 < 2 servers x 1000
            self.assertFalse(self.admitted(r))      # 1998 + 1 just admitted x 2 jobs is not < 2000

    def test_live_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.round(tmp, 'oneshot', 2500)
            self.assertFalse(self.admitted(r))
            (Path(tmp) / 'admission.json').write_text(json.dumps({'target_inflight': 1500}))
            r._adm_at = 0                            # the file is re-read every 60 s
            self.assertTrue(self.admitted(r))
            self.assertIn("admission override: {'target_inflight': 1500}", r.logged)
            (Path(tmp) / 'admission.json').write_text('not json')
            r._adm_at = 0
            self.assertEqual(r.admission('target_inflight'), 1000)   # a broken file falls back to the command line
            (Path(tmp) / 'admission.json').write_text(json.dumps({'target_inflight': 0}))
            r._adm_at = 0
            self.assertTrue(self.admitted(r))        # 0: no gate


class FakeEndpoint:
    def __init__(self):
        self.calls = []

    async def chat_tools(self, messages, tools, max_tokens=None, affinity=None, extra=None):
        self.calls.append(extra)
        return 'reply', '', {'prompt': 1, 'completion': 1, 'finish': 'stop'}, []


class EffortScope(unittest.TestCase):
    def test_effort_reaches_asset_specs_only(self):
        rnd = R.Round.__new__(R.Round)
        rnd.a = Namespace(max_tokens=None, author_tokens=128000)
        rnd.gen, rnd.gen_slots = FakeEndpoint(), asyncio.Semaphore(1)
        rnd.workflow, rnd.lean = 'full', False
        rnd.asset_author_extra = R.with_effort({'chat_template_kwargs': {'x': 1}}, 'high')

        async def go():
            await rnd.author_reply([{'role': 'user', 'content': 'u'}], None, {'spec_id': 'a', 'asset': {'dir': '/x'}})
            await rnd.author_reply([{'role': 'user', 'content': 'u'}], None, {'spec_id': 'b'})
        asyncio.run(go())
        self.assertEqual(rnd.gen.calls[0], {'chat_template_kwargs': {'x': 1, 'reasoning_effort': 'high'},
                                            'reasoning_effort': 'high'})
        self.assertIsNone(rnd.gen.calls[1])     # None: the endpoint's own extra_body (the server's max)

    def test_with_effort_numeric(self):
        self.assertEqual(R.with_effort({}, '75'), {'chat_template_kwargs': {'reasoning_effort': 75}})


class StreamedBinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rnd = R.Round.__new__(R.Round)
        self.rnd.dir = Path(self.tmp.name)
        self.rnd.a = Namespace(bind_window=2)
        self.rnd.log = lambda msg: None
        self.specs = [{'spec_id': f's{i}', **({'bind': 'pending'} if i in (1, 3) else {})} for i in range(7)]
        self.rnd.bind_init(self.specs)
        (self.rnd.dir / 'bind').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, sid, status, asset=None):
        res = {'spec_id': sid, 'status': status, **({'asset': asset} if asset else {})}
        (self.rnd.dir / 'bind' / f'{sid}.json').write_text(json.dumps(res))

    def test_order_window_and_results(self):
        done = {}

        async def go():
            watcher = asyncio.create_task(self.rnd.watch_bind(every=0.02))

            async def one(sp):
                done[sp['spec_id']] = await self.rnd.bind_ready(sp)
            tasks = [asyncio.create_task(one(sp)) for sp in self.specs]
            await asyncio.sleep(0.2)
            # window 2 (at most 2 unresolved pending specs before a spec): every unpending spec may start; s1, s3 wait
            self.assertEqual(sorted(done), ['s0', 's2', 's4', 's5', 's6'])
            self.publish('s3', 'no_fit')     # a later binding resolves first: no head-of-line blocking
            await asyncio.sleep(0.2)
            self.assertEqual(sorted(done), ['s0', 's2', 's3', 's4', 's5', 's6'])
            self.publish('s1', 'bound', {'dir': '/a', 'asset_id': 'x'})
            await asyncio.gather(*tasks)
            watcher.cancel()
        asyncio.run(go())
        self.assertEqual(done['s1']['asset'], {'dir': '/a', 'asset_id': 'x'})
        self.assertEqual(done['s1']['bind'], 'bound')
        self.assertNotIn('asset', done['s3'])
        self.assertEqual(done['s3']['bind'], 'no_fit')
        self.assertNotIn('bind', done['s0'])

    def test_window_counts_unresolved_specs(self):
        rnd = R.Round.__new__(R.Round)
        rnd.dir, rnd.a, rnd.log = self.rnd.dir, Namespace(bind_window=1), (lambda m: None)
        specs = [{'spec_id': 'p0', 'bind': 'pending'}, {'spec_id': 'p1', 'bind': 'pending'}, {'spec_id': 'x'}]
        rnd.bind_init(specs)
        done = {}

        async def go():
            t = asyncio.create_task(rnd.bind_ready(specs[2]))
            await asyncio.sleep(0.05)
            self.assertFalse(t.done())        # 2 unresolved pending specs before it > window 1
            w = asyncio.create_task(rnd.watch_bind(every=0.02))
            (rnd.dir / 'bind' / 'p1.json').write_text(json.dumps({'spec_id': 'p1', 'status': 'no_fit'}))
            done['x'] = await asyncio.wait_for(t, 2)
            w.cancel()
        asyncio.run(go())
        self.assertEqual(done['x'], {'spec_id': 'x'})

    def test_window_live_override(self):
        rnd = R.Round.__new__(R.Round)
        rnd.dir, rnd.a, rnd.log = self.rnd.dir, Namespace(bind_window=1), (lambda m: None)
        specs = [{'spec_id': 'p0', 'bind': 'pending'}, {'spec_id': 'p1', 'bind': 'pending'}, {'spec_id': 'x'}]
        rnd.bind_init(specs)

        async def go():
            w = asyncio.create_task(rnd.watch_bind(every=0.02))
            t = asyncio.create_task(rnd.bind_ready(specs[2]))
            await asyncio.sleep(0.1)
            self.assertFalse(t.done())        # 2 unresolved before it > window 1
            (rnd.dir / 'admission.json').write_text(json.dumps({'bind_window': 2}))
            rnd._adm_at = 0                   # (re-read now instead of within 60 s)
            out = await asyncio.wait_for(t, 2)
            w.cancel()
            return out
        self.assertEqual(asyncio.run(go()), {'spec_id': 'x'})

    def test_round_without_binding_never_waits(self):
        rnd = R.Round.__new__(R.Round)
        rnd.dir, rnd.a = self.rnd.dir, Namespace(bind_window=500)
        rnd.bind_init([{'spec_id': 'a'}, {'spec_id': 'b'}])
        out = asyncio.run(rnd.bind_ready({'spec_id': 'b'}))
        self.assertEqual(out, {'spec_id': 'b'})


class Planner(unittest.TestCase):
    cfg = tomllib.loads((REPO / 'task_synthesis' / 'production' / 'm1.toml').read_text())

    def test_eligible(self):
        ex = {'crypto-material'}
        anchor = lambda c: {'bucket': 'best-generated', 'assets': [{'class': c, 'role': 'anchor'}]}
        self.assertTrue(production.eligible(anchor('tabular-data'), ex))
        self.assertTrue(production.eligible(anchor('network-capture'), ex))
        self.assertFalse(production.eligible(anchor('crypto-material'), ex))
        self.assertFalse(production.eligible({'bucket': 'none', 'assets': []}, ex))
        self.assertFalse(production.eligible({'bucket': 'best-generated',
                                              'assets': [{'class': 'config', 'role': 'companion'}]}, ex))

    def test_sizing(self):
        s = production.sizing(self.cfg, 1_000_000_000, 0.7)
        sp = self.cfg['specs']
        p = self.cfg['recipe']['asset_share'] * 0.7 * sp['bind_rate']
        self.assertAlmostEqual(s['p_asset_tasks'], round(p, 4))
        per_trace = p * sp['tokens_per_task_asset'] + (1 - p) * sp['tokens_per_task_generated']
        self.assertEqual(s['tokens_per_trace_prior'], round(per_trace))
        m = self.cfg['recipe']['traces_per_task']
        self.assertEqual(s['tasks_needed'], math.ceil(1_000_000_000 / (m * per_trace)))
        self.assertEqual(s['specs_sized'], math.ceil(1_000_000_000 / (m * per_trace) * sp['oversupply']))
        self.assertNotIn('specs', s)   # run.json's 'specs' is the drawn count; sizing must not shadow it

    def test_driver_round_args_parse_to_the_recipe(self):
        d = production.Driver.__new__(production.Driver)
        d.cfg, d.pool = self.cfg, 'a+b'
        d.info = {'round': 'x', 'specs': 100, 'seed': 1, 'servers': ['s1', 's2']}
        argv = d.round_args()
        self.assertEqual(argv[1:4], ['-m', 'task_synthesis.pipeline.round', 'submit'])
        a = R.parse_args(argv[3:])
        r = self.cfg['round']
        self.assertEqual((a.real_assets, a.author, a.vocab, a.minutes_mix), (self.cfg['recipe']['real_assets'], 'oneshot',
                                                                             'v2h0', 'easy-heavy'))
        self.assertEqual(a.concurrency, math.ceil(2 * r['knee_inflight_per_server'] * 2 / r['jobs']))
        self.assertEqual((a.target_inflight, a.authors), (r['oneshot_target_inflight'], 0))
        self.assertEqual(a.bind_window, r['bind_window'])

    def test_driver_bind_args(self):
        d = production.Driver.__new__(production.Driver)
        b = self.cfg['bind']
        for cfg_bind, servers, slots, http in (
                (b, ['s1', 's2'], b['concurrency'], b['http_concurrency']),
                ({**b, 'concurrency_per_server': 150, 'http_concurrency_per_server': 32}, ['s1', 's2', 's3', 's4'], 600, 128)):
            d.cfg = {**self.cfg, 'bind': cfg_bind}
            d.pool, d.info = '+'.join(servers + ['s1-spare']), {'round': 'x', 'bind': 'x', 'servers': servers}
            argv = d.bind_args()
            flags = dict(zip(argv[4::2], argv[5::2]))
            self.assertEqual(argv[1:4], ['-m', 'task_synthesis.assets.bind', 'submit'])
            self.assertEqual(flags['--llm'], d.pool)
            self.assertEqual((int(flags['--concurrency']), int(flags['--http-concurrency'])), (slots, http))
            self.assertEqual((flags['--warm-from'], int(flags['--grow-now'])), ('auto', 3))
        self.assertEqual((b['concurrency'], b['warm_from'], b['grow_now']), (600, 'auto', 3))

    def test_config_values(self):
        rec = self.cfg['recipe']
        self.assertEqual((rec['vocab'], rec['minutes_mix'], rec['traces_per_task'], rec['trace_judge'],
                          rec['real_assets'], rec['asset_share']), ('v2h0', 'easy-heavy', 2, False, 'oneshot', 0.64))
        # 0.64 (user, 2026-11-04): ~39% of tokens on real-asset tasks at v2h's measured eligible share
        self.assertAlmostEqual(production.sizing(self.cfg, 10**9, 0.6863)['asset_token_share'], 0.39, delta=0.01)
        self.assertIn(rec['real_assets'], R.REAL_ASSET_PRESETS)
        self.assertEqual((self.cfg['generator']['servers'], self.cfg['campaign']['engines']), (2, 8))


class Driver(unittest.TestCase):
    def driver(self, tmp):
        d = production.Driver.__new__(production.Driver)
        d.cfg = Planner.cfg
        d.code = REPO
        d.stream = Path(tmp) / 'stream'
        d.rd = Path(tmp) / 'round'
        d.records = {}
        d.info = {'campaign_id': '20991231-test', 'tokens_budget': 1_000_000, 'tokens_per_trace_prior': 50000,
                  'milestones': [], 'round': 'r', 'bind': 'r'}
        return d

    def test_first_bind_job_opens_every_slot_and_a_resubmitted_one_ramps(self):
        """#172 entries 3 and 11: at launch the servers are idle and binding is the bottleneck, so the first bind job
        starts with every slot open; a resubmitted one ramps its slots into a loaded pool (bind --ramp-minutes)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            d.state, d.pool = {}, 's1+s2'
            d.info.update({'bind_selected': 5, 'servers': ['s1', 's2']})
            d.bind_done = lambda: False
            d.log = d.save = lambda *a, **k: None
            calls = []
            d.sh = lambda argv, what: calls.append(argv) or '1 job(s) submitted to cluster 9.\n'
            orig = production.BINDS
            try:
                production.BINDS = Path(tmp) / 'binds'          # no job.json: the bind job is not running
                d.bind_up()
                d.bind_up()
            finally:
                production.BINDS = orig
        self.assertEqual(calls[0][-2:], ['--ramp-minutes', '0'])
        self.assertNotIn('--ramp-minutes', calls[1])          # bind's default ramp
        self.assertEqual(d.state['bind_submits'], 2)

    def test_generator_servers_use_the_task_synthesis_profiles(self):
        """#480: the one serve_teacher defaults to the eval servers' profiles; production names its pipeline (vLLM main
        and DSpark-5 since 11-19, no CPU KV tier, 20c/320G for DeepSeek; filled shapes, no memory net)."""
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            d.cfg = {**Planner.cfg, 'generator': {'teacher': 'deepseek-v41-flash-1017', 'gpus': 4, 'bid': 63, 'hours': 48}}
            d.state, d.log, d.save = {}, (lambda *a, **k: None), (lambda *a, **k: None)
            calls = []
            d.sh = lambda argv, what: calls.append(argv) or '1 job(s) submitted to cluster 9.\n'
            orig = production.SERVING
            try:
                production.SERVING = Path(tmp) / 'serving'
                d.submit_server('gen-ds')
            finally:
                production.SERVING = orig
        argv = calls[0]
        self.assertEqual(argv[argv.index('--pipeline') + 1], 'task-synthesis')
        self.assertEqual(argv[argv.index('--teacher') + 1], 'deepseek-v41-flash-1017')

    def test_campaign_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = self.driver(tmp).campaign_toml(['tasksynth-a', 'tasksynth-b', 'tasksynth-c'])
        c = tomllib.loads(text)
        self.assertEqual(c['id'], '20991231-test')
        self.assertEqual((c['train']['trajectories_per_task'], c['train']['tokens_budget'], c['train']['budget_stop'],
                          c['train']['tokens_per_trajectory_prior']), (2, 1_000_000, 'saved', 50000))
        self.assertEqual((c['serving']['jobs'], c['serving']['bid'], c['workers']['count'], c['workers']['max_count']),
                         (8, 61, 32, 104))
        self.assertEqual([(x['task'], x.get('cap')) for x in c['smoke']['attempts']], [('tasksynth-a', None)])
        self.assertFalse(c['eval']['enabled'])
        self.assertEqual(c['train']['overflow_per_task'], 0)   # exactly M traces per task
        self.assertIs(c['train']['count_interrupted_usable'], False)   # interrupted traces are dropped in SFT
        self.assertEqual(c['serving']['autoscale']['start'], 2)   # #234 R1: Super starts with 2 engines of 8
        d = self.driver(tmp)
        d.cfg = {**Planner.cfg, 'campaign': {k: v for k, v in Planner.cfg['campaign'].items() if k != 'autoscale'}}
        with tempfile.TemporaryDirectory() as tmp2:
            d.stream = Path(tmp2)
            old = tomllib.loads(d.campaign_toml(['tasksynth-a']))
        self.assertEqual(old['serving']['autoscale'], {'enabled': False})   # runs planned before #234

    def test_supply(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            (d.stream / 'producer').mkdir(parents=True)
            (d.stream / 'manifest.json').write_text(json.dumps({f't{i}': {} for i in range(10)}))
            (d.stream / 'producer' / 'rejected.json').write_text(json.dumps({'tx': {}}))
            for i in range(15):   # 13 accepted, 2 not
                w = d.rd / 'work' / f's{i}'
                w.mkdir(parents=True)
                (w / 'record.json').write_text(json.dumps({'status': 'valid', 'judge': {'pass': i < 13}}))
            total, parts = d.supply({'per_task': {'tasks': 6, 'available_tasks': 10, 'projected_tokens': 300_000,
                                                  'tokens_per_trajectory_estimate': 40_000}})
        # opened 6 projected 300k + (4 streamed unopened + (13 - 10 - 1) unpublished) x M x 40k
        self.assertEqual(parts['unpublished'], 2)
        self.assertEqual(total, 300_000 + (4 + 2) * Planner.cfg['recipe']['traces_per_task'] * 40_000)


TestHealthy = production.healthy
REAL_TIME = production.time.time


class QuotaAndSwap(unittest.TestCase):
    """#124: /data file-quota warning and the rolling generator-server swap (condor and health mocked)."""

    def driver(self, tmp, spares=True):
        d = production.Driver.__new__(production.Driver)
        servers = ['x-ds1', 'x-ds2']
        d.info = {'servers': servers, 'own_servers': True,
                  'server_spares': {s: s + 'b' for s in servers} if spares else {}}
        d.spares = d.info['server_spares']
        d.cfg = {'generator': {'teacher': 't', 'gpus': 4, 'bid': 61, 'hours': 24}}
        d.rd, d.run = Path(tmp) / 'rd', Path(tmp) / 'run'
        d.rd.mkdir(); d.run.mkdir()
        d.state_path, d.state = d.run / 'state.json', {}
        return d

    def test_quota_only_warns(self):
        orig = production.fast_files
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d = self.driver(tmp)
                for free in (5_000_000, 1_500_000, 200_000):
                    production.fast_files = lambda path='/data', f=free: (41_943_040 - f, 41_943_040)
                    d.quota()
                    self.assertFalse((d.rd / 'STOP').exists())
                log = (d.run / 'driver.log').read_text()
                self.assertEqual(log.count('ALERT /data file quota low'), 1)   # at most hourly
                self.assertNotIn('STOP', log)
        finally:
            production.fast_files = orig

    def test_launch_warns_over_the_file_budget_and_launches(self):
        """#245: launch only warns when the run's expected files exceed /data's free files (user, 2026-11-05); a
        relaunch counts only the tasks not started yet."""
        import contextlib, io
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            info = {'round': 'p1', 'campaign_id': 'c-p1', 'tasks_needed': 1000,
                    'config': {'quota': {'files_per_task': 40, 'campaign_loose_files': 200_000}}}
            (tmp / 'runs' / 'p1').mkdir(parents=True)
            (tmp / 'runs' / 'p1' / 'run.json').write_text(json.dumps(info))
            submitted = []
            with mock.patch.multiple(production, RUNS=tmp / 'runs', ROUNDS=tmp / 'rounds', SYNTH=tmp / 'synth',
                                     frozen_checkout=lambda: tmp / 'code',
                                     fast_files=lambda path='/data': (41_943_040 - 220_000, 41_943_040)), \
                    mock.patch.object(production.subprocess, 'run',
                                      lambda *a, **k: submitted.append(a) or Namespace(stdout='cluster 7.')):
                self.assertEqual(production.file_budget(info, production.fast_files())[:2], (240_000, 220_000))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    production.launch(Namespace(name='p1', bid=35))
                self.assertIn('WARNING the run needs ~240,000 files', out.getvalue())
                self.assertEqual(len(submitted), 1)
                for i in range(600):   # a relaunch: 600 specs started
                    (tmp / 'rounds' / 'p1' / 'work' / f's-{i}').mkdir(parents=True)
                (tmp / 'runs' / 'p1' / 'driver.json').unlink()
                self.assertEqual(production.file_budget(info, production.fast_files())[0], 216_000)
                production.launch(Namespace(name='p1', bid=35))
                self.assertEqual(len(submitted), 2)

    def test_swap_starts_the_spare_first_and_releases_the_old_once_healthy(self):
        clock = [1e9]
        jobs, health, events = {}, set(), []
        patches = {'read_json': production.read_json, 'alive': production.alive, 'healthy': production.healthy}
        orig_time = production.time.time
        try:
            production.time.time = lambda: clock[0]
            production.read_json = lambda path, default=None: (
                dict(jobs[Path(path).parent.name]) if str(path).endswith('job.json') and Path(path).parent.name in jobs
                else (default if str(path).endswith('job.json') else patches['read_json'](path, default)))
            production.alive = lambda j: next(v['alive'] for v in jobs.values() if v['job_id'] == j)
            production.healthy = lambda n: n in health
            with tempfile.TemporaryDirectory() as tmp:
                d = self.driver(tmp)

                def submit(name):
                    jobs[name] = {'job_id': f'{len(events)}.0', 'time': clock[0], 'alive': True}
                    events.append(('submit', name))
                    return 'ok'

                def remove(name, why):
                    jobs[name]['alive'] = False
                    health.discard(name)
                    events.append(('remove', name))
                    return True
                d.submit_server, d.remove_server = submit, remove
                d.servers_up()
                self.assertEqual(events, [('submit', 'x-ds1'), ('submit', 'x-ds2')])
                health.update({'x-ds1', 'x-ds2'})
                clock[0] += 23.2 * 3600
                d.servers_up()
                self.assertEqual(events[2:], [('submit', 'x-ds1b'), ('submit', 'x-ds2b')])
                d.servers_up()                       # spares not healthy yet: the old servers keep serving
                self.assertEqual(len(events), 4)
                health.add('x-ds1b')
                d.servers_up()
                self.assertEqual(events[4:], [('remove', 'x-ds1')])
                self.assertEqual(d.state['server_active'], {'x-ds1': 'x-ds1b'})
                jobs['x-ds2']['alive'] = False       # an old server dying mid-swap is not restarted
                health.discard('x-ds2')
                d.servers_up()
                self.assertEqual(len(events), 5)
        finally:
            production.time.time = orig_time
            for k, v in patches.items():
                setattr(production, k, v)

    def test_decontam_hold_keeps_the_servers_until_released_dead_or_late(self):
        clock, live = [1e9], {'7.0': True}
        orig = production.time.time, production.alive
        try:
            production.time.time = lambda: clock[0]
            production.alive = lambda j: live.get(j, False)
            with tempfile.TemporaryDirectory() as tmp:
                d = self.driver(tmp)
                self.assertFalse(d.decontam_hold())                 # no hold file
                (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0', 'work': 'W'}))
                d.state['round_finished'] = clock[0]
                self.assertTrue(d.decontam_hold())
                clock[0] += 2 * 3600
                self.assertTrue(d.decontam_hold())
                (d.run / 'decontam.HOLD').unlink()                  # last pass done: pool.py removes it
                self.assertFalse(d.decontam_hold())
                (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0'}))
                live['7.0'] = False                                 # the decontam job died
                self.assertFalse(d.decontam_hold())
                self.assertIn('is gone', (d.run / 'driver.log').read_text())
                d.state.pop('hold_dropped'); live['7.0'] = True
                clock[0] += 1.5 * 3600                              # 3.5 h past the round's end
                self.assertFalse(d.decontam_hold())
                self.assertFalse(d.decontam_hold())                 # stays released
                self.assertEqual((d.run / 'driver.log').read_text().count('ALERT decontam hold'), 2)
        finally:
            production.time.time, production.alive = orig

    def test_release_retries_while_condor_fails(self):
        calls, mode = [], ['fail']
        orig_run, orig_read = production.subprocess.run, production.read_json

        def fake_run(argv, **kw):
            calls.append(argv[0])
            if argv[0] == 'condor_q':
                if mode[0] == 'fail':
                    return Namespace(returncode=1, stdout='', stderr='Error: must define SCHEDD_NAME')
                return Namespace(returncode=0, stdout=f'run --name {argv[1].split(".")[0]}\n', stderr='')
            return Namespace(returncode=0, stdout='', stderr='')
        try:
            production.subprocess.run = fake_run
            production.read_json = lambda path, default=None: (
                {'job_id': Path(path).parent.name + '.0'} if str(path).endswith('job.json') else orig_read(path, default))
            with tempfile.TemporaryDirectory() as tmp:
                d = self.driver(tmp, spares=False)
                d.release_servers('test')
                self.assertNotIn('servers_released', d.state)       # condor_q failed: not marked, retried later
                self.assertNotIn('condor_rm', calls)
                mode[0] = 'ok'
                d.release_servers('test')
                self.assertIn('servers_released', d.state)
                self.assertEqual(calls.count('condor_rm'), 2)
        finally:
            production.subprocess.run, production.read_json = orig_run, orig_read

    def hold_driver(self, tmp, live):
        d = self.driver(tmp)
        (d.rd / 'STOP').write_text('budget covered\n')
        work = Path(tmp) / 'dc'
        (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0', 'work': str(work)}))
        serving = Path(tmp) / 'serving'
        for i, n in enumerate(d.server_names()):
            (serving / n).mkdir(parents=True)
            (serving / n / 'endpoints.json').write_text(json.dumps({'urls': [f'http://{n}:8000/v1'], 'healthy_at': 1,
                                                                     'updated_at': production.time.time()}))
            (serving / n / 'job.json').write_text(json.dumps({'job_id': f'{i}.0'}))
            live[f'{i}.0'] = n in ('x-ds1', 'x-ds2')
        live['7.0'] = True
        return d, work, serving

    def test_stop_with_a_hold_keeps_the_busiest_server_and_releases_the_rest(self):
        """#234 R2: at STOP with a decontam hold, one server stays (the judge's busiest), the others go at once; the
        kept one goes within a minute of the hold's removal."""
        live, removed = {}, []
        orig = production.alive, production.SERVING, production.time.sleep
        try:
            production.alive = lambda j: live.get(j, False)
            with tempfile.TemporaryDirectory() as tmp:
                d, work, serving = self.hold_driver(tmp, live)
                production.SERVING = serving
                (work / 'judge-load').mkdir(parents=True)
                # the judge pins more groups on x-ds2 than on x-ds1
                (work / 'judge-load' / 'fp-p.json').write_text(json.dumps({'updated_at': production.time.time(),
                    'targets': [['http://x-ds1:8000/v1', 0, 120], ['http://x-ds2:8000/v1', 0, 200],
                                ['http://x-ds2:8000/v1', 1, 190]]}))
                (work / 'judge-load' / 'old.json').write_text(json.dumps({'updated_at': 0,   # stale: ignored
                    'targets': [['http://x-ds1:8000/v1', 0, 10_000]]}))

                def remove(name, why):
                    removed.append(name)
                    return True
                d.remove_server = remove
                d.decontam_trim()
                self.assertEqual(d.state['decontam_kept']['server'], 'x-ds2')
                self.assertEqual(sorted(removed), ['x-ds1', 'x-ds1b', 'x-ds2b'])
                d.decontam_trim()                                   # idempotent
                self.assertEqual(len(removed), 3)
                self.assertIn('keeping x-ds2 for decontam', (d.run / 'driver.log').read_text())
                # the hold goes during the driver's 300 s sleep: it wakes within one 30 s poll
                slept, clock = [], [production.time.time()]

                def fake_sleep(sec):
                    slept.append(sec)
                    clock[0] += sec
                    if len(slept) == 2:
                        (d.run / 'decontam.HOLD').unlink(missing_ok=True)
                production.time.sleep = fake_sleep
                production.time.time = lambda: clock[0]
                d.nap(300)
                self.assertEqual(slept, [30, 30])
                self.assertTrue(d.hold_just_released())
                slept.clear()
                d.nap(300)                                          # wakes early once only (no busy loop)
                self.assertEqual(sum(slept), 300)
        finally:
            production.alive, production.SERVING, production.time.sleep = orig
            production.time.time = REAL_TIME

    def test_hold_file_names_the_server_and_no_hold_keeps_todays_behaviour(self):
        live, removed = {}, []
        orig = production.alive, production.SERVING
        try:
            production.alive = lambda j: live.get(j, False)
            with tempfile.TemporaryDirectory() as tmp:
                d, work, serving = self.hold_driver(tmp, live)
                production.SERVING = serving
                d.remove_server = lambda name, why: removed.append(name) or True
                (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0', 'work': str(work), 'keep': 'x-ds1'}))
                d.decontam_trim()
                self.assertEqual(d.state['decontam_kept']['server'], 'x-ds1')
                self.assertNotIn('x-ds1', removed)
            removed.clear()
            with tempfile.TemporaryDirectory() as tmp:
                d, work, serving = self.hold_driver(tmp, live)
                production.SERVING = serving
                d.remove_server = lambda name, why: removed.append(name) or True
                (d.run / 'decontam.HOLD').unlink()                  # no decontam: servers stay until round finish
                d.decontam_trim()
                self.assertEqual(removed, [])
                self.assertNotIn('decontam_kept', d.state)
                (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0', 'work': str(work)}))
                (d.rd / 'STOP').unlink()                            # a hold before STOP: generation still needs both
                d.decontam_trim()
                self.assertEqual(removed, [])
                (d.rd / 'STOP').write_text('x')                     # no judge load: the first healthy server
                production.healthy = lambda n: n == 'x-ds2'
                d.decontam_trim()
                self.assertEqual(d.state['decontam_kept']['server'], 'x-ds2')
        finally:
            production.alive, production.SERVING = orig
            production.healthy = TestHealthy

    def test_a_dead_kept_server_is_resubmitted_twice_at_most(self):
        live, submitted = {}, []
        orig = production.alive, production.SERVING
        try:
            production.alive = lambda j: live.get(j, False)
            with tempfile.TemporaryDirectory() as tmp:
                d, work, serving = self.hold_driver(tmp, live)
                production.SERVING = serving
                d.remove_server = lambda name, why: True
                d.submit_server = lambda name: submitted.append(name) or 'ok'
                (d.run / 'decontam.HOLD').write_text(json.dumps({'job': '7.0', 'work': str(work), 'keep': 'x-ds1'}))
                live['0.0'] = False
                for _ in range(4):
                    d.decontam_trim()
                self.assertEqual(submitted, ['x-ds1', 'x-ds1'])
        finally:
            production.alive, production.SERVING = orig

    def test_runs_planned_without_spares_keep_their_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp, spares=False)
            self.assertEqual(d.server_names(), ['x-ds1', 'x-ds2'])


if __name__ == '__main__':
    unittest.main()


class CheapSplit(unittest.TestCase):
    """#242: round jobs go out at --cheap-bid as far as cheap nodes have room, the rest at --bid."""

    def test_split(self):
        from argparse import Namespace
        from unittest import mock
        from task_synthesis.pipeline import round as round_mod
        a = Namespace(cheap_bid=25, bid=56, cpus=24, memory_gib=384)
        with mock.patch('synth.condor.cheap_capacity', lambda *x, **k: 5):
            self.assertEqual(round_mod.cheap_split(12, a), 5)
        with mock.patch('synth.condor.cheap_capacity', lambda *x, **k: 40):
            self.assertEqual(round_mod.cheap_split(12, a), 12)
        with mock.patch('synth.condor.cheap_capacity', mock.Mock(side_effect=RuntimeError('schedd down'))):
            self.assertEqual(round_mod.cheap_split(12, a), 0, 'a failed check sends every job at --bid')
        self.assertEqual(round_mod.cheap_split(12, Namespace(cheap_bid=0, bid=56, cpus=24, memory_gib=384)), 0)


class Port378(unittest.TestCase):
    """#378 port: the round and bind wait for a healthy server; round shards are auto-released on memory holds;
    tasks.tar goes to /data with O_DIRECT; the slot arithmetic is the shared layer's."""

    def driver(self, tmp):
        d = Driver.driver(Driver(), tmp)
        d.state, d.logs = {}, []
        d.log = d.logs.append
        d.save = lambda: None
        d.info['servers'] = ['s1', 's2']
        d.spares = {}
        return d

    def test_round_and_bind_wait_for_a_healthy_server(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            with mock.patch.object(production, 'healthy', lambda s: False):
                self.assertFalse(d.servers_ready())
                self.assertFalse(d.servers_ready())
            self.assertEqual(d.logs, ['round and bind wait for a healthy generator server'])   # logged once
            with mock.patch.object(production, 'healthy', lambda s: s == 's2'):
                self.assertTrue(d.servers_ready())
            self.assertIn('server s2 healthy', d.logs[-1])
            with mock.patch.object(production, 'healthy', lambda s: False):
                self.assertTrue(d.servers_ready(), 'once started, never held back (swaps, restarts)')

    def test_a_running_round_or_the_opt_out_never_waits(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(production, 'healthy', lambda s: False):
            d = self.driver(tmp)
            d.rd.mkdir(parents=True)
            (d.rd / 'job.json').write_text('{}')               # a driver restarted on a running round
            self.assertTrue(d.servers_ready())
            d2 = self.driver(tmp + '/x')
            d2.cfg = {**Planner.cfg, 'round': {**Planner.cfg['round'], 'wait_for_server': False}}
            self.assertTrue(d2.servers_ready())

    def test_round_shards_get_auto_release_other_modules_do_not(self):
        from tbshared.condor_cost import AUTO_RELEASE
        self.assertEqual(R.auto_release('round'), AUTO_RELEASE)
        self.assertEqual(R.auto_release('roundpack'), '')
        self.assertIn('HoldReasonCode == 34', AUTO_RELEASE)

    def test_tasks_tar_is_copied_with_o_direct(self):
        text = (REPO / 'task_synthesis' / 'pipeline' / 'campaign.py').read_text()
        self.assertEqual(text.count("copy_direct(f, out / f'{f.name}.partial')"), 2)
        self.assertNotIn("'dd', f'if={f}'", text)

    def test_fits_is_the_shared_layers(self):
        import synth.condor
        import tbshared.condor_cost
        self.assertIs(synth.condor.fits, tbshared.condor_cost.fits)
        self.assertEqual(synth.condor.fits(48, 400000, 2, 16, 128 * 1024), 2)


class Port378Review(unittest.TestCase):
    """Review follow-ups: spares count as ready, a long wait ALERTs, leftover shards are removed, cheap shards carry
    the shared ladder, and engine_cache keys ignore every --*-port."""

    def driver(self, tmp):
        return Port378.driver(Port378(), tmp)

    def test_a_healthy_spare_after_a_swap_counts(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            d.spares = {'s1': 's1-b'}
            with mock.patch.object(production, 'healthy', lambda s: s == 's1-b'):
                self.assertTrue(d.servers_ready())

    def test_long_wait_alerts_once(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(production, 'healthy', lambda s: False):
            d = self.driver(tmp)
            d.spares = {}
            d.state['dependents_waiting'] = production.time.time() - 4 * 3600
            self.assertFalse(d.servers_ready())
            self.assertFalse(d.servers_ready())
            self.assertEqual(sum('ALERT no generator server healthy' in x for x in d.logs), 1)

    def test_leftover_idle_or_held_shards_removed(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            d = self.driver(tmp)
            d.rd.mkdir(parents=True)
            (d.rd / 'job.json').write_text(json.dumps({'job_id': '77.0'}))
            rm = []
            ok = mock.Mock(returncode=0)
            with mock.patch.object(production, 'job_states', lambda c: {0: 2, 1: 1, 2: 5}), \
                    mock.patch.object(production.subprocess, 'run', lambda argv, **k: rm.append(argv) or ok):
                d.remove_leftover_shards()
            self.assertEqual(rm, [['condor_rm', '77.1'], ['condor_rm', '77.2']])   # the running one exits by itself
            self.assertIn('removed 2 idle or held round jobs', d.logs[-1])
            with mock.patch.object(production, 'job_states', lambda c: {0: 2}), \
                    mock.patch.object(production.subprocess, 'run', lambda argv, **k: rm.append(argv) or ok):
                d.remove_leftover_shards()   # later passes: only what is idle or held then
            self.assertEqual(len(rm), 2)

    def test_cheap_shards_carry_the_ladder(self):
        a = Namespace(cheap_bid=25, bid=56)
        text = R.auto_release('round', a)
        self.assertIn('+AaBidLadder = "25,56"', text)
        self.assertIn('+AaBidLadderSteps = "300"', text)
        self.assertNotIn('AaBidLadder', R.auto_release('round', Namespace(cheap_bid=0, bid=56)))
        self.assertEqual(R.auto_release('bind', a), '')


class HeldServers(unittest.TestCase):
    """#378 decision: a held generator server is removed and restarted (at most 3 times), never left blocking."""

    def test_held_server_removed_then_capped(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            d = Port378.driver(Port378(), tmp)
            calls = []
            with mock.patch.object(production, 'job_states', lambda j: {0: 5}), \
                    mock.patch('tbshared.holds.remove_held', lambda j, has_net=True: calls.append(j) or ('replace', 'cpu401', '21')):
                for _ in range(3):
                    self.assertTrue(d.held_server_removed('ds1', {'job_id': '55.0'}))
                self.assertFalse(d.held_server_removed('ds1', {'job_id': '55.0'}))   # the 4th: left held, ALERT
            self.assertEqual(calls, ['55.0'] * 3)
            self.assertTrue(any('ALERT server ds1 held again' in x for x in d.logs))

    def test_running_or_released_servers_untouched(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            d = Port378.driver(Port378(), tmp)
            with mock.patch.object(production, 'job_states', lambda j: {0: 2}):
                self.assertFalse(d.held_server_removed('ds1', {'job_id': '55.0'}))
            seen = []
            with mock.patch.object(production, 'job_states', lambda j: {0: 5}), \
                    mock.patch('tbshared.holds.remove_held', lambda j, has_net=True: seen.append(has_net) or ('wait', None, '1')):
                self.assertFalse(d.held_server_removed('ds1', {'job_id': '55.0'}))   # a user's hold: left alone
            self.assertEqual(seen, [False])   # no memory net in serve_teacher here: memory holds are replaced too

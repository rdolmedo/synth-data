"""The live loop against a fake cluster: shadow acts on nothing, `on` acts only for hooked runs, by exact id."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from tbsched import loop
from tbsched.loop import Scheduler
from tbsched.state import read

URLS = ['http://n1:1/v1', 'http://n1:2/v1']
WRITES = ('rm', 'submit', 'touch', 'remove')


def metrics(running, waiting=0, kv=0.3, gen=0, hits=0, queries=0):
    return (f'vllm:num_requests_running {running}\nvllm:num_requests_waiting {waiting}\n'
            f'vllm:kv_cache_usage_perc {kv}\nvllm:generation_tokens_total {gen}\n'
            f'vllm:prefix_cache_hits_total {hits}\nvllm:prefix_cache_queries_total {queries}\n'
            'vllm:num_preemptions_total 0\n')


class FakeIO:
    def __init__(self, jobs, running=0):
        self.jobs, self.calls, self.t, self.running = jobs, [], 0, running

    def scrape(self, url, timeout=5.0):
        self.t += 1
        return metrics(self.running, gen=1000 * self.t, hits=900_000 * self.t, queries=1_000_000 * self.t)

    def condor_jobs(self, clusters):
        self.calls.append(('q', sorted(clusters)))
        return {j: v for j, v in self.jobs.items() if j.split('.')[0] in clusters}

    def condor_rm(self, ids):
        if ids:
            self.calls.append(('rm', list(ids)))

    def submit_workers(self, run_dir, n):
        self.calls.append(('submit', n))

    def touch(self, path, text):
        self.calls.append(('touch', Path(path).name))
        Path(path).write_text(text)

    def remove(self, path):
        self.calls.append(('remove', Path(path).name))
        Path(path).unlink(missing_ok=True)

    def writes(self):
        return [c for c in self.calls if c[0] in WRITES]


class Clock:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return self.t


def iso(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def campaign(root, n_items=200, claims=(), mode='shadow', hooked=False, outcomes=(), jobs=None, endpoints=True):
    """claims: (key, worker, url or None (no pin file), claimed seconds ago)."""
    root = Path(root)
    reg = root / 'serving' / 'reg'
    reg.mkdir(parents=True)
    if endpoints:
        (reg / 'endpoints.json').write_text(json.dumps({'urls': URLS, 'engine_urls': URLS, 'job_id': '500.0'}))
    (reg / 'job.json').write_text(json.dumps({'job_id': '500.0'}))
    run = root / 'runs' / 'r1'
    q = run / 'queue'
    for d in ('claims', 'done', 'retry'):
        (q / d).mkdir(parents=True)
    if n_items is not None:
        (q / 'items.txt').write_text(''.join(f't{i} 0\n' for i in range(n_items)))
    (run / 'jobs.json').write_text(json.dumps(jobs if jobs is not None else [{'cluster': '700'}]))
    (run / 'workers').mkdir()
    for k, worker, url, ago in claims:
        (q / 'claims' / k).mkdir()
        (q / 'claims' / k / 'claim.json').write_text(json.dumps({'worker': worker, 'claimed_at': iso(time.time() - ago)}))
        (q / 'claims' / k / 'heartbeat').write_text('1')
        if url:
            (q / 'claims' / k / 'replica.json').write_text(json.dumps({'url': url}))
    for i, (worker, reason) in enumerate(outcomes):
        a = run / 'attempts' / f'x{i}'
        a.mkdir(parents=True)
        (a / 'attempt.json').write_text(json.dumps({'worker': worker}))
        (a / 'outcome.json').write_text(json.dumps({'infrastructure_failure': bool(reason),
                                                    'infrastructure_reason': reason}))
    (root / 'sched').mkdir()
    (root / 'sched' / 'config.json').write_text(json.dumps({
        'mode': mode, 'groups': {'reg': {'registry': str(reg), 'model': 'super'}},
        'runs': {'r1': {'dir': str(run), 'kind': 'agentic', 'groups': {'reg': 1.0}, 'slots': 16, 'hooked': hooked,
                        'model': 'super', 'namespace': 'ns'}}}))
    return root, run


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.user = mock.patch.object(loop, 'USER', 'user')
        self.user.start()

    def tearDown(self):
        self.user.stop()
        self.tmp.cleanup()

    def run_ticks(self, root, io, ticks=1, clock=None):
        clock = clock or Clock()
        s = Scheduler(root, io=io, now=clock)
        s.lease.owner = 'test'
        s.load()
        out = None
        for _ in range(ticks):
            for _ in range(3):
                s.scrape()
                clock.t += 15
            out = s.tick()
            clock.t += 15
        return s, out

    def engine(self, root, url):
        return read(root / 'sched' / 'engines' / (url.replace(':', '_').replace('/', '_') + '.json'))

    def test_shadow_writes_credit_and_events_and_touches_nothing(self):
        root, run = campaign(self.tmp.name, claims=[('t0__a0__r0', '700.0-n9-1', URLS[0], 600)])
        io = FakeIO({'700.0': ('running', 'user'), '701.0': ('running', 'user')}, running=1)
        s, d = self.run_ticks(root, io)
        st = read(root / 'sched' / 'runs' / 'r1.json')
        self.assertEqual(st['mode'], 'shadow')
        unspent0 = st['credit'][URLS[0]] - 1        # cumulative credit includes the live session's own start
        self.assertGreater(st['credit'][URLS[1]], unspent0)   # the engine with the live session gets less new room
        self.assertEqual(io.writes(), [])
        self.assertFalse((run / 'STOP_WORKERS').exists())

    def test_shadow_records_stop_and_release_but_does_none(self):
        root, run = campaign(self.tmp.name, n_items=0, jobs=[{'cluster': '700'}])
        io = FakeIO({'700.0': ('running', 'user')})
        self.run_ticks(root, io, ticks=3)
        events = [json.loads(x) for x in (root / 'sched' / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['kind'] == 'workers' and e['stop'] for e in events))
        self.assertTrue(any(e['kind'] == 'group' and e['action'][0] == 'release' for e in events))
        self.assertTrue(all(e.get('shadow', True) for e in events if e['kind'] in ('workers', 'group')))
        self.assertEqual(io.writes(), [])

    def test_on_mode_acts_only_for_hooked_runs(self):
        outcomes = [('701.0', 'BuildError')] * 6
        for hooked, want in ((False, []), (True, [('rm', ['701.0'])])):
            with tempfile.TemporaryDirectory() as tmp:
                root, run = campaign(tmp, mode='on', hooked=hooked, outcomes=outcomes,
                                     jobs=[{'cluster': '700'}, {'cluster': '701'}, {'cluster': '702'}])
                io = FakeIO({'700.0': ('running', 'user'), '701.0': ('running', 'user'),
                             '702.0': ('running', 'someone')})
                self.run_ticks(root, io)
                self.assertEqual([c for c in io.calls if c[0] == 'rm'], want, hooked)
                self.assertEqual('submit' in [c[0] for c in io.calls], hooked)

    def test_dead_claims_are_work_not_engine_load_and_get_a_worker(self):
        # 200 claims of a worker condor no longer lists: claimable, 0 sessions, and someone must claim/reap them
        claims = [(f't{i}__a0__r0', '799.0-n9-1', URLS[i % 2], 600) for i in range(200)]
        root, run = campaign(self.tmp.name, n_items=200, claims=claims, mode='on', hooked=True)
        io = FakeIO({})
        self.run_ticks(root, io)
        self.assertEqual(self.engine(root, URLS[0])['sessions'], 0)
        pend = read(root / 'sched' / 'runs' / 'r1.json')['pending']
        self.assertEqual((pend['dead'], pend['live']), (200, 0))
        self.assertTrue(any(c[0] == 'submit' and c[1] >= 1 for c in io.calls), io.calls)

    def test_unpinned_live_claims_count_on_their_hashed_engine(self):
        claims = [(f't{i}__a0__r0', '700.0-n9-1', None, 600) for i in range(20)]
        root, _ = campaign(self.tmp.name, claims=claims)
        self.run_ticks(root, FakeIO({'700.0': ('running', 'user')}, running=20))
        n = [self.engine(root, u)['sessions'] for u in URLS]
        self.assertEqual(sum(n), 20)
        self.assertTrue(all(n), n)

    def test_fresh_claims_are_counted_and_old_ones_are_not(self):
        claims = [('t0__a0__r0', '700.0-n9-1', URLS[0], 30), ('t1__a0__r0', '700.0-n9-1', URLS[0], 3600)]
        root, _ = campaign(self.tmp.name, claims=claims)
        s, _ = self.run_ticks(root, FakeIO({'700.0': ('running', 'user')}, running=2))
        runs = s.observe_runs()
        self.assertEqual(runs['r1']['fresh'], {URLS[0]: 1})
        self.assertEqual(runs['r1']['live'], {URLS[0]: 2})

    def test_eval_q35_jobs_json_is_understood(self):
        root, _ = campaign(self.tmp.name, claims=[('t0__a0__r0', '700.0-n9-1', URLS[0], 60)],
                           jobs={'jobs': [{'cluster': '700', 'role': 'worker'}, {'cluster': '500', 'role': 'serve'}]})
        io = FakeIO({'700.0': ('running', 'user'), '500.0': ('running', 'user')}, running=1)
        self.run_ticks(root, io)
        self.assertEqual(read(root / 'sched' / 'runs' / 'r1.json')['pending']['live'], 1)
        self.assertEqual([c for c in io.calls if c[0] == 'q'][0][1], ['700'])

    def test_no_release_when_endpoints_are_unreadable_or_the_engine_is_busy(self):
        for endpoints, running in ((False, 0), (True, 5)):
            with tempfile.TemporaryDirectory() as tmp:
                root, _ = campaign(tmp, n_items=0, mode='on', hooked=True, endpoints=endpoints)
                io = FakeIO({'700.0': ('running', 'user')}, running=running)
                self.run_ticks(root, io, ticks=4)
                self.assertNotIn(('rm', ['500.0']), io.calls, (endpoints, running))

    def test_release_of_an_idle_hooked_group(self):
        root, _ = campaign(self.tmp.name, n_items=0, mode='on', hooked=True)
        io = FakeIO({'700.0': ('running', 'user'), '500.0': ('running', 'user')})
        self.run_ticks(root, io, ticks=3)
        self.assertIn(('rm', ['500.0']), io.calls)
        self.assertEqual(read(root / 'sched' / 'groups' / 'reg.json')['status'], 'released')

    def test_an_unreadable_queue_is_never_stopped(self):
        root, run = campaign(self.tmp.name, n_items=None, mode='on', hooked=True)
        io = FakeIO({'700.0': ('running', 'user')})
        self.run_ticks(root, io)
        self.assertFalse((run / 'STOP_WORKERS').exists())

    def test_stop_workers_written_by_tbsched_is_cleared_when_work_returns(self):
        for touched_by_person, cleared in ((False, True), (True, False)):
            with tempfile.TemporaryDirectory() as tmp:
                root, run = campaign(tmp, n_items=0, mode='on', hooked=True)
                io = FakeIO({})
                clock = Clock()
                self.run_ticks(root, io, clock=clock)
                stop = run / 'STOP_WORKERS'
                self.assertTrue(stop.exists())                     # drained: tbsched stopped it
                if touched_by_person:
                    time.sleep(0.01)
                    stop.write_text('tbsched: nothing left to claim\n')   # same text, a person's write
                (run / 'queue' / 'items.txt').write_text('t9 0\n')      # a new round brought work
                io.calls.clear()
                self.run_ticks(root, io, clock=clock)
                self.assertEqual(not stop.exists(), cleared, touched_by_person)
                self.assertEqual('submit' in [c[0] for c in io.calls], cleared)

    def test_a_run_whose_engines_were_released_stays_stopped(self):
        root, run = campaign(self.tmp.name, n_items=0, mode='on', hooked=True)
        io = FakeIO({'500.0': ('running', 'user')})
        clock = Clock()
        self.run_ticks(root, io, ticks=3, clock=clock)
        self.assertEqual(read(root / 'sched' / 'groups' / 'reg.json')['status'], 'released')
        (run / 'queue' / 'items.txt').write_text('t9 0\n')          # work comes back, but no engine is left
        io.calls.clear()
        self.run_ticks(root, io, ticks=4, clock=clock)
        self.assertTrue((run / 'STOP_WORKERS').exists())
        self.assertEqual(io.writes(), [])

    def test_unhooked_runs_are_not_spread_in_on_mode(self):
        root = Path(self.tmp.name)
        _, _ = campaign(root / 'a', n_items=0, mode='on', hooked=True)
        cfg = json.loads((root / 'a' / 'sched' / 'config.json').read_text())
        reg2 = root / 'a' / 'serving' / 'reg2'
        reg2.mkdir(parents=True)
        (reg2 / 'endpoints.json').write_text(json.dumps({'urls': ['http://n2:1/v1'], 'job_id': '501.0'}))
        (reg2 / 'job.json').write_text(json.dumps({'job_id': '501.0'}))
        _, run_u = campaign(root / 'u', n_items=500)
        cfg['groups']['reg2'] = {'registry': str(reg2), 'model': 'super'}
        cfg['runs']['u'] = {'dir': str(run_u), 'kind': 'agentic', 'groups': {'reg2': 1.0}, 'slots': 16,
                            'hooked': False, 'model': 'super'}
        (root / 'a' / 'sched' / 'config.json').write_text(json.dumps(cfg))
        io = FakeIO({'500.0': ('running', 'user')})
        self.run_ticks(root / 'a', io, ticks=4)
        self.assertNotIn('joined', read(root / 'a' / 'sched' / 'runs' / 'u.json') or {})
        self.assertIn(('rm', ['500.0']), io.calls)                   # the idle group goes instead of idling

    def test_a_release_that_removes_nothing_keeps_the_group(self):
        root, _ = campaign(self.tmp.name, n_items=0, mode='on', hooked=True)
        (root / 'serving' / 'reg' / 'job.json').unlink()
        self.run_ticks(root, FakeIO({}), ticks=4)
        self.assertEqual(read(root / 'sched' / 'groups' / 'reg.json').get('status'), 'active')
        events = (root / 'sched' / 'events.jsonl').read_text()
        self.assertIn('release_blocked', events)

    def test_a_dead_engine_blocks_release_only_for_a_while_and_says_why(self):
        root, _ = campaign(self.tmp.name, n_items=0, mode='on', hooked=True)

        class Dead(FakeIO):
            def scrape(self, url, timeout=5.0):
                if url.startswith('http://n1:2'):
                    raise OSError('connection refused')
                return super().scrape(url, timeout)
        io = Dead({'500.0': ('running', 'user')})
        clock = Clock()
        s, _ = self.run_ticks(root, io, ticks=3, clock=clock)
        self.assertNotIn(('rm', ['500.0']), io.calls)
        self.assertIn('engine_unreadable', (root / 'sched' / 'events.jsonl').read_text())
        clock.t += 700
        s.scrape()
        s.tick()
        s.scrape()
        clock.t += 60
        s.tick()
        self.assertIn(('rm', ['500.0']), io.calls)

    def test_decisions_are_logged_once_per_change(self):
        root, _ = campaign(self.tmp.name, n_items=0)
        self.run_ticks(root, FakeIO({'700.0': ('running', 'user')}), ticks=6)
        kinds = [json.loads(x)['kind'] for x in (root / 'sched' / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(kinds.count('workers'), 1)
        self.assertLessEqual(kinds.count('group'), 3)

    def test_settled_claims_are_read_once(self):
        claims = [(f't{i}__a0__r0', '700.0-n9-1', None, 600) for i in range(5)]
        root, run = campaign(self.tmp.name, claims=claims)
        for i in range(3):
            (run / 'queue' / 'done' / f't{i}__a0__r0.json').write_text('{}')
        s, _ = self.run_ticks(root, FakeIO({'700.0': ('running', 'user')}, running=2))
        cache = s.pin_cache['r1']
        self.assertEqual({k: v for k, v in cache.items() if v == ''}, {f't{i}__a0__r0': '' for i in range(3)})
        self.assertEqual({k for k, v in cache.items() if v is None}, {'t3__a0__r0', 't4__a0__r0'})

    def test_credit_is_published_per_client_url(self):
        # per-replica gates: workers pin the gate urls, the scheduler scrapes the engines behind them
        root, _ = campaign(self.tmp.name)
        gates = ['http://g1:1/v1', 'http://g1:2/v1']
        (root / 'serving' / 'reg' / 'endpoints.json').write_text(json.dumps(
            {'urls': gates, 'engine_urls': URLS, 'job_id': '500.0'}))
        self.run_ticks(root, FakeIO({}, running=0))
        st = read(root / 'sched' / 'runs' / 'r1.json')
        self.assertEqual(set(st['credit']), set(URLS))
        self.assertEqual(st['credit_urls'], {g: st['credit'][e] for g, e in zip(gates, URLS)})

    def test_a_pooled_gate_is_one_engine(self):
        root, _ = campaign(self.tmp.name)
        (root / 'serving' / 'reg' / 'endpoints.json').write_text(json.dumps(
            {'urls': ['http://pool:1/v1'], 'engine_urls': URLS, 'job_id': '500.0'}))
        s, _ = self.run_ticks(root, FakeIO({}, running=0))
        self.assertEqual(list(s.engines), ['http://pool:1/v1'])
        self.assertEqual(set(read(root / 'sched' / 'runs' / 'r1.json')['credit_urls']), {'http://pool:1/v1'})

    def test_unhooked_runs_get_shadow_credit_in_on_mode(self):
        root, _ = campaign(self.tmp.name, mode='on', hooked=False)
        self.run_ticks(root, FakeIO({}))
        self.assertEqual(read(root / 'sched' / 'runs' / 'r1.json')['mode'], 'shadow')

    def test_engines_a_runs_gate_holds_get_no_credit(self):
        root, run = campaign(self.tmp.name, mode='on', hooked=True)
        self.run_ticks(root, FakeIO({}))
        before = read(root / 'sched' / 'runs' / 'r1.json')['credit_urls']
        self.assertGreater(before[URLS[1]], 0)
        (run / 'queue' / 'admission.json').write_text(json.dumps({'gate_held_replicas': [1], 'replicas': 2}))
        self.run_ticks(root, FakeIO({}))
        after = read(root / 'sched' / 'runs' / 'r1.json')
        self.assertEqual(after['credit_urls'], {URLS[0]: after['credit'][URLS[0]]})
        self.assertEqual(after['credit'][URLS[1]], 0)                # unspent credit there was withdrawn

    def test_registry_urls_are_normalized_like_pi_eval(self):
        root, _ = campaign(self.tmp.name)
        (root / 'serving' / 'reg' / 'endpoints.json').write_text(json.dumps(
            {'urls': [u.removesuffix('/v1') + '/' for u in URLS], 'job_id': '500.0'}))
        self.run_ticks(root, FakeIO({}))
        self.assertEqual(set(read(root / 'sched' / 'runs' / 'r1.json')['credit_urls']), set(URLS))

    def test_a_pooled_gate_has_the_pools_limits(self):
        root, _ = campaign(self.tmp.name)
        (root / 'serving' / 'reg' / 'endpoints.json').write_text(json.dumps(
            {'urls': ['http://pool:1/v1'], 'engine_urls': URLS + ['http://n1:3/v1', 'http://n1:4/v1'],
             'job_id': '500.0'}))
        self.run_ticks(root, FakeIO({}))
        self.assertEqual(read(root / 'sched' / 'engines' / 'http___pool_1_v1.json')['limit'], 4 * 96)

    def test_a_mixed_group_ignores_the_hit_rate(self):
        # engines that also serve single-turn lanes tbsched does not see (the aa_evals dispatcher's): no hit cuts
        root, _ = campaign(self.tmp.name, claims=[(f't{i}__a0__r0', '700.0-n9-1', URLS[i % 2], 600) for i in range(90)])
        cfg = json.loads((root / 'sched' / 'config.json').read_text())
        io = FakeIO({'700.0': ('running', 'user')}, running=80)
        io.scrape = lambda url, timeout=5.0, t=[0]: (t.__setitem__(0, t[0] + 1) or metrics(
            80, gen=1000 * t[0], hits=100_000 * t[0], queries=1_000_000 * t[0]))      # 10% hit, loaded
        self.run_ticks(root, io)
        self.assertEqual(self.engine(root, URLS[0])['state'], 'pressure')
        cfg['groups']['reg']['mixed'] = True
        with tempfile.TemporaryDirectory() as tmp:
            root2, _ = campaign(tmp, claims=[(f't{i}__a0__r0', '700.0-n9-1', URLS[i % 2], 600) for i in range(90)])
            (root2 / 'sched' / 'config.json').write_text(json.dumps({**cfg, 'groups': {'reg': {
                **cfg['groups']['reg'], 'registry': str(root2 / 'serving' / 'reg')}}, 'runs': {'r1': {
                **cfg['runs']['r1'], 'dir': str(root2 / 'runs' / 'r1')}}}))
            io2 = FakeIO({'700.0': ('running', 'user')}, running=80)
            io2.scrape = io.scrape
            self.run_ticks(root2, io2)
            self.assertNotEqual(self.engine(root2, URLS[0])['state'], 'pressure')

    def test_run_releases_the_lease_on_exit(self):
        root, _ = campaign(self.tmp.name)
        s = Scheduler(root, io=FakeIO({}), now=Clock())
        loop.run(s, once=True)
        self.assertFalse(list((root / 'sched').glob('lease.*')))

    def test_a_second_instance_is_refused_and_a_superseded_one_stops(self):
        root, _ = campaign(self.tmp.name)
        a, _ = self.run_ticks(root, FakeIO({}))
        b = Scheduler(root, io=FakeIO({}), now=Clock())
        with self.assertRaises(SystemExit):
            b.tick()
        later = Clock()
        later.t += 3600                                  # a stopped renewing long ago: b takes over
        b2 = Scheduler(root, io=FakeIO({}), now=later)
        b2.tick()
        with self.assertRaises(SystemExit):
            a.tick()

    def test_operator_overrides_win_and_survive_ticks(self):
        root, _ = campaign(self.tmp.name)
        ov = root / 'sched' / 'overrides' / 'engines'
        ov.mkdir(parents=True)
        (ov / 'http___n1_1_v1.json').write_text('{"limit": 7}')
        self.run_ticks(root, FakeIO({}), ticks=2)
        self.assertEqual(self.engine(root, URLS[0])['limit'], 7)
        self.assertEqual(json.loads((ov / 'http___n1_1_v1.json').read_text()), {'limit': 7})

    def test_a_condor_failure_acts_on_nothing_and_keeps_idle_timers(self):
        root, run = campaign(self.tmp.name, mode='on', hooked=True)

        class Down(FakeIO):
            def condor_jobs(self, clusters):
                raise RuntimeError('schedd down')
        st = root / 'sched' / 'runs'
        st.mkdir(parents=True)
        (st / 'r1.json').write_text(json.dumps({'name': 'r1', 'idle_since': {'700.0': 123.0}}))
        io = Down({})
        self.run_ticks(root, io)
        self.assertEqual(io.writes(), [])
        self.assertEqual(read(st / 'r1.json')['idle_since'], {'700.0': 123.0})


if __name__ == '__main__':
    unittest.main()

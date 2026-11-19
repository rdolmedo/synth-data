"""Offline tests for watching and changing a live production run (#237, #238): `production status --json` on a
synthetic run with a failing and a working fake condor_q, `production replan` dry-run / apply / refusal on a synthetic
round, and a re-plan of a round whose two jobs are running (they reload specs.jsonl in place).

  python -m unittest task_synthesis.tests.test_live
"""
import asyncio
import hashlib
import json
import os
import stat
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

from task_synthesis.pipeline import live, production as P
from task_synthesis.pipeline.claims import Claims
from task_synthesis.tests.test_round_shards import make_round

PATCHED = ('RUNS', 'ROUNDS', 'STREAMS', 'SERVING', 'BINDS', 'SYNTH')
CFG = {'recipe': {'real_assets': 'oneshot', 'vocab': 'v2h', 'minutes_mix': 'easy-heavy', 'asset_share': 0.5,
                  'traces_per_task': 2, 'bind_excluded_classes': []},
       'round': {'bind_window': 300, 'jobs': 2, 'claim_stale_minutes': 10},
       'generator': {'servers': 1}, 'bind': {}, 'quota': {'alert_free_files': 2_000_000}}


class Sandbox(unittest.TestCase):
    """A synthetic run under a temporary ROOT: production's directories are patched to it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.orig = {k: getattr(P, k) for k in PATCHED}
        for k in PATCHED:
            setattr(P, k, self.root / k.lower())
            getattr(P, k).mkdir()
        self.path = os.environ['PATH']

    def tearDown(self):
        for k, v in self.orig.items():
            setattr(P, k, v)
        os.environ['PATH'] = self.path
        self.tmp.cleanup()

    def run_dir(self, name='x', specs=()):
        info = {'name': name, 'config': json.loads(json.dumps(CFG)), 'tokens_budget': 10**9, 'milestones': [],
                'seed': 5000, 'specs': len(specs), 'bind_selected': sum(1 for s in specs if s.get('bind') == 'pending'),
                'round': name, 'stream': str(P.STREAMS / name), 'bind': name,
                'campaign_id': f'c-{name}', 'servers': [f'{name}-ds1'], 'own_servers': True,
                'server_spares': {f'{name}-ds1': f'{name}-ds1b'}, 'tokens_per_trace_prior': 40000}
        run, rd = P.RUNS / name, P.ROUNDS / name
        run.mkdir()
        rd.mkdir()
        (rd / 'work').mkdir()
        P.write_json(run / 'run.json', info)
        (rd / 'specs.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in specs))
        return run, rd

    def fake_condor_q(self, script):
        b = self.root / 'bin'
        b.mkdir(exist_ok=True)
        (b / 'condor_q').write_text('#!/bin/sh\n' + script)
        (b / 'condor_q').chmod(0o755 | stat.S_IEXEC)
        os.environ['PATH'] = f'{b}:{self.path}'


class StatusJson(Sandbox):
    def make_run(self):
        run, rd = self.run_dir('x', [{'spec_id': f's5000-{i:05d}'} for i in range(4)])
        now = time.time()
        P.write_json(run / 'driver.json', {'job_id': '100.0'})
        P.write_json(run / 'heartbeat.json', {'time': now - 30, 'pass': 7})
        P.write_json(run / 'state.json', {'producer_job': '103.0'})
        (P.SERVING / 'x-ds1').mkdir()
        P.write_json(P.SERVING / 'x-ds1' / 'job.json', {'job_id': '101.0'})
        P.write_json(P.SERVING / 'x-ds1' / 'endpoints.json', {'urls': [], 'healthy_at': now - 600, 'updated_at': now - 20})
        P.write_json(rd / 'job.json', {'job_id': '102.0'})
        (rd / 'shards').mkdir()
        base = {'host': 'g1', 'state': 'working', 'claimed': 10, 'admitted': 8, 'accepted': 5, 'cut': 0, 'errors': 0,
                'done': 6, 'open': 4, 'sessions': 2, 'inflight': 3, 'reload_specs': True}
        P.write_json(rd / 'shards' / 'a.json', {**base, 'owner': 'a', 'shard': 0, 'time': now - 10})
        P.write_json(rd / 'shards' / 'b.json', {**base, 'owner': 'b', 'shard': 1, 'time': now - 3600})   # stale
        P.write_json(rd / 'shards' / 'old.json', {**base, 'owner': 'old', 'shard': 0, 'state': 'exited',
                                                   'time': now - 7200, 'accepted': 2})
        (P.STREAMS / 'x' / 'chunks').mkdir(parents=True)
        P.write_json(P.STREAMS / 'x' / 'manifest.json', {'t1': {}, 't2': {}})
        ops = P.SYNTH / 'c-x' / 'ops'
        (ops / 'controller').mkdir(parents=True)
        P.write_json(ops / 'progress.json', {'phase': 'running', 'updated_at': now - 200, 'train_tokens_saved': 123,
                                             'active_train': 40, 'backends_ready': ['j0r0'],
                                             'per_task': {'missing_trajectories': 100, 'available_tasks': 10, 'tasks': 8}})
        P.write_json(ops / 'owned-jobs.json', {'jobs': [
            {'job_id': '104.0', 'role': 'controller', 'index': 0, 'generation': 0, 'time': 1},
            {'job_id': '105.0', 'role': 'serving', 'index': 0, 'generation': 0, 'time': 1},
            {'job_id': '106.0', 'role': 'worker', 'index': 0, 'generation': 0, 'time': 1}]})
        P.write_json(ops / 'campaign.json', {'serving': {'jobs': 4}})
        P.write_json(ops / 'admission-policy.json', {'serving_jobs': 2})
        return run, rd

    def status(self):
        return live.Status('x', metrics=False).collect()

    def test_failing_condor_q_is_unknown_not_gone(self):
        self.make_run()
        self.fake_condor_q('echo "Error: communication error" >&2\necho "SCHEDD:2:Failed to connect" >&2\nexit 1\n')
        s = self.status()
        self.assertEqual(s['condor'], 'unknown')
        c = s['components']
        self.assertEqual(c['driver']['condor'], 'unknown')
        self.assertEqual(c['servers'][0]['condor'], 'unknown')
        self.assertEqual(c['round']['condor'], 'unknown')
        self.assertEqual(c['stream']['condor'], 'unknown')
        self.assertEqual(c['super']['controller']['condor'], 'unknown')
        kinds = [a['kind'] for a in s['alerts']]
        self.assertIn('condor_unknown', kinds)
        self.assertNotIn('gone', kinds)                       # a failing condor_q is never "job gone" (07:34 outage)
        # heartbeats still speak: the stale shard and the late controller tick are alerts
        self.assertTrue(any(a['kind'] == 'stale_heartbeat' and a['component'] == 'round shard 1' for a in s['alerts']))
        self.assertIn('tick_late', kinds)
        # counters: summed over every job incarnation; live ones only for open / sessions / in flight
        k = c['round']['counters']
        self.assertEqual((k['claimed'], k['accepted'], k['done']), (30, 12, 18))
        self.assertEqual((k['open'], k['sessions'], k['inflight']), (4, 2, 3))
        self.assertEqual(c['super']['engines_cap'], 2)
        self.assertEqual(c['super']['queued_attempts'], 60)
        self.assertEqual(c['stream']['tasks'], 2)
        json.dumps(s)                                          # the --json output

    def test_condor_states_held_and_gone(self):
        self.make_run()
        # driver running, server held, round proc 0 running / proc 1 held, producer gone, controller running
        rows = ['100\t0\t2', '101\t0\t5', '102\t0\t2', '102\t1\t5', '104\t0\t2', '105\t0\t2']
        self.fake_condor_q('printf "' + '\\n'.join(rows) + '\\n"\n')
        s = self.status()
        self.assertEqual(s['condor'], 'ok')
        c = s['components']
        self.assertEqual(c['driver']['condor'], 'running')
        self.assertEqual(c['servers'][0]['condor'], 'held')
        self.assertEqual(c['round']['condor'], {'102.0': 'running', '102.1': 'held'})
        self.assertEqual(c['stream']['condor'], 'gone')
        alerts = {(a['component'], a['kind']) for a in s['alerts']}
        self.assertIn(('server x-ds1', 'held'), alerts)
        self.assertIn(('round', 'held'), alerts)
        self.assertIn(('stream', 'gone'), alerts)               # the producer should run until END
        self.assertNotIn(('driver', 'gone'), alerts)
        self.assertEqual(c['super']['engines_live'], 1)

    def test_cli_json(self):
        self.make_run()
        self.fake_condor_q('exit 1\n')
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            live.status(Namespace(name='x', json=True, no_metrics=True))
        s = json.loads(buf.getvalue())
        self.assertEqual(s['condor'], 'unknown')
        self.assertTrue(s['alerts'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            live.status(Namespace(name='x', json=False, no_metrics=True))
        self.assertIn('alerts (', buf.getvalue())


def fake_draw(n, seed, mix, vocab):
    """specs_mod.sample stand-in: n specs of seed `seed`, every other one eligible for binding."""
    out = []
    for i in range(n):
        s = {'spec_id': f's{seed}-{i:05d}', 'vocab': vocab, 'minutes': 10, 'bucket': 'best-generated',
             'assets': [{'role': 'anchor', 'class': 'python-code'}] if i % 2 == 0 else []}
        out.append(s)
    return out


def tree_digest(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob('*')):
        h.update(str(p).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


class ReplanSynthetic(Sandbox):
    """20 specs: s0-s3 started (claimed by a live round job, s0 with its record), s4-s19 unstarted; s4/s6/s8 go to
    binding, s4 has its result. Window 1: the jobs can start s5 (and s4: bound) ... up to the 2nd unresolved pending."""

    def setUp(self):
        super().setUp()
        self.specs = [{'spec_id': f's4000-{i:05d}', 'vocab': 'v2h', **({'bind': 'pending'} if i in (4, 6, 8) else {})}
                      for i in range(20)]
        self.run, self.rd = self.run_dir('x', self.specs)
        cfg = P.read_json(self.run / 'run.json')
        cfg['config']['round']['bind_window'] = 1
        P.write_json(self.run / 'run.json', cfg)
        (self.rd / 'bind').mkdir()
        P.write_json(self.rd / 'bind' / 's4000-00004.json', {'status': 'bound'})
        self.job = Claims(self.rd, 0, stale_after=600)
        self.job.beat(state='working', reload_specs=True)
        for i in range(4):
            self.assertTrue(self.job.claim(f's4000-{i:05d}'))
        P.write_json(self.rd / 'work' / 's4000-00000' / 'record.json', {'status': 'valid'})
        (P.BINDS / 'x').mkdir()
        P.write_json(P.BINDS / 'x' / 'job.json', {'job_id': '200.0', 'args': {'round': 'x'}})
        self.calls = []
        self.patches = {k: getattr(live, k) for k in ('draw', 'condor_cmd', 'job_owner_and_args', 'submit_bind',
                                                       'submit_round')}
        live.draw = fake_draw
        self.bind_state = {'200.0': 2}

        def condor_cmd(argv, timeout=120):
            self.calls.append(argv)
            if argv[0] == 'condor_hold':
                self.bind_state[argv[1]] = 5
            if argv[0] == 'condor_rm':
                self.bind_state.pop(argv[1], None)
            return True, ''
        live.condor_cmd = condor_cmd
        live.job_owner_and_args = lambda j: ((os.environ.get('USER', 'user'), self.bind_state[str(j)],
                                              '-m task_synthesis.assets.bind run --round x')
                                             if str(j) in self.bind_state else
                                             (os.environ.get('USER', 'user'), 2, '-m round run --round x')
                                             if str(j) == '300' else None)
        live.submit_bind = lambda d: self.calls.append(['bind submit', d.info['bind_selected']]) or '201.0'
        live.submit_round = lambda d: self.calls.append(['round submit']) or '301.0'

    def tearDown(self):
        for k, v in self.patches.items():
            setattr(live, k, v)
        super().tearDown()

    def args(self, **kw):
        return Namespace(**{'name': 'x', 'vocab': 'v2h0', 'mix': None, 'seed': 4100, 'asset_share': None, 'why': 'test',
                            'apply': False, 'restart_old_shards': False, 'confirm_minutes': 0, **kw})

    def test_dry_run_writes_nothing(self):
        before = tree_digest(self.root)
        out = []
        s = live.Replan(self.args(), log=out.append).run()
        self.assertEqual(tree_digest(self.root), before)
        self.assertEqual((s['kept'], s['new']), (4, 16))
        self.assertEqual(s['replaced_vocab'], {'v2h': 16})
        self.assertIn('dry run: nothing written (--apply to re-plan)', out)
        self.assertEqual(self.calls, [])
        # startable: s4 (bound), s5, s7 (1 unresolved pending before it, s6, <= window 1); s6 and s8 wait for their
        # results; s9 on have 2 unresolved before them
        self.assertIn('unstarted specs the round jobs could start now: 3', out[0])

    def test_apply(self):
        out = []
        s = live.Replan(self.args(apply=True), log=out.append).run()
        new = [json.loads(l) for l in (self.rd / 'specs.jsonl').read_text().splitlines()]
        ids = [x['spec_id'] for x in new]
        self.assertEqual(ids[:4], [f's4000-{i:05d}' for i in range(4)])        # started specs kept, in order
        self.assertEqual(ids[4:], [f's4100-{i:05d}' for i in range(16)])       # then the new draws
        self.assertTrue(all(x['vocab'] == 'v2h0' for x in new[4:]))
        self.assertEqual(sum(1 for x in new if x.get('bind') == 'pending'), 4)  # 8 eligible x 0.5
        # the specs the jobs could start were pre-claimed by the re-plan (no job can claim them any more)
        for i in (4, 5, 7):
            owner = json.loads((self.rd / 'work' / f's4000-{i:05d}' / '.claim' / 'owner').read_text())
            self.assertEqual(owner['owner'], s['owner'])
            self.assertFalse(Claims(self.rd, 1).claim(f's4000-{i:05d}'))
        self.assertFalse((self.rd / 'work' / 's4000-00009').exists())         # unreachable: not touched
        self.assertTrue((self.rd / 'specs.before-replan-1.jsonl').exists())
        self.assertEqual(P.read_json(self.rd / 'replan-1.json')['new'], 16)
        info = P.read_json(self.run / 'run.json')
        self.assertEqual(info['config']['recipe']['vocab'], 'v2h0')
        self.assertEqual((info['specs'], info['bind_selected']), (20, 4))
        self.assertEqual(info['replans'][0]['seed'], 4100)
        self.assertIn('production replan 1', info['overrides_log'][-1]['by'])
        # bind job: held before the rewrite, removed after, a new one submitted (it ramps its slots)
        self.assertEqual([c[0] for c in self.calls], ['condor_hold', 'condor_rm', 'bind submit'])
        self.assertEqual(self.calls[-1][1], 4)
        self.assertTrue(any('have not reloaded specs.jsonl' in l for l in out))   # the fake job never reloads
        # the same seed again is refused
        with self.assertRaises(SystemExit):
            live.Replan(self.args(apply=True), log=out.append).run()
        self.assertTrue(any('seed 4100 already drew' in l for l in out))

    def test_jobs_without_reload_are_refused_or_restarted(self):
        self.job.beat(state='working')                           # a job from before #238: no reload_specs
        P.write_json(self.rd / 'job.json', {'job_id': '300.0'})
        out = []
        live.Replan(self.args(), log=out.append).run()
        self.assertTrue(any('cannot reload specs.jsonl' in l for l in out))
        with self.assertRaises(SystemExit):
            live.Replan(self.args(apply=True), log=out.append).run()
        self.assertEqual(self.calls, [])
        live.Replan(self.args(apply=True, restart_old_shards=True), log=out.append).run()
        self.assertEqual([c[0] for c in self.calls], ['condor_hold', 'condor_rm', 'bind submit', 'condor_rm',
                                                       'round submit'])
        self.assertEqual(self.calls[3], ['condor_rm', '300'])

    def test_a_round_job_without_heartbeats_is_old(self):
        (self.rd / 'shards' / f'{self.job.owner}.json').unlink()
        P.write_json(self.rd / 'job.json', {'job_id': '300.0'})     # alive (the fake condor_q), no heartbeat at all
        out = []
        with self.assertRaises(SystemExit):
            live.Replan(self.args(apply=True), log=out.append).run()
        self.assertTrue(any('started before #171' in l for l in out))

    def test_stopping_round_is_refused(self):
        (self.rd / 'STOP').write_text('budget')
        with self.assertRaises(SystemExit):
            live.Replan(self.args(apply=True), log=lambda m: None).run()
        self.assertEqual(self.calls, [])


class OneOrRecord:
    """Like Round.one: a spec that already has its record (shard 0 claims finished specs to report them) returns it."""
    def __init__(self, fake):
        self.fake, self.ran = fake, fake.ran

    async def __call__(self, spec):
        p = self.fake.rnd.work / spec['spec_id'] / 'record.json'
        return json.loads(p.read_text()) if p.exists() else await self.fake(spec)


class ReplanLiveJobs(Sandbox):
    """Two round jobs run 10 specs (one session slot each, 0.6 s per spec); a re-plan at 0.2 s keeps their two started
    sessions, replaces the 8 others with new draws. Every spec of the new file ends with exactly one record, the started
    ones by their own job, and nothing runs that the re-plan removed."""

    def test_jobs_reload_and_keep_their_sessions(self):
        old = [{'spec_id': f's4000-{i:05d}', 'vocab': 'v2h'} for i in range(10)]
        run, rd = self.run_dir('x', old)
        orig = live.draw, live.submit_bind, live.CONFIRM_POLL_S
        live.draw, live.CONFIRM_POLL_S = fake_draw, 0.05
        live.submit_bind = lambda d: '201.0'
        try:
            jobs = [make_round(rd, k, 2, old, delay=0.6) for k in (0, 1)]
            for j in jobs:
                j.author_slots = asyncio.Semaphore(1)
                j.specs_version = j.specs_stat()
                j.one = OneOrRecord(j.one)

            async def beat(j):
                while True:
                    j.claims.beat(**j.beat_status())
                    await asyncio.sleep(0.05)

            async def go():
                specs = {id(j): list(old) for j in jobs}
                for j in jobs:
                    j.task_by_sid = {s['spec_id']: asyncio.create_task(j.guarded(s)) for s in old}
                    j.tasks_all = list(j.task_by_sid.values())
                side = [asyncio.create_task(beat(j)) for j in jobs]
                side += [asyncio.create_task(j.watch_specs(specs[id(j)], every=0.05)) for j in jobs]
                await asyncio.sleep(0.2)
                started = {sid for j in jobs for sid in j.mine}
                self.assertEqual(len(started), 2)
                summary = await asyncio.to_thread(
                    live.Replan(Namespace(name='x', vocab='v2h0', mix=None, seed=4100, asset_share=None, why='',
                                          apply=True, restart_old_shards=False, confirm_minutes=0.2),
                                log=lambda m: None).run)
                for j in jobs:
                    while [t for t in j.tasks_all if not t.done()]:
                        await asyncio.wait([t for t in j.tasks_all if not t.done()])
                for t in side:
                    t.cancel()
                for j in jobs:
                    if getattr(j, 'binding', None):
                        j.binding.cancel()
                return started, summary
            started, summary = asyncio.run(go())
        finally:
            live.draw, live.submit_bind, live.CONFIRM_POLL_S = orig
        self.assertEqual(summary['kept'], 2)
        new = [json.loads(l)['spec_id'] for l in (rd / 'specs.jsonl').read_text().splitlines()]
        self.assertEqual(set(new[:2]), started)
        self.assertEqual(len(new), 10)
        ran = [sid for j in jobs for sid in j.one.ran]
        self.assertEqual(sorted(ran), sorted(new))                       # each spec of the new file once, nothing else
        for sid in new:
            self.assertTrue((rd / 'work' / sid / 'record.json').exists(), sid)
        for sid in started:                                              # the started sessions were not cut or re-run
            rec = json.loads((rd / 'work' / sid / 'record.json').read_text())
            self.assertEqual(rec['status'], 'valid')
            self.assertNotIn('taken_over_from', rec)
        for j in jobs:
            self.assertTrue(any('specs.jsonl changed (re-plan)' in l for l in j.logged))
            self.assertFalse(j.foreign - set(new))                    # removed (pre-claimed) specs are not waited for
        self.assertEqual(sum(j.counts['claimed'] for j in jobs), 10)


class BindRamp(unittest.TestCase):
    def test_slots_open_evenly(self):
        from task_synthesis.assets.bind import ramp_slots, ramp_start
        self.assertEqual((ramp_start(600, 0.1), ramp_start(5, 0.1), ramp_start(600, 1.0)), (60, 1, 600))

        async def go():
            slots, cap, seen = asyncio.Semaphore(2), [2], []
            ramp = asyncio.create_task(ramp_slots(slots, 2, 10, 0.01, on_step=lambda k: cap.__setitem__(0, k)))
            running = [0]

            async def work():
                async with slots:
                    running[0] += 1
                    seen.append(running[0])
                    await asyncio.sleep(2)
                    running[0] -= 1
            ws = [asyncio.create_task(work()) for _ in range(20)]
            await asyncio.sleep(0.05)
            early = running[0]
            await ramp
            await asyncio.sleep(0.05)
            late = running[0]
            for w in ws:
                w.cancel()
            return early, late, cap[0]
        early, late, cap = asyncio.run(go())
        self.assertLessEqual(early, 3)        # 2 at the start, one more every 0.075 s
        self.assertEqual((late, cap), (10, 10))


if __name__ == '__main__':
    unittest.main()

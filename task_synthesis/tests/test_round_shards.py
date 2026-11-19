"""Offline tests for round shards that die or hang (#171): claims with owner + heartbeat, takeover of a lost job's
claims (including two jobs racing for the same stale claim), records written only by the claim's owner, the drained
job that stays for the others, and the exit after the final flush.

  python -m unittest task_synthesis.tests.test_round_shards     (harbor venv)
"""
import asyncio
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path

from task_synthesis.common import REPO
from task_synthesis.pipeline import round as R
from task_synthesis.pipeline.claims import LEGACY, Claims


def age(claims, seconds):
    """Pretend a job last beat `seconds` ago."""
    b = json.loads((claims.beats / f'{claims.owner}.json').read_text())
    b['time'] = time.time() - seconds
    (claims.beats / f'{claims.owner}.json').write_text(json.dumps(b))


class ClaimsUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_claim_owner_heartbeat_and_takeover(self):
        a, b = Claims(self.d, 0, stale_after=60), Claims(self.d, 1, stale_after=60)
        a.beat(state='working')
        b.beat(state='working')
        self.assertTrue(a.claim('s1'))
        self.assertFalse(b.claim('s1'))
        self.assertEqual(a.current('s1')[:2], (0, a.owner))
        self.assertTrue(a.owns('s1') and not b.owns('s1'))
        self.assertIsNone(b.stale('s1'))                 # a beats: not stale
        age(a, 61)
        b._beats_at = 0
        gen, owner, _ = b.stale('s1')
        self.assertEqual((gen, owner), (0, a.owner))
        self.assertTrue(b.take_over('s1', gen))
        self.assertFalse(Claims(self.d, 2, stale_after=60).take_over('s1', gen))   # generation 1 is taken
        self.assertTrue(b.owns('s1') and not a.owns('s1'))
        self.assertEqual(b.current('s1')[:2], (1, b.owner))
        self.assertIsNone(b.stale('s1'))
        # the journals list what each job claimed or took over, before the mkdir
        self.assertEqual(b.journal(a.owner), (['s1'], 3))
        self.assertEqual(b.journal(b.owner)[0], ['s1'])
        self.assertEqual(b.journal(a.owner, 3), ([], 3))

    def test_legacy_and_ownerless_claims(self):
        c = Claims(self.d, 0, stale_after=60)
        c.beat()
        (self.d / 'work' / 'old' / '.claim').mkdir(parents=True)
        (self.d / 'work' / 'old' / '.claim' / 'owner').write_text('cpu202 shard 0 pid 3831316\n')   # m1-hero's format
        self.assertEqual(c.current('old')[1], LEGACY)
        self.assertIsNone(c.stale('old'))                # never taken over automatically
        (self.d / 'work' / 'half' / '.claim').mkdir(parents=True)   # died between mkdir and the owner write
        self.assertIsNone(c.stale('half'))               # fresh: its owner may still be writing
        old = time.time() - 120
        os.utime(self.d / 'work' / 'half' / '.claim', (old, old))
        self.assertEqual(c.stale('half')[:2], (0, None))
        self.assertIsNone(c.current('nobody'))


def _race(round_dir, sids, barrier, q, k):
    c = Claims(round_dir, 10 + k, stale_after=60)
    c.beat(state='working')
    won = []
    barrier.wait()
    for sid in sids:
        cur = c.stale(sid)
        if cur is not None and c.take_over(sid, cur[0]):
            won.append(sid)
    q.put((c.owner, won))


class TakeoverRace(unittest.TestCase):
    def test_two_and_four_jobs_race_for_the_same_stale_claims(self):
        """Separate processes (as on two nodes) take over the same 200 stale claims at once: each claim has exactly one
        new owner, and that owner is the one the claim names."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dead = Claims(d, 0, stale_after=60)
            dead.beat()
            sids = [f's{i}' for i in range(200)]
            for sid in sids:
                self.assertTrue(dead.claim(sid))
            age(dead, 3600)
            for n in (2, 4):
                ctx = mp.get_context('fork')
                barrier, q = ctx.Barrier(n), ctx.Queue()
                ps = [ctx.Process(target=_race, args=(d, sids, barrier, q, k + 10 * n)) for k in range(n)]
                for p in ps:
                    p.start()
                out = [q.get(timeout=60) for _ in ps]
                for p in ps:
                    p.join(30)
                won = [s for _, w in out for s in w]
                self.assertEqual(sorted(won), sorted(sids))          # every claim taken, none twice
                reader = Claims(d, 99, stale_after=60)
                for owner, w in out:
                    for sid in w:
                        self.assertEqual(reader.current(sid)[1], owner)
                # make this round's winners the dead ones for the next race
                for owner, _ in out:
                    b = json.loads((d / 'shards' / f'{owner}.json').read_text())
                    b['time'] = 0
                    (d / 'shards' / f'{owner}.json').write_text(json.dumps(b))


class FakeRun:
    """Stands in for Round.one(): writes the record through write_record after `delay` s."""
    def __init__(self, rnd, delay=0.05):
        self.rnd, self.delay, self.ran = rnd, delay, []

    async def __call__(self, spec):
        self.ran.append(spec['spec_id'])
        (self.rnd.work / spec['spec_id'] / 'attempt-0.reply.txt').write_text('x')
        await asyncio.sleep(self.delay)
        rec = {'spec_id': spec['spec_id'], 'status': 'valid', 'by': self.rnd.claims.owner}
        self.rnd.write_record(spec['spec_id'], rec)
        return rec


def make_round(d, shard, shards, specs, delay=0.05, stale=60):
    r = R.Round.__new__(R.Round)
    r.a = Namespace(resume=False, round='t')
    r.dir, r.work, r.tasks = Path(d), Path(d) / 'work', Path(d) / 'tasks'
    for x in (r.work, r.tasks):
        x.mkdir(parents=True, exist_ok=True)
    r.shard, r.shards = shard, shards
    r.foreign, r.journal_at, r.journal_pending, r.mine, r.lost, r.takeovers = set(), {}, {}, {}, set(), {}
    r.done, r.state, r._last_beat, r.tail_cut, r.open_authors = 0, 'working', time.monotonic(), False, 0
    r.logged = []
    r.log = r.logged.append
    r.claims = Claims(d, shard, stale_after=stale)
    r.claims.beat(state='working')
    r.spec_by_id = {s['spec_id']: s for s in specs}
    r.author_slots = asyncio.Semaphore(1000)
    r.tasks_all = []

    async def ready(spec):
        return spec

    async def admit():
        return None
    r.bind_ready, r.admit = ready, admit
    r.one = FakeRun(r, delay)
    return r


class RoundTakeover(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.specs = [{'spec_id': f's{i}'} for i in range(6)]

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, sid, name='record.json'):
        p = self.d / 'work' / sid / name
        return json.loads(p.read_text()) if p.exists() else None

    def test_lost_job_claims_are_rerun_by_a_live_job(self):
        dead = make_round(self.d, 0, 2, self.specs)
        live = make_round(self.d, 1, 2, self.specs)
        for sid in ('s0', 's1', 's2'):                     # the dead job's in-flight specs
            self.assertTrue(dead.claim({'spec_id': sid}))
            (self.d / 'work' / sid / 'attempt-0.reply.txt').write_text('stale reply')
            (self.d / 'tasks' / sid).mkdir()
        self.assertTrue(dead.claims.claim('s5'))
        dead.write_record('s5', {'spec_id': 's5', 'status': 'valid'})   # finished before it died: left alone

        async def go():
            live.tasks_all = [asyncio.create_task(live.guarded(s)) for s in self.specs]
            await asyncio.gather(*live.tasks_all)
            self.assertEqual(sorted(live.foreign), ['s0', 's1', 's2'])   # s5 has its record: not foreign
            self.assertEqual(await live.take_stale(), [])          # the dead job still looks alive
            age(dead.claims, 3600)
            live.claims._beats_at = 0
            taken = await live.take_stale()
            self.assertEqual(sorted(taken), ['s0', 's1', 's2'])
            await asyncio.gather(*live.tasks_all)
        asyncio.run(go())
        for sid in ('s0', 's1', 's2', 's3', 's4'):
            r = self.rec(sid)
            self.assertEqual(r['by'], live.claims.owner, sid)
        for sid in ('s0', 's1', 's2'):
            self.assertEqual(self.rec(sid)['taken_over_from'], dead.claims.owner)
            self.assertEqual((self.d / 'work' / sid / 'attempt-0.reply.txt').read_text(), 'x')   # from scratch
            self.assertFalse((self.d / 'tasks' / sid).exists())
        self.assertNotIn('by', self.rec('s5'))
        self.assertEqual(sorted(live.one.ran), ['s0', 's1', 's2', 's3', 's4'])
        self.assertTrue(all((self.d / 'work' / s['spec_id'] / 'record.json').exists() for s in self.specs))

    def test_two_live_jobs_race_for_the_same_lost_claims(self):
        """Two live jobs (each its own event loop and thread) sweep the same dead job at the same moment: each spec runs
        once, and the round ends with one record per spec."""
        dead = make_round(self.d, 0, 3, self.specs)
        for s in self.specs:
            self.assertTrue(dead.claim(s))
        age(dead.claims, 3600)
        jobs = [make_round(self.d, k, 3, self.specs) for k in (1, 2)]
        barrier = threading.Barrier(2)

        def run(job):
            async def go():
                job.tasks_all = [asyncio.create_task(job.guarded(s)) for s in self.specs]
                await asyncio.gather(*job.tasks_all)
                job.claims._beats_at = 0
                await asyncio.to_thread(barrier.wait)
                await job.take_stale()
                await asyncio.gather(*job.tasks_all)
            asyncio.run(go())
        ts = [threading.Thread(target=run, args=(j,)) for j in jobs]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        ran = jobs[0].one.ran + jobs[1].one.ran
        self.assertEqual(sorted(ran), sorted(s['spec_id'] for s in self.specs))
        for s in self.specs:
            r = self.rec(s['spec_id'])
            self.assertIn(r['by'], (jobs[0].claims.owner, jobs[1].claims.owner))
            self.assertEqual(r['taken_over_from'], dead.claims.owner)

    def test_a_job_that_lost_its_claim_writes_nothing(self):
        slow = make_round(self.d, 0, 2, self.specs[:1], delay=0.5)
        other = make_round(self.d, 1, 2, self.specs[:1])

        async def go():
            t = asyncio.create_task(slow.guarded(self.specs[0]))
            await asyncio.sleep(0.1)                      # slow owns s0 and is running it
            age(slow.claims, 3600)                        # ... and stalls past the stale limit
            other.foreign.add('s0')
            other.claims._beats_at = 0
            other.tasks_all = []
            self.assertEqual(await other.take_stale(), ['s0'])
            await asyncio.gather(*other.tasks_all)
            self.assertIsNone(await t)                    # slow wakes: LostClaim, no record
        asyncio.run(go())
        self.assertEqual(self.rec('s0')['by'], other.claims.owner)
        self.assertIn('s0: claim lost to another job; dropped', slow.logged)

    def test_stalled_job_drops_lost_claims(self):
        job = make_round(self.d, 0, 2, self.specs[:2], delay=30)
        thief = Claims(self.d, 1, stale_after=60)
        thief.beat()

        async def go():
            job.tasks_all = [asyncio.create_task(job.guarded(s)) for s in self.specs[:2]]
            await asyncio.sleep(0.1)
            self.assertEqual(sorted(job.mine), ['s0', 's1'])
            self.assertTrue(thief.take_over('s0', 0))
            await job.drop_lost_claims()
            res = await asyncio.gather(*job.tasks_all, return_exceptions=True)
            return res
        res = asyncio.run(go())
        self.assertIsNone(res[0])                          # dropped, no record
        self.assertIsNone(self.rec('s0'))
        self.assertTrue(isinstance(res[1], asyncio.CancelledError) or res[1] is None or res[1]['status'] == 'valid')

    def test_takeover_after_a_stop_records_tail_cut(self):
        dead = make_round(self.d, 0, 2, self.specs[:1])
        self.assertTrue(dead.claim(self.specs[0]))
        age(dead.claims, 3600)
        live = make_round(self.d, 1, 2, self.specs[:1])
        live.tail_cut = True
        live.foreign.add('s0')

        async def go():
            await live.take_stale()
            await asyncio.gather(*live.tasks_all)
        asyncio.run(go())
        self.assertEqual(self.rec('s0'), {'spec_id': 's0', 'status': 'tail_cut', 'orphaned_from': dead.claims.owner,
                                          'taken_over_from': dead.claims.owner})
        self.assertEqual(live.one.ran, [])

    def test_exception_records_error_and_finished_accounting(self):
        job = make_round(self.d, 0, 1, self.specs[:1])

        async def boom(spec):
            raise ValueError('bad')
        job.one = boom
        asyncio.run(job.guarded(self.specs[0]))
        self.assertEqual(self.rec('s0', 'error.json')['status'], 'error')
        self.assertTrue(job.spec_finished('s0'))
        self.assertEqual(job.done, 1)

    def test_only_the_lowest_drained_job_stays(self):
        specs = self.specs[:2]
        busy = make_round(self.d, 0, 3, specs)
        self.assertTrue(busy.claim(specs[0]))             # s0 still running in job 0
        j1, j2 = make_round(self.d, 1, 3, specs), make_round(self.d, 2, 3, specs)
        for j in (j1, j2):
            j.foreign.add('s0')
            j.state = 'draining'
            j.claims.beat(state='draining')

        async def go():
            return await j2.may_exit(), await j1.may_exit()
        self.assertEqual(asyncio.run(go()), (True, False))   # job 2 exits, job 1 stays for s0
        busy.write_record('s0', {'spec_id': 's0', 'status': 'valid'})
        self.assertTrue(asyncio.run(j1.may_exit()))          # nothing open: job 1 exits too
        j1.claims.beat(state='exited')
        j2.foreign.add('s1')
        self.assertTrue(busy.claim(specs[1]))
        self.assertFalse(asyncio.run(j2.may_exit()))         # job 1 exited: job 2 is the lowest drained one now

    def test_resume_single_job_takes_its_own_old_claims(self):
        old = make_round(self.d, 0, 1, self.specs[:1])
        self.assertTrue(old.claim(self.specs[0]))
        new = make_round(self.d, 0, 1, self.specs[:1])
        new.a.resume = True
        self.assertTrue(new.claim(self.specs[0]))
        self.assertTrue(new.claims.owns('s0'))
        new2 = make_round(self.d, 0, 2, self.specs[:1])
        self.assertFalse(new2.claim(self.specs[0]))       # several jobs: never by --resume, only by takeover


class StopMarksNotRun(unittest.TestCase):
    def test_lowest_live_job_marks_unclaimed_specs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [{'spec_id': f's{i}'} for i in range(4)]
            a, b = make_round(tmp, 1, 2, specs), make_round(tmp, 2, 2, specs)
            self.assertTrue(a.claim(specs[0]))
            self.assertTrue(a.lowest_live() and not b.lowest_live())
            n = a.mark_not_run(specs) + b.mark_not_run(specs)
            self.assertEqual(n, 3)
            for s in specs[1:]:
                self.assertEqual(json.loads((Path(tmp) / 'work' / s['spec_id'] / 'record.json').read_text())['status'],
                                 'not_run')
            self.assertFalse((Path(tmp) / 'work' / 's0' / 'record.json').exists())


class HeartbeatFromTheLoop(unittest.TestCase):
    def test_a_blocked_loop_stops_beating(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_round(tmp, 0, 1, [])
            path = job.claims.beats / f'{job.claims.owner}.json'

            async def go():
                hb = asyncio.create_task(job.heartbeat(every=0.05))
                await asyncio.sleep(0.2)
                t1 = json.loads(path.read_text())['time']
                time.sleep(0.5)                               # the loop is blocked: no beat
                t2 = json.loads(path.read_text())['time']
                await asyncio.sleep(0.2)
                t3 = json.loads(path.read_text())['time']
                hb.cancel()
                return t1, t2, t3
            t1, t2, t3 = asyncio.run(go())
            self.assertEqual(t1, t2)
            self.assertGreater(t3, t2)


class ExitAfterFlush(unittest.TestCase):
    def test_run_and_exit_does_not_wait_for_stuck_threads(self):
        """A non-daemon thread that never ends (a stuck executor thread at m1-hero's end) must not keep the job alive."""
        code = r'''
import asyncio, sys, threading, time
from task_synthesis.pipeline import round as R
class Fake:
    def __init__(self, a): pass
    async def run(self):
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, time.sleep, 3600)      # a stuck executor thread
        threading.Thread(target=time.sleep, args=(3600,)).start()
        print('final line', flush=True)
R.Round = Fake
R.run_and_exit(None)
'''
        t0 = time.time()
        p = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120,
                           env={**os.environ, 'PYTHONPATH': str(REPO)})
        self.assertLess(time.time() - t0, 60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('final line', p.stdout)
        self.assertIn('busy threads', p.stderr)

    def test_hang_watchdog_exits_a_blocked_job(self):
        code = r'''
import time
from task_synthesis.pipeline import round as R
r = R.Round.__new__(R.Round)
r._last_beat = time.monotonic() - 100      # the event loop has not beaten for 100 s
r.hang_watchdog(2)
'''
        p = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=60,
                           env={**os.environ, 'PYTHONPATH': str(REPO)})
        self.assertEqual(p.returncode, 3)
        self.assertIn('event loop blocked', p.stderr)
        self.assertIn('hang_watchdog', p.stderr)            # the thread stacks

    def test_report_in_job_skips_pairwise_work_on_big_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig, R.ROUNDS = R.ROUNDS, Path(tmp)
            try:
                n = R.REPORT_PAIRS_MAX + 1
                for i in range(n):
                    d = Path(tmp) / 'big' / 'work' / f's{i}'
                    d.mkdir(parents=True)
                    (d / 'record.json').write_text(json.dumps(
                        {'spec_id': f's{i}', 'status': 'valid', 'repairs': 0, 'instruction': f'task number {i} alpha',
                         'judge': {'pass': True, 'scores': {'clarity': 5}, 'flags': {}}, 'attempts': []}))
                t0 = time.time()
                R.report('big', in_job=True)
                self.assertLess(time.time() - t0, 60)
                s = json.loads((Path(tmp) / 'big' / 'summary.json').read_text())
                self.assertTrue(s['near_duplicates_skipped'])
            finally:
                R.ROUNDS = orig


if __name__ == '__main__':
    unittest.main()

"""#378: a pi_eval worker of a growing dispatcher leaves after sitting idle with no start credit (PiEvalWorker
watch_idle / idle_state), and says so in its heartbeat file."""
import asyncio
import json

from aa_evals.common.pi_eval import worker as W


class FakeState:
    def __init__(self, q, items):
        self.q, self._items = q, items

    def _names(self, sub, suffix=''):
        d = self.q / sub
        return {p.name.removesuffix(suffix) for p in d.iterdir()} if d.exists() else set()

    def items(self):
        return self._items

    def complete(self):
        return getattr(self, 'done_all', False)

    def stale(self, max_age):
        import time
        out = []
        for k in self._names('claims') - self._names('done', '.json'):
            d = self.q / 'claims' / k
            if time.time() - max(p.stat().st_mtime for p in [d, *d.iterdir()]) > max_age:
                out.append(k)
        return out


def make(tmp_path, max_starts=2, claims=(), items=(('t1', 0), ('t2', 0), ('t3', 0))):
    run = tmp_path / 'run'
    for sub in ('queue/claims', 'queue/done', 'queue/retry', 'workers'):
        (run / sub).mkdir(parents=True, exist_ok=True)
    for k in claims:
        (run / 'queue' / 'claims' / k).mkdir()
    (run / 'control.json').write_text(json.dumps({'max_starts': max_starts}))
    (run / 'jobs.json').write_text('[]')
    w = object.__new__(W.PiEvalWorker)
    w.run, w.name, w.running, w.ok, w.hold, w.gate_host = run, 'w1', {}, True, '', None
    w.state = FakeState(run / 'queue', list(items))
    w.idle_exit_sec, w.idle_exited, w.leaving = 0.05, False, False
    return w


def test_idle_state(tmp_path):
    w = make(tmp_path, max_starts=2, claims=['t1__a0__r0', 't2__a0__r0'])
    assert w.idle_state() == (True, True)                    # idle, work pending, credit all spent
    (w.run / 'control.json').write_text(json.dumps({'max_starts': 3}))
    assert w.idle_state() == (True, False)                   # one start of credit left: not yet
    w.running = {'t1__a0__r0': {}}
    assert w.idle_state()[0] is False                        # running something
    w.running, w.hold = {}, 'gate holds 3 new conversation(s)'
    assert w.idle_state()[0] is False                        # paused by the gate: the dispatcher holds credit back
    w.hold = ''
    (w.run / 'STOP_WORKERS').write_text('')
    assert w.idle_state()[0] is False
    (w.run / 'STOP_WORKERS').unlink()
    w2 = make(tmp_path / 'b', max_starts=0)
    assert w2.idle_state()[0] is False                       # no credit granted yet (start-up hold): stays
    w4 = make(tmp_path / 'd', max_starts=2, claims=['t1__a0__r0', 't2__a0__r0'])
    (w4.run / 'jobs.json').unlink()
    assert w4.idle_state()[0] is False                       # --no-submit run: nothing would grow it back
    w5 = make(tmp_path / 'e', max_starts=2, claims=['t1__a0__r0', 't2__a0__r0'])
    import os
    os.utime(w5.run / 'queue' / 'claims' / 't1__a0__r0', (1, 1))
    assert w5.idle_state()[0] is False                       # a lost worker's claim: stay to reap it
    w3 = make(tmp_path / 'c', max_starts=3, claims=['t1__a0__r0', 't2__a0__r0', 't3__a0__r0'])
    assert w3.idle_state()[0] is False                       # queue fully claimed: eval_q35's own idle exit


def test_a_worker_that_never_claimed_stays_through_the_start_up_grace(tmp_path):
    # val-sb10b 11-18: a lazy lane lost 2 of its 4 first workers to idle exit during the launch grab, never revived
    import time
    w = make(tmp_path, max_starts=2, claims=['t1__a0__r0', 't2__a0__r0'])
    w.idle_exit_sec, w.claimed_any, w.started_at = 600.0, False, time.time() - 900
    assert w.idle_state()[0] is False                        # never claimed, up 15 min < 2 x 10 min: stays
    w.started_at = time.time() - 1300
    assert w.idle_state()[0] is True                         # past the grace: an idle worker leaves as before
    w.started_at, w.claimed_any = time.time(), True
    assert w.idle_state()[0] is True                         # it claimed before: the usual idle exit


def test_exit_reason_marks_the_no_claimable_exit(tmp_path, monkeypatch):
    # all claimed, nothing running: eval_q35's no-claimable exit
    w = make(tmp_path, max_starts=3, claims=['t1__a0__r0', 't2__a0__r0', 't3__a0__r0'])
    w.idle_exited = False
    monkeypatch.setattr(W.q35, 'STOP', False)
    assert w.exit_reason() == 'no-claimable'
    monkeypatch.setattr(W.q35, 'STOP', True)
    assert w.exit_reason() is None                           # a signal (SIGTERM, condor_rm, eviction): no mark
    monkeypatch.setattr(W.q35, 'STOP', False)
    w.state.done_all = True
    assert w.exit_reason() == 'complete'                     # never makes a lane revivable
    w.state.done_all = False
    w.running = {'t1__a0__r0': {}}
    assert w.exit_reason() is None                           # attempts still running: a stop or a signal
    w.running = {}
    (w.run / 'STOP_WORKERS').write_text('')
    assert w.exit_reason() is None                           # stopped on purpose: not a reason to revive
    (w.run / 'STOP_WORKERS').unlink()
    w.idle_exited = True
    assert w.exit_reason() == 'idle-no-credit'
    w.mark_exited('no-claimable')
    assert json.loads((w.run / 'workers' / 'w1.json').read_text())['exited'] == 'no-claimable'


def test_main_marks_why_it_left(tmp_path, monkeypatch):
    async def nothing(self=None):
        return None

    def go(stop=False, stop_file=False, crash=False):
        w = make(tmp_path / f'm{stop}{stop_file}{crash}', max_starts=3, claims=['t1__a0__r0', 't2__a0__r0'])
        w.idle_exited, w.idle_exit_sec = False, 0

        async def inner(self):
            if crash:
                raise RuntimeError('boom')
            W.q35.STOP = stop
        monkeypatch.setattr(W.q35.Worker, 'main', inner)
        monkeypatch.setattr(W.PiEvalWorker, 'refresh', lambda self: None)
        monkeypatch.setattr(W.PiEvalWorker, 'keep_admission', nothing)
        monkeypatch.setattr(W.q35, 'STOP', False)
        if stop_file:
            (w.run / 'STOP_WORKERS').write_text('')
        try:
            asyncio.run(w.main())
        except RuntimeError:
            pass
        hb = w.run / 'workers' / 'w1.json'
        return json.loads(hb.read_text()).get('exited') if hb.exists() else None

    assert go() == 'no-claimable'
    assert go(stop=True) is None                             # a signal
    assert go(stop_file=True) is None                        # run.py --stop
    assert go(crash=True) is None                            # a crash
    monkeypatch.setattr(W.q35, 'STOP', False)


def test_watch_idle_exits_and_marks(tmp_path, monkeypatch):
    monkeypatch.setattr(W, 'IDLE_EVERY', 0.01)
    w = make(tmp_path, max_starts=1, claims=['t1__a0__r0'])
    try:
        asyncio.run(asyncio.wait_for(w.watch_idle(), 5))
        assert w.idle_exited and W.q35.STOP
    finally:
        W.q35.STOP = False
    w.mark_exited()
    rec = json.loads((w.run / 'workers' / 'w1.json').read_text())
    assert rec['exited'] == 'idle-no-credit' and rec['heartbeat']


def test_watch_idle_stays_while_credit_is_unspent(tmp_path, monkeypatch):
    monkeypatch.setattr(W, 'IDLE_EVERY', 0.01)
    w = make(tmp_path, max_starts=2, claims=['t1__a0__r0'])

    async def run():
        task = asyncio.create_task(w.watch_idle())
        await asyncio.sleep(0.3)
        assert not task.done()
        task.cancel()
    asyncio.run(run())
    assert not w.idle_exited and not W.q35.STOP


def test_try_leave_under_the_claim_lock(tmp_path):
    w = make(tmp_path, max_starts=2, claims=['t1__a0__r0', 't2__a0__r0'])
    (w.run / 'queue' / 'claims' / 't2__a0__r0' / 'claim.json').write_text(json.dumps({'worker': 'w1'}))
    assert w.try_leave() is False and not w.leaving          # its own claim (being started): stays
    (w.run / 'queue' / 'claims' / 't2__a0__r0' / 'claim.json').write_text(json.dumps({'worker': 'w9'}))
    (w.run / 'queue' / 'dispatch.lock').mkdir()
    assert w.try_leave() is False                            # a claim in progress: next round
    (w.run / 'queue' / 'dispatch.lock').rmdir()
    (w.run / 'control.json').write_text(json.dumps({'max_starts': 3}))
    assert w.try_leave() is False                            # credit arrived meanwhile
    (w.run / 'control.json').write_text(json.dumps({'max_starts': 2}))
    assert w.try_leave() is True and w.leaving
    assert not (w.run / 'queue' / 'dispatch.lock').exists()


def test_leaving_worker_claims_nothing(tmp_path, monkeypatch):
    """PiEvalWorker's credit_claim refuses claims, under the claim lock, once the worker decided to leave."""
    import types
    w = make(tmp_path, max_starts=5)
    calls = []
    w.state.claim = lambda worker, avail: calls.append(worker) or 'k'
    w.config = {'dispatch': {'dir': str(tmp_path / 'nodisp')}, 'ctx_gate': False}
    monkeypatch.setattr(W.q35.Worker, '__init__', lambda self, a: None)
    W.PiEvalWorker.__init__(w, types.SimpleNamespace())
    assert w.state.claim('w1', []) == 'k'
    w.leaving = True
    assert w.state.claim('w1', []) is None and calls == ['w1']

"""#531 A5: the hung-session watchdog (hung.py) and its hook in PiEvalWorker.run_item."""
import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from aa_evals.common.pi_eval import hung, request_gaps
from aa_evals.common.pi_eval import worker as W


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def proxy_dir(raw: Path, api='http://127.0.0.1:9/v1', session=True):
    d = raw / '.pi-cont16k' / 'k-abc'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'protocol.json').write_text(json.dumps({'proxy_api': api}))
    if session:
        say(raw, tool_call('a'), tool_result('a'))


def sessions(raw: Path) -> Path:
    d = raw / 'job' / 'trial' / 'agent' / 'pi' / 'sessions'
    d.mkdir(parents=True, exist_ok=True)
    return d / 's.jsonl'


def say(raw: Path, *entries):
    with open(sessions(raw), 'a') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')


def tool_call(i):
    return {'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'toolUse',
                                           'content': [{'type': 'toolCall', 'id': i}]}}


def tool_result(i):
    return {'type': 'message', 'message': {'role': 'toolResult', 'toolCallId': i, 'content': []}}


def test_watchdog_rule(tmp_path):
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    c, clk = {'in_flight': 0, 'requests': 0, 'model_sec': 0.0}, Clock()
    wd = hung.Watchdog(600, read=lambda api: dict(c), clock=clk)
    assert wd.tick(raw) is None
    clk.t += 3600
    assert wd.tick(raw) is None and not wd.armed          # no request yet (env build, setup): not judged
    c.update(requests=1, in_flight=1, model_sec=2.0)
    assert wd.tick(raw) is None and wd.armed
    clk.t += 3600
    assert wd.tick(raw) is None                           # inside a request (or parked at a gate): activity
    c.update(in_flight=0, model_sec=3700.0)
    assert wd.tick(raw) is None                           # it ended since the last tick: activity
    clk.t += 500
    assert wd.tick(raw) is None                           # 500 s idle < 600
    clk.t += 200
    assert wd.tick(raw) == ('idle', 700)                  # 700 s, no request, no tool running: hung
    c.update(requests=2)
    assert wd.tick(raw) is None                           # a new request resets
    clk.t += 700
    assert wd.tick(raw, ok=False) is None                 # an outage the worker sees: not idle (reset)
    clk.t += 599
    assert wd.tick(raw) is None
    clk.t += 2
    assert wd.tick(raw) == ('idle', 601)


def test_no_evidence_never_fires_and_agent_end_disarms(tmp_path):
    raw = tmp_path / 'raw'
    clk = Clock()
    wd = hung.Watchdog(10, read=lambda api: {'in_flight': 0, 'requests': 3, 'model_sec': 1.0}, clock=clk)
    for _ in range(5):                                    # no proxy address yet
        clk.t += 100
        assert wd.tick(raw) is None
    proxy_dir(raw)
    wd.read = lambda api: None                            # the proxy does not answer
    for _ in range(5):
        clk.t += 100
        assert wd.tick(raw) is None
    wd.read = lambda api: {'in_flight': 0, 'requests': 3, 'model_sec': 1.0}
    assert wd.tick(raw) is None
    clk.t += 11
    assert wd.tick(raw) == ('idle', 11)
    agent = raw / 'job' / 'trial' / 'agent'
    agent.mkdir(parents=True, exist_ok=True)
    (agent / 'env_clock.json').write_text('{}')           # the agent ended: the verifier runs, never cut
    clk.t += 1000
    assert wd.tick(raw) is None and wd.ended


def test_tool_calls_and_the_final_answer_are_not_idle(tmp_path):
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    wd = hung.Watchdog(600, read=lambda api: {'in_flight': 0, 'requests': 5, 'model_sec': 9.0}, clock=clk)
    assert wd.tick(raw) is None
    say(raw, tool_call('b'))                              # a long tool call (a training run): Pi waits on it
    for _ in range(12):
        clk.t += 600
        assert wd.tick(raw) is None
    say(raw, tool_result('b'))                            # the tool returned: from here Pi owes a request
    assert wd.tick(raw) is None
    clk.t += 601
    assert wd.tick(raw) == ('idle', 601)
    say(raw, {'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'stop', 'content': []}})
    assert wd.tick(raw) is None                           # the final answer: Pi exits, or Harbor ends it and scores
    clk.t += 5000
    assert wd.tick(raw) is None
    raw2 = tmp_path / 'raw2'
    proxy_dir(raw2, session=False)                        # no session log: no tool evidence, never judged
    wd2 = hung.Watchdog(10, read=lambda api: {'in_flight': 0, 'requests': 5, 'model_sec': 9.0}, clock=clk)
    for _ in range(5):
        clk.t += 100
        assert wd2.tick(raw2) is None
    s = hung.SessionLog()                                 # torn last line: read once complete
    path = sessions(tmp_path / 'raw3')
    path.write_text(json.dumps(tool_call('c'))[:20])
    assert s.poll(tmp_path / 'raw3') == (True, True) and not s.outstanding
    with open(path, 'a') as f:
        f.write(json.dumps(tool_call('c'))[20:] + '\n')
    assert s.poll(tmp_path / 'raw3') == (True, True) and s.outstanding == ['c']


def wire(raw: Path, i: int, started: float, answered: bool):
    d = raw / '.pi-cont16k' / 'k-abc' / 'wire-responses'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{i:06d}.meta.json').write_text(json.dumps({'started': started, 'agent_request': True}))
    if answered:
        (d / f'{i:06d}.response').write_text('data: ...')


def test_the_11_15_hang_is_a_stalled_request(tmp_path):
    # the real signature (m5l_gdpval_t2): a forwarded request never answered; Pi resends it every 1800 s; the proxy has
    # a request in flight the whole time, requests and model_sec keep rising
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    c = {'in_flight': 1, 'requests': 20, 'model_sec': 900.0}
    wd = hung.Watchdog(600, 3660, read=lambda api: dict(c), clock=clk)
    for i in range(1, 21):
        wire(raw, i, clk.t - 30 * (21 - i), True)          # an ordinary session so far
    wire(raw, 21, clk.t, False)                          # forwarded; no response ever starts
    t0 = clk.t
    fired = None
    for minute in range(1, 121):
        clk.t = t0 + 60 * minute
        c['model_sec'] += 60
        if minute % 30 == 0:                             # Pi's request timeout: it resends
            c['requests'] += 1
            wire(raw, 21 + minute // 30, clk.t, False)
        fired = wd.tick(raw)
        if fired:
            break
    assert fired[0] == 'stalled_request' and 3660 < fired[1] <= 3720   # ~61 min, not the 2 h cap


def test_the_guards_accounting_error_ends_the_session_at_once(tmp_path):
    # the cause of the 11-15 hangs: the response-budget guard's sticky accounting failure (every later request: 502)
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    wd = hung.Watchdog(600, 3660, read=lambda api: {'in_flight': 1, 'requests': 9, 'model_sec': 50.0}, clock=clk)
    log = raw / '.pi-cont16k' / 'k-abc' / 'response-budget-wire.jsonl'
    log.write_text(json.dumps({'time': clk.t, 'policy': 'p', 'turn': 3}) + '\n')
    assert wd.tick(raw) is None
    with open(log, 'a') as f:
        f.write(json.dumps({'time': clk.t, 'accounting_error': '', 'upstream': {}})[:30])   # torn: not yet
    assert wd.tick(raw) is None
    with open(log, 'a') as f:
        f.write(json.dumps({'time': clk.t, 'accounting_error': '', 'upstream': {}})[30:] + '\n')
    clk.t += 60
    assert wd.tick(raw) == ('budget_accounting_failure', 0.0)
    assert wd.tick(raw, ok=False) == ('budget_accounting_failure', 0.0)   # sticky: an outage does not excuse it


def test_a_slow_session_that_survives_one_pi_timeout_is_not_stalled(tmp_path):
    # the guard buffers whole replies: .response appears when the reply is complete. A request lost to one Pi timeout
    # (1800 s) and a slow resend (1,250 s) must not fire; error indices answered later and out-of-order answers clear
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    c = {'in_flight': 1, 'requests': 3, 'model_sec': 0.0}
    wd = hung.Watchdog(600, 3660, read=lambda api: dict(c), clock=clk)
    wire(raw, 1, clk.t, True)
    wire(raw, 2, clk.t, False)                           # never answered (Pi gives up after 1800 s)
    for _ in range(30):
        clk.t += 60
        c['model_sec'] += 60
        assert wd.tick(raw) is None
    wire(raw, 3, clk.t, False)                           # the resend: its reply takes 1,250 s
    for _ in range(21):
        clk.t += 60
        c['model_sec'] += 60
        assert wd.tick(raw) is None
    d = raw / '.pi-cont16k' / 'k-abc' / 'wire-responses'
    (d / '000003.response').write_text('x')              # 3,060 s after the first unanswered request: cleared
    assert wd.wire.poll(raw) is None
    wire(raw, 4, clk.t, False)                           # a backend error: meta, no reply
    wire(raw, 5, clk.t, False)
    wire(raw, 6, clk.t, False)
    (d / '000006.response').write_text('x')              # a later index answered first (out of order): clears 4 and 5
    assert wd.wire.poll(raw) is None
    clk.t += 5000
    assert wd.tick(raw) is None


def test_a_long_answered_request_and_gate_waits_are_not_stalls(tmp_path):
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    c = {'in_flight': 1, 'requests': 1, 'model_sec': 0.0}
    wd = hung.Watchdog(600, 1860, read=lambda api: dict(c), clock=clk)
    wire(raw, 1, clk.t, False)
    for _ in range(25):                                  # 1,500 s before its response starts (Snowball at 16k: 1,243 s)
        clk.t += 60
        c['model_sec'] += 60
        assert wd.tick(raw) is None
    (raw / '.pi-cont16k' / 'k-abc' / 'wire-responses' / '000001.response').write_text('x')
    c['requests'] = 2                                    # the next turn waits at the turn gate: in flight, no meta yet
    for _ in range(40):
        clk.t += 60
        c['model_sec'] += 60
        assert wd.tick(raw) is None
    wire(raw, 2, clk.t, False)                           # forwarded after 40 min at the gate: the stall clock starts now
    clk.t += 1800
    assert wd.tick(raw) is None
    assert wd.tick(raw, ok=False) is None                # an outage: stall time before it does not count
    clk.t += 1800
    c['model_sec'] += 1800
    assert wd.tick(raw) is None
    clk.t += 61
    assert wd.tick(raw)[0] == 'stalled_request'
    assert hung.Watchdog(600, None, read=lambda api: dict(c), clock=clk).tick(raw) is None   # stall rule off


def test_guard_turns_are_progress_for_the_stall_rule(tmp_path):
    # the real slow-session pattern (m5l-sf-d1024-full-r2-tbsci_lite specimen-swap-contamination__a1__r0): requests 8
    # and 9 dropped by Pi at ~1800 s each, the guard still finishing turns at +1,947 s and +3,757 s (it serves requests
    # in order), request 10 answered at +5,388 s: not a stall
    raw = tmp_path / 'raw'
    proxy_dir(raw)
    clk = Clock()
    c = {'in_flight': 1, 'requests': 8, 'model_sec': 0.0}
    wd = hung.Watchdog(600, 3660, read=lambda api: dict(c), clock=clk)
    log = raw / '.pi-cont16k' / 'k-abc' / 'response-budget-wire.jsonl'
    big = 'x' * 2_000_000                                # guard lines carry whole upstream records (~1.4 MB)

    def turn(t):
        with open(log, 'a') as f:
            f.write(json.dumps({'time': t, 'policy': 'p', 'accounting_policy': 'a', 'turn': 1, 'upstream': big}) + '\n')
    for i in range(1, 8):
        wire(raw, i, clk.t, True)
    t0 = clk.t
    wire(raw, 8, t0, False)
    for minute in range(1, 91):
        clk.t = t0 + 60 * minute
        c['model_sec'] += 60
        if minute == 30:
            wire(raw, 9, clk.t, False)                    # Pi's first resend
        if minute == 33:
            turn(t0 + 1947)                               # the guard finishes the dropped request 8
        if minute == 60:
            wire(raw, 10, clk.t, False)                   # Pi's second resend
        if minute == 63:
            turn(t0 + 3757)                               # ... and request 9
        assert wd.tick(raw) is None, minute
    (raw / '.pi-cont16k' / 'k-abc' / 'wire-responses' / '000010.response').write_text('x')   # +5,388 s: answered
    assert wd.tick(raw) is None and wd.budget.progress_at == t0 + 3757 and not wd.budget.failed
    # without guard progress, the same timeline is a stall (the rule's backstop)
    wd2 = hung.Watchdog(600, 3660, read=lambda api: dict(c), clock=clk)
    wd2.budget.poll = lambda raw: False
    (raw / '.pi-cont16k' / 'k-abc' / 'wire-responses' / '000010.response').unlink()
    assert wd2.tick(raw)[0] == 'stalled_request'


def test_run_py_records_the_watchdog_in_new_runs_only(tmp_path, monkeypatch, capsys):
    import io
    import tarfile
    from aa_evals.common.pi_eval import run
    from aa_evals.common.pi_eval.test_model_input import ep
    actual = ep(tmp_path, [['server']])
    actual.update(served='opaque', urls=['http://fake.invalid/v1'], request={'overrides': {'temperature': 1}})
    monkeypatch.setattr(run.endpoint, 'alive', lambda *a: (True, actual, 'fake'))
    monkeypatch.setattr(run.endpoint, 'max_model_len', lambda *a: 131072)
    monkeypatch.setattr(run, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(run, 'snapshot_code', lambda *a: str(Path(run.__file__).resolve().parents[3]))
    monkeypatch.setattr(run, 'submit_workers', lambda *a: pytest.fail('submission'))
    monkeypatch.setattr(run.runtime_cache, 'load_manifest', lambda *a: {})
    bundle = tmp_path / 'bundle.tar'
    content = b'version = "1.0"\n[agent]\ntimeout_sec=21600\n'
    with tarfile.open(bundle, 'w') as tf:
        m = tarfile.TarInfo('tasks/fake/task.toml')
        m.size = len(content)
        tf.addfile(m, io.BytesIO(content))
    base = ['--bench', 'tb21', '--model', 'serving:fake', '--seed-namespace', 'x', '--bundle', str(bundle),
            '--no-submit']
    run.main([*base, '--run-id', 'new'])
    cfg = json.loads((tmp_path / 'runs/tb21/new/config.json').read_text())
    assert cfg['hung_watchdog']['idle_sec'] == 600 and cfg['hung_watchdog']['stall_sec'] == 3660
    run.main([*base, '--run-id', 'off', '--hung-watchdog-sec', '0'])
    assert json.loads((tmp_path / 'runs/tb21/off/config.json').read_text())['hung_watchdog'] is None
    run.main([*base, '--run-id', 'new', '--hung-watchdog-sec', '0'])      # an existing run keeps its stored value
    assert json.loads((tmp_path / 'runs/tb21/new/config.json').read_text())['hung_watchdog'] == cfg['hung_watchdog']
    assert '--hung-watchdog-sec / --hung-stall-sec ignored' in capsys.readouterr().err


def test_resolve(tmp_path):
    t = tmp_path / 'gaps.json'
    t.write_text(json.dumps({'generated_at': 'T', 'suites': {'gdpval_pi': {'p99_gap_sec': 40.0},
                                                             'tb21': {'p99_gap_sec': 400.0}}}))
    assert hung.resolve('gdpval_pi', table=t)['idle_sec'] == 600           # floor: 10 min
    assert hung.resolve('tb21', table=t)['idle_sec'] == 1200               # 3 x p99
    unknown = hung.resolve('newbench', table=t)
    assert unknown['idle_sec'] == 1200 and 'largest' in unknown['source']  # the most lenient known threshold
    assert hung.resolve('tb21', 0, table=t) is None                        # --hung-watchdog-sec 0: off
    assert hung.resolve('tb21', 900, table=t)['idle_sec'] == 900
    assert hung.resolve('tb21', table=t)['stall_sec'] == hung.STALL_SEC          # no stall_sec in the table
    t.write_text(json.dumps({'generated_at': 'T', 'stall_sec': 2000, 'suites': {}}))
    assert hung.resolve('tb21', table=t)['stall_sec'] == 2000
    assert hung.resolve('tb21', table=t, stall_override_sec=0)['stall_sec'] == 0  # --hung-stall-sec 0: stall rule off
    assert request_gaps.threshold(None) == 600


def test_the_shipped_table_covers_the_pi_suites():
    t = json.loads(hung.TABLE.read_text())
    for suite in ('tb21', 'tbsci_lite', 'gdpval_pi', 'automationbench_pi', 'critpt_pi'):
        rec = hung.resolve(suite)
        assert rec['p99_gap_sec'] is not None and rec['idle_sec'] >= 600, suite
    assert t['rule'].startswith('idle_sec = max(3 x p99 gap, 600 s)')
    assert t['stall_sec'] > 1800                          # above Pi's own request timeout


# ---------------------------------------------------------------- run_item with a fake Pi session (end to end)

class ClockServer:
    """A Pi proxy's /proxy/model-clock."""

    def __init__(self, state):
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(outer.state).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.state = state
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.api = f'http://127.0.0.1:{self.srv.server_address[1]}/v1'
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class FakeState:
    def __init__(self, q):
        self.q, self.finished = q, {}

    def heartbeat(self, k):
        pass

    def finish(self, k, outcome):
        self.finished[k] = outcome


HARBOR = """#!/bin/bash
# fake gen/run_harbor.sh: a Pi proxy log dir with its address, a session log, then (optionally) the agent's end, then
# a wait
raw="$2"
mkdir -p "$raw/.pi-cont16k/$3-x" "$raw/job/trial/agent/pi/sessions"
echo "{\\"proxy_api\\": \\"$CLOCK_API\\"}" > "$raw/.pi-cont16k/$3-x/protocol.json"
printf '%s\\n' "$SESSION" > "$raw/job/trial/agent/pi/sessions/s.jsonl"
if [ -n "$STALLED" ]; then
  mkdir -p "$raw/.pi-cont16k/$3-x/wire-responses"
  echo "{\\"started\\": $(date +%s), \\"agent_request\\": true}" > "$raw/.pi-cont16k/$3-x/wire-responses/000001.meta.json"
fi
if [ -n "$ACCOUNTING" ]; then echo '{"time": 1, "accounting_error": "TimeoutError: ", "upstream": {}}' > "$raw/.pi-cont16k/$3-x/response-budget-wire.jsonl"; fi
if [ -n "$AGENT_ENDS" ]; then mkdir -p "$raw/job/trial/agent" && echo '{}' > "$raw/job/trial/agent/env_clock.json"; fi
exec sleep "$SLEEP"
"""


DONE = '\n'.join(json.dumps(e) for e in (tool_call('a'), tool_result('a')))    # a tool returned, no request since
RUNNING = json.dumps(tool_call('a'))                                             # a tool still running


def run_attempt(tmp_path, monkeypatch, clock, sleep, idle_sec=1.0, ok=True, agent_ends=False, config_extra=None,
                session=DONE, stall_sec=None, stalled=False, accounting=False):
    srv = ClockServer(clock)
    repo = tmp_path / 'repo'
    (repo / 'gen').mkdir(parents=True)
    (repo / 'gen' / 'run_harbor.sh').write_text(HARBOR)
    run = tmp_path / 'run'
    (run / 'queue' / 'done').mkdir(parents=True)
    w = W.PiEvalWorker.__new__(W.PiEvalWorker)
    w.config = {'model': 'm', 'served': 's', 'agent': 'pi', 'harness': 'pi', 'benchmark': 'gdpval_pi',
                'hung_watchdog': {'idle_sec': idle_sec, 'stall_sec': stall_sec}, **(config_extra or {})}
    w.name, w.run, w.tasks_dir, w.urls = 'w1', run, tmp_path / 'tasks', ['http://127.0.0.1:1/v1']
    w.state = FakeState(run / 'queue')
    w.running = {'t1__a0__r0': {}}
    w.ok, w.proxy_state = ok, (time.time(), True)
    env = {'PATH': os.environ['PATH'], 'CLOCK_API': srv.api, 'SLEEP': str(sleep),
           'AGENT_ENDS': '1' if agent_ends else '', 'SESSION': session, 'STALLED': '1' if stalled else '',
           'ACCOUNTING': '1' if accounting else ''}
    monkeypatch.setattr(W.q35, 'SCRATCH', tmp_path / 'scratch')
    monkeypatch.setattr(W.q35, 'REPO', repo)
    monkeypatch.setattr(W.q35, 'STOP', False)
    monkeypatch.setattr(W.PiEvalWorker, 'POLL_SEC', 0.2)
    monkeypatch.setattr(W.PiEvalWorker, 'env', lambda self, k, task, attempt: dict(env))
    monkeypatch.setattr(W.PiEvalWorker, 'url_for', lambda self, k: 'http://127.0.0.1:1/v1')
    monkeypatch.setattr(W.PiEvalWorker, 'endpoint_gone', lambda self, endpoint: False)
    monkeypatch.setattr(W.PiEvalWorker, 'pack', lambda self, k, raw, root, outcome: {})
    t0 = time.time()
    try:
        asyncio.run(w.run_item('t1__a0__r0'))
    finally:
        srv.close()
    info = json.loads((run / 'attempts' / 't1__a0__r0' / 'attempt.json').read_text())
    return w.state.finished['t1__a0__r0'], time.time() - t0, info


def test_an_idle_pi_ends_as_an_infra_retry(tmp_path, monkeypatch):
    # requests made, none in flight, no tool running, nothing new: Pi stopped
    out, wall, info = run_attempt(tmp_path, monkeypatch, {'in_flight': 0, 'requests': 7, 'model_sec': 42.0}, 600)
    assert out['infrastructure_failure'] and out['infrastructure_reason'] == 'hung_session'
    assert out['exception_type'] == 'HungSession' and out['score'] is None
    assert out['hung_kind'] == 'idle' and out['hung_sec'] >= 1.0 and out['hung_limit_sec'] == 1.0
    assert wall < 60                                       # ended by the watchdog, not by the 600 s session
    assert info['hung_watchdog_sec'] == 1.0                # recorded per attempt


def test_a_stalled_request_ends_as_an_infra_retry(tmp_path, monkeypatch):
    # the 11-15 signature end to end: in flight, a forwarded request whose response never starts
    out, wall, info = run_attempt(tmp_path, monkeypatch, {'in_flight': 1, 'requests': 7, 'model_sec': 42.0}, 600,
                                  idle_sec=600, stall_sec=1.0, stalled=True)
    assert out['infrastructure_reason'] == 'hung_session' and out['hung_kind'] == 'stalled_request'
    assert out['hung_limit_sec'] == 1.0 and info['hung_stall_sec'] == 1.0 and wall < 60


def test_a_kill_that_leaves_the_process_alive_does_not_crash_the_attempt(tmp_path, monkeypatch):
    # kill() can return with the process alive; the next tick (agent ended -> None) must not lose the fired value
    async def slow_kill(proc):
        if not getattr(slow_kill, 'once', False):
            slow_kill.once = True
            agent = next((tmp_path / 'scratch' / 'raw').glob('*')) / 'job' / 'trial' / 'agent'
            (agent / 'env_clock.json').write_text('{}')
            return
        proc.kill()
        await proc.wait()
    monkeypatch.setattr(W.PiEvalWorker, 'kill', staticmethod(slow_kill))
    out, _, _ = run_attempt(tmp_path, monkeypatch, {'in_flight': 0, 'requests': 7, 'model_sec': 42.0}, 3)
    assert out['infrastructure_reason'] == 'hung_session' and out['hung_kind'] == 'idle'


def test_a_session_in_a_request_or_at_a_gate_is_not_hung(tmp_path, monkeypatch):
    out, wall, _ = run_attempt(tmp_path, monkeypatch, {'in_flight': 1, 'requests': 7, 'model_sec': 42.0}, 3)
    assert out.get('infrastructure_reason') != 'hung_session' and wall >= 3


def test_a_session_in_a_long_tool_call_is_not_hung(tmp_path, monkeypatch):
    out, wall, _ = run_attempt(tmp_path, monkeypatch, {'in_flight': 0, 'requests': 7, 'model_sec': 42.0}, 3,
                               session=RUNNING)
    assert out.get('infrastructure_reason') != 'hung_session' and wall >= 3


def test_the_verifier_phase_and_outages_are_not_judged(tmp_path, monkeypatch):
    idle = {'in_flight': 0, 'requests': 7, 'model_sec': 42.0}
    out, _, _ = run_attempt(tmp_path / 'a', monkeypatch, dict(idle), 3, agent_ends=True)
    assert out.get('infrastructure_reason') != 'hung_session'
    out, _, _ = run_attempt(tmp_path / 'b', monkeypatch, dict(idle), 3, ok=False)
    assert out.get('infrastructure_reason') != 'hung_session'


def test_runs_without_the_key_and_the_oracle_run_without_it(tmp_path, monkeypatch):
    idle = {'in_flight': 0, 'requests': 7, 'model_sec': 42.0}
    out, _, info = run_attempt(tmp_path / 'a', monkeypatch, dict(idle), 3, config_extra={'hung_watchdog': None})
    assert out.get('infrastructure_reason') != 'hung_session' and info['hung_watchdog_sec'] is None
    out, _, info = run_attempt(tmp_path / 'b', monkeypatch, dict(idle), 3, config_extra={'agent': 'oracle'})
    assert out.get('infrastructure_reason') != 'hung_session' and info['hung_watchdog_sec'] is None


@pytest.mark.parametrize('sec', [0])
def test_run_py_records_the_watchdog(sec):
    from aa_evals.common.pi_eval import run
    ap = run.build_parser({})
    assert ap.parse_args(['--bench', 'tb21', '--run-id', 'x']).hung_watchdog_sec is None   # default: the table
    assert ap.parse_args(['--bench', 'tb21', '--run-id', 'x', '--hung-watchdog-sec', str(sec)]).hung_watchdog_sec == 0
    assert ap.parse_args(['--bench', 'tb21', '--run-id', 'x', '--hung-stall-sec', '2000']).hung_stall_sec == 2000


def test_a_guard_accounting_failure_ends_as_response_budget_accounting(tmp_path, monkeypatch):
    """#549: the watchdog's accounting rule records the same reason classify() gives when Pi reports the guard's error
    itself (response_budget_accounting), so one failure has one name whichever path ends the attempt."""
    out, wall, info = run_attempt(tmp_path, monkeypatch, {'in_flight': 1, 'requests': 3, 'model_sec': 9.0}, 600,
                                  accounting=True)
    assert out['infrastructure_failure'] and out['infrastructure_reason'] == 'response_budget_accounting'
    assert out['hung_kind'] == 'budget_accounting_failure' and out['exception_type'] == 'ResponseBudgetAccounting'
    assert out['score'] is None and wall < 60

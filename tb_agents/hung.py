"""Hung-session watchdog for Pi attempts (#531 A5; the rule of tbsched's design, #451 workstream 1).

Shared by both pipelines that run Pi through gen/pi_cont16k.py (one rule, #531): aa_evals pi_eval
(aa_evals/common/pi_eval/worker.py; aa_evals/common/pi_eval/hung.py re-exports this module) and task synthesis
(synth/worker.py, campaign config harbor.hung_watchdog). It reads only the attempt's raw dir as gen/pi_cont16k.py and
Harbor write it. In synthesis only the idle rule can fire: its proxy profile (PI_PROXY_CAPTURE_FILES=0) writes no
wire-responses/ (no stall rule), and its campaigns run no response-budget guard (no accounting rule; the 11-15 cause
does not arise there). The agent's end is env_clock.json (Pi's env-clock agent, pi_eval) or trajectory.json (Harbor
writes it as the agent stops, before the verifier: synthesis runs the plain cached agent).

M5, 11-15: 362 GDPval and 53 AutomationBench Pi sessions (synth second traces, m5l_*_t2) sat until the 2 h wall cap.
Healthy GDPval sessions take at most 33 min. Each held a worker slot and the suite's drain for 2 h. The hangs were
random, not task-specific: 273 of 358 recovered on retry. Their cause: the response-budget guard
(tb_agents/response_budget_proxy.py) failed to account a reply (most likely its 30 s /detokenize call timed out), logged
`accounting_error`, and from then on answered every request with a sticky 502. The sticky proxy retries a 502 for
--hold (1800 s) as a backend restart, so Pi never saw the error: it timed out after 1800 s and resent, every 1800 s,
until the cap. The proxy had a request in flight the whole time (model_sec ~7,180 of 7,200 s), and the session ended
with "Request timed out."

The worker ends an attempt as infrastructure failure on any of three rules: `hung_session` for the two timed rules,
`response_budget_accounting` for the guard's accounting failure (the reason classify() also gives when Pi reports the
guard's error itself, which it does within seconds since #549; the watchdog then rarely gets there first). That is a same-seed retry,
capped at eval_q35's INFRA_RETRIES; failures in an outage window are exempt from the cap (queue_state.finish).
- budget accounting failure: the guard logged `accounting_error` (<logs>/response-budget-wire.jsonl). Its failure is
  sticky, so the session cannot recover: ended at once.
- stalled request (stall_sec, 3660 s: two of Pi's 1800 s timeouts and a minute): no progress since the oldest
  unanswered request for longer than stall_sec. Progress is a reply arriving or the guard completing a turn. The guard
  serves requests one at a time and keeps generating a request Pi gave up on, so a resend waits behind it; a healthy
  slow session can lose two Pi timeouts and still recover (TB-Sci-lite, a small model: answered after 5,388 s, the
  guard logging turns at +1,947 and +3,757 s). The proxy writes <logs>/wire-responses/NNNNNN.meta.json when it
  forwards a request, after its turn and KV gates; NNNNNN.response appears when the reply arrives (the guard buffers
  whole replies). Time a request then waits at a server-side admission gate (aa_evals/qwenserve) counts. Without the
  guard (no PI_RESPONSE_BUDGET), only replies are progress and the accounting rule never fires.
- idle (idle_sec): Pi itself has done nothing for longer than idle_sec, as defined below.
- idle_sec = max(3 x the suite's p99 idle gap, 10 min) and stall_sec are in request_gaps.json (measured by
  request_gaps.py), set when the run is built and recorded in its config (`hung_watchdog`).
- Idle means Pi itself is doing nothing: no model request in flight and no tool call outstanding.
  - Time inside a request is activity, including a wait at the turn gate or the KV gate (both inside the proxy's
    request handler). The source is the attempt's Pi proxy: /proxy/model-clock gives in_flight, requests and
    model_sec (gen/pi_cont16k.py records its address in protocol.json).
  - Time inside a tool call is activity too. Pi's session log (<trial>/agent/pi/sessions/*.jsonl, appended as it
    goes) shows each toolCall id until its toolResult arrives. The 11-15 hangs had no tool running. A tool that never
    returns is the model's doing and is left to the environment clock. Measured (request_gaps.py, 3,485 attempts of
    15 M5 runs): with tool time counted as idle, a 10-min limit would have ended 19 healthy attempts (13 of 808
    TB2.1, 4 CritPt, 2 TB-Sci-lite: long training and build commands, the same tasks every time, so a scoring bias);
    with tool time excluded, none that a gate wait does not explain (1 TB-Sci-lite attempt cannot be checked).
- The idle rule is armed after the session's first request, so environment build and agent setup are not judged. Both
  rules are disarmed once the agent has ended (agent/env_clock.json or agent/trajectory.json exists), so a long
  verifier is never cut.
- Never on missing evidence: an unreadable clock, no proxy yet, or no session log keeps the idle rule from firing;
  no wire-responses directory keeps the stall rule from firing. Outage time the worker sees (endpoint down, registry
  stale, cluster proxy failing) counts toward neither timed rule; the cluster-proxy signal (eval_q35 proxy_ok)
  refreshes only when the worker has a free slot, the endpoint and registry signals every 30 s. The idle rule never fires after the agent's final answer
  (its last message stopped with `stop`): Pi exits on its own, else Harbor's agent timeout ends the agent and the
  verifier still scores the work.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

# the idle limit rule (aa_evals/common/pi_eval/request_gaps.py measures the gaps and imports these)
FLOOR_SEC = 600.0     # the watchdog never fires sooner than this
P99_MULT = 3.0        # threshold = max(P99_MULT x suite p99 gap, FLOOR_SEC)


def threshold(p99_gap_sec: float | None) -> float:
    """The watchdog's idle limit for a suite whose p99 gap is p99_gap_sec (None: the floor)."""
    return max(P99_MULT * float(p99_gap_sec or 0.0), FLOOR_SEC)


TABLE = Path(__file__).resolve().parents[1] / 'aa_evals' / 'common' / 'pi_eval' / 'request_gaps.json'
STALL_SEC = 3660.0    # fallback when the table has none: two of Pi's 1800 s request timeouts, plus a minute
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def resolve(benchmark: str, override_sec: float | None = None, table: Path = TABLE,
            stall_override_sec: float | None = None) -> dict | None:
    """The `hung_watchdog` config entry of a new run: {'idle_sec', 'stall_sec', 'p99_gap_sec', 'source'}, or None (off)
    for override_sec 0. A suite without a measured gap gets the largest idle threshold of the table (the most
    lenient). stall_sec is the table's (one value: the stall is Pi's request timeout, not a suite property)."""
    if override_sec is not None and override_sec <= 0:
        return None
    t = json.loads(Path(table).read_text())
    stall = float(stall_override_sec if stall_override_sec is not None else t.get('stall_sec') or STALL_SEC)
    if override_sec is not None:
        return {'idle_sec': float(override_sec), 'stall_sec': stall, 'p99_gap_sec': None,
                'source': 'run.py --hung-watchdog-sec'}
    suites = t.get('suites') or {}
    rec = suites.get(benchmark)
    if rec and rec.get('p99_gap_sec') is not None:
        return {'idle_sec': threshold(rec['p99_gap_sec']), 'stall_sec': stall, 'p99_gap_sec': rec['p99_gap_sec'],
                'source': f"request_gaps.json {t.get('generated_at')} ({benchmark})"}
    worst = max((threshold(r.get('p99_gap_sec')) for r in suites.values()), default=threshold(None))
    return {'idle_sec': worst, 'stall_sec': stall, 'p99_gap_sec': None,
            'source': f"request_gaps.json {t.get('generated_at')} (no entry for {benchmark}: the largest threshold)"}


def proxy_api(raw: Path) -> str | None:
    """The attempt's Pi proxy address (gen/pi_cont16k.py: <jobs>/.pi-cont16k/<name>-*/protocol.json proxy_api)."""
    try:
        files = sorted(Path(raw).glob('.pi-cont16k/*/protocol.json'), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    for p in reversed(files):
        try:
            api = json.loads(p.read_text()).get('proxy_api')
        except (OSError, ValueError):
            continue
        if api:
            return api
    return None


def read_clock(api: str, timeout: float = 5.0) -> dict | None:
    """The proxy's model clock ({in_flight, requests, model_sec, ...}), or None when it does not answer."""
    base = api.rstrip('/').removesuffix('/v1')
    try:
        with OPENER.open(base + '/proxy/model-clock', timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def agent_ended(raw: Path) -> bool:
    """The agent has stopped (the verifier may follow): Pi's env-clock wrapper writes <trial>/agent/env_clock.json
    (pi_eval's agent); Harbor writes <trial>/agent/trajectory.json when it syncs the agent's output, before the verifier
    starts, for every agent (synthesis runs PiCont16kCached, which writes no env clock). In pi_eval trajectory.json
    never comes before env_clock.json, so this changes nothing there."""
    try:
        return any(Path(raw).glob('*/*/agent/env_clock.json')) or any(Path(raw).glob('*/*/agent/trajectory.json'))
    except OSError:
        return False


class SessionLog:
    """Pi's session log, read incrementally: whether it grew, whether a tool call is outstanding, and whether the
    agent gave its final answer."""

    def __init__(self):
        self.path, self.offset, self.partial = None, 0, b''
        self.outstanding: list = []
        self.final = False

    def poll(self, raw: Path) -> tuple[bool, bool]:
        """(found, grew) for this tick."""
        try:
            files = sorted(Path(raw).glob('*/*/agent/pi/sessions/*.jsonl'), key=lambda p: p.stat().st_mtime)
        except OSError:
            return self.path is not None, False
        if not files:
            return False, False
        path = files[-1]
        if path != self.path:   # a new session file (first, or Pi started another): read it from the start
            self.path, self.offset, self.partial, self.outstanding, self.final = path, 0, b'', [], False
        try:
            size = path.stat().st_size
            if size < self.offset:
                self.offset, self.partial, self.outstanding, self.final = 0, b'', [], False
            if size == self.offset:
                return True, False
            with open(path, 'rb') as f:
                f.seek(self.offset)
                chunk = f.read(size - self.offset)
        except OSError:
            return True, False
        self.offset += len(chunk)
        lines = (self.partial + chunk).split(b'\n')
        self.partial = lines.pop()
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = e.get('message') if isinstance(e, dict) and e.get('type') == 'message' else None
            if not isinstance(m, dict):
                continue
            if m.get('role') == 'assistant':
                c = m.get('content') if isinstance(m.get('content'), list) else []
                # a list, not a set: duplicate or empty ids in a parallel call must not hide a running tool
                self.outstanding = [x.get('id') for x in c if isinstance(x, dict) and x.get('type') == 'toolCall']
                self.final = m.get('stopReason') == 'stop' and not self.outstanding
            elif m.get('role') == 'toolResult' and self.outstanding:
                i = m.get('toolCallId')
                self.outstanding.remove(i if i in self.outstanding else self.outstanding[0])
        return True, True


HEAD = 4096   # guard log lines run ~1.4 MB (full upstream records); the keys read here come first


class BudgetLog:
    """The response-budget guard's log (<logs>/response-budget-wire.jsonl), read incrementally: whether it logged an
    accounting error (after which it answers every request with a sticky 502), and the time of its last other record
    (a turn it completed: progress, even on a request Pi has given up on). Only each line's head is parsed."""

    TIME = re.compile(rb'^\{"time": ([0-9.]+)')

    def __init__(self):
        self.path, self.offset, self.failed = None, 0, False
        self.partial = b''          # the head of an incomplete last line
        self.partial_done = False   # its head is complete (the rest of that line is skipped)
        self.progress_at = None

    def poll(self, raw: Path) -> bool:
        if self.failed:
            return True
        try:
            files = sorted(Path(raw).glob('.pi-cont16k/*/response-budget-wire.jsonl'), key=lambda p: p.stat().st_mtime)
            if not files:
                return False
            path = files[-1]
            if path != self.path:
                self.path, self.offset, self.partial, self.partial_done = path, 0, b'', False
            size = path.stat().st_size
            if size <= self.offset:
                return False
            with open(path, 'rb') as f:
                f.seek(self.offset)
                chunk = f.read(size - self.offset)
        except OSError:
            return False
        self.offset += len(chunk)
        start = 0
        while True:
            nl = chunk.find(b'\n', start)
            piece = chunk[start:] if nl < 0 else chunk[start:nl]
            if not self.partial_done:
                self.partial += piece[:HEAD - len(self.partial)]
                self.partial_done = len(self.partial) >= HEAD
            if nl < 0:
                break
            self.line(self.partial)
            self.partial, self.partial_done, start = b'', False, nl + 1
        return self.failed

    def line(self, head: bytes) -> None:
        if not head.strip():
            return
        if b'"accounting_error":' in head:
            self.failed = True
            return
        m = self.TIME.match(head)
        if m:
            t = float(m.group(1))
            self.progress_at = t if self.progress_at is None else max(self.progress_at, t)


class WireLog:
    """The proxy's wire-responses directory: the start of the current unanswered streak (the oldest forwarded request
    after the last one whose reply arrived), or None."""

    def __init__(self):
        self.dir = None
        self.started: dict[str, float] = {}

    def poll(self, raw: Path) -> float | None:
        try:
            dirs = sorted(Path(raw).glob('.pi-cont16k/*/wire-responses'), key=lambda p: p.stat().st_mtime)
            if not dirs:
                return None
            if dirs[-1] != self.dir:
                self.dir, self.started = dirs[-1], {}
            names = os.listdir(dirs[-1])
        except OSError:
            return None
        metas = sorted(n[:-len('.meta.json')] for n in names if n.endswith('.meta.json'))
        answered = {n[:-len('.response')] for n in names if n.endswith('.response')}
        last = max(answered) if answered else ''          # zero-padded indices: string order is request order
        for i in metas:
            if i in answered or i <= last:
                continue
            if i not in self.started:
                try:
                    self.started[i] = float(json.loads((dirs[-1] / f'{i}.meta.json').read_text())['started'])
                except (OSError, ValueError, KeyError, TypeError):
                    return None                           # being written: next tick
            return self.started[i]
        return None


class Watchdog:
    """One attempt's liveness check; tick() once per worker poll (eval_q35's 60 s heartbeat loop)."""

    def __init__(self, idle_sec: float, stall_sec: float | None = STALL_SEC, read=read_clock, clock=time.time):
        self.idle_sec, self.read, self.clock = float(idle_sec), read, clock
        self.stall_sec = float(stall_sec) if stall_sec else None
        self.session = SessionLog()
        self.wire = WireLog()
        self.budget = BudgetLog()
        self.api = None
        self.armed = False
        self.ended = False
        self.last_active = None
        self.outage_at = None     # the last tick that saw an outage: stall time before it does not count
        self.seen = None          # (requests, model_sec) at the last tick
        self.longest = 0.0        # longest idle stretch seen (for the outcome)

    def tick(self, raw: Path, ok: bool = True) -> tuple[str, float] | None:
        """('budget_accounting_failure' | 'stalled_request' | 'idle', seconds) when the session is hung, else None."""
        now = self.clock()
        if self.last_active is None:
            self.last_active = now
        if self.ended or agent_ended(raw):
            self.ended = True
            return None
        if self.budget.poll(raw):
            return 'budget_accounting_failure', 0.0
        if not ok:   # an outage the worker sees: counts toward neither timed rule
            self.last_active = self.outage_at = now
            return None
        if self.stall_sec:
            since = self.wire.poll(raw)
            if since is not None:
                # the guard completing a turn (even one Pi gave up on) is progress: it serves requests in order, so a
                # resend waits behind the dropped one's generation and a slow session can lose two Pi timeouts
                floor = max(x for x in (since, self.outage_at, self.budget.progress_at) if x is not None)
                age = now - floor
                if age > self.stall_sec:
                    return 'stalled_request', age
        if self.api is None:
            self.api = proxy_api(raw)
        c = self.read(self.api) if self.api else None
        if not isinstance(c, dict):
            self.last_active = now   # no evidence: never counted as idle
            return None
        try:
            requests, in_flight = int(c.get('requests') or 0), int(c.get('in_flight') or 0)
            model_sec = float(c.get('model_sec') or 0.0)
        except (TypeError, ValueError):
            self.last_active = now
            return None
        found, grew = self.session.poll(raw)
        if not found or self.session.final:   # no tool evidence, or the agent already answered: never judged
            self.last_active = now
            return None
        seen = (requests, model_sec)
        if requests == 0 or in_flight > 0 or seen != self.seen or grew or self.session.outstanding:
            self.armed = self.armed or requests > 0
            self.last_active, self.seen = now, seen
            return None
        idle = now - self.last_active
        self.longest = max(self.longest, idle)
        return ('idle', idle) if self.armed and idle > self.idle_sec else None

"""Per-suite idle gaps between a Pi session's model requests (#531 A5): the hung-session watchdog's threshold table.

  python -m aa_evals.common.pi_eval.request_gaps RUN [RUN ...] --out aa_evals/common/pi_eval/request_gaps.json

Read from each attempt's trial.tar.gz, without extracting it:
- the agent's model requests: pi-cont16k/wire-responses/NNNNNN.meta.json gives each request's `started` time (agent
  requests only); the matching .response file's mtime is when the response ended. A request with no response file
  never completed for Pi (its client gave up and retried); it counts as in flight until the next request;
- Pi's session log (agent/pi/sessions/*.jsonl): the times of its messages and tool results;
- the agent's end, agent/env_clock.json: its mtime, env_sec (time with no request in flight) and kv_gate_held_sec.

A gap is time in which Pi itself is idle, the watchdog's measure (hung.py): no model request in flight and no tool
call outstanding. It runs from a response's end, or the last session entry after it (a tool result), to the next
request; the tail runs from the last response or entry to the agent's end.

The wire logs are written after the turn gate and the KV gate let a request through, so a turn parked at a gate
reads as a gap here, while the live watchdog counts it as in flight. Gaps are therefore exact only in attempts with
no gate wait (kv_gate_held_sec 0), and the suite's p99 comes from those attempts. Per suite, this reports that p99,
the watchdog's threshold max(3 x p99, 600 s), and the attempts it would have ended: their longest stretch (bounded by
env_sec) is above the threshold and longer than all of their gate waits. Stretches a gate wait can explain are
counted separately. The same gaps with tool time counted as idle (`with_tool_time`) show why tool calls are exempt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tb_agents.hung import FLOOR_SEC, P99_MULT, threshold  # noqa: F401  (the rule lives with the watchdog)

GATE_SLACK_SEC = 5.0  # a stretch at most this much longer than the attempt's gate waits can be one (request set-up)
# The stall rule (hung.py): no progress (a reply arriving, or the response-budget guard completing a turn) since the
# oldest unanswered request, for this long: two of Pi's 1800 s request timeouts and a minute. As live, a streak ends
# when the clearing reply arrives (the guard buffers whole replies) and guard turns inside it restart its clock.
STALL_SEC = 3660.0
BUDGET = re.compile(r'/pi-cont16k/response-budget-wire\.jsonl$')
GUARD_TIME = re.compile(rb'^\{"time": ([0-9.]+)')
META = re.compile(r'(?:^|/)pi-cont16k/wire-responses/(\d+)\.meta\.json$')
SESSION = re.compile(r'/agent/pi/sessions/[^/]+\.jsonl$')


def session_events(lines) -> list[float]:
    """Times of a Pi session's tool results and messages (Pi appends each entry as it happens)."""
    out = []
    for line in lines:
        try:
            e = json.loads(line)
            t = dt.datetime.fromisoformat(str(e.get('timestamp')).replace('Z', '+00:00')).timestamp()
        except (ValueError, TypeError):
            continue
        if e.get('type') == 'message':
            out.append(t)
    return sorted(out)
RESP = re.compile(r'(?:^|/)pi-cont16k/wire-responses/(\d+)\.response$')




def busy_gaps(intervals: list[tuple[float, float]]) -> tuple[list[float], float | None]:
    """Gaps between merged busy stretches of (start, end) request intervals, and the last stretch's end."""
    gaps, end = [], None
    for s, e in sorted(intervals):
        e = max(e, s)
        if end is None:
            end = e
        elif s > end:
            gaps.append(s - end)
            end = e
        else:
            end = max(end, e)
    return gaps, end


def attempt(path: str) -> dict | None:
    """{'key', 'gaps', 'tail', 'n_requests'} of one attempt archive, or None when it has no agent requests."""
    started, ended, agent_end, clock, events, accounting_error, guard = {}, {}, None, {}, [], None, []
    try:
        with tarfile.open(path, 'r:gz') as tf:
            for m in tf:
                name = m.name
                if name.endswith('/agent/env_clock.json'):
                    agent_end = float(m.mtime)
                    f = tf.extractfile(m)
                    try:
                        clock = json.loads(f.read()) if f else {}
                    except ValueError:
                        clock = {}
                    continue
                r = RESP.search(name)
                if r:
                    ended[r.group(1)] = float(m.mtime)
                    continue
                if BUDGET.search(name) and m.isfile():
                    f = tf.extractfile(m)
                    for line in (f.read().splitlines() if f else []):
                        head = line[:4096]              # as hung.BudgetLog: the keys read come first
                        t = GUARD_TIME.match(head)
                        if b'"accounting_error":' in head:
                            if accounting_error is None and t:
                                accounting_error = float(t.group(1))
                        elif t:
                            guard.append(float(t.group(1)))   # a turn the guard completed: progress
                    continue
                if SESSION.search(name) and m.isfile():
                    f = tf.extractfile(m)
                    if f:
                        events += session_events(f.read().decode('utf-8', 'replace').splitlines())
                    continue
                r = META.search(name)
                if r and m.isfile():
                    f = tf.extractfile(m)
                    try:
                        meta = json.loads(f.read()) if f else {}
                    except ValueError:
                        continue
                    if meta.get('agent_request', True) and isinstance(meta.get('started'), (int, float)):
                        started[r.group(1)] = float(meta['started'])
    except (OSError, tarfile.TarError, EOFError):
        return None
    if not started:
        return None
    # A request without a response file never completed for Pi: its client gave up (Pi's request timeout, e.g. 1800 s
    # on a stalled backend) and retried. It was in flight, as the live proxy's in_flight counts it, until the next
    # request (or the agent's end).
    order = sorted(started, key=started.get)
    intervals, missing = [], 0
    for j, i in enumerate(order):
        end = ended.get(i)
        if end is None:
            missing += 1
            end = started[order[j + 1]] if j + 1 < len(order) else (agent_end or started[i])
        intervals.append((started[i], end))
    gaps, last_end = busy_gaps(intervals)
    tail = (agent_end - last_end) if agent_end is not None and last_end is not None else None
    # Pi-side idle (what the watchdog judges): no request in flight and no tool call outstanding. A stretch starts at
    # a response's end or the last session entry after it (a tool result), and ends at the next request.
    events.sort()
    idle, k, merged, cur = [], 0, sorted(intervals), None
    for s0, e0 in merged:
        if cur is not None and s0 > cur:
            while k < len(events) and events[k] < cur:
                k += 1
            last = cur
            j = k
            while j < len(events) and events[j] < s0:
                last = max(last, events[j])
                j += 1
            idle.append(s0 - last)
        cur = e0 if cur is None else max(cur, e0)
    tail_idle = None
    if tail is not None:
        after = [t for t in events if t >= last_end and (agent_end is None or t <= agent_end)]
        tail_idle = agent_end - max([last_end] + after)
    # The wire logs are written after the turn gate and the KV gate let a request through, so a turn parked at a gate
    # reads as a gap here, while the live watchdog counts it as in flight (it is inside the proxy's handler). The env
    # clock's env_sec is the attempt's total time with no request in flight: it bounds any idle stretch, and gaps are
    # exact only in attempts that waited at no gate (gate_held_sec 0).
    # Unanswered streaks (the stall rule): from the first request with no response after the last answered one, to the
    # next answered request's start (recovered) or the agent's end (never answered). Pi resends a request it gave up
    # on (its 1800 s request timeout), so a stalled session shows a streak of resends.
    # As live, a streak is cleared when a later reply arrives, and the guard completing a turn is progress within it:
    # its stall length is the longest stretch with neither.
    guard.sort()

    def longest(a, b):
        pts = [a] + [t for t in guard if a < t < b] + [b]
        return max(y - x for x, y in zip(pts, pts[1:]))
    streaks, first, n = [], None, 0
    for i in order:
        if i in ended:
            if first is not None:
                streaks.append((n, longest(first, ended[i]), True))
            first, n = None, 0
        else:
            first = started[i] if first is None else first
            n += 1
    if first is not None:
        streaks.append((n, longest(first, max(agent_end or first, first)), False))
    return {'key': Path(path).parent.name, 'gaps': gaps, 'tail': tail, 'n_requests': len(started), 'streaks': streaks,
            'n_no_response': missing, 'env_sec': clock.get('env_sec'),
            'gate_held_sec': clock.get('kv_gate_held_sec'), 'idle': idle, 'tail_idle': tail_idle,
            'accounting_error_at': accounting_error,
            # a reply that arrived after the guard's accounting error would make the rule a false positive
            'answered_after_accounting_error': accounting_error is not None and any(
                e > accounting_error + 1.0 for e in ended.values()),
            'has_session': bool(events)}


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def measure(runs: list[Path], jobs: int = 8) -> dict:
    by_suite: dict[str, list[tuple[str, dict]]] = {}
    for run in runs:
        cfg = json.loads((run / 'config.json').read_text())
        suite = cfg.get('benchmark') or run.parent.name
        paths = sorted(str(p) for p in (run / 'attempts').glob('*/trial.tar.gz'))
        with ProcessPoolExecutor(jobs) as ex:
            for rec in ex.map(attempt, paths, chunksize=4):
                if rec is not None:
                    by_suite.setdefault(suite, []).append((run.name, rec))
    out = {}
    for suite, recs in sorted(by_suite.items()):
        # exact stretches: attempts that waited at no gate (a gate wait reads as a stretch in the wire logs)
        # gate_held_sec None (no env_clock.json, e.g. a stopped attempt) is unknown, not "no gate wait"
        exact = [r for _, r in recs if isinstance(r.get('gate_held_sec'), (int, float)) and r['gate_held_sec'] < 1.0]
        idle = [g for r in exact for g in r['idle']]
        p99 = quantile(idle, 0.99)
        limit = threshold(p99)
        cands, gated, unknown = [], 0, 0
        for run, r in recs:
            x = max(r['idle'] + ([r['tail_idle']] if r.get('tail_idle') is not None else []), default=0.0)
            if isinstance(r.get('env_sec'), (int, float)):
                x = min(x, r['env_sec'])     # never more than the attempt's total time with no request in flight
            if x <= limit:
                continue
            if not isinstance(r.get('gate_held_sec'), (int, float)):
                unknown += 1                 # no clock record: cannot tell a gate wait from idle time
            elif r['gate_held_sec'] + GATE_SLACK_SEC >= x:
                gated += 1                   # can be a turn parked at a gate: in flight for the live watchdog
            else:
                cands.append((round(x), run, r['key']))
        wire = [g for r in exact for g in r['gaps']]
        # stall rule check: unanswered streaks by outcome (recovered = a later request was answered)
        st = [(run, r['key'], n, d, ok) for run, r in recs for n, d, ok in r.get('streaks', [])]
        stalls = {
            'n_streaks': len(st),
            'recovered': {'n': sum(1 for x in st if x[4]),
                          'max_sec': round(max([x[3] for x in st if x[4]], default=0.0), 1),
                          'by_unanswered': {str(k): sum(1 for x in st if x[4] and min(x[2], 4) == k) for k in (1, 2, 3, 4)},
                          'over_1860_sec': sum(1 for x in st if x[4] and x[3] > 1860),
                          'over_3660_sec': sum(1 for x in st if x[4] and x[3] > 3660)},
            'stall_sec': STALL_SEC,
            'accounting_error_attempts': sum(1 for _, r in recs if r.get('accounting_error_at') is not None),
            # an accounting error in an attempt whose session still got a later reply would be a false positive
            'accounting_error_then_answered': sum(1 for _, r in recs if r.get('answered_after_accounting_error')),
            'never_answered_with_accounting_error': sum(1 for _, r in recs if r.get('accounting_error_at') is not None
                                                        and any(not ok for _, _, ok in r.get('streaks', []))),
            'would_fire_recovered': sum(1 for x in st if x[4] and x[3] > STALL_SEC),
            'would_fire_never_answered': sum(1 for x in st if not x[4] and x[3] > STALL_SEC),
            'never_answered': {'n': sum(1 for x in st if not x[4]),
                               'over_1860_sec': sum(1 for x in st if not x[4] and x[3] > 1860),
                               'over_3660_sec': sum(1 for x in st if not x[4] and x[3] > 3660),
                               'by_unanswered': {str(k): sum(1 for x in st if not x[4] and min(x[2], 4) == k)
                                                 for k in (1, 2, 3, 4)}},
            'recovered_examples': [[round(x[3]), x[0], x[1], x[2]] for x in sorted(
                (x for x in st if x[4]), key=lambda x: -x[3])[:6]],
        }
        out[suite] = {
            'p99_gap_sec': None if p99 is None else round(p99, 1),
            'p50_gap_sec': None if not idle else round(quantile(idle, 0.5), 1),
            'max_gap_sec': None if not idle else round(max(idle), 1),
            'idle_sec': round(limit),
            'would_fire': len(cands), 'would_fire_examples': [list(x) for x in sorted(cands, reverse=True)[:10]],
            'over_limit_explained_by_gate_waits': gated, 'over_limit_gate_unknown': unknown,
            'n_attempts': len(recs), 'n_attempts_no_gate_wait': len(exact),
            'n_attempts_with_session_log': sum(1 for _, r in recs if r.get('has_session')),
            'n_requests_no_response': sum(r.get('n_no_response', 0) for _, r in recs),
            'runs': sorted({run for run, _ in recs}),
            'stalls': stalls,
            # for reference: the same gaps with tool time counted as idle (the rule without the tool-call exemption)
            'with_tool_time': {'p99_gap_sec': None if not wire else round(quantile(wire, 0.99), 1),
                               'max_gap_sec': None if not wire else round(max(wire), 1),
                               'over_600_sec': sum(1 for _, r in recs if min(
                                   max(r['gaps'] + ([r['tail']] if r['tail'] is not None else []), default=0.0),
                                   r['env_sec'] if isinstance(r.get('env_sec'), (int, float)) else 1e18) > 600 and
                                   not ((r.get('gate_held_sec') or 0.0) >= 600))},
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(argv)
    suites = measure(a.runs, a.jobs)
    table = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'rule': f'idle_sec = max({P99_MULT:g} x p99 gap, {FLOOR_SEC:g} s); gap = no model request in flight and '
                     'no tool call outstanding (Pi itself idle)',
             'stall_sec': STALL_SEC,
             'stall_rule': 'a forwarded request with no response started, and none answered since, for stall_sec',
             'source': 'aa_evals/common/pi_eval/request_gaps.py', 'suites': suites}
    a.out.write_text(json.dumps(table, indent=1, sort_keys=True) + '\n')
    for s, v in suites.items():
        print(s, {k: v[k] for k in v if k not in ('runs',)})


if __name__ == '__main__':
    main()

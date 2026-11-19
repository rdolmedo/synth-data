#!/usr/bin/env python3
"""#344 dispatcher: one per model run. A single in-flight budget for the model's server, shared by every suite, and
the longest predicted attempts first across suites.

  python -m aa_evals.common.dispatch.dispatcher init DIR --budget 132 --model qwen35-4b [--expect-lanes N]
        [--durations TABLE] [--burst 16] [--metrics http://engine:port] [--floor 16]
  python -m aa_evals.common.dispatch.dispatcher run DIR          # the loop (a background process / CPU job)
  python -m aa_evals.common.dispatch.dispatcher status DIR       # budget, in flight by suite, queue head
  python -m aa_evals.common.dispatch.dispatcher report DIR [--out attempts.jsonl]   # per-attempt start/end/pred
  python -m aa_evals.common.dispatch.dispatcher budget DIR N|auto   # live fixed budget (auto off) / auto again
  python -m aa_evals.common.dispatch.dispatcher plan PLAN.json --budget N --model M [--ids-out DIR]   # c30 plan file

Lanes register themselves under DIR/lanes/<name>/registration.json (one lane per run) when their suite starts with
`--dispatch DIR`:
- kind "pi": a pi_eval run (TB 2.1, TB-Sci-lite and the suites ported to Pi). Its queue is eval_q35's (claims, done,
  retries; items.txt written longest first). Starts = its claims so far. The dispatcher writes RUN/control.json
  `max_starts`, which the run's workers check under a lock before each claim (pilane.credit_claim); their own
  admission (burst, serving-gate pause, stale-claim reaper) is unchanged. Capacity = slots of the workers with a
  fresh heartbeat.
- kind "lane": a single-turn client (HLE, Omniscience, GDP.pdf, SciCode) through `lane.Lane`; it publishes
  lane.json (starts, in flight, pending with weights) and events.jsonl and reads DIR/lanes/<name>/control.json.

Allowances are cumulative **start credits**: a lane may start attempts while its starts < its credit. A finished
attempt therefore returns its slot to the dispatcher, not to its lane. Every tick (15 s):
  used(lane) = in flight + outstanding credit (credit - starts: granted, not yet started; counted as in use until the
               lane spends it, so a slow-reading lane never overshoots the budget);
  free = budget - sum(used); the pending attempts of all lanes (after the ones the outstanding credit covers) are
  merged by predicted duration x the suite's online factor, and free slots go to the head of that merged queue, by
  weight, strictly in order, at most each lane's capacity; credit += grant (credits never shrink).
A budget cut or a --metrics backoff stops grants until the in-flight work drains below it; running attempts are never
cut. A single-turn lane whose lane.json is stale for 10 min is counted as 0 (its client died) and logged.
Online factor per suite: geometric mean of actual / table seconds over this run's finished attempts (plus in-flight
ones already longer than predicted), shrunk toward the all-suite factor (k = 5 attempts).
Start-up: no grants until the expected lanes are registered and ready (pi: a worker heartbeat), or 30 min passed, so
the longest attempts of every suite start first. With --metrics, the budget also backs off like adaptive.py (KV
preemptions / waiting requests) and recovers by +2 per calm tick, and (new dirs, #378) --budget is only the start
value: autobudget.py hill-climbs it on the engine's generated tok/s between --budget-min and --budget-max. The
dispatcher runs until DIR/STOP exists (lanes can register or resume at any time).

Files: DIR/dispatch.json (config), DIR/control.json (live {"budget": N} or {"auto": true}), DIR/auto_budget.json (the
auto budget's level and blocked level, kept across restarts), DIR/state.json (each tick), DIR/events.jsonl
(start/end of every attempt with its prediction), DIR/log.txt (allocation changes, warnings).
"""
from __future__ import annotations

import argparse
import datetime as dt
import heapq
import json
import math
import os
import socket
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .. import jsonl
from . import autobudget
from .durations import Durations, DEFAULT_TABLE
from .lane import LANE_FRESH, atomic_json, pid_alive, read_json
from .pilane import OPEN_LIMIT

TICK_SEC = 15.0
HEARTBEAT_FRESH = 180.0     # a pi worker heartbeat (every ~20 s) older than this: worker gone
DEAD_SEC = 600.0            # a single-turn lane.json older than this: its client died; its slots are released
READY_TIMEOUT = 1800.0
SHRINK_K = 5.0
MISSING_WARN = 0.10   # warn when more than this share of a lane's pending items have no prediction
REWRITE_SEC = 300.0   # control files are rewritten at least this often (a heartbeat of the dispatcher)
RESERVE_SEC = 600.0   # pi credit left unspent this long while the gate is clear is withdrawn
GATE_TIMEOUT = 5.0
# Pi worker growth (#378): a pi lane that could take more of the budget than its workers' slots (shortfall) for
# SCALE_TICKS ticks gets workers added, at most one batch per SCALE_COOLDOWN, never past its pending work.
SCALE_TICKS = 4
SCALE_COOLDOWN = 600.0
# Lazy first workers (#378, pi_eval run --lazy-workers): a run registers with no worker; until its first worker
# submission it is ready for the start-up release and offers LAZY_PROBE_SLOTS of capacity, so at most that much
# granted credit waits (counted as used) while its first workers start; its first grant submits them.
LAZY_PROBE_SLOTS = 1
LAZY_RETRY_SEC = 300.0
# ... and it reserves its first fleet's slots (worker.n x slots) in the global longest-first order, as eager first workers
# did: with only the probe slot, the single-turn lanes (unbounded capacity) took the whole budget at launch and the Pi
# lanes got slots only as single-turn items finished (val-sb10b 11-18: SciCode 251/256 in flight after a minute, TB2.1,
# GDPval and AB 1 each; pre-#511: TB2.1 80, GDPval 75, SciCode 114, AB 42). The reserved credit is held unspent while
# the workers start. Bounds: before the first submission is recorded (it keeps failing) the gate reservation's rule
# withdraws it after reserve_sec; once the first submission is older than ready_timeout (the start-up grace eager
# lanes had), the lane's credit is capped at its starts + its live workers' free slots (read from files: a restarted
# dispatcher applies it too), so workers that never start, or a fleet that starts only in part, hold nothing.


def log(d: Path, msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())} {msg}'
    print(line, flush=True)
    with open(d / 'log.txt', 'a') as f:
        f.write(line + '\n')


def iso_epoch(s: str | None) -> float | None:
    """ISO-8601 time as written by eval_q35 (`...+00:00`) or harbor_mswea (`...Z`); naive times are UTC."""
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).timestamp()


def parse_key(k: str) -> tuple[str, int, int]:
    """eval_q35 key task__a<k>__r<retry>."""
    task, a, r = k.rsplit('__', 2)
    return task, int(a[1:]), int(r[1:])


# ---------------------------------------------------------------------------------------------- lane views

class PiLane:
    """A pi_eval run's queue (eval_q35 queue_state layout), read-only except RUN/control.json."""
    kind = 'pi'

    def __init__(self, reg: dict):
        self.reg = reg
        self.name, self.suite, self.run = reg['name'], reg['suite'], Path(reg['run'])
        self.q = self.run / 'queue'
        self.cfg = read_json(self.run / 'config.json', {}) or {}
        self.items = [(p[0], int(p[1])) for p in (ln.split() for ln in (self.q / 'items.txt').read_text().splitlines())
                      if len(p) >= 2]
        self.seen_done: set[str] = set()
        self.started: dict[str, float] = {}
        self.credit = int((read_json(self.control_path(), {}) or {}).get('max_starts') or 0)

    def _names(self, sub, suffix=''):
        try:
            return {p.name.removesuffix(suffix) for p in (self.q / sub).iterdir() if not p.name.startswith('.')}
        except FileNotFoundError:
            return set()

    @property
    def default_slots(self) -> int:
        return int((self.cfg.get('worker') or {}).get('slots') or 1)

    def first_submitted(self) -> float | None:
        """Epoch of the run's first worker submission (RUN/jobs.json submitted_at; the file's mtime without it)."""
        recs = read_json(self.run / 'jobs.json', []) or []
        if not recs:
            return None
        ts = []
        for r in recs if isinstance(recs, list) else []:
            try:
                ts.append(dt.datetime.fromisoformat(str(r.get('submitted_at')).replace('Z', '+00:00')).timestamp())
            except (TypeError, ValueError):
                continue
        if ts:
            return min(ts)
        try:
            return (self.run / 'jobs.json').stat().st_mtime
        except OSError:
            return None

    @property
    def lazy(self) -> bool:
        """A --lazy-workers run whose first workers are not submitted yet (no RUN/jobs.json record)."""
        if not self.cfg.get('lazy_workers') or getattr(self, '_lazy_over', False):
            return False
        if read_json(self.run / 'jobs.json', []) or []:
            self._lazy_over = True
            return False
        return True

    def stop_words(self) -> list[str] | None:
        """RUN/STOP_WORKERS as words (eval_q35 Worker.stop_requested), None when absent."""
        try:
            return (self.run / 'STOP_WORKERS').read_text().split()
        except FileNotFoundError:
            return None
        except OSError:
            return []

    def worker_jobs(self) -> dict[str, tuple[str | None, int]]:
        """condor cluster -> (code snapshot dir name, slots) of the run's worker jobs, from RUN/jobs.json and each
        submit file's `arguments = <code>/aa_evals/common/pi_eval/worker_job.sh RUN SLOTS` (cached per file)."""
        cache = self.__dict__.setdefault('_jobs', {})
        for j in read_json(self.run / 'jobs.json', []) or []:
            c = str(j.get('cluster'))
            if c in cache:
                continue
            snap, slots = None, self.default_slots
            try:
                for line in Path(j['submit_file']).read_text().splitlines():
                    if line.startswith('arguments'):
                        args = line.split('=', 1)[1].split()
                        snap = Path(args[0].split('/aa_evals/', 1)[0]).name
                        slots = int(args[-1])
            except (OSError, KeyError, ValueError, IndexError):
                pass
            cache[c] = (snap, slots)
        return cache

    def control_path(self) -> Path:
        return self.run / 'control.json'

    def snapshot(self, now: float) -> dict:
        claims, done, retry = self._names('claims'), self._names('done', '.json'), self._names('retry', '.json')
        active = claims - done
        # queue_state.claim's order (#545): lowest attempt, then retry index, then the queue's own order (a set's
        # iteration order depends on hash randomization), then the key
        pos = {ta: i for i, ta in enumerate(self.items)}

        def order(k):
            p = parse_key(k)
            return p[1], p[2], pos.get((p[0], p[1]), len(pos)), k
        pend = sorted((k for k in retry if k not in claims), key=order)
        pend += [f'{t}__a{a}__r0' for t, a in self.items if f'{t}__a{a}__r0' not in claims]
        stop = self.stop_words()
        # eval_q35 drain: 'drain <sha> ...' stops only the workers on those code snapshots; anything else stops all
        stopped = stop is not None and stop[:1] != ['drain']
        drained = [c[:12] for c in stop[1:]] if stop and stop[:1] == ['drain'] else []
        cap, newest, n_drained, last_exit, n_left = 0, None, 0, None, 0
        jobs = self.worker_jobs()
        for p in (self.run / 'workers').glob('*.json') if (self.run / 'workers').exists() else []:
            w = read_json(p, {}) or {}
            hb = iso_epoch(w.get('heartbeat'))
            if hb and (newest is None or hb >= newest):
                newest, last_exit = hb, w.get('exited')
            if w.get('exited') in ('idle-no-credit', 'no-claimable'):
                n_left += 1   # left on its own (#378 idle exit, or eval_q35's no-claimable exit under packing)
            if w.get('exited'):
                continue   # a worker that left (#378 idle exit) has no slots, however fresh its last heartbeat
            if hb and now - hb < HEARTBEAT_FRESH:
                snap, slots = jobs.get(p.name.split('.', 1)[0], (None, self.default_slots))
                if snap and any(snap.startswith(c) for c in drained):
                    n_drained += 1   # finishing its attempts, claims nothing: no capacity
                else:
                    cap += slots
        # no worker heartbeat for DEAD_SEC (stopped, abandoned or evicted run): its claims and unspent credit no
        # longer hold server slots. A run that never had a worker yet is not dead (nothing of it can be in flight).
        self.first_seen = getattr(self, 'first_seen', now)
        # #378: every worker left because the lane had no start credit (the last heartbeat says so) and nothing of it
        # is in flight: the lane waits for credit, it is not dead, and its admission.json (written by the workers
        # that left) no longer says anything
        idle_exited = cap == 0 and last_exit == 'idle-no-credit' and not active
        # bool(): `claims` is a set, and an empty one would leak into state.json as the lane's 'dead' (not JSON)
        dead = not idle_exited and ((newest is not None and now - newest > DEAD_SEC) or (
            newest is None and bool(claims) and now - self.first_seen > DEAD_SEC))
        # a lane some of whose workers left on their own (#378 idle exit, or the no-claimable exit) and whose last
        # worker then died: nobody is left to reap that worker's claims, so the lane is dead, but it grows back (a new
        # worker reaps them) when it has the credit. Its open claims count as work: a dead lane's claims belong to lost
        # workers and become same-seed retries once reaped (as tbsched.work.pending counts dead claims as claimable),
        # so a lane whose only work left is a lost straggler's claims (#538: claim packing's last worker evicted) is
        # revived too. A run abandoned on purpose is not: `run.py --stop` (STOP_WORKERS) or the lane's NO_HOLD makes it
        # not scalable (heartbeats alone cannot tell an evicted last worker from a condor_rm'd fleet).
        revivable = dead and n_left > 0 and bool(pend or active) and (self.run / 'jobs.json').exists()
        work_left = len(active) + len(pend)
        if revivable:   # its lost claims are work to grant and size workers for: first, as the retries they become
            # once reaped are claimed before unclaimed attempts (n_pending/head then name keys that are still claimed)
            pend = sorted(active, key=order) + pend
        adm = {} if idle_exited or revivable else read_json(self.q / 'admission.json', {}) or {}
        paused = bool(adm.get('paused')) or stopped
        events, finished = [], []
        for k in active:
            if k not in self.started:
                c = read_json(self.q / 'claims' / k / 'claim.json', {}) or {}
                self.started[k] = iso_epoch(c.get('claimed_at')) or now
                events.append({'ev': 'start', 'key': k, 'item': parse_key(k)[0], 't': round(self.started[k], 2)})
        for k in sorted(done - self.seen_done):
            rec = read_json(self.q / 'done' / f'{k}.json')
            if rec is None:
                continue
            self.seen_done.add(k)
            t1 = iso_epoch(rec.get('finished_at')) or now
            t0 = self.started.pop(k, None)
            if t0 is None:   # started and finished between two ticks
                c = read_json(self.q / 'claims' / k / 'claim.json', {}) or {}
                t0 = iso_epoch(c.get('claimed_at'))
                if t0 is not None:
                    events.append({'ev': 'start', 'key': k, 'item': parse_key(k)[0], 't': round(t0, 2)})
            sec = rec.get('wall_sec') or (t1 - t0 if t0 else None)
            st = ('infra:' + str(rec.get('infrastructure_reason'))) if rec.get('infrastructure_failure') else 'ok'
            end_state = (rec.get('stats') or {}).get('end_state')
            ev = {'ev': 'end', 'key': k, 'item': parse_key(k)[0], 't': round(t1, 2), 'sec': sec, 'status': st,
                  'end_state': end_state, 'score': rec.get('score')}
            events.append(ev)
            if t0 is not None or sec is not None:
                finished.append(ev)
        running = {k: self.started.get(k, now) for k in active}
        lazy = self.lazy and not stopped
        if lazy:   # no worker yet: ready once registered; one probe slot so its first grant comes in priority order
            cap = max(cap, LAZY_PROBE_SLOTS)
        # the slots its first workers will bring, reserved at launch (decide(); LAZY_PROBE_SLOTS comment)
        reserve = self.default_slots * max(1, int((self.cfg.get('worker') or {}).get('n') or 1)) if lazy else 0
        return {'in_flight': 0 if dead else len(active), 'starts': len(claims), 'dead': dead,
                'pending': [(k, parse_key(k)[0], 1) for k in pend], 'n_pending': len(pend),
                'capacity': 0 if paused else cap, 'ready': cap > 0 or idle_exited, 'paused': paused,
                'idle_exited': idle_exited, 'revivable': revivable, 'lazy': lazy, 'lazy_reserve': reserve,
                # server release: stopped on purpose (STOP_WORKERS, not drain) never holds the server as STUCK; a dead
                # lane's own counts (in_flight above reads 0 for it): claims still open + work never claimed
                'stopped': stopped, 'work_left': work_left if dead else 0,
                # paused only because the serving gate holds new conversations (pi_eval worker.refresh): the lane
                # keeps a reservation of its workers' slots (Dispatcher.decide), so it is not starved by the lanes
                # whose requests keep the gate holding
                'gate_paused': paused and not stopped and str(adm.get('reason') or '').startswith('gate holds'),
                # #378 per-replica pause: some replicas' gates hold, and the workers claim only attempts pinned to the
                # others (not paused: the lane still takes credit)
                'gate_partial': not paused and bool(adm.get('gate_held_replicas')),
                'worker_capacity': cap,
                'reason': ('no worker heartbeat for ' + f'{(now - newest) if newest else now - self.first_seen:.0f}s'
                           if dead else 'workers left: no start credit (grows back with credit)'
                           if idle_exited and not stopped else adm.get('reason')
                           or ('STOP_WORKERS' if stopped else None)),
                'draining_workers': n_drained,
                'done': not active and not pend, 'running': running, 'events': events, 'finished': finished}

    def write_control(self, credit: int, burst: int, why: str) -> None:
        atomic_json(self.control_path(), {'max_starts': int(credit), 'max_in_flight': OPEN_LIMIT, 'burst': int(burst),
                                          'why': why, 'epoch': round(time.time(), 1), 'writer': 'dispatch'})


class StLane:
    """A single-turn client's lane (lane.Lane), read through DIR/lanes/<name>/."""
    kind = 'lane'

    def __init__(self, reg: dict, lane_dir: Path):
        self.reg = reg
        self.name, self.suite, self.run, self.dir = reg['name'], reg['suite'], reg.get('run'), lane_dir
        self.offset = 0
        self.started: dict[str, float] = {}
        self.credit = int((read_json(self.control_path(), {}) or {}).get('max_starts') or 0)
        self.shrinkable = False

    def control_path(self) -> Path:
        return self.dir / 'control.json'

    def snapshot(self, now: float) -> dict:
        st = read_json(self.dir / 'lane.json', {}) or {}
        events, finished = [], []
        try:
            with open(self.dir / 'events.jsonl') as f:
                f.seek(self.offset)
                chunk = f.read()
            whole = chunk[:chunk.rfind('\n') + 1]
            self.offset += len(whole.encode())
            for line in whole.split('\n'):   # '\n' only: str.splitlines() also splits inside JSON strings
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except ValueError as x:   # a torn line of a killed writer: skipped, never silently
                    print(f'{self.dir}/events.jsonl: undecodable line skipped ({x})', file=sys.stderr, flush=True)
                    continue
                events.append(e)
                if e['ev'] == 'start':
                    self.started[e['key']] = e['t']
                else:
                    self.started.pop(e['key'], None)
                    finished.append(e)
        except FileNotFoundError:
            pass
        age = now - float(st.get('epoch') or 0) if st else float('inf')
        fresh, closed = age < LANE_FRESH, bool(st.get('closed'))
        dead = bool(st) and not closed and age > DEAD_SEC
        pend = [(p[0], p[1], int(p[3]) if len(p) > 3 else 1) for p in st.get('pending') or []]
        self.shrinkable = bool(st.get('shrinkable'))
        in_flight = 0 if dead else int(st.get('in_flight') or 0)
        return {'in_flight': in_flight, 'starts': int(st.get('starts') or 0), 'dead': dead,
                'pending': [] if dead else pend, 'n_pending': int(st.get('n_pending') or 0),
                # ready for the start-up release once the client has announced its work (lane.expect)
                'capacity': 10 ** 6 if fresh and not closed else 0,
                'ready': fresh and bool(st.get('announced', True)), 'paused': not fresh,
                'reason': None if fresh else ('closed' if closed else f'lane.json stale {age:.0f}s' if st else
                                              'no lane.json'),
                'done': closed and not in_flight, 'running': {} if dead else dict(self.started),
                # #378: the client takes a lower credit (control.json `withdraw`); older clients only grow theirs
                'shrinkable': self.shrinkable,
                # server release: the client will send more model work after CPU work in progress (lane.json)
                'hold_server': bool(st.get('hold_server')) and not dead,
                # a dead client's own counts (in_flight above reads 0 for a dead lane): work it left unrun
                'work_left': (int(st.get('n_pending') or 0) + int(st.get('in_flight') or 0)
                              + bool(st.get('hold_server'))) if dead else 0,
                # opened but its work not announced yet (lane.expect): more model work is coming
                'unannounced': bool(st) and not closed and not dead and not bool(st.get('announced', True)),
                'events': events, 'finished': finished}

    def write_control(self, credit: int, burst: int, why: str) -> None:
        # `exact` (#378, clients that publish `shrinkable`): max_starts is the credit as is, also when lower than
        # before (the single-turn hold, and a lane taken for dead whose unspent credit the budget already got back);
        # older clients keep the larger of theirs and this one
        atomic_json(self.control_path(), {'max_starts': int(credit), 'why': why, 'epoch': round(time.time(), 1),
                                          **({'exact': True} if self.shrinkable else {})})


# ---------------------------------------------------------------------------------------------- core

class Factors:
    """Online rescaling: per-suite log(actual / table) with shrinkage to the all-suite mean."""

    def __init__(self):
        self.sum: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)

    def add(self, suite: str, actual: float, pred: float) -> None:
        if actual and pred and actual > 0 and pred > 0:
            self.sum[suite] += math.log(actual / pred)
            self.n[suite] += 1

    def factors(self, censored: dict[str, list[float]] | None = None) -> dict[str, float]:
        s, n = dict(self.sum), dict(self.n)
        for suite, logs in (censored or {}).items():
            s[suite] = s.get(suite, 0.0) + sum(logs)
            n[suite] = n.get(suite, 0) + len(logs)
        g = sum(s.values()) / (sum(n.values()) + SHRINK_K)
        out = {suite: math.exp((s[suite] + SHRINK_K * g) / (n[suite] + SHRINK_K)) for suite in s}
        out['*'] = math.exp(g)
        return out


def allocate(free: int, room: dict[str, int], queues: dict[str, list[tuple[float, int]]],
             idle: bool = False) -> dict[str, int]:
    """Grant `free` slots to the longest pending attempts across lanes, strictly in order of prediction.
    room[lane] = how many more units the lane can run (capacity - used); queues[lane] = (scaled prediction, weight) of
    its pending attempts in the lane's own start order. The global head waits when its weight does not fit `free`
    (except when nothing is in use: `idle`); a lane at its capacity passes its turn. Returns grants (units)."""
    grants = {n: 0 for n in queues}
    heap = [(-q[0][0], n, 0) for n, q in queues.items() if q]
    heapq.heapify(heap)
    while heap and free > 0:
        _, n, i = heapq.heappop(heap)
        w = queues[n][i][1]
        if w > room.get(n, 0) - grants[n]:
            continue                     # this lane cannot take it now; the others go on
        if w > free and not (idle and not any(grants.values())):
            break                        # strict order: the head waits for enough free slots
        grants[n] += w
        free -= w
        if i + 1 < len(queues[n]):
            heapq.heappush(heap, (-queues[n][i + 1][0], n, i + 1))
    return grants


def decide(views: dict[str, dict], budget: int, gate_queued: int | None, now: float, released: bool = True,
           hold_min: int = 1, reserve_sec: float = RESERVE_SEC,
           flowing: bool = True, st_hold_on: bool = True) -> tuple[dict[str, int], dict[str, int], dict]:
    """The dispatcher's grant policy for one tick (pure; the live tick and the tests' simulation both call it).

    views[lane]: in_flight, starts, credit, capacity (0 when paused), gate_paused, worker_capacity, pending =
    [(scaled prediction, weight)] in the lane's start order, dead, withdrawable (pi: control.json is read raw),
    unspent_since (when its unspent credit last stopped being spent, or None).
    Returns (grants, withdrawals {lane: new credit}, info).
    - used = in flight + unspent credit; a dead lane counts 0.
    - Gate back-off: while the serving gate holds >= hold_min new conversations (gate_queued; None = no gate), nobody
      gets new credit, so the gate drains and the Pi lanes it paused resume.
    - Reservation: a lane paused only by the gate keeps its workers' slots as capacity, so it takes credit in global
      longest-first order (held unspent until its workers resume) instead of losing it to the lanes that keep the
      gate holding. A lazy pi lane (no worker submitted yet, `lazy_reserve` > 0) likewise reserves its first fleet's
      slots, so the single-turn lanes do not take the whole budget at launch while its first workers start.
    - Withdrawal: a gate-paused (or reserved) pi lane's credit unspent for reserve_sec while the gate is not seen
      holding goes back (credit = starts), also when the gate is unreachable or --gate-backoff is off; the lane then
      gets no new reservation until it starts something or its workers see the gate clear (`withdrawn`). Lanes never
      paused by a gate are untouched: without a gate nothing here differs from the round-1 policy.
    - Single-turn hold (#378, `st_hold_on`): while the gate holds (back-off: nobody gets new credit) and a pi lane
      with pending work is gate-paused (pi_eval pauses only while every replica's gate holds), a `shrinkable`
      single-turn lane's unspent credit goes back at once (withdraw: credit = starts). Requests already sent stay, but
      nothing refills the gate queue they sit in, so it drains and the pi lane resumes. (c30 Xhigh: ~46 unspent
      SciCode credits were spent into the gate queue during the back-off, which kept it holding and the pi lanes
      paused ~50 min.)"""
    withdraw, used, room, queues, caps = {}, {}, {}, {}, {}
    gate_hold = gate_queued is not None and gate_queued >= hold_min
    st_hold = st_hold_on and gate_hold and any(
        not v['dead'] and lane_kind(v) == 'pi' and v['pending'] and v.get('gate_paused') for v in views.values())
    for n, v in views.items():
        credit = v['credit']
        if st_hold and lane_kind(v) == 'lane' and v.get('shrinkable') and not v['dead'] and credit > v['starts']:
            withdraw[n] = credit = v['starts']
        # bounded reservation: credit a gate-paused lane has not spent for reserve_sec goes back, unless the gate is
        # seen holding (then the lane is waiting for it to drain, which the back-off is doing). Lanes never paused
        # by a gate (every lane on a server without one) are not touched.
        if v['withdrawable'] and (v.get('gate_paused') or v.get('reserved')) and not gate_hold \
                and credit > v['starts'] and v.get('unspent_since') is not None \
                and now - v['unspent_since'] > reserve_sec:
            withdraw[n] = credit = v['starts']
        out = 0 if v['dead'] else max(0, credit - v['starts'])
        used[n] = 0 if v['dead'] else v['in_flight'] + out
        cap = v['worker_capacity'] if v.get('gate_paused') else v['capacity']
        if v.get('lazy_reserve') and not v.get('withdrawn') and n not in withdraw:
            cap = max(cap, v['lazy_reserve'])   # lazy first workers: their slots are reserved at launch
        # a lane whose reservation was withdrawn gets no new one until it starts something or its gate clears
        if lane_kind(v) == 'pi' and (n in withdraw or v.get('withdrawn')):
            cap = v['capacity'] if not v.get('gate_paused') else 0
        room[n] = max(0, cap - used[n])
        caps[n] = cap
        q, skip = [], out   # the first `out` units are already covered by credit
        for pred, w in v['pending']:
            if skip > 0:
                skip -= w
                continue
            q.append((pred, w))
            if len(q) >= max(budget, 1):
                break
        queues[n] = q
    free = budget - sum(used.values())
    if released and not gate_hold:
        grants = allocate(free, room, queues, idle=sum(used.values()) == 0)
    else:
        grants = {n: 0 for n in views}
    # shortfall (#378): what each scalable pi lane would also take, in the same longest-first order, from the slots
    # left free if its workers had room. Lanes paused by the gate or a stop, and non-scalable lanes, have none.
    shortfall = {n: 0 for n in views}
    left = free - sum(grants.values())
    # a dead lane is scalable only when some of its workers left on their own and work is left (`revivable`)
    scal = [n for n, v in views.items() if v.get('scalable') and not v.get('gate_paused')
            and (not v['dead'] or v.get('revivable'))]
    if left > 0 and scal and released and not gate_hold:
        room2 = {n: (10 ** 6 if n in scal else max(0, room[n] - grants[n])) for n in views}
        queues2 = {}
        for n, q in queues.items():
            skip, rest = grants[n], []
            for pred, w in q:
                if skip > 0:
                    skip -= w
                    continue
                rest.append((pred, w))
            queues2[n] = rest
        extra = allocate(left, room2, queues2)
        shortfall = {n: (extra[n] if n in scal else 0) for n in views}
    # Whole-budget shortfall (#451, the 11-16 Snowflake lesson; val-sb10b 11-18): with the budget full, every slot
    # freed this tick is granted (`left` is 0), so a scalable pi lane at its workers' room shows no shortfall above
    # while unlimited single-turn lanes keep absorbing the freed slots, and the longest lane never grows (TB2.1 stayed
    # at its first fleet, 64 of 256, behind 900 Omniscience items). Its shortfall is also what it would hold of the
    # whole budget in longest-first order if it had room, minus what it holds after this tick's grants. In-flight
    # single-turn units rank at their lane's queue head (they turn over in minutes: those slots will free); in-flight pi
    # sessions are held (they run for an hour or more: counting another pi lane's tail as freeing would grow workers
    # that wait the tail out, review). Only lanes with live workers: their new workers get slots as the single-turn
    # items they outrank finish; a lane whose workers all left (idle exit) grows back gradually through its virtual
    # shortfall below, or it would refill workers that leave again.
    if scal and released and not gate_hold:
        q_all, room4 = {}, {}
        for n, v in views.items():
            pend = list(v['pending'][:max(budget, 1)])
            head = math.inf if lane_kind(v) == 'pi' else (pend[0][0] if pend else 0.0)
            q_all[n] = [] if v['dead'] and n not in scal else [(head, 1)] * (0 if v['dead'] else v['in_flight']) + pend
            room4[n] = 10 ** 6 if n in scal else max(caps[n], used[n])
        ideal = allocate(budget, room4, q_all, idle=True)
        for n in scal:
            v = views[n]
            if v.get('worker_capacity', v['capacity']) > 0 and not (v.get('idle_exited') or v.get('revivable') or v['dead']):
                shortfall[n] = max(shortfall[n], ideal[n] - used[n] - grants[n])
    # #378 idle exit: a lane whose workers all left for lack of credit has no room, so it passes its turn and the slots
    # freed tick by tick go to the lanes after it in longest-first order; it would wait for them to drain. Its
    # shortfall is instead what it would win of the next slots to free (a tenth of the budget while slots are being
    # granted (`flowing`), else what is free now) if every lane that can take credit had room, so it grows back while
    # it is still the head. Dead, stopped, paused and withdrawn lanes keep their real room. Grants are unchanged.
    idle = [n for n in scal if views[n].get('idle_exited') or views[n].get('revivable')]
    if idle and released and not gate_hold:
        room3 = {n: (10 ** 6 if n in scal or (not v['dead'] and not v.get('withdrawn') and not v.get('gate_paused')
                                               and v['capacity'] > 0) else room[n]) for n, v in views.items()}
        virt = allocate(max(free, math.ceil(budget / 10) if flowing else 0), room3, queues, idle=True)
        for n in idle:
            shortfall[n] = max(shortfall[n], virt[n])
    return grants, withdraw, {'used': used, 'free': free, 'gate_hold': gate_hold, 'st_hold': st_hold,
                              'shortfall': shortfall}


def lane_kind(v: dict) -> str:
    """'pi' or 'lane' (single-turn): views carry `kind`; older callers mark pi lanes by `withdrawable` only."""
    return v.get('kind') or ('pi' if v.get('withdrawable') else 'lane')


def alive_worker_slots(lane, runner=None) -> int:
    """Slots of a pi lane's worker jobs that are queued, running or held (condor_q over RUN/jobs.json clusters)."""
    import subprocess
    jobs = lane.worker_jobs()
    if not jobs:
        return 0
    r = (runner or subprocess.run)(['condor_q', *jobs, '-af', 'ClusterId', 'JobStatus'], capture_output=True, text=True,
                                   timeout=120)
    if r.returncode:   # unknown is not zero: growth must not answer a schedd error with a second fleet
        raise RuntimeError(f'condor_q failed: {(r.stderr or "").strip()[:200]}')
    n = 0
    for line in (r.stdout or '').splitlines():
        p = line.split()
        if len(p) == 2 and p[1] in ('1', '2', '5'):   # held too (a memory hold is released): not replaced at each cooldown
            n += jobs.get(p[0], (None, lane.default_slots))[1]
    return n


def workers_to_add(shortfall: int, in_flight: int, outstanding: int, n_pending: int, job_slots: int, slots: int,
                   short_ticks: int, since_last: float | None, budget: int | None = None) -> int:
    """Pi worker growth (pure): how many workers to submit now. `job_slots` = slots of the lane's worker jobs queued or
    running (starting workers count, so a slow start is not answered twice). Needs a shortfall for SCALE_TICKS ticks,
    SCALE_COOLDOWN since the last batch, and never more slots than in flight + pending work."""
    if shortfall <= 0 or short_ticks < SCALE_TICKS or (since_last is not None and since_last < SCALE_COOLDOWN):
        return 0
    want = min(in_flight + outstanding + shortfall, in_flight + n_pending)
    if budget is not None:
        want = min(want, budget)              # a lane never needs more slots than the whole budget
    return max(0, math.ceil((want - job_slots) / max(1, slots)))


class Dispatcher:
    def __init__(self, d: Path):
        self.d = d
        self.cfg = read_json(d / 'dispatch.json')
        if not self.cfg:
            raise SystemExit(f'{d}/dispatch.json missing: run `init` first')
        self.dur = Durations(self.cfg.get('durations') or DEFAULT_TABLE, self.cfg.get('model'))
        self.lanes: dict = {}
        self.factors = Factors()
        self.pred_at_start: dict[tuple[str, str], float] = {}
        self.t0 = time.time()
        self.released = bool((read_json(d / 'state.json', {}) or {}).get('released'))   # survives a restart
        self.target = None       # adaptive (metrics) target
        self.pre = None
        self.auto: autobudget.AutoBudget | None = None   # auto budget (#378), made at the first tick it is on
        self.auto_was_on = False
        self.auto_logged = None
        self.last_alloc = None
        self.drained_logged = False
        self.current: dict[str, float] = {'*': 1.0}
        self.items: dict[tuple[str, str], str] = {}   # (lane, running key) -> item
        self.snaps: dict = {}
        # a restarted dispatcher re-reads every lane from the start: events already logged rebuild the factors only
        self.seen: set = set()
        for e in jsonl.Tolerant().read(d / 'events.jsonl'):   # bad lines are warned on stderr
            self.seen.add((e['lane'], e['key'], e['ev']))

    def discover(self) -> None:
        for reg_path in sorted((self.d / 'lanes').glob('*/registration.json')):
            reg = read_json(reg_path)
            if not reg:
                continue
            old = self.lanes.get(reg['name'])
            if old is not None and old.reg == reg:
                continue
            try:
                lane = PiLane(reg) if reg.get('kind') == 'pi' else StLane(reg, reg_path.parent)
            except (OSError, ValueError, KeyError) as e:
                log(self.d, f'lane {reg.get("name")}: not readable yet ({e!r})')
                continue
            if old is not None:   # a resumed or replaced client: keep the event offset, re-read its credit
                lane.offset, lane.started = getattr(old, 'offset', 0), old.started
                lane.seen_done = getattr(old, 'seen_done', set())
            self.lanes[reg['name']] = lane
            log(self.d, f'lane {reg["name"]} {"re-registered" if old else "registered"}: {reg.get("kind")} '
                        f'suite={reg["suite"]} run={reg.get("run")} pid={reg.get("pid")}')
            if reg['suite'] not in self.dur.table['suites']:
                log(self.d, f'WARNING lane {reg["name"]}: suite {reg["suite"]!r} is not in the duration table '
                            f'({sorted(self.dur.table["suites"])}): every attempt gets the default prediction')

    def registry_root(self) -> Path:
        """The serving registry root of this run: dispatch.json `serving_env` (recorded at init), whatever this
        process has (a dispatcher restarted from another shell keeps reading the run's registry); older dirs without
        it: serving.registry_root() of this process. harbor_mswea.endpoint.SERVING_ROOT (a test's) wins."""
        from aa_evals.common import serving
        from aa_evals.common.harbor_mswea import endpoint as E
        if E.SERVING_ROOT:
            return Path(E.SERVING_ROOT)
        if 'serving_env' in self.cfg:
            return Path((self.cfg['serving_env'] or {}).get(serving.ROOT_ENV) or serving.ROOT)
        return serving.registry_root()

    def registry(self) -> str | None:
        """The model's serving registry: dispatch.json `registry`, else the first pi lane's serving:<name>."""
        if self.cfg.get('registry'):
            return self.cfg['registry']
        for lane in self.lanes.values():
            spec = str((getattr(lane, 'cfg', None) or {}).get('model_spec') or '')
            if spec.startswith('serving:'):
                return spec.split(':', 1)[1]
        return None

    def gate_queued(self) -> int | None:
        """New conversations the model's admission gate holds: the minimum over replicas (registry gate_status_urls).
        None: no gate (registry without gate_status_urls), --gate-backoff off, or no replica answered."""
        if self.cfg.get('gate_backoff', 'queued_new') == 'off':
            return None
        name = self.registry()
        if not name:
            return None
        urls = (read_json(self.registry_root() / name / 'endpoints.json', {}) or {}).get('gate_status_urls') or []
        if not urls:
            return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        held = []
        for u in urls:
            try:
                with opener.open(u, timeout=GATE_TIMEOUT) as r:
                    held.append(int(json.load(r).get('queued_new') or 0))
            except (OSError, ValueError, TypeError):
                continue
        # back off only when every replica holds: one holding replica must not stop credit for the others
        return min(held) if held else None

    def auto_on(self, ctl: dict | None = None) -> bool:
        """Auto budget (#378): dispatch.json auto_budget (new dirs, unless --no-auto-budget) with --metrics, and no
        manual budget in control.json (`budget DIR N` turns it off, `budget DIR auto` back on)."""
        if not (self.cfg.get('auto_budget') and self.cfg.get('metrics')):
            return False
        if ctl is None:
            ctl = read_json(self.d / 'control.json', {}) or {}
        return bool(ctl.get('auto', 'budget' not in ctl))

    def max_num_seqs(self) -> int | None:
        """The engine's --max-num-seqs (per engine): dispatch.json, else the registry's launch.json argv (cached)."""
        if self.cfg.get('max_num_seqs'):
            return int(self.cfg['max_num_seqs'])
        if not getattr(self, '_mns', None) and self.registry() and time.time() - getattr(self, '_mns_tried', 0) > 300:
            self._mns_tried = time.time()
            launch = read_json(self.registry_root() / self.registry() / 'launch.json', {}) or {}
            vals = []
            for rep in launch.get('replicas') or []:
                argv = [str(x) for x in rep.get('argv') or []]
                for i, x in enumerate(argv):
                    try:
                        if x == '--max-num-seqs' and i + 1 < len(argv):
                            vals.append(int(argv[i + 1]))
                        elif x.startswith('--max-num-seqs='):
                            vals.append(int(x.split('=', 1)[1]))
                    except ValueError:
                        pass
            self._mns = max(vals) if vals else None   # not known yet (no registry / launch.json): tried again later
        return getattr(self, '_mns', None)

    def auto_budget(self) -> autobudget.AutoBudget:
        if self.auto is None:
            b0 = int(self.cfg['budget'])
            lo = int(self.cfg.get('budget_min') or min(b0, max(int(self.cfg.get('floor') or 16), b0 // 2)))
            hi = int(self.cfg.get('budget_max') or max(2 * b0, lo))
            self.auto = autobudget.AutoBudget(b0, lo, hi, self.max_num_seqs(), time.time(),
                                              read_json(self.d / 'auto_budget.json'))
        return self.auto

    def budget(self) -> int:
        ctl = read_json(self.d / 'control.json', {}) or {}
        if self.auto_on(ctl):
            return self.auto_budget().budget
        return int(ctl.get('budget', self.cfg['budget']))

    def scrape(self) -> str | None:
        """The engine's /metrics text (--metrics), None when it does not answer."""
        url = self.cfg.get('metrics')
        if not url:
            return None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(url.rstrip('/').removesuffix('/v1') + '/metrics', timeout=10).read().decode()
        except OSError:
            return None

    def adaptive(self, budget: int, in_flight: int, text: str | None = None) -> int:
        if not self.cfg.get('metrics'):
            return budget
        if text is None:
            return min(budget, self.target or budget)
        if 'auto_budget' in self.cfg:   # #378 dirs: KV of the fullest engine (a sum over a DP server's engines
            mm = autobudget.parse_metrics(text, strict=False)   # backed off at half full)
            kv, waiting, pre = mm['kv'], mm['waiting'], mm['preempt_total']
        else:                           # older dirs: as they were
            m = lambda name: sum(float(ln.split()[-1]) for ln in text.splitlines() if ln.startswith(name + '{'))
            kv, waiting = m('vllm:kv_cache_usage_perc'), m('vllm:num_requests_waiting')
            pre = m('vllm:num_preemptions_total')
        dp = 0 if self.pre is None else pre - self.pre
        self.pre = pre
        t = self.target or budget
        if dp > 0 or (waiting > 0 and kv >= 0.95):
            t = max(int(self.cfg.get('floor') or 16), in_flight - 1)
        elif kv < 0.85 and waiting == 0 and in_flight >= t - 1:
            t = t + 2
        self.target = min(budget, t)
        self.metrics_seen = {'kv': round(kv, 3), 'waiting': waiting, 'preempt_delta': dp}
        return self.target

    def record_events(self, snaps: dict, now: float) -> None:
        new_events = []
        for name, s in snaps.items():
            suite = self.lanes[name].suite
            for e in s['events']:
                e = {**e, 'lane': name, 'suite': suite}
                key = (name, e['key'])
                if e['ev'] == 'start':
                    p = self.dur.predict(suite, e['item'])
                    f = self.current.get(suite, self.current['*'])
                    self.pred_at_start[key] = p * f
                    self.items[key] = str(e['item'])
                    e.update(pred_table=p, pred=round(p * f, 1))
                else:
                    e.update(pred_table=self.dur.predict(suite, e['item']), pred=self.pred_at_start.pop(key, None))
                    self.items.pop(key, None)
                    if e.get('sec') and str(e.get('status', 'ok')) == 'ok':
                        self.factors.add(suite, float(e['sec']), e['pred_table'])
                if (name, e['key'], e['ev']) not in self.seen:
                    self.seen.add((name, e['key'], e['ev']))
                    new_events.append(e)
        if new_events:
            with open(self.d / 'events.jsonl', 'a') as f:
                f.write(''.join(json.dumps(e, separators=(',', ':')) + '\n' for e in new_events))
        censored = defaultdict(list)
        pre_f = self.factors.factors()
        for name, s in snaps.items():
            suite = self.lanes[name].suite
            fs = pre_f.get(suite, pre_f['*'])
            for k, t0 in s['running'].items():
                p = self.dur.predict(suite, self.items.get((name, k), k))
                if now - t0 > p * fs:
                    censored[suite].append(math.log((now - t0) / p))
        self.current = self.factors.factors(censored)

    def tick(self) -> dict:
        now = time.time()
        self.discover()
        snaps = {}
        for name, lane in self.lanes.items():
            try:
                snaps[name] = lane.snapshot(now)
            except (OSError, ValueError) as e:
                log(self.d, f'lane {name}: snapshot failed ({e!r})')
        self.record_events(snaps, now)
        for n, s in snaps.items():
            lane = self.lanes[n]
            if s['dead']:
                if not getattr(lane, 'dead_logged', False):
                    lane.dead_logged = True
                    log(self.d, f'WARNING lane {n}: client gone ({s["reason"]}): its slots are released')
                if lane.credit > s['starts']:   # unspent credit of a dead lane is withdrawn (pi: claims so far)
                    lane.write_control(s['starts'], int(self.cfg.get('burst') or 16), 'dispatch: lane dead')
                    lane.credit = s['starts']
            else:
                lane.dead_logged = False
            if s.get('idle_exited') and lane.credit > s['starts']:
                # #378: credit granted just as its last worker left has no one to spend it: back to the budget (the
                # lane grows back through its shortfall)
                lane.write_control(s['starts'], int(self.cfg.get('burst') or 16), 'dispatch: workers left idle')
                lane.credit = s['starts']
            if (lane.kind == 'pi' and lane.cfg.get('lazy_workers') and not s.get('lazy') and not s['dead']
                    and not s.get('gate_paused')):
                # a launch reservation past the start-up grace: no more credit than its live workers can spend
                first = lane.first_submitted()
                cap = s['starts'] + max(0, int(s.get('worker_capacity', s['capacity']) or 0) - s['in_flight'])
                if first is not None and now - first > float(self.cfg.get('ready_timeout') or READY_TIMEOUT) \
                        and lane.credit > max(s['starts'], cap):
                    c = max(s['starts'], cap)
                    log(self.d, f'lane {n}: credit {lane.credit} trimmed to {c} (its starts + live workers\' free '
                                f'slots): first workers submitted {now - first:.0f}s ago')
                    lane.write_control(c, int(self.cfg.get('burst') or 16), 'dispatch: launch reservation trimmed')
                    lane.credit = c
            # since when has this lane's unspent credit not been spent (reset whenever it starts something)
            if lane.credit <= s['starts']:
                lane.unspent_since = None
            elif getattr(lane, 'unspent_since', None) is None or s['starts'] > getattr(lane, 'last_starts', -1):
                lane.unspent_since = now
            if getattr(lane, 'withdrawn', False) and (s['starts'] > getattr(lane, 'last_starts', -1)
                                                       or not (s.get('gate_paused') or s.get('lazy'))):
                lane.withdrawn = False   # it started something, or its workers see the gate clear (a lazy lane whose
                # launch reservation was withdrawn gets no new one until its first submission is recorded)
            # the reserve_sec withdrawal covers a gate reservation and a lazy lane before its first submission is
            # recorded; after it, the start-up grace above bounds the launch reservation (no carry-over)
            keep = getattr(lane, 'reserved', False) and lane.credit > s['starts'] and not getattr(lane, 'reserved_lazy', False)
            lane.reserved = bool(s.get('gate_paused')) or bool(s.get('lazy')) or keep
            lane.reserved_lazy = bool(s.get('lazy')) and not s.get('gate_paused')
            lane.last_starts = s['starts']
        ctl = read_json(self.d / 'control.json', {}) or {}
        auto = self.auto_on(ctl)
        tick_sec = float(self.cfg.get('tick_sec') or TICK_SEC)
        if auto:
            a = self.auto_budget()
            if a.max_num_seqs is None:   # the registry may be known only once a pi lane registered
                a.max_num_seqs = self.max_num_seqs()
            if not self.auto_was_on:   # (re)enabled: judge only what follows
                a.t_step, a.samples, a.last, a.prev = now, [], None, None
            old, (b, why) = a.budget, a.step(now, tick_sec)
            if why and (b != old or why.split('(')[0] != self.auto_logged):
                log(self.d, f'auto budget {why}')
                self.auto_logged = why.split('(')[0]
            if why:
                atomic_json(self.d / 'auto_budget.json', {**a.to_json(), 'epoch': round(now, 1)})
        self.auto_was_on = auto
        budget = self.budget()
        in_flight = sum(s['in_flight'] for s in snaps.values())
        text = self.scrape()
        eff = self.adaptive(budget, in_flight, text)
        expect = int(self.cfg.get('expect_lanes') or 0)
        ready = [n for n, s in snaps.items() if s['ready'] or s['done']]
        live = [n for n, s in snaps.items() if not s['dead']]
        if not self.released:
            if (len(ready) >= max(1, expect) and set(ready) >= set(live)) or now - self.t0 > self.cfg.get(
                    'ready_timeout', READY_TIMEOUT):
                self.released = True
                log(self.d, f'released: {len(ready)} lane(s) ready of {len(snaps)} (expected {expect})')
        views = {}
        for n, s in snaps.items():
            lane = self.lanes[n]
            f = self.current.get(lane.suite, self.current['*'])
            views[n] = {'in_flight': s['in_flight'], 'starts': s['starts'], 'credit': lane.credit,
                        'capacity': s['capacity'], 'gate_paused': s.get('gate_paused', False), 'kind': lane.kind,
                        'gate_partial': s.get('gate_partial', False), 'shrinkable': s.get('shrinkable', False),
                        'worker_capacity': s.get('worker_capacity', s['capacity']), 'dead': s['dead'],
                        'withdrawable': lane.kind == 'pi', 'unspent_since': getattr(lane, 'unspent_since', None),
                        'withdrawn': getattr(lane, 'withdrawn', False), 'reserved': getattr(lane, 'reserved', False),
                        'scalable': lane.kind == 'pi' and bool(self.cfg.get('pi_autoscale')) and not s['paused']
                        and lane.stop_words() is None and not (self.d / 'lanes' / n / 'NO_HOLD').exists(),
                        'idle_exited': bool(s.get('idle_exited')),
                        'revivable': bool(s.get('revivable')), 'lazy_reserve': int(s.get('lazy_reserve') or 0),
                        'pending': [(self.dur.predict(lane.suite, item) * f, w) for _, item, w in s['pending']]}
        gq = self.gate_queued()
        grants, withdraw, info = decide(views, eff, gq, now, self.released, int(self.cfg.get('gate_hold_min') or 1),
                                        float(self.cfg.get('reserve_sec') or RESERVE_SEC),
                                        flowing=now - getattr(self, 'last_grant_t', 0) < SCALE_COOLDOWN / 2,
                                        st_hold_on=self.cfg.get('single_turn_hold', 'on') != 'off')
        if any(grants.values()):
            self.last_grant_t = now
        if info['gate_hold'] != getattr(self, 'gate_hold', False):
            log(self.d, f'gate {"holds " + str(gq) + " new conversation(s): no new credit" if info["gate_hold"] else "clear"}')
        self.gate_hold = info['gate_hold']
        if info['st_hold'] != getattr(self, 'st_hold', False):
            log(self.d, 'single-turn hold ' + ('on: a pi lane waits on the gate: unspent single-turn credit withdrawn'
                                                if info['st_hold'] else 'off'))
        self.st_hold = info['st_hold']
        for n, c in withdraw.items():
            lane = self.lanes[n]
            if lane.kind == 'lane':   # #378 single-turn hold
                log(self.d, f'lane {n}: credit {lane.credit} withdrawn to its starts {c} (a pi lane waits on the gate)')
                lane.write_control(c, int(self.cfg.get('burst') or 16), 'dispatch: single-turn hold')
                lane.credit, lane.unspent_since, lane.last_write = c, None, now
                continue
            why = 'first workers not submitted' if snaps.get(n, {}).get('lazy') else 'gate not holding'
            log(self.d, f'lane {n}: credit {lane.credit} unspent for > '
                        f'{float(self.cfg.get("reserve_sec") or RESERVE_SEC):.0f}s, {why}: withdrawn to {c}')
            lane.write_control(c, int(self.cfg.get('burst') or 16), 'dispatch: unspent credit withdrawn')
            lane.credit, lane.unspent_since, lane.withdrawn = c, None, True
        used = info['used']
        outstanding = {n: max(0, self.lanes[n].credit - s['starts']) if not s['dead'] else 0 for n, s in snaps.items()}
        coverage = {}
        for n, s in snaps.items():   # share of pending items with no table prediction (silent fallbacks)
            suite = self.lanes[n].suite
            items = {item for _, item, _ in s['pending']}
            miss = sum(1 for i in items if not self.dur.known(suite, i)) / len(items) if items else 0.0
            coverage[n] = round(miss, 3)
            if miss > MISSING_WARN and not getattr(self.lanes[n], 'warned', False):
                self.lanes[n].warned = True
                log(self.d, f'WARNING lane {n}: {miss:.0%} of pending items have no prediction in suite {suite!r} '
                            '(fallback: suite median / default); check --dispatch-suite')
        burst = int(self.cfg.get('burst') or 16)
        why = f'dispatch budget={eff} in_flight={in_flight}'
        for n, s in snaps.items():
            lane = self.lanes[n]
            credit = max(lane.credit, s['starts']) + grants[n]
            if credit != lane.credit or now - getattr(lane, 'last_write', 0) > REWRITE_SEC:
                lane.write_control(credit, burst, why)
                lane.credit, lane.last_write = credit, now
        if auto:   # one engine sample for the auto budget: was the budget filled, with work still pending?
            used_now = sum(used.values()) + sum(grants.values())
            pending = sum(s['n_pending'] for s in snaps.values() if not s['dead'])
            filled = (self.released and used_now >= eff - max(2, 0.05 * eff)
                      and pending > sum(outstanding.values()) + sum(grants.values()))
            try:
                m = autobudget.parse_metrics(text) if text is not None else None
            except ValueError:
                m = None
            self.auto.sample(now, m, filled, info['gate_hold'])
        self.first_workers(snaps, grants, now)
        if self.cfg.get('pi_autoscale'):
            self.grow_workers(snaps, info.get('shortfall') or {}, outstanding, grants, now)
        if self.cfg.get('release_server_after'):
            self.maybe_release_server(snaps, in_flight, now)
        alloc = {n: grants[n] for n in snaps}
        if alloc != self.last_alloc and any(alloc.values()):
            log(self.d, f'budget {eff} in flight {in_flight} used {sum(used.values())}: grants {alloc}')
        self.last_alloc = alloc
        head = []
        for n, s in snaps.items():
            suite = self.lanes[n].suite
            f = self.current.get(suite, self.current['*'])
            for k, item, _ in s['pending'][:5]:
                head.append({'lane': n, 'key': k, 'pred': round(self.dur.predict(suite, item) * f)})
        head.sort(key=lambda x: -x['pred'])
        state = {'epoch': round(now, 1), 'budget': budget, 'effective_budget': eff, 'in_flight': in_flight,
                 'auto_budget': self.auto.to_json() if auto else None,
                 'gate_queued': gq, 'gate_hold': info['gate_hold'],
                 'st_hold': info['st_hold'],
                 'used': sum(used.values()) + sum(grants.values()), 'released': self.released, 'metrics': getattr(self, 'metrics_seen', None),
                 'factors': {k: round(v, 3) for k, v in self.current.items()},
                 'factor_n': dict(self.factors.n),
                 'lanes': {n: {'suite': self.lanes[n].suite, 'kind': self.lanes[n].kind, 'in_flight': s['in_flight'],
                               'starts': s['starts'], 'credit': self.lanes[n].credit,
                               'outstanding': outstanding[n] + grants[n], 'grant': grants[n],
                               'n_pending': s['n_pending'],
                               'capacity': s['capacity'] if s['capacity'] < 10 ** 6 else None, 'ready': s['ready'],
                               'paused': s['paused'], 'reason': s['reason'], 'done': s['done'], 'dead': s['dead'],
                               'gate_paused': s.get('gate_paused', False),
                               'gate_partial': s.get('gate_partial', False),
                               'lazy': bool(s.get('lazy')),   # no worker yet: its first grant submits them
                               'no_prediction_share': coverage[n],
                               'warning': ('no prediction for ' + f'{coverage[n]:.0%} of pending items')
                               if coverage[n] > MISSING_WARN else ('client gone' if s['dead'] else None)}
                           for n, s in snaps.items()},
                 'head': head[:10]}
        atomic_json(self.d / 'state.json', state)
        self.snaps = snaps
        return state

    def first_workers(self, snaps: dict, grants: dict, now: float) -> None:
        """Lazy lanes (pi_eval --lazy-workers): submit a run's first workers (its configured worker.n: --workers, or
        launch-small) once it holds credit, in a background thread; retried after LAZY_RETRY_SEC if it fails."""
        import threading
        for n, s in snaps.items():
            lane = self.lanes[n]
            if lane.kind != 'pi' or not s.get('lazy') or s['dead']:
                continue
            if grants.get(n, 0) <= 0 and lane.credit <= s['starts']:
                continue
            if getattr(lane, 'first_submitting', False) or now - getattr(lane, 'lazy_tried', 0) < LAZY_RETRY_SEC:
                continue   # a submission still in flight (a slow schedd) never starts a second first fleet
            lane.lazy_tried, lane.first_submitting = now, True
            threading.Thread(target=self._first_one, args=(n, lane), daemon=True).start()

    def _first_one(self, n, lane) -> None:
        try:
            from aa_evals.common.pi_eval.run import submit_workers
            cfg = read_json(lane.run / 'config.json', {}) or {}
            k = max(1, int((cfg.get('worker') or {}).get('n') or 1))
            cluster = submit_workers(lane.run, cfg, k)
            log(self.d, f'lane {n}: first credit: submitted its first {k} worker(s) x {lane.default_slots} slots '
                        f'(lazy workers; cluster {cluster})')
        except BaseException as e:   # retried after LAZY_RETRY_SEC; the lane keeps its probe slot meanwhile
            log(self.d, f'lane {n}: lazy first worker submission failed ({e!r})')
        finally:
            lane.first_submitting = False

    def grow_workers(self, snaps: dict, shortfall: dict, outstanding: dict, grants: dict, now: float) -> None:
        """Pi worker growth (#378): submit workers for a lane whose slots, not the budget, limit it (pi_eval
        submit_workers, the run's stored worker shape and bid). The condor calls run in a background thread, one at a
        time per lane, so a slow schedd never stalls the tick. A lane grows only after its own first worker
        submission is recorded (RUN/jobs.json older than 2 ticks): runs made with --no-submit, or still submitting,
        are left alone. Added workers stay until their lane's queue is fully claimed (eval_q35 idle exit), or, with
        pi_idle_exit_sec (#378), until they sit idle that long with no start credit; growth is capped at the budget per
        lane. An idle-exited lane has no capacity, so the credit it would take shows as shortfall and it grows back."""
        import threading
        tick = float(self.cfg.get('tick_sec') or TICK_SEC)
        for n, s in snaps.items():
            lane = self.lanes[n]
            if lane.kind != 'pi':
                continue
            short = int(shortfall.get(n) or 0)
            lane.short_ticks = getattr(lane, 'short_ticks', 0) + 1 if short > 0 else 0
            if short <= 0 or lane.short_ticks < SCALE_TICKS or getattr(lane, 'growing', False):
                continue
            last = getattr(lane, 'last_scale', None)
            if last is not None and now - last < SCALE_COOLDOWN:
                continue
            first = min((iso_epoch(j.get('submitted_at')) or now for j in read_json(lane.run / 'jobs.json', []) or []),
                        default=None)
            if first is None or now - first < 2 * tick:
                continue
            lane.growing, lane.last_scale = True, now
            args = (n, lane, short, s['in_flight'], outstanding.get(n, 0) + grants.get(n, 0), s['n_pending'],
                    lane.short_ticks, self.budget())
            threading.Thread(target=self._grow_one, args=args, daemon=True).start()
        ladder = any(l.kind == 'pi' and 'bid_ladder' in (l.cfg.get('worker') or {}) for l in self.lanes.values())
        if ladder and now - getattr(self, 'last_watch_check', 0) > SCALE_COOLDOWN:
            self.last_watch_check = now   # keep the CPU bid-ladder watcher alive while ladder workers exist
            threading.Thread(target=self._ensure_watcher, daemon=True).start()

    def _grow_one(self, n, lane, short, in_flight, outstanding, n_pending, short_ticks, budget) -> None:
        try:
            slots = lane.default_slots
            job_slots = alive_worker_slots(lane)
            add = workers_to_add(short, in_flight, outstanding, n_pending, job_slots, slots, short_ticks, None, budget)
            if add > 0:
                from aa_evals.common.pi_eval.run import submit_workers
                cfg = read_json(lane.run / 'config.json', {}) or {}
                cluster = submit_workers(lane.run, cfg, add)
                log(self.d, f'lane {n}: slot-bound (shortfall {short}, {job_slots} worker slots queued/running): '
                            f'+{add} worker(s) x {slots} slots (cluster {cluster})')
        except BaseException as e:   # growth is best effort; the lane keeps its current capacity
            log(self.d, f'lane {n}: worker growth failed ({e!r})')
        finally:
            lane.growing = False

    def maybe_release_server(self, snaps: dict, in_flight: int, now: float) -> None:
        """#378: release the model's serving job once nothing has been pending or in flight on it for
        `release_server_after` s (dispatch.json; set by `init --release-server-after`, off by default because a
        registry can be shared): every expected lane registered, none with a pending or in-flight attempt, unspent
        credit or `hold_server` (a lane that will send the model more work after CPU work in progress, e.g. a chain's
        next step after grading), at least one attempt started, and every engine of the registry running and queueing
        0 requests (an engine that does not answer counts as busy). A lane need not be `done`: a SciCode step hanging
        in CPU grading kept 2 B200s idle ~1 h (sb2-0p33b, 11-18; eval/release_when_idle.sh). A dead lane with work left
        blocks the release and is logged once as STUCK (release_when_idle v1 released sb2-2p2b with a hung client's
        830 attempts unrun). The job is the registry's current job.json job, removed only if it is ours and its batch
        name carries the registry name. One attempt per dispatcher run."""
        expect = int(self.cfg.get('expect_lanes') or 0)
        if getattr(self, 'server_release_tried', False):
            return
        # an operator lets a lane go (DIR/lanes/<lane>/NO_HOLD: e.g. a client killed to drop a suite)
        let_go = {n for n in snaps if (self.d / 'lanes' / n / 'NO_HOLD').exists()}
        considered = {n: s for n, s in snaps.items() if n not in let_go}
        stuck = sorted(n for n, s in considered.items() if s['dead'] and not s.get('stopped')
                       and ((s.get('n_pending') or 0) > 0 or (s.get('work_left') or 0) > 0))
        if stuck != getattr(self, 'stuck_logged', []):
            if stuck:
                log(self.d, f'STUCK: dead lane(s) with work left {stuck}: not releasing the server')
            self.stuck_logged = stuck
        idle = (self.released and snaps and len(snaps) >= max(1, expect) and in_flight == 0 and not stuck
                and all(s.get('stopped') and s['dead'] or (s.get('n_pending') or 0) == 0 and s['in_flight'] == 0
                        and not s.get('hold_server') and not s.get('unannounced') for s in considered.values())
                and all(self.lanes[n].credit <= s['starts'] for n, s in snaps.items())
                and any(s['starts'] for s in snaps.values()))
        if idle:
            idle = self.engines_idle() is True
        if not idle:
            self.idle_since = None
            return
        self.idle_since = getattr(self, 'idle_since', None) or now
        if now - self.idle_since < float(self.cfg['release_server_after']):
            return
        self.server_release_tried = True
        import threading
        threading.Thread(target=self._release_server, args=(self.registry(),), daemon=True).start()

    def engines_idle(self) -> bool | None:
        """True when every engine of the registry runs and queues 0 requests; None when the registry or an engine
        cannot be read (not idle)."""
        name = self.registry()
        ep = read_json(self.registry_root() / name / 'endpoints.json', None) if name else None
        urls = (ep or {}).get('engine_urls') or (ep or {}).get('urls') or []
        if not urls:
            return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for u in urls:
            try:
                text = opener.open(u.rstrip('/').removesuffix('/v1') + '/metrics', timeout=10).read().decode()
                busy = sum(float(line.split()[-1]) for line in text.splitlines()
                           if line.startswith(('vllm:num_requests_running', 'vllm:num_requests_waiting')))
            except Exception:   # noqa: BLE001 -- unreadable (any error) counts as busy: never a wrong release
                return None
            if busy > 0:
                return False
        return True

    def _release_server(self, name: str | None, runner=None) -> None:
        import getpass
        import subprocess
        run = runner or subprocess.run
        try:
            if not name:
                raise RuntimeError('no serving registry known (init --registry)')
            job = str((read_json(self.registry_root() / name / 'job.json', {}) or {}).get('job_id') or '')
            if not job:
                raise RuntimeError(f'{name}: no job.json job_id')
            ad = run(['condor_q', job, '-af', 'Owner', 'JobBatchName', 'JobStatus'], capture_output=True, text=True,
                     timeout=120).stdout.split()
            if len(ad) < 3:
                log(self.d, f'server {name} ({job}): not in the queue; nothing to release')
                return
            if ad[0] != getpass.getuser() or name not in ad[1]:
                raise RuntimeError(f'refusing to remove {job}: {ad}')
            r = run(['condor_rm', job], capture_output=True, text=True, timeout=120)
            log(self.d, f'nothing pending or in flight {self.cfg["release_server_after"]:.0f}s: released server {name} '
                        f'({job}): {(r.stdout or r.stderr or "").strip()[:200]}')
        except BaseException as e:
            log(self.d, f'server release failed ({e!r})')

    def _ensure_watcher(self) -> None:
        try:
            from aa_evals.common import condor_cost
            condor_cost.ensure_watcher()
        except BaseException as e:
            log(self.d, f'bid-ladder watcher check failed ({e!r})')

    def lock(self) -> None:
        """One dispatcher per DIR: DIR/dispatcher.lock (host, pid). A lock of a dead pid on this host, or of another
        host whose state.json is over 2 min old, is taken over."""
        p = self.d / 'dispatcher.lock'
        me = {'host': socket.gethostname(), 'pid': os.getpid(), 'epoch': round(time.time(), 1)}
        old = read_json(p)
        if old and (old.get('host'), old.get('pid')) != (me['host'], me['pid']):
            st = read_json(self.d / 'state.json', {}) or {}
            if old.get('host') == me['host']:
                live = pid_alive(old.get('pid'))
            else:
                live = time.time() - float(st.get('epoch') or 0) < 120
            if live:
                raise SystemExit(f'{self.d}: a dispatcher is already running (pid {old.get("pid")} on {old.get("host")})')
            log(self.d, f'taking over the lock of a dead dispatcher (pid {old.get("pid")} on {old.get("host")})')
        if old:
            atomic_json(p, me)   # take-over of a dead lock (checked above)
        else:
            try:                 # first start: O_EXCL, so two simultaneous starts cannot both win
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                raise SystemExit(f'{self.d}: another dispatcher started at the same time') from None
            with os.fdopen(fd, 'w') as f:
                json.dump(me, f)

    def owns_lock(self) -> bool:
        return (read_json(self.d / 'dispatcher.lock', {}) or {}).get('pid') == os.getpid() and (
            read_json(self.d / 'dispatcher.lock', {}) or {}).get('host') == socket.gethostname()

    def run(self) -> None:
        self.lock()
        log(self.d, f'dispatcher up: budget {self.budget()}, released {self.released}, '
                    f'durations {self.dur.describe()}')
        try:
            self.loop()
        finally:
            if self.owns_lock():
                (self.d / 'dispatcher.lock').unlink(missing_ok=True)

    def loop(self) -> None:
        while not (self.d / 'STOP').exists():
            if not self.owns_lock():   # another dispatcher took over (this one was stalled): stop writing credits
                log(self.d, f'lost dispatcher.lock to {read_json(self.d / "dispatcher.lock")}; exiting')
                return
            try:
                st = self.tick()
                expect = int(self.cfg.get('expect_lanes') or 0)
                drained = bool(self.lanes) and len(self.lanes) >= expect and all(
                    v['done'] for v in st['lanes'].values())
                if drained and not self.drained_logged:
                    log(self.d, 'all registered lanes drained (still running: touch DIR/STOP to exit)')
                self.drained_logged = drained
            except Exception as e:   # noqa: BLE001 -- keep dispatching through transient FS errors
                log(self.d, f'tick failed: {e!r}')
            time.sleep(float(self.cfg.get('tick_sec') or TICK_SEC))
        log(self.d, 'STOP file: exiting (credits stay as last written)')


# ---------------------------------------------------------------------------------------------- CLI

def cmd_init(a) -> None:
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'lanes').mkdir(exist_ok=True)
    if (d / 'dispatch.json').exists() and not a.force:
        raise SystemExit(f'{d}/dispatch.json exists (--force to overwrite)')
    if a.force:   # a reused dir starts a new run: not released, no budget override, no auto budget state
        for name in ('state.json', 'control.json', 'auto_budget.json'):
            (d / name).unlink(missing_ok=True)
    from aa_evals.common import serving
    dur = Durations(a.durations, a.model)
    cfg = {'created_epoch': round(time.time(), 1), 'budget': a.budget, 'model': a.model,
           'durations': str(dur.path), 'durations_sha256': dur.sha256, 'expect_lanes': a.expect_lanes,
           'burst': a.burst, 'tick_sec': a.tick_sec, 'metrics': a.metrics, 'floor': a.floor,
           'ready_timeout': a.ready_timeout, 'notes': a.notes, 'registry': a.registry, 'gate_backoff': a.gate_backoff,
           'gate_hold_min': a.gate_hold_min, 'reserve_sec': a.reserve_sec,
           'single_turn_hold': a.single_turn_hold,   # #378; absent in older dirs: on
           'pi_autoscale': a.pi_autoscale,   # absent in older dirs: off
           # #378, absent in older dirs: off. Workers of a growing dispatcher's pi lanes leave after this long idle
           # with no start credit; the dispatcher grows them back when credit returns
           'pi_idle_exit_sec': a.pi_idle_exit_sec if a.pi_autoscale else 0,
           # #378, absent in older dirs: off. With --metrics, the budget is a start value the dispatcher moves
           'auto_budget': a.auto_budget, 'budget_min': a.budget_min, 'budget_max': a.budget_max,
           'max_num_seqs': a.max_num_seqs,
           'lazy_workers': a.lazy_workers,   # new pi runs default to lazy first workers; absent in older dirs: eager
           'release_server_after': a.release_server_after,
           # the registry root this run reads ({} = production), for the dispatcher's own reads and release
           'serving_env': serving.forwarded()}
    atomic_json(d / 'dispatch.json', cfg)
    print(json.dumps(cfg, indent=1))


def cmd_status(a) -> None:
    d = Path(a.dir)
    st = read_json(d / 'state.json')
    if not st:
        raise SystemExit('no state.json yet (dispatcher not running?)')
    age = time.time() - st['epoch']
    ab = st.get('auto_budget')
    if ab:
        print(f"auto budget {ab['budget']} in [{ab['lo']}, {ab['hi']}] (max-num-seqs {ab.get('max_num_seqs')}): "
              f"{ab.get('note') or 'first window'}")
    print(f"budget {st['effective_budget']}/{st['budget']}  in flight {st['in_flight']}  "
          f"released {st['released']}  state age {age:.0f}s{'  (STALE: dispatcher down?)' if age > 120 else ''}")
    if st.get('metrics'):
        print(f"engine: {st['metrics']}")
    print(f"budget used (in flight + unspent credit): {st.get('used')}")
    print(f"{'lane':<32}{'kind':<6}{'in fl':>6}{'unspent':>8}{'pend':>7}{'cap':>6}  state")
    for n, v in sorted(st['lanes'].items()):
        flag = 'done' if v['done'] else ('paused: ' + str(v['reason'])) if v['paused'] else 'ready' if v['ready'] else 'not ready'
        print(f"{n:<32}{v['kind']:<6}{v['in_flight']:>6}{v['outstanding']:>8}{v['n_pending']:>7}"
              f"{v['capacity'] if v['capacity'] is not None else '-':>6}  {flag}"
              + (f"  WARNING: {v['warning']}" if v.get('warning') else ''))
    print('factors (actual / table):', {k: v for k, v in st['factors'].items()}, 'n', st.get('factor_n'))
    print('queue head:', ', '.join(f"{h['lane']}:{h['key']} {h['pred']}s" for h in st['head']))


def cmd_report(a) -> None:
    """Per-attempt rows (lane, suite, key, item, start, end, sec, pred, status) from DIR/events.jsonl."""
    d = Path(a.dir)
    start, rows = {}, []
    for e in jsonl.read(d / 'events.jsonl', required=True):
        k = (e['lane'], e['key'])
        if e['ev'] == 'start':
            start[k] = e
        else:
            s = start.pop(k, {})
            rows.append({'lane': e['lane'], 'suite': e['suite'], 'key': e['key'], 'item': e['item'],
                         'start': s.get('t'), 'end': e['t'], 'sec': e.get('sec'), 'pred': s.get('pred'),
                         'pred_table': e.get('pred_table'), 'status': e.get('status'), 'end_state': e.get('end_state')})
    out = open(a.out, 'w') if a.out else None
    for r in rows:
        if out:
            out.write(json.dumps(r) + '\n')
    t = [r for r in rows if r['start']]
    if t:
        t0, t1 = min(r['start'] for r in t), max(r['end'] for r in t)
        print(f'{len(rows)} finished attempts ({len(start)} still running), span {(t1 - t0) / 3600:.2f} h')
        by = Counter(r['suite'] for r in rows)
        print('by suite:', dict(by))
        print('statuses:', dict(Counter(str(r['status']).split(':')[0] for r in rows)))


def simulate(preds: list[float], budget: int) -> float:
    """Makespan (s) of list scheduling on `budget` slots, attempts in the given order (no slowdown with load)."""
    slots = [0.0] * max(1, budget)
    heapq.heapify(slots)
    end = 0.0
    for p in preds:
        t = heapq.heappop(slots) + p
        end = max(end, t)
        heapq.heappush(slots, t)
    return end


def cmd_plan(a) -> None:
    """A c30 plan file (suites.<suite>.items = [{id, repeats}]): attempts, table coverage and predicted attempt-hours
    per suite, and the longest-first makespan on --budget slots with the --cap (list scheduling at the table's
    recorded speed; the #344 replay adds load slowdown). --ids-out DIR writes <suite>.ids (one id per line)."""
    plan = json.load(open(a.plan))
    dur = Durations(a.durations, a.model)
    rows, preds = {}, []
    for suite, v in plan['suites'].items():
        known = dur.table['suites'].get(suite) or {}
        ps = [min(dur.predict(suite, it['id']), a.cap) for it in v['items'] for _ in range(int(it['repeats']))]
        preds += ps
        rows[suite] = {'items': len(v['items']), 'attempts': len(ps),
                       'in_table': sum(1 for it in v['items'] if str(it['id']) in known),
                       'attempt_h': round(sum(ps) / 3600, 1), 'max_pred_h': round(max(ps, default=0) / 3600, 2)}
        if a.ids_out:
            Path(a.ids_out).mkdir(parents=True, exist_ok=True)
            (Path(a.ids_out) / f'{suite}.ids').write_text(''.join(f"{it['id']}\n" for it in v['items']))
    for suite, r in rows.items():
        print(f"{suite:<16} items {r['items']:>4} (in table {r['in_table']:>4})  attempts {r['attempts']:>5}  "
              f"{r['attempt_h']:>7.1f} attempt-h  longest {r['max_pred_h']:.2f} h")
    work = sum(preds)
    span = simulate(sorted(preds, reverse=True), a.budget)
    print(f"total {len(preds)} attempts, {work / 3600:.1f} attempt-h; budget {a.budget}: work/budget "
          f"{work / a.budget / 3600:.2f} h, longest-first makespan {span / 3600:.2f} h (cap {a.cap:g} s)")


def cmd_budget(a) -> None:
    """`budget DIR N`: a fixed budget from now on (also turns the auto budget off); `budget DIR auto`: auto again."""
    d = Path(a.dir)
    if a.n == 'auto':
        cfg = read_json(d / 'dispatch.json', {}) or {}
        if not (cfg.get('auto_budget') and cfg.get('metrics')):
            raise SystemExit('this dispatch dir has no auto budget (init without --metrics, --no-auto-budget, or '
                             'made before #378)')
        atomic_json(d / 'control.json', {'auto': True, 'epoch': round(time.time(), 1)})
        print('budget -> auto')
        return
    atomic_json(d / 'control.json', {'budget': int(a.n), 'auto': False, 'epoch': round(time.time(), 1)})
    print(f'budget -> {int(a.n)}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('dir')
    p.add_argument('--budget', type=int, required=True, help="the server's measured knee (attempts in flight); with "
                   "--metrics and the auto budget (default), the start value")
    p.add_argument('--auto-budget', action=argparse.BooleanOptionalAction, default=True,
                   help='with --metrics: hill-climb the budget on the engine\'s generated tok/s (#378, autobudget.py)')
    p.add_argument('--budget-min', type=int, default=None, help='auto budget floor (default: budget / 2, >= --floor)')
    p.add_argument('--budget-max', type=int, default=None, help='auto budget ceiling (default: 2 x budget)')
    p.add_argument('--max-num-seqs', type=int, default=None,
                   help="the engine's --max-num-seqs per engine (default: read from the registry's launch.json); the "
                        "auto budget stops raising when running reaches it")
    p.add_argument('--model', default=None, help='duration-table model key (e.g. qwen35-4b); none: other models only')
    p.add_argument('--durations', default=str(DEFAULT_TABLE))
    p.add_argument('--expect-lanes', type=int, default=0, help='hold grants until this many lanes are ready')
    p.add_argument('--burst', type=int, default=16, help='pi lanes: attempt starts per minute per run')
    p.add_argument('--tick-sec', type=float, default=TICK_SEC)
    p.add_argument('--metrics', default=None, help='raw vLLM base URL: back off on preemption/waiting like adaptive.py')
    p.add_argument('--floor', type=int, default=16)
    p.add_argument('--registry', default=None, help="serving registry name of the model (default: the first pi "
                   "lane's serving:<name>); its gate_status_urls drive the gate back-off")
    p.add_argument('--gate-backoff', choices=('queued_new', 'off'), default='queued_new',
                   help='no new credit while the admission gate holds >= --gate-hold-min new conversations '
                        '(servers without /gate/status: no effect)')
    p.add_argument('--gate-hold-min', type=int, default=1)
    p.add_argument('--single-turn-hold', choices=('on', 'off'), default='on',
                   help='while the gate holds and a pi lane is gate-paused, unspent single-turn credit is withdrawn '
                        '(#378; needs the gate back-off, --gate-backoff queued_new)')
    p.add_argument('--reserve-sec', type=float, default=RESERVE_SEC,
                   help='pi credit unspent this long with the gate clear is withdrawn')
    p.add_argument('--ready-timeout', type=float, default=READY_TIMEOUT)
    p.add_argument('--release-server-after', type=float, default=0.0,
                   help='release the model server (registry job) this many s after every lane is done and idle '
                        '(#378; 0 = never). Use 600 when the server is this run\'s own, never for a shared one')
    p.add_argument('--pi-autoscale', action=argparse.BooleanOptionalAction, default=True,
                   help='add pi workers to a lane whose slots limit it (#378); launch lanes with few workers')
    p.add_argument('--pi-idle-exit-sec', type=float, default=600.0,
                   help='with --pi-autoscale: a pi worker with nothing running and no start credit for this long '
                        'exits; the dispatcher grows the lane back when credit returns (#378; 0 = never)')
    p.add_argument('--lazy-workers', action=argparse.BooleanOptionalAction, default=True,
                   help='pi runs created under this dir submit no worker until their first grant (#378; the runs\' '
                        'default, a run can override it with --[no-]lazy-workers)')
    p.add_argument('--notes', default='')
    p.add_argument('--force', action='store_true')
    for name in ('run', 'status'):
        p = sub.add_parser(name)
        p.add_argument('dir')
    p = sub.add_parser('report')
    p.add_argument('dir')
    p.add_argument('--out', default=None)
    p = sub.add_parser('plan')
    p.add_argument('plan')
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--model', default=None)
    p.add_argument('--durations', default=str(DEFAULT_TABLE))
    p.add_argument('--cap', type=float, default=7200.0)
    p.add_argument('--ids-out', default=None)
    p = sub.add_parser('budget')
    p.add_argument('dir')
    p.add_argument('n', help="a fixed budget (turns the auto budget off), or 'auto'")
    a = ap.parse_args(argv)
    if a.cmd == 'init':
        cmd_init(a)
    elif a.cmd == 'run':
        Dispatcher(Path(a.dir)).run()
    elif a.cmd == 'status':
        cmd_status(a)
    elif a.cmd == 'report':
        cmd_report(a)
    elif a.cmd == 'plan':
        cmd_plan(a)
    elif a.cmd == 'budget':
        cmd_budget(a)


if __name__ == '__main__':
    main()

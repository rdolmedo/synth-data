"""Claim packing for pi_eval workers (#531, run config `pack_claims`).

Measured on the M5 runs (11-15 synthesis, 11-15..18 evals): 40-52% of the Pi worker slot-hours we paid for ran
nothing. Part of it is a lane holding fewer start credits than its workers have slots (mid-run), part is the tail,
where every worker keeps a few stragglers. Workers claimed in parallel, so scarce work spread over all of them and
none ever emptied.

Rule: when the run's claimable work (unclaimed attempts and retries, capped by the dispatcher's unspent start credit
and the admission limit) fits in the free slots of the live, claiming workers "ahead" of this one, this worker does
not claim. A worker is ahead when it runs more attempts than this one, or the same number and its name sorts first.
So the fullest workers take scarce work, the others empty out, and they leave through the existing exits: #378's idle
exit under a dispatcher (no start credit), and eval_q35's no-claimable exit.

Only *which worker* claims changes. What is claimed next, its seed and its protocol come from the queue (eval_q35
State.claim) and never depend on the worker.

Guards:
- A peer counts only with a fresh heartbeat (workers/<name>.json less than PEER_FRESH s old, no `exited` mark) that
  says it can claim (`claiming`, written by eval_q35's loop: endpoint up, proxy ok, no STOP_WORKERS / drain, gate
  clear, not leaving), with its own `slots` (ours when an older heartbeat lacks them).
- Progress is measured by other workers' claims only (this worker's own claims never reset its wait). If the demand
  has not dropped through others for DEFER_MAX s while this worker deferred, it claims freely for RELEASE_SEC: a fuller
  worker that cannot claim (a one-per-worker task, a burst window) stalls starts for at most DEFER_MAX.
- Off when the tbsched router governs the run (config `sched`): tbsched's lifecycle decides there.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

PEER_FRESH = 120.0    # s: heartbeats are written every eval_q35 POLL_S (20 s)
PEER_GONE = 600.0     # s: a heartbeat this old (or exited) never comes back (the name carries the pid): not re-read
DEFER_MAX = 60.0      # s: no progress by others this long while deferring -> claim freely
RELEASE_SEC = 600.0   # s: how long a worker then claims without deferring
CACHE_SEC = 5.0       # s: one decision (directory scans) per worker per this interval


def _epoch(s) -> float | None:
    try:
        return dt.datetime.fromisoformat(str(s)).timestamp()
    except (TypeError, ValueError):
        return None


def peers(workers_dir: Path, me: str, now: float, slots: int, gone: set | None = None) -> list[tuple[str, int, int]]:
    """(name, running, slots) of the run's other live workers that can claim. `gone` (updated): names to skip."""
    gone = set() if gone is None else gone
    out = []
    try:
        files = [p for p in Path(workers_dir).glob('*.json') if p.name[:-len('.json')] not in gone]
    except OSError:
        return out
    for p in files:
        name = p.name[:-len('.json')]
        if name == me:
            continue
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or 'heartbeat' not in rec:
            gone.add(name)   # not a heartbeat file (one_per_worker.json)
            continue
        hb = _epoch(rec.get('heartbeat'))
        if rec.get('exited') or hb is None or now - hb > PEER_GONE:
            gone.add(name)
            continue
        if now - hb > PEER_FRESH or rec.get('claiming') is False:
            continue
        out.append((name, len(rec.get('running') or []), int(rec.get('slots') or slots)))
    return out


def demand(state, run: Path, dispatched: bool) -> int:
    """Attempts that can still be claimed now: unclaimed first attempts + unclaimed retries, at most the dispatcher's
    unspent start credit and the admission limit's room. Cheapest checks first (most polls in a credit-limited lane
    find no credit left)."""
    claimed = state._names('claims')
    cap = None
    if dispatched:
        try:
            ctl = json.loads((Path(run) / 'control.json').read_text())
            cap = max(0, int(ctl.get('max_starts') or 0) - len(claimed))
        except (OSError, ValueError, TypeError):
            cap = None
        if cap == 0:
            return 0
    try:
        adm = json.loads((Path(run) / 'queue' / 'admission.json').read_text())
        room = max(0, int(adm['limit']) - len(claimed - state.done()))
        cap = room if cap is None else min(cap, room)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if cap == 0:
        return 0
    n = sum(1 for t, k in state.items() if f'{t}__a{k}__r0' not in claimed)   # eval_q35 queue_state.key(t, k, 0)
    n += len(state._names('retry', '.json') - claimed)
    return max(0, n if cap is None else min(n, cap))


def ahead_free(me: str, mine: int, others: list[tuple[str, int, int]]) -> int:
    """Free slots of the claiming workers ahead of this one (more running, or as many and an earlier name)."""
    return sum(s - r for n, r, s in others if r < s and (r > mine or (r == mine and n < me)))


class Packer:
    def __init__(self, worker, slots: int, dispatched: bool, log=print, clock=time.time):
        self.w, self.slots, self.dispatched, self.log, self.clock = worker, int(slots), dispatched, log, clock
        self.cached_at, self.cached = -1e18, False
        self.waiting = None      # (demand, since, my claims then) while this worker defers
        self.free_until = 0.0    # after a released wait: claim without deferring until then
        self.mine = 0            # claims this worker made (they are not progress by others)
        self.deferred = 0        # claim tries skipped (for the log)
        self.gone: set = set()

    def defer(self) -> bool:
        now = self.clock()
        if now - self.cached_at < CACHE_SEC:
            return self.cached
        self.cached_at = now
        self.cached = self._decide(now)
        return self.cached

    def _decide(self, now: float) -> bool:
        if now < self.free_until or len(self.w.running) >= self.slots:
            self.waiting = None
            return False
        try:
            d = demand(self.w.state, self.w.run, self.dispatched)
            if d <= 0:
                self.waiting = None
                return False
            others = peers(self.w.run / 'workers', self.w.name, now, self.slots, self.gone)
        except OSError:
            self.waiting = None
            return False      # unsure: claim as before
        if d > ahead_free(self.w.name, len(self.w.running), others):
            self.waiting = None
            return False
        if self.waiting is None:
            self.waiting = (d, now, self.mine)
            return True
        d0, since, mine0 = self.waiting
        if d0 - d - (self.mine - mine0) > 0:   # the workers ahead claimed: keep leaving it to them
            self.waiting = (d, now, self.mine)
            return True
        if now - since >= DEFER_MAX:
            self.log(f'claim packing: no claims by the {len(others)} fuller worker(s) for {now - since:.0f}s with '
                     f'{d} claimable ({self.deferred} tries deferred): claiming freely for {RELEASE_SEC:.0f}s')
            self.waiting, self.free_until = None, now + RELEASE_SEC
            return False
        return True

    def wrap(self, inner):
        def claim(worker, tasks_available):
            if self.defer():
                self.deferred += 1
                return None
            k = inner(worker, tasks_available)
            if k is not None:
                self.mine += 1
            return k
        return claim


def install(worker, log=print):
    """PiEvalWorker.__init__ (after every other claim wrapper, so it is the outermost and never runs under a claim
    lock): wrap state.claim when the run config has pack_claims and no tbsched router."""
    c = worker.config
    if not c.get('pack_claims') or c.get('sched'):
        return None
    # this worker's --slots (the run's workers share the run's stored shape; peers report their own in heartbeats)
    slots = int(getattr(getattr(worker, 'a', None), 'slots', 0) or (c.get('worker') or {}).get('slots') or 0)
    if slots <= 0:
        return None
    p = Packer(worker, slots, bool(c.get('dispatch')), log=log)
    worker.state.claim = p.wrap(worker.state.claim)
    return p

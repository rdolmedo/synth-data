"""Start credits per (run, engine) and the engine a new session runs on (#451 workstream 1: placement).

Credits are the aa_evals dispatcher's mechanism (dispatcher.decide), applied per engine instead of per model run:
* a credit is a cumulative number of starts; a run may start a session on engine e while its starts there are below
  its credit there. Unspent credit counts as used, so a slow-reading worker cannot overshoot;
* each tick an engine's free room (its limit minus every run's live sessions and unspent credit there) is granted to
  the runs allowed on it: in run-weight order (smooth weighted round robin, `order = "weight"`, generation) or longest
  predicted remaining first (`order = "longest_first"`, eval makespan);
* each grant goes to the run's allowed engine with the most free room relative to its limit: credit goes where room is,
  so routing by credit is routing to the less-loaded engine;
* a burst per engine GROUP per tick bounds the grants (shared by every run on the group: replay rounds started
  together are staggered, 11-15 283 cold prefills in one queue), and replay sessions have their own per-engine cap and
  per-group burst (a replay's first turn is a cold ~100k prefill: 96/engine next to second traces tipped an engine).

`route` is the one engine choice every client uses at claim time (pi_eval workers, single-turn clients, synth's push
dispatch, DP ranks): among the engines where the run has unspent credit, the one with the most; ties (within `tie`)
broken by a hash of (seed namespace, task, attempt). The seed namespace differs per round, so parallel rounds of a task
(every round's attempt index is 0) no longer land on one engine (m5l-tbsci 11-15: 113-122 running / 3% hit next to a
sibling at 10-35). The pin is kept for the session's life.
"""
from dataclasses import dataclass, field
import hashlib
import math

PLACE = {
    'order': 'weight',        # weight | longest_first
    'burst_min': 16,          # grants per group per tick: max(burst_min, burst_frac x the group's limits) ...
    'burst_frac': 0.5,
    'replay_per_engine': 24,  # replay sessions (live + unspent) per engine
    'replay_burst': 8,        # replay grants per engine per tick
}


@dataclass
class Engine:
    name: str
    group: str
    limit: int                # agentic + replay sessions
    single_cap: int = 0       # single-turn requests in flight
    other: int = 0            # sessions on it that no run here accounts for (other campaigns)
    open: bool = True         # False: draining, released or its gate holds new sessions


@dataclass
class Run:
    name: str
    kind: str                 # agentic | replay | single
    engines: list             # engines it may use
    demand: int               # claimable items (work.Pending.claimable)
    credit: dict = field(default_factory=dict)   # engine -> cumulative credit
    starts: dict = field(default_factory=dict)   # engine -> cumulative starts (pins)
    live: dict = field(default_factory=dict)     # engine -> live sessions
    weight: float = 1.0
    priority: float = 0.0     # longest_first: predicted remaining seconds of its head item

    def __post_init__(self):
        # credit never below starts: lost state, a run hooked mid-way or a worker over-starting would otherwise
        # leave the run with no spendable credit for starts/burst ticks
        self.credit = {e: max(self.credit.get(e, 0), self.starts.get(e, 0)) for e in {*self.credit, *self.starts}}

    def unspent(self, e):
        return max(0, self.credit.get(e, 0) - self.starts.get(e, 0))


def burst_budget(engines: dict, p: dict | None = None) -> dict:
    """{group: grants allowed per period}: max(burst_min, burst_frac x the group's limits and single caps)."""
    p = {**PLACE, **(p or {})}
    out = {}
    for e in engines.values():
        out[e.group] = out.get(e.group, 0) + e.limit + e.single_cap
    return {g: max(p['burst_min'], math.ceil(p['burst_frac'] * v)) for g, v in out.items()}


def allocate(runs: list, engines: dict, p: dict | None = None, burst_left: dict | None = None) -> dict:
    """{run name: {engine: credit to add}} for this tick. Mutates nothing. burst_left: {group: grants still allowed in
    the current period} when allocations happen more often than once per period (refills between decisions)."""
    p = {**PLACE, **(p or {})}
    free, sfree, rfree, rburst = {}, {}, {}, {}
    for e in engines.values():
        used = e.other + sum(r.live.get(e.name, 0) + r.unspent(e.name) for r in runs if r.kind != 'single')
        free[e.name] = max(0, e.limit - used) if e.open else 0
        sused = sum(r.live.get(e.name, 0) + r.unspent(e.name) for r in runs if r.kind == 'single')
        sfree[e.name] = max(0, e.single_cap - sused) if e.open else 0
        rused = sum(r.live.get(e.name, 0) + r.unspent(e.name) for r in runs if r.kind == 'replay')
        rfree[e.name] = max(0, p['replay_per_engine'] - rused)
        rburst[e.name] = p['replay_burst']
    burst = burst_budget(engines, p)
    if burst_left is not None:
        burst = {g: max(0, min(b, burst_left.get(g, b))) for g, b in burst.items()}
    need = {r.name: max(0, r.demand - sum(r.unspent(e) for e in r.engines)) for r in runs}
    grants = {r.name: {} for r in runs}

    def pick(r):
        best, score = None, None
        room = sfree if r.kind == 'single' else free
        for e in r.engines:
            if e not in engines or room.get(e, 0) <= 0 or burst[engines[e].group] <= 0:
                continue
            if r.kind == 'replay' and (rfree[e] <= 0 or rburst[e] <= 0):
                continue
            cap = engines[e].single_cap if r.kind == 'single' else engines[e].limit
            s = (room[e] / max(cap, 1), room[e], e)
            if score is None or s > score:
                best, score = e, s
        return best

    def grant(r, e):
        (sfree if r.kind == 'single' else free)[e] -= 1
        burst[engines[e].group] -= 1
        if r.kind == 'replay':
            rfree[e] -= 1
            rburst[e] -= 1
        grants[r.name][e] = grants[r.name].get(e, 0) + 1
        need[r.name] -= 1

    active = [r for r in runs if need[r.name] > 0]
    if p['order'] == 'longest_first':
        for r in sorted(active, key=lambda r: (-r.priority, r.name)):
            while need[r.name] > 0:
                e = pick(r)
                if e is None:
                    break
                grant(r, e)
        return grants
    # smooth weighted round robin: each round the run with the highest current weight gets one unit
    cur = {r.name: 0.0 for r in active}
    while active:
        total = sum(r.weight for r in active)
        for r in active:
            cur[r.name] += r.weight
        r = max(active, key=lambda r: (cur[r.name], r.name))
        e = pick(r)
        if e is None:
            active.remove(r)
            continue
        cur[r.name] -= total
        grant(r, e)
        if need[r.name] <= 0:
            active.remove(r)
    return grants


def withdraw(runs: list, engines: dict) -> dict:
    """{run name: {engine: credit to take back (> 0)}}: unspent credit that no longer has room behind it. Applied
    before allocate (and the runs' credit lowered), so a cut limit, a drain or a gate hold acts at once instead of
    after the old credit is spent:
    * on an engine that is not open: all of it;
    * on an engine whose live sessions + unspent credit exceed its limit (or single cap): the excess, largest first;
    * of a run whose unspent credit exceeds its demand: the excess, from the engines where it has the most."""
    out = {r.name: {} for r in runs}
    left = {(r.name, e): r.unspent(e) for r in runs for e in {*r.engines, *r.credit}}

    def take(r, e, n):
        n = min(n, left[r.name, e])
        if n > 0:
            left[r.name, e] -= n
            out[r.name][e] = out[r.name].get(e, 0) + n
        return n

    for e in engines.values():
        for single in (False, True):
            rs = [r for r in runs if (r.kind == 'single') == single and e.name in r.engines]
            if not e.open:
                for r in rs:
                    take(r, e.name, left[r.name, e.name])
                continue
            cap = e.single_cap if single else e.limit - e.other
            excess = sum(r.live.get(e.name, 0) + left[r.name, e.name] for r in rs) - cap
            for r in sorted(rs, key=lambda r: (-left[r.name, e.name], r.name)):
                if excess <= 0:
                    break
                excess -= take(r, e.name, excess)
    for r in runs:
        for e in r.credit:
            if e not in r.engines or e not in engines:
                take(r, e, left[r.name, e])     # an engine the run may no longer use: its credit is phantom room
        excess = sum(left[r.name, e] for e in r.engines) - r.demand
        for e in sorted(r.engines, key=lambda e: (-left[r.name, e], e)):
            if excess <= 0:
                break
            excess -= take(r, e, excess)
    return {k: v for k, v in out.items() if v}


def key_hash(namespace: str, task: str, attempt: int) -> int:
    return int(hashlib.sha256(f'{namespace}/{task}/{attempt}'.encode()).hexdigest(), 16)


def route(unspent: dict, namespace: str, task: str, attempt: int, tie: float = 0.10) -> str | None:
    """The engine a new session of (task, attempt) starts on: the most unspent credit, hash tie-break."""
    c = {e: u for e, u in unspent.items() if u > 0}
    if not c:
        return None
    m = max(c.values())
    cands = sorted(e for e, u in c.items() if u >= m * (1 - tie))
    return cands[key_hash(namespace, task, attempt) % len(cands)]

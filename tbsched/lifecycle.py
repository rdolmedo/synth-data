"""Worker and engine-group lifecycle (#451 workstream 1). Pure decisions: the loop (or a simulator) applies them.

Workers, per run:
* sized to the run's live sessions + the credit it may still spend (never more than its claimable work): credit is
  bounded by the per-tick burst, so a limit left over from an earlier ramp cannot over-provision (TB-Sci-lite p0's
  stale limit of 207 would have taken a registry to ~440 sessions on a naive backfill);
* a run with nothing claimable gets no new workers ("skip drained runs": +9 workers were added to two finished runs);
* recycled: a worker with >= recycle_failures worker-local infra failures (build failures, a full disk, sandbox setup)
  in the window and at least as many failures as successes fails fast, claims again and burns retries (11-15: 40
  failures to 14 successes in 45 min). It is removed; the next tick's sizing backfills it;
* trimmed: workers idle past idle_grace_sec while the run holds a whole worker's slots too many;
* a drained run (or one with no engine group left) gets STOP_WORKERS.
Everything acts on what condor lists now, so a restarted scheduler cannot re-recycle a worker it removed before.

Engine groups (serving jobs):
* stealing is a weight change: a hungry group (its runs' claimable work < steal_below and its engines below
  hungry_running on average) joins the run with the largest predicted remaining time (claimable x mean session seconds
  / its engines), same model and a compatible kind, at most once per steal_cooldown_sec (a stolen run ramps for
  minutes and still looks hungry: critpt2 stole twice in 3 min);
* end game: once every backlog is below endgame_items, the least-loaded group is drained (no new credit) when the
  other groups' limits hold every live session and claimable item;
* release: a group with no claimable work routed to it (or draining) and no session for release_ticks ticks; with
  tail = "cut" (synth) it may also go at <= tail_sessions sessions held for tail_ticks ticks;
* a run left with no group gets STOP_WORKERS and its workers removed (two workers sat ~1 h on dead claims of a
  released registry).
"""
from dataclasses import dataclass, field
import math

LIFE = {
    'recycle_failures': 5,
    'recycle_window_sec': 2700,
    'idle_grace_sec': 300,
    'heartbeat_fresh_sec': 120,
    'steal_below': 100,
    'steal_min': 40,             # smallest donor backlog worth taking from
    'hungry_running': 50,
    'steal_cooldown_sec': 600,
    'endgame_items': 150,
    'release_ticks': 2,
    'dead_engine_sec': 600,      # an engine unreadable this long with no live claim on it no longer blocks release
    'tail': 'wait',              # wait | cut
    'tail_sessions': 8,
    'tail_ticks': 8,
}
LOCAL_FAILURES = ('BuildError', 'no_harbor_result', 'disk_full', 'ENOSPC', 'userns_setup')


@dataclass
class Worker:
    id: str                    # condor job id
    status: str                # idle | running | held | gone (condor's view now)
    running: int = 0           # sessions it runs (heartbeat)
    hb_age: float | None = None
    idle_for: float = 0.0      # seconds its heartbeat has shown nothing running
    local_failures: int = 0    # in the recycle window
    successes: int = 0


def plan_workers(live: int, unspent: int, claimable: int, slots: int, workers: list, p: dict | None = None,
                 known: bool = True) -> dict:
    """{add, remove (ids), recycle (ids), recycle_wait (ids), stop (bool)} for one run. known=False (its queue could
    not be read): nothing. A black-hole worker is removed only once it runs nothing (removing it would kill its live
    sessions); until then it is listed in recycle_wait."""
    p = {**LIFE, **(p or {})}
    none = {'add': 0, 'remove': [], 'recycle': [], 'recycle_wait': [], 'stop': False}
    if not known:
        return none
    alive = [w for w in workers if w.status in ('idle', 'running')]
    bad = [w for w in alive if w.status == 'running' and w.local_failures >= p['recycle_failures']
           and w.local_failures >= w.successes]
    recycle = [w.id for w in bad if w.running == 0]
    wait = [w.id for w in bad if w.running > 0]
    keep = [w for w in alive if w.id not in recycle]
    if claimable == 0 and live == 0:
        return {**none, 'recycle': recycle, 'recycle_wait': wait, 'stop': True}
    want = live + min(unspent, claimable)
    have = len(keep) * slots
    add = max(0, math.ceil((want - have) / slots)) if claimable > 0 else 0
    if claimable > 0 and not keep:
        add = max(add, 1)   # someone must claim (and reap dead claims): AB r1 sat ~2.5 h with no worker left
    remove = []
    surplus = have - want
    if surplus >= slots and add == 0:
        idle = sorted((w for w in keep if w.status == 'running' and w.running == 0 and w.hb_age is not None
                       and w.hb_age < p['heartbeat_fresh_sec'] and w.idle_for >= p['idle_grace_sec']),
                      key=lambda w: (-w.idle_for, w.id))
        n = surplus // slots
        if claimable > 0:
            n = min(n, len(keep) - 1)   # keep one worker while there is work (else trim and backfill churn)
        remove = [w.id for w in idle[:max(0, n)]]
    return {'add': add, 'remove': remove, 'recycle': recycle, 'recycle_wait': wait, 'stop': False}


@dataclass
class GroupView:
    name: str
    model: str
    engines: list
    status: str = 'active'     # active | draining | released
    sessions: int = 0          # live sessions pinned to its engines
    running: float = 0.0       # mean running requests per engine
    limit: int = 0             # sum of its engines' limits
    busy: bool = False         # its engines report requests (any client), or their state is unknown: never idle


@dataclass
class RunView:
    name: str
    model: str
    kind: str
    groups: dict               # group -> weight
    claimable: int
    live: int = 0
    session_sec: float = 1800.0
    steal: bool = True         # may be spread onto more groups
    engines_per_group: dict = field(default_factory=dict)


def plan_groups(groups: dict, runs: dict, st: dict, now: float, p: dict | None = None, keep=()) -> list:
    """Actions [(verb, ...)] for this tick: ('join', group, run), ('drain', group), ('release', group, cut: bool),
    ('stop_run', run). st: persisted per-group state {group: {last_steal, idle_ticks, tail_ticks}} (mutated).
    keep: groups never released this tick (a pending stage targets them, or an operator override).
    Every kind may go to every engine of its model: single-turn work fills decode headroom on Pi engines too
    (11-15: 160 SciCode workers on the near-idle m5l-ab2 took it from 2.5k to 5.2-7.7k tok/s)."""
    p = {**LIFE, **(p or {})}
    acts = []
    active = {g: v for g, v in groups.items() if v.status == 'active'}
    on = {g: [r for r in runs.values() if g in r.groups] for g in groups}

    def remaining(r):
        n = sum(len(groups[g].engines) for g in r.groups if g in groups and groups[g].status == 'active') or 1
        return r.claimable * r.session_sec / n

    endgame = bool(runs) and all(r.claimable < p['endgame_items'] for r in runs.values())
    # stealing: one join per tick, the hungriest group first; none in the end game (it consolidates instead)
    joined = endgame
    for g in sorted(active, key=lambda g: (active[g].running, g)):
        gs = st.setdefault(g, {})
        if joined or now - gs.get('last_steal', -math.inf) < p['steal_cooldown_sec']:
            continue
        left = sum(r.claimable for r in on[g])
        if left >= p['steal_below'] or active[g].running >= p['hungry_running']:
            continue
        donors = [r for r in runs.values() if g not in r.groups and r.steal and r.model == active[g].model
                  and r.claimable >= p['steal_min']]
        if not donors:
            continue
        r = max(donors, key=lambda r: (remaining(r), r.name))
        acts.append(('join', g, r.name))
        gs['last_steal'] = now
        joined = True
    # end game: consolidate small backlogs onto the groups already serving them
    recent = {g for g in active if now - st.get(g, {}).get('last_steal', -math.inf) < p['steal_cooldown_sec']}
    if len(active) > 1 and endgame and len(active) - len(recent) >= 1 and set(active) - recent:
        g = min(set(active) - recent, key=lambda g: (active[g].sessions, g))
        others = [h for h in active if h != g]
        need = sum(r.live + r.claimable for r in runs.values() if r.model == active[g].model)
        serving_elsewhere = all(any(h in r.groups for h in others) for r in on[g])
        if serving_elsewhere and sum(active[h].limit for h in others if active[h].model == active[g].model) >= need:
            acts.append(('drain', g))
    # release
    joins = {a[1] for a in acts if a[0] == 'join'}
    for g, v in groups.items():
        if v.status == 'released':
            continue
        gs = st.setdefault(g, {})
        routed = 0 if v.status == 'draining' else sum(r.claimable for r in on[g])
        if g in keep or g in joins or v.busy or not v.engines:
            gs['idle_ticks'] = gs['tail_ticks'] = 0   # unknown or busy (another client may use it): never released
            continue
        if routed == 0 and v.sessions == 0:
            gs['idle_ticks'] = gs.get('idle_ticks', 0) + 1
        else:
            gs['idle_ticks'] = 0
        if p['tail'] == 'cut' and routed == 0 and v.sessions <= p['tail_sessions']:
            gs['tail_ticks'] = gs.get('tail_ticks', 0) + 1
        else:
            gs['tail_ticks'] = 0
        if gs['idle_ticks'] >= p['release_ticks']:
            acts.append(('release', g, False))
        elif p['tail'] == 'cut' and gs['tail_ticks'] >= p['tail_ticks']:
            acts.append(('release', g, True))
    released = {a[1] for a in acts if a[0] == 'release'}
    for r in runs.values():
        if r.groups and all(g in released or groups.get(g) is None or groups[g].status == 'released' for g in r.groups):
            acts.append(('stop_run', r.name))
    return acts

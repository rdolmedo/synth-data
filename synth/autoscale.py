"""Serving engines follow demand (#234 R1): a campaign starts with a few engines and the controller adds or drains
engines from its train demand, between `start` and the [serving] jobs ceiling (and the live cap
ops/admission-policy.json serving_jobs).

    demand   = active attempts + attempts still to dispatch (opened tasks' owed traces + streamed, unopened tasks x M)
    capacity = engines in use (starting ones included) x fit, fit = agents one engine admits (the kv_fit limit)
    up       : demand - capacity > up_backlog x fit for up_seconds                 -> one more engine (up_step)
    down     : demand < down_util x capacity for down_seconds, above min_engines   -> drain one engine

An engine takes ~5 min to get healthy and its GPU costs the same idle or busy, so both directions wait (hysteresis);
after any change both timers restart. A drained engine takes no new attempts and is released once its last attempt
ends (the controller's existing drain path); it is the first one back when demand rises again.

The policy is a pure function of (state, observation, parameters), so the controller and the CPU replay
(synth/analysis/autoscale_replay.py) run the same code. Defaults were chosen with that replay on m1-hero's supply
curve (#234).
"""
import math

DEFAULTS = {
    'enabled': False,
    'start': 2,               # engines at campaign start (and the floor unless min_engines says otherwise)
    'min_engines': None,      # floor; None: start
    'up_backlog': 1.0,        # add when demand exceeds capacity by this many engines' fit ...
    'up_seconds': 300,        # ... continuously for this long
    'up_step': 1,             # engines added per decision
    'down_util': 0.5,         # drain when demand is below this share of capacity ...
    'down_seconds': 900,      # ... continuously for this long
    'fit_prior': 367,         # agents per engine before any engine reports a limit (m1-hero kv_fit: 367 per B200)
    'period_seconds': 60,     # the controller decides at most this often (demand is a pass over every attempt)
}


def params(serving):
    """[serving.autoscale] merged over DEFAULTS; a bare `true` enables the defaults."""
    raw = serving.get('autoscale') or {}
    if raw is True:
        raw = {'enabled': True}
    p = {**DEFAULTS, **raw}
    if 'enabled' not in raw and raw:
        p['enabled'] = True    # a [serving.autoscale] table without `enabled` means on
    if p['min_engines'] is None:
        p['min_engines'] = p['start']
    return p


def decide(state, now, *, engines, ceiling, demand, fit, p):
    """-> the number of engines wanted (engines, more, or fewer). `state` (a dict, persisted by the caller) keeps the
    timers: up_since / down_since (when the condition started holding) and changed_at.

    engines: engines in use now (healthy or starting; draining ones excluded); ceiling: the most allowed now
    (min of [serving] jobs and the live cap); demand: active + dispatchable attempts (math.inf: unbounded);
    fit: agents per engine."""
    floor = min(max(1, int(p['min_engines'])), ceiling)
    if engines < floor:                                  # below the floor (start, or the cap was raised): no wait
        _reset(state, now)
        return floor
    if engines > ceiling:                                # the live cap fell: follow it now (the cap's drain path)
        _reset(state, now)
        return ceiling
    fit = max(1.0, float(fit or p['fit_prior']))
    capacity = engines * fit
    over = demand - capacity > p['up_backlog'] * fit
    under = demand < p['down_util'] * capacity
    state['up_since'] = (now if state.get('up_since') is None else state['up_since']) if over else None
    state['down_since'] = (now if state.get('down_since') is None else state['down_since']) if under else None
    if over and engines < ceiling and now - state['up_since'] >= p['up_seconds']:
        step = max(1, int(p['up_step']))
        if demand != math.inf:
            step = min(step, max(1, math.ceil((demand - capacity) / fit - p['up_backlog'] + 1e-9)))
        _reset(state, now)
        return min(ceiling, engines + step)
    if under and engines > floor and now - state['down_since'] >= p['down_seconds']:
        _reset(state, now)
        return engines - 1
    return engines


def _reset(state, now):
    state['up_since'] = state['down_since'] = None
    state['changed_at'] = now

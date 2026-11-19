"""KV-fit admission: run as many agents per backend as maximizes its generated tokens/s within its KV.

vLLM's kv_cache_usage gauge counts only blocks held by requests in flight. An agent waiting on a
tool keeps its conversation cached (that is what makes ~90% of each turn's prompt a prefix hit),
but those blocks count as free. Admitting on the gauge under-filled the 2026-10-17/18 Super
campaigns (4-7% mean usage, 12-17 running sequences per B200).

A backend's working set is measured:

    working set = in-flight KV (gauge x capacity) + idle agents' cached contexts (block-rounded)

Idle contexts come from each agent's context guard through the worker heartbeat; the in-flight part
includes everything a running request holds (for Nemotron Super, ~20k tokens of Mamba/MTP state
blocks beyond its prompt). A backend admits while

    max(working set, live agents x mean demand) <= target x capacity

The mean demand per agent is a slow average of working set / agents, floored by the campaign's
`agent_tokens_prior`: during a ramp every agent is young and short, and following that mean down
over-admits agents that later grow.

The target is found per backend. By default (`target_search = "throughput"`) it hill-climbs
generated tokens/s: the free KV is not idle headroom (every Super request allocates and frees ~20k
tokens of state blocks, so prefixes of agents idle for over a minute are evicted), and throughput
stayed flat-to-rising through moderate eviction, so the best target is found by measuring, not by
a cache-hit threshold. A cache collapse (reuse < `reuse_collapse`), the in-flight gauge, preemptions
and the request queue are brakes. `target_search = "reuse"` is the AIMD alternative on prompt-cache
reuse (the share of the previous turn's prompt vLLM served from cache). Campaigns sharing serving
publish their idle working sets next to the serving job's endpoints, so each backend's budget is shared.

`target_search = "slope"` (issue #6) measures the throughput curve at the scale where it is flat: it multiplies
the target by `slope_grow_factor` per ~10-minute window and keeps growing while each doubling adds at least
`slope_min_gain` generated tokens/s; when a step adds less, or reuse falls below `slope_reuse_drop` of the best
clean reuse seen (the first sign of thrash), it returns to the previous target and holds, probing one step up
again every `slope_reprobe_seconds`. A target the backend cannot fill while the request queue pauses admission
is the engine's concurrency cap (`engine_cap`): it holds there.
"""
import math

DEFAULTS = {
    'kv_target_fraction': 0.40,      # initial target (measured working set / KV capacity)
    'kv_target_min': 0.15,
    'kv_target_max': 0.90,
    'kv_target_fraction_offload': 1.0,  # initial target of a backend with a CPU KV tier (its cap scales with the tier)
    'kv_target_min_offload': 1.0,    # and its floor: below one GPU's worth a target only lowers concurrency (the
                                     # in-flight brakes bind first); the throughput search walked a CalibForge replay
                                     # down to 0.64 on noise, -17% vs the fixed-level peak (2026-10-24)
    'target_search': 'throughput',   # 'throughput': hill-climb the target on gen tok/s; 'reuse': AIMD on reuse
    'search_settle_seconds': 90,     # after a target change, ignore this long (fill/drain, cache rebuild) ...
    'search_measure_seconds': 210,   # ... then average gen tok/s over this long
    'search_step': 0.10,             # relative target change per decision
    'search_tolerance': 0.02,        # throughput changes within +-2% keep the direction
    'inflight_fill_below': 0.0,      # CPU-tier backends: filled to the target with the in-flight KV below this, no
    'inflight_fill_step': 1.30,      # preemption: grow the target this much per decision instead of the search step
                                     # (off by default; Super campaigns set 0.55)
    'reuse_collapse': 0.50,          # throughput mode: cut at once below this reuse (near the target)
    'agent_tokens_prior': 20000,     # KV demand per live agent (idle context + share of in-flight KV); also its floor
    'agent_tokens_floor': 1.0,       # the mean never goes below floor x prior
    'agent_tokens_halflife_seconds': 1200,
    'new_agents_per_backend_per_period': 32,
    'emergency_kv_high': 0.92,       # in-flight gauge: pause admissions
    'resume_kv': 0.85,
    'resume_hold_seconds': 30,
    'max_waiting': 64,
    'preemption_pause_seconds': 120,
    'reuse_window_seconds': 180,
    'reuse_min_expected_tokens': 2_000_000,
    'reuse_low': 0.90,               # shrink the target below this reuse, when operating near the target
    'reuse_high': 0.95,              # grow it above this, when operating at the target
    'target_band': 0.10,             # "near the target": mean working set / capacity >= target - band
    'target_decrease_factor': 0.90,
    'target_increase_step': 0.02,
    'target_decrease_hold_seconds': 60,
    'target_increase_hold_seconds': 180,
    'kv_capacity_tokens': None,      # fallbacks when serving does not publish its KV geometry
    'kv_block_tokens': 16,
    'slope_grow_factor': 1.5,        # slope search: multiplicative step up
    'slope_min_gain': 0.03,          # keep growing while a doubling adds >= 3% generated tok/s
    'slope_reuse_drop': 0.90,        # stop growing below this share of the best clean reuse seen
    'slope_settle_seconds': 150,
    'slope_measure_seconds': 450,
    'slope_reprobe_seconds': 2700,   # while holding, probe one step up this often
}


def footprint(context_tokens, block):
    """KV tokens an agent holds: whole blocks."""
    return math.ceil(max(0, context_tokens) / block) * block if context_tokens else 0


def idle_tokens(context, block):
    """KV an idle agent keeps cached (its context, whole blocks); in flight it is in the gauge."""
    if not context or context.get('inflight'):
        return 0
    return footprint(context.get('context_tokens') or 0, block)


def update_mean(previous, observed, dt, halflife):
    """Exponential average with a half-life in seconds."""
    if observed is None:
        return previous
    w = 1 - 0.5 ** (max(0.0, dt) / halflife)
    return previous + w * (observed - previous)


def step(b, policy, now, obs, mean_tokens):
    """One admission tick for one backend. `b` is its persisted state (mutated); `obs`:
      kv, wait, preempted (bool), stale (bool), capacity, offload (CPU KV tier tokens; 0 without one),
      own_idle_tokens, own_agents, other_idle_tokens, other_agents (other campaigns on the backend),
      reuse_expected, reuse_cached (this tick's deltas).
    Returns the backend's status: room (new agents allowed now), limit (agent capacity at the
    current target), working set and the causes of a pause."""
    p = {**DEFAULTS, **policy}
    capacity = obs['capacity']
    offload = obs.get('offload') or 0
    start = p['kv_target_fraction']
    if offload and capacity:
        # A CPU KV tier keeps evicted prefixes: idle contexts may exceed the GPU KV by the tier's size. The in-flight
        # brakes below still bound what runs on the GPU.
        p['kv_target_max'] *= 1 + offload / capacity
        p['kv_target_min'] = max(p['kv_target_min'], p['kv_target_min_offload'])
        start = p['kv_target_fraction_offload']
    # Live min/max overrides (ops/admission-policy.json) apply to the stored target too.
    alpha = b['target'] = min(p['kv_target_max'], max(p['kv_target_min'], b.get('target', start)))
    measured = working_set(obs)
    agents = obs['own_agents'] + obs['other_agents']
    working = max(measured, agents * mean_tokens)
    # Reuse, and where the backend operated, over a sliding window of its turns.
    history = b.setdefault('reuse', [])
    history.append([now, obs['reuse_expected'], obs['reuse_cached'], measured / capacity if capacity else 0,
                    obs.get('gen_tok_s'), obs['kv']])
    history[:] = [x for x in history if now - x[0] <= p['reuse_window_seconds']]
    expected = sum(x[1] for x in history)
    reuse = sum(x[2] for x in history) / expected if expected >= p['reuse_min_expected_tokens'] else None
    used = sum(x[3] for x in history if len(x) > 3) / len(history)
    causes = []
    if obs['stale'] or not capacity:
        causes.append('stale_metrics' if obs['stale'] else 'unknown_kv_capacity')
    kv = obs['kv'] if obs['kv'] is not None else 1
    if kv >= p['emergency_kv_high']:
        b['kv_paused'] = True
    if b.get('kv_paused'):
        if kv <= p['resume_kv']:
            b['kv_low_since'] = b.get('kv_low_since') or now
            if now - b['kv_low_since'] >= p['resume_hold_seconds']:
                b['kv_paused'] = False
        else:
            b['kv_low_since'] = None
    if b.get('kv_paused'):
        causes.append('kv_inflight_high')
    if obs['preempted']:
        b['preempted_at'] = now
        if now - b.get('last_decrease', -math.inf) >= p['target_decrease_hold_seconds']:
            b['target'] = alpha = max(p['kv_target_min'], alpha * p['target_decrease_factor'])
            b['last_decrease'] = now
    if now - b.get('preempted_at', -math.inf) < p['preemption_pause_seconds']:
        causes.append('preemption')
    if (obs['wait'] or 0) > p['max_waiting']:
        causes.append('inference_queue')
    if p['target_search'] == 'throughput' and capacity:
        b['target'] = alpha = search(b, p, now, reuse, used, alpha, fill=bool(offload))
    elif p['target_search'] == 'slope' and capacity:
        # Filled = admitted up to the budget, judged on the same working set admission counts (the mean demand
        # per agent is floored by the prior, so a teacher whose agents use less KV fills by agent count).
        b['target'] = alpha = slope_search(b, p, now, obs, reuse, used, alpha, working / capacity,
                                           'inference_queue' in causes)
        if b['slope'].get('engine_cap') and 'inference_queue' in causes:
            causes.append('engine_cap')   # the queue, not KV, limits concurrency: raise max_seqs, not the target
    elif reuse is not None and capacity:
        # Judged only while the backend operated near its target: after an overload, evicted agents keep
        # missing while the working set drains, and cutting on that drove the target to its floor
        # (validation run). A cut lands just below where the backend thrashed.
        if (reuse < p['reuse_low'] and used >= alpha - p['target_band']
                and now - b.get('last_decrease', -math.inf) >= p['target_decrease_hold_seconds']):
            b['target'] = alpha = max(p['kv_target_min'], min(alpha, used) * p['target_decrease_factor'])
            b['last_decrease'] = now
            history.clear()  # judge the new target on fresh turns
        elif (reuse >= p['reuse_high'] and used >= alpha - p['target_band'] / 2
              and now - max(b.get('last_increase', -math.inf), b.get('last_decrease', -math.inf)) >= p['target_increase_hold_seconds']):
            b['target'] = alpha = min(p['kv_target_max'], alpha + p['target_increase_step'])
            b['last_increase'] = now
    ceiling = 1 + offload / capacity if capacity else 1
    blind = reuse is None and alpha > ceiling
    # No thrash signal (no per-turn cached tokens in the window): admit no more than the KV tiers hold. On
    # 2026-10-25 live-tmax2 had none, its target sat at a live 2.0 on a server without CPU offload, and it thrashed
    # (prefix hit 0.96 -> 0.05, 5k instead of 30k gen tok/s for 2.5 h); the searches never saw it.
    budget = (ceiling if blind else alpha) * capacity if capacity else 0
    limit = int(budget / mean_tokens) if mean_tokens else 0
    free = max(0, int((budget - working) / max(mean_tokens, 1)))
    room = 0 if causes else min(p['new_agents_per_backend_per_period'], free)
    return {'paused': bool(causes), 'causes': causes, 'room': room, 'free_agents': free, 'limit': limit,
            'target': round(alpha, 4), 'working_set_tokens': int(working), 'measured_tokens': int(measured),
            'inflight_tokens': int((obs['kv'] or 0) * (capacity or 0)),
            'capacity_tokens': capacity, 'agents': agents, 'reuse': None if reuse is None else round(reuse, 4),
            'used': round(used, 4), 'blind_cap': blind}


def search(b, p, now, reuse, used, alpha, fill=False):
    """Hill-climb the target on this backend's generated tokens/s. Throughput, not cache reuse, is the
    objective: in the validation run gen tok/s was flat-to-rising up to ~200 agents per B200 at a 0.75 prefix
    hit rate, so a reuse threshold left 10-20% on the table. Each decision averages gen tok/s after the
    previous change has settled; the direction holds while throughput does not fall by more than the
    tolerance and reverses when it does. A cache collapse near the target cuts at once.

    `fill` (a backend with a CPU KV tier): a window filled to the target whose in-flight KV stayed below
    `inflight_fill_below` without a preemption means the GPU batch is short of requests while the tier holds the
    idle contexts, so the target grows by `inflight_fill_step` at once. Live CalibForge agents spend ~80% of their
    time in tools: the hero's ran 35 of ~185 agents per B200 at 3.6k gen tok/s for an hour at target 1.0, and
    6.0-6.3k once the target was set to 2.0 by hand (2026-10-24); the +-10% search would have taken hours."""
    s = b.setdefault('search', {'direction': 1, 'since': now, 'previous': None})
    if reuse is not None and reuse < p['reuse_collapse'] and used >= alpha - p['target_band']:
        alpha = max(p['kv_target_min'], min(alpha, used) * p['target_decrease_factor'])
        b['search'] = {'direction': -1, 'since': now, 'previous': None, 'last': 'collapse'}
        b['reuse'] = []
        return alpha
    if now - s['since'] < p['search_settle_seconds'] + p['search_measure_seconds']:
        return alpha
    window = [x for x in b.get('reuse', []) if x[0] >= s['since'] + p['search_settle_seconds']
              and len(x) > 4 and x[4] is not None]
    if not window:
        return alpha
    if sum(x[3] for x in window) / len(window) < alpha - p['target_band']:
        s['since'] = now  # not filled to the target (workers, work, a drain): no evidence about it
        return alpha
    gen = sum(x[4] for x in window) / len(window)
    inflight = [x[5] for x in window if len(x) > 5 and x[5]]   # a zero or missing gauge is no evidence
    if (fill and inflight and sum(inflight) / len(inflight) < p['inflight_fill_below']
            and b.get('preempted_at', -math.inf) < window[0][0]):
        s.update(direction=1, previous=None, since=now, last='fill')
        s['history'] = (s.get('history') or [])[-11:] + [[round(now), round(alpha, 4), round(gen)]]
        return min(p['kv_target_max'], max(p['kv_target_min'], alpha * p['inflight_fill_step']))
    previous = s['previous']
    if previous is not None and gen < previous * (1 - p['search_tolerance']):
        s['direction'] = -s['direction']
    s['previous'], s['since'] = gen, now
    s['history'] = (s.get('history') or [])[-11:] + [[round(now), round(alpha, 4), round(gen)]]
    return min(p['kv_target_max'], max(p['kv_target_min'], alpha * (1 + p['search_step'] * s['direction'])))


def slope_search(b, p, now, obs, reuse, used, alpha, fill, queued):
    """Grow the target multiplicatively while each doubling adds >= slope_min_gain generated tok/s; on a flat
    step or the first sign of thrash, return to the previous target and hold (re-probing one step up every
    slope_reprobe_seconds). Each decision averages one window after a settle period; windows the backend could
    not fill to the target give no evidence (with a request queue, that is the engine's cap)."""
    lo, hi = p['kv_target_min'], p['kv_target_max']
    s = b.setdefault('slope', {'phase': 'grow', 'since': now, 'points': [], 'reuse_ref': None})
    if reuse is not None and reuse < p['reuse_collapse'] and used >= alpha - p['target_band']:
        s.update(phase='hold', since=now, hold_since=now, acc=None, last='collapse', points=[])
        b['reuse'] = []
        return max(lo, min(alpha, used) * p['target_decrease_factor'])
    if obs['preempted']:  # step() already lowered the target; judge the new one fresh, then hold
        s.update(phase='hold', since=now, hold_since=now, acc=None, last='preemption')
        return alpha
    if now - s['since'] < p['slope_settle_seconds']:
        return alpha
    if (s['phase'] == 'grow' and s['points'] and s['points'][-1][0] < alpha and s['reuse_ref'] is not None
            and reuse is not None and reuse < s['reuse_ref'] * p['slope_reuse_drop']):
        # A step up is thrashing: back off now rather than at the end of its window.
        s.update(phase='hold', since=now, hold_since=now, acc=None, last='thrash', fails=s.get('fails', 0) + 1)
        b['reuse'] = []
        return s['points'][-1][0]
    acc = s.get('acc') or [0.0, 0, 0.0, 0.0, 0.0, 0]   # gen sum, gen ticks, fill sum, expected, cached, ticks
    if obs.get('gen_tok_s') is not None:
        acc[0], acc[1] = acc[0] + obs['gen_tok_s'], acc[1] + 1
    acc[2], acc[3], acc[4], acc[5] = acc[2] + fill, acc[3] + obs['reuse_expected'], acc[4] + obs['reuse_cached'], acc[5] + 1
    s['acc'] = acc
    if now - s['since'] < p['slope_settle_seconds'] + p['slope_measure_seconds']:
        return alpha
    s['since'], s['acc'] = now, None
    if not acc[1]:
        return alpha
    if acc[2] / acc[5] < alpha - p['target_band']:
        s['engine_cap'] = queued   # not filled to the target: no evidence about it
        return alpha
    s['engine_cap'] = False
    gen = acc[0] / acc[1]
    wreuse = acc[4] / acc[3] if acc[3] >= p['reuse_min_expected_tokens'] else None
    thrash = wreuse is not None and s['reuse_ref'] is not None and wreuse < s['reuse_ref'] * p['slope_reuse_drop']
    if wreuse is not None and not thrash:
        s['reuse_ref'] = max(s['reuse_ref'] or 0.0, wreuse)
    points = s['points'] = (s['points'] + [[round(alpha, 4), round(gen), None if wreuse is None else round(wreuse, 4), round(now)]])[-12:]
    previous = points[-2] if len(points) > 1 and points[-2][0] < alpha else None
    if s['phase'] == 'hold':
        wait = p['slope_reprobe_seconds'] * 2 ** min(s.get('fails', 0), 3)   # failed probes back off, up to 8x
        if now - s.get('hold_since', now) >= wait and not thrash and alpha < hi:
            s['phase'] = 'grow'
            return min(hi, alpha * p['slope_grow_factor'])
        return alpha
    gain = None if previous is None else (gen / previous[1] - 1) / math.log2(alpha / previous[0])
    if thrash or (gain is not None and gain < p['slope_min_gain']) or alpha >= hi:
        failed = previous is not None and (thrash or gain < p['slope_min_gain'])
        s.update(phase='hold', hold_since=now, last='thrash' if thrash else 'flat' if gain is not None else 'max',
                 fails=s.get('fails', 0) + 1 if failed else 0)
        return previous[0] if failed else alpha
    s['fails'] = 0
    return min(hi, alpha * p['slope_grow_factor'])


def working_set(obs):
    """In-flight KV (gauge x capacity) + every campaign's idle agents' cached contexts."""
    return (obs['kv'] or 0) * (obs['capacity'] or 0) + obs['own_idle_tokens'] + obs['other_idle_tokens']

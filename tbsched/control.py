"""Per-engine session limits from load-aware signals (#451 workstream 1: the controller).

Every engine (one vLLM endpoint, or one DP rank) gets its own limit, so pressure on one engine never cuts its sibling;
routing (place.py) sends new sessions to the lighter one instead. Signals come from the engine's /metrics as counter
deltas over a short window (`Window`), not from vllm.log lines.

The prefix-cache hit rate is evidence only where it can mean thrash (M5 ladder, 2026-11-15):
* only on a LOADED engine (running >= load_min, or requests waiting) with enough prompt tokens in the window: a cold
  or lightly used engine's window is stale (flat-d after single-turn SciCode: 23-33% hit at 7 running);
* never on a MIXED engine (it served single-turn or replay sessions in the window): single-turn requests never reuse
  prefixes and a replay's first turn is a cold ~100k prefill, so the rate falls without any thrash (m5l-ab2: 58% at
  149 running, nothing waiting). There pressure is queue, KV and preemption only.

Policies:
* `aimd`: calm (hit >= hit_up or no evidence, waiting <= wait_calm, KV < kv_calm, running < run_max) for calm_ticks
  ticks while the limit binds -> x up (at least +min_step); pressure (a loaded engine with hit < hit_down, waiting >
  wait_max, KV > kv_max, or a preemption) -> min(limit, sessions) x down, at most once per cut_hold_sec (a cut acts
  only as sessions end, so the next reading would cut again on the same overload). Defaults: the M5 Super-hybrid
  preset (AB 50-80/engine at 93% hit, 240 -> 0%; CritPt 84 running 78% -> 173 27%; TB-Sci-lite 96 -> 92%, 192 -> 55%).
* `fixed`: the limit as configured.
* `kv_fit`: synth's admission.step (tbsched/kv_fit.py, verbatim), wired by the synth controller itself.
The single-turn cap of an engine follows `aimd` with the hit rate ignored (decode and KV bound, not prefix bound),
started from the KV it has per sequence (HLE ~100/engine, Omniscience/SciCode ~160-190 on Super).
"""
from dataclasses import dataclass, replace
import math
import re

AIMD = {
    'start': 96,          # first limit (sessions per engine)
    'floor': 32,
    'ceil': 160,          # measured healthy Super range (96-144); 320 once the collapse gate is on (sim: +45%)
    'up': 1.15,
    'min_step': 8,
    'down': 0.85,
    'calm_ticks': 2,      # raise only after this many calm ticks (new sessions take minutes to show in the metrics)
    'cut_hold_sec': 120,
    'binding': 0.9,       # the limit binds when sessions >= binding x limit
    'hit_up': 0.80,
    'hit_down': 0.65,
    'load_min': 60,       # running requests that make an engine "loaded"
    'hit_min_tokens': 200_000,   # prompt tokens the hit window needs to count as evidence
    'fresh_max': 0.2,     # no hit evidence while more than this share of the sessions started within the window
    'run_max': 110,
    'wait_calm': 5,
    'wait_max': 30,
    'kv_calm': 0.85,
    'kv_max': 0.97,
    'reprobe_sec': 2700,  # after a prefix-hit cut, the limit stays below the level that thrashed; one step up this often
    'ctx_prior': None,    # tokens a session holds over its life (suite prior): raise only while sessions x
    'ws_max': 0.9,        # max(prior, measured mean context) fits in ws_max x the engine's KV tiers
}

SINGLE = {**AIMD, 'start': 100, 'floor': 16, 'ceil': 400, 'run_max': 10**9, 'kv_calm': 0.85, 'kv_max': 0.97}


@dataclass
class Obs:
    """One engine over the last window. kv is 0..1 (in-flight KV); hit 0..1 or None."""
    running: float = 0.0
    waiting: float = 0.0
    kv: float = 0.0
    hit: float | None = None
    hit_tokens: float = 0.0
    gen_tok_s: float = 0.0
    preempted: bool = False
    sessions: int = 0       # agentic + replay sessions pinned here (from the pins, not vLLM)
    ctx_tokens: float = 0.0  # their contexts, summed (from the gate's footprints; 0 = unknown)
    kv_tokens: float = 0.0   # the engine's KV tiers (GPU + CPU offload) in tokens (0 = unknown)
    singles: int = 0        # single-turn requests in flight here
    mixed: bool = False     # served single-turn or replay sessions in the window
    fresh: int = 0          # sessions started on it within the window (their first turns are cold by construction)
    reuse: float | None = None   # client-side cache efficiency over reusable prefixes only (the gate's), if known
    stale: bool = False


def hit_evidence(o: Obs, p: dict) -> float | None:
    """The prefix reuse when it can mean thrash, on a loaded engine only. The client-side efficiency (cached tokens /
    the previous prompt, measured by the gate on agentic turns) is exact and ignores single-turn and first turns, so it
    counts even on a mixed engine. The server's hit rate counts only on an agentic-only engine whose sessions are not
    mostly new: a launch wave's first turns are cold by construction (sim: 96 new sessions read 0.17 and cut the
    limit, as on a cold M5 registry)."""
    if not loaded(o, p):
        return None
    if o.reuse is not None:
        return o.reuse
    if (o.mixed or o.hit is None or o.hit_tokens < p['hit_min_tokens']
            or o.fresh > p['fresh_max'] * max(o.sessions, 1)):
        return None
    return o.hit


def loaded(o: Obs, p: dict) -> bool:
    return o.running >= p['load_min'] or o.waiting > p['wait_calm']


def judge(o: Obs, p: dict) -> tuple[str, list[str]]:
    """('pressure' | 'calm' | 'hold', causes)."""
    if o.stale:
        return 'hold', ['stale_metrics']
    hit = hit_evidence(o, p)
    causes = []
    if hit is not None and hit < p['hit_down']:
        causes.append('prefix_hit')
    if o.waiting > p['wait_max']:
        causes.append('queue')
    if o.kv > p['kv_max']:
        causes.append('kv')
    if o.preempted:
        causes.append('preemption')
    if causes:
        return 'pressure', causes
    calm = ((hit is None or hit >= p['hit_up']) and o.waiting <= p['wait_calm'] and o.kv < p['kv_calm']
            and o.running < p['run_max'] and fits(o, p))
    return ('calm' if calm else 'hold'), []


def fits(o: Obs, p: dict) -> bool:
    """Room for the sessions' contexts to grow: a launch cohort is young and looks calm until its contexts grow together
    (sim: TB-Sci-lite-like sessions read calm up to 320/engine, then the hit rate went to 0 and stayed there for an
    hour). Measured mean context per session, floored by the suite's prior, x sessions <= ws_max x the KV tiers."""
    if not p.get('ctx_prior') or not o.kv_tokens:
        return True
    mean = max(p['ctx_prior'], o.ctx_tokens / o.sessions if o.sessions else 0.0)
    return (o.sessions + p['min_step']) * mean <= p['ws_max'] * o.kv_tokens


def aimd(st: dict, o: Obs, p: dict, used: int, now: float, committed: int | None = None) -> dict:
    """One tick for one engine. st: its persisted state (mutated); used: what the limit counts (sessions, or
    single-turn requests in flight); committed: used + credit granted but not yet spent, if known: the limit binds when
    that reaches binding x limit (replay 11-18: short sessions end between credit refills, so live sessions alone sat
    ~15% under an exhausted limit and one engine was never raised). Callers count unspent credit only up to the
    sessions started recently (min(unspent, fresh)): credit nobody spends (no worker slots) is no evidence the limit
    binds. Returns {limit, state, causes, changed}."""
    p = {**AIMD, **p}
    old = st.setdefault('limit', p['start'])
    state, causes = judge(o, p)
    st['calm'] = st.get('calm', 0) + 1 if state == 'calm' else 0
    hi = st.get('hi') or p['ceil']
    if st.get('hi') and now - st.get('hi_at', now) >= p['reprobe_sec']:
        hi = st['hi'] = min(p['ceil'], max(hi + p['min_step'], round(hi * p['up'])))   # probe one step past it again
        st['hi_at'] = now
    if state == 'pressure' and now - st.get('cut_at', -math.inf) >= p['cut_hold_sec']:
        st['limit'] = max(p['floor'], round(min(old, max(used, 1)) * p['down']))
        st['cut_at'] = now
        if 'prefix_hit' in causes:   # a learned ceiling below the level that thrashed (as slope search holds)
            st['hi'], st['hi_at'] = max(p['floor'], min(hi, st['limit'])), now
    elif (state == 'calm' and st['calm'] >= p['calm_ticks'] and old < hi
          and (used if committed is None else max(used, committed)) >= p['binding'] * old):
        st['limit'] = min(hi, max(old + p['min_step'], round(old * p['up'])))
        st['calm'] = 0
    return {'limit': st['limit'], 'state': state, 'causes': causes, 'changed': st['limit'] != old,
            'hi': st.get('hi')}


GUARD = {
    'settle_sec': 120,    # after a limit change, ignore this long (sessions start, caches fill) ...
    'window_sec': 120,    # ... plus the metrics window (Window(120)): a sample's rate covers the window before it
    'measure_sec': 300,   # ... then average generated tok/s over this long
    'min_gain': 0.02,     # keep raising while a step adds at least this share of generated tok/s, and at least
    'noise_z': 2.0,       # noise_z standard errors of the two windows' means (noise must not ratchet the limit up)
    'reprobe_sec': 1800,  # after a flat step or a cut, hold this long (doubling per failed probe, up to 8x)
    'run_max': 10 ** 9,   # no running-request cap and no GPU-KV-gauge cap: with a CPU KV tier both read high while
    'kv_calm': 1.01,      # throughput still grows (replay 11-18: 99% KV, 190 running, the most tok/s of all arms)
    'kv_max': 1.01,
    'wait_max': 64,
    'ceil': 400,
}


def params(overrides: dict | None = None) -> dict:
    """A limit policy's full settings: AIMD, plus GUARD for policy "guard", plus the overrides. Callers that merge
    defaults themselves must use this (merging AIMD over GUARD silently gave the guard aimd's caps)."""
    o = dict(overrides or {})
    return {**AIMD, **(GUARD if o.get('policy') == 'guard' else {}), **o}


def guard(st: dict, o: Obs, p: dict, used: int, now: float, committed: int | None = None) -> dict:
    """policy = "guard": aimd's cuts on thrash evidence (a loaded engine's prefix hit, CPU tier included, below
    hit_down; a preemption; more than wait_max waiting), but a raise only while each step adds generated tok/s:
    at least min_gain and noise_z standard errors of the two windows (measured after settle_sec + window_sec, over
    measure_sec). A slope search on the session limit, as kv_fit's slope search does on the KV target. A step that
    does not pay returns to the previous limit, which becomes a ceiling until a later probe (hold reprobe_sec, doubling
    per failed probe up to 8x). Zero throughput is no evidence for a raise."""
    p = {**AIMD, **GUARD, **p}
    old = st.setdefault('limit', p['start'])
    state, causes = judge(o, p)
    g = st.get('guard') or {'phase': 'measure', 'since': now, 'sum': 0.0, 'sq': 0.0, 'n': 0, 'base': None}
    out = lambda: {'limit': st['limit'], 'state': state, 'causes': causes, 'changed': st['limit'] != old,  # noqa: E731
                   'hi': st.get('hi'), 'phase': st['guard']['phase']}

    def hold(now, failed):
        st['fails'] = st.get('fails', 0) + 1 if failed else 0
        wait = p['reprobe_sec'] * 2 ** min(st.get('fails', 0), 3)
        st['guard'] = {'phase': 'hold', 'since': now, 'sum': 0.0, 'sq': 0.0, 'n': 0, 'base': None, 'until': now + wait}

    if state == 'pressure':
        if now - st.get('cut_at', -math.inf) >= p['cut_hold_sec']:
            st['limit'] = max(p['floor'], round(min(old, max(used, 1)) * p['down']))
            st['cut_at'] = now
            st['hi'] = st['limit']
        hold(now, failed=False)
        return out()
    if g['phase'] == 'hold':
        if now < g['until']:
            st['guard'] = g
            return out()
        st['hi'] = None                    # probe one step past the learned ceiling again
        g = {'phase': 'measure', 'since': now, 'sum': 0.0, 'sq': 0.0, 'n': 0, 'base': None}
    if now - g['since'] >= p['settle_sec'] + p['window_sec'] and not o.stale:
        g['sum'], g['sq'], g['n'] = g['sum'] + o.gen_tok_s, g['sq'] + o.gen_tok_s ** 2, g['n'] + 1
    st['guard'] = g
    if now - g['since'] < p['settle_sec'] + p['window_sec'] + p['measure_sec'] or not g['n']:
        return out()
    n, gen = g['n'], g['sum'] / g['n']
    se = math.sqrt(max(0.0, g['sq'] / n - gen * gen) / n) if n > 1 else 0.0
    base = g['base']
    if gen <= 0 or (base is not None and base[1] <= 0):
        hold(now, failed=False)            # no throughput reading: no evidence either way
        return out()
    if base is not None:
        need = max(p['min_gain'], p['noise_z'] * math.hypot(se / gen, base[2] / base[1]))
        if gen < base[1] * (1 + need):     # the last step did not pay: back, a ceiling there, and hold
            st['limit'], st['hi'] = base[0], base[0]
            hold(now, failed=True)
            st['guard']['last'] = [base[0], round(base[1]), old, round(gen), round(need, 3)]
            return out()
        st['fails'] = 0
    binding = (used if committed is None else max(used, committed)) >= p['binding'] * old
    top = min(p['ceil'], st.get('hi') or p['ceil'])
    if binding and old < top:
        st['limit'] = min(top, max(old + p['min_step'], round(old * p['up'])))
        st['guard'] = {'phase': 'measure', 'since': now, 'sum': 0.0, 'sq': 0.0, 'n': 0, 'base': [old, gen, se]}
    else:                                  # not filled, or at the ceiling: no step to judge; measure afresh
        st['guard'] = {'phase': 'measure', 'since': now, 'sum': 0.0, 'sq': 0.0, 'n': 0, 'base': None}
    return out()


def limit_step(st: dict, o: Obs, p: dict, used: int, now: float, committed: int | None = None) -> dict:
    """The session limit of one engine under p['policy']: 'aimd' (default) or 'guard'."""
    if (p or {}).get('policy') == 'guard':
        return guard(st, o, p, used, now, committed)
    return aimd(st, o, p, used, now, committed)


def single_cap(st: dict, o: Obs, p: dict, now: float) -> dict:
    """The single-turn in-flight cap of an engine: aimd on queue, KV and preemption only."""
    return aimd(st, replace(o, mixed=True, reuse=None), {**SINGLE, **p}, o.singles, now)


def single_start(kv_capacity_tokens: float, p90_sequence_tokens: float, kv_frac: float = 0.9) -> int:
    """First single-turn cap: the sequences that fit in the KV at their p90 length."""
    return max(1, int(kv_frac * kv_capacity_tokens / max(p90_sequence_tokens, 1)))


# ------------------------------------------------------------------------------------------------- /metrics signals
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]Inf)\s*$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

COUNTERS = ('vllm:generation_tokens_total', 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total',
            'vllm:num_preemptions_total')
# Prompt tokens by where they came from. With a CPU KV tier, prefix_cache_hits counts GPU hits only: prefixes restored
# from the tier read as misses (replay 11-18: GPU hit 0.00 at 99% KV while generation was the highest of all arms), so
# the hit rate is 1 - computed / all prompt tokens whenever these series exist.
BY_SOURCE = 'vllm:prompt_tokens_by_source_total'
SOURCES = ('local_compute', 'local_cache_hit', 'external_kv_transfer')
GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc')
FRACTIONS = ('vllm:kv_cache_usage_perc',)   # a fraction per engine: the max over series (data-parallel ranks), not the sum


def parse_metrics(text: str, match: dict | None = None) -> dict:
    """Prometheus text -> {name: value summed over series (FRACTIONS: the max)}; `match` keeps only series with those
    label values (e.g. {'replica': '1'} behind a pooled gate, {'engine': '0'} for one DP rank)."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == '#':
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if match:
            got = dict(_LABEL.findall(labels or ''))
            if any(got.get(k) != str(v) for k, v in match.items()):
                continue
        try:
            v = float(value)
        except ValueError:
            continue
        if name == BY_SOURCE:
            name = f'{name}:{dict(_LABEL.findall(labels or "")).get("source", "")}'
        if math.isfinite(v):
            out[name] = max(out.get(name, v), v) if name in FRACTIONS else out.get(name, 0.0) + v
    return out


class Window:
    """Counter deltas over the last `seconds` of samples of one engine's metrics."""

    def __init__(self, seconds: float = 120.0):
        self.seconds, self.samples = seconds, []

    def add(self, t: float, metrics: dict):
        self.samples.append((t, metrics))
        while len(self.samples) > 2 and t - self.samples[1][0] >= self.seconds:
            self.samples.pop(0)

    def obs(self, **pins) -> Obs:
        """Obs from the window (gauges from the last sample); pins: sessions, singles, mixed."""
        if not self.samples:
            return Obs(stale=True, **pins)
        t1, m1 = self.samples[-1]
        t0, m0 = self.samples[0]
        dt = t1 - t0
        d = {k: max(0.0, m1.get(k, 0.0) - m0.get(k, 0.0)) for k in COUNTERS}
        q = d['vllm:prefix_cache_queries_total']
        hits = d['vllm:prefix_cache_hits_total']
        if any(f'{BY_SOURCE}:{x}' in m1 for x in SOURCES) and any(f'{BY_SOURCE}:{x}' in m0 for x in SOURCES):
            # every prompt-token source, the CPU tier included (both ends: a counter that appears mid-window, after an
            # engine restart or upgrade, is not one window's delta)
            src = {x: max(0.0, m1.get(f'{BY_SOURCE}:{x}', 0.0) - m0.get(f'{BY_SOURCE}:{x}', 0.0)) for x in SOURCES}
            q = sum(src.values())
            hits = q - src['local_compute']
        # gauges: mean over the window (waiting: its max), as the M5 controller read ~2 min of log lines
        g = self.samples[1:] if len(self.samples) > 1 else self.samples
        mean = lambda k: sum(m.get(k, 0.0) for _, m in g) / len(g)   # noqa: E731
        return Obs(running=mean('vllm:num_requests_running'),
                   waiting=max(m.get('vllm:num_requests_waiting', 0.0) for _, m in g),
                   kv=mean('vllm:kv_cache_usage_perc'),
                   hit=(hits / q) if q > 0 else None, hit_tokens=q,
                   gen_tok_s=d['vllm:generation_tokens_total'] / dt if dt > 0 else 0.0,
                   preempted=d['vllm:num_preemptions_total'] > 0, **pins)

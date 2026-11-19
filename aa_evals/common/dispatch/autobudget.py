"""Auto budget (#378): the dispatcher's in-flight budget found by hill-climbing on the server's generated tok/s.

A fixed budget is tuned by hand from the server's knee, and c30 missed in both directions: Super at 182 ran above its
knee (generated tok/s -25% for 4.2 h above ~128 running, KV p90 >= 90% with requests waiting), while Gemma, Qwen3.6
and Qwen3.5-4B were still speeding up at their budgets. `dispatcher init --budget N` now takes N as the start value
and, with `--metrics`, moves the budget between `--budget-min` and `--budget-max` (default N/2 .. 2N;
`--no-auto-budget` keeps it fixed). The controller is pure (no I/O): the dispatcher feeds it one engine sample per
tick and asks it for the budget, and the tests drive it with simulated servers.

Window: the budget is held for DWELL (10 min). The first SETTLE (3 min) after a change is not measured (new attempts
start and prefill; finished ones leave). The rest gives: mean generated tok/s (from vllm:generation_tokens_total),
KV p90 (the fullest engine), the share of samples with requests waiting, preemptions, mean running, the share of ticks
the budget was filled (in flight + unspent credit within 5% of it, with work pending) and the share the serving gate
held new conversations.

Decision at the end of each window, in order:
1. overload: any preemption, or KV p90 >= 90% with requests waiting in >= 10% of samples (Super above its knee):
   -15% on preemptions, else -10%. Raising to >= 95% of the overloaded level is blocked for BLOCK (30 min).
2. not filled (budget filled in < 80% of ticks: ramp-up, drain tail, Pi worker slots, or the gate): no information
   about the budget; hold (a raise still to be judged is judged in the next filled window). So the budget never
   rises past what the lanes' capacity (Pi workers x slots) can fill. Filled is against the effective budget (after
   the per-tick back-off below).
3. engine limit: mean running >= 95% of the scraped engines' max-num-seqs, or waiting in >= 30% of samples (with
   KV below 90%), or the gate holding in >= 50% of ticks: more budget only queues in the engine or at the gate. A
   raise that led here is reverted (and blocked for BLOCK); otherwise hold. This is where Qwen3.6 (max-num-seqs 128)
   stops.
4. after a raise: kept when tok/s gained >= GAIN (3% for +10%: an elasticity of ~0.3, the knee: beyond it every
   attempt slows almost in proportion, so more of them reach the 2 h cap), and raised again if the window is healthy
   (no preemption, KV p90 < 85%, waiting in < 10% of samples, running < 90% of max-num-seqs); reverted and blocked
   for BLOCK when it gained less (tok/s stopped improving or fell).
5. otherwise (start, after a hold, a revert or a cut): raise by 10% when healthy and not blocked; else hold.
The budget never leaves [lo, hi]: a raise stops at hi, and there it holds (logged). The dispatcher's per-tick back-off (adaptive(): a
preemption or waiting at KV >= 95% cuts the effective budget at once) still applies under it, and the gate back-off
(no new credit while the gate holds) is unchanged: the controller only sets the ceiling those work under.
"""
from __future__ import annotations

import math

DWELL = 600.0     # s a budget is held before it is judged
SETTLE = 180.0    # s after a change not measured
UP, DOWN, CUT = 1.10, 0.90, 0.85
GAIN = 0.03       # log tok/s gain a +10% raise must show to be kept (elasticity ~0.3)
BLOCK = 1800.0    # s a failed level stays blocked (then it is probed again)
KV_HIGH, KV_OK = 0.90, 0.85
WAIT_HIGH, WAIT_FULL = 0.10, 0.30
SEQ_FULL, SEQ_OK = 0.95, 0.90
FILLED, GATE_FULL = 0.80, 0.50
MAX_DEFER = 2       # unfilled windows a raise waits to be judged
MIN_SAMPLES = 0.5   # share of the measured part of a window that must have engine samples


def parse_metrics(text: str, strict: bool = True) -> dict:
    """vLLM Prometheus text -> {gen_total, preempt_total, running, waiting (sums over engines), kv (max), engines}.
    strict: no generation counter is an error (the auto budget needs it); otherwise absent metrics are 0."""
    vals: dict[str, list[tuple[str, float]]] = {}
    want = ('vllm:generation_tokens_total', 'vllm:num_preemptions_total', 'vllm:num_requests_running',
            'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc')
    for ln in text.splitlines():
        for name in want:
            if ln.startswith(name + '{') or ln.startswith(name + ' '):
                try:
                    v = float(ln.split()[-1])
                except (ValueError, IndexError):
                    continue
                labels = ln[len(name):ln.find('}') + 1] if ln.startswith(name + '{') else ''
                vals.setdefault(name, []).append((labels, v))
    s = lambda n: sum(v for _, v in vals.get(n, []))
    if strict and 'vllm:generation_tokens_total' not in vals:
        raise ValueError('no vllm:generation_tokens_total in /metrics')
    return {'gen_total': s('vllm:generation_tokens_total'), 'preempt_total': s('vllm:num_preemptions_total'),
            'running': s('vllm:num_requests_running'), 'waiting': s('vllm:num_requests_waiting'),
            'kv': max((v for _, v in vals.get('vllm:kv_cache_usage_perc', [])), default=0.0),
            'engines': max(1, len({lb for lb, _ in vals.get('vllm:num_requests_running', [])}))}


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = q * (len(xs) - 1)
    lo = math.floor(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


class AutoBudget:
    """Hill-climbing budget controller. sample() once per tick, step() once per tick (returns the budget)."""

    def __init__(self, budget: int, lo: int, hi: int, max_num_seqs: int | None = None, now: float = 0.0,
                 state: dict | None = None):
        self.lo, self.hi = int(lo), int(max(lo, hi))
        self.max_num_seqs = max_num_seqs
        self.budget = min(self.hi, max(self.lo, int(budget)))
        self.t_step = now
        self.last: str | None = None            # 'up' when the current level is a raise being judged
        self.prev: tuple[int, float] | None = None   # (budget, log tok/s) of the window before the raise
        self.block: tuple[int, float] | None = None  # (level, until): no raise to >= 95% of level
        self.samples: list[dict] = []
        self.raw: tuple[float, float, float] | None = None   # (t, gen_total, preempt_total) of the last scrape
        self.deferred = 0   # unfilled windows since the current raise
        self.note = ''
        if state:
            self.budget = min(self.hi, max(self.lo, int(state.get('budget', self.budget))))
            self.block = tuple(state['block']) if state.get('block') else None

    def to_json(self) -> dict:
        return {'budget': self.budget, 'lo': self.lo, 'hi': self.hi, 'max_num_seqs': self.max_num_seqs,
                'block': list(self.block) if self.block else None, 'last': self.last, 'note': self.note,
                'since': round(self.t_step, 1)}

    def sample(self, t: float, m: dict | None, filled: bool, gate_hold: bool) -> None:
        """One tick: m = parse_metrics() of the engine scrape (None: scrape failed); filled = the dispatcher had the
        budget filled with work pending; gate_hold = the serving gate held new conversations."""
        if m is None:
            return
        raw, self.raw = self.raw, (t, m['gen_total'], m['preempt_total'])
        if raw is None or t <= raw[0] or m['gen_total'] < raw[1]:   # first scrape, or an engine restart
            return
        self.samples.append({'t': t, 'tps': (m['gen_total'] - raw[1]) / (t - raw[0]), 'running': m['running'],
                             'waiting': m['waiting'], 'kv': m['kv'], 'engines': m['engines'],
                             'preempt': max(0.0, m['preempt_total'] - raw[2]), 'filled': bool(filled),
                             'gate': bool(gate_hold)})

    def stats(self, tick: float) -> dict | None:
        w = [s for s in self.samples if s['t'] >= self.t_step + SETTLE]
        if len(w) < MIN_SAMPLES * (DWELL - SETTLE) / max(tick, 1.0):
            return None
        n = len(w)
        tps = sum(s['tps'] for s in w) / n
        mns = self.max_num_seqs * max(s['engines'] for s in w) if self.max_num_seqs else None
        run = sum(s['running'] for s in w) / n
        return {'n': n, 'tps': tps, 'log_tps': math.log(max(tps, 1.0)), 'kv90': quantile([s['kv'] for s in w], 0.9),
                'wait': sum(1 for s in w if s['waiting'] > 0) / n, 'preempt': sum(s['preempt'] for s in w),
                'running': run, 'seq_share': run / mns if mns else None,
                'filled': sum(1 for s in w if s['filled']) / n, 'gate': sum(1 for s in w if s['gate']) / n}

    def blocked(self, level: int, now: float) -> bool:
        return bool(self.block and now < self.block[1] and level >= 0.95 * self.block[0])

    def _set(self, b: int, now: float, why: str, last: str | None) -> tuple[int, str]:
        old = self.budget
        self.budget = min(self.hi, max(self.lo, int(round(b))))
        self.t_step, self.samples, self.last = now, [], last if self.budget != old else None
        self.deferred = 0
        self.note = f'{old} -> {self.budget}: {why}' if self.budget != old else f'hold {old}: {why}'
        return self.budget, self.note

    def step(self, now: float, tick: float = 15.0) -> tuple[int, str | None]:
        """The budget for this tick and, when the window just ended, what was decided (for the log)."""
        if now - self.t_step < DWELL:
            return self.budget, None
        st = self.stats(tick)
        if st is None:   # engine not answering: keep the budget, measure again
            self.t_step, self.samples, self.last, self.prev = now, [], None, None
            return self.budget, f'hold {self.budget}: too few engine samples'
        b = self.budget
        desc = (f"tok/s {st['tps']:.0f}, KV p90 {st['kv90']:.2f}, waiting {st['wait']:.0%}, preempt "
                f"{st['preempt']:.0f}, running {st['running']:.0f}, filled {st['filled']:.0%}, gate {st['gate']:.0%}")
        healthy = (st['preempt'] == 0 and st['kv90'] < KV_OK and st['wait'] < WAIT_HIGH
                   and (st['seq_share'] is None or st['seq_share'] < SEQ_OK))
        up = min(self.hi, int(round(b * UP)))
        if st['preempt'] > 0 or (st['kv90'] >= KV_HIGH and st['wait'] >= WAIT_HIGH):
            self.block, self.prev = (b, now + BLOCK), None
            return self._set(b * (CUT if st['preempt'] > 0 else DOWN), now, 'overload (' + desc + ')', None)
        if st['filled'] < FILLED:
            self.deferred += 1
            if self.last == 'up' and self.prev and self.deferred <= MAX_DEFER:
                # a raise not judged yet (e.g. grown Pi workers starting): judged in the next filled window, unless
                # that is so late that the tok/s before it no longer compares (then it is just kept)
                self.t_step, self.samples = now, []
                self.note = f'hold {b}: budget not filled, the raise is judged in the next window (' + desc + ')'
                return b, self.note
            self.prev = None
            return self._set(b, now, 'budget not filled (' + desc + ')', None)
        if (st['seq_share'] is not None and st['seq_share'] >= SEQ_FULL) or st['wait'] >= WAIT_FULL \
                or st['gate'] >= GATE_FULL:
            why = 'engine/gate limit (' + desc + ')'
            if self.last == 'up' and self.prev:
                self.block, back = (b, now + BLOCK), self.prev[0]
                self.prev = None
                return self._set(back, now, why + ': raise reverted', None)
            self.prev = None
            return self._set(b, now, why, None)
        if self.last == 'up' and self.prev:
            gain = st['log_tps'] - self.prev[1]
            if gain < GAIN:
                self.block, back = (b, now + BLOCK), self.prev[0]
                self.prev = None
                return self._set(back, now, f'tok/s gain {gain:+.1%} after the raise: reverted (' + desc + ')', None)
            note = f'tok/s gain {gain:+.1%}; '
        else:
            note = ''
        self.prev = (b, st['log_tps'])
        if healthy and up > b and not self.blocked(up, now):
            return self._set(up, now, note + 'raise (' + desc + ')', 'up')
        why = ('at --budget-max' if up <= b else 'blocked after a failed level' if self.blocked(up, now)
               else 'not healthy enough to raise')
        self.prev = None
        return self._set(b, now, note + why + ' (' + desc + ')', None)

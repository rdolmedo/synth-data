"""Context-aware turn gate: the admission policy shared by the live eval (admission.py's gate server, asked by each
rollout's proxy: sticky_proxy --gate-key) and the replay (serve_snowball/replay.py --ctx-gate v3), so what a replay
measures is what an eval runs. Pure logic: callers pass the clock and do the I/O.

Every rollout of an eval is admitted (sandboxes are cheap) and each turn is gated on the cache tokens the rollouts'
contexts occupy, against vLLM's KV cache (kv_cache_size_tokens); no per-GPU or per-benchmark constant:

* allowed set: the rollouts, in claim order, whose contexts fit in fill x the cache. An allowed rollout sends its turns
  without waiting; the others wait at their turn boundary. The set is sticky: it shrinks (youngest first) only when its
  contexts outgrow the budget by `slack`, and it grows (oldest waiting first) only while the next one fits, so a rollout
  is not held and released over and over as contexts grow (each release can cost a full re-prefill).
* parking: a revoked rollout queues behind every rollout never revoked, until it is admitted again. Every attempt
  starts at once, so in claim order alone the residents in mid-run are the oldest rollouts of that start cohort, all
  with large contexts (fewer fit, each decodes slower); parked, the fresh attempts that replace finished ones get in,
  as under a cap, and the parked ones resume later with their head start.
* context of a rollout: prompt + completion of its last finished request, or the prompt of the request it is about to
  send (`need`, the proxy's estimate) when larger.
* pacing: a rollout that waited or was revoked for more than `cold_after` seconds has probably lost its cached prefix;
  at most `cold_rate` such re-admissions per `cold_window` seconds, so a burst of releases cannot evict the residents
  with simultaneous re-prefills.
* fill steers itself by the eviction signal. Every `adjust_every` seconds: cache efficiency = prefix-cache hits (vLLM
  --enable-prompt-tokens-details cached_tokens) / the tokens the rollouts could have reused (their previous prompt, down
  to whole cache blocks: vLLM caches a hybrid model's prefix in blocks of `block` tokens, 1056 for Qwen3.5 with FP8 KV),
  over the turns that were neither held nor revoked since the previous one and that followed it within `eff_gap`
  seconds. With the cache near full vLLM's LRU evicts the contexts of rollouts sitting in long tool calls first, which a
  cap lives with too (2x H100 runs TB2.1 at 0.93 overall); an eviction within a short gap is thrash. Below `eff_cut` ->
  fill x 0.93 (not below `lo`), at most once per `cooldown` seconds: a cut revokes rollouts whose contexts stay in vLLM's
  LRU as the most recently used, so for minutes the residents are evicted first and each reading would cut again (v3
  went 0.70 -> 0.45 in five minutes). At >= `eff_raise` while a rollout waited -> fill + `raise_step`.
* learned ceiling (`learn_hi`, default): the fill rises past its start while the cache keeps the contexts; a cut there
  lowers the ceiling to one raise step below the fill that thrashed, never below the start (below it the gate acts as
  with a fixed ceiling at the start). How many context tokens a cache really holds depends on the model: Snowball's
  sliding-window layers run 1.0-1.2x vLLM's reported kv_cache_size_tokens without thrash, Qwen3.5 thrashes past ~0.7.
  learn_hi=False is v9 (ceiling = the start: when the queue drains every remaining rollout is allowed at once).
* data-parallel engines have separate caches: one gate per engine (gate_server.py, replay --dp-ranks).

Replays (serve_q35/README.md, serve_snowball/README.md): never behind the family's cap beyond run-to-run noise (~1 min)
at any completion quantile, and far ahead where a cap tuned on other traffic thrashes: Qwen3.5-4B on TB2.1 / TB-Lite /
a DeepSeek SFT mix on 1x B200 and 2x H100; Snowball and Datakit on 1x B200, 2x B200 and 2x H100 (Snowball's own TB2.1
on one B200: 48/70/77 vs the cap's 75/96/101 min to 50/90/95%).
"""


class TurnGate:
    def __init__(self, capacity=None, fill=0.7, lo=0.4, hi=None, slack=0.10, cold_after=60.0, cold_rate=4,
                 cold_window=15.0, adjust_every=60.0, block=1, cooldown=180.0, eff_cut=0.95, eff_raise=0.98,
                 raise_step=0.02, eff_window=None, park=True, eff_gap=60.0, learn_hi=True):
        self.capacity, self.fill, self.lo, self.slack, self.block = capacity, fill, lo, slack, block
        # learn_hi: no fixed ceiling above the starting fill; a cut there lowers the ceiling to one raise step below the
        # fill that thrashed, never below the start (below it the gate behaves as with hi = start)
        self.learn_hi, self.fill0 = learn_hi, fill
        self.hi = (float('inf') if learn_hi else fill) if hi is None else hi
        self.cooldown, self.eff_cut, self.eff_raise, self.raise_step = cooldown, eff_cut, eff_raise, raise_step
        self.eff_window = eff_window or adjust_every   # efficiency over the last eff_window seconds
        self.park, self.eff_gap = park, eff_gap
        self.cold_after, self.cold_rate, self.cold_window, self.adjust_every = cold_after, cold_rate, cold_window, adjust_every
        self.r = {}             # key -> rollout state (see register)
        self.allowed = set()
        self.cached = self.ideal = 0.0   # cumulative efficiency sums
        self.hist = []          # (time, cached, ideal) samples over the last efficiency window
        self.waited = False     # some rollout waited since the last adjustment
        self.cold = []          # times of cold re-admissions (pacing)
        self.last_adjust = self.last_cut = None
        self.eff = None
        self.n_revoked = self.n_cold = 0

    # --- rollouts -------------------------------------------------------------------------------------------------
    def register(self, key, order=None, now=0.0):
        if key not in self.r:
            if order is None:
                order = now   # first heard of at its first turn: it queues behind the rollouts already known
            self.r[key] = {'order': order, 'ctx': 0, 'need': 0, 'prev': 0, 'waiting': None, 'out_since': None,
                           'excluded': False, 'parked': False, 'gap': None, 'done_at': None}
        return self.r[key]

    def sync(self, keys, now):
        """keys: {key: order} of the live rollouts; the others are gone (finished or lost)."""
        for k, o in keys.items():
            self.register(k, o, now)
        for k in [k for k in self.r if k not in keys]:
            self.drop(k)

    def drop(self, key):
        self.r.pop(key, None)
        self.allowed.discard(key)

    def ask(self, key, need, now):
        """Rollout `key` is at a turn boundary with a request of about `need` prompt tokens. True: send it now."""
        x = self.register(key, now=now)
        x['need'] = int(need or 0)
        if x['waiting'] is None:
            x['waiting'] = now
        self._rebalance(now)
        if key in self.allowed:
            if now - x['waiting'] >= 2:
                x['excluded'] = True   # its cache may have been evicted while it waited: by design, not a miss
            x['waiting'] = None
            x['gap'] = None if x['done_at'] is None else now - x['done_at']
            return True
        self.waited = True
        return False

    def finished(self, key, prompt, completion, cached, now):
        """A request of `key` finished: prompt/completion tokens and its prefix-cache hit (None if unknown)."""
        x = self.register(key, now=now)
        reusable = min(x['prev'], prompt) // self.block * self.block
        short = self.eff_gap is None or (x['gap'] is not None and x['gap'] <= self.eff_gap)
        if not x['excluded'] and reusable and cached is not None and short:
            self.ideal += reusable
            self.cached += min(cached, reusable)
        x.update(prev=prompt, ctx=prompt + completion, need=0, excluded=False, done_at=now)
        if key not in self.allowed:
            x['excluded'] = True   # revoked while its request ran: it waits next, its cache unprotected
        self._rebalance(now)

    # --- policy ---------------------------------------------------------------------------------------------------
    def _fp(self, k):
        x = self.r[k]
        return max(x['ctx'], x['need'])

    def _rebalance(self, now):
        if not self.capacity:
            self.allowed = set(self.r)
            return
        budget = self.fill * self.capacity
        rank = lambda: sorted(self.r, key=lambda k: (self.park and self.r[k]['parked'], self.r[k]['order']))  # noqa: E731
        order = rank()
        foot = sum(self._fp(k) for k in self.allowed)
        if foot > budget * (1 + self.slack):
            for k in reversed(order):
                if foot <= budget or len(self.allowed) <= 1:
                    break
                if k in self.allowed:
                    self.allowed.discard(k)
                    foot -= self._fp(k)
                    self.r[k].update(out_since=now, excluded=True, parked=True)
                    self.n_revoked += 1
            order = rank() if self.park else order   # the rollouts just revoked queue behind
        self.cold = [t for t in self.cold if now - t < self.cold_window]
        for k in order:
            if k in self.allowed:
                continue
            x, f = self.r[k], self._fp(k)
            if self.allowed and foot + f > budget:
                break   # strict order: the oldest waiting rollout goes first (parked ones last)
            since = x['out_since'] if x['out_since'] is not None else x['waiting']
            cold = x['ctx'] > 0 and since is not None and now - since > self.cold_after
            if cold and len(self.cold) >= self.cold_rate:
                break
            self.allowed.add(k)
            foot += f
            x['out_since'] = None
            x['parked'] = False
            if cold:
                self.cold.append(now)
                self.n_cold += 1

    def tick(self, now):
        """Call every few seconds: adjusts fill once per adjust_every, then rebalances."""
        self.hist.append((now, self.cached, self.ideal))
        while len(self.hist) > 2 and now - self.hist[1][0] >= self.eff_window:
            self.hist.pop(0)
        if self.last_adjust is None:
            self.last_adjust = now
        if now - self.last_adjust >= self.adjust_every:
            (t0, c0, i0), (t1, c1, i1) = self.hist[0], self.hist[-1]
            self.eff = (c1 - c0) / (i1 - i0) if t1 - t0 >= 0.75 * self.eff_window and i1 - i0 > 2e5 else None
            if self.eff is not None and self.eff < self.eff_cut:
                if self.last_cut is None or now - self.last_cut >= self.cooldown:
                    if self.learn_hi:
                        self.hi = min(self.hi, max(self.fill0, self.fill - self.raise_step))
                    self.fill = max(min(self.fill, self.lo), self.fill * 0.93)
                    self.last_cut = now
            elif self.eff is not None and self.eff >= self.eff_raise and self.waited:
                self.fill = min(self.hi, self.fill + self.raise_step)
            self.last_adjust, self.waited = now, False
        self._rebalance(now)

    def stats(self):
        waiting = sum(1 for x in self.r.values() if x['waiting'] is not None)
        return {'ctx_fill': round(self.fill, 3), 'ctx_hi': None if self.hi == float('inf') else round(self.hi, 3),
                'ctx_budget': int(self.fill * (self.capacity or 0)),
                'ctx_footprint': sum(self._fp(k) for k in self.allowed),
                'ctx_total': sum(self._fp(k) for k in self.r), 'ctx_allowed': len(self.allowed),
                'ctx_live': len(self.r), 'ctx_waiting': waiting,
                'ctx_parked': sum(1 for k, x in self.r.items() if x['parked'] and k not in self.allowed),
                'ctx_eff': None if self.eff is None else round(self.eff, 3),
                'ctx_revoked': self.n_revoked, 'ctx_cold': self.n_cold}

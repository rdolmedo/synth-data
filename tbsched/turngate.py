"""eval_q35's context-aware turn gate (eval_q35/turn_gate.py), moved to the core verbatim: the class below is
byte-identical to that file's (tbsched/tests/test_gate.py::Copies checks it until eval_q35 imports it from here).
Its design notes are in eval_q35/turn_gate.py's docstring; tbsched/gate.py wraps it as #477's collapse gate.
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

"""Per-item duration predictions for the #344 dispatcher (longest attempts first).

A table holds suites.<suite>.<item> = {model: seconds}. Two tables ship (built by `build_durations.py`):
- `durations_v2.json` (DEFAULT_TABLE, #378): c30's measured attempts (Pi harness, no compaction, text-only, 2 h cap).
  Full runs only (canaries ran on an idle server). Each value is the model's latent median seconds on the item: a
  censored lognormal fit, so attempts cut at the 2 h cap count as "longer than", not as 7200 s. Single-turn values exclude the recorded server gate wait. Items, models
  or suites c30 lacks come from v1, mapped to c30 per model and suite (log-linear). It also has `lognormal` (per-suite sigma,
  tau, cap) and `aliases` (other names of a model key).
- `durations_v1.json` (#344): mean seconds per attempt from the M3.6 replay (GDPval/CritPt re-timed for no
  compaction; single-request suites from tokens / per-request speed). A dispatch dir keeps the table path it recorded.

predict(suite, item):
- the geometric mean over the other models that ran the item, blended in logs with this model's own value when it has
  one (`model`; weight `own_weight`, 0.5 unless the table says otherwise);
- tables with `lognormal`: the expected seconds under the cap, E[min(T, cap)] for T lognormal around that median, with
  log-sd sqrt(sigma^2 + (1 - w)^2 tau^2) (sigma: one attempt around its model-item median; tau: a model-item median
  around the additive model x item fit, w = 0 without an own value), times the suite's `mean_factor` (c30's capped
  seconds / the fit's expectation; > 1 where the tail is heavier than lognormal), at most the cap;
- an item no model ran: the suite's median prediction; an unknown suite: DEFAULT_SEC.
Online rescaling by this run's own completions happens in the dispatcher (one factor per suite), so it changes the
interleaving of suites and never the order within a suite.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

DEFAULT_TABLE = Path(__file__).with_name('durations_v2.json')
V1_TABLE = Path(__file__).with_name('durations_v1.json')
DEFAULT_SEC = 600.0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def capped_mean(mu: float, sd: float, cap: float | None) -> float:
    """E[min(T, cap)] for log T ~ N(mu, sd^2); no cap: E[T]."""
    if sd <= 0:
        return min(math.exp(mu), cap) if cap else math.exp(mu)
    mean = math.exp(mu + sd * sd / 2)
    if not cap:
        return mean
    lc = math.log(cap)
    return mean * _phi((lc - mu - sd * sd) / sd) + cap * (1.0 - _phi((lc - mu) / sd))


class Durations:
    def __init__(self, path: str | Path | None = None, model: str | None = None):
        self.path = Path(path or DEFAULT_TABLE)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.table = json.loads(raw)
        model = model or None
        self.model = (self.table.get('aliases') or {}).get(model, model)
        if self.model and self.model not in self.table.get('models', []):
            raise ValueError(f'unknown duration model {model!r} (table models: {self.table.get("models")})')
        self.lognormal = self.table.get('lognormal')
        self.own_weight = float(self.table.get('own_weight', 0.5))
        self._median: dict[str, float] = {}

    def describe(self) -> dict:
        return {'path': str(self.path), 'sha256': self.sha256, 'name': self.table.get('name'), 'model': self.model}

    def _item(self, per: dict, suite: str | None = None) -> float | None:
        others = [math.log(max(1.0, v)) for m, v in per.items() if m != self.model and v]
        own = per.get(self.model) if self.model else None
        if not others and not own:
            return None
        w = 1.0 if not others else (self.own_weight if own else 0.0)
        g = (sum(others) / len(others)) if others else 0.0
        if own:
            g = (1 - w) * g + w * math.log(max(1.0, own))
        if not self.lognormal:
            return math.exp(g)
        sig = float((self.lognormal.get('sigma') or {}).get(suite, self.lognormal.get('sigma_default', 0.0)))
        tau = float((self.lognormal.get('tau') or {}).get(suite, self.lognormal.get('tau_default', 0.0)))
        cap = (self.lognormal.get('cap_sec') or {}).get(suite, self.lognormal.get('cap_default'))
        x = capped_mean(g, math.sqrt(sig * sig + ((1 - w) * tau) ** 2), cap)
        x *= float((self.lognormal.get('mean_factor') or {}).get(suite, 1.0))
        return min(x, cap) if cap else x

    def suite_median(self, suite: str) -> float:
        if suite not in self._median:
            vals = [x for x in (self._item(p, suite) for p in (self.table['suites'].get(suite) or {}).values()) if x]
            self._median[suite] = statistics.median(vals) if vals else DEFAULT_SEC
        return self._median[suite]

    def known(self, suite: str, item: str) -> bool:
        return str(item) in (self.table['suites'].get(suite) or {})

    def predict(self, suite: str, item: str) -> float:
        per = (self.table['suites'].get(suite) or {}).get(str(item))
        x = self._item(per, suite) if per else None
        return round(x if x is not None else self.suite_median(suite), 1)

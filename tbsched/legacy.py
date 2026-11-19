"""The 11-15 rules, kept as the baseline every A/B compares against (tests/test_sim.py, bench_client.py `proto`/`hash`).

* pi_eval's engine for an attempt: sha256(task/attempt) over the registry's urls (worker.hashed_url); the round is not
  in the hash, so parallel rounds of a task (attempt 0 each) share an engine.
* The M5 ladder controller (/data/aai/m5/ladder/t2/autoscale.py, per registry every 120 s): calm on EVERY engine ->
  raise each binding run's limit x1.15 (>= +8), after 2 calm ticks; pressure on EITHER engine -> cut every run of the
  registry x0.85; `sibling_hold` is the 05:08 patch (hold when the other engine is under 40% as loaded).
"""
import hashlib

HIT_UP, HIT_DOWN, RUN_MAX, LO, HI = 0.80, 0.65, 110, 32, 640


def hashed(n: int, task: str, attempt: int) -> int:
    return int(hashlib.sha256(f'{task}/{attempt}'.encode()).hexdigest(), 16) % n


def m5_verdict(obs: list, mixed: bool, sibling_hold: bool = True) -> str:
    """'calm' | 'pressure' | 'hold' for one registry from its engines' control.Obs (hit None reads as healthy)."""
    hit = lambda o: 1.0 if o.hit is None else o.hit   # noqa: E731
    loaded = lambda o: ((not mixed and hit(o) < HIT_DOWN and o.running >= 60) or o.waiting > 30 or o.kv > 0.97)  # noqa: E731
    calm = all((mixed or hit(o) >= HIT_UP or o.running < 40) and o.waiting <= 5 and o.kv < 0.85
               and o.running < (RUN_MAX if not mixed else 180) for o in obs)
    press = any(loaded(o) for o in obs)
    if press and sibling_hold and len(obs) == 2:
        a, b = sorted(obs, key=lambda o: o.running + o.waiting)
        if not loaded(a) and a.running + a.waiting < 0.4 * (b.running + b.waiting):
            press = False
    return 'pressure' if press else 'calm' if calm else 'hold'


def m5_limit(cur: int, verdict: str, calm_ticks: int, active: int) -> int:
    """A run's new limit under the registry's verdict (calm_ticks: consecutive calm ticks so far, this one included)."""
    if verdict == 'pressure':
        return max(LO, round(cur * 0.85))
    if verdict == 'calm' and calm_ticks >= 2 and active >= 0.9 * cur:
        return min(HI, max(cur + 8, round(cur * 1.15)))
    return cur

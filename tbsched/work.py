"""The one "pending work" helper (#451 workstream 1). Every driver, controller and release check calls `pending`;
nothing else counts work left. The M5 campaign had four versions of this and two of them had the same bug (retry files
that were already claimed counted as pending: AB finishing r3 waited ~50 min on nothing; m5l-critpt held a registry
with 0 sessions), and a drain check that ignored dead workers deadlocked a round driver for ~2.5 h (AB finishing r1:
121 worker_stopped retries with no worker left).

Queue layout (eval_q35/queue_state.py, shared by pi_eval):
  queue/items.txt             "task attempt" per line
  queue/claims/<key>/         claim.json {worker, ...} + heartbeat;   key = task__a<attempt>__r<retry>
  queue/done/<key>.json       outcome of that physical attempt
  queue/retry/<key>.json      a same-seed retry waiting to be claimed

  unclaimed  items whose r0 key has no claim
  retries    retry files whose key has no claim (a claimed retry file is history, not work)
  live       open claims (no done/) whose worker is alive and whose heartbeat is fresh
  dead       the other open claims: their worker is gone or silent. They are claimable work: the reaper
             (pi_eval reap_stale) records them as infra failures and queues their same-seed retry.
  claimable = unclaimed + retries + dead;  drained = nothing claimable and nothing live
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class Pending:
    unclaimed: int = 0
    retries: int = 0
    live: int = 0
    dead: int = 0
    known: bool = True      # False: the queue could not be read (no items.txt): never "drained", never acted on
    live_keys: frozenset = frozenset()
    dead_keys: frozenset = frozenset()

    @property
    def claimable(self) -> int:
        return self.unclaimed + self.retries + self.dead

    @property
    def drained(self) -> bool:
        return self.known and self.claimable == 0 and self.live == 0


def key(task, attempt, retry=0):
    return f'{task}__a{attempt}__r{retry}'


def parse_key(k):
    task, a, r = k.rsplit('__', 2)
    return task, int(a[1:]), int(r[1:])


def _names(d: Path, suffix=''):
    try:
        return {n.removesuffix(suffix) for n in os.listdir(d) if not n.startswith('.') and n.endswith(suffix)}
    except FileNotFoundError:
        return set()


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def claim_age(claim_dir: Path, now: float) -> float | None:
    """Seconds since the claim's newest file changed (claim.json, heartbeat, replica.json)."""
    try:
        return now - max(p.stat().st_mtime for p in [claim_dir, *claim_dir.iterdir()])
    except (FileNotFoundError, ValueError):
        return None


def pending(run: Path, worker_alive=None, now: float | None = None, stale_sec: float = 1200.0) -> Pending:
    """Work left in one run's queue. worker_alive(worker name) -> True / False / None (unknown: trust the heartbeat);
    a claim is live only if its worker is not known to be gone AND its heartbeat is under stale_sec old."""
    q = Path(run) / 'queue'
    now = time.time() if now is None else now
    try:
        lines = (q / 'items.txt').read_text().splitlines()
    except OSError:
        return Pending(known=False)
    items = [key(p[0], int(p[1])) for p in (ln.split() for ln in lines) if len(p) >= 2]
    claims, done, retry = _names(q / 'claims'), _names(q / 'done', '.json'), _names(q / 'retry', '.json')
    live, dead = set(), set()
    for k in claims - done:
        d = q / 'claims' / k
        age = claim_age(d, now)
        alive = None
        if worker_alive is not None:
            worker = (_read(d / 'claim.json') or {}).get('worker')
            alive = worker_alive(worker) if worker else None
        (dead if alive is False or age is None or age > stale_sec else live).add(k)
    return Pending(unclaimed=sum(1 for k in items if k not in claims), retries=len(retry - claims), live=len(live),
                   dead=len(dead), live_keys=frozenset(live), dead_keys=frozenset(dead))


def pins(run: Path, keys=None, cache=None, settled=()) -> dict:
    """{key: engine} of the run's claims that recorded an engine (claims/<key>/replica.json {"url" | "engine"}).
    cache: {key: engine, '' (no pin, final) or None (no pin yet)} kept by the caller. A pin is written at claim time,
    so each claim is read until it has one or is `settled` (done, or no longer live): then its absence is final."""
    q = Path(run) / 'queue' / 'claims'
    cache = {} if cache is None else cache
    out = {}
    for k in (keys if keys is not None else _names(q)):
        if cache.get(k) is None:
            rec = _read(q / k / 'replica.json')
            pin = (rec.get('engine') or rec.get('url')) if rec else None
            cache[k] = pin or ('' if k in settled else None)
        if cache[k]:
            out[k] = cache[k]
    return out


def hashed(urls: list, key_: str) -> str | None:
    """pi_eval's engine for an attempt with no pin file (worker.hashed_url: sha256(task/attempt) over the urls)."""
    if not urls:
        return None
    import hashlib
    task, attempt, _ = parse_key(key_)
    return urls[int(hashlib.sha256(f'{task}/{attempt}'.encode()).hexdigest(), 16) % len(urls)]

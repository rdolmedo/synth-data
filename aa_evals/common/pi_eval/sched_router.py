"""tbsched routing for pi_eval runs (#451 workstream 1): claim only within the scheduler's start credit per engine and
pin each new attempt to the engine with the most unspent credit. Off unless the run was launched with
`run.py --sched-state FILE` (config `sched`).

FILE is the scheduler's per-run state, CAMPAIGN/sched/runs/<run>.json, written every tick by `python -m tbsched.loop`:
{"mode": "on", "updated": epoch, "credit_urls": {client url: cumulative credit}, ...}. Under a mkdir lock
(queue/sched.lock, like the #344 dispatcher's credit lock), a worker:
  1. reads FILE; unless it is in mode "on" and was updated within STALE_SEC, the run follows the old rules
     (hash pin, #402 re-pin, control.json max_in_flight / burst): a scheduler that stops never stops the run. On the
     way back no claim is made while admission.json still has the limits credit mode opened (until the next refresh);
  2. counts the run's starts per url: every claim's replica.json pin, else pi_eval's hash of (task, attempt), as
     tbsched.loop counts them;
  3. claims only if some url not held by its serving gate has unspent credit (credit - starts > 0);
  4. pins the claimed attempt to the url with the most unspent credit; ties within 10% broken by
     sha256(seed namespace / task / attempt), so parallel rounds (different namespaces) do not share an engine by
     construction; writes claims/<key>/replica.json {"url", "why": "tbsched"} for the run's other workers.
While credit governs, the worker opens eval_q35's in-flight limit and start burst (admission.json): tbsched's credit
already bounds starts per engine and paces them per engine group.

`route` and `hashed` are copies of tbsched.place.route and tbsched.work.hashed, so a frozen pi_eval snapshot (which
does not carry tbsched/) runs this; test_sched_router.py checks they agree.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

STALE_SEC = 300.0      # credit older than this: the scheduler is not running; old rules
LOCK_STALE = 120.0
OPEN = 10 ** 6         # admission.json limit / burst while credit governs


def key_hash(namespace: str, task: str, attempt: int) -> int:
    return int(hashlib.sha256(f'{namespace}/{task}/{attempt}'.encode()).hexdigest(), 16)


def route(unspent: dict, namespace: str, task: str, attempt: int, tie: float = 0.10) -> str | None:
    """= tbsched.place.route: the most unspent credit, hash tie-break."""
    c = {e: u for e, u in unspent.items() if u > 0}
    if not c:
        return None
    m = max(c.values())
    cands = sorted(e for e, u in c.items() if u >= m * (1 - tie))
    return cands[key_hash(namespace, task, attempt) % len(cands)]


def parse_key(k):
    task, a, r = k.rsplit('__', 2)
    return task, int(a[1:]), int(r[1:])


def hashed(urls: list, key_: str) -> str | None:
    """= tbsched.work.hashed = pi_eval worker.hashed_url: sha256(task/attempt) over the urls."""
    if not urls:
        return None
    task, attempt, _ = parse_key(key_)
    return urls[int(hashlib.sha256(f'{task}/{attempt}'.encode()).hexdigest(), 16) % len(urls)]


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _atomic(p: Path, obj) -> None:
    tmp = p.with_name(f'.{p.name}.{os.getpid()}.{time.monotonic_ns()}')
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


class SchedRouter:
    def __init__(self, worker, file, namespace, stale_sec=STALE_SEC, now=time.time, log=print):
        self.w, self.file, self.ns, self.stale_sec, self.now, self.log = worker, Path(file), namespace, stale_sec, now, log
        self.q = Path(worker.run) / 'queue'
        self.cache: dict[str, str | None] = {}     # claim key -> url it counts on (a pin is final once written)
        self.state = None                          # last mode reported: 'credit' | reason for the old rules

    def credit(self) -> dict | None:
        """{url: cumulative credit} while the scheduler governs this run, else None (old rules). Never raises: a
        malformed file is the old rules too (a crash here would stop claiming or admission refreshes)."""
        rec, credit, why = _read(self.file), None, None
        try:
            now = self.now()
            updated = float(rec.get('updated') or 0) if isinstance(rec, dict) else 0.0
            if not isinstance(rec, dict):
                why = 'no state file'
            elif rec.get('mode') != 'on':
                why = f"mode {rec.get('mode')!r}"
            elif not now - self.stale_sec <= updated <= now + self.stale_sec:   # old, or a clock far ahead
                why = 'stale'
            else:
                credit = {str(u): int(c) for u, c in rec['credit_urls'].items()}
        except (TypeError, ValueError, AttributeError, KeyError):
            why, credit = 'malformed state file', None
        state = 'credit' if why is None else why
        if state != self.state:
            self.log(f'tbsched router: {"credit governs claims" if why is None else "old rules (" + why + ")"}')
            self.state = state
        return None if why else credit

    def active(self) -> bool:
        return self.credit() is not None

    def starts(self, urls: list) -> dict:
        """Starts per url: a claim's replica.json pin (cached: final once written), else its hash. The hash is cached
        only once the claim is done: an old-rules worker writes its #402 pin just after the claim's mkdir."""
        try:
            keys = [k for k in os.listdir(self.q / 'claims') if not k.startswith('.')]
        except FileNotFoundError:
            keys = []
        out = {}
        for k in keys:
            u = self.cache.get(k)
            if u is None:
                rec = _read(self.q / 'claims' / k / 'replica.json')
                u = (rec or {}).get('url') if isinstance(rec, dict) else None
                if u or (self.q / 'done' / f'{k}.json').exists():
                    self.cache[k] = u = u or hashed(urls, k)
                else:
                    u = hashed(urls, k)
            if u:
                out[u] = out.get(u, 0) + 1
        return out

    def wrap(self, orig, raw=None):
        """orig: the old rules' claim (with the #402 re-pin); raw: eval_q35's State.claim, used while credit
        governs (the tbsched pin replaces the #402 one)."""
        raw = raw or orig

        def claim(worker, tasks_available):
            credit = self.credit()
            if credit is None:
                adm = _read(self.q / 'admission.json')
                if isinstance(adm, dict) and adm.get('tbsched'):
                    return None   # its limits are still open from credit mode: wait for the next refresh
                return orig(worker, tasks_available)
            urls = list(self.w.urls)
            self.starts(urls)    # warm the cache outside the lock (Lustre reads); inside, only new claims are read
            lock = self.q / 'sched.lock'
            try:
                lock.mkdir()
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > LOCK_STALE:   # holder died inside the section
                        shutil.rmtree(lock, ignore_errors=True)
                except FileNotFoundError:
                    pass
                return None
            try:
                held = getattr(self.w, 'held_urls', frozenset())
                starts = self.starts(urls)
                unspent = {u: credit.get(u, 0) - starts.get(u, 0) for u in urls if u not in held}
                if not any(v > 0 for v in unspent.values()):
                    return None
                k = raw(worker, tasks_available)
                if k is None:
                    return None
                task, attempt, _ = parse_key(k)
                url = route(unspent, self.ns, task, attempt)
                try:
                    _atomic(self.q / 'claims' / k / 'replica.json', {'url': url, 'why': 'tbsched',
                                                                     'unspent': unspent[url]})
                except OSError as e:   # the attempt is claimed: it keeps whatever pin it has
                    self.log(f'{k}: tbsched pin failed ({e!r})')
                    return k
                self.w.pinned[k] = url
                self.cache[k] = url
                return k
            finally:
                shutil.rmtree(lock, ignore_errors=True)
        return claim


def install(worker, raw_claim=None, log=print):
    """Called by PiEvalWorker.__init__ when config `sched` is set: wraps state.claim (the #402 replica claim, the old
    rules) and uses raw_claim (eval_q35's State.claim) while credit governs."""
    cfg = worker.config.get('sched') or {}
    if not cfg.get('file'):
        return None
    r = SchedRouter(worker, cfg['file'], worker.config.get('seed_namespace') or '', log=log)
    worker.state.claim = r.wrap(worker.state.claim, raw_claim)
    return r

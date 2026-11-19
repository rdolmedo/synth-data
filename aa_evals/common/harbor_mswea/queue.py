"""Lustre work queue of one run (Lustre has no file locks: mkdir is the mutex).

  RUN/queue/items.txt          "task repeat" per line, repeat-major (every task's repeat 0 first)
  RUN/queue/claims/<key>/      claim (atomic mkdir) + claim.json + heartbeat; key = <task>__r<repeat>__t<try>
  RUN/queue/done/<key>.json    outcome of that physical attempt
  RUN/queue/retry/<key>.json   the next try of an infrastructure failure, claimed before new items
  RUN/control.json             live-editable {"limit": max attempts in flight over all workers}

An item (task, repeat) resolves at its first non-infrastructure outcome. Infrastructure failures get up to
MAX_RETRIES further tries; if all fail, the item stays an infra_error and is excluded from the score.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

MAX_RETRIES = 3
KEY = re.compile(r'^(?P<task>.+)__r(?P<repeat>\d+)__t(?P<try>\d+)$')


def now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{socket.gethostname()}.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=1, default=str) if not isinstance(value, str) else value)
    os.replace(tmp, path)


def key(task: str, repeat: int, attempt: int = 0) -> str:
    return f'{task}__r{repeat}__t{attempt}'


def parse_key(k: str) -> tuple[str, int, int]:
    m = KEY.match(k)
    if not m:
        raise ValueError(k)
    return m['task'], int(m['repeat']), int(m['try'])


class Queue:
    def __init__(self, run):
        self.run = Path(run)
        self.q = self.run / 'queue'

    # ---- setup / reading
    def init(self, items: list[tuple[str, int]]) -> None:
        for sub in ('claims', 'done', 'retry'):
            (self.q / sub).mkdir(parents=True, exist_ok=True)
        items = sorted(items, key=lambda x: (x[1], x[0]))
        atomic(self.q / 'items.txt', ''.join(f'{t} {r}\n' for t, r in items))

    def items(self) -> list[tuple[str, int]]:
        out = []
        for line in (self.q / 'items.txt').read_text().splitlines():
            p = line.split()
            if len(p) == 2:
                out.append((p[0], int(p[1])))
        return out

    def _names(self, sub: str, suffix: str = '') -> set[str]:
        try:
            return {p.name.removesuffix(suffix) for p in (self.q / sub).iterdir() if not p.name.startswith('.')}
        except FileNotFoundError:
            return set()

    def done_keys(self) -> set[str]:
        return self._names('done', '.json')

    def done(self) -> dict[str, dict]:
        return {k: read(self.q / 'done' / f'{k}.json', {}) for k in self.done_keys()}

    def active(self) -> set[str]:
        return self._names('claims') - self.done_keys()

    def limit(self) -> int:
        c = read(self.run / 'control.json', {}) or {}
        return int(c.get('limit', read(self.run / 'config.json', {}).get('max_concurrent', 32)))

    def resolved(self) -> dict[tuple[str, int], dict]:
        """(task, repeat) -> the record that resolves it: its first non-infrastructure outcome, or its last
        infrastructure failure once MAX_RETRIES retries are spent. Unresolved items are absent."""
        by_item: dict[tuple[str, int], list[tuple[int, dict]]] = {}
        for k, rec in self.done().items():
            t, r, a = parse_key(k)
            by_item.setdefault((t, r), []).append((a, {**rec, 'key': k}))
        out = {}
        for item, recs in by_item.items():
            recs.sort(key=lambda x: x[0])
            ok = [rec for _, rec in recs if not rec.get('infrastructure_failure')]
            if ok:
                out[item] = ok[0]
            elif recs[-1][0] >= MAX_RETRIES:
                out[item] = recs[-1][1]
        return out

    def complete(self) -> bool:
        return len(self.resolved()) >= len(self.items())

    # ---- claiming
    def candidates(self) -> list[str]:
        claimed = self._names('claims')
        items = self.items()
        pos = {tr: i for i, tr in enumerate(items)}
        # ties (same repeat and try) in the queue's own order, not a set's per-process order (#531)
        def order(k):
            p = parse_key(k)
            return p[1], p[2], pos.get(p[:2], len(items)), k
        retries = sorted(self._names('retry', '.json') - claimed, key=order)
        fresh = [key(t, r) for t, r in items if key(t, r) not in claimed]
        return retries + fresh

    def claim(self, worker: str, fits) -> str | None:
        """Claim the first candidate whose task `fits(task)` on this worker, unless the run is at its limit."""
        if len(self.active()) >= self.limit():
            return None
        for k in self.candidates():
            if not fits(parse_key(k)[0]):
                continue
            d = self.q / 'claims' / k
            try:
                d.mkdir(parents=True)
            except FileExistsError:
                continue
            atomic(d / 'claim.json', {'worker': worker, 'host': socket.gethostname(), 'claimed_at': now()})
            self.heartbeat(k)
            return k
        return None

    def heartbeat(self, k: str) -> None:
        try:
            (self.q / 'claims' / k / 'heartbeat').write_text(now())
        except OSError:
            pass

    def claim_info(self, k: str) -> dict:
        return read(self.q / 'claims' / k / 'claim.json', {}) or {}

    def finish(self, k: str, outcome: dict) -> None:
        if (self.q / 'done' / f'{k}.json').exists():
            return
        atomic(self.q / 'done' / f'{k}.json', {**outcome, 'key': k, 'finished_at': now()})
        task, repeat, attempt = parse_key(k)
        if outcome.get('infrastructure_failure') and attempt < MAX_RETRIES:
            atomic(self.q / 'retry' / f'{key(task, repeat, attempt + 1)}.json',
                   {'after': k, 'reason': outcome.get('infrastructure_reason'), 'at': now()})

    def recover_stale(self, max_age: float) -> list[str]:
        """Claims whose heartbeat stopped (worker killed or node lost) become infrastructure failures."""
        out = []
        for k in self.active():
            d = self.q / 'claims' / k
            try:
                t = max(p.stat().st_mtime for p in [d, *d.iterdir()])
            except (FileNotFoundError, ValueError):
                continue
            if time.time() - t > max_age:
                self.finish(k, {'infrastructure_failure': True, 'infrastructure_reason': 'worker_lost',
                                'score': None, 'exception_type': 'WorkerLost', 'claim': self.claim_info(k)})
                out.append(k)
        return out

    def reset_claim(self, k: str) -> None:
        shutil.rmtree(self.q / 'claims' / k, ignore_errors=True)

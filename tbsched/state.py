"""Scheduler state that survives restarts (#451 workstream 1).

<campaign>/sched/
  runs/<run>.json        written by the scheduler only: credit per engine, groups, limits, watchdog gap
  engines/<engine>.json  written by the scheduler only: load snapshot, limit, single cap, gate state
  groups/<group>.json    written by the scheduler only: status (active | draining | released), counters
  overrides/<kind>/<name>.json   written by people only, read by the scheduler (pin a limit, a weight, no-steal, keep)
  events.jsonl           decisions, one whole line per os.write with O_APPEND (no torn lines)
  lease.<n>/             epoch-numbered mkdir lease: one scheduler per campaign

One writer per file, written whole (temp + rename): no read-merge-write, so a hand edit cannot be reverted by a
scheduler tick and no entry another process added can be erased (11-15: rounds.json lost the CritPt resample entry and
a hand edit that way). Lustre has no flock or fcntl locks; mkdir is the atomic primitive (memory: lustre-no-kernel-locks).
"""
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time


def atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def read(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class StateDir:
    KINDS = ('runs', 'engines', 'groups')

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, kind: str, name: str) -> Path:
        assert kind in self.KINDS, kind
        return self.root / kind / f'{safe(name)}.json'

    def get(self, kind: str, name: str, default=None):
        return read(self.path(kind, name), default)

    def put(self, kind: str, name: str, obj) -> None:
        atomic(self.path(kind, name), obj)

    def all(self, kind: str) -> dict:
        d = self.root / kind
        out = {}
        for p in sorted(d.glob('*.json')) if d.exists() else []:
            v = read(p)
            if v is not None:
                out[v.get('name', p.stem)] = v
        return out

    def override(self, kind: str, name: str) -> dict:
        return read(self.root / 'overrides' / kind / f'{safe(name)}.json', {}) or {}

    def event(self, **rec) -> None:
        rec = {'t': round(time.time(), 1), **rec}
        line = (json.dumps(rec, sort_keys=True) + '\n').encode()
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / 'events.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def safe(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)


class Lease:
    """One scheduler per campaign: an epoch-numbered mkdir lease (lease.<n>/owner.json). acquire() takes lease.1 when
    there is none, keeps the current one when it is ours, and takes over a stale one (owner.json not renewed for
    stale_sec) by creating lease.<n+1>: mkdir is atomic, so of two newcomers exactly one gets epoch n+1. held() is
    true only for the owner of the highest epoch, so an old holder that resumes after a takeover sees it lost the
    lease and exits instead of acting (11-15: an old controller that was not killed doubled every step for a tick).
    renew() writes only while held(). stale_sec must exceed the longest tick (condor and submit calls included)."""

    def __init__(self, root: Path, stale_sec: float = 1800.0, owner: str | None = None):
        self.root = Path(root)
        self.stale_sec = stale_sec
        self.owner = owner or f'{socket.gethostname()}:{os.getpid()}:{time.time_ns()}'
        self.epoch = None

    def _epochs(self):
        out = []
        for p in self.root.glob('lease.*'):
            try:
                out.append(int(p.name.split('.', 1)[1]))
            except ValueError:
                continue
        return sorted(out)

    def _rec(self, n):
        return read(self.root / f'lease.{n}' / 'owner.json')

    def acquire(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        self.root.mkdir(parents=True, exist_ok=True)
        eps = self._epochs()
        if eps:
            cur = eps[-1]
            rec = self._rec(cur)
            if rec and rec.get('owner') == self.owner:
                self.epoch = cur
                return True
            beat = (rec or {}).get('beat')
            if beat is None:
                try:
                    beat = (self.root / f'lease.{cur}').stat().st_mtime   # owner.json not written yet
                except FileNotFoundError:
                    beat = -math.inf
            if now - beat < self.stale_sec:
                return False
        n = (eps[-1] if eps else 0) + 1
        try:
            (self.root / f'lease.{n}').mkdir()
        except FileExistsError:
            return False      # another newcomer won this epoch
        self.epoch = n
        atomic(self.root / f'lease.{n}' / 'owner.json', {'owner': self.owner, 'beat': now, 'epoch': n,
                                                         'host': socket.gethostname(), 'pid': os.getpid()})
        for old in eps:
            shutil.rmtree(self.root / f'lease.{old}', ignore_errors=True)
        return True

    def held(self) -> bool:
        eps = self._epochs()
        return (self.epoch is not None and bool(eps) and eps[-1] == self.epoch
                and (self._rec(self.epoch) or {}).get('owner') == self.owner)

    def renew(self, now: float | None = None) -> bool:
        if not self.held():
            return False
        atomic(self.root / f'lease.{self.epoch}' / 'owner.json',
               {'owner': self.owner, 'beat': time.time() if now is None else now, 'epoch': self.epoch,
                'host': socket.gethostname(), 'pid': os.getpid()})
        return True

    def release(self) -> None:
        if self.held():
            shutil.rmtree(self.root / f'lease.{self.epoch}', ignore_errors=True)

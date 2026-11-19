"""Spec claims of a round's jobs (shards): owner, heartbeat and takeover (#171).

A job claims a spec by mkdir of work/<spec>/.claim (atomic on Lustre, which has no file locks) and writes its owner
there: {"owner": <id of this process>, "host", "pid", "shard", "time"}. Every job (incarnation) also writes a heartbeat
<round>/shards/<owner>.json from its event loop (Round.heartbeat). A job that dies (held at its memory limit, m1-hero
17841191.0) or hangs (its event loop stops: 17841191.3, 252 open specs) stops beating, and once its heartbeat is older
than `stale_after` any live job may take over its open claims and re-run them from scratch.

A takeover is race-safe the same way: the n-th takeover of a claim is mkdir of .claim/take-<n>, so of two jobs that
see the same stale owner only one wins; the current owner is the one of the highest take-<n> (or of .claim itself).
A job that wakes from a stall checks its claims (Claims.owns) and drops the ones it lost without writing their records.

Each job also journals the specs it is about to claim or take over, one per line, in <round>/shards/<owner>.claims
(written before the mkdir, so a claim is never missing from it): when a job is lost, the others read its journal
instead of every claim of the round.

Claims written before this module (owner as a text line, m1-hero and earlier) have no heartbeat: they are never taken
over automatically.
"""
import json
import os
import socket
import time
import uuid
from pathlib import Path

LEGACY = 'legacy'   # owner id of a pre-#171 claim (text owner file): never stale


def owner_id(shard):
    return f'{socket.gethostname()}-p{os.getpid()}-s{shard}-{uuid.uuid4().hex[:6]}'


class Claims:
    def __init__(self, round_dir, shard=0, stale_after=600, owner=None):
        self.dir = Path(round_dir)
        self.work = self.dir / 'work'
        self.beats = self.dir / 'shards'
        self.beats.mkdir(parents=True, exist_ok=True)
        self.shard, self.stale_after = shard, stale_after
        self.owner = owner or owner_id(shard)
        self.info = {'owner': self.owner, 'host': socket.gethostname(), 'pid': os.getpid(), 'shard': shard}
        self._beats, self._beats_at = {}, 0.0
        self._journal = None

    # ---- heartbeat
    def beat(self, **status):
        """Write this job's heartbeat (called from the event loop, so a blocked loop stops beating)."""
        value = {**self.info, 'time': time.time(), **status}
        tmp = self.beats / f'.{self.owner}.json.tmp'
        tmp.write_text(json.dumps(value))
        os.replace(tmp, self.beats / f'{self.owner}.json')

    def heartbeats(self, max_age=20):
        """owner -> heartbeat of every job of the round (re-listed at most every `max_age` s)."""
        if time.time() - self._beats_at >= max_age:
            out = {}
            try:
                names = os.listdir(self.beats)
            except OSError:
                names = []
            for n in names:
                if n.endswith('.json') and not n.startswith('.'):
                    try:
                        b = json.loads((self.beats / n).read_text())
                        out[b['owner']] = b
                    except (OSError, ValueError, KeyError):
                        continue
            self._beats, self._beats_at = out, time.time()
        return self._beats

    def alive(self, owner, now=None):
        """Has this owner beaten within stale_after? (legacy owners: always.)"""
        if owner == LEGACY:
            return True
        b = self.heartbeats().get(owner)
        if b is None:   # its heartbeat is not listed yet (a new job): ask again once, then read the file directly
            try:
                b = json.loads((self.beats / f'{owner}.json').read_text())
                self._beats[owner] = b
            except (OSError, ValueError):
                return False
        return (now or time.time()) - b.get('time', 0) < self.stale_after

    def live_jobs(self):
        """Heartbeats of the jobs that are alive now."""
        now = time.time()
        return [b for o, b in self.heartbeats().items() if now - b.get('time', 0) < self.stale_after]

    # ---- claims
    def _note(self, sid):
        """Journal a claim attempt (line-buffered: on disk before the mkdir that makes the claim)."""
        if self._journal is None:
            self._journal = open(self.beats / f'{self.owner}.claims', 'a', buffering=1)
        self._journal.write(sid + '\n')

    def journal(self, owner, offset=0):
        """-> (spec ids a job journaled from byte `offset` on, new offset); a partial last line is left for later."""
        try:
            with open(self.beats / f'{owner}.claims', 'rb') as f:
                f.seek(offset)
                data = f.read()
        except FileNotFoundError:
            return [], offset
        end = data.rfind(b'\n') + 1
        return data[:end].decode().split(), offset + end

    def _write_owner(self, d):
        tmp = d / f'.owner.{os.getpid()}.tmp'
        tmp.write_text(json.dumps({**self.info, 'time': time.time()}))
        os.replace(tmp, d / 'owner')

    def claim(self, sid):
        """Claim a spec nobody has claimed -> True; False if a claim exists (whoever holds it)."""
        wd = self.work / sid
        wd.mkdir(parents=True, exist_ok=True)
        if (wd / '.claim').exists():
            return False
        self._note(sid)
        try:
            os.mkdir(wd / '.claim')
        except FileExistsError:
            return False
        self._write_owner(wd / '.claim')
        return True

    def current(self, sid):
        """-> (generation, owner id or None, seconds since the claim's owner was set); None if unclaimed.
        owner None: the claim directory exists but its owner file was never written (died in between)."""
        c = self.work / sid / '.claim'
        try:
            names = os.listdir(c)
        except FileNotFoundError:
            return None
        gens = [int(n[5:]) for n in names if n.startswith('take-') and n[5:].isdigit()]
        gen = max(gens, default=0)
        d = c / f'take-{gen}' if gen else c
        try:
            text = (d / 'owner').read_text()
        except FileNotFoundError:
            try:
                return gen, None, time.time() - d.stat().st_mtime
            except FileNotFoundError:
                return gen, None, 0.0
        try:
            o = json.loads(text)
            return gen, o['owner'], time.time() - o.get('time', 0)
        except (ValueError, KeyError, TypeError):
            return gen, LEGACY, 0.0     # "host shard N pid P": a claim made before heartbeats existed

    def owns(self, sid):
        cur = self.current(sid)
        return cur is not None and cur[1] == self.owner

    def stale(self, sid):
        """The claim's current state if its owner is gone (no heartbeat within stale_after, or an owner file that was
        never written that long after the claim), else None."""
        cur = self.current(sid)
        if cur is None:
            return None
        gen, owner, age = cur
        if owner is None:
            return cur if age >= self.stale_after else None
        if owner == self.owner or self.alive(owner):
            return None
        return cur

    def take_over(self, sid, gen):
        """Take over generation `gen` of a stale claim -> True if this job won it (mkdir of take-<gen+1>)."""
        d = self.work / sid / '.claim' / f'take-{gen + 1}'
        self._note(sid)
        try:
            os.mkdir(d)
        except FileExistsError:
            return False
        except FileNotFoundError:   # the claim was removed by hand
            return False
        self._write_owner(d)
        return True

"""Read-merge-write of a pi_eval run's live control.json (#451 workstream 2, lesson "shared JSON control files").

RUN/control.json is read by the run's workers before every claim: `max_in_flight`, `burst`, and under a dispatcher
`max_starts`. Several processes write it: a feeder or stream driver (its share of an engine), the adaptive limiter, a
fence (stop at a boundary), and people by hand. A writer that rebuilds the whole file from its own view erases every
key it does not know: M5, 11-16, the rounds driver held rounds.json across a ~4 min launch and wrote it back,
erasing an entry another process had added (CritPt rounds 2-4 sat unscheduled until noticed), and a later tick
reverted a hand edit.

`update(run, {...})` re-reads the live file, changes only the given keys and replaces it atomically (temp + rename),
under a mkdir lock shared by every caller (Lustre has no flock/fcntl; mkdir is atomic):
  - the holder writes a token into the lock and removes the lock only if the token is still its own;
  - a lock older than LOCK_STALE_SEC belongs to a killed holder: it is broken by renaming it to a unique name (one
    breaker wins the rename) and removing that. Remaining race: a breaker that read the old lock's age just before
    another breaker replaced it can move the fresh lock aside, so two writers hold it once (one key update of
    theirs may be lost; periodic writers repeat it next tick). It needs a holder killed AND two writers in that gap;
  - after LOCK_WAIT_SEC the update goes ahead without the lock (still a merge; the race window is one read + rename,
    not a whole tick) -- a live holder stuck that long on Lustre is the remaining, rare, double-holder case;
  - a control.json that does not parse (a hand edit saved half-way) is kept as control.json.bad-<time> and reported
    on stderr before the update writes the given keys alone: its keys cannot be merged, but they are not lost.
A single-owner writer (the dispatcher's credit) may keep writing the whole file: no other writer runs on a dispatched
run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

LOCK_STALE_SEC = 30.0
LOCK_WAIT_SEC = 10.0


class Invalid(ValueError):
    pass


def _load(path: Path) -> dict:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(text)
    except ValueError as e:
        raise Invalid(str(e)) from None
    if not isinstance(value, dict):
        raise Invalid(f'not an object: {type(value).__name__}')
    return value


def read(run) -> dict:
    """The live control.json ({} when missing or not a JSON object; an I/O error such as EIO is raised)."""
    try:
        return _load(Path(run) / 'control.json')
    except Invalid:
        return {}


def _lock(lock: Path, token: str, wait: float) -> bool:
    deadline = time.monotonic() + wait
    while True:
        try:
            lock.mkdir()
            (lock / 'owner').write_text(token)
            return True
        except FileExistsError:
            pass
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_SEC:
                dead = lock.with_name(f'{lock.name}.dead.{token}')
                os.rename(lock, dead)            # only one breaker's rename succeeds
                shutil.rmtree(dead, ignore_errors=True)
                continue
        except FileNotFoundError:
            if time.monotonic() < deadline:
                continue                         # gone meanwhile, or another breaker won: try again
        except OSError:
            pass                                 # e.g. a rename refused: wait like for a live lock
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _unlock(lock: Path, token: str) -> None:
    try:
        if (lock / 'owner').read_text() == token:
            (lock / 'owner').unlink()
            lock.rmdir()
    except OSError:
        pass


def update(run, changes: dict, remove=(), wait: float = LOCK_WAIT_SEC) -> dict:
    """Set `changes` (and drop `remove`) in RUN/control.json, keeping every other key as it is on disk now.
    Returns the written value."""
    run = Path(run)
    path, lock = run / 'control.json', run / '.control.lock'
    token = f'{os.getpid()}.{time.monotonic_ns()}'
    locked = _lock(lock, token, wait)
    try:
        try:
            current = _load(path)
        except Invalid as e:
            bad = run / f'control.json.bad-{time.strftime("%Y%m%dT%H%M%S")}-{token}'
            shutil.copyfile(path, bad)
            print(f'control.update: {path} does not parse ({e}); kept as {bad.name}', file=sys.stderr, flush=True)
            current = {}
        value = {**current, **changes}
        for k in remove:
            value.pop(k, None)
        tmp = run / f'.control.json.{token}.tmp'
        try:
            tmp.write_text(json.dumps(value))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return value
    finally:
        if locked:
            _unlock(lock, token)

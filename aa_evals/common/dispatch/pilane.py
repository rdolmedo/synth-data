"""pi_eval's side of the #344 dispatcher (run.py --dispatch DIR): claim order, initial hold, lane registration.

- plan(): the run's (task, attempt) list longest predicted first (the dispatcher's duration table and model; ties:
  attempt, then the run's task order). eval_q35's claim() takes retries first, then items.txt in file order, so
  this file order is the claim order.
- hold(): RUN/control.json `max_starts` 0, so no worker claims before the dispatcher grants start credits.
- credit_claim(): wraps eval_q35 State.claim in the run's workers (worker.py, dispatched runs only): a claim is made
  only while the run's claims so far (every physical attempt, retries included) < `max_starts`, under a mkdir lock so
  workers cannot overshoot it together. The in-flight limit of eval_q35's admission is left open (max_in_flight huge);
  burst pacing, the serving-gate pause and the stale-claim reaper are unchanged.
- register(): DIR/lanes/<lane>/registration.json (kind pi), which the dispatcher discovers.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import time
from pathlib import Path

from .durations import Durations
from .lane import atomic_json, read_json

MISSING_WARN = 0.10   # warn when more than this share of a lane's tasks have no table prediction


def config(dispatch_dir: str | Path) -> dict:
    cfg = read_json(Path(dispatch_dir) / 'dispatch.json')
    if not cfg:
        raise SystemExit(f'{dispatch_dir}/dispatch.json missing: python -m aa_evals.common.dispatch.dispatcher init')
    return cfg


def plan(dispatch_dir: str | Path, lane: str, suite: str, names: list[str], attempts: int):
    """(record for config.json, [(task, attempt)] in claim order)."""
    d = Path(dispatch_dir).resolve()
    cfg = config(d)
    dur = Durations(cfg.get('durations'), cfg.get('model'))
    if cfg.get('durations_sha256') and cfg['durations_sha256'] != dur.sha256:
        raise SystemExit('duration table changed since dispatcher init')
    rank = {t: i for i, t in enumerate(names)}
    pred = {t: dur.predict(suite, t) for t in names}
    order = sorted(((t, k) for k in range(attempts) for t in names), key=lambda x: (-pred[x[0]], x[1], rank[x[0]]))
    known = sum(1 for t in names if str(t) in (dur.table['suites'].get(suite) or {}))
    rec = {'dir': str(d), 'lane': lane, 'suite': suite, 'claim_order': 'longest_predicted_first',
           'durations': dur.describe(), 'tasks_in_table': known, 'tasks': len(names)}
    missing = 1 - known / max(1, len(names))
    if missing > MISSING_WARN:
        rec['coverage_warning'] = (f'{len(names) - known} of {len(names)} tasks have no prediction for suite '
                                   f'{suite!r}: claim order falls back to the suite median / task order')
        print(f'WARNING (dispatch): {rec["coverage_warning"]}. Table suites: {sorted(dur.table["suites"])}; '
              'pass --dispatch-suite', file=sys.stderr, flush=True)
    return rec, order


OPEN_LIMIT = 10 ** 6   # eval_q35's in-flight admission limit under a dispatcher: credits bound the starts instead
LOCK_STALE = 120.0


def hold(run: Path) -> None:
    atomic_json(Path(run) / 'control.json', {'max_starts': 0, 'max_in_flight': OPEN_LIMIT,
                                             'why': 'dispatch: waiting for the first start credit',
                                             'epoch': round(time.time(), 1), 'writer': 'pi_eval run.py'})


def n_claims(q: Path) -> int:
    try:
        return sum(1 for p in (q / 'claims').iterdir() if not p.name.startswith('.'))
    except FileNotFoundError:
        return 0


def credit_claim(state, run: Path, refuse=None):
    """Wrap state.claim (eval_q35 queue_state.State) with the dispatcher's start credit. refuse(): checked under the
    lock too (a worker leaving for lack of credit, #378)."""
    orig = state.claim
    run = Path(run)
    q = run / 'queue'

    def claim(worker, tasks_available):
        lock = q / 'dispatch.lock'
        try:
            lock.mkdir()
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE:   # holder died inside the critical section
                    shutil.rmtree(lock, ignore_errors=True)
            except FileNotFoundError:
                pass
            return None
        try:
            if refuse is not None and refuse():
                return None
            ctl = read_json(run / 'control.json', {}) or {}
            if 'max_starts' not in ctl or n_claims(q) >= int(ctl['max_starts']):
                return None
            return orig(worker, tasks_available)
        finally:
            shutil.rmtree(lock, ignore_errors=True)
    state.claim = claim
    return claim


def check_owner(rec: dict, run: Path) -> None:
    """Refuse a lane name another run registered (called before the run dir is created, and again at register)."""
    old = read_json(Path(rec['dir']) / 'lanes' / rec['lane'] / 'registration.json')
    if old and Path(old.get('run', '')).resolve() != Path(run).resolve():
        raise SystemExit(f'dispatch lane {rec["lane"]} belongs to another run ({old.get("run")}); '
                         'pass --dispatch-lane NAME')


def register(rec: dict, run: Path) -> None:
    lane_dir = Path(rec['dir']) / 'lanes' / rec['lane']
    check_owner(rec, run)
    atomic_json(lane_dir / 'registration.json', {
        'name': rec['lane'], 'suite': rec['suite'], 'kind': 'pi', 'run': str(Path(run).resolve()),
        'host': socket.gethostname(), 'pid': os.getpid(), 'registered_epoch': time.time()})

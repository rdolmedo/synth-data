"""Model-server registry helpers (#445 item 3): one reader of serve_teacher's `<ROOT>/<name>/endpoints.json` and
`job.json` for every client, so "is the server up", "start dependents only once it is ready" and "release it" mean
the same thing everywhere (today ~15 clients each re-implement the 120 s freshness rule).

  (import it as tbshared.serving, or in aa_evals as aa_evals.common.serving: its ALIASES copy for partial snapshots)
  serving.path(name, root)                     # <root>/<name>/endpoints.json (root: a client's own SERVING_ROOT)
  serving.record(name, root)                   # that file as written (raises OSError / ValueError like json.loads)
  serving.age(rec)                             # seconds since its updated_at (inf without one): > STALE_SEC = dead
  serving.endpoint(name)                       # endpoints.json when fresh and healthy, else None
  serving.state(name)                          # {'state': ready|starting|held|stale|stopped|unknown, 'job_id', ...}
  serving.wait_ready(name, timeout=7200)       # poll until ready -> endpoints.json; None at the timeout or when the
                                               # server's job has left the queue (it will never become ready)
  serving.release(name, reason)                # condor_rm the job.json job, only if it is ours; -> what was done
  serving.forwarded()                          # {TASKSYNTH_SERVING_ROOT: root} when set here, else {}
  serving.pin(recorded)                        # this process onto a run's recorded root (refuses another one)
  serving.to_record()                          # what a run records at plan time ({} for production)
  serving.condor_env()                         # " TASKSYNTH_SERVING_ROOT=<root>" for a condor job's environment, or ''

serve_teacher writes endpoints.json every 30 s while its engines run, with `healthy_at` set once the start-up probes
passed, and removes it when it stops; clients treat a file older than STALE_SEC as dead. The registry is ROOT unless
$TASKSYNTH_SERVING_ROOT (or a `root` argument) names another one; a client passes its module's root (None by
default) so that one variable moves every reader, and a condor job gets it through `forwarded()` / `condor_env()`
(the eval jobs run with getenv = false). Stdlib only.
"""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/data/aai/tbench/tasksynth/serving')   # the production registry (serve_teacher's default)
ROOT_ENV = 'TASKSYNTH_SERVING_ROOT'   # another registry, e.g. for dry runs and canaries (serve_teacher honours it too)


def registry_root(root=None) -> Path:
    """`root` when given, else $TASKSYNTH_SERVING_ROOT, else ROOT: resolved on every call, so a test or a canary can
    point every reader here at its own registry without touching the production one."""
    return Path(root or os.environ.get(ROOT_ENV) or ROOT)

STALE_SEC = 120
POLL_SEC = 30
QUEUE_EVERY = 10          # wait_ready asks condor_q once per QUEUE_EVERY polls (~5 min): many waiters, one schedd


def forwarded(env=None) -> dict:
    """{TASKSYNTH_SERVING_ROOT: root} when it is set here (`env`, default os.environ), else {}: for the environment of
    a condor job whose client reads the registry (the eval jobs run with getenv = false), so the job reads the
    registry its submitter read. The root must be an absolute path without whitespace, quotes or $ (it goes into a
    condor environment line unquoted)."""
    env = os.environ if env is None else env
    v = env.get(ROOT_ENV)
    if not v:
        return {}
    if not v.startswith('/') or any(c.isspace() or c in '\'"$' for c in v):
        raise SystemExit(f'{ROOT_ENV}={v!r}: needs an absolute path without whitespace, quotes or $')
    return {ROOT_ENV: v}


def pin(recorded, what='the run') -> dict:
    """Point this process at a recorded registry root (`recorded` as forwarded() gives it: {ROOT_ENV: root}, or
    {}/None for production) for every reader here and every command and job it starts: set $TASKSYNTH_SERVING_ROOT to
    it, or clear it for production. Refused (SystemExit) when the variable already names another registry, so a
    canary run never reaches production servers and a production run never a canary's. -> the validated record."""
    env = forwarded(recorded or {})
    want, have = env.get(ROOT_ENV), os.environ.get(ROOT_ENV)
    if have and Path(have) != Path(want or ROOT):
        raise SystemExit(f'{what} was planned on registry {want or ROOT}, but {ROOT_ENV}={have}: unset it (it keeps '
                         f'its own registry)')
    if want:
        os.environ[ROOT_ENV] = want
    else:
        os.environ.pop(ROOT_ENV, None)
    return env


def to_record() -> dict:
    """What a run records at plan time: forwarded() when $TASKSYNTH_SERVING_ROOT names a registry other than
    production, else {} (record nothing: the run's files stay as before)."""
    env = forwarded()
    return env if env and Path(env[ROOT_ENV]) != ROOT else {}


def condor_env(fwd=None) -> str:
    """`fwd` (default: forwarded() of this process) as text to append to a condor `environment` line:
    " TASKSYNTH_SERVING_ROOT=<root>", or '' when unset, so the submit file is unchanged."""
    fwd = forwarded() if fwd is None else fwd
    return ''.join(f' {k}={v}' for k, v in fwd.items())


def path(name: str, root=None) -> Path:
    """The server's endpoints.json: <root>/<name>/endpoints.json (root: registry_root)."""
    return registry_root(root) / name / 'endpoints.json'


def record(name: str, root=None) -> dict:
    """The server's endpoints.json as written. Raises OSError (missing, unreadable) or ValueError (not JSON), as the
    clients' own json.loads(path.read_text()) did, so each keeps its own error handling."""
    return json.loads(path(name, root).read_text())


def age(rec: dict, now=None) -> float:
    """Seconds since the record's updated_at (serve_teacher rewrites it every 30 s); inf without one."""
    try:
        return (time.time() if now is None else now) - float(rec.get('updated_at'))
    except (TypeError, ValueError):
        return float('inf')


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _judge(e, now, stale_sec=STALE_SEC):
    """ready / starting (fresh, probes not passed yet) / stale (not updated for stale_sec) / None (no file)."""
    if not e:
        return None
    if now - float(e.get('updated_at') or 0) > stale_sec:
        return 'stale'
    return 'ready' if e.get('healthy_at') else 'starting'


def endpoint(name: str, root=None, now=None, stale_sec=STALE_SEC):
    """The server's endpoints.json when it is fresh (updated < stale_sec ago) and healthy (start-up probes passed),
    else None."""
    e = _read(registry_root(root) / name / 'endpoints.json')
    return e if _judge(e, time.time() if now is None else now, stale_sec) == 'ready' else None


def _queue(job_id: str, runner=None):
    """(JobStatus, Owner) of a job, (None, None) when it is not in the queue, ('?', '?') when condor_q fails."""
    runner = runner or subprocess.run   # resolved per call (a default bound at import would ignore test fakes)
    try:
        r = runner(['condor_q', str(job_id), '-af', 'JobStatus', 'Owner'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return '?', '?'
    if r.returncode:
        return '?', '?'
    parts = (r.stdout or '').split()
    return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)


def state(name: str, root=None, now=None, runner=None) -> dict:
    """ready (fresh, healthy endpoints.json), starting (job idle or running, no healthy endpoint yet), held (JobStatus
    5: a memory net may release it, else it needs a replacement), stale (an endpoints.json older than STALE_SEC and the
    job still queued: hung, or restarting), stopped (no job in the queue), unknown (condor_q failed)."""
    home = registry_root(root) / name
    job = _read(home / 'job.json') or {}
    e = _read(home / 'endpoints.json')                       # one read decides (it may appear at any moment)
    seen = _judge(e, time.time() if now is None else now)
    if seen == 'ready':
        return {'state': 'ready', 'job_id': e.get('job_id') or job.get('job_id'), 'urls': e.get('urls')}
    status, _ = _queue(job['job_id'], runner) if job.get('job_id') else (None, None)
    if status == '?':
        return {'state': 'unknown', 'job_id': job.get('job_id')}
    if status is None:
        return {'state': 'stopped', 'job_id': job.get('job_id')}
    if status == '5':
        return {'state': 'held', 'job_id': job.get('job_id'), 'job_status': status}
    return {'state': 'stale' if seen == 'stale' else 'starting', 'job_id': job.get('job_id'), 'job_status': status}


def wait_ready(name: str, timeout: float = 7200, root=None, poll=POLL_SEC, runner=None, sleep=time.sleep,
               clock=time.time, queue_every=QUEUE_EVERY):
    """Block until the server is ready -> its endpoints.json; None after `timeout` s, or when its job is no longer in
    the queue (it will never become ready; checked with condor_q at the start and every `queue_every` polls). For job
    scripts and launchers that must not start dependents (workers, evals) against a server that is still loading.
    The default timeout covers serve_teacher's start-up limit (5,400 s) plus time idle in the queue."""
    end = clock() + timeout
    n = 0
    while True:
        e = endpoint(name, root)
        if e is not None:
            return e
        if n % max(1, queue_every) == 0 and state(name, root, runner=runner)['state'] == 'stopped':
            return None
        if clock() >= end:
            return None
        n += 1
        sleep(min(poll, max(0.0, end - clock())))


def release(name: str, reason: str, root=None, runner=None, user=None) -> dict:
    """Remove the server's job (the job.json job id, exactly) if it is still queued and ours; record the release in
    <home>/released.jsonl. Never removes another user's job or anything not named in job.json."""
    home = registry_root(root) / name
    job = _read(home / 'job.json') or {}
    jid = job.get('job_id')
    rec = {'time': time.time(), 'name': name, 'job_id': jid, 'reason': reason}
    if not jid:
        return {**rec, 'action': 'none', 'why': 'no job.json'}
    runner = runner or subprocess.run
    status, owner = _queue(jid, runner)
    me = user or os.environ.get('USER') or getpass.getuser()
    if status == '?':
        out = {**rec, 'action': 'none', 'why': 'condor_q failed'}
    elif status is None:
        out = {**rec, 'action': 'none', 'why': 'not in the queue'}
    elif owner != me:
        out = {**rec, 'action': 'none', 'why': f'owner {owner} is not {me}'}
    else:
        try:
            r = runner(['condor_rm', str(jid)], capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        out = {**rec, 'action': 'removed' if ok else 'none', 'why': '' if ok else 'condor_rm failed'}
    try:
        with open(home / 'released.jsonl', 'a') as f:
            f.write(json.dumps(out) + '\n')
    except OSError:
        pass
    return out


if __name__ == '__main__':   # for shell submitters (aa_evals/omniscience/submit.sh)
    import sys
    if sys.argv[1:] != ['condor-env']:
        raise SystemExit('usage: python -m tbshared.serving condor-env   # (or aa_evals.common.serving) condor_env(), '
                         'or nothing when unset')
    print(condor_env(), end='')

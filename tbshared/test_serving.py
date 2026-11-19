"""tbshared.serving: the server registry readers (fake registry and fake condor; no jobs touched)."""
import json
import os
import time
import types

import pytest

from tbshared import serving


def registry(tmp_path, endpoints=None, job_id='7.0'):
    home = tmp_path / 'srv'
    home.mkdir(parents=True, exist_ok=True)
    if job_id:
        (home / 'job.json').write_text(json.dumps({'job_id': job_id}))
    if endpoints is not None:
        (home / 'endpoints.json').write_text(json.dumps(endpoints))
    return home


def condor(queue=None, rm_rc=0):
    """A fake subprocess.run: condor_q prints `queue` ('2 user', '' = not queued, None = condor_q fails)."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == 'condor_q':
            return types.SimpleNamespace(returncode=1 if queue is None else 0, stdout=queue or '')
        return types.SimpleNamespace(returncode=rm_rc, stdout='')
    run.calls = calls
    return run


def test_endpoint_needs_fresh_and_healthy(tmp_path):
    now = time.time()
    registry(tmp_path, {'healthy_at': now - 50, 'updated_at': now - 10, 'urls': ['http://h:1/v1']})
    assert serving.endpoint('srv', tmp_path)['urls'] == ['http://h:1/v1']
    registry(tmp_path, {'healthy_at': now - 50, 'updated_at': now - serving.STALE_SEC - 1})
    assert serving.endpoint('srv', tmp_path) is None                              # stale
    registry(tmp_path, {'healthy_at': None, 'updated_at': now})
    assert serving.endpoint('srv', tmp_path) is None                              # still starting
    assert serving.endpoint('nope', tmp_path) is None


def test_state(tmp_path):
    now = time.time()
    registry(tmp_path, {'healthy_at': now, 'updated_at': now, 'urls': ['u']})
    assert serving.state('srv', tmp_path, runner=condor('2 user'))['state'] == 'ready'
    registry(tmp_path, {'healthy_at': None, 'updated_at': now})                     # fresh, probes not passed yet
    assert serving.state('srv', tmp_path, runner=condor('2 user'))['state'] == 'starting'
    registry(tmp_path, {'healthy_at': now - 900, 'updated_at': now - 600})          # healthy once, silent since
    assert serving.state('srv', tmp_path, runner=condor('2 user'))['state'] == 'stale'
    assert serving.state('srv', tmp_path, runner=condor('5 user'))['state'] == 'held'
    (tmp_path / 'srv' / 'endpoints.json').unlink()
    assert serving.state('srv', tmp_path, runner=condor('1 user'))['state'] == 'starting'
    assert serving.state('srv', tmp_path, runner=condor(''))['state'] == 'stopped'
    assert serving.state('srv', tmp_path, runner=condor(None))['state'] == 'unknown'


def test_wait_ready_returns_when_ready_and_gives_up_when_the_job_is_gone(tmp_path):
    registry(tmp_path)
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s
        if t[0] >= 30:     # the server comes up after 30 s
            now = time.time()
            (tmp_path / 'srv' / 'endpoints.json').write_text(json.dumps({'healthy_at': now, 'updated_at': now}))
    run = condor('2 user')
    e = serving.wait_ready('srv', 600, tmp_path, poll=15, runner=run, sleep=sleep, clock=lambda: t[0], queue_every=10)
    assert e is not None and sleeps == [15, 15] and len(run.calls) == 1        # condor_q once, not every poll
    (tmp_path / 'srv' / 'endpoints.json').unlink()
    assert serving.wait_ready('srv', 600, tmp_path, runner=condor(''), sleep=sleep, clock=lambda: t[0]) is None
    assert serving.wait_ready('srv', 0, tmp_path, runner=condor('1 user'), sleep=sleep, clock=lambda: t[0]) is None


def test_release_only_our_job_from_job_json(tmp_path):
    registry(tmp_path)
    run = condor('2 user')
    out = serving.release('srv', 'done', tmp_path, runner=run, user='user')
    assert out['action'] == 'removed' and run.calls[-1] == ['condor_rm', '7.0']
    other = condor('2 someone')
    assert serving.release('srv', 'x', tmp_path, runner=other, user='user')['action'] == 'none'
    assert not any(c[0] == 'condor_rm' for c in other.calls)
    gone = condor('')
    assert serving.release('srv', 'x', tmp_path, runner=gone, user='user')['why'] == 'not in the queue'
    broken = condor(None)                                     # condor_q fails: never guess, never remove
    assert serving.release('srv', 'x', tmp_path, runner=broken, user='user')['why'] == 'condor_q failed'
    assert not any(c[0] == 'condor_rm' for c in broken.calls)
    rows = [json.loads(x) for x in (tmp_path / 'srv' / 'released.jsonl').read_text().splitlines()]
    assert [r['action'] for r in rows] == ['removed', 'none', 'none', 'none']
    registry(tmp_path / 'b', job_id=None)
    assert serving.release('srv', 'x', tmp_path / 'b', runner=condor('2 user'))['why'] == 'no job.json'


def test_state_survives_endpoints_appearing_between_reads(tmp_path, monkeypatch):
    """Review of PR C: serve_teacher creates endpoints.json once healthy. If it appears after the first read, state()
    must not crash (it used to read the file twice and call .get on the first, None, result)."""
    registry(tmp_path)
    now = time.time()
    (tmp_path / 'srv' / 'endpoints.json').write_text(json.dumps({'healthy_at': now, 'updated_at': now, 'urls': ['u']}))
    real, calls = serving._read, []

    def first_read_misses(path):
        if path.name == 'endpoints.json':
            calls.append(1)
            if len(calls) == 1:
                return None
        return real(path)
    monkeypatch.setattr(serving, '_read', first_read_misses)
    assert serving.state('srv', tmp_path, runner=condor('2 user'))['state'] in ('starting', 'stale')
    assert serving.state('srv', tmp_path, runner=condor('2 user'))['state'] == 'ready'


def test_registry_root_env_override(tmp_path, monkeypatch):
    """$TASKSYNTH_SERVING_ROOT points every reader at another registry (dry runs, canaries); an explicit root wins."""
    now = time.time()
    registry(tmp_path, {'healthy_at': now, 'updated_at': now, 'urls': ['canary']})
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    assert serving.registry_root() == serving.ROOT
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path))
    assert serving.registry_root() == tmp_path and serving.registry_root('/elsewhere') == serving.Path('/elsewhere')
    assert serving.endpoint('srv')['urls'] == ['canary']
    assert serving.state('srv', runner=condor('2 user'))['state'] == 'ready'


def test_path_record_age(tmp_path):
    registry(tmp_path, {'updated_at': 100.0, 'urls': ['u']})
    assert serving.path('srv', tmp_path) == tmp_path / 'srv' / 'endpoints.json'
    assert serving.record('srv', tmp_path)['urls'] == ['u']
    assert serving.age(serving.record('srv', tmp_path), now=130.0) == 30.0
    assert serving.age({}, now=1.0) == float('inf') and serving.age({'updated_at': None}) == float('inf')
    import pytest
    with pytest.raises(OSError):
        serving.record('missing', tmp_path)
    (tmp_path / 'srv' / 'endpoints.json').write_text('{not json')
    with pytest.raises(ValueError):
        serving.record('srv', tmp_path)


def test_pin_and_to_record(tmp_path, monkeypatch):
    """pin: a process onto a recorded registry root ({} = production), refusing another one; to_record: what a run
    records at plan time (only a registry other than production)."""
    from tbshared import serving
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    canary = str(tmp_path / 'canary')
    assert serving.pin({serving.ROOT_ENV: canary}) == {serving.ROOT_ENV: canary}
    assert os.environ[serving.ROOT_ENV] == canary
    assert serving.pin({serving.ROOT_ENV: canary}) == {serving.ROOT_ENV: canary}       # same root again
    with pytest.raises(SystemExit, match='planned on registry'):
        serving.pin({})                                                                # production, canary shell
    monkeypatch.setenv(serving.ROOT_ENV, str(serving.ROOT))
    assert serving.pin(None) == {} and serving.ROOT_ENV not in os.environ              # production path: cleared
    with pytest.raises(SystemExit):
        serving.pin({serving.ROOT_ENV: 'relative/path'})                               # validated as forwarded()
    assert serving.to_record() == {}
    monkeypatch.setenv(serving.ROOT_ENV, str(serving.ROOT))
    assert serving.to_record() == {}
    monkeypatch.setenv(serving.ROOT_ENV, canary)
    assert serving.to_record() == {serving.ROOT_ENV: canary}

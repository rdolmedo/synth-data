"""PiEvalWorker.pack (#378): the archive is built on node-local scratch and copied to the run dir with O_DIRECT."""
import tarfile

import pytest

pytest.importorskip('harbor')
from aa_evals.common.pi_eval import worker as W   # noqa: E402


def test_pack_builds_locally_copies_direct_and_cleans_up(tmp_path, monkeypatch):
    scratch, run = tmp_path / 'scratch' / 'raw', tmp_path / 'fast' / 'attempts' / 'k1'
    raw = scratch / 'k1'
    (raw / 'trial').mkdir(parents=True)
    (raw / 'trial' / 'result.json').write_text('{"reward": 1}')
    run.mkdir(parents=True)
    seen = []
    from aa_evals.common import fastio
    real = fastio.copy_direct
    monkeypatch.setattr(fastio, 'copy_direct', lambda s, d: seen.append((s, d)) or real(s, d))
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert out['trial_archive'] == str(run / 'trial.tar.gz') and 'pack_error' not in out
    src, dst = seen[0]
    assert src.parent == scratch and dst.parent == run                  # built locally, copied into the run dir
    assert not src.exists() and not raw.exists() and not list(run.glob('.trial.tar.gz.*'))
    with tarfile.open(run / 'trial.tar.gz') as tf:
        assert 'k1/trial/result.json' in tf.getnames()


def _raw(tmp_path):
    raw = tmp_path / 'scratch' / 'raw' / 'k1'
    (raw / 'trial').mkdir(parents=True)
    (raw / 'trial' / 'result.json').write_text('{"reward": 1}')
    run = tmp_path / 'fast' / 'attempts' / 'k1'
    run.mkdir(parents=True)
    return raw, run


def _full_scratch(monkeypatch, times):
    """tarfile.open of the local (scratch) archive fails with ENOSPC `times` times."""
    import errno
    real, n = W.tarfile.open, {'left': times}

    def open_(name, *a, **kw):
        if str(name).endswith('.k1.trial.tar.gz') and n['left'] > 0:
            n['left'] -= 1
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real(name, *a, **kw)
    monkeypatch.setattr(W.tarfile, 'open', open_)


def _evict(monkeypatch, tmp_path, calls):
    from harbor_ns import builder
    monkeypatch.setattr(builder, 'evict', lambda root, force=False: calls.append((root, force)) or {'removed': 3})
    monkeypatch.setattr(W.q35, 'SCRATCH', tmp_path / 'scratch')


def test_pack_full_scratch_evicts_and_packs_again(tmp_path, monkeypatch):
    """#451 lesson 7: ENOSPC building the local archive frees harbor_ns images (force) and packs once more."""
    raw, run = _raw(tmp_path)
    calls = []
    _full_scratch(monkeypatch, 1)
    _evict(monkeypatch, tmp_path, calls)
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert calls == [(tmp_path / 'scratch' / 'harbor_ns', True)] and out['pack_evicted'] == {'removed': 3}
    assert out['trial_archive'] == str(run / 'trial.tar.gz') and 'pack_error' not in out and 'pack_direct' not in out
    assert not raw.exists() and (run / 'trial.tar.gz').exists()


def test_pack_still_full_writes_the_archive_straight_to_the_run_dir(tmp_path, monkeypatch):
    raw, run = _raw(tmp_path)
    calls = []
    _full_scratch(monkeypatch, 2)
    _evict(monkeypatch, tmp_path, calls)
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert out['pack_direct'] is True and 'pack_error' not in out and not raw.exists()
    with tarfile.open(run / 'trial.tar.gz') as tf:
        assert 'k1/trial/result.json' in tf.getnames()
    assert not list(run.glob('.trial.tar.gz.*')) and not list(raw.parent.glob('.k1.trial.tar.gz'))


def test_pack_fast_full_does_not_evict_and_keeps_the_trial(tmp_path, monkeypatch):
    """ENOSPC copying to /data (OST or quota) is not this worker's scratch: no eviction, the trial stays as before."""
    import errno
    from aa_evals.common import fastio
    raw, run = _raw(tmp_path)
    calls = []
    _evict(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(fastio, 'copy_direct', lambda s, d: (_ for _ in ()).throw(OSError(errno.ENOSPC, 'full')))
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert calls == [] and 'pack_error' in out and raw.exists() and not (run / 'trial.tar.gz').exists()


def test_pack_evict_failure_never_escapes(tmp_path, monkeypatch):
    from harbor_ns import builder
    raw, run = _raw(tmp_path)
    _full_scratch(monkeypatch, 1)
    monkeypatch.setattr(builder, 'evict', lambda root, force=False: (_ for _ in ()).throw(OSError(28, 'x')))
    monkeypatch.setattr(W.q35, 'SCRATCH', tmp_path / 'scratch')
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert 'error' in out['pack_evicted'] and out['trial_archive'] == str(run / 'trial.tar.gz')


def test_pack_other_error_keeps_the_local_trial(tmp_path, monkeypatch):
    from aa_evals.common import fastio
    raw, run = _raw(tmp_path)
    monkeypatch.setattr(fastio, 'copy_direct', lambda s, d: (_ for _ in ()).throw(PermissionError(13, 'denied')))
    out = W.PiEvalWorker.pack(None, 'k1', raw, run, {'trial_dir': str(raw / 'trial')})
    assert 'pack_error' in out and raw.exists()

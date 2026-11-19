"""synth/venv_stage.py: a venv's tar, its record, and the node-local copy serve_teacher's stage_venv runs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from synth import venv_stage as vs


def fake_venv(root: Path, name='vllm-x') -> Path:
    v = root / name
    sp = v / 'lib' / 'python3.12' / 'site-packages'
    for dist in ('vllm-0.34.0.dist-info', 'torch-2.9.0.dist-info', 'flashinfer_cubin-0.7.4.dist-info'):
        (sp / dist).mkdir(parents=True)
        (sp / dist / 'RECORD').write_text(dist)
    (sp / 'vllm').mkdir()
    (sp / 'vllm' / '__init__.py').write_text('__version__ = "0.34.0"\n')
    (sp / 'with space.txt').write_text('x')   # names with spaces survive the round trip
    (v / 'bin').mkdir()
    (v / 'bin' / 'vllm').write_text('#!/bin/sh\n')
    os.symlink('/usr/bin/python3', v / 'bin' / 'python')   # absolute interpreter link, kept as a link
    os.symlink('lib', v / 'lib64')
    (v / 'pyvenv.cfg').write_text('relocatable = true\n')
    return v


def test_build_check_stage_roundtrip(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    meta = vs.build(v)
    assert vs.tar_path(v).is_file() and json.loads(vs.meta_path(v).read_text()) == meta
    assert meta['top'] == 'vllm-x' and meta['dists'] == 3 and meta['files'] == vs.count_files(v) == 9
    assert vs.check(v) == (meta, None)
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path == vs.copy_path(v, tmp_path / 'node') and path.name == 'vllm-x' and rec['staged'] and not rec.get('reused')
    assert path.parent.parent == tmp_path / 'node' / 'venvs'      # <root>/venvs/<hash of the source path>/<name>
    assert vs.dists(path) == vs.dists(v) and vs.count_files(path) == meta['files']
    assert os.readlink(path / 'bin' / 'python') == '/usr/bin/python3' and os.readlink(path / 'lib64') == 'lib'
    assert (path / 'lib' / 'python3.12' / 'site-packages' / 'with space.txt').read_text() == 'x'
    again, rec2 = vs.stage(v, tmp_path / 'node')   # a second engine of the job: the copy is reused
    assert again == path and rec2['reused']


def test_changed_venv_is_not_staged(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    vs.build(v)
    (v / 'lib' / 'python3.12' / 'site-packages' / 'nvidia_extra-1.0.dist-info').mkdir()   # a wheel added afterwards
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and not rec['staged'] and 'rebuild' in rec['reason']
    assert not vs.copy_path(v, tmp_path / 'node').exists()


def test_reinstalled_wheel_is_not_staged(tmp_path):
    """Same name and version, other files (a rebuilt wheel): its RECORD differs, so does the fingerprint."""
    v = fake_venv(tmp_path / 'fast')
    vs.build(v)
    (v / 'lib' / 'python3.12' / 'site-packages' / 'vllm-0.34.0.dist-info' / 'RECORD').write_text('other files')
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'rebuild' in rec['reason']


def test_same_name_venvs_get_their_own_copies(tmp_path):
    """Two venvs of one job with the same directory name (A/B arms) never share a copy."""
    a, b = fake_venv(tmp_path / 'a'), fake_venv(tmp_path / 'b')
    (b / 'lib' / 'python3.12' / 'site-packages' / 'patched-1.0.dist-info').mkdir()
    vs.build(a)
    vs.build(b)
    pa, _ = vs.stage(a, tmp_path / 'node')
    pb, _ = vs.stage(b, tmp_path / 'node')
    assert pa != pb and pa.is_dir() and pb.is_dir() and vs.dists(pa) == vs.dists(a) and vs.dists(pb) == vs.dists(b)


def test_symlinked_or_non_relocatable_venvs_refused(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    link = tmp_path / 'link'
    os.symlink(v, link)
    assert 'symlink' in vs.check(link)[1]
    with pytest.raises(SystemExit):
        vs.build(link)
    (v / 'pyvenv.cfg').write_text('home = /usr/bin\n')
    assert 'relocatable' in vs.check(v)[1]
    with pytest.raises(SystemExit):
        vs.build(v)


def test_failed_or_slow_extraction_leaves_no_copy(tmp_path, monkeypatch):
    v = fake_venv(tmp_path / 'fast')
    vs.build(v)
    dest = vs.copy_path(v, tmp_path / 'node')

    def broken(cmd, timeout):
        dest.mkdir(parents=True)                       # a partial copy, then the pipe fails
        return subprocess.CompletedProcess(cmd, 2, '', 'tar: Unexpected EOF')
    monkeypatch.setattr(vs, '_bash', broken)
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'extract failed' in rec['reason'] and not dest.exists()

    def slow(cmd, timeout):
        dest.mkdir(parents=True)
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(vs, '_bash', slow)
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'TimeoutExpired' in rec['reason'] and not dest.exists()


def test_copy_differing_from_its_record_is_dropped(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    meta = vs.build(v)
    vs.meta_path(v).write_text(json.dumps({**meta, 'files': meta['files'] + 1}))
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'differs from its record' in rec['reason'] and not vs.copy_path(v, tmp_path / 'node').exists()


def test_missing_or_truncated_tar(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'no vllm-x.tar' in rec['reason']
    vs.build(v)
    with open(vs.tar_path(v), 'r+b') as f:
        f.truncate(1024)
    path, rec = vs.stage(v, tmp_path / 'node')
    assert path is None and 'bytes' in rec['reason']


def test_tar_gib(tmp_path):
    v = fake_venv(tmp_path / 'fast')
    assert vs.tar_gib(v) == 0.0
    vs.build(v)
    assert vs.tar_gib(v) == vs.tar_path(v).stat().st_size / 2 ** 30 > 0

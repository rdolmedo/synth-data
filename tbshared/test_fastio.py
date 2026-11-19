"""tbshared.fastio.copy_direct: O_DIRECT copy, buffered fallback, size check."""
import json
import types
from pathlib import Path

import pytest

from tbshared import fastio


def test_copy_direct_uses_dd_oflag_direct(tmp_path):
    src = tmp_path / 'a.bin'
    src.write_bytes(b'x' * 1234567)
    dst = tmp_path / 'b.bin'
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        dst.write_bytes(src.read_bytes())
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    assert fastio.copy_direct(src, dst, runner=runner) == 'direct'
    assert 'oflag=direct' in calls[0] and dst.read_bytes() == src.read_bytes()


def test_copy_direct_falls_back_when_dd_fails(tmp_path):
    src = tmp_path / 'a.bin'
    src.write_bytes(b'y' * 4097)
    dst = tmp_path / 'b.bin'
    fail = lambda argv, **kw: types.SimpleNamespace(returncode=1, stdout='', stderr='dd: Invalid argument')
    assert fastio.copy_direct(src, dst, runner=fail) == 'buffered'
    assert dst.read_bytes() == src.read_bytes()


def test_copy_direct_short_dd_output_is_redone(tmp_path):
    src = tmp_path / 'a.bin'
    src.write_bytes(b'z' * 10000)
    dst = tmp_path / 'b.bin'

    def short(argv, **kw):
        dst.write_bytes(b'z' * 10)                 # truncated copy: the size check catches it
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    assert fastio.copy_direct(src, dst, runner=short) == 'buffered'
    assert dst.stat().st_size == 10000


def test_copy_tree_direct_builds_partial_then_renames(tmp_path):
    src = tmp_path / 'src'
    (src / 'a' / 'b').mkdir(parents=True)
    (src / 'x.txt').write_text('x' * 5000)
    (src / 'a' / 'b' / 'y.bin').write_bytes(b'\0' * 70000)
    dst = tmp_path / 'out' / 'render'
    dst.parent.mkdir()
    (dst.parent / 'render.partial').mkdir()                 # a killed copy's leftover is replaced
    (dst.parent / 'render.partial' / 'stale').write_text('old')

    def runner(argv, **kw):                                 # dd: copy if= to of=
        a = dict(x.split('=', 1) for x in argv[1:] if '=' in x)
        Path(a['of']).write_bytes(Path(a['if']).read_bytes())
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    assert fastio.copy_tree_direct(src, dst, runner=runner) == 2
    assert (dst / 'x.txt').read_text() == 'x' * 5000 and (dst / 'a' / 'b' / 'y.bin').stat().st_size == 70000
    assert not (dst / 'stale').exists() and not (dst.parent / 'render.partial').exists()
    with pytest.raises(FileExistsError):
        fastio.copy_tree_direct(src, dst, runner=runner)
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert fastio.copy_tree_direct(empty, tmp_path / 'e2') == 0 and (tmp_path / 'e2').is_dir()


def test_append_line_and_trim_torn_tail(tmp_path):
    f = tmp_path / 'items.jsonl'
    assert fastio.trim_torn_tail(f) == 0                    # missing file
    fastio.append_line(f, '{"a": 1}')
    fastio.append_line(f, '{"b": " "}\n')
    assert f.read_text().split('\n') == ['{"a": 1}', '{"b": " "}', '']
    assert fastio.trim_torn_tail(f) == 0
    with open(f, 'a') as h:
        h.write('{"c": 3, "trunc')                          # a kill mid-append
    assert fastio.trim_torn_tail(f) == len('{"c": 3, "trunc')
    fastio.append_line(f, '{"d": 4}')
    assert [json.loads(x) for x in f.read_text().split('\n') if x] == [{'a': 1}, {'b': ' '}, {'d': 4}]
    g = tmp_path / 'one.jsonl'
    g.write_text('{"torn"')                                 # no newline at all
    assert fastio.trim_torn_tail(g) == 7 and g.read_text() == ''

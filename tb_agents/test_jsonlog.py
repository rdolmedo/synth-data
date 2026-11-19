"""Whole-line JSONL appends and torn-line recovery (#451 workstream 2): each case reproduces an M5 11-15 failure."""
import json
import multiprocessing
import os

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tb_agents import jsonlog   # noqa: E402
from tb_agents.jsonlog import append_line, parse, read, recover   # noqa: E402


def big_record(i, size=200_000):
    # a response-budget wire record: the whole history as Pi saw it (MBs in long sessions)
    return {'time': 1.5 + i, 'policy': 'p', 'turn': i, 'messages': [{'role': 'user', 'content': 'x' * size}],
            'response': {'choices': [{'message': {'role': 'assistant', 'content': f'answer {i}'}}]}}


def test_bytes_are_unchanged(tmp_path):
    # writers keep their own json.dumps options: the file holds exactly the old text-mode bytes
    path = tmp_path / 'w.jsonl'
    rec = {'time': 1.0, 'text': 'é ✓', 'n': [1, 2]}
    append_line(path, json.dumps(rec, ensure_ascii=False))
    append_line(path, json.dumps(rec) + '\n')   # a caller that already added the newline gets no second one
    assert path.read_bytes() == (json.dumps(rec, ensure_ascii=False) + '\n' + json.dumps(rec) + '\n').encode()


def test_one_write_per_record(tmp_path, monkeypatch):
    calls = []
    real = os.write
    monkeypatch.setattr(jsonlog.os, 'write', lambda fd, b: calls.append(len(b)) or real(fd, b))
    line = json.dumps(big_record(0), ensure_ascii=False)
    append_line(tmp_path / 'w.jsonl', line)
    assert calls == [len(line.encode()) + 1]


def test_short_write_continues(tmp_path, monkeypatch):
    real = os.write
    monkeypatch.setattr(jsonlog.os, 'write', lambda fd, b: real(fd, bytes(b[:1000])))   # 1000 bytes per call
    append_line(tmp_path / 'w.jsonl', json.dumps(big_record(0, 5000)))
    append_line(tmp_path / 'w.jsonl', json.dumps(big_record(1, 5000)))
    got = read(tmp_path / 'w.jsonl')
    assert got.clean and [r['turn'] for r in got.records] == [0, 1]


def test_a_failed_partial_write_is_ended_so_the_next_record_stands_alone(tmp_path, monkeypatch):
    # the M5 failure shape: a record cut short (e.g. /data over quota), then later appends glued onto it
    real = os.write
    path = tmp_path / 'w.jsonl'
    append_line(path, json.dumps(big_record(0, 100)))

    calls = []

    def fails_after_part(fd, b):   # write(2) returns a short count, then the next call fails
        calls.append(len(b))
        if len(b) == 1:
            return real(fd, b)
        if len(calls) == 1:
            return real(fd, bytes(b[:300]))
        raise OSError(122, 'Disk quota exceeded')
    monkeypatch.setattr(jsonlog.os, 'write', fails_after_part)
    with pytest.raises(OSError):
        append_line(path, json.dumps(big_record(1, 5000)))
    monkeypatch.setattr(jsonlog.os, 'write', real)
    append_line(path, json.dumps(big_record(2, 100)))
    got = read(path)
    assert [r['turn'] for r in got.records] == [0, 2] and got.cut == 1 and not got.bad and not got.recovered


def test_new_files_get_the_umask_default_mode(tmp_path):
    old = os.umask(0o002)
    try:
        append_line(tmp_path / 'w.jsonl', '{}')
    finally:
        os.umask(old)
    assert (tmp_path / 'w.jsonl').stat().st_mode & 0o777 == 0o664   # as open(path, 'a') did


def worker(path, k, n):
    for i in range(n):
        append_line(path, json.dumps(big_record(k * 1000 + i, size=150_000), ensure_ascii=False))


def test_concurrent_appenders_never_interleave(tmp_path):
    path = str(tmp_path / 'shared.jsonl')
    procs = [multiprocessing.Process(target=worker, args=(path, k, 6)) for k in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    got = read(path)
    assert got.clean
    assert sorted(r['turn'] for r in got.records) == sorted(k * 1000 + i for k in range(6) for i in range(6))


def test_cut_last_line_is_skipped_and_counted(tmp_path):
    path = tmp_path / 'w.jsonl'
    for i in range(3):
        append_line(path, json.dumps(big_record(i, 100)))
    with open(path, 'ab') as f:   # a session killed mid-write
        f.write(json.dumps(big_record(3, 100)).encode()[:70])
    got = read(path)
    assert [r['turn'] for r in got.records] == [0, 1, 2]
    assert got.cut_tail and not got.recovered and not got.bad


def test_fragment_then_whole_record_keeps_the_record(tmp_path):
    # 11-15: 8 TB-Sci-lite wire logs had a line = cut fragment + a whole record
    path = tmp_path / 'w.jsonl'
    append_line(path, json.dumps(big_record(0, 1000)))
    with open(path, 'ab') as f:
        f.write(json.dumps(big_record(1, 5000)).encode()[:3000])   # cut fragment, no newline
    append_line(path, json.dumps(big_record(1, 5000)))             # the restarted writer's whole record
    append_line(path, json.dumps(big_record(2, 1000)))
    got = read(path)
    assert [r['turn'] for r in got.records] == [0, 1, 2]
    assert got.recovered == 1 and not got.cut_tail and not got.bad


def test_nested_objects_with_the_same_first_key(tmp_path):
    # candidates inside the whole record (a nested {"time": ...}) fail to parse and are skipped
    rec = {'time': 2, 'inner': {'time': 3, 'deeper': {'time': 4}}, 'x': 'y'}
    line = json.dumps({'time': 1, 'cut': 'abc'}).encode()[:12] + json.dumps(rec).encode()
    assert recover(line) == rec


def test_unrecoverable_middle_line_is_reported_not_raised():
    data = b'{"t":1}\n' + b'garbage tail of an interleaved write\n' + b'{"t":2}\n'
    got = parse(data)
    assert [r['t'] for r in got.records] == [1, 2] and got.bad == [2] and not got.cut_tail and not got.cut


def test_last_line_that_parses_without_newline_is_kept():
    got = parse(b'{"t":1}\n{"t":2}')
    assert [r['t'] for r in got.records] == [1, 2] and got.clean


def test_torn_line_at_end_with_newline_is_recovered_not_dropped():
    # the M5 reader treated any bad LAST line as cut; a torn line that ends in a newline holds a whole record
    data = b'{"t":0}\n' + b'{"t":1,"m":"cu' + b'{"t":1,"m":"cut short"}\n'
    got = parse(data)
    assert [r['m'] if 'm' in r else None for r in got.records] == [None, 'cut short'] and got.recovered == 1


@pytest.mark.parametrize('cut', [1, 4, 7, 8, 30])
def test_short_fragments_are_recovered_too(cut):
    # a write cut within the first key ('{', '{"ti', '{"time"') left no signature before the record
    rec = json.dumps(big_record(1, 200)).encode()
    got = parse(json.dumps(big_record(0, 50)).encode() + b'\n' + rec[:cut] + rec + b'\n')
    assert [r['turn'] for r in got.records] == [0, 1] and got.recovered == 1 and not got.bad


def test_last_line_with_fragment_and_record_but_no_newline_is_recovered():
    rec = json.dumps({'t': 'end', 'id': 1})
    got = parse(b'{"t":"req","id":1}\n' + rec[:9].encode() + rec.encode())
    assert [r['t'] for r in got.records] == ['req', 'end'] and got.recovered == 1 and not got.cut_tail


def test_a_fragment_ending_in_a_nested_object_with_the_same_first_key_is_not_a_record():
    clean = [json.dumps({'time': i, 'policy': 'p', 'inner': {'time': 9, 'x': 3}}) for i in range(3)]
    frag = json.dumps({'time': 5, 'policy': 'p', 'inner': {'time': 2, 'x': 3}})[:-1]   # cut after the nested object
    got = parse(('\n'.join([clean[0], frag, clean[1], clean[2]]) + '\n').encode())
    assert [r['time'] for r in got.records] == [0, 1, 2] and got.cut == 1 and not got.recovered and not got.bad


def test_a_cut_record_ended_by_a_newline_is_one_lost_record_not_garbage():
    # what a failed write leaves with append_line: the fragment, then a newline, then the next record
    rec = json.dumps(big_record(1, 500))
    got = parse((json.dumps(big_record(0, 50)) + '\n' + rec[:200] + '\n' + json.dumps(big_record(2, 50)) + '\n').encode())
    assert [r['turn'] for r in got.records] == [0, 2] and got.cut == 1 and not got.bad

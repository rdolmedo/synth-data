"""The readers of the workers' resources.jsonl read tbshared.reslog's records (#445 item 2) and the older heartbeat
lines alike: synth/analysis/worker_footprint.py and attempt_time.py."""
import io
import json
from contextlib import redirect_stdout

from synth.analysis import attempt_time, worker_footprint

G = 2 ** 30
OLD = [{'time': 100.0 + 15 * i, 'active': 4, 'cpuacct_usage_ns': str(int(30e9 * i)), 'total_rss': str(2 * G),
        'total_cache': str(G), 'memory_bytes': str(3 * G)} for i in range(3)]
NEW = [{'t': 100.0 + 15 * i, 'fmt': 2, 'active': 4, 'cpu_sec': 30.0 * i, 'mem': str(3 * G),
        'stat': {'total_rss': 2 * G, 'total_cache': G}, 'mem_stat': {'rss': 2 * G, 'cache': G}} for i in range(3)]


def write(tmp_path, rows):
    d = tmp_path / '0'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'resources.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    return d


def test_worker_footprint_reads_both_formats(tmp_path):
    old = worker_footprint.samples(write(tmp_path / 'old', OLD))
    new = worker_footprint.samples(write(tmp_path / 'new', NEW))
    assert len(old) == len(new) == 2
    for a, b in zip(old, new):
        assert (a['cores'], a['rss_gib'], a['cache_gib'], a['mem_gib']) == (b['cores'], b['rss_gib'], b['cache_gib'],
                                                                             b['mem_gib']) == (2.0, 2.0, 1.0, 3.0)


def test_attempt_time_load_join_reads_both_formats(tmp_path, monkeypatch):
    outs = []
    for name, rows in (('old', OLD), ('new', NEW)):
        root = tmp_path / name
        write(root / 'camp' / 'ops' / 'workers', rows)
        monkeypatch.setenv('SYNTH_ROOT', str(root))
        cs = [{'worker': '0', 'start': 110.0, 'duration': 10.0, 'category': 'read'}]
        buf = io.StringIO()
        with redirect_stdout(buf):
            attempt_time.load_join(None, 'camp', cs)
        outs.append(buf.getvalue())
    assert outs[0] == outs[1] and 'n=1' in outs[1]

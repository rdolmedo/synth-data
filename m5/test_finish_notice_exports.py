"""#375 sftnotice: the M5 exporters that ship Pi sessions (gdpval, automationbench, critpt, tbsci_lite) ship each trial's
finish notice map as "finish_notices", mark an attempt whose map is lost unusable, refuse v3 runs, and export runs
without a notice as before. Synthetic pi_eval runs; no cluster data."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import pytest

from aa_evals.common.pi_eval import finish_map
from aa_evals.common.pi_eval.test_finish_map import V3, V4, capped_turns, notice_line, trial_tar

SESSION = '\n'.join(json.dumps(e) for e in [
    {'type': 'session', 'id': 's'},
    {'type': 'message', 'id': 'a', 'parentId': None, 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'task'}]}},
    {'type': 'message', 'id': 'b', 'parentId': 'a', 'message': {'role': 'assistant', 'content': [
        {'type': 'text', 'text': 'done'}], 'stopReason': 'stop', 'usage': {'input': 10, 'output': 5}}}]) + '\n'
FIRST = {'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'}], 'tools': [{'name': 'bash'}]}
NEAR = 131072 - 40000 + 300


@pytest.fixture(autouse=True)
def _fresh_settings():
    finish_map.run_settings.cache_clear()
    yield
    finish_map.run_settings.cache_clear()


def make_run(root: Path, name: str, fn, trials: dict, key_fmt='{t}__a0__r0'):
    """A pi_eval run dir: config.json, attempts/<key>/{outcome.json, trial.tar.gz, attempt.json}, queue/done/<key>.json.
    trials: task -> (notices or None (no file; [] = the guard's empty file), turns text or None, max_prompt_tokens)."""
    run = root / name
    (run / 'queue' / 'done').mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'protocol': {'context_window': 131072, 'finish_notice': fn}}))
    for task, (notices, turns_text, served) in trials.items():
        key = key_fmt.format(t=task)
        adir = run / 'attempts' / key
        adir.mkdir(parents=True)
        trial = f'{key}/{key}/{task}__xyz'
        extra = {f'{trial}/artifacts/home/user/outputs/report.docx': b'PK', f'{trial}/verifier/reward.txt': '1',
                 f'{trial}/artifacts/work/answer.py': 'x = 1'}
        trial_tar(adir / 'trial.tar.gz', trial, SESSION, FIRST, notices, turns_text, extra)
        (adir / 'outcome.json').write_text(json.dumps({'trial_archive': str(adir / 'trial.tar.gz'),
                                                       'trial_dir_in_archive': trial,
                                                       'stats': {'end_state': 'finished', 'max_prompt_tokens': served}}))
        (adir / 'attempt.json').write_text('{}')
        (run / 'queue' / 'done' / f'{key}.json').write_text(json.dumps({'key': key, 'max_prompt_tokens': served}))
    return run


# 'lost': notices were given (served prompt near the window) but the archive has no map file, and its turn log is the
# realistic capped-turns-only one, far from the window (the old turns.jsonl heuristic shipped it as usable).
# 'emptynear': the file is there but empty although a served prompt was inside the threshold.
TRIALS = {'ok': ([notice_line()], capped_turns(3000, 19000), NEAR), 'quiet': ([], capped_turns(3000), 20000),
          'lost': (None, capped_turns(3000, 19000, 35000), NEAR), 'emptynear': ([], None, NEAR)}


def test_gdpval_pack(tmp_path):
    from m5.gdpval import export as E
    run = make_run(tmp_path, 'r4', V4, TRIALS)
    plain = make_run(tmp_path, 'r0', None, {'plain': (None, None, NEAR)})
    out = tmp_path / 'out'
    E.pack(argparse.Namespace(out=str(out), runs=[str(run), str(plain)], min_age=0, final=False, delete_archives=False))
    lines = {x['task']: x for x in map(json.loads, gzip.open(out / 'attempts' / 'attempts-00000.jsonl.gz', 'rt'))}
    assert lines['ok']['finish_notices'] == [notice_line()] and lines['ok']['record']['usable']
    assert lines['quiet']['finish_notices'] == [] and lines['quiet']['record']['usable']
    assert 'finish_notices' not in lines['lost'] and not lines['lost']['record']['usable']
    assert lines['lost']['record']['finish_notice_problem'] == 'finish_notices_missing'
    assert not lines['emptynear']['record']['usable']
    assert lines['emptynear']['record']['finish_notice_problem'] == 'finish_notices_empty_near_window'
    E.manifest(argparse.Namespace(out=str(out), exclude=None))
    assert json.loads((out / 'manifest.json').read_text())['finish_notice_map'] == 1
    assert 'finish_notices' not in lines['plain'] and lines['plain']['record']['usable']   # no notice: as before
    v3 = make_run(tmp_path, 'r3', V3, {'x': (None, None, None)})
    with pytest.raises(SystemExit, match='non-sticky'):
        E.pack(argparse.Namespace(out=str(tmp_path / 'o3'), runs=[str(v3)], min_age=0, final=False, delete_archives=False))


def test_critpt_read_attempt(tmp_path):
    from m5.critpt import export as E
    run = make_run(tmp_path, 'r4', V4, TRIALS)
    got = {t: E.read_attempt(run / 'attempts' / f'{t}__a0__r0') for t in TRIALS}
    assert got['ok'][0]['finish_notices'] == [notice_line()] and got['quiet'][0]['finish_notices'] == []
    assert got['lost'] == (None, 'finish_notices_missing')
    assert got['emptynear'] == (None, 'finish_notices_empty_near_window')
    plain = make_run(tmp_path, 'r0', None, {'plain': (None, None, NEAR)})
    r, why = E.read_attempt(plain / 'attempts' / 'plain__a0__r0')
    assert why == 'ok' and 'finish_notices' not in r


def test_critpt_and_tbsci_refuse_v3_before_writing(tmp_path, monkeypatch):
    from m5.critpt import export as C
    from m5.tbsci_lite import export as T
    ok = make_run(tmp_path, 'r4', V4, {'ok': TRIALS['ok']})
    v3 = make_run(tmp_path, 'r3', V3, {'x': (None, None, None)})
    monkeypatch.setattr(sys, 'argv', ['x', '--runs', str(ok), str(v3), '--out', str(tmp_path / 'c'), '--shard-size', '1'])
    with pytest.raises(SystemExit, match='non-sticky'):
        C.main()
    assert not (tmp_path / 'c' / 'attempts').exists() or not any((tmp_path / 'c' / 'attempts').iterdir())
    with pytest.raises(SystemExit, match='non-sticky'):
        T.collect(argparse.Namespace(export=str(tmp_path / 't'), runs=[str(ok), str(v3)]))
    assert not (tmp_path / 't').exists()


def test_tbsci_lite_attempt_line(tmp_path):
    from m5.tbsci_lite import export as E
    run = make_run(tmp_path, 'r4', V4, TRIALS)
    got = {t: E.attempt_line(run / 'attempts' / f'{t}__a0__r0') for t in TRIALS}
    assert got['ok']['finish_notices'] == [notice_line()] and got['ok']['record']['usable']
    assert not got['lost']['record']['usable'] and 'finish_notices' not in got['lost']
    assert got['lost']['meta']['finish_notice_problem'] == 'finish_notices_missing'
    assert not got['emptynear']['record']['usable']
    v3 = make_run(tmp_path, 'r3', V3, {'x': (None, None, None)})
    with pytest.raises(SystemExit):
        E.attempt_line(v3 / 'attempts' / 'x__a0__r0')


def test_automationbench_read_attempt(tmp_path):
    from m5.automationbench import collect as C
    run = make_run(tmp_path, 'r4', V4, TRIALS)
    got = {t: C.read_attempt(run, {'key': f'{t}__a0__r0', 'item_id': t, 'status': 'ok'}) for t in TRIALS}
    assert got['ok']['finish_notices'] == [notice_line()] and got['ok']['why'] is None
    assert got['quiet']['finish_notices'] == []
    assert got['lost']['finish_notices'] is None and got['lost']['why'] == 'finish_notices_missing'
    assert got['emptynear']['why'] == 'finish_notices_empty_near_window'
    plain = make_run(tmp_path, 'r0', None, {'plain': (None, None, NEAR)})
    p = C.read_attempt(plain, {'key': 'plain__a0__r0', 'item_id': 'plain', 'status': 'ok'})
    assert p['finish_notices'] is None and p['why'] is None and p['sessions']

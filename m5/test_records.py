"""m5/records.py: each test reproduces an M5 ladder failure (#451 workstream 2)."""
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m5 import records as R   # noqa: E402

SYSTEM = {'role': 'system', 'content': 'You are an agent.'}
USER = {'role': 'user', 'content': 'Fix the bug.'}
CALL = {'role': 'assistant', 'reasoning_content': 'look first', 'content': '',
        'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"cmd": "ls"}'}}]}
TOOL = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'a.py'}
FINAL = {'role': 'assistant', 'reasoning_content': 'done', 'content': 'Fixed.'}
TOOLS = [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'type': 'object'}}}]


def turn(k, history, message, used, finish='stop'):
    return {'time': 100.0 + k, 'policy': 'p', 'accounting_policy': 'a', 'turn': k, 'used_before': 0, 'used_after': used,
            'cap': 5000, 'replayed': False, 'messages': history, 'tools': TOOLS,
            'response': {'choices': [{'message': message, 'finish_reason': finish}]}}


def lines():
    return [json.dumps(turn(0, [SYSTEM, USER], CALL, 100, 'tool_calls')),
            json.dumps(turn(1, [SYSTEM, USER, CALL, TOOL], FINAL, 220))]


def wire_bytes(ls, tail=b''):
    return ('\n'.join(ls) + '\n').encode() + tail


def attempt(tmp_path, data, name='r1', outcome=None, task='t1', attempt_index=0):
    att = tmp_path / 'runs' / name / 'attempts' / f'{task}__a{attempt_index}__r0'
    att.mkdir(parents=True)
    with tarfile.open(att / 'trial.tar.gz', 'w:gz') as tf:
        info = tarfile.TarInfo(f'{task}__x/agent/pi-cont16k/response-budget-wire.jsonl')
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    (att / 'outcome.json').write_text(json.dumps(outcome or {'stats': {'end_state': 'natural'}}))
    return att / 'trial.tar.gz'


def test_clean_wire_builds_a_natural_record(tmp_path):
    r = R.attempt_record(attempt(tmp_path, wire_bytes(lines())), {'id': 'x', 'group': 'g'})
    assert 'drop' not in r, r
    assert [m['role'] for m in r['messages']] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert r['weights'] == [0, 0, 1, 0, 1] and r['meta']['natural'] and r['meta']['end'] == 'natural'
    assert r['tools'] == TOOLS


def test_torn_wire_line_is_recovered(tmp_path):
    # 11-15: a wire line = a cut fragment + the whole record (8 TB-Sci-lite attempts); the M5 build died on one
    ls = lines()
    ls[1] = ls[1][:200] + ls[1]
    r = R.attempt_record(attempt(tmp_path, wire_bytes(ls)), {'id': 'x', 'group': 'g'})
    assert 'drop' not in r and r['meta']['natural'] and r['meta']['wire_recovered_lines'] == 1


def test_cut_last_line_keeps_the_wire_before_it(tmp_path):
    ls = lines()
    r = R.attempt_record(attempt(tmp_path, wire_bytes(ls[:1], tail=ls[1][:150].encode())), {'id': 'x', 'group': 'g'})
    assert 'drop' not in r and r['meta']['wire_cut_tail'] and not r['meta']['natural']
    assert r['weights'] == [0, 0, 1]


def test_lost_middle_line_is_a_counted_drop(tmp_path):
    ls = lines()
    data = wire_bytes([ls[0], 'garbage of an interleaved write', ls[1]])
    r = R.attempt_record(attempt(tmp_path, data), {'id': 'x', 'group': 'g'})
    assert r['drop'] == 'wire_unreadable_line' and r['id'] == 'x'


def test_unreadable_archive_is_a_counted_drop_not_a_crash(tmp_path):
    # the HLE build and the TB-Sci-lite resample build each died once on an attempt being packed by a live run
    att = tmp_path / 'runs' / 'r1' / 'attempts' / 't1__a0'
    att.mkdir(parents=True)
    (att / 'trial.tar.gz').write_bytes(b'\x1f\x8b not a finished archive')
    r = R.attempt_record(att / 'trial.tar.gz', {'id': 'x', 'group': 'g'})
    assert r['drop'].startswith('unreadable: ') and r['group'] == 'g'


def test_infrastructure_failure_is_a_drop(tmp_path):
    r = R.attempt_record(attempt(tmp_path, wire_bytes(lines()), outcome={'infrastructure_failure': True}),
                         {'id': 'x', 'group': 'g'})
    assert r == {'id': 'x', 'group': 'g', 'drop': 'infrastructure_failure'}


def real(id_, rnd, natural):
    return {'id': id_, 'round': rnd, 'meta': {'natural': natural}}


def drop(id_, rnd):
    return {'id': id_, 'round': rnd, 'drop': 'infrastructure_failure'}


def m5_rule_before_the_fix(records):
    """fin_pi as first written (11-15): 'a later round replaces an unfinished one', drops included."""
    best = {}
    for r in records:
        prev = best.get(r['id'])
        if prev is None or not (prev.get('meta') or {}).get('natural'):
            best[r['id']] = r
    return best


def test_later_round_drop_never_replaces_a_real_record():
    # 11-16 08:55: 1 AB finishing and 18 AB resample records were lost this way in the first AB build
    rounds = [real('a', 'r1', False), drop('a', 'r2'), drop('a', 'r3')]
    assert 'drop' in m5_rule_before_the_fix(rounds)['a']          # the failure
    assert R.keep_best(rounds)['a']['round'] == 'r1'


@pytest.mark.parametrize('rounds, want', [
    ([real('a', 'r1', False), real('a', 'r2', True), real('a', 'r3', True)], 'r2'),    # first natural in round order
    ([real('a', 'r1', True), drop('a', 'r2'), real('a', 'r3', False)], 'r1'),          # a natural finish is final
    ([real('a', 'r1', False), real('a', 'r2', False), drop('a', 'r3')], 'r2'),         # else the last real record
    ([drop('a', 'r1'), real('a', 'r2', False)], 'r2'),                                 # real replaces a drop
    ([drop('a', 'r1'), drop('a', 'r2')], 'r2'),                                        # only drops: counted as one
])
def test_keep_best(rounds, want):
    assert R.keep_best(rounds)['a']['round'] == want


def test_round_key_orders_rounds_and_retries_numerically():
    tars = [f'/x/runs/{r}/attempts/{a}/trial.tar.gz' for r, a in
            [('r10', 't__a0__r0'), ('r2b', 't__a0__r0'), ('r2', 't__a0__r0'), ('r1', 't__a0__r10'),
             ('r1', 't__a0__r2'), ('r1', 't__a0__r0')]]
    assert [R.round_key(t) for t in sorted(tars, key=R.round_key)] == [
        (1, 'r1', 't', 0, 0), (1, 'r1', 't', 0, 2), (1, 'r1', 't', 0, 10), (2, 'r2', 't', 0, 0), (2, 'r2b', 't', 0, 0),
        (10, 'r10', 't', 0, 0)]
    with pytest.raises(ValueError):
        R.round_key('/x/runs/final/attempts/t__a0__r0/trial.tar.gz')


def test_build_rounds_keeps_best_and_turns_base_errors_into_drops(tmp_path):
    natural_r2 = attempt(tmp_path, wire_bytes(lines()), name='r2')
    unfinished_r1 = attempt(tmp_path, wire_bytes(lines()[:1]), name='r1')
    infra_r3 = attempt(tmp_path, wire_bytes(lines()), name='r3', outcome={'infrastructure_failure': True})
    orphan = attempt(tmp_path, wire_bytes(lines()), name='r4', task='not_in_index')
    index = {'t1': {'id': 'x', 'group': 'g'}}
    best = R.build_rounds([infra_r3, natural_r2, unfinished_r1, orphan], lambda t: dict(index[Path(t).parent.name.split('__a')[0]]))
    assert best['x']['meta']['natural'] and best['x']['meta']['wire_turns'] == 2
    assert best[str(orphan)]['drop'] == 'unreadable: KeyError'   # M5: a missing index entry crashed the build


def test_dedupe_keeps_the_lowest_id_for_every_consumer():
    # a split's fence overshoot: task 'g1' was claimed by run A and run B; CritPt's resample-input build asserted one
    # trace per task and crashed, while SFT assembly kept the lowest id. Every consumer now calls the same rule.
    recs = [{'id': 'critpt:t2/runA/g1__a0', 'group': 'g1'}, {'id': 'critpt:t2/runA/g2__a0', 'group': 'g2'},
            {'id': 'critpt:t2/runB/g1__a0', 'group': 'g1'}, {'id': 'critpt:t2/runA/g2__a0', 'group': 'g2'}]
    kept, dropped = R.dedupe_lowest_id(recs)
    assert [r['id'] for r in kept] == ['critpt:t2/runA/g1__a0', 'critpt:t2/runA/g2__a0'] and dropped == 2
    resample_inputs, _ = R.dedupe_lowest_id(reversed(recs))      # another consumer, another read order
    assert {r['id'] for r in resample_inputs} == {r['id'] for r in kept}


def test_dedupe_prefers_a_real_record_over_a_lower_id_drop():
    recs = [{'id': 'a/runA/g1', 'group': 'g1', 'drop': 'infrastructure_failure'}, {'id': 'a/runB/g1', 'group': 'g1'},
            {'id': 'a/runA/g2', 'group': 'g2', 'drop': 'no_wire'}, {'id': 'a/runB/g2', 'group': 'g2', 'drop': 'no_wire'}]
    kept, dropped = R.dedupe_lowest_id(recs)
    assert [r['id'] for r in kept] == ['a/runB/g1', 'a/runA/g2'] and dropped == 2


def test_history_missing_from_the_wire_is_a_drop(tmp_path):
    ls = lines()
    other = dict(CALL, reasoning_content='a different first turn')
    ls[1] = json.dumps(turn(1, [SYSTEM, USER, other, TOOL], FINAL, 220))
    r = R.attempt_record(attempt(tmp_path, wire_bytes(ls)), {'id': 'x', 'group': 'g'})
    assert r['drop'] == 'history_not_on_wire'


def test_credential_like_text_drops_the_record_and_the_dummy_key_is_scrubbed(tmp_path):
    leak = dict(FINAL, content='token ghp_' + 'A' * 30)
    r = R.attempt_record(attempt(tmp_path, wire_bytes([lines()[0], json.dumps(turn(1, [SYSTEM, USER, CALL, TOOL], leak, 220))])),
                         {'id': 'x', 'group': 'g'})
    assert r['drop'] == 'credential_like_text'
    dummy = dict(FINAL, content='key sk-local-dummy used')
    r = R.attempt_record(attempt(tmp_path / 'b', wire_bytes([lines()[0], json.dumps(turn(1, [SYSTEM, USER, CALL, TOOL], dummy, 220))])),
                         {'id': 'y', 'group': 'g'})
    assert r['messages'][-1]['content'] == 'key [REDACTED] used'


def test_a_cut_line_of_the_writer_is_counted_not_a_drop(tmp_path):
    # append_line ends a failed write's fragment with a newline: the same single lost record as a glued fragment
    ls = lines()
    side = json.dumps({'time': 100.5, 'policy': 'p', 'accounting_policy': 'a', 'replay_tool': 1, 'x': 'y' * 300})
    r = R.attempt_record(attempt(tmp_path, wire_bytes([ls[0], side[:120], ls[1]])), {'id': 'x', 'group': 'g'})
    assert 'drop' not in r and r['meta']['wire_cut_lines'] == 1 and r['meta']['natural']


def test_build_rounds_output_dedupes_even_with_base_errors(tmp_path):
    ok = attempt(tmp_path, wire_bytes(lines()), name='r1')
    orphan = attempt(tmp_path, wire_bytes(lines()), name='r1', task='not_in_index')
    stray = tmp_path / 'elsewhere' / 'trial.tar.gz'     # not under a round run: still a counted record
    stray.parent.mkdir()
    stray.write_bytes(ok.read_bytes())
    best = R.build_rounds([orphan, ok, stray], lambda t: {'t1': {'id': 'x', 'group': 'g'}}[Path(t).parent.name.split('__a')[0]])
    kept, dropped = R.dedupe_lowest_id(best.values())
    assert {r['id'] for r in kept} == {'x', str(orphan), str(stray)} and dropped == 0

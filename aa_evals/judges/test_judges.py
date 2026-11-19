"""Offline tests: python3 -m aa_evals.judges.test_judges  (pytest also collects them)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import arms, deterministic as det, report, selfpref, stats, sweep


def test_hle_mc():
    assert det.hle_mc('Explanation: ...\nAnswer: B\nConfidence: 80%', 'B') is True
    assert det.hle_mc('**Answer:** (C)\n', 'B') is False
    assert det.hle_mc('Answer: A\n...\nFinal answer: D', 'D') is True   # the last answer line counts
    assert det.hle_mc('I think it is B.', 'B') is None                  # no answer line: undecided
    assert det.hle_mc('Answer: Because the ...', 'B') is None           # a word, not a letter
    assert det.hle_mc('Final Answer: A and C', 'A') is None             # several letters: not a single choice
    assert det.hle_mc('Answer: **(B).**', 'B') is True


def test_hle_exact():
    assert det.hle_exact('Exact Answer: $7744$\nConfidence: 90%', '7744') is True
    assert det.hle_exact('Exact Answer: 7,744', '7744') is True
    assert det.hle_exact('Exact Answer: 7800', '7744') is None          # within a "small margin"? undecided
    assert det.hle_exact('Exact Answer: 9000', '7744') is False
    assert det.hle_exact('Exact Answer: \\boxed{\\text{Paris}}', 'paris') is True
    assert det.hle_exact('Exact Answer: x^2+1', '1+x^2') is None        # equivalent forms: left to the judge
    assert det.hle_exact('no answer line', '3') is None
    assert det.hle_exact('Exact Answer: 12%', '0.12') is None           # percent vs fraction: left to the judge
    assert det.hle_exact('Exact Answer: 12%', '15%') is False
    assert det.hle_exact('Exact Answer: 7.5', '7.4') is False           # 1.35% apart (#41 audit 1, item 67d51e1b)


def test_omni_expected():
    assert det.omni_expected('Li Shimin', 'Li Shimin') == 'CORRECT'
    assert det.omni_expected('li shimin.', 'Li Shimin') == 'CORRECT'
    assert det.omni_expected("I don't know.", 'Peter Vroom') == 'NOT_ATTEMPTED'
    assert det.omni_expected('I’m not sure, but possibly X', 'Y') is None   # a hedged guess is an attempt
    assert det.omni_expected("I'm not sure.", 'Y') == 'NOT_ATTEMPTED'
    assert det.omni_expected('I am not fully confident. I believe it was **EBITDA**.', 'Y') is None
    assert det.omni_expected('I do not know the exact date.', '4 May') == 'NOT_ATTEMPTED'
    assert det.omni_expected('I do not have sufficient information', 'Y') == 'NOT_ATTEMPTED'
    assert det.omni_expected('19.7 million', '20.2 million') is None    # not plain numbers
    assert det.omni_expected('1.23', 'go1.8') is None
    assert det.omni_expected('42', '30') == 'INCORRECT'
    assert det.omni_expected('Erik Erikson', 'Otto Fenichel') is None
    assert det.omni_expected('Unknown Pleasures', 'Closer') is None     # a title that starts with "Unknown"
    assert det.omni_expected('Unknown.', 'Closer') == 'NOT_ATTEMPTED'
    assert det.omni_expected('Not sure of the year.', '1066') is None   # not a bare phrase: left to the judge
    # audit 1 (#41) follow-up: a phrase followed by an answer, or a second clause, is left to the judge
    assert det.omni_expected('Unknown - Closer', 'Closer') is None
    assert det.omni_expected('Unknown \u2014 Closer', 'Closer') is None
    assert det.omni_expected('Unknown, Closer', 'Closer') is None
    assert det.omni_expected('Not sure!', 'Closer') == 'NOT_ATTEMPTED'
    assert det.omni_expected('I am not sure of the exact name; it was Closer', 'Joy') is None
    assert det.omni_expected("I don't know. It was Closer.", 'Joy') is None
    assert det.omni_expected('I do not know the answer to this question.', 'Joy') == 'NOT_ATTEMPTED'
    assert det.omni_expected("I don't know the exact amount; I need the 2018 statement.", 'Joy') == 'NOT_ATTEMPTED'
    assert det.omni_expected('54', '35%') == 'INCORRECT'                # far under both readings
    assert det.omni_expected('0.35', '35%') is None                     # equal as a fraction: left to the judge


def test_kappa_and_majority():
    assert stats.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(stats.cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) - 0.0) < 1e-9
    assert stats.cohen_kappa(['A', 'A'], ['A', 'A']) == 1.0
    assert stats.majority([True, True, False]) is True
    assert stats.majority([True, False]) is None
    assert stats.rate_ci(0, 0) is None


def test_arms_consistent():
    for a in arms.ARMS.values():
        assert a.role in ('candidate', 'reference')
        if a.role == 'candidate':
            assert a.gpus == 1, a.name
    for server, primary in arms.PRIMARY.items():
        assert arms.ARMS[primary].server == server


def _omni_run(root: Path, rid: str, grades: dict, responses: dict, gold: dict) -> None:
    d = root / 'omniscience' / rid
    d.mkdir(parents=True)
    with (d / 'items.jsonl').open('w') as f:
        for iid, g in grades.items():
            f.write(json.dumps({'item_id': iid, 'repeat': 0, 'status': 'ok', 'grade': g, 'grade_source': 'judge',
                                'response': responses[iid], 'gold': gold[iid], 'judge_prompt_tokens': 2000,
                                'judge_completion_tokens': 300}) + '\n')
    (d / 'summary.json').write_text(json.dumps({'metrics': {'accuracy': {'score': 50.0, 'ci95': [40, 60]},
                                                            'non_hallucination_rate': {'score': 70.0}}}))


def test_report_on_synthetic_study():
    gold = {'1': 'Paris', '2': 'Rome', '3': '42', '4': 'X', '5': 'Y', '6': 'Z'}
    resp = {'1': 'Paris', '2': "I don't know.", '3': '10', '4': 'X-ish', '5': 'Y', '6': 'W'}
    g_qwen = {'1': 'CORRECT', '2': 'NOT_ATTEMPTED', '3': 'INCORRECT', '4': 'CORRECT', '5': 'CORRECT', '6': 'INCORRECT'}
    g_super = {'1': 'CORRECT', '2': 'NOT_ATTEMPTED', '3': 'CORRECT', '4': 'PARTIAL_ANSWER', '5': 'CORRECT',
               '6': 'INCORRECT'}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _omni_run(root, 'j41-t-qwen38-27b-medium', g_qwen, resp, gold)
        _omni_run(root, 'j41-t-super-medium', g_super, resp, gold)
        _omni_run(root, 'j41-t-qwen38-27b-medium-rt', g_qwen, resp, gold)
        orig = sweep.STUDY_ROOT
        sweep.STUDY_ROOT = root / 'judges'
        try:
            for arm, step, rid in [('qwen38-27b-medium', 'omni', 'j41-t-qwen38-27b-medium'),
                                   ('qwen38-27b-medium', 'omni-rt', 'j41-t-qwen38-27b-medium-rt'),
                                   ('super-medium', 'omni', 'j41-t-super-medium')]:
                sweep.append_step('t', {'kind': 'step', 'arm': arm, 'step': step, 'run_id': rid, 'rc': 0,
                                        'bench_dir': str(root / 'omniscience'), 'wall_s': 360.0, 'concurrency': 64})
            rep = report.build('t')
        finally:
            sweep.STUDY_ROOT = orig
    B = rep['benches']['omni']
    q, s = B['arms']['qwen38-27b-medium'], B['arms']['super-medium']
    # deterministic: 1 and 5 sure-correct, 2 sure-abstain, 3 sure-wrong (42 vs 10) -> qwen 4/4, super 3/4
    assert q['det']['n'] == 4 and q['det']['agreement'] == 100.0
    assert s['det']['agreement'] == 75.0 and s['det']['false_accept'] == 50.0
    assert q['retest_kappa_4class'] == 1.0
    assert q['gpu_h_full_run'] == 0.1 * 300 / 6
    assert B['n_common'] == 6 and 'qwen38-27b-medium | super-medium' in B['pairwise_kappa']
    assert s['leniency_pp'] == 0.0  # both call 3 of 6 correct (different ones)
    assert 'qwen38-27b-medium' in report.to_markdown(rep)


def _cells(n: int, q_on_q: int, s_on_q: int, q_on_s: int, s_on_s: int) -> dict:
    """Synthetic omni cells: n items per source; the counts are how many items each judge passes."""
    def judged(k_pass):
        return {(str(i), 0): {'ok': True, 'bin': i < k_pass, 'label': 'CORRECT' if i < k_pass else 'INCORRECT',
                              'response': f'r{i}', 'gold': f'g{i}'} for i in range(n)}
    return {('qwen', 'qwen'): judged(q_on_q), ('qwen', 'super'): judged(s_on_q),
            ('super', 'qwen'): judged(q_on_s), ('super', 'super'): judged(s_on_s)}


def test_selfpref_separates_leniency_from_interaction():
    # Qwen's judge passes 10 more items than Super's on both sources: pure leniency, no self-preference.
    r = selfpref.analyse('omni', _cells(100, 40, 30, 50, 40), n_boot=300)
    assert r['leniency_pp']['estimate'] == 10.0 and r['self_preference_pp']['estimate'] == 0.0
    assert r['intersected'] and r['self_preference_pp']['ci95'][0] <= 0 <= r['self_preference_pp']['ci95'][1]
    # Each judge favours its own family by 10 items: no net leniency, a 20 pp interaction.
    r = selfpref.analyse('omni', _cells(100, 40, 30, 30, 40), n_boot=300)
    assert r['leniency_pp']['estimate'] == 0.0 and r['self_preference_pp']['estimate'] == 20.0
    assert r['self_preference_pp']['ci95'][0] > 0
    # One source only: the 2x2 is incomplete, cells are still reported.
    c = _cells(10, 5, 5, 5, 5)
    del c[('super', 'qwen')]
    r = selfpref.analyse('omni', c)
    assert 'self_preference_pp' not in r and r['sources'] == ['qwen']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)

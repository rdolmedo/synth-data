"""aa_evals.common.aai_open: version labels, the run-config record, env forwarding and the aggregate (#481)."""
import json

import pytest

from aa_evals.common import aai_open as A

# v0 golden values (docs/aai_open/VERSIONS.md): m5l-sb-full's slot scores (M5 ladder, raw convention, 16.05 on
# marin#11563's post-mortem) and DeepSeek V4.1 Flash Max's c30 raw scores (out/table.json, universe convention 41.63
# = marin#11563's open-model table)
SB_FULL = {'tb21': 25.0, 'tbsci_lite': 1.56, 'automationbench': 38.91, 'critpt': 1.43, 'hle': 4.46,
           'omniscience': 14.0, 'omni_nhr': 65.76, 'scicode_main': 14.44, 'gdppdf_allpass': 6.76,
           'gdpval': 978.5368855902548}
SB2_0P33B = {'tb21': 17.86, 'automationbench': 32.64, 'omniscience': 13.11, 'omni_nhr': 61.13, 'scicode_main': 7.78,
             'gdppdf_allpass': 6.76, 'gdpval': 829.2193}


def test_labels():
    assert A.label() == A.CURRENT == 'v0.1'
    assert A.label(None, ['x']) == 'v0.1+custom'
    assert A.label('v0.1+custom', ['x']) == 'v0.1+custom'                 # never doubled
    assert A.label('v0', []) == 'v0'
    for bad in ('0.1', 'v0.1 ', "v0.1'", 'v0.1+other', 'latest', 'v0.1\n'):
        assert not A.valid(bad)
        with pytest.raises(ValueError):
            A.label(bad)


def test_record_from_launcher_client_and_continued_runs():
    assert A.record([], env={}) == {'version': 'v0.1', 'deviations': [], 'source': 'client'}
    assert A.record([], env={A.ENV: 'v0.1'})['source'] == 'launcher'
    r = A.record(['sampling override temperature=0.6'], env={A.ENV: 'v0.1'})
    assert r['version'] == 'v0.1+custom' and r['deviations'] == ['sampling override temperature=0.6']
    assert A.record([], env={A.ENV: 'v0.1+custom'})['version'] == 'v0.1+custom'   # the launcher's own deviations
    # a malformed value (a future label format seen by older code, a typo) never fails a run: CURRENT+custom
    for bad in ('nonsense', 'v0.1\n', "v0.1' X='y", 'v0.2-rc1'):
        r = A.record([], env={A.ENV: bad})
        assert r['version'] == 'v0.1+custom' and r['deviations'][0].startswith('unrecognized AAI_OPEN_VERSION')
    # a grade stage / resume keeps the label of the run it continues; a new deviation (another judge) is added once
    gen = {'aai_open': {'version': 'v0.1', 'deviations': [], 'source': 'launcher'}}
    assert A.record([], source_cfg=gen) == gen['aai_open']
    g = A.record(A.judge_deviations('qwen38-27b-nothink'), source_cfg=gen)
    assert g['version'] == 'v0.1+custom' and g['source'] == 'launcher' and len(g['deviations']) == 1
    assert A.record(A.judge_deviations('qwen38-27b-nothink'), source_cfg={'aai_open': g})['deviations'] == g['deviations']
    # a run started before the field stays unrecorded, whatever this process's env says, and so does a run that
    # continues it (a re-grade of a re-grade)
    old = A.record([], env={A.ENV: 'v0.1'}, source_cfg={})
    assert old['version'] == 'unrecorded'
    assert A.record(A.judge_deviations(None), source_cfg={'aai_open': old})['version'] == 'unrecorded'


def test_client_deviations():
    assert A.client_deviations(context_window=131072, overrides={'temperature': None, 'top_p': None}) == []
    assert A.client_deviations(context_window=262144)[0].startswith('context_window 262144')
    assert A.client_deviations(overrides={'temperature': 0.6}) == ['sampling override temperature=0.6']
    v4 = {**A.NOTICE, 'sticky': True, 'final_at': 24576, 'notice_margin': 256, 'final_slack': 2048}
    assert A.client_deviations(notice=v4) == []
    assert A.client_deviations(notice=None)[0].startswith('finish notice')                   # --finish-notice off
    assert A.client_deviations(notice={**v4, 'warn_at': 30000})[0].startswith('finish notice')
    assert A.client_deviations(notice={**v4, 'text_sha256': '0' * 64})[0].startswith('finish notice')
    assert A.client_deviations(images='on') == ['image inputs on (v0.1: text only)']
    assert A.judge_deviations(A.JUDGE_ARM) == [] and A.judge_deviations(None)


def test_forwarded_and_condor_env():
    assert A.forwarded({}) == {} and A.condor_env({}) == ''
    assert A.forwarded({A.ENV: 'v0.1+custom'}) == {A.ENV: 'v0.1+custom'}
    assert A.condor_env({A.ENV: 'v0.1'}) == " AAI_OPEN_VERSION='v0.1'"
    for bad in ("v0.1' X='y", 'v0.1\n'):                                 # nothing unvalidated reaches a submit file
        assert A.forwarded({A.ENV: bad}) == {A.ENV: 'v0.1+custom'}


def test_aggregate_reproduces_v0():
    assert A.aggregate(SB_FULL)['score'] == pytest.approx(16.0495, abs=1e-3)
    lean = A.aggregate(SB2_0P33B)
    assert lean['points'] == 65 and lean['missing'] == ['tbsci_lite', 'critpt', 'hle']
    assert lean['score'] == pytest.approx(11.76, abs=0.01)
    filled = A.aggregate(SB2_0P33B, fill={k: SB_FULL[k] for k in ('tbsci_lite', 'critpt', 'hle')})
    assert filled['points'] == 100 and filled['filled'] == ['tbsci_lite', 'critpt', 'hle']
    assert filled['score'] == pytest.approx(12.58, abs=0.01)
    # DeepSeek V4.1 Flash Max, c30 (pre-notice) raw scores: the c30 leaderboard convention (raw x retained/universe,
    # GDPval Elo through the clamp) gives marin#11563's 41.6 (41.63); the raw convention would give 48.2
    ds = {'tb21': 75.0, 'tbsci_lite': 39.06, 'automationbench': 63.24, 'critpt': 56.67, 'scicode_main': 68.89,
          'hle': 25.8, 'omniscience': 38.0, 'omni_nhr': 37.63, 'gdppdf_allpass': 32.43, 'gdpval': 1277.7}
    assert A.aggregate(ds, 'universe')['score'] == pytest.approx(41.63, abs=0.005)
    assert A.aggregate(ds)['score'] == pytest.approx(48.24, abs=0.01)
    assert A.briefcase(400) == 0 and A.briefcase(3000) == 100 and A.briefcase(1500) == 50
    assert sum(w for _, w in A.SLOTS) == 100


def test_of_config_and_of_run(tmp_path):
    pi = tmp_path / 'pi'
    pi.mkdir()
    (pi / 'config.json').write_text(json.dumps({'aai_open': {'version': 'v0.1'}}))
    coll = tmp_path / 'collected'
    coll.mkdir()
    (coll / 'config.json').write_text(json.dumps({'harness': {'pi_run': str(pi)}}))
    assert A.of_run(pi) == 'v0.1' and A.of_run(coll) == 'v0.1'
    assert A.of_run(tmp_path / 'missing') == 'unrecorded'
    assert A.of_config({'harness': 'aa_evals.hle: ...'}) == 'unrecorded'


def test_pi_eval_records_what_it_can_see():
    """pi_eval.run.pi_deviations: context window, notice, images and hand-given request settings of a new Pi run."""
    import types
    pi_run = pytest.importorskip('aa_evals.common.pi_eval.run')
    v4 = {**A.NOTICE, 'sticky': True}
    a = types.SimpleNamespace(request_overrides='', request_preset='')
    base = {'context_window': 131072, 'finish_notice': v4, 'model_input': ['text']}
    assert pi_run.pi_deviations(a, base) == []
    assert pi_run.pi_deviations(a, {**base, 'context_window': 262144})
    assert pi_run.pi_deviations(a, {k: v for k, v in base.items() if k != 'finish_notice'})   # --finish-notice off
    assert pi_run.pi_deviations(a, {**base, 'model_input': ['text', 'image']}) == ['image inputs on (v0.1: text only)']
    assert pi_run.pi_deviations(types.SimpleNamespace(request_overrides='{"temperature":0.6}', request_preset=''),
                                base) == ['sampling override temperature=0.6']
    assert pi_run.pi_deviations(types.SimpleNamespace(request_overrides='', request_preset='qwen38-27b'), base)
    # what the run's default notice is: the v0.1 pin (a notice text change would fail here first)
    assert pi_run.pi_deviations(a, {**base, 'finish_notice': pi_run.finish_notice(types.SimpleNamespace())}) == []

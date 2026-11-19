"""aa_evals.campaign.score: slot scores from run summaries, the aggregate and each run's AAI-open version (#481)."""
import json

from aa_evals.campaign import score as S
from aa_evals.campaign.test_launch import campaign
from aa_evals.common import aai_open as A


def write(path, obj):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'summary.json' if isinstance(obj, dict) and 'aai_open' not in obj else path / 'config.json').write_text(
        json.dumps(obj))


def test_scores_versions_and_lean_aggregate(tmp_path):
    C = campaign(tmp_path)
    C.models['qwen36']['suites'] = ['tb21', 'automationbench', 'scicode', 'omniscience', 'gdppdf', 'hle']
    runs, pi = tmp_path / 'runs', tmp_path / 'pi'
    rid = lambda s: C.run_id('qwen36', s, 'full')   # noqa: E731
    write(pi / 'tb21' / rid('tb21'), {'score_resolved_only': 25.0})
    write(pi / 'tb21' / rid('tb21'), {'aai_open': {'version': 'v0.1'}})
    write(runs / 'automationbench' / rid('automationbench'), {'score': 38.91})
    write(pi / 'automationbench_pi' / rid('automationbench'), {'aai_open': {'version': 'v0.1+custom'}})
    write(runs / 'scicode' / rid('scicode'), {'main_problem_accuracy': 14.44, 'sub_step_accuracy': 40.75})
    write(runs / 'omniscience' / f"{rid('omniscience')}-graded",
          {'score': 14.0, 'metrics': {'non_hallucination_rate': {'score': 65.76}}})
    write(runs / 'gdppdf' / f"{rid('gdppdf')}-graded", {'score': 6.76, 'mean_pass': 52.1})
    # HLE generated but not graded: missing; GDPval and the rest not run
    (runs / 'hle' / rid('hle')).mkdir(parents=True)
    r = S.report(C, ['qwen36'], 'full', 'raw', runs=runs, pi_runs=pi)['qwen36']
    assert r['scores'] == {'tb21': 25.0, 'automationbench': 38.91, 'scicode_main': 14.44, 'omniscience': 14.0,
                           'omni_nhr': 65.76, 'gdppdf_allpass': 6.76}
    assert r['extra'] == {'scicode_sub': 40.75, 'gdppdf_meanpass': 52.1}
    assert r['versions']['tb21'] == 'v0.1' and r['versions']['automationbench'] == 'v0.1+custom'
    assert r['versions']['hle'] == 'unrecorded'
    assert r['aai_open_versions'] == ['unrecorded', 'v0.1', 'v0.1+custom']
    agg = r['aggregate']
    assert agg['points'] == 50 and set(agg['missing']) == {'tbsci_lite', 'critpt', 'hle', 'gdpval'}
    assert abs(agg['score'] - A.aggregate(r['scores'])['score']) < 1e-12


def test_scicode_from_items_while_grading(tmp_path):
    d = tmp_path / 'sc'
    d.mkdir()
    rows = [{'problem_id': '1', 'repeat': 0, 'score': 1}, {'problem_id': '1', 'repeat': 0, 'score': 1},
            {'problem_id': '2', 'repeat': 0, 'score': 1}, {'problem_id': '2', 'repeat': 0, 'score': 0}]
    (d / 'items.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    assert S.scicode_scores(d) == (50.0, 75.0)
    assert S.scicode_scores(tmp_path / 'none') is None

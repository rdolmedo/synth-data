"""Synthetic scoring tests: no GPU, serving endpoint, benchmark data or run artifacts."""
import json
from types import SimpleNamespace

from aa_evals.gdpval import run as R


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    (root / 'matches').mkdir(parents=True)
    cfg = {'selection': {'task_ids': ['t']}, 'repeats': 2, 'git_sha': 'generation-sha',
           'model': {'spec': 'serving:offline', 'served_name': 'm', 'body': {}, 'vision': False},
           'context': {'window': 131072}}
    (root / 'config.json').write_text(json.dumps(cfg))
    outcomes = {R.key('t', r): {'status': 'ok', 'end': 'finish'} for r in (0, 1)}
    monkeypatch.setattr(R, 'RUNS', tmp_path)
    monkeypatch.setattr(R, 'load_outcomes', lambda *a: outcomes)
    monkeypatch.setattr(R.T, 'load_tasks', lambda: [SimpleNamespace(task_id='t', deliverable_files=['expert'])])
    args = SimpleNamespace(run_id='run', judge_arm='primary', bootstrap=5)
    return root, cfg, outcomes, args


def match(repeat, order, score=1):
    return {'key': R.key('t', repeat), 'task_id': 't', 'repeat': repeat, 'order': order,
            'winner': 'tie' if score == 0.5 else 'A', 'model_score': score}


def append(root, arm, rows):
    with (root / 'matches' / f'{arm}.jsonl').open('a') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def test_error_and_single_orientation_are_pending_not_final(tmp_path, monkeypatch):
    root, cfg, outcomes, args = fixture(tmp_path, monkeypatch)
    append(root, 'primary', [match(0, 'MA'), match(0, 'MB'), match(1, 'MA', 0),
                             {'key': R.key('t', 1), 'task_id': 't', 'order': 'MB', 'error': 'bad schema'}])
    R.stage_score(args)
    s = json.loads((root / 'summary.json').read_text())
    p = s['per_judge']['primary']
    assert s['generation_complete'] and not s['complete'] and not s['valid'] and s['partial']
    assert s['score'] is not None  # explicitly provisional, so partial arms remain reportable
    assert (s['n_pending_generations'], s['n_pending_judgments'], s['n_pending']) == (0, 1, 1)
    assert (p['n_expected_judgments'], p['n_completed_judgments'], p['n_judgments']) == (4, 3, 3)
    assert p['n_complete_attempts'] == 1 and p['n_partial_attempts'] == 1 and p['judge_errors'] == 1
    assert p['n_expected_anchored_judgments'] == 4 and p['n_pending_anchored_judgments'] == 1
    rows = {r['repeat']: r for r in R.item_rows(root, cfg, list(outcomes), outcomes, 'primary')}
    assert rows[0]['score'] == 1 and rows[0]['judge_status'] == 'complete'
    assert rows[1]['score'] is None and rows[1]['score_MA'] == 0 and rows[1]['score_MB'] is None
    assert rows[1]['judge_status'] == 'partial'


def test_completed_primary_allows_partial_secondary_and_deduplicates_retries(tmp_path, monkeypatch):
    root, cfg, outcomes, args = fixture(tmp_path, monkeypatch)
    append(root, 'primary', [match(r, order, 0.5) for r in (0, 1) for order in ('MA', 'MB')] +
                            [match(0, 'MA', 0.5), {'key': R.key('t', 0), 'order': 'MA', 'error': 'old failure'}])
    append(root, 'secondary', [match(0, 'MA')])
    args.judge_arm = None
    monkeypatch.setattr(R, 'DEFAULT_JUDGE_ARM', 'primary')
    R.stage_score(args)
    s = json.loads((root / 'summary.json').read_text())
    assert s['valid'] and s['complete'] and not s['partial'] and s['n_pending'] == 0
    assert s['per_judge']['primary']['n_completed_judgments'] == 4
    assert s['per_judge']['secondary']['partial'] and not s['per_judge']['secondary']['complete']
    assert s['per_judge']['secondary']['n_pending_judgments'] == 3
    assert s['per_judge']['secondary']['elo'] is not None


def test_generation_without_judgments_never_becomes_valid_score(tmp_path, monkeypatch):
    root, cfg, outcomes, args = fixture(tmp_path, monkeypatch)
    args.judge_arm = None
    R.stage_score(args)
    s = json.loads((root / 'summary.json').read_text())
    assert s['generation_complete'] and s['score'] is None and not s['valid'] and not s['complete']
    assert s['n_pending'] == 2 and s['n_pending_judgments'] == 4
    assert all(r['score'] is None for r in R.item_rows(root, cfg, list(outcomes), outcomes))


def test_generation_infra_is_excluded_from_judgment_expectations(tmp_path, monkeypatch):
    root, cfg, outcomes, args = fixture(tmp_path, monkeypatch)
    outcomes[R.key('t', 1)] = {'status': 'infra_error', 'end': 'infra'}
    append(root, 'primary', [match(0, 'MA', 0.5), match(0, 'MB', 0.5), match(1, 'MA'), match(1, 'MB')])
    R.stage_score(args)
    s = json.loads((root / 'summary.json').read_text())
    assert s['complete'] and not s['valid']  # one of two infra outcomes exceeds2%, independently of judging
    assert s['n_infra'] == 1 and s['per_judge']['primary']['n_expected_judgments'] == 2
    assert s['per_judge']['primary']['n_judgments'] == 2
    rows = R.item_rows(root, cfg, list(outcomes), outcomes, 'primary')
    assert rows[1]['score'] is None and rows[1]['judge_status'] == 'excluded_infra'

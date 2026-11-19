"""aa_evals.gdpval.refs: c30's two-reference draws (checked against c30n's frozen pools: identical, 2026-11-13)."""
from aa_evals.gdpval import refs as R


def test_make_pairs_one_per_reference_and_skips():
    fin = {'run': {'t1': [0, 1], 't2': [0]}, 'refA': {'t1': [0, 1, 2], 't2': [1]}, 'refB': {'t1': [4]}}
    pairs, skipped = R.make_pairs('run', ['refA', 'refB'], fin)
    assert len(pairs) == 2 * 2 + 1 and skipped == [{'task_id': 't2', 'repeat': 0, 'ref': 'refB'}]
    for p in pairs:
        assert p['pair'] == f"{p['task_id']}|{p['a']['run']}|{p['a']['repeat']}|{p['b']['run']}|{p['b']['repeat']}"
        assert (p['a']['run'], p['a']['repeat']) < (p['b']['run'], p['b']['repeat'])
        assert p['order'] == R.order_of(p['pair'])
    assert pairs == R.make_pairs('run', ['refA', 'refB'], fin)[0]          # deterministic
    t1b = [p for p in pairs if p['task_id'] == 't1' and 'refB' in p['pair']]
    assert {p['b']['repeat'] if p['b']['run'] == 'refB' else p['a']['repeat'] for p in t1b} == {4}

"""durations_v2 builder: censored fit and the v1 fallback (synthetic data)."""
import math
import random

import pytest

from aa_evals.common.dispatch import build_durations as bd


def synth(seed=0, cap=7200.0, sigma=0.3):
    """3 models x 30 items, 4 repeats; model effects 0 / +0.5 / -0.5, item medians 100 s .. ~20000 s."""
    rnd = random.Random(seed)
    eff = {'a': 0.0, 'b': 0.5, 'c': -0.5}
    recs, truth = [], {}
    for k in range(30):
        item = f'i{k}'
        base = math.log(100) + k * 0.18
        for m, e in eff.items():
            truth[(m, item)] = base + e
            for rep in range(4):
                t = math.exp(base + e + rnd.gauss(0, sigma))
                cens = t > cap
                recs.append({'model': m, 'suite': 'tb21', 'item': item, 'rep': rep, 'sec': min(t, cap),
                             'sec_raw': min(t, cap), 'censored': cens})
    return recs, truth


def test_fit_recovers_censored_medians():
    recs, truth = synth()
    t = bd.build_v2(recs + [dict(recs[0], stage='canary', sec=1.0, sec_raw=1.0)], None)   # canaries are ignored
    assert t['lognormal']['mean_factor']['tb21'] == pytest.approx(1.0, abs=0.1)
    assert any(r['censored'] for r in recs)
    lg = t['lognormal']
    assert lg['sigma']['tb21'] == pytest.approx(0.3, abs=0.05)
    assert lg['model_log_effect']['tb21']['b'] - lg['model_log_effect']['tb21']['a'] == pytest.approx(0.5, abs=0.1)
    # a cell whose attempts all hit the cap is estimated past it, near the truth; a naive mean would say 7200
    m, i = max(truth, key=truth.get)
    assert all(r['censored'] for r in recs if (r['model'], r['item']) == (m, i))
    est = math.log(t['suites']['tb21'][i][m])
    assert est > math.log(7200) + 0.3
    assert est == pytest.approx(truth[(m, i)], abs=0.35)
    err = [abs(math.log(t['suites']['tb21'][i][mm]) - v) for (mm, i), v in truth.items()]
    assert sum(err) / len(err) < 0.15


def test_v1_fallback_scaled_and_exclude():
    recs, _ = synth(cap=1e9)
    recs = [r for r in recs if not (r['model'] == 'c' and r['item'] == 'i3')]     # a model-item c30 lacks
    v1 = {'models': ['a', 'b', 'c', 'old'], 'suites': {
        'tb21': {**{f'i{k}': {'a': 50.0 * math.exp(k * 0.18), 'b': 80.0 * math.exp(k * 0.18),
                              'c': 30.0 * math.exp(k * 0.18), 'old': 70.0} for k in range(30)},
                 'v1only': {'a': 500.0, 'old': 100.0, 'b': 1.0}},
        'newsuite': {'q': {'a': 42.0}}}}
    t = bd.build_v2(recs, v1)
    s = t['suites']['tb21']
    al, be = t['provenance']['tb21']['v1_map']['a']
    assert be == pytest.approx(1.0, abs=0.1) and al == pytest.approx(math.log(2), abs=0.5)   # c30 'a' ~ 2x v1
    assert s['v1only']['a'] == pytest.approx(math.exp(al + be * math.log(500)), rel=1e-3)
    assert s['v1only']['a'] == pytest.approx(1000, rel=0.15)
    al, be = t['provenance']['tb21']['v1_map']['c']
    assert 'c' in s['i3'] and s['i3']['c'] == pytest.approx(
        math.exp(al + be * math.log(v1['suites']['tb21']['i3']['c'])), rel=1e-3)
    assert 'old' in t['models'] and s['v1only']['old'] > 0              # a model c30 lacks: suite-wide map
    assert 'b' not in s['v1only']                                       # placeholders are not durations
    assert t['suites']['newsuite'] == {'q': {'a': 42.0}}                # a suite c30 lacks: v1 as is
    lomo = bd.build_v2(recs, v1, exclude=['a'])
    assert 'a' not in lomo['models'] and all('a' not in per for per in lomo['suites']['tb21'].values())

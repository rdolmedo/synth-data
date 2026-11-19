"""AA-Omniscience metrics.

With c, i, p, a = counts of CORRECT, INCORRECT, PARTIAL_ANSWER, NOT_ATTEMPTED and N = c + i + p + a:
  accuracy                 = c / N                      (AA index weight 10%)
  hallucination rate       = i / (i + p + a) = i / (N - c)
  non-hallucination rate   = 1 - i / (N - c)            (AA index weight 5%)
  AA-Omniscience Index     = 100 * (c - i) / N          (AA's "net score", -100..100)

Items cut off at the run deadline (no answer; x = their count, label CUTOFF) are not correct for accuracy and the
index (N includes them) but are excluded from the non-hallucination denominator, since an unanswered item says
nothing about hallucination (audit 1, #31). The conservative variants count them as INCORRECT instead.

These reproduce AA's published numbers exactly. For Nemotron 3.5 Super VL with N = 3000: accuracy 0.211667 gives
c = 635, index -2.8667 gives i = 721, and 1 - 721/2365 = 0.695137 is AA's non-hallucination rate.
"""
from __future__ import annotations

import random

GRADE_KEYS = ('CORRECT', 'INCORRECT', 'PARTIAL_ANSWER', 'NOT_ATTEMPTED')
CUTOFF = 'CUTOFF'
METRIC_NAMES = {
    'accuracy': 'AA-Omniscience Accuracy',
    'non_hallucination_rate': 'AA-Omniscience Non-Hallucination Rate',
    'omniscience_index': 'AA-Omniscience Index',
    'non_hallucination_rate_cutoff_incorrect': 'Non-Hallucination Rate, cutoffs counted INCORRECT (conservative)',
    'omniscience_index_cutoff_incorrect': 'Omniscience Index, cutoffs counted INCORRECT (conservative)',
}


def counts(grades) -> dict:
    out = dict.fromkeys(GRADE_KEYS + (CUTOFF,), 0)
    for g in grades:
        out[g] += 1
    return out


def from_counts(c: int, i: int, p: int, a: int, x: int = 0) -> dict:
    """All metrics on AA's display scale (percent for the rates, -100..100 for the index). x = cutoffs."""
    n = c + i + p + a + x
    if n == 0:
        return dict.fromkeys(METRIC_NAMES)
    answered_not_correct = i + p + a
    hall = i / answered_not_correct if answered_not_correct else 0.0
    hall_cons = (i + x) / (n - c) if n - c else 0.0
    return {'accuracy': 100.0 * c / n,
            'non_hallucination_rate': 100.0 * (1.0 - hall),
            'omniscience_index': 100.0 * (c - i) / n,
            'non_hallucination_rate_cutoff_incorrect': 100.0 * (1.0 - hall_cons),
            'omniscience_index_cutoff_incorrect': 100.0 * (c - i - x) / n}


def compute(grades) -> dict:
    k = counts(grades)
    return from_counts(k['CORRECT'], k['INCORRECT'], k['PARTIAL_ANSWER'], k['NOT_ATTEMPTED'], k[CUTOFF])


def bootstrap_ci(grades_by_item: dict[str, list[str]], n_boot: int = 2000, seed: int = 0) -> dict:
    """95% percentile bootstrap over items; an item's repeats are resampled together."""
    keys = sorted(grades_by_item)
    if not keys:
        return {m: None for m in METRIC_NAMES}
    per_item = [counts(grades_by_item[k]) for k in keys]
    rng = random.Random(seed)
    samples = {m: [] for m in METRIC_NAMES}
    for _ in range(n_boot):
        c = i = p = a = x = 0
        for _ in keys:
            k = per_item[rng.randrange(len(keys))]
            c += k['CORRECT']; i += k['INCORRECT']; p += k['PARTIAL_ANSWER']; a += k['NOT_ATTEMPTED']
            x += k[CUTOFF]
        for m, v in from_counts(c, i, p, a, x).items():
            samples[m].append(v)
    out = {}
    for m, vals in samples.items():
        vals.sort()
        out[m] = [round(vals[int(0.025 * (n_boot - 1))], 2), round(vals[int(0.975 * (n_boot - 1))], 2)]
    return out

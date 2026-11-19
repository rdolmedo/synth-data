"""summary.json from items.jsonl rows: pass@1 overall and by slice, CI95, HLE's calibration error."""
import json
import math
from pathlib import Path

AA_LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
AA_KEYS = {'cais/hle-diamond': 'hle-diamond', 'cais/hle': 'humanitys-last-exam'}
MAX_INFRA_RATE = 0.02


def aa_reference(slug, dataset):
    """AA's published score (0-100) for `slug` on `dataset`, or None."""
    if not slug or dataset not in AA_KEYS:
        return None
    try:
        board = json.loads(AA_LEADERBOARD.read_text())
    except (OSError, ValueError):
        return None
    for m in board['models']:
        if m['slug'] == slug:
            v = m['scores'].get(AA_KEYS[dataset])
            return None if v is None else round(100 * v, 2)
    return None


def calib_err(confidence, correct, p='2', beta=100):
    """run_judge_results.py's RMS calibration error (bins of `beta` items, sorted by confidence); None below
    2 bins, where the official code fails or degenerates."""
    n = len(confidence)
    if n < 2 * beta:
        return None
    order = sorted(range(n), key=lambda i: confidence[i])
    conf = [confidence[i] for i in order]
    corr = [correct[i] for i in order]
    bins = [[i * beta, (i + 1) * beta] for i in range(n // beta)]
    bins[-1] = [bins[-1][0], n]
    cerr = 0.0
    for lo, hi in bins[:-1]:   # the official code skips the last bin (sic)
        k = hi - lo
        diff = abs(sum(conf[lo:hi]) / k - sum(corr[lo:hi]) / k)
        cerr += k / n * diff ** 2
    return math.sqrt(cerr)


def pass_at_1(rows):
    """Mean over items of each item's mean score over its scored repeats, on 0-100, with a 95% CI half-width
    from the item means (equals HLE's Wald interval with one repeat)."""
    per_item = {}
    for r in rows:
        if r['score'] is not None:
            per_item.setdefault(r['item_id'], []).append(r['score'])
    means = [sum(v) / len(v) for v in per_item.values()]
    n = len(means)
    if not n:
        return {'score': None, 'ci95': None, 'n': 0}
    mu = sum(means) / n
    var = sum((m - mu) ** 2 for m in means) / n
    return {'score': round(100 * mu, 2), 'ci95': round(100 * 1.96 * math.sqrt(var / n), 2), 'n': n}


def by(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {k: pass_at_1(v) for k, v in sorted(groups.items())}


def model_error_kind(row):
    """Why a model_error scored 0 (all of them are the model's outcome, never infra):
    cutoff              unfinished at the run's no-tails cutoff
    length              hit max_tokens or the served context with no final answer (runaway generation)
    reasoning_unclosed  the model ended its turn inside the reasoning block (EOS before </think>)
    empty_stop          finished normally (stop) with an empty final answer
    empty_<finish>      any other finish_reason with no final answer (labelled with that reason)"""
    if row.get('status') != 'model_error':
        return None
    finish = row.get('finish_reason')
    if finish == 'cutoff':
        return 'cutoff'
    if finish == 'length':
        return 'length'
    if row.get('reasoning_unclosed'):
        return 'reasoning_unclosed'
    return 'empty_stop' if finish == 'stop' else f'empty_{finish}'


def summarize(rows, *, n_items, n_repeats, dataset, aa_slug=None, extra=None):
    scored = [r for r in rows if r['score'] is not None]
    infra = [r for r in rows if r['status'] == 'infra_error']
    expected = n_items * n_repeats
    missing = expected - len(rows)        # rows never attempted count as infra failures too
    n_infra = len(infra) + max(0, missing)
    n_pending = sum(r['status'] == 'pending' for r in rows)
    head = pass_at_1(rows)
    judged = [r for r in scored if r.get('judge_confidence') is not None]
    cal = calib_err([r['judge_confidence'] / 100 for r in judged], [r['score'] for r in judged])
    completion = [r['completion_tokens'] for r in rows if r.get('completion_tokens') is not None]
    ref = aa_reference(aa_slug, dataset)
    out = {
        'benchmark': 'HLE-Diamond' if dataset == 'cais/hle-diamond' else 'HLE',
        'metric': 'pass@1',
        'score': head['score'],
        'ci95': head['ci95'],
        'n_items': n_items,
        'n_repeats': n_repeats,
        'n_scored_items': head['n'],
        'n_scored_rows': len(scored),
        'n_infra_errors': n_infra,
        'infra_error_rate': round(n_infra / expected, 4) if expected else None,
        'valid': bool(expected) and not n_pending and n_infra / expected <= MAX_INFRA_RATE,
        'n_pending': n_pending, 'complete': not n_pending and not missing,
        'generation_complete': not missing and not any(r.get('error_stage') == 'generation' for r in rows),
        'n_model_errors': sum(r['status'] == 'model_error' for r in rows),
        'aa_reference': ref,
        'aa_reference_model': aa_slug,
        'aa_reference_key': AA_KEYS.get(dataset),
        'delta_vs_aa': None if ref is None or head['score'] is None else round(head['score'] - ref, 2),
        'by_partition': by(rows, 'partition'),
        'by_modality': by([{**r, 'modality': 'image' if r.get('has_image') else 'text'} for r in rows], 'modality'),
        'by_answer_type': by(rows, 'answer_type'),
        'by_category': by(rows, 'category'),
        'calibration_error': None if cal is None else round(100 * cal, 2),
        'completion_tokens': {'mean': round(sum(completion) / len(completion)) if completion else None,
                              'max': max(completion) if completion else None, 'total': sum(completion)},
        'finish_reason': _count(rows, 'finish_reason'),
        'status': _count(rows, 'status'),
        'model_errors_by_kind': _count([{'k': model_error_kind(r)} for r in rows if r['status'] == 'model_error'], 'k'),
    }
    out.update(extra or {})
    return out


def _count(rows, key):
    out = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return dict(sorted(out.items()))

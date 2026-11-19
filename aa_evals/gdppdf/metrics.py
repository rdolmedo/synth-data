"""All-pass (headline), Mean Pass, domain cuts and a task-clustered 95% CI, as AA reports GDP.pdf.

AA: All-pass is "the share of all 500 attempts where every criterion passes"; Mean Pass is "each attempt's
criterion pass rate, averaged with equal weight across tasks and repeats"; domain cuts use task-macro Mean
Pass only. Errors and missing attempts score zero (our model_error); infra errors are excluded and counted.
"""
import math
from collections import defaultdict


def _task_means(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r['item_id']].append(r[key])
    return {t: sum(v) / len(v) for t, v in by.items()}


def ci95(task_means):
    """Half-width of a normal 95% CI over task means (attempts clustered by task)."""
    v = list(task_means.values())
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return 1.96 * sd / math.sqrt(n)


def summarize(items, n_tasks_expected, n_repeats):
    """`items`: rows with item_id, domain, status, score (all-pass 0/1), mean_pass (0..1)."""
    scored = [r for r in items if r['status'] in ('ok', 'model_error', 'cutoff', 'ungraded')]   # deadline rows = 0
    infra = [r for r in items if r['status'] == 'infra_error']
    expected = n_tasks_expected * n_repeats
    out = {'n_attempts_expected': expected, 'n_attempts': len(items), 'n_scored': len(scored),
           'n_missing': expected - len(items), 'n_infra_errors': len(infra),
           'n_infra': len(infra),
           'infra_error_rate': round(len(infra) / expected, 4) if expected else None,
           'n_model_errors': sum(r['status'] == 'model_error' for r in items),
           'n_cutoff': sum(r['status'] == 'cutoff' for r in items),
           'n_ungraded': sum(r['status'] == 'ungraded' for r in items),
           'n_pending_judgments': sum(r['status'] == 'pending' for r in items)}
    out['generation_complete'] = out['n_missing'] == 0
    out['complete'] = out['generation_complete'] and out['n_pending_judgments'] == 0
    out['valid'] = (out['complete'] and out['infra_error_rate'] is not None
                    and out['infra_error_rate'] <= 0.02)
    if not scored:
        return dict(out, score=None, mean_pass=None, ci95=None)
    ap = _task_means(scored, 'score')
    mp = _task_means(scored, 'mean_pass')
    out['score'] = round(100 * sum(r['score'] for r in scored) / len(scored), 2)
    out['ci95'] = None if ci95(ap) is None else round(100 * ci95(ap), 2)
    out['score_task_macro'] = round(100 * sum(ap.values()) / len(ap), 2)
    out['mean_pass'] = round(100 * sum(mp.values()) / len(mp), 2)
    out['mean_pass_ci95'] = None if ci95(mp) is None else round(100 * ci95(mp), 2)
    dom = {}
    for r in scored:
        dom[r['item_id']] = r['domain']
    by_dom = defaultdict(list)
    for t, v in mp.items():
        by_dom[dom[t]].append(v)
    out['mean_pass_by_domain'] = {d: {'mean_pass': round(100 * sum(v) / len(v), 2), 'n_tasks': len(v)}
                                  for d, v in sorted(by_dom.items())}
    out['n_tasks_scored'] = len(ap)
    if out['n_pending_judgments']:
        out.update(score=None, mean_pass=None, ci95=None, mean_pass_ci95=None, score_task_macro=None)
    return out

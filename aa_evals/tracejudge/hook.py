"""The summary hook every benchmark's summary code calls (M3.2 `summary.json` contract: `n_flagged`).

    from aa_evals.tracejudge import fold_summary
    summary = fold_summary(run_dir, summary)      # before writing summary.json

It reads `<run_dir>/flags.jsonl` (written by `python -m aa_evals.tracejudge judge|watch`) and adds:
  n_flagged             counted attempts the judge flagged as infra that the pipeline scored as model outcomes:
                        the ones to look at (each is a suspected silent infra failure);
  n_flagged_by_category the same, by category;
  tracejudge            {n_judged, n_attempts_finished?, n_flagged_known (agrees with n_infra / n_cutoff),
                         n_llm_errors, judge, prompt_version, flags: "flags.jsonl"}.
No flags.jsonl: n_flagged is None and tracejudge.status is "not run". It never raises on a bad row.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

THRESHOLD = 0.5
FLAGS = 'flags.jsonl'
_WARNED: set = set()


def _warn_once(path, line: int, e: Exception) -> None:
    import sys
    if (str(path), line) not in _WARNED:
        _WARNED.add((str(path), line))
        print(f'jsonl: {path}:{line}: undecodable line skipped ({e})', file=sys.stderr, flush=True)


def read_flags(run_dir, path=None) -> dict:
    """key -> last row of flags.jsonl (a re-judged attempt appends a newer row)."""
    p = Path(path) if path else Path(run_dir) / FLAGS
    rows = {}
    try:
        with open(p, errors='replace') as f:
            lines = f.read().split('\n')   # '\n' only (aa_evals.common.jsonl)
        for i, line in enumerate(lines, 1):
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError as e:
                    if i < len(lines):   # not the unterminated last line (a row being written): say so
                        _warn_once(p, i, e)
                    continue
                if isinstance(r, dict) and r.get('key'):
                    rows[r['key']] = r
    except OSError:
        return {}
    return rows


def counts(rows: dict, threshold: float = THRESHOLD) -> dict:
    counted = [r for r in rows.values() if r.get('counted', True)]
    new = [r for r in counted if r.get('flagged') and not r.get('known_infra')]
    known = [r for r in counted if r.get('flagged') and r.get('known_infra')]
    by_cat = Counter()
    for r in new:
        for c in sorted({f['category'] for f in r.get('flags') or [] if f.get('confidence', 0) >= threshold}):
            by_cat[c] += 1
    judges = Counter(json.dumps(r.get('judge'), sort_keys=True) for r in counted if r.get('judge'))
    limited, egress = Counter(), Counter()
    for r in counted:
        for h, n in (r.get('remote_limited') or {}).items():
            limited[h] += n
        for h, n in (r.get('egress_blocked') or {}).items():
            egress[h] += n
    return {
        'n_offline_proxy_review_required': sum(bool((r.get('recorded') or {}).get('offline_proxy_review_required')) for r in counted),
        'n_offline_proxy_review_pending': sum(bool((r.get('recorded') or {}).get('offline_proxy_review_required'))
                                               and (r.get('recorded') or {}).get('offline_proxy_review_pending') is not False
                                               for r in counted),
        'n_flagged': len(new),
        'n_flagged_by_category': dict(sorted(by_cat.items())),
        'tracejudge': {
            'n_judged': len(counted), 'n_judged_physical': len(rows),
            'n_flagged_known': len(known),
            # attempts where a remote site refused the model's scraping (429/403/503 from the site, anti-bot pages):
            # the same for every model (no web_search tool), not infra
            'n_remote_site_limited': sum(1 for r in counted if r.get('remote_limited')),
            'remote_limited_hosts': dict(limited.most_common(10)),
            # attempts where the proxy refused CONNECT to one host while others worked in the same call: our egress
            # being blocked or that upstream down (an environment property, the same for every model), not flagged
            'n_egress_blocked': sum(1 for r in counted if r.get('egress_blocked')),
            'egress_blocked_hosts': dict(egress.most_common(10)),
            'n_llm_calls': sum(1 for r in counted if r.get('llm')),
            'n_llm_errors': sum(1 for r in counted if r.get('llm') and not r['llm'].get('ok')),
            'flagged_keys': sorted(r['key'] for r in new)[:200],
            'judge': json.loads(judges.most_common(1)[0][0]) if judges else None,
            'threshold': threshold, 'flags': FLAGS,
        },
    }


def _native_proxy_reviews(run_dir, trace_rows):
    """Discover native markers first; missing prior required sources never certify pending0."""
    import re
    from aa_evals.tracejudge.layouts import _offline_proxy_review
    run_dir = Path(run_dir)
    pattern = re.compile(r'(.+__a(?:0|[1-9]\d*))__r(0|[1-9]\d*)')
    try:
        config = json.loads((run_dir / 'config.json').read_text())
    except (OSError, ValueError):
        config = {}
    groups, outcomes, unreadable = {}, {}, set()
    for path in (run_dir / 'attempts').glob('*/outcome.json'):
        key = path.parent.name
        match = pattern.fullmatch(key)
        if not match:
            continue
        try:
            outcome = json.loads(path.read_text())
            if not isinstance(outcome, dict):
                raise ValueError('native outcome is not an object')
        except (OSError, ValueError):
            unreadable.add(key)
            continue
        outcomes[key] = outcome
        groups.setdefault(match[1], []).append((int(match[2]), key, outcome))
    selected = {}
    for base, group in groups.items():
        group.sort()
        selected[base] = next((key for _, key, outcome in group if not outcome.get('infrastructure_failure')), group[-1][1])
    # These are integrity anchors only; flags never override native retry selection.
    known = {key for key, row in trace_rows.items() if isinstance(row.get('recorded'), dict)
             and row['recorded'].get('offline_proxy_review_required') is True}
    known.update(path.parent.name for path in (run_dir / 'attempts').glob('*/offline-proxy-review.json'))
    required, pending, physical_required, physical_pending, reviewed = set(), set(), set(), set(), {}
    for key in set(outcomes) | known:
        match = pattern.fullmatch(key)
        if not match:
            continue
        outcome = outcomes.get(key)
        actual = isinstance(outcome, dict) and outcome.get('offline_proxy_review_required') is True
        if not actual and key not in known:
            continue
        if actual:
            result = _offline_proxy_review(run_dir, key, outcome, config)
        else:
            result = {'offline_proxy_review_required': True, 'offline_proxy_review_pending': True,
                      'offline_proxy_review_error': 'previously marked native source is missing or unmarked'}
        reviewed[key] = result
        unresolved = result.get('offline_proxy_review_pending') is not False
        physical_required.add(key)
        if unresolved:
            physical_pending.add(key)
        choice = selected.get(match[1])
        counted = choice is None or choice == key
        if not actual and choice is not None and choice != key:
            # A concrete native infra prefix or earlier noninfra outcome still
            # makes this key audit-only. A missing earlier noninfra source cannot
            # be silently replaced by a later retry just because a flag says so.
            chosen_retry = int(pattern.fullmatch(choice)[2])
            counted = (chosen_retry > int(match[2]) and not (outcome or {}).get('infrastructure_failure'))
        if counted:
            required.add(key)
            if unresolved:
                pending.add(key)
    # Unknown completed native bodies also hold audit completeness, without
    # double-counting a known marker whose body became unreadable.
    pending.update(unreadable)
    audit = {'n_offline_proxy_review_required': len(required),
             'n_offline_proxy_review_pending': len(pending),
             'offline_proxy_review_pending_keys': sorted(pending),
             'n_offline_proxy_review_required_physical': len(physical_required),
             'n_offline_proxy_review_pending_physical': len(physical_pending),
             'offline_proxy_review_pending_physical_keys': sorted(physical_pending),
             'n_offline_proxy_review_unreadable_outcomes': len(unreadable)}
    return audit, reviewed


def fold_summary(run_dir, summary: dict | None = None, *, path=None, threshold: float = THRESHOLD) -> dict:
    """summary + n_flagged / n_flagged_by_category / tracejudge (a new dict; the input is not changed)."""
    out = dict(summary or {})
    flag_error = None
    try:
        rows = read_flags(run_dir, path)
    except Exception as exc:   # native review discovery still runs for unreadable flags
        rows, flag_error = {}, exc
    audit, reviewed = _native_proxy_reviews(run_dir, rows)
    out.update(audit)
    if flag_error is not None:
        out.update(n_flagged=None, tracejudge={'status': f'unreadable: {flag_error!r}'})
        return out
    if not rows:
        out.setdefault('n_flagged', None)
        out.setdefault('tracejudge', {'status': 'not run'})
        return out
    # Keep per-row details current where a trace exists; authoritative pending
    # counts come from native outcomes even when flags are absent or malformed.
    for key, result in reviewed.items():
        if key in rows:
            record = rows[key].get('recorded')
            rows[key]['recorded'] = {**(record if isinstance(record, dict) else {}), **result}
    try:
        out.update(counts(rows, threshold))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        out.update(n_flagged=None, tracejudge={'status': f'unreadable flag body: {exc!r}'})
    out.update(audit)
    return out


def fold_file(run_dir, *, path=None, threshold: float = THRESHOLD) -> dict | None:
    """Fold into <run_dir>/summary.json in place (atomic). None when there is no summary.json yet."""
    sp = Path(run_dir) / 'summary.json'
    try:
        summary = json.loads(sp.read_text())
    except (OSError, ValueError):
        return None
    new = fold_summary(run_dir, summary, path=path, threshold=threshold)
    tmp = sp.with_name(f'.summary.json.tj.{os.getpid()}')
    tmp.write_text(json.dumps(new, indent=1))
    tmp.replace(sp)
    return new

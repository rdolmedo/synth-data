"""Per-step view of a finished run, for the fix pass: pass rate per step across repeats, failure categories, and
the steps no repeat passes (candidates for broken tests or targets, to be read by hand or by an agent).

    python -m aa_evals.scicode.analyze --run-id <id> [--run-id <id2> ...] [--out steps.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from .run import RUNS, final_rows

CATS = [('timeout', r'TIMEOUT|outer timeout'), ('assert', r'AssertionError'), ('name', r'NameError'),
        ('type', r'TypeError'), ('value', r'ValueError'), ('index', r'IndexError|KeyError'),
        ('shape', r'broadcast|shape'), ('linalg', r'LinAlgError'), ('import', r'ImportError|ModuleNotFound'),
        ('syntax', r'SyntaxError|IndentationError'), ('memory', r'MemoryError|Killed'),
        ('attr', r'AttributeError'), ('zerodiv', r'ZeroDivision')]


def category(row) -> str:
    if row['status'] == 'infra_error':
        return 'infra'
    if row.get('score'):
        return 'pass'
    if row.get('cutoff'):
        return 'cutoff'
    if row.get('empty_content'):
        return 'empty'
    if row.get('timeouts'):
        return 'timeout'
    tail = row.get('fail_tail') or ''
    for name, pat in CATS:
        if re.search(pat, tail):
            return name
    return 'other'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', action='append', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    steps = defaultdict(lambda: {'n': 0, 'pass': 0, 'cats': Counter(), 'env': Counter()})
    for rid in a.run_id:
        for (sid, r), row in final_rows(RUNS / rid).items():
            st = steps[sid]
            if row['status'] == 'infra_error':
                st['cats']['infra'] += 1
                continue
            st['n'] += 1
            st['pass'] += row['score'] or 0
            st['cats'][category(row)] += 1
            for env, ok in (row.get('per_env') or {}).items():
                st['env'][env] += bool(ok)
    key = lambda s: tuple(int(x) for x in s.split('.'))
    total = Counter()
    for st in steps.values():
        total.update(st['cats'])
    print('failure categories over all step attempts:', dict(total.most_common()))
    never = [s for s in sorted(steps, key=key) if steps[s]['n'] and steps[s]['pass'] == 0]
    always = [s for s in sorted(steps, key=key) if steps[s]['n'] and steps[s]['pass'] == steps[s]['n']]
    env_split = [s for s in sorted(steps, key=key) if len(set(steps[s]['env'].values())) > 1]
    print(f'{len(steps)} steps; never pass: {len(never)}; always pass: {len(always)}')
    print('never pass:', ' '.join(f"{s}({','.join(f'{k}{v}' for k, v in steps[s]['cats'].most_common(2))})" for s in never))
    print('env-dependent (passes in one env more than the other):',
          ' '.join(f"{s}{dict(steps[s]['env'])}" for s in env_split))
    if a.out:
        json.dump({s: {'n': v['n'], 'pass': v['pass'], 'cats': dict(v['cats']), 'env': dict(v['env'])}
                   for s, v in steps.items()}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

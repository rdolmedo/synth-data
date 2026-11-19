"""Drift report: which pipeline branches are behind the shared layer.

  python -m tbshared.drift [--fetch] [--json]

For every pipeline branch (PIPELINES), reads origin/<branch>:tbshared/PIN and counts the commits of origin/shared
that touch tbshared/ after it. Once the trunk `main` holds the canonical copy (tbshared/TRUNK, #480), every row also
lists the tbshared/ files where that copy differs from main's (`differs_from_main`; pipeline-local files excluded):
a fix made on main shows there until the branch has it. `--fetch` first runs `git fetch origin` for these branches.
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

PIPELINES = ('aa-evals', 'aa-serving', 'eval', 'task-synthesis', 'data-synthesis', 'glm-data-synthesis', 'sft')
SHARED = 'origin/shared'
MAIN = 'origin/main'
LOCAL = {'PIN', 'ALIASES', 'pipeline.json', 'TRUNK'}   # sync.LOCAL: pipeline-local, never compared


def git(*args) -> tuple[int, str]:
    r = subprocess.run(['git', *args], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout.strip()


def differs_from_main(rev: str) -> list[str] | None:
    """tbshared/ files where `rev`'s copy differs from the trunk's (pipeline-local files excluded); None without a
    trunk copy."""
    if git('cat-file', '-e', f'{MAIN}:tbshared/TRUNK')[0]:
        return None
    rc, out = git('diff', '--name-only', rev, MAIN, '--', 'tbshared/')
    if rc:
        return None
    return [f for f in out.splitlines() if f.split('/', 1)[-1] not in LOCAL]


def report(pipelines=PIPELINES) -> list[dict]:
    rows = []
    trunk = differs_from_main(SHARED)
    if trunk is not None:   # #480: main holds the canonical copy; shared no longer gets every change
        rows.append({'branch': 'main', 'pin': None, 'behind': None, 'state': 'trunk (canonical)',
                     'shared_differs_from_main': trunk})
    for b in pipelines:
        rc, pin = git('show', f'origin/{b}:tbshared/PIN')
        if rc:
            rows.append({'branch': b, 'pin': None, 'behind': None, 'state': 'not migrated'})
            continue
        if subprocess.run(['git', 'merge-base', '--is-ancestor', pin, SHARED], capture_output=True).returncode:
            # not in this repository, or never merged into shared (pinned from a PR branch, or shared rewritten)
            rows.append({'branch': b, 'pin': pin[:12], 'behind': None, 'state': 'pin not on shared'})
            continue
        rc, n = git('rev-list', '--count', f'{pin}..{SHARED}', '--', 'tbshared/')
        _, log = git('log', '--format=%h %s', f'{pin}..{SHARED}', '--', 'tbshared/')
        rows.append({'branch': b, 'pin': pin[:12], 'behind': int(n), 'state': 'current' if n == '0' else 'behind',
                     'missing': log.splitlines()})
    for r in rows:
        if r['pin'] and trunk is not None:
            r['differs_from_main'] = differs_from_main(f"origin/{r['branch']}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.fetch:
        for b in ('shared', 'main', *PIPELINES):
            rc, _ = git('fetch', '-q', 'origin', b)
            if rc:
                print(f'WARNING: git fetch origin {b} failed: its row may be stale', file=sys.stderr)
    rows = report()
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    _, head = git('rev-parse', '--short=12', SHARED)
    print(f'shared head {head}')
    for r in rows:
        print(f"{r['branch']:<20} {r['state']:<22} pin {r['pin'] or '-':<12} behind {r['behind'] if r['behind'] is not None else '-'}")
        for m in r.get('missing') or []:
            print(f'    missing {m}')
        for f in r.get('differs_from_main') or []:
            print(f'    differs from main: {f}')
        for f in r.get('shared_differs_from_main') or []:
            print(f'    shared differs from main: {f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

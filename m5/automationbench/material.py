"""What the generator reads about the apps it writes data for (#359: the generator reads the material it writes
about): each app's state schema (upstream's pydantic source, the exact input the simulator validates), the REST
endpoints the agent can call on it, and format examples from the 200 unscored simple tasks.

Runs in the AutomationBench venv.
"""
from __future__ import annotations

import functools
import json
import pprint
import random
import re
from pathlib import Path

from m5.automationbench.upstream import load

import automationbench  # noqa: E402

PKG = Path(automationbench.__file__).parent   # the installed package (upstream 4a8e106)
SCHEMA_DIR = PKG / 'schema'
API_SCHEMAS = PKG / 'tools' / 'api' / 'schemas'
LICENSE = re.compile(r'^# Copyright 2026 Zapier, Inc\.\n# SPDX-License-Identifier: MIT\n+', re.M)


@functools.lru_cache(None)
def schema_source(app: str) -> str:
    """The app's schema module(s), license header stripped (one file, or a package's files in import order)."""
    f = SCHEMA_DIR / f'{app}.py'
    files = [f] if f.exists() else sorted((SCHEMA_DIR / app).glob('*.py'), key=lambda p: (p.name != 'base.py', p.name == '__init__.py', p.name))
    out = []
    for p in files:
        text = LICENSE.sub('', p.read_text()).strip()
        if p.name == '__init__.py' and len(text) < 2000 and 'import' in text and 'class ' not in text:
            continue
        out.append(f'# --- automationbench/schema/{p.relative_to(SCHEMA_DIR)}\n{text}\n')
    return '\n'.join(out)


@functools.lru_cache(None)
def _api(app: str) -> dict | None:
    f = API_SCHEMAS / f'{app}.jsonc'
    if not f.exists():
        return None
    return json.loads('\n'.join(x for x in f.read_text().splitlines() if not x.lstrip().startswith('//')))


@functools.lru_cache(None)
def endpoints_text(app: str) -> str:
    """One line per endpoint the agent can reach: METHOD URL - description (required params)."""
    from automationbench.tools.api.search import _compute_url
    d = _api(app)
    if d is None:
        return '(no REST endpoints)'
    lines = []
    for e in d.get('endpoints', []):
        url = _compute_url(d['api'], d['baseUrl'], e['path'])
        req = [k for k, v in (e.get('parameters') or {}).items() if isinstance(v, dict) and v.get('required')]
        desc = (e.get('description') or '').split('. ')[0].strip().rstrip('.')
        lines.append(f'{e["method"]} {url} - {desc}' + (f' (required: {", ".join(req)})' if req else ''))
    return '\n'.join(lines)


def get_endpoints(app: str) -> list[str]:
    """Full URLs of the app's GET endpoints that need no path or required parameters (the problem-side read check)."""
    from automationbench.tools.api.search import _compute_url
    d = _api(app)
    out = []
    for e in (d or {}).get('endpoints', []):
        req = [k for k, v in (e.get('parameters') or {}).items() if isinstance(v, dict) and v.get('required')]
        if e['method'] == 'GET' and '{' not in e['path'] and not req:
            out.append(_compute_url(d['api'], d['baseUrl'], e['path']))
    return out


@functools.lru_cache(None)
def simple_tasks() -> list[dict]:
    """The 200 unscored simple tasks: {'apps', 'trigger', 'state'} (assertions and tool lists dropped)."""
    out = []
    for r in load('simple', noise=False):
        st = r['info']['initial_state']
        out.append({'name': r['task_name'], 'apps': sorted(k for k in st if k != 'meta'), 'trigger': r['trigger'],
                    'state': st})
    return out


def example_code(t: dict) -> str:
    state = pprint.pformat(t['state'], width=118, sort_dicts=False)
    return (f'def build_task() -> dict:\n    trigger = {t["trigger"]!r}\n    initial_state = '
            + state.replace('\n', '\n    ') + '\n    return {"trigger": trigger, "initial_state": initial_state}\n')


def pick_examples(apps: list[str], rng: random.Random, k: int = 3) -> list[dict]:
    """k simple tasks, preferring ones that show the spec's apps (each new app covered first), else random."""
    pool = [t for t in simple_tasks() if t['apps']]
    chosen, covered = [], set()
    for _ in range(k):
        best = max(pool, key=lambda t: (len((set(t['apps']) & set(apps)) - covered), rng.random()))
        if not (set(best['apps']) & set(apps)) - covered:
            best = rng.choice(pool)
        chosen.append(best)
        covered |= set(best['apps'])
        pool = [t for t in pool if t is not best]
    return chosen

"""Run a generated `build_task()` and the problem-side checks (#359).

- exec: the generator's code in a separate `python -I` process (CPU and memory limits, empty working dir, 30 s);
- assemble: upstream's task-file shape (example_id, task, prompt = domain system prompt + trigger, info with
  zapier_tools and initial_state), the trigger's fitted closing sentence, then the domain's own `_noise.py`;
- checks (hard): the apps are the spec's; the state loads into the simulator (`WorldState`, with
  `compute_allowed_services` as the world service does) and survives its JSON round trip; IDs are unique per
  collection; every inferable reference resolves (refs.py); every Slack `#channel` the trigger names exists; the
  app's parameter-free GET endpoints answer without raising;
- warnings (soft, recorded): size and trigger length far from the spec, quoted names in the trigger that appear
  nowhere in the state.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
code = sys.stdin.read()
ns = {"__name__": "generated_task"}
exec(compile(code, "build_task.py", "exec"), ns)
out = ns["build_task"]()
sys.stdout.write(json.dumps(out, default=str))
'''
SKIP = None


def run_code(code: str, timeout: int = 40) -> tuple[dict | None, str | None]:
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, '-I', '-c', RUNNER], input=code, capture_output=True, text=True,
                               timeout=timeout, cwd=d, env={'PATH': '/usr/bin:/bin', 'HOME': d})
        except subprocess.TimeoutExpired:
            return None, 'build_task() did not finish within 30 s'
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        return None, 'build_task() raised: ' + ' | '.join(err[-4:])[:1500]
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return None, 'build_task() did not return JSON-serializable data'
    if not isinstance(out, dict) or not isinstance(out.get('trigger'), str) or not isinstance(out.get('initial_state'), dict):
        return None, 'build_task() must return {"trigger": str, "initial_state": dict}'
    return out, None


def count_records(state: dict) -> int:
    n = 0
    for app, v in state.items():
        if app == 'meta' or not isinstance(v, dict):
            continue
        for coll, x in v.items():
            if isinstance(x, list):
                n += len(x)
                if app == 'google_sheets' and coll == 'spreadsheets':
                    for ss in x:
                        if isinstance(ss, dict):
                            for ws in ss.get('worksheets', []) or []:
                                if isinstance(ws, dict):
                                    n += len(ws.get('rows', []) or [])
    return n


def assemble(spec: dict, out: dict) -> dict:
    """Upstream's task-file shape; noise applied by the domain's own _noise.py (seeded by example_id)."""
    from m5.automationbench.upstream import apply_noise, system_prompt
    state = copy.deepcopy(out['initial_state'])
    state['meta'] = {**(state.get('meta') or {}), 'schema_version': '0.1.0', 'current_time': spec['now']}
    trigger = out['trigger'].strip()
    if spec.get('suffix'):
        trigger = trigger + ' ' + spec['suffix']
    task = {'example_id': spec['example_id'], 'task': f'{spec["domain"]}.{spec["slug"]}',
            'prompt': [{'role': 'system', 'content': system_prompt(spec['domain'])},
                       {'role': 'user', 'content': trigger}],
            'answer': '',
            'info': {'zapier_tools': list(spec.get('zapier_tools', [])), 'initial_state': state}}
    pre = {'state_bytes': len(json.dumps({k: v for k, v in state.items() if k != 'meta'})),
           'n_records': count_records(state), 'trigger_words': len(out['trigger'].split())}
    task = apply_noise(spec['domain'], task)
    task['info']['task_name'] = task.pop('task')
    task['info']['m5'] = {'spec_id': spec['id'], 'pre_noise': pre}
    return task


def _unique_ids(dump: dict) -> list[str]:
    bad = []
    for app, st in dump.items():
        if app == 'meta' or not isinstance(st, dict):
            continue
        for coll, recs in st.items():
            if isinstance(recs, list):
                seen = {}
                for r in recs:
                    if isinstance(r, dict) and r.get('id') not in (None, ''):
                        key = (r.get('spreadsheet_id'), r.get('worksheet_id'), r['id']) if app == 'google_sheets' else r['id']
                        if key in seen:
                            bad.append(f'{app}.{coll}: duplicate id {r["id"]!r}')
                        seen[key] = 1
    return bad


def checks(spec: dict, out: dict, task: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) for an assembled task."""
    global SKIP
    from automationbench.runner import compute_allowed_services, strip_none_values
    from automationbench.schema.world import WorldState
    from automationbench.tools.api import api_fetch
    from m5.automationbench import material, refs
    if SKIP is None:
        SKIP = refs.load_skip(HERE / 'fk_skip.json')
    errors, warns = [], []
    apps = sorted(k for k in out['initial_state'] if k != 'meta')
    if set(apps) != set(spec['apps']):
        errors.append(f'initial_state apps are {apps}; the spec asks for exactly {sorted(spec["apps"])}')
    for a in apps:
        if a not in WorldState.model_fields:
            errors.append(f'{a!r} is not an app of the simulator')
    if errors:
        return errors, warns
    info = task['info']
    initial = strip_none_values(info['initial_state'])
    try:
        world = WorldState(**initial)
        world.meta.allowed_services = compute_allowed_services(initial, [], info.get('zapier_tools', []))
        world = WorldState.model_validate_json(world.model_dump_json())
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        return [f'the state does not load into the simulator: {msg[:3000]}'], warns
    dump = world.model_dump(mode='json')
    errors += _unique_ids(dump)
    try:   # references: the generator's own data (upstream's noise leaves some of its own dangling)
        own = strip_none_values(copy.deepcopy(out['initial_state']))
        own['meta'] = info['initial_state']['meta']
        own_dump = WorldState(**own).model_dump(mode='json')
        errors += [f'dangling reference: {x}' for x in refs.check(own_dump, SKIP)][:20]
        # Sheets: row 1 is the header, data rows are 2, 3, ... (upstream: 924 of 924 worksheets start at 2; pilot
        # traces read an off-by-one when ours started at 1)
        by_ws = {}
        for r in (own_dump.get('google_sheets') or {}).get('rows', []):
            by_ws.setdefault((r.get('spreadsheet_id'), r.get('worksheet_id')), []).append(r.get('row_id'))
        for (ss, ws), ids in by_ws.items():
            if not all(isinstance(x, int) for x in ids):
                continue   # string row ids ('r1'): upstream uses them too, no row-number semantics
            nums = list(ids)
            if nums != list(range(2, 2 + len(nums))):
                errors.append(f'google_sheets {ss}/{ws}: data rows must have row_id 2, 3, 4, ... in order (row 1 is '
                              f'the header); found {nums[:8]}{"..." if len(nums) > 8 else ""}')
    except Exception as e:  # noqa: BLE001
        errors.append(f'the state (before background records) does not load into the simulator: {str(e)[:2000]}')
    trig = out['trigger']
    if 'slack' in apps:
        names = {c.get('name') for c in dump['slack'].get('channels', [])}
        for ch in set(re.findall(r'(?<![\w/])#([a-z0-9][a-z0-9_-]+)', trig)):
            if ch not in names:
                errors.append(f'the request names Slack channel #{ch}, which does not exist (channels: {sorted(n for n in names if n)})')
    blob_all = json.dumps(dump)
    action_log = [a for a in apps if 'actions' in WorldState.model_fields[a].annotation.model_fields]
    for tok in [] if action_log else set(re.findall(r'\b[A-Za-z]{1,6}_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b|\b(?:ss|ws|msg|tkt|opp|acc)_[A-Za-z0-9_]+\b', trig)):
        if tok not in blob_all:
            errors.append(f'the request mentions the ID {tok!r}, which does not exist in the state')
    for a in apps:
        for url in material.get_endpoints(a):
            try:
                api_fetch(copy.deepcopy(world), 'GET', url)
            except Exception as e:  # noqa: BLE001
                errors.append(f'reading {url} on this state raises {type(e).__name__}: {str(e)[:300]}')
    pre = info['m5']['pre_noise']
    if not 0.6 * spec['records'] <= pre['n_records'] <= 1.6 * spec['records']:
        warns.append(f'records {pre["n_records"]} vs spec {spec["records"]}')
    if not 0.5 * spec['trigger_words'] <= pre['trigger_words'] <= 1.8 * spec['trigger_words']:
        warns.append(f'trigger words {pre["trigger_words"]} vs spec {spec["trigger_words"]}')
    blob = json.dumps(initial).lower()
    for q in re.findall(r"['\"‘“]([A-Z][^'\"’”]{3,60})['\"’”]", trig):
        if q.lower() not in blob:
            warns.append(f'quoted name not in state: {q!r}')
    return errors, warns

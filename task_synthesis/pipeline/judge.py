"""LLM audit of one checked task: rubric scores, flags, issues. The pass rule is recomputed here from the scores,
never taken from the judge's own verdict field."""
import json
import re

from .prompts import JUDGE_ASSET_USER, JUDGE_USER, judge_system
from .specs import spec_text

CORE = ('clarity', 'consistency', 'solvability')
SOFT = ('difficulty', 'realism', 'originality')
BLOCKING = ('leaks_solution', 'needs_internet', 'needs_gpu', 'unsafe')
FILE_CHARS, TOTAL_CHARS = 6000, 40000


def render_files(pkg):
    parts, total = [], 0
    for path, body in pkg['files'].items():
        text = body if len(body) <= FILE_CHARS else body[:FILE_CHARS] + f'\n[... {len(body) - FILE_CHARS} more characters]'
        if total + len(text) > TOTAL_CHARS:
            parts.append(f'## {path} ({len(body)} characters, not shown)')
            continue
        total += len(text)
        parts.append(f'## {path}\n{text}')
    return '\n\n'.join(parts) or '(none)'


def render_snapshot(check):
    s = check.get('snapshot', {})
    out = ['## Directory tree (bytes, path)', '\n'.join(s.get('tree', [])) or '(empty)']
    if s.get('heads'):
        out.append('## Heads of the input files')
        for p, h in s['heads'].items():
            out.append(f'### {p}\n{h}')
    if 'defect_rc' in s:
        out.append(f'## defect_check exit code: {s["defect_rc"]}')
    if s.get('services_log'):
        out.append(f'## services log\n{s["services_log"]}')
    if check.get('warnings'):
        out.append('## Check warnings\n' + '\n'.join(check['warnings']))
    return '\n\n'.join(out)


def prompt(spec, pkg, check, asset=None):
    """asset: for a task built on a real asset, its provenance and manifest (task_synthesis.assets.render.for_judge)."""
    return [{'role': 'system', 'content': judge_system(bool(asset))},
            {'role': 'user', 'content': (JUDGE_ASSET_USER.format(asset=asset) if asset else '') + JUDGE_USER.format(
                spec=spec_text(spec), instruction=pkg['instruction'], files=render_files(pkg),
                setup=pkg['setup'], services=pkg['services'] or '(none)', snapshot=render_snapshot(check),
                sketch=pkg['sketch'])}]


def parse(text, asset=False):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('judge reply has no JSON object')
    j = json.loads(m.group(0))
    core = CORE + (('asset_use',) if asset else ())
    scores = {k: int(j['scores'][k]) for k in core + SOFT}
    flags = {k: bool(j.get('flags', {}).get(k, False)) for k in BLOCKING + ('spec_mismatch',)}
    passed = (all(scores[k] >= 4 for k in core) and all(scores[k] >= 3 for k in SOFT)
              and not any(flags[k] for k in BLOCKING))
    return {'scores': scores, 'flags': flags, 'estimated_minutes': j.get('estimated_minutes'),
            'issues': [str(x) for x in j.get('issues', [])][:12], 'judge_verdict': j.get('verdict'), 'pass': passed}


async def judge(llm, spec, pkg, check, tries=3, asset=None, extra=None):
    """extra: the request's extra_body instead of the endpoint's (round --judge-effort on asset tasks only)."""
    last = None
    for _ in range(tries):
        content, _, usage = await llm.chat(prompt(spec, pkg, check, asset), extra=extra)
        try:
            return {**parse(content, bool(asset)), 'usage': usage}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            last = f'{e}: {content[-300:]}'
    return {'pass': False, 'error': f'unparseable judge reply: {last}'}

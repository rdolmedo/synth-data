"""Read-only SciCode protocol/source context for the trace judge, never a verdict.

SciCode's generic transcript omits its prompt and earlier generated functions. The
optional raw generation, chain and config restore those facts without running code.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
from pathlib import Path

from .trace import Event, Trace

SOURCE_LIMIT = 8 * 1024 * 1024
RAW_LIMIT = 32 * 1024 * 1024
LINE_LIMIT = 4 * 1024 * 1024
CONTEXT_CHARS = 600
MAX_SYMBOLS = 8


def _json(path):
    try:
        if path.stat().st_size <= SOURCE_LIMIT:
            obj = json.loads(path.read_text())
            return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, UnicodeError, RecursionError):
        pass
    return {}


def _raw(path, step):
    """Last matching record, with bounded decompression even for incomplete files."""
    out, size = {}, 0
    try:
        with gzip.open(path, 'rb') as f:
            while size < RAW_LIMIT:
                line = f.readline(min(LINE_LIMIT, RAW_LIMIT - size) + 1)
                if not line:
                    return out
                size += len(line)
                if len(line) > LINE_LIMIT or size > RAW_LIMIT:
                    return {}  # do not mistake an early matching record for the last one
                try:
                    obj = json.loads(line)
                except (ValueError, UnicodeError, RecursionError):
                    continue
                if isinstance(obj, dict) and obj.get('step') == step:
                    out = obj
    except (OSError, EOFError):
        return {}
    return {}


def enrich(t: Trace, row: dict, config: dict, run_dir: Path) -> Trace:
    m = re.fullmatch(r'(\d+)\.(\d+)/r(\d+)', t.key)
    if t.bench.replace('-', '_') not in ('scicode', 'scicode_verified') or not m:
        return t
    problem, ordinal, repeat = m.groups()
    step = f'{problem}.{ordinal}'
    t.recorded['scicode_context_version'] = 1

    def add(text, ref):
        t.events.append(Event('scicode-context', text[:CONTEXT_CHARS], ref))

    grading = config.get('grading') or {}
    if isinstance(grading, dict) and grading.get('rule'):
        add('Recorded SciCode grading policy: ' + str(grading['rule']), 'config.json:grading.rule')
    chain_ptr = row.get('chain')
    cp = Path(chain_ptr if isinstance(chain_ptr, str) and chain_ptr else f'chains/{problem}.r{repeat}.json')
    if not cp.is_absolute():
        cp = run_dir / cp
    chain = _json(cp)
    steps = chain.get('steps') if isinstance(chain.get('steps'), dict) else {}
    current = steps.get(step) if isinstance(steps.get(step), dict) else {}
    grade = current.get('grade') or {}
    envs = grade.get('per_env') if isinstance(grade, dict) else None
    if isinstance(envs, dict):
        outcomes = {str(k): {f: v.get(f) for f in ('pass', 'rc', 'timeout')}
                    for k, v in list(envs.items())[:8] if isinstance(v, dict)}
        add('Saved combined grade: ' + json.dumps({'pass': grade.get('pass'), 'per_env': outcomes})
            + (' (additional environments omitted)' if len(envs) > 8 else ''), f'{cp.name}:{step}.grade')

    raw_path = run_dir / 'transcripts' / f'{problem}.r{repeat}.gen.jsonl.gz'
    raw = _raw(raw_path, step)
    prompt = raw.get('prompt') if isinstance(raw.get('prompt'), str) else ''
    final = next((x.text for x in reversed(t.turns) if x.role == 'assistant' and x.kind == 'text'), '')
    matches = bool(raw) and (raw.get('content') or '') == final
    if current.get('prompt_sha') and prompt:
        matches = matches and hashlib.sha256(prompt.encode()).hexdigest()[:16] == current['prompt_sha']
    if matches:
        _, marker, deps = prompt.rpartition('\nDEPENDENCIES:\n')
        if marker:
            deps = deps.split('\nRESPONSE GUIDELINES:', 1)[0].strip()
            bounded = deps[:1800]
            for n in range(0, len(bounded), 500):
                add('Exact saved dependency instruction:\n' + bounded[n:n + 500],
                    f'{raw_path.name}:{step}.prompt:dependencies:{n}')
            if len(deps) > 1800:
                add('Dependency instruction truncated after 1800 characters.', f'{raw_path.name}:{step}.prompt')
        if isinstance(current.get('code'), str):
            try:
                from aa_evals.scicode.prompts import extract_code
            except (ImportError, OSError):
                add('Optional official SciCode extractor unavailable in this judge snapshot; '
                    'saved answer/extracted-code match not checked.', f'{cp.name}:{step}.code')
            else:
                same = extract_code(raw.get('content')) == current['code']
                add('Official SciCode extraction protocol (aa_evals.scicode.prompts.extract_code): '
                    'first Python fence (otherwise first fence or whole response); '
                    'import lines removed; the runner prepends supplied problem dependencies. '
                    f'Saved answer/extracted-code pair matches this extractor: {same}. '
                    'This is protocol context, not an infrastructure verdict.', f'{cp.name}:{step}.code')
    else:
        add('Optional raw generation missing, unreadable or unmatched; dependency/extraction context unavailable.',
            f'{raw_path.name}:{step}')

    # A test.py filename alone cannot distinguish injected generated functions from
    # verifier helpers. Report definitions present, not a claim about shadowing.
    tails = '\n'.join(str(v.get('tail') or '')[-8192:] for v in list((envs or {}).values())[:8]
                      if isinstance(v, dict)) if isinstance(envs, dict) else ''
    if not tails:
        tails = '\n'.join(e.text for e in t.events if e.source.startswith('grader:'))
    names = list(dict.fromkeys(re.findall(r'\bin ([A-Za-z_]\w*)\s*\n', tails)))[:MAX_SYMBOLS]
    if not names:
        return t
    origins = {name: [] for name in names}
    incomplete = len(steps) > 100
    for sid, rec in list(steps.items())[:100]:
        if not isinstance(rec, dict) or not re.fullmatch(r'\d+\.\d+', sid):
            continue
        if sid.split('.')[0] != problem or int(sid.split('.')[1]) > int(ordinal):
            continue
        code = rec.get('code')
        if not isinstance(code, str) or len(code) > SOURCE_LIMIT:
            incomplete = True
            continue
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            incomplete = True
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in origins:
                kind = 'generated' if rec.get('gen_status') == 'ok' else 'saved (generation origin unconfirmed)'
                label = f'{kind} step{sid}'
                if label not in origins[node.name]:
                    origins[node.name].append(label)
    for name, found in origins.items():
        if found:
            add(f'Traceback symbol {name}: definition present in ' + ', '.join(found) +
                '. Presence does not establish which definition ran if test code shadows names.',
                f'{cp.name}:definitions:{name}')
        else:
            add(f'Traceback symbol {name}: no definition found in available cumulative chain code; '
                f'origin unresolved (source incomplete: {incomplete or not bool(steps)}). '
                'A genuine grader failure remains possible.', f'{cp.name}:definitions:{name}')
    return t

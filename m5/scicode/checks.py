"""Problem-side checks on a generated SciCode-format spec (sub-issue #352). None of them tests correctness:
- the spec parses (required keys, 1-15 steps, the requested step count +-1);
- every header (function_header + return_line) is valid Python: one def or class whose bodies are docstrings only,
  closed by a single `return` line;
- the import lines parse, use only the allowed modules, and import cleanly (checked separately in a batch,
  importable());
- step dependencies are consistent: unique names, no step refers to a function that a later step defines;
- the spec carries no solution code: no code fences or code lines in descriptions, nothing but docstrings in headers.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess

ALLOWED_TOP = {'numpy', 'scipy', 'math', 'cmath', 'itertools', 'functools', 'collections', 'copy', 'random', 'sympy',
               'heapq', 'fractions', 'decimal', 'statistics', 'operator', 'string', 're', 'bisect'}
KEYS = ('problem_name', 'problem_description_main', 'problem_io', 'required_dependencies', 'sub_steps')
STEP_KEYS = ('step_description_prompt', 'function_header', 'return_line')
CODE_LINE = re.compile(r'^\s*(def |class |import |from \S+ import |for .+:\s*$|while .+:\s*$|if .+:\s*$|return\b|'
                       r'[A-Za-z_][\w\.\[\], ]*\s*[+\-*/]?=\s*(np|scipy|sp|math)\.)', re.M)


def header_names(header: str, ret: str):
    """(kind, name, method names) of a header, or raises ValueError."""
    src = header.rstrip() + '\n' + ret.rstrip('\n')
    tree = ast.parse(src)
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.ClassDef)):
        raise ValueError('header must hold exactly one def or class')
    node = tree.body[0]

    def doc_only(fn, allow_return):
        body = list(fn.body)
        if not body or not (isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant)
                            and isinstance(body[0].value.value, str)):
            raise ValueError(f'{fn.name}: no docstring')
        rest = body[1:]
        if allow_return and rest and isinstance(rest[-1], ast.Return):
            rest = rest[:-1]
        rest = [s for s in rest if not isinstance(s, ast.Pass)]
        if rest:
            raise ValueError(f'{fn.name}: body beyond docstring (solution code?)')

    if isinstance(node, ast.FunctionDef):
        doc_only(node, True)
        if not node.body or not isinstance(node.body[-1], ast.Return):
            raise ValueError('return_line is not a return statement of the function')
        return 'def', node.name, []
    meths = [m for m in node.body if isinstance(m, ast.FunctionDef)]
    if not meths or len(meths) != len([m for m in node.body if not isinstance(m, ast.Expr)]):
        raise ValueError('class body must be method stubs only')
    for i, m in enumerate(meths):
        doc_only(m, i == len(meths) - 1)
    return 'class', node.name, [m.name for m in meths]


def parse_imports(deps: str):
    mods = []
    for line in deps.splitlines():
        s = line.split('#')[0].strip()
        if not s:
            continue
        tree = ast.parse(s)
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
            raise ValueError(f'not an import line: {s!r}')
        n = tree.body[0]
        mods += [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or '']
    return mods


def check(spec: dict, n_target: int | None = None) -> list[str]:
    """-> list of failure reasons (empty: the spec passes)."""
    errs = []
    for k in KEYS:
        if k not in spec or not spec[k]:
            errs.append(f'missing {k}')
    if errs:
        return errs
    steps = spec['sub_steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 15:
        return ['steps: need a list of 1-15']
    if n_target and abs(len(steps) - n_target) > 1:
        errs.append(f'steps: {len(steps)} for a target of {n_target}')
    if not re.fullmatch(r'[A-Za-z0-9_\-]{3,80}', str(spec['problem_name'])):
        errs.append('problem_name')
    try:
        mods = parse_imports(spec['required_dependencies'])
        bad = [m for m in mods if m.split('.')[0] not in ALLOWED_TOP]
        if bad:
            errs.append(f'imports outside the allowed set: {bad}')
        if not any(m.split('.')[0] == 'numpy' for m in mods):
            errs.append('imports: no numpy')
    except (SyntaxError, ValueError) as e:
        errs.append(f'imports: {e}')
    names = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or any(not isinstance(s.get(k), str) or not s.get(k).strip() for k in STEP_KEYS):
            errs.append(f'step {i + 1}: missing fields')
            continue
        try:
            kind, name, meths = header_names(s['function_header'], s['return_line'])
            names.append((i, name))
        except (SyntaxError, ValueError) as e:
            errs.append(f'step {i + 1}: header: {str(e)[:120]}')
            continue
        d = s['step_description_prompt']
        if '```' in d or CODE_LINE.search(d):
            errs.append(f'step {i + 1}: code in description')
        if len(d) < 80:
            errs.append(f'step {i + 1}: description too short')
        if '```' in s['function_header'] or not s['return_line'].lstrip().startswith('return'):
            errs.append(f'step {i + 1}: return_line')
    if '```' in spec['problem_description_main'] or CODE_LINE.search(spec['problem_description_main']):
        errs.append('main description: code')
    nm = [n for _, n in names]
    if len(set(nm)) != len(nm):
        errs.append(f'duplicate step names: {nm}')
    for i, name in names:   # forward references: step j < i mentions step i's function
        pat = re.compile(r'\b' + re.escape(name) + r'\b')
        for j in range(i):
            s = steps[j]
            if isinstance(s, dict) and pat.search((s.get('step_description_prompt') or '') + (s.get('function_header') or '')):
                errs.append(f'step {j + 1} refers to {name}, defined later in step {i + 1}')
                break
    return errs


def uses_earlier(spec: dict) -> float:
    """Share of non-final steps whose function is named in a later step (a soft chain-coherence metric)."""
    steps = spec['sub_steps']
    if len(steps) < 2:
        return 1.0
    hit = 0
    for i, s in enumerate(steps[:-1]):
        try:
            _, name, _ = header_names(s['function_header'], s['return_line'])
        except Exception:  # noqa: BLE001
            continue
        later = ' '.join(t['step_description_prompt'] + t['function_header'] for t in steps[i + 1:])
        hit += bool(re.search(r'\b' + re.escape(name) + r'\b', later))
    return hit / (len(steps) - 1)


IMPORT_CHECK = r'''
import json, sys
out = {}
for i, deps in enumerate(json.load(sys.stdin)):
    try:
        exec(deps, {})
        out[i] = None
    except Exception as e:
        out[i] = f"{type(e).__name__}: {e}"[:200]
print(json.dumps(out))
'''


def importable(deps_list: list[str], python: str) -> list[str | None]:
    """Run each dependency block in `python` (the grading env's numpy 1.26 / scipy 1.13 stack); None = imports."""
    r = subprocess.run([python, '-I', '-c', IMPORT_CHECK], input=json.dumps(deps_list), capture_output=True,
                       text=True, timeout=600)
    res = json.loads(r.stdout)
    return [res[str(i)] for i in range(len(deps_list))]

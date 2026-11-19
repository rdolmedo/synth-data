"""Problem-side checks (static; no answers involved).

parse_generation: the generator's tagged reply -> dict or a skip/format failure.
static_checks: the template parses, defines `answer` with the signature of the sampled answer type, keeps the
  FILL IN block with `...` placeholders, imports only allowed modules, declares sympy symbols for symbolic
  arguments; every argument appears in the problem or docstring; the description has both parts, is within length
  bounds and does not cite the paper.
"""
from __future__ import annotations

import ast
import re

from .templates import ALLOWED_IMPORTS, ARGS_TYPES

TAG = lambda name: re.compile(rf'<{name}>\s*(.*?)\s*</{name}>', re.S)
FENCE = re.compile(r'```(?:python|py)?[ \t]*\n(.*?)```', re.S)
CITES = re.compile(r'\b(the paper|this paper|the authors|in Ref\.|Refs?\.\s*\[|\\cite|\\ref\{|Fig\.|Figure \d|'
                   r'Eq\.\s*\(\d|Sec\.\s*\d|Section \d|Table \d|arXiv)', re.I)


def parse_generation(text: str) -> dict:
    st = TAG('status').search(text or '')
    if not st:
        m = re.search(r'<status>\s*SKIP\b[\s:]*(.*)', text or '', re.S)   # a skip without the closing tag
        if m:
            return {'ok': False, 'skip': True, 'reason': 'skip: ' + m.group(1).strip()[:500]}
        return {'ok': False, 'reason': 'format: no <status>'}
    s = st.group(1).strip()
    if s.upper().startswith('SKIP'):
        return {'ok': False, 'skip': True, 'reason': 'skip: ' + s[4:].strip(' :')}
    if not s.upper().startswith('OK'):
        return {'ok': False, 'reason': f'format: status {s[:50]!r}'}
    prob, tpl = TAG('problem').search(text), TAG('template').search(text)
    if not prob or not tpl:
        return {'ok': False, 'reason': 'format: missing <problem> or <template>'}
    m = FENCE.search(tpl.group(1))
    code = (m.group(1) if m else tpl.group(1)).strip('\n') + '\n'
    src, tgt = TAG('source').search(text), TAG('target').search(text)
    return {'ok': True, 'description': prob.group(1).strip() + '\n', 'code_template': code,
            'source': src.group(1).strip() if src else None, 'target': tgt.group(1).strip() if tgt else None}


def words(s: str) -> int:
    return len(s.split())


def repair_template(code: str) -> tuple[str, int]:
    """Blank a filled-in FILL IN block: every assignment between the "FILL IN" marker and the closing "# ---" rule
    becomes `name = ...` (multi-line values collapse to one line; a trailing comment survives only if it has no digit).
    -> (code, number of assignments blanked). Assignments outside the block are left for static_checks to reject.
    Unparsable code or a missing block is returned unchanged."""
    lines = code.split('\n')
    try:
        start = next(i for i, l in enumerate(lines) if 'FILL IN' in l)
        end = next(i for i in range(start + 1, len(lines)) if re.match(r'\s*#\s*-{6,}', lines[i]))
        tree = ast.parse(code)
    except (StopIteration, SyntaxError):
        return code, 0
    edits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and start < n.lineno - 1 < end and not (
                isinstance(n.value, ast.Constant) and n.value.value is Ellipsis):
            tgt = ', '.join(ast.get_source_segment(code, t) for t in n.targets[:1])
            tail = lines[n.end_lineno - 1][n.end_col_offset:]
            com = tail[tail.index('#'):] if '#' in tail and not re.search(r'\d', tail) else ''
            indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
            edits.append((n.lineno - 1, n.end_lineno - 1, f'{indent}{tgt} = ...' + (f'  {com}' if com else '')))
    for a, b, new in sorted(edits, reverse=True):
        lines[a:b + 1] = [new]
    # value-bearing comments on placeholder lines too
    out = []
    for i, l in enumerate(lines):
        m = re.match(r'(\s*[\w, ]+=\s*\.\.\.)\s*#.*\d', l)
        out.append(m.group(1) if m else l)
    return '\n'.join(out), len(edits)


def static_checks(desc: str, code: str, answer_type: str) -> list[str]:
    errs = []
    # description
    if '# Problem setup:' not in desc or '# Main problem:' not in desc:
        errs.append('desc: missing "# Problem setup:" or "# Main problem:"')
    else:
        setup, main = desc.split('# Main problem:', 1)
        ws, wm = words(setup), words(main)
        if not 40 <= ws <= 450:
            errs.append(f'desc: setup {ws} words')
        if not 10 <= wm <= 300:
            errs.append(f'desc: main problem {wm} words')
    c = CITES.search(desc) or CITES.search(code)
    if c:
        errs.append(f'cites: {c.group(0)!r}')
    # template
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return errs + [f'template: syntax error ({e.msg}, line {e.lineno})']
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'answer']
    if len(fns) != 1:
        return errs + ['template: no single top-level def answer']
    fn = fns[0]
    args = [a.arg for a in fn.args.args]
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults:
        errs.append('template: answer() has *args/**kwargs/defaults')
    if answer_type in ARGS_TYPES and not args:
        errs.append(f'template: {answer_type} needs arguments')
    if answer_type not in ARGS_TYPES and args:
        errs.append(f'template: {answer_type} takes no arguments, got {args}')
    doc = ast.get_docstring(fn) or ''
    if not re.search(r'Inputs?\s*\n\s*-{4,}', doc) or not re.search(r'Outputs?\s*\n\s*-{4,}', doc):
        errs.append('template: docstring lacks Inputs/Outputs sections')
    if 'FILL IN' not in code:
        errs.append('template: no FILL IN block')
    ell = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and n.value is Ellipsis]
    if not ell:
        errs.append('template: no ... placeholder')
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        errs.append('template: does not end with return')
    # a filled-in answer (any non-placeholder value assigned in the body) is a leak
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and not (isinstance(n.value, ast.Constant) and n.value.value is Ellipsis):
            errs.append('template: body assigns a value other than ... (possible answer/helper code)')
            break
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or '']
            for mname in mods:
                if mname not in ALLOWED_IMPORTS and mname.split('.')[0] not in {'sympy', 'numpy'}:
                    errs.append(f'template: import {mname}')
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not fn:
            errs.append(f'template: extra top-level definition {n.name}')
    if answer_type.startswith('symbolic'):
        declared = set()
        for n in tree.body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    for e in ([t] if isinstance(t, ast.Name) else getattr(t, 'elts', [])):
                        if isinstance(e, ast.Name):
                            declared.add(e.id)
        missing = [a for a in args if a not in declared]
        if missing:
            errs.append(f'template: symbolic arguments not declared with sp.symbols: {missing}')
        if 'sympy' not in code:
            errs.append('template: symbolic answer without sympy import')
    # every argument is mentioned in the problem or docstring
    for a in args:
        base = a.rstrip('_')
        if not re.search(rf'\b{re.escape(a)}\b|\b{re.escape(base)}\b', doc) and base not in desc:
            errs.append(f'template: argument {a} not described')
    return errs

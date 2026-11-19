"""Does a frozen code copy hold everything its job imports? (#451 workstream 5, lesson 14)

Since #446 the HLE / SciCode / Omniscience / GDP.pdf jobs import tb_agents on every item, but their submitters froze
only aa_evals/ (fixed in #449, found by a failing job). Imports inside functions are the usual culprit: an import
check of the entry module passes, and the job fails at its first item.

`unresolved(root, entries, packages, source)` walks every import statement (module level and inside functions) of
the entry modules (and their packages' __init__) and of every module they reach inside the frozen copy `root`, and
lists the imports the copy cannot resolve although the source tree `source` (the checkout the copy is made from)
can: a missing repository package, a missing module of a copied package (`from aa_evals.common import jsonl` with
only some of common/ copied), or a missing sibling module of a script (`from data import x` next to data.py).
Optional imports are not listed: inside the `try:` body of a try whose handlers catch ImportError /
ModuleNotFoundError (not a broad Exception, which is how a per-item import failure gets swallowed), or under
`if TYPE_CHECKING:`.
"""
from __future__ import annotations

import ast
from pathlib import Path


def repo_packages(repo) -> set[str]:
    """Top-level directories of the repository that hold Python code."""
    return {p.name for p in Path(repo).iterdir()
            if p.is_dir() and not p.name.startswith('.') and any(p.glob('*.py'))}


def module_file(root: Path, name: str) -> Path | None:
    p = Path(root).joinpath(*name.split('.'))
    if p.with_suffix('.py').is_file():
        return p.with_suffix('.py')
    if (p / '__init__.py').is_file():
        return p / '__init__.py'
    if p.is_dir():
        return p   # a namespace package
    return None


def _optional_try(node: ast.Try) -> bool:
    for h in node.handlers:
        kinds = [ast.unparse(e) for e in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])] if h.type else []
        if any(k.split('.')[-1] in ('ImportError', 'ModuleNotFoundError') for k in kinds):
            return True
    return False


def _type_checking(node) -> bool:
    return isinstance(node, ast.If) and ast.unparse(node.test).split('.')[-1] == 'TYPE_CHECKING'


def imports(root: Path, path: Path):
    """(module name candidates, optional, line) of every import in `path`: for `from pkg import name` both
    'pkg.name' (if name is a module) and 'pkg'; relative imports resolved against the file's package."""
    tree = ast.parse(path.read_text(errors='replace'))
    rel = path.relative_to(root).with_suffix('').parts
    package = list(rel[:-1])
    out = []

    def visit(node, optional):
        if isinstance(node, ast.Import):
            out.extend(((a.name,), optional, node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                up = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                base = '.'.join([*up, *([node.module] if node.module else [])])
            for a in node.names:
                full = f'{base}.{a.name}' if base else a.name
                out.append(((full, base), optional, node.lineno))
        if isinstance(node, ast.Try) and _optional_try(node):
            for child in node.body:
                visit(child, True)
            for child in [*node.handlers, *node.orelse, *node.finalbody]:
                visit(child, optional)
            return
        if _type_checking(node):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, optional)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            optional = False   # a function defined inside a guarded try runs later, unguarded
        for child in ast.iter_child_nodes(node):
            visit(child, optional)

    visit(tree, False)
    return out


def _binds(init: Path, name: str) -> bool:
    """Does the package __init__ bind `name` itself (an import or an assignment), so `from pkg import name` works
    without pkg/name.py?"""
    try:
        tree = ast.parse(init.read_text(errors='replace'))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and not node.module:
            continue   # `from . import name` imports the package's own module: it needs pkg/name.py, binds nothing
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split('.')[0]) == name
                                                                  for a in node.names):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, 'id', None) == name for t in targets):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
    return False


def unresolved(root, entries, packages, source=None) -> list[str]:
    """Imports reachable from `entries` (module names or paths relative to root) that the frozen copy `root` cannot
    resolve but the source tree can (without `source`: any repository package the copy lacks)."""
    root = Path(root)
    source = Path(source) if source else None
    todo, seen, missing = [], set(), []

    def add(f):
        todo.append(f)
        rel = f.relative_to(root).parts
        for i in range(1, len(rel)):   # the packages a module is imported through
            init = root.joinpath(*rel[:i], '__init__.py')
            if init.is_file():
                todo.append(init)

    for e in entries:
        f = root / e if e.endswith('.py') else module_file(root, e)
        if f is None or not f.exists():
            missing.append(f'entry {e} is not in the frozen copy')
        else:
            add(f)
    while todo:
        f = todo.pop()
        if f in seen or f.is_dir():
            continue
        seen.add(f)
        for names, optional, line in imports(root, f):
            first = names[0]
            top = first.split('.')[0]
            if top not in packages:   # a script's sibling module (sft_qwen35's style), else third party / stdlib
                rel_dir = f.parent.relative_to(root)
                sib = f.parent / f'{top}.py'
                if f.parent != root and sib.is_file():
                    add(sib)
                elif (f.parent != root and source is not None and (source / rel_dir / f'{top}.py').is_file()
                      and not optional):
                    missing.append(f'{f.relative_to(root)}:{line} imports {top} (sibling module not copied)')
                continue
            target = next((t for t in (module_file(root, n) for n in names if n) if t is not None), None)
            in_source = source is None or any(module_file(source, n) for n in names if n)
            wanted = next((n for n in names if n and (source is None or module_file(source, n))), first)
            reexported = (target is not None and target.name == '__init__.py' and wanted != names[-1]
                          and _binds(target, wanted.rsplit('.', 1)[-1]))   # `from pkg import x`, x bound in pkg
            if target is None or (source is not None and not reexported and module_file(root, wanted) is None
                                  and module_file(source, wanted) is not None):
                if not optional and in_source:
                    missing.append(f'{f.relative_to(root)}:{line} imports {wanted}')
                if target is None:
                    continue
            add(target)
    return sorted(set(missing))

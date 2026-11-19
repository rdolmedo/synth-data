"""The server-registry reader (aa_evals/common/serving.py, tbshared/serving.py's ALIASES copy) is in every partial
snapshot whose clients read a serving registry with it (#445 item 3): a snapshot without it fails at import."""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_gdppdf_snapshot_carries_it(tmp_path):
    from aa_evals.gdppdf import submit
    submit.freeze(tmp_path / 'code')
    assert (tmp_path / 'code' / 'aa_evals' / 'common' / 'serving.py').read_bytes() == \
        (REPO / 'tbshared' / 'serving.py').read_bytes()


def test_automationbench_zip_carries_it(tmp_path):
    from aa_evals.automationbench import submit
    dst = tmp_path / 'code.zip'
    submit.freeze(dst)
    with zipfile.ZipFile(dst) as z:
        assert {'aa_evals/common/__init__.py', 'aa_evals/common/serving.py'} <= set(z.namelist())


def test_gdpval_and_omniscience_snapshot_lists_name_it():
    assert "'serving.py'" in (REPO / 'aa_evals' / 'gdpval' / 'submit.py').read_text()
    assert '"$REPO/aa_evals/common/serving.py"' in (REPO / 'aa_evals' / 'omniscience' / 'submit.sh').read_text()


def _imports_in(code_dir, module):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, '-I', '-c', f'import sys; sys.path.insert(0, {str(code_dir)!r}); import {module}'],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stderr[-500:]


def test_clients_import_inside_their_snapshots(tmp_path, monkeypatch):
    from aa_evals.gdppdf import submit as gdppdf_submit
    from aa_evals.gdpval import submit as gdpval_submit
    gdppdf_submit.freeze(tmp_path / 'gdppdf')
    assert _imports_in(tmp_path / 'gdppdf', 'aa_evals.gdppdf.client') == (0, '')
    monkeypatch.setattr(gdpval_submit, 'CODE_ROOT', tmp_path / 'gdpval')
    monkeypatch.setattr(gdpval_submit, 'git_sha', lambda: 'abc')
    code = gdpval_submit.freeze()
    for module in ('aa_evals.gdpval.endpoint', 'aa_evals.hle.client'):
        assert _imports_in(code, module) == (0, ''), module


def test_automationbench_zip_imports_runs_aa_evals_modules(tmp_path):
    """Every aa_evals module run.py imports at module level is in the zip and imports from it alone (jsonl was
    missing from it since 3c7f1eb9)."""
    import ast
    import subprocess
    import sys
    from aa_evals.automationbench import submit
    dst = tmp_path / 'code.zip'
    submit.freeze(dst)
    tree = ast.parse((REPO / 'aa_evals' / 'automationbench' / 'run.py').read_text())
    mods = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('aa_evals'):
            for a in node.names:
                mods.add(f'{node.module}.{a.name}' if (REPO / (node.module.replace('.', '/') + f'/{a.name}.py')).exists()
                         else node.module)
    code = 'import sys; sys.path.insert(0, %r)\n' % str(dst) + ''.join(f'import {m}\n' for m in sorted(mods))
    r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert 'aa_evals.common.jsonl' in mods and 'aa_evals.common.serving' in mods

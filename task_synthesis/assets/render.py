"""An asset as the task author and the judge see it. The author needs the real content to write setup code against
it (plant a bug in a function, pick columns, cut a log), so the selected text files of code-like assets go in whole,
up to a budget; data files are shown by their manifest (types, shapes, heads)."""
import tempfile
from pathlib import Path

from . import manifest as manifest_mod, store

CODE_EXT = {'.py', '.c', '.h', '.cc', '.cpp', '.hpp', '.cu', '.cuh', '.rs', '.go', '.java', '.js', '.ts', '.tsx', '.jsx',
            '.rb', '.pl', '.pm', '.php', '.sh', '.bash', '.lua', '.ml', '.mli', '.hs', '.lean', '.v', '.scm', '.rkt',
            '.lisp', '.clj', '.ex', '.exs', '.erl', '.kt', '.scala', '.swift', '.cs', '.fs', '.f90', '.f', '.cob',
            '.cbl', '.red', '.asm', '.s', '.zig', '.nim', '.jl', '.r', '.sql', '.tcl', '.awk', '.pas', '.ada', '.adb',
            '.ads', '.vhd', '.sv', '.m', '.mk', '.cmake', '.toml', '.cfg', '.ini', '.conf', '.yaml', '.yml', '.json',
            '.xml', '.service', '.md', '.rst', '.txt', '.tex', '.html', '.css', '.urdf', '.xacro', '.ics', '.gcode',
            '.proto', '.graphql', '.dockerfile', '.gradle', '.properties', '.in', '.am', '.ac'}
CODE_NAMES = {'Makefile', 'CMakeLists.txt', 'Dockerfile', 'README', 'LICENSE', 'setup.py', 'pyproject.toml',
              'Cargo.toml', 'package.json', 'go.mod', 'meson.build', 'configure.ac'}
FULL_CLASSES = {'python-code', 'c-cpp-code', 'other-code', 'git-repo', 'running-service', 'config', 'binary',
                'upstream-project', 'document', 'structured-data', 'text-data', 'scientific-data'}
DATA_EXT = {'.csv', '.tsv', '.jsonl', '.log', '.fasta', '.fa', '.gb', '.gbk', '.pdb', '.cif', '.sdf'}


def selection_dir(record, paths, tmp):
    paths = None if not paths or paths == ['.'] else paths
    return store.extract(record, Path(tmp) / 'sel', paths)


def full_texts(root, cls, budget):
    """Whole text files, most central first: shallow code files, then the rest; data-like files only for small ones."""
    if cls not in FULL_CLASSES:
        return []
    files = []
    for p, rel in manifest_mod.walk(root):
        ext = p.suffix.lower()
        size = p.stat().st_size
        if size == 0 or size > 60_000:
            continue
        if ext in DATA_EXT and size > 6_000:
            continue
        if ext not in CODE_EXT and p.name not in CODE_NAMES:
            continue
        if p.name == 'LICENSE' or p.name.startswith('LICENSE'):
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        score = (rel.count('/'), 0 if ext in CODE_EXT - {'.md', '.txt', '.rst', '.json', '.xml', '.html'} else 1, size)
        files.append((score, rel, text))
    files.sort()
    out, left = [], budget
    for _, rel, text in files:
        if len(text) + 200 > left:
            continue
        out.append((rel, text))
        left -= len(text) + 200
        if left < 2000:
            break
    return out


def for_generator(record, paths, cls, budget=50000):
    with tempfile.TemporaryDirectory(prefix='asset-render-') as tmp:
        root = selection_dir(record, paths, tmp)
        m = manifest_mod.build(root)
        if record['manifest']['stats'].get('(git history)'):
            m['stats']['(git history)'] = record['manifest']['stats']['(git history)']
        texts = full_texts(root, cls, budget)
        shown = {rel for rel, _ in texts}
        m['heads'] = {k: v for k, v in m['heads'].items() if k not in shown}
        man = manifest_mod.render(m, 16000)
    parts = [man]
    if texts:
        parts.append('\n\n## Full text of selected files')
        for rel, text in texts:
            parts.append(f'\n### {rel}\n```\n{text.rstrip()}\n```')
    return ''.join(parts)


def for_judge(record, paths, budget=9000):
    """Provenance and a manifest of the shipped selection (the judge also sees the built tree)."""
    sel = None if not paths or paths == ['.'] else paths
    head = (f'Source: {record.get("source")}; {record.get("title")}; license {record.get("license")}; '
            f'{record.get("url")}\nDescription: {(record.get("description") or "")[:500]}\n'
            f'Shipped selection: {", ".join(paths or ["."])}\n\n')
    return head + manifest_mod.render(record['manifest'], budget, paths=sel)

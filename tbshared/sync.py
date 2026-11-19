"""Vendor tbshared/ from branch `shared` into a pipeline worktree, pinned by commit.

  python -m tbshared.sync [COMMIT]          # default origin/shared: replaces tbshared/, writes tbshared/PIN and the
                                            # copies listed in tbshared/ALIASES
  python -m tbshared.sync --check           # exit 1 when tbshared/ or an alias differs from the PIN (test_pin.py)
  python PATH/TO/tbshared/sync.py COMMIT --root PIPELINE_WORKTREE    # first migration (no tbshared/ there yet)

Run in a pipeline worktree after `git fetch origin shared`; the bump is then a normal PR on the pipeline branch (its
own tests must pass). Never edit tbshared/ (or an alias) in a pipeline branch: fix it on `shared`, then sync.

tbshared/ALIASES (pipeline-local, optional): one `DEST SOURCE` pair per line, DEST relative to the repository root,
SOURCE relative to tbshared/. sync writes DEST as a byte-identical copy of SOURCE, and --check / test_pin.py check it.
For pipelines whose code snapshots copy only some directories (aa-evals' per-benchmark snapshots), so the shared code
must also exist at its old path. An alias source must be self-contained (it must not import tbshared). Removing a line
from ALIASES leaves its old copy unchecked: delete that copy in the same change.

tbshared/pipeline.json (pipeline-local, optional): the pipeline's own config (pipeline_config.py). sync keeps it, and
the pin check ignores it. Stdlib only (also runs by path).

The trunk `main` (#480) holds the canonical tbshared/ (tbshared/TRUNK, no PIN): nothing is vendored there, sync
refuses it, and --check / test_pin.py check that every ALIASES copy is byte-identical to tbshared/ itself.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REMOTE_BRANCH = 'origin/shared'
LOCAL = {'PIN', 'ALIASES', 'pipeline.json', 'TRUNK'}   # pipeline-local files, never part of a pinned tree
JUNK_DIRS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def git(*args, cwd='.', text=True) -> str | bytes:
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=text, timeout=120)
    if r.returncode:
        raise RuntimeError(f'git {" ".join(args)}: {(r.stderr or "").strip()[:300]}')
    return r.stdout


def resolve(commit: str, cwd='.') -> str:
    return git('rev-parse', '--verify', f'{commit}^{{commit}}', cwd=cwd).strip()


def on_shared(commit: str, cwd='.', shared=REMOTE_BRANCH) -> bool:
    """Whether `commit` is merged into `shared` (an ancestor of it)."""
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, shared], cwd=cwd,
                          capture_output=True).returncode == 0


def is_shared_tree(root: Path) -> bool:
    """The `shared` branch's own worktree: HEAD tracks nothing at the top level but tbshared/ (untracked files do not
    count) and no tbshared/PIN."""
    r = subprocess.run(['git', 'ls-tree', '--name-only', 'HEAD'], cwd=root, capture_output=True, text=True)
    if r.returncode:
        return False
    names = {n for n in r.stdout.split() if not n.startswith('.')}
    has_pin = subprocess.run(['git', 'cat-file', '-e', 'HEAD:tbshared/PIN'], cwd=root,
                             capture_output=True).returncode == 0
    return names == {'tbshared'} and not has_pin


def is_trunk(root: Path) -> bool:
    """The trunk's canonical tbshared/ (tbshared/TRUNK): edited in place, not vendored, no PIN."""
    return (Path(root) / 'tbshared' / 'TRUNK').is_file()


def trunk_diff(root: Path) -> list[str]:
    """On the trunk: every ALIASES copy byte-identical to its source in tbshared/ (and self-contained), no PIN."""
    root = Path(root)
    have = local_tree(root)
    out = ['tbshared/PIN on the trunk: the canonical copy is not vendored'] if (root / 'tbshared' / 'PIN').exists() else []
    for dest, src in aliases(root):
        p = root / dest
        if src not in have:
            out.append(f'alias source missing in tbshared/: {dest} <- {src}')
        elif not p.is_file():
            out.append(f'alias missing: {dest} <- {src}')
        elif git('hash-object', str(p), cwd=root).strip() != have[src]:
            out.append(f'alias differs: {dest} <- {src}')
        if src in have and re.search(r'^\s*(from|import)\s+tbshared\b|^\s*from\s+\.',
                                     (root / 'tbshared' / src).read_text(), re.M):
            out.append(f'alias source imports tbshared (not self-contained): {dest} <- {src}')
    return out


def pinned_tree(commit: str, cwd='.') -> dict[str, str]:
    """{path under tbshared/: blob sha} of `commit`'s tbshared/ tree (pipeline-local names excluded)."""
    out = git('ls-tree', '-r', commit, '--', 'tbshared/', cwd=cwd)
    files = {}
    for line in out.splitlines():
        meta, path = line.split('\t', 1)
        _, kind, sha = meta.split()
        rel = path[len('tbshared/'):]
        if kind == 'blob' and rel not in LOCAL:
            files[rel] = sha
    return files


def _junk(rel: str) -> bool:
    parts = rel.split('/')
    name = parts[-1]
    return (bool(JUNK_DIRS & set(parts[:-1])) or name.endswith(('.pyc', '~', '.swp', '.swo')) or name.startswith('._')
            or name == '.DS_Store')


def local_tree(root: Path) -> dict[str, str]:
    """{path under tbshared/: blob sha} of the working copy: tracked files plus untracked ones git does not ignore,
    without caches and editor/OS junk (and without the LOCAL files)."""
    out = git('ls-files', '-co', '--exclude-standard', '-z', '--', 'tbshared/', cwd=root)
    files = {}
    for path in sorted(set(p for p in out.split('\0') if p)):
        rel = path[len('tbshared/'):]
        p = root / path
        if rel in LOCAL or _junk(rel) or not p.is_file():
            continue
        files[rel] = git('hash-object', str(p), cwd=root).strip()
    return files


def aliases(root: Path) -> list[tuple[str, str]]:
    """[(dest relative to root, source relative to tbshared/)] from tbshared/ALIASES."""
    f = root / 'tbshared' / 'ALIASES'
    out = []
    for n, line in enumerate(f.read_text().splitlines() if f.exists() else [], 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f'{f}:{n}: expected `DEST SOURCE`, got {line!r}')
        out.append((parts[0], parts[1]))
    return out


def diff(root: Path, commit: str) -> list[str]:
    """Differences between root/tbshared (and its aliases) and `commit`'s tbshared/ ([] = identical)."""
    want, have = pinned_tree(commit, cwd=root), local_tree(root)
    out = [f'missing: {k}' for k in sorted(set(want) - set(have))]
    out += [f'not in the pin: {k}' for k in sorted(set(have) - set(want))]
    out += [f'changed: {k}' for k in sorted(set(want) & set(have)) if want[k] != have[k]]
    for dest, src in aliases(root):
        p = root / dest
        if src not in want:
            out.append(f'alias source not in the pin: {dest} <- {src}')
        elif not p.is_file():
            out.append(f'alias missing: {dest} <- {src}')
        elif git('hash-object', str(p), cwd=root).strip() != want[src]:
            out.append(f'alias differs: {dest} <- {src}')
        if src in want and re.search(r'^\s*(from|import)\s+tbshared\b|^\s*from\s+\.', (root / 'tbshared' / src).read_text()
                                     if (root / 'tbshared' / src).is_file() else '', re.M):
            # an alias copy runs where tbshared/ may be missing (partial snapshots): its source must be self-contained
            out.append(f'alias source imports tbshared (not self-contained): {dest} <- {src}')
    return out


def vendor(root: Path, commit: str, allow_unmerged: bool = False) -> str:
    """Replace root/tbshared with `commit`'s tbshared/, keep its pipeline-local files (ALIASES, pipeline.json), write
    PIN and the alias copies; returns the full commit id. Refuses the `shared` branch's own worktree and (unless
    allow_unmerged) a commit not on shared."""
    root = Path(root)
    if is_shared_tree(root):
        raise SystemExit(f'{root} is branch `shared` itself: sync vendors into a pipeline worktree')
    if is_trunk(root):
        raise SystemExit(f'{root} is the trunk: its tbshared/ is the canonical copy (tbshared/TRUNK), never vendored')
    sha = resolve(commit, cwd=root)
    if not allow_unmerged and not on_shared(sha, cwd=root):
        raise SystemExit(f'{sha[:12]} is not merged into {REMOTE_BRANCH} (`git fetch origin shared`; '
                         '--allow-unmerged for a PR under review)')
    data = git('archive', '--format=tar', sha, 'tbshared', cwd=root, text=False)
    base = root / 'tbshared'
    keep = {n: (base / n).read_bytes() for n in sorted(LOCAL - {'PIN'}) if (base / n).is_file()}
    if base.exists():
        shutil.rmtree(base)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(root, filter='data')
        else:
            tf.extractall(root)
    for name in LOCAL:                       # never take a local file committed to shared by mistake
        (base / name).unlink(missing_ok=True)
    for name, data in keep.items():
        (base / name).write_bytes(data)
    (base / 'PIN').write_text(sha + '\n')
    for dest, src in aliases(root):
        d = root / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / src, d)
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('commit', nargs='?', default=REMOTE_BRANCH)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--root', default='.')
    ap.add_argument('--allow-unmerged', action='store_true', help='pin a commit not (yet) on shared, e.g. a PR')
    a = ap.parse_args(argv)
    root = Path(git('rev-parse', '--show-toplevel', cwd=a.root).strip())
    if a.check and is_trunk(root):
        d = trunk_diff(root)
        print('\n'.join(d) or f'trunk: tbshared/ is canonical; {len(aliases(root))} aliases match it')
        return 1 if d else 0
    if a.check:
        pin = root / 'tbshared' / 'PIN'
        if not pin.exists():
            print(f'{pin} is missing: vendor tbshared/ with `python -m tbshared.sync` first')
            return 1
        d = diff(root, pin.read_text().strip())
        print('\n'.join(d) or f'tbshared/ matches its pin {pin.read_text().strip()[:12]}')
        return 1 if d else 0
    sha = vendor(root, a.commit, a.allow_unmerged)
    print(f'tbshared/ vendored at {sha[:12]} ({root}); aliases: {len(aliases(root))}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

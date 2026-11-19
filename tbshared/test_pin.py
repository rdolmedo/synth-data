"""In a pipeline branch: tbshared/ (and every copy listed in tbshared/ALIASES) must be byte-identical to the `shared`
commit in tbshared/PIN, so a fix made in one pipeline's copy fails here instead of forking. Fix it on branch `shared`
and bump with `python -m tbshared.sync`.

On the trunk `main` (tbshared/TRUNK, #480) tbshared/ is the canonical copy: there is no pin to compare with (branch
`shared` is no longer where changes land), and every ALIASES copy must be byte-identical to tbshared/ itself."""
import shutil
import subprocess
from pathlib import Path

import pytest

from tbshared import sync

HERE = Path(__file__).resolve().parent


def _root():
    if shutil.which('git') is None:
        return None
    r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=HERE, capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 else None


def test_vendored_copy_matches_its_pin():
    root = _root()
    if root is None or root != HERE.parent:
        pytest.skip('not inside a git worktree (a frozen code snapshot, or no git): checked where the copy was made')
    if sync.is_trunk(root):
        d = sync.trunk_diff(root)
        assert not d, 'trunk: an ALIASES copy differs from tbshared/ (copy it again; never edit the alias):\n' + '\n'.join(d)
        return
    pin = HERE / 'PIN'
    if not pin.exists():
        if sync.is_shared_tree(root):
            pytest.skip('branch `shared` itself: nothing vendored')
        pytest.fail('tbshared/PIN is missing: vendor tbshared/ with `python -m tbshared.sync`, never by hand')
    commit = pin.read_text().strip()
    try:
        sync.resolve(commit, cwd=root)
    except RuntimeError:
        pytest.fail(f'PIN {commit[:12]} is not in this repository: `git fetch origin shared`, then rerun')
    d = sync.diff(root, commit)
    assert not d, ('tbshared/ differs from its pin (edit it on branch `shared`, then `python -m tbshared.sync`):\n'
                   + '\n'.join(d))

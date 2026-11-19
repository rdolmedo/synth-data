"""tbshared.sync (vendor + check) and tbshared.drift on a throwaway repository."""
import subprocess

import pytest

from tbshared import drift, sync


def g(cwd, *a):
    return subprocess.run(['git', *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    g(r, 'init', '-q', '-b', 'shared')
    g(r, 'config', 'user.email', 't@t')
    g(r, 'config', 'user.name', 't')
    (r / 'tbshared').mkdir()
    (r / 'tbshared' / 'a.py').write_text('A = 1\n')
    (r / 'tbshared' / 'sub').mkdir()
    (r / 'tbshared' / 'sub' / 'b.txt').write_text('b\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'v1')
    v1 = g(r, 'rev-parse', 'HEAD')
    g(r, 'update-ref', 'refs/remotes/origin/shared', 'shared')
    g(r, 'checkout', '-q', '--orphan', 'eval')
    g(r, 'rm', '-rfq', '.')
    (r / 'pipeline.py').write_text('x = 1\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'pipeline')
    return r, v1


def test_vendor_then_check(repo):
    r, v1 = repo
    assert sync.vendor(r, v1) == v1
    assert (r / 'tbshared' / 'PIN').read_text().strip() == v1
    assert (r / 'tbshared' / 'sub' / 'b.txt').read_text() == 'b\n'
    assert sync.diff(r, v1) == []
    (r / 'tbshared' / '__pycache__').mkdir()
    (r / 'tbshared' / '__pycache__' / 'a.cpython-312.pyc').write_bytes(b'x')
    assert sync.diff(r, v1) == []                                   # caches are not part of the copy
    (r / 'tbshared' / 'a.py').write_text('A = 2  # a local fix\n')
    (r / 'tbshared' / 'c.py').write_text('')
    (r / 'tbshared' / 'sub' / 'b.txt').unlink()
    assert sync.diff(r, v1) == ['missing: sub/b.txt', 'not in the pin: c.py', 'changed: a.py']
    assert sync.main(['--check', '--root', str(r)]) == 1
    sync.vendor(r, v1)                                              # re-vendoring restores the pinned copy
    assert sync.main(['--check', '--root', str(r)]) == 0


def test_drift_report(repo, monkeypatch):
    r, v1 = repo
    sync.vendor(r, v1)
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'vendor tbshared v1')
    g(r, 'checkout', '-q', 'shared')
    (r / 'tbshared' / 'a.py').write_text('A = 3\n')
    g(r, 'commit', '-qam', 'shared: a fix')
    for ref, br in (('shared', 'shared'), ('eval', 'eval')):   # noqa: B007
        g(r, 'update-ref', f'refs/remotes/origin/{ref}', br)
    monkeypatch.chdir(r)
    rows = {x['branch']: x for x in drift.report(('eval', 'sft'))}
    assert rows['eval']['state'] == 'behind' and rows['eval']['behind'] == 1
    assert rows['eval']['missing'][0].endswith('shared: a fix')
    assert rows['sft']['state'] == 'not migrated'
    g(r, 'checkout', '-q', 'eval')
    sync.vendor(r, 'origin/shared')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'bump tbshared')
    g(r, 'update-ref', 'refs/remotes/origin/eval', 'eval')
    assert drift.report(('eval',))[0]['state'] == 'current'


def test_junk_files_are_ignored_and_missing_pin_check(repo):
    r, v1 = repo
    sync.vendor(r, v1)
    for junk in ('._a.py', '.DS_Store', 'a.py~', 'sub/.b.txt.swp'):
        (r / 'tbshared' / junk).write_text('x')
    assert sync.diff(r, v1) == []
    (r / 'tbshared' / 'PIN').unlink()
    assert sync.main(['--check', '--root', str(r)]) == 1


def test_aliases_are_written_and_checked(repo):
    r, v1 = repo
    (r / 'tbshared').mkdir()
    (r / 'tbshared' / 'ALIASES').write_text('old/place/a.py a.py   # partial snapshots copy old/place only\n')
    sync.vendor(r, v1)
    assert (r / 'old' / 'place' / 'a.py').read_text() == 'A = 1\n'
    assert (r / 'tbshared' / 'ALIASES').exists() and sync.diff(r, v1) == []
    (r / 'old' / 'place' / 'a.py').write_text('A = 9\n')
    assert sync.diff(r, v1) == ['alias differs: old/place/a.py <- a.py']
    (r / 'tbshared' / 'ALIASES').write_text('x.py nope.py\n')
    assert sync.diff(r, v1) == ['alias source not in the pin: x.py <- nope.py']


def test_vendor_refuses_unmerged_commits_and_the_shared_tree(repo, tmp_path):
    r, v1 = repo
    g(r, 'checkout', '-q', '-b', 'pr', 'shared')
    (r / 'tbshared' / 'a.py').write_text('A = 5\n')
    g(r, 'commit', '-qam', 'pr')
    pr = g(r, 'rev-parse', 'HEAD')
    g(r, 'checkout', '-q', 'eval')
    with pytest.raises(SystemExit):
        sync.vendor(r, pr)
    assert sync.vendor(r, pr, allow_unmerged=True) == pr
    g(r, 'checkout', '-q', '-f', 'shared')
    (r / 'notes.txt').write_text('untracked files do not make shared look like a pipeline')
    assert sync.is_shared_tree(r)
    with pytest.raises(SystemExit):
        sync.vendor(r, v1)


def test_drift_flags_a_pin_not_on_shared(repo, monkeypatch):
    r, v1 = repo
    g(r, 'checkout', '-q', '-b', 'pr', 'shared')
    (r / 'tbshared' / 'a.py').write_text('A = 5\n')
    g(r, 'commit', '-qam', 'pr')
    pr = g(r, 'rev-parse', 'HEAD')
    g(r, 'checkout', '-q', 'eval')
    sync.vendor(r, pr, allow_unmerged=True)
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'pin a PR commit')
    g(r, 'update-ref', 'refs/remotes/origin/eval', 'eval')
    monkeypatch.chdir(r)
    assert drift.report(('eval',))[0]['state'] == 'pin not on shared'


def test_alias_sources_must_be_self_contained_and_lines_well_formed(repo):
    r, v1 = repo
    g(r, 'checkout', '-q', 'shared')
    (r / 'tbshared' / 'uses.py').write_text('from tbshared import a\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'a module importing a sibling')
    v2 = g(r, 'rev-parse', 'HEAD')
    g(r, 'update-ref', 'refs/remotes/origin/shared', 'shared')
    g(r, 'checkout', '-q', 'eval')
    (r / 'tbshared').mkdir()
    (r / 'tbshared' / 'ALIASES').write_text('old/uses.py uses.py\n')
    sync.vendor(r, v2)
    assert sync.diff(r, v2) == ['alias source imports tbshared (not self-contained): old/uses.py <- uses.py']
    (r / 'tbshared' / 'ALIASES').write_text('just-one-field\n')
    with pytest.raises(SystemExit):
        sync.diff(r, v2)


def test_pipeline_json_is_local_kept_by_sync_and_ignored_by_the_check(repo):
    r, v1 = repo
    assert 'pipeline.json' in sync.LOCAL
    (r / 'tbshared').mkdir()
    (r / 'tbshared' / 'pipeline.json').write_text('{"pipeline": "eval"}\n')
    sync.vendor(r, v1)
    assert (r / 'tbshared' / 'pipeline.json').read_text() == '{"pipeline": "eval"}\n'   # kept across a bump
    assert sync.diff(r, v1) == [] and sync.main(['--check', '--root', str(r)]) == 0
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'eval: vendor tbshared + its pipeline.json')
    assert sync.diff(r, v1) == []                                   # committed in the pipeline: still not in the pin


def test_a_pipeline_json_committed_to_shared_is_never_vendored(repo):
    r, v1 = repo
    g(r, 'checkout', '-q', 'shared')
    (r / 'tbshared' / 'pipeline.json').write_text('{"pipeline": "shared"}\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'a pipeline.json on shared by mistake')
    v2 = g(r, 'rev-parse', 'HEAD')
    g(r, 'update-ref', 'refs/remotes/origin/shared', 'shared')
    assert 'pipeline.json' not in sync.pinned_tree(v2, cwd=r)
    g(r, 'checkout', '-q', 'eval')
    sync.vendor(r, v2)
    assert not (r / 'tbshared' / 'pipeline.json').exists() and sync.diff(r, v2) == []
    (r / 'tbshared' / 'pipeline.json').write_text('{"pipeline": "eval"}\n')
    sync.vendor(r, v2)
    assert (r / 'tbshared' / 'pipeline.json').read_text() == '{"pipeline": "eval"}\n'   # the pipeline's own wins


def test_trunk_is_canonical_and_checks_its_aliases(repo):
    """#480: on the trunk (tbshared/TRUNK) nothing is vendored; aliases are checked against tbshared/ itself."""
    r, v1 = repo
    g(r, 'checkout', '-q', '--orphan', 'main')
    g(r, 'rm', '-rfq', '.')
    (r / 'tbshared').mkdir()
    (r / 'tbshared' / 'a.py').write_text('A = 3  # edited on the trunk\n')
    (r / 'tbshared' / 'TRUNK').write_text('canonical\n')
    (r / 'tbshared' / 'ALIASES').write_text('pkg/a.py a.py\n')
    (r / 'pkg').mkdir()
    (r / 'pkg' / 'a.py').write_text('A = 3  # edited on the trunk\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'trunk')
    assert sync.is_trunk(r) and sync.trunk_diff(r) == []
    assert sync.main(['--check', '--root', str(r)]) == 0
    with pytest.raises(SystemExit, match='trunk'):
        sync.vendor(r, v1)
    (r / 'pkg' / 'a.py').write_text('A = 4  # a local fix in the alias\n')
    assert sync.trunk_diff(r) == ['alias differs: pkg/a.py <- a.py']
    assert sync.main(['--check', '--root', str(r)]) == 1
    (r / 'pkg' / 'a.py').write_text('A = 3  # edited on the trunk\n')
    (r / 'tbshared' / 'PIN').write_text(v1 + '\n')
    assert sync.trunk_diff(r) == ['tbshared/PIN on the trunk: the canonical copy is not vendored']
    (r / 'tbshared' / 'PIN').unlink()
    for f in (r / 'tbshared' / 'a.py', r / 'pkg' / 'a.py'):              # an alias source must stay self-contained
        f.write_text('from tbshared import b\n')
    assert sync.trunk_diff(r) == ['alias source imports tbshared (not self-contained): pkg/a.py <- a.py']


def test_drift_shows_what_differs_from_the_trunk(repo, monkeypatch):
    """#480: with main canonical, a fix on main shows on shared's row and on every branch that lacks it."""
    r, v1 = repo
    sync.vendor(r, v1)
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'vendor tbshared v1')
    g(r, 'update-ref', 'refs/remotes/origin/eval', 'eval')
    g(r, 'checkout', '-q', '-b', 'main')
    (r / 'tbshared' / 'PIN').unlink()
    (r / 'tbshared' / 'TRUNK').write_text('canonical\n')
    (r / 'tbshared' / 'a.py').write_text('A = 9  # a fix on the trunk\n')
    g(r, 'add', '-A')
    g(r, 'commit', '-qm', 'trunk')
    g(r, 'update-ref', 'refs/remotes/origin/main', 'main')
    monkeypatch.chdir(r)
    rows = {x['branch']: x for x in drift.report(('eval',))}
    assert rows['main']['state'] == 'trunk (canonical)'
    assert rows['main']['shared_differs_from_main'] == ['tbshared/a.py']
    assert rows['eval']['state'] == 'current' and rows['eval']['differs_from_main'] == ['tbshared/a.py']

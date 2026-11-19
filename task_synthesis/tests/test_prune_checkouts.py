"""prune-checkouts (#245): a frozen run checkout is removed only when it is old enough, no condor job, local process
or recent run metadata names it, it is clean except __pycache__, and its HEAD is on origin. A dry run by default."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from task_synthesis import common


def sh(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


class PruneCheckouts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.repo, self.root = t / 'repo', t / 'worktrees'
        self.root.mkdir()
        sh('git', 'init', '-q', '--bare', str(t / 'origin.git'))
        sh('git', 'init', '-q', str(self.repo))
        env = ('-c', 'user.name=t', '-c', 'user.email=t@t')
        (self.repo / 'a.py').write_text('x = 1\n')
        sh('git', *env, 'add', 'a.py', cwd=self.repo)
        sh('git', *env, 'commit', '-qm', 'one', cwd=self.repo)
        sh('git', 'remote', 'add', 'origin', str(t / 'origin.git'), cwd=self.repo)
        sh('git', 'push', '-q', 'origin', 'HEAD:main', cwd=self.repo)
        sh('git', 'fetch', '-q', 'origin', cwd=self.repo)
        (self.repo / 'a.py').write_text('x = 2\n')
        sh('git', *env, 'commit', '-qam', 'two (local only)', cwd=self.repo)
        heads = subprocess.run(['git', 'rev-list', 'HEAD'], cwd=self.repo, capture_output=True, text=True).stdout.split()
        self.local_sha, self.pushed_sha = heads
        self.c = {}
        for name, sha in [('old', self.pushed_sha), ('used', self.pushed_sha), ('dirty', self.pushed_sha),
                          ('unpushed', self.local_sha), ('young', self.pushed_sha), ('cache', self.pushed_sha)]:
            path = self.root / f'tasksynth-run-{name}'
            sh('git', 'worktree', 'add', '-q', '--detach', str(path), sha, cwd=self.repo)
            if name != 'young':
                old = time.time() - 5 * 86400
                os.utime(path / '.git', (old, old))
            self.c[name] = path
        (self.c['dirty'] / 'a.py').write_text('edited\n')
        (self.c['cache'] / '__pycache__').mkdir()
        (self.c['cache'] / '__pycache__' / 'a.cpython-312.pyc').write_bytes(b'\0')

    def tearDown(self):
        self.tmp.cleanup()

    def prune(self, remove, **kw):
        logs = []
        args = dict(ads='Iwd = "/x/tasksynth-run-used"\n', local='', metadata={}, log=logs.append)
        args.update(kw)
        return common.prune_checkouts(self.root, 2, 4, remove, **args), logs

    def test_dry_run_then_remove(self):
        verdicts, logs = self.prune(False)
        removable = sorted(p.name for p, w in verdicts.items() if not w)
        self.assertEqual(removable, ['tasksynth-run-cache', 'tasksynth-run-old'])
        why = {p.name: '; '.join(w) for p, w in verdicts.items() if w}
        self.assertIn('condor job', why['tasksynth-run-used'])
        self.assertIn('not clean', why['tasksynth-run-dirty'])
        self.assertIn('not on any origin branch', why['tasksynth-run-unpushed'])
        self.assertIn('days ago', why['tasksynth-run-young'])
        self.assertTrue(all(p.exists() for p in self.c.values()), 'a dry run removes nothing')
        self.prune(True)
        self.assertFalse(self.c['old'].exists() or self.c['cache'].exists())
        self.assertTrue(all(self.c[n].exists() for n in ('used', 'dirty', 'unpushed', 'young')))
        listed = subprocess.run(['git', 'worktree', 'list'], cwd=self.repo, capture_output=True, text=True).stdout
        self.assertNotIn('tasksynth-run-old', listed)

    def test_local_processes_metadata_and_an_unreachable_schedd_keep_it(self):
        verdicts, _ = self.prune(False, ads='', local=f'python -m x {self.c["old"]}',
                                 metadata={'/data/run.json': f'{{"code": "{self.c["cache"]}"}}'})
        self.assertIn('local process', '; '.join(verdicts[self.c['old']]))
        self.assertIn('recent run metadata', '; '.join(verdicts[self.c['cache']]))
        with mock.patch.object(common, 'condor_ads', lambda: None):
            verdicts = common.prune_checkouts(self.root, 2, 4, False, local='', metadata={}, log=lambda m: None)
        self.assertTrue(all(verdicts.values()), 'condor_q failing keeps every checkout')


if __name__ == '__main__':
    unittest.main()

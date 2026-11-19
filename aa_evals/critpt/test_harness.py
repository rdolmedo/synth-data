"""Sandbox wiring without Apptainer: /data/aai/aa_evals/venvs/critpt-stirrup021/bin/python -m unittest
aa_evals.critpt.test_harness. The end-to-end check (a file in /tmp survives to the next command) is
run.sandbox_selftest, which every CritPt job runs on its compute node before starting."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from aa_evals.critpt.harness import OfflineSandbox, sandbox_argv


class TmpBind(unittest.TestCase):
    def test_argv_binds_tmp_when_given(self):
        argv = sandbox_argv('x.sif', 'apptainer-net-none', '/s/w', 'true', '/s/w.tmp')
        i = argv.index('/s/w.tmp:/tmp')
        self.assertEqual(argv[i - 1], '-B')
        self.assertIn('/s/w:/work', argv)
        self.assertLess(i, argv.index('x.sif'))           # an apptainer option, before the image

    def test_argv_without_tmp_unchanged(self):
        self.assertFalse(any(x.endswith(':/tmp') for x in sandbox_argv('x.sif', 'apptainer-net-none', '/s/w', 'true')))

    def test_sandbox_tmp_dir_lifecycle(self):
        async def go(base):
            sb = OfflineSandbox(sif=Path('x.sif'), isolation='apptainer-net-none', work_base=base,
                                exec_slots=asyncio.Semaphore(1))
            async with sb:
                work, tmp = sb.temp_dir, sb.tmp_dir
                self.assertEqual(tmp.parent, work.parent)        # a sibling: /work does not show it
                self.assertEqual(tmp.name, work.name + '.tmp')
                argv = sb.wrap('echo hi')
                self.assertTrue(tmp.is_dir())
                self.assertIn(f'{tmp}:/tmp', argv)
                (tmp / 'f').write_text('x')
            self.assertFalse(work.exists())
            self.assertFalse(tmp.exists())                        # removed with the trajectory
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(go(Path(d)))


if __name__ == '__main__':
    unittest.main()

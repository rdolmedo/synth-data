"""critpt.run: reprobe jobs are sized for the probe sandbox (#378), generation keeps 16 CPU / 64 GB."""
import os
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest import mock

from aa_evals.critpt import run as C


class Shape(unittest.TestCase):
    def submitted(self, argv):
        got = {}
        with mock.patch.object(C, 'submit_condor', lambda a, argv, **k: got.update(cpus=a.cpus, mem=a.mem_gb)):
            C.main(argv)
        return got['cpus'], got['mem']

    def test_defaults(self):
        self.assertEqual(self.submitted(['--run-id', 'x', '--reprobe', '--condor']), (4, 32))
        self.assertEqual(self.submitted(['--run-id', 'x', '--reprobe', '--condor', '--cpus', '16', '--mem-gb', '64']), (16, 64))
        self.assertEqual(self.submitted(['--run-id', 'x', '--model', 'serving:m', '--condor']), (16, 64))

    def test_job_cpus_from_the_job_ad(self):
        ad = Path(mkdtemp()) / '.job.ad'
        ad.write_text('ClusterId = 5\nRequestCpus = 4\n')
        with mock.patch.dict(os.environ, {'_CONDOR_JOB_AD': str(ad)}):
            self.assertEqual(C.job_cpus(), min(4, len(os.sched_getaffinity(0))))
        with mock.patch.dict(os.environ, {'_CONDOR_JOB_AD': ''}):
            self.assertEqual(C.job_cpus(), len(os.sched_getaffinity(0)))


if __name__ == '__main__':
    unittest.main()

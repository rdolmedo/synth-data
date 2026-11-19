"""decontam.pool --production <run.json> judges on the run's servers: it reads the run's registry root (run.json
serving_env, recorded by task_synthesis.pipeline.production plan for a canary), not the shell's."""
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from decontam import pool
from tbshared import serving


class ProductionRoot(unittest.TestCase):
    def run_defaults(self, rec, env):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=False):
            for k in ('TASKSYNTH_SERVING_ROOT', 'DECONTAM_SERVING_ROOT'):
                if k not in env:
                    os.environ.pop(k, None)
            run = Path(d) / 'production' / 'c1'
            run.mkdir(parents=True)
            (run / 'run.json').write_text(json.dumps({'name': 'c1', 'round': 'c1', 'stream': str(Path(d) / 's'),
                                                      'tasks_needed': 1, 'servers': 1, **rec}))
            a = Namespace(production=str(run / 'run.json'), pool=None, expected_tasks=None, checkpoints=None,
                          snapshot_when=None, serving=None, hold=None, publish=None, admission=None)
            try:
                pool.production_defaults(a)
            except (AttributeError, KeyError, TypeError, OSError):
                pass   # later defaults need more of a real run; the root is pinned first
            return os.environ.get('TASKSYNTH_SERVING_ROOT')

    def test_canary_run_from_a_fresh_shell(self):
        self.assertEqual(self.run_defaults({'serving_env': {'TASKSYNTH_SERVING_ROOT': '/x/canary'}}, {}), '/x/canary')

    def test_production_run_clears_and_conflicts_refuse(self):
        self.assertIsNone(self.run_defaults({}, {}))
        self.assertIsNone(self.run_defaults({}, {'TASKSYNTH_SERVING_ROOT': str(serving.ROOT)}))
        with self.assertRaisesRegex(SystemExit, 'planned on registry'):
            self.run_defaults({}, {'TASKSYNTH_SERVING_ROOT': '/x/canary'})
        with self.assertRaisesRegex(SystemExit, 'disagree'):   # the legacy name must agree with the run's root
            self.run_defaults({'serving_env': {'TASKSYNTH_SERVING_ROOT': '/x/canary'}}, {'DECONTAM_SERVING_ROOT': '/x/old'})
        with self.assertRaisesRegex(SystemExit, 'unset it'):   # nor move a production run's judge to a canary
            self.run_defaults({}, {'DECONTAM_SERVING_ROOT': '/x/canary'})
        self.assertIsNone(self.run_defaults({}, {'DECONTAM_SERVING_ROOT': str(serving.ROOT)}))


if __name__ == '__main__':
    unittest.main()

"""synth.datasets.validation_status: verdicts from no-agent nop sweeps."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from synth import datasets


def trial(run, name, task, reward=None, error=None, message='', tests=1):
    d = Path(run) / name
    (d / 'verifier').mkdir(parents=True)
    (d / 'verifier' / 'ctrf.json').write_text(json.dumps({'results': {'summary': {'tests': tests}}}))
    (d / 'result.json').write_text(json.dumps({
        'trial_name': name, 'task_name': task, 'finished_at': '2026-10-27T00:00:00',
        'verifier_result': {'rewards': {'reward': reward}} if reward is not None else None,
        'exception_info': {'exception_type': error, 'exception_message': message} if error else None}))


class ValidationStatus(unittest.TestCase):
    def status(self, runs, **kw):
        out = Path(self.d) / 'validation.json'
        with contextlib.redirect_stdout(io.StringIO()):
            datasets.validation_status(runs, 'nop', out, **kw)
        return json.loads(out.read_text())

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = self._tmp.name
        a, b = f'{self.d}/validate-w1', f'{self.d}/validate-w2'
        trial(a, 'ok__1', 'ok', reward=0)
        trial(a, 'solved__1', 'solved', reward=1)
        trial(a, 'noreward__1', 'noreward', error='RewardFileNotFoundError')
        trial(a, 'unbuilt__1', 'unbuilt', error='BuildError',
              message='local base local/rts-prefix-0001:1 is not published at /data/x')
        trial(a, 'flaky__1', 'flaky', reward=0, tests=0)  # broke before testing
        trial(b, 'flaky__2', 'flaky', reward=0)
        self.runs = [f'{self.d}/validate-w*']

    def tearDown(self):
        datasets._RESULTS.clear()
        self._tmp.cleanup()

    def test_default_one_failure_is_invalid(self):
        self.assertEqual(self.status(self.runs), {'ok': True, 'flaky': True, 'solved': False, 'noreward': False})

    def test_min_failures_defers_but_nop_solved_is_final(self):
        self.assertEqual(self.status(self.runs, min_failures=2), {'ok': True, 'flaky': True, 'solved': False})

    def test_assume_valid_never_overrides_a_verdict(self):
        v = self.status(self.runs, min_failures=2, assume_valid=['unbuilt', 'solved', 'noreward'])
        self.assertEqual((v['unbuilt'], v['noreward'], v['solved']), (True, True, False))


if __name__ == '__main__':
    unittest.main()

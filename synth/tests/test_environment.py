"""synth.environment: verifier runs that failed on the proxy are re-run on the same sandbox."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from synth.environment import verify_with_retries


class VerifierRetries(unittest.TestCase):
    def run_verifier(self, outputs, retries):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stdout, reward, log = d / 'test-stdout.txt', d / 'reward.txt', d / 'infra-retries.jsonl'
            calls, sleeps = [], []

            async def run():
                text, value = outputs[len(calls)]
                calls.append(text)
                stdout.write_text(text)
                reward.write_text(value)
                return len(calls)

            async def sleep(seconds):
                sleeps.append(seconds)

            result = asyncio.run(verify_with_retries(run, stdout, [reward], retries, log, sleep=sleep))
            kept = sorted(p.name for p in d.glob('test-stdout.infra-*.txt'))
            logged = [json.loads(line)['marker'] for line in log.read_text().splitlines()] if log.exists() else []
            return result, reward.read_text(), kept, logged, sleeps

    def test_retries_a_proxy_failure_then_keeps_the_real_result(self):
        proxy = ('curl: (7) Failed to connect to 10.22.0.107 port 8080\n/tests/test.sh: uvx: command not found', '0')
        result, reward, kept, logged, sleeps = self.run_verifier([proxy, ('1 passed', '1')], retries=2)
        self.assertEqual((result, reward, kept, sleeps), (2, '1', ['test-stdout.infra-0.txt'], [60]))
        self.assertEqual(logged, ['Failed to connect to'])

    def test_tests_that_started_are_never_rerun_despite_proxy_noise(self):
        noisy = ('[synth verifier] pytest did not start (try 1, network failure): Connection reset by peer\n'
                 '============================= test session starts ==============================\n1 failed', '0')
        self.assertEqual(self.run_verifier([noisy], retries=3)[:3], (1, '0', []))

    def test_a_real_failure_runs_once_and_retries_are_bounded(self):
        self.assertEqual(self.run_verifier([('1 failed', '0')], retries=2)[:3], (1, '0', []))
        proxy = ('Connection reset by peer', '0')
        result, reward, kept, _, sleeps = self.run_verifier([proxy, proxy, proxy], retries=2)
        self.assertEqual((result, reward, len(kept), sleeps), (3, '0', 2, [60, 120]))
        self.assertEqual(self.run_verifier([proxy], retries=0)[:3], (1, '0', []))   # off by default


if __name__ == '__main__':
    unittest.main()

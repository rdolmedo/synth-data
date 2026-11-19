"""Score-only regression fixtures: no native worker, generation or source writes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aa_evals.common.pi_eval import score


class GradingPendingScoreTests(unittest.TestCase):
    def test_typed_pending_keeps_no_score_even_with_raw_reward(self):
        cases = {
            ('caffe-cifar-10', 0): {'key': 'caffe-cifar-10__a0__r0', 'reward': 1, 'score': None,
                                  'grading_review_required': True, 'grading_pending': True},
            ('normal', 0): {'key': 'normal__a0__r0', 'reward': 1, 'score': 1},
            ('real-infra', 0): {'key': 'real-infra__a0__r0', 'infrastructure_failure': True,
                               'grading_review_required': True, 'grading_pending': True},
            ('legacy-unscored', 0): {'key': 'legacy-unscored__a0__r0', 'unscored': True},
            ('no-outcome', 0): None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text(json.dumps({'code': str(root / 'unused')}))
            state = SimpleNamespace(logical=lambda: cases)
            old_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {'queue_state': SimpleNamespace(State=lambda run: state)}):
                    rows = score.rows(root)
            finally:
                sys.path[:] = old_path
        by_id = {r['item_id']: r for r in rows}
        self.assertEqual((by_id['caffe-cifar-10']['status'], by_id['caffe-cifar-10']['score']), ('grading_pending', None))
        self.assertEqual(by_id['caffe-cifar-10']['reward'], 1)
        self.assertEqual(by_id['normal']['score'], 1)
        self.assertEqual(by_id['real-infra']['status'], 'infra_error')
        self.assertEqual(by_id['legacy-unscored']['status'], 'ok')
        self.assertEqual(by_id['legacy-unscored']['score'], 0)
        self.assertEqual(by_id['no-outcome']['status'], 'pending')

    def test_pending_blocks_primary_ci_valid_complete_without_becoming_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / 'oracle'
            oracle.mkdir()
            (oracle / 'items.jsonl').write_text(json.dumps({'item_id': 'caffe-cifar-10', 'status': 'ok', 'score': 0}) + '\n')
            (root / 'config.json').write_text(json.dumps({'benchmark': 'tb21', 'attempts': 1,
                                                        'oracle_run': str(oracle)}))
            items = [{'item_id': 'normal', 'repeat': 0, 'status': 'ok', 'score': 1},
                     {'item_id': 'caffe-cifar-10', 'repeat': 0, 'status': 'grading_pending', 'score': None}]
            summary = score.summarize(root, items)
            self.assertIsNone(summary['score'])
            self.assertIsNone(summary['ci95'])
            self.assertIsNone(summary['score_excl_oracle_failures'])
            self.assertEqual(summary['score_resolved_only'], 100)
            self.assertEqual(summary['n_pending_grading'], 1)
            self.assertEqual(summary['n_pending'], 0)
            self.assertEqual(summary['n_cutoff'], 0)
            self.assertEqual(summary['n_items'], 2)
            self.assertEqual(summary['n_infra_errors'], 0)
            self.assertFalse(summary['valid'])
            self.assertFalse(summary['complete'])
            # A later grading-only resolution restores the normal result; no
            # queue/generation call or modification is involved in this test.
            items[1] = dict(items[1], status='ok', score=0)
            resolved = score.summarize(root, items)
            self.assertEqual(resolved['score'], 50)
            self.assertIsNotNone(resolved['ci95'])
            self.assertEqual(resolved['n_pending_grading'], 0)
            self.assertTrue(resolved['valid'])
            self.assertTrue(resolved['complete'])

    def test_malformed_metadata_does_not_forge_a_typed_pending_outcome(self):
        for review, pending in [(True, 'true'), ('true', True), (1, 1), (True, None)]:
            with self.subTest(review=review, pending=pending), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'config.json').write_text(json.dumps({'code': str(root)}))
                rec = {'grading_review_required': review, 'grading_pending': pending, 'score': 1}
                old_path = list(sys.path)
                try:
                    state = SimpleNamespace(logical=lambda: {('task', 0): rec})
                    with patch.dict(sys.modules, {'queue_state': SimpleNamespace(State=lambda run: state)}):
                        rows = score.rows(root)
                finally:
                    sys.path[:] = old_path
                self.assertEqual(rows[0]['status'], 'ok')
                self.assertEqual(rows[0]['score'], 1)


if __name__ == '__main__':
    unittest.main()

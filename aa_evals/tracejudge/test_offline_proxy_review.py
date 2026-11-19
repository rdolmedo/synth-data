"""Offline package-index review is source-backed and independent of LLM flags."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from .core import build_row
from .hook import counts, fold_summary
from .layouts import _offline_proxy_review
from .trace import Trace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class OfflineProxyReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.key = 'task__a0__r0'
        self.attempt = self.root / 'attempts' / self.key
        self.attempt.mkdir(parents=True)
        (self.attempt / 'attempt.json').write_text('{"generation":"immutable"}')
        (self.attempt / 'trial.tar.gz').write_bytes(b'immutable synthetic generation archive')
        self.bundle = self.root / 'bundle.tar'
        self.instruction = b'Original synthetic package-neutral scientific task.'
        with tarfile.open(self.bundle, 'w') as tf:
            member = tarfile.TarInfo('tasks/task/instruction.md')
            member.size = len(self.instruction)
            tf.addfile(member, io.BytesIO(self.instruction))
        self.config = {'bundle': str(self.bundle), 'bundle_sha256': sha(self.bundle)}
        self.outcome = {'score': 0, 'infrastructure_failure': False,
                        'offline_proxy_review_required': True,
                        'offline_proxy_review_evidence': [{'tool_call_id': 'call1',
                            'command': 'pip install optional-package', 'tool_result_sha256': 'a' * 64,
                            'reason': 'default_pip_index_not_proven'}]}
        self.image = self.root / 'actual-source-image.sqfs'
        self.image.write_bytes(b'unchanged synthetic native task image')
        self.audit_path = self.root / 'index-audit.json'
        self.receipt_path = self.attempt / 'offline-proxy-review.json'
        bindings = {'archive_sha256': sha(self.attempt / 'trial.tar.gz'),
                    'attempt_sha256': sha(self.attempt / 'attempt.json'),
                    'marker_evidence_sha256': digest(self.outcome['offline_proxy_review_evidence']),
                    'source_image_sha256': sha(self.image),
                    'instruction_sha256': hashlib.sha256(self.instruction).hexdigest(),
                    'bundle_sha256': sha(self.bundle)}
        self.audit = {'schema_version': 1, 'key': self.key, 'verdict': 'expected_policy_denial',
                      'bindings': bindings, 'public_default_index': True,
                      'indexes': ['https://pypi.org/simple'],
                      'evidence': 'Synthetic source audit: pip default source is PyPI; no config/env index overrides.'}
        self.receipt = {'schema_version': 1, 'key': self.key, 'verdict': 'expected_policy_denial',
                        **{k: bindings[k] for k in ('archive_sha256', 'attempt_sha256', 'marker_evidence_sha256')},
                        'source_image': {'path': str(self.image), 'sha256': sha(self.image)},
                        'instruction': {'member': 'tasks/task/instruction.md',
                                        'bundle_sha256': sha(self.bundle),
                                        'sha256': bindings['instruction_sha256']}}

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, receipt=None, audit=None):
        self.audit_path.write_text(json.dumps(self.audit if audit is None else audit))
        receipt = copy.deepcopy(self.receipt if receipt is None else receipt)
        field = {1: 'index_audit', 2: 'source_audit', 3: 'host_probe_audit'}[receipt['schema_version']]
        receipt[field] = {'path': str(self.audit_path), 'sha256': sha(self.audit_path)}
        self.receipt_path.write_text(json.dumps(receipt))

    def review(self):
        return _offline_proxy_review(self.root, self.key, self.outcome, self.config)

    def source_manager(self, manager):
        self.outcome['offline_proxy_review_evidence'][0]['reason'] = {
            'apt': 'apt_effective_sources_not_proven', 'git': 'git_effective_urls_not_proven'}[manager]
        binding = digest(self.outcome['offline_proxy_review_evidence'])
        self.receipt['marker_evidence_sha256'] = binding
        self.audit['bindings']['marker_evidence_sha256'] = binding
        self.receipt['schema_version'] = self.audit['schema_version'] = 2
        self.audit['sources'] = [{'manager': manager, 'public_effective_sources': True,
            'urls': ['https://github.com/public/repository'] if manager == 'git' else ['http://deb.debian.org/debian'],
            'evidence': 'Synthetic pinned source image/config and prior agent overrides reviewed.'}]

    def host_probe(self):
        bridge = 'http://10.22.2.196:45753/v1'
        raw = ('{ "agent": {"env":{"OPENAI_BASE_URL":' + json.dumps(bridge) + '}} }\n').encode()
        relay_sha = hashlib.sha256(raw).hexdigest()
        member_name = 'task__a0__r0/task__a0__r0/trial/config.json'
        with tarfile.open(self.attempt / 'trial.tar.gz', 'w:gz') as tf:
            member = tarfile.TarInfo(member_name)
            member.size = len(raw)
            tf.addfile(member, io.BytesIO(raw))
        targets = ['http://10.22.0.107:8080', 'http://10.22.2.196:45753/v1/models', 'tcp://10.22.2.196:45753']
        self.outcome['offline_proxy_review_evidence'] = [{'reason': 'agent_host_egress_unresolved',
            'bridge_url': bridge, 'bridge_config_sha256': relay_sha, 'known_proxy_url': 'http://10.22.0.107:8080',
            'probe_targets': targets, 'tool_call_id': 'call1', 'command': 'synthetic bounded known-host probes',
            'tool_result_sha256': 'a' * 64}]
        self.receipt.update(schema_version=3, verdict='expected_agent_host_denial',
            archive_sha256=sha(self.attempt / 'trial.tar.gz'), marker_evidence_sha256=digest(self.outcome['offline_proxy_review_evidence']),
            relay_config={'member': member_name, 'sha256': relay_sha})
        self.audit.update(schema_version=3, verdict='expected_agent_host_denial', sources=[],
            probe_targets=targets, known_bridge_url=bridge, known_proxy_url='http://10.22.0.107:8080',
            model_requests_continued=True, non_task_host_probes=True, required_task_service_failure=False)
        self.audit['bindings'].update(archive_sha256=self.receipt['archive_sha256'],
            marker_evidence_sha256=self.receipt['marker_evidence_sha256'], relay_config_sha256=relay_sha,
            host_probe_targets_sha256=digest(targets))

    def test_host_probe_review_is_pending_without_exact_separate_root_attestation(self):
        self.host_probe()
        self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.native()
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        for version in (1, 2):
            receipt, audit = copy.deepcopy(self.receipt), copy.deepcopy(self.audit)
            receipt.update(schema_version=version, verdict='expected_policy_denial')
            audit.update(schema_version=version, verdict='expected_policy_denial')
            self.publish(receipt, audit)
            self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.publish()
        record = self.review()
        self.assertFalse(record['offline_proxy_review_pending'], record)
        self.assertEqual(record['offline_proxy_review_status'], 'resolved_expected_agent_host_denial')
        self.assertEqual(record['offline_proxy_review_host_probe_audit_sha256'], sha(self.audit_path))
        self.assertIn('offline_proxy_review_host_probe_audit_sha256', self.row(record)['recorded'])

    def test_audit_schema_version_requires_an_exact_integer_not_bool_or_float(self):
        for value in (True, 1.0):
            with self.subTest(version=value):
                audit = copy.deepcopy(self.audit)
                audit['schema_version'] = value
                self.publish(audit=audit)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.source_manager('git')
        audit = copy.deepcopy(self.audit)
        audit['schema_version'] = 2.0
        self.publish(audit=audit)
        self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.host_probe()
        audit = copy.deepcopy(self.audit)
        audit['schema_version'] = 3.0
        self.publish(audit=audit)
        self.assertTrue(self.review()['offline_proxy_review_pending'])

    def test_host_probe_forged_stale_malformed_and_private_task_receipts_remain_pending(self):
        self.host_probe()
        for change in ('stale-relay', 'missing-relay', 'bad-relay-type', 'wrong-bridge', 'wrong-proxy',
                       'no-model-continuation', 'task-service', 'required-service-failure', 'stale-targets'):
            with self.subTest(change=change):
                receipt, audit = copy.deepcopy(self.receipt), copy.deepcopy(self.audit)
                if change == 'stale-relay':receipt['relay_config']['sha256'] = '0' * 64
                elif change == 'missing-relay':receipt['relay_config']['member'] = 'outer/config.json'
                elif change == 'bad-relay-type':receipt['relay_config'] = []
                elif change == 'wrong-bridge':audit['known_bridge_url'] = 'http://localhost:8080/v1'
                elif change == 'wrong-proxy':audit['known_proxy_url'] = 'http://service.internal:8080'
                elif change == 'no-model-continuation':audit['model_requests_continued'] = False
                elif change == 'task-service':audit['non_task_host_probes'] = False
                elif change == 'required-service-failure':audit['required_task_service_failure'] = True
                else:audit['probe_targets'] = []
                self.publish(receipt, audit)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
        # Even freshly hash-bound manual metadata cannot approve a different
        # private task target under this narrowly scoped host-probe schema.
        self.outcome['offline_proxy_review_evidence'][0]['probe_targets'].append('http://127.0.0.1:8080/simple')
        evidence_sha = digest(self.outcome['offline_proxy_review_evidence'])
        self.receipt['marker_evidence_sha256'] = evidence_sha
        self.audit['bindings'].update(marker_evidence_sha256=evidence_sha,
            host_probe_targets_sha256=digest(sorted(self.outcome['offline_proxy_review_evidence'][0]['probe_targets'])))
        self.audit['probe_targets'] = sorted(self.outcome['offline_proxy_review_evidence'][0]['probe_targets'])
        self.publish()
        self.assertTrue(self.review()['offline_proxy_review_pending'])

    def test_host_receipt_does_not_silently_approve_separate_apt_source_marker(self):
        self.host_probe()
        self.outcome['offline_proxy_review_evidence'].append({'reason': 'apt_effective_sources_not_proven',
            'tool_call_id': 'apt-call', 'command': 'apt-get install -y git', 'tool_result_sha256': 'b' * 64})
        evidence_sha = digest(self.outcome['offline_proxy_review_evidence'])
        self.receipt['marker_evidence_sha256'] = evidence_sha
        self.audit['bindings']['marker_evidence_sha256'] = evidence_sha
        self.publish()
        self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.audit['sources'] = [{'manager': 'apt', 'public_effective_sources': True,
            'urls': ['http://archive.ubuntu.com/ubuntu'], 'evidence': 'Pinned original apt sources plus prior changes reviewed.'}]
        self.publish()
        self.assertFalse(self.review()['offline_proxy_review_pending'])

    def test_git_and_apt_source_markers_need_hash_bound_manager_specific_receipts(self):
        for manager in ('git', 'apt'):
            with self.subTest(manager=manager):
                self.source_manager(manager)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
                self.publish()
                resolved = self.review()
                self.assertFalse(resolved['offline_proxy_review_pending'], resolved)
                self.assertEqual(resolved['offline_proxy_review_source_audit_sha256'], sha(self.audit_path))
                self.assertIn('offline_proxy_review_source_audit_sha256', self.row(resolved)['recorded'])
                legacy = copy.deepcopy(self.receipt)
                legacy['schema_version'] = 1
                old_audit = copy.deepcopy(self.audit)
                old_audit['schema_version'] = 1
                self.publish(legacy, old_audit)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
                self.receipt_path.unlink()

    def test_source_review_missing_duplicate_private_and_false_manager_proofs_remain_pending(self):
        self.source_manager('git')
        for change in ('missing', 'duplicate', 'wrong-manager', 'private', 'false', 'no-evidence'):
            with self.subTest(change=change):
                audit = copy.deepcopy(self.audit)
                if change == 'missing':audit['sources'] = []
                elif change == 'duplicate':audit['sources'].append(copy.deepcopy(audit['sources'][0]))
                elif change == 'wrong-manager':audit['sources'][0]['manager'] = 'apt'
                elif change == 'private':audit['sources'][0]['urls'] = ['http://localhost:8080/simple']
                elif change == 'false':audit['sources'][0]['public_effective_sources'] = False
                else:audit['sources'][0]['evidence'] = ''
                self.publish(audit=audit)
                self.assertTrue(self.review()['offline_proxy_review_pending'])

    def row(self, recorded):
        trace = Trace('tbsci_lite', self.key, 'task', 0, 'agentic', {'status': 'ok', **recorded})
        return build_row(trace, [], 'clean', {'ok': True, 'flags': []}, None, None)

    def test_marker_requires_review_even_if_llm_is_clean(self):
        record = self.review()
        row = self.row(record)
        self.assertFalse(row['flagged'])
        summary = counts({self.key: row})
        self.assertEqual(summary['n_offline_proxy_review_pending'], 1)
        self.assertTrue(record['offline_proxy_review_required'])
        # Legacy marker-only flags also fail closed without pending=False proof.
        row['recorded'].pop('offline_proxy_review_pending')
        self.assertEqual(counts({self.key: row})['n_offline_proxy_review_pending'], 1)

    def test_valid_receipt_clears_pending_and_preserves_raw_marker_and_generation(self):
        before = {name: (self.attempt / name).read_bytes() for name in ('attempt.json', 'trial.tar.gz')}
        original = copy.deepcopy(self.outcome)
        self.publish()
        record = self.review()
        self.assertFalse(record['offline_proxy_review_pending'])
        self.assertTrue(record['offline_proxy_review_required'])
        self.assertEqual(record['offline_proxy_review_receipt_sha256'], sha(self.receipt_path))
        self.assertEqual(counts({self.key: self.row(record)})['n_offline_proxy_review_pending'], 0)
        self.assertEqual(counts({self.key: self.row(record)})['n_offline_proxy_review_required'], 1)
        self.assertEqual(self.outcome, original)
        self.assertEqual(before, {name: (self.attempt / name).read_bytes() for name in before})

    def test_forged_cross_attempt_archive_evidence_image_instruction_and_audit_reject(self):
        for field in ('key', 'archive_sha256', 'attempt_sha256', 'marker_evidence_sha256'):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.receipt)
                bad[field] = 'forged'
                self.publish(bad)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
        for field in ('source_image', 'instruction'):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.receipt)
                bad[field]['sha256'] = '0' * 64
                self.publish(bad)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
        audit = copy.deepcopy(self.audit)
        audit['bindings']['source_image_sha256'] = '0' * 64
        self.publish(audit=audit)
        self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.publish()
        self.audit_path.write_text('{}')
        self.assertTrue(self.review()['offline_proxy_review_pending'])

    def native(self):
        (self.root / 'config.json').write_text(json.dumps(self.config))
        (self.attempt / 'outcome.json').write_text(json.dumps(self.outcome))

    def test_native_marker_is_discovered_without_any_flags(self):
        self.native()
        result = fold_summary(self.root)
        self.assertEqual(result['n_offline_proxy_review_required'], 1)
        self.assertEqual(result['n_offline_proxy_review_pending'], 1)
        self.assertIsNone(result['n_flagged'])

    def test_partial_missing_or_malformed_flag_coverage_cannot_hide_native_marker(self):
        self.native()
        for row in [{'key': 'other__a0__r0', 'counted': True, 'flagged': False},
                    {'key': self.key, 'counted': False, 'flagged': False},
                    {'key': 'other__a0__r0', 'recorded': ['malformed'], 'flagged': False}]:
            with self.subTest(row=row):
                (self.root / 'flags.jsonl').write_text(json.dumps(row) + '\n')
                self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        (self.root / 'flags.jsonl').write_text('malformed json\n[]\n')
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)

    def test_missing_or_unmarked_native_source_cannot_reuse_stale_approval(self):
        self.native()
        self.publish()
        record = self.review()
        self.assertFalse(record['offline_proxy_review_pending'])
        (self.root / 'flags.jsonl').write_text(json.dumps(self.row(record)) + '\n')
        (self.attempt / 'outcome.json').unlink()
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        (self.attempt / 'outcome.json').write_text(json.dumps({'score': 0, 'infrastructure_failure': False}))
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        # The host receipt filename retains the required-source anchor even if flags disappear.
        (self.root / 'flags.jsonl').unlink()
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)

    def test_stale_marker_in_concrete_infra_prefix_remains_audit_only(self):
        self.native()
        record = self.review()
        (self.root / 'flags.jsonl').write_text(json.dumps(self.row(record)) + '\n')
        (self.attempt / 'outcome.json').write_text(json.dumps({'infrastructure_failure': True}))
        later = self.root / 'attempts/task__a0__r1'
        later.mkdir()
        (later / 'outcome.json').write_text(json.dumps({'score': 0, 'infrastructure_failure': False}))
        result = fold_summary(self.root)
        self.assertEqual(result['n_offline_proxy_review_pending'], 0)
        self.assertEqual(result['n_offline_proxy_review_pending_physical'], 1)

    def test_valid_receipt_resolves_native_marker_without_a_trace_row(self):
        self.native()
        self.publish()
        result = fold_summary(self.root)
        self.assertEqual(result['n_offline_proxy_review_required'], 1)
        self.assertEqual(result['n_offline_proxy_review_pending'], 0)
        self.assertIsNone(result['n_flagged'])  # audit completeness does not manufacture trace coverage

    def test_native_retry_selection_ignores_forged_flag_counted_fields(self):
        self.native()
        later = self.root / 'attempts/task__a0__r1'
        later.mkdir()
        (later / 'outcome.json').write_text(json.dumps({'score': 0, 'infrastructure_failure': False}))
        (self.root / 'flags.jsonl').write_text(json.dumps({'key': self.key, 'counted': False, 'flagged': False}) + '\n')
        # Earlier marked noninfra remains the selected generation despite a later clean retry.
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        # A real backend infra prefix does not count once a later noninfra retry resolves.
        infra = {**self.outcome, 'infrastructure_failure': True, 'infrastructure_reason': 'pi_backend_unreachable'}
        (self.attempt / 'outcome.json').write_text(json.dumps(infra))
        result = fold_summary(self.root)
        self.assertEqual(result['n_offline_proxy_review_pending'], 0)
        self.assertEqual(result['n_offline_proxy_review_pending_physical'], 1)
        (self.attempt / 'outcome.json').write_text('[]')
        self.assertGreater(fold_summary(self.root)['n_offline_proxy_review_pending'], 0)

    def test_fold_revalidates_current_receipt_even_if_stored_llm_row_was_clean(self):
        self.publish()
        record = self.review()
        self.assertFalse(record['offline_proxy_review_pending'])
        (self.root / 'config.json').write_text(json.dumps(self.config))
        (self.attempt / 'outcome.json').write_text(json.dumps(self.outcome))
        (self.root / 'flags.jsonl').write_text(json.dumps(self.row(record)) + '\n')
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 0)
        self.receipt_path.write_text('{}')
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)
        self.publish()
        (self.attempt / 'trial.tar.gz').write_bytes(b'forged changed generation')
        self.assertEqual(fold_summary(self.root)['n_offline_proxy_review_pending'], 1)

    def test_duplicate_receipt_fields_or_private_mixed_index_cannot_approve(self):
        self.publish()
        text = self.receipt_path.read_text()
        self.receipt_path.write_text(text[:-1] + ', "verdict":"expected_policy_denial"}')
        self.assertIn('duplicate', self.review()['offline_proxy_review_error'])
        for indexes in [['https://pypi.org/simple'] * 2,
                        ['https://pypi.org/simple', 'http://127.0.0.1/simple'],
                        ['http://packages.internal/simple'], []]:
            with self.subTest(indexes=indexes):
                audit = copy.deepcopy(self.audit)
                audit['indexes'] = indexes
                self.publish(audit=audit)
                self.assertTrue(self.review()['offline_proxy_review_pending'])
        self.receipt_path.write_text(json.dumps([self.receipt, self.receipt]))
        self.assertTrue(self.review()['offline_proxy_review_pending'])


if __name__ == '__main__':
    unittest.main()

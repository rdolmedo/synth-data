"""Unknown tool origins need a bound manual review, never an automatic exemption."""
import copy
import hashlib
import io
import json
import tarfile
import unittest

from . import test_offline_proxy_review as legacy

sha, digest = legacy.sha, legacy.digest


class ToolNetworkReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = legacy.OfflineProxyReviewTests()
        fixture.setUp()
        self.f = fixture
        self.addCleanup(fixture.tearDown)
        self.bridge = 'http://10.22.2.196:45753/v1'
        self.relay = json.dumps({'agent': {'env': {'OPENAI_BASE_URL': self.bridge}}}).encode()
        self.command = 'cat /etc/apt/sources.list; timeout 5 apt-get update'
        events = [
            {'type': 'message', 'message': {'role': 'assistant', 'content': [{'type': 'toolCall',
                'id': 'call1', 'name': 'bash', 'arguments': {'command': self.command}}]}},
            {'type': 'message', 'message': {'role': 'toolResult', 'toolName': 'bash',
                'toolCallId': 'call1', 'content': [{'type': 'text', 'text': 'ProxyError: Cannot connect to proxy'}]}},
            {'type': 'message', 'message': {'role': 'assistant', 'api': 'openai-completions',
                'provider': 'harbor-endpoint', 'responseId': 'chatcmpl-real', 'usage': {'output': 42},
                'stopReason': 'stop', 'content': [{'type': 'text', 'text': 'Finished'}]}}]
        self.lines = [json.dumps(event) for event in events]
        self.session = ('\n'.join(self.lines) + '\n').encode()
        with tarfile.open(fixture.attempt / 'trial.tar.gz', 'w:gz') as tf:
            for name, raw in [('trial/config.json', self.relay), ('trial/agent/pi/sessions/actual.jsonl', self.session)]:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                tf.addfile(member, io.BytesIO(raw))
        marker = {'reason': 'agent_tool_network_origin_not_proven', 'command': self.command,
            'tool_call_id': 'call1', 'tool_result_sha256': hashlib.sha256(self.lines[1].encode()).hexdigest(),
            'bridge_url': self.bridge, 'bridge_config_sha256': hashlib.sha256(self.relay).hexdigest(),
            'continuation': {'line_index': 2, 'response_id': 'chatcmpl-real',
                'event_sha256': hashlib.sha256(self.lines[2].encode()).hexdigest()}}
        fixture.outcome['offline_proxy_review_evidence'] = [marker]
        bindings = fixture.audit['bindings']
        bindings.update(archive_sha256=sha(fixture.attempt / 'trial.tar.gz'),
                        marker_evidence_sha256=digest([marker]))
        fixture.receipt.update(schema_version=4, verdict='expected_agent_tool_denial',
            archive_sha256=bindings['archive_sha256'], marker_evidence_sha256=bindings['marker_evidence_sha256'],
            relay_config={'member': 'trial/config.json', 'sha256': hashlib.sha256(self.relay).hexdigest()},
            session={'member': 'trial/agent/pi/sessions/actual.jsonl', 'sha256': hashlib.sha256(self.session).hexdigest()})
        bindings.update(relay_config_sha256=hashlib.sha256(self.relay).hexdigest(),
                        session_sha256=hashlib.sha256(self.session).hexdigest())
        fixture.audit.update(schema_version=4, verdict='expected_agent_tool_denial',
            linked_agent_tool_observations_only=True, model_requests_continued=True,
            all_effective_sources_and_prior_mutations_reviewed=True, primary_transport_failure=False,
            evaluator_failure=False, required_task_service_failure=False, unresolved_source_contradiction=False)

    def publish(self):
        f = self.f
        f.audit_path.write_text(json.dumps(f.audit))
        f.receipt['tool_network_audit'] = {'path': str(f.audit_path), 'sha256': sha(f.audit_path)}
        f.receipt_path.write_text(json.dumps(f.receipt))

    def test_exact_review_accepts_without_changing_originals(self):
        f = self.f
        original = [sha(f.attempt / name) for name in ['attempt.json', 'trial.tar.gz']]
        self.assertTrue(f.review()['offline_proxy_review_pending'])
        self.publish()
        result = f.review()
        self.assertFalse(result['offline_proxy_review_pending'], result)
        self.assertEqual(result['offline_proxy_review_status'], 'resolved_expected_agent_tool_denial')
        self.assertEqual(original, [sha(f.attempt / name) for name in ['attempt.json', 'trial.tar.gz']])

    def test_legacy_or_incomplete_audit_cannot_clear(self):
        f = self.f
        original_receipt, original_audit = copy.deepcopy(f.receipt), copy.deepcopy(f.audit)
        for version in [1, 2, 3, True]:
            f.receipt = copy.deepcopy(original_receipt)
            f.receipt['schema_version'] = version
            self.publish()
            self.assertTrue(f.review()['offline_proxy_review_pending'])
        f.receipt = original_receipt
        for field in ['linked_agent_tool_observations_only', 'model_requests_continued',
                      'all_effective_sources_and_prior_mutations_reviewed', 'primary_transport_failure',
                      'evaluator_failure', 'required_task_service_failure', 'unresolved_source_contradiction']:
            with self.subTest(field=field):
                f.audit = copy.deepcopy(original_audit)
                f.audit[field] = not f.audit[field]
                self.publish()
                self.assertTrue(f.review()['offline_proxy_review_pending'])

    def test_rebound_forged_marker_or_archive_selection_cannot_clear(self):
        f = self.f
        original_marker = copy.deepcopy(f.outcome['offline_proxy_review_evidence'][0])
        for field, value in [('command', 'curl private'), ('tool_call_id', 'other'),
                             ('bridge_url', 'http://other/v1'), ('tool_result_sha256', 'e' * 64),
                             ('continuation', {'line_index': True, 'response_id': 'chatcmpl-real',
                                 'event_sha256': hashlib.sha256(self.lines[2].encode()).hexdigest()}),
                             ('continuation', {'line_index': 2, 'response_id': 'chatcmpl-forged',
                                 'event_sha256': hashlib.sha256(self.lines[2].encode()).hexdigest()})]:
            with self.subTest(field=field, value=value):
                marker = copy.deepcopy(original_marker)
                marker[field] = value
                f.outcome['offline_proxy_review_evidence'] = [marker]
                binding = digest([marker])
                f.receipt['marker_evidence_sha256'] = f.audit['bindings']['marker_evidence_sha256'] = binding
                self.publish()
                self.assertTrue(f.review()['offline_proxy_review_pending'])

    def test_mixed_source_markers_still_need_public_manager_audit(self):
        f = self.f
        f.outcome['offline_proxy_review_evidence'].append({'reason': 'apt_effective_sources_not_proven',
            'tool_call_id': 'call2', 'command': 'apt-get update', 'tool_result_sha256': 'a' * 64})
        binding = digest(f.outcome['offline_proxy_review_evidence'])
        f.receipt['marker_evidence_sha256'] = f.audit['bindings']['marker_evidence_sha256'] = binding
        self.publish()
        self.assertTrue(f.review()['offline_proxy_review_pending'])
        f.audit['sources'] = [{'manager': 'apt', 'public_effective_sources': True,
            'urls': ['http://archive.ubuntu.com/ubuntu'], 'evidence': 'Pinned source and preceding mutations reviewed'}]
        self.publish()
        self.assertFalse(f.review()['offline_proxy_review_pending'])
        for url in ['http://private.internal/ubuntu', 'http://127.0.0.1/repo', 'http://user:pass@example.com/repo']:
            f.audit['sources'][0]['urls'] = [url]
            self.publish()
            self.assertTrue(f.review()['offline_proxy_review_pending'])


if __name__ == '__main__':
    unittest.main()

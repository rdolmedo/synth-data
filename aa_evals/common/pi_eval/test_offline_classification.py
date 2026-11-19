"""Offline web denial is a model outcome; evaluator/model failures still retry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .worker import classify, q35

PROXY = 'curl: (7) Failed to connect to 10.22.0.107 port 8080 after 0 ms: Could not connect to server'
PUBLIC = "timeout 15 curl -sS -I https://pubchem.ncbi.nlm.nih.gov 2>&1 | head -20; echo '---'; timeout 15 curl -sS -I https://www.google.com 2>&1 | head -20"


class OfflineClassificationTests(unittest.TestCase):
    def host_probe_events(self, *, continued=True, error=None):
        command = "set -x; timeout 5 bash -c 'echo > /dev/tcp/10.22.2.196/45753' 2>&1 || true; timeout 5 curl -sv --noproxy '*' http://10.22.2.196:45753/v1/models 2>&1 | head -50 || true; timeout 5 curl -sv http://10.22.0.107:8080 2>&1 | head -50 || true; timeout 5 curl -sv https://example.com 2>&1 | head -50 || true"
        self.events(command, error=error)
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[1]['message']['content'][0]['text'] = ('Immediate connect fail for 10.22.2.196: Network is unreachable\n'
            'Immediate connect fail for 10.22.0.107: Network is unreachable\n' + PROXY)
        self.session.write_text('\n'.join(json.dumps(e) for e in (events if continued else events[:2])))
        (self.trial / 'config.json').write_text(json.dumps({'agent': {'env': {'OPENAI_BASE_URL': 'http://10.22.2.196:45753/v1'}}}))
        return command

    def test_known_host_probes_are_terminal_unresolved_not_public_or_model_backend(self):
        import hashlib
        self.host_probe_events()
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertTrue(result['offline_proxy_review_required'])
        evidence = result['offline_proxy_review_evidence'][0]
        self.assertEqual(evidence['reason'], 'agent_host_egress_unresolved')
        self.assertEqual(evidence['bridge_config_sha256'], hashlib.sha256((self.trial / 'config.json').read_bytes()).hexdigest())
        self.assertIn('tcp://10.22.2.196:45753', evidence['probe_targets'])
        self.assertEqual(self.outcome(agent_internet=True)['infrastructure_reason'], 'proxy_outage')
        self.host_probe_events(error=PROXY)
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        self.host_probe_events()
        (self.trial / 'verifier/test-stdout.txt').write_text(PROXY)
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        (self.trial / 'verifier/test-stdout.txt').unlink()
        self.result(None, {'exception_type': 'AgentSetupError', 'exception_message': 'setup failed'})
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_unknown_private_probe_commands_missing_config_or_continuation_stay_infra(self):
        for change in ['other-private', 'localhost', 'logs', 'python', 'missing-config', 'missing-continuation', 'wrong-bridge']:
            with self.subTest(change=change):
                self.host_probe_events(continued=change != 'missing-continuation')
                text = self.session.read_text()
                if change == 'other-private':text = text.replace('/dev/tcp/10.22.2.196/45753', '/dev/tcp/10.1.2.3/80')
                elif change == 'localhost':text = text.replace('http://10.22.2.196:45753/v1/models', 'http://localhost:8080')
                elif change in ('logs', 'python'):
                    events = [json.loads(line) for line in text.splitlines()]
                    events[0]['message']['content'][0]['arguments']['command'] += '; ' + ('cat /logs/backend.log' if change == 'logs' else 'python -c "import requests; requests.get(1)"')
                    text = '\n'.join(json.dumps(e) for e in events)
                elif change == 'missing-config':(self.trial / 'config.json').unlink()
                elif change == 'wrong-bridge':(self.trial / 'config.json').write_text('{"agent":{"env":{"OPENAI_BASE_URL":"http://10.1.2.3:90/v1"}}}')
                self.session.write_text(text)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_causal_pre_model_configuration_failure_is_infrastructure_not_a_model_zero(self):
        message = ("Command failed (exit 1): chmod 600 /tmp/harbor-pi-agent/models.json\nstdout: None\nstderr: "
                   "chmod: changing permissions of '/tmp/harbor-pi-agent/models.json': Operation not permitted\n")
        trace = ('File "/venv/harbor/agents/installed/pi.py", line 401, in run\n'
                 '    await self._write_custom_models_json(environment, models_json)\n'
                 'File "/source/tb_agents/harness_agents.py", line 266, in _write_custom_models_json\n'
                 '    await super()._write_custom_models_json(environment, models_json)\n' + message)
        exception = {'exception_type': 'NonZeroAgentExitCodeError', 'exception_message': message, 'exception_traceback': trace}
        self.result(0, exception)
        self.session.write_text('')
        result = self.outcome(agent_internet=False)
        self.assertEqual(result['infrastructure_reason'], 'agent_startup_configuration')
        self.assertEqual(result['reward'], 0)
        self.assertIsNone(result['score'])
        # The same text after a model turn, absent causal frame, or with request
        # evidence is never enough to relabel a model outcome as startup infra.
        self.events('chmod 600 /tmp/harbor-pi-agent/models.json')
        self.assertNotEqual(self.outcome(agent_internet=False).get('infrastructure_reason'), 'agent_startup_configuration')
        self.session.write_text('')
        bad = {**exception, 'exception_traceback': 'model tool output: ' + message}
        self.result(0, bad)
        self.assertNotEqual(self.outcome(agent_internet=False).get('infrastructure_reason'), 'agent_startup_configuration')
        self.result(0, exception)
        wire = self.trial.parent / 'pi-cont16k/wire-requests.jsonl'
        wire.parent.mkdir()
        wire.write_text('{"request_index":1}\n')
        self.assertNotEqual(self.outcome(agent_internet=False).get('infrastructure_reason'), 'agent_startup_configuration')

    def test_static_directory_status_and_proxy_unset_public_requests(self):
        for command in [
                'cd /tmp && timeout 60 curl -v -L -o data.tar.gz https://public.example.org/data 2>&1 | tail -40; echo "exit=${PIPESTATUS[0]}"; ls -l /tmp/data.tar.gz 2>/dev/null',
                'timeout 10 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY curl -sv https://public.example.org 2>&1 | head -50; echo status:$?',
                'cat /etc/hosts; getent hosts deb.debian.org 2>/dev/null; timeout 10 curl -sv http://deb.debian.org/debian/ 2>&1 | head -50; echo "--- no proxy ---"; timeout 10 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY curl -sv http://deb.debian.org/debian/ 2>&1 | head -50']:
            with self.subTest(command=command):
                self.events(command)
                self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])
        for command in ['cd "$HOME" && curl https://public.example.org',
                        'curl https://public.example.org; echo "$(cat /logs/backend.log)"',
                        'env -u OPENAI_BASE_URL curl https://public.example.org',
                        'env HTTP_PROXY=http://proxy.internal:8080 curl https://public.example.org']:
            with self.subTest(command=command):
                self.events(command)
                self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_bounded_pip_flags_still_require_source_review_and_localhost_does_not_clear(self):
        for command in ['pip install numpy scipy --timeout 15 --retries 1 2>&1 | tail -20',
                        'python3 -m pip install --dry-run pexpect pyte 2>&1 | head -50',
                        'cd /tmp && timeout 20 pip install pexpect --timeout=5 2>&1 | tail -30; echo EXIT:${PIPESTATUS[0]}']:
            with self.subTest(command=command):
                self.pip_events(command)
                result = self.outcome(agent_internet=False)
                self.assertFalse(result['infrastructure_failure'])
                self.assertTrue(result['offline_proxy_review_required'])
        for command in ['pip install scipy --timeout nan', 'pip install scipy --retries -1',
                        'pip install scipy --retries 1.5',
                        'pip install scipy; git ls-remote https://github.com/public/repo HEAD',
                        'pip install scipy; apt-get install -y git',
                        'pip install --index-url http://localhost:8080/simple vectorops==0.1.0']:
            with self.subTest(command=command):
                self.pip_events(command)
                self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_pip_static_version_postlude_requires_review_and_rejects_extra_execution(self):
        prefix = 'cd /tmp && timeout 20 pip install pexpect --disable-pip-version-check --timeout 5 2>&1 | tail -30; '
        self.pip_events(prefix + "python3 -c 'import pexpect; print(pexpect.__version__)' 2>&1")
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertTrue(result['offline_proxy_review_required'])
        self.events("curl https://public.example.org; python3 -c 'import pexpect; print(pexpect.__version__)'")
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        for suffix in ["python3 -c 'import requests; requests.get(\"http://127.0.0.1\")'",
                       "python3 -c 'import pexpect; print(pexpect.__version__); pexpect.run(\"cat /logs/backend.log\")'",
                       "python3 -c 'import pexpect; print(open(\"/logs/backend.log\").read())'",
                       "python3 -c 'import pexpect; print(pexpect.__version__())'",
                       "python3 -c 'import pexpect as alias; print(alias.__version__)'",
                       "python3 -c 'import pexpect; print(other.__version__)'"]:
            with self.subTest(suffix=suffix):
                self.pip_events(prefix + suffix)
                self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_public_git_probe_requires_effective_url_review_and_keeps_benchmark_url_evidence(self):
        command = 'timeout 25 git ls-remote https://github.com/laude-institute/terminal-bench.git HEAD 2>&1 | head -20; echo EXIT:${PIPESTATUS[0]}'
        self.events(command)
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[1]['message']['content'][0]['text'] = "fatal: unable to access 'https://github.com/laude-institute/terminal-bench.git/': " + PROXY
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertTrue(result['offline_proxy_review_required'])
        self.assertEqual(result['offline_proxy_review_evidence'][0]['reason'], 'git_effective_urls_not_proven')
        self.assertEqual(result['offline_proxy_review_evidence'][0]['command'], command)
        self.assertIn('terminal-bench.git', self.session.read_text())
        self.events('git ls-remote http://service.internal/repo HEAD')
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_apt_denial_requires_separate_sources_review_and_never_hides_backend_error(self):
        command = 'DEBIAN_FRONTEND=noninteractive apt-get install -y git 2>&1 | tail -60'
        self.events(command)
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[1]['message']['content'][0]['text'] = ('Err:1 http://archive.ubuntu.com/ubuntu noble/main amd64 git\n'
            'Cannot initiate the connection to 10.22.0.107:8080 (10.22.0.107). - connect (101: Network is unreachable)')
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertTrue(result['offline_proxy_review_required'])
        self.assertEqual(result['offline_proxy_review_evidence'][0]['reason'], 'apt_effective_sources_not_proven')
        events[-1]['message'].update(stopReason='error', errorMessage=PROXY)
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        events[-1]['message'].update(stopReason='stop', errorMessage='')
        events[1]['message']['content'][0]['text'] = events[1]['message']['content'][0]['text'].replace('http://archive.ubuntu.com/ubuntu', 'http://mirror.internal/ubuntu')
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        self.assertIsNotNone(q35._offline_agent_proxy_failure(self.session, review=[]))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        self.assertFalse(self.outcome(agent_internet=False).get('offline_proxy_review_required', False))
        for command in ['apt-get -o Acquire::http::Proxy=http://service.internal install -y git',
                        'apt-get install -y git; cat /logs/backend.log', 'bash /tmp/unknown-installer.sh']:
            with self.subTest(command=command):
                events[0]['message']['content'][0]['arguments']['command'] = command
                self.session.write_text('\n'.join(json.dumps(e) for e in events))
                result = self.outcome(agent_internet=False)
                self.assertTrue(result['infrastructure_failure'])
                self.assertFalse(result.get('offline_proxy_review_required', False))

    def test_literal_public_python_request_proves_policy_reflection_without_hiding_errors(self):
        command = "python - <<'PY'\nimport requests\nfor url in ['https://huggingface.co','https://raw.githubusercontent.com']:\n    try:\n        r=requests.get(url,timeout=10)\n        print(url, r.status_code, len(r.content), r.url)\n    except Exception as e:\n        print(url, type(e).__name__, str(e)[:200])\nPY"
        self.assertTrue(q35._explicit_public_python_request({'name': 'bash', 'arguments': {'command': command}}))
        self.events(command)
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[1]['message']['content'][0]['text'] = "https://huggingface.co ProxyError HTTPSConnectionPool: Unable to connect to proxy 10.22.0.107:8080"
        events[-1]['message']['content'][0]['text'] = 'The proxy 10.22.0.107:8080 which refused means this agent is offline.'
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])
        events[-1]['message'].update(stopReason='error', errorMessage=PROXY)
        self.session.write_text('\n'.join(json.dumps(e) for e in events))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        for bad in [command.replace('https://huggingface.co', 'http://service.internal'),
                    command.replace('import requests', 'import requests\nimport os'),
                    command.replace('requests.get(url,timeout=10)', 'requests.get(url,timeout=10,proxies={})'),
                    command.replace('requests.get(url,timeout=10)', 'requests.post(url,timeout=10)')]:
            with self.subTest(command=bad):
                self.assertFalse(q35._explicit_public_python_request({'name': 'bash', 'arguments': {'command': bad}}))

    def paid_continuation(self):
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[-1]['message'].update(api='openai-completions', provider='harbor-endpoint',
                                    responseId='chatcmpl-actual-response', usage={'output': 123})
        self.session.write_text('\n'.join(json.dumps(event) for event in events) + '\n')
        (self.trial / 'config.json').write_text(json.dumps({
            'agent': {'env': {'OPENAI_BASE_URL': 'http://10.22.2.196:45753/v1'}}}))
        return events

    def test_unrecognized_linked_tool_denials_with_paid_continuation_require_review(self):
        commands = [
            'cat /etc/apt/sources.list; find /etc/apt/sources.list.d -type f -exec cat {} \\;; timeout 5 apt-get update',
            'curl https://huggingface.co/public/config.json; getent hosts huggingface.co; pip download --no-deps flask',
            'getent hosts example.org; curl https://example.org; sed -n 1,20p /logs/agent/pi.txt',
            'curl http://private.internal/unknown-service']
        for command in commands:
            with self.subTest(command=command):
                self.events(command)
                self.paid_continuation()
                result = self.outcome(agent_internet=False)
                self.assertFalse(result['infrastructure_failure'])
                self.assertTrue(result['offline_proxy_review_required'])
                marker = result['offline_proxy_review_evidence'][0]
                self.assertEqual(marker['reason'], 'agent_tool_network_origin_not_proven')
                self.assertEqual(marker['command'], command)
                self.assertEqual(marker['continuation']['response_id'], 'chatcmpl-actual-response')
        self.events(commands[0])
        events = self.paid_continuation()
        events[1]['message']['content'][0]['text'] = ('Cannot initiate the connection to 10.22.0.107:8080 '
            '(10.22.0.107). - connect (101: Network is unreachable)')
        self.session.write_text('\n'.join(json.dumps(event) for event in events))
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_generic_review_needs_bound_actual_continuation_and_keeps_primary_errors(self):
        for mutation in ['no_config', 'output_zero', 'output_bool', 'synthetic_id', 'other_provider', 'unlinked', 'primary_error']:
            with self.subTest(mutation=mutation):
                self.events('bash /tmp/unknown-installer.sh')
                events = self.paid_continuation()
                if mutation == 'no_config':
                    (self.trial / 'config.json').unlink()
                elif mutation.startswith('output_'):
                    events[-1]['message']['usage']['output'] = 0 if mutation == 'output_zero' else True
                elif mutation == 'synthetic_id':
                    events[-1]['message']['responseId'] = 'context-stop'
                elif mutation == 'other_provider':
                    events[-1]['message']['provider'] = 'other'
                elif mutation == 'unlinked':
                    events[1]['message']['toolCallId'] = 'unknown'
                else:
                    events[-1]['message'].update(stopReason='error', errorMessage=PROXY)
                self.session.write_text('\n'.join(json.dumps(event) for event in events))
                self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_assistant_source_or_narrative_is_not_a_transport_error(self):
        self.events('echo local')
        events = self.paid_continuation()
        events[1]['message']['content'][0]['text'] = 'local'
        events[0]['message']['content'][0]['arguments']['command'] = 'echo ' + PROXY
        events[-1]['message']['content'][0]['text'] = 'The tool reported ' + PROXY
        self.session.write_text('\n'.join(json.dumps(event) for event in events))
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertNotIn('offline_proxy_review_required', result)

    def test_typed_grader_pending_resolves_generation_without_zero_or_retry(self):
        self.events('echo local')
        events = self.paid_continuation()
        events[1]['message']['content'][0]['text'] = 'local'
        self.session.write_text('\n'.join(json.dumps(event) for event in events))
        self.result(reward=None, exception={'exception_type': 'RewardFileNotFoundError'})
        marker = {'schema_version': 1, 'kind': 'caffe_data_adapter_unsupported', 'task': 'caffe-cifar-10',
            'grading_pending': True, 'regenerate': False, 'reason': 'Unsupported active MemoryData input',
            'evidence': {name: 'a' * 64 for name in ('submitted_network_sha256', 'official_test_sha256',
                'descriptor_sha256', 'grader_sha256')}}
        marker['evidence']['input_layers'] = [{'name': 'input', 'type': 'MemoryData', 'configuration_sha256': 'b' * 64}]
        path = self.trial / 'verifier/grading-pending.json'
        path.write_text(json.dumps(marker))
        result = self.outcome(agent_internet=False)
        self.assertFalse(result['infrastructure_failure'])
        self.assertIsNone(result['score'])
        self.assertTrue(result['grading_review_required'])
        self.assertNotIn('unscored', result)
        marker['evidence']['input_layers'] = []  # valid legacy net.layers/net.input needs another adapter
        path.write_text(json.dumps(marker))
        self.assertTrue(self.outcome(agent_internet=False)['grading_pending'])
        path.write_text(json.dumps(marker).replace('"schema_version": 1', '"schema_version": 1,"schema_version": 1'))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        marker['schema_version'] = True
        path.write_text(json.dumps(marker))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        marker['schema_version'] = 1
        path.write_text(json.dumps(marker))
        events[-1]['message'].update(stopReason='error', errorMessage='connection error: backend unreachable')
        self.session.write_text('\n'.join(json.dumps(event) for event in events))
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'raw'
        self.trial = self.raw / 'fake' / 'trial'
        self.session = self.trial / 'agent/pi/sessions/fake.jsonl'
        self.session.parent.mkdir(parents=True)
        (self.trial / 'verifier').mkdir()
        self.result()
        self.events()

    def tearDown(self):
        self.tmp.cleanup()

    def result(self, reward=0, exception=None):
        (self.trial / 'result.json').write_text(json.dumps({
            'verifier_result': {'rewards': {'reward': reward}},
            'exception_info': exception}))

    def events(self, command=PUBLIC, *, linked=True, error=None):
        call = {'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'toolUse', 'content': [
            {'type': 'toolCall', 'id': 'call1', 'name': 'bash', 'arguments': {'command': command}}]}}
        result = {'type': 'message', 'message': {'role': 'toolResult', 'toolName': 'bash',
                  'toolCallId': 'call1' if linked else 'unknown', 'content': [{'type': 'text', 'text': PROXY}]}}
        final = {'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'error' if error else 'stop',
                 'errorMessage': error or '', 'content': [{'type': 'text', 'text': 'Finished.'}]}}
        self.session.write_text('\n'.join(json.dumps(e) for e in [call, result, final]) + '\n')

    def outcome(self, **kwargs):
        return classify(self.raw, 1, False, **kwargs)

    def test_offline_linked_public_tool_denial_keeps_reward_and_queues_no_retry(self):
        offline = self.outcome(agent_internet=False)
        self.assertFalse(offline['infrastructure_failure'])
        self.assertEqual(offline['score'], 0)
        run = self.root / 'fake-run'
        for sub in ('queue/done', 'queue/retry', 'queue/claims', 'serve'):
            (run / sub).mkdir(parents=True)
        (run / 'queue/items.txt').write_text('fake 0\n')
        state = q35.State(run)
        with mock.patch.object(type(state), 'in_outage', return_value=False):
            state.finish('fake__a0__r0', offline)
            self.assertEqual(state.logical()['fake', 0]['key'], 'fake__a0__r0')
            self.assertEqual(state.requeue_missing(), [])
            self.assertEqual(list((run / 'queue/retry').iterdir()), [])
            online = self.outcome(agent_internet=True)
            self.assertEqual(online['infrastructure_reason'], 'proxy_outage')
            state.finish('other__a0__r0', online)
            self.assertTrue((run / 'queue/retry/other__a0__r1.json').exists())

    def pip_events(self, command, **kwargs):
        self.events(command, **kwargs)
        events = [json.loads(line) for line in self.session.read_text().splitlines()]
        events[1]['message']['content'] = [{'type': 'text', 'text':
            "WARNING: Retrying after ProxyError('Cannot connect to proxy.', "
            "NewConnectionError('Network is unreachable')): /simple/scipy/"}]
        self.session.write_text('\n'.join(json.dumps(x) for x in events) + '\n')

    def test_default_pip_denial_stops_retry_but_requires_source_review(self):
        for command in ['pip install scipy scikit-image 2>&1 | tail -20',
                        'python3 -m pip install "scipy>=1.2,<2"',
                        'pip install --no-deps --break-system-packages --force-reinstall scipy',
                        '/usr/local/bin/pip3 --no-cache-dir install -q scipy==1.2',
                        'timeout 120 python -m pip install --upgrade scipy[extra] || true']:
            with self.subTest(command=command):
                self.pip_events(command)
                outcome = self.outcome(agent_internet=False)
                self.assertFalse(outcome['infrastructure_failure'])
                self.assertEqual(outcome['score'], 0)
                self.assertTrue(outcome['offline_proxy_review_required'])
                evidence = outcome['offline_proxy_review_evidence']
                self.assertEqual(evidence[0]['tool_call_id'], 'call1')
                self.assertEqual(evidence[0]['command'], command)
                self.assertEqual(len(evidence[0]['tool_result_sha256']), 64)
        run = self.root / 'review-queue'
        for sub in ('queue/done', 'queue/retry', 'queue/claims', 'serve'):
            (run / sub).mkdir(parents=True)
        (run / 'queue/items.txt').write_text('fake 0\n')
        state = q35.State(run)
        with mock.patch.object(type(state), 'in_outage', return_value=False):
            state.finish('fake__a0__r0', outcome)
            self.assertEqual(state.requeue_missing(), [])
            self.assertTrue(state.logical()['fake', 0]['offline_proxy_review_required'])
        self.assertEqual(self.outcome(agent_internet=True)['infrastructure_reason'], 'proxy_outage')

    def test_public_pip_index_still_requires_effective_source_review(self):
        for command in ['pip install --index-url https://pypi.org/simple scipy',
                        'python3 -m pip install scipy -i https://pypi.org/simple 2>&1 | tail -20',
                        'pip -q install --index-url https://pypi.org/simple scipy']:
            with self.subTest(command=command):
                self.pip_events(command)
                outcome = self.outcome(agent_internet=False)
                self.assertFalse(outcome['infrastructure_failure'])
                self.assertTrue(outcome['offline_proxy_review_required'])

    def test_private_mixed_configured_or_unknown_pip_commands_fail_closed(self):
        for command in ['pip install -i http://127.0.0.1:8080/simple scipy',
                        'pip install -i http://127.1/simple scipy',
                        'pip install scipy --index-url http://packages.internal/simple',
                        'pip install scipy --extra-index-url https://pypi.org/simple',
                        'PIP_INDEX_URL=https://pypi.org/simple pip install scipy',
                        'pip install scipy --proxy http://proxy.internal:8080',
                        'pip install -r requirements.txt', 'pip install -e /app/project',
                        'pip install /app/project', 'pip install https://pypi.org/scipy.whl',
                        'pip install scipy; curl http://127.0.0.1:8000',
                        'pip install scipy; cat /logs/backend.log',
                        'pip install scipy; python -c "import requests; requests.get(1)"']:
            with self.subTest(command=command):
                self.pip_events(command)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')
        self.pip_events('pip install scipy')
        events = self.session.read_text().replace('WARNING: Retrying', 'Looking in indexes: http://packages.internal/simple\\nWARNING: Retrying')
        self.session.write_text(events)
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')
        self.pip_events('pip install -i https://pypi.org/simple scipy')
        self.session.write_text(self.session.read_text().replace('WARNING: Retrying', 'Looking in indexes: http://127.0.0.1/simple\\nWARNING: Retrying'))
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')
        self.events('pip install scipy')  # curl-style text cannot establish an ambiguous pip denial
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_default_pip_review_never_hides_model_setup_or_verifier_faults(self):
        for error in [PROXY, 'Connection error ECONNREFUSED 127.0.0.1', 'Request timed out ETIMEDOUT']:
            self.pip_events('pip install scipy', error=error)
            self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        self.pip_events('pip install scipy')
        (self.trial / 'verifier/test-stdout.txt').write_text(PROXY)
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')
        (self.trial / 'verifier/test-stdout.txt').unlink()
        self.result(reward=None, exception={'exception_type': 'EnvironmentSetupError'})
        self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])
        self.result(reward=1)
        self.assertEqual(self.outcome(agent_internet=False)['score'], 1)
        self.assertTrue(self.outcome(agent_internet=False)['offline_proxy_review_required'])

    def test_routine_bounded_curl_flags_have_correct_option_arity(self):
        for command in [
            'curl -sS --max-time 10 --connect-timeout 2 -o /tmp/public.html -w "%{http_code}" https://www.google.com',
            'curl -m10 --connect-timeout=2 --output=/tmp/public.html --write-out="%{http_code}" --url https://www.google.com',
            'curl --url=https://www.google.com -m 10 | head -n 20',
            'curl -sSIL -m 10 https://www.google.com; echo finished',
            'wget --timeout 10 --output-document /tmp/public.html https://www.google.com']:
            with self.subTest(command=command):
                self.events(command)
                self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])
        for command in ['curl --max-time https://www.google.com', 'curl -m 0 https://www.google.com',
                        'curl --output', 'curl --url',
                        'curl --max-time 10 --url http://service:8080',
                        'curl --url https://www.google.com --url http://127.0.0.1:8080']:
            with self.subTest(command=command):
                self.events(command)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_public_requests_with_passive_wrappers_and_tail_remain_model_outcomes(self):
        url='https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVDOC.ZIP'
        for command in [
            'command -v curl wget python3; timeout 30 curl -v -I --max-time 20 '+url+' 2>&1 | tail -40',
            'which curl wget python3 git || true; timeout 20 curl -I '+url+' 2>&1 | head -40 || true',
            'find /opt /root /usr/local /var/cache /tmp -type f 2>/dev/null | head -200; echo "--- network test ---"; timeout 10 curl -I '+url+' 2>&1 | head -20 || true',
            'getent hosts www.povray.org; echo "---"; curl -I --max-time 10 http://archive.ubuntu.com/ubuntu/ 2>&1 | head -20',
            'timeout 15 curl -I -L '+url+' 2>&1 | head -30 || true; timeout 15 getent hosts www.povray.org || true',
            'getent hosts archive.ubuntu.com security.ubuntu.com deb.debian.org povray.org www.povray.org || true; timeout 10 curl -I http://archive.ubuntu.com/ubuntu/ 2>&1 | head -20 || true; cat /opt/aa-runtime-cache/apt/apt.conf /opt/aa-runtime-cache/apt/sources.list']:
            with self.subTest(command=command):
                self.events(command)
                self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_wrappers_do_not_hide_private_targets_dynamic_or_executing_commands(self):
        for command in [
            'command -v curl; curl http://127.0.0.1:8080 | tail -40',
            'which curl; curl https://www.google.com; curl http://service:8080 | tail -40',
            'getent hosts localhost; curl https://www.google.com | tail -40',
            'getent hosts model.internal.cluster.internal; curl https://www.google.com',
            r'find /tmp -exec curl https://www.google.com {} \; ; curl https://www.google.com',
            'cat <(curl https://www.google.com); curl https://www.google.com',
            'command -v curl; curl "$TARGET"; curl https://www.google.com',
            'command -v curl; curl https://www.google.com >/dev/tcp/127.0.0.1/8080']:
            with self.subTest(command=command):
                self.events(command)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'],'proxy_outage')

    def test_public_curl_cannot_suppress_backend_service_log_reads_or_find_actions(self):
        for command in [
            'curl https://www.google.com; cat /logs/backend.log',
            'cat /tmp/service.log; timeout 10 curl https://www.google.com | tail -40',
            'curl https://www.google.com; cat /etc/hosts /logs/model-proxy.log',
            'find /tmp -fprint /logs/backend.log; curl https://www.google.com',
            'find -files0-from /logs/backend.log; curl https://www.google.com',
            'find /tmp -unknown-predicate; curl https://www.google.com']:
            with self.subTest(command=command):
                self.events(command)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'],'proxy_outage')
        self.events('cat /etc/hosts /etc/resolv.conf /opt/aa-runtime-cache/public-packages.json; curl https://www.google.com')
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_narrative_reflection_is_not_observed_transport_failure(self):
        self.events('command -v curl; timeout 30 curl -I https://www.google.com 2>&1 | tail -40')
        events=[json.loads(line)for line in self.session.read_text().splitlines()]
        events[-1]['message']['content']=[{'type':'thinking','thinking':'The proxy at 10.22.0.107 refused connections, so internet is disabled.'}]
        self.session.write_text('\n'.join(json.dumps(x)for x in events)+'\n')
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])
        # Narrative alone does not establish an observed transport failure.
        events[1]['message']['content']=[{'type':'text','text':'no network operation completed'}]
        self.session.write_text('\n'.join(json.dumps(x)for x in events)+'\n')
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])
        # An explicit model error field remains infrastructure even after a known denial.
        events[1]['message']['content']=[{'type':'text','text':PROXY}]
        events[-1]['message']['errorMessage']=PROXY
        self.session.write_text('\n'.join(json.dumps(x)for x in events)+'\n')
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'],'proxy_outage')

    def test_malformed_model_log_shapes_fail_closed_without_crashing(self):
        self.assertFalse(q35._explicit_external_request({'name':'bash','arguments':[]}))
        for event in [[], 'not an event', {'message':[]},
                      {'type':'message','message':{'role':'assistant','content':PROXY}},
                      {'type':'message','message':{'role':'assistant','content':{'text':PROXY}}},
                      {'type':'message','message':{'role':'toolResult','toolName':'bash','toolCallId':[],'content':PROXY}}]:
            with self.subTest(event=event):
                self.session.write_text(json.dumps(event)+'\n')
                outcome=self.outcome(agent_internet=False)
                self.assertEqual(outcome['infrastructure_failure'],PROXY in json.dumps(event))

    def test_non_dict_arguments_in_actual_logged_call_never_crash_proxy_scan(self):
        for arguments in [[], ['not a dict'], 42, 'bad arguments']:
            with self.subTest(arguments=arguments):
                self.events()
                events = [json.loads(line) for line in self.session.read_text().splitlines()]
                events[0]['message']['content'][0]['arguments'] = arguments
                self.session.write_text('\n'.join(json.dumps(e) for e in events) + '\n')
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_malformed_pip_tool_result_text_preserves_proxy_evidence_without_crashing(self):
        for malformed in [{'type': 'text', 'text': None}, {'type': 'text', 'text': []},
                          {'type': 'text', 'text': 42}, 'not a block']:
            with self.subTest(malformed=malformed):
                self.pip_events('pip install scipy')
                events = [json.loads(line) for line in self.session.read_text().splitlines()]
                events[1]['message']['content'].insert(0, malformed)
                self.session.write_text('\n'.join(json.dumps(e) for e in events) + '\n')
                outcome = self.outcome(agent_internet=False)
                self.assertEqual(outcome['infrastructure_reason'], 'proxy_outage')
                self.assertNotIn('offline_proxy_review_required', outcome)

    def test_successful_reward_is_preserved(self):
        self.result(reward=1)
        self.assertEqual(self.outcome(agent_internet=False)['score'], 1)
        self.assertFalse(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_legacy_online_and_unspecified_policy_remain_infrastructure(self):
        for policy in (True, None, 'false'):
            with self.subTest(policy=policy):
                self.assertEqual(self.outcome(agent_internet=policy)['infrastructure_reason'], 'proxy_outage')
        self.assertEqual(self.outcome()['infrastructure_reason'], 'proxy_outage')

    def test_unlinked_result_or_nonrequest_command_fails_closed(self):
        for command, linked in [(PUBLIC, False), ('echo https://www.google.com', True),
                                ('python -c "print(123)"', True), ('curl "$TARGET"', True)]:
            with self.subTest(command=command, linked=linked):
                self.events(command, linked=linked)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_mixed_private_local_model_and_transport_override_targets_remain_infra(self):
        for command in [
            'curl http://127.0.0.1:8080', 'curl http://localhost:8080', 'curl http://service:80',
            'curl http://10.22.2.119:80', 'curl http://[::1]:8080',
            'curl https://www.google.com; curl http://service:80',
            'curl https://www.google.com; curl localhost:8080',
            'curl https://www.google.com; curl 127.0.0.1:8080',
            'curl --resolve www.google.com:443:127.0.0.1 https://www.google.com',
            'curl -x proxy.internal:8080 https://www.google.com',
            'curl http://model.internal.cluster.internal:1234/v1/models']:
            with self.subTest(command=command):
                self.events(command)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')

    def test_assistant_model_proxy_and_backend_errors_are_never_suppressed(self):
        for error, expected in [(PROXY, 'proxy_outage'), ('Connection error ECONNREFUSED 127.0.0.1', 'pi_backend_unreachable'),
                                ('Request timed out ETIMEDOUT', 'pi_backend_timeout'),
                                # #531: the response-budget guard's own 502 (no longer held by the sticky proxy)
                                # (Pi 1.2.0's recorded errorMessage, verified with a real Pi session)
                                ('502: {"message":"response budget accounting error: Backend detokenization failed with '
                                 'status 500","type":"response_budget_accounting_error"}', 'response_budget_accounting'),
                                ('502: {"message":"x","type":"response_budget_accounting_error"}',   # older guards
                                 'response_budget_accounting')]:
            with self.subTest(error=error):
                self.events(error=error)
                self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], expected)

    def test_evaluator_verifier_proxy_failure_still_retries(self):
        (self.trial / 'verifier/test-stdout.txt').write_text(PROXY)
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'proxy_outage')
        self.assertTrue(q35.proxy_failure(self.trial, agent_internet=False).startswith('verifier:'))

    def test_setup_and_local_service_start_failures_remain_infra(self):
        for kind in ('AgentSetupError', 'EnvironmentSetupError', 'EnvironmentStartError', 'ConnectionRefusedError'):
            with self.subTest(kind=kind):
                self.result(reward=None, exception={'exception_type': kind, 'exception_message': 'local service 127.0.0.1 refused'})
                self.assertTrue(self.outcome(agent_internet=False)['infrastructure_failure'])

    def test_verifier_no_reward_and_endpoint_handover_rules_are_unchanged(self):
        self.result(reward=None)
        self.assertEqual(self.outcome(agent_internet=False)['infrastructure_reason'], 'verifier_no_reward')
        self.assertEqual(classify(self.raw, 1, 'endpoint_moved', agent_internet=False)['infrastructure_reason'], 'endpoint_moved')


if __name__ == '__main__':
    unittest.main()

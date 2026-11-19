"""CPU tests for the shared synthesis core (run: python -m pytest synth/tests)."""
import collections
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest

from synth import config as config_module
from synth.config import Campaign, build, load_teacher
from synth.controller import baseline_counts, pick_per_task, prefer_eval
from synth.prepare import check_smokes, rewrite_agent_timeouts
from synth.vllm_launch import replica_command
from synth.wire import WireLog, load, sse_events

# The recorded launchers below served campaigns frozen at a 132,000 context; new campaigns default to 131,072
# (synth.config PROTOCOL_DEFAULTS, 2026-11-14).
RECORDED_PROTOCOL = {**config_module.PROTOCOL_DEFAULTS, 'context_tokens': 132000}

REPO = Path(__file__).resolve().parents[2]
MULTISTEP = b'''schema_version = "2.0"
[verifier]
timeout_sec = 2400.0
[agent]
timeout_sec = 3600.0
[environment]
memory_mb = 2048
[[steps]]
name = "milestone_1"
[steps.agent]
timeout_sec = 1800.0
[steps.verifier]
timeout_sec = 1200.0
[[steps]]
name = "milestone_2"
[steps.agent]
timeout_sec = 1800.0
'''


class Bundles(unittest.TestCase):
    def test_every_agent_limit_and_nothing_else(self):
        after, count, _ = rewrite_agent_timeouts(MULTISTEP, 21600)
        parsed = tomllib.loads(after.decode())
        self.assertEqual(count, 3)
        self.assertEqual(parsed['agent']['timeout_sec'], 21600)
        self.assertEqual([s['agent']['timeout_sec'] for s in parsed['steps']], [21600, 21600])
        self.assertEqual(parsed['verifier']['timeout_sec'], 2400.0)
        self.assertEqual(parsed['steps'][0]['verifier']['timeout_sec'], 1200.0)

    def test_smoke_tasks_must_be_in_their_bundle(self):
        campaign = Campaign({'id': 'x', 'root': '/tmp/x', 'teacher': load_teacher('nemotron35-super'),
                             'protocol': dict(config_module.PROTOCOL_DEFAULTS), 'train': {'task_list': ['a', 'b']},
                             'smoke': {'attempts': [{'task': 'a', 'kind': 'train'}]}})
        check_smokes(campaign)
        campaign.config['smoke']['attempts'].append({'task': 'hero-only', 'kind': 'train'})
        with self.assertRaisesRegex(SystemExit, 'hero-only'):
            check_smokes(campaign)


class Scheduling(unittest.TestCase):
    def test_fewest_first_counts_live_attempts_and_caps(self):
        tasks = ['a', 'b', 'c']
        have = collections.Counter({'a': 4, 'b': 3, 'c': 7})
        active = collections.Counter({'b': 2})
        tried = collections.Counter({'a': 1, 'b': 2})
        caps = {'a': 8, 'b': 8, 'c': 5}
        self.assertEqual(pick_per_task(tasks, 8, have, active, tried, caps), 'a')
        have['a'] = 7
        self.assertEqual(pick_per_task(tasks, 8, have, active, tried, caps), 'b')
        self.assertIsNone(pick_per_task(tasks, 8, collections.Counter({'a': 8, 'b': 6, 'c': 8}),
                                        collections.Counter({'b': 2}), tried, caps))
        self.assertIsNone(pick_per_task(['a'], 8, collections.Counter(), collections.Counter(),
                                        collections.Counter({'a': 9}), {'a': 9}))

    def test_eval_share_from_first_burst(self):
        self.assertTrue(prefer_eval(0, 0, 100, 25))
        self.assertFalse(prefer_eval(10, 25, 100, 25))
        self.assertTrue(prefer_eval(10, 30, 100, 25, train_stopped=True))

    def test_baseline_counts_only_usable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'manifest.json'
            path.write_text(json.dumps({'records': [{'task': 'a', 'usable': True}, {'task': 'a', 'usable': False},
                                                     {'task': 'b', 'usable': True}]}))
            self.assertEqual(baseline_counts([path]), {'a': 1, 'b': 1})


class Wire(unittest.TestCase):
    def test_round_trip_and_torn_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'logs', 'wire.jsonl')
            log = WireLog(path)
            first = log.request(json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'tools': []}).encode(),
                                'http://b', '/v1/chat/completions')
            log.data(first, b'data: {"id":"r1","choices":[{"delta":{"content":"he"}}]}\n\n')
            log.data(first, b'data: {"id":"r1","choices":[{"delta":{"content":"llo"}}]}\n\n')
            log.end(first, 200)
            second = log.request(b'not json', 'http://b', '/v1/chat/completions')
            log.data(second, 'é'.encode()[:1])  # split multi-byte character survives
            log.close()
            with open(path, 'a') as f:
                f.write('{"t":"data","id":2,')
            calls = load(path)
            self.assertEqual([c['id'] for c in calls], [1, 2])
            self.assertTrue(calls[0]['complete'])
            self.assertEqual(''.join(e['choices'][0]['delta']['content'] for e in sse_events(calls[0]['response'])), 'hello')
            self.assertEqual(calls[1]['raw'], b'not json')
            self.assertFalse(calls[1]['complete'])
            self.assertEqual(bytes(calls[1]['response']), 'é'.encode()[:1])

    def test_short_writes_continue_and_a_cut_record_is_ended(self):
        # #451: a short write is continued; a write failing after part of a record went out ends it with a newline,
        # so later records are not glued onto the fragment (load() still reports the malformed line)
        from unittest import mock
        import synth.wire as W
        real = os.write
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'wire.jsonl')
            log = WireLog(path)
            with mock.patch.object(W.os, 'write', lambda fd, b: real(fd, bytes(b[:7]))):   # 7 bytes per call
                first = log.request(json.dumps({'messages': []}).encode(), 'http://b', '/v1/chat/completions')
                log.end(first, 200)
            self.assertEqual([(c['id'], c['complete']) for c in load(path)], [(1, True)])
            calls = []

            def quota(fd, b):
                calls.append(len(b))
                if len(b) == 1:
                    return real(fd, b)
                if len(calls) == 1:
                    return real(fd, bytes(b[:20]))
                raise OSError(122, 'Disk quota exceeded')
            with mock.patch.object(W.os, 'write', quota), self.assertRaises(OSError):
                log.request(json.dumps({'messages': ['x' * 100]}).encode(), 'http://b', '/v1/chat/completions')
            log.request(b'{}', 'http://b', '/v1/chat/completions')
            with open(path, 'rb') as f:
                lines = f.read().split(b'\n')
            self.assertEqual(len(lines[2]), 20)                       # the fragment, on its own line
            self.assertEqual(json.loads(lines[3])['t'], 'req')        # the next record intact
            with self.assertRaises(ValueError):
                load(path)


SUPER_CHECKPOINT = Path('/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4')


class Teachers(unittest.TestCase):
    def test_every_teacher_spec_loads(self):
        for path in sorted(config_module.TEACHERS.glob('*.toml')):
            teacher = load_teacher(path.stem)
            for key in ('model', 'revision', 'checkpoint', 'served_name', 'runtime', 'serving', 'request', 'counting'):
                self.assertIn(key, teacher, f'{path.name} lacks {key}')
            argv = replica_command(teacher, config_module.PROTOCOL_DEFAULTS, port=12345, model_dir='/m',
                                   tunables=teacher['serving'].get('tunables', {}),
                                   rank_ports={'dp_size': 4, 'rpc_port': 1, 'supervisor_port': 2, 'stage_draft': '/d'})
            self.assertIn(str(config_module.PROTOCOL_DEFAULTS['context_tokens']), argv, path.name)
            self.assertTrue(all('{' not in a or a.startswith('{"') or a.startswith('[') for a in argv), argv)

    def test_super_matches_recorded_launcher(self):
        """The spec must reproduce inference/serve_nemotron.sh (HARDWARE=b200) exactly, apart from flags added since the
        recording that cannot change outputs (listed below)."""
        teacher = load_teacher('nemotron35-super')
        tunables = {'mtp': 1, 'max_seqs': 512, 'max_batched': 16384, 'omp_threads': 16}
        with tempfile.TemporaryDirectory() as d:
            venv = Path(d) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'lib/python3.12/site-packages/nvidia/cu13/lib').mkdir(parents=True)
            fake = venv / 'bin' / 'vllm'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = {**os.environ, 'VENV': str(venv), 'MODEL': '/m', 'HARDWARE': 'b200', 'SCRATCH': str(Path(d) / 's'),
                   'MAX_LEN': '132000', 'MAX_SEQS': '512', 'MAX_BATCHED': '16384', 'MTP_TOKENS': '1',
                   'SERVING_SEED': '20261016'}
            out = subprocess.run(['bash', str(REPO / 'inference/serve_nemotron.sh'), '12345'], env=env,
                                 capture_output=True, text=True, check=True).stdout
        recorded = json.loads(out.strip().splitlines()[-1])
        ours = replica_command(teacher, RECORDED_PROTOCOL, port=12345, model_dir='/m', tunables=tunables)
        self.assertEqual(ours[0], f'{teacher["runtime"]["venv"]}/bin/vllm')
        # flags added since the recording that cannot change outputs (throughput only, #451 ws4): the renderer pool
        i = ours.index('--renderer-num-workers')
        self.assertEqual(ours[i + 1], '4')
        self.assertEqual(ours[1:i] + ours[i + 2:], recorded)


@unittest.skipUnless(SUPER_CHECKPOINT.exists(), 'needs the Nemotron Super tokenizer')
class Export(unittest.TestCase):
    def campaign(self, root):
        teacher = load_teacher('nemotron35-super')
        return Campaign({'id': 'unit', 'root': str(root), 'teacher': teacher,
                         'protocol': {**config_module.PROTOCOL_DEFAULTS}, 'train': {'task_list': ['t']}})

    def test_unique_graph_tokens_and_loss(self):
        from synth.export import export_one
        from synth.serializers import serializer_for
        from tb_agents.pi_context_limit import CAP_MARK
        with tempfile.TemporaryDirectory() as d:
            campaign = self.campaign(d)
            folder = Path(d) / 'unit' / 'train' / 'raw' / 'x'
            sessions = folder / 'harbor' / 't__abc' / 'agent' / 'pi' / 'sessions'
            sessions.mkdir(parents=True)
            (folder / 'attempt.json').write_text(json.dumps({'id': 'x', 'kind': 'train', 'task': 't', 'attempt': 0}))
            (folder / 'worker-result.json').write_text(json.dumps({'interrupted': False}))
            events = [{'type': 'session', 'id': 's1'},
                {'type': 'message', 'id': 'u', 'parentId': None, 'message': {'role': 'user', 'content': 'Do it'}},
                {'type': 'message', 'id': 'a1', 'parentId': 'u', 'message': {'role': 'assistant', 'stopReason': 'toolUse',
                    'content': [{'type': 'thinking', 'thinking': 'plan'},
                                {'type': 'toolCall', 'id': 'c1', 'name': 'bash', 'arguments': {'command': 'ls'}}]}},
                {'type': 'message', 'id': 'r1', 'parentId': 'a1', 'message': {'role': 'toolResult', 'toolCallId': 'c1',
                    'toolName': 'bash', 'content': [{'type': 'text', 'text': 'file x'}]}},
                {'type': 'message', 'id': 'a2', 'parentId': 'r1', 'message': {'role': 'assistant', 'stopReason': 'length',
                    'content': [{'type': 'thinking', 'thinking': 'long'}]}},
                {'type': 'message', 'id': 'a3', 'parentId': 'a2', 'message': {'role': 'assistant', 'stopReason': 'stop',
                    'content': [{'type': 'text', 'text': CAP_MARK}]}}]
            (sessions / 's.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            tools = [{'type': 'function', 'function': {'name': 'bash', 'description': 'b', 'parameters': {'type': 'object'}}}]
            log = WireLog(str(folder / 'harbor' / 'pi-cont16k' / 'wire.jsonl'))
            seq = log.request(json.dumps({'messages': [{'role': 'system', 'content': 'You are Pi.'},
                                                       {'role': 'user', 'content': 'Do it'}], 'tools': tools}).encode(), 'b', '/v1')
            log.end(seq, 200)
            log.close()
            record = export_one(campaign, folder, serializer_for(campaign.teacher))
            weights = {m['event_id']: m['loss_weight'] for s in record['segments'] for m in s['messages']}
            self.assertEqual(weights, {'u': 0, 'a1': 1, 'r1': 0, 'a2': 0, 'a3': 0})
            serializer = serializer_for(campaign.teacher)
            whole = serializer.render([record['system_message']] + [m['message'] for s in record['segments']
                                                                     for m in s['messages']], tools)
            self.assertEqual(record['tokens'], serializer.count(whole))
            self.assertEqual(len(record['segments']), 1)


class Config(unittest.TestCase):
    def test_top8_campaign_freezes(self):
        path = REPO / 'synth/campaigns/nemotron35-super-top8.toml'
        raw = tomllib.loads(path.read_text())
        if not Path(raw['train']['tasks']).exists():
            self.skipTest('campaign inputs not mounted')
        frozen = build(path)
        self.assertEqual(len(frozen['train']['task_list']), 479)
        self.assertEqual(frozen['teacher']['slug'], 'nemotron35-super')
        self.assertEqual(frozen['workers']['slots'], 8)
        self.assertEqual(frozen['serving']['gpus_per_job'], 4)
        self.assertEqual(frozen['admission']['initial_limit_per_backend'], 128)
        self.assertFalse(Campaign(frozen).enabled('eval'))


if __name__ == '__main__':
    unittest.main()


def normalized(argv, model, draft=None):
    """vLLM flags as {flag: value}; JSON values parsed, model/draft paths and ports abstracted."""
    start = argv.index('serve') + 1
    tokens = argv[start:]
    flags = {'_model': '{model}' if tokens[0] == model else tokens[0]}
    i = 1
    while i < len(tokens):
        flag = tokens[i]
        assert flag.startswith('--'), (flag, tokens)
        value = True
        if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
            value = tokens[i + 1]
            i += 1
        if isinstance(value, str):
            is_json = value[:2] == '{"' or value[:1] == '['
            value = value.replace(model, '{model}')
            if draft:
                value = value.replace(draft, '{draft}')
            if is_json:
                value = json.loads(value)
            else:
                try:
                    value = float(value)  # '.92' and '0.92' are the same flag value
                except ValueError:
                    pass
        flags[flag] = '{port}' if flag in ('--port',) else value
        i += 1
    return flags


class RecordedCommands(unittest.TestCase):
    """Each teacher spec reproduces the vLLM command its completed campaign recorded."""

    def test_specs_match_recorded_campaign_commands(self):
        recorded = json.loads((Path(__file__).parent / 'fixtures' / 'recorded-commands.json').read_text())
        for slug, entry in recorded.items():
            with self.subTest(slug):
                teacher = load_teacher(slug)
                argv = entry['argv']
                model = argv[argv.index('serve') + 1]
                draft = next((json.loads(argv[i + 1])['model'] for i, a in enumerate(argv)
                              if a == '--speculative-config' and 'model' in json.loads(argv[i + 1])), None)
                tunables = {**teacher['serving'].get('tunables', {})}
                ours = replica_command(teacher, RECORDED_PROTOCOL, port=1, model_dir=model,
                                       tunables=tunables, rank_ports={'dp_size': 4, 'stage_draft': draft or '/d'})
                self.assertEqual(normalized(ours, model, draft), normalized(argv, model, draft))


    def test_deepseek_1017_is_v41_production_with_the_tuned_layout(self):
        """deepseek-v41-flash-1017 = deepseek-v41-flash's production command on its own weights, except the
        layout the tuning campaign selected (DP4/EP4) and vLLM's B200 AgentX profile at 768 seqs per rank
        (cpu401 paired replay 2026-10-23, cpu305 max-num-seqs sweep 2026-10-24)."""
        recorded = json.loads((Path(__file__).parent / 'fixtures' / 'recorded-commands.json').read_text())
        argv = recorded['deepseek-v41-flash']['argv']
        model = argv[argv.index('serve') + 1]
        teacher = load_teacher('deepseek-v41-flash-1017')
        ours = normalized(replica_command(teacher, RECORDED_PROTOCOL, port=1, model_dir=model,
                                          tunables=teacher['serving']['tunables'], rank_ports={'dp_size': 4}), model)
        expected = normalized(['deepseek-v41-flash-1017' if a == 'deepseek-v41-flash' else a for a in argv], model)
        changed = {k for k in set(ours) | set(expected) if ours.get(k) != expected.get(k)}
        self.assertEqual(changed, {'--tensor-parallel-size', '--data-parallel-size', '--enable-expert-parallel',
                                   '--engram-config', '--max-num-seqs', '--gpu-memory-utilization',
                                   '--kernel-config', '--compilation-config'})
        self.assertEqual((ours['--tensor-parallel-size'], ours['--data-parallel-size']), (1.0, 4.0))
        self.assertTrue(ours['--engram-config']['embedding_across_dp'])
        self.assertEqual((ours['--max-num-seqs'], ours['--gpu-memory-utilization']), (768.0, 0.97))
        self.assertEqual(ours['--kernel-config']['moe_backend'], 'deep_gemm_mega_moe')
        self.assertEqual(max(ours['--compilation-config']['cudagraph_capture_sizes']), 8190)
        self.assertEqual(teacher['serving']['mode'], 'dp_internal')
        self.assertEqual(teacher['runtime']['env']['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS'], '1200')
        self.assertEqual(teacher['runtime']['env']['VLLM_USE_V2_MODEL_RUNNER'], '1')


class GLMLauncher(unittest.TestCase):
    def test_glm_matches_dp_ep_launcher(self):
        """glm53-flash.toml == inference/serve_glm_dpep.sh with the campaign's env and extra args."""
        teacher = load_teacher('glm53-flash')
        port = 12345
        with tempfile.TemporaryDirectory() as d:
            venv = Path(d) / 'venv'
            (venv / 'bin').mkdir(parents=True)
            (venv / 'lib/python3.12/site-packages/nvidia/cu13/lib').mkdir(parents=True)
            fake = venv / 'bin' / 'vllm'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = {**os.environ, 'VENV': str(venv), 'MODEL': '/m', 'SCRATCH': str(Path(d) / 's'),
                   'CACHE_TAR': str(Path(d) / 'absent.tar'), 'MAX_LEN': '132000', 'MTP_TOKENS': '2',
                   'PREFIX_MATCH_UNIT': '128', 'FLASHINFER_MOE': '1', 'OFFLOAD_GB': '32', 'MAX_SEQS': '256',
                   'MAX_BATCHED': '16384', 'GPU_MEM_UTIL': '.92'}
            extra = ['--enable-ep-weight-filter', '--enable-auto-tool-choice', '--tool-call-parser', 'glm47',
                     '--generation-config', 'vllm', '--override-generation-config',
                     '{"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 16384}',
                     '--chat-template', '/m/chat_template.jinja', '--seed', '0', '--dp-supervisor-probe-timeout-s', '120']
            out = subprocess.run(['bash', str(REPO / 'inference/serve_glm_dpep.sh'), '0,1,2,3', str(port), *extra],
                                 env=env, capture_output=True, text=True, check=True).stdout
        recorded = ['vllm'] + json.loads(out.strip().splitlines()[-1])
        ours = replica_command(teacher, RECORDED_PROTOCOL, port=port, model_dir='/m',
                               tunables=teacher['serving']['tunables'],
                               rank_ports={'dp_size': 4, 'rpc_port': port + 13000, 'supervisor_port': port + 1155})
        self.maxDiff = None
        self.assertEqual(normalized(ours, '/m'), normalized(recorded, '/m'))

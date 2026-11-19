"""snowball-eval: one spec for every Snowball-architecture checkpoint (serve_teacher --checkpoint), launching eval_q35's
Snowball TB recipe (eval-snowflake-tb 0bacc3c: eval_q35/common.py SNOWBALL_ENV / SNOWBALL_HF_OVERRIDES, submit.py
snowball_config, serve_job.sh profile snowball) at the 128k eval cap."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from synth.common import REPO
from synth.config import load_teacher
from synth.vllm_launch import replica_command, replica_env
from task_synthesis.serve.serve_teacher import PROFILES, check_checkpoint, checkpoint_overrides

# eval_q35/common.py at eval-snowflake-tb 0bacc3c
SNOWBALL_ENV = {'GRUG_SWA_KEEP': '1', 'GRUG_SEEDED_SAMPLER': '1', 'GRUG_TRITON_ROUTING': '1', 'GRUG_FP8_POW2': '1',
                'VLLM_USE_DEEP_GEMM': '0', 'VLLM_ENGINE_READY_TIMEOUT_S': '3600'}
SNOWBALL_HF_OVERRIDES = {'quantization_config': {'quant_method': 'fp8', 'activation_scheme': 'dynamic',
                                                 'weight_block_size': [128, 128]},
                         'architectures': ['GrugMoeServeForCausalLM']}
SNOWBALL_TEMPLATE_SHA256 = 'cb728ba63d3c6393bbce328a7c6a00359f259b7b9b1bbdc074b1ad53e5029f7c'
_TMP = tempfile.TemporaryDirectory()   # engine scratch dirs of replica_env, removed at exit


def launch(checkpoint=None, cuda_arch=None):
    teacher = load_teacher('snowball-eval', {'checkpoint': checkpoint} if checkpoint else None)
    profile = PROFILES['snowball-eval']
    tunables = {**teacher['serving'].get('tunables', {}), **profile.get('tunables', {}),
                **({'fi_aot_arch': cuda_arch} if cuda_arch else {})}
    argv = replica_command(teacher, profile['protocol'], port=20000, model_dir='/tmp/synth/model', tunables=tunables)
    env = replica_env(teacher, gpus=['0'], scratch=tempfile.mkdtemp(prefix='snowball-eval-test-', dir=_TMP.name), tunables=tunables,
                      base_env={'PATH': '/usr/bin:/bin'})
    return teacher, argv, env


def flag(argv, name):
    return argv[argv.index(name) + 1]


def fake_checkpoint(root, template=b'{{ bos_token }}', archs=('GrugMoeForCausalLM',)):
    root = Path(root)
    (root / 'config.json').write_text(json.dumps({'architectures': list(archs)}))
    (root / 'chat_template.jinja').write_bytes(template)
    (root / 'model.safetensors.index.json').write_text('{}')
    return root


class Recipe(unittest.TestCase):
    def test_eval_q35_snowball_flags(self):
        _, argv, _ = launch()
        self.assertEqual(argv[:2], ['/home/user/vllm0201/bin/vllm', 'serve'])
        self.assertEqual(json.loads(flag(argv, '--hf-overrides')), SNOWBALL_HF_OVERRIDES)
        for name, value in [('--quantization', 'fp8'), ('--kv-cache-dtype', 'fp8'), ('--max-num-batched-tokens', '16384'),
                            ('--max-num-seqs', '256'), ('--gpu-memory-utilization', '0.9'), ('--api-server-count', '4'),
                            ('--tensor-parallel-size', '1'), ('--reasoning-parser', 'snowball'),
                            ('--tool-call-parser', 'snowball'), ('--generation-config', 'vllm'), ('--seed', '0'),
                            ('--max-model-len', '131072'), ('--chat-template', '/tmp/synth/model/chat_template.jinja')]:
            self.assertEqual(flag(argv, name), value, name)
        self.assertEqual(json.loads(flag(argv, '--override-generation-config')), {'temperature': 1.0, 'top_p': 0.95})
        parsers = str(REPO / 'serve_snowball' / 'snowball_parsers.py')
        self.assertEqual(flag(argv, '--reasoning-parser-plugin'), parsers)
        self.assertEqual(flag(argv, '--tool-parser-plugin'), parsers)
        self.assertTrue(Path(parsers).is_file())
        for name in ('--enable-prefix-caching', '--enable-auto-tool-choice', '--enable-prompt-tokens-details',
                     '--language-model-only', '--no-enable-log-requests'):
            self.assertEqual(argv.count(name), 1, name)
        self.assertNotIn('--limit-mm-per-prompt', argv)   # text only: aa_evals reads --language-model-only
        self.assertNotIn('--enforce-eager', argv)          # the plugin compiles the model (GRUG_COMPILE default on)

    def test_engine_environment(self):
        _, _, env = launch()
        for k, v in SNOWBALL_ENV.items():
            self.assertEqual(env[k], v, k)
        plugin = REPO / 'serve_snowball' / 'plugin'
        self.assertEqual(env['PYTHONPATH'], str(plugin))
        self.assertTrue((plugin / 'grug_vllm' / '__init__.py').is_file())
        self.assertIn('grug = grug_vllm:register', (plugin / 'grug_vllm-0.1.dist-info' / 'entry_points.txt').read_text())
        self.assertEqual(env['CUDA_HOME'], '/is/software/nvidia/cuda-13.2')
        self.assertTrue(env['PATH'].startswith('/is/software/nvidia/cuda-13.2/bin:/home/user/vllm0201/bin:'))
        self.assertEqual(env['GRUG_FI_AOT_DIR'], '/data/aai/tbenchtrain/snowball_serve/fi_aot/100a')
        self.assertEqual(launch(cuda_arch='90a')[2]['GRUG_FI_AOT_DIR'],
                         '/data/aai/tbenchtrain/snowball_serve/fi_aot/90a')
        for name in ('HOME', 'XDG_CACHE_HOME', 'FLASHINFER_WORKSPACE_BASE', 'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT'):
            self.assertIn('snowball-eval-test-', env[name], name)   # node-local, never the shared home caches
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual((env['CUDA_VISIBLE_DEVICES'], env['CUDA_DEVICE_ORDER']), ('0', 'PCI_BUS_ID'))
        self.assertEqual(env['VLLM_WORKER_MULTIPROC_METHOD'], 'spawn')

    def test_request_and_admission(self):
        teacher, _, _ = launch()
        req = teacher['request']
        # eval_q35's proxy on every Snowball request (no registry overrides there): T 1.0 / top_p 0.95, effort high
        self.assertEqual(req['overrides'], {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'high',
                                            'chat_template_kwargs': {'clear_thinking': False}})
        self.assertNotIn('pi_thinking', req)   # Pi keeps the eval_q35 protocol's own level
        self.assertTrue(req['ctx_gate'])
        self.assertEqual((req['ctx_fill'], req['kv_gate']), (0.7, '0.80,0.72'))
        self.assertNotIn('gate', PROFILES['snowball-eval'])   # the turn gate runs with pi_eval: no gate in front
        self.assertEqual(teacher['serving_resources']['cuda_capability'], '10.0')
        self.assertTrue(teacher['serving_resources']['bypass_lxcfs'])
        self.assertEqual(teacher['serving'].get('probe'), 'log')            # student: probes logged, not fatal
        self.assertEqual(load_teacher('qwen35-4b-eval')['serving'].get('probe', 'fatal'), 'fatal')
        self.assertEqual((teacher['serving_resources']['default_gpus'], teacher['serving_resources']['default_bid']), (1, 61))


class Probe(unittest.TestCase):
    def test_log_mode_records_and_survives(self):
        from unittest import mock
        from task_synthesis.serve import serve_teacher as st
        teacher = load_teacher('snowball-eval')
        with tempfile.TemporaryDirectory() as d, mock.patch.object(st, 'probe', side_effect=RuntimeError('Tool parsing probe failed')):
            rec = st.run_probe(Path(d), 'http://x/v1', teacher)
            self.assertFalse(rec['ok'])
            self.assertIn('Tool parsing', (Path(d) / 'probe.jsonl').read_text())
        other = load_teacher('qwen35-4b-eval')
        with mock.patch.object(st, 'probe', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                st.run_probe(Path('/nonexistent'), 'http://x/v1', other)


class DeviceIndices(unittest.TestCase):
    def test_uuids_map_to_nvidia_smi_indices(self):
        from unittest import mock
        from synth import gpu
        smi = SimpleNamespace(stdout='0, GPU-aaaa-1\n1, GPU-bbbb-2\n')
        with mock.patch.object(gpu.subprocess, 'run', return_value=smi) as run:
            self.assertEqual(gpu.indices(['GPU-bbbb-2', 'GPU-aaaa-1']), ['1', '0'])
            self.assertEqual(gpu.indices(['1']), ['1'])
            self.assertEqual(run.call_count, 1)
            with self.assertRaises(RuntimeError):
                gpu.indices(['GPU-cccc-3'])


class PerCheckpoint(unittest.TestCase):
    def test_checkpoint_override_reaches_the_spec(self):
        with tempfile.TemporaryDirectory() as d:
            teacher, argv, _ = launch(checkpoint=d)
            self.assertEqual(teacher['checkpoint'], d)
            self.assertEqual(argv[2], '/tmp/synth/model')   # the staged copy, the same path for every checkpoint
            self.assertEqual(checkpoint_overrides(SimpleNamespace(checkpoint=d)), {'checkpoint': str(Path(d).resolve())})
            self.assertIsNone(checkpoint_overrides(SimpleNamespace(checkpoint='')))

    def test_checks_accept_a_snowball_export(self):
        tpl = REPO / 'serve_snowball' / 'snowball_chat_template.jinja'
        with tempfile.TemporaryDirectory() as d:
            body = b'snowball template bytes'
            teacher = load_teacher('snowball-eval', {'checkpoint': str(fake_checkpoint(d, body)),
                                                     'checkpoint_checks': {'chat_template_sha256': [
                                                         hashlib.sha256(body).hexdigest()]}})
            self.assertEqual(check_checkpoint(teacher)['checkpoint'], d)
        self.assertIn(SNOWBALL_TEMPLATE_SHA256, load_teacher('snowball-eval')['checkpoint_checks']['chat_template_sha256'])
        self.assertFalse(tpl.exists())   # the template comes from each checkpoint, checked by hash

    def test_checks_refuse_other_templates_and_architectures(self):
        with tempfile.TemporaryDirectory() as d:
            teacher = load_teacher('snowball-eval', {'checkpoint': str(fake_checkpoint(d, b'datakit'))})
            with self.assertRaisesRegex(SystemExit, 'chat_template.jinja sha256'):
                check_checkpoint(teacher)
        with tempfile.TemporaryDirectory() as d:
            teacher = load_teacher('snowball-eval', {'checkpoint': str(fake_checkpoint(d, archs=['Qwen3ForCausalLM']))})
            with self.assertRaisesRegex(SystemExit, 'architectures'):
                check_checkpoint(teacher)

    def test_specs_without_checks_are_unchanged(self):
        teacher = load_teacher('qwen35-4b-eval')
        self.assertEqual(check_checkpoint(teacher), {'checkpoint': teacher['checkpoint']})


if __name__ == '__main__':
    unittest.main()

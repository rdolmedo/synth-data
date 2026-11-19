"""snowball-eval-main (#478): snowball-eval's recipe on vLLM main. Only the engine build and what it needs differ."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from synth.config import load_teacher
from synth.vllm_launch import replica_command, replica_env
from task_synthesis.serve import serve_teacher as st
from task_synthesis.serve.serve_teacher import PROFILES

_TMP = tempfile.TemporaryDirectory(prefix='snowball-main-test-')


def build(slug):
    teacher = load_teacher(slug)
    tunables = {**teacher['serving'].get('tunables', {}), **PROFILES[slug].get('tunables', {})}
    argv = replica_command(teacher, PROFILES[slug]['protocol'], port=8000, model_dir='/model', tunables=tunables)
    env = replica_env(teacher, gpus=['0'], scratch=tempfile.mkdtemp(dir=_TMP.name), tunables=tunables, base_env={})
    return teacher, argv, env


class SnowballEvalMain(unittest.TestCase):
    def test_only_the_build_differs_from_snowball_eval(self):
        old, old_argv, old_env = build('snowball-eval')
        new, new_argv, new_env = build('snowball-eval-main')
        self.assertTrue(new['runtime']['venv'].startswith('/data/aai/venvs/vllm-0.34.0rc1.dev815'))
        self.assertEqual(new['runtime']['vllm_version'], '0.34.0rc1.dev815+g8f3c672dc')
        for key in ('checkpoint_checks', 'request', 'checkpoint', 'served_name'):
            self.assertEqual(new[key], old[key], key)
        res = dict(old['serving_resources']); del res['cpus_per_extra_engine']   # one engine per GPU here
        self.assertEqual(new['serving_resources'], {**res, 'cuda_capabilities': ['10.0']})   # B200 only
        strip = lambda argv: [a.replace('snowball_parsers_v034.py', 'snowball_parsers.py') for a in argv[1:]]
        old_rest = [a for a in old_argv[1:] if a not in ('--gpu-memory-utilization', '0.9')]
        self.assertEqual(strip(new_argv), old_rest)                     # same flags, parsers file and memory flag aside
        self.assertNotIn('--gpu-memory-utilization', new_argv)          # 0.34 sizes its extensible KV after warmup
        self.assertEqual(sum('snowball_parsers_v034.py' in a for a in new_argv), 2)
        # the engine environment: the same except the build's own entries (OMP_NUM_THREADS, caches, ... included)
        scratch = lambda env: {k: v for k, v in env.items() if 'snowball-main-test' not in str(v)}   # per-call dirs
        old_env, new_env = scratch(old_env), scratch(new_env)
        differs = {'PATH', 'PYTHONPATH', 'GRUG_FI_AOT_DIR', 'GRUG_SEEDED_SAMPLER', 'VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT',
                 'FLASHINFER_NO_DOWNLOAD'}
        self.assertEqual({k for k in set(old_env) | set(new_env) if old_env.get(k) != new_env.get(k)}, differs)
        self.assertEqual({k: v for k, v in new_env.items() if k not in differs},
                         {k: v for k, v in old_env.items() if k not in differs})

    def test_engine_environment(self):
        _, _, env = build('snowball-eval-main')
        self.assertEqual(env['VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT'], '600')   # 0.34's 30 s default fails long renders
        self.assertNotIn('GRUG_SEEDED_SAMPLER', env)                        # native on the V2 runner
        self.assertTrue(env['GRUG_FI_AOT_DIR'].endswith('/fi_aot_074/100a'))
        self.assertIn('/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132/bin', env['PATH'])
        self.assertTrue(env['PYTHONPATH'].endswith('/serve_snowball/plugin_v034'))   # never the 0.20.1 plugin
        for name in ('GRUG_SWA_KEEP', 'GRUG_TRITON_ROUTING', 'GRUG_FP8_POW2'):
            self.assertEqual(env[name], '1', name)
        self.assertEqual(env['VLLM_USE_DEEP_GEMM'], '0')

    def test_one_engine_per_gpu(self):
        # no gpu_util to split k ways: a clear refusal at submit, never a KeyError or two engines that each take the GPU
        teacher = load_teacher('snowball-eval-main')
        self.assertNotIn('cpus_per_extra_engine', teacher['serving'])
        tunables = {**teacher['serving'].get('tunables', {}), 'engines_per_gpu': 2}
        with self.assertRaisesRegex(ValueError, 'one engine per GPU'):
            st.engine_tunables(tunables, 2)
        self.assertEqual(st.engine_tunables(tunables, 1), tunables)
        self.assertEqual(st.engine_tunables({'gpu_util': 0.9}, 2)['gpu_util'], 0.45)   # specs with gpu_util: as before

    def submit(self, cuda_cap):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(st, 'ROOT', Path(d)), \
                mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')), \
                mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n')):
            a = mock.Mock(teacher='snowball-eval-main', gpus=1, force=False, tunables='', protocol='', cuda_cap=cuda_cap,
                          hours=1, bid=61, cpus=0, memory_gib=0, checkpoint='', dry_run=False, exclude_nodes='',
                          bypass_lxcfs=False, pipeline='aa-serving', fill=None)
            a.name = 'n'
            st.submit(a)
            return json.loads((Path(d) / 'n' / 'job.json').read_text())

    def test_b200_only(self):
        self.assertEqual(self.submit('')['cuda_cap'], '10.0')
        with self.assertRaisesRegex(SystemExit, 'CUDA capability 10.0 only, not 9.0'):
            self.submit('9.0')


if __name__ == '__main__':
    unittest.main()

"""Effective-numerics records (synth/numerics.py): a stable hash per recipe, spec metadata that matches the weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from synth import engine_cache, numerics
from synth.config import TEACHERS, load_teacher
from synth.vllm_launch import replica_command, replica_env
from task_synthesis.serve.serve_teacher import PROFILES, PROTOCOL, served_identity

RANKS = {'dp_size': 2, 'rpc_port': 21000, 'supervisor_port': 22000, 'stage_draft': '/d'}
NVFP4_REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'


def launch(slug, *, port, root, model_dir=None, tunables=None, base_env=None):
    teacher = load_teacher(slug)
    profile = PROFILES.get(slug, {})
    protocol = profile.get('protocol', PROTOCOL)
    tun = {**teacher['serving'].get('tunables', {}), **profile.get('tunables', {}), **(tunables or {})}
    model_dir = model_dir or str(Path(root) / 'model')
    argv = replica_command(teacher, protocol, port=port, model_dir=model_dir, tunables=tun,
                           rank_ports={**RANKS, 'rpc_port': port + 1, 'supervisor_port': port + 2})
    env = replica_env(teacher, gpus=['0'], scratch=Path(root) / 'engines' / 'r0', tunables=tun,
                      base_env=base_env if base_env is not None else {'PATH': '/usr/bin:/bin'})
    return teacher, protocol, argv, env, model_dir


def fake_checkpoint(d, quant=None, template='{{ messages }}'):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'config.json').write_text(json.dumps({'torch_dtype': 'bfloat16',
                                               **({'quantization_config': quant} if quant else {})}))
    (d / 'chat_template.jinja').write_text(template)
    return d


class Numerics(unittest.TestCase):
    def record(self, slug, port, root, **kw):
        teacher, protocol, argv, env, model_dir = launch(slug, port=port, root=root, **kw)
        base = {'PATH': '/usr/bin:/bin'}
        return numerics.replica_numerics(teacher, argv, {**env, 'CUDA_VISIBLE_DEVICES': f'GPU-{port}'},
                                         model_dir=model_dir, scratch_roots=(root,), protocol=protocol,
                                         gpu=['NVIDIA B200'], base_env=base)

    def test_hash_ignores_ports_scratch_and_gpu_assignment(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            fake_checkpoint(Path(a) / 'model'), fake_checkpoint(Path(b) / 'model')
            for slug in ('qwen38-27b-eval-fast', 'glm53-flash-eval', 'nemotron35-lightning-eval'):
                with self.subTest(slug=slug):
                    one, two = self.record(slug, 20000, a), self.record(slug, 30000, b)
                    self.assertEqual(one['sha256'], two['sha256'])
                    self.assertNotIn(a, json.dumps(one))
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', one['env'])

    def test_hash_moves_with_numerics(self):
        with tempfile.TemporaryDirectory() as d:
            fake_checkpoint(Path(d) / 'model')
            slug = 'nemotron35-lightning-eval'
            base = self.record(slug, 20000, d)
            bf16 = self.record(slug, 20000, d, tunables={'quantization': ''})   # the B200 override
            self.assertNotEqual(base['sha256'], bf16['sha256'])
            self.assertEqual(base['weights']['quantization']['label'], 'fp8 (online, from bfloat16)')
            self.assertEqual(bf16['weights']['quantization']['label'], 'bfloat16')
            env = self.record(slug, 20000, d, tunables={'extra_env': {'VLLM_X': '1'}})
            self.assertNotEqual(base['sha256'], env['sha256'])
            (Path(d) / 'model' / 'chat_template.jinja').write_text('other')
            self.assertNotEqual(base['sha256'], self.record(slug, 20000, d)['sha256'])

    def test_recording_changes_neither_command_nor_cache_key(self):
        with tempfile.TemporaryDirectory() as d:
            fake_checkpoint(Path(d) / 'model')
            teacher, protocol, argv, env, model_dir = launch('qwen38-27b-eval-fast', port=20000, root=d)
            before = (copy.deepcopy(argv), copy.deepcopy(env), copy.deepcopy(teacher))
            key = engine_cache.config_key(argv, env, teacher['runtime']['venv'], ['NVIDIA B200'])[0]
            numerics.replica_numerics(teacher, argv, env, model_dir=model_dir, scratch_roots=(d,), protocol=protocol)
            self.assertEqual((argv, env, teacher), before)
            self.assertEqual(engine_cache.config_key(argv, env, teacher['runtime']['venv'], ['NVIDIA B200'])[0], key)

    def test_a_pinned_venv_declares_its_own_version(self):          # rt-vllm #478: profile tunables venv + vllm_version
        teacher = {'runtime': {'venv': '/nonexistent/spec-venv', 'vllm_version': '0.34'}, 'model': 'm'}
        rec = numerics.replica_numerics(teacher, ['vllm', 'serve', '/m'], {}, model_dir='/nonexistent', protocol={},
                                        venv='/nonexistent/pinned', declared_version='0.32')
        self.assertEqual(rec['vllm']['declared_version'], '0.32')
        rec = numerics.replica_numerics(teacher, ['vllm', 'serve', '/m'], {}, model_dir='/nonexistent', protocol={})
        self.assertEqual(rec['vllm']['declared_version'], '0.34')

    def test_never_raises(self):
        rec = numerics.safe_replica_numerics(None, [], {}, model_dir='/nonexistent')
        self.assertIsNone(rec['sha256'])
        self.assertIn('error', rec)

    def test_quantization_label(self):
        with tempfile.TemporaryDirectory() as d:
            fake_checkpoint(d, {'quant_method': 'fp8', 'fmt': 'e4m3', 'weight_block_size': [128, 128]})
            self.assertEqual(numerics.weight_quantization(['vllm'], d)['label'], 'fp8/e4m3')
            over = json.dumps({'quantization_config': {'quant_method': 'fp8'}})
            q = numerics.weight_quantization(['vllm', '--hf-overrides', over], Path(d) / 'missing')
            self.assertEqual((q['label'], q['checkpoint']['source']), ('fp8', 'hf-overrides'))

    def test_job_hash(self):
        self.assertEqual(numerics.job_numerics_sha256([{'sha256': 'a'}, {'sha256': 'a'}]), 'a')
        self.assertNotIn(numerics.job_numerics_sha256([{'sha256': 'a'}, {'sha256': 'b'}]), ('a', 'b'))
        ident = served_identity({'model': 'm', 'revision': 'r'},
                                [{'weights': {'quantization': {'label': 'fp8'}}, 'sha256': 'a'},
                                 {'weights': {'quantization': {'label': 'bfloat16'}}, 'sha256': 'b'}])
        self.assertEqual(ident['weights_quantization'], ['bfloat16', 'fp8'])


class SpecMetadata(unittest.TestCase):
    """The spec's own identity fields describe the weights it serves (2026-11-14 serving audit, 2.4 / 2.6)."""

    def test_declared_quantization_is_what_the_engine_applies(self):
        checked = 0
        for path in sorted(TEACHERS.glob('*.toml')):
            teacher = load_teacher(path.stem)
            if 'quantization' not in teacher:
                continue
            if not (Path(teacher['checkpoint']) / 'config.json').is_file():
                continue   # weights not on this machine
            with self.subTest(slug=path.stem):
                _, _, argv, _, _ = launch(path.stem, port=20000, root='/tmp/synth-numerics-test',
                                          model_dir=teacher['checkpoint'])
                self.assertEqual(numerics.weight_quantization(argv, teacher['checkpoint'])['label'],
                                 teacher['quantization'])
                checked += 1
        if not checked:
            self.skipTest('no checkpoint readable')

    def test_revision_matches_the_downloaded_build(self):
        for path in sorted(TEACHERS.glob('*.toml')):
            teacher = load_teacher(path.stem)
            meta = Path(teacher['checkpoint']) / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
            try:
                downloaded = meta.read_text().split()[0]
            except (OSError, IndexError):
                continue
            with self.subTest(slug=path.stem):
                self.assertTrue(teacher['revision'].startswith(downloaded), (teacher['revision'], downloaded))

    def test_qwen38_fp8_specs_do_not_name_the_nvfp4_build(self):
        for path in sorted(TEACHERS.glob('qwen38-27b*.toml')):
            teacher = load_teacher(path.stem)
            with self.subTest(slug=path.stem):
                fp8 = teacher['checkpoint'].endswith('Qwen3.8-27B-FP8')
                self.assertEqual(teacher['revision'] == NVFP4_REVISION, not fp8)
                self.assertEqual(teacher['quantization'], 'fp8/e4m3' if fp8 else 'modelopt/MIXED_PRECISION')


if __name__ == '__main__':
    unittest.main()

"""Super eval specs on vLLM main (#478): the engine build and the supersede plugin they load."""
import configparser
import filecmp
import importlib
from pathlib import Path
import re
import sys
import tomllib
import types
import unittest
from unittest import mock

from synth.config import load_teacher
from task_synthesis.serve.serve_teacher import PIPELINE_PROFILES, PROFILES

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / 'synth' / 'vllm_plugins' / 'cpu_supersede'
DEPLOYED = Path('/data/aai/tbench/vllm_plugins/cpu_supersede_v3')
EOT = REPO / 'synth' / 'vllm_plugins' / 'eot_checkpoint'
EOT_DEPLOYED = Path('/data/aai/tbench/vllm_plugins/eot_checkpoint_v5')
SPECS = ('nemotron35-super-mm', 'nemotron35-super-mm-text')
ALL = SPECS + ('nemotron35-super',)   # + the data-generation teacher


class SuperOnMain(unittest.TestCase):
    def test_eval_specs_run_main_with_supersede_v3_and_eot_v5(self):
        for slug in SPECS:
            with self.subTest(slug=slug):
                t = load_teacher(slug)
                self.assertEqual(t['runtime']['venv'], '/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132')
                self.assertEqual(t['runtime']['vllm_version'], '0.34.0rc1.dev815+g8f3c672dc')
                env = t['serving']['env']
                self.assertEqual(env['PYTHONPATH'], f'{DEPLOYED}:{EOT_DEPLOYED}')
                self.assertEqual(env['SYNTH_CPU_SUPERSEDE'], '1')
                self.assertEqual(env['SYNTH_EOT_CHECKPOINT'], '1')
                self.assertEqual(env['VLLM_USE_SIMPLE_KV_OFFLOAD'], '1')
                self.assertEqual(t['runtime']['env']['FLASHINFER_NO_DOWNLOAD'], '1')

    def test_the_data_generation_teacher_runs_main(self):
        t = load_teacher('nemotron35-super')   # its campaigns add the plugins (PluginPairing checks them)
        self.assertEqual(t['runtime']['venv'], '/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132')
        self.assertEqual(t['runtime']['env']['FLASHINFER_NO_DOWNLOAD'], '1')

    def test_vl_profiles_seed_from_the_staged_snapshot(self):
        for slug in SPECS:   # first starts skip the ~11 min compile (#552 review); staged-path JIT modules (#556)
            self.assertEqual(PROFILES[slug]['seed_from'], '15282107916d460058f1', slug)
            self.assertTrue(PROFILES[slug]['tunables']['stage_venv'], slug)

    def test_the_repo_plugin_is_3_0_with_a_version_dispatching_entry_point(self):
        ep = configparser.ConfigParser()
        ep.read(PLUGIN / 'synth_cpu_supersede-3.0.dist-info' / 'entry_points.txt')
        self.assertEqual(ep['vllm.general_plugins']['synth_cpu_supersede'], 'synth_cpu_supersede._entry:register')

    @unittest.skipUnless(DEPLOYED.is_dir() and EOT_DEPLOYED.is_dir(), 'deployed plugins not readable here')
    def test_the_deployed_copies_are_the_repo_packages(self):
        # specs load the read-only copies under /data; they must be this commit's packages and dist-info, byte for byte
        for repo, deployed, subs in ((PLUGIN, DEPLOYED, ('synth_cpu_supersede', 'synth_cpu_supersede-3.0.dist-info')),
                                     (EOT, EOT_DEPLOYED, ('synth_eot_checkpoint', 'synth_eot_checkpoint-5.0.dist-info'))):
            for sub in subs:
                files = sorted(p.name for p in (repo / sub).iterdir() if p.is_file() and p.suffix != '.pyc')
                self.assertEqual(sorted(p.name for p in (deployed / sub).iterdir() if p.is_file() and p.suffix != '.pyc'),
                                 files, sub)
                match, mismatch, errors = filecmp.cmpfiles(repo / sub, deployed / sub, files, shallow=False)
                self.assertEqual((mismatch, errors), ([], []), sub)

    @unittest.skipUnless(Path('/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132').is_dir(), 'venv not readable')
    def test_declared_vllm_version_is_the_installed_one(self):
        site = Path('/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132/lib/python3.12/site-packages')
        for slug in ALL:
            with self.subTest(slug=slug):
                self.assertTrue((site / f"vllm-{load_teacher(slug)['runtime']['vllm_version']}.dist-info").is_dir())

    def test_entry_dispatches_on_the_imported_vllm(self):
        sys.path.insert(0, str(PLUGIN))
        try:
            entry = importlib.import_module('synth_cpu_supersede._entry')
            for version, impl in (('0.32.0rc1.dev190+g3f1bcfd82', 'synth_cpu_supersede'), ('0.33.1', 'synth_cpu_supersede'),
                                  ('0.34.0rc1.dev815+g8f3c672dc', 'synth_cpu_supersede.vllm034'), ('dev', None)):
                with self.subTest(version=version):
                    called = []
                    fakes = {'synth_cpu_supersede': types.SimpleNamespace(register=lambda: called.append('synth_cpu_supersede')),
                             'synth_cpu_supersede.vllm034': types.SimpleNamespace(register=lambda: called.append('synth_cpu_supersede.vllm034'))}
                    fake_pkg = types.ModuleType('synth_cpu_supersede'); fake_pkg.register = fakes['synth_cpu_supersede'].register
                    fake_pkg.vllm034 = fakes['synth_cpu_supersede.vllm034']
                    with mock.patch.dict(sys.modules, {'vllm': types.SimpleNamespace(__version__=version),
                                                       'synth_cpu_supersede': fake_pkg,
                                                       'synth_cpu_supersede.vllm034': fake_pkg.vllm034}), \
                            mock.patch.dict('os.environ', {'SYNTH_CPU_SUPERSEDE': '1'}):
                        entry.register()
                    self.assertEqual(called, [impl] if impl else [])
        finally:
            sys.path.remove(str(PLUGIN))
            for name in [m for m in sys.modules if m.startswith('synth_cpu_supersede')]:
                del sys.modules[name]

    def test_eot_entry_runs_v5_only_on_a_validated_build(self):
        sys.path.insert(0, str(EOT))
        try:
            entry = importlib.import_module('synth_eot_checkpoint._entry')
            cases = (('0.32.0rc1.dev190+g3f1bcfd82', 'v4'), ('0.33.1', 'v4'), ('0.34.0rc1.dev815+g8f3c672dc', 'v5'),
                     ('0.34.0rc1.dev815+g8f3c672dc.cu132', 'v5'), ('0.34.1', None), ('0.35.0', None), ('dev', None))
            for version, impl in cases:
                with self.subTest(version=version):
                    called = []
                    pkg = types.ModuleType('synth_eot_checkpoint'); pkg.register = lambda: called.append('v4')
                    pkg.vllm034 = types.SimpleNamespace(register=lambda: called.append('v5'))
                    with mock.patch.dict(sys.modules, {'vllm': types.SimpleNamespace(__version__=version),
                                                       'synth_eot_checkpoint': pkg,
                                                       'synth_eot_checkpoint.vllm034': pkg.vllm034}), \
                            mock.patch.dict('os.environ', {'SYNTH_EOT_CHECKPOINT': '1'}):
                        entry.register()
                    self.assertEqual(called, [impl] if impl else [])
        finally:
            sys.path.remove(str(EOT))
            for name in [m for m in sys.modules if m.startswith('synth_eot_checkpoint')]:
                del sys.modules[name]


# plugin builds that run only on vLLM < 0.34 (supersede 3.0 / eot_checkpoint 5.0 are the first with a 0.34 path)
OLD_PLUGINS = re.compile(r'(cpu_supersede_v[12]|eot_checkpoint_v[1-4][a-z]*|evict_dead_v1)(?=/|:|$)')


def pythonpath(env):
    return (env or {}).get('PYTHONPATH', '')


class PluginPairing(unittest.TestCase):
    """No spec, profile or campaign pairs a vLLM 0.34 venv with a plugin build written for 0.32 (eot_checkpoint v4
    patches 0.32 internals unguarded; supersede v1/v2 would demote nothing on 0.34)."""

    def check(self, where, venv, env_path):
        if '0.34' in (venv or ''):
            self.assertFalse(OLD_PLUGINS.search(env_path), f'{where}: vLLM 0.34 venv with {env_path}')

    def test_specs_and_profiles(self):
        for spec in sorted((REPO / 'synth' / 'teachers').glob('*.toml')):
            t = load_teacher(spec.stem)
            self.check(spec.name, t['runtime']['venv'], pythonpath(t['serving'].get('env')))
            profiles = [PROFILES.get(spec.stem, {})] + [ps.get(spec.stem, {}) for ps in PIPELINE_PROFILES.values()]
            for prof in profiles:
                tun = prof.get('tunables', {})
                self.check(f'{spec.stem} profile', tun.get('venv') or t['runtime']['venv'],
                           pythonpath(tun.get('extra_env')) or pythonpath(t['serving'].get('env')))

    def test_campaigns(self):
        for path in sorted((REPO / 'synth' / 'campaigns').glob('*.toml')):
            text = re.sub(r'(?m)^@[A-Z_]+@\s*$', '', path.read_text())   # templates: whole-line placeholders
            c = tomllib.loads(re.sub(r'@[A-Z_]+@', '0', text))                  # and value placeholders
            if 'teacher' not in c:
                continue
            t = load_teacher(c['teacher'])
            base = c.get('teacher_overrides', {}).get('serving', {}).get('tunables', {})
            arms = [base] + [{**base, **cand.get('tunables', {})} for cand in c.get('tuning', {}).get('candidates', [])]
            for tun in arms:
                self.check(path.name, tun.get('venv') or t['runtime']['venv'],
                           pythonpath(tun.get('extra_env')) or pythonpath(t['serving'].get('env')))


if __name__ == '__main__':
    unittest.main()

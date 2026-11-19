"""Neutral compatibility-policy tests: no model, task solution, sandbox or job."""
import copy
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import task_environment as policy
from .worker import PiEvalWorker, q35
from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment


def config():
    value = policy.new_policy('tb21', 'pi', False, ['pypi-server', 'headless-terminal'])
    return {'benchmark': 'tb21', 'agent': 'pi', 'agent_internet': False,
            'task_environment_policy': value, 'task_environment_policy_sha256': policy.digest(value),
            'task_names': ['pypi-server', 'headless-terminal'],
            'protocol': {'task_environment_policy': copy.deepcopy(value),
                         'task_environment_policy_sha256': policy.digest(value)}}


class TaskEnvironmentTests(unittest.TestCase):
    def test_new_policy_is_only_offline_tb_pi_and_legacy_config_is_not_upgraded(self):
        cfg = config()
        self.assertEqual(policy.overrides_for(cfg, 'pypi-server'), {policy.FLAG: '1'})
        self.assertEqual(policy.overrides_for(cfg, 'headless-terminal'), {})
        for bench, agent, internet in [('tbsci_lite', 'pi', False), ('tb21', 'pi', True), ('tb21', 'oracle', False)]:
            self.assertIsNone(policy.new_policy(bench, agent, internet, ['pypi-server']))
        self.assertEqual(policy.new_policy('tb21', 'pi', False, ['headless-terminal']), {'schema_version': 1, 'overrides': {}})
        legacy = {'benchmark': 'tb21', 'agent': 'pi', 'agent_internet': False, 'protocol': {}}
        original = copy.deepcopy(legacy)
        self.assertEqual(policy.overrides_for(legacy, 'pypi-server'), {})
        self.assertEqual(legacy, original)

    def test_forged_or_inconsistent_policy_never_dispatches_extra_task_environment(self):
        for changed in ('digest', 'protocol', 'protocol-version-type', 'other-task', 'other-flag', 'version', 'online', 'selection'):
            with self.subTest(changed=changed):
                cfg = config()
                if changed == 'digest':cfg['task_environment_policy_sha256'] = '0' * 64
                elif changed == 'protocol':cfg['protocol']['task_environment_policy'] = {}
                elif changed == 'protocol-version-type':cfg['protocol']['task_environment_policy']['schema_version'] = True
                elif changed == 'other-task':cfg['task_environment_policy']['overrides']['headless-terminal'] = {policy.FLAG: '1'}
                elif changed == 'other-flag':cfg['task_environment_policy']['overrides']['pypi-server']['HTTP_PROXY'] = 'http://invalid'
                elif changed == 'version':cfg['task_environment_policy']['schema_version'] = True
                elif changed == 'selection':cfg['task_names'] = 'pypi-server'
                else:cfg['agent_internet'] = True
                if changed in ('other-task', 'other-flag', 'version'):
                    cfg['task_environment_policy_sha256'] = policy.digest(cfg['task_environment_policy'])
                    cfg['protocol'].update(task_environment_policy=copy.deepcopy(cfg['task_environment_policy']),
                        task_environment_policy_sha256=cfg['task_environment_policy_sha256'])
                with self.assertRaises(ValueError):policy.overrides_for(cfg, 'pypi-server')
        for cfg in [{'task_environment_policy': None}, {'task_environment_policy_sha256': '0' * 64},
                    {'task_environment_policy': None, 'task_environment_policy_sha256': None,
                     'protocol': {'task_environment_policy': None, 'task_environment_policy_sha256': None}}]:
            with self.subTest(presence=cfg):
                with self.assertRaises(ValueError):policy.overrides_for(cfg, 'pypi-server')
        cfg = config()
        cfg['protocol'].pop('task_environment_policy_sha256')
        with self.assertRaises(ValueError):policy.overrides_for(cfg, 'pypi-server')
        with self.assertRaisesRegex(ValueError, 'selected task list'):
            policy.overrides_for(config(), 'not-a-selected-task')

    def test_worker_dispatch_is_fresh_per_attempt_and_does_not_leak_to_other_or_legacy_tasks(self):
        worker = PiEvalWorker.__new__(PiEvalWorker)
        worker.config = config()
        inherited = {'HARBOR_EXTRA_ARGS': '', policy.FLAG: '1'}
        with mock.patch.object(q35.Worker, 'env', side_effect=lambda *args: dict(inherited)):
            fixed = worker.env('pypi-server__a0__r0', 'pypi-server', 0)
            other = worker.env('headless-terminal__a0__r0', 'headless-terminal', 0)
            self.assertEqual(fixed[policy.FLAG], '1')
            self.assertNotIn(policy.FLAG, other)
            self.assertEqual(inherited[policy.FLAG], '1')
            worker.config = {'benchmark': 'tb21', 'agent': 'pi', 'agent_internet': False}
            self.assertNotIn(policy.FLAG, worker.env('pypi-server__a0__r0', 'pypi-server', 0))

    def test_provider_opt_out_preserves_original_configs_and_default_env_i_support(self):
        import aa_evals.common.harbor_mswea.ns_env as module
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'etc/pip.conf'
            target.parent.mkdir()
            env = NsSidecarEnvironment.__new__(NsSidecarEnvironment)
            env._services = {'main': SimpleNamespace(box=SimpleNamespace(init_pid=123,
                base_env={'https_proxy': 'http://proxy.internal:8080'}))}
            with mock.patch.object(module, 'resolve_in_root', return_value=target):
                with mock.patch.dict(module.os.environ, {policy.FLAG: '1'}):
                    env._seed_pip_proxy()
                    self.assertFalse(target.exists())
                    target.write_text('[global]\nindex-url = https://existing.example.org/simple\n')
                    before = target.read_bytes()
                    env._seed_pip_proxy()
                    self.assertEqual(target.read_bytes(), before)
                with mock.patch.dict(module.os.environ, {}, clear=True):
                    env._seed_pip_proxy()
                    self.assertIn('proxy = http://proxy.internal:8080', target.read_text())
                    self.assertIn('index-url = https://existing.example.org/simple', target.read_text())
                    target.write_text('[global]\nproxy = http://existing.example.org:8080\n')
                    before = target.read_bytes()
                    env._seed_pip_proxy()
                    self.assertEqual(target.read_bytes(), before)

    def test_native_attempt_records_effective_policy_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = PiEvalWorker.__new__(PiEvalWorker)
            worker.config = {**config(), 'model': 'fixture', 'served': 'fixture'}
            worker.name = 'fake-no-generation'
            worker.run, worker.tasks_dir, worker.urls = root / 'run', root / 'tasks', ['http://fake.invalid/v1']
            with mock.patch.object(q35, 'SCRATCH', root / 'scratch'), \
                 mock.patch.object(q35.Worker, 'env', return_value={'HARBOR_EXTRA_ARGS': ''}), \
                 mock.patch('asyncio.create_subprocess_exec', new=mock.AsyncMock(side_effect=RuntimeError('pre-Gen interception'))):
                with self.assertRaisesRegex(RuntimeError, 'pre-Gen interception'):
                    asyncio.run(worker.run_item('pypi-server__a0__r0'))
            info = json.loads((worker.run / 'attempts/pypi-server__a0__r0/attempt.json').read_text())
            self.assertEqual(info['task_environment_policy_sha256'], worker.config['task_environment_policy_sha256'])
            self.assertEqual(info['task_environment_overrides'], {policy.FLAG: '1'})
            self.assertEqual(info['environment'][policy.FLAG], '1')

    def test_native_resume_checks_policy_before_submission_and_never_upgrades_legacy(self):
        from . import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'tb21/fake-resume'
            target.mkdir(parents=True)
            path = target / 'config.json'
            args = ['--bench', 'tb21', '--run-id', 'fake-resume', '--resubmit', '1',
                    '--runtime-cache-manifest', '', '--runtime-cache-manifest-sha256', '']
            cfg = {**config(), 'worker': {'n': 1}}
            with mock.patch.object(run, 'RUNS_ROOT', root), mock.patch.object(run, 'status', return_value={}), \
                 mock.patch.object(run, 'submit_workers') as submit:
                path.write_text(json.dumps(cfg))
                original = path.read_bytes()
                run.main(args)
                submit.assert_called_once()
                self.assertEqual(path.read_bytes(), original)
                submit.reset_mock()
                cfg['task_environment_policy_sha256'] = '0' * 64
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):run.main(args)
                submit.assert_not_called()
                legacy = {k: v for k, v in cfg.items() if not k.startswith('task_environment_')}
                legacy['protocol'] = {}
                path.write_text(json.dumps(legacy))
                original = path.read_bytes()
                run.main(args)
                submit.assert_called_once()
                self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

"""Per-job serving control (one A/B arm per serving job) and per-arm engine env."""
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

from synth.common import atomic, harbor_result
from synth.gpu import preflight
from synth.serve import Supervisor, kv_cache_geometry, kv_offload_tokens
from synth.vllm_launch import replica_env


def supervisor(root):
    sup = object.__new__(Supervisor)
    ops = Path(root) / 'ops'
    sup.campaign = types.SimpleNamespace(ops=ops)
    sup.teacher = {'serving': {'tunables': {'mtp': 1, 'gpu_memory_utilization': 0.89}}}
    sup.home = ops / 'serving' / 'job-3'
    sup.home.mkdir(parents=True)
    return sup


class ReadControl(unittest.TestCase):
    def test_defaults_campaign_file_then_job_file(self):
        with tempfile.TemporaryDirectory() as d:
            sup = supervisor(d)
            self.assertEqual(sup.read_control(), {'generation': 0, 'mtp': 1, 'gpu_memory_utilization': 0.89})
            atomic(sup.campaign.ops / 'serving' / 'control.json', {'generation': 2, 'mtp': 1, 'gpu_memory_utilization': 0.89})
            atomic(sup.home / 'control.json', {'generation': 9, 'arm': 'mtp3', 'mtp': 3})
            control = sup.read_control()
            self.assertEqual(control['generation'], 2)  # the controller only admits at the campaign-wide generation
            self.assertEqual((control['arm'], control['mtp'], control['gpu_memory_utilization']), ('mtp3', 3, 0.89))


class ArmEnv(unittest.TestCase):
    def test_extra_env_reaches_the_engine(self):
        teacher = {'runtime': {'venv': '/venv'}, 'serving': {}}
        with tempfile.TemporaryDirectory() as d:
            env = replica_env(teacher, gpus=['0'], scratch=d, base_env={'PYTHONPATH': '/repo'},
                              tunables={'extra_env': {'PYTHONPATH': '/plugins/v1', 'SYNTH_EVICT_DEAD': 1}})
            self.assertEqual((env['PYTHONPATH'], env['SYNTH_EVICT_DEAD']), ('/plugins/v1', '1'))
            env = replica_env(teacher, gpus=['0'], scratch=d, base_env={'PYTHONPATH': '/repo'}, tunables={})
            self.assertNotIn('PYTHONPATH', env)


class OffloadGeometry(unittest.TestCase):
    def test_cpu_tier_tokens_from_the_engine_log(self):
        log = ('INFO [interface.py:1105] Setting attention block size to 4256 tokens to ensure that attention page size '
               '>= mamba page size.\nINFO [kv_cache_utils.py:1067] GPU KV cache size: 11,425,615 tokens, Maximum ...\n'
               '(EngineCore pid=1) INFO [worker.py:213] SimpleCPUOffloadWorker [CPU]: 9 tensors, 10950 CPU blocks (200.00 GB)\n')
        tokens, block = kv_cache_geometry(log)
        self.assertEqual((tokens, block, kv_offload_tokens(log, block)), (11425615, 4256, 10950 * 4256))
        self.assertEqual(kv_offload_tokens(log.splitlines()[0], block), 0)


class Preflight(unittest.TestCase):
    def test_a_gpu_held_by_another_process_fails(self):
        cuda = '{"calls": [{"function": "cuInit", "code": 0, "name": "CUDA_SUCCESS"}], "device_count": 1}'
        for used, passed in (('3 MiB', True), ('43711 MiB', False)):
            smi = f'0, GPU-e3d6, NVIDIA B200, 183359 MiB, 580.178.04, {used}\n'
            runs = [types.SimpleNamespace(returncode=0, stdout=smi, stderr=''),
                    types.SimpleNamespace(returncode=0, stdout=cuda, stderr='')]
            with mock.patch('synth.gpu.subprocess.run', side_effect=runs):
                ok, report = preflight(['GPU-e3d6'])
            self.assertEqual(ok, passed)
            self.assertEqual(report.get('busy'), None if passed else {'GPU-e3d6': 43711})


class HarborResult(unittest.TestCase):
    def test_a_result_cut_mid_write_reads_as_missing(self):
        with tempfile.TemporaryDirectory() as d:
            trial = Path(d) / 'harbor' / 'task_1__abc'
            trial.mkdir(parents=True)
            self.assertIsNone(harbor_result(d))
            (trial / 'result.json').write_text('{"verifier_result": {"rewards": {"reward": 1.0}}}')
            self.assertEqual(harbor_result(d)['verifier_result']['rewards']['reward'], 1.0)
            (trial / 'result.json').write_text('{"verifier_result": {"rewards": {"rew')
            self.assertIsNone(harbor_result(d))


if __name__ == '__main__':
    unittest.main()


def test_dp_groups_split_a_job_into_dp_engines():
    import pytest
    from synth.serve import dp_groups
    gpus = ['a', 'b', 'c', 'd']
    assert dp_groups(gpus, {}) == [gpus]
    assert dp_groups(gpus, {'dp_group_gpus': 2}) == [['a', 'b'], ['c', 'd']]
    with pytest.raises(ValueError):
        dp_groups(gpus, {'dp_group_gpus': 3})


def test_campaign_serving_and_tuning_jobs_request_the_specs_gpu_type():
    """#445: a campaign's serving jobs request the GPU type of its serving_resources.cuda_capability (H100 = 9.0), as
    synth.tune's jobs do (aa-serving #71); without it, 10.0 (B200) as before."""
    from synth import condor
    from synth.config import Campaign
    for cc, want in (('9.0', '(TARGET.CUDACapability == 9.0)'), (None, '(TARGET.CUDACapability == 10.0)')):
        with tempfile.TemporaryDirectory() as d:
            serving = {'jobs': 1, 'gpus_per_job': 1, 'cpus_per_gpu': 8, 'memory_gib_per_gpu': 64, 'disk_gib': 100,
                       'bid': 61, **({'cuda_capability': cc} if cc else {})}
            campaign = Campaign({'id': 'cc', 'root': d, 'teacher': {'slug': 't'}, 'serving': serving, 'protocol': {}})
            run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.', returncode=0))
            with mock.patch.object(condor.subprocess, 'run', run), mock.patch.object(condor, 'update_ledger', lambda c: None):
                condor.submit_serving(campaign, 0, 0)
            sub = (campaign.ops / 'jobs' / 'serving-0-g0.sub').read_text()
            assert want in sub and 'CUDACapability == 1' not in sub.replace(want, '')

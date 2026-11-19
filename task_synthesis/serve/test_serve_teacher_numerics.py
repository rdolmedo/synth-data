"""Every serve_teacher job records its effective numerics (synth/numerics.py): numerics.<job>.json, launch.json and
endpoints.json carry the weights' quantization and one numerics_sha256, after the submit-time --tunables."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from synth.config import load_teacher
from task_synthesis.serve import serve_teacher as st


def fake_run(d, slug, tunables='', edit=None, launched=None, gpus=('GPU-0',)):
    """serve() of `slug` with fake GPUs and processes -> {file name: last value written}. edit(teacher, d): change the
    spec first; launched: a list that gets every Popen's (argv, env)."""
    d = Path(d)
    ckpt = d / 'ckpt'
    ckpt.mkdir()
    (ckpt / 'config.json').write_text(json.dumps({'torch_dtype': 'bfloat16'}))
    (ckpt / 'chat_template.jinja').write_text('{{ messages }}')
    teacher = {**load_teacher(slug), 'checkpoint': str(ckpt)}
    if edit is not None:
        edit(teacher, d)
    writes = {}
    real_atomic = st.atomic

    def atomic(path, value):
        writes[Path(path).name] = value
        real_atomic(path, value)

    def popen(*args, **kwargs):   # vLLM and the gate: up for one loop, then exit
        if launched is not None:
            launched.append((args[0], kwargs.get('env')))
        proc = mock.Mock(returncode=0)
        proc.poll.side_effect = [None, 0, 0, 0]
        return proc

    report = {'nvidia_smi': {'stdout': '0, GPU-0, NVIDIA H100 80GB HBM3\n'}}
    a = mock.Mock(teacher=slug, name='t', tunables=tunables, protocol='', checkpoint='', pipeline='aa-serving')
    a.name = 't'
    patches = [mock.patch.object(st, 'ROOT', d / 'serving'), mock.patch.object(st, 'load_teacher', return_value=teacher),
               mock.patch.object(st, 'assigned', return_value='GPU-0'), mock.patch.object(st, 'expand', return_value=list(gpus)),
               mock.patch.object(st, 'preflight', return_value=(True, report)),
               mock.patch.object(st.engine_cache, 'seed', return_value={'seeded': False}),
               mock.patch.object(st.engine_cache, 'publish', return_value={}),
               mock.patch.object(st.subprocess, 'Popen', side_effect=popen), mock.patch.object(st, 'request'),
               mock.patch.object(st, 'probe'), mock.patch.object(st, 'stop_process'),
               mock.patch.object(st, 'atomic', atomic), mock.patch.object(st.time, 'sleep'),
               mock.patch.dict(os.environ, {'_CONDOR_SCRATCH_DIR': str(d / 'scratch'), 'SERVE_JOB_ID': '42.0'})]
    for p in patches:
        p.start()
    try:
        home = d / 'serving' / 't'
        home.mkdir(parents=True)
        with contextlib.suppress(SystemExit):   # serve() exits when its engines do
            st.serve(a, home)
    finally:
        for p in reversed(patches):
            p.stop()
    return writes


class JobRecordsNumerics(unittest.TestCase):
    def test_launch_numerics_and_endpoints_agree(self):
        with tempfile.TemporaryDirectory() as d:
            writes = fake_run(d, 'nemotron35-lightning-eval')
            rec, launch, ep = writes['numerics.42.0.json'], writes['launch.json'], writes['endpoints.json']
            self.assertTrue(rec['numerics_sha256'])
            self.assertEqual(launch['numerics_sha256'], rec['numerics_sha256'])
            self.assertEqual(ep['numerics_sha256'], rec['numerics_sha256'])
            self.assertEqual(ep['weights_quantization'], 'fp8 (online, from bfloat16)')
            self.assertEqual(ep['served_name'], 'nemotron35-lightning')
            replica = rec['replicas'][0]
            self.assertIn('--quantization', replica['argv'])
            self.assertEqual(replica['gpu'], ['NVIDIA H100 80GB HBM3'])
            self.assertEqual(replica['template']['source'], 'checkpoint')
            self.assertNotIn(d, json.dumps(replica['argv']) + json.dumps(replica['env']))
            self.assertEqual(launch['replicas'][0]['numerics']['sha256'], replica['sha256'])
            # the first engine's command at the top level too: Endpoint._dp_size (task_synthesis/pipeline/llm.py)
            self.assertEqual(launch['argv'], launch['replicas'][0]['argv'])
            self.assertEqual(launch['pipeline'], 'aa-serving')

    def test_dp_internal_launch_pins_m1_sessions(self):
        """M1's client reads launch.json['argv'] for the DP size behind a dp_internal server's one URL (review of #480
        PR A: with only replicas[].argv it sent no X-data-parallel-rank)."""
        from task_synthesis.pipeline.llm import Endpoint
        with tempfile.TemporaryDirectory() as d:
            def four_gpus(teacher, d):
                teacher['stage_checkpoint'] = False
            writes = fake_run(d, 'deepseek-v41-flash-1017', edit=four_gpus, gpus=('GPU-0', 'GPU-1', 'GPU-2', 'GPU-3'))
            home = Path(d) / 'serving' / 't'
            self.assertEqual(Endpoint._dp_size(home / 'endpoints.json', {'urls': ['u']}), 4)
            self.assertIn('--data-parallel-size', writes['launch.json']['argv'])

    def test_serve_stages_extra_dirs_and_kernel_wheels(self):
        """#445: serve() stages a spec's stage_extra dirs and kernel wheels and passes them as synth.serve does
        (nemotron35-bolt's {stage_draft})."""
        import hashlib
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            def extras(teacher, d):
                draft = d / 'draft'
                draft.mkdir()
                (draft / 'config.json').write_text('{}')
                wheel = d / 'k-0.1-py3-none-any.whl'
                with zipfile.ZipFile(wheel, 'w') as z:
                    z.writestr('kmod/__init__.py', '')
                teacher['serving']['stage_extra'] = [{'name': 'draft', 'source': str(draft)}]
                teacher['serving']['kernel_wheels'] = [{'path': str(wheel),
                                                        'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}]
                teacher['serving']['args'] = teacher['serving']['args'] + ['--draft-dir', '{stage_draft}']
            launched = []
            writes = fake_run(d, 'nemotron35-lightning-eval', edit=extras, launched=launched)
            argv, env = launched[0]
            staged = writes['launch.json']['staged']['stage_draft']
            self.assertEqual(argv[argv.index('--draft-dir') + 1], staged)
            self.assertTrue((Path(staged) / 'config.json').is_file())
            provider = env['PYTHONPATH'].split(':')[0]
            self.assertTrue(provider.endswith('kernel-provider') and (Path(provider) / 'kmod').is_dir())

    def test_a_submit_override_shows_in_the_record(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            fp8 = fake_run(a, 'nemotron35-lightning-eval')['endpoints.json']
            over = '{"quantization": "", "h100_attention": 0}'
            bf16 = fake_run(b, 'nemotron35-lightning-eval', over)
            self.assertEqual(bf16['endpoints.json']['weights_quantization'], 'bfloat16')
            self.assertNotEqual(bf16['endpoints.json']['numerics_sha256'], fp8['numerics_sha256'])
            self.assertEqual(bf16['numerics.42.0.json']['replicas'][0]['overrides']['tunables'], over)


if __name__ == '__main__':
    unittest.main()

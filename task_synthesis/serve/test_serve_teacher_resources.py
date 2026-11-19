"""Serving-job resources (#378 audit item 7): the cgroup sampler (tbshared/reslog.py since #445 item 2; its
field tests are in tbshared/test_reslog.py) as serve_teacher runs it, and the memory-hold safety net."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from task_synthesis.serve import serve_teacher as st

class SamplerLifetime(unittest.TestCase):
    def test_exits_with_its_parent(self):
        """#401: the sampler must never outlive the process that started it."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'res.jsonl'
            pidfile = Path(d) / 'pid'
            parent = (f'import subprocess, sys; p = subprocess.Popen([sys.executable, "-I", {str(st.RESLOG)!r}, "run", '
                      f'{str(out)!r}, "1"]); open({str(pidfile)!r}, "w").write(str(p.pid)); import time; time.sleep(2)')
            subprocess.run([sys.executable, '-c', parent], check=True, timeout=30)
            pid = int(pidfile.read_text())
            deadline = time.time() + 5
            while time.time() < deadline and Path(f'/proc/{pid}').exists() and \
                    'Z' not in (Path(f'/proc/{pid}/stat').read_text().split()[2:3] or ['']):
                time.sleep(0.2)
            alive = Path(f'/proc/{pid}').exists() and Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
            self.assertFalse(alive, 'sampler outlived its parent')
            rows = [json.loads(ln) for ln in out.read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn('cgroup', rows[0])

    def test_exits_at_once_if_its_parent_is_already_gone(self):
        """--parent names the expected parent: a sampler whose parent died before it started must not run on."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'res.jsonl'
            r = subprocess.run([sys.executable, '-I', str(st.RESLOG), 'run', str(out), '1', '--parent', '1'],
                               timeout=10)
            self.assertEqual(r.returncode, 0)
            self.assertFalse(out.exists())

    def test_run_stops_the_sampler_on_any_exit(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(st, 'ROOT', Path(d)), \
                mock.patch.object(st, 'serve', side_effect=SystemExit(3)):
            started = []
            real = st.start_sampler
            st_start = lambda home: started.append(real(home)) or started[-1]
            a = mock.Mock()
            a.name = 'x'
            with mock.patch.object(st, 'start_sampler', st_start), self.assertRaises(SystemExit):
                st.run(a)
            self.assertIsNotNone(started[0].poll(), 'sampler still running after run() exited')


class SafetyNet(unittest.TestCase):
    def submit(self, teacher, gpus, pipeline='aa-serving', fill=None, job=None, fills=None):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(st, 'ROOT', Path(d)), \
                mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')), \
                mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n')):
            a = mock.Mock(teacher=teacher, gpus=gpus, force=False, tunables='', protocol='', cuda_cap='', hours=1, bid=61,
                          cpus=0, memory_gib=0, checkpoint='', dry_run=False, exclude_nodes='', bypass_lxcfs=False,
                          pipeline=pipeline, fill=fill)
            a.name = 'n'
            st.submit(a)
            if job is not None:
                job.update(json.loads((Path(d) / 'n' / 'job.json').read_text()))
            if fills is not None:
                f = Path(d) / 'n' / 'fill.jsonl'
                fills[:] = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
            return (Path(d) / 'n' / 'job.sub').read_text()

    def test_memory_request_grows_per_memory_hold_and_releases(self):
        sub = self.submit('qwen38-27b-eval-fast', 1)
        req = next(ln for ln in sub.splitlines() if ln.startswith('request_memory'))
        base = int(st.memory_gib(st.load_teacher('qwen38-27b-eval-fast'), 'qwen38-27b-eval-fast', 1) * 1024)
        self.assertEqual(req, f'request_memory = {base} + {int(base * st.MEMORY_STEP)} * {st.MEMORY_HOLDS}')
        self.assertIn(f'periodic_release = {st.MEMORY_RELEASE}\n', sub)
        self.assertIn('NumHoldsByReason.JobOutOfResources', st.MEMORY_RELEASE)
        self.assertIn('HoldReasonCode =?= 34', st.MEMORY_RELEASE)



class Shapes(unittest.TestCase):
    """#378 measured shapes: (CPUs, GiB) a submit requests per spec and GPU count (report: /data/aai/aa_evals/
    audits/server_shapes_20261114/report.md). A change here is a change to every new server's bill."""
    EXPECTED = {
        ('qwen38-27b-eval-fast', 1): (15, 240), ('qwen38-27b-eval-fast-text', 2): (26, 420),
        ('qwen36-35b-a3b-eval', 1): (15, 240), ('qwen36-35b-a3b-eval-text', 1): (13, 210),
        ('nemotron35-lightning-eval', 1): (10, 168),
        ('gemma4-26b-a4b-it-eval-b200', 1): (6, 64), ('gemma4-26b-a4b-it-eval-b200-text', 1): (6, 64),
        ('qwen35-4b-eval', 1): (6, 64), ('qwen35-4b-eval-text', 1): (6, 64),
        ('nemotron35-super-mm', 1): (18, 288), ('nemotron35-super-mm-text', 1): (16, 256),   # 200 GiB CPU KV tier
        ('glm53-flash-eval', 4): (28, 448), ('glm53-flash-eval-text', 4): (24, 384),
        ('deepseek-v41-flash-1017', 4): (40, 640),
        ('snowball-eval', 1): (4, 64), ('snowball-eval', 2): (8, 128),   # M5 runs --gpus 2; #531 trim (was 8 / 128 per GPU)
        ('snowball-eval-main', 1): (4, 64),                                # #478: snowball-eval's shape on vLLM main
        ('snowflake-eval', 1): (8, 128), ('datakit-eval', 2): (16, 256),   # not trimmed (their own measurements first)
    }

    def test_requests(self):
        for (slug, gpus), (cpus, gib) in self.EXPECTED.items():
            with self.subTest(slug=slug, gpus=gpus):
                sub = SafetyNet.submit(self, slug, gpus)
                self.assertIn(f'request_cpus = {cpus}\n', sub)
                self.assertIn(f'request_memory = {gib * 1024} + ', sub)
                self.assertIn(f'request_gpus = {gpus}\n', sub)

    def test_no_profile_overrides_the_spec_ram(self):
        """c30 and M5 ran DeepSeek at 1.5 TB because a serve_teacher PROFILE memory_gib overrode the spec's. Only a
        pipeline's own profile set may size a server (task synthesis' measured DeepSeek one-shot shape, #281)."""
        self.assertFalse([s for s, p in st.PROFILES.items() if 'memory_gib' in p])
        self.assertEqual({(pl, s) for pl, ps in st.PIPELINE_PROFILES.items() for s, p in ps.items() if 'memory_gib' in p},
                         {('task-synthesis', 'deepseek-v41-flash-1017')})


class Pipelines(unittest.TestCase):
    """--pipeline (#480): each pipeline keeps its branch's submit defaults and profiles."""

    def test_task_synthesis_deepseek_shape_without_net(self):
        job, fills = {}, []
        sub = SafetyNet.submit(self, 'deepseek-v41-flash-1017', 4, pipeline='task-synthesis', job=job, fills=fills)
        self.assertEqual(len(fills), 1, 'task synthesis submits through tbshared condor_cost.fill_plan')
        self.assertNotEqual(fills[0]['decision'], 'off')
        self.assertEqual(fills[0]['pipeline'], 'task-synthesis')   # --pipeline, not task_synthesis/serve's file
        self.assertIn('request_cpus = 20\n', sub)
        self.assertIn(f'request_memory = {320 * 1024}\n', sub, 'no memory net (holds.remove_held(has_net=False))')
        self.assertIn(f'+DevShmSize = {200 * 1024}\n', sub)   # room for the shared Engram tables (#559 review)
        self.assertNotIn('periodic_release', sub)
        self.assertIn("'--pipeline' 'task-synthesis'", sub)
        self.assertEqual((job['pipeline'], job['memory_net'], job['fill'], job['cpus'], job['memory_gib']),
                         ('task-synthesis', False, True, 20, 320))

    def test_aa_serving_default_is_unchanged(self):
        job, fills = {}, []
        sub = SafetyNet.submit(self, 'deepseek-v41-flash-1017', 4, job=job, fills=fills)
        self.assertEqual(fills, [], 'aa-serving: no fill step at all (the submit file as written)')
        self.assertIn('request_cpus = 40\n', sub)
        self.assertIn(f'periodic_release = {st.MEMORY_RELEASE}\n', sub)
        self.assertNotIn('--pipeline', sub)
        self.assertEqual((job['pipeline'], job['memory_net'], job['fill']), ('aa-serving', True, False))

    def test_task_synthesis_honours_memory_gib_per_gpu(self):
        """#445: task-synthesis' copy requested 256 GiB per GPU whatever the spec said."""
        for slug, gpus in (('qwen38-27b-eval-fast', 1), ('glm53-flash-eval', 4), ('snowball-eval', 2)):
            with self.subTest(slug=slug):
                want = st.memory_gib(st.load_teacher(slug), slug, gpus)
                self.assertNotEqual(want, 256 * gpus)
                sub = SafetyNet.submit(self, slug, gpus, pipeline='task-synthesis')
                self.assertIn(f'request_memory = {int(want * 1024)}\n', sub)

    def test_fill_flag_overrides_the_pipeline_default(self):
        job = {}
        SafetyNet.submit(self, 'qwen35-4b-eval', 1, fill=True, job=job)
        self.assertTrue(job['fill'])
        SafetyNet.submit(self, 'deepseek-v41-flash-1017', 4, pipeline='task-synthesis', fill=False, job=job)
        self.assertFalse(job['fill'])

    def test_unknown_pipeline_fails(self):
        with self.assertRaises(SystemExit):
            st.profile_for('deepseek-v41-flash-1017', 'eval')

    def test_bad_nodes_are_tbshared_list(self):
        self.assertEqual(st.bad_nodes(), sorted(st.condor_cost.bad_nodes()))   # one source, no launcher extras
        self.assertTrue({'cpu305', 'cpu201'} <= set(st.condor_cost.BAD_NODES))


class PromptSources(unittest.TestCase):
    def test_sums_every_engine(self):
        a = ('vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 100.0\n'
             'vllm:prompt_tokens_by_source_total{engine="0",source="local_cache_hit"} 50.0\n'
             'vllm:generation_tokens_total{engine="0"} 7.0\n')
        b = a.replace('100.0', '1.0').replace('7.0', '3.0')
        self.assertEqual(st.prompt_sources([a, b]), {'sources': {'local_compute': 101.0, 'local_cache_hit': 100.0},
                                                     'generation_tokens': 10.0})


class ShapeCheck(unittest.TestCase):
    """#451 workstream 4: a request over 1.5x the spec's CPUs or RAM per GPU is refused (exit 2, nothing written)
    unless --allow-shape; job.json records the check."""

    def submit(self, teacher, gpus, cpus=0, memory_gib=0, allow=False, pipeline='aa-serving', dry_run=False, tunables='',
               job=False):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(st, 'ROOT', Path(d)), \
                mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')) as frozen, \
                mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n')):
            a = mock.Mock(teacher=teacher, gpus=gpus, force=False, tunables=tunables, protocol='', cuda_cap='', hours=1, bid=61,
                          cpus=cpus, memory_gib=memory_gib, checkpoint='', dry_run=dry_run, exclude_nodes='',
                          bypass_lxcfs=False, pipeline=pipeline, fill=None, allow_shape=allow)
            a.name = 'n'
            try:
                st.submit(a)
            except SystemExit as e:
                return e.code, None, (Path(d) / 'n').exists() or frozen.called
            j = json.loads((Path(d) / 'n' / 'job.json').read_text())
            return 0, (j if job else j['shape_check']), True

    def test_oversized_overrides_are_refused_before_anything_is_written(self):
        """32 CPUs / 512 GiB for 2 GPUs of a 4c/64G-per-GPU spec: no registry dir, no frozen checkout, exit 2."""
        self.assertEqual(self.submit('snowball-eval', 2, cpus=32, memory_gib=512), (2, None, False))
        self.assertEqual(self.submit('snowball-eval', 2, cpus=32, memory_gib=512, dry_run=True), (2, None, False))

    def test_allow_shape_submits_and_records(self):
        code, check, _ = self.submit('snowball-eval', 2, cpus=32, memory_gib=512, allow=True)
        self.assertEqual(code, 0)
        self.assertEqual((check['ok'], check['allowed'], check['ratio']), (False, True, {'cpus': 4.0, 'memory_gib': 4.0}))
        self.assertEqual(check['requested_per_gpu'], {'cpus': 16.0, 'memory_gib': 256.0})

    def test_allow_shape_is_a_submit_flag(self):
        seen = []
        with mock.patch.object(st, 'submit', lambda a: seen.append(a.allow_shape)), \
                mock.patch('sys.argv', ['serve_teacher', 'submit', '--teacher', 'snowball-eval', '--name', 'n',
                                        '--allow-shape']):
            st.main()
        with mock.patch.object(st, 'submit', lambda a: seen.append(a.allow_shape)), \
                mock.patch('sys.argv', ['serve_teacher', 'submit', '--teacher', 'snowball-eval', '--name', 'n']):
            st.main()
        self.assertEqual(seen, [True, False])

    def test_specs_pass_at_their_size_and_per_gpu_specs_at_any_size(self):
        for f in sorted((st.TS_ROOT_CODE / 'synth' / 'teachers').glob('*.toml')):
            slug = f.stem
            teacher = st.load_teacher(slug)
            res = teacher.get('serving_resources', {})
            base = int(res.get('default_gpus', 4))
            whole_job = 'memory_gib' in res
            for pipeline in st.PIPELINES:
                whole = whole_job or 'memory_gib' in st.profile_for(slug, pipeline)
                for gpus in ((base, 2 * base) if whole else (1, 2, 4, 8)):
                    with self.subTest(slug=slug, pipeline=pipeline, gpus=gpus):
                        c = st.shape_check(teacher, slug, gpus, st.cpus(teacher, slug, gpus, pipeline),
                                           st.memory_gib(teacher, slug, gpus, pipeline), pipeline)
                        self.assertTrue(c['ok'], c)

    def test_whole_job_shapes_on_fewer_gpus_are_caught(self):
        """deepseek-v41-flash-1017 sizes 640 GiB for 4 GPUs (aa-serving) and task synthesis' profile 20c/320G for 4:
        the same request on 2 GPUs is 2x per GPU."""
        self.assertEqual(self.submit('deepseek-v41-flash-1017', 2)[0], 2)
        self.assertEqual(self.submit('deepseek-v41-flash-1017', 2, pipeline='task-synthesis')[0], 2)
        code, check, _ = self.submit('deepseek-v41-flash-1017', 4, memory_gib=900)     # 225 / 160 per GPU = 1.41
        self.assertEqual((code, check['ok']), (0, True))
        code, check, _ = self.submit('deepseek-v41-flash-1017', 4, pipeline='task-synthesis')
        self.assertEqual((code, check['expected_per_gpu']), (0, {'cpus': 5.0, 'memory_gib': 80.0}))


class SnowballTrim(unittest.TestCase):
    """#531 (rt-lean, coordinator-approved): snowball-eval serves at 4c/64G per GPU (measured 2.2-2.9 mean cores,
    17-20 GB hard per 2-GPU server), and keeps 4 cores per extra engine when engines_per_gpu > 1."""

    def shape(self, gpus, tunables='', **kw):
        code, chk, _ = ShapeCheck.submit(self, 'snowball-eval', gpus, tunables=tunables, job=True, **kw)
        return code, chk

    def test_one_engine_per_gpu(self):
        for gpus, want in ((1, (4, 64)), (2, (8, 128)), (4, (16, 256))):
            code, job = self.shape(gpus)
            self.assertEqual(code, 0)
            self.assertEqual((job['shape_check']['derived']['cpus'], job['shape_check']['derived']['memory_gib']), want)
            self.assertNotIn('extra_engine_cpus', job['shape_check'])
            self.assertTrue(job['shape_check']['ok'])

    def test_extra_engines_keep_their_cores(self):
        for k, per_gpu in ((2, 8), (3, 12)):
            code, job = self.shape(2, tunables=json.dumps({'engines_per_gpu': k, 'gpu_util': 0.3}))
            self.assertEqual(code, 0, k)
            chk = job['shape_check']
            self.assertEqual((chk['derived']['cpus'], chk['extra_engine_cpus']), (2 * per_gpu, 2 * 4 * (k - 1)), k)
            self.assertEqual(job['cpus'], 2 * per_gpu, k)                     # what the job requests (billed)
            self.assertEqual(chk['expected_per_gpu']['cpus'], per_gpu)       # the check expects them too
            self.assertTrue(chk['ok'], k)
            self.assertEqual(chk['derived']['memory_gib'], 128)               # RAM stays per GPU

    def test_explicit_old_shape_on_one_engine_is_refused(self):
        """16 CPUs for one engine per GPU on 2 GPUs is 2x the trimmed shape: refused unless --allow-shape (no live
        snowball-eval job passed explicit CPUs or RAM: 21 of 21 recorded job.json used the spec's)."""
        self.assertEqual(self.shape(2, cpus=16)[0], 2)
        self.assertEqual(self.shape(2, cpus=16, allow=True)[0], 0)

    def test_other_modes_have_no_extra_engines(self):
        """dp_internal / dp_external_lb plan one engine per DP group (k = 1): the key adds nothing (and never plans
        their ports with a placeholder that could not advance)."""
        for slug in ('glm53-flash-eval', 'deepseek-v41-flash-1017'):
            teacher = st.load_teacher(slug)
            teacher['serving_resources']['cpus_per_extra_engine'] = 4
            self.assertEqual(st.extra_engine_cpus(teacher, slug, 4, 'aa-serving'), 0, slug)

    def test_specs_without_the_key_are_unchanged(self):
        for slug in ('snowflake-eval', 'datakit-eval'):
            teacher = st.load_teacher(slug)
            self.assertEqual(st.extra_engine_cpus(teacher, slug, 2, 'aa-serving',
                                                  json.dumps({'engines_per_gpu': 2, 'gpu_util': 0.3})), 0)
            self.assertEqual(st.cpus(teacher, slug, 2, 'aa-serving'), 16)


class TierSizing(unittest.TestCase):
    """#451 ws4 (coordinator: size it): a --tunables / replica_tunables kv_offloading_size moves the job's RAM by the
    tier difference over its DP ranks; an explicit --memory-gib is checked against that."""

    def specs(self):
        for f in sorted((st.TS_ROOT_CODE / 'synth' / 'teachers').glob('*.toml')):
            teacher = st.load_teacher(f.stem)
            yield f.stem, teacher, int(teacher.get('serving_resources', {}).get('default_gpus', 4))

    def submit_super(self, **kw):
        return ShapeCheck.submit(self, 'nemotron35-super-mm-text', 2, job=True, **kw)

    def test_defaults_are_unchanged(self):
        for slug, teacher, gpus in self.specs():
            for pipeline in st.PIPELINES:
                with self.subTest(slug=slug, pipeline=pipeline):
                    gib, tier = st.derived_memory_gib(teacher, slug, gpus, pipeline)
                    self.assertEqual(gib, st.memory_gib(teacher, slug, gpus, pipeline))
                    self.assertEqual(tier['job'], tier['spec'])

    def test_every_spec_ram_holds_its_own_tier(self):
        """Sizing down to tier 0 leaves at least 32 GiB per GPU: every spec's RAM includes the tier it sets."""
        for slug, teacher, gpus in self.specs():
            for pipeline in st.PIPELINES:
                with self.subTest(slug=slug, pipeline=pipeline):
                    gib, _ = st.derived_memory_gib(teacher, slug, gpus, pipeline, '{"kv_offloading_size": 0}')
                    self.assertGreaterEqual(gib / gpus, 32)

    def test_super_with_a_48g_tier(self):
        """m5l-gdpval's relaunch: nemotron35-super-mm-text on 2 GPUs with a 48 GiB tier per engine instead of 200."""
        t = st.load_teacher('nemotron35-super-mm-text')
        self.assertEqual(st.derived_memory_gib(t, 'nemotron35-super-mm-text', 2, 'aa-serving', '{"kv_offloading_size": 48}'),
                         (512 - 2 * 152, {'spec': 400.0, 'spec_gpus': 2, 'job': 96.0}))
        code, j, _ = self.submit_super(tunables='{"kv_offloading_size": 48}')
        self.assertEqual((code, j['memory_gib'], j['shape_check']['derived']['memory_gib']), (0, 208.0, 208.0))
        self.assertEqual(j['shape_check']['requested'], {'cpus': 32, 'memory_gib': 208.0})
        self.assertEqual(j['shape_check']['ratio']['memory_gib'], 1.0)

    def test_explicit_ram_is_checked_against_the_tier_sized_shape(self):
        """512 GiB is the spec's shape at its 200 GiB tier, but 2.46x the 48 GiB tier's 104 GiB per GPU."""
        self.assertEqual(self.submit_super(tunables='{"kv_offloading_size": 48}', memory_gib=512)[0], 2)
        self.assertEqual(self.submit_super(memory_gib=512)[0], 0)
        self.assertEqual(self.submit_super(tunables='{"kv_offloading_size": 48}', memory_gib=128)[0], 0)

    def test_replica_tunables_count_per_engine(self):
        t = st.load_teacher('nemotron35-super-mm-text')
        tj = '{"replica_tunables": [{}, {"kv_offloading_size": 100}]}'
        self.assertEqual(st.kv_tier_gib(t, 'nemotron35-super-mm-text', 2, 'aa-serving', tj), 300.0)

    def test_dp_ranks_and_profile_tiers(self):
        """DeepSeek's 640 GiB is sized for its aa-serving profile's 4 x 64 GiB tier (none in task synthesis);
        GLM-5.3-Flash's tier is per DP rank."""
        ds = st.load_teacher('deepseek-v41-flash-1017')
        self.assertEqual(st.derived_memory_gib(ds, 'deepseek-v41-flash-1017', 4, 'aa-serving', '{"kv_offloading_size": 0}'),
                         (384.0, {'spec': 256.0, 'spec_gpus': 4, 'job': 0.0}))
        self.assertEqual(st.kv_tier_gib(ds, 'deepseek-v41-flash-1017', 4, 'task-synthesis'), 0.0)
        glm = st.load_teacher('glm53-flash-eval')
        self.assertEqual(st.kv_tier_gib(glm, 'glm53-flash-eval', 4, 'aa-serving', '{"kv_offloading_size": 32}'), 128.0)

    def test_whole_job_ram_counts_the_spec_tier_on_its_own_gpus(self):
        """DeepSeek's 640 GiB holds 4 ranks x 64 GiB: on 8 GPUs (8 ranks) the tier adds 256, with the tier off it is
        640 - 256 on any count (review: it was 128 on 8 GPUs, below the 191 GiB of Engram)."""
        ds = st.load_teacher('deepseek-v41-flash-1017')
        slug = 'deepseek-v41-flash-1017'
        self.assertEqual(st.derived_memory_gib(ds, slug, 8, 'aa-serving')[0], 896.0)
        self.assertEqual(st.derived_memory_gib(ds, slug, 8, 'aa-serving', '{"kv_offloading_size": 0}')[0], 384.0)
        _, tier = st.derived_memory_gib(ds, slug, 8, 'aa-serving')
        c = st.shape_check(ds, slug, 8, st.cpus(ds, slug, 8, 'aa-serving'), 896.0, 'aa-serving', tier)
        self.assertEqual((c['ok'], c['expected_per_gpu']['memory_gib']), (True, 160.0))

    def test_engines_per_gpu_and_dp_replicas(self):
        """k engines per GPU each pin a full tier; replica_tunables index engines (a DP engine's ranks share it)."""
        q = st.load_teacher('qwen38-27b-eval-fast')
        self.assertEqual(st.kv_tier_gib(q, 'qwen38-27b-eval-fast', 2, 'aa-serving', '{"engines_per_gpu": 2, "gpu_util": 0.9}'), 640.0)
        glm = st.load_teacher('glm53-flash-eval')
        self.assertEqual(st.kv_tier_gib(glm, 'glm53-flash-eval', 4, 'aa-serving',
                                        '{"replica_tunables": [{}, {"kv_offloading_size": 32}]}'), 192.0)

    def test_tiers_match_the_engine_commands(self):
        """kv_tier_gib against the --kv-offloading-size of every engine command serve() would start (x its DP size)."""
        from synth.vllm_launch import replica_command
        cases = [('nemotron35-super-mm-text', 2, 'aa-serving', '{"kv_offloading_size": 48}'),
                 ('nemotron35-super-mm-text', 2, 'aa-serving', '{"replica_tunables": [{}, {"kv_offloading_size": 0}]}'),
                 ('qwen38-27b-eval-fast', 2, 'aa-serving', '{"engines_per_gpu": 2, "gpu_util": 0.9}'),
                 ('glm53-flash-eval', 4, 'aa-serving', '{"replica_tunables": [{"kv_offloading_size": 16}]}'),
                 ('deepseek-v41-flash-1017', 4, 'aa-serving', ''),
                 ('deepseek-v41-flash-1017', 4, 'task-synthesis', '{"kv_offloading_size": 32}')]
        for slug, gpus, pipeline, tj in cases:
            with self.subTest(slug=slug, tunables=tj):
                t = st.load_teacher(slug)
                n = iter(range(20000, 30000, 10))
                plan = st.engine_plan(slug, t, [str(i) for i in range(gpus)], pipeline=pipeline, tunables_json=tj,
                                      ports=lambda k: next(n))
                total = 0.0
                for i, extra in enumerate(plan['extras']):   # as serve()'s engine loop
                    tun = {**plan['tunables'], **(plan['per_replica'][i] if i < len(plan['per_replica']) else {})}
                    argv = replica_command(t, plan['protocol'], port=plan['bases'][i], model_dir='/m', tunables=tun,
                                           rank_ports=extra)
                    if '--kv-offloading-size' in argv:
                        total += float(argv[argv.index('--kv-offloading-size') + 1]) * extra.get('dp_size', 1)
                self.assertEqual(st.kv_tier_gib(t, slug, gpus, pipeline, tj), total)

    def test_unplannable_jobs_fail_at_submit_readably(self):
        for tj in ('{bad', '[1]', '{"engines_per_gpu": 2}'):   # Super has no gpu_util tunable to split
            with self.subTest(tunables=tj):
                code, _, written = self.submit_super(tunables=tj)
                self.assertIn('cannot plan nemotron35-super-mm-text on 2 GPU(s)', str(code))
                self.assertFalse(written)

    def test_a_spec_without_the_tier_arg_ignores_the_tunable(self):
        t = st.load_teacher('snowball-eval')
        self.assertEqual(st.derived_memory_gib(t, 'snowball-eval', 1, 'aa-serving', '{"kv_offloading_size": 64}'),
                         (st.memory_gib(t, 'snowball-eval', 1, 'aa-serving'), {'spec': 0.0, 'spec_gpus': 1, 'job': 0.0}))


if __name__ == '__main__':
    unittest.main()

"""task_synthesis.serve.serve_teacher --pipeline task-synthesis: its DeepSeek profile reaches the vLLM command (#163).
(Was task_synthesis/tests/test_serve_teacher.py on task-synthesis, where this profile was the only one.)

  python -m unittest task_synthesis.serve.test_serve_teacher_tasksynth     (harbor venv)
"""
import json
import unittest

from synth.config import load_teacher
from synth.vllm_launch import replica_command
from task_synthesis.serve.serve_teacher import PROFILES, engine_command, profile_for

SLUG = 'deepseek-v41-flash-1017'
PIPELINE = 'task-synthesis'


def flag(argv, name):
    return argv[argv.index(name) + 1]


class DeepSeekProfile(unittest.TestCase):
    def setUp(self):
        self.teacher = load_teacher(SLUG)
        self.argv, self.tunables, self.ports = engine_command(SLUG, self.teacher, '/model', 4, pipeline=PIPELINE)

    def test_speculative_tokens_and_runtime_are_the_specs(self):
        """DSpark-5 on vLLM main since 2026-11-19 (one-shot replay: main +8.4-9.0% over vllm032, 5 ties 3; #478)."""
        spec = json.loads(flag(self.argv, '--speculative-config'))
        self.assertEqual(spec['method'], 'dspark')
        self.assertEqual(spec['num_speculative_tokens'], 5)
        self.assertTrue(spec['enable_adaptive_verification'])
        profile = profile_for(SLUG, PIPELINE)
        for key in ('spec_tokens', 'venv', 'vllm_version'):   # no pin left: the spec's (both keys go together)
            self.assertNotIn(key, profile['tunables'], key)
        self.assertNotIn('seed_from', profile)                 # 8162cffd... was a vllm032 snapshot
        self.assertTrue(self.argv[0].startswith(self.teacher['runtime']['venv']), self.argv[0])

    def test_rest_of_the_profile_and_teacher_command(self):
        self.assertEqual(flag(self.argv, '--data-parallel-size'), '4')
        self.assertEqual(flag(self.argv, '--max-model-len'), '132000')
        self.assertNotIn('--kv-offloading-size', self.argv, 'no CPU KV tier for task generation (#242)')
        self.assertNotIn('--kv-transfer-config', self.argv)
        self.assertIn('--enable-prompt-tokens-details', self.argv)
        self.assertEqual(flag(self.argv, '--max-num-seqs'), '768')
        self.assertEqual(len(self.ports), 1)

    def test_request_sizes(self):
        self.assertEqual((profile_for(SLUG, PIPELINE)['memory_gib'], profile_for(SLUG, PIPELINE)['cpus']), (320, 20))

    def test_engram_tables_fit_the_jobs_dev_shm(self):
        """vLLM main shares DeepSeek's two offloaded Engram tables (193,365 MiB together) across DP ranks through
        /dev/shm only when they fit; the job wrapper (/usr/local/share/htcondor/setup_local_quota.sh) mounts /dev/shm at
        the job's DevShmSize, else half the slot's memory. Every DeepSeek profile must leave them room, and the shared
        tables (plus ~34 GiB of processes: the #478 replay's 223 GiB peak) must fit the memory request."""
        from task_synthesis.serve import serve_teacher as st
        need_gib = 193365 / 1024
        teacher = load_teacher(SLUG)
        for pipeline in ('aa-serving', PIPELINE):
            p = profile_for(SLUG, pipeline)
            memory = st.memory_gib(teacher, SLUG, 4, pipeline)
            shm = p.get('dev_shm_gib') or teacher['serving_resources'].get('dev_shm_gib') or memory / 2
            self.assertGreaterEqual(shm, need_gib + 8, pipeline)
            self.assertLess(shm, memory, pipeline)
            self.assertGreaterEqual(memory, 223, pipeline)

    def test_the_profile_changes_only_the_tier_and_protocol(self):
        """Against the teacher's command with the profile's tunables: identical (no command override left; the profile
        sets protocol, prompt token details and drops the CPU KV tier through its tunables)."""
        profile = profile_for(SLUG, PIPELINE)
        base = replica_command(self.teacher, profile['protocol'], port=int(flag(self.argv, '--port')),
                               model_dir='/model',
                               tunables={**self.teacher['serving']['tunables'], **profile['tunables']},
                               rank_ports={'dp_size': 4})
        self.assertEqual(base, self.argv)


class OtherPipelines(unittest.TestCase):
    def test_the_default_pipeline_keeps_the_eval_profile(self):
        """aa-serving (every server since 11-14: c30, M5, AA evals) runs DeepSeek with the CPU KV tier and DSpark-5."""
        teacher = load_teacher(SLUG)
        argv, _, _ = engine_command(SLUG, teacher, '/model', 4)
        self.assertEqual(json.loads(flag(argv, '--speculative-config'))['num_speculative_tokens'], 5)
        self.assertIn('--kv-offloading-size', argv)
        self.assertIs(profile_for(SLUG), PROFILES[SLUG])

    def test_teachers_a_pipeline_does_not_name_run_their_profiles_entry(self):
        for slug in ('qwen38-27b-eval-fast', 'glm53-flash-eval'):
            self.assertIs(profile_for(slug, PIPELINE), PROFILES[slug])
        self.assertEqual(profile_for('nemotron35-super', PIPELINE), {})   # no profile: the PROTOCOL default


class StagedExtras(unittest.TestCase):
    def test_bolt_draft_path_fills_its_command(self):
        """#445: nemotron35-bolt's {stage_draft} (its DFlash draft) is staged and passed like synth.serve does."""
        teacher = load_teacher('nemotron35-bolt')
        with self.assertRaises(KeyError):
            engine_command('nemotron35-bolt', teacher, '/model', 4)
        argv, _, _ = engine_command('nemotron35-bolt', teacher, '/model', 4, staged={'stage_draft': '/scratch/draft'})
        self.assertIn('/scratch/draft', ' '.join(argv))



class FrozenCheckouts(unittest.TestCase):
    def test_tasksynth_worktrees_moves_frozen_checkouts(self):
        """aa-serving 09893a53: TASKSYNTH_WORKTREES puts serve_teacher's frozen checkouts elsewhere (default unchanged)."""
        import importlib
        import os
        from unittest import mock
        from task_synthesis import common
        try:
            with mock.patch.dict(os.environ, {'TASKSYNTH_WORKTREES': '/data/x/worktrees'}):
                self.assertEqual(str(importlib.reload(common).WORKTREES), '/data/x/worktrees')
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('TASKSYNTH_WORKTREES', None)
                self.assertEqual(str(importlib.reload(common).WORKTREES), '/opt/aai/worktrees')
        finally:
            importlib.reload(common)


class Registry(unittest.TestCase):
    """serve_teacher status / wait / release (tbshared.serving, #445 item 3) on a fake registry and fake condor."""

    def run_op(self, argv, root, queue):
        import io
        import types
        from contextlib import redirect_stdout
        from unittest import mock
        from task_synthesis.serve import serve_teacher as st
        calls = []

        def fake(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout=queue if cmd[0] == 'condor_q' else '')
        out = io.StringIO()
        from tbshared import serving
        # never a real condor command from a test: the helpers resolve subprocess.run per call, patched here
        with mock.patch.object(st, 'ROOT', root), mock.patch.object(serving.subprocess, 'run', fake), \
                mock.patch('sys.argv', ['serve_teacher', *argv]), redirect_stdout(out), self.assertRaises(SystemExit) as e:
            st.main()
        return e.exception.code, json.loads(out.getvalue()), calls

    def test_status_wait_release(self):
        import tempfile
        import time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'srv'
            home.mkdir()
            (home / 'job.json').write_text(json.dumps({'job_id': '9.0'}))
            code, out, _ = self.run_op(['status', '--name', 'srv'], Path(d), '1 user')
            self.assertEqual((code, out['state']), (0, 'starting'))
            now = time.time()
            (home / 'endpoints.json').write_text(json.dumps({'healthy_at': now, 'updated_at': now, 'urls': ['u']}))
            code, out, _ = self.run_op(['wait', '--name', 'srv', '--timeout', '5'], Path(d), '2 user')
            self.assertEqual((code, out['ready'], out['urls']), (0, True, ['u']))
            import os
            code, out, calls = self.run_op(['release', '--name', 'srv', '--reason', 'done'], Path(d),
                                           f'2 {os.environ.get("USER", "user")}')
            self.assertEqual((code, out['action']), (0, 'removed'))
            self.assertIn(['condor_rm', '9.0'], calls)


class RegistryRoot(unittest.TestCase):
    def test_env_moves_the_registry_and_reaches_the_job(self):
        """TASKSYNTH_SERVING_ROOT: serve_teacher's registry (job.json, endpoints.json) is the canary's, and the job it
        submits gets the same variable; unset, the production registry as before."""
        import importlib
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock
        from task_synthesis.serve import serve_teacher as st
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'TASKSYNTH_SERVING_ROOT': d}):
                st = importlib.reload(st)
                self.assertEqual(st.ROOT, Path(d))
                with mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')), \
                        mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n')):
                    a = mock.Mock(teacher='qwen35-4b-eval', gpus=1, force=False, tunables='', protocol='', cuda_cap='',
                                  hours=1, bid=61, cpus=0, memory_gib=0, checkpoint='', dry_run=False, exclude_nodes='',
                                  bypass_lxcfs=False, pipeline='aa-serving', fill=None)
                    a.name = 'canary'
                    st.submit(a)
                sub = (Path(d) / 'canary' / 'job.sub').read_text()
                self.assertIn(f"TASKSYNTH_SERVING_ROOT='{d}'", sub)
                self.assertTrue((Path(d) / 'canary' / 'job.json').exists())
            env = {k: v for k, v in os.environ.items() if k != 'TASKSYNTH_SERVING_ROOT'}
            with mock.patch.dict(os.environ, env, clear=True), tempfile.TemporaryDirectory() as d:
                st = importlib.reload(st)
                self.assertEqual(st.ROOT, st.TS_ROOT / 'serving')
                with mock.patch.object(st, 'ROOT', Path(d)), \
                        mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')), \
                        mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n')):
                    st.submit(a)
                # a registry set on the module (tests, tools) reaches the job too: it is keyed on ROOT, not the env
                self.assertIn(f"TASKSYNTH_SERVING_ROOT='{d}'", (Path(d) / 'canary' / 'job.sub').read_text())
        finally:
            importlib.reload(st)


if __name__ == '__main__':
    unittest.main()

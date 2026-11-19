"""Engine compile-cache snapshots: stable per-job paths, the configuration key, seed and publish."""
import os
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from synth import engine_cache

MOUNTINFO = '''24 1 9:3 / / rw,relatime shared:1 - xfs /dev/md3 rw
612 24 9:3 /condor_execute/dir_584245/tmp /tmp rw,relatime - xfs /dev/md3 rw,prjquota
613 24 9:3 /condor_execute/dir_584245/var_tmp /var/tmp rw,relatime - xfs /dev/md3 rw,prjquota
'''


class EngineCache(unittest.TestCase):
    def test_job_private_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            info = Path(d) / 'mountinfo'
            info.write_text(MOUNTINFO)
            self.assertTrue(engine_cache.job_private_tmp('/var/lib/condor/execute/dir_584245', info))
            self.assertFalse(engine_cache.job_private_tmp('/var/lib/condor/execute/dir_1', info))
            info.write_text(MOUNTINFO.splitlines()[0] + '\n')
            self.assertFalse(engine_cache.job_private_tmp('/var/lib/condor/execute/dir_584245', info))

    def test_key_ignores_the_port_and_job_specific_env(self):
        with tempfile.TemporaryDirectory() as venv:
            site = Path(venv) / 'lib/python3.12/site-packages'
            (site / 'vllm-0.32.0.dist-info').mkdir(parents=True)
            argv = ['vllm', 'serve', '/tmp/synth/model', '--port', '4000', '--max-num-seqs', '512']
            env = {'VLLM_USE_SIMPLE_KV_OFFLOAD': '1', 'CUDA_VISIBLE_DEVICES': 'GPU-a', 'CUDA_HOME': '/tmp/synth/c'}
            key, facts = engine_cache.config_key(argv, env, venv, ['NVIDIA B200'])
            other, _ = engine_cache.config_key(argv[:4] + ['5000'] + argv[5:], {**env, 'CUDA_VISIBLE_DEVICES': 'GPU-b'},
                                               venv, ['NVIDIA B200'])
            self.assertEqual(key, other)
            self.assertNotIn('CUDA_VISIBLE_DEVICES', facts['env'])
            self.assertEqual(argv[4], '4000')  # the caller's argv is not modified
            changed = [engine_cache.config_key(argv[:-1] + ['256'], env, venv, ['NVIDIA B200'])[0],
                       engine_cache.config_key(argv, {**env, 'VLLM_USE_SIMPLE_KV_OFFLOAD': '0'}, venv, ['NVIDIA B200'])[0],
                       engine_cache.config_key(argv, env, venv, ['NVIDIA H100'])[0]]
            (site / 'flashinfer_python-0.7.2.dist-info').mkdir()
            changed.append(engine_cache.config_key(argv, env, venv, ['NVIDIA B200'])[0])
            self.assertEqual(len({key, *changed}), 5)

    def test_key_ignores_dp_rpc_and_supervisor_ports(self):
        """GLM's dp_external_lb engines get random DP RPC / supervisor ports each start: same key regardless (#378)."""
        with tempfile.TemporaryDirectory() as venv:
            argv = lambda rpc, sup: ['vllm', 'serve', '/m', '--port', '4000', '--data-parallel-rpc-port', str(rpc),
                                     '--data-parallel-supervisor-port', str(sup), '--max-num-seqs', '256']
            a, _ = engine_cache.config_key(argv(23377, 29576), {}, venv, ['NVIDIA B200'])
            b, _ = engine_cache.config_key(argv(26685, 29407), {}, venv, ['NVIDIA B200'])
            self.assertEqual(a, b)

    def test_publish_then_seed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shared, source, target = d / 'shared', d / 'job-a' / 'cache', d / 'job-b' / 'cache'
            (source / 'flashinfer' / 'cached_ops' / 'm').mkdir(parents=True)
            (source / 'flashinfer' / 'cached_ops' / 'm' / 'm.so').write_bytes(os.urandom(3 << 20) + b'tail')
            (source / 'triton' / 'k.json').parent.mkdir(parents=True)
            (source / 'triton' / 'k.json').write_text('{}')
            (source / 'vllm_cache_root' / 'flashinfer_autotune_cache').mkdir(parents=True)
            (source / 'vllm_cache_root' / 'flashinfer_autotune_cache' / 'configs.json').write_text('{}')
            self.assertEqual(engine_cache.seed(target, 'k1', shared)['reason'], 'no snapshot')
            result = engine_cache.publish(source, 'k1', {'argv': []}, d, {'job_id': 'a'}, shared)
            self.assertTrue(result['published'], result)
            self.assertEqual(sorted(p.name for p in shared.iterdir()), ['k1.json', 'k1.tar'])
            self.assertEqual(engine_cache.publish(source, 'k1', {}, d, {}, shared)['reason'], 'exists')
            seeded = engine_cache.seed(target, 'k1', shared)
            self.assertTrue(seeded['seeded'], seeded)
            self.assertEqual((target / 'flashinfer' / 'cached_ops' / 'm' / 'm.so').read_bytes(),
                             (source / 'flashinfer' / 'cached_ops' / 'm' / 'm.so').read_bytes())
            self.assertFalse((target / 'vllm_cache_root').exists(), 'autotune and torch.compile caches stay local')
            again = engine_cache.seed(target, 'k1', shared)
            self.assertEqual((again['seeded'], again['reason']), (False, 'already seeded in this job'))
            (shared / 'k2.lock').mkdir()
            self.assertEqual(engine_cache.publish(source, 'k2', {}, d, {}, shared)['reason'],
                             'another job is publishing')

    def test_the_vllm_root_is_emptied_before_each_start_inside_the_engine_scratch_only(self):
        with tempfile.TemporaryDirectory() as d:
            scratch = Path(d) / 'engines' / 'r0'
            root = scratch / 'cache' / 'vllm_cache_root'
            (root / 'torch_compile_cache').mkdir(parents=True)
            (scratch / 'cache' / 'triton_cache_dir').mkdir(parents=True)
            self.assertTrue(engine_cache.fresh_vllm_root({'VLLM_CACHE_ROOT': str(root)}, scratch))
            self.assertFalse(root.exists())
            self.assertTrue((scratch / 'cache' / 'triton_cache_dir').exists())
            outside = Path(d) / 'elsewhere'
            outside.mkdir()
            self.assertFalse(engine_cache.fresh_vllm_root({'VLLM_CACHE_ROOT': str(outside)}, scratch))
            self.assertTrue(outside.exists())
            self.assertFalse(engine_cache.fresh_vllm_root({}, scratch))

    def test_oversized_caches_are_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'cache').mkdir()
            (d / 'cache' / 'big.so').write_bytes(b'x' * 4096)
            with unittest.mock.patch.object(engine_cache, 'MAX_SNAPSHOT_BYTES', 1024):
                result = engine_cache.publish(d / 'cache', 'k', {}, d, {}, d / 'shared')
            self.assertFalse(result['published'])
            self.assertEqual(sorted(p.name for p in (d / 'shared').iterdir()), [])

    def test_a_broken_snapshot_leaves_a_cold_cache(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'shared').mkdir()
            (d / 'shared' / 'k.tar').write_bytes(b'not a tar archive' * 100)
            result = engine_cache.seed(d / 'cache', 'k', d / 'shared')
            self.assertFalse(result['seeded'])
            self.assertFalse((d / 'cache').exists())


if __name__ == '__main__':
    unittest.main()


def test_renderer_workers_do_not_change_the_key():
    """#451 ws4: Super specs gained --renderer-num-workers 4; their compile snapshots must keep seeding."""
    from synth import engine_cache
    base = ['vllm', 'serve', '/m', '--port', '1', '--max-num-seqs', '8']
    k0, _ = engine_cache.config_key(base, {}, '/nonexistent', 'B200')
    k1, f1 = engine_cache.config_key(base[:3] + ['--renderer-num-workers', '4'] + base[3:], {}, '/nonexistent', 'B200')
    assert k0 == k1 and '--renderer-num-workers' not in f1['argv']
    k2, _ = engine_cache.config_key(base + ['--max-num-batched-tokens', '4'], {}, '/nonexistent', 'B200')
    assert k2 != k0


def test_flashinfer_no_download_does_not_change_the_key():
    """DeepSeek's spec gained FLASHINFER_NO_DOWNLOAD=1 (refuse to fetch a cubin the wheel lacks): its main snapshots
    keep seeding; a compile-relevant FLASHINFER_ variable still changes the key."""
    from synth import engine_cache
    base = ['vllm', 'serve', '/m', '--port', '1']
    env = {'FLASHINFER_WORKSPACE_BASE': '/x'}
    k0, _ = engine_cache.config_key(base, env, '/nonexistent', 'B200')
    k1, f1 = engine_cache.config_key(base, {**env, 'FLASHINFER_NO_DOWNLOAD': '1'}, '/nonexistent', 'B200')
    assert k0 == k1 and 'FLASHINFER_NO_DOWNLOAD' not in f1['env']
    k2, _ = engine_cache.config_key(base, {**env, 'FLASHINFER_JIT_VERBOSE': '1'}, '/nonexistent', 'B200')
    assert k2 != k0


def test_the_code_checkout_does_not_change_the_key():
    """A spec's {repo} paths (the Snowball family's plugin PYTHONPATH and parser files) name the job's frozen checkout,
    one per commit: two commits' engines of one recipe have the same key, so a server seeds across commits."""
    from synth import engine_cache
    def render(root):
        argv = ['/v/bin/vllm', 'serve', '/m', '--port', '1', '--tool-parser-plugin', f'{root}/serve_snowball/p.py']
        return argv, {'PYTHONPATH': f'{root}/serve_snowball/plugin_v034', 'VLLM_CACHE_ROOT': '/tmp/synth/c'}
    a, b = '/w/tasksynth-run-aaaa', '/w/tasksynth-run-bbbb'
    ka, fa = engine_cache.config_key(*render(a), '/nonexistent', 'B200', repo=a)
    kb, _ = engine_cache.config_key(*render(b), '/nonexistent', 'B200', repo=b)
    assert ka == kb and fa['env']['PYTHONPATH'] == '{repo}/serve_snowball/plugin_v034' and '{repo}/serve_snowball/p.py' in fa['argv']
    other, _ = engine_cache.config_key(*render(a), '/nonexistent', 'B200', repo='/elsewhere')   # not the checkout: kept
    assert other != ka


def test_specs_without_repo_paths_keep_their_keys():
    """Only {repo} paths are abstracted: a command without the checkout keys exactly as before (existing snapshots of
    every other spec keep seeding). The value was computed with origin/main's config_key (eb55b1da)."""
    from synth import engine_cache
    argv = ['/v/bin/vllm', 'serve', '/tmp/synth/model', '--port', '1', '--max-num-seqs', '8']
    env = {'CUDA_HOME': '/c', 'VLLM_X': '1', 'PATH': '/v/bin'}
    assert engine_cache.config_key(argv, env, '/nonexistent', 'NVIDIA B200')[0] == '6076dd6b650be830b392'


def test_only_the_checkout_path_itself_is_abstracted():
    """A path that merely starts with the checkout's string (another worktree) is another directory: kept."""
    from synth import engine_cache
    root = '/w/task-synthesis'
    _, f = engine_cache.config_key(['/v/bin/vllm', f'{root}/p.py', f'{root}-v2/p.py'], {'PYTHONPATH': f'{root}-v2:{root}'},
                                   '/nonexistent', 'B200', repo=root)
    assert f['argv'][1:] == ['{repo}/p.py', f'{root}-v2/p.py'] and f['env']['PYTHONPATH'] == f'{root}-v2:{{repo}}'


def test_snowball_spec_keys_match_across_checkouts(tmp_path):
    """The real snowball-eval-main command, rendered in this checkout and as another commit's frozen copy would."""
    from synth import engine_cache
    from synth.common import REPO
    from synth.config import load_teacher
    from synth.vllm_launch import replica_command, replica_env
    t = load_teacher('snowball-eval-main')
    argv = replica_command(t, {'context_tokens': 131072, 'max_output_tokens': 131072}, port=8000,
                           model_dir='/tmp/synth/model', tunables=t['serving'].get('tunables', {}))
    env = replica_env(t, gpus=['0'], scratch=tmp_path, tunables=t['serving'].get('tunables', {}), base_env={})
    assert any(str(REPO) in a for a in argv) and str(REPO) in env['PYTHONPATH']
    other = '/opt/aai/worktrees/tasksynth-run-0123456789'
    k_here, _ = engine_cache.config_key(argv, env, '/nonexistent', 'B200')
    k_other, _ = engine_cache.config_key([a.replace(str(REPO), other) for a in argv],
                                         {k: v.replace(str(REPO), other) for k, v in env.items()},
                                         '/nonexistent', 'B200', repo=other)
    assert k_here == k_other


def test_seed_first_falls_back_in_order(tmp_path):
    """Own snapshot first; else the first fallback that exists; the record names what seeded and the own key."""
    import tarfile
    from synth import engine_cache
    shared = tmp_path / 'shared'
    shared.mkdir()
    def snap(key, name):
        src = tmp_path / f'src-{key}'
        (src / 'x').mkdir(parents=True)
        (src / 'x' / name).write_text(key)
        with tarfile.open(shared / f'{key}.tar', 'w') as t:
            t.add(src, arcname='.')
    snap('fb2', 'kernel.o')
    snap('own', 'own.o')
    r = engine_cache.seed_first(tmp_path / 'c1', 'own', ['fb1', 'fb2'], shared=shared)
    assert r['seeded'] and r['key'] == 'own' and 'own_key' not in r and 'tried' not in r
    r = engine_cache.seed_first(tmp_path / 'c2', 'new', [None, 'fb1', 'fb2', 'fb2'], shared=shared)
    assert r['seeded'] and r['key'] == 'fb2' and r['own_key'] == 'new'
    assert r['tried'] == [{'key': 'new', 'reason': 'no snapshot'}, {'key': 'fb1', 'reason': 'no snapshot'}]
    assert (tmp_path / 'c2' / 'x' / 'kernel.o').read_text() == 'fb2'
    r = engine_cache.seed_first(tmp_path / 'c3', 'new', ['fb1'], shared=shared)
    assert not r['seeded'] and r['reason'] == 'no snapshot' and r['own_key'] == 'new'
    assert [t['key'] for t in r['tried']] == ['new', 'fb1']
    (shared / 'bad.tar').write_bytes(b'not a tar')   # a corrupt own snapshot: its reason stays in the record
    r = engine_cache.seed_first(tmp_path / 'c4', 'bad', ['fb2'], shared=shared)
    assert r['seeded'] and r['key'] == 'fb2' and r['tried'][0]['key'] == 'bad'
    assert r['tried'][0]['reason'].startswith('extract failed')
    r = engine_cache.seed_first(tmp_path / 'c3', 'new', [], shared=shared)   # no fallbacks: exactly seed()
    assert r == engine_cache.seed(tmp_path / 'c3', 'new', shared=shared)
    r = engine_cache.seed_first(tmp_path / 'c2', 'new', ['fb2'], shared=shared)   # already seeded in this job: no retry
    assert r['reason'] == 'already seeded in this job' and 'tried' not in r


def test_synth_serve_seed_fallbacks():
    """Campaign serving.seed_from first, then the spec's; a key or a list; empties and repeats dropped."""
    from synth.serve import seed_fallbacks
    assert seed_fallbacks({}, {'serving': {}}) == []
    assert seed_fallbacks({'serving': {'seed_from': 'a'}}, {'serving': {'seed_from': ['b', 'a', '']}}) == ['a', 'b']
    assert seed_fallbacks({'serving': {}}, {'serving': {'seed_from': 'b'}}) == ['b']
    assert seed_fallbacks({'serving': {'seed_from': 7}}, {'serving': {'seed_from': ['b', 3]}}) == ['b']   # not fatal


def test_seed_from_snapshots_exist():
    """Every seed_from a serve_teacher profile or a spec names is a published snapshot (an unpublished or deleted key
    would silently seed nothing)."""
    import pytest
    from synth import engine_cache
    from synth.config import load_teacher
    from synth.common import REPO
    from task_synthesis.serve import serve_teacher as st
    if not engine_cache.SHARED.is_dir():
        pytest.skip(f'{engine_cache.SHARED} not mounted')
    keys = {p['seed_from'] for p in st.PROFILES.values() if p.get('seed_from')}
    keys |= {p['seed_from'] for ps in st.PIPELINE_PROFILES.values() for p in ps.values() if p.get('seed_from')}
    for f in (REPO / 'synth' / 'teachers').glob('*.toml'):
        value = load_teacher(f.stem)['serving'].get('seed_from')
        keys |= {value} if isinstance(value, str) else set(value or [])
    assert {'15282107916d460058f1', 'cbe361c05603776b9444'} <= keys
    missing = sorted(k for k in keys if not (engine_cache.SHARED / f'{k}.tar').is_file())
    assert not missing, missing

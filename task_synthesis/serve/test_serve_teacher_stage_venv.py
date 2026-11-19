"""serve_teacher tunable stage_venv: an engine on a node-local copy of its venv runs the copy everywhere the venv
appears, records the same numerics as from the source venv, and seeds its first start from the source recipe's
engine-cache snapshot."""
import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest

from synth import engine_cache, numerics
from synth.config import load_teacher
from task_synthesis.serve import serve_teacher as st

MAIN_SPECS = ('deepseek-v41-flash-1017', 'snowball-eval-main', 'nemotron35-super-mm-text')


def fake_copy(source: str, root: Path) -> Path:
    """A stand-in for the node-local copy: its own path, the source's *.dist-info names, and its nvidia/cu13 wheel
    linked in (what a cu13-wheel CUDA_HOME is built over)."""
    copy = root / 'venvs' / Path(source).name
    for dist in Path(source).glob('lib/python3*/site-packages/*.dist-info'):   # the same distributions (config_key
        (copy / dist.relative_to(source)).mkdir(parents=True, exist_ok=True)    # lists them): keys differ by path only
    for cu13 in Path(source).glob('lib/python3*/site-packages/nvidia/cu13'):
        dst = copy / cu13.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(cu13, dst)
    (copy / 'bin').mkdir(parents=True, exist_ok=True)
    return copy


def build(slug, scratch, staged=None, stage=True):
    teacher = load_teacher(slug)
    plan = st.placeholder_plan(teacher, slug, 1, tunables_json='{"stage_venv": %d}' % int(stage))   # profiles set 1
    tun = {**plan['tunables'], **(plan['per_replica'][0] if plan['per_replica'] else {})}
    argv, env, source, copy, unstage = st.replica_launch(
        teacher, tun, plan['protocol'], gpus=['0'], port=8000, model_dir='/tmp/synth/model', rank_ports=plan['extras'][0],
        scratch=scratch, staged_venvs=staged)
    return teacher, plan, argv, env, source, copy, unstage


@pytest.mark.parametrize('slug', MAIN_SPECS)
def test_staged_engine_runs_the_copy_with_the_same_numerics(slug, tmp_path):
    teacher = load_teacher(slug)
    source = teacher['runtime']['venv']
    if not Path(source).is_dir():
        pytest.skip(f'{source} not mounted')
    copy = fake_copy(source, tmp_path / 'node')
    scratch = tmp_path / 'node' / 'engines' / 'r0'
    _, plan, argv0, env0, _, none, _ = build(slug, scratch, staged={source: (copy, {})}, stage=False)
    assert none is None                                  # without the tunable: the source venv, as before
    _, _, argv, env, src, got, unstage = build(slug, scratch, staged={source: (copy, {})})
    assert src == source and got == copy
    assert argv[0].startswith(f'{copy}/bin/') and argv0[0].startswith(f'{source}/bin/')   # vllm or python
    assert all(source not in str(v) for v in [*argv, *env.values()])   # a spec's literal PATH too (Snowball's)
    assert f'{copy}/bin' in env['PATH'].split(':')
    if teacher['runtime'].get('cuda_home') == 'cu13-wheel':           # Super: CUDA_HOME over the copy's wheel
        assert str(copy) in env['LD_LIBRARY_PATH']
    kw = dict(model_dir='/tmp/synth/model', scratch_roots=(tmp_path / 'node',), protocol=plan['protocol'], venv=source,
              gpu='NVIDIA B200')
    staged_rec = numerics.replica_numerics(teacher, [unstage(a) for a in argv], {k: unstage(v) for k, v in env.items()},
                                           base_env=os.environ, **kw)
    plain_rec = numerics.replica_numerics(teacher, argv0, env0, base_env=os.environ, **kw)
    assert staged_rec['sha256'] == plain_rec['sha256']                 # same files, same numerics
    # its own engine-cache key (paths differ, so FlashInfer's ninja lines do), and the fallback seed key is exactly
    # the key a /fast-venv engine of the same recipe publishes under
    key, _ = engine_cache.config_key(argv, env, str(copy), 'NVIDIA B200')
    plain_key, _ = engine_cache.config_key(argv0, env0, source, 'NVIDIA B200')
    src_key, _ = engine_cache.config_key([unstage(a) for a in argv], {k: unstage(v) for k, v in env.items()}, source,
                                         'NVIDIA B200')
    assert key != plain_key and src_key == plain_key


def test_no_copy_runs_the_source(tmp_path):
    """stage_venv set but the venv could not be staged (no tar, mismatch): the engine runs the source venv."""
    _, _, argv, env, source, copy, unstage = build('snowball-eval-main', tmp_path, staged={})
    assert copy is None and argv[0].startswith(f'{source}/bin/') and unstage('x') == 'x'


def test_venvs_to_stage():
    teacher = load_teacher('snowball-eval-main')
    on = st.placeholder_plan(teacher, 'snowball-eval-main', 2, tunables_json='{"stage_venv": 1}')
    off = st.placeholder_plan(teacher, 'snowball-eval-main', 2, tunables_json='{"stage_venv": 0}')
    assert st.venvs_to_stage(on, teacher) == [teacher['runtime']['venv']] and st.venvs_to_stage(off, teacher) == []


@pytest.mark.parametrize('tunables,extra', [('{"stage_venv": 0}', 0), ('', 16), ('{"stage_venv": 1}', 16)])
def test_submit_requests_disk_for_the_copy(tunables, extra, tmp_path):
    """request_disk = the spec's disk_gib + the tar's size (rounded up) with stage_venv (the profile's default), unchanged
    without."""
    teacher = load_teacher('snowball-eval-main')
    with mock.patch.object(st, 'ROOT', tmp_path), mock.patch.object(st, 'frozen_checkout', return_value=Path('/code')), \
            mock.patch.object(st.venv_stage, 'tar_gib', return_value=15.92), \
            mock.patch.object(st.subprocess, 'run', return_value=mock.Mock(stdout='submitted to cluster 7.\n', stderr='')):
        a = mock.Mock(teacher='snowball-eval-main', gpus=1, force=False, tunables=tunables, protocol='', cuda_cap='',
                      hours=1, bid=61, cpus=0, memory_gib=0, checkpoint='', dry_run=True, exclude_nodes='',
                      bypass_lxcfs=False, pipeline='aa-serving', fill=None, allow_shape=False)
        a.name = 'n'
        st.submit(a)
        sub = (tmp_path / 'n' / 'job.sub.dry').read_text()
    disk = int(teacher['serving_resources'].get('disk_gib', 500)) + extra
    assert f'request_disk = {disk}G' in sub


def test_swap_path_whole_paths_only():
    v = '/data/venvs/vllm-x'
    assert st.swap_path(f'{v}/bin:/usr/bin:{v}', v, '/tmp/c') == '/tmp/c/bin:/usr/bin:/tmp/c'
    assert st.swap_path(f'{v}-other/bin', v, '/tmp/c') == f'{v}-other/bin'   # a longer name is another venv
    assert st.swap_path(f'/mnt{v}/bin', v, '/tmp/c') == f'/mnt{v}/bin'       # so is a longer path ending in it
    assert st.swap_path(f'--x={v}/lib', v, '/tmp/c') == '--x=/tmp/c/lib'


MAIN_VENV = '/data/aai/venvs/vllm-0.34.0rc1.dev815-g8f3c672dc-cu132'
# vLLM-main profiles not staged yet, each with its reason (DeepSeek joined after smoke 17980755)
NOT_YET = {}


def all_profiles():
    for slug in st.PROFILES:
        yield 'aa-serving', slug
    for pipeline, profiles in st.PIPELINE_PROFILES.items():
        for slug in profiles:
            yield pipeline, slug


def test_main_build_profiles_stage_their_venv():
    """Every profile of a spec on the vLLM-main venv turns stage_venv on (coordinator, after #554), unless listed in
    NOT_YET; no other profile does (a spec moving to main shows up here)."""
    for pipeline, slug in sorted(set(all_profiles())):
        on = bool(st.profile_for(slug, pipeline).get('tunables', {}).get('stage_venv'))
        main = load_teacher(slug)['runtime'].get('venv') == MAIN_VENV
        assert on == (main and (pipeline, slug) not in NOT_YET), (pipeline, slug, on, main)
    assert ('aa-serving', 'nemotron35-super-mm-text') in set(all_profiles())


def test_caller_tunables_keep_the_profile_stage_venv():
    """M5's own --tunables ({"kv_offloading_size": 48}) still stage; {"stage_venv": 0} turns it off."""
    t = load_teacher('nemotron35-super-mm-text')
    on = st.placeholder_plan(t, 'nemotron35-super-mm-text', 2, tunables_json='{"kv_offloading_size": 48}')
    off = st.placeholder_plan(t, 'nemotron35-super-mm-text', 2, tunables_json='{"kv_offloading_size": 48, "stage_venv": 0}')
    assert st.venvs_to_stage(on, t) == [MAIN_VENV] and st.venvs_to_stage(off, t) == []


def test_staged_profiles_have_a_current_tar():
    """The staged profiles' venv has a tar that matches it: after a change to the venv, rebuild the tar
    (python -m synth.venv_stage build VENV, in a CPU job) or its servers fall back to /data (~2 min slower)."""
    venvs = {load_teacher(slug)['runtime']['venv'] for pipeline, slug in all_profiles()
             if st.profile_for(slug, pipeline).get('tunables', {}).get('stage_venv')}
    assert venvs == {MAIN_VENV}
    if not Path(MAIN_VENV).is_dir():
        pytest.skip(f'{MAIN_VENV} not mounted')
    meta, reason = st.venv_stage.check(MAIN_VENV)
    assert meta, reason

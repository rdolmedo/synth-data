"""Build one teacher replica's vLLM command line and environment from its spec.

Compile/JIT caches live on node-local scratch: shared /data caches cost
thousands of inodes per job and FlashInfer/Triton locks need real flock.
"""
import json
import os
from pathlib import Path


def cuda_home_from_wheel(venvs, target):
    """Synthetic CUDA_HOME over nvidia/cu13 wheels (+ libcuda stubs); earlier venvs win."""
    roots = [r for v in venvs for r in Path(v).glob('lib/python3*/site-packages/nvidia/cu13')]
    if not roots:
        raise FileNotFoundError(f'No nvidia/cu13 wheel in {venvs}')
    target = Path(target)
    (target / 'lib64' / 'stubs').mkdir(parents=True, exist_ok=True)
    def link(src, dst):
        dst = Path(dst)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    for name in ('bin', 'include', 'nvvm', 'cccl', 'share'):
        source = next((r / name for r in roots if (r / name).exists()), None)
        if source:
            link(source, target / name)
    libraries = {}
    for root in roots:
        for lib in (root / 'lib').glob('*'):
            libraries.setdefault(lib.name, lib)
    for lib in libraries.values():
        link(lib, target / 'lib64' / lib.name)
        if '.so.' in lib.name:
            link(lib, target / 'lib64' / (lib.name.split('.so.')[0] + '.so'))
    for dst in (target / 'lib64' / 'stubs' / 'libcuda.so', target / 'lib64' / 'libcuda.so'):
        link('/usr/lib/x86_64-linux-gnu/libcuda.so.1', dst)
    if not (target / 'lib').exists():
        (target / 'lib').symlink_to('lib64')
    return target, roots


def fmt(template, values):
    return template.format(**values)


def replica_command(teacher, protocol, *, port, model_dir, tunables, rank_ports=None):
    serving = teacher['serving']
    runtime = teacher['runtime']
    values = {'venv': runtime['venv'], 'model_dir': model_dir, 'served_name': teacher['served_name'],
              'port': port, 'seed': serving.get('seed', 0), 'max_model_len': protocol['context_tokens'],
              'max_output_tokens': protocol['max_output_tokens'], 'checkpoint': teacher['checkpoint'],
              # the teacher's tunables are defaults under any control file (older ones lack newer keys)
              **(rank_ports or {}), **serving.get('tunables', {}), **tunables}
    argv = [fmt(x, values) for x in serving['entrypoint']] + [fmt(x, values) for x in serving['args']]
    for key, extra in serving.get('optional_args', {}).items():
        if tunables.get(key):
            argv += [fmt(x, values) for x in extra]
    # Per-arm vLLM flags, appended verbatim (as `extra_env` is for the environment): tuning candidates.
    return argv + [str(x) for x in tunables.get('extra_args') or []]


def replica_env(teacher, *, gpus, scratch, tunables, base_env=None):
    runtime = teacher['runtime']
    venv = tunables.get('venv') or runtime['venv']   # a tunable venv (an engine-version A/B arm) as in replica_command
    env = dict(base_env if base_env is not None else os.environ)
    env.pop('PYTHONPATH', None)  # the supervisor's repo path must not shadow engine modules
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    visible = list(gpus)
    if runtime.get('cuda_visible_devices') == 'index':   # engines that need integer device ids (vLLM 0.20.1)
        from .gpu import indices
        visible = indices(visible)
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'          # CUDA's order = nvidia-smi's
    env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES=','.join(visible),
               OMP_NUM_THREADS=str(tunables.get('omp_threads', runtime.get('omp_threads', 8))))
    path = [f'{venv}/bin', env.get('PATH', '/usr/bin:/bin')]
    if runtime.get('cuda_home') == 'cu13-wheel':
        cuda_home, roots = cuda_home_from_wheel([venv, *runtime.get('cu13_fallback_venvs', [])], scratch / 'cuda_home')
        env['CUDA_HOME'] = str(cuda_home)
        path.insert(0, str(cuda_home / 'bin'))
        if runtime.get('cu13_ld_library_path'):
            env['LD_LIBRARY_PATH'] = ':'.join(filter(None, [*runtime.get('extra_ld_library_paths', []),
                *(str(r / 'lib') for r in roots), str(cuda_home / 'lib64'), env.get('LD_LIBRARY_PATH')]))
    elif runtime.get('cuda_home'):
        env['CUDA_HOME'] = runtime['cuda_home']
    env['PATH'] = ':'.join(path)
    if runtime.get('nvcc_append_flags'):
        env['NVCC_APPEND_FLAGS'] = ' '.join(filter(None, [env.get('NVCC_APPEND_FLAGS'), runtime['nvcc_append_flags']]))
    for name in runtime.get('cache_env', []):
        env[name] = str(scratch / 'cache' / name.lower())
    for name, value in runtime.get('env', {}).items():
        env[name] = str(value)
    for name, value in teacher['serving'].get('env', {}).items():
        env[name] = str(value).format(**tunables)
    for name, value in (tunables.get('extra_env') or {}).items():  # per-arm env (plugins, feature switches)
        env[name] = str(value)
    no_proxy = 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal'
    env['no_proxy'] = env['NO_PROXY'] = ','.join(filter(None, [no_proxy, env.get('no_proxy')]))
    return env


def describe(argv, env, keys=('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'OMP_NUM_THREADS', 'LD_LIBRARY_PATH')):
    return {'argv': argv, 'env': {k: env.get(k) for k in keys}, 'argv_json': json.dumps(argv)}

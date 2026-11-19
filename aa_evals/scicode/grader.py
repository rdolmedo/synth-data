"""Step grading, as in the official harness (eval/scripts/test_generated_code.py): one script = the problem's
dependencies + the cumulative code of steps 1..k + the step's targets + all its test cases (one shared namespace),
run in a fresh `python -I`. Exit code 0 = pass.

Each script runs in a grading env; a step passes if it passes in any env (SciCode-Verified's OR rule over a 2024-era
and a 2025-era numpy/scipy). Envs:
  sif:<path>       Apptainer SIF (sandbox/build_sif.sh): --containall --cleanenv --no-home --net --network none; only
                   the work dir, the vendored scicode helpers and the step's target h5 are bound in (read-only). Needs
                   a compute node; the tmux host cannot start contained Apptainer.
  local:<python>   a plain subprocess, no isolation: trusted code only (gold self-tests).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'
SIF_DIR = Path('/data/aai/aa_evals/data/scicode/sif')
DEFAULT_ENVS = {'np126': f'sif:{SIF_DIR}/scicode_np126.sif', 'np22': f'sif:{SIF_DIR}/scicode_np22.sif'}
MEM_KB = 16 * 1024 * 1024        # ulimit -v per script (grading suites can memory-bomb a node)
THREADS = 2
_PROXY_VARS = ('http_proxy', 'https_proxy', 'ftp_proxy', 'all_proxy', 'no_proxy',
               'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'ALL_PROXY', 'NO_PROXY')


class GradeInfraError(Exception):
    pass


def build_script(prob: dict, k: int, codes: list[str], n_tests: int, test_cases: list[str], *, vendor: str,
                 h5: str) -> str:
    """codes[i] is the code of step i (model code, or gold for the skipped steps), i <= k."""
    parts = [f'import sys; sys.path.insert(0, {vendor!r})', prob['required_dependencies'] or '']
    parts += [codes[i] for i in range(k + 1)]
    parts += ['', 'from scicode.parse.parse import process_hdf5_to_tuple', '',
              f"targets = process_hdf5_to_tuple({prob['sub_steps'][k]['step_number']!r}, {n_tests}, {h5!r})"]
    for i, tc in enumerate(test_cases):
        parts += [f'target = targets[{i}]', '', tc]
    return '\n'.join(parts) + '\n'


def _argv(env: str, workdir: str, h5: Path, timeout: int) -> tuple[list[str], str, str]:
    """(argv, vendor path as the script sees it, h5 path as the script sees it)."""
    kind, _, target = env.partition(':')
    inner_env = (f'export HOME=/work MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS={THREADS} '
                 f'OPENBLAS_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS}; ')
    if kind == 'sif':
        inner = (f'ulimit -v {MEM_KB}; {inner_env} cd /work; '
                 f'exec timeout -k 10 {timeout} python -I /work/test.py')
        scrub = ['env', *(x for v in _PROXY_VARS for x in ('-u', v))]
        return ([*scrub, 'apptainer', '--silent', 'exec', '--containall', '--cleanenv', '--no-home',
                 '--net', '--network', 'none', '-B', f'{workdir}:/work', '-B', f'{VENDOR}:/opt/scicode_vendor:ro',
                 '-B', f'{h5}:/data/targets.h5:ro', '--pwd', '/work', target, 'bash', '-c', inner],
                '/opt/scicode_vendor', '/data/targets.h5')
    if kind == 'local':
        inner = (f'ulimit -v {MEM_KB}; {inner_env} cd {workdir}; '
                 f'exec timeout -k 10 {timeout} {target} -I {workdir}/test.py')
        return ['bash', '-c', inner], str(VENDOR), str(h5)
    raise ValueError(f'bad grading env {env!r} (sif:<path> or local:<python>)')


def run_one(script_fn, env: str, h5: Path, timeout: int, scratch: str | None = None) -> dict:
    """Grade in one env. script_fn(vendor, h5) -> script text. Raises GradeInfraError when the sandbox itself failed."""
    workdir = tempfile.mkdtemp(prefix='scicode_', dir=scratch)
    try:
        argv, vendor, h5_in = _argv(env, workdir, h5, timeout)
        Path(workdir, 'test.py').write_text(script_fn(vendor, h5_in))
        t0 = time.time()
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 120)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -9, (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or ''), 'outer timeout'
        secs = time.time() - t0
        if rc == 255 and 'FATAL:' in err and env.startswith('sif:'):
            raise GradeInfraError(f'apptainer failed: {err.strip()[-500:]}')
        return {'pass': rc == 0, 'rc': rc, 'timeout': rc in (124, 137, -9), 'secs': round(secs, 1),
                'tail': (out[-600:] + '\n' + err[-1500:]).strip()}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def grade_step(script_fn, envs: dict[str, str], h5: Path, timeout: int, *, all_envs: bool = True,
               scratch: str | None = None, infra_retries: int = 2) -> dict:
    """OR over envs. all_envs=True runs every env (the per-env breakdown); else stops at the first pass."""
    per_env = {}
    for label, env in envs.items():
        for attempt in range(infra_retries + 1):
            try:
                per_env[label] = run_one(script_fn, env, h5, timeout, scratch)
                break
            except GradeInfraError as e:
                if attempt == infra_retries:
                    raise
                time.sleep(5)
        if per_env[label]['pass'] and not all_envs:
            break
    return {'pass': any(v['pass'] for v in per_env.values()), 'per_env': per_env}


def selftest(envs: dict[str, str], scratch: str | None = None) -> dict:
    """Each env: imports work and the network is off (sif envs)."""
    probe = ('import socket, numpy, scipy, sympy, h5py\n'
             'try:\n    socket.create_connection(("10.22.0.107", 8080), timeout=3); print("NET")\n'
             'except OSError:\n    print("OFFLINE", numpy.__version__, scipy.__version__)\n')
    out = {}
    for label, env in envs.items():
        r = run_one(lambda v, h: probe, env, VENDOR / 'LICENSE', 120, scratch)
        out[label] = r['tail'].splitlines()[-1] if r['tail'] else f"rc={r['rc']}"
        if r['rc'] != 0 or (env.startswith('sif:') and not out[label].startswith('OFFLINE')):
            raise SystemExit(f'grading env {label} ({env}) self-test failed: rc={r["rc"]} {r["tail"][-500:]}\n'
                             'Contained Apptainer needs a compute node (run with --condor).')
    return out


def scratch_dir() -> str | None:
    d = os.environ.get('_CONDOR_SCRATCH_DIR')
    return d if d and os.path.isdir(d) else None

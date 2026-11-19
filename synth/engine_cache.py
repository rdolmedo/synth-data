"""Warm compile caches for vLLM engines: one snapshot per engine configuration on /data.

A Super engine spent ~14.5 of its 13.5-17 start-up minutes building what every earlier job had already built
(2026-10-24 hero, 24 jobs, ~6 B200-hours): FlashInfer JIT modules (fused_moe_trtllm_sm100 alone 8 min, plus
fp4_quantization, fmha_gen, sampling, the Mamba SSU kernel, CuTe-DSL GEMMs), FlashInfer's MoE autotune (90 s),
Triton kernels and torch.compile. Each job kept these caches on its own scratch, so each rebuilt them.

Caches stay node-local (Lustre has no flock, and a shared JIT dir is rebuilt by concurrent jobs: ninja races). A job
extracts the snapshot for its engine configuration into its engine's cache dir before starting vLLM; the first job
whose engine comes up healthy without one publishes it. FlashInfer's ninja build lines embed the cache and
CUDA_HOME paths and vLLM's compile hash includes the model path, so those paths must be identical in every job:
they live under /tmp, which HTCondor mounts per job (MOUNT_UNDER_SCRATCH), so /tmp/synth/... is the same path in
every job and private to it. Each cache validates its own entries (ninja freshness, content hashes), so a stale
snapshot costs a rebuild, never a wrong kernel.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .common import REPO, atomic, read, swap_path

SHARED = Path('/data/aai/tbench/synth/shared/engine-cache')
STABLE_ROOT = Path('/tmp/synth')
ENV_PREFIXES = ('VLLM_', 'FLASHINFER_', 'TORCH', 'TRITON_', 'NVCC', 'CUDA_HOME', 'HUMMING_', 'XDG_CACHE_HOME',
                'HF_HOME', 'PYTHONPATH', 'LD_LIBRARY_PATH')
LOCK_STALE_SECONDS = 7200
# Never shared (and emptied before every start, fresh_vllm_root): vLLM's cache root holds the FlashInfer MoE autotune
# results and torch.compile's AOT artifacts, and an engine that loads them runs 16-19% slower per step (2026-10-25
# replay: 5,837 vs 6,939 generated tok/s on the same GPU). FlashInfer JIT, Triton and inductor caches stay shared.
LOCAL_ONLY = ('./vllm_cache_root',)
MAX_SNAPSHOT_BYTES = 20 << 30


def job_private_tmp(scratch, mountinfo='/proc/self/mountinfo'):
    """True when /tmp is this job's own bind of <scratch>/tmp (HTCondor MOUNT_UNDER_SCRATCH)."""
    try:
        lines = Path(mountinfo).read_text().splitlines()
    except OSError:
        return False
    suffix = f'/{Path(scratch).name}/tmp'
    return any(len(f) > 4 and f[4] == '/tmp' and f[3].endswith(suffix) for f in (line.split() for line in lines))


def stable_root(scratch):
    """Where engine caches, CUDA_HOME and the staged checkpoint live: a fixed path when /tmp is per job."""
    return STABLE_ROOT if job_private_tmp(scratch) else Path(scratch)


def fresh_vllm_root(env, scratch):
    """Empty the engine's vLLM cache root before every start. An engine that loaded a root an earlier start had
    written (FlashInfer MoE autotune results, torch.compile AOT artifacts) ran ~16% slower per step, its own job's
    root as much as another job's (2026-10-25: 5,822 / 5,837 vs 6,939 generated tok/s, same GPU and load). Rebuilding
    it costs ~2 min per start. Only a root inside this engine's scratch is touched."""
    root = Path(env.get('VLLM_CACHE_ROOT') or '')
    if root.is_absolute() and Path(scratch) in root.parents:
        shutil.rmtree(root, ignore_errors=True)
        return True
    return False


# The API server's tokenize / chat-template thread pool (#451 ws4): no compiled kernel depends on it.
KEY_IGNORED_FLAGS = ('--renderer-num-workers',)
# Environment that cannot change compiled code: FLASHINFER_NO_DOWNLOAD only refuses to fetch a cubin the wheel lacks.
KEY_IGNORED_ENV = ('FLASHINFER_NO_DOWNLOAD',)


def config_key(argv, env, venv, gpu_name, repo=REPO):
    """Everything a compiled artifact can depend on: the engine command (ports abstracted), the compile-relevant
    environment, the venv's installed distributions and the GPU model. Every `--*-port` value is abstracted, not just
    `--port`: a dp_external_lb engine's random DP RPC and supervisor ports made GLM's key new on every start, so it
    never seeded and compiled cold each time (#378: 980 s and 21 cores at start-up, c30 and ss14 keys differed).
    So is the code checkout `repo` ({repo} in a spec; a job runs a frozen copy per commit, worktrees/tasksynth-run-
    <sha>): the Snowball family's PYTHONPATH (plugin) and parser paths named it, so every commit made a new key and
    those servers never seeded across commits. What the code compiles to is checked by each cache itself (content
    hashes, ninja freshness): a snapshot from an older commit costs a rebuild of what changed, never a wrong kernel.
    publish never replaces a snapshot, so the first commit's serves later ones: a kernel a later plugin change alters
    (today only GRUG_TRITON_ROUTING's Triton kernel) is rebuilt at every start until the key changes."""
    root = str(repo or '')
    unrepo = (lambda x: swap_path(x, root, '{repo}')) if root else (lambda x: x)
    args = [unrepo(a) for a in argv]
    for i, a in enumerate(args[:-1]):
        if a == '--port' or (a.startswith('--') and a.endswith('-port')):
            args[i + 1] = '{port}'
    for flag in KEY_IGNORED_FLAGS:   # flags that cannot change compiled code: an existing snapshot still applies
        while flag in args[:-1]:
            i = args.index(flag)
            del args[i:i + 2]
    dists = sorted(p.name for p in Path(venv).glob('lib/python3*/site-packages/*.dist-info'))
    facts = {'argv': args, 'env': {k: unrepo(v) for k, v in sorted(env.items())
                                   if k.startswith(ENV_PREFIXES) and k not in KEY_IGNORED_ENV},
             'dists': dists, 'gpu': gpu_name}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:20], facts


def seed(cache_dir, key, shared=SHARED):
    """Extract the snapshot into a fresh cache dir. Returns a record of what happened (never raises)."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / '.synth-engine-cache.json'
    if marker.exists():
        return {**read(marker, {}), 'seeded': False, 'reason': 'already seeded in this job'}
    tarball = shared / f'{key}.tar'
    if not tarball.exists():
        return {'seeded': False, 'reason': 'no snapshot', 'key': key}
    started = time.time()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xf', str(tarball), '-C', str(cache_dir), *(f'--exclude={x}' for x in LOCAL_ONLY)],
                       check=True, capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(cache_dir, ignore_errors=True)  # cold start rather than a half-extracted cache
        return {'seeded': False, 'reason': f'extract failed: {exc!r}'[:500], 'key': key}
    record = {'seeded': True, 'key': key, 'seconds': round(time.time() - started, 1),
              'bytes': tarball.stat().st_size}
    atomic(marker, record)
    return record


def seed_first(cache_dir, key, fallbacks=(), shared=SHARED):
    """seed() from the engine's own key, else from the first fallback snapshot that exists (an older recipe's,
    another spec's of the same model, the /fast-venv recipe of a staged engine): a first start without its own
    snapshot then reuses what still applies instead of compiling cold. Each cache validates its own entries, so a
    fallback costs a rebuild of what differs, never a wrong kernel. The engine still publishes under its own key once
    healthy. The record names the snapshot that seeded (`key`), the engine's own key (`own_key`) when they differ, and
    every key tried with why it did not seed (`tried`: a corrupt own snapshot shows there, "extract failed", even when
    a fallback then seeds)."""
    record = seed(cache_dir, key, shared)
    attempts = [(key, record)]
    for fallback in [f for f in dict.fromkeys(fallbacks) if f and f != key]:
        if record.get('seeded') or record.get('reason') == 'already seeded in this job':
            break
        record = seed(cache_dir, fallback, shared)
        attempts.append((fallback, record))
    if len(attempts) == 1:
        return record
    return {**record, 'own_key': key,
            'tried': [{'key': k, 'reason': r.get('reason')} for k, r in attempts if not r.get('seeded')]}


def publish(cache_dir, key, facts, local_tmp, source, shared=SHARED):
    """Snapshot a healthy engine's cache dir once per key (mkdir mutex on /data; tar on local disk, then dd)."""
    tarball = shared / f'{key}.tar'
    if tarball.exists():
        return {'published': False, 'reason': 'exists'}
    shared.mkdir(parents=True, exist_ok=True)
    lock = shared / f'{key}.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime < LOCK_STALE_SECONDS:
                return {'published': False, 'reason': 'another job is publishing'}
            lock.rmdir()
            lock.mkdir()
        except OSError:
            return {'published': False, 'reason': 'lock contention'}
    local = Path(local_tmp) / f'engine-cache-{key}.tar'
    partial = shared / f'.{key}.tar.{os.getpid()}.tmp'
    started = time.time()
    try:
        size = sum(p.stat().st_size for p in Path(cache_dir).rglob('*') if p.is_file() and not p.is_symlink())
        if size > MAX_SNAPSHOT_BYTES or shutil.disk_usage(local_tmp).free < 2 * size:
            return {'published': False, 'reason': f'{size} bytes: over the cap or the job disk is short'}
        subprocess.run(['tar', '-cf', str(local), '-C', str(cache_dir), '--exclude=./.synth-engine-cache.json',
                        *(f'--exclude={x}' for x in LOCAL_ONLY), '.'], check=True, capture_output=True, timeout=1800)
        subprocess.run(['dd', f'if={local}', f'of={partial}', 'bs=64M', 'oflag=direct', 'status=none'],
                       check=True, capture_output=True, timeout=1800)
        if partial.stat().st_size != local.stat().st_size:
            raise OSError(f'snapshot copy size mismatch: {partial}')
        partial.replace(tarball)
        record = {'published': True, 'key': key, 'bytes': tarball.stat().st_size,
                  'seconds': round(time.time() - started, 1), 'source': source, 'time': time.time(), 'facts': facts}
        atomic(shared / f'{key}.json', record)
        return {k: v for k, v in record.items() if k != 'facts'}
    except (OSError, subprocess.SubprocessError) as exc:
        partial.unlink(missing_ok=True)
        return {'published': False, 'reason': f'{exc!r}'[:500]}
    finally:
        local.unlink(missing_ok=True)
        try:
            lock.rmdir()
        except OSError:
            pass

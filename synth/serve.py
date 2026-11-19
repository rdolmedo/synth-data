"""GPU-job supervisor: own every vLLM replica inside one serving allocation.

  python -m synth.serve --index J --generation G      (SYNTH_CAMPAIGN=<ops/campaign.json>)

* preflight every assigned GPU, then stage the checkpoint to node-local scratch;
* mode "replicas": one `vllm serve` per gpus_per_replica GPUs; mode "dp_external_lb":
  one data-parallel process whose ranks listen on consecutive ports;
* publish ops/serving/job-J/endpoints.json (one healthy record per rank/replica)
  and per-backend Prometheus samples under ops/metrics/backends/;
* ops/serving/control.json (tunables + generation) restarts replicas in place;
  ops/serving/RELEASE stops everything. Control-plane write failures are logged
  and retried, never allowed to stop a healthy engine.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import time
import zipfile

from . import engine_cache, numerics
from .common import append_jsonl, atomic, best_effort, read, sha
from .config import Campaign
from .gpu import assigned, expand, preflight
from .probe import free_port, probe, request, scrape, server_summary
from tbshared import reslog
from .vllm_launch import describe, replica_command, replica_env

LOG_SNAPSHOT_SECONDS = 60
METRICS_SECONDS = 15
KV_TOKENS = re.compile(r'GPU KV cache size: ([\d,]+) tokens')
KV_BLOCK = re.compile(r'Setting attention block size to (\d+) tokens')
OFFLOAD_BLOCKS = re.compile(r'SimpleCPUOffloadWorker \[CPU\]: \d+ tensors, (\d+) CPU blocks')


def kv_cache_geometry(log_text):
    """(KV capacity in tokens per engine core, KV block size in tokens) from a vLLM startup log.
    Hybrid models log their enlarged attention block; others use vLLM's default 16."""
    sizes = [int(m.replace(',', '')) for m in KV_TOKENS.findall(log_text)]
    blocks = [int(m) for m in KV_BLOCK.findall(log_text)]
    return (min(sizes) if sizes else None), (max(blocks) if blocks else 16)


def kv_offload_tokens(log_text, block):
    """Tokens the CPU KV tier holds per engine core (native SimpleCPUOffloadConnector; CPU blocks are GPU-sized)."""
    blocks = [int(m) for m in OFFLOAD_BLOCKS.findall(log_text)]
    return min(blocks) * block if blocks and block else 0


def stop_process(proc, timeout=60):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def dp_groups(gpus, serving):
    """The DP engines of a dp_* serving job: all its GPUs, or `dp_group_gpus` GPUs each (2x DP2 in a 4-GPU job: a
    smaller lockstep group per engine, less KV per GPU than one DP4/EP4 engine)."""
    n = serving.get('dp_group_gpus') or len(gpus)
    if len(gpus) % n:
        raise ValueError(f'dp_group_gpus={n} does not divide the {len(gpus)} GPUs of the job')
    return [gpus[i:i + n] for i in range(0, len(gpus), n)]


def consecutive_ports(n):
    # Below the kernel's ephemeral range (32768-60999 on the cluster): a port picked there can be taken by any outgoing
    # connection before vLLM binds it (2026-10-30: an engine start failed with EADDRINUSE beside two replay clients).
    for _ in range(200):
        base = random.randint(20000, 32767 - n)
        sockets = []
        try:
            for port in range(base, base + n):
                s = socket.socket()
                s.bind(('', port))
                sockets.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in sockets:
                s.close()
    raise RuntimeError('No consecutive free port block')


class Engine:
    """One vLLM process; serves one replica or all ranks of a DP group."""

    def __init__(self, sup, name, gpus, slot=None):
        self.sup, self.name, self.gpus = sup, name, gpus
        self.slot = slot  # the engine's place in its job (r0, r1, ...): its cache paths are the same in every job
        self.proc = None
        self.failures = []
        self.ranks = []
        self.healthy_at = None
        self.probe_future = None
        self.failed = None
        self.kv_geometry = (None, None)
        self.kv_offload = 0

    def start(self):
        teacher, protocol = self.sup.teacher, self.sup.campaign.protocol
        mode = teacher['serving'].get('mode', 'replicas')
        if mode == 'dp_internal':
            # One API server; vLLM's DP ranks are pinned per attempt by X-data-parallel-rank.
            port = free_port()
            extra = {'dp_size': len(self.gpus) // teacher['serving'].get('gpus_per_replica', 1)}
            self.ranks = [{'name': self.name, 'port': port, 'dp_size': extra['dp_size']}]
        elif mode == 'dp_external_lb':
            n = len(self.gpus) // teacher['serving'].get('gpus_per_replica', 1)
            base = consecutive_ports(n)
            extra = {'dp_size': n, 'rpc_port': consecutive_ports(1), 'supervisor_port': consecutive_ports(1)}
            self.ranks = [{'name': f'{self.name}-rank{i}', 'port': base + i} for i in range(n)]
            port = base
        else:
            port = free_port()
            extra = {}
            self.ranks = [{'name': self.name, 'port': port}]
        arm = f'-{self.sup.control["arm"]}-r{self.sup.restarts}' if self.sup.control.get('arm') else ''
        self.run = self.sup.home / f'g{self.sup.generation}' / f'{self.name}-c{self.sup.control["generation"]}{arm}-{len(self.failures)}'
        self.run.mkdir(parents=True, exist_ok=True)
        scratch = self.sup.local / 'engines' / (self.slot or self.name)
        argv = replica_command(teacher, protocol, port=port, model_dir=str(self.sup.model_dir),
                               tunables=self.sup.tunables, rank_ports={**extra, **self.sup.staged})
        env = replica_env(teacher, gpus=self.gpus, scratch=scratch, tunables=self.sup.tunables)
        if self.sup.kernel_provider:
            env['PYTHONPATH'] = ':'.join(filter(None, [str(self.sup.kernel_provider), env.get('PYTHONPATH')]))
        self.cache_dir = scratch / 'cache'
        self.cache_key, self.cache_facts = engine_cache.config_key(
            argv, env, self.sup.tunables.get('venv') or teacher['runtime']['venv'], self.sup.gpu_name)
        # its own snapshot, else the campaign's serving.seed_from, else the spec's [serving] seed_from: a new campaign's
        # first wave seeds from a snapshot of the same model instead of compiling cold per engine (vLLM main's first
        # Super start JIT-compiles FlashInfer modules and NVFP4 CuTe-DSL GEMMs, ~11 min)
        self.cache_seed = (engine_cache.seed_first(self.cache_dir, self.cache_key, self.sup.seed_from)
                           if self.sup.engine_cache else {'seeded': False, 'reason': 'disabled'})
        engine_cache.fresh_vllm_root(env, scratch)
        self.local_log = scratch / f'vllm-{time.time_ns()}.log'
        atomic(self.run / 'launch.json', {**describe(argv, env), 'gpus': self.gpus, 'ranks': self.ranks,
            'control': self.sup.control, 'host': self.sup.host, 'started_at': time.time(),
            'model': teacher['model'], 'revision': teacher['revision'], 'local_log': str(self.local_log),
            'engine_cache': self.cache_seed,
            # effective numerics after the campaign's overrides and tunables, with a stable hash (synth/numerics.py)
            'numerics': numerics.safe_replica_numerics(
                teacher, argv, env, model_dir=str(self.sup.model_dir), scratch_roots=(self.sup.local, scratch),
                protocol=protocol, venv=self.sup.tunables.get('venv') or teacher['runtime']['venv'],
                declared_version=self.sup.tunables.get('vllm_version'), gpu=self.sup.gpu_name, base_env=os.environ)})
        self.log_handle = self.local_log.open('ab')
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.started_at = time.time()
        self.healthy_at = None
        self.probe_future = None
        self.last_log_copy = 0
        self.metrics_prev = {}
        seeded = (f'; compile caches seeded from snapshot {self.cache_seed["key"]} in {self.cache_seed["seconds"]}s'
                  + (f' (own key {self.cache_key} had none)' if self.cache_seed.get('own_key') else '')
                  if self.cache_seed.get('seeded') else '')
        self.sup.campaign.log(f'{self.name}: started vLLM on {self.sup.host} GPUs {self.gpus} ports '
                              f'{[r["port"] for r in self.ranks]}{seeded}')

    def stop(self):
        stop_process(self.proc)
        self.copy_log(force=True)
        self.healthy_at = None

    def copy_log(self, force=False):
        """Snapshot the node-local vLLM log to /data. Periodic copies run in the background: a slow /data
        write must never stall the loop that publishes endpoints.json (controllers stop admitting to a
        backend whose file is 120 s old)."""
        if force:
            best_effort(shutil.copyfile, self.local_log, self.run / 'vllm.log')
        elif time.time() - self.last_log_copy >= LOG_SNAPSHOT_SECONDS and (
                getattr(self, 'log_copy', None) is None or self.log_copy.done()):
            self.last_log_copy = time.time()
            self.log_copy = self.sup.pool.submit(best_effort, shutil.copyfile, self.local_log, self.run / 'vllm.log')

    def urls(self, rank):
        return f'http://127.0.0.1:{rank["port"]}/v1', f'http://{self.sup.host}:{rank["port"]}/v1'

    def poll(self):
        """Returns False when the engine ended (caller decides on restart)."""
        if self.proc.poll() is not None:
            return False
        self.copy_log()
        if self.healthy_at is None:
            if self.probe_future is None:
                try:
                    for rank in self.ranks:
                        request(self.urls(rank)[0] + '/models', timeout=3)
                    self.probe_future = self.sup.pool.submit(
                        lambda: [probe(self.urls(r)[0], self.sup.teacher) for r in self.ranks])
                except OSError:
                    pass
            elif self.probe_future.done():
                result = self.probe_future.result()  # a failed probe raises: wrong parsers/template
                best_effort(atomic, self.run / 'probe.json', result)
                try:
                    text = self.local_log.read_text(errors='replace')
                    self.kv_geometry = kv_cache_geometry(text)
                    self.kv_offload = kv_offload_tokens(text, self.kv_geometry[1])
                except OSError:
                    pass
                self.healthy_at = time.time()
                self.sup.campaign.log(f'{self.name}: healthy after {self.healthy_at - self.started_at:.0f}s')
                if self.sup.engine_cache:
                    self.sup.pool.submit(self.publish_cache)
            if self.healthy_at is None and time.time() - self.started_at > self.sup.teacher['serving'].get('startup_timeout_sec', 5400):
                raise TimeoutError(f'{self.name}: vLLM startup exceeded the configured limit')
            return True
        now = time.time()
        for rank in self.ranks:
            local = self.urls(rank)[0].removesuffix('/v1')
            last = self.metrics_prev.get(rank['name'])
            if last and now - last[0] < METRICS_SECONDS:
                continue
            current = scrape(local)
            if not current:
                continue
            if rank.get('dp_size', 1) > 1:  # gauges summed over engines; admission wants per-rank means
                current['kv'] /= rank['dp_size']
            sample = {'timestamp': now, 'backend': rank['name'], 'raw': current, 'control': self.sup.control}
            if last:
                sample.update(server_summary([last[1]], [current], now - last[0]))
            self.metrics_prev[rank['name']] = (now, current)
            best_effort(atomic, self.sup.campaign.ops / 'metrics' / 'backends' / f'{rank["name"]}.json', sample)
            best_effort(append_jsonl, self.sup.campaign.ops / 'metrics' / f'samples-{self.sup.index}.jsonl', sample)
        return True

    def publish_cache(self):
        """First healthy engine of a configuration: snapshot its compile caches for later jobs (background)."""
        result = engine_cache.publish(self.cache_dir, self.cache_key, self.cache_facts, self.sup.scratch,
                                      {'host': self.sup.host, 'job_id': self.sup.job_id, 'engine': self.name})
        if result.get('published') or result.get('reason') not in ('exists', 'another job is publishing'):
            self.sup.campaign.log(f'{self.name}: engine cache snapshot {self.cache_key}: {result}')

    def records(self):
        phase = 'failed' if self.failed else 'healthy' if self.healthy_at else 'starting'
        tokens, block = self.kv_geometry
        return [{'name': r['name'], 'url': self.urls(r)[1], 'port': r['port'], 'phase': phase, 'engine': self.name,
                 'dp_size': r.get('dp_size', 1),
                 # KV capacity behind this backend (all ranks of an internal-DP backend), for admission.
                 'kv_capacity_tokens': tokens * r.get('dp_size', 1) if tokens else None, 'kv_block_tokens': block,
                 'kv_offload_tokens': self.kv_offload * r.get('dp_size', 1),
                 'gpus': self.gpus, 'started_at': self.started_at, 'healthy_at': self.healthy_at,
                 'failures': len(self.failures), 'control_generation': self.sup.control['generation']}
                for r in self.ranks]


def gpu_name(report, gpus):
    """The assigned GPUs' model from the preflight's nvidia-smi rows (index, uuid, name, ...)."""
    rows = [[f.strip() for f in line.split(',')] for line in report.get('nvidia_smi', {}).get('stdout', '').splitlines()]
    return sorted({r[2] for r in rows if len(r) > 2 and (r[0] in gpus or r[1] in gpus)}) or None


def stage_teacher(teacher, local):
    """A teacher's checkpoint (unless stage_checkpoint = false), its serving.stage_extra model dirs (draft models) and
    its serving.kernel_wheels on node-local scratch `local` -> (model dir, {'stage_<name>': staged path} for the
    command's rank_ports, the unpacked kernel-wheel dir to put first on PYTHONPATH or None). Used by campaign serving
    (Supervisor.stage) and serve_teacher, so a spec launches the same way under both (#445: nemotron35-bolt's
    {stage_draft})."""
    serving = teacher['serving']
    sources = [('model', teacher['checkpoint'])] if teacher.get('stage_checkpoint', True) else []
    sources += [(x['name'], x['source']) for x in serving.get('stage_extra', [])]
    staged = {}
    for name, source in sources:
        source, target = Path(source), Path(local) / name
        shutil.copytree(source, target, dirs_exist_ok=True)
        for path in source.rglob('*'):
            if path.is_file() and (target / path.relative_to(source)).stat().st_size != path.stat().st_size:
                raise RuntimeError(f'Staged size mismatch: {path}')
        staged[f'stage_{name}'] = str(target)
    kernel_provider = None
    for wheel in serving.get('kernel_wheels', []):
        dest = Path(local) / Path(wheel['path']).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel['path'], dest)
        if sha(dest) != wheel['sha256']:
            raise RuntimeError(f'Kernel wheel checksum mismatch: {wheel["path"]}')
        kernel_provider = Path(local) / 'kernel-provider'
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(kernel_provider)
    model_dir = Path(staged['stage_model']) if 'stage_model' in staged else Path(teacher['checkpoint'])
    return model_dir, staged, kernel_provider


def seed_fallbacks(config, teacher) -> list:
    """Snapshots an engine without its own seeds from, in order: the campaign's serving.seed_from, then the spec's
    [serving] seed_from (a key, or a list of keys)."""
    out = []
    for value in (config.get('serving', {}).get('seed_from'), teacher.get('serving', {}).get('seed_from')):
        out += [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
    out = [k for k in out if isinstance(k, str) and k]   # anything else (a number by mistake) is ignored, not fatal
    return [k for i, k in enumerate(out) if k not in out[:i]]


class Supervisor:
    def __init__(self, index, generation):
        self.campaign = Campaign.load()
        self.teacher = self.campaign.teacher
        self.index, self.generation = index, generation
        self.home = self.campaign.ops / 'serving' / f'job-{index}'
        self.home.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(os.environ['_CONDOR_SCRATCH_DIR'])
        # Checkpoint, CUDA_HOME and compile caches at paths that are the same in every job (synth.engine_cache).
        self.local = engine_cache.stable_root(self.scratch)
        # Snapshots only where the paths are the same in every job (else each job would key a new one).
        self.engine_cache = (self.campaign.config.get('serving', {}).get('engine_cache', True)
                             and self.local == engine_cache.STABLE_ROOT)
        self.seed_from = seed_fallbacks(self.campaign.config, self.teacher)   # engine_cache.seed_first
        self.gpu_name = None
        self.job_id = os.environ.get('SYNTH_JOB_ID')
        self.host = socket.getfqdn()
        self.stopping = False
        self.restarts = 0  # control changes so far (names each A/B arm's run directory)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def read_control(self):
        """ops/serving/control.json (or the teacher's tunables), then this job's ops/serving/job-J/control.json
        on top: one arm of a serving A/B per job. The per-job file never sets `generation` (the controller
        only admits to replicas at the campaign-wide generation); any change to it restarts this job's engines."""
        default = {'generation': 0, **self.teacher['serving'].get('tunables', {})}
        control = read(self.campaign.ops / 'serving' / 'control.json', default)
        job = {k: v for k, v in read(self.home / 'control.json', {}).items() if k != 'generation'}
        return {**control, **job}

    @property
    def tunables(self):
        return {k: v for k, v in self.control.items() if k != 'generation'}

    def stage(self):
        """Checkpoint (+ extra model dirs such as draft models) and kernel wheels on node-local scratch."""
        started = time.time()
        model_dir, self.staged, self.kernel_provider = stage_teacher(self.teacher, self.local)
        best_effort(atomic, self.home / f'staging-g{self.generation}.json', {'staged': self.staged,
            'kernel_wheels': self.teacher['serving'].get('kernel_wheels', []), 'seconds': time.time() - started,
            'host': self.host})
        return model_dir

    def publish(self):
        records = [r for e in self.engines for r in e.records()]
        best_effort(atomic, self.home / 'endpoints.json', {'job_id': self.job_id, 'index': self.index,
            'generation': self.generation, 'host': self.host, 'updated_at': time.time(),
            'control': self.control, 'replicas': records})

    def main(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'stopping', True))
        gpus = expand(assigned())
        passed, report = preflight(gpus)
        atomic(self.home / f'preflight-g{self.generation}.json', {**report, 'host': self.host,
            'job_id': self.job_id, 'passed': passed, 'time': time.time()})
        if not passed:
            raise RuntimeError(f'GPU preflight failed on {self.host}; see preflight-g{self.generation}.json')
        self.gpu_name = gpu_name(report, gpus)
        # cgroup CPU/memory every minute (CPU KV tier, staging page cache), for sizing the request (#242)
        reslog.start(self.home / f'resources-g{self.generation}.jsonl', 60)   # tbshared's one sampler (#445 item 2)
        self.model_dir = self.stage()
        per = self.teacher['serving'].get('gpus_per_replica', 1)
        if self.teacher['serving'].get('mode', 'replicas') in ('dp_external_lb', 'dp_internal'):
            groups = dp_groups(gpus, self.teacher['serving'])
        else:
            if len(gpus) % per:
                raise ValueError(f'{len(gpus)} GPUs do not divide into {per}-GPU replicas')
            groups = [gpus[i:i + per] for i in range(0, len(gpus), per)]
        self.control = self.read_control()
        self.engines = [Engine(self, f'j{self.index}r{k}', g, slot=f'r{k}') for k, g in enumerate(groups)]
        for engine in self.engines:
            engine.start()
        while not self.stopping:
            if (self.campaign.ops / 'serving' / 'RELEASE').exists():
                self.campaign.log(f'Serving job {self.index}: RELEASE requested')
                break
            desired = self.read_control()
            if desired != self.control:
                self.campaign.log(f'Serving job {self.index}: control changed {self.control} -> {desired}; restarting engines')
                for engine in self.engines:
                    engine.stop()
                self.control = desired
                self.restarts += 1
                for engine in self.engines:
                    engine.failures = []
                    engine.failed = None
                    engine.start()
            for engine in self.engines:
                if engine.failed:
                    continue
                if not engine.poll():
                    code = engine.proc.returncode
                    engine.copy_log(force=True)
                    engine.failures.append({'time': time.time(), 'returncode': code, 'run': str(engine.run)})
                    best_effort(atomic, self.home / f'{engine.name}-failures.json', engine.failures)
                    recent = [f for f in engine.failures if time.time() - f['time'] < 1800]
                    if len(recent) >= 3:
                        engine.failed = 'three exits within 30 minutes'
                        self.campaign.log(f'{engine.name}: giving up after repeated exits; inspect {engine.run}')
                    else:
                        self.campaign.log(f'{engine.name}: vLLM exited {code}; restarting in place')
                        engine.start()
            self.publish()
            if all(e.failed for e in self.engines):
                raise RuntimeError('Every engine in this serving job failed')
            time.sleep(5)
        for engine in self.engines:
            engine.stop()
        self.publish()



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--generation', type=int, default=0)
    a = ap.parse_args()
    supervisor = Supervisor(a.index, a.generation)
    try:
        supervisor.main()
    except Exception as exc:
        best_effort(atomic, supervisor.home / f'needs-attention-g{a.generation}.json', {'time': time.time(),
            'error': repr(exc), 'host': supervisor.host})
        raise

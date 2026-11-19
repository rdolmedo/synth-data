#!/usr/bin/env python3
"""CPU worker (one HTCondor job): claims attempts of a run and runs each as one Harbor trial of mini-swe-agent in a
harbor_ns sandbox on this node.

  python -m aa_evals.common.harbor_mswea.worker --run RUN [--cpus N --mem-gb M]

Resources. Every trial gets its task's declared CPUs as its own cores of this job (Harbor's `--ek
ns_cpu_affinity=true` pins the sandbox to the cores its process may use, and the worker gives each Harbor process a
disjoint core set: harbor_ns' own counter restarts at 0 in every process). A trial starts only when its cores and its
declared memory are free on this worker.

Outputs. A trial runs in node-local scratch. Once classified (classify.py), its directory is packed into
RUN/trials/<key>.tar.gz, mini's trajectory is kept as RUN/transcripts/<key>.json.gz, and the outcome goes to
RUN/queue/done/<key>.json (no per-step files on /data). Infrastructure failures get another try (queue.py).

Stops: RUN/STOP_WORKERS (no new claims, exit when idle); SIGTERM (running trials killed and retried).
When the run is complete the last worker writes items.jsonl and summary.json (score.py).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import shutil
import signal
import socket
import tarfile
import time
import tomllib
from pathlib import Path

from aa_evals.common import diskcap
from . import endpoint, score
from .classify import TRAJ, classify, find_trial
from .queue import Queue, atomic, now, parse_key, read

HARBOR = os.environ.get('AA_HARBOR', '/opt/aai/venvs/harbor/bin/harbor')
SCRATCH = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or f"/tmp/{os.environ.get('USER', 'u')}/aa_mswea")
STOP = False


def log(msg: str) -> None:
    print(f'{now()} {msg}', flush=True)


CLK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


def tree_usage(root: int) -> tuple[int, float]:
    """(RSS bytes, CPU seconds incl. reaped children) of a process and all its descendants (sandboxes included:
    harbor_ns starts them in new sessions, so the tree is followed by parent pid, not by session)."""
    kids: dict[int, list[int]] = {}
    stats: dict[int, list[str]] = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                st = f.read().rsplit(')', 1)[1].split()
        except OSError:
            continue
        stats[int(d)] = st
        kids.setdefault(int(st[1]), []).append(int(d))
    rss, cpu, todo = 0, 0.0, [root]
    while todo:
        pid = todo.pop()
        st = stats.get(pid)
        if st is None:
            continue
        cpu += sum(int(x) for x in st[11:15]) / CLK          # utime stime cutime cstime
        rss += int(st[21]) * PAGE
        todo.extend(kids.get(pid, []))
    return rss, cpu


def task_resources(task_dir: Path) -> tuple[int, int, float]:
    """(cpus, memory_mb, agent timeout sec) a trial of this task needs: the larger of agent and verifier env."""
    d = tomllib.loads((task_dir / 'task.toml').read_text())
    e = d.get('environment') or {}
    v = (d.get('verifier') or {}).get('environment') or {}
    cpus = max(int(e.get('cpus') or 1), int(v.get('cpus') or 1))
    mem = max(int(e.get('memory_mb') or 2048), int(v.get('memory_mb') or 2048))
    return cpus, mem, float((d.get('agent') or {}).get('timeout_sec') or 0)


def stage_tasks(src: str, names: list[str], dst: Path) -> None:
    """Copy the run's tasks to node-local scratch: from a directory of task dirs, or a tar with tasks/<slug>/."""
    if dst.exists():
        return
    tmp = dst.with_name(dst.name + '.stage')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    want = set(names)
    p = Path(src)
    if p.is_dir():
        for n in names:
            shutil.copytree(p / n, tmp / n, symlinks=True)
    else:
        with tarfile.open(p) as tf:
            members = [m for m in tf.getmembers()
                       if len(Path(m.name).parts) >= 2 and Path(m.name).parts[0] == 'tasks'
                       and Path(m.name).parts[1] in want]
            tf.extractall(tmp / '_x', members=members, filter='data')
        for n in names:
            os.rename(tmp / '_x' / 'tasks' / n, tmp / n)
        shutil.rmtree(tmp / '_x', ignore_errors=True)
    os.rename(tmp, dst)


class Worker:
    def __init__(self, a):
        self.run = Path(a.run)
        self.q = Queue(self.run)
        self.cfg = read(self.run / 'config.json')
        self.name = f"{os.environ.get('CONDOR_JOB_ID', 'local')}-{socket.gethostname()}-{os.getpid()}"
        allowed = sorted(os.sched_getaffinity(0))
        self.cores_free = set(allowed[:a.cpus] if a.cpus else allowed)
        self.mem_free = int(a.mem_gb * 1024)
        self.running: dict[str, dict] = {}
        self.tasks_dir = SCRATCH / 'tasks'
        self.res: dict[str, tuple[int, int, float]] = {}
        self.last_recover = 0.0

    # ---- resources
    def fits(self, task: str) -> bool:
        cpus, mem, _ = self.res[task]
        return cpus <= len(self.cores_free) and mem <= self.mem_free

    def take(self, task: str) -> list[int]:
        cpus, mem, _ = self.res[task]
        cores = sorted(self.cores_free)[:cpus]
        self.cores_free -= set(cores)
        self.mem_free -= mem
        return cores

    def give(self, task: str, cores: list[int]) -> None:
        self.cores_free |= set(cores)
        self.mem_free += self.res[task][1]

    # ---- one attempt
    def harbor_cmd(self, k: str, task_dir: Path, out: Path, api_base: str) -> list[str]:
        c = self.cfg
        base = [HARBOR, 'run', '-p', str(task_dir), '-o', str(out), '--job-name', k, '-n', '1', '-k', '1', '-y', '-q',
                '--env', c['env_class'], '--ek', 'ns_cpu_affinity=true',
                '--agent-setup-timeout-multiplier', '3', '--environment-build-timeout-multiplier', '4']
        if c.get('agent', 'mini') != 'mini':   # oracle / nop task validation
            return base + ['-a', c['agent']] + list(c.get('harbor_extra_args') or [])
        mini = json.loads(json.dumps(c['mini_config']))
        mini['model']['model_kwargs']['api_base'] = api_base
        cmd = base + [
               '-a', 'aa_evals.common.harbor_mswea.agent:MiniSweAgentAA', '-m', f"hosted_vllm/{c['served']}",
               '--ak', f"version={c['mini_version']}", '--ak', 'config=' + json.dumps(mini),
               '--ae', 'MSWEA_API_KEY=sk-aa-local-dummy',
               '--ae', f"MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT={c['api_retries']}",
               '--agent-timeout-multiplier', f"{c['backstop_multiplier']:g}"]
        return cmd + list(c.get('harbor_extra_args') or [])

    async def run_item(self, k: str, cores: list[int]) -> None:
        task, repeat, attempt = parse_key(k)
        raw = SCRATCH / 'trials' / k
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        api_base, ip = '', ''
        if self.cfg.get('agent', 'mini') == 'mini':
            ok, ep, why = endpoint.alive(self.cfg['model'])
            urls = (ep or {}).get('urls') or self.cfg['urls']
            url = urls[int(hashlib.sha256(f'{task}/{repeat}'.encode()).hexdigest(), 16) % len(urls)]
            api_base, ip = endpoint.ip_url(url)
        env = {**os.environ,
               'PYTHONPATH': self.cfg['code'] + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
               'HARBOR_NS_LOCAL': str(SCRATCH / 'harbor_ns'), 'HARBOR_NS_SHARED': self.cfg['image_cache'],
               'HARBOR_NS_NO_PROXY': ip, 'TMPDIR': str(SCRATCH / 'tmp'),
               'HARNESS_AGENT_CACHE_LOCAL': str(SCRATCH / 'agent_cache'), 'AA_MSWEA_CACHE': self.cfg['bundle_root'],
               'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1'}
        diskcap.env(env, self.cfg, raw / 'diskcap.json', task)   # #405: run config "diskcap" (on unless null)
        env['no_proxy'] = env['NO_PROXY'] = ','.join(
            x for x in ['localhost', '127.0.0.1', 'localnet', '.internal.cluster.internal', ip] if x)
        (SCRATCH / 'tmp').mkdir(parents=True, exist_ok=True)
        cmd = self.harbor_cmd(k, self.tasks_dir / task, raw, api_base)
        info = {'key': k, 'task': task, 'repeat': repeat, 'try': attempt, 'worker': self.name,
                'host': socket.gethostname(), 'cores': cores, 'endpoint': api_base, 'started_at': now(),
                'diskcap': diskcap.config(self.cfg, task)}   # #405: the effective cap (None: off)
        t0 = time.time()
        stopped: bool | str = False
        with (raw / 'harbor.log').open('ab', buffering=0) as output:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=output, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True,
                preexec_fn=lambda: os.sched_setaffinity(0, cores))
            self.running[k]['proc'] = proc
            peak_rss, cpu_sec = 0, 0.0
            while proc.returncode is None:
                self.q.heartbeat(k)
                try:
                    rss, cpu = await asyncio.to_thread(tree_usage, proc.pid)
                    peak_rss, cpu_sec = max(peak_rss, rss), max(cpu_sec, cpu)
                except OSError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 60)
                except asyncio.TimeoutError:
                    pass
                if STOP and proc.returncode is None:
                    stopped = 'worker_stopped'
                    await self.kill(proc)
        limit = (self.res[task][2] or None) if self.cfg.get('agent', 'mini') == 'mini' else None
        outcome = classify(raw, proc.returncode, stopped, time_limit_sec=limit,
                           max_model_len=self.cfg.get('max_model_len'))
        outcome.update(info, wall_sec=round(time.time() - t0, 1), time_limit_sec=limit,
                       usage={'peak_rss_gb': round(peak_rss / 2**30, 2), 'cpu_sec': round(cpu_sec),
                              'cores_reserved': len(cores), 'sampled_every_sec': 60},
                       **diskcap.outcome_fields(raw / 'diskcap.json'))   # the box's peak, largest file, kills
        trial = find_trial(raw)
        try:
            outcome.update(self.pack(k, raw, trial))
        except OSError as e:
            outcome['pack_error'] = repr(e)
        self.q.finish(k, outcome)
        shutil.rmtree(raw, ignore_errors=True)
        log(f"done {k}: score={outcome.get('score')} infra={outcome.get('infrastructure_reason')} "
            f"exit={outcome.get('exit_status')} exc={outcome.get('exception_type')} steps={outcome.get('api_calls')} "
            f"{outcome['wall_sec']:.0f}s")

    def pack(self, k: str, raw: Path, trial: Path | None) -> dict:
        out = {}
        tdir = self.run / 'trials'
        tdir.mkdir(exist_ok=True)
        tmp = SCRATCH / f'{k}.tar.gz'
        with tarfile.open(tmp, 'w:gz') as tf:
            tf.add(raw, arcname=k)
        shutil.copyfile(tmp, tdir / f'{k}.tar.gz')
        tmp.unlink()
        out['trial_archive'] = str(tdir / f'{k}.tar.gz')
        if trial is not None and (trial / 'agent' / TRAJ).exists():
            ddir = self.run / 'transcripts'
            ddir.mkdir(exist_ok=True)
            with open(trial / 'agent' / TRAJ, 'rb') as f, gzip.open(ddir / f'{k}.json.gz', 'wb') as g:
                shutil.copyfileobj(f, g)
            out['transcript'] = str(ddir / f'{k}.json.gz')
        return out

    @staticmethod
    async def kill(proc) -> None:
        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), wait)
                return
            except asyncio.TimeoutError:
                pass

    # ---- loop
    def stop_requested(self) -> bool:
        return (self.run / 'STOP_WORKERS').exists()

    async def main(self) -> None:
        names = self.cfg['tasks']
        await asyncio.to_thread(stage_tasks, self.cfg['tasks_src'], names, self.tasks_dir)
        self.res = {n: task_resources(self.tasks_dir / n) for n in names}
        log(f'{self.name}: {len(names)} tasks staged; {len(self.cores_free)} cores, {self.mem_free // 1024} GB')
        idle = 0
        warned = 0.0
        while True:
            for k, item in list(self.running.items()):
                t = item.get('task')
                if t is not None and t.done():
                    if t.exception():
                        log(f'{k} crashed: {t.exception()!r}')
                        self.q.finish(k, {'infrastructure_failure': True, 'infrastructure_reason': 'worker_exception',
                                          'score': None, 'exception_type': 'WorkerException',
                                          'exception_message': repr(t.exception())[:2000]})
                    self.give(parse_key(k)[0], item['cores'])
                    del self.running[k]
            if time.time() - self.last_recover > 300:
                self.last_recover = time.time()
                for k in await asyncio.to_thread(self.q.recover_stale, 900):
                    log(f'{k}: stale claim recovered as worker_lost')
            if STOP:
                if not self.running:
                    return
            elif not self.stop_requested():
                ok, _, why = (True, None, '') if self.cfg.get('agent', 'mini') != 'mini' else \
                    await asyncio.to_thread(endpoint.alive, self.cfg['model'])
                if not ok and time.time() - warned > 600:
                    warned = time.time()
                    log(f'endpoint {self.cfg["model"]} not usable ({why}): not claiming')
                while ok:
                    try:
                        k = await asyncio.to_thread(self.q.claim, self.name, self.fits)
                    except OSError as e:
                        log(f'claim failed: {e!r}')
                        k = None
                    if k is None:
                        break
                    cores = self.take(parse_key(k)[0])
                    self.running[k] = {'cores': cores}
                    self.running[k]['task'] = asyncio.create_task(self.run_item(k, cores))
                    log(f'claimed {k} on cores {cores[0]}-{cores[-1]} ({len(self.running)} running)')
                    await asyncio.sleep(2)
            if not self.running:
                if await asyncio.to_thread(self.q.complete):
                    await asyncio.to_thread(score.finalize, self.run)
                    log('run complete; scored; exiting')
                    return
                idle += 1
                if self.stop_requested() or idle >= 45:   # 15 min with nothing this worker can claim
                    log('nothing to run; exiting')
                    return
            else:
                idle = 0
            atomic(self.run / 'workers' / f'{self.name}.json',
                   {'heartbeat': now(), 'running': sorted(self.running), 'cores_free': len(self.cores_free),
                    'mem_free_gb': self.mem_free // 1024})
            await asyncio.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--cpus', type=int, default=0, help='cores to use (default: this job\'s affinity mask)')
    ap.add_argument('--mem-gb', type=float, default=0, help='memory budget (default: 90%% of the job request)')
    a = ap.parse_args()
    if not a.mem_gb:
        a.mem_gb = 0.9 * float(os.environ.get('AA_WORKER_MEM_GB') or 64)

    def stop(sig, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    asyncio.run(Worker(a).main())


if __name__ == '__main__':
    main()

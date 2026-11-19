"""Inbox rollout worker (CPU job): run assigned attempts through Harbor + Pi cont16k.

  python -m synth.worker --index I --slots N      (SYNTH_CAMPAIGN=<ops/campaign.json>)

The controller is the only publisher: it records an attempt in its state, then
writes ops/workers/I/inbox/<uid>.json. Completion is raw/<uid>/worker-result.json.
Trajectories are written straight to the campaign's raw/ tree on /data, so a
lost worker loses at most its in-flight streams (exported as interrupted).
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tarfile
import time

from .common import PROXY_ENV, REPO, append_jsonl, atomic, best_effort, read
from . import sandbox_limits
from .config import Campaign, effective_protocol
from .memory_guard import MemoryGuard
from .package_proxy import DEFAULT_ALLOW as PACKAGE_PROXY_HOSTS, PIP_INDEX as PACKAGE_PROXY_PIP_INDEX, \
    SANDBOX_URL as PACKAGE_PROXY_URL
from tbshared import reslog   # the one cgroup sampler (#445 item 2)
from tb_agents import hung   # the hung-session watchdog pi_eval runs (#531: one rule for both pipelines)

HUNG_TICK_SEC = 60.0   # one watchdog check per attempt per minute, as pi_eval (a few globs and one local HTTP call)

HEARTBEAT_SECONDS = 15


def apply_stack_limit(kib):
    """Soft RLIMIT_STACK of this process (KiB; inherited by every child) -> the limit set, or None (0: unchanged).
    Same rule as harbor_ns.sandbox.apply_stack_limit, without importing harbor_ns into the worker. kib None: unlimited,
    as eval (the soft limit raised to the hard one; RLIM_INFINITY on HTCondor) -> 'unlimited' or the KiB set."""
    import resource
    if kib is None:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != hard:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
        return 'unlimited' if hard == resource.RLIM_INFINITY else hard // 1024
    if not kib:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = kib * 1024 if hard == resource.RLIM_INFINITY else min(kib * 1024, hard)
    if soft != want:
        resource.setrlimit(resource.RLIMIT_STACK, (want, hard))
    return want // 1024


FINISH_ENV = ('PI_FINISH_WARN_AT', 'PI_FINISH_RESERVE', 'PI_FINISH_STICKY')


def attempt_env(campaign, item, base, attempt_dir=None):
    """Pi cont16k environment for one attempt; identical for train and eval. attempt_dir: where the sandbox's disk-cap
    summary goes (raw/<uid>/diskcap.json)."""
    teacher, protocol = campaign.teacher, campaign.protocol
    request = teacher.get('request', {})
    effective = effective_protocol(protocol, item['task'])   # the campaign's, or its task source's override
    context, output = str(effective['context_tokens']), str(protocol['max_output_tokens'])
    if item.get('smoke_cap'):
        output = str(item['smoke_cap'])
    if item.get('smoke_context_window'):
        context = str(item['smoke_context_window'])
    overrides = {**request.get('overrides', {}), 'seed': item['seed']}
    env = {**base, 'AGENT_KIND': 'pi-cont16k', 'MODEL': request['client_model'],
           'MAX_LEN': context, 'PI_CONTEXT': context, 'PI_CONTEXT_WINDOW': context, 'PI_CONTEXT_PROXY': context,
           'MAX_TOKENS': output, 'PI_MAX_OUTPUT': output, 'PI_CONT16K_CAP': output,
           'PI_CONT16K_RESAMPLES': str(protocol['pi_resamples']), 'PI_VERSION': protocol['pi_version'],
           'PI_THINKING': request.get('pi_thinking', 'high'), 'PI_GRACEFUL_INTERRUPT': '1',
           'PI_REQUEST_OVERRIDES': json.dumps(overrides), 'PI_WIRE_LOG': '1',
           'PI_CLI_PROMPT_SEPARATOR': '1', 'HARBOR_ENVIRONMENT': 'synth.environment:CampaignEnvironment',
           # task-synthesis' harness defaults; the shared harness (#480) defaults to AA evals' native-MCP-off and
           # instruction-on-stdin (#203)
           'PI_NATIVE_MCP': '1', 'PI_INSTRUCTION_ARGV': '1',
           'PI_CONT16K_PROFILE': 'synth',   # gen/pi_cont16k.py: synthesis' proxy defaults (capture, polling, overrides)
           'SYNTH_ATTEMPT': item['id']}
    if protocol.get('pi_cache'):
        env['PI_CONT16K_CACHE'] = protocol['pi_cache']
    if protocol.get('glibc_compat_archive'):
        env['PI_RUNTIME_COMPAT_ARCHIVE'] = protocol['glibc_compat_archive']
    if item.get('dp_rank') is not None:
        env['PI_CAMPAIGN_DP_RANK'] = str(item['dp_rank'])
    if protocol.get('exact_headroom'):
        env['PI_CONTEXT_EXACT_HEADROOM'] = '1'
    harbor = campaign.config.get('harbor', {})
    if harbor.get('resource_guard', True):
        env['PI_CAMPAIGN_RESOURCE_GUARD'] = '1'  # sandbox PSS guard at the task's declared memory
    # Start-up readiness of the attempt's local helpers (#244: 7,209 of m1-hero's 7,642 pre-agent failures were these
    # deadlines missed on I/O-saturated worker hosts): the continuation proxy's first /models (per request, in all)
    # and the sandbox relays' socat (synth.endpoint_relay).
    env['PI_PROXY_READY_TIMEOUT_SEC'] = str(harbor.get('proxy_ready_timeout_sec', 10))
    env['PI_PROXY_READY_BUDGET_SEC'] = str(harbor.get('proxy_ready_budget_sec', 300))
    env['SYNTH_RELAY_READY_SEC'] = str(harbor.get('relay_ready_sec', 60))
    if request.get('pi_compat'):
        env['PI_COMPAT_JSON'] = json.dumps(request['pi_compat'])
    if request.get('chat_template_kwargs_env'):
        env['PI_CHAT_TEMPLATE_KWARGS'] = json.dumps(request['chat_template_kwargs_env'])
    for adapter in request.get('adapters', []):
        env[{'inline_reasoning_history': 'PI_INLINE_REASONING_HISTORY',
             'mirror_reasoning_content': 'PI_MIRROR_REASONING'}[adapter]] = '1'
    env.update({k: str(v) for k, v in request.get('env', {}).items()})
    if env.get('PI_CONTEXT_RENDERER'):   # task-synthesis' guard default: timings with a renderer (shared default: off)
        env.setdefault('PI_CONTEXT_TIMINGS', '1')
    # #375 finish notice (tb_agents/pi_context_limit.py), from the frozen protocol: v4 sticky notices for campaigns
    # created since 2026-11-14; none (v2, as before) for campaigns frozen without it. Off is explicit: a value in the
    # worker's own environment never turns it on.
    for k in FINISH_ENV:
        env.pop(k, None)
    fn = effective['finish_notice']
    if fn:
        env['PI_FINISH_WARN_AT'], env['PI_FINISH_RESERVE'] = str(fn['warn_at']), str(fn['reserve'])
        if fn.get('sticky'):
            env['PI_FINISH_STICKY'] = '1'
    # sandbox memory guard and disk cap (synth.sandbox_limits; campaigns frozen without them keep the old rules)
    sandbox_limits.attempt_env(env, harbor, item['task'], attempt_dir or campaign.root_for(item['kind']) / 'raw' / item['id'])
    return env


class Worker:
    def __init__(self, index, slots):
        self.campaign = Campaign.load()
        self.index, self.slots = index, slots
        self.home = self.campaign.ops / 'workers' / str(index)
        self.scr = Path(os.environ['_CONDOR_SCRATCH_DIR'])
        self.active = {}
        self.extracting = {}  # stream chunk tar -> lock (extract_chunk)
        self.stopping = False
        self.host = socket.getfqdn()
        self.job_id = os.environ.get('SYNTH_JOB_ID', '')
        # LITELLM_LOCAL_MODEL_COST_MAP: Harbor's LiteLLM otherwise fetches its cost map from GitHub
        # through the proxy at the end of every attempt (a 5 s timeout in 58% of 10-23 attempts).
        self.env = {**os.environ, **PROXY_ENV, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1',
                    'TMPDIR': str(self.scr / 'tmp'), 'HARBOR_NS_LOCAL': str(self.scr / 'harbor_ns'),
                    'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}
        # harbor_ns image builds: harbor.persist_images = false keeps task images in scratch (shared prefix bases);
        # network_rebuilds / network_backoff: rebuilds after downloads failed at the proxy, and their spacing.
        harbor = self.campaign.config.get('harbor', {})
        if harbor.get('persist_images') is False:
            self.env['HARBOR_NS_PERSIST_IMAGES'] = '0'
        if 'network_rebuilds' in harbor:
            self.env['HARBOR_NS_NETWORK_REBUILDS'] = str(int(harbor['network_rebuilds']))
        if 'network_backoff' in harbor:
            self.env['HARBOR_NS_NETWORK_BACKOFF'] = ','.join(str(float(x)) for x in harbor['network_backoff'])
        # harbor.offline_no_proxy: sandboxes without network get no proxy variables (tasksynth r4: agents in offline
        # sandboxes found https_proxy in env and spent turns trying it)
        if harbor.get('offline_no_proxy'):
            self.env['HARBOR_NS_OFFLINE_NO_PROXY'] = '1'
        # harbor.package_proxy: offline sandboxes reach the package registries only (synth.package_proxy)
        if harbor.get('package_proxy'):
            self.env['HARBOR_NS_OFFLINE_PROXY_URL'] = PACKAGE_PROXY_URL
            self.env['HARBOR_NS_OFFLINE_PROXY_HOSTS'] = ','.join(harbor.get('package_proxy_allow') or PACKAGE_PROXY_HOSTS)
            if harbor.get('package_cache', True) and harbor.get('package_cache_pip', False):
                self.env['HARBOR_NS_OFFLINE_PIP_INDEX'] = PACKAGE_PROXY_PIP_INDEX
        # harbor.package_cache (default on): one package cache per worker (synth.package_cache) for image builds and,
        # through the package proxy, the sandboxes' apt; its store lives in this job's scratch
        self.package_cache = None
        if harbor.get('package_cache', True):
            self.env['HARBOR_NS_PACKAGE_CACHE'] = str(self.scr / 'pkgcache.json')
        # image builds share the worker with ~48 attempts: a few cores each and a memory ceiling (harbor_ns.sandbox)
        self.env.setdefault('HARBOR_NS_BUILD_CPUS', str(harbor.get('build_cpus', 4)))
        self.env.setdefault('HARBOR_NS_BUILD_MEM_GB', str(harbor.get('build_mem_gb', 24)))
        # harbor_ns on main (#480) keeps aa-evals'/eval's offline no_proxy; synth keeps its own: offline sandboxes whose
        # no_proxy covers private names and ranges and the image's /etc/hosts names (harbor_ns.sandbox.OFFLINE_DOMAINS)
        self.env.setdefault('HARBOR_NS_OFFLINE_BYPASS', '1')
        # #171: HTCondor jobs run with an unlimited stack, where Docker (Terminal-Bench's runtime) gives 8 MiB. harbor, its
        # sandboxes and build steps inherit this process's soft limit, so a runaway recursion in a task gets its SIGSEGV
        # instead of growing until the worker job is held at its memory limit (m1-hero: a round job held at 392 GB from a
        # task setup expecting SIGSEGV; 2 Super worker jobs held for memory). harbor.stack_kb, 0: unchanged.
        # Campaigns frozen since 2026-11-14: unlimited, as eval (synth.sandbox_limits: a 64 MiB soft limit would make
        # glibc's default thread stacks 64 MiB instead of 2 MiB, a train/test difference under RLIMIT_AS caps).
        stack = apply_stack_limit(sandbox_limits.stack_kb(harbor))
        print(f'[worker] sandbox stack limit: {stack if stack is not None else "unchanged"}'
              f'{" KiB" if isinstance(stack, int) else ""}', flush=True)   # the effective value, checkable
        # harbor.worker_memory_guard (default on): synth.memory_guard keeps the job's attempts inside its memory (#244).
        self.guard = None
        if harbor.get('worker_memory_guard', True):
            self.guard = MemoryGuard(kill_fraction=harbor.get('worker_memory_kill_fraction', 0.85),
                                     admit_fraction=harbor.get('worker_memory_admit_fraction', 0.75),
                                     log=lambda event: best_effort(append_jsonl, self.home / 'memory-guard.jsonl',
                                                                   {**event, 'job_id': self.job_id, 'host': self.host}))
        self.admission_held_logged = 0
        (self.scr / 'tmp').mkdir(exist_ok=True)
        (self.scr / 'agent-state').mkdir(exist_ok=True)
        (self.home / 'inbox').mkdir(parents=True, exist_ok=True)

    def stage(self):
        for kind in ('train', 'eval'):
            if not self.campaign.enabled(kind):
                continue
            if kind == 'train' and not self.campaign.tasks(kind) and self.campaign.config['train'].get('stream'):
                continue  # every task comes in a stream chunk (extract_chunk)
            dst = self.scr / kind
            if not (dst / 'READY').exists():
                dst.mkdir(exist_ok=True)
                with tarfile.open(self.campaign.ops / f'{kind}_tasks.tar') as tf:
                    tf.extractall(dst, filter='tar')
                (dst / 'READY').touch()

    async def extract_chunk(self, bundle, kind):
        """A stream chunk's tasks (ops/stream/train-NNNN.tar) into this job's scratch, once, the first time an
        attempt needs one of them (synth/stream.py)."""
        marker = self.scr / kind / '.chunks' / f'{Path(bundle).name}.READY'
        if marker.exists():
            return
        async with self.extracting.setdefault(bundle, asyncio.Lock()):
            if marker.exists():
                return

            def unpack():
                with tarfile.open(bundle) as tf:
                    tf.extractall(self.scr / kind, filter='tar')
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            await asyncio.to_thread(unpack)

    async def run(self, item):
        uid, kind = item['id'], item['kind']
        out = self.campaign.root_for(kind) / 'raw' / uid
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'worker-result.json').exists():
            return
        if 'finish_notice' in self.campaign.protocol:   # campaigns since 2026-11-14: the protocol this attempt runs
            p = effective_protocol(self.campaign.protocol, item['task'])
            item = {**item, 'protocol': {**p, 'context_tokens': int(item.get('smoke_context_window') or p['context_tokens'])}}
        atomic(out / 'attempt.json', item)
        if (out / 'CANCEL').exists():
            atomic(out / 'worker-result.json', {'finished_at': time.time(), 'interrupted': True,
                                                 'infrastructure_failure': 'CancelledBeforeStart'})
            return
        task_kind = item.get('task_kind', kind)
        best_effort(self.free_scratch)  # an attempt cannot start (mkdir) on a full scratch
        if item.get('bundle'):
            await self.extract_chunk(item['bundle'], task_kind)
        selected = self.scr / 'single' / uid
        selected.mkdir(parents=True, exist_ok=True)
        link = selected / item['task']
        if not link.exists():
            link.symlink_to(self.scr / task_kind / 'tasks' / item['task'])
        env = attempt_env(self.campaign, item, self.env, out)
        env['PI_CONTEXT_STATE'] = str(self.state_file(uid))  # the agent's KV footprint, for admission
        task_dir = self.scr / task_kind / 'tasks' / item['task']
        toolchain = self.campaign.config.get('harbor', {}).get('verifier_toolchain')
        if toolchain:  # offline verifier toolchain + wrapper (synth/verifier_toolchain.py); never blocks the attempt
            prep = await asyncio.to_thread(self.prepare_verifier, task_dir, toolchain, uid)
            best_effort(atomic, out / 'verifier-prep.json', prep)
        started = time.time()
        atomic(out / 'worker-start.json', {'time': started, 'host': self.host, 'worker': self.index,
                                         'endpoint': item['endpoint'], 'backend': item['backend'],
                                         'job_id': self.job_id})
        harbor_args = self.campaign.config.get('harbor', {}).get('args', [
            '--environment-build-timeout-multiplier', '3', '--agent-setup-timeout-multiplier', '3'])
        # #531: frozen campaigns without harbor.hung_watchdog (made before it) run without the watchdog
        hw = (self.campaign.config.get('harbor') or {}).get('hung_watchdog')
        wd = hung.Watchdog(hw['idle_sec'], hw.get('stall_sec')) if hw else None
        hung_fired, next_tick = None, time.monotonic() + HUNG_TICK_SEC
        with (out / 'harbor.log').open('ab') as lf:
            proc = await asyncio.create_subprocess_exec('bash', str(REPO / 'gen/run_harbor.sh'),
                str(selected), str(out), 'harbor', item['endpoint'], '1', '-k', '1', *harbor_args,
                env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            self.active[uid]['pid'] = proc.pid
            interrupted = False
            while proc.returncode is None:
                guarded = self.active[uid].get('memory_guard_at')
                if guarded and time.monotonic() - guarded > 300:   # its sandbox is gone; Harbor did not wrap up
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
                    break
                if self.stopping or (out / 'CANCEL').exists():
                    interrupted = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), 120)
                    except asyncio.TimeoutError:
                        os.killpg(proc.pid, signal.SIGKILL)
                        await proc.wait()
                    break
                try:
                    await asyncio.wait_for(proc.wait(), 10)
                except asyncio.TimeoutError:
                    pass
                if proc.returncode is None and wd is not None and time.monotonic() >= next_tick:
                    next_tick = time.monotonic() + HUNG_TICK_SEC
                    try:   # the watchdog must never break an attempt
                        hung_fired = await asyncio.to_thread(wd.tick, out)
                    except Exception as e:   # noqa: BLE001
                        print(f'[worker] {uid}: hung-session watchdog off for this attempt: {e!r}', flush=True)
                        wd = None
                    if hung_fired is not None and proc.returncode is None:   # Harbor may have exited meanwhile
                        print(f'[worker] {uid}: hung session ({hung_fired[0]}, {hung_fired[1] / 60:.1f} min): '
                              'stopping (a mid-run infrastructure failure)', flush=True)
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                            await asyncio.wait_for(proc.wait(), 120)
                        except ProcessLookupError:
                            pass
                        except asyncio.TimeoutError:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            await proc.wait()
                        break
                    hung_fired = None   # fired as Harbor exited on its own: the attempt ended normally
        shutil.rmtree(selected, ignore_errors=True)
        self.state_file(uid).unlink(missing_ok=True)
        if toolchain:
            await asyncio.to_thread(self.release_verifier, task_dir, uid)
        guard = self.active[uid].get('memory_guard')
        hung_rec = {}
        if hung_fired is not None and not guard:
            # the same two names as pi_eval's worker (exception_type there): the guard's sticky accounting failure, or
            # a stalled/idle session. A mid-run infrastructure failure (counted toward max_infrastructure_failures_per_task,
            # like WorkerMemoryGuard; its export counts as usable when train.count_interrupted_usable, the default)
            hung_kind, sec = hung_fired
            hung_rec = {'infrastructure_failure': 'ResponseBudgetAccounting' if hung_kind == 'budget_accounting_failure'
                        else 'HungSession', 'hung_kind': hung_kind, 'hung_sec': round(sec, 1),
                        'hung_limit_sec': {'stalled_request': hw.get('stall_sec'), 'idle': hw['idle_sec']}.get(hung_kind)}
        atomic(out / 'worker-result.json', {'started_at': started, 'finished_at': time.time(),
            'returncode': proc.returncode, 'interrupted': interrupted, 'host': self.host,
            'worker': self.index, 'job_id': self.job_id,
            'termination_reason': 'campaign_interruption' if interrupted else 'memory_guard' if guard
                                  else 'hung_session' if hung_rec else 'harbor_exit',
            **({'infrastructure_failure': 'WorkerMemoryGuard', 'memory_guard': guard} if guard else hung_rec)})

    def prepare_verifier(self, task_dir, toolchain, uid):
        from .verifier_toolchain import prepare_verifier
        try:
            return prepare_verifier(task_dir, toolchain, self.scr / 'verifier-toolchain', holder=uid)
        except Exception as e:  # noqa: BLE001  (the task's own script then runs online, as before)
            return {'task': task_dir.name, 'mode': 'error', 'reason': repr(e)[:500]}

    def release_verifier(self, task_dir, uid):
        from .verifier_toolchain import release_verifier
        try:
            release_verifier(task_dir, self.scr / 'verifier-toolchain', holder=uid)
        except Exception as e:  # noqa: BLE001
            print(f'release_verifier {task_dir.name}: {e!r}', flush=True)

    def free_scratch(self):
        """Evict idle harbor_ns images from this job's scratch (Harbor itself only evicts once a
        build starts, which never happens when scratch is already full)."""
        from harbor_ns.builder import evict
        evict(self.scr / 'harbor_ns')

    def state_file(self, uid):
        return self.scr / 'agent-state' / f'{uid}.json'

    def agent_state(self, uid):
        try:
            return json.loads(self.state_file(uid).read_text())
        except (OSError, ValueError):
            return None

    def post_agent(self):
        """Active attempts whose agent has finished (Harbor wrote the agent's trajectory): they only collect artifacts
        and verify from here on, holding a worker slot but no KV. RTS pilot 2026-10-26: 531 of 639 live attempts were
        in a 4-5 min verifier (apt-get update at ~200 kB/s through the proxy) while counting as engine agents."""
        done = []
        for uid, entry in self.active.items():
            if not entry.get('post_agent'):
                item = entry['item']
                out = self.campaign.root_for(item['kind']) / 'raw' / uid / 'harbor'
                entry['post_agent'] = any(out.glob('*/agent/trajectory.json'))
            if entry['post_agent']:
                done.append(uid)
        return done

    def heartbeat(self):
        record = {'updated_at': time.time(), 'host': self.host, 'job_id': self.job_id,
                  'active': list(self.active), 'capacity': self.slots, 'stopping': self.stopping,
                  'contexts': {uid: c for uid in self.active if (c := self.agent_state(uid))},
                  'post_agent': best_effort(self.post_agent) or [],
                  # the cgroup history (sizing slots per worker; oom, failcnt, shmem, dirty/writeback explain memory
                  # holds RSS does not, 10-23): tbshared.reslog every HEARTBEAT_SECONDS, started in main()
                  'resources_log': str(self.home / 'resources.jsonl')}
        atomic(self.home / 'heartbeat.json', record)

    async def guard_loop(self):
        """synth.memory_guard over the live attempts, in a thread every poll_sec."""
        while True:
            try:
                attempts = {uid: entry.get('pid') for uid, entry in self.active.items()}
                record = await asyncio.to_thread(self.guard.check, attempts)
                if record and record['attempt'] in self.active:
                    self.active[record['attempt']].update(memory_guard=record, memory_guard_at=time.monotonic())
                    print(f'{time.strftime("%H:%M:%S")} memory guard: {record["action"]} of {record["attempt"]} '
                          f'({record["unreclaimable_bytes"] / 2**30:.1f} of {record["budget_bytes"] / 2**30:.1f} GiB)', flush=True)
            except Exception as e:  # noqa: BLE001  (a guard error must not stop the worker)
                print(f'memory guard error: {e!r}', flush=True)
            await asyncio.sleep(self.guard.poll_sec)

    async def main(self):
        # resources.jsonl: tbshared.reslog samples (fmt 2) with the worker's active attempts, every heartbeat period,
        # from a daemon thread (the sample never runs in this event loop)
        reslog.start(self.home / 'resources.jsonl', HEARTBEAT_SECONDS, extra=lambda: {'active': len(self.active)})
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: setattr(self, 'stopping', True))
        guard = asyncio.create_task(self.guard_loop()) if self.guard else None
        if 'HARBOR_NS_PACKAGE_CACHE' in self.env:
            from .package_cache import CacheProcess
            harbor = self.campaign.config.get('harbor', {})
            self.package_cache = await CacheProcess(
                self.scr / 'pkgcache', self.env['HARBOR_NS_PACKAGE_CACHE'],
                args=['--max-gib', str(harbor.get('package_cache_gib', 10)), '--min-free-gib', '15'],
                log=lambda m: print(f'{time.strftime("%H:%M:%S")} {m}', flush=True)).start()
        try:
            await self.loop()
        finally:
            if guard is not None:
                guard.cancel()
            if self.package_cache is not None:
                await self.package_cache.close()

    async def loop(self):
        self.stage()
        last_beat = 0
        while True:
            for uid, entry in list(self.active.items()):
                if entry['task'].done():
                    exc = entry['task'].exception()
                    if exc:
                        item = entry['item']
                        folder = self.campaign.root_for(item['kind']) / 'raw' / uid
                        best_effort(atomic, folder / 'worker-result.json', {'finished_at': time.time(),
                            'interrupted': True, 'infrastructure_failure': 'WorkerRunError', 'error': repr(exc)})
                    del self.active[uid]
            stop = self.home / 'STOP'
            admits = self.guard is None or best_effort(self.guard.admits) is not False
            if not admits and time.time() - self.admission_held_logged > 60:
                self.admission_held_logged = time.time()
                print(f'{time.strftime("%H:%M:%S")} memory guard: no new attempts while '
                      f'{self.guard.last["unreclaimable"] / 2**30:.1f} GiB of {self.guard.last["budget"] / 2**30:.1f} GiB '
                      f'is held', flush=True)
            if admits and not self.stopping and not stop.exists() and not (self.campaign.ops / 'protocol-defect.json').exists():
                for path in sorted((self.home / 'inbox').glob('*.json')):
                    if len(self.active) >= self.slots:
                        break
                    item = read(path)
                    if item is None or item['id'] in self.active:
                        continue
                    if (self.campaign.root_for(item['kind']) / 'raw' / item['id'] / 'worker-result.json').exists():
                        continue
                    self.active[item['id']] = {'item': item}
                    self.active[item['id']]['task'] = asyncio.create_task(self.run(item))
                    await asyncio.sleep(1)
            if time.time() - last_beat >= HEARTBEAT_SECONDS:
                best_effort(self.heartbeat)
                last_beat = time.time()
            if (self.stopping or stop.exists()) and not self.active:
                best_effort(self.heartbeat)
                break
            await asyncio.sleep(5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--slots', type=int, default=8)
    a = ap.parse_args()
    asyncio.run(Worker(a.index, a.slots).main())

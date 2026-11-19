#!/usr/bin/env python3
"""CPU rollout worker of a pi_eval run: eval_q35's worker (imported, not modified) against a serving registry.

  python -m aa_evals.common.pi_eval.worker --run RUN --slots N

eval_q35's worker expects its own serving job to publish RUN/serve/endpoint.url and admission.py to keep
RUN/queue/admission.json fresh. Here this process does both from the run's config:
- endpoint: the registry (serving:<name> -> endpoints.json urls). Each attempt is pinned to one replica (hash of its
  task and attempt, as eval_q35 pins DP ranks), so one rollout's turns reuse one prefix cache. Registry urls are
  checked every 30 s; claiming pauses while the registry is stale or the endpoint does not answer.
- admission: limit = the run's max_in_flight over all workers (claims - done), at most `burst` starts per minute.
- idle exit (#378, dispatch dirs with pi_autoscale and pi_idle_exit_sec, runs with RUN/jobs.json): a worker with
  nothing running, free to claim (endpoint up, gate clear, no STOP_WORKERS) and work pending, but no start credit
  left for pi_idle_exit_sec,
  exits and writes `exited: idle-no-credit` in its heartbeat file, so the dispatcher does not take the lane for dead
  and grows it back when credit returns. eval_q35's own exit: 5 min after the queue is fully claimed (#531; was 15).
- claim packing (#531, run config pack_claims): scarce work goes to the fullest claiming workers (claim_pack.py).
- hung-session watchdog (#531 A5, run config hung_watchdog): an attempt whose Pi session is hung ends as
  infrastructure failure `hung_session` (same-seed retry): the response-budget guard's sticky accounting failure, no
  progress on its oldest unanswered request for stall_sec, or Pi idle (no request in flight, no tool running) for
  idle_sec (hung.py).
Everything else is eval_q35's worker: claims, same-seed infrastructure retries, Pi 1.2.0 pi-cont16k through
gen/run_harbor.sh in harbor_ns, classification. Additions per run config: harbor_ns multi-uid (HARBOR_NS_MULTIUID=1),
an environment class (--env), the environment-clock cap (envclock_mult), and the oracle agent (agent: oracle).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import importlib.util
import os
import shutil
import signal
import socket
import tarfile
import time
import urllib.request
from pathlib import Path

from aa_evals.common import diskcap
from aa_evals.common.harbor_mswea import endpoint as registry
from aa_evals.common.pi_eval import hung, model_input, task_agents, task_environment

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('eval_q35_worker', REPO / 'eval_q35' / 'worker.py')
q35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q35)   # puts eval_q35/ on sys.path for its own imports (common, queue_state, ...)

ADMISSION_EVERY = 30
REAP_EVERY = 120
IDLE_EVERY = 20    # s between idle-exit checks
ORPHAN_SEC = 180   # an active claim with no heartbeat this long belongs to a lost worker: stay to reap it
STALE_SEC = 1200   # eval_q35/admission.py --stale-sec default
_q35_wrap_verifier = q35.wrap_verifier


def separate_verifier(task) -> bool:
    """[verifier] environment_mode = "separate" (or a tests/Dockerfile): the verifier runs in its own image built from
    tests/, whose Dockerfile may COPY test.sh by name (27 TB-Sci / lite tasks)."""
    try:
        cfg = q35.tomllib.loads((Path(task) / 'task.toml').read_text())
    except (OSError, ValueError):
        cfg = {}
    return (cfg.get('verifier') or {}).get('environment_mode') == 'separate' or (Path(task) / 'tests' / 'Dockerfile').exists()


def wrap_shared_verifiers_only(task):
    """eval_q35's verifier retry wrapper renames tests/test.sh to test_task.sh. A separate verifier image copies
    test.sh by name and would get the wrapper without the script; its bootstrap is baked into that image (no network
    at verification), so there is nothing to retry. Leave those tasks' tests/ untouched (also keeps the verifier build
    context identical to the validated, cached image)."""
    if not separate_verifier(task):
        _q35_wrap_verifier(task)


q35.wrap_verifier = wrap_shared_verifiers_only   # Worker.stage looks the module global up at call time

_q35_classify = q35.classify


def classify(raw, returncode, stopped, harness='pi', *, agent_internet=True):
    """eval_q35 classify, plus the #18 verifier rule (audit, #67): an attempt the verifier never scored is a model 0
    only when the verifier timed out (AA's rule: the tests ran out of time on the agent's work); a verifier that
    crashed, failed to bootstrap or never ran (no reward and any other exception, or none) is an infrastructure
    failure, retried with the same seed."""
    out = _q35_classify(raw, returncode, stopped, harness, agent_internet=agent_internet)
    if out.get('unscored') and not out.get('infrastructure_failure') and out.get('exception_type') != 'VerifierTimeoutError':
        out = {**out, 'score': None, 'infrastructure_failure': True, 'infrastructure_reason': 'verifier_no_reward'}
        out.pop('unscored', None)
    return out


def alive(url: str, timeout: float = 20) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip('/') + '/models', timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def gate_queued_new(url: str, timeout: float = 10) -> int | None:
    """New conversations waiting at an admission gate (aa_evals/qwenserve/gate.py GET /gate/status), None if unknown."""
    base = url.rstrip('/').removesuffix('/v1')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + '/gate/status', timeout=timeout) as r:
            return int(json.load(r).get('queued_new') or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class PiEvalWorker(q35.Worker):
    # aa-evals' worker rules, kept where eval_q35's Worker (eval's copy on main, #480) differs: no disk cap without a run
    # config "diskcap" (and so no one-attempt-per-worker tasks), no per-minute resource log. The no-claimable exit is
    # eval_q35's 5 min since #531 (was 15 min): an empty worker past the queue's last claim cost 2-5% of the M5 Pi
    # worker slot-hours. #378's no-credit idle exit and its first-claim grace (idle_state) are separate and unchanged.
    DISKCAP_WITHOUT_KEY = False
    POLL_SEC = 60.0      # run_item's heartbeat / stop / endpoint / watchdog check interval (eval_q35's 60 s)
    IDLE_EXIT_S = 300
    RESOURCE_LOG = False
    INSTRUCTION_ARGV = False   # the instruction on Pi's stdin (aa-evals #203); a run's request env or --harbor-env
    # may still set PI_INSTRUCTION_ARGV=1

    def __init__(self, a):
        super().__init__(a)
        self.urls: list[str] = []
        self.ok = False
        self.gate_host = None
        self.idle_exit_sec, self.idle_exited, self.leaving = 0.0, False, False
        self.started_at, self.claimed_any = time.time(), False   # idle exit waits for the first claim (idle_state)
        if self.config.get('dispatch'):   # #344 dispatcher: claims only within its start credit (pilane.py)
            from aa_evals.common.dispatch.pilane import credit_claim
            # a leaving worker (#378 idle exit) claims nothing more: checked under the claim lock try_leave takes
            credit_claim(self.state, self.run, refuse=lambda: self.leaving)
            try:
                dcfg = json.loads((Path(self.config['dispatch']['dir']) / 'dispatch.json').read_text())
            except (OSError, ValueError, KeyError, TypeError):
                dcfg = {}
            if dcfg.get('pi_autoscale'):   # #378: the dispatcher grows the lane back when credit returns
                self.idle_exit_sec = float(dcfg.get('pi_idle_exit_sec') or 0)
        if self.config.get('ctx_gate'):   # eval_q35's turn gate, hosted by one elected worker (turn_gate_host.py)
            from aa_evals.common.pi_eval.turn_gate_host import TurnGateHost
            self.gate_host = TurnGateHost(self, q35)
        # per-replica gate (#378): an attempt claimed while its hashed replica's gate holds runs on a clear replica
        self.held_urls: frozenset = frozenset()
        self.pinned: dict[str, str] = {}
        raw_claim = self.state.claim
        self.state.claim = self.replica_claim(self.state.claim)
        # tbsched (#451): claims within the scheduler's credit per engine, pinned to the engine with the most unspent
        # credit (sched_router.py); off unless run.py --sched-state; old rules whenever the scheduler is not running
        self.sched = None
        if self.config.get('sched'):
            if self.config.get('dispatch'):
                raise SystemExit('a run takes its start credit from the #344 dispatcher or from tbsched, not both')
            from aa_evals.common.pi_eval import sched_router
            self.sched = sched_router.install(self, raw_claim=raw_claim, log=q35.log)
        # #531 claim packing (run config pack_claims, new runs): scarce work goes to the fullest live workers, so the
        # others empty out and leave by the idle exits below; off when the tbsched router governs the run
        from aa_evals.common.pi_eval import claim_pack
        self.packer = claim_pack.install(self, log=q35.log)

    # ------------------------------------------------------------- endpoint + admission (eval_q35's serve job)
    def refresh(self) -> None:
        c = self.config
        urls, ok, why = [], False, ''
        if c.get('agent') == 'oracle':   # no model: a placeholder endpoint that nothing calls
            urls, ok = ['http://127.0.0.1:9/v1'], True
        else:
            usable, ep, why = registry.alive(c['model_spec'])
            urls = (ep or {}).get('urls') or []
            if ep and ep.get('served') != c['served']:
                usable, why = False, f"registry serves {ep.get('served')!r}, run expects {c['served']!r}"
            if usable and c.get('model_input'):
                try:
                    actual = model_input.resolve(ep, c['model_input']['declaration'])
                    if (c['model_input'].get('resolved_image_capability') is not None
                            and actual['resolved_image_capability'] is None):
                        raise ValueError('current launch no longer establishes frozen Pi model input')
                    if actual['modalities'] != model_input.frozen_modalities(c):
                        raise ValueError('current launch differs from frozen Pi model input')
                except (ValueError, KeyError, TypeError) as exc:
                    usable, why = False, str(exc)
            if usable:
                ok = bool(urls) and all(alive(u) for u in urls)
                why = '' if ok else 'endpoint not answering'
        if ok:
            self.urls = urls
            serve = self.run / 'serve'
            serve.mkdir(exist_ok=True)
            if q35.read(serve / 'urls.json') != urls:
                q35.atomic(serve / 'urls.json', urls)
            if not (serve / 'endpoint.url').exists() or (serve / 'endpoint.url').read_text().strip() != urls[0]:
                tmp = serve / f'.endpoint.url.{os.getpid()}'
                tmp.write_text(urls[0] + '\n')
                tmp.replace(serve / 'endpoint.url')
        if ok and self.gate_host is not None:
            self.gate_host.try_hold()
        if ok != self.ok:
            q35.log(f'endpoint {"up: " + ", ".join(urls) if ok else "DOWN (" + why + "): not claiming"}')
        self.ok = ok
        # Behind an admission gate (registry "gate": true) a new conversation's first request waits at the gate with
        # no response bytes, and Pi's 30-min idle timeout would fire on that queueing (overnight.md CUTOFF and
        # TIMEOUT rules: gate queue time never counts against a request). So no attempt starts on a replica whose gate
        # holds new conversations: the wait happens in our queue, before the attempt and its clocks start.
        # Per replica (#378): an attempt is pinned to one replica when it is claimed (url_for: the hash of its task
        # and attempt, or a clear replica when that one's gate holds), so the run pauses (admission.json `paused`,
        # which the #344 dispatcher reads as "gate-paused") only while every replica's gate holds; until then claims go
        # on in the run's order (longest predicted first). c30 Qwen3.8 Xhigh (2 replicas): one SciCode client filled
        # replica 2's gate while replica 1 had room, and every Pi lane paused ~50 min.
        hold, held_idx = '', []
        if ok and c.get('agent') != 'oracle':
            # probe every replica: harbor_mswea's endpoint record drops the registry's "gate" flag, and an engine
            # without a gate has no /gate/status (None)
            queued = [gate_queued_new(u) for u in urls]
            held_idx = [i for i, q in enumerate(queued) if q]
            n = sum(queued[i] for i in held_idx)
            if held_idx and len(held_idx) == len(urls):
                hold = f'gate holds {n} new conversation(s)'
            note = hold or (f'gate holds {n} new conversation(s) on replica(s) {",".join(map(str, held_idx))} of '
                            f'{len(urls)}: new attempts run on the others' if held_idx else '')
            if note != getattr(self, 'gate_note', ''):
                q35.log(f'admission: {note or "gate clear"}')
            self.gate_note = note
            self.hold = hold
        self.held_urls = frozenset(urls[i] for i in held_idx) if held_idx and not hold else frozenset()
        ctl = q35.read(self.run / 'control.json', {}) or {}   # live overrides: {"max_in_flight": N, "burst": N}
        sched = getattr(self, 'sched', None)
        governed = sched is not None and sched.active()   # tbsched's credit bounds and paces starts per engine
        from aa_evals.common.pi_eval.sched_router import OPEN
        q35.atomic(self.run / 'queue' / 'admission.json', {
            'epoch': time.time(), 'paused': not ok or bool(hold), 'reason': why or hold or None,
            'gate_held_replicas': held_idx, 'replicas': len(urls),
            'limit': OPEN if governed else int(ctl.get('max_in_flight', c['max_in_flight'])),
            'burst': OPEN if governed else int(ctl.get('burst', c.get('burst', 8))), 'window_sec': 60,
            'writer': self.name, **({'tbsched': True} if governed else {})})

    def replica_claim(self, inner):
        """State.claim, then pin the claimed attempt to its replica (pin)."""
        def claim(worker, tasks_available):
            k = inner(worker, tasks_available)
            if k is not None:
                self.pin(k)
            return k
        return claim

    def pin(self, k: str) -> str | None:
        """The attempt's replica for its whole life (one prefix cache): the hash of its task and attempt (url_for),
        unless that replica's gate held new conversations at the last refresh; then a clear one (also hashed),
        recorded in its claim dir (replica.json) for the run's other workers (turn_gate_host). Never raises: the
        attempt is already claimed (on any error it keeps its hashed replica)."""
        held, urls = self.held_urls, list(self.urls)   # one view: refresh may swap both meanwhile
        if not held or not urls:
            return None
        try:
            task, attempt, _ = q35.parse_key(k)
            hashed = urls[int(hashlib.sha256(f'{task}/{attempt}'.encode()).hexdigest(), 16) % len(urls)]
            clear = [u for u in urls if u not in held]
            if hashed not in held or not clear:
                return None
            url = clear[int(hashlib.sha256(f'{k}/repin'.encode()).hexdigest(), 16) % len(clear)]
            self.pinned[k] = url
            q35.atomic(self.run / 'queue' / 'claims' / k / 'replica.json', {'url': url, 'hashed': hashed,
                                                                          'why': 'gate holds'})
            q35.log(f'{k}: replica {urls.index(hashed)} gate holds: runs on replica {urls.index(url)}')
            return url
        except Exception as e:  # noqa: BLE001
            q35.log(f'{k}: replica pin failed ({e!r}): keeps its hashed replica')
            return self.pinned.get(k)

    async def keep_admission(self):
        last_reap = 0.0
        while not q35.STOP:
            await asyncio.to_thread(self.refresh)
            if time.time() - last_reap > REAP_EVERY:
                last_reap = time.time()
                await asyncio.to_thread(self.reap_stale)
            await asyncio.sleep(ADMISSION_EVERY)

    def can_claim(self) -> bool:
        """eval_q35's conditions plus this worker's own: endpoint answering, no gate hold, not leaving (#378)."""
        return super().can_claim() and self.ok and not getattr(self, 'hold', '') and not self.leaving

    def claimable(self):
        """eval_q35's (an unclaimed attempt or retry), or a lost worker's claim (no heartbeat for ORPHAN_SEC) that
        reap_stale will turn into a retry after STALE_SEC: no worker takes the no-claimable exit (IDLE_EXIT_S, 5 min
        since #531) while one waits, or a run without a dispatcher could be left with that attempt and no worker."""
        if super().claimable():
            return True
        try:
            return bool(set(self.state.stale(ORPHAN_SEC)) - set(self.running))
        except OSError:
            return True   # unsure: stay

    def reap_stale(self) -> None:
        """eval_q35/admission.py's stale-claim reaper (admission.py:244-249), which pi_eval runs without: a claim whose
        heartbeat (every 60 s while the attempt runs) is older than STALE_SEC belongs to a lost or held worker; it is
        recorded as an infrastructure failure (stale_claim) and its same-seed retry queued. Every worker runs it; a
        double reap writes the same records. A reaped attempt whose worker comes back later overwrites its record
        with the real outcome, and logical() takes the first non-infrastructure outcome."""
        try:
            for k in self.state.stale(STALE_SEC):
                if k in self.running:
                    continue
                q35.log(f'stale claim {k}: queued retry')
                self.state.finish(k, {'infrastructure_failure': True, 'infrastructure_reason': 'stale_claim',
                                      'score': None, 'exception_type': 'LostWorker'})
        except OSError as e:
            q35.log(f'stale-claim check failed: {e!r}')

    def url_for(self, k: str) -> str:
        """The replica attempt k runs on: pinned at its claim (pin), else the hash of its task and attempt."""
        pinned = getattr(self, 'pinned', {})
        if k in pinned:
            return pinned[k]
        rec = q35.read(self.run / 'queue' / 'claims' / k / 'replica.json')   # pinned by another worker
        return (rec or {}).get('url') or self.hashed_url(k)

    def hashed_url(self, k: str) -> str:
        task, attempt, _ = q35.parse_key(k)
        urls = self.urls or [(self.run / 'serve' / 'endpoint.url').read_text().strip()]
        h = int(hashlib.sha256(f'{task}/{attempt}'.encode()).hexdigest(), 16)
        return urls[h % len(urls)]

    @property
    def diskcap_worker_disk_mb(self):
        # the one-attempt-per-worker threshold (a quarter of the scratch) against this run's worker disk (run.py
        # --worker-disk-gb, default 300), not eval_q35's 200 GiB: GDPval's storage_mb 20480 x 3 fits four times
        disk_gb = (self.config.get('worker') or {}).get('disk_gb')
        return disk_gb * 1024 if disk_gb else super().diskcap_worker_disk_mb

    # ------------------------------------------------------------- per-attempt environment
    def env(self, k, task, attempt):
        e = super().env(k, task, attempt)
        c = self.config
        # #405 per-sandbox disk cap: eval_q35's env (super) sets it from the run config "diskcap" (run.py --diskcap,
        # default 3:30720; no key: off, DISKCAP_WITHOUT_KEY), summary to attempts/<key>/diskcap.json. The worker's own
        # environment never changes who the cap spares (aa_evals/common/diskcap.py), and with a "diskcap" in the run
        # config neither the registry's request env nor --harbor-env changes it (restored below), so attempt.json
        # "diskcap" is what the box runs with.
        e.pop('HARBOR_NS_DISKCAP_SPARE', None)
        cap_env = {x: e.get(x) for x in diskcap.ENV_KEYS}
        e['PI_GATE_CLIENT'] = c['benchmark']
        e['PI_AGENT_INTERNET'] = '1' if c.get('agent_internet', True) else '0'
        e['PI_RUNTIME_CACHE_MANIFEST'] = c.get('runtime_cache_manifest', '')
        e['PI_RUNTIME_CACHE_MANIFEST_SHA256'] = c.get('runtime_cache_manifest_sha256', '')
        extra = e.get('HARBOR_EXTRA_ARGS', '')
        if c.get('env_class'):
            extra += f" --env {c['env_class']}"   # after run_harbor.sh's own --env: the last one wins
        e['HARBOR_EXTRA_ARGS'] = extra.strip()
        if c.get('multiuid'):
            e['HARBOR_NS_MULTIUID'] = '1'
        req = c.get('request') or {}
        if req and c.get('agent') != 'oracle':   # M3.2: the model's own request settings (run.py resolve_request)
            e['PI_REQUEST_OVERRIDES'] = json.dumps(req['overrides'], sort_keys=True)
            for ad in req.get('adapters') or []:
                if ad == 'inline_reasoning_history':
                    e['PI_INLINE_REASONING_HISTORY'] = '1'
            if req.get('pi_thinking'):   # what Pi itself asks for; the proxy's overrides win where they set it
                e['PI_THINKING'] = req['pi_thinking']
            if req.get('pi_compat'):
                e['PI_COMPAT_JSON'] = json.dumps(req['pi_compat'])
            e.update({str(k): str(v) for k, v in (req.get('env') or {}).items()})
        e.update({str(k): str(v) for k, v in (c.get('harbor_env') or {}).items()})   # run.py --harbor-env KEY=VAL
        if 'diskcap' in c:
            for x, v in cap_env.items():
                e.pop(x, None)
                if v is not None:
                    e[x] = v
        if e.get('PI_RESPONSE_REPLAY_INDEX'):
            replay_index = json.loads(Path(e['PI_RESPONSE_REPLAY_INDEX']).read_text())
            e['PI_RESPONSE_REPLAY'] = replay_index[task]
        # This provider flag is dispatched only from the recorded task policy,
        # never a registry/profile/global worker environment inherited by other tasks.
        e.pop(task_environment.FLAG, None)
        e.update(task_environment.overrides_for(c, task))
        if c.get('ctx_gate') and c.get('agent') != 'oracle':
            # turn gate + KV brake of this attempt's replica (turn_gate_host.py writes serve/kv-<i>.json)
            url = self.url_for(k)
            i = self.urls.index(url) if url in self.urls else 0
            e['PI_KV_GATE_FILE'] = str(self.run / 'serve' / f'kv-{i}.json')
        m = (c.get('envclock_mult_by_task') or {}).get(task, c.get('envclock_mult', 1))
        if m != 1 and c.get('agent') != 'oracle':   # a cap below six hours (run.py --agent-cap-sec / --per-task-cap):
            # environment clock scaled to the task's cap; wall-clock backstop = 2 x max(task cap, run cap), so a
            # short-cap task on a slow (shared) endpoint is not cut by model time, which the Pi protocol leaves free
            e['PI_ENVCLOCK_MULT'] = repr(m)
            e['HARBOR_EXTRA_ARGS'] = e['HARBOR_EXTRA_ARGS'].replace(
                f"--agent-timeout-multiplier {c['backstop_multiplier']:g}",
                f"--agent-timeout-multiplier "
                f"{c['backstop_multiplier'] * max(m, float(c.get('agent_cap_sec') or 21600) / 21600):g}")
        if c.get('agent_wall_cap_sec') and c.get('agent') != 'oracle':
            # #344 wall cap: Harbor's agent timeout (agent run only; not env build, setup or verifier) = the cap. It
            # replaces the backstop above in both directions: a cap above 2 x the environment clock leaves model time
            # free on a slow endpoint, as the Pi protocol intends (m5l 11-16: Snowball at 13-26 tok/s per request
            # needs more than 4 h for the capped token budget; run.py bounds the cap at 2 x six hours). On timeout
            # Harbor records AgentTimeoutError and still runs the verifier, so the attempt is graded on the
            # environment it left (classify: score = reward)
            extra = e.get('HARBOR_EXTRA_ARGS', '')
            flag = '--agent-timeout-multiplier '
            cur = float(extra.split(flag, 1)[1].split()[0]) if flag in extra else None
            base = float((c.get('protocol') or {}).get('agent_timeout_sec') or 21600)
            mult = float(c['agent_wall_cap_sec']) / base
            if cur is None:
                extra = f'{extra} {flag}{mult:g}'.strip()
            elif mult != cur:
                extra = extra.replace(f'{flag}{extra.split(flag, 1)[1].split()[0]}', f'{flag}{mult:g}')
            e['HARBOR_EXTRA_ARGS'] = extra
        if c.get('agent') == 'oracle':   # the tasks' reference solutions through the same sandbox path
            for x in [x for x in e if x.startswith('PI_')] + ['TERMINUS']:
                e.pop(x, None)
            e.update(AGENT_KIND='terminus-2', AGENT='oracle')
        else:
            model_input.apply_env(c, e)
            task_agents.apply_env(c, task, e)
        return e

    async def run_item(self, k):
        """eval_q35 Worker.run_item (origin/eval 26be496), verbatim except the endpoint (this attempt's replica) and the
        first-claim flag."""
        self.claimed_any = True   # (pi_eval) idle exit waits for a worker's first claim (idle_state)
        task, attempt, retry = q35.parse_key(k)
        root = self.run / 'attempts' / k
        raw = q35.SCRATCH / 'raw' / k   # pi_eval: node-local; packed into root/trial.tar.gz when done
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        single = q35.SCRATCH / 'single' / k
        single.mkdir(parents=True, exist_ok=True)
        if not (single / task).exists():
            (single / task).symlink_to(self.tasks_dir / task)
        endpoint = self.url_for(k)   # (eval_q35: RUN/serve/endpoint.url)
        env = self.env(k, task, attempt)
        (q35.SCRATCH / 'tmp').mkdir(exist_ok=True)
        cmd = ['bash', str(q35.REPO / 'gen/run_harbor.sh'), str(single), str(raw), k, endpoint, '1', '-k', '1',
               '--environment-build-timeout-multiplier', '4']
        capture_metadata = None
        if self.config.get('required_output_capture'):
            from .required_outputs import plan_for, harbor_args, digest
            config_bytes = (self.run / 'config.json').read_bytes()
            if json.loads(config_bytes) != self.config:
                raise ValueError('run config changed before required-output capture binding')
            prepared = plan_for(self.config, k, self.tasks_dir / task, raw,
                                digest(config_bytes), self.run / 'config.json')
            if prepared is not None:
                capture_plan, capture_data = prepared
                capture_path = root / 'required-output-capture-plan.json'
                with capture_path.open('xb') as stream:
                    stream.write(capture_data)
                cmd.extend(harbor_args(capture_path, capture_data))
                capture_metadata = {'plan_path': str(capture_path), 'plan_sha256': digest(capture_data),
                                    'destination': capture_plan['destination'],
                                    'sources': capture_plan['sources']}
        info = {'key': k, 'task': task, 'attempt': attempt, 'retry': retry, 'seed': q35.seed(task, attempt, self.config.get('seed_namespace', q35.SEED_NAMESPACE)),
                'model': self.config['model'], 'served': self.config['served'], 'endpoint': endpoint,
                'worker': self.name, 'host': socket.gethostname(), 'started_at': q35.now(), 'started_epoch': time.time(),
                'command': cmd, 'protocol': {**q35.PROTOCOL, **self.config.get('protocol', {})}, 'envclock': bool(self.config.get('envclock')),
                'environment': {x: env[x] for x in env if x.startswith(('PI_', 'MAX_', 'HARBOR_EXTRA', 'AGENT_KIND', 'TERMINUS', 'TEMPERATURE', 'TOP_P'))},
                'diskcap': q35.diskcap_config(self.diskcap_cfg, task)}   # #405: the task's effective cap (None: off)
        for f in ('diskcap.json', 'diskcap.verifier.json'):   # a re-run of this key must not report an old box
            (root / f).unlink(missing_ok=True)
        if capture_metadata is not None:
            info['required_output_capture'] = capture_metadata
        if self.config.get('task_environment_policy') is not None:
            info['task_environment_policy_sha256'] = self.config['task_environment_policy_sha256']
            info['task_environment_overrides'] = task_environment.overrides_for(self.config, task)
            info['environment'].update(info['task_environment_overrides'])
        hw = self.config.get('hung_watchdog') if self.config.get('agent') != 'oracle' else None
        # #531 A5 (hung.py); None: off (runs made before it)
        wd = hung.Watchdog(hw['idle_sec'], hw.get('stall_sec', hung.STALL_SEC)) if hw else None
        info['hung_watchdog_sec'] = wd.idle_sec if wd else None
        info['hung_stall_sec'] = wd.stall_sec if wd else None
        q35.atomic(root / 'attempt.json', info)
        stopped, hung_idle = False, None
        with (root / 'harbor.log').open('ab', buffering=0) as output:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=output, stderr=asyncio.subprocess.STDOUT,
                                                        env=env, start_new_session=True)
            self.running[k]['proc'] = proc
            gone, cut_at = 0, None
            while proc.returncode is None:
                self.state.heartbeat(k)
                try:
                    await asyncio.wait_for(proc.wait(), self.POLL_SEC)
                except asyncio.TimeoutError:
                    pass
                if q35.STOP and proc.returncode is None:
                    stopped = True
                    await self.kill(proc)
                elif proc.returncode is None and self.cut_file(k).exists():
                    if cut_at is None:   # the agent stops itself (cut file); acknowledge so orchestrate waits
                        cut_at = time.time()
                        q35.atomic(self.cut_file(k).with_suffix('.ack'), {'worker': self.name, 'at': q35.now()})
                        q35.log(f'{k}: tail cut; the agent stops and the verifier runs')
                    elif time.time() - cut_at > 2700:   # no clean stop within 45 min: end it (scored as a failure)
                        stopped = 'tail_cut'
                        await self.kill(proc)
                elif proc.returncode is None:
                    # The run's serving job moved (e.g. 4B H100 -> B200 takeover, or a restart elsewhere): the agent
                    # keeps calling the old endpoint (Pi's proxy waits out a dead backend for 30 min). Two checks a
                    # minute apart that it refuses connections end the attempt for a same-seed retry.
                    gone = gone + 1 if await asyncio.to_thread(self.endpoint_gone, endpoint) else 0
                    if gone >= 2:
                        q35.log(f'{k}: endpoint {endpoint} gone (the run now serves at another endpoint); stopping')
                        stopped = 'endpoint_moved'
                        await self.kill(proc)
                    elif stopped == 'hung_session':   # kill() left it alive: try again each poll (as STOP does)
                        await self.kill(proc)
                    elif wd is not None and not stopped:   # once fired, never again
                        # an outage the worker sees (endpoint down, stale registry, cluster proxy failing) is not idle
                        ok = bool(getattr(self, 'ok', True)) and bool(getattr(self, 'proxy_state', (0, True))[1])
                        try:
                            fired = await asyncio.to_thread(wd.tick, raw, ok)
                        except Exception as e:   # noqa: BLE001 -- the watchdog must never break an attempt
                            q35.log(f'{k}: hung-session watchdog off for this attempt: {e!r}')
                            wd, fired = None, None
                        if fired is not None:
                            hung_idle = fired
                            q35.log(f'{k}: hung session ({fired[0]}, {fired[1] / 60:.1f} min); stopping for a '
                                    f'same-seed retry')
                            stopped = 'hung_session'
                            await self.kill(proc)
        outcome = classify(raw, proc.returncode, stopped if stopped != 'tail_cut' else True, self.config.get('harness', 'pi'),
                           agent_internet=self.config.get('agent_internet', True))
        if hung_idle is not None:   # #531 A5: an infrastructure failure (same-seed retry, capped; outages exempt);
            # kept even if a later STOP or endpoint_moved overwrote `stopped` while the kill was slow (a tail cut below
            # still takes precedence, as the tail rule says)
            stopped = 'hung_session'
            kind, sec = hung_idle
            # the guard's sticky accounting failure has its own reason, the one classify() gives when Pi itself
            # reports the guard's error (#549): whichever ends the attempt first, the same failure gets one name
            acct = kind == 'budget_accounting_failure'
            outcome.update(infrastructure_failure=True, score=None,
                           infrastructure_reason='response_budget_accounting' if acct else 'hung_session',
                           exception_type='ResponseBudgetAccounting' if acct else 'HungSession',
                           hung_kind=kind, hung_sec=round(sec, 1),
                           hung_limit_sec={'stalled_request': info['hung_stall_sec'], 'idle': info['hung_watchdog_sec']}
                           .get(kind))
        cut = q35.read(self.cut_file(k))
        if cut:   # tail rule: not an infrastructure failure, no retry; scored by the verifier when it ran
            scored = 'verifier' if stopped != 'tail_cut' and isinstance(outcome.get('reward'), (int, float)) else 'fail'
            outcome.update(infrastructure_failure=False, infrastructure_reason=None, tail_cut=True,
                           tail_cut_reason=cut.get('reason'), tail_cut_idle_h=cut.get('idle_h'), tail_cut_scored=scored,
                           score=float(outcome['reward']) if scored == 'verifier' else 0.0)
        outcome['wall_sec'] = round(time.time() - info['started_epoch'], 1)
        outcome.update(host=info.get('host'), worker=info.get('worker'))   # per-node failure rates (node_health.py)
        # #405: the trial's boxes (upper dirs + pod volumes): peak data, largest file, the cap's kills
        outcome.update(diskcap.outcome_fields(root / 'diskcap.json'))
        outcome.update(await asyncio.to_thread(self.pack, k, raw, root, outcome))   # pi_eval: stats + one archive
        if stopped and outcome.get('stats', {}).get('end_state'):
            # the agent's env-clock wrapper reports any external kill as wall_clock_backstop; say who stopped it
            outcome['stats']['end_state_agent'] = outcome['stats']['end_state']
            outcome['stats']['end_state'] = stopped if isinstance(stopped, str) else 'worker_stopped'
        q35.atomic(root / 'outcome.json', outcome)
        if (self.state.q / 'done' / f'{k}.json').exists():
            q35.log(f'{k}: already resolved by the stale-claim monitor; outcome kept in {root}/outcome.json only')
        else:
            self.state.finish(k, outcome)
        q35.log(f"done {k}: score={outcome.get('score')} infra={outcome.get('infrastructure_reason')} "
            f"exc={outcome.get('exception_type')} {outcome['wall_sec']:.0f}s")


    def pack(self, k: str, raw: Path, root: Path, outcome: dict) -> dict:
        """Per-attempt stats (score.attempt_stats) from the local trial, then the trial as one tar.gz in the run dir
        (one file per attempt instead of ~150; the streamed wire responses compress well)."""
        from aa_evals.common.pi_eval.score import attempt_stats
        out = {'stats': attempt_stats(outcome)}
        arc = root / 'trial.tar.gz'
        tmp = root / f'.trial.tar.gz.{os.getpid()}'
        local = raw.parent / f'.{k}.trial.tar.gz'   # node-local scratch, next to the raw trial
        def tar(dest):
            with tarfile.open(dest, 'w:gz', compresslevel=6) as tf:
                tf.add(raw, arcname=k)
        try:
            try:
                tar(local)
                direct = False
            except OSError as e:
                local.unlink(missing_ok=True)
                from harbor_ns.builder import evict, out_of_space
                if not out_of_space(e):
                    raise
                # #451 lesson 7: this worker's scratch is full (once full, every later attempt kept its trial here and
                # failed too). Free harbor_ns images (forced eviction) and build the archive again; if the scratch is
                # still full, write it straight into the run dir (no local copy), so the rollout is kept either way.
                try:
                    out['pack_evicted'] = evict(q35.SCRATCH / 'harbor_ns', force=True)
                except Exception as ee:  # noqa: BLE001
                    out['pack_evicted'] = {'error': repr(ee)}
                try:
                    tar(local)
                    direct = False
                except OSError as e2:
                    local.unlink(missing_ok=True)
                    if not out_of_space(e2):
                        raise
                    tar(tmp)
                    fd = os.open(tmp, os.O_RDONLY)
                    try:
                        os.fsync(fd)   # a write-back error on /data is a pack error, not an archive renamed in place
                    finally:
                        os.close(fd)
                    from aa_evals.common.fastio import drop_cache
                    drop_cache(tmp)   # #378: no dirty page cache left in the worker's memory cgroup
                    direct = out['pack_direct'] = True
            if not direct:
                # #378: O_DIRECT to /data, so the archive leaves no dirty page cache in the worker's memory cgroup
                # (over-quota /data writes back slowly; dirty pages got whole workers held "over memory limit")
                from aa_evals.common.fastio import copy_direct
                copy_direct(local, tmp)
            tmp.replace(arc)
            out['trial_archive'] = str(arc)
            out['trial_dir_in_archive'] = str(Path(outcome['trial_dir']).relative_to(raw.parent)) \
                if outcome.get('trial_dir') else None
            shutil.rmtree(raw, ignore_errors=True)
        except OSError as e:
            out['pack_error'] = repr(e)
            tmp.unlink(missing_ok=True)
        finally:
            local.unlink(missing_ok=True)
        return out

    def endpoint_gone(self, endpoint):
        """The attempt's replica left the registry and refuses connections (eval_q35: same-seed retry)."""
        if endpoint.rstrip('/') in [u.rstrip('/') for u in self.urls]:
            return False
        hostport = endpoint.split('://', 1)[-1].split('/', 1)[0]
        host, port = hostport.rsplit(':', 1)
        try:
            socket.create_connection((host, int(port)), timeout=10).close()
            return False
        except OSError:
            return True

    # ------------------------------------------------------------- idle exit (#378)
    def idle_state(self) -> tuple[bool, bool]:
        """(free and idle: nothing running, endpoint up, gate clear, no STOP_WORKERS, work pending, the dispatcher has
        granted credit before, the run's workers are its own condor jobs, and no lost worker's claim waits to be
        reaped; no unspent start credit now)."""
        from aa_evals.common.dispatch.pilane import n_claims
        ctl = q35.read(self.run / 'control.json', {}) or {}
        granted = int(ctl.get('max_starts') or 0)
        claims = n_claims(self.run / 'queue')
        # jobs.json: the run submits its own workers, so the dispatcher can grow it back (not --no-submit runs)
        free = (not self.running and self.ok and not getattr(self, 'hold', '') and granted > 0
                and (self.run / 'jobs.json').exists() and not self.stop_requested() and self.claimable())
        # a worker that never claimed (a first-fleet or newly grown worker waiting for credit, val-sb10b 11-18: a
        # lazy lane lost half its first fleet during the launch grab) is not idle before 2 idle-exit periods after its
        # start, so it leaves no earlier than 3 periods in all (2 + the idle period watch_idle then requires)
        if free and not getattr(self, 'claimed_any', True) and \
                time.time() - getattr(self, 'started_at', 0.0) < 2 * float(self.idle_exit_sec or 0):
            free = False
        if free and set(self.state.stale(ORPHAN_SEC)) - set(self.running):
            # a held or evicted worker's attempt: stay until this worker's reaper (STALE_SEC) queues its retry, or
            # the lane would be left with an active claim and no worker (dead, never grown back)
            free = False
        if free and self.gate_host is not None and self.gate_host.thread and self.gate_host.thread.is_alive():
            # this worker hosts the run's turn gate: it stays while other workers' attempts use it
            free = not (self.state._names('claims') - self.state._names('done', '.json'))
        return free, claims >= granted

    def try_leave(self) -> bool:
        """Decide to leave under the run's claim lock (pilane.credit_claim's), so no claim of this worker is half
        made: still no unspent credit and no active claim of this worker. Then claims are refused (self.leaving)."""
        from aa_evals.common.dispatch.pilane import n_claims
        lock = self.run / 'queue' / 'dispatch.lock'
        try:
            lock.mkdir()
        except FileExistsError:
            return False   # a claim is being made: decide next round
        try:
            ctl = q35.read(self.run / 'control.json', {}) or {}
            if n_claims(self.run / 'queue') < int(ctl.get('max_starts') or 0) or self.running:
                return False
            for k in self.state._names('claims') - self.state._names('done', '.json'):
                if (q35.read(self.run / 'queue' / 'claims' / k / 'claim.json', {}) or {}).get('worker') == self.name:
                    return False
            self.leaving = True
            return True
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    async def watch_idle(self):
        since = None
        while not q35.STOP:
            await asyncio.sleep(IDLE_EVERY)
            try:
                free, no_credit = await asyncio.to_thread(self.idle_state)
            except OSError:
                continue
            if not free:
                since = None
                continue
            since = since or time.time()
            # idle that long and no unspent credit now (credit left unspent: burst pacing, or another worker about to
            # claim it; a worker that keeps losing claims to the others is surplus and leaves)
            if time.time() - since >= self.idle_exit_sec and no_credit and not self.running:
                try:
                    leave = await asyncio.to_thread(self.try_leave)
                except OSError:
                    leave = False
                if not leave:
                    continue
                q35.log(f'idle {time.time() - since:.0f}s with work pending but no start credit: exiting '
                        '(the dispatcher adds workers back when credit returns)')
                self.idle_exited = True
                q35.STOP = True
                return

    def mark_exited(self, reason: str = 'idle-no-credit') -> None:
        """The dispatcher reads this. idle-no-credit: the lane lost this worker for lack of credit, it is not dead.
        no-claimable: eval_q35's exit with nothing left to claim; if the run's last worker then dies holding claims,
        the lane is revivable (its claims are reaped by a new worker) rather than abandoned."""
        q35.atomic(self.run / 'workers' / f'{self.name}.json',
                   {'heartbeat': q35.now(), 'running': sorted(self.running), 'exited': reason})

    def exit_reason(self) -> str | None:
        """Why main() is returning: the #378 idle exit, the run complete, eval_q35's no-claimable exit (nothing
        running, no stop, no signal: q35.STOP is set by the signal handlers), or None (a stop, a signal, a crash never
        gets here, or attempts still running). Only idle-no-credit and no-claimable make a dead lane revivable."""
        if self.idle_exited:
            return 'idle-no-credit'
        if self.running or q35.STOP or self.stop_requested():
            return None
        try:
            if self.state.complete():
                return 'complete'
        except OSError:
            return None
        return 'no-claimable'

    async def main(self):
        await asyncio.to_thread(self.refresh)
        keeper = asyncio.create_task(self.keep_admission())
        idler = asyncio.create_task(self.watch_idle()) if self.idle_exit_sec > 0 else None
        try:
            await super().main()
        finally:
            keeper.cancel()
            if idler is not None:
                idler.cancel()
        reason = self.exit_reason()
        if reason:
            self.mark_exited(reason)
        if self.state.complete():
            from aa_evals.common.pi_eval import score
            try:
                score.finalize(self.run)
            except Exception as e:   # scoring is re-runnable (run.py --score)
                q35.log(f'scoring failed: {e!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--slots', type=int, default=32)
    a = ap.parse_args()

    def stop(sig, frame):
        q35.STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    asyncio.run(PiEvalWorker(a).main())


if __name__ == '__main__':
    main()

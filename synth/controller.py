"""Persistent campaign owner (CPU job): serve, smoke, admit, export, score, stop, finalize.

  python -m synth.controller          (SYNTH_CAMPAIGN=<ops/campaign.json>)

Stages: smoke -> running -> finalizing -> complete. The controller is the only
writer of controller/state.json and the only publisher of worker inboxes; the
state records an attempt before its inbox file exists. Serving jobs, workers
and exports are separate processes, so a controller restart (code reload, node
loss) never kills a rollout.

Backends are individual vLLM replicas/ranks published by synth.serve. Each
attempt is pinned to one backend (prefix-cache locality) and admission is per
backend. `admission.mode = "kv_fit"` (synth/admission.py) admits while the KV every
live agent holds (its cached context) fits a self-tuning share of the backend's KV
capacity, and scales the workers to match; campaigns frozen without a mode keep the
original limit controller (KV hysteresis, a growth forecast, preemption back-off and
queue depth).

Training modes:
  tokens    cycle all tasks until saved usable tokens >= train.tokens_target;
  per_task  give every task train.trajectories_per_task usable trajectories,
            counting train.baseline_manifests (an existing corpus), fewest first.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import errno
import gzip
import heapq
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import admission as kv_admission
from . import archive
from . import autoscale
from . import condor
from . import stream as stream_chunks
from .common import (PROXY_ENV, PYTHON, REPO, MkdirLease, append_jsonl, atomic, best_effort, harbor_result, read,
                     seed, sha)
from .config import Campaign

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CONTINUATION_TEXT = 'Thinking exceeded the maximum thinking budget, please continue.'
CAP_TEXT = 'harbor-gen context cap reached'


# Harbor's errors for a failed agent command. After model requests they are the agent's own outcome: Harbor names
# the error after network strings anywhere in the command's output (the agent's transcript), and m1-hero's 461
# "NetworkConnectionError" attempts were pi killed by the agent's own `pkill -f` (exit -15/-9), all usable (#244).
AGENT_COMMAND_FAILURES = ('NonZeroAgentExitCodeError', 'NetworkConnectionError')
# Reasons the controller or its operator interrupt attempts for. They say nothing about the task or the worker host and
# count toward no cap (interrupt() also marks its items controller_interruption).
CONTROLLER_INTERRUPTIONS = ('OwnedBackendRestart', 'OwnedServingAllocationLost', 'TrainingClosed', 'TrainingTargetReached',
                            'TaskBlocked', 'ServingCapped', 'HostExcluded', 'CancelledBeforeStart', 'WorkerInterrupted')


def model_requests(folder):
    """True/False: the attempt's continuation proxy logged a model request (pi-cont16k requests.jsonl, under harbor/
    or, when Harbor wrote no job directory, .pi-cont16k/<run>/). None: no Pi cont16k logs at all. m1-hero (#244): no
    requests.jsonl in 399 of 400 sampled attempts that failed before the agent, in none of 400 usable ones."""
    logs = [*Path(folder).glob('harbor/pi-cont16k'), *Path(folder).glob('.pi-cont16k/*')]
    if not logs:
        return None
    for d in logs:
        try:
            if (d / 'requests.jsonl').stat().st_size > 0:
                return True
        except OSError:
            pass
    return False


def infrastructure_failure(result, finished, requests=None):
    """The attempt's infrastructure failure, or None when it ran as an attempt. `requests`: model_requests() of it."""
    if finished.get('infrastructure_failure'):
        return finished['infrastructure_failure']
    if finished.get('interrupted'):
        return 'WorkerInterrupted'  # the worker stopped mid-attempt: truncated
    result = result or {}
    exc_type = (result.get('exception_info') or {}).get('exception_type', '')
    if requests is False:
        # Nothing reached the model, whatever Harbor reports. m1-hero (#244): 2,336 relay start-up timeouts surfaced as
        # AgentTimeoutError and 1,545 proxy start-up failures left no trial result; both counted as real attempts.
        return f'PreAgentFailure/{exc_type or "NoModelRequest"}'
    if exc_type and not (result.get('agent_execution') or {}).get('started_at') and not result.get('agent_result'):
        return f'PreAgentFailure/{exc_type}'
    if requests and exc_type in AGENT_COMMAND_FAILURES:
        return None
    if any(s in exc_type.lower() for s in ('environment', 'setup', 'connect', 'network', 'docker', 'imagepull')):
        return exc_type
    return None


def failure_phase(item):
    """None (no failure), 'interruption' (the controller's), 'pre_agent' or 'mid_run'."""
    failure = item.get('infrastructure_failure')
    if not failure:
        return None
    if item.get('controller_interruption') or failure in CONTROLLER_INTERRUPTIONS:
        return 'interruption'
    return 'pre_agent' if failure.startswith('PreAgentFailure/') else 'mid_run'


def short_host(name):
    return (name or '').split('.')[0] or None


def write_text(path, text, retries=8):
    """common.atomic for an already serialized document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.{attempt}.tmp')
        try:
            with tmp.open('w') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            return
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 10))


class StateJournal:
    """What must survive a crash between state snapshots: attempts at assignment (appended before their inbox is
    published) and the dispatch counters. Segments controller/journal/<first seq>.jsonl; a snapshot records the last
    seq it holds (journal_seq) and the segments it covers are deleted once it is written. m1-hero (#235): the whole
    state (172 MB at the end) was written twice per tick; one tick took 499 s, 222 s of it saving."""
    KEYS = ('train_cursor', 'frontier', 'budget_cursor', 'skipped_invalid', 'train_starved_since')
    LISTS = ('opened', 'invalid')

    def __init__(self, folder):
        self.folder = Path(folder)
        self.seq = 0
        self.segment = None
        self.lengths = {}

    def replay(self, state):
        """Apply records newer than the snapshot to `state`; returns how many were applied."""
        base = seq = state.get('journal_seq', 0)
        applied = 0
        for path in sorted(self.folder.glob('*.jsonl'), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
            try:
                lines = path.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a line cut by a crash
                if record['seq'] <= base:
                    continue
                for item in record.get('items', []):
                    state['items'].setdefault(item['id'], item)
                state.update(record.get('set', {}))
                for key, values in record.get('extend', {}).items():
                    have = state.setdefault(key, [])
                    seen = set(have)
                    have.extend(v for v in values if v not in seen)
                seq = max(seq, record['seq'])
                applied += 1
        self.seq = seq
        self.lengths = {k: len(state.get(k) or []) for k in self.LISTS}
        return applied

    def append(self, state, items):
        self.seq += 1
        record = {'seq': self.seq, 'time': time.time(), 'items': items,
                  'set': {k: state[k] for k in self.KEYS if k in state}}
        extend = {k: (state.get(k) or [])[self.lengths.get(k, 0):] for k in self.LISTS}
        if any(extend.values()):
            record['extend'] = {k: v for k, v in extend.items() if v}
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.segment is None:
            self.segment = self.folder / f'{self.seq}.jsonl'
        with open(self.segment, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.lengths = {k: len(state.get(k) or []) for k in self.LISTS}

    def rotate(self, state):
        """A snapshot of `state` is being taken now: it holds every record so far; the next one opens a new segment."""
        self.segment = None
        self.lengths = {k: len(state.get(k) or []) for k in self.LISTS}
        return self.seq

    def drop_through(self, seq):
        """A snapshot holding records up to seq is on disk: delete the segments it covers."""
        for path in self.folder.glob('*.jsonl'):
            if path.stem.isdigit() and int(path.stem) <= seq and path != self.segment:
                best_effort(path.unlink, missing_ok=True)


def prefer_eval(active_train, active_eval, capacity, reserved, train_stopped=False):
    """Share admission proportionally from the first burst; keep training progressing."""
    if train_stopped:
        return True
    return active_eval < reserved and active_eval * (capacity - reserved) <= active_train * reserved


def engine_signature(config):
    """What a serving job runs for a frozen campaign config: the rendered vLLM command (dummy ports and paths)
    and the engine environment. Client-side request settings and unused optional args do not change it."""
    from .vllm_launch import replica_command
    teacher, protocol = config['teacher'], config['protocol']
    serving = teacher['serving']
    tunables = serving.get('tunables', {})
    ports = {'dp_size': config.get('serving', {}).get('gpus_per_job', 1), 'rpc_port': 1, 'supervisor_port': 2,
             'stage_draft': '/draft'}
    argv = replica_command(teacher, protocol, port=0, model_dir='/model', tunables=tunables, rank_ports=ports)
    runtime = {k: v for k, v in teacher['runtime'].items()}
    return {'argv': argv, 'runtime': runtime, 'serving_env': serving.get('env'), 'extra_env': tunables.get('extra_env'),
            'mode': serving.get('mode'), 'gpus_per_replica': serving.get('gpus_per_replica'),
            # no 'revision': the engine serves the staged local checkpoint (no --revision, HF offline), so the revision
            # is provenance only (recorded by synth/numerics.py); a corrected label must not refuse adoption
            **{k: teacher.get(k) for k in ('model', 'checkpoint', 'served_name')}}


def same_engine(frozen, config):
    """May a campaign with `config` run on serving started for the frozen campaign? Same engine command and
    environment (2026-10-25: a request-env or unused optional-arg change refused adoption and cost a cold start)."""
    try:
        return engine_signature(frozen) == engine_signature(config)
    except (KeyError, IndexError, ValueError):
        same = ('model', 'checkpoint', 'served_name', 'runtime', 'serving', 'request')
        return all(frozen['teacher'].get(k) == config['teacher'].get(k) for k in same)


# What task builds and verifiers fetch through the cluster proxy: apt indices, PyPI, the astral uv installer.
PROXY_PROBE_URLS = ['http://archive.ubuntu.com/ubuntu/dists/jammy/Release.gpg', 'https://pypi.org/simple/pluggy/',
                    'https://astral.sh/uv/0.9.7/install.sh']


def probe_proxy(urls, timeout):
    """(ok, detail): every URL answers through the cluster proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': PROXY_ENV['http_proxy'], 'https': PROXY_ENV['https_proxy']}))
    for url in urls:
        try:
            with opener.open(urllib.request.Request(url, headers={'User-Agent': 'synth-proxy-probe'}), timeout=timeout) as r:
                r.read(4096)
        except Exception as e:  # noqa: BLE001
            return False, f'{url}: {e!r}'[:300]
    return True, 'ok'


def baseline_counts(manifests):
    """Usable, complete trajectories per task in an existing corpus (top-up campaigns). Truncated ones (a bank
    record cut before it finished) stay in the corpus but do not count toward a task's target."""
    counts = collections.Counter()
    for path in manifests:
        for record in read(path, {}).get('records', []):
            if record.get('usable') and not record.get('truncated'):
                counts[record['task']] += 1
    return counts


def pick_per_task(tasks, target, have, active, assigned, cap):
    """Next task under its target counting live attempts; fewest first, then fewest tries."""
    best = None
    for order, task in enumerate(tasks):
        projected = have[task] + active[task]
        if projected >= target or assigned[task] >= cap.get(task, math.inf):
            continue
        key = (projected, assigned[task], order)
        if best is None or key < best[0]:
            best = (key, task)
    return best and best[1]


class Controller:
    def __init__(self, campaign):
        self.c = campaign
        self.ops = campaign.ops
        self.state_path = self.ops / 'controller' / 'state.json'
        self.s = read(self.state_path, {'stage': 'smoke', 'items': {}, 'train_cursor': 0, 'workers': {},
                                        'backends': {}, 'train_stop': False, 'created_at': time.time()})
        self.journal = StateJournal(self.ops / 'controller' / 'journal')
        replayed = self.journal.replay(self.s)
        if replayed:
            campaign.log(f'State: replayed {replayed} journal records newer than controller/state.json')
        self.save_pool = ThreadPoolExecutor(max_workers=1)   # state snapshots are written off the tick
        self._save_future = None
        self._last_snapshot = 0
        self._saved_stage = self.s['stage']
        self.queue_view = None   # condor.QueueView, made by main()
        self.host = short_host(socket.gethostname())
        self.slow_ticks = 0
        self.io_errors = []      # times of I/O errors (EIO) this controller hit
        self.indices = {k: read(campaign.root_for(k) / 'export-index.json', {}) for k in ('train', 'eval', 'smoke')}
        self.export_process = None
        self.export_started_at = None
        self.last_export = 0
        self.archive_process = None
        self.last_archive = 0
        self.last_dispatch = 0
        self.stopping = False
        self.errors = 0
        self.loops = 0
        self.timings = {}
        self._heartbeats = None  # worker heartbeats, read once per tick
        self._reuse_seen = {}    # attempt -> cumulative (expected, cached) prompt-cache tokens last seen
        self._direct = {}        # backend -> (time, raw) of the controller's own last /metrics scrape
        self.pool = ThreadPoolExecutor(max_workers=32)  # parallel small-file reads/writes on /data
        train = campaign.config.get('train') or {}
        self.mode = train.get('mode', 'tokens')
        self.baseline = baseline_counts(train.get('baseline_manifests', []))
        for item in self.s['items'].values():  # recorded before WorkerInterrupted existed
            if item['status'] == 'finished' and item.get('interrupted') and not item.get('infrastructure_failure'):
                item['infrastructure_failure'] = 'WorkerInterrupted'

    # ---------------------------------------------------------------- state
    def save(self, force=False):
        """Snapshot the state to controller/state.json (same format) every controller.state_save_interval_sec (60), on a
        stage change, or now (force). It is serialized here (a consistent copy) and written by a background thread, so a
        slow /data write never holds a tick; attempts are durable before that through the journal (durable()). A
        snapshot still being written defers the next one; force waits for it."""
        self.collect_save()
        interval = self.c.config.get('controller', {}).get('state_save_interval_sec', 60)
        if self._save_future is not None:
            if not force:
                return
            self._save_future.result()
            self.collect_save()
        now = time.time()
        if not force and self.s['stage'] == self._saved_stage and now - self._last_snapshot < interval:
            return
        self.s['updated_at'] = now
        self.s['journal_seq'] = self.journal.rotate(self.s)
        text = json.dumps(self.s, ensure_ascii=False) + '\n'
        self._last_snapshot, self._saved_stage = now, self.s['stage']
        seq = self.s['journal_seq']

        def write():
            started = time.time()
            write_text(self.state_path, text)
            return seq, time.time() - started, len(text)
        self._save_future = self.save_pool.submit(write)
        if force:
            self._save_future.result()
            self.collect_save()

    def collect_save(self):
        if self._save_future is None or not self._save_future.done():
            return
        future, self._save_future = self._save_future, None
        try:
            seq, seconds, size = future.result()
        except Exception as exc:  # noqa: BLE001  (the journal keeps everything since the last snapshot)
            self.c.log(f'State snapshot failed ({exc!r}); the journal keeps the attempts since the last one')
            self.note_io_error(exc)
            return
        self.journal.drop_through(seq)
        self.s['last_state_save'] = {'seconds': round(seconds, 2), 'bytes': size, 'journal_seq': seq}

    def durable(self, items):
        """Record new attempts in the journal before their inboxes are published (ownership precedes publication)."""
        self.journal.append(self.s, items)

    def policy(self):
        """Campaign admission defaults, hot-overridable by ops/admission-policy.json."""
        return {**self.c.config['admission'], **read(self.ops / 'admission-policy.json', {})}

    def active(self, kind=None, backend=None):
        return [i for i in self.s['items'].values() if i['status'] == 'active'
                and (kind is None or i['kind'] == kind) and (backend is None or i['backend'] == backend)]

    def assign(self, kind, task, attempt, worker, backend, retry=0, publish=True, **extra):
        """Record an attempt; publish=False leaves saving the state and writing the inbox to the
        caller (the dispatch loop saves once per tick, then publishes every new inbox)."""
        uid = f'{self.c.slug}-{self.c.id}-{kind}-{task}-a{attempt}-r{retry}'
        if uid in self.s['items']:
            raise ValueError(f'Duplicate attempt assignment {uid}')
        if backend.get('dp_size', 1) > 1:  # internal DP: pin the attempt to its least-loaded rank
            load = collections.Counter(i.get('dp_rank') for i in self.active(backend=backend['name']))
            extra['dp_rank'] = min(range(backend['dp_size']), key=lambda r: (load[r], r))
        if extra.get('task_kind', kind) == 'train' and (bundle := self.stream_bundle(task)):
            extra['bundle'] = bundle  # a streamed task: the worker extracts its chunk first
        item = {'id': uid, 'kind': kind, 'task': task, 'attempt': attempt, 'retry': retry,
                'seed': seed(self.c.id, kind, task, attempt), 'worker': worker,
                'worker_job': (self.heartbeats().get(worker) or {}).get('job_id'), 'backend': backend['name'],
                'backend_identity': backend['identity'], 'endpoint': backend['url'],
                'assigned_at': time.time(), 'status': 'active', 'campaign': self.c.id,
                'model': self.c.teacher['model'], 'revision': self.c.teacher['revision'],
                'agent_timeout_sec': self.c.protocol['agent_timeout_sec'], **extra}
        self.s['items'][uid] = item
        if publish:
            self.durable([item])  # durable ownership precedes publication to a worker
            self.publish_inbox(item)
        return uid

    def publish_inbox(self, item):
        atomic(self.ops / 'workers' / str(item['worker']) / 'inbox' / f'{item["id"]}.json', item)

    def raw(self, item):
        return self.c.root_for(item['kind']) / 'raw' / item['id']

    # ------------------------------------------------------------ resources
    def serving_ops(self, receipt):
        """Where a serving job publishes endpoints/metrics/control: ours, or its original campaign's."""
        return Path(receipt['serving_ops']) if receipt and receipt.get('serving_ops') else self.ops

    def adopt_serving(self, q):
        """[serving] adopt_ops: run on another campaign's serving jobs (same index) instead of new ones.
        That campaign must retain them (ops/serving/RETAIN.json) and serve the same teacher and protocol."""
        source = self.c.config['serving'].get('adopt_ops')
        if not source:
            return
        source = Path(source)
        frozen = read(source / 'campaign.json')
        if frozen is None or not same_engine(frozen, self.c.config):
            raise RuntimeError(f'{source} does not serve this campaign\'s teacher spec; refusing to adopt')
        for key in ('context_tokens', 'max_output_tokens'):
            if frozen['protocol'][key] != self.c.protocol[key]:
                raise RuntimeError(f'{source} serves a different {key}; refusing to adopt')
        if not (source / 'serving' / 'RETAIN.json').exists():
            raise RuntimeError(f'{source}/serving/RETAIN.json is missing: that campaign would release its jobs')
        theirs_all = [r for r in (read(p) for p in (source / 'jobs').glob('serving-*-submission.json')) if r]
        live = condor.query(r['job_id'] for r in theirs_all)  # not in our own queue view until adopted
        for index in range(self.c.config['serving']['jobs']):
            mine = condor.latest(self.c, 'serving', index)
            if mine and q.get(mine['job_id'], {}).get('JobStatus') in (1, 2):
                continue  # our own (submitted or adopted) job is alive
            rows = [r for r in theirs_all if r.get('role') == 'serving' and r.get('index', 0) == index]
            theirs = max(rows, key=lambda r: r['generation']) if rows else None
            if not theirs or live.get(theirs['job_id'], {}).get('JobStatus') not in (1, 2):
                continue
            if mine and mine['job_id'] == theirs['job_id'] and not mine.get('campaign_attr'):
                continue  # adopted this tick; our queue view includes it from the next one
            # (a receipt adopted before the campaign_attr fix stays invisible in our queue view: re-adopt it once, at
            # the next generation, with a clean receipt -- a restarted controller on such a campaign would stall)
            # campaign_attr: the source job carries the SOURCE campaign's +SynthCampaign, never ours, so our queue view
            # must name the adopted job by cluster. Kept, it vanished from the view, read as lost after 120 s, and the
            # controller submitted its own serving and re-adopted every ~2.5 min (uni-canary 11-18).
            condor.adopt(self.c, {k: v for k, v in theirs.items() if k not in ('serving_ops', 'adopted_at', 'campaign_attr')},
                         theirs.get('serving_ops') or source, generation=mine['generation'] + 1 if mine else 0)

    def release_idle_serving(self, q):
        """Training has stopped for good and nothing is in flight or left to assign: release serving now. The final
        exports and verification can take an hour at scale and need no GPU. For good: closed (operator, tail rule or
        a reached "saved" budget), or the flushed, finalized corpus meets the target; a stop on live counts can still
        resume when an export is quarantined."""
        if (self.s.get('serving_released') or not self.s['train_stop'] or self.active() or self.pending_work()
                or self.retained_serving()):
            return
        if not (self.train_closed() or (self.training_flush_complete() and self.train_target_reached(finalized_only=True))):
            return
        (self.ops / 'serving').mkdir(parents=True, exist_ok=True)
        (self.ops / 'serving' / 'RELEASE').touch()
        for r in condor.receipts(self.c):
            if r['role'] == 'serving' and r['job_id'] in q:
                condor.release(self.c, r['job_id'])
        self.s['serving_released'] = time.time()
        self.c.log('Released serving: training stopped and no attempt is in flight; exports continue without GPUs.')

    def serving_jobs(self):
        """Serving jobs in use: the campaign's count, or fewer by the hot override ops/admission-policy.json
        serving_jobs (user, 2026-10-26: "if we can't keep up the concurrency, we could just reduce the number of
        gpus"); jobs above it drain and are released (scale_down)."""
        jobs = self.c.config['serving']['jobs']
        return max(1, min(jobs, int(self.policy().get('serving_jobs', jobs))))

    def autoscale_params(self):
        """[serving.autoscale] (synth/autoscale.py), None when off. Owned serving only: a campaign that adopts another
        campaign's serving jobs (adopt_ops) does not scale them."""
        p = autoscale.params(self.c.config['serving'])
        return p if p['enabled'] and not self.c.config['serving'].get('adopt_ops') else None

    def autoscale_state(self, q):
        """The engines the autoscale policy keeps in use (s['autoscale']['wanted']): on the first call, the jobs already
        running below the cap (a controller restart keeps them) or the first `start` indices."""
        a = self.s.get('autoscale')
        if a is None:
            p, cap = self.autoscale_params(), self.serving_jobs()
            alive = [j for j in range(cap) if (r := condor.latest(self.c, 'serving', j)) and r['job_id'] in q
                     and j not in self.s.get('released_serving', [])]
            wanted = alive or list(range(min(cap, p['start'])))
            a = self.s['autoscale'] = {'wanted': sorted(wanted), 'up_since': None, 'down_since': None,
                                       'changed_at': time.time(), 'events': []}
            self.c.log(f'Serving autoscale: {len(wanted)} engines in use at start ({sorted(wanted)}), ceiling {cap} of '
                       f'{self.c.config["serving"]["jobs"]}')
        return a

    def ensure_serving(self, q):
        """(Re)submit the serving jobs in use. Never above the live cap serving_jobs() (#234: a controller restart with
        ceiling 10 and cap 7 started all 10, m1-hero 2026-11-04), never a released index, and under autoscale only the
        indices the policy keeps in use. Jobs already running are left to scale_down (drain, then release)."""
        if self.s.get('serving_released'):
            return
        jobs, cap = self.c.config['serving']['jobs'], self.serving_jobs()
        wanted = set(self.autoscale_state(q)['wanted']) if self.autoscale_params() else None
        self.adopt_serving(q)
        for index in range(jobs):
            if index in self.s.setdefault('released_serving', []):
                continue
            latest = condor.latest(self.c, 'serving', index)
            if latest and latest['job_id'] in q:
                ad = q[latest['job_id']]
                if ad['JobStatus'] == 5:
                    self.c.log(f'Owned serving job {latest["job_id"]} held: {ad.get("HoldReason")}')
                    condor.remove(self.c, latest['job_id'])
                continue
            if index >= cap or (wanted is not None and index not in wanted):
                continue   # above the live cap, or not in use: not submitted
            if latest and time.time() - latest['time'] < 120:
                continue
            if latest:
                self.exclude_failed_preflight(index, latest)
                self.interrupt(lambda i: i['backend'].startswith(f'j{index}r'), 'OwnedServingAllocationLost')
            source = self.c.config['serving'].get('adopt_ops')
            if source and not (Path(source) / 'completion.json').exists():
                # Its source campaign runs and will (re)submit the shared serving job: wait and adopt that one.
                # 2026-10-25: both controllers restarted together; this one found no live source job, submitted its
                # own 4xB200 and ran two servers for a campaign meant to share one.
                if not self.s.get('waiting_for_adopted_serving'):
                    self.s['waiting_for_adopted_serving'] = time.time()
                    self.c.log(f'Serving {index}: waiting for {Path(source).parent.name} to provide the shared job')
                continue
            self.s.pop('waiting_for_adopted_serving', None)
            condor.submit_serving(self.c, index, latest['generation'] + 1 if latest else 0)

    def exclude_failed_preflight(self, index, receipt):
        generation = receipt.get('serving_generation', receipt['generation'])
        report = read(self.serving_ops(receipt) / 'serving' / f'job-{index}' / f'preflight-g{generation}.json', {})
        if report and not report.get('passed') and report.get('host'):
            node = report['host'].split('.')[0]
            excluded = read(self.ops / 'excluded-nodes.json', [])
            if node not in excluded:
                atomic(self.ops / 'excluded-nodes.json', sorted(excluded + [node]))
                self.c.log(f'Excluding node {node} after a failed GPU preflight')

    def backends(self, q, include_stale=False):
        """Healthy replicas of running owned serving jobs at the current control generation. A job whose
        endpoints.json is older than 120 s is left out (no new attempts); include_stale=True keeps its last
        published replicas, so check_backends interrupts attempts only when an engine restarted or its job
        ended, never because the supervisor was slow to publish (a 4-minute stall writing to /data on one
        node once interrupted every attempt on a healthy engine)."""
        ready = {}
        for index in range(self.c.config['serving']['jobs']):
            latest = condor.latest(self.c, 'serving', index)
            if not latest or q.get(latest['job_id'], {}).get('JobStatus') != 2:
                continue
            home = self.serving_ops(latest)
            control = read(home / 'serving' / 'control.json', {}).get('generation', 0)
            published = read(home / 'serving' / f'job-{index}' / 'endpoints.json', {})
            if published.get('generation') != latest.get('serving_generation', latest['generation']):
                continue
            stale = time.time() - published.get('updated_at', 0) > 120
            for r in published.get('replicas', []):
                if r['phase'] != 'healthy' or r.get('control_generation') != control:
                    continue
                # A slow /data write can stall the supervisor's publishing while its engines serve on; a
                # replica whose job is running and whose engine answers /health directly stays ready.
                if stale and not include_stale and not self.engine_alive(r['url']):
                    continue
                ready[r['name']] = {**r, 'identity': f'{latest["job_id"]}/{r["name"]}/{r["healthy_at"]}',
                                    'metrics': str(home / 'metrics' / 'backends' / f'{r["name"]}.json'),
                                    'serving_home': str(home), 'publisher_stale': stale}
        return ready

    def engine_alive(self, url):
        try:
            with OPENER.open(url.removesuffix('/v1') + '/health', timeout=5) as response:
                return response.status == 200
        except OSError:
            return False

    def backend_sample(self, name, backend, now):
        """The serving job's metrics sample for a backend, or, when that is over 100 s old (its supervisor
        is stalled), the controller's own scrape of the engine's /metrics."""
        sample = best_effort(read, backend.get('metrics') or self.ops / 'metrics' / 'backends' / f'{name}.json', {}) or {}
        if now - sample.get('timestamp', 0) <= 100:
            return sample
        from .probe import scrape, server_summary
        raw = scrape(backend['url'].removesuffix('/v1'))
        if not raw:
            return sample
        direct = {'timestamp': now, 'backend': name, 'raw': raw, 'source': 'controller_scrape'}
        previous = self._direct.get(name)
        if previous and now > previous[0]:
            direct.update(server_summary([previous[1]], [raw], now - previous[0]))
        self._direct[name] = (now, raw)
        return direct

    def check_backends(self, ready):
        """Interrupt attempts whose pinned backend restarted or vanished."""
        seen = self.s.setdefault('backend_missing_since', {})
        now = time.time()
        for item in self.active():
            identity = item['backend_identity']
            if ready.get(item['backend'], {}).get('identity') == identity:
                seen.pop(identity, None)
                continue
            seen.setdefault(identity, now)
            if now - seen[identity] > 90:
                self.interrupt(lambda i, identity=identity: i['backend_identity'] == identity, 'OwnedBackendRestart')
        for identity in list(seen):
            if now - seen[identity] > 3600:
                del seen[identity]

    def interrupt(self, predicate, reason):
        for item in self.active():
            if predicate(item) and not item.get('infrastructure_failure'):
                folder = self.raw(item)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'CANCEL').touch()
                item['infrastructure_failure'] = reason
                item['controller_interruption'] = True   # counts toward no per-task cap (failure_phase)
                self.c.log(f'Interrupting {item["id"]}: {reason}')

    def worker_count(self):
        """Worker indices in use: the live override, else the configured count, raised (never lowered:
        scale_worker_fleet retires workers above the target, so every index keeps being supervised and stopped) to
        the highest target scale_workers set, up to workers.max_count."""
        policy = self.policy()
        if 'worker_count' in policy:
            return int(policy['worker_count'])
        return max(self.c.config['workers']['count'], self.s.get('worker_high_water', 0))

    def heartbeats(self):
        """index -> heartbeat of every worker index in use, read in parallel once per tick."""
        if self._heartbeats is None:
            indices = range(self.worker_count())
            home = self.ops / 'workers'
            self._heartbeats = dict(zip(indices, self.pool.map(
                lambda i: best_effort(read, home / str(i) / 'heartbeat.json', {}) or {}, indices)))
        return self._heartbeats

    def cheap_room(self, q, avoid=(), shape=None):
        """New workers that can go out at workers.cheap_bid: the workers of this shape (the tick's filled shape, else
        the configured one) that fit now on nodes whose
        floor is at most the cheap bid (condor.cheap_capacity), at most workers.max_per_host campaign workers per host
        (#244) and none on the avoided hosts, less the cheap-bid workers already waiting for one (#242). 0 if the
        check fails, so those workers go out at workers.bid."""
        w = {**self.c.config['workers'], **(shape or {})}
        exclude = w.get('exclude_nodes', []) + read(self.ops / 'excluded-nodes.json', []) + list(avoid)
        room = best_effort(condor.cheap_capacity, w['cpus'], w['memory_gib'], max_floor=w['cheap_bid'], exclude=exclude,
                           per_host=w.get('max_per_host', 3), placed=condor.role_hosts(self.c, q, 'worker'))
        waiting = sum(1 for r in condor.receipts(self.c) if r['role'] == 'worker' and r['job_id'] in q
                      and q[r['job_id']].get('JobStatus') == 1 and 1000 + q[r['job_id']].get('JobPrio', 0) < w['bid'])
        return max(0, (room or 0) - waiting)

    def worker_avoid(self, q):
        """Hosts a new worker submission avoids besides the excluded nodes: those already running workers.max_per_host
        (3) of the campaign's workers, and the controller's own host. m1-hero (#244): pre-agent failures were 0.4% of
        attempts with one campaign worker on the host, 0.7% with two, 3.6% with three, 8.1% with four, 13.3% with
        five (start-up deadlines missed on hosts whose /data client was saturated)."""
        per_host = self.c.config['workers'].get('max_per_host', 3)
        hosts = condor.role_hosts(self.c, q, 'worker')
        avoid = {h for h, n in hosts.items() if per_host and n >= per_host}
        host = getattr(self, 'host', None)
        if host and self.c.config.get('controller', {}).get('avoid_worker_hosts', True):
            avoid.add(host)
        return sorted(avoid)

    def pending_workers(self, q, start, count, now):
        """How many workers from `start` this tick would submit (no live or recent job, not retired), at most
        workers.max_submissions_per_tick: the batch the tick's fill decision is made for."""
        n = 0
        for index in range(start, count):
            if index in self.s.setdefault('retired_workers', []):
                continue
            latest = condor.latest(self.c, 'worker', index)
            if latest and (latest['job_id'] in q or now - latest['time'] < 90):
                continue
            n += 1
        return min(n, self.c.config['workers'].get('max_submissions_per_tick', 16))

    def ensure_workers(self, q, count):
        now = time.time()
        overrides = read(self.ops / 'worker-resource-overrides.json', {})
        submissions = 0
        cheap_room = None   # checked once per tick, on the first submission
        avoid = None
        filled = None       # the tick's worker shape at the binding resource (condor.worker_fill), on the first one
        for index in range(count):
            if index in self.s.setdefault('retired_workers', []):
                continue
            latest = condor.latest(self.c, 'worker', index)
            if latest and latest['job_id'] in q:
                ad = q[latest['job_id']]
                hb = self.heartbeats().get(index) or read(self.ops / 'workers' / str(index) / 'heartbeat.json', {})
                if ad['JobStatus'] == 5:
                    self.c.log(f'Owned worker {latest["job_id"]} held: {ad.get("HoldReason")}')
                    condor.remove(self.c, latest['job_id'])
                elif ad['JobStatus'] == 2 and hb.get('job_id') == latest['job_id'] and now - hb['updated_at'] > 900:
                    self.c.log(f'Restarting stalled owned worker {latest["job_id"]}')
                    condor.remove(self.c, latest['job_id'])
                elif ad['JobStatus'] == 1 and self.cheap_bid_expired(ad, latest, now):
                    w = self.c.config['workers']
                    best_effort(condor.rebid, self.c, latest['job_id'], w['bid'])   # its cheap node was taken
                    self.c.log(f'Worker {index} ({latest["job_id"]}) idle {now - latest["time"]:.0f} s at bid '
                               f'{1000 + ad["JobPrio"]}: rebid to {w["bid"]}')
                continue
            if latest and now - latest['time'] < 90:
                continue
            if latest:  # Condor confirms the worker ended: settle its attempts first
                for item in self.active():
                    if item['worker'] == index and not (self.raw(item) / 'worker-result.json').exists():
                        best_effort(atomic, self.raw(item) / 'worker-result.json', {'finished_at': now,
                            'interrupted': True, 'infrastructure_failure': 'OwnedWorkerLost', 'job_id': latest['job_id']})
            if submissions >= self.c.config['workers'].get('max_submissions_per_tick', 16):
                continue  # a large scale-up spreads over ticks instead of stalling one
            shape = {k: v for k, v in condor.worker_shape(overrides, index).items()
                     if k in ('slots', 'cpus', 'memory_gib', 'disk_gib')}
            avoid = self.worker_avoid(q) if avoid is None else avoid
            if filled is None:   # once per tick, for the whole batch (placement-aware on the workers' ladder)
                w = self.c.config['workers']
                try:
                    filled = condor.worker_fill(self.c, self.pending_workers(q, index, count, now),
                                                w.get('exclude_nodes', []) + read(self.ops / 'excluded-nodes.json', [])
                                                + avoid) or {}
                except Exception as exc:  # noqa: BLE001  (the fill is best effort; the configured shape is safe)
                    self.c.log(f'Workers: fill decision failed ({exc!r}); configured shape')
                    filled = {}
            if filled and not ({'cpus', 'memory_gib'} & set(shape)):   # an operator's override is kept as given
                shape = {**shape, **filled}
            bid = {}
            if self.c.config['workers'].get('cheap_bid'):
                if cheap_room is None:
                    cheap_room = self.cheap_room(q, avoid, filled)
                    self.c.log(f'Workers: room for {cheap_room} more at bid {self.c.config["workers"]["cheap_bid"]} '
                               f'(CPU-only and A100 nodes, at most {self.c.config["workers"].get("max_per_host", 3)} '
                               f'per host)')
                bid, cheap_room = {'cheap': cheap_room > 0}, cheap_room - 1
            condor.submit_worker(self.c, index, latest['generation'] + 1 if latest else 0, shape, avoid=avoid, **bid)
            submissions += 1

    def cheap_bid_expired(self, ad, receipt, now):
        """An idle worker submitted at workers.cheap_bid (condor.worker_bid) that has waited
        workers.cheap_bid_wait_sec (600) for a cheap node: rebid it to workers.bid."""
        w = self.c.config['workers']
        prio = ad.get('JobPrio')
        return (bool(w.get('cheap_bid')) and prio is not None and 1000 + prio < w['bid']
                and now - receipt['time'] >= w.get('cheap_bid_wait_sec', 600))

    def ready_workers(self, count):
        """(load, index) of workers with a fresh heartbeat from their current job and a free slot. A worker's
        heartbeat capacity (its own --slots) wins over workers.slots, so a slots change applies to new workers
        without overfilling the inboxes of running ones."""
        slots = self.c.config['workers']['slots']
        load = collections.Counter(i['worker'] for i in self.active())
        excluded = set(self.s.get('auto_excluded_hosts', {}))
        ready = []
        for index in range(count):
            # draining (scale_down, #241): no new attempts, retired once its last one ends
            if index in self.s.get('retired_workers', []) or index in self.s.get('draining_workers', []):
                continue
            hb = self.heartbeats().get(index) or {}
            if excluded and short_host(hb.get('host')) in excluded:
                continue  # its job is being removed (exclude_host)
            latest = condor.latest(self.c, 'worker', index)
            if latest and hb.get('job_id') == latest['job_id'] and time.time() - hb.get('updated_at', 0) < 90 \
                    and not hb.get('stopping') and load[index] < (hb.get('capacity') or slots):
                ready.append((load[index], index))
        return sorted(ready)

    # -------------------------------------------------------------- harvest
    def harvest(self):
        """Finished attempts: their worker result, Harbor result and inbox cleanup are read in the thread pool and the
        log lines written in one append. Serially they took 40-150 s per tick at RTS scale (2026-10-26: ~3,000 live
        attempts of a few minutes each, hundreds finishing per tick), which held dispatch and the proxy probes."""
        active = self.active()
        results = list(self.pool.map(lambda i: best_effort(read, self.raw(i) / 'worker-result.json'), active))
        done = [(item, finished) for item, finished in zip(active, results) if finished is not None]

        def load(pair):
            item, finished = pair
            paths = sorted(self.raw(item).glob('harbor/*__*/result.json'))
            result = None
            if len(paths) == 1:
                try:
                    result = read(paths[0])
                except (OSError, ValueError):  # cut mid-write (see common.harbor_result)
                    result = None
            best_effort((self.ops / 'workers' / str(item['worker']) / 'inbox' / f'{item["id"]}.json').unlink,
                        missing_ok=True)
            return paths, result, best_effort(model_requests, self.raw(item))
        lines = []
        harvested = []
        for (item, finished), (paths, result, requests) in zip(done, self.pool.map(load, done)):
            result = result or {}
            reward = (result.get('verifier_result') or {}).get('rewards', {}).get('reward')
            item.update(status='finished', finished_at=time.time(),
                        interrupted=bool(finished.get('interrupted')),
                        reward=reward if isinstance(reward, (float, int)) and math.isfinite(reward) else None,
                        exception_type=(result.get('exception_info') or {}).get('exception_type'),
                        result_path=str(paths[0]) if len(paths) == 1 else None)
            failure = infrastructure_failure(result, finished, requests)
            if failure and not item.get('infrastructure_failure'):  # keep the controller's own reason
                item['infrastructure_failure'] = failure
            if requests is not None:
                item['model_requests'] = requests
            host = short_host(finished.get('host'))
            if host:
                item['host'] = host
            if failure_phase(item):
                item['failure_phase'] = failure_phase(item)
            harvested.append(item)
            lines.append(f'{item["kind"]} finished {item["task"]} a{item["attempt"]} r{item["retry"]} '
                         f'on {item["backend"]}, reward={item["reward"]}')
        self.c.log.many(lines)
        self.host_health(harvested)

    # ------------------------------------------------------- host exclusion
    def host_health(self, harvested):
        """Exclude a worker host after workers.exclude_host_after (20) consecutive pre-agent failures there, across at
        least workers.exclude_host_min_tasks (5) tasks, unless attempts on the other hosts fail before the agent too
        (> 30% of their last 400: a cluster-wide cause). At most workers.max_auto_excluded_hosts (8); 0 disables. The
        host goes into ops/excluded-nodes.json (remove it there to undo) with the evidence in ops/host-exclusions.jsonl;
        its workers' attempts are interrupted (HostExcluded) and the jobs removed, so they come back elsewhere.
        Pilot: gpu090 failed every attempt in ~15 s (259 failures, 60 tasks) until excluded by hand. m1-hero's longest
        streak on a saturated host was 13."""
        w = self.c.config['workers']
        after, min_tasks = w.get('exclude_host_after', 20), w.get('exclude_host_min_tasks', 5)
        limit = w.get('max_auto_excluded_hosts', 8)
        streaks = self.s.setdefault('host_streaks', {})
        recent = self.s.setdefault('recent_host_outcomes', [])
        for item in sorted(harvested, key=lambda i: i.get('finished_at') or 0):
            host, phase = item.get('host'), failure_phase(item)
            if not host or phase == 'interruption' or item['kind'] == 'smoke':
                continue
            pre = phase == 'pre_agent'
            recent.append([host, pre])
            if pre:
                st = streaks.setdefault(host, {'n': 0, 'tasks': [], 'since': item.get('finished_at')})
                st['n'] += 1
                st['tasks'] = (st['tasks'] + [item['task']])[-50:]
                st['last'] = item['id']
            else:
                streaks.pop(host, None)
        del recent[:-400]
        excluded = self.s.setdefault('auto_excluded_hosts', {})
        for host, st in list(streaks.items()):
            if not after or host in excluded or st['n'] < after or len(set(st['tasks'])) < min_tasks:
                continue
            others = [pre for h, pre in recent if h != host]
            if len(others) >= 20 and sum(others) / len(others) > 0.3:
                if not st.get('held_back'):
                    st['held_back'] = time.time()
                    self.c.log(f'Host {host}: {st["n"]} pre-agent failures in a row, but other hosts fail too '
                               f'({sum(others)}/{len(others)}); not excluding it')
                continue
            if len(excluded) >= limit:
                if not st.get('held_back'):
                    st['held_back'] = time.time()
                    self.c.log(f'Host {host}: {st["n"]} pre-agent failures in a row; not excluded '
                               f'(workers.max_auto_excluded_hosts = {limit} reached)')
                continue
            self.exclude_host(host, st)

    def exclude_host(self, host, st):
        nodes = read(self.ops / 'excluded-nodes.json', []) or []
        if host not in nodes:
            atomic(self.ops / 'excluded-nodes.json', sorted(nodes + [host]))
        evidence = {'time': time.time(), 'host': host, 'consecutive_pre_agent_failures': st['n'],
                    'tasks': sorted(set(st['tasks'])), 'since': st.get('since'), 'last_attempt': st.get('last'),
                    'undo': 'remove the host from ops/excluded-nodes.json; workers moved meanwhile stay where they are'}
        workers = []
        for index, hb in self.heartbeats().items():
            latest = condor.latest(self.c, 'worker', index)
            if latest and hb.get('job_id') == latest['job_id'] and short_host(hb.get('host')) == host:
                workers.append(latest['job_id'])
                self.interrupt(lambda i, index=index: i['worker'] == index, 'HostExcluded')
                best_effort(condor.remove, self.c, latest['job_id'])
        evidence['removed_worker_jobs'] = workers
        self.s.setdefault('auto_excluded_hosts', {})[host] = evidence
        best_effort(append_jsonl, self.ops / 'host-exclusions.jsonl', evidence)
        self.c.log(f'Excluding worker host {host}: {st["n"]} pre-agent failures in a row on {len(set(st["tasks"]))} '
                   f'tasks; removed its workers {workers} (undo: ops/excluded-nodes.json)')
        self.s['host_streaks'].pop(host, None)

    # --------------------------------------------------------------- export
    def export(self, force=False):
        """Background incremental export; force=True joins a pass that starts now."""
        if self.export_process is not None and (force or self.export_process.poll() is not None):
            self.collect_export()
        if force:
            self.launch_export()
            self.collect_export()
        elif self.export_process is None and time.time() - self.last_export >= self.c.config.get('export_interval_sec', 90):
            self.launch_export()

    def launch_export(self):
        self.export_started_at = time.time()
        self.export_log = self.ops / 'controller' / 'exports' / f'export-{time.time_ns()}.log'
        self.export_log.parent.mkdir(parents=True, exist_ok=True)
        with self.export_log.open('ab') as output:
            self.export_process = subprocess.Popen([PYTHON, '-u', '-m', 'synth.export', '--kind', 'all'],
                cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO)}, stdout=output, stderr=subprocess.STDOUT)

    def collect_export(self):
        code = self.export_process.wait()
        self.export_process = None
        self.s['last_export_duration_sec'] = time.time() - self.export_started_at
        if code:
            raise RuntimeError(f'Incremental exporter exited {code}; see {self.export_log}')
        self.s['last_export_covered_after'] = self.export_started_at
        self.reload_indices()
        self.last_export = time.time()
        self.manifests()

    def reload_indices(self):
        for kind in ('train', 'eval', 'smoke'):
            self.indices[kind] = read(self.c.root_for(kind) / 'export-index.json', {})

    # -------------------------------------------------------------- archive
    def archive_policy(self):
        """[archive]: attempt directories are packed into shards (synth/archive.py) by default, because ~27-31 files
        per attempt filled the 41.9M-file /data quota on 2026-10-25. incremental: while the campaign runs, a packer
        subprocess takes shard_size finished train attempts whose exports are final (usable, finalized); unusable
        ones are retried by the exporter and wait for completion. pack_on_complete: everything left (train, eval,
        smoke) is packed after completion.json is written."""
        return {'incremental': True, 'pack_on_complete': True, 'shard_size': 1000, 'workers': 4,
                'min_age_sec': 600, 'interval_sec': 300, **(self.c.config.get('archive') or {})}

    def archive_ready(self, policy):
        """Finished train attempts with a final usable export, finished min_age_sec ago and not packed yet."""
        packed, index, now = archive.load_index(self.c.root, 'train')['attempts'], self.indices['train'], time.time()
        return sorted(i['id'] for i in self.s['items'].values()
                      if i['kind'] == 'train' and i['status'] == 'finished' and i['id'] not in packed
                      and now - i.get('finished_at', now) >= policy['min_age_sec']
                      and (index.get(i['id']) or {}).get('usable') and (index.get(i['id']) or {}).get('finalized')
                      and (self.raw(i) / 'worker-result.json').exists())

    def archive_tick(self):
        policy = self.archive_policy()
        if not policy['incremental']:
            return
        if self.archive_process is not None:
            if self.archive_process.poll() is None:
                return
            self.collect_archive()
        if time.time() - self.last_archive < policy['interval_sec']:
            return
        self.last_archive = time.time()
        ready = self.archive_ready(policy)
        if len(ready) < policy['shard_size']:
            return
        take = ready[:policy['shard_size'] * max(1, len(ready) // policy['shard_size'])][:policy['shard_size'] * policy['workers']]
        self.launch_archive(take, policy)

    def launch_archive(self, uids, policy):
        folder = self.ops / 'archive'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        only, self.archive_log = folder / f'pack-{stamp}.json', folder / f'pack-{stamp}.log'
        atomic(only, uids)
        with self.archive_log.open('ab') as output:
            self.archive_process = subprocess.Popen(
                [PYTHON, '-u', '-m', 'synth.archive', 'pack', str(self.c.root), '--kind', 'train', '--only', str(only),
                 '--shard-size', str(policy['shard_size']), '--workers', str(policy['workers']), '--delete'],
                cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO)}, stdout=output, stderr=subprocess.STDOUT)
        self.c.log(f'Archive: packing {len(uids)} finished attempts (log {self.archive_log.name})')

    def collect_archive(self):
        code = self.archive_process.wait()
        self.archive_process = None
        if code:
            try:  # m1-hero (#235): 4 packs failed with EIO on gpu166 while the controller ran there
                with open(self.archive_log, 'rb') as f:
                    f.seek(max(0, f.seek(0, 2) - 20000))
                    tail = f.read().decode(errors='replace')
                if 'Input/output error' in tail or '[Errno 5]' in tail:
                    self.note_io_error(OSError(errno.EIO, 'archive packer'))
            except OSError:
                pass
            raise RuntimeError(f'Archive packer exited {code}; see {self.archive_log}')

    def close_archive(self):
        if self.archive_process is not None and self.archive_process.poll() is None:
            self.archive_process.terminate()   # an unfinished shard leaves only .partial files; nothing is deleted
            try:
                self.archive_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.archive_process.kill()
                self.archive_process.wait()

    def pack_completed(self):
        """pack_on_complete: pack every remaining attempt directory of a completed campaign, in this job."""
        policy = self.archive_policy()
        if not policy['pack_on_complete']:
            return
        if self.archive_process is not None:
            self.collect_archive()
        workers = max(1, len(os.sched_getaffinity(0)) - 1)
        for kind in ('train', 'eval', 'smoke'):
            if (self.c.root_for(kind) / 'raw').exists():
                try:
                    archive.pack_many([self.c.root], kind, policy['shard_size'], workers, delete=True,
                                      log=lambda m: self.c.log(f'Archive: {m.strip()}'), campaign=self.c.config)
                except Exception as exc:   # the campaign is complete; leftovers can be packed with synth.archive
                    self.c.log(f'Archive: packing {kind} failed ({exc!r}); its attempt dirs stay as they are')

    def close_export(self):
        if self.export_process is not None and self.export_process.poll() is None:
            self.export_process.terminate()
            try:
                self.export_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.export_process.kill()
                self.export_process.wait()

    # ------------------------------------------------------------ accounting
    def totals(self):
        index = self.indices['train']
        key = (id(index), len(index))  # the index is replaced (never edited) by each export
        if getattr(self, '_totals', (None,))[0] != key:
            valid = [v for v in index.values() if v.get('usable')]
            self._totals = (key, (sum(v['tokens'] for v in valid), len(valid),
                                  sum(v.get('finalized', False) for v in valid)))
        return self._totals[1]

    def counted(self, record, finalized_only=False):
        """A new usable trajectory counts toward per-task targets once its attempt finished, interrupted or not
        (train.count_interrupted_usable, default true), so a task gets exactly trajectories_per_task usable ones.
        m1-hero (#244): not counting them gave 382 tasks 597 usable trajectories beyond M = 2 (26M tokens), mostly
        from attempts mislabelled as infrastructure failures. false: only attempts without an infrastructure failure
        count (the rest stay in the corpus)."""
        item = self.s['items'].get(record['id'])
        if not (record.get('usable') and (record.get('finalized') or not finalized_only) and item is not None
                and item['status'] == 'finished'):
            return False
        return not item.get('infrastructure_failure') or (self.c.config.get('train') or {}).get('count_interrupted_usable', True)

    def train_counts(self, finalized_only=False):
        """Counted trajectories per task: baseline corpus + this campaign's exports."""
        counts = collections.Counter(self.baseline)
        for record in self.indices['train'].values():
            if self.counted(record, finalized_only):
                counts[record['task']] += 1
        return counts

    # --------------------------------------------------------------- stream
    def stream_config(self):
        """[train.stream] (synth/stream.py): tasks arrive in chunks while the campaign runs. None: frozen task set."""
        return (self.c.config.get('train') or {}).get('stream') if self.c.enabled('train') else None

    def train_list(self):
        """Train tasks in campaign order: the frozen list, then the tasks of admitted stream chunks."""
        streamed = (self.s.get('stream') or {}).get('tasks')
        if not streamed:
            return self.c.tasks('train')
        if getattr(self, '_train_list', (None,))[0] != len(streamed):
            self._train_list = (len(streamed), self.c.tasks('train') + streamed)
        return self._train_list[1]

    def stream_open(self):
        """More stream chunks will come: no END.json yet, or chunks it counts are not admitted yet. Open-stream
        campaigns are never done, never a tail and never drain serving (2026-10-31: stage 2 of the task-synthesis
        proof of concept cut 745 live attempts when its list ran out before its budget)."""
        if not self.stream_config():
            return False
        st = self.s.get('stream') or {}
        end = st.get('end')
        return end is None or len(st.get('chunks', [])) < end['chunks']

    def stream_wanted(self, have=None):
        """The stream's next chunk could still be used: it is open, and the token budget (if any) still opens tasks.
        A budget that no longer opens tasks ("projected": the opened tasks' trajectories fill it) outranks the stream,
        as it does a list that is not exhausted."""
        return self.stream_open() and (self.budget() is None or self.budget_open(have))

    def stream_tick(self):
        """Admit the next committed stream chunk (synth.stream.admit: check, rewrite agent limits, copy to ops/stream)
        in the thread pool, one at a time in index order; poll the stream every stream.poll_sec (60), at once after an
        admission (a backlog). ops/stream/skip.json ({"<index>": reason}, operator) passes over a chunk that cannot be
        admitted (e.g. its tar no longer matches its record), so the chunks after it and END.json still count."""
        config = self.stream_config()
        if not config:
            return
        st = self.s.setdefault('stream', {'dir': config['dir'], 'chunks': [], 'tasks': [], 'end': None})
        future = getattr(self, '_stream_future', None)
        if future is not None:
            if not future.done():
                return
            self._stream_future = None
            try:
                self.stream_admitted(future.result())
                self._stream_polled = 0
            except Exception as exc:  # noqa: BLE001  (retried at the next poll; progress.json shows the error)
                st['error'] = {'time': time.time(), 'chunk': len(st['chunks']), 'error': repr(exc)[:500]}
                self.c.log(f'Stream: admitting chunk {len(st["chunks"])} failed ({exc!r}); retrying '
                           f'(ops/stream/skip.json passes over it)')
        if time.time() - getattr(self, '_stream_polled', 0) < config.get('poll_sec', 60):
            return
        self._stream_polled = time.time()
        stream = Path(config['dir'])
        st['end'] = read(stream / 'END.json')
        if self.s.get('train_starved_since') and self.stream_open() and time.time() - st.get('waiting_logged', 0) > 1800:
            st['waiting_logged'] = time.time()
            last = max((c['admitted_at'] for c in st['chunks']), default=None)
            self.c.log(f'Stream: nothing to dispatch, waiting for chunk {len(st["chunks"])} '
                       f'(last chunk admitted {f"{(time.time() - last) / 60:.0f} min ago" if last else "never"})')
        index = len(st['chunks'])
        if st['end'] is not None and index >= st['end']['chunks']:
            return
        if (reason := (read(self.ops / 'stream' / 'skip.json', {}) or {}).get(str(index))) is not None:
            st['chunks'].append({'index': index, 'count': 0, 'skipped': reason, 'admitted_at': time.time()})
            st.pop('error', None)
            self.write_stream_chunks()
            self._stream_polled = 0
            self.c.log(f'Stream: chunk {index} skipped (ops/stream/skip.json: {reason})')
            return
        record = read(stream / 'chunks' / f'{index:04d}.json')
        if record is not None:
            self._stream_future = self.pool.submit(stream_chunks.admit, self.c, record, stream)

    def stream_admitted(self, audit):
        """Append an admitted chunk's tasks to the train list (a task already listed keeps its place) and record the
        chunk in the state and in ops/stream/chunks.json."""
        st = self.s['stream']
        if audit['chunk'] != len(st['chunks']):
            raise ValueError(f'stream chunk {audit["chunk"]} admitted out of order (next is {len(st["chunks"])})')
        known = set(self.train_list())
        new = [t for t in audit['tasks_order'] if t not in known]
        st['chunks'].append({'index': audit['chunk'], 'count': len(new), 'invalid': audit.get('invalid', {}),
                             'duplicates_skipped': len(audit['tasks_order']) - len(new), 'tar': audit['path'],
                             'sha256': audit['sha256'], 'source_tar': audit['source'],
                             'source_sha256': audit['source_sha256'], 'source': audit['commit'].get('source'),
                             'published_at': audit['commit'].get('created_at'), 'admitted_at': time.time()})
        st['tasks'].extend(new)
        st.pop('error', None)
        self.write_stream_chunks()
        self.c.log(f'Stream: admitted chunk {audit["chunk"]} ({len(new)} tasks'
                   + (f', {len(audit["tasks_order"]) - len(new)} already listed' if len(new) < len(audit['tasks_order'])
                      else '')
                   + (f', {len(audit["invalid"])} left out as invalid: {sorted(audit["invalid"])[:5]}'
                      if audit.get('invalid') else '') + f'); {len(self.train_list())} train tasks')

    def write_stream_chunks(self):
        """ops/stream/chunks.json: every admitted (or skipped) chunk with its source and sha256."""
        st = self.s['stream']
        atomic(self.ops / 'stream' / 'chunks.json', {'stream': st['dir'], 'chunks': st['chunks'], 'end': st['end'],
                                                    'tasks': len(st['tasks']), 'updated_at': time.time()})

    def stream_bundle(self, task):
        """The admitted chunk tar holding a streamed task (its worker extracts it), or None for frozen tasks."""
        st = self.s.get('stream') or {}
        chunks = st.get('chunks', [])
        if getattr(self, '_stream_bundles', (None,))[0] != len(chunks):
            bundles, offset = {}, 0
            for chunk in chunks:
                bundles.update(dict.fromkeys(st['tasks'][offset:offset + chunk['count']], chunk.get('tar')))
                offset += chunk['count']
            self._stream_bundles = (len(chunks), bundles)
        return self._stream_bundles[1].get(task)

    def stream_summary(self):
        st = self.s.get('stream')
        if not st:
            return None
        return {'dir': st['dir'], 'chunks': len(st['chunks']), 'tasks': len(st['tasks']), 'end': st['end'],
                'open': self.stream_open(), 'error': st.get('error'),
                'last_admitted_at': max((c['admitted_at'] for c in st['chunks']), default=None)}

    def budget(self):
        return (self.c.config.get('train') or {}).get('tokens_budget') if self.mode == 'per_task' else None

    def train_tasks(self):
        """per_task tasks in play: all of them, or under train.tokens_budget the tasks opened so far, in
        the order of the (pre-shuffled) task list. Every opened task gets its full trajectories_per_task."""
        return self.s.get('opened', []) if self.budget() else self.train_list()

    def validation(self):
        """train.validation: {task: true|false} from an environment validation sweep that runs ahead of
        the campaign; a task opens only once validated, failed ones are skipped. None: no gate."""
        path = (self.c.config.get('train') or {}).get('validation')
        if not path:
            return None
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            return {}
        if getattr(self, '_validation', (None, None))[0] != mtime:
            self._validation = (mtime, read(path, {}))
        return self._validation[1]

    def budget_exhausted(self):
        """Every list position is opened or invalid, and no stream chunk is still to come."""
        return self.s.get('budget_cursor', 0) >= len(self.train_list()) and not self.stream_open()

    def open_next_task(self, budget_open):
        """Open the earliest valid, unopened task among the next train.validation_window positions
        after the first unresolved one, so a slow validation does not stall the campaign. A valid task
        behind the furthest opened position opens even after the budget closed: the subset stays
        exactly "every valid task before position P" of the shuffled list."""
        tasks, verdicts = self.train_list(), self.validation()
        opened = set(self.s.setdefault('opened', []))
        invalid = set(self.s.setdefault('invalid', []))
        window = self.c.config['train'].get('validation_window', 256)
        cursor, frontier = self.s.get('budget_cursor', 0), self.s.get('frontier', 0)
        choice = None
        for position in range(cursor, min(len(tasks), cursor + window)):
            task = tasks[position]
            if task in opened or task in invalid:
                continue
            verdict = True if verdicts is None else verdicts.get(task)
            if verdict is False:
                self.s['invalid'].append(task)
                invalid.add(task)
            elif verdict and choice is None and (position < frontier or budget_open):
                choice = position
        if choice is not None:
            self.s['opened'].append(tasks[choice])
            opened.add(tasks[choice])
            self.s['frontier'] = max(frontier, choice + 1)
        while cursor < len(tasks) and (tasks[cursor] in opened or tasks[cursor] in invalid):
            cursor += 1
        self.s['budget_cursor'] = cursor
        frontier = self.s.get('frontier', 0)
        self.s['skipped_invalid'] = sum(self.position(t) < frontier for t in invalid)
        return tasks[choice] if choice is not None else None

    def position(self, task):
        tasks = self.train_list()
        if getattr(self, '_positions', (None,))[0] != len(tasks):  # a stream chunk extends the list
            self._positions = (len(tasks), {t: i for i, t in enumerate(tasks)})
        return self._positions[1][task]

    def budget_pending(self):
        """Positions before the furthest opened one still awaiting validation (or opening)."""
        return self.s.get('budget_cursor', 0) < self.s.get('frontier', 0)

    def trajectory_tokens_estimate(self):
        """Expected saved tokens of one more trajectory. Early finishers are the short ones (the Super
        top-up's running mean was 0.62x its final mean at 50% finished, 0.93x at 90%), and the exports lag the
        harvest by up to an hour and cover the early finishers first, so the measured (exported) mean is used only
        once 90% of started attempts are exported; before that it is floored by the prior. Gating on finished
        attempts instead let the 10-24 TMax hero switch to a 9.4k exported mean (final: 13.96k) at 93% finished."""
        prior = self.c.config['train'].get('tokens_per_trajectory_prior', 60000)
        tokens, n, _ = self.totals()
        if not n:
            return prior
        started = sum(1 for i in self.s['items'].values() if i['kind'] == 'train')
        return tokens / n if n >= 0.9 * started else max(tokens / n, prior)

    def unexported_finished(self):
        """Counted-to-be train attempts that finished but are not exported yet (exports lag the harvest)."""
        return [i for i in self.s['items'].values() if i['kind'] == 'train' and i['status'] == 'finished'
                and not i.get('infrastructure_failure') and i['id'] not in self.indices['train']]

    def optimistic_counts(self):
        """train_counts plus finished attempts not yet exported (a quarantined export drops out later)."""
        have = self.train_counts()
        for item in self.unexported_finished():
            have[item['task']] += 1
        return have

    def estimated_saved(self):
        """Saved tokens now: exported tokens plus the finished, not yet exported trajectories at the running mean
        (an export pass takes ~30 min at 35k trajectories; waiting for it would overshoot the budget). Cached for the
        tick (running() resets it; harvest and export run before it)."""
        index = self.indices['train']
        key = (getattr(self, 'loops', None), id(index), len(index))
        cached = getattr(self, '_saved_cache', None)
        if cached is None or cached[0] != key:
            cached = self._saved_cache = (key, self.totals()[0] + len(self.unexported_finished()) * self.trajectory_tokens_estimate())
        return cached[1]

    def projected_tokens(self, have=None):
        """Saved tokens (estimated) plus the expected tokens of trajectories still owed to opened tasks. `have` must
        count finished but unexported trajectories (train_plan's counts do): they count once, as saved. Counting
        them as done but at zero tokens undercounted the 2026-10-23 TMax campaign by ~45% (297M projected vs ~540M
        real) while exports lagged."""
        target = self.c.config['train']['trajectories_per_task']
        have = have if have is not None else self.optimistic_counts()
        owed = sum(max(0, target - have[t]) for t in self.train_tasks())
        return self.estimated_saved() + owed * self.trajectory_tokens_estimate()

    def budget_policy(self):
        """train.budget_stop: "saved" (default; user, 2026-10-24: "keep running at max concurrency until we hit the
        desired token budget; always max concurrency"): tasks keep opening while the saved tokens are under the
        budget, serving is never drained for a tail, and training closes the moment the budget is reached (live
        attempts are cut, so the last tasks may end below trajectories_per_task). "projected": open tasks only while
        the projected tokens of every opened task fit, and let them finish (the tail rule cuts the end)."""
        return (self.c.config.get('train') or {}).get('budget_stop', 'saved')

    def budget_open(self, have=None):
        """True while more tasks may be opened under the token budget."""
        if self.budget() is None or self.budget_exhausted() or self.s.get('budget_closed'):
            return False
        if self.budget_policy() == 'saved':
            return self.estimated_saved() < self.budget()
        return self.projected_tokens(have) < self.budget()

    def max_concurrency_open(self):
        """Training under the "saved" budget policy is open: keep every backend at its admission limit."""
        return (self.c.enabled('train') and self.budget() is not None and self.budget_policy() == 'saved'
                and not self.s['train_stop'])

    def train_closed(self):
        """Operator file ops/CLOSE_TRAIN.json, the tail rule, or a reached "saved" token budget: training ends with
        what it has (e.g. a single long-tail attempt holding a campaign open). Live attempts are interrupted and kept
        as truncated data; completion.json records the early close and any tasks below target."""
        return read(self.ops / 'CLOSE_TRAIN.json') or self.s.get('tail_closed') or self.s.get('budget_closed')

    def train_target_reached(self, finalized_only=False):
        train = self.c.config.get('train') or {}
        if not self.c.enabled('train'):
            return True
        if self.train_closed():
            return True
        if self.mode == 'tokens':
            tokens = self.totals()[0]
            return tokens >= train['tokens_target']
        target = train['trajectories_per_task']
        counts = self.train_counts(finalized_only)
        caps = self.attempt_caps()
        # finished tries only: a task whose LAST allowed try is still running has not reached its cap. Counting it
        # (as dispatch must, so it never over-issues) closed training at campaign tails and interrupted those live
        # tries (TrainingTargetReached, which counts as no try), so the reopen check would re-dispatch them in a loop
        # (uni-canary 11-18, max_extra_attempts_per_task = 0: all 10 first attempts cut 14 s in).
        tried = self.tried(finished_only=True)
        if self.budget() and (self.budget_pending() or (self.totals()[0] < self.budget() and not self.budget_exhausted())):
            return False
        if self.stream_wanted():  # more tasks are coming
            return False
        return all(counts[t] >= target or tried[t] >= caps[t] for t in self.train_tasks())

    def attempt_caps(self):
        train = self.c.config['train']
        extra = train.get('max_extra_attempts_per_task', 4)
        target = train['trajectories_per_task']
        return {t: max(0, target - self.baseline[t]) + extra for t in self.train_list()}

    def infrastructure_cap(self):
        return self.policy().get('max_infrastructure_failures_per_task',
                                 self.c.config['train'].get('max_infrastructure_failures_per_task', 4))

    def tried(self, finished_only=False):
        """Train attempts per task that count against attempt_caps (finished_only: running ones excluded). Infrastructure failures do not
        (a worker whose scratch filled used up 12 tries per task in seconds); pre-agent and mid-run ones have
        their own cap, train.max_infrastructure_failures_per_task (default 4; it was 32, and m1-hero ran 8 tasks
        with deterministic build or agent-install failures 24-33 times, #244), after which the task stops. The
        controller's own interruptions (failure_phase 'interruption') count toward neither."""
        cap = self.infrastructure_cap()
        tried, infra = collections.Counter(), collections.Counter()
        for i in self.s['items'].values():
            if i['kind'] == 'train' and not (finished_only and i['status'] != 'finished'):
                phase = failure_phase(i)
                if phase is None:
                    tried[i['task']] += 1
                elif phase != 'interruption':
                    infra[i['task']] += 1
        blocked = self.blocked()
        return collections.Counter({t: (math.inf if infra[t] >= cap or t in blocked else tried[t])
                                    for t in set(tried) | set(infra) | blocked})

    def blocked(self):
        """ops/blocked-tasks.json ({task: reason} or [task, ...]): tasks that get no more attempts; live ones are
        cancelled (running). RTS 2026-10-26: tasks whose image build ate a worker's memory (no guard during builds)
        were retried on worker after worker, each loss holding a 48-slot worker."""
        path = self.ops / 'blocked-tasks.json'
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return set()
        if getattr(self, '_blocked', (None, None))[0] != mtime:
            data = read(path, {}) or {}
            self._blocked = (mtime, set(data) if isinstance(data, (list, dict)) else set())
        return self._blocked[1]

    def train_plan(self):
        """The counters next_train needs, built once per dispatch tick (each is a pass over every
        item) and updated by every pick, instead of rebuilt for each assignment."""
        # Attempt numbers (uid, seed) count every issued attempt, infrastructure failures included.
        plan = {'issued': collections.Counter(i['task'] for i in self.s['items'].values() if i['kind'] == 'train')}
        if self.mode != 'tokens':
            plan.update(tried=self.tried(), caps=self.attempt_caps())
            have = self.train_counts()
            # A finished attempt not yet exported is optimistically counted; a
            # quarantined export drops out and its task is rescheduled.
            for item in self.s['items'].values():
                if (item['kind'] == 'train' and item['status'] == 'finished' and not item.get('infrastructure_failure')
                        and item['id'] not in self.indices['train']):
                    have[item['task']] += 1
            plan.update(have=have, active=collections.Counter(i['task'] for i in self.active('train')))
        return plan

    def next_train(self, plan=None):
        """(task, attempt) of the next training attempt; the caller assigns it."""
        plan = plan if plan is not None else self.train_plan()
        tasks = self.train_list()
        if self.mode == 'tokens':
            cursor = self.s['train_cursor']
            if not tasks or (cursor >= len(tasks) and self.stream_open()):
                return None  # the stream's next chunk comes before any task is cycled again
            self.s['train_cursor'] += 1
            task = tasks[cursor % len(tasks)]
        else:
            target = self.c.config['train']['trajectories_per_task']
            have = plan['have']
            task = pick_per_task(self.train_tasks(), target, have, plan['active'], plan['tried'], plan['caps'])
            if task is None and self.budget():
                task = self.open_next_task(self.budget_open(have))
                if task and len(self.s['opened']) % 25 == 1:
                    self.c.log(f'Token budget: opened task {len(self.s["opened"])} ({task}, {self.s.get("skipped_invalid", 0)} '
                               f'skipped as invalid); projected {self.projected_tokens(have) / 1e6:.1f}M of {self.budget() / 1e6:.0f}M')
            if task is None and self.max_concurrency_open() and self.budget_open(have):
                # Nothing to open (validation behind, list exhausted) but the budget is open: keep the GPUs full with
                # up to train.overflow_per_task (1) extra trajectories per opened task instead of idling (user,
                # 2026-10-24: "for some envs we might sample 9 rather than 8 trajs, this is fine").
                extra = self.c.config['train'].get('overflow_per_task', 1)
                task = pick_per_task(self.train_tasks(), target + extra, have, plan['active'], plan['tried'], plan['caps'])
            if task is None:
                return None
            plan['active'][task] += 1
            plan['tried'][task] += 1
        attempt = plan['issued'][task]
        plan['issued'][task] += 1
        return task, attempt

    def resolved_eval(self):
        resolved = {}
        for item in self.s['items'].values():
            ex = self.indices['eval'].get(item['id'], {})
            if (item['kind'] == 'eval' and not item.get('infrastructure_failure') and item.get('reward') is not None
                    and ex.get('usable') and ex.get('finalized')):
                key = (item['task'], item['attempt'])
                if key in resolved:
                    raise ValueError(f'Duplicate scored evaluation attempt {key}')
                resolved[key] = item
        return resolved

    def eval_candidates(self):
        """(task, attempt, retry) of every eval attempt that could still be started."""
        if not self.c.enabled('eval') or self.s.get('tail_closed'):
            return
        resolved = self.resolved_eval()
        max_generations = self.c.config['eval'].get('max_generations', 3)
        generations = collections.defaultdict(list)
        for i in self.s['items'].values():
            if i['kind'] == 'eval':
                generations[i['task'], i['attempt']].append(i)
        for attempt in range(self.c.eval_attempts):
            for task in self.c.tasks('eval'):
                if (task, attempt) in resolved:
                    continue
                prior = generations.get((task, attempt), [])
                if any(i['status'] == 'active' for i in prior):
                    continue
                # A real finite reward is never discarded or rerolled.
                if any(i.get('reward') is not None and not i.get('infrastructure_failure') for i in prior):
                    continue
                if prior and not all(i.get('infrastructure_failure') for i in prior):
                    continue  # e.g. a verifier timeout: final, unscored
                if len(prior) >= max_generations:
                    continue
                yield task, attempt, len(prior)

    def next_eval(self):
        return next(self.eval_candidates(), None)

    # ------------------------------------------------------------------ tail
    def tail_policy(self):
        """The last few attempts of a campaign are not worth holding its GPUs for (user, 2026-10-18).
        Overridable in [tail] and live in ops/tail-policy.json."""
        return {'idle_fraction': 0.10, 'hold_minutes': 20, 'max_remaining_fraction': 0.03,
                'starved_minutes': 5, 'starved_fraction': 0.5, 'starved_list_fraction': 0.99,
                **(self.c.config.get('tail') or {}), **read(self.ops / 'tail-policy.json', {})}

    def remaining_work(self):
        """(attempts still to start, attempts planned) over training and evaluation."""
        planned = self.c.eval_expected
        remaining = sum(1 for _ in self.eval_candidates())
        if self.c.enabled('train') and not self.s['train_stop']:
            if self.mode == 'tokens' or self.budget_open() or self.budget_pending() or self.stream_wanted():
                return math.inf, planned
            target = self.c.config['train']['trajectories_per_task']
            have = self.train_counts()
            active = collections.Counter(i['task'] for i in self.active('train'))
            tried, caps = self.tried(), self.attempt_caps()
            planned += target * len(self.train_tasks())
            remaining += sum(max(0, min(target - have[t] - active[t], caps[t] - tried[t])) for t in self.train_tasks())
        return remaining, planned

    def tail_check(self, status):
        """Close a campaign whose GPUs only serve its tail: at most idle_fraction of the serving
        capacity in flight and at most max_remaining_fraction of the planned attempts left to start,
        for hold_minutes. Live attempts are cut (kept as truncated data); unresolved evaluation
        attempts are reported as such; the campaign finalizes and releases its serving."""
        capacity = sum(v['limit'] for v in status.values())
        if self.s.get('tail_closed') or not capacity:
            return
        if self.serving_shared():
            # A live adopter runs on these GPUs, so they serve more than this tail: let the stragglers finish
            # (2026-10-25: …-calibforge-slots48 cut 911 live attempts while its successor was taking the server).
            self.s.pop('tail_idle_since', None)
            return
        policy = self.tail_policy()
        active = len(self.active())
        starved = self.s.get('train_starved_since')
        resolved = (len(self.s.get('opened', [])) + len(self.s.get('invalid', []))) / max(1, len(self.train_list()))
        if (starved and self.max_concurrency_open() and resolved >= policy['starved_list_fraction']
                and not self.stream_wanted()   # an open task stream is more work coming, not a tail
                and time.time() - starved >= policy['starved_minutes'] * 60
                and active <= policy['starved_fraction'] * capacity):
            # A "saved" budget that the task list cannot reach (all valid tasks opened, overflow spent) keeps
            # remaining_work() infinite, so the rule below never fires: the 10-24 TMax hero drained 3.4k -> 1.1k
            # agents on 12 B200s with the budget open. With the list resolved (opened or invalid; a lagging
            # validation early on is not a tail), no work for starved_minutes and the load below starved_fraction
            # of capacity: cut the tail now (user: max concurrency until the budget, no tails).
            self.s['tail_closed'] = {'time': time.time(), 'active': active, 'capacity': capacity,
                                     'starved_since': starved, 'policy': policy}
            self.c.log(f'Tail rule: nothing to dispatch for {(time.time() - starved) / 60:.0f} min under an open '
                       f'token budget, {active} attempts in flight on capacity {capacity}; cutting the tail.')
            self.interrupt(lambda i: True, 'TailCut')
            return
        remaining, planned = self.remaining_work()
        if active > policy['idle_fraction'] * capacity or remaining > policy['max_remaining_fraction'] * max(planned, 1):
            self.s.pop('tail_idle_since', None)
            return
        since = self.s.setdefault('tail_idle_since', time.time())
        if time.time() - since < policy['hold_minutes'] * 60:
            return
        self.s['tail_closed'] = {'time': time.time(), 'active': active, 'capacity': capacity,
                                 'remaining': remaining, 'planned': planned, 'policy': policy}
        self.c.log(f'Tail rule: {active} attempts in flight on capacity {capacity}, {remaining} of {planned} '
                   f'planned left to start; cutting the tail and finalizing.')
        self.interrupt(lambda i: True, 'TailCut')

    def training_flush_complete(self):
        if self.active('train'):
            return False
        finished = max((i['finished_at'] for i in self.s['items'].values() if i['kind'] == 'train'), default=0)
        return (self.s.get('last_export_covered_after', 0) >= finished
                and all(ex.get('finalized') for ex in self.indices['train'].values()))

    def manifests(self):
        tokens, n, finalized = self.totals()
        common = {'campaign': self.c.id, 'teacher': self.c.slug, 'model': self.c.teacher['model'],
                  'revision': self.c.teacher['revision'], 'protocol': str(self.ops / 'protocol.json'),
                  'updated_at': time.time(),
                  'token_accounting': 'unique native Pi source graph; repeated request prefixes excluded'}
        if self.c.enabled('train'):
            train = self.c.config['train']
            counts = self.train_counts(finalized_only=True)
            new = collections.Counter(r['task'] for r in self.indices['train'].values()
                                      if self.counted(r, finalized_only=True))
            stream = self.stream_summary()  # with every admitted chunk's source and sha256
            atomic(self.c.train / 'manifest.json', {**common, 'kind': 'train', 'mode': self.mode,
                'tokens_target': train.get('tokens_target'), 'trajectories_per_task': train.get('trajectories_per_task'),
                'tokens_budget': train.get('tokens_budget'), 'opened_tasks': len(self.train_tasks()),
                'skipped_invalid_tasks': self.s.get('skipped_invalid', 0), 'validation': train.get('validation'),
                'list_positions_covered': self.s.get('frontier', 0),
                'train_tokens_saved': tokens, 'trajectory_count': n, 'finalized_trajectory_count': finalized,
                'baseline_manifests': train.get('baseline_manifests', []),
                'per_task': {t: {'baseline': self.baseline[t], 'new': new[t], 'total': counts[t]}
                             for t in self.train_tasks()},
                'tasks': self.train_tasks(), 'records': list(self.indices['train'].values()),
                'excluded': {'eval': True, 'smoke': True},
                'stream': stream and {**stream, 'chunks': self.s['stream']['chunks']}})
        if not self.c.enabled('eval'):
            return
        records, resolved = [], self.resolved_eval()
        for task in self.c.tasks('eval'):
            for attempt in range(self.c.eval_attempts):
                histories = [i for i in self.s['items'].values()
                             if i['kind'] == 'eval' and i['task'] == task and i['attempt'] == attempt]
                canonical = resolved.get((task, attempt))
                inventory = self.c.eval / 'inventories'
                records.append({'task': task, 'attempt': attempt, 'seed': seed(self.c.id, 'eval', task, attempt),
                    'resolved': canonical is not None, 'canonical_id': canonical and canonical['id'],
                    'reward': canonical and canonical['reward'],
                    'generations': [{**i, 'trajectory_export': self.indices['eval'].get(i['id']),
                        'artifact_inventory': str(inventory / f'{i["id"]}.json')} for i in histories]})
        atomic(self.c.eval / 'manifest.json', {**common, 'kind': 'eval', 'attempts_per_task': self.c.eval_attempts,
            'attempts_expected': self.c.eval_expected, 'attempts_finished': len(resolved),
            'tasks': self.c.tasks('eval'), 'records': records, 'training_tokens_contributed': 0,
            'retry_policy': 'Retry only explicit infrastructure failures with the same per-attempt seed; keep all originals; never select by reward.'})
        self.score(resolved)

    def score(self, resolved):
        tasks, attempts = self.c.tasks('eval'), self.c.eval_attempts
        expected = len(tasks) * attempts
        per = []
        for a in range(attempts):
            outcomes = [int(i['reward'] >= 1) for (t, k), i in resolved.items() if k == a]
            per.append({'attempt': a, 'expected': len(tasks), 'scored': len(outcomes), 'passed': sum(outcomes),
                        'pass_at_1': sum(outcomes) / len(tasks) if len(outcomes) == len(tasks) else None})
        complete = len(resolved) == expected
        interval = None
        if complete:
            rates = [sum(int(resolved[t, a]['reward'] >= 1) for a in range(attempts)) / attempts for t in tasks]
            rng = random.Random(20261016)
            boot = sorted(sum(rng.choices(rates, k=len(tasks))) / len(tasks) for _ in range(10000))
            interval = [boot[249], boot[9749]]
        passed = sum(i['reward'] >= 1 for i in resolved.values())
        unresolved = collections.Counter()
        for task in tasks:
            for a in range(attempts):
                if (task, a) not in resolved:
                    prior = [i for i in self.s['items'].values() if i['kind'] == 'eval' and i['task'] == task and i['attempt'] == a]
                    last = max(prior, key=lambda i: i['retry']) if prior else {}
                    unresolved[last.get('infrastructure_failure') or last.get('exception_type') or 'not_run'] += 1
        atomic(self.c.eval / 'scores.json', {'complete': complete, 'coverage': len(resolved) / expected,
            'scored': len(resolved), 'expected': expected, 'per_attempt': per,
            'pass_at_1': passed / expected if complete else None,
            # Incomplete (tail rule): unresolved attempts as failures, and over the resolved ones only.
            'pass_at_1_unresolved_as_fail': passed / expected,
            'pass_at_1_resolved_only': passed / len(resolved) if resolved else None,
            'unresolved': dict(unresolved), 'tail_closed': self.s.get('tail_closed'),
            'task_bootstrap_95': interval, 'updated_at': time.time()})

    # ---------------------------------------------------------------- smoke
    def smoke(self, q, ready):
        specs = self.c.config.get('smoke', {}).get('attempts', [])
        if not specs:
            self.s['stage'] = 'running'
            return
        listed = set(self.train_list())
        waiting = [s['task'] for s in specs if s.get('kind', 'train') == 'train' and s['task'] not in listed]
        if waiting:  # streamed smoke tasks: wait for their chunk (synth.prepare checked frozen ones)
            if not self.stream_open():
                raise RuntimeError(f'Smoke tasks {waiting} are not in the train list and the task stream has ended')
            if time.time() - self.s.get('smoke_waiting_logged', 0) > 1800:
                self.s['smoke_waiting_logged'] = time.time()
                self.c.log(f'Smoke waits for {waiting} from the task stream')
            return
        self._heartbeats = None
        workers = self.ready_workers(1)
        if not ready or not workers:
            return
        backend = ready[sorted(ready)[0]]
        latest = {}
        for attempt, spec in enumerate(specs):
            prior = sorted((i for i in self.s['items'].values() if i['kind'] == 'smoke' and i['attempt'] == attempt),
                           key=lambda i: i['retry'])
            # A smoke lost to the infrastructure (e.g. its worker's node) is retried, not judged.
            if not prior or (prior[-1]['status'] == 'finished' and prior[-1].get('infrastructure_failure')):
                if len(prior) >= 3:
                    raise RuntimeError(f'Smoke {spec["task"]} failed on infrastructure three times')
                uid = self.assign('smoke', spec['task'], attempt, workers[0][1], backend, retry=len(prior),
                                  task_kind=spec.get('kind', 'train'), smoke_cap=spec.get('cap'),
                                  smoke_context_window=spec.get('context_window'))
                prior.append(self.s['items'][uid])
            latest[attempt] = prior[-1]
        items = list(latest.values())
        if any(i['status'] != 'finished' for i in items):
            return
        self.export(force=True)
        evidence = []
        for item in items:
            ex = self.indices['smoke'].get(item['id'], {})
            # The capped smoke checks truncation handling; its task's verifier may still time out
            # (e.g. downloading its tools), so only the uncapped smokes must carry a reward.
            if not ex.get('usable') or item.get('infrastructure_failure') or (
                    item.get('reward') is None and not item.get('smoke_cap')):
                raise RuntimeError(f'Production-path smoke requires a usable scored original: {item["id"]}: {ex}')
            with gzip.open(ex['path'], 'rt') as f:
                record = json.load(f)
            messages = [e['message'] for s in record['sessions'] for e in s['entries'] if e.get('type') == 'message']
            parts = [p for m in messages if isinstance(m.get('content'), list) for p in m['content']]
            if not any(p.get('type') == 'thinking' for p in parts):
                raise RuntimeError('Pi smoke did not expose reasoning')
            if not item.get('smoke_context_window') and not any(m.get('role') == 'toolResult' for m in messages):
                raise RuntimeError('Normal-limit Pi smoke did not save tool observations')
            if item.get('smoke_cap'):
                # Pi's patched protocol: a truncated turn without tool calls gets the continuation
                # message; one with tool calls gets "not executed" tool results. Never silence.
                cut = [i for i, m in enumerate(messages) if m.get('role') == 'assistant' and m.get('stopReason') == 'length']
                if not cut:
                    raise RuntimeError('Capped smoke never reached the output cap')
                for i in cut:
                    after = messages[i + 1] if i + 1 < len(messages) else None
                    if after is not None and not (
                            (after.get('role') == 'user' and CONTINUATION_TEXT in str(after.get('content')))
                            or (after.get('role') == 'toolResult' and 'not executed' in str(after.get('content')))):
                        raise RuntimeError(f'Truncated turn {i} was not followed by Pi continuation handling')
            if item.get('smoke_context_window'):
                terminal = [m for seg in record['segments'] for m in seg['messages'] if CAP_TEXT in str(m['message'])]
                if any(m['loss_weight'] != 0 for m in terminal):
                    raise RuntimeError('Synthetic context-limit text must keep zero assistant loss')
            evidence.append({'id': item['id'], 'export': ex, 'result': item['result_path']})
        if not any(i.get('reward') is not None for i in items):
            raise RuntimeError('No production-path smoke was scored')
        crosscheck = self.validate_tokenizer(backend['url'])
        atomic(self.ops / 'smoke' / 'validation.json', {'passed': True, 'time': time.time(), 'items': evidence,
            'tokenizer': crosscheck, 'excluded_from_train_and_eval': True})
        self.s['stage'] = 'running'
        self.c.log('Production-path smokes passed; starting sustained generation.')

    def validate_tokenizer(self, url):
        from .serializers import serializer_for
        serializer = serializer_for(self.c.teacher)
        messages = [{'role': 'system', 'content': 'Serialization validation.'},
                    {'role': 'user', 'content': 'Compute 17 + 25.'},
                    {'role': 'assistant', 'content': '42', 'reasoning_content': '17+25=42'}]
        local = serializer.count(serializer.render(messages, []))
        payload = serializer.tokenize_payload(messages, self.c.teacher)
        if payload is None:
            return {'local': local, 'skipped': 'teacher has no /tokenize chat cross-check'}
        req = urllib.request.Request(url.removesuffix('/v1') + '/tokenize', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        remote = json.load(OPENER.open(req, timeout=120))
        if local != remote['count']:
            raise ValueError(f'Local and vLLM token counts disagree: {local} != {remote["count"]}')
        return {'local': local, 'vllm': remote['count'], 'passed': True, **serializer.identity()}

    # ------------------------------------------------------------- admission
    def admission(self, ready):
        """name -> backend status: paused, limit (agent capacity), room (new attempts allowed now), active."""
        if self.policy().get('mode') == 'kv_fit':
            return self.admission_kv_fit(ready)
        return self.admission_limits(ready)

    def admission_kv_fit(self, ready):
        """synth/admission.py: admit while in-flight KV plus idle agents' cached contexts fit the KV."""
        now, policy = time.time(), self.policy()
        p = {**kv_admission.DEFAULTS, **policy}
        contexts, post_agent = {}, set()
        for hb in self.heartbeats().values():
            contexts.update(hb.get('contexts') or {})
            post_agent.update(hb.get('post_agent') or ())
        by_backend = collections.defaultdict(list)
        pre_agent = collections.Counter()
        for item in self.active():
            if item['id'] in post_agent:  # agent finished, verifying: no KV, never calls the model again
                continue
            if item['id'] not in contexts:  # environment build / agent setup: no KV yet
                pre_agent[item['backend']] += 1
                continue
            by_backend[item['backend']].append(item)
        self._post_agent = len(post_agent & {i['id'] for i in self.active()})
        self._pre_agent = sum(pre_agent.values())
        # admission.pre_agent_weight < 1 counts attempts still building their environment as a fraction of an agent: no KV (RTS 2026-10-27: ~1,000
        # network-bound builds of tasks without a shared base held every engine slot with 2-7% of agents in the
        # model), and the quarter still bounds how many builds pile up behind the proxy.
        pre_weight = policy.get('pre_agent_weight', 1.0)
        others = self.shared_load(ready, now)
        samples = dict(zip(ready, self.pool.map(lambda nb: self.backend_sample(nb[0], nb[1], now), ready.items())))
        observed = {}
        for name, backend in ready.items():
            b = self.s['backends'].setdefault(name, {})
            block = backend.get('kv_block_tokens') or p['kv_block_tokens']
            idle = expected = cached = 0
            for item in by_backend.get(name, []):
                context = contexts.get(item['id'])
                idle += kv_admission.idle_tokens(context, block)
                if context:
                    e, c = context.get('expected_cached_tokens', 0), context.get('cached_tokens', 0)
                    pe, pc = self._reuse_seen.get(item['id'], (e, c))
                    expected, cached = expected + max(0, e - pe), cached + max(0, c - pc)
                    self._reuse_seen[item['id']] = (e, c)
            sample = samples[name]
            raw = sample.get('raw', {})
            preempted = raw.get('preempt', 0) > b.get('last_preempt', raw.get('preempt', 0))
            b['last_preempt'] = raw.get('preempt', b.get('last_preempt', 0))
            other = others.get(name, {})
            observed[name] = (sample, {
                'kv': raw.get('kv'), 'wait': raw.get('wait'), 'preempted': preempted,
                'stale': now - sample.get('timestamp', 0) > 100,
                'capacity': backend.get('kv_capacity_tokens') or p['kv_capacity_tokens'],
                'offload': backend.get('kv_offload_tokens') or 0,
                'own_idle_tokens': idle,
                'own_agents': len(by_backend.get(name, [])) + math.ceil(pre_weight * pre_agent[name]),
                'other_idle_tokens': other.get('idle_tokens', 0), 'other_agents': other.get('agents', 0),
                'reuse_expected': expected, 'reuse_cached': cached,
                'gen_tok_s': sample.get('srv_gen_tok_s') if now - sample.get('timestamp', 0) <= 100 else None})
        live = {i['id'] for items in by_backend.values() for i in items}
        self._reuse_seen = {k: v for k, v in self._reuse_seen.items() if k in live}
        # Mean KV demand per live agent: a slow average of measured working set / agents, floored by the
        # prior (during a ramp every agent is young and short; following that down over-admits).
        fit = self.s.setdefault('kv_fit', {'agent_tokens': p['agent_tokens_prior'], 'updated_at': now})
        fresh = [o for _, o in observed.values() if not o['stale'] and o['capacity'] and o['kv'] is not None]
        agents = sum(o['own_agents'] + o['other_agents'] for o in fresh)
        if agents >= p.get('agent_tokens_min_sample', 16):
            fit['agent_tokens'] = kv_admission.update_mean(
                fit['agent_tokens'], sum(kv_admission.working_set(o) for o in fresh) / agents,
                now - fit['updated_at'], p['agent_tokens_halflife_seconds'])
        fit['updated_at'] = now
        mean = max(fit['agent_tokens'], p['agent_tokens_prior'] * p.get('agent_tokens_floor', 1.0), 1024)
        status, loads = {}, collections.defaultdict(dict)
        for name, backend in ready.items():
            sample, obs = observed[name]
            st = kv_admission.step(self.s['backends'][name], policy, now, obs, mean)
            cap = policy.get('max_agents_per_backend')
            if cap:  # hot override (ops/admission-policy.json): fewer concurrent agents, e.g. for the cluster proxy
                left = max(0, int(cap) - obs['own_agents'])
                st.update(limit=min(st['limit'], int(cap)), free_agents=min(st['free_agents'], left),
                          room=min(st['room'], left))
            raw = sample.get('raw', {})
            status[name] = {**st, 'active': obs['own_agents'], 'kv': raw.get('kv'), 'running': raw.get('run'),
                            'waiting': raw.get('wait'), 'hit': sample.get('hit'), 'gen_tok_s': sample.get('srv_gen_tok_s')}
            if backend.get('serving_home'):
                loads[backend['serving_home']][name] = {'idle_tokens': obs['own_idle_tokens'], 'agents': obs['own_agents']}
        for home, backends in loads.items():
            best_effort(atomic, Path(home) / 'serving' / 'load' / f'{self.c.id}.json',
                        {'campaign': self.c.id, 'updated_at': now, 'backends': backends})
        best_effort(atomic, self.ops / 'admission.json', {'updated_at': now, 'policy': p, 'mode': 'kv_fit',
                                                          'agent_tokens': round(mean), 'backends': status})
        best_effort(append_jsonl, self.ops / 'metrics' / 'admission-samples.jsonl', {'time': now, 'agent_tokens': round(mean),
            'backends': {k: {kk: v[kk] for kk in ('paused', 'limit', 'room', 'active', 'kv', 'waiting', 'running', 'target',
                                                  'working_set_tokens', 'measured_tokens', 'inflight_tokens', 'capacity_tokens',
                                                  'reuse', 'hit', 'gen_tok_s', 'causes')}
                         for k, v in status.items()}})
        return status

    def shared_load(self, ready, now):
        """backend -> {idle_tokens, agents} that other campaigns on the same serving published."""
        total = collections.defaultdict(lambda: {'idle_tokens': 0, 'agents': 0})
        for home in {b['serving_home'] for b in ready.values() if b.get('serving_home')}:
            for path in (Path(home) / 'serving' / 'load').glob('*.json'):
                if path.stem == self.c.id:
                    continue
                load = best_effort(read, path, {}) or {}
                if now - load.get('updated_at', 0) > 120:
                    continue
                for name, value in load.get('backends', {}).items():
                    if name in ready:
                        total[name]['idle_tokens'] += value.get('idle_tokens', 0)
                        total[name]['agents'] += value.get('agents', 0)
        return total

    def admission_limits(self, ready):
        now, policy = time.time(), self.policy()
        status = {}
        for name in ready:
            b = self.s['backends'].setdefault(name, {'limit': policy['initial_limit_per_backend'],
                'last_limit_increase': now, 'history': []})
            sample = read(ready[name].get('metrics') or self.ops / 'metrics' / 'backends' / f'{name}.json', {})
            raw, previous = sample.get('raw', {}), b.get('last_raw', {})
            preempt = raw.get('preempt', 0) > previous.get('preempt', raw.get('preempt', 0))
            history = b['history']
            if raw and (not history or history[-1][0] != sample.get('timestamp')):
                history.append([sample['timestamp'], raw.get('kv', 1)])
            history[:] = [x for x in history if now - x[0] <= 300]
            slope = (max(0, (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0]))
                     if len(history) > 1 and history[-1][0] > history[0][0] else 0)
            forecast = raw.get('kv', 1) + policy['forecast_seconds'] * slope
            causes = [n for n, cond in (('stale_metrics', now - sample.get('timestamp', 0) > 100),
                ('kv_emergency', raw.get('kv', 1) >= policy['emergency_kv_high']),
                ('kv_growth_forecast', forecast >= policy['forecast_kv']),
                ('preemption', preempt), ('inference_queue', raw.get('wait', 0) > policy['max_waiting'])) if cond]
            b['limit'] = min(b['limit'], policy['maximum_limit_per_backend'])
            if causes:
                b['paused'], b['low_since'], b['last_pressure_at'] = True, None, now
                if preempt:
                    b['limit'] = max(16, int(b['limit'] * 0.8))
            elif b.get('paused'):
                if raw.get('kv', 1) <= policy['resume_kv'] and raw.get('wait', 0) < 4:
                    b['low_since'] = b.get('low_since') or now
                    if now - b['low_since'] >= 90:
                        b['paused'] = False
                else:
                    b['low_since'] = None
            active = len(self.active(backend=name))
            if (b['limit'] < policy['maximum_limit_per_backend'] and not causes and not b.get('paused')
                    and raw.get('kv', 1) < policy['increase_below_kv'] and active >= b['limit'] - 4
                    and now - b['last_limit_increase'] > policy['minimum_limit_hold_seconds']
                    and now - b.get('last_pressure_at', 0) > 600):
                b['limit'] = min(policy['maximum_limit_per_backend'], b['limit'] + policy['limit_step'])
                b['last_limit_increase'] = now
            b['last_raw'] = raw
            status[name] = {'paused': b.get('paused', False), 'limit': b['limit'], 'active': active,
                            'room': 0 if b.get('paused') else max(0, b['limit'] - active),
                            'causes': causes, 'kv': raw.get('kv'), 'running': raw.get('run'),
                            'waiting': raw.get('wait'), 'forecast': forecast, 'gen_tok_s': sample.get('srv_gen_tok_s')}
        best_effort(atomic, self.ops / 'admission.json', {'updated_at': now, 'policy': policy, 'backends': status})
        best_effort(append_jsonl, self.ops / 'metrics' / 'admission-samples.jsonl', {'time': now, 'backends': {
            k: {kk: v[kk] for kk in ('paused', 'limit', 'active', 'kv', 'waiting')} for k, v in status.items()}})
        return status

    def pending_work(self):
        """Attempts still to be assigned (upper bound); inf while a token target is open."""
        pending = 0
        if self.c.enabled('eval'):
            active = {(i['task'], i['attempt']) for i in self.active('eval')}
            pending += sum((t, a) not in active for t in self.c.tasks('eval') for a in range(self.c.eval_attempts)
                           if (t, a) not in self.resolved_eval())
        if self.c.enabled('train') and not self.s['train_stop']:
            if self.mode == 'tokens':
                return math.inf
            target = self.c.config['train']['trajectories_per_task']
            have = self.train_counts()
            for item in self.s['items'].values():
                if (item['kind'] == 'train' and item['status'] == 'finished' and not item.get('infrastructure_failure')
                        and item['id'] not in self.indices['train']):
                    have[item['task']] += 1
            active = collections.Counter(i['task'] for i in self.active('train'))
            tried = self.tried()
            caps = self.attempt_caps()
            if self.budget_open(have) or self.budget_pending() or self.stream_wanted(have):
                return math.inf
            pending += sum(max(0, min(target - have[t] - active[t], caps[t] - tried[t])) for t in self.train_tasks())
        return pending

    @staticmethod
    def job_of(backend):
        return int(backend[1:].split('r', 1)[0])

    def serving_shared(self):
        """This campaign's serving jobs that a live campaign has adopted (ops/serving/adopters)."""
        return [r['job_id'] for r in condor.receipts(self.c)
                if r['role'] == 'serving' and not r.get('adopted_at') and condor.attached(self.ops, r['job_id'])]

    def retained_serving(self):
        """Operator file ops/serving/RETAIN.json: the serving jobs outlive this campaign (a successor
        campaign adopts them), so they are neither drained, released nor removed at completion. It holds them while
        an adopter is attached (ops/serving/adopters), or for an hour after it was written so the successor can
        start; after that the jobs are released as usual (2026-10-25: a RETAIN.json whose successor had already
        finished kept 4 B200s idle past completion)."""
        path = self.ops / 'serving' / 'RETAIN.json'
        retain = read(path)
        if not retain:
            return None
        if time.time() - path.stat().st_mtime < 3600:
            return retain
        attached = [r['job_id'] for r in condor.receipts(self.c)
                    if r['role'] == 'serving' and condor.attached(self.ops, r['job_id'])]
        return retain if attached else None

    def scale_down(self, q, status):
        """Give back GPUs and CPUs the remaining work cannot use.

        Long six-hour tails otherwise keep every allocation for a few rollouts. When the
        live attempts plus the work still to assign fit in 75% of the other serving jobs'
        admission limits, the least-loaded job stops taking attempts (draining) and is
        released once its last pinned attempt ends; it comes back if work grows again.
        Idle workers beyond what the remaining work needs are retired the same way."""
        jobs = self.c.config['serving']['jobs']
        released = self.s.setdefault('released_serving', [])
        draining = self.s.setdefault('draining_serving', [])
        cap, capped = self.serving_jobs(), self.s.setdefault('capped_serving', {})
        for j in range(cap, jobs):  # above the serving_jobs override: drain, then release
            if j not in released and j not in draining:
                draining.append(j)
                capped[str(j)] = time.time()
                self.c.log(f'Serving job {j} drains: serving_jobs override {cap} of {jobs}')
        for j in [int(k) for k in capped if int(k) < cap]:  # override raised: bring capped jobs back
            capped.pop(str(j))
            for lst in (draining, released):
                if j in lst:
                    lst.remove(j)
            self.c.log(f'Serving job {j} back: serving_jobs override {cap} of {jobs}')
        grace = self.policy().get('serving_drain_grace_sec', 1800)
        for k, since in capped.items():
            if int(k) in draining and time.time() - since > grace:
                self.interrupt(lambda i, j=int(k): i['backend'].startswith(f'j{j}r'), 'ServingCapped')
        pending = self.pending_work()
        active = self.active()
        load = len(active) + pending
        per_job = collections.Counter(self.job_of(i['backend']) for i in active)
        capacity = collections.Counter()
        for name, value in status.items():
            capacity[self.job_of(name)] += value['limit']
        keep = [j for j in range(min(jobs, cap)) if j not in released and j not in draining]
        if self.retained_serving():
            if draining:
                self.c.log(f'ops/serving/RETAIN.json: serving jobs {draining} stop draining and stay up')
                draining.clear()
        elif self.autoscale_params():
            self.autoscale_tick(q, status, per_job)
        elif load > 0.9 * sum(capacity[j] for j in keep) and any(j < cap for j in draining + released):
            self.c.log(f'Load {load} exceeds the kept serving capacity; restoring drained/released serving jobs.')
            draining[:] = [j for j in draining if j >= cap]
            released[:] = [j for j in released if j >= cap]
        elif len(keep) > 1 and not self.max_concurrency_open() and not self.s.get('serving_released'):
            victim = min(keep, key=lambda j: (per_job[j], -j))
            if capacity[victim] and load <= 0.75 * sum(capacity[j] for j in keep if j != victim):
                draining.append(victim)
                self.c.log(f'Draining serving job {victim}: {load} attempts (live + to assign) fit in the other jobs')
        for index in list(draining):
            if not per_job[index]:
                draining.remove(index)
                released.append(index)
                latest = condor.latest(self.c, 'serving', index)
                if latest and latest['job_id'] in q:
                    condor.release(self.c, latest['job_id'])
                self.c.log(f'Released drained serving job {index}: no attempts pinned to it')
        self.scale_worker_fleet(load, pending, active)

    def scale_worker_fleet(self, load, pending, active):
        """Retire the workers the work does not need (#241). The target is scale_workers' worker_target (kv_fit),
        else what the remaining work needs. Above it:
        - idle workers retire, highest index first;
        - if that is not enough, the least-loaded busy workers drain: dispatch skips them (ready_workers) and they retire when their last attempt ends. Dispatch spreads
          attempts over every ready worker, so without draining no worker ever empties (m1-hero: ~20 agents on each
          of ~101 workers, ~40% of slots used).
        Both wait until the fleet has been above the target for shrink_after_sec, so startup (the configured count
        before supply arrives) and short dips do not cost worker restarts. A higher target cancels drains first, then
        brings retired indices back (ensure_workers resubmits them). Index 0 is never retired."""
        w = self.c.config['workers']
        slots = w['slots']
        count = self.worker_count()
        wanted = count if pending == math.inf else min(count, math.ceil(load / slots) + 2)
        if self.s.get('worker_target') is not None:  # kv_fit: the live demand + headroom (scale_workers)
            wanted = min(wanted, self.s['worker_target'])
        retired = set(self.s.setdefault('retired_workers', []))
        draining = [i for i in self.s.setdefault('draining_workers', []) if i not in retired and i < count]
        load_by_worker = collections.Counter(i['worker'] for i in active)

        def taking():  # workers that take new attempts
            return count - len(retired) - len(draining)

        def retire(index, why):
            retired.add(index)
            (self.ops / 'workers' / str(index)).mkdir(parents=True, exist_ok=True)
            (self.ops / 'workers' / str(index) / 'STOP').write_text('synth-retired\n')
            self.c.log(f'Retired worker {index}: {why} (target {wanted}, {taking()} taking attempts)')

        if taking() < wanted:
            self.s.pop('worker_shrink_since', None)
            while draining and taking() < wanted:
                index = draining.pop()
                self.c.log(f'Worker {index} stops draining: target {wanted}')
            for index in sorted(retired):
                if taking() >= wanted:
                    break
                stop = self.ops / 'workers' / str(index) / 'STOP'
                if stop.exists() and stop.read_text() == 'synth-retired\n':
                    stop.unlink()
                retired.discard(index)
        if taking() > wanted:
            since = self.s.setdefault('worker_shrink_since', time.time())
            if time.time() - since >= w.get('shrink_after_sec', 900):
                for index in reversed(range(1, count)):
                    if taking() <= wanted:
                        break
                    if index not in retired and index not in draining and not load_by_worker[index]:
                        retire(index, 'idle above the target')
                busy = sorted((load_by_worker[i], -i, i) for i in range(1, count) if i not in retired and i not in draining)
                picked = [i for _, _, i in busy[:max(0, taking() - wanted)]]
                draining.extend(picked)
                if picked:
                    self.c.log(f'Draining workers {picked}: target {wanted} of {count - len(retired)} live '
                               f'({len(active)} attempts hold slots)')
        else:
            self.s.pop('worker_shrink_since', None)
        for index in list(draining):
            if not load_by_worker[index]:
                draining.remove(index)
                retire(index, 'drained')
        self.s['draining_workers'] = draining
        self.s['retired_workers'] = sorted(retired)

    def train_demand(self):
        """Active attempts + attempts still to dispatch (#234): opened tasks' owed trajectories (within their attempt
        caps) + streamed, unopened tasks x trajectories_per_task while tasks may still open, + pending eval attempts.
        math.inf for a tokens-mode train target (no per-task bound)."""
        active = len(self.active())
        pending = 0
        if self.c.enabled('eval'):
            live = {(i['task'], i['attempt']) for i in self.active('eval')}
            resolved = self.resolved_eval()
            pending += sum((t, a) not in live and (t, a) not in resolved for t in self.c.tasks('eval')
                           for a in range(self.c.eval_attempts))
        if self.c.enabled('train') and not self.s['train_stop']:
            if self.mode == 'tokens':
                return math.inf
            target = self.c.config['train']['trajectories_per_task']
            plan = self.train_plan()
            have, act, tried, caps = plan['have'], plan['active'], plan['tried'], plan['caps']
            pending += sum(max(0, min(target - have[t] - act[t], caps[t] - tried[t])) for t in self.train_tasks())
            if self.budget() and self.budget_open(have):
                unopened = len(self.train_list()) - len(self.s.get('opened', [])) - len(self.s.get('invalid', []))
                pending += max(0, unopened) * target
        return active + pending

    def autoscale_tick(self, q, status, per_job):
        """#234 R1: set the engines in use from demand (synth/autoscale.py), between [serving.autoscale] start and the
        live cap. Up: un-drain a draining engine first (it is already up), else submit the lowest free index (via
        ensure_serving). Down: drain the least-loaded engine (no new attempts; released once its last attempt ends)."""
        p = self.autoscale_params()
        a = self.autoscale_state(q)
        now = time.time()
        if self.s['train_stop'] or self.s.get('serving_released'):
            return
        released = self.s.setdefault('released_serving', [])
        draining = self.s.setdefault('draining_serving', [])
        cap, jobs = self.serving_jobs(), self.c.config['serving']['jobs']
        wanted = a['wanted'] = [j for j in a['wanted'] if j < cap and j not in released and j not in draining]

        def alive(j):
            r = condor.latest(self.c, 'serving', j)
            return bool(r and r['job_id'] in q)
        for j in range(jobs):   # a running job outside the policy's set (restored by a cap change): drain it
            if j not in wanted and j not in draining and j not in released and alive(j):
                draining.append(j)
                self.c.log(f'Serving autoscale: job {j} is not in use; draining')
        if now - a.get('decided_at', 0) < p['period_seconds']:
            return
        a['decided_at'] = now
        back = [j for j in draining if j < cap and str(j) not in self.s.get('capped_serving', {})]
        free = [j for j in range(cap) if j not in wanted and j not in draining and not alive(j)]
        ceiling = max(len(wanted), min(cap, len(wanted) + len(back) + len(free)))
        limits = sorted(v['limit'] for n, v in status.items() if self.job_of(n) in wanted and v.get('limit'))
        fit = limits[len(limits) // 2] if limits else p['fit_prior']
        demand = self.train_demand()
        n = autoscale.decide(a, now, engines=len(wanted), ceiling=ceiling, demand=demand, fit=fit, p=p)
        a['last'] = {'time': now, 'engines': len(wanted), 'ceiling': ceiling, 'demand': demand if demand != math.inf
                     else 'inf', 'fit': fit, 'wanted': n}
        if n == len(wanted):
            return
        why = f'demand {demand} vs {len(wanted)} x fit {fit}'
        while len(wanted) < n:
            if back:
                j = max(back, key=lambda j: (per_job[j], -j))
                back.remove(j)
                draining.remove(j)
                verb = 'back from draining'
            else:
                j = free.pop(0)
                if j in released:
                    released.remove(j)
                verb = 'submitted'
            wanted.append(j)
            self.c.log(f'Serving autoscale: engine {j} {verb} ({why}); {len(wanted)} in use')
            a['events'] = (a.get('events') or [])[-199:] + [[round(now), 'up', j, demand if demand != math.inf else -1]]
        while len(wanted) > n:
            j = min(wanted, key=lambda j: (per_job[j], -j))
            wanted.remove(j)
            draining.append(j)
            self.c.log(f'Serving autoscale: engine {j} drains ({why}); {len(wanted)} in use')
            a['events'] = (a.get('events') or [])[-199:] + [[round(now), 'down', j, demand if demand != math.inf else -1]]
        a['wanted'] = sorted(wanted)

    # -------------------------------------------------------------- running
    def running(self, q, ready):
        self._heartbeats = None  # this tick's heartbeats (admission, workers, dispatch)
        self._saved_cache = None
        if self.max_concurrency_open() and not self.s.get('budget_closed'):
            saved = self.estimated_saved()
            if saved >= self.budget():
                self.s['budget_closed'] = {'time': time.time(), 'estimated_saved': saved, 'budget': self.budget(),
                                           'active': len(self.active('train'))}
                self.c.log(f'Token budget reached at max concurrency: ~{saved / 1e6:.1f}M of {self.budget() / 1e6:.0f}M '
                           f'saved; closing training ({len(self.active("train"))} live attempts cut).')
        if self.c.enabled('train') and not self.s['train_stop'] and self.train_target_reached():
            self.s['train_stop'] = True
            closed = self.train_closed()
            by = ('ops/CLOSE_TRAIN.json' if read(self.ops / 'CLOSE_TRAIN.json') else
                  'the tail rule' if self.s.get('tail_closed') else 'the token budget')
            self.c.log(f'Training closed by {by}; stopping new training claims.' if closed else
                       'Training target reached on live counts; stopping new training claims.')
            self.interrupt(lambda i: i['kind'] == 'train', 'TrainingClosed' if closed else 'TrainingTargetReached')
        self.release_idle_serving(q)
        blocked = self.blocked()
        if blocked:
            self.interrupt(lambda i: i['kind'] == 'train' and i['task'] in blocked, 'TaskBlocked')
        if self.s['train_stop'] and self.training_flush_complete() and not self.train_target_reached(finalized_only=True):
            self.s['train_stop'] = False
            self.c.log('Finalized corpus fell below the target; resuming training claims.')
        with self.timed('admission'):
            status = self.admission(ready)
        self.scale_workers(status)
        with self.timed('ensure_workers'):
            self.ensure_workers(q, self.worker_count())
        policy = self.policy()
        if (not (self.ops / 'protocol-defect.json').exists()
                and time.time() - self.last_dispatch >= policy['period_seconds']):
            self.last_dispatch = time.time()
            with self.timed('dispatch'):
                self.dispatch(ready, status, policy)
        with self.timed('scale_down'):
            self.scale_down(q, status)
        self.tail_check(status)
        eval_done = ((len(self.resolved_eval()) == self.c.eval_expected or self.s.get('tail_closed'))
                     and not self.active('eval'))
        train_done = (not self.c.enabled('train') or
                      (self.s['train_stop'] and self.training_flush_complete() and not self.active('train')))
        if eval_done and train_done:
            self.s['stage'] = 'finalizing'

    # ------------------------------------------------------------ image affinity + proxy breaker
    def image_group(self, task):
        """A task's shared harbor_ns base (train.image_groups: {task: base id}, from synth.prefix_bases); tasks
        without one are their own group."""
        if getattr(self, '_image_groups', None) is None:
            path = (self.c.config.get('train') or {}).get('image_groups')
            self._image_groups = read(path, {}) if path else {}
        return self._image_groups.get(task) or f'task:{task}'

    def affinity(self):
        """Workers (in their current job, so with the scratch cache intact) that ran each task / image group."""
        beats = self.heartbeats()
        by_task, by_group = collections.defaultdict(set), collections.defaultdict(set)
        for item in self.s['items'].values():
            if item['kind'] != 'train' or item.get('worker_job') is None:
                continue
            if (beats.get(item['worker']) or {}).get('job_id') != item['worker_job']:
                continue
            by_task[item['task']].add(item['worker'])
            by_group[self.image_group(item['task'])].add(item['worker'])
        return by_task, by_group

    def affine_worker(self, task, loads, worker_slots, default, aff, slack):
        """The worker for a training attempt: one that already built this task's image, else one holding its
        shared base, unless it is more than `slack` (fraction of its slots) busier than the least-loaded worker.
        per_task issues a task's attempts back to back, so without this its 8 attempts land on 8 workers, each
        unpacking the base from /data and rebuilding the task layer (and fetching anything it downloads)."""
        by_task, by_group = aff
        floor = loads[default] / worker_slots[default]
        best = None
        for rank, candidates in ((0, by_task.get(task, ())), (1, by_group.get(self.image_group(task), ()))):
            for w in candidates:
                if w in loads and loads[w] / worker_slots[w] <= floor + slack:
                    key = (rank, loads[w] / worker_slots[w], w)
                    best = key if best is None or key < best else best
        stats = self.s.setdefault('affinity_stats', {'task': 0, 'group': 0, 'other': 0})
        stats[('task', 'group')[best[0]] if best else 'other'] += 1
        return best[2] if best else default

    def proxy_ok(self, policy):
        """admission.proxy_probe = {urls, interval_sec, timeout_sec, fail_after, recover_after}: while the cluster
        proxy fails the probes (every URL through it), no new attempts start. Builds and verifiers fetch through it,
        and multi-hour outages (2026-10-24/25) zeroed 31-42% of verifier runs. Probes run in the thread pool so a
        dead proxy never stalls the tick."""
        probe = policy.get('proxy_probe')
        if not probe:
            return True
        st = self.s.setdefault('proxy', {'ok': True, 'fails': 0, 'oks': 0})
        future = getattr(self, '_proxy_future', None)
        if future is not None and future.done():
            self._proxy_future = None
            ok, detail = future.result()
            st.update(checked_at=time.time(), detail=detail)
            st['fails'], st['oks'] = (0, st['oks'] + 1) if ok else (st['fails'] + 1, 0)
            if st['ok'] and st['fails'] >= probe.get('fail_after', 3):
                st.update(ok=False, down_since=time.time())
                self.c.log(f'Proxy down ({detail}); pausing new attempts until it recovers.')
                best_effort(append_jsonl, self.ops / 'controller' / 'proxy-events.jsonl',
                            {'time': time.time(), 'event': 'down', 'detail': detail})
            elif not st['ok'] and st['oks'] >= probe.get('recover_after', 2):
                down = time.time() - st.get('down_since', time.time())
                st['ok'] = True
                self.c.log(f'Proxy recovered after {down / 60:.0f} min; dispatch resumes.')
                best_effort(append_jsonl, self.ops / 'controller' / 'proxy-events.jsonl',
                            {'time': time.time(), 'event': 'up', 'down_sec': round(down)})
        if getattr(self, '_proxy_future', None) is None and \
                time.time() - st.get('checked_at', 0) >= probe.get('interval_sec', 60):
            self._proxy_future = self.pool.submit(probe_proxy, probe.get('urls', PROXY_PROBE_URLS),
                                                  probe.get('timeout_sec', 20))
        return st['ok']

    def dispatch(self, ready, status, policy):
        """Start attempts on backends with room and workers with free slots. Every assignment of the
        tick is recorded first, the state saved once, then the inboxes published (the controller once
        saved its whole state per assignment: 0.3-0.7 s each at TMax scale, which capped dispatch)."""
        if not self.proxy_ok(policy):
            return 0
        slots = self.c.config['workers']['slots']
        beats = self.heartbeats()
        worker_slots = {i: (beats.get(i) or {}).get('capacity') or slots for i in range(self.worker_count())}
        # Fill workers by the fraction of their own slots in use, so workers of different shapes
        # (worker-resource-overrides.json) carry load in proportion to their slots; uniform workers fill as before.
        loads = {i: load for load, i in self.ready_workers(self.worker_count()) if load < worker_slots[i]}
        least = lambda: min(loads, key=lambda i: (loads[i] / worker_slots[i], loads[i], i))
        affinity = self.c.config['workers'].get('affinity') and self.affinity()
        slack = policy.get('affinity_slack', 0.25)
        draining = self.s.get('draining_serving', [])
        room = {n: v['room'] for n, v in status.items()
                if not v['paused'] and v['room'] > 0 and self.job_of(n) not in draining}
        cap = (math.inf if policy.get('mode') == 'kv_fit' else
               policy['burst'] * (len(status) if policy.get('burst_per_backend', True) else 1))
        capacity = sum(v['limit'] for v in status.values())
        reserved = max(1, int(capacity * policy['eval_share'])) if self.c.enabled('eval') else 0
        active_train, active_eval = len(self.active('train')), len(self.active('eval'))
        evals = collections.deque(self.eval_candidates())
        plan, new, starved, out_of_work = None, [], False, False
        while room and loads and len(new) < cap:
            name = max(room, key=lambda n: (room[n], n))
            worker = least()
            train_open = self.c.enabled('train') and not self.s['train_stop']
            if evals and prefer_eval(active_train, active_eval, capacity, reserved, not train_open):
                kind, work = 'eval', evals.popleft()
            else:
                if train_open:
                    plan = plan if plan is not None else self.train_plan()
                    work = self.next_train(plan)
                else:
                    work = None
                if work:
                    kind = 'train'
                elif evals:
                    kind, work = 'eval', evals.popleft()
                else:
                    starved, out_of_work = train_open, True
                    break
            if kind == 'eval':
                uid = self.assign('eval', work[0], work[1], worker, ready[name], retry=work[2], publish=False)
                active_eval += 1
            else:
                if affinity:
                    worker = self.affine_worker(work[0], loads, worker_slots, worker, affinity, slack)
                    affinity[0][work[0]].add(worker)
                    affinity[1][self.image_group(work[0])].add(worker)
                uid = self.assign('train', work[0], work[1], worker, ready[name], publish=False)
                active_train += 1
            new.append(self.s['items'][uid])
            status[name]['active'] += 1
            room[name] -= 1
            if not room[name]:
                del room[name]
            loads[worker] += 1
            if loads[worker] >= worker_slots[worker]:
                del loads[worker]
        # Worker slots ran out while the engines had room and training had work (not starved now, nor at the last
        # dispatch that looked): scale_workers grows the fleet by at most workers.grow_step workers per dispatch, so
        # startup (no worker ready yet) is not read as thousands of agents of demand (#241).
        w = self.c.config['workers']
        open_now = self.c.enabled('train') and not self.s['train_stop']
        has_work = not out_of_work and (bool(new) or (open_now and not self.s.get('train_starved_since')))
        self.s['worker_slot_short'] = (min(sum(room.values()), w.get('grow_step', 8) * slots)
                                       if room and not loads and has_work else 0)
        if starved:  # room and workers but no training attempt to start (see tail_check)
            self.s.setdefault('train_starved_since', time.time())
        elif any(i['kind'] == 'train' for i in new):
            self.s.pop('train_starved_since', None)
        if new:
            self.durable(new)  # durable ownership precedes publication to a worker
            list(self.pool.map(self.publish_inbox, new))
        return len(new)

    def scale_workers(self, status):
        """kv_fit with workers.max_count: worker slots for the attempts that run, plus the ones dispatch could not
        place for lack of a free slot, plus `workers.headroom` (#241).

        Every active attempt holds a worker slot from assignment to its worker result (building, in its agent,
        verifying), so the live demand is len(active). The KV room the engines could still admit is not demand: on
        m1-hero (supply-bound) it sized the fleet for ~4,000-5,000 agents while ~2,000 ran, ~40% of the slots used,
        for ~236k credits at floor prices. Growth comes from dispatch running out of slots (worker_slot_short);
        scale_down drains the workers above the target (worker_target)."""
        w = self.c.config['workers']
        policy = self.policy()
        if policy.get('mode') != 'kv_fit' or not w.get('max_count') or not status:
            return
        headroom = policy.get('worker_headroom', w.get('headroom', 0.25))   # live override: ops/admission-policy.json
        live = len(self.active())
        short = self.s.get('worker_slot_short', 0)
        demand = live + short
        self.s['worker_demand'] = demand
        needed = min(w['max_count'], math.ceil(demand * (1 + headroom) / w['slots']) + w.get('spare', 2))
        self.s['worker_target'] = needed
        if needed > self.s.get('worker_high_water', 0):
            self.s['worker_high_water'] = needed
            self._heartbeats = None
            self.c.log(f'Workers: {live} attempts hold slots, {short} more could start; using up to {needed} workers '
                       f'of {w["slots"]} slots')

    # ------------------------------------------------------------- finalize
    def finalize(self, q):
        self.export(force=True)
        if not self.train_target_reached(finalized_only=True) or (
                len(self.resolved_eval()) != self.c.eval_expected and not self.s.get('tail_closed')):
            self.s['stage'], self.s['train_stop'] = 'running', False
            self.s.pop('artifacts_verified', None)
            if self.s.pop('serving_released', None):  # released early: bring serving back
                (self.ops / 'serving' / 'RELEASE').unlink(missing_ok=True)
            for idx in range(self.worker_count()):
                p = self.ops / 'workers' / str(idx) / 'STOP'
                if p.exists() and p.read_text() == 'synth-finalizing\n':
                    p.unlink()
            self.c.log('Final read-back needs more work; resuming.')
            return
        retain = self.retained_serving()
        (self.ops / 'serving').mkdir(parents=True, exist_ok=True)
        if not retain:
            (self.ops / 'serving' / 'RELEASE').touch()
        serving = {r['job_id'] for r in condor.receipts(self.c) if r['role'] == 'serving'} if retain else set()
        for idx in range(self.worker_count()):
            p = self.ops / 'workers' / str(idx) / 'STOP'
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text('synth-finalizing\n')
        current = os.environ.get('SYNTH_JOB_ID')
        adopted = {r['job_id'] for r in condor.receipts(self.c) if r.get('adopted_at')} - serving
        for jid in adopted & set(q):
            condor.release(self.c, jid)  # detach; removed here only if its own campaign is done with it
        live = [jid for jid in q if jid in condor.owned_ids(self.c) and jid != current and jid not in serving
                and jid not in adopted]
        if live:
            for jid in live:
                condor.remove(self.c, jid)
            return
        if not self.s.get('artifacts_verified'):
            if self.archive_process is not None:
                self.collect_archive()
            for kind in ('train', 'eval'):
                packed = archive.load_index(self.c.root, kind)['attempts']
                for name, ex in self.indices[kind].items():
                    if not ex.get('usable') or name in packed:   # packed shards were verified when written
                        continue
                    assert sha(ex['path']) == ex['sha256'], ex['path']
                    with gzip.open(ex['path'], 'rt') as f:
                        record = json.load(f)
                    assert record['tokens'] == ex['tokens'] and record['finalized'], ex['path']
            for item in self.s['items'].values():
                if item['kind'] == 'eval':
                    inventory = [{'path': str(p.relative_to(self.c.eval)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                 for p in sorted(self.raw(item).rglob('*')) if p.is_file()]
                    atomic(self.c.eval / 'inventories' / f'{item["id"]}.json', inventory)
            self.s['artifacts_verified'] = True
            self.manifests()
            self.save(force=True)
        self.s['stage'] = 'complete'
        completion = {'status': 'complete', 'campaign': self.c.id, 'teacher': self.c.slug,
                      'model': self.c.teacher['model'], 'train_tokens_saved': self.totals()[0],
                      'owned_resources_released': not retain, 'completed_at': time.time()}
        if retain:
            completion['retained_serving'] = {'jobs': sorted(j for j in serving if j in q), 'retain': retain}
        for kind in ('train', 'eval'):
            path = self.c.root_for(kind) / 'manifest.json'
            if path.exists():
                completion[f'{kind}_manifest'] = str(path)
                completion[f'{kind}_manifest_sha256'] = sha(path)
        if self.c.enabled('eval'):
            completion.update(eval_attempts_finished=len(self.resolved_eval()), eval_attempts_expected=self.c.eval_expected)
        if self.train_closed():
            completion['train_closed_early'] = self.train_closed()
        if self.s.get('tail_closed'):
            completion['tail_closed'] = self.s['tail_closed']
        if self.stream_summary():
            completion['stream'] = self.stream_summary()
        if self.mode == 'per_task':
            counts = self.train_counts(finalized_only=True)
            completion['per_task_min'] = min((counts[t] for t in self.train_tasks()), default=0)
            completion['tasks'] = len(self.train_tasks())
            completion['tasks_below_target'] = sorted(t for t in self.train_tasks()
                                                      if counts[t] < self.c.config['train']['trajectories_per_task'])
        atomic(self.ops / 'completion.json', completion)
        self.c.log('Campaign complete: corpus finalized and verified, owned jobs released'
                   + (f' except retained serving jobs {completion["retained_serving"]["jobs"]}.' if retain else '.'))
        self.pack_completed()

    # --------------------------------------------------------------- report
    def progress(self, ready, error=None):
        tokens, n, finalized = self.totals()
        now = time.time()
        history = self.s.setdefault('token_history', [])
        if not history or tokens != history[-1][1] or now - history[-1][0] >= 90:
            history.append([now, tokens])
        history[:] = [x for x in history if now - x[0] <= 600]
        rate = (tokens - history[0][1]) / (now - history[0][0]) if history and now > history[0][0] else 0
        stage = self.s['stage']
        p = {'updated_at': now, 'heartbeat_at': now, 'campaign': self.c.id, 'teacher': self.c.slug,
             'phase': {'smoke': 'setup', 'running': 'running', 'finalizing': 'running', 'complete': 'complete'}[stage],
             'controller_stage': stage, 'controller_error': error, 'autonomous_controller': True,
             'needs_attention': bool(error or (self.ops / 'protocol-defect.json').exists()),
             'owned_job_ids': sorted(condor.owned_ids(self.c)), 'backends_ready': sorted(ready),
             'train_tokens_saved': tokens, 'train_trajectories': n, 'train_finalized': finalized,
             'train_saved_tokens_per_second': rate, 'train_stop': self.s['train_stop'],
             'active_train': len(self.active('train')), 'active_eval': len(self.active('eval')),
             'export_in_progress': self.export_process is not None,
             'last_export_duration_sec': self.s.get('last_export_duration_sec'),
             'worktree_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                               text=True).stdout.strip()}
        if self.mode == 'per_task' and self.c.enabled('train'):
            target = self.c.config['train']['trajectories_per_task']
            counts = self.train_counts(finalized_only=True)
            p['per_task'] = {'target': target, 'tasks': len(self.train_tasks()),
                             'at_target': sum(counts[t] >= target for t in self.train_tasks()),
                             'missing_trajectories': sum(max(0, target - counts[t]) for t in self.train_tasks())}
            if self.budget():
                p['per_task'].update(tokens_budget=self.budget(), available_tasks=len(self.train_list()),
                                     list_cursor=self.s.get('budget_cursor', 0), frontier=self.s.get('frontier', 0),
                                     skipped_invalid=self.s.get('skipped_invalid', 0),
                                     projected_tokens=round(self.projected_tokens()),
                                     tokens_per_trajectory_estimate=round(self.trajectory_tokens_estimate()))
        if self.c.enabled('eval'):
            p.update(eval_attempts_finished=len(self.resolved_eval()), eval_attempts_expected=self.c.eval_expected)
        if self.stream_summary():
            p['stream'] = self.stream_summary()
        if 'proxy' in self.s:
            p['proxy'] = self.s['proxy']
        if self.s.get('autoscale'):
            a = self.s['autoscale']
            p['autoscale'] = {'wanted': a['wanted'], 'draining': self.s.get('draining_serving', []),
                              'last': a.get('last'), 'events': (a.get('events') or [])[-10:]}
        if 'affinity_stats' in self.s:
            p['affinity_stats'] = self.s['affinity_stats']
        p['attempt_failures'] = self.failure_summary()
        if self.s.get('auto_excluded_hosts'):
            p['auto_excluded_hosts'] = sorted(self.s['auto_excluded_hosts'])
        if self.queue_view is not None:
            age = self.queue_view.age()
            p['condor_queue'] = {'age_sec': None if age is None else round(age, 1), 'reads': self.queue_view.reads,
                                 'failures': self.queue_view.failures}
        if 'last_state_save' in self.s:
            p['last_state_save'] = self.s['last_state_save']
        best_effort(atomic, self.ops / 'progress.json', p)

    def failure_summary(self):
        """Train attempts by failure phase, and the tasks stopped at the infrastructure cap (cached per item count)."""
        key = (len(self.s['items']), sum(1 for _ in self.active()))
        if getattr(self, '_failures', (None,))[0] != key:
            phases = collections.Counter(failure_phase(i) or 'none' for i in self.s['items'].values() if i['kind'] == 'train')
            capped = sum(1 for v in self.tried().values() if v == math.inf)
            self._failures = (key, {'pre_agent': phases['pre_agent'], 'mid_run': phases['mid_run'],
                                    'interruption': phases['interruption'], 'tasks_stopped': capped,
                                    'cap_per_task': self.infrastructure_cap()})
        return self._failures[1]

    def timed(self, name):
        """Context manager adding a phase's wall time to this tick's timings."""
        controller = self

        class Phase:
            def __enter__(self):
                self.start = time.monotonic()

            def __exit__(self, *exc):
                controller.timings[name] = controller.timings.get(name, 0) + time.monotonic() - self.start
        return Phase()

    # ------------------------------------------------------------ placement
    def note_io_error(self, exc):
        if isinstance(exc, OSError) and exc.errno == errno.EIO:
            self.io_errors.append(time.time())

    def placement_check(self, elapsed):
        """Move the controller off this node (resubmit at the next generation, excluding the node, then stop) after
        controller.relocate_slow_ticks (10) ticks in a row over controller.relocate_tick_sec (60 s), or
        controller.relocate_io_errors (2) I/O errors (EIO) within 30 min. At most one move per 30 min and
        controller.max_relocations (4) per campaign. m1-hero (#235): on gpu166, then gpu190, ticks took 60-380 s and 4
        archive packs failed with EIO; it was moved by hand (hold, condor_qedit Requirements, release)."""
        c = self.c.config.get('controller', {})
        self.slow_ticks = self.slow_ticks + 1 if elapsed > c.get('relocate_tick_sec', 60) else 0
        now = time.time()
        self.io_errors = [t for t in self.io_errors if now - t < 1800]
        reason = None
        if c.get('relocate_slow_ticks', 10) and self.slow_ticks >= c.get('relocate_slow_ticks', 10):
            reason = f'{self.slow_ticks} ticks in a row over {c.get("relocate_tick_sec", 60)} s'
        elif c.get('relocate_io_errors', 2) and len(self.io_errors) >= c.get('relocate_io_errors', 2):
            reason = f'{len(self.io_errors)} I/O errors (EIO) in 30 min'
        if reason is None or not os.environ.get('SYNTH_JOB_ID'):
            return
        moves = self.s.setdefault('relocations', [])
        if len(moves) >= c.get('max_relocations', 4) or (moves and now - moves[-1]['time'] < 1800):
            if not getattr(self, '_relocation_held', False):
                self._relocation_held = True
                self.c.log(f'Controller on {self.host} is unhealthy ({reason}) but not moving: {len(moves)} moves so far')
            return
        self.relocate(reason)

    def relocate(self, reason):
        nodes = read(self.ops / 'controller' / 'excluded-nodes.json', []) or []
        if self.host and self.host not in nodes:
            atomic(self.ops / 'controller' / 'excluded-nodes.json', sorted(nodes + [self.host]))
        latest = condor.latest(self.c, 'controller')
        generation = latest['generation'] + 1 if latest else 0
        q = None
        try:
            q = self.queue_view.get() if self.queue_view is not None else None
        except Exception:  # noqa: BLE001
            pass
        job = condor.submit_controller(self.c, generation, q=q)
        move = {'time': time.time(), 'from_host': self.host, 'from_job': os.environ.get('SYNTH_JOB_ID'),
                'to_job': job, 'generation': generation, 'reason': reason}
        self.s.setdefault('relocations', []).append(move)
        best_effort(append_jsonl, self.ops / 'controller' / 'relocations.jsonl', move)
        self.c.log(f'Controller moving off {self.host} ({reason}): submitted generation {generation} as {job}; '
                   f'this one stops after saving (undo the node exclusion in ops/controller/excluded-nodes.json)')
        self.stopping = True

    def main(self):
        self.c.log('Controller started.')
        c = self.c.config.get('controller', {})
        self.queue_view = condor.QueueView(self.c, refresh_sec=c.get('queue_refresh_sec', 15),
                                           max_age_sec=c.get('queue_max_age_sec', 60))
        while not self.stopping:
            ready = {}
            self._heartbeats = None
            self.loops += 1
            self.timings, tick = {}, time.monotonic()
            try:
                with self.timed('condor_queue'):
                    q = self.queue_view.get()
                if self.s['stage'] in ('smoke', 'running'):
                    with self.timed('ensure_serving'):
                        self.ensure_serving(q)
                    if self.s['stage'] == 'smoke':
                        self.ensure_workers(q, 1)
                with self.timed('backends'):
                    ready = self.backends(q)
                    self.check_backends(self.backends(q, include_stale=True))
                if self.loops % 20 == 1:  # re-publish lost inboxes (two stats per live attempt)
                    def republish(item):
                        inbox = self.ops / 'workers' / str(item['worker']) / 'inbox' / f'{item["id"]}.json'
                        if not inbox.exists() and not (self.raw(item) / 'worker-result.json').exists():
                            atomic(inbox, item)
                    with self.timed('republish'):
                        list(self.pool.map(republish, self.active()))
                with self.timed('harvest'):
                    self.harvest()
                with self.timed('export'):
                    self.export()
                if self.s['stage'] == 'running':
                    with self.timed('archive'):
                        self.archive_tick()
                if self.s['stage'] in ('smoke', 'running'):
                    with self.timed('stream'):
                        self.stream_tick()
                if self.s['stage'] == 'smoke':
                    self.smoke(q, ready)
                elif self.s['stage'] == 'running':
                    self.running(q, ready)
                elif self.s['stage'] == 'finalizing':
                    self.finalize(q)
                with self.timed('save'):
                    self.save(force=self.s['stage'] == 'complete')
                with self.timed('progress'):
                    self.progress(ready)
                self.errors = 0
                if self.s['stage'] == 'complete':
                    return
            except Exception as exc:
                self.errors += 1
                self.note_io_error(exc)
                self.c.log(f'Controller needs attention ({self.errors}): {exc!r}')
                best_effort(self.progress, ready, repr(exc))
                if self.errors >= 3:
                    best_effort(atomic, self.ops / 'controller' / 'needs-attention.json',
                                {'time': time.time(), 'error': repr(exc)})
            elapsed = time.monotonic() - tick
            if elapsed > 10:  # where a slow tick went (each tick should take well under its 15 s period)
                best_effort(append_jsonl, self.ops / 'controller' / 'slow-ticks.jsonl', {'time': time.time(),
                    'seconds': round(elapsed, 1), **{k: round(v, 2) for k, v in self.timings.items()}})
            best_effort(self.placement_check, elapsed)
            if not self.stopping:
                time.sleep(max(1.0, 15 - elapsed))

    def close(self):
        """Last snapshot before the process ends (a stop, a move, or completion)."""
        try:
            self.save(force=True)
        except Exception as exc:  # noqa: BLE001  (the journal still holds every attempt)
            self.c.log(f'Final state snapshot failed: {exc!r}')


def main():
    campaign = Campaign.load()
    if read(campaign.ops / 'completion.json', {}).get('status') == 'complete':
        return
    current = os.environ.get('SYNTH_JOB_ID')
    if not current:
        raise RuntimeError('The controller must run as a receipted HTCondor job (SYNTH_JOB_ID)')
    # A moved controller (Controller.relocate) starts while its predecessor saves and exits: wait up to 10 min.
    lease = MkdirLease(campaign.ops / 'controller' / 'lease', current,
                       alive=lambda job: job in condor.queue(campaign)).acquire(attempts=120)
    controller = None
    try:
        controller = Controller(campaign)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(controller, 'stopping', True))
        controller.main()
    finally:
        if controller is not None:
            controller.close_export()
            controller.close_archive()
            controller.close()
        lease.release()


if __name__ == '__main__':
    sys.exit(main())

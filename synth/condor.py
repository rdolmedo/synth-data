"""Owned HTCondor jobs: generated submit files, receipts, queue reads, scoped removal.

Every submission writes a receipt before anything depends on the job; only
receipted jobs are ever removed. ops/CANCELLED.json disables new submissions.
Never use condor_ssh_to_job (it kills jobs on this cluster).
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import threading
import time

from .common import PROXY_ENV, PYTHON, REPO, atomic, best_effort, read

ROLES = ('serving', 'worker', 'controller', 'tuning')


_RECEIPTS = {}  # receipt path -> content; receipts are written once and never change
_SUBMITTED = [float('-inf')]  # time.monotonic() of this process's last submission or adoption (QueueView)
_LISTINGS = {}  # jobs dir -> (monotonic time, receipt paths); submit/adopt here invalidate it


def receipts(campaign):
    """Every receipt of the campaign. The controller asks for the latest receipt of every worker
    several times per tick: the directory is listed at most once a second and each file read once."""
    folder = campaign.ops / 'jobs'
    listed = _LISTINGS.get(folder)
    if listed is None or time.monotonic() - listed[0] > 1:
        listed = _LISTINGS[folder] = (time.monotonic(), sorted(folder.glob('*-submission.json')))
    paths = listed[1]
    for path in paths:
        if path not in _RECEIPTS:
            record = read(path)
            if record is None:
                continue
            _RECEIPTS[path] = record
    return [_RECEIPTS[p] for p in paths if p in _RECEIPTS]


def owned_ids(campaign):
    return {r['job_id'] for r in receipts(campaign)}


def latest(campaign, role, index=0):
    rows = [r for r in receipts(campaign) if r['role'] == role and r.get('index', 0) == index]
    return max(rows, key=lambda r: r['generation']) if rows else None


# Receipts whose job left the queue for good (absent from two queue reads, receipted over 5 minutes earlier): queue()
# stops naming them. Per process; a restarted controller names every receipt once.
_GONE = collections.defaultdict(set)
_ABSENT = collections.Counter()


def queue(campaign):
    """Queue ads of this campaign's receipted jobs. Jobs submitted with the campaign attribute (+SynthCampaign) are
    matched by it; the others (adopted serving jobs, older receipts) are named by cluster until they are known to have
    left the queue. m1-hero (#235): one OR over all 206 clusters the campaign ever had, every tick, against a degraded
    schedd made ~60% of ticks slow."""
    rows = receipts(campaign)
    gone = _GONE[str(campaign.ops)]
    named = [r['job_id'] for r in rows if not r.get('campaign_attr') and r['job_id'] not in gone]
    attr = campaign.id if any(r.get('campaign_attr') for r in rows) else None
    owned = {r['job_id'] for r in rows}
    ads = {j: ad for j, ad in query(named, attr=attr).items() if j in owned}
    now = time.time()
    for r in rows:
        job = r['job_id']
        if job in ads:
            _ABSENT.pop(job, None)
        elif not r.get('campaign_attr') and job not in gone and now - r.get('time', now) > 300:
            _ABSENT[job] += 1
            if _ABSENT[job] >= 2:
                gone.add(job)
                _ABSENT.pop(job, None)
    return ads


def query(job_ids, attr=None):
    """Ads of these jobs, plus (attr) every job whose SynthCampaign is attr."""
    clusters = sorted({int(j.split('.')[0]) for j in job_ids})
    terms = [f'SynthCampaign == "{attr}"'] if attr else []
    if not clusters and not terms:
        return {}
    ads = []
    for start in range(0, max(1, len(clusters)), 200):
        expr = ' || '.join(terms + [f'ClusterId == {i}' for i in clusters[start:start + 200]])
        terms = []
        r = subprocess.run(['condor_q', '-constraint', expr, '-json'], capture_output=True, text=True,
                           check=True, timeout=120)
        ads += json.loads(r.stdout) if r.stdout.strip() else []
    return {f'{x["ClusterId"]}.{x["ProcId"]}': x for x in ads}


class QueueView:
    """The campaign's queue, read in a background thread every `refresh_sec`. get() returns the newest read if it
    started at most `max_age_sec` ago; otherwise it waits for the read in flight, and a failed read raises (a failing
    condor_q stays unknown: nothing acts on an empty queue). A read's time is when it started, so a job submitted
    after it is never taken as gone: the controller's 90/120 s grace after a submission exceeds max_age_sec. m1-hero
    (#235): condor_q took 15-22 s of a 15 s tick from 22:00 on, which delayed dispatch every tick."""

    def __init__(self, campaign, refresh_sec=15.0, max_age_sec=60.0, read_queue=None, clock=time.monotonic):
        self.campaign, self.refresh_sec, self.max_age_sec, self.clock = campaign, refresh_sec, max_age_sec, clock
        self.read_queue = read_queue or (lambda: queue(self.campaign))   # module lookup at call time (tests patch it)
        self.snapshot = None   # (started, ads)
        self.future = None
        self.started = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.reads = self.failures = 0
        self.last_error = None

    def _collect(self):
        if self.future is None or not self.future.done():
            return
        future, started, self.future = self.future, self.started, None
        try:
            ads = future.result()
        except Exception as exc:  # noqa: BLE001  (a fresh snapshot still serves; get() raises when none is left)
            self.failures += 1
            self.last_error = exc
            return
        self.reads += 1
        self.last_error = None
        if self.snapshot is None or started >= self.snapshot[0]:
            self.snapshot = (started, ads)

    def _launch(self):
        if self.future is None:
            self.started = self.clock()
            self.future = self.pool.submit(self.read_queue)

    def age(self):
        return None if self.snapshot is None else self.clock() - self.snapshot[0]

    def get(self):
        self._collect()
        age = self.age()
        if age is None or age >= self.refresh_sec:
            self._launch()
        if age is not None and age <= self.max_age_sec:
            return self.snapshot[1]
        # Nothing fresh enough: wait, as the synchronous query did, for a read that started after this process's last
        # submission (an older one may miss that job, and an absent job counts as ended after its 90 s grace).
        if self.started is None or self.started < _SUBMITTED[0]:
            if self.future is not None:
                try:
                    self.future.result()
                except Exception:  # noqa: BLE001
                    pass
                self._collect()
            self._launch()
        self.future.result()   # a failing condor_q raises: the queue stays unknown and nothing acts on it
        self._collect()
        return self.snapshot[1]

    def invalidate(self):
        """The next get() waits for a read that starts now."""
        self.snapshot = None


def host_of(ad):
    """Short host name of a running job's ad (RemoteHost slot1_7@gpu166.internal... -> gpu166)."""
    remote = (ad or {}).get('RemoteHost') or ''
    return remote.split('@')[-1].split('.')[0] or None


def role_hosts(campaign, q, role):
    """Counter: short host -> running owned jobs of this role (latest receipts only)."""
    latest = {}
    for r in receipts(campaign):
        if r['role'] == role:
            key = r.get('index', 0)
            if key not in latest or r['generation'] > latest[key]['generation']:
                latest[key] = r
    return collections.Counter(h for r in latest.values() if (q.get(r['job_id']) or {}).get('JobStatus') == 2
                               and (h := host_of(q[r['job_id']])))


def remove(campaign, job_id):
    if job_id not in owned_ids(campaign):
        raise AssertionError(f'Refusing to remove unowned job {job_id}')
    subprocess.run(['condor_rm', job_id], capture_output=True, text=True, timeout=60)


def bad_nodes() -> set:
    """tbshared's shared node list read at each submit (condor_cost.bad_nodes(): its file's BAD_NODES, so an edit
    reaches a long-lived controller without a restart, plus the live file, which also reaches frozen copies). Every
    synth submission excludes it on top of the campaign's own lists (uni-canary 11-18: controllers landed on gpu103,
    gpu123 and gpu166)."""
    from tbshared import condor_cost
    return set(condor_cost.bad_nodes())


def requirements(exclude, gpu, cuda_cap='10.0'):
    terms = [f'(TARGET.CUDACapability == {cuda_cap})'] if gpu else []
    terms += [f'(TARGET.UtsnameNodename != "{n}")' for n in sorted(set(exclude))]
    return ' && '.join(terms) or 'True'


def submit(campaign, role, *, index=0, generation=0, cpus, memory_gib, disk_gib, bid, gpus=0,
           exclude=(), args=(), extra_env=None, fill=True, cuda_cap='10.0'):
    """One owned job of a role. fill (tbshared 1.5, the user's rule: request both resources at the binding level,
    units = max(CPUs, GB / 16) + 24 per GPU): the job text goes through condor_cost.fill_plan once. GPU roles get the
    node-class caps of their requirements (CUDACapability == cuda_cap: 10.0 = B200, 9.0 = H100); CPU roles the
    placement check at their fixed bid.
    Workers pass fill=False: the controller decides their shape once per tick (worker_fill), for the whole batch.
    Every decision is appended to ops/jobs/fill.jsonl."""
    if role not in ROLES:
        raise ValueError(role)
    exclude = sorted(set(exclude) | bad_nodes())   # every role: the campaign's lists + tbshared's BAD_NODES (#451)
    if (campaign.ops / 'CANCELLED.json').exists():
        raise RuntimeError(f'{campaign.ops}/CANCELLED.json exists; submissions are disabled')
    stem = f'{role}-{index}-g{generation}'
    folder = campaign.ops / 'jobs'
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{stem}-submission.json'
    if receipt.exists():
        return read(receipt)['job_id']
    module = {'serving': 'synth.serve', 'worker': 'synth.worker', 'controller': 'synth.controller',
              'tuning': 'synth.tune'}[role]
    env = {'USER': 'user', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONNOUSERSITE': '1',
           'PYTHONPATH': str(REPO), 'TOKENIZERS_PARALLELISM': 'false', 'SYNTH_CAMPAIGN': str(campaign.frozen_path),
           'SYNTH_JOB_ID': '$(Cluster).$(Process)', **PROXY_ENV, **(extra_env or {})}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    logs = campaign.ops / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    sub = folder / f'{stem}.sub'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m {module} {' '.join(map(str, args))}
initialdir = {REPO}
output = {logs}/{stem}.$(Cluster).out
error = {logs}/{stem}.$(Cluster).err
log = {logs}/{stem}.condor.log
request_cpus = {cpus}
request_memory = {int(memory_gib * 1024)}
request_disk = {int(disk_gib)}G
{f"request_gpus = {gpus}" if gpus else ""}
+JobBatchName = "{campaign.slug}-{campaign.id}-{role}-{index}"
+SynthCampaign = "{campaign.id}"
+MaxRuntime = 1209600
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {requirements(exclude, gpus > 0, cuda_cap)}
queue
''')
    if fill:
        from tbshared import condor_cost
        body, n = condor_cost.split_queue(sub.read_text())
        body = condor_cost.fill_plan(body, n, None if gpus else (int(bid),), sorted(set(exclude)), where=str(sub),
                                     log=campaign.log)[0]
        sub.write_text(body + f'queue {n}\n')
        cpus, memory_gib = condor_cost.parse_shape(body)[:2]
    r = subprocess.run(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, check=True)
    job_id = re.search(r'cluster (\d+)', r.stdout).group(1) + '.0'
    _SUBMITTED[0] = time.monotonic()
    _LISTINGS.pop(folder, None)
    atomic(receipt, {'job_id': job_id, 'role': role, 'index': index, 'generation': generation, 'bid': bid,
                     'gpus': gpus, 'cpus': cpus, 'memory_gib': memory_gib, 'disk_gib': disk_gib,
                     'exclude': sorted(set(exclude)), 'time': time.time(), 'stdout': r.stdout, 'campaign_attr': True})
    best_effort(update_ledger, campaign)  # the receipt above is the ownership record
    campaign.log(f'Submitted owned {role} {index} generation {generation} at bid {bid}: {job_id}')
    return job_id


def adopt(campaign, receipt, serving_ops, generation):
    """Take over another campaign's running serving job: a receipt (ownership) at our next
    generation that records where the job publishes endpoints, metrics and control and under
    which generation it publishes them."""
    path = campaign.ops / 'jobs' / f'{receipt["role"]}-{receipt["index"]}-g{generation}-submission.json'
    if path.exists():
        return read(path)
    now = time.time()
    record = {**receipt, 'generation': generation, 'serving_generation': receipt.get('serving_generation', receipt['generation']),
              'serving_ops': str(serving_ops), 'adopted_at': now, 'time': now}
    _LISTINGS.pop(path.parent, None)
    _SUBMITTED[0] = time.monotonic()
    atomic(path, record)
    atomic(Path(serving_ops) / 'serving' / 'adopters' / f'{campaign.id}-{receipt["job_id"]}.json',
           {'campaign': campaign.id, 'ops': str(campaign.ops), 'job_id': receipt['job_id'], 'adopted_at': now})
    best_effort(update_ledger, campaign)
    campaign.log(f'Adopted serving {receipt["index"]} ({receipt["job_id"]}) from {serving_ops}')
    return record


def attached(serving_ops, job_id):
    """Campaigns that adopted this serving job (published under serving_ops) and have not let go of it: neither
    detached nor complete."""
    out = []
    for path in (Path(serving_ops) / 'serving' / 'adopters').glob('*.json'):
        a = read(path)
        if (a and a['job_id'] == job_id and not a.get('detached_at')
                and not (Path(a['ops']) / 'completion.json').exists()):
            out.append(a['campaign'])
    return out


def release(campaign, job_id):
    """Give back a serving job this campaign is done with. An adopted job is shared with the campaign that runs it
    (and maybe other adopters): detach, and remove it only when that campaign has completed and nobody else is
    attached; otherwise its own campaign releases it (2026-10-25: an adopter that removed it at its own release cut
    the still-training source's attempts and cost it a 15-minute restart)."""
    receipt = next((r for r in receipts(campaign) if r['job_id'] == job_id and r.get('adopted_at')), None)
    if receipt is None:
        remove(campaign, job_id)
        return
    source = Path(receipt['serving_ops'])
    mark = source / 'serving' / 'adopters' / f'{campaign.id}-{job_id}.json'
    state = read(mark, {})
    if not state.get('detached_at'):
        atomic(mark, {'campaign': campaign.id, 'ops': str(campaign.ops), 'job_id': job_id, **state,
                      'detached_at': time.time()})
        campaign.log(f'Detached from adopted serving {job_id}; {source.parent.name} releases it')
    if (source / 'completion.json').exists() and not attached(source, job_id):
        remove(campaign, job_id)


def update_ledger(campaign):
    atomic(campaign.ops / 'owned-jobs.json', {'campaign': campaign.id, 'jobs': receipts(campaign),
                                              'updated_at': time.time()})


def serving_bid(campaign, gpus):
    table = campaign.config['serving'].get('bid_per_job', {})
    bid = campaign.config['serving'].get('bid') or table.get(str(gpus))
    if not bid:
        raise ValueError(f'No serving bid configured for {gpus} GPUs')
    return bid


def submit_serving(campaign, index, generation):
    s = campaign.config['serving']
    gpus = s['gpus_per_job']
    return submit(campaign, 'serving', index=index, generation=generation, gpus=gpus,
                  cpus=s.get('cpus') or s['cpus_per_gpu'] * gpus,
                  memory_gib=s.get('memory_gib') or s['memory_gib_per_gpu'] * gpus,
                  disk_gib=s['disk_gib'], bid=serving_bid(campaign, gpus),
                  exclude=s.get('exclude_nodes', []) + read(campaign.ops / 'excluded-nodes.json', []),
                  args=('--index', index, '--generation', generation),
                  cuda_cap=s.get('cuda_capability', '10.0'))   # the spec's GPU type (9.0 = H100), as synth.tune


def worker_shape(overrides, index):
    """ops/worker-resource-overrides.json entries for one worker: keys are an index ("7") or an inclusive range
    ("0-39"); values may set slots, cpus, memory_gib and disk_gib. The exact index wins over a range."""
    shape = {}
    for key, value in overrides.items():
        lo, _, hi = str(key).partition('-')
        if lo.isdigit() and hi.isdigit() and int(lo) <= index <= int(hi):
            shape.update(value)
    shape.update(overrides.get(str(index), {}))
    return shape


from tbshared.condor_cost import fits  # noqa: E402  (the shared layer: one copy of the slot arithmetic)


def cheap_capacity(cpus, memory_gib, max_floor=25, exclude=(), status=None, per_host=None, placed=None):
    """CPU-only jobs of this shape that fit right now on nodes whose MinRunningPrice (floor per unit-hour) is at most
    `max_floor`, i.e. that a job bidding `max_floor` can match: CPU-only and A100 nodes at 25 (#242). Callers submit
    that many at the cheap bid and the rest at their normal bid, and rebid the cheap ones that still wait. `status`:
    condor_status -af:t UtsnameNodename MinRunningPrice Cpus Memory Gpus lines (tests). per_host (workers.max_per_host):
    a host counts at most per_host - placed[host] more jobs, `placed` being the campaign's jobs already there (#244:
    pre-agent failures 0.4% of attempts with one campaign worker on a host, 13% with five)."""
    if status is None:
        status = subprocess.run(['condor_status', '-const', 'PartitionableSlot =?= true', '-af:t', 'UtsnameNodename',
                                 'MinRunningPrice', 'Cpus', 'Memory', 'Gpus'], capture_output=True, text=True,
                                timeout=120, check=True).stdout
    total, skip, room = 0, set(exclude), collections.Counter()   # room: per host (a host has several slots)
    for line in status.splitlines():
        parts = line.split('\t')
        if len(parts) < 5 or parts[0] in skip:
            continue
        try:
            floor, free_cpus, free_mib = float(parts[1]), int(float(parts[2])), int(float(parts[3]))
            free_gpus = int(float(parts[4])) if parts[4] not in ('undefined', '') else 0
        except ValueError:
            continue
        if floor <= max_floor:
            room[parts[0]] += fits(free_cpus, free_mib, free_gpus, cpus, int(memory_gib * 1024))
    placed = placed or {}
    for host, n in room.items():
        total += min(n, max(0, per_host - placed.get(host, 0))) if per_host else n
    return total


def rebid(campaign_or_none, job_id, bid):
    """Raise a queued job's bid in place (condor_submit_bid sets JobPrio = bid - 1000; an idle job rebid this way
    matches within ~1 min, and the job keeps its id). `campaign_or_none`: check ownership against its receipts."""
    if campaign_or_none is not None and job_id not in owned_ids(campaign_or_none):
        raise AssertionError(f'Refusing to rebid unowned job {job_id}')
    subprocess.run(['condor_qedit', job_id, 'JobPrio', str(int(bid) - 1000)], capture_output=True, text=True,
                   timeout=60, check=True)


def worker_bid(w):
    """The bid a worker is submitted at: workers.cheap_bid when set (and below workers.bid), else workers.bid.

    A job only matches nodes whose MinRunningPrice (floor per unit-hour: B200 55, H100 35, A100 15/25, CPU-only nodes
    5-35) is at most its bid, and its job `rank` cannot steer it to cheap nodes here: the negotiator's
    NEGOTIATOR_PRE_JOB_RANK (fewest GPUs on the node, then the fullest node) orders machines before any job rank
    (2026-11-05 canaries, #242). So a cheap_bid of 25 keeps workers on CPU-only and A100 nodes, where a CPU unit costs
    15-25 instead of 35-55 and no waiting GPU job raises the running price; ensure_workers rebids a worker that stays
    idle for workers.cheap_bid_wait_sec to workers.bid (H100 nodes), so a worker never waits long for a cheap node."""
    cheap = w.get('cheap_bid')
    return min(cheap, w['bid']) if cheap else w['bid']


def submit_worker(campaign, index, generation, shape=None, cheap=True, avoid=()):
    """cheap=False: submit at workers.bid even when workers.cheap_bid is set (no cheap node has room, #242).
    avoid: hosts this submission excludes besides the configured and excluded nodes (the controller passes hosts
    already running workers.max_per_host of the campaign's workers, and its own host)."""
    w = {**campaign.config['workers'], **(shape or {})}
    harbor = campaign.config.get('harbor', {})
    cache_gib = harbor.get('package_cache_gib', 10) if harbor.get('package_cache', True) else 0   # its store's cap
    return submit(campaign, 'worker', index=index, generation=generation, cpus=w['cpus'],
                  memory_gib=w['memory_gib'], disk_gib=w['disk_gib'] + cache_gib, bid=worker_bid(w) if cheap else w['bid'],
                  exclude=w.get('exclude_nodes', []) + read(campaign.ops / 'excluded-nodes.json', []) + list(avoid),
                  args=('--index', index, '--slots', w['slots']), fill=False)   # shape: worker_fill, per tick


def worker_fill(campaign, n, exclude=()):
    """{'cpus', 'memory_gib'} for this tick's n new workers at the binding resource (tbshared condor_cost.fill_plan
    on the workers' ladder: cheap_bid -> bid), or {} when the shape stays as configured (workers.fill = false, filling
    would start any of them on a higher rung even capped, or the node classes are unknown). The decision is appended
    to ops/jobs/fill.jsonl."""
    from tbshared import condor_cost
    w = campaign.config['workers']
    if not w.get('fill', True) or n <= 0:
        return {}
    cheap = w.get('cheap_bid')
    ladder = (cheap, w['bid']) if cheap and cheap < w['bid'] else (w['bid'],)
    folder = campaign.ops / 'jobs'
    folder.mkdir(parents=True, exist_ok=True)
    body = (f'request_cpus = {w["cpus"]}\nrequest_memory = {int(w["memory_gib"] * 1024)}\n'
            f'+JobBatchName = "{campaign.slug}-{campaign.id}-worker"\n')
    body, _, rec = condor_cost.fill_plan(body, n, ladder, sorted(set(exclude) | bad_nodes()), where=str(folder / 'workers.sub'),
                                         log=campaign.log)
    if rec['decision'] not in ('filled', 'capped'):
        return {}
    cpus, memory_gib = condor_cost.parse_shape(body)[:2]
    return {'cpus': cpus, 'memory_gib': memory_gib}


def controller_avoid(campaign, q=None):
    """Hosts the controller job avoids: its own exclusions (ops/controller/excluded-nodes.json, from moves) and, by
    default (controller.avoid_worker_hosts), every host running one of the campaign's workers. m1-hero (#235): on nodes
    shared with its workers (gpu166, gpu190) ticks took 60-380 s from dirty-page writeback throttling."""
    c = campaign.config.get('controller', {})
    avoid = set(read(campaign.ops / 'controller' / 'excluded-nodes.json', []) or [])
    if c.get('avoid_worker_hosts', True):
        try:
            avoid |= set(role_hosts(campaign, queue(campaign) if q is None else q, 'worker'))
        except Exception as exc:  # noqa: BLE001  (placement is best effort; the submission is not)
            campaign.log(f'Controller placement: queue unreadable ({exc!r}); not avoiding worker hosts')
    return sorted(avoid)


def submit_controller(campaign, generation, q=None):
    c = campaign.config.get('controller', {})
    return submit(campaign, 'controller', generation=generation, cpus=c.get('cpus', 8),
                  memory_gib=c.get('memory_gib', 48), disk_gib=c.get('disk_gib', 64), bid=c.get('bid', 56),   # 64: archive shards are staged in scratch
                  exclude=c.get('exclude_nodes', ['gpu132', 'cpu101', 'gpu106']) + controller_avoid(campaign, q))

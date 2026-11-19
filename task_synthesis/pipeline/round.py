"""One audit round: sample specs, generate tasks, build and check them (with repair turns), judge the valid ones,
then score decontamination and diversity and write REPORT.md.

  python -m task_synthesis.pipeline.round submit --round r1 --n 100        # CPU job from a frozen checkout
  python -m task_synthesis.pipeline.round run --round r1 --n 100           # inside the job (resumable)
  python -m task_synthesis.pipeline.round report --round r1                # rewrite summary.json + REPORT.md

State lives in <ROOT>/rounds/<round>/: specs.jsonl, work/<spec>/ (replies, checks, record.json), tasks/<spec>/
(the Harbor task of the last attempt). A rerun skips specs that already have a record.
"""
import argparse
import asyncio
import bisect
import contextlib
import faulthandler
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')  # read when harbor_ns.builder is imported

from synth.common import PROXY_ENV, PYTHON, atomic, read  # noqa: E402
from tbshared import reslog  # noqa: E402  (the one cgroup sampler, #445 item 2)
from ..common import ROOT, frozen_checkout, pin_planned_run  # noqa: E402
from tbshared import serving  # noqa: E402
from . import explore, prompts, specs as specs_mod  # noqa: E402
from .claims import Claims  # noqa: E402
from .package import BASE_REF, PackageError, materialize, parse, rehost  # noqa: E402

ROUNDS = ROOT / 'rounds'
# generator requests in flight per round at most: DeepSeek-V4.1-Flash-1017 on one 4xB200 server peaks at ~2,700-3,000
# concurrent requests (synth/teachers/deepseek-v41-flash-1017.toml); rounds sharing a server should split it
MAX_IN_FLIGHT = 2500
# admission timing (Round.admit, Round.pool_inflight): the load cache, one /metrics read's timeout, the waits while the
# pool is full or its load unknown, and how long an admitted spec counts as load before its requests show in /metrics.
# Module constants so the admission load test (#238) can run the same gate on a compressed clock.
POOL_CACHE_S, METRICS_TIMEOUT_S, ADMIT_POLL_S, UNKNOWN_POLL_S = 15, 30, 15, 5
RECENT_ONESHOT_S, RECENT_WORKSPACE_S = 30, 120
# a re-planned round (production replan, #238): every job re-reads specs.jsonl when it changes, checked this often
SPECS_RELOAD_S = 60
# --real-assets: the author recipe for specs bound to a real asset (#118). #117 chose oneshot (#20 A/Bs: it ties wx at
# ~1/3 of wx's DeepSeek tokens); wx stays available. The 5th A/B's recipe (wy, author effort low, -4.1 pp vs student 1)
# was removed. Specs without an asset always get the one-shot author and student 1's settings (--effort-scope asset).
REAL_ASSET_PRESETS = {
    'off': {},                                    # no binding: every spec is a generated task
    'oneshot': {'author': 'oneshot'},             # 1st A/B (realasset): the asset's rendering in one prompt, ~60k/task
    'wx': {'author': 'workspace', 'workflow': 'lean3', 'judge_repairs': 1, 'judge_effort': 'high',   # 4th A/B (wxasset,
           'reasoning_effort': 'high', 'author_tokens': 128000, 'author_commands': 60},             # iteration 7), 210k
}
BASE_MANIFEST = Path('/data/aai/tbench/harbor_ns/bases/local_tasksynth-base_2.manifest.json')
TOOLS = ('git, curl, wget, tmux, vim, nano, jq, sqlite3, build-essential (gcc, g++, make), cmake, gdb, '
         'zip/unzip, xz, zstd, rsync, openssh-client, iproute2, netcat, dnsutils, procps, lsof, file, tree, bc, '
         'time, xxd, hexdump; `python` runs python3')


def system_prompt(minutes):
    m = json.loads(BASE_MANIFEST.read_text())
    pip = ', '.join(p.split('==')[0] for p in m['pip'] if p.split('==')[0] in
                    ('numpy', 'pandas', 'scipy', 'requests', 'pyyaml', 'pytest'))
    return prompts.GEN_SYSTEM.format(base_ref=BASE_REF, tools=TOOLS, python=m['python'][0].replace('python', ''),
                                     pip=pip, minutes=minutes)


def prompt_hash():
    text = prompts.GEN_SYSTEM + prompts.GEN_USER + prompts.REPAIR + prompts.JUDGE_SYSTEM + prompts.JUDGE_USER + \
        prompts.ASSET_USER + prompts.JUDGE_ASSET_CRITERION + prompts.JUDGE_ASSET_USER
    return hashlib.sha256(text.encode()).hexdigest()[:12]


INFRA_MARKS = ('No space left on device', 'nsenter: failed to execute /bin/sh', 'could not be pinned')


def asset_files(spec):
    """A spec bound to a real asset (task_synthesis.assets.bind export) -> (tar, generator text, judge text);
    (None, None, None) otherwise."""
    a = spec.get('asset')
    if not a:
        return None, None, None
    d = Path(a['dir'])
    return d / 'asset.tar.zst', (d / 'gen.txt').read_text(), (d / 'judge.txt').read_text()


def with_effort(extra, effort):
    """A request extra_body with a reasoning effort (1-100, or low=50 / high=75 / max=100)."""
    e = int(effort) if str(effort).isdigit() else effort
    kw = {**extra.get('chat_template_kwargs', {}), 'reasoning_effort': e}
    return {**{k: v for k, v in extra.items() if k != 'reasoning_effort'}, 'chat_template_kwargs': kw,
            **({'reasoning_effort': e} if isinstance(e, str) else {})}


class LostClaim(Exception):
    """This job no longer owns the spec's claim (taken over while the job stalled): drop the spec, write nothing."""


def infra_attempt(a):
    """An attempt (record entry or check result) that failed on infrastructure, not on the generator's package."""
    return bool(a.get('infra')) or any(m in e for e in a.get('errors') or [] for m in INFRA_MARKS)


class Round:
    specs_version = None   # (inode, size, mtime_ns) of the specs.jsonl this job runs (reload_specs)

    @property
    def counts(self):
        """#237: this job's counters in its heartbeat (claims are not admissions: a claimed spec may still wait for its
        binding and for room on the servers)."""
        return self.__dict__.setdefault('_counts', {'claimed': 0, 'admitted': 0, 'accepted': 0, 'cut': 0, 'errors': 0})

    def __init__(self, a):
        self.a, self.dir = a, ROUNDS / a.round
        self.work, self.tasks = self.dir / 'work', self.dir / 'tasks'
        if (self.dir / 'packed' / 'index.json').exists():   # roundpack: a packed round is finished; never write into it
            raise SystemExit(f'{self.dir} is packed (roundpack, #245): a packed round does not run again')
        self.shard, self.shards = getattr(a, 'shard', 0) or 0, getattr(a, 'shards', 1) or 1
        self.suffix = '' if self.shard == 0 else f'.s{self.shard}'
        for d in (self.work, self.tasks):
            d.mkdir(parents=True, exist_ok=True)
        self.claims = None          # set in run(); None: records are written without an ownership check (tests)
        self.foreign = set()                           # specs found claimed by another job
        self.journal_at, self.journal_pending = {}, {} # lost job -> its claim journal read up to, specs to look at again
        self.mine, self.lost = {}, set()               # this job's open claims (-> task), and claims it lost
        self.takeovers = {}                            # spec -> owner it was taken over from
        self.done, self.state = 0, 'working'
        self._last_beat = time.monotonic()

    def log(self, msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(self.dir / f'round{self.suffix}.log', 'a') as f:
            f.write(line + '\n')

    # ---- shared queue, load-aware admission and tail cut (defaults from 2026-11-02, user: "make sure our defaults
    # ensure that we are not wasteful"). The lean2 third A/B ran its 3 DeepSeek servers at ~46% of full load: static
    # per-round spec lists drained servers at different times, a fixed author count let the batch shrink whenever authors
    # sat in commands or builds, and the last ~5% of sessions ran on near-idle servers. Now every job of a round
    # (submit --jobs N) pulls specs from one queue (a spec is claimed by mkdir, atomic on Lustre), a new author session
    # starts only while the generator pool's in-flight requests are below --target-inflight per server, and once no spec
    # is left unclaimed and at most --tail-frac of them are still open for --tail-minutes, the open sessions are cut.
    #
    # #171: a claim names its owner (this job's incarnation), and every job beats a heartbeat from its event loop
    # (pipeline.claims). A job that dies or hangs stops beating; live jobs then take over its open claims and re-run
    # those specs from scratch (sweep_claims), so a lost job costs only its in-flight work.
    def claim(self, spec):
        sid = spec['spec_id']
        wd = self.work / sid
        if (wd / 'record.json').exists():
            return self.shard == 0          # finished earlier: shard 0 reports it, the others skip it
        if self.claims.claim(sid):
            return True
        if bool(self.a.resume) and self.shards == 1:   # one job re-runs its own round's claims: make them its own
            while not self.claims.owns(sid):
                cur = self.claims.current(sid)
                self.claims.take_over(sid, cur[0] if cur else 0)
            return True
        self.foreign.add(sid)              # another job's; sweep_claims takes it over if that job is lost
        return False

    def write_record(self, sid, rec, name='record.json'):
        """A spec's record (or error.json), written only while this job still owns the spec's claim: a job that stalled
        past the stale limit may have lost it to a takeover, and the new owner's run is the one that counts."""
        if self.claims is not None and not self.claims.owns(sid):
            raise LostClaim(sid)
        if sid in self.takeovers:
            rec = {**rec, 'taken_over_from': self.takeovers[sid]}
        atomic(self.work / sid / name, rec)

    async def heartbeat(self, every=30):
        """This job's heartbeat, from the event loop (not a thread): a job whose loop is blocked stops beating, and its
        claims go stale (m1-hero 17841191.3 kept running with 252 open specs and wrote nothing for 27 min). A gap
        longer than half the stale limit means other jobs may have taken over claims of this one: check them."""
        while True:
            gap = time.monotonic() - self._last_beat
            self._last_beat = time.monotonic()
            try:
                self.claims.beat(**self.beat_status())
            except OSError as e:
                self.log(f'heartbeat: {type(e).__name__}: {e}')
            if gap - every > self.claims.stale_after / 2:   # the gap beyond the sleep between beats
                self.log(f'event loop stalled for {gap:.0f} s (stale limit {self.claims.stale_after:.0f} s): checking '
                         f'this job\'s {len(self.mine)} claims')
                await self.drop_lost_claims()
            await asyncio.sleep(every)

    def beat_status(self):
        """This job's heartbeat fields (<round>/shards/<owner>.json, every 30 s; production status reads them, #237):
        state, open claims, finished specs, the counters (claimed / admitted / accepted / cut / errors, since this job
        started), live sessions (admitted, not finished), its generator and judge requests in flight now, the pool's load
        as last read, and the specs.jsonl version it runs."""
        gen, judge = getattr(self, 'gen', None), getattr(self, 'judge_llm', None)
        inflight = sum(gen.load.values()) if gen is not None else 0
        if judge is not None and judge is not gen:
            inflight += sum(judge.load.values())
        pool = getattr(self, '_pool', None)
        return {'state': self.state, 'open': len(self.mine), 'done': self.done, **self.counts,
                'sessions': getattr(self, 'open_authors', 0), 'inflight': inflight,
                'pool_load': None if pool is None else pool[0], 'pool_servers': None if pool is None else pool[1],
                'pool_read_at': getattr(self, '_pool_at', None),
                'target_inflight': self.admission('target_inflight') if hasattr(self, 'a') else None,
                'specs': len(getattr(self, 'spec_by_id', ()) or ()),
                'specs_version': list(self.specs_version) if self.specs_version else None, 'reload_specs': True}

    async def drop_lost_claims(self):
        mine = dict(self.mine)
        lost = [sid for sid, owned in zip(mine, await asyncio.to_thread(
            lambda: [self.claims.owns(sid) for sid in mine])) if not owned]
        for sid in lost:
            self.lost.add(sid)
            mine[sid].cancel()
        if lost:
            self.log(f'{len(lost)} claims were taken over by other jobs while this one stalled; dropped: {lost[:10]}')

    def hang_watchdog(self, limit):
        """Thread: once the event loop has not beaten for `limit` s, dump every thread's stack to stderr and exit, so a
        hung job frees its node (its claims went stale long before and were taken over)."""
        while True:
            time.sleep(min(30, limit / 4))
            gap = time.monotonic() - self._last_beat
            if gap > limit:
                try:
                    sys.stderr.write(f'{time.strftime("%H:%M:%S")} event loop blocked for {gap:.0f} s (> --hang-exit-'
                                     f'minutes {limit / 60:.0f}): exiting; thread stacks:\n')
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    sys.stderr.flush()
                finally:
                    os._exit(3)

    def spec_finished(self, sid):
        wd = self.work / sid
        return (wd / 'record.json').exists() or (wd / 'error.json').exists()

    def reset_spec(self, sid):
        """A taken-over spec runs from scratch: the lost job's replies, checks and task directory go."""
        wd = self.work / sid
        for f in wd.iterdir() if wd.is_dir() else ():
            if f.name == '.claim':
                continue
            if f.is_dir() and not f.is_symlink():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)
        shutil.rmtree(self.tasks / sid, ignore_errors=True)

    async def sweep_claims(self, every=60):
        while True:
            await asyncio.sleep(every)
            try:
                await self.take_stale()
            except Exception as e:  # noqa: BLE001  (the next sweep tries again)
                self.log(f'claim sweep: {type(e).__name__}: {e}')

    async def take_stale(self):
        """Take over the open claims of lost jobs: a job without a heartbeat within the stale limit (or one that exited)
        is read once from its claim journal, and each of its claims that is still open and still its own is taken over
        (mkdir take-<n>: one winner among the live jobs). A won claim re-runs its spec from scratch (after a stop: a
        tail_cut record). -> the spec ids taken over."""
        beats = await asyncio.to_thread(self.claims.heartbeats)
        now = time.time()
        gone = [o for o, b in beats.items() if o != self.claims.owner
                and (b.get('state') == 'exited' or now - b.get('time', 0) >= self.claims.stale_after)]
        offsets = {o: self.journal_at.get(o, 0) for o in gone}
        pending = {o: list(self.journal_pending.get(o, ())) for o in gone}
        if not gone:
            return []

        def scan():
            out, later = [], {}
            for o in gone:
                new, offsets[o] = self.claims.journal(o, offsets[o])
                for sid in dict.fromkeys(pending[o] + new):
                    if sid not in self.spec_by_id or self.spec_finished(sid):
                        continue
                    cur = self.claims.current(sid)
                    if cur is None or (cur[1] is not None and cur[1] != o):
                        continue          # never claimed (a lost mkdir race), or owned by another job now
                    stale = self.claims.stale(sid)
                    if stale is None:     # its owner file is still being written: look again next sweep
                        later.setdefault(o, []).append(sid)
                    elif self.claims.take_over(sid, stale[0]):
                        out.append((sid, o))
            return out, later
        taken, later = await asyncio.to_thread(scan)
        self.journal_at.update(offsets)
        for o in gone:
            self.journal_pending[o] = later.get(o, [])
        for sid, o in taken:
            self.foreign.discard(sid)
            t = asyncio.create_task(self.guarded(self.spec_by_id[sid], takeover=o))
            self.tasks_all.append(t)
            self.__dict__.setdefault('task_by_sid', {})[sid] = t
        if taken:
            self.log(f'took over {len(taken)} claims of lost jobs ({", ".join(sorted({o for _, o in taken}))}): '
                     f'{[s for s, _ in taken][:10]}')
        return [s for s, _ in taken]

    def lowest_live(self, states=None):
        """Is this the lowest shard (ties: owner id) among the live jobs, or among those in one of `states`?"""
        me = (self.shard, self.claims.owner)
        others = [(b.get('shard', 0), b['owner']) for b in self.claims.live_jobs()
                  if b['owner'] != self.claims.owner and b.get('state') != 'exited'
                  and (states is None or b.get('state') in states)]
        return all(me < o for o in others)

    async def may_exit(self):
        """After this job's own specs: exit unless specs of other jobs are still open and this is the one job that stays
        to take over their claims if their job is lost (the lowest shard among the draining jobs). A job that
        finished does not hold its node for nothing: the others exit at once."""
        await asyncio.to_thread(lambda: self.claims.heartbeats(max_age=0))
        if not self.lowest_live(states=('draining',)):
            self.log('own specs done; a lower drained job stays for the open specs of other jobs: exiting')
            return True
        foreign = list(self.foreign)
        finished = await asyncio.to_thread(lambda: [sid for sid in foreign if self.spec_finished(sid)])
        self.foreign.difference_update(finished)
        n = len(self.foreign)
        if not n:
            return True
        if not getattr(self, '_linger_logged', False):
            self._linger_logged = True
            self.log(f'own specs done; staying for the {n} open specs of other jobs (taken over if their job is lost)')
        return False

    async def pool_inflight(self):
        """(running + waiting requests summed over the generator pool's servers, number of servers); cached 15 s, or
        None if a server's load could not be read (admit() then waits: an unknown load is not an empty server).

        One read at a time per job (m1-hero restart, 2026-11-04 19:37): thousands of specs found the cache stale at once,
        each fired its own /metrics reads (~5 s each under load), most timed out and were counted as 0, and 12 new jobs
        admitted ~1,500 specs in 2 min onto servers already at their target."""
        lock = self.__dict__.setdefault('_pool_lock', asyncio.Lock())
        async with lock:
            if time.time() - getattr(self, '_pool_at', 0) < POOL_CACHE_S:
                return self._pool
            urls = list(self.gen.load)   # live pool members (clients keeps released servers' URLs: a swap must not count them)

            def read(u):
                base = u.rstrip('/').removesuffix('/v1')
                try:
                    t = urllib.request.urlopen(base + '/metrics', timeout=METRICS_TIMEOUT_S).read().decode()
                except OSError:
                    return None
                return sum(float(l.split()[-1]) for l in t.splitlines()
                           if l.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')))
            loads = await asyncio.gather(*(asyncio.to_thread(read, u) for u in urls))
            self._pool = None if any(x is None for x in loads) else (sum(loads), len(urls))
            self._pool_at = time.time()
            return self._pool

    def admission(self, key):
        """--target-inflight / --target-sessions, or a live override from <round>/admission.json, e.g.
        {"target_inflight": 2500}, re-read every 60 s: #124's measured step-up of the servers' load without a restart.
        Also --bind-window ({"bind_window": 1200}; m1-hero 2026-11-04: the 300 window held the round while binding's
        in-flight set, 600 slots of 4-18 min each, was twice as wide)."""
        now = time.time()
        if now - getattr(self, '_adm_at', 0) > 60:
            self._adm_at = now
            try:
                new = json.loads((self.dir / 'admission.json').read_text())
                new = new if isinstance(new, dict) else {}
            except (OSError, ValueError):
                new = {}
            if new != getattr(self, '_adm', {}):
                self.log(f'admission override: {new or "none (command-line targets)"}')
            self._adm = new
        v = self._adm.get(key)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else getattr(self.a, key, 0)

    async def admit(self):
        """Wait until the generator pool has room for one more spec: requests running + waiting below
        --target-inflight per server (every author), and open sessions below --target-sessions (agentic/workspace).
        One-shot specs are gated too (2026-11-04): a fixed --concurrency per job neither fills the servers when a job is
        lost nor holds back when a server is swapped out, and #124's step-up needs a load target."""
        if not self.admission('target_inflight'):
            return
        workspace = self.a.author in ('agentic', 'workspace')
        recent = self.__dict__.setdefault('_admitted', [])
        while True:
            target = self.admission('target_inflight')
            if not target:
                return
            pool = await self.pool_inflight()
            if pool is None:   # a server's load is unknown: wait for the next read rather than admit blind
                await asyncio.sleep(UNKNOWN_POLL_S)
                continue
            load, n = pool
            # specs admitted recently have not sent their first request yet: count them, for every job of the round.
            # A workspace session needs up to ~2 min (sandbox start, first prompt; 4th A/B, 2026-11-03: 4 jobs
            # overshot to ~2,700 in flight on a server targeted at 1,200 during the ramp); a one-shot spec sends its
            # first request at once, so it only has to outlast the 15 s load cache
            now = time.time()
            recent[:] = [t for t in recent if now - t < (RECENT_WORKSPACE_S if workspace else RECENT_ONESHOT_S)]
            # live sessions matter as much as requests in flight: every idle session's context competes for the
            # server's prefix cache (4th A/B: ~2,000 sessions per server -> prefix hit 0.45-0.5 at KV 31%; the 3rd
            # A/B's ~1,000 per server held 0.76-0.79). Each job counts its own sessions x jobs. One-shot specs hold no
            # session between requests, so only their requests count.
            sessions = (self.open_authors + len(recent)) * max(1, self.shards)
            cap = self.admission('target_sessions') if workspace else 0
            # this job's own CPU: a session that cannot get a check_task build slot holds a server session and makes
            # no progress (5th A/B, 2026-11-03: after DP-rank pinning doubled decode speed, 375 sessions per 64-CPU job
            # queued 108-128 builds behind 32 slots and the servers fell from ~750 to ~110 running requests each)
            bs = self.build_slots
            builds_waiting = len(getattr(bs, '_waiters', None) or [])
            if builds_waiting >= self.a.build_parallel:
                if not getattr(self, '_build_held', False):
                    self.log(f'admission held: {builds_waiting} sessions wait for a build slot ({self.a.build_parallel} '
                             f'slots); this job is CPU-bound -> run more jobs (--jobs) with fewer sessions each')
                self._build_held = True
            elif (load + len(recent) * max(1, self.shards) < target * max(1, n)
                    and (not cap or sessions < cap * max(1, n))):
                self._build_held = False
                recent.append(now)
                return
            await asyncio.sleep(ADMIT_POLL_S)

    # ---- streamed binding (#118). The production planner marks the specs that go to binding (bind=pending) in the
    # round's one seeded order; the bind job (assets.bind run --round) works through them in that order and publishes
    # each result as <round>/bind/<spec>.json. A pending spec is authored only once its result is in (bound: with its
    # real asset; otherwise as an ordinary generated task), and a spec starts only while at most --bind-window pending
    # specs before it in the queue still wait for their result. So every stretch of the stream keeps the planned
    # real-asset share up to that many delayed specs (the 330M pilot prefix and the ladder's fit data are time prefixes
    # of the generation, #49), and when binding falls behind, the round slows down instead of drifting. A count, not a
    # distance from the first unresolved spec: the m1-pilot start (2026-11-03) held the round at position 502 for 7+
    # min behind one slow binding (two rounds of searches for a new-class cell) while the servers ran at ~15% load.
    def bind_init(self, specs):
        """The binding view of `specs` (their queue order). Called again by reload_specs: results already read and the
        condition the waiting specs sleep on are kept."""
        self.bind_dir = self.dir / 'bind'
        self.pos = {sp['spec_id']: i for i, sp in enumerate(specs)}
        self.bind_pending = [sp['spec_id'] for sp in specs if sp.get('bind') == 'pending']
        self.bind_results = getattr(self, 'bind_results', None) or {}
        self.bind_head = 0
        while self.bind_head < len(self.bind_pending) and self.bind_pending[self.bind_head] in self.bind_results:
            self.bind_head += 1
        self.unresolved = [self.pos[sid] for sid in self.bind_pending   # sorted queue positions still waiting
                           if sid not in self.bind_results]
        if getattr(self, 'bind_cond', None) is None:
            self.bind_cond = asyncio.Condition()
        self.bind_window = getattr(self.a, 'bind_window', 300) or 0

    def unresolved_before(self, position):
        return bisect.bisect_left(self.unresolved, position)

    def bind_frontier(self):
        """Queue position of the first pending spec without a bind result (inf: none left)."""
        return self.pos[self.bind_pending[self.bind_head]] if self.bind_head < len(self.bind_pending) else float('inf')

    async def watch_bind(self, every=20):
        last_log = 0
        while self.bind_head < len(self.bind_pending):
            try:
                names = set(await asyncio.to_thread(os.listdir, self.bind_dir)) if self.bind_dir.exists() else set()
            except OSError:
                names = set()
            new = [sid for sid in self.bind_pending[self.bind_head:]
                   if sid not in self.bind_results and f'{sid}.json' in names]
            for sid in new:
                try:
                    self.bind_results[sid] = json.loads((self.bind_dir / f'{sid}.json').read_text())
                except (OSError, ValueError):
                    pass   # being replaced; next pass
            while self.bind_head < len(self.bind_pending) and self.bind_pending[self.bind_head] in self.bind_results:
                self.bind_head += 1
            if new:
                self.unresolved = [self.pos[sid] for sid in self.bind_pending[self.bind_head:] if sid not in self.bind_results]
            async with self.bind_cond:
                self.bind_cond.notify_all()
            if time.time() - last_log > 600 and self.bind_head < len(self.bind_pending):
                last_log = time.time()
                try:
                    beat = json.loads((self.bind_dir / 'HEARTBEAT').read_text())
                    age = f'{time.time() - beat["time"]:.0f}s ago'
                except (OSError, ValueError, KeyError):
                    age = 'none'
                self.log(f'binding: {len(self.bind_results)}/{len(self.bind_pending)} results, first unresolved at queue '
                         f'position {self.bind_frontier()}, {len(self.unresolved)} unresolved, bind heartbeat {age}')
            await asyncio.sleep(every)
        async with self.bind_cond:
            self.bind_cond.notify_all()

    async def bind_ready(self, spec):
        """Wait until this spec may start (see bind_init) -> the spec, with its asset if the binding found one."""
        sid = spec['spec_id']
        pending = spec.get('bind') == 'pending'
        if not pending and not self.bind_pending:
            return spec

        def ready():   # the window can be changed live: <round>/admission.json {"bind_window": N} (0: no window)
            window = self.admission('bind_window')
            return ((not pending or sid in self.bind_results)   # a spec a re-plan removed (not in pos) is cancelled
                    and (not window or self.unresolved_before(self.pos.get(sid, 0)) <= window))
        if not ready():
            async with self.bind_cond:
                await self.bind_cond.wait_for(ready)
        if not pending:
            return spec
        r = self.bind_results[sid]
        if r.get('status') == 'bound' and r.get('asset'):
            return {**spec, 'bind': 'bound', 'asset': r['asset']}
        return {**spec, 'bind': r.get('status') or 'unbound'}

    def open_specs(self, specs):
        """(specs not yet claimed by any job, specs claimed but without a record) across all jobs of the round."""
        unclaimed = opened = 0
        for sp in specs:
            wd = self.work / sp['spec_id']
            if (wd / 'record.json').exists() or (wd / 'error.json').exists():
                continue
            if (wd / '.claim').exists():
                opened += 1
            else:
                unclaimed += 1
        return unclaimed, opened

    def accepted(self, specs):
        """Valid, judge-passed tasks of the round so far (records are final once written: each is read once)."""
        seen = self.__dict__.setdefault('_seen', {})
        for sp in specs:
            sid = sp['spec_id']
            if sid not in seen and (self.work / sid / 'record.json').exists():
                try:
                    r = json.loads((self.work / sid / 'record.json').read_text())
                except (OSError, ValueError):
                    continue
                seen[sid] = r.get('status') == 'valid' and bool((r.get('judge') or {}).get('pass'))
        return sum(seen.values())

    async def watch_tail(self, specs, tasks, every=120):
        """Stop rules (user, 2026-11-02: keep the queue fed and stop as soon as there are enough tasks; a small bias
        toward faster specs is accepted, tails are not): --target-tasks accepted tasks across all jobs, or, when the
        specs run out first, the tail rule (--tail-frac open for --tail-minutes). Open sessions are cut (tail_cut)."""
        target = getattr(self.a, 'target_tasks', 0) or 0
        frac, minutes = getattr(self.a, 'tail_frac', 0), getattr(self.a, 'tail_minutes', 15)
        limit, calm = max(5, int(frac * len(specs))), 0
        while True:
            await asyncio.sleep(every)
            why = None
            if (self.dir / 'STOP').exists():   # the production driver: the stream holds enough tasks for the budget
                why = f'STOP file: {(self.dir / "STOP").read_text().strip()[:300]}'
            if why is None and target:
                n = await asyncio.to_thread(self.accepted, specs)
                if n >= target:
                    why = f'target reached: {n} accepted tasks >= --target-tasks {target}'
            if why is None and frac and self.a.author in ('agentic', 'workspace'):
                unclaimed, opened = await asyncio.to_thread(self.open_specs, specs)
                calm = calm + 1 if unclaimed == 0 and opened <= limit else 0
                if calm * every >= minutes * 60:
                    why = f'tail: nothing left to claim and {opened} open sessions (<= {limit}) for {minutes} min'
            if why:
                self.log(f'stop: {why}; cutting this job\'s open sessions')
                self.tail_cut = True
                for t in tasks:
                    t.cancel()
                # specs nobody started get a record, so the round is done (campaign stream --end); by the lowest live
                # shard (shard 0 may be the lost one), each claimed first so that two jobs never both write it
                if await asyncio.to_thread(lambda: (self.claims.heartbeats(max_age=0), self.lowest_live())[1]):
                    n = await asyncio.to_thread(self.mark_not_run, specs)
                    self.log(f'stop: {n} specs nobody started recorded as not_run')
                return

    def mark_not_run(self, specs):
        n = 0
        for sp in specs:
            sid = sp['spec_id']
            wd = self.work / sid
            if (wd / 'record.json').exists() or (wd / 'error.json').exists() or (wd / '.claim').exists():
                continue
            if self.claims.claim(sid):
                atomic(wd / 'record.json', {'spec_id': sid, 'status': 'not_run'})
                n += 1
        return n

    # ---- re-planned rounds (production replan, #238). The re-plan pre-claims the unstarted specs the jobs could start
    # now and replaces specs.jsonl atomically (started specs kept, in their order, then the new draws). Every job notices
    # the new file (inode/size/mtime) within SPECS_RELOAD_S and switches to it in place: specs that left the file and
    # are not this job's open claims are cancelled where they wait (binding, a session slot, the claim), new specs join
    # the queue, and the job's started sessions go on untouched. So no job has to be drained and replaced.
    def specs_stat(self):
        st = os.stat(self.dir / 'specs.jsonl')
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    async def watch_specs(self, specs, every=None):
        while True:
            await asyncio.sleep(every or SPECS_RELOAD_S)
            try:
                v = await asyncio.to_thread(self.specs_stat)
                if v != self.specs_version:
                    await self.reload_specs(specs, v)
            except Exception as e:  # noqa: BLE001  (a transient Lustre error: the next check tries again)
                self.log(f'specs reload: {type(e).__name__}: {e}')

    async def reload_specs(self, specs, version):
        """Switch this job to the current specs.jsonl. `specs` (the list watch_tail reads) is updated in place."""
        new = await asyncio.to_thread(self.load_specs)
        new_ids = {sp['spec_id'] for sp in new}
        old = self.spec_by_id
        removed = [sid for sid in old if sid not in new_ids]
        kept_open = [sid for sid in removed if sid in self.mine]   # started here: runs to its record
        tasks = self.__dict__.setdefault('task_by_sid', {})
        cancelled = 0
        for sid in removed:
            t = tasks.get(sid)
            if sid not in self.mine and t is not None and not t.done():
                t.cancel()
                cancelled += 1
        self.foreign.difference_update(removed)
        added = [] if self.tail_cut else [sp for sp in new if sp['spec_id'] not in old]
        specs[:] = new
        self.spec_by_id = {sp['spec_id']: sp for sp in new}
        for sid in kept_open:
            self.spec_by_id[sid] = old[sid]
        self.bind_init(new)
        for sp in added:
            t = asyncio.create_task(self.guarded(sp))
            tasks[sp['spec_id']] = t
            self.tasks_all.append(t)
        binding = getattr(self, 'binding', None)
        if self.bind_head < len(self.bind_pending) and (binding is None or binding.done()):
            self.binding = asyncio.create_task(self.watch_bind())
        async with self.bind_cond:
            self.bind_cond.notify_all()   # queue positions changed: waiting specs look again
        self.specs_version = version
        self.log(f'specs.jsonl changed (re-plan): {len(new)} specs, {len(removed)} removed ({cancelled} waiting here '
                 f'cancelled, {len(kept_open)} open here kept), {len(added)} new'
                 + (' (not started: the round is stopping)' if self.tail_cut else ''))

    def load_specs(self):
        path = self.dir / 'specs.jsonl'
        if not path.exists():
            drawn = specs_mod.sample(self.a.n, self.a.seed, getattr(self.a, 'minutes', None),
                                     specs_mod.parse_mix(getattr(self.a, 'minutes_mix', None) or specs_mod.DEFAULT_MIX),
                                     getattr(self.a, 'vocab', None))
            path.write_text(''.join(json.dumps(s) + '\n' for s in drawn))
        return [json.loads(l) for l in path.read_text().splitlines()]

    async def checked(self, task_dir):
        """check() with retries on infrastructure failures (full scratch, sandbox errors): those are not the
        generator's to repair."""
        from .check import check
        for i in range(3):
            async with self.build_slots:
                try:
                    chk = await check(self.store, task_dir, memory_mb=getattr(self.a, 'check_memory_mb', 0) or None,
                                      build_memory_mb=1024 * (getattr(self.a, 'build_memory_gb', 24) or 0))
                except Exception as e:  # noqa: BLE001  (sandbox infrastructure)
                    chk = {'ok': False, 'errors': [], 'infra': f'{type(e).__name__}: {e}'}
            if not chk.get('infra'):
                return chk
            await asyncio.sleep(60 * (i + 1))
        return chk

    def stored_replies(self, wd, old):
        """--resume: a spec whose attempts failed on infrastructure (r6: scratch full, the error read as a build
        failure) replays its saved replies; finished specs keep their record. -> ({k: reply file}, {k: usage})."""
        if not self.a.resume:
            return {}, {}
        if old and not (old['status'] in ('env_fail', 'infra_fail') and any(infra_attempt(a) for a in old['attempts'])):
            return None, None
        replies = {int(f.name.split('-')[1].split('.')[0]): f for f in wd.glob('attempt-*.reply.txt')}
        return replies, {a['k']: a.get('usage') for a in (old or {}).get('attempts', [])}

    async def one(self, spec):
        wd = self.work / spec['spec_id']
        rec_path = wd / 'record.json'
        old = read(rec_path) if rec_path.exists() else None
        stored, old_usage = self.stored_replies(wd, old)
        if old and stored is None or old and not self.a.resume:
            return old
        wd.mkdir(parents=True, exist_ok=True)
        asset_tar, asset_gen, asset_judge = asset_files(spec)
        agentic = self.a.author in ('agentic', 'workspace') and bool(spec.get('asset'))
        workspace = agentic and self.a.author == 'workspace'
        if agentic and asset_gen and len(asset_gen) > explore.RENDER_CHARS:
            asset_gen = (asset_gen[:explore.RENDER_CHARS] + '\n[... rendering cut: open the files in the sandbox ...]\n')
        messages = [{'role': 'system', 'content': system_prompt(spec['minutes'])},
                    {'role': 'user', 'content': specs_mod.render(spec, asset_gen, agentic=agentic,
                                                                 asset_first=self.a.order == 'asset', workspace=workspace,
                                                                 tokens=self.a.author_tokens,
                                                                 lean=self.workflow if self.lean else False,
                                                                 commands=self.a.author_commands if workspace
                                                                 else explore.MAX_COMMANDS)}]
        async with contextlib.AsyncExitStack() as stack:
            ex = None
            if agentic:
                ex = explore.Explorer(self.store, asset_tar, spec['spec_id'], live=wd / 'explore.live.jsonl',
                                      max_commands=self.a.author_commands if workspace else explore.MAX_COMMANDS,
                                      package_cache=getattr(self, 'author_cache', None),
                                      fetch_log=wd / 'author-fetch.jsonl')
                ex.token_budget = self.a.author_tokens
                if self.lean:
                    ex.output_cap = explore.LEAN_OUTPUT_CHARS
                if self.workflow in ('lean2', 'lean3'):
                    ex.nudge_at = explore.LEAN2_NUDGE_AT
                ex.workspace = workspace
                async with self.build_slots:
                    await stack.enter_async_context(ex)
            try:
                rec = await self.attempts_loop(spec, wd, messages, stored, old_usage, asset_tar, asset_judge, ex)
            finally:
                self.gen.release(spec['spec_id'])   # its DP rank takes a new session
            if ex is not None:
                rec['explore'] = {'commands': ex.commands, 'output_chars': ex.output_chars, 'checks': ex.checks,
                                  'command_secs': round(sum(x['secs'] for x in ex.log), 1)}
                atomic(wd / 'explore.json', {'log': ex.log, 'messages': messages})
                self.write_record(spec['spec_id'], rec)
        return rec

    async def workspace_check(self, ex, spec):
        """check_task: export /task, build and check it exactly as a submission is checked -> text for the author."""
        try:
            pkg = await explore.export_package(ex)
        except PackageError as e:
            return f'The task in /task cannot be read: {e}'
        d = self.tasks / '.check' / spec['spec_id']
        try:
            materialize(pkg, spec, {'round': self.a.round, 'check_task': True}, d, asset_tar=ex.asset_tar)
            chk = await self.checked(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if chk.get('infra'):
            return f'The check could not run (infrastructure, not your task): {chk["infra"][:300]}. Try again later.'
        warn = ''.join(f'\n- warning: {w}' for w in chk.get('warnings') or [])
        if chk['ok']:
            return f'All checks pass (built in {chk.get("build_seconds")} s).{warn}'
        return 'The checks failed:\n' + '\n'.join(f'- {e}' for e in chk['errors']) + warn

    async def author_reply(self, messages, ex, spec=None):
        """One generator reply. With an Explorer the generator has a bash tool: its calls run in the sandbox and the
        conversation goes on until it answers without a tool call (normally the task package). Once the sandbox
        budget is spent the tool is withdrawn."""
        usage_sum, reasoning_chars, nudges, submitted = {'prompt': 0, 'completion': 0}, 0, 0, False
        while True:
            if submitted:
                return '', reasoning_chars, usage_sum
            compact_at = explore.LEAN2_COMPACT_AT if self.workflow in ('lean2', 'lean3') else explore.LEAN_COMPACT_AT
            stop_at = explore.LEAN2_STOP_AT if self.workflow in ('lean2', 'lean3') else explore.STOP_AT
            if ex is not None and self.lean and ex.last_prompt > compact_at:
                explore.compact(messages, keep_last=explore.LEAN_KEEP_LAST)
                ex.last_prompt = 0
            elif ex is not None and not self.lean and ex.last_prompt > explore.COMPACT_AT:
                explore.compact(messages)
            spent = ex is not None and (ex.spent() or ex.last_prompt > stop_at)
            ws = ex is not None and getattr(ex, 'workspace', False)
            if ws and spent and self.workflow in ('lean2', 'lean3') and not ex.grace and ex.last_prompt <= stop_at:
                ex.grace = True   # a final allowance to finish /task instead of a lost attempt
                grace_tokens, grace_cmds = max(8000, self.a.author_tokens // 6), 8
                ex.token_budget = ex.completion + grace_tokens
                ex.max_commands = ex.commands + grace_cmds
                messages.append({'role': 'user', 'content': prompts.WORKSPACE_GRACE.format(commands=grace_cmds,
                                                                                           tokens=grace_tokens // 1000)})
                spent = False
            if ws:
                tools = [explore.SUBMIT_TOOL] if spent else [explore.BASH_TOOL, explore.CHECK_TOOL, explore.SUBMIT_TOOL]
            else:
                tools = [explore.BASH_TOOL] if ex is not None and not spent else None
            async with self.gen_slots:
                cap = self.a.max_tokens or (explore.REPLY_TOKENS if ws or tools else None)
                content, reasoning, usage, calls = await self.gen.chat_tools(
                    messages, tools, max_tokens=cap, affinity=(spec or {}).get('spec_id'),
                    extra=self.asset_author_extra if (spec or {}).get('asset') else None)
            usage_sum['prompt'] += usage.get('prompt') or 0
            usage_sum['completion'] += usage.get('completion') or 0
            usage_sum['finish'] = usage.get('finish')
            reasoning_chars += len(reasoning or '')
            if ex is not None:
                ex.last_prompt = usage.get('prompt') or 0
                ex.completion += usage.get('completion') or 0
            if not calls or ex is None:
                if ws and nudges < 2:    # workspace mode ends with submit, not with a reply
                    nudges += 1
                    nudge = 'Keep working with the tools; call submit when /task is ready.'
                    if usage.get('finish') == 'length':
                        nudge = (f'Your reply hit the {cap}-token limit for one reply before any tool call. Work in '
                                 'smaller steps (write long files over several commands). ' + nudge)
                    messages += [{'role': 'assistant', 'content': content or ''}, {'role': 'user', 'content': nudge}]
                    continue
                return content, reasoning_chars, usage_sum
            messages.append({'role': 'assistant', 'content': content or '',
                             'tool_calls': [{'id': c['id'], 'type': 'function',
                                             'function': {'name': c['name'], 'arguments': c['arguments']}}
                                            for c in calls]})
            for c in calls:
                cmd = explore.command_of(c)
                if c['name'] == 'submit':
                    submitted = True
                    out = 'Submitted: the task is being built, checked and reviewed.'
                elif c['name'] == 'check_task':
                    if ex.checks >= explore.MAX_CHECKS:
                        out = f'check_task budget used up ({explore.MAX_CHECKS}); call submit'
                    else:
                        ex.checks += 1
                        out = await self.workspace_check(ex, spec)
                elif ex.spent():
                    out = ('[budget used up: call submit now; /task is submitted as it is]' if ws else
                           '[sandbox budget used up: reply with the complete task in the output format, without a tool call]')
                elif cmd is None:
                    out = 'error: call the bash tool with {"command": "<shell command>"}'
                else:
                    out = await ex.run(cmd)
                messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': out})

    async def attempts_loop(self, spec, wd, messages, stored, old_usage, asset_tar, asset_judge, ex):
        attempts, pkg, chk, status = [], None, None, 'format_fail'
        prov = {'model': self.gen.model, 'endpoint': self.gen.name, 'prompts': prompt_hash(), 'round': self.a.round}
        reuse = True   # a stored reply answered the previous attempt's repair message; stop reusing once that changes
        for k in range(1 + self.a.repairs):
            if reuse and k in stored:
                content, usage = stored[k].read_text(), old_usage.get(k) or {}
                att = {'k': k, 'replayed': True, 'usage': usage}
            else:
                reuse = False
                t0 = time.time()
                content, reasoning_chars, usage = await self.author_reply(messages, ex, spec)
                (wd / f'attempt-{k}.reply.txt').write_text(content)
                att = {'k': k, 'gen_seconds': round(time.time() - t0), 'usage': usage, 'reasoning_chars': reasoning_chars}
            ws = ex is not None and getattr(ex, 'workspace', False)
            try:
                if ws:
                    pkg = await explore.export_package(ex)
                    atomic(wd / f'attempt-{k}.package.json', pkg)
                else:
                    if usage.get('finish') == 'length':
                        raise PackageError('your reply hit the output length limit before the task was complete; think '
                                           'less, keep files short, and generate bulk material from setup')
                    pkg = parse(content)
            except PackageError as e:
                att.update(stage='format', errors=[str(e)])
                if ex is not None:   # make room for the repaired package (agentic conversations run long)
                    explore.compact(messages, keep_last=2)
                    ex.last_prompt = 0
                attempts.append(att)
                if ws:
                    ex.max_commands = max(ex.max_commands, ex.commands + self.a.author_commands // 5)
                    ex.token_budget = max(ex.token_budget, ex.completion + self.a.author_tokens // 5)
                    messages.append({'role': 'user', 'content': prompts.WORKSPACE_REPAIR.format(errors=f'- {e}')})
                else:
                    messages += [{'role': 'assistant', 'content': content},
                                 {'role': 'user', 'content': prompts.REPAIR.format(errors=f'- {e}')}]
                pkg = None
                continue
            task_dir = materialize(pkg, spec, prov, self.tasks / spec['spec_id'], asset_tar=asset_tar)
            prev = read(wd / f'attempt-{k}.check.json') if att.get('replayed') else None
            if prev is not None and not infra_attempt(prev):
                chk = prev
            else:
                chk = await self.checked(task_dir)
                atomic(wd / f'attempt-{k}.check.json', chk)
                if att.get('replayed') and reuse:
                    reuse = False   # later stored replies answered an infrastructure error: set them aside
                    for j, f in stored.items():
                        if j > k:
                            f.rename(f.with_name(f'attempt-{j}.stale.reply.txt'))
                            c = wd / f'attempt-{j}.check.json'
                            if c.exists():
                                c.rename(c.with_name(f'attempt-{j}.stale.check.json'))
            att.update(stage='check', ok=chk['ok'], errors=chk['errors'], warnings=chk.get('warnings', []),
                       build_seconds=chk.get('build_seconds'), infra=chk.get('infra'))
            attempts.append(att)
            if chk.get('infra'):
                status = 'infra_fail'
                break
            if chk['ok']:
                status = 'valid'
                break
            status = 'env_fail'
            errors = '\n'.join(f'- {e}' for e in chk['errors'])
            if ws:
                ex.max_commands = max(ex.max_commands, ex.commands + self.a.author_commands // 5)
                ex.token_budget = max(ex.token_budget, ex.completion + self.a.author_tokens // 5)
                explore.compact(messages, keep_last=4)
                ex.last_prompt = 0
                messages.append({'role': 'user', 'content': prompts.WORKSPACE_REPAIR.format(errors=errors)})
            else:
                messages += [{'role': 'assistant', 'content': content},
                             {'role': 'user', 'content': prompts.REPAIR.format(errors=errors)}]
        rec = {'spec_id': spec['spec_id'], 'spec': spec, 'status': status, 'attempts': attempts,
               'repairs': len(attempts) - 1, 'meta': pkg['meta'] if pkg else None}
        e = getattr(self, 'efforts', None)
        if e and (e['author'] or e['judge']):   # what the author and the judge ran at (None: the server's default)
            on = e['scope'] == 'all' or bool(spec.get('asset'))
            rec['effort'] = {'author': e['author'] if on else None, 'judge': e['judge'] if on else None}
        if status == 'valid':
            from .judge import judge
            async with self.judge_slots:
                # asset-first: the generator chose the action, so the judge sees the spec without it
                # (and the companion materials, which it chose too: ag2's spec_mismatch flags were mostly those)
                jspec = ({**{k: v for k, v in spec.items() if k not in ('agent_action', 'agent_works_on')},
                          'assets': spec.get('assets', [])[:1]}
                         if self.a.order == 'asset' and spec.get('asset') else spec)
                rec['judge'] = await judge(self.judge_llm, jspec, pkg, chk, asset=asset_judge,
                                           extra=self.asset_judge_extra if spec.get('asset') else None)
            ws = ex is not None and getattr(ex, 'workspace', False)
            for jr in range(getattr(self.a, 'judge_repairs', 0) if ws else 0):
                # iteration 7: a rejection is not final; the author still has its sandbox and gets the reviewer's issues
                j = rec['judge']
                if j.get('pass') or 'scores' not in j:
                    break
                rec.setdefault('judge_history', []).append(j)
                low = ', '.join(k for k, v in j['scores'].items() if v is not None and v < 4) or 'none'
                flags = ', '.join(k for k, v in (j.get('flags') or {}).items() if v) or 'none'
                issues = '\n'.join(f'- {x}' for x in j.get('issues') or []) or '- (none listed)'
                ex.max_commands = max(ex.max_commands, ex.commands + self.a.author_commands // 5)
                ex.token_budget = max(ex.token_budget, ex.completion + self.a.author_tokens // 5)
                explore.compact(messages, keep_last=4)
                ex.last_prompt = 0
                messages.append({'role': 'user', 'content': prompts.WORKSPACE_JUDGE_REPAIR.format(issues=issues, low=low,
                                                                                                    flags=flags)})
                k, t0 = len(attempts), time.time()
                content, reasoning_chars, usage = await self.author_reply(messages, ex, spec)
                att = {'k': k, 'stage': 'judge_repair', 'gen_seconds': round(time.time() - t0), 'usage': usage,
                       'reasoning_chars': reasoning_chars}
                try:
                    pkg2 = await explore.export_package(ex)
                except PackageError as e:
                    att.update(ok=False, errors=[str(e)])
                    attempts.append(att)
                    break
                atomic(wd / f'attempt-{k}.package.json', pkg2)
                chk2 = await self.checked(materialize(pkg2, spec, prov, self.tasks / spec['spec_id'], asset_tar=asset_tar))
                atomic(wd / f'attempt-{k}.check.json', chk2)
                att.update(ok=chk2['ok'], errors=chk2['errors'])
                attempts.append(att)
                if not chk2['ok']:   # keep the task as judged
                    materialize(pkg, spec, prov, self.tasks / spec['spec_id'], asset_tar=asset_tar)
                    break
                pkg, chk = pkg2, chk2
                rec['meta'], rec['judge_repairs'] = pkg['meta'], jr + 1
                async with self.judge_slots:
                    rec['judge'] = await judge(self.judge_llm, jspec, pkg, chk, asset=asset_judge,
                                               extra=self.asset_judge_extra if spec.get('asset') else None)
            rec['instruction'] = pkg['instruction']
        self.write_record(spec['spec_id'], rec)
        return rec

    async def run(self):
        from .check import make_store
        from .llm import Endpoint, wait_healthy
        from harbor_ns.sandbox import apply_stack_limit
        stack = apply_stack_limit(getattr(self.a, 'sandbox_stack_kb', 8192))   # every build step and sandbox inherits it
        self.log(f'sandbox stack limit {f"{stack} KiB" if stack else "unchanged"}; check memory '
                 f'{getattr(self.a, "check_memory_mb", 0) or "task.toml memory_mb"} MB, build steps '
                 f'{getattr(self.a, "build_memory_gb", 24) or "no cap"} GB')
        self.log(f'waiting for endpoints {self.a.gen} / {self.a.judge}')
        await wait_healthy(self.a.gen)
        await wait_healthy(self.a.judge)
        # generator and judge each get --concurrency requests in flight; 0 (default) = every spec of the round at once,
        # up to MAX_IN_FLIGHT: a round's latency is one generation (~10-50 min of decode) and the server's aggregate
        # throughput grows with the batch, so holding specs back only idles it (2026-10-31: 300 in flight -> 20.7k gen
        # tok/s, KV 2%; 2,273 in flight over four rounds -> 64k gen tok/s, KV 34%, no preemptions)
        if not self.a.concurrency:
            self.a.concurrency = max(1, min(MAX_IN_FLIGHT, len(self.load_specs())))
        self.gen = Endpoint(self.a.gen, max_in_flight=2 * self.a.concurrency)
        effort = getattr(self.a, 'reasoning_effort', None)
        self.workflow = getattr(self.a, 'workflow', 'full')
        self.lean = self.workflow in ('lean', 'lean2', 'lean3')
        # --effort-scope asset (default, #118): --reasoning-effort and --judge-effort apply to specs bound to a real asset
        # only; the other specs keep the server's (max), the one-shot recipe of student 1 and the #25/#26 arms. 'all':
        # every spec (rounds before #118 applied them round-wide; wx0/wy0 had bound specs only, so both scopes agree)
        scope = getattr(self.a, 'effort_scope', 'asset') or 'asset'
        self.asset_author_extra = self.asset_judge_extra = None   # None: the endpoint's own extra_body
        self.efforts = {'scope': scope, 'author': effort, 'judge': getattr(self.a, 'judge_effort', None)}
        if effort:   # the author's reasoning budget (1-100, or low=50 / high=75 / max=100); the judge keeps the server's
            if scope == 'all':
                self.gen.extra = with_effort(self.gen.extra, effort)
            else:
                self.asset_author_extra = with_effort(self.gen.extra, effort)
        jeffort = getattr(self.a, 'judge_effort', None)
        self.judge_llm = (Endpoint(self.a.judge, max_in_flight=self.a.concurrency)
                          if self.a.judge != self.a.gen or effort or jeffort else self.gen)
        if jeffort:   # iteration 7: high agrees with max as well as max agrees with itself, at -34% judge tokens
            if scope == 'all':
                self.judge_llm.extra = with_effort(self.judge_llm.extra, jeffort)
            else:
                self.asset_judge_extra = with_effort(self.judge_llm.extra, jeffort)
        self.gen_slots = asyncio.Semaphore(self.a.concurrency)
        self.build_slots = asyncio.Semaphore(self.a.build_parallel)
        # agentic/workspace authors each hold a sandbox (memory, scratch, the cluster proxy) for their whole session:
        # --authors caps the sessions open at once, and a finished one makes room for the next spec
        self.open_authors = 0
        self.author_slots = asyncio.Semaphore(getattr(self.a, 'authors', 0) or len(self.load_specs()) or 1)
        self.judge_slots = asyncio.Semaphore(self.a.concurrency)
        self.store = make_store()
        self.cache = None
        if getattr(self.a, 'package_cache', 'on') == 'on':
            # one package cache per round job (synth.package_cache): builds' apt/pip repeats come from node-local disk
            from synth.package_cache import CacheProcess
            scratch = Path(os.environ.get('HARBOR_NS_LOCAL') or os.environ.get('TMPDIR') or '/tmp').parent
            self.cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', log=self.log,
                                            args=['--max-gib', str(getattr(self.a, 'package_cache_gib', 20))]).start()
        # authors' sandboxes (internet) use the cache too, through a package proxy in the sandbox (explore.Explorer):
        # apt and pip by default (author replay, 300 wy0 authors: proxy bytes -72%, versions identical); 'apt' keeps
        # pip off it (PIP_INDEX_URL is visible to the author: "Looking in indexes")
        mode = getattr(self.a, 'author_package_cache', 'apt+pip')
        self.author_cache = None if self.cache is None or mode == 'off' else mode
        specs = self.load_specs()
        self.tail_cut = False
        atomic(self.dir / f'config{self.suffix}.json', {**vars(self.a), 'prompts': prompt_hash(), 'host': socket.gethostname(),
                                          'code': str(Path(__file__).resolve().parents[2]), 'started': time.time()})
        self.log(f'round {self.a.round}: {len(specs)} specs, generator {self.gen.model}')
        self.spec_by_id = {sp['spec_id']: sp for sp in specs}
        self.bind_init(specs)
        if getattr(self.a, 'real_assets', None) == 'off' and (self.bind_pending or any(sp.get('asset') for sp in specs)):
            raise SystemExit('specs are bound to real assets or wait for binding, but --real-assets off')
        self.claims = Claims(self.dir, self.shard, stale_after=60 * (getattr(self.a, 'claim_stale_minutes', 10) or 10))
        self.specs_version = self.specs_stat()
        self.claims.beat(**self.beat_status())   # before the first claim
        self.log(f'claims: owner {self.claims.owner}, stale after {self.claims.stale_after / 60:.0f} min without a heartbeat')
        hang = 60 * (getattr(self.a, 'hang_exit_minutes', 0) or 0)
        if hang:
            threading.Thread(target=self.hang_watchdog, args=(hang,), daemon=True, name='hang-watchdog').start()
        beat = asyncio.create_task(self.heartbeat())
        watch = asyncio.create_task(self.watch_resources())
        # cgroup CPU/memory every minute, for sizing the round job (#242: condor gives only averages and RSS)
        reslog.start(self.dir / 'resources' / f'{self.claims.owner}.jsonl', 60)
        self.binding = asyncio.create_task(self.watch_bind())
        self.task_by_sid = {s['spec_id']: asyncio.create_task(self.guarded(s)) for s in specs}
        self.tasks_all = list(self.task_by_sid.values())
        tail = asyncio.create_task(self.watch_tail(specs, self.tasks_all))
        sweep = asyncio.create_task(self.sweep_claims())
        reload = asyncio.create_task(self.watch_specs(specs))
        try:
            while True:
                live = [t for t in self.tasks_all if not t.done()]
                if live:
                    self.state = 'working'
                    await asyncio.wait(live)   # takeovers started meanwhile join tasks_all: look again
                    continue
                self.state = 'draining'
                if await self.may_exit():
                    break
                await asyncio.sleep(30)
        finally:
            watch.cancel()
            tail.cancel()
            self.binding.cancel()
            sweep.cancel()
            reload.cancel()
            if self.cache is not None:
                await self.close_cache()
        if self.shard == 0:
            await asyncio.to_thread(report, self.a.round, True)   # in a thread: the heartbeat goes on
        self.state = 'exited'
        beat.cancel()
        try:
            self.claims.beat(**self.beat_status())
        except OSError:
            pass
        self.log(f'done: {self.done} specs finished by this job')

    async def guarded(self, spec, takeover=None):
        """One spec in this job: claim, wait for binding and admission, run, record. `takeover`: the spec's claim was
        won from a lost job (whose owner id this is), so it runs from scratch, or is recorded as tail_cut after a stop."""
        sid = spec['spec_id']
        claimed = takeover is not None
        fresh = True
        try:
            if takeover is not None:
                self.mine[sid] = asyncio.current_task()
                self.takeovers[sid] = takeover
                if self.tail_cut:
                    rec = {'spec_id': sid, 'status': 'tail_cut', 'orphaned_from': takeover}
                    self.write_record(sid, rec)
                    return self.finished(spec, rec)
                await asyncio.to_thread(self.reset_spec, sid)
            if not (self.work / sid / 'record.json').exists():
                spec = await self.bind_ready(spec)
            async with self.author_slots:
                if takeover is None:
                    if not self.claim(spec):
                        return None
                    claimed = True
                    self.mine[sid] = asyncio.current_task()
                if (self.work / sid / 'record.json').exists():
                    fresh = False   # finished by an earlier job: reported here (shard 0), not counted again
                else:
                    self.counts['claimed'] += 1
                    await self.admit()
                    self.counts['admitted'] += 1
                self.open_authors += 1
                try:
                    rec = await self.one(spec)
                finally:
                    self.open_authors -= 1
        except LostClaim:
            self.log(f'{sid}: claim lost to another job; dropped')
            return None
        except asyncio.CancelledError:
            if sid in self.lost:
                return None
            if not (self.tail_cut and claimed):
                raise
            rec = {'spec_id': sid, 'status': 'tail_cut'}
            try:
                self.write_record(sid, rec)
            except LostClaim:
                return None
        except Exception as e:  # noqa: BLE001
            rec = {'spec_id': sid, 'status': 'error', 'error': f'{type(e).__name__}: {e}',
                   'trace': traceback.format_exc()[-2000:]}
            try:
                self.write_record(sid, rec, 'error.json')
            except LostClaim:
                return None
        finally:
            if self.mine.get(sid) is asyncio.current_task():
                del self.mine[sid]
        return self.finished(spec, rec, count=fresh)

    def finished(self, spec, rec, count=True):
        self.done += 1
        j = rec.get('judge', {})
        st = rec.get('status') if count else None
        if st == 'valid' and j.get('pass'):
            self.counts['accepted'] += 1
        elif st == 'tail_cut':
            self.counts['cut'] += 1
        elif st == 'error':
            self.counts['errors'] += 1
        self.log(f'[{self.done}/{len(self.spec_by_id)}] {spec["spec_id"]} {rec["status"]} repairs={rec.get("repairs")} '
                 f'judge={"pass" if j.get("pass") else ("fail" if j else "-")}'
                 + (f' (taken over from {self.takeovers[spec["spec_id"]]})' if spec['spec_id'] in self.takeovers else ''))
        return rec

    async def close_cache(self):
        """Stop the package cache, keep its stats, and merge what it fetched into the shared seed (warm start of the
        next round job; synth.package_cache.publish)."""
        from synth.package_cache import publish
        stats = self.cache.stats()
        await self.cache.close()
        if stats:
            atomic(self.dir / f'pkgcache-stats{self.suffix}.json', stats)   # one per job (shard)
            self.log(f'package cache: {stats["requests"]} requests, {stats["hit"]} hits, {stats["miss"]} misses, '
                     f'{stats["revalidated"]} revalidated, {stats["upstream_requests"]} upstream requests, '
                     f'{stats["upstream_bytes"] / 2**30:.2f} GiB upstream')
        # one job per round merges into the seed: each merge re-reads and rewrites the whole 12 GiB tar, 16 jobs did
        # so one after another (~10 min of teardown), and two jobs were held for memory right after it (wx0 s0, wy0 s10)
        if stats and stats.get('miss', 0) >= 20 and self.shard == 0:
            try:
                summary = await asyncio.to_thread(publish, self.cache.root)
                self.log(f'package cache seed published: {summary["file"]} ({summary["entries"]} files, '
                         f'{summary["bytes"] / 2**30:.2f} GiB)')
            except Exception as e:  # noqa: BLE001  (the round's results do not depend on it)
                self.log(f'package cache seed publish failed: {type(e).__name__}: {e}')

    async def watch_resources(self, every=600):
        """Scratch and memory use in round.log: 144 workspace authors on 200 GiB filled the job's scratch and broke
        the sandboxes' overlays (ws7, 2026-11-01: '/task does not exist' after 8 checks), and condor's DiskUsage does not
        show it."""
        scratch = os.environ.get('HARBOR_NS_LOCAL') or os.environ.get('TMPDIR') or '/tmp'
        while True:
            await asyncio.sleep(every)
            try:
                du = shutil.disk_usage(scratch)
                du_proc = await asyncio.create_subprocess_exec('du', '-sx', '--block-size=1G', scratch,
                                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                try:
                    out, _ = await asyncio.wait_for(du_proc.communicate(), 300)
                    used = out.split()[0].decode() if out else '?'
                except asyncio.TimeoutError:
                    du_proc.kill(); used = '>300s'
                mem = Path('/proc/meminfo').read_text().split('\n')[:3]
                bs = self.build_slots
                building, waiting = self.a.build_parallel - bs._value, len(getattr(bs, '_waiters', None) or [])
                self.log(f'resources: scratch du {used} GiB; filesystem {du.used / 2**30:.0f}/{du.total / 2**30:.0f} GiB used '
                         f'({du.free / 2**30:.0f} free); authors open {self.open_authors}; builds {building} running, '
                         f'{waiting} waiting for a slot; {" ".join(mem[2].split()[:2])}')
            except Exception as e:  # noqa: BLE001
                self.log(f'resources: {type(e).__name__}: {e}')


def collect(round_name):
    """Every spec's record.json, then error.json of the specs without one; packed or loose (roundpack.RoundFiles)."""
    from .roundpack import RoundFiles
    out = RoundFiles(ROUNDS / round_name).outcomes()
    return [r for r, _ in out.values() if r is not None] + [e for r, e in out.values() if r is None and e is not None]


def pct(a, b):
    return f'{a}/{b} ({100 * a / b:.0f}%)' if b else f'{a}/0'


# near-duplicate pairs and nearest neighbours compare every pair of tasks in pure Python: fine for an audit round, but
# shard 0 of m1-hero (60k valid tasks) sat in it for 30+ min after its last spec, holding its node, and never finished.
# Above this many valid tasks the job's own report skips them (the stream producer checks near-duplicates as it
# publishes); `round report` computes the pairs with the candidate filter (textsim.near_duplicates_fast).
REPORT_PAIRS_MAX = 3000


def report(round_name, in_job=False):
    from collections import Counter, defaultdict
    from .textsim import decon, near_duplicates, near_duplicates_fast
    d = ROUNDS / round_name
    recs = collect(round_name)
    n = len(recs)
    st = Counter(r['status'] for r in recs)
    valid = [r for r in recs if r['status'] == 'valid']
    judged = [r for r in valid if 'judge' in r and 'scores' in r['judge']]
    passed = [r for r in judged if r['judge']['pass']]
    first_try = sum(1 for r in valid if r['repairs'] == 0)
    texts = {r['spec_id']: r['instruction'] for r in valid}
    dc = decon(texts) if texts else {}
    pairs_skipped = in_job and len(texts) > REPORT_PAIRS_MAX
    if pairs_skipped:
        pairs, nearest = [], {}
    elif len(texts) > REPORT_PAIRS_MAX:
        pairs, nearest = near_duplicates_fast(texts), {}
    else:
        pairs, nearest = near_duplicates(texts) if len(texts) > 1 else ([], {})
    lines = [f'# Audit round {round_name}', '',
             f'Generated {time.strftime("%Y-%m-%d %H:%M")} from {d}.', '', '## Funnel', '',
             '| Stage | Count |', '| --- | --- |',
             f'| Specs | {n} |',
             f'| Parsed (any attempt) | {n - st["format_fail"] - st["error"]} |',
             f'| Environment valid | {pct(len(valid), n)} (first try {first_try}) |',
             f'| Judge pass | {pct(len(passed), n)} of specs; {pct(len(passed), len(judged))} of judged |',
             f'| Judge errors (no verdict; rejudge) | {sum(1 for r in valid if "scores" not in r.get("judge", {}))} |',
             f'| Status counts | {dict(st)} |', '']
    # Failure reasons at the last attempt of non-valid specs.
    reasons = Counter()
    for r in recs:
        if r['status'] in ('env_fail', 'format_fail') and r.get('attempts'):
            for e in r['attempts'][-1].get('errors', []):
                reasons[re.sub(r'[:(].*', '', e.split('\n')[0])[:90]] += 1
    if reasons:
        lines += ['## Last-attempt failure reasons', '', '| Reason | Count |', '| --- | --- |']
        lines += [f'| {k} | {v} |' for k, v in reasons.most_common(15)] + ['']
    first = Counter()
    for r in recs:
        for att in r.get('attempts', [])[:1]:
            for e in att.get('errors', []):
                first[re.sub(r'[:(].*', '', e.split('\n')[0])[:90]] += 1
    if first:
        lines += ['## First-attempt failure reasons', '', '| Reason | Count |', '| --- | --- |']
        lines += [f'| {k} | {v} |' for k, v in first.most_common(15)] + ['']
    if judged:
        keys = list(dict.fromkeys(k for r in judged for k in r['judge']['scores']))   # asset tasks add asset_use
        lines += ['## Judge scores (valid tasks)', '', '| Criterion | Mean | Share >= 4 |', '| --- | --- | --- |']
        for k in keys:
            vals = [r['judge']['scores'][k] for r in judged if r['judge']['scores'].get(k) is not None]
            if not vals:
                continue
            lines.append(f'| {k} | {sum(vals) / len(vals):.2f} | {sum(v >= 4 for v in vals) / len(vals):.0%} |')
        flags = Counter(f for r in judged for f, v in r['judge']['flags'].items() if v)
        lines += ['', f'Flags: {dict(flags)}', '']
        ratio = [r['judge']['estimated_minutes'] / r['spec']['minutes'] for r in judged
                 if isinstance(r['judge'].get('estimated_minutes'), (int, float)) and r['spec'].get('minutes')]
        if ratio:
            ratio.sort()
            lines += [f'Judge-estimated / target minutes: median {ratio[len(ratio) // 2]:.2f}, '
                      f'p10 {ratio[len(ratio) // 10]:.2f}, p90 {ratio[9 * len(ratio) // 10]:.2f}', '']
    by = defaultdict(lambda: [0, 0, 0])
    for r in recs:
        spec = r.get('spec') or {}
        keys = [f'bucket:{spec.get("bucket")}', f'action:{spec.get("agent_action")}',
                f'kind:{(r.get("meta") or {}).get("kind")}']
        if spec.get('asset'):
            keys += [f'class:{spec["assets"][0]["class"]}', f'source:{spec["asset"].get("source")}']
        for key in keys:
            by[key][0] += 1
            by[key][1] += r['status'] == 'valid'
            by[key][2] += bool(r.get('judge', {}).get('pass'))
    lines += ['## By slice', '', '| Slice | Specs | Valid | Judge pass |', '| --- | --- | --- | --- |']
    lines += [f'| {k} | {v[0]} | {v[1]} | {v[2]} |' for k, v in sorted(by.items())] + ['']
    flagged = {k: v for k, v in dc.items() if v['flagged']}
    nn = sorted(nearest.values())
    lines += ['## Decontamination and diversity', '',
              f'- Decon (TF-IDF cosine >= 0.5 or 8-gram containment >= 0.1 vs TB2.1 + TB-Lite): {len(flagged)} flagged'
              + (f': {flagged}' if flagged else ''),
              f'- Max cosine to an eval task: {max((v["cosine"] for v in dc.values()), default=0):.2f}',
              (f'- Near-duplicate pairs: skipped in the round job ({len(texts)} valid tasks > {REPORT_PAIRS_MAX}); '
               f'run `round report --round {round_name}`' if pairs_skipped else
               f'- Near-duplicate pairs (cosine >= 0.6): {len(pairs)}' + (f': {pairs[:10]}' if pairs else '')),
              (f'- Nearest-neighbour cosine within the round: median {nn[len(nn) // 2]:.2f}, max {nn[-1]:.2f}' if nn else
               f'- Nearest-neighbour cosine within the round: not computed (> {REPORT_PAIRS_MAX} valid tasks)'
               if len(texts) > REPORT_PAIRS_MAX else '- Nearest-neighbour cosine within the round: median 0.00, max 0.00'),
              '']
    gen_tok = [a['usage']['completion'] for r in recs for a in r.get('attempts', []) if a.get('usage')]
    builds = [a['build_seconds'] for r in recs for a in r.get('attempts', []) if a.get('build_seconds') is not None]
    lines += ['## Cost', '',
              f'- Generator completion tokens per attempt: mean {sum(gen_tok) / max(1, len(gen_tok)):.0f}, '
              f'p90 {sorted(gen_tok)[9 * len(gen_tok) // 10] if gen_tok else 0}, max {max(gen_tok, default=0)}, '
              f'truncated replies {sum(1 for r in recs for a in r.get("attempts", []) if (a.get("usage") or {}).get("finish") == "length")}',
              f'- Build+check seconds per attempt: mean {sum(builds) / max(1, len(builds)):.0f}, max {max(builds, default=0)}', '']
    lines += ['## Tasks', '', '| Spec | Slug | Kind | Action | Status | Repairs | Judge | Scores (cl/co/so/di/re/or) |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(recs, key=lambda r: r['spec_id']):
        meta, j = r.get('meta') or {}, r.get('judge', {})
        sc = '/'.join(str(v) for v in j.get('scores', {}).values())
        lines.append(f'| {r["spec_id"]} | {meta.get("slug", "")} | {meta.get("kind", "")} | '
                     f'{(r.get("spec") or {}).get("agent_action", "")} | {r["status"]} | {r.get("repairs", "")} | '
                     f'{"pass" if j.get("pass") else ("fail" if j else "")} | {sc} |')
    (d / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic(d / 'summary.json', {'specs': n, 'status': dict(st), 'valid': len(valid), 'judged': len(judged),
                                'passed': len(passed), 'first_try_valid': first_try, 'decon_flagged': len(flagged),
                                'near_duplicate_pairs': len(pairs), 'decon_flagged_ids': sorted(flagged),
                                'near_duplicates': pairs, **({'near_duplicates_skipped': True} if pairs_skipped else {}),
                                'time': time.time()})
    print(d / 'REPORT.md')


async def rejudge(a):
    """Judge again the valid tasks of a round that have no verdict (judge errors), with --judge."""
    from .judge import judge
    from .llm import Endpoint, wait_healthy
    d = ROUNDS / a.round
    if (d / 'packed' / 'index.json').exists():
        raise SystemExit(f'{d} is packed (roundpack, #245): rejudge writes record.json; not on a packed round')
    await wait_healthy(a.judge)
    llm = Endpoint(a.judge, max_in_flight=a.concurrency)
    todo = []
    for p in sorted((d / 'work').glob('*/record.json')):
        rec = json.loads(p.read_text())
        if rec['status'] == 'valid' and 'scores' not in rec.get('judge', {}):
            todo.append((p, rec))
    print(f'rejudging {len(todo)} tasks with {a.judge}', flush=True)
    slots = asyncio.Semaphore(a.concurrency)

    async def one(p, rec):
        wd = p.parent
        k = rec['attempts'][-1]['k']
        pkg = parse((wd / f'attempt-{k}.reply.txt').read_text())
        chk = json.loads((wd / f'attempt-{k}.check.json').read_text())
        async with slots:
            rec['judge'] = await judge(llm, rec['spec'], pkg, chk, asset=asset_files(rec['spec'])[2])
        rec.setdefault('rejudged', []).append({'judge': a.judge, 'time': time.time()})
        atomic(p, rec)
        print(rec['spec_id'], 'pass' if rec['judge'].get('pass') else 'fail', rec['judge'].get('scores'), flush=True)

    await asyncio.gather(*(one(p, rec) for p, rec in todo))
    report(a.round)


# `submit` options that shape the condor job, not the job's own run (never forwarded to it)
SUBMIT_ONLY = ('op', 'bid', 'cheap_bid', 'cpus', 'memory_gib', 'disk_gib', 'jobs', 'shard', 'shards', 'fill')


def submit(a, module='round', d=None, name=None):
    """CPU job running `python -m task_synthesis.pipeline.<module> run <a's options>` from a frozen checkout."""
    d = d or ROUNDS / a.round
    if module == 'round':   # a production run's round, also when submitted by hand: the run's registry root
        pin_planned_run(a.round)
    if module == 'round' and (d / 'assets').exists() and not (d / 'specs.jsonl').exists():
        raise SystemExit(f'{d} has asset bindings but no specs.jsonl (failed export?); refusing to sample fresh specs')
    code = frozen_checkout()
    name = name or f'round-{a.round}'
    d.mkdir(parents=True, exist_ok=True)
    args = ' '.join(f'--{k.replace("_", "-")}' + ('' if v is True else f' {v}') for k, v in vars(a).items()
                    if k not in SUBMIT_ONLY
                    and v is not None and v is not False)
    jobs = getattr(a, 'jobs', 1) or 1
    if jobs > 1:   # one queue, N jobs: each claims specs as it has room (Round.claim)
        args += ' --shard $(Process) --shards ' + str(jobs)
    env = {'USER': os.environ.get('USER', 'user'), 'HOME': os.environ.get('HOME', '/home/user'),
           'PYTHONNOUSERSITE': '1', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONPATH': str(code),
           'HARBOR_NS_PERSIST_IMAGES': '0', 'HARBOR_NS_BUILD_CPUS': '2', **PROXY_ENV,
           **serving.forwarded()}   # TASKSYNTH_SERVING_ROOT when set here: the job's llm clients read that registry
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = d / 'job.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export HARBOR_NS_LOCAL=$_CONDOR_SCRATCH_DIR/harbor_ns TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $_CONDOR_SCRATCH_DIR/tmp; exec {PYTHON} -u -m task_synthesis.pipeline.{module} run {args}'"
initialdir = {code}
output = {d}/job.$(Cluster).$(Process).out
error = {d}/job.$(Cluster).$(Process).err
log = {d}/job.condor.log
request_cpus = {a.cpus}
request_memory = {a.memory_gib * 1024}
request_disk = {getattr(a, 'disk_gib', 200) + (getattr(a, 'package_cache_gib', 20) if getattr(a, 'package_cache', 'on') == 'on' else 0)}G
+JobBatchName = "tasksynth-{name}"
+MaxRuntime = 172800
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in ROUND_EXCLUDE)}
''')
    n = jobs if module == 'round' else 1
    body, shape = fill_shards(sub.read_text(), n, a, sub)
    if module == 'round':   # the shared memory net with AUTO_RELEASE: request_memory grows 25% per memory hold
        from tbshared.condor_cost import memory_net
        body = memory_net(body)
    sub.write_text(f'{body}{auto_release(module, a)}queue {n}\n')
    cheap = cheap_split(n, a, *shape)
    r = subprocess.run(['condor_submit_bid', str(a.cheap_bid if cheap else a.bid), str(sub)], capture_output=True,
                       text=True, check=True)
    cluster = re.search(r'cluster (\d+)', r.stdout).group(1)
    job = cluster + '.0'
    for proc in range(cheap, n) if cheap else ():   # the rest at the normal bid
        subprocess.run(['condor_qedit', f'{cluster}.{proc}', 'JobPrio', str(a.bid - 1000)], capture_output=True,
                       text=True, timeout=60, check=True)
    atomic(d / 'job.json', {'job_id': job, 'code': str(code), 'time': time.time(), 'args': vars(a), 'cheap_jobs': cheap,
                            'shape': {'cpus': shape[0], 'memory_gib': shape[1]}})
    if module == 'round' and cheap:
        try:   # the shared per-user watcher raises shards still idle at the cheap bid (tbshared condor_cost)
            from tbshared.condor_cost import ensure_watcher
            ensure_watcher()
        except Exception as e:  # noqa: BLE001  (the driver's rebid_round still raises the first wait)
            print(f'bid-ladder watcher not ensured ({e!r})', file=sys.stderr)
    print(job)


def auto_release(module, a=None):
    """periodic_release on memory holds for round shards (tbshared condor_cost.AUTO_RELEASE, #378 port): a restarted
    shard is safe, because the open claims of a lost job are taken over (Round.take_stale) and finished specs are
    never redone. With --cheap-bid the shards also carry the shared bid ladder's attributes (--cheap-bid -> --bid after
    300 s idle; the driver's rebid_round uses [round] cheap_bid_wait_min, whichever acts first), so the watcher job raises any shard left idle at the cheap bid, a released one
    included (the driver's one-off rebid_round only sees the first wait). Other modules are submitted as before."""
    if module != 'round':
        return ''
    from tbshared.condor_cost import AUTO_RELEASE, ladder_lines
    out = AUTO_RELEASE
    if a is not None and getattr(a, 'cheap_bid', 0) and a.cheap_bid < a.bid:
        out += ladder_lines((a.cheap_bid, a.bid), (300,))
    return out


# Round jobs never run on these nodes (broken /data I/O, bad GPUs or black holes)
ROUND_EXCLUDE = ['gpu132', 'gpu110', 'cpu105', 'cpu301', 'gpu180', 'gpu158', 'cpu208', 'gpu109']


def fill_shards(body, n, a, sub):
    """(body, (cpus, memory_gib)): the jobs' shape at the binding resource (tbshared 1.5, the user's rule: units =
    max(CPUs, GB / 16), so both are requested at the binding level), decided for all n jobs on their ladder
    (--cheap-bid -> --bid, or --bid alone): filled only where it does not start any job on a higher rung, else capped
    at the cheap tier's free GB per free CPU, else as given (condor_cost.fill_plan; the decision goes to fill.jsonl next
    to job.sub). --no-fill submits the shape as given."""
    from tbshared import condor_cost
    cheap = getattr(a, 'cheap_bid', 0)
    ladder = (cheap, a.bid) if cheap and cheap < a.bid else (a.bid,)
    body = condor_cost.fill_plan(body, n, ladder, ROUND_EXCLUDE, fill=getattr(a, 'fill', True), where=str(sub),
                                 log=lambda m: print(m, file=sys.stderr))[0]
    cpus, memory_gib = condor_cost.parse_shape(body)[:2]
    return body, (cpus, memory_gib)


def cheap_split(n, a, cpus=None, memory_gib=None):
    """How many of the `n` jobs go out at --cheap-bid: as many as fit now on nodes whose floor is at most that bid
    (CPU-only and A100 nodes, synth.condor.cheap_capacity); the rest bid --bid (#242: m1-hero's 64-CPU shards at bid 56
    ran on B200 nodes, floor 55). 0 without --cheap-bid or if the check fails."""
    if not getattr(a, 'cheap_bid', 0) or a.cheap_bid >= a.bid:
        return 0
    from synth.condor import cheap_capacity
    try:   # the shape submitted (fill_shards), else the arguments'
        room = cheap_capacity(cpus or a.cpus, memory_gib or a.memory_gib, max_floor=a.cheap_bid, exclude=ROUND_EXCLUDE)
    except Exception as e:  # noqa: BLE001
        print(f'cheap capacity check failed ({type(e).__name__}: {e}); every job at bid {a.bid}', file=sys.stderr)
        return 0
    print(f'{min(n, room)} of {n} jobs at bid {a.cheap_bid} (room for {room} on CPU-only and A100 nodes), the rest at '
          f'{a.bid}', file=sys.stderr)
    return min(n, room)


def rehost_rounds(names):
    """Apply package.rehost to the tasks of these rounds and cohorts (all of both if none are named); each change is
    printed as it is made."""
    from .cohort import COHORTS
    dirs = ([d for n in names for d in (ROUNDS / n, COHORTS / n) if (d / 'tasks').is_dir()] if names else
            sorted(d for root in (ROUNDS, COHORTS) if root.is_dir() for d in root.iterdir() if (d / 'tasks').is_dir()))
    missing = [n for n in names if not any((root / n / 'tasks').is_dir() for root in (ROUNDS, COHORTS))]
    if missing:
        raise SystemExit(f'no tasks/ in rounds or cohorts for: {missing}')
    changed = left = 0
    for d in dirs:
        if (d / 'packed' / 'index.json').exists():   # roundpack: its packed tasks are not rewritten in place
            print(f'{d.parent.name}/{d.name} is packed (roundpack, #245): only its loose tasks are rehosted', flush=True)
        for task in sorted((d / 'tasks').iterdir()):
            try:
                if rehost(task):
                    changed += 1
                    print(f'rehosted {d.parent.name}/{d.name}/{task.name}', flush=True)
            except (PackageError, OSError) as e:
                left += 1
                print(f'left alone: {e}', flush=True)
    print(f'{changed} tasks rehosted, {left} left alone, in {len(dirs)} rounds and cohorts')


def apply_preset(a, parser):
    """--real-assets: set the preset's flags; an explicit flag with another value than the preset's is an error."""
    for k, v in REAL_ASSET_PRESETS[a.real_assets].items():
        cur, default = getattr(a, k), parser.get_default(k)
        if cur != default and str(cur) != str(v):
            raise SystemExit(f'--real-assets {a.real_assets} sets --{k.replace("_", "-")} {v}, but {cur} was passed')
        setattr(a, k, v)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    for op in ('run', 'submit'):
        p = sub.add_parser(op)
        p.add_argument('--round', required=True)
        p.add_argument('--n', type=int, default=100)
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--minutes', type=int, help='difficulty target of every spec (default: drawn from --minutes-mix)')
        p.add_argument('--minutes-mix', default=specs_mod.DEFAULT_MIX,
                       help=f'difficulty-target mix: one of {sorted(specs_mod.MIXES)} (student1 = the mix before #26; use it, '
                            'with --vocab v1, for any comparison with students 1-2) '
                            'or explicit weights, e.g. 15=0.5,30=0.3,60=0.2')
        p.add_argument('--vocab', choices=specs_mod.VOCAB_NAMES, default=specs_mod.DEFAULT_VOCAB,
                       help=f'spec vocabulary (#25), default {specs_mod.DEFAULT_VOCAB} (TB-fitted actions, domains and classes with v2\'s '
                            'sub-domains, 5%% open draws, merged details and languages, TB languages at >= 1%%); v1 = the TB-fitted vocabulary before #25 '
                            '(use it, with --minutes-mix student1, for any comparison with students 1-2); v2, v2d2, v2d3 = the '
                            '#25 arms; v2h = v2 with 5%% open draws (default 11-03 to 11-04)')
        p.add_argument('--gen', default='gen-ds', help='serving name (task_synthesis.serve) or base URL')
        p.add_argument('--judge', default='gen-ds', help='serving name or base URL')
        p.add_argument('--repairs', type=int, default=2)
        p.add_argument('--max-tokens', type=int, help='generator reply cap (default: none; the context bounds it)')
        p.add_argument('--concurrency', type=int, default=0,
                       help='generator requests in flight (default 0: every spec at once, up to MAX_IN_FLIGHT)')
        p.add_argument('--real-assets', choices=sorted(REAL_ASSET_PRESETS),
                       help='author recipe for specs bound to a real asset, as one preset (#118): oneshot (1st A/B), wx '
                            '(4th A/B, iteration 7), off (no bound specs allowed). Sets '
                            '--author/--workflow/--judge-repairs/--reasoning-effort/--judge-effort/--author-tokens/'
                            '--author-commands; passing one of those with a different value is an error')
        p.add_argument('--effort-scope', choices=['asset', 'all'], default='asset',
                       help='--reasoning-effort and --judge-effort apply to specs bound to a real asset only (default; '
                            'the others keep the server\'s, as student 1), or to every spec')
        p.add_argument('--bind-window', type=int, default=300,
                       help='streamed binding: start a spec only while at most this many pending specs before it in the '
                            'queue still wait for their bind result (0: no limit)')
        p.add_argument('--author', choices=['oneshot', 'agentic', 'workspace'], default='oneshot',
                       help='agentic: the generator explores a real asset in a sandbox before writing (pipeline.explore)')
        p.add_argument('--order', choices=['spec', 'asset'], default='spec',
                       help='asset: the generator also chooses the action for the asset (asset-first)')
        p.add_argument('--reasoning-effort', help='author reasoning effort: low|high|max or 1-100 (default: the server\'s, '
                       'max); the round judge keeps the server\'s')
        p.add_argument('--judge-effort', help='round judge reasoning effort: low|high|max or 1-100 (default: the server\'s, '
                       'max); re-judging 300 tasks: high 84%% pass/fail agreement with max (max vs max 87%%) at -34%% tokens, '
                       'low 80%% at -57%%')
        p.add_argument('--workflow', choices=['full', 'lean', 'lean2', 'lean3'], default='full',
                       help='lean: batching prompt, deep-but-rare compaction (prefix-cache friendly), 4k-char tool outputs')
        p.add_argument('--target-inflight', type=int, default=None,
                       help='start a new spec only while the generator pool has fewer than this many requests running + '
                            'waiting per server; <round>/admission.json overrides it live; 0: no gate. Default: 1,200 for '
                            'agentic/workspace authors (DeepSeek-1017 4xB200 sweet spot), no gate for one-shot authors '
                            '(their specs hold no session; production sets one, m1.toml oneshot_target_inflight)')
        p.add_argument('--judge-repairs', type=int, default=0,
                       help='workspace: rounds of fixing a judge-rejected task from the reviewer\'s issues (sandbox kept)')
        p.add_argument('--target-sessions', type=int, default=1000,
                       help='agentic/workspace: at most this many open author sessions per pool server (0: no cap)')
        p.add_argument('--target-tasks', type=int, default=0,
                       help='stop every job of the round once this many tasks are valid and judge-passed, cutting the open '
                            'sessions; give the round more specs than the target needs (~1.4x at a 70-85%% pass rate) so the '
                            'queue never drains first (0: run every spec)')
        p.add_argument('--tail-frac', type=float, default=0.03,
                       help='agentic/workspace: once nothing is left to claim and at most this share of the specs is open '
                            'for --tail-minutes, cut the open sessions (0: wait for every session)')
        p.add_argument('--tail-minutes', type=int, default=15)
        p.add_argument('--check-memory-mb', type=int, default=0,
                       help="memory cap of each check's sandbox (PSS of its processes; the largest is killed and the check "
                            "fails with an out-of-memory error for the generator). 0: the task's own task.toml memory_mb "
                            '(8192 if missing). #171: a task bomb took a 384 GiB round job and its 281 specs down')
        p.add_argument('--build-memory-gb', type=int, default=24,
                       help='memory cap of each image build step (resident memory of its process tree; the step and the '
                            'build fail with an out-of-memory error; 0: no cap)')
        p.add_argument('--sandbox-stack-kb', type=int, default=8192,
                       help="soft stack limit of the job, so of every build step and sandbox (Docker's 8 MiB; condor jobs "
                            'run with an unlimited stack, where a runaway recursion eats the node instead of a SIGSEGV; '
                            '0: unchanged)')
        p.add_argument('--claim-stale-minutes', type=int, default=10,
                       help='a claim whose job has not beaten its heartbeat for this long is taken over by a live job and '
                            'its spec re-run from scratch (#171; the heartbeat is written every 30 s)')
        p.add_argument('--hang-exit-minutes', type=int, default=30,
                       help='a job whose event loop has not run for this long dumps its thread stacks and exits, so a hung '
                            'job frees its node (its claims were taken over after --claim-stale-minutes; 0: never)')
        p.add_argument('--shard', type=int, default=0, help=argparse.SUPPRESS)
        p.add_argument('--shards', type=int, default=1, help=argparse.SUPPRESS)
        p.add_argument('--authors', type=int, default=0,
                       help='specs open at once per job (for agentic/workspace authors: sandboxes); 0: every spec')
        p.add_argument('--author-tokens', type=int, default=explore.TOKEN_BUDGET,
                       help='agentic/workspace: completion tokens per author, then only submit is left')
        p.add_argument('--author-commands', type=int, default=explore.WORKSPACE_MAX_COMMANDS,
                       help='workspace: sandbox commands per author')
        p.add_argument('--build-parallel', type=int, default=48,
                       help='harbor_ns builds at once (each ~1.5 min, 2 CPUs; ~190 over the cluster proxy were fine)')
        p.add_argument('--package-cache', choices=['on', 'off'], default='on',
                       help='builds fetch apt/PyPI packages through a package cache in this job (synth.package_cache)')
        p.add_argument('--package-cache-gib', type=int, default=20, help='package cache store cap (added to the disk request)')
        p.add_argument('--author-package-cache', choices=['apt+pip', 'apt', 'off'], default='apt+pip',
                       help="with --package-cache on: authors' sandboxes fetch apt and PyPI packages (apt: only apt) through it")
        p.add_argument('--resume', action='store_true',
                       help='replay saved replies of specs that failed on infrastructure (finished specs are kept)')
        if op == 'submit':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cheap-bid', type=int, default=0,
                           help='submit as many jobs as fit on nodes with a floor <= this bid at it, the rest at --bid; '
                                'the production driver rebids the ones still idle (#242)')
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=96)
            p.add_argument('--no-fill', dest='fill', action='store_false',
                           help='submit --cpus / --memory-gib as given (default: filled to the binding resource where '
                                'it is free, tbshared condor_cost.fill_plan)')
            p.add_argument('--jobs', type=int, default=1,
                           help='jobs sharing the round\'s spec queue (each claims specs as it has room); size them so the '
                                'pool reaches --target-inflight: ~1,500 workspace authors per 64-CPU job')
            p.add_argument('--disk-gib', type=int, default=200,
                           help='job scratch (sandboxes, builds): 144 workspace authors overflowed 200 GiB (ws7/ws11, 2026-11-01)')
    p = sub.add_parser('report')
    p.add_argument('--round', required=True)
    p = sub.add_parser('rejudge')
    p.add_argument('--round', required=True)
    p.add_argument('--judge', required=True, help='serving name or base URL')
    p.add_argument('--concurrency', type=int, default=32)
    p = sub.add_parser('rehost', help='keep the /etc/hosts setup writes in tasks made before the entrypoint did')
    p.add_argument('--round', action='append', default=[], help='a round or cohort, repeatable (default: all of both)')
    return ap, sub


def parse_args(argv=None):
    ap, sub = build_parser()
    a = ap.parse_args(argv)
    if getattr(a, 'real_assets', None):
        apply_preset(a, sub.choices[a.op])
    if hasattr(a, 'target_inflight') and a.target_inflight is None:
        # one-shot rounds were never load-gated before 2026-11-04: they keep that unless a target is given
        a.target_inflight = 1200 if a.author in ('agentic', 'workspace') else 0
    if getattr(a, 'minutes_mix', None):
        try:
            specs_mod.parse_mix(a.minutes_mix)
        except ValueError as e:
            ap.error(str(e))
    return a


def run_and_exit(a):
    """`round run`, then exit at once after the final flush, even if threads are stuck (m1-hero, 2026-11-05: 2 of 12
    jobs printed "The executor did not finishing joining its threads within 300 seconds" after their last log line and
    then hung in the interpreter's exit, which joins every executor thread; condor held their nodes ~40 min). Stuck
    threads' stacks go to stderr first."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rc = 0
    try:
        loop.run_until_complete(Round(a).run())
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    try:
        stuck = []
        frames = sys._current_frames()
        for t in threading.enumerate():
            if t is threading.main_thread() or t.daemon or t.ident not in frames:
                continue
            stack = traceback.extract_stack(frames[t.ident])
            if any(f.name == '_worker' and 'work_queue.get' in (f.line or '') for f in stack[-3:]):
                continue   # an idle executor thread
            stuck.append(f'thread {t.name}:\n' + ''.join(traceback.format_list(stack[-12:])))
        if stuck:
            print(f'{time.strftime("%H:%M:%S")} exiting with {len(stuck)} busy threads:\n' + '\n'.join(stuck),
                  file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)


def main():
    a = parse_args()
    if a.op == 'run':
        run_and_exit(a)
    elif a.op == 'submit':
        submit(a)
    elif a.op == 'rejudge':
        asyncio.run(rejudge(a))
    elif a.op == 'rehost':
        rehost_rounds(a.round)
    else:
        report(a.round)


if __name__ == '__main__':
    main()

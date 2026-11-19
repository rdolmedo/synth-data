"""Credit rules for CPU jobs of every pipeline (tbshared; from aa-evals #378): a staggered bid ladder, auto-release
on memory holds, and where condor files go. Policy (ladder, steps, exclusions) is each pipeline's, passed as arguments;
the defaults are aa-evals'.

Charges (Confluence "Cluster banking system"): units = max(RequestCpus, RequestMemory_GB / 16) + 24 per GPU, per hour,
on what the job REQUESTS, at the node's running price, which is at least the node floor (MinRunningPrice): CPU-only and
A100 nodes 5-25, H100 35, B200 55. A job only matches nodes whose floor is at most its bid, and job `rank` cannot steer
placement (the negotiator orders machines first). Running jobs are never killed for a higher bid; their price can rise
after the first hour while higher bids wait. So a CPU job should bid as low as gets it a node soon.

Bid ladder (default 26 -> 36 -> 56):
- 26 reaches CPU-only and A100 nodes (floor <= 25), 36 adds H100 nodes and floor-35 CPU nodes, 56 adds B200 nodes.
- At submission, each job starts on the lowest rung whose nodes have room for its shape right now (condor_status;
  partitionable GPU slots keep min(16, 8 x free GPUs) CPUs and min(64000, 32000 x free GPUs) MB for GPU jobs), so a
  full cheap tier costs no waiting. Jobs that fit nowhere start on the top rung.
- A job still idle `steps` seconds after it went idle or was last rebid (default 300 s at 26, 600 s at 36) moves up a
  rung in place (`condor_prio`, what `condor_change_bid` does). The watcher job does this for every job that carries
  `+AaBidLadder`; it is submitted on first use (`ensure_watcher`, 1 CPU / 2 GB) and exits after 2 h with no ladder job
  in the queue. A job's bid never goes down.

Files: condor logs, stdout/stderr and run state go to /data (FAST_ROOT), never to /home: the home quota is
small, and when it fills, condor silently drops event-log records (c30, 11-12) and runs stop.

  python -m tbshared.condor_cost submit JOB.sub      # submit on the ladder (shape read from the file)
  python -m tbshared.condor_cost plan CPUS MEM_GB [N] # where N such jobs would start now
  python -m tbshared.condor_cost escalate [--dry-run] # one watcher pass
  python -m tbshared.condor_cost watch                # the watcher loop (what the watcher job runs)
  python -m tbshared.condor_cost status               # ladder jobs by bid and state

Stdlib only: the watcher runs a frozen copy of this file.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LADDER = (26, 36, 56)
STEPS = (300, 600)            # seconds idle on rung i before rung i + 1 (the last value repeats)
FAST_ROOT = Path('/data/aai/aa_evals')
WATCH_DIR = FAST_ROOT / 'bidladder'
WATCH_BATCH = 'aa-bid-ladder-watch'
# The watcher (1 CPU / 2 GB) bids 36 (CPU-only, A100 and H100 nodes: a 1-CPU job finds room there at once; at 26 it
# could wait behind the very batch it was submitted for, and only another submission's ensure_watcher would move it)
# and is a ladder job itself (36 -> 56): while it waits, every ensure_watcher call runs one escalation pass in its place.
WATCH_SHAPE = (1, 2)
WATCH_LADDER = (36, 56)
WATCH_STEPS = (300,)
WATCH_IDLE_EXIT = 7200
ATTR = 'AaBidLadder'          # "26,36,56"
ATTR_STEPS = 'AaBidLadderSteps'
ATTR_AT = 'AaBidLadderAt'     # epoch of the last rebid
SUBMIT_TIMEOUT = 300
# A job held for memory (HoldReasonCode 34) is released by condor itself, at most 5 times, and restarts: for jobs that
# resume their work on a restart (aa-evals workers, clients and collectors; eval_q35 workers); others pass
# auto_release=False (#378). Most such holds in c30/c30n were unflushed writes to /data counted in the job's memory
# cgroup, not process memory, so a restart (new cgroup) usually clears them.
AUTO_RELEASE_MAX_HOLDS = 5
AUTO_RELEASE = f'periodic_release = (HoldReasonCode == 34) && (NumHolds <= {AUTO_RELEASE_MAX_HOLDS})\n'
# With AUTO_RELEASE, a literal request_memory becomes the serving jobs' memory net (holds.py, 1.5): the base plus
# MEMORY_STEP of it per memory hold so far, so a worker that really needs more does not loop hold -> release -> hold
# at the same size. The base is the job's shape: parse_shape, the ladder's fit check and ledger.read_submit read it.
# Equal to holds.MEMORY_HOLDS / MEMORY_STEP (test_condor_cost); copied because this file must not import tbshared
# (it is also an alias copy that runs alone).
MEMORY_HOLDS = 'ifThenElse(isUndefined(NumHoldsByReason.JobOutOfResources), 0, NumHoldsByReason.JobOutOfResources)'
MEMORY_STEP = 0.25
_MEM_UNIT_MB = {'': 1, 'M': 1, 'K': 1 / 1024, 'G': 1024, 'T': 1024 * 1024}
_MEM_LITERAL = re.compile(r'^([ \t]*request_memory[ \t]*=[ \t]*)(\d+(?:\.\d+)?)[ \t]*([KMGT]?)B?[ \t]*$', re.M | re.I)
# nodes CPU jobs exclude (/data I/O faults, Lustre evictions, slow nodes): the union of aa-evals' (harbor_mswea
# EXCLUDE_NODES) and eval_q35's (submit.BAD_NODES) lists on 2026-11-14; the watcher too. A pipeline may exclude more.
# + gpu103 (2026-11-15): EIO on /data file creates killed two eval jobs within 25 min (m5l Omniscience, SciCode)
# + cpu305 (GPU4 fails cuInit, 11-11 and 11-15; was serve_teacher's own extra list) and cpu201 (bad at the 11-15 re-check):
#   in the live file since 2026-11-19 (coordinator), recorded here so the list has one source
BAD_NODES = ('cpu102', 'cpu201', 'cpu203', 'cpu205', 'cpu207', 'cpu208', 'cpu305', 'gpu103', 'gpu106', 'gpu110', 'gpu123', 'gpu132', 'gpu143',
             'gpu166', 'gpu172')
# Live additions every process sees at its next submission, frozen copies included (node names, one or more per line,
# separated by spaces or commas, '#' comments; $TBSHARED_BAD_NODES_FILE names another file; the directory may need
# creating for the first entry): exclude a node at once here, then add it to BAD_NODES above for the record, and keep
# it here too while processes started from older frozen copies run (they re-read only their own copy's BAD_NODES).
BAD_NODES_FILE = '/data/aai/tbench/shared/bad-nodes.txt'
BAD_NODES_FILE_ENV = 'TBSHARED_BAD_NODES_FILE'
_NODE = re.compile(r'^[A-Za-z]+[0-9]+[A-Za-z0-9-]*$')   # a host's short name (gpu103, cpu404): letters, then digits


def _source_bad_nodes() -> set:
    """BAD_NODES as this module's file says now (its `BAD_NODES = <tuple/list/set of strings>` assignment, evaluated as
    a literal): an edited checkout reaches a process that imported the old list. Anything else (unreadable, not a
    literal collection of names): the imported constant, with a log line."""
    import ast
    try:
        tree = ast.parse(Path(__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'BAD_NODES' for t in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, (tuple, list, set)) and value and all(isinstance(x, str) for x in value):
                    return set(value)
                raise ValueError(f'BAD_NODES is not a collection of node names: {value!r}')
        raise ValueError('no BAD_NODES assignment')
    except (OSError, ValueError, SyntaxError, TypeError) as e:
        print(f'condor_cost.bad_nodes: {__file__}: {e}; using the imported list', file=sys.stderr, flush=True)
        return set(BAD_NODES)


def _live_bad_nodes() -> set:
    """The live file's nodes (missing: none; a line that is not a node name is skipped with a log line)."""
    path = Path(os.environ.get(BAD_NODES_FILE_ENV) or BAD_NODES_FILE)
    try:
        lines = path.read_bytes().decode('utf-8', errors='replace').splitlines()   # one bad byte never stops a submit
    except FileNotFoundError:
        return set()
    except OSError as e:
        print(f'condor_cost.bad_nodes: {path}: {e}; live additions skipped', file=sys.stderr, flush=True)
        return set()
    out = set()
    for line in lines:
        for name in re.split(r'[\s,]+', line.split('#', 1)[0].strip()):
            if not name:
                continue
            short = name.split('.', 1)[0]   # gpu103.internal.cluster...: UtsnameNodename is the short name
            if _NODE.match(short) and all(_NODE.match(x) or x.isalnum() for x in name.split('.')[1:]):
                out.add(short)
            else:   # also reason text without its '#' (it would otherwise exclude 'EIO', 'on', ...)
                print(f'condor_cost.bad_nodes: {path}: not a node name: {name!r}', file=sys.stderr, flush=True)
    return out


def bad_nodes() -> tuple:
    """The nodes every CPU and GPU submission excludes, read when it is called, i.e. at submit time (#531 A9: a
    long-lived grade watcher kept the BAD_NODES it imported and put a collector on gpu103): BAD_NODES as this file says
    now, plus the live file. Sorted. Never read BAD_NODES itself at submission (it is the list at import)."""
    return tuple(sorted(_source_bad_nodes() | _live_bad_nodes()))


class SubmitError(RuntimeError):
    """A submission failed. `submitted` lists the clusters already queued before the failure (record them)."""

    def __init__(self, msg: str, submitted: list | None = None):
        super().__init__(msg)
        self.submitted = submitted or []


# ------------------------------------------------------------------------------------------------ placement
def units(cpus: float, mem_gib: float, gpus: float = 0) -> float:
    """Charged units per hour of a request as given: max(CPUs, RAM GB / 16) + 24 per GPU (RAM up to 16 GB per CPU is
    free). A shape filled by fill_shape costs the same, up to whole-CPU rounding."""
    return max(float(cpus), float(mem_gib) / 16) + 24 * float(gpus)


# Fill to the binding resource (user, 2026-11-14: "since mem is free for enough cpus, make sure to size the other one
# accordingly, since we will pay the same. according to the one that binds"). Units are max(CPUs, GB / 16), so a job
# asks for both at the binding level, GB = max(GB, 16 x CPUs) and CPUs = max(CPUs, ceil(GB / 16)), within what the
# nodes have (no node class here has 16 GB per CPU: B200 and A100 7.9, H100 10.5, the fat CPU nodes 15.7):
# - GPU jobs: CPUs and GB at most the per-GPU share x GPUs of every node class the job may land on (the minimum over
#   every CUDACapability its requirements name, else over every GPU node), so a filled job never strands GPUs. No
#   class info (condor_status down, an unknown class): not filled.
# - CPU jobs: at most the largest node's CPUs and memory (the node with the most memory); on the ladder only where it is free (fill_plan): a filled
#   batch that would start any job on a higher rung is capped at the cheap tier's free GB per free CPU, else unfilled.
# One place fills: the job text (fill_plan, in submit and submit_file), and the bid plan reads that text's shape.
GB_PER_CPU = 16


def node_rows(status: str | None = None, exclude=(), gpu_free_nodes_only: bool = False) -> list[dict]:
    """The partitionable slots in condor_status_slots() with their totals: [{'node', 'floor', 'cpus', 'mem_mb',
    'gpus' (free), 'total_cpus', 'total_mem_mb', 'total_gpus', 'cuda'}]; rows without totals are left out."""
    status = condor_status_slots() if status is None else status
    out, skip = [], set(exclude)
    for line in status.splitlines():
        p = line.split('\t')
        if len(p) < 9 or p[0] in skip:
            continue
        try:
            r = {'node': p[0], 'floor': float(p[1]), 'cpus': int(float(p[2])), 'mem_mb': int(float(p[3])),
                 'gpus': int(_num(p[4], 0) or 0), 'total_gpus': int(_num(p[5], 0) or 0),
                 'total_cpus': float(p[6]), 'total_mem_mb': float(p[7]), 'cuda': _num(p[8], None)}
        except ValueError:
            continue
        if gpu_free_nodes_only and r['total_gpus']:
            continue
        out.append(r)
    return out


def _capabilities(req: str) -> list[float] | None:
    """The CUDACapability values a requirements expression names with == ([] = none named; None = it compares
    CUDACapability some other way, so its class is unknown)."""
    named = [float(x) for x in re.findall(r'CUDACapability\s*==\s*([\d.]+)', req)]
    other = re.findall(r'CUDACapability\s*(?:>=|<=|!=|=!=|=\?=|>|<)', req)
    return None if other else named


def shape_caps(gpus: float = 0, node=None, status: str | None = None, exclude=(),
               gpu_free_nodes_only: bool = False) -> tuple[int, float] | None:
    """(most CPUs, most GB) a job can be filled to, None when unknown. GPU jobs: min over the node classes of
    TotalCpus / TotalGPUs x gpus and TotalMemory / TotalGPUs x gpus; `node` is the target class as (TotalCpus,
    TotalMemory GB, TotalGPUs), a CUDACapability, or a list of them (the requirements' OR), None = every GPU node.
    CPU jobs: (the most CPUs a CPU job can get on any node, ((CPUs, GB) a CPU job can get per node, ...)): fill_shape
    caps GB at the most memory among nodes with at least the filled CPU count. A node with GPUs keeps
    min(16, 8 x GPUs) CPUs and min(64000, 32000 x GPUs) MB for GPU jobs (its Start policy, as in fits)."""
    if gpus and isinstance(node, tuple) and len(node) == 3:
        tc, tm, tg = map(float, node)
        return (int(tc / tg * gpus + 1e-9), int(tm / tg * gpus)) if tg >= gpus else None
    try:
        rows = node_rows(status, exclude, gpu_free_nodes_only)
    except (OSError, subprocess.SubprocessError):
        return None
    if not gpus:
        per = sorted({(int(r['total_cpus'] - (min(16, 8 * r['total_gpus']) if r['total_gpus'] else 0)),
                       int((r['total_mem_mb'] - (min(64000, 32000 * r['total_gpus']) if r['total_gpus'] else 0))
                           / 1024)) for r in rows})
        return (max(c for c, _ in per), tuple(per)) if per else None
    want = None if node is None else {float(x) for x in (node if isinstance(node, (list, tuple)) else [node])}
    cls = {(r['cuda'], r['total_cpus'], r['total_mem_mb'], r['total_gpus']) for r in rows
           if r['total_gpus'] >= gpus and (want is None or r['cuda'] in want)}
    if not cls or (want is not None and not want <= {c[0] for c in cls}):
        return None                                         # a named class with no node here: unknown
    return (min(int(tc / tg * gpus + 1e-9) for _, tc, _, tg in cls),
            min(int(tm / tg * gpus / 1024) for _, _, tm, tg in cls))


def mem_cap_for(caps, cpus: int):
    """The GB cap of shape_caps' result for a job of `cpus` CPUs: GPU jobs' class share, or for CPU jobs the most
    memory among nodes that give a CPU job at least `cpus` CPUs (None: no such node)."""
    if isinstance(caps[1], tuple):
        return max((m for c, m in caps[1] if c >= cpus), default=None)
    return caps[1]


def fill_shape(cpus: float, mem_gb: float, gpus: float = 0, node=None, status: str | None = None,
               log=None, caps=False) -> tuple[int, float]:
    """(CPUs, GB) at the binding resource within shape_caps (never below the request); unchanged when the caps are
    unknown. Capping is logged. `caps` overrides the lookup ((CPUs, GB) or None)."""
    cpus, mem_gb = int(math.ceil(float(cpus) - 1e-9)), float(mem_gb)
    lim = shape_caps(gpus, node, status) if caps is False else caps
    if lim is None:
        if log:
            log(f'fill_shape: {cpus}c/{mem_gb:g}G{f" {gpus:g} GPU" if gpus else ""} not filled: node class unknown')
        return cpus, (int(mem_gb) if mem_gb == int(mem_gb) else mem_gb)
    want_c = max(cpus, int(math.ceil(mem_gb / GB_PER_CPU - 1e-9)))
    new_c = max(cpus, min(want_c, lim[0]))
    want_m = max(mem_gb, float(GB_PER_CPU * new_c))
    mem_cap = mem_cap_for(lim, new_c)
    new_m = max(mem_gb, min(want_m, float(mem_gb if mem_cap is None else mem_cap)))
    if log and (new_c < want_c or new_m < want_m):
        log(f'fill_shape: {cpus}c/{mem_gb:g}G{f" {gpus:g} GPU" if gpus else ""} wants {want_c}c/{want_m:g}G; capped '
            f'at {new_c}c/{new_m:g}G (nodes: at most {lim[0]} CPUs / {mem_cap} GB per job)')
    return new_c, (int(new_m) if new_m == int(new_m) else new_m)


def fits(free_cpus: int, free_mib: int, free_gpus: int, cpus: int, mib: int) -> int:
    """How many CPU-only jobs of (cpus, mib) one partitionable slot takes now. A slot with free GPUs keeps
    min(16, 8 x free GPUs) CPUs and min(64000, 32000 x free GPUs) MB for GPU jobs (its Start policy)."""
    if cpus <= 0 or mib <= 0:
        return 0
    n = 0
    while True:
        lc, lm = free_cpus - cpus, free_mib - mib
        if lc < 0 or lm < 0 or (free_gpus > 0 and (lc < min(16, 8 * free_gpus) or lm < min(64000, 32000 * free_gpus))):
            return n
        n, free_cpus, free_mib = n + 1, lc, lm


STATUS_TTL = 300        # s: one condor_status per submit burst, not 120 s of blocking per call
STATUS_FAIL_TTL = 60    # s: a failed condor_status is not retried within this
_STATUS_CACHE: dict = {}


def condor_status_slots(max_age: float = STATUS_TTL) -> str:
    """The partitionable slots, tab-separated: UtsnameNodename MinRunningPrice Cpus Memory Gpus TotalGpus TotalCpus
    TotalMemory CUDACapability (free resources first, then the node's totals). Cached for max_age s."""
    now = time.time()
    if _STATUS_CACHE.get('out') is not None and now - _STATUS_CACHE['t'] < max_age:
        return _STATUS_CACHE['out']
    if _STATUS_CACHE.get('err') is not None and now - _STATUS_CACHE['err_t'] < STATUS_FAIL_TTL:
        raise _STATUS_CACHE['err']          # a recent failure: do not wait another 120 s
    try:
        out = subprocess.run(['condor_status', '-const', 'PartitionableSlot =?= true', '-af:t', 'UtsnameNodename',
                              'MinRunningPrice', 'Cpus', 'Memory', 'Gpus', 'TotalGpus', 'TotalCpus', 'TotalMemory',
                              'CUDACapability'], capture_output=True, text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        _STATUS_CACHE.update(err=e, err_t=now)
        raise
    _STATUS_CACHE.update(t=now, out=out, err=None)
    return out


def room_by_floor(cpus: int, mem_gib: float, exclude=(), status: str | None = None,
                  gpu_free_nodes_only: bool = False) -> dict[float, int]:
    """floor -> how many jobs of this shape fit now on nodes with that floor (gpu_free_nodes_only: nodes without
    GPUs, for jobs that require them)."""
    status = condor_status_slots() if status is None else status
    room: dict[float, int] = {}
    skip = set(exclude)
    for line in status.splitlines():
        p = line.split('\t')
        if len(p) < 5 or p[0] in skip:
            continue
        if gpu_free_nodes_only and len(p) > 5 and p[5] not in ('0', 'undefined', ''):
            continue
        try:
            floor, fc, fm = float(p[1]), int(float(p[2])), int(float(p[3]))
            fg = int(float(p[4])) if p[4] not in ('undefined', '') else 0
        except ValueError:
            continue
        room[floor] = room.get(floor, 0) + fits(fc, fm, fg, int(cpus), int(mem_gib * 1024))
    return room


def plan_bids(n: int, cpus: int, mem_gib: float, ladder=LADDER, exclude=(), status: str | None = None,
              gpu_free_nodes_only: bool = False) -> list[tuple[int, int]]:
    """[(bid, count)] for n jobs of exactly this shape (the job text's: fill_plan fills before it plans): each rung
    takes what fits on the nodes it newly reaches (floor <= bid and above the previous rung), cheapest first; the rest
    start on the top rung. On a condor_status failure everything starts on the bottom rung and the watcher moves it
    up."""
    try:
        room = room_by_floor(cpus, mem_gib, exclude, status, gpu_free_nodes_only)
    except (OSError, subprocess.SubprocessError):
        return [(ladder[0], n)] if n else []
    out, left, prev = [], n, float('-inf')
    for i, bid in enumerate(ladder):
        tier = sum(k for f, k in room.items() if prev < f <= bid)
        take = left if i == len(ladder) - 1 else min(left, tier)
        if take:
            out.append((bid, take))
        left -= take
        prev = bid
    return out


# ------------------------------------------------------------------------------------------------ submission
def parse_shape(sub_text: str) -> tuple[int, float, float]:
    """(cpus, mem_gib, gpus) from a submit file's request_* lines: memory is the value's leading size (MB unless it has
    a K/M/G/T unit), as before 1.5, so the memory net's `BASE + STEP * <memory holds>` reads as its base."""
    def get(key, default):
        m = re.search(rf'^\s*{key}\s*=\s*(\S+)', sub_text, re.M | re.I)
        return m.group(1) if m else default
    cpus = int(float(get('request_cpus', '1')))
    line = re.search(r'^\s*request_memory\s*=\s*(.*?)\s*$', sub_text, re.M | re.I)
    if line:
        m = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?)B?\b', line.group(1), re.I)
        if not m:
            raise ValueError(f'request_memory = {line.group(1)!r} does not start with a size (MB, or K/M/G/T)')
        mem_gib = float(m.group(1)) * _MEM_UNIT_MB[m.group(2).upper()] / 1024
    else:
        mem_gib = 1.0
    gpus = float(get('request_gpus', '0'))
    return cpus, mem_gib, gpus


_CPU_LITERAL = re.compile(r'^([ \t]*request_cpus[ \t]*=[ \t]*)(\d+)[ \t]*$', re.M | re.I)
_MEM_NET = re.compile(r'^([ \t]*request_memory[ \t]*=[ \t]*)(\d+)[ \t]*\+[ \t]*(\d+)[ \t]*\*[ \t]*(\S.*?)[ \t]*$', re.M | re.I)
FILL_RECORD = 'fill.jsonl'     # one line per batch, next to its submit file (fill-report reads them)
_PIPELINE: dict = {}


def _lines(body: str, key: str) -> list[str]:
    return re.findall(rf'^[ \t]*{key}[ \t]*=.*$', body, re.M | re.I)


def fillable(body: str) -> bool:
    """Whether request_cpus / request_memory / request_gpus are plain numbers (or the memory net) that fill can
    rewrite."""
    gl = _lines(body, 'request_gpus')
    return not ((_lines(body, 'request_cpus') and not _CPU_LITERAL.search(body)) or
                (_lines(body, 'request_memory') and not (_MEM_LITERAL.search(body) or _MEM_NET.search(body))) or
                (gl and not re.fullmatch(r'[^=]*=[ \t]*\d+(?:\.\d+)?[ \t]*', gl[-1])))


def set_shape(body: str, cpus: int, mem_gb: float) -> str:
    """The body with request_cpus / request_memory set (a memory net keeps its step's share of the base)."""
    c0, m0, _ = parse_shape(body)
    mb = int(round(mem_gb * 1024))
    if _lines(body, 'request_cpus'):
        body = _CPU_LITERAL.sub(lambda m: f'{m.group(1)}{cpus}', body)
    elif cpus != c0:
        body += f'request_cpus = {cpus}\n'
    if _MEM_NET.search(body):
        body = _MEM_NET.sub(lambda m: f'{m.group(1)}{mb} + {int(mb * int(m.group(3)) / int(m.group(2)))} * '
                                      f'{m.group(4)}', body)
    elif _MEM_LITERAL.search(body):
        body = _MEM_LITERAL.sub(lambda m: f'{m.group(1)}{mb}', body)
    elif mem_gb != m0:
        body += f'request_memory = {mb}\n'
    return body


PIPELINE_FILE = 'pipeline.json'


def find_pipeline_json(start) -> Path | None:
    """The pipeline config governing `start` (a file or directory): the nearest pipeline.json in a directory at or
    above it, checking `<dir>/pipeline.json` then `<dir>/tbshared/pipeline.json` at each level. On the trunk (#480)
    each pipeline package has its own (aa_evals/, eval_q35/, synth/, task_synthesis/, task_synthesis/serve/); a
    vendored pipeline branch has one, tbshared/pipeline.json, at the top. Stops at the repository root (.git)."""
    p = Path(start).resolve()
    d = p if p.is_dir() else p.parent
    for d in (d, *d.parents):
        for f in (d / PIPELINE_FILE, d / 'tbshared' / PIPELINE_FILE):
            if f.is_file():
                return f
        if (d / '.git').exists():
            return None
    return None


def _library(path: str) -> bool:
    """A file of the interpreter or an installed package (stdlib, site-packages): never a pipeline's caller."""
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    return '/site-packages/' in path or '/dist-packages/' in path or any(path.startswith(r + os.sep) for r in roots)


def caller_file() -> str | None:
    """The source file of the nearest caller outside tbshared/ (and outside this module, wherever its copy lives:
    an alias such as aa_evals/common/condor_cost.py), skipping interpreter and installed-package frames; None from the
    command line (`python -m tbshared.condor_cost ...`) or a thread with no pipeline frame on its stack."""
    here = os.path.realpath(__file__)
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if fn and not fn.startswith('<'):
            rp = os.path.realpath(fn)
            if rp != here and os.path.basename(os.path.dirname(rp)) != 'tbshared' and not _library(rp):
                return rp
        f = f.f_back
    return None


def pipeline_name(start=None) -> str | None:
    """`pipeline` of the pipeline.json governing `start` (default: the calling module's file, caller_file), else of
    the one governing this module's own copy (a vendored branch's tbshared/pipeline.json: so the command line and
    callers outside any package keep the branch's name; on the trunk there is none), else None. Read directly,
    because this module must not import tbshared (it is also an alias copy)."""
    start = start if start is not None else caller_file()
    f = find_pipeline_json(start) if start is not None else None
    if f is None:
        f = find_pipeline_json(__file__)
    if f is None:
        return None
    if str(f) not in _PIPELINE:
        name = None
        try:
            name = json.loads(f.read_text()).get('pipeline')
        except (OSError, ValueError, AttributeError):
            pass
        _PIPELINE[str(f)] = name
    return _PIPELINE[str(f)]


def cheap_gb_per_cpu(bid: float, exclude=(), status: str | None = None, gpu_free_nodes_only: bool = False):
    """Free GB per free CPU on the nodes a bid reaches (floor <= bid), None when unknown."""
    try:
        status = condor_status_slots() if status is None else status
    except (OSError, subprocess.SubprocessError):
        return None
    c = m = 0
    for line in status.splitlines():
        p = line.split('\t')
        if len(p) < 5 or p[0] in set(exclude):
            continue
        if gpu_free_nodes_only and len(p) > 5 and p[5] not in ('0', 'undefined', ''):
            continue
        try:
            if float(p[1]) <= bid:
                c, m = c + int(float(p[2])), m + int(float(p[3]))
        except ValueError:
            continue
    return m / 1024 / c if c else None


def _worse(plan: list, base: list, ladder, nowhere: int = 0, base_nowhere: int = 0) -> bool:
    """Whether `plan` starts any job on a higher rung than `base` (fewer jobs at or below some rung), or leaves more
    jobs fitting on no node now (a job that fits nowhere now but fits unfilled counts as worse)."""
    cum = lambda pl, b: sum(k for bb, k in pl if bb <= b)   # noqa: E731
    return nowhere > base_nowhere or any(cum(plan, b) < cum(base, b) for b in ladder)


def _nowhere(n: int, cpus: int, mem_gib: float, ladder, exclude, status, gpu_free_nodes_only) -> int:
    """How many of n jobs fit on no node a ladder's top bid reaches, now (0 when condor_status is unavailable)."""
    try:
        room = room_by_floor(cpus, mem_gib, exclude, status, gpu_free_nodes_only)
    except (OSError, subprocess.SubprocessError):
        return 0
    return max(0, n - sum(k for f, k in room.items() if f <= ladder[-1]))


def _record(where: str, rec: dict, log) -> None:
    try:
        path = Path(where).parent / FILL_RECORD
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
            f.flush()
            os.fsync(f.fileno())
    except (OSError, TypeError, ValueError) as e:   # the decision is logged anyway; never fail a submit for it
        log(f'{where}: fill record not written ({e!r})')


def fill_plan(body: str, n: int = 1, ladder=None, exclude=(), status: str | None = None,
              gpu_free_nodes_only: bool = False, fill: bool = True, where: str = 'submit', record: bool = True,
              log=print, pipeline: str | None = None) -> tuple[str, list | None, dict]:
    """Fill a submit body to the binding resource where it is free, and plan its bids from the result: (body, plan,
    decision). The one place the shape is filled; the plan reads the returned body's shape.
    - decision 'filled': fill_shape within the node caps (shape_caps); 'capped': with RAM capped (by the nodes, or
      for a CPU batch whose filled plan is worse, at the bottom rung's free GB per free CPU); 'unfilled': caps
      unknown, a request that is not a plain number (a macro: the body is returned as is), or filling, even
      capped, is worse; 'unchanged': already at the binding level; 'off': fill=False.
    - worse: the filled plan starts any job on a higher rung, or leaves more jobs that fit on no node now.
    - ladder: the CPU bid ladder (plan returned); a fixed bid passes (bid,), so the fill is kept only if the room at
      floors <= bid stays at least min(n, the unfilled room). None: no plan and no placement check (GPU jobs, which
      bid by GPU type, are never placement-checked here).
    - record: append the decision to fill.jsonl next to `where` (time, pipeline, batch, shapes, decision, reason,
      plans, jobs that fit nowhere). pipeline: the record's name (default: pipeline_name() of the calling module)."""
    batch = re.search(r'^\s*(?:\+JobBatchName|batch_name)\s*=\s*"?([^"\n]+?)"?\s*$', body, re.M | re.I)
    req = ' '.join(re.findall(r'^\s*requirements\s*=\s*(.*)$', body, re.M | re.I))
    shape = None
    if fillable(body):
        try:
            shape = parse_shape(body)
        except ValueError:
            shape = None
    c, m, g = shape or (None, None, None)
    rec = {'t': round(time.time(), 1), 'pipeline': pipeline if pipeline is not None else pipeline_name(),
           'batch': batch.group(1) if batch else None,
           'submit_file': str(where), 'n': n, 'gpus': g, 'before': {'cpus': c, 'mem_gb': m}}
    plan = lambda cc, mm: plan_bids(n, cc, mm, ladder, exclude, status, gpu_free_nodes_only)  # noqa: E731
    nowhere = lambda cc, mm: _nowhere(n, cc, mm, ladder, exclude, status, gpu_free_nodes_only)  # noqa: E731
    new, decision, reason, plans, lost = (c, m), 'unfilled', '', {}, {}
    if shape is None:
        reason = 'request_cpus / request_memory / request_gpus is not a plain number'
    elif not fill:
        decision, reason = 'off', 'fill=False'
    else:
        if g:
            names = _capabilities(req)
            caps = None if names is None else shape_caps(g, names or None, status, exclude)
        else:
            caps = shape_caps(0, None, status, exclude, gpu_free_nodes_only)
        if caps is None:
            reason = 'node class unknown (condor_status unavailable, or a class with no node here)'
        else:
            full = fill_shape(c, m, g, caps=caps)
            want_c = max(c, int(math.ceil(m / GB_PER_CPU - 1e-9)))
            want = (want_c, max(m, float(GB_PER_CPU * want_c)))
            if full == (c, m) and want == (c, m):
                decision, reason = 'unchanged', 'already at the binding level'
            elif full == (c, m):
                reason = f'node caps ({caps[0]} CPUs / {mem_cap_for(caps, c)} GB per job) leave no room to fill'
            else:
                new = full
                decision = 'capped' if full != want else 'filled'
                reason = (f'node caps: at most {caps[0]} CPUs / {mem_cap_for(caps, full[0])} GB per job'
                          if full != want else 'binding resource')
                if ladder and not g:
                    plans = {'unfilled': plan(c, m), 'filled': plan(*full)}
                    lost = {'unfilled': nowhere(c, m), 'filled': nowhere(*full)}
                    if _worse(plans['filled'], plans['unfilled'], ladder, lost['filled'], lost['unfilled']):
                        r = cheap_gb_per_cpu(ladder[0], exclude, status, gpu_free_nodes_only)
                        capped = (full[0], max(m, min(full[1], float(int((r or 0) * full[0])))))
                        if capped not in (full, (c, m)):
                            plans['capped'], lost['capped'] = plan(*capped), nowhere(*capped)
                        if 'capped' in plans and not _worse(plans['capped'], plans['unfilled'], ladder,
                                                            lost['capped'], lost['unfilled']):
                            new, decision = capped, 'capped'
                            reason = (f'the filled plan is worse: RAM at the bid-{ladder[0]} tier\'s {r:.1f} free GB '
                                      'per free CPU')
                        else:
                            new, decision = (c, m), 'unfilled'
                            reason = ('filling (even at the cheap tier\'s GB per CPU) starts jobs on a higher rung '
                                      'or leaves jobs that fit nowhere')
    if shape is not None and new != (c, m):
        body = set_shape(body, new[0], new[1])
    used = plan(*new) if ladder and shape is not None else None
    plans['used'] = used
    rec.update(after={'cpus': new[0], 'mem_gb': new[1]}, decision=decision, reason=reason, plans=plans,
               nowhere=lost)
    if shape is None:
        log(f'{where}: fill {decision} ({reason})')
    else:
        log(f'{where}: fill {decision}: {c}c/{m:g}G -> {new[0]}c/{new[1]:g}G ({reason})'
            + (f'; plans unfilled {plans.get("unfilled")} filled {plans.get("filled")}, fit nowhere {lost}'
               if 'filled' in plans else ''))
    if record:
        _record(where, rec, log)
    return body, used, rec


def memory_net(body: str) -> str:
    """The body with each literal request_memory as BASE_MB + MEMORY_STEP x BASE_MB per memory hold so far."""
    def grow(m):
        base = int(float(m.group(2)) * _MEM_UNIT_MB[m.group(3).upper()])
        return f'{m.group(1)}{base} + {int(base * MEMORY_STEP)} * {MEMORY_HOLDS}'
    return _MEM_LITERAL.sub(grow, body)


def requirement_hints(sub_text: str) -> tuple[set, bool]:
    """(nodes the submit file's requirements exclude, whether it requires nodes without GPUs)."""
    req = ' '.join(re.findall(r'^\s*requirements\s*=\s*(.*)$', sub_text, re.M | re.I))
    nodes = set(re.findall(r'UtsnameNodename\s*!=\s*"([^"]+)"', req))
    return nodes, bool(re.search(r'GPUs\s*==\s*0', req, re.I))


def with_auto_release(body: str, auto_release: bool = True) -> str:
    """The submit body plus AUTO_RELEASE and the memory net on a literal request_memory (memory_net), unless it
    already sets periodic_release (a job with its own net, e.g. holds.submit_lines: left as written) or auto_release
    is False (jobs that would redo their work from scratch, or duplicate appended output, on a restart)."""
    if not auto_release or re.search(r'^\s*periodic_release\s*=', body, re.M | re.I):
        return body
    return memory_net(body) + AUTO_RELEASE


def ladder_lines(ladder=LADDER, steps=STEPS) -> str:
    return f'+{ATTR} = "{",".join(map(str, ladder))}"\n+{ATTR_STEPS} = "{",".join(map(str, steps))}"\n'


def split_queue(sub_text: str) -> tuple[str, int]:
    """(submit text without its final `queue [N]` line, N)."""
    lines = sub_text.rstrip('\n').split('\n')
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r'^\s*queue\s*(\d*)\s*$', lines[i], re.I)
        if m:
            return '\n'.join(lines[:i]) + '\n', int(m.group(1) or 1)
        if lines[i].strip() and not lines[i].lstrip().startswith('#'):
            break
    raise ValueError('submit file has no plain `queue [N]` last line')


def submit(sub_path, bid: int | None = None, runner=subprocess.run, auto_release: bool = True, fill: bool = True,
           **kw) -> list[dict]:
    """The one entry point for CPU jobs: `bid` None = the ladder (submit_file), else a fixed bid (an explicit
    `--bid`). Returns [{'cluster', 'bid', 'n', 'submit_file'}]. fill: the shape at the binding resource (fill_plan)."""
    if bid is None:
        return submit_file(sub_path, runner=runner, auto_release=auto_release, fill=fill, **kw)
    sub_path = Path(sub_path)
    try:
        body, n = split_queue(sub_path.read_text())
    except ValueError:      # `queue ... from ...` and the like: submitted as written
        body = None
    if body is not None:    # a fixed bid never moves up: fill only if the room at floors <= bid holds (fill_plan)
        body = fill_plan(body, n, (int(bid),), kw.get('exclude', ()), kw.get('status'), fill=fill,
                         where=str(sub_path))[0]
        sub_path.write_text(with_auto_release(body, auto_release) + f'queue {n}\n')
    try:
        r = runner(['condor_submit_bid', str(int(bid)), str(sub_path)], capture_output=True, text=True,
                   timeout=SUBMIT_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise SubmitError(f'condor_submit_bid {bid} {sub_path} timed out') from e
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'(\d+) job\(s\) submitted to cluster (\d+)', out)
    if r.returncode or not m:
        raise SubmitError(f'condor_submit_bid {bid} {sub_path} failed: {out.strip()}')
    return [{'cluster': m.group(2), 'bid': int(bid), 'n': int(m.group(1)), 'submit_file': str(sub_path)}]


def submit_file(sub_path, ladder=LADDER, steps=STEPS, exclude=(), status: str | None = None,
                runner=subprocess.run, watcher: bool = True, auto_release: bool = True,
                fill: bool = True) -> list[dict]:
    """Submit a CPU-only submit file on the bid ladder: [{'cluster', 'bid', 'n', 'submit_file'}], one per starting
    rung. The file's final `queue N` is split across rungs (sibling files `<stem>.bid<B>.sub`). GPU jobs are refused.
    fill (default): the shape at the binding resource where it is free (fill_plan), whose plan is used."""
    sub_path = Path(sub_path)
    text = sub_path.read_text()
    if parse_shape(text)[2]:
        raise ValueError(f'{sub_path}: GPU jobs bid by GPU type, not on the CPU ladder')
    body, n = split_queue(text)
    req_nodes, gpu_free = requirement_hints(text)
    body, plan, _ = fill_plan(body, n, ladder, set(exclude) | req_nodes, status, gpu_free, fill=fill,
                              where=str(sub_path))
    if plan is None:   # a shape fill_plan cannot rewrite (e.g. request_cpus = 4.0, request_memory = 4096 * 2): as 1.4
        cpus, mem_gb, _ = parse_shape(text)
        plan = plan_bids(n, cpus, mem_gb, ladder, set(exclude) | req_nodes, status, gpu_free)
    body = with_auto_release(body, auto_release) + ladder_lines(ladder, steps)
    clusters = []
    for bid, k in plan:
        path = sub_path if len(plan) == 1 else sub_path.with_name(f'{sub_path.stem}.bid{bid}.sub')
        path.write_text(body + f'queue {k}\n')
        try:
            r = runner(['condor_submit_bid', str(bid), str(path)], capture_output=True, text=True,
                       timeout=SUBMIT_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise SubmitError(f'condor_submit_bid {bid} {path} timed out', clusters) from e
        out = (r.stdout or '') + (r.stderr or '')
        m = re.search(r'submitted to cluster (\d+)', out)
        if r.returncode or not m:
            raise SubmitError(f'condor_submit_bid {bid} {path} failed: {out.strip()}', clusters)
        clusters.append({'cluster': m.group(1), 'bid': bid, 'n': k, 'submit_file': str(path)})
        print(f'{path}: {k} job(s) at bid {bid} (cluster {m.group(1)}); `+{ATTR}` moves it up while idle', flush=True)
    if watcher:
        try:
            ensure_watcher(runner=runner)
        except Exception as e:   # the jobs are queued; without a watcher they stay on their first rung
            print(f'WARNING: bid-ladder watcher not ensured ({e!r}); run `python -m tbshared.condor_cost watch`',
                  flush=True)
    return clusters


# ------------------------------------------------------------------------------------------------ escalation
def next_bid(bid: float, ladder, steps, idle_since: float, last_change: float | None, now: float) -> int | None:
    """The rung an idle job should move to now, or None. Rung i = the highest rung <= bid; it waits steps[i] (the last
    value repeats) after max(idle since, last rebid). A bid below the bottom rung moves to it at once."""
    ladder = [int(x) for x in ladder]
    below = [i for i, b in enumerate(ladder) if b <= bid]
    if not below:
        return ladder[0]
    i = below[-1]
    if i + 1 >= len(ladder):
        return None
    wait = steps[min(i, len(steps) - 1)] if steps else STEPS[0]
    t0 = max(idle_since or 0, last_change or 0)
    return ladder[i + 1] if now - t0 >= wait else None


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def ladder_jobs(owner: str, runner=subprocess.run, idle_only: bool = True, watchers: bool = True) -> list[dict]:
    """The owner's jobs on the ladder (idle_only: waiting ones); watchers=False leaves out the watcher jobs, which are
    ladder jobs themselves."""
    const = f'Owner == "{owner}" && {ATTR} =!= undefined' + (' && JobStatus == 1' if idle_only else '')
    if not watchers:
        const += f' && JobBatchName =!= "{WATCH_BATCH}"'
    r = runner(['condor_q', owner, '-constraint', const, '-json', '-attributes',
                f'ClusterId,ProcId,JobStatus,JobPrio,RequestGpus,EnteredCurrentStatus,QDate,{ATTR},{ATTR_STEPS},{ATTR_AT}'],
               capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(f'condor_q failed: {(r.stderr or "").strip()[:300]}')
    return json.loads(r.stdout or '[]') or []


def escalate(now: float | None = None, dry_run: bool = False, owner: str | None = None, runner=subprocess.run,
             log=print) -> list[tuple[str, int, int]]:
    """One pass: move every idle ladder job of `owner` that waited long enough up one rung. Returns (job, old, new)."""
    now = time.time() if now is None else now
    owner = owner or getpass.getuser()
    moves = []
    for j in ladder_jobs(owner, runner):
        if _num(j.get('RequestGpus'), 0):
            continue                  # never a GPU job
        job = f"{j['ClusterId']}.{j['ProcId']}"
        bid = _num(j.get('JobPrio'), -1000) + 1000
        ladder = [int(x) for x in str(j.get(ATTR, '')).split(',') if x.strip().isdigit()] or list(LADDER)
        steps = [int(x) for x in str(j.get(ATTR_STEPS, '')).split(',') if x.strip().isdigit()] or list(STEPS)
        new = next_bid(bid, ladder, steps, _num(j.get('EnteredCurrentStatus'), now), _num(j.get(ATTR_AT)), now)
        if new is None or new <= bid:
            continue
        log(f'{time.strftime("%FT%TZ", time.gmtime(now))} {job}: idle at bid {int(bid)} -> {new}')
        if not dry_run:
            # the rebid time first: without it the next pass would measure the wait from EnteredCurrentStatus and move
            # the job up again at once; if the rebid itself then fails, the job only waits one more step
            try:
                r = runner(['condor_qedit', job, ATTR_AT, str(int(now))], capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                log(f'  condor_qedit {ATTR_AT} timed out; not moved')
                continue
            if r.returncode:
                log(f'  condor_qedit {ATTR_AT} failed: {(r.stderr or r.stdout or "").strip()[:200]}; not moved')
                continue
            try:
                r = runner(['condor_prio', '-p', str(new - 1000), job], capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                log('  condor_prio timed out')
                continue
            if r.returncode:
                log(f'  condor_prio failed: {(r.stderr or r.stdout or "").strip()[:200]}')
                continue
        moves.append((job, int(bid), new))
    return moves


def watcher_jobs(owner: str, runner=subprocess.run, running_only: bool = False) -> list[str]:
    st = 'JobStatus == 2' if running_only else '(JobStatus == 1 || JobStatus == 2)'
    r = runner(['condor_q', owner, '-constraint', f'JobBatchName == "{WATCH_BATCH}" && {st}',
                '-af', 'ClusterId'], capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f'condor_q failed: {(r.stderr or "").strip()[:200]}')
    return sorted((r.stdout or '').split(), key=int)


def ensure_watcher(owner: str | None = None, runner=subprocess.run, status: str | None = None) -> str | None:
    """Submit the watcher job unless one is queued or running. It runs a frozen copy of this file from WATCH_DIR, bids
    36 on its own 36 -> 56 ladder (WATCH_LADDER) and is a ladder job itself; while no watcher runs, this call does one
    escalation pass in its place (which also moves a waiting watcher up)."""
    owner = owner or getpass.getuser()
    if watcher_jobs(owner, runner):
        if not watcher_jobs(owner, runner, running_only=True):
            try:   # never raise for this stand-in pass: the jobs are queued and the watcher will start
                escalate(owner=owner, runner=runner, log=lambda s: print(s, flush=True))
            except Exception as e:  # noqa: BLE001
                print(f'escalation pass in place of the waiting watcher failed: {e!r}', flush=True)
        return None
    src = Path(__file__).resolve()
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    code = WATCH_DIR / 'code' / sha
    code.mkdir(parents=True, exist_ok=True)
    frozen = code / 'condor_cost.py'
    if not frozen.exists():
        shutil.copyfile(src, frozen)
    sub = WATCH_DIR / f'watch-{sha}.sub'
    # an interpreter every node sees (the submitter's, unless it is node-local)
    py = sys.executable if sys.executable.startswith(('/lustre/', '/fast/', '/home/', '/usr/')) else '/usr/bin/python3'
    excluded = bad_nodes()
    reqs = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in excluded)
    sub.write_text(f'''universe = vanilla
executable = {py}
arguments = -I {frozen} watch
initialdir = {WATCH_DIR}
request_cpus = {WATCH_SHAPE[0]}
request_memory = {WATCH_SHAPE[1] * 1024}
request_disk = 1G
requirements = {reqs}
periodic_remove = (JobStatus == 5)
+BypassLXCfs = "true"
+JobBatchName = "{WATCH_BATCH}"
{ladder_lines(WATCH_LADDER, WATCH_STEPS).rstrip()}
getenv = false
environment = "HOME={os.environ.get('HOME', '/home/' + owner)} USER={owner} PATH=/usr/local/bin:/usr/bin:/bin"
output = {WATCH_DIR}/watch.$(Cluster).out
error = {WATCH_DIR}/watch.$(Cluster).err
log = {WATCH_DIR}/watch.log
queue
''')
    bid = plan_bids(1, *WATCH_SHAPE, WATCH_LADDER, excluded, status)[0][0]
    r = runner(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, timeout=SUBMIT_TIMEOUT)
    m = re.search(r'submitted to cluster (\d+)', (r.stdout or '') + (r.stderr or ''))
    if r.returncode or not m:
        raise RuntimeError(f'watcher submit failed: {(r.stdout or "") + (r.stderr or "")}')
    print(f'bid-ladder watcher: cluster {m.group(1)} (log {WATCH_DIR}/watch.<cluster>.out)', flush=True)
    return m.group(1)


def watch(interval: float = 60, idle_exit: float = WATCH_IDLE_EXIT) -> None:
    """The watcher loop: escalate every `interval` s; exit after `idle_exit` s with no ladder job queued or running,
    or when an older watcher exists (one watcher per user)."""
    owner = getpass.getuser()
    last_seen = time.time()
    while True:
        try:
            ws = watcher_jobs(owner, running_only=True)
            mine = _own_cluster()
            if mine and ws and ws[0] != mine and int(ws[0]) < int(mine):
                print(f'older watcher {ws[0]} is running; exiting', flush=True)
                return
            escalate(owner=owner, log=lambda s: print(s, flush=True))
            if any(j.get('JobStatus') in (1, 2) for j in ladder_jobs(owner, idle_only=False, watchers=False)):
                last_seen = time.time()
            elif time.time() - last_seen > idle_exit:
                print(f'no ladder job for {idle_exit:.0f} s; exiting', flush=True)
                return
        except Exception as e:     # transient schedd errors: try again next round
            print(f'{time.strftime("%FT%TZ", time.gmtime())} pass failed: {e!r}', flush=True)
        time.sleep(interval)


def _own_cluster() -> str | None:
    """This job's ClusterId when running under condor (from the job ad file), else None."""
    ad = os.environ.get('_CONDOR_JOB_AD')
    try:
        for line in Path(ad).read_text().splitlines() if ad else []:
            if line.startswith('ClusterId'):
                return line.split('=', 1)[1].strip()
    except OSError:
        pass
    return None


def status(owner: str | None = None) -> dict:
    owner = owner or getpass.getuser()
    out: dict = {}
    for j in ladder_jobs(owner, idle_only=False, watchers=False):
        k = f"bid {int(_num(j.get('JobPrio'), -1000) + 1000)} {'idle' if j.get('JobStatus') == 1 else 'running' if j.get('JobStatus') == 2 else 'status ' + str(j.get('JobStatus'))}"
        out[k] = out.get(k, 0) + 1
    return {'jobs': out, 'watchers': watcher_jobs(owner)}


def fill_report(paths) -> dict:
    """Summary of fill.jsonl records (files, or directories searched recursively): batches and jobs per decision, how
    often the fill was capped, per pipeline, and the rung mix (jobs per starting bid) as submitted against the
    unfilled plan, so a campaign shows whether filling moved jobs off the cheap tier."""
    files = []
    for x in map(Path, paths):
        files += sorted(x.rglob(FILL_RECORD)) if x.is_dir() else [x]
    out = {'files': len(files), 'batches': 0, 'jobs': 0, 'decisions': {}, 'capped_share': None, 'pipelines': {},
           'rungs_used': {}, 'rungs_unfilled': {}, 'bad_lines': 0}
    for f in files:
        try:
            lines = f.read_text(errors='replace').splitlines()
        except OSError:
            out['bad_lines'] += 1                    # an unreadable file counts as one bad line
            continue
        for line in lines:
            try:
                r = json.loads(line)
                d, n = str(r['decision']), int(r.get('n') or 0)
                plans = r.get('plans') or {}
                used = plans.get('used')
                rungs = {name: [(str(bid), int(cnt)) for bid, cnt in (pl or [])]
                         for name, pl in (('rungs_used', used), ('rungs_unfilled', plans.get('unfilled') or used))}
                pk = str(r.get('pipeline'))
            except (ValueError, KeyError, TypeError, AttributeError):
                out['bad_lines'] += 1
                continue
            out['batches'] += 1
            out['jobs'] += n
            k = out['decisions'].setdefault(d, {'batches': 0, 'jobs': 0})
            k['batches'] += 1
            k['jobs'] += n
            out['pipelines'][pk] = out['pipelines'].get(pk, 0) + 1
            for name, pairs in rungs.items():
                for bid, cnt in pairs:
                    out[name][bid] = out[name].get(bid, 0) + cnt
    filled = sum(out['decisions'].get(d, {}).get('batches', 0) for d in ('filled', 'capped'))
    if filled:
        out['capped_share'] = round(out['decisions'].get('capped', {}).get('batches', 0) / filled, 3)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('submit')
    s.add_argument('sub')
    s.add_argument('--ladder', default=','.join(map(str, LADDER)))
    s.add_argument('--no-watcher', action='store_true')
    p = sp.add_parser('plan')
    p.add_argument('cpus', type=int)
    p.add_argument('mem_gb', type=float)
    p.add_argument('n', type=int, nargs='?', default=1)
    e = sp.add_parser('escalate')
    e.add_argument('--dry-run', action='store_true')
    w = sp.add_parser('watch')
    w.add_argument('--interval', type=float, default=60)
    w.add_argument('--idle-exit', type=float, default=WATCH_IDLE_EXIT)
    sp.add_parser('status')
    sp.add_parser('ensure-watcher')
    fr = sp.add_parser('fill-report', help='summarize fill.jsonl decision records (files or directories)')
    fr.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'submit':
        recs = submit_file(a.sub, tuple(int(x) for x in a.ladder.split(',')), watcher=not a.no_watcher)
        print(' '.join(r['cluster'] for r in recs))
    elif a.cmd == 'plan':
        print(plan_bids(a.n, a.cpus, a.mem_gb))
    elif a.cmd == 'escalate':
        escalate(dry_run=a.dry_run)
    elif a.cmd == 'watch':
        watch(a.interval, a.idle_exit)
    elif a.cmd == 'status':
        print(json.dumps(status(), indent=1))
    elif a.cmd == 'fill-report':
        print(json.dumps(fill_report(a.paths), indent=1))
    elif a.cmd == 'ensure-watcher':
        print(ensure_watcher() or 'already queued or running')


if __name__ == '__main__':
    main()

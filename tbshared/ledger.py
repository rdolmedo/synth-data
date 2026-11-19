"""Credit ledger from HTCondor user event logs (tbshared; generalized from the c30 audit, #378).

condor_history keeps only the newest ~10k records on this schedd, so usage is rebuilt from the user event logs
(`log = ...` in a submit file). One run segment per execute event (001), up to the next 004 evicted / 005
terminated / 009 aborted / 012 held / 007 shadow exception / 024 reconnect failed of the same job ('open' if none;
a second 001 without an end closes the previous segment as 'restart-no-end'). A memory hold here is a 007 and a 012
in the same second; the 007 closes the segment, which is then marked held.

Event logs lose events (c30, 11-12: condor drops records when the log's filesystem is full), so `close_stale`
ends open segments of jobs that are not running: at the job's EnteredCurrentStatus when it is still queued
(`condor_q`), else at the later of the last write to its .out/.err files and the segment's last shadow event
(006 / 022 / 023; Pi workers and servers can be silent on .out/.err) ('open-stale'). A real removal here writes a
004 and a 009 in the same second, so a 009 that closes a running segment ("bare") means the end event was lost: when
it was written more than an hour after that last activity it is capped there ('aborted-late'; the logged end is
kept in end_logged), else kept at the 009 time and counted.

Known limitations: an open segment of a job whose hold was lost and that was released since ends at the release
(its EnteredCurrentStatus), so it is charged through the hold. Timestamps in the repeated hour of the October DST
change (02:00-03:00) are ambiguous and read as the first (CEST) one.

Requested resources of a segment come from its own 004/005 resource table ("Partitionable Resources : Usage
Request Allocated"), else the job's last known request before the segment ended (an earlier table, or the memory
limit of a hold message, which is 0.9 x the request), else the first one after it, else another job of the same
cluster, else the submit file (`annotate`: the cluster's `jobs.json` record next to the log, or the newest `*.sub`
next to it), else unknown (not costed; reported as unknown-request hours). A submit file written more than 2 min
after the job was submitted (rewritten since) is not used for the request.

Charges (Confluence "Cluster banking system"): units = max(RequestCpus, RequestMemory_GB / 16) + 24 per GPU, per
hour, on REQUESTED resources, at the node's running price, which is at least the node's floor (MinRunningPrice).
Floors are per node and are read, never inferred from host names: most CPU-only and A100 nodes are 5-25, H100 35,
B200 55, but some A100 and CPU-only nodes are 35 (g181-g184, cpu105 in 11-2026). Hosts missing from every floors
snapshot are 'unknown' and not charged at floor (reported as unknown-floor hours); pass older snapshots to cover
retired nodes. Credits at floor are a lower bound; credits at the job's bid (its jobs.json `bid`, the submission
bid; a ladder rebid is not in the event log) are an estimate, at max(bid, floor).

Event-log timestamps are cluster-local time without a zone: Europe/Berlin, CET/CEST (checked around the
2026-10-25 change; the c30 audit used CET = UTC+1, right only in winter). That is the default; `tz` / `--tz` takes
'local' (the system zone), 'UTC', '+01:00' or an IANA name. Without zoneinfo / tzdata the default falls back to the
system zone, with a warning.

  python -m tbshared.ledger LOG [LOG...] [--since ISO] [--until ISO] [--group batch|log|node-kind|floor|host|job]
      [--floors FILE ...] [--live] [--save-floors FILE] [--queue FILE | --no-queue] [--tz ZONE] [--json OUT]

LOG may be a glob (quote it). --since/--until without a zone are UTC. --group takes a comma list (node-kind,floor).
Without --floors, floors come from a live `condor_status` (--live adds it before the files). Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:          # pragma: no cover (Python < 3.9)
    ZoneInfo = None

CLUSTER_TZ = 'Europe/Berlin'
END_EVENTS = {'004': 'evicted', '005': 'terminated', '009': 'aborted', '012': 'held', '007': 'shadow-exception',
              '024': 'reconnect-failed'}
KINDS = {'10.0': 'B200', '9.0': 'H100', '8.0': 'A100', '7.5': 'RTX6000'}   # CUDACapability -> node kind
REQ_ORDER = ('table', 'earlier', 'later', 'cluster', 'submit', 'submit-guess', 'unknown')   # best first
SUBMIT_SLACK = 120           # s: a submit file written later than this after the job's submission was rewritten
STALE_AFTER = 3600           # s: output older than this (before now, or before a 009) means the job had stopped

HDR = re.compile(r'^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?'
                 r'(Z|[+-]\d{2}:?\d{2})?\s?(.*)$')
HOST = re.compile(r'alias=([A-Za-z0-9-]+)')
CPUT = re.compile(r'Usr (\d+) (\d+):(\d+):(\d+), Sys (\d+) (\d+):(\d+):(\d+)\s+-\s+Run Remote Usage')
RES_HEAD = re.compile(r'^\s*Partitionable Resources\s*:(.*)$')
RES = re.compile(r'^\s+(Cpus|Disk \(KB\)|GPUs|Gpus(?: \(Average\))?|Memory \(MB\))\s*:(.*)$')
MEMU = re.compile(r'^\s+(\d+)\s+-\s+MemoryUsage of job \(MB\)')
HOLDLIM = re.compile(r'memory limit of (\d+) megabytes\. Peak usage: (\d+) megabytes')
RETVAL = re.compile(r'Normal termination \(return value (-?\d+)\)')
# The memory limit in a hold (012) or shadow-exception (007) message is floor(0.9 x RequestMemory) on this cluster:
# 166 of 166 holds checked against their submit files (eval_q35 + c30 logs, 2026-11-14), e.g. 294912 for 327680 MB.
HOLD_LIMIT_FRACTION = 0.9


def request_from_limit(limit_mb: float) -> float:
    """RequestMemory (MB) behind a hold message's memory limit (the smallest request whose limit it is)."""
    return float(math.ceil(limit_mb / HOLD_LIMIT_FRACTION - 1e-6))


def units(cpus: float, mem_gib: float, gpus: float = 0) -> float:
    """Charged units per hour: max(CPUs, RAM GB / 16) + 24 per GPU (the same rule as condor_cost.units; repeated
    so this file runs on its own)."""
    return max(float(cpus), float(mem_gib) / 16) + 24 * float(gpus)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


# ------------------------------------------------------------------------------------------------ time
def parse_tz(s):
    """A zone for event-log timestamps: 'cluster' (Europe/Berlin), 'local' / None (the system zone, as None), 'UTC',
    a fixed offset '+01:00', or an IANA name. ValueError otherwise."""
    if s is None or s == 'local':
        return None
    if isinstance(s, dt.tzinfo):
        return s
    if s == 'cluster':
        z = _cluster_zone()
        if z is not None:
            return z
        _warn_once(f'WARNING: zone {CLUSTER_TZ} unavailable (no zoneinfo / tzdata): event-log times are read in the '
                   'system zone')
        return None
    if s.upper() in ('UTC', 'Z'):
        return dt.timezone.utc
    m = re.fullmatch(r'([+-])(\d{1,2}):?(\d{2})?', s)
    if m:
        off = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return dt.timezone(off if m.group(1) == '+' else -off)
    if ZoneInfo is not None and re.fullmatch(r'[A-Za-z_]+(/[A-Za-z0-9_+-]+)*', s):
        try:
            return ZoneInfo(s)
        except Exception:     # ZoneInfoNotFoundError, or a key that is not a zone file
            pass
    raise ValueError(f'timezone {s!r}: use cluster, local, UTC, +HH:MM or an IANA name (Europe/Berlin)')


def _cluster_zone():
    try:
        return ZoneInfo(CLUSTER_TZ) if ZoneInfo is not None else None
    except Exception:         # ZoneInfoNotFoundError: no tzdata
        return None


_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _stderr(msg)


def _epoch(date: str, clock: str, zone: str | None, tz) -> float:
    if zone:
        z = dt.timezone.utc if zone == 'Z' else dt.timezone(
            (1 if zone[0] == '+' else -1) * dt.timedelta(hours=int(zone[1:3]), minutes=int(zone[-2:])))
        return dt.datetime.fromisoformat(f'{date} {clock}').replace(tzinfo=z).timestamp()
    if tz is None:
        return time.mktime(time.strptime(f'{date} {clock}', '%Y-%m-%d %H:%M:%S'))
    return dt.datetime.fromisoformat(f'{date} {clock}').replace(tzinfo=tz).timestamp()


def parse_time(s: str | None) -> float | None:
    """ISO date/time -> epoch; no zone = UTC."""
    if not s:
        return None
    t = dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).timestamp()


# ------------------------------------------------------------------------------------------------ parsing
def events(path, tz='cluster'):
    """(code, 'cluster.proc', epoch, header rest, body lines) per event, in file order."""
    tz = parse_tz(tz)
    cur, body = None, []
    with open(path, errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            m = HDR.match(line)
            if m:
                if cur:
                    yield (*cur, body)
                code, cl, pr, date, clock, zone, rest = m.groups()
                cur, body = (code, f'{int(cl)}.{int(pr)}', _epoch(date, clock, zone, tz), rest), []
            elif line.strip() == '...':
                if cur:
                    yield (*cur, body)
                cur, body = None, []
            elif cur:
                body.append(line)
    if cur:
        yield (*cur, body)


def _row(text: str, cols) -> dict:
    """Column name -> number of one resource-table row (the text after its ':'). Numbers are right-aligned under the
    header's column names; quoted values (Assigned "GPU-...") are skipped. Without a header: Usage Request Allocated
    when 3 numbers, Request Allocated when 2."""
    toks = [(float(m.group()), m.end()) for m in re.finditer(r'"[^"]*"|\S+', text)
            if re.fullmatch(r'\d+(?:\.\d+)?', m.group())]
    if cols:
        out = {}
        for v, end in toks:
            out[min(cols, key=lambda c: abs(c[1] - end))[0]] = v
        return out
    names = ('Usage', 'Request', 'Allocated') if len(toks) >= 3 else ('Request', 'Allocated')
    return dict(zip(names, (v for v, _ in toks)))


def _resources(body):
    """(request {'cpus', 'mem_mb', 'gpus'}, memory usage MB, CPU seconds) from a 004/005 body. The request is the
    Request column, else Allocated (a row that only shows the allocation)."""
    req, mem_use, cpu, cols = {}, None, None, None
    for b in body:
        m = CPUT.search(b)
        if m and cpu is None:
            g = list(map(int, m.groups()))
            cpu = sum(g[i] * 86400 + g[i + 1] * 3600 + g[i + 2] * 60 + g[i + 3] for i in (0, 4))
        m = RES_HEAD.match(b)
        if m:
            cols = [(c.group(), c.end()) for c in re.finditer(r'\S+', m.group(1))]
            continue
        m = RES.match(b)
        if not m:
            continue
        row = _row(m.group(2), cols)
        rq = row.get('Request', row.get('Allocated'))
        name = m.group(1)
        if name == 'Memory (MB)' and 'Usage' in row:
            mem_use = row['Usage']
        if rq is None:
            continue
        if name == 'Cpus':
            req['cpus'] = rq
        elif name == 'Memory (MB)':
            req['mem_mb'] = rq
        elif name.startswith(('GPUs', 'Gpus')):
            req['gpus'] = rq
    return req, mem_use, cpu


def parse_log(path, tz='cluster', warn=_stderr) -> list[dict]:
    """Run segments of one event log, in start order. Each: log, job ('cluster.proc'), cluster, proc, submit, start,
    end (None if open), host, end_event (code or 'open'), end_how, cpu_s (Run Remote Usage, Usr + Sys),
    return_value, cpus / mem_mb / gpus (requested) with req_src, peak_mem_mb (006 MemoryUsage and the table's
    usage), hold_limit_mb / hold_peak_mb (memory-limit message), held (ended by a hold: a 012, a 012 within 2 s
    after the 007 that ended it, or a memory-limit message), holds (the job's hold count in this log). A 004/005
    without an open segment (its 001 lost) is reported through `warn`; its run time is not counted."""
    tz = parse_tz(tz)
    jobs: dict = {}
    segs: list = []
    orphans: list = []
    for code, job, t, rest, body in events(path, tz):
        st = jobs.setdefault(job, {'submit': None, 'open': None, 'last': None, 'obs': [], 'holds': 0})
        if code == '000':
            st['submit'] = t if st['submit'] is None else st['submit']
        elif code == '001':
            if st['open'] is not None:
                st['open'].update(end=t, end_event='001', end_how='restart-no-end')
                segs.append(st['open'])
            h = HOST.search(rest + ' ' + ' '.join(body))
            cl, pr = job.split('.')
            st['open'] = {'log': str(path), 'job': job, 'cluster': cl, 'proc': int(pr), 'submit': st['submit'],
                          'start': t, 'end': None, 'host': h.group(1).split('.')[0] if h else None,
                          'end_event': 'open', 'end_how': 'open', 'cpu_s': None, 'return_value': None,
                          'peak_mem_mb': None, 'hold_limit_mb': None, 'hold_peak_mb': None, 'held': False,
                          'last_seen': None, '_table': {}}
        elif code in ('022', '023'):      # disconnected / reconnected: the shadow still saw the job
            if st['open'] is not None:
                st['open']['last_seen'] = t
        elif code == '006':
            seg = st['open']
            if seg is not None:
                seg['last_seen'] = t
            for b in body:
                m = MEMU.match(b)
                if m and seg is not None:
                    seg['peak_mem_mb'] = max(seg['peak_mem_mb'] or 0, float(m.group(1)))
        elif code in END_EVENTS:
            table, mem_use, cpu = _resources(body) if code in ('004', '005') else ({}, None, None)
            text = ' '.join(body)
            lim = HOLDLIM.search(text)
            obs = dict(table)
            if lim and 'mem_mb' not in obs:
                obs['mem_mb'] = request_from_limit(float(lim.group(1)))
            if obs:
                st['obs'].append((t, obs))
            if code == '012':
                st['holds'] += 1
            seg = st['open']
            if seg is None:
                last = st['last']
                if code == '012' and last is not None and last['end_event'] == '007' and 0 <= t - last['end'] <= 2:
                    last['held'] = True        # the memory hold: 007 closed the segment, 012 follows
                    if lim and last['hold_limit_mb'] is None:
                        last['hold_limit_mb'], last['hold_peak_mb'] = float(lim.group(1)), float(lim.group(2))
                elif code in ('004', '005'):
                    orphans.append(job)
                continue
            seg.update(end=t, end_event=code, end_how=END_EVENTS[code], cpu_s=cpu, _table=table)
            if mem_use is not None:
                seg['peak_mem_mb'] = max(seg['peak_mem_mb'] or 0, mem_use)
            if lim:
                seg['hold_limit_mb'], seg['hold_peak_mb'] = float(lim.group(1)), float(lim.group(2))
            seg['held'] = code == '012' or bool(lim)
            rv = RETVAL.search(text)
            if rv:
                seg['return_value'] = int(rv.group(1))
            segs.append(seg)
            st['open'], st['last'] = None, seg
    if orphans and warn:
        warn(f'WARNING: {path}: {len(orphans)} end event(s) (004/005) without an execute (001), not counted: jobs '
             f'{", ".join(sorted(set(orphans))[:5])}{" ..." if len(set(orphans)) > 5 else ""}')
    segs.extend(st['open'] for st in jobs.values() if st['open'] is not None)
    by_cluster: dict = {}
    for job, st in jobs.items():
        for _, o in st['obs']:
            by_cluster.setdefault(job.split('.')[0], {}).update(o)
    for seg in segs:
        st = jobs[seg['job']]
        seg['holds'] = st['holds']
        _resolve_request(seg, st['obs'], by_cluster.get(seg['cluster'], {}))
    segs.sort(key=lambda s: (s['start'], s['job']))
    return segs


def _resolve_request(seg, obs, cluster_req):
    table = seg.pop('_table')
    req, src = dict(table), 'table'
    if not ('cpus' in req and 'mem_mb' in req):
        end = math.inf if seg['end'] is None else seg['end']
        before: dict = {}
        for t, o in obs:
            if t <= end:
                before.update(o)
        after = [o for t, o in obs if t > end]
        for name, pool in (('earlier', [before]), ('later', after), ('cluster', [cluster_req])):
            for key in ('cpus', 'mem_mb', 'gpus'):
                if key not in req:
                    v = next((o[key] for o in pool if key in o), None)
                    if v is not None:
                        req[key] = v
                        src = max(src, name, key=REQ_ORDER.index)
        if not ('cpus' in req and 'mem_mb' in req):
            src = 'unknown'
    if 'gpus' not in req and src != 'unknown':
        req['gpus'] = 0.0          # a table without a GPU line, or a request known without one: a CPU job
    seg.update(cpus=req.get('cpus'), mem_mb=req.get('mem_mb'), gpus=req.get('gpus'), req_src=src)


# ------------------------------------------------------------------------------------------------ submit files
_CACHE: dict = {}


def read_submit(path) -> dict:
    """{'cpus', 'mem_mb', 'gpus', 'batch', 'output', 'error', 'initialdir', 'mtime'} from a submit file's literal
    request_* (request_memory: also the base of the memory net's expression), +JobBatchName / batch_name, output /
    error / initialdir lines."""
    key = ('sub', str(path))
    if key in _CACHE:
        return _CACHE[key]
    out: dict = {}
    try:
        txt = Path(path).read_text(errors='replace')
        out['mtime'] = Path(path).stat().st_mtime
    except OSError:
        txt = ''
    for k, name in (('request_cpus', 'cpus'), ('request_memory', 'mem_mb'), ('request_gpus', 'gpus')):
        # request_memory may be the memory net's `BASE + STEP * <memory holds>` (condor_cost.memory_net): the base
        tail = r'(?:[ \t]*\+[^\n]*)?' if name == 'mem_mb' else ''
        m = re.search(rf'^\s*{k}\s*=\s*([\d.]+)\s*([KMGT]?B?)\s*{tail}$', txt, re.M | re.I)
        if m:
            v, unit = float(m.group(1)), m.group(2).upper().rstrip('B')
            if name == 'mem_mb':
                v *= {'K': 1 / 1024, '': 1, 'M': 1, 'G': 1024, 'T': 1024 * 1024}[unit]
            out[name] = v
    m = re.search(r'^\s*(?:\+JobBatchName|batch_name)\s*=\s*"?([^"\n]+?)"?\s*$', txt, re.M | re.I)
    if m:
        out['batch'] = m.group(1)
    for k in ('output', 'error', 'initialdir'):
        m = re.search(rf'^\s*{k}\s*=\s*(\S+)\s*$', txt, re.M | re.I)
        if m:
            out[k] = m.group(1)
    _CACHE[key] = out
    return out


def job_records(log_dir) -> dict:
    """cluster -> record ({'cluster', 'submit_file', 'bid', ...}) from jobs.json in the log's directory or its
    parent (a list of records or {'jobs': [...]}, as eval_q35 and aa-evals write)."""
    log_dir = Path(log_dir)
    key = ('jobs', str(log_dir))
    if key in _CACHE:
        return _CACHE[key]
    recs: dict = {}
    for jj in (log_dir.parent / 'jobs.json', log_dir / 'jobs.json'):     # the nearer file wins
        try:
            data = json.loads(jj.read_text())
        except (OSError, ValueError):
            continue
        for r in (data.get('jobs', []) if isinstance(data, dict) else data):
            if isinstance(r, dict) and r.get('cluster') is not None:
                recs[str(r['cluster'])] = r
    _CACHE[key] = recs
    return recs


def _guess_submit(log_dir, submit_t) -> str | None:
    """The newest *.sub next to the log (or one level up) written by SUBMIT_SLACK after the job was submitted, else
    the oldest one (which `annotate` then uses for the batch name only)."""
    log_dir = Path(log_dir)
    cands = []
    for p in list(log_dir.glob('*.sub')) + list(log_dir.parent.glob('*.sub')):
        try:
            cands.append((p.stat().st_mtime, str(p)))
        except OSError:
            pass
    if not cands:
        return None
    before = [c for c in cands if submit_t is not None and c[0] <= submit_t + SUBMIT_SLACK]
    return max(before)[1] if before else min(cands)[1]


def annotate(segs: list[dict]) -> list[dict]:
    """Add batch, bid and submit_file from the cluster's jobs.json record and submit file, and fill unknown requests
    from the submit file (req_src 'submit', or 'submit-guess' for a *.sub picked by time). A submit file written
    more than SUBMIT_SLACK after the job was submitted has been rewritten: submit_after is set and its request is
    not used (its batch name still is)."""
    for s in segs:
        d = Path(s['log']).parent
        rec = job_records(d).get(s['cluster'], {})
        sub, how = rec.get('submit_file'), 'submit'
        if not sub:
            sub, how = _guess_submit(d, s.get('submit') or s['start']), 'submit-guess'
        info = read_submit(sub) if sub else {}
        t0 = s.get('submit') or s['start']
        s['submit_after'] = bool(info.get('mtime') and info['mtime'] > t0 + SUBMIT_SLACK)
        try:
            s['bid'] = float(rec['bid']) if rec.get('bid') is not None else None
        except (TypeError, ValueError):
            s['bid'] = None
        s['submit_file'] = sub
        s['batch'] = info.get('batch') or rec.get('batch') or rec.get('role')
        if s['req_src'] == 'unknown' and 'cpus' in info and 'mem_mb' in info and not s['submit_after']:
            s.update(cpus=info['cpus'], mem_mb=info['mem_mb'], gpus=info.get('gpus', 0.0), req_src=how)
    return segs


def expand(patterns) -> list[str]:
    """Paths from paths/globs, each real file once."""
    out, seen = [], set()
    for p in patterns:
        for f in (sorted(glob.glob(p)) if glob.has_magic(p) else [p]):
            r = os.path.realpath(f)
            if r not in seen and os.path.isfile(r):
                seen.add(r)
                out.append(f)
    return out


def load(patterns, tz='cluster', with_submit: bool = True, warn=_stderr) -> list[dict]:
    """Segments of every log (globs allowed). The same job can appear in two copies of a log (run dirs copied
    between filesystems): segments are deduplicated by (job, start), keeping the copy that knows most."""
    tz = parse_tz(tz)
    best: dict = {}
    for f in expand(patterns):
        segs = parse_log(f, tz, warn)
        if with_submit:
            annotate(segs)
        for s in segs:
            score = (s['end'] is not None, -REQ_ORDER.index(s['req_src']), s['cpu_s'] is not None)
            k = (s['job'], s['start'])
            if k not in best or score > best[k][0]:
                best[k] = (score, s)
    return sorted((v[1] for v in best.values()), key=lambda s: (s['start'], s['job']))


# ------------------------------------------------------------------------------------------------ stale ends
def parse_queue(text: str) -> dict:
    """'cluster.proc' -> (JobStatus, EnteredCurrentStatus) from `condor_q -af ClusterId ProcId JobStatus
    EnteredCurrentStatus` lines (the c30 audit's queue_now.txt)."""
    q = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 4 and p[0].isdigit() and p[1].isdigit():
            try:
                q[f'{int(p[0])}.{int(p[1])}'] = (int(p[2]), float(p[3]))
            except ValueError:
                pass
    return q


def condor_q_text(clusters, runner=subprocess.run, chunk: int = 200) -> str:
    """Read-only `condor_q <clusters> -af ClusterId ProcId JobStatus EnteredCurrentStatus`."""
    out, clusters = [], sorted(set(map(str, clusters)), key=int)
    for i in range(0, len(clusters), chunk):
        r = runner(['condor_q', *clusters[i:i + chunk], '-af', 'ClusterId', 'ProcId', 'JobStatus',
                    'EnteredCurrentStatus'], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise RuntimeError(f'condor_q failed: {(r.stderr or "").strip()[:200]}')
        out.append(r.stdout or '')
    return ''.join(out)


def output_files(seg) -> list[Path]:
    """The job's .out/.err files: its submit file's output/error ($(Cluster), $(Process) expanded), and
    *<cluster>.<proc>.out/.err next to the log (*<cluster>.out/.err when there are none)."""
    cl, pr = seg['cluster'], str(seg['proc'])
    d = Path(seg['log']).parent
    files = []
    info = read_submit(seg['submit_file']) if seg.get('submit_file') else {}
    base = Path(info.get('initialdir') or Path(seg['submit_file']).parent) if info else d
    for k in ('output', 'error'):
        if info.get(k):
            p = re.sub(r'\$\((?:Cluster|ClusterId)\)', cl, info[k], flags=re.I)
            p = re.sub(r'\$\((?:Process|ProcId)\)', pr, p, flags=re.I)
            if '$(' not in p:
                files.append(base / p)
    for pat in (f'*{cl}.{pr}.out', f'*{cl}.{pr}.err'):
        files += d.glob(pat)
    if not any(f.exists() for f in files):
        for pat in (f'*{cl}.out', f'*{cl}.err'):
            files += d.glob(pat)
    return files


def last_output(seg) -> float | None:
    ms = []
    for f in output_files(seg):
        try:
            ms.append(f.stat().st_mtime)
        except OSError:
            pass
    return max(ms) if ms else None


def close_stale(segs: list[dict], queue: dict | None = None, now: float | None = None,
                stale_after: float = STALE_AFTER) -> dict:
    """End segments the event log left open or closed late, in place. For each job's last segment:
    - open, and the job is queued but not running (`queue`: parse_queue output; None = unknown): end at its
      EnteredCurrentStatus;
    - open, not in the queue: end at its last activity, the later of the last output write and the segment's last
      shadow event (last_seen: 006 / 022 / 023), if the queue is unknown only when that is older than stale_after;
      with no output files it stays open ('open-unresolved' when the queue says the job is gone);
    - ended by a bare 009 (a lost end event) more than stale_after after its last activity: end there (end_logged
      keeps the 009 time); other bare 009s are kept and counted (aborted_kept).
    Fixed segments get end_how 'open-stale' / 'aborted-late' and end_src 'queue' / 'output'. Returns counts."""
    now = time.time() if now is None else now
    last: dict = {}
    for s in segs:
        if s['job'] not in last or s['start'] >= last[s['job']]['start']:
            last[s['job']] = s
    stats = {'open_queue': 0, 'open_output': 0, 'open_running': 0, 'open_left': 0, 'aborted_late': 0,
             'aborted_late_h': 0.0, 'aborted_kept': 0}

    def activity(seg):
        m = last_output(seg)
        return None if m is None else max(m, seg.get('last_seen') or seg['start'])
    for s in last.values():
        if s['end'] is None:
            q = queue.get(s['job']) if queue is not None else None
            if q and q[0] == 2:
                stats['open_running'] += 1
                continue
            if q and q[1] >= s['start']:
                s.update(end=q[1], end_how='open-stale', end_src='queue')
                stats['open_queue'] += 1
                continue
            m = activity(s)
            if m is not None and m >= s['start'] and (queue is not None or now - m > stale_after):
                s.update(end=m, end_how='open-stale', end_src='output')
                stats['open_output'] += 1
                continue
            if queue is not None and q is None:
                s['end_how'] = 'open-unresolved'
            stats['open_left'] += 1
        elif s['end_event'] == '009':
            m = activity(s)
            if m is not None and s['start'] <= m < s['end'] - stale_after:
                stats['aborted_late'] += 1
                stats['aborted_late_h'] += (s['end'] - m) / 3600
                s.update(end_logged=s['end'], end=m, end_how='aborted-late', end_src='output')
            else:
                stats['aborted_kept'] += 1
    return stats


# ------------------------------------------------------------------------------------------------ floors
def parse_floors(text: str) -> dict:
    """host -> (floor, kind) from `condor_status -af UtsnameNodename MinRunningPrice CUDACapability` lines (one per
    slot; dynamic slots without GPUs read CUDACapability undefined, so a host's kind is its first defined capability
    and its floor the highest seen). Quoted, fully qualified host names (the c30 nodes.txt) are accepted."""
    floor: dict = {}
    cap: dict = {}
    for line in text.splitlines():
        m = re.match(r'^\s*"?([^"\s]+)"?\s+(\S+)(?:\s+(\S+))?', line)
        if not m:
            continue
        host = m.group(1).split('.')[0]
        try:
            f = float(m.group(2))
        except ValueError:
            continue
        floor[host] = max(floor.get(host, f), f)
        c = (m.group(3) or 'undefined').strip('"')
        if c != 'undefined' and host not in cap:
            cap[host] = c
    return {h: (f, KINDS.get(cap.get(h), 'CPU' if h not in cap else f'cc{cap[h]}')) for h, f in floor.items()}


def status_floors_text(runner=subprocess.run) -> str:
    r = runner(['condor_status', '-af', 'UtsnameNodename', 'MinRunningPrice', 'CUDACapability'],
               capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f'condor_status failed: {(r.stderr or "").strip()[:200]}')
    return r.stdout or ''


def floors_from_status(runner=subprocess.run) -> dict:
    """host -> (floor, kind), live."""
    return parse_floors(status_floors_text(runner))


def load_floors(*paths) -> dict:
    """host -> (floor, kind) from saved snapshots (the condor_status output above); for a host in several, the
    first file that has it wins (pass the newest first, older ones for retired nodes)."""
    out: dict = {}
    for p in paths:
        for h, v in parse_floors(Path(p).read_text()).items():
            out.setdefault(h, v)
    return out


def node_floor(host, floors: dict | None) -> tuple[float | None, str, str]:
    """(floor, kind, 'status') for a host in the snapshot, else (None, 'unknown', 'unknown'): floors are not
    inferred from names (g181-g184 A100 and g187-g206 H100 are 35, cpu105 is a CPU node at 35)."""
    if floors and host in floors:
        f, kind = floors[host]
        return float(f), kind, 'status'
    return None, 'unknown', 'unknown'


# ------------------------------------------------------------------------------------------------ costs
def segment_cost(seg: dict, floors: dict | None = None, since: float | None = None, until: float | None = None,
                 now: float | None = None, bid: float | None = None) -> dict:
    """Cost of one segment inside [since, until]. An open segment runs to min(until, now). Returns hours, gpu_h,
    units, unit_h, floor, node_kind, floor_src, credits_floor (0 when the floor is unknown), bid, credits_bid (at
    max(bid, floor): a recorded bid below the node's floor was raised in place after submission), cpu_used_h /
    cpu_req_h (where CPU time is known, prorated to the window), mem_peak_gbh / mem_req_gbh (where the peak is
    known). units is None (and nothing is charged) when the request is unknown."""
    now = time.time() if now is None else now
    end = seg['end'] if seg['end'] is not None else (min(until, now) if until is not None else now)
    a = seg['start'] if since is None else max(seg['start'], since)
    b = end if until is None else min(end, until)
    hours = max(0.0, b - a) / 3600
    full_h = max(0.0, end - seg['start']) / 3600
    floor, kind, fsrc = node_floor(seg.get('host'), floors)
    known = seg.get('cpus') is not None and seg.get('mem_mb') is not None
    u = units(seg['cpus'], seg['mem_mb'] / 1024, seg.get('gpus') or 0) if known else None
    bid = seg.get('bid') if bid is None else bid
    cpu_known = known and seg.get('cpu_s') is not None and full_h > 0
    peak_known = known and seg.get('peak_mem_mb') is not None
    return {
        'hours': hours, 'gpu_h': (seg.get('gpus') or 0) * hours, 'units': u, 'unit_h': (u or 0) * hours,
        'floor': floor, 'node_kind': kind, 'floor_src': fsrc, 'credits_floor': (u or 0) * hours * (floor or 0),
        'bid': bid, 'credits_bid': (u or 0) * hours * max(bid, floor or 0) if bid is not None else None,
        'cpu_used_h': seg['cpu_s'] / 3600 * hours / full_h if cpu_known else None,
        'cpu_req_h': seg['cpus'] * hours if cpu_known else None,
        'mem_peak_gbh': seg['peak_mem_mb'] / 1024 * hours if peak_known else None,
        'mem_req_gbh': seg['mem_mb'] / 1024 * hours if peak_known else None,
    }


def costed(segs, floors=None, since=None, until=None, now=None) -> list[dict]:
    """Each segment merged with its cost, for segments with time inside the window."""
    out = []
    for s in segs:
        c = segment_cost(s, floors, since, until, now)
        if c['hours'] > 0:
            out.append({**s, **c})
    return out


GROUPS = {
    'batch': lambda r: r.get('batch') or '(no batch)',
    'log': lambda r: r['log'],
    'node-kind': lambda r: r['node_kind'],
    'floor': lambda r: 'floor ?' if r['floor'] is None else f"floor {r['floor']:g}",
    'host': lambda r: r.get('host') or '?',
    'job': lambda r: r['job'],
}


def group_key(names: str):
    """A key over costed rows from a comma list of GROUPS names ('node-kind,floor')."""
    fns = [GROUPS[n.strip()] for n in names.split(',') if n.strip()]
    return lambda r: ' / '.join(str(f(r)) for f in fns)


def summarize(segs, key=lambda r: 'TOTAL', floors=None, since=None, until=None, now=None) -> dict:
    """group -> totals over the segments inside the window. Rows already costed (costed()) are used as they are
    unless floors / since / until / now are given; then they are costed again. `key` gets the segment merged with
    its cost (node_kind, floor, ...). Totals: segments, jobs, open (still open), fixed (end inferred by
    close_stale), held (ended by a hold), job_h, open_h, gpu_h, unit_h, credits_floor, open_credits_floor,
    credits_bid with bid_h (hours with a known bid), unknown_req_h (not costed), unknown_floor_h (not charged at
    floor), cpu_used_h / cpu_req_h / cpu_used_frac, mem_peak_gbh / mem_req_gbh / mem_peak_frac (time-weighted),
    peak_mem_max_gb."""
    pre = segs and all('hours' in s for s in segs) and all(x is None for x in (floors, since, until, now))
    rows = segs if pre else costed(segs, floors, since, until, now)
    out: dict = {}
    jobs: dict = {}
    for r in rows:
        k = key(r)
        g = out.setdefault(k, dict.fromkeys(
            ('segments', 'open', 'fixed', 'held', 'job_h', 'open_h', 'gpu_h', 'unit_h', 'credits_floor',
             'open_credits_floor', 'credits_bid', 'bid_h', 'unknown_req_h', 'unknown_floor_h', 'cpu_used_h',
             'cpu_req_h', 'mem_peak_gbh', 'mem_req_gbh', 'peak_mem_max_gb'), 0))
        jobs.setdefault(k, set()).add(r['job'])
        g['segments'] += 1
        g['fixed'] += bool(r.get('end_src'))
        g['held'] += bool(r.get('held'))
        g['job_h'] += r['hours']
        g['gpu_h'] += r['gpu_h']
        g['unit_h'] += r['unit_h']
        g['credits_floor'] += r['credits_floor']
        if r['end'] is None:
            g['open'] += 1
            g['open_h'] += r['hours']
            g['open_credits_floor'] += r['credits_floor']
        if r['credits_bid'] is not None:
            g['credits_bid'] += r['credits_bid']
            g['bid_h'] += r['hours']
        if r['units'] is None:
            g['unknown_req_h'] += r['hours']
        if r['floor_src'] == 'unknown':
            g['unknown_floor_h'] += r['hours']
        if r['cpu_used_h'] is not None:
            g['cpu_used_h'] += r['cpu_used_h']
            g['cpu_req_h'] += r['cpu_req_h']
        if r['mem_peak_gbh'] is not None:
            g['mem_peak_gbh'] += r['mem_peak_gbh']
            g['mem_req_gbh'] += r['mem_req_gbh']
            g['peak_mem_max_gb'] = max(g['peak_mem_max_gb'], r['peak_mem_mb'] / 1024)
    for k, g in out.items():
        g['jobs'] = len(jobs[k])
        g['cpu_used_frac'] = g['cpu_used_h'] / g['cpu_req_h'] if g['cpu_req_h'] else None
        g['mem_peak_frac'] = g['mem_peak_gbh'] / g['mem_req_gbh'] if g['mem_req_gbh'] else None
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _pct(x):
    return '-' if x is None else f'{100 * x:.0f}%'


def table(groups: dict, total: dict, title: str = 'group') -> str:
    w = max([len(title), 5] + [len(str(k)) for k in groups])
    head = (f"{title:<{w}} {'segs':>6} {'jobs':>6} {'job-h':>9} {'GPU-h':>8} {'unit-h':>10} {'@floor':>11} "
            f"{'@bid':>11} {'CPU used':>8} {'mem peak':>8} {'held':>5} {'open':>5} {'fixed':>5} {'?req-h':>7} "
            f"{'?floor-h':>8}")

    def row(k, g):
        bid = f"{g['credits_bid']:,.0f}" if g['bid_h'] else '-'
        return (f"{str(k):<{w}} {g['segments']:>6} {g['jobs']:>6} {g['job_h']:>9,.1f} {g['gpu_h']:>8,.1f} "
                f"{g['unit_h']:>10,.0f} {g['credits_floor']:>11,.0f} {bid:>11} {_pct(g['cpu_used_frac']):>8} "
                f"{_pct(g['mem_peak_frac']):>8} {g['held']:>5} {g['open']:>5} {g['fixed']:>5} "
                f"{g['unknown_req_h']:>7,.1f} {g['unknown_floor_h']:>8,.1f}")
    lines = [head, '-' * len(head)]
    lines += [row(k, g) for k, g in sorted(groups.items(), key=lambda kv: (-kv[1]['credits_floor'], str(kv[0])))]
    lines += ['-' * len(head), row('TOTAL', total)]
    return '\n'.join(lines)


def main(argv=None, runner=subprocess.run, out=sys.stdout, err=sys.stderr) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='+', help='event logs or globs')
    ap.add_argument('--since', help='window start, ISO (no zone = UTC)')
    ap.add_argument('--until', help='window end, ISO (no zone = UTC); open segments run to it (or to now)')
    ap.add_argument('--group', default='batch', help=f'comma list of {"|".join(GROUPS)}')
    ap.add_argument('--floors', action='append', default=[],
                    help='saved `condor_status -af UtsnameNodename MinRunningPrice CUDACapability`; repeat for '
                         'older snapshots (the first that has a host wins)')
    ap.add_argument('--live', action='store_true', help='live condor_status floors first, then --floors files')
    ap.add_argument('--save-floors', help='write the live condor_status floors snapshot here')
    ap.add_argument('--queue', help='saved `condor_q -af ClusterId ProcId JobStatus EnteredCurrentStatus` to end '
                                    'open segments with (default: a live, read-only condor_q)')
    ap.add_argument('--no-queue', action='store_true', help='no condor_q: open segments end at stale output only')
    ap.add_argument('--no-fixup', action='store_true', help='keep open segments open and late 009s as logged')
    ap.add_argument('--stale-after', type=float, default=STALE_AFTER, help='seconds (default %(default)s)')
    ap.add_argument('--tz', default='cluster',
                    help=f'zone of event-log timestamps: cluster ({CLUSTER_TZ}, default), local, UTC, +01:00, IANA')
    ap.add_argument('--no-submit', action='store_true', help='do not read jobs.json / submit files (batch, bid)')
    ap.add_argument('--json', help='write groups, total and costed segments here')
    a = ap.parse_args(argv)
    for n in a.group.split(','):
        if n.strip() not in GROUPS:
            ap.error(f'--group {n!r}: one of {", ".join(GROUPS)}')
    try:
        tz = parse_tz(a.tz)
    except ValueError as e:
        ap.error(str(e))
    live = a.live or not a.floors
    if a.save_floors and not live:
        ap.error('--save-floors saves the live snapshot: add --live (or drop --floors)')
    floors, fsrc = {}, []
    if live:
        try:
            text = status_floors_text(runner)
            floors, fsrc = parse_floors(text), ['condor_status (live)']
            if a.save_floors:
                Path(a.save_floors).write_text(text)
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            print(f'WARNING: live floors failed ({e!r})', file=err)
    for h, v in load_floors(*a.floors).items():
        floors.setdefault(h, v)
    fsrc += a.floors
    logs = expand(a.logs)
    if not logs:
        print('no event logs match', file=err)
        return 1
    since, until, now = parse_time(a.since), parse_time(a.until), time.time()
    segs = load(logs, tz, with_submit=not a.no_submit, warn=lambda s: print(s, file=err))
    if not a.no_fixup:
        queue = None
        if a.queue:
            queue = parse_queue(Path(a.queue).read_text())
        elif not a.no_queue:
            open_cl = {s['cluster'] for s in segs if s['end'] is None}
            try:
                queue = parse_queue(condor_q_text(open_cl, runner)) if open_cl else {}
            except (OSError, RuntimeError, subprocess.SubprocessError) as e:
                print(f'WARNING: condor_q failed ({e!r}); open segments end at stale output only', file=err)
        st = close_stale(segs, queue, now, a.stale_after)
        if st['open_queue'] or st['open_output'] or st['aborted_late'] or st['aborted_kept']:
            print(f"NOTE: ended {st['open_queue']} open segment(s) at their queue EnteredCurrentStatus and "
                  f"{st['open_output']} at their last activity (output write / shadow event); bare 009s (end event "
                  f"lost): capped {st['aborted_late']} at their last activity (-{st['aborted_late_h']:,.1f} job-h), "
                  f"kept {st['aborted_kept']} at the 009 time", file=err)
    rows = costed(segs, floors, since, until, now)
    if not rows:
        print('no run time inside the window', file=out)
        return 0
    groups, total = summarize(rows, group_key(a.group)), summarize(rows)['TOTAL']
    if total['open']:
        print(f"WARNING: {total['open']} segment(s) are still open and charged to {a.until or 'now'}: "
              f"{total['open_h']:,.1f} job-h ({_pct(total['open_h'] / total['job_h'])} of job-h), "
              f"{total['open_credits_floor']:,.0f} credits at floor "
              f"({_pct(total['open_credits_floor'] / total['credits_floor']) if total['credits_floor'] else '-'})",
              file=err)
    if total['unknown_floor_h']:
        print(f"WARNING: {total['unknown_floor_h']:,.1f} job-h on hosts in no floors snapshot are not charged at "
              f"floor; pass an older snapshot with --floors", file=err)
    win = f"{a.since or 'start'} .. {a.until or 'now'} (UTC)"
    print(f"{len(logs)} logs, {len(rows)} segments, window {win}, floors: {', '.join(fsrc) or 'none'}", file=out)
    print(table(groups, total, a.group), file=out)
    if a.json:
        Path(a.json).write_text(json.dumps({'logs': logs, 'since': since, 'until': until, 'now': now,
                                            'floors': fsrc, 'group': a.group, 'groups': groups, 'total': total,
                                            'segments': rows}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

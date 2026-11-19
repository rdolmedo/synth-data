"""Per-minute cgroup resource log of a condor job (#378): the memory the job's cgroup really holds (rss + shmem +
tmpfs + page cache) and its CPU time. Holds come from rss + shmem + dirty pages: clean page cache alone filled a job
to 100% of its limit without a hold (optim-servers, 2026-11-14), so `dirty` and `writeback` are logged too. Size job
RAM from these peaks, not from condor's RSS-only MemoryUsage (RSS sizing got every c30 Pi worker held on 2026-11-11).

CPU contention (1.5). Condor jobs here are limited by a cpuset (affinity mask), not by a CFS quota (`cpu_quota_us`:
v1 cpu.cfs_quota_us, v2 the first field of cpu.max; -1 = none), so the CFS throttle counters stay 0 and say nothing.
What a cpuset costs is run-queue wait, read per thread from /proc/<tid>/schedstat (field 1: ns on a CPU, field 2: ns
waiting on a run queue) for the threads of the job's cgroup and its descendants (v1 `tasks` of the cpu controller,
v2 `cgroup.threads`):
- `cpu_runq_wait_ms`, `cpu_run_ms`: the summed deltas since the previous sample over the tids read in both samples
  (wait / run is the fraction of extra time spent queueing). Threads that start or exit within the interval drop
  out, and a negative delta (a reused tid) is skipped, so both are lower bounds. None on the first sample.
  Short-lived processes mostly drop out (compare cpu_run_ms with the cpu_sec delta for coverage), and wait includes
  wakeup latency on idle cores too: compare it across load, not as an absolute delay.
- `cpu_runq_threads`: the job's tid count; `cpu_runq_sampled`: how many were read. Above RUNQ_MAX_TIDS (5,000) a
  deterministic subset is read (a fixed hash of the tid under a threshold, so the subsets of two samples nest);
  `cpu_runq_matched`: tids in both samples.
Every field is None when its files are missing or unreadable; a field never fails the sample.

One sampler for every pipeline (#445 item 2, #480): the fields above, plus
- `hard`: bytes the kernel cannot reclaim (rss + shmem + dirty + writeback, hierarchical where v1 has total_*): size
  RAM from its peak; `mem` includes clean page cache and reads near the limit on a busy node (serving, #378);
- `failcnt`, `oom` (v1 memory.oom_control / v2 memory.events), `ncpu` (the job's cpuset size);
- `cores`: CPU cores used since the previous sample (eval_q35 / synth.telemetry);
- `shm_used`, `scratch_used`: bytes used on /dev/shm and on the job's scratch filesystem (_CONDOR_SCRATCH_DIR), and
  `<name>_used` for every NAME=DIR given;
- `procs` (with --procs, every PROC_EVERY samples): processes over 0.5 GB RSS in the job's cgroup tree, with their
  command lines;
- `mem_stat`: aa-serving's normalized memory.stat (cache, rss, rss_huge, shmem, mapped_file, dirty, writeback,
  unevictable, active/inactive anon/file, swap, pgmajfault; v1 hierarchical totals preferred, v2 names mapped);
  `stat` keeps tbshared's raw keys (plus their total_* forms and the extra counters);
- `fmt`: 2; the first sample also has `cgroup` (the memory / cpu directories actually read) and `env`
  (OMP_NUM_THREADS, scratch, job ids).
Memory and CPU are read from the job's own cgroup (job_dir): the same as before for batch jobs; for an interactive
job's shell child (".../sshd") it is now the job, as aa-serving's sampler did.
aa-serving's serving sampler (before #445 item 2) wrote `limit` (int), `load1` (float), `mem`/`mem_peak` (int) and
the normalized `stat`; here they are `mem_limit`, `loadavg`, `mem`, `mem_peak` (strings, as tbshared always wrote)
and `mem_stat`. `summary` reads all three formats.

  python -I tbshared/reslog.py OUT.jsonl [INTERVAL_SEC]       # one JSON line per sample; exits within 1 s of its parent
  python -I tbshared/reslog.py run OUT.jsonl [INTERVAL_SEC] [--parent PID] [--procs] [NAME=DIR ...]
  python -I tbshared/reslog.py summary OUT.jsonl [--healthy-at T]   # peaks: hard / mem / cores, start-up vs serving
  reslog.start(path, interval=60, extra=None)                 # in-process: a daemon thread (extra(): more fields)

Run the CLI in the background of a job script that does NOT `exec` its main process, and stop and `wait` it before the
script exits (aa-evals #401: a sampler outliving the main process got every worker held at a clean exit). --parent
names the pid that must stay its parent (a Python launcher passes its own; one that died before the sampler started
is noticed too).

Stdlib only; one file per job (Lustre has no shared appends).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


FMT = 2
PROC_EVERY = 6          # --procs: per-process RSS every 6th sample (~1 min at 10 s)
HARD = ('rss', 'shmem', 'dirty', 'writeback')
_V2_NAMES = (('anon', 'rss'), ('file', 'cache'), ('file_dirty', 'dirty'), ('file_writeback', 'writeback'))


MEM_STAT_KEYS = ('cache', 'rss', 'rss_huge', 'shmem', 'mapped_file', 'dirty', 'writeback', 'unevictable', 'active_anon',
                 'inactive_anon', 'active_file', 'inactive_file', 'swap', 'pgmajfault')


def mem_stat(stat: dict) -> dict:
    """aa-serving's normalized memory.stat: v1 hierarchical total_* preferred, v2 names mapped (anon -> rss, file ->
    cache, file_dirty -> dirty, file_writeback -> writeback, file_mapped -> mapped_file)."""
    out = {}
    for k in MEM_STAT_KEYS:
        v = stat.get('total_' + k, stat.get(k))
        if v is not None:
            out[k] = v
    for v2, k in _V2_NAMES + (('file_mapped', 'mapped_file'),):
        if k not in out and v2 in stat:
            out[k] = stat[v2]
    return out


def hard_bytes(stat: dict) -> int | None:
    """Non-reclaimable bytes from a sample's `stat`: v1 hierarchical total_* where present, else the plain or v2 names
    (anon, file_dirty, file_writeback)."""
    if not stat:
        return None
    norm = {k: stat.get('total_' + k, stat.get(k)) for k in HARD}
    for v2, k in _V2_NAMES:
        if k in norm and norm[k] is None and v2 in stat:
            norm[k] = stat[v2]
    return sum(v or 0 for v in norm.values())


def _int(v):
    return int(v) if isinstance(v, str) and v.isdigit() else None


def sample(proc=Path('/proc'), cgroups=Path('/sys/fs/cgroup'), state: dict | None = None,
           extra_dirs=()) -> dict:
    """One sample. `state` (a dict the caller keeps between samples; main() does) carries the per-tid schedstat
    counters for the run-queue deltas, the previous CPU time for `cores` and the sample count; without it they are
    None. `extra_dirs`: (name, dir) pairs whose filesystem's used bytes go to `<name>_used`."""
    st = {} if state is None else state
    out = {'t': round(time.time(), 1), 'fmt': FMT, 'loadavg': (proc / 'loadavg').read_text().split()[0]}
    paths = {}
    for line in (proc / 'self/cgroup').read_text().splitlines():
        _, ctrls, path = line.split(':', 2)
        for c in ctrls.split(','):
            paths[c] = path

    def read(p):
        try:
            return p.read_text().strip()
        except OSError:
            return None
    base = None
    if 'memory' in paths:                                   # cgroup v1
        base = job_dir(cgroups / 'memory', paths['memory'], 'memory.limit_in_bytes', read)
        out['mem'] = read(base / 'memory.usage_in_bytes')
        out['mem_peak'] = read(base / 'memory.max_usage_in_bytes')
        out['mem_limit'] = read(base / 'memory.limit_in_bytes')
        out['failcnt'] = _int(read(base / 'memory.failcnt'))
        events = read(base / 'memory.oom_control') or ''
        stat = read(base / 'memory.stat') or ''
    elif '' in paths:                                       # cgroup v2
        base = job_dir(cgroups, paths[''], 'memory.max', read)
        out['mem'] = read(base / 'memory.current')
        out['mem_peak'] = read(base / 'memory.peak')
        out['mem_limit'] = read(base / 'memory.max')
        out['failcnt'] = None
        events = read(base / 'memory.events') or ''
        stat = read(base / 'memory.stat') or ''
    else:
        stat = events = ''
    keep = ('rss', 'shmem', 'cache', 'mapped_file', 'anon', 'file', 'dirty', 'writeback', 'file_dirty', 'file_writeback',
            'total_rss', 'total_shmem', 'total_cache', 'total_dirty', 'total_writeback', 'total_mapped_file',
            'file_mapped', 'rss_huge', 'total_rss_huge', 'unevictable', 'total_unevictable', 'swap', 'total_swap',
            'pgmajfault', 'total_pgmajfault', 'active_anon', 'inactive_anon', 'active_file', 'inactive_file',
            'total_active_anon', 'total_inactive_anon', 'total_active_file', 'total_inactive_file')
    out['stat'] = {k: int(v) for k, v in (ln.split() for ln in stat.splitlines() if len(ln.split()) == 2)
                   if k in keep and v.lstrip('-').isdigit()}
    out['mem_stat'] = mem_stat(out['stat'])
    out['hard'] = hard_bytes(out['stat'])
    out['oom'] = {k: int(v) for k, v in (ln.split() for ln in events.splitlines() if len(ln.split()) == 2)
                  if v.isdigit()} if events else None
    out['cpu_quota_us'] = cpu_quota(paths, cgroups, read)
    out.update(runq(paths, cgroups, proc, st))
    cpu = None
    own = None
    if 'cpuacct' in paths:
        own = cgroups / 'cpuacct' / paths['cpuacct'].lstrip('/')
        if base is not None and 'memory' in paths:          # the job's own cgroup, as for memory (job_dir)
            same = cgroups / 'cpuacct' / base.relative_to(cgroups / 'memory')
            own = same if (same / 'cpuacct.usage').exists() else own
        cpu = read(own / 'cpuacct.usage')
        out['cpu_sec'] = round(int(cpu) / 1e9, 1) if cpu and cpu.isdigit() else None
    elif '' in paths:
        own = base if base is not None else cgroups / paths[''].lstrip('/')   # v2: the job's own cgroup (job_dir)
        cs = read(own / 'cpu.stat') or ''
        us = dict(ln.split() for ln in cs.splitlines() if len(ln.split()) == 2).get('usage_usec')
        out['cpu_sec'] = round(int(us) / 1e6, 1) if us and us.isdigit() else None
    st['dirs'] = {'memory': str(base) if base is not None else None, 'cpu': str(own) if own is not None else None}
    prev = st.get('cpu')                                    # (t, cpu_sec) of the previous sample
    out['cores'] = (round((out['cpu_sec'] - prev[1]) / (out['t'] - prev[0]), 3)
                    if prev and out.get('cpu_sec') is not None and out['t'] > prev[0] else None)
    if out.get('cpu_sec') is not None:
        st['cpu'] = (out['t'], out['cpu_sec'])
    try:
        out['ncpu'] = len(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        out['ncpu'] = None
    dirs = [('shm', '/dev/shm')] + ([('scratch', os.environ['_CONDOR_SCRATCH_DIR'])]
                                    if os.environ.get('_CONDOR_SCRATCH_DIR') else []) + list(extra_dirs)
    for name, d in dirs:
        try:
            v = os.statvfs(d)
            out[f'{name}_used'] = (v.f_blocks - v.f_bfree) * v.f_frsize
        except OSError:
            out[f'{name}_used'] = None
    n = st.get('n', 0)
    if st.get('procs') and base is not None and n % PROC_EVERY == 0:
        out['procs'] = processes(base, proc, read)
    st['n'] = n + 1
    return out


def job_dir(root: Path, rel: str, limit_file: str, read) -> Path:
    """The job's own cgroup under `root`: the topmost ancestor of ours (ours included) that still carries a limit. A
    batch job's processes sit in the job's cgroup; an interactive job's shell sits in a child ('.../sshd') with the same
    limit, and the condor root above the job is unlimited (aa-serving's serving sampler, #378, merged here)."""
    d = here = root / rel.lstrip('/')
    found = None
    while d != root and root in d.parents:
        v = read(d / limit_file)
        if v and v.isdigit() and int(v) < 1 << 60:
            found = d
        elif found is not None:
            break
        d = d.parent
    return found or here


def processes(mem_dir: Path, proc: Path, read) -> list:
    """Processes of the job's memory cgroup over 0.5 GB RSS: pid, rss_gb, the first 120 characters of the command."""
    rows = []
    try:
        pids = set()
        for d, _, files in os.walk(mem_dir):                # the job's cgroup and its descendants (interactive shells)
            if 'cgroup.procs' in files:
                pids |= {int(x) for x in (read(Path(d) / 'cgroup.procs') or '').split() if x.isdigit()}
        pids = sorted(pids)
        for pid in pids:
            status = read(proc / str(pid) / 'status') or ''
            rss = next((int(ln.split()[1]) for ln in status.splitlines() if ln.startswith('VmRSS:')), 0)
            if rss < 512 * 1024:                            # KB
                continue
            cmd = (read(proc / str(pid) / 'cmdline') or '').replace('\0', ' ')[:120]
            rows.append({'pid': pid, 'rss_gb': round(rss / 2 ** 20, 2), 'cmd': cmd})
    except Exception:                                       # never fail the sample
        return rows
    return rows


RUNQ_MAX_TIDS = 5000
RUNQ_FIELDS = ('cpu_runq_wait_ms', 'cpu_run_ms', 'cpu_runq_threads', 'cpu_runq_sampled', 'cpu_runq_matched')


def _cpu_dir(paths: dict, cgroups: Path):
    """(the job's cpu cgroup directory, its tid-list file name), or (None, None)."""
    for c in ('cpu', 'cpuacct'):                            # cgroup v1 (co-mounted here: cpu,cpuacct)
        if c in paths:
            return cgroups / c / paths[c].lstrip('/'), 'tasks'
    if '' in paths:                                         # cgroup v2
        return cgroups / paths[''].lstrip('/'), 'cgroup.threads'
    return None, None


def cpu_quota(paths: dict, cgroups: Path, read):
    """The cgroup's CFS quota in us per period (-1 = none), None when unreadable. Never raises."""
    try:
        base, tidfile = _cpu_dir(paths, cgroups)
        if base is None:
            return None
        if tidfile == 'tasks':
            q = read(base / 'cpu.cfs_quota_us')
            return int(q) if q and q.lstrip('-').isdigit() else None
        q = (read(base / 'cpu.max') or '').split()
        return -1 if q[:1] == ['max'] else int(q[0]) if q and q[0].isdigit() else None
    except Exception:
        return None


def _keep(tid: int, n: int) -> bool:
    """Deterministic subset of about RUNQ_MAX_TIDS of n tids: a fixed hash under a threshold that falls as n grows,
    so the subsets of two samples nest and their common tids stay comparable."""
    return n <= RUNQ_MAX_TIDS or (tid * 2654435761) % 2 ** 32 < 2 ** 32 * RUNQ_MAX_TIDS // n


def runq(paths: dict, cgroups: Path, proc: Path, state: dict) -> dict:
    """Run-queue wait and run time of the job's threads since the previous call (module docstring). `state['runq']`
    keeps {tid: (run_ns, wait_ns)} of this call for the next. Never raises; unreadable gives None."""
    out = dict.fromkeys(RUNQ_FIELDS)
    try:
        base, tidfile = _cpu_dir(paths, cgroups)
        if base is None or not base.is_dir():
            state.pop('runq', None)
            return out
        tids = set()
        for d, _, files in os.walk(base):                   # the job's cgroup and its descendants
            if tidfile in files:
                try:
                    tids.update(int(x) for x in (Path(d) / tidfile).read_text().split() if x.isdigit())
                except OSError:
                    pass
        n = len(tids)
        cur = {}
        for tid in sorted(tids):
            if not _keep(tid, n):
                continue
            try:
                f = (proc / str(tid) / 'schedstat').read_text().split()
                cur[tid] = (int(f[0]), int(f[1]))
            except (OSError, ValueError, IndexError):       # exited since the tid list was read
                pass
        out['cpu_runq_threads'], out['cpu_runq_sampled'] = n, len(cur)
        prev = state.get('runq')
        if prev is not None:
            run = wait = matched = 0
            for tid, (r, w) in cur.items():
                if tid in prev and r >= prev[tid][0] and w >= prev[tid][1]:
                    run, wait, matched = run + r - prev[tid][0], wait + w - prev[tid][1], matched + 1
            out['cpu_run_ms'], out['cpu_runq_wait_ms'] = round(run / 1e6, 1), round(wait / 1e6, 1)
            out['cpu_runq_matched'] = matched
        state['runq'] = cur
    except Exception:                                       # a sampler field must never fail the sample
        state.pop('runq', None)
        return dict.fromkeys(RUNQ_FIELDS)
    return out


def log(path, interval=60, parent: int | None = None, extra_dirs=(), procs=False, extra=None, stop=None) -> None:
    """Append a sample to `path` every `interval` s until `parent` is no longer our parent (checked every second) or
    `stop` (a threading.Event) is set; with neither, forever (a daemon thread). `extra()`: more fields for each sample
    (best effort). A failed sample or write (e.g. a transient /data error) skips that sample, never the sampler."""
    path = Path(path)
    if parent is not None and os.getppid() != parent:
        return
    state = {'procs': procs}
    first, made = True, False
    while stop is None or not stop.is_set():
        try:
            if not made:                                 # once, until it succeeds (a /data metadata call per sample)
                path.parent.mkdir(parents=True, exist_ok=True)
                made = True
            row = sample(state=state, extra_dirs=extra_dirs)
            if extra is not None:
                try:
                    row.update(extra())
                except Exception as e:  # noqa: BLE001
                    row['extra_error'] = repr(e)[:200]
            if first:
                row['cgroup'] = state.get('dirs') or {}
                row['env'] = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', '_CONDOR_SCRATCH_DIR', 'SERVE_JOB_ID',
                                                             'SYNTH_JOB_ID', 'CONDOR_JOB_ID')}
                first = False
            with open(path, 'a') as f:
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
        except Exception as e:   # never take the job down
            print(f'reslog: {e!r}', file=sys.stderr, flush=True)
        if parent is None:
            if stop is not None:
                stop.wait(interval)
            else:
                time.sleep(interval)
            continue
        for _ in range(max(1, int(interval))):   # check the parent every second: never outlive the job's worker
            if os.getppid() != parent:
                return
            time.sleep(min(1.0, interval))


def start(path, interval=60, extra=None, extra_dirs=(), procs=False):
    """The sampler in-process, as a daemon thread (dies with the process): for long-lived Python jobs that sample
    themselves (synth.telemetry, eval_q35 workers). -> the thread; `thread.stop.set()` ends it after its current
    sample."""
    import threading
    stop = threading.Event()
    t = threading.Thread(target=log, args=(path, interval, None, extra_dirs, procs, extra, stop), name='reslog',
                         daemon=True)
    t.stop = stop
    t.start()
    return t


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def summary(rows: list, healthy_at: float | None = None) -> dict:
    """Peaks of one job's samples, split into start-up (before healthy_at) and serving: memory (`mem`, and `hard` =
    non-reclaimable), per-interval cores (intervals over 3x the median sampling interval, i.e. gaps, are left out),
    /dev/shm. Reads this format (fmt 2), the older tbshared one, and aa-serving's serving sampler's (limit, load1,
    normalized stat)."""
    rows = [r for r in rows if 'stat' in r]
    for r in rows:
        r['_ms'] = r.get('mem_stat') or mem_stat(r['stat'])
        if r.get('hard') is None:
            r['hard'] = hard_bytes(r['stat']) or 0
    gib = lambda b: round(b / 2 ** 30, 1) if b is not None else None   # noqa: E731
    last = rows[-1] if rows else {}
    out = {'samples': len(rows), 'from': rows[0]['t'] if rows else None, 'to': last.get('t'),
           'limit_gib': gib(_num(last.get('mem_limit', last.get('limit')))), 'ncpu': last.get('ncpu'),
           'mem_peak_gib': gib(max((_num(r.get('mem_peak')) or 0) for r in rows)) if rows else None,
           'failcnt': last.get('failcnt'),
           'under_oom_seen': any((r.get('oom') or {}).get('under_oom') for r in rows)}
    steps = sorted(b['t'] - a['t'] for a, b in zip(rows, rows[1:]) if b['t'] > a['t'])
    gap = 3 * steps[len(steps) // 2] if steps else None
    for phase, sel in (('startup', lambda r: healthy_at is not None and r['t'] < healthy_at),
                       ('serving', lambda r: healthy_at is None or r['t'] >= healthy_at)):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        cores = [(b['cpu_sec'] - a['cpu_sec']) / (b['t'] - a['t']) for a, b in zip(rs, rs[1:])
                 if a.get('cpu_sec') is not None and b.get('cpu_sec') is not None and 0 < b['t'] - a['t'] <= gap]
        top = max(rs, key=lambda r: r['hard'])
        out[phase] = {
            'minutes': round((rs[-1]['t'] - rs[0]['t']) / 60, 1),
            'mem_max_gib': gib(max(_num(r.get('mem')) or 0 for r in rs)),
            'hard_max_gib': gib(top['hard']),
            'at_hard_max': {k: gib(top['_ms'].get(k)) for k in ('rss', 'shmem', 'cache', 'mapped_file', 'dirty',
                                                                 'writeback', 'unevictable')},
            'dirty_wb_max_gib': gib(max(r['_ms'].get('dirty', 0) + r['_ms'].get('writeback', 0) for r in rs)),
            'shm_used_max_gib': gib(max(r.get('shm_used') or 0 for r in rs)),
            'cores_mean': round(sum(cores) / len(cores), 2) if cores else None,
            **{f'cores_p{int(q * 100)}': round(_pct(cores, q), 2) if cores else None for q in (0.5, 0.95, 0.99)},
            'cores_max': round(max(cores), 2) if cores else None,
        }
    for r in rows:
        r.pop('_ms', None)
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'summary':
        healthy = float(argv[argv.index('--healthy-at') + 1]) if '--healthy-at' in argv else None
        rows = [json.loads(ln) for ln in open(argv[1]) if ln.strip()]
        print(json.dumps(summary(rows, healthy), indent=1))
        return
    if argv and argv[0] == 'run':
        argv = argv[1:]
    parent = os.getppid()
    if '--parent' in argv:
        i = argv.index('--parent')
        parent = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    procs = '--procs' in argv
    argv = [x for x in argv if x != '--procs']
    path, rest = argv[0], argv[1:]                       # the output path may contain '='; NAME=DIR only after it
    extra_dirs = tuple(tuple(x.split('=', 1)) for x in rest if '=' in x)   # NAME=DIR: that fs's used bytes
    plain = [x for x in rest if '=' not in x]
    log(path, float(plain[0]) if plain else 60, parent, extra_dirs, procs)


if __name__ == '__main__':
    main()

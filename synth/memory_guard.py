"""Worker-level memory guard: keep a worker job's attempts inside the job's memory.

harbor_ns caps each sandbox at its task's memory_mb (a PSS guard), but ~48 attempts x 4 GB can exceed the job's request,
and tmpfs pages (a sandbox's /dev/shm) are invisible to a PSS guard. When the job reaches its limit, HTCondor holds it
and every attempt on it is lost. m1-hero (#244): 17841573 on gpu165 (anon to 94.7 GB) and 17841566 on gpu150 (/dev/shm
tmpfs from 37 to 89 GB in ~1 min, 5 GB RSS) were held, each cutting ~20 attempts.

The guard reads the job's cgroup every `poll_sec` and counts what cannot be reclaimed: anonymous memory plus shmem/tmpfs
(clean page cache is reclaimed first and never causes the hold). Above `admit_fraction` of the job's memory the worker
starts no new attempt; above `kill_fraction` the attempt holding the most memory (its processes' RSS plus its
sandboxes' /dev/shm) is stopped: its sandbox processes when they hold most of it (freeing their tmpfs), otherwise its
whole process tree. One attempt fails (WorkerMemoryGuard) instead of all of them.
"""
import os
from pathlib import Path
import signal
import time

PAGE = os.sysconf('SC_PAGE_SIZE')


def parse_stat(text):
    out = {}
    for line in (text or '').splitlines():
        key, _, value = line.partition(' ')
        if value.strip().lstrip('-').isdigit():
            out[key] = int(value)
    return out


def job_request_bytes(env=None):
    """RequestMemory (MiB) from the HTCondor job ad (_CONDOR_JOB_AD), in bytes; None if unknown."""
    env = os.environ if env is None else env
    path = env.get('_CONDOR_JOB_AD')
    if not path:
        return None
    try:
        for line in Path(path).read_text(errors='replace').splitlines():
            key, _, value = line.partition('=')
            if key.strip() == 'RequestMemory' and value.strip().isdigit():
                return int(value.strip()) * 2 ** 20
    except OSError:
        return None
    return None


def cgroup_memory(proc=Path('/proc'), cgroups=Path('/sys/fs/cgroup')):
    """{'limit', 'usage', 'unreclaimable', 'anon', 'shmem'} in bytes for this process's memory cgroup (v1 or v2)."""
    memberships = {}
    for line in (proc / 'self/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            memberships[controller] = path
    if 'memory' in memberships:   # v1
        base = cgroups / 'memory' / memberships['memory'].lstrip('/')
        stat = parse_stat((base / 'memory.stat').read_text())
        limit = int((base / 'memory.limit_in_bytes').read_text())
        usage = int((base / 'memory.usage_in_bytes').read_text())
        anon, shmem = stat.get('total_rss', stat.get('rss', 0)), stat.get('total_shmem', stat.get('shmem', 0))
    else:                         # v2
        base = cgroups / memberships.get('', '/').lstrip('/')
        stat = parse_stat((base / 'memory.stat').read_text())
        raw = (base / 'memory.max').read_text().strip()
        limit = None if raw == 'max' else int(raw)
        usage = int((base / 'memory.current').read_text())
        anon, shmem = stat.get('anon', 0), stat.get('shmem', 0)
    if limit is not None and limit >= 2 ** 60:   # v1 "unlimited"
        limit = None
    return {'limit': limit, 'usage': usage, 'anon': anon, 'shmem': shmem, 'unreclaimable': anon + shmem}


def process_table(proc=Path('/proc')):
    """pid -> {'ppid', 'rss', 'mnt'} for every process readable here."""
    table = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / 'stat').read_text()
            fields = stat[stat.rindex(')') + 2:].split()
            rss = int((path / 'statm').read_text().split()[1]) * PAGE
            mnt = os.readlink(path / 'ns' / 'mnt')
        except (OSError, ValueError, IndexError):
            continue  # exited, or not ours
        table[int(path.name)] = {'ppid': int(fields[1]), 'rss': rss, 'mnt': mnt}
    return table


def descendants(table, root):
    children = {}
    for pid, row in table.items():
        children.setdefault(row['ppid'], []).append(pid)
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid in table:
            out.append(pid)
        stack.extend(children.get(pid, ()))
    return out


def shm_bytes(pid, proc=Path('/proc')):
    """Used bytes of the /dev/shm that process `pid` sees (its sandbox's tmpfs)."""
    try:
        st = os.statvfs(proc / str(pid) / 'root' / 'dev' / 'shm')
    except OSError:
        return 0
    return (st.f_blocks - st.f_bfree) * st.f_frsize


class MemoryGuard:
    def __init__(self, kill_fraction=0.85, admit_fraction=0.75, poll_sec=2.0, cooldown_sec=15.0, budget=None,
                 read_memory=cgroup_memory, read_table=process_table, read_shm=shm_bytes, kill=os.kill,
                 own_mnt=None, log=None, clock=time.monotonic):
        self.kill_fraction, self.admit_fraction = kill_fraction, admit_fraction
        self.poll_sec, self.cooldown_sec = poll_sec, cooldown_sec
        self.fixed_budget = budget
        self.read_memory, self.read_table, self.read_shm, self.kill = read_memory, read_table, read_shm, kill
        self.own_mnt = own_mnt if own_mnt is not None else _own_mnt()
        self.log = log or (lambda event: None)
        self.clock = clock
        self.last_kill = float('-inf')
        self.last = None
        self.kills = []
        self.request = job_request_bytes()

    def budget(self, memory):
        """The job's memory: the smaller of the cgroup limit and the job's RequestMemory."""
        values = [v for v in (self.fixed_budget, memory.get('limit'), self.request) if v]
        return min(values) if values else None

    def sample(self):
        memory = self.read_memory()
        memory['budget'] = self.budget(memory)
        self.last = memory
        return memory

    def admits(self, memory=None):
        """May the worker start another attempt? (no budget known: yes)"""
        memory = memory or self.last or self.sample()
        return not memory['budget'] or memory['unreclaimable'] < self.admit_fraction * memory['budget']

    def attempt_usage(self, attempts, table):
        """uid -> {'pids', 'sandbox_pids', 'rss', 'sandbox_rss', 'shm', 'total'} for attempts {uid: root pid}."""
        out = {}
        for uid, root in attempts.items():
            if not root:
                continue
            pids = descendants(table, root)
            sandbox = [p for p in pids if table[p]['mnt'] != self.own_mnt]
            shm, seen = 0, set()
            for pid in sandbox:
                if table[pid]['mnt'] not in seen:
                    seen.add(table[pid]['mnt'])
                    shm += self.read_shm(pid)
            rss = sum(table[p]['rss'] for p in pids)
            sandbox_rss = sum(table[p]['rss'] for p in sandbox)
            out[uid] = {'pids': pids, 'sandbox_pids': sandbox, 'rss': rss, 'sandbox_rss': sandbox_rss, 'shm': shm,
                        'total': rss + shm}
        return out

    def check(self, attempts):
        """One guard step over attempts {uid: root pid}; returns the stopped attempt's record, or None."""
        memory = self.sample()
        budget = memory['budget']
        if not budget or memory['unreclaimable'] < self.kill_fraction * budget:
            return None
        if self.clock() - self.last_kill < self.cooldown_sec:
            return None   # the last kill's memory may still be being freed
        usage = self.attempt_usage(attempts, self.read_table())
        if not usage:
            return None
        uid = max(usage, key=lambda u: usage[u]['total'])
        victim = usage[uid]
        record = {'time': time.time(), 'attempt': uid, 'budget_bytes': budget, 'unreclaimable_bytes': memory['unreclaimable'],
                  'anon_bytes': memory['anon'], 'shmem_bytes': memory['shmem'], 'attempt_rss_bytes': victim['rss'],
                  'attempt_shm_bytes': victim['shm'], 'attempts': len(usage)}
        if victim['total'] < 0.05 * budget:
            record['action'] = 'none: no attempt holds 5% of the job memory'
            self.last_kill = self.clock()
            self.log(record)
            return None
        in_sandbox = victim['sandbox_rss'] + victim['shm']
        targets = victim['sandbox_pids'] if victim['sandbox_pids'] and in_sandbox >= 0.5 * victim['total'] else victim['pids']
        record['action'] = ('SIGKILL the attempt\'s sandbox processes' if targets is victim['sandbox_pids']
                            else 'SIGKILL the attempt\'s process tree')
        record['killed_pids'] = len(targets)
        for pid in targets:
            try:
                self.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.last_kill = self.clock()
        self.kills.append(record)
        self.log(record)
        return record


def _own_mnt():
    try:
        return os.readlink('/proc/self/ns/mnt')
    except OSError:
        return None

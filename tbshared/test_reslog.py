"""tbshared.reslog: one cgroup sample per line, and the CLI never outlives its parent (#401)."""
import json
import subprocess
import sys
import time
from pathlib import Path

from tbshared import reslog

HERE = Path(__file__).resolve().parent


def test_sample_has_memory_and_dirty_fields():
    s = reslog.sample()
    assert {'t', 'loadavg', 'stat'} <= set(s)
    assert all(isinstance(v, int) for v in s['stat'].values())


def test_cli_exits_with_its_parent(tmp_path):
    log, pidf = tmp_path / 'r.jsonl', tmp_path / 'pid'
    script = f'"{sys.executable}" -I "{HERE / "reslog.py"}" "{log}" 60 & echo $! > "{pidf}"; sleep 1.5'
    subprocess.run(['bash', '-c', script], check=True, timeout=30)
    pid = int(pidf.read_text())
    for _ in range(40):
        try:
            if Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(0.25)
    else:
        raise AssertionError('reslog outlived its parent')
    assert 'stat' in json.loads(log.read_text().splitlines()[0])


JOB = 'htcondor/condor_var_lib_condor_execute_slot1_10@gpu133'
V1_CGROUP = (f'9:memory:/{JOB}\n8:cpuset:/htcondor-ext/x\n2:cpu,cpuacct:/{JOB}\n'
             '1:name=systemd:/system.slice/condor.service\n0::/system.slice/condor.service\n')   # this cluster's layout
V2_JOB = 'system.slice/job.scope'


def fake(tmp_path, cgroup_lines, files):
    """A fake proc root and cgroup root (sample's `proc` and `cgroups` parameters)."""
    proc, cg = tmp_path / 'proc', tmp_path / 'cg'
    (proc / 'self').mkdir(parents=True)
    (proc / 'loadavg').write_text('1.00 1.00 1.00 1/100 1\n')
    (proc / 'self' / 'cgroup').write_text(cgroup_lines)
    for rel, text in files.items():
        f = cg / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return proc, cg


def threads(proc, cg, tidfile, stats, sub=None):
    """Set the job's threads: {tid: (run_ns, wait_ns)} with their /proc/<tid>/schedstat; `sub` = {child cgroup: tids}."""
    for d in [p for p in proc.iterdir() if p.name.isdigit()]:
        (d / 'schedstat').unlink()
        d.rmdir()
    for tid, (r, w) in stats.items():
        (proc / str(tid)).mkdir()
        (proc / str(tid) / 'schedstat').write_text(f'{r} {w} 7\n')
    sub = sub or {}
    in_sub = {t for ts in sub.values() for t in ts}
    tidfile.write_text(''.join(f'{t}\n' for t in stats if t not in in_sub))
    for name, ts in sub.items():
        (tidfile.parent / name).mkdir(exist_ok=True)
        (tidfile.parent / name / tidfile.name).write_text(''.join(f'{t}\n' for t in ts))


def test_runq_wait_v1_with_threads_appearing_and_exiting(tmp_path):
    proc, cg = fake(tmp_path, V1_CGROUP, {f'cpu/{JOB}/cpu.cfs_quota_us': '-1\n', f'cpu/{JOB}/tasks': ''})
    tasks, st = cg / 'cpu' / JOB / 'tasks', {}
    threads(proc, cg, tasks, {10: (1_000_000, 500_000), 11: (2_000_000, 0), 12: (0, 0)})
    s = reslog.sample(proc, cg, st)
    assert s['cpu_quota_us'] == -1 and (s['cpu_runq_threads'], s['cpu_runq_sampled']) == (3, 3)
    assert s['cpu_runq_wait_ms'] is None and s['cpu_run_ms'] is None and s['cpu_runq_matched'] is None   # first
    # 10 runs 4 ms and waits 3 ms; 11 exits; 13 appears (not counted until it is in two samples); 12 is in a child
    threads(proc, cg, tasks, {10: (5_000_000, 3_500_000), 12: (1_000_000, 2_000_000), 13: (9_000_000, 9_000_000)},
            sub={'sandbox0': [12]})
    s = reslog.sample(proc, cg, st)
    assert (s['cpu_runq_threads'], s['cpu_runq_sampled'], s['cpu_runq_matched']) == (3, 3, 2)
    assert (s['cpu_run_ms'], s['cpu_runq_wait_ms']) == (5.0, 5.0)
    threads(proc, cg, tasks, {10: (5_000_000, 3_500_000), 13: (10_000_000, 9_500_000)})
    s = reslog.sample(proc, cg, st)
    assert (s['cpu_run_ms'], s['cpu_runq_wait_ms'], s['cpu_runq_matched']) == (1.0, 0.5, 2)
    threads(proc, cg, tasks, {13: (1, 1)})                         # tid reused by a new thread: negative delta skipped
    s = reslog.sample(proc, cg, st)
    assert (s['cpu_run_ms'], s['cpu_runq_wait_ms'], s['cpu_runq_matched']) == (0.0, 0.0, 0)
    json.dumps(s)


def test_runq_wait_v2_and_quota(tmp_path):
    proc, cg = fake(tmp_path, f'0::/{V2_JOB}\n', {f'{V2_JOB}/cpu.max': '200000 100000\n',
                                                   f'{V2_JOB}/cgroup.threads': '',
                                                   f'{V2_JOB}/cpu.stat': 'usage_usec 3000000\n'})
    tf, st = cg / V2_JOB / 'cgroup.threads', {}
    threads(proc, cg, tf, {5: (0, 0)})
    assert reslog.sample(proc, cg, st)['cpu_quota_us'] == 200000
    threads(proc, cg, tf, {5: (2_000_000, 1_000_000)})
    s = reslog.sample(proc, cg, st)
    assert (s['cpu_run_ms'], s['cpu_runq_wait_ms'], s['cpu_runq_threads'], s['cpu_sec']) == (2.0, 1.0, 1, 3.0)
    (cg / V2_JOB / 'cpu.max').write_text('max 100000\n')
    assert reslog.sample(proc, cg, st)['cpu_quota_us'] == -1


def test_runq_reads_a_deterministic_nested_subset_above_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(reslog, 'RUNQ_MAX_TIDS', 50)
    proc, cg = fake(tmp_path, V1_CGROUP, {f'cpu/{JOB}/tasks': ''})
    tasks, st = cg / 'cpu' / JOB / 'tasks', {}
    threads(proc, cg, tasks, {t: (0, 0) for t in range(1000, 1400)})
    s = reslog.sample(proc, cg, st)
    assert s['cpu_runq_threads'] == 400 and 25 <= s['cpu_runq_sampled'] <= 75
    first = set(st['runq'])
    assert reslog.sample(proc, cg, {})['cpu_runq_sampled'] == s['cpu_runq_sampled']       # deterministic
    threads(proc, cg, tasks, {t: (1_000_000, 1_000_000) for t in range(1000, 1500)})    # more threads: smaller subset
    s = reslog.sample(proc, cg, st)
    assert set(st['runq']) <= first | set(range(1400, 1500))                            # nested in the first subset
    assert s['cpu_runq_threads'] == 500 and s['cpu_runq_matched'] == len(set(st['runq']) & first) > 0
    assert s['cpu_runq_wait_ms'] == s['cpu_runq_matched'] * 1.0


def test_cpu_fields_are_none_when_unreadable_and_never_fail(tmp_path):
    proc, cg = fake(tmp_path, V1_CGROUP, {f'memory/{JOB}/memory.usage_in_bytes': '1\n'})
    s = reslog.sample(proc, cg, {})                                 # no cpu cgroup dir at all
    assert s['cpu_quota_us'] is None and all(s[k] is None for k in reslog.RUNQ_FIELDS) and s['mem'] == '1'
    (cg / 'cpu' / JOB).mkdir(parents=True)
    (cg / 'cpu' / JOB / 'tasks').mkdir()                            # unreadable tid list (a directory)
    (cg / 'cpu' / JOB / 'cpu.cfs_quota_us').write_text('garbage\n')
    s = reslog.sample(proc, cg, {})
    assert s['cpu_quota_us'] is None and s['cpu_runq_threads'] == 0
    (cg / 'cpu' / JOB / 'tasks').rmdir()
    (cg / 'cpu' / JOB / 'tasks').write_text('77\nnot-a-tid\n')     # a tid without /proc/77, or with a bad schedstat
    assert reslog.sample(proc, cg, {})['cpu_runq_sampled'] == 0
    (proc / '77').mkdir()
    (proc / '77' / 'schedstat').write_text('x y\n')
    assert reslog.sample(proc, cg, {})['cpu_runq_sampled'] == 0
    proc2, cg2 = fake(tmp_path / 'none', '1:name=systemd:/x\n', {})
    assert all(reslog.sample(proc2, cg2)[k] is None for k in reslog.RUNQ_FIELDS)
    st = {'runq': 'corrupt'}
    assert reslog.runq({'cpu': 7}, cg, proc, st) == dict.fromkeys(reslog.RUNQ_FIELDS) and 'runq' not in st


def test_live_sample_has_the_cpu_fields():
    st = {}
    reslog.sample(state=st)
    s = reslog.sample(state=st)
    assert {'cpu_quota_us', *reslog.RUNQ_FIELDS} <= set(s)
    assert not {'cpu_nr_periods', 'cpu_nr_throttled', 'cpu_throttled_usec'} & set(s)
    assert isinstance(s['shm_used'], int)          # the real statvfs (the scratch test fakes it)


# --- one sampler for every pipeline (#445 item 2): serving's and eval_q35's fields in the same record ---------------
def test_hard_oom_and_failcnt_v1(tmp_path):
    stat = 'cache 500\nrss 10\ntotal_rss 100\ntotal_shmem 20\ntotal_cache 900\ntotal_dirty 3\ntotal_writeback 1\n'
    proc, cg = fake(tmp_path, V1_CGROUP, {f'memory/{JOB}/memory.usage_in_bytes': '1024\n',
                                          f'memory/{JOB}/memory.failcnt': '7\n',
                                          f'memory/{JOB}/memory.oom_control': 'oom_kill_disable 1\nunder_oom 0\n',
                                          f'memory/{JOB}/memory.stat': stat})
    s = reslog.sample(proc, cg, {})
    assert s['fmt'] == reslog.FMT and s['hard'] == 124                    # hierarchical total_*: 100 + 20 + 3 + 1
    assert s['failcnt'] == 7 and s['oom'] == {'oom_kill_disable': 1, 'under_oom': 0} and s['mem'] == '1024'


def test_hard_v2_names(tmp_path):
    proc, cg = fake(tmp_path, f'0::/{V2_JOB}\n', {f'{V2_JOB}/memory.current': '9\n',
                                                   f'{V2_JOB}/memory.events': 'oom 0\noom_kill 2\n',
                                                   f'{V2_JOB}/memory.stat': 'anon 50\nfile 70\nshmem 5\n'
                                                                            'file_dirty 2\nfile_writeback 1\n'})
    s = reslog.sample(proc, cg, {})
    assert s['hard'] == 58 and s['oom'] == {'oom': 0, 'oom_kill': 2} and s['failcnt'] is None


def test_cores_scratch_and_extra_dirs(tmp_path, monkeypatch):
    proc, cg = fake(tmp_path, f'0::/{V2_JOB}\n', {f'{V2_JOB}/cpu.stat': 'usage_usec 1000000\n'})
    # Filesystem usage through a fake statvfs: the real one reads a filesystem other tests write to in parallel
    # (test_all), so two reads differed by a few hundred KB.
    scratch, work = tmp_path / 'scratch', tmp_path / 'work'
    blocks = {str(scratch): (1000, 400), str(work): (1000, 900), '/dev/shm': (100, 100)}   # (f_blocks, f_bfree)

    class Vfs:
        def __init__(self, path):
            if str(path) not in blocks:
                raise OSError(2, 'no such filesystem', str(path))
            self.f_blocks, self.f_bfree = blocks[str(path)]
            self.f_frsize = 4096
    monkeypatch.setattr(reslog.os, 'statvfs', Vfs)
    monkeypatch.setenv('_CONDOR_SCRATCH_DIR', str(scratch))
    st = {}
    a = reslog.sample(proc, cg, st, extra_dirs=(('work', str(work)), ('gone', str(tmp_path / 'gone'))))
    assert a['cores'] is None and a['shm_used'] == 0
    assert (a['scratch_used'], a['work_used'], a['gone_used']) == (600 * 4096, 100 * 4096, None)
    (cg / V2_JOB / 'cpu.stat').write_text('usage_usec 3000000\n')
    st['cpu'] = (a['t'] - 4, a['cpu_sec'])                                 # 2 CPU-s over 4 s + the time between samples
    b = reslog.sample(proc, cg, st)
    assert 0 < b['cores'] <= 0.5 and abs(b['cores'] - 2 / (b['t'] - a['t'] + 4)) <= 0.001   # not timing-bound


def test_procs_only_when_asked_and_every_proc_every(tmp_path, monkeypatch):
    big = 'Name:\tvllm\nVmRSS:\t 2097152 kB\n'
    proc, cg = fake(tmp_path, V1_CGROUP, {f'memory/{JOB}/cgroup.procs': '41\n42\n'})
    for pid, status in ((41, big), (42, 'VmRSS:\t 1000 kB\n')):
        (proc / str(pid)).mkdir()
        (proc / str(pid) / 'status').write_text(status)
        (proc / str(pid) / 'cmdline').write_text('python\0-m\0vllm\0')
    assert 'procs' not in reslog.sample(proc, cg, {})
    st = {'procs': True}
    first = reslog.sample(proc, cg, st)
    assert first['procs'] == [{'pid': 41, 'rss_gb': 2.0, 'cmd': 'python -m vllm '}]
    assert all('procs' not in reslog.sample(proc, cg, st) for _ in range(reslog.PROC_EVERY - 1))
    assert 'procs' in reslog.sample(proc, cg, st)


def test_summary_reads_this_and_the_older_formats():
    gib = 2 ** 30
    new = [{'t': 100 + 10 * i, 'fmt': 2, 'mem': str(5 * gib), 'mem_peak': str(6 * gib), 'mem_limit': str(8 * gib),
            'stat': {'total_rss': gib * (1 + i), 'total_shmem': gib, 'total_cache': 2 * gib}, 'cpu_sec': 20.0 * i,
            'ncpu': 8, 'shm_used': gib} for i in range(4)]
    s = reslog.summary(new, healthy_at=115)
    assert s['limit_gib'] == 8.0 and s['startup']['hard_max_gib'] == 3.0 and s['serving']['hard_max_gib'] == 5.0
    assert s['serving']['cores_max'] == 2.0 and s['serving']['at_hard_max']['rss'] == 4.0
    old = [{'t': 100.0, 'load1': 1.0, 'mem': 5 * gib, 'mem_peak': 6 * gib, 'limit': 8 * gib, 'hard': 2 * gib,
            'stat': {'rss': gib, 'shmem': gib, 'cache': 3 * gib}, 'cpu_sec': 1.0}]   # task_synthesis/serve/reslog.py
    o = reslog.summary(old)
    assert o['limit_gib'] == 8.0 and o['serving']['hard_max_gib'] == 2.0 and o['mem_peak_gib'] == 6.0


def test_run_cli_with_parent_and_a_thread(tmp_path):
    log = tmp_path / 'r.jsonl'
    r = subprocess.run([sys.executable, '-I', str(HERE / 'reslog.py'), 'run', str(log), '1', '--parent', '1'],
                       timeout=10)
    assert r.returncode == 0 and not log.exists()                         # its parent is not pid 1: exits at once
    t = reslog.start(tmp_path / 't.jsonl', 0.2, extra=lambda: {'busy_slots': 3})
    for _ in range(50):
        if (tmp_path / 't.jsonl').exists() and len((tmp_path / 't.jsonl').read_text().splitlines()) >= 2:
            break
        time.sleep(0.1)
    t.stop.set()
    t.join(5)
    assert not t.is_alive(), 'start(): thread.stop ends the sampler'
    rows = [json.loads(x) for x in (tmp_path / 't.jsonl').read_text().splitlines()]
    assert t.daemon and rows[0]['busy_slots'] == 3 and 'cgroup' not in rows[1]
    assert set(rows[0]['cgroup']) == {'memory', 'cpu'}                      # the directories actually read


def test_interactive_shell_child_maps_to_the_job(tmp_path):
    """aa-serving's sampler rule, kept: memory and CPU of the job's cgroup (topmost ancestor with a limit), not of the
    interactive shell's child cgroup; a batch job is its own cgroup."""
    job = 'htcondor/slot1_1@x'
    files = {}
    for rel, lim in (('htcondor', 2 ** 63 - 4096), (job, 2 ** 30), (f'{job}/sshd', 2 ** 30)):
        files[f'memory/{rel}/memory.limit_in_bytes'] = f'{lim}\n'
        files[f'memory/{rel}/memory.usage_in_bytes'] = '7\n' if rel == job else '1\n'
        files[f'cpuacct/{rel}/cpuacct.usage'] = '5000000000\n' if rel == job else '1\n'
    proc, cg = fake(tmp_path, f'9:memory:/{job}/sshd\n2:cpuacct:/{job}/sshd\n', files)
    s = reslog.sample(proc, cg, {})
    assert s['mem'] == '7' and s['cpu_sec'] == 5.0
    proc2, cg2 = fake(tmp_path / 'b', f'9:memory:/{job}\n2:cpuacct:/{job}\n', files)
    assert reslog.sample(proc2, cg2, {})['mem'] == '7'


def test_mem_stat_keeps_the_serving_samplers_counters(tmp_path):
    stat = ('rss 0\ntotal_rss 100\ntotal_shmem 20\ntotal_cache 900\ntotal_mapped_file 5\ntotal_unevictable 7\n'
            'total_swap 0\ntotal_pgmajfault 3\ntotal_rss_huge 2\ntotal_active_anon 60\ntotal_inactive_file 400\n')
    proc, cg = fake(tmp_path, V1_CGROUP, {f'memory/{JOB}/memory.stat': stat})
    s = reslog.sample(proc, cg, {})
    assert s['stat']['rss'] == 0 and s['stat']['total_rss'] == 100                      # raw, as tbshared wrote
    assert s['mem_stat'] == {'cache': 900, 'rss': 100, 'rss_huge': 2, 'shmem': 20, 'mapped_file': 5,
                             'unevictable': 7, 'active_anon': 60, 'inactive_file': 400, 'swap': 0, 'pgmajfault': 3}
    proc, cg = fake(tmp_path / 'v2', f'0::/{V2_JOB}\n', {f'{V2_JOB}/memory.stat': 'anon 5\nfile 6\nfile_mapped 1\n'})
    assert reslog.sample(proc, cg, {})['mem_stat'] == {'rss': 5, 'cache': 6, 'mapped_file': 1}


def test_v2_cpu_and_procs_come_from_the_job_cgroup(tmp_path):
    job = 'system.slice/job.scope'
    files = {f'{job}/memory.max': '1073741824\n', f'{job}/cpu.stat': 'usage_usec 9000000\n',
             f'{job}/cgroup.procs': '', f'{job}/sshd/memory.max': '1073741824\n',
             f'{job}/sshd/cpu.stat': 'usage_usec 1000000\n', f'{job}/sshd/cgroup.procs': '41\n'}
    proc, cg = fake(tmp_path, f'0::/{job}/sshd\n', files)
    (proc / '41').mkdir()
    (proc / '41' / 'status').write_text('VmRSS:\t 2097152 kB\n')
    (proc / '41' / 'cmdline').write_text('vllm\0')
    s = reslog.sample(proc, cg, {'procs': True})
    assert s['cpu_sec'] == 9.0                                     # the job's, not the shell child's 1.0
    assert [r['pid'] for r in s['procs']] == [41]                  # found in the child cgroup of the job


def test_cli_output_path_may_contain_an_equals_sign(tmp_path):
    d = tmp_path / 'a=b'
    log, pidf = d / 'r.jsonl', tmp_path / 'pid'
    script = f'"{sys.executable}" -I "{HERE / "reslog.py"}" "{log}" 60 & echo $! > "{pidf}"; sleep 1.5'
    subprocess.run(['bash', '-c', script], check=True, timeout=30, cwd=tmp_path)
    assert log.exists() and not (tmp_path / '60').exists()


def test_summary_keeps_dirty_writeback_and_reads_minute_logs():
    gib = 2 ** 30
    rows = [{'t': 60.0 * i, 'mem': str(gib), 'stat': {'total_rss': gib, 'total_dirty': gib, 'total_writeback': gib},
             'cpu_sec': 120.0 * i} for i in range(5)]
    rows.insert(3, {**rows[2], 't': 600.0, 'cpu_sec': 2000.0})     # a gap: left out of the cores
    rows = sorted(rows, key=lambda r: r['t'])
    s = reslog.summary(rows)['serving']
    assert s['dirty_wb_max_gib'] == 2.0 and s['cores_max'] == 2.0 and s['at_hard_max']['dirty'] == 1.0

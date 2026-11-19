"""tbshared.condor_cost: bid ladder placement, submission and escalation (no condor needed)."""
import json
import types
from pathlib import Path

import pytest

from tbshared import condor_cost as C

# UtsnameNodename MinRunningPrice Cpus Memory(MB) Gpus
STATUS = '\n'.join([
    'gpu101\t15\t40\t300000\t0',        # A100-40, floor 15: 40 CPUs free, no free GPU
    'gpu102\t25\t20\t100000\t1',        # A100, floor 25, 1 free GPU keeps 8 CPUs / 32000 MB
    'c001\t5\t8\t64000\tundefined',   # CPU-only node
    'cpu101\t35\t64\t900000\t0',        # H100
    'cpu301\t55\t200\t3000000\t0',      # B200
])


def test_fits_keeps_gpu_reserve():
    assert C.fits(20, 100000, 1, 4, 16384) == 3    # 20-8=12 CPUs usable -> 3 x 4
    assert C.fits(20, 100000, 0, 4, 16384) == 5
    assert C.fits(8, 64000, 0, 0, 1) == 0


def test_room_by_floor_and_plan_cheapest_first():
    room = C.room_by_floor(4, 16, status=STATUS)
    assert room == {15.0: 10, 25.0: 3, 5.0: 2, 35.0: 16, 55.0: 50}
    kw = {'status': STATUS}
    assert C.plan_bids(10, 4, 16, **kw) == [(26, 10)]
    assert C.plan_bids(20, 4, 16, **kw) == [(26, 15), (36, 5)]
    assert C.plan_bids(40, 4, 16, **kw) == [(26, 15), (36, 16), (56, 9)]
    assert C.plan_bids(200, 4, 16, **kw) == [(26, 15), (36, 16), (56, 169)]   # rest on the top rung
    assert C.plan_bids(5, 4, 16, exclude=('gpu101', 'gpu102', 'c001'), **kw) == [(36, 5)]


def test_plan_falls_back_to_bottom_rung_when_status_fails(monkeypatch):
    def boom():
        raise OSError('no schedd')
    monkeypatch.setattr(C, 'condor_status_slots', boom)
    assert C.plan_bids(7, 4, 16) == [(26, 7)]


def test_parse_shape_and_split_queue():
    text = 'executable = /bin/true\nrequest_cpus = 6\nrequest_memory = 98304\nrequest_disk = 30G\nqueue 12\n'
    assert C.parse_shape(text) == (6, 96.0, 0.0)
    assert C.parse_shape('request_memory = 64G\n')[1] == 64.0
    body, n = C.split_queue(text)
    assert n == 12 and 'queue' not in body and body.endswith('request_disk = 30G\n')
    assert C.split_queue('a = b\nqueue\n# trailing comment\n') == ('a = b\n', 1)
    with pytest.raises(ValueError):
        C.split_queue('queue 2\nrequest_cpus = 1\n')


class Runner:
    def __init__(self):
        self.calls, self.cluster = [], 1000

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] == 'condor_submit_bid':
            self.cluster += 1
            return types.SimpleNamespace(returncode=0, stdout=f'1 job(s) submitted to cluster {self.cluster}.\n', stderr='')
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')


def test_submit_file_splits_queue_across_rungs(tmp_path):
    sub = tmp_path / 'worker.sub'
    sub.write_text('executable = /bin/true\nrequest_cpus = 4\nrequest_memory = 16384\nqueue 20\n')
    r = Runner()
    recs = C.submit_file(sub, status=STATUS, runner=r, watcher=False, fill=False)
    assert [(x['cluster'], x['bid'], x['n']) for x in recs] == [('1001', 26, 15), ('1002', 36, 5)]
    assert recs[0]['submit_file'].endswith('worker.bid26.sub')
    assert [c[:2] for c in r.calls] == [['condor_submit_bid', '26'], ['condor_submit_bid', '36']]
    a, b = (tmp_path / 'worker.bid26.sub').read_text(), (tmp_path / 'worker.bid36.sub').read_text()
    assert a.endswith('queue 15\n') and b.endswith('queue 5\n')
    assert '+AaBidLadder = "26,36,56"' in a and '+AaBidLadderSteps = "300,600"' in a


def test_submit_file_single_rung_rewrites_in_place_and_refuses_gpu(tmp_path):
    sub = tmp_path / 'job.sub'
    sub.write_text('request_cpus = 1\nrequest_memory = 4096\nqueue\n')
    r = Runner()
    assert [x['cluster'] for x in C.submit_file(sub, status=STATUS, runner=r, watcher=False)] == ['1001']
    assert sub.read_text().endswith('+AaBidLadderSteps = "300,600"\nqueue 1\n')
    g = tmp_path / 'gpu.sub'
    g.write_text('request_cpus = 8\nrequest_memory = 4096\nrequest_gpus = 1\nqueue\n')
    with pytest.raises(ValueError):
        C.submit_file(g, status=STATUS, runner=r, watcher=False)


@pytest.mark.parametrize('fill', [True, False])
@pytest.mark.parametrize('shape', ['request_cpus = 4.0\nrequest_memory = 8192\n',
                                   'request_cpus = 4\nrequest_memory = 4096 * 2\n'])
def test_submit_file_plans_shapes_fill_cannot_rewrite(tmp_path, shape, fill):
    """A readable shape fill_plan leaves as written (float CPUs, an arithmetic request) is still planned and submitted
    on the ladder, as in 1.4 (review of 7f8f3221: TypeError on a None plan)."""
    sub = tmp_path / 'odd.sub'
    sub.write_text('executable = /bin/true\n' + shape + 'queue 2\n')
    r = Runner()
    recs = C.submit_file(sub, status=STATUS, runner=r, watcher=False, fill=fill)
    assert sum(x['n'] for x in recs) == 2 and all(c[0] == 'condor_submit_bid' for c in r.calls)


def test_next_bid_waits_per_rung_and_never_lowers():
    lad, st = [26, 36, 56], [300, 600]
    assert C.next_bid(26, lad, st, idle_since=1000, last_change=None, now=1299) is None
    assert C.next_bid(26, lad, st, idle_since=1000, last_change=None, now=1300) == 36
    assert C.next_bid(36, lad, st, idle_since=1000, last_change=1300, now=1899) is None
    assert C.next_bid(36, lad, st, idle_since=1000, last_change=1300, now=1900) == 56
    assert C.next_bid(56, lad, st, idle_since=0, last_change=None, now=10 ** 9) is None
    assert C.next_bid(61, lad, st, idle_since=0, last_change=None, now=10 ** 9) is None   # above the ladder: untouched
    assert C.next_bid(20, lad, st, idle_since=0, last_change=None, now=1) == 26            # below it: to the bottom
    assert C.next_bid(30, lad, st, idle_since=0, last_change=None, now=300) == 36          # between rungs: rung 26's wait


def test_escalate_rebids_only_waiting_cpu_ladder_jobs():
    jobs = [
        {'ClusterId': 1, 'ProcId': 0, 'JobPrio': -974, 'EnteredCurrentStatus': 0, 'AaBidLadder': '26,36,56',
         'AaBidLadderSteps': '300,600'},                                  # waited long enough -> 36
        {'ClusterId': 1, 'ProcId': 1, 'JobPrio': -974, 'EnteredCurrentStatus': 950, 'AaBidLadder': '26,36,56'},  # too soon
        {'ClusterId': 2, 'ProcId': 0, 'JobPrio': -964, 'EnteredCurrentStatus': 0, 'AaBidLadderAt': 500,
         'AaBidLadder': '26,36,56', 'AaBidLadderSteps': '300,600'},       # rebid at 500: waits until 1100
        {'ClusterId': 3, 'ProcId': 0, 'JobPrio': -974, 'EnteredCurrentStatus': 0, 'AaBidLadder': '26,36,56',
         'RequestGpus': 1},                                               # GPU job: never
    ]

    class R(Runner):
        def __call__(self, argv, **kw):
            if argv[0] == 'condor_q':
                self.calls.append(argv)
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(jobs), stderr='')
            return super().__call__(argv, **kw)
    r = R()
    moves = C.escalate(now=1000, owner='me', runner=r, log=lambda s: None)
    assert moves == [('1.0', 26, 36)]
    assert ['condor_prio', '-p', '-964', '1.0'] in r.calls
    assert ['condor_qedit', '1.0', 'AaBidLadderAt', '1000'] in r.calls
    assert r.calls.index(['condor_qedit', '1.0', 'AaBidLadderAt', '1000']) < r.calls.index(['condor_prio', '-p', '-964', '1.0'])
    r2 = R()
    assert C.escalate(now=1200, owner='me', runner=r2, log=lambda s: None, dry_run=True) == [('1.0', 26, 36), ('2.0', 36, 56)]
    assert not any(c[0] in ('condor_prio', 'condor_qedit') for c in r2.calls)


def test_ensure_watcher_submits_once(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'WATCH_DIR', tmp_path)

    class R(Runner):
        running = ''

        def __call__(self, argv, **kw):
            if argv[0] == 'condor_q':
                self.calls.append(argv)
                return types.SimpleNamespace(returncode=0, stdout=self.running, stderr='')
            return super().__call__(argv, **kw)
    r = R()
    status = 'c1\t15\t4\t20000\t0\t0'         # a floor-15 node with room for the watcher (1 CPU / 2 GB, filled: 16 GB)
    assert C.ensure_watcher(owner='me', runner=r, status=status) == '1001'
    sub = next(tmp_path.glob('watch-*.sub')).read_text()
    assert 'request_cpus = 1' in sub and f'+JobBatchName = "{C.WATCH_BATCH}"' in sub and str(tmp_path) in sub
    assert '+AaBidLadder = "36,56"' in sub                         # the watcher is a ladder job itself, from 36
    assert (tmp_path / 'code').exists() and ['condor_submit_bid', '36'] == r.calls[-1][:2]
    r.running = '1001\n'
    assert C.ensure_watcher(owner='me', runner=r) is None


def test_submit_fixed_bid_bypasses_ladder(tmp_path):
    sub = tmp_path / 'job.sub'
    sub.write_text('request_cpus = 1\nrequest_memory = 4096\nqueue 3\n')
    r = Runner()
    r.__class__ = type('R3', (Runner,), {'__call__': lambda self, argv, **kw: (self.calls.append(argv), types.SimpleNamespace(
        returncode=0, stdout='3 job(s) submitted to cluster 77.\n', stderr=''))[1]})
    assert C.submit(sub, 61, runner=r) == [{'cluster': '77', 'bid': 61, 'n': 3, 'submit_file': str(sub)}]
    assert r.calls == [['condor_submit_bid', '61', str(sub)]] and 'AaBidLadder' not in sub.read_text()


def test_partial_ladder_submission_reports_what_was_queued(tmp_path):
    sub = tmp_path / 'worker.sub'
    sub.write_text('request_cpus = 4\nrequest_memory = 16384\nqueue 20\n')

    class Half(Runner):
        def __call__(self, argv, **kw):
            if argv[0] == 'condor_submit_bid' and argv[1] == '36':
                return types.SimpleNamespace(returncode=1, stdout='', stderr='Failed to connect to queue manager')
            return super().__call__(argv, **kw)
    with pytest.raises(C.SubmitError) as e:
        C.submit_file(sub, status=STATUS, runner=Half(), watcher=False, fill=False)
    assert [(x['cluster'], x['bid'], x['n']) for x in e.value.submitted] == [('1001', 26, 15)]
    assert not isinstance(e.value, SystemExit)


def test_requirement_hints_narrow_the_capacity():
    text = ('request_cpus = 4\nrequest_memory = 16384\nrequirements = (TARGET.UtsnameNodename != "gpu101") && '
            '(TARGET.GPUs == 0 || TARGET.GPUs =?= undefined)\nqueue 1\n')
    nodes, gpu_free = C.requirement_hints(text)
    assert nodes == {'gpu101'} and gpu_free
    status = STATUS.replace('gpu101\t15\t40\t300000\t0', 'gpu101\t15\t40\t300000\t0\t0') \
                   .replace('c001\t5\t8\t64000\tundefined', 'c001\t5\t8\t64000\tundefined\t0') \
                   .replace('gpu102\t25\t20\t100000\t1', 'gpu102\t25\t20\t100000\t1\t4')
    assert C.room_by_floor(4, 16, status=status, gpu_free_nodes_only=True)[5.0] == 2
    assert C.room_by_floor(4, 16, status=status, gpu_free_nodes_only=True).get(25.0, 0) == 0   # gpu102 has GPUs


def test_auto_release_on_memory_holds_ladder_and_fixed(tmp_path):
    sub = tmp_path / 'job.sub'
    sub.write_text('request_cpus = 1\nrequest_memory = 4096\nqueue 2\n')
    C.submit_file(sub, status=STATUS, runner=Runner(), watcher=False)
    assert 'periodic_release = (HoldReasonCode == 34) && (NumHolds <= 5)' in sub.read_text()
    fixed = tmp_path / 'fixed.sub'
    fixed.write_text('request_cpus = 1\nrequest_memory = 4096\nqueue 3\n')
    r = Runner()
    r.__class__ = type('R4', (Runner,), {'__call__': lambda self, argv, **kw: (self.calls.append(argv), types.SimpleNamespace(
        returncode=0, stdout='3 job(s) submitted to cluster 77.\n', stderr=''))[1]})
    C.submit(fixed, 61, runner=r)
    assert fixed.read_text().endswith('NumHolds <= 5)\nqueue 3\n')
    own = tmp_path / 'own.sub'
    own.write_text('request_cpus = 1\nperiodic_release = False\nqueue\n')
    C.submit_file(own, status=STATUS, runner=Runner(), watcher=False)
    assert own.read_text().count('periodic_release') == 1                # a job's own policy is kept


def test_units():
    assert C.units(16, 256) == 16 and C.units(8, 256) == 16 and C.units(16, 64, 1) == 40 and C.units(1, 0.5) == 1


def test_auto_release_opt_out():
    assert 'periodic_release' in C.with_auto_release('x = 1\n')
    assert C.with_auto_release('x = 1\n', auto_release=False) == 'x = 1\n'
    assert C.with_auto_release('periodic_release = false\n') == 'periodic_release = false\n'


def test_ensure_watcher_escalates_while_the_watcher_waits(tmp_path, monkeypatch):
    """A queued but not running watcher cannot escalate anything (itself included): ensure_watcher does one pass."""
    monkeypatch.setattr(C, 'WATCH_DIR', tmp_path)
    calls = []
    idle = [{'ClusterId': 9, 'ProcId': 0, 'JobStatus': 1, 'JobPrio': 26 - 1000, 'RequestGpus': 0,
             'EnteredCurrentStatus': 0, 'AaBidLadder': '26,36,56', 'AaBidLadderSteps': '300,600'}]

    def runner(argv, **kw):
        calls.append(argv)
        out = ''
        if argv[0] == 'condor_q' and '-json' in argv:
            out = json.dumps(idle)
        elif argv[0] == 'condor_q':
            out = '9\n' if '||' in ' '.join(argv) else ''   # queued (status 1 or 2), not running
        return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
    assert C.ensure_watcher(owner='me', runner=runner) is None
    assert ['condor_prio', '-p', str(36 - 1000), '9.0'] in calls


def test_watcher_alone_in_the_queue_exits(monkeypatch):
    """The watcher is a ladder job itself: it must not count itself when deciding to idle-exit."""
    clock = iter(range(0, 10**6, 60))
    monkeypatch.setattr(C.time, 'time', lambda: next(clock))
    monkeypatch.setattr(C.time, 'sleep', lambda s: None)
    monkeypatch.setattr(C, '_own_cluster', lambda: '5')
    monkeypatch.setattr(C, 'watcher_jobs', lambda owner, running_only=False: ['5'])
    monkeypatch.setattr(C, 'escalate', lambda **kw: [])
    seen = []

    def ladder_jobs(owner, runner=None, idle_only=True, watchers=True):
        seen.append(watchers)
        return [] if not watchers else [{'JobStatus': 2, 'ClusterId': 5}]
    monkeypatch.setattr(C, 'ladder_jobs', ladder_jobs)
    C.watch(interval=60, idle_exit=600)            # returns (a loop that never exits would hang the test)
    assert seen and not any(seen)


def test_ladder_jobs_can_leave_out_watchers():
    cmds = []

    def runner(argv, **kw):
        cmds.append(argv)
        return types.SimpleNamespace(returncode=0, stdout='[]', stderr='')
    C.ladder_jobs('me', runner=runner, idle_only=False, watchers=False)
    assert f'JobBatchName =!= "{C.WATCH_BATCH}"' in cmds[0][cmds[0].index('-constraint') + 1]


def test_escalate_skips_the_move_when_the_rebid_time_cannot_be_written():
    """Without AaBidLadderAt the next pass would measure from EnteredCurrentStatus and move the job up again at once."""
    jobs = [{'ClusterId': 1, 'ProcId': 0, 'JobPrio': -974, 'EnteredCurrentStatus': 0, 'AaBidLadder': '26,36,56'}]
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        if argv[0] == 'condor_q':
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(jobs), stderr='')
        return types.SimpleNamespace(returncode=1 if argv[0] == 'condor_qedit' else 0, stdout='', stderr='schedd busy')
    assert C.escalate(now=1000, owner='me', runner=runner, log=lambda s: None) == []
    assert not any(c[0] == 'condor_prio' for c in calls)


def test_escalate_prio_failure_or_timeout_after_the_rebid_time():
    """The rebid time written, then condor_prio fails (or times out): not reported, retried after one more step."""
    import subprocess
    jobs = [{'ClusterId': 1, 'ProcId': 0, 'JobPrio': -974, 'EnteredCurrentStatus': 0, 'AaBidLadder': '26,36,56'}]
    for fail in ('rc', 'timeout'):
        calls = []

        def runner(argv, **kw):
            calls.append(argv)
            if argv[0] == 'condor_q':
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(jobs), stderr='')
            if argv[0] == 'condor_prio':
                if fail == 'timeout':
                    raise subprocess.TimeoutExpired(argv, 120)
                return types.SimpleNamespace(returncode=1, stdout='', stderr='busy')
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        assert C.escalate(now=1000, owner='me', runner=runner, log=lambda s: None) == []
        assert ['condor_qedit', '1.0', 'AaBidLadderAt', '1000'] in calls
    assert C.next_bid(26, (26, 36, 56), (300, 600), 0, 1000, 1299) is None     # waits one more step from the At
    assert C.next_bid(26, (26, 36, 56), (300, 600), 0, 1000, 1300) == 36


def test_memory_net_matches_holds_and_keeps_the_base_as_the_shape():
    from tbshared import holds as H
    assert (C.MEMORY_HOLDS, C.MEMORY_STEP) == (H.MEMORY_HOLDS, H.MEMORY_STEP)   # one net for servers and CPU jobs
    assert C.memory_net('request_memory = 98304\n') == f'request_memory = {H.memory_request(96)}\n'
    assert C.memory_net('request_memory = 98304\n') == f'request_memory = 98304 + 24576 * {H.MEMORY_HOLDS}\n'
    assert C.memory_net('Request_Memory = 32G\n') == f'Request_Memory = 32768 + 8192 * {H.MEMORY_HOLDS}\n'
    expr = f'request_memory = 4096 + 1024 * {H.MEMORY_HOLDS}\n'
    assert C.memory_net(expr) == expr and C.memory_net('request_memory = $(mem)\n') == 'request_memory = $(mem)\n'
    assert C.parse_shape('request_cpus = 4\n' + C.memory_net('request_memory = 49152\n')) == (4, 48.0, 0.0)
    assert C.parse_shape('request_cpus = 4\nrequest_memory = 48G + 1\n') == (4, 48.0, 0.0)
    assert C.parse_shape('request_cpus = 4\n')[1] == 1.0                                       # condor's default
    with pytest.raises(ValueError):
        C.parse_shape('request_memory = $(mem)\n')


def test_auto_release_adds_the_memory_net_and_leaves_own_nets_alone():
    from tbshared import holds as H
    body = 'request_cpus = 4\nrequest_memory = 32768\n'
    out = C.with_auto_release(body)
    assert out == f'request_cpus = 4\nrequest_memory = 32768 + 8192 * {H.MEMORY_HOLDS}\n' + C.AUTO_RELEASE
    assert 'NumHolds <= 5' in C.AUTO_RELEASE                                   # the release bound is unchanged
    assert C.with_auto_release(body, auto_release=False) == body             # no release: no growth either
    own = 'request_cpus = 4\n' + H.submit_lines(96)                          # a server's own net (holds.py): as is
    assert C.with_auto_release(own) == own


def test_ladder_fit_check_uses_the_base_and_the_file_gets_the_net(tmp_path, monkeypatch):
    from tbshared import holds as H
    seen = []
    real = C.plan_bids
    monkeypatch.setattr(C, 'plan_bids', lambda n, cpus, mem, *a, **k: (seen.append((n, cpus, mem)), real(n, cpus, mem, *a, **k))[1])
    sub = tmp_path / 'worker.sub'
    sub.write_text('request_cpus = 4\nrequest_memory = 32768\nqueue 3\n')
    C.submit_file(sub, status=STATUS, runner=Runner(), watcher=False, fill=False)
    assert seen == [(3, 4, 32.0)]
    text = sub.read_text()
    assert f'request_memory = 32768 + 8192 * {H.MEMORY_HOLDS}\n' in text and text.count('periodic_release') == 1
    assert C.parse_shape(text) == (4, 32.0, 0.0)                              # a resubmit of the written file: same base
    seen.clear()
    C.submit_file(sub, status=STATUS, runner=Runner(), watcher=False, fill=False)
    assert seen == [(3, 4, 32.0)] and text.count(H.MEMORY_HOLDS) == sub.read_text().count(H.MEMORY_HOLDS) == 1



# condor_status_slots rows with the node totals: name floor Cpus Memory(MB) Gpus TotalGpus TotalCpus TotalMemory CUDA
GB = 1024
NODES = '\n'.join([
    f'b01\t55\t200\t{2000 * GB}\t8\t8\t384\t{3022 * GB}\t10.0',      # B200: 48 CPUs and 377 GB per GPU
    f'h01\t35\t100\t{1500 * GB}\t8\t8\t192\t{3023 * GB}\t9.0',       # H100 8-GPU
    f'h02\t35\t60\t{600 * GB}\t4\t4\t96\t{1007 * GB}\t9.0',          # H100 4-GPU: 24 CPUs, 251 GB per GPU
    f'a01\t25\t100\t{900 * GB}\t4\t4\t128\t{1007 * GB}\t8.0',        # A100 4-GPU
    f'a02\t25\t100\t{1500 * GB}\t8\t8\t128\t{2003 * GB}\t8.0',       # A100 8-GPU: 16 CPUs, 250 GB per GPU
    f't01\t15\t30\t{700 * GB}\t8\t8\t36\t{754 * GB}\t7.5',           # CUDA 7.5: 4.5 CPUs, 94 GB per GPU
    f'c01\t15\t100\t{1000 * GB}\t0\t0\t128\t{1007 * GB}\tundefined',  # CPU-only, floor 15: 10 GB free per free CPU
    f'c02\t35\t192\t{3000 * GB}\t0\t0\t192\t{3023 * GB}\tundefined',  # fat CPU node: the largest memory
])


def body(c, m, extra=''):
    return f'request_cpus = {c}\nrequest_memory = {int(m * 1024)}\n{extra}'


def test_shape_caps_per_gpu_share_or_requirements_and_unknown_classes():
    assert C.shape_caps(8, '10.0', NODES) == (384, 3022)
    assert C.shape_caps(1, ['9.0'], NODES) == (24, 251)                     # min over both H100 classes
    assert C.shape_caps(1, ['9.0', '8.0'], NODES) == (16, 250)              # OR'd: min over every named class
    assert C.shape_caps(1, None, NODES) == (4, 94)                          # unnamed: every GPU node (7.5 binds)
    assert C.shape_caps(4, '8.0', NODES) == (64, 1001)                      # 4-GPU A100: 128 / 4 x 4, 1007 GB
    assert C.shape_caps(1, '12.0', NODES) is None                           # a class with no node here: unknown
    caps = C.shape_caps(0, None, NODES)                                     # CPU jobs: per node, net of GPU reserves
    assert caps[0] == 368 and (192, 3023) in caps[1] and (368, 2959) in caps[1]   # B200: 384 - 16 CPUs, - 64000 MB
    assert C.mem_cap_for(caps, 190) == 3023 and C.mem_cap_for(caps, 256) == 2959 and C.mem_cap_for(caps, 400) is None
    assert C.shape_caps(0, None, STATUS) is None                            # no totals in condor_status: unknown
    assert C.shape_caps(2, (96, 1007, 4)) == (48, 503)                      # an explicit (TotalCpus, GB, GPUs)
    assert C._capabilities('(TARGET.CUDACapability == 9.0) || (TARGET.CUDACapability == 8.0)') == [9.0, 8.0]
    assert C._capabilities('TARGET.CUDACapability >= 8.0') is None and C._capabilities('Cpus > 2') == []


def test_fill_shape_gpu_jobs_within_the_node_class():
    logs = []
    # Snowball 8xB200 at 1050 GB: wants 66 CPUs, under the B200 class's 48 per GPU x 8 = 384; RAM to 1056 GB
    assert C.fill_shape(16, 1050, 8, node='10.0', status=NODES, log=logs.append) == (66, 1056)
    assert not logs
    # 4xB200 at 192c/1000G: RAM capped at the node's share (1511 GB), not 16 x 192 = 3072 GB
    assert C.fill_shape(192, 1000, 4, node='10.0', status=NODES, log=logs.append) == (192, 1511)
    assert 'capped at 192c/1511G' in logs[-1]
    # 1xA100 at 32c: 250 GB per GPU on the 8-GPU class, not 512 GB (which would strand 3 GPUs of a 4-GPU node)
    assert C.fill_shape(32, 64, 1, node='8.0', status=NODES) == (32, 250)
    # H100 1 GPU at 640 GB wants 40 CPUs: capped at the 24 per GPU (RAM stays as asked, above the share)
    assert C.fill_shape(8, 640, 1, node='9.0', status=NODES, log=logs.append) == (24, 640)
    assert C.fill_shape(8, 64, 1, node='9.0', status=NODES) == (8, 128)
    assert C.fill_shape(8, 64, 1, node='12.0', status=NODES, log=logs.append) == (8, 64)   # unknown: not filled
    assert 'not filled: node class unknown' in logs[-1]

    def boom(max_age=None):
        raise OSError('no condor_status')
    orig = C.condor_status_slots
    try:
        C.condor_status_slots = boom
        assert C.fill_shape(8, 640, 1) == (8, 640) and C.fill_shape(4, 32) == (4, 32)   # condor_status down
    finally:
        C.condor_status_slots = orig


def test_fill_shape_cpu_jobs_at_the_same_units_within_the_largest_node():
    for (c, g), want in {(4, 32): (4, 64), (10, 128): (10, 160), (2, 24): (2, 32), (6, 96): (6, 96),
                         (1, 4): (1, 16), (2, 200): (13, 208)}.items():
        assert C.fill_shape(c, g, status=NODES) == want and C.units(*want) == max(want[0], want[1] / 16)
    assert C.units(2, 200) == 12.5 and C.units(13, 208) == 13                # whole CPUs: under 1 unit more
    assert C.fill_shape(190, 100, status=NODES) == (190, 3023)               # never above the largest node's memory
    assert C.fill_shape(256, 100, status=NODES) == (256, 2959)               # only nodes with >= 256 CPUs: the B200s
    assert C.fill_shape(400, 100, status=NODES) == (400, 100)                # more CPUs than any node: RAM not filled


def test_condor_status_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(C, '_STATUS_CACHE', {})
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **k: calls.append(a) or types.SimpleNamespace(stdout=NODES))
    assert C.condor_status_slots() == NODES and C.condor_status_slots() == NODES and len(calls) == 1
    assert 'TotalMemory' in calls[0][0] and 'CUDACapability' in calls[0][0]
    C._STATUS_CACHE['t'] -= C.STATUS_TTL + 1
    C.condor_status_slots()
    assert len(calls) == 2


def test_parse_shape_leading_size_as_before():
    assert C.parse_shape('request_memory = 4096 * 2\n')[1] == 4.0               # the leading integer, as before 1.5
    assert C.parse_shape(f'request_memory = 4096 + 1024 * {C.MEMORY_HOLDS}\n')[1] == 4.0
    assert C.parse_shape('request_memory = 64 GB\n')[1] == 64.0
    with pytest.raises(ValueError, match='does not start with a size'):
        C.parse_shape('request_memory = $(mem)\n')


# placement: cheap tier (floor <= 26) = c01 only (100 CPUs, 1000 GB: 10 GB per free CPU); then H100/fat CPU at 35
PLACE = '\n'.join([f'c01\t15\t100\t{1000 * GB}\t0\t0\t128\t{1007 * GB}\tundefined',
                   f'c02\t35\t192\t{3000 * GB}\t0\t0\t192\t{3023 * GB}\tundefined'])


def test_fill_plan_fills_where_free_caps_at_the_cheap_ratio_or_falls_back(tmp_path):
    logs, where = [], str(tmp_path / 'w.sub')
    # 2 jobs of 4c/32G -> 4c/64G: both still fit at bid 26 (100c/1000G): filled
    b, plan, rec = C.fill_plan(body(4, 32), 2, C.LADDER, status=PLACE, where=where, log=logs.append)
    assert (rec['decision'], rec['after'], plan) == ('filled', {'cpus': 4, 'mem_gb': 64}, [(26, 2)])
    assert C.parse_shape(b)[:2] == (4, 64.0) and rec['plans']['unfilled'] == [(26, 2)]
    # 25 jobs: unfilled 25 x 32G fit (800G); filled 15 x 64G only: capped at 10 GB/CPU = 40G, and 25 x 40 G fit
    b, plan, rec = C.fill_plan(body(4, 32), 25, C.LADDER, status=PLACE, where=where, log=logs.append)
    assert rec['decision'] == 'capped' and rec['after'] == {'cpus': 4, 'mem_gb': 40.0} and plan == [(26, 25)]
    assert rec['plans']['filled'] == [(26, 15), (36, 10)] and 'GB per free CPU' in rec['reason']
    assert C.parse_shape(b)[:2] == (4, 40.0)
    # 31 jobs of 3c/30G: unfilled, 33 fit at bid 26; filled (48G), 20; capped at 10 GB/CPU = 30G is the request: unfilled
    b, plan, rec = C.fill_plan(body(3, 30), 31, C.LADDER, status=PLACE, where=where, log=logs.append)
    assert rec['decision'] == 'unfilled' and C.parse_shape(b)[:2] == (3, 30.0) and plan == [(26, 31)]
    assert 'higher rung' in rec['reason'] and 'plans unfilled' in logs[-1]
    lines = [json.loads(x) for x in (tmp_path / 'fill.jsonl').read_text().splitlines()]
    assert [x['decision'] for x in lines] == ['filled', 'capped', 'unfilled']
    assert set(lines[0]) >= {'t', 'pipeline', 'batch', 'before', 'after', 'decision', 'reason', 'plans'}


def test_fill_plan_gpu_unknown_odd_and_off(tmp_path):
    where = str(tmp_path / 'g.sub')
    gpu = body(8, 640, 'request_gpus = 1\nrequirements = (TARGET.CUDACapability == 9.0) || '
                       '(TARGET.CUDACapability == 8.0)\n+JobBatchName = "serve-x"\n')
    b, plan, rec = C.fill_plan(gpu, 1, None, status=NODES, where=where, log=lambda m: None)
    assert plan is None and rec['decision'] == 'capped' and rec['after'] == {'cpus': 16, 'mem_gb': 640.0}
    assert rec['batch'] == 'serve-x' and 'request_cpus = 16\n' in b                    # A100's 16 per GPU binds
    ge = body(8, 64, 'request_gpus = 1\nrequirements = (TARGET.CUDACapability >= 8.0)\n')
    assert C.fill_plan(ge, 1, None, status=NODES, where=where, log=lambda m: None)[2]['decision'] == 'unfilled'
    odd = 'request_cpus = 4\nrequest_memory = 4096 * 2\n'
    b, _, rec = C.fill_plan(odd, 1, C.LADDER, status=PLACE, where=where, log=lambda m: None)
    assert b == odd and rec['decision'] == 'unfilled' and 'plain number' in rec['reason']
    b, plan, rec = C.fill_plan(body(4, 32), 2, C.LADDER, status=PLACE, fill=False, where=where, log=lambda m: None)
    assert rec['decision'] == 'off' and C.parse_shape(b)[:2] == (4, 32.0) and plan == [(26, 2)]
    assert C.fill_plan(body(4, 64), 1, C.LADDER, status=PLACE, where=where, log=lambda m: None)[2]['decision'] == \
        'unchanged'
    rep = C.fill_report([tmp_path])
    assert rep['batches'] == 5 and rep['decisions']['capped']['batches'] == 1 and rep['capped_share'] == 1.0
    assert rep['rungs_used'] == {'26': 3}     # off, unchanged (GPU batches and the odd one have no plan)


def test_submit_file_plan_matches_the_submitted_text(tmp_path, monkeypatch):
    from tbshared import holds as H
    seen = []
    real = C.plan_bids
    monkeypatch.setattr(C, 'plan_bids', lambda n, c, m, *a, **k: (seen.append((n, c, m)), real(n, c, m, *a, **k))[1])
    sub = tmp_path / 'w.sub'
    sub.write_text('request_cpus = 4\nrequest_memory = 32768\n+JobBatchName = "aa-pi-x"\nqueue 25\n')
    recs = C.submit_file(sub, status=PLACE, runner=Runner(), watcher=False)
    text = sub.read_text()
    assert f'request_memory = 40960 + 10240 * {H.MEMORY_HOLDS}\n' in text                # capped fill, then the net
    assert C.parse_shape(text)[:2] == (4, 40.0) and seen[-1] == (25, 4, 40.0)            # plan = submitted shape
    assert [(r['bid'], r['n']) for r in recs] == [(26, 25)]
    rec = json.loads((tmp_path / 'fill.jsonl').read_text().splitlines()[-1])
    assert rec['batch'] == 'aa-pi-x' and rec['decision'] == 'capped'
    fixed = tmp_path / 'f.sub'
    fixed.write_text('request_cpus = 2\nrequest_memory = 4096\nqueue 1\n')
    r = Runner()
    r.__class__ = type('R5', (Runner,), {'__call__': lambda self, argv, **kw: (self.calls.append(argv), types.SimpleNamespace(
        returncode=0, stdout='1 job(s) submitted to cluster 9.\n', stderr=''))[1]})
    C.submit(fixed, 61, runner=r, status=NODES)
    assert fixed.read_text().startswith(f'request_cpus = 2\nrequest_memory = 32768 + 8192 * {H.MEMORY_HOLDS}\n')


def test_watcher_plan_and_job_agree(tmp_path, monkeypatch):
    seen = []
    real = C.plan_bids
    monkeypatch.setattr(C, 'plan_bids', lambda n, c, m, *a, **k: (seen.append((c, m)), real(n, c, m, *a, **k))[1])
    monkeypatch.setattr(C, 'WATCH_DIR', tmp_path)
    r = Runner()
    r.__class__ = type('RW', (Runner,), {'__call__': lambda self, argv, **kw: types.SimpleNamespace(
        returncode=0, stdout='' if argv[0] == 'condor_q' else '1 job(s) submitted to cluster 5.\n', stderr='')})
    C.ensure_watcher(owner='me', runner=r, status='c1\t15\t4\t20000\t0\t0')
    sub = next(tmp_path.glob('watch-*.sub')).read_text()
    assert seen == [C.WATCH_SHAPE] and C.parse_shape(sub)[:2] == (C.WATCH_SHAPE[0], float(C.WATCH_SHAPE[1]))


def test_pipeline_name_from_the_callers_package(tmp_path, monkeypatch):
    """#480: the pipeline.json nearest above the calling module (a package's own on the trunk; a vendored branch's
    tbshared/pipeline.json at the top), wherever this module's copy lives."""
    monkeypatch.setattr(C, '_PIPELINE', {})
    r = tmp_path / 'repo'
    (r / '.git').mkdir(parents=True)
    for rel, name in (('eval_q35/pipeline.json', 'eval'), ('synth/pipeline.json', 'task-synthesis'),
                      ('task_synthesis/serve/pipeline.json', 'aa-serving')):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(json.dumps({'pipeline': name}))
    (r / 'task_synthesis' / 'pipeline').mkdir(parents=True)
    assert C.pipeline_name(r / 'eval_q35' / 'submit.py') == 'eval'
    assert C.pipeline_name(r / 'synth' / 'tests' / 'x.py') == 'task-synthesis'
    assert C.pipeline_name(r / 'task_synthesis' / 'serve' / 'serve_teacher.py') == 'aa-serving'
    assert C.pipeline_name(r / 'task_synthesis' / 'pipeline' / 'round.py') is None     # no config above: no name
    caller = r / 'eval_q35' / 'caller.py'                    # the default: the calling module's package
    caller.write_text('from tbshared import condor_cost\nNAME = condor_cost.pipeline_name()\n')
    ns = {}
    exec(compile(caller.read_text(), str(caller), 'exec'), ns)
    assert ns['NAME'] == 'eval'
    v = tmp_path / 'vendored'                                # a pipeline branch: tbshared/pipeline.json at the top
    (v / 'tbshared').mkdir(parents=True)
    (v / '.git').mkdir()
    (v / 'tbshared' / 'pipeline.json').write_text('{"pipeline": "aa-evals"}')
    assert C.pipeline_name(v / 'aa_evals' / 'common' / 'pi_eval' / 'run.py') == 'aa-evals'
    assert C.pipeline_name(v / 'tbshared' / 'condor_cost.py') == 'aa-evals'



def test_big_cpu_jobs_count_jobs_that_fit_nowhere(tmp_path):
    # a free B200 takes the capped 256c/2959G (368c / 2959 GB left for CPU jobs after the GPU reserve)
    free = f'b01\t55\t384\t{3022 * GB}\t8\t8\t384\t{3022 * GB}\t10.0'
    assert C.fill_plan(body(256, 100), 1, C.LADDER, status=free, where=str(tmp_path / 'b.sub'),
                       log=lambda m: None)[2]['nowhere'] == {'unfilled': 0, 'filled': 0}
    # a part-used B200 (2500 GB free): 256c/100G fits it, the filled one fits nowhere: unfilled
    b200 = f'b01\t55\t384\t{2500 * GB}\t8\t8\t384\t{3022 * GB}\t10.0'
    b, plan, rec = C.fill_plan(body(256, 100), 1, C.LADDER, status=b200, where=str(tmp_path / 'b.sub'),
                               log=lambda m: None)
    assert rec['nowhere'] == {'unfilled': 0, 'filled': 1} and rec['decision'] == 'unfilled' and plan == [(56, 1)]
    assert C.parse_shape(b)[:2] == (256, 100.0)
    # 190c/100G: fills to 190c/3023G, which needs a whole 192c node; with that node part-used it fits nowhere
    used = f'c02\t55\t192\t{2000 * GB}\t0\t0\t192\t{3023 * GB}\tundefined'
    rec = C.fill_plan(body(190, 100), 1, C.LADDER, status=used, where=str(tmp_path / 'c.sub'), log=lambda m: None)[2]
    assert rec['plans']['filled'] == rec['plans']['unfilled'] == [(56, 1)]       # the plan alone looks the same
    assert rec['nowhere']['filled'] == 1 and rec['decision'] == 'unfilled'
    assert C._worse([(26, 1)], [(26, 1)], C.LADDER, 1, 0) and not C._worse([(26, 1)], [(26, 1)], C.LADDER, 1, 1)


FIXED = f'c01\t15\t64\t{512 * GB}\t0\t0\t128\t{1007 * GB}\tundefined'   # floor 15: 64 free CPUs, 512 GB


def fixed_runner():
    r = Runner()
    r.__class__ = type('RF', (Runner,), {'__call__': lambda self, argv, **kw: types.SimpleNamespace(
        returncode=0, stdout='1 job(s) submitted to cluster 9.\n', stderr='')})
    return r


def test_fixed_bid_keeps_the_fill_only_where_the_room_holds(tmp_path):
    sub = tmp_path / 'f.sub'
    sub.write_text('request_cpus = 16\nrequest_memory = 131072\nqueue 2\n')       # 16c/128G: 4 fit, filled 256G: 2
    C.submit(sub, 25, runner=fixed_runner(), status=FIXED)
    assert C.parse_shape(sub.read_text())[:2] == (16, 256.0)                          # 2 jobs: room 2 >= min(2, 4)
    sub.write_text('request_cpus = 16\nrequest_memory = 131072\nqueue 3\n')
    C.submit(sub, 25, runner=fixed_runner(), status=FIXED)
    assert C.parse_shape(sub.read_text())[:2] == (16, 128.0)                          # 3 jobs: 2 < 3, cap = 8 GB/CPU
    recs = [json.loads(x) for x in (tmp_path / 'fill.jsonl').read_text().splitlines()]
    assert [r['decision'] for r in recs] == ['filled', 'unfilled'] and recs[1]['plans']['unfilled'] == [[25, 3]]


def test_macro_requests_keep_auto_release(tmp_path):
    sub = tmp_path / 'm.sub'
    sub.write_text('request_cpus = $(NCPU)\nrequest_memory = 4096\nqueue 2\n')
    C.submit(sub, 61, runner=fixed_runner(), status=FIXED)
    text = sub.read_text()
    assert C.AUTO_RELEASE in text and f'request_memory = 4096 + 1024 * {C.MEMORY_HOLDS}' in text   # net still added
    assert 'request_cpus = $(NCPU)' in text and text.endswith('queue 2\n')
    rec = json.loads((tmp_path / 'fill.jsonl').read_text())
    assert rec['decision'] == 'unfilled' and rec['before'] == {'cpus': None, 'mem_gb': None}


def test_condor_status_failures_are_cached_briefly(monkeypatch):
    calls = []

    def fail(*a, **k):
        calls.append(a)
        raise C.subprocess.TimeoutExpired('condor_status', 120)
    monkeypatch.setattr(C, '_STATUS_CACHE', {})
    monkeypatch.setattr(C.subprocess, 'run', fail)
    for _ in range(3):
        with pytest.raises(C.subprocess.TimeoutExpired):
            C.condor_status_slots()
    assert len(calls) == 1                                                            # not 3 x 120 s
    assert C.fill_shape(4, 32) == (4, 32)                                             # unknown: not filled, no wait
    C._STATUS_CACHE['err_t'] -= C.STATUS_FAIL_TTL + 1
    with pytest.raises(C.subprocess.TimeoutExpired):
        C.condor_status_slots()
    assert len(calls) == 2


def test_fill_report_never_raises(tmp_path):
    good = {'decision': 'filled', 'n': 2, 'plans': {'used': [[26, 2]], 'unfilled': [[26, 2]]}}
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'fill.jsonl').write_text('\n'.join([json.dumps(good), '[1, 2]', 'not json',
                                                          json.dumps({'decision': 'x', 'plans': 3}),
                                                          json.dumps({'decision': 'x', 'plans': {'used': [[26]]}}),
                                                          json.dumps({'decision': 'x', 'n': 'two'})]))
    (tmp_path / 'b' / 'fill.jsonl').mkdir(parents=True)                               # unreadable: a directory
    rep = C.fill_report([tmp_path, tmp_path / 'missing.jsonl'])
    assert rep['batches'] == 1 and rep['bad_lines'] == 7 and rep['rungs_used'] == {'26': 2}


def _load_copy(path):
    """condor_cost loaded from `path` (an alias copy or a vendored branch's tbshared/condor_cost.py)."""
    import importlib.util
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(C.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(f'cc_copy_{abs(hash(str(path)))}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_name_from_an_alias_copy_threads_and_the_command_line(tmp_path):
    """Review of #480 PR B: an alias copy names its caller's package; a thread with no pipeline frame and the command
    line fall back to the config governing the module's own copy (a vendored branch's file; none on the trunk)."""
    import concurrent.futures
    r = tmp_path / 'trunk'
    (r / '.git').mkdir(parents=True)
    (r / 'eval_q35').mkdir()
    (r / 'eval_q35' / 'pipeline.json').write_text('{"pipeline": "eval"}')
    alias = _load_copy(r / 'aa_evals' / 'common' / 'condor_cost.py')     # no aa_evals/pipeline.json here
    caller = r / 'eval_q35' / 'caller.py'
    ns = {'cc': alias}
    exec(compile('NAME = cc.pipeline_name()\n', str(caller), 'exec'), ns)
    assert ns['NAME'] == 'eval'
    assert alias.pipeline_name() is None                                  # pytest's frames: no package, no fallback
    # a stdlib frame between the caller and the module (sched.py calls the action): skipped, the caller's package
    alias._PIPELINE.clear()
    exec(compile('import sched\ns = sched.scheduler()\ns.enter(0, 1, cc.pipeline_name)\ns.run()\n', str(caller),
                 'exec'), ns)
    assert str(r / 'eval_q35' / 'pipeline.json') in alias._PIPELINE
    v = tmp_path / 'vendored'
    (v / '.git').mkdir(parents=True)
    (v / 'tbshared').mkdir()
    (v / 'tbshared' / 'pipeline.json').write_text('{"pipeline": "aa-evals"}')
    vend = _load_copy(v / 'tbshared' / 'condor_cost.py')
    assert vend.pipeline_name() == 'aa-evals'                             # command line / no caller package
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        assert pool.submit(vend.pipeline_name).result() == 'aa-evals'      # library frames are skipped
    assert vend.caller_file() is None or 'site-packages' not in vend.caller_file()


# ------------------------------------------------------------------------------------------------ bad_nodes (#531 A9)
@pytest.fixture
def live(tmp_path, monkeypatch):
    f = tmp_path / 'bad-nodes.txt'
    monkeypatch.setenv(C.BAD_NODES_FILE_ENV, str(f))
    return f


def test_bad_nodes_reads_the_file_and_the_live_list_at_each_call(tmp_path, monkeypatch, live):
    """A long-lived process that imported the module sees an edited BAD_NODES (its file, re-read) and the live file's
    additions at its next call, without a restart (the m5l grade watcher put a collector on gpu103)."""
    assert C.bad_nodes() == tuple(sorted(C.BAD_NODES))                         # no live file
    live.write_text('# reason, date\ngpu999  # EIO on /data (11-18)\n\n cpu998 \nbad"name\n')
    assert C.bad_nodes() == tuple(sorted({*C.BAD_NODES, 'gpu999', 'cpu998'}))      # comments, blanks; a bad line skipped
    src = tmp_path / 'condor_cost.py'                                          # an edited checkout, same process
    for text, want in (("BAD_NODES = ('g1',  # EIO\n             'g2')\n", {'g1', 'g2'}),
                       ("X = 1\nBAD_NODES = ['g3']\n", {'g3'}),
                       ("BAD_NODES = ('gpu103')\n", set(C.BAD_NODES)),            # a str, not a collection: imported
                       ("BAD_NODES = OTHER + ('g1',)\n", set(C.BAD_NODES))):     # not a literal: imported
        src.write_text(text)
        monkeypatch.setattr(C, '__file__', str(src))
        assert set(C.bad_nodes()) == want | {'gpu999', 'cpu998'}, text
    monkeypatch.setattr(C, '__file__', str(tmp_path / 'missing.py'))
    live.unlink()
    assert C.bad_nodes() == tuple(sorted(C.BAD_NODES))                         # unreadable source: imported


def test_live_file_formats_and_bad_bytes(live):
    """Spaces and commas between names, fully qualified host names (UtsnameNodename is the short name), and a bad
    byte (skipped, never an exception: one bad byte must not stop every submitter)."""
    live.write_bytes(b'gpu901, gpu902 gpu903\ngpu904.internal.cluster.internal\n\xff\xfe\ngpu905\n'
                     b'gpu906 EIO on /data since 11-19\n')   # a reason without its '#': the words are not nodes
    got = set(C.bad_nodes()) - set(C.BAD_NODES)
    assert got == {'gpu901', 'gpu902', 'gpu903', 'gpu904', 'gpu905', 'gpu906'}, got


def test_watcher_submission_excludes_the_list_read_now(tmp_path, monkeypatch, live):
    live.write_text('gpu999\n')
    monkeypatch.setattr(C, 'WATCH_DIR', tmp_path)
    monkeypatch.setattr(C, 'watcher_jobs', lambda *a, **k: [])
    seen = {}
    monkeypatch.setattr(C, 'plan_bids', lambda n, c, m, ladder, exclude, status=None: seen.update(ex=exclude) or [(36, 1)])

    def runner(argv, **kw):
        seen['sub'] = Path(argv[-1]).read_text()
        return types.SimpleNamespace(returncode=0, stdout='1 job(s) submitted to cluster 5.', stderr='')
    C.ensure_watcher(owner='me', runner=runner, status='')
    assert '(TARGET.UtsnameNodename != "gpu999")' in seen['sub'] and 'gpu999' in seen['ex']


def test_no_submitter_binds_the_list_at_import():
    """Every submission path reads the list at submit time, in every package but m5/ (finished campaigns, own lists):
    no module imports BAD_NODES by name or reads condor_cost.BAD_NODES (outside condor_cost itself), no module binds
    bad_nodes() / excluded() / exclude_nodes() at import (a module-level assignment or a default argument), eval_q35's
    own list is only read inside excluded(), and no shell script reads BAD_NODES."""
    import ast
    repo = Path(C.__file__).resolve().parents[1]
    readers = {'bad_nodes', 'excluded', 'exclude_nodes'}
    calls = lambda node: [n for n in ast.walk(node) if isinstance(n, ast.Call) and
                          getattr(n.func, 'id', getattr(n.func, 'attr', None)) in readers]
    bad = []
    import subprocess
    r = subprocess.run(['git', '-C', str(repo), 'ls-files'], capture_output=True, text=True)
    tracked = [repo / x for x in r.stdout.split()] if r.returncode == 0 and r.stdout.strip() else None
    tops = sorted(p.name for p in repo.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name != 'm5')
    for top in tops:
        files = ([p for p in tracked if p.parts[len(repo.parts)] == top] if tracked is not None   # tracked files only
                 else list((repo / top).rglob('*')))
        for p in sorted(x for x in files if x.suffix == '.py'):
            rel = str(p.relative_to(repo))
            if p.name.startswith('test_') or '/tests/' in rel or rel in ('tbshared/condor_cost.py',
                                                                         'aa_evals/common/condor_cost.py'):
                continue
            try:
                tree = ast.parse(p.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and any(a.name == 'BAD_NODES' for a in n.names):
                    bad.append(f'{rel}:{n.lineno} imports BAD_NODES')
                if isinstance(n, ast.Attribute) and n.attr == 'BAD_NODES':
                    bad.append(f'{rel}:{n.lineno} reads .BAD_NODES')
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for d in n.args.defaults + n.args.kw_defaults:
                        if d is not None and calls(d):
                            bad.append(f'{rel}:{n.lineno} binds a bad-node list in a default argument')
            for stmt in tree.body:   # module level
                if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and calls(stmt):
                    bad.append(f'{rel}:{stmt.lineno} binds a bad-node list at import')
            if rel == 'eval_q35/submit.py':
                ok = {id(x) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == 'excluded'
                      for x in ast.walk(f)}
                bad += [f'{rel}:{n.lineno} reads BAD_NODES outside excluded()' for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and n.id == 'BAD_NODES' and isinstance(n.ctx, ast.Load) and id(n) not in ok]
        for p in sorted(x for x in files if x.suffix == '.sh'):
            code = [l for l in p.read_text(errors='replace').splitlines() if not l.lstrip().startswith('#')]
            if any('BAD_NODES' in l for l in code):
                bad.append(f'{p.relative_to(repo)} reads BAD_NODES (use bad_nodes())')
    assert not bad, bad

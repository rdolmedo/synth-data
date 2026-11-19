"""tbshared.ledger: run segments from synthetic HTCondor event logs, floors, costs, stale ends, grouping, the CLI."""
import datetime as dt
import io
import json
import os
import shutil
import time
import types

import pytest

from tbshared import condor_cost
from tbshared import ledger as L

CET = dt.timezone(dt.timedelta(hours=1))
SUBMIT_HOST = ('<10.15.2.1:9618?PrivAddr=%3c10.22.0.101:9618%3fsock%3dschedd_1_2%3e&PrivNet=internal.cluster.is.'
               'localnet&addrs=10.15.2.1-9618&alias=sched.cluster.internal&noUDP&sock=schedd_1_2>')


def ev(code, job, t, head, *body, day='2026-11-10'):
    cl, pr = job.split('.')
    return f'{code} ({cl}.{int(pr):03d}.000) {day} {t} {head}\n' + ''.join(b + '\n' for b in body) + '...\n'


def submitted(job, t):
    return ev('000', job, t, f'Job submitted from host: {SUBMIT_HOST}')


def executing(job, t, host, day='2026-11-10'):
    n = int(host[1:])
    return ev('001', job, t, f'Job executing on host: <172.22.2.{n}:9618?addrs=172.22.2.{n}-9618&alias={host}.'
                             f'internal.cluster.internal&noUDP&sock=startd_5395_1988>', day=day)


def image(job, t, mb):
    return ev('006', job, t, f'Image size of job updated: {mb * 1000}', f'\t{mb}  -  MemoryUsage of job (MB)',
              f'\t{mb * 1000}  -  ResidentSetSize of job (KB)')


def table(cpus, mem_mb, mem_use, gpus=0, gpu_id=None):
    gpu = (f'\t   Gpus (Average)       :                  {gpus}         {gpus} "{gpu_id}"' if gpu_id else
           f'\t   Gpus (Average)       :                  {gpus}         {gpus} ')
    return ['\tPartitionable Resources :    Usage   Request Allocated' + (' Assigned' if gpu_id else ' '),
            f'\t   Cpus                 :     0.15        {cpus:>2}        {cpus:>2} ',
            '\t   Disk (KB)            :     1    209715200 223454124 ',
            gpu,
            f'\t   Memory (MB)          :  {mem_use:>4}       {mem_mb}    {mem_mb} ']


def terminated(job, t, usr, sys_, tab, rv=0):
    return ev('005', job, t, 'Job terminated.', f'\t(1) Normal termination (return value {rv})',
              f'\t\tUsr {usr}, Sys {sys_}  -  Run Remote Usage',
              '\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Run Local Usage',
              f'\t\tUsr {usr}, Sys {sys_}  -  Total Remote Usage',
              '\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Total Local Usage',
              '\t0  -  Run Bytes Sent By Job', '\t0  -  Run Bytes Received By Job', *tab, '',
              '\tJob terminated of its own accord at 2026-11-10T10:00:00Z with exit-code 0.')


def evicted(job, t, usr, sys_, tab):
    return ev('004', job, t, 'Job was evicted.', '\t(0) CPU times', f'\t\tUsr {usr}, Sys {sys_}  -  Run Remote Usage',
              '\t\tUsr 0 00:00:00, Sys 0 00:00:00  -  Run Local Usage', '\t0  -  Run Bytes Sent By Job',
              '\t0  -  Run Bytes Received By Job', *tab)


def memory_hold(job, t, host, limit, peak):
    """A memory hold as this cluster writes it: a 007 shadow exception and a 012 hold in the same second."""
    msg = (f'\tError from slot1_2@{host}.internal.cluster.internal: Job has gone over memory limit of {limit} '
           f'megabytes. Peak usage: {peak} megabytes.')
    return (ev('007', job, t, 'Shadow exception!', msg, '\t0  -  Run Bytes Sent By Job',
               '\t0  -  Run Bytes Received By Job') + ev('012', job, t, 'Job was held.', msg, '\tCode 34 Subcode 0'))


LOG = ''.join([
    submitted('100.0', '09:00:00'), submitted('100.1', '09:00:00'), submitted('101.0', '09:00:05'),
    submitted('102.0', '09:00:10'), submitted('103.0', '09:00:15'),
    # 100.0: one clean run on gpu101 (floor 15), 2 h, 1.5 CPU-h used, peak 3000 MB (006) vs 2941 table usage
    executing('100.0', '10:00:00', 'gpu101'), image('100.0', '10:05:00', 500), image('100.0', '10:10:00', 3000),
    terminated('100.0', '12:00:00', '0 01:00:00', '0 00:30:00', table(16, 262144, 2941)),
    # 100.1: memory hold on cpu401 (B200), released, evicted on gpu102, then running again (open) on gpu103
    executing('100.1', '10:00:00', 'cpu401'), image('100.1', '10:30:00', 230000),
    memory_hold('100.1', '11:00:00', 'cpu401', 235929, 260000),
    ev('013', '100.1', '11:05:00', 'Job was released.', '\tvia periodic_release'),
    executing('100.1', '11:10:00', 'gpu102'),
    evicted('100.1', '12:10:00', '0 00:20:00', '0 00:10:00', table(16, 262144, 1000)),
    executing('100.1', '12:30:00', 'gpu103'), image('100.1', '12:35:00', 800),
    # 101.0: a GPU job on cpu301 (B200), 1 h: units 16 + 24
    executing('101.0', '10:00:00', 'cpu301'),
    terminated('101.0', '11:00:00', '0 00:10:00', '0 00:00:00', table(16, 262144, 5386, 1, 'GPU-060dad9b'), rv=3),
    # 102.0: a second execute without an end event, then removed while running
    executing('102.0', '10:00:00', 'gpu101'), executing('102.0', '10:30:00', 'gpu101'),
    ev('009', '102.0', '11:00:00', 'Job was aborted.', '\tvia condor_rm (by user someone)'),
    # 103.0: a host missing from the floors snapshot, no request anywhere, a shadow exception that is not a hold
    executing('103.0', '10:00:00', 'x999'),
    ev('007', '103.0', '10:20:00', 'Shadow exception!', '\tError from slot1_4@x999: some other error',
       '\t0  -  Run Bytes Sent By Job', '\t0  -  Run Bytes Received By Job'),
])

# condor_status -af UtsnameNodename MinRunningPrice CUDACapability: one line per slot
FLOORS = '\n'.join(['gpu101 15 8.0', 'gpu101 15 undefined', 'gpu102 25 undefined', 'gpu102 25 8.0', 'gpu103 25 undefined',
                    'cpu401 55 10.0', 'cpu401 55 undefined', 'cpu301 55 10.0', 'hgpu101 35 9.0', 'c001 5 undefined',
                    'gpu088 15 7.5'])


def at(clock, tz=CET, day='2026-11-10'):
    return dt.datetime.fromisoformat(f'{day} {clock}').replace(tzinfo=tz).timestamp()


def proc(stdout='', rc=0):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr='')


@pytest.fixture
def log(tmp_path):
    p = tmp_path / 'run' / 'workers' / 'condor.log'
    p.parent.mkdir(parents=True)
    p.write_text(LOG)
    return p


def by(segs):
    return {(s['job'], s['host']): s for s in segs}


def test_parse_segments(log):
    segs = L.parse_log(log, tz=CET)
    assert [(s['job'], s['host'], s['end_how']) for s in segs] == [
        ('100.0', 'gpu101', 'terminated'), ('100.1', 'cpu401', 'shadow-exception'), ('101.0', 'cpu301', 'terminated'),
        ('102.0', 'gpu101', 'restart-no-end'), ('103.0', 'x999', 'shadow-exception'), ('102.0', 'gpu101', 'aborted'),
        ('100.1', 'gpu102', 'evicted'), ('100.1', 'gpu103', 'open')]
    s = by(segs)
    a = s['100.0', 'gpu101']
    assert (a['start'], a['end'], a['submit']) == (at('10:00:00'), at('12:00:00'), at('09:00:00'))
    assert (a['cpus'], a['mem_mb'], a['gpus'], a['req_src']) == (16, 262144, 0, 'table')
    assert a['cpu_s'] == 5400 and a['peak_mem_mb'] == 3000 and a['return_value'] == 0 and a['end_event'] == '005'
    assert not a['held']
    h = s['100.1', 'cpu401']
    # the 007 + 012 memory hold: closed by the 007, held; memory from the limit (0.9 x the request), CPUs later
    assert h['end_event'] == '007' and h['held'] and h['holds'] == 1 and h['end'] == at('11:00:00')
    assert (h['hold_limit_mb'], h['hold_peak_mb'], h['mem_mb'], h['cpus']) == (235929, 260000, 262144, 16)
    assert h['req_src'] == 'later' and h['cpu_s'] is None and h['peak_mem_mb'] == 230000
    e = s['100.1', 'gpu102']
    assert e['end_event'] == '004' and e['cpu_s'] == 1800 and e['req_src'] == 'table' and e['peak_mem_mb'] == 1000
    o = s['100.1', 'gpu103']
    assert o['end'] is None and o['end_event'] == 'open' and o['req_src'] == 'earlier' and o['cpus'] == 16
    g = s['101.0', 'cpu301']
    assert (g['gpus'], g['cpus'], g['return_value'], g['peak_mem_mb']) == (1, 16, 3, 5386)
    r = [x for x in segs if x['job'] == '102.0']
    assert [(x['start'], x['end']) for x in r] == [(at('10:00:00'), at('10:30:00')), (at('10:30:00'), at('11:00:00'))]
    assert all(x['req_src'] == 'unknown' and x['cpus'] is None for x in r)
    u = s['103.0', 'x999']
    assert u['end_event'] == '007' and u['req_src'] == 'unknown' and not u['held']


def test_resource_table_columns():
    head = '\tPartitionable Resources :    Usage   Request Allocated '
    body = [head, '\t   Cpus                 :' + ' ' * 27 + '16 ',                # Allocated only
            '\t   Memory (MB)          :  2941       327680    327680 ',
            '\t   Gpus (Average)       :                  0         0 ']
    assert L._resources(body) == ({'cpus': 16, 'mem_mb': 327680, 'gpus': 0}, 2941, None)
    assert L._resources(['\t   Cpus                 :     0.15        16        16 '])[0] == {'cpus': 16}
    assert L._resources(['\t   Gpus (Average)       :                  1         1 "GPU-1"'])[0] == {'gpus': 1}


def test_orphan_end_event_warns(tmp_path):
    p = tmp_path / 'condor.log'
    p.write_text(terminated('9.0', '12:00:00', '0 00:00:01', '0 00:00:00', table(8, 131072, 10)) +
                 ev('009', '9.1', '12:00:00', 'Job was aborted.', '\tvia condor_rm'))
    msgs = []
    assert L.parse_log(p, warn=msgs.append) == []
    assert len(msgs) == 1 and '1 end event(s)' in msgs[0] and '9.0' in msgs[0]     # a 009 of an idle job is normal


def test_later_cluster_requests_and_timezones(tmp_path):
    p = tmp_path / 'condor.log'
    p.write_text(executing('7.0', '10:00:00', 'gpu101') + ev('024', '7.0', '10:30:00', 'Job reconnection failed',
                 '    Job not found at execution machine') + executing('7.0', '11:00:00', 'gpu101') +
                 terminated('7.0', '12:00:00', '0 00:00:01', '0 00:00:00', table(8, 131072, 10)) +
                 executing('7.1', '12:00:00', 'gpu102'))
    first, second, sibling = L.parse_log(p, tz=dt.timezone.utc)
    assert first['end_how'] == 'reconnect-failed' and first['req_src'] == 'later' and first['cpus'] == 8
    assert second['req_src'] == 'table'
    assert sibling['job'] == '7.1' and sibling['req_src'] == 'cluster' and sibling['mem_mb'] == 131072
    assert first['start'] == dt.datetime(2026, 11, 10, 10, tzinfo=dt.timezone.utc).timestamp()
    assert L.parse_log(p)[0]['start'] == first['start'] - 3600              # default: Europe/Berlin, CET in Nov
    assert L.parse_log(p, tz=CET)[0]['start'] == first['start'] - 3600
    assert L.parse_log(p, tz='local')[0]['start'] == time.mktime((2026, 11, 10, 10, 0, 0, 0, 0, -1))
    summer = tmp_path / 'summer.log'
    summer.write_text(executing('8.0', '10:00:00', 'gpu101', day='2026-07-01'))
    assert L.parse_log(summer)[0]['start'] == at('08:00:00', dt.timezone.utc, '2026-07-01')   # CEST = UTC+2
    q = tmp_path / 'zoned.log'
    q.write_text(executing('8.0', '10:00:00', 'gpu101').replace('2026-11-10 10:00:00', '2026-11-10T10:00:00Z'))
    assert L.parse_log(q, tz=CET)[0]['start'] == first['start']                           # an explicit zone wins
    assert L.parse_tz('+01:00') == CET and L.parse_tz('UTC') == dt.timezone.utc and L.parse_tz('local') is None
    assert str(L.parse_tz('cluster')) == 'Europe/Berlin' == str(L.parse_tz('Europe/Berlin'))
    for bad in ('bogus', 'Mars/Olympus', '+1:2:3'):
        with pytest.raises(ValueError):
            L.parse_tz(bad)
    assert L.parse_time('2026-11-10T10:00') == first['start'] == L.parse_time('2026-11-10T11:00:00+01:00')


def test_floors_and_kinds(tmp_path):
    f = L.parse_floors(FLOORS)
    assert f['gpu101'] == (15, 'A100') and f['gpu102'] == (25, 'A100') and f['gpu103'] == (25, 'CPU')
    assert f['cpu401'] == (55, 'B200') and f['hgpu101'] == (35, 'H100') and f['gpu088'] == (15, 'RTX6000')
    snap = '"gpu100.internal.cluster.internal" 25 8.0 8 128.0 2051936 "NVIDIA A100-SXM4-80GB"\n'
    assert L.parse_floors(snap) == {'gpu100': (25, 'A100')}                  # the c30 nodes.txt format
    assert L.node_floor('gpu101', f) == (15, 'A100', 'status')
    for host in ('cpu305', 'gpu181', 'x999', None):                             # never guessed from the name
        assert L.node_floor(host, f) == (None, 'unknown', 'unknown')
    new, old = tmp_path / 'new.txt', tmp_path / 'old.txt'
    new.write_text('gpu101 25 8.0\n')
    old.write_text('gpu101 15 8.0\ncpu305 55 10.0\n')
    assert L.load_floors(new, old) == {'gpu101': (25, 'A100'), 'cpu305': (55, 'B200')}     # the first file wins
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return proc(FLOORS)
    assert L.floors_from_status(runner) == f
    assert calls == [['condor_status', '-af', 'UtsnameNodename', 'MinRunningPrice', 'CUDACapability']]


def test_units_match_condor_cost_and_hold_limits():
    for shape in ((16, 256, 0), (8, 300, 0), (16, 256, 1), (1, 2, 0)):
        assert L.units(*shape) == condor_cost.units(*shape)
    assert [L.request_from_limit(x) for x in (294912, 235929, 29491, 117964)] == [327680, 262144, 32768, 131072]


def test_segment_cost_clipping_and_prices(log):
    f = L.parse_floors(FLOORS)
    s = by(L.parse_log(log, tz=CET))
    a = s['100.0', 'gpu101']
    c = L.segment_cost(a, f)
    assert (c['hours'], c['units'], c['unit_h'], c['credits_floor']) == (2, 16, 32, 480)
    assert c['cpu_used_h'] == 1.5 and c['cpu_req_h'] == 32 and c['credits_bid'] is None
    assert c['mem_peak_gbh'] == pytest.approx(3000 / 1024 * 2) and c['mem_req_gbh'] == 512
    w = L.segment_cost(a, f, since=at('11:00:00'), until=at('11:30:00'))
    assert w['hours'] == 0.5 and w['credits_floor'] == 0.5 * 16 * 15
    assert w['cpu_used_h'] == pytest.approx(1.5 / 4) and w['cpu_req_h'] == 8                  # prorated
    assert L.segment_cost(a, f, since=at('12:00:00'))['hours'] == 0
    assert L.segment_cost(a, f, bid=25)['credits_bid'] == 2 * 16 * 25
    assert L.segment_cost(a, f, bid=5)['credits_bid'] == 2 * 16 * 15         # a bid below the floor was raised
    g = L.segment_cost(s['101.0', 'cpu301'], f)
    assert (g['units'], g['gpu_h'], g['credits_floor'], g['node_kind']) == (40, 1, 40 * 55, 'B200')
    o = s['100.1', 'gpu103']
    assert L.segment_cost(o, f, until=at('13:30:00'))['hours'] == 1                           # open: to until
    assert L.segment_cost(o, f, now=at('13:00:00'))['hours'] == 0.5                           # ... or to now
    assert L.segment_cost(o, f, until=at('14:00:00'), now=at('13:00:00'))['hours'] == 0.5
    u = L.segment_cost(s['103.0', 'x999'], f)
    assert u['units'] is None and u['credits_floor'] == 0 and u['floor_src'] == 'unknown' and u['hours'] == 1 / 3
    known_req = dict(s['103.0', 'x999'], cpus=4, mem_mb=16384)
    x = L.segment_cost(known_req, f, bid=26)
    assert x['floor'] is None and x['credits_floor'] == 0 and x['credits_bid'] == pytest.approx(4 / 3 * 26)


def _touch(path, t):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    os.utime(path, (t, t))


def test_close_stale_open_segments(log):
    d = log.parent
    segs = L.annotate(L.parse_log(log, tz=CET))
    s = by(segs)
    o = s['100.1', 'gpu103']
    # queued, not running: end at EnteredCurrentStatus
    st = L.close_stale(segs, {'100.1': (5, at('13:15:00'))}, now=at('20:00:00'))
    assert (o['end'], o['end_how'], o['end_src'], st['open_queue']) == (at('13:15:00'), 'open-stale', 'queue', 1)
    # running: stays open
    o.update(end=None, end_how='open', end_src=None)
    assert L.close_stale(segs, {'100.1': (2, at('12:30:00'))}, now=at('20:00:00'))['open_running'] == 1
    assert o['end'] is None
    # gone from the queue: the last .out/.err write
    _touch(d / 'worker.100.1.err', at('13:40:00'))
    st = L.close_stale(segs, {}, now=at('13:50:00'))
    assert (o['end'], o['end_src'], st['open_output']) == (at('13:40:00'), 'output', 1)
    # queue unknown: only output older than stale_after counts
    o.update(end=None, end_how='open', end_src=None)
    assert L.close_stale(segs, None, now=at('14:00:00'))['open_left'] == 1 and o['end'] is None
    assert L.close_stale(segs, None, now=at('15:00:00'))['open_output'] == 1 and o['end'] == at('13:40:00')
    # output older than the segment's last shadow event (006 at 12:35): end at the event
    _touch(d / 'worker.100.1.err', at('12:31:00'))
    o.update(end=None, end_how='open', end_src=None)
    L.close_stale(segs, {}, now=at('20:00:00'))
    assert o['last_seen'] == at('12:35:00') and o['end'] == at('12:35:00') and o['end_src'] == 'output'
    # gone, no output at all: stays open, marked
    os.unlink(d / 'worker.100.1.err')
    o.update(end=None, end_how='open', end_src=None)
    L.close_stale(segs, {}, now=at('20:00:00'))
    assert o['end'] is None and o['end_how'] == 'open-unresolved'


def test_close_stale_late_abort(log):
    segs = L.parse_log(log, tz=CET)
    r = [x for x in segs if x['job'] == '102.0'][-1]           # 10:30 -> 009 at 11:00
    _touch(log.parent / 'job.102.0.out', at('10:40:00'))
    st = L.close_stale(segs, {}, now=at('20:00:00'))
    assert st['aborted_late'] == 0 and st['aborted_kept'] == 1 and r['end'] == at('11:00:00')   # 20 min: kept
    _touch(log.parent / 'job.102.0.out', at('10:40:00'))
    segs2 = L.parse_log(log, tz=CET)
    r = [x for x in segs2 if x['job'] == '102.0'][-1]
    st = L.close_stale(segs2, {}, now=at('20:00:00'), stale_after=600)
    assert st['aborted_late'] == 1 and st['aborted_late_h'] == pytest.approx(1 / 3)
    assert (r['end'], r['end_logged'], r['end_how'], r['end_src']) == (at('10:40:00'), at('11:00:00'),
                                                                         'aborted-late', 'output')
    first = [x for x in segs2 if x['job'] == '102.0'][0]
    assert first['end'] == at('10:30:00') and 'end_src' not in first          # only the job's last segment


def test_late_abort_capped_at_last_shadow_event(tmp_path):
    p = tmp_path / 'condor.log'
    p.write_text(executing('5.0', '10:00:00', 'gpu101') + image('5.0', '10:50:00', 900) +
                 ev('022', '5.0', '10:55:00', 'Job disconnected, attempting to reconnect',
                    '    Socket between submit and execute hosts closed unexpectedly') +
                 ev('023', '5.0', '10:56:00', 'Job reconnected to slot1_1@gpu101') +
                 ev('009', '5.0', '12:00:00', 'Job was aborted.', '\tvia condor_rm (by user someone)'))
    _touch(tmp_path / 'pi.5.0.out', at('10:20:00'))          # silent on .out after 10:20
    (s,) = L.parse_log(p, tz=CET)
    assert s['last_seen'] == at('10:56:00')
    st = L.close_stale([s], {}, now=at('20:00:00'))
    assert st['aborted_late'] == 1 and s['end'] == at('10:56:00') and s['end_logged'] == at('12:00:00')
    (s,) = L.parse_log(p, tz=CET)
    os.unlink(tmp_path / 'pi.5.0.out')                       # no output files: kept at the 009, counted
    assert L.close_stale([s], {}, now=at('20:00:00'))['aborted_kept'] == 1 and s['end'] == at('12:00:00')


def test_cluster_zone_falls_back_to_system_zone(monkeypatch, capsys):
    monkeypatch.setattr(L, 'ZoneInfo', None)
    monkeypatch.setattr(L, '_WARNED', set())
    assert L.parse_tz('cluster') is None and L.parse_tz('cluster') is None
    assert capsys.readouterr().err.count('unavailable (no zoneinfo / tzdata)') == 1     # warned once
    with pytest.raises(ValueError):
        L.parse_tz('Europe/Berlin')                          # an explicit name still needs the zone

    def missing(key):
        raise OSError('no tzdata')
    monkeypatch.setattr(L, 'ZoneInfo', missing)
    assert L.parse_tz('cluster') is None


def test_summarize_groups(log):
    f = L.parse_floors(FLOORS)
    segs = L.parse_log(log, tz=CET)
    until = at('13:30:00')
    groups = L.summarize(segs, L.group_key('node-kind'), floors=f, until=until)
    assert set(groups) == {'A100', 'B200', 'CPU', 'unknown'}
    a = groups['A100']        # 100.0 2 h at 15, 102.0 2 x 0.5 h unknown, 100.1 evicted 1 h at 25
    assert (a['segments'], a['jobs'], a['job_h'], a['unknown_req_h']) == (4, 3, 4, 1)
    assert a['credits_floor'] == 2 * 16 * 15 + 1 * 16 * 25
    assert a['cpu_used_h'] == 2 and a['cpu_req_h'] == 48 and a['cpu_used_frac'] == pytest.approx(2 / 48)
    b = groups['B200']        # 100.1 held (007 + 012) after 1 h, 101.0 GPU 1 h
    assert (b['job_h'], b['gpu_h'], b['unit_h'], b['held']) == (2, 1, 56, 1)
    assert b['credits_floor'] == 56 * 55 and b['peak_mem_max_gb'] == pytest.approx(230000 / 1024)
    c = groups['CPU']
    assert (c['open'], c['open_h'], c['job_h'], c['open_credits_floor']) == (1, 1, 1, 16 * 25)
    assert groups['unknown']['unknown_floor_h'] == pytest.approx(1 / 3) and groups['unknown']['held'] == 0
    tot = L.summarize(segs, floors=f, until=until)['TOTAL']
    assert tot['job_h'] == pytest.approx(4 + 2 + 1 + 1 / 3) and tot['jobs'] == 5
    assert tot['credits_floor'] == sum(g['credits_floor'] for g in groups.values())
    both = L.summarize(segs, L.group_key('node-kind,floor'), floors=f, until=until)
    assert {'A100 / floor 15', 'A100 / floor 25', 'B200 / floor 55', 'unknown / floor ?'} <= set(both)
    win = L.summarize(segs, floors=f, since=at('11:30:00'), until=until)['TOTAL']
    assert win['segments'] == 3 and win['job_h'] == pytest.approx(0.5 + 2 / 3 + 1)            # 100.0, 100.1 x2
    rows = L.costed(segs, f, until=until)
    assert L.summarize(rows)['TOTAL'] == tot                                   # costed rows as they are
    assert L.summarize(rows, since=at('11:30:00'), floors=f, until=until)['TOTAL'] == win     # ... or re-costed
    assert L.summarize(rows, floors={}, until=until)['TOTAL']['credits_floor'] == 0


def test_annotate_from_jobs_json_and_submit_guess(log):
    run = log.parent.parent
    sub = run / 'jobs' / 'workers.sub'
    sub.parent.mkdir()
    sub.write_text('request_cpus = 16\nrequest_memory = 262144\nrequest_gpus = 0\n+JobBatchName = "q35-workers-x"\n'
                   'queue 2\n')
    os.utime(sub, (at('08:59:00'),) * 2)
    (run / 'jobs.json').write_text(json.dumps({'jobs': [{'cluster': '100', 'bid': 25, 'submit_file': str(sub)},
                                                        {'cluster': '102', 'bid': 36, 'role': 'workers'}]}))
    other = log.parent / 'other.sub'
    other.write_text('request_cpus = 2\nrequest_memory = 8G\nbatch_name = guessed\nqueue\n')
    os.utime(other, (at('09:01:00'),) * 2)
    s = by(L.annotate(L.parse_log(log, tz=CET)))
    assert s['100.0', 'gpu101']['batch'] == 'q35-workers-x' and s['100.0', 'gpu101']['bid'] == 25
    assert not s['100.0', 'gpu101']['submit_after']
    r = s['102.0', 'gpu101']           # a record without a submit file: the *.sub next to the log, picked by time
    assert r['bid'] == 36 and r['req_src'] == 'submit-guess' and (r['cpus'], r['mem_mb']) == (2, 8192)
    assert r['batch'] == 'guessed'
    assert s['103.0', 'x999']['bid'] is None and s['103.0', 'x999']['req_src'] == 'submit-guess'


def test_read_submit_takes_the_base_of_the_memory_net(tmp_path):
    from tbshared import condor_cost
    sub = tmp_path / 'w.sub'
    sub.write_text(condor_cost.with_auto_release('request_cpus = 4\nrequest_memory = 49152\n') + 'queue 2\n')
    r = L.read_submit(sub)
    assert (r['cpus'], r['mem_mb']) == (4, 49152)
    sub2 = tmp_path / 'w2.sub'
    sub2.write_text('request_cpus = 4 + 1\nrequest_memory = $(mem)\n')              # unparseable: left unknown
    assert 'cpus' not in L.read_submit(sub2) and 'mem_mb' not in L.read_submit(sub2)


def test_annotate_ignores_requests_of_rewritten_submit_files(log):
    run = log.parent.parent
    sub = run / 'jobs' / 'workers.sub'
    sub.parent.mkdir()
    sub.write_text('request_cpus = 2\nrequest_memory = 8192\nbatch_name = rewritten\nqueue\n')
    os.utime(sub, (at('09:30:00'),) * 2)                                     # after the jobs were submitted
    (run / 'jobs.json').write_text(json.dumps([{'cluster': '102', 'submit_file': str(sub)}]))
    guess = log.parent / 'late.sub'
    guess.write_text('request_cpus = 4\nrequest_memory = 8192\nqueue\n')
    os.utime(guess, (at('09:30:00'),) * 2)
    L._CACHE.clear()
    s = by(L.annotate(L.parse_log(log, tz=CET)))
    r = s['102.0', 'gpu101']
    assert r['submit_after'] and r['req_src'] == 'unknown' and r['cpus'] is None and r['batch'] == 'rewritten'
    u = s['103.0', 'x999']           # the only *.sub was written after submission: no request from it
    assert u['submit_file'] == str(guess) and u['submit_after'] and u['req_src'] == 'unknown'


def test_load_dedupes_copied_logs(log, tmp_path):
    copy = tmp_path / 'copy' / 'condor.log'
    copy.parent.mkdir()
    shutil.copy(log, copy)
    one = L.load([str(log)], tz=CET, with_submit=False)
    both = L.load([str(tmp_path / '*' / 'workers' / 'condor.log'), str(copy), str(log)], tz=CET, with_submit=False)
    assert len(both) == len(one) == 8
    assert L.expand([str(tmp_path / 'nope.log')]) == []


class Runner:
    def __init__(self, queue=''):
        self.queue, self.calls = queue, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] == 'condor_status':
            return proc(FLOORS)
        if argv[0] == 'condor_q':
            return proc(self.queue)
        raise AssertionError(argv)


def test_cli_table_json_and_warnings(log, tmp_path):
    floors = tmp_path / 'floors.txt'
    floors.write_text(FLOORS)
    out_json = tmp_path / 'out.json'
    buf, err, r = io.StringIO(), io.StringIO(), Runner()
    rc = L.main([str(tmp_path / '*' / 'workers' / 'condor.log'), '--tz', '+01:00', '--since', '2026-11-10T09:00Z',
                 '--until', '2026-11-10T12:30Z', '--group', 'node-kind,floor', '--floors', str(floors),
                 '--json', str(out_json)], runner=r, out=buf, err=err)
    text = buf.getvalue()
    assert rc == 0 and '1 logs, 8 segments' in text and 'A100 / floor 15' in text
    assert r.calls == [['condor_q', '100', '-af', 'ClusterId', 'ProcId', 'JobStatus', 'EnteredCurrentStatus']]
    tot_line = [x for x in text.splitlines() if x.startswith('TOTAL')][0]
    assert tot_line.split()[1:3] == ['8', '5']
    # not in the queue, no output: still open, with a warning giving its share
    assert '1 segment(s) are still open' in err.getvalue() and '1.0 job-h (14% of job-h)' in err.getvalue()
    assert '0.3 job-h on hosts in no floors snapshot' in err.getvalue()
    d = json.loads(out_json.read_text())
    assert d['total']['job_h'] == pytest.approx(4 + 2 + 1 + 1 / 3)       # 12:30Z = 13:30 CET
    assert len(d['segments']) == 8 and d['groups']['B200 / floor 55']['gpu_h'] == 1
    assert d['total']['held'] == 1
    # a saved queue snapshot ends the open segment at its EnteredCurrentStatus
    q = tmp_path / 'queue.txt'
    q.write_text(f"100 1 1 {at('13:00:00'):.0f}\n")
    buf, err = io.StringIO(), io.StringIO()
    assert L.main([str(log), '--tz', 'Europe/Berlin', '--until', '2026-11-10T12:30Z', '--floors', str(floors),
                   '--queue', str(q), '--json', str(out_json)], runner=Runner(), out=buf, err=err) == 0
    d = json.loads(out_json.read_text())
    assert d['total']['open'] == 0 and d['total']['fixed'] == 1
    assert d['total']['job_h'] == pytest.approx(4 + 2 + 0.5 + 1 / 3)
    assert 'ended 1 open segment(s) at their queue' in err.getvalue()
    assert 'kept 1 at the 009 time' in err.getvalue()                       # 102.0: a bare 009, no output files
    # live floors through the runner, saved as a snapshot; --live with an older file
    snap = tmp_path / 'snap.txt'
    buf = io.StringIO()
    assert L.main([str(log), '--until', '2026-11-10T14:00Z', '--group', 'batch', '--save-floors', str(snap)],
                  runner=Runner(), out=buf, err=io.StringIO()) == 0
    assert snap.read_text() == FLOORS and '(no batch)' in buf.getvalue() and 'condor_status (live)' in buf.getvalue()
    old = tmp_path / 'old.txt'
    old.write_text('x999 5 undefined\n')
    buf, err = io.StringIO(), io.StringIO()
    assert L.main([str(log), '--until', '2026-11-10T14:00Z', '--live', '--floors', str(old), '--no-queue',
                   '--group', 'node-kind'], runner=Runner(), out=buf, err=err) == 0
    assert 'condor_status (live), ' in buf.getvalue() and 'no floors snapshot' not in err.getvalue()
    buf = io.StringIO()
    assert L.main([str(log), '--until', '2026-11-09', '--floors', str(floors)], runner=Runner(), out=buf,
                  err=io.StringIO()) == 0
    assert 'no run time inside the window' in buf.getvalue()
    assert L.main([str(tmp_path / 'none*.log')], runner=Runner(), out=io.StringIO(), err=io.StringIO()) == 1
    for bad in (['--group', 'bogus'], ['--tz', 'Mars/Olympus'], ['--floors', str(floors), '--save-floors', 'x']):
        with pytest.raises(SystemExit):
            L.main([str(log), *bad], runner=Runner(), out=io.StringIO(), err=io.StringIO())

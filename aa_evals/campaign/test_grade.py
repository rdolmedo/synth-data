"""aa_evals.campaign.grade: collectors, grading steps and the event-driven judge, with a fake condor (no jobs)."""
import json
import re
import time
import types
from pathlib import Path

import pytest

from aa_evals.campaign import grade as GR
from aa_evals.campaign import launch as L
from aa_evals.campaign.test_launch import campaign
from aa_evals.common import condor_cost as CC


class FakeCondor:
    """condor_q -json / condor_submit_bid / condor_rm / wrappers, recorded."""

    def __init__(self):
        self.jobs, self.calls, self.next, self.wrapper_out = [], [], 1000, {}

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        ok = lambda out='': types.SimpleNamespace(returncode=0, stdout=out, stderr='')  # noqa: E731
        if argv[0] == 'condor_q' and '-json' in argv:
            return ok(json.dumps(self.jobs))
        if argv[0] == 'condor_q':
            return ok('')
        if argv[0] == 'condor_submit_bid':
            self.next += 1
            sub = Path(argv[2]).read_text()
            batch = re.search(r'JobBatchName = "([^"]+)"', sub)
            self.jobs.append({'ClusterId': self.next, 'ProcId': 0, 'JobStatus': 1,
                              'JobBatchName': batch.group(1) if batch else None, 'Args': sub})
            return ok(f'1 job(s) submitted to cluster {self.next}.')
        if argv[0] == 'condor_rm':
            self.jobs = [j for j in self.jobs if str(j['ClusterId']) != argv[1]]
            return ok()
        if argv[0] == 'condor_status':
            return ok('')
        for k, v in self.wrapper_out.items():
            if k in ' '.join(argv):
                return types.SimpleNamespace(returncode=0, stdout=v, stderr='')
        return ok('')


@pytest.fixture
def G(tmp_path, monkeypatch):
    C = campaign(tmp_path)
    C.cfg['grading'] = {'runs_root': str(tmp_path / 'runs')}
    monkeypatch.setattr(L, 'REG', tmp_path / 'serving')
    monkeypatch.setattr(GR.condor_cost, 'ensure_watcher', lambda **kw: None)
    # free Cpus / Memory MB / Gpus, then the node's totals (tbshared 1.5 fills within them): TotalGpus TotalCpus TotalMemory CUDA
    monkeypatch.setattr(GR.condor_cost, 'condor_status_slots', lambda *a, **k: 'n1\t15\t64\t512000\t0\t0\t64\t524288\tundefined\n')
    return GR.Grading(C)


def pi_run(G, tmp_path, model, suite, stage='full', resolved=True):
    rid = G.C.run_id(model, suite, stage)
    pr = tmp_path / 'pi' / rid
    (pr / 'queue' / 'done').mkdir(parents=True)
    if resolved:
        (pr / 'queue' / 'done' / 'k1.json').write_text('{}')
    d = G.run_dir(suite, rid)
    d.mkdir(parents=True)
    (d / 'config.json').write_text(json.dumps({'harness': {'pi_run': str(pr)}}))
    return rid


def set_lanes(G, model, lanes, stage='full'):
    d = G.C.dispatch_dir(model, stage)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'state.json').write_text(json.dumps({'lanes': lanes}))


def test_mentions_matches_whole_run_ids_only():
    assert GR.mentions('aa-hle-c1-x-hle', 'c1-x-hle')
    assert GR.mentions('/data/runs/hle/c1-x-hle/job/argv.json', 'c1-x-hle')
    assert GR.mentions('aa-pi-critpt_pi-c1-x-critpt', 'c1-x-critpt')
    assert not GR.mentions('aa-hle-c1-x-hle-graded', 'c1-x-hle')
    assert not GR.mentions('aa-hle-c1-x-hle-canary', 'c1-x-hle')
    assert not GR.mentions('c1-x-hle2', 'c1-x-hle')


def test_clusters_of_skips_the_bid_ladder_watcher():
    out = ('bid-ladder watcher: cluster 77 (log /data/x)\n/data/r/job.sub: 1 job(s) at bid 26 (cluster 123); ...\n'
           '123.0  logs: /data/r/job/job.123.out\n1 job(s) submitted to cluster 124.\n')
    assert GR.clusters_of(out) == ['123', '124']


def test_log_outcome(tmp_path):
    log = tmp_path / 'c.log'
    log.write_text('000 (5.000.000) 2026-11-13 10:00:00 Job submitted from host: <x>\n...\n'
                   '005 (5.000.000) 2026-11-13 11:00:00 Job terminated.\n'
                   '\t(1) Normal termination (return value 0)\n'
                   '005 (6.000.000) 2026-11-13 11:00:00 Job terminated.\n'
                   '\tJob terminated of its own accord at 2026-11-13T10:44:01Z with exit-code 2.\n'
                   '009 (7.000.000) 2026-11-13 11:00:00 Job was aborted.\n\tvia condor_rm\n')
    assert GR.log_outcome(str(log), '5') == 'ok'
    assert GR.log_outcome(str(log), '6') == 'failed'
    assert GR.log_outcome(str(log), '7') == 'failed'
    assert GR.log_outcome(str(log), '8') is None


def test_collector_waits_for_a_resolved_attempt_then_submits_lean_on_the_ladder(G, tmp_path):
    fc = FakeCondor()
    for s in GR.PI_COLLECT:
        pi_run(G, tmp_path, 'gemma26', s, resolved=False)
    st = G.load()
    q = GR.Queue.fetch('me', fc)
    assert GR.collect_pass(G, q, st, ['gemma26'], 'full', False, fc, out=lambda *_: None)['gemma26'] == 'waiting'
    (tmp_path / 'pi' / G.C.run_id('gemma26', 'critpt', 'full') / 'queue' / 'done' / 'a.json').write_text('{}')
    r = GR.collect_pass(G, q, st, ['gemma26'], 'full', False, fc, out=lambda *_: None)
    assert r['gemma26'] == 'submitted'
    subs = [c for c in fc.calls if c[0] == 'condor_submit_bid']
    assert len(subs) == 1 and subs[0][1] == '26'                 # the bottom rung of the ladder
    sub = Path(subs[0][2]).read_text()
    assert CC.parse_shape(sub)[:2] == (1, 16.0)                  # 1 CPU / 4 GB, filled to 16 GB (same units)
    assert f'request_memory = 16384 + 4096 * {CC.MEMORY_HOLDS}\n' in sub   # the memory net on the filled base
    assert 'periodic_release' in sub and '+AaBidLadder' in sub
    assert f'JobBatchName = "aa-collect-c99-gemma26"' in sub
    assert str(G.dir).startswith(str(tmp_path)) and 'log = ' + str(G.collect_dir('gemma26', 'full')) in sub
    sh = (G.collect_dir('gemma26', 'full') / 'collect.sh').read_text()
    for s in GR.PI_COLLECT:
        assert f'collect {s} {GR.COLLECT_PY[s]} c99-gemma26-{s} ' in sh
    assert '--concurrency 2' in sh and '--follow' in sh and '.collected' in sh
    # live: no second job
    q = GR.Queue.fetch('me', fc)
    assert GR.collect_pass(G, q, st, ['gemma26'], 'full', False, fc, out=lambda *_: None)['gemma26'] == 'running'
    assert len([c for c in fc.calls if c[0] == 'condor_submit_bid']) == 1
    # every suite imported: done, whatever the queue says
    for s in GR.PI_COLLECT:
        (G.collect_dir('gemma26', 'full') / f'{s}.collected').write_text('x')
    assert GR.collect_pass(G, q, st, ['gemma26'], 'full', False, fc, out=lambda *_: None)['gemma26'] == 'done'


def test_collector_resubmits_only_unimported_suites_and_is_removed_after_the_grace(G, tmp_path, monkeypatch):
    fc = FakeCondor()
    for s in GR.PI_COLLECT:
        pi_run(G, tmp_path, 'qwen36', s)
    d = G.collect_dir('qwen36', 'full')
    d.mkdir(parents=True)
    (d / 'gdpval.collected').write_text('x')
    st = G.load()
    GR.collect_pass(G, GR.Queue.fetch('me', fc), st, ['qwen36'], 'full', False, fc, out=lambda *_: None)
    sh = (d / 'collect.sh').read_text()
    assert 'collect gdpval' not in sh and 'collect critpt' in sh and 'collect automationbench' in sh
    set_lanes(G, 'qwen36', {f'{GR.PI_BENCH[s]}.c99-qwen36-{s}': {'done': True} for s in GR.PI_COLLECT})
    q = GR.Queue.fetch('me', fc)
    assert GR.collect_pass(G, q, st, ['qwen36'], 'full', False, fc, out=lambda *_: None)['qwen36'] == 'running'
    st['collect']['qwen36.full']['lanes_done_at'] -= G.g['collect_grace'] + 1
    r = GR.collect_pass(G, q, st, ['qwen36'], 'full', False, fc, out=lambda *_: None)
    assert r['qwen36'].startswith('failed') and any(c[0] == 'condor_rm' for c in fc.calls)


def test_hle_grade_waits_for_generation_then_the_judge(G, tmp_path):
    fc = FakeCondor()
    rid = G.C.run_id('qwen36', 'hle', 'full')
    G.run_dir('hle', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'hle.{rid}': {'done': False}})
    st = G.load()
    run = lambda up: GR.grade_pass(G, GR.Queue.fetch('me', fc), st, ['qwen36'], 'full', up, False, fc,  # noqa: E731
                                   out=lambda *_: None)
    assert run(False)[('qwen36', 'hle')] == 'waiting: generation'
    set_lanes(G, 'qwen36', {f'hle.{rid}': {'done': True}})
    fc.jobs.append({'ClusterId': 5, 'JobStatus': 2, 'JobBatchName': f'aa-hle-{rid}'})   # the client still exits
    assert run(False)[('qwen36', 'hle')] == 'waiting: generation'
    fc.jobs.clear()
    assert run(False)[('qwen36', 'hle')] == 'needs judge'
    fc.wrapper_out['aa_evals.hle.submit'] = 'bid-ladder watcher: cluster 9 (log x)\n/x/job.sub: 1 job(s) at bid 26 (cluster 42); y\n42.0  logs: z\n'
    assert run(True)[('qwen36', 'hle')] == 'submitted'
    argv = next(c for c in fc.calls if 'aa_evals.hle.submit' in c)
    assert argv[argv.index('--run-id') + 1] == f'{rid}-graded' and '--force' not in argv
    assert argv[argv.index('--judge-arm') + 1] == 'qwen38-27b-medium-pin'
    rec = st['grade'][f'qwen36.full.hle']['grade']
    assert rec['clusters'] == ['42']
    fc.jobs.append({'ClusterId': 42, 'ProcId': 0, 'JobStatus': 2, 'JobBatchName': f'aa-hle-{rid}-graded',
                    'UserLog': str(tmp_path / 'hle.log')})
    assert run(True)[('qwen36', 'hle')] == 'running'
    fc.jobs.clear()
    (tmp_path / 'hle.log').write_text('005 (42.000.000) 2026-11-13 Job terminated.\n\t(1) Normal termination (return value 1)\n')
    assert run(True)[('qwen36', 'hle')] == 'submitted'              # failed once: resubmitted with --force
    assert '--force' in [c for c in fc.calls if 'aa_evals.hle.submit' in c][-1]
    (tmp_path / 'hle.log').write_text('005 (42.000.000) 2026-11-13 Job terminated.\n\t(1) Normal termination (return value 0)\n')
    assert run(True)[('qwen36', 'hle')] == 'done'


def test_critpt_reprobe_job_then_local_grade(G, tmp_path):
    fc = FakeCondor()
    pi_run(G, tmp_path, 'gemma26', 'critpt')
    st = G.load()
    run = lambda: GR.grade_pass(G, GR.Queue.fetch('me', fc), st, ['gemma26'], 'full', False, False, fc,  # noqa: E731
                                out=lambda *_: None)
    assert run()[('gemma26', 'critpt')] == 'waiting: import'
    d = G.collect_dir('gemma26', 'full')
    d.mkdir(parents=True)
    (d / 'critpt.collected').write_text('x')
    fc.wrapper_out['--reprobe'] = 'x/run.sub: 1 job(s) at bid 26 (cluster 77); y\n'
    assert run()[('gemma26', 'critpt')] == 'submitted'               # not judged: no judge needed
    (tmp_path / 'r.log').write_text('005 (77.000.000) 2026-11-13 Job terminated.\n\t(1) Normal termination (return value 0)\n')
    st['grade']['gemma26.full.critpt']['reprobe']['log'] = str(tmp_path / 'r.log')
    assert run()[('gemma26', 'critpt')] == 'done'
    assert any('--grade' in c for c in fc.calls)


def test_gdpval_judge_is_our_lean_job(G, tmp_path):
    fc = FakeCondor()
    rid = pi_run(G, tmp_path, 'gemma26', 'gdpval')
    d = G.collect_dir('gemma26', 'full')
    d.mkdir(parents=True)
    (d / 'gdpval.collected').write_text('x')
    st = G.load()
    r = GR.grade_pass(G, GR.Queue.fetch('me', fc), st, ['gemma26'], 'full', True, False, fc, out=lambda *_: None)
    assert r[('gemma26', 'gdpval')] == 'submitted'
    sub = Path(next(c for c in fc.calls if c[0] == 'condor_submit_bid')[2]).read_text()
    assert CC.parse_shape(sub)[:2] == (1, 16.0) and f'aa-grade-{rid}' in sub   # 1 CPU / 8 GB, filled to 16 GB
    sh = (G.dir / 'grade' / 'gemma26.full.gdpval' / 'judge.sh').read_text()
    assert 'aa_evals.gdpval.refs plan' in sh and 'aa_evals.gdpval.refs judge' in sh
    assert '--pool c99-refs-gemma26 ' in sh and 'm36-lightning-gdpval-full' in sh


def judge_setup(G, fresh=True):
    reg = L.REG / G.j['registry']
    reg.mkdir(parents=True, exist_ok=True)
    (reg / 'job.json').write_text(json.dumps({'job_id': '500.0'}))
    (reg / 'endpoints.json').write_text(json.dumps({'job_id': '500.0', 'updated_at': time.time() if fresh else 0,
                                                     'engine_urls': ['http://x:1/v1']}))


def test_judge_starts_only_for_waiting_work_and_never_twice(G):
    fc = FakeCondor()
    fc.wrapper_out['serve_teacher'] = 'Submitting job(s).\n1 job(s) submitted to cluster 500.\n500.0\n'
    st = G.load()
    q = GR.Queue.fetch('me', fc)
    assert 'not needed' in GR.judge_pass(G, q, st, False, False, False, fc, out=lambda *_: None)
    r = GR.judge_pass(G, q, st, True, True, False, fc, out=lambda *_: None)
    assert 'started' in r and st['judge']['jobs'] == ['500']
    argv = next(c for c in fc.calls if 'task_synthesis.serve.serve_teacher' in c)
    assert argv[argv.index('--name') + 1] == 'aa-judge-qwen38-27b' and argv[argv.index('--bid') + 1] == '61'
    fc.jobs.append({'ClusterId': 500, 'JobStatus': 1, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    n = len(fc.calls)
    r = GR.judge_pass(G, GR.Queue.fetch('me', fc), st, True, True, False, fc, out=lambda *_: None)
    assert r.startswith('ours:') and 'starting' in r
    assert not any('serve_teacher' in ' '.join(c) for c in fc.calls[n:])


def test_judge_release_after_idle_only_when_ours_and_unused(G, monkeypatch):
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs.append({'ClusterId': 500, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    st = G.load()
    st['judge']['jobs'] = ['500']
    q = GR.Queue.fetch('me', fc)
    no_ps = lambda: ''  # noqa: E731
    jp = lambda flight=lambda e: 0.0, ps=no_ps: GR.judge_pass(G, q, st, False, False, False, fc,  # noqa: E731
                                                              out=lambda *_: None, ps=ps, flight=flight)
    assert '3 requests in flight' in jp(lambda e: 3.0)
    assert 'idle 0 s' in jp()
    st['judge']['idle_since'] -= 601
    # someone else's grade job is live: kept
    fc.jobs.append({'ClusterId': 900, 'JobStatus': 2, 'JobBatchName': 'aa-hle-other-run-graded'})
    q = GR.Queue.fetch('me', fc)
    assert 'kept for other users' in jp()
    fc.jobs.pop()
    q = GR.Queue.fetch('me', fc)
    # a local process naming the registry: kept
    assert 'kept for other users' in jp(ps=lambda: '123 python refs.py judge --judge-endpoint serving:aa-judge-qwen38-27b\n')
    # another campaign's fresh lease: kept
    lease = L.REG / G.j['registry'] / 'leases' / 'c98.json'
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text('{}')
    assert 'kept for other users' in jp()
    lease.unlink()
    # judged work of ours queued: kept, idle timer reset
    assert 'judged work' in GR.judge_pass(G, q, st, False, True, False, fc, out=lambda *_: None, ps=no_ps)
    assert 'idle_since' not in st['judge']
    st['judge']['idle_since'] = time.time() - 601
    r = jp()
    assert 'released 500' in r and not r.startswith('ours:') and ['condor_rm', '500'] in fc.calls


def test_judge_not_started_by_us_is_never_released(G):
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs.append({'ClusterId': 501, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    st = G.load()
    st['judge']['idle_since'] = time.time() - 3600
    r = GR.judge_pass(G, GR.Queue.fetch('me', fc), st, False, False, False, fc, out=lambda *_: None,
                      ps=lambda: '', flight=lambda e: 0.0)
    assert 'not started by c99' in r and not any(c[0] == 'condor_rm' for c in fc.calls)


def test_tick_starts_the_judge_for_ready_work_and_status_writes_nothing(G, tmp_path, monkeypatch):
    fc = FakeCondor()
    fc.wrapper_out['serve_teacher'] = '1 job(s) submitted to cluster 600.\n'
    rid = G.C.run_id('qwen36', 'omniscience', 'full')
    G.run_dir('omniscience', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'omniscience.{rid}': {'done': True}})
    text = GR.status(G, ['qwen36'], 'full', runner=fc, ps=lambda: '', flight=lambda e: 0.0)
    assert 'omniscience' in text and 'needs judge' in text and not G.state_path.exists()
    assert not any(c[0] in ('condor_submit_bid', 'condor_rm') or 'serve_teacher' in ' '.join(c) for c in fc.calls)
    r = GR.tick(G, ['qwen36'], 'full', runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    assert r['grade'][('qwen36', 'omniscience')] == 'needs judge' and 'started' in r['judge']
    assert (L.REG / G.j['registry'] / 'leases' / 'c99.json').exists()
    assert not GR.finished(r)


def test_finished():
    assert GR.finished({'collect': {'a': 'done'}, 'grade': {('a', 'hle'): 'done', ('a', 'tb21'): 'self-graded'},
                        'judge': 'judge: released 5'})
    assert not GR.finished({'collect': {'a': 'done'}, 'grade': {('a', 'hle'): 'done'}, 'judge': 'ours: judge 5: idle 3 s'})
    assert not GR.finished({'collect': {'a': 'running'}, 'grade': {}, 'judge': 'judge: not needed, none live'})


# ------------------------------------------------------------------------------------------------ review regressions
def collected(G, model, suite, stage='full'):
    d = G.collect_dir(model, stage)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{suite}.collected').write_text('x')


def test_a_failure_later_in_a_pass_never_duplicates_earlier_submissions(G, tmp_path, monkeypatch):
    G.C.models['qwen36'] = dict(G.C.models['gemma26'], registry='c99-qwen36')
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs.append({'ClusterId': 500, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    for m in ('gemma26', 'qwen36'):
        pi_run(G, tmp_path, m, 'gdpval')
        collected(G, m, 'gdpval')
    real = GR.condor_cost.submit
    calls = {'n': 0}

    def flaky(sub, bid=None, **kw):
        calls['n'] += 1
        if 'qwen36' in str(sub):
            raise GR.condor_cost.SubmitError('schedd timeout')
        return real(sub, bid, **kw)
    monkeypatch.setattr(GR.condor_cost, 'submit', flaky)
    kw = dict(runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    r = GR.tick(G, ['gemma26', 'qwen36'], 'full', parts=('grade',), **kw)
    assert r['grade'][('gemma26', 'gdpval')] == 'submitted' and r['grade'][('qwen36', 'gdpval')].startswith('retry')
    for _ in range(3):
        GR.tick(G, ['gemma26', 'qwen36'], 'full', parts=('grade',), **kw)
    assert len([j for j in fc.jobs if j['JobBatchName'] == 'aa-grade-c99-gemma26-gdpval']) == 1
    # a live job of a step with no record (state lost) is adopted, not submitted again
    G.save({})
    r = GR.tick(G, ['gemma26'], 'full', parts=('grade',), **kw)
    assert r['grade'][('gemma26', 'gdpval')] == 'running'
    assert len([j for j in fc.jobs if j['JobBatchName'] == 'aa-grade-c99-gemma26-gdpval']) == 1


def test_gdppdf_graded_job_is_not_generation_and_keeps_the_judge(G, tmp_path):
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs.append({'ClusterId': 500, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    rid = G.C.run_id('qwen36', 'gdppdf', 'full')
    G.run_dir('gdppdf', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'gdppdf.{rid}': {'done': True}})
    st = G.load()
    st['judge']['jobs'] = ['500']
    G.save(st)
    fc.wrapper_out['aa_evals.gdppdf.submit'] = 'x/run.sub: 1 job(s) at bid 26 (cluster 61); y\n'
    kw = dict(runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    assert GR.tick(G, ['qwen36'], 'full', **kw)['grade'][('qwen36', 'gdppdf')] == 'submitted'
    fc.jobs.append({'ClusterId': 61, 'ProcId': 0, 'JobStatus': 1, 'JobBatchName': 'aa-gdppdf-run',
                    'Arguments': f'-u -m aa_evals.gdppdf.run --run-id {rid}-graded --stage grade --generations-from {rid}'})
    st = G.load()
    st['judge']['idle_since'] = time.time() - 3600
    G.save(st)
    r = GR.tick(G, ['qwen36'], 'full', **kw)
    assert r['grade'][('qwen36', 'gdppdf')] == 'running' and 'released' not in r['judge']
    assert not any(c[0] == 'condor_rm' for c in fc.calls)


def test_our_judged_jobs_of_another_stage_keep_the_judge(G, tmp_path):
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs += [{'ClusterId': 500, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'},
                {'ClusterId': 70, 'JobStatus': 1, 'JobBatchName': 'aa-hle-c99-gemma26-hle-canary-graded'}]
    G.save({'judge': {'jobs': ['500'], 'idle_since': time.time() - 3600},
            'grade': {'gemma26.canary.hle': {'grade': {'status': 'submitted', 'clusters': ['70']}}}})
    r = GR.tick(G, ['gemma26'], 'full', runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    assert 'judged work' in r['judge'] and not any(c[0] == 'condor_rm' for c in fc.calls)


def test_collector_grace_restarts_with_each_submission_and_with_progress(G, tmp_path):
    fc = FakeCondor()
    for s in GR.PI_COLLECT:
        pi_run(G, tmp_path, 'qwen36', s)
    set_lanes(G, 'qwen36', {f'{GR.PI_BENCH[s]}.c99-qwen36-{s}': {'done': True} for s in GR.PI_COLLECT})
    st = G.load()
    st['collect']['qwen36.full'] = {'submits': [], 'lanes_done_at': time.time() - 99999}   # an earlier collector's
    run = lambda: GR.collect_pass(G, GR.Queue.fetch('me', fc), st, ['qwen36'], 'full', False, fc,  # noqa: E731
                                  out=lambda *_: None)
    assert run()['qwen36'] == 'submitted' and 'lanes_done_at' not in st['collect']['qwen36.full']
    assert run()['qwen36'] == 'running'
    st['collect']['qwen36.full']['lanes_done_at'] -= G.g['collect_grace'] + 1
    (G.run_dir('critpt', 'c99-qwen36-critpt') / 'items.jsonl').write_text('{"a": 1}\n')     # progress
    assert run()['qwen36'] == 'running' and not any(c[0] == 'condor_rm' for c in fc.calls)


def test_a_collector_that_gave_up_fails_its_suites(G, tmp_path):
    fc = FakeCondor()
    pi_run(G, tmp_path, 'gemma26', 'critpt')
    st = G.load()
    st['collect']['gemma26.full'] = {'submits': [], 'gave_up': ['critpt']}
    r = GR.grade_pass(G, GR.Queue.fetch('me', fc), st, ['gemma26'], 'full', False, False, fc, out=lambda *_: None)
    assert r[('gemma26', 'critpt')].startswith('failed')


def test_log_outcome_needs_every_proc_ok(tmp_path):
    log = tmp_path / 'c.log'
    log.write_text('005 (9.000.000) x Job terminated.\n\t(1) Normal termination (return value 3)\n'
                   '005 (9.001.000) x Job terminated.\n\t(1) Normal termination (return value 0)\n')
    assert GR.log_outcome(str(log), '9') == 'failed'


def test_judge_only_pass_does_not_persist_grade_transitions(G, tmp_path):
    fc = FakeCondor()
    (tmp_path / 'g.log').write_text('005 (42.000.000) x Job terminated.\n\t(1) Normal termination (return value 0)\n')
    rid = G.C.run_id('qwen36', 'hle', 'full')
    G.run_dir('hle', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'hle.{rid}': {'done': True}})
    G.save({'grade': {'qwen36.full.hle': {'grade': {'status': 'submitted', 'clusters': ['42'], 'log': str(tmp_path / 'g.log')}}}})
    GR.tick(G, ['qwen36'], 'full', parts=('judge',), runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    assert G.load()['grade']['qwen36.full.hle']['grade']['status'] == 'submitted'


def test_judge_registry_must_be_the_arms_server(tmp_path, monkeypatch):
    C = campaign(tmp_path)
    C.cfg['grading'] = {'judge': {'registry': 'some-other-judge'}}
    with pytest.raises(SystemExit):
        GR.Grading(C)


def test_lock_pidfile(G):
    import os
    with GR.Lock(G) as lock:
        assert lock.mine and lock.pid.exists() and GR.Lock.current is lock
        os.utime(lock.pid, (time.time() - 99999, time.time() - 99999))
        G.save({})                                            # every save refreshes it
        assert time.time() - lock.pid.stat().st_mtime < 60
        with pytest.raises(SystemExit):
            with GR.Lock(G):
                pass
    assert not lock.pid.exists() and GR.Lock.current is None
    lock.pid.write_text('otherhost 1\n')                      # another node's watcher, refreshed recently
    with pytest.raises(SystemExit):
        with GR.Lock(G):
            pass
    os.utime(lock.pid, (time.time() - GR.Lock.STALE_SEC - 1,) * 2)
    with GR.Lock(G) as l2:                                    # stale: taken over
        l2.pid.write_text('otherhost 2\n')                   # ...and taken from us: we never unlink theirs
    assert lock.pid.read_text() == 'otherhost 2\n'
    lock.pid.write_text(f'{GR.socket.gethostname()} 999999999\n')   # this host, process gone
    with GR.Lock(G):
        pass


def test_failed_submit_with_tries_left_is_a_retry_that_keeps_the_watch_and_the_judge(G, tmp_path, monkeypatch):
    fc = FakeCondor()
    judge_setup(G)
    fc.jobs.append({'ClusterId': 500, 'JobStatus': 2, 'JobBatchName': 'tasksynth-serve-aa-judge-qwen38-27b'})
    rid = G.C.run_id('qwen36', 'hle', 'full')
    G.run_dir('hle', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'hle.{rid}': {'done': True}})
    G.save({'judge': {'jobs': ['500'], 'idle_since': time.time() - 3600}})
    fc.wrapper_out['aa_evals.hle.submit'] = 'condor_submit_bid timed out\n'      # rc 0, no cluster id
    kw = dict(runner=fc, out=lambda *_: None, ps=lambda: '', flight=lambda e: 0.0)
    r = GR.tick(G, ['qwen36'], 'full', **kw)
    assert r['grade'][('qwen36', 'hle')].startswith('retry') and not GR.finished(r)
    assert 'released' not in r['judge'] and not any(c[0] == 'condor_rm' for c in fc.calls)
    # the "failed" submit had queued a job after all: adopted even at max tries, never a duplicate
    st = G.load()
    st['grade']['qwen36.full.hle']['grade']['tries'] = G.g['max_tries']
    G.save(st)
    fc.jobs.append({'ClusterId': 88, 'ProcId': 0, 'JobStatus': 1, 'JobBatchName': f'aa-hle-{rid}-graded'})
    r = GR.tick(G, ['qwen36'], 'full', **kw)
    assert r['grade'][('qwen36', 'hle')] == 'running'
    assert G.load()['grade']['qwen36.full.hle']['grade']['clusters'] == ['88']


def test_collector_out_of_tries_fails_its_suites(G, tmp_path):
    fc = FakeCondor()
    for s in GR.PI_COLLECT:
        pi_run(G, tmp_path, 'gemma26', s)
    st = G.load()
    st['collect']['gemma26.full'] = {'submits': [{}] * (G.g['max_tries'] + 1)}
    q = GR.Queue.fetch('me', fc)
    assert GR.collect_pass(G, q, st, ['gemma26'], 'full', False, fc, out=lambda *_: None)['gemma26'].startswith('failed')
    r = GR.grade_pass(G, q, st, ['gemma26'], 'full', True, False, fc, out=lambda *_: None)
    assert all(r[('gemma26', s)].startswith('failed') for s in ('gdpval', 'critpt'))


def test_collector_submit_error_is_per_model(G, tmp_path, monkeypatch):
    fc = FakeCondor()
    for m in ('gemma26', 'qwen36'):
        pi_run(G, tmp_path, m, 'critpt')

    def bad(sub, bid=None, **kw):
        raise GR.condor_cost.SubmitError('schedd down')
    monkeypatch.setattr(GR.condor_cost, 'submit', bad)
    r = GR.collect_pass(G, GR.Queue.fetch('me', fc), G.load(), ['gemma26', 'qwen36'], 'full', False, fc,
                        out=lambda *_: None)
    assert r['gemma26'].startswith('retry') and r['qwen36'].startswith('retry')


def test_sibling_arms_of_the_judge_server_are_other_users(G):
    fc = FakeCondor()
    fc.jobs.append({'ClusterId': 901, 'JobStatus': 2, 'JobBatchName': 'sweep', 'Args': '--arm qwen38-27b-nothink'})
    why = GR.other_users(G, GR.Queue.fetch('me', fc), set(), set(), ps=lambda: '')
    assert any('901' in w for w in why)


def test_a_model_with_a_suites_subset_is_graded_and_collected_only_on_those(G, tmp_path):
    G.C.models['qwen36']['suites'] = ['critpt', 'hle']
    rid = G.C.run_id('qwen36', 'hle', 'full')
    G.run_dir('hle', rid).mkdir(parents=True)
    set_lanes(G, 'qwen36', {f'hle.{rid}': {'done': True}})
    text = GR.status(G, ['qwen36'], 'full', runner=FakeCondor(), ps=lambda: '', flight=lambda e: 0.0)
    rows = [l.split()[0] for l in text.splitlines() if l.startswith('   ')]
    assert rows == ['critpt', 'hle']                      # gdpval / omniscience / ... never wait on this model
    r = GR.tick(G, ['qwen36'], 'full', dry='status', runner=FakeCondor(), out=lambda *_: None, ps=lambda: '',
                flight=lambda e: 0.0)
    assert {k[1] for k in r['grade'] if k[0] == 'qwen36'} == {'critpt', 'hle'}
    assert r['collect']['qwen36'] == 'waiting: no Pi run'   # only critpt is a collected Pi suite here


def test_status_prints_each_runs_aai_open_version(G, tmp_path, monkeypatch):
    """#481: status names the version each run recorded (single-turn config, or pi_eval's run dir for Pi suites)."""
    rid = G.C.run_id('qwen36', 'hle', 'full')
    G.run_dir('hle', rid).mkdir(parents=True)
    (G.run_dir('hle', rid) / 'config.json').write_text(json.dumps({'aai_open': {'version': 'v0.1'}}))
    pi = tmp_path / 'pi_runs'
    monkeypatch.setenv('AA_RUNS_ROOT', str(pi))
    tb = pi / 'tb21' / G.C.run_id('qwen36', 'tb21', 'full')
    tb.mkdir(parents=True)
    (tb / 'config.json').write_text(json.dumps({'aai_open': {'version': 'v0.1+custom'}}))
    text = GR.status(G, ['qwen36'], 'full', runner=FakeCondor(), ps=lambda: '', flight=lambda e: 0.0)
    row = {l.split()[0]: l for l in text.splitlines() if l.startswith('   ')}
    assert row['hle'].endswith('aai-open v0.1') and row['tb21'].endswith('aai-open v0.1+custom')
    assert row['critpt'].endswith('aai-open unrecorded')

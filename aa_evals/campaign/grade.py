#!/usr/bin/env python3
"""Campaign collection, grading and judging with the #378 cost rules (the c30 ops scripts c30_collect.py,
c30_grade.py and grade_loop.sh, made config-driven; the sibling of launch.py, same CONFIG).

  python -m aa_evals.campaign.grade CONFIG status  [MODEL ...] [--stage full|canary]
  python -m aa_evals.campaign.grade CONFIG collect [MODEL ...] [--stage full|canary] [--dry-run]   # one pass
  python -m aa_evals.campaign.grade CONFIG grade   [MODEL ...] [--stage full|canary] [--dry-run]   # one pass
  python -m aa_evals.campaign.grade CONFIG judge   [--stage full|canary] [--dry-run]               # one pass
  python -m aa_evals.campaign.grade CONFIG watch   [MODEL ...] [--stage full|canary] [--every 60]  # all three, looped

Collect: one CPU job per model and stage (1 CPU / 4 GB; c30's used 0.14 cores and <= 0.9 GB RSS) runs the Pi
collectors of GDPval, CritPt and AutomationBench (`pi.run collect --follow`). It is submitted once the model's first
Pi attempt resolves and exits by itself when every attempt is imported (a `<suite>.collected` marker per suite; a
restarted job skips marked suites and the collectors resume where they were). If it is still running `collect_grace`
s (1 h) after all of the model's Pi lanes are done, it is removed and reported.

Grade: each suite's grading is submitted as soon as it can start:
- HLE, Omniscience, GDP.pdf: generation done (dispatcher lane done, or every attempt finished with no live job), then
  the judged grade job (`<run>-graded`, the module's own submit wrapper: lean shape, bid ladder), once the judge is up;
- GDPval: imported (collector marker), then `gdpval.refs` plan + judge against the reference runs (one CPU job);
- CritPt: imported, then `critpt.run --reprobe --condor` (4 CPU / 32 GB), then `critpt.run --grade` here;
- TB 2.1, TB-Sci-lite, SciCode, AutomationBench grade in their runs (verifier, step tests, collector).
A grade job that ends without exit code 0 is resubmitted (the graders resume), at most `max_tries` times in all.

Judge (event-driven, no fixed cadence beyond the watcher's pass): it is started when judged work is ready and no job of
the judge registry is live; it is released when, for `idle_release_sec` (600 s), no judge request was in flight (vLLM
running + waiting, gate in flight) and no judged work of this campaign was queued or running. It is never released
when this campaign did not start it, nor while anyone else may use it: another live condor job or local process that
names the registry or any arm it serves, or another campaign's lease (`<registry>/leases/<campaign>.json`, refreshed
while that campaign has judged work).

Everything it writes is on /data: <dispatch_root>/<name>-grade/ (collector and grade job dirs, state.json, condor
logs), events in <dispatch_root>/<name>-launch/events.jsonl (launch.py's log).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from aa_evals.campaign import launch as L
from aa_evals.common import aai_open, condor_cost
from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

GDPY = '/home/user/python-3.11/bin/python3'            # GDPval (pandas, pyarrow)
ABPY = '/opt/aai/venvs/automationbench/bin/python'
COLLECT_PY = {'gdpval': GDPY, 'critpt': L.STIRRUP, 'automationbench': ABPY}
PI_COLLECT = ('gdpval', 'critpt', 'automationbench')       # Pi suites imported by a collector
PI_BENCH = {'gdpval': 'gdpval_pi', 'critpt': 'critpt_pi', 'automationbench': 'automationbench_pi'}
JUDGED = ('hle', 'omniscience', 'gdppdf', 'gdpval')         # grading needs the judge server
GRADED = ('hle', 'omniscience', 'gdppdf', 'gdpval', 'critpt')
SELF_GRADED = ('tb21', 'tbsci_lite', 'scicode', 'automationbench')
LIVE = (1, 2, 5)                                           # idle, running, held: still in the queue

DEFAULTS = {
    'runs_root': '/data/aai/aa_evals/runs',
    'collector': {'cpus': 1, 'mem_gb': 4, 'disk_gb': 20, 'gdpval_concurrency': 2},
    'collect_grace': 3600,
    'max_tries': 3,
    'gdpval_refs': ['m36-qwen38-xhigh-gdpval-full', 'm36-lightning-gdpval-full'],
    'gdpval_judge': {'cpus': 1, 'mem_gb': 8, 'min_success': 0.98},
    'judge': {'registry': 'aa-judge-qwen38-27b', 'arm': 'qwen38-27b-medium-pin', 'teacher': 'qwen38-27b-eval-fast',
              'gpus': 1, 'gpu': 'B200', 'bid': 61, 'hours': 16,
              'protocol': {'context_tokens': 262144, 'max_output_tokens': 262144},
              'concurrency': 16, 'idle_release_sec': 600, 'lease_sec': 900},
}


def now() -> float:
    return time.time()


def iso(t: float | None = None) -> str:
    return datetime.datetime.fromtimestamp(now() if t is None else t, datetime.timezone.utc).isoformat(timespec='seconds')


# ------------------------------------------------------------------------------------------------ config
class Grading:
    """The grading view of a campaign: launch.Campaign plus the `grading` block of its config (DEFAULTS)."""

    def __init__(self, C: L.Campaign):
        self.C = C
        g = C.cfg.get('grading') or {}
        self.g = {**DEFAULTS, **g}
        for k in ('collector', 'gdpval_judge', 'judge'):
            self.g[k] = {**DEFAULTS[k], **(g.get(k) or {})}
        self.j = self.g['judge']
        from aa_evals.judges.arms import ARMS
        arm = ARMS.get(self.j['arm'])
        if arm is None or arm.server != self.j['registry']:
            raise SystemExit(f"grading.judge: arm {self.j['arm']} is served by {arm.server if arm else '?'}, not "
                             f"{self.j['registry']} (the graders reach the judge through the arm)")
        self.runs_root = Path(self.g['runs_root'])
        self.dir = C.dispatch_root / f'{C.name}-grade'
        self.state_path = self.dir / 'state.json'
        self.user = getpass.getuser()

    def run_dir(self, suite: str, run_id: str) -> Path:
        return self.runs_root / suite / run_id

    def collect_dir(self, model: str, stage: str) -> Path:
        return self.dir / 'collect' / f'{model}.{stage}'

    def collect_batch(self, model: str, stage: str) -> str:
        return f'aa-collect-{self.C.name}-{model}' + ('-canary' if stage == 'canary' else '')

    def pool(self, model: str, stage: str) -> str:
        return f'{self.C.name}-refs-{model}' + ('-canary' if stage == 'canary' else '')

    # state.json: {'collect': {model.stage: {...}}, 'grade': {model.stage.suite: {step: {...}}}, 'judge': {...}}
    def load(self) -> dict:
        try:
            s = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            s = {}
        for k in ('collect', 'grade', 'judge'):
            s.setdefault(k, {})
        return s

    def save(self, s: dict) -> None:
        if Lock.current is not None:
            Lock.current.refresh()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(f'state.json.{os.getpid()}')
        tmp.write_text(json.dumps(s, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, self.state_path)


# ------------------------------------------------------------------------------------------------ condor
class Queue:
    """One condor_q snapshot of this user's jobs."""

    ATTRS = 'ClusterId,ProcId,JobStatus,JobBatchName,Args,Arguments,Environment,UserLog,HoldReason,HoldReasonCode'

    def __init__(self, jobs: list[dict]):
        self.jobs = jobs

    @classmethod
    def fetch(cls, user: str, runner=subprocess.run) -> 'Queue':
        r = runner(['condor_q', user, '-json', '-attributes', cls.ATTRS], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise RuntimeError(f'condor_q failed: {(r.stderr or "").strip()[:300]}')
        return cls(json.loads(r.stdout or '[]') or [])

    def live(self, pred=lambda j: True) -> list[dict]:
        return [j for j in self.jobs if j.get('JobStatus') in LIVE and pred(j)]

    def by_batch(self, batch: str, states=LIVE) -> list[dict]:
        return [j for j in self.jobs if j.get('JobBatchName') == batch and j.get('JobStatus') in states]

    def cluster(self, cluster: str) -> list[dict]:
        return [j for j in self.jobs if str(j.get('ClusterId')) == str(cluster)]


def text_of(job: dict) -> str:
    return ' '.join(str(job.get(k) or '') for k in ('JobBatchName', 'Args', 'Arguments', 'Environment'))


def mentions(text: str, token: str) -> bool:
    """`token` (a run id, registry or arm) as a whole name: not part of a longer one (`c1-x-hle` is not in
    `c1-x-hle-graded` or `c1-x-hle-canary`, but is in `aa-pi-hle-c1-x-hle` and `.../c1-x-hle/job`)."""
    return re.search(rf'(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_-])', text) is not None


def log_outcome(log: str | None, cluster: str) -> str | None:
    """'ok' (exit code 0), 'failed' (another exit code, or aborted) or None (no end in the event log yet)."""
    if not log:
        return None
    try:
        text = Path(log).read_text(errors='replace')
    except OSError:
        return None
    procs = {}
    for m in re.finditer(rf'^(005|009) \({int(cluster):d}\.(\d+)\.\d+\)[^\n]*\n((?:\t[^\n]*\n|\s{{2,}}[^\n]*\n)*)', text, re.M):
        if m.group(1) == '009':
            procs[m.group(2)] = 'failed'
            continue
        code = re.search(r'exit-code (\d+)|return value (\d+)', m.group(3))
        procs[m.group(2)] = 'ok' if code and int(code.group(1) or code.group(2)) == 0 else 'failed'
    if not procs:
        return None
    return 'ok' if all(v == 'ok' for v in procs.values()) else 'failed'


def clusters_of(text: str) -> list[str]:
    """Cluster ids a submit wrapper printed: condor_cost's `(cluster N)`, condor_submit's `submitted to cluster N`,
    hle.submit's `N.0  logs:`. Not the bid-ladder watcher's `watcher: cluster N` (a job of its own)."""
    found = re.findall(r'\(cluster (\d+)\)|submitted to cluster (\d+)|^(\d+)\.0\s', text, re.M)
    return sorted({x for t in found for x in t if x}, key=int)


def submit_job(sub: Path, runner=subprocess.run) -> list[str]:
    recs = condor_cost.submit(sub, None, runner=runner)   # the bid ladder + periodic_release on memory holds
    return [r['cluster'] for r in recs]


def own_sub(d: Path, script: Path, batch: str, cpus: int, mem_gb: int, disk_gb: int) -> str:
    reqs = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in condor_cost.bad_nodes())
    home = os.environ.get('HOME') or f'/home/{getpass.getuser()}'
    return f'''universe = vanilla
executable = /bin/bash
arguments = {script}
initialdir = {d}
output = {d}/job.$(Cluster).out
error = {d}/job.$(Cluster).err
log = {d}/condor.log
request_cpus = {cpus}
request_memory = {int(mem_gb * 1024)}
request_disk = {disk_gb}G
requirements = {reqs}
+BypassLXCfs = "true"
+JobBatchName = "{batch}"
+MaxRuntime = 172800
getenv = false
environment = "HOME={home} USER={getpass.getuser()} PATH=/usr/local/bin:/usr/bin:/bin{serving.condor_env()}"
queue
'''


def dry_submit(sub: Path, runner=subprocess.run) -> str:
    r = runner(['condor_submit', '-dry-run', '/dev/null', str(sub)], capture_output=True, text=True, timeout=120)
    return f'condor_submit -dry-run rc {r.returncode}: {((r.stdout or "") + (r.stderr or "")).strip()[-200:]}'


# ------------------------------------------------------------------------------------------------ run state
def lanes(C: L.Campaign, model: str, stage: str) -> dict:
    try:
        return json.loads((C.dispatch_dir(model, stage) / 'state.json').read_text()).get('lanes') or {}
    except (OSError, ValueError):
        return {}


def lane_of(ls: dict, run_id: str) -> dict | None:
    return next((v for k, v in ls.items() if k.endswith('.' + run_id)), None)


def finished_count(run_dir: Path) -> int:
    try:
        from aa_evals.tracejudge.layouts import open_run
        return len(open_run(str(run_dir)).finished())
    except (Exception, SystemExit):   # noqa: BLE001  (no layout yet, or a run mid-write)
        return -1


def generation_done(G: Grading, q: Queue, ls: dict, c: dict) -> bool:
    """The suite's lane is done and no generation job of its run is live; without a lane (a dispatcher that died),
    every attempt finished and no job is live."""
    rid = c['run_id']
    # a graded job names its source run too (GDP.pdf: --generations-from RID): it is not generation
    live = q.live(lambda j: mentions(text_of(j), rid) and not mentions(text_of(j), rid + '-graded')
                  and not str(j.get('JobBatchName') or '').startswith(('aa-grade-', 'aa-collect-')))
    if live:
        return False
    lane = lane_of(ls, rid)
    if lane is not None and lane.get('done'):
        return True
    return finished_count(G.run_dir(c['suite'], rid)) >= c['attempts']


def pi_run_dir(G: Grading, suite: str, run_id: str) -> Path | None:
    try:
        cfg = json.loads((G.run_dir(suite, run_id) / 'config.json').read_text())
        return Path(cfg['harness']['pi_run'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


PI_BENCH = {'tb21': 'tb21', 'tbsci_lite': 'tbsci_lite', 'gdpval': 'gdpval_pi', 'critpt': 'critpt_pi',
            'automationbench': 'automationbench_pi'}   # pi_eval's run dirs: <AA_RUNS_ROOT>/<bench>/<run_id>


def run_version(G: Grading, suite: str, run_id: str) -> str:
    """The run's recorded AAI-open version (#481; aai_open.of_run): the graded copy's (it adds grade-stage deviations,
    e.g. another judge arm) or the run's own config, the Pi run a collected suite points at, or the pi_eval run dir;
    'unrecorded' for runs from before the field."""
    graded = G.run_dir(suite, run_id + '-graded')
    v = aai_open.of_run(graded if (graded / 'config.json').exists() else G.run_dir(suite, run_id))
    if v == 'unrecorded' and suite in PI_BENCH:
        root = Path(os.environ.get('AA_RUNS_ROOT', '/data/aai/aa_evals/runs_fallback'))
        v = aai_open.of_run(root / PI_BENCH[suite] / run_id)
    return v


def resolved_any(pi_run: Path | None) -> bool:
    try:
        with os.scandir(pi_run / 'queue' / 'done') as it:
            return any(True for _ in it)
    except (OSError, TypeError):
        return False


# ------------------------------------------------------------------------------------------------ collect
def progress(G: Grading, model: str, stage: str, suites: list[str]) -> list:
    """What the collector has imported so far (sizes of items/transcripts files, GDPval outcome and render count)."""
    sig = []
    for s in suites:
        d = G.run_dir(s, G.C.run_id(model, s, stage))
        if s == 'gdpval':
            sig.append(sum(1 for _ in d.glob('attempts/*/outcome.json')) + sum(1 for _ in d.glob('attempts/*/render')))
        else:
            sig.append(sum(f.stat().st_size for f in (d / 'items.jsonl', d / 'transcripts.jsonl') if f.exists()))
    return sig


def collector_script(G: Grading, model: str, stage: str, suites: list[str]) -> str:
    C, d = G.C, G.collect_dir(model, stage)
    lines = ['#!/bin/bash', f'# {C.name} Pi collectors for {model} ({stage}): {", ".join(suites)} (aa_evals.campaign.grade)',
             'set -u', f'cd {shlex.quote(str(C.code))}',
             f'export PYTHONPATH={shlex.quote(str(C.code))} PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1',
             'export TMPDIR=${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}',
             # the run roots the watcher reads (the modules' defaults unless the config's runs_root moves them)
             f'export AA_RUNS_ROOT={shlex.quote(str(G.runs_root))} CRITPT_RUNS={shlex.quote(str(G.runs_root / "critpt"))}',
             f'D={shlex.quote(str(d))}',
             'collect() {   # suite python run_id run_config extra: follow until every attempt is imported (resumable)',
             '  [ -e "$D/$1.collected" ] && { echo "$1: already collected"; return 0; }',
             '  for i in $(seq 120); do [ -e "$4" ] && break; sleep 60; done',
             '  for try in 1 2 3 4 5; do',
             '    "$2" -m aa_evals.$1.pi.run collect --run-id "$3" --follow $5 >> "$D/collect-$1.log" 2>&1 && '
             '{ date -u +%FT%TZ > "$D/$1.collected"; echo "$1: collected"; return 0; }',
             '    echo "$1: collector exited rc $? (try $try); again in 120 s" | tee -a "$D/collect-$1.log"; sleep 120',
             '  done',
             '  return 1',
             '}']
    for s in suites:
        rid = C.run_id(model, s, stage)
        extra = {'gdpval': f"--concurrency {G.g['collector']['gdpval_concurrency']}",
                 'automationbench': f'--out-root {G.runs_root / "automationbench"}'}.get(s, '')
        lines.append(f'collect {s} {COLLECT_PY[s]} {rid} {shlex.quote(str(G.run_dir(s, rid) / "config.json"))} '
                     f'{shlex.quote(extra)} &')
    lines += ['wait', 'for s in ' + ' '.join(suites) + '; do [ -e "$D/$s.collected" ] || exit 1; done',
              'echo all collected']
    return '\n'.join(lines) + '\n'


def collect_pass(G: Grading, q: Queue, st: dict, models: list[str], stage: str, dry: bool | str,
                 runner=subprocess.run, out=print) -> dict:
    """Per model: 'waiting' (no Pi attempt resolved yet), 'running', 'submitted', 'done' or 'failed: ...'."""
    C, res = G.C, {}
    for model in models:
        key, d = f'{model}.{stage}', G.collect_dir(model, stage)
        rec = st['collect'].setdefault(key, {'submits': []})
        rids = {s: C.run_id(model, s, stage) for s in PI_COLLECT if s in L.model_suites(C, model)}
        started = [s for s in rids if (G.run_dir(s, rids[s]) / 'config.json').exists()]
        todo = [s for s in started if not (d / f'{s}.collected').exists()]
        live = q.by_batch(G.collect_batch(model, stage))
        if started and not todo:
            res[model] = 'done'
            continue
        ls = lanes(C, model, stage)
        pi_done = bool(started) and all((lane_of(ls, rids[s]) or {}).get('done') for s in started)
        if live:
            sig = progress(G, model, stage, started)
            if not pi_done:
                rec.pop('lanes_done_at', None)
            elif rec.get('progress') != sig or 'lanes_done_at' not in rec:
                rec['lanes_done_at'] = now()   # the grace runs from the last import progress after the lanes end
            rec['progress'] = sig
            held = [j for j in live if j.get('JobStatus') == 5]
            if pi_done and now() - rec['lanes_done_at'] > G.g['collect_grace']:
                job = str(live[0]['ClusterId'])
                out(f'{model}: collector {job} still running {G.g["collect_grace"]:.0f} s after its lanes ended '
                    f'({", ".join(todo)} not imported): removed; see {d}/collect-*.log')
                if not dry:
                    runner(['condor_rm', job], capture_output=True, text=True, timeout=120)
                    C.log(event='collect_removed', model=model, stage=stage, job=job, missing=todo)
                    rec['gave_up'] = todo
                res[model] = f'failed: not imported {todo}'
                continue
            res[model] = 'held: ' + str(held[0].get('HoldReason'))[:120] if held else 'running'
            continue
        if rec.get('gave_up'):
            res[model] = f"failed: not imported {rec['gave_up']}"
            continue
        if not any(resolved_any(pi_run_dir(G, s, rids[s])) for s in todo):
            res[model] = 'waiting' if started else 'waiting: no Pi run'
            continue
        if len(rec['submits']) >= G.g['max_tries'] + 1:
            if not dry:
                rec['gave_up'] = todo
            res[model] = f"failed: {len(rec['submits'])} collector jobs ended without importing {todo} ({d})"
            continue
        if dry == 'status':
            res[model] = 'would submit'
            continue
        d.mkdir(parents=True, exist_ok=True)
        sh = d / 'collect.sh'
        sh.write_text(collector_script(G, model, stage, todo))
        sub = d / 'collect.sub'
        k = G.g['collector']
        sub.write_text(own_sub(d, sh, G.collect_batch(model, stage), k['cpus'], k['mem_gb'], k['disk_gb']))
        if dry:
            out(f'{model}: would submit {sub} ({", ".join(todo)}); {dry_submit(sub, runner)}')
            res[model] = 'would submit'
            continue
        try:
            clusters = submit_job(sub, runner)
        except (condor_cost.SubmitError, OSError, subprocess.SubprocessError, ValueError) as e:
            clusters = [r['cluster'] for r in getattr(e, 'submitted', None) or []]
            if not clusters:
                rec['submits'].append({'at': now(), 'clusters': [], 'suites': todo, 'error': repr(e)[-300:]})
                out(f'{model}: collector submit failed: {e!r}'[:300])
                res[model] = f'retry: collector submit failed ({e!r})'[:200]
                continue
        rec.pop('lanes_done_at', None)
        rec['submits'].append({'at': now(), 'clusters': clusters, 'suites': todo})
        C.log(event='collect_submit', model=model, stage=stage, suites=todo, clusters=clusters)
        out(f'{model}: collector {",".join(clusters)} for {", ".join(todo)}')
        res[model] = 'submitted'
    return res


# ------------------------------------------------------------------------------------------------ grade
def grade_steps(G: Grading, model: str, stage: str, suite: str, run_id: str) -> list[dict]:
    """[{'name', 'kind': 'job' (a submit wrapper whose condor job is tracked) | 'own' (our job) | 'local', 'argv' or
    'script', 'judged'}] for one suite."""
    j, dst = G.j, run_id + '-graded'
    conc = str(j['concurrency'])
    if suite == 'hle':
        return [{'name': 'grade', 'kind': 'job', 'judged': True, 'argv': [
            L.STIRRUP, '-m', 'aa_evals.hle.submit', '--run-id', dst, '--', '--stage', 'grade', '--generations-from', run_id,
            '--judge-arm', j['arm'], '--judge-concurrency', conc]}]
    if suite == 'omniscience':
        return [{'name': 'grade', 'kind': 'job', 'judged': True, 'argv': [
            'bash', 'aa_evals/omniscience/submit.sh', dst, '--stage', 'grade', '--answers-from', run_id,
            '--judge-arm', j['arm'], '--judge-concurrency', conc]}]
    if suite == 'gdppdf':
        return [{'name': 'grade', 'kind': 'job', 'judged': True, 'argv': [
            L.PDFPY, '-m', 'aa_evals.gdppdf.submit', 'run', '--run-id', dst, '--', '--stage', 'grade',
            '--generations-from', run_id, '--judge-arm', j['arm'], '--judge-concurrency', conc]}]
    if suite == 'gdpval':
        pool = G.pool(model, stage)
        refs = ','.join(G.g['gdpval_refs'])
        pairs = G.runs_root / 'gdpval' / '_pools' / pool / 'pairs.jsonl'
        code = shlex.quote(str(G.C.code))
        script = '\n'.join([
            '#!/bin/bash', f'# {G.C.name}: GDPval {run_id} against {refs} (aa_evals.gdpval.refs)', 'set -euo pipefail',
            f'cd {code}', f'export PYTHONPATH={code} PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1',
            f'[ -e {shlex.quote(str(pairs))} ] || {GDPY} -m aa_evals.gdpval.refs plan --run {run_id} --pool {pool} --refs {refs}',
            f'exec {GDPY} -m aa_evals.gdpval.refs judge --run {run_id} --pool {pool} --refs {refs} '
            f"--judge-arm {j['arm']} --concurrency {conc} --min-success {G.g['gdpval_judge']['min_success']}", ''])
        return [{'name': 'judge', 'kind': 'own', 'judged': True, 'script': script}]
    if suite == 'critpt':
        return [{'name': 'reprobe', 'kind': 'job', 'judged': False,
                 'argv': [L.STIRRUP, '-m', 'aa_evals.critpt.run', '--run-id', run_id, '--reprobe', '--condor']},
                {'name': 'grade', 'kind': 'local', 'judged': False,
                 'argv': [L.STIRRUP, '-m', 'aa_evals.critpt.run', '--run-id', run_id, '--grade']}]
    return []


def ready(G: Grading, q: Queue, st: dict, ls: dict, model: str, stage: str, c: dict) -> str | None:
    """None when the suite's grading can start, else what it waits for."""
    if c['suite'] in PI_COLLECT:
        if (G.collect_dir(model, stage) / f"{c['suite']}.collected").exists():
            return None
        gave_up = (st['collect'].get(f'{model}.{stage}') or {}).get('gave_up') or []
        return 'failed: collector gave up' if gave_up else 'import'
    return None if generation_done(G, q, ls, c) else 'generation'


def step_job_state(q: Queue, rec: dict) -> str:
    """'live', 'held', 'ok' or 'failed' for a step's recorded clusters."""
    clusters = rec.get('clusters') or []
    if not clusters:
        return 'failed'
    jobs = [j for c in clusters for j in q.cluster(c) if j.get('JobStatus') in LIVE]
    if jobs:
        for j in jobs:
            if j.get('UserLog') and not rec.get('log'):
                rec['log'] = j['UserLog']
        return 'held' if all(j.get('JobStatus') == 5 for j in jobs) else 'live'
    outs = [log_outcome(rec.get('log'), c) for c in clusters]
    return 'ok' if outs and all(o == 'ok' for o in outs) else 'failed'


def live_step_jobs(q: Queue, step: dict, run_id: str) -> list[str]:
    """Clusters of live jobs that are this step's: the graded run's job, our GDPval judge job, the CritPt reprobe."""
    if step['kind'] == 'own':
        pred = lambda j: j.get('JobBatchName') == f'aa-grade-{run_id}'  # noqa: E731
    elif step['kind'] == 'job' and step['name'] == 'reprobe':
        pred = lambda j: mentions(text_of(j), run_id) and 'aa_evals.critpt.run' in text_of(j)  # noqa: E731
    elif step['kind'] == 'job':
        pred = lambda j: mentions(text_of(j), run_id + '-graded')  # noqa: E731
    else:
        return []
    return sorted({str(j['ClusterId']) for j in q.live(pred)}, key=int)


def env_for(G: Grading) -> dict:
    return {**os.environ, 'PYTHONPATH': str(G.C.code), 'PYTHONDONTWRITEBYTECODE': '1'}


def run_step(G: Grading, model: str, stage: str, suite: str, run_id: str, step: dict, rec: dict,
             runner=subprocess.run) -> tuple[bool, str]:
    """Submit (or, for 'local', run) one step; fills rec. Returns (ok, message)."""
    rec['tries'] = rec.get('tries', 0) + 1
    rec['at'] = now()
    try:
        return _run_step(G, suite, run_id, step, rec, model, stage, runner)
    except condor_cost.SubmitError as e:
        if e.submitted:
            rec.update(status='submitted', clusters=[r['cluster'] for r in e.submitted])
            return True, f'partly submitted: {e}'
        rec.update(status='failed', out=str(e)[-300:])
        return False, str(e)[-200:]
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        rec.update(status='failed', out=repr(e)[-300:])
        return False, repr(e)[-200:]


def _run_step(G: Grading, suite: str, run_id: str, step: dict, rec: dict, model: str, stage: str, runner) -> tuple:
    C = G.C
    if step['kind'] == 'own':
        d = G.dir / 'grade' / f'{model}.{stage}.{suite}'
        d.mkdir(parents=True, exist_ok=True)
        sh = d / f"{step['name']}.sh"
        sh.write_text(step['script'])
        sub = d / f"{step['name']}.sub"
        k = G.g['gdpval_judge']
        sub.write_text(own_sub(d, sh, f'aa-grade-{run_id}', k['cpus'], k['mem_gb'], 10))
        rec['clusters'], rec['log'], rec['status'] = submit_job(sub, runner), str(d / 'condor.log'), 'submitted'
        return True, f"cluster {','.join(rec['clusters'])}"
    argv = list(step['argv'])
    if step['kind'] == 'job' and suite == 'hle' and rec['tries'] > 1:
        i = argv.index('--')
        argv = argv[:i] + ['--force'] + argv[i:]          # the wrapper's receipt exists; the graded run resumes
    p = runner(argv, cwd=C.code, text=True, capture_output=True, env=env_for(G),
               timeout=1800 if step['kind'] == 'local' else 900)
    full = (p.stdout or '') + (p.stderr or '')
    tail = full.strip()[-800:]
    rec['rc'], rec['out'] = p.returncode, tail[-300:]
    if step['kind'] == 'local':
        rec['status'] = 'done' if p.returncode == 0 else 'failed'
        return p.returncode == 0, tail.splitlines()[-1][:200] if tail else ''
    clusters = clusters_of(full)
    if p.returncode or not clusters:
        rec['status'] = 'failed'
        return False, f'rc {p.returncode}: {tail[-200:]}'
    rec['clusters'], rec['status'] = clusters, 'submitted'
    for c in clusters:   # the job's event log, for its outcome once it leaves the queue
        r = runner(['condor_q', c, '-af', 'UserLog'], capture_output=True, text=True, timeout=120)
        if (r.stdout or '').strip():
            rec['log'] = r.stdout.strip().splitlines()[0]
    return True, f"cluster {','.join(clusters)}"


def grade_pass(G: Grading, q: Queue, st: dict, models: list[str], stage: str, judge_up: bool, dry: bool,
               runner=subprocess.run, out=print, save=lambda: None) -> dict:
    """{(model, suite): state}: 'self-graded', 'waiting: generation|import', 'needs judge', 'running', 'held',
    'submitted', 'done', 'failed: ...'."""
    C, res = G.C, {}
    sel = L.plan_suites(C, stage)
    for model in models:
        ls = lanes(C, model, stage)
        for suite in L.model_suites(C, model):
            c = L.suite_cmd(C, model, suite, sel[suite], stage)
            if suite in SELF_GRADED:
                res[(model, suite)] = 'self-graded'
                continue
            if not (C.dispatch_dir(model, stage).exists() or G.run_dir(suite, c['run_id']).exists()):
                res[(model, suite)] = 'not launched'
                continue
            recs = st['grade'].setdefault(f'{model}.{stage}.{suite}', {})
            # once a step was submitted, readiness is settled (a graded job may name its source run: GDP.pdf)
            why = None if any(x.get('status') for x in recs.values()) else ready(G, q, st, ls, model, stage, c)
            if why:
                res[(model, suite)] = why if why.startswith('failed') else f'waiting: {why}'
                continue
            state = 'done'
            for step in grade_steps(G, model, stage, suite, c['run_id']):
                rec = recs.setdefault(step['name'], {})
                if rec.get('status') == 'done':
                    continue
                if rec.get('status') == 'submitted':
                    js = step_job_state(q, rec)
                    if js in ('live', 'held'):
                        state = 'running' if js == 'live' else 'held'
                        break
                    if js == 'ok':
                        rec['status'] = 'done'
                        if not dry:   # a dry pass changes nothing (its state is not saved either)
                            C.log(event='grade_done', model=model, stage=stage, suite=suite, step=step['name'],
                                  clusters=rec.get('clusters'))
                            out(f'{model} {suite}: {step["name"]} done')
                        continue
                    rec['status'] = 'failed'
                    if not dry:
                        C.log(event='grade_failed', model=model, stage=stage, suite=suite, step=step['name'],
                              clusters=rec.get('clusters'), log=rec.get('log'))
                adopt = live_step_jobs(q, step, c['run_id'])
                if adopt:   # a job of this step is queued but not recorded (a pass or a submit that failed late)
                    rec.update(status='submitted', clusters=adopt, tries=max(1, rec.get('tries', 0)), at=now())
                    for j in q.live(lambda j: str(j.get('ClusterId')) in adopt):
                        rec['log'] = rec.get('log') or j.get('UserLog')
                    state = 'running'
                    break
                if rec.get('status') == 'failed' and rec.get('tries', 0) >= G.g['max_tries']:
                    state = f"failed: {step['name']} after {rec['tries']} tries ({rec.get('log') or rec.get('out')})"
                    break
                if step['judged'] and not judge_up:
                    state = 'needs judge'
                    break
                if dry:
                    out(f'{model} {suite}: would run {step["name"]}: ' +
                        (shlex.join(step['argv']) if 'argv' in step else step['script'].splitlines()[-2]))
                    state = 'would submit'
                    break
                ok, msg = run_step(G, model, stage, suite, c['run_id'], step, rec, runner)
                save()   # a later failure in this pass must not lose this submission
                C.log(event='grade_step', model=model, stage=stage, suite=suite, step=step['name'], ok=ok,
                      tries=rec['tries'], msg=msg)
                out(f'{model} {suite}: {step["name"]} {"ok" if ok else "FAILED"} (try {rec["tries"]}): {msg}')
                if step['kind'] == 'local' and ok:
                    continue
                state = 'submitted' if ok else \
                    (f'retry: {step["name"]} (try {rec["tries"]} failed): {msg}' if rec['tries'] < G.g['max_tries']
                     else f'failed: {step["name"]} after {rec["tries"]} tries: {msg}')
                break
            res[(model, suite)] = state
    return res


# ------------------------------------------------------------------------------------------------ judge
def registry_fresh(registry: str) -> dict | None:
    """endpoints.json of the registry's current job, updated < 120 s ago; else None."""
    try:
        e = serving.record(registry, L.REG)
        j = json.loads((serving.registry_root(L.REG) / registry / 'job.json').read_text())
    except (OSError, ValueError):
        return None
    if e.get('job_id') != j.get('job_id') or serving.age(e, now()) > serving.STALE_SEC:
        return None
    return e


_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def in_flight(e: dict, opener=_NOPROXY) -> float | None:
    """Judge requests in flight: vLLM running + waiting on every engine, plus the gate's in-flight count. None when
    an engine can't be read (counted as busy)."""
    total = 0.0
    try:
        for u in e.get('engine_urls') or e.get('urls') or []:
            t = opener.open(u.rsplit('/v1', 1)[0] + '/metrics', timeout=20).read().decode()
            total += sum(float(v) for v in re.findall(
                r'^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})? ([0-9.e+-]+)$', t, re.M))
        for u in e.get('gate_status_urls') or []:
            total += float(json.loads(opener.open(u, timeout=20).read().decode()).get('inflight') or 0)
    except Exception:   # noqa: BLE001
        return None
    return total


def other_users(G: Grading, q: Queue, ours: set[str], judge_jobs: set[str], ps=None) -> list[str]:
    """Why someone else may be using the judge: a live condor job that is not ours and names its registry or any arm it serves, a
    local process that does, or another campaign's fresh lease."""
    from aa_evals.judges.arms import ARMS
    reg = G.j['registry']
    arms = sorted({a.name for a in ARMS.values() if a.server == reg} | {G.j['arm']})
    why = []
    for j in q.live():
        c = str(j.get('ClusterId'))
        if c in ours or c in judge_jobs:
            continue
        t, b = text_of(j), str(j.get('JobBatchName') or '')
        # c30_grade's guard: graded views, reference pools and judge sweeps are judge clients; wrappers (HLE,
        # Omniscience) keep the arm in a file, so their batch name is what shows
        if mentions(t, reg) or any(mentions(t, a) for a in arms) or b.endswith('-graded') or b.startswith('aa-grade-') or \
                re.search(r'refs|judge', b):
            why.append(f"condor job {c} ({b})")
    try:
        procs = ps() if ps else subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True,
                                               timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        procs = ''
    me = {os.getpid(), os.getppid()}
    for line in procs.splitlines():
        pid, _, args = line.strip().partition(' ')
        if pid.isdigit() and int(pid) not in me and 'aa_evals.campaign.grade' not in args and \
                (mentions(args, reg) or any(mentions(args, a) for a in arms)):
            why.append(f'process {pid} on {socket.gethostname()}: {args[:100]}')
    for f in sorted((serving.registry_root(L.REG) / reg / 'leases').glob('*.json')):
        if f.stem != G.C.name and now() - f.stat().st_mtime < G.j['lease_sec']:
            why.append(f'lease {f.name}')
    return why


def judge_argv(G: Grading) -> list[str]:
    j = G.j
    return [L.HARBOR, '-m', 'task_synthesis.serve.serve_teacher', 'submit', '--teacher', j['teacher'], '--name',
            j['registry'], '--gpus', str(j['gpus']), '--cuda-cap', '10.0' if j['gpu'] == 'B200' else '9.0',
            '--bid', str(j['bid']), '--hours', str(j['hours']),
            '--protocol', json.dumps(j['protocol'], separators=(',', ':')), '--force']


def judge_pass(G: Grading, q: Queue, st: dict, need: bool, busy_work: bool, dry: bool, runner=subprocess.run,
               out=print, ps=None, flight=in_flight) -> str:
    """need: judged work is ready and waits for the judge; busy_work: judged work of this campaign is ready, queued
    or running. Starts, keeps or releases the judge; returns a one-line state, which starts with 'ours:' while a
    judge job this campaign started is live."""
    r = _judge_pass(G, q, st, need, busy_work, dry, runner, out, ps, flight)
    live = {str(j['ClusterId']) for j in q.by_batch(f"tasksynth-serve-{G.j['registry']}", LIVE)}
    mine = live & set(st['judge'].get('jobs') or [])
    return ('ours: ' if mine and 'released' not in r else '') + r


def _judge_pass(G, q, st, need, busy_work, dry, runner, out, ps, flight) -> str:
    C, js, reg = G.C, st['judge'], G.j['registry']
    batch = f'tasksynth-serve-{reg}'
    live = q.by_batch(batch, (1, 2))
    held = q.by_batch(batch, (5,))
    ours = set(js.get('jobs') or [])
    lease = serving.registry_root(L.REG) / reg / 'leases' / f'{C.name}.json'
    if busy_work and not dry:
        lease.parent.mkdir(parents=True, exist_ok=True)
        lease.write_text(json.dumps({'campaign': C.name, 'host': socket.gethostname(), 'pid': os.getpid(),
                                     'at': iso()}) + '\n')
    elif not busy_work and lease.exists() and not dry:
        lease.unlink()
    if not live:
        js.pop('idle_since', None)
        if held:
            return f"judge job {held[0]['ClusterId']} held: {str(held[0].get('HoldReason'))[:120]}"
        if not need:
            return 'judge: not needed, none live'
        if dry:
            out('judge: would start ' + shlex.join(judge_argv(G)))
            return 'judge: would start'
        p = runner(judge_argv(G), cwd=C.cfg['serving_workdir'], text=True, capture_output=True, timeout=600)
        job = (re.findall(r'(\d+)\.0', p.stdout or '') or [None])[-1]
        C.log(event='judge_start', registry=reg, rc=p.returncode, job=job, argv=judge_argv(G))
        if job and p.returncode == 0:
            js.setdefault('jobs', []).append(job)
            js['started_at'] = now()
        out(f"judge: started {reg} job {job} rc {p.returncode} {(p.stderr or '').strip()[-200:]}")
        return f'judge: started job {job}'
    jobs = {str(j['ClusterId']) for j in live}
    e = registry_fresh(reg)
    if busy_work:
        js.pop('idle_since', None)
        return f"judge {','.join(sorted(jobs))}: {'up' if e else 'starting'}, judged work queued or running"
    n = flight(e) if e else 0.0
    if n is None or n > 0:
        js.pop('idle_since', None)
        return f"judge {','.join(sorted(jobs))}: {'unreadable metrics' if n is None else f'{n:.0f} requests in flight'}"
    since = js.setdefault('idle_since', now())
    idle = now() - since
    if idle < G.j['idle_release_sec']:
        return f"judge {','.join(sorted(jobs))}: idle {idle:.0f} s (release at {G.j['idle_release_sec']} s)"
    mine = jobs & ours
    if not mine:
        return f"judge {','.join(sorted(jobs))}: idle, not started by {C.name}: left running"
    others = other_users(G, q, set(st.get('our_clusters') or []), jobs, ps)
    if others:
        return f"judge {','.join(sorted(jobs))}: idle here, kept for other users: {'; '.join(others)[:300]}"
    if dry:
        out(f"judge: would release {','.join(sorted(mine))} (idle {idle:.0f} s)")
        return 'judge: would release'
    for c in sorted(mine):
        r = runner(['condor_rm', c], capture_output=True, text=True, timeout=120)
        C.log(event='judge_release', registry=reg, job=c, rc=r.returncode, idle_sec=round(idle))
        out(f'judge: released {reg} job {c} after {idle:.0f} s idle (rc {r.returncode})')
    js.pop('idle_since', None)
    return f"judge: released {','.join(sorted(mine))}"


# ------------------------------------------------------------------------------------------------ passes
class Lock:
    """One mutating pass or watcher per campaign: an O_EXCL pidfile <grade dir>/watch.pid ("host pid"; flock is not
    used: /data returns ENOSYS for it). It is refreshed on every state save (several times a pass) and is stale when
    its process is gone (same host) or it was not refreshed for STALE_SEC, far above the longest step (30 min). A
    stale file is taken over by an atomic rename, so two starters can't both take it."""
    STALE_SEC = 4 * 3600
    current: 'Lock | None' = None

    def __init__(self, G: Grading):
        self.pid = G.dir / 'watch.pid'
        self.me = f'{socket.gethostname()} {os.getpid()}'
        self.mine = False

    def __enter__(self):
        self.pid.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.pid, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if not self._stale():
                    raise SystemExit(f'{self.pid}: another grade pass or watcher of this campaign is running '
                                     f'({self._holder()})')
                try:   # only one starter's rename succeeds
                    os.rename(self.pid, self.pid.with_name(f'watch.pid.stale.{self.me.replace(" ", ".")}'))
                except FileNotFoundError:
                    pass
                continue
            os.write(fd, (self.me + '\n').encode())
            os.close(fd)
            self.mine = True
            Lock.current = self
            return self
        raise SystemExit(f'{self.pid}: could not take the lock')

    def _holder(self) -> str:
        try:
            return self.pid.read_text().strip()
        except OSError:
            return ''

    def _stale(self) -> bool:
        try:
            host, pid = self._holder().split()
            age = now() - self.pid.stat().st_mtime
        except (OSError, ValueError):
            return True
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0)
                return False
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
        return age > self.STALE_SEC

    def refresh(self):
        if self.mine:
            try:
                os.utime(self.pid)
            except OSError:
                pass

    def __exit__(self, *exc):
        if self.mine and self._holder() == self.me:
            self.pid.unlink(missing_ok=True)
        self.mine = False
        if Lock.current is self:
            Lock.current = None


def tick(G: Grading, models: list[str], stage: str, parts=('collect', 'grade', 'judge'), dry: bool = False,
         runner=subprocess.run, out=print, ps=None, flight=in_flight) -> dict:
    st = G.load()
    q = Queue.fetch(G.user, runner)
    r = {}
    save = (lambda: None) if dry else (lambda: G.save(st))
    try:
        if 'collect' in parts:
            r['collect'] = collect_pass(G, q, st, models, stage, dry, runner, out)
            save()
        judge_up = registry_fresh(G.j['registry']) is not None and bool(
            q.by_batch(f"tasksynth-serve-{G.j['registry']}", (2,)))
        if 'grade' in parts or 'judge' in parts:
            gdry = dry or 'grade' not in parts
            g = grade_pass(G, q, copy.deepcopy(st) if gdry else st, models, stage, judge_up, gdry, runner,
                           out if 'grade' in parts else (lambda *_: None), save)
            if 'grade' in parts:
                r['grade'] = g
            st['our_clusters'] = sorted({c for recs in st['grade'].values() for x in recs.values()
                                         for c in x.get('clusters') or []})
            judged = [v for (m, s), v in g.items() if s in JUDGED]
            need = any(v in ('needs judge', 'would submit') or v.startswith('retry') for v in judged)
            # judged jobs of this campaign still queued or running, whatever this pass's models and stage
            ours_live = any(x.get('status') == 'submitted' and any(q.cluster(c) and q.cluster(c)[0].get('JobStatus') in LIVE
                                                                   for c in x.get('clusters') or [])
                            for k, recs in st['grade'].items() if k.rsplit('.', 1)[-1] in JUDGED
                            for x in recs.values())
            busy = need or ours_live or any(v in ('running', 'held', 'submitted') for v in judged)
            if 'judge' in parts:
                r['judge'] = judge_pass(G, q, st, need, busy, dry, runner, out, ps, flight)
    finally:
        save()
    return r


def finished(r: dict) -> bool:
    """Every collector and grade step ended (done or failed for good) and no judge job of this campaign is live."""
    c = all(v == 'done' or v.startswith('failed') for v in (r.get('collect') or {}).values())
    g = all(v in ('done', 'self-graded') or v.startswith('failed') for v in (r.get('grade') or {}).values())
    return c and g and not r.get('judge', '').startswith('ours:')


def watch(G: Grading, models: list[str], stage: str, every: float):
    with Lock(G) as lock:
        while True:
            lock.refresh()
            try:
                r = tick(G, models, stage, out=lambda s: print(f'{iso()} {s}', flush=True))
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:   # a transient condor or /data error
                print(f'{iso()} pass failed: {e!r}', flush=True)
                r = {}
            if r and finished(r):
                print(f'{iso()} every suite of {", ".join(models)} ({stage}) is graded or failed, judge not ours or '
                      f'released: exiting', flush=True)
                return
            time.sleep(every)


# ------------------------------------------------------------------------------------------------ status
def status(G: Grading, models: list[str], stage: str, runner=subprocess.run, ps=None, flight=in_flight) -> str:
    r = tick(G, models, stage, dry='status', runner=runner, out=lambda *_: None, ps=ps, flight=flight)
    st = G.load()
    lines = [f'{G.C.name} ({stage})']
    for model in models:
        lines.append(f'== {model}: collector {r["collect"].get(model)}')
        for suite in L.model_suites(G.C, model):
            v = r['grade'].get((model, suite))
            if v == 'would submit':
                v = 'ready: submits on the next pass'
            recs = st['grade'].get(f'{model}.{stage}.{suite}') or {}
            steps = ' '.join(f"{k}={x.get('status')}({','.join(x.get('clusters') or [])})" for k, x in recs.items())
            lines.append(f'   {suite:16} {v}' + (f'   [{steps}]' if steps else '')
                         + f'   aai-open {run_version(G, suite, G.C.run_id(model, suite, stage))}')
    js = st['judge']
    lines.append(f"judge {G.j['registry']}: {r['judge']}; started by this campaign: {js.get('jobs') or []}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('op', choices=['status', 'collect', 'grade', 'judge', 'watch'])
    ap.add_argument('models', nargs='*')
    ap.add_argument('--stage', choices=['full', 'canary'], default='full')
    ap.add_argument('--every', type=float, default=60.0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    G = Grading(L.Campaign(a.config))
    models = a.models or list(G.C.models)
    for x in models:
        if x not in G.C.models:
            raise SystemExit(f'unknown model {x}')
    if a.op == 'status':
        print(status(G, models, a.stage))
    elif a.op == 'watch':
        watch(G, models, a.stage, a.every)
    else:
        with Lock(G):
            r = tick(G, models, a.stage, (a.op,) if a.op != 'judge' else ('judge',), a.dry_run)
        for k, v in r.items():
            print(k, json.dumps({f'{x[0]} {x[1]}' if isinstance(x, tuple) else x: y for x, y in v.items()}, indent=1)
                  if isinstance(v, dict) else v)


if __name__ == '__main__':
    main()

"""M1's round driver (task_synthesis.pipeline.round, origin/task-synthesis e9ce566) for TB-Sci-lite tasks (#353).

Same one-shot author, package checks, harbor_ns build, offline sandbox checks, judge, claims and stop rules; patched:
  - the generator's system prompt and spec rendering (m5/tbsci_lite/prompts.py, specs.py), the judge's spec text and
    one more judge sentence on hidden truth;
  - the base image: local/tasksynth-sci:1 (base_image/);
  - every instruction ends with lite's closing sentence (prompts.CLOSING);
  - DeepSeek requests (author + judge) stay under this worker's cap in m5-coord/alloc.json (re-read every 60 s,
    split over the round's jobs), and transport errors are retried for ~1 h (server restarts);
  - rounds live under /data/aai/m5/tbsci_lite/rounds; jobs run from frozen checkouts of both trees.

  python -m m5.tbsci_lite.round_sci submit --round p1 --gen m5-ds --judge m5-ds [round options]
  python -m m5.tbsci_lite.round_sci report --round p1
Write <round>/specs.jsonl first (python -m m5.tbsci_lite.plan): the round never samples M1 specs.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from task_synthesis.pipeline import check as C
from task_synthesis.pipeline import judge as J
from task_synthesis.pipeline import llm as L
from task_synthesis.pipeline import package as K
from task_synthesis.pipeline import prompts as TP
from task_synthesis.pipeline import round as R
from task_synthesis.pipeline import specs as TS

from . import prompts as P
from . import specs as S
from .common import PY, ROOT, TS_CHECKOUT, bad_nodes, frozen_checkout, job_pythonpath, requirements_excluding

ALLOC = Path('/opt/aai/m5-coord/alloc.json')
BENCH = 'tbsci_lite'
BASE_REF = 'local/tasksynth-sci:1'
BASE_MANIFEST = Path('/data/aai/tbench/harbor_ns/bases/local_tasksynth-sci_1.manifest.json')
SHOWN_PIP = ('numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'statsmodels', 'sympy', 'networkx', 'numba',
             'h5py', 'netcdf4', 'xarray', 'pyarrow', 'openpyxl', 'tifffile', 'astropy', 'biopython', 'scikit-image',
             'pillow', 'pywavelets', 'emcee', 'lmfit', 'uncertainties', 'pint', 'cvxpy', 'pulp', 'ortools',
             'python-sat', 'shapely', 'pyproj', 'rdkit', 'ase', 'control', 'joblib', 'tqdm', 'pyyaml', 'requests',
             'pytest')


def system_prompt(minutes):
    m = json.loads(BASE_MANIFEST.read_text())
    have = {p.split('==')[0].lower(): p.split('==')[1] for p in m['pip']}
    pip = ', '.join(f'{p} {have[p]}' for p in SHOWN_PIP if p in have)
    return P.GEN_SYSTEM.format(base_ref=BASE_REF, tools=R.TOOLS, python=m['python'][0].replace('python', ''),
                               pip=pip, minutes=minutes)


def materialize(pkg, spec, provenance, task_dir, asset_tar=None):
    """M1's materialize, with lite's closing sentence appended to the instruction."""
    closing = P.CLOSING.format(seconds=spec.get('closing_seconds', 7200))
    text = pkg['instruction'].rstrip()
    if not text.endswith(closing):
        text += '\n\n' + closing
    return K.materialize({**pkg, 'instruction': text + '\n'}, spec, provenance, task_dir, asset_tar=asset_tar)


DS_ALIAS = 'alloc:ds'   # --gen/--judge alloc:ds: the DeepSeek servers alloc.json lists under ds.serving, as a pool


def ds_serving():
    try:
        names = list(json.loads(ALLOC.read_text())['ds']['serving'])
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    return names or ['m5-ds']


def patch_ds_alias():
    """The coordinator swaps DeepSeek servers by editing alloc.json ds.serving (m5-ds -> m5-ds2, 2026-11-12). The alias
    resolves to those servers on every refresh (each 300 s and after a connection error), so a server that joins is
    used and one that leaves drops out; with several it is task_synthesis's '+' pool (one served model)."""
    if getattr(L.Endpoint, '_sci_alias', False):
        return
    read, wait = L.Endpoint._read, L.wait_healthy

    def _read(self):
        if self.name != DS_ALIAS:
            return read(self)
        names = ds_serving()
        if len(names) == 1:
            return L.Endpoint._read_one(L.ROOT / 'serving' / names[0] / 'endpoints.json')
        saved, self.name = self.name, '+'.join(names)
        try:
            return read(self)
        finally:
            self.name = saved

    async def wait_healthy(name, *args, **kw):
        return await wait(('+'.join(ds_serving())) if name == DS_ALIAS else name, *args, **kw)

    L.Endpoint._read = _read
    L.wait_healthy = wait_healthy
    L.Endpoint._sci_alias = True


class AllocCap:
    """In-flight DeepSeek requests of this job <= alloc.json ds.max_inflight[bench] / jobs (re-read every 60 s)."""

    def __init__(self, jobs=1):
        self.jobs, self.cap, self.read_at, self.inflight = max(1, jobs), 1, 0.0, 0
        self.cond = None

    def current(self):
        if time.time() - self.read_at > 60:
            try:
                total = int(json.loads(ALLOC.read_text())['ds']['max_inflight'][BENCH])
                # requests this worker's other DeepSeek clients (scidata fit checks, decontam) hold: ROOT/ds_reserve.json
                try:
                    reserve = int(json.loads((ROOT / 'ds_reserve.json').read_text()).get('reserve', 0))
                except (OSError, ValueError):
                    reserve = 0
                self.cap = max(1, (total - reserve) // self.jobs)
            except (OSError, ValueError, KeyError):
                pass
            self.read_at = time.time()
        return self.cap

    async def __aenter__(self):
        if self.cond is None:
            self.cond = asyncio.Condition()
        async with self.cond:
            while self.inflight >= self.current():
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout=30)
                except asyncio.TimeoutError:
                    pass
            self.inflight += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.inflight -= 1
            self.cond.notify_all()


MAX_DATA_BYTES = 50 << 20   # #353: data <= 50 MB per task
_check = C.check


async def check(store, task_dir, *args, **kw):
    """M1's build + offline checks, plus the data-size ceiling: the built tree under the task's roots (the snapshot
    the judge sees) must stay within 50 MB."""
    chk = await _check(store, task_dir, *args, **kw)
    if chk.get('ok'):
        total = 0
        for line in (chk.get('snapshot') or {}).get('tree') or []:
            head = line.strip().split(None, 1)[0]
            total += int(head) if head.isdigit() else 0
        chk['data_bytes'] = total
        if total > MAX_DATA_BYTES:
            chk['ok'] = False
            chk.setdefault('errors', []).append(
                f'the task ships {total / 2**20:.0f} MB of materials; keep the data well under 50 MB (usually a few MB): '
                'reduce the sample count, resolution or precision, or subset the dataset')
    return chk


CAP = AllocCap()
_chat_tools = L.Endpoint.chat_tools


async def chat_tools(self, messages, tools, max_tokens=None, retries=6, **kw):
    """Endpoint.chat_tools under the alloc cap; a request that fails on transport for its 6 tries (~13 min) is
    re-queued for up to ~1 h more (a server restart or the 24-h refresh), re-reading endpoints.json meanwhile."""
    t0 = time.time()
    while True:
        async with CAP:
            try:
                return await _chat_tools(self, messages, tools, max_tokens=max_tokens, retries=retries, **kw)
            except RuntimeError as e:
                if 'LLM request failed' not in str(e) or time.time() - t0 > 4500:
                    raise
        print(f'{time.strftime("%H:%M:%S")} {self.name}: transport failures, waiting for the server', flush=True)
        await asyncio.sleep(120)
        self.refresh()


def _dp_size(path, info):
    """DP ranks behind a serving's one URL. M1's Endpoint._dp_size reads launch.json `argv`; serve_teacher now writes
    `replicas[].argv`, so it fell back to 1 and sent no X-data-parallel-rank header (author sessions unpinned)."""
    if len(info['urls']) != 1:
        return 1
    try:
        launch = json.loads((path.parent / 'launch.json').read_text())
        argv = launch.get('argv') or ((launch.get('replicas') or [{}])[0].get('argv')) or []
        if any(a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb', '--data-parallel-rank')):
            return 1
        return max(1, int(argv[argv.index('--data-parallel-size') + 1]))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return 1


# ---- one-shot judge repair (v2a: the judge rejected 21 of 31 model-spec tasks, mostly for real inconsistencies a
# revision can fix; M1 gives only its workspace author a judge-repair turn). With --judge-repairs N, a one-shot task
# that builds but fails the judge gets the reviewer's issues and N chances to return a corrected package, which is
# rebuilt, rechecked and rejudged. The record is written once, after the repairs, so a reader never sees the interim
# rejection. Replies go to jrepair-<j>.reply.txt (outside the attempt-<k>.reply.txt replay set).
_ORIG_LOOP = R.Round.attempts_loop
_ORIG_WRITE = R.Round.write_record


def _write_record(self, sid, rec, name='record.json'):
    if name == 'record.json' and sid in getattr(self, '_jr_defer', ()):
        return None
    return _ORIG_WRITE(self, sid, rec, name)


async def _judge_repairs(self, spec, wd, messages, rec, asset_tar, asset_judge, n):
    sid = spec['spec_id']
    last = [a for a in rec['attempts'] if a.get('stage') == 'check'][-1]
    prev = (wd / f'attempt-{last["k"]}.reply.txt').read_text()
    prov = {'model': self.gen.model, 'endpoint': self.gen.name, 'prompts': R.prompt_hash(), 'round': self.a.round}
    extra = self.asset_judge_extra if spec.get('asset') else None
    for jr in range(n):
        j = rec.get('judge') or {}
        if rec['status'] != 'valid' or j.get('pass') or 'scores' not in j:
            break
        rec.setdefault('judge_history', []).append(j)
        low = ', '.join(k for k, v in j['scores'].items() if v is not None and v < 4) or 'none'
        flags = ', '.join(k for k, v in (j.get('flags') or {}).items() if v) or 'none'
        issues = '\n'.join(f'- {x}' for x in j.get('issues') or []) or '- (none listed)'
        messages += [{'role': 'assistant', 'content': prev},
                     {'role': 'user', 'content': P.JUDGE_REPAIR.format(issues=issues, low=low, flags=flags)}]
        t0 = time.time()
        content, reasoning_chars, usage = await self.author_reply(messages, None, spec)
        (wd / f'jrepair-{jr}.reply.txt').write_text(content)
        att = {'k': len(rec['attempts']), 'stage': 'judge_repair', 'gen_seconds': round(time.time() - t0),
               'usage': usage, 'reasoning_chars': reasoning_chars}
        try:
            if usage.get('finish') == 'length':
                raise R.PackageError('reply hit the output length limit')
            pkg = R.parse(content)
        except R.PackageError as e:
            att.update(ok=False, errors=[str(e)])
            rec['attempts'].append(att)
            break
        chk = await self.checked(R.materialize(pkg, spec, prov, self.tasks / sid, asset_tar=asset_tar))
        R.atomic(wd / f'jrepair-{jr}.check.json', chk)
        att.update(ok=chk['ok'], errors=chk['errors'])
        rec['attempts'].append(att)
        if not chk['ok']:   # keep the task as judged
            R.materialize(R.parse(prev), spec, prov, self.tasks / sid, asset_tar=asset_tar)
            break
        async with self.judge_slots:
            rec['judge'] = await J.judge(self.judge_llm, spec, pkg, chk, asset=asset_judge, extra=extra)
        rec['meta'], rec['instruction'], rec['judge_repairs'] = pkg['meta'], pkg['instruction'], jr + 1
        prev = content
    return rec


async def _attempts_loop(self, spec, wd, messages, stored, old_usage, asset_tar, asset_judge, ex):
    n = getattr(self.a, 'judge_repairs', 0) if ex is None else 0
    if not n:
        return await _ORIG_LOOP(self, spec, wd, messages, stored, old_usage, asset_tar, asset_judge, ex)
    sid = spec['spec_id']
    self.__dict__.setdefault('_jr_defer', set()).add(sid)
    try:
        rec = await _ORIG_LOOP(self, spec, wd, messages, stored, old_usage, asset_tar, asset_judge, ex)
    finally:
        self._jr_defer.discard(sid)
    try:
        rec = await _judge_repairs(self, spec, wd, messages, rec, asset_tar, asset_judge, n)
    except R.LostClaim:
        raise
    except Exception as e:  # noqa: BLE001  (an error in a repair keeps the judged task)
        rec['judge_repair_error'] = f'{type(e).__name__}: {e}'
    self.write_record(sid, rec)
    return rec


def patch(jobs=1):
    R.ROUNDS = ROOT / 'rounds'
    K.BASE_REF = R.BASE_REF = BASE_REF
    R.BASE_MANIFEST = BASE_MANIFEST
    R.system_prompt = system_prompt
    R.materialize = materialize
    patch_ds_alias()
    # tasksynth's report() reads spec["assets"][0] whenever spec["asset"] is set; our real-data specs carry only
    # "asset", so prod1/prod2 crashed there after all records were written (exit 1); a KeyError there now only skips the report.
    if not getattr(R.report, '_sci', False):
        _report = R.report

        def report(rnd, *args, **kw):
            try:
                return _report(rnd, *args, **kw)
            except KeyError as e:   # report only: the records are already written
                print(f'report skipped: KeyError {e}', flush=True)
        report._sci = True
        R.report = report
    TP.GEN_SYSTEM = P.GEN_SYSTEM
    if P.JUDGE_EXTRA.strip() not in TP.JUDGE_SYSTEM:
        anchor = '\n\nAnswer with JSON only:'
        assert anchor in TP.JUDGE_SYSTEM, 'judge prompt anchor moved'
        TP.JUDGE_SYSTEM = TP.JUDGE_SYSTEM.replace(anchor, '\n\n' + P.JUDGE_EXTRA.strip() + anchor, 1)
    TS.render = S.render
    TS.spec_text = J.spec_text = S.spec_text
    CAP.jobs = max(1, jobs)
    L.Endpoint.chat_tools = chat_tools
    L.Endpoint._dp_size = staticmethod(_dp_size)
    C.check = check
    R.Round.attempts_loop = _attempts_loop
    R.Round.write_record = _write_record


def submit(a):
    """CPU job(s) running `python -m m5.tbsci_lite.round_sci run <a's options>` from frozen checkouts."""
    d = R.ROUNDS / a.round
    if not (d / 'specs.jsonl').exists():
        raise SystemExit(f'{d}/specs.jsonl missing: write it with m5.tbsci_lite.plan first')
    code = frozen_checkout()
    jobs = getattr(a, 'jobs', 1) or 1
    args = ' '.join(f'--{k.replace("_", "-")}' + ('' if v is True else f' {v}') for k, v in vars(a).items()
                    if k not in ('op', 'bid', 'cheap_bid', 'cpus', 'memory_gib', 'disk_gib', 'jobs', 'shard', 'shards')
                    and v is not None and v is not False)
    if jobs > 1:
        args += ' --shard $(Process) --shards ' + str(jobs)
    env = {'USER': os.environ.get('USER', 'user'), 'HOME': os.environ.get('HOME', '/home/user'),
           'PYTHONNOUSERSITE': '1', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
           'PYTHONPATH': job_pythonpath(code), 'HARBOR_NS_PERSIST_IMAGES': '0', 'HARBOR_NS_BUILD_CPUS': '2',
           **R.PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    excl = sorted(set(bad_nodes()) | set(R.ROUND_EXCLUDE))
    sub = d / 'job.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export HARBOR_NS_LOCAL=$_CONDOR_SCRATCH_DIR/harbor_ns TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $_CONDOR_SCRATCH_DIR/tmp; exec {PY} -u -m m5.tbsci_lite.round_sci run {args}'"
initialdir = {TS_CHECKOUT}
output = {d}/job.$(Cluster).$(Process).out
error = {d}/job.$(Cluster).$(Process).err
log = {d}/job.condor.log
request_cpus = {a.cpus}
request_memory = {a.memory_gib * 1024}
request_disk = {getattr(a, 'disk_gib', 200) + (getattr(a, 'package_cache_gib', 20) if getattr(a, 'package_cache', 'on') == 'on' else 0)}G
+JobBatchName = "m5-tbsci-{a.round}"
+MaxRuntime = 172800
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {requirements_excluding(excl)}
queue {jobs}
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    cluster = re.search(r'cluster (\d+)', r.stdout).group(1)
    R.atomic(d / 'job.json', {'job_id': cluster + '.0', 'jobs': jobs, 'code': str(code), 'ts_code': str(TS_CHECKOUT),
                              'time': time.time(), 'args': vars(a)})
    print(cluster)


def main():
    a = R.parse_args()
    patch(getattr(a, 'shards', 1) or 1)
    if a.op == 'run':
        R.run_and_exit(a)
    elif a.op == 'submit':
        submit(a)
    elif a.op == 'rejudge':
        asyncio.run(R.rejudge(a))
    else:
        R.report(a.round)


if __name__ == '__main__':
    main()

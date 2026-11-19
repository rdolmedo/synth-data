#!/usr/bin/env python3
"""Our Pi protocol (eval_q35 pi180: the tb21p180 / tbmix evals) against an external serving endpoint.

  python -m aa_evals.common.pi_eval.run --bench tb21 --run-id ID --model serving:aa-qwen --reasoning-effort medium \
      (--bundle SIX_HOUR.tar | --tasks-tar TASKS.tar) [--include a,b | --limit N] [--attempts 3] \
      [--max-in-flight 48] [--agent-cap-sec 7200] [--agent-wall-cap-sec 7200] [--dispatch DIR] [--multiuid] [--env-class MOD:CLASS] [--harbor-env K=V] [--seed-namespace NS] \
      [--agent pi|oracle] [--slots 24 --workers N]
  python -m aa_evals.common.pi_eval.run --bench tb21 --run-id ID --status | --score | --resubmit N | --stop

RUN = $AA_RUNS_ROOT/<bench>/<run_id>/ (default /data/aai/aa_evals/runs_fallback; never /home, which is not
for runs, logs or data: see aa_evals/README.md "Cost and storage rules") in eval_q35's run layout (config.json, queue/, attempts/<key>/trial.tar.gz + outcome.json, serve/) plus items.jsonl and summary.json (aa_evals contract; written by the last worker, or --score).
No serving job of our own: the workers (worker.py) keep serve/endpoint.url and queue/admission.json from the registry.

Protocol (eval_q35 common.PROTOCOL_PI180, unchanged): Pi 1.2.0 pi-cont16k with the bash180 bundle, 132,000-token
context with submit-at-limit (no compaction), 16,384 output tokens per request, temperature 1.0 / top_p 0.95 set by
the proxy, per-attempt seeds sha256("<namespace>/<task>/<attempt>"), task limit 21,600 s of environment time (model
time excluded) with a 2x wall-clock backstop, up to 6 same-seed infrastructure retries. --reasoning-effort goes on
every request (the proxy's own default, max, is outside Qwen3.8's template: xhigh|medium|low).
--agent-cap-sec C < 21600 scales the environment clock (PI_ENVCLOCK_MULT = C / 21600) on the six-hour bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from aa_evals.common import aai_open
from aa_evals.common import condor_cost
from aa_evals.common import diskcap
from aa_evals.common import serving
from aa_evals.common.harbor_mswea import endpoint
from aa_evals.common.harbor_mswea.queue import atomic, now, read
from aa_evals.common.harbor_mswea.run import EXCLUDE_NODES, exclude_nodes, snapshot_code, workers_alive
from . import bundle as bundle_mod
from . import hung, score, runtime_cache, model_input, task_agents, task_environment

HARNESS = 'pi180'
# M3.0 protocol (#173): regular evals cap the agent's context at 128k; the Qwen3.8 calibration runs pass
# --context-window 262144 (its native length) explicitly.
DEFAULT_CONTEXT_WINDOW = 131072
# teacher specs for url: model specs (--request-preset NAME): the [request] block of aa-serving's synth/teachers/NAME.toml
TEACHERS_DIR = Path(os.environ.get('AA_TEACHERS_DIR', '/opt/aai/worktrees/aa-serving/synth/teachers'))
KNOWN_ADAPTERS = {'inline_reasoning_history': 'PI_INLINE_REASONING_HISTORY',
                  # mirror_reasoning_content: always on in the Pi proxy's sampled path (inference/sticky_proxy.py)
                  'mirror_reasoning_content': None}
# AA leaderboard slugs of the M3.1 models (#174), by served-name substring and reasoning effort (None: any)
AA_SLUGS = [('qwen38-27b', 'medium', 'qwen3-8-27b-medium'), ('qwen38-27b', 'xhigh', 'qwen3-8-27b'),
            ('qwen38-27b', 'low', 'qwen3-8-27b-low'),
            ('deepseek-v41-flash-1017', 'max', 'deepseek-v4-1-flash-1017'),
            ('deepseek-v41-flash-1017', 'high', 'deepseek-v4-1-flash-1017-high'),
            ('super', None, 'nemotron-3-5-super-vl'), ('lightning', None, 'nemotron-3-5-lightning'),
            ('qwen36-35b', None, 'qwen3-6-35b-a3b'), ('qwen3.6-35b', None, 'qwen3-6-35b-a3b'),
            ('gemma-4-26b', None, 'gemma-4-26b-a4b'), ('gemma4-26b', None, 'gemma-4-26b-a4b'),
            ('qwen35-4b', None, 'qwen3-5-4b'), ('qwen3.5-4b', None, 'qwen3-5-4b')]


# #375: on for every new Pi run (A/B on Qwen3.8 Xhigh: CritPt agreement 13.6 -> 20.0, answered 36% -> 74%; GDPval
# Elo +98; user 2026-11-13 "make it the default"). `--finish-notice off` opts out.
DEFAULT_FINISH_NOTICE = '40000:16384'
FINISH_OFF = ('off', 'none', '0', '0:0')


def finish_notice(a, default=DEFAULT_FINISH_NOTICE):
    """protocol.finish_notice for a NEW run (#375): --finish-notice WARN_AT:RESERVE, the default for Pi runs when the
    flag is absent, or None for `off` (and for non-Pi agents unless the flag was given, which is an error there)."""
    explicit = getattr(a, 'finish_notice', None)
    spec = default if explicit is None else explicit
    if not spec or str(spec).strip().lower() in FINISH_OFF:
        return None
    if getattr(a, 'agent', 'pi') != 'pi':
        if explicit is None:
            return None
        raise SystemExit('--finish-notice is a Pi context-guard policy')
    from tb_agents import pi_context_limit as G
    try:
        warn, reserve = (int(x) for x in str(spec).split(':'))
    except ValueError:
        raise SystemExit(f'--finish-notice expects WARN_AT:RESERVE or off, got {spec!r}')
    try:
        G.finish_settings({'PI_FINISH_WARN_AT': str(warn), 'PI_FINISH_RESERVE': str(reserve)})   # validates
    except ValueError as e:
        raise SystemExit(f'--finish-notice: {e}')
    texts = G.FINISH_WARN + '\n' + G.FINISH_FINAL
    # v4 (#375 sftnotice): sticky notices, so a trial's final request is the linear trace the model saw and SFT
    # exporters can ship its notice map (finish-notices.jsonl). Runs created with v3 keep it (no 'sticky' key).
    return {'warn_at': warn, 'reserve': reserve, 'policy': G.STICKY_POLICY, 'sticky': True,
            'notice_margin': G.NOTICE_MARGIN,
            'final_slack': G.FINAL_SLACK, 'text_sha256': hashlib.sha256(texts.encode()).hexdigest(),
            # #382 re-review: the final turn starts at reserve + max(FINAL_SLACK, TURN // 2) (the A/B at cf17f3f used
            # reserve + FINAL_SLACK and has no final_at); recorded so the two plans are distinguishable
            'final_at': reserve + max(G.FINAL_SLACK, G.TURN // 2)}


def pi_deviations(a, protocol: dict) -> list[str]:
    """What a new Pi run can check against the AAI-open spec itself (aai_open.client_deviations): its recorded
    context window, finish notice and image inputs, and request settings given by hand (--request-overrides,
    --request-preset); the launcher checks the rest (plan, caps, bundles) and passes its label."""
    dev = aai_open.client_deviations(context_window=protocol.get('context_window'),
                                     overrides=json.loads(a.request_overrides) if a.request_overrides else None,
                                     notice=protocol.get('finish_notice'),
                                     images='on' if 'image' in (protocol.get('model_input') or []) else None)
    if a.request_preset:
        dev.append(f'request preset {a.request_preset} (v0.1: the registry [request] block)')
    return dev


def finish_notice_conflict(cfg, a):
    """Resume check: a stored run keeps the policy it was created with (runs from before the default have none; v3
    runs stay v3). Only an explicit --finish-notice whose WARN_AT:RESERVE differs from the stored values is refused:
    the flag sets those two, not the policy version (#375 sftnotice review NIT-1)."""
    if getattr(a, 'finish_notice', None) is None:
        return False
    stored, wanted = (cfg.get('protocol') or {}).get('finish_notice'), finish_notice(a)
    key = lambda fn: (fn['warn_at'], fn['reserve']) if fn else None   # noqa: E731
    return key(stored) != key(wanted)


def aa_slug_for(served: str, effort: str | None) -> str | None:
    s = (served or '').lower()
    for pat, eff, slug in AA_SLUGS:
        if pat in s and (eff is None or eff == effort):
            return slug
    return None


def resolve_request(ep: dict | None, preset: str, extra: str, effort: str | None) -> dict:
    """The model's request settings (M3.2): the serving registry's [request] block (endpoints.json `request`), or
    --request-preset (a teacher spec, for url: models), plus --request-overrides; --reasoning-effort replaces the
    effort at the top level and in chat_template_kwargs (where the template reads it). Nothing model-specific here."""
    req, source = {}, None
    if preset:
        import tomllib
        path = Path(preset) if preset.endswith('.toml') else TEACHERS_DIR / f'{preset}.toml'
        req, source = dict(tomllib.loads(path.read_text()).get('request') or {}), str(path)
    elif ep and ep.get('request'):
        req, source = dict(ep['request']), ep.get('registry')
    ov = json.loads(json.dumps(req.get('overrides') or {}))
    if extra:
        x = json.loads(extra)
        ctk = {**(ov.get('chat_template_kwargs') or {}), **(x.pop('chat_template_kwargs', None) or {})}
        ov.update(x)
        if ctk:
            ov['chat_template_kwargs'] = ctk
    if not ov:
        raise SystemExit('no request settings for this model: the registry has no [request] overrides; pass '
                         '--request-preset TEACHER or --request-overrides JSON (no built-in sampling defaults)')
    if effort:
        ov['reasoning_effort'] = effort
        if isinstance(ov.get('chat_template_kwargs'), dict) and 'reasoning_effort' in ov['chat_template_kwargs']:
            ov['chat_template_kwargs']['reasoning_effort'] = effort
    bad = [x for x in req.get('adapters') or [] if x not in KNOWN_ADAPTERS]
    if bad:
        raise SystemExit(f'unsupported request adapters {bad} (known: {sorted(KNOWN_ADAPTERS)})')
    return {'source': source, 'overrides': ov, 'adapters': list(req.get('adapters') or []),
            'pi_thinking': req.get('pi_thinking'), 'env': dict(req.get('env') or {}),
            'pi_compat': req.get('pi_compat')}
# Runs, worker condor logs and code snapshots live on /data, even when /data warns that it is over quota (user,
# 2026-11-13). /home is not for runs or logs: the 11-12 home-quota hit dropped condor event-log records and
# stopped c30. /opt/aai/aa_runs_fallback is now a symlink to this tree. AA_RUNS_ROOT overrides.
RUNS_ROOT = Path(os.environ.get('AA_RUNS_ROOT', '/data/aai/aa_evals/runs_fallback'))


def q35_common(code: str):
    sys.path.insert(0, str(Path(code) / 'eval_q35'))
    import common   # eval_q35/common.py of the frozen code
    return common


def bundle_tasks(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        return sorted({Path(m.name).parts[1] for m in tf.getmembers()
                       if len(Path(m.name).parts) == 3 and Path(m.name).parts[2] == 'task.toml'})


# Worker job shapes per benchmark (#378): aa_evals/pipeline.json `worker_shapes` (tbshared.pipeline_config) (slots, CPUs and RAM at the default
# slots, the CPU model and the basis of each), from c30 (big models, 340 worker jobs) and the #378 measurement run
# (memrun/report.md: Qwen3.5-4B, per-minute cgroup samples, 1x and 2x workers).
# - slots: bigger TB workers average out tool-call bursts (cores per slot p95 0.67 -> 0.41 from 16 to 32 slots);
#   TB-Sci-lite keeps 4 heavy sandboxes per worker.
# - CPUs = (base + per_slot x slots) / target (cpu_model): the measured mean with all slots busy at ~80% (tb21 and
#   tbsci_lite: c30's big-model fit; the rest: the measurement). The CPU request is a hard cpuset (affinity mask): the
#   slots of a worker share exactly its requested cores and never borrow idle cores of the node. Sizing at the
#   full-slot mean / 0.8, not the p95, is a deliberate trade-off: the slots mostly wait on vLLM, and some burst delay
#   is accepted for higher average utilization. At the current caps, in minutes with at least 75% of slots busy, about
#   3.7-9% of CPU work is deferred (GDPval 3.7%, CritPt 4-9%, TB 2.1 6%), under 1% of attempt time (coordinator, #378).
# - RAM = slots x (mem_gb / slots of the shape): what the workers need, ~2x the measured process memory (rss + shmem)
#   at full slots. Page cache needs no headroom: trial archives go to /data with O_DIRECT (fastio). TB 2.1 keeps
#   4 GB/slot (c30's big-model TB workers peaked near 100 GB RSS on 16 slots, before the sandbox cap); TB-Sci-lite
#   16 GB/slot (tasks declare up to 16 GB, x1.5).
# - At submission tbshared fills the shape to the binding resource where it is free (condor_cost.fill_plan, 1.5): RAM
#   up to 16 GB per CPU costs no units, so e.g. TB 2.1's 10c/128G goes out as 10c/160G unless that would start jobs on
#   a higher rung. The decision is in RUN/jobs/fill.jsonl; config.json keeps the shape asked for here. A memory hold
#   restarts the job with 25% more memory (condor_cost's memory net), at most 5 times.
# Unknown benchmarks: 0.5 CPU per slot, 16 GB per CPU. Explicit --slots / --worker-cpus / --worker-mem-gb win, but
# launchers should not pass CPUs or RAM (#451 lesson 5: M5's launchers passed pre-#378 shapes, e.g. GDPval workers
# used 2.8 of 8 CPUs and 10 of 128 GB): an explicit value more than SHAPE_WARN_FACTOR off the measured shape at the
# same slots is warned about at launch and recorded in config.json worker.shape_check.
DEFAULT_SLOTS = 24
DEFAULT_CPU_MODEL = {'base': 0.0, 'per_slot': 0.4, 'target': 0.8}     # 0.5 CPU per slot
SHAPE_WARN_FACTOR = 1.5
# Derived bench names of campaign runs (m5l_tbsci_lite_t2, m5l_gdpval_finish, m5l_critpt_t2fin, m5_critpt, gdpval_m5)
# take their base suite's shape (tbsci_lite, gdpval_pi, critpt_pi); before #451 they silently got the default.
DERIVED_PREFIX = re.compile(r'^m\d+[a-z0-9]*_')   # m5_, m5l_, m5l2_
# base suites named differently from their pipeline.json entry (M5's 'tb' finishing ran TB tasks on tb21's shape)
SHAPE_ALIASES = {'tb': 'tb21'}
DERIVED_SUFFIX = re.compile(r'(_m\d+[a-z]*|_t2fin|_t2|_finish|_resample|_pi)$')


def shape_key(bench: str, shapes: dict) -> str | None:
    """The worker_shapes key for `bench`: itself, else its base suite (campaign prefix/suffixes removed, then the
    suite itself or its Pi entry `<suite>_pi`); None when neither is listed."""
    if bench in shapes:
        return bench
    base = DERIVED_PREFIX.sub('', bench)
    while DERIVED_SUFFIX.search(base):
        base = DERIVED_SUFFIX.sub('', base)
    return next((k for k in (base, base + '_pi', SHAPE_ALIASES.get(base, '')) if k and k in shapes), None)


def bench_shape(bench: str) -> dict | None:
    """The benchmark's worker shape from the package's pipeline.json (aa_evals/pipeline.json; None: neither it nor
    its base suite is listed). A missing or malformed pipeline.json raises
    (tbshared.pipeline_config.PipelineConfigError) instead of falling back silently."""
    from tbshared import pipeline_config
    shapes = pipeline_config.load()['worker_shapes']
    key = shape_key(bench, shapes)
    return shapes[key] if key else None


def shape_check(bench: str, slots: int, cpus: int | None, mem_gb: int | None) -> dict:
    """Explicit CPUs / RAM against the measured shape at the same slots: {'shape_key', 'measured', 'warnings'}."""
    from tbshared import pipeline_config
    key = shape_key(bench, pipeline_config.load()['worker_shapes'])
    if not key:
        return {'shape_key': None, 'measured': None, 'warnings': [
            f'no measured worker shape for {bench} in pipeline.json: the generic default (0.5 CPU and 8 GB per slot) '
            f'unless given; add the suite to worker_shapes or SHAPE_ALIASES (#451)']}
    m_cpus, m_mem = worker_shape(bench, slots, None, None)
    warnings = []
    for flag, got, want in (('--worker-cpus', cpus, m_cpus), ('--worker-mem-gb', mem_gb, m_mem)):
        if got is not None and got > 0 and max(got / want, want / got) > SHAPE_WARN_FACTOR:
            warnings.append(f'{flag} {got} is {got / want:.2f}x the measured shape for {bench} ({key}: {want} at '
                            f'{slots} slots, pipeline.json); launchers should not pass worker CPUs/RAM (#451)')
    return {'shape_key': key, 'measured': {'cpus': m_cpus, 'mem_gb': m_mem}, 'warnings': warnings}


def default_slots(bench: str, slots: int | None) -> int:
    return slots or (bench_shape(bench) or {}).get('slots') or DEFAULT_SLOTS


def worker_shape(bench: str, slots: int, cpus: int | None, mem_gb: int | None) -> tuple[int, int]:
    """(CPUs, GB) for a worker job of `slots` slots: explicit values win; else the benchmark's pipeline.json shape,
    scaled to `slots` by its cpu_model and GB per slot. At least 2 CPUs."""
    shape = bench_shape(bench)
    if cpus is None:
        m = (shape or {}).get('cpu_model') or DEFAULT_CPU_MODEL
        cpus = max(2, math.ceil((m['base'] + m['per_slot'] * slots) / m['target'] - 1e-9))
    if mem_gb is None:
        mem_gb = math.ceil(slots * shape['mem_gb'] / shape['slots'] - 1e-9) if shape else 16 * cpus
    return cpus, mem_gb


LAUNCH_SHARE = 0.25   # with dispatcher growth, a lane starts with this share of the model's budget in worker slots
# (once its first workers heartbeat, the dispatcher's whole-budget shortfall may grow it to its longest-first share)


def lazy_workers(flag: bool | None, dispatch_dir: str | None) -> bool:
    """Lazy first workers (#378): the dispatcher submits a run's first workers at its first grant. On by default for a
    run under a dispatcher whose dispatch.json records lazy_workers (dirs made since); eager without a dispatcher (no
    credit to wait for) and under older dirs, so nothing live changes; --[no-]lazy-workers overrides."""
    if not dispatch_dir:
        if flag:
            raise SystemExit('--lazy-workers needs --dispatch: the dispatcher submits the first workers')
        return False
    if flag is not None:
        return bool(flag)
    try:
        return bool(json.loads((Path(dispatch_dir) / 'dispatch.json').read_text()).get('lazy_workers'))
    except (OSError, ValueError):
        return False


def initial_workers(n_items: int, max_in_flight: int, slots: int, dispatch_dir: str | None = None) -> int:
    """Workers to submit at run creation: enough for min(items, max in flight), or, when the run's dispatcher grows
    pi workers (dispatch.json pi_autoscale, #378), only LAUNCH_SHARE of its budget: five lanes sized at 60% of the
    budget each held ~3x the budget in idle slots in c30, and the dispatcher adds workers to a lane its slots limit."""
    want = min(n_items, max_in_flight)
    if dispatch_dir:
        try:
            dcfg = json.loads((Path(dispatch_dir) / 'dispatch.json').read_text())
        except (OSError, ValueError):
            dcfg = {}
        if dcfg.get('pi_autoscale') and dcfg.get('budget'):
            want = min(want, math.ceil(LAUNCH_SHARE * int(dcfg['budget'])))
    return max(1, math.ceil(want / max(1, slots)))


def submit_workers(run: Path, cfg: dict, n: int) -> str:
    jobs = run / 'jobs'
    jobs.mkdir(exist_ok=True)
    w = cfg['worker']
    excluded = exclude_nodes(EXCLUDE_NODES)   # the shared list now + this module's extras (m5 launchers add to them)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{x}")' for x in excluded)
    sub = jobs / f'worker-{time.strftime("%Y%m%dT%H%M%S")}.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = {cfg['code']}/aa_evals/common/pi_eval/worker_job.sh {run} {w['slots']}
request_cpus = {w['cpus']}
request_memory = {int(w['mem_gb'] * 1024)}
request_disk = {w['disk_gb']}G
+BypassLXCfs = "true"
coresize = -1
+JobBatchName = "aa-pi-{cfg['benchmark']}-{cfg['run_id']}"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin CONDOR_JOB_ID=$(Cluster).$(Process){serving.condor_env(cfg.get('serving_env') or {})}"
requirements = {req}
rank = (TARGET.CUDACapability =!= 10.0)
output = {jobs}/worker.$(Cluster).$(Process).out
error = {jobs}/worker.$(Cluster).$(Process).err
log = {jobs}/worker.log
queue {n}
''')
    # bid None (new runs): the CPU bid ladder, one condor cluster per starting rung; older runs keep their fixed bid
    try:
        recs = condor_cost.submit(sub, w.get('bid') or None, exclude=excluded)
    except condor_cost.SubmitError as e:   # record the rungs already queued, so nothing runs untracked
        _record_jobs(run, e.submitted)
        raise
    _record_jobs(run, recs)
    return recs[-1]['cluster']


def _record_jobs(run: Path, recs: list[dict]) -> None:
    if not recs:
        return
    rec = read(run / 'jobs.json', []) or []
    for x in recs:
        rec.append({'cluster': x['cluster'], 'n': x['n'], 'submit_file': x['submit_file'], 'bid': x['bid'],
                    'submitted_at': now()})
    atomic(run / 'jobs.json', rec)


def status(run: Path) -> dict:
    q = run / 'queue'
    names = lambda d, suf='': {p.name.removesuffix(suf) for p in (q / d).glob('*')} if (q / d).exists() else set()
    done, claims = names('done', '.json'), names('claims')
    recs = [read(q / 'done' / f'{k}.json', {}) or {} for k in done]
    s = {'run': str(run), 'items': len((q / 'items.txt').read_text().split('\n')) - 1,
         'physical_done': len(done), 'active': sorted(claims - done),
         'infra_attempts': sum(1 for r in recs if r.get('infrastructure_failure')),
         'passed': sum(1 for r in recs if not r.get('infrastructure_failure') and (r.get('score') or 0) >= 1),
         'admission': read(q / 'admission.json'), 'workers_alive': workers_alive(run)}
    if (run / 'summary.json').exists():
        s['summary'] = read(run / 'summary.json')
    return s


def build_parser(d: dict | None = None) -> argparse.ArgumentParser:
    d = d or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default=d.get('bench'), required='bench' not in d)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--model', default=None, help='serving:<name> | url:<base>@<served> (not needed for --agent oracle)')
    ap.add_argument('--reasoning-effort', default=d.get('reasoning_effort'),
                    help="replaces the registry's effort (top level and chat_template_kwargs); default: the model's "
                         "registry setting (M3.0: DeepSeek max, Qwen3.8 medium|xhigh given here, others reasoning on)")
    ap.add_argument('--request-preset', default='', help='teacher spec NAME (aa-serving synth/teachers/NAME.toml) or a '
                    '.toml path whose [request] block gives the request settings (for url: models; overrides the registry)')
    ap.add_argument('--request-overrides', default='', help='JSON merged over the request settings (recorded)')
    ap.add_argument('--bundle', default=d.get('bundle'), help='a six-hour bundle (tar with tasks/<slug>/), used as is')
    ap.add_argument('--tasks-tar', default=d.get('tasks_tar'), help='tasks tar/dir: rewritten to six-hour TOMLs (bundle.py)')
    ap.add_argument('--include', default=d.get('include', ''),
                    help='comma-separated tasks (default: the benchmark\'s headline task set, if it has one); '
                         '--include all runs every task of the bundle')
    ap.add_argument('--exclude', default=','.join(d.get('exclude', [])))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--attempts', '--repeats', type=int, default=d.get('attempts', 5), dest='attempts',
                    help='repeats per task (M3.0: 5)')
    ap.add_argument('--context-window', type=int, default=d.get('context_window', DEFAULT_CONTEXT_WINDOW),
                    help="the agent's context window in tokens (M3.0 default 131072 = 128k; 262144 only for the "
                         "Qwen3.8 calibration runs; <= the server's max_model_len). The episode ends at the window")
    ap.add_argument('--max-in-flight', type=int, default=d.get('max_in_flight', 48))
    ap.add_argument('--burst', type=int, default=8, help='at most this many attempt starts per minute')
    ap.add_argument('--agent-cap-sec', type=float, default=d.get('agent_cap_sec', 21600.0))
    ap.add_argument('--agent-wall-cap-sec', type=float, default=d.get('agent_wall_cap_sec'),
                    help="#344: wall-clock cap on the agent's run (from agent start; not env build or verifier): "
                         "Harbor's agent timeout (the backstop) becomes this, the environment clock stays "
                         "--agent-cap-sec. A capped attempt ends with AgentTimeoutError and the verifier scores it")
    ap.add_argument('--finish-notice', default=d.get('finish_notice'), metavar='WARN_AT:RESERVE|off',
                    help="#375 finish notice (Pi; default %s for new runs, `off` to disable): the context guard appends "
                         "a harness notice with the tokens left once at most max(WARN_AT, RESERVE + 16384) remain, and "
                         "keeps the last RESERVE tokens for a final turn ('this is your final turn'). The conversation, reasoning included, is "
                         "never edited. Recorded in protocol.finish_notice; a resumed run keeps its stored policy (an "
                         "explicit flag must match it)" % DEFAULT_FINISH_NOTICE)
    ap.add_argument('--dispatch', default=d.get('dispatch'), metavar='DIR',
                    help='#344 dispatcher DIR (dispatcher.py init): items.txt longest predicted first, admission '
                         '0 until the dispatcher grants slots (RUN/control.json), lane registered under DIR/lanes/')
    ap.add_argument('--dispatch-lane', default=None, help='lane name under the dispatcher (default: <bench>.<run-id>)')
    ap.add_argument('--sched-state', default=d.get('sched_state'), metavar='FILE',
                    help='#451 tbsched: CAMPAIGN/sched/runs/<run>.json. Workers claim within its start credit per '
                         'engine and pin each attempt to the engine with the most unspent credit (sched_router.py), '
                         'while it is in mode "on" and fresh; otherwise the old rules (default: off)')
    ap.add_argument('--dispatch-suite', default=None,
                    help='duration-table suite of this run (default: --bench without a trailing _pi, e.g. '
                         'gdpval_pi -> gdpval); recorded in config dispatch.suite')
    ap.add_argument('--per-task-cap', action=argparse.BooleanOptionalAction, default=d.get('per_task_cap', False),
                    help="each task's limit = min(its original [agent] timeout_sec, --agent-cap-sec): environment "
                         'clock and wall backstop scaled per task (default: --agent-cap-sec for every task)')
    ap.add_argument('--multiuid', action=argparse.BooleanOptionalAction, default=d.get('multiuid', True))
    ap.add_argument('--verifier-toolchain-bundle', default=d.get('verifier_toolchain_bundle'),
                    help="#531 B5: stage the offline verifier (eval_q35 verifier_toolchain.py) from this bundle's "
                         "manifest records, for a bundle derived from it (AA's bundle-aptfix.tar from tb21_6h): only "
                         "tasks whose test.sh is byte-identical to the validated one go offline. Default: the "
                         "benchmark's (none = the bundle's own stem)")
    ap.add_argument('--env-class', default=d.get('env_class',
                    'aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment'))
    ap.add_argument('--harbor-env', action='append', default=[], metavar='KEY=VAL',
                    help="extra environment for each attempt's run_harbor.sh (repeatable; e.g. HARBOR_NS_MEMCAP=1)")
    ap.add_argument('--diskcap', default=None, metavar='auto|FACTOR[:MIN_MB]|off',
                    help='#405 per-sandbox disk cap (harbor_ns/diskcap.py; on by default, as eval_q35 since #372): no '
                         'file grows past max(storage_mb x FACTOR, MIN_MB) MB (MIN_MB for tasks without storage_mb), '
                         'and a box whose overlay upper dir is above it loses the process writing the most, instead '
                         'of filling the worker\'s shared scratch. "off": no cap (config "diskcap": null). Default '
                         '"auto": the bench\'s measured per-task caps (eval_q35 DISKCAP_FAMILIES; TB 2.1: rstan-to-pystan '
                         '90 GiB, one per worker, sam-cell-seg 40 GiB, others 32 GiB), 3:30720 for a bench without them')
    ap.add_argument('--required-output-capture-root', default='',
                    help='opt-in independent Lustre evidence for fixed FastText/Caffe outputs after agent exit')
    ap.add_argument('--seed-namespace', default=d.get('seed_namespace'))
    ap.add_argument('--ctx-gate', action=argparse.BooleanOptionalAction, default=None,
                    help="eval_q35's context-aware turn gate (+ the proxies' KV brake), hosted by the run's workers; "
                         "default: the serving registry's request.ctx_gate")
    ap.add_argument('--ctx-fill', type=float, default=None, help='turn gate start fill (default: registry, else 0.7)')
    ap.add_argument('--agent', default='pi', choices=['pi', 'oracle'])
    ap.add_argument('--agent-internet', action=argparse.BooleanOptionalAction, default=False,
                    help='agent egress (M3.5/M3.6: disabled); setup and grading keep their own network policies')
    ap.add_argument('--slots', type=int, default=None,
                    help='attempts per CPU worker job; default: per benchmark or its base suite (aa_evals/pipeline.json, else 24)')
    ap.add_argument('--shape-key', default=None,
                    help="the aa_evals/pipeline.json worker_shapes entry to size workers by, when the workload differs "
                         "from the bench's own (e.g. tbsci_lite_synth for synthesized TB-Sci-lite traces); default: "
                         "the bench or its base suite (#451)")
    ap.add_argument('--workers', type=int, default=0, help='default: enough for min(items, max in flight)')
    ap.add_argument('--worker-cpus', type=int, default=None,
                    help="default (launchers should not pass it, #451): the benchmark's measured CPU model at ~80%% "
                         "(aa_evals/pipeline.json, a derived name's base suite), else 0.5 per slot; at least 2")
    ap.add_argument('--worker-mem-gb', type=int, default=None,
                    help='default (launchers should not pass it, #451): slots x the benchmark\'s GB per slot (aa_evals/pipeline.json), else 16 GB per CPU; tbshared fills it to 16 GB per CPU where free. Never size it from condor '
                         "RSS: the sandboxes' tmpfs and page cache count against the job's cgroup limit")
    ap.add_argument('--worker-disk-gb', type=int, default=300)
    ap.add_argument('--cpu-bid', type=int, default=None,
                    help='fixed bid for the worker jobs; default: the CPU bid ladder 26 -> 36 -> 56 (common/condor_cost.py)')
    ap.add_argument('--aa-slug', default=d.get('aa_slug'))
    ap.add_argument('--aa-metric-key', default=d.get('aa_metric_key'))
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--resubmit', type=int, default=0, help='submit N more workers to an existing run')
    ap.add_argument('--stop', action='store_true', help='workers stop claiming (RUN/STOP_WORKERS); running attempts finish')
    ap.add_argument('--runtime-cache-manifest', default=d.get('runtime_cache_manifest', runtime_cache.DEFAULT_MANIFEST),
                    help='pinned public runtime package cache manifest (evaluator staging; agents install offline)')
    ap.add_argument('--runtime-cache-manifest-sha256', default=d.get('runtime_cache_manifest_sha256', runtime_cache.DEFAULT_MANIFEST_SHA256))
    ap.add_argument('--model-images', choices=('auto', 'on', 'off'), default='auto',
                    help='Pi input metadata: resolve actual serving launch by default; explicit declarations '
                         'support unknown URL endpoints but cannot contradict known launch capability')
    ap.add_argument('--no-submit', action='store_true')
    ap.add_argument('--pack-claims', action=argparse.BooleanOptionalAction, default=True,
                    help='#531: when the claimable work fits in the free slots of fuller live workers, a worker '
                         'leaves it to them, so emptied workers exit (claim_pack.py). Recorded in config.json; runs '
                         'made before it keep the old behavior. Off under --sched-state (tbsched decides)')
    ap.add_argument('--hung-watchdog-sec', type=float, default=d.get('hung_watchdog_sec'),
                    help="#531 A5: hung-session watchdog (hung.py): end a hung attempt as infrastructure failure "
                         "hung_session (same-seed retry). This sets the idle rule's limit: Pi doing nothing (no request "
                         "in flight, no tool running) this long. Default for new runs: max(3 x the suite's p99 idle "
                         "gap, 600 s) from request_gaps.json. 0 turns the whole watchdog off (all three rules). "
                         "Recorded in config.json (hung_watchdog); runs made before it run without")
    ap.add_argument('--hung-stall-sec', type=float, default=d.get('hung_stall_sec'),
                    help="#531 A5: the stall rule's limit: no progress (a reply, or the response-budget guard "
                         "completing a turn) since the oldest unanswered request for this long. Default: "
                         "request_gaps.json stall_sec (3660 s, two of Pi's 1800 s timeouts); 0 = stall rule off")
    ap.add_argument('--lazy-workers', action=argparse.BooleanOptionalAction, default=None,
                    help='#378: submit no worker at creation; the dispatcher submits the first --workers (default '
                         'launch-small) when it first grants this run credit. Default: on under a dispatcher whose '
                         'dispatch.json records lazy_workers (new dirs), off without --dispatch or under older dirs')
    return ap


def toolchain_offline_tasks(stem: str, names) -> int:
    """#531 B5: how many of `names` have an offline record under `stem` in eval_q35's verifier toolchain manifest
    (a mistyped stem would silently stage nothing). Stdlib: the module is loaded by path, as the worker does."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_rt_verifier_toolchain', Path(__file__).resolve().parents[3] / 'eval_q35' / 'verifier_toolchain.py')
    vt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vt)
    tasks = json.loads((Path(vt.TOOLCHAIN) / 'manifest.json').read_text())['tasks']
    return sum(1 for t in names if (tasks.get(f'{stem}/{t}') or {}).get('offline'))


def main(argv=None, defaults: dict | None = None) -> None:
    d = defaults or {}
    a = build_parser(d).parse_args(argv)
    shape_bench = a.shape_key or a.bench
    if a.shape_key:   # an exact entry: only bench names get the derived-name matching
        from tbshared import pipeline_config
        if a.shape_key not in pipeline_config.load()['worker_shapes']:
            raise SystemExit(f'--shape-key {a.shape_key}: no such worker_shapes entry in aa_evals/pipeline.json')
    a.slots = default_slots(shape_bench, a.slots)
    explicit = (a.worker_cpus, a.worker_mem_gb)
    a.worker_cpus, a.worker_mem_gb = worker_shape(shape_bench, a.slots, a.worker_cpus, a.worker_mem_gb)
    run = RUNS_ROOT / a.bench / a.run_id
    if a.status:
        print(json.dumps(status(run), indent=1, default=str))
        return
    if a.score:
        print(json.dumps(score.finalize(run), indent=1))
        return
    if a.stop:
        (run / 'STOP_WORKERS').write_text('stop\n')
        print('workers stop claiming; running attempts finish')
        return
    if (run / 'config.json').exists():
        cfg = read(run / 'config.json')
        diskcap.check_stored(cfg, a.diskcap, d.get('diskcap_family'), cfg.get('task_names') or ())
        if a.required_output_capture_root:
            frozen_capture = cfg.get('required_output_capture') or {}
            if frozen_capture.get('root') != str(Path(a.required_output_capture_root).resolve()):
                raise SystemExit('required-output capture policy differs from this stored run; use a new run ID')
        frozen_input = model_input.frozen_modalities(cfg)
        task_agents.frozen(cfg)
        task_environment.validate(cfg)
        if a.model_images != 'auto' and (frozen_input or ['text']) != (
                ['text', 'image'] if a.model_images == 'on' else ['text']):
            raise SystemExit('model input policy differs from this stored run; use a new run ID')
        if cfg.get('agent') == 'pi' and cfg.get('agent_internet', True) != a.agent_internet:
            raise SystemExit('agent internet policy differs from this stored run; use a new run ID')
        expected_cache = a.runtime_cache_manifest if cfg.get('agent') == 'pi' and not a.agent_internet else ''
        expected_cache_sha = a.runtime_cache_manifest_sha256 if expected_cache else ''
        if (cfg.get('runtime_cache_manifest', ''), cfg.get('runtime_cache_manifest_sha256', '')) != (expected_cache, expected_cache_sha):
            raise SystemExit('runtime package cache policy differs from this stored run; use a new run ID')
        if finish_notice_conflict(cfg, a):
            raise SystemExit('finish notice policy differs from this stored run; use a new run ID')
        if a.verifier_toolchain_bundle and a.verifier_toolchain_bundle != cfg.get('verifier_toolchain_bundle'):
            print(f"WARNING: --verifier-toolchain-bundle {a.verifier_toolchain_bundle} ignored: this run keeps its "
                  f"stored {cfg.get('verifier_toolchain_bundle')!r} (#531 B5)", file=sys.stderr)
        if a.hung_watchdog_sec is not None or a.hung_stall_sec is not None:
            print(f"WARNING: --hung-watchdog-sec / --hung-stall-sec ignored: this run keeps its stored hung_watchdog "
                  f"{cfg.get('hung_watchdog')!r} (#531 A5)", file=sys.stderr)
        if cfg.get('dispatch'):
            from aa_evals.common.dispatch import pilane
            pilane.register(cfg['dispatch'], run)
        # a lazy run before its first grant has no worker on purpose: a relaunch leaves that to the dispatcher
        waiting = cfg.get('lazy_workers') and not (run / 'jobs.json').exists()
        n = a.resubmit or (0 if workers_alive(run) or waiting else cfg['worker']['n'])
        if n and not a.no_submit:
            submit_workers(run, cfg, n)
        print(json.dumps(status(run), indent=1, default=str))
        return

    # ---- new run
    a.shape_check = shape_check(shape_bench, a.slots, *explicit)   # (an existing run keeps its stored shape)
    for w in a.shape_check['warnings']:
        print(f'WARNING worker shape: {w}', file=sys.stderr, flush=True)
    if a.agent == 'pi':
        if not a.model:
            raise SystemExit('--model is required for a Pi run')
        ok, ep, why = endpoint.alive(a.model)
        if not ok and not a.no_submit:
            raise SystemExit(f'model {a.model} not usable now: {why} (the workers wait for it; --no-submit to prepare)')
        ep = ep or endpoint.resolve(a.model)
        served = ep['served']
        inputs = model_input.resolve(ep, a.model_images)
        request = resolve_request(ep, a.request_preset, a.request_overrides, a.reasoning_effort)
        a.reasoning_effort = request['overrides'].get('reasoning_effort') or ''
        a.aa_slug = a.aa_slug or aa_slug_for(served, a.reasoning_effort)
        if ok and a.context_window:
            mml = endpoint.max_model_len(ep['urls'][0], served)
            if mml and a.context_window > mml:
                raise SystemExit(f'--context-window {a.context_window} > the server max_model_len {mml}')
    else:
        served, a.model, request = 'oracle', a.model or 'none', None
        inputs = None
    if not a.seed_namespace:
        raise SystemExit('--seed-namespace is required (per benchmark, fixed across models)')
    if a.bundle:
        bpath = Path(a.bundle)
        manifest = read(bpath.with_suffix('.json'), {}) or {}
    elif a.tasks_tar:
        src = Path(a.tasks_tar)
        key = bundle_mod.sha256(src)[:12] if src.is_file() else src.name
        bpath = RUNS_ROOT / a.bench / 'bundles' / f'pi_six_hour_{key}.tar'
        bpath.parent.mkdir(parents=True, exist_ok=True)
        if not bpath.exists():
            bundle_mod.build(src, bpath)
        manifest = read(bpath.with_suffix('.json'), {}) or {}
    else:
        raise SystemExit('--bundle or --tasks-tar is required')
    all_tasks = bundle_tasks(bpath)
    excluded = [t for t in all_tasks if t in set(filter(None, a.exclude.split(',')))]
    names = [t for t in all_tasks if t not in excluded]
    if a.include and a.include != 'all':
        inc = [t for t in a.include.split(',') if t]
        missing = set(inc) - set(all_tasks)
        if missing:
            raise SystemExit(f'unknown tasks: {sorted(missing)}')
        names = [t for t in names if t in inc]
    if a.limit:
        names = names[:a.limit]
    if a.verifier_toolchain_bundle:   # #531 B5: a stem with no offline records for these tasks is a typo, not a choice
        k = toolchain_offline_tasks(a.verifier_toolchain_bundle, names)
        if not k:
            raise SystemExit(f'--verifier-toolchain-bundle {a.verifier_toolchain_bundle}: no offline toolchain record '
                             f'for any of the {len(names)} tasks')
        print(f'verifier toolchain {a.verifier_toolchain_bundle}: offline records for {k} of {len(names)} tasks '
              '(the test.sh sha check at staging decides)')
    code = snapshot_code(RUNS_ROOT / 'code')
    common = q35_common(code)
    protocol = dict(common.PROTOCOLS[HARNESS])
    protocol_overrides = {}
    if a.context_window and a.context_window != protocol['context_window']:
        # eval_q35 derives MAX_LEN / PI_CONTEXT* (Pi, proxy, context guard) from this one value
        protocol_overrides['context_window'] = {'from': protocol['context_window'], 'to': a.context_window}
        protocol['context_window'] = a.context_window
    # the eval_q35 constants carry the pre-M3 Qwen values; record what this run actually uses
    protocol['attempts'] = a.attempts
    protocol['agent_internet'] = a.agent_internet
    if inputs is not None:
        protocol['model_input'] = inputs['modalities']
    if request:   # eval_q35's worker reads these keys; the proxy's overrides are what reaches the server
        for k in ('temperature', 'top_p'):
            if k in request['overrides']:
                protocol[k] = request['overrides'][k]
        protocol['pi_thinking'] = request.get('pi_thinking') or protocol['pi_thinking']
    if a.agent_cap_sec > protocol['agent_timeout_sec']:
        raise SystemExit('--agent-cap-sec above the six-hour protocol limit')
    if finish_notice(a):
        protocol['finish_notice'] = finish_notice(a)
    if a.agent_wall_cap_sec is not None:
        if not 0 < a.agent_wall_cap_sec <= protocol['agent_timeout_sec'] * 2.0:
            raise SystemExit('--agent-wall-cap-sec must be in (0, 2 x the six-hour limit]')
        protocol['agent_wall_cap_sec'] = a.agent_wall_cap_sec
    dispatch_rec, names_attempts = None, None
    if a.dispatch and a.sched_state:
        raise SystemExit('--dispatch and --sched-state both grant start credit: pick one')
    if a.dispatch:
        from aa_evals.common.dispatch import pilane
        dispatch_rec, names_attempts = pilane.plan(a.dispatch, a.dispatch_lane or f'{a.bench}.{a.run_id}',
                                                   a.dispatch_suite or a.bench.removesuffix('_pi'), names, a.attempts)
        pilane.check_owner(dispatch_rec, run)   # before the run dir exists
    mult = a.agent_cap_sec / protocol['agent_timeout_sec']
    req = (ep.get('request') or {}) if a.agent == 'pi' else {}
    ctx_gate = bool(req.get('ctx_gate')) if a.ctx_gate is None else a.ctx_gate
    if a.agent != 'pi':
        ctx_gate = False
    n_items = len(names) * a.attempts
    # --no-submit: no worker at all, not even at the first grant (prepared while the model is down, or by hand)
    a.lazy_workers = lazy_workers(a.lazy_workers, a.dispatch) and not a.no_submit
    n_workers = a.workers or initial_workers(n_items, a.max_in_flight, a.slots, a.dispatch)
    cache_path = a.runtime_cache_manifest if a.agent == 'pi' and not a.agent_internet else ''
    cache_sha = a.runtime_cache_manifest_sha256 if cache_path else ''
    if cache_path:
        runtime_cache.load_manifest(cache_path, cache_sha)
    diskcap.check_harbor_env({**(d.get('harbor_env') or {}), **dict(x.split('=', 1) for x in a.harbor_env)})
    a.diskcap = a.diskcap if a.diskcap is not None else d.get('diskcap', diskcap.DEFAULT_ARG)
    try:
        cap = diskcap.arg(a.diskcap, d.get('diskcap_family'), names)
    except ValueError:
        raise SystemExit(f'--diskcap {a.diskcap!r}: expected FACTOR[:MIN_MB] or off') from None
    cfg = {
        # eval_q35 run config (worker.py / queue_state.py read these)
        'name': f'{a.bench}-{a.run_id}', 'created_at': now(), 'model': a.model, 'served': served,
        'harness': HARNESS, 'protocol': protocol, 'protocol_overrides': protocol_overrides, 'reasoning_effort': a.reasoning_effort if a.agent == 'pi' else '',
        'benchmark': a.bench, 'seed_namespace': a.seed_namespace, 'bundle': str(bpath),
        'bundle_sha256': bundle_mod.sha256(bpath), 'tasks': len(names), 'attempts': a.attempts,
        'envclock': True, 'envclock_mult': mult, 'backstop_multiplier': 2.0, 'kv_gate': (req.get('kv_gate') or '0.80,0.72') if ctx_gate else '', 'ctx_gate': ctx_gate,
        'ctx_fill': float(a.ctx_fill or req.get('ctx_fill') or 0.7),
        'code': code, 'code_commit': (read(Path(code) / '.complete', {}) or {}).get('git_sha'),
        # pi_eval additions
        'run_id': a.run_id, 'model_spec': a.model, 'agent': a.agent, 'agent_internet': a.agent_internet,
        'harness_label': f'pi-1.2.0-{HARNESS}',
        'runtime_cache_manifest': cache_path, 'runtime_cache_manifest_sha256': cache_sha,
        'per_task_cap': a.per_task_cap,
        'envclock_mult_by_task': {t: min(float(o or a.agent_cap_sec), a.agent_cap_sec) / protocol['agent_timeout_sec']
                                  for t, o in ((t, (manifest.get('tasks') or {}).get(t, {}).get(
                                      'original_agent_timeout_sec')) for t in names)} if a.per_task_cap else {},
        'max_in_flight': a.max_in_flight, 'burst': a.burst, 'agent_cap_sec': a.agent_cap_sec,
        'agent_wall_cap_sec': a.agent_wall_cap_sec,
        'dispatch': dispatch_rec,
        **({'sched': {'file': str(Path(a.sched_state).resolve())}} if a.sched_state else {}),
        'multiuid': a.multiuid, 'env_class': a.env_class,
        # #531 B5: the offline verifier toolchain's bundle for this run's tasks (None: the bundle's own stem)
        'verifier_toolchain_bundle': a.verifier_toolchain_bundle,
        'harbor_env': {**(d.get('harbor_env') or {}), **dict(x.split('=', 1) for x in a.harbor_env)},
        'diskcap': cap,   # #405; aa_evals/common/diskcap.py (null = off)
        'aa_slug': a.aa_slug, 'aa_metric_key': a.aa_metric_key,
        'oracle_run': d.get('oracle_run'),
        'subset': bool(a.include or a.limit), 'task_names': names, 'excluded_tasks': excluded,
        'original_agent_timeouts': {t: (manifest.get('tasks') or {}).get(t, {}).get('original_agent_timeout_sec')
                                    for t in names},
        'categories': d.get('categories_fn', lambda b: {})(bpath),
        'registry_request': ep.get('request') if a.agent == 'pi' else None,
        'registry_dir': str(Path(ep['registry']).parent) if a.agent == 'pi' and ep.get('registry') else None,
        # the registry root this run reads ({} = production): forwarded into every worker job, also the ones the
        # dispatcher submits later from another process
        'serving_env': serving.forwarded(),
        # M3.2: the settings every sampled request gets (worker.py -> PI_REQUEST_OVERRIDES -> the Pi proxy)
        'request': request,
        'lazy_workers': bool(a.lazy_workers),   # the dispatcher submits the first workers at the first grant
        # #531: scarce work to the fullest workers (claim_pack.py); never with tbsched (its lifecycle decides)
        'pack_claims': bool(a.pack_claims) and not a.sched_state,
        # #531 A5: hung-session watchdog ({'idle_sec', 'p99_gap_sec', 'source'}; None = off, and for the oracle)
        'hung_watchdog': hung.resolve(a.bench, a.hung_watchdog_sec, stall_override_sec=a.hung_stall_sec)
        if a.agent == 'pi' else None,
        'worker': {'n': n_workers, 'slots': a.slots, 'cpus': a.worker_cpus, 'mem_gb': a.worker_mem_gb,
                   'disk_gb': a.worker_disk_gb, 'bid': a.cpu_bid,   # None: condor_cost.LADDER
                   'shape_check': a.shape_check,
                   **({} if a.cpu_bid else {'bid_ladder': list(condor_cost.LADDER)})},
        'notes': d.get('notes', ''),
    }
    if a.agent == 'pi':   # #481: the AAI-open version this run claims (the launcher's label + what this run shows)
        cfg['aai_open'] = aai_open.record(pi_deviations(a, protocol))
    if a.required_output_capture_root:
        from .required_outputs import SPECS, VERSION
        capture_root = Path(a.required_output_capture_root).resolve()
        if (a.agent != 'pi' or not capture_root.is_relative_to('/lustre')
                or a.env_class != 'aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment'):
            raise SystemExit('required-output capture needs Pi and an independent absolute Lustre path')
        cfg['required_output_capture'] = {
            'schema_version': VERSION, 'root': str(capture_root),
            'tasks': sorted(set(names) & set(SPECS)),
        }
    if a.agent == 'pi' and task_agents.FIELD in d:
        dispatch = task_agents.selected(d[task_agents.FIELD], names)
        cfg[task_agents.FIELD] = dispatch
        protocol[task_agents.FIELD] = dict(dispatch)
        for task in names:
            for env in ((request or {}).get('env') or {}, cfg.get('harbor_env') or {}):
                task_agents.apply_env(cfg, task, dict(env))
    if inputs is not None:
        cfg['model_input'] = inputs
        # Catch configured reserved overrides before writing a new run; the
        # worker also checks its actual inherited/merged environment per attempt.
        for env in ((request or {}).get('env') or {}, cfg.get('harbor_env') or {}):
            model_input.apply_env(cfg, dict(env))
    policy = task_environment.new_policy(a.bench, a.agent, a.agent_internet, names)
    if policy is not None:
        cfg['task_environment_policy'] = protocol['task_environment_policy'] = policy
        cfg['task_environment_policy_sha256'] = protocol['task_environment_policy_sha256'] = task_environment.digest(policy)
    task_environment.validate(cfg)
    run.mkdir(parents=True, exist_ok=False)
    for sub in ('queue', 'queue/claims', 'queue/done', 'queue/retry', 'attempts', 'serve', 'workers'):
        (run / sub).mkdir()
    order = names_attempts if dispatch_rec else [(t, k) for k in range(a.attempts) for t in names]
    (run / 'queue' / 'items.txt').write_text(''.join(f'{t} {k}\n' for t, k in order))
    atomic(run / 'config.json', cfg)
    if dispatch_rec:   # nothing is claimed until the dispatcher grants slots; then the lane is visible to it
        from aa_evals.common.dispatch import pilane
        pilane.hold(run)
        pilane.register(dispatch_rec, run)
    print(f'{run}: {len(names)} tasks x {a.attempts} attempts, {n_workers} worker(s) x {a.slots} slots, '
          f'max {a.max_in_flight} in flight, code {code}')
    if a.lazy_workers:
        print(f'{run}: lazy workers: the dispatcher submits the first {n_workers} at its first grant')
    elif not a.no_submit:
        submit_workers(run, cfg, n_workers)


if __name__ == '__main__':
    main()

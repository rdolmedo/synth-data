#!/usr/bin/env python3
"""Shared Harbor + mini-swe-agent runner (AA's Terminal-Bench harness): TB 4.0 (aa_evals/tb4) and TB-Science
(aa_evals/tbsci) call it with their own defaults.

  python -m aa_evals.common.harbor_mswea.run --bench tb4 --tasks DIR_OR_TAR --model SPEC --run-id ID \
      [--limit N] [--repeats K] [--step-limit 500] [--sampling JSON] [--include a,b] [--exclude a,b] \
      [--workers N] [--max-concurrent N] [--env-class MOD:CLASS] [--aa-slug SLUG --aa-metric-key KEY]
  python -m aa_evals.common.harbor_mswea.run --bench tb4 --run-id ID --status | --score | --resubmit

RUN = /data/aai/aa_evals/runs/<bench>/<run_id>/: config.json, queue/, trials/<key>.tar.gz, transcripts/,
items.jsonl, summary.json (written by the last worker when every item is resolved; `--score` re-scores).

A new run id: freezes the code (/data/aai/aa_evals/code/<git sha>[-<diff hash>]), writes config.json and the
queue, and submits CPU worker jobs (worker_job.sh). An existing run id resumes: it submits workers if the run is
unfinished and none are alive, after checking that the arguments still match the run's config.

Sampling: --sampling '{"temperature": 1.0, "top_p": 0.95, "max_tokens": N, "reasoning_effort": "high",
"chat_template_kwargs": {...}}'. Default for serving:<name> models: the registry's lab-recommended request
overrides (endpoints.json "request.overrides"); for url: models, AA's default for reasoning models (temperature 0.6).
temperature/top_p/max_tokens go to litellm as is; every other key goes in the request's extra_body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

from . import endpoint, score
from .agent import DEFAULT_VERSION, bundle_path
from .queue import Queue, atomic, now, read
from aa_evals.common import diskcap  # noqa: E402
from aa_evals.common import serving  # noqa: E402

ROOT = Path('/data/aai/aa_evals')
REPO = Path(__file__).resolve().parents[3]
# gen, inference: pi_eval (Pi protocol); tbshared: the shared layer (tbshared/README.md); tbsched: the one
# pending-work helper and scheduler core (pi_eval adaptive/state, #451) -- a frozen run must import what it calls
CODE_DIRS = ('aa_evals', 'harbor_ns', 'tb_agents', 'eval_q35', 'gen', 'inference', 'tbshared', 'tbsched')
# nodes this pipeline excludes on top of the shared list (tbshared condor_cost.bad_nodes(): /data I/O faults, Lustre
# evictions, slow nodes), which every submission reads at submit time (exclude_nodes())
EXCLUDE_NODES = ()


def exclude_nodes(extra=()) -> tuple:
    """The nodes a worker submission excludes now: the shared bad_nodes() (read at each call, #531 A9) + `extra`."""
    from aa_evals.common import condor_cost
    return tuple(dict.fromkeys((*condor_cost.bad_nodes(), *extra)))
LITELLM_KEYS = ('temperature', 'top_p', 'max_tokens')
IMAGE_CACHE = '/data/aai/tbench/harbor_ns'


def list_tasks(src: str) -> dict[str, dict]:
    """task name -> its task.toml, from a directory of task dirs or a tar with tasks/<slug>/task.toml."""
    p = Path(src)
    out = {}
    if p.is_dir():
        for d in sorted(p.iterdir()):
            if (d / 'task.toml').exists():
                out[d.name] = tomllib.loads((d / 'task.toml').read_text())
        return out
    with tarfile.open(p) as tf:
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if len(parts) == 3 and parts[0] == 'tasks' and parts[2] == 'task.toml':
                out[parts[1]] = tomllib.loads(tf.extractfile(m).read().decode())
    return dict(sorted(out.items()))


def mini_config(step_limit: int, sampling: dict) -> dict:
    kw = {'api_key': 'sk-aa-local-dummy', 'drop_params': True}
    extra = {}
    for k, v in sampling.items():
        if k in LITELLM_KEYS:
            kw[k] = v
        elif k == 'extra_body':
            extra.update(v)
        else:
            extra[k] = v
    if extra:
        kw['extra_body'] = extra
    return {'agent': {'step_limit': step_limit, 'cost_limit': 0},
            'model': {'model_kwargs': kw, 'cost_tracking': 'ignore_errors'}}


def snapshot_code(root: Path | None = None) -> str:
    """Frozen copy of the code that jobs run (never the live worktree), under root (default ROOT/code)."""
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    diff = subprocess.run(['git', '-C', str(REPO), 'diff', 'HEAD', '--', *CODE_DIRS], capture_output=True,
                          text=True, check=True).stdout
    untracked = subprocess.run(['git', '-C', str(REPO), 'ls-files', '--others', '--exclude-standard', '--',
                                *CODE_DIRS], capture_output=True, text=True, check=True).stdout.split()
    h = hashlib.sha256(diff.encode())
    for f in sorted(untracked):
        h.update(f.encode() + (REPO / f).read_bytes())
    name = sha[:12] + ('' if not diff and not untracked else '-' + h.hexdigest()[:8])
    dst = (root or ROOT / 'code') / name
    if not (dst / '.complete').exists():
        tmp = dst.with_name(dst.name + f'.tmp{os.getpid()}')
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '--exclude', '__pycache__', '--exclude', '*.pyc',
                        *[str(REPO / d) for d in CODE_DIRS], str(tmp) + '/'], check=True)
        (tmp / '.complete').write_text(json.dumps({'git_sha': sha, 'dirty': bool(diff or untracked), 'at': now()}))
        try:
            tmp.rename(dst)
        except OSError:
            subprocess.run(['rm', '-rf', str(tmp)], check=False)
    return str(dst)


def submit_workers(run: Path, cfg: dict, n: int) -> list[str]:
    jobs = run / 'jobs'
    jobs.mkdir(exist_ok=True)
    w = cfg['worker']
    script = Path(cfg['code']) / 'aa_evals/common/harbor_mswea/worker_job.sh'
    excluded = exclude_nodes(EXCLUDE_NODES)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{x}")' for x in excluded)
    sub = jobs / f'worker-{time.strftime("%Y%m%dT%H%M%S")}.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = {script} {run}
request_cpus = {w['cpus']}
request_memory = {int(w['mem_gb'] * 1024)}
request_disk = {w['disk_gb']}G
+BypassLXCfs = "true"
coresize = -1
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin AA_WORKER_MEM_GB={w['mem_gb']} CONDOR_JOB_ID=$(Cluster).$(Process){serving.condor_env(cfg.get('serving_env') or {})}"
requirements = {req}
output = {jobs}/worker.$(Cluster).$(Process).out
error = {jobs}/worker.$(Cluster).$(Process).err
log = {jobs}/worker.log
queue {n}
''')
    from aa_evals.common import condor_cost
    def record(recs):
        rec = read(run / 'jobs.json', []) or []
        for x in recs:
            rec.append({'cluster': x['cluster'], 'n': x['n'], 'submit_file': x['submit_file'], 'bid': x['bid'],
                        'submitted_at': now()})
        atomic(run / 'jobs.json', rec)
    try:
        recs = condor_cost.submit(sub, w.get('bid') or None, exclude=excluded)   # None: the CPU bid ladder (#378)
    except condor_cost.SubmitError as e:   # record the rungs already queued, so nothing runs untracked
        if e.submitted:
            record(e.submitted)
        raise
    record(recs)
    return [x['cluster'] for x in recs]


def workers_alive(run: Path) -> int:
    clusters = [j['cluster'] for j in read(run / 'jobs.json', []) or []]
    if not clusters:
        return 0
    r = subprocess.run(['condor_q', *clusters, '-af', 'JobStatus'], capture_output=True, text=True)
    return sum(1 for x in r.stdout.split() if x in ('1', '2'))   # idle or running


def reclassify(run: Path) -> dict:
    import shutil
    import tempfile

    from .classify import classify
    from .queue import parse_key
    cfg = read(run / 'config.json')
    q = Queue(run)
    changed = []
    for k, rec in sorted(q.done().items()):
        arc = rec.get('trial_archive')
        if not arc or not Path(arc).exists():
            continue
        tmp = Path(tempfile.mkdtemp(prefix='aa-reclassify-'))
        try:
            with tarfile.open(arc) as tf:
                tf.extractall(tmp, filter='data')
            new = classify(tmp / k, rec.get('returncode'), False, time_limit_sec=rec.get('time_limit_sec'),
                           max_model_len=cfg.get('max_model_len'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        keep = {x: rec[x] for x in ('key', 'task', 'repeat', 'try', 'worker', 'host', 'cores', 'endpoint',
                                    'started_at', 'wall_sec', 'time_limit_sec', 'trial_archive', 'transcript', 'usage',
                                    'finished_at') if x in rec}
        new = {**new, **keep, 'trial_dir': rec.get('trial_dir'), 'reclassified_at': now()}
        if (new.get('score'), new.get('infrastructure_failure')) != (rec.get('score'), rec.get('infrastructure_failure')):
            changed.append(k)
        atomic(q.q / 'done' / f'{k}.json', new)
        task, repeat, attempt = parse_key(k)
        if new.get('infrastructure_failure') and not rec.get('infrastructure_failure'):
            from .queue import MAX_RETRIES, key as qkey   # same retry rule as Queue.finish
            if attempt < MAX_RETRIES:
                atomic(q.q / 'retry' / f'{qkey(task, repeat, attempt + 1)}.json',
                       {'after': k, 'reason': new.get('infrastructure_reason'), 'at': now()})
    s = score.finalize(run)
    return {'changed_score_or_infra': changed, 'summary': s}


def status(run: Path) -> dict:
    q = Queue(run)
    items, res = q.items(), q.resolved()
    done = q.done()
    s = {'run': str(run), 'items': len(items), 'resolved': len(res), 'active': sorted(q.active()),
         'physical_attempts_done': len(done),
         'infra_attempts': sum(1 for r in done.values() if r.get('infrastructure_failure')),
         'passed': sum(1 for r in res.values() if r.get('score') == 1.0),
         'workers_alive': workers_alive(run)}
    if (run / 'summary.json').exists():
        s['summary'] = read(run / 'summary.json')
    return s


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    d = defaults or {}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default=d.get('bench'), required='bench' not in d)
    ap.add_argument('--tasks', default=d.get('tasks'), help='directory of task dirs, or a tar with tasks/<slug>/')
    ap.add_argument('--model', help='serving:<name> or url:<base_url>@<served_name> (none with --agent oracle|nop)')
    ap.add_argument('--agent', default='mini', choices=['mini', 'oracle', 'nop'],
                    help="mini (default); oracle/nop = Harbor's reference-solution / do-nothing agents to validate tasks")
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N tasks (after --include/--exclude)')
    ap.add_argument('--repeats', type=int, default=d.get('repeats', 3))
    ap.add_argument('--step-limit', type=int, default=d.get('step_limit', 0))
    ap.add_argument('--sampling', default=None, help='JSON (see module doc)')
    ap.add_argument('--include', '--tasks-include', dest='include', default='',
                    help='comma-separated task names (only these)')
    ap.add_argument('--exclude', default=','.join(d.get('exclude', [])), help='comma-separated task names')
    ap.add_argument('--mini-version', default=DEFAULT_VERSION)
    ap.add_argument('--api-retries', type=int, default=30, help="AA: up to 30 retries on API failures")
    ap.add_argument('--backstop-multiplier', type=float, default=2.0,
                    help='Harbor wall-clock limit = this x the task agent timeout (env time is the real limit)')
    ap.add_argument('--env-class', default='aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment')
    ap.add_argument('--max-concurrent', type=int, default=32)
    ap.add_argument('--workers', type=int, default=0, help='worker jobs (default: enough for --max-concurrent)')
    ap.add_argument('--worker-cpus', type=int, default=32)
    ap.add_argument('--worker-mem-gb', type=float, default=160)
    ap.add_argument('--worker-disk-gb', type=int, default=400)
    ap.add_argument('--cpu-bid', type=int, default=None, help='fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)')
    ap.add_argument('--aa-slug', default=None, help='AA leaderboard slug of the model (aa_reference)')
    ap.add_argument('--aa-metric-key', default=d.get('aa_metric_key'))
    ap.add_argument('--metric', default=d.get('metric', 'pass@1'))
    ap.add_argument('--n-tasks-benchmark', type=int, default=d.get('n_tasks_benchmark'))
    ap.add_argument('--oracle-run', default=d.get('oracle_run'),
                    help='run dir of an --agent oracle run of these tasks: summary adds score_excl_oracle_failures')
    ap.add_argument('--multiuid', action=argparse.BooleanOptionalAction, default=d.get('multiuid', False),
                    help='harbor_ns multi-uid mode (--ek ns_multiuid=true; needs harbor_ns from PR #47): real subuid '
                         'ids, service entrypoints as the image USER. TB4: fixes live-database-cutover, '
                         'wal-recovery-ordering')
    ap.add_argument('--diskcap', default=None, metavar='auto|FACTOR[:MIN_MB]|off',
                    help='#405 per-sandbox disk cap (harbor_ns/diskcap.py; on by default): no file grows past '
                         'max(storage_mb x FACTOR, MIN_MB) MB, and a box above it loses the process writing the most, '
                         'instead of filling the worker\'s shared scratch. "off": no cap (config "diskcap": null). '
                         'Default "auto": the bench\'s measured per-task caps (defaults "diskcap_family", eval_q35 '
                         'DISKCAP_FAMILIES), 3:30720 for a bench without them')
    ap.add_argument('--harbor-extra-args', default='',
                    help='extra `harbor run` flags, space-separated (diagnostics only, e.g. --verifier-timeout-multiplier 3)')
    ap.add_argument('--no-submit', action='store_true', help='write the run but submit no workers')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--resubmit', type=int, default=0, help='submit N more workers to an existing run')
    ap.add_argument('--reclassify', action='store_true',
                    help='re-derive every finished outcome from its trial archive with the current classify.py, '
                         'then re-score (a changed infra outcome schedules its retry)')
    return ap


def main(argv=None, defaults: dict | None = None) -> None:
    a = build_parser(defaults).parse_args(argv)
    run = ROOT / 'runs' / a.bench / a.run_id
    if a.status:
        print(json.dumps(status(run), indent=1))
        return
    if a.score:
        print(json.dumps(score.finalize(run), indent=1))
        return
    if a.reclassify:
        print(json.dumps(reclassify(run), indent=1))
        return
    if (run / 'config.json').exists():
        cfg = read(run / 'config.json')
        diskcap.check_stored(cfg, a.diskcap, (defaults or {}).get('diskcap_family'), cfg.get('tasks') or ())
        if a.model and a.model != cfg['model']:
            raise SystemExit(f'run {a.run_id} exists with model {cfg["model"]}')
        if Queue(run).complete():
            print(json.dumps(score.finalize(run), indent=1))
            return
        n = a.resubmit or (0 if workers_alive(run) else cfg['worker']['n'])
        if n and not a.no_submit:   # as pi_eval: --no-submit never submits, a new run or a resumed one
            submit_workers(run, cfg, n)
        print(json.dumps(status(run), indent=1))
        return
    if a.agent != 'mini':
        a.model = a.model or 'none'
        ep = {'kind': 'none', 'served': None, 'urls': [], 'request': {}}
    elif not a.model or not a.tasks:
        raise SystemExit('--model and --tasks are required for a new run')
    elif not a.step_limit:
        raise SystemExit('--step-limit is required (TB4 500, TB-Science 1000 per AA)')
    else:
        ok, ep, why = endpoint.alive(a.model)
        if not ok:
            raise SystemExit(f'model {a.model} not usable: {why}')
    bundle = bundle_path(a.mini_version)
    have_bundle = bundle.exists() and bundle.with_name(bundle.name + '.sha256').exists()
    if a.agent == 'mini' and not have_bundle:
        raise SystemExit(f'mini-swe-agent bundle missing: {bundle} (build_bundle.sh)')
    all_tasks = list_tasks(a.tasks)
    excluded = [t for t in all_tasks if t in set(filter(None, a.exclude.split(',')))]
    names = [t for t in all_tasks if t not in excluded]
    if a.include:
        inc = [t for t in a.include.split(',') if t]
        missing = set(inc) - set(all_tasks)
        if missing:
            raise SystemExit(f'unknown tasks: {sorted(missing)}')
        names = [t for t in names if t in inc]
    if a.limit:
        names = names[:a.limit]
    if a.sampling is not None:
        sampling = json.loads(a.sampling)
        sampling_source = 'cli'
    elif ep['kind'] == 'serving' and (ep.get('request') or {}).get('overrides'):
        sampling = ep['request']['overrides']
        sampling_source = f"{ep['registry']} request.overrides (lab-recommended)"
    else:
        sampling, sampling_source = {'temperature': 0.6}, 'AA default for reasoning models'
    n_workers = a.workers or max(1, math.ceil(a.max_concurrent * 3.5 / a.worker_cpus))
    try:
        cap = diskcap.arg(a.diskcap if a.diskcap is not None else (defaults or {}).get('diskcap', diskcap.DEFAULT_ARG),
                          (defaults or {}).get('diskcap_family'), names)
    except ValueError:
        raise SystemExit(f'--diskcap {a.diskcap!r}: expected FACTOR[:MIN_MB] or off') from None
    if 'ns_diskcap' in a.harbor_extra_args:
        raise SystemExit('set the disk cap with --diskcap, not --harbor-extra-args')
    code = snapshot_code()
    sha = (bundle.with_name(bundle.name + '.sha256')).read_text().split()[0] if have_bundle else None
    cfg = {
        'benchmark': a.bench, 'run_id': a.run_id, 'metric': a.metric, 'created_at': now(),
        'git_sha': json.loads((Path(code) / '.complete').read_text())['git_sha'], 'code': code,
        'model': a.model, 'served': ep['served'], 'served_name': ep['served'], 'urls': ep['urls'], 'model_registry': ep.get('registry'),
        'serving_job': ep.get('job_id'), 'agent': a.agent,
        'serving_env': serving.forwarded(),   # the registry root this run reads ({} = production), for its workers
        'max_model_len': endpoint.max_model_len(ep['urls'][0], ep['served']) if ep['urls'] else None,
        'sampling': sampling, 'sampling_source': sampling_source,
        'harness': {'name': 'mini-swe-agent', 'version': a.mini_version, 'config': '-c mini (stock) + overrides',
                    'mode': 'yolo, exit-immediately', 'tools': ['bash'], 'command_timeout_sec': 30,
                    'step_limit': a.step_limit, 'cost_limit': 0, 'max_consecutive_format_errors': 3,
                    'compaction': None, 'harbor': '0.24.0', 'agent_class':
                    'aa_evals.common.harbor_mswea.agent:MiniSweAgentAA', 'bundle': str(bundle), 'bundle_sha256': sha},
        'mini_version': a.mini_version, 'mini_config': mini_config(a.step_limit, sampling),
        'api_retries': a.api_retries, 'backstop_multiplier': a.backstop_multiplier,
        'time_limit': 'task agent timeout counted as environment time (wall - model time); '
                      f'Harbor wall-clock backstop {a.backstop_multiplier:g}x',
        'env_class': a.env_class, 'image_cache': IMAGE_CACHE, 'bundle_root': str(bundle.parents[3]),
        'dataset': {'source': a.tasks, 'n_tasks_source': len(all_tasks)},
        'tasks_src': a.tasks, 'tasks': names, 'excluded_tasks': excluded,
        'subset': bool(a.include or a.limit),
        'n_tasks_benchmark': a.n_tasks_benchmark,
        'categories': {t: (all_tasks[t].get('metadata') or {}).get('category') for t in names},
        'repeats': a.repeats, 'limit': a.limit or None, 'max_concurrent': a.max_concurrent,
        'judge': None, 'multiuid': a.multiuid,
        'diskcap': cap,   # #405; aa_evals/common/diskcap.py (null = off)
        'harbor_extra_args': a.harbor_extra_args.split() + (['--ek', 'ns_multiuid=true'] if a.multiuid else []),
        'aa_slug': a.aa_slug, 'aa_metric_key': a.aa_metric_key, 'oracle_run': a.oracle_run,
        'worker': {'n': n_workers, 'cpus': a.worker_cpus, 'mem_gb': a.worker_mem_gb, 'disk_gb': a.worker_disk_gb,
                   'bid': a.cpu_bid},
    }
    if defaults and defaults.get('dataset'):
        cfg['dataset'].update(defaults['dataset'])
    run.mkdir(parents=True)
    atomic(run / 'config.json', cfg)
    Queue(run).init([(t, r) for t in names for r in range(a.repeats)])
    atomic(run / 'control.json', {'limit': a.max_concurrent})
    print(f'run {run}: {len(names)} tasks x {a.repeats} repeats, model {ep["served"]}, sampling {sampling}')
    if not a.no_submit:
        submit_workers(run, cfg, n_workers)


if __name__ == '__main__':
    main(sys.argv[1:])

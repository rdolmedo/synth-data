"""Re-grade stored generations with each judge arm, one step at a time, recording wall time per step (#41).

  # submit one condor CPU job per judge server (frozen `git archive HEAD aa_evals`; commit first)
  python3 -m aa_evals.judges.sweep submit --study s1 --arms qwen38-27b-nothink,qwen38-27b-medium,qwen38-27b-xhigh \
      --omni-src super-v11-300 --hle-src <hle run id> [--gdp-src <gdp run id> --gdp-args "<selection args>"]

Steps per arm, in order: a probe (one tiny request, to verify that the arm's thinking switch works), the Omniscience
re-grade, its retest (graded again, for test-retest agreement), HLE, and GDP.pdf. Steps run one after another so
that each wall time belongs to one benchmark on a server we use alone. Everything is resumable:
- a step already recorded as finished is skipped;
- each benchmark's own run resumes its unfinished items;
- adding an arm later just runs that arm.
Bench runs land in each benchmark's runs root as j41-<study>-<arm>[-rt]. This study's records are in
/data/aai/aa_evals/runs/judges/<study>/steps.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import arms as arms_mod
from aa_evals.common import jsonl

RUNS = Path(os.environ.get('AA_EVALS_RUNS', '/data/aai/aa_evals/runs'))
STUDY_ROOT = RUNS / 'judges'
CODE_ROOT = Path('/data/aai/aa_evals/code/judges')
REPO = Path(__file__).resolve().parents[2]
PY_DEFAULT = '/home/user/python-3.11/bin/python3'    # stdlib + httpx + pyarrow (HLE's interpreter)
PY_GDP = '/data/aai/aa_evals/envs/gdppdf/bin/python'
HLE_FALLBACK = Path('/data/aai/aa_evals/runs_fallback/hle')   # aa_evals/hle/run.py FALLBACK_RUNS_ROOT
BENCHES = ('omni', 'omni-rt', 'hle', 'gdp')


def run_id(study: str, arm: str, step: str) -> str:
    return f'j41-{study}-{arm}' + ('-rt' if step == 'omni-rt' else '')


def bench_dir(step: str) -> Path:
    return RUNS / {'omni': 'omniscience', 'omni-rt': 'omniscience', 'hle': 'hle', 'gdp': 'gdppdf'}[step]


def read_steps(study: str) -> list[dict]:
    p = STUDY_ROOT / study / 'steps.jsonl'
    return jsonl.read(p)


def append_step(study: str, rec: dict) -> None:
    p = STUDY_ROOT / study / 'steps.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(rec) + '\n')


def probe(arm: arms_mod.Arm, server: str) -> dict:
    """One tiny request with the arm's settings: did the model think? (verifies the thinking switch)"""
    from aa_evals.omniscience import client
    ep = client.resolve(f'serving:{server}')
    body = json.loads(json.dumps((ep.request or {}).get('overrides') or {}))
    body.update(arm.extra)
    body.update(max_tokens=4096, messages=[{'role': 'user', 'content': 'What is 17 * 23? Reply with just the number.'}])
    t0 = time.time()
    resp = client.chat(ep, body, timeout=600, attempts=3)
    out = client.split_response(resp)
    return {'answer': out['answer'][:40], 'reasoning_chars': len(out['reasoning']),
            'completion_tokens': out['usage'].get('completion_tokens'), 'latency_s': round(time.time() - t0, 2),
            'served_name': ep.model, 'urls': ep.urls}


def hle_src_dir(src: str) -> Path:
    """An HLE source run: an absolute dir, or a run id in the /data runs root or the over-quota fallback root."""
    if Path(src).is_absolute():
        return Path(src)
    for root in (RUNS / 'hle', HLE_FALLBACK):
        if (root / src / 'config.json').exists():
            return root / src
    raise SystemExit(f'HLE source run {src} not found')


def step_cmd(step: str, arm: arms_mod.Arm, server: str, a) -> list[str] | None:
    rid = run_id(a.study, arm.name, step)
    # the arm by name (recorded as judge.arm in every run config); a --server override sends its settings elsewhere
    jargs = ['--judge-arm', arm.name] if server == arm.server else \
        ['--judge', f'serving:{server}', '--judge-extra-body', json.dumps(arm.extra)]
    if step in ('omni', 'omni-rt'):
        if not a.omni_src:
            return None
        return [a.py, '-u', '-m', 'aa_evals.omniscience.run', '--answers-from', a.omni_src, *jargs,
                '--run-id', rid, '--concurrency', str(a.concurrency)]
    if step == 'hle':
        if not a.hle_src:
            return None
        src = json.loads((hle_src_dir(a.hle_src) / 'config.json').read_text())
        sel = []  # reproduce the source run's selection
        if src['dataset'].get('ids_file'):
            sel = ['--ids', src['dataset']['ids_file']]
        elif src['dataset'].get('limit'):
            sel = ['--limit', str(src['dataset']['limit'])]
            if str(src['dataset'].get('selection', '')).startswith('stratified'):
                sel.append('--stratify')
        if src.get('n_repeats', 1) != 1:
            sel += ['--repeats', str(src['n_repeats'])]
        if not src['model']['spec'].startswith('serving:'):
            # a url: model has no registry to rebuild the sampling from, so pass the source's own: HLE refuses to
            # judge generations made with other settings
            sel += ['--extra-body', json.dumps(src['generation'])]
        return [a.py, '-u', '-m', 'aa_evals.hle.run', '--model', src['model']['spec'], '--dataset',
                src['dataset']['id'], *sel, '--generations-from', a.hle_src, *jargs,
                '--judge-concurrency', str(a.concurrency), '--run-id', rid] \
            + shlex.split(a.hle_args or '')
    if step == 'gdp':
        if not a.gdp_src:
            return None
        # GDP.pdf's own re-grade path (PR #43): copies the source generations, grades every criterion.
        return [a.py_gdp, '-u', '-m', 'aa_evals.gdppdf.run', '--run-id', rid, '--generations-from', a.gdp_src,
                *jargs, '--judge-concurrency', str(a.gdp_concurrency)] \
            + shlex.split(a.gdp_args or '')
    raise ValueError(step)


def run(a) -> int:
    done = {(r['arm'], r['step']) for r in read_steps(a.study) if r.get('rc') == 0 and r.get('kind') == 'step'}
    rc_all = 0
    for name in a.arms.split(','):
        arm = arms_mod.ARMS[name]
        server = a.server or arm.server
        try:
            pr = probe(arm, server)
            append_step(a.study, {'kind': 'probe', 'arm': name, 'server': server, 'at': time.time(), **pr})
            print(f'[{name}] probe: {pr}', flush=True)
        except Exception as e:  # the server is down: skip this arm, keep going
            append_step(a.study, {'kind': 'probe', 'arm': name, 'server': server, 'at': time.time(), 'error': str(e)})
            print(f'[{name}] probe failed, skipping the arm: {e}', flush=True)
            rc_all = 1
            continue
        for step in [s for s in BENCHES if s in a.steps.split(',')]:
            if (name, step) in done and not a.rerun:
                print(f'[{name}] {step}: done earlier', flush=True)
                continue
            cmd = step_cmd(step, arm, server, a)
            if cmd is None:
                continue
            print(f'[{name}] {step}: {shlex.join(cmd)}', flush=True)
            t0 = time.time()
            code = Path(a.gdp_code) if step == 'gdp' and a.gdp_code else REPO
            env = dict(os.environ, PYTHONPATH=str(code), PYTHONNOUSERSITE='1')
            rc = subprocess.run(cmd, cwd=code, env=env).returncode
            rec = {'kind': 'step', 'arm': name, 'step': step, 'server': server, 'run_id': run_id(a.study, name, step),
                   'bench_dir': str(bench_dir(step)), 'rc': rc, 'started': t0, 'wall_s': round(time.time() - t0, 1),
                   'concurrency': a.concurrency, 'gpus': arm.gpus, 'hardware': arm.hardware,
                   'shared_server': arm.shared_server,
                   'src': {'omni': a.omni_src, 'omni-rt': a.omni_src, 'hle': a.hle_src, 'gdp': a.gdp_src}[step]}
            append_step(a.study, rec)
            print(f'[{name}] {step}: rc={rc} wall={rec["wall_s"]}s', flush=True)
            if rc not in (0, 2):  # 2 = finished but over the infra-error limit
                rc_all = 1
    return rc_all


# the graders it runs (hle/omniscience/gdppdf run.py) import tb_agents' response budget for every item (#449; #451 test)
FROZEN = ('aa_evals', 'tb_agents')


def frozen_copy(ref: str = 'HEAD') -> Path:
    if ref == 'HEAD' and subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', *FROZEN],
                                        capture_output=True, text=True).stdout.strip():
        raise SystemExit(f'{FROZEN} has uncommitted changes: commit first (the job runs a frozen copy of HEAD)')
    sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', ref], text=True).strip()
    dst = CODE_ROOT / f'{sha}-tb'   # -tb: copies from before FROZEN held tb_agents are not reused (#451)
    if not dst.exists():
        tmp = Path(tempfile.mkdtemp(dir=CODE_ROOT.parent if CODE_ROOT.parent.exists() else None))
        archive = subprocess.run(['git', '-C', str(REPO), 'archive', sha, *FROZEN], capture_output=True,
                                 check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=archive, check=True)
        (tmp / 'FROZEN_SHA').write_text(sha + '\n')
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        tmp.rename(dst)
    return dst


def submit(a, raw: list[str]) -> None:
    code = frozen_copy()
    if a.gdp_src and not a.gdp_code:  # GDP.pdf's grader from its own branch until it is merged (PR #43)
        subprocess.run(['git', '-C', str(REPO), 'fetch', '-q', 'origin'], check=True)
        raw = raw + ['--gdp-code', str(frozen_copy(a.gdp_ref))]
    study = STUDY_ROOT / a.study
    logs = study / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    tag = a.arms.split(',')[0].split('-')[0]
    passthrough, skip = [], False
    for x in raw[1:]:  # drop the mode and the submit-only flags
        if skip:
            skip = False
        elif x in ('--dry-run',):
            continue
        elif x == '--bid':
            skip = True
        elif not x.startswith('--bid='):
            passthrough.append(x)
    args = ' '.join(shlex.quote(x) for x in ['-u', '-m', 'aa_evals.judges.sweep', 'run'] + passthrough)
    noproxy = 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'
    from aa_evals.common.condor_cost import bad_nodes   # the shared list, read now (tbshared)
    from aa_evals.common import serving   # the job resolves serving:<server>: forward the registry root
    BAD_NODES_REQ = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    sub = f"""universe = vanilla
executable = {a.py}
arguments = "{args.replace('"', '""')}"
initialdir = {code}
output = {logs}/sweep-{tag}.$(Cluster).out
error = {logs}/sweep-{tag}.$(Cluster).err
log = {logs}/condor.log
request_cpus = 4
request_memory = 8192
request_disk = 4G
+JobBatchName = "aa-judges-{a.study}-{tag}"
+MaxRuntime = 172800
getenv = false
environment = "PYTHONPATH={code} PYTHONNOUSERSITE=1 HOME={os.environ.get('HOME', '/home/user')} PATH=/usr/local/bin:/usr/bin:/bin http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy={noproxy} NO_PROXY={noproxy}{serving.condor_env()}"
requirements = {BAD_NODES_REQ}
queue
"""
    path = logs / f'sweep-{tag}-{int(time.time())}.sub'
    path.write_text(sub)
    if a.dry_run:
        print(sub)
        return
    from aa_evals.common import condor_cost
    condor_cost.submit(path, a.bid)   # default: the CPU bid ladder (#378)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'submit'])
    ap.add_argument('--study', required=True)
    ap.add_argument('--arms', required=True, help=f'comma-separated, from {sorted(arms_mod.ARMS)}')
    ap.add_argument('--server', help='override the arms\' registry name (e.g. a reference arm\'s shared server)')
    ap.add_argument('--steps', default=','.join(BENCHES))
    ap.add_argument('--omni-src', help='Omniscience run id whose answers are re-graded')
    ap.add_argument('--hle-src', help='HLE run id whose generations are re-judged')
    ap.add_argument('--hle-args', help='extra HLE run args reproducing the source selection (e.g. "--limit 40")')
    ap.add_argument('--gdp-src', help='GDP.pdf run id whose attempts are re-graded')
    ap.add_argument('--gdp-args', help='GDP.pdf run args reproducing the source run (model, preset, selection)')
    ap.add_argument('--gdp-ref', default='origin/aa-evals', help='git ref whose aa_evals/gdppdf grades GDP')
    ap.add_argument('--gdp-code', help='a frozen code tree for the GDP step (set by submit from --gdp-ref)')
    ap.add_argument('--gdp-concurrency', type=int, default=32,
                    help='GDP fans out ~13 criteria per attempt, so 32 keeps ~50 requests in flight')
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--rerun', action='store_true', help='redo steps already recorded as finished')
    ap.add_argument('--py', default=PY_DEFAULT)
    ap.add_argument('--py-gdp', default=PY_GDP)
    ap.add_argument('--bid', type=int, default=None, help='fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)')
    ap.add_argument('--dry-run', action='store_true')
    raw = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(raw)
    unknown = [x for x in a.arms.split(',') if x not in arms_mod.ARMS]
    if unknown:
        ap.error(f'unknown arms {unknown}')
    if a.mode == 'submit':
        submit(a, raw)
        return 0
    return run(a)


if __name__ == '__main__':
    sys.exit(main())

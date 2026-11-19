"""Helpers for benchmarks whose agent runs under this Pi harness but whose grading stays in the benchmark (#348).

GDPval, CritPt and AutomationBench build Harbor task bundles (their `pi/tasks.py`), generate with `pi_eval.run`
exactly as TB 2.1 does, then import each resolved attempt into the benchmark's own run layout (their `pi/run.py`):
what the agent left in the sandbox (Harbor artifacts), the Pi session, and the attempt's end and accounting.
"""
from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

from aa_evals.common import jsonl
from aa_evals.common.harbor_mswea.queue import read

ENVIRONMENT_CLASS = 'aa_evals.common.pi_eval.task_inputs_env:NsTaskInputsEnvironment'
STATS = ('end_state', 'model_requests', 'wall_sec', 'model_sec', 'env_sec', 'context_limit', 'output_cap_hits',
         'max_prompt_tokens', 'max_context_tokens', 'n_input_tokens', 'n_output_tokens')


def base_args(bench: str, run_id: str, model: str, bundle: Path, tasks: list[str], repeats: int, seed_namespace: str,
              effort: str | None = None) -> list[str]:
    """pi_eval.run arguments shared by the ported benchmarks: the TB 2.1 protocol (Pi 1.2.0 pi-cont16k, 131,072
    window, no compaction, 16,384 output tokens per request, the model's registry request settings, 2 h environment
    clock, offline agent), single-uid sandboxes, one shared image with staged task inputs, the 16 GB sandbox memory
    cap the Stirrup paths had per command. Text-only model input (#348): `--model-images off`, so pi_eval refuses
    a launch that serves images (only a text-only serving launch, e.g. aa-serving's `*-text` specs, passes)."""
    if not tasks:
        raise SystemExit('empty task selection: refusing to launch (pi_eval would run the whole bundle)')
    args = ['--bench', bench, '--run-id', run_id, '--model', model, '--bundle', str(bundle),
            '--include', ','.join(tasks), '--attempts', str(repeats), '--seed-namespace', seed_namespace,
            '--context-window', '131072', '--agent-cap-sec', '7200', '--no-multiuid', '--env-class', ENVIRONMENT_CLASS,
            '--harbor-env', 'HARBOR_NS_MEMCAP=1', '--model-images', 'off']
    return args + (['--reasoning-effort', effort] if effort else [])


HARBOR_PYTHON = os.environ.get('HARBOR_PYTHON', '/opt/aai/venvs/harbor/bin/python')
REPO = Path(__file__).resolve().parents[3]


def job_missing(python: str, modules, pythonpath=None) -> list[str]:
    """The `modules` that `python` cannot find when run the way jobs run it: `python -s`, PYTHONNOUSERSITE=1,
    PYTHONPATH = pythonpath only. M5 11-15: HLE's jinja2 lived only in ~/.local; a plain `python3 -c 'import jinja2'`
    passed while the job's interpreter could not import it (#451). A check in this process cannot tell (an editable
    or .pth install in the user site points elsewhere), so it runs the interpreter."""
    import subprocess
    import sys
    import tempfile
    code = ('import importlib.util as u, sys\n'
            'def gone(m):\n'
            '    try:\n'
            '        return u.find_spec(m) is None\n'
            '    except ImportError:   # a dotted name whose parent package is missing\n'
            '        return True\n'
            'print(" ".join(m for m in sys.argv[1:] if gone(m)))')
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': os.environ.get('HOME', ''), 'PYTHONNOUSERSITE': '1',
           **({'PYTHONPATH': str(pythonpath)} if pythonpath else {})}
    with tempfile.TemporaryDirectory() as empty:   # `python -c` puts its cwd first on sys.path: never the checkout
        try:
            r = subprocess.run([python, '-s', '-c', code, *modules], env=env, capture_output=True, text=True,
                               timeout=300, cwd=str(pythonpath) if pythonpath else empty)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f'job_missing: {python} could not run: {e!r}', file=sys.stderr, flush=True)
            return list(modules)
    if r.returncode:
        print(f'job_missing: {python} failed: {r.stderr.strip()[-500:]}', file=sys.stderr, flush=True)
        return list(modules)
    return r.stdout.split()


def ensure_python(modules: tuple[str, ...], python: str, module: str) -> None:
    """Re-execute `python -m module argv` when this interpreter, run as jobs run it (job_missing), lacks `modules`
    (e.g. a launcher that runs every port with the harbor venv, which has no pandas/pyarrow)."""
    import sys
    if not job_missing(sys.executable, modules, REPO):
        return
    if os.environ.get('_AA_PI_PORT_REEXEC'):
        raise SystemExit(f'{module}: {python} lacks one of {modules}')
    env = {**os.environ, '_AA_PI_PORT_REEXEC': '1', 'PYTHONNOUSERSITE': '1',
           'PYTHONPATH': str(REPO) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')}
    os.execve(python, [python, '-m', module, *sys.argv[1:]], env)


def runs_root() -> Path:
    """pi_eval.run.RUNS_ROOT, without importing harbor (benchmark venvs lack it)."""
    return Path(os.environ.get('AA_RUNS_ROOT', '/data/aai/aa_evals/runs_fallback'))


def launch(args: list[str]) -> None:
    """`pi_eval.run` with these arguments in the harbor venv (it imports harbor; the benchmarks' venvs don't have
    it, and the harbor venv lacks their data libraries)."""
    import subprocess
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1'}
    r = subprocess.run([HARBOR_PYTHON, '-m', 'aa_evals.common.pi_eval.run', *args], env=env, cwd=str(REPO))
    if r.returncode:
        raise SystemExit(f'pi_eval.run failed ({r.returncode})')


def status(pi_run: Path) -> dict:
    """pi_eval.run.status without its harbor imports: queue counts, resolved attempts by status and end."""
    q = pi_run / 'queue'
    names = lambda d, suf='': {p.name.removesuffix(suf) for p in (q / d).glob('*')} if (q / d).exists() else set()
    done, claims = names('done', '.json'), names('claims')
    rows = resolved(pi_run)
    ends: dict[str, int] = {}
    for r in rows:
        e = end_of(r) if r['status'] == 'ok' else 'infra'
        ends[e] = ends.get(e, 0) + 1
    return {'run': str(pi_run), 'items': len((q / 'items.txt').read_text().split('\n')) - 1,
            'physical_done': len(done), 'active': len(claims - done), 'resolved': len(rows), 'ends': ends,
            'admission': read(q / 'admission.json')}


def resolved(pi_run: Path) -> list[dict]:
    """pi_eval's per-attempt rows (score.rows) that are resolved: status ok or infra_error (retries spent)."""
    from aa_evals.common.pi_eval import score
    return [r for r in score.rows(pi_run) if r['status'] in ('ok', 'infra_error')]


def end_of(row: dict) -> str:
    """How the episode ended: `time_limit` (the environment-clock cap), `context` (the window was full; no
    compaction), else `finish` (the agent stopped by itself, or an agent error after it started)."""
    if row.get('end_state') == 'env_budget_spent' or row.get('exception_type') == 'AgentTimeoutError':
        return 'time_limit'
    if row.get('context_limit'):
        return 'context'
    return 'finish'


def stats(row: dict) -> dict:
    return {k: row.get(k) for k in STATS}


def attempt_record(pi_run: Path, row: dict) -> dict:
    return read(pi_run / 'attempts' / row['key'] / 'outcome.json', {}) or {}


VISIBILITY_SEC = 3600   # how long an attempt's archive may lag its record (Lustre) before it counts as lost


def _younger(path: Path, sec: float = VISIBILITY_SEC) -> bool:
    import time
    try:
        return time.time() - path.stat().st_mtime < sec
    except OSError:
        return True


def extract(pi_run: Path, row: dict, dest: Path, artifacts: list[str]) -> dict:
    """Extract from one attempt's trial archive into `dest`:
    - each artifact path (as in task.toml, e.g. /home/user/outputs) to dest/artifacts/<path>, regular files and dirs
      only (tar 'data' filter; links the agent left are not followed or copied: listed in `links`);
    - Harbor's artifact manifest (`artifacts/manifest.json`): `artifact_status` {source: ok|empty|failed|...};
    - the Pi session (agent/pi/sessions/*.jsonl) as a list of entries;
    - the task-input receipt (task_inputs.json) and the verifier's files.
    `found` is False when the archive, its trial directory or the artifact manifest is missing: the caller must not
    read that as "the agent left nothing" (a lost archive is not a model outcome). `permanent` then says whether
    waiting can help: only an archive that is not visible yet (attempt record or archive younger than
    VISIBILITY_SEC) is retried; a pack error, a record without a trial, an archive gone for longer, a trial
    directory missing from the archive or a missing artifact manifest will not change, and the caller records an
    infrastructure failure."""
    rec = attempt_record(pi_run, row)
    trial = rec.get('trial_dir_in_archive')
    archive = Path(rec.get('trial_archive') or row.get('transcript') or '/nonexistent')
    out = {'found': False, 'permanent': True, 'why': None, 'session': None, 'inputs_receipt': None, 'verifier': {},
           'trial': trial, 'archive': str(archive), 'wall_sec': rec.get('wall_sec'), 'artifact_status': {},
           'links': []}
    rec_path = pi_run / 'attempts' / row['key'] / 'outcome.json'
    if not rec:
        out['permanent'] = not _younger(rec_path) and rec_path.exists()
        out['why'] = 'attempt record not readable' + ('' if out['permanent'] else ' yet')
        return out
    if rec.get('pack_error'):
        out['why'] = f'trial was not archived (pack_error {rec["pack_error"]})'
        return out
    if not trial:
        out['why'] = 'no trial_dir_in_archive in the attempt record'
        return out
    if not archive.is_file():
        out['permanent'] = not _younger(rec_path)
        out['why'] = f'trial archive missing: {archive}'
        return out
    prefixes = [f'{trial}/artifacts/{a.strip("/")}' for a in artifacts]
    manifest_name = f'{trial}/artifacts/manifest.json'
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or None) as tmp, \
            tarfile.open(archive) as tf:
        members, session, has_trial = [], None, False
        for m in tf.getmembers():
            n = m.name
            has_trial = has_trial or n == trial or n.startswith(trial + '/')
            if any(n == p or n.startswith(p + '/') for p in prefixes):
                if m.isfile() or m.isdir():
                    members.append(m)
                else:
                    out['links'].append(n[len(trial) + len('/artifacts'):])
            elif n.startswith(f'{trial}/agent/pi/sessions/') and n.endswith('.jsonl') and m.isfile():
                members.append(m)
                session = n
            elif (n in (f'{trial}/task_inputs.json', manifest_name) or n.startswith(f'{trial}/verifier/')) \
                    and m.isfile():
                members.append(m)
        if not has_trial:
            out['why'] = f'trial directory {trial} not in {archive}'
            return out
        tf.extractall(tmp, members=members, filter='data')
        mf = Path(tmp) / manifest_name
        if not mf.is_file():
            out['why'] = 'no artifacts/manifest.json in the trial (artifact collection did not run)'
            return out
        out['artifact_status'] = {e.get('source'): e.get('status') for e in json.loads(mf.read_text())
                                  if isinstance(e, dict)}
        out['found'] = True
        art = Path(tmp) / trial / 'artifacts'
        import shutil
        target = dest / 'artifacts'
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for p in prefixes:
            src = Path(tmp) / p
            if src.exists():
                rel = src.relative_to(art)
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if src.is_dir() else shutil.copy2)(src, target / rel)
        if session:
            # '\n'-separated: Pi leaves U+2028/U+0085 raw. An agent can overwrite lines of its own session file from
            # inside the sandbox (marin-lr1 hr.handbook_acknowledgment replaced the header with its own text): such
            # lines are recorded in session_bad_lines, not fatal, since the score comes from the extracted state.
            bad = []
            out['session'] = jsonl.read(Path(tmp) / session, bad=bad)
            if bad:
                out['session_bad_lines'] = [{'line': i, 'reason': r} for _, i, r in bad]
        p = Path(tmp) / trial / 'task_inputs.json'
        out['inputs_receipt'] = json.loads(p.read_text()) if p.exists() else None
        v = Path(tmp) / trial / 'verifier'
        out['verifier'] = {f.name: f.read_text(errors='replace')[:200000] for f in v.iterdir() if f.is_file()} \
            if v.exists() else {}
    return out


def verifier_reward(got: dict) -> float | None:
    """The placeholder verifier's reward (tests/test.sh: did the agent leave a submission), None if absent."""
    try:
        return float(got['verifier']['reward.txt'].strip())
    except (KeyError, ValueError, AttributeError):
        return None


def final_text(session: list | None) -> str:
    """The last assistant text of a Pi session (the agent's closing message), '' if none."""
    for e in reversed(session or []):
        msg = e.get('message') if isinstance(e, dict) else None
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue
        parts = msg.get('content')
        if isinstance(parts, str):
            return parts
        text = '\n'.join(p.get('text', '') for p in parts or [] if isinstance(p, dict) and p.get('type') == 'text')
        if text.strip():
            return text
    return ''

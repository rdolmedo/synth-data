"""CritPt v2 (AA Intelligence Index v5.0) on our cluster.

  # run (as a CPU condor job; the sandbox needs a compute node)
  python -m aa_evals.critpt.run --model serving:aa-super --run-id smoke-super --limit 5 --repeats 1 --example --condor
  # grade a finished run: build the 70-problem batch of each repeat, check it, and (with --submit) send it to AA
  python -m aa_evals.critpt.run --run-id smoke-super --grade [--submit]

Outputs in /data/aai/aa_evals/runs/critpt/<run_id>/: config.json, items.jsonl (one row per attempt; the last row of
an item x repeat is its final state), transcripts.jsonl (one packed line per attempt), batches/repeat_<k>.json,
grading.jsonl (AA responses), summary.json. Rerunning the same command resumes: finished item x repeats are kept and
infra errors are retried (up to --max-attempts).

Scoring: the server returns only the accuracy over the whole batch, so per-item scores are unknown (`score` is null
for `ok` rows; 0 for `model_error`, which is submitted blank). Problems not attempted (--limit) and items still in
infra error are submitted blank and excluded by rescaling: accuracy x 70 / n_scored.
"""
from __future__ import annotations

import argparse
import collections
import asyncio
import json
import logging
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from ..common import jsonl
from ..common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)
from . import consensus, data, grading, harness, prompts

# Run outputs (and the frozen code snapshots under _code/). CRITPT_RUNS overrides it (keep it on /data; never /home).
RUNS = Path(os.environ.get('CRITPT_RUNS', '/data/aai/aa_evals/runs/critpt'))
SERVING = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
SIF = data.DATA_ROOT / 'sif' / 'critpt_py313_20261102.sif'
SIF_SHA256 = (data.DATA_ROOT / 'sif' / 'critpt_py313_20261102.sha256')
VENV_PY = Path('/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python')
AA_REF = data.DATA_ROOT / 'aa_reference_2026-11-02.json'
LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
WORKTREE = Path(__file__).resolve().parents[2]

# AA's row(s) per served model, for the reference numbers only. Generation settings come from the serving registry's
# request profile (endpoints.json "request": the teacher spec's [request] block), never from here (M3.2, #175).
AA_ROWS = {
    'nemotron-3.5-super-vl': {'aa_slug': 'nemotron-3-5-super-vl', 'aa_name': 'Nemotron 3.5 Super VL'},
    # AA's (Max) variant: the only DeepSeek V4.1 Flash 1017 row with a published full-70 score (50.0; hard-40 24.0).
    'deepseek-v41-flash-1017': {'aa_slug': 'deepseek-v4-1-flash-1017', 'aa_name': 'DeepSeek V4.1 Flash 1017 (Max)'},
    # AA lists one row per reasoning effort; a run is compared with the row of the effort it used.
    'qwen38-27b': {'aa_slug': 'qwen3-8-27b', 'aa_name': 'Qwen3.8 27B (Xhigh)',
                   'aa_slug_by_effort': {'xhigh': 'qwen3-8-27b', 'medium': 'qwen3-8-27b-medium',
                                         'low': 'qwen3-8-27b-low'}},
}
# Keys the OpenAI client takes as arguments; every other request override goes into extra_body (vLLM's extensions).
OPENAI_ARGS = {'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'seed', 'stop', 'reasoning_effort',
               'max_completion_tokens', 'logprobs', 'top_logprobs', 'n'}
CONTEXT_CAP = 131072      # M3.0 protocol (#173): regular evals cap the context at 128k
ADAPTERS = ('inline_reasoning_history', 'mirror_reasoning_content')


def build_request(profile: dict, a) -> tuple[dict, list[str]]:
    """The per-request settings: the endpoint's request profile (overrides), then CLI flags. --reasoning-effort sets the
    effort both top level and in chat_template_kwargs. History adapters from the profile; models without one get
    mirror_reasoning_content (vLLM >= 0.31 reads earlier turns' thinking from `reasoning`, not `reasoning_content`)."""
    over = {} if a.no_profile else json.loads(json.dumps(profile.get('overrides') or {}))
    if a.temperature is not None:
        over['temperature'] = a.temperature
    if a.top_p is not None:
        over['top_p'] = a.top_p
    ctk = dict(over.pop('chat_template_kwargs', None) or {})
    if a.chat_template_kwargs:
        ctk.update(json.loads(a.chat_template_kwargs))
    if a.reasoning_effort:
        over['reasoning_effort'] = a.reasoning_effort
        ctk['reasoning_effort'] = a.reasoning_effort
    request = {k: v for k, v in over.items() if k in OPENAI_ARGS}
    extra = {k: v for k, v in over.items() if k not in OPENAI_ARGS}
    if ctk:
        extra['chat_template_kwargs'] = ctk
    if extra:
        request['extra_body'] = extra
    if a.adapters is not None:
        adapters = [x for x in a.adapters.split(',') if x]
    else:
        adapters = [] if a.no_profile else [x for x in profile.get('adapters') or [] if x in ADAPTERS]
        if not adapters:
            adapters = ['mirror_reasoning_content']
    bad = [x for x in adapters if x not in ADAPTERS]
    if bad:
        raise SystemExit(f'unknown history adapters {bad}; known: {ADAPTERS}')
    return request, adapters


def request_effort(request: dict) -> str | None:
    ctk = (request.get('extra_body') or {}).get('chat_template_kwargs') or {}
    if ctk.get('enable_thinking') is False or ctk.get('thinking') is False:
        return 'none'
    return request.get('reasoning_effort') or ctk.get('reasoning_effort')


LOG = logging.getLogger('critpt')


# ----------------------------------------------------------------------------------------------- model spec

def resolve_model(spec: str) -> dict:
    """serving:<name> (registry written by task_synthesis.serve.serve_teacher) or url:<base_url>@<served_name>."""
    if spec.startswith('serving:'):
        reg = serving.path(spec.split(':', 1)[1], SERVING)
        info = serving.record(spec.split(':', 1)[1], SERVING)
        age = serving.age(info)
        if age > 120:
            raise RuntimeError(f'{reg} is {age:.0f}s old: the server is dead')
        return {'spec': spec, 'served_name': info['served_name'], 'urls': list(info['urls']),
                'teacher': info.get('teacher'), 'job_id': info.get('job_id'), 'request_profile': info.get('request') or {}}
    if spec.startswith('url:'):
        base, served = spec[4:].rsplit('@', 1)
        return {'spec': spec, 'served_name': served, 'urls': [base.rstrip('/')], 'request_profile': {}}
    raise SystemExit(f'bad --model {spec!r}: use serving:<name> or url:<base_url>@<served_name>')


def max_model_len(url: str, served: str) -> int:
    with urllib.request.urlopen(f'{url}/models', timeout=30) as r:
        models = json.load(r)['data']
    for m in models:
        if m['id'] == served and m.get('max_model_len'):
            return int(m['max_model_len'])
    raise RuntimeError(f'{url}/models does not report max_model_len for {served}')


# ----------------------------------------------------------------------------------------------- files

BAD_LINES = jsonl.Tolerant()   # undecodable lines seen in this run's files: counted in summary.json, which is then invalid


def read_jsonl(path: Path) -> list[dict]:
    """'\n'-separated (rows are written with ensure_ascii=False, so U+2028/U+0085 appear raw); an unterminated last
    line is a row still being appended; any other bad line is recorded in BAD_LINES."""
    return BAD_LINES.read(path)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(path)


_LEGACY_SUMMARY_FAIL = 'Summarizer response contained no text blocks'


def final_states(items: list[dict]) -> dict[tuple[str, int], dict]:
    """Last row per (item_id, repeat), with the number of attempts.

    Rows written before summary failures were model errors (code before ce2d719) recorded them as infra_error and
    reran the item. Under the current rule that first failure is final: it is read as model_error (summary_failed)
    and later attempts of that item x repeat are ignored."""
    out: dict[tuple[str, int], dict] = {}
    locked: dict[tuple[str, int], int] = {}
    for r in items:
        key = (r['item_id'], r['repeat'])
        if key in locked and r['attempt'] > locked[key]:
            continue
        if r['status'] == 'infra_error' and _LEGACY_SUMMARY_FAIL in (r.get('error') or ''):
            r = {**r, 'status': 'model_error', 'score': 0,
                 'error': 'summary_failed: no text in 3 summarizer responses (reclassified from a legacy infra_error)'}
            locked[key] = r['attempt']
        out[key] = {**r, 'attempts': r['attempt']}
    return out


def git_sha() -> str:
    # Condor jobs run a frozen snapshot (no .git of its own) whose SHA submit_condor passes in. Check it first: the
    # snapshot can sit inside another repository's tree, whose HEAD is not this code.
    if os.environ.get('CRITPT_CODE_SHA'):
        return os.environ['CRITPT_CODE_SHA']
    try:
        return subprocess.run(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - frozen snapshots have no .git; the SHA is in their directory name
        return os.environ.get('CRITPT_CODE_SHA', 'unknown')


# ----------------------------------------------------------------------------------------------- run

def select_problems(a) -> list[data.Problem]:
    probs = data.load_problems()
    if a.problems:
        want = a.problems.split(',')
        probs = [p for p in probs if p.problem_id in want]
        if len(probs) != len(want):
            raise SystemExit(f'unknown problem ids in --problems: {sorted(set(want) - {p.problem_id for p in probs})}')
    elif a.limit:
        probs = random.Random(20261102).sample(probs, a.limit)   # a fixed, spread-out subset
        probs.sort(key=lambda p: int(p.problem_id.split('_')[1]))
    return probs


def sandbox_selftest(isolation: str) -> None:
    """Offline, packages present, and /tmp persistent across two commands of one trajectory (per-trajectory bind)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='critpt_selftest_') as d, \
            tempfile.TemporaryDirectory(prefix='critpt_selftest_tmp_') as t:
        probe = ('python -c "import socket,numpy,scipy,sympy,matplotlib\n'
                 'try:\n socket.create_connection((\'10.22.0.107\',8080),timeout=3); print(\'NET\')\n'
                 'except OSError: print(\'OFFLINE\')"; echo persist > /tmp/selftest_tmp')
        r = subprocess.run(harness.sandbox_argv(str(SIF), isolation, d, probe, t), capture_output=True, text=True,
                           timeout=300)
        r2 = subprocess.run(harness.sandbox_argv(str(SIF), isolation, d, 'cat /tmp/selftest_tmp', t),
                            capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or r.stdout.strip() != 'OFFLINE':
        raise SystemExit(f'sandbox self-test failed (rc={r.returncode}): {r.stdout.strip()} {r.stderr.strip()[-400:]}\n'
                         'Run on a compute node (--condor); the tmux host cannot start contained Apptainer.')
    if r2.returncode != 0 or r2.stdout.strip() != 'persist':
        raise SystemExit(f'sandbox self-test: /tmp did not persist across commands (rc={r2.returncode}): '
                         f'{r2.stdout.strip()} {r2.stderr.strip()[-300:]}')


async def run_all(a, run_dir: Path, cfg: dict) -> None:
    problems = select_problems(a)
    examples = {p.problem_id: p for p in data.load_examples()} if a.example else {}
    ex_reps = a.example_repeats if a.example_repeats is not None else a.repeats
    work = ([(p, k) for k in range(a.repeats) for p in problems]
            + [(p, k) for k in range(ex_reps) for p in examples.values()])
    sys_prompt = prompts.system_prompt()
    items_path, tr_path = run_dir / 'items.jsonl', run_dir / 'transcripts.jsonl'
    states = final_states(read_jsonl(items_path))

    todo = []
    for p, k in work:
        st = states.get((p.problem_id, k))
        if st and st['status'] in ('ok', 'model_error'):
            continue
        if st and st['attempts'] >= a.max_attempts:
            continue
        todo.append((p, k, (st or {}).get('attempts', 0)))
    LOG.info('%d item x repeats in the run, %d to do', len(work), len(todo))

    ncpu = len(os.sched_getaffinity(0))
    exec_slots = asyncio.Semaphore(max(2, ncpu - 1))
    traj_slots = asyncio.Semaphore(a.concurrency)
    work_base = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / f'critpt_work_{os.getpid()}'
    work_base.mkdir(parents=True, exist_ok=True)
    gen = cfg['generation']

    def fresh_urls():
        return resolve_model(cfg['model']['spec'])['urls']

    gate = None
    if a.max_requests:
        gate = harness.LoadGate(a.max_requests, fresh_urls, start=a.start_requests, recover_after_s=a.grow_after_s,
                                override_path=run_dir / 'load_gate_override.json')
        gate_task = asyncio.create_task(gate.run())   # noqa: F841 - runs for the life of the event loop

    def client_factory():
        return harness.EndpointClient(model=cfg['model']['served_name'], base_urls=cfg['model']['urls'],
                                      context_window_tokens=gen['context_window_tokens'],
                                      max_output_tokens=gen['max_output_tokens'], sampling=gen['request'],
                                      adapters=gen.get('adapters', ['mirror_reasoning_content']),
                                      request_timeout=gen['request_timeout_s'], resolve=fresh_urls, gate=gate,
                                      endpoint_wait_s=a.endpoint_wait_h * 3600)

    def sandbox_factory():
        return harness.OfflineSandbox(sif=SIF, isolation=a.isolation, work_base=work_base, exec_slots=exec_slots,
                                      cmd_timeout=a.cmd_timeout)

    def run_python(prog: str, timeout: int, stdin: str | None = None):
        import tempfile
        with tempfile.TemporaryDirectory(dir=work_base) as d:
            Path(d, 'check.py').write_text(prog)
            return subprocess.run(harness.sandbox_argv(str(SIF), a.isolation, d, 'python -I check.py'),
                                  capture_output=True, text=True, timeout=timeout, input=stdin)

    async def checks(row: dict, pid: str) -> None:
        """Offline checks of an `ok` answer, in the sandbox: format/execution for every item, and correctness for
        the example items (the only ones with public answers)."""
        if row['status'] != 'ok':
            return
        row['probe'] = await asyncio.to_thread(consensus.probe, row['answer_code'], run_python)
        row['probe_version'] = consensus.PROBE_VERSION
        if pid in examples:
            row['local_check'] = await asyncio.to_thread(grading.check_example, row['answer_code'],
                                                         examples[pid].golden_code, run_python)
            row['score'] = int(row['local_check']['correct'])

    done = {'n': 0}
    t_start = time.time()

    async def one(p: data.Problem, k: int, prev_attempts: int):
        async with traj_slots:
            tr = await harness.run_trajectory(
                client_factory=client_factory, system_prompt=sys_prompt,
                user_prompt=prompts.user_prompt(p, max_turns=a.max_turns, cmd_timeout=a.cmd_timeout),
                sandbox_factory=sandbox_factory, max_turns=a.max_turns, label=f'{p.problem_id} r{k}')
        attempt = prev_attempts + 1
        key = f'{p.problem_id}/r{k}/a{attempt}'
        append_jsonl(tr_path, {'key': key, 'item_id': p.problem_id, 'repeat': k, 'attempt': attempt,
                               'status': tr.status, 'error': tr.error, 'reason': tr.reason,
                               'derivation': tr.derivation, 'answer_code': tr.answer_code, 'history': tr.history})
        row = {'item_id': p.problem_id, 'repeat': k, 'attempt': attempt, 'status': tr.status,
               'score': 0 if tr.status == 'model_error' else None, 'error': tr.error,
               'answer_code': tr.answer_code, 'turns': tr.turns, 'input_tokens': tr.input_tokens,
               'output_tokens': tr.output_tokens, 'reasoning_tokens': tr.reasoning_tokens,
               'n_commands': tr.n_commands, 'n_api_retries': tr.n_api_retries, 'n_summaries': tr.n_summaries,
               'n_cap_overshoot': tr.n_cap_overshoot,
               'seconds': tr.seconds, 'transcript': f'transcripts.jsonl#{key}',
               'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        if p.problem_id in examples and tr.status != 'ok':
            row['score'] = 0
        await checks(row, p.problem_id)
        append_jsonl(items_path, row)
        done['n'] += 1
        LOG.info('[%d/%d %.0fs] %s r%d: %s %s turns=%d out=%d %s', done['n'], len(todo), time.time() - t_start,
                 p.problem_id, k, tr.status, tr.error, tr.turns, tr.output_tokens,
                 json.dumps(row.get('local_check') or {k: v for k, v in (row.get('probe') or {}).items() if k != 'fp'}))
        return tr.status

    # Infra errors are retried in later rounds (fresh trajectory), up to --max-attempts in total.
    while todo:
        results = await asyncio.gather(*(one(p, k, n) for p, k, n in todo))
        states = final_states(read_jsonl(items_path))
        todo = [(p, k, states[(p.problem_id, k)]['attempts']) for (p, k, _), s in zip(todo, results)
                if s == 'infra_error' and states[(p.problem_id, k)]['attempts'] < a.max_attempts]
        if todo:
            LOG.warning('retrying %d infra errors', len(todo))
            await asyncio.sleep(30)

    if gate is not None:
        atomic_json(run_dir / 'load_gate.json', {'max_requests': gate.max_limit, 'final_limit': gate.limit,
                                                 'changes': gate.history})

    # Backfill the offline checks for rows written before they existed (same attempt number: not a new attempt).
    for (pid, k), st in final_states(read_jsonl(items_path)).items():
        if (st['status'] == 'ok' and st.get('probe_version', 1 if 'probe' in st else 0) != consensus.PROBE_VERSION
                and (pid in examples or pid in {p.problem_id for p in problems})):
            row = {x: y for x, y in st.items() if x != 'attempts'}
            await checks(row, pid)
            append_jsonl(items_path, row)


def job_cpus() -> int:
    """The CPUs this process may use: a condor job's RequestCpus (condor does not pin jobs to cores here, so the
    affinity mask is the whole node), else the affinity mask."""
    n = len(os.sched_getaffinity(0))
    ad = os.environ.get('_CONDOR_JOB_AD')
    try:
        for line in Path(ad).read_text().splitlines() if ad else []:
            k, _, v = line.partition('=')
            if k.strip() == 'RequestCpus':
                return max(1, min(n, int(float(v.strip()))))
    except (OSError, ValueError):
        pass
    return n


async def reprobe(a, run_dir: Path, cfg: dict) -> None:
    """Recompute the offline checks (probe fingerprint, example local_check) of every final `ok` row whose probe is
    older than consensus.PROBE_VERSION. Appends a row with the same attempt number (not a new attempt)."""
    import tempfile
    examples = {p.problem_id: p for p in data.load_examples()}
    work_base = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / f'critpt_reprobe_{os.getpid()}'
    work_base.mkdir(parents=True, exist_ok=True)

    def run_python(prog: str, timeout: int, stdin: str | None = None):
        with tempfile.TemporaryDirectory(dir=work_base) as d:
            Path(d, 'check.py').write_text(prog)
            return subprocess.run(harness.sandbox_argv(str(SIF), a.isolation, d, 'python -I check.py'),
                                  capture_output=True, text=True, timeout=timeout, input=stdin)

    items_path = run_dir / 'items.jsonl'
    todo = [st for st in final_states(read_jsonl(items_path)).values()
            if st['status'] == 'ok' and st.get('probe_version', 1 if 'probe' in st else 0) != consensus.PROBE_VERSION]
    LOG.info('reprobe %s: %d ok rows below probe version %d', cfg['run_id'], len(todo), consensus.PROBE_VERSION)
    slots = asyncio.Semaphore(max(2, job_cpus() - 1))

    async def one(st: dict) -> None:
        row = {x: y for x, y in st.items() if x != 'attempts'}
        async with slots:
            row['probe'] = await asyncio.to_thread(consensus.probe, row['answer_code'], run_python)
            row['probe_version'] = consensus.PROBE_VERSION
            if row['item_id'] in examples:
                row['local_check'] = await asyncio.to_thread(grading.check_example, row['answer_code'],
                                                             examples[row['item_id']].golden_code, run_python)
                row['score'] = int(row['local_check']['correct'])
        append_jsonl(items_path, row)

    await asyncio.gather(*(one(st) for st in todo))
    LOG.info('reprobe %s: done', cfg['run_id'])


def write_config(a, run_dir: Path) -> dict:
    for waited in range(0, 7200, 60):                  # the shared server may be restarting
        try:
            m = resolve_model(a.model)
            break
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            LOG.warning('model %s not available (%s); waiting', a.model, e)
            time.sleep(60)
    else:
        raise SystemExit(f'{a.model} unavailable for 2 h')
    profile = m.pop('request_profile', {})
    if a.serving_record:                               # a url: spec standing in for a registry entry (gate bypass)
        reg = json.loads((serving.registry_root(SERVING) / a.serving_record / 'endpoints.json').read_text())
        profile = reg.get('request') or {}
        m['serving_record'] = a.serving_record
    request, adapters = build_request(profile, a)
    served_len = max_model_len(m['urls'][0], m['served_name'])
    ctx = min(served_len, a.context_window) if a.context_window else served_len
    sp = prompts.system_prompt()
    cfg = {
        'benchmark': 'critpt-v2', 'run_id': a.run_id, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'git_sha': git_sha(),
        'model': {**m, 'aa_slug': a.aa_slug or AA_ROWS.get(m['served_name'], {}).get('aa_slug'),
                  'aa_name': AA_ROWS.get(m['served_name'], {}).get('aa_name')},
        'generation': {'request': request, 'adapters': adapters, 'request_profile': profile,
                       'reasoning_effort': request_effort(request),
                       'max_output_tokens': a.max_output_tokens,
                       'context_window_tokens': ctx, 'served_max_model_len': served_len,
                       'context_cap': a.context_window or None, 'request_timeout_s': harness.REQUEST_TIMEOUT,
                       'note': 'request = the endpoint request profile (overrides) + CLI flags; context_window_tokens = '
                               'min(served max_model_len, --context-window): every request\'s max_completion_tokens is '
                               'clamped to the room left under it (harness.EndpointClient), and Stirrup summarizes at '
                               '70% of it; max_output_tokens: an extra per-response cap (null = none)'},
        'dataset': {**data.check_pins(), 'n_problems_total': data.N_PROBLEMS,
                    'problems': [p.problem_id for p in select_problems(a)], 'example': bool(a.example),
                    'example_repeats': (a.example_repeats if a.example_repeats is not None else a.repeats) if a.example else 0},
        'harness': {'name': 'stirrup', 'version': '0.2.1', 'max_turns': a.max_turns, 'finish_tool': 'finish(reason, derivation, answer)',
                    'cmd_timeout_s': a.cmd_timeout, 'context_summarization_cutoff': 0.7,
                    'sandbox': {'sif': str(SIF), 'sif_sha256': SIF_SHA256.read_text().split()[0],
                                'isolation': a.isolation, 'mem_limit_kb': harness.SANDBOX_MEM_KB,
                                'threads': harness.SANDBOX_THREADS},
                    'system_prompt_sha': prompts.sha(sp), 'system_prompt': sp,
                    'user_prompt_sha': prompts.sha(prompts.user_prompt(data.load_example(), max_turns=a.max_turns,
                                                                       cmd_timeout=a.cmd_timeout))},
        'repeats': a.repeats, 'limit': a.limit, 'concurrency': a.concurrency, 'max_requests': a.max_requests, 'start_requests': a.start_requests, 'max_attempts': a.max_attempts,
        'grader': {'default': {'name': 'consensus proxy', 'reference': a.reference,
                               'reference_run': consensus.DEFAULT_REFERENCE_RUN,
                               'note': 'agreement with the reference model\'s majority answer on the problems it '
                                       'answers consistently; tolerances from each problem\'s stated precision '
                                       '(consensus.problem_rules); parameters are stored in the reference file'},
                   'exact_optional': {'name': 'CritPt authors\' server via AA', 'url': grading.API_URL,
                                      'note': 'aggregate accuracy per 70-problem batch; one batch per repeat; needs '
                                              'an AA key with CritPt approval'}},
        'judge': None,
        'note': a.note,
    }
    old = run_dir / 'config.json'
    if old.exists():
        prev = json.loads(old.read_text())
        for k in ('model', 'generation', 'dataset', 'repeats', 'harness'):
            was, now = prev.get(k), cfg.get(k)
            if k == 'model':
                was = {x: y for x, y in (was or {}).items() if x not in ('urls', 'job_id')}
                now = {x: y for x, y in now.items() if x not in ('urls', 'job_id')}
            if was != now and not a.force_config:
                raise SystemExit(f'{old}: "{k}" differs from this invocation (resume with the same settings, or pass '
                                 f'--force-config):\n  was {json.dumps(was)[:400]}\n  now {json.dumps(now)[:400]}')
    atomic_json(old, cfg)
    return cfg


# ----------------------------------------------------------------------------------------------- grading / summary

def aa_slug_for(cfg: dict) -> str | None:
    """AA's row for this run: the row for the reasoning effort used, else the configured slug."""
    rows = AA_ROWS.get(cfg['model']['served_name'], {})
    configured = cfg['model'].get('aa_slug')
    if configured and configured != rows.get('aa_slug'):
        return configured                               # an explicit --aa-slug wins
    return (rows.get('aa_slug_by_effort') or {}).get(run_effort(cfg)) or configured


def run_effort(cfg: dict) -> str | None:
    g = cfg['generation']
    return g.get('reasoning_effort') or request_effort(g['request'])


def aa_reference(cfg: dict) -> dict | None:
    slug = aa_slug_for(cfg)
    if not slug:
        return None
    ref = json.loads(AA_REF.read_text())
    out = {'source': ref['source'], 'retrieved': ref['retrieved']}
    r = ref['models'].get(slug)
    if r:
        out.update({'name': r['name'], 'full70': round(100 * r['full70'], 1), 'hard40': round(100 * r['hard40'], 1)})
    else:
        lb = json.loads(LEADERBOARD.read_text())['models']
        for m in lb:
            if m.get('slug') == slug and m['scores'].get('critpt-v2-hard') is not None:
                out.update({'name': m['name'], 'full70': None, 'hard40': round(100 * m['scores']['critpt-v2-hard'], 1)})
    if 'hard40' not in out:
        return None
    out['slug'] = slug
    got = run_effort(cfg)
    by_effort = AA_ROWS.get(cfg['model']['served_name'], {}).get('aa_slug_by_effort') or {}
    if by_effort and got not in by_effort:
        out['note'] = f'AA lists no row for reasoning_effort {got}; this is {out["name"]}'
    if out.get('full70') is None:
        out['full70_note'] = 'AA publishes no full-70 score for this variant (only the hard-40 index number)'
    return out


def grade(a, run_dir: Path) -> None:
    cfg = json.loads((run_dir / 'config.json').read_text())
    problems = data.load_problems()
    run_ids = set(cfg['dataset']['problems'])
    states = final_states(read_jsonl(run_dir / 'items.jsonl'))
    gpath = run_dir / 'grading.jsonl'
    graded = {r['repeat']: r for r in read_jsonl(gpath) if r.get('http_status') == 200}
    (run_dir / 'batches').mkdir(exist_ok=True)
    key = grading.api_key() if a.submit else None
    if a.submit and not key:
        raise SystemExit('no AA API key: set AA_API_KEY or write it to ' + str(grading.KEY_FILE))
    per_repeat = []
    for k in range(cfg['repeats']):
        st = {pid: states.get((pid, k)) for pid in run_ids}
        pending = [pid for pid, s in st.items() if s is None or (s['status'] == 'infra_error'
                                                                  and s['attempts'] < cfg['max_attempts'])]
        # Deadline cutoff (--cutoff): items with no answer yet are submitted blank and count as wrong (0).
        cutoff = sorted(pending) if getattr(a, 'cutoff', False) else []
        if cutoff:
            pending = []
        n_infra = sum(1 for s in st.values() if s and s['status'] == 'infra_error')
        answers = {pid: s['answer_code'] for pid, s in st.items() if s and s['status'] == 'ok'}
        n_scored = len(run_ids) - n_infra
        info = {'repeat': k, 'n_run': len(run_ids), 'n_ok': len(answers),
                'n_model_error': sum(1 for s in st.values() if s and s['status'] == 'model_error'),
                'n_infra_error': n_infra, 'n_pending': len(pending), 'n_cutoff': len(cutoff), 'n_scored': n_scored}
        if pending:
            info['state'] = 'incomplete'
            per_repeat.append(info)
            continue
        body = grading.build_batch(problems, answers, model=cfg['model']['served_name'],
                                   generation_config={'run_id': cfg['run_id'], 'repeat': k,
                                                      'harness': ('pi-1.2.0' if cfg['harness'].get('name') == 'pi'
                                                                  else 'stirrup-0.2.1'),
                                                      **cfg['generation']['request']},
                                   batch_metadata={'run_id': cfg['run_id'], 'repeat': k, 'n_answered': len(answers)})
        bpath = run_dir / 'batches' / f'repeat_{k}.json'
        atomic_json(bpath, body)
        info['batch'] = str(bpath.relative_to(run_dir))
        if k in graded:
            g = graded[k]
            info.update(state='graded', accuracy_server=g['response']['accuracy'])
        elif key:
            LOG.info('submitting repeat %d (%d answered of %d)', k, len(answers), data.N_PROBLEMS)
            res = grading.submit_batch(body, key=key)
            row = {'repeat': k, 'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%S'), **res}
            append_jsonl(gpath, row)
            if res['http_status'] != 200:
                raise SystemExit(f'AA returned HTTP {res["http_status"]}: {json.dumps(res["response"])[:500]}')
            info.update(state='graded', accuracy_server=res['response']['accuracy'],
                        judge_error_count=res['response'].get('judge_error_count'),
                        server_timeout_count=res['response'].get('server_timeout_count'))
        else:
            info['state'] = 'ready'
        if info.get('accuracy_server') is not None:
            # blanks (not attempted, infra) are wrong on the server; rescale them out
            info['accuracy'] = info['accuracy_server'] * data.N_PROBLEMS / max(1, n_scored)
        per_repeat.append(info)
    write_summary(run_dir, cfg, states, per_repeat, reference=getattr(a, 'reference', consensus.DEFAULT_REFERENCE),
                  cutoff=getattr(a, 'cutoff', False))


def write_summary(run_dir: Path, cfg: dict, states: dict, per_repeat: list[dict], *,
                  reference: str = consensus.DEFAULT_REFERENCE, cutoff: bool = False) -> None:
    """Headline: AA-server full-70 pass@1 when repeats were graded there (the exact path); otherwise agreement with
    the consensus reference (the default path, user decision 2026-11-02), which is a ranking signal on the problems
    the reference solves consistently, not an AA-comparable score."""
    run_ids = set(cfg['dataset']['problems'])
    rows = [s for (pid, _), s in states.items() if pid in run_ids]
    n_infra = sum(1 for s in rows if s['status'] == 'infra_error')
    n_total = len(run_ids) * cfg['repeats']
    accs = [r['accuracy'] for r in per_repeat if r.get('accuracy') is not None]
    full = len(run_ids) == data.N_PROBLEMS
    score = 100 * statistics.fmean(accs) if accs else None
    if len(accs) >= 2:
        t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(len(accs), 2.0)
        ci = t * 100 * statistics.stdev(accs) / math.sqrt(len(accs))
    elif accs:
        n = sum(r['n_scored'] for r in per_repeat if r.get('accuracy') is not None)
        ci = 196 * math.sqrt(max(accs[0] * (1 - accs[0]), 1e-9) / max(n, 1))
    else:
        ci = None
    ex = [s for (pid, _), s in states.items() if pid.startswith('quantum_error_correction')]
    fc = [s.get('probe') or {} for s in rows if s['status'] == 'ok']
    toks = [s['output_tokens'] for s in rows if s['status'] != 'infra_error']
    proxy = consensus.proxy_summary(cfg['run_id'], reference, cutoff=cutoff)
    aa_graded = bool(accs)
    if aa_graded:
        metric, headline, head_ci, grading_path = (
            'critpt_v2_full70_pass@1' if full else 'critpt_v2_subset_pass@1', score, ci, 'aa-server')
    else:   # all 70 problems against the reference's plurality answers (user decision, #185)
        allp = proxy.get('all_problems') or {}
        metric, headline, head_ci, grading_path = (
            f'critpt_v2_all70_agreement@{reference}' if full else f'critpt_v2_subset_agreement@{reference}',
            allp.get('agreement_with_plurality'), allp.get('ci95'), 'consensus-proxy')
    gen = cfg['generation']
    summary = {
        'benchmark': 'critpt-v2',
        'metric': metric,
        'score': None if headline is None else round(headline, 2),
        'grading': grading_path,
        'n_items': len(run_ids), 'n_repeats': cfg['repeats'],
        'ci95': head_ci,
        'ci95_interval': (None if headline is None or head_ci is None
                          else [round(headline - head_ci, 2), round(headline + head_ci, 2)]),   # ci95 is a half-width
        # M3.2 protocol fields (#175)
        'model': cfg['model']['spec'], 'served_name': cfg['model']['served_name'],
        'reasoning_effort': run_effort(cfg),
        'context_window': gen.get('context_window_tokens'), 'context_cap': gen.get('context_cap'),
        'judge': None, 'code_sha': cfg.get('git_sha'),
        'n_infra': n_infra,
        'headline_note': ('all 70 problems (proxy.all_problems): per problem, the share of the run\'s repeats whose '
                          'answer matches the reference\'s most common answer; problems without a comparable '
                          'reference answer count 0; CI over problems' if not aa_graded else 'AA server pass@1'),
        'proxy': proxy,
        'aa_server': {'metric': 'critpt_v2_full70_pass@1' if full else 'critpt_v2_subset_pass@1',
                      'score': None if score is None else round(score, 2), 'n_repeats_graded': len(accs),
                      'ci95': None if ci is None else round(ci, 2),
                      'ci95_note': 'over repeats (t), or binomial for one repeat; the server returns no per-item '
                                   'results'},
        'n_infra_errors': n_infra, 'infra_error_rate': round(n_infra / max(1, n_total), 4),
        'n_unfinished': n_total - len(rows), 'cutoff_applied': cutoff,
        'n_cutoff': (n_total - len(rows)) if cutoff else 0,
        # valid: infra errors <= 2%, nothing unfinished unless a deadline cutoff was applied (cutoffs count as 0),
        # and the headline was actually computed
        'n_bad_jsonl_lines': BAD_LINES.count(run_dir),
        'valid': (n_infra / max(1, n_total) <= 0.02 and (cutoff or n_total - len(rows) == 0)
                  and (len(accs) == cfg['repeats'] if aa_graded else proxy.get('status') == 'ok')
                  and not BAD_LINES.count(run_dir)),
        'partial': not full,
        'repeats': {'run': cfg['repeats'], 'aa': 5,
                    'note': f'{cfg["repeats"]} repeat(s) per problem; AA runs 5 and reports their mean pass@1'},
        'ci95_over': ('repeats (t over per-repeat accuracies; binomial over items for one repeat)' if aa_graded
                      else 'problems (normal approx. over the per-problem agreement, each the mean of its repeats)'),
        'aa_reference': aa_reference(cfg),
        'aa_reference_note': ('context only, not comparable: AA\'s numbers are pass@1 from the authors\' grader '
                              '(full70 = all 70, hard40 = the private 40-problem index subset); our headline is '
                              + ('the AA-server pass@1 on all 70 (comparable to full70)' if aa_graded else
                                 f'agreement with the {reference} reference answers on all 70 problems, a different '
                                 'scale (closest to AA\'s full70 in coverage, not in grading)')),
        'per_repeat': per_repeat,
        'n_cap_overshoot': sum(s.get('n_cap_overshoot') or 0 for s in rows),
        # Pi harness (#348): where each ok answer came from (file, or the final-message fallback); Stirrup: finish
        'answer_source': dict(collections.Counter(s.get('answer_source') or 'finish_tool'
                                                  for s in rows if s['status'] == 'ok')),
        'model_errors': {e: sum(1 for s in rows if s['status'] == 'model_error' and (s['error'] or '').startswith(e))
                         for e in ('no_finish', 'output_limit', 'context_overflow', 'summary_failed', 'response_timeout')},
        'mean_turns': round(statistics.fmean([s['turns'] for s in rows]), 1) if rows else None,
        'mean_output_tokens': round(statistics.fmean(toks)) if toks else None,
        'format_check': {'n_ok_answers': len(fc), 'checked': sum(1 for f in fc if f),
                         'unfilled': sum(1 for f in fc if f.get('unfilled')),
                         'exec_error': sum(1 for f in fc if f and not f.get('executes')),
                         'no_answer_fn': sum(1 for f in fc if f.get('executes') and not f.get('has_answer')),
                         'returns_placeholder': sum(1 for f in fc if f.get('result_is_placeholder'))},
        'example_local_check': [{'item_id': s['item_id'], 'repeat': s['repeat'], 'status': s['status'],
                                 **(s.get('local_check') or {})} for s in sorted(ex, key=lambda s: (s['item_id'], s['repeat']))] or None,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        # the code that graded (may differ from config.json's git_sha, the code that ran), and the reference build
        'graded_with': {'git_sha': git_sha(), 'reference': reference,
                        'reference_created_at': proxy.get('reference_created_at')},
    }
    from aa_evals.tracejudge import fold_summary   # M3.4 trace judge: n_flagged from flags.jsonl
    summary = fold_summary(run_dir, summary)
    atomic_json(run_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_repeat'}, indent=1))
    for r in per_repeat:
        print(json.dumps(r))


# ----------------------------------------------------------------------------------------------- condor

# the probe sandbox: 3 answers at a time. RAM: answers can allocate GBs for a moment, and the sandbox's memory is
# not in the RSS (8 GB was held "over memory limit" at a 38 MB RSS, 2026-11-13; c30's 64 GB jobs: 1 hold in 39);
# 8 GB per CPU is the cheap nodes' ratio and costs no units (units = max(CPUs, GB / 16))
REPROBE_CPUS, REPROBE_MEM_GB = 4, 32


def submit_condor(a, argv: list[str], module: str = 'aa_evals.critpt.run', job_dir: Path | None = None) -> None:
    dirty = subprocess.run(['git', '-C', str(WORKTREE), 'status', '--porcelain', '--', 'aa_evals'], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise SystemExit(f'commit aa_evals first (jobs run a frozen snapshot of HEAD):\n{dirty}')
    sha = git_sha()
    code = RUNS / '_code' / sha[:12]
    if not code.exists():
        tmp = code.with_name(code.name + '.partial')
        tmp.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(['git', '-C', str(WORKTREE), 'archive', sha, 'aa_evals'], capture_output=True, check=True)
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=arc.stdout, check=True)
        tmp.rename(code)
    run_dir = job_dir or RUNS / a.run_id
    (run_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    args = [x for x in argv if x != '--condor']
    # The job reads its arguments from a JSON file: free text (--note, --chat-template-kwargs) never has to survive
    # HTCondor's argument quoting.
    stamp = time.strftime("%Y%m%dT%H%M%S")
    argv_file = run_dir / 'jobs' / f'argv.{stamp}.json'
    argv_file.write_text(json.dumps(args))
    from aa_evals.common.condor_cost import bad_nodes   # the shared list, read now (tbshared; at submission)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    sub = f'''universe = vanilla
executable = {VENV_PY}
arguments = "-m {module} --argv-file {argv_file}"
initialdir = {code}
output = {run_dir}/jobs/run.$(Cluster).out
error = {run_dir}/jobs/run.$(Cluster).err
log = {run_dir}/jobs/condor.log
request_gpus = 0
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = 20G
+BypassLXCfs = "true"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin CRITPT_CODE_SHA={sha} CRITPT_RUNS={RUNS} PYTHONUNBUFFERED=1 APPTAINER_CACHEDIR=/data/aai/apptainer-cache{serving.condor_env()}"
requirements = {req} && (TARGET.GPUs == 0 || TARGET.GPUs =?= undefined)
queue
'''
    path = run_dir / 'jobs' / f'run.{stamp}.sub'
    path.write_text(sub)
    from aa_evals.common import condor_cost
    condor_cost.submit(path, getattr(a, 'cpu_bid', None))   # default: the CPU bid ladder (#378)


# ----------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--argv-file']:                    # condor jobs: arguments written by submit_condor
        argv = json.loads(Path(argv[1]).read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='serving:<name> or url:<base_url>@<served_name>')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int, default=None, help='a fixed random subset of N problems (smoke tests)')
    ap.add_argument('--problems', default='', help='comma-separated problem ids (instead of --limit)')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--example', action='store_true', help='also run the example challenge (graded locally)')
    ap.add_argument('--example-repeats', type=int, default=None, help='repeats for the example items (default: --repeats)')
    ap.add_argument('--concurrency', type=int, default=64, help='trajectories in flight')
    ap.add_argument('--max-requests', type=int, default=None,
                    help='requests in flight, backing off while the server queues (shared servers)')
    ap.add_argument('--start-requests', type=int, default=None,
                    help='with --max-requests: start here and grow toward --max-requests while the server has headroom')
    ap.add_argument('--grow-after-s', type=float, default=600.0,
                    help='with --max-requests: seconds with no queue and KV < 0.9 before the limit grows by 8')
    ap.add_argument('--endpoint-wait-h', type=float, default=2.0,
                    help='how long a trajectory waits for a down server before it becomes an infra error')
    ap.add_argument('--max-turns', type=int, default=harness.MAX_TURNS)
    ap.add_argument('--cmd-timeout', type=int, default=harness.CMD_TIMEOUT)
    ap.add_argument('--max-attempts', type=int, default=3, help='per item x repeat, counting infra-error reruns')
    ap.add_argument('--isolation', default='apptainer-net-none', choices=['apptainer-net-none', 'unshare-inside'])
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--reasoning-effort', default=None)
    ap.add_argument('--chat-template-kwargs', default=None, help='JSON, e.g. {"enable_thinking": true}')
    ap.add_argument('--max-output-tokens', type=int, default=None, help='default: no cap (fill the context)')
    ap.add_argument('--context-window', type=int, default=CONTEXT_CAP,
                    help='context cap in tokens (M3.0: 131072 for regular evals; 0 = the served max_model_len, only '
                         'for the explicit Qwen3.8 full-context calibration runs)')
    ap.add_argument('--no-profile', action='store_true', help="ignore the endpoint's request profile (overrides, adapters)")
    ap.add_argument('--adapters', default=None, help='comma list replacing the history adapters '
                    '(inline_reasoning_history | mirror_reasoning_content)')
    ap.add_argument('--serving-record', default=None,
                    help='with a url: spec (e.g. a gate bypass): take the request profile from serving/<NAME>')
    ap.add_argument('--aa-slug', default=None, help='AA model slug for the reference numbers')
    ap.add_argument('--note', default=None, help='free text recorded in config.json (e.g. a server substitution)')
    ap.add_argument('--force-config', action='store_true')
    ap.add_argument('--condor', action='store_true', help='submit this command as a CPU condor job')
    # c30 (#378): reprobe jobs at 16 CPU / 64 GB used 0.6 cores (p50; p90 5, max 9 for minutes) and RSS <= 0.7 GB
    ap.add_argument('--cpus', type=int, default=None, help='condor job CPUs (default: 16 to generate, 4 for --reprobe)')
    ap.add_argument('--mem-gb', type=int, default=None, help='condor job RAM (default: 64 to generate, 32 for --reprobe)')
    ap.add_argument('--cpu-bid', type=int, default=None, help='fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)')
    ap.add_argument('--grade', action='store_true', help='build the batches and summary of a finished run')
    ap.add_argument('--submit', action='store_true', help='with --grade: send ungraded batches to AA (exact path)')
    ap.add_argument('--reprobe', action='store_true',
                    help='recompute the offline checks of a finished run with the current probe (sandbox: use --condor)')
    ap.add_argument('--cutoff', action='store_true',
                    help='with --grade: the run deadline passed; unanswered items count as 0 (reported as n_cutoff)')
    ap.add_argument('--reference', default=consensus.DEFAULT_REFERENCE, help='consensus reference for the proxy')
    a = ap.parse_args(argv)
    if a.cpus is None:
        a.cpus = REPROBE_CPUS if a.reprobe else 16
    if a.mem_gb is None:
        a.mem_gb = REPROBE_MEM_GB if a.reprobe else 64
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for name in ('httpx', 'httpx2', 'httpcore', 'openai'):
        logging.getLogger(name).setLevel(logging.WARNING)
    run_dir = RUNS / a.run_id
    if a.grade:
        grade(a, run_dir)
        return
    if a.reprobe:
        if a.condor:
            submit_condor(a, argv)
            return
        sandbox_selftest(a.isolation)
        asyncio.run(reprobe(a, run_dir, json.loads((run_dir / 'config.json').read_text())))
        return
    if not a.model:
        ap.error('--model is required to run')
    if (run_dir / 'config.json').exists() and (json.loads((run_dir / 'config.json').read_text()).get('harness')
                                               or {}).get('name') == 'pi':
        raise SystemExit('a Pi-harness run generates with aa_evals.common.pi_eval (aa_evals/critpt/pi/run.py)')
    if a.condor:
        submit_condor(a, argv)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = write_config(a, run_dir)
    LOG.info('run %s on %s: %s', a.run_id, socket.gethostname(), json.dumps(cfg['generation']))
    sandbox_selftest(a.isolation)
    asyncio.run(run_all(a, run_dir, cfg))
    grade(argparse.Namespace(submit=False, reference=a.reference), run_dir)


if __name__ == '__main__':
    main()

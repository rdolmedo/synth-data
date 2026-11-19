"""CritPt under the TB 2.1 Pi harness (#348): generation by aa_evals.common.pi_eval, grading unchanged.

  python -m aa_evals.critpt.pi.run init    --run-id ID --model serving:NAME [--reasoning-effort E] [--repeats 2] \
      [--problems a,b | --limit N] [-- PI_EVAL_ARGS...]
  python -m aa_evals.critpt.pi.run collect --run-id ID [--follow]      # import answers (CPU, no sandbox)
  python -m aa_evals.critpt.pi.run status  --run-id ID
  # then, as for a Stirrup run (the probe runs answers in the SIF: a compute node, so --condor):
  python -m aa_evals.critpt.run --run-id ID --reprobe --condor
  python -m aa_evals.critpt.run --run-id ID --grade                     # consensus proxy vs ds41-max, unchanged

Two run directories with the same id: the CritPt run (`$CRITPT_RUNS/<id>`: config.json, items.jsonl,
transcripts.jsonl, summary.json, the layout `run.py --reprobe/--grade`, the consensus scorer and the trace judge
read) and the pi_eval run (`<pi_eval RUNS_ROOT>/critpt_pi/<id>`: queue, seeds, retries, trial archives).

`collect` writes one items.jsonl row per resolved attempt. The submission is `/work/answer.py` as the episode left
it (a Harbor artifact), fences stripped as `harness.strip_fences` does: `ok` with that `answer_code` when it is
non-empty valid Python, else `model_error` (`no_finish: ...`, scored 0, as a Stirrup trajectory without a
successful `finish`; the Stirrup finish tool refused empty or unparsable answers, so neither ever reached grading).
An attempt whose infrastructure retries ran out is `infra_error`. `--reprobe` then fingerprints the ok answers in
the sandbox, and `--grade` scores agreement with the reference exactly as for Stirrup runs.
"""
from __future__ import annotations

if __name__ == '__main__':   # a launcher may use the harbor venv: re-execute where the data libraries are
    from aa_evals.common.pi_eval.port import ensure_python
    ensure_python(('pyarrow', 'jinja2', 'yaml', 'stirrup'), '/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python', 'aa_evals.critpt.pi.run')

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

from aa_evals.common import fastio
from aa_evals.common.pi_eval import port
from aa_evals.critpt import consensus, data
from aa_evals.critpt import run as C
from aa_evals.critpt.harness import strip_fences
from aa_evals.critpt.pi import tasks as PT

BENCH = 'critpt_pi'
SEED_NAMESPACE = 'critpt-pi-v1'
REPEATS = 2


def pi_runs_root() -> Path:
    return port.runs_root()


def init(a) -> dict:
    run_dir = C.RUNS / a.run_id
    if (run_dir / 'config.json').exists():
        sys.exit(f'{run_dir} exists; use a new --run-id (resume the generation with pi_eval.run --run-id)')
    bundle = Path(a.bundle) if a.bundle else PT.OUT / f'{PT.BUNDLE_VERSION}.tar'
    manifest = json.loads(bundle.with_suffix('.json').read_text())
    problems = [p.problem_id for p in C.select_problems(argparse.Namespace(problems=a.problems, limit=a.limit))]
    port.launch(port.base_args(BENCH, a.run_id, a.model, bundle, problems, a.repeats, SEED_NAMESPACE,
                               a.reasoning_effort) + list(a.pi_args))
    pi_run = pi_runs_root() / BENCH / a.run_id
    pcfg = json.loads((pi_run / 'config.json').read_text())
    req = pcfg.get('request') or {}
    served = pcfg['served']
    cfg = {
        'benchmark': 'critpt-v2', 'run_id': a.run_id, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'git_sha': C.git_sha(),
        'model': {'spec': a.model, 'served_name': served,
                  'aa_slug': a.aa_slug or C.AA_ROWS.get(served, {}).get('aa_slug'),
                  'aa_name': C.AA_ROWS.get(served, {}).get('aa_name')},
        'generation': {'request': req.get('overrides') or {}, 'adapters': req.get('adapters') or [],
                       'request_profile': pcfg.get('registry_request'),
                       'reasoning_effort': (req.get('overrides') or {}).get('reasoning_effort'),
                       'max_output_tokens': pcfg['protocol']['max_output_tokens'],
                       'context_window_tokens': pcfg['protocol']['context_window'],
                       'context_cap': pcfg['protocol']['context_window'], 'request_timeout_s': None,
                       'note': 'Pi harness (#348): pi_eval protocol pi180; context_policy '
                               f'{pcfg["protocol"]["context_policy"]} (no compaction)',
                       'finish_notice': pcfg['protocol'].get('finish_notice')},   # #375/#382
        'dataset': {**data.check_pins(), 'n_problems_total': data.N_PROBLEMS, 'problems': problems,
                    'example': False, 'example_repeats': 0},
        'harness': {'name': 'pi', 'version': pcfg['protocol']['pi_version'], 'label': pcfg.get('harness_label'),
                    'pi_run': str(pi_run), 'pi_eval_code': pcfg.get('code'), 'agent_cap_sec': pcfg['agent_cap_sec'],
                    'answer_file': PT.ANSWER, 'max_turns': None, 'prompt_disclosures': [PT.END_DISCLOSURE],
                    'bundle': {'path': str(bundle), 'sha256': manifest['sha256'], 'version': manifest['bundle_version'],
                               'base_ref': manifest['base_ref']},
                    'sandbox': {'sif': str(PT.SIF), 'sif_sha256': PT.SIF_SHA256, 'isolation': 'harbor_ns offline',
                                'mem_limit_kb': 16 * 1024 * 1024, 'threads': 4},
                    'system_prompt_sha': manifest['system_prompt_sha']},
        'repeats': a.repeats, 'limit': a.limit, 'concurrency': None, 'max_requests': None, 'start_requests': None,
        'max_attempts': 1,
        'grader': {'default': {'name': 'consensus proxy', 'reference': consensus.DEFAULT_REFERENCE,
                               'reference_run': consensus.DEFAULT_REFERENCE_RUN}},
        'judge': None, 'note': a.note,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_json(run_dir / 'config.json', cfg)
    print(f'{run_dir}: {len(problems)} problems x {a.repeats}; generation {pi_run}')
    return cfg


def answer_of(text: str | None) -> tuple[str | None, str | None]:
    """(answer_code, None) for a gradable answer file, else (None, why)."""
    if text is None:
        return None, f'no {PT.ANSWER}'
    code = strip_fences(text)
    if not code.strip():
        return None, f'{PT.ANSWER} is empty'
    try:
        ast.parse(code)
    except SyntaxError as e:
        return None, f'{PT.ANSWER} is not valid Python ({e.msg}, line {e.lineno})'
    return code, None


FENCE = re.compile(r'```(?:python|py)[ \t]*\n(.*?)```', re.S)


def final_message_answer(session: list | None) -> str | None:
    """The fallback answer (coordinator decision 2026-11-11, #348): the last ```python block of the agent's final
    message, if it passes the answer file's checks (strip_fences + ast.parse); None otherwise. Only for episodes the
    agent ended itself (end=finish) without writing /work/answer.py; never for context or cap endings, which stop
    mid-thought."""
    blocks = FENCE.findall(port.final_text(session))
    return answer_of('```python\n' + blocks[-1] + '```')[0] if blocks else None


def collection_problem(got: dict, file_ok: bool) -> str | None:
    """Why the collected answer file can't be trusted as what the episode left (infrastructure), or None. Harbor's
    artifact manifest gives the file's collection status; the placeholder verifier (tests/test.sh, in the sandbox
    after collection) rewards 1 iff /work/answer.py existed and parsed. A missing file is the model's outcome only
    when the verifier also found none."""
    status = got['artifact_status'].get(PT.ANSWER)
    reward = port.verifier_reward(got)
    if status == 'ok':
        return None if file_ok else 'artifact manifest says ok but no answer file was extracted'
    if status == 'failed':   # Harbor marks a source that does not exist as failed too
        return None if reward == 0 else f'answer file not collected (status failed) but verifier reward {reward}'
    return f'answer file missing from the artifact manifest (status {status!r})'


def import_row(run_dir: Path, cfg: dict, row: dict, scratch: Path) -> dict | None:
    """One items.jsonl row; None when the trial archive can't be read yet (retried on the next pass)."""
    pid, k = row['item_id'], row['repeat']
    base = {'item_id': pid, 'repeat': k, 'attempt': 1, 'harness': 'pi', 'pi_key': row.get('key'),
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if row['status'] != 'ok':
        return {**base, 'status': 'infra_error', 'score': None, 'error': f'infra: {row.get("infrastructure_reason")}',
                'answer_code': None, 'turns': 0, 'input_tokens': 0, 'output_tokens': 0, 'seconds': None}
    d = scratch / row['key']
    got = port.extract(Path(cfg['harness']['pi_run']), row, d, [PT.ANSWER])
    if not got['found']:
        if not got['permanent']:
            print(time.strftime('%H:%M:%S'), 'not imported (retry later):', row['key'], got['why'], flush=True)
            return None
        s = port.stats(row)   # nothing to retry: an infrastructure failure, never "no answer"
        return {**base, 'status': 'infra_error', 'score': None, 'error': f'infra: collection: {got["why"]}',
                'end': port.end_of(row), 'answer_code': None, 'turns': s.get('model_requests') or 0,
                'input_tokens': s.get('n_input_tokens'), 'output_tokens': s.get('n_output_tokens') or 0,
                'seconds': got['wall_sec'], 'pi': s}
    f = d / 'artifacts' / PT.ANSWER.lstrip('/')
    end = port.end_of(row)
    s = port.stats(row)
    bad = collection_problem(got, f.is_file())
    if bad:
        return {**base, 'status': 'infra_error', 'score': None, 'error': f'infra: {bad}', 'end': end,
                'answer_code': None, 'turns': s.get('model_requests') or 0, 'input_tokens': s.get('n_input_tokens'),
                'output_tokens': s.get('n_output_tokens') or 0, 'seconds': got['wall_sec'], 'pi': s}
    code, why = answer_of(f.read_text(errors='replace') if f.is_file() else None)
    source = 'file' if code else None
    if code is None and end == 'finish' and not f.is_file():
        code = final_message_answer(got['session'])
        source = 'final_message' if code else None
    key = f'{pid}/r{k}/a1'
    append_row(run_dir / 'transcripts.jsonl', {
        'key': key, 'item_id': pid, 'repeat': k, 'attempt': 1, 'status': 'ok' if code else 'model_error',
        'error': None if code else why, 'reason': end, 'derivation': port.final_text(got['session']),
        'answer_code': code, 'answer_source': source,
        'answer_file': f.read_text(errors='replace')[:200000] if f.is_file() else None,
        'history': got['session'], 'pi_archive': got['archive']})
    return {**base, 'status': 'ok' if code else 'model_error', 'score': None if code else 0,
            'error': None if code else f'no_finish: {why} (end={end})', 'end': end, 'answer_code': code,
            'answer_source': source,
            'turns': s.get('model_requests') or 0, 'input_tokens': s.get('n_input_tokens'),
            'output_tokens': s.get('n_output_tokens') or 0, 'reasoning_tokens': None, 'n_commands': None,
            'n_api_retries': 0, 'n_summaries': 0, 'n_cap_overshoot': 0, 'seconds': got['wall_sec'],
            'pi': s, 'transcript': f'transcripts.jsonl#{key}', **({'session_bad_lines': got['session_bad_lines']} if got.get('session_bad_lines') else {})}


def append_row(path: Path, row: dict) -> None:
    """C.append_jsonl, fsync'd and dropped from the page cache (#378): Pi sessions are large, and dirty pages of /data
    writes count against the collector's memory cgroup."""
    fastio.append_line(path, json.dumps(row, ensure_ascii=False))


def collect(a) -> dict:
    import shutil
    import tempfile
    run_dir = C.RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    pi_run = Path(cfg['harness']['pi_run'])
    for f in ('transcripts.jsonl', 'items.jsonl'):   # a collector killed mid-append left a torn last line
        if fastio.trim_torn_tail(run_dir / f):
            print(time.strftime('%H:%M:%S'), f'{f}: removed a torn last line (an interrupted append)', flush=True)
    while True:
        have = {(r['item_id'], r['repeat']): r.get('pi_key') for r in C.read_jsonl(run_dir / 'items.jsonl')}
        rows = port.resolved(pi_run)
        new = [r for r in rows if a.reimport or have.get((r['item_id'], r['repeat'])) != r.get('key')]
        a.reimport = False   # once: re-grades every resolved attempt with the current import rules
        scratch = Path(tempfile.mkdtemp(prefix='critpt_pi_'))
        try:
            for r in new:
                out = import_row(run_dir, cfg, r, scratch)
                if out is None:
                    continue
                append_row(run_dir / 'items.jsonl', out)
                print(time.strftime('%H:%M:%S'), 'imported', r['key'], out['status'], out.get('error') or '',
                      flush=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        n_items = len(cfg['dataset']['problems']) * cfg['repeats']
        n = len({(r['item_id'], r['repeat']) for r in C.read_jsonl(run_dir / 'items.jsonl')})
        print(f'collected {n}/{n_items}', flush=True)
        if not a.follow or n >= n_items:
            return {'done': n, 'items': n_items}
        time.sleep(120)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pi_args = []
    if '--' in argv:
        i = argv.index('--')
        argv, pi_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['init', 'collect', 'status'])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--model')
    ap.add_argument('--reasoning-effort', default=None)
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--problems', default='')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--aa-slug', default=None)
    ap.add_argument('--note', default=None)
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--reimport', action='store_true',
                    help='collect: re-import every resolved attempt (appends rows; then --reprobe and --grade)')
    a = ap.parse_args(argv)
    a.pi_args = pi_args
    if a.cmd == 'init':
        if not a.model:
            ap.error('--model is required')
        return init(a)
    if a.cmd == 'collect':
        return collect(a)
    cfg = json.loads((C.RUNS / a.run_id / 'config.json').read_text())
    st = C.final_states(C.read_jsonl(C.RUNS / a.run_id / 'items.jsonl'))
    print(json.dumps({'imported': len(st), 'by_status': {s: sum(1 for x in st.values() if x['status'] == s)
                                                          for s in ('ok', 'model_error', 'infra_error')},
                      'pi': port.status(Path(cfg['harness']['pi_run']))},
                     indent=1, default=str))


if __name__ == '__main__':
    main()

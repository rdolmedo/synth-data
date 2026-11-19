"""AutomationBench under the TB 2.1 Pi harness (#348): generation by aa_evals.common.pi_eval, grading unchanged.

  python -m aa_evals.automationbench.pi.run init    --run-id ID --model serving:NAME [--reasoning-effort E] \
      [--repeats 1] [--tasks a,b | --tasks @IDS_FILE | --limit N] [PI_EVAL_ARGS...]
  python -m aa_evals.automationbench.pi.run collect --run-id ID [--follow]   # grade + items.jsonl + summary.json
  python -m aa_evals.automationbench.pi.run status  --run-id ID

Arguments `init` does not know (e.g. `--dispatch DIR`, `--agent-wall-cap-sec 7200`, `--max-in-flight N`) go to
`aa_evals.common.pi_eval.run` unchanged, after the protocol defaults (`port.base_args`), so they win; so does
anything after `--`. Item ids are `<domain>.<task>`, the upstream path's `item_id`.

Runs in the AutomationBench venv (re-executes itself, as run.py does). Two run directories with the same id: the
AutomationBench run (`--out-root`/<id>: config.json, items.jsonl, transcripts.jsonl, summary.json, the layout of
upstream-harness runs, read by the trace judge and `run.py --summarize-only`) and the pi_eval run
(`<pi_eval RUNS_ROOT>/automationbench_pi/<id>`: queue, seeds, retries, trial archives).

`collect` grades each resolved attempt as `run.run_item` grades an upstream rollout: the task's assertions on the
initial world and on the final world (the world service's `/state/world.json` when the episode ended, collected by
Harbor), AA's score (`scoring.aa_task_score`; an exception in an assertion handler fails that assertion and is
listed in `grading_exceptions`), and upstream's own partial credit / strict pass. Every end of the episode (the
agent stopped, the 131k window filled, the 2 h environment-clock cap) is `ok` and graded on the world it left: under
this protocol the window just ends the episode (#348), where the upstream path scored a context overflow 0. No
collected world (the world service failed) or exhausted infrastructure retries is `infra_error`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = 'automationbench_pi'
SEED_NAMESPACE = 'automationbench-pi-v1'
REPEATS = 1   # AA and the M3 policy (#173): one repeat


def ensure_venv():
    try:
        import automationbench  # noqa: F401
        import verifiers  # noqa: F401
    except ImportError:
        reexec_module('aa_evals.automationbench.pi.run')


def reexec_module(mod: str):
    from aa_evals.automationbench.run import VENV
    py = os.path.join(VENV, 'bin', 'python')
    if os.environ.get('_AA_AB_REEXEC'):
        raise SystemExit(f'automationbench is not importable from {sys.executable}')
    root = str(HERE.parents[2])
    env = dict(os.environ, _AA_AB_REEXEC='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=root + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
    os.execve(py, [py, '-m', mod, *sys.argv[1:]], env)


def pi_runs_root() -> Path:
    from aa_evals.common.pi_eval import port
    return port.runs_root()


def init(a) -> dict:
    from aa_evals.automationbench import run as AB
    from aa_evals.automationbench.pi import tasks as PT
    from aa_evals.common.pi_eval import port
    run_dir = Path(a.out_root) / a.run_id
    if (run_dir / 'config.json').exists():
        sys.exit(f'{run_dir} exists; use a new --run-id (resume the generation with pi_eval.run --run-id)')
    bundle = Path(a.bundle) if a.bundle else PT.OUT / f'{PT.BUNDLE_VERSION}.tar'
    manifest = json.loads(bundle.with_suffix('.json').read_text())
    rows = AB.load_tasks(PT.DOMAINS)
    if a.tasks:
        want = task_ids(a.tasks)
        rows = [r for r in rows if r['_task_name'] in want]
        if len(rows) != len(want):
            sys.exit(f'unknown tasks: {sorted(set(want) - {r["_task_name"] for r in rows})}')
    elif a.limit:
        rows = AB.stratified(rows, a.limit, a.subset_seed)
    names = [r['_task_name'] for r in rows]
    pi_args = pi_eval_args(list(a.pi_args))
    port.launch(port.base_args(BENCH, a.run_id, a.model, bundle, names, a.repeats, SEED_NAMESPACE,
                               a.reasoning_effort) + pi_args)
    pi_run = pi_runs_root() / BENCH / a.run_id
    pcfg = json.loads((pi_run / 'config.json').read_text())
    req = pcfg.get('request') or {}
    sampling = req.get('overrides') or {}
    cfg = {
        'benchmark': AB.BENCHMARK, 'metric': AB.METRIC, 'run_id': a.run_id, 'run_dir': str(run_dir),
        **{k: v for k, v in AB.git_info().items()},
        'model': {'spec': a.model, 'served_name': pcfg['served']},
        'generation': {'sampling': sampling, 'adapters': req.get('adapters') or [],
                       'context_window': pcfg['protocol']['context_window'],
                       'context_cap': pcfg['protocol']['context_window'],
                       'max_output_tokens': pcfg['protocol']['max_output_tokens'],
                       'reasoning_effort': AB.reasoning_effort_of(sampling)},
        'dataset': {'source': AB.UPSTREAM_REPO, 'commit': AB.UPSTREAM_COMMIT, 'version': AB.UPSTREAM_VERSION,
                    'split': 'public', 'domains': PT.DOMAINS, 'n_tasks': len(rows),
                    'task_contract_digest': AB.contract_digest(rows),
                    'subset': ('--tasks' if a.tasks else f'stratified --limit {a.limit}' if a.limit else 'all')},
        'harness': {'name': 'pi', 'version': pcfg['protocol']['pi_version'], 'label': pcfg.get('harness_label'),
                    'pi_run': str(pi_run), 'pi_eval_code': pcfg.get('code'), 'agent_cap_sec': pcfg['agent_cap_sec'],
                    'toolset': 'api', 'tools_via': 'ab CLI -> world service (pi/tasks.py)', 'max_turns': None,
                    'bundle': {'path': str(bundle), 'sha256': manifest['sha256'],
                               'version': manifest['bundle_version'], 'images': manifest.get('images')}},
        'judge': None,
        'scoring': 'aa_evals.automationbench.scoring.aa_task_score on the collected final world; every episode end '
                   'graded; infra_error excluded',
        'limits': {'limit': a.limit, 'tasks': a.tasks, 'repeats': a.repeats},
        'pi_eval_args': pi_args,
        'repeats': a.repeats,
        'aa_reference': AB.aa_reference(pcfg['served'], sampling, a.aa_name),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=1))
    print(f'{run_dir}: {len(names)} tasks x {a.repeats}; generation {pi_run}')
    return cfg


def pi_eval_args(extra: list[str]) -> list[str]:
    """Extra pi_eval arguments; with the #344 dispatcher, its duration table's suite is `automationbench` (the
    pi_eval bench is `automationbench_pi`)."""
    has = lambda f: any(x == f or x.startswith(f + '=') for x in extra)  # noqa: E731
    if has('--dispatch') and not has('--dispatch-suite'):
        extra = extra + ['--dispatch-suite', 'automationbench']
    return extra


def task_ids(spec: str) -> list[str]:
    """`a,b,c`, or `@FILE` / an existing file path: ids one per line or comma-separated (# comments allowed)."""
    path = spec[1:] if spec.startswith('@') else spec
    if spec.startswith('@') or (os.sep in spec and Path(spec).is_file()):
        text = '\n'.join(x.split('#', 1)[0] for x in Path(path).read_text().splitlines())
    else:
        text = spec
    out = [t.strip() for t in text.replace(',', '\n').splitlines() if t.strip()]
    if len(set(out)) != len(out):
        sys.exit('duplicate task ids in --tasks')
    return out


def _dec(v):
    if isinstance(v, dict):
        if set(v) == {'__set__'}:
            return {_dec(x) if not isinstance(x, list) else tuple(_dec(y) for y in x) for x in v['__set__']}
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def restore_hidden(world, hidden: dict | None) -> None:
    """Put back the attributes the world service saved in hidden.json (image/world_server.py hidden_state), as
    the api tools set them (instance attributes outside the pydantic fields)."""
    for name, attrs in (hidden or {}).items():
        model = getattr(world, name) if name else world
        for k, v in attrs.items():
            object.__setattr__(model, k, _dec(v))


def grade(row: dict, final_json: str, hidden: dict | None = None) -> dict:
    """run.run_item's grading of one rollout, on a collected final world (world.json plus hidden.json)."""
    from automationbench.rubric import partial_credit, task_completed_correctly
    from automationbench.rubric.registry import AssertionRegistry
    from automationbench.runner import strip_none_values
    from automationbench.schema.world import WorldState
    from aa_evals.automationbench.scoring import aa_task_score
    info = json.loads(row['info']) if isinstance(row['info'], str) else row['info']
    assertions = [strip_none_values(x) for x in info['assertions']]
    initial_dict = strip_none_values(info.get('initial_state', {}))
    initial = WorldState(**initial_dict)
    final = WorldState.model_validate_json(final_json)
    restore_hidden(final, hidden)
    exc: list[str] = []

    def safe(world, which):
        def check(x):
            try:
                return AssertionRegistry.check(world, x)
            except Exception as e:  # noqa: BLE001  (run.py: a handler exception fails the assertion)
                exc.append(f'{which}:{x.get("type")}: {type(e).__name__}: {str(e)[:200]}')
                return False
        return check

    aa = aa_task_score(assertions, safe(initial, 'initial'), safe(final, 'final'))
    state = {'info': {**info, 'assertions': assertions}, 'world': final, 'initial_state': initial_dict}
    try:
        pc = partial_credit(state)
        strict = task_completed_correctly(state)
    except Exception as e:  # noqa: BLE001
        pc = strict = None
        exc.append(f'upstream partial_credit: {type(e).__name__}: {str(e)[:200]}')
    out = {'final_state_score': aa['score'], 'n_objectives': aa['n_objectives'],
           'n_objectives_met': aa['n_objectives_met'], 'n_guardrails': aa['n_guardrails'],
           'n_guardrails_broken': aa['n_guardrails_broken'], 'task_completed': aa['task_completed'],
           'upstream_partial_credit': pc, 'upstream_strict_pass': strict,
           'assertion_results': state.get('_assertion_results')}
    if exc:
        out['grading_exceptions'] = exc
    return out


def session_tokens(session: list | None) -> dict:
    """Prompt and completion tokens summed over the Pi session's assistant messages (`usage`: input + cacheRead,
    output, which includes reasoning); pi_eval's attempt stats carry no totals."""
    if not session:
        return {'input_tokens': None, 'output_tokens': None}
    i = o = 0
    for e in session:
        m = e.get('message') if isinstance(e, dict) else None
        u = m.get('usage') if isinstance(m, dict) and m.get('role') == 'assistant' else None
        if isinstance(u, dict):
            i += int(u.get('input') or 0) + int(u.get('cacheRead') or 0)
            o += int(u.get('output') or 0)
    return {'input_tokens': i, 'output_tokens': o}


def import_row(cfg: dict, task: dict, row: dict, scratch: Path) -> tuple[dict, dict]:
    from aa_evals.common import jsonl
    from aa_evals.common.pi_eval import port
    base = {'item_id': row['item_id'], 'domain': row['item_id'].split('.')[0], 'repeat': row['repeat'],
            'harness': 'pi', 'pi_key': row.get('key'), 'finished_at': time.time()}
    if row['status'] != 'ok':
        rec = {**base, 'status': 'infra_error', 'score': None, 'error': f'infra: {row.get("infrastructure_reason")}'}
        return rec, {'item_id': row['item_id'], 'repeat': row['repeat'], 'status': 'infra_error', 'messages': []}
    d = scratch / row['key']
    got = port.extract(Path(cfg['harness']['pi_run']), row, d, ['/state'])
    world = d / 'artifacts' / 'state' / 'world.json'
    calls_f = d / 'artifacts' / 'state' / 'calls.jsonl'
    calls = jsonl.read(calls_f)
    s = port.stats(row)
    rec = {**base, 'end': port.end_of(row), 'turns': s.get('model_requests') or 0, 'n_tool_calls': len(calls),
           'n_tool_errors': sum(not c.get('ok', True) for c in calls), 'stop_condition': port.end_of(row),
           'max_context_tokens': s.get('max_context_tokens'), **session_tokens(got['session']), 'elapsed_s': got['wall_sec'], 'pi': s, 'n_cap_overshoot': 0,
           **({'session_bad_lines': got['session_bad_lines']} if got.get('session_bad_lines') else {})}
    if not world.is_file():
        rec.update(status='infra_error', score=None, error='no final world collected from the world service')
        g = {}
    else:
        import pydantic
        hidden_f = world.parent / 'hidden.json'
        try:
            g = grade(task, world.read_text(), json.loads(hidden_f.read_text()) if hidden_f.is_file() else None)
        except pydantic.ValidationError as e:
            # #375: the Pi world service accepted a write the WorldState model rejects (Qwen3.6: a dict in a
            # list[str] field). Upstream grades the in-memory world and never re-validates; here the attempt is
            # ungradeable: infra_error, reported, never a crash that stops the collector.
            msg = ('ungradeable: final world fails WorldState validation (the Pi world service accepted an untyped '
                   'write): ' + str(e)[:400])
            print('UNGRADEABLE', row.get('key'), msg.replace('\n', ' | '), flush=True)
            rec.update(status='infra_error', score=None, error=msg, ungradeable=True)
            g = {}
        else:
            rec.update(status='ok', score=g['final_state_score'], error=None,
                       **{k: v for k, v in g.items() if k != 'assertion_results'})
    transcript = {'item_id': row['item_id'], 'repeat': row['repeat'], 'status': rec['status'], 'harness': 'pi',
                  'pi_session': got['session'], 'final_message': port.final_text(got['session']),
                  'tool_calls': calls, 'assertion_results': g.get('assertion_results'), 'pi_archive': got['archive']}
    return rec, transcript


def collect(a) -> dict:
    import shutil
    import tempfile
    from aa_evals.automationbench import run as AB
    from aa_evals.common import fastio
    from aa_evals.common.pi_eval import port
    from aa_evals.tracejudge import fold_summary
    run_dir = Path(a.out_root) / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    tasks = {r['_task_name']: r for r in AB.load_tasks(cfg['dataset']['domains'])}
    items, trs = run_dir / 'items.jsonl', run_dir / 'transcripts.jsonl'
    for f in (trs, items):   # a collector killed mid-append left a torn last line (#378)
        if fastio.trim_torn_tail(f):
            print(time.strftime('%H:%M:%S'), f'{f.name}: removed a torn last line (an interrupted append)', flush=True)
    while True:
        have = {}
        if items.exists():
            for x in items.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    have[(r['item_id'], r['repeat'])] = r.get('pi_key')
        rows = [r for r in port.resolved(Path(cfg['harness']['pi_run']))
                if have.get((r['item_id'], r['repeat'])) != r.get('key')]
        scratch = Path(tempfile.mkdtemp(prefix='ab_pi_'))
        try:
            for r in rows:
                rec, tr = import_row(cfg, tasks[r['item_id']], r, scratch)
                # fsync'd appends, dropped from the page cache: dirty pages of /data writes count against the
                # collector's memory cgroup (#378)
                rec['transcript'] = str(trs)
                rec['transcript_offset'] = trs.stat().st_size if trs.exists() else 0
                fastio.append_line(trs, json.dumps(tr, default=str))
                fastio.append_line(items, json.dumps(rec, default=str))
                print(time.strftime('%H:%M:%S'), 'graded', r['key'], rec['status'], rec.get('score'),
                      rec.get('end'), flush=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        last = {}
        for x in items.read_text().splitlines() if items.exists() else []:
            if x.strip():
                r = json.loads(x)
                last[(r['item_id'], r['repeat'])] = r
        summary = fold_summary(run_dir, AB.summarize(sorted(last.values(), key=lambda r: (r['item_id'], r['repeat'])),
                                                     cfg))
        n_items = cfg['dataset']['n_tasks'] * cfg['repeats']
        summary.update(harness='pi', n_expected=n_items, complete=len(last) >= n_items,
                       ends={e: sum(1 for r in last.values() if r.get('end') == e)
                             for e in ('finish', 'context', 'time_limit')})
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
        print(f'collected {len(last)}/{n_items}; score {summary.get("score")}', flush=True)
        if not a.follow or len(last) >= n_items:
            return summary
        time.sleep(120)


def main(argv=None):
    ensure_venv()
    from aa_evals.automationbench.run import RUNS_ROOT
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
    ap.add_argument('--tasks', default='')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--subset-seed', type=int, default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--aa-name', default=None)
    ap.add_argument('--out-root', default=str(RUNS_ROOT))
    ap.add_argument('--follow', action='store_true')
    a, unknown = ap.parse_known_args(argv)
    if unknown and a.cmd != 'init':
        ap.error(f'unrecognized arguments: {unknown}')
    a.pi_args = unknown + pi_args
    if a.cmd == 'init':
        if not a.model:
            ap.error('--model is required')
        return init(a)
    if a.cmd == 'collect':
        return collect(a)
    cfg = json.loads((Path(a.out_root) / a.run_id / 'config.json').read_text())
    from aa_evals.common.pi_eval import port
    print(json.dumps(port.status(Path(cfg['harness']['pi_run'])), indent=1, default=str))


if __name__ == '__main__':
    main()

"""GDPval-AA under the TB 2.1 Pi harness (#348): generation by aa_evals.common.pi_eval, grading unchanged.

  python -m aa_evals.gdpval.pi.run init    --run-id ID --model serving:NAME [--reasoning-effort E] [--repeats 2] \
      [--task-set text-only-v1 | --task-ids a,b | --limit N] [-- PI_EVAL_ARGS...]
  python -m aa_evals.gdpval.pi.run collect --run-id ID [--follow] [--concurrency 8]   # import + render (CPU job)
  python -m aa_evals.gdpval.pi.run status  --run-id ID
  # then, as for a Stirrup run:
  python -m aa_evals.gdpval.run --run-id ID --stage judge --judge-arm ARM --judge-endpoint SPEC
  python -m aa_evals.gdpval.run --run-id ID --stage score

Two run directories with the same run id:
- the GDPval run, `$AA_RUNS_ROOT/gdpval/<id>` (default /data/aai/aa_evals/runs): config.json (`harness.name` =
  'pi', the pi_eval run path, the bundle), attempts/<task>__r<k>/ (outcome.json, transcript.json.gz,
  deliverables/, render/), matches/, items.jsonl, summary.json: the layout the judge, score, pool and trace-judge
  code read, so grading is the Stirrup path's;
- the pi_eval run, `$PI_RUNS_ROOT/gdpval_pi/<id>` (pi_eval's RUNS_ROOT): its queue, seeds, retries, admission,
  attempts/<key>/trial.tar.gz, exactly as a TB 2.1 run.

`collect` turns each resolved pi_eval attempt into a GDPval attempt: the files in /home/user/outputs when the
episode ended (Harbor artifacts) become deliverables/, the Pi session becomes the transcript, and the end is
`finish` (the agent stopped), `context` (the 131k window was full: no compaction under this protocol), or
`time_limit` (the 2 h environment-clock cap). Every end keeps its deliverables and is judged; an attempt whose
infrastructure retries ran out is `infra_error`, as before. Then render.py renders the deliverables (inside the
SIF), so the judge stage sees the same inputs as for a Stirrup run.

Writes to /data (#378): an attempt is built on node-local scratch (`_CONDOR_SCRATCH_DIR`, else TMPDIR) and its
deliverables, transcript and renders are copied with O_DIRECT (`common.fastio`), so the collector's memory cgroup
never fills with dirty pages (c30n's collectors were held "over memory limit" with a 55-82 MB RSS). outcome.json is
written last and render/ appears complete or not at all, so a collector killed at any point resumes cleanly.
"""
from __future__ import annotations

import os

if __name__ == '__main__':   # a launcher may use the harbor venv: re-execute where the data libraries are
    from aa_evals.common.pi_eval.port import ensure_python
    ensure_python(('pandas', 'pyarrow'), os.environ.get('AA_GDPVAL_PI_PYTHON', '/home/user/python-3.11/bin/python3'), 'aa_evals.gdpval.pi.run')

import argparse
import concurrent.futures as cf
import gzip
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from aa_evals.common import fastio, protocol
from aa_evals.common.pi_eval import port
from aa_evals.gdpval import elo
from aa_evals.gdpval import run as G
from aa_evals.gdpval import runtime_cache as RC
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.network import POLICY as NETWORK_POLICY
from aa_evals.gdpval.pi import tasks as PT

BENCH = 'gdpval_pi'
SEED_NAMESPACE = 'gdpval-pi-v1'
REPEATS = 2


def pi_runs_root() -> Path:
    return port.runs_root()


def bundle_path() -> Path:
    return PT.OUT / f'{PT.BUNDLE_VERSION}.tar'


def pi_eval_args(a, names: list[str], bundle: Path) -> list[str]:
    return port.base_args(BENCH, a.run_id, a.model, bundle, names, a.repeats, SEED_NAMESPACE,
                          a.reasoning_effort) + list(a.pi_args)


def init(a) -> dict:
    run_dir = G.RUNS / a.run_id
    if (run_dir / 'config.json').exists():
        sys.exit(f'{run_dir} exists; use a new --run-id (resume the generation with pi_eval.run --run-id)')
    bundle = Path(a.bundle) if a.bundle else bundle_path()
    manifest = json.loads(bundle.with_suffix('.json').read_text())
    tasks = G.select(a)
    names = [t.task_id for t in tasks]
    if a.task_ids and (unknown := sorted(set(a.task_ids.split(',')) - set(names))):
        sys.exit(f'--task-ids not in --task-set {a.task_set}: {unknown}')
    if not names:
        sys.exit('empty task selection')
    port.launch(pi_eval_args(a, names, bundle))
    pi_run = pi_runs_root() / BENCH / a.run_id
    pcfg = json.loads((pi_run / 'config.json').read_text())
    body = (pcfg.get('request') or {}).get('overrides') or {}
    aa_name, aa_slug = protocol.aa_model(pcfg['served'], body)
    sif = Path(a.sif) if a.sif else G.SIF
    cfg = {
        'benchmark': 'GDPval-AA', 'version': 'AA v2.1 method, Pi harness (#348)', 'run_id': a.run_id,
        'git_sha': G.git_sha(), 'started_at': G.now(), 'argv': sys.argv, 'agent_network': dict(NETWORK_POLICY),
        'model': {'spec': a.model, 'served_name': pcfg['served'], 'body': body, 'effort': protocol.effort_of(body),
                  'vision': (pcfg.get('protocol') or {}).get('model_input') == ['text', 'image'],
                  'vision_rule': 'pi_eval model_input (frozen from the serving launch)',
                  'max_output': pcfg['protocol']['max_output_tokens'], 'aa_name': aa_name,
                  'adapters': (pcfg.get('request') or {}).get('adapters') or [], 'aa_slug': a.aa_slug or aa_slug},
        'context': {'window': pcfg['protocol']['context_window'], 'rule': 'pi_eval --context-window',
                    'policy': pcfg['protocol']['context_policy'], 'summarize_at': None,
                    'finish_notice': pcfg['protocol'].get('finish_notice')},   # #375/#382
        'harness': {'name': 'pi', 'version': pcfg['protocol']['pi_version'], 'label': pcfg.get('harness_label'),
                    'pi_run': str(pi_run), 'pi_eval_code': pcfg.get('code'),
                    'agent_cap_sec': pcfg['agent_cap_sec'], 'max_output_tokens': pcfg['protocol']['max_output_tokens'],
                    'tools': 'Pi 1.2.0 (bash, read, write, edit)', 'deliverables': PT.OUTPUT_DIR,
                    'prompts': 'AA v2.1 minus the finish-tool and step-budget text (pi/tasks.py pi_instruction)',
                    'prompt_disclosures': [PT.END_DISCLOSURE]},
        'bundle': {'path': str(bundle), 'sha256': manifest['sha256'], 'version': manifest['bundle_version'],
                   'base_ref': manifest['base_ref']},
        'sandbox': {'sif': str(sif), 'sif_sha256': PT.SIF_SHA256,
                    'kind': 'harbor_ns sandbox on the SIF root filesystem (pi/tasks.py); render.py in the SIF',
                    'runtime_packages': {'version': 1, 'mode': 'pi-bundle-inputs',
                                         'tasks': sorted(set(names) & RC.REQUIRED_TASKS)}},
        'limits': {'agent_cap_sec': pcfg['agent_cap_sec'], 'max_infra_tries': 7},
        'dataset': {'id': 'openai/gdpval', 'revision': T.REVISION, 'parquet_sha256': T.parquet_sha(),
                    'n_tasks_total': T.N_TASKS},
        'repeats': a.repeats,
        'selection': {'limit': a.limit, 'task_ids': names, **({'task_set': a.task_set} if a.task_set != 'all' else {})},
        'anchor': {'player': elo.EXPERT, 'elo': elo.EXPERT_ELO,
                   'why': "AA v2 human baseline 1000, moved onto the v2.1 scale (-32; #183)"},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for d in ('attempts', 'matches', 'jobs'):
        (run_dir / d).mkdir(exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=1, default=str) + '\n')
    print(f'{run_dir}: {len(names)} tasks x {a.repeats}; generation {pi_run}')
    return cfg


# ------------------------------------------------------------------------------------------------------- collect
def _outputs_check(got: dict) -> str | None:
    """Why the collected outputs can't be trusted as what the episode left (an infrastructure failure), or None.
    Harbor's artifact manifest gives the collection status of the output directory; the placeholder verifier
    (tests/test.sh, run in the sandbox after collection) lists the regular files it saw there (outputs.tsv) and
    rewards 1 iff there was at least one. Both must agree with what was collected."""
    status = got['artifact_status'].get(PT.OUTPUT_DIR)
    reward = port.verifier_reward(got)
    tsv = [ln.split('\t', 1)[1] for ln in (got['verifier'].get('outputs.tsv') or '').splitlines() if '\t' in ln]
    out = Path('artifacts') / PT.OUTPUT_DIR.lstrip('/')
    collected = set(got.get('collected_files') or [])
    if status == 'ok':
        missing = [f for f in tsv if f not in collected]
        if missing:
            return f'artifact collection incomplete: {len(missing)} file(s) the verifier saw, e.g. {missing[:3]}'
        return None
    if status in ('empty', 'failed'):
        if reward == 0 and not tsv:
            return None   # the agent left no files (or removed the directory): a model outcome
        return f'output directory collected as {status!r} but the verifier saw {len(tsv)} file(s) (reward {reward})'
    return f'output directory missing from the artifact manifest (status {status!r}; {out})'


SUPERSEDED = 'superseded'   # matches/superseded/<arm>.jsonl: outside every '*.jsonl' glob of matches/


def _supersede(run_dir: Path, k: str) -> bool:
    """A different pi_eval attempt now resolves this key (e.g. a same-seed retry after a reclassification): drop the
    old render and the old attempt's vs-expert matches, so it is rendered and judged again; old match rows go to
    matches/superseded/<arm>.jsonl. Refused (False: the import waits) while a judge stage is active on the run
    (G.judge_active): it appends to the match file it opened, so rewriting the file under it would lose verdicts.
    Pool (model-vs-model) matches that used the old render are not touched; re-plan a pool after a supersede."""
    if G.judge_active(run_dir):
        return False
    d = run_dir / 'attempts' / k
    for sub in ('render', 'deliverables'):
        shutil.rmtree(d / sub, ignore_errors=True)
    for f in sorted((run_dir / 'matches').glob('*.jsonl')):
        rows = G.read_jsonl(f)
        old = [r for r in rows if r.get('key') == k]
        if not old:
            continue
        (f.parent / SUPERSEDED).mkdir(exist_ok=True)
        with open(f.parent / SUPERSEDED / f.name, 'a') as fh:
            fh.writelines(json.dumps(r) + '\n' for r in old)
        tmp = f.with_name(f.name + f'.partial.{os.getpid()}')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows if r.get('key') != k))
        os.replace(tmp, f)
    return True


def import_attempt(run_dir: Path, cfg: dict, row: dict) -> dict | None:
    """The GDPval attempt of one resolved pi_eval attempt; None when its trial archive can't be read yet (missing
    archive, trial or artifact manifest: retried on the next collect pass, never scored as "no files")."""
    tid, rep = row['item_id'], row['repeat']
    d = run_dir / 'attempts' / G.key(tid, rep)
    base = {'task_id': tid, 'repeat': rep, 'harness': 'pi', 'pi_key': row.get('key'),
            'agent_network': cfg['agent_network'], 'finished_at': time.time()}
    if row['status'] != 'ok':
        d.mkdir(parents=True, exist_ok=True)
        outcome = {**base, 'status': 'infra_error', 'end': 'infra', 'error': row.get('infrastructure_reason'),
                   'turns': 0, 'n_submitted': 0, 'submitted_files': []}
        (d / 'outcome.json').write_text(json.dumps(outcome, indent=1))
        return outcome
    with tempfile.TemporaryDirectory(dir=local_scratch()) as tmp:
        return _import_ok(d, cfg, row, base, Path(tmp))


def local_scratch() -> str | None:
    """Node-local disk for building attempts: the condor job's scratch dir, else TMPDIR (None: tempfile's default)."""
    return os.environ.get('_CONDOR_SCRATCH_DIR') or os.environ.get('TMPDIR') or None


def _write_gz_json(local: Path, dst: Path, obj) -> None:
    with gzip.open(local, 'wt') as f:
        json.dump(obj, f)
    fastio.copy_direct(local, dst)


def _import_ok(d: Path, cfg: dict, row: dict, base: dict, tmp: Path) -> dict | None:
    tid, rep = row['item_id'], row['repeat']
    scratch = tmp / 'import'
    got = port.extract(Path(cfg['harness']['pi_run']), row, scratch, [PT.OUTPUT_DIR])
    if not got['found']:
        if not got['permanent']:
            print(G.now(), 'not imported (retry later):', G.key(tid, rep), got['why'], flush=True)
            return None
        d.mkdir(parents=True, exist_ok=True)   # nothing to retry: an infrastructure failure, never "no files"
        outcome = {**base, 'status': 'infra_error', 'end': 'infra', 'error': f'collection: {got["why"]}',
                   'episode_end': port.end_of(row), 'turns': row.get('model_requests') or 0, 'n_submitted': 0,
                   'submitted_files': [], 'pi': port.stats(row), 'wall_sec': got['wall_sec']}
        (d / 'outcome.json').write_text(json.dumps(outcome, indent=1, default=str))
        return outcome
    d.mkdir(parents=True, exist_ok=True)
    src = scratch / 'artifacts' / PT.OUTPUT_DIR.lstrip('/')
    got['collected_files'] = sorted(str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()) \
        if src.is_dir() else []
    deliv = d / 'deliverables'
    if deliv.exists():
        shutil.rmtree(deliv)
    if got['collected_files']:
        fastio.copy_tree_direct(src, deliv)
    submitted = sorted(str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()) if src.is_dir() else []
    end = port.end_of(row)
    stats = port.stats(row)
    bad = _outputs_check(got)
    events = [{'source': 'harness', 'text': f'attempt end={end} requests={row.get("model_requests")} '
                                            f'submitted={len(submitted)} wall={got["wall_sec"]}s'
                                            + (f' infra: {bad}' if bad else '')},
              {'source': 'pi', 'text': json.dumps(stats, default=str)[:800]}]
    _write_gz_json(tmp / 'transcript.json.gz', d / 'transcript.json.gz',
                   {'harness': 'pi', 'pi_session': got['session'], 'events': events,
                    'final_message': port.final_text(got['session']), 'pi_archive': got['archive']})
    outcome = {**base, 'status': 'infra_error' if bad else 'ok', 'end': 'infra' if bad else end,
               'error': bad or row.get('pi_error'), 'episode_end': end,
               'exception_type': row.get('exception_type'), 'submitted_files': submitted,
               'n_submitted': len(submitted), 'turns': row.get('model_requests') or 0, 'pi': stats,
               'client': {'output_cap_hits': row.get('output_cap_hits') or 0},
               'artifact_status': got['artifact_status'].get(PT.OUTPUT_DIR),
               'verifier_reward': port.verifier_reward(got), 'links_not_collected': got['links'],
               'inputs_receipt': got['inputs_receipt'], 'wall_sec': got['wall_sec'],
               **({'session_bad_lines': got['session_bad_lines']} if got.get('session_bad_lines') else {})}
    (d / 'outcome.json').write_text(json.dumps(outcome, indent=1, default=str))
    return outcome


def render_direct(d: Path, sif: str, code_dir: Path) -> bool:
    """G.render into node-local scratch, then render/ copied to /data with O_DIRECT in one rename (a killed copy
    leaves render.partial, which the next attempt replaces)."""
    with tempfile.TemporaryDirectory(dir=local_scratch()) as tmp:
        ok = G.render(d / 'deliverables', Path(tmp) / 'render', sif, code_dir)
        shutil.rmtree(d / 'render', ignore_errors=True)   # no manifest.json: an older, unfinished render
        fastio.copy_tree_direct(Path(tmp) / 'render', d / 'render')
    return ok


def collect(a) -> dict:
    run_dir = G.RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    pi_run = Path(cfg['harness']['pi_run'])
    code_dir = Path(G.__file__).resolve().parent
    sif = cfg['sandbox']['sif']
    log = lambda *x: print(G.now(), *x, flush=True)  # noqa: E731

    def one(row):
        k = G.key(row['item_id'], row['repeat'])
        d = run_dir / 'attempts' / k
        try:
            prev = json.loads((d / 'outcome.json').read_text())
        except (OSError, ValueError):
            prev = None
        if not prev or prev.get('pi_key') != row.get('key'):
            if prev:
                if not _supersede(run_dir, k):
                    log('supersede waits (a judge stage is active):', k, prev.get('pi_key'), '->', row.get('key'))
                    return
                log('superseded', k, prev.get('pi_key'), '->', row.get('key'))
            o = import_attempt(run_dir, cfg, row)
            if o is None:
                return
            log('imported', k, o['status'], o['end'], f'files={o["n_submitted"]}')
        if G.attempt_state(run_dir, k) == 'render':
            ok = render_direct(d, sif, code_dir)
            log('rendered', k, ok)

    while True:
        rows = port.resolved(pi_run)
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            for f in [ex.submit(one, r) for r in rows]:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    log('collect-error', type(e).__name__, str(e)[:300])
        n_items = len(cfg['selection']['task_ids']) * cfg['repeats']
        done = sum(G.attempt_state(run_dir, G.key(t, r)) == 'done'
                   for t in cfg['selection']['task_ids'] for r in range(cfg['repeats']))
        log(f'collected {done}/{n_items}')
        if not a.follow or done >= n_items:
            return {'done': done, 'items': n_items}
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
    ap.add_argument('--task-set', choices=T.TASK_SETS, default=T.DEFAULT_TASK_SET)
    ap.add_argument('--task-ids', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--sif', default=None)
    ap.add_argument('--aa-slug', default=None)
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--concurrency', type=int, default=8)
    a = ap.parse_args(argv)
    a.pi_args = pi_args
    if a.cmd == 'init':
        if not a.model:
            ap.error('--model is required')
        return init(a)
    if a.cmd == 'collect':
        return collect(a)
    run_dir = G.RUNS / a.run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    G.status(argparse.Namespace(run_id=a.run_id))
    print(json.dumps(port.status(Path(cfg['harness']['pi_run'])),
                     indent=1, default=str))


if __name__ == '__main__':
    main()

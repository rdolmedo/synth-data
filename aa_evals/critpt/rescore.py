"""Re-score every stored CritPt run against a reference, with no model re-runs (#185).

  # one command: re-probe what is stale, rebuild a run-based reference if its probes changed, re-grade every run
  python -m aa_evals.critpt.rescore --reference ds41-max --condor

  # a new reference from answers supplied from elsewhere (e.g. a stronger API model, or an answer key), then re-score
  python -m aa_evals.critpt.rescore --import-answers answers.json --reference NAME --condor

`--import-answers FILE`: JSON {problem_id: answer_code | [answer_code, ...]} (or JSONL rows {"problem_id",
"answer_code"}); ids may omit the `_main` suffix. The answers become run `ref-<NAME>` (one repeat per supplied answer)
and the reference NAME is built from it: every problem whose majority answer is comparable is kept (--min-share 0.5,
--min-samples 1 by default; a single answer per problem is its own majority).

What it does, in order:
1. every run dir under CRITPT_RUNS (or `--runs a,b`): `ok` rows probed before consensus.PROBE_VERSION are re-probed
   in the sandbox (this needs a compute node: `--condor` submits the whole command as one CPU job);
2. a run-based reference whose probe version is older than PROBE_VERSION is rebuilt from the same runs and parameters
   (the old file is kept as <name>.probe<v>.json);
3. every run is graded against the reference (`run.py --grade`, keeping each run's deadline cutoff), which refreshes
   summary.json (headline: all 70 problems) and proxy_<reference>.json (per problem and repeat);
4. a table of all runs goes to stdout and CRITPT_RUNS/_rescore/<reference>.<timestamp>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from . import consensus, data
from . import run as runmod
from aa_evals.common import jsonl

LOG = logging.getLogger('critpt')


def stored_runs(only: str = '') -> list[str]:
    if only:
        return [x for x in only.split(',') if x]
    return sorted(p.name for p in runmod.RUNS.iterdir()
                  if p.is_dir() and not p.name.startswith('_') and (p / 'config.json').exists()
                  and (p / 'items.jsonl').exists())


def stale_rows(run_id: str) -> int:
    states = runmod.final_states(runmod.read_jsonl(runmod.RUNS / run_id / 'items.jsonl'))
    return sum(1 for st in states.values() if st['status'] == 'ok'
               and st.get('probe_version', 1 if 'probe' in st else 0) != consensus.PROBE_VERSION)


def _pid(x: str, known: set[str]) -> str:
    if x in known:
        return x
    if f'{x}_main' in known:
        return f'{x}_main'
    raise SystemExit(f'unknown problem id {x!r}')


def import_answers(path: Path, name: str) -> str:
    """Supplied answers -> run ref-<name> (status ok, not yet probed)."""
    known = {p.problem_id for p in data.load_problems()}
    text = path.read_text()
    try:
        obj = json.loads(text)
        pairs = [(k, v) for k, vs in obj.items() for v in (vs if isinstance(vs, list) else [vs])]
    except json.JSONDecodeError:
        pairs = [(r['problem_id'], r['answer_code']) for r in jsonl.loads(text, where=str(path))]
    by: dict[str, list[str]] = {}
    for k, v in pairs:
        by.setdefault(_pid(k, known), []).append(runmod.harness.strip_fences(v))
    run_id = f'ref-{name}'
    run_dir = runmod.RUNS / run_id
    if (run_dir / 'items.jsonl').exists():
        raise SystemExit(f'{run_dir} exists: pick another --reference name')
    run_dir.mkdir(parents=True)
    repeats = max(len(v) for v in by.values())
    runmod.atomic_json(run_dir / 'config.json', {
        'benchmark': 'critpt-v2', 'run_id': run_id, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'git_sha': runmod.git_sha(), 'imported_from': str(path.resolve()),
        'model': {'spec': f'external:{name}', 'served_name': f'external:{name}', 'urls': []},
        'generation': {'request': {}, 'note': 'answers supplied from outside the harness'},
        'dataset': {**data.check_pins(), 'n_problems_total': data.N_PROBLEMS, 'problems': sorted(by)},
        'repeats': repeats, 'max_attempts': 1, 'judge': None})
    for pid, codes in sorted(by.items()):
        for k, code in enumerate(codes):
            runmod.append_jsonl(run_dir / 'items.jsonl', {
                'item_id': pid, 'repeat': k, 'attempt': 1, 'status': 'ok', 'score': None, 'error': '',
                'answer_code': code, 'turns': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0,
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'imported': True})
    LOG.info('imported %d answers for %d problems into %s', len(pairs), len(by), run_dir)
    return run_id


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--argv-file']:                     # condor job: arguments written by run.submit_condor
        argv = json.loads(Path(argv[1]).read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default=consensus.DEFAULT_REFERENCE)
    ap.add_argument('--runs', default='', help='comma-separated run ids (default: every stored run)')
    ap.add_argument('--import-answers', default=None, help='build reference --reference from these answers first')
    ap.add_argument('--min-share', type=float, default=None, help='reference build (default: 0.5 imported, else the '
                    "reference's own)")
    ap.add_argument('--min-samples', type=int, default=None)
    ap.add_argument('--isolation', default='apptainer-net-none', choices=['apptainer-net-none', 'unshare-inside'])
    ap.add_argument('--condor', action='store_true', help='run the whole command as one CPU condor job')
    ap.add_argument('--cpus', type=int, default=16)
    ap.add_argument('--mem-gb', type=int, default=64)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.condor:
        job_dir = runmod.RUNS / '_rescore'
        runmod.submit_condor(argparse.Namespace(run_id='_rescore', cpus=a.cpus, mem_gb=a.mem_gb), argv,
                             module='aa_evals.critpt.rescore', job_dir=job_dir)
        return

    ref_path = consensus.REFS / f'{a.reference}.json'
    imported = import_answers(Path(a.import_answers), a.reference) if a.import_answers else None
    runs = stored_runs(a.runs)
    if imported and imported not in runs:
        runs.append(imported)
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    ref_runs = ref['runs'] if ref else ([imported] if imported else [])
    # 1. re-probe stale rows (the reference's own runs included)
    need = {r: stale_rows(r) for r in sorted(set(runs) | set(ref_runs))}
    need = {r: n for r, n in need.items() if n}
    if need:
        LOG.info('re-probing (probe version %d): %s', consensus.PROBE_VERSION, need)
        runmod.sandbox_selftest(a.isolation)
        for r in need:
            cfg = json.loads((runmod.RUNS / r / 'config.json').read_text())
            asyncio.run(runmod.reprobe(argparse.Namespace(isolation=a.isolation), runmod.RUNS / r, cfg))
    # 2. (re)build the reference when it is new or its fingerprints are older than the probe
    if ref is None or ref['params'].get('probe_version') != consensus.PROBE_VERSION or imported:
        if not ref_runs:
            raise SystemExit(f'{ref_path} does not exist; pass --import-answers or build it with consensus reference')
        p = (ref or {}).get('params') or {}
        min_share = a.min_share if a.min_share is not None else (0.5 if imported else p.get('min_share', 0.6))
        min_samples = a.min_samples if a.min_samples is not None else (1 if imported else p.get('min_samples', 3))
        if ref is not None:
            old = ref_path.with_name(f'{a.reference}.probe{p.get("probe_version")}.json')
            if not old.exists():
                old.write_text(ref_path.read_text())
        new = consensus.build_reference(ref_runs, min_share=min_share, min_models=p.get('min_models', 1),
                                        rtol=p.get('rtol', consensus.RTOL), min_samples=min_samples)
        new['name'] = a.reference
        if ref is not None:
            new['rebuilt_from'] = {'probe_version': p.get('probe_version'), 'created_at': ref.get('created_at'),
                                   'n_kept': ref.get('n_kept')}
        consensus.REFS.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(new, indent=1))
        LOG.info('reference %s: %d of %d kept (was %s)', a.reference, new['n_kept'], new['n_problems'],
                 (ref or {}).get('n_kept'))
        ref = new
    # 3. grade every run
    table = []
    for r in runs:
        run_dir = runmod.RUNS / r
        prev = json.loads((run_dir / 'summary.json').read_text()) if (run_dir / 'summary.json').exists() else {}
        cut = bool(prev.get('cutoff_applied'))
        try:
            runmod.grade(argparse.Namespace(submit=False, reference=a.reference, cutoff=cut), run_dir)
        except (SystemExit, KeyError, ValueError) as e:
            LOG.warning('%s: not graded (%s)', r, e)
            table.append({'run': r, 'error': str(e)[:200]})
            continue
        s = json.loads((run_dir / 'summary.json').read_text())
        px = s.get('proxy') or {}
        table.append({'run': r, 'model': s.get('served_name'), 'effort': s.get('reasoning_effort'),
                      'repeats': s.get('n_repeats'), 'context_window': s.get('context_window'),
                      'metric': s.get('metric'), 'score': s.get('score'), 'ci95': s.get('ci95'),
                      'score_before': prev.get('score'), 'metric_before': prev.get('metric'),
                      'kept_agreement': px.get('agreement'), 'n_kept': px.get('n_kept'),
                      'n_cutoff': s.get('n_cutoff'), 'valid': s.get('valid'), 'cutoff_applied': cut})
    out = {'reference': a.reference, 'reference_created_at': ref.get('created_at'), 'n_kept': ref.get('n_kept'),
           'probe_version': consensus.PROBE_VERSION, 'git_sha': runmod.git_sha(),
           'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'runs': table}
    (runmod.RUNS / '_rescore').mkdir(exist_ok=True)
    (runmod.RUNS / '_rescore' / f'{a.reference}.{time.strftime("%Y%m%dT%H%M%S")}.json').write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

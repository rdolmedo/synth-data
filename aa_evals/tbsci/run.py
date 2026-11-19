#!/usr/bin/env python3
"""Terminal-Bench-Science 0.1 as AA runs it (issue #29; method, deviations and validation in README.md).

  python -m aa_evals.tbsci.run --model serving:aa-super --run-id smoke-super-v1 --include a,b,c --repeats 1
  python -m aa_evals.tbsci.run --run-id ID --status | --score

Defaults: all 70 tasks of terminal-bench-science v0.1.0 (pinned, Hub-digest-verified, patches.json applied),
stock mini-swe-agent bash-only through the shared runner (aa_evals.common.harbor_mswea), 1,000 steps, 30 s per
command, task.toml timeouts and resources, separate verifier, 3 repeats, pass@1, harbor_ns multi-uid sandboxes
(real distinct UIDs, as e2b has; --no-multiuid for the single-uid runtime). Every option of
aa_evals.common.harbor_mswea.run applies. The run is asynchronous (condor workers); the last worker scores it.

`--score` (or any call once summary.json exists) also writes the TB-Science fields of summary.json:
  score_excl_oracle_failures   pass@1 without the tasks whose reference solution fails in our runtime
  oracle_failing_tasks         those tasks (from validate.py: DATA/validation.json for multi-uid runs,
                               DATA/validation_single_uid.json for --no-multiuid)
  per_domain                   pass@1 per domain (life / physical / mathematical / engineering / earth)
`--aa-slug` defaults from the served model (AA_SLUGS), so summary.json carries aa_reference.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from aa_evals.common.harbor_mswea import endpoint
from aa_evals.common.harbor_mswea.run import ROOT, main
from aa_evals.common import jsonl

DATA = Path('/data/aai/aa_evals/data/tbsci/v0.1.0')
VALIDATION = {True: DATA / 'validation.json', False: DATA / 'validation_single_uid.json'}   # by multi-uid
DATASET = {'name': 'terminal-bench-science/terminal-bench-science', 'tag': 'v0.1', 'git_tag': 'v0.1.0',
           'commit': 'f81afac4f11048e77a15dfc8fb1dbfb897fea0ce',
           'hub_version_hash': 'sha256:91531bf50016a7c64f6cc60794a17c64c6b2c14858a8ae0de39ca16f2abd611a',
           'manifest': str(DATA / 'manifest.json'), 'patches': 'aa_evals/tbsci/patches.json'}
DEFAULTS = {'bench': 'tbsci', 'tasks': str(DATA / 'tasks.tar'), 'repeats': 3, 'step_limit': 1000,
            'aa_metric_key': 'terminal-bench-science',
            'metric': 'Terminal-Bench-Science 0.1 pass@1 (mean of 3 repeats)', 'n_tasks_benchmark': 70,
            'dataset': DATASET, 'multiuid': True}
# served model -> the AA leaderboard entry whose reasoning setting the run mirrors
AA_SLUGS = {
    'nemotron-3.5-super-vl': 'nemotron-3-5-super-vl',
    'deepseek-v41-flash-1017': 'deepseek-v4-1-flash-1017',   # (Max), the serving spec's default effort
    'glm-5.3-flash': 'glm-5-3-flash',
    'qwen38-27b': 'qwen3-8-27b',                                # (Xhigh), aa-qwen's default reasoning_effort
}


def oracle_failing(validation: Path) -> list[str]:
    if not validation.exists():
        return []
    val = json.loads(validation.read_text())
    # 'infra': the oracle never got a reward (traffic-flux-inversion's verifier kills itself in single-uid mode)
    return sorted(t for t, v in val.items() if v['verdict'] in ('oracle_fails', 'oracle_flaky', 'infra'))


def pass_at_1(items: list[dict]) -> tuple[float | None, int]:
    per = defaultdict(list)
    for r in items:
        if r['status'] in ('ok', 'model_error') and r.get('score') is not None:
            per[r['item_id']].append(float(r['score']))
    if not per:
        return None, 0
    return round(100 * sum(sum(v) / len(v) for v in per.values()) / len(per), 2), len(per)


def domain_of(m: dict) -> str:
    return m.get('path', 'tasks/unknown/').split('/')[1]


def finalize(run: Path, data: Path = DATA, validation: dict = VALIDATION, domain=domain_of) -> dict | None:
    """Add the TB-Science fields to the shared scorer's summary.json (also used by aa_evals.tbsci_lite)."""
    if not (run / 'summary.json').exists():
        return None
    items = jsonl.read(run / 'items.jsonl', required=True)
    s = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())['tasks']
    val = validation[bool(json.loads((run / 'config.json').read_text()).get('multiuid'))]
    failing = oracle_failing(val)
    excl, n_excl = pass_at_1([r for r in items if r['item_id'] not in failing])
    dom = defaultdict(list)
    for r in items:
        dom[domain(manifest.get(r['item_id'], {}))].append(r)
    s.update({
        'score_excl_oracle_failures': excl,
        'n_items_excl_oracle_failures': n_excl,
        'oracle_failing_tasks': failing,
        'oracle_validation': str(val) if val.exists() else None,
        'per_domain': {d: dict(zip(('score', 'n_items'), pass_at_1(rs))) for d, rs in sorted(dom.items())},
        'patched_tasks': sorted(t for t, m in manifest.items() if m.get('patches')),
    })
    s.pop('per_category', None)   # task.toml has no metadata.category here; per_domain replaces it
    (run / 'summary.json').write_text(json.dumps(s, indent=1))
    return s


def default_slug(argv: list[str]) -> list[str]:
    """Add --aa-slug for a known served model unless given."""
    if '--aa-slug' in argv or '--model' not in argv:
        return argv
    spec = argv[argv.index('--model') + 1]
    try:
        _, ep, _ = endpoint.alive(spec)
        slug = AA_SLUGS.get(ep.get('served') or '')
    except Exception:  # noqa: BLE001  (the runner reports an unusable model itself)
        slug = None
    return argv + ['--aa-slug', slug] if slug else argv


def cli(argv: list[str], defaults: dict = DEFAULTS, fin=finalize) -> None:
    argv = default_slug(list(argv))
    main(argv, defaults)
    if '--run-id' in argv:
        s = fin(ROOT / 'runs' / defaults['bench'] / argv[argv.index('--run-id') + 1])
        if s is not None and ('--score' in argv or '--status' not in argv):
            print(json.dumps({k: s.get(k) for k in ('score', 'ci95', 'n_items', 'n_infra_errors', 'valid',
                                                    'score_excl_oracle_failures', 'aa_reference')}))


if __name__ == '__main__':
    cli(sys.argv[1:])

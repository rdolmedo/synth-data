#!/usr/bin/env python3
"""TB-Science-lite (#68) under our Pi protocol (user decision 2026-11-02 ~23:40, #66): the headline harness.

  python -m aa_evals.tbsci_lite.pi --model serving:aa-qwen-any --reasoning-effort medium --run-id r1-medium --include a,b
  python -m aa_evals.tbsci_lite.pi --run-id ID --status | --score | --stop | --resubmit N
  python -m aa_evals.tbsci_lite.pi --pool v2 ...       # pool v2 (audit #68, 2026-11-05; README "Pool v2")

Same as aa_evals.tbsci.pi on the pinned lite pool (fetch.py: DATA/tasks.tar, agent timeouts already capped at 2 h),
seed namespace aa-tbsci-lite-v1, at most 32 in flight.  No AA reference.
"""
import json
import sys
from pathlib import Path

from aa_evals.tbsci import pi
from aa_evals.tbsci_lite.fetch import POOL_SHA256

def _pool(argv: list[str]) -> str:
    """`--pool v1|v2|v2.1` (default v1), removed from argv before pi_eval parses it."""
    for i, x in enumerate(argv):
        if x == '--pool' and i + 1 < len(argv):
            v = argv[i + 1]; del argv[i:i + 2]; return v
        if x.startswith('--pool='):
            v = x.split('=', 1)[1]; del argv[i]; return v
    return 'v1'


POOL = _pool(sys.argv)
if POOL not in ('v1', 'v2', 'v2.1'):
    raise SystemExit(f'--pool must be v1, v2 or v2.1, not {POOL}')
DATA = Path(f'/data/aai/aa_evals/data/tbsci_lite/{POOL}')


def domains(_bundle) -> dict:
    return {t: m['domain'] for t, m in json.loads((DATA / 'manifest.json').read_text())['tasks'].items()}


ORACLE_RUN = '/data/aai/aa_evals/runs_fallback/tbsci_lite/oracle-pi-v1'


def headline_tasks() -> str:
    """The 125 tasks whose reference solution passes on the Pi path (ORACLE_RUN, all 141 x 1; README "Validity"):
    the set of every reported lite result since round 3. `--include all` runs the whole pool."""
    return ','.join(sorted(t for t, v in pi.oracle_verdicts(ORACLE_RUN).items() if v == 'pass'))


TASKS_2H = '/data/aai/aa_evals/runs_fallback/data/tbsci_lite/tasks_2h.tar'   # cap.py on DATA/tasks.tar
DEFAULTS = {**pi.DEFAULTS, 'bench': 'tbsci_lite', 'tasks_tar': TASKS_2H,
            'seed_namespace': 'aa-tbsci-lite-v1', 'max_in_flight': 32, 'aa_metric_key': None,
            'task_agent_classes': {'unknown-payload-balancing': 'PiCont16kEnvClockMuslCompat'},
            'attempts': 5,   # M3.0 (#173, 11-04 update): 5 repeats
            'categories_fn': domains,
            'validation': [DATA / 'validation.json'], 'pool': {'n_tasks': 141, 'name': 'TB-Science-lite pool v1'},
            'oracle_run': ORACLE_RUN, 'include': headline_tasks(),
            # deterministic environment defects on the Pi path (audit 2); their Pi-path oracle passes
            'excluded_tasks': {
                'ra-taper-risk': 'verifier_no_reward: agent setup fails with PermissionError writing '
                                 '/tmp/harbor-pi-agent/models.json in the sandbox (all 7 attempts of every run)',
                'refinement-ordered-optimization': 'pi_backend_unreachable: Pi\'s first request fails with "Connection '
                                                   'error." within 30 ms and the proxy logs no request (all 7 attempts of every run)'},
            'notes': 'Pi protocol (user decision 2026-11-02 ~23:40, #66); 2 h agent cap (environment time); '
                     'TB-Science-lite pool v1 (aa_evals/tbsci_lite/README.md)'}

# Pool v2 (audit #68, 2026-11-05): v1 + the audit's patch set (v2_changelog.json), run on the harbor_ns multi-uid
# fixes (CPU view, id bands, owner replay, ...). Scored set: the frozen validation manifest's retained tasks,
# each with three oracle passes and a no-op failure, minus the audit's exclusions. v1's two
# excluded tasks are fixed in v2 (refinement-ordered-optimization: patch; ra-taper-risk: harbor_ns uploads).
TASKS_2H_V2 = '/data/aai/aa_evals/runs_fallback/data/tbsci_lite/tasks_2h_v2.tar'   # cap.py on v2 DATA/tasks.tar
ORACLE_RUN_V2 = '/data/aai/aa_evals/runs_fallback/tbsci_lite/oracle-v2-x3'


def headline_tasks_v2(pool: str = 'v2') -> str:
    """Require the completed x3/no-op gate; never infer validity from a partial or later run."""
    validation = json.loads(Path(__file__).with_name(f'{pool.replace(".", "_")}_validation.json').read_text())
    if validation['pool_sha256'] != POOL_SHA256[pool]:
        raise ValueError(f'frozen {pool} validation does not match the pinned pool')
    kept = set(validation['retained_tasks'])
    tasks = validation['tasks']
    if len(tasks) != 141 or len(kept) != validation['n_retained']:
        raise ValueError('inconsistent frozen v2 task counts')
    expected = {t for t, row in tasks.items() if row['retained']}
    if kept != expected or kept & set(validation['audit_exclusions']):
        raise ValueError('inconsistent frozen v2 retained tasks')
    for t in kept:
        if tasks[t]['oracle'] != {'attempts': 3, 'pass': 3, 'fail': 0, 'infra': 0, 'pending': 0}:
            raise ValueError(f'{t}: incomplete frozen oracle validation')
        if tasks[t]['noop'] != {'attempts': 1, 'pass': 0, 'fail': 1, 'infra': 0, 'pending': 0}:
            raise ValueError(f'{t}: incomplete frozen no-op validation')
    return ','.join(sorted(kept))


if POOL in ('v2', 'v2.1'):
    tasks_tar = TASKS_2H_V2 if POOL == 'v2' else '/data/aai/aa_evals/data/tbsci_lite/v2.1/tasks_2h.tar'
    oracle_run = ORACLE_RUN_V2 if POOL == 'v2' else (
        ORACLE_RUN_V2 + ',/data/aai/aa_evals/runs_fallback/tbsci_lite/audit-nonnegative-v21r2-oracle-x3')
    DEFAULTS = {**DEFAULTS, 'tasks_tar': tasks_tar, 'oracle_run': oracle_run, 'include': headline_tasks_v2(POOL),
                'pool': {'n_tasks': 141, 'name': f'TB-Science-lite pool {POOL}'}, 'excluded_tasks': {},
                'validation': [DATA / 'validation.json'],
                'notes': DEFAULTS['notes'].replace('pool v1', f'pool {POOL} (audit #68; frozen validation manifest)')}

if __name__ == '__main__':
    pi.cli(sys.argv[1:], DEFAULTS)

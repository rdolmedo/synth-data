#!/usr/bin/env python3
"""TB-Science-lite (issue #68): our in-house pool of 141 `lite` TB-Science tasks, run with TB-Science 0.1's method.

  python -m aa_evals.tbsci_lite.run --model serving:aa-qwen --run-id r1-qwen --include a,b,c --repeats 1
  python -m aa_evals.tbsci_lite.run --run-id ID --status | --score

Same as aa_evals.tbsci.run (stock mini-swe-agent bash-only through the shared runner, 1,000 steps, 30 s per
command, task.toml resources and verifier, pass@1, harbor_ns multi-uid sandboxes) on the pinned pool
(fetch.py: DATA/tasks.tar), except the agent timeout, capped at 2 h (README: Agent time).  There is no AA
reference.  summary.json gets per_domain, score_excl_oracle_failures and oracle_failing_tasks (DATA/validation.json,
from `aa_evals/tbsci/validate.py --bench tbsci_lite`).
"""
import sys
from pathlib import Path

from aa_evals.tbsci import run as tbsci

DATA = Path('/data/aai/aa_evals/data/tbsci_lite/v1')
VALIDATION = {True: DATA / 'validation.json', False: DATA / 'validation_single_uid.json'}
DATASET = {'name': 'tbsci-lite pool v1 (open `lite` PRs of harbor-framework/terminal-bench-science, pinned heads)',
           'pool_tar': '/data/aai/tbench/tbsci_lite/tbsci_lite_pool_v1.tar',
           'pool_sha256': 'ddaa18f089bdcab8c7f56604f713efd956d8883fa6caf2a7fd2d1d503421a8b7',
           'manifest': str(DATA / 'manifest.json'), 'agent_cap_sec': 7200}
DEFAULTS = {'bench': 'tbsci_lite', 'tasks': str(DATA / 'tasks.tar'), 'repeats': 3, 'step_limit': 1000,
            'metric': 'TB-Science-lite v1 pass@1', 'n_tasks_benchmark': 141, 'dataset': DATASET, 'multiuid': True}


def finalize(run: Path) -> dict | None:
    return tbsci.finalize(run, DATA, VALIDATION, domain=lambda m: m.get('domain', 'unknown'))


if __name__ == '__main__':
    tbsci.cli(sys.argv[1:], DEFAULTS, finalize)

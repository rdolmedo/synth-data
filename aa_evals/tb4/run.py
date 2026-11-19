#!/usr/bin/env python3
"""Terminal-Bench 4.0 as AA runs it (issue #28; method and deviations in README.md).

  python -m aa_evals.tb4.run --model serving:aa-super --run-id smoke-super-v1 --include a,b,c --repeats 1
  python -m aa_evals.tb4.run --run-id ID --status | --score

Defaults: the 63 CPU tasks of terminal-bench/terminal-bench@4.0.0 (the 3 H100 tasks excluded, user decision
2026-11-02; summary.json also reports the 66-task lower bound), stock mini-swe-agent 2.4.6, bash only,
500 steps, 3 repeats, pass@1. Every option of aa_evals.common.harbor_mswea.run applies.
"""
import sys

from aa_evals.common.harbor_mswea.run import main

DATASET = {'name': 'terminal-bench/terminal-bench', 'tag': '4.0.0', 'revision': 4,
           'content_hash': 'sha256:39d9f44b40420cde',
           'downloaded_with': 'harbor download terminal-bench/terminal-bench@4.0.0 (Harbor 0.24.0), 2026-11-02'}
TASKS = '/data/aai/aa_evals/data/tb4/tasks_4.0.0/terminal-bench'
GPU_TASKS = ['fp8-rmsnorm-gemm', 'jax-speedrun-gpu', 'math-eval-grader']   # need an H100; harbor_ns has no GPUs

DEFAULTS = {'bench': 'tb4', 'tasks': TASKS, 'repeats': 3, 'step_limit': 500, 'exclude': GPU_TASKS,
            'aa_metric_key': 'terminalbench-4-0', 'metric': 'Terminal-Bench 4.0 pass@1 (mean of 3 repeats)',
            'n_tasks_benchmark': 66, 'dataset': DATASET,
            # oracle validation of the 63 tasks (README); later runs re-validate tasks after runtime fixes
            'oracle_run': ','.join(f'/data/aai/aa_evals/runs/tb4/{r}' for r in
                                   ('oracle-v1', 'oracle-netns-v1', 'oracle-ks-x3-v1', 'oracle-v3-payments'))}

if __name__ == '__main__':
    main(sys.argv[1:], DEFAULTS)

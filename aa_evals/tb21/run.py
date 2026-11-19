#!/usr/bin/env python3
"""Terminal-Bench 2.1 under our Pi protocol (issue #67; method, AA's own method and deviations in README.md).

  python -m aa_evals.tb21.run --model serving:aa-qwen --reasoning-effort medium --run-id r1-medium --include a,b,...
  python -m aa_evals.tb21.run --agent oracle --run-id oracle-v1
  python -m aa_evals.tb21.run --run-id ID --status | --score | --stop | --resubmit N

Defaults: the 89 tasks of eval_q35's six-hour bundle tb21_6h.tar (the bundle of every past tb21p180 / tbmix eval;
same task content as terminal-bench/terminal-bench-2-1 rev 6, README), seed namespace q35-tb21-v1 (the past runs'
seeds), 3 attempts, Pi 1.2.0 pi180, at most 48 attempts in flight, harbor_ns single-uid (as past runs). Every option of
aa_evals.common.pi_eval.run applies.
"""
import sys
import tarfile
import tomllib
from pathlib import Path

from aa_evals.common.pi_eval.run import main

BUNDLE = '/data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar'   # sha256 363d5b93...
DATASET = {'name': 'terminal-bench/terminal-bench-2-1', 'revision': 6,
           'content_hash': 'sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a',
           'downloaded_to': '/data/aai/aa_evals/data/tb21/tasks_rev6/terminal-bench-2-1',
           'equals': '/data/aai/tbench/eval_tb21.tar except .gitignore files and sanitize-git-repo/tests/'
                     'test_outputs.py, whose fake secrets are written as concatenated literals there'}


def categories(bundle: Path) -> dict:
    out = {}
    with tarfile.open(bundle) as tf:
        for m in tf.getmembers():
            p = Path(m.name).parts
            if len(p) == 3 and p[2] == 'task.toml':
                cfg = tomllib.loads(tf.extractfile(m).read().decode())
                out[p[1]] = (cfg.get('metadata') or {}).get('category', '?')
    return out


DEFAULTS = {'bench': 'tb21', 'bundle': BUNDLE, 'seed_namespace': 'q35-tb21-v1', 'attempts': 5,   # M3.0: 5 repeats
            'max_in_flight': 48,
            # M3.5 (user, 2026-11-05): 2 h of environment time per task (was 6 h), wall backstop 2 x 2 h = 4 h, as
            # TB-Sci-lite; uniform for every task (no --per-task-cap). M3.4 Qwen medium: 9/444 attempts over 2 h wall
            'agent_cap_sec': 7200.0,
            # single-uid, as every past tb21p180 run: the oracle passes 86/89 single-uid vs 82/89 multi-uid, and
            # multi-uid fixes no TB 2.1 task (README, oracle validation)
            'multiuid': False, 'aa_metric_key': 'terminalbench-2-1',
            # #405 (coordinator, 2026-11-18): eval's per-task disk caps, one rule with eval_q35 (DISKCAP_FAMILIES['tb2.1']:
            # rstan-to-pystan 90 GiB, one per worker, since passing boxes reached 55 GiB on eval (#425); sam-cell-seg
            # 40 GiB; others 32 GiB), --diskcap auto (the default) writes them to config.json
            'diskcap_family': 'tb2.1',
            # per-sandbox memory cap (harbor_ns memcap watchdog), as TB-Sci-lite (user, 2026-11-13, #378): above 1.5 x
            # the task's declared memory_mb (2-8 GB) the sandbox's largest process is killed, so one runaway task can
            # no longer get the whole worker job held (every attempt on it lost). Before: no cap (c30 TB workers
            # reached ~100 GB RSS on 16 slots). The official harness enforces 1.0x through Docker's mem_limit.
            'harbor_env': {'HARBOR_NS_MEMCAP': '1.5'},
            'categories_fn': categories,
            # reference-solution validation, single-uid (README): the last run with a task decides
            'oracle_run': ','.join(f'/data/aai/aa_evals/runs_fallback/tb21/{r}'
                                   for r in ('oracle-fail-nomu', 'oracle-v1-nomu')),
            'notes': 'Pi protocol (user decision 2026-11-02 ~23:40, #66), not AA\'s Terminus 2 harness; dataset '
                     + str(DATASET)}

if __name__ == '__main__':
    main(sys.argv[1:], DEFAULTS)

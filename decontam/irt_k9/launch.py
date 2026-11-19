#!/usr/bin/env python3
"""User 2026-10-28: k=9 on the ~1/3 most discriminative tasks (2PL IRT, irt_select.py) of TB2.1 and TB-Lite, for the
five runs of the CalibForge decontamination contrasts. Starts each eval the way orchestrate.process_requests does
(submit.py init / adaptive control / serve / workers, same code snapshot and protocol: pi180, 2x H100 TP2), plus two
changes: --tasks = the IRT subset, and a fresh seed namespace so attempts 0-8 never replay the seeds of the k=3 evals
(seeds are per (namespace, task, attempt) and identical across models, so the five runs stay paired).
The ops loop then manages these runs like any other directory under eval_q35/ (workers, KV limits, tail cuts, scoring);
babysit.py restarts dead servers, which the ops loop does only for requests.json runs. Idempotent.
"""
import json
import sys
from pathlib import Path

EVAL = '/opt/aai/worktrees/eval-super-scaling/eval_q35'
sys.path.insert(0, EVAL)
from common import ROOT, atomic, now, read  # noqa: E402
from orchestrate import GPU_BID, PY, SLOTS, adaptive_params, sh  # noqa: E402

D = Path(__file__).resolve().parent
RUNS = ['super-subset-4b-calibforge-decon-1b', 'super-subset-4b-calibforge-1b-fastpath', 'super-subset-4b-calibforge-1b',
        'super-subset-4b-tmax-cfcontam-1b', 'super-subset-4b-tmax-cfclean-1b']
BENCH = {'tb2.1': ('tb21p180', 'tasks_tb21.txt', 'q35-tb21-irt33k9-v1'),
         'tblite': ('tblp180', 'tasks_tblite.txt', 'q35-tblite-irt33k9-v1')}
ATTEMPTS = 9


def names():
    return [f'{pre}irt-{r}' for pre, _, _ in BENCH.values() for r in RUNS]


def main(dry):
    reqs = {r['name']: r for r in read(ROOT / 'requests.json')['requests']}
    for bench, (pre, task_file, ns) in BENCH.items():
        tasks = (D / task_file).read_text().split()
        for run in RUNS:
            ref = reqs[f'{pre}-{run}']   # the k=3 eval's recorded request: same model, template, harness
            assert ref['benchmark'] == bench and ref['harness'] == 'pi180', ref
            name = f'{pre}irt-{run}'
            if (ROOT / name / 'config.json').exists():
                print(f'{name}: exists', flush=True)
                continue
            model = ref['model']
            sh(PY, 'submit.py', 'init', '--name', name, '--model', model, '--family', 'qwen3_5',
               '--chat-template', ref['chat_template'], '--benchmark', bench, '--harness', 'pi180', '--serve-gpu', 'h100x2',
               '--attempts', ATTEMPTS, '--api-server-count', 4, '--tasks', ','.join(tasks), dry=dry)
            if dry:
                continue
            params, gate = adaptive_params(model, 'qwen3_5', 'h100x2')
            atomic(ROOT / name / 'control.json', {'adaptive': True, 'adaptive_owner': 'admission', **params})
            cfg = read(ROOT / name / 'config.json')
            atomic(ROOT / name / 'config.json', {
                **cfg, 'kv_gate': gate, 'seed_namespace': ns, 'seed_namespace_benchmark_default': cfg['seed_namespace'],
                'irt_subset': {'tasks': str(D / task_file), 'selection': str(D / f'irt_{bench}.json'), 'reference_run': ref['name'],
                               'note': 'user 2026-10-28: k=9 on the IRT top-1/3 tasks, fresh seed namespace', 'at': now()}})
            sh(PY, 'submit.py', 'serve', '--name', name, '--gpu-bid', ref.get('gpu_bid', GPU_BID))
            sh(PY, 'submit.py', 'workers', '--name', name, '--n', 2, '--slots', SLOTS, '--cpus', 16, '--mem-gb', 320,
               '--disk-gb', 200)


if __name__ == '__main__':
    main('--dry-run' in sys.argv)

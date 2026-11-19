"""Trace-path smoke: one pi_eval run (the traces.py protocol) over a tasks dir on one Super engine.

  python -m m5.tbsci_lite.job --name smoke -- m5.tbsci_lite.smoke --run-id ID --tasks DIR --engine m5-super-1
"""
import argparse

from . import traces as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--engine', default='m5-super-1')
    ap.add_argument('--max-in-flight', type=int, default=2)
    a = ap.parse_args()
    from aa_evals.common.harbor_mswea import run as hm
    from aa_evals.common.pi_eval import run as pe
    from .common import bad_nodes
    hm.EXCLUDE_NODES = tuple(sorted(set(hm.EXCLUDE_NODES) | set(bad_nodes())))
    pe.main(['--run-id', a.run_id, '--model', f'serving:{a.engine}', '--tasks-tar', a.tasks,
             '--max-in-flight', str(a.max_in_flight), '--attempts', '1', *T.WORKER], T.DEFAULTS)


if __name__ == '__main__':
    main()

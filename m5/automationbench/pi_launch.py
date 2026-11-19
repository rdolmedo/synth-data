"""pi_eval.run with the worker submits excluding nodes.excluded() as well as pi_eval's own EXCLUDE_NODES.

  python -m m5.automationbench.pi_launch PI_EVAL_ARGS...   (harbor venv)
"""
import sys

from aa_evals.common.pi_eval import run as R
from m5.automationbench.nodes import excluded

R.EXCLUDE_NODES = tuple(sorted(set(R.EXCLUDE_NODES) | set(excluded())))

if __name__ == '__main__':
    R.main(sys.argv[1:])

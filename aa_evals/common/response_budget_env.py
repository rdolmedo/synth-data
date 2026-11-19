"""PI_RESPONSE_BUDGET (tb_agents/response_budget.py) into the HTCondor jobs of the single-turn suites (capped protocol).

The campaign launcher (aa_evals/campaign/launch.py, config key `capped`) sets PI_RESPONSE_BUDGET in the environment
of the submitter it runs; hle/submit.py, omniscience/submit.sh, gdppdf/submit.py and scicode/run.py --condor forward
it into the job's `environment` (the jobs run with getenv = false), only when it is set: unset, their submit files are
unchanged. Before forwarding, the job's interpreter is checked on the frozen code (`preflight`): a budget the job
could not apply fails at submission, not in every item of a queued job.

  python -m aa_evals.common.response_budget_env condor-env PYTHON CODE   # " PI_RESPONSE_BUDGET='...'" or nothing
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ENV = 'PI_RESPONSE_BUDGET'
# What a budgeted single-turn request needs in the job (tb_agents/response_budget_single.py): the adapter, the
# Snowball template + tokenizer (retained bodies), and Super's tokenizer for token_counting 'super'.
PROBE = ('from tb_agents import response_budget as B, response_budget_single; p = B.settings(); B.template(); '
         'B.tokenizer(); p.get("token_counting") == "super" and B.super_tokenizer()')


def value(env=None) -> str | None:
    """The submitter's PI_RESPONSE_BUDGET, validated by response_budget.settings and re-serialized compactly (no
    newline can reach a submit file); None when unset or empty."""
    env = os.environ if env is None else env
    if not env.get(ENV):
        return None
    from tb_agents import response_budget
    v = json.dumps(response_budget.settings({ENV: env[ENV]}), separators=(',', ':'))
    if '$(' in v:   # condor would expand it as a submit macro
        raise SystemExit(f'{ENV} contains "$(": it cannot go into a condor environment unchanged')
    return v


def condor_quote(v: str) -> str:
    """One value in condor's environment syntax (inside its outer double quotes): single-quoted, ' and " doubled."""
    return "'" + str(v).replace("'", "''").replace('"', '""') + "'"


def condor_env(env: dict) -> str:
    """name='value' pairs for `environment = "..."`."""
    return ' '.join(f'{k}={condor_quote(v)}' for k, v in env.items())


def preflight(python: str, code, v: str) -> None:
    """The job's interpreter can apply budget `v` from the frozen code dir `code` (as the job runs: PYTHONPATH=code,
    no user site), else SystemExit."""
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': os.environ.get('HOME', ''), 'PYTHONPATH': str(code),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', ENV: v}
    try:
        r = subprocess.run([python, '-c', PROBE], cwd=str(code), env=env, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f'{ENV} is set, but the job interpreter {python} could not be checked: {e}')
    if r.returncode:
        last = ((r.stderr or r.stdout).strip().splitlines() or ['?'])[-1]
        raise SystemExit(f'{ENV} is set, but the job interpreter {python} cannot apply it from {code}: {last}')


def forwarded(python: str | None = None, code=None, env=None) -> dict:
    """{PI_RESPONSE_BUDGET: compact JSON} for the job's environment when the submitter has it set (after the
    preflight, when `python` is given); {} otherwise."""
    v = value(env)
    if v is None:
        return {}
    if python:
        preflight(python, code, v)
    return {ENV: v}


if __name__ == '__main__':
    if sys.argv[1:2] != ['condor-env'] or len(sys.argv) != 4:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    fw = forwarded(sys.argv[2], sys.argv[3])
    if fw:
        print(' ' + condor_env(fw))

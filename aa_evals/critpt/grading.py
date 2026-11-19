"""Submissions and grading.

The CritPt authors' server, run with AA, grades through `POST https://artificialanalysis.ai/api/v2/critpt/evaluate`
(header `x-api-key`). It accepts only complete batches covering all 70 problems, returns only aggregates
(`accuracy`, `timeout_rate`, `server_timeout_count`, `judge_error_count`), and allows 10 requests per 24 h. Access
also needs CritPt approval from AA (403 without it). One repeat = one batch.

`generated_code` is the answer code in a single fenced ```python block: the authors' own pipeline submitted the
model's whole markdown reply, so the server extracts code blocks, and one block leaves it no choice to make.
Problems a partial run did not attempt, and model failures, are submitted blank (they score zero).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .data import N_PROBLEMS, Problem

API_URL = 'https://artificialanalysis.ai/api/v2/critpt/evaluate'
KEY_FILE = Path('/data/aai/aa_evals/secrets/aa_api_key')


def generated_code(answer_code: str) -> str:
    return f'```python\n{answer_code.strip()}\n```\n' if answer_code.strip() else ''


def build_batch(problems: list[Problem], answers: dict[str, str], *, model: str, generation_config: dict,
                batch_metadata: dict) -> dict:
    """The request body for one repeat: all 70 problems, blanks where there is no answer."""
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    subs = [{'problem_id': p.problem_id, 'generated_code': generated_code(answers.get(p.problem_id, '')),
             'model': model, 'timestamp': ts, 'generation_config': generation_config} for p in problems]
    body = {'submissions': subs, 'batch_metadata': batch_metadata}
    validate_batch(body, problems)
    return body


def validate_batch(body: dict, problems: list[Problem]) -> None:
    """What the server checks before grading (all 70 ids, none extra) plus the field schema."""
    ids = [s['problem_id'] for s in body['submissions']]
    want = {p.problem_id for p in problems}
    if len(ids) != N_PROBLEMS or set(ids) != want or len(set(ids)) != len(ids):
        raise ValueError(f'batch must hold each of the {N_PROBLEMS} problem ids exactly once')
    for s in body['submissions']:
        for k, t in (('problem_id', str), ('generated_code', str), ('model', str), ('generation_config', dict)):
            if not isinstance(s.get(k), t):
                raise ValueError(f'{s.get("problem_id")}: field {k} must be {t.__name__}')
    if not isinstance(body.get('batch_metadata'), dict):
        raise ValueError('batch_metadata must be an object')
    json.dumps(body)


def api_key() -> str | None:
    key = os.environ.get('AA_API_KEY')
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    return key or None


def submit_batch(body: dict, *, key: str, timeout: float = 3600.0) -> dict[str, Any]:
    """POST one batch. Returns {'http_status', 'response', 'rate_limit'}; never retries a request the server may have
    counted, except after a 429 (which is not graded)."""
    import httpx
    headers = {'x-api-key': key, 'Content-Type': 'application/json'}
    for attempt in range(3):
        r = httpx.post(API_URL, headers=headers, json=body, timeout=timeout)
        rate = {k: r.headers.get(k) for k in ('X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset',
                                              'Retry-After')}
        try:
            payload = r.json()
        except ValueError:
            payload = {'text': r.text[:2000]}
        if r.status_code == 429 and attempt < 2:
            wait = int(r.headers.get('Retry-After') or 60)
            if wait > 900:
                return {'http_status': r.status_code, 'response': payload, 'rate_limit': rate}
            time.sleep(wait)
            continue
        return {'http_status': r.status_code, 'response': payload, 'rate_limit': rate}
    raise AssertionError


def check_example(answer_code: str, golden_code: str, run_python, *, timeout: int = 120) -> dict:
    """Grade the example challenge locally: run the submitted answer and compare it with the golden SymPy expression.

    `run_python(program, timeout)` runs a Python program in the offline sandbox (the answer is model-written code) and
    returns a CompletedProcess. This checks the format path (template filled, code extracted, function returns a
    SymPy expression), not the authors' grader.
    """
    import ast
    import subprocess
    golden_name = ast.parse(golden_code).body[0].targets[0].id     # e.g. F_logical = <expr>
    prog = f'''
import sympy as sp
ns = {{}}
exec(compile({answer_code!r}, "answer", "exec"), ns)
p = sp.symbols("p")
got = sp.sympify(ns["answer"](p))
gns = {{"sp": sp, "p": p}}
exec({golden_code!r}, gns)
gold = gns[{golden_name!r}]
diff = sp.simplify(sp.together(got - gold))
num = all(abs(complex(sp.N((got - gold).subs(p, v)))) < 1e-9 for v in (0.01, 0.1, 0.3))
print("CORRECT" if (diff == 0 or num) else "WRONG", str(got)[:300])
'''
    try:
        r = run_python(prog, timeout)
    except subprocess.TimeoutExpired:
        return {'executes': False, 'correct': False, 'detail': 'timeout'}
    out = (r.stdout or '').strip()
    if r.returncode != 0:
        return {'executes': False, 'correct': False, 'detail': (r.stderr or '').strip()[-500:]}
    return {'executes': True, 'correct': out.startswith('CORRECT'), 'detail': out[:400]}


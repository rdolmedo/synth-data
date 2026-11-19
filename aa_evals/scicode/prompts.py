"""SciCode prompts and code extraction, as in the official harness (scicode-bench/SciCode eval/scripts/gencode.py).

- With background: `multistep_template.txt` (AA's setting: "scientist-annotated background ... in the prompt").
- Without background: `background_comment_template.txt` (the model writes the background as a comment first).
Both templates are the upstream files (md5 5de36afe... and 3a376df5..., as vendored by SciCode-Verified).

Previous steps appear in the prompt (and in the cumulative graded code) with the model's own full extracted code,
as in an uninterrupted official run (gencode.py keeps `extract_python_script(response)` in memory). The gold code of
the skipped steps 13.6/62.1/76.3 is injected in full, as SciCode-Verified's runner does; the official harness cuts it
with `get_function_from_code`, which breaks problems 13/62/76 under inspect_ai (SciCode issue #59).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_BG = (HERE / 'templates' / 'multistep_template.txt').read_text()
TEMPLATE_NOBG = (HERE / 'templates' / 'background_comment_template.txt').read_text()
_IMPORT_RE = re.compile(r'^\s*(import .*|from .*\s+import\s+.*)', flags=re.MULTILINE)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_code(response: str | None) -> str:
    """The official `extract_python_script`: the first ```python block (else the first ``` block, else the whole
    response), with import lines removed (the harness prepends the problem's dependencies)."""
    if not response:
        return ''
    if '```' in response:
        if '```python' in response:
            s = response.split('```python')[1].split('```')[0]
        else:
            s = response.split('```')[1].split('```')[0]
    else:
        s = response
    return _IMPORT_RE.sub('', s)


def build_prompt(prob: dict, k: int, prev_codes: list[str], with_background: bool) -> str:
    """Prompt for step index k; prev_codes[i] is the code of step i < k."""
    def desc(i):
        s = prob['sub_steps'][i]
        return s['step_description_prompt'] + '\n' + s['step_background'] if with_background \
            else s['step_description_prompt']
    lines = []
    for i in range(k):
        lines += [desc(i), prev_codes[i], '------']
    cur = prob['sub_steps'][k]
    next_step = '\n\n'.join([desc(k), f"{cur['function_header']}\n\n{cur['return_line']}"])
    template = TEMPLATE_BG if with_background else TEMPLATE_NOBG
    return template.format(problem_steps_str='\n\n'.join(lines[:-1]), next_step_str=next_step,
                           dependencies=prob['required_dependencies'])

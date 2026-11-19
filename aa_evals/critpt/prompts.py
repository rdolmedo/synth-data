"""Prompts.

System prompt: the CritPt authors' one-step prompt (templates/prompt_template_default_one_step.yaml at the pinned
"CritPt v2" commit), rendered with their Jinja keys. AA uses "the CritPt authors' one-step prompt, in its version for
models with code tools"; the code-tools variant is not in the public repo, so tool use is described in our user
prompt instead.

User prompt: AA's text is unpublished. Ours covers the four items AA lists: the problem, the installed packages, the
execution limits and the submission rules.
"""
from __future__ import annotations

import hashlib

import jinja2
import yaml

from .data import GH_DIR, Problem

ONE_STEP = GH_DIR / 'templates' / 'prompt_template_default_one_step.yaml'


def system_prompt() -> str:
    spec = yaml.safe_load(ONE_STEP.read_text())
    base = yaml.safe_load((ONE_STEP.parent / spec.pop('IMPORTS')).read_text())
    keys = {**base, **spec}                                    # one-step overrides PROMPT_SPECS_STYLE
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    ctx = {k: v for k, v in keys.items() if k.startswith('PROMPT_SPECS_')}
    # SYSTEM_PROMPT_* keys reference each other; render until nothing changes.
    rendered = {k: str(v) for k, v in keys.items() if k.startswith('SYSTEM_PROMPT')}
    for _ in range(10):
        nxt = {k: env.from_string(v).render(**ctx, **rendered) for k, v in rendered.items()}
        if nxt == rendered:
            break
        rendered = nxt
    else:
        raise RuntimeError('system prompt keys did not converge')
    assert ctx.get('PROMPT_SPECS_STYLE') == 'one-step'
    return rendered['SYSTEM_PROMPT'].strip() + '\n'


def user_prompt(p: Problem, *, max_turns: int, cmd_timeout: int) -> str:
    return f"""{p.description.strip()}

# Code template
Your final answer must be submitted by filling in this Python code template:

```python
{p.code_template.strip()}
```

# Environment
- You can run shell commands with the `code_exec` tool, in an offline Linux sandbox. Python 3.13 is installed with numpy, scipy, sympy (with mpmath) and matplotlib. There is no internet access, and no other packages can be installed.
- Each command may run for at most {cmd_timeout} seconds. Files you create stay in the working directory for the rest of the task.
- You have at most {max_turns} turns. Every response you give uses one turn, whether or not it calls a tool.

# Submission
- Submit by calling the `finish` tool once, with three fields:
  - `reason`: why you are finishing;
  - `derivation`: your derivation of the result, step by step;
  - `answer`: the complete code template above with your final answer filled in. Keep the function name and signature, and import nothing beyond what the template already imports.
- Only the `answer` field is graded. If you do not call `finish` before your turns run out, the problem scores zero.
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

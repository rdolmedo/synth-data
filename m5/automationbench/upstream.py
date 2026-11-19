"""Access to upstream AutomationBench (v1.0.6, MIT): scored tasks, the 200 simple tasks, system prompts, noise.

Runs in the AutomationBench venv (/opt/aai/venvs/automationbench).
Test items (the 600 scored tasks) are read here only for statistics, labels and decontamination; they never reach
the generator.
"""
from __future__ import annotations

import contextlib
import importlib
import json

DOMAINS = ['sales', 'marketing', 'operations', 'support', 'finance', 'hr']
REPO = '/data/aai/aa_evals/data/automationbench/repo_4a8e106'
UPSTREAM_COMMIT = '4a8e1061254004d9dac807054eed33fad7d1ff14'


def system_prompt(domain: str) -> str:
    return importlib.import_module(f'automationbench.domains.{domain}.tasks').SYSTEM_PROMPT


def apply_noise(domain: str, task: dict) -> dict:
    """The domain's own noise injector (seeded by example_id); sales/marketing/operations take a list."""
    import inspect
    fn = importlib.import_module(f'automationbench.domains.{domain}._noise').apply_noise
    if next(iter(inspect.signature(fn).parameters)) == 'tasks':
        fn([task])
        return task
    return fn(task)


@contextlib.contextmanager
def _no_noise(domain):
    mod = importlib.import_module(f'automationbench.domains.{domain}.tasks')
    orig = getattr(mod, 'apply_noise', None)
    if orig is not None:
        mod.apply_noise = lambda t: t
    try:
        yield
    finally:
        if orig is not None:
            mod.apply_noise = orig


def load(domain: str, noise: bool = True) -> list[dict]:
    """Upstream rows of one domain: {'example_id', 'prompt', 'info' (dict, with task_name), ...}."""
    from automationbench.domains import get_domain_dataset
    ctx = contextlib.nullcontext() if noise else _no_noise(domain)
    with ctx:
        rows = get_domain_dataset(domain).to_list()
    for r in rows:
        r['info'] = json.loads(r['info']) if isinstance(r['info'], str) else r['info']
        r['task_name'] = r['info']['task_name']
        r['domain'] = domain
        r['trigger'] = [m['content'] for m in r['prompt'] if m['role'] == 'user'][0]
    return rows


def scored(noise: bool = True) -> list[dict]:
    return [r for d in DOMAINS for r in load(d, noise)]

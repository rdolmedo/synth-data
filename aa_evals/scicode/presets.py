"""Per-model request settings and AA reference rows.

Request settings start from the serving registry's `request.overrides` (the lab's recommended sampling and thinking
switches, as pinned in the teacher spec on aa-serving), so any served model works without code changes. `--effort`
then sets the reasoning effort everywhere the registry put one (top level and chat_template_kwargs), because Qwen-style
templates read it from chat_template_kwargs and the registry pins its own (Qwen3.8: xhigh).

M3.0 efforts (#173/#174): DeepSeek V4.1 Flash 1017 at Max; Qwen3.8-27B at Medium and Xhigh; every other model with
reasoning on at the registry's settings. A url: endpoint has no registry: pass --extra-body / --effort explicitly.

AA_ROWS maps (served_name, effort) to AA's leaderboard slug (aa_leaderboard_2026-11-02.json); `scicode` there is
sub-step accuracy over 288 steps x 3 repeats (e.g. Qwen3.8 27B Medium 0.390046 = 337/864).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

AA_LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')

AA_ROWS = {
    ('qwen38-27b', 'xhigh'): 'qwen3-8-27b',
    ('qwen38-27b', 'medium'): 'qwen3-8-27b-medium',
    ('qwen38-27b', 'low'): 'qwen3-8-27b-low',
    ('deepseek-v41-flash-1017', 'max'): 'deepseek-v4-1-flash-1017',
    ('deepseek-v41-flash-1017', 'high'): 'deepseek-v4-1-flash-1017-high',
    ('nemotron-3.5-super-vl', None): 'nemotron-3-5-super-vl',
    ('nemotron-3.5-lightning', None): 'nemotron-3-5-lightning',
    ('nemotron-3.5-lightning-30b-a3b', None): 'nemotron-3-5-lightning',
    ('qwen36-35b-a3b', None): 'qwen3-6-35b-a3b',
    ('qwen3.6-35b-a3b', None): 'qwen3-6-35b-a3b',
    ('gemma-4-26b-a4b-it', None): 'gemma-4-26b-a4b',
    ('gemma4-26b-a4b-it', None): 'gemma-4-26b-a4b',
    ('qwen35-4b', None): 'qwen3-5-4b',
    ('qwen3.5-4b', None): 'qwen3-5-4b',
}


def request_settings(endpoint, *, effort=None, temperature=None, top_p=None, extra_body=None) -> dict:
    gen = {}
    if endpoint.kind == 'serving':
        gen = copy.deepcopy((endpoint.request or {}).get('overrides') or {})
    if effort:
        gen['reasoning_effort'] = effort
        ctk = gen.get('chat_template_kwargs')
        if isinstance(ctk, dict) and 'reasoning_effort' in ctk:
            gen['chat_template_kwargs'] = {**ctk, 'reasoning_effort': effort}
    if temperature is not None:
        gen['temperature'] = temperature
    if top_p is not None:
        gen['top_p'] = top_p
    if extra_body:
        gen.update(json.loads(extra_body))
    gen.setdefault('temperature', 0.6)   # AA's default for reasoning models when the lab gives none
    gen.pop('max_tokens', None)          # set per request from the context cap
    return gen


def effective_effort(gen: dict):
    ctk = gen.get('chat_template_kwargs') or {}
    return gen.get('reasoning_effort') or ctk.get('reasoning_effort')


def aa_reference(served_name: str, effort, slug: str | None = None) -> dict | None:
    slug = slug or AA_ROWS.get((served_name, effort)) or AA_ROWS.get((served_name, None))
    if not slug:
        return None
    try:
        models = json.loads(AA_LEADERBOARD.read_text())['models']
    except (OSError, ValueError, KeyError):
        return None
    for m in models:
        if m.get('slug') == slug:
            v = (m.get('scores') or {}).get('scicode')
            return {'slug': slug, 'name': m.get('name'), 'scicode': None if v is None else round(100 * v, 2),
                    'source': str(AA_LEADERBOARD), 'metric': 'sub-step accuracy, original SciCode, with background'}
    return {'slug': slug, 'scicode': None}

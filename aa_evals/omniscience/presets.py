"""Generation presets: each mirrors the AA variant we compare against.

AA's rule: the lab's recommended temperature (else 0 for non-reasoning models, 0.6 for reasoning models) and the
creator's max output tokens. Our sampling follows the teacher specs in synth/teachers/*.toml (branch
task-synthesis), which use the labs' recommended settings. `max_tokens = "auto"` means the endpoint's context length
minus a margin, the closest we get to AA's "creator max". The preset is picked from the served name unless --preset
is given; CLI flags override single fields.

`aa_name` is the model's name in AA's leaderboard (/data/aai/tbench/index_audit/aa_leaderboard_*.json).
"""
from __future__ import annotations

PRESETS: dict[str, dict] = {
    # Thinking on at full effort (medium_effort off), NVIDIA's recommended sampling.
    'nemotron-3.5-super-vl': {
        'aa_name': 'Nemotron 3.5 Super VL',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 'auto',
                'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}},
    },
    'deepseek-v41-flash-1017-max': {
        'aa_name': 'DeepSeek V4.1 Flash 1017 (Max)',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 'auto', 'reasoning_effort': 'max',
                'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'max'}},
    },
    'deepseek-v41-flash-1017-high': {
        'aa_name': 'DeepSeek V4.1 Flash 1017 (High)',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 'auto', 'reasoning_effort': 'high',
                'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'high'}},
    },
    # AA's "Qwen3.8 27B (Xhigh)": thinking on at xhigh effort, Qwen's recommended sampling (the serving registry's
    # request.overrides for qwen38-27b).
    'qwen3.8-27b-xhigh': {
        'aa_name': 'Qwen3.8 27B (Xhigh)',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
                'repetition_penalty': 1.0, 'max_tokens': 'auto', 'reasoning_effort': 'xhigh',
                'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True,
                                         'reasoning_effort': 'xhigh'}},
    },
    # AA's "Qwen3.8 27B (Medium)": the same, at medium effort.
    'qwen3.8-27b-medium': {
        'aa_name': 'Qwen3.8 27B (Medium)',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
                'repetition_penalty': 1.0, 'max_tokens': 'auto', 'reasoning_effort': 'medium',
                'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True,
                                         'reasoning_effort': 'medium'}},
    },
    # AA lists a single GLM-5.3-Flash variant (reasoning); our teacher spec serves it at max effort.
    'glm-5.3-flash': {
        'aa_name': 'GLM-5.3-Flash',
        'gen': {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 'auto', 'reasoning_effort': 'max'},
    },
    # AA's grader. OpenAI reasoning models take no temperature/top_p; the default max output applies.
    'gpt-5.6-luna': {
        'aa_name': None,
        'gen': {'reasoning_effort': 'medium'},
    },
}

# Served name -> preset when --preset is not given.
DEFAULT_FOR_SERVED = {
    'nemotron-3.5-super-vl': 'nemotron-3.5-super-vl',
    'deepseek-v41-flash-1017': 'deepseek-v41-flash-1017-max',
    'glm-5.3-flash': 'glm-5.3-flash',
    'qwen38-27b': 'qwen3.8-27b-xhigh',
    'gpt-5.6-luna': 'gpt-5.6-luna',
}

# A self-hosted judge's output cap (reasoning included); the same as HLE's judge (aa_evals/hle --judge-max-tokens).
JUDGE_LOCAL_MAX_TOKENS = 32768


def for_model(served_name: str, preset: str | None) -> tuple[str | None, dict]:
    name = preset or DEFAULT_FOR_SERVED.get(served_name)
    if name is None:
        return None, {'aa_name': None, 'gen': {'max_tokens': 'auto'}}
    if name not in PRESETS:
        raise ValueError(f'unknown preset {name!r}; known: {sorted(PRESETS)}')
    p = PRESETS[name]
    return name, {'aa_name': p['aa_name'], 'gen': {k: (dict(v) if isinstance(v, dict) else v)
                                                   for k, v in p['gen'].items()}}

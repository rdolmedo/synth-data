"""Judge arms for the open-judge study (#41).

An arm is one judge server plus one request setting. Its `extra` dict is passed unchanged as --judge-extra-body to
every benchmark's grader (Omniscience, HLE, GDP.pdf). Each grader merges it shallowly over the serving registry's
request.overrides (the lab's sampling), so `chat_template_kwargs` is spelled out in full, and `reasoning_effort:
None` clears a registry effort that would otherwise turn thinking back on. Several arms share a server, so a
cheaper setting costs no extra GPU.

Roles: `candidate` judges are the ones the user picks from, and must be deployable on 1×H100 (best) or 1×B200.
`reference` arms are larger models that run later on the shared servers of the stronger-model runs (their GPUs are
amortized there). They anchor the agreement measures but are not candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arm:
    name: str
    server: str            # serving registry name: /data/aai/tbench/tasksynth/serving/<server>/endpoints.json
    model: str
    hardware: str
    gpus: int
    effort: str            # human label of the thinking setting
    extra: dict = field(default_factory=dict)
    role: str = 'candidate'
    shared_server: bool = False  # wall time is then approximate (other sessions load the same GPUs)


def _qwen(effort: str | None) -> dict:
    if effort is None:
        return {'reasoning_effort': None, 'chat_template_kwargs': {'enable_thinking': False}}
    return {'reasoning_effort': effort, 'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': effort}}


ARMS: dict[str, Arm] = {a.name: a for a in [
    # Qwen3.8 27B, official FP8 build, 1×H100 (teacher qwen38-27b-fp8; registry default xhigh).
    Arm('qwen38-27b-nothink', 'aa-judge-qwen38-27b', 'Qwen3.8 27B FP8', '1×H100', 1, 'thinking off', _qwen(None)),
    # Thinking off, output capped at 4k: under HLE's strict JSON schema the uncapped arm ran away to 32k tokens on
    # ~10% of attempts (each retried, ~30 min of decode apiece). A cap makes the runaway cheap; retries still recover.
    Arm('qwen38-27b-nothink-cap4k', 'aa-judge-qwen38-27b', 'Qwen3.8 27B FP8', '1×H100', 1, 'thinking off, 4k cap',
        {**_qwen(None), 'max_tokens': 4096}),
    Arm('qwen38-27b-medium', 'aa-judge-qwen38-27b', 'Qwen3.8 27B FP8', '1×H100', 1, 'medium', _qwen('medium')),
    Arm('qwen38-27b-xhigh', 'aa-judge-qwen38-27b', 'Qwen3.8 27B FP8', '1×H100', 1, 'xhigh', _qwen('xhigh')),
    # Nemotron 3.5 Super VL NVFP4, 1×B200, the shared aa-super. medium_effort is Super's own reduced-thinking mode.
    Arm('super-medium', 'aa-super', 'Nemotron 3.5 Super VL NVFP4', '1×B200', 1, 'medium_effort',
        {'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': True}}, shared_server=True),
    Arm('super-full', 'aa-super', 'Nemotron 3.5 Super VL NVFP4', '1×B200', 1, 'full thinking',
        {'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}}, shared_server=True),
    # The pinned production judges (coordinator, after audit 1 on #41): every benchmark grades Qwen's runs with these
    # two, spelled out in full so no benchmark's own judge defaults (GDP.pdf: T 0.6, 16k) leak in. Qwen: medium with
    # the registry's sampling, 32k. Super: the registry defaults (T 1.0, top_p 0.95, full thinking), 32k.
    Arm('qwen38-27b-medium-pin', 'aa-judge-qwen38-27b', 'Qwen3.8 27B FP8', '1×H100', 1, 'medium, 32k (pinned)',
        {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, **_qwen('medium'), 'max_tokens': 32768}),
    Arm('super-full-pin', 'aa-super', 'Nemotron 3.5 Super VL NVFP4', '1×B200', 1, 'full thinking, 32k (pinned)',
        {'temperature': 1.0, 'top_p': 0.95, 'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False},
         'max_tokens': 32768}, shared_server=True),
    # Study s3 (#41, user 2026-11-04): two Gemma 4 judges, each on its own 1xH100 (specs synth/teachers/gemma4-*.toml).
    # Gemma 4 has one thinking switch and no effort levels; thinking on is the closest to AA's "medium" judge.
    # Google's sampling (T 1.0, top_p 0.95, top_k 64), 32k output, spelled out as the pinned arms are.
    Arm('gemma4-26b-a4b-pin', 'aa-judge-gemma4-26b', 'Gemma 4 26B-A4B-it (bf16)', '1×H100', 1, 'thinking on, 32k',
        {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'min_p': 0.0,
         'chat_template_kwargs': {'enable_thinking': True}, 'max_tokens': 32768}),
    Arm('gemma4-31b-fp8-pin', 'aa-judge-gemma4-31b', 'Gemma 4 31B-it FP8', '1×H100', 1, 'thinking on, 32k',
        {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'min_p': 0.0,
         'chat_template_kwargs': {'enable_thinking': True}, 'max_tokens': 32768}),
    # M3.5 gold/reference judge: the same one shared DeepSeek server used for generation, at Max.
    # Explicit full settings avoid inheriting High from the older study arm or another grader's defaults.
    Arm('dsv41-flash-max-pin', 'aa-ds', 'DeepSeek V4.1 Flash 1017', '4×B200', 4, 'max, 32k (pinned)',
        {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'max',
         'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'max', 'drop_thinking': False},
         'max_tokens': 32768}, role='reference', shared_server=True),
    # Upper references, later, on the shared stronger-model servers (registry names are set when they exist).
    Arm('glm53-flash-max', 'TBD-glm53-flash', 'GLM-5.3-Flash NVFP4', '≥2×B200', 2, 'max', {},
        role='reference', shared_server=True),
    Arm('dsv41-flash-high', 'TBD-dsv41-flash', 'DeepSeek V4.1 Flash 1017', '4×B200', 4, 'high',
        {'reasoning_effort': 'high', 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'high'}},
        role='reference', shared_server=True),
]}

# The pinned judge per side of the self-preference 2x2 (Qwen's runs are graded by both).
PINNED = {'qwen': 'qwen38-27b-medium-pin', 'super': 'super-full-pin'}

# The arm that represents each server in leave-one-out majorities (closest to AA's "medium" grader setting).
PRIMARY = {'aa-judge-qwen38-27b': 'qwen38-27b-medium', 'aa-super': 'super-medium',
           'TBD-glm53-flash': 'glm53-flash-max', 'TBD-dsv41-flash': 'dsv41-flash-high'}


def apply_arm(name: str, judge: str, extra_json: str | None, default_judge: str) -> tuple[str, str]:
    """--judge-arm for every grader: the arm's server becomes the judge and its settings the extra body (an explicit
    --judge-extra-body merges over them). A --judge other than the default must name the arm's own server."""
    import json
    if name not in ARMS:
        raise SystemExit(f'unknown --judge-arm {name}; one of {sorted(ARMS)}')
    arm = ARMS[name]
    if judge not in (default_judge, f'serving:{arm.server}'):
        raise SystemExit(f'--judge {judge} conflicts with --judge-arm {name} (server {arm.server})')
    return f'serving:{arm.server}', json.dumps({**arm.extra, **json.loads(extra_json or '{}')})

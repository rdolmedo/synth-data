"""The M3.0 eval protocol (#173, #175) for the single-turn judged benchmarks: Omniscience, HLE-Diamond, GDP.pdf.

One place for the rules every runner applies, so a run of any M3.1 model is one command:
- **Context:** 131,072 tokens in total (prompt + output) for regular evals. Full native context only for explicit
  Qwen3.8 calibration runs (`--context-window 0` = the endpoint's own).
- **Model settings:** the serving registry's `request.overrides` (the lab's sampling, thinking on, the spec's default
  effort), then `--reasoning-effort`, which is set wherever the model's request already carries an effort (top level
  and `chat_template_kwargs`, so a template that reads only the kwarg can't silently keep the registry's), then
  explicit CLI fields. A model without effort levels refuses an effort instead of ignoring it.
- **Image inputs:** from the serving launch argv. `--language-model-only`, or no image allowance in
  `--limit-mm-per-prompt`, means a text-only model, which gets AA's text-only path.
- **AA reference:** looked up by served name and effort, so a Medium run can't pick up the Xhigh entry.
- **Summary fields:** model, effort, context cap, judge arm, code SHA and repeats here; `n_flagged` from the M3.4
  trace judge's hook (`aa_evals.tracejudge.fold_summary`, null until it has read the run), which each runner calls
  right before writing summary.json.
"""
from __future__ import annotations

import json

CONTEXT_WINDOW = 131072
SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry

# (served name, effort) -> (AA leaderboard name, AA slug); effort None = the model has no effort levels (reasoning
# on). Names and slugs from /data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json. The M3.1 list (#174).
AA_MODELS: dict[tuple[str, str | None], tuple[str, str]] = {
    ('qwen38-27b', 'medium'): ('Qwen3.8 27B (Medium)', 'qwen3-8-27b-medium'),
    ('qwen38-27b', 'xhigh'): ('Qwen3.8 27B (Xhigh)', 'qwen3-8-27b'),
    ('qwen38-27b', 'low'): ('Qwen3.8 27B (Low)', 'qwen3-8-27b-low'),
    ('deepseek-v41-flash-1017', 'max'): ('DeepSeek V4.1 Flash 1017 (Max)', 'deepseek-v4-1-flash-1017'),
    ('deepseek-v41-flash-1017', 'high'): ('DeepSeek V4.1 Flash 1017 (High)', 'deepseek-v4-1-flash-1017-high'),
    ('nemotron-3.5-super-vl', None): ('Nemotron 3.5 Super VL', 'nemotron-3-5-super-vl'),
    ('nemotron35-lightning', None): ('Nemotron 3.5 Lightning', 'nemotron-3-5-lightning'),
    ('qwen36-35b-a3b', None): ('Qwen3.6 35B A3B (Reasoning)', 'qwen3-6-35b-a3b'),
    ('gemma-4-26b-a4b-it', None): ('Gemma 4 26B A4B (Reasoning)', 'gemma-4-26b-a4b'),
    ('qwen35-4b', None): ('Qwen3.5 4B (Reasoning)', 'qwen3-5-4b'),
    ('glm-5.3-flash', 'max'): ('GLM-5.3-Flash', 'glm-5-3-flash'),
}


def effort_of(body: dict) -> str | None:
    """The reasoning effort a request asks for, else None (a model without effort levels)."""
    kw = body.get('chat_template_kwargs') or {}
    return body.get('reasoning_effort') or kw.get('reasoning_effort')


def thinking_of(body: dict) -> bool | None:
    kw = body.get('chat_template_kwargs') or {}
    for k in ('enable_thinking', 'thinking'):
        if k in kw:
            return bool(kw[k])
    return None


def apply_effort(body: dict, effort: str) -> dict:
    """`body` with `effort` wherever the model's request already carries an effort (both places when either is
    there, since vLLM templates read the kwarg and a top-level value alone changes nothing on Qwen3.8)."""
    kw = dict(body.get('chat_template_kwargs') or {})
    if 'reasoning_effort' not in body and 'reasoning_effort' not in kw:
        raise ValueError('this model has no effort levels in its serving request (it runs with reasoning on); '
                         'drop --reasoning-effort, or set it with --extra-body if the model really takes one')
    out = dict(body, reasoning_effort=effort)
    out['chat_template_kwargs'] = dict(kw, reasoning_effort=effort)
    return out


def aa_model(served_name: str | None, body: dict) -> tuple[str | None, str | None]:
    """(AA name, AA slug) of the variant this request reproduces, or (None, None)."""
    return AA_MODELS.get((served_name or '', effort_of(body)), (None, None))


def launch(spec: str) -> dict | None:
    """The serving launch record (argv, env) of a `serving:` spec, or None."""
    kind, _, name = spec.partition(':')
    if kind != 'serving':
        return None
    from aa_evals.common import serving   # here: frozen copies that carry this module carry serving.py too
    try:
        rec = json.loads((serving.registry_root(SERVING_ROOT) / name / 'launch.json').read_text())
    except (OSError, ValueError):
        return None
    return rec['replicas'][0] if 'replicas' in rec else rec


def image_inputs(spec: str) -> tuple[bool | None, str]:
    """(does the endpoint take images, why). None: unknown (a url: spec), the caller's flag decides."""
    rec = launch(spec)
    if rec is None or not rec.get('argv'):
        return None, 'no serving launch record'
    argv = rec['argv']
    if '--language-model-only' in argv:
        return False, 'served with --language-model-only'
    for i, a in enumerate(argv[:-1]):
        if a == '--limit-mm-per-prompt':
            try:
                n = int(json.loads(argv[i + 1]).get('image', 0))
            except (ValueError, AttributeError):
                return None, f'unparsed --limit-mm-per-prompt {argv[i + 1]!r}'
            return n > 0, f'--limit-mm-per-prompt image={n}'
    return False, 'no --limit-mm-per-prompt image allowance in the launch argv'


def window(endpoint_ctx: int | None, requested: int | None) -> tuple[int | None, str]:
    """The context a run may use: the protocol cap (requested; 0 = no cap), never above the endpoint's own."""
    if not requested:
        return endpoint_ctx, 'endpoint context (no cap: Qwen3.8 calibration runs only)'
    if endpoint_ctx is None or endpoint_ctx >= requested:
        return requested, f'{requested}-token cap (M3.0 protocol, #173)'
    return endpoint_ctx, f'endpoint context {endpoint_ctx} (below the {requested}-token cap)'


def summary_fields(*, model: str | None, served_name: str | None, body: dict, context: dict,
                   judge_arm: str | None, judge_settings_key: str | None, code_sha: str | None,
                   repeats: int, image_inputs: bool | None) -> dict:
    """The protocol block every judged benchmark's summary.json carries (#175)."""
    name, slug = aa_model(served_name, body)
    return {'protocol': {
        'model': model, 'served_name': served_name, 'effort': effort_of(body), 'thinking': thinking_of(body),
        'aa_variant': name, 'context_cap': context, 'image_inputs': image_inputs, 'repeats': repeats,
        'judge_arm': judge_arm, 'judge_settings_key': judge_settings_key, 'code_sha': code_sha}}

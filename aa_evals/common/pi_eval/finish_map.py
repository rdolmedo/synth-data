"""The Pi finish notice map for SFT exporters (#375 sftnotice, user 2026-11-13: SFT traces keep the notice "to avoid
train/test gaps in this aspect").

The context guard (tb_agents/pi_context_limit.py, policy STICKY_POLICY) joins a notice to the outgoing request only;
Pi's session never contains it. With sticky notices the final request of a trial is the one linear trace the model
saw, and each new notice is logged in the trial's pi-cont16k/finish-notices.jsonl with its place:
  index        position in the forwarded request (system prompt included; prepare.py's message list has the same
               indices: system prompt from the first wire request, then the session's messages)
  role         'user' or 'tool'
  base_sha256  sha256 of the message's normalized text before the notice (pi_context_limit.notice_base_text)
  text         the notice as joined (pi_context_limit.join_notice)
  appended     True only if the guard had to add a user message (never seen from Pi): such a trace is not usable
Exporters ship the parsed lines as "finish_notices" next to "sessions" in each archive line; sft_qwen35/prepare.py
joins them to the training messages (or drops the attempt on any mismatch). When one index has several lines
(a retried request), the last one is what later requests carried.

A v4 guard creates finish-notices.jsonl (empty) at startup, so in a v4 run an archive without it has lost its map:
the attempt is unusable ('finish_notices_missing'), with no heuristic. Secondary check: an empty map while the
largest served prompt (pi_eval stats.max_prompt_tokens, from the usage of every wire response) was inside the
notice threshold contradicts the guard ('finish_notices_empty_near_window').
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from tb_agents import pi_context_limit as G

NOTICES = 'pi-cont16k/finish-notices.jsonl'
MAP_KEYS = ('index', 'role', 'base_sha256', 'text', 'appended')


def member_kind(name: str) -> str | None:
    """'notices' for the trial archive member that is the guard's notice log."""
    return 'notices' if name.endswith('/' + NOTICES) else None


def run_policy(cfg: dict, run) -> dict | None:
    """protocol.finish_notice of a pi_eval run, for SFT export: None (no notice; exported as before) or the v4 sticky
    policy. A run with the v3 notice (not sticky, no map: the trace does not show what the model saw) is refused."""
    fn = (cfg.get('protocol') or {}).get('finish_notice')
    if not fn:
        return None
    if not fn.get('sticky') or fn.get('policy') != G.STICKY_POLICY:
        raise SystemExit(f'{run}: generated with the non-sticky Pi finish notice ({fn.get("policy")}); its traces do '
                         f'not record where the model saw the notices and are not usable for SFT. Generate with '
                         f'{G.STICKY_POLICY} (the pi_eval default since #375 sftnotice) or --finish-notice off')
    return fn


@functools.lru_cache(maxsize=None)
def run_settings(run: str) -> tuple[dict | None, int]:
    """(run_policy, context window) of a pi_eval run directory (its config.json; none = no notice)."""
    p = Path(run) / 'config.json'
    cfg = json.loads(p.read_text()) if p.exists() else {}
    return run_policy(cfg, run), int((cfg.get('protocol') or {}).get('context_window') or 131072)


def parse(text: str) -> list[dict]:
    """finish-notices.jsonl -> its lines (ValueError if a line is not a v4 map entry). Split on '\\n' only: JSON
    strings may hold U+2028 and other characters str.splitlines() splits on."""
    out = []
    for line in text.split('\n'):
        if not line.strip():
            continue
        x = json.loads(line)
        if x.get('policy') != G.STICKY_POLICY or any(k not in x for k in MAP_KEYS):
            raise ValueError(f'not a {G.STICKY_POLICY} notice map line: {line[:200]}')
        if not isinstance(x['index'], int) or x['role'] not in ('user', 'tool') or not isinstance(x['text'], str):
            raise ValueError(f'bad notice map entry: {line[:200]}')
        out.append(x)
    return out


def threshold(fn: dict) -> int:
    """Remaining context at or below which the guard always gives a notice (finish_plan: warn at <= warn_at, clamp
    below reserve + TURN, final below that)."""
    return max(int(fn.get('warn_at') or 0), int(fn.get('reserve') or 0) + G.TURN - 1 if fn.get('reserve') else 0)


def near_window(fn: dict, window: int, max_prompt_tokens) -> bool:
    """Whether the largest served prompt was inside the notice threshold: that request had a notice (a served prompt
    includes at most one new notice, < NOTICE_MARGIN tokens, on top of the count the guard decided on)."""
    return max_prompt_tokens is not None and int(max_prompt_tokens) >= window - threshold(fn) + G.NOTICE_MARGIN


def attempt_notices(fn: dict | None, window: int, notices_text: str | None,
                    max_prompt_tokens: int | None = None) -> tuple[list | None, str | None]:
    """(finish_notices for the archive line, None) or (None, why the attempt is unusable for SFT).
    fn None (run without a notice): (None, None), nothing to ship. In a v4 run the guard creates the notice file at
    startup, so an archive without it has lost the map. max_prompt_tokens (outcome stats): an empty map while the
    largest served prompt was inside the threshold is a lost map too."""
    if not fn:
        return None, None
    if notices_text is None:
        return None, 'finish_notices_missing'
    try:
        notices = parse(notices_text)
    except ValueError as e:
        return None, f'finish_notices_malformed: {str(e)[:120]}'
    if any(x['appended'] for x in notices):
        return None, 'finish_notice_appended'
    if not notices and near_window(fn, window, max_prompt_tokens):
        return None, 'finish_notices_empty_near_window'
    return notices, None

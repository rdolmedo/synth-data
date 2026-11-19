"""The AA-Omniscience grader: AA's prompt verbatim, one user message, a single letter back."""
from __future__ import annotations

import re

from . import client
from .prompts import GRADES, LETTER_TO_GRADE, OMNISCIENCE_GRADER_TEMPLATE

GRADE_TO_LETTER = {v: k for k, v in LETTER_TO_GRADE.items()}
_WRAP = r'[\s"\'`*_()\[\].:]*'
_LETTER_ONLY = re.compile(rf'{_WRAP}([ABCD]){_WRAP}')
_LETTER_AND_WORD = re.compile(rf'{_WRAP}([ABCD])\s*[:\-)]\s*({"|".join(GRADES)}){_WRAP}', re.I)
_WORD_ONLY = re.compile(rf'{_WRAP}({"|".join(GRADES)}){_WRAP}', re.I)


def parse_letter(text: str) -> str | None:
    """Accept exactly what the prompt asks for ("A".."D"), tolerating quotes, bold, a trailing period, or the
    letter echoed with its label ("B: INCORRECT"). Anything else is unparseable and gets retried."""
    t = (text or '').strip()
    m = _LETTER_ONLY.fullmatch(t)
    if m:
        return m.group(1)
    m = _LETTER_AND_WORD.fullmatch(t)
    if m and GRADE_TO_LETTER[m.group(2).upper()] == m.group(1):
        return m.group(1)
    m = _WORD_ONLY.fullmatch(t)
    if m:
        return GRADE_TO_LETTER[m.group(1).upper()]
    return None


def render(question: str, target: str, predicted: str) -> str:
    return OMNISCIENCE_GRADER_TEMPLATE.format(question=question, target=target, predicted_answer=predicted)


def grade(ep: client.Endpoint, gen: dict, *, question: str, target: str, predicted: str, timeout: float,
          parse_attempts: int = 3, http_attempts: int = 8) -> dict:
    """Returns {'grade', 'letter', 'source', 'calls': [...]} or raises client.InfraError.

    An empty predicted answer is NOT_ATTEMPTED by the prompt's own rule ("e.g. empty response"); we assign it
    without a call. Unparseable grader output is retried, then counts as an infra error, never as a model grade."""
    if not predicted.strip():
        return {'grade': 'NOT_ATTEMPTED', 'letter': 'D', 'source': 'rule_empty', 'calls': []}
    body = {'messages': [{'role': 'user', 'content': render(question, target, predicted)}]}
    body.update(gen)
    calls = []
    for _ in range(parse_attempts):
        errors: list = []
        try:
            resp = client.chat(ep, body, timeout=timeout, attempts=http_attempts, log=errors)
        except client.InfraError as e:
            calls.append({'errors': errors, 'final_error': str(e)})
            raise client.InfraError(f'judge: {e}') from e
        out = client.split_response(resp)
        letter = parse_letter(out['answer'])
        calls.append({'output': out['answer'], 'reasoning_chars': len(out['reasoning']),
                      'finish_reason': out['finish_reason'], 'usage': out['usage'], 'errors': errors,
                      'latency_s': resp.get('_latency_s')})
        if letter:
            return {'grade': LETTER_TO_GRADE[letter], 'letter': letter, 'source': 'judge', 'calls': calls}
    err = client.InfraError(f'judge output unparseable {parse_attempts}x: {calls[-1]["output"][:200]!r}')
    err.calls = calls  # type: ignore[attr-defined]
    raise err

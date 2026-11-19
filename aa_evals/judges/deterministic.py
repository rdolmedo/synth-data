"""Deterministic verdicts, wherever exact checking is possible (#41, measure 1).

Each function returns the verdict a correct judge must give, or None when a rule can't decide; a judge is never
wrong about an item a rule can't decide. The rules are deliberately conservative: an item gets a verdict only
when no reasonable grader could disagree.
"""
from __future__ import annotations

import math
import re

# The letter must end the line (after an optional ')' / '.' / markdown): "Answer: A and C" is not a single letter.
_MC_LINE = re.compile(r'(?im)^[\s>*#_]*(?:final\s+|exact\s+)?answer[\s*_]*:[\s*_]*\(?\s*([A-Z])\s*\)?[\s.*_]*$')
_EXACT_LINE = re.compile(r'(?im)^[\s>*#_]*(?:exact\s+answer|final\s+answer|answer)[\s*_]*:[\s*_]*(.+?)\s*$')


def norm(s: str) -> str:
    s = (s or '').strip()
    s = re.sub(r'^\$+|\$+$', '', s).strip()
    m = re.fullmatch(r'\\boxed\{(.*)\}', s)
    if m:
        s = m.group(1)
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}', r'\1', s)
    s = s.replace('\\,', '').replace('\\!', '').replace('\\ ', ' ').replace('\\%', '%')
    s = s.strip().strip('*_`"\'').strip()
    s = s.rstrip('.').strip()
    return re.sub(r'\s+', ' ', s).lower()


def as_number(s: str) -> float | None:
    """A plain number (optional sign, thousands commas, decimals, exponent, trailing %), else None."""
    t = norm(s).replace(' ', '')
    t = t[:-1] if t.endswith('%') else t
    if re.fullmatch(r'[-+]?\d{1,3}(,\d{3})+(\.\d+)?', t):
        t = t.replace(',', '')
    if not re.fullmatch(r'[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?', t):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _num_verdict(pred: str, gold: str, wrong_rel: float) -> bool | None:
    p, g = as_number(pred), as_number(gold)
    if p is None or g is None:
        return None
    if norm(pred).endswith('%') != norm(gold).endswith('%'):
        # 12% vs 0.12 (or 54 vs 54%): a judge may read them as equal. Sure-wrong only when both readings are far;
        # never sure-correct.
        alt = p / 100 if norm(pred).endswith('%') else p * 100
        far = all(abs(x - g) > wrong_rel * max(abs(g), 1e-12) for x in (p, alt))
        return False if far else None
    diff = abs(p - g)
    scale = max(abs(g), 1e-12)
    if diff <= 1e-9 * scale:
        return True
    if diff > wrong_rel * scale:
        return False
    return None  # within a judge's "small margin": undecided


# ---------- HLE ----------

def hle_mc(content: str, gold_letter: str) -> bool | None:
    """Multiple choice: the last `Answer: X` line's letter against the gold letter."""
    hits = _MC_LINE.findall(content or '')
    if not hits:
        return None
    return hits[-1].upper() == gold_letter.strip().upper()


def hle_exact(content: str, gold: str) -> bool | None:
    """Exact-match questions: sure-correct when the normalized final answer equals the gold; sure-wrong only when
    both are plain numbers more than 1% apart. Everything else (formulas, prose, near misses) is undecided."""
    hits = _EXACT_LINE.findall(content or '')
    if not hits:
        return None
    pred = hits[-1]
    if norm(pred) == norm(gold) and norm(gold):
        return True
    return _num_verdict(pred, gold, wrong_rel=0.01)


# ---------- Omniscience ----------

_ABSTAIN = re.compile(
    r"^(?:i(?:\s+|(?=['’]))(?:do\s+not|don['’]t|am\s+not|['’]m\s+not|cannot|can['’]t|could\s+not|couldn['’]t|"
    r"am\s+unable|['’]m\s+unable|lack|have\s+no)\b"
    # a bare phrase must be the whole reply, up to final punctuation: "Unknown Pleasures" and "Unknown - Closer"
    # are answers (a phrase followed by more text is left to the judge)
    r"|(?:unknown|not\s+sure|no\s+idea|insufficient\s+information|cannot\s+(?:be\s+)?determined?)"
    r"\s*[.!]*\s*$)", re.I)

# A hedge followed by an answer ("I'm not sure, but I believe it is **X**") is an attempt under the grader prompt
# (graded on X), so an opening abstention counts as sure-abstain only in a short reply with none of these markers.
_HEDGED_ANSWER = re.compile(r"\b(?:believe|think|likely|probably|possibly|perhaps|might|maybe|guess|best|but|however|"
                            r"although|though|suspect|recall|appears?|seems?)\b|\*\*|:", re.I)
ABSTAIN_MAX_CHARS = 200
# Phrases that introduce a candidate answer after an opening abstention ("I'm not sure of the name; it was X"). The
# grader prompt can grade such a reply either way (example 4 vs a hedged attempt), so it is left to the judge.
_ANSWER_INTRO = re.compile(r"\b(?:it\s+(?:is|was|'s)|it[’']s|the\s+answer\s+is|(?:would|could|should|may)\s+be|"
                           r"i\s+(?:guess|believe|think))\b", re.I)


def omni_expected(answer: str, gold: str) -> str | None:
    """The grade AA's grader must give, when a rule can tell:
    - CORRECT: the normalized answer equals the normalized gold;
    - NOT_ATTEMPTED: a short answer (<= ABSTAIN_MAX_CHARS) that opens by saying it doesn't know and offers no
      hedged guess (the grader prompt: "explicitly states it does not know ... even though it has other
      information", example 4);
    - INCORRECT: both are plain numbers more than 5% apart (and the answer does not abstain)."""
    a = (answer or '').strip()
    if not a:
        return 'NOT_ATTEMPTED'
    if norm(a) == norm(gold) and norm(gold):
        return 'CORRECT'
    if _ABSTAIN.match(norm(a)):
        if len(a) <= ABSTAIN_MAX_CHARS and not _HEDGED_ANSWER.search(a) and not _ANSWER_INTRO.search(a):
            return 'NOT_ATTEMPTED'
        return None  # a hedge that may go on to answer: left to the judge
    if _num_verdict(a, gold, wrong_rel=0.05) is False:
        return 'INCORRECT'
    return None

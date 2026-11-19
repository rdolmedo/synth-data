"""A compact text view of a Trace for the LLM judge.

Long agentic traces (TB attempts run to 100-500 turns and 262k tokens) are condensed, not chunked:
  1. a header with what the pipeline recorded (status, score, end state, exit code, exception);
  2. the harness events (they decide most infra failures), then the pre-filter's hits;
  3. the task head;
  4. a one-line-per-step timeline: tool calls and tool outputs clipped, except steps that match an error pattern,
     which keep a window around the match;
  5. the last steps in more detail (how the attempt ended is where most infra failures show).
The middle of the timeline is dropped first when the digest is over budget, with a marker saying how much.
"""
from __future__ import annotations

import json
import re

from aa_evals.tracejudge.rules import Hit
from aa_evals.tracejudge.trace import Trace

ERRPAT = re.compile(
    r'Traceback|Error|error:|ERROR|Killed|killed|signal \d|exit(ed)? (code |with code )?-?\d{2,3}|No space|Read-only|'
    r'refused|reset by peer|unreachable|timed? ?out|Resource temporarily|Cannot allocate|Too many open|'
    r'Temporary failure|Could not resolve|proxy|50[0-9] |OOM|oom|Segmentation fault|core dumped|Permission denied')

BUDGET = 28000        # characters (~7-9k tokens): what the judges were measured with
TAIL_STEPS = 8


def _clip(s: str, n: int) -> str:
    s = (s or '').replace('\r', '')
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + f' …[{len(s) - n} chars]… ' + s[-(n - h):]


def _window(s: str, n: int) -> str:
    """Keep the text around the first error match (and the very end, where the error usually is)."""
    m = ERRPAT.search(s or '')
    if not m or len(s) <= n:
        return _clip(s, n)
    a = max(0, m.start() - n // 3)
    head = s[a:a + n // 2]
    tail = s[-n // 3:]
    return ('…' if a else '') + head + (' … ' + tail if a + n // 2 < len(s) - n // 3 else '')


def _line(x, detail=False) -> str:
    n = 1200 if detail else 220
    if x.kind == 'task':
        return f'[{x.ref}] {x.role.upper()} (task): {_clip(x.text, 1500 if detail else 600)}'
    if x.kind == 'thinking':
        return f'[{x.ref}] THINKING: {_clip(x.text, 600 if detail else 120)}'
    if x.kind == 'tool_call':
        return f'[{x.ref}] CALL {_clip(x.text, 700 if detail else 260)}'
    if x.kind == 'tool_result':
        flag = ' (isError)' if x.meta.get('is_error') else ''
        dur = f" ({x.meta['duration_ms'] / 1000:.0f}s)" if x.meta.get('duration_ms') else ''
        body = _window(x.text, n * 2 if ERRPAT.search(x.text or '') else n)
        return f'[{x.ref}] OUTPUT{flag}{dur} [{len(x.text or "")} chars]: {body}'
    if x.kind == 'error':
        return f'[{x.ref}] HARNESS ERROR: {_clip(x.text, 800)}'
    meta = {k: v for k, v in x.meta.items() if v not in (None, '', [])}
    return f'[{x.ref}] {x.role.upper()}: {_clip(x.text, 1500 if detail else 300)}' + (f'  {json.dumps(meta)}' if meta else '')


def render(t: Trace, hits: list[Hit], budget: int = BUDGET) -> str:
    rec = {k: v for k, v in t.recorded.items() if v is not None and v is not False and v != '' and v != [] and v != {}}
    head = [f'BENCHMARK: {t.bench}   ATTEMPT: {t.key}   KIND: {t.kind}',
            f'PIPELINE RECORD: {json.dumps(rec, default=str)[:1500]}']
    ev = [f'- [{e.ref}] {e.source}: {_clip(e.text, 700)}' for e in t.events]
    head.append('HARNESS EVENTS:\n' + ('\n'.join(ev) if ev else '- (none)'))
    if hits:
        head.append('PRE-FILTER HITS (deterministic hints; may be wrong; "note" = context about protocol limits):\n' +
                    '\n'.join(f'- {h.rule} [{h.polarity}{"/" + h.category if h.category else ""}] {_clip(h.evidence, 300)} ({h.ref})'
                              for h in hits))
    head = '\n\n'.join(head)

    if t.kind == 'single':
        body = []
        for x in t.turns:
            if x.kind == 'task':
                body.append(f'[{x.ref}] REQUEST: {_clip(x.text, 1500)}')
            elif x.kind == 'thinking':
                body.append(f'[{x.ref}] REASONING [{len(x.text)} chars]: {_clip(x.text, 2500)}')
            else:
                meta = {k: v for k, v in x.meta.items() if v not in (None, '', [])}
                body.append(f'[{x.ref}] ANSWER [{len(x.text or "")} chars]: {_clip(x.text, 4000) or "(empty)"}'
                            + (f'  {json.dumps(meta)}' if meta else ''))
        text = head + '\n\nRESPONSE:\n' + '\n'.join(body)
        return text[:budget + 4000]

    turns = t.turns
    task = [x for x in turns if x.kind == 'task'][:2]
    rest = [x for x in turns if x.kind != 'task']
    tail = rest[-TAIL_STEPS:]
    early = rest[:-TAIL_STEPS] if len(rest) > TAIL_STEPS else []
    lines = [_line(x) for x in early if x.kind != 'thinking' or ERRPAT.search(x.text or '')]
    tail_lines = [_line(x, detail=True) for x in tail]
    task_lines = [_line(x) for x in task]
    fixed = len(head) + sum(len(s) + 1 for s in task_lines + tail_lines) + 400
    room = max(2000, budget - fixed)
    if sum(len(s) + 1 for s in lines) > room:
        # keep error lines, the start and the end of the timeline; drop the middle
        keep = [i for i, s in enumerate(lines) if ERRPAT.search(s)]
        chosen, size = set(), 0
        for i in list(range(min(15, len(lines)))) + keep[:40] + list(range(max(0, len(lines) - 30), len(lines))):
            if i not in chosen and size + len(lines[i]) < room:
                chosen.add(i)
                size += len(lines[i]) + 1
        out, gap = [], 0
        for i, s in enumerate(lines):
            if i in chosen:
                if gap:
                    out.append(f'…[{gap} steps omitted]…')
                    gap = 0
                out.append(s)
            else:
                gap += 1
        if gap:
            out.append(f'…[{gap} steps omitted]…')
        lines = out
    n_calls = sum(1 for x in turns if x.kind == 'tool_call')
    return (head + f'\n\nTASK:\n' + '\n'.join(task_lines) +
            f'\n\nTIMELINE ({n_calls} tool calls; early steps clipped):\n' + '\n'.join(lines) +
            f'\n\nLAST {len(tail)} STEPS (detail):\n' + '\n'.join(tail_lines))

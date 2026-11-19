"""Pairwise judging (AA: "a judge ... blindly ranks two submissions for the same task"; "The initial task, all
reference files, and all submission files are parsed and provided as context to the judge"; the grading prompt
"asks the judge to determine which of Submission A and B better responds to the task"; ties count half).

AA's judge prompt is not published; ours is below. Inputs come from render.py manifests (text + page images). The
prompt retains the legacy per-file character shares and first-page image caps. Exact aggregate token admission
enforces the actual judge context, less the reserved output and template margin. Legacy rendered bytes are unchanged
when they fit this limit. Over-budget file context is excerpted with recorded limits; the task
and rubric stay verbatim. Every match is judged in both orders (A/B swapped), each a half-match.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from aa_evals.gdpval.render_cache import RenderIntegrityError
from aa_evals.gdpval.endpoint import GATE_WAIT_S, gate_retry_after, resolve

JUDGE_PROMPT_VERSION = 'gdpval-pairwise-v2-context-budget'

INSTRUCTIONS = """You are an expert reviewer of professional work. Below is a task that was given to a professional \
({occupation}, in the {sector} sector), the reference files that came with it, and two submissions made in response: \
Submission A and Submission B. Each submission's files were converted to text and to page images for you, so some \
formatting may be lost in conversion; judge the work itself.

Decide which submission better responds to the task, as an experienced professional in this occupation would judge \
it. Consider, in this order: whether it does what the task asks (every requested deliverable, in the requested form \
and file type), whether its content is correct and accurate given the task and the reference files, and the quality, \
completeness and professionalism of the work. A submission that is missing requested deliverables, or that submitted \
no files at all, should lose to one that delivers usable work. Do not prefer a submission because it is longer or \
because of its position (A or B). Answer "tie" only if the two are genuinely equal in quality.

Explain your reasoning briefly, then end your reply with exactly one line of JSON: {{"winner": "A"}}, \
{{"winner": "B"}} or {{"winner": "tie"}}."""


@dataclass
class Budget:
    ref_chars: int = 40_000      # all reference files together
    side_chars: int = 60_000     # each submission
    ref_images: int = 4
    side_images: int = 8
    context_margin: int = 256

    def describe(self):
        return dict(self.__dict__)


def _cut(text: str, n: int) -> tuple[str, bool]:
    if len(text) <= n:
        return text, False
    return text[:n] + f'\n[... truncated: {len(text) - n:,} more characters not shown]', True


def _pick_images(files: list[dict], cap: int) -> list[tuple[int, str]]:
    """(file index, image name): first pages of every file, round-robin, up to `cap`."""
    queues = [[(i, im) for im in f.get('images', [])] for i, f in enumerate(files)]
    out, k = [], 0
    while len(out) < cap and any(k < len(q) for q in queues):
        for q in queues:
            if k < len(q) and len(out) < cap:
                out.append(q[k])
        k += 1
    return out


def side_content(title: str, render_dir: Path | None, chars: int, n_images: int, stats: dict,
                 empty_note: str) -> list[dict]:
    """OpenAI content parts for one side (references or a submission)."""
    parts = [{'type': 'text', 'text': f'\n\n========== {title} ==========\n'}]
    man = None
    if render_dir is not None and (render_dir / 'manifest.json').exists():
        man = json.loads((render_dir / 'manifest.json').read_text())
        if man.get('render_integrity_error'):
            raise RenderIntegrityError(render_dir / 'manifest.json', man['render_integrity_error'])
    files = [f for f in (man or {}).get('files', []) if f.get('kind') != 'zip']
    if not files:
        parts[0]['text'] += empty_note + '\n'
        return parts
    share = max(2000, chars // len(files))
    picks = _pick_images(files, n_images)
    by_file: dict[int, list[str]] = {}
    for i, im in picks:
        by_file.setdefault(i, []).append(im)
    for i, f in enumerate(files):
        head = f'\n--- File: {f["name"]} ({f["kind"]}, {f["size"]:,} bytes' + (
            f', {f["n_pages"]} pages' if f.get('n_pages') else '') + ') ---\n'
        notes = list(f.get('notes') or [])
        n_img = len(f.get('images') or [])
        shown = by_file.get(i, [])
        if n_img:
            notes.append(f'{len(shown)} of {n_img} rendered page images shown' if len(shown) < n_img
                         else f'{n_img} page image(s) shown')
        text, cut = _cut(f.get('text') or '', share)
        stats['text_truncated'] = stats.get('text_truncated', 0) + int(cut)
        stats['images_dropped'] = stats.get('images_dropped', 0) + max(0, n_img - len(shown))
        body = head + ('Notes: ' + '; '.join(notes) + '\n' if notes else '') + (
            f'Text content:\n{text}\n' if text.strip() else '(no text content)\n')
        parts.append({'type': 'text', 'text': body})
        for im in shown:
            image_path = render_dir / 'img' / im
            raw = image_path.read_bytes()
            if not raw:
                raise RenderIntegrityError(image_path, 'selected legacy PNG is empty')
            data = base64.b64encode(raw).decode()
            parts.append({'type': 'text', 'text': f'[{f["name"]}: {im}]'})
            parts.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{data}'}})
    return parts


def build_messages(task, ref_dir: Path | None, a_dir: Path | None, b_dir: Path | None, budget: Budget,
                   images: bool, stats: dict) -> list[dict]:
    content = [{'type': 'text', 'text': INSTRUCTIONS.format(occupation=task.occupation, sector=task.sector)
                + '\n\n========== TASK ==========\n' + task.prompt}]
    ni = (lambda n: n if images else 0)
    content += side_content('REFERENCE FILES', ref_dir if task.reference_files else None, budget.ref_chars,
                            ni(budget.ref_images), stats, '(this task has no reference files)')
    content += side_content('SUBMISSION A', a_dir, budget.side_chars, ni(budget.side_images), stats,
                            '(Submission A submitted no files)')
    content += side_content('SUBMISSION B', b_dir, budget.side_chars, ni(budget.side_images), stats,
                            '(Submission B submitted no files)')
    content.append({'type': 'text', 'text': '\n\n========== END ==========\nWhich submission better responds to the '
                    'task? End with one line of JSON: {"winner": "A"}, {"winner": "B"} or {"winner": "tie"}.'})
    return [{'role': 'user', 'content': content}]


WIN_RE = re.compile(r'\{\s*"winner"\s*:\s*"(A|B|tie)"\s*\}', re.I)


def parse_winner(text: str) -> str | None:
    m = WIN_RE.findall(text or '')
    if not m:
        return None
    w = m[-1]
    return 'tie' if w.lower() == 'tie' else w.upper()


def context_excerpt(messages, fraction):
    """Keep task/rubric/end and section headings verbatim; shorten rendered files and first-page images equally
    within each side. Called only when the exact tokenizer says the original prompt exceeds its input budget."""
    out = copy.deepcopy(messages)
    for message in out:
        content = message['content']
        if isinstance(content, str):
            continue  # a plain user message is all mandatory task/rubric; never truncate it
        section = None
        image_counts = {}
        for part in content:
            if part['type'] == 'text' and part['text'].startswith('\n\n========== '):
                section = part['text'].split('\n')[2]
            if part['type'] == 'image_url':
                image_counts[section] = image_counts.get(section, 0) + 1
        section, seen = None, {}
        kept = []
        for i, part in enumerate(content):
            heading = part['type'] == 'text' and part['text'].startswith('\n\n========== ')
            if heading:
                section = part['text'].split('\n')[2]
            if part['type'] == 'text':
                if i == 0 or i == len(content) - 1 or heading:
                    kept.append(part)
                else:
                    text = part['text'][:int(len(part['text']) * fraction)]
                    if text:
                        kept.append({'type': 'text', 'text': text})
            elif part['type'] == 'image_url':
                seen[section] = seen.get(section, 0) + 1
                if seen[section] <= int(image_counts[section] * fraction):
                    kept.append(part)
        kept.insert(1, {'type': 'text', 'text': '[Rendered file context truncated by the recorded judge context budget.]'})
        message['content'] = kept
    return out


class ContextBudgetError(ValueError):
    """Mandatory task/rubric or output reservation cannot fit; retrying unchanged input cannot help."""


class Judge:
    """One judge arm on one endpoint. `extra` is the arm's request body (aa_evals/judges/arms.py)."""

    def __init__(self, spec: str, extra: dict, max_tries: int = 8, timeout: float = 1800.0, budget: Budget | None = None):
        self.spec = spec
        self.extra = dict(extra)
        self.max_tries = max_tries
        self.budget = budget or Budget()
        self.contexts = {}
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0))

    async def fit_context(self, ep, url, messages, stats):
        if url not in self.contexts:
            response = await self.http.get(f'{url}/models')
            response.raise_for_status()
            model = next(m for m in response.json()['data'] if m.get('id') == ep['served_name'])
            self.contexts[url] = int(model['max_model_len'])
        actual = self.contexts[url]
        reserve = int(self.extra.get('max_completion_tokens') or self.extra.get('max_tokens') or 32768)
        window = actual
        limit = window - reserve - self.budget.context_margin
        if limit <= 0:
            raise ContextBudgetError(f'judge output reserve {reserve} leaves no prompt room in context {window}')
        fraction, candidate = 1.0, messages
        for step in range(14):
            body = {'model': ep['served_name'], 'messages': candidate, 'add_generation_prompt': True}
            if self.extra.get('chat_template_kwargs'):
                body['chat_template_kwargs'] = self.extra['chat_template_kwargs']
            base = url[:-3] if url.endswith('/v1') else url
            response = await self.http.post(f'{base}/tokenize', json=body)
            lower_bound = response.status_code == 400 and any(t in response.text.lower() for t in ('context', 'too long'))
            if lower_bound:
                count = actual + 1  # tokenizer stopped at its endpoint limit: a lower bound, already too large
            else:
                response.raise_for_status()
                count = int(response.json()['count'])
            if step == 0:
                stats['initial_prompt_tokens'] = count
                stats['initial_prompt_tokens_lower_bound'] = lower_bound
            stats['context_budget'] = {'endpoint_context_tokens': actual, 'context_tokens': window,
                                       'output_reserve': reserve, 'margin': self.budget.context_margin,
                                       'input_limit': limit, 'prompt_tokens': count, 'token_count_lower_bound': lower_bound,
                                       'excerpt_fraction': fraction,
                                       'images_before': sum(p['type'] == 'image_url' for m in messages if isinstance(m['content'], list) for p in m['content']),
                                       'images_after': sum(p['type'] == 'image_url' for m in candidate if isinstance(m['content'], list) for p in m['content']),
                                       'policy': 'halve_rendered_text_and_images_per_side; preserve_task_and_rubric'}
            if count <= limit:
                return candidate
            if fraction == 0:
                raise ContextBudgetError(f'judge task/rubric alone exceeds the {limit}-token input budget')
            fraction = fraction / 2 if step < 12 else 0
            candidate = context_excerpt(messages, fraction)
        raise ContextBudgetError('judge context fitting did not converge')

    async def ask(self, messages: list[dict], *, stats: dict | None = None) -> dict:
        """{'winner', 'text', 'usage', 'tries', 'error'}; infra failures raise after max_tries."""
        stats = stats if stats is not None else {}
        last = None
        parse_fail = 0
        gate_wait = 0.0
        attempt = -1
        while attempt + 1 < self.max_tries:
            attempt += 1
            try:
                ep = resolve(self.spec)
                url = random.choice(ep['urls'])
                fitted = await self.fit_context(ep, url, messages, stats)
                body = {'model': ep['served_name'], 'messages': fitted, **self.extra}
                body = {k: v for k, v in body.items() if v is not None}
                r = await self.http.post(f'{url}/chat/completions', json=body)
                wait = gate_retry_after(r)
                if wait is not None and gate_wait < GATE_WAIT_S:   # admission gate backpressure: uses no try
                    gate_wait += wait
                    attempt -= 1
                    await asyncio.sleep(wait * (0.8 + 0.4 * random.random()))
                    continue
                if r.status_code == 400:
                    return {'winner': None, 'error': f'bad request: {r.text[:500]}', 'tries': attempt + 1}
                r.raise_for_status()
                d = r.json()
                ch = d['choices'][0]
                text = ch['message'].get('content') or ''
                w = parse_winner(text)
                if w is None or ch.get('finish_reason') == 'length':
                    parse_fail += 1
                    last = f'unparsed verdict (finish_reason={ch.get("finish_reason")})'
                    if parse_fail >= 3:
                        return {'winner': None, 'error': last, 'tries': attempt + 1, 'text': text[-2000:]}
                    continue
                return {'winner': w, 'text': text[-4000:], 'usage': d.get('usage'), 'tries': attempt + 1,
                        'error': None}
            except ContextBudgetError as e:
                return {'winner': None, 'error': str(e), 'tries': attempt + 1}
            except (httpx.HTTPError, KeyError, ValueError, OSError) as e:
                last = f'{type(e).__name__}: {str(e)[:300]}'
                await asyncio.sleep(min(300, 5 * 2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f'judge {self.spec}: {self.max_tries} tries failed: {last}')

    async def aclose(self):
        await self.http.aclose()


def score_for(order: str, winner: str) -> float:
    """The model's score in one judged order: order 'MA' = model is A (opponent B), 'MB' = model is B."""
    if winner == 'tie':
        return 0.5
    model_letter = 'A' if order == 'MA' else 'B'
    return 1.0 if winner == model_letter else 0.0

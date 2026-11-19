"""Per-criterion rubric judge.

AA: "GPT-5.6 Luna Medium judges every atomic criterion independently". The judge sees the task prompt, the
answer and one criterion, never the PDF or the contestant's identity, and "We accept a task grading only
when every criterion has a verdict." AA does not publish its judge prompt.

We use the judge prompt, verdict schema and strict parsing of Surge's harness (surge-ai/gdp-pdf@7a72a51,
src/gdp_pdf/scorer.py), with the task prompt added to the JSON object the judge reads, since AA's judge sees
it. A criterion passes iff its score rounds (half-up, 4 places) to >= 1. A judge output that fails to parse
is retried (5 tries in all); a criterion still without a verdict leaves the attempt ungraded (infra_error).
"""
import asyncio
import copy
import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from aa_evals.gdppdf.client import Client, InfraError

PARSE_TRIES = 5

JUDGE_PROMPT_TEMPLATE = """Read this json object:
{
  prompt: {{ task.prompt }},
  response: {{ rubric.response_to_grade }},
  criteria: {{ rubric.criterion }},
}

Score the response based on the criteria.
If the response perfectly satisfies the criteria, give it a score of "1".
If it does not satisfy the criteria at all, give it a score of "0".
Return the score and rationale in a JSON object with two keys, 'score' and 'rationale'.
'score' should be a string that's either "0" or "1".
'rationale' should be a description of why you gave the response that score.

Here are some examples of how to score a response:
{ score: "1", rationale: "The response is a perfect match for the criteria." }
{ score: "0", rationale: "The response does not satisfy the criteria at all." }
Unlike these examples, when you create a rationale, be specific and detailed. Give reasons specific to this response and criteria. Don't be generic."""

SCHEMA = {
    'type': 'object',
    'required': ['rationale', 'score'],
    'properties': {
        'rationale': {'type': 'string', 'description': 'Text describing the evaluation criterion'},
        'score': {'type': 'string', 'enum': ['0', '1'],
                  'description': "Score value: '0' (does not satisfy) or '1' (perfectly satisfies)"},
    },
    'additionalProperties': False,
}
RESPONSE_FORMAT = {'type': 'json_schema',
                   'json_schema': {'name': 'individual_criteria_score', 'schema': SCHEMA, 'strict': True}}

# Judge request settings. The user decided (2026-11-02, #34) against closed API judges: the judge is an open model
# we self-host. Settings are PINNED across all judged benchmarks (audit #41 issuecomment-6597330339, 2026-11-03):
#   Qwen3.8-27B judge: the registry's sampling (T 1.0, top_p 0.95, top_k 20) at reasoning_effort medium;
#   Super VL judge:    the registry's defaults (T 1.0, top_p 0.95, thinking on, full effort);
#   both: max_tokens 32768, and a reply cut at the cap is re-asked (up to PARSE_TRIES).
JUDGE_MAX_TOKENS = 32768
JUDGE_PRESETS = {
    # AA's own judge, kept for reference only: closed API judges are not approved (needs --allow-external-api).
    'openai:gpt-5.6-luna': {'reasoning_effort': 'medium', 'max_completion_tokens': 32768},
}
_REQUEST_KEYS = ('temperature', 'top_p', 'top_k', 'min_p', 'reasoning_effort', 'chat_template_kwargs')


class ParseError(ValueError):
    pass


def judge_prompt(task_prompt, response, criterion):
    return (JUDGE_PROMPT_TEMPLATE.replace('{{ task.prompt }}', task_prompt)
            .replace('{{ rubric.response_to_grade }}', response)
            .replace('{{ rubric.criterion }}', criterion))


def parse_verdict(raw):
    """Surge's strict parse: strip fences, JSON object with 'score' (number or string) and 'rationale' (str)."""
    cleaned = (raw or '').strip()
    cleaned = re.sub(r'\A```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```\Z', '', cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f'invalid JSON: {e}') from e
    if not isinstance(payload, dict) or 'score' not in payload or 'rationale' not in payload:
        raise ParseError("judge output must be an object with 'score' and 'rationale'")
    s = payload['score']
    if isinstance(s, bool) or not isinstance(s, (int, float, str)):
        raise ParseError("'score' must be a number or string")
    if not isinstance(payload['rationale'], str):
        raise ParseError("'rationale' must be a string")
    try:
        score = Decimal(str(s).strip())
    except InvalidOperation as e:
        raise ParseError(f"'score' is not a decimal: {s!r}") from e
    return score, payload['rationale']


def passes(score):
    return score.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP) >= 1


def judge_settings(spec, served_name='', registry_request=None, extra=None):
    """Pinned judge settings (see above): a preset by spec, else the serving registry's sampling and chat-template
    kwargs (Qwen at reasoning_effort medium), else temperature 0; max_tokens 32768; then `extra`."""
    if spec in JUDGE_PRESETS:
        out = copy.deepcopy(JUDGE_PRESETS[spec])
    else:
        overrides = (registry_request or {}).get('overrides') or {}
        out = copy.deepcopy({k: v for k, v in overrides.items() if k in _REQUEST_KEYS}) or {'temperature': 0.0}
        if 'qwen' in (served_name or '').lower():
            out['reasoning_effort'] = 'medium'
            out['chat_template_kwargs'] = dict(out.get('chat_template_kwargs') or {}, reasoning_effort='medium')
        out['max_tokens'] = JUDGE_MAX_TOKENS
    out.update(extra or {})
    return out


class Judge:
    def __init__(self, spec, concurrency=64, extra=None):
        self.spec = spec
        self.extra = extra or {}
        self.client = Client(spec, concurrency=concurrency, max_attempts=10)
        self.settings = None

    async def setup(self):
        ep = await self.client.endpoint()
        self.settings = judge_settings(self.spec, ep.served_name, ep.request, self.extra)
        return {'spec': self.spec, 'served_name': ep.served_name, 'settings': self.settings,
                'prompt': 'surge-ai/gdp-pdf@7a72a51 scorer.py template + task prompt', 'parse_tries': PARSE_TRIES}

    async def criterion(self, task_prompt, response, criterion):
        """{'result': pass|fail|unscored, 'score', 'rationale', 'tries', 'usage', 'error'}."""
        body = dict(self.settings, messages=[{'role': 'user', 'content': judge_prompt(task_prompt, response, criterion)}],
                    response_format=RESPONSE_FORMAT)
        usage, last = [], None
        for i in range(PARSE_TRIES):
            try:
                out = await self.client.chat(body, stream=False)
            except InfraError as e:
                return {'result': 'unscored', 'score': None, 'rationale': None, 'tries': i + 1, 'usage': usage,
                        'error': str(e)[:300]}
            usage.append(out.get('usage') or {})
            try:
                score, rationale = parse_verdict(out['content'])
            except ParseError as e:
                last = f'{e}; finish={out.get("finish_reason")}; raw={out["content"][:200]!r}'
                continue
            return {'result': 'pass' if passes(score) else 'fail', 'score': str(score), 'rationale': rationale,
                    'tries': i + 1, 'usage': usage, 'error': None}
        return {'result': 'unscored', 'score': None, 'rationale': None, 'tries': PARSE_TRIES, 'usage': usage,
                'error': f'parse failed {PARSE_TRIES}x: {last}'}

    async def grade(self, task, answer):
        """Grade one attempt: every criterion in parallel."""
        verdicts = await asyncio.gather(*(self.criterion(task.prompt, answer, c.text) for c in task.criteria))
        grades = [dict(v, id=c.id) for c, v in zip(task.criteria, verdicts)]
        unscored = sum(g['result'] == 'unscored' for g in grades)
        passed = sum(g['result'] == 'pass' for g in grades)
        n = len(grades)
        return {'grades': grades, 'n_criteria': n, 'n_passed': passed, 'n_unscored': unscored,
                'all_pass': None if unscored else float(passed == n),
                'mean_pass': None if unscored else passed / n}

    async def close(self):
        await self.client.close()

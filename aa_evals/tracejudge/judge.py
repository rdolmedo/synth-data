"""The LLM pass: one chat call per digest, JSON verdict back.

The client is aa_evals.hle.client.ChatClient (serving:<name> follows restarts; transport errors and 5xx retried).
Sampling comes from the serving registry's request.overrides; --effort and --thinking override the reasoning knobs.
"""
from __future__ import annotations

import json
import re
import time

from aa_evals.tracejudge.trace import CATEGORIES

PROMPT_VERSION = 'tj-v5'

SYSTEM = """You audit ONE attempt from an LLM benchmark run. Your only job: decide whether INFRASTRUCTURE failed \
during this attempt, that is, whether something outside the model (sandbox, container, network, serving endpoint, \
harness, chat template, tool-call parser, run scheduler, verifier/grader) broke in a way that could have changed \
the attempt's outcome. You are not grading the model.

Infra categories:
{cats}

Do NOT flag (these are model outcomes, protocol limits or benchmark properties):
- wrong answers, bad code, failing tests, giving up, refusing, "I don't know", hallucinations, slow solutions;
- the model running out of context window, hitting the per-turn output cap, or the agent/episode time limit;
- the sandbox memory cap killing a process THE MODEL started (that is the box's memory limit, like any real \
machine); only a kill of the agent itself (pi/node) or of the whole sandbox is infra. This exception is for \
memory only: a full disk, fork failures or fd exhaustion are infra (sandbox_kill) unless the model's own writes or \
processes visibly caused them;
- the model killing its own agent or shell: when the agent exits on a signal (exit -9, -15, 137, 143) right \
after the model ran pkill -f / kill / killall (look at the last CALL steps, including one marked in flight), the \
model did it, not the sandbox;
- errors in the model's own code or commands (tracebacks, typos, permission errors on files the task never \
granted), and tools or packages missing from the task image (python: command not found, ModuleNotFoundError);
- an exception or error message returned AS A TOOL RESULT by the benchmark's tools or simulated API (e.g. \
"'list' object has no attribute 'split'" after the model passed a list where the API takes a string): that is the \
environment answering the model's call, not the harness. harness_error is only for OUR harness: the client, \
runner, chat template or tool-call parser (an error in the pipeline record or harness events, or the model's \
call never reaching the tool);
- failed external tool internet calls when the pipeline explicitly records agent_internet=false or \
agent_network.mode=offline: external egress is intentionally denied, including the proxy path. Do not infer \
that policy from an error message alone. It does NOT explain broken task-local servers/services, the model \
endpoint or relay, or harness/evaluator networking; those failures remain reviewable;
- "Network is unreachable" / DNS failures on direct connections in a sandbox without direct egress, when HTTP(S) \
through the proxy works. Without an explicit offline-agent policy, a proxy error (503, refused, tunnel failed) \
can indicate infra and must be assessed from the trace;
- a verifier timeout on the model's work, or tests failing on the model's output; a crash (segfault, abort) \
under the verifier whose crashing frame is the model's solution (/app or the task's workdir, or a module or C \
extension the agent built) is the solution failing; grader_crash is only a crash in the test harness itself \
(the test or verifier code, pytest, the reward writer);
- a response that ends inside its reasoning or runs out of max tokens at the protocol cap;
- a dynamic per-request output budget that is smaller than the total context window because the prompt \
already uses part of that window. Compare a length finish to its own request budget, not the total window. \
Agentic trajectory token totals can span multiple turns and summaries and cannot establish a per-call cap hit. \
If the exact request budget is absent, do not invent it or treat the pre-filter's truncation label as proof;
- benchmark flaws (ambiguous task, wrong reference answer, broken task environment that the oracle also fails).
Signatures that ARE infra even though they look quiet:
- an empty answer with 0 (or almost 0) completion tokens, or no finish_reason at all: the model never answered \
(silent_zero); an agent that ends right after an empty model turn (no text, no tool call) is the same;
- finish_reason "length" at a completion count far below a known applicable per-response output budget, \
or a request budget unexpectedly below the configured policy: the harness cut the model off (harness_cutoff). \
An empty final text after nonempty reasoning that exhausts its normal output room is a protocol outcome;
- raw tool-call markup (<tool_call>, <function=...>, a JSON {{"name": ..., "arguments": ...}}) in the model's TEXT \
with no tool executed afterwards: the tool-call parser failed (harness_error);
- chat-template tokens (<|im_start|>, <|im_end|>, <start_of_turn>, ...) in the answer (harness_error);
- repeated connection errors / 5xx / gateway errors from the model endpoint (network).
Pipeline records marked infra_error / infrastructure_failure are already known; confirm them if the evidence \
agrees. The pre-filter hits are hints from string rules and can be wrong in both directions.

Answer with JSON only:
{{"flags": [{{"category": one of {names}, "evidence": "<a short quote from the digest>", "ref": "<the [ref] it \
came from>", "confidence": <0..1>}}], "reason": "<one or two sentences>"}}
Return "flags": [] when nothing infra-related went wrong. Flag only with concrete evidence from the digest; \
use confidence below 0.5 when unsure."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'flags': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {'category': {'type': 'string', 'enum': list(CATEGORIES)},
                           'evidence': {'type': 'string', 'maxLength': 400},
                           'ref': {'type': 'string', 'maxLength': 80},
                           'confidence': {'type': 'number'}},
            'required': ['category', 'evidence', 'ref', 'confidence']}},
        'reason': {'type': 'string', 'maxLength': 600},
    },
    'required': ['flags', 'reason'],
}


def system_prompt() -> str:
    cats = '\n'.join(f'- {k}: {v}' for k, v in CATEGORIES.items())
    return SYSTEM.format(cats=cats, names=json.dumps(list(CATEGORIES)))


def parse(content: str) -> dict | None:
    if not content:
        return None
    s = content.strip()
    s = re.sub(r'^```(?:json)?|```$', '', s, flags=re.M).strip()
    for cand in (s, (re.search(r'\{.*\}', s, re.S) or [None])[0]):
        if not cand:
            continue
        try:
            d = json.loads(cand)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get('flags'), list):
            flags = []
            for f in d['flags']:
                if isinstance(f, dict) and f.get('category') in CATEGORIES:
                    try:
                        c = float(f.get('confidence', 0.5))
                    except (TypeError, ValueError):
                        c = 0.5
                    flags.append({'category': f['category'], 'evidence': str(f.get('evidence', ''))[:400],
                                  'ref': str(f.get('ref', ''))[:80], 'confidence': max(0.0, min(1.0, c))})
            return {'flags': flags, 'reason': str(d.get('reason', ''))[:600]}
    return None


class LLMJudge:
    def __init__(self, spec: str, *, effort: str | None = None, thinking: str | None = None,
                 max_tokens: int = 8192, structured: bool = True, timeout: float = 900,
                 save_raw: bool = False):
        from aa_evals.hle.client import ChatClient
        self.spec, self.effort, self.thinking = spec, effort, thinking
        self.max_tokens, self.structured = max_tokens, structured
        self.save_raw = save_raw
        self.client = ChatClient(spec, timeout=timeout, max_attempts=8)
        self.system = system_prompt()

    def describe(self) -> dict:
        ep = self.client.endpoint
        return {'spec': self.spec, 'served': getattr(ep, 'served_name', None), 'effort': self.effort,
                'thinking': self.thinking, 'max_tokens': self.max_tokens, 'prompt_version': PROMPT_VERSION}

    def _body(self, digest: str) -> dict:
        ep = self.client._current()
        over = json.loads(json.dumps((ep.request or {}).get('overrides') or {}))
        ctk = over.setdefault('chat_template_kwargs', {})
        if self.effort:
            over['reasoning_effort'] = self.effort
            if 'reasoning_effort' in ctk:
                ctk['reasoning_effort'] = self.effort
        if self.thinking in ('on', 'off'):
            ctk['enable_thinking'] = self.thinking == 'on'
            if self.thinking == 'off':
                over.pop('reasoning_effort', None)
                ctk.pop('reasoning_effort', None)
        if not ctk:
            over.pop('chat_template_kwargs')
        body = {**over, 'messages': [{'role': 'system', 'content': self.system},
                                     {'role': 'user', 'content': digest}],
                'max_tokens': self.max_tokens, 'seed': 0}
        if self.structured:
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': 'infra_verdict', 'schema': SCHEMA}}
        return body

    async def judge(self, digest: str) -> dict:
        from aa_evals.hle.client import InfraError, message_fields
        t0 = time.time()
        tries = []

        def totals():
            # A missing usage field is unknown, even if another response has usage.
            return {key: sum(t[key] for t in tries) if all(type(t.get(key)) is int and t[key] >= 0
                                                        for t in tries) else None
                    for key in ('prompt_tokens', 'completion_tokens')}

        for attempt in range(2):   # one re-ask when the JSON doesn't parse
            call = {'call': attempt + 1, 'max_tokens': self.max_tokens, 'request_body_created': False}
            try:
                body = self._body(digest)
                call['request_body_created'] = True
                data, _ = await self.client.chat(body)
            except InfraError as e:
                tries.append({**call, 'error_kind': 'InfraError', 'finish_reason': None, 'usage': None,
                              'prompt_tokens': None, 'completion_tokens': None})
                return {'ok': False, 'error': f'judge endpoint: {str(e)[:300]}', 'latency_s': time.time() - t0,
                        'calls': tries, **totals()}
            content, reasoning, finish, usage = message_fields(data)
            v = parse(content)
            call.update({'finish_reason': finish, 'response_id': data.get('id'), 'usage': data.get('usage'),
                         'prompt_tokens': (usage or {}).get('prompt_tokens'),
                         'completion_tokens': (usage or {}).get('completion_tokens')})
            if self.save_raw:
                # Only the returned model JSON: no client, request headers, endpoint or environment.
                call.update({'content': content, 'reasoning': reasoning, 'raw_response': data})
            tries.append(call)
            if v is not None:
                return {'ok': True, **v, 'latency_s': round(time.time() - t0, 2), 'calls': tries,
                        **totals()}
        return {'ok': False, 'error': f'unparseable verdict: {str(content)[:300]}', 'calls': tries,
                'latency_s': round(time.time() - t0, 2),
                **totals()}

    async def close(self):
        await self.client.close()

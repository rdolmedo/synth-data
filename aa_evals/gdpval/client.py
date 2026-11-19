"""Stirrup's ChatCompletionsClient adapted to our vLLM endpoints and the M3.0 context cap.

Stirrup itself is unmodified; this subclass only changes how one request is sent:
- **Context cap.** Every request is counted with vLLM's /tokenize (the same messages, tools and template kwargs),
  and `max_completion_tokens` = min(output cap, cap - prompt). A prompt within `MIN_ROOM` of the cap raises Stirrup's
  ContextOverflowError, so the agent compacts as it would on a provider whose context is the cap. A reply cut at the
  remaining room is the same overflow; a reply cut at the output cap (below the room) is Stirrup's
  OutputTokenLimitError, which ends the attempt.
- **vLLM's reasoning field** (`reasoning`, older `reasoning_content`) becomes Stirrup's ReasoningBlock, which Stirrup
  sends back as `reasoning_content` on later turns. Frozen registry history adapters mirror it for vLLM or
  inline it for Nemotron; both token counting and generation use the adapted messages.
- **Request settings** from the run config: the registry's `request.overrides` merged with the effort (protocol.py),
  sent as `extra_body` except the OpenAI top-level sampling fields.
- **Infra.** Transport errors, timeouts and 408/409/429/5xx are retried with backoff (re-resolving a `serving:` spec
  every try). After `max_tries` the attempt raises InfraFailure, which the runner records as infra (never scored).
"""
from __future__ import annotations

import asyncio
import copy
import json
import random
import zlib
import uuid
from time import perf_counter
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, BadRequestError

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import AssistantMessage, ReasoningBlock, TextBlock, TokenUsage, ToolCall

from aa_evals.gdpval.endpoint import GATE_WAIT_S, InfraFailure, gate_retry_after, resolve  # noqa: F401  (re-exported)

MIN_ROOM = 1024
TOP_LEVEL = {'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'seed', 'reasoning_effort', 'stop', 'n'}
CONTEXT_TEXT = ('maximum context length', 'context length', 'too many tokens', 'is too long', 'exceeds the model')


def split_body(body: dict) -> tuple[dict, dict]:
    """(top-level OpenAI kwargs, extra_body) of a request body."""
    top = {k: v for k, v in body.items() if k in TOP_LEVEL and v is not None}
    extra = {k: v for k, v in body.items() if k not in TOP_LEVEL and k not in ('max_tokens', 'max_completion_tokens')}
    return top, extra


HISTORY_ADAPTERS = ('mirror_reasoning_content', 'inline_reasoning_history')


def history_adapters(request: dict) -> list[str]:
    """Freeze the registry's declared adapters, with the usual vLLM compatibility default."""
    adapters = list(request.get('adapters') or ['mirror_reasoning_content'])
    unknown = set(adapters).difference(HISTORY_ADAPTERS)
    if unknown:
        raise ValueError(f'unknown GDPval history adapters: {sorted(unknown)}')
    return adapters


def adapt_history(messages: list[dict], adapters) -> list[dict]:
    """Adapt only the served copy; tokenizer and completion receive this exact payload."""
    unknown = set(adapters).difference(HISTORY_ADAPTERS)
    if unknown:
        raise ValueError(f'unknown GDPval history adapters: {sorted(unknown)}')
    messages = copy.deepcopy(messages)
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        if 'mirror_reasoning_content' in adapters and message.get('reasoning_content') is not None:
            message.setdefault('reasoning', message['reasoning_content'])
        if 'inline_reasoning_history' in adapters:
            reasoning = message.get('reasoning_content') or message.get('reasoning')
            if isinstance(reasoning, str) and reasoning.strip():
                prefix = '<think>\n' + reasoning + '</think>'
                content = message.get('content') or ''
                message['content'] = ([{'type': 'text', 'text': prefix}, *content]
                                      if isinstance(content, list) else prefix + content)
            message.pop('reasoning_content', None)
            message.pop('reasoning', None)
    return messages


class CappedVLLMClient(ChatCompletionsClient):
    def __init__(self, *, spec: str, served_name: str, body: dict, context_cap: int, max_output: int,
                 pin_key: str = '', max_tries: int = 12, timeout: float = 3600.0, stats: dict | None = None,
                 adapters: list[str] | tuple[str, ...] = ()):
        adapt_history([], adapters)   # fail closed before constructing HTTP clients
        super().__init__(model=served_name, max_tokens=min(max_output, context_cap), context_window_tokens=context_cap,
                         base_url='http://unused.invalid/v1', api_key='EMPTY', max_retries=0)
        self.adapters = tuple(adapters)
        self.spec = spec
        self.body = dict(body)
        self.cap = context_cap
        self.max_output = min(max_output, context_cap)
        self.pin_key = pin_key
        self.conversation_id = uuid.uuid4().hex
        self.max_tries = max_tries
        self.timeout = timeout
        self.stats = stats if stats is not None else {}
        self._clients: dict[str, AsyncOpenAI] = {}
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        self._last = None   # (prompt + completion tokens, prompt chars) of the previous request

    # one replica per attempt (prefix cache), re-resolved on every try so a restarted server is picked up
    def _pick(self) -> str:
        urls = resolve(self.spec)['urls']
        url = urls[zlib.crc32(self.pin_key.encode()) % len(urls)].rstrip('/')
        self.stats.setdefault('urls', {})
        self.stats['urls'][url] = self.stats['urls'].get(url, 0) + 1
        return url

    def _client_for(self, url: str) -> AsyncOpenAI:
        if url not in self._clients:
            self._clients[url] = AsyncOpenAI(base_url=url, api_key='EMPTY', timeout=self.timeout, max_retries=0)
        return self._clients[url]

    async def _count(self, url: str, msgs: list, tools: list | None) -> int | None:
        _, extra = split_body(self.body)
        req: dict[str, Any] = {'model': self._model, 'messages': msgs, 'add_generation_prompt': True}
        if tools:
            req['tools'] = tools
        if extra.get('chat_template_kwargs'):
            req['chat_template_kwargs'] = extra['chat_template_kwargs']
        base = url[:-3] if url.endswith('/v1') else url
        r = await self._http.post(f'{base}/tokenize', json=req)
        if r.status_code == 200:
            return int(r.json()['count'])
        if r.status_code >= 500 or r.status_code in (408, 409, 429):
            raise httpx.HTTPStatusError('tokenize', request=r.request, response=r)
        # a 400 here is usually the prompt itself being too long for the server's own window
        if any(t in r.text.lower() for t in CONTEXT_TEXT):
            raise ContextOverflowError(r.text[:500])
        self.stats['tokenize_unavailable'] = self.stats.get('tokenize_unavailable', 0) + 1
        self.stats['tokenize_last_error'] = r.text[:300]
        return self._estimate(msgs)

    def _estimate(self, msgs: list) -> int:
        """Fallback prompt size: the previous request's prompt + reply, plus ~1 token per 3 chars of what was
        appended since (an over-estimate for code and prose, so the cap still holds)."""
        chars = len(json.dumps(msgs, ensure_ascii=False))
        if self._last is None:
            return chars // 3
        prev_tokens, prev_chars = self._last
        return prev_tokens + max(0, chars - prev_chars) // 3

    async def generate(self, messages, tools):  # noqa: C901
        msgs = adapt_history(to_openai_messages(messages), self.adapters)
        otools = to_openai_tools(tools) if tools else None
        top, extra = split_body(self.body)
        last_exc = None
        attempt = -1
        while True:
            attempt += 1
            if attempt >= self.max_tries:
                raise InfraFailure(f'{self.max_tries} tries failed: {type(last_exc).__name__}: {str(last_exc)[:500]}')
            try:
                url = self._pick()
                prompt = await self._count(url, msgs, otools)
                room = self.cap - prompt if prompt is not None else self.max_output
                if room < MIN_ROOM:
                    self.stats['context_overflows'] = self.stats.get('context_overflows', 0) + 1
                    raise ContextOverflowError(f'prompt {prompt} tokens leaves {room} of the {self.cap}-token cap')
                max_tok = min(self.max_output, room)
                req: dict[str, Any] = {'model': self._model, 'messages': msgs, 'max_completion_tokens': max_tok,
                                       'extra_headers': {'X-Conversation-Id': self.conversation_id,
                                                         'X-Gate-Client': 'gdpval'},
                                       **top}
                if extra:
                    req['extra_body'] = extra
                if otools:
                    req['tools'] = otools
                    req['tool_choice'] = 'auto'
                t0 = perf_counter()
                response = await self._client_for(url).chat.completions.create(**req)
                t1 = perf_counter()
                break
            except ContextOverflowError:
                raise
            except BadRequestError as e:
                text = str(e).lower()
                if e.code == 'context_length_exceeded' or any(t in text for t in CONTEXT_TEXT):
                    self.stats['context_overflows'] = self.stats.get('context_overflows', 0) + 1
                    raise ContextOverflowError(str(e)[:500]) from e
                last_exc = e   # a malformed request: deterministic, retried a couple of times only
                if attempt >= 2:
                    raise InfraFailure(f'bad request: {str(e)[:500]}') from e
            except (APIConnectionError, APITimeoutError, httpx.TransportError, httpx.HTTPStatusError, InfraFailure,
                    OSError, ValueError) as e:
                last_exc = e
            except APIStatusError as e:
                wait = gate_retry_after(e.response)
                if wait is not None and self.stats.get('gate_wait_s', 0) < GATE_WAIT_S:
                    # the admission gate's backpressure: the server is busy, not failing. Uses no try.
                    self.stats['gate_rejections'] = self.stats.get('gate_rejections', 0) + 1
                    self.stats['gate_wait_s'] = self.stats.get('gate_wait_s', 0) + wait
                    attempt -= 1
                    await asyncio.sleep(wait * (0.8 + 0.4 * random.random()))
                    continue
                if e.status_code not in (408, 409, 429) and e.status_code < 500:
                    raise InfraFailure(f'HTTP {e.status_code}: {str(e)[:500]}') from e
                last_exc = e
            self.stats['infra_retries'] = self.stats.get('infra_retries', 0) + 1
            await asyncio.sleep(min(300, 5 * 2 ** attempt) * (0.5 + random.random()))

        choice = response.choices[0]
        usage = response.usage
        self.stats['requests'] = self.stats.get('requests', 0) + 1
        if usage:
            self._last = ((usage.prompt_tokens or 0) + (usage.completion_tokens or 0),
                          len(json.dumps(msgs, ensure_ascii=False)))
            self.stats['max_context_tokens'] = max(self.stats.get('max_context_tokens', 0),
                                                   (usage.prompt_tokens or 0) + (usage.completion_tokens or 0))
            self.stats['output_tokens'] = self.stats.get('output_tokens', 0) + (usage.completion_tokens or 0)
        if choice.finish_reason in ('max_tokens', 'length'):
            if max_tok < self.max_output:     # cut at the remaining room: the context is full
                self.stats['context_overflows'] = self.stats.get('context_overflows', 0) + 1
                raise ContextOverflowError(f'reply cut at the remaining {max_tok} tokens of the {self.cap} cap')
            self.stats['output_cap_hits'] = self.stats.get('output_cap_hits', 0) + 1
            raise OutputTokenLimitError(model_slug=self._model, max_tokens=max_tok,
                                        provider_reason=choice.finish_reason)
        msg = choice.message
        blocks = []
        extra_fields = getattr(msg, 'model_extra', None) or {}
        reasoning = (getattr(msg, 'reasoning_content', None) or extra_fields.get('reasoning_content')
                     or extra_fields.get('reasoning'))
        if reasoning:
            blocks.append(ReasoningBlock(content=reasoning))
        if msg.content:
            blocks.append(TextBlock(text=msg.content))
        blocks.extend(ToolCall.from_provider(provider_id=tc.id, name=tc.function.name,
                                             arguments=tc.function.arguments or '')
                      for tc in (msg.tool_calls or []))
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0
        rtok = 0
        if usage and getattr(usage, 'completion_tokens_details', None):
            rtok = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0) or 0
        return AssistantMessage(blocks=blocks, token_usage=TokenUsage(input=inp, answer=out - rtok, reasoning=rtok),
                                request_start_time=t0, request_end_time=t1)

    async def aclose(self):
        await self._http.aclose()
        for c in self._clients.values():
            await c.close()

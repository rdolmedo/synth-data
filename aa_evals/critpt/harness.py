"""One CritPt trajectory in Stirrup (AA's open-source harness), against an OpenAI-compatible endpoint.

Pieces on top of stock Stirrup 0.2.1:
- `EndpointClient`: Stirrup's ChatCompletionsClient with (a) our own retry loop for transient API failures (AA retries
  up to 30 times), (b) vLLM's reasoning field read under both names, (c) vLLM's context-length 400 mapped to Stirrup's
  ContextOverflowError so the agent compacts as it would at a provider, (d) the context cap (M3.0: 128k): every
  request's max_completion_tokens is clamped to the room left under `context_window_tokens`, which may be below the
  server's max_model_len, plus an optional extra per-response output cap, and (e) the endpoint's history adapters
  (earlier turns' reasoning sent as `reasoning`, or inline as <think>...</think> for Nemotron).
- `OfflineSandbox`: Stirrup's local code-exec provider, with every command run inside the CritPt SIF (Python 3.13 +
  numpy/scipy/sympy/matplotlib), contained and without network. Only the trial's work dir is bound in.
- `finish(reason, derivation, answer)`: AA's three finish fields.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import random
import re
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import openai
from pydantic import BaseModel, Field
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import (AssistantMessage, ReasoningBlock, TextBlock, TokenUsage, Tool, ToolCall, ToolResult,
                                 ToolUseCountMetadata)
from stirrup.tools.code_backends.base import CommandResult
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.utils.logging import AgentLoggerBase

LOG = logging.getLogger('critpt')

MAX_TURNS = 50                 # AA: at most 50 turns, one per response
CMD_TIMEOUT = 300              # Stirrup's default per-command limit (AA's is unpublished)
SANDBOX_MEM_KB = 16 * 1024 * 1024
SANDBOX_THREADS = 4


class InfraError(Exception):
    """The endpoint or the sandbox failed; the trajectory must be rerun, never scored."""


class RunawayError(Exception):
    """A response outlived the per-request timeout while the server stayed healthy: a runaway generation, which is a
    model outcome (model_error), never infra."""


REQUEST_TIMEOUT = 8 * 3600     # long on purpose: max_tokens and the turn cap bound a response, not the clock


# ----------------------------------------------------------------------------------------------- model client

_CTX_RE = re.compile(r'maximum context length is (\d+) tokens.*?(?:request has|messages resulted in|prompt contains|'
                     r'your request has) (\d+) (?:input )?tokens', re.S | re.I)


class LoadGate:
    """Requests in flight on a shared server, with a limit that backs off while the server queues.

    Every `poll_s` it reads vLLM's /metrics: num_requests_waiting (summed over engines and replicas) and
    kv_cache_usage_perc (the highest). The server counts as queueing when more than `queue_frac` of its running
    requests are waiting (a busy production server always admits with a few requests briefly waiting), or when it
    preempts more than `preempt_frac` of its running requests per poll (num_preemptions_total). A trickle of
    preemptions is the steady state of a KV-bound vLLM whose admitted requests keep growing their contexts; only a
    higher rate is thrash. The limit
    starts at `start` (default `max_limit`). If the server has been queueing for `backoff_after_s`, it halves (down
    to `min_limit`). After `recover_after_s` without queueing and with KV usage below `kv_ceiling`, it grows by
    `step`, up to `max_limit`. Changes apply as requests finish; none is cancelled.

    `override_path` (optional): a JSON file read every poll; {"max_limit": N, "min_limit": N, "limit": N} adjusts
    the gate without a restart.
    """

    def __init__(self, max_limit: int, urls_fn, *, min_limit: int = 4, poll_s: float = 30.0,
                 backoff_after_s: float = 120.0, recover_after_s: float = 600.0, step: int = 8,
                 start: int | None = None, kv_ceiling: float = 0.9, queue_frac: float = 0.01,
                 preempt_frac: float = 0.1,
                 override_path: Path | None = None) -> None:
        self.max_limit, self.min_limit, self.step = max_limit, min_limit, step
        self.limit = min(max_limit, start or max_limit)
        self.kv_ceiling, self.queue_frac, self.preempt_frac = kv_ceiling, queue_frac, preempt_frac
        self._override_path, self._override_mtime = override_path, None
        self._urls_fn, self._poll_s = urls_fn, poll_s
        self._backoff_after_s, self._recover_after_s = backoff_after_s, recover_after_s
        self._inflight = 0
        self._cond = asyncio.Condition()
        self.history: list[dict] = []          # limit changes, for the summary

    async def __aenter__(self):
        async with self._cond:
            await self._cond.wait_for(lambda: self._inflight < self.limit)
            self._inflight += 1

    async def __aexit__(self, *exc):
        async with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    @staticmethod
    def _load(urls: list[str]) -> tuple[float | None, float, float, float]:
        """(requests waiting, highest KV usage, requests running, preemptions so far); waiting is None when no
        metrics could be read."""
        import urllib.request
        total, seen, kv, running, preempt = 0.0, False, 0.0, 0.0, 0.0
        for u in urls:
            base = u.rstrip('/').removesuffix('/v1')
            try:
                with urllib.request.urlopen(f'{base}/metrics', timeout=10) as r:
                    for line in r.read().decode().splitlines():
                        if line.startswith('vllm:num_requests_waiting{'):
                            total += float(line.rsplit(' ', 1)[1])
                            seen = True
                        elif line.startswith('vllm:num_preemptions_total{'):
                            preempt += float(line.rsplit(' ', 1)[1])
                        elif line.startswith('vllm:num_requests_running{'):
                            running += float(line.rsplit(' ', 1)[1])
                        elif line.startswith('vllm:kv_cache_usage_perc{'):
                            kv = max(kv, float(line.rsplit(' ', 1)[1]))
            except Exception:  # noqa: BLE001 - metrics unavailable: no signal this round
                pass
        return (total if seen else None), kv, running, preempt

    async def _set(self, new: int, why: str) -> None:
        async with self._cond:
            if new != self.limit:
                LOG.warning('load gate: limit %d -> %d (%s)', self.limit, new, why)
                self.history.append({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'from': self.limit, 'to': new,
                                     'why': why})
                self.limit = new
                self._cond.notify_all()

    async def _read_override(self) -> None:
        import json
        p = self._override_path
        try:
            if p is None or not p.exists() or p.stat().st_mtime == self._override_mtime:
                return
            self._override_mtime = p.stat().st_mtime
            o = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001 - a torn or bad file: ignore it this round
            LOG.warning('load gate override %s unreadable: %s', p, e)
            return
        self.max_limit = int(o.get('max_limit', self.max_limit))
        self.min_limit = int(o.get('min_limit', self.min_limit))
        new = int(o.get('limit', self.limit))
        await self._set(max(self.min_limit, min(self.max_limit, new)), f'override {o}')

    async def run(self) -> None:
        waiting_since = calm_since = last_preempt = None
        while True:
            try:
                urls = self._urls_fn()
            except Exception:  # noqa: BLE001
                urls = []
            await self._read_override()
            w, kv, running, preempt = (await asyncio.to_thread(self._load, urls) if urls
                                       else (None, 0.0, 0.0, 0.0))
            now = time.time()
            preempted = last_preempt is not None and preempt - last_preempt > self.preempt_frac * max(running, 1)
            if w is not None:
                last_preempt = preempt
            if w is None:
                pass
            elif w > self.queue_frac * running or preempted:
                calm_since = None
                waiting_since = waiting_since or now
                if now - waiting_since >= self._backoff_after_s and self.limit > self.min_limit:
                    await self._set(max(self.min_limit, self.limit // 2),
                                    f'{w:.0f} waiting, {preempt:.0f} preemptions so far, pressure for '
                                    f'{now - waiting_since:.0f}s')
                    waiting_since = now
            elif kv >= self.kv_ceiling:
                waiting_since = calm_since = None          # no queue but no KV headroom: hold
            else:
                waiting_since = None
                calm_since = calm_since or now
                if now - calm_since >= self._recover_after_s and self.limit < self.max_limit:
                    await self._set(min(self.max_limit, self.limit + self.step),
                                    f'no queue for {now - calm_since:.0f}s, KV {kv:.2f}')
                    calm_since = now
            await asyncio.sleep(self._poll_s)


def apply_history_adapters(msgs: list[dict], adapters) -> list[dict]:
    """Earlier assistant turns' reasoning as the served template expects it (the endpoint's request profile;
    same adapters as aa_evals/automationbench and synth's sticky proxy). Stirrup sends it as `reasoning_content`, which
    vLLM >= 0.31 drops from request messages:
    - mirror_reasoning_content: also send it as `reasoning`, the field vLLM maps into the template (Qwen, DeepSeek);
    - inline_reasoning_history: put it back in the content as <think>...</think> (Nemotron's template)."""
    for m in msgs:
        if m.get('role') != 'assistant':
            continue
        if 'inline_reasoning_history' in adapters:
            r = m.pop('reasoning_content', None) or m.pop('reasoning', None)
            m.pop('reasoning', None)
            if isinstance(r, str) and r.strip():
                prefix = '<think>\n' + r + '</think>'
                content = m.get('content') or ''
                m['content'] = ([{'type': 'text', 'text': prefix}] + list(content) if isinstance(content, list)
                                else prefix + content)
        elif 'mirror_reasoning_content' in adapters:
            if m.get('reasoning_content') is not None and 'reasoning' not in m:
                m['reasoning'] = m['reasoning_content']
    return msgs


class EndpointClient(ChatCompletionsClient):
    """ChatCompletionsClient for our vLLM endpoints (see module docstring)."""

    def __init__(self, *, model: str, base_urls: list[str], context_window_tokens: int,
                 max_output_tokens: int | None, sampling: dict[str, Any], request_timeout: float = REQUEST_TIMEOUT,
                 max_attempts: int = 30, resolve=None, endpoint_wait_s: float = 7200.0,
                 gate: LoadGate | None = None, adapters: list[str] | tuple[str, ...] = ('mirror_reasoning_content',)) -> None:
        super().__init__(model=model, max_tokens=context_window_tokens, context_window_tokens=context_window_tokens,
                         base_url=base_urls[0], api_key='EMPTY', timeout=request_timeout, max_retries=0)
        self._timeout = request_timeout
        self._set_urls(base_urls)
        self._rr = random.randrange(len(self._clients))
        self._cap = max_output_tokens
        self._sampling = dict(sampling)
        self._ctx = context_window_tokens
        self._adapters = tuple(adapters)
        self._last: tuple[int, str, int] | None = None   # (messages sent, first message digest, prompt + completion)
        self.n_cap_overshoot = 0
        self._max_attempts = max_attempts
        self._resolve = resolve                 # () -> current urls, or None while the server is down
        self._endpoint_wait_s = endpoint_wait_s
        self._gate = gate
        self.n_retries = 0
        # Every response's usage, including one cut off by the length limit (which never reaches the history).
        self.usage = {'responses': 0, 'input': 0, 'output': 0, 'reasoning': 0}
        self.down_seconds = 0.0

    def _set_urls(self, urls: list[str]) -> None:
        self._urls = list(urls)
        self._clients = [openai.AsyncOpenAI(base_url=u, api_key='EMPTY', timeout=self._timeout, max_retries=0)
                         for u in urls]

    async def _wait_for_endpoint(self) -> None:
        """After a failure: follow the server if it moved; while it is down (no fresh registry), wait without
        spending attempts, up to endpoint_wait_s in total (a server restart must not burn trajectories)."""
        if self._resolve is None:
            return
        while True:
            try:
                urls = self._resolve()
            except Exception:  # noqa: BLE001 - registry missing, stale or torn
                urls = None
            if urls:
                if urls != self._urls:
                    LOG.warning('endpoint moved: %s -> %s', self._urls, urls)
                    self._set_urls(urls)
                return
            if self.down_seconds >= self._endpoint_wait_s:
                raise InfraError(f'endpoint down for {self.down_seconds:.0f}s')
            await asyncio.sleep(60)
            self.down_seconds += 60

    async def _create(self, req: dict[str, Any]) -> Any:
        """One request with retries on transient failures; raises InfraError when they run out."""
        delay = 5.0
        for attempt in range(1, self._max_attempts + 1):
            client = self._clients[self._rr % len(self._clients)]
            try:
                if self._gate is None:
                    return await client.chat.completions.create(**req)
                async with self._gate:
                    return await client.chat.completions.create(**req)
            except openai.BadRequestError:
                raise
            except openai.APITimeoutError as e:
                # The client timeout starts when the request is sent (our LoadGate wait is before it), but a server-side
                # admission queue counts. If the server answers a health check, the response itself ran away.
                if await asyncio.to_thread(self._healthy, self._urls[self._rr % len(self._urls)]):
                    raise RunawayError(f'no response within {self._timeout:.0f}s from a healthy server') from e
                self.n_retries += 1
                self._rr += 1
                LOG.warning('endpoint timeout and failed health check (attempt %d/%d)', attempt, self._max_attempts)
                if attempt == self._max_attempts:
                    raise InfraError(f'endpoint failed {attempt} times: {str(e)[:300]}') from e
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 120.0)
                await self._wait_for_endpoint()
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    openai.RateLimitError, openai.APIStatusError) as e:
                if isinstance(e, openai.APIStatusError) and e.status_code < 500 and e.status_code not in (408, 409, 429):
                    raise InfraError(f'endpoint HTTP {e.status_code}: {str(e)[:300]}') from e
                self.n_retries += 1
                self._rr += 1                       # next DP rank
                LOG.warning('endpoint error (attempt %d/%d): %s', attempt, self._max_attempts, str(e)[:200])
                if attempt == self._max_attempts:
                    raise InfraError(f'endpoint failed {attempt} times: {str(e)[:300]}') from e
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 120.0)
                await self._wait_for_endpoint()
        raise AssertionError

    @staticmethod
    def _healthy(url: str) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(f'{url.rstrip("/")}/models', timeout=30) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _prompt_estimate(self, msgs: list[dict], tools: list | None) -> int:
        """An upper estimate of the prompt's tokens: the previous response's exact prompt + completion plus ~1 token
        per 2.5 characters of what was added since (tool results, the next user message); the whole prompt at that
        rate when there is no previous response or the history was rewritten (a context summary)."""
        first = json.dumps(msgs[0], ensure_ascii=False, default=str)[:4000] if msgs else ''
        if self._last and len(msgs) > self._last[0] and self._last[1] == first:
            new = json.dumps(msgs[self._last[0] + 1:], ensure_ascii=False, default=str)
            return self._last[2] + int(len(new) / 2.5) + 8 * (len(msgs) - self._last[0])
        text = json.dumps(msgs, ensure_ascii=False, default=str) + json.dumps(tools or [], default=str)
        return int(len(text) / 2.5) + 8 * len(msgs)

    async def generate(self, messages: list, tools: dict[str, Tool]) -> AssistantMessage:
        msgs = apply_history_adapters(to_openai_messages(messages), self._adapters)
        req: dict[str, Any] = {'model': self._model, 'messages': msgs, **self._sampling}
        if tools:
            req['tools'] = to_openai_tools(tools)
            req['tool_choice'] = 'auto'
        # Context cap: the response may use only the room left under context_window_tokens (estimated from below, so
        # the cap holds); with no room left, Stirrup compacts (ContextOverflowError).
        room = self._ctx - self._prompt_estimate(msgs, req.get('tools')) - 16
        if room < 256:
            raise ContextOverflowError(f'prompt estimate leaves {room} tokens under the {self._ctx}-token context cap')
        req['max_completion_tokens'] = min(room, self._cap) if self._cap else room
        t0 = perf_counter()
        for _ in range(2):
            try:
                resp = await self._create(req)
                break
            except openai.BadRequestError as e:
                msg = str(e)
                m = _CTX_RE.search(msg)
                if m and 'max_completion_tokens' in req:
                    room = int(m.group(1)) - int(m.group(2)) - 16
                    if room > 256 and room < req['max_completion_tokens']:
                        req['max_completion_tokens'] = room     # cap > room left: shrink and retry once
                        continue
                if re.search(r'context length|maximum context|too long|exceeds the', msg, re.I):
                    raise ContextOverflowError(msg[:500]) from e
                raise InfraError(f'endpoint rejected the request: {msg[:300]}') from e
        else:
            raise ContextOverflowError('no room left in the context')
        t1 = perf_counter()
        if resp.usage:
            total = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
            if (resp.usage.prompt_tokens or 0) > self._ctx:
                raise ContextOverflowError(f'prompt of {resp.usage.prompt_tokens} tokens is over the {self._ctx}-token cap')
            if total > self._ctx:
                self.n_cap_overshoot += 1   # the estimate was low by less than the response; counted, not cut
            self._last = (len(msgs), json.dumps(msgs[0], ensure_ascii=False, default=str)[:4000] if msgs else '',
                          total)
            self.usage['responses'] += 1
            self.usage['input'] += resp.usage.prompt_tokens or 0
            self.usage['output'] += resp.usage.completion_tokens or 0
            det = getattr(resp.usage, 'completion_tokens_details', None)
            self.usage['reasoning'] += (getattr(det, 'reasoning_tokens', 0) or 0) if det else 0
        choice = resp.choices[0]
        if choice.finish_reason in ('max_tokens', 'length'):
            raise OutputTokenLimitError(model_slug=self._model, max_tokens=req.get('max_completion_tokens') or -1,
                                        provider_reason=choice.finish_reason)
        msg = choice.message
        extra = msg.model_extra or {}
        reasoning = getattr(msg, 'reasoning_content', None) or extra.get('reasoning_content') or extra.get('reasoning')
        blocks: list = []
        if reasoning:
            blocks.append(ReasoningBlock(content=reasoning))
        if msg.content:
            blocks.append(TextBlock(text=msg.content))
        blocks.extend(ToolCall.from_provider(provider_id=tc.id, name=tc.function.name,
                                             arguments=tc.function.arguments or '') for tc in (msg.tool_calls or []))
        u = resp.usage
        rtok = 0
        if u and getattr(u, 'completion_tokens_details', None):
            rtok = getattr(u.completion_tokens_details, 'reasoning_tokens', 0) or 0
        out = u.completion_tokens if u else 0
        return AssistantMessage(blocks=blocks, token_usage=TokenUsage(input=u.prompt_tokens if u else 0,
                                                                      answer=out - rtok, reasoning=rtok),
                                request_start_time=t0, request_end_time=t1)


# ----------------------------------------------------------------------------------------------- sandbox

_PROXY_VARS = ('http_proxy', 'https_proxy', 'ftp_proxy', 'all_proxy', 'no_proxy',
               'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'ALL_PROXY', 'NO_PROXY')


def sandbox_argv(sif: str, isolation: str, workdir: str, cmd: str, tmpdir: str | None = None) -> list[str]:
    """argv running `cmd` (bash) in the SIF: contained (no host /data, /home), no host environment (no proxy
    settings), no network, `workdir` as /work, and `tmpdir` (when given) as /tmp.

    Each command is its own `apptainer exec`, and --containall gives every exec a fresh private /tmp. A trajectory's
    sandbox therefore binds a per-trajectory host dir as /tmp, so /tmp persists across its commands as in AA's E2B
    sandbox (#32: 3-4% of trajectories lost files they had written to /tmp)."""
    inner = (f'ulimit -v {SANDBOX_MEM_KB}; export HOME=/work MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl '
             f'OMP_NUM_THREADS={SANDBOX_THREADS} OPENBLAS_NUM_THREADS={SANDBOX_THREADS} '
             f'MKL_NUM_THREADS={SANDBOX_THREADS}; cd /work; ' + cmd)
    # Apptainer passes the host's proxy variables through even with --cleanenv; unset them so the model never sees
    # the cluster's proxy (the network is off either way).
    scrub = ['env', *(x for v in _PROXY_VARS for x in ('-u', v))]
    base = [*scrub, 'apptainer', '--silent', 'exec', '--containall', '--cleanenv', '--no-home', '-B', f'{workdir}:/work',
            *(['-B', f'{tmpdir}:/tmp'] if tmpdir else []), '--pwd', '/work']
    if isolation == 'apptainer-net-none':
        return [*base, '--net', '--network', 'none', sif, 'bash', '-c', inner]
    if isolation == 'unshare-inside':
        return [*base, sif, 'unshare', '-rn', 'bash', '-c', inner]
    raise ValueError(isolation)


class OfflineSandbox(LocalCodeExecToolProvider):
    """Stirrup's local code exec, each command inside the CritPt SIF: contained, no network, only the work dir."""

    DESCRIPTION = ('Execute a shell command (bash) in an offline Linux sandbox with Python 3.13, numpy, scipy, sympy '
                   'and matplotlib. Returns exit code, stdout, and stderr as XML.')

    def __init__(self, *, sif: Path, isolation: str, work_base: Path, exec_slots: asyncio.Semaphore,
                 cmd_timeout: int = CMD_TIMEOUT) -> None:
        super().__init__(temp_base_dir=work_base, description=self.DESCRIPTION, shell_timeout=cmd_timeout)
        self._sif = str(sif)
        self._isolation = isolation
        self._slots = exec_slots
        self.n_commands = 0

    def _check_absolute_paths(self, argv: list[str] | None, cmd: str) -> CommandResult | None:
        return None        # commands run inside the container, as with Stirrup's Docker and E2B providers

    @property
    def tmp_dir(self) -> Path | None:
        """The trajectory's /tmp: a sibling of its work dir (not inside it, so the model's /work stays clean)."""
        return None if self.temp_dir is None else self.temp_dir.with_name(self.temp_dir.name + '.tmp')

    def wrap(self, cmd: str) -> str:
        self.tmp_dir.mkdir(exist_ok=True)
        return shlex.join(sandbox_argv(self._sif, self._isolation, str(self.temp_dir), cmd, str(self.tmp_dir)))

    async def __aexit__(self, *exc):
        tmp = self.tmp_dir
        try:
            return await super().__aexit__(*exc)
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        async with self._slots:           # queueing for a CPU slot does not count against the command's limit
            self.n_commands += 1
            res = await super().run_command(self.wrap(cmd), timeout=timeout)
        if res.exit_code == 255 and re.search(r'^FATAL: ', res.stderr or '', re.M):
            raise InfraError(f'sandbox failed to start: {res.stderr.strip()[:300]}')
        return res


# ----------------------------------------------------------------------------------------------- finish tool

class FinishParams(BaseModel):
    """Submit the final answer and end the task."""

    reason: Annotated[str, Field(description='Reason for finishing.')]
    derivation: Annotated[str, Field(description='Your step-by-step derivation of the result.')]
    answer: Annotated[str, Field(description='The complete code template from the problem, with your final answer '
                                             'filled in (Python code).')]


def strip_fences(code: str) -> str:
    """The answer code without surrounding markdown fences (models often add them)."""
    s = code.strip()
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', s, re.S)
    if blocks:
        return blocks[-1].strip('\n')
    return s


async def _finish_executor(params: FinishParams) -> ToolResult[ToolUseCountMetadata]:
    code = strip_fences(params.answer or '')
    if not code.strip():
        return ToolResult(content='ERROR: `answer` is empty. Fill in the code template and call finish again.',
                          metadata=ToolUseCountMetadata(), success=False)
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ToolResult(content=f'ERROR: `answer` is not valid Python ({e.msg}, line {e.lineno}). Fix it and call '
                                  'finish again.', metadata=ToolUseCountMetadata(), success=False)
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


FINISH_TOOL: Tool[FinishParams, ToolUseCountMetadata] = Tool[FinishParams, ToolUseCountMetadata](
    name='finish',
    description=('Submit your final answer and end the task. Note that you will need a separate turn to finish. '
                 'Fields: reason, derivation, answer (the filled-in code template).'),
    parameters=FinishParams,
    executor=_finish_executor,
)


# ----------------------------------------------------------------------------------------------- one trajectory

class QuietLogger(AgentLoggerBase):
    """No console output (dozens of agents share one process); the transcript is kept from the history."""

    def __init__(self, label: str = '') -> None:
        self.label = label
        self.name = 'critpt'
        self.model = None
        self.max_turns = None
        self.depth = 0
        self.n_summaries = 0

    def __enter__(self): return self
    def __exit__(self, *a): return None
    def on_step(self, turn, tool_calls=0, input_tokens=0, output_tokens=0, *a, **k):
        if turn % 10 == 0:
            LOG.info('%s: turn %d, %d tool calls, %d output tokens', self.label, turn, tool_calls, output_tokens)
    def assistant_message(self, *a, **k): return None
    def user_message(self, *a, **k): return None
    def task_message(self, *a, **k): return None
    def tool_result(self, *a, **k): return None
    def context_summarization_start(self, *a, **k): self.n_summaries += 1
    def context_summarization_complete(self, *a, **k): return None
    def debug(self, m, *a): LOG.debug(m, *a)
    def info(self, m, *a): LOG.debug(m, *a)
    def warning(self, m, *a): LOG.info(m, *a)
    def error(self, m, *a): LOG.info(m, *a)


@dataclass
class Trajectory:
    status: str                    # ok | model_error | infra_error
    answer_code: str = ''          # stripped answer code ('' unless ok)
    derivation: str = ''
    reason: str = ''
    error: str = ''
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    n_commands: int = 0
    n_api_retries: int = 0
    n_summaries: int = 0
    n_cap_overshoot: int = 0       # responses whose prompt + completion passed the context cap (estimate too low)
    seconds: float = 0.0
    history: list = field(default_factory=list)


def _dump_history(full: list[list]) -> list:
    out = []
    for group in full:
        out.append([m.model_dump(mode='json', exclude_none=True) for m in group])
    return out


async def run_trajectory(*, client_factory, system_prompt: str, user_prompt: str, sandbox_factory,
                         max_turns: int = MAX_TURNS, label: str = '') -> Trajectory:
    t0 = time.time()
    client: EndpointClient = client_factory()
    sandbox: OfflineSandbox = sandbox_factory()
    log = QuietLogger(label)
    agent = Agent(client=client, name='critpt', max_turns=max_turns, system_prompt=system_prompt,
                  tools=[sandbox], finish_tool=FINISH_TOOL, logger=log)
    tr = Trajectory(status='infra_error')
    history: list = []
    session = None
    try:
        async with agent.session(cache_on_interrupt=False) as session:
            finish, history, meta = await session.run(user_prompt)
        usage = meta.get('token_usage') or []
        tr.input_tokens = sum(u.input for u in usage)
        tr.output_tokens = sum(u.answer + u.reasoning for u in usage)
        tr.reasoning_tokens = sum(u.reasoning for u in usage)
        tr.turns = sum(1 for g in history for m in g if isinstance(m, AssistantMessage))
        if finish is None:
            tr.status, tr.error = 'model_error', 'no_finish'
        else:
            tr.status = 'ok'
            tr.answer_code = strip_fences(finish.answer)
            tr.derivation, tr.reason = finish.derivation, finish.reason
    except InfraError as e:
        tr.status, tr.error = 'infra_error', str(e)
    except OutputTokenLimitError as e:
        tr.status, tr.error = 'model_error', f'output_limit: {e.provider_reason}'
    except RunawayError as e:
        tr.status, tr.error = 'model_error', f'response_timeout: {e}'
    except ContextOverflowError as e:
        tr.status, tr.error = 'model_error', f'context_overflow: {str(e)[:200]}'
    except openai.BadRequestError as e:
        tr.status, tr.error = 'infra_error', f'bad_request: {str(e)[:300]}'
    except RuntimeError as e:
        if 'Summarizer response contained no text blocks' in str(e):
            # Stock Stirrup asked for a context summary three times (the last with tools withheld) and the model
            # answered with reasoning only: the model's failure under the harness, scored 0 like a missing finish.
            tr.status, tr.error = 'model_error', 'summary_failed: no text in 3 summarizer responses'
        else:
            LOG.exception('trajectory crashed')
            tr.status, tr.error = 'infra_error', f'{type(e).__name__}: {str(e)[:300]}'
    except Exception as e:  # noqa: BLE001 - anything else is ours, not the model's
        LOG.exception('trajectory crashed')
        tr.status, tr.error = 'infra_error', f'{type(e).__name__}: {str(e)[:300]}'
    if not history and session is not None and getattr(session, '_current_run_state', None) is not None:
        st = session._current_run_state           # the run raised: keep what it had so far
        history = [*st.full_msg_history, st.msgs]
    tr.history = _dump_history(history) if history else []
    if tr.status != 'ok' and not tr.output_tokens:
        # The run raised (or stopped) before Stirrup reported usage: count from the client and the partial history.
        tr.input_tokens, tr.output_tokens = client.usage['input'], client.usage['output']
        tr.reasoning_tokens = client.usage['reasoning']
        tr.turns = sum(1 for g in history for m in g if isinstance(m, AssistantMessage))
    tr.n_commands = sandbox.n_commands
    tr.n_api_retries = client.n_retries
    tr.n_summaries = log.n_summaries
    tr.n_cap_overshoot = client.n_cap_overshoot
    tr.seconds = round(time.time() - t0, 1)
    return tr

"""Model specs and an async OpenAI-compatible chat client that tells infra failures apart.

Model specs (aa_evals/README.md):
  serving:<name>              <registry>/<name>/endpoints.json (dead if older than 120 s; $TASKSYNTH_SERVING_ROOT)
  url:<base_url>@<served_name>
  openai:<model>              api.openai.com through the cluster proxy; external spend, so run.py asks for
                              --allow-external-api before using it
"""
import asyncio
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
STALE_SEC = 120
OPENAI_BASE = 'https://api.openai.com/v1'
PROXY = 'http://proxy.internal:8080'


class InfraError(Exception):
    """A call that failed for reasons outside the model (transport, server, judge format)."""

    def __init__(self, message, attempts=None):
        super().__init__(message)
        self.attempts = attempts or []


class Retryable(Exception):
    pass


@dataclass
class Endpoint:
    spec: str
    kind: str
    served_name: str
    urls: list
    api_key: str = None
    request: dict = field(default_factory=dict)   # the serving registry's request config (sampling overrides)
    registry: str = None
    updated_at: float = None

    def describe(self):
        return {'spec': self.spec, 'kind': self.kind, 'served_name': self.served_name, 'urls': self.urls,
                'registry': self.registry, 'request': self.request}


def secret(name):
    """An API key from the environment, else from the `export NAME=...` line of ~/.bashrc (whose interactive
    guard keeps condor jobs from sourcing it). Keys never go into submit files, job ads or run artifacts."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in Path('~/.bashrc').expanduser().read_text().splitlines():
            m = re.match(rf'\s*export\s+{re.escape(name)}=(.*)$', line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def resolve(spec, *, allow_stale=False):
    kind, _, rest = spec.partition(':')
    if kind == 'serving':
        path = serving.path(rest, SERVING_ROOT)
        try:
            reg = serving.record(rest, SERVING_ROOT)
        except (OSError, ValueError) as e:
            raise Retryable(f'{path}: {e}')
        age = serving.age(reg)
        if age > STALE_SEC and not allow_stale:
            raise Retryable(f'{path} is {age:.0f}s old (server dead or restarting)')
        return Endpoint(spec, kind, reg['served_name'], list(reg['urls']), None, reg.get('request') or {},
                        str(path), reg.get('updated_at'))
    if kind == 'url':
        base, sep, name = rest.rpartition('@')
        if not sep or not base or not name:
            raise ValueError(f'url spec must be url:<base_url>@<served_name>, got {spec!r}')
        return Endpoint(spec, kind, name, [base.rstrip('/')])
    if kind == 'openai':
        return Endpoint(spec, kind, rest, [OPENAI_BASE], secret('OPENAI_API_KEY'))
    raise ValueError(f'unknown model spec {spec!r} (serving:, url: or openai:)')


def resolve_now(spec, wait=0):
    """resolve() that waits up to `wait` seconds for a serving registry to be fresh."""
    deadline = time.time() + wait
    while True:
        try:
            return resolve(spec)
        except Retryable:
            if time.time() >= deadline:
                raise
            time.sleep(10)


def gate_retry_after(r) -> float | None:
    """Seconds to wait if `r` is an admission gate's backpressure 503 (server_overloaded + Retry-After), else None."""
    if r.status_code != 503:
        return None
    try:
        err = (r.json() or {}).get('error') or {}
    except ValueError:
        err = {}
    if err.get('type') != 'server_overloaded' and r.headers.get('X-Gate') != '1':
        return None
    try:
        return max(1.0, min(600.0, float(r.headers.get('Retry-After', 60))))
    except ValueError:
        return 60.0


class ChatClient:
    """POSTs /chat/completions with retries. Returns the response JSON; raises InfraError when every attempt
    failed. A transport error, timeout, 408/409/429 or 5xx, malformed body or stale registry is retried up to
    `max_attempts`; another 4xx (deterministic: bad request, context overflow) up to `max_4xx` times.

    An admission gate's backpressure (aa_evals/qwenserve/gate.py: 503 `server_overloaded` with Retry-After, sent while
    the server is busy with other conversations) is not a failure: it uses no attempt, the client waits Retry-After and
    asks again, for up to `gate_wait_s` in total. (2026-11-05, M3.4 HLE on 1xH100 at 128 in flight: 35 of the first
    250 items became infra errors after 30 gate rejections, about 25 min, each.)"""

    def __init__(self, spec, *, timeout=7200, max_attempts=30, max_4xx=3, backoff_cap=60, gate_wait_s=12 * 3600,
                 transport=None):
        self.spec = spec
        try:
            self.endpoint = resolve(spec, allow_stale=True)
        except Retryable:   # registry not there (server stopped): resolved again on first use
            self.endpoint = None
        self.max_attempts, self.max_4xx, self.backoff_cap = max_attempts, max_4xx, backoff_cap
        self.gate_wait_s = gate_wait_s
        external = spec.startswith('openai:')
        kwargs = dict(timeout=httpx.Timeout(timeout, connect=30), transport=transport, trust_env=False,
                      limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))
        if external and transport is None:
            kwargs['proxy'] = os.environ.get('https_proxy') or PROXY
        self.http = httpx.AsyncClient(**kwargs)
        self._rr = random.randrange(1 << 16)

    async def close(self):
        await self.http.aclose()

    def _current(self):
        if self.endpoint is None or self.endpoint.kind == 'serving':
            self.endpoint = resolve(self.spec)   # follows a restarted server; raises Retryable when stale
        return self.endpoint

    async def count_tokens(self, text):
        """Tokens in `text` by the endpoint's tokenizer (vLLM /tokenize), else a chars/3 estimate (above the true
        count for natural text and LaTeX alike, so a cap computed from it stays under the window)."""
        try:
            ep = self._current()
            if ep.kind != 'openai':
                root = ep.urls[0][:-3] if ep.urls[0].endswith('/v1') else ep.urls[0]
                for _ in range(3):
                    try:
                        r = await self.http.post(root + '/tokenize', json={'model': ep.served_name, 'prompt': text,
                                                                           'add_special_tokens': False})
                        if r.status_code == 200:
                            return int(r.json()['count'])
                    except (httpx.TransportError, ValueError, KeyError):
                        pass
                    await asyncio.sleep(2)
        except Retryable:
            pass
        return len(text) // 3 + 1

    async def chat(self, body):
        attempts, n4xx, gate = [], 0, {'rejections': 0, 'wait_s': 0.0}
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            t0 = time.time()
            url = None
            try:
                ep = self._current()
                self._rr += 1
                url = ep.urls[self._rr % len(ep.urls)]
                headers = {'Authorization': f'Bearer {ep.api_key}'} if ep.api_key else {}
                r = await self.http.post(url + '/chat/completions', json={'model': ep.served_name, **body},
                                         headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    if not data.get('choices'):
                        raise Retryable(f'200 without choices: {str(data)[:300]}')
                    attempts.append({'attempt': attempt, 'url': url, 'status': 200, 'latency_s': time.time() - t0})
                    if gate['rejections']:
                        attempts.append({'gate_rejections': gate['rejections'], 'gate_wait_s': round(gate['wait_s'])})
                    return data, attempts
                detail = r.text[:500]
                wait = gate_retry_after(r)
                if wait is not None and gate['wait_s'] < self.gate_wait_s:
                    gate['rejections'] += 1
                    if gate['rejections'] == 1:
                        attempts.append({'attempt': attempt, 'url': url, 'status': 503, 'gate': True,
                                         'error': detail[:200], 'latency_s': time.time() - t0})
                    attempt -= 1   # backpressure, not a failed attempt
                    d = wait * (0.75 + 0.5 * random.random())
                    gate['wait_s'] += d
                    await asyncio.sleep(d)
                    continue
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise Retryable(f'HTTP {r.status_code}: {detail}')
                n4xx += 1
                attempts.append({'attempt': attempt, 'url': url, 'status': r.status_code, 'error': detail,
                                 'latency_s': time.time() - t0})
                if n4xx >= self.max_4xx:
                    raise InfraError(f'HTTP {r.status_code} x{n4xx}: {detail}', attempts)
                delay = min(2, self.backoff_cap)
            except (Retryable, httpx.TransportError, ValueError) as e:
                # ValueError covers an unparseable 200 body (json.JSONDecodeError).
                attempts.append({'attempt': attempt, 'url': url, 'error': f'{type(e).__name__}: {str(e)[:500]}',
                                 'latency_s': time.time() - t0})
                delay = min(self.backoff_cap, 2 ** min(attempt, 10))
                if isinstance(e, Retryable) and e.args and str(e.args[0]).startswith('HTTP 429'):
                    delay = min(max(delay, 10), self.backoff_cap)
            if attempt < self.max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))
        if gate['rejections']:
            attempts.append({'gate_rejections': gate['rejections'], 'gate_wait_s': round(gate['wait_s'])})
        raise InfraError(f'{self.max_attempts} attempts failed; last: {attempts[-1].get("error")}', attempts)


def message_fields(data):
    """(content, reasoning, finish_reason, usage) of a chat completion."""
    choice = data['choices'][0]
    msg = choice.get('message') or {}
    reasoning = msg.get('reasoning_content') or msg.get('reasoning')
    return msg.get('content'), reasoning, choice.get('finish_reason'), data.get('usage') or {}

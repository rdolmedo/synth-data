"""Model specs, endpoint limits, and a streaming OpenAI-compatible chat client that tells failure kinds apart.

Model specs (aa_evals/README.md):
  serving:<name>               <registry>/<name>/endpoints.json (dead if > 120 s old; $TASKSYNTH_SERVING_ROOT)
  url:<base_url>@<served_name> any OpenAI-compatible endpoint
  openai:<model>               api.openai.com (external spend: run.py requires --allow-external-api)

Failure kinds:
  ContextRejected  the endpoint said the prompt is too long (or too many images) -> the caller adapts the input
  InfraError       transport errors, timeouts, 408/409/429/5xx, a stale registry, malformed bodies, after
                   `max_attempts` tries -> never scored as a model failure
"""
import asyncio
import json
import os
import random
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
STALE_SEC = 120
OPENAI_BASE = 'https://api.openai.com/v1'


class InfraError(Exception):
    pass


class Retryable(Exception):
    pass


class GateBusy(Retryable):
    """An admission gate's backpressure 503 (server_overloaded): wait and retry, not a failed attempt."""


class Fatal(InfraError):
    """A deterministic rejection (bad request, auth): retried once, then reported as infra."""


class ContextRejected(Exception):
    def __init__(self, message, max_images=None, max_decoded_bytes=None):
        super().__init__(message)
        self.max_images = max_images
        self.max_decoded_bytes = max_decoded_bytes


def secret(name):
    """An API key from the environment, else from the `export NAME=...` line of ~/.bashrc (condor jobs do not
    source it). Keys never go into submit files, logs or run artifacts."""
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


@dataclass
class Endpoint:
    spec: str
    kind: str
    served_name: str
    urls: list
    api_key: str = None
    request: dict = field(default_factory=dict)
    registry: str = None
    updated_at: float = None

    def describe(self):
        return {'spec': self.spec, 'kind': self.kind, 'served_name': self.served_name, 'urls': self.urls,
                'registry': self.registry}

    @property
    def external(self):
        return self.kind == 'openai'


def resolve(spec, allow_stale=False):
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
        return Endpoint(spec, kind, reg['served_name'], [u.rstrip('/') for u in reg['urls']], None,
                        reg.get('request') or {}, str(path), reg.get('updated_at'))
    if kind == 'url':
        base, sep, name = rest.rpartition('@')
        if not sep or not base or not name:
            raise ValueError(f'url spec must be url:<base_url>@<served_name>, got {spec!r}')
        return Endpoint(spec, kind, name, [base.rstrip('/')])
    if kind == 'openai':
        key = secret('OPENAI_API_KEY')
        if not key:
            raise ValueError('OPENAI_API_KEY is not set')
        return Endpoint(spec, kind, rest, [OPENAI_BASE], key)
    raise ValueError(f'unknown model spec {spec!r} (serving:, url: or openai:)')


async def resolve_wait(spec, wait=1800):
    deadline = time.time() + wait
    while True:
        try:
            return resolve(spec)
        except Retryable as e:
            if time.time() >= deadline:
                raise InfraError(str(e))
            print(f'[client] waiting for {spec}: {e}', flush=True)
            await asyncio.sleep(20)


def serving_launch_limits(spec):
    """(max_model_len, max_images, max_decoded_bytes) from a serving dir's launch.json, or Nones."""
    kind, _, rest = spec.partition(':')
    if kind != 'serving':
        return None, None, None
    try:
        launch = json.loads((serving.registry_root(SERVING_ROOT) / rest / 'launch.json').read_text())
        if 'replicas' in launch:      # multi-replica / gated recipes: one launch per replica (all the same)
            launch = launch['replicas'][0]
        argv = launch['argv']
    except (OSError, ValueError, KeyError):
        return None, None, None
    dec = (launch.get('env') or {}).get('VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES')
    mml = imgs = None
    for i, a in enumerate(argv[:-1]):
        if a == '--max-model-len':
            mml = int(argv[i + 1])
        if a == '--limit-mm-per-prompt':
            try:
                imgs = int(json.loads(argv[i + 1]).get('image', 0))
            except (ValueError, AttributeError):
                pass
    return mml, imgs, int(dec) if dec else None


def root_url(v1_url):
    return v1_url[:-3] if v1_url.endswith('/v1') else v1_url


_IMAGE_CAP_RE = re.compile(r'[Aa]t most (\d+) image')
_DECODE_RE = re.compile(r'VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES=(\d+)')
_CONTEXT_RE = re.compile(r'maximum context length|context length|too long|exceeds|max_model_len|longer than', re.I)


class Client:
    """One httpx client per endpoint spec; round-robins DP ranks and re-reads a serving registry on failure."""

    def __init__(self, spec, *, timeout=10800, max_attempts=12, backoff_cap=120, concurrency=64, gate_wait_s=12 * 3600):
        self.spec = spec
        self.ep = None
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.gate_wait_s = gate_wait_s
        self.backoff_cap = backoff_cap
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30), trust_env=True,
                                      limits=httpx.Limits(max_connections=concurrency * 2))
        self._rr = 0

    async def endpoint(self, refresh=False):
        if self.ep is None or (refresh and self.ep.kind == 'serving'):
            self.ep = await resolve_wait(self.spec)
        return self.ep

    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.ep and self.ep.api_key:
            h['Authorization'] = f'Bearer {self.ep.api_key}'
        return h

    def _url(self):
        self._rr += 1
        return self.ep.urls[self._rr % len(self.ep.urls)]

    async def close(self):
        await self.http.aclose()

    # --- discovery ---
    async def limits(self):
        """{'max_model_len', 'max_images'} as far as the endpoint tells us."""
        ep = await self.endpoint()
        mml, imgs, dec = serving_launch_limits(self.spec)
        if ep.kind != 'openai':
            try:
                r = await self.http.get(f'{ep.urls[0]}/models', headers=self._headers())
                for m in r.json().get('data', []):
                    if m.get('id') == ep.served_name and m.get('max_model_len'):
                        mml = int(m['max_model_len'])
            except (httpx.HTTPError, ValueError):
                pass
        return {'max_model_len': mml, 'max_images': imgs, 'max_decoded_bytes': dec}

    async def count_tokens(self, text):
        """Tokens in `text` by the endpoint's tokenizer (vLLM /tokenize); a chars/3 estimate if unavailable."""
        ep = await self.endpoint()
        if ep.kind != 'openai':
            for attempt in range(5):
                try:
                    r = await self.http.post(f'{root_url(ep.urls[0])}/tokenize', headers=self._headers(),
                                             json={'model': ep.served_name, 'prompt': text, 'add_special_tokens': False})
                    if r.status_code == 200:
                        return int(r.json()['count'])
                    if r.status_code < 500:
                        break
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
                await asyncio.sleep(2 + 3 * attempt)
                await self.endpoint(refresh=True)
        return len(text) // 3 + 1

    # --- chat ---
    async def chat(self, body, stream=True, first_token=None):
        """POST /chat/completions with retries. Returns {'content', 'reasoning', 'finish_reason', 'usage',
        'attempts', 'seconds'}; raises ContextRejected or InfraError. `first_token` (an asyncio.Event) is set
        when the first streamed token arrives, i.e. once the prompt is prefilled (and prefix-cached)."""
        attempts = []
        t0 = time.time()
        gate = {'rejections': 0, 'wait_s': 0.0}
        async with self.sem:
            attempt = 0
            while attempt < self.max_attempts:
                try:
                    ep = await self.endpoint(refresh=attempt > 0)
                    b = dict(body, model=ep.served_name)
                    out = await (self._stream(b, first_token) if stream else self._plain(b))
                    if gate['rejections']:
                        attempts.append({'gate_rejections': gate['rejections'], 'gate_wait_s': round(gate['wait_s'])})
                    out['attempts'] = attempts
                    out['seconds'] = round(time.time() - t0, 1)
                    return out
                except ContextRejected:
                    raise
                except GateBusy as e:   # An admission gate's backpressure (aa_evals/qwenserve/gate.py: 503 `server_overloaded`, Retry-After) uses no attempt: the client waits and asks again, for up to `gate_wait_s` in total (see hle/client.py).
                    if gate['wait_s'] < self.gate_wait_s:
                        gate['rejections'] += 1
                        d = 60 * (0.75 + 0.5 * random.random())
                        gate['wait_s'] += d
                        await asyncio.sleep(d)
                        continue
                    attempts.append({'at': round(time.time() - t0, 1), 'error': f'GateBusy: {e}'[:500]})
                    attempt += 1
                except (Retryable, httpx.HTTPError, InfraError) as e:
                    msg = f'{type(e).__name__}: {e}'[:500]
                    attempts.append({'at': round(time.time() - t0, 1), 'error': msg})
                    if isinstance(e, Fatal) and sum(a['error'].startswith('Fatal') for a in attempts) >= 2:
                        break
                    delay = min(self.backoff_cap, 5 * 2 ** attempt) * (0.5 + random.random())
                    attempt += 1
                    await asyncio.sleep(delay)
        if gate['rejections']:
            attempts.append({'gate_rejections': gate['rejections'], 'gate_wait_s': round(gate['wait_s'])})
        raise InfraError(f'{len(attempts)} attempts failed; last: {attempts[-1]["error"] if attempts else "?"}')

    def _check_status(self, status, text):
        if status == 200:
            return
        if status == 400 or status == 413:
            m = _IMAGE_CAP_RE.search(text)
            if m:
                raise ContextRejected(text[:300], max_images=int(m.group(1)))
            m = _DECODE_RE.search(text)
            if m:
                raise ContextRejected(text[:300], max_decoded_bytes=int(m.group(1)))
            if _CONTEXT_RE.search(text) or status == 413:
                raise ContextRejected(text[:300])
            raise Fatal(f'HTTP {status}: {text[:300]}')   # bad request: our bug or config, not the model's
        if status in (401, 403, 404):
            raise Fatal(f'HTTP {status}: {text[:300]}')
        if status == 503 and 'server_overloaded' in text:
            raise GateBusy(f'HTTP 503: {text[:300]}')
        raise Retryable(f'HTTP {status}: {text[:300]}')

    async def _plain(self, body):
        r = await self.http.post(f'{self._url()}/chat/completions', headers=self._headers(), json=body)
        self._check_status(r.status_code, r.text)
        try:
            j = r.json()
            ch = j['choices'][0]
            msg = ch.get('message') or {}
        except (ValueError, KeyError, IndexError) as e:
            raise Retryable(f'malformed response: {e}')
        return {'content': msg.get('content') or '', 'reasoning': msg.get('reasoning_content') or msg.get('reasoning') or '',
                'finish_reason': ch.get('finish_reason'), 'usage': j.get('usage') or {}}

    async def _stream(self, body, first_token=None):
        body = dict(body, stream=True, stream_options={'include_usage': True})
        content, reasoning, finish, usage = [], [], None, {}
        async with self.http.stream('POST', f'{self._url()}/chat/completions', headers=self._headers(), json=body) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors='replace')
                self._check_status(r.status_code, text)
            async for line in r.aiter_lines():
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    raise Retryable(f'malformed stream chunk: {data[:200]}')
                if j.get('error'):
                    err = json.dumps(j['error'])[:300]
                    if _CONTEXT_RE.search(err):
                        raise ContextRejected(err)
                    raise Retryable(f'stream error: {err}')
                if j.get('usage'):
                    usage = j['usage']
                for ch in j.get('choices') or []:
                    d = ch.get('delta') or {}
                    if first_token is not None and not first_token.is_set():
                        first_token.set()
                    if d.get('content'):
                        content.append(d['content'])
                    rc = d.get('reasoning_content') or d.get('reasoning')
                    if rc:
                        reasoning.append(rc)
                    if ch.get('finish_reason'):
                        finish = ch['finish_reason']
        if finish is None:
            raise Retryable('stream ended without a finish_reason')
        return {'content': ''.join(content), 'reasoning': ''.join(reasoning), 'finish_reason': finish, 'usage': usage}

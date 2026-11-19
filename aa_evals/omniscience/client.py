"""Model specs and a stdlib OpenAI-compatible chat client with retries.

Model specs (aa_evals/README.md contract, plus one extension for external APIs):
  serving:<name>                 <registry>/<name>/endpoints.json (stale after 120 s; $TASKSYNTH_SERVING_ROOT)
  url:<base_url>@<served_name>   any OpenAI-compatible endpoint
  openai:<model>                 api.openai.com, key from OPENAI_API_KEY (external spend: needs the user's approval)

Every failure that is not the model's fault (connection errors, timeouts, 429/5xx, a dead or stale registry,
a response without choices) raises InfraError after the retries run out. Callers mark the item infra_error.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
if os.environ.get('AA_SERVING_ROOT'):   # read until 11-18: refuse rather than silently reach the production registry
    raise SystemExit('AA_SERVING_ROOT is no longer read: set TASKSYNTH_SERVING_ROOT (every aa_evals reader honours it)')
STALE_SEC = 120
OPENAI_BASE = 'https://api.openai.com/v1'


class InfraError(RuntimeError):
    pass


class ModelTimeout(RuntimeError):
    """No response within the request timeout while the server stays healthy: a runaway generation, which is the
    model's behavior (model_error), not infra (audit 1, #31). Raised only when the caller opts in."""


@dataclass
class Endpoint:
    spec: str
    kind: str                   # serving | url | openai
    model: str                  # served name sent as "model"
    urls: list[str]
    api_key_env: str | None = None
    request: dict = field(default_factory=dict)  # the serving registry's "request" block (lab sampling settings)
    _rr: itertools.count = field(default_factory=itertools.count, repr=False)

    def next_url(self) -> str:
        return self.urls[next(self._rr) % len(self.urls)]

    def describe(self) -> dict:
        return {'spec': self.spec, 'kind': self.kind, 'served_name': self.model, 'urls': self.urls}


def _read_registry(name: str) -> dict:
    path = serving.path(name, SERVING_ROOT)
    if not path.exists():
        raise InfraError(f'{path} missing: serving:{name} is not up')
    reg = serving.record(name, SERVING_ROOT)
    age = serving.age(reg)
    if age > STALE_SEC:
        raise InfraError(f'{path} is {age:.0f} s old (> {STALE_SEC} s): serving:{name} is dead')
    if not reg.get('urls'):
        raise InfraError(f'{path} lists no urls')
    return reg


def resolve(spec: str) -> Endpoint:
    kind, _, rest = spec.partition(':')
    if kind == 'serving':
        reg = _read_registry(rest)
        return Endpoint(spec, 'serving', reg['served_name'], [u.rstrip('/') for u in reg['urls']],
                        request=reg.get('request') or {})
    if kind == 'url':
        base, sep, name = rest.rpartition('@')
        if not sep or not base or not name:
            raise ValueError(f'bad spec {spec!r}: want url:<base_url>@<served_name>')
        return Endpoint(spec, 'url', name, [base.rstrip('/')])
    if kind == 'openai':
        if not rest:
            raise ValueError(f'bad spec {spec!r}: want openai:<model>')
        return Endpoint(spec, 'openai', rest, [OPENAI_BASE], api_key_env='OPENAI_API_KEY')
    raise ValueError(f'unknown model spec {spec!r} (serving:, url: or openai:)')


def _refresh(ep: Endpoint) -> None:
    """Re-read a serving registry so a restarted server's new URLs are picked up."""
    if ep.kind == 'serving':
        reg = _read_registry(ep.spec.partition(':')[2])
        ep.urls = [u.rstrip('/') for u in reg['urls']]
        ep.model = reg['served_name']


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_ENV_PROXY = urllib.request.build_opener()


def _opener(ep: Endpoint):
    # Cluster endpoints are reached directly; only external APIs go through the proxy from the environment.
    return _ENV_PROXY if ep.kind == 'openai' else _NO_PROXY


def _headers(ep: Endpoint) -> dict:
    h = {'Content-Type': 'application/json'}
    if ep.api_key_env:
        key = os.environ.get(ep.api_key_env)
        if not key:
            raise InfraError(f'{ep.api_key_env} is not set')
        h['Authorization'] = f'Bearer {key}'
    return h


def get_json(ep: Endpoint, path: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(ep.next_url() + path, headers=_headers(ep))
    with _opener(ep).open(req, timeout=timeout) as r:
        return json.loads(r.read())


def max_model_len(ep: Endpoint) -> int | None:
    """vLLM reports max_model_len in /v1/models; other servers may not."""
    try:
        data = get_json(ep, '/models').get('data') or []
    except Exception:
        return None
    for m in data:
        if m.get('id') == ep.model and m.get('max_model_len'):
            return int(m['max_model_len'])
    return None


_RETRYABLE_HTTP = {404, 408, 409, 425, 429, 500, 502, 503, 504}
_backoff_lock = threading.Lock()


def healthy(ep: Endpoint) -> bool:
    try:
        get_json(ep, '/models', timeout=30)
        return True
    except Exception:
        return False


def chat(ep: Endpoint, body: dict, *, timeout: float, attempts: int = 8, log: list | None = None,
         timeout_is_model: bool = False, gate_wait_s: float = 12 * 3600) -> dict:
    """POST /chat/completions with retries. Returns the parsed response; raises InfraError when retries run out.

    `log` (if given) collects one entry per failed attempt for the transcript. With `timeout_is_model`, a request
    that was accepted but got no response within `timeout` while the server still answers /models raises
    ModelTimeout (not retried); connection failures and timeouts of an unhealthy server stay infra.
    An admission gate's backpressure (aa_evals/qwenserve/gate.py: 503 `server_overloaded`, Retry-After) uses no
    attempt: the client waits and asks again, for up to `gate_wait_s` in total (see hle/client.py)."""
    payload = dict(body, model=ep.model)
    last = None
    bad_request_seen = False
    gate_n, gate_wait, gate_delay = 0, 0.0, None
    attempt = -1
    while attempt + 1 < attempts:
        if gate_delay is not None:   # the gate asked us to come back later: no attempt used
            time.sleep(gate_delay)
            gate_delay = None
        else:
            attempt += 1
            if attempt:
                delay = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
                time.sleep(delay)
        try:
            _refresh(ep)
            payload['model'] = ep.model
            url = ep.next_url() + '/chat/completions'
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=_headers(ep), method='POST')
            t0 = time.time()
            with _opener(ep).open(req, timeout=timeout) as r:
                gate_ms = r.headers.get('X-Gate-Queue-Ms')
                resp = json.loads(r.read())
            if gate_ms is not None:
                resp['_gate_queue_ms'] = gate_ms
            if not resp.get('choices'):
                raise InfraError(f'response without choices: {str(resp)[:300]}')
            resp['_latency_s'] = round(time.time() - t0, 3)
            resp['_attempts'] = attempt + 1
            if gate_n:
                resp['_gate_rejections'], resp['_gate_wait_s'] = gate_n, round(gate_wait)
            return resp
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode(errors='replace')[:500]
            except Exception:
                pass
            last = f'HTTP {e.code}: {detail}'
            if e.code == 503 and 'server_overloaded' in detail and gate_wait < gate_wait_s:
                try:
                    ra = max(1.0, min(600.0, float(e.headers.get('Retry-After', 60))))
                except (TypeError, ValueError):
                    ra = 60.0
                gate_delay = ra * (0.75 + 0.5 * random.random())
                gate_n, gate_wait = gate_n + 1, gate_wait + gate_delay
                if log is not None and gate_n == 1:
                    log.append({'attempt': attempt, 'error': last[:200], 'gate': True})
                continue
            if e.code not in _RETRYABLE_HTTP:
                # A 400/401/403 is a request or config problem, not a model failure. Retry once in case it is a
                # transient server state, then give up.
                if bad_request_seen:
                    if log is not None:
                        log.append({'attempt': attempt, 'error': last})
                    raise InfraError(last)
                bad_request_seen = True
        except (socket.timeout, TimeoutError) as e:  # read timeout: the request was sent and accepted
            last = f'{type(e).__name__}: {e}'
            if timeout_is_model and healthy(ep):
                if log is not None:
                    log.append({'attempt': attempt, 'error': last, 'server_healthy': True})
                raise ModelTimeout(f'no response in {timeout:.0f}s while the server is healthy')
        except (urllib.error.URLError, ConnectionError, json.JSONDecodeError, InfraError, OSError) as e:
            last = f'{type(e).__name__}: {e}'
        if log is not None:
            log.append({'attempt': attempt, 'error': last})
    raise InfraError(f'{attempts} attempts failed; last: {last}')


_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.S)


def split_response(resp: dict) -> dict:
    """Final answer text (reasoning stripped), reasoning text, finish reason and usage."""
    choice = resp['choices'][0]
    msg = choice.get('message') or {}
    content = msg.get('content') or ''
    reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
    # Servers without a reasoning parser leave the thinking inline.
    if '</think>' in content:
        inline = content.rsplit('</think>', 1)
        reasoning = (reasoning + '\n' + inline[0]).strip()
        content = inline[1]
    content = _THINK_BLOCK.sub('', content).strip()
    # Datakit checkpoints (Marin releases) can leak a second <|start_think|>...<|end_think|> span into the content
    # (marin#10400): it is reasoning, as an inline </think> span is above.
    if '<|end_think|>' in content:
        inline = content.rsplit('<|end_think|>', 1)
        reasoning = (reasoning + '\n' + inline[0].replace('<|start_think|>', '')).strip()
        content = inline[1].strip()
    if '<|start_think|>' in content:   # a span never closed (cut, or the model stopped inside it): reasoning too
        content, _, tail = content.partition('<|start_think|>')
        reasoning = (reasoning + '\n' + tail.replace('<|start_think|>', '')).strip()
        content = content.strip()
    return {'answer': content, 'reasoning': reasoning, 'finish_reason': choice.get('finish_reason'),
            'usage': resp.get('usage') or {}}

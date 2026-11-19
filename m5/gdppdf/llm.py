"""Async OpenAI-compatible client for the shared M5 servers (DeepSeek generator, Super teacher).

- Reads /data/aai/tbench/tasksynth/serving/<name>/endpoints.json; a file older than 120 s is a dead or
  restarting server: wait and retry, never crash.
- Pins each stream (an affinity key, e.g. a document id) to one (url, DP rank) target: one URL per rank, or one URL
  with N internal ranks addressed by the X-data-parallel-rank header.
- In-flight requests are capped by m5-coord/alloc.json (re-read every 60 s); the cap can change while running.
- Long timeouts (6 h per request), retries with backoff on transport errors and 5xx/429.
"""
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

SERVING = Path('/data/aai/tbench/tasksynth/serving')
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
BENCH = 'gdppdf'
STALE = 120


class Dead(Exception):
    pass


def read_registry(name):
    p = SERVING / name / 'endpoints.json'
    try:
        info = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise Dead(f'{p}: {e}')
    if time.time() - float(info.get('updated_at', 0)) > STALE:
        raise Dead(f'{p}: stale ({time.time() - float(info.get("updated_at", 0)):.0f}s)')
    dp = 1
    if len(info['urls']) == 1:
        try:
            argv = json.loads((p.parent / 'launch.json').read_text())
            argv = (argv.get('replicas') or [argv])[0]['argv'] if 'replicas' in argv else argv['argv']
            if not any(a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb')):
                dp = int(argv[argv.index('--data-parallel-size') + 1]) if '--data-parallel-size' in argv else 1
        except (OSError, ValueError, KeyError, IndexError):
            dp = 1
    info['_dp'] = dp
    return info


def alloc_cap(kind, engine=None, default=8):
    """kind 'ds': max in-flight DeepSeek requests; 'super': max in-flight sessions per Super engine."""
    try:
        a = json.loads(ALLOC.read_text())
        if kind == 'ds':
            return int(a['ds']['max_inflight'][BENCH])
        return int(a['super']['max_inflight_per_engine'][BENCH])
    except (OSError, ValueError, KeyError):
        return default


class _Null:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *e):
        return None


class DynSem:
    """A semaphore whose limit can change (re-read from alloc.json)."""

    def __init__(self, limit):
        self.limit, self.used, self.cond = limit, 0, asyncio.Condition()

    async def set(self, limit):
        async with self.cond:
            self.limit = limit
            self.cond.notify_all()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.used < self.limit)
            self.used += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.used -= 1
            self.cond.notify_all()


class Server:
    """One serving name (e.g. m5-ds, or one m5-super-<k> engine)."""

    def __init__(self, name, cap_kind='ds', timeout=6 * 3600, cap=None):
        self.name, self.cap_kind, self.fixed_cap = name, cap_kind, cap
        self.timeout = timeout
        self.info = None
        self.sem = DynSem(cap or alloc_cap(cap_kind))
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=60), trust_env=False,
                                      limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512))
        self._read_at = 0
        self._task = None
        self.stats = {'requests': 0, 'retries': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cached_tokens': 0}

    async def start(self):
        await self.refresh(wait=True)
        if self._task is None:
            self._task = asyncio.create_task(self._watch())
        return self

    async def _watch(self):
        while True:
            await asyncio.sleep(60)
            if not self.fixed_cap:
                await self.sem.set(alloc_cap(self.cap_kind, default=self.sem.limit))
            try:
                await self.refresh()
            except Exception:
                pass

    async def refresh(self, wait=False):
        while True:
            try:
                self.info = read_registry(self.name)
                self._read_at = time.time()
                return self.info
            except Dead as e:
                if not wait and self.info is not None:
                    return self.info
                print(f'[llm] {self.name}: {e}; waiting', flush=True)
                await asyncio.sleep(30)

    @property
    def served_name(self):
        return self.info['served_name']

    @property
    def request(self):
        return self.info.get('request') or {}

    def targets(self):
        urls = [u.rstrip('/') for u in self.info['urls']]
        if len(urls) == 1 and self.info.get('_dp', 1) > 1:
            return [(urls[0], r) for r in range(self.info['_dp'])]
        return [(u, None) for u in urls]

    def pick(self, affinity):
        ts = self.targets()
        if affinity is None:
            return random.choice(ts)
        h = int(hashlib.sha1(str(affinity).encode()).hexdigest()[:12], 16)
        return ts[h % len(ts)]

    async def close(self):
        if self._task:
            self._task.cancel()
        await self.http.aclose()

    async def tokenize(self, text, affinity=None):
        for attempt in range(20):
            url, _ = self.pick(affinity)
            root = url[:-3] if url.endswith('/v1') else url
            try:
                r = await self.http.post(f'{root}/tokenize', json={'model': self.served_name, 'prompt': text,
                                                                   'add_special_tokens': False}, timeout=600)
                if r.status_code == 200:
                    return int(r.json()['count'])
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            await asyncio.sleep(min(120, 5 * 2 ** attempt))
            await self.refresh(wait=True)
        raise RuntimeError(f'{self.name}: /tokenize failed')

    async def chat(self, body, affinity=None, max_attempts=12, on_first_token=None, hold=True):
        """Streaming chat completion. body: OpenAI request without 'model'/'stream'.
        -> {'content', 'reasoning', 'finish_reason', 'usage', 'seconds', 'attempts', 'target'}.
        Raises RuntimeError after max_attempts transport/server failures; ValueError on a 400 (our bug or a
        context rejection: the caller decides)."""
        t0 = time.time()
        errors = []
        async with (self.sem if hold else _Null()):
            for attempt in range(max_attempts):
                if time.time() - self._read_at > 60:
                    await self.refresh()
                url, rank = self.pick(affinity)
                headers = {'Content-Type': 'application/json'}
                if rank is not None:
                    headers['X-data-parallel-rank'] = str(rank)
                b = dict(body, model=self.served_name, stream=True, stream_options={'include_usage': True})
                content, reasoning, finish, usage = [], [], None, {}
                try:
                    async with self.http.stream('POST', f'{url}/chat/completions', json=b, headers=headers) as r:
                        if r.status_code != 200:
                            txt = (await r.aread()).decode(errors='replace')
                            if r.status_code == 400:
                                raise ValueError(f'HTTP 400: {txt[:500]}')
                            raise httpx.HTTPStatusError(f'HTTP {r.status_code}: {txt[:300]}', request=r.request,
                                                        response=r)
                        async for line in r.aiter_lines():
                            if not line.startswith('data:'):
                                continue
                            d = line[5:].strip()
                            if d == '[DONE]':
                                break
                            j = json.loads(d)
                            if j.get('error'):
                                raise httpx.HTTPError(f'stream error: {json.dumps(j["error"])[:300]}')
                            if j.get('usage'):
                                usage = j['usage']
                            for ch in j.get('choices') or []:
                                dd = ch.get('delta') or {}
                                if dd.get('content'):
                                    content.append(dd['content'])
                                rc = dd.get('reasoning_content') or dd.get('reasoning')
                                if rc:
                                    reasoning.append(rc)
                                if on_first_token and (dd.get('content') or rc):
                                    on_first_token()
                                    on_first_token = None
                                if ch.get('finish_reason'):
                                    finish = ch['finish_reason']
                    if finish is None:
                        raise httpx.HTTPError('stream ended without finish_reason')
                    self.stats['requests'] += 1
                    self.stats['prompt_tokens'] += usage.get('prompt_tokens') or 0
                    self.stats['completion_tokens'] += usage.get('completion_tokens') or 0
                    self.stats['cached_tokens'] += ((usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                                                    or 0)
                    return {'content': ''.join(content), 'reasoning': ''.join(reasoning), 'finish_reason': finish,
                            'usage': usage, 'seconds': round(time.time() - t0, 1), 'attempts': errors,
                            'target': [url, rank]}
                except ValueError as e:
                    if str(e).startswith('HTTP 400'):
                        raise
                    errors.append(f'{type(e).__name__}: {e}'[:300])
                except (httpx.HTTPError, OSError, json.JSONDecodeError) as e:
                    errors.append(f'{type(e).__name__}: {e}'[:300])
                self.stats['retries'] += 1
                await asyncio.sleep(min(300, 10 * 2 ** attempt) * (0.5 + random.random()))
                await self.refresh(wait=True)
        raise RuntimeError(f'{self.name}: {len(errors)} attempts failed; last: {errors[-1] if errors else "?"}')

"""Async OpenAI-compatible clients for the M5 shared servers.

DeepSeek (generator, judge): /data/aai/tbench/tasksynth/serving/m5-ds/endpoints.json, `urls` one entry per DP rank.
Each request stream is pinned to a rank by a key (X-data-parallel-rank = the url's index), so prefixes stay cached.
A registry older than 120 s means a dead or restarting server: wait and re-read, never crash. Concurrency follows the
coordinator's caps in m5-coord/alloc.json (re-read every 60 s). Pool follows alloc.json ds.serving: DeepSeek servers
come and go (a replacement is listed, the old one removed), and a request on a failed or delisted server moves on.
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
STALE = 120


def read_registry(name):
    info = json.loads((SERVING / name / 'endpoints.json').read_text())
    if time.time() - float(info.get('updated_at', 0)) > STALE:
        raise RuntimeError(f'{name}: registry stale')
    return info


class Cap:
    """A concurrency limit that follows alloc.json (key path e.g. ('ds', 'max_inflight', 'hle'))."""

    def __init__(self, path, default, override=None):
        self.path, self.default, self.override = path, default, override
        self.value, self.read_at, self.inflight = default, 0, 0
        self.cond = asyncio.Condition()

    def current(self):
        if self.override:
            return self.override
        if time.time() - self.read_at > 60:
            self.read_at = time.time()
            try:
                v = json.loads(ALLOC.read_text())
                for k in self.path:
                    v = v[k]
                self.value = int(v)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return self.value

    async def __aenter__(self):
        async with self.cond:
            while self.inflight >= self.current():
                try:
                    await asyncio.wait_for(self.cond.wait(), 30)
                except asyncio.TimeoutError:
                    pass
            self.inflight += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.inflight -= 1
            self.cond.notify_all()


class Fatal(RuntimeError):
    """A non-retryable response (4xx): the request itself is bad, so another server will not help."""


class Server:
    """One serving registry (all its URLs). chat(key, body) pins `key` to one URL / DP rank."""

    def __init__(self, name, timeout=6 * 3600 + 600, wait=True):
        self.name, self.wait = name, wait   # wait=False: a missing or stale registry raises instead of waiting
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30), trust_env=False,
                                      limits=httpx.Limits(max_connections=None, max_keepalive_connections=512))
        self.info, self.read_at = None, 0

    async def registry(self):
        while True:
            if self.info is None or time.time() - self.read_at > 60:
                try:
                    self.info, self.read_at = read_registry(self.name), time.time()
                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    self.info = None
                    if not self.wait:
                        raise RuntimeError(f'{self.name}: {e}')
                    print(f'{self.name}: {e}; waiting', flush=True)
                    await asyncio.sleep(60)
                    continue
            return self.info

    def dp_size(self):
        """DP ranks behind a single dp-internal url (launch.json), else 1."""
        if getattr(self, '_dp', None) is None:
            self._dp = 1
            try:
                l = json.loads((SERVING / self.name / 'launch.json').read_text())
                argv = l['replicas'][0]['argv'] if 'replicas' in l else l['argv']
                if not any(x in argv for x in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb')):
                    self._dp = int(argv[argv.index('--data-parallel-size') + 1])
            except (OSError, ValueError, KeyError, IndexError):
                pass
        return self._dp

    def target(self, info, key):
        """(url, headers, rank): `key` pins to one url and, behind a dp-internal url, one DP rank."""
        urls = info['urls']
        h = int(hashlib.sha256(str(key).encode()).hexdigest(), 16)
        if len(urls) > 1:   # one url per DP rank: the rank is the url's index
            i = h % len(urls)
            return urls[i], {'X-data-parallel-rank': str(i)}, i
        dp = self.dp_size()
        if dp > 1:
            r = h % dp
            return urls[0], {'X-data-parallel-rank': str(r)}, r
        return urls[0], {}, 0

    async def chat(self, key, body, attempts=40):
        """-> (response json, meta). Retries transport errors, 408/429/5xx and stale registries (servers restart)."""
        last = None
        for attempt in range(attempts):
            info = await self.registry()
            url, headers, rank = self.target(info, key)
            t0 = time.time()
            try:
                r = await self.http.post(url + '/chat/completions', json={'model': info['served_name'], **body},
                                         headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('choices'):
                        return data, {'url': url, 'rank': rank, 'latency_s': round(time.time() - t0, 1),
                                      'attempts': attempt + 1}
                    last = f'200 without choices: {str(data)[:200]}'
                elif r.status_code in (408, 409, 429) or r.status_code >= 500:
                    last = f'HTTP {r.status_code}: {r.text[:300]}'
                else:
                    raise Fatal(f'HTTP {r.status_code}: {r.text[:500]}')
            except (httpx.TransportError, ValueError) as e:
                last = f'{type(e).__name__}: {str(e)[:300]}'
                self.info = None   # re-read the registry: the server may have moved
            await asyncio.sleep(min(300, 5 * 2 ** min(attempt, 6)) * (0.5 + random.random()))
        raise RuntimeError(f'{self.name}: {attempts} attempts failed; last {last}')

    async def tokenize(self, text, key=0):
        info = await self.registry()
        url, _, _ = self.target(info, key)
        root = url[:-3] if url.endswith('/v1') else url
        for _ in range(5):
            try:
                r = await self.http.post(root + '/tokenize', json={'model': info['served_name'], 'prompt': text,
                                                                   'add_special_tokens': False},
                                         timeout=60)   # never the 6-h chat timeout: a stalled API must not hang us
                if r.status_code == 200:
                    return int(r.json()['count'])
            except (httpx.TransportError, ValueError, KeyError):
                pass
            await asyncio.sleep(3)
        return None

    async def close(self):
        await self.http.aclose()


def ds_serving(fallback='m5-ds'):
    try:
        names = json.loads(ALLOC.read_text())['ds']['serving']
        return [str(n) for n in names] or [fallback]
    except (OSError, ValueError, KeyError, TypeError):
        return [fallback]


class Pool:
    """The DeepSeek servers in alloc.json ds.serving (re-read every 60 s), or one pinned server. A key goes to one
    listed server with a fresh registry (prefix cache); when that server fails, goes stale or leaves the list, the
    request retries on the next one. Same chat() interface as Server."""

    def __init__(self, pinned=None, fallback='m5-ds', timeout=6 * 3600 + 600):
        self.pinned, self.fallback, self.timeout = pinned, fallback, timeout
        self.servers, self.names, self.read_at = {}, [], 0

    def serving(self):
        if self.pinned:
            return [self.pinned]
        if time.time() - self.read_at > 60:
            self.names, self.read_at = ds_serving(self.fallback), time.time()
        return self.names

    def live(self):
        out = []
        for n in self.serving():
            try:
                read_registry(n)
                out.append(n)
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
        return out

    def server(self, name):
        if name not in self.servers:
            self.servers[name] = Server(name, timeout=self.timeout, wait=False)
        return self.servers[name]

    async def chat(self, key, body, rounds=12):
        h = int(hashlib.sha256(str(key).encode()).hexdigest(), 16)
        last = None
        for rnd in range(rounds):
            names = self.live()
            if not names:
                last = f'no live server in {self.serving()}'
                await asyncio.sleep(60)
                continue
            name = names[(h + rnd) % len(names)]
            try:
                return await self.server(name).chat(key, body, attempts=5)
            except Fatal:
                raise
            except RuntimeError as e:
                last = str(e)
                if rnd:
                    await asyncio.sleep(30)
        raise RuntimeError(f'ds pool: {rounds} rounds failed; last {last}')

    async def close(self):
        for s in self.servers.values():
            await s.close()


def fields(data):
    ch = data['choices'][0]
    msg = ch.get('message') or {}
    return (msg.get('content') or '', msg.get('reasoning_content') or msg.get('reasoning') or '',
            ch.get('finish_reason'), data.get('usage') or {})


METRIC_KEYS = ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total', 'vllm:num_requests_running',
               'vllm:num_requests_waiting',
               'vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc', 'vllm:prefix_cache_hits_total',
               'vllm:prefix_cache_queries_total')


async def metrics_snapshot(server_name, http=None, timeout=30):
    """Sum of selected vLLM /metrics counters over the server's engines (for GPU-hour shares and health)."""
    try:
        info = read_registry(server_name)
    except (OSError, ValueError, KeyError, RuntimeError):
        return None
    own = http is None
    http = http or httpx.AsyncClient(timeout=timeout, trust_env=False)
    out = {'server': server_name, 't': time.time()}
    try:
        for u in info.get('engine_urls') or info['urls']:
            root = u[:-3] if u.endswith('/v1') else u
            try:
                r = await http.get(root + '/metrics')
            except httpx.HTTPError:
                continue
            for line in r.text.splitlines():
                if line.startswith('#'):
                    continue
                name = line.split('{')[0].split(' ')[0]
                if name in METRIC_KEYS:
                    try:
                        out[name] = out.get(name, 0.0) + float(line.rsplit(' ', 1)[1])
                    except ValueError:
                        pass
    finally:
        if own:
            await http.aclose()
    return out

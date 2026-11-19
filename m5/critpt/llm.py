"""Async client for the shared m5 serving endpoints (DeepSeek generator), cap-aware.

- Servers: alloc.json ds.serving (re-read every 60 s and on errors); a server whose endpoints.json is older than
  120 s is skipped; none fresh: back off and retry (never crash). A server that errors is avoided for 2 min.
- Each request is pinned to one DP rank (`X-data-parallel-rank`, round-robin over ranks with the fewest in flight);
  a single-shot request needs no stickier pinning.
- The in-flight cap is re-read from m5-coord/alloc.json every 60 s (ds.max_inflight.critpt).
- No output cap (max_tokens unset), 6 h client timeout.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx

SERVING = Path('/data/aai/tbench/tasksynth/serving')
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
WORKER = 'critpt'


def with_effort(extra: dict, effort: str) -> dict:
    kw = {**extra.get('chat_template_kwargs', {}), 'reasoning_effort': effort}
    return {**{k: v for k, v in extra.items() if k != 'reasoning_effort'}, 'chat_template_kwargs': kw,
            'reasoning_effort': effort}


class Server:
    """name='auto': the servers listed in alloc.json ds.serving (re-read every 60 s); each request goes to the
    (server, DP rank) with the fewest of our requests in flight among servers whose endpoints.json is fresh.
    A server that errors is avoided for 2 min (unless it is the only one); a removed server's in-flight requests finish
    or fail over on their retry. A fixed name pins one server."""

    def __init__(self, name: str = 'auto', effort: str | None = 'high', cap_key: str = 'ds', timeout: float = 6 * 3600):
        self.name, self.effort, self.cap_key = name, effort, cap_key
        self.infos: dict[str, dict] = {}
        self.read_at, self.cap, self.cap_at = 0.0, 1, 0.0
        self.serving_last = [name] if name != 'auto' else ['m5-ds']
        self.bad_until: dict[str, float] = {}
        self.inflight = 0
        self.gate: 'Server | None' = None
        self.rank_load: dict[tuple, int] = {}
        self.cond = asyncio.Condition()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=60), trust_env=False,
                                        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000))
        self.stats = {'requests': 0, 'errors': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

    # ---------------------------------------------------------------- endpoint + cap
    def _read_cap(self):
        if time.time() - self.cap_at < 60:
            return
        self.cap_at = time.time()
        try:
            a = json.loads(ALLOC.read_text())
            self.cap = int((a[self.cap_key]['max_inflight'] if self.cap_key == 'ds' else
                            a[self.cap_key]['max_inflight_per_engine'])[WORKER])
        except Exception as e:   # keep the last cap
            print(f'alloc.json unreadable ({e}); cap stays {self.cap}', flush=True)

    def _serving(self) -> list[str]:
        if self.name != 'auto':
            return [self.name]
        try:
            names = list(json.loads(ALLOC.read_text())[self.cap_key]['serving'])
            if names:
                self.serving_last = names
        except Exception as e:   # keep the last list
            print(f'alloc.json serving unreadable ({e}); keeping {self.serving_last}', flush=True)
        return self.serving_last

    async def _endpoints(self) -> dict[str, dict]:
        """{server: endpoints.json} for the listed servers that are healthy and fresh (< 120 s)."""
        while True:
            if not self.infos or time.time() - self.read_at > 60:
                infos = {}
                for n in self._serving():
                    try:
                        info = json.loads((SERVING / n / 'endpoints.json').read_text())
                        if info.get('healthy_at') and time.time() - info['updated_at'] < 120:
                            infos[n] = info
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                if set(infos) != set(self.infos):
                    print(f'{time.strftime("%H:%M:%S")} serving: {sorted(infos) or "none fresh"}', flush=True)
                self.infos, self.read_at = infos, time.time()
            if self.infos:
                return self.infos
            print(f'{time.strftime("%H:%M:%S")} {self._serving()}: none healthy/fresh; waiting', flush=True)
            await asyncio.sleep(60)

    @staticmethod
    def _launch_dp(name: str) -> int | None:
        """`--data-parallel-size` of the (single) replica in the server's launch.json."""
        try:
            argv = json.loads((SERVING / name / 'launch.json').read_text())['replicas'][0]['argv']
            return int(argv[argv.index('--data-parallel-size') + 1])
        except (OSError, ValueError, KeyError, IndexError):
            return None

    def _ranks(self, name: str, info) -> list[tuple[str, str, int | None]]:
        urls = info['urls']
        dp = info.get('dp_size') or info.get('data_parallel_size') or self._launch_dp(name)
        if len(urls) == 1 and dp:
            return [(name, urls[0], r) for r in range(int(dp))]
        if info.get('dp_ranks') and len(info['dp_ranks']) == len(urls):
            return [(name, u, r) for u, r in zip(urls, info['dp_ranks'])]
        return [(name, u, i) if len(urls) > 1 and info.get('pin_header', True) else (name, u, None)
                for i, u in enumerate(urls)]

    async def _acquire(self):
        async with self.cond:
            while True:
                self._read_cap()
                if self.inflight < self.cap:
                    self.inflight += 1
                    return
                try:
                    await asyncio.wait_for(self.cond.wait(), 30)
                except asyncio.TimeoutError:
                    pass

    async def _release(self):
        async with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    # ---------------------------------------------------------------- chat
    async def chat(self, messages: list[dict], retries: int = 1000, **kw) -> dict:
        """-> {'content', 'reasoning', 'prompt_tokens', 'completion_tokens', 'finish', 'seconds'}."""
        await (self.gate or self)._acquire()   # a gate shares one in-flight cap across Servers
        try:
            for attempt in range(retries):
                infos = await self._endpoints()
                now = time.time()
                ok = [n for n in infos if self.bad_until.get(n, 0) < now] or list(infos)
                targets = [x for n in ok for x in self._ranks(n, infos[n])]
                t = min(targets, key=lambda x: (self.rank_load.get(x, 0), random.random()))
                info = infos[t[0]]
                extra = dict((info.get('request') or {}).get('overrides') or {})
                if self.effort:
                    extra = with_effort(extra, self.effort)
                self.rank_load[t] = self.rank_load.get(t, 0) + 1
                body = {'model': info['served_name'], 'messages': messages, **extra, **kw}
                n_ranks = sum(1 for x in targets if x[0] == t[0])
                headers = {'X-data-parallel-rank': str(t[2])} if t[2] is not None and n_ranks > 1 else {}
                t0 = time.time()
                try:
                    r = await self.client.post(t[1].rstrip('/') + '/chat/completions', json=body, headers=headers)
                    if r.status_code >= 500 or r.status_code == 429:
                        raise httpx.HTTPStatusError(f'{r.status_code}: {r.text[:300]}', request=r.request, response=r)
                    if r.status_code >= 400:
                        raise ValueError(f'HTTP {r.status_code}: {r.text[:500]}')
                    j = r.json()
                    ch = j['choices'][0]
                    msg = ch['message']
                    u = j.get('usage') or {}
                    self.stats['requests'] += 1
                    self.stats['prompt_tokens'] += u.get('prompt_tokens', 0)
                    self.stats['completion_tokens'] += u.get('completion_tokens', 0)
                    return {'content': msg.get('content') or '',
                            'reasoning': msg.get('reasoning_content') or msg.get('reasoning') or '',
                            'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
                            'finish': ch.get('finish_reason'), 'seconds': round(time.time() - t0, 1),
                            'rank': t[2], 'server': t[0]}
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    self.stats['errors'] += 1
                    self.read_at = 0.0   # re-read alloc.json + endpoints.json
                    self.bad_until[t[0]] = time.time() + 120   # fail over to another listed server meanwhile
                    wait = min(300, 15 * 2 ** min(attempt, 4)) * (0.5 + random.random())
                    print(f'{time.strftime("%H:%M:%S")} {t[0]} {type(e).__name__}: {str(e)[:200]}; retry in '
                          f'{wait:.0f}s', flush=True)
                    await asyncio.sleep(wait)
                finally:
                    self.rank_load[t] -= 1
            raise RuntimeError('retries exhausted')
        finally:
            await (self.gate or self)._release()

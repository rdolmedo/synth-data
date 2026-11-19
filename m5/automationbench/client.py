"""DeepSeek (m5-ds) chat client for the generator: DP-rank pinning per conversation, the worker's live cap from
alloc.json (re-read every 60 s), and survival of server restarts (a stale endpoints.json = dead server: back off and
re-read, never crash). Modelled on task_synthesis/pipeline/llm.py (origin/task-synthesis).
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient

SERVING = Path('/data/aai/tbench/tasksynth/serving')
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
WORKER = 'automationbench'
TIMEOUT = 6 * 3600 + 600   # >= 6 h per request (#359 generator rule)


def read_cap(kind: str = 'ds', default: int = 8) -> int:
    try:
        a = json.loads(ALLOC.read_text())
        return int(a[kind]['max_inflight' if kind == 'ds' else 'max_inflight_per_engine'][WORKER])
    except (OSError, ValueError, KeyError, TypeError):
        return default


class Cap:
    """A semaphore whose size follows alloc.json."""

    def __init__(self, kind='ds', fixed: int | None = None):
        self.kind, self.fixed = kind, fixed
        self.n = fixed or read_cap(kind)
        self.used = 0
        self.cond = asyncio.Condition()
        self.t = time.time()

    async def __aenter__(self):
        async with self.cond:
            while True:
                if self.fixed is None and time.time() - self.t > 60:
                    self.n, self.t = read_cap(self.kind, self.n), time.time()
                if self.used < self.n:
                    self.used += 1
                    return self
                try:
                    await asyncio.wait_for(self.cond.wait(), 30)
                except asyncio.TimeoutError:
                    pass

    async def __aexit__(self, *exc):
        async with self.cond:
            self.used -= 1
            self.cond.notify_all()


class DS:
    def __init__(self, name='m5-ds', cap: Cap | None = None, effort='high'):
        self.path = SERVING / name / 'endpoints.json'
        self.cap = cap or Cap('ds')
        self.effort = effort
        self.clients, self.urls, self.dp, self.model, self.extra, self.sampling = {}, [], {}, None, {}, {}
        self.sticky = {}
        self.read_t = 0.0

    def _read(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return False
        if not info.get('healthy_at') or time.time() - info.get('updated_at', 0) > 120:
            return False
        self.model = info['served_name']
        ov = dict((info.get('request') or {}).get('overrides') or {})
        self.sampling = {k: ov.pop(k) for k in ('temperature', 'top_p') if k in ov}
        if self.effort:   # high reasoning effort (#359), wherever the profile sets one
            ov['reasoning_effort'] = self.effort
            if isinstance(ov.get('chat_template_kwargs'), dict) and 'reasoning_effort' in ov['chat_template_kwargs']:
                ov['chat_template_kwargs'] = {**ov['chat_template_kwargs'], 'reasoning_effort': self.effort}
        self.extra = ov
        self.urls = list(info['urls'])
        for u in self.urls:
            if u not in self.clients:
                lim = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
                self.clients[u] = AsyncOpenAI(base_url=u, api_key='none', timeout=TIMEOUT, max_retries=0,
                                              http_client=DefaultAsyncHttpxClient(limits=lim, timeout=TIMEOUT))
            self.dp[u] = self._dp_size(info)
        self.read_t = time.time()
        return True

    def _dp_size(self, info) -> int:
        if len(info['urls']) != 1:
            return 1
        try:
            L = json.loads((self.path.parent / 'launch.json').read_text())
            if 'argv' in L:
                argv = L['argv']
            else:   # synth launches: replicas[].argv_json (one replica behind the one URL)
                argv = json.loads(L['replicas'][0]['argv_json'])
            if any(a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb', '--data-parallel-rank')):
                return 1
            return max(1, int(argv[argv.index('--data-parallel-size') + 1]))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return 1

    async def ready(self):
        while not self._read():
            print(time.strftime('%H:%M:%S'), 'm5-ds endpoint stale or missing; waiting', flush=True)
            await asyncio.sleep(60)

    def _pin(self, key):
        t = self.sticky.get(key)
        if t is None or t[0] not in self.urls:
            counts = {}
            for v in self.sticky.values():
                counts[v] = counts.get(v, 0) + 1
            targets = [(u, r) for u in self.urls for r in range(self.dp.get(u, 1))]
            t = min(targets, key=lambda x: (counts.get(x, 0), random.random()))
            self.sticky[key] = t
        return t

    def release(self, key):
        self.sticky.pop(key, None)

    async def chat(self, messages, key, retries=12):
        """-> (content, reasoning, usage). Pinned to one (url, DP rank) per conversation `key`."""
        err = None
        for attempt in range(retries):
            if time.time() - self.read_t > 60 or not self.urls:
                await self.ready()
            url, rank = self._pin(key)
            try:
                async with self.cap:
                    r = await self.clients[url].chat.completions.create(
                        model=self.model, messages=messages, **self.sampling, extra_body=self.extra,
                        extra_headers={'X-data-parallel-rank': str(rank)} if self.dp.get(url, 1) > 1 else None)
                m = r.choices[0].message
                reasoning = getattr(m, 'reasoning_content', None) or getattr(m, 'reasoning', None) or ''
                return m.content or '', reasoning, {'prompt': r.usage.prompt_tokens,
                                                    'completion': r.usage.completion_tokens,
                                                    'finish': r.choices[0].finish_reason}
            except (APIConnectionError, APITimeoutError) as e:
                err = e
                self.release(key)
                self.read_t = 0
            except APIStatusError as e:
                if e.status_code < 500 and e.status_code not in (429, 408):
                    raise
                err = e
            await asyncio.sleep(min(600, 20 * 2 ** attempt))
        raise RuntimeError(f'DeepSeek request failed after {retries} attempts: {err}')

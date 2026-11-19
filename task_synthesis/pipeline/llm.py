"""OpenAI-compatible client over a task_synthesis.serve endpoint (all DP ranks, least-loaded first)."""
import asyncio
import json
import random
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient

from tbshared import serving   # the registry reader (#445 item 3): $TASKSYNTH_SERVING_ROOT, else production


def url_info(url, timeout=10):
    """Endpoint info for a server we did not start (an OpenAI-compatible base URL): its model, and the request
    overrides of the synth teacher spec of the same name (reasoning effort, chat template kwargs)."""
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url.rstrip('/') + '/models', timeout=timeout) as r:
        model = json.load(r)['data'][0]['id']
    try:
        from synth.config import load_teacher
        request = load_teacher(model).get('request', {})
    except (FileNotFoundError, ValueError):
        request = {}
    return {'served_name': model, 'urls': [url.rstrip('/')], 'request': request, 'healthy_at': time.time(),
            'updated_at': time.time()}


async def wait_healthy(name, timeout=3 * 3600):
    """name: a task_synthesis.serve serving name, a base URL (http://host:port/v1), or several serving names joined
    by '+' (a pool: healthy once any member is)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for n in name.split('+') if not name.startswith('http') else [name]:
            try:
                if n.startswith('http'):
                    await asyncio.to_thread(url_info, n)
                    return
                if serving.endpoint(n) is not None:   # fresh and healthy (tbshared.serving)
                    return
            except (OSError, ValueError, KeyError):
                pass
        await asyncio.sleep(30)
    raise RuntimeError(f'{name}: no healthy endpoint after {timeout}s')


class Endpoint:
    def __init__(self, name, max_in_flight_per_rank=96, max_in_flight=None, timeout=6 * 3600):
        """max_in_flight caps requests across all URLs (default 96 per URL). A dp_internal server publishes one URL
        for all its ranks, so its callers pass their own cap (r1-r3 ran at most 96 requests whatever --concurrency).
        timeout is per request and includes the time queued on the server; a timed-out request is retried from
        scratch, so it must exceed the slowest legitimate reply. Under full load DeepSeek decodes ~17 tok/s per
        request and replies run to ~100k tokens: the old 1 h cut 14% of generator replies (#25 d2/d3 rounds, ~23% of
        generator decode thrown away) and left trace-judge passes waiting hours on their longest verdicts."""
        self.name, self.path = name, serving.path(name)
        info = self._read()
        self.model = info['served_name']
        overrides = dict(info.get('request', {}).get('overrides', {}))
        self.sampling = {k: overrides.pop(k) for k in ('temperature', 'top_p') if k in overrides}
        self.extra = overrides  # reasoning_effort, chat_template_kwargs, ...
        n = max_in_flight or max_in_flight_per_rank * len(info['urls'])
        # the openai client pools at most 1,000 connections; requests beyond that queue silently for one (a judge at
        # concurrency 3000 on a two-server pool ran ~2,000 requests), so size each client's pool to the slots
        self._limits, self._timeout = httpx.Limits(max_connections=n, max_keepalive_connections=n), timeout
        self.clients = {u: self._client(u) for u in info['urls']}
        self.load = {u: 0 for u in info['urls']}
        self.dp = dict(info.get('dp', {}))
        self.slots = asyncio.Semaphore(n)
        self.refreshed = time.time()
        self.sticky = {}             # affinity key -> (url, DP rank)
        self.sessions = {}           # (url, DP rank) -> live sticky sessions

    def _client(self, u):
        return AsyncOpenAI(base_url=u, api_key='none', timeout=self._timeout, max_retries=0,
                           http_client=DefaultAsyncHttpxClient(limits=self._limits, timeout=self._timeout))

    def refresh(self):
        """Re-read the server or pool: servers that came up join the rotation and released ones leave it (2026-11-02:
        pools are the default for sandbox rounds, and servers are released as soon as the queue cannot fill them)."""
        self.refreshed = time.time()
        try:
            info = self._read()
            urls = info['urls']
        except (OSError, ValueError, KeyError, RuntimeError):
            return
        self.dp.update(info.get('dp', {}))
        for u in urls:
            if u not in self.clients:
                self.clients[u] = self._client(u)
            self.load.setdefault(u, 0)
        for u in [u for u in self.load if u not in urls]:
            del self.load[u]

    def _read(self):
        if self.name.startswith('http'):
            return url_info(self.name)
        if '+' in self.name:   # a pool of servers of one model: the URLs of every healthy member
            infos = []
            for n in self.name.split('+'):
                try:
                    infos.append(Endpoint._read_one(serving.path(n)))
                except FileNotFoundError:
                    pass   # a member that never started (e.g. a production run's spare server name)
                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    print(f'{n}: left out of the pool ({e})', flush=True)
            if not infos:
                raise RuntimeError(f'{self.name}: no healthy member')
            if len({i['served_name'] for i in infos}) > 1:
                raise RuntimeError(f'{self.name}: members serve different models')
            return {**infos[0], 'urls': [u for i in infos for u in i['urls']],
                    'dp': {u: n for i in infos for u, n in i['dp'].items()}}
        return Endpoint._read_one(self.path)

    @staticmethod
    def _read_one(path):
        info = json.loads(path.read_text())
        if not info.get('healthy_at') or serving.age(info) > serving.STALE_SEC:   # tbshared.serving's rule
            raise RuntimeError(f'{path}: endpoint not healthy or stale')
        return {**info, 'dp': {u: Endpoint._dp_size(path, info) for u in info['urls']}}

    @staticmethod
    def _dp_size(path, info):
        """DP ranks behind the one URL of a dp_internal server, from its launch.json (1 if unknown). vLLM sends each
        request without an X-data-parallel-rank header to its least-loaded rank, and every rank has its own prefix
        cache: unpinned agent steps landed on a cold rank most of the time (2026-11-03: prefix hit ~0.5 at ~4% KV use
        on the 5th A/B pool, with server-level affinity). synth campaigns pin per attempt the same way."""
        if len(info['urls']) != 1:
            return 1
        try:
            argv = json.loads((path.parent / 'launch.json').read_text())['argv']
            if any(a in argv for a in ('--data-parallel-external-lb', '--data-parallel-hybrid-lb', '--data-parallel-rank')):
                return 1
            return max(1, int(argv[argv.index('--data-parallel-size') + 1]))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return 1

    def _pin(self, affinity):
        """(url, rank) for a session: its sticky target while the URL is live, else the target with the fewest live
        sessions (then the fewest requests in flight)."""
        if affinity is None:
            return min(self.load, key=lambda u: (self.load[u], random.random())), None
        t = self.sticky.get(affinity)
        if t is None or t[0] not in self.load:
            self.release(affinity)
            targets = [(u, r) for u in self.load for r in range(self.dp.get(u, 1))]
            t = min(targets, key=lambda t: (self.sessions.get(t, 0), self.load[t[0]], random.random()))
            self.sticky[affinity] = t
            self.sessions[t] = self.sessions.get(t, 0) + 1
        return t[0], (t[1] if self.dp.get(t[0], 1) > 1 else None)

    def release(self, affinity):
        """The session `affinity` ended: its (url, rank) takes a new session."""
        t = self.sticky.pop(affinity, None)
        if t is not None and self.sessions.get(t, 0) > 0:
            self.sessions[t] -= 1

    async def chat(self, messages, max_tokens=None, retries=6, **kw):
        """-> (content, reasoning, usage dict). Retries transport errors; raises after `retries`.
        max_tokens=None leaves the reply bounded only by the server's context window."""
        content, reasoning, usage, _ = await self.chat_tools(messages, None, max_tokens=max_tokens, retries=retries, **kw)
        return content, reasoning, usage

    async def chat_tools(self, messages, tools, max_tokens=None, retries=6, affinity=None, extra=None, **kw):
        """chat() with OpenAI-style tools (the server parses the model's native tool calls, e.g. DeepSeek's DSML)
        -> (content, reasoning, usage, [{'id', 'name', 'arguments'}]). extra: this request's extra_body instead of the
        endpoint's (e.g. a reasoning effort for some sessions only; round --effort-scope asset)."""
        if max_tokens is not None:
            kw['max_tokens'] = max_tokens
        if tools:
            kw['tools'] = tools
        async with self.slots:
            for attempt in range(retries):
                if time.time() - self.refreshed > 300 or not self.load:
                    self.refresh()
                # session affinity (2026-11-03): an agent's next step re-sends its whole conversation, which is cached
                # only on the server AND DP rank that served the last step. Per-request least-loaded routing over a
                # 3-server pool sent ~2/3 of steps to a cold server, and vLLM's own DP balancer most of the rest to a
                # cold rank (prefix hit ~0.5).
                url, rank = self._pin(affinity)
                self.load[url] += 1
                try:
                    r = await self.clients[url].chat.completions.create(
                        model=self.model, messages=messages, **{**self.sampling, **kw},
                        extra_body=self.extra if extra is None else extra,
                        extra_headers={'X-data-parallel-rank': str(rank)} if rank is not None else None)
                    msg = r.choices[0].message
                    reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None) or ''
                    usage = {'prompt': r.usage.prompt_tokens, 'completion': r.usage.completion_tokens,
                             'finish': r.choices[0].finish_reason}
                    calls = [{'id': c.id, 'name': c.function.name, 'arguments': c.function.arguments}
                             for c in (getattr(msg, 'tool_calls', None) or [])]
                    return msg.content or '', reasoning, usage, calls
                except (APIConnectionError, APITimeoutError) as e:
                    err = e
                    if isinstance(e, APIConnectionError):
                        self.refresh()   # a released pool member drops out; the retry goes to a live one
                        self.release(affinity)
                except APIStatusError as e:
                    if e.status_code < 500 and e.status_code != 429:
                        raise
                    err = e
                finally:
                    if url in self.load:
                        self.load[url] -= 1
                await asyncio.sleep(min(300, 15 * 2 ** attempt))
            raise RuntimeError(f'LLM request failed after {retries} attempts: {err}')

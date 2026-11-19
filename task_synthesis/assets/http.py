"""Async HTTP for the sourcing adapters: the cluster proxy (from the environment), per-host pacing, retries with
backoff, capped streaming downloads with a SHA-256, and the GitHub token for GitHub hosts."""
import asyncio
import contextvars
import hashlib
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

UA = 'tasksynth-research/0.1 (academic research)'
# Seconds between requests to one host. GitHub search allows 30/min (code search 10/min) per token.
PACE = {'api.github.com': 0.25, 'commons.wikimedia.org': 0.2, 'zenodo.org': 2.0, 'pubchem.ncbi.nlm.nih.gov': 0.25, 'huggingface.co': 0.1,
        'eutils.ncbi.nlm.nih.gov': 0.4, 'export.arxiv.org': 3.0, 'arxiv.org': 3.0, 'www.ebi.ac.uk': 0.2}
GITHUB_HOSTS = ('api.github.com', 'codeload.github.com', 'raw.githubusercontent.com', 'github.com')
# GitHub search buckets: gate key and seconds between requests (code search 10/min, search 30/min per token; a
# little under the limits). A query that would wait more than MAX_QUEUE seconds behind a rate-limit hold fails at once.
BUCKETS = {'code_search': ('gh-code-search', 7.5), 'search': ('gh-search', 2.4)}
MAX_QUEUE = 90
# Requests made while LOW is set (bind.py sets it in a grow's background task: the queries a cell needs only for later
# specs) wait while any normal request waits for the same gate, so they use the gates' idle time only.
LOW = contextvars.ContextVar('tasksynth_http_low', default=False)


class TooLarge(Exception):
    pass


class HTTPError(Exception):
    def __init__(self, status, url, body=''):
        super().__init__(f'HTTP {status} for {url}: {body[:200]}')
        self.status = status


def github_token():
    """GITHUB_TOKEN, else `gh auth token` (gh lives in ~/.local/bin, which job PATHs lack), else gh's hosts.yml.
    Never put the token in a submit file: job ads are readable by other users."""
    tok = os.environ.get('GITHUB_TOKEN')
    if tok:
        return tok
    for gh in (shutil.which('gh'), str(Path.home() / '.local' / 'bin' / 'gh')):
        if gh and Path(gh).exists():
            try:
                tok = subprocess.run([gh, 'auth', 'token'], capture_output=True, text=True, timeout=30).stdout.strip()
                if tok:
                    return tok
            except (OSError, subprocess.SubprocessError):
                pass
    hosts = Path.home() / '.config' / 'gh' / 'hosts.yml'
    if hosts.exists():
        m = re.search(r'oauth_token:\s*(\S+)', hosts.read_text())
        if m:
            return m.group(1)
    return None


class RateGate:
    """Spacing per key; GitHub search buckets (search, code_search) are paced separately from core calls. Two
    priorities: a low-priority request goes only when no normal request waits for the key."""
    def __init__(self):
        self.next, self.locks, self.normal = {}, {}, {}

    async def wait(self, key, interval, low=False):
        lock = self.locks.setdefault(key, asyncio.Lock())
        if not low:
            self.normal[key] = self.normal.get(key, 0) + 1
        try:
            while True:
                async with lock:
                    if not (low and self.normal.get(key)):
                        now = time.monotonic()
                        t = self.next.get(key, 0)
                        if t > now:
                            await asyncio.sleep(t - now)
                        self.next[key] = max(now, t) + interval
                        return
                await asyncio.sleep(max(interval, 0.05))    # low: yield to the normal requests waiting
        finally:
            if not low:
                self.normal[key] -= 1

    def penalize(self, key, seconds):
        """A rate-limit answer: nobody on this key sends before now + seconds."""
        self.next[key] = max(self.next.get(key, 0), time.monotonic() + seconds)

    def blocked_for(self, key):
        return self.next.get(key, 0) - time.monotonic()


class Http:
    def __init__(self, concurrency=32, timeout=120):
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=True,
                                        headers={'User-Agent': UA}, limits=httpx.Limits(max_connections=concurrency))
        self.gate = RateGate()
        self.slots = asyncio.Semaphore(concurrency)
        self.gh = github_token()
        self.stats = {'requests': 0, 'bytes': 0, 'errors': 0}

    async def close(self):
        await self.client.aclose()

    def _headers(self, url, headers):
        h = dict(headers or {})
        host = urlparse(url).hostname or ''
        if self.gh and host in GITHUB_HOSTS and 'Authorization' not in h:
            h['Authorization'] = f'Bearer {self.gh}'
        if host == 'api.github.com':
            h.setdefault('Accept', 'application/vnd.github+json')
            h.setdefault('X-GitHub-Api-Version', '2022-11-28')
        return h

    async def _pace(self, url, bucket=None):
        host = urlparse(url).hostname or ''
        share = float(os.environ.get('TASKSYNTH_GH_SHARE', '1'))   # processes sharing the GitHub token
        low = LOW.get()
        if bucket in BUCKETS:
            key, interval = BUCKETS[bucket]
            if self.gate.blocked_for(key) > MAX_QUEUE:
                raise HTTPError(429, url, f'{bucket} rate-limited for {self.gate.blocked_for(key):.0f}s more (not queued)')
            await self.gate.wait(key, interval * share, low)
        await self.gate.wait(host, PACE.get(host, 0.05), low)

    async def request(self, method, url, *, params=None, headers=None, bucket=None, retries=5, **kw):
        last = None
        for attempt in range(retries):
            await self._pace(url, bucket)
            async with self.slots:
                try:
                    r = await self.client.request(method, url, params=params, headers=self._headers(url, headers), **kw)
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last = e
                    self.stats['errors'] += 1
                    await asyncio.sleep(min(120, 5 * 2 ** attempt * (1 + random.random())))
                    continue
            self.stats['requests'] += 1
            if r.status_code in (429, 500, 502, 503, 504) or (r.status_code == 403 and 'rate limit' in r.text.lower()):
                self.stats['errors'] += 1
                wait = min(600, 10 * 2 ** attempt)
                reset = r.headers.get('x-ratelimit-reset')
                again = re.search(r'try again in ([0-9.]+)s', r.text)
                if reset and r.headers.get('x-ratelimit-remaining') == '0':
                    wait = max(5, min(3600, int(reset) - time.time() + 2))
                elif r.headers.get('retry-after', '').isdigit():
                    wait = min(600, int(r.headers['retry-after']) + 1)
                elif again:
                    wait = min(900, float(again.group(1)) + 1)
                last = HTTPError(r.status_code, url, r.text)
                if bucket in BUCKETS and r.status_code in (403, 429):
                    # GitHub's secondary limit: every request that keeps knocking extends it (bf1, 2026-10-31: waits
                    # grew to 440 s and stalled whole cells). Hold the bucket for everyone; fail this query fast
                    # when the hold is long, so the cell grows from its other catalogs.
                    self.gate.penalize(BUCKETS[bucket][0], wait)
                    if wait > MAX_QUEUE:
                        raise last
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 400:
                raise HTTPError(r.status_code, url, r.text)
            return r
        raise last if isinstance(last, Exception) else RuntimeError(f'{url}: retries exhausted')

    async def json(self, url, **kw):
        return (await self.request('GET', url, **kw)).json()

    async def text(self, url, **kw):
        return (await self.request('GET', url, **kw)).text

    async def download(self, url, dest, max_bytes, headers=None, retries=4):
        """Stream url to dest; raises TooLarge past max_bytes. -> (sha256 hex, bytes)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(retries):
            await self._pace(url)
            h = hashlib.sha256()
            n = 0
            try:
                async with self.slots:
                    async with self.client.stream('GET', url, headers=self._headers(url, headers)) as r:
                        if r.status_code in (429, 500, 502, 503, 504):
                            raise HTTPError(r.status_code, url)
                        if r.status_code >= 400:
                            raise HTTPError(r.status_code, url)
                        size = r.headers.get('content-length')
                        if size and size.isdigit() and int(size) > max_bytes:
                            raise TooLarge(f'{url}: {size} bytes > {max_bytes}')
                        with open(dest, 'wb') as f:
                            async for chunk in r.aiter_bytes(1 << 20):
                                n += len(chunk)
                                if n > max_bytes:
                                    raise TooLarge(f'{url}: more than {max_bytes} bytes')
                                h.update(chunk)
                                f.write(chunk)
                self.stats['requests'] += 1
                self.stats['bytes'] += n
                return h.hexdigest(), n
            except TooLarge:
                dest.unlink(missing_ok=True)
                raise
            except HTTPError as e:
                dest.unlink(missing_ok=True)
                if e.status < 500 and e.status != 429:
                    raise
                last = e
            except (httpx.TransportError, httpx.TimeoutException) as e:
                dest.unlink(missing_ok=True)
                last = e
            self.stats['errors'] += 1
            await asyncio.sleep(min(120, 5 * 2 ** attempt))
        raise last

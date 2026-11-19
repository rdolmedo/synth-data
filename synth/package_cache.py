"""Package cache for image builds and agent sandboxes (issue #60): apt (plain HTTP) and a PyPI pull-through index on
node-local disk, one cache process per job (task-synthesis round job, campaign worker), upstream through the cluster
proxy. Design, alternatives and operations: synth/README.md "Package cache".

What is cached, and how fresh it is:
- apt (Ubuntu and Debian archives, absolute-form requests: builds' http_proxy points here, the sandbox package proxy
  forwards here). /pool/ files and /dists/.../by-hash/ indexes are immutable: cached until evicted, by-hash files
  checked against the digest in their URL. Other /dists/ files (InRelease, Release) are mutable: revalidated upstream
  with a conditional GET once older than --fresh seconds (default 300, a mirror's lag), so a build installs the
  versions upstream lists, as without the cache.
- PyPI (origin-form /pypi/simple/<project>/ and /pypi/packages/...; builds get PIP_INDEX_URL pointing here, optionally
  under /t/<tag>/ to label their requests). A project page is pypi.org's page (JSON or HTML, as the client asks) with
  the files.pythonhosted.org links made relative, so files come through here too. Pages are mutable like InRelease;
  files are immutable and checked against the BLAKE2b-256 digest in their path.
- Everything else (other hosts, other paths, CONNECT) passes through to the cluster proxy unchanged.

Robustness:
- Concurrent requests for one file share one upstream fetch; every client streams from the growing file and its last
  64 KiB is held back until the length (and digest) check passed: a client never receives a complete-looking corrupt
  file, and a truncated or corrupt download is never stored.
- Upstream unreachable: retries with backoff, then a breaker (consecutive failures -> fail fast for 30-300 s). A
  mutable file validated within --stale-max seconds (default 6 h) is then served (logged "stale"); anything else is a
  503, which apt/pip report like a proxy failure (harbor_ns rebuilds those builds, never charging the task).
- Upstream slow: a revalidation that has no answer after --revalidate-wait seconds (default 10) serves the stored copy
  (within --stale-max, logged "stale") and finishes for later clients. The cluster proxy sometimes holds a request for
  a minute; apt gives up on an index after about 60 s and pip after 15 s, and a stalled apt-get update then hung a
  build for 30 minutes (replay, s67-00285).
- A large download crawling on its proxy connection (below 1 MB/s for 30 s with over 50 MB to go; the cluster
  proxy's speed differs per connection) or broken mid-body continues on a new connection with a Range request, if the
  origin answers 206 for exactly the rest (at most 3 times). Every build waiting on the file shares that download.
- Disk full or the store over its cap: requests are streamed straight through (nothing stored); the store evicts least
  recently used files beyond --max-gib and stops writing below --min-free-gib free on the scratch filesystem.

  python -m synth.package_cache serve --root DIR --registry FILE [--seed-dir DIR]   # what CacheProcess runs
  python -m synth.package_cache stats --registry FILE                              # counters of a running cache
  python -m synth.package_cache publish --root DIR [--seed-dir DIR]                # merge a store into the seed
  python -m synth.package_cache seed-info [--seed-dir DIR]
"""
import argparse
import asyncio
import base64
import contextlib
import email.utils
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from urllib.parse import urlsplit

SEED_DIR = Path('/data/aai/tbench/tasksynth/pkgcache/seed')
APT_HOSTS = ('archive.ubuntu.com', 'security.ubuntu.com', 'ports.ubuntu.com', 'deb.debian.org', 'security.debian.org',
             'archive.debian.org')
PYPI_PAGES = 'https://pypi.org/simple/'
PYPI_FILES = 'https://files.pythonhosted.org/packages/'
FILES_LINK = PYPI_FILES.encode()
RELATIVE_FILES = b'../../packages/'   # from <prefix>/pypi/simple/<project>/ to <prefix>/pypi/packages/
HOLD = 64 << 10
CHUNK = 256 << 10
HEAD_LIMIT = 64 << 10
TAG = re.compile(r'^[A-Za-z0-9_.-]{1,120}$')
DIGEST_ALGS = {'SHA256': 'sha256', 'SHA512': 'sha512', 'MD5Sum': 'md5', 'SHA1': 'sha1'}


def now():
    return time.time()


def apt_host(host):
    host = (host or '').lower().rstrip('.')
    return any(host == h or host.endswith('.' + h) for h in APT_HOSTS)


def apt_kind(path):
    """'immutable', 'mutable' or None (not an archive file: passed through)."""
    if '/dists/' in path:
        return 'immutable' if '/by-hash/' in path else 'mutable'
    if '/pool/' in path and not path.endswith('/'):
        return 'immutable'
    return None


def by_hash_digest(path):
    """(hashlib name, hex) for .../by-hash/SHA256/<hex>, else None."""
    m = re.search(r'/by-hash/([A-Za-z0-9]+)/([0-9a-fA-F]+)$', path)
    if m and m.group(1) in DIGEST_ALGS:
        return DIGEST_ALGS[m.group(1)], m.group(2).lower()
    return None


def pypi_file_digest(rel):
    """files.pythonhosted.org stores a file at packages/<b2[:2]>/<b2[2:4]>/<b2[4:]>/<name>, b2 = its BLAKE2b-256."""
    m = re.match(r'^([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{60})/[^/]+$', rel)
    if m and not rel.endswith('.metadata'):
        return 'blake2b_256', ''.join(m.groups())
    return None


def new_hasher(alg):
    return hashlib.blake2b(digest_size=32) if alg == 'blake2b_256' else hashlib.new(alg)


def normalize_project(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def describe(url):
    """(package, version) from an archive or PyPI file name, for the fetch logs (pinning, licensing)."""
    name = url.rsplit('/', 1)[-1]
    m = re.match(r'^([^_]+)_([^_]+)_[^_]+\.u?deb$', name)
    if m:
        return m.group(1), m.group(2).replace('%3a', ':').replace('%3A', ':')
    m = re.match(r'^([A-Za-z0-9_.]+)-([0-9][^-]*?)(-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl(\.metadata)?$', name)
    if m:
        return normalize_project(m.group(1)), m.group(2)
    m = re.match(r'^([A-Za-z0-9_.-]+?)-([0-9][A-Za-z0-9_.+!-]*)\.(tar\.gz|zip|tar\.bz2)(\.metadata)?$', name)
    if m:
        return normalize_project(m.group(1)), m.group(2)
    return None, None


class Request:
    def __init__(self, method, target, version, headers, raw):
        self.method, self.target, self.version, self.headers, self.raw = method, target, version, headers, raw
        self.h = {}
        for k, v in headers:
            self.h[k.lower()] = v

    def get(self, name, default=None):
        return self.h.get(name.lower(), default)

    @property
    def keep_alive(self):
        conn = (self.get('connection') or self.get('proxy-connection') or '').lower()
        if self.version == 'HTTP/1.0':
            return 'keep-alive' in conn
        return 'close' not in conn


async def read_request(reader):
    head = await reader.readuntil(b'\r\n\r\n')
    if len(head) > HEAD_LIMIT:
        raise ValueError('request head too large')
    line, *rest = head.decode('latin-1').split('\r\n')
    method, target, version = line.split(' ', 2)
    headers = []
    for h in rest:
        if h and ':' in h:
            k, v = h.split(':', 1)
            headers.append((k.strip(), v.strip()))
    return Request(method, target, version, headers, head)


class Spec:
    """One cacheable resource: key, upstream URL, freshness class, integrity check and (for PyPI pages) rewrite."""

    def __init__(self, key, url, kind, mutable, digest=None, rewrite=False, accept=None):
        self.key, self.url, self.kind, self.mutable = key, url, kind, mutable
        self.digest, self.rewrite, self.accept = digest, rewrite, accept


class Entry:
    __slots__ = ('key', 'url', 'size', 'sha256', 'mutable', 'etag', 'last_modified', 'content_type', 'cache_control',
                 'validated', 'stored', 'used', 'path')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def meta(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != 'path'}


class UpstreamError(Exception):
    pass


class Fetch:
    """One upstream download into a temporary file, shared by every client that wants the same key."""

    def __init__(self, path):
        self.path = path
        self.written = 0
        self.length = None
        self.status = None            # upstream status
        self.headers = {}
        self.body = b''               # small non-200 bodies, relayed uncached
        self.result = None            # 'stored' | 'revalidated' | 'relay' | 'unstored' | 'error'
        self.local = False            # the error was this node's disk, not the download
        self.entry = None
        self.error = None
        self.ready = asyncio.Event()  # status/headers known (or failed)
        self.done = asyncio.Event()
        self.progress = asyncio.Event()
        self.verified = False

    def pulse(self):
        self.progress.set()
        self.progress = asyncio.Event()


class LogWriter:
    """Request log lines written by a thread, so a slow disk never stalls the event loop (stress: opening a tag file
    blocked the loop for 3.3 s, all clients with it). flush() returns once everything queued before it is written."""

    def __init__(self, root):
        self.root = Path(root)
        self.q = queue.SimpleQueue()
        self.thread = threading.Thread(target=self.run, name='log-writer', daemon=True)
        self.thread.start()

    def put(self, line, tag):
        self.q.put((line, tag))

    def flush(self, timeout=10.0):
        ev = threading.Event()
        self.q.put(ev)
        return ev.wait(timeout)

    def close(self, timeout=10.0):
        self.flush(timeout)
        self.q.put(None)

    def run(self):
        main = None
        while True:
            items = [self.q.get()]
            while len(items) < 4096:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            lines, tags, events, stop = [], {}, [], False
            for it in items:
                if it is None:
                    stop = True
                elif isinstance(it, threading.Event):
                    events.append(it)
                else:
                    lines.append(it[0])
                    if it[1]:
                        tags.setdefault(it[1], []).append(it[0])
            try:
                if lines:
                    if main is None:
                        main = open(self.root / 'logs' / 'requests.jsonl', 'a')
                    main.write(''.join(lines))
                    main.flush()
                for tag, ls in tags.items():
                    with open(self.root / 'tags' / f'{tag}.jsonl', 'a') as f:
                        f.write(''.join(ls))
            except OSError:
                pass
            for ev in events:
                ev.set()
            if stop:
                if main is not None:
                    main.close()
                return


class PackageCache:
    def __init__(self, root, upstream='http://proxy.internal:8080', fresh=300.0, stale_max=6 * 3600.0, max_gib=20.0,
                 min_free_gib=10.0, upstream_conns=24, store=True, pypi_pages=PYPI_PAGES, pypi_files=PYPI_FILES,
                 log=True, revalidate_wait=10.0):
        self.root = Path(root) if root else None
        self.upstream = upstream or None
        up = urlsplit(upstream) if upstream else None
        self.up_host, self.up_port = (up.hostname, up.port or 8080) if up else (None, None)
        self.fresh, self.stale_max, self.revalidate_wait = fresh, stale_max, revalidate_wait
        self.max_bytes, self.min_free = int(max_gib * 2**30), int(min_free_gib * 2**30)
        self.store_enabled = store and self.root is not None
        self.pypi_pages, self.pypi_files = pypi_pages, pypi_files
        self.entries = {}
        self.total = 0
        self.fetches = {}
        self.client = None
        self.upstream_slots = asyncio.Semaphore(upstream_conns)
        self.upstream_conns = upstream_conns
        self.servers = []
        self.port = None
        self.no_store_until = 0.0
        self.breaker_until, self.breaker_backoff, self.failures = 0.0, 30.0, 0
        self.stats = {'requests': 0, 'hit': 0, 'miss': 0, 'coalesced': 0, 'revalidated': 0, 'stale': 0, 'bypass': 0,
                      'passthrough': 0, 'relay': 0, 'error': 0, 'bytes_out': 0, 'upstream_requests': 0, 'upstream_bytes': 0,
                      'upstream_errors': 0, 'breaker_opens': 0, 'not_stored': 0, 'evicted': 0, 'resumed': 0,
                      'clients': 0,
                      'clients_peak': 0, 'started': now()}
        self.log_enabled = log and self.root is not None
        self.writer = None
        self.restoring, self.restored, self.closing = False, 0, False
        # a large download crawling on one proxy connection (0.4 MB/s while new connections got 2-25 MB/s; every
        # build waiting for that file waits on it) resumes on a new connection: below slow_rate bytes/s over
        # slow_window seconds with more than slow_min_remaining bytes to go, at most max_resumes times
        self.slow_window, self.slow_rate, self.slow_min_remaining, self.max_resumes = 30.0, 1 << 20, 50 << 20, 3
        self.restore_task = None

    # ------------------------------------------------------------------ store
    def obj(self, key):
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.root / 'objects' / h[:2] / h

    def load(self):
        """Index the store (files restored from a seed included); drop partial or orphaned files."""
        if not self.root:
            return
        for d in ('objects', 'tmp', 'logs', 'tags'):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        for p in (self.root / 'tmp').iterdir():
            p.unlink(missing_ok=True)
        shutil.rmtree(self.root / 'restore', ignore_errors=True)   # an interrupted seed restore's staging files
        for meta_path in (self.root / 'objects').glob('*/*.json'):
            data = meta_path.with_suffix('')
            try:
                m = json.loads(meta_path.read_text())
                st = data.stat()
            except (OSError, ValueError):
                meta_path.unlink(missing_ok=True)
                data.unlink(missing_ok=True)
                continue
            if st.st_size != m.get('size'):
                meta_path.unlink(missing_ok=True)
                data.unlink(missing_ok=True)
                continue
            e = Entry(**m, path=data)
            e.used = e.used or e.stored or 0
            self.entries[e.key] = e
            self.total += e.size
        for data in (self.root / 'objects').glob('*/*'):
            if data.suffix != '.json' and not data.with_suffix('.json').exists():
                data.unlink(missing_ok=True)

    def can_store(self):
        if not self.store_enabled or now() < self.no_store_until:
            return False
        try:
            st = os.statvfs(self.root)
        except OSError:
            return False
        return st.f_bavail * st.f_frsize > self.min_free

    def commit(self, spec, fetch, sha):
        """Move a finished download into the store. -> Entry (or None when it could not be stored)."""
        old = self.entries.pop(spec.key, None)
        if old:
            self.total -= old.size
        path = self.obj(spec.key)
        h = fetch.headers
        e = Entry(key=spec.key, url=spec.url, size=fetch.written, sha256=sha, mutable=spec.mutable,
                  etag=h.get('etag'), last_modified=h.get('last-modified'), content_type=h.get('content-type'),
                  cache_control=h.get('cache-control'), validated=now(), stored=now(), used=now(), path=path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(fetch.path, path)
            tmp_meta = self.root / 'tmp' / (path.name + '.json.' + secrets.token_hex(4))
            tmp_meta.write_text(json.dumps(e.meta()))
            os.replace(tmp_meta, path.with_suffix('.json'))
        except OSError:
            self.no_store_until = now() + 60
            path.unlink(missing_ok=True)
            return None
        self.entries[spec.key] = e
        self.total += e.size
        if self.total > self.max_bytes:
            self.evict()
        return e

    def save_meta(self, e):
        try:
            tmp = self.root / 'tmp' / (e.path.name + '.json.' + secrets.token_hex(4))
            tmp.write_text(json.dumps(e.meta()))
            os.replace(tmp, e.path.with_suffix('.json'))
        except OSError:
            pass

    def drop(self, e):
        if self.entries.get(e.key) is e:
            del self.entries[e.key]
            self.total -= e.size
        for p in (e.path.with_suffix('.json'), e.path):
            with contextlib.suppress(OSError):
                p.unlink()

    def evict(self):
        """Least recently used first, down to 90% of the cap (open files stay readable until closed)."""
        target = int(self.max_bytes * 0.9)
        for e in sorted(self.entries.values(), key=lambda e: e.used or 0):
            if self.total <= target:
                break
            if e.key in self.fetches:
                continue
            self.drop(e)
            self.stats['evicted'] += 1

    # ------------------------------------------------------------------ warm restore, while serving
    def start_restore(self, seed_dir=None):
        """Restore the current seed in the background (see restore). -> the task, or None without a seed."""
        if self.root is None or seed_current(seed_dir or SEED_DIR) is None:
            return None
        self.restore_task = asyncio.ensure_future(self.restore(seed_dir or SEED_DIR))
        return self.restore_task

    async def restore(self, seed_dir=SEED_DIR):
        """Unpack the current seed while the cache serves: a thread extracts the tar into <root>/restore and the event
        loop moves each finished file into the store, unless its key was fetched meanwhile or is being fetched (the
        live copy stays). Until then those requests are misses, as on a cold store; none waits for the restore (a
        10 GiB seed takes 15 s to minutes to read from /data). A marker file records a finished restore: a cache
        restarted in the same job resumes an interrupted one and skips a finished one. Stops when the cache closes
        or the disk reaches its floor. -> files restored."""
        seed = seed_current(seed_dir)
        done = self.root / 'seed-restored'
        if seed is None or done.exists():
            return 0
        self.restoring = True
        staging = self.root / 'restore'
        t0 = now()
        loop = asyncio.get_running_loop()
        finished = loop.create_future()

        def settle(error=None):
            if not finished.done():
                finished.set_exception(error) if error else finished.set_result(None)

        def run():
            try:
                self._extract_seed(seed, staging, loop)
            except BaseException as e:  # noqa: BLE001  (handed to the event loop)
                with contextlib.suppress(RuntimeError):   # the loop is closed: the cache has exited
                    loop.call_soon_threadsafe(settle, e)
            else:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(settle)
        # a daemon thread, not the default executor: a read stuck on a sick Lustre node (gpu143: no progress in 40 min)
        # must neither keep the cache alive after its driver dies nor block its exit
        threading.Thread(target=run, name='seed-restore', daemon=True).start()
        try:
            await finished
            if not self.closing and self.can_store():
                done.write_text(json.dumps({'seed': seed.name, 'restored': self.restored, 't': now()}) + '\n')
            print(f'seed {seed.name}: {self.restored} files restored in {now() - t0:.0f}s', flush=True)
        except (OSError, tarfile.TarError, ValueError) as e:
            print(f'seed restore failed after {self.restored} files: {e}', flush=True)
        finally:
            self.restoring = False
            if not self.closing:
                await asyncio.to_thread(shutil.rmtree, staging, True)   # else the next start's load() removes it
        return self.restored

    def _extract_seed(self, seed, staging, loop):
        """(thread) Extract data + metadata pairs; hand them to the loop in batches and wait for each to be adopted."""
        import concurrent.futures
        batch, half, last = [], set(), time.monotonic()

        def hand_over(items):
            fut = asyncio.run_coroutine_threadsafe(self._adopt(items), loop)
            while not self.closing:
                try:
                    return fut.result(timeout=1)
                except concurrent.futures.TimeoutError:
                    continue
            return False
        with tarfile.open(seed, 'r|') as tar:
            for m in tar:
                if self.closing:
                    return
                if not m.isfile() or not re.fullmatch(r'objects/[0-9a-f]{2}/[0-9a-f]{64}(\.json)?', m.name):
                    continue
                tar.extract(m, staging, filter='data')
                base = m.name[:-5] if m.name.endswith('.json') else m.name
                if base in half:
                    half.discard(base)
                    batch.append(staging / base)
                else:
                    half.add(base)
                if batch and (len(batch) >= 256 or time.monotonic() - last > 1):
                    if not hand_over(batch):
                        return
                    batch, last = [], time.monotonic()
        if batch:
            hand_over(batch)

    async def _adopt(self, bases):
        """(loop) Move restored files into the store. -> False to stop the restore (disk at its floor)."""
        for base in bases:
            meta_path = base.with_name(base.name + '.json')
            try:
                m = json.loads(meta_path.read_text())
                size = base.stat().st_size
            except (OSError, ValueError):
                continue
            key = m.get('key') if isinstance(m, dict) else None
            path = self.obj(key) if isinstance(key, str) else None
            if path is None or path.name != base.name or size != m.get('size') or key in self.entries \
                    or key in self.fetches:
                for q in (base, meta_path):
                    q.unlink(missing_ok=True)
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(base, path)
                os.replace(meta_path, path.with_suffix('.json'))
            except OSError:
                for q in (path, path.with_suffix('.json'), base, meta_path):
                    with contextlib.suppress(OSError):
                        q.unlink()
                continue
            e = Entry(**m, path=path)
            e.used = e.used or e.stored or 0
            self.entries[key] = e
            self.total += e.size
            self.restored += 1
        if self.total > self.max_bytes:
            self.evict()
        return self.can_store()

    # ------------------------------------------------------------------ logging
    def log(self, tag, **entry):
        if not self.log_enabled:
            return
        if self.writer is None:
            self.writer = LogWriter(self.root)
        self.writer.put(json.dumps({'t': round(now(), 3), 'tag': tag, **entry}) + '\n', tag)

    def flush_logs(self, timeout=10.0):
        """(blocking) Until the log lines so far are on disk."""
        return self.writer.flush(timeout) if self.writer is not None else True

    # ------------------------------------------------------------------ upstream
    def breaker_open(self):
        return now() < self.breaker_until

    def upstream_ok(self):
        self.failures = 0
        self.breaker_backoff = 30.0

    def upstream_failed(self):
        self.stats['upstream_errors'] += 1
        self.failures += 1
        if self.failures >= 5 and not self.breaker_open():
            self.breaker_until = now() + self.breaker_backoff
            self.breaker_backoff = min(300.0, self.breaker_backoff * 2)
            self.stats['breaker_opens'] += 1
            self.failures = 0

    def http(self):
        if self.client is None:
            import httpx
            self.client = httpx.AsyncClient(
                proxy=self.upstream, trust_env=False, follow_redirects=False, http2=False,
                timeout=httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=None),
                limits=httpx.Limits(max_connections=self.upstream_conns, max_keepalive_connections=self.upstream_conns,
                                    keepalive_expiry=60.0))
        return self.client

    async def open_upstream(self, spec, headers):
        """GET the upstream URL (retrying transport errors and proxy 5xx) -> an open httpx streaming response."""
        import httpx
        client = self.http()
        last = None
        for attempt in range(3):
            if self.breaker_open():
                raise UpstreamError('upstream breaker open')
            try:
                self.stats['upstream_requests'] += 1
                req = client.build_request('GET', spec.url, headers=headers)
                resp = await client.send(req, stream=True, follow_redirects=spec.kind == 'pypi-page')
            except (httpx.TransportError, OSError) as e:
                last = f'{type(e).__name__}: {e}'
                self.upstream_failed()
            else:
                if resp.status_code in (502, 503, 504):
                    last = f'upstream {resp.status_code}'
                    await resp.aclose()
                    self.upstream_failed()
                else:
                    self.upstream_ok()
                    return resp
            if attempt < 2:
                await asyncio.sleep((1, 4)[attempt])
        raise UpstreamError(last or 'upstream failed')

    async def resume_upstream(self, spec, headers, offset, length):
        """The rest of a download on a new connection. -> the response if it is the matching 206, else None."""
        try:
            resp = await self.open_upstream(spec, {**headers, 'Range': f'bytes={offset}-'})
        except UpstreamError:
            return None
        if resp.status_code == 206 and (resp.headers.get('content-range') or '').replace(' ', '') == \
                f'bytes{offset}-{length - 1}/{length}':
            return resp
        await resp.aclose()
        return None

    async def run_fetch(self, spec, entry, fetch, ua):
        """Download spec into fetch.path; commit to the store when complete and verified."""
        headers = {'Accept-Encoding': 'identity'}
        if ua:
            headers['User-Agent'] = ua
        if spec.accept:
            headers['Accept'] = spec.accept
        if entry is not None and spec.mutable:
            if entry.etag:
                headers['If-None-Match'] = entry.etag
            if entry.last_modified:
                headers['If-Modified-Since'] = entry.last_modified
        try:
            async with self.upstream_slots:
                resp = await self.open_upstream(spec, headers)
                try:
                    fetch.status = resp.status_code
                    fetch.headers = {k.lower(): v for k, v in resp.headers.items()}
                    if resp.status_code == 304 and entry is not None:
                        entry.validated = now()
                        self.save_meta(entry)
                        fetch.result, fetch.entry = 'revalidated', entry
                        fetch.ready.set()
                        return
                    if resp.status_code != 200:
                        body = b''
                        async for chunk in resp.aiter_raw():
                            body += chunk
                            if len(body) > 1 << 20:
                                break
                        fetch.body, fetch.result = body, 'relay'
                        fetch.ready.set()
                        return
                    length = resp.headers.get('content-length')
                    fetch.length = int(length) if length and not spec.rewrite else None
                    fetch.ready.set()
                    sha = hashlib.sha256()
                    check = new_hasher(spec.digest[0]) if spec.digest else None
                    body = bytearray() if spec.rewrite else None
                    resumable = fetch.length is not None and not spec.mutable and \
                        'bytes' in (resp.headers.get('accept-ranges') or '').lower()
                    import httpx
                    with open(fetch.path, 'wb') as f:
                        chunks = resp.aiter_raw(CHUNK).__aiter__()
                        win_t, win_b, resumes = now(), 0, 0
                        while True:
                            try:
                                chunk = await chunks.__anext__()
                            except StopAsyncIteration:
                                break
                            except httpx.TransportError:   # the connection stalled (read timeout) or broke
                                new = await self.resume_upstream(spec, headers, fetch.written, fetch.length) \
                                    if resumable and resumes < self.max_resumes else None
                                if new is None:
                                    raise
                                await resp.aclose()
                                resp, chunks, resumes = new, new.aiter_raw(CHUNK).__aiter__(), resumes + 1
                                self.stats['resumed'] += 1
                                win_t, win_b = now(), fetch.written
                                continue
                            if body is not None:
                                body += chunk
                                continue
                            f.write(chunk)
                            sha.update(chunk)
                            if check:
                                check.update(chunk)
                            fetch.written += len(chunk)
                            self.stats['upstream_bytes'] += len(chunk)
                            if fetch.length is not None:
                                f.flush()
                                fetch.pulse()
                            if resumable and now() - win_t >= self.slow_window:
                                if (fetch.written - win_b) / (now() - win_t) < self.slow_rate and \
                                        resumes < self.max_resumes and \
                                        fetch.length - fetch.written > self.slow_min_remaining:
                                    new = await self.resume_upstream(spec, headers, fetch.written, fetch.length)
                                    if new is not None:   # else keep the slow connection
                                        await resp.aclose()
                                        resp, chunks, resumes = new, new.aiter_raw(CHUNK).__aiter__(), resumes + 1
                                        self.stats['resumed'] += 1
                                win_t, win_b = now(), fetch.written
                        if body is not None:
                            self.stats['upstream_bytes'] += len(body)
                            data = bytes(body).replace(FILES_LINK, RELATIVE_FILES)
                            if self.pypi_files != PYPI_FILES:
                                data = data.replace(self.pypi_files.encode(), RELATIVE_FILES)
                            f.write(data)
                            sha.update(data)
                            fetch.written = len(data)
                finally:
                    await resp.aclose()
            if fetch.length is not None and fetch.written != fetch.length:
                raise UpstreamError(f'truncated: {fetch.written} of {fetch.length} bytes')
            if check and check.hexdigest() != spec.digest[1]:
                raise UpstreamError(f'{spec.digest[0]} mismatch')
            fetch.verified = True
            fetch.entry = self.commit(spec, fetch, sha.hexdigest())
            if fetch.entry is None:
                self.stats['not_stored'] += 1
            fetch.result = 'stored' if fetch.entry else 'unstored'
        except (UpstreamError, OSError, Exception) as e:  # noqa: BLE001  (httpx read errors, ENOSPC, ...)
            if isinstance(e, OSError):   # the store's disk (ENOSPC, EDQUOT, EIO): stop storing for a while
                self.no_store_until = now() + 60
                fetch.local = True
            fetch.error = f'{type(e).__name__}: {e}'
            fetch.result = 'error'
        finally:
            fetch.ready.set()
            fetch.done.set()
            fetch.pulse()
            Path(fetch.path).unlink(missing_ok=True)   # clients streaming it keep their open file
            self.fetches.pop(spec.key, None)

    def start_fetch(self, spec, entry, ua):
        f = self.fetches.get(spec.key)
        if f is not None:
            return f, False
        f = Fetch(self.root / 'tmp' / secrets.token_hex(12))
        self.fetches[spec.key] = f
        asyncio.ensure_future(self.run_fetch(spec, entry, f, ua))
        return f, True

    # ------------------------------------------------------------------ client side
    @staticmethod
    def head(code, reason, headers, keep):
        lines = [f'HTTP/1.1 {code} {reason}']
        lines += [f'{k}: {v}' for k, v in headers.items() if v is not None]
        if 'Date' not in headers:   # pip's HTTP cache keeps a max-age response only with a Date (or an ETag)
            lines.append('Date: ' + email.utils.formatdate(usegmt=True))
        lines.append('Connection: ' + ('keep-alive' if keep else 'close'))
        return ('\r\n'.join(lines) + '\r\n\r\n').encode('latin-1')

    async def simple(self, writer, code, reason, keep, body=b'', ctype='text/plain', extra=None, head_only=False):
        if isinstance(body, str):
            body = body.encode()
        writer.write(self.head(code, reason, {'Content-Type': ctype, 'Content-Length': len(body), **(extra or {})},
                               keep) + (b'' if head_only else body))
        await writer.drain()
        return code

    @staticmethod
    def etag_of(e):
        return f'"{e.sha256[:32]}"' if e.sha256 else None

    def not_modified(self, req, e):
        inm = req.get('if-none-match')
        if inm and self.etag_of(e) and self.etag_of(e) in inm:
            return True
        ims = req.get('if-modified-since')
        if ims and e.last_modified and not inm:
            from email.utils import parsedate_to_datetime
            try:
                return parsedate_to_datetime(e.last_modified) <= parsedate_to_datetime(ims)
            except (TypeError, ValueError):
                return False
        return False

    async def send_entry(self, req, writer, e, keep, how):
        """A stored file, with Range and conditional requests honored. -> (status, bytes sent)."""
        e.used = now()
        base = {'Content-Type': e.content_type or 'application/octet-stream', 'Last-Modified': e.last_modified,
                'ETag': self.etag_of(e), 'Cache-Control': e.cache_control, 'X-Pkgcache': how,
                'X-Pkgcache-Sha256': e.sha256, 'Accept-Ranges': 'bytes'}
        if self.not_modified(req, e):
            writer.write(self.head(304, 'Not Modified', base, keep))
            await writer.drain()
            return 304, 0
        start, end = 0, e.size - 1
        code, reason = 200, 'OK'
        rng = req.get('range')
        m = re.fullmatch(r'bytes=(\d+)-(\d*)', rng or '')
        if m and (not req.get('if-range') or req.get('if-range') in (self.etag_of(e), e.last_modified)):
            s, t = int(m.group(1)), int(m.group(2)) if m.group(2) else e.size - 1
            if s >= e.size:
                await self.simple(writer, 416, 'Range Not Satisfiable', keep,
                                  extra={'Content-Range': f'bytes */{e.size}'})
                return 416, 0
            start, end, code, reason = s, min(t, e.size - 1), 206, 'Partial Content'
            base['Content-Range'] = f'bytes {start}-{end}/{e.size}'
        n = end - start + 1 if e.size else 0
        writer.write(self.head(code, reason, {**base, 'Content-Length': n}, keep))
        if req.method == 'HEAD' or not n:
            await writer.drain()
            return code, 0
        with open(e.path, 'rb') as f:
            loop = asyncio.get_running_loop()
            await writer.drain()
            await loop.sendfile(writer.transport, f, start, n)
        return code, n

    @staticmethod
    def open_stream(fetch):
        """The download's file, opened before it can move: the temporary file, or the stored one once committed.
        None when it is gone (finished unstored): the caller makes its own request."""
        for path in (fetch.path, fetch.entry.path if fetch.entry else None):
            if path is not None:
                with contextlib.suppress(FileNotFoundError):
                    return open(path, 'rb')
        return None

    async def send_stream(self, writer, fetch, f, keep, method, how='miss'):
        """Stream a download in progress from its file; the last HOLD bytes wait for verification."""
        hdrs = fetch.headers
        # Cache-Control as upstream sent it (PyPI files: immutable), as for a stored file: pip keeps a response in its
        # own HTTP cache only when the headers allow it, and a build's ~/.cache/pip should fill as without the cache
        writer.write(self.head(200, 'OK', {'Content-Type': hdrs.get('content-type') or 'application/octet-stream',
                                           'Content-Length': fetch.length, 'Last-Modified': hdrs.get('last-modified'),
                                           'Cache-Control': hdrs.get('cache-control'), 'X-Pkgcache': how}, keep))
        if method == 'HEAD':
            await writer.drain()
            return 0
        sent = 0
        with f:
            while True:
                progress = fetch.progress
                finished = fetch.done.is_set()
                limit = fetch.written if finished and fetch.verified else max(0, fetch.written - HOLD)
                if finished and not fetch.verified:
                    raise ConnectionAbortedError(fetch.error or 'download failed')
                while sent < limit:
                    data = f.read(min(CHUNK, limit - sent))
                    if not data:
                        break
                    writer.write(data)
                    sent += len(data)
                    await writer.drain()
                if finished:
                    return sent
                await progress.wait()

    async def serve_spec(self, req, writer, spec, tag, keep):
        """A cacheable request. -> keep the connection open?"""
        t0 = now()
        self.stats['requests'] += 1
        e = self.entries.get(spec.key)
        how, code, sent, err = None, None, 0, None
        if e is not None and not e.path.exists():
            self.drop(e)
            e = None
        fetch = None
        if e is not None and (not spec.mutable or now() - (e.validated or 0) < self.fresh):
            how = 'hit'
        elif spec.key not in self.fetches and not self.can_store():
            # storeless mode, disk full or store over its floor: one direct request for this client
            return await self.serve_direct(req, writer, spec, tag, keep, t0)
        else:
            fetch, leader = self.start_fetch(spec, e, req.get('user-agent'))
            if e is not None and spec.mutable and now() - (e.validated or 0) < self.stale_max:
                # a slow revalidation: serve the stored copy now; the revalidation completes for later clients
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(fetch.ready.wait(), self.revalidate_wait)
                if not fetch.ready.is_set():
                    how, err, fetch = 'stale', f'revalidation slower than {self.revalidate_wait:g}s', None
            else:
                await fetch.ready.wait()
        if fetch is not None:
            f = None
            if fetch.result in (None, 'stored', 'unstored') and fetch.length is not None and fetch.status == 200:
                f = self.open_stream(fetch)
                if f is None:
                    return await self.serve_direct(req, writer, spec, tag, keep, t0)
            if f is not None:
                how = 'miss' if leader else 'coalesced'
                try:
                    sent = await self.send_stream(writer, fetch, f, keep, req.method, how)
                    code = 200
                except ConnectionAbortedError as x:
                    err = str(x)
                    keep, code = False, 0
                    writer.transport.abort()
            else:
                await fetch.done.wait()
                if fetch.result == 'revalidated':
                    how, e = 'revalidated', fetch.entry
                elif fetch.result == 'stored':
                    how, e = ('miss' if leader else 'coalesced'), fetch.entry
                elif fetch.result == 'unstored':
                    # stored nowhere (disk full): this client is served by a direct request of its own
                    return await self.serve_direct(req, writer, spec, tag, keep, t0)
                elif fetch.result == 'relay':
                    how, code = 'relay', fetch.status
                    hdr = {'Content-Type': fetch.headers.get('content-type') or 'text/plain', 'X-Pkgcache': 'relay'}
                    if fetch.headers.get('location'):
                        hdr['Location'] = fetch.headers['location']
                    await self.simple(writer, code, 'Upstream', keep, fetch.body, hdr['Content-Type'], hdr,
                                      head_only=req.method == 'HEAD')
                    sent = len(fetch.body)
                elif fetch.local and fetch.written == 0:
                    # the store's disk failed before anything was sent: a direct request for this client
                    return await self.serve_direct(req, writer, spec, tag, keep, t0)
                else:   # error
                    err = fetch.error
                    if e is not None and spec.mutable and now() - (e.validated or 0) < self.stale_max:
                        how = 'stale'
                    else:
                        how, code = 'error', 503
                        await self.simple(writer, 503, 'Service Unavailable', keep,
                                          f'pkgcache: upstream unreachable ({err})\n')
        if how in ('hit', 'revalidated', 'stale') or how in ('miss', 'coalesced') and code is None:
            code, sent = await self.send_entry(req, writer, e, keep, how)
        self.stats[how if how in self.stats else 'error'] += 1
        self.stats['bytes_out'] += sent
        if fetch is not None and fetch.entry is not None and how in ('miss', 'coalesced', 'revalidated'):
            sha = fetch.entry.sha256
        elif how in ('hit', 'stale') and e is not None:
            sha = e.sha256
        else:
            sha = None
        pkg, ver = describe(spec.url)
        self.log(tag, kind=spec.kind, url=spec.url, status=code, cache=how, bytes=sent, ms=round(1000 * (now() - t0)),
                 **({'sha256': sha} if sha else {}), **({'package': pkg, 'version': ver} if pkg else {}),
                 **({'error': err[:200]} if err else {}))
        return keep

    async def serve_direct(self, req, writer, spec, tag, keep, t0):
        """No store (disk full, storeless mode): one upstream request streamed straight to this client."""
        headers = {'Accept-Encoding': 'identity'}
        if spec.accept:
            headers['Accept'] = spec.accept
        code, sent, err = 503, 0, None
        try:
            async with self.upstream_slots:
                resp = await self.open_upstream(spec, headers)
                try:
                    code = resp.status_code
                    ctype = resp.headers.get('content-type') or 'application/octet-stream'
                    if spec.rewrite or code != 200:
                        body = await resp.aread()
                        self.stats['upstream_bytes'] += len(body)
                        if spec.rewrite and code == 200:
                            body = body.replace(FILES_LINK, RELATIVE_FILES)
                            if self.pypi_files != PYPI_FILES:
                                body = body.replace(self.pypi_files.encode(), RELATIVE_FILES)
                        await self.simple(writer, code, 'OK' if code == 200 else 'Upstream', keep, body, ctype,
                                          {'X-Pkgcache': 'bypass'}, head_only=req.method == 'HEAD')
                        sent = len(body)
                    else:
                        length = resp.headers.get('content-length')
                        check = new_hasher(spec.digest[0]) if spec.digest else None
                        writer.write(self.head(200, 'OK', {'Content-Type': ctype, 'Content-Length': length,
                                                           'X-Pkgcache': 'bypass'}, keep and length is not None))
                        if length is None:
                            keep = False
                        held = b''
                        async for chunk in resp.aiter_raw(CHUNK):
                            self.stats['upstream_bytes'] += len(chunk)
                            if check:
                                check.update(chunk)
                            data = held + chunk
                            held, data = data[-HOLD:], data[:-HOLD]
                            if data and req.method != 'HEAD':
                                writer.write(data)
                                sent += len(data)
                                await writer.drain()
                        if length is not None and sent + len(held) != int(length):
                            raise UpstreamError('truncated')
                        if check and check.hexdigest() != spec.digest[1]:
                            raise UpstreamError(f'{spec.digest[0]} mismatch')
                        if req.method != 'HEAD':
                            writer.write(held)
                            sent += len(held)
                            await writer.drain()
                finally:
                    await resp.aclose()
            how = 'bypass'
        except UpstreamError as e:
            err, how = str(e), 'error'
            if not sent and not writer.is_closing():
                await self.simple(writer, 503, 'Service Unavailable', keep, f'pkgcache: upstream unreachable ({e})\n')
                code = 503
            else:
                keep = False
                writer.transport.abort()
        self.stats[how] += 1
        self.stats['bytes_out'] += sent
        self.log(tag, kind=spec.kind, url=spec.url, status=code, cache=how, bytes=sent, ms=round(1000 * (now() - t0)),
                 **({'error': err[:200]} if err else {}))
        return keep

    # ------------------------------------------------------------------ pass-through
    async def passthrough(self, req, reader, writer, tag):
        """Not cacheable: the request goes to the cluster proxy as is (Connection: close), the reply comes back."""
        self.stats['passthrough'] += 1
        t0 = now()
        try:
            up_r, up_w = await asyncio.wait_for(asyncio.open_connection(self.up_host, self.up_port), 30)
        except (OSError, asyncio.TimeoutError):
            await self.simple(writer, 502, 'Bad Gateway', False, 'pkgcache: cluster proxy unreachable\n')
            self.log(tag, kind='pass', url=req.target[:300], status=502, cache='pass')
            return False
        if req.method == 'CONNECT':
            up_w.write(f'CONNECT {req.target} HTTP/1.1\r\nHost: {req.target}\r\n\r\n'.encode())
        else:
            kept = [(k, v) for k, v in req.headers if k.lower() not in
                    ('connection', 'proxy-connection', 'keep-alive', 'proxy-authorization', 'x-pkgcache-tag')]
            up_w.write(('\r\n'.join([f'{req.method} {req.target} {req.version}', *(f'{k}: {v}' for k, v in kept),
                                     'Connection: close', '', ''])).encode('latin-1'))
        await up_w.drain()
        await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))
        self.log(tag, kind='pass', url=req.target[:300], method=req.method, cache='pass', ms=round(1000 * (now() - t0)))
        return False

    # ------------------------------------------------------------------ routing
    def route(self, req):
        """-> (Spec | 'local' | 'pass' | 'redirect', tag, path)"""
        tag = req.get('x-pkgcache-tag')
        auth = req.get('proxy-authorization') or ''
        if not tag and auth.lower().startswith('basic '):
            with contextlib.suppress(ValueError, UnicodeDecodeError):
                tag = base64.b64decode(auth[6:].strip()).decode().split(':', 1)[0]
        target = req.target
        if req.method == 'CONNECT':
            return 'pass', tag, None
        if target.startswith('http://'):
            u = urlsplit(target)
            if u.hostname in ('127.0.0.1', 'localhost') and (u.port or 80) == self.port:
                target = target[len(f'http://{u.netloc}'):] or '/'
            elif req.method in ('GET', 'HEAD') and apt_host(u.hostname) and (u.port or 80) == 80 and not u.query:
                kind = apt_kind(u.path)
                if kind:
                    url = f'http://{u.hostname.lower()}{u.path}'
                    return Spec('apt:' + url, url, 'apt', kind == 'mutable', digest=by_hash_digest(u.path)), tag, None
                return 'pass', tag, None
            else:
                return 'pass', tag, None
        if not target.startswith('/'):
            return 'pass', tag, None
        path = target.split('?', 1)[0]
        m = re.match(r'^/t/([^/]+)(/.*)$', path)
        if m:
            tag, path = (m.group(1) if TAG.match(m.group(1)) else tag), m.group(2)
        if tag and not TAG.match(tag):
            tag = None
        if req.method not in ('GET', 'HEAD'):
            return 'local', tag, path
        m = re.match(r'^/pypi/simple/([^/]+)(/?)$', path)
        if m:
            if not m.group(2):
                return 'redirect', tag, target.split('?', 1)[0] + '/'
            project = normalize_project(m.group(1))
            accept = req.get('accept') or ''
            json_ = 'application/vnd.pypi.simple.v1+json' in accept
            variant = 'json' if json_ else 'html'
            return Spec(f'pypi-page:{variant}:{project}', f'{self.pypi_pages}{project}/', 'pypi-page', True,
                        rewrite=True, accept=('application/vnd.pypi.simple.v1+json' if json_ else
                                              'application/vnd.pypi.simple.v1+html, text/html;q=0.1')), tag, path
        m = re.match(r'^/pypi/packages/(.+)$', path)
        if m and '..' not in m.group(1).split('/'):
            rel = m.group(1)
            return Spec(f'pypi-file:{rel}', self.pypi_files + rel, 'pypi-file', False,
                        digest=pypi_file_digest(rel)), tag, path
        return 'local', tag, path

    async def handle(self, reader, writer):
        self.stats['clients'] += 1
        self.stats['clients_peak'] = max(self.stats['clients_peak'], self.stats['clients'])
        try:
            while True:
                try:
                    req = await asyncio.wait_for(read_request(reader), 300)   # idle keep-alive
                except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError,
                        ConnectionError):
                    return
                keep = req.keep_alive
                spec, tag, path = self.route(req)
                if isinstance(spec, Spec):
                    keep = await self.serve_spec(req, writer, spec, tag, keep)
                elif spec == 'pass':
                    keep = await self.passthrough(req, reader, writer, tag)
                elif spec == 'redirect':
                    await self.simple(writer, 301, 'Moved Permanently', keep, extra={'Location': path})
                elif path == '/_pkgcache/flush':   # the builder reads a build's tag file after this
                    ok = await asyncio.to_thread(self.flush_logs, 10.0)
                    await self.simple(writer, 200 if ok else 503, 'OK' if ok else 'Busy', keep, 'flushed\n')
                elif path == '/_pkgcache/health':
                    await self.simple(writer, 200, 'OK', keep, 'ok\n')
                elif path == '/_pkgcache/stats':
                    await self.simple(writer, 200, 'OK', keep, json.dumps(self.snapshot()) + '\n', 'application/json')
                else:
                    await self.simple(writer, 404, 'Not Found', keep, 'pkgcache: not a cached registry path\n')
                if not keep:
                    return
        except (ConnectionError, OSError):
            return
        finally:
            self.stats['clients'] -= 1
            with contextlib.suppress(Exception):
                writer.close()

    def snapshot(self):
        return {**self.stats, 'entries': len(self.entries), 'store_bytes': self.total, 'fetches': len(self.fetches),
                'breaker_open': self.breaker_open(), 'storing': self.can_store(), 'port': self.port,
                'restoring': self.restoring, 'restored': self.restored}

    async def start(self, host='127.0.0.1', port=0, sock=None):
        if not self.entries:
            self.load()
        srv = await asyncio.start_server(self.handle, host, port, limit=HEAD_LIMIT * 2, backlog=1024)
        self.servers.append(srv)
        self.port = srv.sockets[0].getsockname()[1]
        if sock:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(sock)
            s = await asyncio.start_unix_server(self.handle, path=str(sock), limit=HEAD_LIMIT * 2, backlog=1024)
            os.chmod(sock, 0o600)
            self.servers.append(s)
        return self

    async def close(self):
        self.closing = True   # the restore thread stops at its next file; adopted files are complete (os.replace)
        if self.restore_task is not None and not self.restore_task.done():
            self.restore_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.restore_task
        for s in self.servers:
            s.close()
        for s in self.servers:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(s.wait_closed(), 5)
        if self.client is not None:
            await self.client.aclose()
        if self.writer is not None:
            await asyncio.to_thread(self.writer.close)


async def pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


# ---------------------------------------------------------------------- seed (warm restore, few large files on /data)
def seed_current(seed_dir=SEED_DIR):
    try:
        cur = json.loads((Path(seed_dir) / 'current.json').read_text())
        path = Path(seed_dir) / cur['file']
        return path if path.exists() else None
    except (OSError, ValueError, KeyError):
        return None


class SeedLock:
    """mkdir mutex (Lustre has no flock); a holder older than `stale` seconds is presumed dead."""

    def __init__(self, path, stale=3600):
        self.path, self.stale = Path(path), stale

    def __enter__(self):
        while True:
            try:
                self.path.mkdir()
                return self
            except FileExistsError:
                try:
                    if now() - self.path.stat().st_mtime > self.stale:
                        self.path.rmdir()
                        continue
                except OSError:
                    continue
                time.sleep(5)

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            self.path.rmdir()


def publish(root, seed_dir=SEED_DIR, max_gib=12.0, tmp=None):
    """Merge a store into a new seed tar: this store's files plus the current seed's, most recently used first, up to
    max_gib. Built in node-local scratch, copied to /data with dd (direct I/O), then current.json is switched; the
    seed before the previous one is deleted (a reader still holding it open keeps reading). -> summary dict."""
    root, seed_dir = Path(root), Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp or os.environ.get('TMPDIR') or '/tmp')
    with SeedLock(seed_dir / '.lock'):
        items = {}
        for meta_path in (root / 'objects').glob('*/*.json'):
            with contextlib.suppress(OSError, ValueError):
                m = json.loads(meta_path.read_text())
                if meta_path.with_suffix('').stat().st_size == m['size']:
                    items[m['key']] = ('local', meta_path, m)
        old = seed_current(seed_dir)
        if old:
            with tarfile.open(old, 'r|') as tar:
                for mem in tar:
                    if mem.name.endswith('.json') and mem.isfile():
                        m = json.loads(tar.extractfile(mem).read())
                        if m['key'] not in items:
                            items[m['key']] = ('seed', mem.name, m)
        chosen, total = [], 0
        for src, ref, m in sorted(items.values(), key=lambda x: -(x[2].get('used') or 0)):
            if total + m['size'] > max_gib * 2**30:
                continue
            chosen.append((src, ref, m))
            total += m['size']
        from_seed = {ref[:-5] for src, ref, m in chosen if src == 'seed'}
        name = f'seed-{time.strftime("%Y%m%d-%H%M%S")}-{secrets.token_hex(3)}.tar'
        work = tmp / name
        with tarfile.open(work, 'w') as out:
            for src, ref, m in chosen:
                if src == 'local':
                    out.add(ref.with_suffix(''), arcname=f'objects/{ref.parent.name}/{ref.with_suffix("").name}')
                    out.add(ref, arcname=f'objects/{ref.parent.name}/{ref.name}')
            if old and from_seed:
                with tarfile.open(old, 'r|') as tar:
                    for mem in tar:
                        base = mem.name[:-5] if mem.name.endswith('.json') else mem.name
                        if base in from_seed and mem.isfile():
                            out.addfile(mem, tar.extractfile(mem))
        dst = seed_dir / name
        subprocess.run(['dd', f'if={work}', f'of={dst}.partial', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
        os.replace(f'{dst}.partial', dst)
        work.unlink(missing_ok=True)
        summary = {'file': name, 'entries': len(chosen), 'bytes': total, 'created': now(),
                   'previous': old.name if old else None}
        cur_tmp = seed_dir / f'current.json.{secrets.token_hex(4)}'
        cur_tmp.write_text(json.dumps(summary, indent=1) + '\n')
        os.replace(cur_tmp, seed_dir / 'current.json')
        keep = {name, old.name if old else None}
        for p in seed_dir.glob('seed-*.tar'):
            if p.name not in keep:
                p.unlink(missing_ok=True)
        return summary


# ---------------------------------------------------------------------- process supervision (inside the job)
class CacheProcess:
    """Runs `python -m synth.package_cache serve` as a child of this job's driver (round job, campaign worker) and
    keeps it up: a dead cache is restarted after a backoff, and its registry file is removed while it is down so
    builds fall back to the cluster proxy (harbor_ns.sandbox.package_cache). Sets HARBOR_NS_PACKAGE_CACHE for this
    process and its children."""

    def __init__(self, root, registry=None, seed_dir=SEED_DIR, log=print, args=()):
        self.root = Path(root)
        self.registry = Path(registry or self.root.parent / 'pkgcache.json')
        self.seed_dir, self.logf, self.args = seed_dir, log, list(args)
        self.proc = None
        self.port = None
        self.watch = None
        self.stopping = False
        self.restarts = 0

    async def spawn(self):
        self.registry.unlink(missing_ok=True)
        cmd = [sys.executable, '-m', 'synth.package_cache', 'serve', '--root', str(self.root),
               '--registry', str(self.registry), *self.args]
        if self.port:   # a restarted cache takes its old port back if free: apt retrying in a running step reconnects
            cmd += ['--port', str(self.port)]
        if self.seed_dir:
            cmd += ['--seed-dir', str(self.seed_dir)]
        self.root.mkdir(parents=True, exist_ok=True)
        err = open(self.root.parent / 'pkgcache.err', 'ab')
        self.proc = await asyncio.create_subprocess_exec(*cmd, stdin=subprocess.DEVNULL, stdout=err, stderr=err,
                                                         env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        err.close()
        for _ in range(1200):   # interpreter start (the seed is restored in the background, after this)
            if self.proc.returncode is not None:
                return False
            if self.registry.exists():
                with contextlib.suppress(OSError, ValueError, KeyError):
                    self.port = json.loads(self.registry.read_text())['port']
                return True
            await asyncio.sleep(0.1)
        return False

    async def start(self):
        os.environ['HARBOR_NS_PACKAGE_CACHE'] = str(self.registry)
        ok = await self.spawn()
        self.logf(f'package cache: {"up " + self.url() if ok else "did not start; builds use the cluster proxy"} '
                  f'(root {self.root}, log {self.root.parent / "pkgcache.err"})')
        self.watch = asyncio.ensure_future(self.supervise())
        return self

    def url(self):
        try:
            return json.loads(self.registry.read_text())['url']
        except (OSError, ValueError, KeyError):
            return None

    async def supervise(self):
        backoff = 5
        while not self.stopping:
            if self.proc is not None and self.proc.returncode is None:
                await self.proc.wait()
            if self.stopping:
                return
            self.registry.unlink(missing_ok=True)
            self.restarts += 1
            self.logf(f'package cache exited (rc {self.proc.returncode if self.proc else "?"}); builds use the cluster '
                      f'proxy; restart {self.restarts} in {backoff}s')
            await asyncio.sleep(backoff)
            backoff = min(300, backoff * 2)
            if await self.spawn():
                self.logf(f'package cache restarted: {self.url()}')
                backoff = 5

    def stats(self):
        url = self.url()
        if not url:
            return None
        import urllib.request
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + '/_pkgcache/stats',
                                                                                 timeout=5) as r:
                return json.loads(r.read())
        except (OSError, ValueError):
            return None

    async def close(self):
        self.stopping = True
        if self.watch:
            self.watch.cancel()
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.registry.unlink(missing_ok=True)


async def serve(a):
    import resource
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    with contextlib.suppress(ValueError, OSError):   # one descriptor per client and per open file
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    cache = PackageCache(root, upstream=a.upstream, fresh=a.fresh, stale_max=a.stale_max, max_gib=a.max_gib,
                         min_free_gib=a.min_free_gib, upstream_conns=a.upstream_conns,
                         revalidate_wait=a.revalidate_wait)
    try:
        await cache.start(port=a.port, sock=root / 'cache.sock')
    except OSError:   # the preferred port is taken
        await cache.start(sock=root / 'cache.sock')
    restoring = cache.start_restore(a.seed_dir) if a.seed_dir and a.seed_dir != 'none' else None
    reg = {'url': f'http://127.0.0.1:{cache.port}', 'port': cache.port, 'pid': os.getpid(), 'root': str(root),
           'sock': str(root / 'cache.sock'), 'host': socket.gethostname(), 'started': now(),
           'restoring': restoring is not None, 'entries': len(cache.entries)}
    tmp = Path(a.registry).with_name(Path(a.registry).name + f'.{os.getpid()}')
    tmp.write_text(json.dumps(reg) + '\n')
    os.replace(tmp, a.registry)
    print(f'package cache {reg["url"]} root {root}: {len(cache.entries)} files ({cache.total / 2**30:.2f} GiB)'
          + (f', restoring {seed_current(a.seed_dir)} in the background' if restoring else ''), flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    beat = asyncio.ensure_future(heartbeat(cache))
    start_watchdog(cache)
    parent = os.getppid()
    last = 0
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), 5)
        if os.getppid() != parent:   # the driver died: never outlive the job's driver
            break
        if now() - last > 600:
            last = now()
            print(json.dumps({'t': round(now()), **cache.snapshot()}), flush=True)
    with contextlib.suppress(FileNotFoundError):
        if json.loads(Path(a.registry).read_text()).get('pid') == os.getpid():
            os.unlink(a.registry)
    await cache.close()


_BEAT = [time.monotonic()]


async def heartbeat(cache):
    """Event-loop lag: how late a 0.5 s sleep wakes up (stats loop_lag_max_s)."""
    while True:
        t = time.monotonic()
        _BEAT[0] = t
        await asyncio.sleep(0.5)
        cache.stats['loop_lag_max_s'] = round(max(cache.stats.get('loop_lag_max_s', 0), time.monotonic() - t - 0.5), 3)


def start_watchdog(cache, threshold=3.0):
    """A thread that prints the event loop's stack when the loop has not run for `threshold` seconds: everything the
    cache does happens on that loop, so a stall is one stack (stress, 2,400 sandboxes: ~20 s stalls)."""
    import traceback
    main = threading.main_thread().ident

    def watch():
        reported = False
        while True:
            time.sleep(1)
            lag = time.monotonic() - _BEAT[0]
            if lag > threshold and not reported:
                frame = sys._current_frames().get(main)
                print(f'event loop blocked for {lag:.1f}s; its stack:\n' + ''.join(traceback.format_stack(frame)),
                      flush=True)
                cache.stats['loop_stalls'] = cache.stats.get('loop_stalls', 0) + 1
                reported = True
            elif lag < 1:
                reported = False
    threading.Thread(target=watch, name='loop-watchdog', daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('serve')
    p.add_argument('--root', required=True, help='store directory on node-local disk (never Lustre)')
    p.add_argument('--registry', required=True, help='JSON file the cache writes its URL into')
    p.add_argument('--seed-dir', default=str(SEED_DIR), help="warm-restore source ('none' to start cold)")
    p.add_argument('--upstream', default=os.environ.get('PKGCACHE_UPSTREAM', 'http://proxy.internal:8080'))
    p.add_argument('--fresh', type=float, default=300.0, help='seconds a mutable index is served without revalidation')
    p.add_argument('--stale-max', type=float, default=6 * 3600.0,
                   help='upstream down: serve a mutable index validated at most this many seconds ago')
    p.add_argument('--max-gib', type=float, default=20.0)
    p.add_argument('--min-free-gib', type=float, default=10.0)
    p.add_argument('--upstream-conns', type=int, default=24)
    p.add_argument('--revalidate-wait', type=float, default=10.0,
                   help='seconds a client waits on a revalidation before it gets the stored copy (within --stale-max)')
    p.add_argument('--port', type=int, default=0, help='preferred port (a restart reuses the old one if free)')
    p = sub.add_parser('stats')
    p.add_argument('--registry', required=True)
    p = sub.add_parser('publish')
    p.add_argument('--root', required=True)
    p.add_argument('--seed-dir', default=str(SEED_DIR))
    p.add_argument('--max-gib', type=float, default=12.0)
    p = sub.add_parser('seed-info')
    p.add_argument('--seed-dir', default=str(SEED_DIR))
    a = ap.parse_args()
    if a.op == 'serve':
        asyncio.run(serve(a))
    elif a.op == 'stats':
        url = json.loads(Path(a.registry).read_text())['url']
        import urllib.request
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + '/_pkgcache/stats') as r:
            print(json.dumps(json.loads(r.read()), indent=1))
    elif a.op == 'publish':
        print(json.dumps(publish(a.root, a.seed_dir, a.max_gib), indent=1))
    else:
        print((Path(a.seed_dir) / 'current.json').read_text())


if __name__ == '__main__':
    main()

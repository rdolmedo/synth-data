"""Package cache (synth/package_cache.py, issue #60) against a fake upstream proxy: what is stored and for how long,
integrity (truncated or corrupt downloads never stored or completed), upstream failures, the PyPI index, the seed,
and the clients (harbor_ns build env, synth.package_proxy chain)."""
import asyncio
import hashlib
import json
import re
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from synth import package_cache as pc
from synth.package_cache import PackageCache


class FakeUpstream:
    """The cluster proxy and the origins behind it: absolute-form GETs answered from `files` (url -> bytes), with
    ETag/If-None-Match, plus failure modes."""

    def __init__(self):
        self.files, self.etags, self.count = {}, {}, {}
        self.mode = 'ok'          # ok | 503 | truncate | slow | refuse
        self.server, self.port = None, None
        self.passed = []
        self.hold = 0.0           # seconds before answering (a stalled proxy)
        self.ranges = False       # answer Range requests (206) and advertise Accept-Ranges
        self.trickle_first = 0    # the first GET of a URL sends this many bytes per 50 ms (a crawling connection)
        self.break_first = False  # the first GET of a URL sends half the body and closes
        self.ranged = []
        self.cache_control = None # Cache-Control on 200 responses

    def put(self, url, data):
        self.files[url] = data
        self.etags[url] = '"' + hashlib.md5(data).hexdigest() + '"'

    async def start(self):
        self.server = await asyncio.start_server(self.handle, '127.0.0.1', 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def close(self):
        self.server.close()

    async def handle(self, reader, writer):
        try:
            while True:
                head = await reader.readuntil(b'\r\n\r\n')
                line, *hs = head.decode().split('\r\n')
                method, url, _ = line.split(' ', 2)
                h = {k.lower(): v.strip() for k, _, v in (x.partition(':') for x in hs if x)}
                self.count[url] = self.count.get(url, 0) + 1
                if self.hold:
                    await asyncio.sleep(self.hold)
                if self.mode == 'refuse':
                    writer.close()
                    return
                if self.mode == '503':
                    writer.write(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
                    await writer.drain()
                    continue
                if url not in self.files:
                    self.passed.append(url)
                    writer.write(b'HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found')
                    await writer.drain()
                    if h.get('connection') == 'close':
                        writer.close()
                        return
                    continue
                data, etag = self.files[url], self.etags[url]
                if h.get('if-none-match') == etag:
                    writer.write(f'HTTP/1.1 304 Not Modified\r\nETag: {etag}\r\n\r\n'.encode())
                    await writer.drain()
                    continue
                m = re.fullmatch(r'bytes=(\d+)-', h.get('range', '')) if self.ranges else None
                if m:
                    off = int(m.group(1))
                    self.ranged.append((url, off))
                    writer.write(f'HTTP/1.1 206 Partial Content\r\nContent-Length: {len(data) - off}\r\n'
                                 f'Content-Range: bytes {off}-{len(data) - 1}/{len(data)}\r\nETag: {etag}\r\n'
                                 f'Accept-Ranges: bytes\r\n\r\n'.encode() + data[off:])
                    await writer.drain()
                    continue
                length = len(data) + (100 if self.mode == 'truncate' else 0)
                writer.write((f'HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nETag: {etag}\r\n'
                              + ('Accept-Ranges: bytes\r\n' if self.ranges else '')
                              + (f'Cache-Control: {self.cache_control}\r\n' if self.cache_control else '')
                              + 'Content-Type: application/octet-stream\r\n\r\n').encode())
                if self.break_first and self.count[url] == 1:
                    writer.write(data[:len(data) // 2])
                    await writer.drain()
                    writer.close()
                    return
                if self.trickle_first and self.count[url] == 1:
                    try:
                        for i in range(0, len(data), self.trickle_first):
                            writer.write(data[i:i + self.trickle_first])
                            await writer.drain()
                            await asyncio.sleep(0.05)
                    except ConnectionError:
                        return
                    continue
                if self.mode == 'slow':
                    for i in range(0, len(data), 1000):
                        writer.write(data[i:i + 1000])
                        await writer.drain()
                        await asyncio.sleep(0.01)
                else:
                    writer.write(data)
                await writer.drain()
                if self.mode == 'truncate' or h.get('connection') == 'close':
                    writer.close()
                    return
        except (asyncio.IncompleteReadError, ConnectionError):
            pass


async def get(port, target, headers=None, host='127.0.0.1'):
    """One request to the cache -> (status, headers, body); raises ConnectionError on a short body."""
    reader, writer = await asyncio.open_connection(host, port)
    hs = ''.join(f'{k}: {v}\r\n' for k, v in (headers or {}).items())
    writer.write(f'GET {target} HTTP/1.1\r\nHost: x\r\n{hs}Connection: close\r\n\r\n'.encode())
    await writer.drain()
    head = await reader.readuntil(b'\r\n\r\n')
    line, *rest = head.decode('latin-1').split('\r\n')
    h = {k.lower(): v.strip() for k, _, v in (x.partition(':') for x in rest if x)}
    n = int(h.get('content-length', 0))
    try:
        body = await reader.readexactly(n)
    except asyncio.IncompleteReadError as e:
        raise ConnectionError(f'short body: {len(e.partial)} of {n}') from None
    finally:
        writer.close()
    return int(line.split()[1]), h, body


POOL = 'http://archive.ubuntu.com/ubuntu/pool/main/c/cowsay/cowsay_3.03_all.deb'
INREL = 'http://archive.ubuntu.com/ubuntu/dists/noble/InRelease'


class CacheTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.up = await FakeUpstream().start()
        self.caches = []

    async def asyncTearDown(self):
        for c in self.caches:
            await c.close()
        await self.up.close()
        self.tmp.cleanup()

    async def cache(self, **kw):
        base = f'http://127.0.0.1:{self.up.port}'
        kw.setdefault('min_free_gib', 0)
        c = PackageCache(Path(self.tmp.name) / 'store', upstream=base, pypi_pages='http://pypi.test/simple/',
                         pypi_files='http://files.test/packages/', **kw)
        await c.start()
        self.caches.append(c)
        return c

    async def test_immutable_file_is_fetched_once(self):
        self.up.put(POOL, b'deb' * 50000)
        c = await self.cache()
        a = await get(c.port, POOL)
        b = await get(c.port, POOL)
        self.assertEqual((a[0], a[2]), (200, b'deb' * 50000))
        self.assertEqual((a[1]['x-pkgcache'], b[1]['x-pkgcache']), ('miss', 'hit'))
        self.assertEqual(b[2], a[2])
        self.assertEqual(self.up.count[POOL], 1)
        tagged = await get(c.port, POOL, {'X-Pkgcache-Tag': 'build-1'})
        self.assertEqual(tagged[0], 200)
        r = await get(c.port, '/_pkgcache/flush')     # what the builder calls before it reads a build's tag file
        self.assertEqual(r[0], 200)
        log = [json.loads(x) for x in (c.root / 'tags' / 'build-1.jsonl').read_text().splitlines()]
        self.assertEqual((log[0]['cache'], log[0]['package'], log[0]['version']), ('hit', 'cowsay', '3.03'))

    async def test_mutable_index_revalidates_after_fresh(self):
        self.up.put(INREL, b'v1' * 1000)
        c = await self.cache(fresh=0.3)
        self.assertEqual((await get(c.port, INREL))[1]['x-pkgcache'], 'miss')
        self.assertEqual((await get(c.port, INREL))[1]['x-pkgcache'], 'hit')
        await asyncio.sleep(0.4)
        r = await get(c.port, INREL)
        self.assertEqual((r[1]['x-pkgcache'], r[2]), ('revalidated', b'v1' * 1000))
        self.up.put(INREL, b'v2' * 1000)
        await asyncio.sleep(0.4)
        r = await get(c.port, INREL)
        self.assertEqual((r[1]['x-pkgcache'], r[2]), ('miss', b'v2' * 1000))
        self.assertEqual(self.up.count[INREL], 3)

    async def test_stale_index_only_within_stale_max_when_upstream_down(self):
        self.up.put(INREL, b'v1' * 1000)
        c = await self.cache(fresh=0.1, stale_max=60)
        await get(c.port, INREL)
        await asyncio.sleep(0.2)
        self.up.mode = '503'
        with mock.patch('asyncio.sleep', new=fast_sleep):
            r = await get(c.port, INREL)
        self.assertEqual((r[0], r[1]['x-pkgcache'], r[2]), (200, 'stale', b'v1' * 1000))
        c.stale_max = 0
        c.breaker_until = 0
        with mock.patch('asyncio.sleep', new=fast_sleep):
            r = await get(c.port, INREL)
        self.assertEqual(r[0], 503)
        self.assertIn(b'pkgcache: upstream unreachable', r[2])

    async def test_slow_revalidation_serves_the_stored_copy(self):
        """A revalidation the proxy holds longer than revalidate_wait: the stored index now (logged stale), the
        revalidation completes for later clients; a cold miss still waits (nothing to serve)."""
        self.up.put(INREL, b'v1' * 1000)
        c = await self.cache(fresh=0.1, stale_max=60, revalidate_wait=0.3)
        await get(c.port, INREL)
        await asyncio.sleep(0.2)
        self.up.put(INREL, b'v2' * 1000)
        self.up.hold = 1.0
        t0 = time.monotonic()
        r = await get(c.port, INREL)
        self.assertLess(time.monotonic() - t0, 0.9)
        self.assertEqual((r[0], r[1]['x-pkgcache'], r[2]), (200, 'stale', b'v1' * 1000))
        await asyncio.sleep(1.2)                       # the revalidation finished meanwhile
        self.up.hold = 0
        c.fresh = 60
        r = await get(c.port, INREL)
        self.assertEqual((r[1]['x-pkgcache'], r[2]), ('hit', b'v2' * 1000))
        self.up.hold = 0.5
        t0 = time.monotonic()
        self.up.put(POOL, b'p' * 5000)
        r = await get(c.port, POOL)
        self.assertGreaterEqual(time.monotonic() - t0, 0.5)
        self.assertEqual((r[1]['x-pkgcache'], r[2]), ('miss', b'p' * 5000))
        c.flush_logs()
        log = [json.loads(x) for x in (c.root / 'logs' / 'requests.jsonl').read_text().splitlines()]
        self.assertIn('revalidation slower', next(x for x in log if x['cache'] == 'stale')['error'])

    async def test_crawling_download_resumes_on_a_new_connection(self):
        """A large file arriving far below slow_rate continues with a Range request on a new connection; the stored
        file and every client's copy are intact (incremental hashes), followers included."""
        data = os.urandom(400_000)
        self.up.put(POOL, data)
        self.up.ranges, self.up.trickle_first = True, 2_000      # 40 KB/s on the first connection
        c = await self.cache()
        c.slow_window, c.slow_rate, c.slow_min_remaining = 0.3, 1_000_000, 10_000
        with mock.patch.object(pc, 'CHUNK', 4096):     # httpx yields whole chunks (1 MiB in production)
            rs = await asyncio.wait_for(asyncio.gather(*(get(c.port, POOL) for _ in range(5))), 5)
        self.assertEqual({r[2] for r in rs}, {data})
        self.assertEqual(c.stats['resumed'], 1)
        self.assertEqual(len(self.up.ranged), 1)
        self.assertEqual(c.entries[f'apt:{POOL}'].sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual((await get(c.port, POOL))[1]['x-pkgcache'], 'hit')

    async def test_broken_download_resumes(self):
        data = os.urandom(300_000)
        self.up.put(POOL, data)
        self.up.ranges, self.up.break_first = True, True
        c = await self.cache()
        with mock.patch.object(pc, 'CHUNK', 4096):
            self.assertEqual((await get(c.port, POOL))[2], data)
        self.assertEqual(c.stats['resumed'], 1)
        (url, offset), = self.up.ranged          # from what was written (whole chunks), never past it
        self.assertTrue(url == POOL and 0 < offset <= 150_000)

    async def test_no_resume_without_range_support(self):
        data = os.urandom(100_000)
        self.up.put(POOL, data)
        self.up.trickle_first = 20_000                           # slow-ish, no Accept-Ranges
        c = await self.cache()
        c.slow_window, c.slow_rate, c.slow_min_remaining = 0.05, 10_000_000, 1_000
        with mock.patch.object(pc, 'CHUNK', 4096):
            self.assertEqual((await get(c.port, POOL))[2], data)
        self.assertEqual((c.stats['resumed'], self.up.ranged), (0, []))

    async def test_concurrent_misses_share_one_download(self):
        self.up.put(POOL, os.urandom(300_000))
        self.up.mode = 'slow'
        c = await self.cache()
        rs = await asyncio.gather(*(get(c.port, POOL) for _ in range(20)))
        self.assertEqual({r[2] for r in rs}, {self.up.files[POOL]})
        self.assertEqual(self.up.count[POOL], 1)
        self.assertEqual(sorted(r[1]['x-pkgcache'] for r in rs).count('miss'), 1)

    async def test_truncated_download_is_never_completed_or_stored(self):
        self.up.put(POOL, os.urandom(200_000))
        self.up.mode = 'truncate'
        c = await self.cache()
        with self.assertRaises(ConnectionError):
            await get(c.port, POOL)
        await asyncio.sleep(0.05)
        self.assertNotIn('apt:' + POOL, c.entries)
        self.up.mode = 'ok'
        r = await get(c.port, POOL)
        self.assertEqual(r[2], self.up.files[POOL])

    async def test_by_hash_digest_mismatch_is_never_completed(self):
        data = os.urandom(150_000)
        good = 'http://archive.ubuntu.com/ubuntu/dists/noble/main/binary-amd64/by-hash/SHA256/' + \
            hashlib.sha256(data).hexdigest()
        bad = good[:-8] + '00000000'
        self.up.put(good, data)
        self.up.put(bad, data)
        c = await self.cache()
        self.assertEqual((await get(c.port, good))[2], data)
        with self.assertRaises(ConnectionError):
            await get(c.port, bad)
        await asyncio.sleep(0.05)
        self.assertNotIn('apt:' + bad, c.entries)

    async def test_pypi_pages_are_rewritten_and_files_verified(self):
        whl = os.urandom(70_000)
        b2 = hashlib.blake2b(whl, digest_size=32).hexdigest()
        rel = f'{b2[:2]}/{b2[2:4]}/{b2[4:]}/six-1.16.0-py2.py3-none-any.whl'
        self.up.put('http://files.test/packages/' + rel, whl)
        page = json.dumps({'files': [{'url': 'http://files.test/packages/' + rel}]}).encode()
        self.up.put('http://pypi.test/simple/six/', page)
        c = await self.cache()
        r = await get(c.port, '/t/b1/pypi/simple/Six/', {'Accept': 'application/vnd.pypi.simple.v1+json'})
        self.assertEqual(json.loads(r[2])['files'][0]['url'], '../../packages/' + rel)
        r = await get(c.port, '/t/b1/pypi/packages/' + rel)
        self.assertEqual((r[0], r[2]), (200, whl))
        self.assertIn('pypi-file:' + rel, c.entries)
        bad = f'{b2[:2]}/{b2[2:4]}/{"0" * 60}/six-1.16.0-py2.py3-none-any.whl'
        self.up.put('http://files.test/packages/' + bad, whl)
        with self.assertRaises(ConnectionError):
            await get(c.port, '/pypi/packages/' + bad)
        r = await get(c.port, '/pypi/simple/six')
        self.assertEqual((r[0], r[1]['location']), (301, '/pypi/simple/six/'))

    async def test_disk_floor_serves_without_storing(self):
        self.up.put(POOL, b'x' * 100_000)
        c = await self.cache(min_free_gib=1e9)
        for _ in range(2):
            r = await get(c.port, POOL)
            self.assertEqual((r[2], r[1]['x-pkgcache']), (b'x' * 100_000, 'bypass'))
        self.assertEqual((self.up.count[POOL], len(c.entries)), (2, 0))

    async def test_write_error_mid_download_stops_storing(self):
        self.up.put(POOL, b'y' * 100_000)
        c = await self.cache()
        real = open

        def failing(path, mode='r', *a, **kw):
            if 'w' in mode and '/tmp/' in str(path):
                raise OSError(28, 'No space left on device')
            return real(path, mode, *a, **kw)
        with mock.patch('builtins.open', failing):
            r = await get(c.port, POOL)   # the disk failed, not the download: served straight through
        self.assertEqual((r[2], r[1]['x-pkgcache']), (b'y' * 100_000, 'bypass'))
        self.assertGreater(c.no_store_until, time.time())
        r = await get(c.port, POOL)
        self.assertEqual((r[2], r[1]['x-pkgcache']), (b'y' * 100_000, 'bypass'))

    async def test_breaker_fails_fast(self):
        self.up.mode = '503'
        c = await self.cache()
        with mock.patch('asyncio.sleep', new=fast_sleep):
            for i in range(3):
                self.assertEqual((await get(c.port, POOL.replace('3.03', f'{i}')))[0], 503)
        self.assertTrue(c.breaker_open())
        before = sum(self.up.count.values())
        self.assertEqual((await get(c.port, POOL))[0], 503)
        self.assertEqual(sum(self.up.count.values()), before)

    async def test_other_requests_pass_through(self):
        c = await self.cache()
        r = await get(c.port, 'http://example.com/data.csv')
        self.assertEqual(r[0], 404)   # the fake proxy's answer, relayed
        self.assertEqual(self.up.passed, ['http://example.com/data.csv'])

    async def test_eviction_and_restart(self):
        c = await self.cache(max_gib=250_000 / 2**30)
        for i in range(4):
            url = POOL.replace('3.03', f'1.{i}')
            self.up.put(url, bytes([i]) * 100_000)
            await get(c.port, url)
        self.assertLessEqual(c.total, 250_000)
        self.assertGreater(c.stats['evicted'], 0)
        kept = set(c.entries)
        await c.close()
        self.caches.remove(c)
        c2 = await self.cache()
        self.assertEqual(set(c2.entries), kept)
        url = POOL.replace('3.03', '1.3')
        self.assertEqual((await get(c2.port, url))[1]['x-pkgcache'], 'hit')

    async def test_seed_publish_and_restore(self):
        self.up.put(POOL, b'z' * 10_000)
        c = await self.cache()
        await get(c.port, POOL)
        seed = Path(self.tmp.name) / 'seed'
        summary = await asyncio.to_thread(pc.publish, c.root, seed, 1.0, self.tmp.name)
        self.assertEqual(summary['entries'], 1)
        fresh_root = Path(self.tmp.name) / 'store2'
        c2 = PackageCache(fresh_root, upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c2.start()
        self.caches.append(c2)
        self.assertEqual(await c2.start_restore(seed), 1)
        self.assertEqual((await get(c2.port, POOL))[1]['x-pkgcache'], 'hit')
        self.assertEqual(self.up.count[POOL], 1)
        self.assertTrue((fresh_root / 'seed-restored').exists())
        self.assertFalse((fresh_root / 'restore').exists())
        # a restart in the same job skips a finished restore
        await c2.close()
        self.caches.remove(c2)
        c3 = PackageCache(fresh_root, upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c3.start()
        self.caches.append(c3)
        self.assertIsNotNone(c3.start_restore(seed))
        self.assertEqual(await c3.restore_task, 0)
        self.assertEqual(len(c3.entries), 1)

    async def test_stuck_seed_read_blocks_neither_serving_nor_close(self):
        """A seed read that never returns (a sick Lustre node): the cache serves, and closes at once."""
        seed = Path(self.tmp.name) / 'seed'
        seed.mkdir()
        fifo = seed / 'seed-stuck.tar'
        os.mkfifo(fifo)                      # open() for reading blocks until a writer comes
        (seed / 'current.json').write_text(json.dumps({'file': fifo.name}))
        self.up.put(POOL, b'q' * 1000)
        c = PackageCache(Path(self.tmp.name) / 'store2', upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c.start()
        try:
            c.start_restore(seed)
            await asyncio.sleep(0.2)
            self.assertTrue(c.restoring)
            self.assertEqual((await get(c.port, POOL))[1]['x-pkgcache'], 'miss')
            t0 = time.monotonic()
            await c.close()
            self.assertLess(time.monotonic() - t0, 2)
            self.assertFalse((c.root / 'seed-restored').exists())
        finally:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))   # let the reader thread finish

    async def test_restore_runs_while_serving_and_keeps_live_copies(self):
        """Requests are served while the seed is unpacked; a key fetched before the restore reaches it keeps the
        fetched copy, and a restore interrupted by close resumes on the next start."""
        other = POOL.replace('3.03', '9.9')
        self.up.put(POOL, b'z' * 10_000)
        self.up.put(other, b'y' * 20_000)
        c = await self.cache()
        await get(c.port, POOL)
        await get(c.port, other)
        seed = Path(self.tmp.name) / 'seed'
        await asyncio.to_thread(pc.publish, c.root, seed, 1.0, self.tmp.name)
        fresh_root = Path(self.tmp.name) / 'store2'
        c2 = PackageCache(fresh_root, upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c2.start()
        self.caches.append(c2)
        status, hdr, body = await get(c2.port, POOL)     # before any restore: a miss, fetched upstream
        self.assertEqual((status, hdr['x-pkgcache'], body), (200, 'miss', b'z' * 10_000))
        live = c2.entries[next(k for k in c2.entries if k.endswith(POOL))]
        self.assertEqual(await c2.start_restore(seed), 1)   # only the other file comes from the seed
        self.assertIs(c2.entries[live.key], live)
        self.assertEqual((await get(c2.port, other))[1]['x-pkgcache'], 'hit')
        self.assertEqual(self.up.count[other], 1)
        # interrupted: no marker, staging removed on the next start, the restore resumes
        root3 = Path(self.tmp.name) / 'store3'
        c3 = PackageCache(root3, upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c3.start()
        c3.closing = True
        await c3.start_restore(seed)
        await c3.close()
        self.assertFalse((root3 / 'seed-restored').exists())
        c4 = PackageCache(root3, upstream=f'http://127.0.0.1:{self.up.port}', min_free_gib=0)
        await c4.start()
        self.caches.append(c4)
        self.assertEqual(await c4.start_restore(seed), 2)
        self.assertEqual((await get(c4.port, POOL))[1]['x-pkgcache'], 'hit')


_real_sleep = asyncio.sleep


async def fast_sleep(*_a, **_k):
    """No backoff waits in tests."""
    await _real_sleep(0)


class ClientsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.up = await FakeUpstream().start()
        self.cache = PackageCache(Path(self.tmp.name) / 'store', upstream=f'http://127.0.0.1:{self.up.port}',
                                  min_free_gib=0)
        await self.cache.start()
        self.reg = Path(self.tmp.name) / 'reg.json'
        self.reg.write_text(json.dumps({'url': f'http://127.0.0.1:{self.cache.port}', 'port': self.cache.port,
                                        'pid': os.getpid(), 'root': str(self.cache.root)}))

    async def asyncTearDown(self):
        await self.cache.close()
        await self.up.close()
        self.tmp.cleanup()

    async def test_build_env_follows_cache_health(self):
        from harbor_ns import sandbox
        with mock.patch.dict(os.environ, {'HARBOR_NS_PACKAGE_CACHE': str(self.reg)}):
            sandbox._PACKAGE_CACHE['t'] = -1e9
            env = sandbox.package_cache_env('b1')
            self.assertEqual(env['http_proxy'], f'http://b1@127.0.0.1:{self.cache.port}')
            self.assertEqual(env['PIP_INDEX_URL'], f'http://127.0.0.1:{self.cache.port}/t/b1/pypi/simple/')
            self.assertEqual(env['PIP_TRUSTED_HOST'], '127.0.0.1')   # pip fills ~/.cache/pip as without the cache
            self.assertNotIn('PIP_INDEX_URL', sandbox.package_cache_env('b1', pip=False))
            self.reg.write_text(json.dumps({'port': 1, 'pid': os.getpid(), 'root': '/nonexistent'}))
            sandbox._PACKAGE_CACHE['t'] = -1e9
            self.assertEqual(sandbox.package_cache_env('b1'), {})
            self.reg.unlink()
            sandbox._PACKAGE_CACHE['t'] = -1e9
            self.assertEqual(sandbox.package_cache_env('b1'), {})
        sandbox._PACKAGE_CACHE['t'] = -1e9

    def test_pip_configured_contexts_keep_pip_settings(self):
        from harbor_ns.builder import _pip_configured
        d = Path(self.tmp.name) / 'ctx'
        (d / 'tasksynth').mkdir(parents=True)
        (d / 'Dockerfile').write_text('FROM x\nRUN pip install six\n')
        with mock.patch.dict(os.environ, {'HARBOR_NS_PACKAGE_CACHE': str(self.reg)}):
            (d / 'tasksynth' / 'setup.sh').write_text('python3 make_data.py\n')
            self.assertFalse(_pip_configured(d))
            self.assertTrue(_pip_configured(d, budget=0))   # too big to read: pip keeps its own settings
            # command-line options beat PIP_INDEX_URL: the rest of the build's pip installs use the cache
            (d / 'tasksynth' / 'setup.sh').write_text('pip install -i https://download.pytorch.org/whl/cpu torch\n'
                                                      'pip install --index-url https://x/simple --find-links /w y\n')
            self.assertFalse(_pip_configured(d))
            (d / 'tasksynth' / 'setup.sh').write_text('cat > /etc/pip.conf <<EOF\n[global]\nfind-links = /w\nEOF\n')
            self.assertTrue(_pip_configured(d))
            (d / 'tasksynth' / 'setup.sh').write_text('pip3 config set global.index-url https://x/simple\n')
            self.assertTrue(_pip_configured(d))
            (d / 'tasksynth' / 'setup.sh').write_text('true\n')
            (d / 'files' / 'etc').mkdir(parents=True)
            (d / 'files' / 'etc' / 'pip.conf').write_text('[global]\nindex-url = https://mirror/simple\n')
            self.assertTrue(_pip_configured(d))     # a config file shipped in the context
        self.assertFalse(_pip_configured(d))   # no cache in this job: nothing to decide

    async def test_package_proxy_goes_through_cache_and_falls_back(self):
        from synth.package_proxy import PackageProxy
        self.up.put(POOL, b'd' * 50_000)
        reg = json.loads(self.reg.read_text())
        state = {'reg': reg}
        sock = Path(self.tmp.name) / 'p.sock'
        log = Path(self.tmp.name) / 'requests.jsonl'
        proxy = await PackageProxy(sock, log, f'http://127.0.0.1:{self.up.port}', cache=lambda: state['reg'],
                                   tag='trial-1').start()
        try:
            async def via_proxy():
                r, w = await asyncio.open_unix_connection(str(sock))
                w.write(f'GET {POOL} HTTP/1.1\r\nHost: archive.ubuntu.com\r\n\r\n'.encode())
                await w.drain()
                data = await r.read()
                w.close()
                return data
            first, second = await via_proxy(), await via_proxy()
            self.assertTrue(first.endswith(b'd' * 50_000) and second.endswith(b'd' * 50_000))
            self.assertEqual(self.up.count[POOL], 1)
            state['reg'] = None   # cache down: straight to the cluster proxy
            third = await via_proxy()
            self.assertTrue(third.endswith(b'd' * 50_000))
            self.assertEqual(self.up.count[POOL], 2)
            entries = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual([e.get('cache') for e in entries], ['miss', 'hit', None])
            self.cache.flush_logs()
            self.assertTrue((self.cache.root / 'tags' / 'trial-1.jsonl').exists())
        finally:
            await proxy.close()

    async def test_package_proxy_any_host_for_internet_sandboxes(self):
        """allow=None (an author's sandbox, which has internet): plain HTTP on port 80 to any host goes through the
        cache (packages cached, the rest passed through), other ports straight to the proxy, local names refused."""
        from synth.package_proxy import PackageProxy
        self.up.put(POOL, b'f' * 20_000)
        self.up.put('http://example.com/page', b'hello')
        self.up.put('http://example.com:8081/page', b'other port')
        sock = Path(self.tmp.name) / 'p4.sock'
        log = Path(self.tmp.name) / 'r4.jsonl'
        reg = json.loads(self.reg.read_text())
        open_proxy = await PackageProxy(sock, log, f'http://127.0.0.1:{self.up.port}', allow=None, cache=lambda: reg,
                                        tag='author-1').start()
        sock2 = Path(self.tmp.name) / 'p5.sock'
        registries = await PackageProxy(sock2, Path(self.tmp.name) / 'r5.jsonl', f'http://127.0.0.1:{self.up.port}',
                                        cache=lambda: reg).start()

        async def via(s, url):
            r, w = await asyncio.open_unix_connection(str(s))
            w.write(f'GET {url} HTTP/1.1\r\nHost: x\r\n\r\n'.encode())
            await w.drain()
            data = await asyncio.wait_for(r.read(), 10)
            w.close()
            return data
        try:
            for _ in range(2):
                self.assertTrue((await via(sock, POOL)).endswith(b'f' * 20_000))
                self.assertTrue((await via(sock, 'http://example.com/page')).endswith(b'hello'))
            self.assertEqual(self.up.count[POOL], 1)                       # the package: cached
            self.assertEqual(self.up.count['http://example.com/page'], 2)  # anything else: passed through
            self.assertTrue((await via(sock, 'http://example.com:8081/page')).endswith(b'other port'))
            self.assertIn(b'403', (await via(sock, 'http://db/x')).split(b'\r\n', 1)[0])
            self.assertIn(b'is local to the sandbox', await via(sock, 'http://127.0.0.1:5000/x'))
            self.assertIn(b'403', (await via(sock2, 'http://example.com/page')).split(b'\r\n', 1)[0])
            self.assertEqual(self.cache.stats['passthrough'], 2)            # ... by the cache
            entries = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual([(e['host'], e.get('cache')) for e in entries if e['allowed']],
                             [('archive.ubuntu.com', 'miss'), ('example.com', None), ('archive.ubuntu.com', 'hit'),
                              ('example.com', None), ('example.com', None)])
        finally:
            await open_proxy.close()
            await registries.close()

    async def test_streamed_miss_keeps_upstream_cache_control(self):
        """A file served while it downloads carries upstream's Cache-Control, like a stored one (pip caches by it)."""
        self.up.put(POOL, b'w' * 300_000)
        self.up.cache_control = 'max-age=365000000, immutable, public'
        _, h1, _ = await get(self.cache.port, POOL)
        _, h2, _ = await get(self.cache.port, POOL)
        self.assertEqual((h1.get('x-pkgcache'), h2.get('x-pkgcache')), ('miss', 'hit'))
        self.assertEqual(h1.get('cache-control'), 'max-age=365000000, immutable, public')
        self.assertIn('date', h1)   # with max-age, what pip's HTTP cache needs to keep a streamed file
        self.assertEqual(h2.get('cache-control'), 'max-age=365000000, immutable, public')

    async def test_package_proxy_reaches_the_cache_by_its_unix_socket(self):
        """With a socket in the registry the sandbox proxy uses it (no loopback ports: one connection per request)."""
        from synth.package_proxy import PackageProxy
        csock = Path(self.tmp.name) / 'cache.sock'
        c2 = await PackageCache(Path(self.tmp.name) / 'store2', upstream=f'http://127.0.0.1:{self.up.port}',
                                min_free_gib=0).start(sock=csock)
        self.up.put(POOL, b'e' * 10_000)
        reg = {'url': 'http://127.0.0.1:9', 'port': 9, 'pid': os.getpid(), 'root': str(c2.root), 'sock': str(csock)}
        sock = Path(self.tmp.name) / 'p3.sock'
        proxy = await PackageProxy(sock, Path(self.tmp.name) / 'r3.jsonl', f'http://127.0.0.1:{self.up.port}',
                                   cache=lambda: reg, tag='trial-3').start()
        try:
            r, w = await asyncio.open_unix_connection(str(sock))
            w.write(f'GET {POOL} HTTP/1.1\r\nHost: archive.ubuntu.com\r\n\r\n'.encode())
            await w.drain()
            data = await asyncio.wait_for(r.read(), 10)
            w.close()
            self.assertTrue(data.endswith(b'e' * 10_000))
            self.assertEqual(c2.stats['miss'], 1)    # served by the cache on the socket, not port 9
        finally:
            await proxy.close()
            await c2.close()

    async def test_package_proxy_pypi_without_cache(self):
        from synth.package_proxy import PackageProxy
        page = b'<a href="https://files.pythonhosted.org/packages/aa/bb/x/six-1.0.tar.gz">six</a>'
        self.up.put('http://pypi.org/simple/six/', page)
        sock = Path(self.tmp.name) / 'p2.sock'
        proxy = await PackageProxy(sock, Path(self.tmp.name) / 'r2.jsonl', f'http://127.0.0.1:{self.up.port}',
                                   cache=lambda: None, pip=True).start()
        try:
            with mock.patch.object(pc.PackageCache, 'http', fake_http(self.up.port)):
                r, w = await asyncio.open_unix_connection(str(sock))
                w.write(b'GET /pypi/simple/six/ HTTP/1.1\r\nHost: 127.0.0.1:3128\r\n\r\n')
                await w.drain()
                data = await asyncio.wait_for(r.read(), 10)
                w.close()
            self.assertIn(b'../../packages/aa/bb/x/six-1.0.tar.gz', data)
        finally:
            await proxy.close()


def fake_http(port):
    """PackageCache.http with https:// URLs sent to the fake upstream as plain absolute-form requests."""
    import httpx

    def http(self):
        if self.client is None:
            class Plain(httpx.AsyncHTTPTransport):
                async def handle_async_request(self, request):
                    request.url = request.url.copy_with(scheme='http')
                    return await super().handle_async_request(request)
            self.client = httpx.AsyncClient(mounts={'all://': Plain(proxy=f'http://127.0.0.1:{port}')},
                                            trust_env=False)
        return self.client
    return http


if __name__ == '__main__':
    unittest.main()


class BuildLogTest(unittest.TestCase):
    def test_cache_failures_count_as_network_failures(self):
        from harbor_ns.builder import proxy_failures
        apt = ('E: Failed to fetch http://archive.ubuntu.com/ubuntu/pool/main/c/cowsay/cowsay_3.03_all.deb  '
               '503  Service Unavailable [IP: 127.0.0.1 41297]\n')
        pip503 = ("ERROR: Could not install packages due to an OSError: HTTPConnectionPool(host='127.0.0.1', port=41297): "
                  "Max retries exceeded with url: /t/b/pypi/packages/aa/bb/x.whl (Caused by ResponseError('too many 503 "
                  "error responses'))\n")
        pip_down = ("WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after "
                    "connection broken by 'NewConnectionError(\"Failed to establish a new connection: [Errno 111] "
                    "Connection refused\")': /t/b/pypi/simple/six/\n"
                    "ERROR: Could not find a version that satisfies the requirement six (from versions: none)\n")
        for text in (apt, pip503, pip_down):
            self.assertGreater(proxy_failures(text, 8080), 0, text)
        task_error = ('E: Unable to locate package nosuchpkg\n'
                      'ERROR: Could not find a version that satisfies the requirement nosuchpkg (from versions: none)\n')
        self.assertEqual(proxy_failures(task_error, 8080), 0)


class NetworkFailuresTest(unittest.TestCase):
    """check(): a build the check's timeout stopped during a network rebuild is infrastructure, not slow setup."""

    def test_timeout_during_network_rebuild(self):
        from types import SimpleNamespace
        from task_synthesis.pipeline.check import network_failures
        with tempfile.TemporaryDirectory() as d:
            store = SimpleNamespace(shared_root=Path(d))
            (Path(d) / 'logs').mkdir()
            log = Path(d) / 'logs' / 'tasksynth-x.abc.failed.log'
            npm = ('npm ERR! request to https://registry.npmjs.org/@types%2fnode failed, reason: Proxy connection '
                   'ended before receiving CONNECT response\n')
            log.write_text(npm + '[harbor_ns] 1 downloads failed at the proxy (build failed: ...); rebuilding from '
                           'scratch in 43s (attempt 2/3)\n--> [2] RUN apt-get update\n')
            self.assertEqual(network_failures(store, 'tasksynth-x'), 0)                    # last attempt: clean
            self.assertEqual(network_failures(store, 'tasksynth-x', timed_out=True), 1)    # cut short by the timeout
            log.write_text('--> [5] RUN bash setup.sh\nsleep 9999\n')                       # a hung setup stays the task's
            self.assertEqual(network_failures(store, 'tasksynth-x', timed_out=True), 0)

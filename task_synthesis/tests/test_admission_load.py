"""Load test of the round's admission gate (#238 R7; m1-hero debrief #172 section 3C, entry 11).

m1-hero, 2026-11-04 19:37: after the v2h0 switch, 12 new round jobs found their load cache stale at once, each waiting
spec fired its own /metrics reads (~5 s each under load), most timed out and counted as load 0, and DeepSeek went from
2,000 to ~2,850 running requests per server (-27% throughput for ~30 min). PR #170: one read at a time per job
(single flight), and an unreadable load holds admission. Its tests were unit-level; this one runs the real
Round.admit / Round.pool_inflight of 12 jobs x 120 waiting specs (a burst) against two fake vLLM servers whose /metrics
answers slower the more reads it serves at once, and that stop answering for 2 s (timeouts) mid-run. The gate's clock
(cache, poll, timeout, recent window) is compressed 50x: round's module constants.

Asserts: running stays at or below the target (+10%), nothing is admitted while a server's load is unknown, and the
servers fill again once /metrics answers. The pre-#170 gate (a failed read = load 0, no single flight) fails the same
run: the sensitivity check.

  python -m unittest task_synthesis.tests.test_admission_load
"""
import asyncio
import os
import random
import tempfile
import threading
import time
import unittest
import urllib.request
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from task_synthesis.pipeline import round as R

SCALE = 50                       # production seconds per test second
CLOCK = {'POOL_CACHE_S': 15 / SCALE, 'METRICS_TIMEOUT_S': 30 / SCALE, 'ADMIT_POLL_S': 15 / SCALE,
         'UNKNOWN_POLL_S': 5 / SCALE, 'RECENT_ONESHOT_S': 30 / SCALE, 'RECENT_WORKSPACE_S': 120 / SCALE}
JOBS, WAITERS, TARGET, SERVERS = 12, 120, 60, 2
TOL = 0.10
OUTAGE = (4.0, 6.0)              # test seconds: server b's /metrics stops answering (each read outlasts the timeout)
END = 9.0


class FakeServer:
    """A vLLM stand-in: requests in flight (spec coroutines call start/end), and /metrics over HTTP whose latency grows
    with the reads it serves at once (base + per_read x concurrent reads); `down`: every read hangs past the timeout."""

    def __init__(self, base=0.01, per_read=0.02):   # 12 reads at once: 0.25 s; 384 (12 jobs x 32 threads): 7.7 s
        self.active, self.reads, self.down = 0, 0, False
        self.base, self.per_read = base, per_read
        self.lock = threading.Lock()
        srv = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                with srv.lock:
                    srv.reads += 1
                    n, run = srv.reads, srv.active
                try:
                    time.sleep(3 * CLOCK['METRICS_TIMEOUT_S'] if srv.down else srv.base + srv.per_read * n)
                    body = (f'vllm:num_requests_running{{model_name="m"}} {float(run)}\n'
                            f'vllm:num_requests_waiting{{model_name="m"}} 0.0\n').encode()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with srv.lock:
                        srv.reads -= 1

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.httpd.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/v1'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class OldGate(R.Round):
    """Round.pool_inflight before PR #170: no single flight, and a failed read counts as load 0."""

    async def pool_inflight(self):
        now = time.time()
        if now - getattr(self, '_pool_at', 0) < R.POOL_CACHE_S:
            return self._pool
        urls = list(self.gen.load)

        def read(u):
            base = u.rstrip('/').removesuffix('/v1')
            try:
                t = urllib.request.urlopen(base + '/metrics', timeout=R.METRICS_TIMEOUT_S).read().decode()
            except OSError:
                return 0
            return sum(float(l.split()[-1]) for l in t.splitlines()
                       if l.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')))
        loads = await asyncio.gather(*(asyncio.to_thread(read, u) for u in urls))
        self._pool, self._pool_at = (sum(loads), len(urls)), now
        return self._pool


def job(cls, servers, tmp):
    r = cls.__new__(cls)
    r.a = Namespace(author='oneshot', target_inflight=TARGET, target_sessions=0, build_parallel=10**6)
    r.dir, r.shards, r.open_authors = Path(tmp), JOBS, 0
    r.build_slots = asyncio.Semaphore(10**6)
    r.gen = Namespace(load={s.url: 0 for s in servers})
    r.logged = []
    r.log = r.logged.append
    return r


def simulate(cls):
    """-> (samples [(t, total running)], admission times) of JOBS jobs x WAITERS specs, server b down in OUTAGE. Each
    job runs in its own thread and event loop, with its own thread pool for the /metrics reads, as each round job is
    its own process."""
    servers = [FakeServer() for _ in range(SERVERS)]
    admitted, samples, lock = [], [], threading.Lock()
    stop = threading.Event()
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        def run_job(k):
            rng = random.Random(k)

            async def spec(r):
                await r.admit()
                with lock:
                    admitted.append(time.monotonic() - t0)
                s = min(servers, key=lambda s: s.active)       # the pool's least-loaded member
                with s.lock:
                    s.active += 1
                try:
                    await asyncio.sleep(rng.uniform(1.0, 2.0))  # one request: 50-100 s at production speed
                finally:
                    with s.lock:
                        s.active -= 1

            async def go():
                r = job(cls, servers, tmp)
                tasks = [asyncio.create_task(spec(r)) for _ in range(WAITERS)]
                while not stop.is_set():
                    await asyncio.sleep(0.05)
                for x in tasks:
                    x.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            asyncio.run(go())
        threads = [threading.Thread(target=run_job, args=(k,), daemon=True) for k in range(JOBS)]
        try:
            for th in threads:
                th.start()
            while (t := time.monotonic() - t0) < END:
                servers[1].down = OUTAGE[0] <= t < OUTAGE[1]
                samples.append((t, sum(s.active for s in servers)))
                time.sleep(0.02)
        finally:
            stop.set()
            for th in threads:
                th.join(30)
            for s in servers:
                s.close()
    return samples, admitted


class AdmissionLoad(unittest.TestCase):
    def setUp(self):
        self.orig = {k: getattr(R, k) for k in CLOCK}
        for k, v in CLOCK.items():
            setattr(R, k, v)
        self.env = {k: os.environ.get(k) for k in ('no_proxy', 'NO_PROXY')}
        for k in self.env:   # the cluster's HTTP proxy would answer for 127.0.0.1 (403)
            os.environ[k] = ','.join(x for x in (self.env[k], '127.0.0.1', 'localhost') if x)

    def tearDown(self):
        for k, v in self.orig.items():
            setattr(R, k, v)
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_gate_holds_the_target_and_recovers(self):
        samples, admitted = simulate(R.Round)
        cap = TARGET * SERVERS
        peak = max(n for _, n in samples)
        self.assertLessEqual(peak, cap * (1 + TOL), f'running peaked at {peak} > {cap} + {TOL:.0%}')
        # the burst fills the servers (~75% of the target on average: specs admitted in the last 0.6 s count as load
        # x 12 jobs until /metrics shows them, a deliberate overestimate)
        before = [n for t, n in samples if 1.0 <= t < OUTAGE[0]]
        self.assertGreaterEqual(max(before), 0.9 * cap, before[::10])
        self.assertGreaterEqual(sum(before) / len(before), 0.6 * cap, before[::10])
        # unknown load holds admission: nothing admitted once the cached load expired (cache + one read)
        hold_from = OUTAGE[0] + R.POOL_CACHE_S + R.METRICS_TIMEOUT_S
        during = [t for t in admitted if hold_from <= t < OUTAGE[1]]
        self.assertEqual(during, [], f'{len(during)} specs admitted while server b\'s load was unknown')
        low = min(n for t, n in samples if OUTAGE[0] < t < OUTAGE[1])
        self.assertLess(low, 0.3 * cap)                       # the running specs finished meanwhile
        # recovery: within ~1 s (a few cache periods) the servers are >= 70% full again, and stay under the target
        after = [n for t, n in samples if OUTAGE[1] + 1.0 <= t < END]
        self.assertGreaterEqual(max(after), 0.8 * cap, after[::10])
        self.assertGreaterEqual(sum(after) / len(after), 0.5 * cap, after[::10])
        self.assertGreater(len([t for t in admitted if t >= OUTAGE[1]]), cap)

    def test_old_gate_fails_the_same_run(self):
        """Sensitivity: the pre-#170 gate admits blind while /metrics times out, and overshoots the target."""
        samples, admitted = simulate(OldGate)
        cap = TARGET * SERVERS
        hold_from = OUTAGE[0] + R.POOL_CACHE_S + R.METRICS_TIMEOUT_S
        during = [t for t in admitted if hold_from <= t < OUTAGE[1]]
        peak = max(n for _, n in samples)
        self.assertTrue(during, 'the old gate held admission while blind')
        self.assertGreater(peak, cap * (1 + TOL))


if __name__ == '__main__':
    unittest.main()

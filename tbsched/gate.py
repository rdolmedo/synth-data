"""#477's client-side gate: hold requests, not agents, when an engine's prefix cache collapses.

In the 10-25 DeepSeek over-admission the hit rate fell from 0.96 to 0.05 and recovery took 2.5 h: admission stopped new
agents, but the running ones kept thrashing until they finished, and the KV gauge showed nothing (#6). A thrashing
engine recovers in minutes if fewer sessions send turns at once, so their contexts fit in the cache again.

`CollapseGate` is eval_q35's TurnGate (tbsched/turngate.py) with one more mode:
* open (the default): every turn goes at once; the gate only measures cache efficiency (prefix hits / the tokens the
  sessions could have reused, over turns that followed their previous one within eff_gap seconds);
* collapse: on a loaded engine, efficiency below collapse_eff over a window with enough tokens (or a trigger from the
  scheduler's server-side hit rate) sets the gate's budget to start_fill x the sessions' current context footprint.
  From then on the TurnGate admits turns for an allowed set (oldest claims first, sticky with slack), parks the
  youngest sessions' next turns (in-flight requests and agents are never touched), paces cold re-admissions and steers
  the budget by efficiency (< 0.95: x0.93; >= 0.98 with sessions waiting: +0.02);
* it opens again once every session is allowed and efficiency stayed >= eff_raise for reopen_sec.
The budget is relative to the footprint at the trigger, so it does not depend on the KV capacity vLLM reports (which is
~70x too small for some hybrid models).

`GateServer` serves one gate per engine over HTTP: POST /turn {key, engine, need} -> {go}; POST /done {key, engine,
prompt, completion, cached}; POST /end {key, engine} when a session ends; POST /sync {engine, keys: {key: order}} with
the live sessions (the scheduler, from the claims), so a session whose client died without /end is dropped too;
GET /stats. `wait_turn` is the client side: it fails open (a gate that does not answer within `timeout`, or holds a
turn longer than `max_wait`, lets the turn go), and time spent waiting is queue time, never model time.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

from tbsched.turngate import TurnGate


class CollapseGate:
    def __init__(self, collapse_eff=0.5, start_fill=0.8, lo=0.4, reopen_sec=300.0, eff_raise=0.98, raise_step=0.05,
                 **kw):
        self.g = TurnGate(capacity=None, fill=start_fill, lo=lo, eff_raise=eff_raise, raise_step=raise_step, **kw)
        self.collapse_eff, self.start_fill, self.reopen_sec = collapse_eff, start_fill, reopen_sec
        self.mode, self.clean_since, self.triggers = 'open', None, []

    def ask(self, key, need, now):
        return self.g.ask(key, need, now)

    def finished(self, key, prompt, completion, cached, now):
        self.g.finished(key, prompt, completion, cached, now)

    def sync(self, keys, now):
        self.g.sync(keys, now)

    def end(self, key):
        self.g.drop(key)

    def trigger(self, now, why='server_hit', capacity=None):
        """Start holding: budget = start_fill x the live sessions' current footprint. `capacity` (the engine's KV tiers in
        tokens, if known) only sets the unit of the budget's later steps (+raise_step x capacity per adjustment), so the
        budget keeps up with contexts that grow; without it the unit is the footprint at the trigger."""
        if self.mode == 'hold':
            return
        foot = sum(self.g._fp(k) for k in self.g.r)
        if foot <= 0:
            return
        g = self.g
        cap = float(capacity) if capacity and capacity > foot / 4 else float(foot)
        fill = self.start_fill * foot / cap
        g.capacity, g.fill, g.fill0, g.hi = cap, fill, fill, float('inf')
        g.lo = min(g.lo, fill)
        g.allowed = set()            # rebuilt from the oldest claims within the new budget
        g.last_cut, g.last_adjust = now, now
        self.mode, self.clean_since = 'hold', None
        self.triggers.append((round(now), why))
        g._rebalance(now)

    def tick(self, now, loaded=True):
        g = self.g
        g.tick(now)
        if self.mode == 'open':
            if loaded and g.eff is not None and g.eff < self.collapse_eff:
                self.trigger(now, 'gate_efficiency')
            return
        everyone = len(g.allowed) == len(g.r)
        if everyone and (g.eff is None or g.eff >= g.eff_raise):   # no traffic left to measure counts as clean
            self.clean_since = self.clean_since if self.clean_since is not None else now
        elif not everyone or (g.eff is not None and g.eff < g.eff_cut):
            self.clean_since = None
        if self.clean_since is not None and now - self.clean_since >= self.reopen_sec:
            g.capacity, self.mode, self.clean_since = None, 'open', None
            g._rebalance(now)

    def stats(self):
        return {**self.g.stats(), 'mode': self.mode, 'triggers': self.triggers[-5:]}


class GateServer:
    """One gate per engine. Engines are added with add(name); requests name their engine (default: the only one)."""

    def __init__(self, host='0.0.0.0', port=0, tick=5.0, make=CollapseGate):
        self.gates, self.make, self.lock = {}, make, threading.Lock()
        gs = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                try:
                    d = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                    if self.path == '/sync':
                        with gs.lock:
                            g = gs.of(d.get('engine'))
                            if g is not None:
                                g.sync({str(k): float(v) for k, v in (d.get('keys') or {}).items()}, time.time())
                        return self.reply(200, {})
                    key = str(d['key'])
                except (ValueError, KeyError, TypeError):
                    return self.reply(400, {})
                now = time.time()
                with gs.lock:
                    g = gs.of(d.get('engine'))
                    if g is None:
                        return self.reply(200, {'go': True, 'gate': 'none'})
                    if self.path == '/turn':
                        return self.reply(200, {'go': g.ask(key, int(d.get('need') or 0), now)})
                    if self.path == '/end':
                        g.end(key)
                        return self.reply(200, {})
                    if self.path == '/done':
                        cached = d.get('cached')
                        g.finished(key, int(d.get('prompt') or 0), int(d.get('completion') or 0),
                                   None if cached is None else int(cached), now)
                        return self.reply(200, {})
                self.reply(404, {})

            def do_GET(self):
                if self.path != '/stats':
                    return self.reply(404, {})
                with gs.lock:
                    self.reply(200, {e: g.stats() for e, g in gs.gates.items()})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.url = f'http://{host if host != "0.0.0.0" else "127.0.0.1"}:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.tick_sec, self.loaded = tick, {}

        def clock():
            while True:
                time.sleep(self.tick_sec)
                with self.lock:
                    for e, g in self.gates.items():
                        g.tick(time.time(), self.loaded.get(e, True))
        threading.Thread(target=clock, daemon=True).start()

    def add(self, engine):
        with self.lock:
            return self.gates.setdefault(engine, self.make())

    def of(self, engine):
        if engine is None and len(self.gates) == 1:
            return next(iter(self.gates.values()))
        return self.gates.get(engine)

    def close(self):
        self.httpd.shutdown()


def _post(url, path, body, timeout):
    req = urllib.request.Request(url.rstrip('/') + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read() or b'{}')


def wait_turn(url, key, engine=None, need=0, timeout=5.0, poll=1.0, max_wait=1800.0, sleep=time.sleep) -> float:
    """Block until the gate lets this turn go; returns the seconds waited. Fails open on any gate error and after
    max_wait (a gate bug must never hold a session forever)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait:
        try:
            if _post(url, '/turn', {'key': key, 'engine': engine, 'need': need}, timeout).get('go', True):
                break
        except Exception:   # noqa: BLE001 -- a gate that is down must never stop a session
            break
        sleep(poll)
    return time.monotonic() - t0


def report_end(url, key, engine=None, timeout=5.0) -> None:
    try:
        _post(url, '/end', {'key': key, 'engine': engine}, timeout)
    except Exception:   # noqa: BLE001
        pass


def report_done(url, key, engine=None, prompt=0, completion=0, cached=None, timeout=5.0) -> None:
    try:
        _post(url, '/done', {'key': key, 'engine': engine, 'prompt': prompt, 'completion': completion,
                             'cached': cached}, timeout)
    except Exception:   # noqa: BLE001
        pass

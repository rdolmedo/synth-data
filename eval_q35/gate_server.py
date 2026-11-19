"""HTTP front of the turn gate (turn_gate.py) inside the serving job (admission.py starts it for control "ctx_gate").

Each rollout's proxy (sticky_proxy --gate-key) asks it before every turn and reports every finished turn, so the gate
decides on current contexts in memory:
  POST /turn {key, need}                        -> {"go": bool}   (need: the proxy's estimate of the prompt tokens)
  POST /done {key, prompt, completion, cached}  -> {}             (cached: vLLM's prompt_tokens_details.cached_tokens)
  GET  /stats                                   -> the gate's state
The v2 gate instead scanned the proxies' response logs on Lustre from the serving node: an update took up to ~40 s while
the proxies were writing them, and a decision gone stale (vLLM busy, the loop blocked) released every held rollout at
once. Here the gate answers in milliseconds and keeps its own clock (fill adjustment, pacing) in a thread.
"""
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common import dp_rank
from turn_gate import TurnGate


def epoch(iso, default):
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return default


class GateServer:
    def __init__(self, host, capacity, fill=0.7, block=1, tick=5.0, ranks=1, learn_hi=True):
        # vLLM data-parallel engines have separate caches: one gate per engine with its share of the summed capacity,
        # and a rollout competes only within the engine it is pinned to (common.dp_rank, as worker.py pins it)
        self.gates = [TurnGate(capacity=capacity / ranks, fill=fill, block=block, learn_hi=learn_hi)
                      for _ in range(ranks)]
        self.lock = threading.Lock()
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
                    key = str(d['key'])
                except (ValueError, KeyError):
                    return self.reply(400, {})
                now = time.time()
                if self.path == '/turn':
                    with gs.lock:
                        go = gs.of(key).ask(key, int(d.get('need') or 0), now)
                    return self.reply(200, {'go': go})
                if self.path == '/done':
                    cached = d.get('cached')
                    with gs.lock:
                        gs.of(key).finished(key, int(d.get('prompt') or 0), int(d.get('completion') or 0),
                                         None if cached is None else int(cached), now)
                    return self.reply(200, {})
                self.reply(404, {})

            def do_GET(self):
                if self.path != '/stats':
                    return self.reply(404, {})
                with gs.lock:
                    self.reply(200, gs.stats())

        self.httpd = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f'http://{host}:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

        def clock():
            while True:
                time.sleep(tick)
                with self.lock:
                    for g in self.gates:
                        g.tick(time.time())
        threading.Thread(target=clock, daemon=True).start()

    def of(self, key):
        return self.gates[dp_rank(key, len(self.gates))] if len(self.gates) > 1 else self.gates[0]

    def stats(self):
        """One gate: its stats. Per-engine gates: counts summed, fill, ceiling and efficiency the lowest engine's (None
        where no engine has one), plus each engine's own stats under "ranks"."""
        if len(self.gates) == 1:
            return self.gates[0].stats()
        per = [g.stats() for g in self.gates]
        low = ('ctx_fill', 'ctx_hi', 'ctx_eff')
        out = {k: sum(s[k] for s in per) for k in per[0] if k not in low}
        for k in low:
            vals = [s[k] for s in per if s.get(k) is not None]
            out[k] = min(vals) if vals else None
        return {**out, 'ranks': per}

    def sync(self, claims, now):
        """claims: {key: claimed_at (ISO)} of the live rollouts. Returns the gate's state."""
        with self.lock:
            for g in self.gates:
                g.sync({k: epoch(v, now) for k, v in claims.items() if self.of(k) is g}, now)
            return self.stats()

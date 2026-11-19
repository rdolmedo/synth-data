"""eval_q35's context-aware turn gate (eval_q35/turn_gate.py v11, gate_server.py) for a pi_eval run.

In eval_q35 the gate lives in the run's serving job (admission.py, control "ctx_gate"): it learns the cache size from
vLLM, keeps the run's live rollouts in sync with the claim queue, and publishes its address and vLLM's KV reading in
RUN/serve/kv.json, which every rollout's proxy reads (sticky_proxy --kv-gate-file, --gate-key). A pi_eval run has no
serving job of its own (the model is served by serve_teacher), so one of the run's workers does the same:

* election: the worker holding RUN/serve/turn_gate.lock (mkdir mutex; Lustre has no file locks) runs the gate. It
  refreshes RUN/serve/turn_gate.json every TICK seconds; a holder silent for STALE_SEC loses the lock to the next worker
  (a new gate with fresh state: it relearns the contexts from the proxies' next turns, as after an eval_q35 restart).
* one gate per replica (pi_eval pins each attempt to one replica, worker.url_for): RUN/serve/kv-<i>.json for replica i,
  each with that engine's capacity (/metrics cache_config_info kv_cache_size_tokens, else the registry's vllm.log line
  "GPU KV cache size") and KV usage, plus "gate_url". Attempts read their replica's file (PI_KV_GATE_FILE).
* sync: every TICK, the run's active claims (queue_state, claim times as order) go to their replica's gate; finished
  or lost rollouts drop out (gate_server.sync).
Policy, fill (ctx_fill, default 0.7) and the proxies' KV brake (kv_gate 0.80,0.72) as eval_q35's defaults.
"""
from __future__ import annotations

import os
import shutil
import socket
import threading
import time
from pathlib import Path

TICK = 5.0
SYNC_EVERY = 15.0
STALE_SEC = 180.0


class TurnGateHost:
    def __init__(self, worker, q35):
        self.w, self.q35 = worker, q35
        self.serve = worker.run / 'serve'
        self.lock = self.serve / 'turn_gate.lock'
        self.gates: dict[str, object] = {}   # replica url -> GateServer
        self.thread = None

    # ------------------------------------------------------------- election
    def try_hold(self) -> bool:
        if self.thread and self.thread.is_alive():
            return True
        try:
            self.lock.mkdir()
        except FileExistsError:
            info = self.q35.read(self.serve / 'turn_gate.json', {}) or {}
            try:
                age = time.time() - max(float(info.get('t') or 0), self.lock.stat().st_mtime)
            except OSError:
                return False
            if age < STALE_SEC:
                return False
            # steal a dead holder's lock: rename is atomic, so one worker wins
            try:
                gone = self.serve / f'turn_gate.lock.stale.{os.getpid()}.{int(time.time())}'
                self.lock.rename(gone)
                shutil.rmtree(gone, ignore_errors=True)
                self.lock.mkdir()
            except OSError:
                return False
            self.q35.log(f'turn gate: took over from {info.get("worker")} (silent {age:.0f}s)')
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return True

    # ------------------------------------------------------------- the gate
    def capacity(self, url: str, metrics: dict) -> float | None:
        from admission import log_kv_tokens   # eval_q35 (on sys.path via the q35 worker module)
        cap = metrics.get('kv_cache_size_tokens')
        if cap:
            return cap
        reg = self.w.config.get('registry_dir')
        if reg and len(self.w.urls) == 1:
            return log_kv_tokens(Path(reg) / 'vllm.log')
        if reg:
            i = self.w.urls.index(url)
            return log_kv_tokens(Path(reg) / ('vllm.log' if i == 0 else f'vllm-r{i}.log'))
        return None

    def loop(self):
        from admission import scrape
        from gate_server import GateServer
        c = self.w.config
        host = socket.getfqdn()
        last_sync, last_kv = 0.0, {}
        while not self.q35.STOP:
            try:
                now = time.time()
                urls = list(self.w.urls)
                pubs = {}
                for i, url in enumerate(urls):
                    base = url.rstrip('/').removesuffix('/v1')
                    try:
                        m = scrape(base)
                    except OSError:
                        m = {}
                    if url not in self.gates:
                        cap = self.capacity(url, m)
                        if not cap:
                            continue
                        self.gates[url] = GateServer(host, cap, fill=float(c.get('ctx_fill', 0.7)),
                                                     block=int(m.get('block_size', 1)), ranks=1,
                                                     learn_hi=bool(c.get('ctx_learn_hi', True)))
                        self.q35.log(f'turn gate for {url} at {self.gates[url].url} ({cap:.0f} cache tokens, '
                                     f'block {m.get("block_size", 1)})')
                    if m:
                        last_kv[url] = (now, m.get('vllm:kv_cache_usage_perc', 0.0))
                    pubs[url] = i
                if now - last_sync >= SYNC_EVERY:
                    last_sync = now
                    st = self.w.state
                    claims = {k: st.claimed_at(k) for k in st.active()}
                    by = {u: {} for u in self.gates}
                    for k, at in claims.items():
                        u = self.w.url_for(k)
                        if u in by:
                            by[u][k] = at
                    self.stats = {u: g.sync(by[u], now) for u, g in self.gates.items()}
                for url, i in pubs.items():
                    g = self.gates.get(url)
                    t_kv, kv = last_kv.get(url, (0.0, 0.0))
                    self.q35.atomic(self.serve / f'kv-{i}.json', {
                        't': t_kv or now, 'kv': kv, 'real_kv': kv, 'engine': url,
                        **({'gate_url': g.url, **(getattr(self, 'stats', {}).get(url) or {})} if g else {})})
                self.q35.atomic(self.serve / 'turn_gate.json', {
                    't': now, 'worker': self.w.name, 'host': host, 'pid': os.getpid(),
                    'gates': {u: g.url for u, g in self.gates.items()}})
            except Exception as e:  # noqa: BLE001 -- a bookkeeping error must not stop the gate (proxies keep asking)
                self.q35.log(f'turn gate loop: {e!r}')
            time.sleep(TICK)
        shutil.rmtree(self.lock, ignore_errors=True)

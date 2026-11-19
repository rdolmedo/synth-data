"""Replay a gate's request log against its engine's metrics through gate.py's admission code (#378).

  python -m aa_evals.qwenserve.gate_replay --registry DIR --gate 0 --metrics vllm_metrics.csv.gz --cluster JOB \
      [--replica 0] [--old-gate OLD_gate.py] [--out replay.jsonl]

Open loop: the requests arrive, are admitted and end exactly as the gate log (`--log` of gate.py) recorded them; every
`--tick` seconds the gate's tick (Gate.observe) runs on the engine state of that time, taken from vLLM's 10-s stat
lines (the audit's vllm_metrics.csv.gz: kv_pct, running, waiting, gen and prompt tok/s, prefix hit %). Per tick it
records what the gate would decide: room for new conversations, causes, brakes, under-use, live and idle tokens. With
--old-gate, the same replay runs through another gate.py (e.g. `git show origin/aa-serving:aa_evals/qwenserve/gate.py`)
and both decisions are written side by side.

Approximations: a first turn's key in the log is its first-two-messages prefix (shared by repeats), so each logged
`new` multi-turn request starts its own conversation, and a later turn with an unknown key claims the client's oldest
finished first turn. A multi-turn request logged with status 200 and no usage was cut mid-stream by its client (c30:
they cluster at the worker kills); the gate before #378 recorded it as a 200 (context = request bytes / 3), the
current gate as `client_gone` (handler cancelled), so each version is fed what it would have seen. The current gate's
text estimate of a prompt (prompt_estimate) is replaced by request bytes x the client's prompt tokens per byte, learned
from its requests with usage (so the estimator itself is not tested here). Only clients whose multi-turn 200s carry
usage (>= 95%) have their usage-less 200s treated as cut. max_model_len: launch.json's --max-model-len. No preemption counter is in the stat lines (preemptions: 0).
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

CUT = 'client_gone'


def load_gate(path: str | None):
    if not path:
        from aa_evals.qwenserve import gate
        return gate
    spec = importlib.util.spec_from_file_location(f'gate_{abs(hash(path))}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_metrics(path: str, registry: str, cluster: str, replica: int) -> list[dict]:
    rows = []
    with gzip.open(path, 'rt') as f:
        for r in csv.DictReader(f):
            if r['registry'] == registry and r['cluster'] == cluster and int(r['replica']) == replica:
                rows.append(r)
    rows.sort(key=lambda r: float(r['t']))
    out, gen, comp, hit = [], 0.0, 0.0, 0.0
    prev = None
    for r in rows:
        t = float(r['t'])
        dt = t - prev if prev is not None else 10.0
        prev = t
        num = lambda k: float(r[k]) if r.get(k) not in (None, '') else 0.0
        gen += num('gen_tps') * dt
        prompt = num('prompt_tps') * dt
        frac = num('prefix_pct') / 100
        # prompt tok/s is computed (not cached) prefill; cached tokens = computed x hit / (1 - hit)
        cached = prompt * frac / max(1e-6, 1 - frac) if frac < 1 else 0.0
        comp += prompt
        hit += cached
        out.append({'t': t, 'kv': num('kv_pct') / 100, 'wait': num('waiting'), 'run': num('running'), 'gen': gen,
                    'preempt': 0.0, 'src_compute': comp, 'src_gpu': hit, 'src_cpu': 0.0})
    return out


def read_log(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


class Replay:
    def __init__(self, mod, rows, snaps, capacity, block, offload, tick, idle_ttl, max_len, new_view):
        self.mod, self.rows, self.snaps, self.tick = mod, rows, snaps, tick
        self.clock = [0.0]
        mod.time = types.SimpleNamespace(time=lambda: self.clock[0])
        self.gate = mod.Gate('http://replay', capacity=capacity, block=block, offload=offload, idle_ttl=idle_ttl,
                             tick=tick)
        e = self.gate.engines[0]
        if hasattr(e, 'max_len'):
            e.max_len = max_len
        self.new_view = new_view   # feed cut streams as client_gone (the current gate's view)
        self.seq = 0
        self.firsts = collections.defaultdict(collections.deque)   # client -> finished first turns' conv keys
        self.ratio = self.bytes_ratio()
        self.usage_clients = self.reports_usage()

    def reports_usage(self):
        """Clients whose multi-turn responses carry usage (>= 95% of their 200s): only for them is a 200 without
        usage a stream their client cut."""
        n, u = collections.Counter(), collections.Counter()
        for r in self.rows:
            if r.get('multi') and r.get('status') == 200:
                n[r['client']] += 1
                u[r['client']] += bool(r.get('total_tokens'))
        return {c for c in n if u[c] >= 0.95 * n[c]}

    def bytes_ratio(self):
        """Per client: prompt tokens per request byte (requests with usage)."""
        tok, byt = collections.Counter(), collections.Counter()
        for r in self.rows:
            if r.get('total_tokens') and r.get('completion_tokens') is not None and r.get('req_bytes'):
                tok[r['client']] += r['total_tokens'] - r['completion_tokens']
                byt[r['client']] += r['req_bytes']
        return {c: tok[c] / byt[c] for c in byt if byt[c]}

    def admit(self, row):
        g, mod = self.gate, self.mod
        e = g.engines[0]
        client = g.clients.get(row['client']) or g.clients.setdefault(row['client'],
                                                                     mod.Client(g.policy['output_tokens_prior']))
        multi = bool(row.get('multi'))
        key = row['key']
        if multi and row.get('class') in ('new', 'short') and key not in g.convs:
            key = f"{key}#{self.seq}"   # a first turn: its own conversation
            self.seq += 1
            first = True
        else:
            first = False
            if multi and key not in g.convs:
                q = self.firsts[row['client']]
                while q:
                    old = q.popleft()
                    c = g.convs.get(old)
                    if c is not None and not c['inflight']:
                        g.convs[key] = g.convs.pop(old)
                        break
        est = int((row.get('req_bytes') or 0) * self.ratio.get(row['client'], 1 / 3.0))
        r = mod.Request(None, key, multi, row['client'], 0 if multi else est, None, t=row['t_enq'])
        if hasattr(r, 'est'):
            r.est = est
        r.engine, r.cls, r.t_admit = e, row.get('class') or 'new', row['t_admit']
        if r.cls == 'short':
            e.short_inflight += 1
        e.inflight += 1
        e.running.add(r)
        client.inflight += 1
        client.running.add(r)
        client.last_admit = row['t_admit']
        conv = g.convs.setdefault(key, {'tokens': 0, 'last': row['t_admit'], 'inflight': 0, 'multi': multi})
        conv['inflight'] += 1
        conv['engine'] = 0
        conv.pop('gone', None)   # as Gate.dispatch: a turn of a dropped conversation revives it
        return r, client, first

    def end(self, r, client, first, row):
        usage = None
        if row.get('total_tokens'):
            usage = {'total_tokens': row['total_tokens'], 'completion_tokens': row.get('completion_tokens')}
        status = row.get('status')
        if status == 200 and usage is None and r.multi and self.new_view and row['client'] in self.usage_clients:
            status = CUT
        r.stats.update(status=status, usage=usage, events=int(row.get('completion_tokens') or 0))
        self.gate.release(r, client, r.cls, int(row.get('req_bytes') or 0))
        if first and r.multi and r.key in self.gate.convs and status == 200:
            self.firsts[row['client']].append(r.key)

    def run(self):
        events = []
        for i, row in enumerate(self.rows):
            if row.get('t_admit') is None or row.get('t_end') is None:
                continue
            events.append((row['t_admit'], 0, i))
            events.append((row['t_end'], 1, i))
        events.sort()
        t0 = max(self.snaps[0]['t'], events[0][0]) if events else self.snaps[0]['t']
        t1 = self.snaps[-1]['t']
        live, out, j, si = {}, [], 0, 0
        t = t0
        while t <= t1:
            while j < len(events) and events[j][0] <= t:
                tt, kind, i = events[j]
                self.clock[0] = tt
                if kind == 0:
                    live[i] = self.admit(self.rows[i])
                elif i in live:
                    self.end(*live.pop(i), self.rows[i])
                j += 1
            while si + 1 < len(self.snaps) and self.snaps[si + 1]['t'] <= t:
                si += 1
            snap = dict(self.snaps[si])
            if t - snap['t'] > 60:
                snap = None   # no stat line for a minute: stale, as the gate would see a failed scrape
            self.clock[0] = t
            self.gate.observe([snap], t)
            e = self.gate.engines[0]
            st = e.status
            out.append({'t': round(t, 1), 'room': e.room, 'causes': list(e.causes), 'brakes': list(e.brakes),
                        'underused': st.get('underused'), 'live': st.get('live_conversations'),
                        'idle_tokens': st.get('idle_tokens'), 'mean_tokens': st.get('mean_tokens'),
                        'target': st.get('target'), 'convs': len(self.gate.convs), 'gpu_free': int(e.gpu_free),
                        'kv': None if snap is None else snap['kv'], 'wait': None if snap is None else snap['wait'],
                        'run': None if snap is None else snap['run']})
            t += self.tick
        return out


def queued_new_at(rows, ts):
    """New (non-short) requests waiting at the real gate (> 1 s) at each time."""
    import bisect
    enq = sorted(r['t_enq'] for r in rows if r.get('class') == 'new' and (r.get('queue_ms') or 0) > 1000)
    adm = sorted(r['t_admit'] for r in rows if r.get('class') == 'new' and (r.get('queue_ms') or 0) > 1000)
    return [bisect.bisect_right(enq, t) - bisect.bisect_right(adm, t) for t in ts]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', required=True, help='serving registry dir (gate-r<i>.jsonl, gate-r<i>.status.json)')
    ap.add_argument('--gate', type=int, default=0, help='gate index i (gate-r<i>.jsonl)')
    ap.add_argument('--metrics', required=True, help='vllm_metrics.csv.gz (registry, cluster, replica, t, kv_pct, ...)')
    ap.add_argument('--cluster', required=True, help="the server job's condor cluster id in the metrics file")
    ap.add_argument('--replica', type=int, default=None, help='vLLM replica in the metrics file (default: --gate)')
    ap.add_argument('--capacity', type=int, default=None, help='KV tokens (default: gate-r<i>.status.json)')
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=None, help="max_model_len (default: launch.json's --max-model-len)")
    ap.add_argument('--tick', type=float, default=5.0)
    ap.add_argument('--idle-ttl', type=float, default=300.0)
    ap.add_argument('--old-gate', default=None, help='another gate.py to replay side by side')
    ap.add_argument('--out', required=True, help='per-tick JSON lines')
    a = ap.parse_args(argv)
    reg = Path(a.registry)
    status = json.loads((reg / f'gate-r{a.gate}.status.json').read_text())
    capacity = a.capacity or status.get('capacity') or (status.get('engines') or [{}])[0].get('capacity')
    offload = status.get('offload') or 0
    max_len = a.max_len
    if max_len is None:
        try:
            argv = json.loads((reg / 'launch.json').read_text())['replicas'][0]['argv']
            max_len = int(argv[argv.index('--max-model-len') + 1])
        except (OSError, ValueError, KeyError, IndexError):
            max_len = 0
    rows = read_log(reg / f'gate-r{a.gate}.jsonl')
    snaps = read_metrics(a.metrics, reg.name, a.cluster, a.gate if a.replica is None else a.replica)
    if not snaps:
        sys.exit('no metrics rows for this registry / cluster / replica')
    lo, hi = snaps[0]['t'] - 600, snaps[-1]['t'] + 600
    rows = [r for r in rows if r.get('t_enq') and lo <= r['t_enq'] <= hi]
    versions = {'new': (load_gate(None), True)}
    if a.old_gate:
        versions['old'] = (load_gate(a.old_gate), False)
    res = {}
    for name, (mod, new_view) in versions.items():
        res[name] = Replay(mod, rows, snaps, capacity, a.block, offload, a.tick, a.idle_ttl, max_len,
                           new_view).run()
    ts = [x['t'] for x in res['new']]
    q = queued_new_at(rows, ts)
    with open(a.out, 'w') as f:
        for i, t in enumerate(ts):
            f.write(json.dumps({'t': t, 'queued_new': q[i], **{k: v[i] for k, v in res.items()}}) + '\n')
    print(f'{reg.name} gate {a.gate}: {len(ts)} ticks, {len(rows)} requests -> {a.out}')


if __name__ == '__main__':
    main()

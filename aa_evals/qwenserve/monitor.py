"""Production monitor for the Qwen3.8 eval servers behind the admission gate (#110).

  python -m aa_evals.qwenserve.monitor                       # one table, deltas over a 60 s window
  python -m aa_evals.qwenserve.monitor --loop --every 60 --out /data/aai/aa_evals/runs/qwenserve/monitor.jsonl

Reads every `endpoints.json` matching --registry (default: the aa-qwen* serving registries), scrapes each public URL's
/metrics (vLLM's metrics plus the gate's) and /gate/status, and reports per server: KV usage (per engine for a pool),
requests running / waiting in vLLM, gen tok/s, prefix hit rate, preemptions, gate queue p50/p95 by class (from the
gate:queue_seconds histogram over the window), 503s, and per client: queued (new / live turns), in flight, its KV and
share (gate v2.1), expected output, completions, seconds since its last admission.

Alerts (the `alerts` field; the rollout's success criteria): kv_high (any engine's KV > 0.92), preemptions (any in the
window), starved:<client> (a client with queued requests and no admission for --starve seconds), vllm_waiting
(requests waiting inside vLLM: the gate let too much through), no_gate (a server without a gate), unreachable.
--loop appends one JSON line per server per sample to --out (one file, never a file per sample).
"""
import argparse
import glob
import json
import re
import sys
import time
import urllib.request

from aa_evals.common import serving

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)')
LABEL = re.compile(r'(\w+)="([^"]*)"')
REGISTRY = str(serving.registry_root() / 'aa-qwen*' / 'endpoints.json')   # $TASKSYNTH_SERVING_ROOT, else production


def parse(text):
    """Prometheus text -> [(name, labels, value)]."""
    out = []
    for line in text.splitlines():
        if not line or line[0] == '#':
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        try:
            v = float(m[3])
        except ValueError:
            continue
        out.append((m[1], dict(LABEL.findall(m[2] or '')), v))
    return out


def total(samples, name, **match):
    return sum(v for n, lab, v in samples if n == name and all(lab.get(k) == x for k, x in match.items()))


def per_engine(samples, name):
    """One value per engine behind a gate (a pooled gate labels each engine's samples replica="i")."""
    vals = {}
    for n, lab, v in samples:
        if n == name:
            k = lab.get('replica', '0')
            vals[k] = vals.get(k, 0) + v
    return [vals[k] for k in sorted(vals, key=int)]


def hist(samples, cls):
    """Cumulative bucket counts {le: count} of gate:queue_seconds for a class."""
    return {float(lab['le']): v for n, lab, v in samples
            if n == 'gate:queue_seconds_bucket' and lab.get('class') == cls}


def quantile(h0, h1, q):
    """Quantile of the window's queue times from two cumulative histograms (linear inside a bucket)."""
    les = sorted(h1)
    d = [h1[le] - h0.get(le, 0) for le in les]
    n = d[-1] if d else 0
    if n <= 0:
        return None
    target, prev_le, prev_c = q * n, 0.0, 0.0
    for le, c in zip(les, d):
        if c >= target:
            if le == float('inf'):
                return prev_le
            return round(prev_le + (le - prev_le) * (target - prev_c) / max(c - prev_c, 1e-9), 2)
        prev_le, prev_c = le, c
    return prev_le


def fetch(url, timeout=15):
    return OPENER.open(url, timeout=timeout).read().decode()


def snapshot(reg):
    ep = json.loads(open(reg).read())
    name = reg.split('/')[-2]
    out = {'server': name, 'job': ep.get('job_id'), 'gate': bool(ep.get('gate')), 'servers': []}
    for url in ep.get('urls', []):
        base = url.removesuffix('/v1').rstrip('/')
        s = {'url': base, 't': time.time()}
        try:
            s['metrics'] = parse(fetch(base + '/metrics'))
        except OSError as exc:
            s['error'] = repr(exc)[:200]
        if ep.get('gate'):
            try:
                s['status'] = json.loads(fetch(base + '/gate/status'))
            except (OSError, ValueError):
                s['status'] = None
        out['servers'].append(s)
    return out


def summarize(a, b, starve):
    """One row per public URL from two snapshots of a registry."""
    rows = []
    for s0, s1 in zip(a['servers'], b['servers']):
        row = {'server': b['server'], 'url': s1['url'], 'job': b['job'], 't': round(s1['t']), 'alerts': []}
        if 'error' in s1 or 'error' in s0:
            row['alerts'].append('unreachable')
            rows.append(row)
            continue
        m0, m1 = s0['metrics'], s1['metrics']
        dt = max(s1['t'] - s0['t'], 1e-3)
        kv = per_engine(m1, 'vllm:kv_cache_usage_perc')
        d = lambda name, **k: total(m1, name, **k) - total(m0, name, **k)
        hit_c = d('vllm:prompt_tokens_by_source_total', source='local_cache_hit') + \
            d('vllm:prompt_tokens_by_source_total', source='external_kv_transfer')
        hit_all = hit_c + d('vllm:prompt_tokens_by_source_total', source='local_compute')
        row.update(kv=[round(x, 3) for x in kv], running=total(m1, 'vllm:num_requests_running'),
                   waiting=total(m1, 'vllm:num_requests_waiting'),
                   gen_tok_s=round(d('vllm:generation_tokens_total') / dt),
                   prefix_hit=round(hit_c / hit_all, 3) if hit_all else None,
                   preemptions=d('vllm:num_preemptions_total'), window_s=round(dt))
        if not b['gate']:
            row['alerts'].append('no_gate')
        if any(x > 0.92 for x in kv):
            row['alerts'].append('kv_high')
        if row['preemptions'] > 0:
            row['alerts'].append('preemptions')
        if row['waiting'] > 0:
            row['alerts'].append('vllm_waiting')
        if total(m1, 'gate:engines') or total(m1, 'gate:inflight') or any(n.startswith('gate:') for n, _, _ in m1):
            row['queue'] = {}
            for cls in ('cont', 'new', 'short'):
                h0, h1 = hist(m0, cls), hist(m1, cls)
                if h1:
                    row['queue'][cls] = {'n': int(h1.get(float('inf'), 0) - h0.get(float('inf'), 0)),
                                         'p50_s': quantile(h0, h1, 0.5), 'p95_s': quantile(h0, h1, 0.95)}
            row['rejected'] = d('gate:requests_total', outcome='rejected')
            row['client_gone'] = d('gate:requests_total', outcome='client_gone')
            row['inflight'] = total(m1, 'gate:inflight')
        st = s1.get('status') or {}
        clients = {}
        for name, c in (st.get('clients') or {}).items():
            clients[name] = {k: c.get(k) for k in ('queued_new', 'queued_cont', 'inflight', 'kv_tokens', 'share_tokens',
                                                   'held', 'expected_out', 'completed', 'last_admit_age_s')
                             if c.get(k) is not None}
            queued = (c.get('queued_new') or 0) + (c.get('queued_cont') or 0)
            if queued and (c.get('last_admit_age_s') or 0) > starve:
                row['alerts'].append(f'starved:{name}')
        row['clients'] = clients
        if st.get('engines'):
            row['engines'] = [{k: e.get(k) for k in ('kv', 'running', 'waiting', 'target', 'room', 'gpu_free_tokens',
                                                     'reserved_tokens', 'brakes', 'causes')} for e in st['engines']]
        rows.append(row)
    return rows


def table(rows):
    lines = []
    for r in rows:
        if 'kv' not in r:
            lines.append(f"{r['server']:<22} {' '.join(r['alerts'])}")
            continue
        q = r.get('queue') or {}
        qs = ' '.join(f"{c}:{v['p50_s']}/{v['p95_s']}s(n={v['n']})" for c, v in q.items() if v['n'])
        lines.append(f"{r['server']:<22} kv {','.join(f'{x:.2f}' for x in r['kv'])} run {r['running']:.0f} "
                     f"wait {r['waiting']:.0f} gen {r['gen_tok_s']}/s hit {r['prefix_hit']} "
                     f"preempt {r['preemptions']:.0f} 503 {r.get('rejected', 0):.0f} queue p50/p95 [{qs}] "
                     f"ALERTS {','.join(r['alerts']) or '-'}")
        for name, c in r.get('clients', {}).items():
            share = f" kv {c['kv_tokens'] / 1e3:.0f}k/{c['share_tokens'] / 1e3:.0f}k" if 'share_tokens' in c else ''
            lines.append(f"    {name[:28]:<28} queued {c.get('queued_new', 0)}+{c.get('queued_cont', 0)} "
                         f"inflight {c.get('inflight', 0)}{share} out~{c.get('expected_out')} "
                         f"done {c.get('completed')} last admit {c.get('last_admit_age_s', '?')}s"
                         + (' HELD' if c.get('held') else ''))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=REGISTRY, help='glob of endpoints.json files')
    ap.add_argument('--every', type=float, default=60.0, help='window / sample period, seconds')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--out', default='', help='JSON lines (with --loop)')
    ap.add_argument('--starve', type=float, default=600.0, help='alert after this long without an admission')
    a = ap.parse_args()
    regs = sorted(glob.glob(a.registry))
    if not regs:
        sys.exit(f'no registry matches {a.registry}')
    prev = {}
    for r in regs:
        try:
            prev[r] = snapshot(r)
        except (OSError, ValueError):
            pass
    while True:
        time.sleep(a.every)
        rows = []
        for r in sorted(glob.glob(a.registry)):
            try:
                cur = snapshot(r)
            except (OSError, ValueError):
                continue
            if r in prev and prev[r].get('job') == cur.get('job'):
                rows += summarize(prev[r], cur, a.starve)
            prev[r] = cur
        print(time.strftime('%FT%TZ', time.gmtime()) + '\n' + table(rows), flush=True)
        if a.out:
            with open(a.out, 'a') as f:
                f.write(''.join(json.dumps(x) + '\n' for x in rows))
        if not a.loop:
            return


if __name__ == '__main__':
    main()

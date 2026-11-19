"""Per-arm serving comparison for a campaign whose serving jobs run different arms (ops/serving/job-J/control.json).

  python -m synth.ab_report <ops> [--since EPOCH] [--window 600] [--warmup 1200] [--json OUT]

From each job's ops/metrics/samples-J.jsonl (vLLM counters every ~15 s) and the controller's admission samples:
per 10-minute window and arm, generated tok/s, prefix hit rate (GPU and CPU-offload hits over prompt tokens),
recomputed prompt tokens per finished request, preemptions, MTP acceptance, running requests and admitted agents.
An engine restart resets vLLM's counters, so intervals with a negative delta are dropped, and the first `warmup`
seconds of every arm segment (engine start, admission ramp, target search) are excluded. Each arm is reported
against the mean of the baseline arms over the same windows.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

COUNTERS = ('gen', 'prompt', 'src_compute', 'src_gpu', 'src_cpu', 'preempt', 'spec_acc', 'spec_draft', 'req_ok')


def load_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def segments(samples):
    """Split one job's samples into runs of the same arm with continuous counters."""
    segs, cur = [], []
    for s in samples:
        raw = s.get('raw')
        if not raw:
            continue
        arm = (s.get('control') or {}).get('arm', 'baseline')
        if cur:
            prev = cur[-1]
            reset = any(raw.get(k, 0) < prev['raw'].get(k, 0) for k in ('gen', 'prompt'))
            if reset or prev['arm'] != arm:
                segs.append(cur)
                cur = []
        cur.append({'t': s['timestamp'], 'raw': raw, 'arm': arm})
    if cur:
        segs.append(cur)
    return segs


def window_stats(points, t0, t1):
    """Counter deltas and gauge means of one segment inside [t0, t1)."""
    inside = [p for p in points if t0 <= p['t'] < t1]
    if len(inside) < 3:
        return None
    a, b = inside[0], inside[-1]
    dt = b['t'] - a['t']
    if dt < 0.5 * (t1 - t0):
        return None
    d = {k: b['raw'].get(k, 0) - a['raw'].get(k, 0) for k in COUNTERS}
    gauges = {k: sum(p['raw'].get(k, 0) for p in inside) / len(inside) for k in ('run', 'wait', 'kv')}
    return {'seconds': dt, **d, **gauges}


def summarize(w):
    prompt = w['prompt'] or math.nan
    return {
        'gen_tok_s': w['gen'] / w['seconds'],
        'hit': (w['src_gpu'] + w['src_cpu']) / prompt,
        'hit_cpu': w['src_cpu'] / prompt,
        'computed_per_req': w['src_compute'] / w['req_ok'] if w['req_ok'] else math.nan,
        'prefill_tok_s': w['src_compute'] / w['seconds'],
        'preempt': w['preempt'],
        'accept': w['spec_acc'] / w['spec_draft'] if w['spec_draft'] else math.nan,
        'running': w['run'], 'waiting': w['wait'], 'kv': w['kv'],
    }


def agents_by_backend(ops, t0, t1):
    """Mean admitted agents and kv_fit target per backend over [t0, t1) from the controller's admission samples."""
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    for s in load_jsonl(Path(ops) / 'metrics' / 'admission-samples.jsonl'):
        if not t0 <= s.get('time', 0) < t1:
            continue
        for name, b in (s.get('backends') or {}).items():
            a = acc[name]
            a[0] += 1
            a[1] += b.get('active') or 0
            a[2] += b.get('target') or 0
    return {k: {'agents': v[1] / v[0], 'target': v[2] / v[0]} for k, v in acc.items() if v[0]}


def report(ops, since=None, window=600, warmup=1200):
    ops = Path(ops)
    jobs = sorted(ops.glob('metrics/samples-*.jsonl'), key=lambda p: int(p.stem.split('-')[1]))
    per_job = {}
    t_end = 0
    for path in jobs:
        j = int(path.stem.split('-')[1])
        segs = [s for s in segments(load_jsonl(path)) if s]
        per_job[j] = segs
        t_end = max([t_end] + [s[-1]['t'] for s in segs])
    t_start = since or min((s[0]['t'] for segs in per_job.values() for s in segs), default=0)
    rows = []  # (window start, job, arm, stats)
    start = t_start
    while start + window <= t_end:
        agents = agents_by_backend(ops, start, start + window)
        for j, segs in per_job.items():
            for seg in segs:
                if seg[0]['t'] + warmup > start or seg[-1]['t'] < start + window * 0.5:
                    continue
                w = window_stats(seg, start, start + window)
                if w:
                    row = {'start': start, 'job': j, 'arm': seg[0]['arm'], **summarize(w)}
                    row.update(agents.get(f'j{j}r0', {}))
                    rows.append(row)
        start += window
    # relative to the mean of the baseline arms in the same window
    base = defaultdict(list)
    for r in rows:
        if r['arm'].startswith('baseline'):
            base[r['start']].append(r['gen_tok_s'])
    for r in rows:
        b = base.get(r['start'])
        r['vs_baseline'] = r['gen_tok_s'] / (sum(b) / len(b)) - 1 if b else math.nan
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r['job'], r['arm'])].append(r)
    summary = []
    for (j, arm), rs in sorted(by_arm.items()):
        def mean(key):
            xs = [r[key] for r in rs if r.get(key) is not None and not math.isnan(r[key])]
            return sum(xs) / len(xs) if xs else math.nan
        rel = [r['vs_baseline'] for r in rs if not math.isnan(r['vs_baseline'])]
        sd = math.sqrt(sum((x - sum(rel) / len(rel)) ** 2 for x in rel) / (len(rel) - 1)) if len(rel) > 1 else math.nan
        summary.append({'job': j, 'arm': arm, 'windows': len(rs), 'gen_tok_s': mean('gen_tok_s'),
                        'vs_baseline': mean('vs_baseline'), 'vs_baseline_sd': sd, 'hit': mean('hit'),
                        'hit_cpu': mean('hit_cpu'), 'computed_per_req': mean('computed_per_req'),
                        'prefill_tok_s': mean('prefill_tok_s'), 'preempt': sum(r['preempt'] for r in rs),
                        'accept': mean('accept'), 'running': mean('running'), 'agents': mean('agents'),
                        'target': mean('target'), 'kv': mean('kv')})
    return {'window': window, 'warmup': warmup, 'from': t_start, 'to': t_end, 'arms': summary, 'windows': rows}


def fmt(x, spec):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ops')
    ap.add_argument('--since', type=float)
    ap.add_argument('--window', type=int, default=600)
    ap.add_argument('--warmup', type=int, default=1200)
    ap.add_argument('--json')
    a = ap.parse_args()
    r = report(a.ops, a.since, a.window, a.warmup)
    print(f"{'job':>3} {'arm':<14} {'win':>3} {'gen/s':>6} {'vs base':>8} {'sd':>5} {'hit':>5} {'cpu':>5} "
          f"{'recomp':>6} {'pre/s':>6} {'preempt':>7} {'accept':>6} {'run':>5} {'agents':>6} {'target':>6}")
    for s in r['arms']:
        print(f"{s['job']:>3} {s['arm']:<14} {s['windows']:>3} {fmt(s['gen_tok_s'], '6.0f')} "
              f"{fmt(s['vs_baseline'] * 100 if not math.isnan(s['vs_baseline']) else math.nan, '+7.1f')}% "
              f"{fmt(s['vs_baseline_sd'] * 100 if not math.isnan(s['vs_baseline_sd']) else math.nan, '5.1f')} "
              f"{fmt(s['hit'], '5.2f')} {fmt(s['hit_cpu'], '5.2f')} {fmt(s['computed_per_req'], '6.0f')} "
              f"{fmt(s['prefill_tok_s'], '6.0f')} {s['preempt']:>7.0f} {fmt(s['accept'], '6.2f')} "
              f"{fmt(s['running'], '5.0f')} {fmt(s['agents'], '6.0f')} {fmt(s['target'], '6.2f')}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()

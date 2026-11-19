"""Markdown tables from replay_arms results (#110): one row per arm (or per variant of a twin A/B arm, clients named
`<client>@<variant>`), with server gen tok/s, per-client completions / gen tok/s / gate queue, preemptions, KV max and
B200-hours.

  python -m aa_evals.qwenserve.arms_table RUNS [--arms a,b] [--serving NAME]

With --serving (the engine start's serving/<NAME> directory) each variant also gets its own engine's server-side
gen tok/s, prompt tok/s, running, KV and prefix hit averaged over the measured window from vLLM's 10-s log lines:
client-side tok/s counts a request when it completes, which is lumpy for 30-60k-token answers (A/A: 10% apart).
"""
import argparse
import collections
import json
from pathlib import Path
import re
import time

LINE = re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation '
                  r'throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: '
                  r'([\d.]+)%, Prefix cache hit rate: ([\d.]+)%')


def engine_window(log, t0, t1, year):
    """Means of vLLM's periodic stats lines with a local timestamp inside [t0, t1]."""
    rows = []
    for line in Path(log).read_text(errors='replace').splitlines():
        m = LINE.search(line)
        if not m:
            continue
        t = time.mktime(time.strptime(f'{year}-{m[1]}', '%Y-%m-%d %H:%M:%S'))
        if t0 <= t <= t1:
            rows.append([float(x) for x in m.groups()[1:]])
    if not rows:
        return None
    mean = [sum(c) / len(rows) for c in zip(*rows)]
    return {'gen': mean[1], 'prompt': mean[0], 'running': mean[2], 'waiting': mean[3], 'kv': mean[4] / 100,
            'hit': mean[5] / 100, 'lines': len(rows)}


def engine_of(runs, arm, gate, launch):
    st = json.loads((runs / 'arms' / arm / f'gate-{gate}.final.json').read_text())
    port = int(st['engines'][0]['upstream'].rsplit(':', 1)[1])
    return [r['port'] for r in launch['replicas']].index(port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--arms', default='')
    ap.add_argument('--serving', default='', help='serving/<NAME> of the engine start, for per-engine stats')
    a = ap.parse_args()
    runs = Path(a.runs)
    only = set(filter(None, a.arms.split(',')))
    rows = [json.loads(x) for x in (runs / 'mixed.jsonl').read_text().splitlines() if x.strip()]
    cost = {}
    for x in (runs / 'arms.jsonl').read_text().splitlines() if (runs / 'arms.jsonl').exists() else []:
        r = json.loads(x)
        cost[r['arm']] = r
    groups = collections.OrderedDict()
    for r in rows:
        if only and r['arm'] not in only:
            continue
        client, _, variant = r['client'].partition('@')
        groups.setdefault((r['arm'], variant), {})[client] = r
    clients = sorted({c for g in groups.values() for c in g})
    head = ['arm', 'gen tok/s (clients)', 'server gen tok/s', 'preempt', 'KV max'] + \
        [f'{c}: done / tok/s / queue p50,p95 s / err' for c in clients] + ['B200-h']
    engine_rows = []
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for (arm, variant), g in groups.items():
        any_row = next(iter(g.values()))
        cells = [arm + (f' [{variant}]' if variant else ''),
                 f"{sum(r['client_gen_tok_s'] for r in g.values()):,.0f}",
                 f"{any_row['srv_gen_tok_s']:,.0f}" + (' (all engines)' if variant else ''),
                 f"{any_row['preemptions']:.0f}", f"{any_row['kv_max']:.2f}"]
        for c in clients:
            r = g.get(c)
            cells.append('' if r is None else
                         f"{r['turns']} / {r['client_gen_tok_s']:,.0f} / {r.get('gate_queue_p50_s')},"
                         f"{r.get('gate_queue_p95_s')} / {r['errors']}")
        cells.append(str(cost.get(arm, {}).get('b200_h', '')))
        print('| ' + ' | '.join(cells) + ' |')
        if a.serving and variant and arm in cost:
            launch = json.loads((runs / 'serving' / a.serving / 'launch.json').read_text())
            i = engine_of(runs, arm, variant, launch)
            log = runs / 'serving' / a.serving / ('vllm.log' if i == 0 else f'vllm-r{i}.log')
            start = cost[arm]['t2'] - cost[arm]['replay_s'] + any_row['warmup_s']
            w = engine_window(log, start, start + any_row['measure_s'], time.gmtime(start).tm_year)
            engine_rows.append((f'{arm} [{variant}]', i, w))
    if engine_rows:
        print()
        print('| arm | engine | server gen tok/s | prompt tok/s | running | waiting | KV | prefix hit |')
        print('|---|---|---|---|---|---|---|---|')
        for name, i, w in engine_rows:
            if w:
                print(f"| {name} | {i} | {w['gen']:,.0f} | {w['prompt']:,.0f} | {w['running']:.1f} | {w['waiting']:.1f} | "
                      f"{w['kv']:.2f} | {w['hit']:.2f} |")


if __name__ == '__main__':
    main()

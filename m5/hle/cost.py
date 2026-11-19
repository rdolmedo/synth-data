"""GPU-hours as our share of engine-hours, by generated tokens (decode dominates both servers).

    python -m m5.hle.cost --rows 'P/b*.jsonl' --metrics P/gen.metrics.jsonl --gpus 4 --kind gen \
        [--host-map cpu401=m5-ds,cpu406=m5-ds2]
    python -m m5.hle.cost --rows T/traces.jsonl --metrics T/traces.jsonl.metrics.jsonl --gpus 1 --kind trace

Per server, the metrics snapshots are cut into segments at counter resets (a restarted engine) and at gaps over
30 min. For each segment: engine-hours x (our completion tokens finished in it / the server's generation-token delta
over it, all clients). Our tokens come from our rows (usage); a generator row's server is the host of the url it was
served from (--host-map), a trace row names its engine.
"""
import argparse
import collections
import glob
import json

GAP = 1800


def segments(S):
    S = sorted(S, key=lambda s: s['t'])
    seg = [S[0]]
    for s in S[1:]:
        if s['vllm:generation_tokens_total'] < seg[-1]['vllm:generation_tokens_total'] or s['t'] - seg[-1]['t'] > GAP:
            if len(seg) > 1:
                yield seg
            seg = [s]
        else:
            seg.append(s)
    if len(seg) > 1:
        yield seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True, help='jsonl file or glob')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--gpus', type=float, required=True, help='GPUs per server (m5-ds 4, a Super engine 1)')
    ap.add_argument('--kind', choices=['gen', 'trace'], required=True)
    ap.add_argument('--host-map', default='', help='gen rows: host=server,... (default: all rows on every server)')
    a = ap.parse_args()
    snaps = collections.defaultdict(list)
    for l in open(a.metrics):
        s = json.loads(l)
        if 'vllm:generation_tokens_total' in s:
            snaps[s['server']].append(s)
    hosts = dict(x.split('=', 1) for x in a.host_map.split(',') if x)
    mine_at = collections.defaultdict(list)   # server -> [(finished_at, completion tokens)]
    for path in sorted(glob.glob(a.rows)):
        for l in open(path):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if a.kind == 'gen':
                url = (r.get('server') or {}).get('url', '')
                host = url.split('//')[-1].split('.')[0].split(':')[0]
                srv = hosts.get(host) if hosts else None
                tok = (r.get('usage') or {}).get('completion') or 0
            else:
                srv, tok = r.get('engine'), r.get('completion_tokens') or 0
            for name in ([srv] if srv else list(snaps)):
                mine_at[name].append((r.get('finished_at') or 0, tok))
    out, tot = {}, 0.0
    for srv, S in sorted(snaps.items()):
        o = {'segments': 0, 'window_h': 0.0, 'server_gen_tokens': 0.0, 'our_gen_tokens': 0, 'gpu_hours': 0.0}
        for seg in segments(S):
            t0, t1 = seg[0]['t'], seg[-1]['t']
            dgen = seg[-1]['vllm:generation_tokens_total'] - seg[0]['vllm:generation_tokens_total']
            mine = sum(k for t, k in mine_at[srv] if t0 < t <= t1)
            h = (t1 - t0) / 3600
            o['segments'] += 1
            o['window_h'] += h
            o['server_gen_tokens'] += dgen
            o['our_gen_tokens'] += mine
            if dgen > 0:
                o['gpu_hours'] += h * a.gpus * min(1.0, mine / dgen)
        o['share'] = round(o['our_gen_tokens'] / o['server_gen_tokens'], 4) if o['server_gen_tokens'] else None
        o['window_h'], o['gpu_hours'] = round(o['window_h'], 2), round(o['gpu_hours'], 2)
        tot += o['gpu_hours']
        out[srv] = o
    out['total_gpu_hours'] = round(tot, 2)
    out['our_tokens_outside_windows'] = sum(k for srv in mine_at for t, k in mine_at[srv]) - sum(
        o['our_gen_tokens'] for o in out.values() if isinstance(o, dict))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

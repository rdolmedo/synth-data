"""Mixed-client replay against one server (#70 gate fairness): several synth.bench replays at once, each its own
client (`X-Gate-Client`), e.g. Pi agents + long single-turn QA + short single-turn QA, as the AA evals share a server.

  python -m aa_evals.qwenserve.mixed_replay --base http://host:port/v1 --served-name qwen38-27b \
      --clients clients.json --warmup 600 --measure 1200 --out mixed.jsonl

clients.json: [{"name": "agents", "corpus": ".../corpus.json", "count": 96, "closed_loop": true,
                "history": "reasoning_field", "overrides": {...}, "base": "http://...", "gap_scale": 2}, ...] (corpus:
a synth.bench corpus file; `base`, optional, sends that client to its own server instead of --base: per-benchmark
servers; `gap_scale` stretches the recorded tool gaps, synth.bench.Replay).
--metrics: the Prometheus bases (no /v1) whose counters give the server totals, comma-separated (default: --base;
for several servers, list each engine).
One row per client over the same measured window: turns (completed requests), gen tok/s by that client, latency,
gate queue p50/p95 (X-Gate-Queue-Ms), errors; the server totals are in every row.
"""
import argparse
import asyncio
import json
from pathlib import Path

from synth.bench import Replay
from synth.common import read


async def main(a):
    specs = json.loads(Path(a.clients).read_text())
    replays = [Replay(read(c['corpus'])['trajectories'], [{'base': c.get('base') or a.base, 'rank': None}],
                      a.served_name, seed=a.seed,
                      closed_loop=c.get('closed_loop', False), history=c.get('history', 'inline_think'),
                      headers={'X-Gate-Client': c['name']}, overrides=c.get('overrides'),
                      gap_cap=c.get('gap_cap', 300), gap_scale=c.get('gap_scale', 1.0)) for c in specs]
    metrics = [m.removesuffix('/v1') for m in (a.metrics or a.base).split(',') if m]
    rows = await asyncio.gather(*[r.run([(c['count'], a.warmup, a.measure)], metrics, tag=c['name'])
                                  for r, c in zip(replays, specs)])
    with open(a.out, 'a') as f:
        for c, rs in zip(specs, rows):
            for row in rs:
                f.write(json.dumps({'client': c['name'], 'arm': a.arm, **row}) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--served-name', required=True)
    ap.add_argument('--metrics', default='')
    ap.add_argument('--clients', required=True)
    ap.add_argument('--warmup', type=float, default=600)
    ap.add_argument('--measure', type=float, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--arm', default='')
    ap.add_argument('--out', required=True)
    asyncio.run(main(ap.parse_args()))

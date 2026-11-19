"""Trace metrics for the pilot / production reports, from an export root (export.py) and the pi_eval runs.

  python -m m5.gdpval.report --export ROOT [--metrics-before F --metrics-after F]

Per trace: end (finish / context / time_limit), submitted (at least one file in /home/user/outputs), sequence tokens
(the final request's prompt + output as served by Super: the whole linear sequence), output tokens, turns.
GPU-hours: each engine's hours over the metrics window times this worker's share of its generated tokens
(prefill is mostly prefix-cache hits; decode dominates), scaled to all traces by output tokens.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load(root: Path):
    index = json.loads((root / 'attempts' / 'index.json').read_text())
    recs = []
    for sh in index['shards'].values():
        with gzip.open(root / 'attempts' / sh['lines'], 'rt') as f:
            for ln in f:
                recs.append(json.loads(ln)['record'])
    return recs


def parse_metrics(path):
    out, cur = {}, None
    for ln in Path(path).read_text().splitlines():
        if ln.startswith('m5-super'):
            name, _, ts = ln.split()
            cur = out.setdefault(name, {'t': float(ts)})
        elif cur is not None and ln.startswith('vllm:'):
            k, v = ln.split('{')[0], float(ln.rsplit(' ', 1)[1])
            cur[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True)
    ap.add_argument('--metrics-before')
    ap.add_argument('--metrics-after')
    ap.add_argument('--target-tokens', type=float, default=330e6)
    a = ap.parse_args()
    R = [r for r in load(Path(a.export)) if not r['infra']]
    n = len(R)
    seq = np.array([r.get('seq_tokens') or 0 for r in R])
    out = np.array([r.get('output_tokens') or 0 for r in R])
    sub = np.array([r['submitted'] for r in R])
    ctx = np.array([r['end'] == 'context' for r in R])
    rep = {'n_traces': n, 'n_infra': sum(r['infra'] for r in load(Path(a.export))),
           'submit_share': round(float(sub.mean()), 3), 'context_limit_share': round(float(ctx.mean()), 3),
           'time_limit_share': round(float(np.mean([r['end'] == 'time_limit' for r in R])), 3),
           'submitted_given_context_limit': round(float(sub[ctx].mean()), 3) if ctx.any() else None,
           'seq_tokens': {'p10': int(np.percentile(seq, 10)), 'p50': int(np.median(seq)), 'mean': int(seq.mean()),
                          'p90': int(np.percentile(seq, 90))},
           'usable_tokens_per_trace': int(seq[sub].mean()) if sub.any() else 0,
           'output_tokens_mean': int(out.mean()), 'turns_median': int(np.median([r['assistant_turns'] for r in R if r.get('assistant_turns') is not None]))}
    per_problem = seq[sub].sum() / n
    rep['problems_for_target'] = int(a.target_tokens / per_problem)
    rep['problems_for_target_x1.4'] = int(1.4 * a.target_tokens / per_problem)
    if a.metrics_before and a.metrics_after:
        b, e = parse_metrics(a.metrics_before), parse_metrics(a.metrics_after)
        gh = 0.0
        for name in b:
            if name in e:
                dt = (e[name]['t'] - b[name]['t']) / 3600
                dg = e[name]['vllm:generation_tokens_total'] - b[name]['vllm:generation_tokens_total']
                rep.setdefault('engines', {})[name] = {'hours': round(dt, 2), 'gen_tokens': int(dg),
                                                       'prefix_hit': round((e[name]['vllm:prefix_cache_hits_total'] -
                                                                            b[name]['vllm:prefix_cache_hits_total']) /
                                                                           max(1, e[name]['vllm:prefix_cache_queries_total'] -
                                                                               b[name]['vllm:prefix_cache_queries_total']), 3)}
                gh += dt / max(dg, 1)   # GPU-h per generated token on this engine (1 GPU each)
        gph = gh / max(len(rep.get('engines', {})), 1)
        rep['gpu_h_per_output_token'] = gph
        rep['super_gpu_h_this_set'] = round(gph * out.sum(), 2)
        rep['super_gpu_h_per_trace'] = round(gph * out.mean(), 4)
        rep['super_gpu_h_projected_target'] = round(gph * out.mean() * rep['problems_for_target'], 1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

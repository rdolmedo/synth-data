#!/usr/bin/env python3
"""GPU-hours per full AAI-open eval of a model under test, from output tokens and a served generation rate (#176).

  python eval_gpu_hours.py [--ratio 1.24] [--rate h100=1750,b200=4900] [--util 0.75] [--json]

GPU-h = sum over benchmarks of (output tokens x ratio) / (rate x util) / 3600.
* output tokens: Qwen3.8-27B medium's measured totals on our harnesses at ~128k (BASE below, with their runs), the only
  model with every benchmark run; `--ratio` scales them to the model (Qwen3.6-35B-A3B: 1.24, its TB 2.1 output per
  attempt at a 131k window, 34.4k, over Qwen3.8 medium's at 132k, 27.8k). Each benchmark's own ratio is unmeasured.
* rate: the server's sustained gen tok/s at its operating point (closed-loop replay of the model's own TB 2.1 traffic
  at the knee); single-turn benchmarks decode at a similar rate when the gate keeps the engine full (mixed replay).
* util: the fraction of the server's time it runs at that rate (start-up, eval tails bound by tool time, ramps).
Numbers are outputs of these assumptions, not measurements of a full eval.
"""
import argparse
import json

# benchmark: (Qwen3.8 medium output tokens for one full pass at M3.0 repeats, source)
BASE = {
    'tb21': (89 * 27_751, 'tb21/r3-medium: 27,751 per attempt x 89 tasks x 1'),
    'tbsci_lite': (141 * 83_002, 'tbsci_lite/r3-qwen-medium-v2: 10.79M over 130 attempts, x 141 tasks x 1'),
    'automationbench': (7_012_374, 'automationbench/r3-qwen-medium-600'),
    'omniscience': (3_948_928, 'omniscience/m34-qwen-medium-128k'),
    'hle': (25_618_846, 'hle/qwen-diamond-medium (1,000 items; run at 262k, an upper bound at 128k)'),
    'gdppdf': (100 * 5 * 4_750, 'gdppdf/r3-qwen38-medium: 4,750 per item x 100 x k=5'),
    'scicode': (3_077_091, 'scicode/hc-orig-qwen38-med'),
    'gdpval': (None, 'not recorded (gdpval/q38m-m34-1105 has no token counts)'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=1.24)
    ap.add_argument('--rate', default='h100=1750,b200=4900')
    ap.add_argument('--util', type=float, default=0.75)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    rates = {k: float(v) for k, v in (x.split('=') for x in a.rate.split(','))}
    rows, total = [], 0.0
    for b, (tok, src) in BASE.items():
        t = tok * a.ratio if tok else None
        total += t or 0
        rows.append({'benchmark': b, 'output_tokens_M': None if t is None else round(t / 1e6, 2), 'source': src,
                     **{f'{g}_h': None if t is None else round(t / (r * a.util) / 3600, 2) for g, r in rates.items()}})
    out = {'assumptions': {'ratio': a.ratio, 'rates_gen_tok_s': rates, 'util': a.util}, 'rows': rows,
           'total_output_tokens_M': round(total / 1e6, 1),
           'total_gpu_h': {g: round(total / (r * a.util) / 3600, 1) for g, r in rates.items()},
           'missing': [r['benchmark'] for r in rows if r['output_tokens_M'] is None]}
    if a.json:
        print(json.dumps(out, indent=1))
        return
    g = list(rates)
    print(f"{'benchmark':16} {'Mtok':>7} " + ' '.join(f'{x + "-h":>8}' for x in g))
    for r in rows:
        print(f"{r['benchmark']:16} {r['output_tokens_M'] if r['output_tokens_M'] is not None else '-':>7} "
              + ' '.join(f"{r[x + '_h'] if r[x + '_h'] is not None else '-':>8}" for x in g))
    print(f"{'total':16} {out['total_output_tokens_M']:>7} " + ' '.join(f"{out['total_gpu_h'][x]:>8}" for x in g)
          + f"   (missing: {', '.join(out['missing']) or 'none'}; ratio {a.ratio}, util {a.util})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Markdown table of replay_bench.py results (one row per config x concurrency level).

  tput_table.py /data/aai/tbench/throughput/*/results.jsonl [--scale A0=2,P3=2]

--scale multiplies throughput columns (e.g. x2 to show a 4-GPU group as a node-level
2xDP4 number); concurrency is scaled too so rows compare at equal node load.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scale", default="", help="tag=factor,...")
    ap.add_argument("--partial", action="store_true", help="include windows cut short")
    a = ap.parse_args()
    scale = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.scale.split(",") if kv}
    rows = []
    for f in a.files:
        for line in open(f):
            r = json.loads(line)
            if r.get("partial") and not a.partial:
                continue
            rows.append(r)
    cols = ["config", "agents", "gen tok/s", "prefill tok/s", "computed/turn", "hit (gpu/cpu)", "KV mean/max",
            "running", "preempt", "spec acc (len)", "turns/s", "rollouts/h", "lat p50/p90 s", "ITL ms", "B tok/day"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        s = scale.get(r["tag"], 1.0)
        g = r.get("srv_gen_tok_s", r["gen_tok_s"]) * s
        hit = (f"{r['hit']:.0%} ({r['hit_gpu']:.0%}/{r['hit_cpu']:.0%})" if r.get("hit") is not None else "")
        spec = f"{r['spec_accept']:.0%} ({r['spec_mean_len']})" if r.get("spec_accept") is not None else "-"
        print("| " + " | ".join([
            r["tag"] + (f" x{s:g}" if s != 1 else ""), f"{r['concurrency'] * s:g}", f"{g:,.0f}",
            f"{r.get('srv_prefill_computed_tok_s', 0) * s:,.0f}", f"{r.get('computed_per_turn', 0) / 1e3:.1f}k", hit,
            f"{r.get('kv_mean', 0):.0%}/{r.get('kv_max', 0):.0%}", f"{r.get('running_mean', 0) * s:.0f}",
            f"{r.get('preemptions', 0) * s:.0f}", spec, f"{r['turns_s'] * s:.2f}", f"{r['rollouts_h'] * s:.0f}",
            f"{r['lat_p50_s']}/{r['lat_p90_s']}", f"{r.get('itl_mean_ms')}", f"{g * 86400 / 1e9:.2f}"]) + " |")


if __name__ == "__main__":
    main()

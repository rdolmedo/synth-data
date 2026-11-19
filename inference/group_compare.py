#!/usr/bin/env python3
"""Compare vLLM rank groups over an interval from Prometheus counters: finished requests, prompt
tokens split by source (computed / GPU prefix hit / external = CPU offload), generated tokens.

  group_compare.py --interval 600 g1=http://localhost:8101,http://localhost:8102 g2=http://localhost:8105,...
"""
import argparse
import re
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PATS = {
    "compute": r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\} ([0-9.e+]+)',
    "gpu_hit": r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\} ([0-9.e+]+)',
    "cpu_hit": r'^vllm:prompt_tokens_by_source_total\{[^}]*source="external_kv_transfer"[^}]*\} ([0-9.e+]+)',
    "gen": r'^vllm:generation_tokens_total\{[^}]*\} ([0-9.e+]+)',
    "req": r'^vllm:request_success_total\{[^}]*finished_reason="stop"[^}]*\} ([0-9.e+]+)',
    "kv": r'^vllm:kv_cache_usage_perc\{[^}]*\} ([0-9.e+-]+)',
    "run": r'^vllm:num_requests_running\{[^}]*\} ([0-9.e+]+)',
}


def snap(base):
    try:
        txt = OPENER.open(base + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for k, p in PATS.items():
        m = re.search(p, txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    groups = {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in a.groups}
    s0 = {b: snap(b) for bs in groups.values() for b in bs}
    time.sleep(a.interval)
    s1 = {b: snap(b) for bs in groups.values() for b in bs}
    for name, bs in groups.items():
        d = {k: 0.0 for k in PATS}
        kv, run, n = 0.0, 0.0, 0
        for b in bs:
            if not s0[b] or not s1[b]:
                continue
            for k in ("compute", "gpu_hit", "cpu_hit", "gen", "req"):
                d[k] += s1[b][k] - s0[b][k]
            kv += s1[b]["kv"]; run += s1[b]["run"]; n += 1
        prompt = d["compute"] + d["gpu_hit"] + d["cpu_hit"]
        req = max(d["req"], 1)
        print(f"{time.strftime('%H:%M')} {name}: ranks={n} run={run:.0f} kv={kv / max(n, 1):.0%} "
              f"gen={d['gen'] / a.interval:.0f}tok/s turns={d['req']:.0f} prompt/turn={prompt / req / 1e3:.1f}k "
              f"computed/turn={d['compute'] / req / 1e3:.1f}k hit={1 - d['compute'] / max(prompt, 1):.0%} "
              f"(gpu {d['gpu_hit'] / max(prompt, 1):.0%} cpu {d['cpu_hit'] / max(prompt, 1):.0%}) "
              f"prefill={d['compute'] / a.interval:.0f}tok/s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-line load summary per vLLM replica over an interval.

  vllm_stats.py [--interval 30] http://host:8001 http://host:8002
"""
import argparse
import re
import time
import urllib.request

KEYS = ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "generation_tokens_total",
        "prompt_tokens_total", "prefix_cache_hits_total", "prefix_cache_queries_total",
        "num_preemptions_total", "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"]


def scrape(base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        txt = opener.open(base + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001  (replica down / restarting)
        return None
    out = {}
    for k in KEYS:
        m = re.search(rf"^vllm:{k}\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+")
    ap.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    s0 = {b: scrape(b) for b in a.bases}
    time.sleep(a.interval)
    for b in a.bases:
        s1 = scrape(b)
        if s0[b] is None or s1 is None:
            print(f"{b.split('//')[-1]}: unreachable")
            continue
        d = {k: s1[k] - s0[b][k] for k in KEYS}
        hit = d["prefix_cache_hits_total"] / max(d["prefix_cache_queries_total"], 1)
        acc = d["spec_decode_num_accepted_tokens_total"] / max(d["spec_decode_num_draft_tokens_total"], 1)
        print(f"{b.split('//')[-1]}: run={s1['num_requests_running']:.0f} wait={s1['num_requests_waiting']:.0f} "
              f"kv={100 * s1['kv_cache_usage_perc']:.0f}% gen={d['generation_tokens_total'] / a.interval:.0f}tok/s "
              f"prefill={d['prompt_tokens_total'] / a.interval:.0f}tok/s hit={100 * hit:.0f}% "
              f"spec_acc={100 * acc:.0f}% preempt+={d['num_preemptions_total']:.0f}")


if __name__ == "__main__":
    main()

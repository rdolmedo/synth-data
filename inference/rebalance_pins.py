#!/usr/bin/env python3
"""Keep pinned conversations from thrashing one vLLM rank: move them via sticky_proxy's --pins-override.

Every --interval s: reads each backend's KV cache usage; for each rank above --hot (hottest first), moves up to
--max-moves of its active conversations (a request in the last --active-min minutes; largest prompt first, i.e. the
most KV) to the coolest ranks below --cool.  A move takes effect at the conversation's next request (one re-prefill
on the new rank).  Conversations moved in the last --cooldown-min minutes are left alone (no ping-pong).
Conversation -> rank and prompt size come from the proxy's --req-log.

  rebalance_pins.py --req-log req.jsonl --override pins_override.json http://localhost:8101 ... http://localhost:8114
"""
import argparse
import json
import os
import re
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def kv(base):
    try:
        txt = OPENER.open(base + "/metrics", timeout=10).read().decode()
        m = re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\} ([0-9.e+-]+)", txt, re.M)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="+")
    ap.add_argument("--req-log", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--hot", type=float, default=0.92)
    ap.add_argument("--cool", type=float, default=0.80)
    ap.add_argument("--max-moves", type=int, default=4)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--active-min", type=float, default=10)
    ap.add_argument("--cooldown-min", type=float, default=30)
    a = ap.parse_args()
    conv = {}                      # key -> [backend, last_t, bytes]
    moved_at = {}
    override = {}
    if os.path.exists(a.override):
        try:
            override = json.load(open(a.override))
        except ValueError:
            override = {}
    pos = 0
    while True:
        try:
            with open(a.req_log) as f:
                f.seek(pos)
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    conv[r["key"]] = [r["backend"], r["t"], r.get("bytes", 0)]
                pos = f.tell()
        except OSError:
            pass
        load = {b: kv(b) for b in a.backends}
        now = time.time()
        hot = sorted([b for b, v in load.items() if v is not None and v > a.hot], key=lambda b: -load[b])
        moves = []
        for h in hot:
            cool = sorted([b for b, v in load.items() if v is not None and v < a.cool], key=lambda b: load[b])
            if not cool:
                break
            cands = sorted([k for k, (b, t, n) in conv.items() if b == h and now - t < a.active_min * 60
                            and now - moved_at.get(k, 0) > a.cooldown_min * 60], key=lambda k: -conv[k][2])
            for i, k in enumerate(cands[:a.max_moves]):
                dst = cool[i % len(cool)]
                override[k] = dst
                conv[k][0] = dst
                moved_at[k] = now
                moves.append((h.rsplit(":", 1)[1], dst.rsplit(":", 1)[1], conv[k][2]))
        if moves:
            tmp = a.override + ".tmp"
            with open(tmp, "w") as f:
                json.dump(override, f)
            os.replace(tmp, a.override)
            kvs = " ".join(f"{b.rsplit(':', 1)[1]}={v:.0%}" for b, v in load.items() if v is not None)
            print(f"{time.strftime('%H:%M:%S')} moved {len(moves)}: "
                  + ", ".join(f"{s}->{d} ({n // 1000}kB)" for s, d, n in moves) + f" | KV {kvs}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correctness check for fine-grained prefix hits (--prefix-match-unit) on the hybrid
KDA+MLA model: an agent turn whose prompt extends the previous turn's prompt must give the
same greedy continuation whether it is prefilled cold or restored from the previous turn's
cached state (which with --prefix-match-unit lands inside an 8832-token block).

  prefix_check.py --results '/data/aai/tbench/runs/glm-v1/results/*' --test http://localhost:8301
(The server needs VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache.)

For several turn pairs (A = messages up to user turn j, B = A + recorded assistant + user turn j+1):
  cold:  reset cache, generate on B
  warm:  reset cache, generate on A (stores A's tail state), then generate on B
Prints the prompt-token sources of the warm B request (local_cache_hit should be ~len(A)) and
the greedy token agreement (and max logprob gap up to the first divergence) of cold vs warm,
next to cold vs a second cold run (the run-to-run nondeterminism floor).
"""
import argparse
import glob
import json
import random
import re
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(OPENER.open(req, timeout=1800).read() or b"{}")


def sources(base):
    txt = OPENER.open(base + "/metrics", timeout=10).read().decode()
    return {m.group(1): float(m.group(2)) for m in
            re.finditer(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="([a-z_]+)"[^}]*\} ([0-9.e+]+)', txt, re.M)}


def gen(base, messages, n):
    r = post(base + "/v1/chat/completions", {
        "model": "glm-5.3-flash", "messages": messages, "temperature": 0.0, "max_tokens": n,
        "logprobs": True, "top_logprobs": 1})
    ch = r["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    return [t["token"] for t in lp], [t["logprob"] for t in lp], r["usage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.results.rstrip("/") + "/*__*/result.json"))
    rng = random.Random(a.seed)
    rng.shuffle(files)
    done = 0
    for f in files:
        if done >= a.pairs:
            break
        try:
            m = json.load(open(f))["agent_result"]["metadata"]["all_messages"]
        except Exception:  # noqa: BLE001
            continue
        users = [i for i, x in enumerate(m) if x["role"] == "user"]
        cands = [j for j in users[2:-1] if 30_000 < sum(len(x.get("content") or "") + len(x.get("reasoning_content") or "")
                                                         for x in m[:j + 1]) < 300_000]
        if not cands:
            continue
        j = rng.choice(cands)
        nxt = next(i for i in users if i > j)
        A, B = m[:j + 1], m[:nxt + 1]
        post(a.test + "/reset_prefix_cache")
        c0 = sources(a.test)
        cold = gen(a.test, B, a.n)
        c1 = sources(a.test)
        post(a.test + "/reset_prefix_cache")
        cold2 = gen(a.test, B, a.n)   # nondeterminism floor: same prompt, cold again
        post(a.test + "/reset_prefix_cache")
        first = gen(a.test, A, 1)
        s0 = sources(a.test)
        warm = gen(a.test, B, a.n)
        s1 = sources(a.test)
        d = {k: int(s1.get(k, 0) - s0.get(k, 0)) for k in s1}
        def cmp(x, y):
            n = 0
            for p, q in zip(x[0], y[0]):
                if p != q:
                    break
                n += 1
            # logprob gap over the tokens both runs agree on (and the first divergent position)
            dl = max((abs(p - q) for p, q in zip(x[1][:n + 1], y[1][:n + 1])), default=None)
            return f"{n}/{min(len(x[0]), len(y[0]))}", round(dl, 3) if dl is not None else None
        print(json.dumps({"traj": f.split("/")[-2], "prompt_A": first[2]["prompt_tokens"], "prompt_B": warm[2]["prompt_tokens"],
                          "cold_sources": {k: int(c1.get(k, 0) - c0.get(k, 0)) for k in c1}, "warm_sources": d,
                          "cold_vs_cold2": cmp(cold, cold2), "cold_vs_warm": cmp(cold, warm),
                          "cold2_vs_warm": cmp(cold2, warm)}), flush=True)
        done += 1


if __name__ == "__main__":
    main()

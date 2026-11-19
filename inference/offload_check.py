#!/usr/bin/env python3
"""Correctness check for CPU KV offloading: the same long prompt, greedy, against
(a) a reference rank without offloading, (b) the offload rank cold, then (c) the
offload rank after a GPU-only prefix-cache reset, so (c) must load its prefix
from CPU memory.  Prints the generated text, token agreement, and the change in
prompt-token sources (local compute / local cache / external) for (c).

  offload_check.py --traj shard.jsonl.gz --ref http://localhost:8101 --test http://localhost:8105
  offload_check.py --results '/data/aai/tbench/runs/glm-v1/results/*' --model nemotron-3.5-super-vl --ref ... --test ...
(The test server needs VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache. --results reads Harbor runs' all_messages,
as inference/replay_bench.py does.)
"""
import argparse
import gzip
import json
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


def gen(base, messages, n, model="glm-5.3-flash"):
    r = post(base + "/v1/chat/completions", {
        "model": model, "messages": messages, "temperature": 0.0, "max_tokens": n,
        "logprobs": True, "top_logprobs": 1})
    ch = r["choices"][0]
    toks = [t["token"] for t in (ch.get("logprobs") or {}).get("content") or []]
    lps = [t["logprob"] for t in (ch.get("logprobs") or {}).get("content") or []]
    text = (ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or "") + "|" + (ch["message"].get("content") or "")
    return text, toks, lps, r["usage"]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj", help="SFT shard (jsonl.gz of {messages})")
    src.add_argument("--results", help="Harbor result dir glob(s), comma-separated (all_messages)")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--min-chars", type=int, default=120_000)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    msgs = None
    if a.results:
        from replay_bench import load_trajs
        records = ({"messages": [{**x, "content": x.get("content") or ""} for x in t]} for t in load_trajs(a.results, 5000))
    else:
        records = (json.loads(line) for line in gzip.open(a.traj, "rt"))
    for rec in records:
        m = rec["messages"]
        total = 0
        for i, x in enumerate(m):
            total += len(x["content"]) + len(x.get("reasoning_content", ""))
            if x["role"] == "user" and total > a.min_chars:
                msgs = m[:i + 1]
                break
        if msgs:
            break
    print("prompt messages", len(msgs), "chars", sum(len(x["content"]) + len(x.get("reasoning_content", "")) for x in msgs))
    ref = gen(a.ref, msgs, a.n, a.model)
    cold = gen(a.test, msgs, a.n, a.model)
    print("reset", post(a.test + "/reset_prefix_cache"))
    s0 = sources(a.test)
    warm = gen(a.test, msgs, a.n, a.model)
    s1 = sources(a.test)
    print("prompt tokens", warm[3]["prompt_tokens"], "source delta", {k: s1.get(k, 0) - s0.get(k, 0) for k in s1})
    for name, r in (("ref", ref), ("cold", cold), ("warm", warm)):
        print(f"--- {name}: {r[0][:300]!r}")
    def agree(x, y):
        n = 0
        for p, q in zip(x[1], y[1]):
            if p != q:
                break
            n += 1
        return n, min(len(x[1]), len(y[1]))
    print("token prefix agreement cold/warm", agree(cold, warm), "ref/cold", agree(ref, cold), "ref/warm", agree(ref, warm))
    k = min(16, len(cold[2]), len(warm[2]))
    print("max |dlogprob| first", k, "cold/warm", max((abs(p - q) for p, q in zip(cold[2][:k], warm[2][:k])), default=None))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU KV offload correctness by next-token distributions, not greedy text.

Greedy text diverges between runs of an NVFP4 / FP8-KV engine after a few tokens even without offload, so comparing
generations cannot isolate offload errors. Instead, one long prompt made unique by a nonce (no cache anywhere holds it)
is scored three ways on the same engine, each repeated:

  fresh  the first request computes the whole prompt;
  gpu    the same request again: the prefix comes from the GPU prefix cache;
  cpu    after POST /reset_prefix_cache (GPU only): the prefix must come back from the CPU tier.

For each request it records the top-k logprobs of the first generated token (and the prompt-token sources from
/metrics). A correct CPU restore gives cpu-vs-fresh differences of the same size as gpu-vs-fresh ones.

  offload_logprob_check.py --results '/data/aai/tbench/runs/glm-v1/results/*' --model nemotron-3.5-super-vl \
      --test http://host:port [--test ...]      (each test server needs VLLM_SERVER_DEV_MODE=1)
"""
import argparse
import json
import math
import time
import uuid

from offload_check import OPENER, post, sources
from replay_bench import load_trajs


def first_token(base, model, messages, k):
    r = post(base + "/v1/chat/completions", {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 1,
                                             "logprobs": True, "top_logprobs": k})
    lp = r["choices"][0]["logprobs"]["content"][0]
    return {t["token"]: t["logprob"] for t in lp["top_logprobs"]}, lp["token"]


def diff(a, b):
    """Max |logprob difference| over tokens in both top-k sets, and whether the argmax agrees."""
    common = set(a) & set(b)
    return (max((abs(a[t] - b[t]) for t in common), default=math.inf), max(a, key=a.get) == max(b, key=b.get))


def check(base, model, messages, k, repeats):
    nonce = [{"role": "system", "content": f"Session {uuid.uuid4()}."}]
    msgs = nonce + [m for m in messages if m["role"] != "system"]
    runs = []
    def scored(kind):
        s0 = sources(base)
        top, tok = first_token(base, model, msgs, k)
        s1 = sources(base)
        runs.append({"kind": kind, "top": top, "token": tok,
                     "sources": {key: s1.get(key, 0) - s0.get(key, 0) for key in s1}})
    scored("fresh")
    for _ in range(repeats):
        scored("gpu")
        post(base + "/reset_prefix_cache")
        time.sleep(1)
        scored("cpu")
    fresh = runs[0]["top"]
    out = {"base": base, "runs": runs}
    for kind in ("gpu", "cpu"):
        ds = [diff(fresh, r["top"]) for r in runs if r["kind"] == kind]
        out[kind] = {"max_abs_dlogprob": max(d[0] for d in ds), "argmax_agree": sum(d[1] for d in ds), "n": len(ds),
                     "external_tokens": [r["sources"].get("external_kv_transfer", 0) for r in runs if r["kind"] == kind]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", action="append", required=True)
    ap.add_argument("--min-chars", type=int, default=120_000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompts", type=int, default=2)
    a = ap.parse_args()
    prompts = []
    for t in load_trajs(a.results, 5000):
        total = 0
        for i, m in enumerate(t):
            total += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
            if m["role"] == "user" and total > a.min_chars:
                prompts.append([{**x, "content": x.get("content") or ""} for x in t[:i + 1]])
                break
        if len(prompts) >= a.prompts:
            break
    for base in a.test:
        for p in prompts:
            r = check(base.removesuffix("/v1"), a.model, p, a.k, a.repeats)
            print(json.dumps({"base": r["base"], "gpu": r["gpu"], "cpu": r["cpu"],
                              "fresh_sources": r["runs"][0]["sources"], "fresh_token": r["runs"][0]["token"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agentic replay benchmark for an OpenAI-compatible vLLM endpoint set.

Simulates N concurrent terminus-2-style agents using recorded GLM trajectories
(the `all_messages` Harbor stores): each agent replays one trajectory's user
turns (task prompt + recorded terminal observations), but the assistant turns
are whatever the server generates now (fed back with reasoning_content, as
terminus-2 does with interleaved thinking), capped at ~1.5x the recorded length.
Between turns the agent "runs tools" for a sampled think-time.  This reproduces the
growing-context, prefix-cache-heavy load of the real rollouts, so serving
configs can be compared on generated tokens/s at a given concurrency.

  replay_bench.py --results '/data/aai/tbench/runs/glm-v1/results/*' \
      --bases http://localhost:8001/v1,http://localhost:8002/v1 --concurrency 256 --duration 900

Steady-state mode (what inference/THROUGHPUT.md uses):

  replay_bench.py --results '/data/aai/tbench/runs/glm-v*/results/*' --bases http://localhost:8300/v1 \
      --levels 128:720:900,192:480:900 --random-start --tool-dist lognormal --cap-tokens 132000 \
      --metrics-bases http://localhost:8301,http://localhost:8302,... --out results.jsonl --tag A0

* The trajectory sample is fixed: result files are sorted, and agent i draws its sequence of
  trajectories (and start offsets) from its own RNG seeded by (--seed, i), so every config
  replays the same work in the same order, independent of timing.
* --random-start: an agent's first rollout starts at a random recorded turn (history =
  recorded messages up to there), so the run reaches the steady-state mix of short and long
  contexts within one warmup instead of all agents starting at turn 1.
* A rollout ends when its recorded turns run out, or when the next prompt would reach
  --cap-tokens (client-side estimate), or when a capping proxy (inference/sticky_proxy.py
  --cap-tokens) answers with its cap marker; the agent then starts a new trajectory.
* --levels c:warmup:measure[,c:warmup:measure...] runs concurrency levels back to back
  (agents are added, never removed); each level is measured after its own warmup.
* --metrics-bases: the vLLM ranks' base URLs (no /v1).  Their Prometheus counters are
  differenced over each measurement window (server-side generated tokens, prompt tokens by
  source = computed / GPU prefix hit / CPU-offload hit, preemptions, spec-decode acceptance,
  latency histograms) and KV usage / running / waiting are sampled every --sample seconds.
"""
import argparse
import asyncio
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import signal
import statistics
import time
import urllib.request

import httpx

CAP_MARK = "context cap reached"   # substring of sticky_proxy.CAP_MARK


def load_trajs(pattern: str, max_n: int) -> list[list[dict]]:
    out = []
    files = []
    for p in pattern.split(","):
        files += glob.glob(p.rstrip("/") + "/*__*/result.json")
    for rf in sorted(set(files)):
        try:
            r = json.load(open(rf))
        except Exception:  # noqa: BLE001
            continue
        msgs = ((r.get("agent_result") or {}).get("metadata") or {}).get("all_messages")
        if msgs and len(msgs) >= 4:
            out.append(msgs)
        if len(out) >= max_n:
            break
    return out


def est_tokens(m: dict) -> float:
    """Rough token count of a message (GLM tokenizer: ~3.0 chars/token for terminal
    output, ~4.1 for assistant reasoning+JSON)."""
    c = len(m.get("content") or "") if isinstance(m.get("content") or "", str) else 4000
    r = len(m.get("reasoning_content") or "")
    return (c + r) / (3.0 if m.get("role") == "user" else 4.1) + 8


class Stats:
    def __init__(self):
        self.gen = 0
        self.prompt = 0
        self.cached = 0
        self.turns = 0
        self.errors = 0
        self.rollouts = 0       # finished episodes (trajectory exhausted or capped)
        self.capped = 0
        self.lat = []           # (t_end, seconds) per completed LLM turn
        self.in_llm = 0         # agents currently waiting on the server


def sample_tool(rng: random.Random, a) -> float:
    if a.tool_dist == "lognormal":   # fit to glm-v* rollouts: median 3.6 s, mean 10.7 s per turn
        return min(a.tool_cap, rng.lognormvariate(math.log(a.tool_median), a.tool_sigma))
    return rng.uniform(0.5, 1.5) * a.tool_seconds


async def agent(i, trajs, bases, a, stats, stop, client, route, admission=None):
    rng = random.Random(a.seed * 1000003 + i)
    first = True
    while not stop.is_set():
        if admission is not None:
            await admission.enter(i)
        traj = trajs[rng.randrange(len(trajs))]
        hist = [traj[0]]
        k = 1
        if first and a.random_start:
            # random recorded turn below the cap: history = recorded messages before it
            cum, ok = est_tokens(traj[0]), [1]
            for j in range(1, len(traj) - 1, 2):
                cum += est_tokens(traj[j]) + est_tokens(traj[j + 1])
                if a.cap_tokens and cum >= 0.95 * a.cap_tokens:
                    break
                ok.append(j + 2)
            k = rng.choice(ok)
            if k >= len(traj):
                k = 1
            hist = [dict(m) for m in traj[:k]]
        first = False
        base = route(i)
        last_prompt = 0
        while not stop.is_set():
            if k >= len(traj):
                break   # recorded trajectory exhausted: rollout done
            rec = traj[k] if traj[k]["role"] == "assistant" else None
            want = 1024
            if rec:
                want = max(64, min(a.max_tokens_cap, int(1.3 * (len(rec.get("content") or "") + len(rec.get("reasoning_content") or "")) / 3.5)))
            body = {"model": a.model, "messages": hist, "temperature": 1.0, "max_tokens": want}
            t0 = time.time()
            stats.in_llm += 1
            try:
                r = await client.post(base + "/chat/completions", json=body, timeout=7200)
                r.raise_for_status()
                d = r.json()
            except Exception as exc:  # noqa: BLE001
                stats.errors += 1
                stats.in_llm -= 1
                print(json.dumps({"event": "replay_request_error", "timestamp": time.time(),
                                  "client": i, "error_type": type(exc).__name__,
                                  "error": str(exc)}), flush=True)
                await asyncio.sleep(5)
                continue
            stats.in_llm -= 1
            m = d["choices"][0]["message"]
            content = m.get("content") or ""
            if CAP_MARK in content:      # capping proxy ended the episode
                stats.capped += 1
                break
            stats.lat.append((time.time(), time.time() - t0))
            u = d.get("usage") or {}
            stats.gen += u.get("completion_tokens", 0)
            stats.prompt += u.get("prompt_tokens", 0)
            stats.cached += ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            stats.turns += 1
            last_prompt = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            asst = {"role": "assistant", "content": content}
            if m.get("reasoning_content") or m.get("reasoning"):
                asst["reasoning_content"] = m.get("reasoning_content") or m.get("reasoning")
            hist.append(asst)
            nxt = traj[k + 1] if k + 1 < len(traj) and traj[k + 1]["role"] == "user" else None
            if nxt is None:
                break
            hist.append({"role": "user", "content": nxt["content"]})
            k += 2
            if a.cap_tokens and last_prompt + est_tokens(nxt) >= a.cap_tokens:
                stats.capped += 1
                break
            await asyncio.sleep(sample_tool(rng, a))
        else:
            break
        stats.rollouts += 1
        if admission is not None:
            admission.leave(i)


# ---------------- server-side metrics ----------------
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
COUNTERS = {
    "gen": r"vllm:generation_tokens_total",
    "prompt": r"vllm:prompt_tokens_total",
    "src_compute": r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"',
    "src_gpu": r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"',
    "src_cpu": r'vllm:prompt_tokens_by_source_total\{[^}]*source="external_kv_transfer"',
    "preempt": r"vllm:num_preemptions_total",
    "spec_acc": r"vllm:spec_decode_num_accepted_tokens_total",
    "spec_draft": r"vllm:spec_decode_num_draft_tokens_total",
    "spec_drafts": r"vllm:spec_decode_num_drafts_total",
    "req_ok": r"vllm:request_success_total",
    "ttft_sum": r"vllm:time_to_first_token_seconds_sum",
    "ttft_n": r"vllm:time_to_first_token_seconds_count",
    "itl_sum": r"vllm:inter_token_latency_seconds_sum",
    "itl_n": r"vllm:inter_token_latency_seconds_count",
    "queue_sum": r"vllm:request_queue_time_seconds_sum",
    "queue_n": r"vllm:request_queue_time_seconds_count",
    "prefill_sum": r"vllm:request_prefill_time_seconds_sum",
    "prefill_n": r"vllm:request_prefill_time_seconds_count",
}
GAUGES = {"kv": r"vllm:kv_cache_usage_perc", "run": r"vllm:num_requests_running",
          "wait": r"vllm:num_requests_waiting"}


def scrape(base: str) -> dict | None:
    try:
        txt = OPENER.open(base + "/metrics", timeout=20).read().decode()
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for k, p in {**COUNTERS, **GAUGES}.items():
        rx = re.compile(r"^" + p + (r"[^ ]*" if "{" in p else r"(?:\{[^}]*\})?") + r" ([0-9.e+-]+)$", re.M)
        out[k] = sum(float(x) for x in rx.findall(txt))
    return out


async def ascrape(bases):
    return await asyncio.gather(*[asyncio.to_thread(scrape, b) for b in bases])


async def sampler(bases, period, samples, stop, path="", tag=""):
    while not stop.is_set():
        snaps = await ascrape(bases)
        ok = [s for s in snaps if s]
        if ok:
            samples.append({"t": time.time(), "kv_mean": statistics.mean(s["kv"] for s in ok),
                            "kv_max": max(s["kv"] for s in ok), "run": sum(s["run"] for s in ok),
                            "wait": sum(s["wait"] for s in ok), "gen": sum(s["gen"] for s in ok),
                            "compute": sum(s["src_compute"] for s in ok),
                            "gpu_hit": sum(s["src_gpu"] for s in ok), "cpu_hit": sum(s["src_cpu"] for s in ok)})
            if path:
                with open(path, "a") as f:
                    f.write(json.dumps({"tag": tag, **samples[-1]}) + "\n")
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass


def server_summary(s0, s1, dt):
    d = {k: 0.0 for k in COUNTERS}
    for a, b in zip(s0, s1):
        if a and b:
            for k in COUNTERS:
                d[k] += b[k] - a[k]
    prompt = d["src_compute"] + d["src_gpu"] + d["src_cpu"]
    out = {"srv_gen_tok_s": round(d["gen"] / dt), "srv_prefill_computed_tok_s": round(d["src_compute"] / dt),
           "srv_prompt_tok_s": round(prompt / dt),
           "hit": round(1 - d["src_compute"] / prompt, 3) if prompt else None,
           "hit_gpu": round(d["src_gpu"] / prompt, 3) if prompt else None,
           "hit_cpu": round(d["src_cpu"] / prompt, 3) if prompt else None,
           "srv_turns": int(d["req_ok"]),
           "computed_per_turn": round(d["src_compute"] / max(d["req_ok"], 1)),
           "preemptions": int(d["preempt"]),
           "spec_accept": round(d["spec_acc"] / d["spec_draft"], 3) if d["spec_draft"] else None,
           "spec_mean_len": round(1 + d["spec_acc"] / d["spec_drafts"], 2) if d["spec_drafts"] else None,
           "ttft_mean_s": round(d["ttft_sum"] / d["ttft_n"], 2) if d["ttft_n"] else None,
           "itl_mean_ms": round(1000 * d["itl_sum"] / d["itl_n"], 1) if d["itl_n"] else None,
           "queue_mean_s": round(d["queue_sum"] / d["queue_n"], 2) if d["queue_n"] else None,
           "prefill_mean_s": round(d["prefill_sum"] / d["prefill_n"], 2) if d["prefill_n"] else None}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="glob(s) of run result dirs, comma-separated")
    ap.add_argument("--bases", required=True, help="OpenAI base URLs (…/v1), comma-separated")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--warmup", type=float, default=120)
    ap.add_argument("--levels", default="", help="c:warmup:measure,... (overrides --concurrency/--warmup/--duration)")
    ap.add_argument("--tool-seconds", type=float, default=8)
    ap.add_argument("--tool-dist", choices=["uniform", "lognormal"], default="uniform")
    ap.add_argument("--tool-median", type=float, default=3.6)
    ap.add_argument("--tool-sigma", type=float, default=1.48)
    ap.add_argument("--tool-cap", type=float, default=120)
    ap.add_argument("--max-tokens-cap", type=int, default=16384)
    ap.add_argument("--cap-tokens", type=int, default=0, help="end a rollout when the next prompt would reach this")
    ap.add_argument("--random-start", action="store_true")
    ap.add_argument("--max-trajs", type=int, default=5000)
    ap.add_argument("--metrics-bases", default="")
    ap.add_argument("--sample", type=float, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    # Optional per-profile plan is read when this benchmark starts, allowing an
    # already queued GPU study to use an updated, explicitly recorded load sweep.
    plan_path = Path(a.out).with_name("replay-plan.json") if a.out else None
    plan = json.loads(plan_path.read_text()) if plan_path and plan_path.is_file() else {}
    if plan.get("levels"):
        a.levels = plan["levels"]
    random.seed(a.seed)
    trajs = load_trajs(a.results, a.max_trajs)
    bases = a.bases.split(",")
    mbases = [b for b in a.metrics_bases.split(",") if b]
    levels = ([tuple(float(x) for x in lv.split(":")) for lv in a.levels.split(",")] if a.levels
              else [(a.concurrency, a.warmup, a.duration)])
    if any(c < 1 or c != int(c) or w < 0 or d <= 0 for c, w, d in levels):
        raise ValueError("Invalid replay concurrency or measurement window")
    if any(b[0] <= a[0] for a, b in zip(levels, levels[1:])):
        raise ValueError("Replay concurrency levels must increase")
    extend_max = int(plan.get("extend_max_clients", levels[-1][0]))
    if plan_path and plan:
        plan_path.with_name("effective-replay-plan.json").write_text(json.dumps({
            **plan, "initial_levels": levels, "started": time.time(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print(f"{len(trajs)} trajectories, levels {levels}, {len(bases)} endpoints, {len(mbases)} metric ranks", flush=True)
    stats = Stats()
    stop = asyncio.Event()
    samples: list = []
    maxc = max(extend_max, int(max(lv[0] for lv in levels)))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required_fds = maxc + 256
    if soft != resource.RLIM_INFINITY and soft < required_fds:
        target = required_fds if hard == resource.RLIM_INFINITY else min(hard, required_fds)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        if target < required_fds:
            raise RuntimeError("Replay client's file-descriptor limit cannot support this sweep")
    limits = httpx.Limits(max_connections=maxc + 16, max_keepalive_connections=maxc + 16)
    route = lambda i: bases[i % len(bases)]  # noqa: E731
    results = []
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        tasks = []
        spath = a.out.replace(".jsonl", "") + ".samples.jsonl" if a.out else ""
        samp = asyncio.create_task(sampler(mbases, a.sample, samples, stop, spath, a.tag)) if mbases else None
        abort = asyncio.Event()   # SIGTERM/SIGINT: close the current window early, still report it
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, abort.set)

        async def pause(sec):
            try:
                await asyncio.wait_for(abort.wait(), timeout=max(sec, 0))
            except asyncio.TimeoutError:
                pass
        t_start = time.time()
        for conc, warm, dur in levels:
            level_errors0 = stats.errors
            for i in range(len(tasks), int(conc)):
                tasks.append(asyncio.create_task(agent(i, trajs, bases, a, stats, stop, client, route)))
            # progress lines during warmup and measurement
            t_lv = time.time()
            while time.time() - t_lv < warm and not abort.is_set():
                await pause(min(60, warm - (time.time() - t_lv)))
                s = samples[-1] if samples else {}
                print(f"[{a.tag} c={int(conc)} warm {time.time() - t_lv:.0f}s] turns={stats.turns} rollouts={stats.rollouts} "
                      f"in_llm={stats.in_llm} err={stats.errors} kv={s.get('kv_mean', 0):.2f}/{s.get('kv_max', 0):.2f} "
                      f"run={s.get('run', 0):.0f} wait={s.get('wait', 0):.0f}", flush=True)
            if abort.is_set():
                break
            snap0 = await ascrape(mbases) if mbases else []
            g0, p0, c0, tn0, r0, cap0, e0 = (stats.gen, stats.prompt, stats.cached, stats.turns, stats.rollouts,
                                             stats.capped, stats.errors)
            t0 = time.time()
            ns0 = len(samples)
            while time.time() - t0 < dur and not abort.is_set():
                await pause(min(60, dur - (time.time() - t0)))
                s = samples[-1] if samples else {}
                print(f"[{a.tag} c={int(conc)} meas {time.time() - t0:.0f}s] turns={stats.turns - tn0} "
                      f"gen={(stats.gen - g0) / (time.time() - t0):.0f}tok/s in_llm={stats.in_llm} "
                      f"kv={s.get('kv_mean', 0):.2f}/{s.get('kv_max', 0):.2f} run={s.get('run', 0):.0f} "
                      f"wait={s.get('wait', 0):.0f}", flush=True)
            dt = time.time() - t0
            snap1 = await ascrape(mbases) if mbases else []
            lat = sorted(x for t, x in stats.lat if t >= t0)
            win = [s for s in samples[ns0:]]
            res = {"tag": a.tag, "concurrency": int(conc), "measure_s": round(dt), "warmup_s": warm,
                   "since_start_s": round(t0 - t_start), "partial": abort.is_set(),
                   "gen_tok_s": round((stats.gen - g0) / dt), "prompt_tok_s": round((stats.prompt - p0) / dt),
                   "turns_s": round((stats.turns - tn0) / dt, 3),
                   "rollouts_h": round((stats.rollouts - r0) * 3600 / dt, 1),
                   "capped_h": round((stats.capped - cap0) * 3600 / dt, 1),
                   "gen_per_turn": round((stats.gen - g0) / max(stats.turns - tn0, 1)),
                   "prompt_per_turn": round((stats.prompt - p0) / max(stats.turns - tn0, 1)),
                   "errors": stats.errors - e0,
                   "warmup_errors": e0 - level_errors0,
                   "lat_p50_s": round(lat[len(lat) // 2], 1) if lat else None,
                   "lat_p90_s": round(lat[int(len(lat) * 0.9)], 1) if lat else None}
            if mbases:
                res.update(server_summary(snap0, snap1, dt))
                if win:
                    res.update({"kv_mean": round(statistics.mean(s["kv_mean"] for s in win), 3),
                                "kv_max": round(max(s["kv_max"] for s in win), 3),
                                "running_mean": round(statistics.mean(s["run"] for s in win), 1),
                                "waiting_mean": round(statistics.mean(s["wait"] for s in win), 1)})
            res["tok_per_day_B"] = round(res.get("srv_gen_tok_s", res["gen_tok_s"]) * 86400 / 1e9, 3)
            load_stop = None
            if plan:
                throughput = res.get("srv_gen_tok_s", res["gen_tok_s"])
                previous_best = max((r.get("srv_gen_tok_s", r["gen_tok_s"]) for r in results), default=0)
                if res["errors"] or res["warmup_errors"]:
                    load_stop = "Request errors; do not increase load further"
                elif previous_best and throughput < previous_best * 0.8:
                    load_stop = "Throughput fell more than 20% below the best completed lower-load window"
                elif conc == levels[-1][0] and conc < extend_max and throughput > previous_best * 1.05:
                    next_conc = min(extend_max, int(conc) * 2)
                    levels.append((next_conc, max(warm, 420), max(dur, 600)))
                    res["next_adaptive_concurrency"] = next_conc
                res["replay_plan"] = str(plan_path)
                res["load_stop_reason"] = load_stop
            print(json.dumps(res), flush=True)
            results.append(res)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
            if abort.is_set() or load_stop:
                break
        stop.set()
        for t in tasks:
            t.cancel()
        if samp:
            await samp


if __name__ == "__main__":
    asyncio.run(main())

"""Isolated paired comparison of rollout-admission recovery thresholds.

Uses recorded terminal observations, freshly generated answers and simulated
tool delays. These are serving measurements, never additional pass@1 attempts.
Only the owning nemotron_study process may invoke this against its local server.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from replay_bench import agent, ascrape, load_trajs, Stats, server_summary
from nemotron_admission_report import report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from nemotron_admission import AdmissionGuard

PATTERN = ",".join(f"/data/aai/tbench/runs/glm-v{v}/results/*" for v in (1, 2, 3, 5))


class Gate:
    def __init__(self):
        self.open = asyncio.Event()
        self.open.set()
        self.active = set()
        self.admitted = 0

    async def enter(self, client_id):
        await self.open.wait()
        self.active.add(client_id)
        self.admitted += 1

    def leave(self, client_id):
        self.active.discard(client_id)

    def set_paused(self, paused):
        self.open.clear() if paused else self.open.set()


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


async def reset_idle_cache(client, base):
    """No forced resets: a phase starts only after all prior requests drain."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        raw = (await ascrape([base]))[0]
        if raw and raw["run"] == raw["wait"] == 0:
            response = await client.post(base + "/reset_prefix_cache", timeout=60)
            response.raise_for_status()
            if response.json().get("success"):
                return
        await asyncio.sleep(5)
    raise TimeoutError("Owned replay server did not drain and reset; refusing another phase")


async def run_phase(base, out, low, seed, index, trajs, warmup, measure):
    out.mkdir()
    (out / "replay/queue").mkdir(parents=True)
    metrics = out / "metrics.jsonl"
    guard = AdmissionGuard(out, ["replay"], metrics, high=0.90, low=low, recovery_seconds=90)
    gate, stats, stop = Gate(), Stats(), asyncio.Event()
    a = SimpleNamespace(seed=seed, random_start=True, cap_tokens=132000,
                        max_tokens_cap=16384, model="nemotron-3.5-super-vl",
                        tool_dist="lognormal", tool_cap=120, tool_median=3.6,
                        tool_sigma=1.48)
    limits = httpx.Limits(max_connections=160, max_keepalive_connections=160)
    samples, tasks = [], []
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        await reset_idle_cache(client, base.removesuffix("/v1"))
        start = time.time()
        initial = (await ascrape([base.removesuffix("/v1")]))[0]
        if not initial:
            raise RuntimeError("Missing initial server counters")
        # All policies use the same 128-client demand and per-client seed. This
        # deliberately retains the initial burst: rollout admission cannot
        # revoke already running conversations, just as in live evaluation.
        for i in range(128):
            tasks.append(asyncio.create_task(agent(i, trajs, [base], a, stats, stop,
                                                   client, lambda _: base, gate)))
        first_measure = None
        previous = initial
        last_control = start - 60
        previous_sample_time = start
        try:
            while time.time() - start < warmup + measure:
                await asyncio.sleep(min(30, max(0, warmup + measure - (time.time() - start))))
                now = time.time()
                raw = (await ascrape([base.removesuffix("/v1")]))[0]
                if raw:
                    row = {"timestamp": now, "raw": raw,
                           "preemptions": raw["preempt"] - previous["preempt"],
                           "phase_seconds": now - start,
                           "active_rollouts": len(gate.active), "admitted_rollouts": gate.admitted}
                    append(metrics, row)
                    if now - last_control >= 59:
                        guard.update(now)
                        last_control = now
                    gate.set_paused(guard.state.get("paused", False))
                    row["paused"] = not gate.open.is_set()
                    samples.append(row)
                    print(json.dumps({"phase": index, "low": low, "seed": seed,
                          "seconds": round(now - start), "kv": round(raw["kv"], 3),
                          "tok_s": round((raw["gen"] - previous["gen"]) / (now - previous_sample_time)),
                          "preemptions": raw["preempt"] - initial["preempt"],
                          "paused": row["paused"], "active_rollouts": len(gate.active)}), flush=True)
                    previous, previous_sample_time = raw, now
                else:
                    guard.update(now)
                    gate.set_paused(True)
                if first_measure is None and now - start >= warmup and raw:
                    first_measure = (now, raw, stats.errors)
            if first_measure is None or not samples:
                raise RuntimeError("No valid measurement window")
            begin, initial_measure, warmup_errors = first_measure
            end = samples[-1]["timestamp"]
            selected = [r for r in samples if r["timestamp"] >= begin]
            # Gauges are trapezoid-time-weighted. Counters are differenced,
            # including TTFT denominators, rather than averaging interval rates.
            seconds = end - begin
            weighted = lambda key: sum((b["timestamp"] - a["timestamp"]) *
                (a["raw"][key] + b["raw"][key]) / 2 for a, b in zip(selected, selected[1:])) / seconds
            paused_s = sum((b["timestamp"] - a["timestamp"]) * a["paused"]
                           for a, b in zip(selected, selected[1:]))
            result = {"phase": index, "seed": seed, "high": 0.90, "low": low,
                      "recovery_seconds": 90, "control_seconds": 60, "sample_seconds": 30,
                      "concurrency": 128, "warmup_seconds": begin - start,
                      "measure_seconds": seconds, "start": begin, "end": end,
                      "errors": stats.errors - warmup_errors, "warmup_errors": warmup_errors,
                      "warmup_preemptions": initial_measure["preempt"] - initial["preempt"],
                      "kv_time_mean": weighted("kv"), "kv_max": max(r["raw"]["kv"] for r in selected),
                      "running_time_mean": weighted("run"), "waiting_time_mean": weighted("wait"),
                      "paused_fraction": paused_s / seconds,
                      "pause_transitions": sum(not a["paused"] and b["paused"] for a, b in zip(samples, samples[1:])),
                      "resume_transitions": sum(a["paused"] and not b["paused"] for a, b in zip(samples, samples[1:])),
                      "generated_tokens": samples[-1]["raw"]["gen"] - initial_measure["gen"],
                      "prompt_tokens": samples[-1]["raw"]["prompt"] - initial_measure["prompt"],
                      "cached_prompt_tokens": samples[-1]["raw"]["src_gpu"] - initial_measure["src_gpu"],
                      "ttft_sum": samples[-1]["raw"]["ttft_sum"] - initial_measure["ttft_sum"],
                      "ttft_count": samples[-1]["raw"]["ttft_n"] - initial_measure["ttft_n"],
                      **server_summary([initial_measure], [samples[-1]["raw"]], seconds)}
            (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reset_idle_cache(client, base.removesuffix("/v1"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--measure", type=int, default=900)
    a = ap.parse_args()
    if urlsplit(a.base).hostname != "127.0.0.1":
        raise ValueError("Admission study must target its own loopback-only server")
    if a.warmup < 60 or a.measure < 120:
        raise ValueError("Insufficient warmup or measurement duration")
    trajs = load_trajs(PATTERN, 5000)
    if not trajs:
        raise RuntimeError("No replay trajectories")
    plan = [(0, low) for low in (0.75, 0.60, 0.85)] + [(1, low) for low in (0.85, 0.60, 0.75)]
    manifest = {"created": time.time(), "workload": "GLM terminal-trajectory replay; not live pass@1",
                "worktree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
                "trajectory_count": len(trajs), "trajectory_pattern": PATTERN,
                "trajectory_sha256": hashlib.sha256(json.dumps(trajs, sort_keys=True).encode()).hexdigest(),
                "order": plan, "warmup_seconds": a.warmup, "measure_seconds": a.measure,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "replay_sha256": hashlib.sha256(Path(__file__).with_name("replay_bench.py").read_bytes()).hexdigest(),
                "guard_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "eval/nemotron_admission.py").read_bytes()).hexdigest(),
                "limitations": ["Two seeds and reversed order reduce, but do not eliminate, workload variation",
                    "Live trial lengths and tool latencies differ from replay", "90% pause threshold held fixed",
                    "A resume threshold only controls new rollouts; active contexts can still exceed capacity"]}
    (a.out / "admission-plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for index, (seed, low) in enumerate(plan):
        result = await run_phase(a.base, a.out / f"phase{index}-low{int(low*100)}-seed{seed}",
                                 low, seed, index, trajs, a.warmup, a.measure)
        results.append(result)
        append(a.out / "admission-results.jsonl", result)
        (a.out / "admission-progress.json").write_text(json.dumps({"updated": time.time(),
            "completed_phases": len(results), "total_phases": len(plan), "results": results}, indent=2) + "\n")
        (a.out / "admission-recommendation.json").write_text(json.dumps(report(results), indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

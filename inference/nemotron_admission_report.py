"""Summarize paired admission experiments without claiming a global optimum."""
from collections import defaultdict
import json
from pathlib import Path
import sys


def report(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["low"]].append(row)
    candidates = []
    for low, rows in sorted(groups.items()):
        seconds = sum(row["measure_seconds"] for row in rows)
        stable = all(not row["preemptions"] and not row["errors"] and not row["warmup_errors"] for row in rows)
        candidates.append({
            "low_watermark": low, "seeds": sorted(row["seed"] for row in rows),
            "generated_tokens_per_second": sum(row["generated_tokens"] for row in rows) / seconds,
            "throughput_by_seed": {str(row["seed"]): row["generated_tokens"] / row["measure_seconds"] for row in rows},
            "measurement_preemptions": sum(row["preemptions"] for row in rows),
            "warmup_preemptions": sum(row["warmup_preemptions"] for row in rows),
            "request_errors": sum(row["errors"] + row["warmup_errors"] for row in rows),
            "resume_transitions": sum(row["resume_transitions"] for row in rows),
            "measurement_stable": stable,
            "prefix_hit_fraction": sum(row["cached_prompt_tokens"] for row in rows) /
                max(1, sum(row["prompt_tokens"] for row in rows)),
            "ttft_mean_seconds": sum(row["ttft_sum"] for row in rows) / max(1, sum(row["ttft_count"] for row in rows)),
        })
    complete = len(results) == 6 and len(candidates) == 3 and all(row["seeds"] == [0, 1] for row in candidates)
    output = {"status": "complete" if complete else "provisional", "completed_phases": len(results),
              "candidates": candidates, "recommended_low_watermark": None,
              "recommendation": "Await all six paired phases; live policy unchanged",
              "scope": "128-client GLM trajectory replay, 1 B200 MTP5; not a global optimum or live pass@1 comparison"}
    if not complete:
        return output
    stable = [row for row in candidates if row["measurement_stable"] and row["resume_transitions"] >= 2]
    if not stable:
        output["recommendation"] = "No threshold demonstrated stable throughput with enough recovery cycles; tune admission pacing or active load before changing the live threshold"
        return output
    best = max(stable, key=lambda row: row["generated_tokens_per_second"])
    baseline = next(row for row in candidates if row["low_watermark"] == 0.75)
    output["best_measured_low_watermark"] = best["low_watermark"]
    paired_gains = {seed: best["throughput_by_seed"][seed] / baseline["throughput_by_seed"][seed] - 1
                    for seed in ("0", "1")}
    output["paired_gain_over_75"] = paired_gains
    if best["warmup_preemptions"]:
        output["recommendation"] = "Best measured recovery threshold still suffered startup preemptions; validate paced admission before live deployment"
    elif best["low_watermark"] != 0.75 and (all(gain > 0.05 for gain in paired_gains.values()) or not baseline["measurement_stable"]):
        output["recommended_low_watermark"] = best["low_watermark"]
        output["recommendation"] = "Candidate for a bounded live validation; repeated replay supports this threshold, but live trajectories differ"
    else:
        output["recommendation"] = "No consistent material gain over 75%; retain the current live policy pending more evidence"
    return output


if __name__ == "__main__":
    root = Path(sys.argv[1])
    results = [json.loads(line) for line in (root / "admission-results.jsonl").read_text().splitlines()]
    print(json.dumps(report(results), indent=2))

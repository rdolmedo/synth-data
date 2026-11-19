"""Summarize measured serving windows without calling unfinished sweeps final."""
import argparse
import json
from pathlib import Path

ROOT = Path("/data/aai/tbench/nemotron-super")
STUDIES = {
    "b200-study-v1": ("b200", [5, 1, 0]),
    "b200-omp1-v2": ("b200", [5]),
    "h100-80gb-study-v4": ("h100", [3, 1, 0]),
}


def report(root=ROOT):
    profiles, windows = [], []
    for study, (hardware, mtps) in STUDIES.items():
        for mtp in mtps:
            run = root / study / f"mtp{mtp}"
            complete = (run / "COMPLETE").is_file()
            profile = {"study": study, "hardware": hardware, "gpus": 2 if hardware == "h100" else 1,
                       "mtp": mtp, "complete": complete, "source": str(run)}
            profiles.append(profile)
            path = run / "results.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A live writer may not have finished its last line yet.
                    continue
                row = {**row, **profile}
                row["eligible"] = not row.get("partial", False) and not row.get("errors", 0) and not row.get("warmup_errors", 0)
                row["generated_tokens_per_gpu_second"] = row.get("srv_gen_tok_s", 0) / profile["gpus"]
                windows.append(row)
    best = {}
    for hardware in ["h100", "b200"]:
        candidates = [row for row in windows if row["hardware"] == hardware and row["eligible"] and row.get("srv_gen_tok_s", 0) > 0]
        stable = [row for row in candidates if row.get("preemptions", 0) == 0]
        best[hardware] = {
            "best_measured_window": max(candidates, key=lambda r: r["srv_gen_tok_s"], default=None),
            "best_without_preemptions": max(stable, key=lambda r: r["srv_gen_tok_s"], default=None),
        }
    return {"status": "complete" if all(p["complete"] for p in profiles) else "provisional",
            "workload": "GLM seed-trajectory replay with generated Nemotron answers and simulated tool delays; not pass@1 trials",
            "capacity_scope": "Measured per serving replica: one B200 or two H100 80 GB GPUs. No multi-replica scaling claim.",
            "profiles": profiles, "best_by_hardware": best, "windows": windows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(report(args.root), indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")

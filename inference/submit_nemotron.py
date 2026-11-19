"""Submit a Nemotron-only GPU job; writes the submit file before submission."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/aai/tbench/nemotron-super")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hardware", choices=["h100", "b200"])
    ap.add_argument("--mode", choices=["smoke", "study", "serve", "admission"], default="smoke")
    ap.add_argument("--name", required=True)
    ap.add_argument("--bid", type=int, default=61)
    ap.add_argument("--exclude-node", action="append", default=[])
    ap.add_argument("--omp-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="vLLM RNG seed; use distinct seeds for attempt replicas")
    ap.add_argument("--mtps", default="", help="Comma-separated speculative draft lengths; default is the hardware sweep")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.omp_threads < 1 or a.omp_threads > 16 or (a.mtps and not re.fullmatch(r"\d+(,\d+)*", a.mtps)):
        raise SystemExit("Invalid thread count or MTP list")
    if not a.name.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("name must contain only letters, digits, underscores and hyphens")
    run = ROOT / a.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "submission.json").exists() and not a.dry_run:
        raise SystemExit(f"Already submitted: {run}")
    # Exclude the 94 GB H100 NVL nodes: the baseline is two 80 GB H100s.
    resources = (2, 9.0, " && (TARGET.CUDAGlobalMemoryMb < 90000)") if a.hardware == "h100" else (1, 10.0, "")
    if any(not node.replace("-", "").isalnum() for node in a.exclude_node):
        raise SystemExit("invalid node name")
    exclusions = "".join(f' && (TARGET.UtsnameNodename != "{node}")' for node in
                         dict.fromkeys(["cpu404", "gpu132", "cpu307", "cpu207", *a.exclude_node]))
    sub = run / "job.sub"
    sub.write_text(f'''universe = vanilla
executable = {HERE}/nemotron_job.sh
arguments = {a.hardware} {a.mode} {run}
output = {run}/job.$(Cluster).out
error = {run}/job.$(Cluster).err
log = {run}/condor.log
request_gpus = {resources[0]}
request_cpus = 16
request_memory = 393216
request_disk = 180G
+JobBatchName = "nemotron-super-{a.name}"
+MaxRuntime = {129600 if a.mode == 'serve' else 14400 if a.mode == 'admission' else 43200}
+BypassLXCfs = "true"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin OMP_NUM_THREADS={a.omp_threads} BENCH_MTPS={a.mtps} SERVING_SEED={a.seed}"
requirements = (TARGET.CUDACapability == {resources[1]}){resources[2]}{exclusions}
queue
''')
    subprocess.run(["condor_submit", "-dry-run", str(run / "job.ad"), str(sub)], check=True, capture_output=True)
    print(sub)
    if not a.dry_run:
        ready = ROOT / "setup/READY"
        if not ready.is_file():
            raise SystemExit(f"Setup incomplete: missing {ready}")
        result = subprocess.run(["condor_submit_bid", str(a.bid), str(sub)], check=True, capture_output=True, text=True)
        (run / "submission.json").write_text(json.dumps({"time": time.time(), "stdout": result.stdout,
                                                       "stderr": result.stderr}, indent=2))
        print(result.stdout[-1000:])


if __name__ == "__main__":
    main()

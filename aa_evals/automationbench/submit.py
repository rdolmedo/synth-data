"""Submit an AutomationBench-AA run as one CPU HTCondor job, from a frozen copy of this code.

    python -m aa_evals.automationbench.submit --model serving:aa-super --run-id smoke-super --limit 12

Every argument after the submit-only flags is passed to run.py unchanged. The driver is a single
asyncio process (the simulated SaaS world is in-process Python); the GPU work happens on the
model endpoint. Outputs and job logs land in /data/aai/aa_evals/runs/automationbench/<run_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VENV = "/opt/aai/venvs/automationbench"
RUNS_ROOT = Path("/data/aai/aa_evals/runs/automationbench")
from aa_evals.common.condor_cost import bad_nodes  # noqa: E402  (the shared node list, read at submit time)
from aa_evals.common import serving  # noqa: E402


def freeze(dst: Path) -> dict:
    """Zip aa_evals/{__init__.py,common/{__init__,serving,jsonl}.py,tracejudge/*.py,automationbench/*} into dst (one file: /data has a file-count quota).

    The job puts the zip on PYTHONPATH; Python imports straight from it.
    """
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout  # noqa: E731
    frozen = ("aa_evals/automationbench", "aa_evals/common/serving.py", "aa_evals/common/jsonl.py")
    diff = git("diff", "HEAD", "--", *frozen)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *frozen).split()
    prov = {
        "git_sha": git("rev-parse", "HEAD").strip(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "git_dirty": bool(diff.strip() or untracked),
        "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff.strip() else None,
        "untracked": untracked,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_to": str(dst),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(REPO / "aa_evals" / "__init__.py", "aa_evals/__init__.py")
        for f in ("__init__.py", "serving.py", "jsonl.py"):   # run.py's aa_evals.common imports (registry, JSONL)
            z.write(REPO / "aa_evals" / "common" / f, f"aa_evals/common/{f}")
        for f in sorted((REPO / "aa_evals" / "tracejudge").glob("*.py")):   # the summary hook (fold_summary)
            z.write(f, f"aa_evals/tracejudge/{f.name}")
        for f in sorted(HERE.iterdir()):
            if f.is_file() and f.suffix in (".py", ".sh", ".md"):
                z.write(f, f"aa_evals/automationbench/{f.name}")
        z.writestr("aa_evals/automationbench/FROZEN_FROM.json", json.dumps(prov, indent=1))
        if diff.strip():
            z.writestr("aa_evals/automationbench/dirty.diff", diff)
    return prov


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bid", type=int, default=None, help="fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem-gb", type=int, default=24)
    p.add_argument("--disk-gb", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    a, run_args = p.parse_known_args()
    if "--run-id" not in run_args or "--model" not in run_args:
        p.error("pass at least --model and --run-id (forwarded to run.py)")
    run_id = run_args[run_args.index("--run-id") + 1]
    out_root = Path(run_args[run_args.index("--out-root") + 1]) if "--out-root" in run_args else RUNS_ROOT
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    code = run_dir / "code" / f"{stamp}.zip"
    prov = freeze(code)

    reqs = " && ".join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    # condor 'arguments' (new syntax), always: each arg in single quotes with inner ones doubled, the whole in double
    # quotes. An argument with spaces (a --note) was split into words when only quote characters triggered this.
    args = " ".join("'" + x.replace("'", "''") + "'" for x in ["-m", "aa_evals.automationbench.run", *run_args])
    args = '"' + args.replace('"', '""') + '"'
    sub = f"""universe = vanilla
executable = {VENV}/bin/python
arguments = {args}
initialdir = {run_dir}
output = {run_dir}/job.{stamp}.out
error = {run_dir}/job.{stamp}.err
log = {run_dir}/job.{stamp}.log
request_gpus = 0
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = {a.disk_gb}G
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={code} PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 AA_AUTOMATIONBENCH_VENV={VENV} CONDOR_JOB_ID=$(Cluster).$(Process){serving.condor_env()}"
requirements = {reqs}
queue
"""
    sub_path = run_dir / f"job.{stamp}.sub"
    sub_path.write_text(sub)
    if a.dry_run:
        print(sub)
        return 0
    from aa_evals.common import condor_cost
    recs = condor_cost.submit(sub_path, a.bid)   # default: the CPU bid ladder (#378); exits on failure
    rec = {"cluster": recs[0]["cluster"], "bid": recs[0]["bid"], "submit_file": recs[0]["submit_file"],
           "run_args": run_args, "code": prov, "submitted_at": stamp}
    with (run_dir / "submissions.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

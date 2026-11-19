#!/usr/bin/env python3
"""Validate the pinned TB-Science tasks in our runtime: Harbor's oracle (reference solution) and nop agents
on HTCondor CPU workers with the harbor_ns environment.

    python3 validate.py [--bench tbsci_lite] submit --run oracle-r1 --agent oracle --attempts 3 --shards 24 [--dry-run]
    python3 validate.py submit --run nop-r1 --agent nop --attempts 1 --shards 10
    python3 validate.py report --oracle 'oracle-r1-s*' --nop 'nop-r1-s*' [--oracle-rerun 'oracle-fix-s*'] \\
        [--out validation.json]

`submit` snapshots the committed repo (git archive HEAD -> /data/aai/aa_evals/code/<sha>/, never a live
checkout) and shards the tasks over gen/worker.sh jobs, each running gen/run_harbor.sh with AGENT=oracle|nop.
Every Harbor job lands in <runs>/<run>-sNNN/ (synced every 2 min).

`report` classifies every trial: pass (reward >= 1), fail (a verifier reward < 1), infra (a Harbor exception
before a reward: build/start errors, harbor_ns errors, a worker lost mid-trial), and per task the oracle
solve/verify minutes.  A task is valid when every oracle attempt passes and nop fails.
"""
import argparse
import collections
import datetime
import glob
import json
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCHES = {  # --bench: (pinned data dir with tasks.tar + manifest.json, validation runs dir)
    "tbsci": ("/data/aai/aa_evals/data/tbsci/v0.1.0", "/data/aai/aa_evals/runs/tbsci/validation"),
    "tbsci_lite": ("/data/aai/aa_evals/data/tbsci_lite/v1", "/data/aai/aa_evals/runs/tbsci_lite/validation"),
}
DATA, RUNS = (Path(x) for x in BENCHES["tbsci"])
CODE = Path("/data/aai/aa_evals/code")
import sys  # noqa: E402
if str(REPO) not in sys.path:   # documented as `python3 validate.py ...` (by path): make aa_evals importable
    sys.path.insert(0, str(REPO))
from aa_evals.common.condor_cost import bad_nodes  # noqa: E402  (the shared list, read at submit time)


def snapshot() -> Path:
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                      capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit first: the snapshot is git HEAD")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.strip()
    dst = CODE / sha[:12]
    if not (dst / ".complete").exists():
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"git archive {sha} | tar -x -C {dst}", shell=True, cwd=REPO, check=True)
        (dst / ".complete").write_text(sha + "\n")
    return dst


def submit(a):
    if a.tasks_root:  # another benchmark's task directory (e.g. TB4, for the harbor_ns multi-uid checks)
        tasks = sorted(p.parent.name for p in Path(a.tasks_root).glob("*/task.toml"))
    else:
        tasks = sorted(json.loads((DATA / "manifest.json").read_text())["tasks"])
    if a.tasks_file:
        want = {t.strip() for t in open(a.tasks_file) if t.strip() and not t.startswith("#")}
        tasks = [t for t in tasks if t in want]
    code = snapshot()
    run_dir = RUNS / a.run
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    lines = []
    for i in range(a.shards):
        part = tasks[i::a.shards]
        if part:
            sf = run_dir / "shards" / f"shard_{i:03d}.txt"
            sf.write_text("\n".join(part) + "\n")
            lines.append(f"{sf} {a.run}-s{i:03d} none")
    (run_dir / "queue.txt").write_text("\n".join(lines) + "\n")
    extra = f"--environment-build-timeout-multiplier {a.build_mult}"
    if a.env_class:
        extra += f" --env {a.env_class}"  # after run_harbor.sh's own --env: the last one wins
    if a.cpu_affinity:
        extra += " --ek ns_cpu_affinity=true"
    if a.multiuid:
        extra += " --ek ns_multiuid=true"
    env = ("HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin "
           "http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy=localnet "
           f"AGENT={a.agent} "
           + (f"TASKS_ROOT={a.tasks_root} " if a.tasks_root else f"TASKS_TAR={DATA / 'tasks.tar'} ")
           + f"RUNS_ROOT={RUNS} "
           f"HARBOR_EXTRA_ARGS='{extra}'")
    req = " && ".join(["(TARGET.GPUs == 0 || TARGET.GPUs =?= undefined)",
                       *(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())])
    (run_dir / "worker.sub").write_text(f"""universe = vanilla
executable = {code}/{'aa_evals/tbsci/probe_worker.sh' if a.multiuid else 'gen/worker.sh'}
arguments = $(shard) $(name) $(api) {a.n_concurrent} {a.attempts}
output = {run_dir}/logs/$(name).out
error = {run_dir}/logs/$(name).err
log = {run_dir}/logs/condor.log
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = {a.disk_gb}G
coresize = -1
+BypassLXCfs = "true"
+MaxRuntime = 172800
getenv = false
environment = "{env}"
requirements = {req}
queue shard, name, api from {run_dir}/queue.txt
""")
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "code": str(code), "tasks": len(tasks)}, indent=1))
    print(f"{len(tasks)} tasks in {len(lines)} shards, code {code} -> {run_dir}/worker.sub")
    if not a.dry_run:
        out = subprocess.run(["condor_submit_bid", str(a.bid), str(run_dir / "worker.sub")],
                             capture_output=True, text=True, cwd=run_dir)
        print(out.stdout[-500:], out.stderr[-500:])


def _ts(x):
    return datetime.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def _span(d):
    d = d or {}
    return (_ts(d["finished_at"]) - _ts(d["started_at"])) / 60 if d.get("finished_at") and d.get("started_at") else None


def trials(pattern):
    for jd in sorted(glob.glob(str(RUNS / pattern))):
        for td in Path(jd).glob("*__*/"):
            rj = td / "result.json"
            if not rj.exists():  # trial in flight (synced before it finished)
                yield {"task": td.name.rsplit("__", 1)[0], "trial": td.name, "reward": None, "finished": False,
                       "exc": None, "msg": "", "build_min": None, "solve_min": None, "verify_min": None,
                       "total_min": None, "path": str(td)}
                continue
            try:
                r = json.loads(rj.read_text())
            except Exception:
                continue
            if not r.get("trial_name"):
                continue
            rew = (r.get("verifier_result") or {}).get("rewards") or {}
            val = rew.get("reward")
            exc = r.get("exception_info") or {}
            yield {"task": r.get("task_name", "").split("/")[-1], "trial": r["trial_name"], "reward": val,
                   "finished": r.get("finished_at") is not None,
                   "exc": exc.get("exception_type"), "msg": (exc.get("exception_message") or "")[:300],
                   "build_min": _span(r.get("environment_setup")), "solve_min": _span(r.get("agent_execution")),
                   "verify_min": _span(r.get("verifier")), "total_min": _span(r), "path": str(rj.parent)}


def outcome(t):
    if not t["finished"]:
        return "running"
    if t["reward"] is not None:
        return "pass" if t["reward"] >= 1 else "fail"
    if t["exc"] in ("AgentTimeoutError", "VerifierTimeoutError"):
        return "timeout"
    return "infra"


def report(a):
    tasks = sorted(json.loads((DATA / "manifest.json").read_text())["tasks"])
    orc, nop = collections.defaultdict(list), collections.defaultdict(list)
    for p in a.oracle:
        for t in trials(p):
            orc[t["task"]].append(t)
    for p in a.nop:
        for t in trials(p):
            nop[t["task"]].append(t)
    # re-runs (after a runtime fix or patch) replace every earlier trial of their tasks, in the order given
    for pats, dst in ((a.oracle_rerun, orc), (a.nop_rerun, nop)):
        for p in pats:
            newer = collections.defaultdict(list)
            for t in trials(p):
                newer[t["task"]].append(t)
            dst.update(newer)
    rows, verdicts = {}, collections.Counter()
    for task in tasks:
        o, n = orc.get(task, []), nop.get(task, [])
        oc = collections.Counter(outcome(t) for t in o)
        nc = collections.Counter(outcome(t) for t in n)
        if oc.get("running") or not o:
            v = "pending"
        elif oc.get("infra") and not (oc.get("fail") or oc.get("timeout")):
            v = "infra"
        elif oc.get("fail") or oc.get("timeout"):
            v = "oracle_fails" if not oc.get("pass") else "oracle_flaky"
        elif nc.get("pass"):
            v = "nop_passes"
        elif not n or nc.get("running") or not (nc.get("fail") or nc.get("timeout")):
            v = "oracle_ok_nop_pending"
        else:
            v = "valid"
        verdicts[v] += 1
        med = lambda k: statistics.median([t[k] for t in o if t[k] is not None] or [float("nan")])
        rows[task] = {"verdict": v, "oracle": dict(oc), "nop": dict(nc),
                      "build_min": med("build_min"), "solve_min": med("solve_min"),
                      "verify_min": med("verify_min"), "total_min": med("total_min"),
                      "exceptions": sorted({f"{t['exc']}: {t['msg'][:120]}" for t in o + n if t["exc"]}),
                      "trials": [t["path"] for t in o + n]}
    print(dict(verdicts))
    for task, r in rows.items():
        if r["verdict"] not in ("valid", "pending"):
            print(f"  {r['verdict']:22s} {task:40s} oracle={r['oracle']} nop={r['nop']} {r['exceptions'][:2]}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="tbsci", choices=sorted(BENCHES), help="TB-Science 0.1 or the lite pool (#68)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit")
    s.add_argument("--run", required=True)
    s.add_argument("--agent", required=True, choices=["oracle", "nop"])
    s.add_argument("--attempts", type=int, default=1)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--n-concurrent", type=int, default=4)
    s.add_argument("--tasks-file")
    s.add_argument("--tasks-root", help="directory of task dirs instead of the pinned TB-Science tar")
    s.add_argument("--cpus", type=int, default=16)
    s.add_argument("--mem-gb", type=int, default=96)
    s.add_argument("--disk-gb", type=int, default=300)
    s.add_argument("--build-mult", type=float, default=3.0, help="x each task's build_timeout_sec (cold caches)")
    s.add_argument("--env-class", default="aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment",
                   help="Harbor environment import path ('' = run_harbor.sh's harbor_ns default)")
    s.add_argument("--cpu-affinity", action=argparse.BooleanOptionalAction, default=True,
                   help="pin each trial to the task's declared cpus (harbor_ns ns_cpu_affinity)")
    s.add_argument("--multiuid", action="store_true",
                   help="harbor_ns multi-uid mode (real subordinate ids), plus a foreign-uid probe after the job")
    s.add_argument("--bid", type=int, default=30)
    s.add_argument("--dry-run", action="store_true")
    r = sp.add_parser("report")
    r.add_argument("--oracle", action="append", default=[])
    r.add_argument("--nop", action="append", default=[])
    r.add_argument("--oracle-rerun", action="append", default=[],
                   help="run patterns whose trials replace the --oracle trials of the same tasks")
    r.add_argument("--nop-rerun", action="append", default=[])
    r.add_argument("--out")
    a = ap.parse_args()
    global DATA, RUNS
    DATA, RUNS = (Path(x) for x in BENCHES[a.bench])
    submit(a) if a.cmd == "submit" else report(a)


if __name__ == "__main__":
    main()

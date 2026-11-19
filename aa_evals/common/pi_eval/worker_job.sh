#!/bin/bash
# HTCondor CPU job: one pi_eval worker (eval_q35's Pi worker against a serving registry).
#   worker_job.sh RUN_DIR SLOTS
# Runs from the run's frozen code snapshot (config.json "code"), never from the live worktree. Environment as
# eval_q35/worker_job.sh.
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../../.." && pwd)
export USER=user HOME=/home/user PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH=$CODE
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 HTTPS_PROXY=http://proxy.internal:8080
export no_proxy=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal NO_PROXY=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
# condor coresize = -1 gives a hard core limit of unlimited (default: 0), so a task's `ulimit -c unlimited`
# works under harbor_ns multi-uid (TB 2.1 custom-memory-heap-crash); no core files unless a task asks
ulimit -S -c 0 2>/dev/null || true
echo "[pi_eval worker] $(date -u +%FT%TZ) $(hostname) cpus=$(nproc) code=$CODE"
cd "$CODE"
# per-minute cgroup memory/CPU of this job (#378): size worker RAM from these peaks, never from condor's RSS-only
# MemoryUsage (sandbox tmpfs and page cache count against the job's limit).
/opt/aai/venvs/harbor/bin/python -I "$CODE/aa_evals/common/reslog.py" \
    "$1/jobs/res.${CONDOR_JOB_ID:-$$}.jsonl" 60 >/dev/null 2>&1 &
SAMPLER=$!
# The worker runs as a child, not exec'd, so the sampler is stopped before the job ends: with `exec`, the orphaned
# sampler outlived the worker and condor held every worker "over memory limit" at its clean exit (ExitCode 0, peak
# ~30 MB; smk1 smoke 2026-11-13, bisected against c30's code on the same node). Signals are forwarded to the worker.
/opt/aai/venvs/harbor/bin/python -u -m aa_evals.common.pi_eval.worker --run "$1" --slots "$2" &
WORKER=$!
trap 'kill -TERM $WORKER 2>/dev/null' TERM INT
wait $WORKER; rc=$?
while kill -0 $WORKER 2>/dev/null; do wait $WORKER; rc=$?; done
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
exit $rc

#!/bin/bash
# HTCondor CPU job: one harbor_mswea worker (worker.py) for a run.   worker_job.sh RUN_DIR
# Runs from the run's frozen code snapshot (config.json "code"), never from the live worktree.
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../../.." && pwd)
export USER=user HOME=/home/user PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH=$CODE
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 HTTPS_PROXY=http://proxy.internal:8080
export no_proxy=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal NO_PROXY=$no_proxy
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
# condor coresize = -1 gives a hard core limit of unlimited (default: 0), so a task's `ulimit -c unlimited`
# works under harbor_ns multi-uid (TB 2.1 custom-memory-heap-crash); no core files unless a task asks
ulimit -S -c 0 2>/dev/null || true
echo "[worker] $(date -u +%FT%TZ) $(hostname) cpus=$(nproc) affinity=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //') code=$CODE"
cd "$CODE"
exec /opt/aai/venvs/harbor/bin/python -u -m aa_evals.common.harbor_mswea.worker --run "$1"

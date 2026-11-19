#!/bin/bash
# gen/worker.sh, then the multi-uid teardown probe: list every file on this job's scratch (and in the
# synced Harbor job dir) that is not owned by us.  harbor_ns multi-uid mode must leave none.
#   probe_worker.sh <same args as gen/worker.sh>
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
NAME=$2
RUNS_ROOT=${RUNS_ROOT:-/data/aai/tbench/runs}
"$REPO/gen/worker.sh" "$@"; RC=$?
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/worker-$NAME}
out="$RUNS_ROOT/$NAME/foreign_uid_probe.txt"
{
  echo "host $(hostname) uid $(id -u) scratch $SCR rc=$RC $(date -Is)"
  # condor's own job/machine ads in the scratch root are expected (owner condor)
  find "$SCR" "${HARBOR_NS_LOCAL:-$SCR/harbor_ns}" "$RUNS_ROOT/$NAME" -xdev \
       \( -path "$SCR/.job.ad" -o -path "$SCR/.machine.ad" \) -prune -o \
       \( ! -uid "$(id -u)" -o ! -gid "$(id -g)" \) -printf '%u:%g %p\n' 2>&1 | head -200
  echo "end"
} > "$out"
exit $RC

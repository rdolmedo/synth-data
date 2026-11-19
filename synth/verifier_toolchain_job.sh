#!/bin/bash
# HTCondor CPU job (needs internet through the cluster proxy): build, extend or validate the synth offline verifier
# toolchain (synth/verifier_toolchain.py). Run it from a frozen detached checkout (CODE), never a live worktree.
#
#   verifier_toolchain_job.sh CODE build OUT LIMIT [REUSE_ROOT]
#   verifier_toolchain_job.sh CODE validate TOOLCHAIN SAMPLE OUTDIR NCONC
#
# Submit with condor_submit_bid 40 (see the .sub next to the build/validation outputs).
set -uo pipefail
CODE=$1; CMD=$2; shift 2
export USER=user HOME=$_CONDOR_SCRATCH_DIR/home PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
mkdir -p "$HOME"
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 HTTPS_PROXY=http://proxy.internal:8080
export no_proxy=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal NO_PROXY=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
PY=/opt/aai/venvs/harbor/bin/python
RTS=/data/aai/tbench/synth/shared/datasets/rts
cd "$CODE" || exit 2
echo "[job] $(date -u +%FT%TZ) $(hostname) cpus=$(nproc) code=$CODE cmd=$CMD $*"
case $CMD in
  build)
    OUT=$1; LIMIT=$2; REUSE=${3:-}
    "$PY" -u -m synth.verifier_toolchain build --tasks-tar "$RTS/rts-harbor-v2.tar" --order "$RTS/order.txt" \
      --limit "$LIMIT" --out "$OUT" --work "$_CONDOR_SCRATCH_DIR/vtc" --threads 24 ${REUSE:+--reuse "$REUSE"}
    ;;
  validate)
    TOOLCHAIN=$1; SAMPLE=$2; OUTDIR=$3; NCONC=$4
    "$PY" -u -m synth.verifier_toolchain validate --toolchain "$TOOLCHAIN" --sample "$SAMPLE" --out "$OUTDIR" \
      --tasks-tar "$RTS/rts-harbor-v2.tar" --n "$NCONC" --code "$CODE"
    ;;
  *) echo "unknown command $CMD" >&2; exit 2 ;;
esac
rc=$?
echo "[job] $(date -u +%FT%TZ) rc=$rc"
exit $rc

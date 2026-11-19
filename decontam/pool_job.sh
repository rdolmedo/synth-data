#!/bin/bash
# CPU job wrapper for decontam.pool (issue #119): python -m decontam.pool "$@" from this checkout (submit from a
# snapshot worktree, never a checkout being edited), with the model cache on node-local scratch.
# DECONTAM_MODULE runs another module the same way (decontam.pool_mock, decontam.pool_check).
set -euo pipefail
SNAP=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=${_CONDOR_SCRATCH_DIR:-/tmp/$USER-decontam}/hf PYTHONNOUSERSITE=1 PYTHONPATH=$SNAP PYTHONUNBUFFERED=1
export http_proxy=${http_proxy:-http://proxy.internal:8080} https_proxy=${https_proxy:-http://proxy.internal:8080} no_proxy=${no_proxy:-localnet}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} DECONTAM_TORCH_THREADS=${DECONTAM_TORCH_THREADS:-2}
PY=${DECONTAM_PY:-/data/aai/tbenchtrain/decontam_calibforge/venv/bin/python}
cd $SNAP
exec $PY -u -m ${DECONTAM_MODULE:-decontam.pool} "$@"

#!/bin/bash
# CPU job wrapper for the TB-Sci-lite decontamination (#353): python -m m5.tbsci_lite.decontam "$@" (or $DECON_MODULE)
# from the frozen checkout this script sits in, with the decontam venv. bge-small comes from a local dir, put into a
# node-local HF cache layout (HF_HOME on /data hangs on SoftFileLock), with the hub offline.
set -euo pipefail
SNAP=$(cd "$(dirname "$0")/../.." && pwd)
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER-decon-tbsci}
export HF_HOME=$SCR/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
REV=0000000000000000000000000000000000000001
HUB=$HF_HOME/hub/models--BAAI--bge-small-en-v1.5
mkdir -p $HUB/snapshots/$REV $HUB/refs
[ -f $HUB/snapshots/$REV/model.safetensors ] || cp -r /data/aai/models/bge-small-en-v1.5/. $HUB/snapshots/$REV/
printf $REV > $HUB/refs/main
export PYTHONPATH=$SNAP
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} DECONTAM_TORCH_THREADS=${DECONTAM_TORCH_THREADS:-2}
cd $SNAP
echo "host $(hostname) snap $SNAP $(git -C $SNAP rev-parse HEAD 2>/dev/null || true)"
exec /data/aai/tbenchtrain/decontam_calibforge/venv/bin/python -u -m ${DECON_MODULE:-m5.tbsci_lite.decontam} "$@"

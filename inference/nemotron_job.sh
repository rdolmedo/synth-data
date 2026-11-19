#!/bin/bash
# One finite throughput study or a serving endpoint, isolated from GLM jobs.
set -euo pipefail
export HARDWARE=$1
MODE=${2:-study}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${3:?output directory required}
SCR=${_CONDOR_SCRATCH_DIR:?this script runs inside HTCondor}
export SCRATCH=$RUN/cache MODEL=$SCR/model
export PYTHONNOUSERSITE=1
mkdir -p "$RUN" "$MODEL" "$SCRATCH"
echo "$(date -Is) $(hostname) mode=$MODE hardware=$HARDWARE devices=${CUDA_VISIBLE_DEVICES:-unset}"
timeout 60 nvidia-smi -q > "$RUN/gpu.txt"
timeout 60 nvidia-smi topo -m > "$RUN/topology.txt"
echo "$(date -Is) staging checkpoint"
cp /data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/* "$MODEL/"
echo "$(date -Is) checkpoint staged"
exec /opt/aai/venvs/harbor/bin/python -u \
  "$REPO/inference/nemotron_study.py" --hardware "$HARDWARE" --mode "$MODE" --out "$RUN"

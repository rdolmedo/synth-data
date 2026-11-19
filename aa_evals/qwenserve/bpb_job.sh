#!/usr/bin/env bash
# One GPU job: served BPB (traj_bpb.py score) of several engine configs on one docs file, one arm after another.
#   ARMS file: one arm per line, "NAME<TAB>CHECKPOINT<TAB>ENGINE_ARGS_JSON[<TAB>EXTRA score flags]"
#   condor: executable = this script, arguments = DOCS ARMS OUTDIR
# Checkpoints are staged once to node scratch. vLLM 0.32 (venvs/vllm032), CUDA from its cu13 wheel (as serve_teacher).
set -uo pipefail
DOCS=$1; ARMS=$2; OUT=$3
V=/opt/aai/venvs/vllm032
HERE=$(cd "$(dirname "$0")" && pwd)
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/bpb.$$}
mkdir -p "$SCR" "$OUT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 OMP_NUM_THREADS=8
REPO=$(cd "$HERE/../.." && pwd)
export CUDA_HOME=$(cd "$REPO" && "$V/bin/python" -B -c "from synth.vllm_launch import cuda_home_from_wheel as f; print(f(['$V'], '$SCR/cuda_home')[0])")
export PATH=$CUDA_HOME/bin:$V/bin:$PATH
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
for n in VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE XDG_CACHE_HOME; do
  export $n="$SCR/cache/${n,,}"
done
cp "$HERE/traj_bpb.py" "$DOCS" "$SCR/"
echo "[bpb] $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) CUDA_HOME=${CUDA_HOME:-none}"
while IFS=$'\t' read -r NAME CKPT EARGS EXTRA; do
  [ -z "$NAME" ] || [ "${NAME:0:1}" = "#" ] && continue
  M=$SCR/models/$(basename "$CKPT")
  if [ ! -f "$M/.staged" ]; then
    mkdir -p "$M" && ( cd "$CKPT" && find . -maxdepth 1 -type f -print0 | xargs -0 -P 16 -I{} cp "{}" "$M/{}" ) && touch "$M/.staged"
  fi
  [ -f "$OUT/$NAME.json" ] && { echo "[bpb] $NAME done already"; continue; }
  echo "[bpb] $(date -u +%T) $NAME: $CKPT $EARGS $EXTRA"
  "$V/bin/python" -B "$SCR/traj_bpb.py" score --docs "$SCR/$(basename "$DOCS")" --model "$M" --name "$NAME" \
    --out "$OUT/$NAME.json" --engine-args "$EARGS" $EXTRA > "$OUT/$NAME.log" 2>&1 < /dev/null; rc=$?
  echo "[bpb] $(date -u +%T) $NAME rc=$rc $(grep -o '"bpb": [0-9.]*' "$OUT/$NAME.log" | tail -1)"
done < "$ARMS"

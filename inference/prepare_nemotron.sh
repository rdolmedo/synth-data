#!/bin/bash
# Reproducible setup. Run on a CPU host with at least 100 GB of local disk.
set -euo pipefail
ROOT=/data/aai/tbench/nemotron-super/setup
VENV=/opt/aai/venvs/vllm032
REV=82d93699acd3be9e4617c53360fe5b3f5dbeb1b3
VLLM_REV=3f1bcfd82bb2e93fcd364fc7a5491718453a77e2
MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
LOCAL=${LOCAL:-/tmp/user/nemotron-super-model}
SHARED=/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
mkdir -p "$ROOT" "$LOCAL"
if [ ! -e "$ROOT/MODEL_READY" ]; then
  hf download "$MODEL_ID" --revision "$REV" --local-dir "$LOCAL"
  mkdir -p "$SHARED.partial"
  for f in "$LOCAL"/*; do
    [ -f "$f" ] || continue
    if [ "$(stat -c%s "$f")" -gt 1048576 ]; then
      dd if="$f" of="$SHARED.partial/$(basename "$f")" bs=64M oflag=direct conv=fsync status=none
    else
      cp "$f" "$SHARED.partial/"
    fi
  done
  [ ! -e "$SHARED" ] || { echo "Existing checkpoint lacks provenance marker; inspect it first" >&2; exit 1; }
  mv "$SHARED.partial" "$SHARED"
  printf '%s\n' "$REV" > "$ROOT/MODEL_READY"
fi
[ -x "$VENV/bin/python" ] || uv venv --python /opt/aai/venvs/harbor/bin/python "$VENV"
UV_CACHE_DIR=/tmp/user/uv-cache uv pip install --python "$VENV/bin/python" \
  "https://wheels.vllm.ai/$VLLM_REV/vllm-0.32.0rc1.dev190%2Bg3f1bcfd82-cp38-abi3-manylinux_2_28_x86_64.whl"
UV_CACHE_DIR=/tmp/user/uv-cache uv pip install --python "$VENV/bin/python" \
  --no-deps --index-url https://flashinfer.ai/whl 'flashinfer-cubin==0.7.1'
"$VENV/bin/python" -c 'import torch, vllm, flashinfer; print(torch.__version__, vllm.__version__, flashinfer.__version__)'
uv pip freeze --python "$VENV/bin/python" > "$ROOT/serving-packages.txt"
printf '%s\n' "$VLLM_REV" > "$ROOT/READY"

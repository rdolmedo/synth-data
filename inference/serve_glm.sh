#!/bin/bash
# Serve GLM-5.3-Flash with vLLM 0.31 on a group of GPUs (one replica).
#   serve_glm.sh <cuda_visible_devices> <port> [extra vllm args...]
# e.g. serve_glm.sh 0,1,2,3 8001 ; serve_glm.sh 4,5,6,7 8002
# Env overrides: MODEL (weights dir), VENV, TP, MAX_LEN, MAX_SEQS, MTP_TOKENS (0 = off).
set -euo pipefail
GPUS=$1; PORT=$2; shift 2
VENV=${VENV:-/opt/aai/venvs/vllm031}
MODEL=${MODEL:-/tmp/$USER/models/GLM-5.3-Flash}   # node-local copy; canonical copy on /data/aai/models/GLM-5.3-Flash
TP=${TP:-4}
MAX_LEN=${MAX_LEN:-1048576}      # model max_position_embeddings (1M); hybrid KDA + sparse MLA keeps KV small
MAX_SEQS=${MAX_SEQS:-512}
MTP_TOKENS=${MTP_TOKENS:-3}
SCRATCH=${SCRATCH:-/tmp/$USER/vllm-cache}
mkdir -p "$SCRATCH"
# First start on a node compiles DeepGEMM/FlashInfer/Triton kernels (~15 min);
# reuse a saved cache if there is one (made with: tar -cf ... -C /tmp/$USER vllm-cache).
CACHE_TAR=${CACHE_TAR:-/data/aai/tbench/vllm-cache-backup.tar}
if [ ! -d "$SCRATCH/vllm" ] && [ -f "$CACHE_TAR" ]; then
  tar -xf "$CACHE_TAR" -C "$(dirname "$SCRATCH")" || true
fi

export CUDA_VISIBLE_DEVICES=$GPUS
# deep_gemm asserts a CUDA_HOME at import; point it at the venv's bundled cu13.
# No system CUDA toolkit on the nodes: assemble a conventional CUDA_HOME from the
# venv's nvidia/cu13 wheels (DeepGEMM/FlashInfer JIT need bin/nvcc, include/,
# lib64/libcudart.so and lib64/stubs/libcuda.so; the wheels ship only lib/*.so.13).
CU13=$VENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$SCRATCH/cuda_home
mkdir -p "$CUDA_HOME/lib64/stubs"
for d in bin include nvvm cccl share; do [ -e "$CU13/$d" ] && ln -sfn "$CU13/$d" "$CUDA_HOME/$d"; done
for f in "$CU13"/lib/*; do ln -sfn "$f" "$CUDA_HOME/lib64/$(basename "$f")"; done
for f in "$CU13"/lib/*.so.*; do b=$(basename "$f"); ln -sfn "$f" "$CUDA_HOME/lib64/${b%%.so.*}.so"; done
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"
ln -sfn lib64 "$CUDA_HOME/lib"
# The venv's nvcc is 13.4 but torch pins the 13.0 runtime headers; DeepGEMM's JIT
# then trips CCCL's compiler/CTK version check. Newer nvcc + older headers is fine.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export VLLM_CACHE_ROOT=$SCRATCH/vllm TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor TRITON_CACHE_DIR=$SCRATCH/triton
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/flashinfer XDG_CACHE_HOME=$SCRATCH/xdg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export PYTHONNOUSERSITE=1

SPEC=()
if [ "$MTP_TOKENS" -gt 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser glm47 \
  --trust-remote-code \
  --no-enable-log-requests \
  "${SPEC[@]}" "$@"

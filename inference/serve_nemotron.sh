#!/bin/bash
# NVIDIA's pinned Super 3.5 VL recipe; only this worktree uses this environment.
# serve_nemotron.sh <port> [extra vllm args...]
set -euo pipefail
PORT=$1; shift
VENV=${VENV:-/opt/aai/venvs/vllm032}
MODEL=${MODEL:-/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
HARDWARE=${HARDWARE:?set HARDWARE to hgpu100 or b200}
SCRATCH=${SCRATCH:?set a per-job cache directory}
mkdir -p "$SCRATCH"
export CUDA_HOME=$SCRATCH/cuda_home
CU13=$VENV/lib/python3.12/site-packages/nvidia/cu13
mkdir -p "$CUDA_HOME/lib64/stubs"
for d in bin include nvvm cccl share; do [ ! -e "$CU13/$d" ] || ln -sfn "$CU13/$d" "$CUDA_HOME/$d"; done
for f in "$CU13"/lib/*; do ln -sfn "$f" "$CUDA_HOME/lib64/$(basename "$f")"; done
for f in "$CU13"/lib/*.so.*; do b=$(basename "$f"); ln -sfn "$f" "$CUDA_HOME/lib64/${b%%.so.*}.so"; done
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"
ln -sfn lib64 "$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
# Humming's NVRTC helper dlopens libnvrtc-builtins by soname in a subprocess.
# The CUDA toolkit's symlink tree alone does not put it on the loader path.
export LD_LIBRARY_PATH="$CU13/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HUMMING_CACHE_DIR=$SCRATCH/humming HUMMING_TMP_DIR=$SCRATCH/humming-tmp
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export VLLM_CACHE_ROOT=$SCRATCH/vllm TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor TRITON_CACHE_DIR=$SCRATCH/triton
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/flashinfer XDG_CACHE_HOME=$SCRATCH/xdg
export HF_HOME=$SCRATCH/hf PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export no_proxy="localhost,127.0.0.1,::1,localnet,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,::1,localnet,${NO_PROXY:-}"
OPT=()
case "$HARDWARE" in
  hgpu100) TP=2; SEQS=128; BATCH=16384; PMU=64; MTP=3
    OPT+=(--block-size 64 --compilation_config.max_cudagraph_capture_size 256) ;;
  b200) TP=1; SEQS=512; BATCH=32768; PMU=16; MTP=5
    OPT+=(--kernel-config '{"linear_backend_per_quant":{"fp8_w8a8":"torch"}}'
      --compilation-config '{"pass_config":{"fuse_attn_quant":true},"use_inductor_graph_partition":true,"cudagraph_mode":"FULL_AND_PIECEWISE"}') ;;
  *) echo "Unknown HARDWARE=$HARDWARE" >&2; exit 2 ;;
esac
MTP_TOKENS=${MTP_TOKENS:-$MTP}
if [ "$MTP_TOKENS" -gt 0 ]; then
  OPT+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi
exec "$VENV/bin/vllm" serve "$MODEL" --served-model-name nemotron-3.5-super-vl \
  --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --seed "${SERVING_SEED:-0}" \
  --tensor-parallel-size "${TP_SIZE:-$TP}" \
  --max-model-len "${MAX_LEN:-1048576}" --gpu-memory-utilization "${GPU_MEM_UTIL:-0.89}" \
  --max-num-seqs "${MAX_SEQS:-$SEQS}" --max-num-batched-tokens "${MAX_BATCHED:-$BATCH}" \
  --enable-prefix-caching --prefix-match-unit "${PREFIX_MATCH_UNIT:-$PMU}" \
  --enable-mamba-shared-prefix-checkpoint --mamba-cache-mode align \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --async-scheduling --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 --no-enable-log-requests \
  --generation-config vllm --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' "${OPT[@]}" "$@"

#!/bin/bash
# GLM-5.3-Flash with expert parallelism + data-parallel attention (one engine per GPU),
# one OpenAI API server per DP rank on consecutive ports (external LB: the client pins
# each conversation to one rank, keeping its prefix cache local).
#   serve_glm_dpep.sh <cuda_visible_devices> <base_port> [extra vllm args...]
# e.g. serve_glm_dpep.sh 4,5,6,7 8002   -> ranks on 8002..8005
# Env overrides (defaults = the 2026-10-14 production flags): MODEL, MAX_LEN, MAX_SEQS, MTP_TOKENS,
# GPU_MEM_UTIL, MAX_BATCHED, OFFLOAD_GB, ALL2ALL, PREFIX_MATCH_UNIT, FLASHINFER_MOE, TOOL_CALL_PARSER.
# Extra args are passed to `vllm serve` last
# (they win over the flags above).  Recommended agentic-generation config: serve_node.sh.
# Why: with TP the MLA latent KV is replicated on every TP rank; with DP attention each
# sequence's KV lives on one GPU, so the node holds several times more context.
set -euo pipefail
GPUS=$1; PORT=$2; shift 2
VENV=${VENV:-/opt/aai/venvs/vllm031}
# This campaign has observed NFS RPC stalls while four API workers import the
# shared runtime. Other experiments retain their existing paths and behavior.
GLM53_RUNTIME_READY=/data/aai/tbench/pi-cont16k-110m/20261015T215325Z/ops/glm53-flash/runtime/ready.json
if [ -n "${CAMPAIGN_JOB_ID:-}" ] && [ -f "$GLM53_RUNTIME_READY" ] && \
   [ "$VENV" = /opt/aai/venvs/vllm031 ]; then
  VENV=$(/opt/aai/venvs/harbor/bin/python \
    "$(dirname "$0")/../gen/glm53_runtime.py" stage)
  # Humming otherwise starts two extra CUDA-import/build processes per API
  # worker even when NVFP4 is served by FlashInfer. Build on demand instead.
  export HUMMING_DISABLE_PARALLEL_BUILD=1
fi
MODEL=${MODEL:-/tmp/$USER/models/GLM-5.3-Flash}
MAX_LEN=${MAX_LEN:-1048576}
MAX_SEQS=${MAX_SEQS:-256}
MTP_TOKENS=${MTP_TOKENS:-3}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
MAX_BATCHED=${MAX_BATCHED:-16384}   # --max-num-batched-tokens (prefill chunk per step)
OFFLOAD_GB=${OFFLOAD_GB:-0}         # --kv-offloading-size per DP rank (GiB pinned host RAM); 0 = off
ALL2ALL=${ALL2ALL:-}                # --all2all-backend (default: vLLM's allgather_reducescatter)
FLASHINFER_MOE=${FLASHINFER_MOE:-0}  # 1 = put the synthetic CUDA_HOME's nvcc on PATH, which vLLM requires
                                    # (has_flashinfer()) before using any FlashInfer MoE kernel; with 0 (the
                                    # 2026-10-14 behaviour) FP8 MoE runs on DeepGEMM and NVFP4 MoE on weight-only Marlin
PREFIX_MATCH_UNIT=${PREFIX_MATCH_UNIT:-0}  # --prefix-match-unit (tokens; 128 = prefix hits inside the 8832-token
                                    # hybrid block, see THROUGHPUT.md); 0 = vLLM default (whole blocks)
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-}  # glm47 = native tool calling (--enable-auto-tool-choice), needed by the
                                    # claude-code / opencode harnesses (gen/run_harbor.sh AGENT_KIND); empty = off
                                    # (terminus-2 needs none).  vLLM 0.31 also serves the Anthropic Messages API
                                    # (/v1/messages, /v1/messages/count_tokens) on every rank by default.
RPC_PORT=${RPC_PORT:-$((PORT + 13000))}
SUP_PORT=${SUP_PORT:-$((PORT + 1155))}   # supervisor port (vLLM default 9256 collides between groups)
SCRATCH=${SCRATCH:-/tmp/$USER/vllm-cache}
mkdir -p "$SCRATCH"
CACHE_TAR=${CACHE_TAR:-/data/aai/tbench/vllm-cache-backup.tar}
if [ ! -d "$SCRATCH/vllm" ] && [ -f "$CACHE_TAR" ]; then tar -xf "$CACHE_TAR" -C "$(dirname "$SCRATCH")" || true; fi
NDP=$(echo "$GPUS" | tr ',' '\n' | wc -l)

if [[ "$GPUS" == *GPU-* ]]; then
  GPUS=$("$VENV/bin/python" "$(dirname "$0")/resolve_gpu_ids.py" "$GPUS")
fi
export CUDA_VISIBLE_DEVICES=$GPUS
CU13=$VENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$SCRATCH/cuda_home
mkdir -p "$CUDA_HOME/lib64/stubs"
for d in bin include nvvm cccl share; do [ -e "$CU13/$d" ] && ln -sfn "$CU13/$d" "$CUDA_HOME/$d"; done
for f in "$CU13"/lib/*; do ln -sfn "$f" "$CUDA_HOME/lib64/$(basename "$f")"; done
for f in "$CU13"/lib/*.so.*; do b=$(basename "$f"); ln -sfn "$f" "$CUDA_HOME/lib64/${b%%.so.*}.so"; done
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/stubs/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_HOME/lib64/libcuda.so"
ln -sfn lib64 "$CUDA_HOME/lib"
if [ "$FLASHINFER_MOE" = 1 ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export VLLM_CACHE_ROOT=$SCRATCH/vllm TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor TRITON_CACHE_DIR=$SCRATCH/triton
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/flashinfer XDG_CACHE_HOME=$SCRATCH/xdg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"

# Campaign-only resource evidence. Existing serving jobs/defaults are unchanged.
CAMPAIGN_RESOURCE_DIR="/data/aai/tbench/pi-cont16k-110m/20261015T215325Z/ops/glm53-flash/server-${CAMPAIGN_JOB_ID:-unset}"
if [ -n "${CAMPAIGN_JOB_ID:-}" ] && [ -d "$CAMPAIGN_RESOURCE_DIR" ]; then
  (
    findmnt -T "$MODEL" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL || true
    findmnt -T "$SCRATCH" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL || true
    CGREL=$(awk -F: '$2 == "memory" {print $3}' /proc/self/cgroup)
    while true; do
      date -u +%FT%TZ
      for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events \
        "/sys/fs/cgroup/memory/$CGREL/memory.usage_in_bytes" "/sys/fs/cgroup/memory/$CGREL/memory.max_usage_in_bytes"; do
        if [ -f "$f" ]; then echo "$f"; cat "$f"; fi
      done
      ps -u "$USER" -o pid,ppid,pcpu,rss,stat,wchan:24,etime,args --sort=-pcpu | head -35 || true
      timeout -k 5 30 nvidia-smi --id="$GPUS" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
      sleep 60
    done
  ) >> "$CAMPAIGN_RESOURCE_DIR/resources.log" 2>&1 &
fi

SPEC=()
if [ "$MTP_TOKENS" -gt 0 ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}"); fi
OPT=()
if [ "$OFFLOAD_GB" != 0 ]; then OPT+=(--kv-offloading-size "$OFFLOAD_GB"); fi
if [ -n "$ALL2ALL" ]; then OPT+=(--all2all-backend "$ALL2ALL"); fi
if [ "$PREFIX_MATCH_UNIT" != 0 ]; then OPT+=(--prefix-match-unit "$PREFIX_MATCH_UNIT"); fi
if [ -n "$TOOL_CALL_PARSER" ]; then OPT+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER"); fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$NDP" --data-parallel-size-local "$NDP" \
  --data-parallel-multi-port-external-lb --data-parallel-rpc-port "$RPC_PORT" \
  --data-parallel-supervisor-port "$SUP_PORT" \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --reasoning-parser glm47 \
  --trust-remote-code \
  --no-enable-log-requests \
  "${SPEC[@]}" "${OPT[@]}" "$@"

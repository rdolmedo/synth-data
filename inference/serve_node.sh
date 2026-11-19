#!/bin/bash
# One-command GLM-5.3-Flash server for agentic rollout generation on one 8-GPU node:
# vLLM DP+EP group(s) (serve_glm_dpep.sh) + a capping, conversation-sticky proxy on one port.
# Clients (Harbor terminus-2 workers) use only http://<host>:<PROXY_PORT>/v1.
#   serve_node.sh [proxy_port [extra vllm args...]]   (default 8001; rank ports start at BASE_PORT=8101)
# Env (defaults = the recommended config measured in THROUGHPUT.md):
#   QUANT=nvfp4 | fp8    nvidia/GLM-5.3-Flash-NVFP4 (MODEL default /tmp/$USER/models/GLM-5.3-Flash-NVFP4)
#                        or the official FP8 checkpoint (/tmp/$USER/models/GLM-5.3-Flash)
#   LAYOUT=dp4x2 | dp8   two DP4+EP4 groups (GPUs 0-3, 4-7), or one DP8+EP8 group
#   MTP_TOKENS, PREFIX_MATCH_UNIT, FLASHINFER_MOE, OFFLOAD_GB (per rank), GPU_MEM_UTIL, MAX_BATCHED, MODEL,
#   TOOL_CALL_PARSER     — see serve_glm_dpep.sh (TOOL_CALL_PARSER=glm47 for the claude-code / opencode
#                        harnesses; the Anthropic /v1/messages route needs no flag)
#   CAP=132000           end an episode when its prompt reaches this many tokens (0 = off)
#   LOGDIR=/tmp/$USER/glm-serve   vLLM/proxy logs; PINS persist conversation->rank pins across proxy restarts
# Stops everything it started on SIGTERM/SIGINT.  Startup ~4-10 min (kernel cache restored from /data).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROXY_PORT=${1:-8001}; shift || true
QUANT=${QUANT:-nvfp4}
LAYOUT=${LAYOUT:-dp4x2}
BASE_PORT=${BASE_PORT:-8101}
CAP=${CAP:-132000}
LOGDIR=${LOGDIR:-/tmp/$USER/glm-serve}
PROXY_PY=${PROXY_PY:-/opt/aai/venvs/harbor/bin/python}
export MTP_TOKENS=${MTP_TOKENS:-1}
export PREFIX_MATCH_UNIT=${PREFIX_MATCH_UNIT:-128}
export FLASHINFER_MOE=${FLASHINFER_MOE:-1}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-}
EXTRA=()
case "$QUANT" in
  nvfp4) export MODEL=${MODEL:-/tmp/$USER/models/GLM-5.3-Flash-NVFP4}; EXTRA=(--enable-ep-weight-filter) ;;
  fp8)   export MODEL=${MODEL:-/tmp/$USER/models/GLM-5.3-Flash} ;;
  *) echo "unknown QUANT=$QUANT" >&2; exit 2 ;;
esac
if [ ! -f "$MODEL/config.json" ] && [ -f "/data/aai/models/$(basename "$MODEL")/config.json" ]; then
  echo "$(date +%T) copying /data/aai/models/$(basename "$MODEL") -> $MODEL (node-local weights load much faster)"
  mkdir -p "$(dirname "$MODEL")"
  cp -r "/data/aai/models/$(basename "$MODEL")" "$MODEL.partial" && mv "$MODEL.partial" "$MODEL"
fi
export OFFLOAD_GB=${OFFLOAD_GB:-120}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
export MAX_BATCHED=${MAX_BATCHED:-16384}
mkdir -p "$LOGDIR"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

case "$LAYOUT" in
  dp8)   GROUPS_=("0,1,2,3,4,5,6,7:$BASE_PORT") ;;
  dp4x2) GROUPS_=("0,1,2,3:$BASE_PORT" "4,5,6,7:$((BASE_PORT + 10))") ;;
  *) echo "unknown LAYOUT=$LAYOUT" >&2; exit 2 ;;
esac
RANKS=()
for g in "${GROUPS_[@]}"; do
  gpus=${g%%:*}; port=${g##*:}; n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  bash "$HERE/serve_glm_dpep.sh" "$gpus" "$port" "${EXTRA[@]}" "$@" > "$LOGDIR/vllm_$port.log" 2>&1 &
  PIDS+=($!)
  for ((i = 0; i < n; i++)); do RANKS+=("http://localhost:$((port + i))"); done
done
BACKENDS=$(IFS=,; echo "${RANKS[*]}")
echo "$(date +%T) starting $QUANT $LAYOUT (model $MODEL): ranks $BACKENDS (logs $LOGDIR)"
for b in "${RANKS[@]}"; do
  until curl -sf -m 5 "$b/health" > /dev/null; do
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "vLLM exited during startup, see $LOGDIR" >&2; exit 1; }; done
    sleep 10
  done
done
echo "$(date +%T) all ranks up; proxy on :$PROXY_PORT (cap $CAP tokens)"
"$PROXY_PY" "$HERE/sticky_proxy.py" --port "$PROXY_PORT" --backends "$BACKENDS" --cap-tokens "$CAP" \
  --cap-log "$LOGDIR/cap.log" --pins-file "$LOGDIR/pins_$PROXY_PORT.json" >> "$LOGDIR/proxy_$PROXY_PORT.log" 2>&1 &
PIDS+=($!)
wait -n "${PIDS[@]}"
echo "$(date +%T) a server process exited; stopping the rest" >&2
exit 1

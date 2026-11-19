#!/bin/bash
# Steady-state agentic replay benchmark of one running vLLM DP group (see THROUGHPUT.md).
# Starts a capping sticky proxy in front of the given rank ports, waits for the ranks,
# runs replay_bench.py over the concurrency levels, then stops the proxy.
#   bench_config.sh <tag> <proxy_port> <rank_port,rank_port,...> [levels]
# e.g. bench_config.sh A0 8300 8301,8302,8303,8304 128:720:900,192:480:900
# Env: OUT (results dir), RESULTS (trajectory globs), CAP (132000), SEED (0).
set -euo pipefail
TAG=$1; PPORT=$2; RANKS=$3; LEVELS=${4:-128:720:900,192:480:900}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/data/aai/tbench/throughput}
RESULTS=${RESULTS:-/data/aai/tbench/runs/glm-v1/results/*,/data/aai/tbench/runs/glm-v2/results/*,/data/aai/tbench/runs/glm-v3/results/*,/data/aai/tbench/runs/glm-v5/results/*}
CAP=${CAP:-132000}
SEED=${SEED:-0}
PROXY_PY=${PROXY_PY:-/opt/aai/venvs/harbor/bin/python}
BENCH_PY=${BENCH_PY:-/opt/aai/venvs/vllm031/bin/python}
mkdir -p "$OUT/$TAG"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}" PYTHONNOUSERSITE=1
BACKENDS=$(echo "$RANKS" | tr ',' '\n' | sed 's#^#http://localhost:#' | paste -sd,)
for b in ${BACKENDS//,/ }; do
  until curl -sf -m 5 "$b/health" > /dev/null; do sleep 10; done
done
echo "$(date +%T) ranks ready: $BACKENDS"
"$PROXY_PY" "$HERE/sticky_proxy.py" --port "$PPORT" --backends "$BACKENDS" --cap-tokens "$CAP" \
  --cap-log "$OUT/$TAG/cap.log" > "$OUT/$TAG/proxy.log" 2>&1 &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT
until curl -sf -m 5 "http://localhost:$PPORT/metrics" > /dev/null; do sleep 2; done
"$BENCH_PY" "$HERE/replay_bench.py" --results "$RESULTS" --bases "http://localhost:$PPORT/v1" \
  --levels "$LEVELS" --random-start --tool-dist lognormal --cap-tokens "$CAP" --seed "$SEED" \
  --metrics-bases "$BACKENDS" --out "$OUT/$TAG/results.jsonl" --tag "$TAG" 2>&1 | tee -a "$OUT/$TAG/bench.log"

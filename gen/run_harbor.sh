#!/bin/bash
# Run Harbor (an agent harness on a vLLM endpoint; GLM-5.3-Flash by default) over a directory
# of tasks with the harbor_ns rootless environment.
#
#   run_harbor.sh <tasks_dir> <jobs_dir> <job_name> <api_base> <n_concurrent> [extra harbor args...]
#
# e.g. run_harbor.sh /tmp/x/tasks /tmp/x/jobs glm-shard00 http://cpu404:8001/v1 64 -k 8
# AGENT_KIND selects the harness (api_base = the OpenAI-style base, http://host:port/v1):
#   terminus-2   (default) Harbor's terminus-2, driven from this host through litellm.
#                AGENT=oracle forces the oracle (api_base ignored).
#   claude-code  Claude Code CLI (CC_VERSION, default 2.1.207) inside the sandbox, Anthropic
#                Messages API at http://host:port (vLLM's /v1/messages, via inference/sticky_proxy.py).
#   opencode     OpenCode CLI (OC_VERSION, default 1.18.35) inside the sandbox, OpenAI-compatible
#                provider at http://host:port/v1 (streaming chat/completions).
#   pi           Pi coding agent (PI_VERSION, default 1.2.0) inside the sandbox, Harbor's custom
#                endpoint (model_api openai-completions) at http://host:port/v1, --thinking PI_THINKING.
#   pi-cont16k   Patched Pi 1.2.0 + automatic proxy: 16,384 output tokens, no resampling by
#                default, then continuation in Pi's history. Requires six-hour task TOMLs.
# Both CLIs are installed from the pinned-binary cache (tb_agents/harness_agents.py) and talk to
# the endpoint directly from the sandbox (its IP is put on the sandbox no_proxy list; internet
# access keeps going through the cluster proxy).  MAX_TOKENS is the per-response output cap
# for every harness; sampling is left at the server defaults for the CLI harnesses.
#
# Overrides (unset = defaults below; pi-cont16k campaign options live in gen/pi_cont16k.py):
#   HARBOR_ENVIRONMENT    Harbor --env import path (default harbor_ns.environment:NsEnvironment;
#                         alias HARBOR_ENV_CLASS)
#   PI_AGENT_CLASS        Pi agent import path (alias PI_AGENT_IMPORT), e.g. a campaign subclass
#                         of tb_agents.harness_agents:PiCont16kCached
#   HARBOR_WRAPPER_SCRIPT python script exec'd (with $HARBOR_PYTHON) instead of harbor, receiving
#                         the same arguments
set -euo pipefail
# The continuation wrapper owns its proxy and preserves all original Harbor arguments.
if [ "${AGENT_KIND:-}" = pi-cont16k ] && [ "${PI_CONT16K_PROXY_READY:-}" != 1 ]; then
  exec "${HARBOR_PYTHON:-/opt/aai/venvs/harbor/bin/python}" \
    "$(dirname "$0")/pi_cont16k.py" "$@"
fi
TASKS=$1; JOBS=$2; NAME=$3; API_BASE=$4; NCONC=$5; shift 5
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
HARBOR=${HARBOR:-/opt/aai/venvs/harbor/bin/harbor}
AGENT=${AGENT:-terminus-2}
# terminus-2 variant: the name, or an import path such as
# tb_agents.terminus_envclock:Terminus2EnvClock (time limit counts environment time only)
TERMINUS=${TERMINUS:-terminus-2}
MAX_LEN=${MAX_LEN:-1048576}        # must match vLLM --max-model-len
MAX_TOKENS=${MAX_TOKENS:-65536}    # per-response cap; Z.ai's Terminal-Bench setting
if [ "${AGENT_KIND:-}" = pi-cont16k ]; then
  MAX_TOKENS=${PI_CONT16K_CAP:-16384}
fi
MODEL=${MODEL:-hosted_vllm/glm-5.3-flash}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-}                   # empty = server/model default
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-}   # JSON object, e.g. {"truncate_history_thinking":false}

export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export HARBOR_NS_LOCAL=${HARBOR_NS_LOCAL:-/tmp/$USER/harbor_ns}
export HARBOR_NS_SHARED=${HARBOR_NS_SHARED:-/data/aai/tbench/harbor_ns}
# The LLM endpoint is on the cluster network: never send it through the proxy.
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,.internal.cluster.internal,cpu404" NO_PROXY="$no_proxy"
export HOSTED_VLLM_API_KEY=${HOSTED_VLLM_API_KEY:-EMPTY}
ulimit -n 1048576 2>/dev/null || true

AGENT_KIND=${AGENT_KIND:-terminus-2}
SERVED=${MODEL##*/}                # served model name (vLLM --served-model-name)
CC_VERSION=${CC_VERSION:-2.1.207}  # Z.ai's TB2.1 Claude Code version
CC_EFFORT=${CC_EFFORT:-max}        # Claude Code --effort -> Anthropic output_config.effort -> vLLM
                                   # reasoning_effort.  GLM-5.3's template: low|high honoured, anything
                                   # else = Max (its default, also what terminus-2/OpenCode get);
                                   # Claude Code's own default would send "high".  "" = CLI default.
OC_VERSION=${OC_VERSION:-1.18.35}
OC_CONTEXT=${OC_CONTEXT:-$MAX_LEN} # OpenCode model limit.context (its compaction threshold is
                                   # context - output)
PI_VERSION=${PI_VERSION:-1.2.0}
PI_THINKING=${PI_THINKING:-max}    # Pi --thinking -> reasoning_effort (max = GLM's default Max effort)
PI_CONTEXT=${PI_CONTEXT:-$MAX_LEN} # Pi model contextWindow (auto-compaction at contextWindow - 16384)
AGENT_SETUP_MULT=${AGENT_SETUP_MULT:-3}   # x Harbor's 360 s agent-setup timeout (apt installs on cold mirrors)

HARBOR_ENVIRONMENT=${HARBOR_ENVIRONMENT:-${HARBOR_ENV_CLASS:-harbor_ns.environment:NsEnvironment}}
ARGS=(run -p "$TASKS" -o "$JOBS" --job-name "$NAME" -n "$NCONC" -y -q
      --env "$HARBOR_ENVIRONMENT")
if [ "$AGENT_KIND" != terminus-2 ]; then
  # In-sandbox CLIs: sandboxes have no cluster search domain, so use the endpoint's IP, and
  # keep it off the proxy (HARBOR_NS_NO_PROXY -> sandbox no_proxy).
  hostport=${API_BASE#*://}; hostport=${hostport%%/*}; host=${hostport%:*}; port=${hostport##*:}
  ip=$(getent ahostsv4 "$host" | awk 'NR==1{print $1}')
  [ -n "$ip" ] || { echo "cannot resolve $host" >&2; exit 2; }
  ROOT="http://$ip:$port"
  export HARBOR_NS_NO_PROXY="$ip${HARBOR_NS_NO_PROXY:+,$HARBOR_NS_NO_PROXY}"
  ARGS+=(--agent-setup-timeout-multiplier "$AGENT_SETUP_MULT")
fi
if [ "$AGENT_KIND" = claude-code ]; then
  export CLAUDE_CODE_MAX_OUTPUT_TOKENS=$MAX_TOKENS   # Harbor's claude-code agent reads it from its own env
  ARGS+=(-a tb_agents.harness_agents:ClaudeCodeCached -m "$SERVED"
         --ak "version=$CC_VERSION"
         --ae "ANTHROPIC_BASE_URL=$ROOT"
         --ae ANTHROPIC_API_KEY=sk-local-dummy
         --ae "ANTHROPIC_SMALL_FAST_MODEL=$SERVED"   # Harbor sets ANTHROPIC_MODEL, DEFAULT_{OPUS,SONNET,HAIKU}_MODEL
                                                    # and CLAUDE_CODE_SUBAGENT_MODEL to -m itself
         --ae CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
         --ae DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
         --ae DISABLE_TELEMETRY=1 --ae DISABLE_ERROR_REPORTING=1 --ae DISABLE_AUTOUPDATER=1
         --ae DISABLE_BUG_COMMAND=1 --ae CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1
         --ae API_TIMEOUT_MS=3600000)
  [ -n "$CC_EFFORT" ] && ARGS+=(--ak "reasoning_effort=$CC_EFFORT")
elif [ "$AGENT_KIND" = opencode ]; then
  OC_CONFIG=$(python3 - "$ROOT/v1" "$SERVED" "$OC_CONTEXT" "$MAX_TOKENS" <<'PY'
import json, sys
base, served, ctx, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
print(json.dumps({
    "autoupdate": False, "share": "disabled",
    "provider": {"vllm": {
        "npm": "@ai-sdk/openai-compatible", "name": "vLLM",
        # timeout = whole request (1 h, as terminus-2's litellm timeout); header/chunk = idle limits
        "options": {"baseURL": base, "apiKey": "sk-local-dummy", "timeout": 3600000,
                    "headerTimeout": 1800000, "chunkTimeout": 1800000},
        "models": {served: {"name": served, "tool_call": True, "reasoning": True, "temperature": True,
                            # send reasoning back as reasoning_content (GLM keeps past thinking)
                            "interleaved": {"field": "reasoning_content"},
                            "limit": {"context": ctx, "output": out}}}}}}))
PY
)
  # OpenCode clamps output to 32000 unless OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX is set; the agent
  # class sets it from this (not --ae: Harbor would scrub "65536" from every trial file as a secret)
  export OC_OUTPUT_TOKEN_MAX=$MAX_TOKENS
  ARGS+=(-a tb_agents.harness_agents:OpenCodeCached -m "vllm/$SERVED"
         --ak "version=$OC_VERSION"
         --ak "opencode_config=$OC_CONFIG"
         --ae OPENCODE_DISABLE_AUTOUPDATE=1 --ae OPENCODE_DISABLE_SHARE=1
         --ae OPENCODE_DISABLE_MODELS_FETCH=1)
elif [ "$AGENT_KIND" = pi ] || [ "$AGENT_KIND" = pi-cont16k ]; then
  PI_CLASS=PiCached
  # PI_CONT16K_CLASS=PiCont16kEnvClock: time limit counts environment time only
  [ "$AGENT_KIND" = pi-cont16k ] && PI_CLASS=${PI_CONT16K_CLASS:-PiCont16kCached}
  export PI_CONTEXT_WINDOW=$PI_CONTEXT PI_MAX_OUTPUT=$MAX_TOKENS   # read by PiCached (models.json)
  ARGS+=(-a "${PI_AGENT_CLASS:-${PI_AGENT_IMPORT:-tb_agents.harness_agents:$PI_CLASS}}" -m "openai/$SERVED"
         --ak "version=$PI_VERSION"
         --ak model_api=openai-completions
         --ak "thinking=$PI_THINKING"
         --ae "OPENAI_BASE_URL=$ROOT/v1"
         --ae OPENAI_API_KEY=sk-local-dummy
         --ae PI_OFFLINE=1 --ae PI_SKIP_VERSION_CHECK=1 --ae PI_TELEMETRY=0)
elif [ "$AGENT_KIND" = mini-swe-agent-2 ] || [ "$AGENT_KIND" = mini-swe-agent-2-editor ]; then
  MINI_CONFIG=$("${HARBOR%/harbor}/python" "$REPO/eval/nemotron_mini_config.py" "$AGENT_KIND" "$ROOT/v1")
  ARGS+=(-a tb_agents.nemotron_mini:MiniSweAgentCached -m "openai/$SERVED"
         --ak version=2.4.6 --ak "max_tokens=$MAX_TOKENS" --ak "config=$MINI_CONFIG"
         --ae "OPENAI_BASE_URL=$ROOT/v1" --ae MSWEA_API_KEY=sk-local-dummy)
elif [ "$AGENT_KIND" != terminus-2 ]; then
  echo "unknown AGENT_KIND=$AGENT_KIND" >&2; exit 2
elif [ "$AGENT" = terminus-2 ]; then
  MODEL_INFO="{\"max_input_tokens\":$MAX_LEN,\"max_output_tokens\":$MAX_TOKENS,\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
  LLM_KWARGS="\"max_tokens\":$MAX_TOKENS,\"timeout\":3600"
  [ -n "$TOP_P" ] && LLM_KWARGS+=",\"top_p\":$TOP_P"
  [ -n "$CHAT_TEMPLATE_KWARGS" ] && LLM_KWARGS+=",\"extra_body\":{\"chat_template_kwargs\":$CHAT_TEMPLATE_KWARGS}"
  ARGS+=(-a "$TERMINUS" -m "$MODEL"
         --ak "api_base=$API_BASE"
         --ak "temperature=$TEMPERATURE"
         --ak "model_info=$MODEL_INFO"
         --ak "llm_kwargs={$LLM_KWARGS}"
         --ak interleaved_thinking=true
         --ak record_terminal_session=false
         --ak store_all_messages=true
         --ak 'trajectory_config={"raw_content":true,"linear_history":true}')
else
  ARGS+=(-a "$AGENT")
fi
# e.g. HARBOR_EXTRA_ARGS="--environment-build-timeout-multiplier 4" (image pulls/builds on a cold cache)
read -r -a EXTRA <<< "${HARBOR_EXTRA_ARGS:-}"
if [ -n "${HARBOR_WRAPPER_SCRIPT:-}" ]; then
  exec "${HARBOR_PYTHON:-/opt/aai/venvs/harbor/bin/python}" \
    "$HARBOR_WRAPPER_SCRIPT" "${ARGS[@]}" "${EXTRA[@]}" "$@"
fi
exec "$HARBOR" "${ARGS[@]}" "${EXTRA[@]}" "$@"

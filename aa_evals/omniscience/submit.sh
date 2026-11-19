#!/bin/bash
# Run aa_evals.omniscience.run as an HTCondor CPU job, from a frozen copy of the code.
#
#   aa_evals/omniscience/submit.sh <run_id> <run.py args, without --run-id>
#   aa_evals/omniscience/submit.sh smoke-super --model serving:aa-super --judge serving:aa-super --limit 60
#
# The client is a light HTTP loop (no GPU); the model and judge are remote endpoints. The code is copied to
# <run_dir>/code-<stamp>/ so later edits never touch a running job. An openai: judge reads OPENAI_API_KEY from
# ~/.bashrc inside the job, so the key never lands in the submit file or the job ad. Re-submitting the same run_id
# resumes it. Env knobs: BID (default 30), PYTHON, RUNS_ROOT (default /data/aai/aa_evals/runs/omniscience),
# DRY_RUN=1 (write job.sub, submit nothing). PI_RESPONSE_BUDGET (the capped protocol), when set here, is forwarded
# into the job (aa_evals/common/response_budget_env.py).
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
RUN_ID=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RUNS_ROOT=${RUNS_ROOT:-/data/aai/aa_evals/runs/omniscience}
RUN_DIR=$RUNS_ROOT/$RUN_ID
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CODE=$RUN_DIR/code-$STAMP
PYTHON=${PYTHON:-/opt/aai/venvs/harbor/bin/python}
BID=${BID:-}   # empty: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py, #378)

mkdir -p "$CODE/aa_evals" "$RUN_DIR/logs"
cp "$REPO/aa_evals/__init__.py" "$CODE/aa_evals/"
rsync -a --exclude __pycache__ "$REPO/aa_evals/omniscience" "$REPO/aa_evals/judges" "$REPO/aa_evals/tracejudge" "$CODE/aa_evals/"
mkdir -p "$CODE/aa_evals/common"
cp "$REPO/aa_evals/common/__init__.py" "$REPO/aa_evals/common/protocol.py" "$REPO/aa_evals/common/condor_cost.py" "$REPO/aa_evals/common/serving.py" "$REPO/aa_evals/common/aai_open.py" "$CODE/aa_evals/common/"
rsync -a --exclude __pycache__ --exclude 'test_*' "$REPO/aa_evals/common/dispatch" "$CODE/aa_evals/common/"   # --dispatch (#344)
# run.py imports tb_agents' response budget for every item (budget or not)
rsync -a --exclude __pycache__ --exclude 'test_*' --include '*/' --include '*.py' --exclude '*' "$REPO/tb_agents" "$CODE/"
# the shared bad-node list (tbshared condor_cost.bad_nodes(): the frozen copy's BAD_NODES + the live file), read now
REQ=$("$PYTHON" -I -c "import importlib.util as u; s = u.spec_from_file_location('cc', '$CODE/aa_evals/common/condor_cost.py'); m = u.module_from_spec(s); s.loader.exec_module(m); print(' && '.join(f'(TARGET.UtsnameNodename != \"{n}\")' for n in m.bad_nodes()))")
[ -n "$REQ" ] || { echo "could not read bad_nodes() from $CODE/aa_evals/common/condor_cost.py" >&2; exit 1; }
SHA=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$([ -n "$(git -C "$REPO" status --porcelain -- aa_evals/omniscience aa_evals/judges aa_evals/tracejudge aa_evals/common/protocol.py aa_evals/common/serving.py aa_evals/common/aai_open.py tb_agents)" ] && echo true || echo false)
printf '{"sha": "%s", "dirty": %s, "source": "%s", "frozen": "%s"}\n' "$SHA" "$DIRTY" "$REPO" "$CODE" > "$CODE/GIT_SHA"

NEED_OPENAI=0
for a in "$@"; do case $a in openai:*) NEED_OPENAI=1;; esac; done

cat > "$CODE/job.sh" <<EOF
#!/bin/bash
set -uo pipefail
cd "$CODE"
if [ $NEED_OPENAI = 1 ]; then
  # Only the one export; ~/.bashrc returns early in non-interactive shells.
  eval "\$(grep -E '^export OPENAI_API_KEY=' ~/.bashrc)"
fi
exec "$PYTHON" -u -m aa_evals.omniscience.run --run-id "$RUN_ID" --runs-root "$RUNS_ROOT" $(printf '%q ' "$@")
EOF
chmod +x "$CODE/job.sh"

NOPROXY='localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'
# " PI_RESPONSE_BUDGET='<json>'" in condor's environment syntax when set here (checked against the job's $PYTHON)
BUDGET_ENV=""
if [ -n "${PI_RESPONSE_BUDGET:-}" ]; then
  BUDGET_ENV=$(PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 "$PYTHON" -m aa_evals.common.response_budget_env condor-env "$PYTHON" "$CODE")
fi
# " TASKSYNTH_SERVING_ROOT=<root>" when set here: the job reads the registry this submitter reads
ROOT_ENV=""
if [ -n "${TASKSYNTH_SERVING_ROOT:-}" ]; then
  ROOT_ENV=$(PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 "$PYTHON" -m aa_evals.common.serving condor-env)
fi
# " AAI_OPEN_VERSION='<label>'" when set here (#481): the job records the campaign launcher's label
AAI_ENV=""
if [ -n "${AAI_OPEN_VERSION:-}" ]; then
  AAI_ENV=$(PYTHONPATH="$REPO" PYTHONNOUSERSITE=1 "$PYTHON" -m aa_evals.common.aai_open condor-env)
fi
cat > "$CODE/job.sub" <<EOF
universe = vanilla
executable = /bin/bash
arguments = $CODE/job.sh
initialdir = $CODE
output = $RUN_DIR/logs/job.\$(Cluster).out
error = $RUN_DIR/logs/job.\$(Cluster).err
log = $RUN_DIR/logs/job.condor.log
request_cpus = 1
request_memory = 4096
request_disk = 2G
+JobBatchName = "aa-omniscience-$RUN_ID"
+MaxRuntime = 86400
getenv = false
environment = "USER='user' HOME='$HOME' PATH='/usr/local/bin:/usr/bin:/bin' PYTHONNOUSERSITE='1' PYTHONDONTWRITEBYTECODE='1' http_proxy='http://proxy.internal:8080' https_proxy='http://proxy.internal:8080' HTTP_PROXY='http://proxy.internal:8080' HTTPS_PROXY='http://proxy.internal:8080' no_proxy='$NOPROXY' NO_PROXY='$NOPROXY'$BUDGET_ENV$ROOT_ENV$AAI_ENV"
requirements = $REQ
queue
EOF

if [ -n "${DRY_RUN:-}" ]; then
  echo "dry run: $CODE/job.sub (not submitted)"
  exit 0
elif [ -n "$BID" ]; then
  condor_submit_bid "$BID" "$CODE/job.sub"
else
  PYTHONPATH="$CODE" "$PYTHON" -m aa_evals.common.condor_cost submit "$CODE/job.sub"
fi
echo "run dir: $RUN_DIR  (logs/, items.jsonl, summary.json)"

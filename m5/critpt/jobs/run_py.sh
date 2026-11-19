#!/bin/bash
# Generic condor wrapper: extract commit SHA of the m5-critpt repo into the job's scratch dir (a frozen, detached
# copy that lives only for the job) and run `python -m MODULE ARGS...` there with the m5-critpt venv on /data.
set -euo pipefail
SHA=$1; shift
REPO=${M5_CRITPT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}   # the checkout this script lives in
CODE=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}/code-$SHA
mkdir -p $CODE
git -C $REPO archive $SHA | tar -x -C $CODE
cd $CODE
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} M5_CRITPT_CODE_SHA=$SHA
echo "host $(hostname) start $(date -u) code $SHA"
${M5_CRITPT_PY:-/data/aai/venvs/m5-critpt/bin/python} -u "$@"
echo "end $(date -u)"

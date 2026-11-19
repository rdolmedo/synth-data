#!/bin/bash
# The CritPt sandbox SIF as a harbor_ns base image for the Pi harness (#348; aa_evals/common/pi_eval/sif_base.py).
#   build_ns_base.sh        # a CPU condor job; run from the repo root with PYTHONPATH=.
set -euo pipefail
eval "$(python3 -c '
from aa_evals.critpt.pi.tasks import SIF, BASE_REF, BASE_ENV, WORKDIR
import shlex
print("SIF=" + shlex.quote(str(SIF)), "REF=" + shlex.quote(BASE_REF), "WORKDIR=" + WORKDIR)
print("ENVS=(" + " ".join(shlex.quote("--env=" + k + "=" + v) for k, v in BASE_ENV.items()) + ")")')"
python3 -m aa_evals.common.pi_eval.sif_base "$SIF" "$REF" --workdir "$WORKDIR" "${ENVS[@]}"

#!/bin/bash
# The GDPval sandbox SIF as a harbor_ns base image for the Pi harness (#348; aa_evals/common/pi_eval/sif_base.py).
#   build_ns_base.sh [SIF]        # a CPU condor job (~10 GB node scratch); run from the repo root with PYTHONPATH=.
# The image environment is the SIF's %environment (PATH with /opt/venv first, HOME=/home/user, locale, Playwright),
# plus /home/user/.local/bin first on PATH as the Stirrup path's runtime-cache tasks have it (gdpval/runtime_cache.py).
set -euo pipefail
SIF=${1:-/data/aai/aa_evals/sif/gdpval-v1.sif}
REF=$(python3 -c 'from aa_evals.gdpval.pi.tasks import BASE_REF; print(BASE_REF)')
python3 -m aa_evals.common.pi_eval.sif_base "$SIF" "$REF" --workdir /home/user --path-prefix /home/user/.local/bin

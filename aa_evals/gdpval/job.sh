#!/bin/bash
# Condor entry point: unpack the relocatable GDPval venv (one tar on /data; small-file writes to Lustre are slow)
# into node scratch, then run aa_evals.gdpval.run with the job's arguments (`--module M ...`: python -m M ...).
set -euo pipefail
TAR=${GDPVAL_VENV_TAR:-/data/aai/aa_evals/envs/gdpval-venv.tar}
V=${_CONDOR_SCRATCH_DIR:-/tmp/gdpval_job_$$}/venv
if [ ! -x "$V/bin/python" ]; then
  mkdir -p "$V"
  tar -C "$V" -xf "$TAR"
fi
MOD=aa_evals.gdpval.run
if [ "${1:-}" = "--module" ]; then MOD=$2; shift 2; fi
exec "$V/bin/python" -B -m "$MOD" "$@"

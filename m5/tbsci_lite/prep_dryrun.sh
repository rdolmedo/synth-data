#!/bin/bash
# CPU dry run of the SFT prep (branch sft: sft_qwen35/prepare.py --source pi_native) on an M5 export (#353):
# proves the export is in the format M1's production traces used. Writes to node-local scratch only, then copies
# the small report files (stats.json, report.md, sequences.csv, sample_rendered.txt) next to the export.
#   condor job: prep_dryrun.sh SFT_CHECKOUT EXPORT_DIR
set -euo pipefail
CODE=$1 EXPORT=$2
PY=/opt/aai/venvs/qwen35sft/bin/python
export HOME=/home/user HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
OUT=${_CONDOR_SCRATCH_DIR:-/tmp}/prep
rm -rf "$OUT"   # a dry run: a previous run's output (outside condor, /tmp) is not kept (prepare.py refuses a finished out)
cd "$CODE"
"$PY" sft_qwen35/prepare.py --source pi_native --arms nemotron35-super --inputs /nonexistent --synth-roots "$EXPORT" --out "$OUT" \
    --max-tokens 2e9 --val-groups 0 --exclude-tasks none --workers 4 --model /data/aai/models/qwen3.5-2b
mkdir -p "$EXPORT/prep_dryrun"
for f in stats.json report.md sequences.csv sample_rendered.txt; do [ -f "$OUT/$f" ] && cp "$OUT/$f" "$EXPORT/prep_dryrun/"; done
echo PREP_DRYRUN_DONE

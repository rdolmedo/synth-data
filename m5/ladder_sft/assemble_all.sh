#!/bin/bash
# The nested M5 ladder SFT datasets, smallest first (condor CPU job; assemble.sub). Args: [--exclude=FILE] name=budget ...
set -euo pipefail
cd /data/aai/m5/ladder/sft
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/opt/aai/venvs/qwen35sft/bin/python
EXCL=()
for x in "$@"; do
  case "$x" in
    --exclude=*) EXCL=(--exclude "${x#--exclude=}"); continue ;;   # 11-17: decontamination exclusions (mixture v2, m5l2-*)
  esac
  name=${x%%=*}
  budget=${x#*=}
  # --allow-incomplete: only matters with --keep-m1; by default (11-17) tasks without a full second trace are dropped (M=2)
  $P -u assemble.py --budget "$budget" --name "$name" --allow-incomplete "${EXCL[@]}"
done
echo ASSEMBLE_DONE

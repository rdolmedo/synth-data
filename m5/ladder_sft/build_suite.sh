#!/bin/bash
# records + render pool for one source of the M5 ladder mixture (condor CPU job; build.sub).
set -euo pipefail
SUITE=$1
W=${2:-14}
P=/opt/aai/venvs/qwen35sft/bin/python
cd /data/aai/m5/ladder/sft
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case $SUITE in
  omniscience|gdppdf|hle|scicode) T2=t2_flat; FIN=fin_flat;;
  automationbench|gdpval|critpt|tbsci_lite) T2=t2_pi; FIN=fin_pi;;
  tb) T2=; FIN=fin_pi;;
  *) echo "bad suite $SUITE"; exit 1;;
esac
if [ -n "$T2" ]; then $P -u records.py "$T2" "$SUITE" --workers "$W"; fi
$P -u records.py "$FIN" "$SUITE" --workers "$W"
if [ "$T2" = t2_pi ] && [ -d "/data/aai/m5/ladder/t2/resample/runs/m5l_${SUITE}_t2fin" ]; then $P -u records.py fin_t2 "$SUITE" --workers "$W"; fi
$P -u render_pool.py "$SUITE" --workers "$W"
echo "BUILD_SUITE_DONE $SUITE"

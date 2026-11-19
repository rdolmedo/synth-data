#!/bin/bash
# One-off (2026-11-12): copy m5-critpt's venv, work dir and pi_eval run dirs from /home to /data and verify
# (checksum dry run must list nothing). The home copies are deleted by hand after this job reports VERIFIED.
set -euo pipefail
H=/opt/aai
pairs=("$H/venvs/m5-critpt/ /data/aai/venvs/m5-critpt/"
       "$H/m5work/critpt/ /data/aai/m5/critpt/work/"
       "$H/aa_runs_fallback/m5_critpt/ /data/aai/m5/critpt/runs/m5_critpt/")
echo "host $(hostname) start $(date -u)"
for p in "${pairs[@]}"; do set -- $p; mkdir -p $2; rsync -a $1 $2; echo "copied $1 -> $2"; done
bad=0
for p in "${pairs[@]}"; do
  set -- $p
  n=$(rsync -anci --delete $1 $2 | wc -l)
  echo "verify $2: $n differences"; [ $n -eq 0 ] || bad=1
done
[ $bad -eq 0 ] && echo VERIFIED || { echo VERIFY-FAILED; exit 1; }
du -sh /data/aai/venvs/m5-critpt /data/aai/m5/critpt/work /data/aai/m5/critpt/runs
echo "end $(date -u)"

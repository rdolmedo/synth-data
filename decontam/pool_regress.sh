#!/bin/bash
# Regression of decontam.pool on #19's student-1 pool (CPU job): rebuild its 7,296 tasks from the campaign tars,
# compare the signals with decontam.synth's, count unjudged window pairs under #19's verdicts (expect ~0), and
# rebuild the list from those verdicts (expect #19's 36 tasks). Usage: pool_regress.sh WORK [WORKERS]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:?work dir}; N=${2:-30}
OLD=/data/aai/tbenchtrain/decontam_tasksynth/work
LIST=/data/aai/tbench/tasksynth/decon/variants
C=/data/aai/tbench/tasksynth/campaigns
mkdir -p $W
t0=$(date +%s)
$HERE/pool_job.sh update --work $W --pool $C/r5/tasks.tar $C/s1/tasks.tar $C/s2/tasks.tar --workers $N
echo "update seconds $(( $(date +%s) - t0 ))"
for f in verdicts-fp.jsonl verdicts-nofp.jsonl verdicts-nofp-lite.jsonl; do
    [ -s $W/$f ] || cp $OLD/$f $W/$f.partial && mv $W/$f.partial $W/$f 2>/dev/null || true
done
t0=$(date +%s)
$HERE/pool_job.sh snapshot --work $W --dry-run --save-scores
echo "dry-run seconds $(( $(date +%s) - t0 ))"
$HERE/pool_job.sh final --work $W --out $W/final-check
cd $HERE/.. && ${DECONTAM_PY:-/data/aai/tbenchtrain/decontam_calibforge/venv/bin/python} -m decontam.pool_check \
    --work $W --old $OLD --list $LIST

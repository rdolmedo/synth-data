#!/bin/bash
# Issue #19 stage 2b (CPU job): the no-fingerprint arm on TB-Lite. Calibration on 261 Opus-labelled CF/TB2.1 pairs
# (CF session, 2026-10-31): DeepSeek fingerprints P 0.79 / R 0.77, Opus 0.83 / 0.85, no fingerprint 0.74 / 0.81, union
# DeepSeek-fp OR no-fp R 0.90. TB-Lite has DeepSeek fingerprints only, so its pairs at best rank <= 10 are also judged
# with rubric_reuse_nofp_synth.md (2 samples, 6 more for any non-NO); after stage 2 the list is the union of both arms.
# Publishes it (if <= 5% excluded) and resubmits the retrain driver.
set -euo pipefail
SNAP=$(cd "$(dirname "$0")/.." && pwd)
D=/data/aai/tbenchtrain/decontam_tasksynth
W=$D/work
L=$D/stage2
EP=/data/aai/tbenchtrain/decontam_calibforge/variants/serve3/endpoint.json
FP=$W/fingerprints.jsonl
OFP=$W/verdicts-fp.jsonl
ONOFP=$W/verdicts-nofp.jsonl
OLITE=$W/verdicts-nofp-lite.jsonl
PUB=/data/aai/tbench/tasksynth/decon/variants
export PYTHONNOUSERSITE=1 PYTHONPATH=$SNAP
PY=/data/aai/tbenchtrain/decontam_calibforge/venv/bin/python
cd $SNAP
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/events.log; }
touch $OLITE
judge() {  # judge PAIRS SAMPLES OFFSET TAG
    if [ ! -s "$1" ]; then return 0; fi
    $PY -u -m decontam.judge --work $W --out $OLITE --pairs $1 --signals signals-near.jsonl --samples $2 --sample-offset $3 \
        --effort max --concurrency 256 --ramp 32 --labels REUSE,WEAK_REUSE,NO_REUSE --endpoint $EP \
        --rubric decontam/rubric_reuse_nofp_synth.md --summary $L/$4.summary.json > $L/$4.log 2>&1
    log "$4: $(grep '^{' $L/$4.log | tail -1 | cut -c1-220)"
}
$PY -m decontam.synth nearest --work $W --fingerprints $FP --k 10 --out $L/near.jsonl | tee -a $L/events.log
$PY -m decontam.synth signals --work $W --pairs $L/near.jsonl --out-name signals-near.jsonl | tee -a $L/events.log
judge $L/near.jsonl 2 0 nofplite-p0
$PY -m decontam.variants resample --mode nonno --work $W --verdicts $OLITE --out $L/rs-nofplite.jsonl | tail -1 | tee -a $L/events.log
judge $L/rs-nofplite.jsonl 6 2 nofplite-rs
log "nofp-lite arm done; waiting for stage 2 (job 17804269)"
until grep -qE "PUBLISHED|HOLD" $L/events.log; do sleep 60; done
$PY -m decontam.synth final --work $W --fingerprints $FP --verdicts "$OFP,$ONOFP;$OLITE" --arm-names fingerprint,nofp-tblite \
    --out $L/final-union | tee -a $L/events.log
RATE=$(python3 -c "import json; print(json.load(open('$L/final-union/decontamination-variants.json'))['counts']['exclusion_rate'])")
if python3 -c "import sys; sys.exit(0 if $RATE <= 0.05 else 1)"; then
    mkdir -p $PUB
    cp $L/final-union/decontamination-variants.pairs.jsonl $L/final-union/exclude-tasks.txt $PUB/
    cp $L/final-union/decontamination-variants.json $PUB/.decontamination-variants.json.partial
    mv $PUB/.decontamination-variants.json.partial $PUB/decontamination-variants.json
    log "PUBLISHED-UNION $PUB (exclusion rate $RATE)"
    log "retrain driver: $(condor_submit_bid 35 $D/retrain.sub | tail -1)"
else
    log "HOLD-UNION: exclusion rate $RATE > 5%; list in $L/final-union, not published, retrain not submitted"
fi

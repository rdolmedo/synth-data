#!/bin/bash
# Issue #19 stage 2 (CPU job; judge on the shared DeepSeek server): pairs -> 2 samples -> 6 more for pairs with a
# non-NO label -> top-50 windows after exclusions until none unjudged -> final list. Publishes the list for
# synth_retrain.py only if it excludes <= 5% of the candidates (else HOLD for a human look).
set -euo pipefail
SNAP=$(cd "$(dirname "$0")/.." && pwd)
D=/data/aai/tbenchtrain/decontam_tasksynth
W=$D/work
L=$D/stage2
EP=/data/aai/tbenchtrain/decontam_calibforge/variants/serve3/endpoint.json
FPS="/data/aai/tbenchtrain/decontam_calibforge/variants/fp/fingerprints_v3.jsonl /data/aai/tbenchtrain/decontam_calibforge/variants/fp/fingerprints_tblite_ds.jsonl"
FP=$W/fingerprints.jsonl
OFP=$W/verdicts-fp.jsonl
ONOFP=$W/verdicts-nofp.jsonl
PUB=/data/aai/tbench/tasksynth/decon/variants
export HF_HOME=${_CONDOR_SCRATCH_DIR:-/tmp}/hf PYTHONNOUSERSITE=1 PYTHONPATH=$SNAP
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy=localnet
PY=/data/aai/tbenchtrain/decontam_calibforge/venv/bin/python
cd $SNAP
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/events.log; }
touch $OFP $ONOFP

log "waiting for stage 1, $L/FP_READY and the judge server"
until [ -e $W/STAGE1_DONE ] && [ -e $L/FP_READY ] && [ -s $EP ] && curl -s --noproxy '*' -m 10 \
        "$(python3 -c "import json; print(json.load(open('$EP'))['url'])")/models" | grep -q '"id"'; do sleep 60; done
log "ready: server $(python3 -c "import json; print(json.load(open('$EP'))['url'])")"
cat $FPS > $FP.partial && mv $FP.partial $FP
log "fingerprints: $(wc -l < $FP) ($(sha1sum $FP | cut -c1-12))"

SIG=""
signals() {  # signals PAIRS PAIRS_NOFP TAG
    $PY -m decontam.synth signals --work $W --pairs $1,$2 --out-name signals-$3.jsonl | tail -1 | tee -a $L/events.log
    SIG=${SIG:+$SIG,}signals-$3.jsonl
}
judge() {  # judge PAIRS OUT SAMPLES OFFSET TAG fp|nofp CONCURRENCY
    if [ ! -s "$1" ]; then return 0; fi
    local rubric="--rubric decontam/rubric_reuse_synth.md --fingerprints $FP"
    [ "$6" = nofp ] && rubric="--rubric decontam/rubric_reuse_nofp_synth.md"
    $PY -u -m decontam.judge --work $W --out $2 --pairs $1 --signals "$SIG" --samples $3 --sample-offset $4 \
        --effort max --concurrency $7 --ramp 32 --labels REUSE,WEAK_REUSE,NO_REUSE --endpoint $EP $rubric \
        --summary $L/$5.summary.json > $L/$5.log 2>&1
    log "$5: $(tail -1 $L/$5.log | cut -c1-220)"
}
both() {  # both PAIRS PAIRS_NOFP SAMPLES OFFSET TAG
    judge $1 $OFP $3 $4 fp-$5 fp 512 & local p1=$!
    judge $2 $ONOFP $3 $4 nofp-$5 nofp 128 & local p2=$!
    wait $p1; wait $p2
}
resample() {  # every pair with a non-NO label among its first two samples gets samples 2-7 (8 in all)
    $PY -m decontam.variants resample --mode nonno --work $W --verdicts $OFP --out $L/rs-fp-$1.jsonl | tail -1 | tee -a $L/events.log
    $PY -m decontam.variants resample --mode nonno --work $W --verdicts $ONOFP --out $L/rs-nofp-$1.jsonl | tail -1 | tee -a $L/events.log
    both $L/rs-fp-$1.jsonl $L/rs-nofp-$1.jsonl 6 2 rs-$1
}

$PY -m decontam.synth pairs --work $W --fingerprints $FP --k 50 --out $L/p0.jsonl --out-nofp $L/p0-nofp.jsonl \
    | tee -a $L/events.log
signals $L/p0.jsonl $L/p0-nofp.jsonl p0
both $L/p0.jsonl $L/p0-nofp.jsonl 2 0 p0
resample p0
for it in 1 2 3 4 5; do
    $PY -m decontam.synth window --work $W --fingerprints $FP --verdicts $OFP,$ONOFP --k 50 \
        --out $L/w$it.jsonl --out-nofp $L/w$it-nofp.jsonl | tee -a $L/events.log
    if [ ! -s $L/w$it.jsonl ] && [ ! -s $L/w$it-nofp.jsonl ]; then log "windows complete"; break; fi
    signals $L/w$it.jsonl $L/w$it-nofp.jsonl w$it
    both $L/w$it.jsonl $L/w$it-nofp.jsonl 2 0 w$it
    resample w$it
done
$PY -m decontam.synth final --work $W --fingerprints $FP --verdicts $OFP,$ONOFP --out $L/final | tee -a $L/events.log
RATE=$(python3 -c "import json; print(json.load(open('$L/final/decontamination-variants.json'))['counts']['exclusion_rate'])")
if python3 -c "import sys; sys.exit(0 if $RATE <= 0.05 else 1)"; then
    mkdir -p $PUB
    cp $L/final/decontamination-variants.pairs.jsonl $L/final/exclude-tasks.txt $PUB/
    cp $L/final/decontamination-variants.json $PUB/.decontamination-variants.json.partial
    mv $PUB/.decontamination-variants.json.partial $PUB/decontamination-variants.json
    log "PUBLISHED $PUB (exclusion rate $RATE)"
else
    log "HOLD: exclusion rate $RATE > 5%; list in $L/final, not published"
fi

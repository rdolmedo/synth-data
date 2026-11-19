#!/bin/bash
# final_export.sh OUT_DIR BUDGET SEED RUN_DIR...   (condor job, run via submit.sh-style wrapper or directly)
#   OUT_DIR/full      every kept trace of RUN_DIR... (one per problem; no decontamination: deferred)
#   OUT_DIR/cut       seeded random-by-task subset of full with max_context_tokens summing to <= BUDGET
#   OUT_DIR/prep-cut  stats/report of sft_qwen35/prepare.py (origin/sft) on cut: a dry run, the packed data stays
#                     in the job's scratch
# Code: this checkout's HEAD (git archive), prepare.py from origin/sft of the same checkout. Every shard is gzip-tested
# and listed in SHA256SUMS.
set -euo pipefail
OUT=$1 BUDGET=$2 SEED=$3; shift 3
REPO=${M5_CRITPT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
T=$(mktemp -d -p ${_CONDOR_SCRATCH_DIR:-/tmp} final_export.XXXX)   # fresh per run
trap 'rm -rf "$T"' EXIT
SHA=$(git -C $REPO rev-parse HEAD)
SFT=$(git -C $REPO rev-parse origin/sft)
mkdir -p $T/code $T/sft $T/empty $T/prep $OUT
git -C $REPO archive $SHA | tar -x -C $T/code
git -C $REPO archive $SFT sft_qwen35 gen | tar -x -C $T/sft   # prepare.py imports gen/glm53_pi_export
echo "host $(hostname) start $(date -u) code $SHA sft $SFT"
PY="env PYTHONPATH=$T/code PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /data/aai/venvs/m5-critpt/bin/python -u"
cd $T/code
[ -e $OUT/full/manifest.json ] || $PY -m m5.critpt.export --runs "$@" --out $OUT/full
$PY -m m5.critpt.export --runs "$@" --out $OUT/cut --cut-from $OUT/full/manifest.json --budget-tokens $BUDGET --seed $SEED
for d in full cut; do
  (cd $OUT/$d && for f in attempts/shard-*.jsonl.gz; do gzip -t $f; done && sha256sum manifest.json attempts/index.json attempts/shard-*.jsonl.gz > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c --quiet SHA256SUMS)
  echo "$d: shards gzip ok, SHA256SUMS verified"
done
cd $T/sft
PYTHONNOUSERSITE=1 /opt/aai/venvs/qwen35sft/bin/python -u sft_qwen35/prepare.py \
  --source pi_native --arms nemotron35-super --synth-roots $OUT/cut --inputs $T/empty --out $T/prep --max-tokens 1e12 2>&1 | tail -40
mkdir -p $OUT/prep-cut
for f in stats.json report.md sample_rendered.txt; do [ -e $T/prep/$f ] && cp $T/prep/$f $OUT/prep-cut/ && cmp $T/prep/$f $OUT/prep-cut/$f; done
echo "{\"code\": \"$SHA\", \"sft\": \"$SFT\", \"budget\": $BUDGET, \"seed\": $SEED, \"at\": \"$(date -u +%FT%TZ)\"}" > $OUT/BUILD.json
echo "end $(date -u)"

#!/bin/bash
# One-off: download proof-pile-2 arxiv (train/validation/test) + arXiv metadata (CC0), sha256, build the joined index,
# copy to /data/aai/m5/shared/arxiv/ with dd oflag=direct. Runs as a condor CPU job (not on gpu123).
set -euo pipefail
CODE=$1
DEST=/data/aai/m5/shared/arxiv
PY=/data/aai/venvs/m5-critpt/bin/python
export https_proxy=http://proxy.internal:8080 http_proxy=http://proxy.internal:8080 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
W=/tmp/m5-arxiv-$$; mkdir -p $W; cd $W
trap 'rm -rf $W' EXIT
echo "host $(hostname) start $(date -u)"
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
for i in range(5):
    try:
        snapshot_download('EleutherAI/proof-pile-2', repo_type='dataset', allow_patterns=['arxiv/*/*.jsonl.zst'],
                          local_dir='pp2', max_workers=8)
        snapshot_download('librarian-bots/arxiv-metadata-snapshot', repo_type='dataset',
                          allow_patterns=['data/*.parquet', 'README.md'], local_dir='meta', max_workers=8)
        break
    except Exception as e:
        print('retry', i, e, flush=True)
PYEOF
echo "downloaded $(date -u)"; du -sh pp2 meta
mkdir -p out/proof-pile-2-arxiv out/metadata
mv pp2/arxiv/* out/proof-pile-2-arxiv/
mv meta/data/*.parquet out/metadata/; cp meta/README.md out/metadata/README.hf.md
(cd out && find . -type f \( -name '*.zst' -o -name '*.parquet' \) | sort | xargs -P 8 -n 4 sha256sum | sort -k2 > SHA256SUMS)
mkdir -p $DEST
copy() {  # copy() SRC_ROOT: every file under it to $DEST (dd oflag=direct for big files)
  (cd $1 && find . -type f -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$DEST/$(dirname "$f")"
    if [ $(stat -c%s "$f") -gt 1048576 ]; then dd if="$f" of="$DEST/$f" bs=64M oflag=direct status=none
    else cp "$f" "$DEST/$f"; fi
  done)
}
copy out
(cd $DEST && sha256sum -c --quiet SHA256SUMS) && echo "copied+verified $(date -u)"
mkdir -p idx
$PY $CODE/m5/critpt/arxiv_index.py $DEST idx
echo "indexed $(date -u)"
copy idx
echo "done $(date -u)"

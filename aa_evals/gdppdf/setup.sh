#!/bin/bash
# One-time setup for aa_evals/gdppdf: the venv, the pinned dataset mirror and the pinned Tesseract model.
# Light (downloads ~430 MB through the proxy); the LiteParse OCR prep is a separate condor job:
#   python -m aa_evals.gdppdf.submit prepare
set -euo pipefail
source ~/.bashrc >/dev/null 2>&1 || true
export UV_CACHE_DIR=${UV_CACHE_DIR:-/data/aai/caches/uv}
ENV=/data/aai/aa_evals/envs/gdppdf
DATA=/data/aai/aa_evals/data/gdppdf
TESS_SHA=e12c65a915945e4c28e237a9b52bc4a8f39a0cec

if [ ! -x "$ENV/bin/python" ]; then
  uv venv -q -p 3.11 "$ENV"
fi
VIRTUAL_ENV=$ENV uv pip install -q liteparse==2.5.0 pandas==3.1.0 pyarrow==26.0.0 pillow==13.0.0 httpx==0.28.1 numpy==2.4.6 huggingface_hub==1.7.2

mkdir -p "$DATA"
if [ ! -f "$DATA/hf_90feac14/data.parquet" ]; then
  "$ENV/bin/python" -I -c "
from huggingface_hub import snapshot_download
snapshot_download('tolo6474/GDP.pdf', repo_type='dataset', revision='90feac141ae15abbcb1d725f6282cd83e40ea077',
                  local_dir='$DATA/hf_90feac14', max_workers=4)"
fi
if [ ! -f "$DATA/tessdata_best_${TESS_SHA:0:8}/eng.traineddata" ]; then
  mkdir -p "$DATA/tessdata_best_${TESS_SHA:0:8}"
  curl -sSL -o "$DATA/tessdata_best_${TESS_SHA:0:8}/eng.traineddata" \
    "https://github.com/tesseract-ocr/tessdata_best/raw/$TESS_SHA/eng.traineddata"
fi
echo "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba  $DATA/tessdata_best_${TESS_SHA:0:8}/eng.traineddata" | sha256sum -c -
cd "$(dirname "$0")/../.." && PYTHONPATH=. "$ENV/bin/python" -m aa_evals.gdppdf.data

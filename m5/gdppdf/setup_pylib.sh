#!/bin/bash
# Extra packages for m5/gdppdf on top of the GDP.pdf eval venv (/data/aai/aa_evals/envs/gdppdf: LiteParse 2.5.0,
# pandas 3.1, numpy 2.4.6, httpx), packed as /data/aai/m5/gdppdf/code/pylib-v1.tar.gz; condor jobs unpack it into
# scratch (m5/gdppdf/condor.py run wrapper). A venv or a --target dir directly on /data took >1 h of small-file
# writes (and once corrupted a .so). Build in a scratch dir, then pack:
set -euo pipefail
T=${1:-$(mktemp -d)}
https_proxy=http://proxy.internal:8080 UV_CACHE_DIR=$T/uvc ~/.local/bin/uv pip install -q \
  --python /data/aai/aa_evals/envs/gdppdf/bin/python --target $T/pylib zstandard scikit-learn tokenizers
tar -czf /data/aai/m5/gdppdf/code/pylib-v1.tar.gz.tmp -C $T/pylib . && \
  mv /data/aai/m5/gdppdf/code/pylib-v1.tar.gz.tmp /data/aai/m5/gdppdf/code/pylib-v1.tar.gz

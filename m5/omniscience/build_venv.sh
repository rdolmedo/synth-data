#!/bin/bash
# Put the m5-omniscience venv on /data (CPU job). uv cannot lock on this Lustre mount, so the venv built on home
# (uv, python 3.11, packages listed below) is copied; it is used only via `python -m`, so script shebangs don't matter.
#   packages: bm25s PyStemmer pyarrow numpy scipy scikit-learn zstandard tokenizers httpx orjson torch(cpu)
#             sentence-transformers transformers jinja2
set -euo pipefail
SRC=/opt/aai/m5-venvs/omniscience
V=/data/aai/venvs/m5-omniscience
mkdir -p /data/aai/venvs
rm -rf "$V.tmp"; cp -a "$SRC" "$V.tmp"; rm -rf "$V"; mv "$V.tmp" "$V"
# bge-small lives as a plain directory in /data/aai/models/bge-small-en-v1.5 (hub downloads hang on Lustre SoftFileLock)
"$V/bin/python" -c "import bm25s, Stemmer, sklearn, pyarrow, zstandard, sentence_transformers, transformers, torch; print('venv ok', torch.__version__)"

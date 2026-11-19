#!/bin/bash
# Build the SciCode grading sandboxes: Python 3.11 + the scientific stack of the two SciCode-Verified grading envs
# (a 2024-era and a 2025-era numpy/scipy; a step passes if it passes in either, the Verified paper's OR rule).
# Runs as a CPU condor job (submit_build.sub). Built unprivileged: a writable sandbox dir on node-local /tmp, then
# squashed into one SIF on /data. Afterwards it checks the runtime isolation the grader uses.
#   build_sif.sh <out_dir>        -> <out_dir>/scicode_np126.sif, <out_dir>/scicode_np22.sif (+ .freeze.txt, .sha256)
set -euo pipefail
OUT=${1:?usage: build_sif.sh <out_dir>}
BASE=${BASE:-docker://python:3.11-slim}
declare -A PKGS=(
  [np126]="numpy==1.26.4 scipy==1.13.1 sympy==1.12 mpmath==1.4.1 matplotlib==3.8.4 h5py==3.16.0"
  [np22]="numpy==2.2.6 scipy==1.15.2 sympy==1.14.0 mpmath==1.3.0 matplotlib==3.11.2 h5py==3.16.0"
)
source ~/.bashrc >/dev/null 2>&1 || true
# the image pull and pip need the cluster proxy; condor jobs (getenv = false) do not inherit it
export https_proxy=${https_proxy:-http://proxy.internal:8080} http_proxy=${http_proxy:-http://proxy.internal:8080}
export HTTPS_PROXY=$https_proxy HTTP_PROXY=$http_proxy
export APPTAINER_CACHEDIR=/data/aai/apptainer-cache
mkdir -p "$OUT"
for tag in np126 np22; do
  W=$(mktemp -d /tmp/scicode_sif.XXXXXX)
  apptainer build --sandbox "$W/root" "$BASE"
  apptainer exec --writable --no-home "$W/root" \
    env PIP_NO_CACHE_DIR=1 HTTPS_PROXY="${https_proxy:-}" HTTP_PROXY="${http_proxy:-}" \
    python -m pip install --no-warn-script-location ${PKGS[$tag]}
  apptainer exec --no-home "$W/root" python -m pip freeze > "$W/freeze.txt"
  apptainer exec --no-home "$W/root" python -c "import sys; print(sys.version)" >> "$W/freeze.txt"
  mkdir -p "$W/root/work" "$W/root/opt/scicode_vendor" "$W/root/data"
  apptainer build "$W/out.sif" "$W/root"
  dst="$OUT/scicode_$tag.sif"
  dd if="$W/out.sif" of="$dst.partial" bs=64M oflag=direct status=none
  mv "$dst.partial" "$dst"
  cp "$W/freeze.txt" "${dst%.sif}.freeze.txt"
  sha256sum "$dst" | tee "${dst%.sif}.sha256"
  chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"
done

echo "== isolation checks"
T=$(mktemp -d /tmp/scicode_iso.XXXXXX)
probe='import socket,os
try:
    socket.create_connection(("pypi.org",443),timeout=5); print("NET: reachable (BAD)")
except Exception as e: print("NET: blocked", type(e).__name__)
print("ifaces:", os.listdir("/sys/class/net"), "fast visible:", os.path.exists("/data/aai"), "home:", os.path.exists("/home"))
import numpy, scipy, sympy, h5py, matplotlib; print("pkgs ok", numpy.__version__, scipy.__version__, sympy.__version__)'
for tag in np126 np22; do
  apptainer exec --containall --cleanenv --no-home --net --network none -B "$T:/work" --pwd /work "$OUT/scicode_$tag.sif" python -c "$probe" || echo "$tag failed rc=$?"
done
rm -rf "$T"

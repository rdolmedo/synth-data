#!/bin/bash
# Build the CritPt code sandbox: Python 3.13 + numpy, scipy, sympy, matplotlib (AA's package list), as one SIF.
# Runs as a CPU condor job (submit_build.sub). Built unprivileged: a writable sandbox dir on node-local /tmp
# (owned by us, so pip can write into it), then squashed into a single file on /data.
# Afterwards it checks the runtime isolation the agent uses (no network, only /work bound).
set -euo pipefail
OUT=${1:?usage: build_sif.sh <out.sif>}
BASE=${BASE:-docker://python:3.13-slim}
PKGS=${PKGS:-"numpy==2.5.4 scipy sympy==1.14.0 mpmath matplotlib"}
source ~/.bashrc >/dev/null 2>&1 || true            # proxy for the image pull and pip
export APPTAINER_CACHEDIR=/data/aai/apptainer-cache
W=$(mktemp -d /tmp/critpt_sif.XXXXXX)
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"' EXIT
apptainer build --sandbox "$W/root" "$BASE"
# pip into the writable sandbox; no cache, no bytecode left behind in odd places
apptainer exec --writable --no-home "$W/root" \
  env PIP_NO_CACHE_DIR=1 HTTPS_PROXY="${https_proxy:-}" HTTP_PROXY="${http_proxy:-}" \
  python -m pip install --no-warn-script-location $PKGS
apptainer exec --no-home "$W/root" python -m pip freeze > "$W/freeze.txt"
apptainer exec --no-home "$W/root" python -c "import sys; print(sys.version)" >> "$W/freeze.txt"
mkdir -p "$W/root/work"
apptainer build "$W/out.sif" "$W/root"
dd if="$W/out.sif" of="$OUT.partial" bs=64M oflag=direct status=none
mv "$OUT.partial" "$OUT"
cp "$W/freeze.txt" "${OUT%.sif}.freeze.txt"
sha256sum "$OUT" | tee "${OUT%.sif}.sha256"
cat "${OUT%.sif}.freeze.txt"

echo "== isolation checks"
T=$(mktemp -d /tmp/critpt_iso.XXXXXX)
probe='import socket,os,sys
try:
    socket.create_connection(("pypi.org",443),timeout=5); print("NET: reachable (BAD)")
except Exception as e: print("NET: blocked", type(e).__name__)
print("ifaces:", os.listdir("/sys/class/net"))
print("fast visible:", os.path.exists("/data/aai"), "home:", os.path.exists("/home"))
import numpy, scipy, sympy, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.plot([1,2]); plt.savefig("t.png"); print("pkgs ok", numpy.__version__, scipy.__version__, sympy.__version__, matplotlib.__version__, os.getcwd(), os.listdir("."))'
echo "-- A: apptainer --containall --net --network none"
apptainer exec --containall --no-home --net --network none -B "$T:/work" --pwd /work "$OUT" python -c "$probe" || echo "A failed rc=$?"
echo "-- B: apptainer --containall + unshare -rn inside"
apptainer exec --containall --no-home -B "$T:/work" --pwd /work "$OUT" unshare -rn python -c "$probe" || echo "B failed rc=$?"
echo "-- C: unshare -rn outside apptainer"
unshare -rn apptainer exec --containall --no-home -B "$T:/work" --pwd /work "$OUT" python -c "$probe" || echo "C failed rc=$?"
rm -rf "$T"

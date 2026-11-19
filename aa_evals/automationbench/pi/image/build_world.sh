#!/bin/bash
# Build the AutomationBench images (world.def: the world service; agent.def: the agent's sandbox) as SIFs.
# sif_base.py then puts them into harbor_ns's base cache. A CPU condor job.
#   build_world.sh OUT_DIR
set -euo pipefail
OUTDIR=${1:?usage: build_world.sh OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=/data/aai/aa_evals/data/automationbench/repo_4a8e106/automationbench
export APPTAINER_CACHEDIR=${_CONDOR_SCRATCH_DIR:-/tmp}/apptainer-cache
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 HTTPS_PROXY=http://proxy.internal:8080
W=$(mktemp -d "${_CONDOR_SCRATCH_DIR:-/tmp}/abworld.XXXXXX")
export APPTAINER_TMPDIR=$W/tmp; mkdir -p "$APPTAINER_TMPDIR"
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"' EXIT
mkdir -p "$W/ctx/automationbench"
cp "$SRC/__init__.py" "$SRC/tool_wrapper.py" "$W/ctx/automationbench/"
for d in schema tools utils; do cp -r "$SRC/$d" "$W/ctx/automationbench/"; done
find "$W/ctx" -name __pycache__ -prune -exec rm -rf {} +
cp "$HERE/world.def" "$HERE/agent.def" "$HERE/world_server.py" "$W/"
mkdir -p "$OUTDIR"
for img in world agent; do
  (cd "$W" && apptainer build --fakeroot "$W/$img.sif" $img.def)
  cp "$W/$img.sif" "$OUTDIR/$img.sif.partial" && mv "$OUTDIR/$img.sif.partial" "$OUTDIR/$img.sif"
  sha256sum "$OUTDIR/$img.sif" | tee "$OUTDIR/$img.sha256"
done
apptainer exec "$OUTDIR/world.sif" cat /opt/abworld/pip_freeze.txt > "$OUTDIR/world.pip_freeze.txt"

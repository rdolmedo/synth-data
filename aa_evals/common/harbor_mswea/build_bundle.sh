#!/bin/bash
# Build the pinned mini-swe-agent bundle that agent.py uploads into every sandbox (run as a CPU condor job:
# `python -m aa_evals.common.harbor_mswea.bundle submit`).
#
#   build_bundle.sh [VERSION] [CACHE_ROOT]
#
# Result: CACHE_ROOT/mini-swe-agent/<VERSION>/linux-x64/aa-mswea.tar (+ .sha256, freeze.txt), one tar holding
#   /opt/aa-mswea/            a relocatable uv-managed CPython 3.12 with stock mini-swe-agent==VERSION from PyPI
#                             and its dependencies (constraints.txt) installed into its own site-packages
#   /opt/aa-mswea/mini-swe-agent   launcher: that Python, -m minisweagent.run.mini (the `mini-swe-agent` CLI)
# Nothing outside /opt/aa-mswea is touched in the sandbox: the task's own python3/PATH stay as the image has them.
set -euo pipefail
VERSION=${1:-2.4.6}
CACHE=${2:-/data/aai/aa_evals/agent_cache}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(readlink -f "$(mktemp -d "${_CONDOR_SCRATCH_DIR:-/tmp}/aa-mswea.XXXXXX")")
trap 'rm -rf "$W"' EXIT
export UV_PYTHON_INSTALL_DIR=$W/pythons UV_PYTHON_BIN_DIR=$W/pybin   # nothing lands in ~/.local/bin
# node-local uv cache: uv locks its cache with flock, which hangs on Lustre (the /data cache stalled a build)
export UV_CACHE_DIR=$W/uv-cache
uv python install 3.12
PYBIN=$(uv python find --managed-python 3.12)
PYROOT=$(dirname "$(dirname "$(readlink -f "$PYBIN")")")
case "$PYROOT" in "$W"/*) ;; *) echo "unexpected python root $PYROOT" >&2; exit 2;; esac
P=$W/bundle/opt/aa-mswea
mkdir -p "$P"
cp -a "$PYROOT/." "$P/"
rm -f "$P"/lib/python3.12/EXTERNALLY-MANAGED
uv pip install --python "$P/bin/python3.12" --no-progress \
  -c "$HERE/constraints.txt" "mini-swe-agent==$VERSION"
cat > "$P/mini-swe-agent" <<'EOS'
#!/bin/sh
export MSWEA_SILENT_STARTUP=1 PYTHONNOUSERSITE=1
exec /opt/aa-mswea/bin/python3.12 -m minisweagent.run.mini "$@"
EOS
chmod 755 "$P/mini-swe-agent"
got=$(MSWEA_SILENT_STARTUP=1 "$P/bin/python3.12" -c 'import minisweagent, litellm; print(minisweagent.__version__)')
[ "$got" = "$VERSION" ] || { echo "bundle has mini-swe-agent $got, want $VERSION" >&2; exit 2; }
# the stock CLI must parse its options and load the builtin mini config
MSWEA_SILENT_STARTUP=1 MSWEA_CONFIGURED=true "$P/bin/python3.12" -m minisweagent.run.mini --help >/dev/null
uv pip freeze --python "$P/bin/python3.12" > "$P/freeze.txt"
printf '%s\n' "$VERSION" > "$P/VERSION"
find "$P" -name '__pycache__' -prune -exec rm -rf {} +
tar -cf "$W/aa-mswea.tar" -C "$W/bundle" opt
D=$CACHE/mini-swe-agent/$VERSION/linux-x64
mkdir -p "$D"
dd if="$W/aa-mswea.tar" of="$D/aa-mswea.tar.partial" bs=64M oflag=direct conv=fsync status=none
mv "$D/aa-mswea.tar.partial" "$D/aa-mswea.tar"
sha256sum "$D/aa-mswea.tar" | cut -d' ' -f1 > "$D/aa-mswea.tar.sha256"
cp "$P/freeze.txt" "$D/freeze.txt"
echo "bundle ready: $D/aa-mswea.tar $(cat "$D/aa-mswea.tar.sha256") $(du -h "$D/aa-mswea.tar" | cut -f1)"

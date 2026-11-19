#!/usr/bin/env bash
# Fetch and pin upstream AutomationBench, then build its venv (Python 3.13, uv.lock frozen).
# Idempotent. Light enough for the login/tmux host (~30 s, one git clone + wheel installs).
#   bash aa_evals/automationbench/setup.sh
set -euo pipefail
COMMIT=4a8e1061254004d9dac807054eed33fad7d1ff14          # v1.0.6 release + README (2026-08-04)
REPO_URL=https://github.com/zapier/AutomationBench.git
SRC=${AA_AUTOMATIONBENCH_SRC:-/data/aai/aa_evals/data/automationbench/repo_${COMMIT:0:7}}
VENV=${AA_AUTOMATIONBENCH_VENV:-/opt/aai/venvs/automationbench}
PY313=${PY313:-$(uv python find 3.13)}

if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone -q "$REPO_URL" "$SRC"
fi
git -C "$SRC" checkout -q "$COMMIT"
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SRC" status --porcelain)" || { echo "pinned source $SRC is modified" >&2; exit 1; }

# uv needs file locks: /data (Lustre) has none, so the cache stays node-local and the venv sits
# on NFS next to the repo's other venvs. --no-editable copies the package into the venv.
export UV_CACHE_DIR=/tmp/$USER/uvcache UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VENV"
(cd "$SRC" && uv sync --frozen --no-dev --no-editable --python "$PY313")
echo "$COMMIT" > "$VENV/AUTOMATIONBENCH_COMMIT"
"$VENV/bin/python" -c "import automationbench, verifiers; from importlib.metadata import version as v; print('automation-bench', v('automation-bench'), 'verifiers', v('verifiers'))"

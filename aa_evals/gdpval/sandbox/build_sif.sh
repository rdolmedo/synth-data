#!/bin/bash
# Build the GDPval-AA sandbox SIF from gdpval.def as a CPU condor job (submit with build_sif.sub).
# Node-local build dir, then dd to /data with an atomic rename. Afterwards: a smoke check of the toolchain.
set -euo pipefail
OUT=${1:?usage: build_sif.sh <out.sif>}
HERE=$(cd "$(dirname "$0")" && pwd)
export APPTAINER_CACHEDIR=/data/aai/apptainer-cache
export http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 HTTP_PROXY=http://proxy.internal:8080 HTTPS_PROXY=http://proxy.internal:8080
W=$(mktemp -d "${_CONDOR_SCRATCH_DIR:-/tmp}/gdpval_sif.XXXXXX")
export APPTAINER_TMPDIR=$W/tmp; mkdir -p "$APPTAINER_TMPDIR"
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"' EXIT
cp "$HERE/gdpval.def" "$HERE/aa_apt_packages.txt" "$HERE/aa_pip_packages.txt" "$W/"
cd "$W"
apptainer build --fakeroot "$W/out.sif" gdpval.def
dd if="$W/out.sif" of="$OUT.partial" bs=64M status=none
mv "$OUT.partial" "$OUT"
apptainer exec "$OUT" cat /opt/gdpval/missing.txt > "${OUT%.sif}.missing.txt"
apptainer exec "$OUT" cat /opt/gdpval/versions_apt.txt > "${OUT%.sif}.versions_apt.txt"
apptainer exec "$OUT" cat /opt/gdpval/versions_pip.txt > "${OUT%.sif}.versions_pip.txt"
sha256sum "$OUT" | tee "${OUT%.sif}.sha256"
echo "== missing ($(wc -l < "${OUT%.sif}.missing.txt"))"; cat "${OUT%.sif}.missing.txt"

echo "== smoke"
T=$(mktemp -d "$W/smoke.XXXXXX")
apptainer exec --containall --cleanenv -B "$T:/home/user" --pwd /home/user "$OUT" bash -c '
set -x
python --version; which python pip
python -c "import numpy, pandas, docx, pptx, openpyxl, fitz, reportlab, matplotlib; print(\"py ok\")"
python -c "import docx; d=docx.Document(); d.add_paragraph(\"hi\"); d.save(\"t.docx\")"
soffice --headless --convert-to pdf t.docx >/dev/null 2>&1; ls -la t.pdf
pdflatex --version | head -1; pandoc --version | head -1; tesseract --version 2>&1 | head -1
ffmpeg -version | head -1; convert -version | head -1; java -version 2>&1 | head -1; chromium --version
'

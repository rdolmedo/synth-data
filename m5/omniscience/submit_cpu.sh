#!/bin/bash
# Submit a CPU job for m5-omniscience from a frozen detached worktree of the current HEAD (commit first).
#   m5/omniscience/submit_cpu.sh NAME CPUS MEM_MB DISK_GB -- command args...   (command runs from the frozen repo root)
set -euo pipefail
NAME=$1; CPUS=$2; MEM=$3; DISK=$4; shift 4; [ "$1" = "--" ] && shift
REPO=$(git rev-parse --show-toplevel)
SHA=$(git -C "$REPO" rev-parse --short=10 HEAD)
[ -z "$(git -C "$REPO" status --porcelain -- m5/omniscience)" ] || { echo "commit m5/omniscience first" >&2; exit 2; }
# frozen code lives on /data (home is a small NFS quota); the job extracts it itself (no bulk writes from gpu123)
FROZEN=/data/aai/m5/omniscience/code/$SHA
GITDIR=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
LOGS=/data/aai/m5/omniscience/logs/$NAME
mkdir -p "$LOGS"
PY=${PY:-/data/aai/venvs/m5-omniscience/bin/python}
JOB=$LOGS/job.sh
{ echo '#!/bin/bash'; echo "set -e; if [ ! -f $FROZEN/.done ]; then mkdir -p $FROZEN; git --git-dir=$GITDIR archive $SHA m5 aa_evals | tar -x -C $FROZEN; echo $SHA > $FROZEN/FROZEN_SHA; touch $FROZEN/.done; fi; set +e"; echo "cd $FROZEN"; echo "export PYTHONPATH=$FROZEN PYTHONNOUSERSITE=1 PY=$PY"; printf '%q ' "$@"; echo; } > "$JOB"
chmod +x "$JOB"
BAD="cpu203 cpu205 cpu207 cpu208 gpu110 gpu132 gpu123 gpu143 gpu166 gpu172 gpu106 cpu305 ${EXTRA_BAD:-} $(grep -v '^#' /opt/aai/m5-coord/excluded-nodes.txt 2>/dev/null | awk 'NF{print $1}' | tr '\n' ' ')"
REQ=$(for n in $BAD; do printf '(TARGET.UtsnameNodename != "%s") && ' "$n"; done | sed 's/ && $//')
NOPROXY='localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'
cat > "$LOGS/job.sub" <<SUB
universe = vanilla
executable = /bin/bash
arguments = $JOB
initialdir = $LOGS
output = $LOGS/job.\$(Cluster).out
error = $LOGS/job.\$(Cluster).err
log = $LOGS/job.condor.log
request_cpus = $CPUS
request_memory = $MEM
request_disk = ${DISK}G
+JobBatchName = "m5-omni-$NAME"
+MaxRuntime = 172800
getenv = false
environment = "USER='user' HOME='$HOME' PATH='/usr/local/bin:/usr/bin:/bin' PYTHONNOUSERSITE='1' HF_HUB_OFFLINE='1' HF_HOME='/data/aai/hf_cache' http_proxy='http://proxy.internal:8080' https_proxy='http://proxy.internal:8080' HTTP_PROXY='http://proxy.internal:8080' HTTPS_PROXY='http://proxy.internal:8080' no_proxy='$NOPROXY' NO_PROXY='$NOPROXY'"
requirements = $REQ
queue
SUB
condor_submit_bid ${BID:-36} "$LOGS/job.sub"
echo "logs: $LOGS  frozen: $FROZEN"

#!/bin/bash
# submit.sh NAME CPUS MEM_GB DISK_GB -- python args...   (the job runs HEAD's commit, extracted into its scratch dir)
# M5_CRITPT_PY=<python> picks another venv (decon: the decon venv, which has sentence-transformers)
set -euo pipefail
NAME=$1 CPUS=$2 MEM=$3 DISK=$4; shift 5
REPO=${M5_CRITPT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}   # the checkout this script lives in
C=$(git -C $REPO rev-parse --short=12 HEAD)
L=/data/aai/m5/critpt/logs
mkdir -p $L
REQ='(TARGET.UtsnameNodename != "gpu103") && (TARGET.UtsnameNodename != "cpu203") && (TARGET.UtsnameNodename != "cpu205") && (TARGET.UtsnameNodename != "cpu207") && (TARGET.UtsnameNodename != "cpu208") && (TARGET.UtsnameNodename != "gpu110") && (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "gpu123") && (TARGET.UtsnameNodename != "gpu143") && (TARGET.UtsnameNodename != "gpu166") && (TARGET.UtsnameNodename != "gpu172") && (TARGET.UtsnameNodename != "gpu106") && (TARGET.UtsnameNodename != "cpu305")'
# the coordinator's shared exclusions, read at submit time
for n in $(sed 's/#.*//' /opt/aai/m5-coord/excluded-nodes.txt 2>/dev/null); do
  REQ="$REQ && (TARGET.UtsnameNodename != \"$n\")"
done
ARGS="$*"
cat > $L/$NAME.sub <<SUB
universe = vanilla
executable = /bin/bash
arguments = "$REPO/m5/critpt/jobs/run_py.sh $C $ARGS"
request_cpus = $CPUS
request_memory = $((MEM*1024))
request_disk = ${DISK}G
+BypassLXCfs = "true"
+JobBatchName = "m5-critpt-$NAME"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin OMP_NUM_THREADS=$CPUS M5_CRITPT_LAUNCH_REPO=$REPO M5_CRITPT_PY=${M5_CRITPT_PY:-/data/aai/venvs/m5-critpt/bin/python}"
requirements = $REQ
rank = (TARGET.CUDACapability =!= 10.0)
output = $L/$NAME.\$(Cluster).out
error = $L/$NAME.\$(Cluster).err
log = $L/$NAME.log
queue
SUB
condor_submit_bid 36 $L/$NAME.sub

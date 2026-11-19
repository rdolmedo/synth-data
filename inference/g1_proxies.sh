#!/bin/bash
# Put capping/holding proxies in front of DP group 1: per-rank proxies on the old rank ports
# (8101-8104, used directly by older dispatch workers) and the group proxy on 8001, all
# forwarding to the ranks' new ports 8111-8114.  Waits until the old ports are free.
PY=/opt/aai/venvs/harbor/bin/python
P=/opt/aai/repo/inference/sticky_proxy.py
L=/data/aai/tbench/logs
CAP=${CAP:-132000}
free() { ! ss -ltnH "sport = :$1" | grep -q LISTEN; }
until free 8101 && free 8102 && free 8103 && free 8104 && free 8001; do sleep 0.2; done
echo "$(date +%T.%N) ports free, starting proxies"
for i in 1 2 3 4; do
  $PY $P --port 810$i --backends http://localhost:811$i --cap-tokens $CAP --cap-log $L/cap.log >> $L/rank_proxy_810$i.log 2>&1 &
done
$PY $P --port 8001 --backends http://localhost:8111,http://localhost:8112,http://localhost:8113,http://localhost:8114 \
  --cap-tokens $CAP --cap-log $L/cap.log --pins-file $L/pins_8001.json >> $L/sticky_proxy_8001.log 2>&1 &
wait

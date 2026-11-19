#!/bin/bash
# Issue #25 v2-arm driver (HTCondor CPU job). Student 1's recipe throughout, only the spec vocabulary differs:
#   Super campaign (stream, no budget) -> trace judge (DeepSeek, incremental) -> task_defect exclusions
#   -> SFT prep (student 1's snapshot; seeded random subset capped at student 1's 321,688,447 train tokens, 5M val)
#   -> train (Qwen3.5-4B, lr 6e-6, 3 epochs, bf16 --fp8 none, seed 0, 4xB200) -> export checks
#   -> eval request tbmixp180-<run> (ops loop) -> irt_mix compare vs student 1.
# State: $D/events.log (one line per step), $D/state/<step>.done markers make a resubmitted driver resume.
# v4 (driver4.sh): 64 GB job (the judge holds every prepared trace in memory), final passes repeat until <= 0.5%
# unjudged. v3: judge concurrency 1024 (192 left gen-ds idle: ~15 min per judge call). v2: trace-judge passes run in the background; the final pass tolerates <= 0.5% runaway judge calls.
set -uo pipefail
C=20261031-nemotron35-super-tasksynth-v2
CAMP=/data/aai/tbench/synth/$C
STREAM=/data/aai/tbench/tasksynth/streams/v2
TRACES=/data/aai/tbench/tasksynth/traces/$C
TS=/opt/aai/worktrees/tasksynth-run-22f74a98ea      # frozen task_synthesis code (scale rounds)
PY=/opt/aai/venvs/harbor/bin/python
SNAP=/data/aai/tbenchtrain/sft_q35_code/2604353786e4bcdc4ab85765acf22dd0a5bcc012  # student 1's SFT snapshot
EVAL=/opt/aai/worktrees/eval-super-scaling/eval_q35
DS=tasksynth-v2-330m
RUN=tasksynth-4b-v2-330m-3ep
BASE_RUN=tbmixp180-tasksynth-4b-ds-330m-3ep
DATA=/data/aai/tbenchtrain/sft_data_q35/$DS
CKPT=/data/aai/tbenchtrain/sft_q35_ckpts/$RUN
D=/data/aai/tbench/tasksynth/vocab/v2/ab
mkdir -p $D/state
export PYTHONNOUSERSITE=1 http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 \
       no_proxy=localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $D/events.log; }
done_() { [ -e $D/state/$1.done ]; }
mark() { date -u +%FT%TZ > $D/state/$1.done; }
fail() { log "FAILED: $*"; exit 1; }
prog() { python3 -c "import json;p=json.load(open('$CAMP/ops/progress.json'));print(p.get('$1'))" 2>/dev/null; }
judge() {   # one resumable pass; runs in the background, pid in $D/judge.pid
  (cd $TS && PYTHONPATH=$TS exec $PY -m task_synthesis.pipeline.traces run --campaign $CAMP --bundle $STREAM \
     --judge gen-ds --concurrency 1024 >> $D/trace-judge.log 2>&1) &
  echo $! > $D/judge.pid
}
judging() { [ -s $D/judge.pid ] && kill -0 $(cat $D/judge.pid) 2>/dev/null; }
judged() { python3 -c "import json;print(sum(1 for l in open('$TRACES/records.jsonl') if json.loads(l).get('verdict')))" 2>/dev/null || echo 0; }
jobstate() {  # running|idle|held|gone
  local s; s=$(condor_q $1 -af JobStatus 2>/dev/null)
  case "$s" in 1) echo idle;; 2) echo running;; 5) echo held;; "") echo gone;; *) echo "s$s";; esac
}
log "driver start on $(hostname)"

# 1. campaign, judging finished traces every hour while it runs
if ! done_ campaign; then
  last=0; noted=""
  while [ "$(prog phase)" != complete ]; do
    if [ "$(prog needs_attention)" = True ] && [ -z "$noted" ]; then
      log "campaign needs attention: $(prog controller_error)"; noted=1
    fi
    if [ $(( $(date +%s) - last )) -gt 3600 ] && [ "$(prog train_finalized)" != 0 ] && ! judging; then
      log "campaign $(prog phase): saved $(prog train_tokens_saved) tok, $(prog train_finalized) trajectories, $(judged) judged; trace judge pass"
      judge; last=$(date +%s)
    fi
    sleep 300
  done
  log "campaign complete: saved $(prog train_tokens_saved) tok, $(prog train_finalized) trajectories"
  mark campaign
fi

# 2. final trace-judge pass and the task_defect exclusion list (student 1's rule; all outcomes kept)
if ! done_ exclusions; then
  # final pass: wait for it, but a few runaway judge replies (no output cap, 1 h client timeout, retries) must not hold
  # the arm: once <= 0.5% of finished attempts are unjudged and the count has not moved for 30 min, go on without them
  if judging; then kill $(cat $D/judge.pid); sleep 10; fi   # an incremental pass may sit on a runaway call
  total=$(prog train_finalized)
  for pass in 1 2 3 4; do   # repeat passes until <= 0.5% of finished attempts are unjudged (a pass can die early)
    judge; prev=-1; still=0
    while judging; do
      n=$(judged)
      if [ "$n" = "$prev" ]; then still=$((still + 1)); else still=0; prev=$n; fi
      if [ $still -ge 30 ] && python3 -c "import sys; sys.exit(0 if $total - $n <= 0.005 * $total else 1)"; then
        log "final judge pass: $n of $total judged, no progress for 30 min; stopping it"; kill $(cat $D/judge.pid); break
      fi
      sleep 60
    done
    n=$(judged)
    python3 -c "import sys; sys.exit(0 if $total - $n <= 0.005 * $total else 1)" && break
    log "final judge pass $pass ended with $n of $total judged; another pass"
  done
  python3 - $TRACES/records.jsonl $D/exclude-tasks.txt <<'PY' | tee -a $D/events.log || fail "exclusions"
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
judged = [r for r in recs if r.get('verdict')]
bad = sorted({r['task'] for r in judged if r['verdict'].get('task_defect')})
open(sys.argv[2], 'w').write(''.join(t + '\n' for t in bad))
print(f'trace judge: {len(recs)} attempts, {len(judged)} judged, {len(bad)} tasks with a task_defect excluded '
      f'({len({r["task"] for r in recs})} tasks)')
PY
  mark exclusions
fi

# 3. SFT prep (student 1's arguments; --max-tokens = student 1's train tokens)
if ! done_ prep; then
  if [ ! -s $DATA/stats.json ]; then
    mkdir -p $D/no-submit-inputs
    [ -z "$(ls -A $D/no-submit-inputs)" ] || fail "no-submit-inputs is not empty"
    out=$(condor_submit_bid 61 -a CODE=$SNAP -a NAME=$DS -a OUT=$DATA -a request_memory=262144 -a request_disk=120G \
      -a "ARGS=--source pi_native --arms nemotron35-super --inputs $D/no-submit-inputs --synth-roots $CAMP/train --workers 30 --load-workers 28 --max-tokens 321688447 --val-tokens 5e6 --exclude-tasks $D/exclude-tasks.txt" \
      -append 'requirements = (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "gpu110") && (TARGET.UtsnameNodename != "cpu105") && (TARGET.UtsnameNodename != "gpu180") && (TARGET.UtsnameNodename != "cpu301")' \
      $SNAP/sft_qwen35/prep.sub 2>&1) || fail "prep submit: $out"
    job=$(echo "$out" | grep -o 'cluster [0-9]*' | grep -o '[0-9]*'); log "prep job $job -> $DATA"
    while [ "$(jobstate $job)" != gone ]; do [ "$(jobstate $job)" = held ] && fail "prep job $job held"; sleep 120; done
  fi
  [ -s $DATA/stats.json ] || fail "prep left no stats.json"
  log "prep: $(python3 -c "import json;s=json.load(open('$DATA/stats.json'));p=s.get('packs',{});print({'train_tokens':p.get('train',{}).get('tokens'),'train_packs':p.get('train',{}).get('n_packs'),'val_tokens':p.get('val',{}).get('tokens')})" 2>&1)"
  mark prep
fi

# 4. train (student 1's flags)
if ! done_ train; then
  if [ ! -s $CKPT/sft_meta.json ]; then
    out=$(condor_submit_bid 63 -a NAME=$RUN -a CC=10.0 -a NGPU=4 -a CODE=$SNAP -a MODEL=/data/aai/models/qwen3.5-4b -a RUNTIME=43200 \
      -a "ARGS=sft_qwen35/train.py --model @MODEL@ --data $DATA --out /data/aai/tbenchtrain/sft_q35_logs/runs/$RUN --lr 6e-6 --accum 1 --balance-window 32 --epochs 3 --max-tokens 1e10 --val-every 96 --val-limit 0 --fp8 none --export-dir @SCR@/export --export-to $CKPT" \
      -append 'requirements = (TARGET.CUDACapability == 10.0) && (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "cpu207") && (TARGET.UtsnameNodename != "cpu307") && (TARGET.UtsnameNodename != "cpu403") && (TARGET.UtsnameNodename != "cpu405") && (TARGET.UtsnameNodename != "cpu203") && (TARGET.UtsnameNodename != "cpu205") && (TARGET.UtsnameNodename != "cpu208")' \
      $SNAP/sft_qwen35/q35.sub 2>&1) || fail "train submit: $out"
    job=$(echo "$out" | grep -o 'cluster [0-9]*' | grep -o '[0-9]*'); log "train job $job -> $CKPT"
    while [ "$(jobstate $job)" != gone ]; do [ "$(jobstate $job)" = held ] && fail "train job $job held"; sleep 300; done
  fi
  [ -s $CKPT/sft_meta.json ] || fail "train left no sft_meta.json (see sft_q35_logs/$RUN.*.err)"
  log "train done: $(tail -c 600 /data/aai/tbenchtrain/sft_q35_logs/runs/$RUN/train_log.jsonl | tr '\n' ' ' | cut -c1-400)"
  mark train
fi

# 5. export checks (RUNBOOK step 4)
if ! done_ export_check; then
  python3 - $CKPT $DATA <<'PY' | tee -a $D/events.log || fail "export check"
import hashlib, json, sys
ck, data = sys.argv[1:]
sha = hashlib.sha256(open(f'{ck}/chat_template.jinja', 'rb').read()).hexdigest()
want = json.load(open(f'{data}/stats.json'))['template_sha256']
mt = json.load(open(f'{ck}/config.json')).get('model_type')
assert sha == want, f'chat template {sha} != data {want}'
assert mt == 'qwen3_5', f'model_type {mt}'
print(f'export ok: template {sha[:12]}, model_type {mt}')
PY
  mark export_check
fi

# 6. eval request (RUNBOOK step 5): append atomically, then verify it is there
if ! done_ eval_request; then
  python3 - $RUN $CKPT $DATA <<'PY' | tee -a $D/events.log || fail "eval request"
import json, os, shutil, sys, time
run, ck, data = sys.argv[1:]
Q = '/data/aai/tbenchtrain/eval_q35/requests.json'
name = f'tbmixp180-{run}'
entry = {'name': name, 'model': ck, 'family': 'qwen3_5', 'chat_template': f'{ck}/chat_template.jinja',
         'benchmark': 'tbmix-irt100', 'harness': 'pi180', 'attempts_per_task': 5,
         'teacher_arm': 'nemotron35-super (tasksynth v2-vocabulary DeepSeek tasks rv2a-rv2i, 1 traj/task, 3 epochs)',
         'data': data, 'note': '#25 v2 arm: student 1 recipe (322M Super Pi tokens x 3 epochs, bf16, lr 6e-6) on tasks '
         'from the broadened spec vocabulary (vocab_v2.json); compare with tbmixp180-tasksynth-4b-ds-330m-3ep',
         'requested_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
for attempt in range(5):
    q = json.load(open(Q))
    if any(r['name'] == name for r in q['requests']):
        print(f'eval request {name} present'); break
    shutil.copy(Q, Q + f'.bak-{run}')
    q['requests'].append(entry)
    q['closed'] = False
    tmp = Q + f'.tmp-{os.getpid()}'
    json.dump(q, open(tmp, 'w'), indent=1)
    os.replace(tmp, Q)
    time.sleep(20)
else:
    sys.exit('eval request kept disappearing')
PY
  mark eval_request
fi

# 7. wait for the score, then the paired comparisons
S=/data/aai/tbenchtrain/eval_q35/tbmixp180-$RUN/scores.json
while [ ! -s $S ]; do sleep 900; done
log "scores: $(python3 -c "import json;s=json.load(open('$S'));i=s.get('irt',{});print({'pass_at_1':s.get('pass_at_1'),'projTB2.1':i.get('projected_tb2.1'),'projTBLite':i.get('projected_tblite')})" 2>&1 | cut -c1-300)"
(cd $EVAL && python3 -B irt_mix.py compare $BASE_RUN tbmixp180-$RUN > $D/compare-vs-student1.txt 2>&1;
 python3 -B irt_mix.py compare tb21p180-base-4b+tblp180-base-4b tbmixp180-$RUN > $D/compare-vs-base.txt 2>&1)
log "compare vs student 1: $(tail -5 $D/compare-vs-student1.txt | tr '\n' ' ' | cut -c1-400)"
log "DONE"

# aa_evals/common/dispatch: one dispatcher per model run (#344)

A model's suites share **one in-flight budget** on its server, sized to the server's measured knee, and start the
**longest predicted attempts first** across suites. Every attempt has a **2 h wall cap** and is graded on what it
produced. The #344 replay (30% plan, 8 models) estimates 106 → ~50 generation GPU-h: the shared queue −37%, the cap
another −16%.

Everything is opt-in. Without `--dispatch` and without the cap flags, every runner behaves exactly as before, so
M3.5/M3.6 runs stay reproducible.

## Use

```
DIR=/data/aai/aa_evals/dispatch/<model-run>
python -m aa_evals.common.dispatch.dispatcher init $DIR --budget 132 --model qwen35-4b --expect-lanes 9
nohup python -m aa_evals.common.dispatch.dispatcher run $DIR > $DIR/dispatcher.out 2>&1 &     # light; any node
# each suite, with its usual flags plus:
python -m aa_evals.tb21.run ... --dispatch $DIR --agent-wall-cap-sec 7200 --max-in-flight 132   # pi_eval suites
python -m aa_evals.hle.submit --run-id X -- ... --dispatch $DIR --request-cap-sec 7200
bash aa_evals/omniscience/submit.sh X ... --dispatch $DIR --timeout 7200
python -m aa_evals.gdppdf.submit run --run-id X -- ... --dispatch $DIR --request-cap-sec 7200
python -m aa_evals.scicode.run ... --dispatch $DIR --chain-cap-sec 7200 --condor
python -m aa_evals.common.dispatch.dispatcher status $DIR     # budget, in flight by suite, queue head
python -m aa_evals.common.dispatch.dispatcher budget $DIR 120 # live fixed budget (turns the auto budget off)
python -m aa_evals.common.dispatch.dispatcher budget $DIR auto # back to the auto budget
python -m aa_evals.common.dispatch.dispatcher report $DIR --out attempts.jsonl   # per-attempt start/end/pred
```

- `--budget`: attempts (conversations) in flight on the model's server. M3.6 knees: ~66 per GPU for Qwen3.8,
  120–180 for the small models, 247 for GLM on 4 GPUs.
- `--model`: the duration-table key of the model (`qwen38-xhigh`, `qwen38-medium`, `super-vl`, `qwen36`, `gemma26`,
  `lightning`, `qwen35-4b`, `glm53-flash`). The model's own stored times are blended with the other models' times.
  Omit it for a new model (other models only).
- `--expect-lanes N`: no slot is granted until N lanes are registered and ready, at most `--ready-timeout` (30 min).
  Otherwise the first suite to come up would take the whole budget with its own longest items.
- `--metrics http://engine:port`: the budget also backs off on preemptions or waiting requests at high KV (the fullest
  engine's), like `pi_eval/adaptive.py`, and recovers by 2 per calm tick. Do not run `adaptive.py` on the same runs.
- **Auto budget (#378, `autobudget.py`, on by default for new dirs with `--metrics` (campaigns launched by `aa_evals/campaign/launch.py` pass `--no-auto-budget` unless they set `auto_budget: true`, #531 B7); `--no-auto-budget`; dirs made
  before it have it off):** `--budget` is the start value. Every 10 min the dispatcher judges the budget on the
  engine's generated tok/s (`vllm:generation_tokens_total`), measured after 3 min of settling, and moves it within
  `--budget-min`/`--budget-max` (default budget/2 .. 2×budget):
  - +10% while the last raise gained ≥ 3% tok/s (an elasticity of ~0.3: past it each attempt slows almost in
    proportion) and the engine is healthy (KV p90 < 85%, requests waiting in < 10% of samples, no preemption);
  - −10% when KV p90 ≥ 90% with requests waiting in ≥ 10% of samples, −15% on a preemption (c30 Super above ~128
    running); a raise that gained < 3% is reverted. A failed level is not tried again for 30 min.
  - **Limits:** it holds when the budget was not filled (ramp-up, drain tail, too few Pi worker slots, gate holds:
    so it never rises past what the lanes can fill), at `--budget-max` (logged), and when the engine is at
    max-num-seqs (mean running ≥ 95% of `--max-num-seqs` × engines, read from the registry's `launch.json` when not
    given) or requests wait in ≥ 30% of samples, or the gate holds ≥ 50% of the time; a raise that hit one of these
    is reverted.
  - The per-tick back-off above and the gate back-off still work under it. `budget DIR N` fixes the budget (auto
    off); `budget DIR auto` resumes. Its level survives a restart (`DIR/auto_budget.json`); `state.json` and
    `status` show it, `log.txt` every change with the window's numbers.
  - Offline check (`test_autobudget.py`): servers simulated from c30's knee curves, workload drift and noise as
    measured; from their c30 budgets it settles at Super ~136 (running ~120; c30 182), Gemma ~215 (121), Qwen3.6
    ~167 (130, then held by max-num-seqs 128), Qwen3.5-4B ~218 (132), Lightning ~153 (139); simulated tok/s +27%,
    +26%, +14%, +16%, +3% over the fixed budgets. Above c30's highest running the curves are extrapolated (the
    last elasticity, decaying), so the gains of Gemma and 4B are a model, not a measurement.
- **pi_eval workers:** a pi lane gets at most its live workers × slots. Slots that a lane cannot use go to the other
  lanes.
- **Lazy first workers** (#378; default for new dispatch dirs, `--no-lazy-workers` to opt out): the run registers with no worker job. The
  dispatcher counts it ready for the start-up release and reserves its first fleet's slots (`--workers` x slots,
  default launch-small) in longest-first order, as eager first workers did: the granted credit is held unspent while
  the workers start, and the first grant submits them. With only a probe slot (#511 as merged) the single-turn lanes
  took the whole budget at launch (val-sb10b 11-18: SciCode 251/256, TB2.1/GDPval/AB 1 each). Once its first
  submission is older than `ready_timeout` (the start-up grace eager lanes had), its credit is capped at its starts +
  its live workers' free slots, so workers that never start, or a fleet that starts in part, hold nothing (read from
  files: a restarted dispatcher applies it too). Before a submission is recorded (it keeps failing) the reservation
  goes back after `reserve_sec`, and the lane keeps only its probe slot (`LAZY_PROBE_SLOTS`) until it is recorded. After that, launch-small and autoscale as usual.
- **Worker growth (#378, `--pi-autoscale`, on by default for new dirs; dirs made before it have it off):** when a pi
  lane's slots, not the budget, limit it, the dispatcher submits more workers for it. `decide()` reports a
  **shortfall**: what the lane would also take, in the same longest-first order, from the slots left free. Workers are
  added after 4 ticks (1 min) of shortfall, at most one batch per 10 min, sized to in flight + unspent credit +
  shortfall minus the slots of the lane's queued or running worker jobs, and never beyond in flight + pending work.
  They use the run's stored shape and bid (pi_eval `submit_workers`, so run the dispatcher with the harbor venv). The
  condor calls run in a background thread. A lane grows only after its own first worker submission is in
  `RUN/jobs.json` (not for `--no-submit` runs), never past the budget, and a `condor_q` error means no growth.
  Workers leave by themselves 5 min (15 before #531) after their lane's queue is fully claimed, so pi_eval launches a lane of a
  growing dispatcher with only 25% of the budget in worker slots (`LAUNCH_SHARE`; `--workers` overrides). c30's lanes
  got 60% each: 5 lanes left about 3× the budget in idle slots. A lane with live workers also counts
  what it would hold of the **whole budget** in longest-first order if it had room, minus what it holds: with the
  budget full, unlimited single-turn lanes absorb every freed slot, and only this view lets the longest Pi lane grow
  (val-sb10b 11-18: TB2.1 stayed at its first fleet, 64 of 256, behind 900 Omniscience items). Other lanes'
  in-flight Pi sessions count as held (they run for hours); in-flight single-turn items as freeing. Once a lane's first
  workers heartbeat, this can grow it past `LAUNCH_SHARE` within `SCALE_TICKS`, up to its longest-first share. A worker
  that never claimed does not idle-exit before 3 idle-exit periods after its start (pi_eval).
- **Claim packing (#531, pi_eval run config `pack_claims`, on for runs made since; off under `--sched-state`):** when
  a run's claimable work (unclaimed attempts and retries, at most the unspent start credit and the admission room)
  fits in the free slots of fuller live workers, a worker leaves it to them (`pi_eval/claim_pack.py`; bounded: after
  60 s with no progress it claims anyway). Scarce work then fills a few workers, and the emptied ones leave through the
  idle exit below or, once the queue is fully claimed, eval_q35's no-claimable exit (5 min for pi_eval workers since
  #531, was 15). Only which worker claims changes: the queue order and the seeds do not depend on the worker. M5
  (11-15..18): 40-52% of paid Pi slot-hours ran nothing. Packing puts a run's stragglers on few workers, and since
  #540 a lane whose workers are all gone while claims are still open is revived (its lost claims count as work): a
  bare `condor_rm` of a packed run's workers brings new ones back. To abandon a run on purpose, use `run.py --stop`
  (STOP_WORKERS) or the lane's `NO_HOLD` file first.
- **Idle exit (#378, `--pi-idle-exit-sec 600`, new dirs with `--pi-autoscale`; older dirs have none):** a worker
  with nothing running, free to claim (endpoint up, gate clear, no `STOP_WORKERS`), work pending and no start credit
  left for 10 min exits, and writes `exited: idle-no-credit` in its heartbeat file (`RUN/workers/<w>.json`). It
  decides under the run's claim lock (no claim of its own half made, still no credit) and claims nothing after. Not
  before the lane's first credit, not for `--no-submit` runs (no `RUN/jobs.json`), not while a lost worker's claim
  (no heartbeat for 3 min) waits for its reaper, not while it hosts the turn gate for attempts still running. The
  dispatcher counts an exited worker as no slots. A lane whose last worker left that way and has nothing in flight
  is **waiting, not dead**: its stale `admission.json` is ignored, credit granted as it left is withdrawn, and its
  shortfall is what it would win of the next slots to free (a tenth of the budget while slots are being granted)
  among the lanes that can take credit, in global longest-first order, so it grows back while it is still the head
  instead of after the shorter lanes drain (a grown worker takes a few minutes to start). A lane that lost its
  workers any other way is dead after 10 min without a heartbeat, as before; if some of its workers had left on
  their own (`exited: idle-no-credit`, or `no-claimable`: eval_q35's exit with nothing to claim, the usual one under
  claim packing) and work is left, including the lost workers' open claims (reaped into retries by the new worker),
  it still grows back when it has the credit. To abandon a pi lane, use `run.py --stop` (`STOP_WORKERS`) or its
  `NO_HOLD`: either makes it not scalable (removing its workers alone looks like an eviction).
- **Server release (#378, `init --release-server-after 600`, off by default):** once nothing has been pending or in
  flight on the server for that long, the dispatcher removes the model's serving job (`--registry`'s current job.json
  job, only if it is ours and its batch name carries the registry name; one attempt). Idle means: every expected lane
  registered and announced, none with a pending or in-flight attempt, unspent credit or `hold_server` (`Lane.hold_server`:
  the client will send more model work after CPU work in progress), at least one attempt started, and every engine of the
  registry running and queueing 0 requests (an engine that does not answer counts as busy). A lane need not be `done`: a
  SciCode step hung in CPU grading kept 2 B200s idle ~1 h (#531 B1). A dead lane with work left blocks the release and
  is logged once as STUCK (#531 B2); a lane stopped on purpose (`STOP_WORKERS`) does not, and an operator lets any lane
  go with an empty `DIR/lanes/<lane>/NO_HOLD` file. c30's servers ran 7-21 min past their last result (3.1 GPU-h).
  Never set it for a shared registry.

## How it works

| Piece | What it does |
|---|---|
| Credits | Each lane's allowance is a cumulative **start credit**: a lane may start attempts while its starts so far < its credit. When an attempt finishes, its slot goes back to the dispatcher, not to its lane, and the next tick gives it to the longest pending attempt of any suite. Credits never shrink. A budget cut or `--metrics` backoff stops new credit until the work in flight drains below the new budget. Attempts in flight are never cut. |
| Lane (pi_eval) | One lane per run, named `--dispatch-lane` (default `<bench>.<run-id>`; a name registered by another run is refused), suite `--dispatch-suite` (default `--bench` without `_pi`). `run.py --dispatch DIR` writes `queue/items.txt` longest predicted first, which is eval_q35's claim order (retries first, then file order). It also writes `RUN/control.json` `max_starts 0` and registers `DIR/lanes/<lane>/`. Workers check `max_starts` against the run's claims so far, retries included, under a mkdir lock before every claim (`pilane.credit_claim`). eval_q35's in-flight limit is left open. Burst pacing, serving-gate pause, stale-claim reaper and turn gate are unchanged. Capacity = slots of workers with a heartbeat under 3 min; a paused admission or `STOP_WORKERS` gives 0. A run with no worker heartbeat for 10 min is dead: it counts as 0 and its credit is cut back to its claims so far. |
| Lane (single-turn) | `lane.Lane`, through `open_dispatch()`. One lane per run: `DIR/lanes/<suite>.<run dir name>/`. A lane still owned by a live client is refused. A resumed client keeps counting from the credit it already used. Waiting attempts start longest prediction first, only while starts + weight ≤ the credit (`control.json` `max_starts`, polled every 5 s). If attempts wait with no credit update for 10 min, the client logs a warning. Each poll also writes `lane.json` (in flight, starts, credit, pending with weights) and appends to `events.jsonl`. One slot is one HLE or Omniscience request, one SciCode chain while it generates, or one GDP.pdf repeat; a task takes `weight` = its repeats. |
| Dispatcher tick (15 s) | used = in flight + unspent credit, per lane. Unspent credit counts as in use until the lane spends it, so a lane that reads its control file late (Pi: once per claim round) cannot push the total over the budget. free = budget − used. The pending attempts of all lanes, after those the unspent credit already covers, are merged by table prediction × the suite's online factor. Free slots go to that merged queue strictly in order, by weight and at most each lane's capacity, and are written as new credit. A single-turn lane whose `lane.json` is 10 min stale is counted as 0 and logged; a re-registration replaces it. The dispatcher runs until `DIR/STOP`, so lanes can register or resume at any time. One dispatcher per DIR (`dispatcher.lock`); a restarted one keeps `released` and the credits. |
| Gated servers (round 2) | A serving admission gate (registry `gate_status_urls`; `--registry`, default: the first pi lane's `serving:<name>`) holds new conversations when its expected-KV budget is full. pi_eval workers then claim nothing, so the work-conserving round-1 policy handed every slot to the single-turn lanes, whose requests kept the gate holding: the long Pi head ran last. In the c30 Qwen3.8 Xhigh run, SciCode had 127–129 chains in flight and Pi had 1–3, with KV at 0.37. Now: (1) **gate back-off**: no new credit while the gate holds ≥ `--gate-hold-min` (1) new conversations, so it drains; (2) **reservation**: a Pi lane paused only by the gate keeps its workers' slots as capacity, so it takes credit in global longest-first order and holds it unspent until its workers resume (≤ 30 s); (3) **bound**: Pi credit unspent for `--reserve-sec` (600 s) while the gate is clear is withdrawn. `--gate-backoff off` and servers without `/gate/status` behave exactly as before. The policy is `dispatcher.decide()`, a pure function; `test_gate_sim.py` replays the loop: Pi's first start moves from 2.5 h to 1 min, and the makespan from 6.5 h to 5.5 h for 2 h Pi attempts. |
| Gated servers (round 3, #378) | Two loops left after round 2, both seen in c30 Qwen3.8 Xhigh (2 replicas, Pi lanes paused ~50 min). (1) **Per-replica Pi pause** (pi_eval worker): a worker paused while ANY replica's gate held, although each attempt is pinned to one replica; one SciCode flood on replica 2 paused every Pi lane while replica 1 had room. Now the run pauses (`admission.json` `paused`, i.e. "gate-paused" here) only while EVERY replica's gate holds; until then claims go on in longest-first order, and a new attempt whose hashed replica holds runs on a clear replica for its whole life (`queue/claims/<key>/replica.json`). `admission.json` also lists `gate_held_replicas`; the lane's state shows `gate_partial`. (2) **Single-turn hold** (`decide()`, `--single-turn-hold on`, the default; absent in older dirs = on; needs the back-off): during the back-off, a single-turn client at its own concurrency cap still started new requests from credit granted before the hold, and these kept the gate queue full. Now, while the gate holds (no new credit for anyone) and a Pi lane with pending work is gate-paused, single-turn unspent credit is withdrawn at once (control.json `exact`: the client takes the lower value; clients that publish lane.json `shrinkable`, i.e. this code; older clients keep theirs until restarted). Requests already sent are never recalled. Mixed worker code in one run (old and new pi_eval workers writing the same `admission.json`) flips the pause; workers the dispatcher grows use the run's recorded code, so only a manual resubmit from a newer checkout mixes them. `test_gate_sim.py` (Xhigh-like start: 84 SciCode chains in flight, 46 credits unspent, replica rooms 50/30, 2 h Pi): the 120th Pi start moves from 3.1 h to 0.7 h, Pi lanes are paused 6% of the time instead of 60%, and the makespan drops from 6.8 h to 5.6 h. One replica (fix 2 alone): the 120th Pi start moves from 0.95 h to 0.5 h, with the same makespan. |
| Online factor | Per suite: the geometric mean of actual / table seconds over this run's finished non-infra attempts, plus in-flight ones already past their prediction. It is shrunk toward the all-suite factor (k = 5). It changes how suites interleave, never the order within a suite. |
| Duration table | `durations_v2.json` (`DEFAULT_TABLE`, #378), built by `build_durations.py v2` from c30's finished non-infra attempts (Pi harness, no compaction, text-only, 2 h cap; 23,185 full-run attempts of 9 models. Dropped: the 11-11 memory-hold and worker-kill outcomes, endpoint moves, killed single-turn clients, other infra, and canaries, which ran on an idle server at 0.2–0.5× the full run's time). Per suite, a censored lognormal fit: an attempt cut at the cap counts as "longer than", so a model-item whose attempts all hit the cap is estimated past it (Gemma CritPt: 80 capped). A value is the model's latent median on the item. `predict` uses the model's own value when it has one (`own_weight` 1.0), otherwise the geometric mean of the other models. It returns E[min(T, cap)] under the suite's fitted spread, times the suite's `mean_factor` (Omniscience 1.13: a heavier tail than lognormal; the others 0.97–1.04), and never exceeds the cap. Single-turn values exclude the server-gate wait the clients record: Omniscience's held-queue time (`X-Gate-Queue-Ms`), and HLE's 503 back-off. HLE's held-queue time is not recorded and stays in; it was ≤ 12% of Omniscience's time. GDP.pdf and SciCode record none. Fallback: items c30 did not run (HLE 843, tbsci_lite 103, AutomationBench 154, ...), and GLM's unfinished model-items, take exp(α + β·log v1), with α and β fitted on the model's items shared by v1 and c30 in that suite. One scale factor won't do: v1's error shrinks as items get longer. v1's ≤ 1 s zero-token placeholders are skipped (8 GDP.pdf items lose their row and get the suite median). A suite c30 lacks keeps v1. `ds-max` is a model key (alias `ds-max-m35`). `durations_v1.json` (#344 replay, M3.6 + compaction/Stirrup) stays; a dispatch dir keeps the table path it recorded. Validation: `/data/aai/aa_evals/audits/durations_v2_20261113/validate.md`. For a re-run model (own row from its even repeats, predicting its odd repeats), v2 is better on all 9 models. Per-attempt log RMSE is 0.39–0.73 vs 0.58–0.98 for v1; with each predictor's bias removed (centered mean abs) it is 0.27–0.50 vs 0.37–0.77. Actual/predicted per model is 0.91–1.15 vs 0.96–1.83. For a model in no table (leave-one-model-out), v2 corrects most of v1's under-prediction: median actual/predicted over the 9 models goes from 1.61 to 1.11. The cross-suite error improves only a little: SD of log suite ratio 0.33–0.91 with v1, 0.29–0.79 with v2. A new model's speed per suite is its own, and the online factor learns it. Unknown item: the suite median. The keys are each runner's item ids, so ported suites must keep them as pi task names. |
| Observability | `DIR/state.json` (each tick), `DIR/events.jsonl` (start/end of every attempt with its prediction at start and the table prediction), `DIR/log.txt`, `status` and `report`. |

## 2 h caps (graded on output)

| Suite | Flag | At the cap |
|---|---|---|
| Pi suites | `--agent-wall-cap-sec 7200` (counts time a turn waits at a serving gate: agent wall time) | Harbor's agent timeout = the cap, measured on the agent run only (not env build, agent setup or verifier). Harbor records `AgentTimeoutError` and **still runs the verifier**, so `classify` scores the reward (end state `wall_clock_backstop`). It is not `n_cutoff`. The environment clock stays `--agent-cap-sec`. Recorded in config and protocol as `agent_wall_cap_sec`. |
| SciCode | `--chain-cap-sec 7200` | The cap counts the chain's generation time, kept across resumes (`chain_gen_sec` in the chain file); downtime between resumes is not counted. Steps not generated within it are model errors (`chain_cap`, score 0, kept on resume); a request in flight at the cap is closed. Earlier steps are graded. Recorded as `generation.chain_cap_sec`. |
| HLE, GDP.pdf | `--request-cap-sec 7200` | The model call, retries included, is bounded. At the cap the item has no answer: a model error (`request_cap`), scored 0 and kept on resume. |
| Omniscience | `--timeout 7200` | Already a model outcome (`timeout_is_model`). Its default is 3600, which is stricter. |

## Hung Pi sessions (#531 A5)

A hung Pi session used to sit until the 2 h cap, holding a worker slot and its suite's drain. In M5 on 11-15 this hit
362 GDPval and 53 AutomationBench second traces. The cause: the response-budget guard failed to account a reply (most
likely a 30 s /detokenize timeout) and from then on answered every request with a sticky 502. The sticky proxy
retries a 502 for 30 min, so Pi only timed out and resent, every 1800 s. pi_eval runs made since #531 A5 carry a
watchdog (run config `hung_watchdog`, `pi_eval/hung.py`; `run.py --hung-watchdog-sec`, 0 = the whole watchdog off;
`--hung-stall-sec`, 0 = the stall rule off).
The worker ends an attempt as infrastructure failure `hung_session`, a same-seed retry capped by `INFRA_RETRIES`
(failures in an outage window are exempt from the cap), on any of three rules:

- **Budget accounting failure:** the guard logged `accounting_error`. Its failure is sticky, so the session is ended
  at once, recorded as `response_budget_accounting` (the reason the worker also gives when Pi reports the guard's
  error itself: since #549 the guard retries `/detokenize` once and the sticky proxy passes its 502 straight to Pi).
- **Stalled request** (`stall_sec`, 3660 s, two Pi timeouts and a minute): no progress since the oldest unanswered
  request, where progress is a reply arriving or the guard completing a turn. The guard serves requests one at a time
  and finishes a request Pi gave up on, so a resend waits behind it; slow small-model sessions lost two Pi timeouts
  and still recovered. A request counts from when the proxy forwards it, after its own turn and KV gates; time at a
  server-side admission gate counts. This rule is the backstop for stalls with other causes.
- **Idle** (`idle_sec = max(3 x the suite's p99 idle gap, 600 s)`): Pi itself has done nothing, meaning no request in
  flight (gate waits count as in flight) and no tool call outstanding. Every suite's p99 is under a second, so every
  suite gets 600 s. Long tool calls are the model's time, left to the environment clock. Counting them as idle would
  have ended 19 healthy eval attempts, 13 of 808 in TB2.1, the same long-command tasks every time.
- Both limits come from `pi_eval/request_gaps.json`, measured by `request_gaps.py` on 15 M5 eval runs and the 16 t2
  runs (16,104 attempts), with the live rules' semantics. The accounting rule catches all 490 t2 hangs and 3 eval
  ones, and no session was ever answered after an accounting error. The stall rule catches 488 of them at ~61 min
  and ends no healthy attempt. The idle rule ends none that a gate wait does not explain (one TB-Sci-lite attempt has
  no clock record to check).
- The watchdog judges only the agent's run: the idle rule is armed after the first request, and every rule is
  disarmed at the agent's end, so environment build, setup and the verifier are never cut. It never fires on missing
  evidence (no proxy clock or session log for the idle rule, no wire log for the stall rule). Outage time the worker
  sees counts toward neither timed rule (the endpoint and registry signals refresh every 30 s; the cluster-proxy
  signal only when the worker has a free slot). The idle rule never fires after the agent's final answer.
- Recorded per run (`config.json` `hung_watchdog`) and per attempt (`attempt.json` `hung_watchdog_sec`,
  `hung_stall_sec`). Outcomes carry `hung_kind` (`budget_accounting_failure`, `stalled_request` or `idle`),
  `hung_sec` and `hung_limit_sec` (null for the accounting rule, which has no limit). Runs
  made before it, and the oracle, run without.

## Files

- `dispatcher.py`: `init` / `run` / `status` / `report` / `budget`.
- `autobudget.py`: the auto budget controller (pure) and the vLLM `/metrics` parser.
- `lane.py`: the single-turn lane and `open_dispatch()`.
- `pilane.py`: pi_eval claim order, hold and registration. The table suite is `--dispatch-suite` (default: `--bench` without a trailing `_pi`); more than 10% of tasks without a prediction warns at run creation, in `log.txt` and in `status`.
- `durations.py` / `durations_v2.json` (default) / `durations_v1.json` / `build_durations.py` (`v1`, `extract-c30`, `v2`): predictions.
- `test_dispatch.py`, `test_clients.py`, `test_autoscale.py` (growth, idle exit), `test_gate_sim.py`,
  `test_autobudget.py` (the auto budget against simulated c30 servers), `test_build_durations.py`.

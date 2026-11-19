# Qwen3.8-27B FP8 serving: throughput study and admission gate (#70)

The model under test in the AA evals (`serving:aa-qwen`, spec `synth/teachers/qwen38-27b-eval.toml`) and a judge
candidate. Goal: the most generated tok/s per GPU at maximum concurrency on 1×H100, a matched 1×B200 point, and
an admission gate so eval clients need no hand-set concurrency caps.

## Method

* **Load, by replay** (`synth.tune` → `synth.bench`, a fresh engine per arm, the same recorded requests per arm,
  fixed agent levels measured over a window after a warm-up; server Prometheus counters give gen tok/s):
  * **Agentic:** Qwen3.8-27B's own Pi cont16k traffic, campaign `20261015T215325Z` (train + eval attempts,
    reasoning xhigh, 16,384 output cap, 132k window; NVFP4 on B200 at the time). Median prompt 51k tokens (p90
    103k), mean output 3.4k (11% capped), median tool gap 0.4 s. Read by `synth.bench.index_api`
    (`raw/<trial>/pi-cont16k/api/<start ns>-<id>/{request.json,response-0.json}`). Closed loop: each agent's history
    carries its own answers as Pi builds them for Qwen (`history = "reasoning_field"`).
  * **QA:** `qa_corpus.py`, HLE-Diamond single-turn prompts as `aa_evals/hle` sends them (by-type system prompt,
    images inline), half at reasoning `medium`, half `xhigh`, sampling as the registry.
* Campaigns: `synth/campaigns/qwen38-27b-eval-tuning-{a,b,q,x,g}.toml`; results under
  `/opt/aai/qwenserve-runs/<id>/ops/tuning/{results.jsonl,state.json,plan.json}`
  (/data was over its byte hard limit on 2026-11-02, so these small artifacts live on /home; the QA corpus
  holds eval data and stays there, never in git).
* Numerics gate: `state_check.py` (port of `inference/glm53_state_check.py`): first-token top-20 logprobs on the
  longest recorded contexts, fresh and as a prefix hit; cross-engine differences judged against each engine's own
  hit-vs-fresh noise.

### Deviations from production load

| | replay | production evals |
|---|---|---|
| agentic requests | recorded Pi turns of a 10-15 data-generation campaign (TB-like seed tasks, xhigh) | TB 2.1 / TB-Sci / AutomationBench via Pi, mostly `medium` tonight |
| QA output cap | 32,768 tokens (a window reaches steady state in minutes) | the whole context |
| idle agents | recorded tool gaps (capped 300 s); no setup/verification time | live agents also idle in setup and verification |
| mix | one workload per arm | all benchmarks share one server |

## Admission gate (`gate.py`)

A reverse proxy in front of each vLLM engine; clients use the gate's URL (`endpoints.json` `urls`; the engines are
`engine_urls`) with a high in-flight cap of their own.

* Transparent: all paths/methods forwarded, SSE relayed chunk by chunk, upstream status kept. Only POST
  `/v1/chat/completions` and `/v1/completions` are gated.
* Conversations are keyed by their first two messages (as `inference/sticky_proxy.py`). Multi-turn ones (tools or
  more than two messages) stay live (engine, turn priority) while idle up to `--idle-ttl` (300 s); their context counts
  as cached KV for the first `idle_count_seconds` (120 s) of it (#378).
* **Footprints that track reality** (#378, c30-gemma26: 112 "live" conversations, 38M idle tokens against ~4.6M real,
  KV ~1%, room 0 for ~15 min after worker kills): a conversation whose client leaves mid-turn (a killed attempt) is
  dropped at once; a turn without usage records its prompt estimated from its text plus what streamed (was request
  bytes / 3: 324k "tokens" per cut CritPt turn), capped at the engine's max_model_len; an under-used engine (in-flight
  KV < 0.3, nothing waiting, no brake) admits at least `underused_room_min` (2) new conversations per tick.
  `gate_replay.py` replays a gate log (`--log`) against the engine's stat lines through any gate.py version; c30
  replay: `/data/aai/aa_evals/audits/optim_gate_20261113/report.md`.
* New conversations are admitted by `synth.admission.step` (the data-generation pipeline's kv_fit): working set =
  in-flight KV + idle conversations' contexts, against a target fraction of the KV (scaled by a CPU tier) that is
  hill-climbed on gen tok/s, with preemptions, in-flight KV ≥ 0.92, vLLM's waiting queue and a prefix-reuse collapse
  as brakes. Turns of live conversations go first and skip the budget. Everything is capped at `max_num_seqs`.
* Queue time is never model time: nothing reaches vLLM before admission; each response carries
  `X-Gate-Queue-Ms`; `/metrics` appends `gate:queue_seconds` (histogram by class `cont`/`new`), `gate:inflight`,
  `gate:queued`, `gate:target`, `gate:room`, `gate:live_conversations`, `gate:requests_total` to vLLM's metrics;
  `/gate/status` shows the controller state; `--log` writes one JSON line per request.
* Shared servers (2026-11-03 production issues, fixed in the second gate PR; details in the `gate.py` docstring):
  * **Fairness between clients**: a client is the `X-Gate-Client` header, else the source host. Each client has its
    own queues and admission goes round-robin across clients, live turns first within each. Single-turn requests cost
    room by their expected KV; small ones (expected KV < 16k) take a short lane past room and the decode budget, up
    to `max_num_seqs/8` at once.
  * **Decode growth**: each in-flight request reserves its client's p90 output (recent completions plus in-flight
    output so far) minus what it has streamed. A new request starts only if prompt + expected output fits 0.9 × GPU KV
    after vLLM's in-flight KV and every reservation. A client with fewer than 5 completions gets a 32k prior and at
    most 16 uncapped single-turn requests in flight.
  * **Hard brakes**: vLLM's waiting queue non-empty on two ticks; 120 s after any preemption, which also shrinks the
    GPU budget fraction.
  * **Under-used engine** (#97, aa-qwen-tbsci live-lock): with in-flight KV < 0.3, no waiting queue and no brake, the
    target cannot sink below its start and new conversations may start up to 0.9 × (GPU + CPU KV).
  * **Backpressure**: when a client stalls (no admission for it), its new conversations get 503 + `Retry-After: 60`
    after 600 s queued, or on arrival with 64 already queued after 60 s of no admission. Live turns are never
    rejected.
  * **Client disconnects**: a queued request whose connection closed is dropped; an in-flight one is cancelled within
    1 s, which closes the upstream connection and vLLM aborts it.
  * `X-Gate: 1` on every response; endpoints.json lists `gate_status_urls`.
* Serving: a `gate` entry in the serve_teacher PROFILE (`task_synthesis/serve/serve_teacher.py`) starts one gate per
  replica. Tests: `aa_evals/qwenserve/test_gate.py`.

## Results (2026-11-03)

Recipe: `synth/teachers/qwen38-27b-eval-fast.toml` (serve_teacher puts the gate in front). Versus `qwen38-27b-eval`:
bf16 GDN recurrent state (`--mamba-ssm-cache-dtype bfloat16`), `--gpu-memory-utilization 0.94`, a CPU KV tier
(SimpleCPUOffloadConnector + synth-cpu-supersede v2; 160 GiB on B200, `--tunables '{"kv_offloading_size": 80}'` on
H100), `--enable-prompt-tokens-details`. Model, sampling, parsers, template, 262k context and images unchanged.

### 1xB200, agentic closed-loop replay (campaign x), gen tok/s per GPU

| arm | 32 agents | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| base (`qwen38-27b-eval` as running: fp32 state, 0.88, no CPU tier) | 1,867 | | 1,343 | 679 | 881 |
| bf16 state + 0.94 | 1,996 | **2,669** | 1,946 | 979 | |
| + 160 GiB CPU tier (= recipe, fixed agents) | 2,117 | 2,651 | 2,146 | 2,281 | |
| recipe + admission gate (no manual cap) | 2,135 | | 2,384 | **2,458** | 2,281 |

* KV per B200: 3.41M tokens (base) vs 3.89M (bf16 state + 0.94; the hybrid block falls from 1,600 to 832 tokens);
  the tier adds 4.93M tokens (5,930 CPU blocks).
* Peak +43% (2,669 vs 1,867). Past the knee the base engine collapses (prefix hit 0.46 at 64 agents, 0.0 at 128,
  preemptions); with the recipe + gate, 96-128 offered agents give 2.3-2.5k, 2.6-3.6x the base at the same load.
* The gate found the knee by itself: target 0.60 -> 0.53 -> 0.85, ~31-55 requests running, prefix hit 0.83-0.94,
  no preemptions, 0 errors in 3,041 requests (the 79 non-200 outcomes are the replay's shutdown, last 29 s).
  Queue time: live-conversation turns p50 0 / p99 0 ms; new conversations p50 6.0 min / p95 11.8 min with 96-128
  agents offered against ~50 running (held, not failed). A recovery step down after the burst was not measured
  (replay levels only grow).
* MTP depth at the 48-agent knee (recipe): MTP2 2,505 (mean acceptance length 2.29), **MTP3 2,651**, MTP4 2,550
  (2.9): MTP3 stays.
* Startup 200-325 s per arm (engine-cache snapshots off: /data over quota).

### Numerics (state_check.py, 24 recorded contexts of 54-116k tokens, greedy first token, top-20)

| comparison | TV median | TV max | argmax agree |
|---|---|---|---|
| base (fp32 state): prefix hit vs fresh | 0.000 | 0.086 | 24/24 |
| recipe (bf16 state): prefix hit vs fresh | 0.037 | 0.116 | 24/24 |
| base vs recipe, fresh | 0.038 | 0.108 | 24/24 |

The recipe moves the next-token distribution as much as serving a prompt from its own prefix cache does (GLM's
bf16-state switch: 0.084). The startup probe (text + bash tool call through the qwen3_coder and qwen3 reasoning
parsers) passes on every arm. A small fixed-seed quality set (HLE/Omniscience) was not run: GPUs were short.

### 1xH100 (#111, 2026-11-03): `qwen38-27b-eval-h100`

The H100 jobs of #70 (campaigns a, b, q, g) waited behind higher-priority GPU jobs at bid 51 all night. #111 ran
them as campaigns `synth/campaigns/qwen38-27b-eval-h100-h{1..9}.toml` (runs under
`/opt/aai/qwenserve-runs/20261103-qwen38-27b-eval-h100-*`), with the same agentic replay as
campaign x. Start point = `qwen38-27b-eval-fast` with an 80 GiB CPU tier (`h-ref`).

**Gen tok/s per H100 (80GB HBM3) against agents, agentic closed-loop replay (5-min windows):**

| arm | GPU KV tokens | 8 | 16 | 24 | 32 | 48 | H100-h |
|---|---|---|---|---|---|---|---|
| h-ref (= `qwen38-27b-eval-h100`) | 1,138,835 | 628 | 778 | 942 | **949** | 910 | 1.04 |
| h-ref repeat (gpu205, round 2) | 1,138,835 | | | 945 | **1,152** | 1,141 | 0.78 |
| 160 GiB CPU tier | 1,138,835 | 614 | 796 | 858 | **1,046** | 1,038 | 1.02 |
| gpu_memory_utilization 0.96 | 1,185,142 | 648 | 778 | 804 | **914** | 909 | 1.30 |
| max_num_seqs 64 | 1,158,456 | 660 | 731 | 899 | 918 | **954** | 1.05 |
| max_num_batched_tokens 8192 | 1,184,357 | 657 | 760 | 895 | **945** | 909 | 1.04 |
| max_num_batched_tokens 32768 | 1,047,791 | 628 | 765 | 918 | 983 | **1,005** | 1.10 |
| MTP2 | 1,157,076 | 627 | 692 | 784 | 776 | **894** | 0.98 |
| MTP4 | 1,123,712 | 575 | 750 | 791 | **937** | 868 | 1.23 |
| max_model_len 131072 | 1,080,599 | | 843 | **1,052** | | | 0.81 |
| all capacity knobs (0.96, 8k, 64 seqs, 160 GiB), paired with the repeat | | | | 1,033 | **1,163** | 1,136 | 0.68 |
| H100 NVL 95 GB, h-ref | 1,519,493 | 551 | 736 | 863 | **943** | 833 | 1.30 |

**Two H100s (one 2-GPU job, same node), total gen tok/s:**

| arm | GPU KV tokens | 32 | 48 | 64 | 80 | H100-h |
|---|---|---|---|---|---|---|
| TP2 (one engine) | 3,199,098 | 1,695 | **2,055** | 1,778 | 1,706 | 1.91 |
| 2x TP1 replicas | 2 x 1,138,835 | 1,662 | **2,012** | 1,876 | | 1.98 |

* **The H100 plateau is ~950-1,150 gen tok/s per GPU from ~24 agents on, and no knob moves it beyond noise.**
  Single arms vary ~10-20% between nodes and runs: h-ref measured 949 on cpu206 and 1,152 on gpu205. Capacity knobs that
  add 2-4% KV (memory 0.96, 8k batched tokens, 64 seqs) or a bigger CPU tier were flat once paired on one node.
* Why: decode is memory-bandwidth-bound on long contexts. Every step reads each running sequence's KV (~32 KB per
  token over 16 attention layers, ~60k tokens per Pi agent), so time per token grows with the sequences running
  (TP2: 38 / 54 / 90 ms at 25 / 42 / 59 running) while KV still has room. More KV (TP2: +40% per GPU; NVL: +33%)
  runs more sequences without raising throughput. B200 has ~2.4x the bandwidth: same picture, 2.6x the plateau.
* MTP3 stays: MTP2 is 6-18% lower, MTP4 equal to lower (longer accepted runs, costlier steps).
* TP2 = 2x TP1 per GPU (2,055 vs 2,012): use TP1 replicas (simpler, independent failures).
* max_model_len 131072 does not add KV (1.08M vs 1.14M tokens); AA needs 262k anyway. Keep 262k.
* H100 NVL (95 GB, 12 4-GPU nodes): same peak (943) with 33% more KV; the spec runs on both types.
* **H100 / B200 per GPU** at each plateau: ~950-1,150 vs 2,669 (B200 recipe, 48 agents) = **0.36-0.43x**: one B200
  replaces ~2.5 H100s for this workload.

**Gate on H100** (40 agents offered, past the ~24 knee; 10-min windows; each gated arm paired with a no-gate arm on
the same node):

| pair | no gate | gated | gated / no gate | H100-h (both arms) |
|---|---|---|---|---|
| default gate (#108 policy) | 1,028 (0 preemptions) | 882 (20 running, KV ≤ 0.59) | 0.86 | 0.97 |
| H100 brakes only (0.98 / 0.94, budget 0.97) | 1,028 | 996 (25 running) | 0.97 | 0.45 |
| H100 policy (+ target start 1.2, floor 0.8), a | 1,147 (26 preemptions) | 986 (0 preemptions) | 0.86 | 1.10 |
| H100 policy, b | 1,128 (20 preemptions) | 982 (0 preemptions) | 0.87 | ~1.0 |

* Behind the gate an H100 runs ~23 sequences with no preemptions. Without it, ~31 run, with a waiting queue and
  20-26 preemptions per 10 min; that oversubscription is worth ~13% on a pure agent load. The gate's value is
  elsewhere: fairness between clients, the decode budget for long single-turn answers, backpressure and disconnects
  (#108).
* Gate policies are within noise of each other at steady state. The default (#70's B200 values: kv_fit target 0.6
  with a CPU tier, in-flight brakes 0.92 / 0.85) climbs slowly on an H100: in 5-min windows (round h7) it held 16
  running at KV 0.6 with 32 offered (876 gen tok/s) and 14 with 64 (670). `qwen38-27b-eval-h100` therefore uses the
  H100 policy (serve_teacher PROFILE): target start 1.2 (idle contexts spill to the 2.5M-token CPU tier), floor 0.8,
  brakes 0.98 / 0.94, decode budget 0.97.

**Mixed replay on one H100** (`qwen38-27b-eval-h100` via serve_teacher with the default gate, before the H100 policy; 24 agents + 32 HLE at
medium with no output cap + 8 short; 30 min measured): 901 gen tok/s, no preemptions, KV max 0.90; short requests
960 done (queue p50 0 s); HLE 11 done (mean answer 73k tokens; 89 503s while stalled); agents only 90 turns. A few
long HLE answers take most of an H100's 1.1M-token KV: on H100s, give long single-turn benchmarks their own replicas
rather than mixing them with agents on one GPU.

**GPU time.** 22.5 H100-h in all: round 1 (h1-h5, 12 arms) 14.8; round 2: combined knobs (stopped after one
flat pair) 1.65, gate burst 1.15, mixed replay 1.2; gate policy 1.6 + 2.1.

**Availability (2026-11-03).** 30 H100 nodes: 18 80GB HBM3 (8 GPUs each; 104 usable after the bad nodes) and 12 NVL
(4 GPUs each, 48). At bid 51 the jobs started within 10 min in the morning but idled 30+ min in the afternoon with
H100s free; bid 101 started them in minutes. B200: see #70.

### Gate on a shared server (mixed replay, `mixed_replay.py`, 1xB200, `qwen38-27b-eval-fast`)

Several synth.bench replays at once against one gate, each its own client (`X-Gate-Client`): Pi agents (closed loop,
campaign x corpus), HLE single-turn (QA corpus) and a short single-turn stand-in for Omniscience (the same prompts at
reasoning `low`, max_tokens 1024). Base = the gate of #71 (FIFO, continuations first, kv_fit). Runs under
`/opt/aai/qwenserve-runs/fairval/` (`mixed.jsonl`, per-arm `serving/<arm>/gate-r0.jsonl`).

**Fairness** (96 agents + 48 HLE at max_tokens 32k + 16 short; 10 min warm-up, 20 min measured):

| gate | server gen tok/s | preemptions | agents turns | HLE done (queue p50) | short done (queue p50 / p95) |
|---|---|---|---|---|---|
| base | 2,362 | 0 | 1,094 | 35 (436 s) | 6 (742 s / 807 s) |
| per-client fair queuing | 3,182 | 0 | 1,162 | 101 (107 s) | 405 (2.4 s / 204 s) |

**Long reasoning** (64 agents + 96 HLE at reasoning medium, no output cap + 16 short; 30 min warm-up, 30 min measured):

| gate | server gen tok/s | preemptions | KV max | agents turns | HLE done (queue p50) | short done (queue p50 / p95) | 503s in the window |
|---|---|---|---|---|---|---|---|
| base | 2,850 | 25 | 0.998 | 762 | 55 (749 s) | 13 (869 s / 888 s) | 0 |
| fair + decode budget + brakes (first cut) | 2,648 | 0 | 0.96 | 964 | 20 (535 s) | 118 (95 s / 487 s) | 3,046 |
| + stall-only backpressure, short lane past the budget (= PR) | 3,002 | 0 | 0.92 | 1,295 | 26 (490 s) | 1,023 (2.9 s / 16 s) | 201 (HLE) |
| + head-of-line hold for large requests (reverted) | 2,164 | 0 | 0.55 | 398 | 16 (521 s) | 1,646 (5.0 s / 8.6 s) | 511 (HLE) |

* The first cut rejected on queue length alone (503 storms while the queue moved; the replay client retries every
  10 s, ignoring Retry-After) and kept small requests behind the decode budget; both fixed.
* The decode budget trades HLE concurrency for no preemptions: HLE's p90 answer was ~57k tokens, so ~12 fit beside the
  agents, and HLE finished 26 requests in the window against 55 on the base gate (which got them by overcommitting
  the KV: 25 preemptions, agents and short requests starved). Open: large requests still lose freed KV to small
  ones. Holding other clients' new conversations for a large request that waited 30 s (head-of-line) did not help
  HLE (16) and left the engine under-filled (KV max 0.55; the throughput hill-climb then lowered the kv_fit target to
  0.47), so it was reverted. A per-client KV share would be the next thing to try.

**Agents only** (96 agents, 10 min warm-up, 15 min measured): 2,646 gen tok/s through the PR's gate (480da02, before
the stall-only backpressure and the short lane past the budget) vs 2,458 for #71's gate at 96 agents (campaign x), prefix hit 0.88, no preemptions: the
reservations do not cost agentic throughput.


## Production throughput round (#110, 2026-11-03)

Code: `gate.py` now fronts one engine or a **pool** of N (`--upstream a,b,...`): client queues, conversations, fairness
and the KV share are the pool's; kv_fit room, decode budget, brakes and in-flight caps are each engine's. A live turn
goes back to its conversation's engine (prefix cache); a new request goes to the engine with the most free decode
budget. Two gate fixes: **short-lane demotion** (a short-lane request that streams past `short_tokens` or runs past
`short_seconds` leaves the lane) and a **per-client KV share** (`client_share`, on in the `qwen38-27b-eval-fast`
profile; off by default in `gate.py`: max-min fair split of the
pool's GPU budget over demand; while a client under its share waits, clients over theirs start no new non-short
requests; the protected client needs only the decode budget, not kv_fit room). Monitor: `monitor.py`. Serving:
tunables `gate_pool` (one gate over every replica of a job), `gate_policy`, `replica_tunables` (per-replica overrides,
for twin A/B engines in one job).

**Method.** `replay_arms.py`: one engine start, arms chained on it; per arm, wait for idle engines, reset vLLM's prefix
cache and CPU tier (`/reset_prefix_cache?reset_external=true`, `VLLM_SERVER_DEV_MODE=1`), start the arm's gates
fresh, run `mixed_replay.py`, stop the gates. Engine-flag comparisons run as **twin engines** in one job (same node,
same time, identical load per engine). Load per engine = #108's long mix: 64 Pi agents (closed loop, campaign x corpus)
+ 96 HLE at reasoning medium, no output cap + 16 short (QA corpus at reasoning low, max_tokens 1024); 20 min warm-up,
20 min measured. Server gen tok/s per engine is averaged over the measured window from vLLM's 10-s log lines
(`arms_table.py --serving`); client-side tok/s counts a request when it completes and is lumpy for 30-60k-token
answers. **A/A noise** (two identical engines, recipe1): 2,949 vs 3,063 server gen tok/s (~4%); client totals 10%
apart. Runs: `/opt/aai/qwenserve-runs/servprod/` (`arms.jsonl`, `mixed.jsonl`,
`arms/<arm>/gate-*.jsonl`).

### Pooled gate vs per-benchmark servers (2xB200, same offered load: 96 agents + 96 HLE + 16 short)

| arm | server gen tok/s (sum) | engine KV | agents tok/s | HLE done / 503s | short queue p50 / p95 | preemptions | B200-h |
|---|---|---|---|---|---|---|---|
| per-benchmark (agents on one engine, HLE + short on the other) | 5,830 | 0.58 / 0.78 | 2,428 | 42 / 324 | 10 s / 38 s | 0 | 1.17 |
| one gate over both engines | **6,449 (+11%)** | 0.80 / 0.65 | 3,065 | 80 / 0 | 2.5 s / 41 s | 0 | 1.17 |

| one gate over both engines, gate v2.1 (short-lane demotion + per-client KV share) | **6,751 (+16%)** | 0.79 / 0.78 | 2,998 | 101 / 0 | 0.0 s / 0.2 s | 0 | 1.17 |

**Imbalanced load** (24 agents + 96 HLE + 16 short, the morning's shape: one light benchmark beside a backlogged one):

| arm | server gen tok/s (sum) | engine running / KV | agents done / turn latency p50, p90 | HLE done / 503s / latency p50 | short latency p50 | B200-h |
|---|---|---|---|---|---|---|
| per-benchmark (24 agents alone on r0; HLE + short on r1) | 6,281 | 19 / 0.19 and 72 / 0.81 | 776 / 5.1 s, 70 s | 69 / 35 / 624 s | 16 s | 0.84 |
| one gate over both (v2.1) | **7,686 (+22%)** | 61 / 0.66 and 59 / 0.61 | 403 / 8.4 s, 159 s | 127 / 0 / 177 s | 15 s | 0.84 |

The pool's gain is the idle engine's capacity going to the backlog: HLE finishes 1.8x the answers. The price is that
the light closed-loop client decodes in a batch of ~60 instead of ~19, so its turns take 1.6-2.3x longer and it
completes half as many. Pooling wins total GPU throughput, and the total eval wall clock when every benchmark has a backlog.
A latency-bound benchmark on the critical path (few long agent sessions) is better served on its own engine;
the KV share doesn't help here (it splits KV, not decode speed).

### Gate variants on one engine (twin engines, long mix)

| arm | server gen tok/s | agents tok/s | HLE tok/s (done) | HLE queue p50 / p95 | short done (queue p50 / p95) | B200-h |
|---|---|---|---|---|---|---|
| v2 (#108) | 3,096 | 1,799 | 978 (33) | 445 s / 1,182 s | 535 (9.0 s / 70 s) | 1.34 for the pair |
| v2 + equal share, target search frozen while holding (first cut) | 2,924 | 1,535 | 607 (23) | 1,209 s / 1,254 s | 782 (8.8 s / 19 s) | |
| v2.1 = v2 + short-lane demotion | 3,217 | 1,077 | 1,038 (25) | 239 s / 597 s | 919 (0.0 s / 0.0 s) | 1.34 for the pair |
| v2.1 + max-min share (protected requests skip kv_fit room) | 3,305 | 1,012 | 1,184 (30) | 255 s / 381 s | 1,041 (0.0 s / 0.0 s) | |

* v2's short lane was clogged: HLE's first completions are its short ones, so its learned p90 let uncapped HLE requests
  into the short lane, where they grew to tens of thousands of tokens and held all 16 short slots (16 short requests
  queued, none in flight). Demotion frees the slot; the short client's queue went to 0 s.
* The first share cut lost 14% client throughput and starved HLE as well: freezing the target search while holding
  kept the kv_fit target at 0.63 (v2 climbed to 1.16), room stayed 0, so the protected HLE requests could not start
  either, stalled, and drew 1,344 backpressure 503s. Fixed by not freezing and letting protected requests skip room.
* The fixed share is neutral in total (+2.7%, inside the A/A noise) and moves throughput toward the long answers it is
  for: HLE +14% tok/s, HLE queue p95 -36%, agents -6%.
* Pairs on different days/nodes are not comparable to each other (v2 and v2.1 above ran on different engine starts).

### Recipe round 2 (twin engines in one 4xB200 job, gate v2 on each, long mix)

| engine | server gen tok/s | prompt tok/s | running | KV | prefix hit |
|---|---|---|---|---|---|
| recipe (qwen38-27b-eval-fast) | 2,949 | 3,345 | 61.0 | 0.72 | 0.86 |
| recipe again (A/A) | 3,063 | 2,816 | 65.1 | 0.81 | 0.70 |
| max_num_batched_tokens 32,768 (16,384) | 2,958 | 3,243 | 60.1 | 0.66 | 0.86 |
| CPU KV tier 320 GiB (160) | 3,090 | 2,864 | 59.7 | 0.65 | 0.85 |

3.17 B200-h for the job. And vLLM 0.33.0 (released 2026-11-02, same torch 2.14; the supersede plugin applies) vs the
0.32 build, twin engines (1.59 B200-h): 3,386 vs 3,458 server gen tok/s (-2%) at fewer running (66 vs 77) and lower KV
(0.68 vs 0.81); mean MTP acceptance 2.69 vs 2.84.

* All flat within the ~4% A/A noise, so each axis stopped after one arm: the recipe stays. MTP depth was already
  measured (#70: MTP3 best at the agent knee). Prefix caching is on; agents' prefixes hit 0.80-0.86 of prompt tokens
  (GPU + CPU tier), so there is no shared-prefix headroom to chase. Batched 8k and chunked-prefill variants were not
  run after 32k came out flat. vLLM 0.33 is not worth a production switch on this evidence (a fixed-concurrency twin
  without the gate would separate its per-step speed from the gate's admission path; not run).

## Production runbook (#110)

Servers: one `serve_teacher` job per registry (`/data/aai/tbench/tasksynth/serving/<name>/`), spec
`qwen38-27b-eval-fast`, 1xB200 each, gate in front (endpoints.json `urls` = gate, `engine_urls` = raw vLLM).
`serve_teacher submit` freezes the checkout it runs from (`job.json` `code`), so **the commit you submit from is the
gate you get**. Restart only between eval rounds; the coordinator does restarts.

**Roll out (or update) a gate on server NAME**
1. Pick the code: a worktree of `aa-serving` at the wanted commit (gate v2.1 = the merge of #110's PR, which adds
   short-lane demotion, the per-client KV share (on in the profile) and pooling; gate v2 = 0f27ea4, #108). Check it:
   `git -C <wt> log --oneline -1`; `grep -c X-Gate-Client <wt>/aa_evals/qwenserve/gate.py` (> 0 means v2).
2. Stop the old job: `condor_rm <job_id from serving/NAME/job.json>`; wait for `endpoints.json` to disappear (the
   job deletes it on exit; ~30 s).
3. Submit from the worktree: `cd <wt> && /opt/aai/venvs/harbor/bin/python -m
   task_synthesis.serve.serve_teacher submit --teacher qwen38-27b-eval-fast --name NAME --gpus 1 --bid 61 --force`.
   Healthy after ~5-7 min (staging + compile + probe through the gate).
4. Check: `curl -s --noproxy '*' <gate>/gate/status` has `clients` and `engines`; responses carry `X-Gate: 1`.
5. Clients: send `X-Gate-Client: <benchmark>`; set a high in-flight cap of their own (the gate decides what runs);
   treat 503 + `Retry-After` (`server_overloaded`) as "retry later", never as a failed sample; subtract
   `X-Gate-Queue-Ms` from latency. Never bypass the gate on the same engine (the raw port's load is invisible to it).
6. Watch for the first 30 min, then periodically:
   `python -m aa_evals.qwenserve.monitor --loop --every 60 --out /data/aai/aa_evals/runs/qwenserve/monitor.jsonl`
   (one line per server per minute in one file). Healthy: no `kv_high` (KV <= 0.92), no `preemptions`, no
   `starved:<client>`, `vllm_waiting` at most transient, short-class queue p95 under ~30 s.
7. Roll back: `condor_rm`, then submit the same NAME from the previous worktree (gate v1 = 37c5d9e,
   `worktrees/tasksynth-run-37c5d9e0e4`).
8. After a clean round on v2, retire the client-side limiters (HLE, CritPt, pi_eval) one at a time: raise the cap,
   watch one monitor window.

**Pooled serving (recommended once the evals share servers):** one job, N B200s, one gate over all replicas:
`... serve_teacher submit --teacher qwen38-27b-eval-fast --name aa-qwen-pool --gpus 4 --bid 63 --tunables
'{"gate_pool": 1}'` (64 CPUs, 1,280 GiB). endpoints.json lists one URL (the pool gate) and N `engine_urls`; every
benchmark sends its `X-Gate-Client` to that one URL. The gate keeps each agent conversation on its engine and puts
new work where the decode budget is freest, so an idle benchmark's GPU serves the others instead of idling (this
morning: tb21 at KV 0.00 beside qa2 with 455 queued). One replica dying stops the job (serve_teacher exits on any
dead process), as before for multi-replica jobs. Measured: +11-16% server gen tok/s on a balanced mix, +22% when one
benchmark is light, but a light latency-bound client's turns get ~2x slower (tables above); keep such a benchmark on
its own server when it is the critical path. `/metrics` labels each engine's vLLM samples `replica="i"`;
`gate:*` gauges per engine carry `engine="i"`; the monitor shows KV per engine.

**Alerts and what to do**
| alert | meaning | action |
|---|---|---|
| `kv_high` | an engine's KV > 0.92 | transient during bursts is fine; sustained with `preemptions`: lower `inflight_kv_max` (gate policy) |
| `preemptions` | vLLM preempted requests | the gate holds new work 120 s and shrinks its budget itself; repeated: a client bypasses the gate? |
| `starved:<client>` | queued, nothing admitted for 10 min | check `/gate/status` `brakes`/`causes`; a single-turn client with a huge p90 waits for the decode budget (#108 trade-off; see the per-client KV share) |
| `vllm_waiting` | requests queue inside vLLM | sustained: something bypasses the gate (raw port) |
| `unreachable` | /metrics failed | the job died or the node is gone: `condor_q`, `serving/NAME/serve.log` |

## Qwen3.6-35B-A3B (M3.3, #176, 2026-11-05)

Specs: `synth/teachers/qwen36-35b-a3b-eval-h100.toml` (1xH100, preferred) and `qwen36-35b-a3b-eval.toml` (1xB200).
Official Qwen FP8 checkpoint (block-128), reasoning on (Qwen3.6 has no effort levels; the template reads
`enable_thinking` / `preserve_thinking` only), Qwen's "thinking mode for general tasks" sampling (T 1.0, top_p 0.95,
top_k 20, presence 1.5), 131,072-token context, images up to 256, vLLM 0.32, FP8 KV (H100: FA3; B200: TRTLLM-gen),
bf16 GDN state, MTP3, CPU KV tier (80 GiB H100, 160 GiB B200), 128 seqs, 16,384 batched tokens, gate v2.1 with the
per-client KV share (H100: #111's H100 policy). Runs: `/opt/aai/qwenserve-runs/m33-qwen36/`
(BPB, gate, corpus) and `20261105-qwen36-35b-a3b-eval-{h100-t1,b200-t2}/`.

**Correctness** (`aa_evals.tb21.run --context-window 131072`, pi180; `parse_audit.py` over every streamed response):

| run | tasks | pass@1 | infra | responses | tool calls | markup in content | think in content | output cap |
|---|---|---|---|---|---|---|---|---|
| smoke `m33-qwen36-smoke1` | 6 x 1 | 1/6 | 0 | 214 | 226 | 0 | 0 | 1 |
| live `m33-qwen36-x1` (H100 NVL, gate) | 89 x 1 | **40.4 [30.3, 50.6]** (AA 44.9) | 0 | 3,810 | 3,940 | 0 | 0 | 39 |

The only responses without reasoning are the harness's own context-cap message (23 = the 23 attempts that hit the
131k window). Two long training tasks (caffe-cifar-10, train-fasttext) were tail-cut at 2.2 h and scored by their
verifiers (both 0). Mean 34.4k output tokens per attempt (Qwen3.8 medium at 132k: 27.8k).

**Numerics** (`traj_bpb.py`: served BPB of the model's own 48 TB 2.1 trajectories rendered with its template, 1.14M
tokens, 395k assistant label tokens; vLLM 0.32 prompt log-probs):

| arm | B200 | H100 80GB |
|---|---|---|
| BF16 weights, bf16 KV | 0.15077 | (does not fit with log-prob buffers) |
| BF16 weights, FP8 KV | +0.00014 | |
| FP8 weights (official), bf16 KV | -0.00003 | 0.14989 (reference for the H100 column) |
| FP8 weights, FP8 KV (default backend: B200 TRTLLM-gen / H100 FA3) | +0.00025 | +0.00014 |
| FP8 weights, FP8 KV, FlashInfer bf16 query | | +0.00011 |
| + bf16 GDN state (= spec; prefill only) | +0.00022 | +0.00001 (FlashInfer) |

Every numerics change is below 0.0003 BPB (Lightning accepted +0.0017): all adopted. H100 and B200 serve 0.0009 apart
on the same weights (different MoE/GEMM kernels); compare within a column.

**Throughput** (`synth.tune`, closed-loop replay of `m33-qwen36-x1` + smoke, 80-95 trajectories; 300 s windows;
server gen tok/s / mean running / KV max; one engine per arm, same node per GPU type):

| 1x H100 80GB (KV 2.76-2.84M tokens) | 20 | 40 | 60 | 80 |
|---|---|---|---|---|
| h-ref (FlashInfer bf16 query, MTP2, 80 GiB tier) | 904 | 1,158 | 1,294 / 51 / 0.82 | 1,587 / 61 / 1.00 |
| MTP3 | 950 | 1,248 | 1,344 | 1,589 |
| MTP0 | 825 | 1,053 | 1,312 | 1,361 |
| FA3 (vLLM default) | 1,101 | 1,416 | 1,523 | 1,703 |
| no CPU tier | 927 | 1,085 | 589 (hit 0.57) | 257 (hit 0.00) |
| **h-final = spec (FA3, MTP3, 80 GiB tier)** | 1,010 | **1,622** | **1,626 / 47 / 0.93** | **1,797 / 56 / 1.00** |

| 1x B200 (KV 10.2-10.5M tokens, 160 GiB tier, TRTLLM-gen) | 40 | 80 | 120 | 160 |
|---|---|---|---|---|
| MTP2 | 2,491 | 3,612 | 4,565 / 85 / 0.38 | 4,804 / 109 / 0.53 |
| **MTP3 = spec** | **3,004** | **3,807** | **4,743** | **4,995 / 111 / 0.56** |

* FA3 beats FlashInfer with a bf16 query by 7-29% here, unlike Qwen3.5-4B (#15), at the same BPB.
* The CPU tier is essential on H100: without it the cache thrashes from ~60 agents (589 and 257 gen tok/s).
* H100: the GPU KV fills at ~55-60 running sequences (80 offered agents; a few preemptions at 80); throughput is
  still rising there, so ~80 agents per H100 is the operating point. B200: compute-bound near 160 agents with KV at
  0.56, so a B200 holds 160+ agents; one B200 = ~2.8 H100s.
* **Max concurrent trajectories at 128k:** GPU KV alone holds 21 (H100 80GB), 30 (H100 NVL), 78 (B200) full 131k
  contexts; with the CPU tier and TB-shaped contexts, ~80 agents per H100 (56 running) and 160+ per B200 (111 running).

**Admission gate** (mixed replay on the H100 NVL server, 64 agents + 32 HLE at no output cap + 8 short, 15-min
window; engine = h-ref):

| arm | server gen tok/s | KV max | running | agent turns | HLE done / queue p50 / 503s |
|---|---|---|---|---|---|
| no gate | 1,850 | 0.96 | 93 | 2,769 | 13 / - / 0 |
| gate v2.1, default policy | 1,635 | 0.68 | 62 | 1,886 | 9 / 373 s / 17 |
| **gate v2.1, #111 H100 policy (= profile)** | **1,854** | 0.96 | 93 | 2,678 | 14 / 101 s / 0 |

No preemptions in any arm. The B200 profile keeps #70's default policy (not re-measured on Qwen3.6; its 10M-token KV
never passed 0.56 in the replays).

**GPU-hours per full eval** (`eval_gpu_hours.py --rate hgpu100=1800,b200=5000`: Qwen3.8 medium's measured output tokens
per benchmark x 1.24, util 0.75; GDPval not counted): ~70M output tokens, **~14 H100-h or ~5.2 B200-h**. The live
TB 2.1 pass (89 x 1, at most 64 in flight) reached 50 / 90 / 95% of its attempts in 23 / 46 / 50 min on one H100 NVL.

Not done: vLLM 0.33 and max_num_batched_tokens arms (flat for Qwen3.8 in #110); 256 seqs (H100 runs < 64 sequences,
KV-bound).

GPU time: 8.4 H100-h (smoke + live server 2.6, 4B smoke 0.3, BPB 0.3, tuning 5.1), 1.9 B200-h (BPB 0.3, tuning 1.6).

## Qwen3.5-4B (M3.3, #176): eval_q35's tuned serving ported

Spec `synth/teachers/qwen35-4b-eval.toml` (no re-tuning): eval_q35's fp8kv (#15) = FP8 KV, `--mamba-cache-mode
align`, FlashInfer with a bf16 query on H100 (`{"flashinfer_bf16q": 0}` + `--cuda-cap 10.0` for B200's TRTLLM-gen),
256 seqs, util 0.90. Differences: vLLM 0.32 instead of the 0.27.1 SIF (serve_q35 measured both), the official
checkpoint and its own template, Qwen's thinking sampling, images on, 131,072 context. 1x H100 80GB: 3.62M KV tokens.

Admission: eval_q35's context-aware turn gate (v11), which belongs with the eval run (it syncs with the run's claims
and each rollout's proxy asks it per turn). aa_evals' pi_eval hosts it when the registry's `request.ctx_gate` is true
(branch `aa-evals-q35-turngate`, `aa_evals/common/pi_eval/turn_gate_host.py`); the profile puts no aa_evals gate in
front. Smokes through aa_evals TB 2.1 (`m33-q35-4b-gate-smoke1`, fill 0.05: 8 tasks; `-smoke2`, fill 0.008: 4
tasks): gate elected on a worker, capacity 3.6M from /metrics, claims synced (finished rollouts drop out), turns held
and revoked (4 revocations, 2 parked) and re-admitted as the learned fill rose (0.008 -> 0.028 -> 0.048); 559
responses, 553 tool calls, no parser leaks, no infrastructure errors in the 9 attempts that finished (0/9 passed: a
plumbing smoke; AA TB 2.1 25.8); smoke1's last 3 attempts were stopped when the server was released.

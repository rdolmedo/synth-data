# aa_evals: the AA Intelligence Index v5.0 benchmarks on our cluster

This directory holds the eval code for the benchmarks in the Artificial Analysis Intelligence Index v5.0. The audit and benchmark choice are in issue #27.

We reproduce AA's methodology as closely as we reasonably can. It will not be exact. When the test set is private, we evaluate on the public set. Every deviation from AA is listed in the benchmark's own README.

| Benchmark | Dir | AA weight | Issue |
|---|---|---|---|
| Terminal-Bench 4.0 | `tb4/` | 10% | #28 |
| Terminal-Bench-Science 0.1 | `tbsci/` | 15% | #29 |
| AutomationBench-AA | `automationbench/` | 5% | #30 |
| AA-Omniscience 1.1 | `omniscience/` | 15% | #31 |
| CritPt v2 (hard subset) | `critpt/` | 10% | #32 |
| HLE-Diamond | `hle/` | 10% | #33 |
| GDP.pdf | `gdppdf/` | 5% | #34 |
| SciCode-Verified + our fix pass (AAI-open stand-in for FrontierCode 2.0) | `scicode/` | 15% | #184 |

AA-Briefcase and FrontierCode 2.0 (30% of the index) have no usable public data, so they are out of scope.

## Contract every benchmark follows

**One entry point:**
```
python -m aa_evals.<bench>.run --model <model spec> --run-id <id> [--limit N] [--repeats K]
```
`--limit` takes a small subset for smoke tests.

**Model spec.** A model is named in one of two ways:
- `serving:<name>` is resolved from `<registry>/<name>/endpoints.json`: the registry is `$TASKSYNTH_SERVING_ROOT`, else `/data/aai/tbench/tasksynth/serving` (`aa_evals/common/serving.py`; every reader and every eval condor job follows the variable). That file comes from `task_synthesis.serve.serve_teacher` and holds `served_name`, `urls` (one per DP rank) and `updated_at`. A file older than 120 s means the server is dead.
- `url:<base_url>@<served_name>` gives any OpenAI-compatible endpoint directly.

The generation settings (reasoning effort, temperature, max tokens) are part of the run config. They must mirror the AA setting we compare against, for example AA's "(Max)" or "(High)" variant.

**Text only (#348, user decision 2026-11-11).** The models we fine-tune are text-only, so every model is evaluated
without vision:
- **Serving:** the multimodal models run aa-serving's text-only variants, `<spec>-text` (the M3.6 spec with
  `--language-model-only`, so the vision tower is not built): `qwen38-27b-eval-fast-text` (Qwen3.8 Xhigh and Medium),
  `nemotron35-super-mm-text` (Super VL), `qwen36-35b-a3b-eval-text`, `gemma4-26b-a4b-it-eval-b200-text`,
  `qwen35-4b-eval-text` and `glm53-flash-eval-text`. Lightning (`nemotron35-lightning-eval`, a text-only architecture)
  and DeepSeek (`deepseek-v41-flash-1017`, already `--language-model-only`) are unchanged.
- **Runs** read image support from the serving launch argv (`protocol.image_inputs`, pi_eval `model_input`), so on
  these servers every harness records text only (`['text']` for Pi).
- **Item sets:** GDPval `text-only-v1` (207 of 220, #346), HLE-Diamond `text-only-v1` (728 of 1,000), GDP.pdf
  `text-only-v1` (75 of 100) with the text alone (`--vision off`). `--task-set all` (GDP.pdf: also `--vision auto`)
  reproduces the earlier protocol.

**Outputs** go under `/data/aai/aa_evals/runs/<bench>/<run_id>/`:
- `config.json`: git SHA, model spec and served name, generation settings, dataset id plus version/SHA, judge model and settings, harness, limits.
- `items.jsonl`: one row per item × repeat, with `item_id`, `repeat`, `score`, `status` (`ok`, `model_error` or `infra_error`) and a path to the raw transcript.
- `summary.json`: `benchmark`, `metric` (named as AA names it), `score` on AA's 0–100 scale, `n_items`, `n_repeats`, `ci95`, `n_infra_errors`, and `aa_reference` (AA's published score for this model, when AA lists it).

**Infra failures are never scored as model failures** (lessons of #18):
- Retry them, or mark them `infra_error` and exclude them from the score.
- Report how many there were.
- A run with more than 2% infra errors is not a valid result.

**Data** lives in `/data/aai/aa_evals/data/<bench>/`, pinned by version or SHA, and is not committed to git. HLE-Diamond's questions may not be redistributed, and every public eval set stays out of training data.

**Reference numbers.** AA's scores for 278 models are in `/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json`, in each model's `scores` field. Our first sanity check is whether our numbers for the same model land near AA's.

## Cost and storage rules (#378)

The c30 audit (#378) found that 88% of a campaign's unit-hours were CPU/RAM, not GPU, and most of it was waste:
oversized and idle CPU jobs, CPU jobs on GPU-node floor prices, idle servers, and servers above or below their knee.

**Storage.** Runs, condor event logs (`log =`), stdout/stderr, code snapshots and audit files go to `/data`, even when
`/data` warns that it is over quota. `/home` is for code checkouts and venvs only, never for runs, logs or data:
its per-user quota is small, and when it filled on 2026-11-12 condor silently dropped event-log records, then every
writer stopped. The old `/opt/aai/aa_runs_fallback` is a symlink to
`/data/aai/aa_evals/runs_fallback`, the pi_eval default root.

**Writes to /data count against a job's memory until written back.** Dirty page cache sits in the job's cgroup and
/data's writeback is often slow (20 small buffered writes: 3.5 min; with O_DIRECT: ~10 ms), so jobs get held "over
memory limit" with a tiny RSS (c30n's GDPval workers and collectors, writing archives). Bulk outputs are built on
node-local disk and copied with `aa_evals/common/fastio.py copy_direct` (O_DIRECT; pi_eval's trial archives and the
GDPval collector's attempts and renders do); appends of large records (CritPt and AutomationBench collectors) use
`fastio.append_line` (fsync, then drop the cached pages).
More RequestMemory only delays such holds: every aa-evals CPU job submitted through `condor_cost` is released by
condor after a memory hold (code 34) and restarts, at most 5 times (`periodic_release`); its work resumes. Since
tbshared 1.5 each restart asks 25% more of the base RAM (the memory net, `holds.py`'s expression), so a job that really
needs more stops looping at the same size; above 16 GB per CPU that costs units, which is intended.

**Charges.** units = max(RequestCpus, RequestMemory_GB / 16) + 24 per GPU, per hour, on what a job *requests*, at the
node's running price (at least its floor: CPU-only and A100 nodes 5-25, H100 35, B200 55). RAM up to 16 GB per
requested CPU is free; above that it costs as much as CPUs.

**Fill to the binding resource (tbshared 1.5).** "Since mem is free for enough cpus, make sure to size the other one
accordingly, since we will pay the same. according to the one that binds" (user, 2026-11-14). Every job submitted
through `condor_cost` (`submit`, `submit_file`, the CLI) goes out with GB = max(GB, 16 x CPUs) and CPUs = max(CPUs,
ceil(GB / 16)), at the same units (`condor_cost.fill_plan`):
- **Within the nodes.** Never more than the largest node that gives the job its CPUs.
- **Placement-aware.** The batch is planned unfilled and filled. If the filled plan would start any job on a higher rung, or leave one fitting on no node, RAM is capped at the bottom rung's free GB per free CPU; if that is still worse, the batch stays as asked. A fixed `--bid` keeps the fill only if the room at its floor holds. Filling costs RAM, the scarce resource on the cheap tier.
- **Decisions as data.** Each batch's decision (time, pipeline, batch, before/after, decision, reason, plans) is appended to `fill.jsonl` next to its submit file. `python -m tbshared.condor_cost fill-report DIR...` counts decisions and capped fills, and compares the rung mix with the unfilled plans.
- **Opt-out:** `fill=False` per call. Shapes in the code stay what the jobs need; the fill happens at submission.

**CPU job shapes: lean, with a small safety.**
- pi_eval workers: slots, CPUs and RAM per benchmark from c30 and the #378 measurement run (`pi_eval/README.md`
  table; e.g. TB 2.1: 32 slots on 10 CPUs / 128 GB = 0.31 units per slot, was 32 CPUs / 256 GB for 24 slots). CPU is a hard cap: condor pins each job to an affinity mask of exactly `request_cpus` cores (#378: measured in every serving job; CritPt workers at 12 CPUs never exceeded 11.95 cores), so the slots of a pi_eval worker share exactly its requested cores. Size CPUs from measured p99 with headroom, not from averages, except for pi_eval workers: they are deliberately sized at the full-slot mean / 0.8, because their slots mostly wait on vLLM and some burst delay is accepted for higher average utilization. At the current caps, in minutes with at least 75% of slots busy, about 3.7-9% of CPU work is deferred (GDPval 3.7%, CritPt 4-9%, TB 2.1 6%), under 1% of attempt time (coordinator, #378). Memory is a hard limit too (below).
- Size RAM from the job's cgroup peak (pi_eval workers log it: `RUN/jobs/res.<job>.jsonl`, `aa_evals/common/reslog.py`
  for any job), never from condor's `MemoryUsage`, which is RSS only: RSS-sized RAM got every c30 Pi worker held.
- Clients and bookkeeping jobs, as asked (submitted filled where free, e.g. 1 CPU / 16 GB):
  - HLE, Omniscience and GDP.pdf (text-only): 1 CPU / 4 GB;
  - SciCode: 8 CPU / 32 GB (grading runs beside generation);
  - Pi collectors: 1 CPU / 4 GB (one job per model; c30's used 0.14 cores);
  - CritPt reprobe: 4 CPU / 32 GB (c30: 16 / 64 using 2.4 cores on average);
  - GDPval reference judging: 1 CPU / 8 GB.

**Shared layer.** `common/condor_cost.py`, `common/reslog.py` and `common/fastio.py` are pinned, byte-identical
copies of the shared layer (`tbshared/`, branch `shared`, `tbshared/README.md`; listed in `tbshared/ALIASES` because
per-benchmark snapshots copy only parts of `aa_evals/`). Never edit them here: fix them on `shared`, then bump with
`python -m tbshared.sync` (`tbshared/test_pin.py` fails on a local edit). The bad-node list is
`condor_cost.bad_nodes()` (BAD_NODES as the file says now, plus the live file `/data/aai/tbench/shared/bad-nodes.txt`,
read at every submission: #531 A9), used by every aa-evals submitter (harbor_mswea, pi_eval, AutomationBench, HLE, CritPt,
SciCode, GDP.pdf, GDPval, judge sweeps, Omniscience, TB-Science validation); m5/ adds its own list on top.

**CPU bids: the ladder** (`common/condor_cost.py`). Every aa-evals CPU job (pi_eval and harbor_mswea workers, SciCode,
CritPt, GDP.pdf, GDPval, HLE, Omniscience, AutomationBench, judge sweeps) bids **26 -> 36 -> 56** unless `--bid` /
`--cpu-bid` is given:
- 26 reaches CPU-only and A100 nodes, 36 adds H100 nodes, 56 adds B200 nodes.
- At submission each job starts on the lowest rung with room for its shape now, so a full cheap tier costs no wait.
- A job still idle after 5 min at 26, or 10 min at 36, is rebid in place one rung up by the watcher job
  (`aa-bid-ladder-watch`, 1 CPU, started on first use, exits after 2 h without ladder jobs; log in
  `/data/aai/aa_evals/bidladder/`). Bids never go down. Running jobs are never killed for a higher bid.
- `python -m aa_evals.common.condor_cost status | plan CPUS MEM_GB N | escalate --dry-run | submit JOB.sub`.

**Campaign checklist** (the c30 lessons, each with its measured cost). `aa_evals/campaign/launch.py` does items 1-6
and 9 by default, `aa_evals/campaign/grade.py` items 6 (for the judge) and 11; the rest are practice:
1. **Submit 4xB200 servers first** (DS, GLM), then 2-GPU, then 1-GPU servers, packed onto few nodes. In c30, DS and GLM
   waited 5.1-5.5 h for whole nodes while our own 1-GPU servers sat on 9 B200 nodes. GLM was left unfinished.
2. **Start each model's full run as soon as its own canary is clean.** Canaries cost 10.4 GPU-h, 5.7 of it idle.
3. **Check decode tok/s at the canary** against the model's reference before the full run. c30's 4B server ran
   4.5 h at half speed on a wrong attention config.
4. **Budgets at the knee, per server:**
   - Nemotron Super thrashes above about 130 sessions per B200 (c30: budget 182, 25% lower tok/s for 4.2 h).
   - Without compaction, dense Qwen3.8 is KV-bound at about 48-64 per GPU: serve Xhigh and Medium as one 2-GPU pool
     from the start.
   - Small models (Gemma, Qwen3.6, 4B) were still rising at their caps: raise their budgets (and max-num-seqs) when
     Pi slots allow.
5. **Pi capacity follows demand, not 60% of the budget per lane.** Five lanes at 60% each hold 3x the budget in idle
   slots. With a growing dispatcher (`--pi-autoscale`, the default for new dispatch dirs) pi_eval launches each lane
   with 25% of the budget in worker slots and the dispatcher adds workers to a lane whose slots limit it. Workers exit
   5 min (15 before #531) after their lane's queue is fully claimed, not while it only waits for credit, so do not over-launch with
   `--workers`. Without a dispatcher, use `pi_eval.run ... --resubmit N`.
   **Lazy first workers** (default for Pi runs created with `--dispatch` under dispatch dirs made since; recorded as
   `lazy_workers` in dispatch.json and the run's config.json): a run submits no worker at creation. The dispatcher
   counts it ready for the start-up release and gives it one probe slot, so its first grant comes in longest-first
   order and at most one granted slot waits while its workers start; that grant submits the run's first workers
   (launch-small, or `--workers`). Runs without a dispatcher, and runs under older dispatch dirs, start their workers
   at creation as before. `pi_eval.run --no-lazy-workers` (or `dispatcher init --no-lazy-workers`) keeps them eager.
6. **Release every server after 10 idle minutes,** judge and side servers included: engine idle, gate empty, and no
   attempt that can still call it. For a model's own server, `dispatcher init --release-server-after 600 --registry
   <name>` does it when all its lanes are done. Watchers must check server activity, not progress (c30's
   sanity-check server idled 4 h on 2 B200s).
7. **Never `condor_rm` workers that hold attempts to reshape them;** size them right at submission. Kills and RSS-sized
   memory holds cost c30 about 5.6 GPU-h of redone work, plus the tails.
8. **Run the coordinator as a small batch job** with its state and restart notes on /data, not as an 8-CPU/256-GB
   interactive job with notes in node-local `/tmp`.
9. **Serving shapes come from the aa-serving specs (`serving_resources`), sized from measured cgroup peaks (#378,
   aa-serving >= da6ba73, #414).** Never copy an older launch's RAM or CPUs: a 4xB200 server with 1.5 TB RAM is
   billed 96 units for RAM (64 more than its 32 CPUs), as much as its four GPUs. Every `serve_teacher` job logs its cgroup every 10 s to
   `serving/<name>/resources.<job>.jsonl`; size from its non-reclaimable peak (`hard` = rss + shmem + dirty + writeback;
   `python -I tbshared/reslog.py summary FILE`, which reads the older serving logs too). Clean page cache (the staged checkpoint: up to 490 GiB
   for DS) fills any limit but is reclaimed, never held (verified: a 2 GiB job pushed 6 GiB through the cache without
   a hold). A server held for memory (code 34) restarts with +25% RAM, at most twice (the restart costs one start-up on
   a new node, plus the attempts in flight on the old server: pi_eval ends them as `endpoint_moved` and queues a
   same-seed retry. While the server is down, `endpoints.json` is missing (deleted at SIGTERM) or stale (after a hard
   kill); workers stop claiming and pick up the new URLs on their next 30 s refresh. c30's 4B server was replaced
   mid-run this way: 4 attempts retried, the lanes carried on).

   | spec (per GPU unless noted) | was | now | measured peak (non-reclaimable) | serving cores p99 |
   |---|---|---|---|---|
   | qwen38-27b-eval-fast-text, qwen36-35b-a3b-eval-text | 16c / 320G | 13c / 210G | 168 GiB (160 GiB pinned CPU KV tier) | 1-2 |
   | qwen38-27b-eval-fast (judge), qwen36-35b-a3b-eval (images) | 16c / 320G | 15c / 240G | 184-193 GiB (condor RSS, image prompts) | 1-2 |
   | nemotron35-lightning-eval (H100) | 16c / 240G | 10c / 168G | 133 GiB (128 GiB tier) | 2.0 |
   | gemma4-26b-a4b-it-eval-b200(-text), qwen35-4b-eval(-text) | 16c / 200G, 16c / 128G | 6c / 64G | 5-6 GiB (19 / 15 with images) | 1.9-2.0 |
   | glm53-flash-eval-text, 4xB200 job | 32c / 1536G | 24c / 384G | 297 GiB (4 x 64 GiB tier + 4 API servers) | 13.6 |
   | glm53-flash-eval (images), 4xB200 job | 32c / 1536G | 28c / 448G | 347 GiB (condor RSS) | |
   | deepseek-v41-flash-1017, 4xB200 job | 32c / 1536G (the spec said 768, but a serve_teacher PROFILE `memory_gib: 1536` overrode it for every submission, c30 and M5 included; removed) | 40c / 640G | 476 GiB: Engram 193 GiB shm + 4 x 64 GiB tier + processes (long runs up to ~500) | 7.5 |

   Validated live at the new shapes on the same replay load as at the old ones (2026-11-14): no holds, decode
   0.96-1.09x the old shape's tok/s, start-up within 10 s except Lightning (224 -> 255 s on another node; its history
   is 224-252 s) and GLM (980 -> 481 s, the engine-cache fix). DS ran at 36c/600G (476.5 GiB peak, 18.2k gen tok/s at
   389 running) with no old-shape baseline; 640 GiB adds room for long runs. The image variants were sized from
   condor RSS (no live image load). Report: `/data/aai/aa_evals/audits/server_shapes_20261114/report.md`. Saving per
   c30-sized campaign: ~914 server unit-h of 4,276 (CPU/RAM unit-h 1,940 -> 1,026), ~49.5k credits at floor, ~56k at bid.
10. **Report cost as unit-hours split into CPU/RAM and GPU, plus GPU-h.** `condor_history` keeps only about 12 h here,
    so rebuild from the jobs' event logs: `/data/aai/aa_evals/audits/c30_efficiency_20261113/credits/ledger.py`.
11. **Collect, grade and judge on events, not on a timer.** c30's judge sat idle 2.33 of its 5.8 GPU-h: 0.53 h from
    ready to its first client (a 20-minute grading loop), 0.9 h between batches and 0.9 h in tails. Submit each
    suite's grading as soon as its generation (or import) is done, start the judge only when judged work waits, and
    release it after 10 minutes with no request in flight and nothing queued; never release a judge another campaign
    or session may use. Collectors: one small job per model from its first resolved Pi attempt until every attempt is
    imported (c30: 12 CPU / 32 GB at bid 61 using 0.14 cores).

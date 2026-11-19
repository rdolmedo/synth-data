# aa_evals/campaign: run an eval campaign with the #378 cost rules built in

**Evaluating a Snowball-family SFT checkpoint?** Start from [`SFT_EVAL.md`](SFT_EVAL.md): `example_sft_snowball.json`
(3 fields to edit), the commands, every default with its PR, expected throughput / completion times
(`expected/sft_snowball.json`, checked live by `gen/campaign_rate.py --expect`) and what to do when a check warns.

`launch.py` runs one campaign: every model in a config, every suite of a plan; `grade.py` collects, grades and
judges it (below). It is c30's launcher
(`aa-evals-coord/c30/ops/c30_launch.py`) made config-driven, with the c30 audit's checklist
(`aa_evals/README.md`, "Cost and storage rules") as defaults instead of reminders:

| c30 cost | What the launcher does now |
|---|---|
| DS and GLM waited 5.1-5.5 h for whole 4xB200 nodes | `servers` submits the largest servers first and the smaller ones only after those run (or `--stagger-min`) |
| canaries idled 5.7 GPU-h waiting for the other models | `watch` / `promote` start each model's full run as soon as its own canary passes |
| the 4B server ran 4.5 h at half speed | the canary gate compares decode tok/s per running-bin with `ref_decode_per_gpu` (c30's logs); below 0.5x it does not promote |
| the auto budget raised m5l's 3B from 256 to 499 agents on 2 GPUs (same tok/s per GPU, slower requests) | a fixed dispatcher budget by default (`--no-auto-budget`); `auto_budget: true` (model or campaign) opts into the hill-climb (#531 B7) |
| Pi lanes at 60% of the budget each (3x the budget in idle slots) | no worker flags: pi_eval launches 25% of the budget and the dispatcher grows lanes whose slots bind |
| 32-CPU/256-GB workers at bid 61 | no shape or bid flags: pi_eval's lean shapes and the CPU bid ladder (26 -> 36 -> 56) |
| servers ran 7-21 min past their last result | full-run dispatchers get `--release-server-after 600` (never for `shared_server` models) |
| DS launched with 1.5 TB RAM; servers' CPU/RAM unit-h were 1.9x the measured shapes' in total (1.5x Qwen/Lightning to 4x GLM) | no RAM override: the aa-serving specs' measured shapes (#378: DS 640 GiB, GLM 384 GiB text / 448 with images, 1-GPU Qwen/Lightning/Gemma/4B 6-15 CPUs / 64-240 GiB, Super unchanged for now; `aa_evals/README.md` item 9) with a memory-hold safety net (+25% RAM per hold, twice) |

## Use

```
cp aa_evals/campaign/example_c30.json /data/aai/aa_evals/plans/<name>/campaign.json   # set name, plan, models
L="python -m aa_evals.campaign.launch /data/aai/aa_evals/plans/<name>/campaign.json"
$L show --stage full                  # every command, for review
$L servers --stagger-min 10           # 4-GPU, then 2-GPU, then 1-GPU servers
$L launch <model> --stage canary      # per model, once its server is healthy
nohup $L watch --every 300 > /data/aai/aa_evals/dispatch/<name>-launch/watch.out 2>&1 &   # promote each model
$L check <model>                      # what the gate sees; `promote <model> --force` overrides it
```

Config (`example_c30.json` is c30's, with two corrections from the audit: Super's budget 130, Medium on 2 GPUs):
- `name`: the prefix of every registry, run id and dispatch dir; `{campaign}` in any string is replaced by it.
- `plan`: the item plan (#181 format). `code`: the eval checkout commands run from. `serving_workdir`: the aa-serving
  checkout servers start from.
- per model: `registry`, `teacher`, `gpus`, `gpu`, `bid`, `protocol`, `effort`, `budget` (the server's knee; see the
  checklist), `m36_key` (duration table), optional `tunables`, `ref_decode_per_gpu` (median generated tok/s per
  engine by running bin; models without one skip the decode gate), `shared_server` (never released).
- per model `suites` (optional): a list of suites, or a set name, `"lean"` (TB2.1, AutomationBench, GDPval, SciCode,
  Omniscience, GDP.pdf: the M5 Snowball SFT evals' six) or `"full"` (every suite; the default).
- `tbsched_shadow` (off by default; model or campaign): after a full launch, a read-only tbsched loop (mode shadow)
  records what it would decide next to the run, in `<dispatch_root>/<name>-<model>-sched/` (`launch.py CONFIG shadow
  MODEL --stage full` by hand); scores and scheduling unchanged.
- `release_server_after` (600), and `suites` (context window, cap, caches, bundle, canary items).
  `suites.tb_toolchain_bundle` (example: `["tb21_6h", <bundle-aptfix.tar sha256>]`, pinned to the `tb_bundle` it was
  validated with; another bundle is refused) stages TB2.1's offline verifier from that bundle's toolchain records:
  60 of 89 tasks, every one byte-identical to its tb21_6h original, verify offline (#531 B5: paired oracle runs gave
  identical verdicts on all 88 conclusive tasks; torch-tensor-parallelism's tests hang under CPU contention in about
  5 of 8 oracle attempts on either verifier path, its bootstrap is now offline). The online bootstrap of torch-class
  tasks timed out 13 times in SB12B's eval. Unset: every TB2.1 verifier bootstraps online, as before.
- `capped` (off by default): a `capped_protocol.json` (e.g. `/data/aai/m5/ladder/eval/capped_protocol.json`),
  at campaign level or per model (the model's wins; `"capped": null` opts a model out). Each suite gets
  `PI_RESPONSE_BUDGET` = warn + allowance in Super tokens (`near_window` for single-turn suites that set it, the
  benchmark's notice sentence, e.g. Omniscience's): Pi suites as pi_eval `--harbor-env`, single-turn suites in their
  submitter's environment, which forwards it into the condor job after checking that the job's interpreter can apply
  it (`aa_evals/common/response_budget_env.py`). A suite without an entry, or with a null allowance, is an error.
  `show` prints the file and each budget; `build()` records them (`capped`, each suite's `response_budget`).
  With it, `suites.wall_cap_sec` (the model-time caps: Pi wall clock, SciCode chain cap, HLE/GDP.pdf request caps,
  Omniscience timeout) defaults to 43200 s, a hang backstop, instead of `cap_sec` (#531 B6); an explicit value wins
  (at most 43200, pi_eval's limit); the Pi environment clock stays `cap_sec`. Without it the model-time caps stay
  `cap_sec` (2 h, #344). Unset everywhere: every command as before.
- `worker_profile` (off by default; per model or campaign, the model's wins, `null` opts a model out): `"slow"` runs
  TB2.1's workers on pipeline.json `tb21_slow` (32 slots / 6 CPU, filled to 96 GB, instead of 10 CPU / 160 GB) via
  pi_eval `--shape-key`, recorded in the run config (`worker.shape_check`). **Only for slow-served models: a prior TB2.1
  run of the model had tool share env_sec/(env_sec+model_sec) <= 0.25** (the tested Snowball SFT: ~0.22; Super-speed
  traces ~0.5). #531 A/B on sb2-0p33b: pass +2.4 pp [-1.6, +6.4], tool time per model request +7% [-11%, +30%] (not
  significant), workers capped in 44% vs 22% of busy minutes, worker cost -40% per worker-hour. New runs only: an
  existing run keeps its stored worker shape. Fast models keep the default.
- `request_overrides` (off by default): sampling for every suite's requests, `temperature` and/or `top_p` (e.g.
  `{"temperature": 0.6}` for a temperature-only eval), at campaign level or per model (the model's wins; `null` opts a
  model out, as `capped`), over the registry's [request] overrides. Pi suites get
  pi_eval `--request-overrides` (run config `request.overrides`), single-turn suites their client's `--temperature` /
  `--top-p` (run config `generation`); `show`, `build()` and the launch log record them. Other keys, non-finite
  numbers, a negative temperature and a top_p outside (0, 1] are errors. Unset: every command as before. A Pi run that
  already exists resumes with its stored request settings (pi_eval), so set the key before a model's first launch.
  Keep the default T 1.0 / top_p 0.95 for SFT evals: T=0.6 on the Snowball 10.73B SFT was not better (11-19; GDPval
  −68 Elo [−121, −15], other suites within noise; #480 comments 6996073899, 7015700314).

Everything it writes is on /data: dispatch dirs `<dispatch_root>/<name>-<model>[-canary]` (never reused), id files
and the launch log `<dispatch_root>/<name>-launch/events.jsonl`.

## Collection, grading and the judge: `grade.py`

`grade.py` replaces c30's `c30_collect.py`, `c30_grade.py` and `grade_loop.sh` (same CONFIG):

| c30 cost | What `grade.py` does now |
|---|---|
| collectors at 12 CPU / 32 GB, then 2 / 8, bid 61, using 0.14 cores (RSS <= 0.9 GB) | one collector job per model at 1 CPU / 4 GB on the bid ladder, submitted when the model's first Pi attempt resolves; it exits when every attempt is imported |
| collectors held "over memory limit" with a 55-82 MB RSS (dirty pages of /data writes) | GDPval attempts and renders built on node-local scratch and copied with O_DIRECT; CritPt and AutomationBench rows appended with fsync + page-cache drop; a killed collector resumes (torn last lines trimmed) |
| CritPt graders 16 CPU / 64 GB using 2.4 cores on average; GDP.pdf graders 8 / 32 using ~0 | reprobe at 4 CPU / 32 GB (`critpt.run --reprobe` default; 8 GB was held for sandbox memory); GDP.pdf, HLE, Omniscience graders 1 CPU / 4 GB; all on the ladder |
| judge idle 2.33 of its 5.8 GPU-h: 0.53 h before its first client (20-min polling), 0.9 h between batches, 0.9 h in tails | started as soon as judged work is ready, its graders submitted within a pass (60 s) of it being up, released after 10 min with no request in flight and no judged work queued |
| GDPval judged against references from a script outside the repo, on the login node | `aa_evals.gdpval.refs` (c30's draws, verified identical on three c30n pools) as one 1 CPU / 8 GB job per model |

```
G="python -m aa_evals.campaign.grade /data/aai/aa_evals/plans/<name>/campaign.json"
$G status                              # per model and suite: generation, import, grade steps, judge
$G collect --dry-run; $G grade --dry-run; $G judge --dry-run   # one pass each, nothing submitted
nohup $G watch --every 60 > /data/aai/aa_evals/dispatch/<name>-grade/watch.out 2>&1 &
```
`watch` (start it with the launcher's `watch`) runs collect, grade and judge every 60 s and exits when every suite is
graded (or failed for good) and no judge job it started is live. `--stage canary` grades canaries (off by default:
grading a few canary items would start the judge for minutes of work). One watcher or pass per campaign at a time (a pidfile in the grade dir, refreshed on every save).

What each suite needs before grading, and its grading:
- GDPval, CritPt, AutomationBench: imported by the model's collector (`<dispatch_root>/<name>-grade/collect/<model>.<stage>/`,
  a `<suite>.collected` marker each). AutomationBench is graded by its collector. GDPval: `gdpval.refs plan` + `judge`
  against `grading.gdpval_refs` (pool `<name>-refs-<model>`), exit 1 below 98% judged. CritPt: `critpt.run --reprobe
  --condor`, then `critpt.run --grade` in the watcher.
- HLE, Omniscience, GDP.pdf: generation done (the dispatcher lane is done and no job of the run is live; without a lane,
  every attempt finished), then the grade job into `<run>-graded` with the judge arm, once the judge is up.
- TB 2.1, TB-Sci-lite, SciCode: graded in their runs.
A grade job that ends without exit code 0 is resubmitted (graders resume), at most `max_tries` (3) times. A collector
still running an hour after its lanes are done is removed and reported.

The judge is never released while someone else may be using it: only a judge job this campaign started is released,
and not while another condor job or a process on this node names its registry or any arm it serves (or is a `-graded`, `aa-grade-`,
refs or judge job), nor while another campaign's lease (`<registry>/leases/<name>.json`, refreshed each pass while
that campaign has judged work) is fresh. It is never started while a job of its registry is queued, running or held.

Config (`grading`, all optional; defaults in `grade.py` DEFAULTS): `judge` (registry, arm, teacher, gpus, gpu, bid,
hours, protocol, concurrency 16, idle_release_sec 600), `gdpval_refs`, `collector` (1 CPU, 4 GB, 20 GB disk, GDPval
render concurrency 2), `collect_grace` (3600 s), `max_tries`, `runs_root` (tests only: the modules' default roots).

State: `<dispatch_root>/<name>-grade/state.json` (submissions, clusters, tries, the judge's idle timer and jobs);
events in `<dispatch_root>/<name>-launch/events.jsonl` (`collect_submit`, `grade_step`, `grade_done`, `judge_start`,
`judge_release`, ...). Not covered: alerting (`c30_monitor.py`).

## Forecasting GPU-h

New dispatch dirs use the dispatcher's default duration table, `durations_v2.json`, built from c30's measured attempts.
`m36_key` selects a model's own row. Valid keys are the table's `models` plus its `aliases`; DS is `ds-max-m35` → `ds-max`.
A model with no row is predicted from the other models' rows.
For a forecast, list-schedule the plan longest-predicted-first on the model's budget. Then divide by the run's
efficiency: c30 did not run at the ideal makespan of its own exact durations. For the six 1–2-GPU models
(validation in `/data/aai/aa_evals/audits/durations_v2_20261113/validate.md`, section 4):

| generation GPU-h, six 1–2-GPU models | forecast | actual / forecast |
|---|---:|---:|
| #344 replay (Phase A input) | 25.5 | 1.65x |
| v1 table (own M3.6 rows) | 21.4 | 1.96x |
| v2 table (own c30 rows, in-sample) | 29.6 | 1.42x |
| exact c30 durations, no stalls | 30.1 | 1.39x |
| actual | 42.0 | |

v2 closes the table error: 29.6 vs 30.1 with exact durations. Held-out repeats put the work error at 0.91–1.15x per
model. The remaining 1.39x is execution: gate pauses, the 11-11 memory holds and worker kills (redone work), drain
and stalls (the c30 audit's attribution). Forecast = scheduled GPU-h × ~1.4 until a campaign runs cleaner.
A model with no row is forecast at the other models' average speed: per model that is 0.55–1.41x off in c30 (median 1.11x).
Budget its canary as the measurement.

- Optional per-model `checkpoint` (passed to `serve_teacher submit --checkpoint`, e.g. a fine-tuned student served with
  aa-serving's `snowball-eval` spec) and `serve_extra` (extra submit arguments, verbatim). Absent: the server command is unchanged.

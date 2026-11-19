# Evaluating a Snowball-family SFT checkpoint on AAI-open v0.1 from `main`

This page covers: what to edit, the commands to run, what each default does, the numbers to expect, and what to do when a check warns.

- **Config:** [`example_sft_snowball.json`](example_sft_snowball.json).
- **Expected numbers:** [`expected/sft_snowball.json`](expected/sft_snowball.json), checked live by `gen/campaign_rate.py --expect`.
- **Measurement sources:** `/data/aai/uni/eval-runbook/` (`measure.py` → `runs.json`, `show/`).

## 1. Set up: a fresh checkout and 3 fields

```bash
N=sb3-1p0b        # campaign name = the model's id: prefixes its registry, run ids and dispatch dirs
R=/opt/aai
git -C $R/repo fetch -q origin && git -C $R/repo worktree add --detach $R/worktrees/$N-main origin/main
cd $R/worktrees/$N-main      # serving and eval both run from this checkout (config: code, serving_workdir)
mkdir -p /data/aai/aa_evals/plans/$N /data/aai/aa_evals/dispatch/$N-grade
cp aa_evals/campaign/example_sft_snowball.json /data/aai/aa_evals/plans/$N/campaign.json
```

Edit three fields in `campaign.json`:

- `name`: set it to `$N`.
- `models.sft.checkpoint`: the exported checkpoint dir, e.g. `/data/aai/tbenchtrain/snowball_sft/ckpts/<export>`.
- `models.sft.suites`: `"lean"` or `"full"`.

Optionally, `models.sft.name` sets the display name. To evaluate several checkpoints, copy the `sft` block under new keys. Each copy gets its own server and dispatcher.

Paths with `{campaign}` (`code`, `serving_workdir`, `capped`) resolve to this checkout. Nothing in it is hand-labelled with an AAI-open version: the launcher computes the label (see §2).

Pick the suite set:

| Set | Suites | Aggregate | Cost (2xB200) | Use for |
|---|---|---|---|---|
| `lean` | TB2.1, AutomationBench, GDPval, SciCode, Omniscience, GDP.pdf | 65 of 100 points | ~6 B200-h, ~3 h | SFT ladders and recipe comparisons (as the sb2-* evals) |
| `full` | lean + TB-Sci-lite, CritPt, HLE (c30 `cheap_full_v1`) | 100 points | ~11-13 B200-h, ~5.5-6.5 h | a headline point, or a comparison with the leaderboard |

The lean set drops CritPt, HLE and TB-Sci-lite (user decision, 11-17). At this scale they sat at the floor (18k-1B → 12B: HLE 5.4 → 4.5, CritPt 1.0 → 1.4), and they cost 34% of the 12B eval's GPU-h.

## 2. Check, serve, launch

```bash
PY=/opt/aai/venvs/harbor/bin/python
L="$PY -m aa_evals.campaign.launch /data/aai/aa_evals/plans/$N/campaign.json"
$L show --stage full     # every command, nothing submitted
$L servers               # serve_teacher submit: snowball-eval-main, 2xB200, bid 61
$L launch sft --stage full     # once the server is healthy (~2.5-6 min); dispatcher + lanes, detached on this node
G="$PY -m aa_evals.campaign.grade /data/aai/aa_evals/plans/$N/campaign.json"
nohup $G watch --every 60 > /data/aai/aa_evals/dispatch/$N-grade/watch.out 2>&1 &
```

**What `show` must print:**
- AAI-open `v0.1` with no deviations. Anything else means the config left the v0.1 pins, and the runs will record `v0.1+custom` (#481);
- capped protocol `sha256 403ad91eb358`;
- `--teacher snowball-eval-main --gpus 2 --bid 61`;
- dispatcher `--budget 256 --no-auto-budget --release-server-after 600`;
- for TB2.1, `--verifier-toolchain-bundle tb21_6h --shape-key tb21_slow`;
- no `--request-overrides` and no `--temperature`;
- `2217 attempts` for lean, `2805` for full.

Against the val-sb10b run (`uval`), the rendered commands differ in only three ways: the teacher, the two TB2.1 flags above, and the dropped T=0.6 overrides. The diff is at `/data/aai/uni/eval-runbook/show/diff_vs_uval.txt`.

**Where to launch from:** the devbox (gpu118). The dispatcher and the grade watcher stay there for the whole campaign; never use a login node.

**No canary for Snowball SFTs.** The sb2-* evals, uval and ucab all launched the full stage directly. A canary adds ~25 min of makespan and ~1 B200-h. It's worth it for a new model family or serving spec: run `$L launch sft --stage canary`, then `$L watch` (it promotes the model once the canary is clean).

**Abandoning a run:** use `run.py --stop` or `NO_HOLD`, not a bare `condor_rm` (#540).

## 3. Monitor

```bash
python3 gen/campaign_rate.py --dispatch /data/aai/aa_evals/dispatch/$N-sft \
  --expect aa_evals/campaign/expected/sft_snowball.json     # OK / WARN / -- per check; exit 2 if any WARN
```

Run it at +15 min. At that point:
- the budget is full (about 256 in flight);
- every lane has started;
- there are 0 infra outcomes;
- the server runs at least 80 requests per GPU, with gen tok/s per GPU in range.

After that, run it hourly. Because it exits with status 2 on any WARN, a watcher can alert on it. Expected timeline for the lean set (hours after the first start):

| Event | Expected | Measured on |
|---|---|---|
| server healthy after the job starts | 2.5-6 min; up to 10 min if it compiles; plus 0-10 min in the queue at bid 61 | 12 Snowball server jobs, 11-16..19 |
| first starts | 1.5-3 min after healthy (~5 min after the server job starts), when launched at once | uval, ucab, sb2-* |
| AB, GDPval, GDP.pdf, SciCode, Omniscience generation done | 0.9-2.15 h | uval, sb2-2p2b, sb2-4p8b |
| TB2.1 generation done (the tail; longest attempts 140-171 min) | 2.4-3.0 h | same |
| server released | 10 min after its last lane (never while work is pending or a lane is stuck) | #533, #537, #540 |
| last score (SciCode CPU grading, ~1.8 h after its generation) | 3.6-4.3 h after launch | uval: 3.8 h |

## 4. Scores

```bash
/home/user/python-3.11/bin/python3 -m aa_evals.campaign.score /data/aai/aa_evals/plans/$N/campaign.json sft \
  --json /data/aai/aa_evals/plans/$N/scores.json
```

This prints the per-suite headline scores and the aggregate (#481, `aa_evals/common/aai_open.py`). It also prints the AAI-open version each run recorded: `v0.1`, `v0.1+custom`, or `unrecorded` for runs from before #481.
- The aggregate uses the AAI-open slot weights, with GDPval scored as 100 × clamp((Elo − 500) / 2000).
- The lean set scores out of 65 points and the full set out of 100. Missing slots are listed, never filled.
- Use a numpy python: the harbor venv has no numpy.

Fixed-task 95% intervals and paired differences: `/data/aai/uni/val-sb10b/analysis/compare.py` (not on `main`).

## 5. Defaults this config uses

| Setting | What it does | PR |
|---|---|---|
| `teacher: snowball-eval-main` | snowball-eval on vLLM main: +12.9% gen tok/s at 128 agents, same served BPB, 2714/2714 identical prompt ids | #550 |
| staged venv and engine-cache seeding | a node-local venv (~120 s faster start); compile caches seeded across commits | #554, #556, #555, #558 |
| trimmed shape | 4 CPU / 64 GiB per GPU (8c/128G per server), omp 4; measured 4.1-4.6 of 8 cores and 16-19 GiB at load | #535, #547 |
| live bad nodes | `bad_nodes()` read at submit, plus the live file `/data/aai/tbench/shared/bad-nodes.txt` | #536, #554 |
| 2xB200, bid 61, fixed budget 256 | 128 agents per GPU, the measured knee; no auto budget | #534 (user 11-16) |
| server release | 600 s after every lane is done and the engines are idle; a dead lane with work left blocks the release (STUCK) | #533, #537, #540 |
| `capped` protocol | response budgets (capped-sft-eval-v1, `aa_evals/campaign/capped_protocol_v1.json`); the model-time caps become a 12-h hang backstop, the env clock stays 2 h | #449, #539 |
| `tb_toolchain_bundle` | 60 of 89 TB2.1 verifiers run offline (no proxy downloads); paired oracle gate 88/88 identical | #543 |
| `worker_profile: slow` | TB2.1 workers 32 slots / 6 CPU instead of 10 CPU (−40% worker cost, pass within noise) | #553 |
| Pi workers | lazy first fleet with a reserved share, growth over the whole budget, claim packing, 5-min idle exit, hung-session watchdog | #522, #529, #538, #548, #549, #551 |
| SciCode lane | closes before CPU grading, so it doesn't hold the server | #537 |
| sampling | T 1.0 / top_p 0.95 from the spec's generation config: no `request_overrides` | #562 (decision 11-19) |
| AAI-open label | the launcher checks the config against the v0.1 pins (plan, context, caps, capped file, TB2.1 bundle and toolchain, caches, no sampling overrides) and every run records `v0.1` or `v0.1+custom` | #481 |
| grading | `grade.py watch`: 1-CPU collectors; the shared judge starts on demand and is released after 10 min idle | aa_evals/campaign/README.md |

## 6. Opt-ins: when to use them and when not

- **`worker_profile: "slow"`** is on in this example and off on `main` by default. It fits slow-served models only, meaning a prior TB2.1 run had tool share env_sec/(env_sec+model_sec) ≤ 0.25 (Snowball SFTs: ~0.22). Set `null` for a fast-served model (e.g. a Super-speed model at ~0.5): its tool calls would wait on 6 CPUs. An existing run keeps its stored shape.
- **`tbsched_shadow: true`** is off by default. After the full launch, a read-only tbsched loop (mode `shadow`, no run hooked) records what it would decide under `<dispatch_root>/$N-sft-sched/`. It stops when the dispatcher releases the server. Pi lanes that register more than 3 min after launch are left out; re-run the `shadow` op to include them. That is the evidence the coordinator needs for the tbsched "on" canary (#451). It never touches scores or scheduling. You can also start it by hand with `$L shadow sft --stage full`. Don't use it for a run whose server is shared with another campaign (`shared_server`): tbsched would count the other campaign's traffic as unknown load. And don't start two loops for one campaign; the second one exits on the lease.
- **Not for SFT evals:**
  - `request_overrides`: T=0.6 cost GDPval −68 Elo [−121, −15] (#480). If a sampling experiment needs it, record the run as `v0.1+custom`.
  - `auto_budget`: retired (#531 B7).

## 7. Expected numbers

Everything below is for 2xB200 at budget 256. "Measured" means observed on the runs named; "est." means a measured number adjusted for what `main` changed since, with the adjustment given. Per-run detail is in `/data/aai/uni/eval-runbook/runs.json`.

**Server (`snowball-eval-main`)**

| Metric | Expected | WARN | Basis |
|---|---|---|---|
| time to healthy | 140-360 s seeded; ≤ 600 s when it compiles | > 600 s | measured: 144-243 s (0.20.1, 12 server jobs, incl. 22-101 s checkpoint staging), 354 s with slow staging; `main` staged venv 145-149 s; `main` first start of a new cache key 412 s. Seeded or not: `launch.json` `engine_cache` |
| gen tok/s per GPU (≥ 80 running/GPU) | 4.5-6.5k | < 3.5k | est.: `main` replay 5.25-5.33k at ~100 running/GPU, 5.8-6.3k at ~157; live 0.20.1 4.0-4.7k (trimmed) and 4.3-6.1k (untrimmed); +13% for `main` (#550). Not checked while HLE holds > 25% of the attempts in flight (full set): an HLE-only server read 2.4k → 0.46k at 124 → 60 running/GPU, and there's no reference for it |
| prefix-cache hit (≥ 40 running/GPU) | 0.90-0.97 | < 0.85, sustained | est.: replay 0.94-0.96 at 128-192 agents; the first `main` campaign records live mixed-lane values |
| KV use (max) / waiting | 0.3-0.8 / 0 | ≥ 0.95 / > 5% of running | measured: replay 0.44-0.75; live 0.21-0.59 |
| host CPU / hard memory | 4.1-4.6 of 8 cores / 16-20 GiB of 128 | > 90% / > 80% | measured: trimmed rtab-sb2-0p33b; every 2-GPU Snowball server 16.5-20.2 GiB |

**Per suite, lean set** (uval on `main` e678b535, and sb2-2p2b / sb2-4p8b on frozen e3dc54f9)

| Suite | Attempts | Generation done | B200-h share | Attempt p50 / max | After generation |
|---|---|---|---|---|---|
| TB2.1 | 252 | 2.4-3.0 h | 1.3-3.3 | 36-43 / 140-171 min | graded in the run |
| SciCode | 360 | 1.95-2.15 h | 1.2-1.4 | 19-26 / 117-126 min | CPU grading ~1.8 h |
| AutomationBench | 446 | 0.9-2.1 h | 0.4-0.5 | 4-6 / 17-29 min | collector, minutes |
| GDPval | 185 | 0.9-1.95 h | 0.35-0.55 | 7.5-21.5 / 37-67 min | judge 0.35-0.85 h |
| Omniscience | 900 | 1.9-2.15 h (one run: uval) | 0.3-0.4 | 1-1.6 / 16-29 min | 0.1-0.25 h |
| GDP.pdf | 74 (37 units) | 1.45-2.1 h | 0.03-0.06 | 3.6-8 / 18-24 min | 0.1-0.9 h |

The full set adds TB-Sci-lite (64 attempts, p50 78-103 min, 0.6-1.0 B200-h), CritPt (210, p50 92-114 min, 2.0-2.7) and HLE (314, p50 22-28 min, 1.6-4.0). Measured on m5l-sb-full (4xB200) and 18k-1p0b/3p0b (2xB200).

**Infra outcomes**
- Single-turn suites, AutomationBench and GDPval had none in all 11 runs.
- Other Pi outcomes were one-offs (≤ 1 per run): TB2.1 `verifier_no_reward`, `pi_backend_*` and `worker_stopped`. sb2-0p33b/1p0b's CritPt and TB-Sci-lite `worker_stopped` were the suites stopped by hand on 11-17.
- TB2.1's `verifier_bootstrap_timeout` follows the proxy download rate of the run's time window, not the code (val-codeab):
  - 0 at 6-8 MB/s (sb2-*);
  - 13 / 38 / 131 at 0.65 / 0.29 / 0.22 MB/s (sb-full, uval, ucab).
- With the offline toolchain (#543), expect ≤ 2% (est.; rtab: 0). WARN thresholds: above 5% for Pi suites, above 2% for single-turn suites.
- Infra attempts retry by themselves and are not scored. But a task whose every attempt ends infra drops out of the resolved-only score (ucab: 11 items unresolved).

**Totals**

| | Lean (measured; adjusted) | Full (est.) |
|---|---|---|
| generation makespan | 2.6-3.2 h | 5.0-6.5 h |
| server B200-h | 5.2-6.4 | 10.5-13.5 |
| all scores after launch | 3.6-4.3 h | 6.0-7.5 h |
| CPU worker unit-h (billed) | 75-95 | 250-330 |

How these were derived:
- **Lean:** uval took 2.97 h = 5.93 B200-h, and sb2-4p8b 3.12 h = 6.24 B200-h. #522 (launch order) saves ~0.4 h on uval's measured durations. The TB2.1 tail is the floor.
- **Full:** m5l-sb-full's 1317 attempt-hours, list-scheduled on 256 slots, take 5.2-5.6 h. The 2xB200 9-suite runs of weaker models took 6.2-7.7 h (12.5-15.5 B200-h).
- **CPU:** uval's TB2.1 workers took 98.5 unit-h at 10 CPU. The slow profile halves that (rtab: 51.3 vs 99.9 unit-h). The full set adds sb-full's TB-Sci-lite (156) and CritPt (74) workers.

## 8. When a check WARNs

| Check | Look at | Do |
|---|---|---|
| `startup` > 600 s | `serve.log` and `vllm.log` in the registry: staging stuck on /data, JIT compile (not seeded), node | if it never gets healthy: remove it by exact id and resubmit; report the node |
| `gen_tok_s_per_gpu` low | `job.json` (teacher, `cuda_cap`), prefix hit, KV, `host_cpu`, node | compare with `/data/aai/uni/uni-serving/sbmain/samples.jsonl`; tell the coordinator |
| `prefix_hit` < 0.85 | re-check after 5 min (launch waves read low); KV ≥ 0.9 means thrash | sustained: tell the coordinator (256 is the knee) |
| `kv_max` / `waiting` | preemptions in `vllm.log` | as above |
| `server` WARN (no metrics) | the server job (`condor_q`), the registry's `serve.log`; once the dispatcher has released it this reads `--` | if the server died with work left, resubmit it (`$L servers`): the lanes resume on the new endpoints |
| `server_busy` (0 running with work pending) | dispatcher `state.json` lanes, worker jobs (`condor_q`), the proxy | a dead single-turn lane with work left: its client log, then re-run that suite's command into the same dispatcher (`$L launch sft --only <suite>`, as M5 did for Omniscience on 11-18); more TB2.1 workers: `tb21.run ... --resubmit N` with the launch argv (val-sb10b) |
| `<suite>.starts` (no start for 45 min with budget room, or STUCK) | `lanes/<lane>/`, the client or worker jobs, `dispatcher.out` | same; the Omniscience lane hang of 11-18 was the client, not the node |
| `<suite>.infra` | TB2.1: `suites.tb_toolchain_bundle` set? Infra attempts retry by themselves, but a task with no resolved attempt drops out of the score | other suites: the outcome reasons in the run's `items.jsonl` |
| `<suite>.progress` (later than 1.5× expected) | the suite's attempt durations vs the table; one task looping (e.g. install-windows-3.11, 0 of 15 in every Snowball run) | a tail cut is a judgment call: the ladder cut at the longest prior TB2.1 attempt (170 min) once its siblings were resolved |
| `host_cpu` / `host_memory` | the reslog in the registry | memory: the net raises RAM 25% per hold; CPU: tell the coordinator (trim too tight?) |

## 9. Rollback

- **Serving, same numerics as v0 and the M5 ladder:** set `"teacher": "snowball-eval"` (vLLM 0.20.1). Once `snowball-eval` itself moves to vLLM main (next section), the 0.20.1 build stays available as `snowball-eval-v0201`. That name also covers H100.
- **Code from before the unification:** tags `frozen/aa-evals-2026-11-18` and `frozen/aa-serving-2026-11-18`. The exact v0 code (M5 ladder) will get `aai-open-v0*` tags (#481, worker aai-version).
- Check out a tag in its own worktree and point `code` / `serving_workdir` at it.

## 10. The first run on `snowball-eval-main` is its validation

`snowball-eval` switches to vLLM main only after one real campaign has run on `snowball-eval-main` (one-name plan, #550). For that run:

- Tell the coordinator the campaign name at launch.
- The sampler `/data/aai/uni/uni-serving/sbmain/watch_sbmain.py` (read-only, until ~11-22 04:20) records start-up and matched-load gen tok/s per GPU into `sbmain/samples.jsonl`. The 0.20.1 baseline is `../sbtrim/samples.jsonl`.
- The coordinator also checks gate waits, 500s, parse errors and the omp-4 start-up.
- If it is clean, uni-serving's PR moves `snowball-eval` to `main` and keeps `snowball-eval-v0201`. After that, `teacher: "snowball-eval"` and `snowball-eval-main` are the same build.
- If you evaluate after ~11-22, the sampler has stopped. The `--expect` server checks then cover the same ground.

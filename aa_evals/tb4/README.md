# Terminal-Bench 4.0 (AA Intelligence Index v5.0, 10%)

Issue #28. We evaluate a model served on our cluster on Terminal-Bench 4.0 the way Artificial Analysis does: AA's harness (stock mini-swe-agent, bash only) on the public task set. The shared runner lives in `aa_evals/common/harbor_mswea/`, and TB-Science (#29) uses it too.

```bash
PY=/opt/aai/venvs/harbor/bin/python
# smoke: a few tasks x 1 repeat
$PY -m aa_evals.tb4.run --model serving:aa-super --run-id smoke-super-v1 --repeats 1 \
    --include music-harmony,session-window-debug --aa-slug nemotron-3-5-super-vl --max-concurrent 8 --workers 1
# full run: 63 tasks x 3 repeats (sampling defaults to the serving registry's lab-recommended overrides)
$PY -m aa_evals.tb4.run --model serving:<name> --run-id <id> --aa-slug <AA slug> --max-concurrent 64
$PY -m aa_evals.tb4.run --run-id <id> --status        # progress
$PY -m aa_evals.tb4.run --run-id <id> --score         # re-score (the last worker scores automatically)
$PY -m aa_evals.tb4.run --run-id <id>                 # resume: resubmits workers if none are alive
# task validation: Harbor's oracle (reference solution) through the same workers
$PY -m aa_evals.tb4.run --agent oracle --run-id oracle-v1 --repeats 1
```

Outputs are written to `/data/aai/aa_evals/runs/tb4/<run_id>/`:

| File | Contents |
|---|---|
| `config.json` | everything that defines the run (see below) |
| `items.jsonl` | one row per task × repeat |
| `summary.json` | the scores and failure counts |
| `trials/<key>.tar.gz` | the whole Harbor trial |
| `transcripts/<key>.json.gz` | mini-swe-agent's trajectory |
| `queue/` | claims, done records and retries |

`config.json` records the git SHA and the frozen code copy, the model spec, served name and serving job, the sampling, the mini config, the dataset tag and hash, the limits, the env class and the worker resources.

Each `items.jsonl` row has `item_id`, `repeat`, `score`, `status` (`ok`, `model_error` or `infra_error`), `transcript` and `trial_archive`, plus mini's exit status, its step count, its finish reasons, and its model time and environment time.

## AA's method
From [AA's methodology page](https://artificialanalysis.ai/methodology/intelligence-benchmarking), 2026-11-02:
- **Tasks:** the full TB 4.0, 66 tasks.
- **Harness:** mini-swe-agent with "the native bash tool". *"All other agent configuration follows mini-swe-agent defaults, including the interactive mini config and prompts."*
- **Context:** no compaction or summarization.
- **Steps:** max 500.
- **Limits:** *"Task timeouts and sandbox resources follow the upstream task definitions."*
- **Grading:** each verifier runs in its own container. A verifier that exceeds its timeout counts as a failure. A task passes only if every test passes.
- **Score:** pass@1 averaged over 3 repeats.
- **General settings:** temperature 0.6 for reasoning models unless the lab recommends another; max output = the model's maximum; up to 30 API retries; e2b sandboxes.
- **Reference scores** are read from `/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json`, key `terminalbench-4-0`, and reported as `aa_reference` (pass `--aa-slug`).

## What we run
- **Data:** Harbor registry `terminal-bench/terminal-bench@4.0.0`. Tag `4.0.0` is `latest`: revision 4, content hash `sha256:39d9f44b40420cde…`.
  - Revisions 1–3 are 3.0.0 and 4.0.0-rc1/rc2.
  - The tasks sit at `/data/aai/aa_evals/data/tb4/tasks_4.0.0/terminal-bench/` (66 tasks, 480 MB, not in git). They were downloaded with `harbor download terminal-bench/terminal-bench@4.0.0`.
  - Categories: software 18, science 14, ML 11, operations 9, hardware 5, security 5, media 4. These match AA.
  - Every task has an 8 h agent timeout and a separate verifier on digest-pinned prebuilt images (`harborframework/terminal-bench@sha256:…`). 11 are compose tasks, and 8 collect artifacts from sidecar services.
- **Harness:** Harbor 0.24.0's own `mini-swe-agent` agent (`aa_evals/common/harbor_mswea/agent.py:MiniSweAgentAA`), with stock mini-swe-agent 2.4.6 (PyPI, 2026-07-23).
  - **Command:** Harbor's `mini-swe-agent --yolo --exit-immediately -c mini -c model.model_kwargs.timeout=3600 -c <ours>`. Ours sets only `agent.step_limit=500`, `cost_limit=0`, the endpoint and the sampling. `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=30`.
  - **Model class:** mini's `LitellmModel` sends chat completions with the single `bash` tool, so the server needs `--enable-auto-tool-choice` and the model's tool parser.
  - **Mini defaults left unchanged:** prompts, a 30 s timeout per command, and the run ends after 3 consecutive responses without a valid bash tool call.
  - **Install:** a pinned bundle instead of uv/PyPI in every sandbox; see Deviations.
- **Runtime:** `harbor_ns` rootless sandboxes on HTCondor CPU nodes.
  - The env is `aa_evals/common/harbor_mswea/ns_env.py:NsSidecarEnvironment`: harbor_ns plus Harbor's sidecar operations, from aa-tbsci. Without it, sidecar artifacts never reach the separate verifier.
  - Every trial gets its task's declared CPUs as dedicated cores (`ns_cpu_affinity`) and is admitted against its declared memory.
- **Sampling:** the lab-recommended settings, taken by default from the serving registry's `request.overrides`. For Super that is temperature 1.0, top-p 0.95 and thinking on. `max_tokens` is unset, so the server allows up to the remaining context. Use `--sampling` to match an AA variant, e.g. reasoning effort for "(Max)".
- **Time limit (user decision, 2026-11-02):** the task's 8 h counts environment time only (wall time minus model time), with Harbor's wall clock as a 16 h backstop.
  - Environment time is measured from mini's per-message timestamps.
  - An attempt over 8 h of environment time scores 0.
  - With 30 s commands and 500 steps, environment time stays under about 4.2 h, so the 500-step cap is what binds, as for AA with fast APIs.
- **Scoring:** reward ≥ 1 passes. A missing reward (a verifier timeout or an agent crash) scores 0 as `model_error`. pass@1 is the mean over tasks of the mean over repeats, with a 95% interval from a task bootstrap.

## Infrastructure vs. model failures (rules of #18)
These count as infrastructure failures:
- environment build/start errors, agent-setup errors and sandbox errors without a reward;
- no Harbor result, or a worker stopped or lost;
- the model backend unreachable after 30 retries (litellm connection, timeout or 5xx errors);
- an HTTP 400 that is not about context length;
- mini never starting;
- a cluster-proxy refusal in the agent's tool outputs, or in a verifier whose tests never started;
- a verifier timeout during its network bootstrap.

Infrastructure failures get up to 3 more tries. If all fail, the item is `infra_error` and excluded. A run with more than 2% of items `infra_error` is invalid (`summary.valid`).

Everything else is the model's: the step limit, repeated format errors, context overflow (no compaction, so the verifier scores the final state), verifier timeouts and the backstop.

## Deviations from AA
| | AA | Ours | Why |
|---|---|---|---|
| Sandbox | e2b | harbor_ns rootless namespaces on HTCondor CPU nodes | no Docker on the cluster |
| GPU tasks (fp8-rmsnorm-gemm, jax-speedrun-gpu, math-eval-grader) | run on H100 | excluded. `score` is over 63 tasks; `score_all_tasks_lower_bound` counts the 3 as failures over 66 | harbor_ns has no GPU passthrough (user decision 2026-11-02) |
| Time limit | 8 h of wall clock | 8 h of environment time, with a 16 h wall-clock backstop | our serving is slower than API serving (user decision 2026-11-02) |
| Disk limit | `storage_mb` | not enforced | harbor_ns overlays on node scratch |
| mini install | `uv tool install mini-swe-agent` in each sandbox, plus apt curl/git/build tools when missing | the same version from a pinned bundle in `/opt/aa-mswea`; only `tar` is ensured | proxy fragility, and the apt installs would change the task image |
| Context window | the provider's | the serving's `--max-model-len` (`config.json` `max_model_len`; aa-super has 131,072) | our serving |
| MCP (medical-claims-processing: a Playwright SSE sidecar) | unknown | Harbor's mini agent appends the server URL to the instruction; the agent can only reach it through bash | stock Harbor behavior |

## Notes
- **Decontamination:** the TB 4.0 tasks (and TB-Science) must be added to our decontamination references. They are not in #19's references, and that work is outside this issue.
- **/data file quota:** trials run in node-local scratch and land on /data as a single `.tar.gz` each, with no per-step files.

## Smoke: Nemotron 3.5 Super VL (2026-11-02, `runs/tb4/smoke-super-v1`)
- **Setup:** `serving:aa-super` (NVFP4, one B200, vLLM 0.32, `qwen3_coder` tool parser, `nemotron_v3` reasoning parser, `max_model_len` 131,072). The registry's sampling: T 1.0, top-p 0.95, thinking on.
- **Tasks:** embedding-drift-monitor, foodstuff-beta-activity, heat-pump-warranty (compose + sidecar artifacts), interleaved-vigenere, music-harmony and session-window-debug, × 1 repeat.
- **Result:** pass@1 **0/6** (AA: 0.0). 0 infra errors; the run is valid.
  - 4 attempts submitted normally after 17–28 steps with wrong answers.
  - interleaved-vigenere filled the 131k context at step 48 and ended as a context overflow. Its last three responses were cut at the end of the window and became format errors.
  - music-harmony ended at step 43 on a format hazard, described next.
- **The pipeline works end to end:** bundle install, tool calls (including parallel ones), reasoning split off by the server, separate verifier, sidecar artifacts, packing, scoring.
- **Format hazard (Qwen3-Coder XML tool calls):** Super writes tool calls as `<tool_call><function=bash><parameter=command>…</parameter></function></tool_call>`.
  - When a command contains `</function>` (MusicXML has a `<function>` element), vLLM 0.32's `qwen3_coder` parser ends the call there and returns `bash` with `{}`. Mini reports that as a format error ("Missing 'command' argument"), and three in a row end the episode.
  - A command containing `</parameter>` is silently truncated.
  - Confirmed offline with vLLM's parser. `summary.json` counts affected attempts (`n_attempts_with_empty_tool_args`).
  - This is a property of the model's tool-call format and of the parser, not of the runner. AA would see the same behavior with any server that parses this format.

### Why AA scores Super 0% (assessment)
- **Not because the harness can't drive Super.** With stock mini 2.4.6 on our vLLM, Super calls tools normally and submits. Earlier, with a 2.4.6 fork, it passed 48% of TB2.1 and 77% of TB-Lite attempts (`/data/aai/tbench/nemotron-super/pass1-6h-v1`).
- **A mini-specific integration failure on AA's side is the likeliest cause.** AA's Super is non-zero on its other tool-using benchmarks (AutomationBench 14.8, FrontierCode 2.0 6.1, Briefcase 999 Elo), which use other harnesses. It is exactly 0 only on the two mini-swe-agent benchmarks: TB4 at 0/198 attempts and TB-Science at 0/210.
- **One mechanism fits a uniform 0, and it is the leading hypothesis:** Super's raw output is Qwen3-Coder XML (checked on aa-super). On an endpoint that doesn't run `qwen3_coder`, every response comes back with no `tool_calls`, mini's `LitellmModel` raises a format error on each, and the run ends with `RepeatedFormatError` at step 3 having executed nothing.
- **TB4 is hard, though.** The smoke's 0/6 doesn't separate "harness" from "weak" on its own. A 63 × 1 Super run would; see the proposal on #28.

## Oracle validation (Harbor's oracle agent through the same runner)
- **Runs:** `runs/tb4/oracle-v1` (all 63), then re-validations after runtime fixes: `oracle-netns-v1`, `oracle-ks-x3-v1`, `oracle-v3-payments`.
- **Scoring:** TB4 runs read all four runs by default (`--oracle-run`; the last run with a task decides). `summary.json` then adds `score_excl_oracle_failures` and `oracle_failing_tasks`.
- **The headline `score` keeps every task.** AA scores all of them.

| Task | Cause | Status |
|---|---|---|
| ctr-optimization, nextjs-performance | compose `network_mode: "service:X"`, which harbor_ns lacks: main could not reach the sidecar on localhost | **fixed** by `ns_env.py`'s netns relay; both pass |
| payments-pipeline-fix | the verifier runs pip under `env -i` (no proxy variables) and aliases `kafka customer` to 127.0.0.1 at runtime (requests went to the proxy) | **fixed**: `/etc/pip.conf` proxy, plus the task's compose service names in `no_proxy`; passes |
| cumulative-layout-shift | the verifier aliases `barber-shop-data-backend` at runtime, and its headless browser sent the fetches to the proxy | **fixed**: `<local>` and the service names in `no_proxy`; passes. (Its first try was a verifier-image pull past the start timeout: infra, retried) |
| ks-solver-cpp | one verifier stall with no output (1 of 5 runs) | passes 4 of 5 (`oracle-diag-ks-v1`, `oracle-ks-x3-v1`): flaky in our runtime |
| live-database-cutover, wal-recovery-ordering | `setpriv: setresgid` EPERM under harbor_ns' single-uid emulation (nsfake) | **fixed with `--multiuid`**: harbor_ns multi-uid mode, PR #47 by aa-tbsci, oracle 3/3 and nop 0. Use `--multiuid` once #47 is merged; still failing without it |
| medical-claims-processing | the Playwright sidecar runs as `pwuser`, and the image's build-time `chown` is lost when harbor_ns unpacks its image cache (`unsquashfs -no-xattrs`), so VNC can't write `~/.vnc` | **unsupported** in our runtime (needs a harbor_ns builder change) |
| uefi-bootkit | the verifier boots QEMU with `accel=kvm:tcg` under a 300 s budget per boot; we have no `/dev/kvm` (not in the `kvm` group), and the software fallback misses the budget | **unsupported** without KVM |
| cad-model | the reference solution `pip install`s build123d unpinned; today's `ocp_gordon` needs a newer `cadquery-ocp` | task drift (the reference solution only) |
| distributed-dedup | the reference `SubmissionDedup.scala` imports `tb.dedup.task.Hashing`, which the shipped package lacks | task bug (the reference doesn't compile) |
| risk-scorer-replay | the oracle's decisions and scores differ from the expected rows in both uid modes | task or oracle side |

These were found through validation:
- three gaps that all models would hit, now fixed in the env wrapper (network_mode, pip under `env -i`, runtime host aliases);
- a classifier rule: an error before agent setup is infra.

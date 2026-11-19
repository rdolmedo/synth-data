# AutomationBench-AA

This runs Zapier's AutomationBench (v1.0.6) on our cluster and scores it as Artificial Analysis does. It is 5% of AA's Intelligence Index v5.0. Tracking issue: #30.

## Quick start

```bash
bash aa_evals/automationbench/setup.sh        # once: pinned clone + py3.13 venv (~30 s)

# smoke on the login/tmux host (light: one asyncio process; the GPU work is on the endpoint)
python -m aa_evals.automationbench.run --model serving:aa-super --run-id smoke-super --limit 12

# full run as a CPU HTCondor job, from a frozen copy of this code
python -m aa_evals.automationbench.submit --model serving:aa-super --run-id super-full --concurrency 64
```

- `run.py` re-executes itself in the harness venv (`/opt/aai/venvs/automationbench`), so any `python` can start it.
- Outputs go to `/data/aai/aa_evals/runs/automationbench/<run_id>/`:
  - `config.json`, `items.jsonl` and `summary.json`, as in `aa_evals/README.md`;
  - `transcripts.jsonl`: one line per rollout with the full message list, the upstream assertion results, debug info and timings. Each `items.jsonl` row points at its line through `transcript` and `transcript_offset`, so `f.seek(offset); json.loads(f.readline())` reads it;
  - for submitted runs, `code/<ts>.zip` (the frozen code, imported from the zip) and the `job.*` logs.
  - That is 4 files per run, plus 5 per condor submission. /data enforces a file-count quota, so do not add per-item or per-step files.
- **Resume:** rerun the same command. Rows with status `ok` or `model_error` are kept, and `infra_error` rows are rerun. A run dir refuses a different model or generation config unless you pass `--force`.

Useful flags:
- `--limit N`: a subset, round-robin over the 6 domains. By default it takes each domain's *first* tasks in upstream file order, the earliest-written ones; the Super smoke and Qwen round 1 used these. Add `--subset-seed S` for a seeded random draw within each domain. `config.json` records which one was used.
- `--tasks a,b`: run only these tasks.
- `--domains`
- `--repeats K`
- `--concurrency`: tasks per endpoint URL. Each URL is one DP rank, and tasks stay on one rank for prefix-cache locality.
- `--sampling '<json>'`: merged over the endpoint's request profile; `null` deletes a key.
- `--no-profile`, `--adapters`
- `--client-retries`: 40 by default, as upstream; AA uses 30.
- `--endpoint-wait S`: before each infra retry round, re-read a `serving:` endpoint and wait up to S seconds (default 1800) for a live heartbeat. A restarted endpoint usually comes back on another node, so the retry round follows the new URL (logged as `endpoint_moves` in `config.json`).
- `--aa-name`: the AA leaderboard name to compare against.
- `--deadline HH:MM` (or epoch seconds): the cutoff. Items still running are cancelled and recorded as `status: cutoff` with score 0. They count in the headline, which follows the no-tails rule, and are rerun on resume. `summary.json` also gives `n_cutoff` and `score_finished_only`.
- `--extend-deadline`: on resume, replace the stored deadline with a later `--deadline`. Without it the stored deadline wins. Each resume also appends the earlier attempt's model, endpoint, serving snapshot, deadline and git SHA to `config.json` → `previous_attempts`, so an endpoint switch stays on record.
- `--concurrency-file PATH`: a file holding an integer. The per-URL in-flight cap follows it, re-read every 30 s, so you can raise or lower load without a restart.
- `--serving-record NAME`: with a `url:` spec, for example when bypassing an admission gate to talk to the vLLM engine directly, snapshot `serving/NAME` into `config.json` anyway. A `url:` spec carries no request profile, so pass the full sampling in `--sampling`.
- `--note`: free text recorded in `config.json`, for example why a run used a stand-in endpoint. `config.json` also records the endpoint's `max_model_len`, read from `/v1/models`.

## M3 protocol (M3.2, #175): one command for any model

```bash
R=/data/aai/aa_evals/runs_fallback/automationbench   # pi_eval's default runs root
# any served model: its lab settings come from the serving registry's request profile
python -m aa_evals.automationbench.submit --model serving:<name> --run-id <id> --concurrency 64 --out-root $R
# Qwen3.8-27B: the profile pins xhigh, so name the effort (both top level and in chat_template_kwargs)
python -m aa_evals.automationbench.submit --model serving:<qwen> --reasoning-effort medium --run-id <id> --out-root $R
# after the trace judge wrote <run>/flags.jsonl: refresh n_flagged without model calls
python -m aa_evals.automationbench.run --summarize-only --run-id <id> --out-root $R
```

- **Request settings:** the endpoint's `request` block (`request.overrides` + `adapters`, from the teacher spec) supplies
  sampling, thinking and effort; nothing here is model-specific except the `xml_null_args` workaround for Super's
  template. `--reasoning-effort E` overrides the profile's effort in both places. DeepSeek V4.1 Flash: its profile
  pins Max. Super, Lightning, Gemma 4, Qwen3.6, Qwen3.5-4B: reasoning on via their profiles.
- **Reasoning history:** every model whose profile has no history adapter gets `mirror_reasoning_content` (vLLM ≥ 0.31
  reads earlier turns' thinking from `reasoning` only); `inline_reasoning_history` profiles (Nemotron) keep theirs.
- **Context cap:** 131,072 tokens by default (`--context-window`, M3.0 decision on #173). The client sets each
  request's `max_tokens` to the room left under the cap: the previous response's exact prompt + completion plus ~1
  token per 2.5 characters of what was added since (an upper estimate). With no room left the rollout ends as a
  context overflow (`model_error`, scored 0, as AA scores errored tasks). `--context-window 0` uses the served
  `max_model_len` instead; only for the explicit Qwen3.8 full-context calibration runs. Rows record
  `max_context_tokens`; `summary.json` gives `n_context_limit` and `n_cap_overshoot` (responses whose actual total
  passed the cap because the estimate was low; counted, not cut).
- **Repeats:** 1 (AA: 1; the M3 policy, #173 update of 11-04).
- **Text only:** every task, tool and result is text (simulated SaaS JSON); there are no image inputs, so text-only
  models (DeepSeek `--language-model-only`) run the same path.
- **`summary.json`** adds `model`, `served_name`, `reasoning_effort`, `context_window`, `context_cap`, `judge` (none),
  `code_sha`, `n_infra`, `n_cutoff` (already there) and, through `aa_evals.tracejudge.fold_summary`, `n_flagged`,
  `n_flagged_by_category` and `tracejudge` from the M3.4 trace judge's `flags.jsonl` (null until it has run; run
  `python -m aa_evals.tracejudge watch --run-dir <run> --judge serving:aa-judge-gemma4-31b` next to the run).
  Per-item results stay in `items.jsonl`. The frozen code zip includes `aa_evals/tracejudge/`.

## What runs

- **Harness:** upstream [`zapier/AutomationBench`](https://github.com/zapier/AutomationBench) at `4a8e106` (v1.0.6 plus a README commit, MIT). Clone: `/data/aai/aa_evals/data/automationbench/repo_4a8e106`.
  - We import its `AutomationBenchEnv` (a `verifiers` `StatefulToolEnv`), dataset builders, assertions and `RetryingOpenAIChatCompletionsClient`, and modify none of them.
  - The SaaS world is simulated in-process: no containers, no sandbox, no internet.
- **Toolset:** `api`, AA's choice and the upstream default. Three tools: `api_search` (BM25 over the simulated REST endpoints), `api_fetch` (method, URL, params/body as JSON strings) and `base64_encode`.
  - Tool calls are native OpenAI `tools`. The vLLM tool parser turns the model's output into them (qwen3_coder for Super).
- **Prompt:** each task's own upstream system prompt (per domain) and trigger message, unchanged.
- **Turn cap:** 50 model responses, as AA and upstream use.
- **Data:** the 600 public tasks, 100 each in sales, marketing, operations, support, finance and hr.
  - The 200-task `simple` warm-up domain is skipped; upstream does not score it either.
  - Task generation is deterministic (noise is seeded by `example_id`). `config.json` records a sha256 over the upstream task-contract fingerprints of the tasks run.
  - Eval data: do not train on it, and do not commit it.

### Request profile and reasoning history

With `serving:<name>`, the endpoint's `request` block in `endpoints.json` (written by `serve_teacher` from `synth/teachers/<teacher>.toml`) supplies the default sampling overrides and history adapters. This is the same convention our synthesis clients use.

For Super that means:
- temperature 1.0 and top_p 0.95, the model card's recommendation for text;
- `enable_thinking`;
- the `inline_reasoning_history` adapter.

The upstream client replays each earlier turn's reasoning as `reasoning_content`, but vLLM 0.32 drops that field before the template sees it. Without the adapter, Nemotron would see `<think></think>` for every earlier turn. The adapter puts the reasoning back inline as `<think>…</think>`, which the template renders as designed; it keeps history thinking after the last user turn, and every turn here is after the only user turn. `mirror_reasoning_content` (GLM / vLLM 0.31) copies it into `reasoning` instead.

**Qwen3.8 (`qwen38-27b`):** `mirror_reasoning_content`, on by default (since M3.2 for every model without a history adapter; before, through `EXTRA_ADAPTERS` for Qwen only). Qwen's template reads `reasoning_content` and, with `preserve_thinking: true` (in the serving profile), renders it for every earlier assistant turn. We probed vLLM 0.32's `/tokenize` on the served model:
- `reasoning_content` alone renders as an empty `<think>\n\n</think>`;
- `reasoning` renders the text.

The adapter therefore copies the field into `reasoning`. Inlining it, as `inline_reasoning_history` does, would be wrong for Qwen: the template adds its own `<think>` block, so the thinking would be doubled. Qwen's template serializes non-string tool arguments with `tojson`, so a null replays as `null` and `xml_null_args` is not needed.

`max_tokens` is the room left under the context cap (above; before M3.2 none was sent and the server's remaining context applied). This follows AA's rule of using the maximum the lab discloses, within the M3 128k cap.

### Null tool arguments (`xml_null_args`, on for Super)

`api_fetch` takes `params` and `body` as optional JSON strings. Nemotron's qwen3_coder-style XML template renders every argument with `| string`, which breaks on null:
- The model writes `<parameter=body>null</parameter>`, and vLLM's parser turns that into JSON `null`. The call works.
- When the call is replayed in history, the template renders `null` as the Python text `None`.
- The model then copies `None` into every later call. The parser passes `None` through as the *string* `"None"`, and the tool answers `Invalid JSON`.

In the first Super smoke (`smoke-super-12`, adapter off), every task's first `api_fetch` sent a real `null` and nearly every later call sent `"None"`. That produced 293 string-`None` arguments and 234 `Invalid JSON` tool errors in 552 calls. In several transcripts the model then concluded that `api_fetch` was "unavailable".

This is a defect in the serving stack: the template and the parser disagree about null. It is not a gap in the model's ability. The `xml_null_args` history adapter therefore replays a null argument as the literal text `null`, which is exactly what the model wrote. Only the request sent to the server changes; tool execution and the stored transcript stay as they were.

`EXTRA_ADAPTERS` in `run.py` enables it per served name. It is only correct for templates that render arguments as raw text (the qwen3_coder XML format). JSON-rendering templates such as GLM's and DeepSeek's must not use it. `python3 -m aa_evals.automationbench.inspect_run <run_dir>` reports `bad_json` errors and string-`None` arguments, so a new model's smoke shows whether it needs the adapter.

### Scoring (`scoring.py`)

AA's rules, from the methodology page, 2026-11:
- Every assertion is an **objective** (it must be made true) or a **guardrail** (it passes initially and must not break).
- A task scores 0 if any guardrail breaks. Otherwise its score is the fraction of objectives met.
- The headline is the mean over tasks. Errored tasks score 0.

Upstream does not label assertions. We take an assertion that already passes in the initial world as a guardrail, and every other assertion as an objective. Upstream's own `partial_credit` uses the same initial-world check, but breaking a guardrail there only adds one failure to the denominator.
- On the public set this gives 5,729 objectives and 3,854 guardrails. 551 of 600 tasks have at least one guardrail, every task has at least one objective, and 29 informational (`scored: false`) assertions are ignored.

`summary.json` also reports:
- `tasks_completed_pct`: AA's "Tasks Completed". It equals Zapier's strict pass rate, `zapier_pass_rate_pct`, which is Zapier's own headline.
- `upstream_partial_credit_pct`, `objectives_completed_pct`, `guardrail_violations_per_task` and the per-domain scores.

### Error classification

| Status | When | Scored as |
|---|---|---|
| `ok` | the rollout ended normally: the model stopped calling tools, or the 50-turn cap | the AA task score |
| `model_error` | the model side failed: context overflow, a 400 such as unparseable tool-call history, empty responses after retries | 0, as AA scores errored tasks; `final_state_score` keeps the graded end state |
| `infra_error` | the endpoint failed after 40 backoff retries (connection, 5xx, 429), an auth or 404 error (wrong URL or name), or the harness crashed outside an assertion handler | excluded; rerun in up to 2 extra rounds; reported in `n_infra_errors` |
| `cutoff` | no result at the `--deadline` | 0; reported in `n_cutoff`, next to `score_finished_only` |

An exception raised by an assertion handler while grading (e.g. a `ValueError` parsing model-written state) fails that assertion. Upstream already does this for `AttributeError`, `TypeError` and `KeyError`. It is a model or task outcome, never infra, and is listed per item in `grading_exceptions`; `summary.json` gives `n_items_with_grading_exceptions`.

The per-request HTTP timeout is 24 h (`REQUEST_TIMEOUT_S`), following the #66 audit rules:
- time a request spends queued at an admission gate does not count against it;
- a long generation from a healthy server ends at `max_tokens` (the remaining context) or the 50-turn cap, so it is scored as a model outcome and is never an infra timeout;
- context overflow is a `model_error`, and items unfinished at `--deadline` are `cutoff`, which scores 0.

A run with more than 2% `infra_error` sets `"valid": false` and exits with code 3.

## Smoke results (Nemotron 3.5 Super VL, `serving:aa-super`, 2026-11-02)

60 tasks (10 per domain, stratified), 1 repeat, concurrency 32. Both arms used the same tasks, the endpoint profile (T=1.0, top_p 0.95, thinking on) and `inline_reasoning_history`; they differ only in `xml_null_args`.

| Run | `xml_null_args` | AA score (95% CI) | Tasks completed | Objectives met | Tasks with a violation | Turn cap hit | Tool calls | `Invalid JSON` errors | String `"None"` args |
|---|---|---|---|---|---|---|---|---|---|
| `smoke-super-60` | on (default) | **34.3** (23.5–45.2) | 20.0% | 63.2% | 43.3% | 0% | 1,542 | 20 | 15 |
| `smoke-super-60-noNullFix` | off | 24.1 (14.2–34.0) | 11.7% | 55.2% | 40.0% | 10% | 2,205 | 829 | 1,121 |

- AA reports **14.8** for this model, on its private 657-task set, which Zapier says is harder than the public one.
- Paired over the 60 tasks, the adapter adds +10.2 points (95% CI +3.3 to +17.1; 11 tasks up, 1 down).
- 0 infra errors and 0 model errors in both arms. Wall time was ~6 min with the adapter and ~23 min without it, because the broken calls drive longer rollouts.
- Per-domain scores with the adapter: finance 18.3, hr 41.8, marketing 7.3, operations 55.5, sales 44.2, support 38.8.
- Our score sits above AA's even with the adapter off. The plausible causes are the easier public set and reasoning history being kept; the 60-task CI is wide.

## Round 1: Qwen3.8 27B (Medium), 2026-11-03 (#66)

The run used 60 tasks: the **first 10 per domain in upstream file order** (the default `--limit`, not a random draw), the same tasks as the Super smoke. It ran with 1 repeat and the lab's sampling (T=1.0, top_p 0.95, top_k 20). Reasoning was set to medium (`reasoning_effort` and `chat_template_kwargs.reasoning_effort`), as the user decided. The endpoint was `serving:aa-qwen-any`, the `qwen38-27b-eval` spec on 1×B200 (262k context), standing in while `aa-qwen` was queued. The server was shared with the other benchmarks' rounds, and our runs had 32 and 24 requests in flight.

| Run | History adapter | AA score (95% CI) | Tasks completed | Objectives met | Tasks with a violation | Wall |
|---|---|---|---|---|---|---|
| `r1-qwen-medium-60` | `mirror_reasoning_content` (default) | **55.0** (44.2–65.8) | 28.3% | 84.9% | 28.3% | 69 min |
| `r1-qwen-medium-60-noMirror` | none: history thinking renders empty | 49.7 (38.4–60.9) | 26.7% | 82.2% | 31.7% | 47 min |

- AA reports **40.3** for Qwen3.8 27B (Medium) and 48.2 for (Xhigh), on its private set. Our public-set score is +14.7 above AA's. Super showed a similar gap: 34.3 here against AA's 14.8, or 24.1 against 14.8 without its null-argument fix. That is consistent with Zapier's note that the private set is harder.
- Paired over the 60 tasks, the history adapter adds +5.3 (95% CI −5.1 to +15.7; 12 tasks up, 8 down). That is not significant at n=60.
- Gates:
  - 0 infra errors and 0 model errors. All 60 rollouts ended with a final text answer, none hit the 50-turn cap, and none was truncated.
  - Every assistant turn carried reasoning, and no `<think>` leaked into the content.
  - Tool calls parsed cleanly: 0 leaked, no string-`None` arguments, and 22 `Invalid JSON` errors in 1,945 calls.
- What the `Invalid JSON` errors are:
  - 19 are model-written `body`/`params` strings that are not valid JSON. 15 of these are in `finance.monthend_journal_entries`, where a 4-level-deep Gmail payload closes only 3 braces and the model retried it repeatedly.
  - The other 3 are the simulated Gmail API rejecting an unknown `body` field, which is API misuse (audit recount).
  - We checked for a parser bug: 19 other bodies that end in 4 or more closing brackets parse fine, and the same pattern appears in Super's runs. We count these as model errors.
- Failure taxonomy, from the 10 lowest-scoring transcripts:
  - The zeros are guardrail violations, often with every objective met: logging a decoy order (`ORD-45210`), enrolling excluded contacts, emailing a contact who should not get mail, creating a calendar event for an excluded window, or adding a note to the wrong one of three same-name contacts.
  - The other tool errors are API misuse: unknown campaign or spreadsheet IDs, unsupported GAQL resources, or an app the task has not connected.
  - No harness, template or context failures.
- Cost: on average about 12k output tokens and 280k input tokens per task. The median rollout took 259 s and the slowest 3,814 s (34 turns, 75k output tokens). Streams decoded at about 23 tok/s on the shared B200.

## Round 3: Qwen3.8 27B, full public 600 (#66)

| Run | Reasoning effort | AA score (95% CI) | Tasks completed (= Zapier pass rate) | Objectives met | Tasks with a violation | Turn cap hit | AA reference |
|---|---|---|---|---|---|---|---|
| `r3-qwen-medium-600` | medium | **57.2** (53.8–60.7) | 35.2% | 83.2% | 29.2% | 1.5% | 40.3 |
| `r3-qwen-xhigh-600` | xhigh | **59.9** (56.4–63.3) | 36.5% | 83.9% | 27.2% | 2.7% | 48.2 |
| `r1-qwen-medium-60-noMirror` (round 1, first 10 tasks per domain; no 600-task run) | medium, reasoning-history fix off | 49.7 (38.4–60.9) | 26.7% | | | | |

- **Medium run:** 600 tasks, 1 repeat, the reasoning-history adapter on.
  - 0 infra errors, 0 model errors, 0 cutoffs, 0 grading exceptions.
  - None of the 10,146 assistant turns was truncated, every one carried reasoning, and none leaked `<think>`.
  - 9 tasks hit the 50-turn cap.
  - Tool calls: 116 `Invalid JSON` errors in 20,851 calls, and no string `"None"` arguments.
- **Per domain:** finance 61.3, hr 54.2, marketing 64.1, operations 64.0, sales 51.6, support 48.2.
- **Agreement with round 1:** on round 1's 60 tasks this run scores 52.4, against 55.0 in round 1, which is within noise. The two runs used different serving and run on different days.
- **Same-split anchor:** a Zapier pass rate of 35.2% sits next to Claude Sonnet 5 (max) at 34.7% in Zapier's public-600 table. The models differ, so this is a sanity check, not a validation.
- **Serving:** `config.json` → `previous_attempts` records the run's three attempts.
  - 75 tasks ran on `aa-qwen-qa2`, until it was starved at the admission gate behind other benchmarks' long conversations.
  - 142 ran on the dedicated `aa-qwen-ab`, through its gate.
  - 383 ran on the same engine with the gate bypassed (`url:`), as the coordinator instructed because of gate live-lock bugs.
  - All three use the same spec (`qwen38-27b-eval-fast`) and identical sampling.
  - In-flight rollouts that were cancelled at each switch were rerun from the start. Nothing was scored twice.
- **Cost:** 7.0M output and 197M input tokens, about 11.7k and 329k per task.
- **Xhigh run:** 600 tasks, 1 repeat, the reasoning-history adapter on, same sampling as medium.
  - 0 infra errors, 0 model errors, 0 cutoffs, 0 grading exceptions.
  - None of the 11,122 assistant turns was truncated, every one carried reasoning, and none leaked `<think>`. 592 rollouts ended with a final text answer.
  - 16 tasks hit the 50-turn cap.
  - Tool calls: 191 `Invalid JSON` errors in 22,463 calls, and no string `"None"` arguments.
  - Per domain: finance 60.2, hr 61.1, marketing 64.3, operations 60.1, sales 58.1, support 55.3.
  - Serving: all on `aa-qwen-ab`'s engine with the gate bypassed. A first attempt overloaded the engine (KV 0.99, 44 waiting) alongside medium's last tasks; it was stopped with 25 tasks kept and resumed under a shared in-flight budget (`--concurrency-file`, medium first) that held vLLM waiting near 0. `config.json` → `previous_attempts` records it.
  - Cost: 8.8M output and 247M input tokens, about 14.6k and 412k per task. Wall time 77 min for the resumed attempt.
- **Medium vs xhigh:** paired over the 600 tasks, xhigh adds +2.7 (95% CI −0.4 to +5.7; 116 tasks up, 94 down). Not significant. AA reports +8.0 between the two settings on its private set. On round 1's 60 tasks xhigh scores 59.6.
- **Same-split anchor (xhigh):** a Zapier pass rate of 36.5%, between Claude Sonnet 5 (max, 34.7%) and GPT-5.6 Terra (37.2%).

## Pi harness (#348, c30)

User decision 2026-11-11: AutomationBench runs under the TB 2.1 Pi harness, "so that it runs faster. performances may
drop but that is ok, as long as relative rankings are similar". The upstream `verifiers` path above stays as is
(`run.py` / `submit.py`; M3.5/M3.6 reproduce with it). Code: `pi/`.

```bash
python -m aa_evals.automationbench.pi.tasks build        # once: 600-task bundle, /data/aai/aa_evals/data/automationbench/pi_bundle
bash aa_evals/automationbench/pi/image/build_world.sh OUT  # once (CPU job): world/agent SIFs, then pi_eval.sif_base
python -m aa_evals.automationbench.pi.run init --run-id ID --model serving:NAME --tasks @IDS_FILE --repeats 1 \
    [--reasoning-effort E] [--dispatch DIR --agent-wall-cap-sec 7200 ...]   # unknown flags go to pi_eval.run
python -m aa_evals.automationbench.pi.run collect --run-id ID [--follow]    # grade -> items.jsonl, transcripts.jsonl, summary.json
python -m aa_evals.automationbench.pi.run status --run-id ID
python -m aa_evals.automationbench.pi.replay RUN_DIR      # equivalence check against a stored upstream run (CPU)
```

| | Upstream path (M3.5/M3.6) | Pi path |
|---|---|---|
| Harness | `verifiers` StatefulToolEnv, native tool calls, 50-turn cap | Pi 1.2.0 `pi-cont16k` (`pi180`), bash tool, no turn cap, **no compaction**: the episode ends at the 131,072-token window |
| Tools | `api` toolset (api_search, api_fetch, base64_encode) as OpenAI `tools` | the same three tools as a shell command, `ab <tool> '<json>'` (or `-` and stdin); their upstream schemas are in the prompt and in `ab tools` |
| World | in-process `WorldState` | the same `WorldState` in a compose sidecar `world` (`pi/image/world_server.py`), its own sandbox: the agent changes it only through the tools; calls are serialized and the state persists for the episode |
| Prompt | the task's system prompt + trigger message | the same texts in one instruction: system prompt (its "~50 tool-using turns" budget sentence replaced by "Avoid duplicate searches.", since there is no turn cap), a Tools section, then the request verbatim |
| Output per request | the room left under the cap | 16,384 tokens (continuation in Pi's history) |
| Context full | `model_error`, scored 0 | the episode ends and is graded on the world it left |
| Time limit | none (24 h request timeout) | 2 h of environment time (`agent_cap_sec` 7200); with the #344 dispatcher also a 2 h wall cap |
| Grading | assertions on the in-memory world; AA score (`scoring.py`) | the same code (`pi/run.py grade`) on the world the sidecar wrote after its last call (`/state/world.json` plus `hidden.json`, collected by Harbor) |
| Infra | 40 client retries, then `infra_error` | pi_eval's same-seed retries (up to 6), then `infra_error` (excluded) |
| Seeds | none | namespace `automationbench-pi-v1` |

- **Ids** are upstream's `<domain>.<task>` (`items.jsonl` `item_id`), as in the upstream path. `--tasks` takes a
  comma list or `@FILE` (one id per line, `#` comments). With `--dispatch`, `--dispatch-suite automationbench` is
  added, so the #344 duration table's AutomationBench entries apply.
- **Images** (`pi/image/`): `world.def` = python:3.13-slim + upstream's `schema/`, `tools/`, `utils/` and pydantic at
  the AB venv's versions (no task builders, assertions or rubric); `agent.def` = python:3.13-slim (bash, python3).
  Both are harbor_ns bases (`aa-evals.local/automationbench-{world,agent}:v1`). `world_server.py` and `ab` ship in
  each task of the bundle (manifest `scripts` sha256), so changing them needs a new bundle, not new images.
- **Hidden world state.** Two api tools keep state outside the pydantic fields, which `model_dump_json` drops:
  `google_sheets._updated_row_keys` (read by the `google_sheets_row_(not_)updated` assertions) and
  `google_ads._offline_jobs`. The world service saves every such attribute in `/state/hidden.json`, and the grader
  restores them (`restore_hidden`). Without this, a row-update guardrail broken by the model graded as intact (found
  by the replay: Qwen3.8 Xhigh `finance.capex_approval` scored 1.0 instead of 0.0).
- **Equivalence check (`pi/replay.py`).** It replays a stored upstream rollout's tool calls through the world service
  (in order, as `ab` sends them; calls that did not parse, or that upstream never answered because the rollout hit
  its turn cap, are skipped), compares every result with the stored tool message (random ids, Slack `ts`, epoch and
  ISO times aligned and mapped into later arguments), and grades the final world with `pi/run.py grade`.
  Result on six M3.6 runs (Qwen3.5-4B, Qwen3.8 Xhigh, Gemma 4 26B, Super VL, Qwen3.6, Lightning; 3,600 rollouts,
  112,070 executed calls; `/data/aai/aa_evals/code/c30ab/replay3/`): **3,595 grade identically** to the stored
  score (objectives, guardrails, upstream partial credit and strict pass). The 5 others:
  - 2 (`sales.zoom_crm_meeting`, Qwen3.6 and Xhigh): Zoom meeting ids are random integers of varying length, which
    the replay did not yet map. With them mapped, both match (re-checked).
  - 3 (`sales.docusign_deal_workspace` Super, `support.zendesk_sf_case_sync` Qwen3.6, `support.hiver_inbox_report`
    Qwen3.5-4B): an assertion's `text_contains` is a single digit ("4", "5"), and the model pasted random ids
    (workspace UUIDs, case ids) into the checked Slack message, so the outcome depends on the ids drawn. That is
    upstream's own randomness, the same in both paths (repeated replays of the same calls flip between the scores).
  Tool results: 112,018 of 112,070 match after id alignment, the residue being list orderings and ids that follow an
  unaligned one.
- **Collect** writes the upstream run layout (`items.jsonl`, `transcripts.jsonl` with the Pi session, the `ab` call
  log and the assertion results; `summary.json` via `run.summarize`), so `run.py --summarize-only` and the trace
  judge read it. Rows add `harness: pi`, `end` (`finish` / `context` / `time_limit`) and `n_tool_calls` /
  `n_tool_errors` from the world service's call log.

### Validation smoke: Qwen3.5-4B, 96 tasks (2026-11-11)

`c30ab-smoke2-q35-4b-96`: 16 tasks per domain drawn from the c30 plan's 446 ids (`random.Random(348)`, ids in
`/data/aai/aa_evals/smoke/c30ab/smoke96.txt`), 1 repeat, `serving:c30ab-q35-4b-text` (spec
`qwen35-4b-eval-text`, 1×B200, text-only, its turn gate), bundle `automationbench-pi-v2`. Compared with M3.6's
upstream-path run of the same model on the same 96 tasks:

| | Pi path | M3.6 upstream path |
|---|---|---|
| AA score (95% CI) | **12.7** (7.9–17.5) | 10.1 |
| Paired difference | +2.6 (bootstrap CI −1.4 to +7.0); 19 tasks up, 11 down | |
| Tasks completed / with a guardrail violation | 1 / 17 | 0 / 20 |
| Objectives met | 199 of 924 | 214 of 924 |
| Ends | 85 finish, 11 context (all scored 0), 0 time limit | 69 stopped, 26 at the 50-turn cap (score 1.1), 1 overflow |
| Median turns / world calls | 74 / 59 | 36 |
| Median peak context | 60k | 33k |
| Wall per task, median / mean / max | 174 s / 222 s / 1,080 s | 230 s / 276 s / 2,093 s |
| Tokens per task: output / prompt (mostly prefix-cached) | 19k / 4.8M | 13.3k / 0.74M |

- No infra errors, model errors or grading exceptions. The whole smoke took ~20 min of wall time on one B200.
- **Transcripts and scored states read.** For example, `sales.soft_matching` (1.0) makes exactly the one
  Salesforce PATCH and the note its three objectives need. `marketing.content_expiration_alert` (12 of 16
  objectives, 0) sends one alert that also goes to partnerships, legal and marketing, which are the three broken
  `gmail_message_not_sent_to` guardrails; the model's own summary lists those recipients.
- **Context ends** are runaway loops that upstream's 50-turn cap would have cut, e.g. `finance.grant_expense_tracking`,
  702 world calls before the window filled. They score on the world they left: here 0 objectives met.
- **First smoke (`c30ab-smoke-q35-4b-96`, bundle v1, stopped after 6 tasks).** With the tools described as a bare
  list, the 4B model called `ab` as a native tool ("Tool ab not found") and searched the sandbox's filesystem for the
  task's "files" (they live in the apps). The v2 instruction says where the data is and that the tools are `ab`
  subcommands run in bash, with shell examples, and `ab` checks the tool name before calling. Over v2's first 20
  attempts there were no native `ab` calls, and 53 interface errors (bad quoting or JSON) in 1,422 `ab`
  commands. Some attempts still look around the filesystem or draft emails into `/tmp`; that is the model's own
  wandering, since the prompt is explicit.

## Reference points

**AA's number is not a validation.** AA scores the private 657-task split, which Zapier says is harder, and sometimes makes harder still between versions. Both models we ran score well above AA:
- Super VL: +19.5 with `xml_null_args`, +9.3 without it;
- Qwen3.8 27B (Medium): +14.7 on round 1's 60 tasks, +16.9 on the full 600;
- Qwen3.8 27B (Xhigh): +11.6 on the full 600.

Part of the gap may come from our serving choices, for example reasoning history (+5.3 for Qwen, not significant). The gap to AA is reported, but it cannot confirm or refute this pipeline.

**Same-split anchor:** Zapier's README table of strict pass rates on the public 600, which our `tasks_completed_pct` / `zapier_pass_rate_pct` measures. The models are different, and none of them is on AA's leaderboard under the same name, so this is a cross-model sanity check, not a same-model match:

| Model (Zapier, public 600, highest effort) | Pass rate |
|---|---|
| Claude Opus 5 (max) | 50.3% |
| Kimi K3 (max) | 46.7% |
| Claude Fable 5 (max) | 46.2% |
| GPT-5.6 Sol (max) | 45.8% |
| Gemini 3.6 Flash (high) | 45.0% |
| Claude Opus 4.8 (max) | 41.0% |
| Gemini 3.5 Flash (high) | 38.3% |
| GPT-5.6 Terra (max) | 37.2% |
| Claude Sonnet 5 (max) | 34.7% |
| GLM 5.2 (max) | 26.2% |
| *ours: Qwen3.8 27B (Medium), round 1, 60 tasks* | *28.3%* |
| *ours: Nemotron 3.5 Super VL, smoke, 60 tasks* | *20.0%* |

## Deviations from AA

| Aspect | AA | Ours | Effect |
|---|---|---|---|
| Task set | private held-out, 657 tasks, v1.0.6 | public 600, v1.0.6 | Zapier says the private set is "purposely harder" and expects only directional agreement; expect our scores to be **higher** than AA's |
| Harness/client | AA's harness. Its page describes the upstream environment, but AA's client code is unpublished | upstream env and upstream retrying chat-completions client against our vLLM | small |
| Model serving | the provider APIs AA uses | our vLLM (Super: NVFP4 checkpoint, qwen3_coder tool parser, nemotron_v3 reasoning parser) | quantization and parser differences are possible |
| Model serving, Qwen3.8 27B | the provider APIs AA uses (Alibaba's own serving) | vLLM 0.32, FP8 checkpoint `/data/aai/models/Qwen3.8-27B-FP8`, fp8_e4m3 KV cache, MTP-3 speculative decoding, 1×B200, qwen3_coder and qwen3 parsers, 262k context. Round 1 (`aa-qwen-any`, spec `qwen38-27b-eval`) and round 3 (`aa-qwen-qa2`, spec `qwen38-27b-eval-fast`) differ only in engine tuning: qa2 adds `--mamba-ssm-cache-dtype bfloat16`, 160 GB CPU KV offload, `--gpu-memory-utilization 0.94` and an admission gate. Sampling is identical. `config.json` → `serving` records the launch argv, job, GPU and teacher spec (with sha256) | FP8 weights, FP8 KV cache and speculative decoding can shift scores slightly; round 1 vs round 3 mixes serving and sampling noise |
| Sampling | lab-recommended, with temperature 0.6 as the fallback for reasoning models | the endpoint's request profile, i.e. lab-recommended (Super: T=1.0, top_p=0.95) | none intended |
| Max output | the maximum the lab discloses | the room left under the context cap | none intended |
| Context | the model's native context | **capped at 131,072 tokens** (M3.0 protocol, #173); a rollout that runs out of room is a `model_error` (0). Full native context only for the Qwen3.8 calibration runs. The #66 round-3 finals ran uncapped (262k) | depends on the model: at most 26 of 600 tasks for the other models, but DeepSeek V4.1 Flash Max (M3.5) hit the cap on 275 of 600 (192 overflowed and scored 0, 83 had the last turn cut off at the remaining room); `n_context_limit` counts the cases |
| Reasoning history | | `mirror_reasoning_content` for every model without a history adapter (M3.2) | see below |
| Null tool arguments in history | unknown. A provider serving the official template through vLLM would show the same `None` bug | the `xml_null_args` adapter re-renders null as the model wrote it (Super) | ≥ AA if AA's provider has the bug |
| Reasoning history | unknown, provider-dependent | kept: earlier turns' thinking goes back to the model as its template expects (Super `inline_reasoning_history`, Qwen `mirror_reasoning_content`; see above) | likely ≥ AA. Qwen medium: +5.3 (CI −5.1 to +15.7) on 60 tasks |
| Objective/guardrail labels | AA's internal | inferred from the initial world, as upstream does | probably identical |
| Context overflow / model errors | "errored tasks score 0" | 0 (`model_error`); upstream would instead grade the partial end state | matches AA |
| Retries | up to 30 per API call | 40 (upstream), then an `infra_error` excluded from the score | none intended |
| Repeats | 1 | 1 by default (`--repeats`) | — |

## Lessons

- Build the venv with uv's cache on node-local `/tmp`. `/data` has no `flock`, so `uv sync` with a `/data` cache fails with `os error 38`.
- The shell has `http_proxy` set. `run.py` adds the endpoint hosts to `no_proxy`, because otherwise localhost and cluster URLs go to the squid proxy (which returns a 403 HTML page). Condor jobs run with `getenv = false`, so they have no proxy at all.
- Upstream wraps transport errors in `vf.ModelError`, so classification walks the `__cause__` chain. Auth and 404 errors escape the rollout entirely; they are caught and marked `infra_error`.
- Upstream's own `_usage` counter skips the final turn, so token counts come from `verifiers`' usage tracker.
- Endpoints move. aa-super was restarted on another node at the moment a smoke job started; the job had resolved the old URL and spent 3.6 h in upstream's 40-attempt backoff (three rounds) before scoring all 60 items `infra_error`. Retry rounds now re-resolve the endpoint. A dead endpoint shows up in `job.*.out` as `[retry] chat: attempt N/40 after APIConnectionError`; check `endpoints.json` when you see it.
- /data's file-count quota (41.9M files, strictly enforced) is why transcripts are packed into one JSONL and the frozen code is a single zip.

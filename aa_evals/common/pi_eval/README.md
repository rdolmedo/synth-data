# aa_evals/common/pi_eval: our Pi protocol against a serving registry

Runs eval_q35's Pi protocol (harness `pi180`, the protocol of our tb21p180 / tbmix evals) on a model served elsewhere
(`serving:<name>` registry), with no eval_q35 serving job. User decision 2026-11-02 ~23:40 (#66): TB 2.1, TB-Science
and TB-Science-lite use this protocol. The AA-faithful mini-swe-agent runner (`../harbor_mswea`) stays available.

```
python -m aa_evals.common.pi_eval.run --bench B --run-id ID --model serving:NAME --reasoning-effort medium \
    (--bundle SIX_HOUR.tar | --tasks-tar TASKS.tar) --seed-namespace NS [--include a,b | --limit N] [--attempts 3] \
    [--max-in-flight 48] [--agent-cap-sec 7200] [--[no-]multiuid] [--env-class MOD:CLASS] [--agent pi|oracle]
python -m aa_evals.common.pi_eval.run --bench B --run-id ID --status | --score | --stop | --resubmit N
```

Benchmarks wrap it with their defaults (`aa_evals/tb21/run.py`).

## Offline agents (M3.5/M3.6)

New runs default to `--no-agent-internet`, using `NsSidecarEnvironment`. The run config and
protocol record `agent_internet=false`. Setup runs before isolation; grading may use networking
under the task's evaluator policy. Agent processes, including background children, retain their
offline network namespace through teardown. Local task services and compose service traffic remain
available. Task dependencies must be installed during evaluator setup; agent-initiated downloads fail.

Pi reaches its dedicated model proxy through a fixed local HTTP relay. Only model/tokenization API
routes are accepted, with no redirects, query strings or remote media URLs. This preserves model
connectivity without giving tools a general network proxy. The full agent launch shell drops
`CAP_SYS_ADMIN` and `CAP_SYS_PTRACE`, clears ambient capabilities and enables no-new-privileges;
these restrictions also apply to its descendants and are recorded in trial metadata. Tasks requiring
these privileges during agent execution therefore run under a changed capability protocol.

The static capability launcher is compiled from the frozen `capdrop.c` on the CPU worker, so a host
C compiler/static libc is required. Missing namespace support or launcher/relay setup fails closed.
Controlled CPU canaries cover direct/proxy denial, model streaming, local services, separate grading
networking, background processes, single/multi-UID and non-root agent users. They are bounded
validation of this protocol, not a claim that every task dependency is available offline.

Historical configs lacking `agent_internet` mean the original online protocol. Resuming them with
the new default is rejected; use a new run ID for offline generation. Explicit `--agent-internet`
is retained for reproducing historical runs. Status/scoring do not change old generation policy.
Oracle reference execution remains trusted evaluator work and does not inherit agent-only limits.

## Model input modalities

New Pi runs freeze `model_input` in config and protocol from the resolved registry's
actual `launch.json`, with its exact bytes (base64), path, SHA-256, launch timestamp and
every replica's capability. Frozen snapshot hashes are checked before use. The job ID
is explicitly labeled as registry evidence, rather than asserted launch identity.
Each known launch must name the resolved endpoint in its explicit `--served-model-name`
aliases; a mismatch is rejected and missing identity leaves capability unknown. An explicit
positive `--limit-mm-per-prompt` image allowance enables `['text', 'image']`;
`--language-model-only` or an explicit zero allowance selects `['text']`.
Conflicting replicas are rejected, including known conflicts alongside malformed peers.
Partial unknown evidence retains every known replica's flags and cannot override a
contradictory declaration; AUTO remains unknown/text. Model names never infer vision.

`--model-images auto` is the default. Missing/unparsed launch evidence and URL endpoints
retain text input with recorded unknown provenance. For unknown endpoints only,
`--model-images on` explicitly declares image support; declarations contradicting a
known launch are rejected. Workers pause when known launch evidence disappears or
current capabilities differ. A resumed run keeps its frozen metadata; historical
runs without it retain Pi's original text-only default.

The frozen policy owns reserved `PI_MODEL_INPUT_JSON`; conflicting request or worker
environment overrides fail. Pi's custom model receives its explicit `input` list, and
each attempt's protocol records that list. A CPU test runs pinned Pi 1.2.0 against a
deterministic fake endpoint: a native image `read` produces an embedded image in the
next HTTP request for image input, and none for explicit text or legacy controls.

Requests containing images or other non-text content always use the full backend
`/tokenize` path before generation budgeting. Compressed image bytes do not bound
the processor-expanded token count, so the append-only byte shortcut is restricted
to text content in both the submit guard and the compact-policy proxy. The backend's
expanded count determines remaining room within the configured context window;
existing context-limit termination and backend-error handling remain in force.

A vision-model canary must freeze image capability true and actually send an image
after a native tool read before release. Selecting an image task or falling back to
unknown/text alone does not establish that path. This is a new generation protocol;
existing stored outcomes and historical text-only comparisons are not upgraded.

## Task-specific runtime compatibility

New TB-Science-lite Pi configs explicitly freeze `task_agent_classes` in config and
protocol. Only `unknown-payload-balancing` selects `PiCont16kEnvClockMuslCompat`;
the other selected tasks use standard `PiCont16kEnvClock`. The worker validates the
recorded map and rejects global class overrides that would spread the adapter.
Each adapted attempt records the selected task/class and a runtime receipt.
Historical configs without this map retain their existing class on resume.

The adapter requires that exact task and `x86_64-musl`. Its original Alpine image
lacks the glibc interpreter required by the pinned Pi Node binary. The adapter
installs the unchanged verified Pi bundle with the existing SHA-pinned private
glibc loader under `/opt/harbor-pi`, and launches only Pi through that loader.
It does not replace task libc/Node or set a global library environment. Missing or
mismatched runtime assets fail before agent execution. CPU proof on the exact
cached task image verifies non-root Pi 1.2.0, unchanged task Node 22.23.4/musl, and
one native bash tool round trip through a deterministic fake model endpoint.
That synthetic canary does not run or validate the benchmark's scientific verifier.

## What is eval_q35 and what is added

| Piece | Source |
|---|---|
| Agent, proxy, sandbox | eval_q35 `Worker` → `gen/run_harbor.sh` → `gen/pi_cont16k.py` (Pi 1.2.0 + `inference/sticky_proxy.py`) in harbor_ns. Imported from the run's frozen code, not modified. |
| Protocol | `eval_q35/common.py` `PROTOCOL_PI180`, copied into config.json: 132,000-token context, submit at the limit (no compaction), 16,384 output tokens per request (0 resamples), T 1.0 / top_p 0.95 (set by the proxy), bash tool default timeout 180 s, task limit 21,600 s of **environment time** (model time excluded) with a 2× wall-clock backstop |
| Seeds | `sha256("<namespace>/<task>/<attempt>")[:4]`, +1 per request; the namespace is per benchmark and fixed across models (TB 2.1: `q35-tb21-v1`, the seeds of every past TB 2.1 eval) |
| Queue, retries, classification | eval_q35 `queue_state.State` (Lustre mkdir claims) and `worker.classify`: up to 6 same-seed retries of infrastructure failures (environment start/build, Pi before its first request, backend unreachable or timing out, proxy outage, verifier bootstrap stall, worker stopped); everything else is the model's outcome; a missing verifier reward scores 0 only after a **verifier timeout** (AA's rule); a verifier that crashed or never ran is an infra retry (`verifier_no_reward`, pi_eval) |
| Verifier | the task's tests; eval_q35's offline verifier toolchain where it has the task (TB 2.1, TB-Lite), the task's own online bootstrap otherwise. eval_q35's verifier retry wrapper is **not** installed for separate-verifier tasks (`environment_mode = "separate"` or a tests/Dockerfile: their image copies test.sh by name; bootstrap baked in, no network) |
| **Endpoint** (pi_eval) | `worker.py` resolves the registry every 30 s; each attempt is pinned to one replica by hash of (task, attempt); claiming pauses while the registry is stale or a replica does not answer |
| **Admission** (pi_eval) | `queue/admission.json` from the run's `max_in_flight` (claims − done over all workers) and `burst` starts per minute; paused while a serving admission gate (registry `gate: true`) holds new conversations, so gate queueing never meets Pi's 30-min request timeout (per replica, #378: paused only while every replica's gate holds; a new attempt whose hashed replica holds is pinned to a clear one, `queue/claims/<key>/replica.json`); claims whose heartbeat is > 20 min old (lost or held workers) are requeued as infra retries (`stale_claim`, eval_q35 admission.py's reaper); **live override**: `RUN/control.json` `{"max_in_flight": N, "burst": N}` (more than workers × slots needs `--resubmit N`) |
| **Reasoning effort** (pi_eval) | `--reasoning-effort` → the proxy's per-request `reasoning_effort` (its default, `max`, is outside Qwen3.8's template). Checked on aa-qwen-any: prompt tokens of a one-word chat are xhigh 53, medium 11, low 41, the same with the field in `chat_template_kwargs`, so vLLM passes the top-level field to the template |
| **Reasoning history** (serving) | The Qwen3.8 27B servers (aa-serving `synth/teachers/qwen38-27b-eval-fast.toml`, 569db98) set `--default-chat-template-kwargs {"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "xhigh"}` (line 38; line 61 sets the same for synth clients). The proxy sends only `reasoning_effort` and `chat_template_kwargs: {"clear_thinking": false}`, which vLLM merges over the default, so **every earlier turn's reasoning stays in the prompt**; Pi's bundle keeps reasoning-only messages and resends them (eval_q35 README, issue #1). The prompt therefore grows by each turn's full output, not just its visible text, and this largely decides when the window runs out: on TB-Science and lite the median episode generates 92–95k output tokens at a 132k window (median peak context 128k) and 172–190k at 262k (median peak 258k). The request's `reasoning_effort` does override the template default: the first request of the same task has 2,037 prompt tokens at xhigh and 1,999 at medium. Any comparison with AA (whose harness and endpoint handling of history are not stated) or with another serving spec must hold this setting fixed |
| **Context window** (pi_eval) | `--context-window N` sets the protocol's `context_window` (default 132,000); eval_q35 derives Pi's window, the proxy's and the context guard's from it, so the episode still ends at the window. Per attempt the stats record the largest request context (`max_context_tokens`, from the proxy's wire responses) |
| **Time cap** (pi_eval) | `gen/pi_cont16k.py` runs only six-hour task TOMLs. `--tasks-tar` rewrites `[agent] timeout_sec` to 21600 and nothing else (`bundle.py`, verified by re-parsing; originals in the manifest); `--agent-cap-sec C` scales the environment clock (`PI_ENVCLOCK_MULT = C / 21600`) |
| **Sandbox options** (pi_eval) | `--multiuid` → `HARBOR_NS_MULTIUID=1` (harbor_ns multi-uid, PR #47; default on); `--env-class` → `--env` after run_harbor.sh's own (e.g. `aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment` for sidecar artifacts and compose env); `--harbor-env KEY=VAL` (repeatable) → extra environment for each attempt's `run_harbor.sh`, e.g. harbor_ns switches such as `HARBOR_NS_MEMCAP` (workers run with `getenv = false`) |
| **Disk cap** (pi_eval, #405) | `--diskcap auto\|FACTOR[:MIN_MB]\|off` (default `auto`, as eval_q35's submit.py: a bench whose defaults name a `diskcap_family` gets eval_q35's measured per-task caps, `DISKCAP_FAMILIES`; TB 2.1: rstan-to-pystan 90 GiB, run one attempt at a time per worker, sam-cell-seg 40 GiB, others 32 GiB; every other bench `3:30720`, as eval since #372): each sandbox is capped at max(storage_mb × factor, min_mb), e.g. max(storage_mb × 3, 30 GiB): no file grows past it (RLIMIT_FSIZE: SIGXFSZ / EFBIG), and a box whose overlay upper dir and pod volumes are above it loses its top writer (at 1.5× everything but init), so one runaway `dd` cannot fill the worker's shared scratch (#371). Config `"diskcap"` (null = off; eval_q35's format, so per-task `"tasks"` caps and `"one_per_worker"` work too; a config without the key = off: AA runs before #405 ran uncapped, `PiEvalWorker.DISKCAP_WITHOUT_KEY`); eval_q35's worker env sets it. A task whose cap is above a quarter of the worker disk (`--worker-disk-gb`, default 300) runs one attempt at a time per worker (eval_q35's rule; with the default caps 4 TB-Science and 7 TB-Science-lite tasks, storage_mb ≥ 25 GiB, and TB 2.1's rstan-to-pystan); each attempt records `attempt.json` `"diskcap"`, `attempts/<key>/diskcap.json` (limit, peak, largest file, kills) and outcome `diskcap_limit_mb`, `sandbox_disk_peak_mb`, `sandbox_largest_file_mb`, `diskcap_kills`. Multi-uid boxes included (harbor_ns/diskcap.py, `aa_evals/common/diskcap.py`) |
| **Turn gate** (pi_eval, opt-in) | eval_q35's context-aware turn gate (`eval_q35/turn_gate.py` v11 via `gate_server.py`, the default admission of eval_q35's Qwen3.5-4B evals, serve_q35/README.md) plus the proxies' KV brake (0.80/0.72). Off by default; on when the serving registry's `request.ctx_gate` is true (spec `qwen35-4b-eval` on aa-serving) or with `--ctx-gate`; `--ctx-fill` / `request.ctx_fill` sets its start fill (0.7). eval_q35 runs the gate in its serving job; here one elected worker hosts it (`turn_gate_host.py`: mkdir lock `RUN/serve/turn_gate.lock`, taken over after 180 s of silence), one gate per replica with that engine's cache size (`/metrics` cache_config_info, else the registry's vllm.log), synced every 15 s with the run's live claims; each attempt's proxy reads its replica's `RUN/serve/kv-<i>.json` (KV reading + `gate_url`) and asks before every turn (`PI_GATE_KEY`). Use it only on a server whose spec has no aa_evals admission gate in front |
| **Oracle** (pi_eval) | `--agent oracle`: the reference solutions through the same worker, sandbox and verifier path (Harbor's oracle agent) |
| **Storage** (pi_eval) | trials run in node-local scratch; each attempt becomes `attempts/<key>/trial.tar.gz` (+ `outcome.json`, `attempt.json`, `harbor.log`) with its stats computed before packing. A Pi attempt is ~30 MB in ~150 files, mostly the proxy's streamed wire responses |
| **Scoring** (pi_eval) | `score.py`: `items.jsonl` + `summary.json` (aa_evals contract). pass@1 over all attempts (reward ≥ 1) with deadline cutoffs counted 0 (`n_cutoff`; resolved-only score secondary), task bootstrap ci95 (seed 731), infra counts and validity (infra ≤ 2%), end states, context-limit and output-cap counts, mean model requests, per category |

Runs, worker condor logs (`RUN/jobs/`) and code snapshots (`$AA_RUNS_ROOT/code/`) live under
`$AA_RUNS_ROOT/<bench>/<run_id>/`, default `/data/aai/aa_evals/runs_fallback`. They stay on /data even when /data
warns that it is over quota: /home is not for runs, logs or data (see aa_evals/README.md "Cost and storage
rules"). The old `/home/.../aa_runs_fallback` path is a symlink to this tree.

**Worker jobs (#378).** Default slots, CPUs and RAM per benchmark are in `aa_evals/pipeline.json` (`worker_shapes`, with
each one's CPU model and basis), from c30 and the #378 measurement run
(`/data/aai/aa_evals/audits/c30_efficiency_20261113/memrun/report.md`). tbshared fills the RAM at submission:

| bench | slots | CPUs | RAM asked | RAM submitted (filled, where free) | units per slot |
|---|---:|---:|---:|---:|---:|
| tb21 | 32 | 10 | 128 GB | 160 GB | 0.31 |
| tbsci_lite | 4 | 4 | 64 GB | 64 GB | 1.0 |
| tbsci_lite_synth (`--shape-key`, m5/tbsci_lite synth traces) | 16 | 8 | 96 GB | 128 GB | 0.5 |
| critpt_pi | 24 | 4 | 32 GB | 64 GB | 0.17 |
| gdpval_pi | 16 | 2 | 24 GB | 32 GB | 0.125 |
| automationbench_pi | 32 | 4 | 32 GB | 64 GB | 0.125 |

- **CPUs** = (base + per slot x slots) / 0.8: the measured mean with all slots busy, at ~80% (`cpu_model`). The CPU request is a hard cpuset (affinity mask): the slots of a worker share exactly its requested cores, never idle cores of the node. Sizing at the full-slot mean / 0.8 rather than the p95 is a deliberate trade-off: the slots mostly wait on vLLM, and some burst delay is accepted for higher average utilization. At the current caps, in minutes with at least 75% of slots busy, about 3.7-9% of CPU work is deferred (GDPval 3.7%, CritPt 4-9%, TB 2.1 6%), under 1% of attempt time (coordinator, #378).
- **RAM asked** = slots x the shape's GB per slot: what the workers need, ~2x the measured process memory at full slots. Trial archives use O_DIRECT, so page cache needs no headroom.
- **RAM submitted.** RAM up to 16 GB per CPU costs no units, so `condor_cost.fill_plan` (tbshared 1.5) raises it to 16 x CPUs where that is free. It plans the batch filled and unfilled; if filling would start any worker on a higher rung, or leave one fitting on no node, RAM is capped at the bottom rung's free GB per free CPU, or left as asked. The decision is in `RUN/jobs/fill.jsonl`, and `config.json` keeps the shape asked for.
- **Memory holds.** A memory hold restarts the worker with 25% more RAM, at most 5 times (`periodic_release`). Above 16 GB per CPU that costs units, which is intended: a worker that needs more pays for it.
- **Derived names (#451):** a campaign's bench name takes its base suite's shape: prefixes `m5_`, `m5l_`, `m5l2_`, ..., suffixes `_t2`, `_finish`, `_t2fin`, `_resample`, `_pi`, `_m5`, then `<suite>` or `<suite>_pi` (`SHAPE_ALIASES`: `tb` -> `tb21`). A workload that differs from its bench's (synthesized TB-Sci-lite traces) names its entry with `--shape-key` (`tbsci_lite_synth`).
- **Other benchmarks:** 24 slots, 0.5 CPU per slot, 16 GB per CPU, with a launch warning that no measured shape exists. `--slots` / `--worker-cpus` / `--worker-mem-gb` override the asked shape; launchers should not pass CPUs or RAM: an explicit value more than 1.5x off the measured shape at the same slots is warned about and recorded in `config.json` `worker.shape_check`.
- **Bid:** the CPU bid ladder 26 -> 36 -> 56 (`aa_evals/common/condor_cost.py`) unless `--cpu-bid` is given.
- **Logs:** each worker logs its cgroup memory, CPU and run-queue wait every minute to `RUN/jobs/res.<cluster>.<proc>.jsonl`. Size RAM from `mem_peak` there, never from condor's RSS-only MemoryUsage.

## Watching a run

- `--status`: physical attempts done, active claims, infra attempts, passes, admission state, live workers.
- Worker logs: `RUN/jobs/worker.<cluster>.<proc>.out` (`claimed`, `done <key>: score=… infra=… exc=…`, endpoint up/down).
- One attempt: `tar xzf RUN/attempts/<key>/trial.tar.gz`; the Pi session is in `<key>/<trial>/agent/`, the proxy's
  per-request logs in `<key>/pi-cont16k/`.

## M3 protocol (M3.2, #175)

- **Context:** `--context-window` defaults to 131,072 (M3.0's 128k cap); pass 262144 only for the Qwen3.8
  calibration runs. Run creation refuses a window above the server's `max_model_len`. `config.json`
  `protocol_overrides` records the change from eval_q35's 132,000.
- **Repeats:** `--repeats` (alias `--attempts`), default 5.
- **Request settings, per model** (`run.py resolve_request`): the serving registry's `[request]` block
  (`endpoints.json` → `request`), or `--request-preset TEACHER` (aa-serving `synth/teachers/TEACHER.toml`, for
  `url:` specs), plus `--request-overrides JSON`. `--reasoning-effort` replaces the effort at the top level and in
  `chat_template_kwargs` when the template reads it there; without it the registry's effort stands (DeepSeek: max)
  and models without one get none. A model with no settings is an error: there are no built-in sampling defaults.
  The worker passes them as `PI_REQUEST_OVERRIDES` (merged by the Pi proxy into every sampled request),
  `PI_INLINE_REASONING_HISTORY` (adapter `inline_reasoning_history`), `PI_THINKING` (`pi_thinking`),
  `PI_COMPAT_JSON` and the registry `env` (DeepSeek: `PI_CONTEXT_RENDERER` / `PI_CONTEXT_TOKENIZER`, read by the
  context guard). `test_request.py` covers it.
- **Model input:** historical runs use Pi's text-only custom-endpoint default. New runs
  freeze explicit modalities from actual launch evidence as described above.
- **Finish notice (#375, default for new Pi runs since #382):** without compaction, long reasoners used to fill the
  window and end with no answer. The context guard (`tb_agents/pi_context_limit.py`, policy
  `submit-at-context-limit-v4-sticky-notice` for new runs; runs created with `...-v3-finish-notice` keep v3)
  appends a short harness notice with the tokens left to each
  forwarded request once at most 40,000 remain, and keeps the last 16,384 for a final turn ("this is your final
  turn"). Nothing in the conversation is edited or dropped (earlier reasoning stays); the notice is joined to the
  trailing user or tool message, never added as a new user turn. Qwen3.8 Xhigh A/B (#375): CritPt agreement
  13.6 -> 20.0, answered 36% -> 74%; GDPval Elo +98. `--finish-notice WARN_AT:RESERVE` changes it, `--finish-notice
  off` disables it; the value is frozen in `protocol.finish_notice` and resumed runs keep their stored policy (an
  explicit flag on resume must match the stored WARN_AT:RESERVE; the policy version stays the stored one).
  Notices are logged to the attempt's `pi-cont16k/finish-notices.jsonl`.
  **v4 is the default for every new Pi run, evals included** (coordinator decision on the #375 sftnotice review,
  following the user's no-train/test-gap rule: evals and SFT trace generation use the same notice policy). New eval
  runs are therefore NOT protocol-identical with the c30n/c30fn results, which are v3: in v4 every earlier notice
  stays in the context (p50 5, p90 12, max 135 notices per trial on the c30fn/c30n trials), so the model also sees its
  earlier warnings and "final turn" notices. Compare v4 runs with v4 runs.
  **Sticky notices (v4, #375 sftnotice) and SFT traces.** The notice reaches the model but not Pi's session (nor the
  sticky proxy's `wire-requests.jsonl`, which sit upstream of the guard). In v4 a notice, once joined to the message
  at index i, is joined again with the same text to that message in every later request that still has it (same
  index, role and normalized text), before `/tokenize`, so the count includes it; the trailing message's notice is
  decided afresh each request (a retried request gets it again, replacing the entry). Entries are keyed by index, role
  and text, so another conversation through the same guard (an agent querying the endpoint from its sandbox, seen in
  c30n DeepSeek trials) neither gets nor clears them. The final request of a trial is
  then the one linear trace the model saw. Each new notice line records the map entry: `index` (in the forwarded
  request, system prompt included), `role`, `base_sha256` (sha256 of the message's normalized text before the
  notice: a str as is, a list's text parts joined with a blank line; `notice_base_text`), `text` (joined as
  `join_notice` joins it: str + blank line + text, or one more text part), `appended` (only if the guard had to add
  a user message; never seen from Pi). SFT trace generation runs with the default notice: the M5 exporters
  (`m5/{gdpval,automationbench,critpt,tbsci_lite}`) ship each trial's parsed lines as `finish_notices` next to
  `sessions` in the archive line (`finish_map.py`) and write `"finish_notice_map": 1` in `manifest.json`. A v4 guard
  creates `finish-notices.jsonl` (empty) at startup, so an attempt whose archive lacks it has lost its map and is
  unusable (`finish_notices_missing`); so is one with an empty map although its largest served prompt
  (`stats.max_prompt_tokens`) was inside the threshold (`finish_notices_empty_near_window`), or a malformed map. v3
  runs are refused, before anything is written. Runs without a notice export as before, except that gdpval/tbsci_lite
  records now carry `finish_notice_problem: null` (training input unchanged). `sft_qwen35/prepare.py` (branch `sft`) joins each notice to the training message at `index` after checking
  role and `base_sha256`, and drops the attempt on any mismatch (`drop:finish_notice_mismatch`): its message list
  has the forwarded request's indices (system prompt from the first wire request at 0, then the session's
  messages). Checked on the 1,323 c30fn/c30n trials with notices (maps rebuilt from their v3 logs): 1,201 of 1,222
  records spliced, 21 dropped: 19 image tool results (Pi's wire text adds `(tool image omitted: model does not
  support images)`, which the session and so the training text lack: a gap of its own) and 2 sessions misaligned with
  the wire.
  `--finish-notice off` still gives plain v2 traces.
- **summary.json** adds `n_infra`, `n_flagged` (`aa_evals.tracejudge.fold_summary` over `RUN/flags.jsonl`, M3.4; `null` before it ran),
  `context_cap`, `judge: null`, `request_overrides`, `request_source`, `request_adapters`.
- **Adaptive limiter** for long runs on one engine: `python -m aa_evals.common.pi_eval.adaptive --metrics
  http://<engine host:port> RUN [RUN ...] --start 24 --ceil 64` writes each run's `control.json` from the engine's
  KV usage, waiting requests and preemptions (runs in priority order).
# Task-scoped evaluator compatibility

New offline TB2.1 Pi configurations record a schema1
`task_environment_policy` and canonical SHA256 in both config and protocol.
The only permitted override is `pypi-server` →
`HARBOR_NS_SKIP_PIP_PROXY_SEED=1`: the task explicitly installs from a localhost
index, whereas an evaluator-added pip global proxy overrides localhost bypass.
The provider opt-out leaves every preexisting image pip configuration unchanged.
Other tasks retain global proxy seeding for evaluator pip under `env -i`.

Workers validate the exact allowlisted policy and protocol digest, dispatch a
fresh task-specific environment, and record effective overrides/digest in each
attempt's immutable metadata. Stored configurations without the field dispatch
no override and are never upgraded. This repair changes the generation
environment contract; it is not a scoring/classification-only normalization or
permission to carry old affected task generations into a repaired environment.

# Benchmarks ported to this harness (#348)

GDPval, CritPt and AutomationBench run their agent here and keep their grading (`aa_evals/<bench>/pi/`). Shared
pieces, all opt-in (no existing run changes):
- `sif_base.py`: an Apptainer SIF as a harbor_ns base image (shared base cache), so task.toml `docker_image` can
  name it: one image for every task of a benchmark, no per-task build.
- `task_inputs.py` / `task_inputs_env.py:NsTaskInputsEnvironment` (`--env-class`): `NsSidecarEnvironment` plus
  per-task input files (`<task>/inputs/<abs path>`, inventoried with SHA-256 in `inputs.json`) copied into the
  agent's sandbox after it starts; a receipt (`task_inputs.json`) goes into the trial directory. Tasks without
  `inputs.json` are unaffected.
- `port.py`: the pi_eval arguments the ports share (131,072 window, 2 h environment clock, single-uid, the inputs
  environment, `HARBOR_NS_MEMCAP=1`, `--model-images off`: text-only serving enforced; an empty selection refused), launching pi_eval in the harbor venv, and reading resolved attempts back
  (Harbor artifacts = what the episode left, with the artifact manifest's collection status, the Pi session, the
  end: `finish` / `context` / `time_limit`). `extract` reports `found: False` for a missing archive, trial or
  artifact manifest, which the ports never read as "the agent left nothing".
The bundles' `tests/test.sh` only record that the submission exists (and write a CTRF report so eval_q35's verifier
wrapper does not retry); the benchmark score is computed by the benchmark's own grader after import.

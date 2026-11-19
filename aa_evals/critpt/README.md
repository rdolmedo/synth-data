# CritPt v2 (AA Intelligence Index v5.0, 10%)

Research-level physics. There are 70 public test challenges; their answers are hidden, and only the CritPt authors' server grades them. Issue #32 has the plan and decisions.

## M3 protocol (M3.2 #175, proxy v2 #185)

```bash
PY=/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python
export CRITPT_RUNS=/data/aai/aa_evals/runs_fallback/critpt   # the 11-02..11-13 runs; new runs may use the default
# any served model, 70 problems x 5 repeats, 128k context cap; settings from the serving registry's request profile
$PY -m aa_evals.critpt.run --model serving:<name> --run-id <id> --condor
# Qwen3.8-27B: its profile pins xhigh, so name the effort (both top level and in chat_template_kwargs)
$PY -m aa_evals.critpt.run --model serving:<qwen> --reasoning-effort medium --run-id <id> --condor
# re-score EVERY stored run against a reference (re-probes stale answers, rebuilds a run-based reference whose
# fingerprints are stale, re-grades all runs): one CPU job, no model calls
$PY -m aa_evals.critpt.rescore --reference ds41-max --condor
# new reference answers from elsewhere (e.g. a stronger API model): import them, then the same re-score
$PY -m aa_evals.critpt.rescore --import-answers answers.json --reference <name> --condor
```

- **Headline (user decision, #185): all 70 problems** (`critpt_v2_all70_agreement@<reference>`, the former
  `proxy.all_problems`): per problem, the share of the run's repeats whose answer matches the reference's most common
  answer; a problem without a comparable reference answer counts 0; the CI (`ci95`, a half-width; `ci95_interval`)
  is over problems. Agreement on the reference's kept problems stays in `proxy.agreement`. Per problem and repeat:
  `proxy_<reference>.json` → `per_problem_all[pid].by_repeat` (the item × model matrix for M3.8). Runs that belong
  to the reference are scored leave-one-out on both views.
- **Request settings:** the endpoint's `request` block (`request.overrides` and `adapters` from the teacher spec);
  `--reasoning-effort`, `--temperature`, `--top-p`, `--chat-template-kwargs` override it, `--no-profile` drops it.
  Keys the OpenAI client does not take (top_k, min_p, repetition_penalty, chat_template_kwargs) go in `extra_body`.
  DeepSeek V4.1 Flash: Max from its profile. Qwen3.8: Medium or Xhigh by flag. Every other M3.1 model: reasoning on
  via its profile. `AA_ROWS` in run.py only maps a served model (and effort) to its AA row.
- **Reasoning history:** Stirrup sends earlier turns' thinking as `reasoning_content`, which vLLM ≥ 0.31 drops. The
  client now applies the profile's history adapter: `mirror_reasoning_content` (also sent as `reasoning`) by
  default, `inline_reasoning_history` (as `<think>…</think>` in the content) for Nemotron. **The #66 runs
  (`ref-ds41-max`, `qwen-full-*`) predate this: the model saw its earlier turns' thinking as empty.**
- **Context cap:** 131,072 tokens (`--context-window`, M3.0 decision on #173). Every request's
  `max_completion_tokens` is the room left under the cap (the previous response's exact prompt + completion, plus ~1
  token per 2.5 characters added since: an upper estimate), and Stirrup summarizes at 70% of the cap. With no room
  left Stirrup compacts; a single response that fills the room is `output_limit` (scored 0, as at AA).
  `--context-window 0` = the served `max_model_len`, only for the explicit Qwen3.8 full-context calibration runs.
  `summary.json` → `n_cap_overshoot` counts responses whose actual total passed the cap (estimate too low).
- **Repeats:** 5 (AA: 5).
- **Text only:** all 70 problems and the example are text (LaTeX and code templates, no figures), so text-only
  models (DeepSeek `--language-model-only`) need no separate path.
- **`summary.json`** adds `model`, `served_name`, `reasoning_effort`, `context_window`, `context_cap`, `judge` (none),
  `code_sha`, `n_infra`, `n_cutoff`, and, through `aa_evals.tracejudge.fold_summary`, `n_flagged`,
  `n_flagged_by_category` and `tracejudge` from the M3.4 trace judge's `flags.jsonl` (null until it has run; run
  `python -m aa_evals.tracejudge watch --run-dir <run> --judge serving:aa-judge-gemma4-31b` next to the run;
  `run.py --grade` refreshes them).
- **Probe v3 (#185):** answers built from `sympy.physics.quantum` inner products are evaluated numerically: each
  `InnerProduct(Bra(a), Ket(b))` becomes `v_a^† v_b` on fixed complex 3-vectors per state label, with `|x*⟩` the
  complex conjugate of `|x⟩` (the problems' own notation). Algebraically equal bra-ket answers now match (Challenge_30:
  a factor distributed over a sum, `|⟨a|b⟩|²` vs `⟨a|b⟩⟨b|a⟩`, `⟨φ*|ψ⟩⟨ψ|φ*⟩` vs `⟨φ|ψ*⟩⟨φ*|ψ⟩`). Only labels are
  used, so `Bra(Ket(φ))` (Bra of a Ket object) stays a different state, as a strict SymPy grader would treat it, and
  states are not assumed normalized (the problem does not say so). Anything still holding a state or operator falls
  back to the expression tree. Tests: `test_consensus.BraKet`.

## Pi harness (#348, c30)

User decision 2026-11-11: CritPt runs under the TB 2.1 Pi harness. The Stirrup path stays for M3.5/M3.6.

```bash
python -m aa_evals.critpt.pi.tasks build                     # once: 70 tasks, /data/aai/aa_evals/data/critpt/pi_bundle
bash aa_evals/critpt/pi/build_ns_base.sh                     # once (CPU job): the SIF as harbor_ns base image
python -m aa_evals.critpt.pi.run init --run-id ID --model serving:NAME [--reasoning-effort E] [--repeats 2] [--problems a,b]
python -m aa_evals.critpt.pi.run collect --run-id ID --follow
python -m aa_evals.critpt.run --run-id ID --reprobe --condor && python -m aa_evals.critpt.run --run-id ID --grade
```

- Harness: Pi 1.2.0 `pi-cont16k`, 131,072-token window **without compaction**, 16,384 output tokens per request, no
  turn cap (AA: 50 turns), 2 h of environment time; the CritPt SIF's root filesystem as the sandbox (`/work`, HOME
  `/work`, 4 BLAS threads, 16 GB memory cap), offline.
- Prompt: the authors' one-step system prompt, then `prompts.user_prompt` with the finish tool and the turn budget
  replaced by the answer file: the filled-in template in `/work/answer.py` when the episode ends is the submission.
- Import (`collect`): fences stripped as before; an empty, missing or unparsable file is `model_error`
  (`no_finish: ...`, scored 0; Stirrup's finish tool never accepted those). Grading (probe + consensus vs ds41-max)
  is unchanged. Seeds: namespace `critpt-pi-v1`.
- **Collection integrity:** an attempt is imported only when its trial archive and Harbor's artifact manifest are
  readable. An archive not visible yet (record < 1 h old) is retried on the next `collect` pass; a pack error, a
  lost archive or a missing trial/manifest is `infra_error` (never `no_finish`), so `collect --follow` ends. A missing answer file is the model's `no_finish` only when the
  placeholder verifier (in the sandbox, after collection) also found no parsable `/work/answer.py`; if the verifier
  saw one that the collection lacks, the attempt is `infra_error`. Text only: the port passes `--model-images off`.
- **Deviation, where the answer comes from (coordinator decision 2026-11-11, #348; grading only):** under Stirrup the
  `finish` tool captured the answer; under Pi it is `/work/answer.py`, with one fallback: when the agent ended the
  episode itself (`end=finish`) and wrote no file, the last ```python block of its final message is graded if it
  passes the same checks (fences stripped, valid Python). Never for context or cap endings (they stop mid-thought).
  Each row records `answer_source` (`file` | `final_message`); `summary.json` counts them (`answer_source`; Stirrup
  runs: `finish_tool`). c30 canary: Lightning ended 2 of 3 attempts with the filled template in its last message
  and no file. Re-grade stored runs with `pi.run collect --reimport`, then `run.py --reprobe` and `--grade`.
- **Deviation, end disclosure (accepted 2026-11-11, #348):** one added line says the session can end without warning
  (e.g. the window is full) and `/work/answer.py` is graded as it is then, so write it early and update it
  (`pi/tasks.py END_DISCLOSURE`; `config.json` `harness.prompt_disclosures`). Without it, Qwen3.8 Medium's reasoning
  filled the 131k window before any answer file in 2 of 3 smoke problems; with it, 3 of 6 still did.

## Usage (before M3)

There are two grading paths:
- **Consensus proxy (default).** No public answer key exists, so the default score is agreement with a consensus reference built from DeepSeek V4.1 Flash 1017 (Max), 5 samples per problem. It is a ranking signal on the problems that model solves consistently, not an AA-comparable score. See "Consensus proxy" below.
- **AA's server (optional, exact).** Use it only if the user decides to go through AA's platform: it needs an API key plus CritPt approval. It gives the exact full-70 pass@1, comparable with AA's published full-70.

```bash
PY=/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python   # Python 3.13 + stirrup==0.2.1
# 1. Run a model: 70 problems x 5 repeats (+ the example items, graded locally), as a CPU condor job.
#    The sandbox needs a compute node; the tmux host cannot start contained Apptainer. Keep --concurrency within
#    what the shared server allows (aa-super: <= 16).
$PY -m aa_evals.critpt.run --model serving:<name> --run-id <id> [--limit N | --problems a,b] [--repeats 5] [--example] --condor

# 2. The reference (once; the "stronger-model" stage): DeepSeek V4.1 Flash 1017 (Max) itself, then the consensus.
$PY -m aa_evals.critpt.run --model serving:<deepseek> --run-id ref-ds41-max --repeats 5 --example --condor
$PY -m aa_evals.critpt.consensus reference --runs ref-ds41-max --name ds41-max     # keeps problems with >= 60% agreement

# 3. Score: refreshes summary.json (headline = agreement with ds41-max; the DeepSeek run itself is scored leave-one-out).
$PY -m aa_evals.critpt.run --run-id <id> --grade
#    Exact path, optional: also send each repeat's 70-problem batch to AA (key + CritPt approval needed).
$PY -m aa_evals.critpt.run --run-id <id> --grade --submit
```

**A CritPt score needs one of two things**, for Qwen3.8 27B as for any model: an AA API key with CritPt approval (exact path), or a reference model's run to build the consensus from (proxy path). Without either, a run validates everything except correctness: `finish` use, answer extraction, the format/exec probe, the caps, and the example items graded locally. That is what the 2026-11-02 overnight round 1 did, since the DeepSeek reference was descoped. AA lists one row per reasoning effort, and `summary.json` compares a run with the row for the effort it used (`AA_ROWS[...]['aa_slug_by_effort']`): Qwen3.8 27B (Xhigh) is 10.6 full-70 and 4.0 hard-40; Qwen3.8 27B (Medium) is 16.5 hard-40, with no full-70 published. Our headline is full-70 (or, on the proxy path, agreement), never hard-40.

- **Example items:** `--example-repeats N` runs them at their own repeat count, for example `--repeats 1 --example --example-repeats 5`.
- **Jobs:** `--condor` exports the committed `aa_evals/` to `runs/critpt/_code/<sha>`, so commit first. It submits at bid 61 with `+BypassLXCfs` and `request_disk`.
- **Resuming:** rerunning the same command resumes the run. Finished item × repeats are kept. Infra errors are rerun from scratch, up to `--max-attempts` (3) in total.
- **Generation settings:** the endpoint's request profile (since M3.2; a per-model `PRESETS` table before). Flags override it: `--temperature`, `--top-p`, `--reasoning-effort`, `--chat-template-kwargs`, `--max-output-tokens`.
- **API key:** `AA_API_KEY`, or the file `/data/aai/aa_evals/secrets/aa_api_key` (chmod 600). The account also needs CritPt approval from AA (critpt@artificialanalysis.ai); without it the API returns 403.

**Outputs** in `/data/aai/aa_evals/runs/critpt/<run_id>/`. Set `CRITPT_RUNS` to use another root; condor jobs inherit it. Runs made 11-02..11-13 are in `/data/aai/aa_evals/runs_fallback/critpt/` (moved off /home; runs and logs never go there):
- `config.json`
- `items.jsonl`: one row per attempt; the last row per item × repeat is final
- `transcripts.jsonl`: one packed line per attempt
- `batches/repeat_<k>.json`
- `grading.jsonl`: AA's responses (exact path only)
- `summary.json`

## AA's method vs ours

AA ([methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#critpt-v2)):
- Runs all 70 challenges in Stirrup, in an offline Linux sandbox with Python 3.13, numpy, scipy, sympy and matplotlib.
- System prompt: the authors' one-step prompt for models with code tools.
- At most 50 turns, one per response. The model submits with `finish(reason, derivation, answer)`, and no submission scores 0.
- 5 repeats, each graded as one batch by the authors' server.
- The index uses a **private 40-problem hard subset**: the 70 minus the 30 that seven frontier models solved 5/5.

| Item | AA | Ours | Why |
|---|---|---|---|
| Scored set | private hard 40 | **full 70** | The 40-problem list is private, and the API returns only one accuracy per 70-problem batch, with no per-problem results. AA publishes full-70 scores (field `critptAgentic` on the CritPt v2 page; snapshot `/data/aai/aa_evals/data/critpt/aa_reference_2026-11-02.json`). AA's hard-40 number is shown for context only. |
| Grader | the authors' server (via AA) | **default: agreement with a DeepSeek V4.1 Flash 1017 (Max) consensus** on the problems it answers consistently (`critpt_v2_agreement@ds41-max`). **Optional exact path:** the authors' server via AA (`critpt_v2_full70_pass@1`). | The user hasn't decided whether to use AA's platform (2026-11-02). The proxy is not on AA's scale: it favours easy problems and DeepSeek-like answers. |
| Harness | Stirrup (version unknown) | Stirrup 0.2.1 (PyPI; repo `c092cee`), stock agent loop, `EndpointClient` (below) | — |
| System prompt | authors' one-step, code-tools version | authors' `prompt_template_default_one_step.yaml` at `bc37a22` ("CritPt v2"), rendered | The code-tools variant is not public. Tool use is described in the user prompt instead. Stirrup prepends its own base prompt, as at AA. |
| User prompt | unpublished | ours (`prompts.user_prompt`): the problem, the code template, packages, limits (time per command, turns), submission rules | — |
| Sandbox | E2B (AA's primary provider) | Apptainer SIF `critpt_py313_20261102.sif` (python:3.13-slim + numpy 2.5.4, scipy 1.18.1, sympy 1.14.0, mpmath 1.3.0, matplotlib 3.11.2) with `--containall --net --network none`. Only the trial's work dir (as `/work`) and, since #32's M3.4 finding, a per-trial `/tmp` are bound in. Each command is its own `apptainer exec`; without the `/tmp` bind every command got a fresh private `/tmp`, unlike E2B, where `/tmp` lives for the whole session. In the runs before the fix (#66 `qwen-full-*`, M3.4 `m34-qwen-medium`) 3-4% of trajectories wrote files to `/tmp` and lost them at the next command; the model usually rebuilt them (the prompt says files persist in the working directory). `run.sandbox_selftest` checks that `/tmp` persists across two commands. Verified on a compute node: interfaces `lo` only; proxy and cluster unreachable; `/data` and home not visible. | — |
| Per-command limits | unpublished | 300 s (Stirrup default), `ulimit -v` 16 GB, 4 BLAS threads | — |
| Context management | Stirrup | Stirrup defaults: summarize at 70% of the context. A vLLM context-length 400 is mapped to Stirrup's `ContextOverflowError`, so the agent compacts. | — |
| Finish tool | reason, derivation, answer | same fields. The executor rejects an empty or syntactically invalid `answer` so the model can retry. | AA's validation is unknown. |
| What is submitted | unpublished | `generated_code` = the answer code (fences stripped) in one fenced ```python block. No `messages`. | The authors' pipeline submitted the whole markdown reply, so the server extracts code blocks; one block leaves it nothing to choose. |
| Context | the model's native context | **131,072-token cap** (M3.0, #173); the #66 runs used the native 262k | more `output_limit` / compaction for long reasoners (Qwen3.8 xhigh filled 262k in 13 of 70) |
| Reasoning history | provider-dependent | sent back as the template expects (history adapters, M3.2); the #66 runs sent it as `reasoning_content`, which vLLM dropped | the #66 runs saw empty earlier thinking |
| Headline | pass@1 from the authors' server, hard-40 in the index | **agreement with the reference on all 70** (#185) | a different scale |
| Sampling | T=0.6 for reasoning models unless the lab recommends otherwise; the maximum output the creator discloses | the lab's recommended sampling per model (the serving registry's request profile since M3.2; `PRESETS` before). Super VL: T=1.0, top_p=0.95, thinking on, template defaults, output capped at 128k per response. Qwen3.8 27B: T=1.0, top_p=0.95, top_k=20 (server default), `enable_thinking`, `preserve_thinking`, `reasoning_effort: xhigh`, no output cap (the card allows 262,144 reasoning tokens, the whole context). | — |
| API retries | up to 30 | up to 30 per request (backoff to 120 s, round-robin over DP ranks), then the trajectory is an infra error and is rerun | — |
| Repeats | 5 | 5 by default. The 2026-11-03 round 3 (`qwen-full-medium`, `qwen-full-xhigh`) runs **1 repeat** to fit the morning deadline, plus more if time allows. `summary.json` states the repeat count (`repeats`) and what the CI covers (`ci95_over`). | With 1 repeat on the proxy path, the CI is over about 40 kept problems with one sample each, roughly ±13-15 pp. |

## Infra vs model failures

- **`model_error`** (submitted blank, scores 0, as at AA):
  - no `finish` within 50 turns (`no_finish`);
  - the output filled the context (`output_limit`);
  - the context overflowed even after compaction (`context_overflow`).
  - Stirrup's context summary failed (`summary_failed`): Stirrup asked three times, the last time with tools withheld, and the model answered with reasoning only. Stock Stirrup aborts here.
    - `ref-ds41-max` ran on code from before this rule. Its summary failures were classified as infra errors and rerun at the end of the pass, which is harmless for a reference: a rerun adds one more sample of the model's answer.
  - a response outlived the 8 h per-request timeout while the server still answered a health check (`response_timeout`): a runaway generation, which is a model outcome.
- **`infra_error`** (never submitted as a model answer):
  - the endpoint failed after retries;
  - the sandbox failed to start (an Apptainer `FATAL:` exit 255 inside a tool call aborts the trajectory instead of being shown to the model);
  - a crash in our code.

  These are rerun. An item still failing after 3 attempts is blanked in the batch and excluded by rescaling: accuracy × 70 / n_scored. A run is `valid` only with infra errors ≤ 2%, nothing unfinished (unless `--cutoff` was applied, which scores unfinished items 0), and a computed headline.
- **Deadline cutoffs** (rules for every benchmark, 2026-11-03):
  - **Scoring:** items with no answer at the run deadline count as wrong (0). They are reported as `n_cutoff` with `--grade --cutoff` (exact path: submitted blank) and `consensus score --cutoff` (proxy: each missing repeat scores 0). Without `--cutoff` they are `n_pending` / `n_unfinished`, and the score covers finished items only.
  - **Timeouts:** our load gate waits before a request is sent, so its queue time never counts. A server-side admission queue does count against the client timeout, which is why the timeout is 8 h: `max_tokens` and the 50-turn cap bound a response, not the clock.
  - **Code:** final numbers come from merged code; `config.json` records `git_sha`.
- **Per-item scores are unknown.** The server returns one accuracy per batch, so `items.jsonl` has `score: null` for `ok` rows. The CI is over repeats.

## Validating without AA's server

**No answer key exists for the 70 challenges:**
- On HF, `answer_code` is the template, `answer_only_code` is empty and `testcases` is null.
- None of the 70 repo notebooks has an answer cell.
- There is no validation split, and the 190 checkpoints are unreleased.

The only public answers are the authors' example challenge ([[4,2,2]] QEC, not one of the 70) and its checkpoints. Two offline checks run inside the sandbox on every `ok` answer:
- **`local_check`**, the example items only. `--example` runs `quantum_error_correction_main` and its two distinct checkpoints (`_sub_1`, `_sub_2`; checkpoint 3 repeats the main question). The extracted answer is executed and compared with the golden SymPy expression. This validates the template → `finish` → extraction → execution path, but not the authors' grader (its tolerances, function test cases or judge).
- **`format_check`**, every answer:
  - is the template's `...` placeholder still there;
  - does the code execute and define `answer`;
  - does `answer` return something other than a placeholder. This is checked when the function takes no arguments or the template defines them at module level, which holds for 67 of the 70; Challenge_45 and Challenge_62 take test-case arguments.

  `summary.json` counts the failures. This says nothing about correctness.

## Consensus proxy (the default grading)

The default reference is DeepSeek V4.1 Flash 1017 (Max) (user decision 2026-11-02; no closed API). The assessment and cost are on issue #32.

**How `ref-ds41-max` was run** (user decision 2026-11-02, overnight):
- **Server:** the task-synthesis production DeepSeek server, `serving:gen-ds5` (4×B200), shared with its production load.
- **Settings:** `reasoning_effort: max`, AA's (Max) variant; 70×5 plus the 3 example items ×5.
- **Load:** at most 64 trajectories, and at most 64 requests in flight through `--max-requests 64`. That limit halves when the server has been queueing for 2 min, meaning more than 1% of its running requests are waiting; a busy production server always shows a few. The limit grows back by 8 after 10 min without queueing, if KV usage is below 0.9. Changes are logged to `load_gate.json`.
  - **Adjusting live:** write `{"limit": N, "min_limit": N, "max_limit": N}` to `<run>/load_gate_override.json`; the gate reads it every 30 s.
  - **First version:** it required exactly 0 waiting, and so sat at 4 requests from 00:32 to 02:30 while gen-ds5 kept a queue of 2-5 among about 1.5k running. The job was restarted at 02:31.
  - **Restart:** from 02:31 the run also uses the sandbox without host proxy variables. The first 54 rows ran with them visible, but the network was off either way.
- **Preemptions:** the load gate also backs off when vLLM preempts more than 10% of its running requests per poll (`num_preemptions_total`), not only on a queue. A trickle of preemptions is normal at high KV use, as admitted requests grow their contexts; only a higher rate is thrash. The first version reacted to any preemption and drove both Qwen runs to about 7 requests within 15 min. Their floors were set to 16 through `load_gate_override.json` at 09:36. From 2026-11-03 ~09:10 the Qwen round-3 runs use `aa-qwen-qa2`'s raw vLLM port (`url:...:32586`) without its admission gate, as the coordinator asked, so our gate is the only cap.
- **Restarts:** `--endpoint-wait-h 6` makes a trajectory wait up to 6 h for a restarted or moved server, following the registry, and never scores that as a model error.
- **Deviation:** that server's protocol caps context, and so output, at 132,000 tokens. AA uses the creator's maximum, so trajectories that would need more show up as `output_limit` or `context_overflow` model errors. The run's `summary.json` counts them.

**How it works:**
- Every `ok` answer gets a `probe`: it is executed in the sandbox and reduced to a canonical fingerprint (see `consensus.py`).
- `consensus reference` clusters equivalent answers per problem across the reference runs. It keeps a problem when the majority class holds at least `--min-share` of the samples, is the majority of at least `--min-models` models, and the problem has at least `--min-samples` samples (default 3, counting `ok` and `model_error`). The sample minimum stops a problem with a single answer from counting as 100% agreement.
- `consensus score` (or `run.py --grade`) reports a run's agreement with the reference on the kept problems: per problem, the share of the run's samples in the reference's majority class, averaged over problems. The CI is taken over problems.
- **All problems (secondary):** `proxy.all_problems` scores every challenge of the run, kept or not, against the reference's most common answer (its plurality class, whatever its share). A problem without a comparable reference answer, or a cutoff, counts 0. It covers the whole benchmark, but on the 26 problems the reference does not answer consistently (ds41-max keeps 44 of 70), its plurality answer is weak evidence.
- **Leave-one-out:** the reference model's own samples are each compared with the majority of the *other* reference samples, never with a class they helped form.

```bash
$PY -m aa_evals.critpt.consensus validate                      # the equivalence checker on the example (passes)
$PY -m aa_evals.critpt.consensus reference --runs ref-ds41-max --name ds41-max --min-share 0.6
$PY -m aa_evals.critpt.consensus score --run-id <run>              # --reference ds41-max by default
```

**Equivalence** follows the paper's grading description as far as it is public:
- **Tolerances:** the authors' expert per-problem tolerances are not public, so each problem's own wording sets them (`consensus.problem_rules`, 17 problems; audit 2026-11-03).
  - "N decimal places", "N-decimal precision" or "N digits after the decimal point" set an absolute tolerance of 10⁻ᴺ: two independently rounded values may differ by one unit in the last place.
  - "N significant figures/digits" sets a relative tolerance of 10¹⁻ᴺ.
  - With several stated precisions (Challenge_52), the loosest applies to the whole answer.
  - Otherwise numbers are compared within a relative tolerance of 1e-4, with **no absolute floor**: an absolute tolerance comes only from stated decimal places. CritPt answers in SI units are often 1e-20 or smaller, and the earlier floor of 1e-12 made any two of them equal (audit 2, PR #144; erratum below).
- **Sign:** "up to an overall sign" (Challenge_6) lets a vector of numbers match its negation; the other fields keep their sign.
- **Constants:** "omit constants" (Challenge_49) compares expressions up to an additive constant.
- **Expressions:** SymPy expressions are compared by their values at 6 fixed points per symbol name.
  - Physical units (`sympy.physics.units`) are replaced by their SI values, one quantity at a time.
  - Points where both sides evaluate to exactly 0 carry no information (an `exp` term underflowing in float64), and at least 2 informative points are needed. An answer, or a component of one, that is 0 at every point is not verifiable and matches nothing.
  - Answers with no numeric value (traces of operators, bra-kets) fall back to the simplified expression tree (`srepr`), so only identical forms match.
- **Composites:** composite answers match only if every element matches. Sets are unordered; intervals compare their endpoints and open/closed flags.
- **Arrays:** NumPy arrays and SymPy matrices are fingerprinted as nested lists, so an array and the equivalent list match.
- **Test-case functions:** functions of test-case inputs are probed on fixed inputs, and are not comparable when that fails.
- **Re-probing:** fingerprints are computed in the sandbox when a row is written (`probe_version` in each row). After the probe changes (`consensus.PROBE_VERSION`), recompute them with `run.py --run-id <id> --reprobe --condor`. The reference records the versions in its data (`params.probe_versions_in_data`).
- **Known limits:**
  - Challenge_30 (bra-ket algebra) only matched identical forms before probe v3: the same expression with a factor distributed over a sum did not match (qwen-full-xhigh, audit 2 minor f). Fixed in probe v3 (#185, above).
  - Challenge_65 (a basis of gauge-invariant operators, a set of `tr(...)` expressions) is compared as a set of expression trees: an equivalent but different basis does not match.
  - Challenge_29: one component of every DeepSeek answer underflows to 0 at all probe points, so the problem is not verifiable and is not kept.
  - Challenge_61 asks for a protocol whose rounded total runtime is graded, so many different answers are equally right. A value comparison can't judge it, and it is rarely kept.

**Biases:**
- The kept problems are those the references solve reliably, close to the complement of AA's hard-40. Weak models are inflated and strong ones hit a ceiling.
- Models that resemble the references get credit for the references' shared errors.
- The proxy is a ranking signal, not an AA-comparable score.

**Erratum (audit 2, 2026-11-04):** until PR #144 the comparison had a fixed absolute tolerance of 1e-12 and no informative-point rule. Rebuilding `ds41-max` from the same 5 samples per problem with the fixed rule drops 4 kept problems: Challenge_25, 56 and 57 (the samples differ by orders of magnitude, or by 11% at 2 significant figures) and Challenge_29 (underflow, above). No problem gains kept status, and no other class changes. Kept goes from 48 to 44. The superseded reference and scores are in `<runs>/_erratum_atol1e-12/`.

| Run | kept, before → after | all 70, before → after |
|---|---|---|
| `qwen-full-xhigh` | 43.75 (21/48) → **38.64 (17/44)** | 30.0 → **24.29 (17/70)** |
| `qwen-full-medium` | 54.17 (26/48) → **52.27 (23/44)** | 37.14 → **32.86 (23/70)** |
| `ref-ds41-max` (leave-one-out) | 84.58 → **85.0** | — |

## Data

- **The 70 challenges:** HF `CritPt-Benchmark/CritPt` at `5b60585` (parquet sha256 `a3b8a3d9…`, checked at start), in `/data/aai/aa_evals/data/critpt/hf_5b60585/`.
- **Prompt templates and example:** GitHub `CritPt-Benchmark/CritPt` at `bc37a22`, in `/data/aai/aa_evals/data/critpt/src/critpt_repo/`.
- **Not for training:** the dataset card forbids training on it. Keep it out of git and out of training data.

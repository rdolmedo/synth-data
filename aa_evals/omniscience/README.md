# AA-Omniscience 1.1

This directory holds the infrastructure for [AA-Omniscience](https://artificialanalysis.ai/evaluations/omniscience) v1.1 (issue #31). It is 15% of the AA Intelligence Index v5.0: accuracy counts for 10% and the non-hallucination rate for 5%.

Models answer short factual questions with no context and no tools, and they are told that abstaining beats a wrong answer. An LLM grader then labels each answer CORRECT, INCORRECT, PARTIAL_ANSWER or NOT_ATTEMPTED.

## Usage

**M3 protocol (#175): one command per model.** Every M3.1 model runs the same way: 5 repeats, a 128k context cap,
the model's own request settings from its serving registry, and the pinned judge.

```bash
aa_evals/omniscience/submit.sh <run-id> --model serving:<name>                              # Super, Gemma, Lightning, ...
aa_evals/omniscience/submit.sh <run-id> --model serving:<qwen> --reasoning-effort medium     # Qwen3.8 Medium (or xhigh)
aa_evals/omniscience/submit.sh <run-id> --model serving:<deepseek>                          # DeepSeek: registry effort max
```

- **Settings:** a `serving:` model's request starts from its registry's `request.overrides`: the lab's sampling,
  thinking on, and the spec's default effort. `--reasoning-effort X` sets the effort both at the top level and in
  `chat_template_kwargs`, and is refused for a model without effort levels. `--preset` (or a `url:` model) uses
  `presets.py` instead. Rules shared with HLE and GDP.pdf: `aa_evals/common/protocol.py`.
- **Context:** `--context-window 131072` (default) caps prompt + output; `max_tokens` = the window minus 2048.
  `--context-window 0` gives the endpoint's full context, for Qwen3.8 calibration runs only.
- **Judge:** with no `--judge`, the benchmark's pinned arm (`aa_evals/judges/pins.py`): `qwen38-27b-medium-pin` until
  study s3 (#41) reports; final pick against DeepSeek in M3.5.
- **AA reference:** found from the served name and effort, e.g. `Qwen3.8 27B (Medium)`.
- **summary.json** adds a `protocol` block (model, effort, thinking, AA variant, context cap, repeats, judge arm, code
  SHA) and `n_flagged` (the M3.4 trace judge's `flags.jsonl`; null until it has read the run).
- **Trace judge (M3.4, #177):** run it next to the benchmark, in tmux:
  `python -m aa_evals.tracejudge watch --run-dir <run dir> --judge serving:aa-judge-gemma4-31b`
  (`judge ... --fold` for a finished run). It writes `flags.jsonl`, and the runner folds `n_flagged`,
  `n_flagged_by_category` and `tracejudge` into summary.json through `aa_evals.tracejudge.fold_summary`.

Earlier usage (smokes and regrades):

```bash
# Smoke: 60 questions (10 per domain), Super answering and judging itself
python -m aa_evals.omniscience.run --model serving:aa-super --judge serving:aa-super --run-id smoke-super --limit 60

# Same thing as an HTCondor CPU job, run from a frozen copy of the code (preferred on the shared host)
aa_evals/omniscience/submit.sh smoke-super --model serving:aa-super --judge serving:aa-super --limit 60

# Full public set; the judge defaults to serving:aa-judge (a self-hosted open model)
aa_evals/omniscience/submit.sh dsv41-max --model serving:<name>

# Regrade an earlier run's answers with another judge (no model calls)
python -m aa_evals.omniscience.run --answers-from smoke-super --judge serving:<judge> --run-id smoke-super-j2 --limit 60
```

**Judge:** the user decided on 2026-11-02 not to use a closed API judge. The grader is a self-hosted open model, served as `serving:aa-judge` (the default `--judge`). Which model it is depends on a judge test (#31). AA's own grader, `openai:gpt-5.6-luna`, still works, but every `openai:` spec is refused unless `--allow-external-api` is passed, and that flag needs a fresh approval.

- **Model specs** follow the contract in `aa_evals/README.md` (`serving:<name>`, `url:<base>@<served_name>`), plus `openai:<model>` for the OpenAI API (refused without `--allow-external-api`).
  - With `serving:`, the registry is re-read on every retry, so a restarted server's new URL is picked up.
  - `openai:` requests go through the proxy in the environment. Cluster endpoints are reached directly.
- **Generation settings** come from a preset in `presets.py`, chosen by served name or by `--preset`. Each preset mirrors the AA variant it is compared against: sampling, reasoning effort and chat-template kwargs. Flags (`--temperature`, `--top-p`, `--reasoning-effort`, `--max-tokens`, `--chat-template-kwargs`, `--extra-body`) override single fields.
  - `--max-tokens auto`, the preset default, is the endpoint's `max_model_len` minus 2048.
  - Presets: `nemotron-3.5-super-vl`, `qwen3.8-27b-xhigh` (AA's "Qwen3.8 27B (Xhigh)": thinking at xhigh, T 1.0, top_p 0.95, top_k 20; the default for served name `qwen38-27b`), `qwen3.8-27b-medium` (the same at medium effort), the DeepSeek and GLM variants, and `gpt-5.6-luna`.
- **Judge arm:** `--judge-arm <name>` takes the judge server and settings from `aa_evals/judges/arms.py` and records the name as `judge.arm`. Qwen runs are graded with the pinned arms `qwen38-27b-medium-pin` and `super-full-pin` (#41, audit 1).
- **Judge load:** `--judge-concurrency N` caps judge requests in flight independently of `--concurrency` (keep a shared judge server at 16 or fewer).
- **No tails:** `--deadline-s N` ends the run N seconds after it starts. Items still unanswered then count as 0 (`cutoff`, see Statuses), and the summary is written right away. Answers already in hand at the deadline, including those waiting for a judge slot, are still graded. In-flight model requests are abandoned, not cancelled. A resume with the same run id retries the cutoffs.
- **Timeouts:** `--timeout` bounds one model request, including any time it waits in a serving gate's queue (recorded per item as `gate_queue_ms` from `X-Gate-Queue-Ms`). Set it long and let `max_tokens` bound a generation: a timeout on a healthy server is scored as the model's runaway (`model_error`).
- **Resume:** re-running the same command resumes the run.
  - Finished items are kept and infra errors are retried.
  - An item whose answer exists but whose grading failed is regraded without asking the model again.
- **Stopping:** after 25 infra errors in a row on the model side, the run stops submitting work and exits with code 3, so you can resume it once the endpoint is back.
- **Tests:** `python -m aa_evals.omniscience.test_omniscience`. They are offline and include an end-to-end run against a fake server.

## Data

| `--dataset` | HF repo @ revision | Questions | Use |
|---|---|---|---|
| `v1.1` (default) | `ArtificialAnalysis/AA-Omniscience-Public-v1.1` @ `1f9cbce` | 300 | The 10% public slice of v1.1's 3,000 questions, with the same domain split |
| `v1` | `ArtificialAnalysis/AA-Omniscience-Public` @ `e4883ed` | 600 | **Not comparable** to AA's current numbers: v1.1 dropped all of these questions |

- The files live in `/data/aai/aa_evals/data/omniscience/` and are checked against a pinned SHA-256 when loaded. They are fetched on first use.
- Both sets are Apache-2.0 and ungated. They are eval-only: never commit them, and keep them out of training data.
- AA reports how the v1.1 public set tracks the private one over 86 models at launch:
  - Index: r = 0.991, mean gap (public − private) −0.2 points.
  - Accuracy: r = 0.994, gap −0.2 pp.
  - Hallucination rate: r = 0.993, gap 0.0 pp.
- `--limit N` takes the first N of a fixed order: each domain is shuffled with `--subset-seed`, then the domains are interleaved. A subset is a smoke test, not a score.

## Metrics

c, i, p and a count the CORRECT, INCORRECT, PARTIAL_ANSWER and NOT_ATTEMPTED grades, and N = c + i + p + a.

| Metric | Formula | Weight in the AA index |
|---|---|---|
| AA-Omniscience Accuracy | c / N | 10% |
| AA-Omniscience Non-Hallucination Rate | 1 − i / (N − c) | 5% |
| AA-Omniscience Index (net score) | 100 · (c − i) / N | reported, not weighted |

- **These formulas reproduce AA's published numbers exactly.** For Nemotron 3.5 Super VL with N = 3000:
  - accuracy 21.17% gives c = 635;
  - an index of −2.87 gives i = 721;
  - 1 − 721/2365 = 69.51%, which is AA's non-hallucination rate.
- Partial answers count like abstentions: they are not correct and not hallucinations.
- In `summary.json`:
  - `score` is accuracy, with `metric` = "AA-Omniscience Accuracy".
  - `metrics` holds all three, each with a 95% bootstrap CI over items. An item's repeats are resampled together.
  - `grades` holds the four counts, and `aa_reference` holds AA's three numbers for the model.

## Statuses

| Status | Meaning | Scored |
|---|---|---|
| `ok` | The model answered and the grader returned a parseable letter | yes |
| `model_error` | The answer was cut off at `max_tokens` (`finish_reason=length`), or no response came within `--timeout` while the server stayed healthy (`finish_reason=timeout`: a runaway generation, not retried) | yes, graded as is; an empty answer counts as NOT_ATTEMPTED |
| `infra_error` | The model or grader endpoint failed after retries (connection, 429/5xx, a stale registry, a timeout while the server is down), or the grader output was unparseable 3 times | no: excluded and counted; a run with more than 2% is `valid: false` |
| `cutoff` | Still unanswered at `--deadline-s` (no tails) | yes: not correct for accuracy and the index, left out of the non-hallucination denominator; the `*_cutoff_incorrect` metrics count it INCORRECT; `n_cutoff` in the summary; a resume retries it |

- An empty answer is graded NOT_ATTEMPTED without a grader call. The grader prompt itself says "e.g. empty response". Such grades show `grade_source: rule_empty`.
- The grader sees only the final `content`. Reasoning (`reasoning_content`, `reasoning`, or an inline `<think>` block) is stripped first.

## Deviations from AA

| | AA (v1.1, index v5.0) | Ours | Effect |
|---|---|---|---|
| Questions | 3,000 (2,700 private + these 300 public) | the 300 public v1.1 questions | AA measures a −0.2 pp mean gap. Sampling noise at N = 300 is about ±4.6 pp on accuracy (95%) |
| Answer prompt | `OMNISCIENCE_ANSWER_PROMPT` as the system message | the same, verbatim | none |
| User turn | not published | the bare question text | probably none |
| Grader | GPT-5.6 Luna, medium reasoning | a self-hosted open model, pinned per benchmark (`judges/pins.py`: `qwen38-27b-medium-pin`, Qwen3.8 27B FP8 at medium, until study s3 #41 reports). No closed API judge, by user decision | a judge effect, measured in #41 by regrading the same answers with `--answers-from`. For Qwen3.8 runs the judge is the same checkpoint as the model |
| Grader prompt | `OMNISCIENCE_GRADER_TEMPLATE` | the same, verbatim (a test checks it against the card), sent as one user message | none |
| Serving | the provider's API | our vLLM build (Super is NVFP4) | quantization and serving differences |
| Sampling | the lab's recommended temperature, else 0.6 for reasoning models | the lab's recommended values from the serving registry (Super: 1.0 / 0.95, thinking on) | none intended |
| Max output tokens | the creator's maximum | the M3 128k context cap minus 2048 (`--context-window`; full context only for Qwen3.8 calibration runs) | small: the longest Qwen3.8 answer was 68k tokens (xhigh, #31 r3); truncations are counted |
| Repeats | 1 (over 3,000 questions) | 5 over the 300 public questions (M3 protocol, #173) | none for the mean; a tighter CI |
| Text-only models | n/a (text benchmark) | n/a | none |
| API failures | up to 30 retries, then manual review | 8 retries with backoff per call, then `infra_error` (excluded and counted); a re-run resumes | none for a valid run |

## Judge cost

The user declined closed API judges on 2026-11-02, so this is for reference only. A self-hosted judge costs GPU time: the 60-question smoke took one B200 a few minutes.

- GPT-5.6 Luna costs $0.20 per 1M input tokens ($0.02 cached) and $1.20 per 1M output.
- The grader prompt is about 2k tokens and is a shared prefix. Output at medium reasoning is roughly 0.5–2k tokens.
- A 300-question run therefore costs under $1.

## Smoke results

**`smoke-super-60`**, 2026-11-02:
- Setup: Nemotron 3.5 Super VL (NVFP4, `serving:aa-super`, one B200) on 60 v1.1 questions (`--limit 60`, 10 per domain). Super was also the judge.
- Run: a condor CPU job at concurrency 16 took 307 s.
- Health: 0 infra errors, 0 truncations.
- Tokens: completion tokens per question had a median of 2.6k and a max of 20.7k, 254k in total.

| | Ours (60 Qs, Super judge) | AA (3,000 Qs, GPT-5.6 Luna judge) |
|---|---|---|
| Accuracy | 20.0 [10.0, 31.7] | 21.2 |
| Non-hallucination rate | 68.8 [55.1, 81.6] | 69.5 |
| Omniscience Index | −5.0 [−23.3, 11.7] | −2.9 |
| Grades c / i / p / a | 12 / 15 / 0 / 33 | — |

- AA's numbers are inside our CIs.
- The Super judge never chose PARTIAL_ANSWER, even for verbose answers. Regrading the same answers with `--answers-from smoke-super-60 --judge <spec>` measures how much a judge moves the score.

**`super-v11-300`**, 2026-11-02, the full public v1.1 set:
- Setup: the same model and judge as the smoke, but `aa-super` was restarted with a 1M-token context (the smoke ran at 131k). All 300 questions, concurrency 48, `--timeout 21600`.
- Run: 18,347 s wall (5.1 h). Without the straggler below, the job would have finished in well under an hour.
- Health: 0 infra errors.
- Tokens: completion tokens per question had a median of 2.7k, 2.6M in total. Answer latency had a median of 51 s and a p95 of 388 s.
- The straggler was item 205. Super generated 1,046,528 tokens, filling the whole context, in 17,909 s and never answered. It is recorded as truncated (`model_error`) and graded NOT_ATTEMPTED by the empty-answer rule.
- Judge: 299 calls; 470k prompt tokens and 151k completion tokens.

| | Ours (300 Qs, Super judge) | AA (3,000 Qs, GPT-5.6 Luna judge) |
|---|---|---|
| Accuracy | 23.0 [18.7, 28.0] | 21.2 |
| Non-hallucination rate | 66.7 [60.6, 72.4] | 69.5 |
| Omniscience Index | −2.7 [−10.3, 5.0] | −2.9 |
| Grades c / i / p / a | 69 / 77 / 2 / 152 | — |

- AA's numbers are inside our CIs again.
- Per domain (c / i / p / a): SEM 22/15/1/22, Software Engineering 17/27/0/16, Finance 14/8/0/23, Health 3/17/0/25, Humanities 7/6/0/32, Law 6/4/1/34.
- With an unbounded context, one runaway generation sets the wall time. When wall time matters, use a `max_tokens` cap or `--timeout`. Either one trades the occasional very long answer for a bounded run.
- The judge settings for the planned default `serving:aa-judge` are being chosen in #41.

## Lessons

- **The public set changed with v1.1.** `AA-Omniscience-Public`, with 600 questions, is v1. v1.1 removed all of those questions and ships its own 300-question public set, `AA-Omniscience-Public-v1.1`. Scores on the v1 set are not comparable to AA's current numbers.
- **AA changed its grader** from Gemini 2.5 Flash (in the paper) to Gemini 3 Flash, then to GPT-5.6 Luna (medium) as of index v4.1.1. The grader prompt is the same in both dataset cards.
- **AA's leaderboard exposes only accuracy, the non-hallucination rate and the net index.** The definition hallucination rate = i / (i + p + a) is what reproduces all three together.

### Batch judging after generation

Use the same run ID for both stages. Generation records answers with `status=pending` and `score=null`, without resolving or contacting a judge. Its summary remains incomplete and invalid as a score, while `generation_complete` reports whether generation finished. Resume generation with the same command before grading any missing outputs.

```bash
python -m aa_evals.omniscience.run --stage generate --model serving:aa-model --run-id model-omni
python -m aa_evals.omniscience.run --stage grade --run-id model-omni --judge serving:aa-shared-judge
```

Grading uses the stored model settings, context, subset and generation code SHA. The original model server may be offline; missing generations remain infra outcomes and are never regenerated by the grading stage. The grading commit is recorded separately as `stage_git`. `--stage all` retains the original combined behavior, and `--answers-from` can still seed another run for a separate judge arm. The CPU submit shell script forwards these arguments.

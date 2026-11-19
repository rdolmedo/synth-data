# HLE-Diamond (AA Intelligence Index v5.0, weight 10%)

Issue #33. This directory evaluates a model served on our cluster on HLE-Diamond, following Artificial Analysis's methodology. The contract is in `../README.md`.

## Judge: self-hosted, by user decision
- **Decision (user, 2026-11-02):** no closed-API judge. The judge is an open model that we host ourselves.
- **Default:** `serving:aa-judge`, the judge spec shared by every AA eval. The user chooses its model after a judge test.
- **Override:** `--judge serving:<name>` or `url:<base>@<name>`.
- **AA's judge:** `openai:gpt-5.6-luna` stays in the code but is refused unless `--allow-external-api` is passed. That spend is not approved.
- **Effect:** every score we report is graded by a different judge than AA's. The judge test should measure how often the two disagree.

## AA's method (methodology page, read 2026-11-02)
- **Data:** `cais/hle-diamond`, all 1,000 questions.
  - 500 reasoning and 500 knowledge questions, given by the `partition` column.
  - 272 questions include an image.
- **No tools.** 1 repeat, scored as pass@1. AA reports the score overall and per partition.
- **Prompt:** the official HLE system prompt as first published, in two variants.
  - Exact-answer questions are asked for `Explanation: / Exact Answer: / Confidence:`.
  - Multiple-choice questions are asked for `Explanation: / Answer: / Confidence:`.
  - Models without system-prompt support get the prompt prepended to the user turn.
- **Images:** image-capable models get the image after the question text. AA downscales it only where a provider's limit requires.
  - Text-only models get only the text and are still graded on all 1,000 questions.
- **Judge:** GPT-5.6 Luna at medium reasoning, with the official HLE judge prompt and structured response. AA uses the same equality checker for HLE and AA-LCR.
- **General settings:**
  - Reasoning models run at temperature 0.6 unless the lab recommends another value.
  - Max output is the creator's disclosed limit.
  - API failures are retried up to 30 times, and questions that still fail are reviewed by hand.

## Question set (#348, user decision 2026-11-11)
- **`--task-set text-only-v1` (default):** the 728 HLE-Diamond questions without an image. `--limit`, `--stratify`
  and `--ids` select within the set, and an `--ids` file with an image question is refused with a hint.
- **`--task-set all`:** all 1,000 (AA's set). Every run before #348 used it: a resume or `--stage grade` of such a run
  keeps its recorded items, and resuming one without `--task-set all` is refused (`config.json` without
  `dataset.task_set` means `all`).
- `config.json` records `dataset.task_set` and `dataset.n_task_set`; `summary.json` records `task_set`.
- **Re-scoring a stored run** on a set needs no model or judge: `python3 -m aa_evals.hle.rescore --task-set
  text-only-v1 --out <dir> <run_id> ...` restricts its `items.jsonl` rows to the set and writes
  `<dir>/<run_id>.summary.text-only-v1.json` (the run directory is not touched). On `--task-set all` it reproduces
  each stored score exactly.
- **M3.6 runs on the 728** (judge `qwen38-27b-medium-pin`; `/data/aai/aa_evals/audits/hle_textonly_20261111/`):

  | Model | All 1,000 | Text-only 728 | Change | Image 272 |
  |---|---|---|---|---|
  | DeepSeek V4.1 Flash Max (M3.5, text-only) | 20.9 ± 2.5 | 26.7 ± 3.2 | +5.8 | 5.5 |
  | Qwen3.8 Xhigh | 12.1 ± 2.0 | 12.5 ± 2.4 | +0.4 | 11.0 |
  | Super VL | 8.7 ± 1.8 | 10.2 ± 2.2 | +1.5 | 4.8 |
  | Qwen3.8 Medium (M3.4) | 8.2 ± 1.7 | 8.2 ± 2.0 | +0.0 | 8.1 |
  | Qwen3.6 | 7.5 ± 1.6 | 8.0 ± 2.0 | +0.5 | 6.3 |
  | Gemma 26B | 5.7 ± 1.4 | 5.4 ± 1.6 | −0.3 | 6.6 |
  | Qwen3.5-4B | 5.2 ± 1.4 | 4.7 ± 1.5 | −0.5 | 6.6 |
  | Lightning (text-only) | 4.0 ± 1.2 | 4.7 ± 1.5 | +0.7 | 2.2 |
  | GLM-5.3-Flash (250-item subset) | 17.2 ± 4.7 (250) | 18.2 ± 5.8 (170) | +1.0 | 15.0 (80) |

## Usage

**M3 protocol (#175): one command per model**, 1 repeat over the question set (since #348 the 728 text-only questions; before, all 1,000), a 128k context cap, the model's own
request settings from its serving registry, and the pinned judge (`aa_evals/common/protocol.py`):

```bash
python3 -m aa_evals.hle.submit --run-id <id> -- --model serving:<name>                          # any M3.1 model
python3 -m aa_evals.hle.submit --run-id <id> -- --model serving:<qwen> --reasoning-effort medium # Qwen3.8 Medium / xhigh
```

- **Context:** `--context-window 131072` (default). When the endpoint's own context is larger, each request gets
  `max_tokens` = 131,072 − (the prompt text by the endpoint's `/tokenize` + 64 + 5,120 per image). The image
  allowance bounds HLE's largest image prompts (4.8k tokens on Qwen3.8, 4.0k on Super), so prompt + output never
  passes the cap; `summary.json` counts rows over it (`n_over_window`, expected 0). `--context-window 0` = the
  endpoint's full context, for Qwen3.8 calibration runs only.
- **Effort:** `--reasoning-effort X` sets the effort at the top level and in `chat_template_kwargs` (Qwen3.8's
  template reads only the kwarg); refused for models without effort levels (Super, Gemma, Lightning, Qwen3.6).
- **Text-only models** (DeepSeek served `--language-model-only`, Lightning): detected from the serving launch argv;
  the images are dropped and the item is still graded, as AA does for models without image input (methodology
  page). `config.json` records `image_inputs`.
- **AA reference:** the slug follows the served name and effort, so a Medium run compares with `qwen3-8-27b-medium`.
- **Judge:** with no `--judge`, the pinned arm for HLE (`aa_evals/judges/pins.py`; `qwen38-27b-medium-pin` until
  study s3 on #41 reports).
- **summary.json** adds `judge_arm`, `n_over_window`, a `protocol` block (model, effort, thinking, AA variant, context
  cap, repeats, judge arm and settings key, code SHA, image inputs) and `n_flagged` (M3.4 trace judge; null until it
  has read the run).
- **Trace judge (M3.4, #177):** run it next to the benchmark, in tmux:
  `python -m aa_evals.tracejudge watch --run-dir <run dir> --judge serving:aa-judge-gemma4-31b`
  (`judge ... --fold` for a finished run). It writes `flags.jsonl`, and the runner folds `n_flagged`,
  `n_flagged_by_category` and `tracejudge` into summary.json through `aa_evals.tracejudge.fold_summary`.

Earlier usage:
```bash
# once: download the pinned revision to /data/aai/aa_evals/data/hle/ (mode 700; never commit)
python3 -m aa_evals.hle.data fetch --dataset cais/hle-diamond
python3 -m aa_evals.hle.data stats                      # aggregate counts only

# smoke, run in place (light: an async HTTP client); the judge defaults to serving:aa-judge
python3 -m aa_evals.hle.run --model serving:aa-super --run-id smoke-super --limit 30

# full run as a condor CPU job from a frozen copy of HEAD (commit first)
python3 -m aa_evals.hle.submit --run-id super-full -- --model serving:aa-super

# stratified subset (proportional over partition x modality x answer type) with a no-tails cutoff
python3 -m aa_evals.hle.submit --run-id qwen-r1 -- --model serving:aa-qwen --limit 100 --stratify \
    --cutoff-min 110 --concurrency 48 --judge serving:aa-judge-qwen38-27b

# re-judge another run's generations with a different judge (no model calls; the model server may be gone)
python3 -m aa_evals.hle.run --model serving:aa-super --run-id smoke-super-hle40-aajudge --dataset cais/hle --limit 40 \
    --generations-from smoke-super-hle40 --judge serving:aa-judge

python3 -m unittest aa_evals.hle.test_hle               # synthetic items, no data needed
```
- **Python:** `/home/user/python-3.11/bin/python3`, which has httpx, pyarrow and huggingface_hub.
- **Outputs:** written to `/data/aai/aa_evals/runs/hle/<run_id>/`. Only ids and aggregates from them may be shared.
  - `config.json` holds the git SHA, the model spec and registry, the exact sampling body, the prompt policy, the dataset revision and sha256, and the judge spec and settings.
  - `items.jsonl` has one row per item × repeat: `score`, `status`, `error_stage`, partition, modality, finish reason, tokens, and the judge's verdict and confidence.
  - `transcripts/<id>.r<k>.json` holds the model's content and reasoning, usage, every attempt, and the judge's output.
  - `summary.json` has pass@1 with a Wald CI95 and the slices by partition, modality, answer type and category. It also holds HLE's RMS calibration error, the infra-error count, `valid` (at most 2% infra errors), and `aa_reference`.
- **Runs root:** `AA_HLE_RUNS_ROOT` (and `AA_HLE_CODE_ROOT`) move new runs; the 11-02..11-13 runs are in `/data/aai/aa_evals/runs_fallback/hle`. Runs and logs stay on /data, never /home.
  - `submit.py` passes the root on to the job.
  - `--generations-from` finds a source run in any root.
  - The fallback directory is mode 700: it holds question-bearing transcripts.
- **Resume:** run the same `--run-id` again. Finished items are kept, and infra errors and missing judgments are retried. A changed model, sampling, prompt, dataset or judge config is refused unless you pass `--force`.
- **No-tails cutoff:** `--cutoff-min M` stops waiting M minutes after the run starts.
  - Every generation still queued or in flight is cancelled, which closes the HTTP request so vLLM aborts it. Each one scores 0 as a `model_error` with `finish_reason: cutoff`, and `summary.json` reports `n_cutoff`.
  - Cut-off items get no transcript, so a resume regenerates them.
  - Generations finished before the cutoff are still judged.
- **Re-judging (`--generations-from`):** the run never calls the model.
  - An item the source run cut off is carried over as a cut-off 0.
  - An item without a finished source generation (still running, or an infra error) is an `infra_error` here. Rerun the same re-judge run id once the source is finished or resumed, and only the new generations are seeded.
- **Adaptive in-flight limit (`--metrics-url http://<host>:<port>/metrics`):** `--concurrency` becomes the maximum, and the limit follows vLLM.
  - It drops to what vLLM actually runs when vLLM queues (waiting > 0) or preempts.
  - It rises by 4 while KV < 0.85 and there is more work.
  - In-flight requests are never cancelled, so only the scheduling changes, not the results.
  - Why: single-turn reasoning generations grow their KV during decode. A fixed cap that fits while answers are short (Qwen medium at 80: KV 0.93, 4.8k tok/s) thrashes once the long tails dominate (KV 0.99, 35 waiting, 141 preemptions within 10 min).
  - Point it at the raw vLLM port, not an admission gate. Our gate doesn't cancel a request whose client disconnected, so a killed run leaves orphaned generations behind it (qa1 had to be restarted to flush about 700).
- **A second endpoint on the same items (`--reverse`, `python3 -m aa_evals.hle.merge`):**
  - Run the same items on another server as a separate run id with `--ids` and `--reverse`. It starts from the end of the order while the first run works from the front, so they overlap only where they meet.
  - `merge --out <id> --sources <first> <second>` takes, per item, the first source with a finished generation. The rule never looks at the score, so a later source only fills items an earlier one did not finish.
  - It hard-links the winning transcripts, so `--generations-from <merged>` re-judges the result, and it records the sources in `config.json` and `summary.json`.
  - A stratified subset of 600 contains the stratified 300 (quotas grow with N, and each stratum takes its first items by sha256). So `--limit 600 --stratify` minus `--limit 300 --stratify`, merged with the 300, is the stratified 600.
- **Model errors are labelled** (`model_error_kind` per row, `model_errors_by_kind` in `summary.json`). Every one scores 0 and is the model's outcome, never infra (infra is only server or transport failure):
  - `cutoff`: unfinished at the no-tails cutoff.
  - `length`: a runaway generation that hit `max_tokens` or the served context with no final answer.
  - `reasoning_unclosed`: EOS before `</think>` (see the Qwen3.8 note).
  - `empty_stop`: a normal stop with an empty answer.
  - `empty_<finish>`: any other finish reason, labelled with it.
  - The label is derived from row fields, so `scoring.model_error_kind` also labels rows from older runs.
- **Judge settings (pinned for every judged benchmark, #41 audit):**
  - Qwen judge (`serving:aa-judge-qwen38-27b`): `--judge-reasoning-effort medium`, the registry's sampling (T 1.0, top_p 0.95, top_k 20) and `max_tokens` 32768. Output that hits the cap is off-schema and re-asked (`--judge-format-attempts`).
  - Super judge (`serving:aa-super`): the registry defaults (T 1.0, top_p 0.95, thinking on) and `max_tokens` 32768. Pass no `--judge-reasoning-effort`.
  - Lesson: a served judge used to ignore `--judge-reasoning-effort`. Qwen's template reads the effort from `chat_template_kwargs`, which the registry pins to xhigh, so a top-level `reasoning_effort` alone changes nothing. The flag now sets both, and `config.json` records the exact judge body.
  - Each judgment stores a `judge_key` (a hash of the judge spec and settings). `--rejudge` re-grades on resume every judgment made with another judge or other settings. A `--force` resume that changes the judge turns it on by itself, so old and new judgments never mix. The model is never re-asked.
- **Stratified subsets:** `--limit N --stratify` splits N over partition × modality × answer type in proportion to the full set, with at least 1 item per stratum. Each stratum takes its first items by sha256(id). The subset's unweighted mean estimates the full-set score.

## Generation settings per model
Sampling defaults to the serving registry's `request.overrides`. These are the lab-recommended values pinned in the teacher spec, which is AA's own rule. CLI flags override them, and the exact body is written to `config.json`. `max_tokens` is the rest of the 128k window (above); with `--context-window 0` it is left unset, so vLLM uses the rest of the served context, our closest equivalent of AA's "creator's limit".

| Model (AA variant) | AA slug | Command additions |
|---|---|---|
| Nemotron 3.5 Super VL | `nemotron-3-5-super-vl` | none (registry: temperature 1.0, top_p 0.95, `enable_thinking`) |
| DeepSeek V4.1 Flash 1017 (Max) | `deepseek-v4-1-flash-1017` | none (registry: `reasoning_effort=max`; text-only, detected) |
| DeepSeek V4.1 Flash 1017 (High) | `deepseek-v4-1-flash-1017-high` | `--reasoning-effort high` |
| Qwen3.8 27B (Medium) / (Xhigh) | `qwen3-8-27b-medium` / `qwen3-8-27b` | `--reasoning-effort medium` / none (registry: xhigh) |
| Nemotron 3.5 Lightning, Qwen3.6 35B A3B, Gemma 4 26B A4B, Qwen3.5 4B | from `protocol.AA_MODELS` | none (registry: thinking on) |
| GLM-5.3-Flash | `glm-5-3-flash` | none (registry: `reasoning_effort=max`) |
| a text-only `url:` model | | `--text-only` |

## How failures are counted
- **`ok`:** the judge graded a final answer, scoring 1 or 0.
- **`model_error`** (scores 0): the model returned no final `content`. Usually it spent its budget reasoning (`finish_reason=length`), or it ended without an answer. AA's grader would see an empty response too.
- **`infra_error`** (excluded and counted):
  - The model call still failed after 30 attempts. Retried causes are transport errors, timeouts, 408/409/429/5xx, malformed bodies, and a stale registry while the server restarts. A deterministic 4xx is tried at most 3 times.
  - Or the judge failed, including judge output that stays off-schema after 5 re-asks.
  - A run with more than 2% infra errors has `valid: false`, and resuming it retries them.
- **`reasoning_unclosed`** (per row; `n_reasoning_unclosed` in the summary) marks a `model_error` where the model ended its turn before `</think>`. The finish reason is `stop`, and every completion token except the end token is reasoning, so vLLM's reasoning parser returns empty content.
  - **On Qwen3.8-27B it is the model's behavior at medium effort, not a serving defect** (#33, 2026-11-03).
    - At medium it hits 30-47% of generations of at least 29k tokens, and it does so equally with MTP on or off, and with an fp8 or bf16 KV cache:

      | Arm | Unclosed / finished ≥29k |
      |---|---|
      | MTP off | 5/16 |
      | MTP on, fp8 | 7/21 |
      | bf16 KV | 14/46 |
      | round 1 | 9/19 |

    - At xhigh, on the same recipe, it hits 8 of 152 (5%), even past 120k tokens.
    - `stop_reason` shows the normal `<|im_end|>`. The cut is sometimes right after an answer written inside the reasoning, and sometimes mid-expression.
    - Qwen's template adds no reasoning instruction at medium.
  - These items score 0, as AA's grader would see an empty response. The medium full run has 66 of them out of 1,000.
- **Judge diagnostics:** private transcripts save `judgment.calls`, one record per
  `ChatClient.chat` invocation, including the exact returned JSON, content, reasoning,
  response ID, finish reason and usage. Missing usage stays `null`. Each record pins
  the caller-body SHA-256, requested max tokens and structured-output schema. Raw
  responses survive format failures and later client errors. These invocation records
  are **not physical request counts**: a half-open `transport_attempts_range` points
  into the existing `attempts` list, which also contains local format-error and gate
  summary rows. Failed internal transport attempts may still have unknown usage.
- **What the judge sees:** only the final `content`. The reasoning is stored in the transcript but never judged.

## Deviations from AA
| Aspect | AA | Ours | Why / effect |
|---|---|---|---|
| Data | `cais/hle-diamond`, all 1,000 | pinned at revision `04eeb7e`. **New runs (#348): the 728 questions without an image** (`--task-set text-only-v1`, default); `--task-set all` keeps the 1,000 (M3.0–M3.6 runs) | Every model is served without vision (#348, user 2026-11-11). Without the image, the 272 image questions are near the floor (text-only Lightning, M3.6: 6 of 272) and take 9–27% of each model's output tokens. Scores on the 728 run higher than on the 1,000 for text-only models (DeepSeek Max +5.8 pp) and move ≤1.5 pp for the rest; see the re-score below |
| System prompt | the two-variant official prompt | same (`--system-prompt by_type`) | The official two-prompt code tested `answer_type == 'exact_match'`, but the data says `exactMatch`, so the code sent the MC prompt to every question. `--system-prompt code` reproduces that. The two prompts differ only in "Exact Answer:" vs "Answer:". |
| Judge | GPT-5.6 Luna (medium), official prompt | **a self-hosted open judge**, pinned by arm (`judges/pins.py`: `qwen38-27b-medium-pin` until study s3 on #41 reports; user decision 2026-11-02: no closed API judge), with the official prompt | This is the largest deviation, because grading differs from AA's. Strict structured output additionally constrains confidence to the prompt's 0–100 range; see the schema deviation below. The judge's token cap (including reasoning) is 32,768; AA's is unknown. Luna is still possible with `--judge openai:gpt-5.6-luna --allow-external-api`, but it is not approved. |
| Sampling | temperature 0.6 unless the lab recommends otherwise | the lab-recommended values from the serving registry; 0.6 when there are none | follows AA's rule |
| Max output | the creator's disclosed limit | the rest of the M3 128k window (prompt + output ≤ 131,072; #173). Before M3.2: the rest of the served context (Super 1M, Qwen3.8 262k) | Long reasoners are cut short where AA allows more: in the 262k r3 runs of Qwen3.8, 56 of 1,000 medium generations went past 128k (6 of them correct) and 39 of 300 xhigh ones (11 correct), so the cap costs up to 0.6 pp at medium and about 3.7 pp at xhigh |
| Text-only models | the question text only, graded on all items | the same: images dropped when the serving launch has no image input (DeepSeek is served `--language-model-only`, although AA lists it as image-capable, so its 272 image items lose the image here) | DeepSeek's image items are harder for it than on AA |
| Images | downscaled only where a provider limit requires | sent as-is; vLLM's processor applies the served model's resizing. Qwen3.8 27B servers run `--mm-processor-kwargs '{"max_pixels":4194304}'`, so 12 of the 272 Diamond images (4.2-41.6 MP) are downscaled to at most 4 MP, aspect kept | minor: affects only those 12 items |
| Persistent API failures | reviewed by hand; the result is not published | `infra_error`, excluded and counted; >2% makes the run invalid | |
| Repeats | 1 | 1 (M3 protocol: HLE-Diamond gets 1, #173) | |

## Smoke runs
| Run | Data | Model / judge | Result |
|---|---|---|---|
| `smoke-super-hle40` (2026-11-02, condor job 17815276, 17.7 min) | `cais/hle`, 40 items by sha256(id) (8 image, 13 MC); Diamond access was pending | Super VL (`serving:aa-super`, images on, temperature 1.0 / top_p 0.95, thinking on), also the judge | pass@1 **25.0 ± 13.4**; 0 infra errors; 1 `model_error` (reached the 131k served context); mean / max completion 34.5k / 131k tokens. AA's HLE for Super VL is 30.7 (2,158 text-only questions, so not the same set). |

## Full runs
| Run | Model / judge | Result |
|---|---|---|
| `super-diamond-full` (2026-11-02, condor job 17815546; 8.1 h wall, 48 in flight) | Super VL (`serving:aa-super` served at 1M context, images on, registry sampling), with **aa-super as a provisional judge** (it graded its own answers) | pass@1 **10.5 ± 1.9** vs AA's **8.8**. Reasoning 13.8, knowledge 7.2. Text 12.6 (728), image 4.8 (272). exactMatch 9.9, MC 12.8. 0 infra errors. 18 `model_error`: 17 reached the 1M context with no final answer, and 1 stopped with no final answer. Completion tokens: mean 46.5k, median 17.5k, max 1.05M. |

| `qwen-diamond-medium` (2026-11-03; aa-qwen-qa1 tuned recipe, MTP on, fp8 KV; raw vLLM port; adaptive in-flight limit from 09:50; done 12:13) | Qwen3.8 27B at **medium** (registry sampling, `reasoning_effort` medium), full Diamond. Qwen judge pinned (medium, 32k; self-judging) | pass@1 **9.9 ± 1.85** (Super judge: **10.1 ± 1.87**; the judges agree on 910 of 914) vs AA (Medium) **6.7**. Reasoning 14.6, knowledge 5.2. Text 10.2 (728), image 9.2 (272). exactMatch 9.3, MC 12.3. 0 infra errors, 0 cutoffs. 86 `model_error`: 66 `reasoning_unclosed`, 20 `length` (262k context). Completion tokens: mean 25.6k, max 262k. |
| `qwen-diamond-xhigh-strat300` (2026-11-03; aa-qwen-hlex same recipe; raw port, adaptive limit; done 11:36) | Qwen3.8 27B at **xhigh**, a **stratified 300-item subset** (not the full set). Qwen judge pinned | pass@1 **18.7 ± 4.4** (Super judge: **17.3 ± 4.3**) vs AA (Xhigh) **15.3**. Reasoning 28.9, knowledge 8.7. Text 24.0 (217), image 4.9 (82). 0 infra errors, 0 cutoffs. 20 `model_error`: 10 `length` (262k), 10 `reasoning_unclosed`. Mean completion 57.7k tokens. The two judges agree on 276 of 280; the other 4 are Qwen-only yes. |

- **Running Qwen at scale (round 3):**
  - Single-turn reasoning generations grow their KV during decode, so a fixed in-flight cap that fits early thrashes later. Use `--metrics-url` on the raw vLLM port.
  - The admission gate doesn't cancel a disconnected client's requests: about 700 orphaned generations had to be flushed by restarting qa1.
  - 1×B200 holds only about 22-36 of the long tails at once, so the last 10% of items took about half the wall time.
- **Tail rule:** the last 11 items were still generating at 6.2 h, so the interim report (`summary.interim.json`) counted them 0 under the user's no-tails rule. They finished by 8.1 h: 10 with no final answer and 1 judged wrong. The final `summary.json` therefore has the same score.
- **Re-grading:** the score is not comparable to AA's until #41's judge arms re-grade these generations. They use `--generations-from super-diamond-full --ids …/finished-989.ids`. Passing `--ids` keeps a re-judge run from regenerating the unfinished items on the live model.

**Judge check on the smoke run:**
- All 4 "yes" verdicts whose extracted answer differs from the gold string are real equivalences: rounding within margin, a LaTeX rewrite of the same expression, and the same ordinal with a label.
- 5 of 39 judgments first hit the 16,384-token cap while Super reasoned (`finish_reason=length`, no JSON). A re-ask fixed each one, and none became an infra error.

## Data facts (counts only)
- **Pinned revision:** `04eeb7e`, which was the dataset head on 2026-11-02. The parquet's sha256 is `2b342190…` (recorded in `manifest.json`).
- **Composition:** 1,000 items, 272 with an image, 500 reasoning and 500 knowledge, 796 `exactMatch` and 204 `multipleChoice`. This matches AA's description.
- **Overlap with `cais/hle`:** 603 Diamond ids also appear in `cais/hle`. Among those, 80 answers and 124 question texts were revised, so a `cais/hle` run cannot stand in for Diamond.
- **Images:** 186 PNG, 81 JPEG, 4 WebP and 1 GIF (single frame). All 272 decode with PIL, and 4 are wider or taller than 4,096 px. vLLM's processor resizes those.

## Notes
- **Context cap:** the served context caps long reasoners, while AA allows the creator's limit (1M for Super VL).
  - The smoke run used `aa-super` at 131,072 tokens, and 1 of 40 items hit that cap.
  - The full run used 1M. 17 of 1,000 items still reasoned to the cap without a final answer.
- **Long tail:** at 1M context a few generations run for hours. The median item took 17.5k completion tokens, but the last 1% were still running at 6.2 h (48 requests in flight). The run ended at 8.1 h, and 10 of those 11 never gave an answer.
  - Plan around the tail: report at the coordinator's cutoff with the unfinished items counted 0, and let them finish in the background.
  - `--timeout` is per attempt and timeouts are retried, so it does not cap an item's total time.
- **Access:** `cais/hle-diamond` is gated with automatic approval. The HF account behind `HF_TOKEN` must accept the terms on the dataset page, or the download returns 403.
  - `cais/hle` (2,500 questions, same schema) is the older HLE that AA dropped from the index in v5.0. `--dataset cais/hle` runs it.
  - On `cais/hle`, `aa_reference` is AA's text-only HLE score. AA ran that eval on 2,158 text-only questions, so pass `--text-only` and expect a mismatch on the image items.
- **Judge structured output:** the official judge uses the OpenAI SDK's `parse()` with a pydantic model. We send its strict JSON schema with confidence restricted to the prompt's integer 0–100 range. The finite enumeration prevents an unbounded confidence integer from consuming the entire output budget with trailing zeros. This is a grading-schema deviation from the upstream unbounded integer; question/answer messages, correctness criteria and model generation are unchanged. A grade-only resume retries failed judgments and saved judgments whose confidence is outside this range, preserving every other judgment and all generations. The actual schema is recorded in `config.json` alongside the grading execution SHA.
- **Endpoint registry:** the client re-reads `endpoints.json` before every attempt, so a restarted server is picked up mid-run. A registry older than 120 s counts as a retryable infra failure.
- **API keys:** they are read from the environment or from `~/.bashrc`'s `export` line. Condor jobs can't source `~/.bashrc` because of its interactive guard. Keys never enter submit files or job ads.

### Batch judging after generation

Use the same run ID for both stages. Generation records answers with `status=pending` and `score=null`, without resolving or contacting a judge. Its summary remains incomplete and invalid as a score, while `generation_complete` reports whether generation finished. Resume generation with the same command before grading any missing outputs.

```bash
python -m aa_evals.hle.run --stage generate --model serving:aa-model --run-id model-hle
python -m aa_evals.hle.run --stage grade --run-id model-hle --judge serving:aa-shared-judge
```

Grading uses the stored model settings, context, item selection and generation code SHA. The original model server may be offline; a missing generation remains an infra outcome and is never regenerated by the grading stage. The grading commit is recorded separately as `stage_git`. `--stage all` retains the original combined behavior, and `--generations-from` can still seed another run for a separate judge arm. The CPU submit wrapper forwards `--stage` after its `--` separator.

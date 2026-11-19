# GDP.pdf (AA Intelligence Index v5.0, weight 5%)

Issue #34. Surge AI's GDP.pdf as Artificial Analysis runs it: 100 tasks over long professional PDFs in 10 domains (4,592 pages, 1,275 rubric criteria). Each task is answered in a single turn with no tools, from the document's LiteParse text plus page images. An LLM judge grades every rubric criterion separately, and the headline metric is **All-pass**: the share of attempts in which every criterion passes. There are 5 repeats per task.

## Quick start

**M3 protocol (#175): one command per model**: 100 tasks x 5 repeats, a 128k context cap, the model's own request
settings, AA's text-alone path for text-only models, and the pinned judge (`aa_evals/common/protocol.py`):

```bash
python -m aa_evals.gdppdf.submit run --run-id <id> -- --model serving:<name>                          # any M3.1 model
python -m aa_evals.gdppdf.submit run --run-id <id> -- --model serving:<qwen> --reasoning-effort medium # Qwen3.8 Medium
```

- **Settings:** the registry's `request.overrides` (the lab's sampling, thinking on, the spec's default effort).
  Only Super keeps a preset (`super-vl`: NVIDIA's vision sampling, T 0.6). `--reasoning-effort X` sets the effort at
  the top level and in `chat_template_kwargs`; refused for models without effort levels.
- **Context:** `--context-window 131072` (default) is the planning window. Inputs are planned so that at least
  32,768 tokens stay free for the output (`--output-reserve`), and each request gets `max_tokens` = the window minus
  its planned prompt (an overestimate of the real one, so prompt + output never passes the cap; `n_over_window`
  counts any that do). `--context-window 0`: the endpoint's full context with a 65,536 reserve (the #66 protocol),
  for Qwen3.8 calibration runs only.
- **Text only for every model (#348, default `--vision off`):** every attempt gets the full LiteParse text alone, AA's
  path for models without image input, whatever the endpoint takes. `--vision auto` restores page images where the
  serving launch allows them (runs before #348; resuming such a run without it is refused). `config.json` records
  `inputs.vision` and `image_inputs.sent`; the summary's `protocol.image_inputs` is what was sent.
- **Text-only task set (#348; the user approved the 25 exclusions on 2026-11-11):** new runs draw from the 75 tasks
  of `--task-set text-only-v1` (default). `data.TEXT_ONLY_EXCLUDED` lists the 25 left out, whose answers sit in a
  figure, map, plan, photo or handwriting that the text doesn't carry. `--limit`/`--task-ids` select within the set,
  and an excluded id is refused with a hint. `--task-set all` keeps the 100: runs before #348 used it, resume and
  re-grade on it, and their `summary.json` adds `task_sets.text-only-v1` scored from the same rows. `config.json`
  records `selection.task_set` (absent = `all`); `summary.json` records `task_set`. Classification and signals:
  `/data/aai/aa_evals/audits/gdppdf_textonly_20261111/`.
- **Judge:** with no `--judge`, the pinned arm for GDP.pdf (`aa_evals/judges/pins.py`; `qwen38-27b-medium-pin` until
  study s3 on #41 reports).
- **summary.json** adds `judge_arm`, `n_over_window`, a `protocol` block (model, effort, thinking, AA variant, context
  cap, repeats, judge arm and settings key, code SHA, image inputs) and `n_flagged` (M3.4 trace judge).
- **Trace judge (M3.4, #177):** run it next to the benchmark, in tmux:
  `python -m aa_evals.tracejudge watch --run-dir <run dir> --judge serving:aa-judge-gemma4-31b`
  (`judge ... --fold` for a finished run). It writes `flags.jsonl`, and the runner folds `n_flagged`,
  `n_flagged_by_category` and `tracejudge` into summary.json through `aa_evals.tracejudge.fold_summary`.

Earlier usage:

```bash
aa_evals/gdppdf/setup.sh                          # venv, pinned dataset mirror, pinned Tesseract model (light)
python -m aa_evals.gdppdf.submit prepare          # LiteParse 2.5.0 + English OCR over 4,592 pages (condor, ~25 min)

# a run (condor CPU job; the model is served elsewhere)
python -m aa_evals.gdppdf.submit run --run-id super-smoke -- \
    --model serving:aa-super --preset super-vl --judge serving:aa-super --limit 10 --repeats 2
# the judge defaults to the shared open endpoint serving:aa-judge; any serving:/url: spec works.
# Closed API judges are not approved (#34); openai: specs are refused without --allow-external-api.
    ... --judge serving:<judge-endpoint> [--judge-extra-body '{"reasoning_effort": "medium"}']

python -m aa_evals.gdppdf.test_gdppdf              # offline tests

# re-grade stored generations with another judge (judge study #41): no new model calls
python -m aa_evals.gdppdf.run --run-id super-smoke-judgeX --generations-from super-smoke \
    --judge serving:<judge> [--judge-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}']
```

Use the venv python, `/data/aai/aa_evals/envs/gdppdf/bin/python`, with `PYTHONPATH=<repo>`.

`run.py` follows the shared contract: `--model serving:<name> | url:<base>@<served_name> | openai:<model>`, `--run-id`, `--limit N` and `--repeats K` (default 5). `--limit` takes the first N tasks in a fixed domain-stratified order; `--task-ids` names tasks explicitly. Stages (`--stage generate|grade|summarize|all`) resume: rerunning the same command redoes only missing or infra-failed work.

Completed answers awaiting this judge are `pending`, with `n_pending_judgments`, rather than infrastructure
errors. A pending summary has no final score and is invalid until judging finishes. Explicit failed judge
results remain infrastructure errors; deadline-mode ungraded answers still count zero. Grade/summarize on
an existing run reuse its stored selection, generation settings, context/image policy and generation SHA,
even after the generation server has stopped; the grading code SHA is recorded separately.

## AA's method, and what we do

| | AA (methodology page, 2026-11) | Ours |
|---|---|---|
| Tasks | 100 tasks, 10 domains, 4,592 pages, 1,275 criteria | the same 100 (see Data); since #348 new runs use the 75 of `text-only-v1` |
| Text | LiteParse 2.5.0 with English OCR; "the full extracted text of every page", never truncated or summarized | LiteParse **2.5.0** (PyPI wheel) with English OCR. Tesseract `tessdata_best/eng` is pinned at `e12c65a`, which is the model LiteParse downloads by default. Per-page `text` output. |
| Images | 150 DPI page images, sent first in page order, then the task and the text, all in one user message | the same, rendered by LiteParse's own pdfium renderer and flattened from RGBA onto white |
| Context/payload limits | DPI reduced to a 72 minimum; opaque pages may switch PNG→JPEG | DPI ladder 150→120→96→72; at each DPI, JPEG (q=90) is tried before stepping down. The payload cap defaults to 100 MB. |
| Image count caps | labelled 2-page composites, or 4 where the cap is tighter; beyond that, images cover only the leading pages; the prompt says so | the same; composites are 1×2 or 2×2 grids, each page under a grey "Page N" label |
| Nothing fits | the complete extracted text alone | the same; if even the text doesn't fit, the attempt is `model_error/input_too_long` (AA: "terminal input failures score as zero") |
| Turns | single turn, no tools | the same; no system prompt |
| Judge | GPT-5.6 Luna (medium), one call per criterion; sees task prompt, answer and criterion, never the PDF | **an open self-hosted model**, as the user decided on 2026-11-02 (no closed API judges). The default spec is the shared `serving:aa-judge`, whose model is chosen after a judge test; candidates are `serving:aa-judge-qwen38-27b` and `serving:aa-super`, with settings **pinned across judged benchmarks** (audit, #41 issuecomment-6597330339): Qwen judge = the registry's sampling (T 1.0, top_p 0.95, top_k 20) at reasoning_effort medium; Super judge = the registry's defaults (T 1.0, top_p 0.95, thinking on); both max_tokens 32,768, and a reply cut at the cap is re-asked. Grades are keyed by spec + a hash of these settings, so a settings change re-grades. One call per criterion, with the same inputs as AA. |
| Judge prompt | not published | Surge's harness template (surge-ai/gdp-pdf@7a72a51 `scorer.py`), with the task prompt added to the JSON object; strict JSON verdict `"0"`/`"1"` plus a rationale; up to 5 tries on unparseable output |
| Grading accepted | "only when every criterion has a verdict" | an attempt with any unjudged criterion is `infra_error` and is regraded on the next run |
| Metrics | All-pass over 500 attempts; Mean Pass (task-macro); domain cuts on Mean Pass only | the same, plus a task-clustered 95% CI |
| Errors | API failures retried up to 30×; errors and missing attempts score 0 | infra failures are retried with backoff (12 tries), then excluded and counted (a run with more than 2% is invalid). Model errors (empty answer, truncation before any answer, input too long) score 0. |
| Sampling | temperature 0.6 for reasoning models unless the lab recommends otherwise; max output = the creator's disclosed maximum | per preset (below) |

### Deviations from AA

| # | Deviation | Why | Expected effect |
|---|---|---|---|
| 1 | **Data is a community mirror.** We use `tolo6474/GDP.pdf@90feac14` instead of `surgeai/GDP.pdf`. | The official repo became private or deleted between 2026-07-15 and 2026-10-11 (Wayback). | None if the copy is faithful. Four independent mirrors are byte-identical, and the counts match AA exactly (see Data). |
| 2 | **Contestant prompt wording is ours.** | AA doesn't publish its prompt. We follow its description: images first in page order, then the task, then the full text, with notes on composites and coverage. | Small, but unknown. |
| 3 | **Judge prompt** is Surge's template with the task prompt added. | AA doesn't publish its judge prompt. | Unknown. All-pass is sensitive to judge strictness. |
| 4 | **Judge model** is an open self-hosted model, not GPT-5.6 Luna: the arm pinned in `judges/pins.py` (`qwen38-27b-medium-pin` until study s3 on #41 reports; final pick against DeepSeek in M3.5). | The user decided on 2026-11-02 against closed API judges. | Potentially large for all-pass, since one strict "0" zeroes an attempt. On Qwen3.8's r3 answers the Qwen and Super judges agree on 94-95% of criteria and move all-pass by up to 1.6 pp. |
| 5 | **Adaptation thresholds** come from our endpoint (`max_model_len`, `--limit-mm-per-prompt`, vLLM's decoded-media cap, a 100 MB payload cap). AA's come from each provider's own caps. | Our serving differs. | None when the endpoint has 1M context, ≥200 images per prompt and a 2 GiB decode cap. With vLLM's default decode cap, about 15 large documents drop to 120 or 96 DPI. `summary.json` counts these under `input_modes`. |
| 6 | **Serving** is self-hosted vLLM 0.32 with NVFP4 weights, not AA's API provider. | Self-hosted. | Small quantization and serving differences. |
| 7 | **Sampling and output cap.** The registry's lab sampling for every model except Super (`super-vl`: NVIDIA's vision recommendation, 0.6 / top_p 0.95). The output gets the rest of the window, with at least 32,768 tokens kept free when planning the inputs under the 128k cap (65,536 with the full context). | No lab discloses an output cap AA would use. The reserve trades image resolution against room to answer. | Truncations are reported (`error_kinds`, `truncated_with_answer`). In the #66 r3 runs, 3 of 500 medium and 28 of 500 xhigh completions were longer than 32k. |
| 11 | **The M3 128k context cap (#173)** bounds prompt + output; AA gives each model its full context (Qwen3.8: 262k). | The user's cost decision for regular evals. | **Large on long documents.** The LiteParse text alone of 3 of the 100 documents is more than 131k tokens, and that of 9 is more than the 98k left after the 32k reserve: those attempts are `model_error/input_too_long` (scored 0, AA's rule for terminal input failures). Of the rest, many more fall to 72 DPI or text alone. In r3 at 262k, the 10 tasks whose text is above ~97k tokens had 3 of Qwen3.8 medium's 52 all-pass attempts. |
| 12 | **Text-only models get the text alone.** DeepSeek is served `--language-model-only` (AA treats it as image-capable). Since #348 every model gets the text alone (`--vision off`). | Our serving spec; since #348 the models we fine-tune are text-only. | Plans, drawings and scanned tables lose their images. The `text-only-v1` set (#348, default) leaves out the 25 tasks that need them. |
| 8 | **Infra errors are excluded** rather than scored 0. | Shared contract (#18). | Unbiased when infra errors are rare; above 2% the run is invalid. |
| 9 | **LiteParse via the Python binding**, which may differ from AA's (unknown) entry point. Some OCR text carries UTF-8-as-Latin-1 mojibake, which we keep as is. | Same version, same OCR language. | Small. |
| 10 | **Qwen's image processor caps each image at 4,194,304 pixels** (the served recipe's `--mm-processor-kwargs {"max_pixels": 4194304}`, about 4k tokens). Pages larger than that at the planned DPI are downscaled by the server. The audit counts 82 such pages in 9 PDFs at 150 DPI: large-format sheets such as plans and drawings. | It's the serving recipe's setting. AA's Qwen provider may use another cap. | Fine detail on large sheets is lost. Letter pages at 150 DPI (2.1 MP) are unaffected. |

## Data

- **Location:** `/data/aai/aa_evals/data/gdppdf/`. It is not committed: the licence is CC-BY-4.0, the data carries canary `329d96c2-986f-486e-81db-4b6bd8539ead`, and it must never go into training data.
- **Approved by the user on 2026-11-02 (#34):** the mirror below, for evaluation only, never redistributed.
- **Where the official dataset went:** `surgeai/GDP.pdf` (named by AA and by Surge's harness) now returns 404 even with our HF token. Wayback has captures returning 200 up to 2026-07-15 and 401 from 2026-10-11.
- **The mirrors agree with each other:**
  - Five HF mirrors exist.
  - **sleepyheeler**, **tolo6474**, **Oitaaa** and **SigridYiiiiiii** were uploaded between June and October. They hold the same `data.parquet` (sha256 `2ba18b4f…`) and the same 100 PDFs, by LFS sha256. Only their README cards differ.
  - **cloudybean** changed the parquet and 2 PDFs and relicensed them MIT, so we don't use it.
- **The copy matches AA:**
  - The pinned copy has **100 tasks, 1,275 criteria and 4,592 pages**, AA's exact counts.
  - The PDF file names match a file URL Wayback archived from the official repo.
  - `data.load_tasks()` checks the parquet hash and the counts.
- **Harness:** [surge-ai/gdp-pdf](https://github.com/surge-ai/gdp-pdf)@`7a72a51` is cloned for reference under `data/gdppdf/src/harness`. It is an Inspect AI task that attaches the PDF as a native document input, which vLLM can't take, so we reuse only its judge prompt, schema and parsing.
- **One broken task:** `efb22c1d` (Insurance; 11 pages of hail-damage photos; 26 criteria) has an annotation note where its prompt should be: "The issue here can be resolved by editing either the prompt, context, or rubric. Full details are below in the context/rubric section." The task is unanswerable.
  - Its 26 criteria are part of AA's 1,275, so AA scores it as is, and so do we.
  - It caps All-pass at 99% for every model.
- **Domains** (tasks per domain): STEM/Research 15, Healthcare 11, and 10 each for Real Estate, Engineering, Manufacturing/Supply Chains and Construction. Insurance, HR and Finance/Investing have 9 each, and Legal 7. Pages per document: mean 46, median 26.5, max 200; 24 documents run past 64 pages.

## Input construction (`inputs.py`)

- **Images per request:** with an endpoint cap of C, documents of n > C pages become 2-up composites (if ⌈n/2⌉ ≤ C), else 4-up. If even 4-up exceeds C, images cover the first 4C pages.
- **Token budget:** a candidate fits if text tokens + image tokens + 8 per image (when there are several) + 512 + max_tokens ≤ max_model_len, and the decoded image bytes (W×H×4, summed) stay under the server's `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`.
  - Text tokens come from the endpoint's `/tokenize`.
  - Image tokens come from the model's own image rule:
    - `nemotron_vl` copies vLLM 0.32's `DynamicResolutionImageTiler`: a 16-px patch grid capped at 13,312 patches, with 2×2 pixel shuffle, plus 2 delimiter tokens. It is checked equal to vLLM's tiler on 12 sizes and to the live `/tokenize` on 4; a 150 DPI letter page costs 2,082 tokens.
    - `probe` is the default for any other vLLM endpoint. It measures one blank image per distinct size with `/tokenize` (chat messages count image tokens), so it is exact for any dynamic-resolution model.
  - The budget matters: **when the images overflow the context, vLLM's Nemotron processor shrinks every image silently instead of rejecting the request.** Without our own budget, the server would pick the resolution.
- **Rejections:** a context rejection from the server moves the task to the next rung. An "at most N images" rejection lowers the image cap, and a decoded-bytes rejection sets the byte cap the server reports. Each attempt's `plan` (mode, DPI, k, coverage, JPEG, estimated tokens, rungs tried) is stored in `generations.jsonl`.
- **Prefix caching:** the 5 repeats of a task share one message. Repeat 0 is sent first, and the others start once its first token streams back, so they reuse its prefix cache.

## Serving needs

- **Endpoint:** `serving:aa-super` (Nemotron 3.5 Super VL, NVFP4, one B200), restarted with `--max-model-len 1048576` and `--limit-mm-per-prompt '{"image":256,"video":0}'`, as approved on #34 (the change is in aa-serving 42546cd). It should also set `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES=2147483648`; otherwise documents over about 85 pages fall back to 120 or 96 DPI.
- **Prompt sizes:** measured with Super's tokenizer on the prepared text, plus the image rule.
  - LiteParse text averages **815 tokens per page**; dense pages run to about 2k, because the layout output pads columns with spaces. A 150 DPI page image adds about **2,100**.
  - Prompt per task at 150 DPI: **mean 134k, median 88k, max 570k** (`9847a092`, 200 pages).
  - **At 1M context, all 100 tasks fit at full 150 DPI** with a 65,536-token output budget.
  - At the old 131k context with 64 images, only 40 tasks fit at 150 DPI, 64 fit at 72 DPI, and **20 don't fit even as text alone**.
- **Prep:** one 16-CPU condor job, 22 min. Extracted text totals 16.05M characters. 29 pages came out empty, and 27 documents carry 1,224 mojibake sequences (UTF-8 read as Latin-1, e.g. `Â£`), which we keep as LiteParse emits them.

## Smoke: Nemotron 3.5 Super VL (2026-11-02, run `smoke-super-1`)

**Setup**
- **Tasks:** 11 × 2 repeats: one task per domain (`--limit 10` order) plus the 200-page `9847a092`.
- **Model:** `serving:aa-super` (1M context, 256 images), preset `super-vl`.
- **Judge:** `serving:aa-super` itself, the stand-in. The job was one 8-CPU condor job, 20 min wall on the shared server.

| | |
|---|---|
| All-pass | **0.0** (0/22 attempts; AA: **8.0**. With p = 0.08, 0 of 22 has probability 0.16, so this is consistent) |
| Mean Pass (task-macro) | 41.2 ± 15.7 |
| Near misses | 10/11, 16/17 and two 5/6 criteria passed |
| infra_error | **0** (the judge's 191 per-criterion calls all parsed on the first try) |
| model_error | 2 (`truncated_before_answer`: thinking looped to the 65,536-token cap with no answer) |
| Input modes | 16 attempts at 150 DPI; 6 at 96 DPI (the 168–200-page documents, because of vLLM's decoded-media cap, below) |
| Prompt / completion tokens | mean 163k / 14.8k per attempt |
| Judge tokens | 1.8k in / 380 out per criterion call (Super, medium effort) |

**Findings**
- **Token estimates hold.** The client's prompt-token estimates were within 0.1–0.3% of vLLM's counts. Two misses were found and fixed:
  - With more than one image, Nemotron's chat template labels each image `<image N>`, about 6 extra tokens per image.
  - vLLM caps the decoded image bytes per request: `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`, 715,827,880 by default, counted as W×H×4 per image.
- **The decoded-bytes cap downgrades large documents.** It allows only about 85 letter pages at 150 DPI, so about 15 of the 100 documents drop to 120 or 96 DPI.
  - The ladder handles this; the client now knows the cap and avoids the rejected round trips.
  - To send every document at AA's 150 DPI, the server needs `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES=2147483648`, which covers 255 pages.

## Outputs

**`reasoning_unclosed`** (row flag; `n_reasoning_unclosed` in the summary) marks a `model_error` where the turn ended with `finish_reason` stop and no answer, with every completion token reasoning except at most 4 closing ones: EOS before `</think>` (hle's definition), or `</think>`, optional whitespace and `<|im_end|>` straight after a reasoning cut mid-sentence (seen on Qwen3.8 at medium, r1 f41c14f1). A serving defect (MTP speculative decoding) is suspected and is being tested on aa-qwen-nomtp.

**Repeats and CI:** `summary.json` labels `k_repeats` and `ci95_method` (a normal approximation over task means, attempts clustered by task). With k=1 the CI covers task sampling only.

**Deadline cutoffs** (overnight rule, 2026-11-03): `--stage summarize --cutoff` counts every attempt with no generation as `cutoff`, scored 0 and reported as `n_cutoff`. Gate queue time never times a request out (the timeout is per read, 3 h), and a long or length-capped generation is a model outcome, never infra.

The runs root is `/data/aai/aa_evals/runs/gdppdf/` unless `AA_GDPPDF_RUNS` is set (submit.py passes it to the job). Runs made 11-02..11-13 are in `/data/aai/aa_evals/runs_fallback/gdppdf/`. Transcripts never contain page images.

`/data/aai/aa_evals/runs/gdppdf/<run_id>/` holds `config.json`, `generations.jsonl` (the transcripts, with reasoning), `grades.jsonl`, `items.jsonl` and `summary.json`, as in the shared contract. Everything is packed one file per kind, never one file per request, because of the `/data` file quota.

## Lessons (non-obvious)

- **The official HF dataset vanished.** We found it by comparing LFS hashes across mirrors and verified it against AA's counts.
- **LiteParse's Python wheel downloads `tessdata_best/eng` at run time** into `~/.tesseract-rs`. Pin it and pass `tessdata_path`, or condor jobs fetch it through the proxy.
- **LiteParse screenshots are RGBA PNGs.** Flatten them onto white yourself rather than rely on server-side conversion.
- **LiteParse text is token-heavy:** 815 tokens per page on average and about 2k on dense pages (as much as the page image), because its layout output pads columns with spaces. Budget context with it.
- **vLLM refuses requests whose decoded images exceed `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`** (715,827,880 by default, counted as W×H×4). That is about 85 letter pages at 150 DPI, which is far below what 1M context allows. Raise it on servers meant for long documents.
- **Nemotron's chat template numbers the images (`<image N>`) when there are several.** A one-image `/tokenize` probe therefore undercounts by about 6 tokens per image.
- **vLLM's Nemotron VL processor silently downsizes all images** to fit `max_model_len` minus the text. Budget images client-side so the DPI is decided, and logged, by us.

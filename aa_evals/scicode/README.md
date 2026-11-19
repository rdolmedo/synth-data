# SciCode-Verified (+ our fix pass): AAI-open stand-in for FrontierCode 2.0 (15%)

FrontierCode 2.0 is private. The M3.0 decision (#173, item 4) puts **SciCode-Verified** in its slot, plus our own fix pass. Build and fix pass: #184. Audit that motivates it: `/data/aai/aa_evals/audits/scicode_verified/REPORT.md`, which found Verified "usable with caveats".

This is the one benchmark where **fixing the benchmark is in scope**. Every change is listed in a versioned patch list (`PATCHES.md`), with evidence.

The original SciCode also runs in the same harness. It serves as a harness check against AA, which scored the original SciCode up to Index v4.3.

## Usage

```bash
PY=/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python      # httpx; the grading stack lives in the SIFs
# headline: Verified + patch list v3 (the default), without background, 5 repeats, 128k context cap (a CPU condor job: the grader
# needs contained Apptainer, which the tmux host cannot start)
$PY -m aa_evals.scicode.run --model serving:<name> --run-id <id> [--effort medium] --condor
# harness check vs AA: original SciCode, with background, AA's 300 s grading timeout
$PY -m aa_evals.scicode.run --model serving:<name> --run-id <id> --dataset original --repeats 3 --condor
# summary.json from items.jsonl; regrade a run's saved code against another dataset version (patch analysis)
$PY -m aa_evals.scicode.run --run-id <id> --summarize
$PY -m aa_evals.scicode.run --run-id <id> --regrade --dataset verified-aa-v3 --condor
# a finished run, upgraded to a newer patch version: only what the patches touch reruns
$PY -m aa_evals.scicode.run --run-id <new-id> --upgrade-from <id> --dataset verified-aa-v3 --condor
```

**Rerun scope follows the fix (#184 policy).**
- A grading or scoring fix re-scores the stored outputs (`--regrade`, `--summarize`).
- A patch to specific problems (a new `verified-aa-vN`) reruns only what it touches (`--upgrade-from`). In each chain, with the run's own earlier code, a step whose prompt and tests are unchanged keeps its generation and grade. A step whose tests or targets changed is regraded. From the first step whose prompt changed (a text patch), that step and everything after it in the chain is regenerated. Every items row records the dataset and code it is scored under, plus `carried` (origin run, code and dataset) for carried steps. `upgrade.json` and `summary.json` (`upgrade`) list the touched problems.
- A generation-wide change reruns everything.

**Model settings.**
- Request settings come from the serving registry's `request.overrides`, which hold the lab's sampling and thinking switches.
- `--effort X` sets the reasoning effort wherever the registry pins one: top level and `chat_template_kwargs`.
- M3.0 efforts: DeepSeek V4.1 Flash 1017 runs at `--effort max`, and Qwen3.8-27B at `--effort medium` and `--effort xhigh`. Every other model runs at the registry defaults, with reasoning on.
- A `url:` endpoint has no registry, so pass `--extra-body` explicitly.

**Text-only models.** SciCode is text-only, so no image path is needed.

**Other options.**
- `--limit N` / `--problems a,b`: smoke subsets.
- `--repeats K`: 5 by default.
- `--context-window`: 131072 by default, the M3.0 cap. Use the native context only for explicit Qwen3.8 calibration runs.
- Rerunning the same command resumes the run. Finished steps are kept and infra errors are retried. The config check refuses a resume on another commit or other settings, because finals never mix commits.

**Outputs** go to `/data/aai/aa_evals/runs/scicode/<run_id>/`:
- `config.json`
- `items.jsonl`: one row per step × repeat, with `score`, `status`, `cutoff`, per-env pass and a failure tail
- `chains/<pid>.r<k>.json`: per step, the extracted code, finish reason, usage and grade
- `transcripts/<pid>.r<k>.gen.jsonl.gz`: prompt, reasoning and response; `transcripts/<pid>.r<k>.jsonl`: the trace judge's per-step view (response + generation/grading events)
- `summary.json`: headline score with CI, `main_problem_accuracy`, `sub_step_accuracy`, per-env accuracy, `n_infra_errors`, `n_cutoff`, `n_flagged` (the trace judge's `flags.jsonl`), repeats, model, effort, context cap, code SHA and the AA reference row

## Datasets (`data.py`, data in `/data/aai/aa_evals/data/scicode/`, pinned by md5)

| Name | What | Problems / scored steps |
|---|---|---|
| `original` | HF `SciCode1/SciCode` test split + upstream `test_data.h5` (md5 96d5d815…) | 65 / 288 (AA's 288) |
| `verified-v2` | flyingwagner/scicode-verified 0b40bd4, release v2 (`test_data_cleaned.h5` md5 2b41a7df…). Problem 2 dropped. | 64 / 287 |
| `verified-aa-vN` | `verified-v2` + our patch list vN (`patches/build.py`) | 64 / 287 minus excluded steps |

Steps 13.6, 62.1 and 76.3 are never generated or scored, as in the official harness. Their upstream gold code is injected, in full, into the cumulative code of later steps.

## Method

The prompts are the official harness's templates, byte-identical to upstream `gencode.py` on all 582 step × background cases (checked).
- Without background: `background_comment_template.txt`. The model writes the background as a comment, then the code.
- With background: `multistep_template.txt`.
- Earlier steps appear with the model's own full extracted code.

One user message per step, no system prompt. Code extraction is the official `extract_python_script`: the first ```python block, with import lines stripped.

**Grading.** As in `test_generated_code.py`, one script per step: dependencies + the code of steps 1..k + the step's targets + all its test cases, in a fresh `python -I`.
- The script runs in two Apptainer SIFs, built by `sandbox/build_sif.sh`: Python 3.11, with **np126** (numpy 1.26.4, scipy 1.13.1) and **np22** (numpy 2.2.6, scipy 1.15.2).
- Sandbox: `--containall --cleanenv --network none`, `ulimit -v` 16 GB, 2 BLAS threads.
- A step passes if it passes in either env. This is Verified's OR rule; per-env scores are reported too.

**Scores.**
- **Main-problem accuracy** (Verified headline): a problem counts only if every scored step passes, averaged over repeats.
- **Sub-step accuracy**: AA's metric.
- CI95 comes from a cluster bootstrap over problems.
- Infra failures (endpoint failures after 30 retries, sandbox start failures) are never scored. A run is valid at ≤2% infra.
- A response cut at the context cap (`finish_reason=length`) scores whatever it produced (usually 0) and is counted in `n_cutoff`.

## AA's method vs ours

AA ([methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking), SciCode, a legacy eval since v5.0) evaluates the original SciCode with scientist-annotated background in the prompt, scores at the sub-problem level (pass@1), and grades with a 300 s timeout on isolated executors. Its published numbers are 3 repeats × 288 steps; for example, Qwen3.8 27B (Medium) is 0.390046 = 337/864.

| Item | AA (original SciCode) | Ours: stand-in (`verified-aa-vN`) | Ours: harness check (`original`) |
|---|---|---|---|
| Data | original, 288 steps | Verified v2 + our patches | original, 288 steps |
| Background | on | **off**: Verified with background saturates frontier models at 84–98% sub / 69–92% main (audit) | on |
| Headline | sub-step accuracy | **main-problem accuracy** (sub-step reported too) | sub-step accuracy |
| Prompt | official template | official template | official template |
| Grading env | unpublished | two envs, OR rule | two envs, OR rule (per-env reported) |
| Grading timeout | 300 s | 1800 s (SciCode, Verified) | 300 s |
| Skip-step gold | unpublished | full gold | full gold |
| Repeats | 3 | 5 (M3.0) | 3 |
| Context | the model's maximum | 128k cap (M3.0) | 128k cap |
| Sampling | lab's recommended, else T=0.6 | the registry's lab settings | the registry's lab settings |

**Verified scores are not comparable to AA's SciCode numbers.** Every model in the aggregate is run here.

## Patch list (our fix pass)

`PATCHES.md` lists every patch of each version: id, step, kind, what changed, why, evidence and validation. The payloads are eval data and stay out of git: new tests, reference implementations, wrong-physics mutants and valid alternatives, all in `/data/aai/aa_evals/data/scicode/patches/<version>/`. The build (`python -m aa_evals.scicode.patches.build --version vN`) records their sha256 in the dataset manifest.

The build refuses to write a version unless every tests patch passes three checks, in both grading envs:
- its reference passes;
- every listed mutant fails;
- every listed valid alternative passes.

Text patches (spec edits) are checked by writing an implementation from the patched text alone and running it against the tests (`patches/<version>/textcheck/`).

## Deviations from Verified's own protocol
- Problem 2 stays dropped, as in Verified.
- Grading is the same as Verified's: the two-env OR rule and a 1800 s timeout.
- The patch list: see `PATCHES.md`.

Final audit scope, limits and version-upgrade policy: [AUDIT.md](AUDIT.md).

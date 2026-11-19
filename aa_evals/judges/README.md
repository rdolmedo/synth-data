# Open-judge study (#41)

This directory picks the self-hosted judge that will be served as `serving:aa-judge` for Omniscience (#31), HLE-Diamond (#33) and GDP.pdf (#34). No closed API judge is used (user decision, 2026-11-02). The protocol is in [#41](internal issue 41).

Every judge arm re-grades the same stored generations through each benchmark's own grader code and prompt. There are no new model calls:
- Omniscience: `--answers-from`.
- HLE: `--generations-from`.
- GDP.pdf: `--generations-from` (merged in #43), run from a frozen copy of `--gdp-ref` (default `origin/aa-evals`).

## Arms (`arms.py`)

An arm is a judge server plus one request setting. Its `extra` dict is passed unchanged as `--judge-extra-body` to all three graders. Each grader merges it over the serving registry's `request.overrides`, which hold the lab's sampling.

| Arm | Server | Hardware | Role |
|---|---|---|---|
| `qwen38-27b-nothink`, `-medium`, `-xhigh` | `aa-judge-qwen38-27b` (Qwen3.8 27B FP8) | 1×H100 | candidates |
| `super-medium`, `super-full` | `aa-super` (Super VL NVFP4, shared) | 1×B200 | candidates |
| `glm53-flash-max` | the shared GLM-5.3-Flash server, later | ≥2×B200 | upper reference |
| `dsv41-flash-high` | the shared DeepSeek V4.1 Flash server, later | 4×B200 | upper reference |

| `qwen38-27b-medium-pin`, `super-full-pin` | the pinned production judges (after audit 1) | 1×H100 / 1×B200 | candidates |
| `gemma4-26b-a4b-pin` | `aa-judge-gemma4-26b` (Gemma 4 26B-A4B-it, bf16; spec `gemma4-26b-a4b-it`) | 1×H100 | candidate (s3) |
| `gemma4-31b-fp8-pin` | `aa-judge-gemma4-31b` (Gemma 4 31B-it, RedHat FP8-dynamic; spec `gemma4-31b-it-fp8`) | 1×H100 | candidate (s3) |

Every grader takes `--judge-arm <name>` (Omniscience, HLE and GDP.pdf): it sets the judge server and request settings from the arm and records `judge.arm` in `config.json`. The sweep passes arms by name.

The reference arms are too large for one GPU. They run later on the servers of the stronger-model runs, so their GPUs are shared with that work. Pass `--server <registry>` when you run them.

## Usage

```bash
# one condor CPU job per judge server; resumable, so adding arms later just runs them
python3 -m aa_evals.judges.sweep submit --study s1 --arms qwen38-27b-nothink,qwen38-27b-medium,qwen38-27b-xhigh \
    --omni-src super-v11-300 --hle-src super-diamond-full
python3 -m aa_evals.judges.sweep submit --study s1 --arms super-medium,super-full \
    --omni-src super-v11-300 --hle-src super-diamond-full
python3 -m aa_evals.judges.report --study s1      # writes report.json and report.md in the study dir
python3 -m aa_evals.judges.test_judges            # offline tests

# leniency vs self-preference: judges {qwen, super} x answer sources {qwen, super}; a cell is
# <bench>:<answer source>:<judge>=<run id or absolute run dir>
python3 -m aa_evals.judges.selfpref --name r1 \
    --cell omni:super:qwen=j41-s1-qwen38-27b-medium --cell omni:super:super=j41-s1-super-medium \
    --cell omni:qwen:qwen=qwen-medium-r1-60 --cell omni:qwen:super=<run dir>
```

The sweep runs its steps in order: probe, then Omniscience, then the Omniscience retest, then HLE, then GDP.pdf. Because only one step runs at a time, each step's wall time belongs to one benchmark on that server.

- `/data/aai/aa_evals/runs/judges/<study>/steps.jsonl` records each step: run id, return code, wall time, GPUs, and whether the server is shared.
- Each re-grade run lands in its benchmark's runs root as `j41-<study>-<arm>[-rt]`.
- The job runs from a frozen `git archive HEAD aa_evals` under `/data/aai/aa_evals/code/judges/<sha>/`, so commit first.

## Measures (`report.py`)

| Measure | How |
|---|---|
| Agreement with deterministic verdicts | `deterministic.py` gives a verdict only where no reasonable grader could disagree; everything else is undecided and excluded. |
| Agreement between judges | Pairwise Cohen's κ. Omniscience is 4-class (CORRECT / INCORRECT / PARTIAL_ANSWER / NOT_ATTEMPTED); HLE and GDP are binary. Each arm also gets κ with the leave-one-out majority of the *other servers'* primary arms. |
| Leniency | The arm's pass rate minus that majority's, on the same items. On one answer source this can't separate a lenient judge from self-preference. |
| Self-preference | `selfpref.py`: the 2×2 of judges {Qwen, Super} × answer sources {Qwen, Super}. Leniency is the judge gap averaged over both sources. Self-preference is the interaction, with a paired bootstrap CI over items (GDP.pdf: tasks). Each cell also gets its error against the deterministic verdicts. |
| AA reproduction | Super's score under each arm, against AA's 21.2 / 69.5 (Omniscience), 8.8 (HLE-Diamond) and 8.0 (GDP.pdf). |
| Cost | Judge tokens per judgment; step wall time × GPUs, extrapolated to one full run: 300, 1,000 and about 6.4k GDP criteria. Wall time on a shared server is marked approximate. |
| Reliability | Judge infra errors, and the Omniscience test-retest κ (the judges sample at temperature > 0). |

The deterministic checks:
- **HLE multiple choice:** the letter on the response's last `Answer:` line is compared with the gold letter.
- **HLE exact:** *sure-correct* means the normalized answer equals the gold. *Sure-wrong* means both are plain numbers more than 1% apart.
- **Omniscience:**
  - *sure-correct* means the normalized answer equals the gold;
  - *sure-abstain* means a short answer (200 characters or fewer) that opens by saying the model doesn't know and offers no hedged guess, which the grader prompt calls NOT_ATTEMPTED. A hedge followed by an answer ("I'm not sure, but I believe it is **X**") is an attempt, graded on X, so it is left to the judge;
  - *sure-wrong* means both are plain numbers more than 5% apart.

The report writes only ids and aggregates. HLE questions may not be redistributed.

## Study s3: four judges on Qwen3.8's final generations (`s3.py`)

Judges: `qwen38-27b-medium-pin`, `super-full-pin` (a voter), `gemma4-26b-a4b-pin`, `gemma4-31b-fp8-pin`. The Qwen and Super verdicts on Qwen's r3 runs already exist (the runs and their `-jsuper` siblings); the Gemma sweeps write `j41-s3q{m,x}-<arm>` (Qwen medium / xhigh answers) and `j41-s3s-<arm>` (Super's s1 answers).

```bash
python3 -m aa_evals.judges.s3_probe --arm gemma4-31b-fp8-pin   # template, reasoning parser, HLE + GDP schemas; before any re-grade
python3 -m aa_evals.judges.s3 report                            # /data/aai/aa_evals/runs/judges/s3/report.{json,md}
```

All measures are on non-deterministic items only: pairwise disagreements and κ (Omniscience also 4-class), each judge's agreement with the majority of the other three, split patterns, leniency against that majority (paired bootstrap; GDP.pdf resamples tasks), gaps between judges with CIs, deterministic false accepts/rejects, Qwen's scores under each judge, reliability and cost. Hand adjudications of contested items (ids and labels only) are in `<study dir>/adjudication.json`.

**HLE's strict schema and Gemma.** HLE's official judge schema requires a trailing `strict: true` field that the prompt never mentions (OpenAI shows the schema to its model; vLLM does not). Gemma closes `confidence` and then, with free whitespace allowed, pads newlines until max_tokens on every probe. The Gemma servers therefore run xgrammar with compact JSON (`disable_any_whitespace`), which removes the whitespace runaway. About a quarter of HLE judgments still run away inside the unbounded `confidence` integer (`90821552…`); the grader's format re-asks recover them, at the cost of the wasted decode. Qwen showed the same failure without thinking in s1. Compact JSON also makes Gemma occasionally start a string with a stray `: ` (its natural `": "` token is not allowed); no field that is scored is affected.

M3.5 uses `--judge-arm dsv41-flash-max-pin` for the DeepSeek gold/reference pass. It targets the one shared `serving:aa-ds` endpoint with Max thinking, lab sampling and32k judge output, and leaves each benchmark's current default judge unchanged. Batch these requests after the stage's generation finishes. This explicit arm also works for GDPval, whose CLI requires a registered judge arm; the older `dsv41-flash-high` arm remains a separate historical study setting.

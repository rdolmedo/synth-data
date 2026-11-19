# Task synthesis

Generating Terminal-Bench-style tasks (Harbor format) as SFT environments, run locally on HTCondor (pilot: Nemotron 3.5 Super traces through Pi, Qwen3.5-4B student). This is separate from the `data-synthesis` branch, which turns existing tasks into training trajectories.

| File | What it is |
| --- | --- |
| [BLUEPRINT.md](BLUEPRINT.md) | **Start here.** Goal, decisions, generation tree, checks, first experiments, open questions. |
| [NOTES.md](NOTES.md) | Longer working notes (2026-10-17): sampler state, sourcing, detailed tree. |
| [TB_ASSET_CLASSES.md](TB_ASSET_CLASSES.md) | Asset classes per task for TB 1.0–4.0 and tbench-lite, plus suggested roles for TB 2.0. |
| [asset_sampler/](asset_sampler/) | Spec sampler fitted to TB 2.1 + 3.0 (`python3 sample_assets.py --n 100 --out specs.jsonl`; `--check` for fit diagnostics), labels, and the smoothing CV. |
| [decon/](decon/) | Reference copies of the seed decontamination scripts. |
| [PIPELINE.md](PIPELINE.md) | Task pipeline v0 (generate, build, check, judge) and the audit protocol with its quality bar. "M1 production" runs M1's data generation (#118). |
| [production/m1.toml](production/m1.toml) | **M1's production configuration** (#118): every decided value; `python -m task_synthesis.pipeline.production`. |
| [AUDIT_LOG.md](AUDIT_LOG.md) | Results and changes per audit round. |

**Comparing with student 1 (or student 2)? Generate the new rounds with `--vocab v1 --minutes-mix student1`.** Two defaults have changed since student 1: the minutes mix is easy-heavy since #26 (2026-11-02; tbmix 38.4 to 40.4), and the spec vocabulary is v2h0 since #25 (2026-11-04: v1's TB-fitted actions, domains and classes plus v2's sub-domains, merged defect details and other-code languages; untrained as a whole). A study on the defaults against student 1 mixes those shifts into its own effect. Compare against student 1's fp8kv eval `tbmixp180-tasksynth-4b-ds-330m-3ep-fp8kv` (38.4; its bf16 eval, 39.2, ran on different serving). Other baselines that exist: easy-heavy with v1 (`--vocab v1`; `tbmixp180-tasksynth-4b-m26-easy-3ep`, 40.4) and v2 with the student-1 mix (`--vocab v2 --minutes-mix student1`; `tbmixp180-tasksynth-4b-v2-330m-3ep`, 42.2 at k=10, and its training-seed replicate `-seed1`, 39.6: level with student 1 over two seeds per arm, 1.3 pp below it on full TB2.1). Nothing has been trained on both new defaults together (v2h0 with easy-heavy) yet.

**Trace generation of new campaigns (since 2026-11-14) follows the eval protocol**: the Pi finish notice (v4 sticky, as aa-evals #395) at a 131,072 context, and sandbox limits closer to the eval but still a bit stricter (memory max(1.5 x `memory_mb`, 24 GiB) sparing `pi`, an unlimited stack as in eval, eval's disk cap). Campaigns frozen before keep their recorded protocol (no notice, 132,000). Per-benchmark context and notice overrides, and the details: [synth/README.md](../synth/README.md), "Protocol and sandbox limits of new campaigns". Comparisons with students trained on earlier traces mix in this change.

Seeds live in [`../seeds/`](../seeds/).

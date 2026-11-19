# synth-data

Code that generates the synthetic training data behind our AAI-open results: task synthesis for terminal tasks,
per-benchmark problem generators, teacher trace generation with the Pi harness, and decontamination against the
evaluation benchmarks. The data itself is published on Hugging Face; the evaluation harness is in the companion
repository `aai-open`.

## Data

| Dataset | What |
|---|---|
| [`ricdomolm/aai-open-sft-v0`](https://huggingface.co/datasets/ricdomolm/aai-open-sft-v0) | the decontaminated SFT mixture (`m5l2-full`), one split per target benchmark, training records in message form; built by `m5/ladder_sft/` (frozen renderer pinned by tag `aai-open-sft-v0-build`) |

Status: dry-run export, under review. Generated from private trunk commit `6ac53bcf97aa` (see `CORE_VERSION`).

## What is here

| Path | What |
|---|---|
| `task_synthesis/` | terminal-task synthesis pipeline (asset sampling, task generation, verification, decontamination checks) |
| `synth/` | synthesis controller, workers, package cache and the serving launcher modules; `synth/teachers/` holds the teacher serving specs |
| `m5/` | per-benchmark generators (AutomationBench, GDPval, GDP.pdf, Omniscience, CritPt, HLE, SciCode, TB-Science-lite): taxonomies, problem generation, trace generation |
| `decontam/` | decontamination of generated tasks against benchmark items (retrieval + judge) |
| `seeds/`, `lists/` | Terminal-Bench seed task pointers and exclusion lists |
| `m5/ladder_sft/`, `sft_qwen35/data.py` | the pipeline that formatted and assembled the released SFT mixture (records -> render -> assemble) and its chat-template renderer |
| `aa_evals/`, `harbor_ns/`, `tb_agents/`, `inference/`, `tbshared/`, `tbsched/` | the shared core and the eval runners the generators use to produce traces (identical bytes to `aai-open` at the same `CORE_VERSION`) |

## Running

Generators talk to a teacher through an OpenAI-compatible endpoint (`url:<base_url>@<served_name>`, or a served
model registry). The campaign runners submit to HTCondor with Apptainer images, as on the cluster where the data was
produced; that path is kept for reproducibility and documented as internal. Internal default paths were rewritten
at export to neutral ones (`/data/aai`, `/opt/aai`). A pluggable local executor and a quickstart are planned.

## License

MIT (`LICENSE`), except third-party components listed in `NOTICE`. Generated data carries its own license and the
teacher models' terms; see each dataset card.

# M5 GDPval: synthetic SFT data (#358)

Problems only: each problem is a GDPval-style professional request (an occupation's work activity, the requester's
context, named deliverables) plus any input files it references. Super solves each one once under the eval's own Pi
harness (#348); the traces are the SFT data. The GDPval test set (`openai/gdpval`) is read only for class-level
statistics, Opus labels used to fit the sampler, the spirit check and decontamination; no test item is in the
generator's context.

## Pipeline

| Step | Module | Runs as |
|---|---|---|
| Opus labels of the 220 test tasks (task type, inputs, input roles, provenance, deliverables, web reliance) | `label_test.py`, `vocab.py`, `opus.py` | local |
| Leaves: 44 occupations × work activities (Opus, blind to test items) | `brainstorm.py` | local |
| Spec sampler: per-sector fits shrunk toward global (k by leave-one-occupation-out CV) | `sampler.py` | local |
| Real-document lane: pool from m5-gdppdf's shared cache (reference types, 2–40 pages) | `docpool.py`, `docs.py` | CPU job |
| Generator: spec → DeepSeek V4.1 Flash (effort high) → prompt + input-building script | `generate.py`, `prompts.py`, `llm.py` | CPU job |
| Problem checks: the script runs in the GDPval SIF offline; every listed file opens and has text; ≤ 2 repairs | `stage.py`, `sif_validate.py` | CPU job |
| Producer: generate → check → batches of 250 (Harbor task bundles via `pi/tasks.py pi_instruction`) | `producer.py` | CPU job |
| Traces: `aa_evals.common.pi_eval` with the GDPval port's `base_args` | `traces.py` | CPU job (Pi workers) |
| Driver: launches READY batches on the Super engine with most spare sessions, follows `alloc.json`, replaces held workers, periodic exports, stop rule | `drive.py` | process on a CPU node |
| Export: synth/ layout shards (`sft_qwen35/prepare.py --source pi_native --synth-roots`), manifest, verify | `export.py` | CPU job |
| 330M budget copy (seeded task order, whole traces, line flags rewritten) | `export.py budget` | CPU job |
| Spirit check, repeat meter, 8-gram overlap | `spirit.py`, `textsim.py` | local |
| Decontamination: retrieval (8-gram, TF-IDF, bge-small, hashes, canary) + DeepSeek reuse judge × 2 | `decontam.py`, `rubric_reuse_gdpval.md` | CPU jobs |
| Trace metrics, GPU-h by generated-token share | `report.py` | local |

`jobs.py submit --name N [--cpus --mem-gb --hours] -- COMMAND` submits any step as a CPU job (bid 36, BAD_NODES +
`m5-coord/excluded-nodes.txt` excluded) from a frozen detached worktree of HEAD under `/data/aai/m5/gdpval/code/`.

## Commands (production)

```bash
G=/data/aai/m5/gdpval
# two producers over spec indices 1000-8399 (DeepSeek cap shared)
python3 -m m5.gdpval.jobs submit --name prod-p0 --cpus 8 --mem-gb 40 -- python -m m5.gdpval.producer --root $G/production/p0 --start 1000 --n 3700 --batch 250 --check-workers 7 --docs $G/production/docpool-v2.jsonl --part 0 --nparts 2 --producers 2
python3 -m m5.gdpval.jobs submit --name prod-p1 --cpus 8 --mem-gb 40 -- python -m m5.gdpval.producer --root $G/production/p1 --start 4700 --n 3700 --batch 250 --check-workers 7 --docs $G/production/docpool-v2.jsonl --part 1 --nparts 2 --producers 2
# driver (one process): batches -> pi_eval runs gdpval_m5/prod-*, export every 90 min, stop at launched x 62k >= 330M
python3 -m m5.gdpval.drive --roots $G/production/p0 $G/production/p1 --phase prod --export $G/production/export --target-tokens 330e6   # worker CPUs/RAM: gdpval_pi in aa_evals/pipeline.json (#451)
# decontamination over every generated problem (pilot + production)
python3 -m m5.gdpval.jobs submit --name decon-retrieve --cpus 8 --mem-gb 32 -- decontam-python -m m5.gdpval.decontam retrieve --gen <gen.jsonl ...> --out $G/production/decon
python3 -m m5.gdpval.jobs submit --name decon-judge -- python -m m5.gdpval.decontam judge --out $G/production/decon --k 30 --window 10 --samples 2
# final export, manifests with decontam exclusions, verify, 330M copy
python -m m5.gdpval.export pack --runs $G/runs/gdpval_m5/prod-* --out $G/production/export --delete-archives --final
python -m m5.gdpval.export manifest --out $G/production/export --exclude $G/production/decon/excluded.json
python -m m5.gdpval.export verify --out $G/production/export
python -m m5.gdpval.export budget --roots $G/pilot/export $G/production/export --out $G/production/export-330m --budget 330e6 --seed 358
```

## Data

- Full export: `/data/aai/m5/gdpval/pilot/export` and `/data/aai/m5/gdpval/production/export` (two roots;
  pass both to `--synth-roots`).
- 330M copy: `/data/aai/m5/gdpval/production/export-330m` (pilot + production traces; deliverables stay in the
  source roots, which its index names).

`sft_qwen35/prepare.py` takes a shard line whose own `record.usable` is true even when the manifest excludes it. In
these exports every line's `record.usable` equals the manifest's (`export verify` checks it), and the 330M copy
rewrites the lines' flags (`usable=false`, `excluded="budget"` beyond the budget). Train from these roots only, never
from `runs/` directly.

## Protocol notes

- Request settings, window (131,072), output per request (16,384), 2 h environment clock, offline agent, text-only
  input, single uid, staged task inputs, 16 GB sandbox cap: `port.base_args` plus the Super registry's `request`
  block, as in the eval. One attempt per problem; seeds: namespace `gdpval-m5-v1`.
- Tokens per trace: the final request's prompt + output as Super served it (the whole linear sequence; no
  compaction). Usable = resolved, not infra, at least one file in `outputs/`, no canary, not decontam-excluded.

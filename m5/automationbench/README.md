# M5.8 AutomationBench: synthetic SFT data (#359)

Problems only: each task is a colleague's request plus the starting state of the simulated SaaS apps, in upstream
AutomationBench's task-file shape (no assertions). Super solves each one once under the eval's own Pi harness
(#348); the traces are the SFT data.

## Pipeline

| Step | Module | Runs as |
|---|---|---|
| Test-set statistics (class level) | `test_stats.py` | local, seconds |
| Labels of the 600 scored tasks (workflow, traps, targets, rule sources) | `label_test.py` (Opus via `claude -p`) | local |
| Leaves (scenarios per domain × workflow), company and name pools | `brainstorm.py` (Opus, blind to test items) | local |
| Leaf check: block leaves resembling one scored task | `leafcheck.py` (TF-IDF + bge-small) | local / CPU job |
| Taxonomy fit and spec sampler | `taxonomy.py` → `data/fit.json`, `specs.jsonl` | local |
| Generator: spec → DeepSeek → `build_task()` → upstream task + `_noise.py` → checks, ≤ 2 repairs | `gen.py`, `prompt.py`, `material.py`, `check.py`, `refs.py`, `client.py` | CPU job |
| Bundles (the eval's `pi/tasks.py build_task`) and Pi runs on Super | `bundle.py`, `traces.py`, `pi_launch.py`, `feed.py` | CPU job (feeder) |
| Packing traces (M1 synth-archive lines: sessions + first request) | `collect.py` | CPU job |
| Spirit check / pilot metrics | `spirit.py`, `pilot_report.py`, `snap.py` | local |
| Decontamination: retrieval (8-gram, TF-IDF, bge, copied strings) + DeepSeek judge × 2 | `decon.py`, `rubric_reuse_ab.md` | CPU jobs |
| Export (manifest over the shards; `sft_qwen35/prepare.py --source pi_native --synth-roots`) | `export.py` | local / CPU job |
| Export shards whose lines carry the manifest's flags, optionally cut to a token budget; checksums | `export_lines.py` | CPU job |
| GPU-hours by generation-token share | `gpu_hours.py` | CPU job |

`jobs.py NAME [--cpus --mem-gb --venv] -- MODULE ARGS` submits any step as a CPU job (bid 36, BAD_NODES +
`m5-coord/excluded-nodes.txt` excluded) from a frozen detached worktree of HEAD under
`/data/aai/m5/automationbench/code/<sha>`.

## Commands (production)

```bash
P=/data/aai/m5/automationbench/production
python -m m5.automationbench.taxonomy sample --fit fit.json --leaves LEAVES --pools POOLS --n 9500 --seed 360 --start 1000 --out $P/specs.jsonl
python3 m5/automationbench/jobs.py gen-prod --cpus 3 --mem-gb 12 -- m5.automationbench.gen --specs $P/specs.jsonl --out $P/gen --workers 3
python3 m5/automationbench/jobs.py feed-prod --venv harbor -- m5.automationbench.feed --accepted $P/gen/accepted.jsonl --state $P/feed --prefix prod --batch 200 --slots 64 --target-tokens 340e6
python3 m5/automationbench/jobs.py collect-prod --venv harbor -- m5.automationbench.collect --runs '/data/aai/m5/automationbench/pi_runs/m5_automationbench/prod-*' --out $P/traces --follow
# after the stop: decontamination, then the export
python3 m5/automationbench/jobs.py decon-prep -- m5.automationbench.decon prepare --accepted $P/gen/accepted.jsonl --work $P/decon
python3 m5/automationbench/jobs.py decon-retrieve --cpus 4 --mem-gb 24 --venv m5-hle-decon -- m5.automationbench.decon retrieve --work $P/decon
python3 m5/automationbench/jobs.py decon-judge -- m5.automationbench.decon judge --work $P/decon
python3 m5/automationbench/jobs.py export-prod --mem-gb 16 --venv harbor -- m5.automationbench.export --traces $P/traces --exclude-tasks $P/decon/excluded_tasks.txt --drop-no-submit
E=/data/aai/m5/automationbench/export
python3 m5/automationbench/jobs.py export-lines -- m5.automationbench.export_lines $P/traces $E/nemotron35-super
python3 m5/automationbench/jobs.py export-330m -- m5.automationbench.export_lines $P/traces $E/nemotron35-super-330m --budget 330e6 --seed 359
```

`sft_qwen35/prepare.py` takes a shard line whose own `record.usable` is true even when the manifest excludes it, so
train only from `export/*` (lines carry the manifest's flags), never from `production/traces` (collect.py marks every
attempt with a session usable).

## Protocol notes

- Request settings, window (131,072), output per request (16,384), 2 h environment clock + 2 h wall cap, offline
  agent, text-only input: `port.base_args` plus the Super registry's `request` block, as in the eval. Seeds:
  namespace `m5-automationbench-v1`.
- Token counts: each attempt's largest request context as the Super server counted it (prompt + completion of the
  last request of a linear session = the full sequence under Super's chat template).
- Stop rule: 340M tokens over traces that finished by themselves; export drops context/time-limit endings
  (no-submit) and decontamination exclusions, keeping every attempt listed and flagged in the manifest.
- Exports: `export/nemotron35-super` (all kept traces) and `export/nemotron35-super-330m` (330M-token budget,
  whole traces in a seeded random order of tasks, for budget parity across benchmarks).
- Data: `/data/aai/m5/automationbench/` (`fit/`, `pilot/`, `production/`, `pi_runs/`, `export/`). Test labels
  and decontamination work files stay there (derived from eval data); `data/` here holds only class-level counts,
  our brainstormed leaves and pools.

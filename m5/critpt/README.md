# M5 CritPt: synthetic SFT data (#356)

Problems only: each problem is a CritPt-style research-level physics challenge (a "Problem setup" plus a "Main
problem" asking for a checkable final answer in `/work/answer.py`), grounded in one arXiv physics paper. Super solves
each problem once under the eval's own Pi harness (#348, `aa_evals.critpt.pi.tasks.build_task`); the traces are the
SFT data. Generated problems have no reference answers (`reward` is null); a trace is kept when it submitted.

The CritPt test set is read only for class-level statistics (field, answer type, grounding counts fitted into the
taxonomy), the spirit check and seed blocking. No test item is in the generator's context; the only CritPt text the
generator sees is the public example challenge, as a format anchor.

## Pipeline

| Step | Module | Runs as |
|---|---|---|
| Taxonomy: 11 fields × 287 subtopics × techniques (Opus brainstorm, blind to test items), weights fitted to the 70 | `taxonomy/brainstorm.py`, `taxonomy/build.py` | local |
| Seeds: physics papers of the shared arXiv index; TF-IDF candidates per subtopic; papers test items name or sit near (25 per item) blocked | `arxiv_index.py`, `seeds_build.py` | CPU job |
| Generator: leaf draw (field by fitted weight, answer type corrected by measured accept rate) + seed paper → DeepSeek (effort high) → problem; static checks + DeepSeek well-posedness check | `gen.py`, `prompts.py`, `templates.py`, `checks.py`, `llm.py` | CPU job |
| Bundles: Harbor tasks exactly as the eval builds them (prompt, CritPt SIF, 4 CPU / 16 GB, offline) | `traces.py bundle` | local / streamer |
| Traces: `aa_evals.common.pi_eval` with the port's shared arguments (Pi 1.2.0, 131,072 window, 16,384 output per request, 2 h clock, text-only), 1 attempt per problem | `traces.py launch / resubmit` | CPU job (Pi workers) |
| Streamer: batches new problems, spreads runs over the Super engines `alloc.json` lists (water-fill), splits each engine's cap across runs, adds workers to fill it, moves queued work when engines join or leave, stops at the token target (kills the tail) | `stream.py` | CPU job |
| Export: synth/ layout shards for `sft_qwen35/prepare.py --source pi_native --synth-roots`; seeded random-by-task token-budget cut | `export.py` | CPU job |
| Full export + cut + gzip/SHA256SUMS + prepare dry run | `jobs/final_export.sh` | CPU job |
| Spirit check vs the 70 (mix, length, spread) and 8-gram overlap | `spirit.py` | local |

Writes on `/data` go through `safeio.py` (temp file, read-back and parse, rename; retried on EDQUOT/EIO).
`jobs/submit.sh NAME CPUS MEM_GB DISK_GB -- ARGS` runs `python ARGS` as a CPU job from a `git archive` of this
checkout's HEAD (bid 36, BAD_NODES + `m5-coord/excluded-nodes.txt` excluded).

## Commands (production)

```bash
C=/data/aai/m5/critpt
# generation (target and abort via $C/gen/prod1/control.json; pipelines follow alloc.json ds.max_inflight.critpt)
bash m5/critpt/jobs/submit.sh gen-prod1b 1 10 5 -- -m m5.critpt.gen --taxonomy m5/critpt/taxonomy/taxonomy.json \
  --seeds $C/seeds --out $C/gen/prod1 --n 12400 --exclude-used $C/gen/proto1,$C/gen/proto2,$C/gen/pilot
# streamer: stop when kept tokens reach 331M (pilot counted at its max_context tokens), kill in-flight traces
bash m5/critpt/jobs/submit.sh stream 1 4 5 -- -m m5.critpt.stream --gen $C/gen/prod1 \
  --base-tokens 7933488 --stop-kept-tokens 331e6 --kill-tail
# full export + seeded 330M cut + prepare dry run
bash m5/critpt/jobs/final_export.sh $C/export/final 330e6 356 $C/runs/m5_critpt/m5c-pilot1-e* $C/runs/m5_critpt/m5c-prod-*
```

## Data

- Problems: `/data/aai/m5/critpt/gen/{pilot,prod1}/problems.jsonl` (12,418), with `calls.jsonl` / `rejects.jsonl`.
- Traces: `/data/aai/m5/critpt/runs/m5_critpt/` (pi_eval runs `m5c-pilot1-*`, `m5c-prod-*`).
- Export: `/data/aai/m5/critpt/export/final/full` (10,107 traces, 332.8M tokens) and `.../cut` (10,025 traces,
  330.0M tokens, seed 356); `SHA256SUMS` in each; `prep-cut/` holds the prepare dry run's stats.

Decontamination (retrieval plus a same-core-idea judge against the 70) is pending: it runs only if the dataset shows
gains. The seeds already exclude the papers test items name and each item's 25 nearest papers.

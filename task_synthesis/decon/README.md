# Seed decontamination scripts (reference)

These are the scripts that built [`../../seeds/usable_seed_tasks.csv`](../../seeds/usable_seed_tasks.csv); the method is in [`../../seeds/METHODOLOGY.md`](../../seeds/METHODOLOGY.md). They were run once, on 2026-10-14, from a local working folder, and are copied here as is.

They are **not runnable from this repo yet**:

- they read and write `<parent>/data/*.json` and `<parent>/decon_bundles/`, which are intermediate files not included here (task texts, embeddings, judge ratings);
- `extract.py` expects local clones of the Terminal-Bench repos;
- `decon_score.py`, `decon_propagate.py` and `openpr_score.py` need `numpy` and `fastembed`;
- the LLM adjudication steps (Sonnet, then Opus) were run by Claude Code subagents over the bundles from `decon_bundle.py`, not by a script.

Order: `fetch_prs.py` → `decon_extract.py` → `decon_score.py` → `decon_flag.py` → `decon_bundle.py` → (LLM review) → `decon_propagate.py` → `openpr_docs.py` → `openpr_score.py` → (LLM review) → `decon_finalize.py`.

For task synthesis, the same scoring (n-gram containment on instruction, tests, solution and environment files, plus instruction embeddings) should be adapted to check **generated tasks** against TB2 and the OOD eval. `SEED_SURVEY_README.md` is the original folder's README.

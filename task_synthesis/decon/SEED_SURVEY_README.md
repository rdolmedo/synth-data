# Terminal-Bench seed survey + decontamination (2026-10-14)

Main output: `data/seed_candidates_decon.csv`, with one row per non-test task from the harbor-framework TB repos
(TB1, TB2, TB2.1, TB-continuous). Filter on `usable_seed == True` (554 rows).

Columns: `tier` (A CI-validated / A maintainer-merged / B unvalidated / C oracle failed / Q / X), CI signals,
`decon_status` (clean, clean (flagged, cleared), contaminated, quarantine, junk), `decon_reason`, `decon_ref`.

## What "decontaminated" means here
Reference set (must not leak): TB1-core 0.1.0/0.1.1, TB2.0, TB2.1, TB 3.0.0/4.0.0/HEAD, pre-acceptance PR
versions of all of those, OT-TB-Lite (DCAgent/dev_set_v2 @377118f), TB-Science main.
Future-risk set (quarantine): open PRs in harbor-framework/terminal-bench and terminal-bench-science.

Pipeline (scripts/):
1. `decon_extract.py`: full text of every task version (instruction, tests, solution, Dockerfile, env files, file hashes, authors).
2. `decon_score.py` (mumwelt venv): per pair, n-gram containment for instr/tests/solution/docker/env, shared
   non-boilerplate files (git blob sha), bge-small instruction cosine, same-author (person-level, not vendor-level).
   Calibrated on 25 known rejected-then-accepted pairs: 25/25 caught, 0.23% of random pairs flagged.
   Instructions alone are NOT enough: several accepted tasks had fully rewritten instructions but 90%+ shared tests/solutions.
3. `decon_flag.py` → `decon_bundle.py`: Sonnet adjudicates flagged pairs side by side; Opus reviews every
   non-"unrelated", low-confidence or high-overlap pair (EXCLUDE/KEEP, err toward exclude).
4. Embedding band 0.75–0.80 (top-2 per seed): Sonnet instruction-only screen, Opus for UNSURE.
5. `decon_propagate.py`: exclusions propagate to near-copy sibling candidates.
6. `openpr_docs.py` + `openpr_score.py`: seeds vs all open PRs (resubmissions), same Sonnet→Opus review.
7. `decon_finalize.py`: writes the CSV.

## Re-run before each use / each TB release
Re-fetch PRs (`fetch_prs.py`), add new release tags to the reference set, re-run steps 1–7.
Closed PRs get resubmitted (we found 10 seeds whose work reappears in open PRs).

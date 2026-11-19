# decontam: CalibForge vs Terminal-Bench 2.1

Finds CalibForge training tasks that leak Terminal-Bench 2.1 test tasks (issue #14) and writes exclusion lists that
training drops (`sft_qwen35/prepare.py --exclude-tasks`, on branch `sft`, defaults to the current list).

## Lists

| list | criterion | tasks |
|---|---|---|
| `calibforge/decontamination-variants.json` (current, 2026-11-01) | a **variant of one specific TB2.1 or TB-Lite task**: carries its crux, workaround, instance or several of its distinctive choices in any wording, or copies its distinctive form (instruction template, starter files); genre never counts | 405 / 5,431 |
| `calibforge/decontamination-rewrites.json` (2026-10-27) | the same problem as a TB2.1 task (duplicate or rewrite) | 281 / 5,431 |

Both lists are in `/data/aai/tbench/synth/shared/datasets/calibforge/`, next to their `.pairs.jsonl` (every judged
pair). `decontamination-variants.README.md` there gives the method, sources and audit results (pair precision ~0.8,
recall ~0.9). Working data: `/data/aai/tbenchtrain/decontam_calibforge/` (fingerprints, verdicts, audits;
`variants/work-g106.tar` is the work dir the variants list was built from).

Round 6 (2026-11-01, `variants/run_round6.sh`): round 5's `fp` signal came from a cache older than the v3
fingerprints, so 318 fingerprint-window pairs were judged afterwards; all NO_REUSE, list unchanged (`variants/final-r6`).

Public versions: the list (task IDs, reasons, matched test tasks; no data copy) at
huggingface.co/datasets/ricdomolm/CalibForge-decontaminated, and a clean port of this pipeline (OpenAI-compatible
endpoints, API second opinion) at github.com/rdolmedo/calibforge-decontam. On this run's inputs the port reproduces the
retrieval scores, bundles and judge prompts exactly.

## Pipeline

Each module's docstring has its commands.

1. `retrieve` (CPU): candidate (TB2.1, CalibForge) pairs, the top-k under several lexical and embedding signals.
2. `render`: one text bundle per task (instruction, tests, solution, Dockerfile, environment files) and per-pair
   overlap signals.
3. `serve_judge`: DeepSeek-V4.1-Flash-1017 on 4xB200 (condor bid 63); release with `touch <home>/RELEASE`.
4. `judge`: single-turn verdicts per pair. `rubric.md` (DUPLICATE / REWRITE / RELATED / UNRELATED) produced the
   rewrites list. `rubric_reuse.md` with `--fingerprints` (REUSE / WEAK_REUSE / NO_REUSE against each TB2.1 task's
   candidate-blind fingerprint) produced the variants list.
5. `apply` (rewrites list) and `variants` (variants list): near-copy propagation, deepening, re-sampling, audit
   samples and scoring, and the final list (`variants final --rule any --all-weak`).
6. `compare`, `compare_irt`, `irt_k9/`: pass@1 of the decontaminated SFT runs against their references.

`tasks.py` turns Harbor task directories into text for retrieval and rendering.

Known gap: four TB2.1 tasks (crack-7z-hash, vulnerable-secret, filter-js-from-html, break-filter-js-from-html) have no
fingerprint; their issue-#14 exclusions are carried over unchanged (see #14).

## task_synthesis tasks vs TB2.1 + TB-Lite (issue #19)

`synth.py` runs the same variants criterion on synthetic task bundles (student 1: campaigns r5, s1, s2; 7,296 tasks).
Its docstring has the commands, and `synth_stage2.sh` / `synth_stage2b.sh` are the CPU jobs that ran it.

- **Rubrics:** `rubric_reuse_synth.md` and `rubric_reuse_nofp_synth.md`. The generator never saw TB in its prompt, and
  the candidates ship no tests, so the author's solution sketch stands in for them.
- **Forced pairs:** each task is paired with the TB2.1 task that is the only source of one of its sampler spec details.
- **TB-Lite:** DeepSeek fingerprints plus a no-fingerprint arm on each test task's closest candidates, as a per-arm union.
- **Result (2026-10-31):** 36 tasks (0.49%), list in `/data/aai/tbench/tasksynth/decon/variants/`, work dir
  `/data/aai/tbenchtrain/decontam_tasksynth/`. The real cases trace to the spec detail `database:wal-journal`
  (db-wal-recovery). No retrain: the only flagged test task in tbmix-irt100 scores 0/5 for student 1. Details on #19.

## Production pools (issue #119)

`pool.py` runs #19's method on the task-synthesis production layout, from task streams (`<stream>/chunks/NNNN.tar`)
or bundle tars, and can run while generation is still going. Its docstring has the commands. `pool_job.sh` is the
CPU-job wrapper; submit it from a snapshot worktree.

- **Same decision rule.** It uses the same rubrics, fingerprints (TB2.1 Opus v3, TB-Lite DeepSeek v1), arms
  (fingerprint, no-fingerprint, TB-Lite no-fingerprint) and union. A pair is excluded on any REUSE or when every
  sample says WEAK_REUSE. Each test task's top-50 window per signal is judged; listed candidates then leave the
  window, and this repeats until no window holds an unjudged pair.
- **Input.** Tasks are read straight from the tars, with no extraction. A real-asset task's
  `environment/tasksynth/asset.tar.zst` is opened: its files count as environment files (hashed for the
  identical-file signal, text heads shingled and shown), and the judge also sees the asset's title, source and URL.
- **One pass near the end (user, 2026-11-03).** Only the final top-50 windows matter, so judging earlier wastes
  judge time. While generation runs, chunks are only read into `W/units/` (CPU). `run --follow --checkpoints 0.9
  --expected-tasks N --snapshot-when <round>/STOP` starts the main pass at ~90% of the planned pool, so it overlaps the
  last stretch of generation on the generator servers. If the round's STOP (token budget covered) comes first, that
  starts it instead.
  At the streams' END a last snapshot recomputes the signals over the whole pool and judges only the pairs new in the
  windows, with the same stopping rule. It publishes unless more than 5% is excluded. Lists go to `W/lists/<tasks>/`
  and `W/lists/LATEST.json`. `--hold FILE` exists until then: the generation driver should keep a DeepSeek server
  while it does (#118's driver releases them when the round finishes).
- **Changes from #19 for scale.** Forced spec-detail pairs are a windowed signal instead of all being judged.
  The boilerplate thresholds (8 tasks for shingles, 3 for files) scale with pool size from student 1's 7,296 tasks.
  Above 40k tasks, boilerplate among a candidate's own shingles is estimated on a stable hash sample; the test tasks'
  shingles keep their exact counts.
- **Five windowed signals, deeper windows where flags are (user, 2026-11-04).** Windows come from instr,
  instr_tests, embed, fp and forced (`--signals`). ids and overlap are still computed and shown to the judge, but get
  no windows: on student 1 (36 flagged) and the #125 pilot (11) neither found a flagged task alone, and they were
  ~26% of the judged pairs. #19's deep rule is built in (`--k-max 100`): a flagged pair in the bottom half of a test
  task's window doubles that window. Student 1, replaying pool.py's window loop with #19's verdicts: seven signals
  at top-50 judged 43,285 pairs and found 34/36 of #19's list (misses: db-wal-recovery variants at ranks 64 and 93);
  five signals at top-50, 31,747 pairs, 34/36; five signals with the deep rule, 32,034 pairs, 36/36 (db-wal-recovery
  and git-repo-forensics deepened to 100).
- **Admission during the main pass (user, 2026-11-04).** The generation round admits new specs while its servers
  hold fewer than `target_inflight` requests each, and it counts the judge's requests too. A snapshot taken while
  the streams are open raises `<round>/admission.json` `target_inflight` by the judge's in-flight requests per server
  (`--concurrency` + 2 x `--concurrency-small` over the healthy servers: 2,000 -> 2,384 on two servers) and restores
  it afterwards (`--admission`, `--admission-base`; `--production` fills both). Other keys stay; a target changed by
  hand in between is left alone; `W/admission-bump.json` lets a restarted run undo a bump left by a killed one.
- **Judge on the generation engines.** `--serving gen-ds+gen-ds2` reads task-synthesis serving files. Each test task
  is pinned to one server and DP rank (`X-data-parallel-rank`), so its shared prefix stays cached.
- **SFT.** `prepare.py --exclude-tasks` takes comma-separated lists (branch sft), so the M1 prep can pass
  `<CalibForge list>,<pool list>`.
- **Checks.** `pool_regress.sh` rebuilds student 1's pool and compares it with `synth.py`'s run (`pool_check.py`).
  `pool_mock.py` runs a growing stream (main pass on a STOP file, last pass at END) against a mock judge with two DP ranks. `pool_cost.py` is the cost model.

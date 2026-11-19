# m5-omniscience: synthetic SFT data for AA-Omniscience (#354)

Grounded, long-tail factual questions in AA-Omniscience's style, one Super trace each with the eval's own prompt,
harness and limits. Problems only: the generator's answer is kept solely for the problem-side checks and the spirit
check, and is never exported.

## Pipeline

| Step | Code | Where it runs |
|---|---|---|
| Taxonomy top levels: 6 domains (v1.1 split 20/20/15/15/15/15), AA's 86 subtopics with smoothed weights | `taxonomy/fit.py` -> `subtopics.json` | local (seconds) |
| Leaves: ~4.7k brainstormed by Opus per subtopic, each with 4 BM25 queries and preferred sources; single-test-item guard | `taxonomy/brainstorm.py`, `taxonomy/build_leaves.py` -> `leaves.json` | local `claude -p` |
| Sources: docs/changelog repos, WG14 C drafts, PubMed Central, Common Pile CAP, EDGAR-CORPUS (the swezero changelog extraction hung on Lustre and was dropped) | `sources/download.py` | CPU job |
| Passage pool: BM25 per source (+ the mmlu-search Wikipedia index), seed blocking against test items, canary drop, per-source interleave shares (pool2: 70/leaf, Wikipedia 30%) | `pool/build.py` | CPU job |
| SWE top-up: language-matched doc windows per SWE subtopic that pool2's 2-per-document cap left unused, seed blocking re-run (pool2b: +8,170) | `pool/augment.py` | CPU job |
| Generator (DeepSeek V4.1 Flash, high effort, no cap, 6 h timeout; prompt `gen/prompt.md` v5) -> rule checks -> within-leaf near-dup check -> DeepSeek critic at low effort (`gen/critic.md`: ambiguous / unstable / off-subtopic / not short / garbled; no difficulty criterion) -> Super traces, streamed | `gen/pipeline.py` | CPU job |
| Pilot metrics (repeat meter projected to 66k, 8-gram, spread, spirit stats), grade mix (check only) and DeepSeek difficulty proxy with the eval's ds-max settings | `analysis/meter.py`, `analysis/grade.py`, `analysis/proxy.py` | CPU job / local |
| Decontamination: TF-IDF + char + bge retrieval, DeepSeek judge x2 (SAME_FACT / SCAFFOLD exclude) | `decon/run.py` | CPU job |
| Final decontamination of a mixture's problems (#354): 6 signals over all 300 x N pairs, deepening, audit, planted controls, DeepSeek x2; scan of the trained text for canaries / verbatim test text | `decon/full.py`, `decon/scan.py`, `decon/controls.json` | CPU jobs |
| Export to sft/prepare_data.py's format; `--token-target` with `--seed` gives a seeded random-by-problem cut; SHA256SUMS | `export.py` | CPU job |

Jobs are submitted with `submit_cpu.sh NAME CPUS MEM_MB DISK_GB -- cmd` from a frozen detached worktree of HEAD.
Data lives in `/data/aai/m5/omniscience/` (sources/, pool2/, pool2b/, pilot2/, production/, decon-prod/, export*/, logs/).

## Sampling and serving
- Subtopic weight = taxonomy share / max(0.25, acceptance rate) x clamp(remaining deficit, 0.5, 2), so the accepted mix
  tracks the targets; subtopics under 12% acceptance after 50 calls (pool barely covers them) keep their plain share.
- Passage choice among the leaf's next 8 by retrieval rank, weighted 0.85^rank x source weight
  (`--source-weights wiki=0.2,web=0.6,Finance:web=0.05`: Wikipedia is easiest for models; FineWeb Finance pages are
  mostly generic). A leaf that runs dry reuses accepted passages once for a different detail (pid suffix `:r`).
- DeepSeek client follows alloc.json `ds.serving` (any server with endpoints.json < 120 s old), pinned per
  (server, DP rank) with `X-data-parallel-rank`; failed requests retry on the current list. Super sessions per engine
  follow alloc.json too.
- Every /data write goes through `fsio.py` (fsync + size check + rollback/retry for appends; temp + sha256 verify +
  rename for files).

## Trace settings (= M3.6 `m36-super-vl-omniscience-full`)
System: AA's answer prompt with domain and subtopic (85% exact, 15% one of six paraphrases); user: the bare question;
temperature 1.0, top_p 0.95, `enable_thinking` true, `medium_effort` false, `truncate_history_thinking` false,
`max_tokens` 129,024 (131k context), 3,600 s timeout scored as the model's runaway. Every ending is kept upstream;
export drops no-submit traces and decon exclusions.

# Audit log

One entry per round: what ran, the results against the quality bar (PIPELINE.md), what the reading found, and what
changed before the next round. Newest first.

## Trace judge dropped from production, issue #57 (2026-11-03)

**A/B:** student 1 (`tasksynth-4b-ds-330m-3ep`) against the same data without the trace judge's task-defect filter.
The 326 flagged tasks (327 traces) went back in, 6.1% of train tokens. Same token cap (321.7M), val set, code and
recipe (3 epochs, bf16, lr 6e-6), and the same eval profile (bf16 KV, no turn gate, 2x H100).
**Result:** tbmix 44.4% vs 39.2%, paired task bootstrap +5.2 pp [-0.0, +10.6] (p 0.053); projected TB2.1 +2.0
[-0.4, +4.2]; val BPB on the same val set 0.23527 vs 0.23529.
**Decision (rule fixed before the run):** drop the judge unless the no-judge arm is significantly worse. It is not,
so production campaigns run without `traces run`, and SFT prep keeps every task's traces (no `--drop-tasks` from
verdicts). This saves ~50 B200-h per 1B training tokens.
**Reading:** "no harm", not "the judge hurts". Student 1's eval reproduces (fp8kv re-eval 38.4%), and real assets
(#20) had the same +5.2 on tbmix, then 0.0 [-3.1, +3.1] on full TB2.1. The gap is likely checkpoint noise that the
task bootstrap does not include; #105 measures it with a seed replicate of student 1. Record:
/data/aai/tbench/tasksynth/ab/nojudge-ab/PLAN.md.

## Spec vocabulary v2 and the diversity dose-response, issue #25 (2026-10-31 to 11-03)

**Vocabulary v2** (`asset_sampler/vocab_v2.json`, built by `task_synthesis.vocab.build` from an Opus 5.5 brainstorm):
4 new actions, 5 domains, 5 material classes, 1,737 sub-domains, 271 formats, 84 languages. Terminal-Bench's options
keep their fitted shape; new options get at most 25% of each dimension; 20% of specs draw no sub-domain (open draws).
Two flatter designs use the same options: `v2d2` (core tempered 2x, flat new-option shares, open draws 10%) and
`v2d3` (uniform, sub-domain always).

**Arms.** Student 1's recipe throughout (Qwen3.5-4B, 321.7M Super Pi train tokens x 3 epochs, bf16, lr 6e-6); only the
spec vocabulary differs. Rounds `rv2a-i`, `rd2a-i`, `rd3a-i` (seeds 101-110, 111-119, 121-129; student-1 minutes mix).
Eval tbmix-irt100, default profile (fp8 KV + turn gate), same code and per-attempt seeds.

| | Student 1 (v1, k=5) | v2 (k=10) | D2 flatter (k=10) | D3 flat (k=10) |
| --- | --- | --- | --- | --- |
| pass@1 | 38.4 | **42.2** | 38.9 | 38.2 |
| Projected TB2.1 / TB-Lite | 19.4 / 36.4 | 21.2 / 39.3 | 20.0 / 37.4 | 19.3 / 36.2 |
| Raw TB2.1 (35 tasks) / TB-Lite (65) | 37.1 / 39.1 | 34.0 / **46.6** | 32.3 / 42.5 | 36.3 / 39.2 |

- Paired at k=10: D3 - v2 -4.0 [-7.7, -0.1] (p = 0.046; 0.064 without 12 attempts lost to a full disk), TB-Lite -7.4
  [-12.0, -2.8]; D2 - v2 -3.3 [-7.4, +0.6] (p = 0.09). v2 - student 1 on matched attempts 0-4: +5.0 [-0.6, +10.2]
  (p = 0.08), TB-Lite +8.0 [+0.9, +15.4]. The v2 top-up to k=10 replicated its TB-Lite score (47.1 -> 46.2 on new seeds).
- **Flattening the weights hurts**, mostly on TB-Lite. v2's gain is on TB-Lite; on TB2.1 no arm differs significantly
  (35 tasks; the TB-Lite vs TB2.1 gap of v2's gain is not significant, p = 0.12). Full TB2.1 for v2 (89 x 5,
  `tb21p180-tasksynth-4b-v2-330m-3ep`): 19.1% vs student 1's 20.0%, paired -0.9 [-3.6, +2.0] (p = 0.56), so v2's
  gain costs nothing measurable on TB2.1.
- Spec quality falls with flatness (judge pass on no-material specs 88.1 / 87.4 / 80.4%; reading audit 19/20, 9/10,
  7/10: forced combinations, materials silently dropped); trace-level quality barely moves. Why D2 loses is open.

**Hero vocabulary `v2h`** (`vocab_v2h.json`): v2 with open draws 20% -> 5%, nothing else. In v2's 8,740 tasks
(TF-IDF cosine of instructions, > 0.3 = same task concept), open draws are 20% of tasks but 40% of the tasks in
same-concept clusters and 75-100% of every cluster of 4+ (signed APT repository 13, DEM-hydrology CLI 10). Without a
sub-domain the generator falls back on its favourite task for the action and domain. Open-draw tasks add no coverage
(nearest-neighbour similarity to a sub-domain pool 0.130 vs 0.129) and match sub-domain tasks on yield, judge pass, Super
solve rate, defects and similarity to TB2.1/TB-Lite. Projected same-concept repeats per task at ~60k tasks (6B Super
tokens, 2 traces per task): 0.27 with v2, 0.15 with v2h. Sampler check (25k specs each): 4.8% open; action, domain
and anchor mix unchanged (chi-square z +1.6 / -0.8 / -0.3).

**Decision (2026-11-03).** `v2h` is the default vocabulary (`specs.DEFAULT_VOCAB`; `--vocab v1` for the old one): v2
beat v1 on tbmix at no measurable TB2.1 cost, and the hero run should not depend on a flag. The hero run uses both
defaults, `v2h` with the easy-heavy minutes mix; that pair is untested together (v2 was trained on the student-1 mix,
easy-heavy on v1). Easy-heavy traces are ~18% shorter (~41k Super tokens instead of ~50k), so 6B tokens is ~146k
traces, ~73k tasks at 2 per task. Under those inputs the projection is ~0.18 [0.11, 0.25] same-concept neighbours
per task with v2h; it is an output of stated assumptions (below), not a threshold.

**Use.** Hero: `round run|submit` with the defaults. Reproducing or comparing with earlier rounds: name the vocabulary
and the mix, `--vocab v1|v2|v2d2|v2d3 --minutes-mix student1`; this reproduces the specs of `r6a` (v1), `rv2a` and
`rd3a` byte-identically.

**Rechecking repetition when conditions change** (task count, traces per task, trace length, open share, minutes
mix): `/data/aai/tbench/tasksynth/vocab/v2/repetition/` has `repeat_check.py` (`measure` pair rates on any
rounds, `lengths` per minutes mix, `project` to a target) and a README with the assumptions, their current values
and sources: a repeat is TF-IDF cosine > 0.3 between instructions; repeats per task grow linearly with the task
count (checked only up to 8.7k tasks); v2's group pair rates carry over to other open shares and mixes; tasks =
tokens / (tokens per trace x traces per task); the acceptable level is not measured. It also lists the vocabulary
levers and what was measured about each (M1 note: #49).

**Seed replicates and `v2h0` (2026-11-04; supersedes the 11-03 reading that v2 beat v1).** Training-seed replicates
(`train.py --seed 1`, same data and arguments) of v2 and student 1, same eval profile and per-attempt seeds:

| | v2 seed 0 / 1 | Student 1 seed 0 / 1 | v2 - student 1, both seeds pooled |
| --- | --- | --- | --- |
| Full TB2.1 (89 x 5) | 19.1 / 18.7 | 20.0 / 20.4 | -1.3 [-3.7, +1.0], p = 0.24 |
| tbmix, attempts 0-4 | 43.4 / 39.4 | 38.4 / 42.6 (seed 1: bf16 profile) | +0.9 [-3.3, +4.8], p = 0.66 |

- v2's +5 on tbmix was seed luck on both sides; training seed alone moves tbmix by 3-4 pp. With D2/D3 compared against
  v2's lucky seed, "flattening hurts" is weaker too.
- TB2.1: both v2 runs are below all 11 full-TB2.1 runs on v1-vocabulary data (student 1 seeds and the #20 asset arms,
  20.0-22.7%). The deficit is all on mid-difficulty tasks (the 29 that v1 models solve 20-100% of the time) and on
  code-anchored software engineering (port-migrate, reverse-engineer, debug-fix; other-code, C/C++, upstream-project
  anchors). Mechanism: `cap_new` 0.25 gives ~24% of each dimension to options TB2.1 does not test; code-anchored specs
  are 37% of TB2.1, 33% of v1, 25% of v2 (23% of v2's actual tasks). The super-scaling ladder (~2 pp TB2.1 per data
  doubling) predicts ~-0.8 pp for a 25% dilution.
- The overall Python/C shift seen under v2 (Python 43% -> 50% of code assets) came from `cap_new`, not from languages:
  v2 changes languages only in the `other-code` class.
- Other-code languages, the three weights: real-world share = one Opus 5.5 order-of-magnitude guess of each language's
  share of public code (`vocab/brainstorm.py`, floor 0.001%); v2 weight = share^0.4 over 84 languages (JavaScript 7.4%,
  floor languages 0.16%); TB weight (v1) = per-task counts over the ~25 TB2.1/TB3.0 tasks with other-code assets, plus
  pseudo-counts for 11 hand-picked languages, square-root tempered, which gives 30 languages 2.2-6.4% each (one TB task
  is enough for ~3%) and every other language 0. TB weight / v2 weight: ~0.8 for JavaScript/TypeScript/Java, 10-28x for
  Terminal-Bench's niche languages (Redcode 28x, POV-Ray 20x, Coq 15x, Lean 12x, COBOL 10x).
- The user's call: v2 down-weights the tail too much, and Terminal-Bench may weight those languages for a reason, so
  every language used in a TB release gets at least 1% of other-code draws (below).

**`v2h0`** (`vocab_v2h0.json`, `round --vocab v2h0`) = `v2h` with `cap_new` 0 and a language floor: TB-fitted
actions, domains and material classes; keeps the sub-domains (1,216 in TB domains), 5% open draws, merged defect details
(the leakage fix: 0.11% vs 1.03% flagged) and v2's formats and 84 languages. Language floor (`language_floor`): the 22
languages seen in any TB release's other-code assets (`asset_labels.json`, all releases; HTML excluded as markup) get at
least 1% of other-code draws (~115 hero tasks each); 11 of them are lifted (COBOL, Coq, Fortran, Lean, OCaml, POV-Ray
SDL, Redcode, Sage, Scheme, Stan, VBA) and the other languages are scaled by 0.93 (JavaScript 7.4% -> 6.9%).
`language_weights` holds the resulting probabilities. Check (100k specs): every non-language dimension identical to
`cap_new` 0 without the floor (TV 0.000); floored languages drawn at 0.84-1.15% of other-code assets (~200 draws each). Production-path check (`specs.sample`, easy-heavy,
20k specs each): new options 0%; action / domain / anchor / works-on vs v1 TV 0.011 / 0.011 / 0.015 / 0.018 (chi-square
z -0.9 / -1.1 / -0.3 / +1.2); code-anchored 29.1% (v1 30.0%, v2h 22.7%); eligible for binding 72.2% (v2h 69.3%: at
`asset_share` 0.64 real-asset tokens 41.2% instead of 39.8%; 0.617 keeps 39.8%). Cost: ~1.37x v2h's same-concept
repetition (fewer action x domain cells and sub-domains), ~0.18 repeats per task at the hero's ~56k tasks. Untested
in training: a 330M arm with two seeds per side has a TB2.1 interval of about +-2.4 pp, too wide for a ~1 pp effect.

**Decision (user, 2026-11-04).** `v2h0` is the default vocabulary (`specs.DEFAULT_VOCAB`; `--vocab v2h` still draws
the 11-03 default) and the vocabulary for the rest of the M1 hero run (#165). `asset_share` stays 0.64: ~41% instead
of ~40% of Super tokens on real-asset tasks is too small a difference to matter.

## Difficulty target (`minutes`) study, issue #26 (2026-10-31 to 11-02)

**Question.** Each spec carries a difficulty target (expert minutes 15/30/45/60/90/120/180). It reaches the traces only
through the generator prompt: prepare.py gives every task the same agent timeout (21,600 s). Does the mix of targets
matter for the student?

**Effect on traces** (student 1's specs and Super traces; the target is drawn independently of the other spec fields,
so per-target comparisons are randomized). Each doubling of the target multiplies Super's saved tokens by 1.24 (median
24.8k at 15 min, 56.8k at 180). From 15 to 180 min, turns go from 13 to 26, context-cap hits from 2.3% to 15.1%, and
trace-judge solves from 81% to 57%. Instructions and sketches get longer; the kind mix, diversity and defect rates don't
change.

**Arms.** Each arm is a Qwen3.5-4B SFT run that matches student 1 (1,857 steps, 974M SFT tokens, lr 6e-6, bf16 fast
path); only the data differs. New tasks come from rounds m26-* and m26r2-* (`round --minutes`), with one Super trace
each from the streamed campaigns tasksynth-m26 and tasksynth-m26r2. Trace-judge defects are left out, as for student 1.
Mixes are task shares over 15/30/45/60/90/120/180 min.

| Arm | Data | tbmix pass@1 | TB2.1 / TB-Lite | vs student 1 | vs easy-heavy |
| --- | --- | --- | --- | --- | --- |
| very-easy | 75/20/5/0/0/0/0, 3 epochs | 37.6 | 29.1 / 42.2 | -0.8 [-6.2, +4.6] | -2.8 [-9.0, +3.0] |
| easy-heavy | 37/28/14/12/5/3/1, 3 epochs | **40.4** | 41.1 / 40.0 | +2.0 [-3.0, +7.2] | |
| student 1 | 8/17/15/20/15/15/10, 3 epochs | 38.4 | 37.1 / 39.1 | | -2.0 [-7.2, +3.0] |
| pooled | student 1's tasks + all m26 tasks, ~1.6 epochs | 37.6 | 32.0 / 40.6 | -0.8 [-6.2, +4.6] | -2.8 [-8.8, +2.6] |
| hard-heavy | 1/2/4/9/15/27/42, 3 epochs | 36.4 | 37.7 / 35.7 | -2.0 [-7.8, +3.8] | -4.0 [-9.4, +1.2] |
| truncated | student 1's mix; each trace cut at the turn end nearest easy-heavy's length at the same quantile; 3 epochs | 41.0 | 41.7 / 40.6 | +2.6 [-2.8, +7.8] | +0.6 [-5.0, +5.8] |

Eval: tbmix-irt100, pi180, k=5 (500 attempts), on fp8kv + turn-gate serving. Student 1 was re-evaluated on this
serving: 38.4, against 39.2 on bf16 in the entry below. Differences are in pp, with 95% CIs from a paired task
bootstrap.

- A somewhat easier mix than student 1's helps a little and a harder one hurts a little, but very-easy falls below
  easy-heavy: there is a sweet spot near easy-heavy. Over round 1's four arms, pass@1 falls 2.1 pp per doubling of the
  token-weighted geometric-mean target (p = 0.15). With very-easy included, a linear fit is flat (-0.7, p = 0.53).
- Truncated ties easy-heavy: with easy-heavy's trace lengths, student 1's harder task mix does as well. The gain over
  student 1 fits shorter traces as well as it fits easier tasks, and the data can't tell the two apart (the
  truncated vs easy-heavy CI is about ±5 pp).
- Repetition makes no difference: pooled spreads its tokens over more tasks at ~1.6 epochs.
- No single difference is significant.

**Decision (2026-11-02).** Easy-heavy is the default target mix (`specs.DEFAULT_MIX`). `round --minutes-mix student1`
reproduces the earlier rounds' specs exactly (checked on r5 and r6a; since #25's default change on 2026-11-03,
together with `--vocab v1`), and explicit weights work too
(`--minutes-mix 15=0.5,30=0.5`). **A study compared with student 1 must generate with `--minutes-mix student1`** (and, since 2026-11-03, `--vocab v1`) (and use student 1's fp8kv eval, 38.4); on the default mix, compare with easy-heavy (40.4). Easy-heavy matches truncated without cutting traces (truncation threw away 17% of the generated tokens), and its traces are cheaper to generate (fewer cap
hits, more solves). Revisit if the teacher or student changes. An RL task pool may still want long targets.

**Eval incident.** The first truncated eval was served from cpu208 (511 generated tok/s, against ~2,200 on other
nodes). 126 attempts ended with Pi's "Request timed out." after its retries and were scored 0: `worker.classify`'s
`pi_backend_unreachable` rule doesn't match timeouts (shared eval code, left unchanged). Moving the server to another
node (`submit.py serve` takeover) and retrying only the cut-off attempts gave 41.6, biased toward solves because cut-off
attempts are the long ones. A clean full re-run gave 41.0. When many attempts are cut off, re-run the whole eval.

**Where things are.**
- Study scripts and outputs: `/data/aai/tbench/tasksynth/_checks/minutes26/`.
  - Step-1 analysis, `select_arms.py`, `ops/sft_chain.py`.
  - Round 2 in `r2/`: `select_r2.py`, `cut_map.py`, `sft_chain_r2.py`.
  - Tables in `analysis/`.
- Selections: `/data/aai/tbench/tasksynth/sft/m26/` and `sft/m26r2/`.
- SFT prep options (`--cut-len-map`, `--drop-tasks` for trace-judge defects) are on the sft branch since 1657a0d (the runs used snapshot 7e4ce8b, where `--drop-tasks` was still called `--exclude-tasks`).
- Under `/data/aai/tbenchtrain/`:
  - datasets: `sft_data_q35/tasksynth-m26*`;
  - checkpoints: `sft_q35_ckpts/tasksynth-4b-m26*`;
  - evals: `eval_q35/tbmixp180-tasksynth-4b-m26*`. Use `...-m26r2-trunc-3ep-rerun` for truncated.

## Real-asset A/B result (2026-11-01)

Treatment `tasksynth-4b-ds-330m-realasset-3ep` vs student 1 `tasksynth-4b-ds-330m-3ep`. Both use the same config, and the
treatment has 319.6M vs 321.7M train tokens. Both were scored on tbmix-irt100 with student 1's eval profile (bf16, no turn
gate, 2xH100). The comparison is a paired task bootstrap (`irt_mix.py compare`, `ab/swap-ab/compare.json`):

| | Student 1 | Real-asset | Diff (95% CI) | p |
| --- | --- | --- | --- | --- |
| Raw pass@1 | 39.2% | 44.4% | +5.2 pp [+0.2, +10.4] | 0.047 |
| Joint ability | 0.597 | 0.771 | +0.17 [-0.01, +0.36] | 0.068 |
| Projected TB2.1 | 20.2% | 22.4% | +2.2 pp [-0.2, +4.5] | 0.068 |
| Projected TB-Lite | 37.7% | 41.3% | +3.6 pp [-0.2, +7.4] | 0.068 |
| TB2.1, own tasks | | | -0.0 pp [-3.8, +3.8] | 0.97 |
| TB-Lite, own tasks | | | +5.3 pp [+0.4, +9.9] | 0.039 |

**What the comparison shows:**
- TB-Lite gains clearly, while TB2.1 on its own tasks is flat. The ability scale corrects for difficulty, and TB-Lite still gains more than TB2.1.
- Code-primary tasks gain most: ability +0.40 vs +0.14 for the rest. This is a post-hoc slice.
- A full TB2.1 eval of both students (fp8kv + turn gate, 89 x 5) was requested to measure TB2.1 directly.

**Qualitative read (non-code; `ab/swap-ab/qual/SUMMARY.md`):**
- Real non-code assets exercise the same skills on messier material.
- About a third of bindings are decorative, and `asset_use >= 4` let them all through.
- Defects come from the author writing the spec against an asset it cannot see.
- The real-asset student gains from end-of-task discipline. It loses by digging in after a wrong diagnosis, often until the context cap, which may be inherited from real-asset traces (cap hit 11% vs 4%).

**Next (proposed):** an agentic author that explores the asset in a sandbox, a grounding step where the reference solution is run and the expected values come from its output, a counterfactual centrality judge, and an asset-first arm. Test these as a generation-side A/B first.

## raf0-raf7 + swap campaign: the real-asset A/B data (2026-10-31 / 11-01)

**Binding (bf1, 5,269 specs):** 4,256 bound (81%); with the 101 audit specs, 4,315 of student 1's 5,370 best-generated
specs have a real asset. Weakest classes: database 55% (fit check rejects), ml-model 63% (few candidates), text-data
70%, python-code 75% (202 cells without candidates). 34 late second-pass bindings were not generated (they keep
student 1's tasks).

**Generation (prompts unchanged since the gate):**

| Rounds | Specs | Environment-valid (first try) | Judge pass of valid | asset_use |
| --- | --- | --- | --- | --- |
| raf0-raf7 | 4,115 | 3,959 = 96.2% (72%) | 3,517 = 88.8% | 4.78 |
| raf1-raf2 (300 in flight per round) | 1,678 | 97-98% (74-77%) | 88-89% | |
| raf3-raf6 (all 2,258 in flight, 192 builds) | 2,258 | 95-96% (68-74%) | 87-90% | |

Streamed to the campaign: 3,516 (one near-duplicate dropped, cosine 0.66; decon 0). With ra2 + ra3 (pilot, 178):
3,694 real-asset tasks.

**Super traces** (`20261031-nemotron35-super-tasksynth-swap`, streamed; pilot traces reused): 212.5M saved tokens
on 3,516 tasks (5 without a usable trace). Trace judge (gen-ds, then gen-m26a, same model and settings): solved 2,204
/ partial 959 / failed 483 of 3,646 attempts; **task defects on 416 of 3,689 traced real-asset tasks (11.3%; student 1
4.5%)**.

**Selection** (`assets.ab select`, seed 0, `ab/swap-ab/selection.json`): all 3,694 swapped student-1 tasks out
(3,484 of them in student 1's train split, 161.6M of its 321.7M train tokens); 2,826 of 3,273 defect-free real-asset
tasks in, in seeded random order to the same tokens.

**SFT data** (`sft_data_q35/tasksynth-ds-330m-realasset`, student 1's prepare.py arguments + the two real-asset
roots): 319.6M train tokens (student 1 321.7M, -0.64%; the cap did not bind: Qwen tokens per saved token came out
slightly lower on real-asset traces), 171.3M loss tokens (-2.1%: more context per assistant token), 6,217 train
sequences (6,913).

## swap-pilot: Super traces on real-asset tasks (2026-10-31)

One Super Pi trace per judge-passed ra2 + ra3 task (178 tasks, student 1's campaign recipe; campaign
`20261031-nemotron35-super-tasksynth-swap-pilot`, trace judge `traces/<campaign>/REPORT.md`). Compared with student
1's traces of the same 178 specs (their generated-asset tasks):

| | Real-asset pilot | Student-1 originals | All student 1 |
| --- | --- | --- | --- |
| Saved tokens per trace (mean / median) | 56.9k / 45.6k | 41.3k / 32.8k | 45.3k-46.6k mean |
| Assistant turns (median) | 22 | 16 | 17 |
| Tokens per assistant turn (median) | 2,096 | 2,014 | 2,009 |
| Ended at the context cap | 11.2% | 3.9% | 4.9% |
| Judge: solved / partial / failed | 64% / 29% / 7% | 75% / 21% / 4% | 68% / 23% / 8% |
| Trace-judge task defects | 9.0% | 3.4% | 4.5% |
| Install failed offline | 2 / 185 | | ~1% |

- **Longer because there is more to do, not because of big reads:** tokens per turn are unchanged; the agent takes
  more turns (real projects, real data quirks). Traces are longer than the originals for 113 of 178 tasks.
- **Twice the task-defect rate.** Most defects are the author's guesses about real content it could not inspect: a
  sketch expecting 32 rows where the workbook has 31, a key using the textbook 727 °C where the real chart prints
  723 °C, a filter no real run passes, a real repository shipping a failing test the instruction says must pass, title
  rows in a real workbook that a literal rule counts. Defect tasks leave the SFT data, as in student 1. Follow-up
  (not in the A/B, it would reset the audit gate): let setup compute the numbers the instruction and sketch depend on,
  and run the asset's own tests at build time.
- **Offline is fine:** 41 traces tried an install, 2 failed offline; the package proxy blocked only telemetry and
  GitHub fetches.

**Consequence for the A/B:** at the agreed token match, the replaced student-1 tokens buy traces for about three
quarters of the swapped tasks (the swap campaign's random task order decides which); every swapped task's student-1
traces leave the data, and the SFT prep caps train tokens at student 1's 321,688,447.

## ra2 + ra3: real-asset audit gate passed (2026-10-31)

Two rounds of 100 random bf1 bindings each, prompts and code unchanged between them (1e33dba), gen-ds generating and
judging (`rounds/ra2/REPORT.md`, `rounds/ra3/REPORT.md`).

| Metric | ra2 | ra3 | Bar |
| --- | --- | --- | --- |
| Environment-valid share of specs | 98/100 (first try 67) | 98/100 (first try 65) | >= 85% |
| Judge pass share of valid tasks (asset_use >= 4) | 90/98 (92%) | 88/98 (90%) | >= 75% |
| Random judge-passed tasks read and acceptable | 20/20 | 20/20 (2 borderline) | >= 18/20 |
| Near-duplicate pairs | 0 | 0 | <= 2% |
| Decon-flagged tasks | 0 (max cosine 0.18) | 0 (max cosine 0.18) | <= 2% |

- **No empty replies** after telling the generator it cannot inspect the asset; no truncated replies.
- **The asset carries the task** in almost every task read: fixing a real Python binding's 64-bit eventfd bug
  (linuxfd), porting a real ledger kernel to Julia against its tests, modelling a real Ubuntu static library's terminfo
  acceptance in pure Python, repairing corrupted copies of a real SRF-cavity SQLite campaign, a supply-chain audit of a
  real ViT safetensors checkpoint, recovering a force-reset git history (theft), rebuilding a lost BIC scorer from a
  `.pyc` oracle (ges), a Raft conflict-resolution speed-up in real Zig, index pruning on a real Postgres schema,
  byte-exact repair of a damaged real JPEG, a real Danish Electra checkpoint with a broken config behind a service.
- **Borderline (ra3):** a real image used only as a steganography carrier; Basque labels to read off a phase diagram
  without OCR tools.
- Judge scores: asset_use mean 4.83 / 4.85; difficulty 4.11 / 4.07; estimated/target minutes median 1.00.

**Gate passed.** Next: full generation of every bound spec with these prompts (raf0 = ra1's 93 regenerated, raf1+ =
bf1 bindings; ra2 and ra3 tasks are kept as they are), and a Super trace pilot on the 178 judge-passed ra2+ra3 tasks
(`synth/campaigns/nemotron35-super-tasksynth-swap-pilot.toml`).

## ra1: first real-asset round (2026-10-31)

The 93 bk1-targeted bindings, DeepSeek-V4.1-Flash-1017 generating and judging (gen-ds), code at bcb49d1
(`rounds/ra1/REPORT.md`).

| Metric | ra1 | Bar |
| --- | --- | --- |
| Environment-valid share of specs | 85/93 (91%; first try 56) | >= 85% |
| Judge pass share of valid tasks (asset_use >= 4 required) | 77/85 (91%) | >= 75% |
| Random judge-passed tasks read and acceptable | 20/20 (4 with partial asset use) | >= 18/20 |
| Near-duplicate pairs | 0 (nearest-neighbour cosine median 0.09) | <= 2% |
| Decon-flagged tasks | 0 (max cosine to an eval task 0.24) | <= 2% |

- **Reading:** reverse-engineering the compile-time settings of a real phase-field code (MPF) from its binary,
  restoring a rewritten git history from the object database (nix-ninja), nested-CV on real Sable Island microbiome
  tables, fixing a CDS checker against real GenBank records, a feature in a real invoice matcher, a real C PDF parser
  behind a TS gateway, a real dbt adapter with a removed plugin registration. Partial asset use in 4: the queuety broker
  reimplemented in Python with the real Go source only as a tarball, an authored Lua gate next to the real one, a single
  real HTML template in an authored builder, a real word list feeding an authored packer.
- **Judge fails (8) are real defects:** wrong answer keys in sketches (an IPFIX element, a G-code lead-out), stale
  captures, editable acceptance inputs, and one asset_use 2 (vendored source as backdrop, bug in authored code).
- **Build weakness:** first-try validity 61% (r2: 76%); 8 env failures. Two causes found:
  - harbor_ns retries a failed RUN once; the asset RUN deleted the tar before setup, so the retry failed on
    `tar: Cannot open` and that, not the setup's own error, reached the generator. Some tails were apt/pip/cargo
    progress only.
  - Real projects need newer toolchains than Ubuntu 24.04 ships (two Rust crates failed on rustc 1.75).
- **8 empty first replies:** the generator tried to inspect `/tasksynth/asset` (one reply is a literal
  `<bash>ls -la /tasksynth/asset/</bash>`; the rest were tool calls the server parsed out of the content).

**Changes before ra2** (4483a5f, 1e33dba): the asset tar survives RUN retries; the generator sees the first failure with
progress noise removed and error lines first; prompt: when the agent works on the anchor's kind of material the work
happens in the asset's own content; old-toolchain hint (install a pinned release or ship a part that builds); the
generator is told it cannot inspect the asset, the listing is all it gets; asset text budget 70k -> 50k characters.
ra1's specs are regenerated in the full run (its prompts predate these changes).

## Real assets: sourcing bake-off bk1 (2026-10-31, issue #20)

Design agreed with the user: route by fit, not by role label. Each of student 1's 5,370 bundled best-generated specs
keeps its spec; a real anchor is sourced by class x formats x domain (no action in the query), a fit check decides,
and a spec without a fitting asset keeps its generated task in the A/B. 101 of the 5,370 specs, stratified by anchor
class (`assets/specs/audit1.jsonl`), bound twice on gen-ds: query planner with the domain target vs without it
(BLUEPRINT first experiment #2, reduced to the two agent-search strategies).

| Metric | Targeted (`binds/bk1-targeted`) | Untargeted (`binds/bk1-untargeted`) |
| --- | --- | --- |
| Specs bound | 93/101 (92%) | 84/101 (83%) |
| Fit checks passed | 93/107 (87%) | 84/119 (71%) |
| Nearest-neighbour cosine of bound assets (title+description), median / p90 | 0.09 / 0.21 | 0.09 / 0.27 |
| Near-duplicate asset pairs | 0 | 0 |
| License verdicts of bound assets | 93 redistributable | 84 redistributable |
| HTTP requests / bytes | 1,070 / 764 MB | 1,430 / 1.27 GB |
| LLM completion tokens | 1.79M | 1.68M |

- **Targeted wins** on coverage, fit rate, tail diversity and cost; untargeted assets are often off-domain (a 6G FIPS
  library for a systems-os spec, a Redis clone for a data-science spec). Default: targeted.
- **Fit decisions read sensible:** GenBank records for a parsing fix, a 982-row thermal-conductivity table for CPU ML,
  a Rust Postgres-wire proxy for a deploy task; refusals for diagrams as reverse-engineering targets and Wikipedia
  articles as security artifacts (those specs keep generated assets).
- Sources of the 93 targeted bindings: GitHub repos 51, GitHub files 18, Zenodo 12, Commons 6, HF models 4, NCBI 1,
  Ubuntu packages 1. Asset tars: median 50 KB, max 177 MB, 493 MB in total.
- **Asset decon added** (`assets/decon.py`): candidates whose GitHub repo, HF id or URL an eval task references are
  blocked (98 repos, 2,498 URLs), and assets sharing any non-boilerplate file (>= 512 B) with a TB2.1/TB-Lite task are
  rejected (1,988 file hashes). The 93 exported bindings had no hits.

**Next:** audit round ra1 (the 93 targeted bindings) through generation, checks and the judge with asset_use.

## Proof of concept: 330M Super tokens, two students (2026-10-30/31)

**Data.** Rounds r6a-r6h (8,000 specs; 97.6-98.9% valid, 7,069 judge-passed, 0 decon flags) traced by Super Pi in
campaigns s1 (3,434 tasks, 163M), s2 (3,011 complete, 131M) and s2b (586 tasks s2's budget-mode tail rule cut, ~43M),
plus the r5 pilot (12M): ~346M saved tokens, one trajectory per task. Sonnet cohort tasks (1,031 judge-passed) traced
8 times each in s3-sonnet and s4-sonnet (492M). Trace-judge task defects (DeepSeek 4.6%, Sonnet 2.4%) and truncated or
corrupted attempts were left out of SFT (`/data/aai/tbench/tasksynth/sft/`).

**Students** (Qwen3.5-4B, bf16 fast path, lr 6e-6, branch sft-tasksynth 2604353; eval tbmix-irt100, pi180, k=5):

| Student | Data | SFT tokens | tbmix pass@1 | Projected TB2.1 | Projected TB-Lite |
| --- | --- | --- | --- | --- | --- |
| 1: DeepSeek tasks | r5+s1+s2+s2b, 321.7M x 3 epochs | 974M | **39.2%** | **20.2%** [18.3, 22.1] | **37.7%** |
| 2: Sonnet + DeepSeek | Sonnet tasks x8 traj once (479M) + student 1's data x1.51 | 965M | 29.8% | 15.9% [14.4, 17.5] | 30.5% |

Paired IRT differences in projected TB2.1 (irt_mix.py compare):

| Comparison | Student 1 | Student 2 |
| --- | --- | --- |
| vs base Qwen3.5-4B (16.9) | **+3.1 [+0.5, +5.9]** | -1.2 [-3.7, +1.5] |
| vs CalibForge-1B fast path (18.5) | +1.7 [-0.6, +4.1] | **-2.6 [-4.7, -0.3]** |
| vs CalibForge-decon-1B (18.1) | +2.0 [-0.2, +4.4]; own TB2.1 tasks **+3.8 [+0.7, +7.0]** | |
| vs 3B-token mix (19.7) | +0.5 [-2.0, +3.1] | |
| vs RTS 1B synthetic tasks (15.1) | **+5.0 [+2.5, +7.6]** | |
| student 2 vs student 1 | | **-4.2 [-6.6, -2.0]** |

- **Student 1 is the best 4B student so far**: above base on TB2.1 and TB-Lite, above the RTS synthetic-task student,
  level with or above every curated-Super student at 1B tokens and with the 3B-token mix. Synthetic DeepSeek tasks with
  one Super trace each work. Val BPB bottomed in epoch 2 (0.2335) and rose slightly in epoch 3 (0.2353).
- **Student 2 is worse**, although Sonnet's tasks were better as tasks (generator cohort above). On tbmix it hits
  the context limit in 45% of attempts (student 1 34%, CalibForge 34%) and its median attempt runs 41 min (21, 25): it
  learned the long, often unfinished trajectories Super produced on Sonnet's harder tasks (59k mean saved tokens, ~11 pp
  fewer judged solves). Confounds: 8 trajectories per task over ~1,000 tasks (half the tokens) vs 6,900 distinct
  tasks, and DeepSeek data seen 1.5x instead of 3x. Not a test of generator quality alone.

**Next questions:** Sonnet tasks with one trajectory each at matched task count; length or success filters on
trajectories (both students keep all outcomes); more DeepSeek tasks instead of epochs (student 1 overfit mildly in
epoch 3); streaming tasks into trace campaigns (BLUEPRINT open item).

## Generator cohort: Sonnet 5.5 vs DeepSeek-V4.1-Flash on the same specs (2026-10-31)

Research cohort (user, 2026-10-30): Claude Sonnet 5.5 (medium effort, the user's subscription through headless
`claude -p` and early subagents) wrote one reply to the exact generator prompts of 1,349 specs of r5, r6c and r6h; no
repair turns. Each reply went through the same parse, build, checks and DeepSeek judge as a round's attempt 0
(`pipeline/cohort.py`), paired with DeepSeek's attempt 0 on the same spec. Valid, judge-passed Sonnet tasks then got
8 Super Pi trajectories each (campaigns s3-sonnet, s4-sonnet) for student 2.

| Paired measure | Sonnet 5.5 | DeepSeek a0 | Difference |
| --- | --- | --- | --- |
| First-try valid (1,339 specs; infra excluded) | 84.6% | 78.5% | +6.1 pp (McNemar p = 4e-5) |
| Judge-passed per spec | 76.8% | 69.8% | +7.0 pp (p = 6e-5) |
| Median reply characters | 11-13k | 16-18k | Sonnet terser |
| Super trace saved tokens (869 twin specs) | 58.7k | 46.2k | +12.5k [+9.7, +15.3] |
| Assistant turns | 26.7 | 22.3 | +4.5 [+3.0, +5.9] |
| Trace judge "solved" | 62.9% | 74.2% | -11.3 pp [-15.3, -7.3] |
| Requirements met | 0.84 | 0.92 | -0.08 [-0.10, -0.06] |
| Task defect flagged | 2.8% | 4.0% | -1.3 pp [-3.0, +0.4] |

Sonnet's tasks are more often valid without repairs and draw longer, harder Super trajectories (the trace judge is
lenient on outcome, but equally so for both arms). Caveats: the judge is DeepSeek (may favour its own tasks);
Sonnet ran at medium effort vs DeepSeek at max; some Sonnet failures are refusals after safety stops (counted as
format failures). Corrupted task dirs from two overlapping cohort jobs (30 tasks) were rebuilt and rechecked; their
s3 trajectories are excluded from training. Numbers: `/data/aai/tbench/tasksynth/cohorts/generator_study_numbers.txt`.

## r4 (2026-10-30) — first trace pilot

300 specs (seed 4), the post-gate prompts and parser (7f653d3), 300 requests in flight (r1–r3 were capped at 96
by the client; fixed), base image v2 (`rounds/r4/REPORT.md`).

| Metric | r4 | Bar |
| --- | --- | --- |
| Environment-valid share of specs | 292/300 (97%; 228 on the first try) | ≥ 85% |
| Judge pass share of judged valid tasks | 262/292 (90%) | ≥ 75% |
| Near-duplicate pairs | 0 (nearest-neighbour cosine median 0.14, max 0.47) | ≤ 2% |
| Decon-flagged tasks | 0 (max cosine to an eval task 0.29) | ≤ 2% |

- **Post-gate fixes held:** no `leaks_solution` flags (r2: 3), no format failure on the last attempt.
- **Throughput:** ~32.6k generated tok/s with ~300 requests running (r3: ~9k at 96). 297 specs in 24 min; the last
  three took another 40 min, each stuck in image builds that hit the 25-minute timeout (3 of 382 builds; successful
  builds in r2–r4 took ≤ 222 s). The build timeout is now 10 minutes with its own repair message (cd6eb09).
- **Judge fails (30)** read as real defects: wrong expected values in the author's sketch, sources the build never
  creates, README claims the data contradicts, an empty environment for an instruction that promises a package.
- **Traces:** the 262 judge-passed tasks (no decon flags, no near-duplicates) ran one Super Pi trajectory each in
  campaign `20261030-nemotron35-super-tasksynth-r4`, with no network (smokes passed; 255 tasks done in ~30 min on one
  B200).

**Traces (r4).** Saved tokens per trajectory: mean 49.0k, median 37.4k, p90 110k (262 usable); median 15 assistant
turns, 6.5 agent-minutes. Super on TB2.1 (eval campaign 20261026): mean 51.7k, median 33.9k, p90 128k, so the tasks
draw TB2.1-sized traces; earlier synthetic sets are much shorter (CalibForge 24k, RTS 18k, TMax 11k mean).

- **Trace judge (`pipeline/traces.py`) is lenient on outcome.** Calibrated on verifier-scored Super traces
  (`traces/calibration/`, 60 per campaign, no final state): it calls ~70% of failed attempts "solved" (CalibForge
  21/28, TMax 10/14, RTS 36/53) while catching ~94% of real solves. Its 74% "solved" on r4 is therefore not a solve
  rate, and outcome is not used to filter training data. Its task-defect reports read as real and are what it is
  used for.
- **Task defects found in traces (12 of 205 first judged):** 6 are installs that cannot work offline (PostgreSQL,
  nginx, gunicorn/flask) although the prompts promise a package mirror; the rest are logic contradictions the static
  judge missed (a feature already implemented, an FIFO example contradicting its rule, an efficiency bound that
  contradicts the required method, EXDEV on overlayfs renames).
- **Proxy variables leaked into offline sandboxes:** harbor_ns set the cluster proxy in every sandbox; 11/206 agents
  found it and spent turns trying it.

**Changes before r5:** `synth.package_proxy` gives offline sandboxes the Ubuntu, PyPI and npm registries (and
nothing else) at 127.0.0.1:3128, relayed like the model endpoint, logging every request (tested in an offline
harbor_ns sandbox: pip, apt and npm work, GitHub and direct egress fail); the sandbox's proxy variables point at it;
image builds time out after 10 minutes.

## r3 (2026-10-30) — audit gate passed

100 specs (seed 3), same code and prompts as r2 (ddb422d), concurrency 128 (`rounds/r3/REPORT.md`).

| Metric | r3 | Bar |
| --- | --- | --- |
| Environment-valid share of specs | 96/100 (63 on the first try) | ≥ 85% |
| Judge pass share of judged valid tasks | 84/96 (88%) | ≥ 75% |
| Random judge-passed tasks read and acceptable | 20/20 | ≥ 18/20 |
| Near-duplicate pairs | 0 (nearest-neighbour cosine median 0.11, max 0.24) | ≤ 2% |
| Decon-flagged tasks | 0 (max cosine to an eval task 0.21) | ≤ 2% |

- **r2 and r3 meet the bar with unchanged prompts: the audit gate is passed.** The next stage (a small Super
  trajectory pilot) needs the user's approval.
- **Speed:** 100 tasks in 15 min at concurrency 128 (r2: 58 min at 64). Replies averaged 35.7k completion tokens
  (p90 60.4k, max 100.7k).
- **Format failures (3)** were all a missing `</task_json>` after a long inline `defect_check`; the repair message
  ("missing <task_json> block") misled the generator into repeating it. Fixed in the parser after the round
  (an unclosed block ends at the next block); all 9 of those replies parse with the fix.
- Two first attempts had setups that broke `/bin/sh`; the check caught it and one repair fixed both.
- **Judge scores:** clarity 4.38, consistency 4.40, solvability 4.69, difficulty 4.26 (estimated/target minutes
  median 1.00, p90 1.00), realism 4.55, originality 4.04.
- **Reading:** 20/20 good (ETL contract repair with ISO weeks in local time, Y4M reader after a firmware change,
  Scheme G-code maze decoder with arc rasterization, a packer for a vendor format written against its unpacker,
  OME-TIFF + MP4 migration of a time-lapse, model-blob recovery in a C++ classifier, a Colebrook hydraulic network
  solver, a Supervisor-managed relay deploy from an HTML runbook, ...). Recurring pattern among no-materials ML
  specs: "no data yet, synthesize it and train"; acceptable, worth watching for diversity at scale.

**Applied after the gate (not yet audited):** the judge is told that `/tasksynth` is deleted after setup (r2's one
false positive); the generator is told never to leave expected outputs or answer keys where the agent can read
them (r2's real `golden.json` leak); the parser fix above.

## r2 (2026-10-30)

100 specs (seed 2), DeepSeek-V4.1-Flash-1017 generating and judging on our own server without an output cap (job
17797911, cpu308), base image v2, code and prompts at ddb422d (`rounds/r2/REPORT.md`).

| Metric | r2 | Bar |
| --- | --- | --- |
| Environment-valid share of specs | 99/100 (76 on the first try) | ≥ 85% |
| Judge pass share of judged valid tasks | 84/99 (85%) | ≥ 75% |
| Random judge-passed tasks read and acceptable | 20/20 | ≥ 18/20 |
| Near-duplicate pairs | 0 (nearest-neighbour cosine median 0.11, max 0.41) | ≤ 2% |
| Decon-flagged tasks | 0 (max cosine to an eval task 0.29) | ≤ 2% |

- **Without the cap** replies average 41.8k completion tokens (p90 64.9k, max 100.4k); none truncated. First-try
  validity went from 26 to 76. First-attempt failures left are image builds (18), mostly generator bugs fixed by
  one repair turn.
- **Judge scores rose** (clarity 4.24, consistency 4.28, solvability 4.62, difficulty 4.25, realism 4.48,
  originality 4.07). Estimated/target minutes: median 1.00, p10 0.83, p90 1.17.
- **Judge fails (15) are real defects** except one: a mate-in-2 the author's check missed, a scheduler trace with a
  task on two CPUs at once, an encrypted bundle whose `-iter` the docs omit, a package the setup never installed, a
  `golden.json` holding the expected responses where the agent can read it. The false positive (s2-00078,
  `leaks_solution`) blamed a readable `/tasksynth/gen_data.py`, but `/tasksynth` is deleted after setup: the judge
  prompt never says so.
- **Reading:** 20/20 good (C++ dedup indexer to make 5× faster byte-for-byte, mail-sealing port against a Haskell
  reference, multi-tenant Flask leak with a response cache, a stripped binary's container format to reverse and
  re-encode, DNS service discovery with health checks, DST edge-case audit, nginx + Supervisor HTTPS game service,
  SFTP chroot, Pick's-theorem certificates from a PPM, ...).
- **Weak slices:** security-attack-defend 2/5 judge pass; `leaks_solution` 3 (one false positive).
- **Throughput at concurrency 64:** 100 tasks in 58 min (~8.5M generated tokens incl. judge, ~2.4k tok/s); far
  below the server's capacity, which matters only at scale.

**Next:** r3 with r2's code and prompts unchanged (the bar asks for two consecutive rounds). Pending after the gate:
tell the judge that `/tasksynth` is deleted after setup; tell the generator never to leave expected outputs where
the agent can read them.

## r1 (2026-10-30)

100 specs (seed 1, buckets none + best-generated), DeepSeek-V4.1-Flash-1017 generating and judging on the
decontam-judge server (job 17796961, cpu404), whose output cap is 32,768 tokens. `rounds/r1/REPORT.md`.

| Metric | r1 | Bar |
| --- | --- | --- |
| Environment-valid share of specs | 93/100 | ≥ 85% |
| Judge pass share of judged valid tasks | 78/93 (84%) after rejudging the 18 unverdicted tasks uncapped (14 pass, 4 fail) | ≥ 75% |
| Random judge-passed tasks read and acceptable | 19/20 | ≥ 18/20 |
| Near-duplicate pairs | 0 (nearest-neighbour cosine median 0.11, max 0.46) | ≤ 2% |
| Decon-flagged tasks | 0 (max cosine to an eval task 0.19) | ≤ 2% |

- **The 32k cap shaped the round.** Median reply 23k tokens; 76 of the replies were cut at 32,768, including 71
  first attempts, so only 26 tasks were valid on the first try. Repair turns rescued most of them. The 18 tasks
  without a verdict are judge replies that ran out of tokens before their JSON; they need a rejudge without the cap.
- **Judge vs reading.** The 11 real judge fails all have real defects: a data generator that encodes fullwidth
  text with the wrong offset, an acceptance run that cannot exist in the generated data, a binary built to a
  different path than the instruction names, a requirement that is unsatisfiable for all seeds, `/usr/bin/time`
  missing from the image, sketch maths that contradict the instruction. Of 20 judge-passed tasks read, 19 are
  good training environments (signed APT repo with conffile-preserving upgrade, SQLite v2→v3 migration with a
  running service, ledger journal recovery against a compiled exporter, oscilloscope container reverse
  engineering, int8 checkpoint conversion, a C bytecode checker to debug, ...). The borderline one (s1-00062)
  asks the agent to reproduce an exact random process described in prose; the judge failed another task
  (s1-00055) for the same pattern.
- **Environment failures (7)** are bugs in the generator's own setup scripts (`python` not on PATH, a KeyError
  in a data generator, a setup self-test failing), mostly after a truncated first attempt used up a repair turn.
- **Judge scores:** clarity 4.17, consistency 4.37, solvability 4.59, difficulty 4.04 (estimated/target minutes
  median 1.00), realism 4.25, originality 3.88 (lowest).

**Changes before r2:** our own DeepSeek server without an output cap; rejudge r1's 18 unverdicted tasks; base image
v2 adds standard tools tasks reached for (`python` → python3, `time`, `xxd`, `hexdump`); prompt: tasks without
materials must not hinge on reproducing an exact random process; the judge's tree snapshot skips system
directories (`/usr/local/bin`, `/var/lib`, `/run` noise).

## r0-capped16k (aborted, 2026-10-30)

DeepSeek-V4.1-Flash-1017 generator and judge, 25 of 100 specs finished before the stop. State kept in
`rounds/r0-capped16k/`.

- **Why aborted:** the server's default output length (`--override-generation-config max_new_tokens 16384`, the
  production campaigns' value) is a hard per-request cap in vLLM. 47 of 48 format failures were replies cut at exactly
  16,384 tokens (max reasoning effort), and one judge reply was cut too.
- **What the 9 valid tasks showed:** detailed, realistic tasks (a TCP load balancer with health checks, a mail-filter
  DSL linter over 36 generated policies, SQLite WAL salvage). The judge's issues were specific and correct, including
  a task whose instruction says three backends "already run" with no services script (consistency 1).
- **Changes:** output cap 65,536 for generator and judge (server restarted; compile caches seeded from the campaigns'
  snapshot, which the new cap does not affect); truncated replies get their own repair message; instruction paths
  whose top-level directory does not exist (HTTP routes, placeholders) are warnings instead of repair errors; prompt
  rule that anything the instruction says exists or runs must be built by files, setup or services; "no materials"
  specs say there are no running services either.

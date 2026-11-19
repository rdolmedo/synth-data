# SciCode-Verified: our patch list (#184)

Verified v2 is the base (`verified-v2`). Each version below is built by `python -m aa_evals.scicode.patches.build --version vN` into `/data/aai/aa_evals/data/scicode/verified_aa_vN/`. The payloads live in `/data/aai/aa_evals/data/scicode/patches/vN/`: new tests, reference implementations, mutants, valid alternatives, text-check solvers, and each patch's full rationale and evidence. Their sha256 is below and in the dataset manifest.

**Eval data stays out of git.** This file holds only redacted public text, rendered by `python -m aa_evals.scicode.patches.render` from `patches/public/vN.json`: the defect class and the kind of fix. It never quotes target values, test parameters, formulas, function names or the conventions a step needs. For those, read the payload file named in each row.

**Validation rule.** A `tests` patch builds only if, in both grading stacks (numpy 1.26 and numpy 2.2):
- the reference passes;
- every listed wrong-physics mutant fails;
- every listed valid alternative passes.

A `text` patch is checked by an implementation written from the patched prompt and header alone (no background, the setting we run), which must pass the step's tests (`textcheck/`).

**Policy.**
- Fix wrong or over-lenient tests and targets.
- Undo spec edits that hand over the implementation, but keep every convention the frozen target needs, stated scientifically.
- Because we run **without background**, conventions that Verified added only to `step_background` are moved into the prompt.

**Selection rule (from v2 on).** A model failing a step is not evidence that the step is broken; it may just be hard, and a pass can hide a defect too. A patch needs independent evidence of a defect: a wrong or irreproducible target; a test that rejects a valid reading of the stated spec, or a spec that is underdetermined while the tests pick one option; an over-strict or over-lenient test; a contradiction; or a leak. A patch never makes a problem easier (no hints, no loosening of sound tolerances, no text that gives away the approach). When the evidence is unclear, the step is documented and left alone.

**How v2 was found.**
- The 36 first-failing ("chain-head") steps that Qwen3.8 Medium never passed in Phase B (5 repeats, no background) were each investigated, using executed evidence, the with-background original run and a with-background diagnostic run.
- Every step that Qwen passes in at least 4 of 5 repeats (158 steps) got generic wrong-answer probes. Text Verified added was scanned for leaks.
- 20 new patches survived, out of about 30 proposals.

**Investigated in v2 and left alone** (the reasons are in the payload's investigation notes):
- **Model errors:** 8.1, 11.2, 11.4, 12.3, 13.9, 13.13, 25.2, 26.2, 28.2, 32.1, 37.2, 42.3, 43.3, 54.1, 56.4, 57.2, 57.5, 58.4, 69.4, 73.2, 73.8. In each, an implementation written from the stated spec passes.
- **Dropped after review:** 11.12, 21.1 and 56.3. The failing alternative was not a valid reading of the stated prompt.
- **67.6 (our v1 patch):** re-checked. It rejects no valid implementation, and no tolerance could accept the failing method while still rejecting the wrong physics.
- **Probe survivors judged sound:** 18 steps whose tolerance accepts only physically equivalent answers, or whose test is statistical or qualitative.

**Open items.**
- 67.6: one shortcut implementation still passes, as it did in the original.
- 67.4: its tolerance stays as loose as in Verified.
- 55.4: one test case was dropped for grading runtime.
- 60.5: after v2 the slowest correct implementation still runs close to the grading limit (we keep the official 1800 s).
- Problem 28: its own dependency block does not import in the numpy-2.2 stack, so it is graded in the numpy-1.26 stack only (the OR rule), as in Verified. This is documented, not changed, because every 28.x prompt shows that block.
- 67.2: its own tests do not pin the complex-frequency case that later steps use. That case is pinned downstream.
- Dangling step references in 62.3, 62.6, 69.6, 69.7 and 69.8 are left alone (no grading effect).

**Final CPU audit (v3).** The remaining first-failing steps and selected cascades were reviewed against actual v2 artifacts. The 58 lower-pass steps and three newly higher-pass steps received additional probes; dedicated domain checks cover structured outputs and earlier survivors. The remaining prompt additions were read against original SciCode. Nine additions address independently demonstrated underdetermination, avoidable solution material, omitted interface/initial-condition checks, and physically invalid outputs accepted by statistical checks. No sound tolerance was loosened. Full scope, reproducibility and honest limits: [AUDIT.md](AUDIT.md).

Unchanged inherited targets are preserved exactly. Four target generators now read the pinned frozen v2 overlay to avoid incidental floating-point drift; their reference-generation code remains in the payload. This changes neither those steps' tests nor their target digests.

## v3

Base `verified-v2`; 61 patches; problems md5 `cea6b77b1760a0c6780d67fbabde5867`; overlay h5 md5 `286b3d95c5e939b0296beea375d02b52`; overlay steps 11.5, 12.2, 28.1, 31.3, 35.2, 35.3, 48.4, 50.1, 55.4, 60.3, 60.5, 62.5, 63.1, 63.2, 63.4, 64.6, 67.6, 68.3, 72.6, 76.2, 77.12, 80.7; excluded steps none.

Payload (eval data, not in git): `/data/aai/aa_evals/data/scicode/patches/v3/`, tree sha256 `405d98d4bef1117c76695b44851cb454a09e63c77d573d467d8769c116bd8766`.

| file | sha256 |
|---|---|
| `patches_a_rng.json` | `07d9c8266c2fc1143ccef27c9a17041a7efbb8c4c1e3ee2a7707de2cf3d402df` |
| `patches_b_mc.json` | `2f26dbc722c757c857c4d2d2e0aea43c3321bff4da607c3e6d771bfce7ff95f6` |
| `patches_c_phys.json` | `bb808e2a296ce8651dfc520a335b77ce9f7d88d3b16af7bcf34827f92d20113b` |
| `patches_d_text.json` | `0fb37e50d6ebf4d85f8c406453b67f5ca3bf46ef2827f21781c371834b6fcbc8` |
| `patches_e_followup.json` | `97b5ae7affb85f52602f677de4bac6880abba70e0cd27ecb6bf49fe8a0159ddf` |
| `patches_f_b1.json` | `66d010e24da28725dcdd108300750ee5a9873bb3e04defe41c39695707cdd67a` |
| `patches_g_b2.json` | `f356f2435f57f973083b1ac1851fcc06d3a05e140838e212a850b442fa1e549d` |
| `patches_h_b3.json` | `628cfe84c1b7cea91586b7ace8a1d15e0c68e36e02ca428a3e595b9fbb36f009` |
| `patches_i_b4.json` | `0a57c7ddf94772622171d49db46b18ff2219598be476d759abd3b2ddd4e4b623` |
| `patches_j_pass.json` | `ac61a88f20c6d9e60e563946d4785410fdad6442d2dbbaee33741749814e414e` |
| `patches_k_final.json` | `8f2a7ead8abbaafa88232b75e8cf2b7e0139715df68cfdf89bb5ac7b0e6f5e4d` |

| id | step | kind | change | validation | full rationale and evidence |
|---|---|---|---|---|---|
| `v1-72.6-tests` | 72.6 | tests | Verified left the step with no test cases: add tests that do not depend on the random stream | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_a_rng.json` |
| `v1-50.1-tests` | 50.1 | tests | Replace an over-lenient statistical test with a deterministic limit check and exact-enumeration statistical checks | ref passes; 8/8 mutants rejected; 5/5 alternatives accepted | `patches_b_mc.json` |
| `v1-60.3-tests` | 60.3 | tests | Replace a test that cannot see the temperature convention with exact-valued statistical tests at two temperatures | ref passes; 9/9 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-62.5-tests` | 62.5 | tests | Add truncating cases checked by basis-invariant quantities, so a skipped or wrong basis transform fails | ref passes; 8/8 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-67.6-tests` | 67.6 | tests | Restore a tight numeric-vs-analytic consistency check, and add one more case | ref passes; 5/5 mutants rejected; 6/6 alternatives accepted | `patches_c_phys.json` |
| `v1-76.4-text` | 76.4 | text | Make the window score use the same probability convention as the patched 76.2 | text check (`textcheck/`) | `patches_c_phys.json` |
| `v1-76.2-tests` | 76.2 | tests | Verified's own test cases and targets, registered so the build validates the patched convention | ref passes; 6/6 mutants rejected; 3/3 alternatives accepted | `patches_c_phys.json` |
| `v1-55.4-tests` | 55.4 | tests | Restore the value checks Verified dropped as 'chaotic', with regenerated targets | ref passes; 6/6 mutants rejected; 5/5 alternatives accepted | `patches_c_phys.json` |
| `v1-72.9-text` | 72.9 | text | State the two conventions the target needs scientifically instead of dictating code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.3-text` | 55.3 | text | Keep the load-bearing binning and peak conventions, drop the transcribed code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-66.1-text` | 66.1 | text | Describe the lattice convention geometrically instead of listing the vectors | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-71.2-text` | 71.2 | text | State the operand-order convention without naming the library call | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-37.3-text` | 37.3 | text | Drop a docstring sentence that states a test's expected outcome | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-73.1-text` | 73.1 | text | Remove a leftover authoring marker | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-69.3-text` | 69.3 | text | Fill a dangling step reference | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-68.6-text` | 68.6 | text | State the conventions the target needs instead of the closed-form result the step must compute | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.2-text` | 55.2 | text | State the layout and normalization conventions mathematically instead of as library one-liners | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-45.3-text` | 45.3 | text | Keep the stencil convention, drop the solved update that Verified added | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-79.3-text` | 79.3 | text | Keep the scheme choice, let the solver derive its coefficients | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.5-text` | 80.5 | text | Replace the literal loop body with the physical statement | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-32.2-text` | 32.2 | text | No background: move the conventions the target needs into the prompt | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-22.2-text` | 22.2 | text | No background: state the convention that Verified added only to the background | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.1-text` | 55.1 | text | No background: state the numerical scheme the frozen target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-60.2-text` | 60.2 | text | No background: state the cutoff convention (Verified fixed it only in the background) | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.3-text` | 63.3 | text | No background: state the discretization scheme whose coefficients the target holds | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-74.1-text` | 74.1 | text | No background: state the sign convention and the square-case convention the target needs | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-76.2-text` | 76.2 | text | No background: state the convention the regenerated target encodes | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.3-text` | 80.3 | text | No background: replace an undefined term with the definition the target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.1-tests` | 63.1 | tests | Fix test calls that pass two bounds in the wrong order; regenerate the targets | ref passes; 2/2 mutants rejected; 0/0 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-tests` | 63.2 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 4/4 mutants rejected; 1/1 alternatives accepted | `patches_e_followup.json` |
| `v1-63.4-tests` | 63.4 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-rows-text` | 63.2 | text | Restore the physical labels of the boundary rows, which are correct again after the argument-order fix | text check (`textcheck/`) | `patches_e_followup.json` |
| `v2-11.5-text` | 11.5 | text | Underdetermined spec: the meaning of a model parameter was stated only in the background | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-11.5-tests` | 11.5 | tests | Over-strict test: the step compared one ordered representation of an object that is defined only up to equivalence; the tests now compare the object itself | ref passes; 5/5 mutants rejected; 3/3 alternatives accepted | `patches_f_b1.json` |
| `v2-13.8-text` | 13.8 | text | Underdetermined spec: several standard boundary conditions fit the prompt and the tests accept only one | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-22.1-text` | 22.1 | text | Underdetermined spec: the prompt gave only a magnitude where the target depends on a direction | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-22.3-text` | 22.3 | text | Underdetermined spec that contradicts an earlier step: define the requested quantity and its direction convention, which only the background fixed | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-33.1-text` | 33.1 | text | Underdetermined spec whose tests pick one of several valid conventions: state the conventions the target needs, without the matrix the step asks for | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-39.1-text` | 39.1 | text | Underdetermined spec whose tests pick one of several valid conventions: state the convention the target needs | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-31.3-tests` | 31.3 | tests | Over-strict tests whose outcome depended on the random initialization: score against ground truth, replace an ill-posed case, keep the threshold | ref passes; 6/6 mutants rejected; 7/7 alternatives accepted | `patches_g_b2.json` |
| `v2-43.1-text` | 43.1 | text | Underdetermined spec: an input quantity has several standard definitions and the tests encode one; state the definition | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-43.1-header-text` | 43.1 | text | Interface mismatch: the tests call the function in a form the documented interface did not allow; correct the documented argument shapes | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-48.4-text` | 48.4 | text | Underdetermined convention: the theorem the step relies on has several normalizations and the tests encode one; state the normalization only | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-48.4-tests` | 48.4 | tests | Inconsistent tolerance: a downstream check was stricter than the upstream steps whose accepted alternative it consumes; match the upstream tolerance, targets unchanged | ref passes; 5/5 mutants rejected; 2/2 alternatives accepted | `patches_h_b3.json` |
| `v2-60.5-tests` | 60.5 | tests | Infeasible runtime: shorten the run lengths of a test that checks an exact identity; targets and the statistical test unchanged | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_i_b4.json` |
| `v2-80.7-tests` | 80.7 | tests | Test rejects a valid answer: the statistical check also accepts a second, equally valid discretization | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_i_b4.json` |
| `v2-79.2-text` | 79.2 | text | Background-only convention: name the integration scheme the targets encode | text check (`textcheck/`) | `patches_i_b4.json` |
| `v2-69.6-text` | 69.6 | text | Background-only convention: state the experimental geometry and normalization the targets assume | text check (`textcheck/`) | `patches_i_b4.json` |
| `v2-28.1-tests` | 28.1 | tests | Over-lenient test: only integrated sums were checked, so a mis-positioned output passed; add checks on the output's position and peak value | ref passes; 3/3 mutants rejected; 3/3 alternatives accepted | `patches_j_pass.json` |
| `v2-28.1-text` | 28.1 | text | Leak: Verified added the explicit formula the step should derive; keep the method choice, drop the formula | text check (`textcheck/`) | `patches_j_pass.json` |
| `v2-68.3-tests` | 68.3 | tests | Over-lenient test: one method of the step's class was never exercised; add a direct check of it, mirroring the fix Verified made on the twin step | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_j_pass.json` |
| `v2-63.2-text` | 63.2 | text | Leak: the prompt dictated the whole step as literal assignments; state the conventions instead | text check (`textcheck/`) | `patches_j_pass.json` |
| `v3-79.4-step_description_prompt` | 79.4 | text | Specify the numerical method required by the frozen target. | text check (`textcheck/`) | `patches_k_final.json` |
| `v3-43.3-step_description_prompt` | 43.3 | text | Remove nonessential solver settings and initialization; retain the physical branch and output convention. | text check (`textcheck/`) | `patches_k_final.json` |
| `v3-43.3-function_header` | 43.3 | text | Describe the returned quantity and sampling without literal library calls. | text check (`textcheck/`) | `patches_k_final.json` |
| `v3-58.5-step_description_prompt` | 58.5 | text | Keep the physical normalization convention and remove the solved output formula. | text check (`textcheck/`) | `patches_k_final.json` |
| `v3-12.2-tests` | 12.2 | tests | Add exact initial-condition checks missed by normalized comparisons. | ref passes; 3/3 mutants rejected; 2/2 alternatives accepted | `patches_k_final.json` |
| `v3-35.2-tests` | 35.2 | tests | Enforce the output order already required by the prompt. | ref passes; 2/2 mutants rejected; 2/2 alternatives accepted | `patches_k_final.json` |
| `v3-35.3-tests` | 35.3 | tests | Enforce the output order already required by the prompt. | ref passes; 2/2 mutants rejected; 2/2 alternatives accepted | `patches_k_final.json` |
| `v3-64.6-tests` | 64.6 | tests | Reject unphysical outputs admitted by a signed error calculation and check the remaining documented outputs. | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_k_final.json` |
| `v3-77.12-tests` | 77.12 | tests | Add exact mechanical limits missed by a statistical convergence test. | ref passes; 5/5 mutants rejected; 2/2 alternatives accepted | `patches_k_final.json` |

### Why

- **`v1-72.6-tests`.** Verified removed the original seeded tests (an RNG-protocol defect) and left zero tests, so a do-nothing implementation passes. The new tests check limits with deterministic outcomes, plus an exact-enumeration statistical check that any valid random stream passes.
- **`v1-50.1-tests`.** Verified's single test compared a Monte Carlo average with another Monte Carlo estimate, inside a window several times too wide, so wrong-temperature and quench implementations pass. The new checks compare against exact Boltzmann averages of a small instance, within a few standard errors.
- **`v1-60.3-tests`.** Verified's one test at a single temperature cannot tell the correct temperature dependence from a common inverted one, and it also accepts a potential convention that the previous step excludes. The new tests use a configuration whose expectation is known exactly, at two temperatures.
- **`v1-62.5-tests`.** Verified rightly switched to basis-invariant checks, but its only case does not truncate, so any orthogonal transform passes, including the identity. The new cases truncate at gaps in the spectrum, where the kept subspace is unique.
- **`v1-67.6-tests`.** Verified loosened this check by several orders of magnitude, which lets a known wrong-physics implementation pass (it failed the original). The audit measured every valid form far inside the restored tolerance, and the wrong one far outside it.
- **`v1-76.4-text`.** This keeps the 76.4 tests consistent with the convention stated by v1-76.2-text. Without it, one natural reading of the patched 76.2 fails a 76.4 test. Tests unchanged.
- **`v1-76.2-tests`.** Verified's 76.2 targets encode a convention its text does not state, which contradicts 76.1 (the #184 contradiction). The targets are kept bit-for-bit and recomputed independently; v1-76.2-text states the convention. This patch makes the build check that the literal reading fails and that other valid implementations pass.
- **`v1-55.4-tests`.** The audit shows the dynamics are not chaotic. The original targets carry a defect of the gold itself. Targets are regenerated from a reference that 55.1's own tests already pin, and every valid loop form passes.
- **`v1-72.9-text`.** Verified pinned two load-bearing conventions correctly, but wrote them as the literal implementation. The new wording keeps both and drops the code.
- **`v1-55.3-text`.** The frozen targets need specific binning and peak-window conventions, so these stay. Verified also spelled out library calls and an explicit formula, which hand over the implementation.
- **`v1-66.1-text`.** The tests need one specific choice among several valid parameterisations, so the choice must be stated, but Verified gave the full construction as coordinates. The new wording fixes the same choice and leaves the derivation to the solver.
- **`v1-71.2-text`.** Only the order is a convention; the library call names the implementation.
- **`v1-37.3-text`.** The sentence gives away the asserted result and the sign of every target. The sign convention is already fixed elsewhere in the step.
- **`v1-73.1-text`.** Cosmetic. Verified removed the same marker elsewhere (69-d1). No grading effect.
- **`v1-69.3-text`.** Cosmetic. Verified made the identical fix in the sibling problem 67. No grading effect.
- **`v1-68.6-text`.** Computing this quantity is the step's whole job, and Verified wrote out the result. The new wording keeps the conventions it needs and the definitions it follows from.
- **`v1-55.2-text`.** The conventions are load-bearing and stay; the library calls hand over the implementation.
- **`v1-45.3-text`.** The solved update follows uniquely from the stated convention and is the step's job.
- **`v1-79.3-text`.** Several schemes of this order exist, so naming the scheme is load-bearing. Its coefficients follow uniquely and are part of what the step asks the solver to know.
- **`v1-80.5-text`.** The pair sum and the displacement direction are the conventions. The loop body is code.
- **`v1-32.2-text`.** Without background, the prompt does not fix the constants and conventions the frozen target needs. Several were only in the background, some added there by Verified. They are stated physically, and the derivable quantities are left to the solver.
- **`v1-22.2-text`.** The target depends on a convention a solver cannot infer from the prompt. A different common convention changes signs and fails.
- **`v1-55.1-text`.** The prompt names only the method family, and variants of it differ from the target at first order in the step size. The scheme is stated; the operators are left to derive.
- **`v1-60.2-text`.** The prompt and header never say which convention applies, and the other one fails a test.
- **`v1-63.3-text`.** Other standard schemes give a different matrix. The coefficients are left for the solver to derive.
- **`v1-74.1-text`.** The result is unique only up to signs. Verified added the deciding convention to the background only.
- **`v1-76.2-text`.** Verified's regenerated 76.2 targets assume a convention its text does not state, which contradicts the literal reading of 76.1's output (the #184 contradiction). Stating it keeps Verified's targets and changes only text.
- **`v1-80.3-text`.** Verified corrected the definition only in the background. Without background, the prompt's term has two wrong readings, and one test separates them.
- **`v1-63.1-tests`.** The 63.1/63.2/63.4 test cases pass two arguments swapped with respect to the signature, which reverses the grid and turns the boundary conditions into wrong finance. The later steps and the main-problem tests already use signature order. The regenerating reference reproduces every Verified target of 63.1-63.6 when called as Verified does.
- **`v1-63.2-tests`.** See v1-63.1-tests.
- **`v1-63.4-tests`.** See v1-63.1-tests.
- **`v1-63.2-rows-text`.** Verified had removed them because its tests reversed the grid.
- **`v2-11.5-text`.** Several standard readings of the parameter give different channels at the tested points, and only the background fixed which one the targets use. The prompt now states it.
- **`v2-11.5-tests`.** Equivalent representations of the same channel (another order, another valid set) failed. The new check accepts exactly the equivalent ones, and every wrong channel still fails.
- **`v2-13.8-text`.** The background itself says the condition can be implemented in different ways, and the standard alternatives fail. The prompt now states which physical assumption applies, not how to implement it.
- **`v2-22.1-text`.** The opposite direction negates some targets exactly, and only the background's starting value fixed it. The prompt now fixes the direction.
- **`v2-22.3-text`.** The requested coefficients were never defined. Taking the direction convention of the earlier step reverses the sign of every target. The prompt now defines the quantity; the method is not given.
- **`v2-33.1-text`.** The tests compare a full matrix, which depends on conventions (orientation, sublattice order, gauge, sign) set only in the background. Several equally valid choices fail.
- **`v2-39.1-text`.** Several conventions give matrices with the same physics, and only some pass. One test cannot tell them apart.
- **`v2-31.3-tests`.** The prescribed algorithm passed one case only for some random starts, and another case was scored against an earlier estimate rather than the true sources. The threshold is unchanged; the false-fail rate is measured in the payload.
- **`v2-43.1-text`.** A complete implementation using the other common definition fails. The definition appeared only in later material.
- **`v2-43.1-header-text`.** An implementation that follows the documented interface with correct physics crashes on the tests' calls.
- **`v2-48.4-text`.** Implementations using other standard normalizations fail. Only the normalization is stated; the relation the step derives is not.
- **`v2-48.4-tests`.** Verified deliberately accepts an alternative in the earlier steps of this problem, and this step consumes their outputs. A fully valid implementation using that alternative failed here only on tolerance. The new tolerance equals the upstream one, and the wrong-physics mutants still fail.
- **`v2-60.5-tests`.** An implementation that follows the energy function the problem prescribes timed out on the official limit, mostly in a test whose expected value holds exactly at any run length. The statistical test is unchanged (shortening it would be unsafe); wrong-physics mutants still fail.
- **`v2-80.7-tests`.** A standard exact discretization of the collision process biases the measured rate outside the window, although its physics is correct. The width is unchanged, and the wrong discretizations and wrong temperatures still fail.
- **`v2-79.2-text`.** The targets are bit-exact to one splitting scheme, given only in the background. An exact integration and another symmetric splitting both fail. The prompt names the scheme, not its update equations.
- **`v2-69.6-text`.** Only one geometry and normalization reproduces the targets, and other valid readings fail. The prompt states the geometry, not the intensity formula.
- **`v2-28.1-tests`.** A wrong-physics result passes the original checks. Valid alternative methods still pass the new ones. Validated in one grading stack only: the problem's own dependency block cannot import in the other (documented, not changed).
- **`v2-28.1-text`.** The targets do not depend on the formula (an exact alternative also passes), so it fixed no convention and only handed over the physics.
- **`v2-68.3-tests`.** Wrong and zero implementations of that method passed, and later steps use it.
- **`v2-63.2-text`.** A verbatim copy of the given lines passed with no domain knowledge. The new text keeps every convention the targets need (grid layout, which boundary goes where, precedence).
- **`v3-79.4-step_description_prompt`.** A valid method satisfying the prior prompt fails the frozen trajectory, while the required method passes independently. The correction names the already-defined method without a recipe, coefficients or outputs.
- **`v3-43.3-step_description_prompt`.** Verified supplied a solver recipe that is not needed to reproduce the targets. A different solver initialization and accuracy setting passes unchanged tests. The required physical branch and sampling remain explicit.
- **`v3-43.3-function_header`.** The output interface needs the physical quantity and sample positions; a particular library accessor is not required.
- **`v3-58.5-step_description_prompt`.** The boundary condition determines the requested output. An independent calculation using that condition passes unchanged targets; the prompt leaves the derivation to the solver.
- **`v3-12.2-tests`.** Normalizing every output before comparison hid incorrect solution amplitudes despite specified initial conditions. The original checks and targets remain, with exact limit cases added.
- **`v3-35.2-tests`.** Sorting inside the grader erased a documented interface requirement. Existing value checks, targets and tolerances remain, with an order assertion added.
- **`v3-35.3-tests`.** Sorting inside the grader erased a documented interface requirement. Existing value checks, targets and tolerances remain, with an order assertion added.
- **`v3-64.6-tests`.** An error denominator computed from an unchecked returned quantity lets impossible signed outputs pass. Keep the statistical criterion and target, require physical validity, and check the other returned quantities in the exact limit already exercised.
- **`v3-77.12-tests`.** An implementation that destroys the returned physical state still passes the original controlled-convergence check. Keep that check and target, and add small exact limits with control couplings disabled, as already allowed by the prompt. Independent valid formulations pass while unphysical states and incorrect forces fail.


## v2

Base `verified-v2`; 52 patches; problems md5 `b5b08aec0df537fb2e3af8114c2c6923`; overlay h5 md5 `65a0de676583a0fe23a929602e2d4ad2`; overlay steps 11.5, 28.1, 31.3, 48.4, 50.1, 55.4, 60.3, 60.5, 62.5, 63.1, 63.2, 63.4, 67.6, 68.3, 72.6, 76.2, 80.7; excluded steps none.

Payload (eval data, not in git): `/data/aai/aa_evals/data/scicode/patches/v2/`, tree sha256 `528ab137bdb3f3ce0a8f6eb3be914e1615e72b5c8a5ffa6fd0878aeb1e369600`.

| file | sha256 |
|---|---|
| `patches_a_rng.json` | `07d9c8266c2fc1143ccef27c9a17041a7efbb8c4c1e3ee2a7707de2cf3d402df` |
| `patches_b_mc.json` | `2f26dbc722c757c857c4d2d2e0aea43c3321bff4da607c3e6d771bfce7ff95f6` |
| `patches_c_phys.json` | `bb808e2a296ce8651dfc520a335b77ce9f7d88d3b16af7bcf34827f92d20113b` |
| `patches_d_text.json` | `0fb37e50d6ebf4d85f8c406453b67f5ca3bf46ef2827f21781c371834b6fcbc8` |
| `patches_e_followup.json` | `97b5ae7affb85f52602f677de4bac6880abba70e0cd27ecb6bf49fe8a0159ddf` |
| `patches_f_b1.json` | `66d010e24da28725dcdd108300750ee5a9873bb3e04defe41c39695707cdd67a` |
| `patches_g_b2.json` | `f356f2435f57f973083b1ac1851fcc06d3a05e140838e212a850b442fa1e549d` |
| `patches_h_b3.json` | `628cfe84c1b7cea91586b7ace8a1d15e0c68e36e02ca428a3e595b9fbb36f009` |
| `patches_i_b4.json` | `0a57c7ddf94772622171d49db46b18ff2219598be476d759abd3b2ddd4e4b623` |
| `patches_j_pass.json` | `ac61a88f20c6d9e60e563946d4785410fdad6442d2dbbaee33741749814e414e` |

| id | step | kind | change | validation | full rationale and evidence |
|---|---|---|---|---|---|
| `v1-72.6-tests` | 72.6 | tests | Verified left the step with no test cases: add tests that do not depend on the random stream | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_a_rng.json` |
| `v1-50.1-tests` | 50.1 | tests | Replace an over-lenient statistical test with a deterministic limit check and exact-enumeration statistical checks | ref passes; 8/8 mutants rejected; 5/5 alternatives accepted | `patches_b_mc.json` |
| `v1-60.3-tests` | 60.3 | tests | Replace a test that cannot see the temperature convention with exact-valued statistical tests at two temperatures | ref passes; 9/9 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-62.5-tests` | 62.5 | tests | Add truncating cases checked by basis-invariant quantities, so a skipped or wrong basis transform fails | ref passes; 8/8 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-67.6-tests` | 67.6 | tests | Restore a tight numeric-vs-analytic consistency check, and add one more case | ref passes; 5/5 mutants rejected; 6/6 alternatives accepted | `patches_c_phys.json` |
| `v1-76.4-text` | 76.4 | text | Make the window score use the same probability convention as the patched 76.2 | text check (`textcheck/`) | `patches_c_phys.json` |
| `v1-76.2-tests` | 76.2 | tests | Verified's own test cases and targets, registered so the build validates the patched convention | ref passes; 6/6 mutants rejected; 3/3 alternatives accepted | `patches_c_phys.json` |
| `v1-55.4-tests` | 55.4 | tests | Restore the value checks Verified dropped as 'chaotic', with regenerated targets | ref passes; 6/6 mutants rejected; 5/5 alternatives accepted | `patches_c_phys.json` |
| `v1-72.9-text` | 72.9 | text | State the two conventions the target needs scientifically instead of dictating code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.3-text` | 55.3 | text | Keep the load-bearing binning and peak conventions, drop the transcribed code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-66.1-text` | 66.1 | text | Describe the lattice convention geometrically instead of listing the vectors | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-71.2-text` | 71.2 | text | State the operand-order convention without naming the library call | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-37.3-text` | 37.3 | text | Drop a docstring sentence that states a test's expected outcome | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-73.1-text` | 73.1 | text | Remove a leftover authoring marker | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-69.3-text` | 69.3 | text | Fill a dangling step reference | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-68.6-text` | 68.6 | text | State the conventions the target needs instead of the closed-form result the step must compute | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.2-text` | 55.2 | text | State the layout and normalization conventions mathematically instead of as library one-liners | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-45.3-text` | 45.3 | text | Keep the stencil convention, drop the solved update that Verified added | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-79.3-text` | 79.3 | text | Keep the scheme choice, let the solver derive its coefficients | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.5-text` | 80.5 | text | Replace the literal loop body with the physical statement | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-32.2-text` | 32.2 | text | No background: move the conventions the target needs into the prompt | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-22.2-text` | 22.2 | text | No background: state the convention that Verified added only to the background | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.1-text` | 55.1 | text | No background: state the numerical scheme the frozen target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-60.2-text` | 60.2 | text | No background: state the cutoff convention (Verified fixed it only in the background) | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.3-text` | 63.3 | text | No background: state the discretization scheme whose coefficients the target holds | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-74.1-text` | 74.1 | text | No background: state the sign convention and the square-case convention the target needs | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-76.2-text` | 76.2 | text | No background: state the convention the regenerated target encodes | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.3-text` | 80.3 | text | No background: replace an undefined term with the definition the target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.1-tests` | 63.1 | tests | Fix test calls that pass two bounds in the wrong order; regenerate the targets | ref passes; 2/2 mutants rejected; 0/0 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-tests` | 63.2 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 4/4 mutants rejected; 1/1 alternatives accepted | `patches_e_followup.json` |
| `v1-63.4-tests` | 63.4 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-rows-text` | 63.2 | text | Restore the physical labels of the boundary rows, which are correct again after the argument-order fix | text check (`textcheck/`) | `patches_e_followup.json` |
| `v2-11.5-text` | 11.5 | text | Underdetermined spec: the meaning of a model parameter was stated only in the background | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-11.5-tests` | 11.5 | tests | Over-strict test: the step compared one ordered representation of an object that is defined only up to equivalence; the tests now compare the object itself | ref passes; 5/5 mutants rejected; 3/3 alternatives accepted | `patches_f_b1.json` |
| `v2-13.8-text` | 13.8 | text | Underdetermined spec: several standard boundary conditions fit the prompt and the tests accept only one | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-22.1-text` | 22.1 | text | Underdetermined spec: the prompt gave only a magnitude where the target depends on a direction | text check (`textcheck/`) | `patches_f_b1.json` |
| `v2-22.3-text` | 22.3 | text | Underdetermined spec that contradicts an earlier step: define the requested quantity and its direction convention, which only the background fixed | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-33.1-text` | 33.1 | text | Underdetermined spec whose tests pick one of several valid conventions: state the conventions the target needs, without the matrix the step asks for | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-39.1-text` | 39.1 | text | Underdetermined spec whose tests pick one of several valid conventions: state the convention the target needs | text check (`textcheck/`) | `patches_g_b2.json` |
| `v2-31.3-tests` | 31.3 | tests | Over-strict tests whose outcome depended on the random initialization: score against ground truth, replace an ill-posed case, keep the threshold | ref passes; 6/6 mutants rejected; 7/7 alternatives accepted | `patches_g_b2.json` |
| `v2-43.1-text` | 43.1 | text | Underdetermined spec: an input quantity has several standard definitions and the tests encode one; state the definition | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-43.1-header-text` | 43.1 | text | Interface mismatch: the tests call the function in a form the documented interface did not allow; correct the documented argument shapes | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-48.4-text` | 48.4 | text | Underdetermined convention: the theorem the step relies on has several normalizations and the tests encode one; state the normalization only | text check (`textcheck/`) | `patches_h_b3.json` |
| `v2-48.4-tests` | 48.4 | tests | Inconsistent tolerance: a downstream check was stricter than the upstream steps whose accepted alternative it consumes; match the upstream tolerance, targets unchanged | ref passes; 5/5 mutants rejected; 2/2 alternatives accepted | `patches_h_b3.json` |
| `v2-60.5-tests` | 60.5 | tests | Infeasible runtime: shorten the run lengths of a test that checks an exact identity; targets and the statistical test unchanged | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_i_b4.json` |
| `v2-80.7-tests` | 80.7 | tests | Test rejects a valid answer: the statistical check also accepts a second, equally valid discretization | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_i_b4.json` |
| `v2-79.2-text` | 79.2 | text | Background-only convention: name the integration scheme the targets encode | text check (`textcheck/`) | `patches_i_b4.json` |
| `v2-69.6-text` | 69.6 | text | Background-only convention: state the experimental geometry and normalization the targets assume | text check (`textcheck/`) | `patches_i_b4.json` |
| `v2-28.1-tests` | 28.1 | tests | Over-lenient test: only integrated sums were checked, so a mis-positioned output passed; add checks on the output's position and peak value | ref passes; 3/3 mutants rejected; 3/3 alternatives accepted | `patches_j_pass.json` |
| `v2-28.1-text` | 28.1 | text | Leak: Verified added the explicit formula the step should derive; keep the method choice, drop the formula | text check (`textcheck/`) | `patches_j_pass.json` |
| `v2-68.3-tests` | 68.3 | tests | Over-lenient test: one method of the step's class was never exercised; add a direct check of it, mirroring the fix Verified made on the twin step | ref passes; 4/4 mutants rejected; 2/2 alternatives accepted | `patches_j_pass.json` |
| `v2-63.2-text` | 63.2 | text | Leak: the prompt dictated the whole step as literal assignments; state the conventions instead | text check (`textcheck/`) | `patches_j_pass.json` |

### Why

- **`v1-72.6-tests`.** Verified removed the original seeded tests (an RNG-protocol defect) and left zero tests, so a do-nothing implementation passes. The new tests check limits with deterministic outcomes, plus an exact-enumeration statistical check that any valid random stream passes.
- **`v1-50.1-tests`.** Verified's single test compared a Monte Carlo average with another Monte Carlo estimate, inside a window several times too wide, so wrong-temperature and quench implementations pass. The new checks compare against exact Boltzmann averages of a small instance, within a few standard errors.
- **`v1-60.3-tests`.** Verified's one test at a single temperature cannot tell the correct temperature dependence from a common inverted one, and it also accepts a potential convention that the previous step excludes. The new tests use a configuration whose expectation is known exactly, at two temperatures.
- **`v1-62.5-tests`.** Verified rightly switched to basis-invariant checks, but its only case does not truncate, so any orthogonal transform passes, including the identity. The new cases truncate at gaps in the spectrum, where the kept subspace is unique.
- **`v1-67.6-tests`.** Verified loosened this check by several orders of magnitude, which lets a known wrong-physics implementation pass (it failed the original). The audit measured every valid form far inside the restored tolerance, and the wrong one far outside it.
- **`v1-76.4-text`.** This keeps the 76.4 tests consistent with the convention stated by v1-76.2-text. Without it, one natural reading of the patched 76.2 fails a 76.4 test. Tests unchanged.
- **`v1-76.2-tests`.** Verified's 76.2 targets encode a convention its text does not state, which contradicts 76.1 (the #184 contradiction). The targets are kept bit-for-bit and recomputed independently; v1-76.2-text states the convention. This patch makes the build check that the literal reading fails and that other valid implementations pass.
- **`v1-55.4-tests`.** The audit shows the dynamics are not chaotic. The original targets carry a defect of the gold itself. Targets are regenerated from a reference that 55.1's own tests already pin, and every valid loop form passes.
- **`v1-72.9-text`.** Verified pinned two load-bearing conventions correctly, but wrote them as the literal implementation. The new wording keeps both and drops the code.
- **`v1-55.3-text`.** The frozen targets need specific binning and peak-window conventions, so these stay. Verified also spelled out library calls and an explicit formula, which hand over the implementation.
- **`v1-66.1-text`.** The tests need one specific choice among several valid parameterisations, so the choice must be stated, but Verified gave the full construction as coordinates. The new wording fixes the same choice and leaves the derivation to the solver.
- **`v1-71.2-text`.** Only the order is a convention; the library call names the implementation.
- **`v1-37.3-text`.** The sentence gives away the asserted result and the sign of every target. The sign convention is already fixed elsewhere in the step.
- **`v1-73.1-text`.** Cosmetic. Verified removed the same marker elsewhere (69-d1). No grading effect.
- **`v1-69.3-text`.** Cosmetic. Verified made the identical fix in the sibling problem 67. No grading effect.
- **`v1-68.6-text`.** Computing this quantity is the step's whole job, and Verified wrote out the result. The new wording keeps the conventions it needs and the definitions it follows from.
- **`v1-55.2-text`.** The conventions are load-bearing and stay; the library calls hand over the implementation.
- **`v1-45.3-text`.** The solved update follows uniquely from the stated convention and is the step's job.
- **`v1-79.3-text`.** Several schemes of this order exist, so naming the scheme is load-bearing. Its coefficients follow uniquely and are part of what the step asks the solver to know.
- **`v1-80.5-text`.** The pair sum and the displacement direction are the conventions. The loop body is code.
- **`v1-32.2-text`.** Without background, the prompt does not fix the constants and conventions the frozen target needs. Several were only in the background, some added there by Verified. They are stated physically, and the derivable quantities are left to the solver.
- **`v1-22.2-text`.** The target depends on a convention a solver cannot infer from the prompt. A different common convention changes signs and fails.
- **`v1-55.1-text`.** The prompt names only the method family, and variants of it differ from the target at first order in the step size. The scheme is stated; the operators are left to derive.
- **`v1-60.2-text`.** The prompt and header never say which convention applies, and the other one fails a test.
- **`v1-63.3-text`.** Other standard schemes give a different matrix. The coefficients are left for the solver to derive.
- **`v1-74.1-text`.** The result is unique only up to signs. Verified added the deciding convention to the background only.
- **`v1-76.2-text`.** Verified's regenerated 76.2 targets assume a convention its text does not state, which contradicts the literal reading of 76.1's output (the #184 contradiction). Stating it keeps Verified's targets and changes only text.
- **`v1-80.3-text`.** Verified corrected the definition only in the background. Without background, the prompt's term has two wrong readings, and one test separates them.
- **`v1-63.1-tests`.** The 63.1/63.2/63.4 test cases pass two arguments swapped with respect to the signature, which reverses the grid and turns the boundary conditions into wrong finance. The later steps and the main-problem tests already use signature order. The regenerating reference reproduces every Verified target of 63.1-63.6 when called as Verified does.
- **`v1-63.2-tests`.** See v1-63.1-tests.
- **`v1-63.4-tests`.** See v1-63.1-tests.
- **`v1-63.2-rows-text`.** Verified had removed them because its tests reversed the grid.
- **`v2-11.5-text`.** Several standard readings of the parameter give different channels at the tested points, and only the background fixed which one the targets use. The prompt now states it.
- **`v2-11.5-tests`.** Equivalent representations of the same channel (another order, another valid set) failed. The new check accepts exactly the equivalent ones, and every wrong channel still fails.
- **`v2-13.8-text`.** The background itself says the condition can be implemented in different ways, and the standard alternatives fail. The prompt now states which physical assumption applies, not how to implement it.
- **`v2-22.1-text`.** The opposite direction negates some targets exactly, and only the background's starting value fixed it. The prompt now fixes the direction.
- **`v2-22.3-text`.** The requested coefficients were never defined. Taking the direction convention of the earlier step reverses the sign of every target. The prompt now defines the quantity; the method is not given.
- **`v2-33.1-text`.** The tests compare a full matrix, which depends on conventions (orientation, sublattice order, gauge, sign) set only in the background. Several equally valid choices fail.
- **`v2-39.1-text`.** Several conventions give matrices with the same physics, and only some pass. One test cannot tell them apart.
- **`v2-31.3-tests`.** The prescribed algorithm passed one case only for some random starts, and another case was scored against an earlier estimate rather than the true sources. The threshold is unchanged; the false-fail rate is measured in the payload.
- **`v2-43.1-text`.** A complete implementation using the other common definition fails. The definition appeared only in later material.
- **`v2-43.1-header-text`.** An implementation that follows the documented interface with correct physics crashes on the tests' calls.
- **`v2-48.4-text`.** Implementations using other standard normalizations fail. Only the normalization is stated; the relation the step derives is not.
- **`v2-48.4-tests`.** Verified deliberately accepts an alternative in the earlier steps of this problem, and this step consumes their outputs. A fully valid implementation using that alternative failed here only on tolerance. The new tolerance equals the upstream one, and the wrong-physics mutants still fail.
- **`v2-60.5-tests`.** An implementation that follows the energy function the problem prescribes timed out on the official limit, mostly in a test whose expected value holds exactly at any run length. The statistical test is unchanged (shortening it would be unsafe); wrong-physics mutants still fail.
- **`v2-80.7-tests`.** A standard exact discretization of the collision process biases the measured rate outside the window, although its physics is correct. The width is unchanged, and the wrong discretizations and wrong temperatures still fail.
- **`v2-79.2-text`.** The targets are bit-exact to one splitting scheme, given only in the background. An exact integration and another symmetric splitting both fail. The prompt names the scheme, not its update equations.
- **`v2-69.6-text`.** Only one geometry and normalization reproduces the targets, and other valid readings fail. The prompt states the geometry, not the intensity formula.
- **`v2-28.1-tests`.** A wrong-physics result passes the original checks. Valid alternative methods still pass the new ones. Validated in one grading stack only: the problem's own dependency block cannot import in the other (documented, not changed).
- **`v2-28.1-text`.** The targets do not depend on the formula (an exact alternative also passes), so it fixed no convention and only handed over the physics.
- **`v2-68.3-tests`.** Wrong and zero implementations of that method passed, and later steps use it.
- **`v2-63.2-text`.** A verbatim copy of the given lines passed with no domain knowledge. The new text keeps every convention the targets need (grid layout, which boundary goes where, precedence).


## v1 (superseded by v2; kept for the record)

Base `verified-v2`; 32 patches; problems md5 `082c17db6191966f399687581708a921`; overlay h5 md5 `2d48dac1455a971952afc4a49e1325f9`; overlay steps 50.1, 55.4, 60.3, 62.5, 63.1, 63.2, 63.4, 67.6, 72.6, 76.2; excluded steps none.

Payload (eval data, not in git): `/data/aai/aa_evals/data/scicode/patches/v1/`, tree sha256 `8930de9346c5a81912fa95334c32bda3b8852f6347cb1c595c5b6876ae89b8f6`.

| file | sha256 |
|---|---|
| `patches_a_rng.json` | `07d9c8266c2fc1143ccef27c9a17041a7efbb8c4c1e3ee2a7707de2cf3d402df` |
| `patches_b_mc.json` | `2f26dbc722c757c857c4d2d2e0aea43c3321bff4da607c3e6d771bfce7ff95f6` |
| `patches_c_phys.json` | `bb808e2a296ce8651dfc520a335b77ce9f7d88d3b16af7bcf34827f92d20113b` |
| `patches_d_text.json` | `0fb37e50d6ebf4d85f8c406453b67f5ca3bf46ef2827f21781c371834b6fcbc8` |
| `patches_e_followup.json` | `97b5ae7affb85f52602f677de4bac6880abba70e0cd27ecb6bf49fe8a0159ddf` |

| id | step | kind | change | validation | full rationale and evidence |
|---|---|---|---|---|---|
| `v1-72.6-tests` | 72.6 | tests | Verified left the step with no test cases: add tests that do not depend on the random stream | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_a_rng.json` |
| `v1-50.1-tests` | 50.1 | tests | Replace an over-lenient statistical test with a deterministic limit check and exact-enumeration statistical checks | ref passes; 8/8 mutants rejected; 5/5 alternatives accepted | `patches_b_mc.json` |
| `v1-60.3-tests` | 60.3 | tests | Replace a test that cannot see the temperature convention with exact-valued statistical tests at two temperatures | ref passes; 9/9 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-62.5-tests` | 62.5 | tests | Add truncating cases checked by basis-invariant quantities, so a skipped or wrong basis transform fails | ref passes; 8/8 mutants rejected; 4/4 alternatives accepted | `patches_b_mc.json` |
| `v1-67.6-tests` | 67.6 | tests | Restore a tight numeric-vs-analytic consistency check, and add one more case | ref passes; 5/5 mutants rejected; 6/6 alternatives accepted | `patches_c_phys.json` |
| `v1-76.4-text` | 76.4 | text | Make the window score use the same probability convention as the patched 76.2 | text check (`textcheck/`) | `patches_c_phys.json` |
| `v1-76.2-tests` | 76.2 | tests | Verified's own test cases and targets, registered so the build validates the patched convention | ref passes; 6/6 mutants rejected; 3/3 alternatives accepted | `patches_c_phys.json` |
| `v1-55.4-tests` | 55.4 | tests | Restore the value checks Verified dropped as 'chaotic', with regenerated targets | ref passes; 6/6 mutants rejected; 5/5 alternatives accepted | `patches_c_phys.json` |
| `v1-72.9-text` | 72.9 | text | State the two conventions the target needs scientifically instead of dictating code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.3-text` | 55.3 | text | Keep the load-bearing binning and peak conventions, drop the transcribed code | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-66.1-text` | 66.1 | text | Describe the lattice convention geometrically instead of listing the vectors | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-71.2-text` | 71.2 | text | State the operand-order convention without naming the library call | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-37.3-text` | 37.3 | text | Drop a docstring sentence that states a test's expected outcome | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-73.1-text` | 73.1 | text | Remove a leftover authoring marker | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-69.3-text` | 69.3 | text | Fill a dangling step reference | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-68.6-text` | 68.6 | text | State the conventions the target needs instead of the closed-form result the step must compute | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.2-text` | 55.2 | text | State the layout and normalization conventions mathematically instead of as library one-liners | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-45.3-text` | 45.3 | text | Keep the stencil convention, drop the solved update that Verified added | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-79.3-text` | 79.3 | text | Keep the scheme choice, let the solver derive its coefficients | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.5-text` | 80.5 | text | Replace the literal loop body with the physical statement | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-32.2-text` | 32.2 | text | No background: move the conventions the target needs into the prompt | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-22.2-text` | 22.2 | text | No background: state the convention that Verified added only to the background | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-55.1-text` | 55.1 | text | No background: state the numerical scheme the frozen target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-60.2-text` | 60.2 | text | No background: state the cutoff convention (Verified fixed it only in the background) | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.3-text` | 63.3 | text | No background: state the discretization scheme whose coefficients the target holds | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-74.1-text` | 74.1 | text | No background: state the sign convention and the square-case convention the target needs | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-76.2-text` | 76.2 | text | No background: state the convention the regenerated target encodes | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-80.3-text` | 80.3 | text | No background: replace an undefined term with the definition the target uses | text check (`textcheck/`) | `patches_d_text.json` |
| `v1-63.1-tests` | 63.1 | tests | Fix test calls that pass two bounds in the wrong order; regenerate the targets | ref passes; 2/2 mutants rejected; 0/0 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-tests` | 63.2 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 4/4 mutants rejected; 1/1 alternatives accepted | `patches_e_followup.json` |
| `v1-63.4-tests` | 63.4 | tests | Same argument-order fix (see v1-63.1-tests) | ref passes; 6/6 mutants rejected; 2/2 alternatives accepted | `patches_e_followup.json` |
| `v1-63.2-rows-text` | 63.2 | text | Restore the physical labels of the boundary rows, which are correct again after the argument-order fix | text check (`textcheck/`) | `patches_e_followup.json` |

### Why

- **`v1-72.6-tests`.** Verified removed the original seeded tests (an RNG-protocol defect) and left zero tests, so a do-nothing implementation passes. The new tests check limits with deterministic outcomes, plus an exact-enumeration statistical check that any valid random stream passes.
- **`v1-50.1-tests`.** Verified's single test compared a Monte Carlo average with another Monte Carlo estimate, inside a window several times too wide, so wrong-temperature and quench implementations pass. The new checks compare against exact Boltzmann averages of a small instance, within a few standard errors.
- **`v1-60.3-tests`.** Verified's one test at a single temperature cannot tell the correct temperature dependence from a common inverted one, and it also accepts a potential convention that the previous step excludes. The new tests use a configuration whose expectation is known exactly, at two temperatures.
- **`v1-62.5-tests`.** Verified rightly switched to basis-invariant checks, but its only case does not truncate, so any orthogonal transform passes, including the identity. The new cases truncate at gaps in the spectrum, where the kept subspace is unique.
- **`v1-67.6-tests`.** Verified loosened this check by several orders of magnitude, which lets a known wrong-physics implementation pass (it failed the original). The audit measured every valid form far inside the restored tolerance, and the wrong one far outside it.
- **`v1-76.4-text`.** This keeps the 76.4 tests consistent with the convention stated by v1-76.2-text. Without it, one natural reading of the patched 76.2 fails a 76.4 test. Tests unchanged.
- **`v1-76.2-tests`.** Verified's 76.2 targets encode a convention its text does not state, which contradicts 76.1 (the #184 contradiction). The targets are kept bit-for-bit and recomputed independently; v1-76.2-text states the convention. This patch makes the build check that the literal reading fails and that other valid implementations pass.
- **`v1-55.4-tests`.** The audit shows the dynamics are not chaotic. The original targets carry a defect of the gold itself. Targets are regenerated from a reference that 55.1's own tests already pin, and every valid loop form passes.
- **`v1-72.9-text`.** Verified pinned two load-bearing conventions correctly, but wrote them as the literal implementation. The new wording keeps both and drops the code.
- **`v1-55.3-text`.** The frozen targets need specific binning and peak-window conventions, so these stay. Verified also spelled out library calls and an explicit formula, which hand over the implementation.
- **`v1-66.1-text`.** The tests need one specific choice among several valid parameterisations, so the choice must be stated, but Verified gave the full construction as coordinates. The new wording fixes the same choice and leaves the derivation to the solver.
- **`v1-71.2-text`.** Only the order is a convention; the library call names the implementation.
- **`v1-37.3-text`.** The sentence gives away the asserted result and the sign of every target. The sign convention is already fixed elsewhere in the step.
- **`v1-73.1-text`.** Cosmetic. Verified removed the same marker elsewhere (69-d1). No grading effect.
- **`v1-69.3-text`.** Cosmetic. Verified made the identical fix in the sibling problem 67. No grading effect.
- **`v1-68.6-text`.** Computing this quantity is the step's whole job, and Verified wrote out the result. The new wording keeps the conventions it needs and the definitions it follows from.
- **`v1-55.2-text`.** The conventions are load-bearing and stay; the library calls hand over the implementation.
- **`v1-45.3-text`.** The solved update follows uniquely from the stated convention and is the step's job.
- **`v1-79.3-text`.** Several schemes of this order exist, so naming the scheme is load-bearing. Its coefficients follow uniquely and are part of what the step asks the solver to know.
- **`v1-80.5-text`.** The pair sum and the displacement direction are the conventions. The loop body is code.
- **`v1-32.2-text`.** Without background, the prompt does not fix the constants and conventions the frozen target needs. Several were only in the background, some added there by Verified. They are stated physically, and the derivable quantities are left to the solver.
- **`v1-22.2-text`.** The target depends on a convention a solver cannot infer from the prompt. A different common convention changes signs and fails.
- **`v1-55.1-text`.** The prompt names only the method family, and variants of it differ from the target at first order in the step size. The scheme is stated; the operators are left to derive.
- **`v1-60.2-text`.** The prompt and header never say which convention applies, and the other one fails a test.
- **`v1-63.3-text`.** Other standard schemes give a different matrix. The coefficients are left for the solver to derive.
- **`v1-74.1-text`.** The result is unique only up to signs. Verified added the deciding convention to the background only.
- **`v1-76.2-text`.** Verified's regenerated 76.2 targets assume a convention its text does not state, which contradicts the literal reading of 76.1's output (the #184 contradiction). Stating it keeps Verified's targets and changes only text.
- **`v1-80.3-text`.** Verified corrected the definition only in the background. Without background, the prompt's term has two wrong readings, and one test separates them.
- **`v1-63.1-tests`.** The 63.1/63.2/63.4 test cases pass two arguments swapped with respect to the signature, which reverses the grid and turns the boundary conditions into wrong finance. The later steps and the main-problem tests already use signature order. The regenerating reference reproduces every Verified target of 63.1-63.6 when called as Verified does.
- **`v1-63.2-tests`.** See v1-63.1-tests.
- **`v1-63.4-tests`.** See v1-63.1-tests.
- **`v1-63.2-rows-text`.** Verified had removed them because its tests reversed the grid.


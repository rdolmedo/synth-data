# SciCode final audit and freeze

The recommended dataset is `verified-aa-v3`: 64 problems, 287 scored steps, 61 versioned patches, no excluded steps. The public changes and payload hashes are in [PATCHES.md](PATCHES.md). Detailed scientific evidence and executable checks remain with the evaluation data under `/data/aai/aa_evals/data/scicode/patches/final_audit_20261106/`.

The final pass resumed the incomplete v2 audit using actual completed run artifacts. It reviewed all remaining first-failing steps, checked selected dependent failures with corrected prerequisites, extended mutation probes to steps with lower pass rates, tested domain-specific mutants for earlier probe survivors including structured return objects, and read the remaining word-level prompt additions against original SciCode. A model failure selected an investigation; it never established a defect.

Nine additions survived independent review:

- One underdetermined numerical-method choice is stated without coefficients or implementation instructions.
- Three text edits remove unnecessary solver instructions, literal library calls and a solved output expression while retaining required physical conventions.
- Five test edits restore omitted initial-condition, output-order, physical-validity and mechanical-limit requirements. Existing checks, targets and sound tolerances remain.

References, valid alternatives and wrong implementations are executed inside both frozen grading SIFs. The final build passes all 192 reference/mutant/alternative cases; the three edited text steps pass both SIFs separately. The manifest and `validation.json` record the build outcomes. All 61 added generic baselines pass; 60 generic suites complete, with the remaining expensive suite superseded by its dedicated physical-output checks after a decisive defect. No GPU, model endpoint or full evaluation was used by this audit.

This is a finite audit. There was no same-setting second-model SciCode run available. Selected dependent failures were investigated rather than reimplementing every never-passing step. Statistical and qualitative graders do not establish internal algorithm fidelity or eliminate all shortcuts. Existing limits remain documented in PATCHES.md, including runtime margin, dependency compatibility, incomplete coverage and retained cases whose evidence did not justify a change. These limits do not warrant additional hints, relaxed sound thresholds or automatic exclusions.

For an existing v2 run, use `--upgrade-from` with `--dataset verified-aa-v3`. Only three final-step prompts require new generation; five steps need regrading under stronger tests. At five repeats the offline plan is 15 regenerations, 25 regrades and 1395 carried attempts, across seven problems. Earlier-version upgrades additionally include their intervening patches. All new model comparisons should use the same frozen dataset version.

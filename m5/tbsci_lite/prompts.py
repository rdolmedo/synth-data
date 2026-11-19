"""Generator prompts for TB-Sci-lite-style tasks (#353). M1's one-shot author (task_synthesis/pipeline/prompts.py
GEN_SYSTEM, same output format, same package checks and judge) with the science brief: one coherent problem of the
spec's kind of work and main input; 4-12 expert hours (v2, after the pilot's spirit check); measured data from a
simulator the author writes (or a real public dataset the author is shown), or a model/problem specification with no
dataset; the answer never readable. The generator never sees a lite task: only the spec, the base image's contents and this
style guide. Class-level facts about lite (instruction length, deliverable style, closing line) are written in."""

GEN_SYSTEM = """You design one realistic scientific-computing task for training AI agents, in the style of Terminal-Bench-Science. A task is a concrete piece of computational science that a researcher or engineer would do at a Linux shell: analyse a dataset, fit or build a model, simulate, reconstruct, design or infer, and deliver precisely specified result files. An agent will later receive only the instruction and the environment you build; it never sees anything else you write.

# The environment
- Ubuntu 24.04 container built FROM {base_ref}. Preinstalled: {tools}. Python {python} with {pip} (installed system-wide; the PEP 668 marker is kept, so further pip installs need a venv or --break-system-packages). Also R (Rscript), ngspice, gnuplot, graphviz, gfortran.
- You may preinstall a few extra standard apt or pip packages for the task (the build reaches a package mirror). The agent has no network at all and cannot install anything: whatever it needs must be in the image.
- Resources: 2 CPUs, 8 GB RAM, 10 GB disk, no GPU. The intended analysis must run within about 10 minutes of compute on that.
- The working directory is /app. Put the task's materials under /app (e.g. /app/data, /app/docs, /app/tools) and ask for outputs under /app/output unless the scenario needs other paths.
- Your files are copied into the image first. Then a setup script runs once at build time, as root (10-minute limit, 2 CPUs): use it to run your data-generating programs and delete anything the agent must not see. Files under /tasksynth/ are construction-only and are deleted automatically after setup.

# The science and the data
- One coherent scientific problem with a definite, checkable result. Depth over breadth; do not stack independent sub-tasks.
- The specification's kind of work and main data define the task; build exactly that kind of task. A simulation task has the agent implement and run a numerical model; a design or optimization task has it find a design, controller or solution that meets stated targets; a reconstruction task has it invert indirect or degraded measurements; a pipeline task has it turn raw data into derived products. Do not turn the task into fitting parameters to a dataset unless the kind of work is fitting or estimation.
- When the main data is a model or problem specification, there is no measured dataset: ship the specification as a colleague would hand it over (governing equations, parameters and boundary conditions in a document; a netlist, geometry, mesh or configuration file; a problem instance; a partial code base) and ask for the solution, design or computed quantities. The hidden truth is then the solution, which your setup may compute and keep under /tasksynth/, or which the solution sketch describes.
- When the task has measured data and the specification gives you no real dataset, the data comes from a simulator you write: a program under /tasksynth/ with a fixed seed that implements the actual physics, chemistry, biology or engineering of the scenario (governing equations, realistic parameter values with units, measurement noise, instrument response, missing values, outliers, calibration offsets, nuisance effects) and writes the files the agent gets. Do not type more than about 30 lines of data by hand.
- The hidden truth (true parameters, labels, the clean signal, the optimum) must never be readable by the agent: keep it only in /tasksynth/ (deleted) or do not write it at all. The data files, file names, headers, metadata and comments must not leak it, and must not make it trivially recoverable (e.g. a column that is the label in disguise).
- The task must be solvable from what is shipped: the data must actually carry the signal at the stated noise level, and every constant, unit, convention, geometry or calibration the agent needs is either in the files or stated in the instruction.
- Data size: what the science needs, usually well under 20 MB, never over 50 MB.
- Materials should look like a real lab's or project's: realistic file names, headers, units, metadata, a short README or data dictionary where natural. Disclosed helper code is fine when the scenario has it (an instrument model, a forward simulator the agent may call, a format reader, a checker of the output format), as long as it does not solve the task.

# The instruction (all the agent sees)
- 250 to 500 words. Write it like a request from a scientist colleague: a short scientific context, what data is where (absolute paths, formats, columns, units, conventions), the scientific question, and exactly what to deliver.
- Name every output file with its exact format (columns, JSON keys, units, ordering, precision). Say how results will be judged, as concrete acceptance criteria a strict automated grader could check against the hidden truth (e.g. "the estimated diffusion coefficient must be within 5% of the true value", "balanced accuracy on the hidden labels of at least 0.85", "the design must meet every limit in the table"). No grader is shipped, but the criteria must be meaningful and achievable with the shipped data.
- State every required behavior and constraint. Give no hints on the method beyond what a colleague would naturally say, do not describe how the data was generated, and never reveal the hidden truth.
- Markdown is fine (short sections, a table of limits or files).

# Difficulty
- Target: a domain expert needs about {minutes} minutes, working by hand at the shell. The work is substantial: typically a few hundred lines of careful code (a solver, an estimator, an inversion, a processing chain, an optimizer) plus validation of it, and the time goes into understanding the problem and the data, choosing and validating the method, and getting details right (units, conventions, numerical stability, artifacts, edge cases). The compute itself must still fit the limits above.
- It must require real investigation and several interacting steps: no single library call or textbook formula applied to clean data. It must also be finishable: a strong expert agent working at the shell should complete it within two hours.

# Originality and safety
- Invent an original scenario. Do not reproduce known benchmark tasks, textbook exercises, competition problems or famous datasets (Iris, MNIST, Titanic, ...).
- No real personal or patient data; medical data must be simulated or public de-identified research data.

# Output format: exactly these blocks, nothing outside them
<task_json>
{{"slug": "kebab-case-name", "title": "...", "kind": "substrate|scaffold|none", "expert_minutes": 180,
  "apt": [], "pip": [],
  "inputs": ["/abs/path"],
  "outputs": ["/abs/path"],
  "defect_check": "",
  "services": false,
  "summary": "one sentence: what the agent must do"}}
</task_json>
<instruction>
...
</instruction>
<file path="/abs/path/to/file">
...
</file>
<setup>
#!/bin/bash
set -euo pipefail
...
</setup>
<solution_sketch>
...
</solution_sketch>

Notes on the blocks:
- kind: substrate = data the agent analyses; scaffold = code, model or specification files the agent builds on (a model specification, a partial solver, a netlist, a configuration, a problem instance); none = no task-specific files.
- inputs: every path the instruction relies on that must exist when the agent starts. outputs: paths the agent must create; they must not exist at the start. Files the agent must modify in place are inputs.
- Repeat <file> once per file; text files only (binary data such as .npy, HDF5, NetCDF, FITS, images or Parquet is produced by setup from your simulator). Leave defect_check empty and services false unless the scenario truly has a planted defect or a running daemon (then add a <services> block as in Terminal-Bench tasks).
- solution_sketch (never shown to the agent): how an expert solves it, the hidden true values or where the simulator keeps them, what a correct result looks like, and why it takes about the target time."""

GEN_USER = """Design a task for this specification.
- Field: {field}
- Subtopic: {subtopic}
- Kind of work: {archetype}
- Main data: {modality}{fmt}
- Data source: {source}
- Target expert time: about {minutes} minutes
"""

OPEN_SUBTOPIC = 'any area of this field you choose (pick something specific and less common)'

SOURCE_SIM = 'a simulator you write (see "The science and the data")'
SOURCE_REAL = 'the real public dataset described below (the core of the task)'

# The real-data block replaces M1's ASSET_USER for science datasets (from the shared scidata cache).
ASSET_INTRO = ("This task must be built around the real, public dataset below. Its files will be available during the "
               "image build under /tasksynth/asset/, laid out exactly as listed. You cannot run commands or open the "
               "files now: the listing, statistics and file contents below are everything you get, so write the "
               "complete task in this one reply, in the output format. ")

ASSET_USER = """
# The real dataset (the core of the task)
{intro}/tasksynth is deleted after setup, so your setup script must copy what the agent needs into place (e.g. /app/data), and may convert, subset, resample or re-split it.
- The dataset's real content is the core of the task. The instruction may only promise what the data actually contains (check the files and statistics below); do not invent columns, units or properties it lacks.
- Never leave the answer readable: if the task asks the agent to recover something the dataset records (labels, a measured quantity, later observations), your setup must hold it back (remove the column, keep the held-out part only in /tasksynth/), and the solution sketch says how it was held back.
- You may add simulated companion material only where the scenario needs it (e.g. an instrument response, a perturbation the agent must correct, a held-out split), never replace the real data with generated data.
- Refer to the data as a colleague would; the instruction must not depend on outside knowledge of where it came from, and the agent cannot download anything.
- Do not paste the dataset's content into <file> blocks; read it from /tasksynth/asset/ in setup.

Dataset: {title} ({source}; license {license}). {description}

## Dataset files under /tasksynth/asset/
{render}
"""

# The judge (M1's JUDGE_SYSTEM) gets one sentence more, on hidden truth (a leak there is a science task's commonest
# defect: the agent would read the answer instead of doing the work).
JUDGE_EXTRA = """
For scientific tasks also check: the hidden truth (true parameters, labels, the clean signal) is not readable by the agent in the BUILT environment and not trivially recoverable from file names, headers or metadata (if it is, set leaks_solution), and the shipped data really carries the signal needed to meet the stated acceptance criteria (if not, solvability <= 2)."""

# One-shot judge repair (round_sci.py, --judge-repairs): the reviewer's findings go back to the author.
JUDGE_REPAIR = """An independent reviewer audited the built task and rejected it.

Issues found:
{issues}

Scores below 4: {low}. Flags raised: {flags}.

Fix every issue. Check the whole task for the same kind of problem: units and conventions that agree across the instruction, the files and your simulator; acceptance criteria that the shipped data and a correct method actually meet; a hidden reference that does what the instruction asks; nothing that leaks the answer. If the scenario cannot be made consistent, redesign it within the same specification. Return the complete corrected task in the same output format, with all blocks and every file in full."""

# Lite's instructions all end with this sentence (141/141). The eval's tasks_2h.tar states min(upstream, 7200 s), the
# 2 h agent cap (aa_evals/tbsci/cap.py): 7200 in 137 tasks, 5400 in 1, 1800 in 2, 1200 in 1. Appended to every
# instruction, with the time drawn from that mix, so the agent's prompt keeps the eval's form.
CLOSING = 'You have {seconds} seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.'
CLOSING_SECONDS = [(7200, 137), (5400, 1), (1800, 2), (1200, 1)]

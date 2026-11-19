"""Generator, repair and judge prompts. The generator never sees Terminal-Bench tasks: only a spec, the base image's
contents and the written style guide below (BLUEPRINT.md: no TB tasks in prompts)."""

ACTIONS = {
    'build-from-spec': 'build something new (a program, tool, pipeline or artifact) from a specification',
    'debug-fix': 'find and fix what is broken',
    'analyze-answer': 'analyze data, code or logs and report precise answers',
    'configure-deploy': 'configure, install or deploy software or services',
    'optimize': 'make something faster, smaller or cheaper while keeping its behavior',
    'reverse-engineer': 'work out how an undocumented artifact (format, binary, protocol, program) works',
    'implement-feature': 'add a feature to existing code',
    'train-evaluate-ml': 'train, tune or evaluate a machine-learning model (CPU only)',
    'security-attack-defend': 'a sandboxed security exercise: find, exploit or close a planted weakness, or analyze planted artifacts',
    'port-migrate': 'port or migrate code, data or configuration to another language, format or version',
    'transform-convert': 'transform or convert data or documents between formats',
    'recover-data': 'recover data that was lost, deleted, corrupted or encrypted',
    'prove-verify': 'prove, verify or formally check a property',
}

ROLES = {
    'anchor': 'the main material the agent works on',
    'authored': 'written for this task',
    'derived': 'produced from other materials during setup (e.g. compiled, rendered, exported)',
    'stock': 'standard software or packages installed in the image',
    'sourced': 'would normally come from the web; write a realistic self-made stand-in instead',
}

GEN_SYSTEM = """You design one realistic terminal task for training AI agents, in the style of Terminal-Bench. A task is a concrete job that a skilled engineer, scientist or analyst would do at a Linux shell, inside an environment you build. An agent will later receive only the instruction and the environment; it never sees anything else you write.

# The environment
- Ubuntu 24.04 container built FROM {base_ref}. Preinstalled: {tools}. Python {python} with {pip} (installed system-wide; the PEP 668 marker is kept, so further pip installs need a venv or --break-system-packages).
- You may preinstall a few extra standard apt or pip packages for the task. The agent has no general internet access; apt, pip and npm work through a package mirror.
- Resources: 2 CPUs, 8 GB RAM, 10 GB disk, no GPU. Anything the agent must run should finish within minutes on that.
- The working directory is /app. Put the task's materials under /app unless the scenario needs system paths (/etc, /var/log, /srv, /home/<user>, ...).
- Your files are copied into the image first. Then a setup script runs once at build time, as root: use it to run data-generating programs, create git history, users and permissions, compile binaries, plant defects, and delete anything the agent must not see. Files under /tasksynth/ are construction-only and are deleted automatically after setup.
- If the task needs a daemon running when the agent starts (database, web API, queue, ...), write a services script: it runs as root at container start, before the agent arrives, with no internet. It must start the daemons in the background and wait until they accept connections. Daemons must be installed in the image (apt/pip list or setup). The services script stays visible in the image at /etc/tasksynth/services.sh.

# Realism and data
- Materials must look real. Generate data with a program (fixed random seed, explicit distributions, realistic noise and edge cases) placed under /tasksynth/ and run from setup. Do not type more than about 30 lines of data by hand.
- Code the agent works on should read like real project code (modules, configuration, a README where natural), not a toy.
- Never ask for anything the environment cannot support.
- Anything the instruction says already exists, is installed or is running must be created by your files, setup script or services script. Reviewers check the built environment, not your intentions.
- Never leave expected outputs, answer keys or golden results anywhere the agent can read them; keep them in /tasksynth/ or out of the image.

# The instruction (all the agent sees)
- Write it like a request from a colleague or a ticket: a sentence or two of context, then exactly what to do.
- Use absolute paths. Name every output file with its exact format (columns, JSON keys, units, ordering, precision).
- State every required behavior and constraint. A strict automated grader must be able to check the result, although none is shipped.
- Give no hints on how to solve it. Do not describe how the environment was built or where a defect is.
- 80 to 400 words. Plain prose; a short list is fine.

# Difficulty
- Target: an expert needs about {minutes} minutes. It must require investigation and several steps: no single obvious command or one-line fix.
- One coherent goal; depth over breadth.

# Defects
- When the task is about something broken, lost or hidden, plant it in the files or in setup, and give `defect_check`: a shell command that exits 0 if and only if the defect is present when the agent starts.

# Originality and safety
- Invent an original scenario. Do not reproduce known benchmark tasks, textbook exercises or famous datasets (Titanic, Iris, MNIST, ...).
- Security tasks stay inside the sandbox: CTF-style, defensive, or analysis of planted artifacts. No real malware, credentials or personal data.

# Output format: exactly these blocks, nothing outside them
<task_json>
{{"slug": "kebab-case-name", "title": "...", "kind": "none|scaffold|puzzle|substrate", "expert_minutes": 60,
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
<services>
#!/bin/bash
...
</services>
<solution_sketch>
...
</solution_sketch>

Notes on the blocks:
- kind: none = no task-specific files, the agent starts from the stock image; scaffold = files written for the task that the agent builds on; puzzle = the environment carries a planted defect, secret or corruption; substrate = realistic data or code is the material to analyze or process.
- inputs: every path the instruction relies on that must exist when the agent starts. outputs: paths the agent must create; they must not exist at the start. Files the agent must modify in place are inputs.
- Repeat <file> once per file; text files only (binaries, archives, databases and media are produced by setup). Omit <services> unless "services" is true. <setup> may be just the shebang and set line.
- solution_sketch (never shown to the agent): how an expert solves it, the expected answers or key values where they are determined, and why it takes about the target time."""

GEN_USER = """Design a task for this specification.
- Domain: {domain}
- What the agent mainly does: {action}
{materials}- Target expert time: about {minutes} minutes
{extra}"""

NO_ASSETS = """- Materials: none. The environment has no task-specific files, data or running services; the agent starts from the stock image, may install packages, and builds everything the task needs. Leave <file> blocks out and keep setup empty unless the scenario needs system configuration (users, directories). If the agent needs data to work with, let it create or choose its own; never make the task hinge on reproducing an exact random process described in prose."""

REPAIR = """The task failed automated checks after building it:

{errors}

Fix every problem and return the complete corrected task in the same format, with all blocks."""

JUDGE_SYSTEM = """You audit a synthesized terminal task before it is used to train AI agents. The agent will see only the instruction and the built environment. Judge strictly: a weak task wastes expensive training compute.

You receive the specification, the instruction, the author's files, setup and services scripts, a snapshot of the BUILT environment (directory tree, heads of the input files, defect check result), and the author's private solution sketch. Files under /tasksynth/ are construction-only: the build deletes that directory after setup, so the agent never sees them; the snapshot shows what the agent sees.

Score each criterion from 1 (bad) to 5 (excellent):
- clarity: unambiguous; outputs, formats and constraints fully specified; a strict grader could check the result.
- consistency: everything the instruction states or implies about the environment is true of the BUILT environment (files, fields, values, data properties, versions, behaviors).
- solvability: an expert can solve it here with 2 CPUs, 8 GB RAM, no GPU, no internet except package managers, and all needed information is available.
- difficulty: fit to the target expert time (5 = matches; 1 = trivial, or far beyond the target).
- realism: a plausible real-world job with realistic materials that exercises a useful skill.
- originality: not a stock textbook exercise or a well-known benchmark task.

Flags (true/false): leaks_solution (the instruction gives away the method or answer), needs_internet, needs_gpu, unsafe, spec_mismatch (ignores the specification).

Answer with JSON only:
{"scores": {"clarity": n, "consistency": n, "solvability": n, "difficulty": n, "realism": n, "originality": n}, "estimated_minutes": n, "flags": {"leaks_solution": b, "needs_internet": b, "needs_gpu": b, "unsafe": b, "spec_mismatch": b}, "issues": ["short, specific problems"], "verdict": "pass" or "fail"}

Verdict pass only if clarity, consistency and solvability are at least 4, difficulty, realism and originality at least 3, and no flag other than spec_mismatch is true."""

JUDGE_USER = """# Specification
{spec}

# Instruction
{instruction}

# Author's files
{files}

# Setup script
{setup}

# Services script
{services}

# Built environment (automated snapshot)
{snapshot}

# Author's solution sketch (private)
{sketch}"""

# One-shot authoring (the A/B rounds): the generator sees only the rendering below.
ASSET_ONESHOT_INTRO = ("This task must be built around a real, public asset: the first material listed above. Its files will be "
                       "available during the image build under /tasksynth/asset/, laid out exactly as listed below. You cannot "
                       "run commands or open the files now: the listing, statistics and file contents below are everything you "
                       "get, so write the complete task in this one reply, in the output format. ")
ASSET_AGENTIC_INTRO = ("This task must be built around a real, public asset: the first material listed above. Its files will be "
                       "available during the image build under /tasksynth/asset/, laid out as listed below; explore them in the "
                       "sandbox described next before you write. ")

# Agentic authoring (2026-11-01): the generator explores the asset in a sandbox first; the paragraph above that says it
# cannot run commands is replaced by this one (pipeline.explore, round --author agentic).
ASSET_EXPLORE = """Before you write the task, get to know the asset in a sandbox: a container of the task's base image with the asset's files under /tasksynth/asset/ (a writable copy; the preinstalled tools and Python are there, and pip, apt and npm reach a package mirror). Run shell commands there with the bash tool, one command per call; you get its exit code and output (long output is cut in the middle, so prefer head, grep, wc, small scripts). You have up to {commands} commands for the whole task, including fixing it after the build checks, and each may run up to 10 minutes. As the conversation grows, older command outputs are shortened to save room, so write what you will need (facts, values, paths, decisions) to /tmp/notes.md as you go and read it back before you write the task. Take the time you need:
- learn what the asset really contains and how it behaves: structure, sizes, encodings, schemas, quirks, history, how its programs build and run;
- find what in it makes a realistic, demanding task for this specification, so the task depends on the asset's real content rather than on material you invent around it;
- compute or try everything your instruction, setup and solution sketch rely on: counts, values, expected answers, the defect you plant and how it shows, commands the agent would run, your setup steps.
Nothing you do in the sandbox carries over to the build: the image is built from your files and setup script as usual, with the asset again under /tasksynth/asset/. When you are ready, reply with the complete task in the output format, without a tool call. After the build checks you may use the sandbox again to debug.
"""

# Workspace authoring (iteration 2, 2026-11-01): the author builds the task as files in the sandbox and submits them;
# replaces ASSET_EXPLORE in round --author workspace. Build, not solve (iteration 4, user 2026-11-01: "it just needs to
# build it, not build it and then solve it"): ws5/ws6 authors spent 46% of their commands writing and rerunning their own
# reference solutions (~450k tokens per author vs ~60k one-shot). Scope paragraph (iteration 3): agentic prototypes ag1/ag2 stacked
# several jobs into one task; Super solved 39-48% of them vs 74% one-shot and hit the context cap 21-27% vs 4%.
ASSET_WORKSPACE = """You build this task inside a sandbox: a container of the task's base image with the asset's files under /tasksynth/asset/ (a writable copy; the preinstalled tools and Python are there, and pip, apt and npm reach a package mirror). Run shell commands there with the bash tool, one command per call; you get its exit code and output (long output is cut in the middle, so prefer head, grep, wc, small scripts). You have up to {commands} commands, each up to 10 minutes, and about {tokens}k tokens of your own output in total (thinking included); plan the work to fit both. As the conversation grows, older command outputs are shortened to save room, so keep what you need (facts, values, paths, decisions) in /tmp/notes.md and read it back when you need it.

Work in this order, and take the time you need:
1. Get to know the asset: what it really contains and how it behaves (structure, sizes, encodings, schemas, quirks, history, how its programs build and run), and what in it makes a realistic, demanding task for this specification, one that depends on the asset's real content rather than on material you invent around it.
   Scope: one coherent goal that an expert finishes in about {minutes} minutes, built on the most interesting thing you found in the asset. Do not stack independent jobs (an analysis plus a deployment fix plus a report): pick one and do it in depth. If an expert could not finish it in that time, cut the task down.
   The agent has no internet but can install public packages from a mirror (pip, npm, apt). If the work amounts to reimplementing what a public package already provides, the instruction must say whether that package may be used.
2. Write the task as files under /task (instead of replying with the output format; each file holds what the matching block of the output format describes):
   - /task/task.json: the <task_json> object
   - /task/instruction.md: the <instruction>
   - /task/files/...: each <file>, at its absolute path under /task/files (/task/files/app/config.yaml becomes /app/config.yaml); text only
   - /task/setup.sh: the <setup> script (the asset is under /tasksynth/asset/ while it runs)
   - /task/services.sh: the <services> script, only if task.json says "services": true
   - /task/solution_sketch.md: the <solution_sketch>: how an expert would solve it and what a correct result looks like (the reviewer of the agent's work reads it); give exact expected values only where a quick look at the data settles them (a count, a field, the planted defect), not by solving the task
3. Check the build, not the solution: you build the task; solving it is the agent's job, so do not implement or run a solution yourself. The check_task tool builds /task exactly as the pipeline will and runs the automated checks: use it, read what it reports, and fix /task until it passes. Make sure the built environment is what the instruction says it is: the files and services exist where it says, facts it states about the asset (names, counts, formats, behaviour) are true, a planted problem is really there and shows as described, and the instruction, files and sketch agree with each other.
4. Call submit. The task is then built, checked and reviewed; if it fails, you get the problems back, and the sandbox and /task are still yours to fix it.
"""

# --workflow lean (iteration 5, 2026-11-02): workspace authors averaged 86 commands, each re-reading ~33k tokens of
# context and spending most of its tokens on reasoning; fewer, larger steps cost less for the same work.
ASSET_LEAN = """Work economically: every tool call re-reads the whole conversation, so take fewer, larger steps. Combine related commands in one call (a short script, or commands joined with && or ;), write each file whole with one heredoc rather than through many small edits, look into large files with targeted commands (head, grep, wc, a few lines of Python), and run check_task when the task is complete, then again only after fixing what it reported."""

# --workflow lean2 (iteration 6, 2026-11-02): in every workspace arm most first attempts ended without /task: the author
# wrote the task last, the budget ran out first, and the refusal told it to reply in the output format (a text dump that
# cannot be exported), so the "repair" attempt wrote the task. Draft early, fix by targeted edits, finish on a grace.
ASSET_LEAN2 = """Work economically: every tool call re-reads the whole conversation, so take fewer, larger steps. Combine related commands in one call (a short script, or commands joined with && or ;), and look into large files with targeted commands (head, grep, wc, a few lines of Python).
Write a complete first draft of /task (task.json, instruction.md, setup.sh, solution_sketch.md and the files) as soon as you have chosen the task, well before half of your budget is spent, then improve it: whatever is in /task when the budget runs out is what gets submitted. Write each file whole once; after that fix it with targeted edits (sed, or a few lines of Python) rather than rewriting it. Run check_task once the draft is complete, and again only after fixing what it reported."""

# --workflow lean3 (iteration 7, 2026-11-02): in the lean2 production run 30% of specs were not accepted. 20% built a
# valid task that the round judge rejected (mostly consistency, clarity, solvability, spec_mismatch), 3% never wrote
# task.json/instruction.md, and 3% hit the 10-minute image build limit. They cost about a third of all author tokens.
ASSET_LEAN3_EXTRA = """Write /task/task.json and /task/instruction.md first, as soon as you have chosen the task (drafts are fine), then the rest.
setup.sh runs once at image build with a 10-minute limit on 2 CPUs: no large compiles or test-suite runs, no downloads beyond the package mirror; build only what the task needs.
Before you call submit, review /task the way the reviewer will, and fix what fails: (1) consistency: every path, name, count, value and format in the instruction matches the files, setup.sh and the solution sketch; (2) clarity: one unambiguous goal with exact output requirements (paths, formats); (3) solvability: an expert can do it offline with the tools present, in about the target time; (4) it matches the specification's action and domain; (5) the instruction does not give the solution away."""

WORKSPACE_JUDGE_REPAIR = """A reviewer read the task in /task and did not accept it.

Issues:
{issues}

Scores below 4 (out of 5): {low}. Flags: {flags}.

Fix these problems in /task with targeted edits (do not start the task over), make the instruction, files and solution sketch agree, run check_task once, and call submit."""

WORKSPACE_GRACE = """[Your budget is used up. You have a final allowance of {commands} commands and about {tokens}k tokens: finish /task with what you have (write any missing file, fix what is broken), run check_task at most once, and call submit. Do not start anything new.]"""

WORKSPACE_REPAIR = """The submitted task failed:

{errors}

Fix it in /task (the sandbox and your files are still there), check it with check_task, and call submit again."""

ASSET_FIRST_USER = """Design a task around the real asset below.
- Domain: {domain}
- What the agent mainly does: choose the kind of work that this asset most naturally supports for a realistic, demanding task, from: {actions}. Say which in the summary of task_json.
- Materials: the real asset is the anchor; add companion files, data or services only where the scenario needs them.
- Target expert time: about {minutes} minutes
"""

# Real assets (issue #20). The generator's system prompt is unchanged; an asset task's user prompt ends with this
# block, and the judge gets one more criterion (asset_use) and the asset's provenance and manifest.
ASSET_USER = """
# The real asset (the anchor)
{intro}/tasksynth is deleted after setup, so your setup script must copy what the agent needs to its place (under /app, or wherever the scenario needs it), and may convert, subset or compile it, or plant a defect in a copy.
- The asset's real content is the core of the task. Do not replace it with generated data and do not invent properties it lacks: the instruction may only promise what the asset actually contains (check the files and statistics below). Companion materials the specification lists as written for the task are still yours to write.
- Keep it real: trim or convert only where size, runtime or the scenario requires it; keep its names, structure and quirks.
- When the agent mainly works on the anchor's kind of material (see the specification), the work happens in the asset's own content: plant the defect, request the change or point the analysis at the asset's real code, data or history, not at wrapper code you write around it.
- The agent works offline: anything the asset needs at run time (data it would download, external services, a GPU) must be provided by you or avoided by the task. Fetch build dependencies (cargo, npm, pip, go modules) during setup.
- Ubuntu 24.04's toolchains can be older than the asset expects (rustc 1.75, Go 1.22, Node 18, CMake 3.28, GCC 13). If the asset needs newer ones, install a pinned official release in setup (the build reaches the web through a proxy), or ship a part of the asset that builds with what is there.
- Refer to the material as a colleague would. The instruction must not depend on outside knowledge of where it came from.
- Do not paste the asset's content into <file> blocks; read it from /tasksynth/asset/ in setup.

Asset: {title} ({source}; license {license}). {description}
A reviewer's note on how it could anchor the task (a suggestion, not a requirement): {idea}

## Asset files under /tasksynth/asset/
{render}
"""

JUDGE_ASSET_CRITERION = """- asset_use: the task rests on the real asset's actual content (shown under "Real asset"): the asset is the core material rather than decoration next to invented data, and everything the instruction says about it holds in the built environment (5 = the task could not exist without this asset's real content; 1 = the asset is ignored or replaced by generated data).
"""

JUDGE_ASSET_USER = """# Real asset (copied to /tasksynth/asset/ during the build; the setup script places it)
{asset}

"""


def judge_system(asset):
    if not asset:
        return JUDGE_SYSTEM
    s = JUDGE_SYSTEM.replace('\n\nFlags (true/false)', '\n' + JUDGE_ASSET_CRITERION.rstrip('\n') + '\n\nFlags (true/false)', 1)
    s = s.replace('"originality": n}', '"originality": n, "asset_use": n}', 1)
    s = s.replace('Verdict pass only if clarity, consistency and solvability are at least 4,',
                  'Verdict pass only if clarity, consistency, solvability and asset_use are at least 4,', 1)
    assert 'asset_use' in s and s.count('asset_use') >= 3, 'judge prompt anchors moved'
    return s

You audit training data for leakage from a held-out benchmark. Terminal-Bench 2.1 (TB2.1) is a held-out benchmark of terminal tasks: each task gives an agent an instruction and a Docker environment, and hidden tests check the end state. CalibForge is a large synthetic training set written in the same format. Its generator appears to have been shown Terminal-Bench tasks while writing, so some candidates were built by taking one specific test task and changing parts of it.

You will see one TB2.1 test task in full, then deterministic overlap signals for the pair, then one CalibForge candidate task in full (long environment files may be excerpted). Decide whether the candidate is a **variant of this specific test task**, or **its own task** that merely belongs to the same family.

Both errors matter. Training tasks in the same family as a test task (other build-from-source tasks, other bug fixes, other video-analysis scripts) are expected and wanted, and flagging them throws away good training data. But a candidate that carries this test task's specific choices leaks the test task, whether it copies the text or rewrites every sentence.

## First: what is distinctive about the test task

Before you compare, read the test task on its own, as a careful reader who has not seen the candidate, and separate:

- **Genre** and **conventions**: what many independently written tasks share (the kind of task, its domain, tools and skills, format conventions such as `/app`, writing a result to a file, "make the tests pass"). These never count, however close the candidate is on them.
- **Distinctive choices**: what this task's author picked that another author in the same genre would be unlikely to repeat (the specific instance, the crux or a given-away workaround, the deliverable and interface, unusual constraints, specific tools or data, distinctive wording, section skeleton and starter files). Weight each:
  - **decisive**: one match alone makes the candidate a variant (the crux, a given-away workaround, the same specific instance, the task itself with surface changes);
  - **minor**: several together make a variant; one alone usually does not.

Fix this list before you look for matches, and do not promote something to distinctive just because the candidate happens to share it: ask whether an independent author in the same genre would plausibly have made the same choice.

## How to compare

Go through the test task's distinctive choices and decide which ones the candidate carries, **in any wording**.

- **Rewrites count.** Rewording every sentence and renaming every file does not help a candidate that keeps the choices (e.g. the same tokenizer, dataset type and bare-integer answer; the same workaround; the same interface around a swapped component).
- **Copied form is enough on its own.** If the candidate's instruction is recognisably this test task's instruction with other names filled in (most of its distinctive sentences, or its skeleton of opening, sequence of requirements and closing), or the candidate reproduces this test task's starter-file set with its naming scheme and roles, label REUSE even when the problem inside and every decisive choice differ: the candidate was made by editing the test task. A hidden test suite that is the test task's suite with names swapped supports this finding. The copied sentences must be distinctive: an instruction in the generic shape any author would write for this kind of task (complete the proof in X.v and compile it with coqc; fix the bug so the tests pass; train a model and save it to a path) is genre, even when the candidate follows it closely.
- **Swapping the core problem does not rescue a variant.** A candidate that keeps the test task's frame, deliverable and constraints but asks to detect a different event, build a different package, port from a different language or compute a different function is still a variant.
- **Genre never counts.** Sharing the kind of task, domain, tool, skill or format conventions is fine. These count for nothing on their own, even next to each other: a stock sentence ("Fix the bug so that the tests pass", "Recover as many records as possible"), a conventional result file (a JSON list or a text answer under `/app`), a generic organisation or person name, a placeholder domain (`example.com`, `company.local`), a standard port (8080, 8443, 2222), a textbook technique (XOR with a key found from known magic bytes, recovering a commit through the reflog). The exception is when that very item is this test task's crux or instance: then it is distinctive here.
- **Containing the test task counts.** A candidate that contains the test task's problem (its crux in the same scenario) as one of several parts is a variant, however much it adds around it.
- **A match must be specific.** The candidate matches a choice only if it makes the same choice, not merely a choice of the same type (another library restriction, another output file, another dataset is not a match).
- **Hidden test code** counts only when it checks the test task's own success criterion or scenario; test utilities and scoring helpers copied on their own do not count.

Then step back and ask: taken together, does the candidate reproduce this test task with parts swapped, or is it a task of its own that shares the genre plus at most a detail?

## Illustrations (made up, not from either dataset)

- Test task: merge two CSV exports keyed on customer ID, where one export carries a legacy `C-` prefix on IDs and duplicates are resolved by the latest timestamp. Candidate: merge two JSON order exports keyed on order number, where one source has a legacy `ORD` prefix and the most recent update wins. Different data and wording, but the crux (prefix mismatch) and the resolution rule match: REUSE. Another candidate that joins two CSV files on a key, with its own data quirks: genre only, NO_REUSE.
- Test task: "Write a script named `plan.py` at `/app/plan.py`. You may only use the standard library. Write the schedule as YAML to `/app/plan.yaml` with exactly the keys `start` and `end`." Candidate: a different scheduling problem whose instruction has the same opening sentence with another file name, the same standard-library-only clause, and the same YAML deliverable: three minor matches that together reproduce the frame: REUSE.

## Labels

- **REUSE**: the candidate is a variant of this test task: it carries a decisive choice; or it copies the test task's distinctive form (templated instruction, or starter-file set and naming); or it carries several minor choices (three or more) that together reproduce the test task's frame, deliverable or setup, in copied or reworded form.
- **WEAK_REUSE**: one or two minor choices match and the rest is the candidate's own; derivation is plausible but so is coincidence. Two minor matches alone are WEAK_REUSE, not REUSE.
- **NO_REUSE**: the candidate shares at most the genre and conventions (and perhaps a single unremarkable detail).

Quote your evidence: for each match give a short quote or file name from both tasks. Do not label REUSE without matches quoted from both sides.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"choice": "<short name of the distinctive choice>", "weight": "decisive" | "minor", "form": "copied" | "reworded", "test_task": "<short quote or file name>", "candidate": "<short quote or file name>"}, ...], "reason": "<one or two sentences>"}

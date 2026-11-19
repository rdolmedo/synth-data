You audit training data for leakage from a held-out benchmark. GDPval is a held-out benchmark of professional knowledge-work tasks: each task is a written request from a manager or client to a professional in one of 44 occupations (accountants, nurses, lawyers, sales managers, engineers, ...), sometimes with input files (spreadsheets, documents, PDFs), and the agent delivers files (Word, Excel, PDF, PowerPoint). The candidates come from a large synthetic training set in the same format, written by a language model from a short specification (an occupation, a generic work activity such as "monthly budget variance analysis", a kind of work product, file formats) and a written style guide. The model was never shown a GDPval task while writing, but it may have read GDPval during pretraining, so a candidate could reproduce a test task from memory, copied or reworded. That is what you check.

You will see one test task (its prompt and the names of its input files), deterministic overlap signals for the pair, then one candidate task (its prompt and excerpts of its input files). Decide whether the candidate is a **variant of this specific test task**, or **its own task** that merely belongs to the same occupation and kind of work.

Both errors matter. Training tasks of the same occupation and genre as a test task (another month-end close, another nursing care plan, another sales forecast, another lease abstract) are expected and wanted, and flagging them throws away good training data. But a candidate that carries this test task's specific choices leaks the test task, whether it copies the text or rewrites every sentence.

## First: what is distinctive about the test task

Read the test task on its own and separate:

- **Genre** and **conventions**: what many independently written requests in this occupation share (the kind of work product, the domain, standard sections of such a document, common formats, "You are a ... at ...", "deliver a PDF", "state your assumptions"). These never count.
- **Distinctive choices**: what this task's author picked that another author in the same genre would be unlikely to repeat: the specific scenario and instance (the particular organisation situation, product, case facts, event), the specific combination of required outputs and their structure (named tabs, specific sections in a specific order), specific figures, names, dates and constraints, the specific input-file set and its roles, distinctive wording. Weight each:
  - **decisive**: one match alone makes the candidate a variant (the same specific scenario or case, the same instance with surface changes, the same set of input files with the same roles and the same asks);
  - **minor**: several together make a variant; one alone usually does not.

Fix this list before you look for matches, and do not promote something to distinctive just because the candidate happens to share it.

## How to compare

- **Rewrites count.** Renaming the company, people and files does not help a candidate that keeps the scenario, the asks and their structure.
- **Copied form is enough on its own.** If the candidate's request is recognisably this test task's request with other names filled in (most of its distinctive sentences, or its skeleton: the same opening situation, the same sequence of specific requirements, the same closing), label REUSE even if the subject differs: the candidate was made by editing the test task (scaffold reuse). The skeleton must be distinctive: the generic shape any author would write for this kind of request is genre.
- **Swapping the subject does not rescue a variant.** Keeping the test task's frame, deliverable structure and constraints while changing the product, the city or the numbers is still a variant.
- **Genre never counts.** Same occupation, same kind of document, standard sections, standard calculations (NPV, variance, reorder point), common formats: fine, even all together.
- **A match must be specific.** The same type of choice (another set of named tabs, another word limit) is not a match; the same choice is.

Then step back: does the candidate reproduce this test task with parts swapped, or is it a task of its own that shares the genre plus at most a detail?

## Illustrations (made up, not from either dataset)

- Test task: a regional bakery chain's operations manager must build a 3-tab staffing workbook for a holiday weekend from a roster file and a sales forecast file, with tabs "Coverage", "Overtime" and "Gaps". Candidate: a hardware store's manager builds a 3-tab staffing workbook for a holiday weekend from a roster and a forecast with tabs "Coverage", "Overtime cost" and "Open shifts": same frame, inputs and deliverable structure: REUSE. Another candidate that schedules nurses for a ward from an availability sheet into a rota: genre only, NO_REUSE.
- Test task: draft a will for a named Texas client with a testamentary trust, specific executors and a spendthrift clause. Candidate: draft a will for a Florida client with different family facts, no trust: genre (estate planning) plus one minor detail: NO_REUSE or at most WEAK_REUSE.

## Labels

- **REUSE**: the candidate is a variant of this test task: it carries a decisive choice; or it copies the test task's distinctive form; or it carries three or more minor choices that together reproduce the test task's scenario, deliverable structure or input set.
- **WEAK_REUSE**: one or two minor choices match and the rest is the candidate's own; derivation is plausible but so is coincidence.
- **NO_REUSE**: the candidate shares at most the genre and conventions (and perhaps one unremarkable detail).

Quote your evidence: for each match give a short quote from both tasks. Do not label REUSE without matches quoted from both sides.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"choice": "<short name>", "weight": "decisive" | "minor", "form": "copied" | "reworded", "test_task": "<short quote>", "candidate": "<short quote>"}, ...], "reason": "<one or two sentences>"}

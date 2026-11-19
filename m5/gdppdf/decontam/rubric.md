You audit training data for leakage from a held-out benchmark. GDP.pdf is a held-out benchmark of professional document-analysis tasks: each test task is a request that a professional sends to an AI assistant together with one long PDF document (a filing, guideline, policy, manual, report, contract, record ...); the assistant answers from the document's text alone. The candidates come from a large synthetic training set in the same format: a language model read a different real document and wrote requests about it, from a short specification (a domain, a task type such as "check against rules", a task family such as "determine coverage for a described claim under an auto policy", a number of asks and pitfalls). The model was never shown a GDP.pdf task while writing, but it may have read GDP.pdf during pretraining, so a candidate could reproduce a test task from memory, copied or reworded. That is what you check.

You will see one test task (its request, and a short description of its document), then overlap signals for the pair, then one candidate (its request and a short description of its document). Decide whether the candidate is a **variant of this specific test task**, or **its own task** that merely belongs to the same family.

Both errors matter. Training tasks in the same family as a test task (other coverage determinations, other compensation-table calculations, other guideline-based patient work-ups) are expected and wanted, and flagging them throws away good training data. But a candidate that carries this test task's specific choices leaks the test task, whether it copies the text or rewrites every sentence.

## First: what is distinctive about the test task

Read the test task on its own and separate:

- **Genre and conventions**: what many independently written requests share (the kind of work, the domain, the document type, generic asks such as "summarize", "make a table", "show your work", common professional framings such as "I'm an HR manager"). These never count, however close the candidate is on them.
- **Distinctive choices**: what this task's author picked that another author would be unlikely to repeat: the specific document or the same issuer and series (another year of the same company's filing, the same insurer's policy form, the same agency report series), the specific case (the scenario's parties, events, dates and amounts), the specific combination of asks and constraints, the crux or trap, the specific items or figures to compile, distinctive wording or the request's skeleton. Weight each:
  - **decisive**: one match alone makes a variant (the same document or issuer+series with the same kind of ask; the same case scenario; the same crux on the same kind of document; the task itself with surface changes);
  - **minor**: several together make a variant; one alone usually does not.

## How to compare

- **Rewrites count.** Rewording every sentence and renaming parties does not help a candidate that keeps the choices.
- **Copied form is enough on its own.** If the candidate's request is recognisably this test request with other names filled in (most of its distinctive sentences, or its skeleton of context, sequence of asks and constraints), label REUSE even when the document differs: reusing a test task's scaffold counts as contamination.
- **Swapping the document does not rescue a variant** when the scenario, asks and constraints are the test task's.
- **Genre never counts.** Same domain, same document type, a generic ask, a generic persona, standard formatting requests: these count for nothing on their own, even together.
- **A match must be specific.** The same choice, not merely a choice of the same type (another claim scenario, another compensation table, another guideline is not a match).

Then step back: does the candidate reproduce this test task with parts swapped, or is it a task of its own that shares the genre plus at most a detail?

## Illustrations (made up, not from either dataset)

- Test: an adjuster asks whether liability, collision and towing coverages apply to a two-car accident where the named insured's teenage son drove an unlisted vehicle, under the attached personal auto policy. Candidate: a claims assistant asks which coverages apply when the insured's nephew, not listed on the policy, crashed a newly bought second car, under another insurer's auto policy, with the same coverage list to check. Same case structure and crux (unlisted driver/vehicle, the same coverages to adjudicate): REUSE. Another candidate that asks whether water damage from a burst pipe is covered under a homeowners form: genre only, NO_REUSE.
- Test: compute year-over-year changes in a named executive's stock awards from a proxy statement and point to the page explaining the drivers. Candidate: compute YoY changes in the CEO's total compensation from a different company's proxy statement. One minor match (compensation YoY from a proxy) is genre: NO_REUSE; if it also asks for the same index lookup and the same follow-up on why one officer's award changed, WEAK_REUSE or REUSE by the count of matches.

## Labels

- **REUSE**: a decisive match; or copied form; or three or more minor matches that together reproduce the test task's frame.
- **WEAK_REUSE**: one or two minor matches, the rest the candidate's own.
- **NO_REUSE**: at most the genre and conventions (and perhaps one unremarkable detail).

Quote your evidence: for each match give a short quote from both sides. Do not label REUSE without matches quoted from both.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"choice": "<short name>", "weight": "decisive" | "minor", "form": "copied" | "reworded", "test_task": "<short quote>", "candidate": "<short quote>"}, ...], "reason": "<one or two sentences>"}

# Style guide: expert closed-ended questions (HLE style)

You write one question for an exam of extremely hard, expert-level, closed-ended questions. A strong frontier model
answering with no tools, no internet and no images must find it hard; a world expert in the field must find it fair.

## What every question must satisfy
1. **One short, unambiguous, verifiable answer.** Exactly one answer is correct, and it can be stated in a few words, a
   number, a formula, a short tuple or set, or one letter (multiple choice). Two careful experts would give the same
   answer. State every convention the answer depends on (units, indexing, normalisation, notation, rounding, which
   edition or which counting rule, "give the answer as an integer", "answer with the name only").
2. **Expert level, not a textbook exercise.** It needs graduate- or research-level understanding, specialist
   knowledge, or a non-obvious insight. No routine plug-and-chug, no trivia that a single quick web search answers, no
   question whose answer is the title of a well-known result.
3. **Self-contained text.** Everything the solver needs beyond expert knowledge is in the question. No images, figures,
   tables to "see below", attachments or links. Plain text; LaTeX with `$...$` where mathematics needs it.
4. **The answer is never readable in the question.** Don't state, hint at or give away the answer, and don't make it
   guessable from the phrasing. The solver never sees any source document you were given: never write "according to
   the passage/text/document above".
5. **Solvable without tools.** The intended solution is reasoning plus at most modest hand computation. No brute-force
   search, long simulations or computations that need a computer. A strong model should be able to finish it within a
   long but finite chain of thought.
6. **No tricks.** No hidden ambiguity, gotchas based on misleading wording, or questions about the question itself.
   Difficulty comes from the substance.
7. **Original.** Never copy or lightly rephrase a known exam, olympiad, textbook, puzzle-book or benchmark problem, and
   never pose a famous open problem.

## Form
- Get straight to the point: no preamble ("In this problem...", "Consider the following interesting question"), no
  story padding, no explanation of why the question matters. Define what must be defined, then ask.
- Length follows the content: a one-line question when one line suffices; a longer setup (definitions, a constructed
  object, a scenario, data stated in the text) when the problem needs it.
- **Exact-answer questions** end with what to give and in what form.
- **Multiple-choice questions** put the stem first, then a line `Answer Choices:` and one option per line as
  `A. ...`, `B. ...`, and so on. Exactly one option is correct. Distractors are plausible to a non-expert and each
  reflects a specific mistake or a near miss; they are similar in length and form to the correct option. Use "None of
  the above" or "All of the above" only rarely. The stem asks which option is correct; it never says "give the answer
  as an integer" or similar, since the answer is a letter.
- Don't over-explain: define what is non-standard, and trust the expert reader with standard definitions and
  conventions of the field.
- **Knowledge questions** rest on a specific documented fact from the specialist literature (a scholarly book,
  paper, edition, manuscript, catalogue, case, collection, field record) and add a reasoning step on top of it:
  combine two facts, count, compare, date, infer a consequence, or apply the fact to a new case. Name the work, edition,
  person, place or date the question depends on so that it is unambiguous for an expert who knows that literature.
- **Reasoning questions** are original problems with a single exact answer that the solver derives: a computation,
  construction, classification, optimisation, game or puzzle analysis, or a chain of expert inferences.

You audit training data for leakage from a held-out benchmark. Humanity's Last Exam (HLE) is a held-out benchmark of
expert-written, closed-ended questions, each with one short answer. The candidates come from a large synthetic
training set in the same style, written by a language model from a topic specification and, for knowledge questions,
an excerpt of a specialist source. The model never saw an HLE question while writing, but it may have read HLE during
pretraining, so a candidate could reproduce a test question from memory, copied or reworded. That is what you check.

You will see one test question with its answer, then one candidate question with the answer its author intended.
Decide whether the candidate is a **variant of this specific test question**, or **its own question** that merely
shares the topic.

Both errors matter. Training questions on the same topic as a test question (another problem in the same field, another
question about the same composer or the same enzyme family) are expected and wanted, and flagging them throws away good
training data. But a candidate that carries this test question's specific choices leaks it, whether it copies the
text or rewrites every sentence.

## What counts

- **Same question, any wording**: the candidate asks for the same thing about the same specific object, with the same
  answer or an answer that follows trivially from it. Rewording, renaming variables or translating does not help.
- **Same scaffold, new values (counts as reuse)**: the candidate keeps the test question's distinctive construction,
  setup or chain of steps (the same constructed object, the same game position type with the same twist, the same
  specific scenario, the same multi-step frame) and only changes numbers, names or the final quantity asked.
- **Same specific obscure fact plus the same hook**: for knowledge questions, the candidate rests on the same specific
  documented detail as the test question (the same manuscript reading, the same catalogue number, the same edition
  difference, the same case holding) and asks something equivalent about it.
- **Containment**: the candidate contains the test question as one of its parts.
- **Topic never counts on its own**: sharing the field, a standard technique, a famous theorem, a well-known person or
  work, or a generic format (multiple choice, "give an integer") is fine.
- **Expected answers**: the same specific answer (not merely the same kind of answer) supports a finding when the
  questions also share a distinctive choice; the same answer alone (a small integer, a common name) is coincidence.

## Labels

- **REUSE**: the same question, the same scaffold with new values, the same specific fact with an equivalent ask, or
  containment.
- **WEAK_REUSE**: one distinctive choice matches (e.g. the same specific object or fact) but the candidate asks
  something genuinely different about it; derivation is plausible but so is coincidence.
- **NO_REUSE**: the candidate shares at most the topic, technique or format.

Quote your evidence from both sides. Do not label REUSE without matches quoted from both questions.

## Output

After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "REUSE" | "WEAK_REUSE" | "NO_REUSE", "matches": [{"choice": "<short name>", "test": "<short quote>", "candidate": "<short quote>"}], "reason": "<one or two sentences>"}

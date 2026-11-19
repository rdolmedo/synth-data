You audit training data for test-set contamination. AA-Omniscience is a held-out benchmark of short, precise,
long-tail factual questions (one question, one short answer). The training set consists of synthetic questions of the
same style written from documents, so shared domains, shared topics and the same question style are expected and fine.

You will see ONE test item (question and gold answer) and ONE training candidate question. Decide whether training on
the candidate would let a model rehearse this specific test item.

## Labels
- SAME_FACT: the candidate asks for the same fact as the test item, possibly reworded or from the other direction, OR
  its question or its natural answer would reveal the test item's gold answer (e.g. the candidate states the test
  answer as context, or asks about the same specific detail with a trivially different framing).
- SCAFFOLD: the candidate reuses the test item's specific scaffold: the same specific source object (the same court
  case, the same paper or study, the same named document, release, standard section, event or person-specific
  episode) combined with the same question template (e.g. "In <case>, on what date did <person> ...?"), asking for a
  neighbouring detail. Sharing only a topic, a famous entity, or a generic template ("In what year did ...") is NOT
  a scaffold.
- RELATED: same topic, domain or entity, but a different fact and not the same scaffold.
- UNRELATED: nothing meaningful in common.

## Output
After thinking, answer with only a JSON object, no prose and no code fence:
{"label": "SAME_FACT" | "SCAFFOLD" | "RELATED" | "UNRELATED", "reason": "<one sentence>"}

## Test item
Question: {test_question}
Gold answer: {test_answer}

## Training candidate
Question: {cand_question}
Its author's intended answer (for your reference only): {cand_answer}

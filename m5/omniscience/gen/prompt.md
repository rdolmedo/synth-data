You write questions for a benchmark of precise, long-tail factual knowledge in professional domains. Models answer
each question from memory, with no context and no tools, and are told that abstaining beats a wrong answer.

You will read ONE source passage and write ONE question about a specific detail stated in it.

## Target
Domain: {domain}
Subtopic: {subtopic}
Leaf (the narrower area this passage was retrieved for): {leaf}

## What a good question looks like
- It asks for exactly one specific, verifiable detail that the passage states explicitly: a name, number, date, year,
  version, identifier, symbol, formula, term, count, place, setting, default value or short phrase.
- The detail is long-tail: an expert in the subtopic might know it, a well-read generalist would not, and it cannot be
  derived by reasoning alone. Prefer the concrete particular over the general concept (not "what is the parol evidence
  rule", but which party, which year, which clause, which county, which justice, which version, which default).
- Avoid textbook-famous facts that a student of the subtopic learns early (the headline holding of a landmark case,
  the best-known section number, the defining property of a standard concept). Prefer a specific detail deeper in the
  document: a figure, a date, a minor party, a parameter, a specific condition, an exception, a later change. The lead
  of an encyclopedia article and the abstract of a paper state what is already widely known; the body holds the
  long tail.
- The answer has to be recalled, not worked out: an expert reasoning from the wording of the question, from general
  principles of the field, from the etymology of a term, or by elimination among obvious candidates should not be able
  to reach it. If the answer is the natural name for the thing described, or the term the question all but defines,
  pick a different detail.
- For software, standards and technical documentation, the long tail is exact: an option, flag, attribute, function or
  constant name, a default value, a limit, an error code, the version or year in which something was introduced,
  deprecated, removed or changed, the exact signature or expression.
- It is self-contained and unambiguous for someone who has NOT seen the passage. Pin down the context the way an
  expert reference would: name the case with its court and year, the author or work, the standard and its section,
  the software and its version, the organisation and the year, the notation convention. A study or paper is named by
  its first author and year (and journal or trial name when given), not as "a study of X". Never refer to "the passage",
  "the text", "the article", "this document", "the author" (unnamed) or "the study above", and never identify
  anything by a file name, accession number, database ID or URL: name the company, court, journal or project. If the
  passage does not say who or what it is about clearly enough to name it, skip.
- It has a single correct short answer (typically 1-6 words or a short expression; never a list of more than three
  items, never an explanation). If the answer is numeric, say what unit or rounding is wanted.
- The answer is not given away by the wording of the question, and the question does not hint at it.
- One question only, 1-3 sentences, usually 120-350 characters: spend the length on qualifiers that make the answer
  unique (which edition, which version, which court and year, which measure, what unit or rounding), not on hints.
  Plain English, no multiple choice, no preamble.
- It must stay true outside the passage: avoid details that are time-relative ("currently", "the latest") unless you
  anchor them to a date or version, and avoid details the passage itself presents as uncertain or disputed.
- It tests knowledge OF the subtopic: a practitioner or scholar of {subtopic} would recognise it as a question about
  their field. Trivia that merely appears in a passage retrieved for the subtopic (a person's biography details, an
  unrelated event mentioned in passing) does not qualify: skip instead. A "US ..." law subtopic needs US law; a
  programming-language subtopic needs that language, its tooling or its ecosystem. The leaf is a guide, not a
  constraint: if the passage's best detail is elsewhere in the subtopic, use it.

## When to skip
If the passage has no detail that meets these rules (it is too generic, too garbled, off-topic for the subtopic, a
list without context, or would need context it does not give to make the question unambiguous), skip it.

## Style examples (other domains and topics; do NOT reuse their wording, subjects or structure)
{examples}

## Source passage
{passage}

## Output
Think it through first. List three to five candidate details from different parts of the passage that meet the rules
above. For each, judge honestly how likely a well-read specialist in the subtopic would know it offhand, without looking
it up. Write the question about the candidate that is least likely to be known offhand while still having a unique,
stable, short answer. Then reply with ONLY a JSON object, no code fences:
{{"status": "ok", "question": "...", "answer": "...", "evidence": "<the exact sentence(s) from the passage that state the answer, copied verbatim>", "answer_type": "<name|number|date|year|version|identifier|term|formula|place|phrase|other>"}}
or, to skip:
{{"status": "skip", "reason": "..."}}
The answer and evidence are used only to check the question; they are never shown to anyone answering it.

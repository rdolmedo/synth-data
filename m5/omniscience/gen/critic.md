You review one question written for a closed-book benchmark of long-tail professional factual knowledge. The person
answering sees ONLY the question: no source, no context, no tools. It was written from a source passage that you do not
see either. Be a strict reviewer: in past audits about a third of such questions had a defect below.

Domain: {domain}
Subtopic: {subtopic}
Question: {question}
Answer the writer intends: {answer}

Hardness and obscurity are NOT defects: the benchmark asks things like the date of a specific email in a named 2009
Fourth Circuit contract case. Work through these steps.

1. Referent. Using only the words of the question, name the single specific thing it asks about (the case, company,
   filing, study, trial, report, book edition, standard version, software release, event, person). If the question
   identifies it only by description ("the randomized trial comparing X with Y", "a 69-year-old man", "the $1.75
   billion credit facility", "research drawing on ...", "in patients with X, what incidence is reported", a 10-K
   identified by a date rather than the company) and an expert could not tell which one exactly, it is ambiguous.
   General facts about a well-defined concept, rule, statute, standard or named work need no further anchor.
2. Uniqueness. Would experts agree on one answer? If different sources, editions, datasets or authorities commonly give
   different figures and the question does not say whose, or several answers are defensible, it is ambiguous (or
   unstable, if the fact changes over time and the question has no date anchor).
3. Field. Would a practitioner or scholar of "{subtopic}" see this as a question about their field? Incidental trivia
   that merely appeared in a source about the subtopic is off_subtopic: biography details, an archive location, page
   ranges or appendix titles of a textbook, settings of an obscure app, a fact belonging to a different field (a
   criminal statute under a tax subtopic, a genetics study under a surgery subtopic, folk dance under religion).
4. Form. not_short: the intended answer is an explanation, interpretation, definition or a list of more than three
   items. garbled: the question is confused or self-contradictory, mixes up two things, does not match the intended
   answer, or the answer can be computed or read off from the question itself.

Reply with ONLY a JSON object, no code fences:
{{"referent": "<what the question is about, in a few words>", "flags": [<zero or more of "ambiguous", "unstable", "off_subtopic", "not_short", "garbled">], "note": "<under 15 words>"}}

You are helping build a taxonomy for synthetic training questions in the style of AA-Omniscience, a benchmark of
short, precise, long-tail factual questions across six professional domains. Each question asks for one specific
detail (a name, number, date, year, version, identifier, symbol, formula, term, place, count or short phrase) that an
expert in the field might know but that is rarely common knowledge. Questions are about 1-3 sentences (median ~180
characters) and pin down their context precisely (the source, edition, version, case, jurisdiction, year, notation).
Answers are short (median ~10 characters). Each synthetic question will be written by reading one real document
passage retrieved for a leaf of this taxonomy.

Domain: {domain}
Subtopic (AA's label, used verbatim in the benchmark's system prompt): {subtopic}

Propose {n} distinct LEAVES below this subtopic. A leaf is a specific sub-area of the subtopic, narrow enough that
documents about it share vocabulary, broad enough to support ~20 different detail questions from different documents
(e.g. for "US Contract Law": "Parol evidence rule and merger clauses"; for "python": "asyncio and concurrency APIs
across CPython 3.x releases"; for "Ancient history": "Ptolemaic Egypt administration and dynastic politics").
Cover the whole breadth of the subtopic as practitioners, textbooks and reference works would define it, including
obscure corners. Leaves must not overlap each other much.{avoid}

For each leaf also give:
- "queries": 4 different BM25 keyword queries (3-8 words each, no boolean syntax) that would retrieve passages rich in
  specific facts about the leaf from {sources_desc}. Vary the queries (different entities, sub-aspects, terms).
- "sources": the 1-3 best-suited source types for the leaf, from: {source_types}.

Return ONLY a JSON array, no prose, of objects: {{"leaf": str, "scope": str (one sentence), "queries": [str x4],
"sources": [str]}}.

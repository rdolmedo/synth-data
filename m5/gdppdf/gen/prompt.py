"""The generator prompt: DeepSeek reads one real document's LiteParse text and writes 2-3 work requests for it.

Only class-level information about GDP.pdf reaches this prompt (domain names, task-type definitions, fitted slot
parameters, brainstormed task families). No test prompt, PDF or rubric is ever shown.
"""
import re

TASK_TYPE_TEXT = {
    'extract_compile': 'Extract and compile: locate specific facts, figures, items or provisions spread over the '
                       'document and compile them (list, table, structured summary) under conditions the requester '
                       'sets (filters, groupings, sorting, exclusions).',
    'calculate': 'Calculate from tables: compute values from figures in the document (changes between periods, '
                 'totals, ratios, payouts, premiums, unit conversions, derived engineering or clinical quantities). '
                 'The inputs must be in the document or the request; the results must not be.',
    'apply_rules': 'Check against rules: apply the document\'s rules, criteria, guideline, specification, policy or '
                   'contract terms to a concrete case the requester describes (eligibility, coverage, compliance, '
                   'clinical or technical decision), reaching a determination for each part.',
    'reconcile_check': 'Reconcile: cross-check items against each other or against stated requirements (a checklist '
                       'in the request, another section of the document, a table vs. the text) and report what is '
                       'missing, inconsistent, superseded or non-compliant.',
    'assess_recommend': 'Assess and recommend: weigh specific evidence from several parts of the document to assess '
                        'a situation, choose among options or recommend a course of action, with the justification '
                        'tied to the document\'s facts.',
}

PITFALL_KINDS = [
    'a superseded, amended or conditional provision that overrides the obvious one',
    'a distractor number: a similar figure for another year, segment, unit, table or scenario',
    'an exclusion, exception or footnote that removes some items from scope',
    'a case fact in the request that changes the outcome (a date, threshold, status or quantity)',
    'a scope limit in the request (only certain items, periods, categories) that must be respected',
    'a tempting inference the document does not support (the answer should not overreach)',
    'an item the request names that does not exist or does not apply in the document (should be flagged, not invented)',
    'a unit, period or basis mismatch between parts of the document',
]

FORMAT_TEXT = {
    'table': 'a table', 'bulleted_list': 'a bulleted list', 'numbered_list': 'a numbered list',
    'short_answer': 'a short direct answer', 'prose_summary': 'a concise prose summary',
    'memo_or_report': 'a short memo or report', 'worked_calculation': 'the calculation shown step by step',
    'mixed': 'a mix (e.g. a table plus a short explanation)',
}

INLINE_TEXT = {
    'case_scenario': 'a concrete case scenario (events, parties, dates, amounts) that the document\'s rules must be '
                     'applied to',
    'client_or_patient_facts': 'facts about a specific client, employee, patient, property or product',
    'added_rules_or_checklist': 'an extra list of requirements, criteria or instructions the requester brings (e.g. '
                                'from their organisation) that the document must be checked against',
    'data_or_list': 'a short list or small table of items/values the requester brings for comparison',
    'other': 'any other context the requester would naturally supply',
}

INTRO = """You write realistic work requests for a dataset of professional document analysis. A professional working \
in {domain} has the PDF document below and asks an AI assistant to do a piece of work with it. The assistant will \
receive your request followed by the document's full extracted text, and must answer in one reply from the document \
(and any facts given in the request) alone, with no tools and no other sources.

Document: {doc_desc}

Read the whole document carefully first. Then write {k} different requests, one for each slot specification below.

Requirements for every request:
1. Grounded. Everything needed to answer is in the document's text or the request's own material. Never invent \
figures, sections, tables, products or rules the document does not contain.
2. Spread. Answering must require combining information from several places of the document that are far apart \
(different sections or pages), not a single paragraph or table.
3. No answers. Never state, hint at or partially compute the answer. Do not quote the passages that contain it. \
Numbers in the request are inputs (case facts, thresholds, targets), never results.
4. Pitfalls. Build in the slot's number of natural pitfalls: places where a careless reader would go wrong. Make \
them come from the document itself (a similar figure elsewhere, an exception, a superseded or conditional rule, a \
footnote, a case fact that changes the outcome), not from extra exclusions in the request. Never \
announce, label or hint at them: no "watch for", "note the exception", "be careful", no naming the clause, table or \
footnote that holds the catch. Use what this document offers, for example: {pitfalls}. Each request must still have one \
defensible correct answer.
5. Real voice. Write as that professional would write to an assistant: their goal, their context, their \
constraints, in their own words, with the slot's length and output format. Refer to the document naturally (the \
attached report, policy, filing, manual, PDF, ...); point to a section, table or page only when a real user would. \
Vary openings and phrasing; do not start with "Using the attached document" or "Based on the document". Vary the \
register as real users do: some formal, some brisk or conversational, the occasional loose phrasing.
6. Text only. The assistant sees only extracted text (with parser artifacts), not images: do not ask about \
charts, photos, maps or drawings whose content is not in the text.
7. Finishable. A careful expert could answer in one reply of at most about 1,500 words: no exhaustive enumerations \
of hundreds of items, no open-ended research.
8. Distinct. The {k} requests must use different parts of the document and different skills; none may paraphrase \
another.
9. Invented case facts (names, dates, amounts) are welcome where a slot asks for inline material. Keep them \
plausible and consistent with the document, so that the document's rules decide the outcome.

{slots}

Return one JSON object and nothing else:
{{"requests": [{{"slot": 1, "task_family": "<the letter you chose>", "family_fit": "good" or "loose", "prompt": "<the request, exactly as \
the professional would send it>", "pages_used": [<the page numbers whose content the answer needs>]}}, ...]}}

If the document cannot support a realistic request from a professional in {domain} (wrong field, too thin, mostly \
unreadable text), return {{"requests": [], "reason": "<why>"}} instead.

--- DOCUMENT TEXT ({n_pages} pages) ---
{doc}"""

SLOT = """Slot {i}:
- Task type: {task_type}
- Task family: pick the one of these that best fits this document and write the request as an instance of it,
  adapting it to what the document actually contains (another role or purpose in the same field is fine). Always
  pick one; report whether the fit is "good" or "loose":
{leaves}
- Distinct asks (questions or deliverables) in the request: {n_asks}
- Pitfalls to build in: {n_pitfalls}
- Inline material from the requester: {inline}
- Request length: about {words} words{long_note}, in natural sentences (never telegraphic)
- Output the requester asks for: {fmt}
- Explicit constraints the request states (exclusions, filters, sorting, limits, "do not ..."): {constraints}
- The requester {persona}"""


def doc_desc(doc):
    bits = [doc.get('title') or 'untitled']
    if doc.get('doc_type'):
        bits.append(doc['doc_type'].replace('_', ' '))
    if doc.get('issuer'):
        bits.append(f'issued by {doc["issuer"]}')
    return '; '.join(bits) + f' ({doc["n_pages"]} pages)'


def slot_text(i, s):
    inline = INLINE_TEXT[s['inline']] if s.get('inline') else 'none (the request is just the ask and its context)'
    long_note = (' (most of it the inline material)' if s.get('inline') and s['words'] >= 150 else '')
    leaves = '\n'.join(f'  ({chr(97 + j)}) {l}' + (f' [{r}]' if r else '') for j, (l, r) in
                       enumerate(zip(s['leaf_candidates'], s.get('leaf_roles') or [None] * 9)))
    return SLOT.format(i=i, task_type=TASK_TYPE_TEXT[s['task_type']], leaves=leaves, n_asks=s['n_asks'],
                       n_pitfalls=s['n_pitfalls'], inline=inline, words=s['words'], long_note=long_note,
                       fmt=FORMAT_TEXT[s['format']],
                       constraints=s['constraints'] if s.get('constraints') is not None else 'as natural',
                       persona=('states their role or situation' if s['persona'] else
                                'does not state their role, job title or organisation; the request is just the ask'))


_RUNS = re.compile(r' {3,}')


def generator_doc_text(pages):
    """The document as the generator reads it: the eval's page markers, with runs of 3+ spaces shortened to two
    (LiteParse pads columns with spaces; this keeps columns apart at about 30% fewer tokens)."""
    return ''.join(f'\n--- Page {i} ---\n{_RUNS.sub("  ", t.rstrip())}\n' for i, t in enumerate(pages, 1))


def build(doc, pages, slots):
    import random
    rng = random.Random(doc['doc_id'])
    pit = rng.sample(PITFALL_KINDS, 4)
    return INTRO.format(domain=doc['domain'], doc_desc=doc_desc(doc), k=len(slots), pitfalls='; '.join(pit),
                        slots='\n\n'.join(slot_text(i, s) for i, s in enumerate(slots, 1)), n_pages=len(pages),
                        doc=generator_doc_text(pages))

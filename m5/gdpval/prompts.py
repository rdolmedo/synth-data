"""Generator prompts (DeepSeek, high effort, one call per problem). Only class-level statistics of GDPval appear here;
no test prompt, file or rubric is ever shown to the generator."""
from __future__ import annotations

FORMAT_NAMES = {'docx': 'Word document (.docx)', 'xlsx': 'Excel workbook (.xlsx)', 'pdf': 'PDF (.pdf)',
                'pptx': 'PowerPoint deck (.pptx)', 'csv': 'CSV file (.csv)', 'md': 'Markdown file (.md)',
                'txt': 'plain-text file (.txt)', 'code': 'source code file (e.g. .py, .sql, .yaml)'}

SYSTEM = """You write realistic professional work assignments for evaluating and training AI agents on real-world
knowledge work. Each assignment is the kind of request a working professional receives from a manager, client or
colleague and completes at a computer in one to a few hours, delivering one or more files (Word, Excel, PDF,
PowerPoint, sometimes CSV, Markdown or code).

The agent that will receive the assignment works alone in an offline Linux sandbox (Python with pandas, openpyxl,
python-docx, python-pptx, reportlab, matplotlib; LibreOffice; Pandoc). It cannot browse the web, ask questions, make
calls or meet anyone. Input files you provide are placed in its home directory, and it saves its deliverables to an
output directory (the harness tells it where, so you never need to mention directories).

What good assignments look like (house style):
- Written in the second person by the person asking for the work. Usually opens by setting the professional's role and
  workplace ("You are a ... at ..."), then the situation and why the deliverable is needed, who will use it, then the
  specific requirements.
- Concrete and self-contained: names of the organisation, people, products, places, dates, figures and constraints the
  work depends on are given (in the prompt or in the input files), so an expert could finish without asking anything.
  Where an expert would apply general professional knowledge (laws, standards, common practice), the prompt may rely on
  it, as a real request would.
- Specific about the deliverable: what files, in which format, what they must contain (sections, tabs, columns, length,
  audience, tone), and any naming. Requirements may be a numbered or bulleted list, or prose.
- Refers to every input file by its exact file name and says what it is and how to use it.
- Realistic messiness is fine (a requester's informal tone, some redundant context), as in real work.
- Scope: substantial but finishable by a strong agent in one sitting: the kind of thing an expert does in 1 to 3 hours.
  Prefer a focused deliverable done well over a sprawling multi-part project. Typical outputs are a 2 to 8 page
  document, a workbook with a few tabs, a 6 to 15 slide deck, or a set of 2 to 3 related files.
- Invent distinctive, realistic names (no "Acme", "ABC Corp", "John Doe", "XYZ"); vary the settings (company size,
  US region or occasionally another country, public or private sector, year 2023 to 2026).

You never write the solution, a rubric, the expected answers or hints about them."""

SYNTH_RULES = """Input files for this assignment are SYNTHETIC: you write a Python script that creates them.
- The script uses only the Python standard library plus openpyxl, python-docx, reportlab, matplotlib, pandas and numpy;
  it writes each file into the current working directory under the exact names you list; it uses a fixed random seed,
  needs no network and runs in under a minute.
- Make the files look like real business material: realistic headers, units, dates, IDs, notes, a cover sheet or
  instructions tab where natural, formatting (bold headers, column widths, number formats), and natural messiness that
  fits the situation (inconsistent spellings or date formats, a few blanks, duplicates, outdated rows, extra columns,
  free-text comments). Plant the issues the task expects a careful professional to notice, without labelling them,
  and do not list them in the assignment text either (a requester may at most say the data has not been cleaned).
- Data must be internally consistent so that a correct deliverable exists: amounts add up where they should, IDs match
  across files, dates make sense.
- The files must NOT contain the answer: no precomputed totals, summaries, conclusions or flags that the task asks for,
  no finished or partially finished version of the deliverable, no notes revealing the planted issues. A template the
  task asks to fill must be left blank where the agent's work goes.
- Size: large enough to be realistic work (typically 20 to 400 rows per table, 1 to 6 pages per document), small enough
  to read in full."""

REAL_RULES = """The main input file for this assignment is a REAL public document, given below as extracted text. It
will be placed in the agent's home directory under the file name shown. Write an assignment in which this professional
must actually use the document's content: requirements that a real requester would ask for, whose correct completion
depends on facts, figures, rules or details found in the document (ideally from several different parts of it). Do not
copy the document's key content into the prompt; the agent must read the document. Do not misrepresent what the
document is; you may place it in a fictional organisational situation (the organisation received, is subject to, or is
reviewing this document)."""

COMPANION_RULES = """In addition to the real document, the assignment also has SYNTHETIC companion input files (listed
below), which you create with a Python script under the same rules as any synthetic input:
""" + SYNTH_RULES.split('\n', 1)[1]

OUTPUT_FORMAT = """Reply in exactly this format (the tags on their own lines):

<task_prompt>
(the full assignment text, as the requester would write it)
</task_prompt>
<input_files>
(one line per SYNTHETIC input file you create with the script, never the real document if there is one: "<file name> | <one-line description>"; write NONE if there are
no synthetic input files)
</input_files>
<script>
(the complete Python script that creates the synthetic input files; write NONE if there are none)
</script>"""


def spec_text(spec: dict) -> str:
    fm = [FORMAT_NAMES.get(f, f) for f in spec['formats']]
    lines = [
        f"Occupation of the professional: {spec['occupation']} (sector: {spec['sector']})",
        f"Work activity: {spec['leaf']['name']}: {spec['leaf']['description']}",
        f"Kind of work product: {spec['task_type_desc']}",
        f"Deliverable file format(s): {', '.join(fm)}" + (
            f" ({spec['n_deliverables']} separate files)" if spec['n_deliverables'] > 1 else ' (one file)'),
        f"Target length of the assignment text: about {spec['target_words']} words (stay within 20% of it)",
    ]
    if spec.get('web_refs'):
        lines.append('Public sources: as real requests often do, point the professional to public material by name '
                     'or URL (a regulation, standard, agency or vendor web page, published guidance or data source) '
                     'that a careful expert would consult. The agent is offline and cannot open them, so the task must '
                     'still be completable from the provided material and general professional knowledge; do not '
                     'say that the sources are unavailable.')
    if spec.get('template_input'):
        lines.append('One of the input files is a template, form or partly completed workbook/document from the '
                     'organisation that the professional must complete or update (its blanks are where the work goes).')
    if spec.get('setting'):
        lines.append(f"Setting: {spec['setting']}")
    if spec['lane'] == 'none':
        lines.append('Input files: none. Everything the agent needs must be in the assignment text itself.')
    elif spec['lane'] == 'synthetic':
        lines.append(f"Input files: {spec['n_inputs']} synthetic file(s) of these types: "
                     f"{', '.join(spec['input_exts'])}.")
    else:
        lines.append(f"Input files: the real document below" + (
            f", plus {len(spec['input_exts'])} synthetic companion file(s) of these types: "
            f"{', '.join(spec['input_exts'])}" if spec.get('input_exts') else ' (no other input files)') + '.')
    if spec.get('open'):
        lines[1] = 'Work activity: your choice, any realistic file-producing desk work of this occupation'
    return '\n'.join(lines)


def user_prompt(spec: dict, doc: dict | None = None) -> str:
    parts = ['Write one assignment for this specification:\n', spec_text(spec), '']
    if spec['lane'] == 'synthetic':
        parts += [SYNTH_RULES, '']
    elif spec['lane'] == 'real':
        parts += [REAL_RULES, '']
        if spec.get('input_exts'):
            parts += [COMPANION_RULES, '']
        parts += [f"Real document file name: {doc['name']}" + (f" ({doc['pages']} pages)" if doc.get('pages') else ''),
                  '<document_text>', doc['text'], '</document_text>', '']
    parts += ['Before writing, think through a concrete, specific scenario that fits the specification (not the most '
              'obvious generic one), what an expert would need to produce, and that everything needed is provided.', '',
              OUTPUT_FORMAT]
    return '\n'.join(parts)


def parse(reply: str) -> dict:
    """-> {'task_prompt', 'input_files': [(name, desc)], 'script'}; raises ValueError on a malformed reply."""
    import re

    def tag(t):
        m = re.findall(rf'<{t}>\s*\n?(.*?)\n?\s*</{t}>', reply, re.S)
        if not m:
            raise ValueError(f'missing <{t}>')
        return m[-1].strip()
    prompt = tag('task_prompt')
    files_raw = tag('input_files')
    script = tag('script')
    files = []
    if files_raw.strip().upper() != 'NONE':
        for ln in files_raw.splitlines():
            ln = ln.strip().lstrip('-* ').strip()
            if not ln:
                continue
            name, _, desc = ln.partition('|')
            name = name.strip().strip('`"\'')
            if name:
                files.append((name, desc.strip()))
    if script.strip().upper() == 'NONE':
        script = ''
    script = re.sub(r'^```(?:python)?\s*\n', '', script)
    script = re.sub(r'\n```\s*$', '', script)
    return {'task_prompt': prompt, 'input_files': files, 'script': script}

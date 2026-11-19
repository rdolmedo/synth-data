"""Brainstorm the taxonomy's breadth below GDPval's 44 occupations with Opus (#358): work activities per occupation.

No test item reaches these calls: Opus sees the occupation, its sector and the class vocabularies only.

Stages (cached, resumable):
  leaves   per occupation: a first list of work activities, one "what is still missing" pass, then a cleaning call
           (merge overlaps, drop audio/video-only or offline-infeasible work); each leaf is tagged with the task types
           it fits, its deliverable formats and its typical inputs (none / private business files / public documents)

  python -m m5.gdpval.brainstorm leaves     -> WORK/leaves/<occupation slug>.json, WORK/leaves.json
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from m5.gdpval.opus import WORK, ask_json
from m5.gdpval.vocab import OCC_SECTOR, OCCUPATIONS, PROVENANCE, TASK_TYPES

FORMATS = ['docx', 'xlsx', 'pdf', 'pptx', 'csv', 'md', 'txt', 'code']

SYSTEM = """You help design the taxonomy of a generator of synthetic training tasks for AI agents doing professional
knowledge work. Each task is a realistic assignment that a working professional in a given occupation would receive
from a manager, client or colleague, and that an expert completes in one to a few hours at a computer, producing one or
more files (Word, Excel, PDF, PowerPoint, sometimes CSV, Markdown or code). Tasks may come with input files (data
exports, spreadsheets, forms, documents, regulations, reports) or with none (all needed facts are in the request).

The agent works alone in an offline Linux sandbox with Python, LibreOffice and document libraries. It cannot browse
the web, make phone calls, meet people, or handle physical objects, images, audio or video as core material.

The taxonomy entries are categories of work, never individual tasks: each must fit many different concrete tasks."""

LEAF_FIELDS = '''Each work activity is an object:
{"name": "<3-8 word name>", "description": "<one sentence: what the professional produces and why>",
 "task_types": [<1-3 of the task types below that this activity typically yields>],
 "formats": [<deliverable formats, from: ''' + ', '.join(FORMATS) + '''>],
 "inputs": [<typical inputs, any of: "none", "private_business", "public_published">]}

Task types:
''' + '\n'.join(f'- {k}: {v}' for k, v in TASK_TYPES.items()) + '''

Input kinds:
- none: the request itself carries all needed facts
''' + '\n'.join(f'- {k}: {v}' for k, v in PROVENANCE.items() if k != 'mixed_other')


def slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:60]


def first(occ: str):
    user = f'''Occupation: {occ} (sector: {OCC_SECTOR[occ]}).

List 45 distinct work activities in which a {occ} working in this sector produces a file deliverable at a computer.
Cover the occupation's whole range of desk work: routine and occasional, analytical and drafting, internal and
client-facing, small and larger assignments, different employers and settings within the sector. Keep each one broad
enough for many different tasks (no single scenario, no named companies, no specific numbers). Leave out work whose
core material is images, audio or video, and work that needs live web research, phone calls or physical presence.

{LEAF_FIELDS}

Reply with one ```json block: a list of 45 objects.'''
    return ask_json(SYSTEM, user, f'leaves1-{slug(occ)}')


def missing(occ: str, cur: list):
    names = '\n'.join(f'- {x["name"]}: {x["description"]}' for x in cur)
    user = f'''Occupation: {occ} (sector: {OCC_SECTOR[occ]}).

Current work activities:
{names}

What common kinds of file-producing desk work of a {occ} are still missing? Think about other employers and settings
within the sector, other audiences, periodic vs one-off work, and the deliverable formats (Word, Excel, PDF,
PowerPoint) that are under-represented. Propose up to 15 new activities that do not overlap the current ones, with
the same rules (broad categories, no audio/video or web research as core).

{LEAF_FIELDS}

Reply with one ```json block: a list of objects (may be empty).'''
    return ask_json(SYSTEM, user, f'leaves2-{slug(occ)}')


def clean(occ: str, cur: list):
    user = f'''Occupation: {occ} (sector: {OCC_SECTOR[occ]}).

Candidate work activities (JSON):
{json.dumps(cur, indent=0)}

Clean this list:
1. Merge activities that overlap so much that a typical task would fit both; keep the broader name.
2. Drop activities that are not realistic for this occupation in this sector, need images/audio/video as core
   material, need live web research, or are too narrow to fit many different tasks.
3. Fix tags: task_types (1-3 from the vocabulary), formats, inputs.
4. Keep everything else unchanged.

{LEAF_FIELDS}

Reply with one ```json block: the cleaned list.'''
    return ask_json(SYSTEM, user, f'leaves3-{slug(occ)}')


def valid(x) -> bool:
    return (isinstance(x, dict) and x.get('name') and x.get('description')
            and all(t in TASK_TYPES for t in x.get('task_types') or [])
            and x.get('task_types') and x.get('formats'))


def leaves_one(occ: str):
    out = WORK / 'leaves' / f'{slug(occ)}.json'
    if out.exists():
        return json.loads(out.read_text())
    a = [x for x in first(occ) if valid(x)]
    b = [x for x in missing(occ, a) if valid(x)]
    c = [x for x in clean(occ, a + b) if valid(x)]
    for x in c:
        x['formats'] = [f for f in x['formats'] if f in FORMATS] or ['docx']
        x['inputs'] = [i for i in x.get('inputs') or [] if i in ('none', 'private_business', 'public_published')] \
            or ['none']
    res = {'occupation': occ, 'sector': OCC_SECTOR[occ], 'n_first': len(a), 'n_missing': len(b), 'leaves': c}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f'{occ}: {len(a)} + {len(b)} -> {len(c)}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['leaves'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    occs = [o for o in OCCUPATIONS if not a.only or a.only.lower() in o.lower()]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(leaves_one, occs))
    if not a.only:
        (WORK / 'leaves.json').write_text(json.dumps({r['occupation']: r['leaves'] for r in res}, indent=1))
        print(sum(len(r['leaves']) for r in res), 'leaves')


if __name__ == '__main__':
    main()

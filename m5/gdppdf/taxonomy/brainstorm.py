"""Brainstorm the leaves of the taxonomy: professional task families per domain, with Opus via `claude -p` (the
user's subscription, ~10 calls). Only class-level inputs: the domain name, the broad document types and the task
types. No GDP.pdf item is shown.

  python -m m5.gdppdf.taxonomy.brainstorm --out m5/gdppdf/taxonomy/leaves.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from m5.gdppdf.docs.classify import DOC_TYPES, DOMAINS
from m5.gdppdf.taxonomy.label_requests import TASK_TYPES

WORK = Path('/data/aai/m5/gdppdf/work/opus')
MODEL = 'claude-opus-5-5'

SYSTEM = 'You are an expert in professional work across industries. You answer with JSON only.'

USER = """We are building a dataset of realistic requests that professionals send to an AI assistant together with \
one long, real PDF document (a report, filing, policy, manual, guideline, contract, dataset, record ...). The \
assistant must answer from the document alone, in text (it cannot see images).

Domain: {domain}

Brainstorm {n} distinct, concrete task families for this domain. A task family is a recurring kind of work a \
specific professional does with a document, e.g. "check a subcontractor's pay application against the contract's \
retainage and change-order clauses" or "tabulate dose adjustments for renal impairment across a drug label's \
sections" (these two are only illustrations of the level of detail; do not reuse them).

Cover:
- many different professional roles and sub-fields of {domain} (breadth matters more than depth);
- all of these task types: {task_types};
- the document types a professional in {domain} actually receives: {doc_types}.

Each family must be doable from a document's text alone and answerable in one reply. Avoid families that need \
figures, photos, maps or drawings, outside data, or legal/medical advice beyond what a document states.

Return a JSON list; each item: {{"leaf": "<task family, 8-20 words>", "role": "<who asks>", "doc_types": [<1-3 of \
the document type keys>], "task_types": [<1-2 of the task type keys>]}}.

Document type keys: {doc_keys}
Task type keys: {task_keys}"""


def call(user, tag, effort='high', timeout=3600):
    key = hashlib.sha256(json.dumps([MODEL, effort, SYSTEM, user]).encode()).hexdigest()[:16]
    out = WORK / f'{tag}-{key}.json'
    if out.exists():
        return json.loads(out.read_text())['result']
    WORK.mkdir(parents=True, exist_ok=True)
    sysf = WORK / f'{tag}-{key}.system.txt'
    sysf.write_text(SYSTEM)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE')}
    env['DISABLE_AUTOUPDATER'] = '1'
    for attempt in range(3):
        t0 = time.time()
        r = subprocess.run(['claude', '-p', '--model', MODEL, '--effort', effort, '--system-prompt-file', str(sysf),
                            '--tools', '', '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json'],
                           input=user, capture_output=True, text=True, env=env, timeout=timeout)
        try:
            res = json.loads(r.stdout)
        except ValueError:
            res = {'is_error': True, 'result': (r.stderr or r.stdout)[-500:]}
        if not res.get('is_error') and res.get('result'):
            out.write_text(json.dumps({'tag': tag, 'user': user, 'result': res['result'], 'usage': res.get('usage'),
                                       'models': list(res.get('modelUsage') or {}), 'seconds': round(time.time() - t0)}))
            print(f'{tag} ok {time.time() - t0:.0f}s', flush=True)
            return res['result']
        print(f'{tag} attempt {attempt} failed: {str(res.get("result"))[:200]}', flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(tag)


USER_CELL = """We are building a dataset of realistic requests that professionals send to an AI assistant together with \
one long, real PDF document. The assistant must answer from the document's text alone (it cannot see images).

Domain (the requester's field): {domain}
Document type: {doc_type} ({doc_desc})

List {n} distinct task families for this cell: recurring kinds of work that a professional in {domain} does with a \
document of this type. Each must apply to MOST documents of this type (it names the kind of content it works on, \
not one specific document), and be concrete about the work: what is located, computed, checked or decided, and for \
what purpose. Example of the level of detail (another cell; do not reuse): "tabulate dose adjustments by renal \
function across a drug label's dosing, warnings and pharmacology sections for a pharmacy protocol".

Cover several different roles in {domain}, and these task types: {task_types}. Avoid families that need figures, \
photos, maps or drawings, outside data, or advice beyond what a document states.

Return a JSON list; each item: {{"leaf": "<task family, 10-25 words>", "role": "<who asks>", "task_types": [<1-2 \
of: {task_keys}>]}}."""


def parse_list(text):
    m = re.search(r'\[.*\]', text, re.S)
    return json.loads(m.group(0))


def cells(a):
    """v2: leaves per (domain x document type) cell, generic to the cell's documents."""
    def one(cell):
        d, t = cell
        user = USER_CELL.format(domain=d, doc_type=t, doc_desc=DOC_TYPES[t], n=a.n,
                                task_types='; '.join(f'{k} ({v})' for k, v in TASK_TYPES.items()),
                                task_keys=', '.join(TASK_TYPES))
        tag = 'cell-' + re.sub(r'\W+', '_', f'{d}-{t}').strip('_').lower()
        try:
            items = parse_list(call(user, tag))
        except Exception as e:
            print(f'{tag}: {e}', flush=True)
            return cell, []
        ok = [dict(it, doc_types=[t]) for it in items if it.get('leaf') and it.get('task_types')
              and set(it['task_types']) <= set(TASK_TYPES)]
        return cell, ok

    todo = [(d, t) for d in DOMAINS for t in DOC_TYPES]
    with cf.ThreadPoolExecutor(6) as ex:
        res = dict(ex.map(one, todo))
    out = {d: [l for t in DOC_TYPES for l in res[(d, t)]] for d in DOMAINS}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print({d: len(v) for d, v in out.items()}, 'empty cells:', [c for c, v in res.items() if not v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--cells', action='store_true', help='v2: per domain x document type')
    a = ap.parse_args()
    if a.cells:
        return cells(a)

    def one(d):
        user = USER.format(domain=d, n=a.n, task_types='; '.join(f'{k} ({v})' for k, v in TASK_TYPES.items()),
                           doc_types='; '.join(f'{k} ({v})' for k, v in DOC_TYPES.items()),
                           doc_keys=', '.join(DOC_TYPES), task_keys=', '.join(TASK_TYPES))
        items = parse_list(call(user, 'leaves-' + re.sub(r'\W+', '_', d).strip('_').lower()))
        ok = [it for it in items if it.get('leaf') and set(it.get('doc_types') or []) <= set(DOC_TYPES)
              and set(it.get('task_types') or []) <= set(TASK_TYPES) and it.get('doc_types') and it.get('task_types')]
        return d, ok

    with cf.ThreadPoolExecutor(5) as ex:
        res = dict(ex.map(one, DOMAINS))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print({d: len(v) for d, v in res.items()})


if __name__ == '__main__':
    main()

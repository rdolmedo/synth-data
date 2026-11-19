"""Label the GDPval test tasks with the taxonomy's class-level dimensions (Opus, analysis only).

The labels feed only class-level statistics (counts per sector) into the sampler; no test text reaches the generator.

  python -m m5.gdpval.label_test     -> WORK/test_labels.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from m5.gdpval.opus import WORK, ask_json
from m5.gdpval.vocab import INPUT_ROLES, PROVENANCE, TASK_TYPES

PARQUET = '/data/aai/aa_evals/data/gdpval/hf_11e7900/data/train-00000-of-00001.parquet'

SYSTEM = """You classify professional work tasks for a statistics table. For each task you get its prompt, the names of
its input files and the names of the files a human expert delivered. Answer with labels only, from the given
vocabularies. Be consistent across tasks."""


def vocab_text() -> str:
    t = 'task_type (the main kind of work product; pick one primary, optionally one secondary):\n'
    t += '\n'.join(f'- {k}: {v}' for k, v in TASK_TYPES.items())
    t += '\n\ninput role (per input file, what the task does with it):\n'
    t += '\n'.join(f'- {k}: {v}' for k, v in INPUT_ROLES.items())
    t += '\n\ninput provenance (per input file, what kind of document it is):\n'
    t += '\n'.join(f'- {k}: {v}' for k, v in PROVENANCE.items())
    return t


def batch_prompt(rows) -> str:
    parts = [vocab_text(), '\nTasks:']
    for r in rows:
        refs = [p.split('/')[-1] for p in r['reference_files']]
        dels = [p.split('/')[-1] for p in r['deliverable_files']]
        parts.append(f"\n=== task {r['task_id'][:8]} ({r['occupation']})\ninput files: {refs}\nexpert deliverable "
                     f"files: {dels}\nprompt:\n{r['prompt']}\n")
    parts.append('''
Reply with one ```json block: a list with one object per task, in order:
{"id": "<8-char id>", "task_type": "<primary>", "task_type_2": "<secondary or null>",
 "deliverable_formats": ["pdf"|"docx"|"xlsx"|"pptx"|"csv"|"md"|"txt"|"code"|"image"|"audio_video"|"other", ...] (what the
   prompt asks the agent to deliver; use the expert files as a hint),
 "n_deliverables": <number of separate files requested>,
 "inputs": [{"file": "<name>", "role": "<input role>", "provenance": "<provenance>"}, ...] (one per input file; [] if none),
 "needs_web": true|false (the task as written relies on looking things up online: URLs to consult, current prices,
   regulations to fetch),
 "work_activity": "<5-12 word generic description of the work activity, no names, no numbers>"}''')
    return '\n'.join(parts)


def main():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    rows = [dict(task_id=r.task_id, occupation=r.occupation, sector=r.sector, prompt=r.prompt,
                 reference_files=list(r.reference_files), deliverable_files=list(r.deliverable_files))
            for r in df.itertuples()]
    batches = [rows[i:i + 10] for i in range(0, len(rows), 10)]

    def one(b):
        return ask_json(SYSTEM, batch_prompt(b), 'label', effort='high')

    with ThreadPoolExecutor(int(os.environ.get('LABEL_WORKERS', '6'))) as ex:
        outs = list(ex.map(one, batches))
    labels = {}
    for b, o in zip(batches, outs):
        by = {x['id']: x for x in o}
        for r in b:
            x = by.get(r['task_id'][:8])
            if x is None:
                print('missing label', r['task_id'], file=sys.stderr)
                continue
            labels[r['task_id']] = {**x, 'sector': r['sector'], 'occupation': r['occupation'],
                                    'n_words': len(r['prompt'].split()),
                                    'ref_ext': [Path(p).suffix.lower() for p in r['reference_files']],
                                    'del_ext': [Path(p).suffix.lower() for p in r['deliverable_files']]}
    (WORK / 'test_labels.json').write_text(json.dumps(labels, indent=1))
    print(len(labels), 'labels ->', WORK / 'test_labels.json')


if __name__ == '__main__':
    main()

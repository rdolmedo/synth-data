"""The GDPval gold set (openai/gdpval) and AA's task prompt.

Data: `openai/gdpval@11e7900` downloaded to /data/aai/aa_evals/data/gdpval/hf_11e7900 (not committed; the
public eval set stays out of training data). 220 tasks, 9 sectors, 44 occupations; each task has a prompt, 0+
reference files and the human expert's deliverable files.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from aa_evals.gdpval.aa_prompts import SYSTEM_PROMPT, TASK_TEMPLATE

DATA = Path(os.environ.get('AA_GDPVAL_DATA', '/data/aai/aa_evals/data/gdpval/hf_11e7900'))
REVISION = '11e7900cdcac61bc4daf59e65feb238acda98fbf'
PARQUET = 'data/train-00000-of-00001.parquet'
N_TASKS = 220
HOME = '/home/user'
REF_DIR = HOME   # where reference files land, as Stirrup's upload puts them (the sandbox home)

# Text-only task set (#346, 2026-11-11). The models we fine-tune are text-only and every model is served without
# vision, so tasks whose core input or deliverable is an image, audio or video are left out of new runs. Our judge
# sees audio/video as metadata only (render.py), so those were judged blind anyway. Classification and per-task
# signals: /data/aai/aa_evals/audits/gdpval_textonly_20261111/ (classification.json, feat.py).
# 13 tasks: 9 enter Elo, 4 have no expert deliverable.
TEXT_ONLY_EXCLUDED = {
    'f2986c1f-2bbf-4b83-bc93-624a9d617f45': 'identify medications from a photo',
    '57b2cdf2-ad62-4591-aa91-aad489740320': 'check the report against surveillance photos',
    'a46d5cd2-55fe-48fa-a4c6-6aaf6b9991b5': 'check the report against photos of the house and vehicles',
    'a45bc83b-22f9-4def-8d89-9c5661b2b86f': 'understand and restyle an architecture diagram',
    'fe0d3941-e32c-4bf1-a643-b566d2b4cb3c': 'device schematic shown only as an image',
    'a73fbc98-90d4-4134-a54f-2b1d0c838791': 'assign vendor tables from an arena layout drawing',
    '75401f7c-396d-406d-b08e-938874ad1045': 'cut a show reel from 13 video clips',
    'e222075d-5d62-4757-ae3c-e34b0846583b': 'MP4 deliverable from online stock footage',
    '38889c3b-e3d4-49c8-816a-3cc8e5313aba': 'drum reference track in, WAV stems out',
    '45c6237b-f9c9-4526-9a8d-6a5c404624ec': 'product photos in the order list (no expert file)',
    'a941b6d8-4289-4500-b45a-f8e4fc94a724': 'composite an actor between two video shots (no expert file)',
    'ff85ee58-bc9f-4aa2-806d-87edeabb1b81': 'resync a sax track to a mix (no expert file)',
    '4b894ae3-1f23-4560-b13d-07ed1132074e': 'mix a bass stem (no expert file)',
}
TASK_SETS = ('text-only-v1', 'all')
DEFAULT_TASK_SET = 'text-only-v1'


@dataclass
class Task:
    task_id: str
    sector: str
    occupation: str
    prompt: str
    reference_files: list[str] = field(default_factory=list)    # dataset-relative paths
    deliverable_files: list[str] = field(default_factory=list)  # the human expert's files

    def ref_targets(self) -> list[tuple[Path, str]]:
        """(host source, sandbox path) per reference file. Basenames are kept (as Stirrup's upload does); a
        repeated basename gets its dataset hash dir as a prefix so no file is lost."""
        names = [Path(r).name for r in self.reference_files]
        out = []
        for r, n in zip(self.reference_files, names):
            dest = n if names.count(n) == 1 else f'{Path(r).parent.name}_{n}'
            out.append((DATA / r, f'{REF_DIR}/{dest}'))
        return out


def parquet_sha() -> str:
    return hashlib.sha256((DATA / PARQUET).read_bytes()).hexdigest()


def load_tasks() -> list[Task]:
    import pandas as pd
    df = pd.read_parquet(DATA / PARQUET)
    tasks = [Task(r.task_id, r.sector, r.occupation, r.prompt, list(r.reference_files), list(r.deliverable_files))
             for r in df.itertuples()]
    assert len(tasks) == N_TASKS, len(tasks)
    missing = [p for t in tasks for p in t.reference_files + t.deliverable_files if not (DATA / p).exists()]
    assert not missing, f'{len(missing)} files missing, e.g. {missing[:3]}'
    return tasks


def task_set(tasks: list[Task], name: str) -> list[Task]:
    """The tasks of a named set, in the given order. 'all' is the full 220; 'text-only-v1' drops TEXT_ONLY_EXCLUDED."""
    if name == 'all':
        return list(tasks)
    if name == 'text-only-v1':
        return [t for t in tasks if t.task_id not in TEXT_ONLY_EXCLUDED]
    raise ValueError(f'unknown task set {name!r}; one of {TASK_SETS}')


def stratified(tasks: list[Task]) -> list[Task]:
    """A fixed order that interleaves sectors, so `--limit N` is a spread sample."""
    by: dict[str, list[Task]] = {}
    for t in sorted(tasks, key=lambda t: hashlib.sha256(t.task_id.encode()).hexdigest()):
        by.setdefault(t.sector, []).append(t)
    out, i = [], 0
    while len(out) < len(tasks):
        for s in sorted(by):
            if i < len(by[s]):
                out.append(by[s][i])
        i += 1
    return out


def task_prompt(task: Task) -> str:
    """AA's Task Submission Prompt with the task and its reference paths filled in."""
    text = TASK_TEMPLATE
    note = '(This section appears only when the task includes reference files.)\n\n'
    assert note in text
    text = text.replace(note, '')
    refs = task.ref_targets()
    if refs:
        text = text.replace('- [absolute path to each reference file]', '\n'.join(f'- {p}' for _, p in refs))
    else:
        start = text.index('## Reference Files Location')
        end = text.index('## Completing Your Work')
        text = text[:start] + text[end:]
    assert '[task description]' in text
    return text.replace('[task description]', task.prompt)


def system_prompt() -> str:
    return SYSTEM_PROMPT

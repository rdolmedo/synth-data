"""The pinned GDP.pdf task set: 100 tasks, 100 PDFs (4,592 pages), 1,275 rubric criteria.

Surge AI's official dataset `surgeai/GDP.pdf` went private or was deleted between 2026-07-15 (last public
Wayback capture) and 2026-10-11. Four independent community mirrors (sleepyheeler, tolo6474, Oitaaa,
SigridYiiiiiii) hold byte-identical copies of `data.parquet` and all 100 PDFs, and the copy matches AA's
published counts exactly (100 tasks, 4,592 pages, 1,275 criteria). A fifth mirror (cloudybean) changed the
parquet and two PDFs and is not used. We pin tolo6474/GDP.pdf at one commit and check the hashes on load.

The data is CC-BY-4.0 and carries a canary GUID; it never goes into git or into training data.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path('/data/aai/aa_evals/data/gdppdf')
HF_REPO = 'tolo6474/GDP.pdf'
HF_REVISION = '90feac141ae15abbcb1d725f6282cd83e40ea077'
HF_DIR = DATA_ROOT / 'hf_90feac14'
PARQUET_SHA256 = '2ba18b4facc482a3520a6a6763f0294b0f8f3369f9cf4375b17af8fc0e025704'
CANARY = '329d96c2-986f-486e-81db-4b6bd8539ead'

# AA's published task-set size; a mismatch means the data is not the set AA evaluates.
EXPECTED = {'tasks': 100, 'criteria': 1275, 'pages': 4592}

# Text-only task set (#348; the user approved all 25 exclusions on 2026-11-11). Every model is served without vision,
# and GDP.pdf sends the LiteParse text alone. These tasks' answers sit in a figure, map, plan, photo or handwriting
# that the text does not carry (or OCR leaves unreadable). New runs draw from the 75 others (--task-set text-only-v1,
# the default); --task-set all keeps the 100. Classification, per-task signals and the M3.6 re-score:
# /data/aai/aa_evals/audits/gdppdf_textonly_20261111/ (feat.py, signals.json, classification.json).
# 'clear': the core answer is visual; 'borderline': read as visual after review.
TEXT_ONLY_EXCLUDED = {
    'aff4bf86-58c3-4460-a7d6-e4df28f74b02': ('clear', 'list the centered features of a one-page engineering drawing'),
    'fa00520d-bb0d-403a-9964-4a9f42c08264': ('clear', 'improvements and dates handwritten on a scanned form (OCR unreadable)'),
    '56a40a31-5bc3-4dd4-9f9e-3a5954b6b575': ('clear', 'life expectancy by sex only in a chart'),
    '8b7a7757-3ac9-4a31-9207-1571ad837acc': ('clear', 'peak rates read off line charts (graded with ranges)'),
    '48e0e149-282d-4a54-b195-42f7f8b57521': ('clear', 'count lakes on a wetlands map'),
    '6ed541f1-4832-444a-8d62-bc1e6f84cc8f': ('clear', 'areas inside a drive-time boundary on a map'),
    '9a33920e-63d0-4164-a19d-a21f815a7e56': ('clear', 'readings from bar and line charts'),
    'f8e8f09e-16f8-4bd6-9f93-a762fe197105': ('clear', 'arrow-annotated panel of a figure'),
    '9cf542c7-eda7-4297-a58e-137b9fd38931': ('clear', 'button row in a photo of an instrument'),
    '31fddf6a-85cd-4461-8126-78546c3e5cae': ('clear', 'count light-fixture symbols on floor plans'),
    '2a74d072-a8fb-4962-a6ef-77b1fca4df7b': ('clear', 'trace a wire on a scanned wiring diagram'),
    '6070d5f8-d8f0-4557-aa60-fde07763513f': ('clear', 'lots next to ponds on a site plan'),
    'b69eac3c-8531-46e8-98e8-0be36675e6c5': ('clear', '4 of 10 criteria read a bar chart with no values in the text'),
    'fc62a5ad-f57a-4b37-8f44-1b049b3d3d60': ('clear', 'edges and edge colours of a network figure'),
    'dead128f-4662-456a-b07f-a0fa879aceeb': ('clear', 'values read off a stock performance graph'),
    'cc1c6b05-8470-47b3-a474-a6a0e4f79b97': ('clear', 'streams on a process flow diagram (stream table is an image)'),
    '39305def-757a-49aa-9839-dd05c776a972': ('clear', 'divergence between curves in a figure'),
    '678d01ed-d88e-40ae-a4ed-50bdf011ad28': ('clear', 'above-ceiling obstructions on MEP drawings'),
    '07bd6ec8-f0c4-40db-b0c7-8eead030b772': ('clear', 'blow counts on a handwritten field record'),
    '95696d8f-3993-4640-97c8-a4232bb9c96b': ('borderline', 'key table is a rotated scan whose OCR is unreadable'),
    '0d60f331-d255-414c-aef9-64e608b794f2': ('borderline', '6 of 20 criteria are part numbers only in an exploded view'),
    'e2129d8a-d96b-4726-b0c3-f07c6e0b11b0': ('borderline', 'socket counts per room from electrical-plan symbols'),
    'bc7a2336-04fa-48aa-beb4-5f3df7417de3': ('borderline', 'kitchen layout from a floor plan (shape, dimensions)'),
    '14b56b44-c251-47a2-9de9-dd80d09899e2': ('borderline', 'window counts per floor from plans'),
    'efb22c1d-4a15-4f00-98a7-f176e7f8d6ec': ('borderline', 'photos only (the prompt is also broken)'),
}
TASK_SETS = ('text-only-v1', 'all')
DEFAULT_TASK_SET = 'text-only-v1'


def task_set(tasks, name):
    """The tasks of a named set, in the given order: 'all', or 'text-only-v1' (drops TEXT_ONLY_EXCLUDED)."""
    if name == 'all':
        return list(tasks)
    if name == 'text-only-v1':
        return [t for t in tasks if (t.task_id if hasattr(t, 'task_id') else t) not in TEXT_ONLY_EXCLUDED]
    raise ValueError(f'unknown task set {name!r}; one of {TASK_SETS}')


_RUBRIC_RE = re.compile(r'rubric\s*-\s*(\d+)\.\s*criterion', re.IGNORECASE)


@dataclass
class Criterion:
    id: str
    text: str
    meta: dict = field(default_factory=dict)   # type / severity / implicitness / subjectiveness / failure_mode


@dataclass
class Task:
    task_id: str
    domain: str
    prompt: str
    pdf_path: Path
    criteria: list

    @property
    def pdf_name(self):
        return self.pdf_path.name


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_tasks(hf_dir=HF_DIR, verify=True):
    """All 100 tasks in parquet order, with rubric criteria ordered by their number."""
    import pandas as pd
    parquet = Path(hf_dir) / 'data.parquet'
    if verify and sha256(parquet) != PARQUET_SHA256:
        raise RuntimeError(f'{parquet} does not match the pinned sha256 {PARQUET_SHA256}')
    df = pd.read_parquet(parquet)
    tasks = []
    for row in df.to_dict('records'):
        found = []
        for col, val in row.items():
            m = _RUBRIC_RE.fullmatch(str(col).strip())
            if m and isinstance(val, str) and val.strip():
                n = int(m.group(1))
                prefix = str(col).strip()
                meta = {k[len(prefix) + 1:]: row[k] for k in row
                        if str(k).startswith(prefix + '_') and row[k] not in (None, '')}
                found.append((n, Criterion(id=f'r{n}', text=val, meta=meta)))
        found.sort(key=lambda p: p[0])
        tasks.append(Task(task_id=str(row['task_id']), domain=str(row['domain']), prompt=str(row['prompt']),
                          pdf_path=Path(hf_dir) / row['pdf_path'], criteria=[c for _, c in found]))
    if verify:
        n_crit = sum(len(t.criteria) for t in tasks)
        if len(tasks) != EXPECTED['tasks'] or n_crit != EXPECTED['criteria']:
            raise RuntimeError(f'unexpected task set: {len(tasks)} tasks, {n_crit} criteria; expected {EXPECTED}')
        missing = [t.pdf_path for t in tasks if not t.pdf_path.exists()]
        if missing:
            raise RuntimeError(f'{len(missing)} PDFs missing, e.g. {missing[0]}')
    return tasks


def select(tasks, limit=None, task_ids=None):
    """A deterministic subset: explicit ids, else the first `limit` tasks in a fixed domain-stratified order
    (round-robin over domains, parquet order within each), so small smokes cover several domains."""
    if task_ids:
        want = set(task_ids)
        out = [t for t in tasks if t.task_id in want]
        if len(out) != len(want):
            raise SystemExit(f'unknown task ids: {sorted(want - {t.task_id for t in out})}')
        return out
    if not limit:
        return list(tasks)
    by_domain = {}
    for t in tasks:
        by_domain.setdefault(t.domain, []).append(t)
    order, i = [], 0
    while len(order) < len(tasks):
        for d in sorted(by_domain):
            if i < len(by_domain[d]):
                order.append(by_domain[d][i])
        i += 1
    return order[:limit]


def dataset_info():
    return {'hf_repo': HF_REPO, 'hf_revision': HF_REVISION, 'parquet_sha256': PARQUET_SHA256,
            'official_repo': 'surgeai/GDP.pdf (unavailable since ~2026-10)', 'expected': EXPECTED, 'canary': CANARY}


if __name__ == '__main__':
    ts = load_tasks()
    print(json.dumps({'tasks': len(ts), 'criteria': sum(len(t.criteria) for t in ts),
                      'domains': sorted({t.domain for t in ts})}, indent=1))

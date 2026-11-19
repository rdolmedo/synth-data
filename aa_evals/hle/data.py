"""HLE data: fetch a pinned revision from Hugging Face to /data, load it as plain dicts.

The questions may not be redistributed. They live only under DATA_ROOT (mode 700), never in git,
issue comments, PRs or logs we share; run artifacts reference questions by `id`.

    python -m aa_evals.hle.data fetch [--dataset cais/hle-diamond]
    python -m aa_evals.hle.data stats [--dataset cais/hle-diamond]    # aggregate counts only
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

DATA_ROOT = Path('/data/aai/aa_evals/data/hle')
PARQUET = 'data/test-00000-of-00001.parquet'
# Pinned revisions (HF dataset commit SHAs, checked 2026-11-02). Diamond is what AA's index uses;
# cais/hle is AA's older HLE eval (removed from the index in v5.0) and the schema we develop against.
REVISIONS = {
    'cais/hle-diamond': '04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23',
    'cais/hle': '5a81a4c7271a2a2a312b9a690f0c2fde837e4c29',
}
COLUMNS = ['id', 'question', 'image', 'answer', 'answer_type', 'category', 'raw_subject', 'partition']
# Question sets (#348, user decision 2026-11-11). Every model is served without vision, and without its image an
# image question is near the floor (text-only Lightning, M3.6: 6 of 272), so new runs score only the questions
# without an image. 'all' keeps every question (AA's set; the M3.0-M3.6 runs).
TASK_SETS = ('text-only-v1', 'all')
DEFAULT_TASK_SET = 'text-only-v1'
# The size of each set at the pinned revision, checked when a run selects it.
TASK_SET_SIZES = {('cais/hle-diamond', 'text-only-v1'): 728, ('cais/hle-diamond', 'all'): 1000}


def local_dir(dataset, revision=None):
    return DATA_ROOT / dataset.replace('/', '__') / (revision or REVISIONS[dataset])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def fetch(dataset, revision=None):
    """Download the test parquet at the pinned revision and write manifest.json next to it."""
    from huggingface_hub import hf_hub_download
    revision = revision or REVISIONS[dataset]
    dest = local_dir(dataset, revision)
    if (dest / 'manifest.json').exists():
        return json.loads((dest / 'manifest.json').read_text())
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_ROOT, 0o700)
    with tempfile.TemporaryDirectory(prefix='hle-fetch-') as tmp:
        # A 403 here means the HF account behind HF_TOKEN has not accepted the dataset's terms.
        path = hf_hub_download(dataset, PARQUET, repo_type='dataset', revision=revision, local_dir=tmp,
                               token=os.environ.get('HF_TOKEN'))
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest / 'test.parquet')
    rows = load(dataset, revision, check=False)
    manifest = {'dataset': dataset, 'revision': revision, 'file': PARQUET, 'sha256': sha256(dest / 'test.parquet'),
                'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **stats(rows)}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load(dataset, revision=None, check=True):
    """The dataset's rows as dicts with COLUMNS (`partition` is None where the dataset has none)."""
    import pyarrow.parquet as pq
    dest = local_dir(dataset, revision)
    path = dest / 'test.parquet'
    if not path.exists():
        raise SystemExit(f'{path} missing: run `python -m aa_evals.hle.data fetch --dataset {dataset}` first')
    if check:
        manifest = json.loads((dest / 'manifest.json').read_text())
        if os.path.getsize(path) and manifest.get('sha256') and manifest['sha256'] != sha256(path):
            raise SystemExit(f'{path} does not match its manifest sha256')
    table = pq.read_table(path)
    names = set(table.column_names)
    table = table.select([c for c in COLUMNS if c in names])
    rows = table.to_pylist()
    for row in rows:
        for c in COLUMNS:
            row.setdefault(c, None)
        row['image'] = row['image'] or ''
    return rows


def task_set(rows, name):
    """The rows of a named question set, in the given order: 'all', or 'text-only-v1' (the rows without an image)."""
    if name == 'all':
        return list(rows)
    if name == 'text-only-v1':
        return [r for r in rows if not r['image']]
    raise ValueError(f'unknown task set {name!r}; one of {TASK_SETS}')


def stats(rows):
    """Aggregate counts only (safe to share)."""
    def count(key):
        out = {}
        for r in rows:
            out[str(r[key])] = out.get(str(r[key]), 0) + 1
        return dict(sorted(out.items()))
    return {'n_items': len(rows), 'n_image': sum(bool(r['image']) for r in rows), 'partition': count('partition'),
            'answer_type': count('answer_type'), 'category': count('category')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('op', choices=['fetch', 'stats'])
    ap.add_argument('--dataset', default='cais/hle-diamond', choices=sorted(REVISIONS))
    ap.add_argument('--revision')
    a = ap.parse_args()
    out = fetch(a.dataset, a.revision) if a.op == 'fetch' else stats(load(a.dataset, a.revision))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

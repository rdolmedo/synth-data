"""AA-Omniscience public question sets, pinned by HF revision and file SHA-256.

The data is never committed. It lives in /data/aai/aa_evals/data/omniscience/<dir>/ and is fetched on
first use from the pinned HF revision. Both sets are Apache-2.0 and ungated, but they are eval-only: keep them
out of any training data.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(os.environ.get('AA_EVALS_DATA', '/data/aai/aa_evals/data')) / 'omniscience'
CSV_NAME = 'AA-Omniscience_dataset_public.csv'


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    revision: str
    sha256: str
    n: int
    aa_version: str  # the AA-Omniscience version this public set belongs to

    @property
    def dir(self) -> Path:
        tag = 'public-v1.1' if self.name == 'v1.1' else 'public-v1'
        return DATA_ROOT / f'{tag}@{self.revision[:7]}'

    @property
    def csv_path(self) -> Path:
        return self.dir / CSV_NAME

    def url(self, filename: str) -> str:
        return f'https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{filename}'


DATASETS = {
    # Default. The 10% public slice of AA-Omniscience v1.1 (3,000 questions, the version in AA index v5.0).
    'v1.1': DatasetSpec('v1.1', 'ArtificialAnalysis/AA-Omniscience-Public-v1.1',
                        '1f9cbcec472000b28f9782205f1ef9498b45bd21',
                        'f4fbdd1ddf906d16b39899f7aaf1c96375135ac8f12ab83dd1dd41997b5cc0d7', 300, '1.1'),
    # The v1.0 public set. v1.1 removed these questions, so scores on it are NOT comparable to AA's v1.1 numbers.
    'v1': DatasetSpec('v1', 'ArtificialAnalysis/AA-Omniscience-Public',
                      'e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849',
                      '1e04603dafa3bd0d16d8151f07a5eb74c43d90c3a85d2fca120da2359174d02f', 600, '1.0'),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(spec: DatasetSpec) -> Path:
    """Download the pinned CSV and card if missing (mkdir mutex: Lustre has no file locks)."""
    if spec.csv_path.exists():
        return spec.csv_path
    spec.dir.mkdir(parents=True, exist_ok=True)
    lock = spec.dir / '.fetch.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'{lock} exists: another process is fetching (remove it if stale)')
    try:
        headers = {}
        if os.environ.get('HF_TOKEN'):
            headers['Authorization'] = f'Bearer {os.environ["HF_TOKEN"]}'
        for name in ('README.md', CSV_NAME):
            req = urllib.request.Request(spec.url(name), headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            tmp = spec.dir / f'.{name}.tmp'
            tmp.write_bytes(body)
            tmp.rename(spec.dir / name)
        (spec.dir / 'SOURCE').write_text(f'{spec.repo} {spec.revision}\n')
    finally:
        lock.rmdir()
    return spec.csv_path


def load(name: str = 'v1.1') -> tuple[DatasetSpec, list[dict]]:
    spec = DATASETS[name]
    path = fetch(spec)
    got = _sha256(path)
    if got != spec.sha256:
        raise RuntimeError(f'{path}: sha256 {got} != pinned {spec.sha256}')
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != spec.n:
        raise RuntimeError(f'{path}: {len(rows)} rows, expected {spec.n}')
    items = [{'item_id': str(r['question_id']), 'domain': r['domain'], 'subtopic': r['subtopic'],
              'question': r['question'], 'answer': r['answer']} for r in rows]
    if len({it['item_id'] for it in items}) != len(items):
        raise RuntimeError(f'{path}: duplicate question_id')
    return spec, items


def subset(items: list[dict], limit: int | None, seed: int = 0) -> list[dict]:
    """The first `limit` items of a fixed, domain-balanced order: each domain is shuffled with `seed`, then the
    domains are interleaved round-robin. --limit 60 on v1.1 gives 10 questions per domain."""
    if not limit or limit >= len(items):
        return list(items)
    by_domain: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda it: int(it['item_id'])):
        by_domain.setdefault(it['domain'], []).append(it)
    rng = random.Random(seed)
    queues = []
    for domain in sorted(by_domain):
        q = by_domain[domain][:]
        rng.shuffle(q)
        queues.append(q)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order[:limit]

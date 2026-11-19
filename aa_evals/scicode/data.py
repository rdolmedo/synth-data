"""SciCode datasets, pinned by md5. The data lives in /data/aai/aa_evals/data/scicode/ and is not committed.

  original      upstream SciCode test split (HF SciCode1/SciCode, 65 problems, 288 scored steps) + test_data.h5.
                This is what AA scored (sub-step accuracy over 288 steps); our harness check against AA.
  verified-v2   SciCode-Verified v2 (flyingwagner/scicode-verified 0b40bd4: 64 problems, 287 scored steps; problem 2
                dropped) + test_data_cleaned.h5.
  verified-aa-vN  verified-v2 with our patch list vN applied (aa_evals/scicode/patches/). Built by
                `python -m aa_evals.scicode.patches.build --version vN`; the build writes a problems file plus an
                overlay h5 holding only the patched steps' targets (the other steps read the v2 h5).

Steps 13.6, 62.1 and 76.3 are never generated or scored (the official harness skips them); their gold code from
upstream eval/data/ is injected into the cumulative code of the later steps.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from aa_evals.common import jsonl

ROOT = Path('/data/aai/aa_evals/data/scicode')
SKIP_STEPS = {'13.6', '62.1', '76.3'}
SKIP_GOLD_MD5 = {'13.6': 'a09eb1732efe7588722adb77b1f500fb', '62.1': '78804c388c69487c02e336375cba1089',
                 '76.3': 'f2ae41955825aaa3ced4eeb37186faca'}

BASE = {
    'original': {
        'problems': ROOT / 'original/problems_test.jsonl', 'problems_md5': 'c176df46e558ffa6fb8f18116b1a7361',
        'h5': ROOT / 'original/test_data.h5', 'h5_md5': '96d5d815aee54434deba01eb27646f22',
        'source': 'SciCode1/SciCode (HF) test split; test_data.h5 = upstream harness pin',
        'n_problems': 65, 'n_steps': 288,
    },
    'verified-v2': {
        'problems': ROOT / 'verified_v2/problems_test.jsonl', 'problems_md5': '5c604d8dbf52642bd94e13b92c8f52eb',
        'h5': ROOT / 'verified_v2/test_data_cleaned.h5', 'h5_md5': '2b41a7df40ddc23ce651ec05b8ecb6f8',
        'source': 'flyingwagner/scicode-verified 0b40bd4, release "data" v2 (manifest.json)',
        'n_problems': 64, 'n_steps': 287,
    },
}


@dataclass
class Dataset:
    name: str
    problems: list[dict]
    h5: Path                      # base targets
    overlay_h5: Path | None       # patched steps' targets (verified-aa-vN)
    overlay_steps: set[str]
    pins: dict
    excluded_steps: set[str] = field(default_factory=set)   # generated (later steps build on them), never scored
    overlay_step_sha: dict = field(default_factory=dict)       # per overlay step: sha256 of its targets (builder)

    def target_key(self, step: str) -> tuple:
        """Identity of a step's targets, to tell whether two dataset versions grade a step against the same values.
        Overlay steps without a recorded digest fall back to the whole overlay's md5 (conservative: any change of
        the overlay counts as a change)."""
        if self.overlay_h5 and step in self.overlay_steps:
            return ('overlay', self.overlay_step_sha.get(step) or self.pins.get('overlay_h5_md5'))
        return ('base', (self.pins.get('base_pins') or self.pins).get('h5_md5'))

    def h5_for(self, step: str) -> Path:
        return self.overlay_h5 if (self.overlay_h5 and step in self.overlay_steps) else self.h5

    def scored_steps(self) -> list[str]:
        return [s['step_number'] for p in self.problems for s in p['sub_steps']
                if s['step_number'] not in SKIP_STEPS and s['step_number'] not in self.excluded_steps]


def md5(path: Path, _cache: dict = {}) -> str:
    key = (str(path), path.stat().st_mtime, path.stat().st_size)
    if key not in _cache:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
        _cache[key] = h.hexdigest()
    return _cache[key]


def read_jsonl(path: Path) -> list[dict]:
    return jsonl.read(path, required=True)


def skip_gold(step: str) -> str:
    p = ROOT / 'skip_gold' / f'{step}.txt'
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != SKIP_GOLD_MD5[step]:
        raise SystemExit(f'{p}: md5 {got} != pinned {SKIP_GOLD_MD5[step]}')
    return p.read_text()


def load(name: str, *, check_h5: bool = True) -> Dataset:
    """Load and verify a dataset. check_h5=False skips hashing the ~1 GB target files (generation-only callers)."""
    if name in BASE:
        b = BASE[name]
        pin = b['problems_md5']
        got = md5(b['problems'])
        if got != pin:
            raise SystemExit(f'{b["problems"]}: md5 {got} != pinned {pin}')
        if check_h5 and md5(b['h5']) != b['h5_md5']:
            raise SystemExit(f'{b["h5"]}: md5 != pinned {b["h5_md5"]}')
        probs = read_jsonl(b['problems'])
        ds = Dataset(name, probs, b['h5'], None, set(),
                     {'dataset': name, 'source': b['source'], 'problems_md5': got, 'h5_md5': b['h5_md5']})
    elif name.startswith('verified-aa-'):
        version = name.removeprefix('verified-aa-')
        d = ROOT / f'verified_aa_{version}'
        man = json.loads((d / 'manifest.json').read_text())
        base = load(man['base'], check_h5=check_h5)
        if md5(d / 'problems_test.jsonl') != man['problems_md5']:
            raise SystemExit(f'{d}/problems_test.jsonl: md5 != its manifest (rebuild with patches.build)')
        overlay = d / 'targets_overlay.h5' if man.get('overlay_steps') else None
        if overlay and check_h5 and md5(overlay) != man['overlay_h5_md5']:
            raise SystemExit(f'{overlay}: md5 != its manifest')
        ds = Dataset(name, read_jsonl(d / 'problems_test.jsonl'), base.h5, overlay, set(man.get('overlay_steps', [])),
                     {'dataset': name, 'base': man['base'], 'base_pins': base.pins, 'patch_version': version,
                      'patch_list_sha256': man['patch_list_sha256'], 'problems_md5': man['problems_md5'],
                      'overlay_h5_md5': man.get('overlay_h5_md5'), 'excluded_steps': man.get('excluded_steps', []),
                      'n_patches': man['n_patches']}, set(man.get('excluded_steps', [])),
                     dict(man.get('overlay_step_sha256') or {}))
    else:
        raise SystemExit(f'unknown dataset {name!r} (original, verified-v2, verified-aa-<version>)')
    ds.pins['n_problems'] = len(ds.problems)
    ds.pins['n_scored_steps'] = len(ds.scored_steps())
    for s in SKIP_STEPS:
        skip_gold(s)
    return ds

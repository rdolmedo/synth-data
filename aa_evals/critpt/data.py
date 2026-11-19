"""CritPt v2 problems, pinned.

The 70 test challenges come from the HF dataset `CritPt-Benchmark/CritPt` at a fixed revision (answers are hidden;
only the authors' server grades). The example challenge (`quantum_error_correction_main`, not one of the 70) comes
from the authors' GitHub repo at the "CritPt v2" commit and ships its golden SymPy answer, which we use to check our
answer extraction offline. Nothing here is committed to git: the data lives under DATA_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path('/data/aai/aa_evals/data/critpt')
HF_REPO = 'CritPt-Benchmark/CritPt'
HF_REVISION = '5b60585631e5c15ee34c32e21a430a4e58b1cfb3'
HF_PARQUET = DATA_ROOT / 'hf_5b60585' / 'data' / 'train-00000-of-00001.parquet'
HF_PARQUET_SHA256 = 'a3b8a3d9b6b1d9c6aae2c4f098c91fd382dd71a9a23e6494f74019280cb115ed'
GH_REPO = 'https://github.com/CritPt-Benchmark/CritPt'
GH_COMMIT = 'bc37a22bedb8744dff8996ccce7f9eedec548915'   # "CritPt v2"
GH_DIR = DATA_ROOT / 'src' / 'critpt_repo'
EXAMPLE_ID = 'quantum_error_correction_main'
N_PROBLEMS = 70


@dataclass(frozen=True)
class Problem:
    problem_id: str            # e.g. Challenge_12_main (the grading server's id)
    description: str           # markdown problem statement (setup + main problem)
    code_template: str         # the Python template the answer must fill
    golden_code: str | None = None   # only for the example challenge


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_pins() -> dict:
    """Verify the pinned files; returns the provenance block for config.json."""
    got = _sha256(HF_PARQUET)
    if got != HF_PARQUET_SHA256:
        raise SystemExit(f'{HF_PARQUET}: sha256 {got} != pinned {HF_PARQUET_SHA256}')
    head = subprocess.run(['git', '-C', str(GH_DIR), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != GH_COMMIT:
        raise SystemExit(f'{GH_DIR}: HEAD {head} != pinned {GH_COMMIT}')
    return {'hf_repo': HF_REPO, 'hf_revision': HF_REVISION, 'hf_parquet_sha256': HF_PARQUET_SHA256,
            'gh_repo': GH_REPO, 'gh_commit': GH_COMMIT}


def _challenge_num(pid: str) -> int:
    return int(re.match(r'Challenge_(\d+)_main$', pid).group(1))


def load_problems() -> list[Problem]:
    """The 70 test challenges, ordered Challenge_1 .. Challenge_70."""
    import pyarrow.parquet as pq
    rows = pq.read_table(HF_PARQUET).to_pylist()
    probs = [Problem(r['problem_id'], r['problem_description'], r['code_template']) for r in rows]
    probs.sort(key=lambda p: _challenge_num(p.problem_id))
    ids = [p.problem_id for p in probs]
    if len(probs) != N_PROBLEMS or len(set(ids)) != N_PROBLEMS:
        raise SystemExit(f'expected {N_PROBLEMS} unique problems, got {len(probs)}')
    return probs


def load_examples() -> list[Problem]:
    """The example challenge and its distinct checkpoints (subproblems), each with its golden answer.

    These are the only CritPt problems with public answers. Subproblem 3 asks the main question again (same answer),
    so it is skipped.
    """
    main = load_example()
    nb = json.loads((GH_DIR / 'data' / 'example_challenges' / 'quantum_error_correction.ipynb').read_text())
    cells = [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    setup = next(s for t, s in cells if t == 'markdown' and 'Problem setup' in s)
    out = [main]
    for i, (t, s) in enumerate(cells):
        m = re.match(r'##\s*Subproblem\s*(\d+):\s*', s)
        if t != 'markdown' or not m:
            continue
        j_tpl = next(j for j in range(i + 1, len(cells)) if 'Parsing template' in cells[j][1])
        j_ans = next(j for j in range(i + 1, len(cells)) if 'Answer code' in cells[j][1])
        golden = cells[j_ans + 1][1]
        if golden.split('=', 1)[1].strip() == main.golden_code.split('=', 1)[1].strip():
            continue
        desc = setup.rstrip() + '\n\n# Main problem:\n' + s[m.end():].strip() + '\n'
        out.append(Problem(f'quantum_error_correction_sub_{m.group(1)}', desc, cells[j_tpl + 1][1], golden))
    return out


def load_example() -> Problem:
    """The example challenge with its golden answer code (from the authors' notebook)."""
    nb = json.loads((GH_DIR / 'data' / 'example_challenges' / f'{EXAMPLE_ID}.ipynb').read_text())
    cells = [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    setup = next(s for t, s in cells if t == 'markdown' and 'Problem setup' in s)
    main = next(s for t, s in cells if t == 'markdown' and 'Main problem' in s)
    i_tpl = next(i for i, (t, s) in enumerate(cells) if 'Parsing template' in s)
    i_ans = next(i for i, (t, s) in enumerate(cells) if 'Answer code' in s)
    template = cells[i_tpl + 1][1]
    golden = cells[i_ans + 1][1]
    # Same layout as the HF rows: setup, then the main problem.
    description = setup.rstrip() + '\n\n' + main.strip() + '\n'
    return Problem(EXAMPLE_ID, description, template, golden)

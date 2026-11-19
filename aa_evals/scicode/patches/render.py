"""Render the public patch list (the aa_evals/scicode/PATCHES.md section for one version) from a built version.

    python -m aa_evals.scicode.patches.render --version v1 > /tmp/v1.md   (then paste it under the PATCHES.md intro)

Eval data stays out of git. PATCHES.md holds only the redacted public text of each patch: the defect class and the
kind of fix, from patches/public/<version>.json, which is committed. It also holds validation counts and the sha256
of every payload file. It never prints target values, test parameters, formulas, function names, conventions,
mutant or alternative names, or the payload's rationale and evidence; those stay with the eval data on /data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .. import data
from .build import PAYLOAD_ROOT

PUBLIC = Path(__file__).parent / 'public'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    a = ap.parse_args(argv)
    root = PAYLOAD_ROOT / a.version
    out = data.ROOT / f'verified_aa_{a.version}'
    man = json.loads((out / 'manifest.json').read_text())
    val = json.loads((out / 'validation.json').read_text())['results']
    pub = json.loads((PUBLIC / f'{a.version}.json').read_text())
    by_patch = defaultdict(list)
    for v in val:
        by_patch[v['patch']].append(v)
    patches = [(f, pt) for f in man['patch_files'] for pt in json.loads((root / f).read_text())['patches']]
    missing = [pt['id'] for _, pt in patches if pt['id'] not in pub]
    if missing:
        raise SystemExit(f'no public text for {missing} in {PUBLIC}/{a.version}.json')
    print(f'## {a.version}\n')
    print(f'Base `{man["base"]}`; {man["n_patches"]} patches; problems md5 `{man["problems_md5"]}`; overlay h5 md5 '
          f'`{man["overlay_h5_md5"]}`; overlay steps {", ".join(man["overlay_steps"]) or "none"}; excluded steps '
          f'{", ".join(man["excluded_steps"]) or "none"}.\n')
    print(f'Payload (eval data, not in git): `{root}/`, tree sha256 `{man["patch_list_sha256"]}`.\n')
    print('| file | sha256 |')
    print('|---|---|')
    for f in man['patch_files']:
        print(f'| `{f}` | `{hashlib.sha256((root / f).read_bytes()).hexdigest()}` |')
    print()
    print('| id | step | kind | change | validation | full rationale and evidence |')
    print('|---|---|---|---|---|---|')
    for f, pt in patches:
        rows = by_patch.get(pt['id'], [])
        if pt['kind'] == 'tests':
            mut = [r for r in rows if r['want'] == 'fail']
            alt = [r for r in rows if r['want'] == 'pass' and r['case'] != 'ref']
            ref = [r for r in rows if r['case'] == 'ref']
            vtxt = (f"ref {'passes' if ref and ref[0]['ok'] else 'FAILS'}; {sum(r['ok'] for r in mut)}/{len(mut)} "
                    f"mutants rejected; {sum(r['ok'] for r in alt)}/{len(alt)} alternatives accepted")
        elif pt['kind'] == 'text':
            vtxt = 'text check (`textcheck/`)'
        else:
            vtxt = '-'
        where = pt.get('step') or f"problem {pt.get('problem')}"
        print(f"| `{pt['id']}` | {where} | {pt['kind']} | {pub[pt['id']]['summary']} | {vtxt} | `{f}` |")
    print('\n### Why\n')
    for _, pt in patches:
        print(f"- **`{pt['id']}`.** {pub[pt['id']]['why']}")
    print()


if __name__ == '__main__':
    main()

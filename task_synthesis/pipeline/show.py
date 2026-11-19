"""Print one task of a round for reading: spec, instruction, files, setup, built tree, judge.

  python -m task_synthesis.pipeline.show --round r1 s1-00007 [--files] [--sample 20 --seed 0 --passed]"""
import argparse
import json
import random
import tempfile
from pathlib import Path

from .round import ROUNDS


def show(rd, sid, files=False):
    from .roundpack import RoundFiles
    rf = RoundFiles(rd)
    if not (rd / 'work' / sid / 'record.json').exists() and rf.is_packed():   # packed (roundpack): a temporary copy
        with tempfile.TemporaryDirectory(prefix='tasksynth-show-') as tmp:
            for area in ('work', 'tasks'):
                try:
                    rf.copy_area(sid, area, Path(tmp) / area / sid)
                except FileNotFoundError:
                    pass
            return show(Path(tmp), sid, files)
    wd = rd / 'work' / sid
    rec = json.loads((wd / 'record.json').read_text())
    td = rd / 'tasks' / sid
    print('=' * 100)
    print(f'{sid}  status={rec["status"]}  repairs={rec.get("repairs")}  spec={json.dumps(rec["spec"])[:400]}')
    if rec.get('meta'):
        print('meta:', json.dumps(rec['meta'])[:800])
    a = rec['spec'].get('asset')
    if a:
        print(f'asset: {a["asset_id"]} {a.get("source")} "{a.get("title")}" {a.get("license")} {a.get("url")}\n'
              f'       use_paths={a.get("use_paths")} tar={a.get("tar_bytes")} B\n       idea: {a.get("idea", "")[:300]}')
    if (td / 'instruction.md').exists():
        print('\n--- instruction\n' + (td / 'instruction.md').read_text())
    env = td / 'environment' / 'files'
    if env.exists():
        listing = sorted(str(p.relative_to(env)) for p in env.rglob('*') if p.is_file())
        print('--- author files:', listing)
        if files:
            for p in sorted(env.rglob('*')):
                if p.is_file():
                    print(f'\n### /{p.relative_to(env)}\n{p.read_text(errors="replace")[:3000]}')
    if (td / 'environment' / 'tasksynth' / 'setup.sh').exists():
        print('--- setup\n' + (td / 'environment' / 'tasksynth' / 'setup.sh').read_text()[:2000])
    checks = sorted(wd.glob('attempt-*.check.json'))
    if checks:
        c = json.loads(checks[-1].read_text())
        print('--- check errors:', c.get('errors'), 'warnings:', c.get('warnings'))
        print('--- built tree\n' + '\n'.join(c.get('snapshot', {}).get('tree', [])[:60]))
    for a in rec.get('attempts', []):
        if a.get('errors'):
            print(f'--- attempt {a["k"]} errors:', [e[:300] for e in a['errors']])
    if rec.get('judge'):
        j = rec['judge']
        print('--- judge:', 'PASS' if j.get('pass') else 'FAIL', j.get('scores'), 'est', j.get('estimated_minutes'),
              'flags', {k: v for k, v in (j.get('flags') or {}).items() if v})
        for i in j.get('issues', []):
            print('   -', i)
    if (td / 'solution' / 'sketch.md').exists():
        print('--- sketch\n' + (td / 'solution' / 'sketch.md').read_text()[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('ids', nargs='*')
    ap.add_argument('--files', action='store_true')
    ap.add_argument('--sample', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--passed', action='store_true')
    ap.add_argument('--status')
    a = ap.parse_args()
    rd = ROUNDS / a.round
    ids = a.ids
    if a.sample:
        from .roundpack import RoundFiles
        recs = RoundFiles(rd).records()
        if a.passed:
            recs = [r for r in recs if r.get('judge', {}).get('pass')]
        if a.status:
            recs = [r for r in recs if r['status'] == a.status]
        ids = [r['spec_id'] for r in random.Random(a.seed).sample(recs, min(a.sample, len(recs)))]
    for sid in ids:
        show(rd, sid, a.files)


if __name__ == '__main__':
    main()

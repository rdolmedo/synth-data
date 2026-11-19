"""Issue #125: summary of a decontam.pool list (flag rates by real-asset vs generated task, by test task and bench,
and the spec details behind each flag, as in #19's audit).

  python -m decontam.pool_report --work W [--list W/lists/<n>/decontamination-variants.json] [--out report.json]

Reads the pool's stream chunks (W/pool.json units) for each task's tasksynth.json and whether it carries a real asset
(environment/tasksynth/asset.tar.zst). Prints a Markdown summary; --out writes the numbers as JSON.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import tarfile

ASSET = 'environment/tasksynth/asset.tar.zst'
STUDENT1 = (36, 7296)   # #19


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def details(spec):
    """Spec details a task drew: asset class:format / service / language / git state ... (one string each)."""
    out = []
    for a in spec.get('assets') or []:
        cls = a.get('class')
        for key, val in a.items():
            if key in ('class', 'role'):
                continue
            for v in (val if isinstance(val, list) else [val]):
                out.append(f'{cls}:{v}')
    for key in ('agent_action', 'anchor_class'):
        if spec.get(key):
            out.append(f'{key}={spec[key]}')
    return out


def read_pool(work):
    st = json.loads((Path(work) / 'pool.json').read_text())
    tasks = {}
    for u in st['units']:
        with tarfile.open(u['tar']) as t:
            for m in t.getmembers():
                parts = m.name.lstrip('./').split('/')
                if len(parts) < 3 or parts[0] != 'tasks':
                    continue
                rec = tasks.setdefault(parts[1], {'unit': u['uid'], 'real_asset': False, 'spec': {}})
                rest = '/'.join(parts[2:])
                if rest == ASSET:
                    rec['real_asset'] = True
                elif rest == 'tasksynth.json':
                    rec['spec'] = json.loads(t.extractfile(m).read()).get('spec', {})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--list')
    ap.add_argument('--out')
    a = ap.parse_args()
    work = Path(a.work)
    lst = Path(a.list) if a.list else Path(json.loads((work / 'lists' / 'LATEST.json').read_text())['list'])
    L = json.loads(lst.read_text())
    tasks = read_pool(work)
    ex = L['excluded']
    unknown = sorted(set(ex) - set(tasks))
    groups = {'all': list(tasks), 'real-asset': [t for t, r in tasks.items() if r['real_asset']],
              'generated': [t for t, r in tasks.items() if not r['real_asset']]}
    rows = {}
    for g, names in groups.items():
        k = sum(t in ex for t in names)
        rows[g] = {'tasks': len(names), 'flagged': k, 'rate': k / max(1, len(names)), 'ci95': wilson(k, len(names))}
    k1, n1 = STUDENT1
    rows['student 1 (#19)'] = {'tasks': n1, 'flagged': k1, 'rate': k1 / n1, 'ci95': wilson(k1, n1)}
    by_tb = collections.defaultdict(lambda: {'flagged': 0, 'real_asset': 0, 'bench': None})
    for t, e in ex.items():
        for v in e['verdicts']:
            b = by_tb[v['tb']]
            b['bench'] = v.get('bench')
        for tb in set(e['tb_tasks']):
            by_tb[tb]['flagged'] += 1
            by_tb[tb]['real_asset'] += bool(tasks.get(t, {}).get('real_asset'))
    # spec details: draws in the pool vs flags (single-source details leaked in #19)
    draws, flags = collections.Counter(), collections.Counter()
    for t, r in tasks.items():
        ds = set(details(r['spec']))
        draws.update(ds)
        if t in ex:
            flags.update(ds)
    flagged = []
    for t, e in sorted(ex.items()):
        r = tasks.get(t, {})
        flagged.append({'task': t, 'tb_tasks': e['tb_tasks'], 'sources': e['sources'], 'real_asset': r.get('real_asset'),
                        'domain': r.get('spec', {}).get('domain'), 'details': details(r.get('spec', {})),
                        'reason': (e['verdicts'][0].get('reason') or '')[:400]})
    rep = {'list': str(lst), 'counts': L['counts'], 'groups': rows, 'by_test_task': dict(by_tb),
           'details_in_flags': {d: {'flags': flags[d], 'draws': draws[d]} for d in flags},
           'flagged': flagged, 'unknown_flagged': unknown}
    print(f'list {lst}\n')
    print('| Group | Tasks | Flagged | Rate | 95% CI |\n|---|---|---|---|---|')
    for g, r in rows.items():
        print(f"| {g} | {r['tasks']:,} | {r['flagged']} | {100 * r['rate']:.2f}% | "
              f"{100 * r['ci95'][0]:.2f}-{100 * r['ci95'][1]:.2f}% |")
    print('\n| Test task | Bench | Flagged | of which real-asset |\n|---|---|---|---|')
    for tb, b in sorted(by_tb.items(), key=lambda x: -x[1]['flagged']):
        print(f"| {tb} | {b['bench']} | {b['flagged']} | {b['real_asset']} |")
    print('\nSpec details on flagged tasks (flags / draws in the pool):')
    for d, n in flags.most_common(25):
        print(f'  {d}: {n} / {draws[d]}')
    if unknown:
        print(f'\nWARNING: {len(unknown)} flagged tasks not in the pool: {unknown[:5]}')
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

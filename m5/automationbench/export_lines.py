"""Export shards whose line records agree with the manifest (sft_qwen35/prepare.py trusts a shard line's own
record.usable over manifest.json), optionally cut to a token budget.

  python -m m5.automationbench.export_lines SRC_ROOT DST_ROOT [--budget 330e6 --seed 359]
SRC_ROOT: collect.py output + export.py manifest. --budget keeps whole usable traces in a seeded random order of
tasks (not collection order, so the cut favours neither early nor short traces) while the total stays within the
budget, stopping at the first trace that would exceed it; the rest are marked usable=false, excluded="budget".
DST_ROOT gets attempts/shard_*.jsonl.gz (same lines, record.usable and
record.excluded from the manifest), attempts/index.json, manifest.json, SHA256SUMS. Each file is written to a temp name,
read back (gzip, line count, ids, flags) and renamed; EIO/EDQUOT are retried with backoff.
"""
import argparse, gzip, hashlib, json, os, random, time
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--budget', type=float, default=0)
ap.add_argument('--seed', type=int, default=359)
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
(dst / 'attempts').mkdir(parents=True, exist_ok=True)
man = json.loads((src / 'manifest.json').read_text())
if a.budget:
    tasks = sorted({r['task'] for r in man['records'] if r['usable']})
    random.Random(a.seed).shuffle(tasks)
    rank = {t: i for i, t in enumerate(tasks)}
    total, full = 0, False
    for r in sorted((r for r in man['records'] if r['usable']), key=lambda r: (rank[r['task']], r['id'])):
        if full or total + r['tokens'] > a.budget:
            full = True
            r['usable'], r['excluded'] = False, 'budget'
        else:
            total += r['tokens']
    man['counts'] = dict(Counter(r['excluded'] or 'kept' for r in man['records']))
    man['kept_tokens'] = total
    man['budget'] = {'tokens': a.budget, 'seed': a.seed, 'order': 'seeded random order of tasks, whole traces'}
by_id = {r['id']: r for r in man['records']}
idx = json.loads((src / 'attempts' / 'index.json').read_text())


def retry(fn, what):
    for k in range(8):
        try:
            return fn()
        except OSError as e:
            print(f'{what}: {e!r}, retry {k + 1}', flush=True)
            time.sleep(min(300, 10 * 2 ** k))
    raise SystemExit(f'{what}: gave up')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


sums = {}
for n in sorted(idx['shards'], key=int):
    name = idx['shards'][n]['lines']
    out = dst / 'attempts' / name
    if out.exists() and (dst / 'attempts' / (name + '.ok')).exists():
        sums[f'attempts/{name}'] = (dst / 'attempts' / (name + '.ok')).read_text().strip()
        continue
    with gzip.open(src / 'attempts' / name, 'rt') as f:
        lines = [json.loads(x) for x in f]
    for x in lines:
        r = by_id[x['id']]
        x['record'] = {**x['record'], 'usable': bool(r['usable']), 'excluded': r['excluded']}

    def write():
        tmp = out.with_name(name + '.partial')
        with gzip.open(tmp, 'wt') as g:
            for x in lines:
                g.write(json.dumps(x) + '\n')
        with gzip.open(tmp, 'rt') as g:
            back = [json.loads(y) for y in g]
        assert [y['id'] for y in back] == [x['id'] for x in lines]
        assert all(y['record']['usable'] == bool(by_id[y['id']]['usable']) for y in back)
        os.replace(tmp, out)
        h = sha(out)
        (dst / 'attempts' / (name + '.ok')).write_text(h + '\n')
        return h
    sums[f'attempts/{name}'] = retry(write, name)
    print(name, len(lines), sum(x['record']['usable'] for x in lines), flush=True)


def small(rel, obj):
    def w():
        p = dst / rel
        tmp = p.with_name(p.name + '.tmp')
        tmp.write_text(json.dumps(obj, indent=1) if rel == 'manifest.json' else json.dumps(obj))
        assert json.loads(tmp.read_text()) == obj
        os.replace(tmp, p)
        return sha(p)
    sums[rel] = retry(w, rel)


small('attempts/index.json', idx)
small('manifest.json', man)
txt = ''.join(f'{h}  {p}\n' for p, h in sorted(sums.items()))
retry(lambda: (dst / 'SHA256SUMS').write_text(txt), 'SHA256SUMS')
for p, h in sums.items():
    assert sha(dst / p) == h, p
for f in (dst / 'attempts').glob('*.ok'):
    f.unlink()
print('done', len(sums), 'files; kept', man['counts'], man['kept_tokens'])

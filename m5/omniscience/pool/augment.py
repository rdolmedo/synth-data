"""Add language-matched software-doc passages to an existing pool (no new downloads).

pool2 capped every parent document at 2 passages and retrieved over all software docs at once, so large manuals
(R manuals, WG14 C drafts, the Rust reference, Go docs) were barely used and Rust/Go leaves got Swift or PHP pages.
Here each SWE subtopic gets its own BM25 index over the windows of the docs whose `subtopic_hint` matches, minus
windows already in the base pool; leaves take up to --add-per-leaf new passages (a document may serve a leaf up to
--per-doc-leaf times and the whole addition up to --per-doc times). Seed decontamination is re-run on the new
indexes (same rule as build.py) and the base pool's blocked docs stay blocked.

The merged pool keeps every base row and pid. New rows get the next pid numbers of their leaf and are written before
the leaf's still-unused base rows, so the sampler serves them first.

Usage: python -m m5.omniscience.pool.augment OUT --base POOL --leaves taxonomy/leaves.json --used RUN/gen.jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import random
from pathlib import Path

import zstandard

from m5.omniscience import fsio
from m5.omniscience.pool.build import (CANARY, SOURCES, Index, log, read_zst, seed_block)

DOMAIN = 'Software Engineering'
JAVA = {'java', 'Java testing libraries', 'Java general purpose libraries'}
HINTS = {'html': {'html/javascript'}, 'javascript': {'javascript', 'html/javascript'}}
HINTS.update({s: JAVA for s in JAVA})


def thash(t):
    return hashlib.sha1(' '.join(t.split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--base', required=True)
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--used', action='append', default=[], help='gen.jsonl of runs drawing from the base pool')
    ap.add_argument('--add-per-leaf', type=int, default=30)
    ap.add_argument('--per-doc', type=int, default=40)
    ap.add_argument('--per-doc-leaf', type=int, default=3)
    ap.add_argument('--win-cap', type=int, default=400)
    ap.add_argument('--k', type=int, default=80)
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=3541)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    leaves = [L for L in json.loads(Path(args.leaves).read_text())['leaves'] if L['domain'] == DOMAIN]
    subs = sorted({L['subtopic'] for L in leaves})

    base = list(read_zst(Path(args.base) / 'passages.jsonl.zst'))
    have = {thash(p['text']) for p in base if p['domain'] == DOMAIN}
    blocked = {json.loads(l)['doc_id'] for l in open(Path(args.base) / 'blocked.jsonl')}
    used = set()
    for u in args.used:
        for l in open(u):
            used.add(json.loads(l)['pid'])
    log('base', len(base), 'swe texts', len(have), 'blocked', len(blocked), 'used', len(used))

    idx = {s: Index(s) for s in subs}
    for path in ('docs.jsonl.zst', 'cstd.jsonl.zst'):
        for d in read_zst(SOURCES / path):
            h = d.get('subtopic_hint', '')
            for s in subs:
                if h in HINTS.get(s, {s}):
                    idx[s].add_doc(d['doc_id'], d.get('title', '')[:200],
                                   f"{d.get('repo', '')} {d.get('path', '')}".strip(), d['text'], cap=args.win_cap,
                                   rng=rng)
    for s in subs:
        keep = [i for i, t in enumerate(idx[s].texts) if thash(t) not in have]
        idx[s].meta = [idx[s].meta[i] for i in keep]
        idx[s].texts = [idx[s].texts[i] for i in keep]
    idx = {s: ix for s, ix in idx.items() if ix.texts}
    log('new windows', {s: len(ix.texts) for s, ix in idx.items()})
    for ix in idx.values():
        ix.build(args.threads)
    nb = seed_block(idx, args.threads)
    blocked_new = {d: w for d, w in nb.items() if d not in blocked}
    blocked |= set(nb)
    log('blocked by new indexes', len(blocked_new))

    n_base = collections.Counter(p['leaf_id'] for p in base)
    added = collections.defaultdict(list)
    per_doc, used_win = collections.Counter(), set()
    for s, ix in idx.items():
        ls = [L for L in leaves if L['subtopic'] == s]
        qs, owners = [], []
        for L in ls:
            for q in L['queries'][:4]:
                qs.append(q); owners.append(L['leaf_id'])
        res, sc = ix.search(qs, args.k, args.threads)
        cands = collections.defaultdict(list)
        for o, r, v in zip(owners, res, sc):
            top = float(v[0]) or 1.0
            cands[o] += [(float(x) / top, int(w)) for w, x in zip(r, v) if w >= 0 and x > 0]
        order = {lid: sorted(set(c), reverse=True) for lid, c in cands.items()}
        pos, leaf_doc = collections.Counter(), collections.Counter()
        active = list(order)
        while active:
            rng.shuffle(active)
            nxt = []
            for lid in active:
                lst = order[lid]
                while pos[lid] < len(lst):
                    score, wid = lst[pos[lid]]
                    pos[lid] += 1
                    p = ix.passage(wid)
                    if thash(p['text']) in used_win or p['doc_id'] in blocked or per_doc[p['doc_id']] >= args.per_doc:
                        continue
                    if leaf_doc[(lid, p['doc_id'])] >= args.per_doc_leaf or CANARY.search(p['text']):
                        continue
                    used_win.add(thash(p['text'])); per_doc[p['doc_id']] += 1; leaf_doc[(lid, p['doc_id'])] += 1
                    L = next(x for x in ls if x['leaf_id'] == lid)
                    k = n_base[lid] + len(added[lid])
                    added[lid].append({'pid': f'{lid}:{k}', 'leaf_id': lid, 'domain': L['domain'],
                                       'subtopic': L['subtopic'], 'leaf': L['leaf'], 'source': 'sw',
                                       'score': round(score, 3), 'augment': 1, **p})
                    break
                if len(added[lid]) < args.add_per_leaf and pos[lid] < len(lst):
                    nxt.append(lid)
            active = nxt

    # merged pool: base rows in order, except each SWE leaf's unused base rows go after its new rows
    held = collections.defaultdict(list)
    rows = []
    for p in base:
        if p['leaf_id'] in added and p['pid'] not in used:
            held[p['leaf_id']].append(p)
        else:
            rows.append(p)
    for lid in sorted(set(added) | set(held)):
        rows += added.get(lid, []) + held.get(lid, [])
    pids = [p['pid'] for p in rows]
    assert len(pids) == len(set(pids)) and len(rows) == len(base) + sum(map(len, added.values()))
    blob = zstandard.ZstdCompressor(level=10).compress(
        ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in rows).encode())
    n_check = len(zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 34).splitlines())
    assert n_check == len(rows), (n_check, len(rows))
    fsio.write_atomic(out / 'passages.jsonl.zst', blob)
    fsio.verify_file(out / 'passages.jsonl.zst')
    fsio.write_atomic(out / 'blocked.jsonl', ''.join(json.dumps({'doc_id': d}) + '\n' for d in sorted(blocked)))
    by_sub = collections.Counter()
    for lid, a in added.items():
        if a:
            by_sub[a[0]['subtopic']] += len(a)
    stats = {'n_passages': len(rows), 'n_base': len(base), 'n_added': sum(by_sub.values()),
             'added_by_subtopic': dict(by_sub.most_common()), 'blocked_new': len(blocked_new),
             'new_windows': {s: len(ix.texts) for s, ix in idx.items()}, 'args': vars(args)}
    fsio.write_atomic(out / 'pool_stats.json', json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == '__main__':
    main()

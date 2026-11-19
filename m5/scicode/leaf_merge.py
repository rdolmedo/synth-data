"""Merge near-identical taxonomy leaves across cells, so that each subject is owned by one leaf (the pilot's repeat
meter traced its near-duplicate problems to leaves on one subject in different cells; #53 lesson 2: fix repeats in the
taxonomy). Candidates: each leaf's bge-small neighbours (cosine > CAND_COS, at most K). A DeepSeek judge sees one leaf
and its candidates and lists those on the same subject. Then a greedy pass in a fixed random order keeps a leaf only
if no leaf already kept is its judged duplicate.

    python -m m5.scicode.leaf_merge run      # WORK/leaf_merge.jsonl (resumable)
    python -m m5.scicode.leaf_merge write    # taxonomy/leaf_merge.json (titles dropped as duplicates)
"""
import asyncio
import json
import re
import sys

import numpy as np

from . import taxonomy as T
from .common import Alloc, append_line, read_jsonl, safe_write

EMB = T.WORK / 'emb' / 'leaves_bge.npy'   # rows of all_leaf_rows(), in order (re-embedded after round 2)
CAND_COS, K = 0.76, 10   # the pilot's known duplicate leaf pairs sat at 0.76-0.93

RUBRIC = """You deduplicate a list of topic seeds for scientific programming problems. Each seed names a system or model and a quantity to compute. Two seeds are DUPLICATES when a problem written from one would very likely be essentially the same problem as one written from the other: the same specific system or model and the same main computed quantity (or one seed's computation is the core of the other's), whatever the wording, parameter values or method details. Seeds on the same broad topic but with a different system, a different main observable, or a clearly different computation are NOT duplicates.

You see one seed (S) and a numbered list of candidates. Answer with only a JSON object: {"duplicates": [<numbers of the candidates that duplicate S>], "reason": "<one short sentence>"}"""


def rows():
    return T.all_leaf_rows()


def candidates():
    E = np.load(EMB)
    S = E @ E.T
    np.fill_diagonal(S, 0)
    out = {}
    for i in range(len(S)):
        idx = [int(j) for j in np.argsort(-S[i])[:K] if S[i, j] > CAND_COS]
        if idx:
            out[i] = idx
    return out


async def run():
    from .generate import DS
    R = rows()
    C = candidates()
    path = T.WORK / 'leaf_merge.jsonl'
    done = {r['i'] for r in read_jsonl(path)}
    todo = [i for i in C if i not in done]
    print(f'{len(C)} leaves with candidates, {len(todo)} to judge', flush=True)
    ds, alloc = DS(), Alloc()

    async def one(i):
        cs = C[i]
        user = (f"S: {R[i][1]['title']}: {R[i][1]['gist']}\n\nCandidates:\n"
                + '\n'.join(f"{n + 1}. {R[j][1]['title']}: {R[j][1]['gist']}" for n, j in enumerate(cs)))
        for _ in range(3):
            g = await ds.chat(user, system=RUBRIC)
            m = re.search(r'\{.*\}', g['content'] or '', re.S)
            try:
                v = json.loads(m.group(0))
                dups = [cs[int(n) - 1] for n in v.get('duplicates', []) if 1 <= int(n) <= len(cs)]
                append_line(path, json.dumps({'i': i, 'cands': cs, 'dups': dups, 'reason': v.get('reason', '')}))
                return
            except (AttributeError, ValueError, TypeError):
                pass

    pending = set()
    for i in todo:
        while len(pending) >= max(1, alloc.ds_cap()):
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.add(asyncio.create_task(one(i)))
    await asyncio.gather(*pending)


def write():
    R = rows()
    edges = {}
    for r in read_jsonl(T.WORK / 'leaf_merge.jsonl'):
        for j in r['dups']:
            edges.setdefault(r['i'], set()).add(j)
            edges.setdefault(j, set()).add(r['i'])
    n1 = sum(1 for _, x in R if x.get('round', 1) == 1)   # round-1 leaves first, then round 2 (appended)
    rng = np.random.default_rng(0)
    order = list(rng.permutation(n1)) + list(n1 + rng.permutation(len(R) - n1))
    kept, drop = set(), []
    for i in order:
        if edges.get(int(i), set()) & kept:
            drop.append(R[int(i)][1]['title'])
        else:
            kept.add(int(i))
    safe_write(T.HERE / 'leaf_merge.json', json.dumps(
        {'note': 'Leaf titles dropped as duplicates of a kept leaf on the same subject (m5/scicode/leaf_merge.py).',
         'n_edges': sum(len(v) for v in edges.values()) // 2, 'drop_titles': sorted(set(drop))}, indent=0,
        ensure_ascii=False))
    print(len(edges), 'leaves in duplicate pairs;', len(drop), 'dropped')


if __name__ == '__main__':
    asyncio.run(run()) if sys.argv[1] == 'run' else write()

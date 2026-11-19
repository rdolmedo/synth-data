"""GDPval: place one run against fixed reference runs (c30's grading, #375/#378): one judged pair per finished
attempt per reference, in one A/B order.

  python -m aa_evals.gdpval.refs plan  --run RUN --pool POOL [--refs R1,R2]
  python -m aa_evals.gdpval.refs judge --run RUN --pool POOL [--judge-arm ARM] [--judge-endpoint SPEC] [--concurrency 16]

This is c30's `aa-evals-coord/c30/ops/gdpval_refs/refs.py` (itself gdpval_pool_20261109's ladder.py), with the run,
the pool, the references and the arm as arguments; the draws are unchanged, so a c30 pool re-planned here is the same.
The default references are the two M3.6 runs every c30 model was placed against (#183).

plan:  for every finished attempt of RUN (pool.finished: status ok, render done), one pair per reference run: the
       reference attempt of the same task is drawn by sha256("ladder-ref|731|<run>|<task>|<repeat>|<ref run>")[:16]
       mod (number of finished reference repeats, sorted). Pair ids are pool.plan's (sides sorted). Each pair gets
       one order, by Stage 1's order hash: sha256("stage-order|731|<pair>")[:16] mod 2, 0 = AB. A reference with no
       finished attempt on the task skips that pair (counted in pool.json). The plan is frozen: plan refuses to
       overwrite pairs.jsonl.
judge: pool.judge_pool on just the planned (pair, order) items (the other order of every pair is marked done in
       memory). Pass 2 retries once every item without a winner (an error row, or no row after an infra exception).
       Retried rows are appended; the last row per (pair, order) is the result, and matches/<arm>.jsonl is rewritten
       that way if a retry happened (the raw log is kept beside it). Re-running judge resumes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from aa_evals.gdpval import pool as P
from aa_evals.gdpval.run import now

SEED = 731
REFS = ('m36-qwen38-xhigh-gdpval-full', 'm36-lightning-gdpval-full')
ARM = 'qwen38-27b-medium-pin'


def u(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def order_of(pid: str) -> str:
    return 'AB' if u('stage-order', SEED, pid) % 2 == 0 else 'BA'


def make_pairs(run: str, refs: list[str], fin: dict) -> tuple[list[dict], list[dict]]:
    """(pairs, skipped) from fin = {run id: {task: [finished repeats]}}."""
    pairs, skipped = [], []
    for tid, reps in sorted(fin[run].items()):
        for rep in sorted(reps):
            for ref in refs:
                rr = sorted(fin[ref].get(tid, []))
                if not rr:
                    skipped.append({'task_id': tid, 'repeat': rep, 'ref': ref})
                    continue
                orep = rr[u('ladder-ref', SEED, run, tid, rep, ref) % len(rr)]
                a, b = sorted([(run, rep), (ref, orep)])
                pid = f'{tid}|{a[0]}|{a[1]}|{b[0]}|{b[1]}'
                pairs.append({'pair': pid, 'task_id': tid, 'a': {'run': a[0], 'repeat': a[1]},
                              'b': {'run': b[0], 'repeat': b[1]}, 'order': order_of(pid)})
    assert len({p['pair'] for p in pairs}) == len(pairs)
    return pairs, skipped


def plan(run: str, pool: Path, refs: list[str]) -> dict:
    if (pool / 'pairs.jsonl').exists():
        raise SystemExit(f'{pool}/pairs.jsonl exists (frozen plan)')
    fin = {r: P.finished(r) for r in [run, *refs]}
    pairs, skipped = make_pairs(run, refs, fin)
    pool.mkdir(parents=True, exist_ok=True)
    meta = {'runs': [run, *refs], 'run': run, 'refs': list(refs), 'seed': SEED, 'planned_at': now(),
            'n_run_attempts': sum(map(len, fin[run].values())), 'n_pairs': len(pairs),
            'orders': {o: sum(p['order'] == o for p in pairs) for o in ('AB', 'BA')},
            'pairs_per_ref': {r: sum(r in (p['a']['run'], p['b']['run']) for p in pairs) for r in refs},
            'n_skipped': len(skipped), 'skipped': skipped,
            'ref_hash': 'sha256("ladder-ref|731|<run>|<task>|<repeat>|<ref run>")[:16] mod n finished ref repeats (sorted)',
            'order_hash': 'sha256("stage-order|731|<pair>")[:16] mod 2: 0 = AB, 1 = BA (as Stage 1)',
            'note': 'one order per pair; "order" in pairs.jsonl is the judged order'}
    (pool / 'pool.json').write_text(json.dumps(meta, indent=1) + '\n')   # before pairs.jsonl: that one freezes
    tmp = pool / 'pairs.jsonl.partial'
    tmp.write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    tmp.rename(pool / 'pairs.jsonl')
    print(json.dumps({k: v for k, v in meta.items() if k != 'skipped'}), flush=True)
    return meta


def judge(pool: Path, arm: str, endpoint: str | None, concurrency: int, images: int, deadline: float = 0.0) -> dict:
    out = pool / 'matches' / f'{arm}.jsonl'
    out.parent.mkdir(exist_ok=True)
    plan_rows = P.read_jsonl(pool / 'pairs.jsonl')
    wanted = {(r['pair'], r['order']) for r in plan_rows}
    other = [{'pair': r['pair'], 'order': 'BA' if r['order'] == 'AB' else 'AB', 'winner': '__not_judged__'}
             for r in plan_rows]
    orig = P.read_jsonl
    state = {'retry': False}

    def read_jsonl(path):
        rows = orig(path)
        if Path(path) == out:
            if state['retry']:   # an item without a winner is not done: judge it once more
                rows = [r for r in rows if r.get('winner')]
            return rows + other
        return rows

    if deadline:
        def stop():
            print(f'deadline {deadline} reached: exiting (budget cap)', flush=True)
            os._exit(3)
        timer = threading.Timer(max(0.0, deadline - time.time()), stop)
        timer.daemon = True
        timer.start()
    print(f'{len(wanted)} (pair, order) items wanted', flush=True)
    P.read_jsonl = read_jsonl
    try:
        asyncio.run(P.judge_pool(pool, arm, endpoint, concurrency, images))
        rows = orig(out)
        ok = {(r['pair'], r['order']) for r in rows if r.get('winner')}
        todo = wanted - ok
        print(now(), f'pass 1: {len(ok & wanted)} with a winner, {len(todo)} to retry', flush=True)
        if todo:
            state['retry'] = True
            asyncio.run(P.judge_pool(pool, arm, endpoint, concurrency, images))
            rows = orig(out)
            last = {}
            for r in rows:
                last[(r['pair'], r['order'])] = r
            if len(last) < len(rows):
                raw = out.with_name(out.stem + '.raw.jsonl')
                raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                tmp = out.with_name(out.name + '.partial')
                tmp.write_text(''.join(json.dumps(r) + '\n' for r in last.values()))
                tmp.rename(out)
            ok = {k for k, r in last.items() if r.get('winner')}
    finally:
        P.read_jsonl = orig
    res = {'wanted': len(wanted), 'with_winner': len(ok & wanted), 'failures': len(wanted - ok)}
    print(now(), f"final: {res['with_winner']} of {res['wanted']} with a winner; failures {res['failures']}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['plan', 'judge'])
    ap.add_argument('--run', required=True, help='the GDPval run to place, e.g. c31-gemma26-gdpval')
    ap.add_argument('--pool', required=True, help='pool name under runs/gdpval/_pools, e.g. c31-refs-gemma26')
    ap.add_argument('--refs', default=','.join(REFS), help='reference runs (comma-separated)')
    ap.add_argument('--judge-arm', default=ARM)
    ap.add_argument('--judge-endpoint', default=None, help="default: the arm's server")
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--judge-images', type=int, default=1)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--min-success', type=float, default=0.0,
                    help='judge: exit 1 if fewer than this share of the planned items have a winner')
    a = ap.parse_args(argv)
    refs = [r for r in a.refs.split(',') if r]
    if a.run in refs:
        raise SystemExit(f'{a.run} is one of the references')
    pool = P.POOLS / a.pool
    if a.what == 'plan':
        plan(a.run, pool, refs)
        return 0
    res = judge(pool, a.judge_arm, a.judge_endpoint, a.concurrency, a.judge_images, a.deadline_epoch)
    if res['wanted'] and res['with_winner'] < a.min_success * res['wanted']:
        raise SystemExit(1)
    return 0


if __name__ == '__main__':
    main()

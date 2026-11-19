"""M5 ladder SFT datasets (user 2026-11-15): equal shares across 9 sources (the 8 M5 benchmarks + TB), M=2 traces per
task, finishing at the train cap, nested task subsets by budget; E=2 is the trainer's --epochs.

  * First traces: the 3B arms' packed records (V/data/W-3b, S-3b) and TB's share of ib-6p0b (tb/include.txt), reused
    byte for byte, except records above the train cap, which are replaced by their finishing version (pool rows with
    `replaces`). Val = every first-trace val record (unchanged, in every budget).
  * Second traces: pool rows of kind t2_flat / t2_pi (same task groups).
  * Budget D SFT tokens: every source takes the prefix of its task order whose unique train tokens (first + second
    traces, Snowball tokens) are closest to D / (E x 9); a smaller source contributes all its tasks. 'full' = all.
    Task order: M5 sources sha256(f'{seed}:{group}'); TB its ladder order (tb/tasks.txt, nests the TB ladder cuts).
    Budgets nest: a smaller budget's tasks are a prefix of a larger one's in every source.
  * --exclude FILE (11-17, decontamination): JSON {source: [task ids]}; those tasks' train records (first, finishing and
    second traces) are dropped before the budget cut. The remaining tasks keep their order, so every budget is still a
    prefix, and a subset of the same budget without --exclude. Val records are unchanged (never trained on).
  * M=2 always (user 11-17): every selected task has exactly two traces (first + second; SciCode: two per step; TB: its
    two ib-6p0b records). Tasks without a full second trace are dropped before the budget cut, so the budget is met by
    the number of tasks, never by M=1 tasks. --keep-m1 restores the v1 behaviour (m5l-* v1 kept 41 such tasks: AB 1,
    GDPval 10, TB-Sci-lite 30, via --allow-incomplete).

  python assemble.py --budget 3e9 --name m5l-3p0b [--exclude excl.json] [--dry-run]   # --budget full = the whole mixture
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

L = Path('/data/aai/m5/ladder')
S = L / 'sft'
V = Path('/data/aai/m5/validation/data')
IB = Path('/data/aai/grug-b200/snowflake/tb/intermediate/data/ib-6p0b')
sys.path.insert(0, str(S))
import render as RD   # noqa: E402
from data import write_dataset   # noqa: E402

FIRST = {'automationbench': V / 'W-3b', 'gdpval': V / 'W-3b', 'gdppdf': V / 'W-3b', 'omniscience': V / 'W-3b',
         'critpt': V / 'S-3b', 'hle': V / 'S-3b', 'scicode': V / 'S-3b', 'tbsci_lite': V / 'S-3b', 'tb': IB}
E, PACK_LEN = 2, 131072


def first_rows(suite):
    """[(row index in its dataset, row)] of the suite's first-trace records (train: TB only its share; val: all)."""
    d = FIRST[suite]
    inc = set((L / 'tb' / 'include.txt').read_text().split()) if suite == 'tb' else None
    out = []
    for i, r in enumerate(csv.DictReader((d / 'sequences.csv').open())):
        if suite == 'tb':
            if r['split'] == 'val' or r['id'] in inc:
                out.append((i, {**r, 'arm': 'tb'}))
        elif r['arm'] == suite:
            out.append((i, r))
    if suite == 'tb':
        assert sum(r['split'] == 'train' for _, r in out) == len(inc)
    return out


def pool_rows(suite):
    p = S / 'pool' / suite / 'index.jsonl'
    return [json.loads(l) for l in p.open()] if p.exists() else []


def task_id(group):
    return group.split(':', 1)[1] if ':' in group else group


def plan(budget, seed, require_complete, exclude=None, keep_m1=False):
    """Per source: selected train groups (in order), the rows to write, token counts."""
    out = {}
    for suite in FIRST:
        excl = set((exclude or {}).get(suite, ()))
        rows, pool = first_rows(suite), pool_rows(suite)
        repl = {p['replaces']: k for k, p in enumerate(pool) if p.get('replaces') and p['kind'] in ('fin_flat', 'fin_pi')}
        t2repl = {p['replaces']: k for k, p in enumerate(pool) if p['kind'] == 'fin_t2'}   # resampled second traces
        t2 = collections.defaultdict(list)
        for k, p in enumerate(pool):
            if p['kind'] in ('t2_flat', 't2_pi') and task_id(p['group']) not in excl:
                t2[p['group']].append(t2repl.get(p['id'], k))
        tok = collections.Counter()
        train = collections.defaultdict(list)
        excluded_found = set()
        for i, r in rows:
            if r['split'] != 'train':
                continue
            if task_id(r['group']) in excl:
                excluded_found.add(task_id(r['group']))
                continue
            train[r['group']].append(i)
            tok[r['group']] += pool[repl[r['id']]]['n_tokens'] if r['id'] in repl else int(r['n_tokens'])
        dup_t2 = 0
        if suite in ('automationbench', 'gdpval', 'critpt', 'tbsci_lite'):   # Pi: one second trace per task
            for g, ks in t2.items():   # a split run's overshoot (t2/fence.py): keep the lowest id (run A's)
                if len(ks) > 1:
                    dup_t2 += len(ks) - 1
                    t2[g] = [min(ks, key=lambda k: pool[k]['id'])]
        for g, ks in t2.items():
            assert g in train, (suite, g, 'second trace of a task without a first trace')
            tok[g] += sum(pool[k]['n_tokens'] for k in ks)
        if suite == 'tb':
            order = [g for g in (L / 'tb' / 'tasks.txt').read_text().split() if g in train]
        else:
            order = sorted(train, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        assert set(order) == set(train), (suite, len(order), len(train))
        if suite == 'tb':   # TB's ib-6p0b share carries both traces as first-trace records
            not_m2 = [g for g in train if len(train[g]) != 2]
        else:               # one second trace per first-trace record (SciCode: per step)
            not_m2 = [g for g in train if len(t2.get(g, ())) != len(train[g])]
        missing_t2 = len(not_m2)
        if require_complete and missing_t2 and keep_m1:
            raise SystemExit(f'{suite}: {missing_t2} train tasks have no full second trace (pass --allow-incomplete)')
        if not keep_m1:
            for g in not_m2:
                del train[g]
                t2.pop(g, None)
            order = [g for g in order if g in train]
        cums = np.cumsum([tok[g] for g in order])
        if budget == 'full':
            n = len(order)
        else:
            target = float(budget) / E / len(FIRST)
            n = min(len(order), int(np.argmin(np.abs(cums - target))) + 1)
        sel = order[:n]
        out[suite] = {'rows': rows, 'pool': pool, 'repl': repl, 't2': t2, 'groups': sel, 'train_groups': len(train),
                      'unique_tokens': int(cums[n - 1]), 'missing_t2_tasks': missing_t2, 'duplicate_t2_dropped': dup_t2,
                      'replaced_in_selection': sum(1 for g in sel for i in train[g] if rows_by_index(rows)[i]['id'] in repl),
                      't2_records': sum(len(t2[g]) for g in sel), 'excluded_listed': len(excl),
                      'excluded_train_tasks': len(excluded_found), 'dropped_not_m2': 0 if keep_m1 else missing_t2}
        if not keep_m1:   # M=2 on every selected task
            assert all((len(train[g]) == 2) if suite == 'tb' else (len(t2[g]) == len(train[g])) for g in sel), suite
    return out


_RBI = {}


def rows_by_index(rows):
    key = id(rows)
    if key not in _RBI:
        _RBI[key] = {i: r for i, r in rows}
    return _RBI[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', required=True, help="SFT tokens at E=2 (e.g. 3e9) or 'full'")
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-incomplete', action='store_true')
    ap.add_argument('--exclude', help='JSON {source: [task ids]} of decontamination exclusions')
    ap.add_argument('--keep-m1', action='store_true', help='v1 behaviour: keep tasks without a second trace (M=1)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    exclude = json.loads(Path(a.exclude).read_text()) if a.exclude else None
    P = plan(a.budget, a.seed, not a.allow_incomplete, exclude, a.keep_m1)
    summary = {s: {k: v for k, v in p.items() if k not in ('rows', 'pool', 'repl', 't2', 'groups')} | {'tasks': len(p['groups'])}
               for s, p in P.items()}
    total = sum(p['unique_tokens'] for p in P.values())
    print(json.dumps({'budget': a.budget, 'unique_tokens': total, 'sft_tokens_at_E2': total * E, 'sources': summary}, indent=1),
          flush=True)
    if a.dry_run:
        return
    final = S / 'data' / a.name
    assert not (final / 'COMPLETE.json').exists(), final
    # stage on node-local disk, verify there, dd to /data (user 11-15: writing .npy straight to Lustre + re-reading it
    # for the verify was ~half of a 1 h 50 assembly); assemble.sub's request_disk must hold the largest budget (~28 GB)
    out = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.mkdtemp(dir='/tmp')) / f'assemble-{a.name}'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    records, results, verify = [], [], []
    for suite, p in P.items():
        d = FIRST[suite]
        tok = np.load(d / 'tokens.npy', mmap_mode='r')
        msk = np.load(d / 'loss_mask.npy', mmap_mode='r')
        offs = np.load(d / 'seq_offsets.npy', mmap_mode='r')
        ptok = np.load(S / 'pool' / suite / 'tokens.npy', mmap_mode='r') if p['pool'] else None
        pmsk = np.load(S / 'pool' / suite / 'mask.npy', mmap_mode='r') if p['pool'] else None
        poff = np.load(S / 'pool' / suite / 'offsets.npy', mmap_mode='r') if p['pool'] else None
        sel = set(p['groups'])

        def pool_res(k):
            q = p['pool'][k]
            lo, hi = int(poff[k]), int(poff[k + 1])
            return {'ids': np.array(ptok[lo:hi]), 'mask': np.array(pmsk[lo:hi]), 'n_full': q['n_full'], 'n_turns': q['n_turns'],
                    'kept_turns': q['kept_turns'], 'truncated': q['truncated'], 'special_in_text': q['special_in_text'],
                    'prefix_checked': 0, 'history_retok_turns': 0, 'empty_reasoning_turns': 0}

        for i, r in p['rows']:
            if r['split'] == 'train' and r['group'] not in sel:
                continue
            if r['split'] == 'train' and r['id'] in p['repl']:
                k = p['repl'][r['id']]
                records.append({'id': r['id'], 'group': r['group'], 'arm': suite, 'split': 'train', 'source': 'finish'})
                results.append(pool_res(k))
                continue
            lo, hi = int(offs[i]), int(offs[i + 1])
            records.append({'id': r['id'], 'group': r['group'], 'arm': suite, 'split': r['split'], 'source': 'first'})
            results.append({'ids': np.array(tok[lo:hi]), 'mask': np.array(msk[lo:hi]), 'n_full': int(r['n_tokens_full']),
                            'n_turns': int(r['n_turns']), 'kept_turns': int(r['kept_turns']),
                            'truncated': r['truncated'] == 'True', 'special_in_text': int(r['special_in_text']),
                            'prefix_checked': 0, 'history_retok_turns': 0, 'empty_reasoning_turns': 0})
            verify.append((len(results) - 1, d, i))
        for g in p['groups']:
            for k in p['t2'].get(g, []):
                q = p['pool'][k]
                records.append({'id': q['id'], 'group': q['group'], 'arm': suite, 'split': 'train', 'source': 'second'})
                results.append(pool_res(k))
        print(suite, 'collected', len(records), round(time.time() - t0), 's', flush=True)
    for r in results:
        assert len(r['ids']) <= PACK_LEN + 1, 'record longer than the pack'
    meta = {'source': 'm5-ladder', 'budget': a.budget, 'seed': a.seed, 'exclude': a.exclude, 'm2_only': not a.keep_m1, 'epochs_intended': E, 'sources': summary,
            'first_traces': {s: str(d) for s, d in FIRST.items()}, 'pool': str(S / 'pool'),
            'rules': 'records.py / render.py docstrings; finishing replaces first traces above the train cap',
            'template': RD.B.TEMPLATE, 'tokenizer': RD.B.TOKENIZER}
    R = RD.renderer()
    stats = write_dataset(out, records, results, PACK_LEN, R.tok, meta, seed=a.seed)
    # unchanged first traces: byte for byte
    nt, nm, no = (np.load(out / f, mmap_mode='r') for f in ('tokens.npy', 'loss_mask.npy', 'seq_offsets.npy'))
    cache = {}
    for j, d, i in verify:
        if d not in cache:
            cache[d] = tuple(np.load(d / f, mmap_mode='r') for f in ('tokens.npy', 'loss_mask.npy', 'seq_offsets.npy'))
        t, m, o = cache[d]
        assert np.array_equal(nt[no[j]:no[j + 1]], t[o[i]:o[i + 1]]) and np.array_equal(nm[no[j]:no[j + 1]], m[o[i]:o[i + 1]])
    by = collections.Counter((r['arm'], r['split'], r['source']) for r in records)
    (out / 'manifest.json').write_text(json.dumps({'sources': summary, 'records': {'|'.join(k): v for k, v in sorted(by.items())},
                                                   'unique_tokens': total, 'builder': str(Path(__file__).resolve())}, indent=1))
    part = final.with_name(final.name + '.partial')
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    for f in sorted(out.iterdir()):
        if f.stat().st_size > 1 << 20:
            subprocess.run(['dd', f'if={f}', f'of={part / f.name}', 'bs=64M', 'oflag=direct', 'conv=fsync', 'status=none'], check=True)
        else:
            shutil.copyfile(f, part / f.name)
        assert (part / f.name).stat().st_size == f.stat().st_size, f
    if final.exists():   # an earlier incomplete attempt (no COMPLETE.json, asserted above)
        shutil.rmtree(final)
    part.rename(final)
    shutil.rmtree(out, ignore_errors=True)
    (final / 'COMPLETE.json').write_text(json.dumps({'first_traces_verified': len(verify), 'train_packs': stats['packs']['train']['n_packs'],
                                                     'elapsed_s': round(time.time() - t0)}, indent=1))
    print('DONE', final, stats['packs'], flush=True)


if __name__ == '__main__':
    main()

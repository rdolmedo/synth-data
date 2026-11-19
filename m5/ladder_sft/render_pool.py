"""Render every new record of one source (records/{t2,fin}_*/<suite>.jsonl) once, with the cut-span loss rule, into
sft/pool/<suite>/: tokens.npy (int32), mask.npy (uint8), offsets.npy, index.jsonl (one row per record: id, group, arm,
split, kind, replaces, the Renderer.process scalars, n_tokens, n_loss, meta). assemble.py packs budgets from these.
  python render_pool.py <suite> [--workers 32]"""
import argparse
import collections
import concurrent.futures
import json
from pathlib import Path
import sys
import time

import numpy as np

S = Path('/data/aai/m5/ladder/sft')
sys.path.insert(0, str(S))
import render as RD   # noqa: E402

FIELDS = ('n_full', 'n_turns', 'kept_turns', 'truncated', 'special_in_text', 'prefix_checked', 'history_retok_turns',
          'empty_reasoning_turns')
R = None


def init():
    global R
    R = RD.renderer()


def one(line):
    rec = json.loads(line)
    try:
        res = RD.render(R, rec)
    except AssertionError as e:
        return None, {'id': rec['id'], 'error': str(e)[:300]}
    row = {k: rec.get(k) for k in ('id', 'group', 'arm', 'split', 'kind', 'replaces', 'round')}
    row.update({k: (bool(res[k]) if k == 'truncated' else int(res[k])) for k in FIELDS})
    row.update(n_tokens=len(res['ids']), n_loss=int(res['mask'].sum()), cut_audit=res['cut_audit'],
               meta={k: v for k, v in (rec.get('meta') or {}).items() if k in ('natural', 'warned', 'cut', 'end',
                                                                                'context_limit', 'near_window_bound')})
    return (np.asarray(res['ids'], dtype=np.int32), res['mask']), row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suite')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    files = [f for k in ('t2_flat', 't2_pi', 'fin_flat', 'fin_pi', 'fin_t2') for f in [S / 'records' / k / f'{a.suite}.jsonl'] if f.exists()]
    out = S / 'pool' / a.suite
    out.mkdir(parents=True, exist_ok=True)
    ids, masks, rows, errors, t0 = [], [], [], [], time.time()
    for f in files:
        with f.open() as fh, concurrent.futures.ProcessPoolExecutor(a.workers, initializer=init) as pool:
            for k, (arr, row) in enumerate(pool.map(one, fh, chunksize=16), 1):
                if arr is None:
                    errors.append({**row, 'file': f.name})
                    continue
                ids.append(arr[0])
                masks.append(arr[1])
                rows.append(row)
                if k % 5000 == 0:
                    print(f.parent.name, k, round(time.time() - t0), 's', flush=True)
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in ids])
    np.save(out / 'tokens.npy', np.concatenate(ids) if ids else np.zeros(0, np.int32))
    np.save(out / 'mask.npy', np.concatenate(masks).astype(np.uint8) if masks else np.zeros(0, np.uint8))
    np.save(out / 'offsets.npy', offsets)
    with (out / 'index.jsonl').open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    (out / 'errors.json').write_text(json.dumps(errors, indent=1))
    c = collections.Counter(r['kind'] for r in rows)
    s = {'suite': a.suite, 'files': [str(f) for f in files], 'records': dict(c), 'errors': len(errors),
         'tokens': int(offsets[-1]), 'loss_tokens': int(sum(r['n_loss'] for r in rows)),
         'truncated': sum(r['truncated'] for r in rows), 'cut_spans': sum(len(r['cut_audit']) for r in rows),
         'cut_tokens_untrained_boundary': sum(x['real_tokens'] - x['trained'] for r in rows for x in r['cut_audit']),
         'elapsed_s': round(time.time() - t0)}
    (out / 'stats.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)


if __name__ == '__main__':
    main()

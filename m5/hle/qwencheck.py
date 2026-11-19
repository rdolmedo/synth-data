"""Qwen3.5 render check of exported shards with the SFT branch's renderer (sft_qwen35/data.py Renderer.process:
the shipped chat template, append-only spans, token-boundary and prefix checks). Run with the qwen35sft venv.

    python -m m5.hle.qwencheck --code SFT_CODE --inputs 'E/traces-*.jsonl.gz' --out E/prepcheck/qwen.json
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import sys
from collections import Counter

R = None


def init(code, model):
    global R
    sys.path.insert(0, f'{code}/sft_qwen35')
    from data import Renderer
    R = Renderer(model)


def work(line):
    rec = json.loads(line)
    r = R.process(rec['messages'], max_len=131072)
    if 'error' in r:
        return rec['trial'], r['error'], 0, 0, False
    return rec['trial'], None, len(r['ids']), int(r['mask'].sum()), bool(r.get('truncated'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='/data/aai/models/qwen3.5-4b')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    lines = [l for f in sorted(glob.glob(a.inputs)) for l in gzip.open(f, 'rt', encoding='utf-8')]
    errors, tok, loss, trunc, n_ok = Counter(), 0, 0, 0, 0
    examples = {}
    with mp.Pool(a.workers, initializer=init, initargs=(a.code, a.model)) as pool:
        for trial, err, n, m, t in pool.imap_unordered(work, lines, chunksize=16):
            if err:
                errors[err] += 1
                examples.setdefault(err, trial)
            else:
                n_ok, tok, loss, trunc = n_ok + 1, tok + n, loss + m, trunc + t
    res = {'records': len(lines), 'ok': n_ok, 'errors': dict(errors), 'error_examples': examples,
           'qwen_tokens': tok, 'qwen_loss_tokens': loss, 'truncated_at_131072': trunc}
    with open(a.out, 'w') as f:
        f.write(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()

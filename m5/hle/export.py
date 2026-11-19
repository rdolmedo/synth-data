"""Export kept Super traces as SFT shards in the format sft/prepare_data.py consumes (as M1's production traces).

    python -m m5.hle.export --traces 'T/traces*.jsonl' --problems 'P/problems*.jsonl' \
        --excluded W/excluded.json --out /data/aai/m5/hle/export [--target-tokens 330e6]

Kept = the trace ended with a final answer (finish_reason stop, non-empty content), its problem passed the
problem-side checks, decontamination did not flag it, and no HLE canary marker appears anywhere in the record. Every other trace stays in the raw trace files, flagged.
With --target-tokens, traces are taken in completion order until the budget (prompt + completion tokens by the Super
tokenizer, as the server counted the full sequence) is covered; the tail is cut.

Shard record: {"trial", "task", "run", "messages": [system, user, assistant{content, reasoning_content}], "meta"}.
The user content is the eval's single text part as a string (the chat template renders it identically).
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from .decon import CANARY_MARKERS

PER_SHARD = 1000


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def put(src, dst):
    if src.stat().st_size > (1 << 20):
        subprocess.run(['dd', f'if={src}', f'of={dst}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    else:
        subprocess.run(['cp', str(src), str(dst)], check=True)
    if dst.stat().st_size != src.stat().st_size:
        raise SystemExit(f'size mismatch {dst}')
    want, got = sha256(src), sha256(dst)
    if want != got:
        raise SystemExit(f'checksum mismatch {dst}')
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--problems', required=True)
    ap.add_argument('--excluded')
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-tokens', type=float, default=0)
    ap.add_argument('--run-name', default='m5-hle-v1')
    ap.add_argument('--scratch', default=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp')
    a = ap.parse_args()
    probs = {}
    for p in sorted(glob.glob(a.problems)):
        for l in open(p):
            r = json.loads(l)
            probs[r['id']] = r
    excl = set(json.loads(Path(a.excluded).read_text())['excluded']) if a.excluded else set()
    traces = {}
    for p in sorted(glob.glob(a.traces)):
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get('status') == 'ok' and r['id'] not in traces:   # one trace per problem: the first
                traces[r['id']] = r
    why = collections.Counter()
    keep = []
    for t in sorted(traces.values(), key=lambda t: t.get('finished_at') or 0):
        p = probs.get(t['id'])
        if p is None or p.get('status') != 'ok':
            why['problem_not_ok'] += 1
        elif t['id'] in excl:
            why['decontam'] += 1
        elif t['hit_context']:
            why['context_limit'] += 1
        elif not t['has_answer']:
            why['no_answer'] += 1
        elif any(m in json.dumps(t['messages'], ensure_ascii=False) + t['content'] + t['reasoning_content']
                 for m in CANARY_MARKERS):
            why['canary'] += 1
        else:
            keep.append(t)
    total, cut = 0, 0
    final = []
    for t in keep:
        if a.target_tokens and total >= a.target_tokens:
            cut += 1
            continue
        final.append(t)
        total += t['tokens_total']
    why['tail_cut'] = cut
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch) / 'm5hle-export'
    scratch.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    shards = {}
    for si in range(0, len(final), PER_SHARD):
        local = scratch / f'traces-{si // PER_SHARD:04d}.jsonl.gz'
        with gzip.open(local, 'wt', encoding='utf-8') as f:
            for t in final[si:si + PER_SHARD]:
                sysm, user = t['messages']
                utext = user['content'] if isinstance(user['content'], str) else ''.join(
                    c['text'] for c in user['content'] if c.get('type') == 'text')
                rec = {'trial': t['id'], 'task': t['id'], 'run': a.run_name,
                       'messages': [{'role': 'system', 'content': sysm['content']},
                                    {'role': 'user', 'content': utext},
                                    {'role': 'assistant', 'content': t['content'],
                                     'reasoning_content': t['reasoning_content']}],
                       'meta': {k: t['spec'].get(k) for k in ('partition', 'category', 'subject', 'answer_type', 'leaf')}
                       | {'prompt_tokens': t['prompt_tokens'], 'completion_tokens': t['completion_tokens'],
                          'engine': t.get('engine')}}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                stats[t['spec']['partition']] += 1
        n = len(final[si:si + PER_SHARD])
        with gzip.open(local, 'rt', encoding='utf-8') as f:
            if sum(1 for _ in f) != n:
                raise SystemExit(f'{local}: line count != {n}')
        shards[local.name] = {'records': n, 'bytes': local.stat().st_size, 'sha256': put(local, out / local.name),
                              'tokens': sum(t['tokens_total'] for t in final[si:si + PER_SHARD])}
        local.unlink()
    summary = {'run': a.run_name, 'n_traces_raw': len(traces), 'n_exported': len(final), 'tokens': total,
               'excluded': dict(why), 'by_partition': dict(stats), 'target_tokens': a.target_tokens,
               'shards': shards,
               'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k, v in sorted(shards.items())))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

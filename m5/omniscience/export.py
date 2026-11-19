"""Export m5-omniscience Super traces for SFT (the format sft/prepare_data.py consumes, as M1's production traces).

  python -m m5.omniscience.export OUT --runs RUN [RUN ...] [--excluded WORK/excluded.jsonl] [--token-target T]

Every trace is kept in traces.jsonl upstream; exclusions happen here and are counted:
  decon        flagged by decon/run.py (judge or canary)
  no_submit    no final answer (includes context-limit and timeout endings)
  dup          a problem traced twice (resume overlap): the first trace wins
With --token-target, rows are kept in trace order (or, with --seed, in a seeded random order over problems; one trace
per problem, so this is random-by-task) until the Super-side total tokens reach the target.
Shards and problems.jsonl.zst are listed with their sha256 in OUT/SHA256SUMS (sha256sum -c format).
Writes OUT/trajectories-NNNN.jsonl.gz (10k rows each), OUT/problems.jsonl.zst (questions only, no answers),
OUT/export_stats.json.
"""
import argparse
import collections
import gzip
import json
import random
from pathlib import Path

import zstandard

from m5.omniscience import fsio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--excluded', default=None)
    ap.add_argument('--token-target', type=int, default=0)
    ap.add_argument('--shard', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=None, help='seeded random order over problems before the token cut')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    excl = set()
    if a.excluded:
        excl = {json.loads(l)['problem_id'] for l in open(a.excluded)}
    gen = {}
    traces = []
    for r in a.runs:
        for l in open(Path(r) / 'gen.jsonl'):
            g = json.loads(l)
            if g.get('accepted'):
                gen[g['problem_id']] = g
        traces += [json.loads(l) for l in open(Path(r) / 'traces.jsonl')]
    counts = collections.Counter()
    seen, ok = set(), []
    for t in traces:
        counts['traces'] += 1
        pid = t['problem_id']
        if pid in seen:
            counts['drop:dup'] += 1
            continue
        seen.add(pid)
        if pid in excl:
            counts['drop:decon'] += 1
            continue
        if not t['answer'].strip():
            counts['drop:no_submit'] += 1
            counts[f"drop:no_submit:{t['finish_reason']}"] += 1
            continue
        ok.append(t)
    if a.seed is not None:
        random.Random(a.seed).shuffle(ok)
    kept, tok = [], 0
    for t in ok:
        if a.token_target and tok >= a.token_target:
            counts['drop:over_target'] += 1
            continue
        tok += t.get('total_tokens') or 0
        kept.append(t)
    shards = []
    for s in range(0, len(kept), a.shard):
        p = out / f'trajectories-{s // a.shard:04d}.jsonl.gz'
        lines = []
        for t in kept[s:s + a.shard]:
            g = gen.get(t['problem_id'], {})
            msgs = t['messages'] + [{'role': 'assistant', 'content': t['answer'],
                                     'reasoning_content': t['reasoning']}]
            lines.append(json.dumps({
                'trial': t['problem_id'], 'task': t['problem_id'], 'run': 'm5-omniscience',
                'benchmark': 'AA-Omniscience', 'domain': t['domain'], 'subtopic': t['subtopic'], 'leaf': t['leaf'],
                'prompt_id': t['prompt_id'], 'source': g.get('source'), 'doc_id': g.get('doc_id'),
                'finish_reason': t['finish_reason'], 'capped': t['finish_reason'] == 'length', 'success': None,
                'usage': t.get('usage'), 'teacher': 'nemotron-3.5-super-vl', 'gen_settings': t.get('gen_settings'),
                'messages': msgs}, ensure_ascii=False) + '\n')
        blob = gzip.compress(''.join(lines).encode(), compresslevel=6)
        fsio.write_atomic(p, blob)
        n = sum(1 for _ in gzip.decompress(p.read_bytes()).splitlines())   # gzip -t and line count
        assert n == len(lines), (p, n, len(lines))
        shards.append({'name': p.name, 'lines': n, 'bytes': len(blob), 'sha256': fsio.verify_file(p)[1]})
    rows = []
    for t in kept:
        g = gen.get(t['problem_id'], {})
        rows.append(json.dumps({'problem_id': t['problem_id'], 'domain': t['domain'], 'subtopic': t['subtopic'],
                                'leaf': t['leaf'], 'question': t['messages'][-1]['content'],
                                'source': g.get('source'), 'doc_id': g.get('doc_id')}, ensure_ascii=False) + '\n')
    blob = zstandard.ZstdCompressor(level=10).compress(''.join(rows).encode())
    fsio.write_atomic(out / 'problems.jsonl.zst', blob)
    assert len(zstandard.ZstdDecompressor().decompress(blob).splitlines()) == len(kept)
    psha = fsio.verify_file(out / 'problems.jsonl.zst')[1]
    sums = ''.join(f"{x['sha256']}  {x['name']}\n" for x in shards) + f'{psha}  problems.jsonl.zst\n'
    fsio.write_atomic(out / 'SHA256SUMS', sums)
    stats = {'kept': len(kept), 'kept_total_tokens_api': tok, 'seed': a.seed, 'token_target': a.token_target,
             'shards': shards, 'problems_sha256': psha, **counts}
    fsio.write_atomic(out / 'export_stats.json', json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()

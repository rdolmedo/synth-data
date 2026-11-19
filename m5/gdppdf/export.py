"""Export Super traces as SFT data in the format sft/prepare_data.py consumes (jsonl.gz shards; M1's production
trace format): {"trial", "task", "run", "messages": [{"role": "user", "content"}, {"role": "assistant", "content",
"reasoning_content"}], ...metadata}.

  python -m m5.gdppdf.export --traces '<prod>/traces/traces-*.jsonl' --problems '<prod>/problems-*.jsonl' \
      --index <index.jsonl> --doc-flags <doc_flags.jsonl> --excluded <decontam work>/excluded.json \
      --target-tokens 330e6 --out /data/aai/m5/gdppdf/export

The user message is exactly what the eval sent (aa_evals.gdppdf.inputs.render_text with the problem's pdf_name; the
eval sends it as one text part, which the chat template renders identically). Tokens per sequence = the server's
prompt_tokens + completion_tokens (Super's tokenizer on the whole templated sequence; a sample is re-checked
locally with the tokenizer and chat template). Kept: status ok (a non-empty final answer) and finish_reason stop,
not excluded by decontamination (problem judge, document MinHash/hash, canary). Every other trace is listed with its
reason in excluded.jsonl. If the kept tokens exceed --target-tokens, documents are dropped whole (a deterministic
shuffle) until the total fits: one trace per problem, all tasks of a document together.
"""
import argparse
import collections
import glob
import gzip
import json
import random
from pathlib import Path

from aa_evals.gdppdf import inputs
from m5.gdppdf.docs.store import DocStore

CANARY = '329d96c2-986f-486e-81db-4b6bd8539ead'
SHARD = 2000


def write_verified(path, lines, tries=8):
    """Write a gzip shard to a temp name, read it back (gzip parse, line count, every line JSON), record sha256, then
    rename. Retries with backoff on EDQUOT/EIO or a failed check (the /data byte quota is enforced, 2026-11-12)."""
    import hashlib
    import os
    import time
    data = gzip.compress(''.join(lines).encode(), compresslevel=6)
    tmp = Path(str(path) + '.tmp')
    for k in range(tries):
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            back = tmp.read_bytes()
            text = gzip.decompress(back).decode()
            rows = text.split('\n')[:-1]
            if back != data or len(rows) != len(lines) or any(not json.loads(r).get('messages') for r in rows):
                raise OSError('read-back check failed')
            os.replace(tmp, path)
            return {'file': path.name, 'lines': len(lines), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        except (OSError, ValueError) as e:
            print(f'[export] {path.name}: {e}; retry {k + 1}', flush=True)
            time.sleep(15 * (k + 1))
    raise OSError(f'{path}: could not write a verified shard')


def load(pattern, key):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                r = json.loads(l)
                out[r[key]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--problems', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--doc-flags', default=None)
    ap.add_argument('--excluded', default=None)
    ap.add_argument('--target-tokens', type=float, default=330e6)
    ap.add_argument('--run', default='m5-gdppdf-v1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=357)
    a = ap.parse_args()
    traces = load(a.traces, 'problem_id')
    probs = load(a.problems, 'problem_id')
    store = DocStore(a.index)
    flagged_docs = set()
    if a.doc_flags:
        flagged_docs = {r['doc_id'] for r in map(json.loads, open(a.doc_flags)) if r.get('flag')}
    excl = json.load(open(a.excluded))['excluded'] if a.excluded else {}
    keep, drop = [], []
    for pid, t in traces.items():
        p = probs.get(pid)
        u = t.get('usage') or {}
        tok = (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0)
        why = None
        if p is None:
            why = 'no_problem'
        elif t['status'] != 'ok':
            why = t['status']
        elif t.get('finish_reason') != 'stop':
            why = f'finish_{t.get("finish_reason")}'
        elif pid in excl:
            why = 'decontam_problem'
        elif t['doc_id'] in flagged_docs or store.meta.get(t['doc_id'], {}).get('canary'):
            why = 'decontam_document'
        elif CANARY in (p['prompt'] + t.get('answer', '') + t.get('reasoning', '')):
            why = 'canary'
        (drop if why else keep).append((pid, t, p, tok, why))
    total = sum(x[3] for x in keep)
    cut = 0
    if total > a.target_tokens:
        docs = sorted({x[1]['doc_id'] for x in keep})
        random.Random(a.seed).shuffle(docs)
        by_doc = collections.defaultdict(list)
        for x in keep:
            by_doc[x[1]['doc_id']].append(x)
        kept, run = [], 0
        for d in docs:
            s = sum(x[3] for x in by_doc[d])
            if run + s > a.target_tokens:
                drop += [(x[0], x[1], x[2], x[3], 'over_target') for x in by_doc[d]]
                cut += len(by_doc[d])
                continue
            kept += by_doc[d]
            run += s
        keep = kept
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep.sort(key=lambda x: x[0])
    shards, manifest = 0, []
    for i in range(0, len(keep), SHARD):
        name = f'trajectories-{i // SHARD:04d}.jsonl.gz'
        lines = []
        for pid, t, p, tok, _ in keep[i:i + SHARD]:
            d = store.get(t['doc_id'])
            text = inputs.render_text(p['prompt'], [x['text'] for x in d['pages']], p['pdf_name'], None)
            rec = {'trial': pid, 'task': pid, 'run': a.run, 'success': None,
                   'messages': [{'role': 'user', 'content': text},
                                {'role': 'assistant', 'content': t['answer'], 'reasoning_content': t['reasoning']}],
                   'doc_id': t['doc_id'], 'domain': p['domain'], 'doc_type': p.get('doc_type'),
                   'task_type': p['spec']['task_type'], 'leaf': p['spec']['leaf'], 'n_pages': p['n_pages'],
                   'tokens': tok, 'prompt_tokens': t['usage'].get('prompt_tokens'),
                   'completion_tokens': t['usage'].get('completion_tokens'), 'teacher': 'nemotron-3.5-super-vl',
                   'sampling': t.get('body'), 'max_tokens': t.get('max_tokens')}
            lines.append(json.dumps(rec, ensure_ascii=False) + '\n')
        manifest.append(write_verified(out / name, lines))
        shards += 1
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    with open(out / 'excluded.jsonl', 'w') as f:
        for pid, t, p, tok, why in drop:
            f.write(json.dumps({'problem_id': pid, 'doc_id': t['doc_id'], 'reason': why, 'tokens': tok}) + '\n')
    summary = {'kept_traces': len(keep), 'kept_tokens': sum(x[3] for x in keep), 'kept_docs': len({x[1]['doc_id'] for x in keep}),
               'shards': shards, 'excluded': dict(collections.Counter(x[4] for x in drop)),
               'traces_total': len(traces), 'tokens_mean': round(sum(x[3] for x in keep) / max(1, len(keep))),
               'target_tokens': a.target_tokens, 'cut_over_target': cut}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

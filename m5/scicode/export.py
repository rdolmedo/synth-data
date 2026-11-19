"""Export Super's SciCode traces as SFT records (the sft/prepare_data.py input format, as M1's production traces):
one record per step (a single-turn conversation: the eval's step prompt -> Super's reasoning + answer), gzipped
jsonl shards.

    python -m m5.scicode.export --gen G1[,G2] --traces T1[,T2] --out DIR [--exclude decon_list.json] [--token-cap N]

Exclusions (each counted in DIR/export_stats.json):
- problems the decontamination flags (--exclude);
- chains that did not finish (the tail cut at the token target);
- steps that hit the context limit (finish_reason length / prompt over context: no complete answer) and steps that
  never submit an answer (empty content); the rest of their chain is kept, since the eval
  conditions later steps on whatever such a step produced, exactly as the traces did. A reply without a ``` block is
  kept: the eval's extractor grades the whole reply then.
--token-cap: keep whole chains until the Super-tokenizer count reaches N (cuts the tail): in completion order, or with
--seed in a seeded random order over problems (a random-by-task subset, as the other M5 benches' 330M exports).
--dup-flags (repeat.py dup_flags.json) only labels records (near_dup clear|borderline, near_dup_identical, the
keep-one redundancy bits); nothing is dropped for it.
Tokens are counted with the Super tokenizer on the full rendered sequence (Super's chat template, prompt plus
reasoning plus answer).
"""
import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

from .common import read_jsonl, safe_write

SUPER = '/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'
SHARD = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--dup-flags', default=None, help='repeat.py dup_flags.json: problem_id -> clear|borderline (flag only)')
    ap.add_argument('--token-cap', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None, help='random-by-task order for --token-cap')
    ap.add_argument('--run', default='m5-scicode')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SUPER)
    specs = {}
    for g in a.gen.split(','):
        for s in read_jsonl(Path(g) / 'specs.jsonl'):
            specs[s['problem_id']] = s
    excl = set(json.loads(Path(a.exclude).read_text())['exclude']) if a.exclude else set()
    dflags = json.loads(Path(a.dup_flags).read_text())['flags'] if a.dup_flags else {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = collections.Counter()
    shard, buf, total = 0, [], 0
    manifest = []
    sums = {}

    def flush():
        nonlocal shard, buf
        if buf:
            p = out / f'traces-{shard:04d}.jsonl.gz'
            raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in buf).encode()
            z = gzip.compress(raw)
            assert gzip.decompress(z) == raw
            safe_write(p, z)
            sums[p.name] = {'sha256': hashlib.sha256(z).hexdigest(), 'records': len(buf), 'bytes': len(z)}
            shard += 1
            buf = []

    steps, chains = collections.defaultdict(dict), []
    for t in a.traces.split(','):
        for r in read_jsonl(Path(t) / 'steps.jsonl'):
            steps[r['problem_id']][r['k']] = r
        chains += read_jsonl(Path(t) / 'chains.jsonl')
    if a.seed is not None:
        random.Random(a.seed).shuffle(chains)
    for c in chains:
        pid = c['problem_id']
        st['chains'] += 1
        if pid in excl:
            st['chains_excluded_decon'] += 1
            continue
        if a.token_cap and total >= a.token_cap:
            st['chains_cut_tail'] += 1
            continue
        recs, ctoks = [], 0
        for k in range(c['n_steps']):
            r = steps[pid][k]
            st['steps'] += 1
            if r['finish_reason'] in ('length', 'prompt_over_context'):
                st['steps_excluded_context_limit'] += 1
                continue
            if not (r['content'] or '').strip():
                st['steps_excluded_no_submit'] += 1
                continue
            st['steps_no_code_fence_kept'] += '```' not in r['content']   # the eval grades the whole reply then
            msgs = [{'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': r['content'], 'reasoning_content': r['reasoning'] or ''}]
            text = tok.apply_chat_template(msgs, tokenize=False, truncate_history_thinking=False)
            n = len(tok(text, add_special_tokens=False)['input_ids'])
            ctoks += n
            recs.append({'messages': msgs, 'task': pid, 'trial': r['step_number'], 'run': a.run,
                         'success': None, 'n_tokens_super': n, 'finish_reason': r['finish_reason'],
                         'cell': specs.get(pid, {}).get('cell'), 'step': k + 1, 'n_steps': c['n_steps'],
                         'near_dup': (dflags.get(pid) or {}).get('level'),
                         'near_dup_of': (dflags.get(pid) or {}).get('partners', []),
                         'near_dup_identical': bool((dflags.get(pid) or {}).get('identical')),
                         'near_dup_redundant_identical': bool((dflags.get(pid) or {}).get('redundant_identical')),
                         'near_dup_redundant_clear': bool((dflags.get(pid) or {}).get('redundant_clear')),
                         'near_dup_redundant_any': bool((dflags.get(pid) or {}).get('redundant_any'))})
        if not recs:
            st['chains_no_steps_kept'] += 1
            continue
        total += ctoks
        st['chains_kept'] += 1
        st['steps_kept'] += len(recs)
        manifest.append({'problem_id': pid, 'tokens': ctoks, 'steps_kept': len(recs),
                         'near_dup': (dflags.get(pid) or {}).get('level')})
        f = dflags.get(pid) or {}
        tags = ([f['level']] if f.get('level') else []) + [x for x in ('identical',) if f.get(x)] + \
            [f'redundant_{x}' for x in ('identical', 'clear', 'any') if f.get(f'redundant_{x}')]
        for tg in tags:
            st[f'chains_near_dup_{tg}'] += 1
            st[f'tokens_near_dup_{tg}'] += ctoks
        buf += recs
        if len(buf) >= SHARD:
            flush()
    flush()
    stats = {**st, 'tokens_super': total, 'shards': shard, 'excluded_list': a.exclude, 'seed': a.seed,
             'token_cap': a.token_cap, 'dup_flags': a.dup_flags}
    safe_write(out / 'manifest.jsonl.gz', gzip.compress(''.join(json.dumps(m) + '\n' for m in manifest).encode()))
    safe_write(out / 'SHA256SUMS.json', json.dumps(sums, indent=1))
    safe_write(out / 'export_stats.json', json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()

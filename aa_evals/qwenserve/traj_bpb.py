#!/usr/bin/env python3
"""Served bits-per-byte of a model's own recorded agent trajectories under one engine config (M3.3, #176).

The numerics check of the serving studies (serve_q35 / serve_lightning bpb.sh: offline vLLM prompt log-probs, one
GPU), on the model's own Pi traffic instead of an SFT val split:

  build: each pi_eval attempt (RUN/attempts/*/trial.tar.gz) -> its last recorded request (wire-requests.jsonl deltas
         replayed) rendered with the checkpoint's chat template (add_generation_prompt false, the serving template
         kwargs, e.g. preserve_thinking, so every assistant turn keeps its reasoning as served). Label tokens = the
         assistant turns (everything after "<|im_start|>assistant\\n" through "<|im_end|>"): the model's own outputs.
  score: NLL of the label tokens from vLLM prompt log-probs with --engine-args (e.g. {"kv_cache_dtype": "fp8_e4m3"},
         {"quantization": "fp8"}); BPB = NLL / (ln 2 x label UTF-8 bytes). Compare configs on the same docs file.

  python traj_bpb.py build --runs RUN [RUN ...] --model CKPT --out docs.jsonl [--max-tokens 131071] [--kwargs JSON]
  python traj_bpb.py score --docs docs.jsonl --model CKPT --out arm.json --name ARM [--engine-args JSON]
"""
import argparse
import glob
import json
import math
import tarfile
import time
from pathlib import Path


def last_request(tar_path):
    """(messages, tools) of the attempt's last forwarded request, or None."""
    with tarfile.open(tar_path) as tf:
        m = next((x for x in tf.getmembers() if x.name.endswith('pi-cont16k/wire-requests.jsonl')), None)
        if m is None:
            return None
        lines = tf.extractfile(m).read().decode().splitlines()
    messages, tools = [], None
    for line in lines:
        d = json.loads(line)
        delta = d['messages_delta']
        messages = messages[:delta['copy_previous_prefix_messages']] + delta['append_messages']
        if d.get('tools'):
            tools = d['tools']
    return (messages, tools) if messages else None


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kwargs = json.loads(a.kwargs)
    tars = sorted(t for r in a.runs for t in glob.glob(str(Path(r) / 'attempts' / '*' / 'trial.tar.gz')))
    n = 0
    with open(a.out, 'w') as f:
        for t in tars:
            req = last_request(t)
            if not req:
                continue
            msgs, tools = req
            for m in msgs:   # Pi's history field -> the template's
                if m.get('role') == 'assistant' and m.get('reasoning') and not m.get('reasoning_content'):
                    m['reasoning_content'] = m.pop('reasoning')
                for tc in m.get('tool_calls') or []:   # vLLM parses JSON-string arguments before templating
                    fn = tc.get('function') or {}
                    if isinstance(fn.get('arguments'), str):
                        try:
                            fn['arguments'] = json.loads(fn['arguments'])
                        except ValueError:
                            pass
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False, **kwargs)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc['input_ids'], enc['offset_mapping']
            spans, i, head = [], 0, '<|im_start|>assistant\n'
            while True:
                s = text.find(head, i)
                if s < 0:
                    break
                e = text.find('<|im_end|>', s)
                e = len(text) if e < 0 else e + len('<|im_end|>')
                spans.append((s + len(head), e))
                i = e
            mask, j = [], 0
            for (cs, ce) in offs:
                while j < len(spans) and spans[j][1] <= cs:
                    j += 1
                mask.append(int(j < len(spans) and cs >= spans[j][0] and ce <= spans[j][1] and ce > cs))
            ids, mask = ids[:a.max_tokens], mask[:a.max_tokens]
            if sum(mask) < 16:
                continue
            f.write(json.dumps({'id': str(Path(t).parent.name), 'ids': ids, 'mask': mask}) + '\n')
            n += 1
    print(f'{n} docs from {len(tars)} attempts -> {a.out}')


def score(a):
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    docs = [json.loads(line) for line in open(a.docs)]
    docs = docs[:a.max_docs] if a.max_docs else docs
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, max_num_batched_tokens=a.max_batched,
              gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False, max_num_seqs=4, max_logprobs=1,
              **json.loads(a.engine_args))
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False, flat_logprobs=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=d['ids']) for d in docs], sp)
    per = []
    for d, o in zip(docs, outs):
        x, m = np.asarray(d['ids']), np.asarray(d['mask'], dtype=bool)
        flat = o.prompt_logprobs
        st = np.asarray(flat.start_indices)
        ids_all, lp_all = np.asarray(flat.token_ids), np.asarray(flat.logprobs, dtype=np.float64)
        first = st[1:]
        assert np.array_equal(ids_all[first], x[1:]), 'log-probs are not of the prompt tokens'
        lp = lp_all[first]                     # entry p - 1 = position p
        want = np.flatnonzero(m[1:]) + 1
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
        nbytes = sum(len(tok.decode(x[lo:hi].tolist(), skip_special_tokens=False).encode())
                     for lo, hi in zip(edges[::2], edges[1::2]))
        per.append({'id': d['id'], 'tokens': int(len(x)), 'label_tokens': int(m[1:].sum()), 'label_bytes': int(nbytes),
                    'nll_nats': float(-lp[want - 1].sum())})
    N, T, B = (sum(r[k] for r in per) for k in ('nll_nats', 'label_tokens', 'label_bytes'))
    out = {'name': a.name, 'model': a.model, 'engine_args': json.loads(a.engine_args), 'docs': len(per),
           'tokens': sum(r['tokens'] for r in per), 'label_tokens': T, 'label_bytes': B,
           'nll_per_token': N / T, 'bpb': N / (math.log(2) * B), 'elapsed_s': round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({**out, 'per_doc': per}, open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    b = sub.add_parser('build')
    b.add_argument('--runs', nargs='+', required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--max-tokens', type=int, default=131071)
    b.add_argument('--kwargs', default='{"enable_thinking": true, "preserve_thinking": true}')
    s = sub.add_parser('score')
    s.add_argument('--docs', required=True)
    s.add_argument('--model', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--name', required=True)
    s.add_argument('--engine-args', default='{}')
    s.add_argument('--max-model-len', type=int, default=131072)
    s.add_argument('--max-batched', type=int, default=8192)
    s.add_argument('--gpu-util', type=float, default=0.85)
    s.add_argument('--max-docs', type=int, default=0)
    a = ap.parse_args()
    (build if a.op == 'build' else score)(a)


if __name__ == '__main__':
    main()

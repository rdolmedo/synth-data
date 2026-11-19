#!/usr/bin/env python3
"""Is the server's sampling faithful to the target distribution? A randomized PIT check on real DeepSeek contexts.

For speculative decoding (DSpark block verification, vllm#61114 / issue #9) the sampled token at every position should
follow the target model's distribution at that position given the sampled prefix. The server returns the target's
top-k logprobs at every generated position (temperature 1, top_p 1, so the raw logprobs are the sampling
distribution). For a sampled token s with probability p_s and probability mass F_s above it (tokens ranked before
it), u = F_s + U * p_s (U uniform) is exactly Uniform(0, 1) under faithful sampling, at every position, whatever the
prompt. A token outside the top-k is one bucket after the top-k (a coarser partition keeps the PIT exact).

Statistics per position band (j = 0 is sampled right after prefill and never verified; j >= 1 go through draft
verification): the Kolmogorov-Smirnov distance of u from uniform with its asymptotic p-value, decile counts, and the
top-group rate: how often s is (one of) the target's argmax token(s) vs the expected mass (z-score).
See pit() for two properties of vLLM's reported logprobs that the statistics account for.

    sampling_calibration.py --base http://127.0.0.1:58869 --model deepseek-v41-flash-1017 --corpus corpus.json \
        --out X.json [--prompts 64 --samples 32 --max-tokens 64]
    sampling_calibration.py --report X.json [Y.json ...]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, body, rank, timeout=1800):
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'X-data-parallel-rank': str(rank)})
    return json.loads(OPENER.open(request, timeout=timeout).read())


def load_prompts(corpus, count, min_chars, max_chars, seed=7):
    """Deterministic: a random (seeded) turn of each of the first trajectories, sized within [min_chars, max_chars]."""
    trajectories = sorted(json.loads(Path(corpus).read_text())['trajectories'], key=lambda t: t['run'])
    rng, out = random.Random(seed), []
    for t in trajectories:
        if len(out) >= count:
            break
        k = rng.randrange(len(t['requests']))
        try:
            body = json.loads(Path(t['requests'][k]).read_text())
        except (OSError, ValueError):
            continue
        if not min_chars <= len(json.dumps(body['messages'])) <= max_chars:
            continue
        messages = [dict(m) for m in body['messages']]
        system = next((j for j, m in enumerate(messages) if m.get('role') == 'system'), None)
        nonce = f'[calibration {len(out):03d}] '
        if system is None:
            messages.insert(0, {'role': 'system', 'content': nonce})
        else:
            messages[system]['content'] = nonce + (messages[system].get('content') or '')
        out.append({'id': len(out), 'run': t['run'], 'turn': k, 'body': {**body, 'messages': messages}})
    return out


def sample(base, body, k, max_tokens, seed, rank):
    body = {**body, 'stream': False, 'temperature': 1.0, 'top_p': 1.0, 'max_tokens': max_tokens, 'seed': seed,
            'logprobs': True, 'top_logprobs': k, 'return_tokens_as_token_ids': True}
    for key in ('stream_options', 'max_completion_tokens', 'top_k', 'min_p'):
        body.pop(key, None)
    out = post(base, body, rank)
    rows = []
    for c in out['choices'][0]['logprobs']['content']:
        # First k by logprob: some builds also list the sampled token when it is outside the top k; keeping it would
        # make the partition depend on the sample and break the PIT.
        top = sorted(((t['token'], t['logprob']) for t in c['top_logprobs']), key=lambda x: -x[1])[:k]
        rows.append([c['token'], c['logprob'], top])
    return rows


def pit(token, top, rng):
    """(u, p_top_group, in_top_group) for one sampled position.

    Two properties of vLLM's reported logprobs would bias a naive PIT (found on DeepSeek V4.1, 2026-11-18):
    - `top_logprobs` always contains the sampled token: when it is outside the top k, it takes the k-th slot and the
      true k-th token is dropped. Only the first k-1 entries are trusted; a sampled token outside them is the tail.
    - Logits can be coarse (DeepSeek V4.1: logprob steps of 0.125 nats, exact ties for the top token at ~2% of
      positions), and the API lists the sampled token first among equal values. Tokens with equal logprobs are one
      bucket, so their order cannot leak the sample.
    """
    groups = {}
    for t, lp in top[:-1]:
        groups.setdefault(lp, []).append(t)
    above, first = 0.0, None
    for lp in sorted(groups, reverse=True):
        mass = math.exp(lp) * len(groups[lp])
        if first is None:
            first = (mass, set(groups[lp]))
        if token in groups[lp]:
            return above + rng.random() * mass, first[0], token in first[1]
        above += mass
    rest = max(0.0, 1.0 - above)
    return min(1.0, above + rng.random() * rest), first[0], False


def ks(us):
    us = sorted(us)
    n = len(us)
    if not n:
        return None, None
    d = max(max((i + 1) / n - u, u - i / n) for i, u in enumerate(us))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, max(0.0, min(1.0, p))


def analyze(samples, seed=0):
    rng = random.Random(seed)
    bands = {'j=0': lambda j: j == 0, 'j=1-5': lambda j: 1 <= j <= 5, 'j>=6': lambda j: j >= 6, 'j>=1': lambda j: j >= 1}
    acc = {b: {'u': [], 'top1_obs': 0, 'top1_exp': 0.0, 'top1_var': 0.0, 'informative': 0} for b in bands}
    for s in samples:
        for j, (token, _, top) in enumerate(s['tokens']):
            u, p1, is1 = pit(token, top, rng)
            for b, f in bands.items():
                if f(j):
                    a = acc[b]
                    a['u'].append(u)
                    a['top1_obs'] += is1
                    a['top1_exp'] += p1
                    a['top1_var'] += p1 * (1 - p1)
                    a['informative'] += p1 < 0.9
    report = {}
    for b, a in acc.items():
        d, p = ks(a['u'])
        deciles = [0] * 10
        for u in a['u']:
            deciles[min(9, int(u * 10))] += 1
        report[b] = {'tokens': len(a['u']), 'informative_tokens(p_top1<0.9)': a['informative'],
                     'ks_d': round(d, 5) if d is not None else None, 'ks_p': round(p, 5) if p is not None else None,
                     'deciles': deciles, 'top1_obs': a['top1_obs'], 'top1_exp': round(a['top1_exp'], 1),
                     'top1_z': round((a['top1_obs'] - a['top1_exp']) / math.sqrt(a['top1_var']), 2) if a['top1_var'] else None}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base')
    ap.add_argument('--model')
    ap.add_argument('--corpus')
    ap.add_argument('--out')
    ap.add_argument('--prompts', type=int, default=64)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--min-chars', type=int, default=4_000)
    ap.add_argument('--max-chars', type=int, default=150_000)
    ap.add_argument('--dp', type=int, default=4)
    ap.add_argument('--workers', type=int, default=256)
    ap.add_argument('--report', nargs='+')
    a = ap.parse_args()
    if a.report:
        for path in a.report:
            r = json.loads(Path(path).read_text())
            print(json.dumps({'file': path, 'meta': r.get('meta'), 'report': analyze(r['samples'])}, indent=1))
        return
    prompts = load_prompts(a.corpus, a.prompts, a.min_chars, a.max_chars)
    meta = json.loads(OPENER.open(a.base + '/version', timeout=60).read())
    seeds = random.Random(12345)
    jobs = [(p, s, seeds.randrange(2**31 - 1)) for s in range(a.samples) for p in prompts]
    t0, errors = time.time(), []

    def one(job):
        p, s, seed = job
        try:
            return {'prompt': p['id'], 'sample': s, 'seed': seed,
                    'tokens': sample(a.base, {**p['body'], 'model': a.model}, a.k, a.max_tokens, seed, p['id'] % a.dp)}
        except Exception as exc:
            errors.append(repr(exc)[:300])
            return None

    # The first sample of each prompt alone, so the others hit its cached prefix.
    with ThreadPoolExecutor(a.workers) as pool:
        first = list(pool.map(one, jobs[:len(prompts)]))
        rest = list(pool.map(one, jobs[len(prompts):]))
    samples = [x for x in first + rest if x]
    report = analyze(samples)
    meta.update(prompts=len(prompts), samples=a.samples, max_tokens=a.max_tokens, k=a.k, seconds=round(time.time() - t0),
                errors=len(errors), error_sample=errors[:3])
    Path(a.out).write_text(json.dumps({'meta': meta, 'report': report, 'samples': samples}))
    print(json.dumps({'meta': meta, 'report': report}, indent=1), flush=True)


if __name__ == '__main__':
    main()

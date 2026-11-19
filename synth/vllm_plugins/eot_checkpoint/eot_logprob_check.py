"""GPU correctness check for synth-eot-checkpoint: next-token distributions after an end-state hit vs a fresh prefill.

Each conversation takes a recorded Pi request (a unique nonce in the system prompt, so no cache holds its prefix),
generates turn 1 on the engine, builds turn 2 as Pi would (the answer re-rendered + a short tool result), and scores
turn 2's first token (top-k logprobs, temperature 0) two ways on the same engine:

  hit    turn 2 as served: resumed from turn 1's end state (variant eot-gpu), from it after the GPU prefix cache was
         reset so it comes back from the CPU tier (eot-cpu), or with the plugin skipped for the request so it resumes
         at turn 1's prompt boundary as stock vLLM does (boundary, the noise floor of an ordinary prefix hit);
  fresh  the same prompt under a random cache_salt: computed from scratch.

  eot-gpu-settled  eot-gpu scored after settle() (a few scheduler steps and 4 s), without the reset: separates a
         wrong end state (both GPU variants off) from turn 2 reading it before the stop step is done (only eot-gpu off).

  paired  one conversation, every way, each turn 2 with its own tool-result text so it diverges right after turn 1's
         end (no variant can resume from another's turn-2 prefix): the end-state hit right after turn 1 (imm), the
         same end state after settle() (set), a boundary hit with the plugin skipped (bnd), each against its own
         fresh prefill; max |dlogprob| and total variation over the top-k union. A read before the stop step is done
         shows as imm worse than set; a wrong end state as imm and set worse than bnd.

A correct end state gives hit-vs-fresh differences like the boundary variant's. `cached` must equal turn 1's prompt +
output tokens for the eot variants. The server needs VLLM_SERVER_DEV_MODE=1 (reset_prefix_cache).

  eot_logprob_check.py --base http://host:port --model nemotron-3.5-super-vl --corpus <corpus.json> --out <json>
"""
import argparse
import json
import math
import random
import statistics
import time
import urllib.request
import uuid

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read() or b'null')


def generate(base, body):
    """Turn 1 streamed, as Pi sends it, with token ids (a non-streaming response right-strips the reasoning and the
    text, so a history built from it is not what was generated). Returns (message, prompt ids, output ids, finish)."""
    body = {**body, 'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    reasoning, content, calls, prompt_ids, output_ids, finish = [], [], {}, [], [], None
    with OPENER.open(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            prompt_ids = chunk.get('prompt_token_ids') or prompt_ids
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
                content.append(delta.get('content') or '')
                for t in delta.get('tool_calls') or []:
                    c = calls.setdefault(t.get('index', 0), {'id': None, 'name': '', 'arguments': ''})
                    c['id'] = t.get('id') or c['id']
                    c['name'] += (t.get('function') or {}).get('name') or ''
                    c['arguments'] += (t.get('function') or {}).get('arguments') or ''
                output_ids += choice.get('token_ids') or []
                finish = choice.get('finish_reason') or finish
    reasoning, content = ''.join(reasoning), ''.join(content)
    out = {'role': 'assistant', 'content': ('<think>\n' + reasoning + '</think>' + content) if reasoning else content}
    if calls:
        out['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {
            'name': c['name'], 'arguments': json.dumps(json.loads(c['arguments'] or '{}'), separators=(',', ':'),
                                                       ensure_ascii=False)}} for c in (calls[k] for k in sorted(calls))]
    return out, prompt_ids, output_ids, finish


def score(base, body, k, **extra):
    b = {**body, **extra, 'stream': False, 'max_tokens': 1, 'temperature': 0.0, 'logprobs': True, 'top_logprobs': k,
         'return_token_ids': True}
    for key in ('stream_options', 'max_completion_tokens', 'seed'):
        b.pop(key, None)
    r = post(base + '/v1/chat/completions', b)
    lp = r['choices'][0]['logprobs']['content'][0]
    cached = ((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    return {t['token']: t['logprob'] for t in lp['top_logprobs']}, cached, r.get('prompt_token_ids') or []


def tv(a, b):
    """Total variation over the union of two top-k lists (a token missing from one list counts as 0 there)."""
    return 0.5 * sum(abs(math.exp(a.get(t, -math.inf)) - math.exp(b.get(t, -math.inf))) for t in set(a) | set(b))


def diff(a, b):
    common = set(a) & set(b)
    return (max((abs(a[t] - b[t]) for t in common), default=math.inf), max(a, key=a.get) == max(b, key=b.get))


def settle(base, model):
    """Drive a few scheduler steps (finished CPU stores flush on the next step), then let them complete."""
    for _ in range(2):
        post(base + '/v1/chat/completions', {'model': model, 'max_tokens': 8,
                                             'messages': [{'role': 'user', 'content': f'hi {uuid.uuid4()}'}]})
        time.sleep(2)


def reset_gpu_cache(base, tries=15):
    """Reset the GPU prefix cache (the CPU tier stays: reset_external defaults to false). The endpoint answers
    {"success": false} while blocks are still held, e.g. by in-flight CPU-tier stores; a failed reset would let
    turn 2 hit on the GPU and mislabel an eot-cpu row, so retry."""
    for _ in range(tries):
        r = post(base + '/reset_prefix_cache', None)
        if not isinstance(r, dict) or r.get('success', True):
            return r
        time.sleep(2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--corpus', required=True, help='synth.bench corpus cache (json with trajectories)')
    ap.add_argument('--n', type=int, default=8, help='conversations per variant')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', default='boundary,eot-gpu,eot-cpu', help='comma-separated, each run --n times '
                    '(also eot-gpu-settled)')
    a = ap.parse_args()
    trajectories = json.load(open(a.corpus))['trajectories']
    rng = random.Random(a.seed)
    rows = []
    names = a.variants.split(',')
    variants = names * a.n
    for i, variant in enumerate(variants):
        t = trajectories[rng.randrange(len(trajectories))]
        k = rng.randrange(1, max(2, len(t['requests']) // 2))
        body = json.load(open(t['requests'][k]))
        body['model'] = a.model
        nonce = f'Session {uuid.uuid4()}.\n'
        body['messages'] = [dict(m) for m in body['messages']]
        body['messages'][0]['content'] = nonce + body['messages'][0]['content']
        row = {'i': i, 'variant': variant, 'request': t['requests'][k]}
        try:
            message, p1, g1, finish = generate(a.base, body)
            row.update(prompt1=len(p1), output1=len(g1), finish1=finish, last1=g1[-1] if g1 else None)
            if finish != 'stop' and finish != 'tool_calls':
                row['skipped'] = f'turn 1 finished with {finish}'
                rows.append(row)
                continue
            tool = [{'role': 'tool', 'content': 'ok\n', 'tool_call_id': c['id']} for c in message.get('tool_calls', [])]
            turn2 = {**body, 'messages': body['messages'] + [message] + (tool or [{'role': 'user', 'content': 'ok'}])}
            if variant == 'paired':
                def turn2_with(text):
                    tl = [{'role': 'tool', 'content': text, 'tool_call_id': c['id']} for c in message.get('tool_calls', [])]
                    return {**body, 'messages': body['messages'] + [message] + (tl or [{'role': 'user', 'content': text}])}
                t_imm, t_set, t_bnd = turn2_with('ok\n'), turn2_with('done, exit code 0\n'), turn2_with('result: none\n')
                imm, c_imm, ids2 = score(a.base, t_imm, a.k)
                settle(a.base, a.model)
                sett, c_set, _ = score(a.base, t_set, a.k)
                bnd, c_bnd, _ = score(a.base, t_bnd, a.k, vllm_xargs={'synth_eot_skip': 1})
                ref = {k: score(a.base, t, a.k, cache_salt=str(uuid.uuid4()))[0]
                       for k, t in (('imm', t_imm), ('set', t_set), ('bnd', t_bnd))}
                pairs = {'imm': (imm, ref['imm']), 'set': (sett, ref['set']), 'bnd': (bnd, ref['bnd'])}
                row.update(prompt2=len(ids2), expected_end=len(p1) + len(g1), cached={'imm': c_imm, 'set': c_set, 'bnd': c_bnd},
                           **{f'mad_{k}': diff(x, y)[0] for k, (x, y) in pairs.items()},
                           **{f'tv_{k}': round(tv(x, y), 6) for k, (x, y) in pairs.items()},
                           **{f'argmax_{k}': diff(x, y)[1] for k, (x, y) in pairs.items()})
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            extra = {'vllm_xargs': {'synth_eot_skip': 1}} if variant == 'boundary' else {}
            if variant in ('eot-cpu', 'eot-gpu-settled'):
                settle(a.base, a.model)
            if variant == 'eot-cpu':
                row['reset'] = reset_gpu_cache(a.base)
            hit, cached, ids2 = score(a.base, turn2, a.k, **extra)
            fresh, cached_fresh, _ = score(a.base, turn2, a.k, cache_salt=str(uuid.uuid4()))
            d, same = diff(hit, fresh)
            before = p1 + g1
            lcp = next((i for i, (x, y) in enumerate(zip(ids2, before)) if x != y), min(len(ids2), len(before)))
            row.update(prompt2=len(ids2), lcp=lcp, cached=cached, cached_fresh=cached_fresh, expected_end=len(before),
                       max_abs_diff=d, argmax_same=same, top_hit=max(hit, key=hit.get), top_fresh=max(fresh, key=fresh.get))
        except Exception as exc:  # one failed conversation must not end the check
            row['error'] = repr(exc)[:500]
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    paired = [r for r in rows if r['variant'] == 'paired' and 'tv_imm' in r]
    if paired:
        q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
        summary['paired'] = {'n': len(paired),
                             'end_state_hits': {k: sum((r['cached'][k] or 0) >= r['expected_end'] for r in paired) for k in ('imm', 'set')},
                             'boundary_hits_before_end': sum((r['cached']['bnd'] or 0) < r['prompt1'] for r in paired),
                             **{f'{m}_{k}': {'median': round(statistics.median(r[f'{m}_{k}'] for r in paired), 5),
                                             'p90': round(q([r[f'{m}_{k}'] for r in paired], .9), 5),
                                             'max': round(max(r[f'{m}_{k}'] for r in paired), 5)}
                                for m in ('tv', 'mad') for k in ('imm', 'set', 'bnd')},
                             **{f'argmax_{k}': sum(r[f'argmax_{k}'] for r in paired) for k in ('imm', 'set', 'bnd')}}
    for variant in names:
        if variant == 'paired':
            continue
        rs = [r for r in rows if r['variant'] == variant and 'max_abs_diff' in r]
        if not rs:
            continue
        ds = sorted(r['max_abs_diff'] for r in rs)
        summary[variant] = {'n': len(rs), 'max_abs_diff_median': statistics.median(ds), 'max_abs_diff_max': ds[-1],
                            'argmax_same': sum(r['argmax_same'] for r in rs),
                            'hit_at_end': sum(r['cached'] == r['expected_end'] for r in rs),
                            'answer_rerendered': sum(r['lcp'] >= r['expected_end'] for r in rs),
                            'cached_minus_prompt1_median': statistics.median(r['cached'] - r['prompt1'] for r in rs),
                            'reset_failed': sum(isinstance(r.get('reset'), dict) and not r['reset'].get('success', True)
                                                for r in rs)}
    print(json.dumps(summary, indent=1))
    json.dump({'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

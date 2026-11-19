#!/usr/bin/env python3
"""Greedy and first-token logprob parity between two vLLM builds on real recorded Pi requests (DeepSeek V4.1, Super).

Adapted from inference/glm53_state_check.py (data-synthesis). Prompts are fixed and deterministic: for the first
trajectories of a synth.bench corpus (sorted by run path), one long request (the latest one under --max-chars) and
one short request (the first turn), each made unique by a nonce at the start of the system prompt, so no replay
shares their cache and two engines see byte-identical prompts.

Each prompt is sent twice to the same DP rank (X-data-parallel-rank, so the second really hits the first's cache):
`fresh` (whole prompt computed) and `hit` (served from the GPU prefix cache). Each request is greedy (temperature 0)
for --max-tokens tokens with the top-k logprobs of every position, token ids as strings ("token_id:N").

    parity_check.py --base http://127.0.0.1:58869 --model deepseek-v41-flash-1017 --corpus corpus.json --out A.json
    parity_check.py --compare A.json B.json            # A vs B fresh, judged against each build's hit-vs-fresh
    parity_check.py ... --images 8                     # also 8 prompts with a synthetic PNG (kind `image`, VL specs)
    parity_check.py ... --cpu-tier                     # also `cpu`: after POST /reset_prefix_cache (GPU cache only;
                                                       # server needs VLLM_SERVER_DEV_MODE=1) the prefix must come back
                                                       # from the CPU KV tier; sequential (resets would hit other prompts)

Per prompt and pair: first-token max |dlogprob| over the common top-k, total variation over the top-k union (missing
entries count as 0), argmax agreement, and the common prefix of the greedy token sequences.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
import zlib

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, body, rank, timeout=1800):
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'X-data-parallel-rank': str(rank)})
    return json.loads(OPENER.open(request, timeout=timeout).read())


def greedy(base, body, k, max_tokens, rank):
    body = {**body, 'stream': False, 'temperature': 0.0, 'max_tokens': max_tokens, 'logprobs': True,
            'top_logprobs': k, 'return_tokens_as_token_ids': True}
    for key in ('stream_options', 'max_completion_tokens', 'seed', 'top_p'):
        body.pop(key, None)
    t0 = time.time()
    out = post(base, body, rank)
    content = out['choices'][0]['logprobs']['content']
    usage = out.get('usage') or {}
    return {'tokens': [c['token'] for c in content],
            'top': [{t['token']: t['logprob'] for t in c['top_logprobs']} for c in content[:4]],
            'prompt_tokens': usage.get('prompt_tokens'),
            'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'finish': out['choices'][0].get('finish_reason'), 'seconds': round(time.time() - t0, 2)}


def external_tokens(base):
    """Prompt tokens served by a KV connector so far (vllm:prompt_tokens_by_source{source="external_kv_transfer"})."""
    text = OPENER.open(base + '/metrics', timeout=60).read().decode()
    return sum(float(l.rsplit(' ', 1)[1]) for l in text.splitlines()
               if l.startswith('vllm:prompt_tokens_by_source') and 'external_kv_transfer' in l)


def load_prompts(corpus, count, min_chars, max_chars):
    trajectories = sorted(json.loads(Path(corpus).read_text())['trajectories'], key=lambda t: t['run'])
    out = []
    for t in trajectories:
        if len(out) >= 2 * count:
            break
        bodies = []
        for path in t['requests']:
            try:
                bodies.append(json.loads(Path(path).read_text()))
            except (OSError, ValueError):
                break
        sized = [(len(json.dumps(b['messages'])), b) for b in bodies]
        long = [b for c, b in sized if min_chars <= c <= max_chars]
        if not long or not bodies:
            continue
        for kind, body in (('long', long[-1]), ('short', bodies[0])):
            i = len(out)
            messages = [dict(m) for m in body['messages']]
            system = next((j for j, m in enumerate(messages) if m.get('role') == 'system'), None)
            nonce = f'[parity-check {i:03d}] '
            if system is None:
                messages.insert(0, {'role': 'system', 'content': nonce})
            else:
                messages[system]['content'] = nonce + (messages[system].get('content') or '')
            out.append({'id': f'{i:03d}-{kind}', 'run': t['run'], 'kind': kind, 'body': {**body, 'messages': messages}})
    return out


def first_diff(a, b):
    common = set(a) & set(b)
    dmax = max((abs(a[t] - b[t]) for t in common), default=math.inf)
    tv = 0.5 * sum(abs(math.exp(a.get(t, -math.inf)) - math.exp(b.get(t, -math.inf))) for t in set(a) | set(b))
    return {'max_dlogprob': dmax, 'tv': tv, 'argmax_agree': max(a, key=a.get) == max(b, key=b.get)}


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pair(x, y):
    d = first_diff(x['top'][0], y['top'][0])
    d['greedy_lcp'] = lcp(x['tokens'], y['tokens'])
    d['greedy_len'] = min(len(x['tokens']), len(y['tokens']))
    return d


def png(i, size=448):
    """A deterministic RGB PNG: a 4x4 grid of colored cells, a diagonal band and a border, varied by i."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            cx, cy = x * 4 // size, y * 4 // size
            r, g, b = (37 * (cx + i) + 90 * cy) % 256, (71 * cy + 23 * i + 40 * cx) % 256, (113 * (cx ^ cy) + 51 * i) % 256
            if abs(x - y - (i * 29) % 200 + 100) < 18:
                r, g, b = 250, 250, 250
            if min(x, y, size - 1 - x, size - 1 - y) < 6:
                r, g, b = 10, 10, 10
            row += bytes((r, g, b))
        rows.append(bytes(row))
    def chunk(tag, data):
        return len(data).to_bytes(4, 'big') + tag + data + zlib.crc32(tag + data).to_bytes(4, 'big')
    ihdr = size.to_bytes(4, 'big') * 2 + bytes([8, 2, 0, 0, 0])
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b'')


QUESTIONS = ('Describe this image: the colors of the grid cells, the band and the border.',
             'How many grid cells does this image have, and which one is the brightest?',
             'What shapes and colors do you see? Answer in two sentences.',
             'Is the diagonal band lighter or darker than the cells it crosses? Explain briefly.')


def image_prompts(count):
    """`count` image prompts (kind `image`): a nonce system message, one user turn with a PNG and a question."""
    out = []
    for i in range(count):
        url = 'data:image/png;base64,' + base64.b64encode(png(i)).decode()
        msgs = [{'role': 'system', 'content': f'[parity-check image {i:03d}] You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                             {'type': 'text', 'text': QUESTIONS[i % len(QUESTIONS)]}]}]
        out.append({'id': f'image-{i:03d}', 'run': 'synthetic', 'kind': 'image', 'body': {'messages': msgs}})
    return out


def kinds(rows):
    return sorted({x['kind'] for x in rows}, key=('long', 'short', 'image').index)


def summarize(ds):
    if not ds:
        return {}
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    return {'n': len(ds), 'max_dlogprob_median': round(statistics.median(d['max_dlogprob'] for d in ds), 4),
            'max_dlogprob_p90': round(q([d['max_dlogprob'] for d in ds], .9), 4),
            'tv_median': round(statistics.median(d['tv'] for d in ds), 4), 'tv_p90': round(q([d['tv'] for d in ds], .9), 4),
            'tv_max': round(max(d['tv'] for d in ds), 4), 'tv_nonzero': sum(d['tv'] > 0 for d in ds),
            'argmax_agree': sum(d['argmax_agree'] for d in ds),
            'greedy_lcp_median': statistics.median(d['greedy_lcp'] for d in ds),
            'greedy_full_match': sum(d['greedy_lcp'] == d['greedy_len'] for d in ds)}


def compare(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    report = {'builds': [r.get('meta', {}) for r in runs]}
    for name, r in zip(paths, runs):
        rows = [x for x in r['rows'] if x.get('hit')]
        for kind in kinds(rows):
            label = Path(name).parent.name if Path(name).stem == 'parity' else Path(name).stem   # <arm>/parity.json
            report[f'hit_vs_fresh {label} {kind}'] = summarize(
                [pair(x['fresh'], x['hit']) for x in rows if x['kind'] == kind])
    a, b = ({x['id']: x for x in r['rows']} for r in runs[:2])
    for kind in kinds(list(a.values())):
        report[f'fresh_a_vs_fresh_b {kind}'] = summarize(
            [pair(a[k]['fresh'], b[k]['fresh']) for k in sorted(set(a) & set(b)) if a[k]['kind'] == kind])
    print(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base')
    ap.add_argument('--model')
    ap.add_argument('--corpus')
    ap.add_argument('--out')
    ap.add_argument('--prompts', type=int, default=24, help='trajectories: one long + one short prompt each')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--min-chars', type=int, default=150_000)
    ap.add_argument('--max-chars', type=int, default=330_000)
    ap.add_argument('--dp', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compare', nargs=2)
    ap.add_argument('--cpu-tier', action='store_true', help='also score a CPU-tier restore (sequential)')
    ap.add_argument('--images', type=int, default=0, help='also this many synthetic image prompts (VL specs)')
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    prompts = load_prompts(a.corpus, a.prompts, a.min_chars, a.max_chars) + image_prompts(a.images)
    meta = json.loads(OPENER.open(a.base + '/version', timeout=60).read()) if a.base else {}

    def one(i_p):
        i, p = i_p
        body, rank = {**p['body'], 'model': a.model}, i % a.dp
        row = {'id': p['id'], 'run': p['run'], 'kind': p['kind'], 'rank': rank}
        try:
            row['fresh'] = greedy(a.base, body, a.k, a.max_tokens, rank)
            row['hit'] = greedy(a.base, body, a.k, a.max_tokens, rank)
            if a.cpu_tier:
                OPENER.open(urllib.request.Request(a.base + '/reset_prefix_cache', data=b'', method='POST'), timeout=120).read()
                time.sleep(1)
                e0 = external_tokens(a.base)
                row['cpu'] = greedy(a.base, body, a.k, a.max_tokens, rank)
                row['cpu']['external_tokens'] = external_tokens(a.base) - e0
                row['cpu_vs_fresh'] = pair(row['fresh'], row['cpu'])
            d = pair(row['fresh'], row['hit'])
            print(json.dumps({'id': p['id'], 'prompt_tokens': row['fresh']['prompt_tokens'],
                              'hit_cached': row['hit']['cached'], **d}), flush=True)
        except Exception as exc:   # one rejected prompt must not lose the others
            row['error'] = repr(exc)[:300]
            print(json.dumps({'id': p['id'], 'error': row['error']}), flush=True)
        return row

    with ThreadPoolExecutor(1 if a.cpu_tier else a.workers) as pool:
        rows = list(pool.map(one, enumerate(prompts)))
    Path(a.out).write_text(json.dumps({'meta': meta, 'rows': [r for r in rows if 'fresh' in r],
                                       'errors': [r for r in rows if 'error' in r]}, indent=1))
    ok = [r for r in rows if r.get('hit')]
    summary = {kind: summarize([pair(r['fresh'], r['hit']) for r in ok if r['kind'] == kind]) for kind in kinds(ok)}
    if a.cpu_tier:
        cpu = [r for r in ok if r.get('cpu')]
        summary['cpu_vs_fresh'] = {kind: summarize([r['cpu_vs_fresh'] for r in cpu if r['kind'] == kind]) for kind in kinds(cpu)}
        summary['cpu_external_tokens'] = [round(r['cpu']['external_tokens']) for r in cpu]
    print(json.dumps({'summary': summary, 'errors': sum('error' in r for r in rows)}), flush=True)


if __name__ == '__main__':
    main()

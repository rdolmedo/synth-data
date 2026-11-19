#!/usr/bin/env python3
"""Multi-page image requests (rendered PDF pages, as GDP.pdf's vision mode sends them) against a VL server: prompt
tokens and first-token top-k logprobs, fresh and from the prefix cache, to compare two vLLM builds.

Each request is one user turn: N page images (PNG data URIs, in page order) and a question, behind a nonce system
message so no other request shares its cache. Greedy, --max-tokens tokens, top-k logprobs of the first positions.

    mm_pages_check.py --base http://host:port --model nemotron-3.5-super-vl --pages DIR --out A.json [--sizes 16,64,160]
    mm_pages_check.py --compare A.json B.json      (prompt tokens equal? first-token TV / argmax, cross-build vs hit)
"""
import argparse
import base64
import json
import math
from pathlib import Path
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
QUESTION = 'These are the pages of one document, in order. What is the document about? Answer in three sentences.'


def post(base, body, timeout=3600):
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.loads(OPENER.open(req, timeout=timeout).read())


def ask(base, model, pages, nonce, k, max_tokens):
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()}}
               for p in pages] + [{'type': 'text', 'text': QUESTION}]
    body = {'model': model, 'messages': [{'role': 'system', 'content': f'[mm-pages {nonce}] You are a helpful assistant.'},
                                         {'role': 'user', 'content': content}],
            'temperature': 0.0, 'max_tokens': max_tokens, 'logprobs': True, 'top_logprobs': k,
            'return_tokens_as_token_ids': True}
    t0 = time.time()
    r = post(base, body)
    lp = r['choices'][0]['logprobs']['content']
    usage = r.get('usage') or {}
    return {'prompt_tokens': usage.get('prompt_tokens'),
            'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'tokens': [c['token'] for c in lp], 'top': [{t['token']: t['logprob'] for t in c['top_logprobs']} for c in lp[:2]],
            'seconds': round(time.time() - t0, 1)}


def tv(a, b):
    """Total variation over the union of two top-k lists (a token missing from one list counts as 0 there)."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(math.exp(a.get(t, -math.inf)) - math.exp(b.get(t, -math.inf))) for t in keys)


def pair(x, y):
    ta, tb = x['top'][0], y['top'][0]
    lcp = next((i for i, (s, t) in enumerate(zip(x['tokens'], y['tokens'])) if s != t), min(len(x['tokens']), len(y['tokens'])))
    return {'tv': round(tv(ta, tb), 5), 'argmax_agree': max(ta, key=ta.get) == max(tb, key=tb.get),
            'same_prompt_tokens': x['prompt_tokens'] == y['prompt_tokens'], 'greedy_lcp': lcp}


def compare(pa, pb):
    a, b = (json.loads(Path(p).read_text()) for p in (pa, pb))
    out = {'builds': [a.get('meta'), b.get('meta')], 'rows': []}
    for ra, rb in zip(a['rows'], b['rows']):
        assert ra['pages'] == rb['pages']
        out['rows'].append({'pages': ra['pages'], 'prompt_tokens': [ra['fresh']['prompt_tokens'], rb['fresh']['prompt_tokens']],
                            'cross_fresh': pair(ra['fresh'], rb['fresh']),
                            'hit_vs_fresh': [pair(ra['fresh'], ra['hit']), pair(rb['fresh'], rb['hit'])]})
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base')
    ap.add_argument('--model')
    ap.add_argument('--pages')
    ap.add_argument('--out')
    ap.add_argument('--sizes', default='16,64,160')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=16)
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    pages = sorted(Path(a.pages).glob('*.png'))
    meta = json.loads(OPENER.open(a.base + '/version', timeout=60).read())
    rows = []
    for n in map(int, a.sizes.split(',')):
        row = {'pages': n}
        try:
            row['fresh'] = ask(a.base, a.model, pages[:n], f'{n}', a.k, a.max_tokens)
            row['hit'] = ask(a.base, a.model, pages[:n], f'{n}', a.k, a.max_tokens)
        except Exception as exc:   # noqa: BLE001  (one rejected size must not lose the others)
            row['error'] = repr(exc)[:500]
        print(json.dumps({'pages': n, **({k: row['fresh'][k] for k in ('prompt_tokens', 'seconds')} if 'fresh' in row else
                                          {'error': row.get('error')})}), flush=True)
        rows.append(row)
    Path(a.out).write_text(json.dumps({'meta': meta, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()

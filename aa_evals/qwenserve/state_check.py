#!/usr/bin/env python3
"""Next-token distributions on long recorded contexts, for comparing engine numerics (e.g. GDN state dtype).
Ported from inference/glm53_state_check.py (data-synthesis) for Qwen3.8-27B's api-dir corpus (#70): request bodies
are read from the corpus' request.json paths (no wire log).

A synth.tune candidate `check` (run against the fresh engine before its replay):

    glm53_state_check.py --base http://host:port --model glm-5.3-flash --corpus ops/tuning/corpus.json --out X.json
                         [--prompts 24] [--k 20] [--min-tokens 60000]

Prompts are the latest recorded request under --max-chars (~110k tokens: the last ones sit at the 132k context cap,
which the nonce would push over) of the longest corpus trajectories (deterministic), each made unique to this
check by a fixed nonce at the start of the system prompt, so two engines see byte-identical prompts that no replay
shares. Each prompt is scored twice on the engine: `fresh` (the whole prompt computed) and `hit` (the same request
again, served from the GPU prefix cache). The record holds the top-k logprobs of the first generated token (greedy,
max_tokens 1). Compare engines with --compare a.json b.json: cross-engine fresh-vs-fresh differences are judged
against each engine's own hit-vs-fresh differences (the noise floor of cached vs recomputed numerics).
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def first_token(base, body, k):
    body = {**body, 'stream': False, 'temperature': 0.0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': k,
            'seed': 0}
    for key in ('stream_options', 'max_completion_tokens'):
        body.pop(key, None)
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    out = json.loads(OPENER.open(request, timeout=1800).read())
    content = out['choices'][0]['logprobs']['content'][0]
    return {'top': {t['token']: t['logprob'] for t in content['top_logprobs']}, 'token': content['token'],
            'prompt_tokens': out['usage']['prompt_tokens'], 'cached': (out['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')}


def prompts(corpus, count, min_tokens, max_chars=330_000):
    def bodies(t):
        if 'wire' in t:
            from synth.bench import wire_bodies
            return wire_bodies(t)
        return [json.loads(Path(r).read_text()) for r in t['requests']]
    trajectories = json.loads(Path(corpus).read_text())['trajectories']
    # Longest trajectories by request count first; the last request carries the longest context.
    trajectories = sorted(trajectories, key=lambda t: (-len(t['requests']), t['run']))
    out = []
    for t in trajectories:
        if len(out) >= count:
            break
        if 'wire' in t:
            sized = [(sum(len(json.dumps(m)) for m in b['messages']), b) for b in bodies(t)]
            sized = [(c, b) for c, b in sized if min_tokens * 3 <= c <= max_chars]
            if not sized:
                continue
            body = sized[-1][1]
        else:   # request.json files: size them on disk, load only the chosen one
            import os
            fit = [r for r in t['requests'] if min_tokens * 3 <= os.path.getsize(r) <= max_chars]
            if not fit:
                continue
            body = json.loads(Path(fit[-1]).read_text())
        nonce = f'[state-check {len(out):03d}] '
        messages = [dict(m) for m in body['messages']]
        system = next(i for i, m in enumerate(messages) if m.get('role') == 'system')
        messages[system]['content'] = nonce + messages[system]['content']
        out.append({'run': t['run'], 'body': {**body, 'messages': messages}})
    return out


def diff(a, b):
    common = set(a) & set(b)
    return max((abs(a[t] - b[t]) for t in common), default=math.inf), max(a, key=a.get) == max(b, key=b.get)


def compare(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    report = {}
    for name, r in zip(paths, runs):
        d = [diff(x['fresh']['top'], x['hit']['top']) for x in r['rows'] if x.get('hit')]
        report[f'hit_vs_fresh {Path(name).stem}'] = {'max_dlogprob_median': statistics.median(x[0] for x in d),
                                                     'max_dlogprob_max': max(x[0] for x in d),
                                                     'argmax_agree': sum(x[1] for x in d), 'n': len(d)}
    a, b = ({x['run']: x for x in r['rows']} for r in runs[:2])
    d = [diff(a[k]['fresh']['top'], b[k]['fresh']['top']) for k in sorted(set(a) & set(b))]
    report['fresh_a_vs_fresh_b'] = {'max_dlogprob_median': statistics.median(x[0] for x in d),
                                    'max_dlogprob_max': max(x[0] for x in d), 'argmax_agree': sum(x[1] for x in d),
                                    'n': len(d)}
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base')
    ap.add_argument('--model')
    ap.add_argument('--corpus')
    ap.add_argument('--out')
    ap.add_argument('--prompts', type=int, default=24)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--min-tokens', type=int, default=60000)
    ap.add_argument('--max-chars', type=int, default=330_000)
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    rows = []
    for p in prompts(a.corpus, a.prompts, a.min_tokens, a.max_chars):
        body = {**p['body'], 'model': a.model}
        try:
            row = {'run': p['run'], 'fresh': first_token(a.base, body, a.k)}
            row['hit'] = first_token(a.base, body, a.k)
        except Exception as exc:   # one rejected prompt must not lose the others
            print(json.dumps({'run': Path(p['run']).name, 'error': repr(exc)[:300]}), flush=True)
            continue
        rows.append(row)
        print(json.dumps({'run': Path(p['run']).name, 'prompt_tokens': row['fresh']['prompt_tokens'],
                          'hit_cached': row['hit']['cached'], 'd': diff(row['fresh']['top'], row['hit']['top'])}),
              flush=True)
    Path(a.out).write_text(json.dumps({'rows': rows}, indent=1))


if __name__ == '__main__':
    main()

"""Snapshot vLLM token counters of serving endpoints (GPU-hour accounting by token share, #359).

  python3 m5/automationbench/snap.py OUT.jsonl m5-ds m5-super-1 ...

Appends {"t", "name", "url", "prompt_tokens", "generation_tokens", "prefix_hit"} per engine URL. Our GPU-hours =
engine GPUs x hours between two snapshots x (our generation tokens / the engine's generation tokens) in that window.
"""
import json
import sys
import time
import urllib.request

OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def counters(base: str) -> dict:
    url = base.rstrip('/').removesuffix('/v1') + '/metrics'
    text = OPEN.open(url, timeout=20).read().decode()
    out = {'prompt_tokens': 0.0, 'generation_tokens': 0.0, 'prefix_queries': 0.0, 'prefix_hits': 0.0}
    for ln in text.splitlines():
        if ln.startswith('#'):
            continue
        name = ln.split('{')[0].split(' ')[0]
        v = float(ln.rsplit(' ', 1)[-1])
        if name == 'vllm:prompt_tokens_total':
            out['prompt_tokens'] += v
        elif name == 'vllm:generation_tokens_total':
            out['generation_tokens'] += v
        elif name in ('vllm:prefix_cache_queries_total', 'vllm:gpu_prefix_cache_queries_total'):
            out['prefix_queries'] += v
        elif name in ('vllm:prefix_cache_hits_total', 'vllm:gpu_prefix_cache_hits_total'):
            out['prefix_hits'] += v
    return out


def main():
    out = sys.argv[1]
    with open(out, 'a') as f:
        for name in sys.argv[2:]:
            try:
                e = json.load(open(f'/data/aai/tbench/tasksynth/serving/{name}/endpoints.json'))
            except (OSError, ValueError) as ex:
                print(name, 'no endpoints.json', ex)
                continue
            for u in e.get('engine_urls') or e.get('urls') or []:
                try:
                    c = counters(u)
                except Exception as ex:  # noqa: BLE001
                    print(name, u, 'metrics failed', ex)
                    continue
                rec = {'t': time.time(), 'name': name, 'url': u, 'job': e.get('job_id'), **c}
                f.write(json.dumps(rec) + '\n')
                print(json.dumps(rec))


if __name__ == '__main__':
    main()

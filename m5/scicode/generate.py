"""Generate SciCode-format problems with DeepSeek V4.1 Flash (m5-ds) from taxonomy leaves.

    python -m m5.scicode.generate --out DIR --n N [--seed S]

Each draw: cell (subfield|method) by its fitted weight -> a leaf of that cell (least-used first; 3% open draws get
the cell only) -> a step count from the test set's smoothed step-count distribution. The generator sees the class
descriptions, the leaf and the step count, never a test item. High reasoning effort, no output cap (the server's
context bounds it), 6 h client timeout. A reply is parsed and run through checks.check(); a failing draw is retried
once with a fresh sample, then recorded as rejected.

Out: DIR/draws.jsonl (every draw), DIR/specs.jsonl (accepted specs, streamed to the Super runner),
DIR/rejects.jsonl, DIR/replies.jsonl (raw reply + reasoning per attempt; packed with zstd at the end).
Resumable: draws already in draws.jsonl are not redrawn; unfinished ones are generated again.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
import time
from pathlib import Path

from . import checks
from .common import Alloc, JsonlWriter, read_jsonl, safe_write
from .opus import parse_json

HERE = Path(__file__).resolve().parent
VOCAB = HERE / 'taxonomy' / 'vocab.json'
CLASSES = json.loads((HERE / 'taxonomy' / 'classes.json').read_text())
SYSTEM = (HERE / 'gen_prompt.md').read_text()
OPEN_SHARE = 0.03
EFFORT = 'high'


class Sampler:
    def __init__(self, vocab: dict, seed: int, used: collections.Counter | None = None):
        self.v, self.rng = vocab, random.Random(seed)
        self.cells = list(vocab['cells'])
        self.w = [vocab['cells'][c] for c in self.cells]
        self.used = used or collections.Counter()
        self.steps = [int(k) for k in vocab['p_steps']]
        self.sw = [vocab['p_steps'][str(k)] for k in self.steps]

    def fresh(self, cell) -> bool:
        return any(self.used[x['id']] == 0 for x in self.v['leaves'].get(cell) or [])

    def draw(self, i: int) -> dict:
        cell = self.rng.choices(self.cells, self.w)[0]
        if not self.fresh(cell):   # cell out of unused leaves: redraw among cells that still have some (no leaf reuse
            alt = [(c, w) for c, w in zip(self.cells, self.w) if self.fresh(c)]   # until the vocab runs out; ~1.5%
            if alt:                                                                # of draws at 9k move cells)
                cell = self.rng.choices([c for c, _ in alt], [w for _, w in alt])[0]
        n = self.rng.choices(self.steps, self.sw)[0]
        leaves = self.v['leaves'].get(cell) or []
        d = {'draw': i, 'cell': cell, 'subfield': cell.split('|')[0], 'method': cell.split('|')[1], 'n_steps': n}
        if self.rng.random() < OPEN_SHARE or not leaves:
            d['leaf'] = None
        else:
            lo = min(self.used[x['id']] for x in leaves)
            leaf = self.rng.choice([x for x in leaves if self.used[x['id']] == lo])
            self.used[leaf['id']] += 1
            d['leaf'] = leaf
        return d


def user_prompt(d: dict) -> str:
    s, m = d['subfield'], d['method']
    f = s.split('/')[0]
    if d['leaf']:
        topic = f"Topic: {d['leaf']['title']}\n{d['leaf']['gist']}"
    else:
        topic = 'Topic: choose a specific, less common topic of your own in this subfield that suits this method family.'
    return (f"Field: {CLASSES['fields'][f]}\n"
            f"Subfield: {CLASSES['subfields'][s]}\n"
            f"Method family: {CLASSES['methods'][m]}\n"
            f"{topic}\n"
            f"Number of steps: {d['n_steps']}\n\n"
            f"Write one problem on this topic with exactly {d['n_steps']} step{'s' if d['n_steps'] > 1 else ''}, "
            "following the format and rules above.")


def ds_body(user: str, request: dict, system: str | None = None) -> dict:
    ov = dict((request or {}).get('overrides') or {})
    ctk = {**(ov.pop('chat_template_kwargs', None) or {}), 'reasoning_effort': EFFORT}
    ov.pop('max_tokens', None)
    return {'messages': [{'role': 'system', 'content': system or SYSTEM}, {'role': 'user', 'content': user}],
            **ov, 'chat_template_kwargs': ctk, 'reasoning_effort': EFFORT}


class DS:
    """DeepSeek requests over every server alloc.json lists in ds.serving (re-read each registry refresh, so servers can
    be added and retired mid-run; a fixed `name` pins one server instead). Each server's endpoints.json counts only
    when healthy and fresh (updated < 120 s ago); requests go to the least-loaded (server URL, DP rank), pinned with
    X-data-parallel-rank. No usable server: wait and re-read."""

    def __init__(self, name: str | None = None, timeout=6 * 3600):
        import httpx
        self.name = name
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30), trust_env=False,
                                      limits=httpx.Limits(max_connections=None, max_keepalive_connections=512))
        self.load = collections.Counter()
        self.alloc = Alloc()
        self.servers, self.t = {}, 0.0
        self._dp = {}

    def names(self) -> list[str]:
        if self.name:
            return [self.name]
        return list((self.alloc.get().get('ds') or {}).get('serving') or [])

    async def registry(self) -> dict:
        """{server name: endpoints info} for usable servers."""
        from .common import SERVING
        while True:
            if not self.servers or time.time() - self.t > 30:
                live = {}
                for n in self.names():
                    try:
                        info = json.loads((SERVING / n / 'endpoints.json').read_text())
                        if info.get('healthy_at') and time.time() - info['updated_at'] < 120 and info.get('urls'):
                            live[n] = info
                    except (OSError, ValueError, KeyError):
                        pass
                if set(live) != set(self.servers):
                    print(f'{time.strftime("%H:%M:%S")} DeepSeek servers: {sorted(live)}', flush=True)
                self.servers, self.t = live, time.time()
            if self.servers:
                return self.servers
            print(f'{time.strftime("%H:%M:%S")} no usable DeepSeek server in {self.names()}, waiting', flush=True)
            await asyncio.sleep(60)

    def dp_size(self, server, url):
        """DP ranks behind one URL, from the server's launch.json (vLLM routes headerless requests itself)."""
        from .common import SERVING
        if (server, url) in self._dp:
            return self._dp[(server, url)]
        n = 1
        try:
            launch = json.loads((SERVING / server / 'launch.json').read_text())
            reps = launch.get('replicas') or [launch]
            for rep in reps:
                argv = rep['argv']
                port = argv[argv.index('--port') + 1] if '--port' in argv else None
                if len(reps) == 1 or (port and url.rstrip('/').endswith(f':{port}/v1')):
                    n = int(argv[argv.index('--data-parallel-size') + 1]) if '--data-parallel-size' in argv else 1
        except (OSError, ValueError, KeyError, IndexError):
            pass
        self._dp[(server, url)] = n
        return n

    async def chat(self, user: str, max_attempts=20, system: str | None = None):
        import httpx
        err = None
        for attempt in range(max_attempts):
            servers = await self.registry()
            targets = [(n, u, r) for n, info in servers.items() for u in info['urls'] for r in range(self.dp_size(n, u))]
            server, url, rank = min(targets, key=lambda t: (self.load[t[1:]], random.random()))
            info = servers[server]
            self.load[(url, rank)] += 1
            t0 = time.time()
            try:
                r = await self.http.post(url + '/chat/completions', json={'model': info['served_name'],
                                                                          **ds_body(user, info.get('request'), system)},
                                         headers={'X-data-parallel-rank': str(rank)})
                if r.status_code == 200:
                    data = r.json()
                    ch = data['choices'][0]
                    msg = ch.get('message') or {}
                    return {'content': msg.get('content') or '',
                            'reasoning': msg.get('reasoning_content') or msg.get('reasoning') or '',
                            'finish_reason': ch.get('finish_reason'), 'usage': data.get('usage') or {},
                            'latency_s': round(time.time() - t0, 1), 'server': server, 'rank': rank,
                            'attempts': attempt + 1}
                err = f'{server} HTTP {r.status_code}: {r.text[:300]}'
                if 400 <= r.status_code < 500 and r.status_code not in (408, 409, 429):
                    if attempt >= 2:
                        break
            except (httpx.HTTPError, ValueError, KeyError) as e:
                err = f'{server} {type(e).__name__}: {str(e)[:300]}'
                self.t = 0.0   # re-read the registries before the retry
            finally:
                self.load[(url, rank)] -= 1
            await asyncio.sleep(min(300, 10 * 2 ** min(attempt, 5)) * (0.5 + random.random()))
        raise RuntimeError(f'DS request failed: {err}')


def normalize(spec: dict) -> dict:
    """Light, non-semantic cleanup: drop unknown keys, renumber steps."""
    out = {k: spec.get(k) for k in checks.KEYS}
    steps = []
    for s in spec.get('sub_steps') or []:
        if isinstance(s, dict):
            steps.append({k: s.get(k) for k in checks.STEP_KEYS})
    out['sub_steps'] = steps
    if isinstance(out.get('required_dependencies'), list):
        out['required_dependencies'] = '\n'.join(out['required_dependencies'])
    return out


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(a.vocab or VOCAB).read_text())
    draws_w, specs_w, rej_w, rep_w = (JsonlWriter(out / f) for f in
                                      ('draws.jsonl', 'specs.jsonl', 'rejects.jsonl', 'replies.jsonl'))
    old_draws = {d['draw']: d for d in read_jsonl(out / 'draws.jsonl')}
    finished = {r['draw'] for r in read_jsonl(out / 'specs.jsonl')} | {r['draw'] for r in read_jsonl(out / 'rejects.jsonl')}
    used = collections.Counter(d['leaf']['id'] for d in old_draws.values() if d.get('leaf'))
    for spec in (a.used_from.split(',') if a.used_from else []):   # GEN_DIR:TRACES_DIR, leaves of traced problems
        g, t = spec.split(':')
        traced = {c['problem_id'] for c in read_jsonl(Path(t) / 'chains.jsonl')}
        dmap = {d['draw']: d for d in read_jsonl(Path(g) / 'draws.jsonl')}
        for r in read_jsonl(Path(g) / 'specs.jsonl'):
            d = dmap.get(r['draw'])
            if r['problem_id'] in traced and d and d.get('leaf'):
                used[d['leaf']['id']] += 1
    print(f'{sum(used.values())} leaf uses carried in', flush=True)
    sampler = Sampler(vocab, a.seed, used)
    dedup = None
    if a.dedup:
        from .dedup import Filter
        prior = []
        for g in (a.dedup_against.split(',') if a.dedup_against else []):
            prior += read_jsonl(Path(g) / 'specs.jsonl')
        dedup = Filter(vocab, prior + read_jsonl(out / 'specs.jsonl'))
        print(f'dedup filter over {len(dedup.specs)} admitted specs', flush=True)
    for _ in range(len(old_draws)):   # advance the RNG past old draws so new ones continue the sequence
        sampler.rng.random()
    ds = DS(a.serving)
    alloc = Alloc()
    inflight = 0
    stop_file = out / 'STOP'
    stats = collections.Counter()

    async def one(d):
        nonlocal inflight
        try:
            for attempt in range(2):
                user = user_prompt(d)
                try:
                    g = await ds.chat(user)
                except RuntimeError as e:
                    await rep_w.write({'draw': d['draw'], 'attempt': attempt, 'error': str(e)})
                    stats['infra'] += 1
                    return   # unfinished: a rerun generates it again
                await rep_w.write({'draw': d['draw'], 'attempt': attempt, **g})
                errs = []
                try:
                    spec = normalize(parse_json(g['content']))
                    if not isinstance(spec, dict):
                        raise ValueError
                    errs = checks.check(spec, d['n_steps'])
                except (ValueError, AttributeError, TypeError):
                    spec, errs = None, ['reply holds no JSON spec']
                if g['finish_reason'] == 'length':
                    errs.append('generator hit the context limit')
                pid = f"syn{a.prefix}{d['draw']:06d}"
                if not errs and dedup is not None:
                    dup, dlog = await dedup.is_duplicate(ds, {**spec, 'problem_id': pid})
                    if dup:
                        await rej_w.write({'draw': d['draw'], 'cell': d['cell'], 'errors': ['duplicate'], 'dedup': dlog})
                        stats['duplicate'] += 1
                        return
                if not errs:
                    if dedup is not None:
                        dedup.add({**spec, 'problem_id': pid})
                    await specs_w.write({'problem_id': pid, 'draw': d['draw'], 'attempt': attempt, 'cell': d['cell'],
                                         'leaf_id': (d['leaf'] or {}).get('id'), 'n_steps_target': d['n_steps'],
                                         'gen_usage': g['usage'], **spec})
                    stats['accepted'] += 1
                    return
                stats[f'fail_attempt{attempt}'] += 1
                last = errs
            await rej_w.write({'draw': d['draw'], 'cell': d['cell'], 'errors': last})
            stats['rejected'] += 1
        finally:
            inflight -= 1

    tasks = []
    todo = [old_draws[i] for i in sorted(old_draws) if i not in finished]
    i = len(old_draws)
    t0 = time.time()
    last_print = 0
    while True:
        if stop_file.exists():
            break
        if not todo and i >= a.n:
            break
        cap = alloc.ds_cap() if a.max_inflight is None else min(a.max_inflight, alloc.ds_cap())
        while inflight < cap and (todo or i < a.n):
            if todo:
                d = todo.pop(0)
            else:
                d = sampler.draw(i)
                i += 1
                await draws_w.write(d)
            inflight += 1
            tasks.append(asyncio.create_task(one(d)))
        if time.time() - last_print > 300:
            print(f'{time.strftime("%H:%M:%S")} drawn {i} inflight {inflight} cap {cap} {dict(stats)} '
                  f'{time.time() - t0:.0f}s', flush=True)
            last_print = time.time()
        await asyncio.sleep(2)
    await asyncio.gather(*tasks)
    safe_write(out / 'DONE', json.dumps({'stats': dict(stats), 'at': time.time()}))
    print(f'done: {dict(stats)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, required=True, help='total draws')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prefix', default='', help='problem-id prefix (e.g. p for pilot)')
    ap.add_argument('--serving', default=None, help='pin one DeepSeek server (default: alloc.json ds.serving)')
    ap.add_argument('--vocab', default=None)
    ap.add_argument('--dedup', action='store_true', help='streaming near-duplicate filter (dedup.py; disabled by default: the coordinator put repeat fixes in the taxonomy, #53 lesson 2)')
    ap.add_argument('--dedup-against', default='', help='earlier gen dirs whose specs count as admitted')
    ap.add_argument('--used-from', default='', help='GEN_DIR:TRACES_DIR[,...]: leaves of earlier traced problems count as used')
    ap.add_argument('--max-inflight', type=int, default=None, help='below the alloc.json cap')
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == '__main__':
    main()

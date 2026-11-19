"""CritPt problem generator: taxonomy leaf + seed paper -> one challenge (DeepSeek, high effort, no output cap).

  python -m m5.critpt.gen --taxonomy T.json --seeds DIR --out DIR --n N [--concurrency C] [--seed S]

Per call: sample a leaf (field by fitted weight; subtopic uniform within the field, OPEN_SHARE open draws;
technique uniform within the field; answer type by fitted weight; work mode | answer type; grounding), take an
unused seed paper from the subtopic's candidates (each paper is used once), write the challenge, parse it, run the
static problem checks and the LLM well-posedness check. A skip or a failed check retries the leaf with another
paper (up to 3 papers), then the leaf is redrawn.
Outputs (append-only, resumable): OUT/problems.jsonl (accepted), OUT/rejects.jsonl, OUT/calls.jsonl (usage).
Nothing in OUT holds an answer.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import os
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import zstandard

from . import checks, prompts, safeio
from .llm import Server

PAPER_CHARS = 120000
MIN_MATH = 15


def load_example():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from aa_evals.critpt import data
    return data.load_example()


def clean_paper(text: str) -> str:
    text = re.sub(r'\\begin\{thebibliography\}.*?(\\end\{thebibliography\}|$)', '', text, flags=re.S)
    text = re.sub(r'(?m)^%.*\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    if len(text) > PAPER_CHARS:
        text = text[:PAPER_CHARS] + '\n\n[... the rest of the paper is truncated ...]'
    return text


def read_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


class Sampler:
    def __init__(self, tax: dict, cand: dict, papers: dict, used: set, rng: random.Random):
        self.tax, self.cand, self.papers, self.used, self.rng = tax, cand, papers, used, rng
        self.fields = list(tax['fields'])
        self.fw = [tax['fields'][f]['weight'] for f in self.fields]
        self.at_boost = {k: 1.0 for k in tax['answer_types']}

    def rebalance(self, outdir: Path, min_draws: int = 30):
        """Answer types differ in accept rate (the generator skips symbolic_expr more often). Draw each type with
        weight target / rate (rate = accepted / drawn calls of that type in OUT, normalized by the overall rate;
        boost clipped to [0.5, 2.5]) so the ACCEPTED mix tracks the taxonomy. Non-compounding: recomputed from the
        files each time."""
        drawn = {}
        for r in read_jsonl(outdir / 'calls.jsonl'):
            drawn[r['k']] = r['leaf']['answer_type']
        acc = collections.Counter(r['leaf']['answer_type'] for r in read_jsonl(outdir / 'problems.jsonl'))
        nd = collections.Counter(drawn.values())
        tot_rate = sum(acc.values()) / max(1, len(drawn))
        for k in self.tax['answer_types']:
            if nd.get(k, 0) >= min_draws and acc.get(k, 0) and tot_rate:
                self.at_boost[k] = min(2.5, max(0.5, tot_rate / (acc[k] / nd[k])))

    def leaf(self) -> dict:
        r = self.rng
        f = r.choices(self.fields, self.fw)[0]
        fd = self.tax['fields'][f]
        if r.random() < self.tax['open_share']:
            sub = {'id': f'{f}::open', 'name': f'any subtopic of {f} that the paper fits',
                   'scope': 'open draw: choose the subtopic from the paper'}
        else:
            sub = r.choice(fd['subtopics'])
        tech = r.choice(fd['techniques'])
        ats = list(self.tax['answer_types'])
        at = r.choices(ats, [self.tax['answer_types'][k] * self.at_boost[k] for k in ats])[0]
        wmd = self.tax['work_mode_given_answer_type'][at]
        wm = r.choices(list(wmd), list(wmd.values()))[0]
        gd = self.tax['grounding']
        g = r.choices(list(gd), list(gd.values()))[0]
        return {'field': f, 'subtopic_id': sub['id'], 'subtopic': sub['name'], 'subtopic_scope': sub['scope'],
                'technique': tech['name'], 'work_mode': wm, 'answer_type': at, 'grounding': g}

    def paper(self, leaf: dict, top: int = 80):
        c = [x for x, _ in self.cand.get(leaf['subtopic_id'], [])[:top] if x not in self.used and x in self.papers]
        if not c:
            c = [x for x, _ in self.cand.get(f'{leaf["field"]}::open', []) if x not in self.used and x in self.papers]
        if not c:
            return None
        # prefer higher-ranked candidates: weight 1/sqrt(rank)
        x = self.rng.choices(c[:top], [1 / (i + 1) ** 0.5 for i in range(len(c[:top]))])[0]
        self.used.add(x)
        return x


def load_seeds(seeds: Path, need: set | None = None) -> dict:
    """arxiv_id -> record with zstd-compressed text (keeps memory small)."""
    cc = zstandard.ZstdCompressor(level=3)
    out = {}
    with open(seeds / 'seeds.jsonl.zst', 'rb') as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding='utf-8'):
            r = json.loads(line)
            if (need is not None and r['arxiv_id'] not in need) or r['n_math'] < MIN_MATH:
                continue
            r['ztext'] = cc.compress(r.pop('text').encode())
            out[r['arxiv_id']] = r
    return out


async def one(server: Server, check_server: Server, sampler: Sampler, example, out: Path, k: int, lock):
    leaf = sampler.leaf()
    for attempt in range(3):
        pid = sampler.paper(leaf)
        if pid is None:
            leaf = sampler.leaf()
            continue
        paper = sampler.papers[pid]
        text = clean_paper(zstandard.ZstdDecompressor().decompress(paper['ztext']).decode())
        prompt = prompts.gen_prompt(leaf, paper, text, example)
        r = await server.chat([{'role': 'user', 'content': prompt}])
        call = {'k': k, 'attempt': attempt, 'arxiv_id': pid, 'leaf': leaf, 'prompt_tokens': r['prompt_tokens'],
                'completion_tokens': r['completion_tokens'], 'finish': r['finish'], 'seconds': r['seconds'],
                'reasoning_chars': len(r['reasoning']), 'at': time.time()}
        g = checks.parse_generation(r['content'])
        rec = {'k': k, 'arxiv_id': pid, 'title': paper.get('title'), 'categories': paper.get('categories'),
               'leaf': leaf, 'gen': {kk: call[kk] for kk in ('prompt_tokens', 'completion_tokens', 'finish',
                                                             'seconds', 'reasoning_chars')}}
        if not g['ok']:
            await append(out / 'rejects.jsonl', {**rec, 'stage': 'skip' if g.get('skip') else 'format',
                                                 'reason': g['reason'], 'raw': r['content'][-3000:]}, lock)
            await append(out / 'calls.jsonl', call, lock)
            continue
        g['code_template'], rec['template_blanked'] = checks.repair_template(g['code_template'])
        errs = checks.static_checks(g['description'], g['code_template'], leaf['answer_type'])
        rec.update(description=g['description'], code_template=g['code_template'], source=g['source'],
                   target=g['target'])
        if errs:
            await append(out / 'rejects.jsonl', {**rec, 'stage': 'static', 'reason': errs}, lock)
            await append(out / 'calls.jsonl', call, lock)
            continue
        cr = await check_server.chat([{'role': 'user', 'content': prompts.check_prompt(g['description'],
                                                                                      g['code_template'])}])
        call['check_tokens'] = [cr['prompt_tokens'], cr['completion_tokens']]
        await append(out / 'calls.jsonl', call, lock)
        try:
            t = cr['content']
            verdict = json.loads(t[t.index('{'): t.rindex('}') + 1])
        except ValueError:
            verdict = {'parse_error': cr['content'][-500:]}
        bad = [x for x in ('defined', 'self_contained', 'no_leak', 'single_target', 'template_ok')
               if verdict.get(x) is not True]
        rec['check'] = verdict
        if bad:
            await append(out / 'rejects.jsonl', {**rec, 'stage': 'llm_check', 'reason': bad}, lock)
            continue
        rec['problem_id'] = 'cp-' + hashlib.sha256((pid + g['description']).encode()).hexdigest()[:10]
        rec['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        await append(out / 'problems.jsonl', rec, lock)
        return True
    return False


async def append(path: Path, rec: dict, lock):
    async with lock:   # verified append, retried on EDQUOT/EIO (blocking: a retry pauses the other workers too)
        await asyncio.to_thread(safeio.append_line, path, json.dumps(rec))


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tax = json.loads(Path(a.taxonomy).read_text())
    cand = json.loads((Path(a.seeds) / 'candidates.json').read_text())
    papers = load_seeds(Path(a.seeds))
    print(f'{len(papers)} seed papers', flush=True)
    used = {r['arxiv_id'] for r in read_jsonl(out / 'calls.jsonl')}
    if a.exclude_used:
        for p in a.exclude_used.split(','):
            used |= {r['arxiv_id'] for r in read_jsonl(Path(p) / 'calls.jsonl')}
    done = len(read_jsonl(out / 'problems.jsonl'))
    rng = random.Random(f'{a.seed}/{len(used)}')
    sampler = Sampler(tax, cand, papers, used, rng)
    example = load_example()
    server = Server(a.server, effort='high')
    check_server = Server(a.server, effort=None)
    check_server.gate = server   # generation and checks share the one alloc cap (ds.max_inflight.critpt)
    if a.concurrency:
        import m5.critpt.llm as L
        server._read_cap = check_server._read_cap = lambda: None
        server.cap = check_server.cap = a.concurrency
    lock = asyncio.Lock()
    need = a.n - done
    print(f'{done} accepted already; {need} to go', flush=True)
    sem = asyncio.Semaphore(10 ** 6)
    state = {'accepted': done, 'k': len(used)}
    stop = asyncio.Event()

    ctl_path = Path(a.control) if a.control else out / 'control.json'
    sampler.rebalance(out)

    async def control():
        """OUT/control.json, re-read every 60 s: {"n": N} moves the target; {"abort": true} exits at once (in-flight
        calls are dropped: cuts the tail when the token target is reached). Also rebalances answer types."""
        last = None
        while not stop.is_set():
            try:
                c = json.loads(ctl_path.read_text()) if ctl_path.exists() else {}
            except (OSError, ValueError) as e:
                c = last or {}
                print(f'control unreadable ({e!r}); keeping {c}', flush=True)
            if c != last:
                print(f'{time.strftime("%H:%M:%S")} control {c}', flush=True)
                last = c
            if c.get('n'):
                a.n = int(c['n'])
            if c.get('abort'):
                print(f'abort: {state["accepted"]} accepted, dropping in-flight calls', flush=True)
                os._exit(0)
            if state['accepted'] >= a.n:
                stop.set()
            sampler.rebalance(out)
            for _ in range(60):
                if stop.is_set():
                    return
                await asyncio.sleep(1)

    async def worker():
        while not stop.is_set():
            state['k'] += 1
            ok = await one(server, check_server, sampler, example, out, state['k'], lock)
            if ok:
                state['accepted'] += 1
                if state['accepted'] >= a.n:
                    stop.set()
            if state['k'] % 20 == 0:
                print(f'{time.strftime("%H:%M:%S")} calls {state["k"]} accepted {state["accepted"]}/{a.n} '
                      f'{server.stats} boost {dict((k, round(v, 2)) for k, v in sampler.at_boost.items())}',
                      flush=True)

    tasks = []

    def grow():
        """Each worker holds at most one call, so the pipeline count follows the live cap: --workers is the
        starting count; when alloc raises the cap, workers are added (cap + 10); a lowered cap just blocks them."""
        server._read_cap()
        want = max(a.workers or 0, (a.concurrency or server.cap) + 10)
        if len(tasks) < want and not stop.is_set():
            print(f'{time.strftime("%H:%M:%S")} workers {len(tasks)} -> {want} (cap {server.cap})', flush=True)
            tasks.extend(asyncio.create_task(worker()) for _ in range(want - len(tasks)))

    grow()
    ctl = asyncio.create_task(control())
    while not stop.is_set():
        await asyncio.sleep(60)
        grow()
    await asyncio.gather(ctl, *tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taxonomy', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, required=True, help='stop after this many accepted problems')
    ap.add_argument('--server', default='auto', help="'auto': alloc.json ds.serving; or one server name")
    ap.add_argument('--concurrency', type=int, default=0, help='fixed cap (default: alloc.json, re-read every 60 s)')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', default='critpt-gen-v1')
    ap.add_argument('--exclude-used', default='', help='other OUT dirs whose papers are not reused')
    ap.add_argument('--control', default='', help='live control JSON (default OUT/control.json): {"n": N} / {"abort": true}')
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == '__main__':
    main()

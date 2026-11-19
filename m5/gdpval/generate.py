"""Generate GDPval-style problems with DeepSeek (high effort, no output cap, 6 h client timeout), one call per spec.

  python -m m5.gdpval.generate --out DIR --start 0 --n 200 [--endpoint m5-ds] [--docs DOCS_INDEX]

Writes DIR/gen.jsonl (one record per spec index: spec, document reference, parsed problem, usage, static-check
result). Resumable: indices already in gen.jsonl are skipped. Concurrency follows the worker's cap in alloc.json
(re-read every 60 s). Survives server restarts: a failed request waits for a healthy endpoints.json and is retried.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from m5.gdpval import prompts as P
from m5.gdpval.llm import Endpoint, wait_healthy
from m5.gdpval.sampler import Sampler

ALLOC = Path('/opt/aai/m5-coord/alloc.json')
WORKER = 'gdpval'
EXT_OK = {'xlsx', 'pdf', 'docx', 'pptx', 'csv', 'txt', 'md', 'json'}


def cap() -> int:
    try:
        return int(json.loads(ALLOC.read_text())['ds']['max_inflight'][WORKER])
    except (OSError, ValueError, KeyError):
        return 8


class Gate:
    """A semaphore whose size follows alloc.json (this process's share of the worker's DeepSeek cap)."""

    def __init__(self, share: float = 1.0):
        self.share = share
        self.n, self.limit, self.cond, self.t = 0, self._cap(), asyncio.Condition(), time.time()

    def _cap(self):
        return max(1, int(cap() * self.share))

    async def __aenter__(self):
        async with self.cond:
            while True:
                if time.time() - self.t > 60:
                    self.limit, self.t = self._cap(), time.time()
                if self.n < self.limit:
                    self.n += 1
                    return
                try:
                    await asyncio.wait_for(self.cond.wait(), 30)
                except asyncio.TimeoutError:
                    pass

    async def __aexit__(self, *a):
        async with self.cond:
            self.n -= 1
            self.cond.notify_all()


def static_check(spec: dict, prob: dict) -> list[str]:
    """Cheap problem-side checks before the sandbox run (that one runs the script and opens the files)."""
    bad = []
    words = len(prob['task_prompt'].split())
    if words < 60 or words > 2000:
        bad.append(f'prompt_words={words}')
    if spec['lane'] == 'real' and spec.get('_doc_name'):   # the real document is staged by the bundle, not the script
        prob['input_files'] = [(n, d) for n, d in prob['input_files'] if n != spec['_doc_name']]
        if spec['_doc_name'] not in prob['task_prompt']:
            bad.append('real_doc_not_named_in_prompt')
    names = [n for n, _ in prob['input_files']]
    if spec['lane'] == 'synthetic' or (spec['lane'] == 'real' and spec.get('input_exts')):
        if not names or not prob['script']:
            bad.append('synthetic_inputs_missing')
    if spec['lane'] == 'none' and names:
        bad.append('unexpected_inputs')
    for n in names:
        if '/' in n or n.startswith('.') or Path(n).suffix.lstrip('.').lower() not in EXT_OK:
            bad.append(f'bad_input_name:{n}')
        elif n not in prob['task_prompt']:
            bad.append(f'input_not_named_in_prompt:{n}')
    if len(set(names)) != len(names):
        bad.append('duplicate_input_names')
    if re.search(r'/home/user/outputs', prob['task_prompt']):
        pass   # harmless: the harness names the same directory
    return bad


async def one(ep: Endpoint, gate: Gate, spec: dict, doc: dict | None, effort: str, out, lock):
    user = P.user_prompt(spec, doc)
    msgs = [{'role': 'system', 'content': P.SYSTEM}, {'role': 'user', 'content': user}]
    extra = dict(ep.extra)
    extra['reasoning_effort'] = effort
    if isinstance(extra.get('chat_template_kwargs'), dict) and 'reasoning_effort' in extra['chat_template_kwargs']:
        extra['chat_template_kwargs'] = {**extra['chat_template_kwargs'], 'reasoning_effort': effort}
    rec = {'idx': spec['idx'], 'spec': spec, 'doc': {k: v for k, v in (doc or {}).items() if k != 'text'} or None,
           'effort': effort, 'started': time.time()}
    async with gate:
        for outer in range(8):
            try:
                content, reasoning, usage, _ = await ep.chat_tools(msgs, None, extra=extra,
                                                                   affinity=f'gen-{spec["idx"]}')
                ep.release(f'gen-{spec["idx"]}')
                break
            except Exception as e:  # noqa: BLE001  (server restart, overload: wait for a healthy endpoint)
                rec.setdefault('errors', []).append(f'{type(e).__name__}: {str(e)[:200]}')
                if 'BadRequest' in type(e).__name__ or '400' in str(e)[:20]:
                    content, reasoning, usage = '', '', {'error': str(e)[:300]}
                    break
                await asyncio.sleep(60)
                await wait_healthy(ep.name)
                ep.refresh()
        else:
            content, reasoning, usage = '', '', {'error': 'gave up'}
    rec.update(usage=usage, seconds=round(time.time() - rec['started']), reasoning_chars=len(reasoning),
               raw=content)
    try:
        prob = P.parse(content)
        rec['problem'] = prob
        rec['static_bad'] = static_check(spec, prob)
    except ValueError as e:
        rec['parse_error'] = str(e)
    async with lock:
        out.write(json.dumps(rec) + '\n')
        out.flush()
    return rec


def done_indices(path: Path) -> set:
    if not path.exists():
        return set()
    s = set()
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get('usage', {}).get('error') is None:
            s.add(r['idx'])
    return s


async def main_async(a):
    await wait_healthy(a.endpoint)
    ep = Endpoint(a.endpoint, max_in_flight=max(cap(), 256), timeout=6 * 3600)
    sampler = Sampler()
    docs = None
    if a.docs:
        from m5.gdpval.docs import DocPicker
        docs = DocPicker(a.docs)  # noqa (single process)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'gen.jsonl'
    skip = done_indices(path)
    gate, lock = Gate(), asyncio.Lock()
    tasks = []
    with open(path, 'a') as out:
        for i in range(a.start, a.start + a.n):
            if i in skip:
                continue
            spec = sampler.spec(i, a.seed)
            if a.lanes and spec['lane'] not in a.lanes.split(','):
                continue
            doc = None
            if spec['lane'] == 'real':
                if docs is None:
                    continue
                doc = docs.pick(spec)
                if doc is None:
                    continue
            tasks.append(asyncio.create_task(one(ep, gate, spec, doc, a.effort, out, lock)))
        n_ok = 0
        for f in asyncio.as_completed(tasks):
            r = await f
            n_ok += 'problem' in r
            print(time.strftime('%H:%M:%S'), r['idx'], r['spec']['lane'], r['usage'], r.get('parse_error'),
                  r.get('static_bad'), flush=True)
    print('done', len(tasks), 'ok', n_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', default='gdpval-m5-v1')
    ap.add_argument('--endpoint', default='m5-ds')
    ap.add_argument('--effort', default='high')
    ap.add_argument('--docs', default=None)
    ap.add_argument('--lanes', default=None, help='only these lanes (comma list of none,synthetic,real)')
    a = ap.parse_args()
    asyncio.run(main_async(a))


if __name__ == '__main__':
    main()

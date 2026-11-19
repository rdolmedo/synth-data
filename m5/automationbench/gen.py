"""Generation driver (#359): specs -> DeepSeek (high effort, no output cap, >= 6 h timeout) -> build_task() ->
assembled upstream task -> problem-side checks, with up to 2 repair turns in the same conversation (same DP rank).

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.gen --specs SPECS.jsonl --out DIR \
      [--limit N] [--workers 2] [--cap N]

Writes DIR/accepted.jsonl (one record per accepted task: spec, TASK_META, the upstream-shaped task, warnings, usage)
and DIR/rejected.jsonl (spec id, the last errors), appending; a rerun skips spec ids already in either file.
Run it as a CPU job (exec and checks are CPU work); it is light on the submit host only for a handful of specs.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import json
import time
from pathlib import Path

from m5.automationbench import prompt as P

MAX_REPAIRS = 2


def cpu_stage(spec: dict, reply: str) -> dict:
    """In a worker process: parse, exec, assemble, check."""
    from m5.automationbench import check as C
    meta, code = P.parse(reply)
    if code is None:
        return {'errors': ['no ```python block with build_task() found in your answer'], 'meta': meta}
    out, err = C.run_code(code)
    if err:
        return {'errors': [err], 'meta': meta, 'code': code}
    spec = {**spec, 'slug': _slug(meta, spec)}
    try:
        task = C.assemble(spec, out)
    except Exception as e:  # noqa: BLE001
        return {'errors': [f'the state could not be prepared: {type(e).__name__}: {str(e)[:1000]}'], 'meta': meta,
                'code': code}
    errors, warns = C.checks(spec, out, task)
    return {'errors': errors, 'warnings': warns, 'meta': meta, 'code': code, 'task': task if not errors else None,
            'own_state': out['initial_state'] if not errors else None}


def _slug(meta, spec):
    """Neutral task names: the generator's slug can name the traps (seen: renewal_triage_lookalike_exclusion), and
    the task name reaches the sandbox's paths, so it stays in TASK_META only."""
    return f'm5_{spec["id"]}'


async def one(ds, pool, spec, out_dir, lock, stats):
    key = spec['id']
    messages = [{'role': 'user', 'content': P.build(spec)}]
    usage = []
    t0 = time.time()
    res = None
    for rnd in range(MAX_REPAIRS + 1):
        try:
            content, reasoning, u = await ds.chat(messages, key)
        except Exception as e:  # noqa: BLE001
            res = {'errors': [f'generator request failed: {e}']}
            break
        usage.append(u)
        res = await asyncio.get_running_loop().run_in_executor(pool, cpu_stage, spec, content)
        res['round'] = rnd
        if not res['errors']:
            break
        messages += [{'role': 'assistant', 'content': content}, {'role': 'user', 'content': P.repair(res['errors'])}]
    ds.release(key)
    rec = {'spec_id': spec['id'], 'spec': spec, 'usage': usage, 'sec': round(time.time() - t0, 1),
           'rounds': (res or {}).get('round', 0) + 1, 'meta': (res or {}).get('meta')}
    async with lock:
        if res and not res['errors']:
            rec.update(task=res['task'], own_state=res['own_state'], warnings=res['warnings'], code=res['code'])
            with (out_dir / 'accepted.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            stats['ok'] += 1
        else:
            rec.update(errors=(res or {}).get('errors'), code=(res or {}).get('code'))
            with (out_dir / 'rejected.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            stats['bad'] += 1
    print(time.strftime('%H:%M:%S'), 'ok' if res and not res['errors'] else 'REJECT', spec['id'], spec['domain'],
          f'rounds={rec["rounds"]} sec={rec["sec"]} tok={sum(x["completion"] for x in usage)}',
          f'[{stats["ok"]} ok / {stats["bad"]} rejected]', flush=True)


async def main_async(a):
    from m5.automationbench.client import DS, Cap
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for n in ('accepted.jsonl', 'rejected.jsonl'):
        f = out_dir / n
        if f.exists():
            done |= {json.loads(x)['spec_id'] for x in f.read_text().splitlines() if x.strip()}
    specs = [json.loads(x) for x in Path(a.specs).read_text().splitlines() if x.strip()]
    specs = [s for s in specs if s['id'] not in done][:a.limit or None]
    print(f'{len(specs)} specs to run ({len(done)} done)', flush=True)
    ds = DS(a.serving, cap=Cap('ds', a.cap), effort=None if a.serving != 'm5-ds' else 'high')
    await ds.ready()
    lock = asyncio.Lock()
    stats = {'ok': 0, 'bad': 0}
    window = max(a.cap or 0, 400)   # specs in flight in the driver; the Cap bounds requests to the server
    sem = asyncio.Semaphore(window)
    with cf.ProcessPoolExecutor(a.workers) as pool:
        async def guarded(s):
            async with sem:
                await one(ds, pool, s, out_dir, lock, stats)
        await asyncio.gather(*(guarded(s) for s in specs))
    print('done', stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cap', type=int, default=None, help='fixed in-flight cap (default: alloc.json, live)')
    ap.add_argument('--serving', default='m5-ds', help='serving name (plumbing tests only: another endpoint)')
    asyncio.run(main_async(ap.parse_args()))


if __name__ == '__main__':
    main()

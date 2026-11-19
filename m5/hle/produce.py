"""Production: specs in batches -> seeds -> DeepSeek problems -> Super traces, all streaming.

Three CPU jobs, each restartable (they skip work already on disk):

    # 1. seeder: batches of specs (one RNG seed per batch), knowledge seeds attached; stops at --batches or STOP
    python -m m5.hle.produce seed --root R --batches 9 --batch-size 2000
    # 2. generator: follows R/specs/b*.jsonl and writes R/problems/b*.jsonl (DeepSeek, cap from alloc.json)
    python -m m5.hle.produce gen --root R
    # 3. traces: m5.hle.traces --problems 'R/problems/b*.jsonl' --out R/traces/traces.jsonl --follow \
    #        --target-tokens 330e6 --cut-tails

Done markers: R/specs/SEED.DONE when the seeder finishes; R/problems/b*.jsonl.DONE per batch (the trace driver's
follow mode ends when every problem file is DONE); R/STOP stops the seeder and the generator from starting new work.
The doc-use counts (R/seeds_used.json) carry across batches so no excerpt repeats and books are used at most 3 times.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from .safeio import append_line, verify_jsonl, write_atomic


def reseed(a, b):
    """Knowledge specs batch b left without an excerpt (seed_missing), re-seeded with the current fallbacks into
    batch 100 + b (the generator follows every b*.jsonl; batch b's own file is never rewritten)."""
    from . import seeds
    root = Path(a.root)
    src, out = root / 'specs' / f'b{b:03d}.jsonl', root / 'specs' / f'b{100 + b:03d}.jsonl'
    if out.exists() or not src.exists():
        return
    missing = [l for l in src.read_text().splitlines() if l.strip() and json.loads(l).get('seed_missing')]
    raw, tmp = root / 'specs' / f'.b{100 + b:03d}.raw.jsonl', root / 'specs' / f'.b{100 + b:03d}.tmp'
    write_atomic(raw, ''.join(l + '\n' for l in missing), check=lambda p: verify_jsonl(p, len(missing)))
    t0 = time.time()
    seeds.attach(raw, tmp, a.blocklist, root / 'seeds_used.json', a.local_copy, seed=100 + b)
    if not verify_jsonl(tmp, len(missing)):
        raise SystemExit(f'reseed {b}: {tmp} failed verification')
    tmp.rename(out)
    raw.unlink()
    print(f'reseed batch {b}: {len(missing)} specs -> {out} ({time.time() - t0:.0f}s)', flush=True)


def seed_batches(a):
    from . import gen, seeds
    root = Path(a.root)
    (root / 'specs').mkdir(parents=True, exist_ok=True)
    for b in range(a.reseed_upto):   # batches seeded before the fallbacks existed
        if not (root / 'STOP').exists():
            reseed(a, b)
    for b in range(a.start, a.batches):
        if (root / 'STOP').exists():
            break
        out = root / 'specs' / f'b{b:03d}.jsonl'
        if out.exists():
            continue
        raw = root / 'specs' / f'.b{b:03d}.raw.jsonl'
        specs = gen.make_specs(a.batch_size, seed=1000 + b, prefix=f'b{b:03d}-')
        write_atomic(raw, ''.join(json.dumps(sp, ensure_ascii=False) + '\n' for sp in specs),
                     check=lambda p: verify_jsonl(p, len(specs)))
        tmp = root / 'specs' / f'.b{b:03d}.tmp'
        t0 = time.time()
        seeds.attach(raw, tmp, a.blocklist, root / 'seeds_used.json', a.local_copy, seed=b)
        if not verify_jsonl(tmp, len(specs)):
            raise SystemExit(f'batch {b}: {tmp} failed verification')
        tmp.rename(out)
        raw.unlink()
        print(f'batch {b}: {out} ({time.time() - t0:.0f}s)', flush=True)
    write_atomic(root / 'specs' / 'SEED.DONE', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    print('SEED DONE', flush=True)


async def gen_follow(a):
    """Generate every spec batch as it appears, in order, keeping the DeepSeek cap busy across batch boundaries."""
    from . import gen as genmod
    from .gen import GEN_SETTINGS, build_prompt, checks, ngrams, parse
    genmod.COMPOSE = not a.no_compose
    from .llm import Cap, Pool, fields, metrics_snapshot
    root = Path(a.root)
    (root / 'problems').mkdir(parents=True, exist_ok=True)
    srv = Pool(pinned=a.server)
    cap = Cap(('ds', 'max_inflight', 'hle'), 40, override=a.concurrency)
    stats = {'ok': 0, 'skip': 0, 'fail': 0, 'err': 0}
    locks = {}
    pending = {}   # batch -> outstanding count
    t_start = time.time()

    async def one(b, spec, out):
        async with cap:
            if (root / 'STOP').exists():
                pending[b] -= 1
                return
            body = {'messages': [{'role': 'user', 'content': build_prompt(spec)}], **GEN_SETTINGS}
            t0 = time.time()
            try:
                data, meta = await srv.chat(spec['id'], body)
            except Exception as e:
                stats['err'] += 1
                print(f'{spec["id"]}: {e}', flush=True)
                pending[b] -= 1
                return
        content, reasoning, finish, usage = fields(data)
        prob = parse(content)
        row = {'id': spec['id'], 'spec': spec, **prob, 'finish': finish,
               'usage': {'prompt': usage.get('prompt_tokens'), 'completion': usage.get('completion_tokens')},
               'gen_seconds': round(time.time() - t0), 'reasoning_chars': len(reasoning), 'server': meta,
               'finished_at': time.time()}
        if 'skip' in prob:
            row['status'] = 'skip'
        else:
            row['checks'] = checks(spec, prob)
            row['status'] = 'ok' if not row['checks'] else 'fail'
        stats[row['status']] += 1
        if spec.get('seed'):
            g = ngrams(prob.get('question') or '')
            row['seed_8gram_share'] = round(len(g & ngrams(spec['seed']['text'])) / max(1, len(g)), 4)
            row['spec'] = {**spec, 'seed': {k: v for k, v in spec['seed'].items() if k != 'text'}}
        async with locks[b]:
            await asyncio.to_thread(append_line, out, json.dumps(row, ensure_ascii=False))
        pending[b] -= 1
        n = sum(stats.values())
        if n % 100 == 0:
            print(f'{time.strftime("%H:%M:%S")} {n} done {stats} inflight {cap.inflight} cap {cap.current()} '
                  f'{time.time() - t_start:.0f}s', flush=True)

    async def snapshots():
        while True:
            for name in srv.serving():
                snap = await metrics_snapshot(name)
                if snap:
                    snap['my_inflight'] = cap.inflight
                    await asyncio.to_thread(append_line, root / 'problems' / 'gen.metrics.jsonl', json.dumps(snap))
            await asyncio.sleep(300)

    snap_task = asyncio.create_task(snapshots())
    tasks, started = set(), set()
    while True:
        for sf in sorted((root / 'specs').glob('b*.jsonl')):
            b = sf.stem
            if b in started:
                continue
            out = root / 'problems' / f'{b}.jsonl'
            done = set()
            if out.exists():
                for l in out.read_text().splitlines():
                    try:
                        done.add(json.loads(l)['id'])
                    except (ValueError, KeyError):
                        pass
            specs = [json.loads(l) for l in sf.read_text().splitlines() if l.strip()]
            specs = [s for s in specs if s['id'] not in done and not s.get('seed_missing')]
            started.add(b)
            locks[b] = asyncio.Lock()
            pending[b] = len(specs)
            print(f'{b}: {len(specs)} specs to generate ({len(done)} done)', flush=True)
            for s in specs:
                t = asyncio.create_task(one(b, s, out))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
        for b in list(pending):
            if pending[b] == 0:
                out = root / 'problems' / f'{b}.jsonl'
                out.touch()
                write_atomic(str(out) + '.DONE', json.dumps(stats) + '\n')
                del pending[b]
                print(f'{b}: DONE', flush=True)
        seeding_done = (root / 'specs' / 'SEED.DONE').exists() or (root / 'STOP').exists()
        if seeding_done and not tasks and not pending:
            break
        await asyncio.sleep(20)
    snap_task.cancel()
    await srv.close()
    write_atomic(root / 'problems' / 'GEN.DONE', json.dumps(stats) + '\n')
    print('GEN DONE', stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('seed')
    s.add_argument('--root', required=True)
    s.add_argument('--batches', type=int, required=True)
    s.add_argument('--start', type=int, default=0)
    s.add_argument('--batch-size', type=int, default=2000)
    s.add_argument('--reseed-upto', type=int, default=0, help='re-seed seed_missing specs of batches < N first')
    s.add_argument('--blocklist', default='/data/aai/m5/hle/private/blocklist.json')
    s.add_argument('--local-copy', default=os.environ.get('_CONDOR_SCRATCH_DIR'))
    g = sub.add_parser('gen')
    g.add_argument('--root', required=True)
    g.add_argument('--server', help='pin one DeepSeek server (default: follow alloc.json ds.serving)')
    g.add_argument('--concurrency', type=int)
    g.add_argument('--no-compose', action='store_true', help='single-technique reasoning prompts (v2)')
    a = ap.parse_args()
    if a.cmd == 'seed':
        seed_batches(a)
    else:
        asyncio.run(gen_follow(a))


if __name__ == '__main__':
    main()

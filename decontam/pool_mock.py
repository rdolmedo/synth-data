"""End-to-end test of decontam.pool without GPUs (issue #119): a mock OpenAI-compatible judge with DP ranks, a stream
that grows chunk by chunk, and `pool run --follow --snapshot-when STOP` until END: chunks are only read until STOP,
the main pass runs then (a chunk streamed during it is read during it, #234), more chunks are read without a pass, and
the last pass at END judges the new window pairs.

  python -m decontam.pool_mock --work W --source STREAM [--chunks 4]

The mock answers REUSE for a deterministic 0.2% of pairs (WEAK_REUSE for another 0.2%), so resampling, exclusions and
window iterations all run. Checks: every test task stayed on one DP rank (per judge process); exactly two snapshots
(STOP, END); the published list equals the pairs the mock flagged (any REUSE / all WEAK_REUSE); the hold is released.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from aiohttp import web

SEEN = collections.defaultdict(set)   # (judge process, test task) -> DP ranks that served it
NAME = re.compile(r'## TEST TASK \([^)]*\): `([^`]+)`')
CAND = re.compile(r'## CANDIDATE TASK[^`]*`([^`]+)`')
DELAY = [0.01]   # seconds per judge reply (--judge-delay)


def label(tb, cf, sample_text):
    h = int(hashlib.sha1(f'{tb}|{cf}'.encode()).hexdigest(), 16) % 1000
    return 'REUSE' if h < 2 else 'WEAK_REUSE' if h < 4 else 'NO_REUSE'


async def chat(request):
    body = await request.json()
    user = body['messages'][1]['content']
    tb, cf = NAME.search(user).group(1), CAND.search(user).group(1)
    SEEN[(request.headers.get('X-Decontam-Judge'), tb)].add(request.headers.get('X-data-parallel-rank'))
    await asyncio.sleep(DELAY[0])
    verdict = {'label': label(tb, cf, user), 'reason': 'mock', 'matches': []}
    return web.json_response({'choices': [{'message': {'content': json.dumps(verdict), 'reasoning_content': ''},
                                           'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': len(user) // 4, 'completion_tokens': 20}})


async def models(request):
    return web.json_response({'data': [{'id': 'mock-judge'}]})


def serve(port):
    app = web.Application(client_max_size=64 << 20)
    app.add_routes([web.post('/v1/chat/completions', chat), web.get('/v1/models', models)])
    web.run_app(app, host='127.0.0.1', port=port, print=None, handle_signals=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', required=True, help='a real stream to copy chunks from')
    ap.add_argument('--chunks', type=int, default=4)
    ap.add_argument('--port', type=int, default=18931)
    ap.add_argument('--judge-delay', type=float, default=1.0, help='seconds per mock reply: the main pass must last '
                                                                    'longer than a few --interval (20 s) polls')
    a = ap.parse_args()
    DELAY[0] = a.judge_delay
    root = Path(a.work)
    shutil.rmtree(root, ignore_errors=True)
    serving, stream, work, pub = root / 'serving' / 'mock', root / 'stream', root / 'work', root / 'published'
    (stream / 'chunks').mkdir(parents=True)
    serving.mkdir(parents=True)
    threading.Thread(target=serve, args=(a.port,), daemon=True).start()
    time.sleep(2)

    def heartbeat():
        while True:
            (serving / 'launch.json').write_text(json.dumps({'argv': ['vllm', 'serve', '--data-parallel-size', '2']}))
            tmp = serving / 'endpoints.json.partial'
            tmp.write_text(json.dumps({'served_name': 'mock-judge', 'urls': [f'http://127.0.0.1:{a.port}/v1'],
                                       'healthy_at': time.time(), 'updated_at': time.time()}))
            os.replace(tmp, serving / 'endpoints.json')
            time.sleep(20)
    threading.Thread(target=heartbeat, daemon=True).start()
    src = Path(a.source)
    sizes = []

    def add(i):
        rec = json.loads((src / 'chunks' / f'{i:04d}.json').read_text())
        os.symlink(src / rec['tar'], stream / 'chunks' / f'{i:04d}.tar')
        rec['tar'] = f'chunks/{i:04d}.tar'
        (stream / 'chunks' / f'{i:04d}.json').write_text(json.dumps(rec))
        sizes.append(rec['count'])
    add(0)
    # the wrapper runs decontam.pool; the mock registry only through TASKSYNTH_SERVING_ROOT
    env = {k: v for k, v in os.environ.items() if k not in ('DECONTAM_MODULE', 'DECONTAM_SERVING_ROOT')}
    env['TASKSYNTH_SERVING_ROOT'] = str(root / 'serving')
    here = Path(__file__).resolve().parent
    stop = root / 'round' / 'STOP'
    stop.parent.mkdir()
    cmd = [str(here / 'pool_job.sh'), 'run', '--work', str(work), '--pool', str(stream), '--serving', 'mock',
           '--follow', '--interval', '20', '--snapshot-when', str(stop), '--publish', str(pub), '--workers', '8',
           '--hold', str(root / 'decontam.HOLD'), '--hold-rate', '1', '--concurrency', '64', '--concurrency-small', '16', '--k', '10',
           '--k-lite', '5']
    proc = subprocess.Popen(cmd, env=env)

    def wait(cond, what, limit=3600):
        t0 = time.time()
        while not cond():
            if proc.poll() is not None:
                sys.exit(f'run exited early ({proc.returncode}) waiting for {what}')
            if time.time() - t0 > limit:
                sys.exit(f'timeout waiting for {what}')
            time.sleep(5)
        print(f'{what}: {time.time() - t0:.0f}s', flush=True)

    def read_units():
        try:
            return len(json.loads((work / 'pool.json').read_text())['units'])
        except (OSError, ValueError, KeyError):
            return 0
    latest = lambda: json.loads((work / 'lists' / 'LATEST.json').read_text()) if (work / 'lists' / 'LATEST.json').exists() else {}
    half = max(1, a.chunks // 2)
    for i in range(1, half):
        add(i)
    wait(lambda: read_units() == half, f'{half} chunks read')
    no_early = not (work / 'lists').exists()            # reading only: no judge pass before STOP
    stop.touch()                                         # the round's token budget is covered
    main_n = sum(sizes)
    wait(lambda: (work / 'passes').exists(), 'main pass started')
    add(half)                                            # a chunk streamed while the main pass judges (#234 R2)
    wait(lambda: latest().get('tasks') == main_n, 'main pass')
    main_pairs = latest()['counts']['pairs_judged']
    events = (work / 'events.log').read_text()
    read_during_pass = 'update during the pass' in events
    for i in range(half + 1, a.chunks):
        add(i)
    wait(lambda: read_units() == a.chunks, f'{a.chunks} chunks read')
    no_extra = latest().get('tasks') == main_n               # still only the main pass
    (stream / 'END.json').write_text(json.dumps({'chunks': a.chunks}))
    rc = proc.wait()
    lists = sorted(p.name for p in (work / 'lists').iterdir() if p.is_dir())
    final = json.loads((pub / 'decontamination-variants.json').read_text())
    # expected: candidates with a pair the mock flags, among pairs that were judged
    pairs = [json.loads(l) for l in open(pub / 'decontamination-variants.pairs.jsonl')]
    expect = {p['cf'] for p in pairs if any(label(p['tb'], p['cf'], '') in ('REUSE', 'WEAK_REUSE')
                                            for _ in [0])}
    report = {'exit': rc, 'snapshots': lists, 'tasks': sum(sizes), 'excluded': len(final['excluded']),
              'no_pass_before_stop': no_early, 'no_pass_between_stop_and_end': no_extra,
              'pairs_main_pass': main_pairs, 'pairs_last_pass': final['counts']['pairs_judged'] - main_pairs,
              'hold_released': not (root / 'decontam.HOLD').exists(), 'chunk_read_during_main_pass': read_during_pass,
              'judge_load_files': sorted(p.name for p in (work / 'judge-load').glob('*.json')),
              'expected_from_mock': len(expect), 'list_matches_mock': set(final['excluded']) == expect,
              'test_tasks_on_one_rank': all(len(r) == 1 for r in SEEN.values()),
              'ranks_used': dict(collections.Counter(next(iter(r)) for r in SEEN.values())),
              'judge_processes': len({k[0] for k in SEEN}),
              'pairs_judged': final['counts']['pairs_judged'], 'by_arm': {k: v['pairs_judged'] for k, v in final['arms'].items()}}
    print(json.dumps(report, indent=1))
    ok = (rc == 0 and report['list_matches_mock'] and report['test_tasks_on_one_rank'] and len(lists) == 2 and no_early
          and no_extra and report['hold_released'] and read_during_pass and report['judge_load_files'])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

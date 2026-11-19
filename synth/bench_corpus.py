"""Turn a synth campaign's recorded Pi traffic into a synth.bench corpus.

  python -m synth.bench_corpus <campaign train/raw> <out dir> [--size N] [--seed S] [--min-turns K]

synth.bench replays `<run>/harbor/run/pi-cont16k/{requests.jsonl, api-records/NNNN.request.json,
NNNN.transport1.meta.json}` (the pi-submit layout). Synth campaigns record the same traffic in the sticky proxy's
`harbor/pi-cont16k/wire.jsonl`: one `req` record per request with the exact forwarded body and its start time, and
an `end` record with its completion time. Only completed (HTTP 200) requests are kept, in order, so the replayed tool
gaps (next start - this end) are the recorded ones.
"""
import argparse
import json
import os
from pathlib import Path
import random

from .common import atomic


def convert(attempt, out, min_turns):
    wires = list((Path(attempt) / 'harbor').glob('*/wire.jsonl'))
    if not wires:
        return 0
    reqs, ends = {}, {}
    with open(wires[0]) as f:
        for line in f:
            d = json.loads(line)
            if d['t'] == 'req':
                reqs[d['id']] = d
            elif d['t'] == 'end':
                ends[d['id']] = d
    kept = [i for i in sorted(reqs) if ends.get(i, {}).get('status') == 200 and 'body' in reqs[i]]
    if len(kept) < min_turns:
        return 0
    pi = Path(out) / Path(attempt).name / 'harbor' / 'run' / 'pi-cont16k'
    (pi / 'api-records').mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(kept):
        (pi / 'api-records' / f'{n:04d}.request.json').write_text(json.dumps(reqs[i]['body']))
        (pi / 'api-records' / f'{n:04d}.transport1.meta.json').write_text(json.dumps({'time': ends[i]['time']}))
    (pi / 'requests.jsonl').write_text(''.join(json.dumps({'t': reqs[i]['time']}) + '\n' for i in kept))
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('out')
    ap.add_argument('--size', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-turns', type=int, default=2)
    a = ap.parse_args()
    attempts = sorted(os.listdir(a.raw))
    random.Random(a.seed).shuffle(attempts)
    done = turns = 0
    for name in attempts:
        if done >= a.size:
            break
        n = convert(Path(a.raw) / name, a.out, a.min_turns)
        if n:
            done += 1
            turns += n
    atomic(Path(a.out) / 'corpus.json', {'source': a.raw, 'size': done, 'turns': turns, 'seed': a.seed})
    print(f'{done} trajectories, {turns} requests -> {a.out}')


if __name__ == '__main__':
    main()

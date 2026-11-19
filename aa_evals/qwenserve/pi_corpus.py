#!/usr/bin/env python3
"""Replay corpus from pi_eval runs (M3.3, #176): each attempt's packed trial (RUN/attempts/<key>/trial.tar.gz) ->
<out>/<run>__<key>/raw/harbor/pi-cont16k/{wire-requests.jsonl, wire-responses/*} (mtimes kept), the layout
synth.bench.index_wire reads (the GLM pi-submit corpus layout). A tuning campaign then names
corpus = "<out>/*" with closed_loop = true.

  python pi_corpus.py --runs RUN [RUN ...] --out DIR
"""
import argparse
import glob
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out, n = Path(a.out), 0
    for run in a.runs:
        for t in sorted(glob.glob(str(Path(run) / 'attempts' / '*' / 'trial.tar.gz'))):
            key = Path(t).parent.name
            dest = out / f'{Path(run).name}__{key}' / 'raw' / 'harbor' / 'pi-cont16k'
            if (dest / 'wire-requests.jsonl').exists():
                continue
            with tarfile.open(t) as tf:
                for m in tf.getmembers():
                    if not m.isfile() or '/pi-cont16k/' not in m.name:
                        continue
                    rel = m.name.split('/pi-cont16k/', 1)[1]
                    if rel != 'wire-requests.jsonl' and not rel.startswith('wire-responses/'):
                        continue
                    m.name = rel
                    tf.extract(m, dest, filter='data')
            n += (dest / 'wire-requests.jsonl').exists()
    print(f'{n} attempts -> {out}')


if __name__ == '__main__':
    main()

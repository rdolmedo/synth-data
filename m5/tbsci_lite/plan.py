"""Write a round's specs.jsonl (#353): n specs drawn from vocab.json, with real datasets from the shared scidata
cache bound at the vocabulary's real share.

  python -m m5.tbsci_lite.plan --round p1 --n 200 --seed 1 [--real-share 0.25] [--no-real]
"""
import argparse
import collections
import json

from . import scidata, specs
from .common import ROOT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--start', type=int, default=0, help='first spec index (append a later batch to a round)')
    ap.add_argument('--real-share', type=float)
    ap.add_argument('--no-real', action='store_true')
    ap.add_argument('--append', action='store_true')
    ap.add_argument('--ms-boost', type=float, default=1.0, help='multiply P(model-spec | archetype) (specs.sample)')
    ap.add_argument('--minutes', choices=['vocab', 'v1'], default='vocab', help='target mix: vocab.json, or v1 (2-6 h)')
    a = ap.parse_args(argv)
    d = ROOT / 'rounds' / a.round
    path = d / 'specs.jsonl'
    if path.exists() and not a.append:
        raise SystemExit(f'{path} exists (use --append with --start)')
    datasets = {} if a.no_real else scidata.usable()
    prior = collections.Counter()   # datasets bound in earlier rounds go after the unused ones
    for f in (ROOT / 'rounds').glob('*/specs.jsonl'):
        for line in open(f):
            aid = (json.loads(line).get('asset') or {}).get('id')
            if aid:
                prior[aid] += 1
    out = specs.sample(a.n, a.seed, datasets=datasets, real_share=0.0 if a.no_real else a.real_share, start=a.start,
                       minutes=specs.MINUTES_V1 if a.minutes == 'v1' else None, prior_uses=prior, ms_boost=a.ms_boost)
    d.mkdir(parents=True, exist_ok=True)
    with open(path, 'a' if a.append else 'w') as f:
        for s in out:
            f.write(json.dumps(s) + '\n')
    c = lambda k: collections.Counter(s.get(k) for s in out).most_common()  # noqa: E731
    print(json.dumps({'round': a.round, 'n': len(out), 'datasets_available': len(datasets),
                      'sci_domain': c('sci_domain'), 'field': c('domain'), 'archetype': c('agent_action'),
                      'modality': c('modality'), 'source': c('source'), 'minutes': c('minutes'),
                      'open_subtopic': sum(1 for s in out if not s.get('subtopic'))}, indent=1))


if __name__ == '__main__':
    main()

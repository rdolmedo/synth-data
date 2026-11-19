"""Does the fingerprint source matter? Judge decisions on the Opus-labelled pairs under three configurations.

  python -m decontam.compare_fp --work W --variants V [--out FILE]

The 262 pairs (round-1 and round-2 audits, calibration set) were judged fresh in round 5b with rubric_reuse.md and Opus
fingerprints (opus), rubric_reuse.md and DeepSeek fingerprints (ds), rubric_reuse_nofp.md (nofp): 2 samples, 8 for
pairs with a non-NO label. A pair is excluded if any sample says REUSE or all say WEAK_REUSE (the list rule). Reference:
the audits' Opus label (REUSE = variant); the calibration set's positive / negative (borderline: its lean). Pairs of
the four TB2.1 tasks the fingerprint prompt uses as examples are reported separately (their DeepSeek fingerprints
had help).
"""
import argparse
import collections
import json
from pathlib import Path

from . import apply as A

EXAMPLES = {'build-pmars', 'video-processing', 'count-dataset-tokens', 'fix-code-vulnerability'}
CALIBRATION = '/data/aai/tbenchtrain/decontam_calibforge/reuse/calibration.json'


def reference(work, variants):
    """{(tb, cf): (source, is_variant)}: audit labels first, then the calibration set."""
    ref = {}
    for key, out in (('audit_key.json', 'audit'), ('audit_key_r2.json', 'audit2')):
        ids = {k['id']: (k['tb'], k['cf']) for k in json.loads((Path(work) / key).read_text())['pairs']}
        for path in sorted((Path(variants) / out).glob('out_*.jsonl')):
            for r in map(json.loads, open(path)):
                if r.get('id') in ids and r.get('label'):
                    ref[ids[r['id']]] = (out, r['label'] == 'REUSE')
    cal = json.loads(Path(CALIBRATION).read_text())
    for kind in ('positive', 'negative', 'borderline'):
        for e in cal[kind]:
            if (e[0], e[1]) not in ref:
                ref[(e[0], e[1])] = ('calibration', kind == 'positive' or (kind == 'borderline' and e[-1] == 'exclude'))
    return ref


def excluded(labels):
    return 'REUSE' in labels or (bool(labels) and all(l == 'WEAK_REUSE' for l in labels))


def score(rows):
    tp = sum(r['ref'] and r['pred'] for r in rows)
    fp = sum(not r['ref'] and r['pred'] for r in rows)
    fn = sum(r['ref'] and not r['pred'] for r in rows)
    return {'pairs': len(rows), 'variants': tp + fn, 'excluded': tp + fp, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / (tp + fp), 3) if tp + fp else None,
            'recall': round(tp / (tp + fn), 3) if tp + fn else None,
            'agreement': round(sum(r['ref'] == r['pred'] for r in rows) / len(rows), 3) if rows else None}


def main(a):
    ref = reference(a.work, a.variants)
    cfgs = {c: {p: [s['verdict']['label'] for s in ss] for p, ss in
                A.read_verdicts([str(Path(a.work) / f'r5-val-{c}-verdicts.jsonl')]).items()} for c in ('opus', 'ds', 'nofp')}
    common = set(ref) & set.intersection(*(set(v) for v in cfgs.values()))
    report = {'pairs_with_reference': len(ref), 'pairs_judged_all_three': len(common)}
    for subset, keep in (('all', lambda p: True), ('without_example_tasks', lambda p: p[0] not in EXAMPLES),
                         ('example_tasks', lambda p: p[0] in EXAMPLES)):
        for src in ('all', 'audit', 'audit2', 'calibration'):
            ps = [p for p in sorted(common) if keep(p) and (src == 'all' or ref[p][0] == src)]
            report[f'{subset}/{src}'] = {c: score([{'ref': ref[p][1], 'pred': excluded(cfgs[c][p])} for p in ps])
                                         for c in cfgs}
    flips = collections.Counter()
    for p in common:
        o, d = excluded(cfgs['opus'][p]), excluded(cfgs['ds'][p])
        if o != d:
            flips[('opus only' if o else 'ds only', 'variant' if ref[p][1] else 'not')] += 1
    report['opus_vs_ds_disagreements'] = {f'{k[0]} ({k[1]})': v for k, v in sorted(flips.items())}
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--variants', default='/data/aai/tbenchtrain/decontam_calibforge/variants')
    ap.add_argument('--out')
    main(ap.parse_args())

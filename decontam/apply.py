"""Steps 5-6: propagate exclusions to CalibForge near-copies, self-validate, write the exclusion list.

  python -m decontam.apply near-copies --work W --verdicts V[,V2] --out W/propagate-N.jsonl
      CalibForge near-copies of every judge-excluded task, paired with the same TB2.1 task, minus pairs already
      judged: judge them next (judge.py --pairs), then pass their verdicts here again until none are new.
  python -m decontam.apply final --work W --verdicts V[,V2,...] --calibration C --out DIR
      Pair verdicts, calibration, agreement, exclusions by source (judge / propagated / canary / lexical), sanity
      cap; writes DIR/decontamination-rewrites.json and DIR/decontamination-rewrites.pairs.jsonl.

A pair is excluded if any of its samples says DUPLICATE or REWRITE. Near-copy (seeds/METHODOLOGY.md 6.6): shingle
containment >= 0.5 in tests or instruction (shingles in more than 5 tasks removed), or bge-small instruction cosine
>= 0.93. CalibForge has no reference solutions, so the solution field never fires.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from . import tasks as T

EXCLUDE = ('DUPLICATE', 'REWRITE')
NEAR_CONTAIN, NEAR_COSINE, NEAR_DF, MIN_SHINGLES = 0.5, 0.93, 5, 25
CAP_WARN, CAP_HARD = 0.10, 0.15


def read_verdicts(paths):
    """{(tb, cf): [sample rows]} from judge outputs (later files win per (tb, cf, sample))."""
    rows = {}
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:   # a line cut short by a killed judge
                continue
            if r.get('verdict'):
                rows[(r['tb'], r['cf'], r['sample'])] = r
    pairs = collections.defaultdict(list)
    for (tb, cf, _), r in sorted(rows.items()):
        pairs[(tb, cf)].append(r)
    return pairs


def pair_verdict(samples):
    labels = [s['verdict']['label'] for s in samples]
    hit = [s for s in samples if s['verdict']['label'] in EXCLUDE]
    best = hit[0] if hit else samples[0]
    return {'labels': labels, 'excluded': bool(hit), 'label': best['verdict']['label'],
            'reason': best['verdict'].get('reason', ''), 'copied_elements': best['verdict'].get('copied_elements', []),
            'agree': len(set(labels)) == 1, 'agree_binary': len({l in EXCLUDE for l in labels}) == 1,
            'truncated': bool(samples[0].get('tb_truncated') or samples[0].get('cf_truncated'))}


def near_copies(work, names):
    """{excluded CF task: [near-copy CF tasks]} under the 6.6 rule."""
    F = pickle.loads((Path(work) / 'features.pkl').read_bytes())
    cf = F['cf']
    idx = {c['name']: j for j, c in enumerate(cf)}
    E = np.load(Path(work) / 'cf-bge-small.npy')
    fields = ('tests', 'instruction')
    df = {f: collections.Counter(x for c in cf for x in c['shingles'][f]) for f in fields}
    sets = [{f: {x for x in c['shingles'][f] if df[f][x] <= NEAR_DF} for f in fields} for c in cf]
    index = {f: collections.defaultdict(list) for f in fields}
    for j, s in enumerate(sets):
        for f in fields:
            for x in s[f]:
                index[f][x].append(j)
    out = {}
    for name in names:
        i = idx[name]
        found = set(np.nonzero(E @ E[i] >= NEAR_COSINE)[0].tolist())
        for f in fields:
            mine = sets[i][f]
            if len(mine) < MIN_SHINGLES:
                continue
            hits = collections.Counter(j for x in mine for j in index[f][x])
            found |= {j for j, n in hits.items() if len(sets[j][f]) >= MIN_SHINGLES and n / min(len(mine), len(sets[j][f])) >= NEAR_CONTAIN}
        out[name] = sorted(cf[j]['name'] for j in found if j != i)
    return out


def cmd_near_copies(a):
    pairs = read_verdicts(a.verdicts.split(','))
    excluded = collections.defaultdict(set)
    for (tb, cf), samples in pairs.items():
        if pair_verdict(samples)['excluded']:
            excluded[cf].add(tb)
    tb_names = set(json.loads((Path(a.work) / 'retrieval.json').read_text()).get('tb_names', ())) or {tb for tb, _ in pairs}
    for cf, hits in json.loads(Path(T.CF_LEXICAL).read_text())['flagged'].items():  # lexical flags vs a TB2.1 task
        excluded[cf] |= {h['reference'] for h in hits if h['reference'] in tb_names}
    excluded = {cf: tbs for cf, tbs in excluded.items() if tbs}
    copies = near_copies(a.work, sorted(excluded))
    todo = sorted({(tb, y) for x, tbs in excluded.items() for y in copies[x] for tb in tbs} - set(pairs))
    with open(a.out, 'w') as f:
        for tb, cf in todo:
            via = sorted(x for x, tbs in excluded.items() if tb in tbs and cf in copies[x])
            f.write(json.dumps({'tb': tb, 'cf': cf, 'via': via}) + '\n')
    print(f'{len(excluded)} excluded tasks (judge, or lexical vs TB2.1), {sum(map(len, copies.values()))} near-copy links, '
          f'{len(todo)} new pairs to judge -> {a.out}')


def calibration(pairs):
    cal = T.read_calibration()
    out = {}
    for kind, want in (('exclude', True), ('keep', False), ('expected_exclude_unconfirmed', True)):
        rows = []
        for tb, cf in cal[kind]:
            v = pair_verdict(pairs[(tb, cf)]) if (tb, cf) in pairs else None
            rows.append({'tb': tb, 'cf': cf, 'labels': v and v['labels'], 'pass': bool(v) and v['excluded'] == want,
                         'samples_correct': v and sum((l in EXCLUDE) == want for l in v['labels']),
                         'reason': v and v['reason']})
        out[kind] = {'passed': sum(r['pass'] for r in rows), 'of': len(rows),
                     'per_sample_correct': sum(r['samples_correct'] or 0 for r in rows),
                     'samples': sum(len(r['labels'] or []) for r in rows), 'pairs': rows}
    out['pass'] = out['exclude']['passed'] == out['exclude']['of'] and out['keep']['passed'] == out['keep']['of']
    return out


def cmd_final(a):
    work = Path(a.work)
    pairs = read_verdicts(a.verdicts.split(','))
    propagated_pairs = set()
    for path in (a.propagated or '').split(','):
        if path:
            propagated_pairs |= {(r['tb'], r['cf']) for r in map(json.loads, open(path))}
    retrieval = json.loads((work / 'retrieval.json').read_text())
    lexical = json.loads(Path(T.CF_LEXICAL).read_text())['flagged']
    canary = retrieval['cf_canary_tasks']
    cal = calibration(read_verdicts(a.calibration.split(','))) if a.calibration else None
    verdicts = {p: pair_verdict(s) for p, s in pairs.items()}
    excluded = {}

    def add(task, source, **info):
        e = excluded.setdefault(task, {'sources': [], 'tb_tasks': [], 'verdicts': []})
        if source not in e['sources']:
            e['sources'].append(source)
        if info.get('tb') and info['tb'] not in e['tb_tasks']:
            e['tb_tasks'].append(info['tb'])
        if info:
            e['verdicts'].append({'source': source, **info})

    for (tb, cf), v in sorted(verdicts.items()):
        if v['excluded']:
            add(cf, 'propagated' if (tb, cf) in propagated_pairs else 'judge', tb=tb, label=v['label'],
                labels=v['labels'], reason=v['reason'], copied_elements=v['copied_elements'])
    for cf in canary:
        add(cf, 'canary')
    for cf, hits in lexical.items():
        for h in hits:
            add(cf, 'lexical', tb=h['reference'], field=h['field'],
                **({'containment': h['containment']} if 'containment' in h else {}))
    n_cf = retrieval['cf_tasks']
    labels = collections.Counter(l for v in verdicts.values() for l in v['labels'])
    pair_labels = collections.Counter(v['label'] for v in verdicts.values())
    by_source = collections.Counter(s for e in excluded.values() for s in e['sources'])
    only = collections.Counter(e['sources'][0] for e in excluded.values() if len(e['sources']) == 1)
    first = collections.Counter(next(s for s in ('judge', 'propagated', 'canary', 'lexical') if s in e['sources'])
                                for e in excluded.values())
    two = [v for v in verdicts.values() if len(v['labels']) >= 2]
    rate = len(excluded) / n_cf
    report = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'issue': 'internal issue 14',
        'test_set': 'TB2.1 (89 tasks, /data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar)',
        'candidates': f'CalibForge ({n_cf} tasks, {T.CF_TAR})',
        'rule': 'exclude a task if any judge sample labels it DUPLICATE or REWRITE of a TB2.1 task (pair judged by '
                'DeepSeek-V4.1-Flash-1017, thinking on, 2 samples), or it is a judged-positive near-copy of an excluded '
                'task (propagated), or it carries a Terminal-Bench canary GUID (canary), or it was flagged by the '
                'lexical decontamination (lexical, synth.datasets.decontaminate vs 189 references)',
        'judge': a.judge_info and json.loads(Path(a.judge_info).read_text()),
        'rubric_sha256': hashlib.sha256(Path(__file__).with_name('rubric.md').read_bytes()).hexdigest(),
        'counts': {'pairs_judged': len(verdicts), 'samples': sum(len(v['labels']) for v in verdicts.values()),
                   'pairs_from_retrieval': len(verdicts) - len(propagated_pairs & set(verdicts)),
                   'pairs_from_propagation': len(propagated_pairs & set(verdicts)),
                   'sample_labels': dict(labels), 'pair_labels': dict(pair_labels),
                   'pairs_excluded': sum(v['excluded'] for v in verdicts.values()),
                   'pairs_truncated': sum(v['truncated'] for v in verdicts.values()),
                   'excluded_tasks': len(excluded), 'exclusion_rate': round(rate, 4),
                   'excluded_by_source_any': dict(by_source), 'excluded_by_source_only': dict(only),
                   'excluded_by_first_source': dict(first)},
        'agreement': {'pairs_with_2_samples': len(two),
                      'label_agreement': round(sum(v['agree'] for v in two) / max(1, len(two)), 4),
                      'exclude_decision_agreement': round(sum(v['agree_binary'] for v in two) / max(1, len(two)), 4),
                      'excluded_by_one_sample_only': sum(v['excluded'] and not v['agree_binary'] for v in two),
                      'excluded_by_both_samples': sum(v['excluded'] and v['agree_binary'] for v in two)},
        'calibration': cal,
        'sanity_cap': {'rate': round(rate, 4), 'warn_above': CAP_WARN, 'hard_above': CAP_HARD,
                       'status': 'ok' if rate <= CAP_WARN else 'investigate' if rate <= CAP_HARD else 'too-many'},
        'excluded': dict(sorted(excluded.items())),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'decontamination-rewrites.json').write_text(json.dumps(report, indent=1) + '\n')
    with open(out / 'decontamination-rewrites.pairs.jsonl', 'w') as f:
        for (tb, cf), v in sorted(verdicts.items()):
            samples = pairs[(tb, cf)]
            f.write(json.dumps({'tb': tb, 'cf': cf, 'source': 'propagated' if (tb, cf) in propagated_pairs else 'retrieval',
                                **v, 'samples': [{k: s.get(k) for k in ('sample', 'verdict', 'finish_reason', 'usage')}
                                                 for s in samples]}) + '\n')
    print(json.dumps({k: report[k] for k in ('counts', 'agreement', 'sanity_cap')}, indent=1))
    if cal:
        print('calibration pass:', cal['pass'], {k: (cal[k]['passed'], cal[k]['of']) for k in
                                                  ('exclude', 'keep', 'expected_exclude_unconfirmed')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['near-copies', 'final'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--verdicts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--propagated', help='final: comma list of near-copies outputs (their pairs count as propagated)')
    ap.add_argument('--calibration', help='final: comma list of the calibration run verdicts')
    ap.add_argument('--judge-info', help='final: JSON describing the judge server and run')
    a = ap.parse_args()
    cmd_near_copies(a) if a.cmd == 'near-copies' else cmd_final(a)

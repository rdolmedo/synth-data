"""The TB-Sci-lite exclusion list for the SFT mixture (#353): canary scan of the pool and the final list.

  python -m m5.tbsci_lite.decon_final scan  --work W --pool <tars>      lite canary GUIDs/markers in every file of every
        pool task (assets opened) and the content hash of each task copy in each tar (a task traced from several
        chunks must be the same task everywhere) -> W/scan.json
  python -m m5.tbsci_lite.decon_final final --work W --out DIR [--sequences CSV] [--controls controls.json]
        -> DIR/excluded.json {"excluded": [mixture task ids], "report": {...}}, DIR/pairs.jsonl (every judged pair:
        windows that brought it, labels, excluded), DIR/verdicts.jsonl (every sample: label, matches, reason),
        DIR/excluded_pairs.jsonl

The list holds the mixture's tasks (arm tbsci_lite of the SFT sequences, ids without the 'tbsci_lite:' prefix) that
the pool method lists (any REUSE sample or WEAK_REUSE in every sample of a pair, or a canary), plus any task the scan
finds a lite canary in. Positive controls (s900-*) are reported, never listed.
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from .decontam import LITE_CANARIES, patch

SEQUENCES = '/data/aai/m5/ladder/sft/data/m5l-full/sequences.csv'
ARM = 'tbsci_lite'
MARKERS = (re.compile(rb'harbor-canary', re.I), re.compile(rb'BENCHMARK DATA SHOULD NEVER APPEAR', re.I),
           re.compile(rb'terminal-bench-canary', re.I))


def _scan_tar(tar):
    from decontam import pool as PL
    out = {}
    guids = [g.encode() for g in LITE_CANARIES] + [g.encode() for g in PL.T.CANARIES]
    for raw in PL.read_tasks(tar, assets=True):
        h = hashlib.sha256()
        hits = set()
        blobs = [('instruction.md', (raw['instruction'] or '').encode()), ('task.toml', raw['toml'].encode())]
        blobs += [(r, d) for r, _, d in raw['members']] + [('asset:' + p, d) for p, _, d in raw['assets']]
        for rel, data in sorted(blobs, key=lambda x: x[0]):
            h.update(rel.encode() + b'\0' + hashlib.sha256(data).digest())
            for g in guids:
                if g in data:
                    hits.add(f'{rel}: {g.decode()}')
            for m in MARKERS:
                if m.search(data):
                    hits.add(f'{rel}: {m.pattern.decode()}')
        out[raw['name']] = {'sha': h.hexdigest(), 'hits': sorted(hits), 'asset_error': raw['info'].get('asset_error')}
    return tar, out


def cmd_scan(a):
    patch()
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    copies = collections.defaultdict(dict)
    hits, errors = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for tar, res in ex.map(_scan_tar, a.pool):
            for name, r in res.items():
                copies[name][Path(tar).stem] = r['sha']
                if r['hits']:
                    hits.setdefault(name, []).extend(f'{Path(tar).stem}/{x}' for x in r['hits'])
                if r['asset_error']:
                    errors[f'{Path(tar).stem}/{name}'] = r['asset_error']
    differ = {n: v for n, v in copies.items() if len(set(v.values())) > 1}
    rep = {'tars': len(a.pool), 'tasks': len(copies), 'task_copies': sum(len(v) for v in copies.values()),
           'tasks_in_several_tars': sum(len(v) > 1 for v in copies.values()),
           'tasks_whose_copies_differ': differ, 'canary_hits': hits, 'asset_errors': errors,
           'guids': list(LITE_CANARIES), 'markers': [m.pattern.decode() for m in MARKERS],
           'seconds': round(time.time() - t0)}
    Path(a.work, 'scan.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in rep.items()}, indent=1))


def mixture_tasks(path):
    with open(path) as f:
        return {r['group'].split(':', 1)[1] for r in csv.DictReader(f) if r['arm'] == ARM}


def read_samples(path):
    """{(tb, cf): [sample rows with a verdict]}, the number of lines without one."""
    out, none = collections.defaultdict(list), 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('verdict'):
                out[(r['tb'], r['cf'])].append(r)
            else:
                none += 1
    for v in out.values():
        v.sort(key=lambda r: r['sample'])
    return out, none


def cmd_final(a):
    PL = patch()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = json.loads((work / 'lists' / 'LATEST.json').read_text()) if not a.list else {'list': a.list}
    lst = json.loads(Path(latest['list']).read_text())
    pool = PL.Pool(work)
    cands = set(pool.cfn)
    mix = mixture_tasks(a.sequences)
    controls = json.loads(Path(a.controls).read_text()) if a.controls else []
    ctl = {c['name']: c for c in controls}
    scan = json.loads((work / 'scan.json').read_text()) if (work / 'scan.json').exists() else None
    samples, no_verdict = read_samples(work / PL.ARMS['nofp'][0])
    labels = {p: [r['verdict']['label'] for r in v] for p, v in samples.items()}

    via = collections.defaultdict(set)
    first_pass = {}
    for f in sorted(glob.glob(str(work / 'passes' / '*' / 'pairs-nofp.jsonl'))):
        for r in map(json.loads, open(f)):
            via[(r['tb'], r['cf'])] |= set(r.get('via', []))
            first_pass.setdefault((r['tb'], r['cf']), Path(f).parent.name)

    listed = lst['excluded']                          # candidate -> {sources, tb_tasks, verdicts}
    canary = set(scan['canary_hits']) if scan else set()
    flagged = (set(listed) | canary) - set(ctl)
    excluded = sorted(flagged & mix)
    outside = sorted(flagged - mix)

    judged_c = {c for _, c in labels}
    cover_c = {p[1] for p, v in via.items() if p in labels and any(s.startswith('cover_') for s in v)}
    listed_before_cover = {c for c in mix if c in listed}
    mix_in_pool = mix & cands
    coverage = {
        'mixture_tasks': len(mix), 'in_pool': len(mix_in_pool), 'missing_from_pool': sorted(mix - cands),
        'scored_by_retrieval_against_all_141_lite_tasks': len(mix_in_pool),
        'judged_in_at_least_one_pair': len(mix & judged_c),
        'judged_against_nearest_lite_task': len(mix & cover_c),
        'not_judged_but_listed': sorted((mix - judged_c) & set(listed)),
        'not_judged': sorted((mix & cands) - judged_c - set(listed)),
        'pool_tasks': len(cands), 'pool_tasks_outside_mixture': len(cands - mix - set(ctl)),
        'mixture_listed_tasks': len(listed_before_cover)}

    def best(rows):
        return next((r for r in rows if r['verdict']['label'] == 'REUSE'), rows[0])['verdict']

    def why(c):
        e = listed.get(c)
        if not e:
            return {'task': c, 'source': 'canary', 'hits': scan['canary_hits'].get(c) if scan else None}
        v = [x for x in e['verdicts'] if x.get('tb')]
        return {'task': c, 'sources': e['sources'], 'tb_tasks': e['tb_tasks'],
                'pairs': [{'tb': x['tb'], 'labels': x.get('labels'), 'reason': x.get('reason', ''),
                           'matches': x.get('matches', [])} for x in v],
                'canary': scan['canary_hits'].get(c) if scan else None}

    ctl_rep = {}
    for n, c in ctl.items():
        e = listed.get(n)
        pairs = {t: labels[(t, n)] for (t, cc) in labels if cc == n}
        ctl_rep[n] = {'source': c['source'], 'kind': c['kind'], 'in_pool': n in cands,
                      'judged_against_source': labels.get((c['source'], n)),
                      'windows_with_source': sorted(via.get((c['source'], n), [])),
                      'caught': bool(e), 'caught_by_source': bool(e and c['source'] in e['tb_tasks']),
                      'tb_tasks': e['tb_tasks'] if e else [], 'pairs_judged': pairs,
                      'reason': best(samples[(c['source'], n)]).get('reason', '') if (c['source'], n) in samples else None}
    recall = sum(v['caught_by_source'] for v in ctl_rep.values())

    by_test = collections.Counter(t for c in excluded for t in (listed.get(c, {}).get('tb_tasks') or ['(canary)']))
    lab_counts = collections.Counter(l for v in labels.values() for l in v)
    pattern = collections.Counter(','.join(f'{k}x{n}' for k, n in sorted(collections.Counter(v).items()))
                                  for v in labels.values())
    summaries = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(work / 'passes' / '*' / '*.summary.json')))]
    events = (work / 'events.log').read_text().splitlines()
    retrieval = json.loads((work / 'retrieval.json').read_text())
    via_counts = collections.Counter(s for v in via.values() for s in v)
    report = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'issue': 'internal issue 353',
        'test_set': {'name': 'TB-Science-lite v2.1 (all 141 pool tasks)', 'tar': '/data/aai/aa_evals/data/tbsci_lite/v2.1/tasks.tar'},
        'mixture': {'sequences': a.sequences, 'arm': ARM, 'tasks': len(mix)},
        'rule': lst['rule'] + '; also any task carrying a lite canary GUID or benchmark marker in any file (scan.json)',
        'method': {'pool': 'decontam/pool.py (data-synthesis 60f67da) via m5/tbsci_lite/decontam.py',
                   'rubric': 'm5/tbsci_lite/rubric_reuse_lite.md', 'judge': 'DeepSeek V4.1 Flash (m5-ds), effort high',
                   'windows': latest.get('windows'), 'coverage_pairs': 'each candidate vs its nearest lite task by '
                   'embedding and by instr+tests TF-IDF', 'samples': '2 per pair; 6 more when the first two are not both '
                   'NO_REUSE'},
        'counts': {'excluded': len(excluded), 'excluded_rate_of_mixture': round(len(excluded) / max(1, len(mix)), 5),
                   'excluded_pool_tasks_outside_mixture': outside, 'canary_tasks': sorted(canary),
                   'pairs_judged': len(labels), 'judge_samples': sum(map(len, labels.values())),
                   'samples_without_verdict': no_verdict, 'labels': dict(lab_counts),
                   'pair_label_patterns': dict(pattern.most_common()),
                   'pairs_by_window': dict(via_counts.most_common()),
                   'passes': latest.get('passes'), 'excluded_by_test_task': dict(by_test.most_common())},
        'coverage': coverage,
        'positive_controls': {'planted': len(ctl), 'caught_by_source': recall,
                              'recall': round(recall / max(1, len(ctl)), 3), 'controls': ctl_rep},
        'retrieval': {k: retrieval.get(k) for k in ('refs', 'cands', 'duplicates_skipped', 'units', 'boilerplate_df',
                                                    'blob_df', 'forced_pairs', 'tb_named', 'cf_canary_tasks',
                                                    'asset_tasks', 'coverage_pairs')},
        'scan': ({k: scan[k] for k in ('tars', 'tasks', 'task_copies', 'tasks_in_several_tars')} |
                 {'tasks_whose_copies_differ': len(scan['tasks_whose_copies_differ']),
                  'asset_errors': len(scan['asset_errors'])}) if scan else None,
        'judge_seconds': {'sum_of_runs': sum(s.get('seconds', 0) for s in summaries),
                          'runs': len(summaries),
                          'first_pass_start': next((e[:20] for e in events if ' pass 0:' in e), None),
                          'last_event': events[-1][:20] if events else None},
        'list': latest['list'], 'work': str(work),
        'excluded_detail': [why(c) for c in excluded],
    }
    tmp = out / '.excluded.json.partial'
    tmp.write_text(json.dumps({'excluded': excluded, 'report': report}, indent=1))
    os.replace(tmp, out / 'excluded.json')
    hit = {(t, c) for (t, c), v in labels.items() if 'REUSE' in v or (v and all(x == 'WEAK_REUSE' for x in v))}
    with open(out / 'pairs.jsonl', 'w') as f:
        for p in sorted(set(labels) | set(via)):
            f.write(json.dumps({'tb': p[0], 'cf': p[1], 'via': sorted(via.get(p, [])), 'pass': first_pass.get(p),
                                'labels': labels.get(p), 'excluded_pair': p in hit, 'in_mixture': p[1] in mix,
                                'control': p[1] in ctl}) + '\n')
    with open(out / 'verdicts.jsonl', 'w') as f:
        for p, rows in sorted(samples.items()):
            for r in rows:
                f.write(json.dumps({'tb': p[0], 'cf': p[1], 'sample': r['sample'], 'label': r['verdict']['label'],
                                    'matches': r['verdict'].get('matches', []), 'reason': r['verdict'].get('reason', ''),
                                    'finish_reason': r.get('finish_reason'), 'usage': r.get('usage')}) + '\n')
    with open(out / 'excluded_pairs.jsonl', 'w') as f:
        for p in sorted(hit):
            f.write(json.dumps({'tb': p[0], 'cf': p[1], 'labels': labels[p], 'in_mixture': p[1] in mix,
                                'control': p[1] in ctl, 'best': best(samples[p])}) + '\n')
    for name in ('retrieval.json', 'events.log', 'scan.json'):
        if (work / name).exists():
            shutil.copy(work / name, out / name)
    print(json.dumps({'excluded': len(excluded), 'rate': report['counts']['excluded_rate_of_mixture'],
                      'coverage': {k: v for k, v in coverage.items() if not isinstance(v, list)},
                      'controls': f'{recall}/{len(ctl)}', 'pairs': len(labels),
                      'samples': report['counts']['judge_samples'], 'by_test': dict(by_test.most_common(10))}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['scan', 'final'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--pool', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out')
    ap.add_argument('--sequences', default=SEQUENCES)
    ap.add_argument('--controls')
    ap.add_argument('--list', help='final: a decontamination-variants.json other than lists/LATEST.json (tests)')
    a = ap.parse_args()
    {'scan': cmd_scan, 'final': cmd_final}[a.cmd](a)


if __name__ == '__main__':
    main()

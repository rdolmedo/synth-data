"""Variants pass (issue #14, criterion agreed 2026-10-29): is a CalibForge task a variant of one specific TB2.1 task?

A variant carries the test task's crux / given-away workaround, or several of its distinctive choices, in any wording;
sharing the genre is fine. Each TB2.1 task has a fingerprint (genre, conventions, distinctive choices weighted
decisive/minor) written before any candidate was seen; the judge (rubric_reuse.md) compares candidates against it.

  python -m decontam.variants pairs --work W --fingerprints F --k 50 --out W/v-pairs-50.jsonl
      Candidates per TB2.1 task: best rank <= k under any retrieval signal (the five of retrieve.py plus `fp`, bge-small
      cosine between the fingerprint's distinctive choices and the candidate's instruction), plus forced pairs: the
      calibration set and every pair behind an issue-#14 exclusion.
  python -m decontam.variants deepen --work W --fingerprints F --verdicts V --k-from 50 --k-to 100 --out P
      TB2.1 tasks whose variants reach the outer half of the judged depth get the next band of candidates.
  python -m decontam.variants resample --work W --verdicts V --out P
      Pairs whose first two samples include WEAK_REUSE but no REUSE: judge two more samples (--sample-offset 2).
      The pair rule (calibrated 2026-10-29, effort max): excluded if any sample says REUSE.
  python -m decontam.variants near --work W --verdicts V --out P
      CalibForge near-copies (apply.py's rule) of excluded tasks, paired with the same TB2.1 task, not yet judged.
  python -m decontam.variants audit-sample --work W --fingerprints F --verdicts V[,V2] --out AUDIT_DIR
      Stratified sample for the independent audit (blind to the judge): excluded pairs, WEAK-only pairs, all-NO pairs
      among each TB2.1 task's closest candidates, random all-NO pairs; no calibration pairs. Shuffled into batches.
  python -m decontam.variants audit-score --work W --verdicts V[,V2] --out AUDIT_DIR
      Judge vs audit per stratum, with stratum sizes for population estimates.
  python -m decontam.variants window --work W --fingerprints F --verdicts V[,V2] --k 50 --out P --out-nofp P2
      The user's stopping rule (2026-10-31): pairs never judged in a TB2.1 task's top-k per signal once excluded
      tasks are taken out. Judge them, exclude, repeat until none are new. TB2.1 tasks without a fingerprint (no `fp`
      signal) go to P2, judged with rubric_reuse_nofp.md.
  python -m decontam.variants band --work W --verdicts V[,V2] --out DIR
      The judge's uncertain pairs (one or two REUSE votes, all or some WEAK_REUSE) whose task no other source
      excludes, in batches for a blind second opinion (same brief as the audits); `final --adjudicated` applies it.
  python -m decontam.variants final --work W --fingerprints F --verdicts V[,V2] --out DIR
      Writes DIR/decontamination-variants.json (+ .pairs.jsonl).
  python -m decontam.variants union --lists tb21=DIR1,tblite=DIR2[,...] --out DIR
      One list from several (per test set, or per judge): a task is excluded if any list excludes it; sources are
      prefixed with the list's name. Pair files are concatenated with a `list` field.
"""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np

from . import apply as A
from . import tasks as T

SIGNALS = ('instr', 'instr_tests', 'ids', 'overlap', 'embed', 'fp')
OLD = '/data/aai/tbench/synth/shared/datasets/calibforge/decontamination-rewrites.json'
CALIBRATION = '/data/aai/tbenchtrain/decontam_calibforge/reuse/calibration.json'
# TB2.1 tasks without a fingerprint (Claude's safety filter stopped the extraction): their issue-#14 exclusions are
# carried over unchanged and their pairs are not re-judged.
NO_FINGERPRINT = ('break-filter-js-from-html', 'crack-7z-hash', 'filter-js-from-html', 'vulnerable-secret')


def read_fingerprints(path):
    return {r['tb']: r for r in map(json.loads, open(path))}


def fp_query(fp):
    return ' '.join(c['choice'] for c in fp['choices'])


def ranks(work, fps):
    """{signal: rank matrix (TB x CF)}, tb names, cf names; `fp` is computed (and cached) here."""
    work = Path(work)
    z = np.load(work / 'scores.npz')
    tbn, cfn = [str(x) for x in z['tb']], [str(x) for x in z['cf']]
    R = {s: np.where(z[f'score_{s}'] > 0, z[f'rank_{s}'], 10 ** 9) for s in SIGNALS if s != 'fp'}
    cache = work / 'fp-ranks.npy'
    if cache.exists():
        R['fp'] = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
        Q = model.encode([fp_query(fps[t]) if t in fps else '' for t in tbn], normalize_embeddings=True)
        S = Q @ np.load(work / 'cf-bge-small.npy').T
        order = np.argsort(-S, axis=1, kind='stable')
        R['fp'] = np.empty_like(order)
        R['fp'][np.arange(len(tbn))[:, None], order] = np.arange(1, len(cfn) + 1)
        np.save(cache, R['fp'])
    return R, tbn, cfn


def best_ranks(R):
    return np.minimum.reduce([R[s] for s in SIGNALS])


def old_exclusions():
    """{cf: {'sources': [...], 'tb_tasks': [...]}} of the issue-#14 list."""
    return json.loads(Path(OLD).read_text())['excluded']


def calibration_pairs():
    cal = json.loads(Path(CALIBRATION).read_text())
    return [(e[0], e[1]) for kind in ('positive', 'negative', 'borderline') for e in cal[kind]]


def write_pairs(rows, out):
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def cmd_pairs(a):
    fps = read_fingerprints(a.fingerprints)
    R, tbn, cfn = ranks(a.work, fps)
    B = best_ranks(R)
    ti, ci = {n: i for i, n in enumerate(tbn)}, {n: j for j, n in enumerate(cfn)}
    rows = {}
    for t in fps:
        i = ti[t]
        for j in np.nonzero(B[i] <= a.k)[0]:
            rows[(t, cfn[j])] = {'tb': t, 'cf': cfn[j], 'best': int(B[i, j]),
                                 'via': [s for s in SIGNALS if R[s][i, j] <= a.k], 'forced': []}
    forced = [((t, c), 'calibration') for t, c in calibration_pairs()]
    forced += [((t, c), 'issue14:' + '+'.join(e['sources'])) for c, e in old_exclusions().items() for t in e['tb_tasks']]
    for (t, c), why in forced:
        if t not in fps or t not in ti or c not in ci:
            continue
        r = rows.setdefault((t, c), {'tb': t, 'cf': c, 'best': int(B[ti[t], ci[c]]), 'via': [], 'forced': []})
        r['forced'].append(why)
    out = sorted(rows.values(), key=lambda r: (r['tb'], r['best'], r['cf']))
    write_pairs(out, a.out)
    per = collections.Counter(r['tb'] for r in out)
    print(f'{len(out)} pairs over {len(per)} TB2.1 tasks (min {min(per.values())}, max {max(per.values())} per task), '
          f'{sum(bool(r["forced"]) and not r["via"] for r in out)} forced only; by signal: '
          f'{ {s: sum(s in r["via"] for r in out) for s in SIGNALS} } -> {a.out}')


def pair_labels(paths):
    """{(tb, cf): [labels]} from judge outputs."""
    return {p: [s['verdict']['label'] for s in samples] for p, samples in A.read_verdicts(paths).items()}


def decide(labels, rule, weak=False):
    """Pair decision from its sample labels: 'any' sample, a 'majority', or at least k ('k2', 'k3') say REUSE
    (or WEAK_REUSE, with weak). Round 2 gives every pair with a non-NO label four samples, so k counts out of four."""
    n = sum(l in (('REUSE', 'WEAK_REUSE') if weak else ('REUSE',)) for l in labels)
    if rule == 'any':
        return n >= 1
    if rule == 'majority':
        return 2 * n > len(labels)
    return n >= int(rule[1:])


def cmd_deepen(a):
    fps = read_fingerprints(a.fingerprints)
    R, tbn, cfn = ranks(a.work, fps)
    B = best_ranks(R)
    ti, ci = {n: i for i, n in enumerate(tbn)}, {n: j for j, n in enumerate(cfn)}
    labels = pair_labels(a.verdicts.split(','))
    deep = collections.defaultdict(int)
    for (t, c), ls in labels.items():
        b = int(B[ti[t], ci[c]])
        trigger = ('REUSE',) if a.trigger == 'reuse' else ('REUSE', 'WEAK_REUSE')
        if a.k_from // 2 < b <= a.k_from and any(l in trigger for l in ls):
            deep[t] += 1
    rows = []
    for t in sorted(deep):
        i = ti[t]
        for j in np.nonzero((B[i] > a.k_from) & (B[i] <= a.k_to))[0]:
            if (t, cfn[j]) not in labels:
                rows.append({'tb': t, 'cf': cfn[j], 'best': int(B[i, j]),
                             'via': [s for s in SIGNALS if R[s][i, j] <= a.k_to], 'forced': []})
    write_pairs(rows, a.out)
    print(f'{len(deep)} TB2.1 tasks with {a.trigger} verdicts at best rank {a.k_from // 2 + 1}-{a.k_from}: {dict(deep)}')
    print(f'{len(rows)} new pairs at best rank {a.k_from + 1}-{a.k_to} -> {a.out}')


def excluded_pairs(labels, rule, weak):
    return {p for p, ls in labels.items() if decide(ls, rule, weak)}


def all_weak(labels):
    return {p for p, ls in labels.items() if ls and all(l == 'WEAK_REUSE' for l in ls)}


def listed_tasks(work, labels, carried=True):
    """Every CalibForge task the list excludes (judge any-REUSE or all-WEAK, canary, TB-named, carried over unless
    `carried` is False: the issue-#14 carry-overs belong to TB2.1)."""
    work = Path(work)
    retrieval = json.loads((work / 'retrieval.json').read_text())
    tb21 = sorted(str(x) for x in np.load(work / 'scores.npz')['tb'])
    out = {c for _, c in excluded_pairs(labels, 'any', False) | all_weak(labels)}
    out |= set(retrieval['cf_canary_tasks']) | set(tb_named(work, tb21))
    if carried:
        out |= {c for c, e in old_exclusions().items() for v in e['verdicts']
                if v.get('tb') in NO_FINGERPRINT or (v.get('source') == 'lexical' and v.get('tb') not in tb21)}
    return out


def split_by_fingerprint(rows, fps, out, out_nofp):
    """Pairs of fingerprinted TB2.1 tasks to `out`, the others to `out_nofp` (judged with rubric_reuse_nofp.md)."""
    write_pairs([r for r in rows if r['tb'] in fps], out)
    write_pairs([r for r in rows if r['tb'] not in fps], out_nofp)
    return sum(r['tb'] in fps for r in rows), sum(r['tb'] not in fps for r in rows)


def cmd_window(a):
    fps = read_fingerprints(a.fingerprints)
    R, tbn, cfn = ranks(a.work, fps)
    labels = pair_labels(a.verdicts.split(','))
    listed = listed_tasks(a.work, labels, not a.drop_carried)
    gone = np.array([c in listed for c in cfn])
    rows = {}
    for i, t in enumerate(tbn):
        for s in SIGNALS:
            if s == 'fp' and t not in fps:  # no fingerprint, no fp signal
                continue
            r = R[s][i]
            order = np.argsort(r, kind='stable')[:a.k + int(gone.sum())]
            for j in [j for j in order if r[j] < 10 ** 9 and not gone[j]][:a.k]:
                if (t, cfn[j]) not in labels:
                    rows.setdefault((t, cfn[j]), {'tb': t, 'cf': cfn[j], 'via': [], 'forced': []})['via'].append(s)
    n, m = split_by_fingerprint(sorted(rows.values(), key=lambda r: (r['tb'], r['cf'])), fps, a.out, a.out_nofp)
    per = collections.Counter(t for t, _ in rows)
    print(f'{len(listed)} listed tasks; top-{a.k} windows: {n} unjudged pairs -> {a.out}, {m} without a fingerprint '
          f'-> {a.out_nofp}; most: {per.most_common(6)}')


def cmd_resample(a):
    rows = A.read_verdicts(a.verdicts.split(','))
    out = []
    for (t, c), samples in sorted(rows.items()):
        first = [s['verdict']['label'] for s in samples if s['sample'] < 2]
        if a.mode == 'weak':
            want = len(first) == 2 and 'REUSE' not in first and 'WEAK_REUSE' in first
        else:  # nonno: every pair whose first two samples are not both NO_REUSE
            want = len(first) == 2 and first != ['NO_REUSE', 'NO_REUSE']
        if want:
            out.append({'tb': t, 'cf': c})
    write_pairs(out, a.out)
    print(f'{len(out)} of {len(rows)} pairs selected ({a.mode}) for two more samples -> {a.out}')


def cmd_near(a):
    labels = pair_labels(a.verdicts.split(','))
    hit = excluded_pairs(labels, a.rule, a.weak) | (all_weak(labels) if a.all_weak else set())
    by_cf = collections.defaultdict(set)
    for t, c in hit:
        by_cf[c].add(t)
    copies = A.near_copies(a.work, sorted(by_cf))
    todo = sorted({(t, y) for x, ts in by_cf.items() for y in copies[x] for t in ts} - set(labels))
    rows = [{'tb': t, 'cf': c, 'via': ['near-copy'], 'forced': []} for t, c in todo]
    if a.out_nofp:
        n, m = split_by_fingerprint(rows, read_fingerprints(a.fingerprints), a.out, a.out_nofp)
    else:
        write_pairs(rows, a.out)
        n, m = len(rows), 0
    print(f'{len(by_cf)} excluded tasks, {sum(map(len, copies.values()))} near-copy links, {n} new pairs -> {a.out}'
          + (f', {m} without a fingerprint -> {a.out_nofp}' if a.out_nofp else ''))


STRATA = (('excluded', 32), ('weak', 24), ('close_no', 24), ('random_no', 10),
          ('issue14_dropped', 0))  # round 1 (issue14_dropped added by hand)
STRATA_R2 = (('reuse_3of4', 24), ('reuse_2of4', 22), ('reuse_1of4', 14), ('weak', 20), ('close_no', 12),
             ('random_no', 6), ('issue14_kept', 12))


def strata(labels, B, ti, ci, votes=False):
    """{pair: stratum} for every judged pair (calibration pairs and TB2.1 tasks without a fingerprint left out)."""
    cal = set(calibration_pairs())
    out = {}
    old = old_exclusions()
    for (t, c), ls in labels.items():
        if (t, c) in cal or t in NO_FINGERPRINT:
            continue
        if votes:
            n = sum(l == 'REUSE' for l in ls)
            out[(t, c)] = ('reuse_3of4' if n >= 3 else 'reuse_2of4' if n == 2 else 'reuse_1of4' if n == 1 else
                           'weak' if 'WEAK_REUSE' in ls else
                           'issue14_kept' if c in old and t in old[c]['tb_tasks'] else
                           'close_no' if B[ti[t], ci[c]] <= 3 else 'no')
            continue
        if decide(ls, 'any'):
            out[(t, c)] = 'excluded'
        elif 'WEAK_REUSE' in ls:
            out[(t, c)] = 'weak'
        elif B[ti[t], ci[c]] <= 3:
            out[(t, c)] = 'close_no'
        else:
            out[(t, c)] = 'no'
    return out


def cmd_audit_sample(a):
    import random
    import shutil
    fps = read_fingerprints(a.fingerprints)
    R, tbn, cfn = ranks(a.work, fps)
    B = best_ranks(R)
    ti, ci = {n: i for i, n in enumerate(tbn)}, {n: j for j, n in enumerate(cfn)}
    st = strata(pair_labels(a.verdicts.split(',')), B, ti, ci, votes=a.round2)
    seen = {(k['tb'], k['cf']) for path in (a.exclude or '').split(',') if path
            for k in json.loads(Path(path).read_text())['pairs']}
    st = {p: s for p, s in st.items() if p not in seen}  # earlier audit rounds: fresh pairs only
    rng = random.Random(a.seed)
    pools = collections.defaultdict(list)
    for p, s in sorted(st.items()):
        pools['random_no' if s == 'no' else s].append(p)
    picked = []
    for name, n in (STRATA_R2 if a.round2 else STRATA):
        picked += [(p, name) for p in rng.sample(pools[name], min(n, len(pools[name])))]
    rng.shuffle(picked)
    out = Path(a.out)
    (out / 'cf').mkdir(parents=True, exist_ok=True)
    dirs = T.extract(T.CF_TAR, Path(a.work) / 'cf')
    key, nb = [], a.batches
    for k, ((t, c), name) in enumerate(picked):
        key.append({'id': k, 'tb': t, 'cf': c, 'stratum': name})
        if not (out / 'cf' / c).exists():
            shutil.copytree(dirs[c], out / 'cf' / c, symlinks=True)
    for b in range(nb):
        write_pairs([{'id': r['id'], 'tb': r['tb'], 'cf': r['cf']} for r in key[b::nb]], out / f'pairs_{b}.jsonl')
    sizes = collections.Counter(s if s != 'no' else 'random_no' for s in st.values())
    key_path = Path(a.work) / a.key_name
    key_path.write_text(json.dumps({'pairs': key, 'stratum_sizes': sizes}, indent=1))
    print(f'{len(key)} pairs in {nb} batches -> {out}; stratum sizes {dict(sizes)}; key -> {key_path}')


def in_band(ls):
    """The judge's uncertain pairs: one or two REUSE votes, or WEAK_REUSE votes without REUSE."""
    k = ls.count('REUSE')
    return 1 <= k <= 2 or (k == 0 and 'WEAK_REUSE' in ls)


def cmd_band(a):
    import shutil
    labels = pair_labels(a.verdicts.split(','))
    seen = {(k['tb'], k['cf']) for path in (a.exclude or '').split(',') if path
            for k in json.loads(Path(path).read_text())['pairs']}
    other = {c for (t, c), ls in labels.items() if ls.count('REUSE') >= 3} | listed_tasks(a.work, {}, not a.drop_carried)
    picked = []
    for (t, c), ls in sorted(labels.items()):
        if t in NO_FINGERPRINT or (t, c) in seen or c in other or not in_band(ls):
            continue
        k, w = ls.count('REUSE'), ls.count('WEAK_REUSE')
        picked.append(((t, c), f'reuse_{k}' if k else 'all_weak' if w == len(ls) else 'some_weak'))
    import random
    random.Random(a.seed).shuffle(picked)
    out = Path(a.out)
    (out / 'cf').mkdir(parents=True, exist_ok=True)
    dirs = T.extract(T.CF_TAR, Path(a.work) / 'cf')
    key, nb = [], a.batches
    for k, ((t, c), name) in enumerate(picked):
        key.append({'id': k, 'tb': t, 'cf': c, 'stratum': name})
        if not (out / 'cf' / c).exists():
            shutil.copytree(dirs[c], out / 'cf' / c, symlinks=True)
    for b in range(nb):
        write_pairs([{'id': r['id'], 'tb': r['tb'], 'cf': r['cf']} for r in key[b::nb]], out / f'pairs_{b}.jsonl')
    key_path = Path(a.work) / a.key_name
    key_path.write_text(json.dumps({'pairs': key, 'stratum_sizes': collections.Counter(n for _, n in picked)}, indent=1))
    print(f'{len(key)} pairs in {nb} batches -> {out}; {dict(collections.Counter(n for _, n in picked))}; key -> {key_path}')


def read_adjudicated(spec):
    """{(tb, cf): label} from KEY:OUTDIR items (blind second opinions: band batches or audits)."""
    out = {}
    for item in spec.split(','):
        key_path, out_dir = item.split(':')
        ids = {k['id']: (k['tb'], k['cf']) for k in json.loads(Path(key_path).read_text())['pairs']}
        for path in sorted(Path(out_dir).glob('out_*.jsonl')):
            for r in map(json.loads, open(path)):
                if r.get('id') in ids and r.get('label'):
                    out[ids[r['id']]] = r
    return out


def cmd_audit_score(a):
    key = json.loads((Path(a.work) / a.key_name).read_text())
    labels = pair_labels(a.verdicts.split(','))
    audit = {}
    for path in sorted(Path(a.out).glob('out_*.jsonl')):
        for r in map(json.loads, open(path)):
            audit[r['id']] = r
    rows = collections.defaultdict(list)
    for k in key['pairs']:
        r = audit.get(k['id'])
        if r:
            rows[k['stratum']].append({**k, 'judge': labels[(k['tb'], k['cf'])], 'audit': r['label'],
                                       'confidence': r.get('confidence'), 'reason': r.get('reason', ''),
                                       'fingerprint_note': r.get('fingerprint_note', '')})
    report = {}
    for name, _ in STRATA + STRATA_R2:
        if name in report or not rows[name]:
            continue
        rs = rows[name]
        c = collections.Counter(r['audit'] for r in rs)
        report[name] = {'audited': len(rs), 'population': key['stratum_sizes'].get(name, 0), 'audit_labels': dict(c),
                        'audit_reuse_rate': round(c['REUSE'] / len(rs), 3) if rs else None}
    print(json.dumps(report, indent=1))
    (Path(a.out) / 'score.json').write_text(json.dumps({'strata': report, 'rows': rows}, indent=1))


def tb_named(work, tb_names):
    """{cf: [TB2.1 task names]} for CalibForge instructions that name a TB2.1 task verbatim. In CalibForge these are
    generic 'environment analyzer' tasks 'for the <TB task> domain' whose /app is cloned from that TB task's
    environment, so the TB task's starter files are agent-visible."""
    import pickle
    import re
    pat = re.compile(r'(?<![\w/-])(' + '|'.join(map(re.escape, tb_names)) + r')(?![\w-])')
    cf = pickle.loads((Path(work) / 'features.pkl').read_bytes())['cf']
    return {c['name']: sorted(set(pat.findall(c['instruction']))) for c in cf if pat.search(c['instruction'])}


def cmd_union(a):
    lists = [item.split('=', 1) for item in a.lists.split(',')]
    excluded, rules, counts, pairs = {}, {}, {}, []
    for name, d in lists:
        rep = json.loads((Path(d) / 'decontamination-variants.json').read_text())
        rules[name], counts[name] = rep['rule'], rep['counts']
        for c, e in rep['excluded'].items():
            u = excluded.setdefault(c, {'sources': [], 'tb_tasks': [], 'verdicts': []})
            u['sources'] += [f'{name}:{s}' for s in e['sources'] if f'{name}:{s}' not in u['sources']]
            u['tb_tasks'] += [t for t in e['tb_tasks'] if t not in u['tb_tasks']]
            u['verdicts'] += [{**v, 'list': name} for v in e['verdicts']]
        pf = Path(d) / 'decontamination-variants.pairs.jsonl'
        if pf.exists():
            pairs += [{**json.loads(l), 'list': name} for l in open(pf)]
    by_list = {name: sum(any(s.startswith(name + ':') for s in e['sources']) for e in excluded.values()) for name, _ in lists}
    report = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'issue': 'internal issue 14',
              'rule': 'excluded if any of these lists excludes the task: ' + '; '.join(f'{n}: {r}' for n, r in rules.items()),
              'counts': {'excluded_tasks': len(excluded), 'excluded_by_list_any': by_list, 'lists': counts},
              'excluded': dict(sorted(excluded.items()))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / '.decontamination-variants.json.partial'  # atomic: a training driver waits for this file
    tmp.write_text(json.dumps(report, indent=1))
    with open(out / 'decontamination-variants.pairs.jsonl', 'w') as f:
        for r in pairs:
            f.write(json.dumps(r) + '\n')
    tmp.rename(out / 'decontamination-variants.json')
    print(json.dumps(report['counts']['excluded_by_list_any']), len(excluded), 'tasks ->', out)


def cmd_final(a):
    fps = read_fingerprints(a.fingerprints)
    work = Path(a.work)
    rows = A.read_verdicts(a.verdicts.split(','))
    labels = {p: [s['verdict']['label'] for s in samples] for p, samples in rows.items()}
    hit = excluded_pairs(labels, a.rule, a.weak)
    weak = {p for p, ls in labels.items() if p not in hit and all(l == 'WEAK_REUSE' for l in ls)} if a.all_weak else set()
    adj = {}
    if a.adjudicated:  # a blind second opinion decides the judge's uncertain pairs it has seen
        adj = {p: r for p, r in read_adjudicated(a.adjudicated).items()
               if p in labels and p[0] not in NO_FINGERPRINT and in_band(labels[p])}
        hit, weak = hit - set(adj), weak - set(adj)
    retrieval = json.loads((work / 'retrieval.json').read_text())
    tb21 = {str(x) for x in np.load(work / 'scores.npz')['tb']}
    excluded = {}

    def add(task, source, tb=None, **info):
        e = excluded.setdefault(task, {'sources': [], 'tb_tasks': [], 'verdicts': []})
        if source not in e['sources']:
            e['sources'].append(source)
        if tb and tb not in e['tb_tasks']:
            e['tb_tasks'].append(tb)
        e['verdicts'].append({'source': source, 'tb': tb, **info})

    for (t, c) in sorted(hit):
        samples = rows[(t, c)]
        best = next(s for s in samples if s['verdict']['label'] == 'REUSE') if 'REUSE' in labels[(t, c)] else samples[0]
        add(c, 'judge' if t in fps else 'judge-nofp', t, labels=labels[(t, c)], reason=best['verdict'].get('reason', ''),
            matches=best['verdict'].get('matches', []))
    for (t, c) in sorted(weak):  # the judge's grey zone: about one in five such pairs is a variant per the audits
        v = rows[(t, c)][0]['verdict']
        add(c, 'judge-all-weak' if t in fps else 'judge-all-weak-nofp', t, labels=labels[(t, c)],
            reason=v.get('reason', ''), matches=v.get('matches', []))
    confirmed = {p for p, r in adj.items() if r['label'] == 'REUSE'}
    for (t, c) in sorted(confirmed):
        add(c, 'adjudicated', t, labels=labels[(t, c)], reason=adj[(t, c)].get('reason', ''),
            matches=adj[(t, c)].get('matches', []))
    for c in retrieval['cf_canary_tasks']:
        add(c, 'canary')
    if not a.no_tb_named:
        for c, names in tb_named(work, sorted(tb21)).items():
            for t in names:
                add(c, 'tb-named-environment', t)
    for c, e in ({} if a.drop_carried else old_exclusions()).items():  # carried over (TB2.1 only): tasks without a
        for v in e['verdicts']:                                             # fingerprint, references outside TB2.1
            if v.get('tb') in NO_FINGERPRINT:
                add(c, 'carried:no-fingerprint', v['tb'], issue14=v)
            elif v.get('source') == 'lexical' and v.get('tb') not in tb21:
                add(c, 'carried:lexical-non-tb21', v['tb'], issue14=v)
    n_cf = retrieval['cf_tasks']
    by_source = collections.Counter(s for e in excluded.values() for s in e['sources'])
    only = collections.Counter(e['sources'][0] for e in excluded.values() if len(e['sources']) == 1)
    old = old_exclusions()
    report = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'issue': 'internal issue 14',
        'test_set': 'Terminal-Bench 2.1 (89 tasks)', 'candidates': f'CalibForge ({n_cf} tasks)',
        'rule': (f'exclude a task if it is a variant of a specific TB2.1 task (judge: rubric_reuse.md with candidate-blind '
                 f'fingerprints, pair excluded if {a.rule} of its samples say REUSE'
                 + (' or WEAK_REUSE' if a.weak else '')
                 + (', or all of its samples say WEAK_REUSE (judge-all-weak)' if a.all_weak else '')
                 + ('; the judge\'s uncertain pairs (one or two REUSE votes, or WEAK_REUSE without REUSE) are decided by '
                    'a blind second opinion where one exists (adjudicated)' if a.adjudicated else '')
                 + '; TB2.1 tasks without a fingerprint are judged with rubric_reuse_nofp.md (judge-nofp)'
                 + '), or it carries a Terminal-Bench canary GUID (canary); '
                 'issue-#14 exclusions against TB2.1 tasks without a fingerprint are carried over'
                 + ('' if a.no_tb_named else '; or its instruction names a TB2.1 task verbatim (tb-named-environment: '
                    'generic analyzer tasks run over an environment cloned from that TB2.1 task)')),
        'counts': {'pairs_judged': len(labels), 'pairs_excluded': len(hit | weak | confirmed), 'excluded_tasks': len(excluded),
                   'pairs_adjudicated': len(adj), 'pairs_adjudicated_reuse': len(confirmed),
                   'exclusion_rate': round(len(excluded) / n_cf, 4), 'excluded_by_source_any': dict(by_source),
                   'excluded_by_source_only': dict(only),
                   'pair_labels': dict(collections.Counter(','.join(sorted(ls)) for ls in labels.values()).most_common()),
                   'vs_issue14': {'both': len(set(excluded) & set(old)), 'new_only': len(set(excluded) - set(old)),
                                  'issue14_only': len(set(old) - set(excluded))}},
        'excluded': dict(sorted(excluded.items())),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'decontamination-variants.json').write_text(json.dumps(report, indent=1))
    with open(out / 'decontamination-variants.pairs.jsonl', 'w') as f:
        for (t, c), ls in sorted(labels.items()):
            row = {'tb': t, 'cf': c, 'labels': ls, 'excluded': (t, c) in hit | weak | confirmed}
            if (t, c) in adj:
                row['adjudicated'] = adj[(t, c)]['label']
            f.write(json.dumps(row) + '\n')
    print(json.dumps(report['counts'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pairs', 'deepen', 'resample', 'near', 'window', 'band', 'audit-sample',
                                    'audit-score', 'final', 'union'])
    ap.add_argument('--lists', help='union: comma list of NAME=DIR (each with decontamination-variants.json)')
    ap.add_argument('--batches', type=int, default=5)
    ap.add_argument('--key-name', default='audit_key.json', help='audit key file under --work')
    ap.add_argument('--exclude', help='audit-sample: comma list of earlier audit keys whose pairs are not drawn again')
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--no-tb-named', action='store_true', help='final: leave out the tb-named-environment source')
    ap.add_argument('--mode', choices=['weak', 'nonno'], default='weak', help='resample: which pairs get two more samples')
    ap.add_argument('--round2', action='store_true', help='audit-sample: strata by REUSE votes out of four')
    ap.add_argument('--trigger', choices=['reuse', 'weak'], default='reuse',
                    help='deepen: which labels in the outer half of the judged depth trigger the next band')
    ap.add_argument('--work')
    ap.add_argument('--fingerprints')
    ap.add_argument('--verdicts')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--k-from', type=int)
    ap.add_argument('--k-to', type=int)
    ap.add_argument('--rule', choices=['any', 'majority', 'k2', 'k3'], default='any')
    ap.add_argument('--weak', action='store_true', help='WEAK_REUSE samples count as REUSE')
    ap.add_argument('--all-weak', action='store_true',
                    help='final: also exclude pairs whose samples all say WEAK_REUSE (source judge-all-weak)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out-nofp', help='window/near: pairs of TB2.1 tasks without a fingerprint go here')
    ap.add_argument('--adjudicated', help='final: comma list of KEY:OUTDIR blind second opinions (band, audits)')
    ap.add_argument('--drop-carried', action='store_true',
                    help='window/band/final: no issue-#14 carry-overs (another test set than TB2.1, or the TB2.1 tasks '
                         'without a fingerprint judged instead)')
    a = ap.parse_args()
    {'pairs': cmd_pairs, 'deepen': cmd_deepen, 'resample': cmd_resample, 'near': cmd_near,
     'window': cmd_window, 'band': cmd_band, 'audit-sample': cmd_audit_sample, 'audit-score': cmd_audit_score,
     'final': cmd_final, 'union': cmd_union}[a.cmd](a)

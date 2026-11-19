"""Issue #19: task_synthesis training tasks vs the held-out TB2.1 and TB-Lite tasks, under issue #14's variants criterion.

A candidate is excluded if it is a variant of one specific test task (rubric_reuse_synth.md, judged against that task's
candidate-blind fingerprint; rubric_reuse_nofp_synth.md for the TB2.1 tasks without one): any REUSE among its pooled
samples, or WEAK_REUSE in all of them, as for CalibForge; or it carries a Terminal-Bench canary GUID.

  python -m decontam.synth setup    --work W --scratch S        extract references + candidates to S (node-local)
  python -m decontam.synth retrieve --work W --scratch S        features, five lexical/embedding signals, bundles
  python -m decontam.synth pairs    --work W --fingerprints F --k 50 --out P --out-nofp P2
        top-k per signal (+ `fp`: fingerprint choices vs candidate instruction), plus forced pairs: each candidate with
        the TB2.1 task that is the only source (<= 2 tasks) of one of its spec's details (the sampler's formats, services
        and git states were fitted on TB 2.1 + 3.0 labels), and candidates whose instruction names a test task
  python -m decontam.synth signals  --work W --pairs P[,P2] --out-name N     overlap-signal blocks for these pairs
  python -m decontam.synth window   --work W --fingerprints F --verdicts V --k 50 --out P --out-nofp P2
        pairs never judged in a test task's top-k per signal once excluded candidates are taken out (repeat until none)
  python -m decontam.synth final    --work W --fingerprints F --verdicts V --out DIR

W holds only derived files (features, scores, bundles, signals, pairs, verdicts); S holds the extracted task trees.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np

from . import render as RD
from . import retrieve as R
from . import tasks as T
from . import variants as V

REFS = {'tb21': ('Terminal-Bench 2.1', '/data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar'),
        'tblite': ('Terminal-Bench Lite', '/data/aai/tbenchtrain/eval_q35/bundle/tblite_6h.tar')}
CAMPAIGNS = Path('/data/aai/tbench/tasksynth/campaigns')
CANDIDATES = ('r5', 's1', 's2')  # student 1's bundles (s2b reran s2 tasks from the s2 bundle)
SAMPLER = Path('/opt/aai/worktrees/task-synthesis/task_synthesis/asset_sampler')
SIGNALS = V.SIGNALS  # instr, instr_tests, ids, overlap, embed, fp
SPEC_MAX_SUPPORT = 2
CANDIDATE_TITLE = 'CANDIDATE TASK (synthetic training data)'


def index(work):
    return json.loads((Path(work) / 'index.json').read_text())


def cmd_setup(a):
    work, scratch = Path(a.work), Path(a.scratch)
    work.mkdir(parents=True, exist_ok=True)
    refs, cands = {}, {}
    for slug, (label, tar) in REFS.items():
        for name, d in T.extract(tar, scratch / 'tb' / slug).items():
            assert name not in refs, f'duplicate reference task {name}'
            refs[name] = {'bench': label, 'dir': str(d)}
    dup = 0
    for c in CANDIDATES:
        for name, d in T.extract(CAMPAIGNS / c / 'tasks.tar', scratch / 'cf' / c).items():
            if name in cands:
                dup += 1
                continue
            cands[name] = {'campaign': c, 'dir': str(d)}
    (work / 'index.json').write_text(json.dumps({'refs': refs, 'cands': cands}, indent=0))
    print(f'{len(refs)} references ({collections.Counter(r["bench"] for r in refs.values())}), {len(cands)} candidates '
          f'({collections.Counter(c["campaign"] for c in cands.values())}; {dup} duplicates across bundles skipped)')


def candidate_features(d):
    """retrieve.features, with the author's solution sketch standing in for the tests (the tasks ship none)."""
    f = R.features(d)
    task = T.load(d)
    sketch = T.field_text(task, 'solution')
    f['tests'] = sketch
    f['ids'] |= {x for x in T.IDENT.findall(sketch) if x != '/app'}
    f['spec'] = json.loads((Path(d) / 'tasksynth.json').read_text()).get('spec', {})
    return f


def _render_init(bench, n):
    RD.REF_BENCH.update(bench)
    RD.CANDIDATE_TITLE = CANDIDATE_TITLE
    RD.CANDIDATE_POOL = f'{n:,} synthetic training tasks'


def _render(args):
    return RD.render(*args)


def configure_render(ix):
    _render_init({n: r['bench'] for n, r in ix['refs'].items()}, len(ix['cands']))


def cmd_retrieve(a):
    work = Path(a.work)
    ix = index(work)
    configure_render(ix)
    tbn, cfn = sorted(ix['refs']), sorted(ix['cands'])
    cache = work / 'features.pkl'
    if cache.exists():
        F = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            F = {'tb': list(pool.map(R.features, [ix['refs'][n]['dir'] for n in tbn], chunksize=4)),
                 'cf': list(pool.map(candidate_features, [ix['cands'][n]['dir'] for n in cfn], chunksize=32))}
        cache.write_bytes(pickle.dumps(F))
    tb, cf = F['tb'], F['cf']
    assert [t['name'] for t in tb] == tbn and [c['name'] for c in cf] == cfn
    t0 = time.time()
    S = {'instr': R.cosine_tfidf([t['instruction'] for t in tb], [c['instruction'] for c in cf]),
         'instr_tests': R.cosine_tfidf([t['instruction'] + '\n' + t['tests'] for t in tb],
                                       [c['instruction'] + '\n' + c['tests'] for c in cf]),
         'ids': R.identifier_scores(tb, cf)}
    contain, shared_files = R.overlap_scores(tb, cf)
    S['overlap'] = np.maximum.reduce([contain[f] for f in R.FIELDS]) + shared_files
    E = R.embeddings([t['instruction'] for t in tb] + [c['instruction'] for c in cf], work / 'bge-small.npy')
    S['embed'] = E[:len(tb)] @ E[len(tb):].T
    np.save(work / 'cf-bge-small.npy', E[len(tb):])
    np.savez_compressed(work / 'scores.npz', tb=np.array(tbn), cf=np.array(cfn), shared_files=shared_files,
                        **{f'score_{s}': S[s] for s in S}, **{f'rank_{s}': R.ranks(S[s]) for s in S},
                        **{f'contain_{f}': contain[f] for f in R.FIELDS})
    pat = re.compile(r'(?<![\w/-])(' + '|'.join(map(re.escape, sorted(tbn, key=len, reverse=True))) + r')(?![\w-])')
    report = {'refs': len(tbn), 'cands': len(cfn), 'signal_seconds': round(time.time() - t0),
              'cf_canary_tasks': sorted(c['name'] for c in cf if c['canaries']),
              'tb_with_canary': sum(bool(t['canaries']) for t in tb),
              'tb_named': {c['name']: sorted(set(pat.findall(c['instruction']))) for c in cf if pat.search(c['instruction'])}}
    (work / 'retrieval.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if not isinstance(v, (dict, list)) else len(v)) for k, v in report.items()}))
    if not (work / 'bundles.jsonl').exists():
        jobs = [(ix['refs'][n]['dir'], 'tb') for n in tbn] + [(ix['cands'][n]['dir'], 'cf') for n in cfn]
        bench = {n: r['bench'] for n, r in ix['refs'].items()}
        stats = collections.defaultdict(list)
        tmp = work / 'bundles.jsonl.partial'
        with ProcessPoolExecutor(a.workers, initializer=_render_init, initargs=(bench, len(cfn))) as pool, \
                tmp.open('w') as f:
            for row in pool.map(_render, jobs, chunksize=16):
                f.write(json.dumps(row) + '\n')
                stats[row['side']].append((row['tokens'], row['truncated']))
        os.replace(tmp, work / 'bundles.jsonl')
        out = {s: {'n': len(v), 'p50': sorted(t for t, _ in v)[len(v) // 2], 'max': max(t for t, _ in v),
                   'truncated': sum(x for _, x in v)} for s, v in stats.items()}
        (work / 'render.json').write_text(json.dumps(out, indent=1))
        print('bundles', json.dumps(out))


def read_fps(paths):
    fps = {}
    for p in paths.split(','):
        fps.update(V.read_fingerprints(p))
    return fps


def ranks(work, fps):
    """{signal: rank matrix (refs x candidates), 10**9 where the signal has no score}; `fp` computed and cached here,
    only for references with a fingerprint."""
    work = Path(work)
    z = np.load(work / 'scores.npz')
    tbn, cfn = [str(x) for x in z['tb']], [str(x) for x in z['cf']]
    R_ = {s: np.where(z[f'score_{s}'] > 0, z[f'rank_{s}'], 10 ** 9) for s in SIGNALS if s != 'fp'}
    key = sorted(t for t in tbn if t in fps)
    cache = work / 'fp-ranks.npz'
    if cache.exists() and list(np.load(cache)['refs']) == key:
        R_['fp'] = np.load(cache)['ranks']
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
        Q = model.encode([V.fp_query(fps[t]) if t in fps else '' for t in tbn], normalize_embeddings=True)
        S = Q @ np.load(work / 'cf-bge-small.npy').T
        r = R.ranks(S)
        r[[i for i, t in enumerate(tbn) if t not in fps]] = 10 ** 9
        np.savez(cache, refs=np.array(key), ranks=r)
        R_['fp'] = r
    return R_, tbn, cfn


def spec_sources(tb21):
    """{(class, option): [TB2.1 tasks]} for spec details the sampler fitted on at most SPEC_MAX_SUPPORT TB2.1 + 3.0
    tasks, keeping the TB2.1 ones (TB 3.0 is not a reference here)."""
    sys.path.insert(0, str(SAMPLER))
    import sample_assets as S
    base = {(r['set'], r['task']): r for r in json.load(open(SAMPLER / 'asset_labels.json'))}
    support = collections.defaultdict(set)
    for a in json.load(open(SAMPLER / 'anchor_labels.json')):
        r = base[(a['set'], a['task'])]
        classes = set(r['classes']) | {x['class'] for x in a['assets']}
        for c, (_, opts) in S.DETAILS.items():
            if c in classes:
                for o, toks in opts.items():
                    if S.matches(toks, r['ext'], r['description'].lower()):
                        support[(c, o)].add((a['set'], a['task']))
    return {k: sorted(t for s, t in v if s == 'TB2.1' and t in tb21) for k, v in support.items()
            if len(v) <= SPEC_MAX_SUPPORT and any(s == 'TB2.1' and t in tb21 for s, t in v)}


def spec_details(spec):
    for x in spec.get('assets', []):
        state = x.get('state')
        state = state if isinstance(state, list) else [state] if state else []
        for o in list(x.get('formats', [])) + list(x.get('services', [])) + state:
            yield (x['class'], o)


def forced_pairs(work, tbn):
    F = pickle.loads((Path(work) / 'features.pkl').read_bytes())
    src = spec_sources(set(tbn))
    out = collections.defaultdict(list)
    for c in F['cf']:
        for d in spec_details(c['spec']):
            for t in src.get(d, ()):
                out[(t, c['name'])].append('spec:' + ':'.join(d))
    for c, names in json.loads((Path(work) / 'retrieval.json').read_text())['tb_named'].items():
        for t in names:
            out[(t, c)].append('tb-named')
    return out, src


def write_split(rows, fps, out, out_nofp):
    return V.split_by_fingerprint(sorted(rows, key=lambda r: (r['tb'], r['cf'])), fps, out, out_nofp)


def cmd_pairs(a):
    fps = read_fps(a.fingerprints)
    R_, tbn, cfn = ranks(a.work, fps)
    missing = [t for t in tbn if t not in fps]
    B = np.minimum.reduce([R_[s] for s in SIGNALS])
    rows = {}
    for i, t in enumerate(tbn):
        for j in np.nonzero(B[i] <= a.k)[0]:
            rows[(t, cfn[j])] = {'tb': t, 'cf': cfn[j], 'best': int(B[i, j]),
                                 'via': [s for s in SIGNALS if R_[s][i, j] <= a.k], 'forced': []}
    forced, src = forced_pairs(a.work, tbn)
    ti, ci = {n: i for i, n in enumerate(tbn)}, {n: j for j, n in enumerate(cfn)}
    for (t, c), why in forced.items():
        r = rows.setdefault((t, c), {'tb': t, 'cf': c, 'best': int(B[ti[t], ci[c]]), 'via': [], 'forced': []})
        r['forced'] += why
    n, m = write_split(rows.values(), fps, a.out, a.out_nofp)
    per = collections.Counter(t for t, _ in rows)
    print(f'{len(rows)} pairs ({n} fingerprinted -> {a.out}, {m} without a fingerprint -> {a.out_nofp}); references without '
          f'a fingerprint: {missing}; per reference min {min(per.values())} max {max(per.values())}; forced-only '
          f'{sum(bool(r["forced"]) and not r["via"] for r in rows.values())}; by signal '
          f'{ {s: sum(s in r["via"] for r in rows.values()) for s in SIGNALS} }; spec sources {len(src)}: '
          f'{ {":".join(k): v for k, v in sorted(src.items())} }')


def cmd_signals(a):
    wanted = [json.loads(l) for p in a.pairs.split(',') if p and Path(p).exists() for l in open(p)]
    n = RD.pair_signals(Path(a.work), RD.bundle_index(a.work), RD.score_pairs(a.work, wanted), a.out_name)
    print(f'{n} pair signal blocks -> {a.out_name}')


def pair_labels(paths):
    return V.pair_labels([p for p in paths.split(',') if p and Path(p).exists()])


def listed(work, labels):
    """Candidates the list excludes: any REUSE, all WEAK_REUSE, or a canary."""
    out = {c for _, c in V.excluded_pairs(labels, 'any', False) | V.all_weak(labels)}
    return out | set(json.loads((Path(work) / 'retrieval.json').read_text())['cf_canary_tasks'])


def cmd_window(a):
    fps = read_fps(a.fingerprints)
    R_, tbn, cfn = ranks(a.work, fps)
    labels = pair_labels(a.verdicts)
    gone = np.array([c in listed(a.work, labels) for c in cfn])
    rows = {}
    for i, t in enumerate(tbn):
        for s in SIGNALS:
            r = R_[s][i]
            order = np.argsort(r, kind='stable')[:a.k + int(gone.sum())]
            for j in [j for j in order if r[j] < 10 ** 9 and not gone[j]][:a.k]:
                if (t, cfn[j]) not in labels:
                    rows.setdefault((t, cfn[j]), {'tb': t, 'cf': cfn[j], 'via': [], 'forced': []})['via'].append(s)
    n, m = write_split(rows.values(), fps, a.out, a.out_nofp)
    print(f'{int(gone.sum())} listed candidates; top-{a.k} windows: {n} unjudged pairs -> {a.out}, {m} without a '
          f'fingerprint -> {a.out_nofp}')


def cmd_nearest(a):
    """Pairs of one benchmark's test tasks with their closest candidates (best rank <= k over all signals), for a
    second judge arm (the no-fingerprint rubric) on top of the fingerprint arm."""
    fps = read_fps(a.fingerprints)
    R_, tbn, cfn = ranks(a.work, fps)
    ix = index(a.work)
    B = np.minimum.reduce([R_[s] for s in SIGNALS])
    rows = [{'tb': t, 'cf': cfn[j], 'best': int(B[i, j]), 'via': [s for s in SIGNALS if R_[s][i, j] <= a.k],
             'forced': []}
            for i, t in enumerate(tbn) if ix['refs'][t]['bench'] == a.bench for j in np.nonzero(B[i] <= a.k)[0]]
    V.write_pairs(rows, a.out)
    print(f'{len(rows)} pairs ({a.bench}, best rank <= {a.k}) -> {a.out}')


def cmd_final(a):
    """Each arm (';'-separated groups of verdict files) is decided on its own (any REUSE, or all WEAK_REUSE, among
    that arm's samples); a candidate is excluded if any arm excludes one of its pairs."""
    work = Path(a.work)
    ix = index(work)
    groups = [[p for p in g.split(',') if p and Path(p).exists()] for g in a.verdicts.split(';')]
    names = a.arm_names.split(',') if a.arm_names else [f'arm{i}' for i in range(len(groups))]
    report, labels, hit = final_report(
        {n: r['bench'] for n, r in ix['refs'].items()}, len(ix['cands']),
        json.loads((work / 'retrieval.json').read_text())['cf_canary_tasks'], read_fps(a.fingerprints), groups, names,
        f'task_synthesis bundles {", ".join(CANDIDATES)} ({len(ix["cands"])} tasks)', a.fingerprints)
    write_final(a.out, report, labels, hit)
    print(json.dumps(report['counts'], indent=1))


def final_report(bench, n_cf, canaries, fps, groups, names, candidates, fingerprints,
                 issue='internal issue 19'):
    """(report, {pair: {arm: labels}}, excluded pairs). bench: reference task -> test set; groups: verdict files per
    arm; canaries: candidates carrying a Terminal-Bench canary GUID."""
    assert len(names) == len(groups), (names, groups)
    arms = []
    for name, paths in zip(names, groups):
        rows_ = V.A.read_verdicts(paths)
        labels_ = {p: [s['verdict']['label'] for s in samples] for p, samples in rows_.items()}
        hit_ = V.excluded_pairs(labels_, 'any', False)
        arms.append((name, rows_, labels_, hit_, V.all_weak(labels_) - hit_))
    labels = {}
    for name, _, labels_, *_ in arms:
        for p, ls in labels_.items():
            labels.setdefault(p, {})[name] = ls
    hit = set().union(*(h for *_, h, _ in arms))
    weak = set().union(*(w for *_, w in arms)) - hit
    excluded = {}

    def add(task, source, tb=None, **info):
        e = excluded.setdefault(task, {'sources': [], 'tb_tasks': [], 'verdicts': []})
        if source not in e['sources']:
            e['sources'].append(source)
        if tb and tb not in e['tb_tasks']:
            e['tb_tasks'].append(tb)
        e['verdicts'].append({'source': source, 'tb': tb, **info})

    for name, rows_, labels_, hit_, weak_ in arms:
        for (t, c) in sorted(hit_ | weak_):
            samples = rows_[(t, c)]
            best = next((s for s in samples if s['verdict']['label'] == 'REUSE'), samples[0])['verdict']
            add(c, f"{name}:" + ('judge' if (t, c) in hit_ else 'judge-all-weak') + ('' if t in fps else '-nofp'), t,
                bench=bench[t], labels=labels_[(t, c)], reason=best.get('reason', ''),
                matches=best.get('matches', []))
    for c in canaries:
        add(c, 'canary')
    report = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'issue': issue,
        'test_sets': {b: sum(v == b for v in bench.values()) for b in sorted(set(bench.values()))},
        'candidates': candidates,
        'rule': ('exclude a task if it is a variant of a specific test task (judge: rubric_reuse_synth.md with candidate-blind '
                 'fingerprints, rubric_reuse_nofp_synth.md for test tasks without one; a pair is excluded if any of its '
                 'samples says REUSE or all of them say WEAK_REUSE; with several judge arms, each arm decides on its own '
                 'samples and any arm excluding a pair excludes it), or it carries a Terminal-Bench canary GUID'),
        'arms': {name: {'verdicts': g, 'pairs_judged': len(l_), 'pairs_excluded': len(h_ | w_)}
                 for (name, _, l_, h_, w_), g in zip(arms, groups)},
        'fingerprints': fingerprints,
        'counts': {'pairs_judged': len(labels), 'pairs_excluded': len(hit | weak), 'excluded_tasks': len(excluded),
                   'exclusion_rate': round(len(excluded) / n_cf, 5),
                   'excluded_by_source': dict(collections.Counter(s for e in excluded.values() for s in e['sources'])),
                   'excluded_by_bench': dict(collections.Counter(bench[t] for e in excluded.values()
                                                                 for t in e['tb_tasks'])),
                   'pair_labels': {name: dict(collections.Counter(','.join(sorted(ls)) for ls in l_.values()).most_common())
                                   for name, _, l_, *_ in arms}},
        'excluded': dict(sorted(excluded.items())),
    }
    return report, labels, hit | weak


def write_final(out, report, labels, excluded_pairs):
    """decontamination-variants.json (the list SFT prep reads: --exclude-tasks), its pairs and a plain name list; the
    JSON is written last, atomically."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'decontamination-variants.pairs.jsonl', 'w') as f:
        for (t, c), ls in sorted(labels.items()):
            f.write(json.dumps({'tb': t, 'cf': c, 'labels': ls, 'excluded': (t, c) in excluded_pairs}) + '\n')
    (out / 'exclude-tasks.txt').write_text(''.join(f'{c}\n' for c in sorted(report['excluded'])))
    tmp = out / '.decontamination-variants.json.partial'
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out / 'decontamination-variants.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['setup', 'retrieve', 'pairs', 'signals', 'window', 'nearest', 'final'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--scratch')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--fingerprints', help='comma list of fingerprint JSONL files')
    ap.add_argument('--verdicts', help='comma list of verdict JSONL files')
    ap.add_argument('--pairs', help='signals: comma list of pair JSONL files')
    ap.add_argument('--out-name', default='pair_signals.jsonl')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--out')
    ap.add_argument('--out-nofp')
    ap.add_argument('--bench', default='Terminal-Bench Lite', help='nearest: which test set')
    ap.add_argument('--arm-names', help="final: comma list naming the ';'-separated verdict groups")
    a = ap.parse_args()
    {'setup': cmd_setup, 'retrieve': cmd_retrieve, 'pairs': cmd_pairs, 'signals': cmd_signals, 'window': cmd_window,
     'nearest': cmd_nearest, 'final': cmd_final}[a.cmd](a)

"""Step 1 (CPU): candidate (TB2.1 task, CalibForge task) pairs for the judge.

  python -m decontam.retrieve --work /tmp/decontam [--k 50]

For each TB2.1 task, the union of the top-k CalibForge tasks under each signal:
  instr     instruction TF-IDF (unigrams + bigrams)
  instr_tests  instruction + tests TF-IDF
  ids       rare identifiers in instruction + tests (paths, filenames, snake/camel names), idf-weighted
  overlap   synth.datasets shingle containment (instruction, tests, Dockerfile, env files; boilerplate shingles in
            more than 8 tasks dropped) plus shared non-boilerplate files, with no flag threshold
  embed     bge-small-en-v1.5 instruction cosine (seeds/METHODOLOGY.md 6.1)
Writes <work>/pairs.jsonl (one pair per line, with every signal's rank and score) and <work>/retrieval.json (recall
of the known positives: the lexical flags against TB2.1, the hand-found rewrites and the calibration seed).
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sp

from synth.datasets import shingles
from . import tasks as T

SIGNALS = ('instr', 'instr_tests', 'ids', 'overlap', 'embed')
FIELDS = {'instruction': 'instruction', 'tests': 'tests', 'dockerfile': 'dockerfile', 'env': 'environment',
          'solution': 'solution'}
SHINGLE_FIELD = {'instruction': 'instruction', 'tests': 'tests', 'dockerfile': 'dockerfile', 'env': 'env',
                 'solution': 'tests'}  # solution uses code-token 10-grams like tests
BOILERPLATE_DF = 8
MIN_SHINGLES = 25
# Rewrites found by hand in the preliminary analysis (issue #14), on top of the calibration seed.
HAND = [('query-optimize', 'data-science_20260604_003814_005'), ('polyglot-c-py', 'software-engineering_20260612_191200_017'),
        ('polyglot-rust-c', 'software-engineering_20260612_191200_017'), ('portfolio-optimization', 'optimization_20260604_194036_001'),
        ('count-dataset-tokens', 'model-training_20260503_030115_001'), ('cancel-async-tasks', 'software-engineering_20260429_103257_003'),
        ('rstan-to-pystan', 'data-science_20260604_002443_007'), ('build-pmars', 'software-engineering_20260513_204536_001'),
        ('raman-fitting', 'scientific-computing_20260419_065313_007'), ('model-extraction-relu-logits', 'mathematics_20260604_002443_004'),
        ('model-extraction-relu-logits', 'mathematics_20260612_191200_004'), ('model-extraction-relu-logits', 'mathematics_20260617_210354_004')]


def features(task_dir):
    return task_features(T.load(task_dir))


def task_features(task):
    text = {'instruction': task['instruction'], **{f: T.field_text(task, p) for f, p in FIELDS.items() if f != 'instruction'}}
    sh = {f: shingles(text[f], SHINGLE_FIELD[f]) if text[f] else set() for f in FIELDS}
    blobs = {f['sha1'] for f in task['files'] if f['size'] >= 300}
    return {'name': task['name'], 'instruction': task['instruction'], 'tests': text['tests'], 'shingles': sh,
            'blobs': blobs, 'ids': T.identifiers(task), 'canaries': T.canaries(task)}


def load_all(work):
    cache = Path(work) / 'features.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    tb = T.extract(T.TB_TAR, Path(work) / 'tb')
    cf = T.extract(T.CF_TAR, Path(work) / 'cf')
    with ProcessPoolExecutor() as pool:
        out = {'tb': list(pool.map(features, tb.values(), chunksize=4)),
               'cf': list(pool.map(features, cf.values(), chunksize=32))}
    cache.write_bytes(pickle.dumps(out))
    return out


def tok(s):
    import re
    return re.findall(r'[a-z0-9_]+', s.lower())


def tfidf(docs):
    feats = [collections.Counter((lambda w: w + [a + ' ' + b for a, b in zip(w, w[1:])])(tok(d))) for d in docs]
    df = collections.Counter(f for c in feats for f in c)
    vocab = {f: i for i, f in enumerate(f for f, n in df.items() if n >= 2)}
    r, c, v = [], [], []
    for i, cnt in enumerate(feats):
        for f, n in cnt.items():
            j = vocab.get(f)
            if j is not None:
                r.append(i); c.append(j); v.append((1 + math.log(n)) * math.log(len(docs) / df[f]))
    X = sp.csr_matrix((v, (r, c)), shape=(len(docs), len(vocab)))
    return sp.diags(1 / np.sqrt(X.multiply(X).sum(1).A1 + 1e-12)) @ X


def cosine_tfidf(tb_docs, cf_docs):
    X = tfidf(tb_docs + cf_docs)
    return (X[:len(tb_docs)] @ X[len(tb_docs):].T).toarray()


def identifier_scores(tb, cf):
    df = collections.Counter(x for t in cf for x in t['ids'])
    S = np.zeros((len(tb), len(cf)))
    for i, t in enumerate(tb):
        for j, c in enumerate(cf):
            shared = t['ids'] & c['ids']
            if shared:
                S[i, j] = sum(math.log(len(cf) / df[x]) for x in shared)
    return S


def overlap_scores(tb, cf, boilerplate=BOILERPLATE_DF):
    """Per-field shingle containment |A∩B| / min(|A|,|B|) (both sides >= 25 shingles) and shared files, TB x CF."""
    df = {f: collections.Counter() for f in FIELDS}
    blob_df = collections.Counter()
    for t in tb + cf:
        for f in FIELDS:
            df[f].update(t['shingles'][f])
        blob_df.update(t['blobs'])
    keep = lambda f, s: {x for x in s if df[f][x] <= boilerplate}
    cf_sets = [{f: keep(f, c['shingles'][f]) for f in FIELDS} for c in cf]
    index = {f: collections.defaultdict(list) for f in FIELDS}
    for j, sets in enumerate(cf_sets):
        for f in FIELDS:
            for x in sets[f]:
                index[f][x].append(j)
    blob_index = collections.defaultdict(list)
    for j, c in enumerate(cf):
        for b in c['blobs']:
            if blob_df[b] <= 3:
                blob_index[b].append(j)
    contain = {f: np.zeros((len(tb), len(cf))) for f in FIELDS}
    shared_files = np.zeros((len(tb), len(cf)))
    for i, t in enumerate(tb):
        for f in FIELDS:
            mine = keep(f, t['shingles'][f])
            if len(mine) < MIN_SHINGLES:
                continue
            hits = collections.Counter(j for x in mine for j in index[f].get(x, ()))
            for j, n in hits.items():
                other = len(cf_sets[j][f])
                if other >= MIN_SHINGLES:
                    contain[f][i, j] = n / min(len(mine), other)
        for b in t['blobs']:
            if blob_df[b] <= 3:
                for j in blob_index.get(b, ()):
                    shared_files[i, j] += 1
    return contain, shared_files


def embeddings(texts, cache):
    if Path(cache).exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(cache, E)
    return E


def ranks(S):
    """Rank of every column per row (1 = best); ties share the better rank."""
    order = np.argsort(-S, axis=1, kind='stable')
    R = np.empty_like(order)
    rows = np.arange(S.shape[0])[:, None]
    R[rows, order] = np.arange(1, S.shape[1] + 1)
    return R


def positives(tb_names, cf_names):
    lexical = json.loads(Path(T.CF_LEXICAL).read_text())['flagged']
    pos = {(h['reference'], c): 'lexical' for c, hs in lexical.items() for h in hs if h['reference'] in tb_names}
    for a, b in HAND:
        pos.setdefault((a, b), 'hand')
    cal = T.read_calibration()
    for a, b in cal['exclude']:
        pos.setdefault((a, b), 'calibration-exclude')
    return {p: s for p, s in pos.items() if p[0] in tb_names and p[1] in cf_names}  # TB2.1 seeds only apply to TB2.1


def main(work, k):
    work = Path(work)
    F = load_all(work)
    tb, cf = F['tb'], F['cf']
    tbn, cfn = [t['name'] for t in tb], [c['name'] for c in cf]
    S = {'instr': cosine_tfidf([t['instruction'] for t in tb], [c['instruction'] for c in cf]),
         'instr_tests': cosine_tfidf([t['instruction'] + '\n' + t['tests'] for t in tb],
                                     [c['instruction'] + '\n' + c['tests'] for c in cf]),
         'ids': identifier_scores(tb, cf)}
    contain, shared_files = overlap_scores(tb, cf)
    S['overlap'] = np.maximum.reduce([contain[f] for f in FIELDS]) + shared_files
    E = embeddings([t['instruction'] for t in tb] + [c['instruction'] for c in cf], work / 'bge-small.npy')
    S['embed'] = E[:len(tb)] @ E[len(tb):].T
    np.save(work / 'cf-bge-small.npy', E[len(tb):])
    R = {s: ranks(S[s]) for s in SIGNALS}
    np.savez_compressed(work / 'scores.npz', tb=np.array(tbn), cf=np.array(cfn), shared_files=shared_files,
                        **{f'score_{s}': S[s] for s in SIGNALS}, **{f'rank_{s}': R[s] for s in SIGNALS},
                        **{f'contain_{f}': contain[f] for f in FIELDS})
    pairs = []
    for i, t in enumerate(tbn):
        for j, c in enumerate(cfn):
            hit = [s for s in SIGNALS if R[s][i, j] <= k and S[s][i, j] > 0]
            if hit:
                pairs.append({'tb': t, 'cf': c, 'signals': hit, 'rank': {s: int(R[s][i, j]) for s in SIGNALS},
                              'score': {s: round(float(S[s][i, j]), 4) for s in SIGNALS},
                              'containment': {f: round(float(contain[f][i, j]), 3) for f in FIELDS if contain[f][i, j]},
                              'shared_files': int(shared_files[i, j])})
    with (work / 'pairs.jsonl').open('w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    pos = positives(set(tbn), set(cfn))
    ti, ci = {n: i for i, n in enumerate(tbn)}, {n: j for j, n in enumerate(cfn)}
    best = {p: min(int(R[s][ti[p[0]], ci[p[1]]]) for s in SIGNALS) for p in pos}
    recall = {} if not pos else {str(kk): {**{s: round(float(np.mean([R[s][ti[a], ci[b]] <= kk for a, b in pos])), 3) for s in SIGNALS},
                        'union': round(float(np.mean([best[p] <= kk for p in pos])), 3)} for kk in (15, 50, 100)}
    cal = T.read_calibration()
    paired = {(p['tb'], p['cf']) for p in pairs}
    report = {'k': k, 'pairs': len(pairs), 'tb_tasks': len(tbn), 'cf_tasks': len(cfn),
              'pairs_per_tb_task': {'min': min(collections.Counter(p['tb'] for p in pairs).values()),
                                    'max': max(collections.Counter(p['tb'] for p in pairs).values())},
              'pairs_by_signal': {s: sum(s in p['signals'] for p in pairs) for s in SIGNALS},
              'positives': len(pos), 'positives_by_source': collections.Counter(pos.values()),
              'recall': recall, 'positives_missed': sorted([*p, pos[p], best[p]] for p in pos if p not in paired),
              'calibration_retrieved': {kind: {f'{a} | {b}': (a, b) in paired for a, b in cal[kind]}
                                        for kind in ('exclude', 'keep', 'expected_exclude_unconfirmed')},
              'cf_canary_tasks': sorted(c['name'] for c in cf if c['canaries']),
              'tb_with_canary': sum(bool(t['canaries']) for t in tb)}
    (work / 'retrieval.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ('cf_canary_tasks', 'calibration_retrieved')}, indent=1))
    print('calibration exclude retrieved:', sum(report['calibration_retrieved']['exclude'].values()), '/', len(cal['exclude']))
    print('canary CF tasks:', len(report['cf_canary_tasks']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--k', type=int, default=50)
    a = ap.parse_args()
    main(a.work, a.k)

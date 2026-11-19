"""Decontaminate m5-omniscience problems against the 300 AA-Omniscience v1.1 public questions (once, at the end).

  python -m m5.omniscience.decon.run WORK --gen RUN/gen.jsonl [--gen ...] [--ds m5-ds] [--concurrency 150]

1. Retrieval: for each test item, the top-K candidates under word TF-IDF (1-2 grams), char 4-gram TF-IDF and
   bge-small-en-v1.5 embedding cosine, plus every candidate whose question contains the test item's gold answer
   (answers of >= 4 characters) while sharing >= 2 of the test question's rarest content words.
2. Judge: DeepSeek (rubric.md: SAME_FACT / SCAFFOLD / RELATED / UNRELATED), 2 samples per pair. A candidate is
   excluded when any sample says SAME_FACT or SCAFFOLD (#119's "any REUSE" rule; scaffold reuse counts).
   Deepening: a test item with any exclusion gets its window doubled once.
3. Canary: questions, and the traces' text if given, containing a benchmark canary string are excluded.
Outputs in WORK: pairs.jsonl (every judged pair with both verdicts), excluded.jsonl, summary.json.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from m5.omniscience import fsio
from m5.omniscience.gen.checks import norm, parse
from m5.omniscience.gen.pipeline import DS, Alloc

V11 = Path('/data/aai/aa_evals/data/omniscience/public-v1.1@1f9cbce/AA-Omniscience_dataset_public.csv')
RUBRIC = (Path(__file__).parent / 'rubric.md').read_text()
CANARY = re.compile(r'canary GUID|BENCHMARK DATA SHOULD NEVER APPEAR', re.I)
K = 20
BAD = {'SAME_FACT', 'SCAFFOLD'}


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def topk(S, k):
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return idx


def embed(texts, cache):
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            return E
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('/data/aai/models/bge-small-en-v1.5', device='cpu')  # plain dir, no hub locks
    E = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(cache, E)
    return E


def retrieve(test, cands, work, k):
    tq = [t['question'] for t in test]
    cq = [c['question'] for c in cands]
    pairs = collections.defaultdict(set)
    w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=1).fit(cq + tq)
    S = (w.transform(tq) @ w.transform(cq).T).toarray()
    for i, row in enumerate(topk(S, k)):
        for j in row:
            pairs[i].add((int(j), 'tfidf'))
    ch = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), sublinear_tf=True, min_df=2).fit(cq + tq)
    S2 = (ch.transform(tq) @ ch.transform(cq).T).toarray()
    for i, row in enumerate(topk(S2, k)):
        for j in row:
            pairs[i].add((int(j), 'char'))
    E = embed(tq + cq, work / f'bge_{len(cq)}.npy')
    S3 = E[:len(tq)] @ E[len(tq):].T
    for i, row in enumerate(topk(S3, k)):
        for j in row:
            pairs[i].add((int(j), 'embed'))
    df = collections.Counter(t for q in cq for t in set(norm(q).split()))
    for i, t in enumerate(test):
        a = norm(t['answer'])
        if len(a) < 4:
            continue
        rare = sorted(set(norm(t['question']).split()), key=lambda x: df.get(x, 0))[:6]
        for j, c in enumerate(cands):
            h = f" {norm(c['question'])} "
            if f' {a} ' in h and sum(f' {r} ' in h for r in rare) >= 2:
                pairs[i].add((j, 'answer'))
    return pairs, (S, S2, S3)


def judge_pair(ds, t, c, slot, stop):
    prompt = (RUBRIC.replace('{test_question}', t['question']).replace('{test_answer}', t['answer'])
              .replace('{cand_question}', c['question']).replace('{cand_answer}', c.get('answer_check_only', '')))
    errs = []
    resp = ds.chat([{'role': 'user', 'content': prompt}], slot, stop, errs)
    if not resp or resp.get('_error'):
        return {'label': 'ERROR'}
    content = resp['choices'][0]['message'].get('content') or ''
    m = re.search(r'"label"\s*:\s*"([A-Z_]+)"', content)
    d = parse(content) or {}
    return {'label': (d.get('label') or (m.group(1) if m else 'UNPARSED')), 'reason': str(d.get('reason', ''))[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work')
    ap.add_argument('--gen', nargs='+', required=True)
    ap.add_argument('--traces', nargs='*', default=[])
    ap.add_argument('--ds', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=150)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--k', type=int, default=K)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    test = list(csv.DictReader(open(V11, encoding='utf-8')))
    cands = []
    for g in a.gen:
        cands += [json.loads(l) for l in open(g)]
    cands = [c for c in cands if c.get('accepted')]
    log('candidates', len(cands), 'test', len(test))
    excluded = {}
    for c in cands:
        if CANARY.search(c['question']):
            excluded[c['problem_id']] = {'why': 'canary'}
    for tf in a.traces:
        for l in open(tf):
            r = json.loads(l)
            if CANARY.search(r.get('answer', '') + r.get('reasoning', '')):
                excluded.setdefault(r['problem_id'], {'why': 'canary_in_trace'})
    ds = DS(a.ds, Alloc(a.ds))
    stop = threading.Event()
    done = {}
    pf = work / 'pairs.jsonl'
    if pf.exists():
        for l in open(pf):
            r = json.loads(l)
            done[(r['test_id'], r['problem_id'])] = r
    lock = threading.Lock()

    def run_pairs(todo):
        def one(n, i, j, sig):
            t, c = test[i], cands[j]
            key = (t['question_id'], c['problem_id'])
            if key in done:
                return done[key]
            vs = [judge_pair(ds, t, c, n * a.samples + s, stop) for s in range(a.samples)]
            r = {'test_id': t['question_id'], 'problem_id': c['problem_id'], 'signals': sorted(sig),
                 'labels': [v['label'] for v in vs], 'reasons': [v.get('reason') for v in vs],
                 'test_question': t['question'], 'cand_question': c['question']}
            with lock:
                done[key] = r
                fsio.append_line(pf, json.dumps(r, ensure_ascii=False))
            return r
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            futs = [ex.submit(one, n, i, j, sig) for n, (i, j, sig) in enumerate(todo)]
            for k_, f in enumerate(cf.as_completed(futs)):
                f.result()
                if k_ % 500 == 0:
                    log('judged', k_, '/', len(futs))

    k = a.k
    pairs, _ = retrieve(test, cands, work, k)
    todo = {}
    for i, ps in pairs.items():
        for j, sig in ps:
            todo.setdefault((i, j), set()).add(sig)
    log('pairs', len(todo))
    run_pairs([(i, j, s) for (i, j), s in todo.items()])
    # deepening: double the window for test items with any exclusion
    hot = {i for i, t in enumerate(test) if any(
        done.get((t['question_id'], cands[j]['problem_id']), {}).get('labels') and
        set(done[(t['question_id'], cands[j]['problem_id'])]['labels']) & BAD for j, _ in pairs[i])}
    if hot:
        pairs2, _ = retrieve(test, cands, work, 2 * k)
        extra = {}
        for i in hot:
            for j, sig in pairs2[i]:
                if (i, j) not in todo:
                    extra.setdefault((i, j), set()).add(sig)
        log('deepening', len(hot), 'items', len(extra), 'pairs')
        run_pairs([(i, j, s) for (i, j), s in extra.items()])
    for (tid, pid), r in done.items():
        if set(r['labels']) & BAD:
            excluded.setdefault(pid, {'why': 'judge', 'test_ids': []})
            excluded[pid].setdefault('test_ids', []).append(tid)
    fsio.write_atomic(work / 'excluded.jsonl',
                      ''.join(json.dumps({'problem_id': pid, **why}) + '\n' for pid, why in excluded.items()))
    lab = collections.Counter(l for r in done.values() for l in r['labels'])
    summ = {'candidates': len(cands), 'pairs_judged': len(done), 'labels': dict(lab), 'excluded': len(excluded),
            'excluded_rate': round(len(excluded) / max(1, len(cands)), 5),
            'test_items_with_exclusions': len({t for r in done.values() if set(r['labels']) & BAD
                                               for t in [r['test_id']]}),
            'by_reason': dict(collections.Counter(v['why'] for v in excluded.values()))}
    fsio.write_atomic(work / 'summary.json', json.dumps(summ, indent=1))
    log(json.dumps(summ))


if __name__ == '__main__':
    main()

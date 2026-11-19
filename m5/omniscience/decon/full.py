"""Final decontamination of the m5-omniscience problems in an SFT mixture against the AA-Omniscience items c30 scores
(the 300 public v1.1 questions, aa_evals.omniscience.data 'v1.1'; c30 plan cheap_full_v1: all 300, no id filter).

    # CPU job (m5-omniscience venv): retrieval over every mixture problem -> retrieved.jsonl, audit.jsonl, retrieval.json
    python -m m5.omniscience.decon.full retrieve WORK --mixture SEQUENCES.csv --gen production/gen.jsonl
    # CPU job (tokens of the mixture itself): canary markers and verbatim test text in the trained sequences
    python -m m5.omniscience.decon.scan WORK --data M5L_DATA_DIR
    # CPU job: DeepSeek judge, 2 samples per pair -> pairs.jsonl; excluded.json
    python -m m5.omniscience.decon.full judge WORK [--ds m5-ds] [--concurrency 400]

Candidates: every problem with arm == omniscience in the mixture's sequences.csv (group = omniscience:<problem id>),
looked up in gen.jsonl (all must be found), plus planted positive controls (controls.json: verbatim copies,
paraphrases, reverse-direction and answer-revealing rewrites and scaffold reuses of real test items).
Signals, each over the full 300 x N similarity matrix (every candidate is scored against every test item):
  tfidf  word 1-2 gram TF-IDF cosine (questions)        char  char_wb 4-gram TF-IDF cosine (questions)
  embq   bge-small-en-v1.5 cosine of the questions      embqa bge cosine of "question + answer" (catches reversals)
  ngram  a shared word 6-gram (>= 3 content words) in <= 20 candidate questions  answer gold-answer matching: the test answer inside the
         candidate question, equal answers, or the candidate's answer inside the test question, each with >= 2 (or 1
         for a long answer) of the test question's rarest content words (each in <= 0.2% of the candidate
         questions) in the candidate question.
Windows: each test item's top-K candidates per similarity signal (ranks kept to 4K for deepening), each candidate's
top-1 test item per similarity signal when it is in the top --cand-frac of candidates, and every ngram/answer hit.
Audit: a random sample of mixture candidates outside every window, judged against their nearest test item (embq),
measures what the windows miss.
Trace scan (decon.scan): canary hits are excluded outright; each verbatim test-text hit in the trained text is judged
as a pair whose candidate carries the overlapping trace excerpt.
Judge: rubric.md (SAME_FACT / SCAFFOLD / RELATED / UNRELATED), 2 samples per pair; a candidate is excluded when any
sample says SAME_FACT or SCAFFOLD. Deepening: a test item with an exclusion among mixture candidates gets its window
doubled (2K), and doubled again (4K) if the 2K ring added exclusions.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from m5.omniscience import fsio
from m5.omniscience.decon.run import BAD, CANARY, V11, judge_pair
from m5.omniscience.gen.checks import norm

HERE = Path(__file__).parent
CONTROLS = HERE / 'controls.json'
SIM = ('tfidf', 'char', 'embq', 'embqa')
VALID = {'SAME_FACT', 'SCAFFOLD', 'RELATED', 'UNRELATED'}
EMBED_MODEL = '/data/aai/models/bge-small-en-v1.5'


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def code_sha():
    for p in (HERE.parents[2] / 'FROZEN_SHA',):
        if p.exists():
            return p.read_text().strip()
    try:
        return subprocess.run(['git', '-C', str(HERE), 'rev-parse', '--short=10', 'HEAD'], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ''


def load_test():
    rows = list(csv.DictReader(open(V11, encoding='utf-8')))
    assert len(rows) == 300, len(rows)
    return [{'id': str(r['question_id']), 'question': r['question'], 'answer': r['answer'], 'domain': r['domain']}
            for r in rows]


def mixture_ids(seq_csv, arm):
    ids = set()
    for r in csv.DictReader(open(seq_csv)):
        if r['arm'] == arm:
            ids.add(r['group'].split(':', 1)[1])
    return ids


def load_cands(seq_csv, arm, gens):
    want = mixture_ids(seq_csv, arm)
    got = {}
    for g in gens:
        for line in open(g):
            d = json.loads(line)
            pid = d.get('problem_id')
            if pid in want and pid not in got and d.get('question'):
                got[pid] = {'id': pid, 'question': d['question'], 'answer': d.get('answer_check_only') or '',
                            'control': None}
    missing = want - set(got)
    if missing:
        raise SystemExit(f'{len(missing)} mixture problems not found in {gens}: {sorted(missing)[:5]}')
    cands = [got[i] for i in sorted(got)]
    controls = json.loads(CONTROLS.read_text())
    for c in controls:
        cands.append({'id': c['id'], 'question': c['question'], 'answer': c['answer'],
                      'control': {'test_id': c['test_id'], 'kind': c['kind']}})
    return cands, len(want)


def embed(texts, cache):
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            return E
    os.environ['HF_HOME'] = str(Path(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir()) / 'hf')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS') or len(os.sched_getaffinity(0))))
    m = SentenceTransformer(EMBED_MODEL, device='cpu')
    t0 = time.time()
    E = m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    log('embedded', len(texts), f'{time.time() - t0:.0f}s')
    np.save(cache, E)
    return E


STOP = None


def content_words(q):
    global STOP
    if STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP = set(ENGLISH_STOP_WORDS)
    return [w for w in norm(q).split() if len(w) >= 3 and w not in STOP]


def grams(words, n=6):
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def retrieve(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    test = load_test()
    cands, n_mix = load_cands(a.mixture, a.arm, a.gen)
    N, T = len(cands), len(test)
    log('test', T, 'candidates', N, '(mixture', n_mix, '+ controls', N - n_mix, ')')
    tq, cq = [t['question'] for t in test], [c['question'] for c in cands]
    sims = {}
    w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=1,
                        dtype=np.float32).fit(cq + tq)
    sims['tfidf'] = (w.transform(tq) @ w.transform(cq).T).toarray().astype(np.float32)
    ch = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), sublinear_tf=True, min_df=2,
                         dtype=np.float32).fit(cq + tq)
    sims['char'] = (ch.transform(tq) @ ch.transform(cq).T).toarray().astype(np.float32)
    log('tfidf done')
    # the embedding caches are keyed by the exact text list (controls change it)
    key = hashlib.sha1('\n'.join(tq + cq).encode()).hexdigest()[:10]
    E = embed(tq + cq, work / f'bge_q_{N}_{key}.npy')
    sims['embq'] = E[:T] @ E[T:].T
    qa = [f"{x['question']}\nAnswer: {x['answer']}" for x in test + cands]
    E2 = embed(qa, work / f'bge_qa_{N}_{key}.npy')
    sims['embqa'] = E2[:T] @ E2[T:].T
    del E, E2
    log('embeddings done')
    K4 = 4 * a.k
    pairs = collections.defaultdict(dict)   # (i, j) -> {signal: rank or detail}
    for s in SIM:
        S = sims[s]
        top = np.argpartition(-S, K4 - 1, axis=1)[:, :K4]
        for i in range(T):
            row = top[i][np.argsort(-S[i, top[i]], kind='stable')]
            for r, j in enumerate(row):
                pairs[(i, int(j))][s] = r
        # candidate side: each candidate's nearest test item, when that similarity is in the top cand_frac
        best_i, best_v = S.argmax(axis=0), S.max(axis=0)
        floor = float(np.quantile(best_v[:n_mix], 1 - a.cand_frac))
        for j in np.nonzero(best_v >= floor)[0]:
            pairs[(int(best_i[j]), int(j))][f'{s}_c'] = round(float(best_v[j]), 4)
        sims[s + '_floor'] = floor
    log('windows done', len(pairs))
    # rare 6-grams and gold-answer matching
    cw = [content_words(q) for q in cq]
    df = collections.Counter(x for ws in cw for x in set(ws))
    tw = [content_words(t['question']) for t in test]
    rare_max = max(20, int(a.rare_frac * N))   # a 'rare' word is in at most this many candidate questions
    rare = [sorted({x for x in ws if 1 <= df.get(x, 0) <= rare_max}, key=lambda x: (df[x], x))[:8] for ws in tw]
    nq = [f" {norm(q)} " for q in cq]
    na = [norm(c['answer']) for c in cands]
    tnq = [f" {norm(t['question'])} " for t in test]

    def n_rare(i, j):
        return sum(f' {r} ' in nq[j] for r in rare[i])

    tg = collections.defaultdict(set)
    for i, t in enumerate(test):
        for g in grams(norm(t['question']).split()):
            tg[g].add(i)
    gdf, hits = collections.Counter(), collections.defaultdict(set)
    for j, q in enumerate(cq):
        ws = norm(q).split()
        for g in grams(ws):
            if g in tg:
                gdf[g] += 1
                hits[g].add(j)
    stopish = 0
    for g, js in hits.items():
        if gdf[g] > a.ngram_max_df or len(content_words(g)) < 3:
            stopish += 1
            continue
        for i in tg[g]:
            for j in js:
                pairs[(i, j)]['ngram'] = g
    by_ans = collections.defaultdict(list)
    for j, x in enumerate(na):
        if x:
            by_ans[x].append(j)
    n_ans = collections.Counter()
    for i, t in enumerate(test):
        ta = norm(t['answer'])
        if not ta:
            continue
        need = 1 if (len(ta) >= 8 and re.search('[a-z]', ta)) else 2
        if len(ta) >= 4:
            pat = f' {ta} '
            for j in range(N):
                if pat in nq[j] and n_rare(i, j) >= need:
                    pairs[(i, j)]['answer_in_q'] = ta
                    n_ans['answer_in_q'] += 1
        for j in by_ans.get(ta, ()):
            if n_rare(i, j) >= need:
                pairs[(i, j)]['answer_eq'] = ta
                n_ans['answer_eq'] += 1
    for j, x in enumerate(na):
        if len(x) < 5 or not re.search('[a-z]', x):
            continue
        pat = f' {x} '
        for i in range(T):
            if pat in tnq[i] and n_rare(i, j) >= 2:
                pairs[(i, j)]['answer_rev'] = x
                n_ans['answer_rev'] += 1
    log('ngram/answer done', dict(n_ans))
    # write all retrieved pairs (ranks up to 4K) with the four similarities
    out = []
    for (i, j), sig in pairs.items():
        out.append({'test_id': test[i]['id'], 'problem_id': cands[j]['id'], 'signals': sig,
                    'sims': {s: round(float(sims[s][i, j]), 4) for s in SIM},
                    'control': cands[j]['control']})
    fsio.write_atomic(work / 'retrieved.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    # audit: random mixture candidates outside every first-pass window, vs their nearest test item (embq)
    k = a.k
    first = {(i, j) for (i, j), sig in pairs.items() if in_first_pass(sig, k)}
    covered = {j for _, j in first}
    rng = random.Random(a.seed)
    pool = [j for j in range(n_mix) if j not in covered]
    sample = rng.sample(pool, min(a.audit, len(pool)))
    best = sims['embq'].argmax(axis=0)
    audit = [{'test_id': test[int(best[j])]['id'], 'problem_id': cands[j]['id'], 'signals': {'audit': 1},
              'sims': {s: round(float(sims[s][int(best[j]), j]), 4) for s in SIM}, 'control': None} for j in sample]
    fsio.write_atomic(work / 'audit.jsonl', ''.join(json.dumps(r) + '\n' for r in audit))
    # canary markers in the candidate questions (the trained text is scanned by decon.scan)
    canary = [c['id'] for c in cands[:n_mix] if CANARY.search(c['question'])]
    mix_best = {s: sims[s][:, :n_mix].max(axis=0) for s in SIM}
    ctrl = {}
    for j in range(n_mix, N):
        c = cands[j]
        i = next(ii for ii, t in enumerate(test) if t['id'] == c['control']['test_id'])
        sig = pairs.get((i, j), {})
        ctrl[c['id']] = {**c['control'], 'retrieved_first_pass': in_first_pass(sig, k), 'signals': sig,
                         'sims': {s: round(float(sims[s][i, j]), 4) for s in SIM}}
    meta = {'test_set': str(V11), 'test_items': T, 'mixture': a.mixture, 'arm': a.arm, 'gen': a.gen,
            'mixture_problems': n_mix, 'controls': N - n_mix, 'scored_against_all_test_items': n_mix,
            'k': k, 'k_max': K4, 'cand_frac': a.cand_frac, 'ngram_max_df': a.ngram_max_df, 'rare_max_df': rare_max,
            'cand_floors': {s: round(sims[s + '_floor'], 4) for s in SIM},
            'pairs_retrieved_upto_4k': len(pairs), 'first_pass_pairs': len(first),
            'first_pass_mixture_candidates': sum(1 for j in covered if j < n_mix), 'answer_hits': dict(n_ans),
            'ngram_common_skipped': stopish, 'audit_pairs': len(audit), 'canary_in_questions': canary,
            'nearest_test_sim_quantiles': {s: {q: round(float(np.quantile(v, q)), 4) for q in (0.5, 0.9, 0.99, 0.999)}
                                           for s, v in mix_best.items()},
            'controls': ctrl, 'code': code_sha(), 'wall_s': round(time.time() - t_start)}
    fsio.write_atomic(work / 'retrieval.json', json.dumps(meta, indent=1))
    log('retrieval written', json.dumps({k_: v for k_, v in meta.items() if k_ != 'controls'}))
    log('controls retrieved', sum(v['retrieved_first_pass'] for v in ctrl.values()), '/', len(ctrl))


def in_first_pass(sig, k):
    return any((s in SIM and r < k) or s.endswith('_c') or s in ('ngram', 'answer_in_q', 'answer_eq', 'answer_rev')
               for s, r in sig.items())


def ring(sig, lo, hi):
    """True when the pair's best test-side rank is in [lo, hi)."""
    r = min((v for s, v in sig.items() if s in SIM), default=None)
    return r is not None and lo <= r < hi


def kf(r):
    """A pair's key: trace pairs (scan hits, judged with the trace excerpt) are separate from question pairs."""
    return r['test_id'], r['problem_id'] + ('#trace' if r.get('kind') == 'trace' else '')


TRACE_NOTE = ("\n\n[This candidate's training trace (the teacher's reasoning and answer, trained on together with the "
              "question) contains text that overlaps the test item verbatim. Excerpt of the trained text around the "
              "overlap:]\n<<<\n{excerpt}\n>>>")


def judge(a):
    from m5.omniscience.gen.pipeline import DS
    work = Path(a.work)
    meta = json.loads((work / 'retrieval.json').read_text())
    test = {t['id']: t for t in load_test()}
    cands = {}
    gen_ids = set()
    for r in map(json.loads, open(work / 'retrieved.jsonl')):
        gen_ids.add(r['problem_id'])
    audit = [json.loads(l) for l in open(work / 'audit.jsonl')]
    gen_ids |= {r['problem_id'] for r in audit}
    scan = json.loads((work / 'scan.json').read_text()) if (work / 'scan.json').exists() else {}
    traces = [{'test_id': h['test_id'], 'problem_id': h['problem_id'], 'kind': 'trace', 'excerpt': h['excerpt'],
               'signals': {'trace_text': h['grams']}, 'sims': {}, 'control': None}
              for h in scan.get('test_text_hits', [])]
    gen_ids |= {r['problem_id'] for r in traces}
    for c in json.loads(CONTROLS.read_text()):
        cands[c['id']] = {'question': c['question'], 'answer_check_only': c['answer']}
    for g in meta['gen']:
        for line in open(g):
            d = json.loads(line)
            if d.get('problem_id') in gen_ids and d['problem_id'] not in cands:
                cands[d['problem_id']] = {'question': d['question'], 'answer_check_only': d.get('answer_check_only', '')}
    retrieved = [json.loads(l) for l in open(work / 'retrieved.jsonl')]
    k = meta['k']
    ds = DS(a.ds, None)   # pinned to one server; alloc.json is not consulted
    stop = threading.Event()
    pf = work / 'pairs.jsonl'
    done = {}
    if pf.exists():
        for line in open(pf):
            r = json.loads(line)
            if all(l in VALID for l in r['labels']):
                done[kf(r)] = r
    lock = threading.Lock()
    stats = collections.Counter()
    t_judge = [0.0]

    def judge_one(n, r, stage):
        key = kf(r)
        if key in done:
            return done[key]
        t, c = test[r['test_id']], cands[r['problem_id']]
        if r.get('kind') == 'trace':
            c = {**c, 'question': c['question'] + TRACE_NOTE.format(excerpt=r['excerpt'])}
        labels, reasons = [], []
        for s in range(a.samples):
            for attempt in range(a.retries):
                v = judge_pair(ds, t, c, n * a.samples + s + attempt, stop)
                with lock:
                    stats['requests'] += 1
                if v['label'] in VALID:
                    break
                with lock:
                    stats['retries'] += 1
            labels.append(v['label'])
            reasons.append(v.get('reason'))
        out = {'test_id': r['test_id'], 'problem_id': r['problem_id'], 'kind': r.get('kind', 'question'),
               'stage': stage, 'signals': r['signals'],
               'sims': r['sims'], 'control': r['control'], 'labels': labels, 'reasons': reasons,
               'test_question': t['question'], 'test_answer': t['answer'], 'cand_question': c['question'],
               'cand_answer': c['answer_check_only']}
        with lock:
            done[key] = out
            fsio.append_line(pf, json.dumps(out, ensure_ascii=False))
        return out

    def run(todo, stage):
        todo = [r for r in todo if kf(r) not in done]
        log(stage, 'to judge', len(todo))
        if not todo:
            return
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            futs = [ex.submit(judge_one, n, r, stage) for n, r in enumerate(todo)]
            for m, f in enumerate(cf.as_completed(futs)):
                f.result()
                if m % 1000 == 0 or m == len(futs) - 1:
                    el = time.time() - t0
                    log(stage, 'judged', m + 1, '/', len(futs), f'{el:.0f}s', f'{(m + 1) / max(el, 1):.1f} pairs/s',
                        'requests', stats['requests'], 'retries', stats['retries'])
        t_judge[0] += time.time() - t0

    def bad(r):
        return bool(set(r['labels']) & BAD)

    def hot_items(rows):
        return {r['test_id'] for r in rows if r['control'] is None and bad(r)}

    t_all = time.time()
    first = [r for r in retrieved if in_first_pass(r['signals'], k)]
    first_keys = {kf(r) for r in first}
    run(first, 'w1')
    deep = []
    hot = hot_items(done[kf(r)] for r in first)
    lo, hi = k, 2 * k
    while hot and hi <= meta['k_max']:
        ringp = [r for r in retrieved if r['test_id'] in hot and r['control'] is None and ring(r['signals'], lo, hi)
                 and kf(r) not in first_keys]
        run(ringp, f'deep{hi}')
        new_hot = hot_items(done[kf(r)] for r in ringp if kf(r) in done)
        deep.append({'window': hi, 'items': len(hot), 'pairs': len(ringp), 'items_with_new_exclusions': len(new_hot),
                     'new_excluding_pairs': sum(1 for r in ringp if kf(r) in done and bad(done[kf(r)]))})
        hot, lo, hi = new_hot, hi, 2 * hi
    run(audit, 'audit')
    run(traces, 'trace')
    finalize(a, work, meta, done, deep, stats, t_judge[0], time.time() - t_all)


def finalize(a, work, meta, done, deep, stats, judge_s, total_s):
    rows = list(done.values())
    bad = [r for r in rows if set(r['labels']) & BAD]
    real_bad = [r for r in bad if r['control'] is None]
    excluded = {}
    for r in real_bad:
        e = excluded.setdefault(r['problem_id'], {'why': 'judge' if r.get('kind') != 'trace' else 'judge_trace',
                                                  'test_ids': []})
        e['test_ids'].append(r['test_id'])
    for pid in meta['canary_in_questions']:
        excluded.setdefault(pid, {'why': 'canary_question'})
    scan = {}
    if (work / 'scan.json').exists():
        scan = json.loads((work / 'scan.json').read_text())
        for pid in scan.get('canary_problems', []):
            excluded.setdefault(pid, {'why': 'canary_trace'})
    n_mix = meta['mixture_problems']
    ctrl = {}
    for cid, c in meta['controls'].items():
        r = done.get((c['test_id'], cid))
        ctrl[cid] = {'kind': c['kind'], 'test_id': c['test_id'], 'retrieved': c['retrieved_first_pass'],
                     'labels': r['labels'] if r else None, 'caught': bool(r and set(r['labels']) & BAD)}
    by_kind = collections.defaultdict(lambda: [0, 0])
    for c in ctrl.values():
        by_kind[c['kind']][0] += c['caught']
        by_kind[c['kind']][1] += 1
    real = [r for r in rows if r['control'] is None]
    k = meta['k']

    def window_signals(sig):
        # the signals that put a pair in the first-pass window ('deepening' / 'audit' otherwise)
        return {s for s, v in sig.items() if (s in SIM and v < k) or s.endswith('_c') or s.startswith(('ngram', 'answer'))}

    sig_first, only = collections.Counter(), collections.Counter()
    for r in real_bad:
        names = window_signals(r['signals']) or {r['stage']}
        for s in names:
            sig_first[s] += 1
        if len(names) == 1:
            only[next(iter(names))] += 1
    audit_rows = [r for r in rows if r['stage'] == 'audit']
    trace_rows = [r for r in rows if r['stage'] == 'trace']
    labels = collections.Counter(l for r in real for l in r['labels'])
    report = {
        'test_set': {'path': meta['test_set'], 'items': meta['test_items'],
                     'note': 'c30 cheap_full_v1 scores all 300 AA-Omniscience public v1.1 questions'},
        'mixture': meta['mixture'], 'mixture_problems': n_mix,
        'covered_by_retrieval': meta['scored_against_all_test_items'],
        'mixture_problems_judged': len({r['problem_id'] for r in real}),
        'pairs_judged': len(real), 'judge_samples': sum(len(r['labels']) for r in real),
        'requests_incl_controls_and_retries': stats['requests'], 'retries': stats['retries'],
        'labels': dict(labels),
        'unresolved_pairs': sum(1 for r in real if not all(l in VALID for l in r['labels'])),
        'excluded': len(excluded), 'excluded_pct': round(100 * len(excluded) / n_mix, 4),
        'excluded_by_reason': dict(collections.Counter(v['why'] for v in excluded.values())),
        'excluding_pairs': len(real_bad),
        'excluding_pairs_both_samples': sum(1 for r in real_bad if all(l in BAD for l in r['labels'])),
        'test_items_with_exclusions': len({r['test_id'] for r in real_bad}),
        'excluding_pairs_by_signal': dict(sig_first), 'excluding_pairs_single_signal': dict(only),
        'deepening': deep,
        'audit': {'pairs': len(audit_rows), 'labels': dict(collections.Counter(l for r in audit_rows for l in r['labels'])),
                  'excluded': sum(1 for r in audit_rows if set(r['labels']) & BAD)},
        'controls': {'n': len(ctrl), 'retrieved': sum(c['retrieved'] for c in ctrl.values()),
                     'caught': sum(c['caught'] for c in ctrl.values()),
                     'recall': round(sum(c['caught'] for c in ctrl.values()) / max(1, len(ctrl)), 4),
                     'by_kind': {k_: f'{v[0]}/{v[1]}' for k_, v in by_kind.items()}, 'detail': ctrl},
        'canary': {'questions': len(meta['canary_in_questions']), 'scan': {k_: v for k_, v in scan.items()
                                                                          if k_ not in ('test_text_hits',)}},
        'trace_text_hits': {'pairs': len(trace_rows), 'excluded': sum(1 for r in trace_rows if set(r['labels']) & BAD),
                            'detail': [{'test_id': r['test_id'], 'problem_id': r['problem_id'],
                                        'grams': r['signals'].get('trace_text'), 'labels': r['labels'],
                                        'reasons': r['reasons']} for r in trace_rows]},
        'retrieval': {k_: meta[k_] for k_ in ('k', 'k_max', 'cand_frac', 'cand_floors', 'pairs_retrieved_upto_4k',
                                               'first_pass_pairs', 'first_pass_mixture_candidates', 'answer_hits',
                                               'nearest_test_sim_quantiles')},
        'judge': {'server': a.ds, 'samples': a.samples, 'concurrency': a.concurrency, 'rule': 'exclude on any '
                  'SAME_FACT or SCAFFOLD sample', 'wall_s_judging': round(judge_s), 'wall_s_total': round(total_s)},
        'code': {'retrieve': meta['code'], 'judge': code_sha()},
        'finished': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    }
    fsio.write_atomic(work / 'excluded.json', json.dumps({'excluded': sorted(excluded), 'report': report}, indent=1))
    fsio.write_atomic(work / 'excluded_detail.jsonl',
                      ''.join(json.dumps({'problem_id': p, **v}) + '\n' for p, v in sorted(excluded.items())))
    log('final', json.dumps({k_: v for k_, v in report.items() if k_ not in ('controls',)})[:3000])
    log('controls', json.dumps(report['controls']['by_kind']), 'recall', report['controls']['recall'])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('retrieve')
    r.add_argument('work')
    r.add_argument('--mixture', required=True, help="the mixture's sequences.csv")
    r.add_argument('--arm', default='omniscience')
    r.add_argument('--gen', nargs='+', required=True)
    r.add_argument('--k', type=int, default=25)
    r.add_argument('--cand-frac', type=float, default=0.02)
    r.add_argument('--ngram-max-df', type=int, default=20)
    r.add_argument('--rare-frac', type=float, default=0.002)
    r.add_argument('--audit', type=int, default=1000)
    r.add_argument('--seed', type=int, default=354)
    j = sub.add_parser('judge')
    j.add_argument('work')
    j.add_argument('--ds', default='m5-ds')
    j.add_argument('--concurrency', type=int, default=400, help='pairs in flight (= requests in flight)')
    j.add_argument('--samples', type=int, default=2)
    j.add_argument('--retries', type=int, default=4)
    a = ap.parse_args()
    {'retrieve': retrieve, 'judge': judge}[a.cmd](a)


if __name__ == '__main__':
    main()

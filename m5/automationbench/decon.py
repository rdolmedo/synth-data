"""Decontamination of synthetic tasks against AutomationBench's 600 scored tasks (#359; #119 recipe), once at the end.

  # AutomationBench venv: texts of both sides
  python -m m5.automationbench.decon prepare --accepted 'GLOB' --work W [--examples]
  # CPU job, a venv with sentence-transformers + sklearn: retrieval -> W/pairs.jsonl
  python -m m5.automationbench.decon retrieve --work W [--k 10]
  # AutomationBench venv, DeepSeek judge, 2 samples per pair -> W/verdicts.jsonl, W/excluded.json
  python -m m5.automationbench.decon judge --work W

Signals: word 8-gram containment of the candidate's request in the test request (closing boilerplate sentences
stripped on both sides), TF-IDF (word 1-2 grams) cosine of the requests, bge-small-en-v1.5 cosine of request + data
skeleton (apps, sheet/worksheet titles, channels, subjects, record names), and copied distinctive strings (people,
companies, titles, IDs that occur in at most 2 scored tasks) shared with the test task. Windows: each test task's
top-K candidates per signal, plus every candidate's top-1 test task per signal above a floor. Judge:
rubric_reuse_ab.md (scaffold reuse counts), 2 samples; excluded on any REUSE or when both samples say WEAK_REUSE.
`--examples` prepares the 200 simple tasks (the generator's format examples) as the test side instead: the pilot's
copying check.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BGE = '/data/aai/m5/automationbench/sources/bge-small-en-v1.5'   # BAAI/bge-small-en-v1.5 @5c38ec7, MIT

HERE = Path(__file__).resolve().parent
NAME_KEYS = ('name', 'first_name', 'last_name', 'full_name', 'title', 'subject', 'company', 'company_name',
             'display_name', 'real_name', 'email', 'from_', 'to', 'id', 'account_name', 'vendor', 'customer')


def strings_of(state: dict) -> set[str]:
    """Distinctive-looking strings of a state: values of name/title/subject/email/id fields and sheet cells named so."""
    out = set()

    def walk(o, key=None):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for x in o:
                walk(x, key)
        elif isinstance(o, str) and key is not None:
            k = str(key).lower()
            if (k in NAME_KEYS or k.endswith('_id') or k.endswith('name') or k.endswith('email')) and 4 <= len(o) <= 80:
                out.add(o.strip().lower())
    walk({k: v for k, v in state.items() if k != 'meta'})
    return out


def skeleton(state: dict) -> str:
    parts = ['apps: ' + ', '.join(sorted(k for k in state if k != 'meta'))]
    gs = state.get('google_sheets') or {}
    for ss in gs.get('spreadsheets', []) or []:
        parts.append('sheet ' + str(ss.get('title')) + ': ' + ', '.join(str(w.get('title')) for w in ss.get('worksheets', []) or []))
    for c in (state.get('slack') or {}).get('channels', []) or []:
        parts.append('#' + str(c.get('name')))
    for m in ((state.get('gmail') or {}).get('messages', []) or [])[:20]:
        parts.append('email: ' + str(m.get('subject')))
    return '\n'.join(parts)[:3000]


def compact(state: dict, limit=6000) -> str:
    s = json.dumps({k: v for k, v in state.items() if k != 'meta'}, separators=(',', ':'))
    return s if len(s) <= limit else s[:limit] + ' ...[truncated]'


def prepare(a):
    from m5.automationbench.taxonomy import strip_suffixes
    from m5.automationbench.upstream import DOMAINS, load
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    test = []
    for d in (['simple'] if a.examples else DOMAINS):
        for r in load(d, noise=False):
            st = r['info']['initial_state']
            test.append({'id': r['task_name'], 'domain': d, 'trigger': strip_suffixes(r['trigger'])[0],
                         'skeleton': skeleton(st), 'strings': sorted(strings_of(st)), 'state': compact(st)})
    cands = []
    for p in sorted(glob.glob(a.accepted)):
        for line in open(p):
            rec = json.loads(line)
            t = rec['task']
            trig = [m['content'] for m in t['prompt'] if m['role'] == 'user'][0]
            # the generator's own data (before upstream's noise, which is shared boilerplate on both sides)
            own = {k: v for k, v in t['info']['initial_state'].items()}
            st = rec.get('own_state') or own
            cands.append({'id': t['info']['task_name'], 'domain': t['info']['task_name'].split('.')[0],
                          'trigger': strip_suffixes(trig)[0], 'skeleton': skeleton(st),
                          'strings': sorted(strings_of(st)), 'state': compact(st)})
    (work / 'test.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in test))
    (work / 'cands.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in cands))
    print(len(test), 'test,', len(cands), 'candidates')


def grams(t, n=8):
    w = re.findall(r'\w+', t.lower())
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def retrieve(a):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    work = Path(a.work)
    test = [json.loads(x) for x in open(work / 'test.jsonl')]
    cands = [json.loads(x) for x in open(work / 'cands.jsonl')]
    T, C = [t['trigger'] for t in test], [c['trigger'] for c in cands]
    S = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words='english').fit(T + C)
    S['tfidf'] = (vec.transform(T) @ vec.transform(C).T).toarray().astype(np.float32)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(BGE, device='cpu')
    ET = m.encode([t['trigger'] + '\n' + t['skeleton'] for t in test], batch_size=64, normalize_embeddings=True)
    EC = m.encode([c['trigger'] + '\n' + c['skeleton'] for c in cands], batch_size=64, normalize_embeddings=True)
    S['embed'] = (ET @ EC.T).astype(np.float32)
    inv = defaultdict(list)
    for i, t in enumerate(T):
        for g in grams(t):
            inv[g].append(i)
    ng = np.zeros((len(T), len(C)), dtype=np.float32)
    for j, c in enumerate(C):
        gs = grams(c)
        cnt = Counter(i for g in gs for i in inv.get(g, ()))
        for i, k in cnt.items():
            ng[i, j] = k / len(gs)
    S['ngram8'] = ng
    # copied distinctive strings: strings of a test task that occur in at most 2 test tasks
    # specific-looking strings only (a digit, an email, or 2+ words), rare on both sides: generic titles, worksheet
    # ids and common names are shared by chance (pilot preview: 'operations manager', 'ws_routing', 'maya chen')
    df = Counter(s for t in test for s in set(t['strings']))
    dfc = Counter(s for c in cands for s in set(c['strings']))
    def specific(s):
        return (bool(re.search(r'\d', s)) and len(s) >= 6) or '@' in s or len(s.split()) >= 3
    inv_s = defaultdict(list)
    for i, t in enumerate(test):
        for s in t['strings']:
            if df[s] <= 2 and dfc[s] <= 3 and specific(s) \
                    and not re.fullmatch(r'[\w.+-]*@?(company|example)\.example\.com|\d+', s):
                inv_s[s].append(i)
    cp = np.zeros((len(T), len(C)), dtype=np.float32)
    shared = defaultdict(list)
    for j, c in enumerate(cands):
        for s in c['strings']:
            for i in inv_s.get(s, ()):
                cp[i, j] += 1
                if len(shared[(i, j)]) < 10:
                    shared[(i, j)].append(s)
    S['copied'] = cp
    floors = {'tfidf': 0.35, 'embed': 0.85, 'ngram8': 0.05, 'copied': 2}
    pairs = defaultdict(set)
    for name, M in S.items():
        k = min(a.k, M.shape[1])
        top = np.argpartition(-M, k - 1, axis=1)[:, :k]
        for i in range(M.shape[0]):
            for j in top[i]:
                if M[i, j] > 0:
                    pairs[(i, int(j))].add(name)
        best = M.argmax(axis=0)
        for j in range(M.shape[1]):
            i = int(best[j])
            if M[i, j] >= floors[name]:
                pairs[(i, j)].add(name)
    with open(work / 'pairs.jsonl', 'w') as f:
        for (i, j), win in pairs.items():
            f.write(json.dumps({'test_id': test[i]['id'], 'cand_id': cands[j]['id'],
                                'signals': {n: round(float(S[n][i, j]), 4) for n in S},
                                'shared_strings': shared.get((i, j), []), 'window': sorted(win)}) + '\n')
    summary = {'n_test': len(test), 'n_cand': len(cands), 'n_pairs': len(pairs),
               'max': {n: float(M.max()) for n, M in S.items()},
               'cand_over_floor': {n: int((S[n].max(axis=0) >= floors[n]).sum()) for n in S},
               'ngram8_any': int((S['ngram8'].max(axis=0) > 0).sum()),
               'ngram8_mean_max': float(S['ngram8'].max(axis=0).mean()),
               'copied_any': int((S['copied'].max(axis=0) > 0).sum())}
    (work / 'retrieve.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), flush=True)


def judge_prompt(rubric, t, c, shared):
    return (f'{rubric}\n\n## Test task\nRequest: {t["trigger"]}\nStarting data: {t["state"]}\n\n'
            f'## Candidate task\nRequest: {c["trigger"]}\nStarting data: {c["state"]}\n\n'
            f'Strings found in both: {shared or "none"}\n')


def verdict_of(text: str) -> str | None:
    m = re.findall(r'VERDICT:\s*(REUSE|WEAK_REUSE|NO_REUSE)', text or '')
    return m[-1] if m else None


async def judge_async(a):
    from m5.automationbench.client import DS, Cap
    work = Path(a.work)
    rubric = (HERE / 'rubric_reuse_ab.md').read_text()
    test = {x['id']: x for x in map(json.loads, open(work / 'test.jsonl'))}
    cands = {x['id']: x for x in map(json.loads, open(work / 'cands.jsonl'))}
    pairs = [json.loads(x) for x in open(work / 'pairs.jsonl')]
    out_p = work / 'verdicts.jsonl'
    done = set()
    if out_p.exists():
        done = {(v['test_id'], v['cand_id'], v['sample']) for v in map(json.loads, open(out_p))}
    ds = DS(cap=Cap('ds', a.concurrency), effort='high')
    await ds.ready()
    lock = asyncio.Lock()
    n = {'done': 0}

    async def one(p, s):
        if (p['test_id'], p['cand_id'], s) in done:
            return
        msg = [{'role': 'user', 'content': judge_prompt(rubric, test[p['test_id']], cands[p['cand_id']], p['shared_strings'])}]
        try:
            content, _, _ = await ds.chat(msg, key=f'{p["test_id"]}#{s}')   # one rank per test task: shared prefix
            v = verdict_of(content)
        except Exception as e:  # noqa: BLE001
            content, v = f'error: {e}', None
        async with lock:
            with out_p.open('a') as f:
                f.write(json.dumps({'test_id': p['test_id'], 'cand_id': p['cand_id'], 'sample': s, 'verdict': v,
                                    'text': content[-1500:]}) + '\n')
            n['done'] += 1
            if n['done'] % 200 == 0:
                print(n['done'], 'verdicts', flush=True)

    await asyncio.gather(*(one(p, s) for p in pairs for s in range(2)))
    summarize(work)


def summarize(work: Path):
    by = defaultdict(list)
    for v in map(json.loads, open(work / 'verdicts.jsonl')):
        by[(v['test_id'], v['cand_id'])].append(v['verdict'])
    excl = {}
    for (t, c), vs in by.items():
        if 'REUSE' in vs or (len(vs) >= 2 and all(x == 'WEAK_REUSE' for x in vs)):
            excl.setdefault(c, []).append({'test_id': t, 'verdicts': vs})
    n_c = sum(1 for _ in open(work / 'cands.jsonl'))
    res = {'n_candidates': n_c, 'n_pairs_judged': len(by), 'n_excluded': len(excl),
           'rate': len(excl) / max(1, n_c), 'unparsed': sum(v.count(None) for v in by.values()),
           'by_test': Counter(x['test_id'] for xs in excl.values() for x in xs).most_common(20), 'excluded': excl}
    (work / 'excluded.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'excluded'}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'retrieve', 'judge', 'summarize'])
    ap.add_argument('--accepted')
    ap.add_argument('--work', required=True)
    ap.add_argument('--examples', action='store_true')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--concurrency', type=int, default=None)
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(a)
    elif a.cmd == 'retrieve':
        retrieve(a)
    elif a.cmd == 'judge':
        asyncio.run(judge_async(a))
    else:
        summarize(Path(a.work))


if __name__ == '__main__':
    main()

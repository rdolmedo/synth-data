"""Problem-level decontamination against the 100 GDP.pdf test tasks, once at the end (#119's method).

  python -m m5.gdppdf.decontam.problems retrieve --problems '<glob>' --index <index.jsonl> --work <dir>
  python -m m5.gdppdf.decontam.problems judge --work <dir> [--server m5-ds]
  python -m m5.gdppdf.decontam.problems decide --work <dir>

retrieve (CPU job): candidate (test task, problem) pairs = the union of each test task's top-K window under four
  signals: word 8-gram overlap count, TF-IDF cosine (word 1-2 grams) of the requests, bge-small-en-v1.5 embedding
  cosine of the requests, and TF-IDF cosine of the documents' descriptions (title, issuer, series, summary: same
  issuer/series). Writes pairs.jsonl.
judge: DeepSeek, rubric.md (REUSE / WEAK_REUSE / NO_REUSE; scaffold reuse counts), 2 samples per pair, high effort.
  #19's deep rule: a REUSE in the bottom half of a test task's window doubles that window (up to K_MAX), and the new
  pairs are judged too.
decide: a problem is excluded on any REUSE sample, or when both samples say WEAK_REUSE. Writes excluded.json.
The canary GUID and test-document hashes are checked separately (decontam/docs.py, export).
"""
import argparse
import asyncio
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

K, K_MAX = 30, 120
RUBRIC = Path(__file__).with_name('rubric.md')


def words(t):
    return re.findall(r'[a-z0-9]+', t.lower())


def ng(t, n=8):
    w = words(t)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_problems(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                r = json.loads(l)
                if r.get('status') == 'accepted':
                    out[r['problem_id']] = r
    return list(out.values())


def test_items(doc_labels):
    from aa_evals.gdppdf import data
    lab = {}
    if doc_labels and Path(doc_labels).exists():
        lab = {json.loads(l)['file_name']: json.loads(l)['label'] for l in open(doc_labels)}
    items = []
    for t in data.load_tasks():
        l = lab.get(t.task_id) or {}
        items.append({'id': t.task_id, 'domain': t.domain, 'prompt': t.prompt,
                      'doc': '; '.join(str(x) for x in (l.get('title'), l.get('issuer'), l.get('series'),
                                                         l.get('summary')) if x)})
    return items


EMBED_PY = """import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
texts = json.load(open(sys.argv[2]))
m = SentenceTransformer(sys.argv[1], device='cpu')
np.save(sys.argv[3], m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))
"""


def embed(texts, model_dir, python=None, work=Path('.')):
    """bge-small-en-v1.5 embeddings, in another venv's python if this one lacks sentence-transformers."""
    if python is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_dir, device='cpu')
        return m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    import subprocess
    (work / 'embed_in.json').write_text(json.dumps(texts))
    (work / 'embed.py').write_text(EMBED_PY)
    subprocess.run([python, '-I', str(work / 'embed.py'), model_dir, str(work / 'embed_in.json'),
                    str(work / 'embed.npy')], check=True, env={'HF_HUB_OFFLINE': '1', 'PATH': '/usr/bin:/bin'})
    return np.load(work / 'embed.npy')


def retrieve(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from m5.gdppdf.docs.store import DocStore
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    probs = load_problems(a.problems)
    if a.traces:   # only problems with a kept trace (status ok) can reach the export
        ok = set()
        for p in glob.glob(a.traces):
            for l in open(p):
                if l.endswith('\n') and l.strip():
                    t = json.loads(l)
                    if t.get('status') == 'ok':
                        ok.add(t['problem_id'])
        probs = [p for p in probs if p['problem_id'] in ok]
    meta = DocStore(a.index).meta
    tests = test_items(a.test_doc_labels)
    P = [p['prompt'] for p in probs]
    T = [t['prompt'] for t in tests]
    pdesc = ['; '.join(str(x) for x in (meta.get(p['doc_id'], {}).get(k) for k in ('title', 'issuer', 'series',
                                                                                    'summary')) if x) for p in probs]
    S = {}
    tg = [ng(t) for t in T]
    pg = [ng(p) for p in P]
    inv = collections.defaultdict(list)
    for j, g in enumerate(pg):
        for x in g:
            inv[x].append(j)
    ov = np.zeros((len(T), len(P)), dtype=np.float32)
    for i, g in enumerate(tg):
        for x in g:
            for j in inv.get(x, ()):
                ov[i, j] += 1
    S['ngram8'] = ov
    v = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True, min_df=2).fit(P + T)
    S['tfidf'] = (v.transform(T) @ v.transform(P).T).toarray()
    v2 = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True).fit(pdesc + [t['doc'] for t in tests])
    S['doc'] = (v2.transform([t['doc'] for t in tests]) @ v2.transform(pdesc).T).toarray()
    if a.embed_model:
        E = embed(T + P, a.embed_model, a.embed_python, work)
        S['embed'] = E[:len(T)] @ E[len(T):].T
    np.savez_compressed(work / 'signals.npz', **S)
    json.dump({'problems': [p['problem_id'] for p in probs], 'tests': [t['id'] for t in tests]},
              open(work / 'ids.json', 'w'))
    with open(work / 'items.jsonl', 'w') as f:
        for p, d in zip(probs, pdesc):
            f.write(json.dumps({'kind': 'problem', 'id': p['problem_id'], 'domain': p['domain'], 'prompt': p['prompt'],
                                'doc': d}) + '\n')
        for t in tests:
            f.write(json.dumps({'kind': 'test', **t}) + '\n')
    pairs = window_pairs(S, K)
    write_pairs(work / 'pairs.jsonl', pairs, S, probs, tests)
    print(json.dumps({'problems': len(P), 'tests': len(T), 'pairs': len(pairs), 'signals': list(S)}), flush=True)


def window_pairs(S, k, only=None):
    out = set()
    for name, M in S.items():
        for i in range(M.shape[0]):
            if only is not None and i not in only:
                continue
            kk = only.get(i, k) if only else k
            top = np.argsort(-M[i])[:kk]
            out |= {(i, int(j)) for j in top if M[i, j] > 0}
    return out


def write_pairs(path, pairs, S, probs, tests, mode='w'):
    with open(path, mode) as f:
        for i, j in sorted(pairs):
            f.write(json.dumps({'test': tests[i]['id'], 'problem': probs[j]['problem_id'],
                                **{k: round(float(M[i, j]), 4) for k, M in S.items()}}) + '\n')


JUDGE_USER = """## Test task (GDP.pdf)
Domain: {t_domain}
Document: {t_doc}
Request:
<<<
{t_prompt}
>>>

## Overlap signals
{signals}

## Candidate (synthetic training task)
Domain: {c_domain}
Document: {c_doc}
Request:
<<<
{c_prompt}
>>>"""


def parse_label(text):
    m = re.search(r'\{.*\}', text, re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except ValueError:
        j = None
    lab = (j or {}).get('label')
    return lab if lab in ('REUSE', 'WEAK_REUSE', 'NO_REUSE') else None, j


async def judge_pairs(a, pairs, items, out_path):
    from m5.gdppdf.llm import Server
    done = set()
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done.add((r['test'], r['problem'], r['sample']))
    server = await Server(a.server, 'ds', cap=a.concurrency).start()
    rubric = RUBRIC.read_text()
    out = open(out_path, 'a')
    sem = asyncio.Semaphore(4 * (a.concurrency or 100))

    async def one(pr, s):
        async with sem:
            t, c = items[('test', pr['test'])], items[('problem', pr['problem'])]
            sig = ', '.join(f'{k}={v}' for k, v in pr.items() if k not in ('test', 'problem'))
            body = {'messages': [{'role': 'system', 'content': rubric},
                                 {'role': 'user', 'content': JUDGE_USER.format(
                                     t_domain=t['domain'], t_doc=t['doc'] or '(not described)', t_prompt=t['prompt'],
                                     signals=sig, c_domain=c['domain'], c_doc=c['doc'] or '(not described)',
                                     c_prompt=c['prompt'])}],
                    'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'high',
                    'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'high'}}
            for _ in range(3):
                try:
                    res = await server.chat(body, affinity=pr['test'])
                except (RuntimeError, ValueError) as e:
                    print(e, flush=True)
                    continue
                lab, j = parse_label(res['content'])
                if lab:
                    out.write(json.dumps({'test': pr['test'], 'problem': pr['problem'], 'sample': s, 'label': lab,
                                          'verdict': j, 'usage': res['usage']}) + '\n')
                    out.flush()
                    return

    await asyncio.gather(*(one(pr, s) for pr in pairs for s in range(2) if (pr['test'], pr['problem'], s) not in done))
    out.close()
    await server.close()


def judge(a):
    work = Path(a.work)
    items = {}
    for l in open(work / 'items.jsonl'):
        r = json.loads(l)
        items[(r['kind'], r['id'])] = r
    S = dict(np.load(work / 'signals.npz'))
    ids = json.load(open(work / 'ids.json'))
    k = {i: K for i in range(len(ids['tests']))}
    while True:
        pairs = [json.loads(l) for l in open(work / 'pairs.jsonl')]
        asyncio.run(judge_pairs(a, pairs, items, work / 'verdicts.jsonl'))
        # deep rule: a REUSE in the bottom half of a test task's window doubles it
        ver = [json.loads(l) for l in open(work / 'verdicts.jsonl')]
        pj = {p: j for j, p in enumerate(ids['problems'])}
        ti = {t: i for i, t in enumerate(ids['tests'])}
        deepen = {}
        for v in ver:
            if v['label'] != 'REUSE':
                continue
            i, j = ti[v['test']], pj[v['problem']]
            for M in S.values():
                rank = int((M[i] > M[i, j]).sum())
                if k[i] // 2 <= rank < k[i] and k[i] < K_MAX:
                    deepen[i] = min(K_MAX, 2 * k[i])
        if not deepen:
            break
        k.update(deepen)
        have = {(p['test'], p['problem']) for p in pairs}
        new = {(i, j) for i, j in window_pairs(S, K, only=deepen)
               if (ids['tests'][i], ids['problems'][j]) not in have}
        print(f'deepening {len(deepen)} test windows: {len(new)} new pairs', flush=True)
        if not new:
            break

        probs = [{'problem_id': p} for p in ids['problems']]
        tests = [{'id': t} for t in ids['tests']]
        write_pairs(work / 'pairs.jsonl', new, S, probs, tests, mode='a')


def decide(a):
    work = Path(a.work)
    by = collections.defaultdict(list)
    for l in open(work / 'verdicts.jsonl'):
        v = json.loads(l)
        by[(v['test'], v['problem'])].append(v['label'])
    excl = {}
    for (t, p), labs in by.items():
        if 'REUSE' in labs or (len(labs) >= 2 and all(x == 'WEAK_REUSE' for x in labs)):
            excl.setdefault(p, []).append({'test': t, 'labels': labs})
    n_prob = len(json.load(open(work / 'ids.json'))['problems'])
    rep = {'pairs_judged': len(by), 'excluded': len(excl), 'rate': round(len(excl) / max(1, n_prob), 5),
           'by_test': collections.Counter(x['test'] for v in excl.values() for x in v).most_common(10)}
    json.dump({'excluded': excl, 'report': rep}, open(work / 'excluded.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['retrieve', 'judge', 'decide'])
    ap.add_argument('--problems', default=None)
    ap.add_argument('--traces', default=None, help='glob of trace files: keep only problems with an ok trace')
    ap.add_argument('--index', default='/data/aai/m5/shared/docs/index.jsonl')
    ap.add_argument('--test-doc-labels', default='/data/aai/m5/gdppdf/work/test_doc_labels.jsonl')
    ap.add_argument('--embed-model', default='/data/aai/m5/gdppdf/models/bge-small-en-v1.5', help='a local bge-small-en-v1.5 directory')
    ap.add_argument('--embed-python', default='/data/aai/m5/hle/venv-decon/bin/python',
                    help='a python with sentence-transformers (m5-hle\'s decon venv, read-only)')
    ap.add_argument('--work', required=True)
    ap.add_argument('--server', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=None)
    a = ap.parse_args()
    {'retrieve': retrieve, 'judge': judge, 'decide': decide}[a.cmd](a)


if __name__ == '__main__':
    main()

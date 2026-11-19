"""Repeat meter with judge confirmation: the share of problems that have a true near-duplicate, projected to production.

    python -m m5.scicode.repeat --gen DIR[,DIR...] --out DIR [--cand-cos 0.25] [--n-prod 9000]

Candidate pairs are TF-IDF neighbours (the dedup.py vectorizer, fitted on the vocab leaves; the pilot-v1 true
near-duplicates sat at cosine 0.38-0.40 under it, so 0.25 leaves margin for recall). Each candidate pair goes to the
DeepSeek same-core-problem judge (dedup.RUBRIC), both orders; a pair counts when either order says DUPLICATE.
Projection: pairwise probability p = confirmed / all pairs, share of N problems with >= 1 duplicate = 1 - (1 - p)^(N - 1),
with a 95% Poisson upper bound on the confirmed count.
"""
import argparse
import asyncio
import json
import math
import re
from pathlib import Path

import numpy as np

from .common import Alloc, append_line, read_jsonl, safe_write
from .dedup import RUBRIC, render
from .meter import spec_text

VOCAB = Path(__file__).resolve().parent / 'taxonomy' / 'vocab.json'

RUBRIC3 = """You check a synthetic training set of scientific programming problems for near-duplicates. Classify the pair:
- DUPLICATE: the same specific system or model and the same main computed quantity (or one problem's main computation is the core of the other's), so that a solution of one is largely a solution of the other, in any wording, with any parameter values, materials or function names.
- SAME_CORE: the same specific system or model and a largely shared core computation (e.g. the same simulation, solver or derivation), but a different main observable, or substantial analysis that one has and the other lacks.
- DISTINCT: anything else, e.g. the same broad topic, or the same method applied to different systems.

Answer with only a JSON object: {"verdict": "DUPLICATE" | "SAME_CORE" | "DISTINCT", "reason": "<one sentence>"}"""
LABELS = {'strict': (RUBRIC, ('DUPLICATE', 'DISTINCT')), 'three': (RUBRIC3, ('DUPLICATE', 'SAME_CORE', 'DISTINCT'))}


def poisson_upper(k, conf=0.95):
    """Exact upper limit of a Poisson mean given k events (chi-square via bisection on the CDF)."""
    lo, hi = 0.0, 10.0 + 5 * k
    for _ in range(100):
        mid = (lo + hi) / 2
        cdf = sum(math.exp(-mid) * mid ** i / math.factorial(i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > 1 - conf else (lo, mid)
    return hi


def candidates(specs, cand_cos, per_spec=5, emb=None, emb_cos=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vocab = json.loads(VOCAB.read_text())
    texts = [f"{x['title']}. {x['gist']}" for xs in vocab['leaves'].values() for x in xs]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=2).fit(texts)
    X = vec.transform([spec_text(s) for s in specs])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    out = {}
    for i in range(len(S)):
        for j in np.argsort(-S[i])[:per_spec]:
            if S[i, j] > cand_cos:
                a, b = sorted((i, int(j)))
                out[(a, b)] = float(S[i, j])
    if emb is not None:   # bge-small rows in specs order: adds neighbours that TF-IDF misses (reworded duplicates)
        if emb == 'auto':   # needs sentence-transformers (job-local venv: logs/jobs/venvjob.sh)
            from sentence_transformers import SentenceTransformer
            E = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu').encode(
                [spec_text(s) for s in specs], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        else:
            E = np.load(emb)
        assert len(E) == len(specs), (len(E), len(specs))
        for i in range(len(E)):
            c = E @ E[i]
            c[i] = 0
            for j in np.argsort(-c)[:per_spec]:
                if c[j] > emb_cos:
                    a, b = sorted((i, int(j)))
                    out.setdefault((a, b), float(S[a, b]))
    return out, S


async def judge(specs, pairs, out_path, serving, rubric='strict'):
    from .generate import DS
    done = {(r['a'], r['b'], r['order']) for r in read_jsonl(out_path)}
    jobs = [(a, b, o) for (a, b) in pairs for o in (0, 1) if (specs[a]['problem_id'], specs[b]['problem_id'], o) not in done]
    print(f'{len(pairs)} candidate pairs, {len(jobs)} judgments to run', flush=True)
    ds, alloc = DS(serving), Alloc()

    async def one(a, b, o):
        x, y = (specs[a], specs[b]) if o == 0 else (specs[b], specs[a])
        user = f"## Problem A\n{render(x)}\n\n## Problem B\n{render(y)}"
        for _ in range(3):
            g = await ds.chat(user, system=LABELS[rubric][0])
            m = re.search(r'\{.*\}', g['content'] or '', re.S)
            try:
                v = json.loads(m.group(0))
            except (AttributeError, ValueError):
                continue
            if v.get('verdict') in LABELS[rubric][1]:
                append_line(out_path, json.dumps({'a': specs[a]['problem_id'], 'b': specs[b]['problem_id'], 'order': o,
                                                  'cos': round(pairs[(a, b)], 3), 'verdict': v['verdict'],
                                                  'reason': v.get('reason', '')}))
                return

    pending = set()
    for job in jobs:
        while len(pending) >= max(1, alloc.ds_cap()):
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.add(asyncio.create_task(one(*job)))
    if pending:
        await asyncio.gather(*pending)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cand-cos', type=float, default=0.25)
    ap.add_argument('--n-prod', type=int, default=9000)
    ap.add_argument('--serving', default=None, help='pin one DeepSeek server (default: alloc.json ds.serving)')
    ap.add_argument('--traces', default=None, help='only problems with a chain in these trace dirs')
    ap.add_argument('--rubric', default='strict', choices=list(LABELS))
    ap.add_argument('--emb', default=None, help="bge-small embeddings of spec_text (npy, rows in specs order), or 'auto' to encode here")
    ap.add_argument('--emb-cos', type=float, default=0.85)
    ap.add_argument('--per-spec', type=int, default=5, help='neighbours per spec and signal')
    a = ap.parse_args()
    specs = [s for g in a.gen.split(',') for s in read_jsonl(Path(g) / 'specs.jsonl')]
    if a.traces:
        traced = {c['problem_id'] for t in a.traces.split(',') for c in read_jsonl(Path(t) / 'chains.jsonl')}
        specs = [s for s in specs if s['problem_id'] in traced]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, S = candidates(specs, a.cand_cos, per_spec=a.per_spec, emb=a.emb, emb_cos=a.emb_cos)
    vpath = out / ('repeat_verdicts.jsonl' if a.rubric == 'strict' else f'repeat_verdicts_{a.rubric}.jsonl')
    asyncio.run(judge(specs, pairs, vpath, a.serving, a.rubric))
    if a.rubric == 'three':
        return flags3(specs, pairs, vpath, out, a.n_prod)
    dup = {}
    for r in read_jsonl(vpath):
        k = (r['a'], r['b'])
        dup[k] = dup.get(k, False) or r['verdict'] == 'DUPLICATE'
    conf = sorted(k for k, v in dup.items() if v)
    n = len(specs)
    n_pairs = n * (n - 1) // 2
    cos = {(specs[a]['problem_id'], specs[b]['problem_id']): c for (a, b), c in pairs.items()}
    proj = lambda k: 1 - (1 - k / n_pairs) ** (a.n_prod - 1)   # noqa: E731
    in_dup = {x for k in conf for x in k}
    nn = S.max(1)
    res = {'n_specs': n, 'pairs_total': n_pairs, 'cand_cos': a.cand_cos, 'candidate_pairs': len(pairs),
           'judged_pairs': len(dup), 'confirmed_pairs': len(conf), 'specs_in_confirmed_pair': len(in_dup),
           'confirmed_cos': sorted(round(cos[k], 3) for k in conf),
           'pairwise_p': len(conf) / n_pairs, 'n_prod': a.n_prod, 'projected_share_with_dup': proj(len(conf)),
           'projected_share_with_dup_95up': proj(poisson_upper(len(conf))),
           'nn_cos_mean': float(nn.mean()), 'nn_cos_p90': float(np.quantile(nn, 0.9)),
           'confirmed': [{'a': k[0], 'b': k[1], 'cos': round(cos[k], 3)} for k in conf]}
    safe_write(out / 'repeat.json', json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'confirmed'}, indent=1))
    for k in conf:
        print('DUP', k, round(cos[k], 3))


def flags3(specs, pairs, vpath, out, n_prod):
    """Clear = both samples (orders) flag the pair (DUPLICATE or SAME_CORE); borderline = one sample flags it. The judge
    almost never says DUPLICATE (calibration: every known and hand-labelled duplicate on pilots v1/v2 got SAME_CORE
    twice), so agreement of the two samples is the usable split. Writes dup_flags.json:
    problem_id -> {level, identical, partners, redundant_identical, redundant_clear, redundant_any}: identical = some
    partner pair got a DUPLICATE vote (the near-identical tier); redundant_* = what a keep-one pass (earliest problem
    kept) over identical pairs, clear pairs, or all flagged pairs would drop."""
    votes = {}
    for r in read_jsonl(vpath):
        votes.setdefault((r['a'], r['b']), []).append(r['verdict'])
    level, ndup = {}, {k: vs.count('DUPLICATE') for k, vs in votes.items()}
    for k, vs in votes.items():
        n_flag = sum(v != 'DISTINCT' for v in vs)
        if n_flag >= 2:
            level[k] = 'clear'
        elif n_flag == 1:
            level[k] = 'borderline'
    order = {s['problem_id']: i for i, s in enumerate(specs)}
    rank = {'clear': 2, 'borderline': 1}
    flags = {}
    for (x, y), lv in level.items():
        for u, w in ((x, y), (y, x)):
            f = flags.setdefault(u, {'level': lv, 'partners': []})
            f['partners'].append({'id': w, 'level': lv, 'dup_votes': ndup[(x, y)]})
            f['identical'] = f.get('identical', False) or ndup[(x, y)] > 0
            if rank[lv] > rank[f['level']]:
                f['level'] = lv
    tiers = (('identical', lambda p: p['dup_votes'] > 0), ('clear', lambda p: p['level'] == 'clear'),
             ('any', lambda p: True))
    for name, ok in tiers:
        kept = set()
        for pid in sorted(flags, key=order.get):
            if any(p['id'] in kept and ok(p) for p in flags[pid]['partners']):
                flags[pid][f'redundant_{name}'] = True
            else:
                kept.add(pid)
    n = len(specs)
    res = {'n_specs': n, 'candidate_pairs': len(pairs), 'judged_pairs': len(votes),
           'pairs_clear': sum(v == 'clear' for v in level.values()),
           'pairs_borderline': sum(v == 'borderline' for v in level.values()),
           'pairs_identical': sum(v > 0 for k, v in ndup.items() if k in level),
           'share_in_identical_pair': sum(f['identical'] for f in flags.values()) / n,
           'share_redundant_identical': sum(bool(f.get('redundant_identical')) for f in flags.values()) / n,
           'share_in_clear_pair': sum(f['level'] == 'clear' for f in flags.values()) / n,
           'share_in_any_pair': len(flags) / n,
           'share_redundant_clear': sum(bool(f.get('redundant_clear')) for f in flags.values()) / n,
           'share_redundant_any': sum(bool(f.get('redundant_any')) for f in flags.values()) / n,
           'flags': flags}
    safe_write(out / 'dup_flags.json', json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'flags'}, indent=1))
    return res


if __name__ == '__main__':
    main()

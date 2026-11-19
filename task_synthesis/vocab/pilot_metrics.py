"""Diversity and quality of task sets made from different spec vocabularies (#25 pilot).

  python -m task_synthesis.vocab.pilot_metrics --set v2=<round> --set v1=<round> [--set student1=student1] [--out f.json]

A set is a round under ROOT/rounds, or `student1` (student 1's bundled tasks: r5 + s1 + s2). Needs numpy.
Metrics:
- quality: valid share, judge pass share among valid, mean judge scores, spec_mismatch flags;
- themes: per domain, the share of tasks whose slug contains the domain's most frequent slug word (mean over domains
  with >= 30 tasks, at 30 tasks each), and the global share of the 10 most frequent slug words;
- similarity: TF-IDF cosine of instructions, each task's nearest neighbour within its set, at equal N (subsampled,
  mean of 5 draws): mean, p90, share above 0.5;
- packages: share installing nothing beyond the base image, distinct packages per 100 installing tasks (rarefied);
- v2 usage: share of specs drawing a new action / domain / class / format, and with a sub-domain."""
import argparse
import collections
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from ..common import ROOT

BASE = set('ca-certificates curl wget git openssh-client rsync tmux vim nano less tree file procps psmisc lsof iproute2 '
           'iputils-ping netcat-openbsd dnsutils unzip zip xz-utils bzip2 zstd patch diffutils bc gawk jq sqlite3 '
           'build-essential make cmake pkg-config gdb python3 python3-pip python3-venv python3-dev python-is-python3 '
           'time xxd bsdextrautils gcc g++ numpy pandas scipy requests pyyaml pytest pip setuptools wheel'.split())
STOP = {'to', 'and', 'the', 'of', 'v2', 'v3', 'a', 'for', 'in', 'from', 'with', 'tool', 'cli'}
WORD = re.compile(r'[a-z0-9_]+')


def records(name):
    if name == 'student1':
        out = []
        for c in ('r5', 's1', 's2'):
            man = json.loads((ROOT / 'campaigns' / c / 'manifest.json').read_text())
            for v in man.values():
                rec = ROOT / 'rounds' / v['round'] / 'work' / v['spec_id'] / 'record.json'
                out.append(json.loads(rec.read_text()))
        return out
    return [json.loads(p.read_text()) for r in name.split(',')   # a comma list pools several rounds
            for p in sorted((ROOT / 'rounds' / r / 'work').glob('*/record.json'))]


def pkgs(meta):
    p = set(meta.get('apt') or []) | {re.split(r'[=<>\[ ]', x)[0].lower() for x in (meta.get('pip') or [])}
    return p - BASE


def tfidf_nn(texts):
    toks = [WORD.findall(t.lower()) for t in texts]
    docs = [collections.Counter(t + [f'{a}_{b}' for a, b in zip(t, t[1:])]) for t in toks]
    df = collections.Counter(w for d in docs for w in d)
    vocab = {w: i for i, w in enumerate(w for w, c in df.items() if c >= 2)}
    n = len(docs)
    X = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for w, c in d.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = (1 + math.log(c)) * math.log(n / df[w])
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    S = X @ X.T
    np.fill_diagonal(S, -1)
    return S.max(axis=1)


def metrics(recs, n_eq, seed=0):
    valid = [r for r in recs if r['status'] == 'valid']
    judged = [r for r in valid if r.get('judge')]
    passed = [r for r in judged if r['judge'].get('pass')]
    out = {'specs': len(recs), 'valid': len(valid) / max(len(recs), 1),
           'judge_pass_of_valid': len(passed) / max(len(judged), 1)}
    sc = collections.defaultdict(list)
    for r in judged:
        for k, x in (r['judge'].get('scores') or {}).items():
            if isinstance(x, (int, float)):
                sc[k].append(x)
    out['judge_scores'] = {k: round(sum(v) / len(v), 3) for k, v in sorted(sc.items())}
    out['spec_mismatch'] = sum(bool((r['judge'].get('flags') or {}).get('spec_mismatch')) for r in judged) / max(len(judged), 1)
    # themes
    by = collections.defaultdict(list)
    for r in valid:
        by[r['spec']['domain']].append([w for w in set((r['meta'].get('slug') or '').split('-')) if w and w not in STOP])
    tops, rng = {}, random.Random(seed)
    for d, ws in by.items():   # at 30 tasks per domain (mean of 20 draws), so sets of different sizes compare
        if len(ws) >= 30:
            w = collections.Counter(x for s in ws for x in s).most_common(1)[0][0]
            share = sum(collections.Counter(x for s in rng.sample(ws, 30) for x in s).most_common(1)[0][1] / 30
                        for _ in range(20)) / 20
            tops[d] = (w, round(share, 3))
    out['domain_top_word'] = dict(sorted(tops.items(), key=lambda kv: -kv[1][1]))
    out['domain_top_word_mean'] = sum(v for _, v in tops.values()) / max(len(tops), 1)
    allw = collections.Counter(x for ws in by.values() for s in ws for x in s)
    out['top10_words'] = [(w, round(c / len(valid), 3)) for w, c in allw.most_common(10)]
    # similarity at equal N
    rng = random.Random(seed)
    nn = []
    for _ in range(5):
        sub = rng.sample(valid, min(n_eq, len(valid)))
        nn.append(tfidf_nn([r.get('instruction') or '' for r in sub]))
    nn = np.concatenate(nn)
    out['nn_sim'] = {'n': min(n_eq, len(valid)), 'mean': round(float(nn.mean()), 4),
                     'p90': round(float(np.quantile(nn, 0.9)), 4), 'gt0.5': round(float((nn > 0.5).mean()), 4)}
    # packages
    ps = [pkgs(r.get('meta') or {}) for r in valid]
    inst = [p for p in ps if p]
    out['install_nothing'] = 1 - len(inst) / max(len(ps), 1)
    k = min(100, len(inst))
    out['distinct_pkgs_per_100_installing'] = (sum(len(set().union(*rng.sample(inst, k))) for _ in range(50)) / 50
                                               if k == 100 else None)
    # slug richness at equal N
    out['distinct_slug_words_eq'] = sum(len({x for r in rng.sample(valid, min(n_eq, len(valid)))
                                             for x in (r['meta'].get('slug') or '').split('-')}) for _ in range(5)) / 5
    return out


def v2_usage(recs, vocab):
    new = {'action': {x['name'] for x in vocab['actions']}, 'domain': {x['name'] for x in vocab['domains']},
           'class': {x['name'] for x in vocab['classes']},
           'format': {f['name'] for fs in vocab['formats'].values() for f in fs}}
    use = collections.Counter()
    for r in recs:
        s = r['spec']
        use['action'] += s['agent_action'] in new['action']
        use['domain'] += s['domain'] in new['domain']
        cl = {a['class'] for a in s.get('assets', [])}
        use['class'] += bool(cl & new['class'])
        fm = {f for a in s.get('assets', []) for k in ('formats', 'services', 'state') for f in (a.get(k) or [])}
        use['format'] += bool(fm & new['format'])
        use['subdomain'] += bool(s.get('subdomain'))
        use['any_new_toplevel'] += (s['agent_action'] in new['action'] or s['domain'] in new['domain']
                                    or bool(cl & new['class']))
    return {k: round(v / max(len(recs), 1), 3) for k, v in use.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--set', action='append', required=True, help='label=round (or label=student1)')
    ap.add_argument('--n-eq', type=int, default=0, help='N for similarity (default: smallest set of valid tasks)')
    ap.add_argument('--out')
    a = ap.parse_args()
    sets = dict(s.split('=', 1) for s in a.set)
    recs = {k: records(v) for k, v in sets.items()}
    n_eq = a.n_eq or min(sum(r['status'] == 'valid' for r in rs) for rs in recs.values())
    res = {k: metrics(rs, n_eq) for k, rs in recs.items()}
    for k, rs in recs.items():
        vname = next((r['spec'].get('vocab') for r in rs if r['spec'].get('vocab')), None)
        if vname:
            vocab = json.loads((Path(__file__).resolve().parents[1] / 'asset_sampler' / f'vocab_{vname}.json').read_text())
            res[k]['v2_usage'] = v2_usage(rs, vocab)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Text similarity for decontamination (generated instructions vs TB2.1 and TB-Lite) and diversity (near-duplicates
within a set): word 8-gram containment and TF-IDF cosine, in pure Python (a few thousand short documents)."""
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path

from ..common import ROOT

EVAL_TARS = {'tb21': '/data/aai/tbench/eval_tb21.tar', 'tblite': '/data/aai/tbench/eval_tblite.tar'}
CACHE = ROOT / 'decon' / 'eval_instructions.json'
WORD = re.compile(r'[a-z0-9_]+')


def eval_instructions():
    """{'tb21/<task>': text, 'tblite/<task>': text}, read once from the eval bundles."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    for name, path in EVAL_TARS.items():
        with tarfile.open(path) as tar:
            for m in tar:
                if m.isfile() and m.name.endswith('/instruction.md') and m.name.count('/') == 2:
                    out[f'{name}/{m.name.split("/")[1]}'] = tar.extractfile(m).read().decode(errors='replace')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def words(text):
    return WORD.findall(text.lower())


def ngrams(text, n=8):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def containment(a, b, n=8):
    """Share of a's n-grams found in b."""
    ga = ngrams(a, n)
    return len(ga & ngrams(b, n)) / len(ga) if ga else 0.0


class Tfidf:
    def __init__(self, docs):
        self.df = Counter(t for d in docs for t in set(words(d)))
        self.n = len(docs)

    def vec(self, text):
        tf = Counter(words(text))
        v = {t: (1 + math.log(c)) * math.log((1 + self.n) / (1 + self.df.get(t, 0))) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}


def cosine(u, v):
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(t, 0.0) for t, x in u.items())


def decon(texts, cos_flag=0.5, ngram_flag=0.1):
    """texts: {id: instruction}. -> {id: {'nearest', 'cosine', 'ngram8', 'flagged'}} against TB2.1 + TB-Lite."""
    ev = eval_instructions()
    model = Tfidf(list(ev.values()) + list(texts.values()))
    evv = {k: model.vec(t) for k, t in ev.items()}
    out = {}
    for tid, text in texts.items():
        v = model.vec(text)
        best = max(evv, key=lambda k: cosine(v, evv[k]))
        c = cosine(v, evv[best])
        g = max(containment(text, ev[k]) for k in ev)
        out[tid] = {'nearest': best, 'cosine': round(c, 3), 'ngram8': round(g, 3),
                    'flagged': c >= cos_flag or g >= ngram_flag}
    return out


def near_duplicates(texts, threshold=0.6):
    """Pairs of ids within a set whose instructions have TF-IDF cosine >= threshold."""
    ids = list(texts)
    model = Tfidf(list(texts.values()))
    vecs = [model.vec(texts[i]) for i in ids]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            c = cosine(vecs[i], vecs[j])
            if c >= threshold:
                pairs.append((ids[i], ids[j], round(c, 3)))
    nearest = {}
    for i in range(len(ids)):
        best = max((cosine(vecs[i], vecs[j]) for j in range(len(ids)) if j != i), default=0.0)
        nearest[ids[i]] = round(best, 3)
    return pairs, nearest


def near_duplicates_fast(texts, threshold=0.6, max_df=0.02, min_partial=0.25):
    """near_duplicates for thousands of texts without numpy: candidate pairs share rare terms (document frequency
    <= max_df of the set); their rare-term dot product must reach min_partial, then the exact cosine decides.
    Pairs at cosine >= 0.6 share many rare words, so the candidate filter loses little."""
    ids = list(texts)
    model = Tfidf(list(texts.values()))
    vecs = [model.vec(texts[i]) for i in ids]
    cap = max(2, int(max_df * len(ids)))
    postings = {}
    for k, v in enumerate(vecs):
        for t, x in v.items():
            if model.df.get(t, 0) <= cap:
                postings.setdefault(t, []).append((k, x))
    partial = Counter()
    for plist in postings.values():
        for a in range(len(plist)):
            i, xi = plist[a]
            for j, xj in plist[a + 1:]:
                partial[(i, j)] += xi * xj
    pairs = []
    for (i, j), p in partial.items():
        if p >= min_partial:
            c = cosine(vecs[i], vecs[j])
            if c >= threshold:
                pairs.append((ids[i], ids[j], round(c, 3)))
    return sorted(pairs)


def near_duplicates_new(texts, new, threshold=0.6, max_df=0.02, min_partial=0.25):
    """near_duplicates_fast restricted to pairs with at least one id in `new` (a stream chunk against every task
    published before it): the candidate pass grows with the new texts, not with the square of the whole set.
    Pairs are (earlier id, later id) in the order of `texts`."""
    ids = list(texts)
    model = Tfidf(list(texts.values()))
    vecs = [model.vec(texts[i]) for i in ids]
    cap = max(2, int(max_df * len(ids)))
    postings = {}
    for k, v in enumerate(vecs):
        for t, x in v.items():
            if model.df.get(t, 0) <= cap:
                postings.setdefault(t, []).append((k, x))
    new = set(new)
    fresh = {k for k, i in enumerate(ids) if i in new}
    partial = Counter()
    for i in sorted(fresh):
        for t, xi in vecs[i].items():
            for j, xj in postings.get(t, ()):
                if j != i and (j not in fresh or j > i):  # a pair of new texts is counted once, from its first
                    partial[(min(i, j), max(i, j))] += xi * xj
    pairs = []
    for (i, j), p in partial.items():
        if p >= min_partial:
            c = cosine(vecs[i], vecs[j])
            if c >= threshold:
                pairs.append((ids[i], ids[j], round(c, 3)))
    return sorted(pairs)

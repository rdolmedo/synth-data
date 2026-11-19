"""Small text-similarity helpers (no sklearn on the login python): TF-IDF cosine and word n-gram overlap."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = set('''a an the and or of to in on for with by at from as is are was were be been being this that these those it
its into your you we our their they them he she his her not no but if then than so such can will would should may
must each any all per via about over under within between including include using use also other more most'''.split())


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'\-]*", s.lower())


def terms(s: str, bigrams=True) -> list[str]:
    w = [x for x in words(s) if x not in STOP and len(x) > 1]
    return w + ([f'{a} {b}' for a, b in zip(w, w[1:])] if bigrams else [])


def fit_idf(fit_docs: list[str]) -> tuple[dict, int]:
    df = Counter()
    for d in fit_docs:
        df.update(set(terms(d)))
    n = len(fit_docs)
    return {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}, n


def vectorize(d: str, idf: dict, n: int) -> dict:
    tf = Counter(terms(d))
    v = {t: (1 + math.log(c)) * idf.get(t, math.log(1 + n) + 1) for t, c in tf.items()}
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / z for t, x in v.items()}


def tfidf_matrix(docs: list[str], fit_docs: list[str] | None = None):
    """Rows L2-normalised sublinear TF-IDF over a shared vocabulary (dense float32; fine for ~10^4 docs x 10^5
    terms only via sparse dict products, so this returns a list of dicts)."""
    fit_docs = fit_docs if fit_docs is not None else docs
    df = Counter()
    for d in fit_docs:
        df.update(set(terms(d)))
    n = len(fit_docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    out = []
    for d in docs:
        tf = Counter(terms(d))
        v = {t: (1 + math.log(c)) * idf.get(t, math.log(1 + n) + 1) for t, c in tf.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / z for t, x in v.items()})
    return out


def cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def sim_matrix(A: list[dict], B: list[dict]) -> np.ndarray:
    """Cosine between every row of A and B via an inverted index on B."""
    inv: dict[str, list[tuple[int, float]]] = {}
    for j, b in enumerate(B):
        for t, x in b.items():
            inv.setdefault(t, []).append((j, x))
    S = np.zeros((len(A), len(B)), dtype=np.float32)
    for i, a in enumerate(A):
        row = S[i]
        for t, x in a.items():
            for j, y in inv.get(t, ()):
                row[j] += x * y
    return S


def ngrams(s: str, n: int = 8) -> set:
    w = words(s)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}

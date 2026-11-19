"""Streaming near-duplicate filter for generated specs, applied before a spec is admitted to Super (production
oversupply absorbs the rejects). The pilot's repeat meter projected far above 2% near-duplicates at production scale:
leaves of one subject survive in different cells under different wording, and the generator converges on the same
problem. This is a diversity filter, not a difficulty or quality screen.

A new spec's TF-IDF neighbours among the admitted specs (cosine above CAND_COS, at most MAX_CAND) go to a DeepSeek
judge (1 sample): is it the same core problem? Any YES rejects the spec as a duplicate. The TF-IDF model is fitted
once on the leaf texts of the vocabulary (fixed IDF).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .meter import spec_text

CAND_COS = 0.3   # pilot: the true near-duplicates sat at 0.38-0.40 under this vectorizer
MAX_CAND = 3

RUBRIC = """You check a synthetic training set of scientific programming problems for near-duplicates. Two problems are DUPLICATE when they pose the same core problem: the same specific system or model and the same main computed quantity, so that a solution of one is largely a solution of the other, in any wording, with any parameter values or function names. Problems on the same broad topic, or using the same method on different systems, or computing different quantities of the same model, are DISTINCT.

Answer with only a JSON object: {"verdict": "DUPLICATE" | "DISTINCT", "reason": "<one sentence>"}"""


def render(s):
    out = [f"Name: {s['problem_name']}", s['problem_description_main']]
    out += [f"Step {i + 1}: {x['step_description_prompt']}" for i, x in enumerate(s['sub_steps'])]
    return '\n'.join(out)


class Filter:
    def __init__(self, vocab: dict, admitted: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [f"{x['title']}. {x['gist']}" for xs in vocab['leaves'].values() for x in xs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english', min_df=2).fit(texts)
        self.specs, self.rows = [], []
        for s in admitted:
            self.add(s)

    def add(self, s):
        self.specs.append(s)
        self.rows.append(self.vec.transform([spec_text(s)]))

    def candidates(self, s):
        if not self.rows:
            return []
        import scipy.sparse as sp
        M = sp.vstack(self.rows)
        v = self.vec.transform([spec_text(s)])
        c = (M @ v.T).toarray().ravel()
        idx = [i for i in np.argsort(-c)[:MAX_CAND] if c[i] > CAND_COS]
        return [(self.specs[i], float(c[i])) for i in idx]

    async def is_duplicate(self, ds, s) -> tuple[bool, list]:
        log = []
        for other, cos in self.candidates(s):
            user = f"## Problem A\n{render(other)}\n\n## Problem B\n{render(s)}"
            v = None
            for _ in range(3):
                g = await ds.chat(user, system=RUBRIC)
                m = re.search(r'\{.*\}', g['content'] or '', re.S)
                try:
                    v = json.loads(m.group(0))
                    if v.get('verdict') in ('DUPLICATE', 'DISTINCT'):
                        break
                except (AttributeError, ValueError):
                    v = None
            log.append({'other': other['problem_id'], 'cos': round(cos, 3), **(v or {'verdict': 'unparsed'})})
            if v and v['verdict'] == 'DUPLICATE':
                return True, log
        return False, log

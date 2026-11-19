"""Sample the generator's slot specifications for one document from the fitted taxonomy (taxonomy/taxonomy.json).

Top level (domain, document type) comes from the document itself; the selection already matched their fitted
frequencies. Below it, each slot draws task type | domain, then a specific leaf (task family) of that domain that
fits the document type and task type, then asks, pitfalls, inline material, length, format and persona.
A slot with no fitting leaf is an open draw (counted; target <= 5%).
"""
import json
import math
import random
from pathlib import Path

TAXONOMY = Path(__file__).resolve().parents[1] / 'taxonomy' / 'taxonomy.json'


def wchoice(rng, d):
    ks = list(d)
    return rng.choices(ks, weights=[d[k] for k in ks])[0]


class Sampler:
    def __init__(self, path=TAXONOMY):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.t = json.loads(Path(path).read_text())
        self.leaf_use = {}
        self.vec, self.mats = {}, {}
        for d, leaves in self.t['leaves'].items():
            texts = [f"{l['leaf']} {l.get('role') or ''} {' '.join(l['doc_types'])}" for l in leaves]
            v = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True).fit(texts)
            self.vec[d], self.mats[d] = v, v.transform(texts)

    def task_types(self, domain):
        return self.t['task_types'][domain]

    def candidates(self, rng, doc, task_type, n=12, pool=24):
        """n leaves for the generator to choose from: the leaves of the document's own (domain, document type) cell
        that list this task type (else any of the cell, else the domain's), ordered by TF-IDF similarity to the
        document's description; of the `pool` most similar, the least offered first."""
        d = doc['domain']
        leaves = self.t['leaves'][d]
        desc = ' '.join(str(doc.get(k) or '') for k in ('title', 'summary', 'series', 'issuer', 'doc_type'))
        sim = (self.mats[d] @ self.vec[d].transform([desc]).T).toarray().ravel()
        order = sorted(range(len(leaves)), key=lambda i: -sim[i])
        cell = [i for i in order if doc.get('doc_type') in leaves[i]['doc_types']]
        idx = ([i for i in cell if task_type in leaves[i]['task_types']] or cell
               or [i for i in order if task_type in leaves[i]['task_types']])[:pool]
        idx.sort(key=lambda i: (self.leaf_use.get(leaves[i]['leaf'], 0), rng.random()))
        out = [leaves[i] for i in idx[:n]]
        for l in out:
            self.leaf_use[l['leaf']] = self.leaf_use.get(l['leaf'], 0) + 1
        return out

    def slots(self, doc, k, seed):
        rng = random.Random(seed)
        tts = dict(self.task_types(doc['domain']))
        out = []
        for i in range(k):
            tt = wchoice(rng, tts)
            tts[tt] *= 0.7             # different task types within a document, softly (0.35 under-drew the main type)
            cands = self.candidates(rng, doc, tt)
            inline = rng.random() < self.t['inline'][tt]
            kind = wchoice(rng, self.t['inline_kind'][tt]) if inline else None
            pool = self.t['words']['inline' if inline else 'plain']
            w = rng.choice(pool) * math.exp(rng.gauss(0, 0.25))
            w = int(max(20, min(1100, w)) // 5 * 5)
            out.append({'task_type': tt, 'leaf_candidates': [c['leaf'] for c in cands],
                        'leaf_roles': [c.get('role') for c in cands],
                        'n_asks': int(wchoice(rng, self.t['n_asks'][tt])),
                        'n_pitfalls': int(wchoice(rng, self.t['pitfalls'])), 'inline': kind, 'words': w,
                        'format': wchoice(rng, self.t['format'][tt]), 'persona': rng.random() < self.t['persona'],
                        'constraints': int(wchoice(rng, self.t['constraints'][tt])) if 'constraints' in self.t else None})
        return out

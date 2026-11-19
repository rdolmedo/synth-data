"""The spec sampler: GDPval's 44 occupations at uniform weight, Opus-brainstormed work activities below them, and
class-level dimensions fitted per sector with shrinkage toward the global counts (strength by leave-one-occupation-out
CV, as TB's smoothing_cv.py).

  python -m m5.gdpval.sampler fit            -> WORK/sampler.json (fitted tables + CV report)
  python -m m5.gdpval.sampler sample -n 20   -> specs as JSON lines (for inspection)

Fitted from WORK/test_labels.json (the text-only-v1 set: 207 tasks). Only counts per sector reach the specs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random

from m5.gdpval.opus import WORK
from m5.gdpval.vocab import OCC_SECTOR, OCCUPATIONS, SECTORS_OCCUPATIONS, TASK_TYPES

K_GRID = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 1e9]
EPS = 0.2
OPEN_SHARE = 0.04          # at most 5% open draws (#53 lessons)
TEMPLATE_SHARE = 0.25        # of synthetic-input tasks: one input is a template/partly done file to complete (test: 20 of 261 files, ~15% of input tasks; labeller undercounts)
REAL_SHARE = 0.10          # of tasks with inputs: real-document lane (GDPval inputs: 3% public documents; coordinator msg 11-11)
INPUT_EXTS = ['xlsx', 'pdf', 'docx', 'pptx', 'csv', 'txt']
DELIV_FMTS = ['pdf', 'docx', 'xlsx', 'pptx', 'csv', 'md', 'txt', 'code']
TEXT_ONLY_EXCLUDED = {   # aa_evals/gdpval/tasks.py TEXT_ONLY_EXCLUDED (media-core tasks)
    'f2986c1f', '57b2cdf2', 'a46d5cd2', 'a45bc83b', 'fe0d3941', 'a73fbc98', '75401f7c', 'e222075d', '38889c3b',
    '45c6237b', 'a941b6d8', 'ff85ee58', '4b894ae3'}


def load_rows():
    labels = json.loads((WORK / 'test_labels.json').read_text())
    rows = []
    for tid, x in labels.items():
        if tid[:8] in TEXT_ONLY_EXCLUDED:
            continue
        exts = [e.lstrip('.') for e in x['ref_ext']]
        exts = [e if e in INPUT_EXTS else ('txt' if e in ('md', 'json', 'step') else None) for e in exts]
        fm = [f for f in (x.get('deliverable_formats') or []) if f in DELIV_FMTS]
        rows.append({'task': tid, 'sector': x['sector'], 'occupation': x['occupation'],
                     'task_type': x['task_type'] if x['task_type'] in TASK_TYPES else None,
                     'has_inputs': 'yes' if x['ref_ext'] else 'no', 'input_exts': [e for e in exts if e],
                     'n_inputs': len(x['ref_ext']), 'formats': fm, 'n_deliverables': max(1, int(x.get('n_deliverables')
                                                                                              or 1)),
                     'n_words': x['n_words'], 'provenance': [i.get('provenance') for i in x.get('inputs') or []],
                     'needs_web': bool(x.get('needs_web'))})
    return rows


# ------------------------------------------------------------------------------------------------- fit + CV
def tally(rows, val, w=1.0):
    """Counts of val(r) per sector and globally; val may return a list (each element counted 1/len)."""
    by, glob = collections.defaultdict(collections.Counter), collections.Counter()
    for r in rows:
        v = val(r)
        vs = v if isinstance(v, list) else [v]
        vs = [x for x in vs if x is not None]
        for x in vs:
            by[r['sector']][x] += w / len(vs)
            glob[x] += w / len(vs)
    return by, glob


def drop_single(rows, val):
    """Sector cells that rest on a single test task are merged into the global prior (#358): their sector count is
    removed (the value keeps its global share)."""
    by_tasks = collections.defaultdict(set)
    for r in rows:
        v = val(r)
        for x in (v if isinstance(v, list) else [v]):
            if x is not None:
                by_tasks[(r['sector'], x)].add(r['task'])
    return {key for key, ts in by_tasks.items() if len(ts) == 1}


def model(train, val, voc, k, singles=frozenset()):
    by, glob = tally(train, val)
    prior = {v: glob.get(v, 0) + EPS for v in voc}
    z = sum(prior.values())
    prior = {v: c / z for v, c in prior.items()}

    def p(sector):
        cnt = by.get(sector, {})
        q = {v: (0 if (sector, v) in singles else cnt.get(v, 0)) + k * prior[v] for v in voc}
        s = sum(q.values())
        return {v: c / s for v, c in q.items()}
    return p


def cv(rows, val, voc, k):
    """Leave-one-occupation-out mean log-likelihood per held-out label."""
    tot, n = 0.0, 0
    for o in OCCUPATIONS:
        test = [r for r in rows if r['occupation'] == o]
        train = [r for r in rows if r['occupation'] != o]
        p = model(train, val, voc, k, frozenset(drop_single(train, val)))
        for r in test:
            v = val(r)
            for x in (v if isinstance(v, list) else [v]):
                if x is None:
                    continue
                tot += math.log(p(r['sector'])[x])
                n += 1
    return tot / max(n, 1)


DIMS = {   # name -> (value function, vocabulary)
    'task_type': (lambda r: r['task_type'], list(TASK_TYPES)),
    'has_inputs': (lambda r: r['has_inputs'], ['yes', 'no']),
    'input_ext': (lambda r: r['input_exts'] or None, INPUT_EXTS),
    'format': (lambda r: r['formats'] or None, DELIV_FMTS),
}


def fit():
    rows = load_rows()
    out = {'n_tasks': len(rows), 'dims': {}}
    for name, (val, voc) in DIMS.items():
        grid = {k: cv(rows, val, voc, k) for k in K_GRID}
        best = max(grid, key=grid.get)
        singles = drop_single(rows, val)
        p = model(rows, val, voc, best, frozenset(singles))
        by, glob = tally(rows, val)
        out['dims'][name] = {'k': best, 'cv_ll': grid, 'global': dict(glob),
                             'singles_merged': sorted(f'{s}|{v}' for s, v in singles),
                             'p': {s: p(s) for s in SECTORS_OCCUPATIONS}}
        print(f'{name:<11} best k={best:<5} ll={grid[best]:.3f}  (k=0.25: {grid[0.25]:.3f}, k=100: {grid[100]:.3f}) '
              f'singles merged: {len(singles)}')
    inp = [r for r in rows if r['n_inputs']]
    out['n_inputs'] = dict(collections.Counter(min(r['n_inputs'], 5) for r in inp))
    out['n_deliverables'] = dict(collections.Counter(min(r['n_deliverables'], 4) for r in rows))
    out['n_words'] = sorted(r['n_words'] for r in rows)
    prov = collections.Counter(p for r in rows for p in r['provenance'])
    out['provenance'] = dict(prov)
    out['needs_web'] = sum(r['needs_web'] for r in rows) / len(rows)
    print('n_inputs', out['n_inputs'], 'n_deliverables', out['n_deliverables'], 'provenance', dict(prov),
          'needs_web', round(out['needs_web'], 3))
    (WORK / 'sampler.json').write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------- sampling
REGIONS = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia',
           'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Tennessee',
           'Massachusetts', 'Indiana', 'Maryland', 'Missouri', 'Colorado', 'Wisconsin', 'Minnesota',
           'South Carolina', 'Alabama', 'Louisiana', 'Kentucky', 'Oregon', 'Oklahoma', 'Connecticut', 'Utah', 'Iowa',
           'Nevada', 'Arkansas', 'Kansas', 'Mississippi', 'New Mexico', 'Nebraska', 'Idaho', 'Hawaii', 'Maine',
           'New Hampshire', 'Rhode Island', 'Montana', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska',
           'Vermont', 'Wyoming', 'West Virginia']
ABROAD = ['Canada', 'the United Kingdom', 'Australia', 'Ireland', 'Germany', 'Singapore', 'India', 'Mexico']
ORG_SIZES = ['a small organisation (under 50 people)', 'a mid-sized organisation (50 to 1,000 people)',
             'a large organisation (over 1,000 people)', 'a public agency, nonprofit or institution',
             'an independent practice, consultancy or freelancer setting']


def pick(rng, p: dict, allowed=None):
    items = [(v, w) for v, w in p.items() if (allowed is None or v in allowed) and w > 0]
    if not items:
        return None
    tot = sum(w for _, w in items)
    x = rng.random() * tot
    for v, w in items:
        x -= w
        if x <= 0:
            return v
    return items[-1][0]


class Sampler:
    def __init__(self, fitted=None, leaves=None):
        self.f = fitted or json.loads((WORK / 'sampler.json').read_text())
        self.leaves = leaves or json.loads((WORK / 'leaves.json').read_text())
        missing = [o for o in OCCUPATIONS if not self.leaves.get(o)]
        assert not missing, missing
        self.compat = {o: {t for lf in self.leaves[o] for t in lf['task_types']} for o in OCCUPATIONS}

    def spec(self, idx: int, seed: str = 'gdpval-m5-v1') -> dict:
        rng = random.Random(hashlib.sha256(f'{seed}/{idx}'.encode()).hexdigest())
        occ = OCCUPATIONS[rng.randrange(len(OCCUPATIONS))]       # uniform over the 44
        sector = OCC_SECTOR[occ]
        d = self.f['dims']
        tt = pick(rng, d['task_type']['p'][sector], self.compat[occ]) or pick(rng, d['task_type']['p'][sector])
        has_inputs = pick(rng, d['has_inputs']['p'][sector]) == 'yes'
        lane = 'none'
        if has_inputs:
            lane = 'real' if rng.random() < REAL_SHARE else 'synthetic'
        is_open = rng.random() < OPEN_SHARE
        cands = [lf for lf in self.leaves[occ] if tt in lf['task_types']] or self.leaves[occ]
        if lane == 'real':
            pub = [lf for lf in cands if 'public_published' in lf.get('inputs', [])]
            if not pub:   # no activity of this type uses public documents: try the occupation's other activities
                pub = [lf for lf in self.leaves[occ] if 'public_published' in lf.get('inputs', [])]
            cands = pub or cands
            if not pub:
                lane = 'synthetic'
        elif lane == 'synthetic':
            cands = [lf for lf in cands if 'private_business' in lf.get('inputs', [])] or cands
        else:
            cands = [lf for lf in cands if 'none' in lf.get('inputs', [])] or cands
        leaf = rng.choice(cands)
        tt = tt if tt in leaf['task_types'] else rng.choice(leaf['task_types'])
        n_del = int(pick(rng, {k: v for k, v in self.f['n_deliverables'].items()}))
        fmts = []
        allowed = set(leaf['formats'])
        if allowed & {'docx', 'pptx', 'xlsx', 'md'}:
            allowed.add('pdf')   # any written or tabular deliverable can be asked for as a PDF (GDPval: pdf 38%)
        if tt in ('spreadsheet_model', 'data_processing', 'plan_schedule', 'form_template_checklist'):
            allowed.add('xlsx')  # tabular work products (pilot: xlsx 19% vs GDPval 27% with leaf formats only)
        for _ in range(n_del):
            f = pick(rng, d['format']['p'][sector], allowed - set(fmts)) or pick(rng, d['format']['p'][sector],
                                                                                  set(DELIV_FMTS) - set(fmts))
            fmts.append(f)
        if tt == 'presentation' and 'pptx' not in fmts:
            fmts[0] = 'pptx'
        n_inputs, exts = 0, []
        if lane != 'none':
            n_inputs = int(pick(rng, self.f['n_inputs']))
            k = n_inputs - 1 if lane == 'real' else n_inputs   # the real document is one of the inputs
            exts = [pick(rng, d['input_ext']['p'][sector]) for _ in range(k)]
        words = self.f['n_words'][rng.randrange(len(self.f['n_words']))]
        words = int(round(words * rng.uniform(0.85, 1.15), -1))
        where = rng.choice(REGIONS) if rng.random() < 0.9 else rng.choice(ABROAD)
        setting = f'{rng.choice(ORG_SIZES)} in {where}, year {rng.choice([2023, 2024, 2024, 2025, 2025, 2026])}'
        web_refs = rng.random() < self.f.get('needs_web', 0.4)
        template_input = lane == 'synthetic' and any(e in ('xlsx', 'docx', 'pptx') for e in exts) and \
            rng.random() < TEMPLATE_SHARE
        return {'web_refs': web_refs, 'template_input': template_input, 'idx': idx, 'seed': seed, 'occupation': occ, 'sector': sector, 'task_type': tt,
                'task_type_desc': TASK_TYPES[tt], 'leaf': {k: leaf[k] for k in ('name', 'description')},
                'open': is_open, 'lane': lane, 'n_inputs': n_inputs, 'input_exts': exts, 'formats': fmts,
                'n_deliverables': n_del, 'target_words': max(100, words), 'setting': setting}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fit', 'sample'])
    ap.add_argument('-n', type=int, default=20)
    ap.add_argument('--start', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'fit':
        fit()
    else:
        s = Sampler()
        for i in range(a.start, a.start + a.n):
            print(json.dumps(s.spec(i)))


if __name__ == '__main__':
    main()

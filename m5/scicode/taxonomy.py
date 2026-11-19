"""SciCode taxonomy: field x subfield x method, fitted to the 64 SciCode-Verified test problems, with brainstormed
breadth below the fitted levels (Opus 5.5 via claude -p).

    python -m m5.scicode.taxonomy fit          # weights from taxonomy/test_labels.json (class-level counts only)
    python -m m5.scicode.taxonomy areas        # stage 1: topic areas + method compatibility per subfield (Opus)
    python -m m5.scicode.taxonomy leaves       # stage 2: specific leaves per (subfield, method) cell (Opus)
    python -m m5.scicode.taxonomy build        # dedup + block list -> taxonomy/vocab.json

Fitted levels (sub-issue #352): the field marginal is the test set's, unflattened. Subfields and methods are the
hand-labelled test classes, with every class seen in a single test problem merged into a sibling. Smoothing:
P(sub|field) gets 0.5 pseudo-counts per subfield; P(method|field) is shrunk toward the global method marginal (3
pseudo-observations); P(method|subfield) is the subfield's own counts shrunk toward that (2 pseudo-observations),
then multiplied by Opus' compatibility of (subfield, method) (0, 0.4, 1; a pair seen in the
test set counts as at least 'occasional'). Only these class-level numbers and the class descriptions in
classes.json reach the brainstorm and the generator; test items never do.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import math
import random
import re
from pathlib import Path

from . import opus
from .common import safe_write

HERE = Path(__file__).resolve().parent / 'taxonomy'
CLASSES = json.loads((HERE / 'classes.json').read_text())
WORK = opus.WORK
N_PLAN = 9500            # specs to draw: 6.6k problems x 1.4 oversupply + pilot + margin
STEPS_ALPHA, SUB_ALPHA, METH_LAMBDA, SUB_METH_LAMBDA = 0.25, 0.5, 3.0, 2.0
COMPAT_W = {0: 0.0, 1: 0.4, 2: 1.0}


def labels():
    return json.loads((HERE / 'test_labels.json').read_text())['labels']


def fit():
    lab = labels().values()
    n = len(lab)
    fc = collections.Counter(x['field'] for x in lab)
    sc = collections.Counter(x['subfield'] for x in lab)
    mc = collections.Counter(x['method'] for x in lab)
    fm = collections.Counter((x['field'], x['method']) for x in lab)
    sm = collections.Counter((x['subfield'], x['method']) for x in lab)
    steps = collections.Counter(x['n_steps'] for x in lab)
    assert min(sc.values()) >= 2 and min(mc.values()) >= 2, 'a class rests on a single test problem'
    p_field = {f: fc[f] / n for f in CLASSES['fields']}
    p_sub = {}
    for s in CLASSES['subfields']:
        f = s.split('/')[0]
        k = sum(1 for t in CLASSES['subfields'] if t.startswith(f + '/'))
        p_sub[s] = (sc[s] + SUB_ALPHA) / (fc[f] + SUB_ALPHA * k)
    p_meth_global = {m: mc[m] / n for m in CLASSES['methods']}
    p_meth_field = {f: {m: (fm[(f, m)] + METH_LAMBDA * p_meth_global[m]) / (fc[f] + METH_LAMBDA)
                        for m in CLASSES['methods']} for f in CLASSES['fields']}
    p_steps = {k: (steps[k] + STEPS_ALPHA) / (n + 15 * STEPS_ALPHA) for k in range(1, 16)}
    return {'n_test': n, 'p_field': p_field, 'p_sub_given_field': p_sub, 'p_method_global': p_meth_global,
            'p_method_given_field': p_meth_field, 'p_steps': p_steps,
            'sm_counts': {f'{s}|{m}': c for (s, m), c in sm.items()}, 'sub_counts': dict(sc),
            'test_pairs': sorted(f'{s}|{m}' for s, m in sm), 'mean_steps': sum(k * v for k, v in p_steps.items())}


SYSTEM = """You are a senior computational scientist helping design a large, diverse curriculum of research-style scientific programming problems.

Each problem in the curriculum is a multi-step Python implementation task (typically 2-6 steps, occasionally 1 or up to 15) written the way a domain scientist would set it: a short main description of a scientific computation, then a chain of steps where each step asks for one function (or class) with a precise mathematical specification, and later steps build on the functions of earlier steps to produce a final scientific quantity. Solutions use only numpy, scipy, sympy and the Python standard library, run in seconds to minutes on a CPU, and return well-defined numerical (or structured) outputs that a test could check.

Problems range from graduate-course exercises to small research computations across physics, chemistry, materials science, biology and applied mathematics."""


def cell_compat(areas):
    out = {}
    for s, a in areas.items():
        for m, v in (a.get('compat') or {}).items():
            out[(s, m)] = int(v)
    return out


def stage_areas(jobs=5):
    meth = '\n'.join(f'- {m}: {d}' for m, d in CLASSES['methods'].items())

    def one(s):
        user = f"""Subfield: {CLASSES['fields'][s.split('/')[0]]} / {CLASSES['subfields'][s]}

Method families used across the curriculum:
{meth}

Tasks:
1. List 45-70 distinct topic areas inside this subfield that are good sources of such problems. Cover the subfield broadly (classic and modern, theory-heavy and application-heavy, different physical/biological/chemical systems and scales), not just the most famous textbook examples. Each area: a short name (3-8 words) and one sentence on the kinds of computations it offers.
2. Rate how naturally each method family serves problems in this subfield: 2 = common, 1 = occasional, 0 = rare or unnatural.

Reply with JSON only, in one ```json block:
{{"areas": [{{"name": "...", "about": "..."}}, ...], "compat": {{"<method>": 0|1|2, ...}}}}"""
        return s, opus.ask_json(SYSTEM, user, 'areas-' + s.replace('/', '-'))

    out = {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        for s, r in ex.map(one, CLASSES['subfields']):
            out[s] = r
    safe_write(WORK / 'areas.json', json.dumps(out, indent=1))
    return out


def cell_weights(fitd, areas):
    compat = cell_compat(areas)
    tp = {tuple(x.split('|')) for x in fitd['test_pairs']}
    w = {}
    for s in CLASSES['subfields']:
        f = s.split('/')[0]
        raw = {}
        for m in CLASSES['methods']:
            c = max(compat.get((s, m), 0), 1 if (s, m) in tp else 0)
            ps = ((fitd['sm_counts'].get(f'{s}|{m}', 0) + SUB_METH_LAMBDA * fitd['p_method_given_field'][f][m])
                  / (fitd['sub_counts'][s] + SUB_METH_LAMBDA))
            raw[m] = ps * COMPAT_W[c]
        z = sum(raw.values())
        for m, v in raw.items():
            if v > 0:
                w[f'{s}|{m}'] = fitd['p_field'][f] * fitd['p_sub_given_field'][s] * v / z
    return w


def leaf_targets(w):
    return {c: max(12, math.ceil(1.1 * N_PLAN * p)) for c, p in w.items()}


def stage_leaves(jobs=6, per_call=90):
    fitd = fit()
    areas = json.loads((WORK / 'areas.json').read_text())
    w = cell_weights(fitd, areas)
    tgt = leaf_targets(w)
    print(f'{len(tgt)} cells, {sum(tgt.values())} leaves wanted', flush=True)

    def run_cell(cell):
        s, m = cell.split('|')
        rng = random.Random(cell)
        al = areas[s]['areas']
        got, i = [], 0
        while len(got) < tgt[cell] and i < 40:
            k = min(per_call, tgt[cell] - len(got) + 10)
            focus = rng.sample(al, min(10, len(al)))
            prev = '\n'.join(f'- {x["title"]}' for x in got) or '(none yet)'
            user = f"""Subfield: {CLASSES['fields'][s.split('/')[0]]} / {CLASSES['subfields'][s]}
Method family: {m}: {CLASSES['methods'][m]}

Propose {k} specific problem topics ("leaves") in this subfield whose core computation uses this method family. Draw mainly on these topic areas this time (others are fine if they fit better):
{chr(10).join(f'- {a["name"]}: {a["about"]}' for a in focus)}

Each leaf names one concrete scientific system or model and the quantity to compute, specific enough that two problem writers given the same leaf would write closely related problems, e.g. the model or equation, the observable, and the main computational ingredients. Requirements:
- distinct from each other and from the leaves already listed below (no rewordings, no same system with a trivially different observable);
- implementable as a chain of 1-15 Python functions with numpy/scipy/sympy, running in seconds to minutes, with well-defined outputs;
- a realistic mix of difficulty: mostly graduate-course level, some simpler, some small-research level;
- varied systems, scales and applications, not only the canonical textbook example of each area.

Leaves already listed for this cell:
{prev}

Reply with JSON only, in one ```json block: [{{"title": "<= 14 words", "gist": "1-2 sentences: system/model, what is computed, main computational ingredients", "area": "<topic area name>"}}, ...]"""
            try:
                r = opus.ask_json(SYSTEM, user, f'leaves-{s.replace("/", "-")}-{m}-{i}')
            except ValueError:
                i += 1
                continue
            seen = {norm(x['title']) for x in got}
            for x in r if isinstance(r, list) else []:
                if isinstance(x, dict) and x.get('title') and x.get('gist') and norm(x['title']) not in seen:
                    seen.add(norm(x['title']))
                    got.append({'title': x['title'].strip(), 'gist': x['gist'].strip(), 'area': x.get('area', '')})
            i += 1
        return cell, got

    out = {}
    cells = sorted(tgt, key=lambda c: -tgt[c])
    with cf.ThreadPoolExecutor(jobs) as ex:
        for cell, got in ex.map(run_cell, cells):
            out[cell] = got
            print(f'{cell}: {len(got)}/{tgt[cell]}', flush=True)
    safe_write(WORK / 'leaves_raw.json', json.dumps(out, indent=1))
    return out


def stage_breadth(jobs=6, per_call=90, margin=1.15, n_traced=9000):
    """Round 2 (after leaf_merge): more leaves for the cells whose distinct leaves fall short of their expected
    traced draws (margin x n_traced x weight). The prompt lists every kept leaf of the subfield as subjects to avoid,
    so the new leaves add subjects instead of rewording old ones; leaf_merge then checks them against everything."""
    fitd = fit()
    areas = json.loads((WORK / 'areas.json').read_text())
    w = cell_weights(fitd, areas)
    raw = json.loads((WORK / 'leaves_raw.json').read_text())
    merged = json.loads((HERE / 'leaf_merge.json').read_text())
    blocked = set(json.loads((HERE / 'leaf_block.json').read_text())['block_titles'])
    drop = set(merged['drop_titles']) | blocked
    kept = {c: [x for x in xs if x['title'] not in drop] for c, xs in raw.items()}
    need = {c: math.ceil(margin * n_traced * p) - len(kept.get(c, [])) for c, p in w.items()}
    need = {c: k for c, k in need.items() if k > 0}
    print(f'{len(need)} cells short, {sum(need.values())} leaves wanted', flush=True)
    by_sub = collections.defaultdict(list)
    for c, xs in kept.items():
        by_sub[c.split('|')[0]] += [x['title'] for x in xs]

    def run_cell(cell):
        s, m = cell.split('|')
        rng = random.Random('b2' + cell)
        al = areas[s]['areas']
        got, i = [], 0
        while len(got) < need[cell] and i < 12:
            k = min(per_call, need[cell] - len(got) + 10)
            focus = rng.sample(al, min(12, len(al)))
            avoid = '\n'.join(f'- {t}' for t in by_sub[s] + [x['title'] for x in got])
            user = f"""Subfield: {CLASSES['fields'][s.split('/')[0]]} / {CLASSES['subfields'][s]}
Method family: {m}: {CLASSES['methods'][m]}

Propose {k} NEW specific problem topics ("leaves") in this subfield whose core computation uses this method family. The subjects listed at the end are already covered (in any method family): do not repeat any of them, reworded or with a different method, parameter or observable of the same system. Go for different systems, models, phenomena, scales and applications; less common corners of the subfield are welcome as long as they stay standard enough for a domain scientist. Draw on these topic areas this time (others are fine):
{chr(10).join(f'- {a["name"]}: {a["about"]}' for a in focus)}

Each leaf names one concrete scientific system or model and the quantity to compute, specific enough that two problem writers given the same leaf would write closely related problems. Requirements: implementable as a chain of 1-15 Python functions with numpy/scipy/sympy, running in seconds to minutes, with well-defined outputs; mostly graduate-course level, some simpler, some small-research level.

Subjects already covered:
{avoid}

Reply with JSON only, in one ```json block: [{{"title": "<= 14 words", "gist": "1-2 sentences: system/model, what is computed, main computational ingredients", "area": "<topic area name>"}}, ...]"""
            try:
                r = opus.ask_json(SYSTEM, user, f'breadth-{s.replace("/", "-")}-{m}-{i}')
            except ValueError:
                i += 1
                continue
            seen = {norm(t) for t in by_sub[s]} | {norm(x['title']) for x in got}
            for x in r if isinstance(r, list) else []:
                if isinstance(x, dict) and x.get('title') and x.get('gist') and norm(x['title']) not in seen:
                    seen.add(norm(x['title']))
                    got.append({'title': x['title'].strip(), 'gist': x['gist'].strip(), 'area': x.get('area', ''),
                                'round': 2})
            i += 1
        return cell, got

    out = {}
    with cf.ThreadPoolExecutor(jobs) as ex:
        for cell, got in ex.map(run_cell, sorted(need, key=lambda c: -need[c])):
            out[cell] = got
            print(f'{cell}: {len(got)}/{need[cell]}', flush=True)
    safe_write(WORK / 'leaves_round2.json', json.dumps(out, indent=1))
    return out


def all_leaf_rows():
    """[(cell, leaf)] of round 1 (leaves_raw.json) followed by round 2 (leaves_round2.json, if any): appending keeps
    the round-1 indices stable for the judges' resumable logs."""
    rows = [(c, x) for c, xs in json.loads((WORK / 'leaves_raw.json').read_text()).items() for x in xs]
    p2 = WORK / 'leaves_round2.json'
    if p2.exists():
        rows += [(c, x) for c, xs in json.loads(p2.read_text()).items() for x in xs]
    return rows


def norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['fit', 'areas', 'leaves', 'breadth', 'build'])
    ap.add_argument('--jobs', type=int, default=5)
    a = ap.parse_args()
    if a.stage == 'fit':
        print(json.dumps(fit(), indent=1))
    elif a.stage == 'areas':
        stage_areas(a.jobs)
    elif a.stage == 'leaves':
        stage_leaves(a.jobs)
    elif a.stage == 'breadth':
        stage_breadth(a.jobs)
    elif a.stage == 'build':
        from . import build_vocab
        build_vocab.main()


if __name__ == '__main__':
    main()

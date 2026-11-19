"""Fit the taxonomy to GDP.pdf's class-level statistics (taxonomy/taxonomy.json).

Inputs (class-level only; no test text reaches the generator):
- test request labels (label_requests.py --test): task type, asks, inline material, format, persona, length;
- test document labels (docs/classify.py --test): document type per domain;
- the rubric's Dodged-Bullet counts per task (pitfalls per request);
- brainstormed leaves (brainstorm.py).
Domain weights: half the text-only-v1 set (75 tasks, what the eval scores), half all 100 (breadth). Per-domain
distributions are shrunk toward the global ones: P(x|D) = (n_Dx + a * P(x)) / (n_D + a), a = 10, so a domain cell
never rests on a single test task. Lengths: the test prompts' word counts pooled over domains (by inline vs not).

  python -m m5.gdppdf.taxonomy.fit --req-labels R.jsonl --doc-labels D.jsonl --leaves leaves.json --out taxonomy.json
"""
import argparse
import collections
import json
from pathlib import Path

from m5.gdppdf.docs.classify import DOC_TYPES, DOMAINS
from m5.gdppdf.taxonomy.label_requests import FORMATS, TASK_TYPES

ALPHA = 10
INLINE_KINDS = ['case_scenario', 'client_or_patient_facts', 'added_rules_or_checklist', 'data_or_list', 'other']


def norm(c, keys, floor=0.0):
    tot = sum(c.get(k, 0) + floor for k in keys)
    return {k: (c.get(k, 0) + floor) / tot for k in keys}


def shrink(counts_by_d, keys, alpha=ALPHA, floor=0.5):
    glob = collections.Counter()
    for c in counts_by_d.values():
        glob.update(c)
    g = norm(glob, keys, floor)
    out = {}
    for d, c in counts_by_d.items():
        # a cell resting on a single test task gets no domain-specific mass (#19: no leaf from one test item); it
        # keeps only the global prior's share
        c = collections.Counter({k: v for k, v in c.items() if v >= 2})
        n = sum(c.values())
        out[d] = {k: (c.get(k, 0) + alpha * g[k]) / (n + alpha) for k in keys}
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--req-labels', required=True)
    ap.add_argument('--doc-labels', required=True)
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from aa_evals.gdppdf import data
    tasks = {t.task_id: t for t in data.load_tasks()}
    req = {r['id']: r for r in map(json.loads, open(a.req_labels))}
    doc = {r['file_name']: r for r in map(json.loads, open(a.doc_labels))}

    # domains: half text-only share, half all-100 share
    allc = collections.Counter(t.domain for t in tasks.values())
    toc = collections.Counter(t.domain for t in data.task_set(list(tasks.values()), 'text-only-v1'))
    domains = {d: 0.5 * allc[d] / sum(allc.values()) + 0.5 * toc[d] / sum(toc.values()) for d in DOMAINS}

    by_d = lambda f: {d: collections.Counter(f(i) for i, t in tasks.items() if t.domain == d and f(i) is not None)
                      for d in DOMAINS}
    lab = lambda i: (req.get(i) or {}).get('label') or {}
    tt_of = lambda i: lab(i).get('task_type') if lab(i).get('task_type') in TASK_TYPES else None
    task_types, tt_glob = shrink(by_d(tt_of), list(TASK_TYPES))
    dt_of = lambda i: ((doc.get(i) or {}).get('label') or {}).get('doc_type') if (
        ((doc.get(i) or {}).get('label') or {}).get('doc_type') in DOC_TYPES) else None
    doc_types, dt_glob = shrink(by_d(dt_of), list(DOC_TYPES), floor=0.25)

    # per task type (pooled over domains, shrunk toward all tasks)
    def per_tt(f, keys, floor=0.5):
        c = {tt: collections.Counter(f(i) for i in tasks if tt_of(i) == tt and f(i) is not None) for tt in TASK_TYPES}
        return shrink(c, keys, alpha=8, floor=floor)[0]

    def asks(i):
        n = lab(i).get('n_asks')
        return str(min(6, max(1, int(n)))) if isinstance(n, (int, float)) else None
    n_asks = per_tt(asks, [str(k) for k in range(1, 7)], floor=0.2)
    fmt = per_tt(lambda i: lab(i).get('output_format') if lab(i).get('output_format') in FORMATS else None, FORMATS,
                 floor=0.3)
    inline_c = {tt: [bool(lab(i).get('inline_material')) for i in tasks if tt_of(i) == tt] for tt in TASK_TYPES}
    all_inline = [bool(lab(i).get('inline_material')) for i in tasks if lab(i)]
    p_all = sum(all_inline) / len(all_inline)
    inline = {tt: (sum(v) + 8 * p_all) / (len(v) + 8) for tt, v in inline_c.items()}
    inline_kind = per_tt(lambda i: lab(i).get('inline_kind') if lab(i).get('inline_material') and
                         lab(i).get('inline_kind') in INLINE_KINDS else None, INLINE_KINDS, floor=0.3)
    def cons(i):
        c = lab(i).get('constraints')
        return str(min(4, int(c))) if isinstance(c, (int, float)) else None
    constraints = per_tt(cons, [str(k) for k in range(5)], floor=0.3)
    persona = (sum(bool(lab(i).get('persona')) for i in tasks if lab(i)) + 1) / (len([i for i in tasks if lab(i)]) + 2)
    db = collections.Counter(min(4, sum(c.meta.get('type') == 'Dodged Bullet' for c in t.criteria))
                             for t in tasks.values())
    pitfalls = norm({str(k): v for k, v in db.items()}, [str(k) for k in range(5)], floor=0.5)
    wl = {'inline': [len(t.prompt.split()) for i, t in tasks.items() if lab(i).get('inline_material')],
          'plain': [len(t.prompt.split()) for i, t in tasks.items() if lab(i) and not lab(i).get('inline_material')]}

    leaves = json.loads(Path(a.leaves).read_text())
    tax = {'version': 'v1', 'alpha': ALPHA, 'domains': domains, 'doc_types': doc_types, 'doc_types_global': dt_glob,
           'task_types': task_types, 'task_types_global': tt_glob, 'n_asks': n_asks, 'pitfalls': pitfalls,
           'inline': inline, 'inline_kind': inline_kind, 'format': fmt, 'persona': persona,
           'constraints': constraints,
           'words': {k: sorted(v) for k, v in wl.items()}, 'leaves': leaves,
           'note': 'class-level statistics of GDP.pdf (100 tasks; domains half text-only-v1); per-domain cells '
                   f'shrunk toward global counts with a={ALPHA}; leaves brainstormed by Opus (no test items)'}
    Path(a.out).write_text(json.dumps(tax, indent=1) + '\n')
    print(json.dumps({k: tax[k] for k in ('domains', 'task_types_global', 'doc_types_global', 'pitfalls', 'inline',
                                           'persona')}, indent=1))


if __name__ == '__main__':
    main()

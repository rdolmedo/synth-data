"""Fit the top two taxonomy levels to AA-Omniscience: domain and subtopic weights.

Only class-level statistics leave this script (counts per label). Test questions never enter generator context.
- Domains: v1.1's fixed split (the public set is a stratified 10% of the private 3,000): 20/20/15/15/15/15%.
- Subtopics: AA's 86 v1.1 labels. Within a domain, p = 0.6 * v1.1 share + 0.25 * v1.0 share + 0.15 * uniform
  (v1.0-only labels are dropped: they are not in v1.1's label set, and no new top-level options are added).
"""
import collections, csv, json, sys
from pathlib import Path

D = Path('/data/aai/aa_evals/data/omniscience')
V11 = D / 'public-v1.1@1f9cbce/AA-Omniscience_dataset_public.csv'
V10 = D / 'public-v1@e4883ed/AA-Omniscience_dataset_public.csv'


def rows(p):
    return list(csv.DictReader(open(p, encoding='utf-8')))


def main(out):
    r11, r10 = rows(V11), rows(V10)
    dom = collections.Counter(r['domain'] for r in r11)
    c11 = collections.Counter((r['domain'], r['subtopic']) for r in r11)
    c10 = collections.Counter((r['domain'], r['subtopic']) for r in r10)
    tax = {'source': 'AA-Omniscience public v1.1 (label counts) + v1.0 (label counts)', 'domains': {}}
    for d in sorted(dom):
        subs = sorted(s for (dd, s) in c11 if dd == d)
        n11 = sum(c11[(d, s)] for s in subs)
        n10 = sum(c10[(d, s)] for s in subs) or 1
        w = {s: 0.6 * c11[(d, s)] / n11 + 0.25 * c10[(d, s)] / n10 + 0.15 / len(subs) for s in subs}
        z = sum(w.values())
        tax['domains'][d] = {'weight': dom[d] / len(r11),
                             'subtopics': {s: {'weight': round(w[s] / z, 5), 'n_v11': c11[(d, s)],
                                               'n_v10': c10[(d, s)]} for s in subs}}
    Path(out).write_text(json.dumps(tax, indent=1, ensure_ascii=False))
    for d, v in tax['domains'].items():
        print(f"{d}: {v['weight']:.3f}, {len(v['subtopics'])} subtopics")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name('subtopics.json'))

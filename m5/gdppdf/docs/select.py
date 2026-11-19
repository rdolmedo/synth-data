"""Select the documents of the cache from the classified PDFA pool, matching GDP.pdf's class-level mix.

  python -m m5.gdppdf.docs.select --labels <classified.jsonl> --taxonomy <taxonomy.json> --n 6500 --out sel.jsonl

Filters: usable, one of the 10 domains, richness >= 2, estimated Super tokens <= 95k (~1.9 tokens per PDFA
word; the index then drops >92k by exact count), not a test PDF (sha256), not from a test document's issuer + series (exclusions.py), one copy per near-duplicate
(title+issuer, snippet MinHash), at most 3 documents per host and 2 per issuer+series.
Quotas: N x domain weight x document type | domain (taxonomy.json), and within each cell GDP.pdf's page-count bins
(half text-only-v1, half all 100); a short bin hands its quota to the next longer bin, then shorter ones; a short
cell hands its quota to the domain's other document types.
"""
import argparse
import collections
import glob
import hashlib
import json
import random
import re

from m5.gdppdf.docs.classify import DOC_TYPES, DOMAINS
from m5.gdppdf.docs.exclusions import excluded
from m5.gdppdf.docs.store import read_zst_jsonl

TOK_PER_WORD = 1.9   # median Super tokens per PDFA word on extracted docs (1.4-3.0)
MAX_EST_TOKENS = 95000
# GDP.pdf page bins over the test documents whose text fits the eval's window (89 of 100; the others score 0 as
# input_too_long whatever the model): (lo, hi, share) = mean of the text-only-v1 (68 fit) and all-100 shares
BINS = [(1, 4, 0.056), (5, 9, 0.184), (10, 19, 0.178), (20, 39, 0.276), (40, 79, 0.173), (80, 119, 0.095),
        (120, 300, 0.039)]


# A domain short of documents may borrow documents labelled with another domain whose text is about its work (the
# domain is the requester's profession: a health plan's benefit booklet serves an insurance claims analyst too).
BORROW = {
    'Insurance': re.compile(r'(insurer|insured|insurance|policyholder|underwrit|deductible|premium|reinsur|actuar|'
                            r'evidence of coverage|copay|coinsurance|endorsement)', re.I),
    'Manufacturing/Supply Chains': re.compile(r'(manufactur|supplier|supply chain|inventory|warehouse|logistic|'
                                              r'procurement|production|quality (control|assurance|manual)|'
                                              r'packaging|shipping|distribution center)', re.I),
    'HR': re.compile(r'(employee|personnel|human resources|payroll|benefits|leave|collective bargaining|'
                     r'compensation|recruit|workforce)', re.I),
    'Construction': re.compile(r'(construction|contractor|specification|bid\b|tender|building code|site work|'
                               r'concrete|earthwork|change order|submittal)', re.I),
    'Real Estate': re.compile(r'(real estate|property|lease|tenant|landlord|zoning|appraisal|mortgage|housing|'
                              r'homeowners|condominium|parcel)', re.I),
    'Engineering': re.compile(r'(engineering|design criteria|specification|load|pump|valve|hydraulic|electrical|'
                              r'mechanical|structural)', re.I),
}


def bin_of(n):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= n <= hi:
            return i
    return len(BINS) - 1


def shingles(t, k=5):
    ws = re.findall(r'[a-z0-9]+', t.lower())
    return {' '.join(ws[i:i + k]) for i in range(max(0, len(ws) - k + 1))}


_P = (1 << 61) - 1
_RS = __import__('numpy').random.RandomState(357)
_A = _RS.randint(1, 1 << 30, size=64).astype('uint64')
_B = _RS.randint(0, 1 << 30, size=64).astype('uint64')


def minhash(sh, n=64):
    """64 MinHash values from one 30-bit hash per shingle and universal hashing (vectorised)."""
    import numpy as np
    if not sh:
        return None
    h = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), 'little') & ((1 << 30) - 1)
                     for s in sh), dtype='uint64')
    v = (np.outer(h, _A[:n]) + _B[:n]) % np.uint64(_P)
    return tuple(int(x) for x in v.min(axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--triage', default='/data/aai/m5/shared/docs/cc2021/triage/pdfa-*.jsonl.zst')
    ap.add_argument('--provenance', default='/data/aai/m5/shared/docs/cc2021/meta/provenance.parquet')
    ap.add_argument('--taxonomy', required=True)
    ap.add_argument('--test-texts', default='/data/aai/aa_evals/data/gdppdf/prepared/'
                    'liteparse-2.5.0_eng_tessbest-e12c65a9/texts.jsonl')
    ap.add_argument('--n', type=int, default=6500)
    ap.add_argument('--have', nargs='*', default=[], help='earlier selections: their documents are kept out of the '
                    'pool and count against the quotas (a top-up)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=357)
    ap.add_argument('--long-cap', type=float, default=MAX_EST_TOKENS, help='estimate cap for 80+ page documents')
    ap.add_argument('--long-boost', action='store_true', help='after the quotas, fill the 80+ page bins of the '
                    'combined (--have + new) selection from any cell, domain-proportionally')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tax = json.load(open(a.taxonomy))
    tri = {}
    for p in sorted(glob.glob(a.triage)):
        for r in read_zst_jsonl(p):
            tri[r['id']] = r
    prov = {}
    try:
        import pandas as pd
        df = pd.read_parquet(a.provenance, columns=['id', 'url', 'host'])
        prov = dict(zip(df['id'], zip(df['url'], df['host'])))
    except Exception as e:
        print(f'no provenance ({e}); URLs left empty', flush=True)
    test_sha = {json.loads(l)['pdf_sha256'] for l in open(a.test_texts)}
    have = [json.loads(l) for p in a.have for l in open(p) if l.strip()]
    prev = {d['doc_id'] for d in have}
    have_n = collections.Counter((d['domain'], d['doc_type'], bin_of(d['n_pages'])) for d in have)
    drops = collections.Counter()
    pool = []
    for l in open(a.labels):
        r = json.loads(l)
        lab, t = r.get('label') or {}, tri.get(r['file_name'])
        if not t:
            drops['no_triage'] += 1
            continue
        if lab.get('usable') is not True:
            drops['unusable'] += 1
            continue
        if lab.get('domain') not in DOMAINS or lab.get('doc_type') not in DOC_TYPES:
            drops['no_domain'] += 1
            continue
        if not isinstance(lab.get('richness'), (int, float)) or lab['richness'] < 2:
            drops['thin'] += 1
            continue
        est = TOK_PER_WORD * t['words']
        # long documents get a looser estimate cap (tokens per word vary 1.4-3.0; the index drops any whose exact
        # Super count passes 92k), so the 80+ page bins can reach GDP.pdf's shares
        if est > (a.long_cap if t['n_pages'] >= 80 else MAX_EST_TOKENS):
            drops['too_long'] += 1
            continue
        if t.get('sha256') in test_sha:
            drops['test_pdf'] += 1
            continue
        url, host = prov.get(t['id'], (None, None))
        hit = excluded(lab.get('issuer'), lab.get('series'), lab.get('title'), url, (t.get('snippet') or '')[:1500])
        if hit:
            drops['test_issuer_series'] += 1
            continue
        doc_id = f'cc21-{t["id"]}'
        if doc_id in prev:
            drops['already_selected'] += 1
            continue
        pool.append({'_text': ' '.join(str(x) for x in (lab.get('title'), lab.get('summary'), lab.get('series'),
                                                       (t.get('snippet') or '')[:3000])),
                     'doc_id': doc_id, 'source': 'pdfa-eng-wds', 'id': t['id'], 'shard': t['shard'],
                     'domain': lab['domain'], 'doc_type': lab['doc_type'], 'title': lab.get('title'),
                     'issuer': lab.get('issuer'), 'series': lab.get('series'), 'year': lab.get('year'),
                     'summary': lab.get('summary'), 'richness': lab['richness'], 'url': url, 'host': host,
                     'n_pages': t['n_pages'], 'words': t['words'], 'est_tokens': int(est), 'sha256': t['sha256'],
                     '_mh': minhash(shingles(t.get('snippet') or ''))})
    print(f'pool {len(pool)}; drops {dict(drops)}', flush=True)
    # near-duplicates: same title+issuer, or MinHash bands (8 bands x 8 rows) collide
    rng.shuffle(pool)
    pool.sort(key=lambda d: -d['richness'])
    seen_ti, bands, kept = set(), set(), []
    for d in pool:
        ti = (re.sub(r'\W+', ' ', str(d['title']).lower()).strip(), str(d['issuer']).lower())
        bs = {(b, d['_mh'][b * 8:(b + 1) * 8]) for b in range(8)} if d['_mh'] else set()
        if ti in seen_ti or bs & bands:
            drops['near_duplicate'] += 1
            continue
        seen_ti.add(ti)
        bands |= bs
        kept.append(d)
    pool = kept
    cells = collections.defaultdict(list)
    for d in pool:
        cells[(d['domain'], d['doc_type'])].append(d)
    host_n, series_n = collections.Counter(), collections.Counter()
    for d in have:   # earlier documents count against the host and series caps too
        if d.get('host'):
            host_n[d['host']] += 1
        series_n[(str(d['issuer']).lower(), str(d['series']).lower())] += 1

    def take(cands, n):
        out = []
        for d in cands:
            if len(out) >= n:
                break
            sk = (str(d['issuer']).lower(), str(d['series']).lower())
            if d['host'] and host_n[d['host']] >= 3 or series_n[sk] >= 2:
                continue
            out.append(d)
            if d['host']:
                host_n[d['host']] += 1
            series_n[sk] += 1
        return out

    selected, report = [], {}
    for dom in DOMAINS:
        q_dom = a.n * tax['domains'][dom]
        dts = tax['doc_types'][dom]
        got_dom = []
        carry = 0.0
        order = sorted(dts, key=lambda t: -dts[t])
        for dt in order:
            q = q_dom * dts[dt] + carry
            cands = cells.get((dom, dt), [])
            by_bin = collections.defaultdict(list)
            for d in cands:
                by_bin[bin_of(d['n_pages'])].append(d)
            for v in by_bin.values():
                # weighted shuffle (Efraimidis-Spirakis), weight = estimated tokens per page: within a page bin, lean
                # toward denser documents (GDP.pdf's fitting test documents average ~815 Super tokens per page, the
                # CC pool ~570), so the text lengths come closer to the eval's as well as the page counts
                v.sort(key=lambda d: -rng.random() ** (1.0 / max(50.0, d['est_tokens'] / max(1, d['n_pages']))))
            got = []
            spill = 0.0
            for i in reversed(range(len(BINS))):   # longest first: spill a shortage downward
                want = int(round(q * BINS[i][2] + spill))
                already = have_n[(dom, dt, i)]
                g = take(by_bin[i], max(0, want - already))
                spill = max(0, want - already - len(g))
                got += g
            if spill > 0:                            # still short: any remaining doc of the cell
                rest = [d for d in cands if d not in got]
                got += take(rest, int(spill))
            n_have = sum(have_n[(dom, dt, i)] for i in range(len(BINS)))
            carry = max(0.0, q - len(got) - n_have)
            got_dom += got
            report[f'{dom} | {dt}'] = {'quota': round(q, 1), 'got': len(got), 'avail': len(cands)}
        n_have_dom = sum(v for (d0, _, _), v in have_n.items() if d0 == dom)
        report[dom] = {'quota': round(q_dom), 'got': len(got_dom) + n_have_dom, 'new': len(got_dom),
                       'short': round(carry)}
        selected += got_dom
    # second pass: short domains borrow keyword-matching documents from the unselected pool of other domains
    chosen = {d['doc_id'] for d in selected}
    for dom in DOMAINS:
        short = report[dom]['quota'] - report[dom]['got']
        if short <= 0 or dom not in BORROW:
            continue
        hits = lambda d: BORROW[dom].findall(d['_text'])
        cands = [d for d in pool if d['doc_id'] not in chosen and d['domain'] != dom
                 and len(hits(d)) >= 5 and len({h.lower() for h in hits(d)}) >= 3
                 and d['doc_type'] not in ('research_article',)]
        rng.shuffle(cands)
        cands.sort(key=lambda d: -len(BORROW[dom].findall(d['_text'])))
        got = take(cands, int(short))
        for d in got:
            d['domain_labelled'] = d['domain']
            d['domain'] = dom
            chosen.add(d['doc_id'])
        selected += got
        report[dom]['borrowed'] = len(got)
        report[dom]['got'] += len(got)
    if a.long_boost:   # the combined selection's long bins up to their targets, from any cell
        chosen = {d['doc_id'] for d in selected}
        allsel = have + selected
        n_all = len(allsel)
        for i in (len(BINS) - 1, len(BINS) - 2):
            want = int(round(BINS[i][2] * n_all)) - sum(bin_of(d['n_pages']) == i for d in allsel)
            if want <= 0:
                continue
            cands = [d for d in pool if d['doc_id'] not in chosen and bin_of(d['n_pages']) == i]
            rng.shuffle(cands)
            by_dom = collections.defaultdict(list)
            for d in cands:
                by_dom[d['domain']].append(d)
            added = []
            while len(added) < want and any(by_dom.values()):
                for dom in sorted(by_dom, key=lambda x: -tax['domains'][x]):
                    if by_dom[dom] and len(added) < want:
                        g = take([by_dom[dom].pop()], 1)
                        added += g
            for d in added:
                chosen.add(d['doc_id'])
            selected += added
            report[f'long_boost_bin_{BINS[i][0]}-{BINS[i][1]}'] = {'wanted': want, 'added': len(added),
                                                                   'avail': len(cands)}
    for d in selected:
        d.pop('_mh', None)
        d.pop('_text', None)
    with open(a.out, 'w') as f:
        for d in selected:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    bins = collections.Counter(bin_of(d['n_pages']) for d in selected)
    print(json.dumps({'selected': len(selected), 'drops': dict(drops),
                      'domains': {d: report[d] for d in DOMAINS},
                      'page_bins': {f'{BINS[i][0]}-{BINS[i][1]}': (bins[i], round(bins[i] / max(1, len(selected)), 3),
                                                                   BINS[i][2]) for i in range(len(BINS))}}, indent=1))
    with open(a.out + '.report.json', 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()

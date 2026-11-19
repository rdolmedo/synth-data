"""Build the passage pool: retrieve source passages for every taxonomy leaf (one CPU job, nothing indexed on disk).

Sources are read in place. Per source type an in-memory BM25 index is built over ~700-word windows; Wikipedia uses
the existing mmlu-search BM25 index (2018 dump, 100-word chunks, expanded to +-3 neighbouring chunks of the same
article). Each leaf's 4 queries go to the leaf's preferred sources; candidates are interleaved across sources and
assigned greedily so that every window serves at most one leaf and every parent document at most MAX_PER_DOC windows.

Seed decontamination (#354, shared rules): every v1.1 test question is also run against every index, and a parent
document is blocked when a top hit contains the test answer together with most of the question's rare terms (it is
the source the item was written from, or cites the same fact). Windows with a canary string are dropped.

Output (OUT dir): passages.jsonl.zst (one row per selected passage), blocked.jsonl, pool_stats.json.
Usage: python -m m5.omniscience.pool.build OUT --leaves taxonomy/leaves.json [--web-shards 12] [--per-leaf 45]
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import bm25s
import numpy as np
import pyarrow.parquet as pq
import Stemmer
import zstandard

from m5.omniscience import fsio

WIKI = Path('/data/aai/mmlu-search/wiki_bm25_index')
FINEWEB = Path('/data/aai/nanoswe/base_data_fineweb')
SCOTUS = Path('/data/aai/caselaw/sc_data/caselaw_sc.jsonl')
SOURCES = Path('/data/aai/m5/omniscience/sources')
TEST_V11 = Path('/data/aai/aa_evals/data/omniscience/public-v1.1@1f9cbce/AA-Omniscience_dataset_public.csv')
SRC_MAP = {'wikipedia': 'wiki', 'web_edu': 'web', 'case_law': 'law', 'software_docs': 'sw', 'changelog': 'sw',
           'pubmed': 'pubmed', 'sec_filings': 'edgar'}
CANARY = re.compile(r'canary GUID|BENCHMARK DATA SHOULD NEVER APPEAR', re.I)
WIN_WORDS = 700
MAX_WIN_PER_DOC = 12
MAX_PER_DOC = 2
STEM = Stemmer.Stemmer('english')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_zst(path):
    with open(path, 'rb') as f:
        r = zstandard.ZstdDecompressor().stream_reader(f)
        buf = b''
        while True:
            b = r.read(1 << 24)
            if not b:
                break
            buf += b
            *lines, buf = buf.split(b'\n')
            for l in lines:
                if l.strip():
                    yield json.loads(l)
        if buf.strip():
            yield json.loads(buf)


def windows(text, words=WIN_WORDS):
    """Split on paragraph boundaries into ~`words`-word windows (long paragraphs are cut by words)."""
    paras = [p.strip() for p in re.split(r'\n\s*\n|\n(?=[A-Z#*\-=])', text) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        w = p.split()
        while len(w) > words * 1.5:
            if cur:
                out.append('\n\n'.join(cur)); cur, n = [], 0
            out.append(' '.join(w[:words])); w = w[words:]
        if n + len(w) > words and cur:
            out.append('\n\n'.join(cur)); cur, n = [], 0
        cur.append(' '.join(w) if len(w) != len(p.split()) else p); n += len(w)
    if cur:
        out.append('\n\n'.join(cur))
    return [o for o in out if len(o.split()) >= 120]


class Index:
    """In-memory BM25 over windows of one source type. Each window keeps (doc_id, title, header, text)."""

    def __init__(self, name):
        self.name, self.meta, self.texts = name, [], []

    def add_doc(self, doc_id, title, header, text, cap=MAX_WIN_PER_DOC, rng=None, header_in_text=False):
        ws = windows(text)
        if len(ws) > cap:
            ws = sorted((rng or random).sample(ws, cap), key=ws.index)
        for i, w in enumerate(ws):
            if CANARY.search(w):
                continue
            self.meta.append((doc_id, title, '' if (header_in_text and i == 0) else header, i))
            self.texts.append(w)

    def build(self, threads):
        log(self.name, 'indexing', len(self.texts))
        toks = bm25s.tokenize(self.texts, stopwords='en', stemmer=STEM, show_progress=False)
        self.bm = bm25s.BM25(method='lucene', k1=1.5, b=0.75)
        self.bm.index(toks, show_progress=False)
        self.vocab = toks.vocab
        del toks
        log(self.name, 'indexed')

    def search(self, queries, k, threads):
        q = bm25s.tokenize(queries, stopwords='en', stemmer=STEM, return_ids=False, show_progress=False)
        q = [[t for t in qq if t in self.vocab] for qq in q]
        ok = [i for i, qq in enumerate(q) if qq]
        res = np.full((len(queries), k), -1, dtype=np.int64)
        sc = np.zeros((len(queries), k), dtype=np.float32)
        if ok:
            r, s = self.bm.retrieve([q[i] for i in ok], k=min(k, len(self.texts)), n_threads=threads,
                                    show_progress=False)
            res[ok, :r.shape[1]], sc[ok, :r.shape[1]] = r, s
        return res, sc

    def passage(self, wid):
        doc_id, title, header, i = self.meta[wid]
        return {'doc_id': doc_id, 'title': title, 'header': header, 'window': i, 'text': self.texts[wid]}


class Wiki:
    """The mmlu-search Wikipedia BM25 index (bm25s, 21M 100-word chunks, read in place via mmap)."""
    name = 'wiki'

    def __init__(self):
        self.bm = bm25s.BM25.load(str(WIKI), mmap=True, load_corpus=False)
        self.offsets = json.loads((WIKI / 'corpus.mmindex.json').read_text())
        self.f = open(WIKI / 'corpus.jsonl', 'rb')

    def chunk(self, i):
        if i < 0 or i >= len(self.offsets) - 1:
            return None
        self.f.seek(self.offsets[i])
        return json.loads(self.f.readline())

    def search(self, queries, k, threads):
        q = bm25s.tokenize(queries, stopwords='en', stemmer=STEM, return_ids=False, show_progress=False)
        q = [[t for t in qq if t in self.bm.vocab_dict] or ['zzzzunk'] for qq in q]
        return self.bm.retrieve(q, k=k, n_threads=threads, show_progress=False)

    def passage(self, wid):
        c = self.chunk(wid)
        parts = [c['text']]
        lo = hi = wid
        for _ in range(3):
            p = self.chunk(lo - 1)
            if p and p['title'] == c['title']:
                parts.insert(0, p['text']); lo -= 1
            n = self.chunk(hi + 1)
            if n and n['title'] == c['title']:
                parts.append(n['text']); hi += 1
        text = ' '.join(parts)
        if text.startswith(c['title'] + ' '):
            text = text[len(c['title']) + 1:]
        return {'doc_id': f"wiki:{c['title']}", 'title': c['title'], 'header': '', 'window': lo, 'text': text,
                'span': [lo, hi]}


def load_sources(args, rng):
    idx = {k: Index(k) for k in ('web', 'law', 'sw', 'pubmed', 'edgar')}
    shards = sorted(glob.glob(str(FINEWEB / 'shard_*.parquet')))
    for s in rng.sample(shards, args.web_shards):
        t = pq.read_table(s, columns=['text']).column('text').to_pylist()
        for j, text in enumerate(t):
            if len(text) < 2500:
                continue
            first = text.strip().split('\n', 1)[0][:150]
            idx['web'].add_doc(f'web:{Path(s).stem}:{j}', first, '', text, cap=3, rng=rng)
        log('web', s, len(idx['web'].texts))
    with open(SCOTUS) as f:
        for nl, line in enumerate(f):
            if args.max_docs and nl >= args.max_docs:
                break
            d = json.loads(line)['caselaw']
            ops = ((d.get('casebody') or {}).get('data') or {}).get('opinions') or []
            text = '\n\n'.join(o.get('text', '') for o in ops)
            if len(text.split()) < 500:
                continue
            cite = ', '.join(c['cite'] for c in d.get('citations', [])[:2])
            header = f"{d['name']}, {cite} (Supreme Court of the United States, decided {d['decision_date']})"
            idx['law'].add_doc(f"scotus:{d['id']}", d['name_abbreviation'], header, header + '\n\n' + text, rng=rng,
                               header_in_text=True)
    log('law scotus', len(idx['law'].texts))
    for path, key in (('cap.jsonl.zst', 'law'), ('docs.jsonl.zst', 'sw'), ('cstd.jsonl.zst', 'sw'),
                      ('changelogs.jsonl.zst', 'sw'), ('pmc.jsonl.zst', 'pubmed'), ('edgar.jsonl.zst', 'edgar')):
        p = SOURCES / path
        if not p.exists():
            log('missing source', p)
            continue
        for nd, d in enumerate(read_zst(p)):
            if args.max_docs and nd >= args.max_docs:
                break
            if key == 'edgar' and rng.random() > args.edgar_keep:
                continue
            text = d['text']
            if key == 'law':
                header = text.strip()[:700]
                cap = 6
            elif key == 'sw':
                header = f"{d.get('repo', '')} {d.get('path', '')}".strip()
                cap = 40 if d['source'] == 'software_docs' else 12
            elif key == 'pubmed':
                header = d['title']
                cap = 4
            else:
                header = d['title']
                cap = 3
            idx[key].add_doc(d['doc_id'], d.get('title', '')[:200], header, text, cap=cap, rng=rng,
                             header_in_text=key == 'law')
        log(key, path, len(idx[key].texts))
    return idx


def rare_terms(q, df, n=8):
    toks = bm25s.tokenize([q], stopwords='en', stemmer=STEM, return_ids=False, show_progress=False)[0]
    toks = sorted(set(toks), key=lambda t: df.get(t, 0))
    return toks[:n]


def norm(s):
    return re.sub(r'[^0-9a-z]+', ' ', s.lower()).strip()


def seed_block(indexes, threads):
    """Parent documents that test items were written from (or that state the same fact)."""
    items = list(csv.DictReader(open(TEST_V11, encoding='utf-8')))
    qs = [it['question'] for it in items]
    blocked = {}
    for name, ix in indexes.items():
        res, sc = ix.search(qs, 20, threads)
        for it, row in zip(items, res):
            ans = norm(it['answer'])
            qt = bm25s.tokenize([it['question']], stopwords='en', stemmer=STEM, return_ids=False,
                                show_progress=False)[0]
            for wid in row:
                if wid < 0:
                    continue
                p = ix.passage(int(wid))
                body = p['header'] + ' ' + p['text']
                pt = set(bm25s.tokenize([body], stopwords='en', stemmer=STEM, return_ids=False,
                                        show_progress=False)[0])
                cover = sum(t in pt for t in set(qt)) / max(1, len(set(qt)))
                has_ans = len(ans) >= 2 and f' {ans} ' in f' {norm(body)} '
                # short answers ("No", "15", "C99") match by chance: they need more of the question's terms
                if (has_ans and (cover >= 0.7 or (len(ans) >= 4 and cover >= 0.5))) or cover >= 0.8:
                    blocked[p['doc_id']] = {'item_id': it['question_id'], 'index': name, 'cover': round(cover, 2),
                                            'has_answer': has_ans}
    return blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--leaves', required=True)
    ap.add_argument('--web-shards', type=int, default=12)
    ap.add_argument('--edgar-keep', type=float, default=1.0)
    ap.add_argument('--max-docs', type=int, default=0, help='smoke tests: cap docs read per source file')
    ap.add_argument('--per-leaf', type=int, default=45)
    ap.add_argument('--source-share', default='', help='per-leaf interleave shares, e.g. wiki=0.4')
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=354)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    leaves = json.loads(Path(args.leaves).read_text())['leaves']
    idx = load_sources(args, rng)
    for k in list(idx):
        if not idx[k].texts:
            del idx[k]
            continue
        idx[k].build(args.threads)
    idx['wiki'] = Wiki()
    log('indexes', {k: (len(v.texts) if hasattr(v, 'texts') else 'wiki') for k, v in idx.items()})

    blocked = seed_block(idx, args.threads)
    log('blocked docs', len(blocked))
    fsio.write_atomic(out / 'blocked.jsonl', ''.join(json.dumps({'doc_id': d, **why}) + '\n' for d, why in blocked.items()))

    # candidates per leaf: (normalized score, index name, window id)
    cands = collections.defaultdict(list)
    for name, ix in idx.items():
        qs, owners = [], []
        for L in leaves:
            srcs = {SRC_MAP.get(s) for s in L.get('sources', [])} - {None}
            if not srcs & set(idx):
                srcs = {'wiki', 'web'}
            if name in srcs:
                for q in L['queries'][:4]:
                    qs.append(q); owners.append(L['leaf_id'])
        if not qs:
            continue
        log(name, 'queries', len(qs))
        for s0 in range(0, len(qs), 4000):
            res, sc = ix.search(qs[s0:s0 + 4000], args.k, args.threads)
            for o, r, s in zip(owners[s0:s0 + 4000], res, sc):
                top = float(s[0]) or 1.0
                for wid, v in zip(r, s):
                    if wid >= 0 and v > 0:
                        cands[o].append((float(v) / top, name, int(wid)))
    # interleave sources within a leaf, best first
    share = {k: float(v) for k, v in (x.split('=') for x in args.source_share.split(',') if x)}
    order = {}
    for lid, cs in cands.items():
        by = collections.defaultdict(list)
        for c in sorted(set(cs), reverse=True):
            by[c[1]].append(c)
        # weighted round-robin: a source with share w contributes w passages per round (wiki is easiest for models)
        names = list(by)
        rng.shuffle(names)
        merged, credit, pos_s = [], collections.Counter(), collections.Counter()
        while any(pos_s[n] < len(by[n]) for n in names):
            for n in names:
                if pos_s[n] >= len(by[n]):
                    continue
                credit[n] += share.get(n, 1.0)
                while credit[n] >= 1 and pos_s[n] < len(by[n]):
                    merged.append(by[n][pos_s[n]])
                    pos_s[n] += 1
                    credit[n] -= 1
        order[lid] = merged
    used_win, per_doc, picked = set(), collections.Counter(), collections.defaultdict(list)
    pos = collections.Counter()
    active = [L['leaf_id'] for L in leaves if order.get(L['leaf_id'])]
    by_id = {L['leaf_id']: L for L in leaves}
    cctx = zstandard.ZstdCompressor(level=10)
    tmp_local = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or os.environ.get('TMPDIR') or '/tmp') / 'passages.jsonl.zst'
    fout = open(tmp_local, 'wb')
    w = cctx.stream_writer(fout)
    n_out, src_count = 0, collections.Counter()
    while active:
        rng.shuffle(active)
        nxt = []
        for lid in active:
            lst = order[lid]
            while pos[lid] < len(lst):
                score, name, wid = lst[pos[lid]]
                pos[lid] += 1
                if (name, wid) in used_win:
                    continue
                p = idx[name].passage(wid)
                if p['doc_id'] in blocked or per_doc[p['doc_id']] >= MAX_PER_DOC:
                    continue
                if p['doc_id'] in picked[lid]:
                    continue
                if CANARY.search(p['text']) or len(p['text'].split()) < 120:
                    continue
                if name == 'wiki':
                    for c in range(p['span'][0], p['span'][1] + 1):
                        used_win.add((name, c))
                used_win.add((name, wid))
                per_doc[p['doc_id']] += 1
                picked[lid].append(p['doc_id'])
                L = by_id[lid]
                row = {'pid': f'{lid}:{len(picked[lid]) - 1}', 'leaf_id': lid, 'domain': L['domain'],
                       'subtopic': L['subtopic'], 'leaf': L['leaf'], 'source': name, 'score': round(score, 3), **p}
                row.pop('span', None)
                w.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
                n_out += 1
                src_count[(L['domain'], name)] += 1
                break
            if len(picked[lid]) < args.per_leaf and pos[lid] < len(order[lid]):
                nxt.append(lid)
        active = nxt
    w.close()
    fout.close()
    blob = tmp_local.read_bytes()
    n_check = sum(1 for _ in zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 34).splitlines())
    assert n_check == n_out, (n_check, n_out)
    fsio.write_atomic(out / 'passages.jsonl.zst', blob)
    fsio.verify_file(out / 'passages.jsonl.zst')
    per_leaf = [len(picked[L['leaf_id']]) for L in leaves]
    stats = {'n_passages': n_out, 'n_leaves': len(leaves), 'leaves_lt_20': sum(x < 20 for x in per_leaf),
             'leaves_zero': sum(x == 0 for x in per_leaf), 'per_leaf_mean': sum(per_leaf) / len(per_leaf),
             'by_domain_source': {f'{d}|{s}': n for (d, s), n in sorted(src_count.items())},
             'blocked_docs': len(blocked),
             'index_sizes': {k: len(v.texts) for k, v in idx.items() if hasattr(v, 'texts')}}
    fsio.write_atomic(out / 'pool_stats.json', json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == '__main__':
    main()

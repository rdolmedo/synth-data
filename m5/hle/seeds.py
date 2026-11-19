"""Seed excerpts for knowledge questions: a full-text index over the literature subsample, retrieval by leaf queries,
seed blocking (works HLE cites, canary), and extraction of the chosen excerpts.

    # one CPU job per source (after m5.hle.literature): chunk every document, build a contentless FTS5 index
    python -m m5.hle.seeds index --source peS2o
    # attach a seed to every knowledge spec (CPU job): retrieval, blocking, doc reuse cap, one decompression pass
    python -m m5.hle.seeds attach --specs specs.jsonl --out specs_seeded.jsonl [--blocklist blocked.json]

Index: /data/aai/m5/hle/index/<source>.sqlite with `fts(title, body)` (contentless: no stored text) and `chunks(id, part, line, start, end, doc_id, title, nchars)`. A chunk is a ~CHUNK_CHARS window of a document
cut at a paragraph boundary. Ranking: distinct leaf-query phrases a chunk contains, then BM25 (Retriever.search).
"""
import argparse
import io
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from .safeio import verify_jsonl, write_atomic

LIT = Path('/data/aai/m5/shared/literature')
INDEX = Path('/data/aai/m5/hle/index')
CHUNK_CHARS = 20000
MIN_CHUNK = 3000
MAX_CHUNKS_PER_DOC = 100
NEAR_SPAN = 12
MIN_SCORE = 9.0   # one rare phrase (in <~1/8000 of chunks), or two moderately rare ones
SOURCE_NAMES = {'doab': 'open-access academic book (DOAB)', 'libretexts': 'LibreTexts textbook',
                'project_gutenberg': 'Project Gutenberg book', 'pre_1929_books': 'pre-1929 public-domain book',
                'library_of_congress': 'Library of Congress digitized book',
                'biodiversity_heritage_library': 'Biodiversity Heritage Library item', 'peS2o': 'open-access paper',
                'caselaw_access_project': 'US court opinion', 'arxiv': 'arXiv paper'}
STOP = set('a an and the of in on for to with by from at as is are was were be or not this that these those its it '
           'into over under between via about'.split())


def zst_lines(path):
    p = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    for line in io.TextIOWrapper(p.stdout, encoding='utf-8'):
        yield line
    p.wait()


def chunk_spans(text):
    spans, i, n = [], 0, len(text)
    while i < n and len(spans) < MAX_CHUNKS_PER_DOC:
        j = min(n, i + CHUNK_CHARS)
        if j < n:
            k = text.rfind('\n\n', i + CHUNK_CHARS // 2, j)
            if k < 0:
                k = text.rfind('\n', i + CHUNK_CHARS // 2, j)
            if k > 0:
                j = k
        if j - i >= MIN_CHUNK or not spans:
            spans.append((i, j))
        i = j
    return spans


def build_index(source, scratch):
    """Index one source. Builds in local scratch, then copies the db to /data."""
    parts = sorted((LIT / source).glob('part-*.jsonl.zst'))
    db_local = Path(scratch) / f'{source}.sqlite'
    db_local.unlink(missing_ok=True)
    con = sqlite3.connect(db_local)
    con.execute('pragma journal_mode=off')
    con.execute('pragma synchronous=off')
    con.execute("create virtual table fts using fts5(title, body, content='')")
    con.execute('create table chunks(id integer primary key, part text, line integer, start integer, end integer, '
                'doc_id text, title text, nchars integer)')
    cid = 0
    t0 = time.time()
    for part in parts:
        rows_c, rows_f = [], []
        for ln, line in enumerate(zst_lines(part)):
            d = json.loads(line)
            text = d['text']
            for (s, e) in chunk_spans(text):
                cid += 1
                rows_c.append((cid, part.name, ln, s, e, d['id'], (d.get('title') or '')[:300], e - s))
                rows_f.append((cid, d.get('title') or '', text[s:e]))
            if len(rows_f) >= 5000:
                con.executemany('insert into chunks values (?,?,?,?,?,?,?,?)', rows_c)
                con.executemany('insert into fts(rowid, title, body) values (?,?,?)', rows_f)
                rows_c, rows_f = [], []
        con.executemany('insert into chunks values (?,?,?,?,?,?,?,?)', rows_c)
        con.executemany('insert into fts(rowid, title, body) values (?,?,?)', rows_f)
        con.commit()
        print(f'{source}: {part.name} -> {cid} chunks ({time.time() - t0:.0f}s)', flush=True)
    con.execute("create virtual table vocab using fts5vocab(fts, 'row')")
    con.execute("insert into fts(fts) values('optimize')")
    con.commit()
    con.close()
    INDEX.mkdir(parents=True, exist_ok=True)
    tmp = INDEX / f'.{source}.sqlite.tmp'
    subprocess.run(['dd', f'if={db_local}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    if tmp.stat().st_size != db_local.stat().st_size:
        raise SystemExit('size mismatch')
    tmp.rename(INDEX / f'{source}.sqlite')
    db_local.unlink()
    print(f'{source}: DONE {cid} chunks', flush=True)


ARXIV = Path('/data/aai/m5/shared/arxiv')
# primary-archive sample sizes for the arXiv sub-index (knowledge seeds for Math, Physics, CS/AI, plus some bio/stat)
ARXIV_SAMPLE = {'math': 50000, 'cs': 40000, 'math-ph': 4000, 'hep-th': 6000, 'hep-ph': 6000, 'gr-qc': 4000,
                'quant-ph': 8000, 'cond-mat': 10000, 'astro-ph': 8000, 'physics': 6000, 'nucl-th': 2000,
                'stat': 5000, 'q-bio': 5000, 'eess': 3000, 'nlin': 2000, 'econ': 1500, 'q-fin': 1500}
ARXIV_MAX_CHUNKS = 3


def doc_path(source, part):
    if source == 'arxiv':
        return ARXIV / 'proof-pile-2-arxiv' / part
    return LIT / source / part


def build_index_arxiv(scratch, seed=0):
    """Index a stratified sample of proof-pile-2 arXiv papers (first ARXIV_MAX_CHUNKS chunks each), streaming the
    shared shards (never copied). part = '<split>/<shard>', line = the record's line in the decompressed shard."""
    import pyarrow.parquet as pq
    idx = pq.read_table(ARXIV / 'index.parquet', columns=['arxiv_id', 'split', 'shard', 'line', 'n_chars', 'title',
                                                            'primary_category', 'doi']).to_pylist()
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in idx:
        if r['n_chars'] and r['n_chars'] >= MIN_CHUNK:
            by[(r['primary_category'] or '').split('.')[0]].append(r)
    chosen = []
    for arch, n in ARXIV_SAMPLE.items():
        L = by.get(arch, [])
        chosen += rng.sample(L, min(n, len(L)))
    want = defaultdict(dict)
    for r in chosen:
        want[f'{r["split"]}/{r["shard"]}'][r['line']] = r
    print(f'arxiv: {len(chosen)} papers in {len(want)} shards', flush=True)
    db_local = Path(scratch) / 'arxiv.sqlite'
    db_local.unlink(missing_ok=True)
    con = sqlite3.connect(db_local)
    con.execute('pragma journal_mode=off')
    con.execute('pragma synchronous=off')
    con.execute("create virtual table fts using fts5(title, body, content='')")
    con.execute('create table chunks(id integer primary key, part text, line integer, start integer, end integer, '
                'doc_id text, title text, nchars integer)')
    cid, t0 = 0, time.time()
    for part, lines in sorted(want.items()):
        rows_c, rows_f = [], []
        for ln, line in enumerate(zst_lines(doc_path('arxiv', part))):
            r = lines.get(ln)
            if r is None:
                continue
            text = json.loads(line)['text']
            for (st, en) in chunk_spans(text)[:ARXIV_MAX_CHUNKS]:
                cid += 1
                rows_c.append((cid, part, ln, st, en, r['arxiv_id'], (r['title'] or '')[:300], en - st))
                rows_f.append((cid, r['title'] or '', text[st:en]))
        con.executemany('insert into chunks values (?,?,?,?,?,?,?,?)', rows_c)
        con.executemany('insert into fts(rowid, title, body) values (?,?,?)', rows_f)
        con.commit()
        print(f'arxiv: {part} -> {cid} chunks ({time.time() - t0:.0f}s)', flush=True)
    finish_index(con, db_local, 'arxiv', cid)


def finish_index(con, db_local, source, cid):
    con.execute("create virtual table vocab using fts5vocab(fts, 'row')")
    con.execute("insert into fts(fts) values('optimize')")
    con.commit()
    con.close()
    INDEX.mkdir(parents=True, exist_ok=True)
    tmp = INDEX / f'.{source}.sqlite.tmp'
    subprocess.run(['dd', f'if={db_local}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    if tmp.stat().st_size != db_local.stat().st_size:
        raise SystemExit('size mismatch')
    tmp.rename(INDEX / f'{source}.sqlite')
    db_local.unlink()
    print(f'{source}: DONE {cid} chunks', flush=True)


# ---------------------------------------------------------------------------------------------------------- retrieval

def terms(q):
    return [t for t in re.findall(r"[^\W_]+", q.lower()) if t not in STOP and len(t) > 1]


class Retriever:
    def __init__(self, sources=None, local_copy=None):
        self.cons, self.ndocs = {}, {}
        for db in sorted(INDEX.glob('*.sqlite')):
            name = db.stem
            if sources and name not in sources:
                continue
            path = db
            if local_copy:   # sqlite over Lustre is slow for random reads: copy to local scratch first
                path = Path(local_copy) / db.name
                if not path.exists() or path.stat().st_size != db.stat().st_size:
                    subprocess.run(['cp', str(db), str(path)], check=True)
            con = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
            self.cons[name] = con
            self.ndocs[name] = con.execute('select max(id) from chunks').fetchone()[0] or 1
        self.idf_cache = {}

    def idf(self, source, term):
        k = (source, term)
        if k not in self.idf_cache:
            r = self.cons[source].execute('select doc from vocab where term = ?', (term,)).fetchone()
            df = r[0] if r else 0
            self.idf_cache[k] = math.log((self.ndocs[source] + 1) / (df + 0.5))
        return self.idf_cache[k]

    def search(self, queries, sources=None, cap=5000, top=300):
        """-> [(score, source, chunk_id)] best first.

        Each query is a phrase (2+ real words). A phrase matching `cap` or more chunks of a source is too common and
        ignored; otherwise it is worth log(n_chunks / its chunk frequency). A chunk scores the sum over the phrases it
        contains, plus up to 2 points of BM25 (title x3) for the OR of the kept phrases, scaled within the source."""
        scores = defaultdict(float)
        for src in (sources or self.cons):
            if src not in self.cons:
                continue
            con = self.cons[src]
            # the phrase keeps every token (the FTS tokenizer indexes stopwords too, so dropping them breaks
            # adjacency); it needs 2+ content words of 3+ letters
            phrases = [' '.join(re.findall(r"[^\W_]+", q.lower())) for q in queries
                       if sum(1 for t in terms(q) if len(t) >= 3 and not t.isdigit()) >= 2]
            phrases = [p for p in dict.fromkeys(phrases) if p]
            kept = []
            for ph in phrases:
                # exact phrase, else its content words within NEAR_SPAN tokens of one another (0.8 weight); a chunk
                # gets the better of the two per query, each weighted by log(n_chunks / its chunk frequency)
                content = [t for t in ph.split() if t not in STOP and len(t) > 1]
                best = {}
                for expr, f in ((f'"{ph}"', 1.0),
                                ('NEAR(' + ' '.join(f'"{t}"' for t in content) + f', {NEAR_SPAN})', 0.8)):
                    if f < 1.0 and len(content) < 2:
                        continue
                    try:
                        rows = con.execute('select rowid from fts where fts match ? limit ?', (expr, cap)).fetchall()
                    except sqlite3.OperationalError:
                        continue
                    if not rows or len(rows) >= cap:
                        continue
                    w = f * math.log(self.ndocs[src] / len(rows))
                    for (rid,) in rows:
                        best[rid] = max(best.get(rid, 0.0), w)
                    kept.append(expr)
                for rid, w in best.items():
                    scores[(src, rid)] += w
            if not kept:
                continue
            try:
                bm = {rid: -b for rid, b in con.execute(
                    'select rowid, bm25(fts, 3.0, 1.0) from fts where fts match ? order by bm25(fts, 3.0, 1.0) limit ?',
                    (' OR '.join(kept), top))}
            except sqlite3.OperationalError:
                bm = {}
            mx = max(bm.values()) if bm else 0
            for rid, b in bm.items():
                if (src, rid) in scores and mx > 0:
                    scores[(src, rid)] += 2.0 * b / mx
        jit = random.Random(len(scores))
        return [x[1:] for x in sorted(((s + 1e-3 * jit.random(), s, src, rid) for (src, rid), s in scores.items()),
                                      reverse=True)]

    def chunk(self, source, cid):
        r = self.cons[source].execute('select part, line, start, end, doc_id, title, nchars from chunks where id = ?',
                                      (cid,)).fetchone()
        return dict(zip(('part', 'line', 'start', 'end', 'doc_id', 'title', 'nchars'), r))


# ---------------------------------------------------------------------------------------------------------- blocking

def norm_title(t):
    t = re.sub(r'[^a-z0-9 ]', ' ', (t or '').lower())
    return re.sub(r'\s+', ' ', t).strip()


class Blocklist:
    """Works cited in cais/hle rationales (and named in questions): arXiv ids, DOIs and titles."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
        self.arxiv = set(d.get('arxiv') or [])
        self.doi = {x.lower() for x in d.get('doi') or []}
        self.titles = {norm_title(t) for t in d.get('titles') or [] if len(norm_title(t)) >= 5}   # exact match
        self.long_titles = [t for t in self.titles if len(t.split()) >= 4]

    def blocked(self, doc_id, title, url=''):
        nt = norm_title(title)
        if nt and nt in self.titles:
            return 'title'
        if nt and len(nt.split()) >= 4 and any(t in nt for t in self.long_titles):
            return 'title_contains'
        s = f'{doc_id} {url}'.lower()
        if any(x in s for x in self.doi if len(x) > 8):
            return 'doi'
        if any(x in s for x in self.arxiv):
            return 'arxiv'
        return None


# ---------------------------------------------------------------------------------------------------------- attach

def attach(specs_path, out_path, blocklist_path, used_path, local_copy, seed=0, max_uses_book=3, top=15):
    """Attach a seed chunk to every knowledge spec that lacks one. Reasoning specs pass through unchanged."""
    rng = random.Random(seed)
    specs = [json.loads(l) for l in Path(specs_path).read_text().splitlines() if l.strip()]
    block = Blocklist(blocklist_path)
    used = json.loads(Path(used_path).read_text()) if used_path and Path(used_path).exists() else {}
    R = Retriever(local_copy=local_copy)
    all_sources = list(R.cons)
    picks, stats = {}, defaultdict(int)
    tax = None

    def pool_for(sp, relaxed):
        # relaxed (fallback once the leaf's strong matches are used up; production batches 1-6 lost 12-55% of
        # knowledge specs without it): every source at half the threshold, books up to twice the uses
        queries = list(sp.get('queries') or [])
        if sp.get('leaf'):
            queries.append(sp['leaf'])
        if not queries:   # open draw: the subject (and category) name
            queries = [sp['subject'] if sp['subject'] != 'broad' else sp['category'].split('/')[0]]
        pref = [s for s in (sp.get('sources') or []) if s in R.cons]
        if relaxed:
            cands = [c for c in R.search(queries, all_sources) if c[0] >= MIN_SCORE / 2]
        else:
            cands = [c for c in R.search(queries, pref or all_sources) if c[0] >= MIN_SCORE]
            if len(cands) < top and pref:   # other sources only on stronger evidence (two phrases)
                cands += [c for c in R.search(queries, [s for s in all_sources if s not in pref])
                          if c[0] >= 2 * MIN_SCORE]
        pool = []
        for score, src, cid in cands:
            c = R.chunk(src, cid)
            if c['nchars'] < MIN_CHUNK:
                continue
            key = f'{src}:{c["doc_id"]}'
            book = c['nchars'] >= CHUNK_CHARS // 2 and src not in ('peS2o', 'caselaw_access_project', 'libretexts')
            limit = (max_uses_book * (2 if relaxed else 1)) if book else 1
            if used.get(key, 0) >= limit or f'{src}:{cid}' in used:
                stats['reuse_skip'] += 1
                continue
            why = block.blocked(c['doc_id'], c['title'])
            if why:
                stats[f'blocked_{why}'] += 1
                continue
            pool.append((score, src, cid, c, key))
            if len(pool) >= top:
                break
        return pool

    for sp in specs:
        if sp['partition'] != 'knowledge' or sp.get('seed'):
            continue
        sp.pop('seed_missing', None)
        pool = pool_for(sp, relaxed=False) or pool_for(sp, relaxed=True)
        if not pool and sp.get('leaf'):
            # the leaf's documents are used up: try other leaves of the same subject node, which keeps the
            # partition x category x answer-type spread (production batch 0 lost 27% of knowledge specs here)
            if tax is None:
                from .gen import Taxonomy
                tax = Taxonomy()
            others = [l for l in tax.leaves.get(('knowledge', sp['category'], sp['subject'])) or []
                      if l['leaf'] != sp['leaf']]
            alts = [{**sp, 'leaf': l['leaf'], 'scope': l.get('scope'), 'queries': l.get('queries') or [],
                     'sources': l.get('sources') or []} for l in rng.sample(others, min(3, len(others)))]
            for relaxed in (False, True):
                for alt in alts:
                    pool = pool_for(alt, relaxed=relaxed)
                    if pool:
                        sp.update({k: alt[k] for k in ('leaf', 'scope', 'queries', 'sources')},
                                  leaf_resampled_from=sp['leaf'])
                        stats['leaf_resampled'] += 1
                        break
                if pool:
                    break
        if not pool:   # last resort: a leaf of another subject in the same category (keeps the category spread)
            if tax is None:
                from .gen import Taxonomy
                tax = Taxonomy()
            cat_leaves = [(subj, l) for (part, cat, subj), ls in tax.leaves.items()
                          if part == 'knowledge' and cat == sp['category'] for l in ls]
            for subj, l in rng.sample(cat_leaves, min(6, len(cat_leaves))):
                alt = {**sp, 'subject': subj, 'leaf': l['leaf'], 'scope': l.get('scope'),
                       'queries': l.get('queries') or [], 'sources': l.get('sources') or []}
                pool = pool_for(alt, relaxed=True)
                if pool:
                    sp.update({k: alt[k] for k in ('subject', 'leaf', 'scope', 'queries', 'sources')},
                              leaf_resampled_from=sp.get('leaf'), subject_resampled=True)
                    stats['category_resampled'] += 1
                    break
        if pool:
            # favour the best matches, keep some spread: pick from the top `top` with weights ~ score
            ws = [p[0] ** 2 for p in pool]   # favour chunks holding more of the leaf's phrases
            score, src, cid, c, key = rng.choices(pool, weights=ws)[0]
            used[key] = used.get(key, 0) + 1
            used[f'{src}:{cid}'] = 1
            picks[sp['id']] = (src, cid, c, score)
            stats['seeded'] += 1
        else:
            stats['no_seed'] += 1
    # one decompression pass per part to read the chosen chunks
    want = defaultdict(list)
    for sid, (src, cid, c, score) in picks.items():
        want[(src, c['part'])].append((c['line'], sid))
    texts = {}
    for (src, part), lst in want.items():
        lines = defaultdict(list)
        for ln, sid in lst:
            lines[ln].append(sid)
        for ln, line in enumerate(zst_lines(doc_path(src, part))):
            if ln in lines:
                d = json.loads(line)
                for sid in lines[ln]:
                    _, _, c, _ = picks[sid]
                    texts[sid] = (d, d['text'][c['start']:c['end']])
                del lines[ln]
                if not lines:
                    break
    rows = []
    for sp in specs:
        if sp['id'] in texts:
            src, cid, c, score = picks[sp['id']]
            d, text = texts[sp['id']]
            sp['seed'] = {'source': src, 'source_name': SOURCE_NAMES.get(src, src), 'chunk': cid,
                          'doc_id': d.get('id') or c['doc_id'], 'title': d.get('title') or c['title'] or '',
                          'author': d.get('author') or '',
                          'date': str(d.get('date') or (d.get('meta') or {}).get('timestamp') or '')[:20],
                          'url': d.get('url') or (d.get('meta') or {}).get('url') or '',
                          'score': round(score, 2), 'text': text}
        elif sp['partition'] == 'knowledge' and not sp.get('seed'):
            sp['seed_missing'] = True
        rows.append(json.dumps(sp, ensure_ascii=False) + '\n')
    write_atomic(out_path, ''.join(rows), check=lambda p: verify_jsonl(p, len(rows)))
    if used_path:
        write_atomic(used_path, json.dumps(used))
    print(json.dumps(dict(stats)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('index')
    i.add_argument('--source', required=True)
    i.add_argument('--scratch', default=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp')
    a_ = sub.add_parser('attach')
    a_.add_argument('--specs', required=True)
    a_.add_argument('--out', required=True)
    a_.add_argument('--blocklist', default='/data/aai/m5/hle/blocklist.json')
    a_.add_argument('--used', default=None, help='json of doc/chunk use counts, shared across batches')
    a_.add_argument('--local-copy', default=os.environ.get('_CONDOR_SCRATCH_DIR'))
    a_.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'index':
        for s in a.source.split(','):
            build_index_arxiv(a.scratch) if s == 'arxiv' else build_index(s, a.scratch)
    else:
        attach(a.specs, a.out, a.blocklist, a.used, a.local_copy, a.seed)


if __name__ == '__main__':
    main()

"""Build the joined index of proof-pile-2 `arxiv` with the arXiv metadata snapshot (CC0).

Usage: python arxiv_index.py SRC [DST]   (SRC holds proof-pile-2-arxiv/{train,validation,test}/*.jsonl.zst and metadata/*.parquet)

Writes OUT/index.parquet: one row per proof-pile-2 record:
  arxiv_id (normalized to the metadata form), split, shard (file name), line (0-based line in the decompressed shard),
  n_chars, pp2_timestamp, pp2_yymm, and the joined metadata: title, categories, primary_category, abstract, doi,
  journal_ref, license, first_version_date; matched (bool).
Also OUT/index_stats.json (counts per split, match rate, records per primary archive).
Records are read by streaming the shards; nothing is decompressed to disk.
"""
import io
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

OLD = re.compile(r'^([a-z\-]+(?:\.[A-Z]{2})?)/?(\d{7})(v\d+)?$')
NEW = re.compile(r'^(\d{4}\.\d{4,5})(v\d+)?$')


def norm_id(x: str) -> str:
    x = (x or '').strip()
    x = re.sub(r'^arxiv:', '', x, flags=re.I)
    m = NEW.match(x)
    if m:
        return m.group(1)
    m = OLD.match(x)
    if m:
        return f'{m.group(1).split(".")[0]}/{m.group(2)}'
    return x


def scan(path: str):
    p = Path(path)
    rows = []
    with open(p, 'rb') as f:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding='utf-8')
        for i, line in enumerate(reader):
            try:
                r = json.loads(line)
            except Exception:
                continue
            meta = r.get('meta') or {}
            rows.append({'raw_id': str(meta.get('arxiv_id') or meta.get('id') or ''),
                         'arxiv_id': norm_id(str(meta.get('arxiv_id') or '')),
                         'split': p.parent.name, 'shard': p.name, 'line': i, 'n_chars': len(r.get('text') or ''),
                         'pp2_timestamp': str(meta.get('timestamp') or ''), 'pp2_yymm': str(meta.get('yymm') or '')})
    return rows


def main(src: str, dst: str | None = None):
    out = Path(src)
    dst = Path(dst or src)
    shards = sorted(str(p) for p in (out / 'proof-pile-2-arxiv').glob('*/*.jsonl.zst'))
    with Pool(8) as pool:
        parts = pool.map(scan, shards, chunksize=1)
    rows = [r for part in parts for r in part]
    print('records', len(rows), flush=True)
    meta = pq.read_table(sorted(str(x) for x in (out / 'metadata').glob('*.parquet')), columns=['id', 'title', 'categories', 'abstract', 'doi', 'journal-ref',
                                                    'license', 'versions']).to_pylist()
    m = {}
    for r in meta:
        v = r.get('versions') or []
        m[norm_id(r['id'])] = {'title': r['title'], 'categories': r['categories'], 'abstract': r['abstract'],
                               'doi': r['doi'], 'journal_ref': r['journal-ref'], 'license': r['license'],
                               'first_version_date': (v[0].get('created') if v else None)}
    del meta
    empty = dict.fromkeys(['title', 'categories', 'abstract', 'doi', 'journal_ref', 'license', 'first_version_date'])
    st = Counter()
    for r in rows:
        j = m.get(r['arxiv_id'])
        r.update(j or empty)
        r['matched'] = j is not None
        r['primary_category'] = (j['categories'] or '').split()[0] if j and j['categories'] else None
        st[(r['split'], r['matched'])] += 1
    pq.write_table(pa.Table.from_pylist(rows), dst / 'index.parquet', compression='zstd')
    arch = Counter((r['primary_category'] or '?').split('.')[0] for r in rows)
    stats = {'records': len(rows), 'by_split_matched': {f'{a}/{b}': c for (a, b), c in st.items()},
             'match_rate': sum(r['matched'] for r in rows) / max(1, len(rows)),
             'by_primary_archive': dict(arch.most_common()), 'unmatched_examples': [r['raw_id'] for r in rows
                                                                                  if not r['matched']][:30],
             'metadata_records': len(m)}
    (dst / 'index_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != 'by_primary_archive'}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])

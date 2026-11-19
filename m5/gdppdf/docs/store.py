"""Read access to the shared document cache (/data/aai/m5/shared/docs).

index.jsonl: one line per document: doc_id, source, texts (the .jsonl.zst shard holding its LiteParse pages),
n_pages, chars, domain, doc_type, title, issuer, series, url, licence, sha256, ... (see docs/README.md).
"""
import collections
import json
from pathlib import Path

import zstandard

DOCS = Path('/data/aai/m5/shared/docs')


def read_zst_jsonl(path):
    with open(path, 'rb') as f:
        data = zstandard.ZstdDecompressor().stream_reader(f).read()
    return [json.loads(l) for l in data.decode().split('\n') if l.strip()]


class DocStore:
    def __init__(self, index=DOCS / 'index.jsonl', cache_shards=64):
        self.index_path = Path(index)
        self.meta = {}
        self.reload()
        self.cache = collections.OrderedDict()
        self.cache_shards = cache_shards

    def reload(self):
        for l in open(self.index_path):
            if l.strip():
                r = json.loads(l)
                self.meta[r['doc_id']] = r

    def get(self, doc_id):
        if doc_id not in self.meta:
            self.reload()
        m = self.meta[doc_id]
        shard = m['texts']
        if shard not in self.cache:
            self.cache[shard] = {r['doc_id']: r for r in read_zst_jsonl(shard) if 'pages' in r}
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return {**m, **self.cache[shard][doc_id]}

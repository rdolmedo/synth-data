"""SafeDocs provenance for the CC PDF lane: file id -> source URL, host, TLD, country (CC-MAIN-2021-31-PDF-UNTRUNCATED
metadata, digitalcorpora S3; ~0.35 MB/s per stream through the proxy, so ranged parallel downloads).

Writes /data/aai/m5/shared/docs/cc2021/meta/provenance.parquet (id, url, host, tld, country, fetched_digest,
fetched_length; the first URL per file) and sources.json (file URLs, sizes, sha256). Condor job: 4 CPUs, 24 GB.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from m5.gdppdf.docs.pdfa import download_url

BASE = 'https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED'
FILES = {'prov': 'metadata/cc-provenance-20230303.csv.gz', 'hosts': 'metadata/cc-hosts-20230303.csv.gz'}
OUT = Path('/data/aai/m5/shared/docs/cc2021/meta')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main():
    tmp = Path(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp')) / 'ccmeta'
    tmp.mkdir(parents=True, exist_ok=True)
    src = {}
    for k, rel in FILES.items():
        p = tmp / Path(rel).name
        download_url(f'{BASE}/{rel}', p, parts=32)
        src[k] = {'url': f'{BASE}/{rel}', 'bytes': p.stat().st_size, 'sha256': sha256(p)}
    prov = pd.read_csv(tmp / Path(FILES['prov']).name, usecols=['url_id', 'file_name', 'url', 'fetched_digest',
                                                                  'fetched_length'], dtype=str, keep_default_na=False)
    hosts = pd.read_csv(tmp / Path(FILES['hosts']).name, usecols=['url_id', 'host', 'tld', 'country'], dtype=str,
                        keep_default_na=False)
    df = prov.merge(hosts, on='url_id', how='left')
    df['uid'] = pd.to_numeric(df['url_id'], errors='coerce')
    df = df.sort_values('uid').drop_duplicates('file_name')
    df['id'] = df['file_name'].str.replace('.pdf', '', regex=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df[['id', 'url', 'host', 'tld', 'country', 'fetched_digest', 'fetched_length']].to_parquet(
        OUT / 'provenance.parquet', index=False)
    (OUT / 'sources.json').write_text(json.dumps({'corpus': BASE, 'files': src, 'rows': len(df),
                                                  'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=1))
    print('rows', len(df), flush=True)


if __name__ == '__main__':
    main()

"""Seed blocklist: the works cited in the rationales (and named in the questions) of all cais/hle and HLE-Diamond items.

    python -m m5.hle.blocklist --out /data/aai/m5/hle/private/blocklist.json [--server m5-ds]

Regex pass (arXiv ids, DOIs, URLs) plus an LLM pass on DeepSeek that lists every cited or named work (title, authors,
year, ids), 10 items per request. The output holds test-derived strings, so it lives under /data/aai/m5/hle/private
(mode 700), never in git or in shared posts.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from aa_evals.hle import data as hledata

from .llm import Cap, Server, fields

AX = re.compile(r'(?:arxiv\.org/(?:abs|pdf)/|arXiv:\s?)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})', re.I)
DOI = re.compile(r'\b(10\.\d{4,9}/[^\s"<>\])},;]+)', re.I)
URL = re.compile(r'https?://[^\s)\]>"]+')

PROMPT = """Below are {n} items, each a question with its author's rationale. For EACH item, list every specific published or
archival work that the text cites, quotes, names or relies on: papers, books, editions, chapters, articles, theses,
manuscripts, catalogues, datasets, databases, websites, standards, court cases, statutes, musical or literary works
when a specific edition or source is meant. Include works mentioned by title, by author and year, or by identifier.
Do not include general subject names or famous theorems unless a specific source is named.

Reply with only JSON: {{"items": [{{"id": "<item id>", "works": [{{"title": "...", "authors": "...", "year": "...",
"doi": "...", "arxiv": "...", "url": "..."}}]}}]}} (empty strings for unknown fields; "works": [] when none).

{items}"""


def load_items():
    rows = {}
    for ds in ('cais/hle', 'cais/hle-diamond'):
        import pyarrow.parquet as pq
        path = hledata.local_dir(ds) / 'test.parquet'
        t = pq.read_table(path, columns=['id', 'question', 'rationale'])
        for r in t.to_pylist():
            rows[r['id']] = r
    return list(rows.values())


async def llm_pass(items, server, out_dir, concurrency):
    srv = Server(server)
    cap = Cap(('ds', 'max_inflight', 'hle'), 40, override=concurrency)
    cache = out_dir / 'blocklist_llm.jsonl'
    done = set()
    if cache.exists():
        for l in cache.read_text().splitlines():
            for it in json.loads(l).get('items', []):
                done.add(it.get('id'))
    todo = [it for it in items if it['id'] not in done]
    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    lock = asyncio.Lock()

    async def one(b):
        txt = '\n\n'.join(f'### Item {it["id"]}\nQuestion: {it["question"][:4000]}\nRationale: {(it["rationale"] or "")[:6000]}'
                          for it in b)
        body = {'messages': [{'role': 'user', 'content': PROMPT.format(n=len(b), items=txt)}],
                'temperature': 0.6, 'chat_template_kwargs': {'thinking': False}}
        async with cap:
            for attempt in range(3):
                try:
                    data, _ = await srv.chat(b[0]['id'], body)
                    content = fields(data)[0]
                    m = re.search(r'\{.*\}', content, re.S)
                    obj = json.loads(m.group(0))
                    break
                except Exception as e:
                    print('batch error', e, flush=True)
                    obj = None
            if obj is None:
                return
        async with lock:
            with open(cache, 'a') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    await asyncio.gather(*(one(b) for b in batches))
    await srv.close()
    works = []
    for l in cache.read_text().splitlines():
        for it in json.loads(l).get('items', []):
            for w in it.get('works') or []:
                works.append({**w, 'item': it.get('id')})
    return works


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/aai/m5/hle/private/blocklist.json')
    ap.add_argument('--server', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=60)
    ap.add_argument('--no-llm', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(out.parent, 0o700)
    items = load_items()
    arxiv, doi, urls = set(), set(), set()
    for it in items:
        t = f'{it["question"]} {it["rationale"] or ""}'
        arxiv.update(x.lower() for x in AX.findall(t))
        doi.update(x.rstrip('.').lower() for x in DOI.findall(t))
        urls.update(URL.findall(t))
    titles = []
    works = [] if a.no_llm else asyncio.run(llm_pass(items, a.server, out.parent, a.concurrency))
    for w in works:
        if w.get('title'):
            titles.append(w['title'])
        if w.get('doi'):
            doi.add(w['doi'].lower().replace('https://doi.org/', ''))
        if w.get('arxiv'):
            arxiv.add(w['arxiv'].lower().replace('arxiv:', '').strip())
    res = {'n_items': len(items), 'arxiv': sorted(arxiv), 'doi': sorted(doi), 'urls': sorted(urls),
           'titles': sorted(set(titles)), 'works': works}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0) + '\n')
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in res.items()}), flush=True)


if __name__ == '__main__':
    main()

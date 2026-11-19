"""Generate GDP.pdf-style problems: DeepSeek V4.1 Flash (high reasoning effort, no output cap, 6 h timeout) reads one
real document's text and writes 2-3 requests for it; problem-side checks; accepted problems are appended as they
come, so the Super trace runner can stream them.

  python -m m5.gdppdf.gen.generate --docs <docs.jsonl> --out <dir> [--limit N] [--k 2|3|auto]

<docs.jsonl>: rows of the docs index to use (doc_id, domain, doc_type, title, issuer, n_pages, texts, ...).
Output <dir>/problems-<tag>.jsonl (one line per request, status accepted|rejected:<reason>) and
<dir>/gencalls-<tag>.jsonl (one line per generator call: usage, seconds, slots, raw reply for audits).
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

from m5.gdppdf.docs.store import DocStore
from m5.gdppdf.gen import checks, prompt as P
from m5.gdppdf.gen.slots import Sampler
from m5.gdppdf.llm import Server
from m5.gdppdf.util import Appender

EFFORT = 'high'


def ds_body(server, text):
    ov = json.loads(json.dumps((server.request or {}).get('overrides') or {}))
    body = {'temperature': ov.get('temperature', 1.0), 'top_p': ov.get('top_p', 0.95), 'reasoning_effort': EFFORT}
    ctk = dict(ov.get('chat_template_kwargs') or {})
    ctk.update({'thinking': True, 'reasoning_effort': EFFORT})
    body['chat_template_kwargs'] = ctk
    body['messages'] = [{'role': 'user', 'content': text}]
    return body   # no max_tokens: the context bounds the reply


def parse_reply(text):
    """-> (requests list or None if unparseable, reason when the generator declined the document)."""
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None, None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None, None
    reqs = j.get('requests') if isinstance(j, dict) else None
    return (reqs, j.get('reason')) if isinstance(reqs, list) else (None, None)


def chosen_leaf(spec, tf):
    """The leaf the generator picked from the slot's candidates ('a'-'d'), or an open draw."""
    t = str(tf or '').strip().lower().strip('()., ')
    if len(t) >= 1 and t[0] in 'abcdefghijklmnop' and (len(t) == 1 or not t[1].isalpha()):
        j = ord(t[0]) - 97
        if j < len(spec['leaf_candidates']):
            return spec['leaf_candidates'][j], False
    return 'open', True


def pdf_name(rng):
    return f'{uuid.UUID(int=rng.getrandbits(128), version=4)}_{uuid.UUID(int=rng.getrandbits(128), version=4)}_' \
           f'{uuid.UUID(int=rng.getrandbits(128), version=4)}.pdf'


def k_for(doc, mode):
    if mode != 'auto':
        return int(mode)
    return 3 if doc['n_pages'] >= 8 else 2


async def run(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f'{os.environ.get("_CONDOR_CLUSTER", "local")}-{int(time.time())}'
    docs = [json.loads(l) for l in open(a.docs) if l.strip()]
    done = set()
    for p in glob.glob(str(out_dir / 'gencalls-*.jsonl')):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                r = json.loads(l)
                if r.get('status') == 'ok':
                    done.add(r['doc_id'])
    docs = [d for d in docs if d['doc_id'] not in done]
    if a.limit:
        docs = docs[:a.limit]
    print(f'[gen] {len(docs)} documents to do ({len(done)} done)', flush=True)
    store = DocStore(a.index)
    sampler = Sampler(a.taxonomy) if a.taxonomy else Sampler()
    server = await Server(a.server, 'ds', cap=a.concurrency).start()
    fprob = Appender(out_dir / f'problems-{tag}.jsonl')
    fcall = Appender(out_dir / f'gencalls-{tag}.jsonl')
    stats = {'docs': 0, 'accepted': 0, 'rejected': 0, 'failed': 0, 'declined': 0}
    sem = asyncio.Semaphore(4 * (a.concurrency or 200))

    async def one(meta):
        async with sem:
            doc = store.get(meta['doc_id'])
            pages = [p['text'] for p in doc['pages']]
            k = k_for(doc, a.k)
            slots = sampler.slots(doc, k, seed=f'{a.seed}:{doc["doc_id"]}')
            text = P.build(doc, pages, slots)
            t0 = time.time()
            rec = {'doc_id': doc['doc_id'], 'slots': slots, 'k': k, 'at': None}
            try:
                res = await server.chat(ds_body(server, text), affinity=doc['doc_id'])
            except (RuntimeError, ValueError) as e:
                rec.update(status='failed', error=str(e)[:500])
                fcall.write(rec)
                stats['failed'] += 1
                return
            reqs, declined = parse_reply(res['content'])
            rec.update(status='ok' if reqs else 'declined' if reqs == [] else 'unparseable', declined=declined, usage=res['usage'], seconds=round(time.time() - t0),
                       finish_reason=res['finish_reason'], reply=res['content'][:20000],
                       reasoning_chars=len(res['reasoning']), at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            fcall.write(rec)
            if not reqs:
                stats['declined' if reqs == [] else 'failed'] += 1
                return
            doc_ng = checks.ngrams(checks.words(' '.join(pages)))
            rows = []
            rng = random.Random(f'pdf:{doc["doc_id"]}')
            for j, r in enumerate(reqs[:k]):
                if not isinstance(r, dict):
                    continue
                si = r.get('slot') if isinstance(r.get('slot'), int) and 1 <= r.get('slot') <= k else j + 1
                reason, flags = checks.check_request(r, len(pages), doc_ng)
                spec = dict(slots[si - 1])
                spec['leaf'], spec['open_draw'] = chosen_leaf(spec, r.get('task_family'))
                spec['family_fit'] = r.get('family_fit')
                rows.append({'problem_id': f'{doc["doc_id"]}-{si}', 'doc_id': doc['doc_id'], 'slot': si,
                             'prompt': (r.get('prompt') or '').strip(), 'task_family': r.get('task_family'),
                             'pdf_name': pdf_name(rng),
                             'domain': doc['domain'], 'doc_type': doc.get('doc_type'), 'n_pages': len(pages),
                             'spec': spec, 'flags': flags,
                             'status': 'accepted' if reason is None else f'rejected:{reason}'})
            ok = [i for i, x in enumerate(rows) if x['status'] == 'accepted']
            for i in checks.tfidf_dupes([rows[i]['prompt'] for i in ok]):
                rows[ok[i]]['status'] = 'rejected:within_doc_repeat'
            for x in rows:
                fprob.write(x)
                stats['accepted' if x['status'] == 'accepted' else 'rejected'] += 1
            stats['docs'] += 1
            if stats['docs'] % 20 == 0 or stats['docs'] <= 3:
                print(f'[gen] {stats} ds={server.stats}', flush=True)

    await asyncio.gather(*(one(d) for d in docs))
    fprob.close()
    fcall.close()
    await server.close()
    print(f'[gen] done {stats} ds={server.stats}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', required=True)
    ap.add_argument('--index', default='/data/aai/m5/shared/docs/index.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--taxonomy', default=None)
    ap.add_argument('--server', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=None, help='default: alloc.json cap (re-read every 60 s)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--k', default='auto')
    ap.add_argument('--seed', default='m5gdppdf-v1')
    ap.add_argument('--tag', default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()

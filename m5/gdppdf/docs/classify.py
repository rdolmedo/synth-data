"""Classify triaged documents into GDP.pdf's 10 domains and broad document types with DeepSeek (no thinking).

  python -m m5.gdppdf.docs.classify --triage '/data/aai/m5/shared/docs/cc2021/triage/*.jsonl.zst' \
      --out /data/aai/m5/shared/docs/cc2021/classified.jsonl

Light CPU (HTTP only); runs on gpu123 or as a small job. Resumable (skips file_names already in --out).
The same prompt labels the GDP.pdf test documents (class-level statistics only, --test).
"""
import argparse
import hashlib
import asyncio
import glob
import io
import json
import re
from pathlib import Path

import zstandard

from m5.gdppdf.llm import Server

DOMAINS = ['STEM/Research', 'Healthcare', 'Real Estate', 'Engineering', 'Manufacturing/Supply Chains', 'Construction',
           'Insurance', 'HR', 'Finance/Investing', 'Legal']
DOC_TYPES = {
    'research_article': 'peer-reviewed paper, review article, conference paper, thesis or dissertation chapter',
    'technical_report': 'technical, statistical or research report by an agency, lab, NGO or company; white paper',
    'guideline_standard': 'clinical/practice guideline, standard, specification, code, regulation text, design manual',
    'product_manual': 'operating/service/installation manual, product catalog, datasheet, technical selection guide',
    'handbook_policy': 'employee handbook, benefits guide, institutional policy, procedure manual, program guide',
    'contract_policy': 'insurance policy or coverage booklet, contract, agreement, collective bargaining agreement, '
                       'lease, bylaws, declaration, deed',
    'financial_report': 'annual report, 10-K/10-Q, proxy statement, financial statements, budget, fund/bank report',
    'legal_proceeding': 'court opinion, complaint, brief, order, attorney-general or administrative opinion, '
                        'proposed rules or amendments with commentary',
    'business_record': 'operational records: inspection/appraisal/assessment report, invoices, bills, logs, '
                       'schedules, price lists, claim files, minutes, bid tabulations, permit packages',
    'data_tables': 'mostly tabular statistical release, data appendix, rate or fee tables',
    'form_instructions': 'forms with their instructions, application packages, checklists',
}

PROMPT = """You are triaging PDF documents for a dataset of realistic professional document-analysis tasks. A knowledge \
worker (analyst, clinician, engineer, HR specialist, underwriter, attorney, project manager, ...) would receive such a \
document and be asked questions that require reading it closely.

Below are the source URL, page count and extracted text of the first pages (and one middle page) of a PDF. The text \
comes from a PDF parser without layout cleanup.

Decide:
- "usable": true if this is a substantive, information-dense document a professional in one of the domains below \
would plausibly analyze (rules, figures, tables, specifications, findings, terms). False for: marketing flyers, \
newsletters, event programs, slide decks with little text, menus, children's or religious material, fiction, \
personal documents, non-English documents, blank forms with no content, catalogs of unrelated short items, broken \
or garbled text, documents shorter in substance than ~3 pages.
- "domain": one of {domains}, or "none".
- "doc_type": one of:
{types}
- "issuer": the organization (or author group) that issued the document, as precisely as possible.
- "series": the document series or kind, e.g. "Form 10-K", "MMWR Recommendations and Reports", "employee handbook", \
"personal auto policy", "NASA Technical Note", journal name for papers.
- "year": publication year if visible, else null.
- "title": a short title.
- "richness": 1-3, how well the document supports multi-step questions (3 = many specific facts, tables, rules or \
numbers spread over many pages; 1 = thin or repetitive).
- "summary": one sentence on what the document contains.

URL: {url}
Pages: {pages}

--- first pages ---
{snippet}

--- middle page ({mid_page}) ---
{mid}

Answer with one JSON object only, keys: usable, domain, doc_type, issuer, series, year, title, richness, summary."""


def build(rec):
    types = '\n'.join(f'  - "{k}": {v}' for k, v in DOC_TYPES.items())
    return PROMPT.format(domains=', '.join(f'"{d}"' for d in DOMAINS), types=types, url=rec.get('url') or '(unknown)',
                         pages=rec.get('n_pages') or rec.get('pages_meta'), snippet=(rec.get('snippet') or '')[:4000],
                         mid_page=rec.get('mid_page') or '-', mid=(rec.get('mid_text') or '(none)')[:1000])


def parse_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def read_zst_jsonl(path):
    with open(path, 'rb') as f:
        data = zstandard.ZstdDecompressor().stream_reader(f).read()
    return [json.loads(l) for l in data.decode().split('\n') if l.strip()]


async def classify_all(recs, out_path, server, concurrency):
    done = set()
    if Path(out_path).exists():
        done = {json.loads(l)['file_name'] for l in open(out_path) if l.strip()}
    for r in recs:
        r.setdefault('file_name', r.get('id'))
        r.setdefault('pages_meta', r.get('n_pages'))
    todo = [r for r in recs if r['file_name'] not in done and not r.get('error') and (r.get('snippet') or '').strip()]
    print(f'[classify] {len(todo)} to do ({len(done)} done)', flush=True)
    out = open(out_path, 'a')
    sem = asyncio.Semaphore(concurrency)
    n = 0

    async def one(r):
        nonlocal n
        async with sem:
            body = {'messages': [{'role': 'user', 'content': build(r)}], 'temperature': 0.2, 'max_tokens': 600,
                    'chat_template_kwargs': ({'enable_thinking': False} if 'super' in server.name else
                                             {'thinking': False})}
            lab = None
            for _ in range(3):
                try:
                    res = await server.chat(body, affinity=r['file_name'])
                except (RuntimeError, ValueError) as e:
                    print(f'[classify] {r["file_name"]}: {e}', flush=True)
                    continue
                lab = parse_json(res['content'])
                if lab:
                    break
            if lab is None:
                return
            out.write(json.dumps({'file_name': r['file_name'], 'shard': r.get('shard'), 'model': server.served_name, 'sha256': r.get('sha256'),
                                  'url': r.get('url'), 'host': r.get('host'), 'pages': r.get('pages_meta'),
                                  'textless': r.get('textless'), 'label': lab}, ensure_ascii=False) + '\n')
            out.flush()
            n += 1
            if n % 500 == 0:
                print(f'[classify] {n}/{len(todo)} {server.stats}', flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    out.close()


def wanted(r):
    """Classification candidates: text-bearing PDFs, all of 20+ pages and a deterministic sample of shorter ones
    (GDP.pdf's median is 27 pages; the CC pool's is ~3)."""
    n, w = r.get('n_pages') or 0, r.get('words') or 0
    if r.get('error') or n < 1 or w < 300 or w / max(1, n) < 40:
        return False
    h = int(hashlib.sha1(str(r['id']).encode()).hexdigest()[:8], 16) / 16 ** 8
    share = 1.0 if n >= 20 else 0.3 if n >= 10 else 0.1 if n >= 5 else 0.03
    return h < share


THIN = re.compile(
    r'(policy period|declarations page|named insured|coverage part|endorsement|exclusions?\b.*\bcoverage|deductible|'
    r'underwriting|insurer|policyholder|claims? (handling|procedure|adjust)|premium rat|loss payee|subrogation|'
    r'certificate of insurance|summary of benefits|evidence of coverage|'
    r'bill of materials|purchase order|supplier|inventory|warehouse|logistics|lean manufacturing|quality manual|'
    r'packing list|production schedule|kanban|procurement|'
    r'employee handbook|collective bargaining|personnel polic|human resources|job description|pay scale|'
    r'salary schedule|leave of absence|benefits guide|open enrollment|'
    r'specifications? for|bid documents?|general conditions|scope of work|change order|submittals?|'
    r'appraisal|lease agreement|purchase agreement|zoning|property management|homeowners? association)', re.I)


def boosted(r):
    """Thin-domain keyword hits (insurance, supply chain, HR, construction, real estate), any length >= 3 pages."""
    n, w = r.get('n_pages') or 0, r.get('words') or 0
    if r.get('error') or n < 3 or w < 600 or w / max(1, n) < 40:
        return False
    return bool(THIN.search((r.get('snippet') or '') + ' ' + (r.get('mid_text') or '')))


async def amain(a):
    server = await Server(a.server, 'super' if 'super' in a.server else 'ds', cap=a.concurrency).start()
    try:
        if a.test:
            recs = [json.loads(l) for l in open(a.test)]
        else:
            recs = [r for p in sorted(glob.glob(a.triage)) for r in read_zst_jsonl(p)]
            recs = [r for r in recs if (boosted(r) if a.boost else wanted(r))]
        await classify_all(recs, a.out, server, a.concurrency)
    finally:
        await server.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--triage', default='/data/aai/m5/shared/docs/cc2021/triage/*.jsonl.zst')
    ap.add_argument('--test', default=None, help='jsonl of test-document snippets (same fields as triage rows)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--server', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--boost', action='store_true', help='thin-domain keyword candidates instead of the length sample')
    a = ap.parse_args()
    asyncio.run(amain(a))


if __name__ == '__main__':
    main()

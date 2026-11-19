"""Documents and long text: arXiv (PDF, LaTeX source), Europe PMC open-access articles (PDF via PMC's AWS open-data
bucket, else Europe PMC full-text XML), Wikipedia (HTML or plain text, pinned to a revision) and Project Gutenberg
(plain-text books, public domain in the US).
"""
import csv
import gzip
import html
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from ..http import TooLarge
from .base import MB, Reject, Source, cap, unpack

ARXIV_API = 'https://export.arxiv.org/api/query'
ARXIV_DEFAULT = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
ATOM = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}


def words(query):
    return [w for w in re.findall(r'[\w\-]+', query) if len(w) > 1][:6]


def strip_tags(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s or ''))).strip()


def cache_dir():
    d = Path(os.environ.get('TASKSYNTH_ASSET_CACHE') or Path(tempfile.gettempdir()) / 'tasksynth-asset-cache')
    d.mkdir(parents=True, exist_ok=True)
    return d


class Arxiv(Source):
    """export.arxiv.org search; PDFs, and the e-print source for LaTeX targets. The licence is read from the abstract
    page (arXiv's API carries none): CC links map to CC ids, the default to arXiv's non-exclusive licence (pointer)."""
    name = 'arxiv'
    classes = ('document',)

    async def arxiv_wait(self):
        # arXiv asks for one request per 3 s on all its hosts (export.arxiv.org is paced in http.PACE)
        await self.http.gate.wait('arxiv.org', 3.0)

    async def search(self, target, query, n):
        import xml.etree.ElementTree as ET
        ws = words(query)
        if not ws:
            return []
        params = {'search_query': ' AND '.join(f'all:{w}' for w in ws), 'max_results': min(50, n * 3),
                  'sortBy': random.choice(['relevance', 'relevance', 'submittedDate'])}
        text = await self.http.text(ARXIV_API, params=params)
        root = ET.fromstring(text)
        out = []
        for e in root.findall('a:entry', ATOM):
            vid = e.findtext('a:id', '', ATOM).rsplit('/abs/', 1)[-1]
            if not re.match(r'^[\w.\-/]+v\d+$', vid):
                continue
            authors = [a.findtext('a:name', '', ATOM) for a in e.findall('a:author', ATOM)]
            cat = e.find('arxiv:primary_category', ATOM)
            out.append({'source': self.name, 'key': vid, 'url': f'https://arxiv.org/pdf/{vid}',
                        'page_url': f'https://arxiv.org/abs/{vid}',
                        'title': re.sub(r'\s+', ' ', e.findtext('a:title', '', ATOM)).strip(),
                        'description': re.sub(r'\s+', ' ', e.findtext('a:summary', '', ATOM)).strip()[:400],
                        'license_raw': None, 'size': None, 'formats': list(target.get('formats') or ['pdf']),
                        'meta': {'authors': authors[:8], 'n_authors': len(authors),
                                 'category': cat.get('term') if cat is not None else None,
                                 'published': e.findtext('a:published', '', ATOM)[:10],
                                 'comment': (e.findtext('arxiv:comment', '', ATOM) or '')[:200]},
                        'query': query})
        return out

    async def resolve_license(self, cand):
        await self.arxiv_wait()
        page = await self.http.text(cand['page_url'])
        m = re.search(r'href="(https?://creativecommons\.org/[^"]+|https?://arxiv\.org/licenses/[^"]+)"', page)
        lic = m.group(1) if m else ARXIV_DEFAULT
        if 'arxiv.org/licenses/nonexclusive-distrib' in lic:
            lic = ARXIV_DEFAULT
        cand['license_raw'] = lic
        return cand

    async def fetch(self, cand, staging, target):
        staging = Path(staging)
        if cand.get('license_raw') is None:
            await self.resolve_license(cand)
        from ..license import classify
        if classify(cand['license_raw'])[1] == 'excluded':
            raise Reject(f'licence {cand["license_raw"]} is excluded by the policy')
        vid = cand['key']
        latex = 'latex' in (target.get('formats') or [])
        await self.arxiv_wait()
        if latex:
            tmp = staging.with_name(staging.name + '.eprint')
            try:
                digest, nbytes = await self.http.download(f'https://arxiv.org/e-print/{vid}', tmp, cap(target))
            except TooLarge as e:
                raise Reject(str(e))
            try:
                head = tmp.read_bytes()[:8]
                if head.startswith(b'%PDF'):
                    raise Reject('the e-print is a PDF only (no LaTeX source)')
                if tarfile.is_tarfile(tmp):
                    unpack(tmp, staging, strip_single_top=False)
                elif head[:2] == b'\x1f\x8b':
                    with gzip.open(tmp) as g, open(staging / 'main.tex', 'wb') as f:
                        data = g.read(cap(target) + 1)
                        if len(data) > cap(target):
                            raise Reject('decompressed source too large')
                        f.write(data)
                else:
                    raise Reject(f'unrecognized e-print format {head!r}')
            finally:
                tmp.unlink(missing_ok=True)
            if not any(p.suffix == '.tex' for p in staging.rglob('*')):
                raise Reject('no .tex file in the e-print source')
            url = f'https://arxiv.org/e-print/{vid}'
        else:
            safe = vid.replace('/', '_')
            try:
                digest, nbytes = await self.http.download(cand['url'], staging / f'{safe}.pdf', cap(target))
            except TooLarge as e:
                raise Reject(str(e))
            url = cand['url']
        authors = cand['meta']['authors']
        who = ', '.join(authors[:3]) + (' et al.' if cand['meta']['n_authors'] > 3 else '')
        return {'version': vid, 'sha256': digest, 'download_bytes': nbytes, 'url': url,
                'license_raw': cand['license_raw'],
                'attribution': f'{who}, "{cand["title"]}", arXiv:{vid}, {cand["license_raw"]}'}


EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
PMC_S3 = 'https://pmc-oa-opendata.s3.amazonaws.com'
EPMC_LICENSES = ['cc by', 'cc0', 'cc by-sa', 'cc by-nc', 'cc by-nc-sa']


class EuropePMC(Source):
    """Open-access articles with a CC licence. The PDF comes from PMC's open-data bucket on AWS (pinned by the PMC
    article version, md5 in the metadata); without a PDF there, Europe PMC's full-text XML."""
    name = 'europepmc'
    classes = ('document', 'text-data')

    async def search(self, target, query, n):
        lic = ' OR '.join(f'LICENSE:"{x}"' for x in EPMC_LICENSES)
        q = f'({query}) AND OPEN_ACCESS:y AND IN_EPMC:y AND ({lic})'
        params = {'query': q, 'format': 'json', 'resultType': 'core', 'pageSize': min(100, n * 3),
                  'sort': random.choice(['', '', 'P_PDATE_D desc'])}
        j = await self.http.json(f'{EPMC}/search', params={k: v for k, v in params.items() if v})
        out = []
        for r in j.get('resultList', {}).get('result', []):
            pmcid = r.get('pmcid')
            if not pmcid:
                continue
            out.append({'source': self.name, 'key': pmcid, 'url': f'https://europepmc.org/article/PMC/{pmcid}',
                        'page_url': f'https://europepmc.org/article/PMC/{pmcid}',
                        'title': strip_tags(r.get('title', ''))[:300],
                        'description': strip_tags(r.get('abstractText', ''))[:400],
                        'license_raw': r.get('license'), 'size': None,
                        'formats': list(target.get('formats') or ['pdf']),
                        'meta': {'authors': r.get('authorString', '')[:300], 'year': r.get('pubYear'),
                                 'journal': ((r.get('journalInfo') or {}).get('journal') or {}).get('title'),
                                 'doi': r.get('doi'), 'has_pdf': r.get('hasPDF')},
                        'query': query})
        return out

    async def fetch(self, cand, staging, target):
        staging = Path(staging)
        pmcid = cand['key']
        want_xml = any(f in (target.get('formats') or []) for f in ('xml', 'html')) or target['cls'] == 'text-data'
        meta = None
        for v in (1, 2, 3):
            try:
                meta = await self.http.json(f'{PMC_S3}/metadata/{pmcid}.{v}.json', retries=2)
                newer = await self._exists(f'{PMC_S3}/metadata/{pmcid}.{v + 1}.json')
                if not newer:
                    break
            except Exception:  # noqa: BLE001  (no such version)
                break
        if meta and not meta.get('is_pmc_openaccess', True):
            raise Reject('not in the PMC open-access subset')
        key = 'xml_url' if want_xml else 'pdf_url'
        s3 = (meta or {}).get(key) or (meta or {}).get('pdf_url' if want_xml else 'xml_url')
        if s3:
            path = s3.split('s3://pmc-oa-opendata/', 1)[1].split('?')[0]
            url = f'{PMC_S3}/{path}'
            dest = staging / Path(path).name
            version = f'{pmcid}.{meta.get("version")}'
            lic = meta.get('license_code') or cand['license_raw']
        else:
            url = f'{EPMC}/{pmcid}/fullTextXML'
            dest = staging / f'{pmcid}.xml'
            version = f'{pmcid} (Europe PMC full text, {time.strftime("%Y-%m-%d")})'
            lic = cand['license_raw']
        try:
            digest, nbytes = await self.http.download(url, dest, cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        authors = cand['meta'].get('authors') or 'unknown authors'
        return {'version': version, 'sha256': digest, 'download_bytes': nbytes, 'url': url, 'license_raw': lic,
                'attribution': f'{authors[:150]} "{cand["title"][:150]}", {cand["meta"].get("journal") or ""} '
                               f'{cand["meta"].get("year") or ""}, {pmcid}, {lic}'}

    async def _exists(self, url):
        try:
            r = await self.http.client.head(url)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False


WIKI_API = 'https://en.wikipedia.org/w/api.php'
WIKI_REST = 'https://en.wikipedia.org/api/rest_v1'
from .commons import WM_HEADERS  # noqa: E402


def html_to_text(doc):
    """Readable plain text from Parsoid HTML: paragraphs, headings and list items; no references, tables, infoboxes
    or maintenance boxes."""
    def inline(x):
        return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', x))).strip()
    doc = re.sub(r'(?is)<head\b.*?</head>', ' ', doc)
    doc = re.sub(r'(?is)<(script|style|table|sup|figure|math|noscript)\b.*?</\1>', ' ', doc)
    doc = re.sub(r'(?is)<div\b[^>]*class="[^"]*(?:navbox|reflist|hatnote|shortdescription|ambox)[^"]*"[^>]*>.*?</div>',
                 ' ', doc)
    doc = re.sub(r'(?is)<(h[1-6])\b[^>]*>(.*?)</\1>', lambda m: f'\n\n{inline(m.group(2))}\n', doc)
    doc = re.sub(r'(?is)<li\b[^>]*>(.*?)</li>', lambda m: f'\n- {inline(m.group(1))}', doc)
    doc = re.sub(r'(?is)<p\b[^>]*>(.*?)</p>', lambda m: f'\n\n{inline(m.group(1))}', doc)
    doc = re.sub(r'(?s)<[^>]+>', '', doc)
    lines = [re.sub(r'[ \t]+', ' ', html.unescape(l)).strip() for l in doc.split('\n')]
    lines = [l for l in lines if l != '-' and 'Learn how and when to remove this' not in l]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip() + '\n'


class Wikipedia(Source):
    """English Wikipedia articles pinned to a revision: Parsoid HTML (document html) or plain text derived from it
    (text-data plain-text). CC BY-SA 4.0."""
    name = 'wikipedia'
    classes = ('document', 'text-data')

    async def search(self, target, query, n):
        params = {'action': 'query', 'list': 'search', 'srsearch': query, 'srlimit': min(50, n * 3),
                  'srprop': 'size|wordcount|snippet|timestamp', 'format': 'json', 'formatversion': 2}
        j = await self.http.json(WIKI_API, params=params, headers=WM_HEADERS)
        out = []
        for r in (j.get('query') or {}).get('search', []):
            if r.get('wordcount', 0) < 400 or r['title'].startswith(('List of', 'Index of')):
                continue
            t = r['title']
            out.append({'source': self.name, 'key': t, 'url': f'https://en.wikipedia.org/wiki/{quote(t.replace(" ", "_"))}',
                        'page_url': f'https://en.wikipedia.org/wiki/{quote(t.replace(" ", "_"))}', 'title': t,
                        'description': strip_tags(r.get('snippet', ''))[:400], 'license_raw': 'CC-BY-SA-4.0',
                        'size': r.get('size'), 'formats': list(target.get('formats') or ['html']),
                        'meta': {'wordcount': r.get('wordcount'), 'pageid': r.get('pageid')}, 'query': query})
        return out

    async def fetch(self, cand, staging, target):
        staging = Path(staging)
        t = cand['key']
        j = await self.http.json(WIKI_API, params={'action': 'query', 'prop': 'revisions', 'rvprop': 'ids|timestamp',
                                                   'titles': t, 'format': 'json', 'formatversion': 2},
                                 headers=WM_HEADERS)
        page = j['query']['pages'][0]
        if page.get('missing'):
            raise Reject('page missing')
        rev = page['revisions'][0]
        revid, ts = rev['revid'], rev['timestamp']
        slug = quote(t.replace(' ', '_'), safe='')
        url = f'{WIKI_REST}/page/html/{slug}/{revid}'
        fname = re.sub(r'[^A-Za-z0-9._\-]+', '_', t)[:100]
        dest = staging / f'{fname}.html'
        try:
            digest, nbytes = await self.http.download(url, dest, cap(target), headers=WM_HEADERS)
        except TooLarge as e:
            raise Reject(str(e))
        fmts = target.get('formats') or []
        if target['cls'] == 'text-data' or ('plain-text' in fmts and 'html' not in fmts):
            text = html_to_text(dest.read_text(errors='replace'))
            dest.unlink()
            (staging / f'{fname}.txt').write_text(text)
        return {'version': f'revision {revid} ({ts})', 'sha256': digest, 'download_bytes': nbytes,
                'url': f'https://en.wikipedia.org/w/index.php?title={slug}&oldid={revid}',
                'attribution': f'"{t}", Wikipedia contributors, English Wikipedia revision {revid}, CC BY-SA 4.0'}


PG_CATALOG = 'https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv'


class Gutenberg(Source):
    """Project Gutenberg plain-text books (public domain in the US), searched in the catalog CSV by title, subject and
    bookshelf keywords. The texts keep Gutenberg's header and footer (the generator decides what to do with them)."""
    name = 'gutenberg'
    classes = ('text-data', 'document')
    _rows = None

    async def catalog(self):
        if Gutenberg._rows is None:
            path = cache_dir() / 'pg_catalog.csv'
            if not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400:
                tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
                await self.http.download(PG_CATALOG, tmp, 200 * MB)
                tmp.replace(path)
            with open(path, newline='', encoding='utf-8') as f:
                Gutenberg._rows = [r for r in csv.DictReader(f) if r.get('Type') == 'Text']
        return Gutenberg._rows

    async def search(self, target, query, n):
        rows = await self.catalog()
        ws = [w.lower() for w in words(query) if len(w) > 2]
        if not ws:
            return []
        scored = []
        for r in rows:
            hay = f'{r["Title"]} {r["Subjects"]} {r["Bookshelves"]}'.lower()
            s = sum(w in hay for w in ws)
            if s:
                scored.append((s + random.random() * 0.5 + (0.3 if r.get('Language') == 'en' else 0), r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, r in scored[: n * 3]:
            i = r['Text#']
            out.append({'source': self.name, 'key': i, 'url': f'https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt',
                        'page_url': f'https://www.gutenberg.org/ebooks/{i}', 'title': re.sub(r'\s+', ' ', r['Title'])[:300],
                        'description': f'{r["Authors"][:120]}; subjects: {r["Subjects"][:200]}; '
                                       f'shelves: {r["Bookshelves"][:120]}'[:400],
                        'license_raw': 'public-domain', 'size': None, 'formats': ['plain-text'],
                        'meta': {'authors': r['Authors'], 'issued': r['Issued'], 'language': r['Language']},
                        'query': query})
        return out

    async def fetch(self, cand, staging, target):
        i = cand['key']
        try:
            digest, nbytes = await self.http.download(cand['url'], Path(staging) / f'pg{i}.txt', cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        if nbytes < 5000:
            raise Reject(f'text has only {nbytes} bytes')
        authors = cand['meta']['authors'] or 'Anonymous'
        return {'version': f'ebook #{i} (issued {cand["meta"]["issued"]}, fetched {time.strftime("%Y-%m-%d")})',
                'sha256': digest, 'download_bytes': nbytes, 'url': cand['url'],
                'attribution': f'"{cand["title"][:150]}" by {authors[:120]}, Project Gutenberg eBook #{i}, '
                               f'public domain in the USA'}

"""Wikimedia Commons: photos, scans, diagrams, recordings and videos (media), plus PDFs (document). Licenses and
attribution come from the file's extmetadata (LicenseShortName / License / LicenseUrl, Artist, Credit). Each file is
pinned by its upload timestamp and SHA-1; the download computes the SHA-256. ppm targets fetch a PNG or JPEG (the
task's setup derives the PPM).
"""
import html
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..http import TooLarge
from ..license import normalize
from .base import MB, Reject, Source

API = 'https://commons.wikimedia.org/w/api.php'
# format -> search keywords (CirrusSearch filemime:/filetype:)
FILTERS = {'png': ['filemime:image/png'], 'jpeg': ['filemime:image/jpeg'], 'gif': ['filemime:image/gif'],
           'svg': ['filemime:image/svg+xml'], 'ppm': ['filemime:image/png', 'filemime:image/jpeg'],
           'tiff': ['filemime:image/tiff'], 'video': ['filemime:video/webm', 'filetype:video'],
           'audio': ['filetype:audio'], 'pdf': ['filemime:application/pdf'], 'djvu': ['filemime:image/vnd.djvu']}
CAP = {'video': 60 * MB, 'audio': 40 * MB, 'pdf': 40 * MB}
IMAGE_CAP = 20 * MB
# Wikimedia's User-Agent policy wants a contact URL: generic agents get HTTP 403 (http.UA has none)
WM_HEADERS = {'User-Agent': 'tasksynth-research/0.1 (academic research)'}
# icons, logos and heraldry are not the real-world material tasks want
JUNK = re.compile(r'\b(logo|logos|icon|icons|flag of|flags|coat of arms|emblem|seal of|wordmark|signature|'
                  r'favicon|symbol|pictogram|button|blank map)\b', re.I)
MIN_BYTES = {'video': 200_000, 'audio': 50_000, 'pdf': 30_000, 'svg': 3_000}


def strip_html(s, n=400):
    s = re.sub(r'<[^>]+>', ' ', s or '')
    return re.sub(r'\s+', ' ', html.unescape(s)).strip()[:n]


def kind_of(mime):
    if mime.startswith('video/'):
        return 'video'
    if mime.startswith('audio/') or mime in ('application/ogg',):
        return 'audio'
    if mime == 'application/pdf':
        return 'pdf'
    if mime == 'image/svg+xml':
        return 'svg'
    return 'image'


def license_from(em):
    for k in ('License', 'LicenseShortName', 'LicenseUrl'):
        v = (em.get(k) or {}).get('value')
        if v and normalize(strip_html(v, 200)):
            return strip_html(v, 200)
    v = (em.get('LicenseShortName') or {}).get('value')
    return strip_html(v, 200) if v else None


class Commons(Source):
    name = 'commons'
    classes = ('media', 'document')

    def filters(self, target):
        fmts = [f for f in (target.get('formats') or []) if f in FILTERS]
        if not fmts:
            fmts = ['pdf'] if target['cls'] == 'document' else ['png', 'jpeg']
        return fmts

    async def search(self, target, query, n):
        out = []
        for fmt in self.filters(target):
            for flt in FILTERS[fmt]:
                params = {'action': 'query', 'generator': 'search', 'gsrnamespace': 6,
                          'gsrsearch': f'{query} {flt}', 'gsrlimit': min(50, max(10, n * 3)),
                          'prop': 'imageinfo', 'iiprop': 'url|size|sha1|mime|extmetadata|timestamp',
                          'iiextmetadatafilter': 'LicenseShortName|License|LicenseUrl|Artist|Credit|'
                                                 'ImageDescription|Categories|Restrictions',
                          'format': 'json', 'formatversion': 2}
                j = await self.http.json(API, params=params, headers=WM_HEADERS)
                for p in (j.get('query') or {}).get('pages', []):
                    c = self.cand(p, fmt, query)
                    if c:
                        out.append(c)
                if len(out) >= n * 3:
                    break
            if len(out) >= n * 3:
                break
        # smaller files first among equals: bundles stay lean (videos prefer <= 20 MB)
        out.sort(key=lambda c: (c['size'] > 20 * MB, c['meta']['rank']))
        seen, uniq = set(), []
        for c in out:
            if c['key'] not in seen:
                seen.add(c['key'])
                uniq.append(c)
        return uniq[: n * 3]

    def cand(self, p, fmt, query):
        info = (p.get('imageinfo') or [None])[0]
        if not info:
            return None
        em = info.get('extmetadata') or {}
        title = p['title']
        cats = strip_html((em.get('Categories') or {}).get('value', ''), 300)
        kind = kind_of(info.get('mime', ''))
        size = info.get('size') or 0
        if JUNK.search(title) or JUNK.search(cats):
            return None
        if size < MIN_BYTES.get(kind, 20_000):
            return None
        if kind == 'image' and min(info.get('width') or 0, info.get('height') or 0) < 200:
            return None
        if size > (CAP.get(kind) or IMAGE_CAP):
            return None
        if (em.get('Restrictions') or {}).get('value'):
            restr = strip_html(em['Restrictions']['value'], 100)
        else:
            restr = ''
        url = info['url'].split('?')[0]
        desc = strip_html((em.get('ImageDescription') or {}).get('value', ''), 300)
        return {'source': self.name, 'key': title, 'url': url, 'page_url': info.get('descriptionurl') or
                f'https://commons.wikimedia.org/wiki/{title.replace(" ", "_")}',
                'title': title[len('File:'):] if title.startswith('File:') else title,
                'description': (desc + (f' [categories: {cats}]' if cats else ''))[:400],
                'license_raw': license_from(em), 'size': size, 'formats': [fmt],
                'meta': {'mime': info.get('mime'), 'kind': kind, 'width': info.get('width'),
                         'height': info.get('height'), 'duration': info.get('duration'), 'sha1': info.get('sha1'),
                         'timestamp': info.get('timestamp'), 'restrictions': restr,
                         'artist': strip_html((em.get('Artist') or {}).get('value', ''), 200),
                         'credit': strip_html((em.get('Credit') or {}).get('value', ''), 200),
                         'rank': p.get('index', 0)},
                'query': query}

    async def fetch(self, cand, staging, target):
        m = cand['meta']
        cap = CAP.get(m['kind'], IMAGE_CAP)
        name = unquote(Path(urlparse(cand['url']).path).name)
        name = re.sub(r'[^A-Za-z0-9._\-]+', '_', name)[-120:] or 'file'
        try:
            digest, nbytes = await self.http.download(cand['url'], Path(staging) / name, cap, headers=WM_HEADERS)
        except TooLarge as e:
            raise Reject(str(e))
        who = m.get('artist') or 'unknown author'
        credit = f'; {m["credit"]}' if m.get('credit') and m['credit'] not in who else ''
        return {'version': f'{m.get("timestamp")} sha1:{m.get("sha1")}', 'sha256': digest, 'download_bytes': nbytes,
                'url': cand['url'], 'page_url': cand['page_url'],
                'attribution': f'"{cand["title"]}" by {who}{credit} (Wikimedia Commons), {cand["license_raw"]}'}

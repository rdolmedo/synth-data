"""OpenML datasets (tabular). The full active-dataset list (~6.7k names with shape statistics) is fetched once and
cached in memory; a search scores names against the query's words, then reads the descriptions of the best matches
(one call each) to rank them and to get the licence. Fetch takes the parquet export; csv/tsv targets get it converted
with pyarrow (noted in the record), arff targets get the original ARFF.

Excluded: toy and benchmark classics (iris, titanic, mnist, adult, ...), stream/synthetic generators (Hyperplane,
RandomRBF, BNG(...), fri_c*, ...), and tables over 2M cells or 500 columns.
"""
import asyncio
import re
from pathlib import Path

from ..http import TooLarge
from .base import Reject, Source, cap

API = 'https://www.openml.org/api/v1/json'
FAMOUS = {'iris', 'titanic', 'mnist_784', 'mnist', 'adult', 'wine', 'wine-quality-red', 'wine-quality-white',
          'wine_quality', 'boston', 'diabetes', 'breast-cancer', 'breast-w', 'wdbc', 'credit-g', 'car', 'mushroom',
          'kr-vs-kp', 'spambase', 'letter', 'fashion-mnist', 'cifar_10', 'covertype', 'higgs', 'bank-marketing',
          'blood-transfusion-service-center', 'phoneme', 'pima-indians-diabetes', 'sonar', 'ionosphere', 'glass',
          'heart-statlog', 'vehicle', 'segment', 'soybean', 'zoo', 'tic-tac-toe', 'balance-scale', 'abalone',
          'house_prices', 'california_housing', 'cholesterol', 'penguins', 'diamonds', 'auto-mpg', 'cars'}
SYNTHETIC = re.compile(r'^(bng\(|fri_c|hyperplane|randomrbf|random_rbf|randomtree|sea[_(]|led(\d|_|\(|$)|agrawal|stagger|'
                       r'waveform|friedman|mv$|puma|2dplanes|synthetic|madelon|hill-valley|electricity_prices_ice|'
                       r'pol$|fried$|.*_seed_\d|.*\(\d+\)$)', re.I)
STOP = {'the', 'a', 'an', 'of', 'and', 'or', 'for', 'in', 'on', 'data', 'dataset', 'with', 'to', 'by'}


def words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return {w for w in re.split(r'[^a-z0-9]+', s.lower()) if len(w) > 2 and w not in STOP}


def quality(d, name):
    for q in d.get('quality', []):
        if q['name'] == name:
            try:
                return float(q['value'])
            except (TypeError, ValueError):
                return None
    return None


class OpenML(Source):
    name = 'openml'
    classes = ('tabular-data',)
    _list = None
    _lock = asyncio.Lock()

    async def datasets(self):
        async with OpenML._lock:
            if OpenML._list is None:
                j = await self.http.json(f'{API}/data/list/limit/10000/status/active')
                keep = []
                for d in j['data']['dataset']:
                    name = d['name']
                    rows, cols = quality(d, 'NumberOfInstances'), quality(d, 'NumberOfFeatures')
                    if name.lower() in FAMOUS or SYNTHETIC.match(name):
                        continue
                    if rows is None or cols is None or rows < 50 or cols > 500 or rows * cols > 2e6:
                        continue
                    keep.append({'did': d['did'], 'name': name, 'version': d['version'], 'rows': int(rows),
                                 'cols': int(cols), 'words': words(name)})
                # one version per name: the latest
                best = {}
                for d in keep:
                    if d['name'] not in best or d['version'] > best[d['name']]['version']:
                        best[d['name']] = d
                OpenML._list = list(best.values())
        return OpenML._list

    async def describe(self, did):
        return (await self.http.json(f'{API}/data/{did}'))['data_set_description']

    async def search(self, target, query, n):
        q = words(query)
        if not q:
            return []
        ds = await self.datasets()
        scored = sorted(((len(q & d['words']), -abs(len(d['words']) - len(q)), d) for d in ds
                         if q & d['words']), key=lambda x: (-x[0], -x[1]))
        top = [d for _, _, d in scored[: n * 3]]
        descs = await asyncio.gather(*(self.describe(d['did']) for d in top), return_exceptions=True)
        out = []
        for d, desc in zip(top, descs):
            if isinstance(desc, Exception):
                continue
            text = re.sub(r'\s+', ' ', desc.get('description') or '').strip()
            if re.search(r'\bsynthetic(ally)? generated\b|\bartificial(ly generated)? data\b', text, re.I):
                continue
            out.append({'source': self.name, 'key': str(d['did']),
                        'url': desc.get('parquet_url') or desc.get('url'),
                        'page_url': f'https://www.openml.org/search?type=data&id={d["did"]}',
                        'title': d['name'], 'description': text[:400], 'license_raw': desc.get('licence'),
                        'size': None, 'formats': ['parquet', 'arff'],
                        'meta': {'did': d['did'], 'version': desc.get('version'), 'rows': d['rows'], 'cols': d['cols'],
                                 'arff_url': desc.get('url'), 'parquet_url': desc.get('parquet_url'),
                                 'md5': desc.get('md5_checksum'), 'citation': (desc.get('citation') or '')[:200],
                                 'creator': desc.get('creator'), 'original_data_url': desc.get('original_data_url'),
                                 'score': len(q & d['words'])},
                        'query': query})
        out.sort(key=lambda c: -c['meta']['score'])
        return out[: n * 3]

    async def fetch(self, cand, staging, target):
        staging, limit = Path(staging), cap(target)
        m = cand['meta']
        fmts = target.get('formats') or ['csv']
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', cand['title']).strip('_')[:80] or f'openml_{m["did"]}'
        note = None
        try:
            if 'arff' in fmts and m.get('arff_url'):
                d, nb = await self.http.download(m['arff_url'], staging / f'{safe}.arff', limit)
            elif m.get('parquet_url'):
                pq = staging / f'{safe}.parquet'
                d, nb = await self.http.download(m['parquet_url'], pq, limit)
                if not set(fmts) & {'parquet', 'feather'}:
                    sep = '\t' if 'tsv' in fmts else ','
                    out = staging / f'{safe}.{"tsv" if sep == chr(9) else "csv"}'
                    await asyncio.to_thread(self.to_csv, pq, out, sep)
                    pq.unlink()
                    note = f'converted from the OpenML parquet export to {out.suffix[1:]}'
            elif m.get('arff_url'):
                d, nb = await self.http.download(m['arff_url'], staging / f'{safe}.arff', limit)
            else:
                raise Reject('no download URL')
        except TooLarge as e:
            raise Reject(str(e))
        who = m.get('creator') or 'OpenML uploader'
        if isinstance(who, list):
            who = ', '.join(who[:3])
        res = {'version': f'openml:{m["did"]} v{m.get("version")}', 'sha256': d, 'download_bytes': nb,
               'url': f'https://www.openml.org/search?type=data&id={m["did"]}',
               'attribution': f'{who}, "{cand["title"]}" (OpenML dataset {m["did"]}), {cand["license_raw"]}'}
        if note:
            res['derived'] = note
        return res

    @staticmethod
    def to_csv(pq, out, sep):
        import pyarrow.csv as pc
        import pyarrow.parquet as pp
        t = pp.read_table(pq)
        pc.write_csv(t, out, write_options=pc.WriteOptions(delimiter=sep))

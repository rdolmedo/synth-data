"""Scientific data: structures (RCSB PDB, COD), sequences (NCBI), small molecules (PubChem) and mass spectra
(MassBank). Each candidate is one coherent set (one structure, one gene family's records, one compound class,
one contributor's spectra), so a task can work across several related records.

Licenses: RCSB PDB and COD data are CC0; NCBI sequence records and PubChem are US-government databases without
use restrictions (NCBI notes that some submitters may claim rights in their records; the attribution says so);
MassBank records carry their own LICENSE line (mostly CC BY / CC BY-SA, some NC), read at fetch time.
"""
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

from ..http import TooLarge
from ..license import classify
from .base import Reject, Source, cap

UTC = lambda: time.strftime('%Y-%m-%d', time.gmtime())  # noqa: E731


def slug(s, n=40):
    return re.sub(r'[^a-z0-9]+', '-', str(s).lower()).strip('-')[:n] or 'data'


# ------------------------------------------------------------------------------------------------ RCSB PDB
RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_GQL = 'https://data.rcsb.org/graphql'


class Rcsb(Source):
    """Macromolecular structures; formats pdb (legacy PDB format, absent for very large entries) and cif (mmCIF)."""
    name = 'rcsb'
    classes = ('scientific-data',)

    async def search(self, target, query, n):
        start = random.choice([0, 0, 0, 25, 50, 100])
        body = {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
                    {'type': 'terminal', 'service': 'full_text', 'parameters': {'value': query}},
                    {'type': 'terminal', 'service': 'text', 'parameters': {
                        'attribute': 'rcsb_entry_info.deposited_atom_count', 'operator': 'less', 'value': 60000}}]},
                'return_type': 'entry', 'request_options': {'paginate': {'start': start, 'rows': max(10, n * 3)}}}
        r = await self.http.request('POST', RCSB_SEARCH, json=body)
        if r.status_code == 204 or not r.content:
            return []
        ids = [x['identifier'] for x in r.json().get('result_set', [])]
        if not ids and start:
            body['request_options']['paginate']['start'] = 0
            r = await self.http.request('POST', RCSB_SEARCH, json=body)
            ids = [x['identifier'] for x in (r.json().get('result_set', []) if r.content else [])]
        if not ids:
            return []
        gql = ('{entries(entry_ids: %s) {rcsb_id struct {title} rcsb_accession_info {revision_date} '
               'rcsb_entry_info {deposited_atom_count experimental_method resolution_combined} '
               'struct_keywords {pdbx_keywords}}}' % json.dumps(ids))
        meta = (await self.http.request('POST', RCSB_GQL, json={'query': gql})).json()['data']['entries']
        out = []
        for e in meta:
            info = e.get('rcsb_entry_info') or {}
            atoms = info.get('deposited_atom_count') or 0
            title = (e.get('struct') or {}).get('title') or ''
            res = info.get('resolution_combined') or []
            out.append({'source': self.name, 'key': e['rcsb_id'], 'url': f'https://files.rcsb.org/download/{e["rcsb_id"]}.cif',
                        'page_url': f'https://www.rcsb.org/structure/{e["rcsb_id"]}',
                        'title': f'PDB {e["rcsb_id"]}: {title}'[:200],
                        'description': (f'{info.get("experimental_method")}, {atoms} atoms'
                                        + (f', resolution {res[0]:.2f} A' if res else '')
                                        + f'; keywords: {(e.get("struct_keywords") or {}).get("pdbx_keywords")}')[:400],
                        'license_raw': 'CC0-1.0', 'size': atoms * 100, 'formats': ['cif', 'pdb'],
                        'meta': {'revision_date': (e.get('rcsb_accession_info') or {}).get('revision_date'),
                                 'atoms': atoms}, 'query': query})
        return out[: n * 3]

    async def fetch(self, cand, staging, target):
        pid = cand['key']
        fmts = target.get('formats') or ['cif']
        want = ['pdb', 'cif'] if 'pdb' in fmts and 'cif' not in fmts else ['cif', 'pdb'] if 'cif' in fmts else ['cif']
        got, last = None, None
        for ext in want:
            url = f'https://files.rcsb.org/download/{pid}.{ext}'
            try:
                digest, nbytes = await self.http.download(url, Path(staging) / f'{pid}.{ext}', cap(target))
                got = (url, digest, nbytes)
                break
            except TooLarge as e:
                raise Reject(str(e))
            except Exception as e:  # noqa: BLE001  (PDB format does not exist for large entries: 404)
                last = e
        if not got:
            raise Reject(f'no {want} file for {pid}: {last}')
        rev = (cand['meta'].get('revision_date') or '')[:10]
        return {'version': f'{pid} revision {rev}', 'sha256': got[1], 'download_bytes': got[2], 'url': got[0],
                'attribution': f'RCSB PDB entry {pid} ({cand["title"][len(pid) + 6:][:120]}), revision {rev}, CC0-1.0'}


# ------------------------------------------------------------------------------------------------ NCBI
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
PROTEIN_WORDS = re.compile(r'\b(protein|proteins|enzyme|enzymes|peptide|kinase|protease|antibod|receptor|domain)', re.I)


class Ncbi(Source):
    """Sequence records: FASTA (format fasta) or GenBank flat files (genbank) of 5-40 records matching one query
    (a gene family, a pathogen's segments, ...). Records longer than 2 Mb are excluded."""
    name = 'ncbi'
    classes = ('scientific-data',)

    def db_for(self, target, query):
        return 'protein' if PROTEIN_WORDS.search(query) and 'genbank' not in (target.get('formats') or []) \
            else 'nucleotide'

    async def search(self, target, query, n):
        db = self.db_for(target, query)
        term = f'({query}) AND ("200"[SLEN] : "2000000"[SLEN])'
        j = await self.http.json(f'{EUTILS}/esearch.fcgi', params={'db': db, 'term': term, 'retmax': 300,
                                                                     'retmode': 'json', 'sort': 'relevance'})
        ids = j.get('esearchresult', {}).get('idlist', [])
        if len(ids) < 3:
            return []
        groups, i = [], 0
        rng = random.Random(f'{query}{len(ids)}')
        while i < len(ids) and len(groups) < n * 2:
            k = rng.randint(5, 40)
            groups.append(ids[i:i + k])
            i += k
        firsts = [g[0] for g in groups]
        summ = await self.http.json(f'{EUTILS}/esummary.fcgi', params={'db': db, 'id': ','.join(firsts),
                                                                         'retmode': 'json'})
        res = summ.get('result', {})
        out = []
        for g in groups:
            s = res.get(g[0], {})
            slen = int(s.get('slen') or 0)
            out.append({'source': self.name, 'key': f'{db}:{",".join(g)}', 'url': f'{EUTILS}/efetch.fcgi',
                        'page_url': f'https://www.ncbi.nlm.nih.gov/{db}/{s.get("accessionversion", g[0])}',
                        'title': f'{len(g)} NCBI {db} records for "{query}", e.g. {s.get("title", "")}'[:200],
                        'description': f'first record {s.get("accessionversion")}: {s.get("title", "")} '
                                       f'({slen} {"aa" if db == "protein" else "bp"})'[:400],
                        'license_raw': 'public-domain', 'size': slen * len(g) * (3 if 'genbank' in (target.get('formats') or []) else 1),
                        'formats': ['fasta', 'genbank'], 'meta': {'db': db, 'ids': g}, 'query': query})
        return out

    async def fetch(self, cand, staging, target):
        db, ids = cand['meta']['db'], cand['meta']['ids']
        gb = 'genbank' in (target.get('formats') or []) and 'fasta' not in (target.get('formats') or []) \
            or ('genbank' in (target.get('formats') or []) and random.random() < 0.5)
        rettype, ext = ('gb', 'gb') if gb else ('fasta', 'fasta')
        dest = Path(staging) / f'{slug(cand["query"])}.{ext}'
        url = f'{EUTILS}/efetch.fcgi?db={db}&id={",".join(ids)}&rettype={rettype}&retmode=text'
        try:
            digest, nbytes = await self.http.download(url, dest, cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        text = dest.read_text(errors='replace')
        if gb:
            accs = re.findall(r'^VERSION\s+(\S+)', text, re.M)
        else:
            accs = re.findall(r'^>(\S+)', text, re.M)
        if len(accs) < 1 or nbytes < 200:
            raise Reject(f'efetch returned {nbytes} bytes, {len(accs)} records')
        return {'version': ' '.join(accs)[:2000], 'sha256': digest, 'download_bytes': nbytes, 'url': url,
                'formats': [ 'genbank' if gb else 'fasta'],
                'attribution': f'NCBI {db} records {", ".join(accs[:5])}{" ..." if len(accs) > 5 else ""} '
                               f'({len(accs)} records), retrieved {UTC()}. NCBI places no restrictions on these data; '
                               'some submitters may claim rights in their records.'}


# ------------------------------------------------------------------------------------------------ PubChem
PUG = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


class PubChem(Source):
    """A related set of 5-200 compounds (Entrez text search over PubChem Compound) as SDF, plus a CSV of their
    computed properties (format molecule)."""
    name = 'pubchem'
    classes = ('scientific-data',)

    async def pace(self):
        await self.http.gate.wait('pubchem-extra', 0.25)  # PubChem: at most 5 requests/s

    async def search(self, target, query, n):
        j = await self.http.json(f'{EUTILS}/esearch.fcgi', params={'db': 'pccompound', 'term': query, 'retmax': 400,
                                                                     'retmode': 'json', 'sort': 'relevance'})
        cids = j.get('esearchresult', {}).get('idlist', [])
        if len(cids) < 3:
            return []
        rng = random.Random(query)
        groups, i = [], 0
        while i < len(cids) and len(groups) < n * 2:
            k = rng.randint(5, 120)
            groups.append(cids[i:i + k])
            i += k
        if len(groups) > 1 and len(groups[-1]) < 5:
            groups[-2] += groups.pop()
        await self.pace()
        firsts = ','.join(g[0] for g in groups)
        try:
            props = (await self.http.json(f'{PUG}/compound/cid/{firsts}/property/Title,MolecularFormula/JSON'))
            titles = {str(p['CID']): p for p in props['PropertyTable']['Properties']}
        except Exception:  # noqa: BLE001
            titles = {}
        out = []
        for g in groups:
            t = titles.get(g[0], {})
            out.append({'source': self.name, 'key': f'cids:{",".join(g)}', 'url': f'{PUG}/compound/cid/.../SDF',
                        'page_url': f'https://pubchem.ncbi.nlm.nih.gov/#query={quote(query)}',
                        'title': f'{len(g)} PubChem compounds for "{query}", e.g. {t.get("Title", g[0])}'[:200],
                        'description': f'first compound CID {g[0]} {t.get("Title", "")} {t.get("MolecularFormula", "")}'[:400],
                        'license_raw': 'public-domain', 'size': 4000 * len(g), 'formats': ['molecule'],
                        'meta': {'cids': g}, 'query': query})
        return out

    async def fetch(self, cand, staging, target):
        cids = cand['meta']['cids']
        staging = Path(staging)
        name = slug(cand['query'])
        await self.pace()
        url = f'{PUG}/compound/cid/{",".join(cids)}/SDF'
        try:
            digest, nbytes = await self.http.download(url, staging / f'{name}.sdf', cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        await self.pace()
        fields = 'Title,MolecularFormula,MolecularWeight,ExactMass,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,' \
                 'RotatableBondCount,HeavyAtomCount,Charge,IUPACName,InChIKey,ConnectivitySMILES'
        try:
            csv = await self.http.text(f'{PUG}/compound/cid/{",".join(cids)}/property/{fields}/CSV')
            (staging / f'{name}_properties.csv').write_text(csv)
        except Exception:  # noqa: BLE001  (the SDF alone is the asset)
            pass
        n_mol = (staging / f'{name}.sdf').read_text(errors='replace').count('$$$$')
        if n_mol < 1:
            raise Reject('SDF holds no molecules')
        return {'version': f'CIDs {",".join(cids)[:1500]} retrieved {UTC()}', 'sha256': digest,
                'download_bytes': nbytes, 'url': url,
                'attribution': f'PubChem Compound records ({n_mol} compounds, CIDs {",".join(cids[:5])}...), '
                               f'NCBI/NLM, retrieved {UTC()}; US government database, no use restrictions'}


# ------------------------------------------------------------------------------------------------ MassBank
MB_REPO = 'MassBank/MassBank-data'
LICENSE_RANK = ['cc0-1.0', 'public-domain', 'cc-by-4.0', 'cc-by-3.0', 'cc-by-sa-4.0', 'cc-by-sa-3.0', 'cc-by-nc-4.0',
                'cc-by-nc-3.0', 'cc-by-nc-sa-4.0', 'cc-by-nc-sa-3.0']


class MassBank(Source):
    """Mass spectra (format spectrum): 1-20 MassBank records from one contributor directory matching a query, pinned
    to a commit of MassBank/MassBank-data. The MassBank REST API refused connections in testing ("too many
    clients"), so search uses GitHub code search on the data repository."""
    name = 'massbank'
    classes = ('scientific-data',)

    def __init__(self, http):
        super().__init__(http)
        self.commit = None

    async def head(self):
        if not self.commit:
            j = await self.http.json(f'https://api.github.com/repos/{MB_REPO}/commits/dev')
            self.commit = j['sha']
        return self.commit

    async def search(self, target, query, n):
        j = await self.http.json('https://api.github.com/search/code', bucket='code_search',
                                 params={'q': f'{query} repo:{MB_REPO}', 'per_page': 100})
        by_dir = {}
        for it in j.get('items', []):
            p = it['path']
            if not p.endswith('.txt') or '/' not in p:
                continue
            by_dir.setdefault(p.split('/')[0], []).append(p)
        out = []
        for d, paths in by_dir.items():
            paths = sorted(paths)
            for i in range(0, len(paths), 20):
                g = paths[i:i + 20]
                out.append({'source': self.name, 'key': f'{d}:{g[0]}:{len(g)}', 'url': f'https://github.com/{MB_REPO}',
                            'page_url': f'https://github.com/{MB_REPO}/tree/dev/{d}',
                            'title': f'{len(g)} MassBank spectra from contributor {d} matching "{query}"',
                            'description': f'records {", ".join(Path(p).stem for p in g[:6])}', 'license_raw': None,
                            'size': 6000 * len(g), 'formats': ['spectrum'], 'meta': {'paths': g, 'dir': d},
                            'query': query})
        random.shuffle(out)
        return out[: n * 2]

    async def resolve_license(self, cand):
        sha = await self.head()
        text = await self.http.text(f'https://raw.githubusercontent.com/{MB_REPO}/{sha}/{cand["meta"]["paths"][0]}')
        m = re.search(r'^LICENSE:\s*(.+)$', text, re.M)
        cand['license_raw'] = m.group(1).strip() if m else None
        return cand

    async def fetch(self, cand, staging, target):
        sha = await self.head()
        staging = Path(staging)
        kept, lic, digests, total = [], [], [], 0
        for p in cand['meta']['paths']:
            url = f'https://raw.githubusercontent.com/{MB_REPO}/{sha}/{p}'
            dest = staging / Path(p).name
            try:
                digest, nbytes = await self.http.download(url, dest, cap(target))
            except TooLarge as e:
                raise Reject(str(e))
            text = dest.read_text(errors='replace')
            m = re.search(r'^LICENSE:\s*(.+)$', text, re.M)
            spdx, verdict = classify(m.group(1).strip() if m else None)
            if verdict == 'excluded':
                dest.unlink()
                continue
            kept.append(Path(p).stem)
            lic.append(spdx)
            digests.append(digest)
            total += nbytes
        if not kept:
            raise Reject('no record with an allowed license')
        worst = max(lic, key=lambda s: LICENSE_RANK.index(s) if s in LICENSE_RANK else len(LICENSE_RANK))
        import hashlib
        return {'version': f'{MB_REPO}@{sha}', 'sha256': hashlib.sha256(''.join(digests).encode()).hexdigest(),
                'download_bytes': total, 'url': f'https://github.com/{MB_REPO}/tree/{sha}/{cand["meta"]["dir"]}',
                'license_raw': worst,
                'attribution': f'MassBank records {", ".join(kept[:6])}{" ..." if len(kept) > 6 else ""} '
                               f'({len(kept)} spectra, contributor {cand["meta"]["dir"]}), {MB_REPO} commit {sha[:12]}; '
                               f'licenses {sorted(set(lic))}'}


# ------------------------------------------------------------------------------------------------ COD
COD = 'http://www.crystallography.net/cod'  # https timed out through the cluster proxy; http works


class Cod(Source):
    """Crystal structures from the Crystallography Open Database (format cif): 1-8 entries matching a query (one
    mineral family, one ligand series...), CC0."""
    name = 'cod'
    classes = ('scientific-data',)

    async def search(self, target, query, n):
        try:
            rows = await self.http.json(f'{COD}/result', params={'text': query, 'format': 'json'}, timeout=90)
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(rows, list) or not rows:
            return []
        rows = [r for r in rows if not r.get('duplicateof') and r.get('status') in (None, '', 'ok', 'OK')] or rows
        rng = random.Random(query)
        rng.shuffle(rows)
        out, i = [], 0
        while i < len(rows) and len(out) < n * 2:
            k = rng.choice([1, 1, 2, 3, 5, 8])
            g = rows[i:i + k]
            i += k
            name = g[0].get('mineral') or g[0].get('chemname') or g[0].get('commonname') or g[0].get('formula')
            out.append({'source': self.name, 'key': ','.join(r['file'] for r in g), 'url': f'{COD}/{g[0]["file"]}.cif',
                        'page_url': f'http://www.crystallography.net/cod/{g[0]["file"]}.html',
                        'title': f'{len(g)} COD structure(s) for "{query}", e.g. {name}'[:200],
                        'description': '; '.join(f'{r["file"]}: {r.get("formula")} {r.get("sg")} ({r.get("year")}) '
                                                 f'{(r.get("title") or "")[:80]}' for r in g)[:400],
                        'license_raw': 'CC0-1.0', 'size': 30000 * len(g), 'formats': ['cif'],
                        'meta': {'entries': [{'file': r['file'], 'rev': r.get('svnrevision'), 'doi': r.get('doi'),
                                              'authors': r.get('authors')} for r in g]}, 'query': query})
        return out

    async def fetch(self, cand, staging, target):
        import hashlib
        digests, total, refs = [], 0, []
        for e in cand['meta']['entries']:
            url = f'{COD}/{e["file"]}.cif'
            try:
                digest, nbytes = await self.http.download(url, Path(staging) / f'{e["file"]}.cif', cap(target))
            except TooLarge as err:
                raise Reject(str(err))
            if nbytes > 3 * (1 << 20) and len(cand['meta']['entries']) > 1:
                (Path(staging) / f'{e["file"]}.cif').unlink()  # structure-factor-laden CIFs bloat a set
                continue
            digests.append(digest)
            total += nbytes
            refs.append(f'COD {e["file"]} (svn r{e["rev"]}; {(e.get("authors") or "")[:60]}'
                        + (f'; doi:{e["doi"]}' if e.get('doi') else '') + ')')
        if not digests:
            raise Reject('every CIF of the set is larger than 3 MB')
        kept = {r.split(' ')[1] for r in refs}
        version = ' '.join(f'{e["file"]}@r{e["rev"]}' for e in cand['meta']['entries'] if e['file'] in kept)
        return {'version': version, 'sha256': digests[0] if len(digests) == 1 else
                hashlib.sha256(''.join(digests).encode()).hexdigest(), 'download_bytes': total,
                'url': f'{COD}/{cand["meta"]["entries"][0]["file"]}.cif',
                'attribution': 'Crystallography Open Database: ' + '; '.join(refs)[:800] + '; CC0-1.0'}

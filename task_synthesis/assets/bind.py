"""Bind specs to real assets: plan catalog queries per cell (class x formats x domain, no agent action), search,
shortlist per spec from catalog metadata, fetch into the store, and fit-check the fetched asset against the spec.
A spec whose candidates all fail stays unbound (in the A/B it keeps student 1's generated task).

  python -m task_synthesis.assets.bind submit --name b1 --specs <specs.jsonl> [--strategy targeted|untargeted]
  python -m task_synthesis.assets.bind run    --name b1 --specs <specs.jsonl>        # inside the job (resumable)
  python -m task_synthesis.assets.bind report --name b1
  python -m task_synthesis.assets.bind export --name b1 --round ra1 --prefix ra      # bound specs -> a round

State in <ROOT>/assets/binds/<name>/: specs.jsonl, plan.jsonl (queries per cell and catalog), pool.jsonl
(candidates), results.jsonl (one line per spec), fits.jsonl (every fit check), REPORT.md.
"""
import argparse
import asyncio
import importlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from ..common import ROOT, frozen_checkout
from . import decon, http as http_mod, manifest as manifest_mod, prompts as P, store
from .license import classify
from .sources.base import Reject
from .targets import cell, routes, target

BINDS = ROOT / 'assets' / 'binds'
PYTHON = '/data/aai/venvs/tasksynth-assets/bin/python'
ADAPTERS = {'github-repo': ('github', 'GithubRepos'), 'github-file': ('github', 'GithubFiles'),
            'zenodo': ('zenodo', 'Zenodo'), 'hf-dataset': ('hf', 'HfDatasets'), 'hf-model': ('hf', 'HfModels'),
            'openml': ('openml', 'OpenML'), 'commons': ('commons', 'Commons'), 'arxiv': ('docs', 'Arxiv'),
            'europepmc': ('docs', 'EuropePMC'), 'wikipedia': ('docs', 'Wikipedia'), 'gutenberg': ('docs', 'Gutenberg'),
            'rcsb': ('science', 'Rcsb'), 'ncbi': ('science', 'Ncbi'), 'pubchem': ('science', 'PubChem'),
            'massbank': ('science', 'MassBank'), 'cod': ('science', 'Cod'), 'debian': ('debian', 'DebianBinaries')}
MAX_USES = 2          # many assets, few tasks each (BLUEPRINT decision 5)


async def ramp_slots(slots, start, total, minutes, on_step=None):
    """Grow `slots` (a semaphore created with `start` permits) evenly to `total` permits over `minutes`. A bind job
    started into a loaded generator pool opens its slots gradually: m1-hero's replacement bind job (2026-11-04 19:56,
    #172 entry 11) started all 600 at once, on top of the new round shards' admissions."""
    extra = total - start
    if extra <= 0 or minutes <= 0:
        for _ in range(max(0, extra)):
            slots.release()
        if on_step:
            on_step(total)
        return
    step = minutes * 60 / extra
    for i in range(extra):
        await asyncio.sleep(step)
        slots.release()
        if on_step:
            on_step(start + i + 1)


def ramp_start(total, frac):
    return max(1, min(total, math.ceil(total * frac)))
SHOW = 16             # candidates shown to the shortlist


def load_sources(http):
    out = {}
    for name, (mod, cls) in ADAPTERS.items():
        try:
            m = importlib.import_module(f'.sources.{mod}', __package__)
            out[name] = getattr(m, cls)(http)
        except (ImportError, AttributeError) as e:
            print(f'source {name} unavailable: {e}', flush=True)
    return out


def jparse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object')
    return json.loads(m.group(0))


def spec_brief(spec):
    from ..pipeline.prompts import ACTIONS
    from ..pipeline.specs import CLASS_NAMES, SCALE, describe_asset
    lines = [f'- Domain: {spec["domain"].replace("-", " ")}',
             f'- What the agent mainly does: {ACTIONS.get(spec["agent_action"], spec["agent_action"])}',
             '- Materials (the first is the anchor, to be the real asset):']
    lines += [describe_asset(a) for a in spec['assets']]
    if spec.get('agent_works_on'):
        lines.append(f'- The agent mainly works on: {CLASS_NAMES.get(spec["agent_works_on"], spec["agent_works_on"])}')
    if spec.get('code_scale') in SCALE:
        lines.append(f'- Size of the code involved: {SCALE[spec["code_scale"]]}')
    lines.append(f'- Target expert time: about {spec.get("minutes", 60)} minutes')
    return '\n'.join(lines)


def cand_line(i, c):
    meta = c.get('meta') or {}
    bits = [f'[{i}] {c["source"]}: {c["title"]}', f'license {c.get("license_raw")}']
    if c.get('size'):
        bits.append(f'{manifest_mod.human(c["size"])}')
    for k in ('stars', 'language', 'topics', 'mime', 'files', 'downloads', 'library'):
        if meta.get(k):
            bits.append(f'{k} {meta[k]}')
    line = '; '.join(str(b) for b in bits)
    if c.get('description'):
        line += f'\n    {c["description"][:300]}'
    return line


class Binder:
    def __init__(self, a):
        self.a = a
        self.dir = BINDS / a.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = defaultdict(list)       # cell -> candidates
        self.pool_keys = set()
        self.planned = defaultdict(int)     # (cell, source) -> queries run
        self.uses = Counter()               # candidate key -> tasks bound (or reserved)
        self.failed = set()                 # candidate keys that failed fetch or license
        self.search_failed = Counter()      # cell -> searches that errored (rate limits): the cell may grow again
        self.cell_locks = defaultdict(asyncio.Lock)
        self.growing = {}                   # cell -> background task finishing its grow (--grow-now)
        self.urgent = set()                 # cells whose background grow a spec waits for: normal gate priority
        self.fetch_locks = defaultdict(asyncio.Lock)
        self.tokens = Counter()
        self.counts = Counter()

    def log(self, msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(self.dir / 'bind.log', 'a') as f:
            f.write(line + '\n')

    def append(self, name, obj):
        with open(self.dir / name, 'a') as f:
            f.write(json.dumps(obj) + '\n')

    async def llm(self, system, user, kind):
        content, _, usage = await self.ep.chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
        self.tokens[f'{kind}_prompt'] += usage.get('prompt') or 0
        self.tokens[f'{kind}_completion'] += usage.get('completion') or 0
        return content

    # ---------------------------------------------------------------- pools
    async def plan_queries(self, t, source, k):
        detail = t['language'] or ', '.join(t['formats']) or ', '.join(t['services'])
        targeted = self.a.strategy == 'targeted'
        user = P.PLAN_USER.format(cls_text=P.CLASS_TEXT.get(t['cls'], t['cls']),
                                  detail=f' ({detail})' if detail else '',
                                  domain=t['domain'].replace('-', ' ') if targeted else 'any (no domain target)',
                                  source_text=P.SOURCE_TEXT.get(source, source), k=k,
                                  domain_clause=' within this domain' if targeted else ' across many unrelated subjects')
        for _ in range(3):
            try:
                qs = [str(q).strip() for q in jparse(await self.llm(P.PLAN_SYSTEM, user, 'plan'))['queries']]
                return [q for q in qs if q][:k]
            except (ValueError, KeyError, TypeError):
                continue
        return []

    async def grow(self, t, c, n_specs, round_no):
        """Add candidates to a cell's pool from its routed catalogs. GitHub code search (github-file) is a fallback,
        queried only when the other catalogs left the cell short of 3 candidates (m1-pilot, 2026-11-04: at ~10
        queries/min with 429 holds, cells routed to it first held bind slots and binding fell to ~9 specs/min; in bf1 it
        was also the weakest source, 63% fit vs 86-89%); a cell with no other catalog still uses it.

        --grow-now N (m1-bind, 2026-11-04): k is sized for the cell's specs in the whole bind (up to 24 queries), but the
        spec that triggers the grow needs only a few. Each catalog runs its first N queries now; the rest run in a
        background task at low priority on the rate gates (http.LOW), so the cell ends with the same queries and pool
        while the GitHub search gate (25/min) serves the specs waiting now first. A cell left with fewer than 3 usable
        candidates after the first N runs every query now, as before (the github-file fallback decision is unchanged)."""
        stat = {'added': 0, 'failed': 0}
        srcs = routes(t, self.sources)
        primary = [x for x in srcs if x != 'github-file'] or srcs
        ks = [(min(24, math.ceil(0.5 * n_specs) + 2) if i == 0 else min(8, math.ceil(0.25 * n_specs) + 1))
              for i in range(len(primary))]
        ks = [k if round_no == 0 else max(2, k // 2) for k in ks]
        # the catalogs' plans do not depend on each other's results: ask for them at once (one LLM wait, not one per catalog)
        plans = await asyncio.gather(*(self.plan_queries(t, s, k) for s, k in zip(primary, ks)))
        later = []
        for source, queries in zip(primary, plans):
            self.append('plan.jsonl', {'cell': c, 'source': source, 'round': round_no, 'queries': queries,
                                       'strategy': self.a.strategy})
            now = getattr(self.a, 'grow_now', 0) or len(queries)
            for q in queries[:now]:
                await self.search(t, c, source, q, stat)
            later += [(source, q) for q in queries[now:]]
        if later and (len(self.pool[c]) < 3 or len(self.usable(c)) < 3):
            for source, q in later:
                await self.search(t, c, source, q, stat)
            later = []
        for source in [x for x in srcs if x not in primary]:
            if len(self.pool[c]) >= 3:
                continue
            k = min(8, math.ceil(0.25 * n_specs) + 1)
            queries = await self.plan_queries(t, source, k if round_no == 0 else max(2, k // 2))
            self.append('plan.jsonl', {'cell': c, 'source': source, 'round': round_no, 'queries': queries,
                                       'strategy': self.a.strategy})
            for q in queries:
                await self.search(t, c, source, q, stat)
        if later:
            task = asyncio.create_task(self.grow_later(t, c, round_no, later, stat))
            self.growing[c] = task
            task.add_done_callback(lambda _t, c=c: (self.growing.pop(c, None), self.urgent.discard(c)))
        else:
            self.grown(c, round_no, stat)
        return stat['added']

    async def grow_later(self, t, c, round_no, later, stat):
        for source, q in later:
            http_mod.LOW.set(c not in self.urgent)      # this task only (asyncio tasks copy the context)
            await self.search(t, c, source, q, stat)
        self.grown(c, round_no, stat)

    def grown(self, c, round_no, stat):
        self.search_failed[c] += stat['failed']
        self.append('grown.jsonl', {'cell': c, 'round': round_no, 'added': stat['added'], 'failed': stat['failed']})

    async def search(self, t, c, source, q, stat):
        try:
            found = await self.sources[source].search(t, q, 10)
        except Reject as e:   # the catalog cannot serve this target: not a transient failure, no regrow
            self.counts[f'search_reject:{source}'] += 1
            self.append('errors.jsonl', {'stage': 'search_reject', 'source': source, 'query': q,
                                         'error': repr(e)[:300], 'cell': c})
            return
        except Exception as e:  # noqa: BLE001
            self.counts[f'search_error:{source}'] += 1
            stat['failed'] += 1
            self.append('errors.jsonl', {'stage': 'search', 'source': source, 'query': q, 'error': repr(e)[:300],
                                         'cell': c})
            return
        self.counts[f'searches:{source}'] += 1
        for cand in found:
            self.counts[f'candidates:{source}'] += 1
            lic = cand.get('license_raw')
            if lic is not None and classify(lic)[1] == 'excluded':
                self.counts[f'license_excluded:{source}'] += 1
                continue
            if cand['key'] in self.pool_keys:
                continue
            if decon.blocked_candidate(cand):
                self.counts[f'decon_blocked:{source}'] += 1
                continue
            cand = {**cand, 'query': q, 'cell': c, 'strategy': self.a.strategy}
            self.pool_keys.add(cand['key'])
            self.pool[c].append(cand)
            self.append('pool.jsonl', cand)
            stat['added'] += 1

    def usable(self, c):
        return [x for x in self.pool[c] if x['key'] not in self.failed and self.uses[x['key']] < MAX_USES]

    async def candidates(self, spec, t, c):
        async with self.cell_locks[c]:
            if len(self.usable(c)) < 3 and c in self.growing:
                # the cell's earlier grow still runs in the background and a spec now needs it: no longer low priority
                # (bench, 2026-11-04: left at low priority, these waits held slots while the gate's backlog grew)
                self.urgent.add(c)
                await self.growing[c]
            for round_no in range(2):
                avail = self.usable(c)
                if len(avail) >= 3 or self.planned[c] > round_no:
                    break
                self.planned[c] += 1
                await self.grow(t, c, self.cell_sizes[c], round_no)
            avail = self.usable(c)
        rng = random.Random(f'{self.a.seed}-{spec["spec_id"]}')
        fresh = [x for x in avail if not self.uses[x['key']]]
        used = [x for x in avail if self.uses[x['key']]]
        rng.shuffle(fresh)
        rng.shuffle(used)
        show = (fresh + used)[:SHOW]
        # code search hits carry no license until their repository is looked up
        for x in show:
            if x.get('license_raw') is None and hasattr(self.sources[x['source']], 'resolve_license'):
                try:
                    await self.sources[x['source']].resolve_license(x)
                except Exception:  # noqa: BLE001
                    x['license_raw'] = ''
                if classify(x['license_raw'])[1] == 'excluded':
                    self.failed.add(x['key'])
                    self.counts[f'license_excluded:{x["source"]}'] += 1
        return [x for x in show if x['key'] not in self.failed]

    # ---------------------------------------------------------------- fetch
    async def fetch(self, cand, t):
        aid = store.asset_id(cand['source'], cand['key'])
        async with self.fetch_locks[aid]:
            if aid in self.index:
                return self.index[aid]
            if cand['key'] in self.failed:
                return None
            staging = Path(tempfile.mkdtemp(prefix=f'{aid}-', dir=self.stage_root))
            base = {'asset_id': aid, 'source': cand['source'], 'key': cand['key'], 'url': cand.get('url'),
                    'page_url': cand.get('page_url'), 'title': cand.get('title'), 'description': cand.get('description'),
                    'license_raw': cand.get('license_raw'), 'cls': t['cls'], 'formats': t['formats'],
                    'language': t['language'], 'target': cell(t), 'query': cand.get('query'),
                    'strategy': cand.get('strategy'), 'size': cand.get('size'), 'meta': cand.get('meta')}
            try:
                prov = await self.sources[cand['source']].fetch(cand, staging, t)
                rec = {**base, **prov}
                spdx, verdict = classify(rec.get('license_raw'))
                if verdict == 'excluded':
                    raise Reject(f'license {rec.get("license_raw")!r} excluded')
                rec.update(license=spdx, license_verdict=verdict)
                if not any(p.is_file() for p in staging.rglob('*')):
                    raise Reject('nothing fetched')
                hits = await asyncio.to_thread(decon.shared_files, staging)
                if hits or decon.blocked_candidate(rec):
                    self.counts['decon_rejected'] += 1
                    raise Reject(f'decon: shares files with eval tasks: {hits or decon.blocked_candidate(rec)}')
                scrub = rec.pop('scrub_in_tree', True)
                rec = await asyncio.to_thread(self.writer.put, staging, rec, scrub)
                self.index[aid] = rec
                self.counts[f'fetched:{cand["source"]}'] += 1
                return rec
            except Exception as e:  # noqa: BLE001  (Reject, HTTP errors, broken archives)
                self.failed.add(cand['key'])
                self.counts[f'fetch_fail:{cand["source"]}'] += 1
                self.writer.reject(base, f'{type(e).__name__}: {e}')
                return None
            finally:
                subprocess.run(['chmod', '-R', 'u+rwX', str(staging)], capture_output=True)
                shutil.rmtree(staging, ignore_errors=True)

    # ---------------------------------------------------------------- per spec
    async def fit(self, spec, rec):
        user = P.FIT_USER.format(spec=spec_brief(spec), source=rec['source'], title=rec.get('title'),
                                 license=rec.get('license'), description=(rec.get('description') or '')[:600],
                                 manifest=manifest_mod.render(rec['manifest'], 16000))
        for _ in range(3):
            try:
                j = jparse(await self.llm(P.FIT_SYSTEM, user, 'fit'))
                paths = [str(p).strip().lstrip('/') or '.' for p in j.get('use_paths') or []]
                known = [r for _, r in rec['manifest']['tree']]
                ok = [p for p in paths if p in ('.', './') or any(r == p.rstrip('/') or r.startswith(p.rstrip('/') + '/')
                                                                 for r in known)]
                return {'fit': bool(j.get('fit')) and bool(ok or paths == []), 'reason': str(j.get('reason', ''))[:600],
                        'use_paths': ok or ['.'], 'idea': str(j.get('idea', ''))[:600],
                        'dropped_paths': [p for p in paths if p not in ok]}
            except (ValueError, KeyError, TypeError):
                continue
        return {'fit': False, 'reason': 'unparseable fit reply', 'use_paths': [], 'idea': ''}

    async def one(self, spec):
        t, c = target(spec), cell(target(spec))
        res = {'spec_id': spec['spec_id'], 'cell': c, 'cls': t['cls'], 'tries': [], 'status': 'no_candidates'}
        cands = await self.candidates(spec, t, c)
        if not cands:
            return res
        user = P.SHORTLIST_USER.format(spec=spec_brief(spec), cands='\n'.join(cand_line(i, x) for i, x in enumerate(cands)))
        picks = []
        for _ in range(3):
            try:
                j = jparse(await self.llm(P.SHORTLIST_SYSTEM, user, 'shortlist'))
                picks = [int(i) for i in j.get('picks', []) if str(i).isdigit() and int(i) < len(cands)][:self.a.tries]
                res['shortlist_why'] = str(j.get('why', ''))[:300]
                break
            except (ValueError, KeyError, TypeError):
                continue
        if not picks:
            res['status'] = 'no_shortlist'
            return res
        res['status'] = 'no_fit'
        for i in picks:
            cand = cands[i]
            if self.uses[cand['key']] >= MAX_USES:
                continue
            self.uses[cand['key']] += 1          # reserve while fetching and checking
            rec = await self.fetch(cand, t)
            if rec is None:
                self.uses[cand['key']] -= 1
                res['tries'].append({'key': cand['key'], 'fetch': 'failed'})
                continue
            f = await self.fit(spec, rec)
            self.append('fits.jsonl', {'spec_id': spec['spec_id'], 'asset_id': rec['asset_id'], 'cls': t['cls'],
                                       'source': rec['source'], **f})
            res['tries'].append({'key': cand['key'], 'asset_id': rec['asset_id'], 'fit': f['fit'], 'reason': f['reason']})
            if f['fit']:
                res.update(status='bound', asset_id=rec['asset_id'], use_paths=f['use_paths'], idea=f['idea'],
                           source=rec['source'])
                return res
            self.uses[cand['key']] -= 1
        return res

    def load_done(self):
        done = {}
        if (self.dir / 'results.jsonl').exists():
            for l in (self.dir / 'results.jsonl').read_text().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                if r['status'] not in ('error', 'no_candidates'):   # no_candidates: pools may have grown since
                    done[r['spec_id']] = r
        return done

    async def run(self):
        specs = [json.loads(l) for l in Path(self.a.specs).read_text().splitlines() if l.strip()]
        shutil.copyfile(self.a.specs, self.dir / 'specs.jsonl') if Path(self.a.specs) != self.dir / 'specs.jsonl' else None
        done = self.load_done()
        await self._setup(specs, done)
        todo = [s for s in specs if s['spec_id'] not in done]
        self.log(f'bind {self.a.name}: {len(specs)} specs, {len(done)} done, strategy {self.a.strategy}, '
                 f'sources {sorted(self.sources)}')
        await self._bind_all(specs, todo)
        await self.http.close()
        report(self.a.name)

    async def _setup(self, specs, done):
        from ..pipeline.llm import Endpoint, wait_healthy
        from .http import Http
        self.cell_sizes = Counter(cell(target(s)) for s in specs)
        self.stage_root = Path(os.environ.get('TMPDIR', tempfile.gettempdir())) / 'bind-staging'
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.writer = store.Writer(f'{self.a.name}-{socket.gethostname()}-{os.getpid()}')
        self.index = store.load_index()
        self.warm_start(set(self.cell_sizes))
        # resume: pools, uses and failures from earlier runs of this bind
        if (self.dir / 'pool.jsonl').exists():
            for l in (self.dir / 'pool.jsonl').read_text().splitlines():
                x = json.loads(l)
                if x['key'] not in self.pool_keys:
                    self.pool_keys.add(x['key'])
                    self.pool[x['cell']].append(x)
            # a cell counts as grown only when its searches finished (grown.jsonl); binds from before that marker
            # existed: a cell with at least 3 candidates counts as grown once, otherwise it is grown again
            # (2026-10-31: a round whose searches errored, e.g. GitHub rate limits, and left the cell short of 3
            # candidates is not counted, so the cell grows again)
            grown = self.dir / 'grown.jsonl'
            if grown.exists():
                plans = defaultdict(set)
                for l in (self.dir / 'plan.jsonl').read_text().splitlines():
                    p = json.loads(l)
                    plans[(p['cell'], p['round'])].update((p['source'], q) for q in p['queries'])
                errs = set()
                if (self.dir / 'errors.jsonl').exists():
                    for l in (self.dir / 'errors.jsonl').read_text().splitlines():
                        e = json.loads(l)
                        if e.get('stage') == 'search':
                            errs.add((e['source'], e['query']))
                for l in grown.read_text().splitlines():
                    g = json.loads(l)
                    failed = g['failed'] if 'failed' in g else len(plans[(g['cell'], g['round'])] & errs)
                    if failed and len(self.pool[g['cell']]) < 3:
                        continue
                    self.planned[g['cell']] = max(self.planned[g['cell']], g['round'] + 1)
            for l in (self.dir / 'plan.jsonl').read_text().splitlines():
                p = json.loads(l)
                if not self.planned[p['cell']] and len(self.pool[p['cell']]) >= 3:
                    self.planned[p['cell']] = 1
        keys = {store.asset_id(x['source'], x['key']): x['key'] for xs in self.pool.values() for x in xs}
        for r in done.values():
            if r['status'] == 'bound':
                self.uses[keys.get(r['asset_id'], r['asset_id'])] += 1
        await wait_healthy(self.a.llm)
        self.ep = Endpoint(self.a.llm, max_in_flight=self.a.concurrency)
        self.http = Http(concurrency=self.a.http_concurrency)
        self.sources = load_sources(self.http)

    def warm_start(self, cells):
        """--warm-from (m1-bind, 2026-11-04): start this bind's cells with the candidates earlier binds found for the
        same cells (same cell key and strategy), so the first spec of a known cell skips its plan and searches. In
        m1-pilot a spec whose cell was new held a bind slot 18 min on average (GitHub search gate queue) against 3.8 min
        for a known cell. Earlier binds' bound specs count toward MAX_USES, candidates whose fetch was rejected for a
        lasting reason (Reject, TooLarge) stay out, and license and decon are checked again. A cell grows once its
        imported candidates run short, with k sized for this bind, as before. Re-imported on every start (not
        written to this bind's pool.jsonl), before this bind's own pool, so pool order survives a resume."""
        names = (getattr(self.a, 'warm_from', None) or 'none').strip()
        if names == 'none':
            return
        if names == 'auto':
            dirs = [d for d in sorted(BINDS.iterdir()) if d.name != self.a.name and (d / 'pool.jsonl').exists()]
        else:
            dirs = [BINDS / n for n in names.split(',') if n and n != self.a.name]
        for r in store.load_rejected():
            if str(r.get('reason', '')).startswith(('Reject', 'TooLarge')) and r.get('key'):
                self.failed.add(r['key'])
        n_cand, prior, aid_key = 0, 0, {}
        for d in dirs:
            if not (d / 'pool.jsonl').exists():
                continue
            with open(d / 'pool.jsonl') as f:
                for line in f:
                    try:
                        x = json.loads(line)
                    except ValueError:
                        continue
                    aid_key[store.asset_id(x['source'], x['key'])] = x['key']
                    if (x.get('cell') not in cells or x['key'] in self.pool_keys
                            or x.get('strategy', 'targeted') != self.a.strategy):
                        continue
                    lic = x.get('license_raw')
                    if (lic is not None and classify(lic)[1] == 'excluded') or decon.blocked_candidate(x):
                        continue
                    self.pool_keys.add(x['key'])
                    self.pool[x['cell']].append({**x, 'warm_from': d.name})
                    n_cand += 1
        for d in dirs:
            last = {}
            if (d / 'results.jsonl').exists():
                for line in (d / 'results.jsonl').read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        last[r['spec_id']] = r
            for r in last.values():
                if r.get('status') == 'bound' and r.get('asset_id'):
                    key = (self.index.get(r['asset_id']) or {}).get('key') or aid_key.get(r['asset_id'])
                    if key:
                        self.uses[key] += 1
                        prior += 1
        warm = sum(n for c, n in self.cell_sizes.items() if len(self.usable(c)) >= 3)
        self.log(f'warm start from {[d.name for d in dirs]}: {n_cand} candidates in '
                 f'{sum(1 for c in cells if self.pool[c])} of {len(cells)} cells, {prior} earlier uses, '
                 f'{len(self.failed)} rejected keys; {warm} of {sum(self.cell_sizes.values())} specs start in a cell '
                 f'with >= 3 usable candidates')

    async def _bind_all(self, specs, todo):
        slots = asyncio.Semaphore(self.a.concurrency)
        n = 0

        async def guarded(spec):
            nonlocal n
            async with slots:
                try:
                    res = await self.one(spec)
                except Exception as e:  # noqa: BLE001
                    res = {'spec_id': spec['spec_id'], 'status': 'error', 'error': repr(e)[:500],
                           'trace': traceback.format_exc()[-1500:]}
            self.append('results.jsonl', res)
            n += 1
            if n % 20 == 0 or n == len(todo):
                self.log(f'[{n}/{len(todo)}] {Counter(json.loads(l)["status"] for l in (self.dir / "results.jsonl").read_text().splitlines())} '
                         f'http {self.http.stats} tokens {dict(self.tokens)}')

        await asyncio.gather(*(guarded(s) for s in todo))
        # cells left empty by search errors (rate-limit holds have expired by now): grow them once more
        last = {}
        for l in (self.dir / 'results.jsonl').read_text().splitlines():
            r = json.loads(l)
            last[r['spec_id']] = r
        retry = [s for s in specs if last.get(s['spec_id'], {}).get('status') == 'no_candidates'
                 and self.search_failed[cell(target(s))]]
        if retry:
            for c in {cell(target(s)) for s in retry}:
                self.planned[c] = 0
                self.search_failed[c] = 0
            self.log(f'second pass: {len(retry)} no-candidate specs in cells whose searches errored')
            todo, n = retry, 0
            await asyncio.gather(*(guarded(s) for s in todo))
        self.stop_growing()
        self.append('costs.jsonl', {'time': time.time(), 'http': self.http.stats, 'tokens': dict(self.tokens),
                                    'counts': dict(self.counts)})

    def stop_growing(self):
        """Every spec has its result: background grows (pools for specs that will not come) are cancelled."""
        for task in list(self.growing.values()):
            task.cancel()

    # ---------------------------------------------------------------- streamed into a running round (#118)
    async def stream(self):
        """Bind a round's specs marked bind=pending, in the round's queue order, while the round runs: each result is
        exported into the round at once (<round>/assets/<spec>/) and published as <round>/bind/<spec>.json, which the
        round waits for before it authors that spec (pipeline.round, --bind-window). Unbound results (no candidates,
        no fit, decon-flagged asset, error) make the spec an ordinary generated task. Resumable: finished specs keep
        their result. Stops early when the round stops (<round>/STOP, or every spec of the round has a record)."""
        rd = ROOT / 'rounds' / self.a.round
        out = rd / 'bind'
        out.mkdir(parents=True, exist_ok=True)
        specs = [s for s in (json.loads(l) for l in (rd / 'specs.jsonl').read_text().splitlines() if l.strip())
                 if s.get('bind') == 'pending']
        atomic_write(self.dir / 'specs.jsonl', ''.join(json.dumps(s) + '\n' for s in specs))
        done = self.load_done()
        await self._setup(specs, done)
        idx = store.load_index()
        by_id = {s['spec_id']: s for s in specs}
        # resume: publish results that were decided before a restart but never published
        for sid, r in done.items():
            if sid in by_id and not (out / f'{sid}.json').exists():
                await self.publish(rd, by_id[sid], r, idx)
        todo = [s for s in specs if s['spec_id'] not in done]
        self.log(f'bind {self.a.name} -> round {self.a.round}: {len(specs)} pending specs, {len(done)} done, '
                 f'strategy {self.a.strategy}, sources {sorted(self.sources)}')
        minutes = getattr(self.a, 'ramp_minutes', 0) or 0
        open_now = ramp_start(self.a.concurrency, getattr(self.a, 'ramp_start', 1.0)) if minutes else self.a.concurrency
        slots = asyncio.Semaphore(open_now)
        cap = [open_now]
        ramp = asyncio.create_task(ramp_slots(slots, open_now, self.a.concurrency, minutes,
                                              on_step=lambda k: cap.__setitem__(0, k)))
        if minutes:
            self.log(f'slots ramp: {open_now} now, {self.a.concurrency} after {minutes} min')
        stop = asyncio.Event()
        n = 0
        finished_at = []   # completion times of the last 10 min: the rate in HEARTBEAT (production status, #237)

        async def heartbeat():
            beats = 0
            while not stop.is_set():
                now = time.time()
                finished_at[:] = [t for t in finished_at if now - t < 600]
                atomic_write(out / 'HEARTBEAT', json.dumps({'time': now, 'host': socket.gethostname(),
                                                            'pid': os.getpid(), 'done': n, 'todo': len(todo),
                                                            'slots': cap[0], 'concurrency': self.a.concurrency,
                                                            'rate_per_min': round(len(finished_at) / 10, 2)}))
                # STOP every minute; the record scan (one stat per spec) every 10 min
                if (rd / 'STOP').exists() or (beats % 10 == 9 and await asyncio.to_thread(round_finished, rd)):
                    self.log('round stopped or finished: no new bindings')
                    stop.set()
                    return
                beats += 1
                await asyncio.sleep(60)

        async def guarded(spec):
            nonlocal n
            async with slots:
                if stop.is_set():
                    return
                res = await self.one_safe(spec)
            if res['status'] == 'no_candidates' and self.search_failed[res.get('cell')] and not stop.is_set():
                # searches of this cell errored (GitHub rate-limit holds): grow it again once the holds expire
                await asyncio.sleep(600)
                async with slots:
                    if stop.is_set():
                        return
                    c = res.get('cell')
                    async with self.cell_locks[c]:
                        if self.search_failed[c]:
                            self.planned[c], self.search_failed[c] = 0, 0
                    res = await self.one_safe(spec)
            self.append('results.jsonl', res)
            await self.publish(rd, spec, res, idx)
            n += 1
            finished_at.append(time.time())
            if n % 50 == 0 or n == len(todo):
                self.log(f'[{n}/{len(todo)}] {Counter(json.loads(l)["status"] for l in (self.dir / "results.jsonl").read_text().splitlines() if l.strip())} '
                         f'http {self.http.stats} tokens {dict(self.tokens)}')

        beat = asyncio.create_task(heartbeat())
        await asyncio.gather(*(guarded(s) for s in todo))
        stop.set()
        ramp.cancel()
        self.stop_growing()
        await beat
        self.append('costs.jsonl', {'time': time.time(), 'http': self.http.stats, 'tokens': dict(self.tokens),
                                    'counts': dict(self.counts)})
        await self.http.close()
        report(self.a.name)

    async def one_safe(self, spec):
        try:
            return await self.one(spec)
        except Exception as e:  # noqa: BLE001
            return {'spec_id': spec['spec_id'], 'status': 'error', 'error': repr(e)[:500],
                    'trace': traceback.format_exc()[-1500:]}

    async def publish(self, rd, spec, r, idx):
        """A bind result -> the round: export the asset if bound, then <round>/bind/<spec>.json (atomic). Every spec
        gets a result, or the round would wait for it: a failed export makes it an ordinary generated task."""
        sid = spec['spec_id']
        res = {'spec_id': sid, 'status': r['status'], 'bind': self.a.name, 'time': time.time()}
        if r['status'] == 'bound':
            try:
                rec = self.index.get(r['asset_id']) or idx.get(r['asset_id'])
                if rec is None:
                    raise KeyError(f'{r["asset_id"]} not in the asset store index')
                asset, reason = await asyncio.to_thread(export_one, rec, r, rd, sid)
                if asset is None:
                    res.update(status='decon_flagged', reason=reason, asset_id=r['asset_id'])
                else:
                    res['asset'] = asset
            except Exception as e:  # noqa: BLE001  (Lustre errors, a broken tar)
                self.log(f'{sid}: export of {r["asset_id"]} failed ({type(e).__name__}: {e}); the spec stays generated')
                res.update(status='export_error', error=f'{type(e).__name__}: {e}'[:500], asset_id=r['asset_id'])
        for i in range(5):
            try:
                atomic_write(rd / 'bind' / f'{sid}.json', json.dumps(res))
                return
            except OSError as e:   # a transient Lustre EIO: the round waits for this file
                self.log(f'{sid}: publishing the bind result failed ({e}); retry {i + 1}')
                await asyncio.sleep(30)
        raise RuntimeError(f'{sid}: could not publish its bind result')


def report(name):
    d = BINDS / name
    specs = {json.loads(l)['spec_id']: json.loads(l) for l in (d / 'specs.jsonl').read_text().splitlines()}
    res = {}
    for l in (d / 'results.jsonl').read_text().splitlines():
        r = json.loads(l)
        res[r['spec_id']] = r
    fits = [json.loads(l) for l in (d / 'fits.jsonl').read_text().splitlines()] if (d / 'fits.jsonl').exists() else []
    costs = [json.loads(l) for l in (d / 'costs.jsonl').read_text().splitlines()] if (d / 'costs.jsonl').exists() else []
    st = Counter(r['status'] for r in res.values())
    by = defaultdict(Counter)
    for r in res.values():
        by[r.get('cls') or specs[r['spec_id']]['assets'][0]['class']][r['status']] += 1
    lines = [f'# Asset binding {name}', '', f'{len(res)} of {len(specs)} specs processed: {dict(st)}', '',
             f'Bound share: {st["bound"] / max(1, len(res)):.0%}', '', '## By anchor class', '',
             '| Class | Specs | Bound | No fit | No candidates | Other |', '| --- | --- | --- | --- | --- | --- |']
    for k, v in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(v.values())
        lines.append(f'| {k} | {tot} | {v["bound"]} ({v["bound"] / tot:.0%}) | {v["no_fit"]} | {v["no_candidates"]} | '
                     f'{tot - v["bound"] - v["no_fit"] - v["no_candidates"]} |')
    fs = defaultdict(lambda: [0, 0])
    for f in fits:
        fs[f['source']][0] += 1
        fs[f['source']][1] += bool(f['fit'])
    lines += ['', '## Fit checks by source', '', '| Source | Checks | Fit |', '| --- | --- | --- |']
    lines += [f'| {k} | {v[0]} | {v[1]} ({v[1] / v[0]:.0%}) |' for k, v in sorted(fs.items(), key=lambda x: -x[1][0])]
    bound = [r for r in res.values() if r['status'] == 'bound']
    reuse = Counter(r['asset_id'] for r in bound)
    lines += ['', '## Diversity', '', f'- Distinct assets bound: {len(reuse)} for {len(bound)} specs; '
              f'used twice: {sum(1 for v in reuse.values() if v > 1)}']
    if bound:
        idx = store.load_index(with_manifest=False)
        lic = Counter(idx[r['asset_id']].get('license_verdict') for r in bound if r['asset_id'] in idx)
        srcs = Counter(r.get('source') for r in bound)
        lines += [f'- Sources of bound assets: {dict(srcs)}', f'- License verdicts: {dict(lic)}']
        try:
            from ..pipeline.textsim import near_duplicates
            texts = {r['asset_id']: f'{idx[r["asset_id"]].get("title", "")} {idx[r["asset_id"]].get("description", "")}'
                     for r in bound if r['asset_id'] in idx}
            pairs, nearest = near_duplicates(texts) if len(texts) > 1 else ([], {})
            nn = sorted(nearest.values())
            lines.append(f'- Asset title+description near-duplicate pairs (cosine >= 0.6): {len(pairs)}; nearest-neighbour '
                         f'cosine median {nn[len(nn) // 2] if nn else 0:.2f}')
        except Exception as e:  # noqa: BLE001
            lines.append(f'- (diversity scan failed: {e})')
    if costs:
        c = costs[-1]
        lines += ['', '## Cost', '', f'- HTTP: {c["http"]}', f'- LLM tokens: {c["tokens"]}', f'- Counts: {c["counts"]}']
    reasons = Counter(re.sub(r'[^a-z ]', '', f['reason'].lower())[:60] for f in fits if not f['fit'])
    lines += ['', '## Bound specs', '', '| Spec | Class | Action | Asset | Source |', '| --- | --- | --- | --- | --- |']
    for r in sorted(bound, key=lambda r: r['spec_id']):
        s = specs.get(r['spec_id'])
        if s is None:   # a result for a spec no longer in the round's specs (m1-hero's v2h0 re-plan, #165)
            continue
        lines.append(f'| {r["spec_id"]} | {r["cls"]} | {s["agent_action"]} | {r["asset_id"]} | {r.get("source")} |')
    (d / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(d / 'REPORT.md')
    return st


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def round_finished(rd):
    """Every spec of the round has its record (or error)."""
    try:
        sids = [json.loads(l)['spec_id'] for l in (rd / 'specs.jsonl').read_text().splitlines() if l.strip()]
    except OSError:
        return False
    from ..pipeline.roundpack import RoundFiles   # packed or loose (#245)
    return RoundFiles(rd).has_outcome(sids)


def export_one(rec, r, rd, sid):
    """One bound spec's asset -> <rd>/assets/<sid>/ (asset.tar.zst = the shipped selection, gen.txt = the asset as the
    generator sees it, judge.txt = provenance and manifest for the judge). -> (the spec's `asset` dict, None) or
    (None, reason) when the asset-level decon flags it (repo/URL references, files shared with eval tasks)."""
    from . import render
    with tempfile.TemporaryDirectory(prefix='decon-') as tmp:
        hits = decon.shared_files(store.extract(rec, Path(tmp) / 'a'))
    reason = decon.blocked_candidate(rec) or (f'shares files with eval tasks: {hits}' if hits else None)
    if reason:
        return None, reason
    adir = rd / 'assets' / sid
    adir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='export-') as tmp:
        tar = store.task_tar(rec, Path(tmp) / 'asset.tar.zst', None if r['use_paths'] == ['.'] else r['use_paths'])
        store.copy_to_fast(tar, adir / 'asset.tar.zst')
    (adir / 'gen.txt').write_text(render.for_generator(rec, r['use_paths'], rec['cls']))
    (adir / 'judge.txt').write_text(render.for_judge(rec, r['use_paths']))
    return {'asset_id': r['asset_id'], 'use_paths': r['use_paths'], 'idea': r.get('idea', ''),
            'dir': str(adir), 'tar_bytes': (adir / 'asset.tar.zst').stat().st_size,
            **{k: rec.get(k) for k in ('source', 'title', 'description', 'url', 'page_url', 'version',
                                       'license', 'license_verdict', 'attribution', 'sha256')}}, None


def export(a):
    """Bound specs -> <ROOT>/rounds/<round>/specs.jsonl (spec ids prefixed; the asset binding rides on the spec), and
    per spec <round>/assets/<spec id>/: asset.tar.zst (the shipped selection), gen.txt (the asset as the generator
    sees it), judge.txt (provenance and manifest for the judge). Runs in the assets venv, which can read every
    format the manifest describes. (Production rounds bind while they run instead: `run --round`, Binder.stream.)"""
    d = BINDS / a.name
    rd = ROOT / 'rounds' / a.round
    if (rd / 'specs.jsonl').exists() and not a.force:
        raise SystemExit(f'{rd}/specs.jsonl exists (--force to overwrite)')
    idx = store.load_index()
    out = []
    specs = {json.loads(l)['spec_id']: json.loads(l) for l in (d / 'specs.jsonl').read_text().splitlines()}
    only = set(Path(a.only).read_text().split()) if a.only else None
    for l in (d / 'results.jsonl').read_text().splitlines():
        r = json.loads(l)
        if r['status'] != 'bound' or (only is not None and r['spec_id'] not in only):
            continue
        s = dict(specs[r['spec_id']])
        s.pop('round', None)
        rec = idx[r['asset_id']]
        sid = f'{a.prefix}-{r["spec_id"]}'
        # asset-level decon (binds before decon.py skipped it): repo/URL references and shared files
        asset, reason = export_one(rec, r, rd, sid)
        if asset is None:
            print(f'{r["spec_id"]}: {r["asset_id"]} decon-flagged ({reason}); not exported')
            continue
        s.update(spec_id=sid, orig_spec_id=r['spec_id'], bind=a.name, asset=asset)
        out.append(s)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / 'specs.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in sorted(out, key=lambda s: s['spec_id'])))
    print(f'{rd}/specs.jsonl: {len(out)} bound specs')


def submit(a):
    from ..common import pin_planned_run
    pin_planned_run(a.name)   # a production run's bind, also when submitted by hand: the run's registry root
    pin_planned_run(getattr(a, 'round', None))   # and the run whose round it binds (refused if the two disagree)
    code = frozen_checkout()
    d = BINDS / a.name
    d.mkdir(parents=True, exist_ok=True)
    source = f'--round {a.round}' if a.round else f'--specs {Path(a.specs).resolve()}'
    args = (f'--name {a.name} {source} --strategy {a.strategy} --llm {a.llm} --concurrency {a.concurrency} '
            f'--http-concurrency {a.http_concurrency} --tries {a.tries} --seed {a.seed} --warm-from {a.warm_from} '
            f'--grow-now {a.grow_now} --ramp-minutes {a.ramp_minutes} --ramp-start {a.ramp_start}')
    from synth.common import PROXY_ENV
    from tbshared import serving
    env = {'USER': 'user', 'HOME': '/home/user', 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(code),
           'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'TASKSYNTH_GH_SHARE': str(a.gh_share), **PROXY_ENV,
           **serving.forwarded()}   # TASKSYNTH_SERVING_ROOT when set here: the job's llm clients read that registry
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = d / 'job.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $TMPDIR; exec {PYTHON} -u -m task_synthesis.assets.bind run {args}'"
initialdir = {code}
output = {d}/job.$(Cluster).out
error = {d}/job.$(Cluster).err
log = {d}/job.condor.log
request_cpus = {a.cpus}
request_memory = {a.memory_gib * 1024}
request_disk = 150G
+JobBatchName = "tasksynth-bind-{a.name}"
+MaxRuntime = 172800
getenv = false
environment = "{environment}"
requirements = (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "gpu110") && (TARGET.UtsnameNodename != "cpu105") && (TARGET.UtsnameNodename != "cpu301") && (TARGET.UtsnameNodename != "gpu180") && (TARGET.UtsnameNodename != "cpu208")
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    job = re.search(r'cluster (\d+)', r.stdout).group(1) + '.0'
    (d / 'job.json').write_text(json.dumps({'job_id': job, 'code': str(code), 'time': time.time(), 'args': vars(a)}))
    print(job)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    for op in ('run', 'submit'):
        p = sub.add_parser(op)
        p.add_argument('--name', required=True)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('--specs', help='specs to bind (offline: export them into a round afterwards)')
        g.add_argument('--round', help='stream: bind this round\'s bind=pending specs while it runs (#118)')
        p.add_argument('--strategy', default='targeted', choices=('targeted', 'untargeted'))
        p.add_argument('--llm', default='gen-ds')
        p.add_argument('--concurrency', type=int, default=48)
        p.add_argument('--http-concurrency', type=int, default=24)
        p.add_argument('--tries', type=int, default=3, help='fetched candidates fit-checked per spec')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--warm-from', default='none',
                       help='earlier binds whose pools start this one\'s cells: comma list, auto (every other bind), none')
        p.add_argument('--grow-now', type=int, default=0,
                       help='queries per catalog a grow runs before the spec proceeds; the rest in the background at '
                            'low priority (0: all now)')
        p.add_argument('--ramp-minutes', type=float, default=15,
                       help='stream mode: open the --concurrency slots evenly over this many minutes (0: all at once)')
        p.add_argument('--ramp-start', type=float, default=0.1, help='share of the slots open at the start of the ramp')
        if op == 'submit':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=8)
            p.add_argument('--memory-gib', type=int, default=32)
            p.add_argument('--gh-share', type=int, default=1, help='bind jobs sharing the GitHub token at once')
    p = sub.add_parser('report')
    p.add_argument('--name', required=True)
    p = sub.add_parser('export')
    p.add_argument('--name', required=True)
    p.add_argument('--round', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--only', help='file of spec ids to export (default: every bound spec)')
    p.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.op == 'run':
        asyncio.run(Binder(a).stream() if a.round else Binder(a).run())
    elif a.op == 'submit':
        submit(a)
    elif a.op == 'report':
        report(a.name)
    else:
        export(a)


if __name__ == '__main__':
    main()

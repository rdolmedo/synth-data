"""Offline tests for streamed-binding throughput (m1-bind, #133): the rate gate's low-priority lane, grows that run
their first queries now and the rest in the background, and the warm start from earlier binds' pools."""
import asyncio
import json
import tempfile
import time
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest import mock

from task_synthesis.assets import bind as B, http as H


class GateTest(unittest.TestCase):
    def test_low_priority_waits_for_normal_requests(self):
        async def go():
            gate, order = H.RateGate(), []

            async def req(name, low, delay=0):
                await asyncio.sleep(delay)
                if low:
                    H.LOW.set(True)
                await gate.wait('k', 0.02, H.LOW.get())
                order.append(name)
            # n1, n2 queue behind n0's spacing; l0, l1 arrive while they wait and go after them (a low request that
            # took the gate before any normal one waited keeps its turn: at most one interval)
            await asyncio.gather(req('n0', False), req('n1', False, 0.001), req('n2', False, 0.001),
                                 req('l0', True, 0.002), req('l1', True, 0.002))
            return order
        order = asyncio.run(go())
        self.assertEqual(order[:3], ['n0', 'n1', 'n2'])
        self.assertEqual(sorted(order[3:]), ['l0', 'l1'])

    def test_spacing_holds_across_priorities(self):
        async def go():
            gate, times = H.RateGate(), []

            async def req(low):
                await gate.wait('k', 0.03, low)
                times.append(time.monotonic())
            await asyncio.gather(*(req(i % 2 == 0) for i in range(6)))
            return times
        t = sorted(asyncio.run(go()))
        self.assertTrue(all(b - a >= 0.029 for a, b in zip(t, t[1:])))


class StubSource:
    def __init__(self, name, per_query=4):
        self.name, self.per_query, self.queries = name, per_query, []

    async def search(self, t, q, n):
        self.queries.append((q, H.LOW.get()))
        await asyncio.sleep(0.001)
        return [{'source': self.name, 'key': f'{self.name}:{q}:{i}', 'title': q, 'license_raw': 'MIT'}
                for i in range(self.per_query)]


def make_binder(tmp, grow_now=0, warm_from='none', name='b'):
    with mock.patch.object(B, 'BINDS', Path(tmp)):
        b = B.Binder(Namespace(name=name, strategy='targeted', grow_now=grow_now, warm_from=warm_from, seed=0))
    b.log = lambda m: None
    return b


class GrowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.t = {'cls': 'tabular-data', 'formats': ['csv'], 'language': None, 'services': [], 'domain': 'x',
                  'scale': 'none'}     # routes: zenodo, hf-dataset, openml, github-file (fallback)
        self.c = B.cell(self.t)

    def binder(self, grow_now, per_query=4):
        b = make_binder(self.tmp, grow_now, name=f'g{grow_now}{per_query}')
        b.sources = {s: StubSource(s, per_query) for s in ('zenodo', 'hf-dataset', 'openml', 'github-file')}
        b.plans_at = []

        async def plan(t, source, k):
            b.plans_at.append(time.monotonic())
            await asyncio.sleep(0.05)
            return [f'{source}-q{i}' for i in range(k)]
        b.plan_queries = plan
        return b

    def run_grow(self, b, n_specs=40):
        async def go():
            await b.grow(self.t, self.c, n_specs, 0)
            fg = {s: list(src.queries) for s, src in b.sources.items()}
            while b.growing:
                await asyncio.gather(*list(b.growing.values()))
            return fg
        return asyncio.run(go())

    def test_first_queries_now_rest_in_background_same_pool(self):
        full, split = self.binder(0), self.binder(3)
        self.run_grow(full)
        fg = self.run_grow(split)
        self.assertEqual(len(fg['zenodo']), 3)                     # k = 22 for 40 specs; 3 before the spec proceeds
        self.assertEqual(len(split.sources['zenodo'].queries), 22)
        self.assertTrue(all(low for _, low in split.sources['zenodo'].queries[3:]))
        self.assertFalse(any(low for _, low in split.sources['zenodo'].queries[:3]))
        self.assertEqual(sorted(x['key'] for x in full.pool[self.c]), sorted(x['key'] for x in split.pool[self.c]))
        self.assertEqual(split.sources['github-file'].queries, [])   # enough candidates: no code-search fallback
        grown = (Path(self.tmp) / split.a.name / 'grown.jsonl').read_text().splitlines()
        self.assertEqual(len(grown), 1)
        self.assertEqual(json.loads(grown[0])['added'], len(full.pool[self.c]))
        self.assertLess(max(split.plans_at) - min(split.plans_at), 0.04)   # catalogs planned at once

    def test_short_cell_runs_everything_now(self):
        b = self.binder(1, per_query=0)       # nothing found: every query now, then the github-file fallback
        fg = self.run_grow(b)
        self.assertEqual(len(fg['zenodo']), 22)
        self.assertEqual(len(fg['github-file']), 8)      # fallback k: min(8, ceil(40 / 4) + 1)
        self.assertEqual(b.growing, {})

    def test_candidates_waits_for_a_background_grow_when_short(self):
        b = self.binder(1)
        b.cell_sizes = Counter({self.c: 40})
        b.uses = Counter()

        async def go():
            await b.grow(self.t, self.c, 40, 0)
            await asyncio.sleep(0.005)                       # a few background queries at low priority first
            b.planned[self.c] = 1
            for x in b.pool[self.c]:
                b.uses[x['key']] = B.MAX_USES               # every candidate found so far is used up
            got = await b.candidates({'spec_id': 's'}, self.t, self.c)
            return got
        got = asyncio.run(go())
        self.assertTrue(got)                                  # waited for the background part, no round-1 grow
        self.assertEqual(b.planned[self.c], 1)
        lows = [low for _, low in b.sources['zenodo'].queries]
        self.assertTrue(lows[1] and not lows[-1])             # promoted to normal priority once a spec waited
        self.assertEqual(b.urgent, set())


class WarmStartTest(unittest.TestCase):
    def test_imports_cells_counts_uses_and_skips_rejects(self):
        tmp = Path(tempfile.mkdtemp())
        old = tmp / 'old'
        old.mkdir()
        pool = [{'source': 'zenodo', 'key': f'k{i}', 'cell': 'c1', 'strategy': 'targeted', 'license_raw': 'cc-by-4.0'}
                for i in range(4)]
        pool += [{'source': 'zenodo', 'key': 'u0', 'cell': 'c1', 'strategy': 'untargeted', 'license_raw': 'MIT'},
                 {'source': 'zenodo', 'key': 'x0', 'cell': 'other', 'strategy': 'targeted', 'license_raw': 'MIT'}]
        (old / 'pool.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in pool))
        aid = B.store.asset_id('zenodo', 'k0')
        (old / 'results.jsonl').write_text(json.dumps({'spec_id': 'a', 'status': 'bound', 'asset_id': aid}) + '\n'
                                           + json.dumps({'spec_id': 'b', 'status': 'no_fit'}) + '\n')
        b = make_binder(tmp, warm_from='auto', name='new')
        b.cell_sizes, b.index = Counter({'c1': 5}), {}
        rejected = [{'key': 'k1', 'reason': 'Reject: 9000 files'}, {'key': 'k2', 'reason': 'HTTPError: 503'}]
        with mock.patch.object(B, 'BINDS', tmp), mock.patch.object(B.store, 'load_rejected', return_value=rejected), \
                mock.patch.object(B.decon, 'blocked_candidate', return_value=None):
            b.warm_start({'c1'})
        self.assertEqual(sorted(x['key'] for x in b.pool['c1']), ['k0', 'k1', 'k2', 'k3'])
        self.assertNotIn('other', b.pool)
        self.assertEqual(b.uses['k0'], 1)
        self.assertIn('k1', b.failed)                         # lasting reject
        self.assertNotIn('k2', b.failed)                      # transient: may be fetched again
        self.assertEqual(sorted(x['key'] for x in b.usable('c1')), ['k0', 'k2', 'k3'])
        self.assertEqual(b.planned['c1'], 0)                  # grows (with this bind's k) once the imports run short

    def test_none_imports_nothing(self):
        b = make_binder(tempfile.mkdtemp(), warm_from='none')
        b.warm_start({'c1'})
        self.assertEqual(dict(b.pool), {})


if __name__ == '__main__':
    unittest.main()

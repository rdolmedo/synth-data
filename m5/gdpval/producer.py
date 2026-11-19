"""Producer (one CPU condor job per range of spec indices): generate with DeepSeek, run the problem-side checks in the
SIF, repair failed input scripts (the error goes back to the generator, at most REPAIRS rounds), and close a Harbor
bundle every BATCH accepted problems, so problems stream into Super while generation goes on.

  python -m m5.gdpval.producer --root ROOT --start S --n N [--batch 250] [--check-workers 6] [--docs INDEX]

ROOT/gen.jsonl            every generator reply (also repairs, with "repair": k)
ROOT/check.jsonl          the final check result per problem
ROOT/batches/bNNN/        check.jsonl + inputs.tar of the batch's accepted problems, then bundle.tar(+.json),
                          then READY (the gpu123 driver launches a pi_eval run on it)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import tarfile
import time
from pathlib import Path

from m5.gdpval import prompts as P
from m5.gdpval import stage
from m5.gdpval.generate import Gate, static_check
from m5.gdpval.llm import Endpoint, wait_healthy
from m5.gdpval.sampler import Sampler

REPAIRS = 2
REPAIR_MSG = '''The script failed when it was run in the sandbox (current working directory = the directory where the
files must be written). Error output:

{err}

Reply with the corrected complete script only, between <script> and </script> tags. Keep the same file names and
the same content plan; fix the error (and anything else that would fail).'''


async def ask(ep: Endpoint, msgs, effort, affinity, log):
    extra = dict(ep.extra)
    extra['reasoning_effort'] = effort
    if isinstance(extra.get('chat_template_kwargs'), dict) and 'reasoning_effort' in extra['chat_template_kwargs']:
        extra['chat_template_kwargs'] = {**extra['chat_template_kwargs'], 'reasoning_effort': effort}
    for _ in range(8):
        try:
            content, reasoning, usage, _ = await ep.chat_tools(msgs, None, extra=extra, affinity=affinity)
            ep.release(affinity)
            return content, reasoning, usage
        except Exception as e:  # noqa: BLE001  (restart, overload: wait for a healthy endpoints.json)
            log.append(f'{type(e).__name__}: {str(e)[:200]}')
            if 'BadRequest' in type(e).__name__:
                return '', '', {'error': str(e)[:300]}
            await asyncio.sleep(60)
            await wait_healthy(ep.name)
            ep.refresh()
    return '', '', {'error': 'gave up'}


class Producer:
    def __init__(self, a):
        self.a, self.root = a, Path(a.root)
        (self.root / 'batches').mkdir(parents=True, exist_ok=True)
        self.sampler = Sampler()
        self.docs = None
        if a.docs:
            from m5.gdpval.docs import DocPicker
            self.docs = DocPicker(a.docs, a.part, a.nparts)
        self.lock = asyncio.Lock()
        self.check_sem = asyncio.Semaphore(a.check_workers)
        self.scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / 'm5prod'
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.done = self._done()
        self.pending = []   # accepted, not yet in a closed batch: (check result, input blobs)
        self.nbatch = len(list((self.root / 'batches').glob('b*')))

    def _done(self):
        p = self.root / 'check.jsonl'
        return {json.loads(x)['idx'] for x in p.read_text().splitlines()} if p.exists() else set()

    async def write(self, name, rec):
        async with self.lock:
            with open(self.root / name, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    async def check(self, rec):
        async with self.check_sem:
            return await asyncio.to_thread(stage.check_one, rec, self.scratch)

    async def one(self, ep, gate, idx):
        spec = self.sampler.spec(idx, self.a.seed)
        if self.a.lanes and spec['lane'] not in self.a.lanes.split(','):
            return
        doc = None
        if spec['lane'] == 'real':
            doc = self.docs.pick(spec) if self.docs else None
            if doc is None:   # no real document available: this index waits for a later run with docs
                return
        msgs = [{'role': 'system', 'content': P.SYSTEM}, {'role': 'user', 'content': P.user_prompt(spec, doc)}]
        rec = {'idx': idx, 'spec': spec, 'doc': {k: v for k, v in (doc or {}).items() if k != 'text'} or None,
               'effort': self.a.effort, 'started': time.time(), 'errors': []}
        async with gate:
            content, reasoning, usage = await ask(ep, msgs, self.a.effort, f'gen-{idx}', rec['errors'])
        rec.update(usage=usage, seconds=round(time.time() - rec['started']), reasoning_chars=len(reasoning),
                   raw=content)
        try:
            rec['problem'] = P.parse(content)
            rec['static_bad'] = static_check({**spec, '_doc_name': (doc or {}).get('name')}, rec['problem'])
        except ValueError as e:
            rec['parse_error'] = str(e)
        await self.write('gen.jsonl', rec)
        res, blobs = await self.check(rec)
        k = 0
        while (not res['accepted'] and k < REPAIRS and res.get('why', '').split(':')[0] in
               ('script_failed', 'script_timeout', 'file_check', 'validate_failed')):
            k += 1
            err = res.get('stderr') or res.get('why')
            msgs2 = msgs + [{'role': 'assistant', 'content': content},
                            {'role': 'user', 'content': REPAIR_MSG.format(err=err[-3000:])}]
            async with gate:
                c2, r2, u2 = await ask(ep, msgs2, self.a.effort, f'rep-{idx}', rec['errors'])
            try:
                import re
                m = re.findall(r'<script>\s*\n?(.*?)\n?\s*</script>', c2, re.S)
                script = m[-1].strip() if m else ''
                script = re.sub(r'^```(?:python)?\s*\n', '', script)
                script = re.sub(r'\n```\s*$', '', script)
            except Exception:  # noqa: BLE001
                script = ''
            await self.write('gen.jsonl', {'idx': idx, 'repair': k, 'usage': u2, 'raw': c2, 'error_in': err[-1500:]})
            if not script:
                break
            rec['problem'] = {**rec['problem'], 'script': script}
            rec['repairs'] = k
            content = c2
            res, blobs = await self.check(rec)
        res['repairs'] = k
        res['usage'] = rec['usage']
        if res['accepted']:
            await self.write('final.jsonl', rec)
        await self.write('check.jsonl', res)
        if res['accepted']:
            async with self.lock:
                self.pending.append((rec, res, blobs))
                if len(self.pending) >= self.a.batch:
                    batch, self.pending = self.pending[:self.a.batch], self.pending[self.a.batch:]
                    await self.close(batch)

    async def close(self, batch):
        name = f'b{self.nbatch:03d}'
        self.nbatch += 1
        await asyncio.to_thread(self._close, name, batch)

    def _close(self, name, batch):
        d = self.root / 'batches' / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'gen.jsonl', 'w') as f:
            for rec, _, _ in batch:
                f.write(json.dumps(rec) + '\n')
        with open(d / 'check.jsonl', 'w') as f:
            for _, res, _ in batch:
                f.write(json.dumps(res) + '\n')
        local = self.scratch / f'{name}-inputs.tar'
        with tarfile.open(local, 'w') as tf:
            for _, res, blobs in batch:
                for n, b in blobs.items():
                    ti = tarfile.TarInfo(f'{res["pid"]}/{n}')
                    ti.size, ti.mtime = len(b), 0
                    tf.addfile(ti, io.BytesIO(b))
        import shutil
        shutil.copyfile(local, d / 'inputs.tar.partial')
        os.replace(d / 'inputs.tar.partial', d / 'inputs.tar')
        local.unlink()
        args = argparse.Namespace(gen=[str(d / 'gen.jsonl')], check=[str(d)], out=str(d / 'bundle.tar'), limit=0)
        stage.bundle(args)
        (d / 'READY').write_text(json.dumps({'n': len(batch), 'at': time.time()}))
        print(time.strftime('%H:%M:%S'), 'closed', name, len(batch), flush=True)

    async def recover(self):
        """Accepted problems of an earlier run of this producer that never made it into a closed batch: re-run their
        (seeded) input scripts to get the files back, and put them back in the pending list."""
        batched = set()
        for b in (self.root / 'batches').glob('b*/check.jsonl'):
            batched |= {json.loads(x)['idx'] for x in b.read_text().splitlines()}
        p = self.root / 'final.jsonl'
        if not p.exists():
            return
        recs = {}
        for x in p.read_text().splitlines():
            r = json.loads(x)
            if r['idx'] not in batched:
                recs[r['idx']] = r
        for r in recs.values():
            res, blobs = await self.check(r)
            if res['accepted']:
                self.pending.append((r, res, blobs))
        print('recovered', len(self.pending), 'accepted problems not yet batched', flush=True)
        while len(self.pending) >= self.a.batch:
            batch, self.pending = self.pending[:self.a.batch], self.pending[self.a.batch:]
            await self.close(batch)

    async def run(self):
        a = self.a
        await wait_healthy(a.endpoint)
        ep = Endpoint(a.endpoint, max_in_flight=512, timeout=6 * 3600)
        gate = Gate(1.0 / max(1, a.producers))
        todo = [i for i in range(a.start, a.start + a.n) if i not in self.done]
        await self.recover()
        print(len(todo), 'specs to generate', flush=True)
        # bounded fan-out: keep at most ~2x the cap of coroutines alive (the gate does the real limiting)
        sem = asyncio.Semaphore(1024)

        async def guarded(i):
            async with sem:
                try:
                    await self.one(ep, gate, i)
                except Exception as e:  # noqa: BLE001
                    print('error', i, type(e).__name__, str(e)[:300], flush=True)
        await asyncio.gather(*[guarded(i) for i in todo])
        if self.pending and a.flush:
            await self.close(self.pending)
            self.pending = []
        print('producer done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--batch', type=int, default=250)
    ap.add_argument('--check-workers', type=int, default=6)
    ap.add_argument('--seed', default='gdpval-m5-v1')
    ap.add_argument('--endpoint', default='m5-ds')
    ap.add_argument('--effort', default='high')
    ap.add_argument('--docs', default=None, help='docpool.py pool JSONL (real lane)')
    ap.add_argument('--part', type=int, default=0, help='this producer\'s share of the doc pool')
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--lanes', default=None, help='only these lanes (prototyping)')
    ap.add_argument('--producers', type=int, default=1, help='producers sharing the DeepSeek cap')
    ap.add_argument('--flush', action=argparse.BooleanOptionalAction, default=True,
                    help='close a last partial batch at the end')
    a = ap.parse_args()
    asyncio.run(Producer(a).run())


if __name__ == '__main__':
    main()

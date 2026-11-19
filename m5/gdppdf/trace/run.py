"""Super traces for GDP.pdf-style problems with the eval's exact text-only protocol (aa_evals/gdppdf, #348):

- one user message, no system prompt, no tools: content = [{'type': 'text', 'text': render_text(prompt, pages,
  pdf_name)}] (aa_evals.gdppdf.inputs.render_text with note=None, the text-alone path);
- the super-vl preset body (aa_evals.gdppdf.run.PRESETS: T 0.6, top_p 0.95, enable_thinking, medium_effort off);
- 131,072-token window: planned prompt = /tokenize(text) + 512; refused (input_too_long) if it leaves less than the
  32,768 output reserve; max_tokens = 131072 - planned prompt (run.py's cap_window rule).

  python -m m5.gdppdf.trace.run --problems '<dir>/problems-*.jsonl' --docs <docs index> --out <dir>/traces

Streams: problems files are re-scanned every 60 s, so accepted problems flow in while generation runs. Every trace
is kept (ok, no_answer, length, input_too_long), one per problem; resumable (problem_ids already in the output are
skipped). Engines and per-engine session caps come from m5-coord/alloc.json, re-read every 60 s.
"""
import argparse
import asyncio
import glob
import json
import os
import random
import time
from pathlib import Path

from aa_evals.gdppdf import inputs
from aa_evals.gdppdf.run import CAPPED_OUTPUT_RESERVE, PRESETS
from m5.gdppdf.docs.store import DocStore
from m5.gdppdf.llm import ALLOC, Server
from m5.gdppdf.util import Appender

WINDOW = 131072
BODY = PRESETS['super-vl']['body']   # {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 'rest', chat_template_kwargs}


def super_body():
    b = {k: v for k, v in BODY.items() if k != 'max_tokens'}
    return json.loads(json.dumps(b))


def engines_from_alloc():
    try:
        return list(json.loads(ALLOC.read_text())['super']['serving'])
    except (OSError, ValueError, KeyError):
        return []


class Pool:
    def __init__(self):
        self.servers = {}

    async def sync(self):
        for name in engines_from_alloc():
            if name not in self.servers:
                try:
                    self.servers[name] = await asyncio.wait_for(Server(name, 'super').start(), 600)
                    print(f'[trace] engine {name} joined', flush=True)
                except asyncio.TimeoutError:
                    print(f'[trace] engine {name} not reachable yet', flush=True)

    def pick(self):
        live = [s for s in self.servers.values() if s.sem.used < s.sem.limit]
        if not live:
            return None
        return min(live, key=lambda s: (s.sem.used / max(1, s.sem.limit), random.random()))


async def run(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'traces-{os.environ.get("_CONDOR_CLUSTER", "local")}-{int(time.time())}.jsonl'
    done = set()
    for p in glob.glob(str(out_dir / 'traces-*.jsonl')):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                done.add(json.loads(l)['problem_id'])
    print(f'[trace] {len(done)} traces already done', flush=True)
    store = DocStore(a.docs)
    pool = Pool()
    await pool.sync()
    out = Appender(out_path)
    queue = []
    seen = set(done)
    stop_at = a.max_tokens_total
    totals = {'traces': 0, 'tokens': 0}
    for p in glob.glob(str(out_dir / 'traces-*.jsonl')):
        for l in open(p):
            if l.endswith('\n') and l.strip():
                u = json.loads(l).get('usage') or {}
                totals['tokens'] += (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0)
    inflight = set()

    def scan():
        new = 0
        for p in sorted(glob.glob(a.problems)):
            for l in open(p):
                if not l.endswith('\n') or not l.strip():
                    continue
                r = json.loads(l)
                if r.get('status', 'accepted') != 'accepted' or r['problem_id'] in seen:
                    continue
                seen.add(r['problem_id'])
                queue.append(r)
                new += 1
        return new

    async def one(srv, prob):
        t0 = time.time()
        doc = store.get(prob['doc_id'])
        pages = [p['text'] for p in doc['pages']]
        text = inputs.render_text(prob['prompt'], pages, prob['pdf_name'], None)
        rec = {'problem_id': prob['problem_id'], 'doc_id': prob['doc_id'], 'pdf_name': prob['pdf_name'],
               'engine': srv.name, 'body': super_body()}
        try:
            base = await srv.tokenize(text, affinity=prob['problem_id'])
            est = base + inputs.TOKEN_MARGIN
            rec['plan'] = {'mode': 'text', 'text_tokens': base, 'est_prompt_tokens': est}
            if est + CAPPED_OUTPUT_RESERVE > WINDOW:
                rec.update(status='input_too_long')
            else:
                body = dict(super_body(), max_tokens=WINDOW - est,
                            messages=[{'role': 'user', 'content': inputs.message_content([], text)}])
                res = await srv.chat(body, affinity=prob['problem_id'], hold=False)
                rec.update(answer=res['content'], reasoning=res['reasoning'], finish_reason=res['finish_reason'],
                           usage=res['usage'], max_tokens=body['max_tokens'], retries=res['attempts'])
                rec['status'] = ('ok' if res['content'].strip() else
                                 'length_no_answer' if res['finish_reason'] == 'length' else 'no_answer')
                u = res['usage']
                totals['tokens'] += (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0)
        except (RuntimeError, ValueError) as e:
            rec.update(status='infra_error', error=str(e)[:500])
        rec['seconds'] = round(time.time() - t0, 1)
        rec['at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        if rec['status'] != 'infra_error':
            out.write(rec)
            totals['traces'] += 1
        else:
            seen.discard(prob['problem_id'])   # retried on the next scan
        u = rec.get('usage') or {}
        print(f'[trace] {totals["traces"]} {prob["problem_id"]} {srv.name} {rec["status"]} '
              f'prompt={u.get("prompt_tokens")} completion={u.get("completion_tokens")} {rec["seconds"]}s '
              f'total={totals["tokens"] / 1e6:.1f}M', flush=True)

    async def snap():
        """Engine-wide vLLM counters, for the GPU-hour share (my tokens / the engine's tokens x engine-hours)."""
        rec = {'at': time.time(), 'mine': dict(totals), 'engines': {}}
        for name, srv in pool.servers.items():
            vals = {}
            for url, _ in srv.targets():
                root = url[:-3] if url.endswith('/v1') else url
                try:
                    r = await srv.http.get(f'{root}/metrics', timeout=30)
                    for line in r.text.splitlines():
                        for k in ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
                                  'vllm:num_requests_running', 'vllm:kv_cache_usage_perc',
                                  'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total'):
                            if line.startswith(k + '{') or line.startswith(k + ' '):
                                vals[k] = vals.get(k, 0.0) + float(line.rsplit(' ', 1)[1])
                except Exception:
                    pass
            rec['engines'][name] = vals
        with open(out_dir / 'engine_metrics.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')

    last_scan = last_snap = 0
    while True:
        if time.time() - last_snap > 300:
            last_snap = time.time()
            asyncio.create_task(snap())
        if time.time() - last_scan > 60 and not (stop_at and totals['tokens'] >= stop_at):
            await pool.sync()
            n = scan()
            last_scan = time.time()
            if n:
                print(f'[trace] +{n} problems queued ({len(queue)} waiting, {len(inflight)} in flight)', flush=True)
        if stop_at and totals['tokens'] >= stop_at:
            if queue:
                print(f'[trace] token target {stop_at} reached; draining', flush=True)
            queue.clear()
            if a.exit_when_idle and not Path(a.exit_when_idle).exists():
                Path(a.exit_when_idle).touch()      # tells the generator driver to stop
        while queue:
            srv = pool.pick()
            if srv is None:
                break
            prob = queue.pop(0)
            t = asyncio.create_task(srv_slot(srv, one, prob))
            inflight.add(t)
            t.add_done_callback(inflight.discard)
            await asyncio.sleep(0)
        target = stop_at and totals['tokens'] >= stop_at
        if not queue and not inflight and a.exit_when_idle and (target or time.time() - last_scan < 5):
            if Path(a.exit_when_idle).exists() or target:
                break
        await asyncio.sleep(1)
    out.close()


async def srv_slot(srv, fn, prob):
    """One session slot on `srv` for the whole problem (tokenize + chat)."""
    async with srv.sem:
        return await fn(srv, prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True, help='glob of problems jsonl files (status accepted)')
    ap.add_argument('--docs', required=True, help='docs index (DocStore)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens-total', type=float, default=None, help='stop dispatching at this many trace tokens')
    ap.add_argument('--exit-when-idle', default=None, help='a flag file: exit once it exists and the queue is empty')
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()

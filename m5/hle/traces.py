"""Super traces for accepted problems, with the eval's exact prompt, sampling and limits (aa_evals/hle, M3.6 config).

    python -m m5.hle.traces --problems 'P/problems*.jsonl' --out T/traces.jsonl [--target-tokens 330e6] [--follow]

- Messages: aa_evals.hle.prompts.build_messages (official HLE system prompt by answer type, text only, system role).
- Sampling: the M3.6 Super run's generation block (temperature 1.0, top_p 0.95, chat_template_kwargs
  enable_thinking / medium_effort false / truncate_history_thinking false), checked against each engine's registry.
- Limits: 131,072-token window: max_tokens = 131072 - (/tokenize(system + user text) + 64), as run.py does.
- Engines: m5-coord/alloc.json `super.serving`; per-engine in-flight cap `super.max_inflight_per_engine.hle`
  (re-read every 60 s). A request stays on its engine across retries. Servers restart: wait, re-read, resume.
- Streaming: with --follow, new accepted problems (status ok) are picked up as they land; the run stops admitting when
  kept tokens + an estimate for in-flight traces reach --target-tokens, or when <out>.STOP exists. Rows are appended;
  a restart skips problems already traced.
"""
import argparse
import asyncio
import collections
import glob
import json
import random
import time
from pathlib import Path

from aa_evals.hle import prompts

from .safeio import append_line
from .llm import ALLOC, Server, fields, metrics_snapshot

WINDOW = 131072
TEMPLATE_SLACK = 64
GEN = {'temperature': 1.0, 'top_p': 0.95,
       'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False, 'truncate_history_thinking': False}}


def alloc():
    try:
        a = json.loads(ALLOC.read_text())
        return list(a['super']['serving']), int(a['super']['max_inflight_per_engine']['hle'])
    except (OSError, ValueError, KeyError):
        return None, None


def spread_targets():
    """partition x category x answer-type shares from the fitted taxonomy (as in report.py)."""
    fit = json.loads((Path(__file__).resolve().parent / 'data' / 'fit.json').read_text())
    tgt = {}
    for part, cw in fit['category'].items():
        for c, w in cw.items():
            mc = fit['mc_share'][part][c]
            tgt[(part, c, 'multipleChoice')] = fit['partition'][part] * w * mc
            tgt[(part, c, 'exactMatch')] = fit['partition'][part] * w * (1 - mc)
    return tgt


def cell_of(p):
    s = p['spec']
    return (s['partition'], s['category'], s['answer_type'])


def item_of(p):
    return {'id': p['id'], 'question': p['question'], 'answer_type': p['spec']['answer_type'], 'image': ''}


class Driver:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.stop_file = Path(str(self.out) + '.STOP')
        self.done, self.kept_tokens, self.n_kept = set(), 0, 0
        if self.out.exists():
            for line in self.out.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                self.done.add(r['id'])
                if r.get('kept'):
                    self.kept_tokens += r['tokens_total']
                    self.n_kept += 1
        self.offsets = {}
        self.queue = []
        self.targets, self.cells = (spread_targets(), collections.Counter()) if a.balance else (None, None)
        if a.balance and self.out.exists():   # dispatched-and-finished counts survive a restart
            for line in self.out.read_text().splitlines():
                try:
                    self.cells[cell_of(json.loads(line))] += 1
                except (ValueError, KeyError):
                    pass
        self.seen = set(self.done)
        self.engines, self.cap = {}, 16
        self.inflight = {}
        self.dispatched = {}
        self.errors = collections.Counter()
        self.load, self.since = {}, {}   # server-wide running+waiting per engine (all clients), our dispatches since
        self.lock = asyncio.Lock()
        self.mean_tokens = 40000
        self.t0 = time.time()

    def refresh_engines(self):
        names, cap = alloc()
        if self.a.engines:
            names = self.a.engines.split(',')
        if self.a.cap:
            cap = min(self.a.cap, cap) if cap else self.a.cap
        if names:
            for n in names:
                if n not in self.engines:
                    self.engines[n] = Server(n, timeout=6 * 3600)
                    self.inflight.setdefault(n, 0)
            self.active = [n for n in names]
        if cap:
            self.cap = cap

    def problem_files(self):
        # metrics side files (problems.jsonl.metrics.jsonl) match 'problems*.jsonl' but are not problems and never get
        # a .DONE marker: the pilot driver waited on one forever
        return sorted(p for p in glob.glob(self.a.problems) if not p.endswith('.metrics.jsonl'))

    async def refresh_load(self):
        """Server-wide load per engine from /metrics (10-s timeout: a stalled engine counts as fully loaded)."""
        snaps = await asyncio.gather(*(metrics_snapshot(n, timeout=10) for n in self.active))
        for n, s in zip(self.active, snaps):
            ok = s and 'vllm:num_requests_running' in s
            self.load[n] = (s['vllm:num_requests_running'] + s.get('vllm:num_requests_waiting', 0)) if ok else None
            self.since[n] = 0

    def engine_cost(self, e):
        """Least-loaded engine first, by everyone's requests, not only ours (coordinator 11-12)."""
        if self.load.get(e) is None:
            return 1e9 + self.inflight[e]   # unknown or stalled: use only when nothing else is free
        return self.load[e] + self.since.get(e, 0)

    def poll_problems(self):
        for path in self.problem_files():
            off = self.offsets.get(path, 0)
            with open(path) as f:
                f.seek(off)
                while True:
                    line = f.readline()
                    if not line or not line.endswith('\n'):
                        break
                    off = f.tell()
                    try:
                        p = json.loads(line)
                    except ValueError:
                        continue
                    if p.get('status') == 'ok' and p['id'] not in self.seen and not p.get('exclude'):
                        self.seen.add(p['id'])
                        self.queue.append(p)
            self.offsets[path] = off

    def next_index(self):
        """FIFO, or with --balance the queued problem whose cell is furthest below its fitted share of dispatches.
        Never idles a slot: if every deficit cell is empty, the best available cell goes."""
        if not self.targets:
            return 0
        n = sum(self.cells.values()) + 1
        best, best_i = None, 0
        seen = set()
        for i, p in enumerate(self.queue):
            c = cell_of(p)
            if c in seen:
                continue
            seen.add(c)
            d = self.targets.get(c, 0.0) * n - self.cells[c]
            if best is None or d > best:
                best, best_i = d, i
        self.cells[cell_of(self.queue[best_i])] += 1
        return best_i

    def budget_left(self):
        if not self.a.target_tokens:
            return True
        n_in = sum(self.inflight.values())
        return self.kept_tokens + n_in * self.mean_tokens < self.a.target_tokens

    async def trace(self, eng, p):
        srv = self.engines[eng]
        item = item_of(p)
        messages = prompts.build_messages(item, policy='by_type', text_only=True, system_role='system')
        text = '\n'.join(m['content'] if isinstance(m['content'], str) else
                         '\n'.join(c.get('text', '') for c in m['content']) for m in messages)
        n_text = await srv.tokenize(text)
        if n_text is None:
            n_text = len(text) // 3 + 1
        body = {'messages': messages, 'stream': False, **GEN, 'max_tokens': WINDOW - (n_text + TEMPLATE_SLACK)}
        t0 = time.time()
        try:
            data, meta = await srv.chat(p['id'], body, attempts=60)
        except Exception as e:
            return {'id': p['id'], 'engine': eng, 'status': 'infra_error', 'error': str(e)[:500]}
        content, reasoning, finish, usage = fields(data)
        has_answer = bool(content and content.strip()) and finish == 'stop'
        pt, ct = usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0
        return {'id': p['id'], 'engine': eng, 'status': 'ok', 'messages': messages, 'content': content,
                'reasoning_content': reasoning, 'finish_reason': finish, 'stop_reason': data['choices'][0].get('stop_reason'),
                'prompt_tokens': pt, 'completion_tokens': ct, 'tokens_total': pt + ct, 'max_tokens': body['max_tokens'],
                'has_answer': has_answer, 'hit_context': finish == 'length', 'kept': has_answer,
                'latency_s': round(time.time() - t0), 'server': meta, 'spec': p['spec'],
                'problem_file_id': p['id'], 'finished_at': time.time()}

    async def worker_done(self, eng, task_p):
        row = await task_p
        async with self.lock:
            p_done = self.dispatched.pop(row['id'], None)
            self.inflight[eng] -= 1
            if row['status'] == 'ok':
                if row['kept']:
                    self.kept_tokens += row['tokens_total']
                    self.n_kept += 1
                n = self.n_kept or 1
                self.mean_tokens = 0.9 * self.mean_tokens + 0.1 * row['tokens_total'] if self.n_kept > 20 else self.mean_tokens
                await asyncio.to_thread(append_line, self.out, json.dumps(row, ensure_ascii=False))
            else:   # infra error: back in the queue (poll_problems only reads new lines, so it would be lost)
                p = p_done
                self.errors[row['id']] += 1
                if p is not None and self.errors[row['id']] <= 2:   # a request that always fails is not retried forever
                    self.queue.append(p)
                    if self.cells is not None:
                        self.cells[cell_of(p)] -= 1
                print(f'{row["id"]}: {row.get("error")}', flush=True)

    async def main(self):
        self.refresh_engines()
        await self.refresh_load()
        last_refresh, last_log = time.time(), 0
        tasks = set()
        while True:
            if time.time() - last_refresh > 60:
                self.refresh_engines()
                await self.refresh_load()
                last_refresh = time.time()
            self.poll_problems()
            stopping = self.stop_file.exists() or not self.budget_left()
            if not stopping:
                random.shuffle(self.queue) if self.a.shuffle else None
                while self.queue:
                    free = [e for e in self.active if self.inflight[e] < self.cap]
                    if not free:
                        break
                    eng = min(free, key=self.engine_cost)
                    p = self.queue.pop(self.next_index())
                    self.dispatched[p['id']] = p
                    self.inflight[eng] += 1
                    self.since[eng] = self.since.get(eng, 0) + 1
                    t = asyncio.create_task(self.worker_done(eng, self.trace(eng, p)))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
            if time.time() - last_log > 300:
                last_log = time.time()
                for n in self.active:
                    snap = await metrics_snapshot(n)
                    if snap:
                        snap['my_inflight'] = self.inflight.get(n, 0)
                        await asyncio.to_thread(append_line, str(self.out) + '.metrics.jsonl', json.dumps(snap))
                print(f'{time.strftime("%H:%M:%S")} kept {self.n_kept} traces {self.kept_tokens / 1e6:.1f}M tok; '
                      f'inflight {dict(self.inflight)} cap {self.cap}; queue {len(self.queue)}; '
                      f'{"STOPPING " if stopping else ""}{time.time() - self.t0:.0f}s', flush=True)
            if self.a.until:   # production: new problem files keep appearing until the generator is done
                sources_done = Path(self.a.until).exists()
            else:
                files = self.problem_files()
                sources_done = bool(files) and all(Path(p + '.DONE').exists() for p in files)
            if not tasks and (stopping or (not self.queue and (not self.a.follow or sources_done))):
                break
            if stopping and tasks and self.a.target_tokens and self.kept_tokens >= self.a.target_tokens and self.a.cut_tails:
                print(f'target reached: cutting {len(tasks)} in-flight traces', flush=True)
                for t in list(tasks):
                    t.cancel()
                break
            await asyncio.sleep(5)
        for s in self.engines.values():
            await s.close()
        print(f'DONE kept {self.n_kept} traces {self.kept_tokens / 1e6:.1f}M tokens', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True, help='glob of problem jsonl files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-tokens', type=float, default=0)
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--balance', action='store_true', help='dispatch toward the fitted partition x category x '
                    'answer-type spread (the oversupply lets the kept set match it)')
    ap.add_argument('--cut-tails', action='store_true')
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--until', help='with --follow: finish only once this file exists (and the queue is empty)')
    ap.add_argument('--engines', help='override alloc.json super.serving (comma list)')
    ap.add_argument('--cap', type=int, help='override the per-engine cap (never above alloc.json)')
    a = ap.parse_args()
    asyncio.run(Driver(a).main())


if __name__ == '__main__':
    main()

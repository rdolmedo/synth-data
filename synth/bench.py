"""Pi replay benchmark: drive one serving configuration with recorded Pi traffic.

Each simulated agent replays a recorded Pi trajectory turn by turn with the exact request
bodies the proxy sent (system prompt, tool schemas, full history, sampling, seed), waiting
the recorded tool time between a response and the next request (capped). The server generates
its own answers and they are discarded: the next request carries the recorded history, so
context growth and prefix reuse match production while output lengths are the served model's.
(Each turn re-prefills the recorded assistant message instead of reusing the generated one.)

A level `c:warmup:measure` runs c agents (agents are only added, so levels run in increasing
order); throughput comes from the servers' Prometheus counters over the measurement window,
latency and per-stream speed from the client.

Adaptive mode (`--adapt-hours`, issue #6): instead of fixed levels, synth.admission decides every tick how many
new agents may start, from the server's counters and the agents' cached contexts, as the campaign controller does.
Each agent replays one recorded trajectory from its first turn and ends, like a campaign attempt. With several
client processes (`--clients N --client-index k --coord-dir D`), client 0 runs admission on everyone's reports and
hands each client its share of the new agents.

Closed loop (`closed_loop`): from the turn an agent starts at, each request's history carries the agent's own
generated answers, built as Pi builds them, instead of the recorded ones (tool outputs stay recorded). The next
prompt then extends this engine's own prompt + output as in production, so a cache hit through the previous answer
is possible. Requests ask for token ids, and each turn records how far the new prompt matches the previous prompt +
output (`turn_stats`).

Corpus: `api-records/*.request.json` + `requests.jsonl` (start times) + `*.transport1.meta.json`
(completion times) of a finished campaign's attempts, e.g. the DeepSeek V4.1 campaign 20261016T171932Z. GLM-5.3-Flash's
attempts of that campaign logged a delta wire format instead (`index_wire`); their bodies are rebuilt in memory.
"""
import array
import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
import glob
import itertools
import uuid
import json
import os
from pathlib import Path
import random
import statistics
import time

import httpx

from .common import atomic, append_jsonl, read
from .probe import scrape, server_summary


def index_wire(run):
    """GLM-5.3-Flash's pi-submit attempts (campaign 20261016T171932Z, `<attempt>/raw/harbor/pi-cont16k`): its proxy
    logged each forwarded request to `wire-requests.jsonl` as a delta (keep the previous request's first n messages,
    append these; `tools` only when they change; every other field under `parameters`) and its SSE stream to
    `wire-responses/<request_index>.response`, whose mtime (whole seconds) is the completion time. Requests whose
    stream never reached [DONE] are applied to the history but not replayed. `requests` holds wire line numbers."""
    pi = Path(run) / 'raw' / 'harbor' / 'pi-cont16k'
    try:
        records = [json.loads(line) for line in (pi / 'wire-requests.jsonl').read_text().splitlines()]
    except (OSError, ValueError):
        return None
    sent = []
    for n, d in enumerate(records):
        path = pi / 'wire-responses' / f'{d["request_index"]:06d}.response'
        try:
            with open(path, 'rb') as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - 64))
                if b'[DONE]' in f.read():
                    sent.append((n, d['time'], os.stat(path).st_mtime + 0.5))
        except OSError:
            continue
    if len(sent) < 2:
        return None
    gaps = [max(0.0, sent[i + 1][1] - end) for i, (_, _, end) in enumerate(sent[:-1])] + [None]
    return {'run': str(run), 'wire': str(pi / 'wire-requests.jsonl'), 'requests': [n for n, _, _ in sent], 'gaps': gaps}


def wire_bodies(trajectory):
    """The full request bodies of an `index_wire` trajectory, in replay order (message dicts are shared)."""
    messages, tools, bodies = [], None, {}
    wanted = set(trajectory['requests'])
    with open(trajectory['wire']) as f:
        for n, line in enumerate(f):
            d = json.loads(line)
            delta = d['messages_delta']
            messages = messages[:delta['copy_previous_prefix_messages']] + delta['append_messages']
            if 'tools' in d:
                tools = d['tools']
            if n in wanted:
                bodies[n] = {**d['parameters'], 'messages': messages, **({'tools': tools} if tools else {})}
    return [bodies[n] for n in trajectory['requests']]


async def recorded_body(trajectory, k, loaded):
    """Turn k's recorded body as a fresh top-level dict; `loaded` (one list per trajectory) caches the rebuilt bodies
    of an `index_wire` trajectory."""
    if 'wire' not in trajectory:
        return json.loads(await asyncio.to_thread(Path(trajectory['requests'][k]).read_text))
    if not loaded:
        loaded.extend(await asyncio.to_thread(wire_bodies, trajectory))
    return dict(loaded[k])


def index_api(run):
    """Qwen3.8-27B's pi-cont16k attempts (campaign 20261015T215325Z, `<attempt>/raw/<trial>/pi-cont16k`): the proxy
    kept each request body as `api/<start ns>-<id>/request.json` and the response headers as `response-0.json`, whose
    `time` is the completion time. Only HTTP 200 requests are replayed, in start order."""
    pis = list(Path(run).glob('raw/*/pi-cont16k/api'))
    if not pis:
        return None
    sent = []
    for d in sorted(pis[0].iterdir(), key=lambda p: int(p.name.split('-')[0])):
        try:
            meta = json.loads((d / 'response-0.json').read_text())
        except (OSError, ValueError):
            continue
        if meta.get('status') == 200 and (d / 'request.json').exists():
            sent.append((str(d / 'request.json'), int(d.name.split('-')[0]) / 1e9, meta['time']))
    if len(sent) < 2:
        return None
    gaps = [max(0.0, sent[i + 1][1] - end) for i, (_, _, end) in enumerate(sent[:-1])] + [None]
    return {'run': str(run), 'requests': [r for r, _, _ in sent], 'gaps': gaps}


def index_run(run):
    """One recorded trajectory: request bodies in order and the tool gap after each."""
    pi = Path(run) / 'harbor' / 'run' / 'pi-cont16k'
    if not pi.exists() and (Path(run) / 'raw' / 'harbor' / 'pi-cont16k').exists():
        return index_wire(run)
    if not pi.exists() and (Path(run) / 'raw').is_dir() and not (Path(run) / 'raw' / 'harbor').exists():
        return index_api(run)
    try:
        starts = [json.loads(line)['t'] for line in (pi / 'requests.jsonl').read_text().splitlines()]
    except (OSError, ValueError, KeyError):
        return None
    requests = sorted(str(p) for p in (pi / 'api-records').glob('*.request.json'))
    if len(requests) != len(starts) or len(requests) < 2:
        return None
    gaps = []
    for i, path in enumerate(requests):
        try:
            end = json.loads(Path(path.replace('.request.json', '.transport1.meta.json')).read_text())['time']
            gaps.append(max(0.0, starts[i + 1] - end) if i + 1 < len(starts) else None)
        except (OSError, ValueError, KeyError):
            gaps.append(None)
    return {'run': str(run), 'requests': requests, 'gaps': gaps}


def build_corpus(pattern, size, path, seed=0):
    """Fixed random sample of recorded trajectories; cached at `path`. A `pattern` ending in .json is a prebuilt corpus
    in this cache format (e.g. task_synthesis/serve/oneshot_corpus.py: single-request trajectories): its first `size`
    trajectories."""
    if pattern.endswith('.json'):
        out = read(pattern)['trajectories'][:size]
        atomic(path, {'pattern': pattern, 'size': size, 'seed': seed, 'trajectories': out})
        return out
    cached = read(path)
    if cached and cached['pattern'] == pattern and cached['size'] == size and cached['seed'] == seed:
        return cached['trajectories']
    runs = sorted(glob.glob(pattern))
    random.Random(seed).shuffle(runs)
    out = []
    with ThreadPoolExecutor(32) as pool:
        for start in range(0, len(runs), 4 * size):
            out += [t for t in pool.map(index_run, runs[start:start + 4 * size]) if t]
            if len(out) >= size:
                break
    out = out[:size]
    atomic(path, {'pattern': pattern, 'size': size, 'seed': seed, 'trajectories': out})
    return out


def levels(spec):
    return [tuple(float(x) if i else int(x) for i, x in enumerate(part.split(':'))) for part in spec.split(',')]


def pi_message(reasoning, content, tool_calls, history='inline_think'):
    """The assistant message Pi appends to its history from a streamed answer, tool-call arguments re-serialized
    compactly. `inline_think` (Super's `inline_reasoning_history`, checked against 824 hero turns): the reasoning
    inline in the content. `reasoning_field` (GLM's recorded Pi turns): the streamed reasoning in `reasoning` and a
    null content when the answer has no text."""
    if history == 'reasoning_field':
        message = {'role': 'assistant', 'content': content or None}
        if reasoning:
            message['reasoning'] = reasoning
    else:
        message = {'role': 'assistant',
                   'content': ('<think>\n' + reasoning + '</think>' + content) if reasoning else content}
    calls = []
    for call in tool_calls:
        try:
            arguments = json.dumps(json.loads(call['arguments'] or '{}'), separators=(',', ':'), ensure_ascii=False)
        except ValueError:
            arguments = call['arguments']
        calls.append({'id': call['id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': arguments}})
    if calls:
        message['tool_calls'] = calls
    return message


def with_generated_history(body, generated, k):
    """Turn k's `body` with its j-th assistant message replaced by `generated[j]` (and the following tool results'
    ids remapped by position). Returns None unless the history holds exactly one assistant message per earlier turn
    (Pi compacted it, or a failed request was retried), so the recorded body is sent instead."""
    messages = body['messages']
    turns = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    if not generated or len(turns) != k:
        return None
    messages = list(messages)
    for j, i in enumerate(turns):
        if j not in generated:
            continue
        old_ids = [c.get('id') for c in messages[i].get('tool_calls') or []]
        new = generated[j]
        new_ids = dict(zip(old_ids, (c['id'] for c in new.get('tool_calls') or [])))
        messages[i] = new
        for k in range(i + 1, len(messages)):
            if messages[k].get('role') != 'tool':
                break
            if messages[k].get('tool_call_id') in new_ids:
                messages[k] = {**messages[k], 'tool_call_id': new_ids[messages[k]['tool_call_id']]}
    return {**body, 'messages': messages}


def common_prefix(a, b):
    """Length of the common prefix of two int arrays (chunked C-speed slice compares)."""
    n, i, step = min(len(a), len(b)), 0, 4096
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    i = min(i, n)
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Replay:
    def __init__(self, corpus, targets, served_name, *, seed=0, gap_cap=300, random_start=True, gap_scale=1.0,
                 closed_loop=False, history='inline_think', headers=None, overrides=None, stream=True):
        """targets: [{'base': .../v1, 'rank': int|None}] assigned round-robin to agents. `gap_scale` stretches every
        recorded tool gap (after the cap): replay agents skip environment setup and verification, so at the agent
        counts a campaign runs they sit idle far less than live agents do (2026-10-25: ~5 s vs ~24 s per turn on TMax)
        and the engine queues instead of caching idle contexts. `closed_loop`: see the module docstring. `headers`: sent
        with every request (e.g. X-Gate-Client); `overrides`: merged into every body (e.g. a shorter max_tokens).
        `stream=False` sends non-streamed requests, as task synthesis's round client does (no client TTFT then; the
        server's histograms still have it)."""
        self.corpus, self.targets, self.served = corpus, targets, served_name
        self.stream = stream
        self.seed, self.gap_cap, self.random_start, self.gap_scale = seed, gap_cap, random_start, gap_scale
        self.closed_loop, self.history = closed_loop, history
        self.headers, self.overrides = headers or {}, overrides or {}
        self.conversation_prefix = uuid.uuid4().hex
        self.turns = []
        self.turn_stats = []
        self.errors = []
        self.stop = asyncio.Event()

    async def call(self, client, target, body, collect=False, conversation=None):
        """One streamed request. `collect`: also return the answer as Pi would append it and the token ids.
        `conversation`: sent as X-Conversation-Id, so an admission gate tells the agents that replay the same recorded
        trajectory apart (their requests share their first messages, and their first model turn too when they start past
        it)."""
        headers = {**self.headers, **({} if target['rank'] is None else {'X-data-parallel-rank': str(target['rank'])}),
                   **({'X-Conversation-Id': conversation} if conversation else {})}
        t0 = time.time()
        if not body.get('stream', True):
            r = await client.post(target['base'] + '/chat/completions', json=body, headers=headers)
            if r.status_code != 200:
                raise RuntimeError(f'HTTP {r.status_code}: {r.content[:300]!r}')
            data = r.json()
            end, usage = time.time(), data.get('usage') or {}
            return {'t': end, 'latency': end - t0, 'ttft': None, 'finish': (data.get('choices') or [{}])[0].get('finish_reason'),
                    'completion': usage.get('completion_tokens', 0), 'prompt': usage.get('prompt_tokens', 0),
                    'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens')}
        ttft = usage = finish = None
        reasoning, content, calls, prompt_ids, output_ids = [], [], {}, None, array.array('i')
        async with client.stream('POST', target['base'] + '/chat/completions', json=body, headers=headers) as r:
            queue_ms = r.headers.get('X-Gate-Queue-Ms')   # time held by an admission gate (aa_evals/qwenserve)
            if r.status_code != 200:
                raise RuntimeError(f'HTTP {r.status_code}: {(await r.aread())[:300]!r}')
            async for line in r.aiter_lines():
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                chunk = json.loads(line[6:])
                if chunk.get('usage'):
                    usage = chunk['usage']
                if collect and chunk.get('prompt_token_ids') is not None:
                    prompt_ids = array.array('i', chunk['prompt_token_ids'])
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if ttft is None and (delta.get('content') or delta.get('reasoning') or
                                         delta.get('reasoning_content') or delta.get('tool_calls')):
                        ttft = time.time() - t0
                    finish = choice.get('finish_reason') or finish
                    if collect:
                        reasoning.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
                        content.append(delta.get('content') or '')
                        for t in delta.get('tool_calls') or []:
                            c = calls.setdefault(t.get('index', 0), {'id': None, 'name': '', 'arguments': ''})
                            c['id'] = t.get('id') or c['id']
                            c['name'] += (t.get('function') or {}).get('name') or ''
                            c['arguments'] += (t.get('function') or {}).get('arguments') or ''
                        output_ids.extend(choice.get('token_ids') or [])
        end = time.time()
        usage = usage or {}
        out = {'t': end, 'latency': end - t0, 'ttft': ttft, 'finish': finish,
               'completion': usage.get('completion_tokens', 0), 'prompt': usage.get('prompt_tokens', 0),
               'cached': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
               'queue_s': int(queue_ms) / 1000 if queue_ms and queue_ms.isdigit() else None}
        if collect:
            out['message'] = pi_message(''.join(reasoning), ''.join(content), [calls[k] for k in sorted(calls)],
                                        self.history)
            out['ids'] = (prompt_ids, output_ids)
        return out

    def stream_args(self):
        if self.stream or self.closed_loop:   # closed loop needs the streamed answer and token ids
            return {'stream': True, 'stream_options': {'include_usage': True}}
        return {'stream': False}

    def closed_turn(self, i, k, turn, generated, prev, closed):
        """Record turn k's answer for later histories and how far its prompt matches the previous prompt + output."""
        generated[k] = turn.pop('message')
        prompt_ids, output_ids = turn.pop('ids')
        stat = {'t': round(turn['t'], 1), 'agent': i, 'k': k, 'closed': closed, 'prompt': turn['prompt'],
                'cached': turn['cached'], 'completion': turn['completion'], 'finish': turn['finish']}
        if prev and prompt_ids is not None:
            before = prev['prompt'] + prev['output']
            lcp = common_prefix(prompt_ids, before)
            stat.update(prev_prompt=len(prev['prompt']), prev_output=len(prev['output']), lcp=lcp,
                        prev_finish=prev['finish'], new_next=list(prompt_ids[lcp:lcp + 3]),
                        old_next=list(before[lcp:lcp + 3]))
        self.turn_stats.append(stat)
        return {'prompt': prompt_ids, 'output': output_ids, 'finish': turn['finish']} if prompt_ids is not None else None

    async def agent(self, i, client):
        rng = random.Random(f'{self.seed}/{i}')
        target = self.targets[i % len(self.targets)]
        first = True
        for n in itertools.count():
            if self.stop.is_set():
                return
            trajectory = self.corpus[rng.randrange(len(self.corpus))]
            start = rng.randrange(len(trajectory['requests'])) if first and self.random_start else 0
            first = False
            conversation = f'{self.conversation_prefix}/{i}/{n}'
            generated, prev, loaded = {}, None, []
            for k in range(start, len(trajectory['requests'])):
                if self.stop.is_set():
                    return
                try:
                    body = await recorded_body(trajectory, k, loaded)
                except (OSError, ValueError, KeyError):   # unreadable or empty record: next trajectory
                    break
                body.update({'model': self.served, **self.stream_args(), **self.overrides})
                closed = None
                if self.closed_loop:
                    closed = with_generated_history(body, generated, k)
                    body = closed or body
                    body['return_token_ids'] = True
                try:
                    turn = await self.call(client, target, body, collect=self.closed_loop, conversation=conversation)
                    if self.closed_loop:
                        prev = self.closed_turn(i, k, turn, generated, prev, closed is not None)
                    self.turns.append(turn)
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    self.errors.append({'t': time.time(), 'agent': i, 'error': repr(exc)[:300]})
                    await asyncio.sleep(10)
                    break
                gap = trajectory['gaps'][k]
                if gap:
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=min(gap, self.gap_cap) * self.gap_scale)
                    except asyncio.TimeoutError:
                        pass

    def window(self, t0, t1):
        turns = [t for t in self.turns if t0 <= t['t'] < t1]
        decode = [t['completion'] / (t['latency'] - t['ttft']) for t in turns
                  if t['ttft'] is not None and t['completion'] > 1 and t['latency'] > t['ttft']]
        q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None
        return {'turns': len(turns), 'errors': sum(t0 <= e['t'] < t1 for e in self.errors),
                'client_gen_tok_s': round(sum(t['completion'] for t in turns) / (t1 - t0)),
                'completion_mean': round(statistics.mean(t['completion'] for t in turns)) if turns else None,
                'length_capped': sum(t['finish'] == 'length' for t in turns),
                'latency_p50_s': q([t['latency'] for t in turns], .5), 'latency_p90_s': q([t['latency'] for t in turns], .9),
                'ttft_p50_s': q([t['ttft'] for t in turns if t['ttft'] is not None], .5),
                'stream_tok_s_p50': round(q(decode, .5), 1) if decode else None,
                'stream_tok_s_p10': round(q(decode, .1), 1) if decode else None,
                **({'gate_queue_p50_s': q(gq, .5), 'gate_queue_p95_s': q(gq, .95)}
                   if (gq := [t['queue_s'] for t in turns if t.get('queue_s') is not None]) else {})}

    async def run(self, plan, metrics, *, kv_divisor=1, sample_seconds=15, healthy=None, out=None, tag=None,
                  stats_out=None):
        """plan: [(agents, warmup_s, measure_s)]; metrics: Prometheus bases (no /v1). `stats_out`: closed-loop turn
        stats (jsonl), written at the end."""
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        results, tasks = [], []
        async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30), limits=limits, trust_env=False) as client:
            try:
                for count, warmup, measure in plan:
                    while len(tasks) < count:
                        tasks.append(asyncio.create_task(self.agent(len(tasks), client)))
                    await self.sleep(warmup, healthy)
                    s0 = await asyncio.gather(*[asyncio.to_thread(scrape, m) for m in metrics])
                    t0, samples, cpu0 = time.time(), [], time.process_time()
                    while time.time() - t0 < measure:
                        await self.sleep(min(sample_seconds, measure - (time.time() - t0)), healthy)
                        snaps = [s for s in await asyncio.gather(*[asyncio.to_thread(scrape, m) for m in metrics]) if s]
                        if snaps:
                            samples.append({'kv': max(s['kv'] for s in snaps) / kv_divisor,
                                            'run': sum(s['run'] for s in snaps), 'wait': sum(s['wait'] for s in snaps)})
                    s1 = await asyncio.gather(*[asyncio.to_thread(scrape, m) for m in metrics])
                    t1 = time.time()
                    row = {'tag': tag, 'agents': count, 'warmup_s': warmup, 'measure_s': round(t1 - t0),
                           **server_summary(s0, s1, t1 - t0), **self.window(t0, t1),
                           'kv_max': round(max((s['kv'] for s in samples), default=0), 4),
                           'running_mean': round(statistics.mean(s['run'] for s in samples), 1) if samples else None,
                           'running_max': max((s['run'] for s in samples), default=None),
                           'waiting_mean': round(statistics.mean(s['wait'] for s in samples), 1) if samples else None,
                           # replay process CPU (all threads) per wall second: near 1 the client, not the server, binds
                           'client_cpu': round((time.process_time() - cpu0) / (t1 - t0), 2)}
                    results.append(row)
                    if out:
                        append_jsonl(out, row)
                    print(json.dumps(row), flush=True)
            finally:
                self.stop.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if stats_out and self.turn_stats:
                    Path(stats_out).write_text(''.join(json.dumps(s) + '\n' for s in self.turn_stats))
        return results

    async def attempt(self, i, client, target, contexts):
        """One campaign attempt: a recorded trajectory from its first turn; `contexts[i]` is its cached context."""
        trajectory = self.corpus[random.Random(f'{self.seed}/attempt/{i}').randrange(len(self.corpus))]
        state = contexts[i] = {'context_tokens': 0, 'inflight': False}
        loaded = []
        try:
            for k in range(len(trajectory['requests'])):
                if self.stop.is_set():
                    return
                try:
                    body = await recorded_body(trajectory, k, loaded)
                except (OSError, ValueError, KeyError):   # unreadable or empty record: end the attempt
                    return
                body.update(model=self.served, **self.stream_args())
                state['inflight'] = True
                try:
                    turn = await self.call(client, target, body, conversation=f'{self.conversation_prefix}/attempt/{i}')
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    self.errors.append({'t': time.time(), 'agent': i, 'error': repr(exc)[:300]})
                    return
                finally:
                    state['inflight'] = False
                self.turns.append(turn)
                state['context_tokens'] = turn['prompt'] + turn['completion']
                gap = trajectory['gaps'][k]
                if gap:
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=min(gap, self.gap_cap) * self.gap_scale)
                    except asyncio.TimeoutError:
                        pass
        finally:
            contexts.pop(i, None)

    async def adapt(self, metrics, policy, capacity, *, hours, block=256, kv_divisor=1, tick=15, healthy=None,
                    out=None, tag=None, clients=1, index=0, coord=None, offload=0):
        """Admission-driven replay: synth.admission.step sets how many new attempts start each tick."""
        from . import admission
        p = {**admission.DEFAULTS, **policy}
        coord = Path(coord) if coord else None
        b, tasks, contexts, mean, granted, started = {}, {}, {}, p['agent_tokens_prior'], 0, 0
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        end = time.time() + hours * 3600
        prev = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30), limits=limits, trust_env=False) as client:
            try:
                while time.time() < end:
                    await self.sleep(tick, healthy)
                    for i in [i for i, (t, _) in tasks.items() if t.done()]:
                        del tasks[i]
                    idle = sum(admission.idle_tokens(c, block) for c in contexts.values())
                    report = {'t': time.time(), 'agents': len(tasks), 'idle_tokens': idle}
                    if coord:
                        atomic(coord / f'report-{index}.json', report)
                    room = 0
                    if index == 0:
                        reports = [report] + [read(coord / f'report-{k}.json') or {} for k in range(1, clients)]
                        snaps = [x for x in await asyncio.gather(*[asyncio.to_thread(scrape, m) for m in metrics]) if x]
                        now = time.time()
                        if snaps and prev:
                            d = {k: sum(x[k] for x in snaps) - sum(x[k] for x in prev[1]) for k in
                                 ('gen', 'src_compute', 'src_gpu', 'src_cpu', 'preempt')}
                            agents = sum(r.get('agents', 0) for r in reports)
                            o = {'kv': max(x['kv'] for x in snaps) / kv_divisor, 'wait': sum(x['wait'] for x in snaps),
                                 'preempted': d['preempt'] > 0, 'stale': False, 'capacity': capacity, 'offload': offload,
                                 'own_idle_tokens': sum(r.get('idle_tokens', 0) for r in reports), 'own_agents': agents,
                                 'other_idle_tokens': 0, 'other_agents': 0,
                                 'reuse_expected': d['src_compute'] + d['src_gpu'] + d['src_cpu'],
                                 'reuse_cached': d['src_gpu'] + d['src_cpu'], 'gen_tok_s': d['gen'] / (now - prev[0])}
                            if agents >= 16:
                                mean = admission.update_mean(mean, admission.working_set(o) / agents, now - prev[0],
                                                             p['agent_tokens_halflife_seconds'])
                            m = max(mean, p['agent_tokens_prior'] * p.get('agent_tokens_floor', 1.0), 1024)
                            st = admission.step(b, policy, now, o, m)
                            room, granted = st['room'], granted + 1
                            if coord:
                                atomic(coord / 'grant.json', {'n': granted, 'per_client': -(-room // clients)})
                            row = {'tag': tag, 't': round(now), 'agents': agents, 'room': room, 'target': st['target'],
                                   'gen_tok_s': round(o['gen_tok_s']), 'kv': round(o['kv'], 4), 'wait': o['wait'],
                                   'run': sum(x['run'] for x in snaps), 'working_set': st['working_set_tokens'],
                                   'mean_tokens': round(m), 'reuse': st['reuse'], 'causes': st['causes'],
                                   'phase': (b.get('slope') or {}).get('phase'), 'errors': len(self.errors)}
                            if out:
                                append_jsonl(out, row)
                            room = -(-room // clients) if clients > 1 else room
                        prev = (now, snaps) if snaps else prev
                    else:
                        g = read(coord / 'grant.json') or {}
                        if g.get('n', 0) > granted:
                            granted, room = g['n'], g.get('per_client', 0)
                    for _ in range(room):
                        load = collections.Counter(r for _, r in tasks.values())
                        r = min(range(len(self.targets)), key=lambda r: load[r])   # least-loaded rank, as the controller
                        started += 1
                        i = (index << 32) + started
                        tasks[i] = (asyncio.create_task(self.attempt(i, client, self.targets[r], contexts)), r)
            finally:
                self.stop.set()
                for t, _ in tasks.values():
                    t.cancel()
                await asyncio.gather(*[t for t, _ in tasks.values()], return_exceptions=True)

    async def sleep(self, seconds, healthy):
        end = time.time() + seconds
        while time.time() < end:
            await asyncio.sleep(min(15, max(0, end - time.time())))
            if healthy and not await asyncio.to_thread(healthy):
                raise RuntimeError('serving engine exited during the benchmark')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, help='glob of recorded run dirs (…/attempts/*/run-*)')
    ap.add_argument('--corpus-size', type=int, default=600)
    ap.add_argument('--corpus-cache', required=True)
    ap.add_argument('--bases', required=True, help='comma-separated …/v1 bases')
    ap.add_argument('--dp-ranks', type=int, default=0, help='pin agents to X-data-parallel-rank 0..n-1')
    ap.add_argument('--served-name', required=True)
    ap.add_argument('--levels', default='64:300:300')
    ap.add_argument('--out', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--adapt-hours', type=float, default=0, help='admission-driven replay for this long (no --levels)')
    ap.add_argument('--policy', default='{}', help='admission policy JSON, e.g. {"target_search": "slope"}')
    ap.add_argument('--kv-capacity-tokens', type=int, default=0, help='KV tokens behind the server (all ranks)')
    ap.add_argument('--kv-block-tokens', type=int, default=256)
    ap.add_argument('--kv-offload-tokens', type=int, default=0, help='CPU KV tier tokens behind the server (admission scales its cap)')
    ap.add_argument('--clients', type=int, default=1)
    ap.add_argument('--client-index', type=int, default=0)
    ap.add_argument('--coord-dir', default='', help='shared directory for multi-client admission')
    ap.add_argument('--tag', default='')
    ap.add_argument('--sched', default='', choices=('', 'sched', 'hash', 'proto'),
                    help='#451: --adapt-hours with tbsched as the admission client (tbsched/bench_client.py): sched = '
                         'per-engine limits + credit routing; hash / proto = the 11-15 rules (pi_eval hash pin, + M5 autoscale)')
    ap.add_argument('--sched-gate', action='store_true', help='with --sched sched: the #477 collapse gate')
    ap.add_argument('--sched-limit', type=int, default=96, help='--sched hash/proto: the run limit over all engines')
    ap.add_argument('--sched-aimd', default='{}', help='--sched sched: tbsched.control.AIMD overrides (JSON; '
                         '{"policy": "guard"} for the throughput guard)')
    ap.add_argument('--sched-stationary', action='store_true',
                    help='--sched: the initial fill joins its trajectories at a random turn (a mix of context ages)')
    ap.add_argument('--gap-scale', type=float, default=1.0, help='stretch every recorded tool gap (Replay gap_scale)')
    a = ap.parse_args()
    if a.sched and (not a.adapt_hours or a.clients > 1 or a.coord_dir or a.policy != '{}'):
        raise SystemExit('--sched runs with --adapt-hours only, as one client, without --policy/--coord-dir')
    bases = a.bases.split(',')
    targets = ([{'base': bases[0], 'rank': r} for r in range(a.dp_ranks)] if a.dp_ranks
               else [{'base': b, 'rank': None} for b in bases])
    replay = Replay(build_corpus(a.corpus, a.corpus_size, a.corpus_cache), targets, a.served_name, seed=a.seed,
                    gap_scale=a.gap_scale)
    metrics = [b.removesuffix('/v1') for b in bases]
    if a.adapt_hours and a.sched:
        from tbsched import bench_client
        sched = bench_client.SchedClient(
            [{'metrics': metrics[0] + '/metrics', 'match': {'engine': str(r)}} for r in range(a.dp_ranks)] if a.dp_ranks
            else [{'metrics': m + '/metrics'} for m in metrics], a.sched, gate=a.sched_gate, limit=a.sched_limit,
            aimd=json.loads(a.sched_aimd), namespace=a.tag or a.sched)
        asyncio.run(bench_client.replay(replay, sched, hours=a.adapt_hours, out=a.out or None, tag=a.tag or a.sched,
                                        seed=a.seed, stationary=a.sched_stationary))
    elif a.adapt_hours:
        if a.coord_dir:
            Path(a.coord_dir).mkdir(parents=True, exist_ok=True)
        asyncio.run(replay.adapt(metrics, json.loads(a.policy), a.kv_capacity_tokens, hours=a.adapt_hours,
                                 block=a.kv_block_tokens, kv_divisor=max(1, a.dp_ranks), out=a.out or None,
                                 tag=a.tag or None, clients=a.clients, index=a.client_index, coord=a.coord_dir or None,
                                 offload=a.kv_offload_tokens))
    else:
        asyncio.run(replay.run(levels(a.levels), metrics, kv_divisor=max(1, a.dp_ranks), out=a.out or None))

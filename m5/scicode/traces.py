"""Super traces for generated SciCode problems, exactly as the eval generates them (aa_evals/scicode/run.py):
one user message per step from the official no-background template (prompts.build_prompt, with_background=False),
earlier steps carrying Super's own extracted code, no system prompt, the harness's per-model request settings
(presets.request_settings on the serving registry = the M3.6 run's config), max_tokens = 131072 - prompt - 256,
non-streaming requests through the eval's ChatClient. Nothing is executed or graded.

    python -m m5.scicode.traces --specs DIR/specs.jsonl --out OUT [--follow] [--token-target N]

Each chain (problem) is pinned to one Super engine (prefix cache: step k's prompt extends step k-1's); engines and
the per-engine session cap come from alloc.json (re-read every 60 s). Specs are streamed: with --follow the runner
picks up new lines of specs.jsonl until the generator's DONE file appears (or the token target is reached).

Out: OUT/steps.jsonl (one row per generated step: prompt, reasoning, content, finish_reason, usage, engine),
OUT/chains.jsonl (one row per finished chain: per-step summary and flags). Resumable from steps.jsonl.
At --token-target (prompt + completion tokens of finished steps), no new chain starts; running chains finish
(the tail is cut at export).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import time
from pathlib import Path

import httpx

from aa_evals.hle.client import ChatClient, InfraError, message_fields, resolve
from aa_evals.scicode import presets, prompts
from aa_evals.scicode.run import Budget, CTX_MARGIN

from .common import Alloc, JsonlWriter, read_jsonl, safe_write

CONTEXT = 131072
# the M3.6 Super run's request settings (/data/aai/aa_evals/runs/scicode/m36-super-vl-scicode-full/config.json)
M36_REQUEST = {'temperature': 1.0, 'top_p': 0.95,
               'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False,
                                        'truncate_history_thinking': False}}


def to_problem(spec: dict) -> dict:
    return {'problem_id': spec['problem_id'], 'required_dependencies': spec['required_dependencies'],
            'sub_steps': [{'step_number': f"{spec['problem_id']}.{i + 1}", **s} for i, s in enumerate(spec['sub_steps'])]}


class Engines:
    def __init__(self, alloc: Alloc):
        self.alloc = alloc
        self.clients: dict[str, ChatClient] = {}
        self.budgets: dict[str, Budget] = {}
        self.sessions = collections.Counter()
        self.gen = {}

    def live(self) -> list[str]:
        out = []
        for e in self.alloc.super_engines():
            try:
                ep = resolve(f'serving:{e}')
            except Exception:  # noqa: BLE001  (stale or missing registry: not live)
                continue
            if e not in self.clients:
                self.clients[e] = ChatClient(f'serving:{e}', timeout=4 * 3600)
                self.budgets[e] = Budget(CONTEXT, ep)
                gen = presets.request_settings(ep)
                if gen != M36_REQUEST:
                    raise SystemExit(f'{e}: request settings {gen} differ from the M3.6 eval {M36_REQUEST}')
                self.gen[e] = gen
            out.append(e)
        return out

    def free(self) -> str | None:
        cap = self.alloc.super_cap()
        live = [e for e in self.live() if self.sessions[e] < cap]
        if not live:
            return None
        return min(live, key=lambda e: (self.sessions[e], random.random()))


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps_w, chains_w = JsonlWriter(out / 'steps.jsonl'), JsonlWriter(out / 'chains.jsonl')
    done_chains = {r['problem_id'] for r in read_jsonl(out / 'chains.jsonl')}
    prior = collections.defaultdict(dict)
    tokens = 0
    for r in read_jsonl(out / 'steps.jsonl'):
        prior[r['problem_id']][r['k']] = r
    for pid, ss in prior.items():
        if pid in done_chains:
            tokens += sum((r['usage'].get('prompt_tokens') or 0) + (r['usage'].get('completion_tokens') or 0)
                          for r in ss.values())
    alloc = Alloc()
    eng = Engines(alloc)
    http = httpx.AsyncClient(trust_env=False, timeout=120)
    started, running = set(done_chains), set()
    stats = collections.Counter()
    t0 = time.time()

    async def chain(spec: dict, e: str):
        nonlocal tokens
        prob = to_problem(spec)
        pid = prob['problem_id']
        codes, recs = [], []
        try:
            for k, s in enumerate(prob['sub_steps']):
                r = prior[pid].get(k)
                if r is None:
                    prompt = prompts.build_prompt(prob, k, codes, with_background=False)
                    while True:
                        messages = [{'role': 'user', 'content': prompt}]
                        budget, gen = eng.budgets[e], eng.gen[e]
                        if eng.clients[e].endpoint is not None:   # follow a restarted engine's URL
                            if budget.endpoint is not eng.clients[e].endpoint:
                                budget.endpoint, budget.mode = eng.clients[e].endpoint, None
                        try:
                            max_tokens, n_prompt = await budget.max_tokens(http, messages, gen)
                            if max_tokens < 1024:
                                content = reasoning = None
                                finish, usage, status = 'prompt_over_context', {}, 'model_error'
                                lat = 0
                            else:
                                tq = time.time()
                                resp, _ = await eng.clients[e].chat({'messages': messages, **gen, 'max_tokens': max_tokens})
                                content, reasoning, finish, usage = message_fields(resp)
                                status, lat = 'ok', round(time.time() - tq, 1)
                            break
                        except InfraError as err:   # engine gone: move the chain to another engine
                            stats['infra_retry'] += 1
                            print(f'{pid} step {k + 1} on {e}: {str(err)[:200]}', flush=True)
                            eng.sessions[e] -= 1
                            while (ne := eng.free()) is None:
                                await asyncio.sleep(60)
                            e = ne
                            eng.sessions[e] += 1
                    r = {'problem_id': pid, 'k': k, 'step_number': s['step_number'], 'engine': e, 'status': status,
                         'prompt': prompt, 'reasoning': reasoning, 'content': content, 'finish_reason': finish,
                         'usage': usage, 'max_tokens': max_tokens, 'prompt_tokens_est': n_prompt, 'latency_s': lat,
                         'prompt_sha': prompts.sha(prompt), 'at': time.time()}
                    await steps_w.write(r)
                code = prompts.extract_code(r['content'])
                codes.append(code)
                recs.append(r)
            u = [x['usage'] or {} for x in recs]
            row = {'problem_id': pid, 'n_steps': len(recs), 'engine': e,
                   'prompt_tokens': sum(x.get('prompt_tokens') or 0 for x in u),
                   'completion_tokens': sum(x.get('completion_tokens') or 0 for x in u),
                   'finish': [x['finish_reason'] for x in recs],
                   'no_code_block': [bool(x['content']) and '```' not in x['content'] for x in recs],
                   'empty_content': [not x['content'] for x in recs],
                   'cutoff_any': any(x['finish_reason'] in ('length', 'prompt_over_context') for x in recs),
                   'submitted_all': all(x['content'] and '```' in x['content'] for x in recs), 'at': time.time()}
            tokens += row['prompt_tokens'] + row['completion_tokens']
            await chains_w.write(row)
            stats['chains'] += 1
        finally:
            eng.sessions[e] -= 1
            running.discard(pid)

    specs_path = Path(a.specs)
    done_file = specs_path.parent / 'DONE'
    seen_lines = 0
    queue: list[dict] = []
    tasks = []
    last = 0
    while True:
        if (Path(a.out) / 'STOP').exists():
            break
        new = read_jsonl(specs_path)
        for s in new[seen_lines:]:
            if s['problem_id'] not in started and (not a.limit or len(started) + len(queue) < a.limit + len(done_chains)):
                queue.append(s)
        seen_lines = len(new)
        target_hit = a.token_target and tokens >= a.token_target
        if target_hit and not (specs_path.parent / 'STOP').exists():
            safe_write(specs_path.parent / 'STOP', f'token target {a.token_target} reached\n')   # stops the generator
        while queue and not target_hit:
            e = eng.free()
            if e is None:
                break
            s = queue.pop(0)
            started.add(s['problem_id'])
            running.add(s['problem_id'])
            eng.sessions[e] += 1
            tasks.append(asyncio.create_task(chain(s, e)))
        if time.time() - last > 300:
            print(f'{time.strftime("%H:%M:%S")} chains done {stats["chains"]} running {len(running)} queued {len(queue)} '
                  f'tokens {tokens / 1e6:.2f}M sessions {dict(eng.sessions)} {time.time() - t0:.0f}s', flush=True)
            last = time.time()
        finished_input = (not a.follow or done_file.exists()) and not queue
        if (finished_input or target_hit) and not running:
            break
        await asyncio.sleep(15)
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f'done: {dict(stats)} tokens {tokens}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--token-target', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='at most N chains (pilot)')
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == '__main__':
    main()

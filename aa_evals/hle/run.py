"""HLE-Diamond as Artificial Analysis runs it: official HLE prompts, no tools, LLM equality-check judge, pass@1.

    python -m aa_evals.hle.run --model serving:aa-super --run-id smoke-super --limit 30 \
        --judge serving:aa-super
    python -m aa_evals.hle.run --model serving:<name> --run-id <id>    # judge: serving:aa-judge (self-hosted)

Outputs go to /data/aai/aa_evals/runs/hle/<run_id>/ (config.json, items.jsonl, summary.json,
transcripts/). Rerunning the same --run-id resumes: finished items are kept, infra errors and missing
judgments are retried. See aa_evals/hle/README.md for the method and every deviation from AA.
"""
import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from aa_evals.common import aai_open, protocol
from aa_evals.judges import pins
from aa_evals.tracejudge import fold_summary

from . import data, prompts, scoring
from .client import ChatClient, Endpoint, InfraError, Retryable, message_fields, resolve_now

DEFAULT_RUNS_ROOT = Path('/data/aai/aa_evals/runs/hle')
FALLBACK_RUNS_ROOT = Path('/data/aai/aa_evals/runs_fallback/hle')   # runs made 11-02..11-13 (moved off /home)
# AA_HLE_RUNS_ROOT moves new runs (submit.py passes it on to the job); runs are looked up in every root.
RUNS_ROOT = Path(os.environ.get('AA_HLE_RUNS_ROOT') or DEFAULT_RUNS_ROOT)


def run_path(run_id):
    """An existing run's directory in any runs root, else where a new run goes (RUNS_ROOT)."""
    for root in dict.fromkeys([RUNS_ROOT, DEFAULT_RUNS_ROOT, FALLBACK_RUNS_ROOT]):
        if (root / run_id / 'config.json').exists():
            return root / run_id
    return RUNS_ROOT / run_id
# User decision 2026-11-02 (#33): no closed-API judge. The judge is an open model we self-host, registered as
# serving:aa-judge (which model is chosen after a judge test). AA's own judge, openai:gpt-5.6-luna, stays
# available behind --allow-external-api.
DEFAULT_JUDGE = 'serving:aa-judge'
# The AA leaderboard slug comes from the served name and the effort (aa_evals/common/protocol.py AA_MODELS), so a
# Medium run can't pick up the Xhigh entry (#33 audit 1, finding 7); --aa-slug overrides.
# Under a context cap, max_tokens = cap - (the prompt's text tokens by the endpoint's /tokenize + slack for the chat
# template + an allowance per image). The allowance is an upper bound on HLE's image prompts (max 4.8k tokens for the
# whole prompt on Qwen3.8 at 4 MP, 4.0k on Super), so the cap is never exceeded; summary.json counts any row over it.
TEMPLATE_SLACK = 64
IMAGE_ALLOWANCE = 5120
BENCH = 'hle'


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def git_state():
    repo = Path(__file__).resolve().parents[2]
    def git(*a):
        r = subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    # A frozen copy made by submit.py first: it has no .git of its own, and under a code root inside another
    # repo (the Lustre fallback sits under the home repo) `git rev-parse` would report that repo's HEAD.
    if (repo / 'FROZEN_SHA').exists():
        return {'sha': (repo / 'FROZEN_SHA').read_text().strip(), 'dirty': False, 'frozen_copy': str(repo)}
    sha = git('rev-parse', 'HEAD')
    return {'sha': sha, 'dirty': bool(git('status', '--porcelain', '--', 'aa_evals')), 'repo': str(repo)}


def stratum(item):
    return (item['partition'], bool(item['image']), item['answer_type'])


def select(items, limit=None, ids=None, stratify=False):
    """A deterministic subset: the given ids, else the first `limit` items ordered by sha256(id), which mixes
    partitions, modalities and answer types without depending on the dataset's row order.

    With `stratify`, the `limit` is split across partition x modality x answer type in proportion to the
    full set (largest remainder, at least 1 per stratum), and each stratum takes its first items by sha256.
    The subset's unweighted mean then estimates the full-set score with lower variance."""
    if ids:
        wanted = set(ids)
        chosen = [it for it in items if it['id'] in wanted]
        if len(chosen) != len(wanted):
            raise SystemExit(f'{len(wanted) - len(chosen)} requested ids are not in the dataset')
        return chosen
    ordered = sorted(items, key=lambda it: hashlib.sha256(it['id'].encode()).hexdigest())
    if not limit or limit >= len(ordered):
        return ordered
    if not stratify:
        return ordered[:limit]
    groups = {}
    for it in ordered:
        groups.setdefault(stratum(it), []).append(it)
    keys = sorted(groups)
    if limit < len(keys):
        raise SystemExit(f'--stratify needs --limit >= {len(keys)} (one per stratum)')
    quota = {k: max(1, int(limit * len(groups[k]) / len(ordered))) for k in keys}
    rema = sorted(keys, key=lambda k: -(limit * len(groups[k]) / len(ordered) - quota[k]))
    i = 0
    while sum(quota.values()) < limit:
        k = rema[i % len(rema)]
        if quota[k] < len(groups[k]):
            quota[k] += 1
        i += 1
    while sum(quota.values()) > limit:   # minimums pushed the total over: trim the largest strata
        quota[max(keys, key=lambda k: quota[k])] -= 1
    chosen = {it['id'] for k in keys for it in groups[k][:quota[k]]}
    return [it for it in ordered if it['id'] in chosen]


def endpoint_context(endpoint):
    """The served context (vLLM /v1/models max_model_len), or None when the endpoint doesn't say."""
    if endpoint.kind == 'openai' or not endpoint.urls:
        return None
    try:
        r = httpx.get(endpoint.urls[0] + '/models', timeout=30, trust_env=False)
        for m in r.json().get('data') or []:
            if m.get('id') == endpoint.served_name and m.get('max_model_len'):
                return int(m['max_model_len'])
    except (httpx.HTTPError, ValueError):
        pass
    return None


def generation_settings(a, endpoint):
    """The sampling body sent with every model request: the serving registry's request overrides (the lab's
    recommended sampling, as the teacher spec pins them), then the CLI's explicit values."""
    gen = {}
    if a.endpoint_defaults and endpoint.kind == 'serving':
        gen.update(json.loads(json.dumps((endpoint.request or {}).get('overrides') or {})))
    if a.reasoning_effort is not None:
        # wherever the request carries an effort: a top-level value alone leaves Qwen3.8's template kwarg (xhigh)
        try:
            gen = protocol.apply_effort(gen, a.reasoning_effort)
        except ValueError as e:
            if gen:
                raise SystemExit(f'--reasoning-effort {a.reasoning_effort}: {e}')
            gen['reasoning_effort'] = a.reasoning_effort   # a url:/openai: model with no request profile
    for key in ('temperature', 'top_p'):
        if getattr(a, key) is not None:
            gen[key] = getattr(a, key)
    if a.max_tokens:
        gen['max_tokens'] = a.max_tokens
    if a.chat_template_kwargs:
        gen['chat_template_kwargs'] = {**gen.get('chat_template_kwargs', {}), **json.loads(a.chat_template_kwargs)}
    if a.extra_body:
        gen.update(json.loads(a.extra_body))
    if endpoint.kind != 'openai':
        gen.setdefault('temperature', 0.6)   # AA's default for reasoning models when the lab gives none
    return gen


def judge_settings(a, endpoint):
    s = {'response_format': prompts.JUDGE_RESPONSE_FORMAT}
    if endpoint.kind == 'openai':
        s['max_completion_tokens'] = a.judge_max_tokens
        s['reasoning_effort'] = a.judge_reasoning_effort or 'medium'   # AA's judge setting
    else:
        if endpoint.kind == 'serving':
            s.update((endpoint.request or {}).get('overrides') or {})
        if a.judge_reasoning_effort:
            # Served Qwen-style templates read the effort from chat_template_kwargs, which the registry pins
            # (xhigh), so a top-level reasoning_effort alone changes nothing: set both.
            s['reasoning_effort'] = a.judge_reasoning_effort
            if 'reasoning_effort' in (s.get('chat_template_kwargs') or {}):
                s['chat_template_kwargs'] = {**s['chat_template_kwargs'], 'reasoning_effort': a.judge_reasoning_effort}
        s['max_tokens'] = a.judge_max_tokens
    if a.judge_extra_body:
        s.update(json.loads(a.judge_extra_body))
    return s


class AdaptiveLimiter:
    """At most `limit` model requests in flight. With a vLLM /metrics URL, `limit` follows the server: it drops to
    what vLLM actually runs when vLLM queues requests (waiting > 0) or preempts, and rises by 4 while the KV cache
    is below `kv_target` and the client has more work. In-flight requests are never cancelled, so results don't
    change, only the scheduling. Long single-turn generations grow their KV during decode, so a fixed cap that
    fits early (short answers) thrashes later (long tails)."""

    def __init__(self, max_limit, metrics_url=None, start=None, min_limit=8, kv_target=0.85, period=30):
        self.max, self.min = max_limit, min(min_limit, max_limit)
        self.limit = max_limit if metrics_url is None else max(self.min, min(max_limit, start or max_limit))
        self.metrics_url, self.kv_target, self.period = metrics_url, kv_target, period
        self.inflight, self.last_preempt = 0, None
        self.cond = asyncio.Condition()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.inflight < self.limit)
            self.inflight += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    def adjust(self, waiting, kv, preemptions):
        new = self.limit
        preempted = None not in (self.last_preempt, preemptions) and preemptions > self.last_preempt
        if self.inflight > self.limit:
            pass   # still draining to an earlier cut: vLLM's queue lags the client, so don't cut again yet
        elif (waiting or 0) > 0 or preempted:
            new = max(self.min, self.inflight - max(2, int(waiting or 0)))
        elif kv is not None and kv < self.kv_target and self.inflight >= self.limit:
            new = min(self.max, self.limit + 4)
        self.last_preempt = preemptions
        if new != self.limit:
            print(f'[{time.strftime("%H:%M:%S")}] in-flight limit {self.limit} -> {new} (vLLM waiting {waiting}, '
                  f'kv {kv}, preemptions {preemptions})', flush=True)
        self.limit = new

    @staticmethod
    def parse(text):
        out = {}
        for line in text.splitlines():
            for key in ('num_requests_waiting', 'kv_cache_usage_perc', 'num_preemptions_total'):
                if (line.startswith(f'vllm:{key}{{') or line.startswith(f'vllm:{key} ')) and key not in out:
                    out[key] = float(line.rsplit(' ', 1)[1])
        return out

    async def control(self):
        async with httpx.AsyncClient(trust_env=False, timeout=10) as http:
            while True:
                await asyncio.sleep(self.period)
                try:
                    m = self.parse((await http.get(self.metrics_url)).text)
                except Exception:   # a metrics hiccup leaves the limit as it is
                    continue
                async with self.cond:
                    self.adjust(m.get('num_requests_waiting'), m.get('kv_cache_usage_perc'),
                                m.get('num_preemptions_total'))
                    self.cond.notify_all()


class Runner:
    def __init__(self, a, items, model, judge, gen, jset, run_dir):
        self.a, self.items, self.model, self.judge = a, {it['id']: it for it in items}, model, judge
        self.gen, self.jset, self.run_dir = gen, jset, run_dir
        # which judge and settings made a judgment; --rejudge re-grades stored judgments made otherwise
        self.judge_key = hashlib.sha256(json.dumps([getattr(a, 'judge', None), jset], sort_keys=True)
                                        .encode()).hexdigest()[:16]
        self.tdir = run_dir / 'transcripts'
        self.window = getattr(a, 'window', None)   # the run's context cap; None: the endpoint's own context
        self.msem = AdaptiveLimiter(a.concurrency, getattr(a, 'metrics_url', None),
                                    start=getattr(a, 'start_concurrency', None),
                                    min_limit=getattr(a, 'min_concurrency', 8) or 8)
        self.jsem = asyncio.Semaphore(a.judge_concurrency)
        # #344 dispatcher (--dispatch DIR): the model-request allowance comes from the run's dispatcher (shared
        # in-flight budget, longest predicted first) instead of --concurrency / the adaptive limiter
        self.lane, self.pred = None, None
        if getattr(a, 'dispatch', None) and getattr(a, 'stage', 'all') != 'grade':
            from aa_evals.common.dispatch.lane import open_dispatch
            self.lane, self.pred, _ = open_dispatch(a.dispatch, 'hle', run_dir)
        self.request_cap = getattr(a, 'request_cap_sec', None)
        self.counts = {'done': 0, 'ok': 0, 'model_error': 0, 'infra_error': 0, 'pending': 0, 'correct': 0}
        self.total = 0
        self.deadline = None
        # Re-judging another run (--generations-from) never calls the model. An item the source run lacks is
        # carried over as the source scored it: cut off (0) or an infra error (resume the source run).
        self.source = getattr(a, 'generations_from', None) or (a.run_id if getattr(a, 'stage', 'all') == 'grade' else None)
        self.source_cutoff = set()
        if self.source:
            src_items = run_path(self.source) / 'items.jsonl'
            if src_items.exists():
                for line in src_items.read_text().splitlines():
                    r = json.loads(line)
                    if r.get('finish_reason') == 'cutoff':
                        self.source_cutoff.add((r['item_id'], r['repeat']))

    def path(self, item_id, repeat):
        return self.tdir / f'{item_id}.r{repeat}.json'

    async def max_tokens_for(self, item, messages):
        """Under the context cap: the cap minus an upper bound on this prompt's tokens."""
        text = '\n'.join(m['content'] if isinstance(m['content'], str) else
                         '\n'.join(c.get('text', '') for c in m['content']) for m in messages)
        n_text = await self.model.count_tokens(text)
        n_img = sum(1 for m in messages if not isinstance(m['content'], str)
                    for c in m['content'] if c.get('type') == 'image_url')
        bound = n_text + TEMPLATE_SLACK + IMAGE_ALLOWANCE * n_img
        return self.window - bound, bound

    async def generate(self, item, repeat=0):
        if self.lane:   # the slot is taken before tokenization, so every item waits in the lane in priority order
            async with self.lane.aslot(f"{item['id']}.r{repeat}", item['id'], self.pred(item['id'])):
                return await self._generate(item, contextlib.nullcontext())
        return await self._generate(item, self.msem)

    async def _generate(self, item, slot):
        messages = prompts.build_messages(item, policy=self.a.system_prompt, text_only=self.a.text_only,
                                          system_role=self.a.system_role)
        t0 = time.time()
        body = {'messages': messages, 'stream': False, **self.gen}
        prompt_bound = None
        if self.window and 'max_tokens' not in self.gen:
            body['max_tokens'], prompt_bound = await self.max_tokens_for(item, messages)
        from tb_agents import response_budget as response_policy
        from tb_agents.response_budget_single import run_async_chat
        budget = response_policy.with_benchmark(response_policy.settings(), 'hle')   # capped protocol (off unless set)
        tries = []

        async def budgeted(b):
            d, n = await self.model.chat(b)
            tries.append(n)
            return d

        async def request():
            if budget:
                d = await run_async_chat(budgeted, body, budget)
                return d, (sum(tries) if all(isinstance(t, int) for t in tries) else tries)
            return await self.model.chat(body)
        async with slot:
            try:
                if self.request_cap:
                    # #344 2 h cap: the whole model call (all its retries) is bounded; at the cap the request is
                    # closed (vLLM aborts it) and the item is the model's: no answer, scored 0, kept on resume
                    data_, attempts = await asyncio.wait_for(request(), self.request_cap)
                else:
                    data_, attempts = await request()
            except InfraError as e:
                return {'status': 'infra_error', 'error': str(e), 'attempts': e.attempts, 'latency_s': time.time() - t0}
            except TimeoutError:
                if not self.request_cap:
                    raise
                return {'status': 'model_error', 'error': f'request cap: no answer within {self.request_cap:g} s',
                        'finish_reason': 'request_cap', 'latency_s': time.time() - t0,
                        'max_tokens': body.get('max_tokens'), 'prompt_bound': prompt_bound}
        content, reasoning, finish, usage = message_fields(data_)
        out = {'content': content, 'reasoning': reasoning, 'finish_reason': finish, 'usage': usage,
               'attempts': attempts, 'latency_s': time.time() - t0, 'response_id': data_.get('id'),
               'max_tokens': body.get('max_tokens'), 'prompt_bound': prompt_bound,
               # vLLM: the stop token id or string that ended the turn (None for EOS of the tokenizer / length)
               'stop_reason': data_['choices'][0].get('stop_reason'),
               'response_budget': data_.get('_response_budget')}
        if content is None or not content.strip():
            # No final answer: the model spent its budget reasoning (finish_reason length) or closed its turn
            # without one. AA's grader sees an empty response, so this scores 0.
            out['status'] = 'model_error'
            out['error'] = f'no final answer (finish_reason={finish})'
        else:
            out['status'] = 'ok'
        return out

    async def generate_by_deadline(self, item, repeat=0):
        """generate(), abandoned at the run's --cutoff-min (the no-tails rule): an unfinished item, queued or in
        flight, scores 0 as a cut-off model_error. Cancelling closes the HTTP request, so vLLM aborts it."""
        if self.deadline is None:
            return await self.generate(item, repeat)
        t0 = time.time()
        try:
            return await asyncio.wait_for(self.generate(item, repeat), max(0.0, self.deadline - t0))
        except TimeoutError:
            return {'status': 'cutoff', 'error': f'unfinished at the {self.a.cutoff_min:g}-min cutoff',
                    'finish_reason': 'cutoff', 'latency_s': time.time() - t0}

    async def grade(self, item, response):
        t0 = time.time()
        attempts, calls, last = [], [], None
        async with self.jsem:
            for k in range(self.a.judge_format_attempts):
                body = {'messages': prompts.judge_messages(item, response), **self.jset}
                # One entry per ChatClient.chat invocation, NOT per physical HTTP/model call.
                # Transport retries and gate summaries remain solely in `attempts`; this range
                # references those rows and excludes local format-error markers.
                call = {'invocation_index': k + 1, 'status': 'unknown',
                        'request_body_sha256': hashlib.sha256(json.dumps(
                            body, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                        'request_body_scope': 'ChatClient.chat argument, before client adds model',
                        'request_max_tokens': body.get('max_tokens'),
                        'request_response_format': copy.deepcopy(body.get('response_format')),
                        'judge_key': self.judge_key, 'raw_response': None, 'response_id': None,
                        'content': None, 'reasoning': None, 'finish_reason': None, 'usage': None,
                        'transport_attempts_range': [len(attempts), len(attempts)]}
                calls.append(call)
                try:
                    data_, att = await self.judge.chat(body)
                except InfraError as e:
                    attempts += e.attempts
                    call.update(status='infra_error', error=str(e))
                    call['transport_attempts_range'][1] = len(attempts)
                    return {'status': 'infra_error', 'error': str(e), 'attempts': attempts, 'calls': calls}
                except Exception as e:
                    # Caller/client failures do not establish that any HTTP request or paid
                    # completion occurred. Preserve prior responses and leave usage unknown.
                    error = f'{type(e).__name__}: {e}'
                    call.update(status='caller_error', error=error)
                    return {'status': 'infra_error', 'error': error, 'attempts': attempts, 'calls': calls}
                attempts += att
                call['transport_attempts_range'][1] = len(attempts)
                call.update(status='returned', raw_response=copy.deepcopy(data_),
                            response_id=data_.get('id'), usage=copy.deepcopy(data_.get('usage')))
                try:
                    content, reasoning, finish, usage = message_fields(data_)
                    call.update(content=content, reasoning=reasoning, finish_reason=finish)
                    j = prompts.parse_judgment(content)
                except ValueError as e:   # off-schema judge output: retried, never scored as the model's fault
                    last = f'{e} (finish_reason={call["finish_reason"]})'
                    call.update(status='off_schema', error=last)
                    attempts.append({'format_error': last})
                    continue
                except Exception as e:
                    error = f'{type(e).__name__}: {e}'
                    call.update(status='caller_error', error=error)
                    return {'status': 'infra_error', 'error': error, 'attempts': attempts, 'calls': calls}
                call['status'] = 'ok'
                return {'status': 'ok', **j, 'finish_reason': finish, 'usage': usage, 'attempts': attempts,
                        'latency_s': time.time() - t0, 'judge_key': self.judge_key, 'calls': calls}
        return {'status': 'infra_error', 'error': f'judge output off-schema {self.a.judge_format_attempts}x: {last}',
                'attempts': attempts, 'calls': calls}

    async def one(self, item_id, repeat):
        item = self.items[item_id]
        path = self.path(item_id, repeat)
        rec = json.loads(path.read_text()) if path.exists() else None
        if rec is None or rec['generation']['status'] == 'infra_error':
            if self.source:
                # not persisted, so rerunning after the source run finishes seeds the generation
                rec = {'item_id': item_id, 'repeat': repeat, 'judgment': None, 'generation': (
                    {'status': 'cutoff', 'error': f'cut off in source run {self.source}'}
                    if (item_id, repeat) in self.source_cutoff else
                    {'status': 'infra_error', 'error': f'no finished generation in source run {self.source}'})}
            else:
                gen = await self.generate_by_deadline(item, repeat)
                rec = {'item_id': item_id, 'repeat': repeat, 'generation': gen, 'judgment': None}
                if gen['status'] != 'cutoff':   # a cut-off item is regenerated on resume
                    atomic_json(path, rec)
        gen = rec['generation']
        jud = rec.get('judgment')
        if getattr(self.a, 'stage', 'all') != 'generate' and gen['status'] == 'ok' and (
                jud is None or jud['status'] == 'infra_error' or
                (jud.get('status') == 'ok' and not prompts.valid_confidence(jud.get('confidence'))) or
                (getattr(self.a, 'rejudge', False) and jud.get('judge_key') != self.judge_key)):
            rec['judgment'] = await self.grade(item, gen['content'])
            atomic_json(path, rec)
        row = self.row(item, rec)
        self.counts['done'] += 1
        self.counts[row['status']] += 1
        self.counts['correct'] += row['score'] == 1
        return row

    def row(self, item, rec):
        gen, jud = rec['generation'], rec.get('judgment')
        if gen['status'] == 'infra_error':
            status, score, stage = 'infra_error', None, 'generation'
        elif getattr(self.a, 'stage', 'all') == 'generate' and jud is None:
            status, score, stage = 'pending', None, None
        elif gen['status'] in ('model_error', 'cutoff'):
            status, score, stage = 'model_error', 0, None
        elif jud is None or jud['status'] != 'ok':
            status, score, stage = 'infra_error', None, 'judge'
        else:
            status, score, stage = 'ok', int(jud['correct'] == 'yes'), None
        usage = gen.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        row = {'item_id': item['id'], 'repeat': rec['repeat'], 'score': score, 'status': status,
                'error_stage': stage, 'error': (gen.get('error') if stage == 'generation' else
                                               (jud or {}).get('error') if stage == 'judge' else gen.get('error')),
                'transcript': str(self.path(item['id'], rec['repeat'])), 'generation_status': gen['status'],
                'partition': item['partition'], 'has_image': bool(item['image']), 'answer_type': item['answer_type'],
                'category': item['category'],
                'finish_reason': 'cutoff' if gen['status'] == 'cutoff' else gen.get('finish_reason'),
                'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                'reasoning_tokens': details.get('reasoning_tokens'), 'latency_s': round(gen.get('latency_s') or 0, 1),
                'model_attempts': len(gen.get('attempts') or []), 'stop_reason': gen.get('stop_reason'),
                'max_tokens': gen.get('max_tokens'),
                # no final answer because the model ended its turn inside the reasoning block (EOS before </think>):
                # all completion tokens but the end token were reasoning
                'reasoning_unclosed': bool(gen['status'] == 'model_error' and gen.get('finish_reason') == 'stop'
                                           and details.get('reasoning_tokens') is not None
                                           and details['reasoning_tokens'] >= (usage.get('completion_tokens') or 0) - 1),
                'judge_correct': (jud or {}).get('correct'), 'judge_confidence': (jud or {}).get('confidence'),
                'extracted_answer': (jud or {}).get('extracted_final_answer')}
        row['model_error_kind'] = scoring.model_error_kind(row)
        return row

    async def progress(self):
        t0 = time.time()
        while True:
            await asyncio.sleep(self.a.progress_every)
            c = self.counts
            inf, lim = (self.lane.in_flight, self.lane.limit) if self.lane else (self.msem.inflight, self.msem.limit)
            print(f'[{time.strftime("%H:%M:%S")}] {c["done"]}/{self.total} done | ok {c["ok"]} model_error '
                  f'{c["model_error"]} infra_error {c["infra_error"]} | correct {c["correct"]} | '
                  f'{(time.time() - t0) / 60:.1f} min | in flight {inf}/{lim}', flush=True)

    async def run(self, pairs):
        self.total = len(pairs)
        if getattr(self.a, 'cutoff_min', None):
            self.deadline = time.time() + 60 * self.a.cutoff_min
        ticker = asyncio.create_task(self.progress())
        control = asyncio.create_task(self.msem.control()) if self.msem.metrics_url and not self.lane else None
        if self.lane:   # announce the work so the dispatcher sees this lane's queue before tokenization
            self.lane.expect([(f'{i}.r{k}', i, self.pred(i)) for i, k in pairs if not self.path(i, k).exists()])
        try:
            rows = await asyncio.gather(*(self.one(i, k) for i, k in pairs))
        finally:
            ticker.cancel()
            if control:
                control.cancel()
            if self.lane:
                self.lane.close()
            if self.model is not None:
                await self.model.close()
            if self.judge is not None:
                await self.judge.close()
        return rows


FINGERPRINT = ('model_served_name', 'generation', 'prompt', 'dataset_revision', 'judge_spec', 'judge_settings',
               'n_repeats', 'context_window')


GENERATION_KEYS = ('model_served_name', 'generation', 'prompt', 'dataset_revision')


def seed_generations(src_dir, dst_dir, pairs):
    """Copy the finished generations of another run (judgments dropped) so this run only judges them.
    Returns how many were copied; infra-failed generations are left for this run to redo."""
    n = 0
    for item_id, repeat in pairs:
        src = Path(src_dir) / 'transcripts' / f'{item_id}.r{repeat}.json'
        dst = Path(dst_dir) / 'transcripts' / f'{item_id}.r{repeat}.json'
        if dst.exists() or not src.exists():
            continue
        rec = json.loads(src.read_text())
        if rec['generation']['status'] == 'infra_error':
            continue
        atomic_json(dst, {**rec, 'judgment': None, 'generation_from': str(src)})
        n += 1
    return n


def fingerprint(cfg):
    return {'model_served_name': cfg['model']['served_name'], 'generation': cfg['generation'],
            'prompt': cfg['prompt'], 'dataset_revision': cfg['dataset']['revision'],
            'judge_spec': cfg['judge']['spec'], 'judge_settings': cfg['judge']['settings'],
            'n_repeats': cfg['n_repeats'], 'context_window': (cfg.get('context') or {}).get('window')}


def response_budget_on() -> dict | None:
    """The capped protocol's HLE response budget (PI_RESPONSE_BUDGET), or None."""
    from tb_agents import response_budget as response_policy
    return response_policy.with_benchmark(response_policy.settings(), 'hle')


def per_item_window(stage, generations_from, context, capped):
    """The window each item's max_tokens is fitted to (window - prompt bound), or None: no max_tokens, the endpoint's
    own context bounds the reply. Needed where the endpoint's context is larger than the window, and whenever a capped
    response budget is on: the budget sets max_tokens itself, and without the item's room a request whose prompt + the
    budget's cap passes the endpoint's context is rejected (HTTP 400: m5l SFT models served at exactly 131,072,
    2026-11-15, every prompt over 222 tokens) or runs past the window in its continuations; the budget's near-window
    rule needs that room as well."""
    if stage == 'grade' or generations_from or not context or not context.get('window'):
        return None
    ctx = context.get('endpoint_max_model_len')
    if ctx and context['window'] >= ctx and not capped:
        return None
    return context['window']


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['all', 'generate', 'grade'], default='all',
                    help='generate saves ungraded answers without a live judge; grade reuses answers without a live model')
    ap.add_argument('--model', help='serving:<name> | url:<base>@<served_name> | openai:<model>')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int, help='first N items by sha256(id) (smoke subsets)')
    ap.add_argument('--stratify', action='store_true',
                    help='with --limit: proportional over partition x modality x answer type (at least 1 each)')
    ap.add_argument('--ids', help='file with one item id per line')
    ap.add_argument('--repeats', type=int, default=1, help='AA runs 1')
    ap.add_argument('--dataset', default='cais/hle-diamond', choices=sorted(data.REVISIONS))
    ap.add_argument('--task-set', choices=data.TASK_SETS, default=data.DEFAULT_TASK_SET,
                    help="text-only-v1 (default, #348): the questions without an image (728 of HLE-Diamond's 1,000); "
                         "'all': every question (AA's set, M3.0-M3.6 runs). --limit/--ids select within the set")
    ap.add_argument('--text-only', action='store_true', help='drop images (AA: text-only models get the text). '
                    'Automatic for a serving: model launched without image input (--language-model-only, or no '
                    'image allowance in --limit-mm-per-prompt)')
    ap.add_argument('--context-window', type=int, default=protocol.CONTEXT_WINDOW,
                    help='total tokens per request, prompt + output (M3.0: 131072 for regular evals; 0 = the '
                         "endpoint's full context, for Qwen3.8 calibration runs only)")
    ap.add_argument('--system-prompt', default='by_type', choices=prompts.SYSTEM_POLICIES)
    ap.add_argument('--system-role', default='system', choices=['system', 'user'],
                    help="'user' prepends the prompt to the user turn (models without a system prompt)")
    g = ap.add_argument_group('generation (mirror the AA variant compared against)')
    g.add_argument('--no-endpoint-defaults', dest='endpoint_defaults', action='store_false',
                   help="ignore the serving registry's request overrides (lab-recommended sampling)")
    g.add_argument('--temperature', type=float)
    g.add_argument('--top-p', type=float)
    g.add_argument('--max-tokens', type=int, help='default: unset, i.e. the rest of the served context')
    g.add_argument('--reasoning-effort', help='set wherever the model request carries an effort (top level and '
                                             'chat_template_kwargs); refused for models without effort levels')
    g.add_argument('--chat-template-kwargs', help='JSON, merged over the registry')
    g.add_argument('--extra-body', help='JSON merged into every model request')
    g.add_argument('--concurrency', type=int, default=128, help='max model requests in flight')
    g.add_argument('--metrics-url', help="the model server's vLLM /metrics: the in-flight limit then adapts "
                                          '(down when vLLM queues or preempts, up while KV < 0.85)')
    g.add_argument('--start-concurrency', type=int, help='initial adaptive limit (default --concurrency)')
    g.add_argument('--min-concurrency', type=int, default=8)
    g.add_argument('--timeout', type=float, default=7200, help='per model request, seconds')
    g.add_argument('--max-attempts', type=int, default=30, help='AA retries API failures up to 30 times')
    g.add_argument('--dispatch', metavar='DIR', help='#344: take the in-flight allowance from this dispatcher '
                   '(aa_evals.common.dispatch; replaces --concurrency and --metrics-url), longest predicted first')
    g.add_argument('--request-cap-sec', type=float, help='#344: wall cap on one model call including retries; at the '
                   'cap the item has no answer (model_error, finish_reason request_cap, scored 0)')
    j = ap.add_argument_group('judge')
    j.add_argument('--judge', default=DEFAULT_JUDGE,
                   help=f'model spec (default {DEFAULT_JUDGE}, our self-hosted open judge; AA uses openai:gpt-5.6-luna)')
    j.add_argument('--judge-reasoning-effort',
                   help="openai: judges default to medium (AA); served judges keep the registry's effort unless "
                        "set (pinned #41: medium for the Qwen judge, registry defaults for Super)")
    j.add_argument('--judge-max-tokens', type=int, default=32768,
                   help='includes judge reasoning (Super as judge passed 16k on 5 of 39 smoke items)')
    j.add_argument('--judge-extra-body', help='JSON merged into every judge request')
    j.add_argument('--judge-arm', help='a judge arm from aa_evals/judges/arms.py (e.g. qwen38-27b-medium-pin): sets '
                                       '--judge to its server and its settings as the extra body; recorded as '
                                       f'judge.arm. Default: the pinned arm for HLE, {pins.default_arm(BENCH)}, '
                                       'unless --judge names another judge')
    j.add_argument('--judge-concurrency', type=int, default=32)
    j.add_argument('--judge-format-attempts', type=int, default=5, help='re-asks after off-schema judge output')
    ap.add_argument('--allow-external-api', action='store_true',
                    help='required for openai: model or judge specs (not approved for this project as of 2026-11-02)')
    ap.add_argument('--aa-slug', help='AA leaderboard slug of the model, for aa_reference')
    ap.add_argument('--wait-for-endpoint', type=float, default=600, help='seconds to wait for a fresh registry')
    ap.add_argument('--generations-from', metavar='RUN_ID',
                    help="reuse another run's generations (same model, sampling, prompt, dataset) and only judge, "
                         "e.g. to re-judge with a different judge")
    ap.add_argument('--force', action='store_true', help='resume even if the run config changed')
    ap.add_argument('--reverse', action='store_true',
                    help='start items in reverse order (another run works the same items from the front)')
    ap.add_argument('--rejudge', action='store_true',
                    help='on resume, re-grade stored judgments made with another judge or judge settings')
    ap.add_argument('--progress-every', type=float, default=60)
    ap.add_argument('--note', help='free text recorded in config.json (e.g. why a substitute endpoint was used)')
    ap.add_argument('--cutoff-min', type=float,
                    help='no-tails rule: after this many minutes, unfinished generations score 0 (finish_reason '
                         'cutoff); a resume regenerates them')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    source_cfg = None
    if a.stage == 'grade':
        source_dir = run_path(a.generations_from or a.run_id)
        source_cfg = json.loads((source_dir / 'config.json').read_text())
        a.model = source_cfg['model']['spec']
        a.dataset = source_cfg['dataset']['id']
        a.repeats = source_cfg['n_repeats']
        a.limit = source_cfg['dataset'].get('limit')
        a.task_set = source_cfg['dataset'].get('task_set', 'all')   # runs before #348 drew from all questions
        a.text_only = source_cfg['prompt']['text_only']
        a.system_prompt = source_cfg['prompt']['system_prompt']
        a.system_role = source_cfg['prompt']['system_role']
        a.metrics_url = None
    elif not a.model:
        raise SystemExit('--model is required for generation')
    if a.judge_arm is None and a.judge == DEFAULT_JUDGE:
        a.judge_arm = pins.default_arm(BENCH)   # the benchmark's pinned judge (#41 audit 1, #175)
    if a.judge_arm:
        from aa_evals.judges.arms import apply_arm
        a.judge, a.judge_extra_body = apply_arm(a.judge_arm, a.judge, a.judge_extra_body, DEFAULT_JUDGE)
    for spec in ((a.model,) if a.stage == 'generate' else (a.judge,) if a.stage == 'grade' else (a.model, a.judge)):
        if spec.startswith('openai:') and not a.allow_external_api:
            raise SystemExit(f'{spec} is billed by the provider: pass --allow-external-api once spend is approved')
    if source_cfg:
        m = source_cfg['model']
        model_ep = Endpoint(m['spec'], m['kind'], m['served_name'], m['urls'], None, m.get('request') or {}, m.get('registry'))
    try:
        if not source_cfg:
            model_ep = resolve_now(a.model, 0 if a.generations_from else a.wait_for_endpoint)
    except Retryable as e:
        if not a.generations_from:
            raise SystemExit(f'model {a.model} is not serving ({e})')
        # re-judging another run's generations needs no live model: describe it as that run recorded it
        m = json.loads((run_path(a.generations_from) / 'config.json').read_text())['model']
        model_ep = Endpoint(m['spec'], m['kind'], m['served_name'], m['urls'], None, m.get('request') or {},
                            m.get('registry'))
    judge_ep = Endpoint(a.judge, 'deferred', None, [])
    try:
        if a.stage != 'generate':
            judge_ep = resolve_now(a.judge, a.wait_for_endpoint)
    except Retryable as e:
        raise SystemExit(f'judge {a.judge} is not serving ({e}); start it or pass --judge serving:<name>')
    if judge_ep.kind == 'openai' and not judge_ep.api_key:
        raise SystemExit('OPENAI_API_KEY not found (environment or ~/.bashrc)')

    items = data.load(a.dataset)
    manifest = json.loads((data.local_dir(a.dataset) / 'manifest.json').read_text())
    ids = [l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()] if a.ids else None
    if source_cfg:
        ids = source_cfg.get('item_ids') or ids
        if not ids:
            a.stratify = source_cfg['dataset'].get('selection', '').startswith('stratified')
    pool = data.task_set(items, a.task_set)
    expected = data.TASK_SET_SIZES.get((a.dataset, a.task_set))
    if expected and manifest['revision'] == data.REVISIONS.get(a.dataset) and len(pool) != expected:
        raise SystemExit(f'--task-set {a.task_set}: {len(pool)} questions at the pinned revision, expected {expected}')
    excluded = {it['id'] for it in items} - {it['id'] for it in pool}
    if ids and (out := excluded & set(ids)):
        raise SystemExit(f'--ids: {len(out)} ids are outside --task-set {a.task_set} (questions with an image, '
                         '#348); pass --task-set all')
    chosen = select(pool, a.limit, ids, a.stratify)

    gen = source_cfg['generation'] if source_cfg else generation_settings(a, model_ep)
    jset = judge_settings(a, judge_ep) if a.stage != 'generate' else {}
    images, images_why = (None, 'stored generation protocol') if source_cfg else protocol.image_inputs(a.model)
    if images is False and not a.text_only:
        a.text_only = True   # AA: a model without image input gets the question text and is graded on all items
        print(f'text-only model ({images_why}): images dropped', flush=True)
    capped = bool(response_budget_on())
    if source_cfg:
        context = source_cfg.get('context')
    elif a.generations_from:
        context = json.loads((run_path(a.generations_from) / 'config.json').read_text()).get('context')
    else:
        ctx = endpoint_context(model_ep)
        win, why = protocol.window(ctx, a.context_window)
        context = {'window': win, 'rule': why, 'endpoint_max_model_len': ctx, 'requested': a.context_window,
                   'max_tokens': ('--max-tokens' if a.max_tokens else
                                  None if not win or (ctx and win >= ctx and not capped) else
                                  f'per item: window - (/tokenize text + {TEMPLATE_SLACK} + {IMAGE_ALLOWANCE}/image)')}
        if a.max_tokens and win and a.max_tokens >= win:
            raise SystemExit(f'--max-tokens {a.max_tokens} exceeds the {win}-token context window')
    a.window = per_item_window(a.stage, a.generations_from, context, capped)
    run_dir = RUNS_ROOT / a.run_id
    cfg = {
        'run_id': a.run_id, 'benchmark': 'hle', 'stage': a.stage, 'item_ids': [it['id'] for it in chosen],
        'git': source_cfg.get('git') if source_cfg else git_state(), 'stage_git': git_state(), 'argv': sys.argv if argv is None else argv,
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'host': os.uname().nodename,
        'condor_job': os.environ.get('HLE_JOB_ID'),
        'model': model_ep.describe(), 'generation': gen, 'context': context, 'effort': protocol.effort_of(gen),
        'image_inputs': {'sent': not a.text_only, 'endpoint': images, 'why': images_why},
        'prompt': {'system_prompt': a.system_prompt, 'system_role': a.system_role, 'text_only': a.text_only,
                   'image_placement': 'after question text' if not a.text_only else 'dropped'},
        'dataset': {'id': a.dataset, 'revision': manifest['revision'], 'sha256': manifest['sha256'],
                    'n_total': len(items), 'task_set': a.task_set, 'n_task_set': len(pool),
                    'n_selected': len(chosen), 'limit': a.limit, 'ids_file': a.ids,
                    'selection': 'ids file' if ids else ('stratified by partition x modality x answer type, '
                                 'first by sha256(id)' if a.stratify else 'first N by sha256(id)') if a.limit
                                 else 'all'},
        'judge': {'spec': a.judge, 'arm': a.judge_arm, 'served_name': judge_ep.served_name, 'kind': judge_ep.kind,
                  'settings': {k: v for k, v in jset.items() if k != 'response_format'},
                  'prompt': 'official HLE judge prompt (run_judge_results.py)',
                  'response_format': 'json_schema ExtractedAnswer (strict)',
                  'response_format_schema': prompts.JUDGE_SCHEMA,
                  'format_attempts': a.judge_format_attempts},
        'harness': 'aa_evals.hle: official HLE prompts, async OpenAI-compatible chat client, no tools',
        'n_repeats': a.repeats, 'concurrency': a.concurrency, 'judge_concurrency': a.judge_concurrency,
        'max_attempts': a.max_attempts, 'timeout_s': a.timeout, 'cutoff_min': a.cutoff_min,
        'dispatch': a.dispatch and str(Path(a.dispatch).resolve()), 'request_cap_sec': a.request_cap_sec,
        'adaptive_concurrency': a.metrics_url and {'metrics_url': a.metrics_url, 'start': a.start_concurrency,
                                                   'min': a.min_concurrency},
        'aa_slug': a.aa_slug or protocol.aa_model(model_ep.served_name, gen)[1],
        'note': a.note, 'rejudge': a.rejudge, 'reverse': a.reverse,
    }
    # #481: the AAI-open version (the launcher's label, or that of the run whose generations this one uses)
    inherit = source_cfg if source_cfg is not None else (
        json.loads((run_path(a.generations_from) / 'config.json').read_text()) if a.generations_from else None)
    dev = [] if inherit is not None else aai_open.client_deviations(
        context_window=(context or {}).get('window'), overrides={'temperature': a.temperature, 'top_p': a.top_p},
        flags={'--extra-body': a.extra_body, '--chat-template-kwargs': a.chat_template_kwargs,
               '--max-tokens': a.max_tokens, '--no-endpoint-defaults': not a.endpoint_defaults})
    dev += aai_open.judge_deviations(a.judge_arm) if a.stage != 'generate' else []
    cfg['aai_open'] = aai_open.record(dev, source_cfg=inherit)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'transcripts').mkdir(exist_ok=True)
    old = run_dir / 'config.json'
    if old.exists():
        prev = json.loads(old.read_text())
        ignored = {'judge_spec', 'judge_settings'} if prev.get('stage') == 'generate' or a.stage == 'generate' else set()
        diff = [k for k in FINGERPRINT if k not in ignored and fingerprint(prev)[k] != fingerprint(cfg)[k]]
        if diff and not a.force:
            raise SystemExit(f'{run_dir} exists with a different config ({diff}); new --run-id or --force')
        prev_set = prev['dataset'].get('task_set', 'all')   # runs before #348 drew from all questions
        if prev_set != a.task_set and not a.force:
            raise SystemExit(f'{run_dir} was started on --task-set {prev_set}; pass --task-set {prev_set} to resume it '
                             '(or use a new --run-id)')
        cfg['resumed_from'] = prev.get('started_at')
        cfg['aai_open'] = aai_open.record(dev, source_cfg=prev)   # a resumed run keeps the label it started with
        cfg['stage_history'] = prev.get('stage_history', []) + [{'stage': prev.get('stage', 'all'), 'git': prev.get('stage_git', prev.get('git'))}]
        if a.stage == 'generate' and prev.get('stage') != 'generate':
            cfg['judge'] = prev['judge']
        if any(fingerprint(prev)[k] != fingerprint(cfg)[k] for k in ('judge_spec', 'judge_settings')):
            a.rejudge = cfg['rejudge'] = True   # a forced judge change never leaves old judgments mixed in
    pairs = [(it['id'], k) for it in chosen for k in range(a.repeats)]
    if a.reverse:   # a second endpoint working the same items from the other end (merged with hle.merge)
        pairs.reverse()
    if a.generations_from:
        src_dir = run_path(a.generations_from)
        src = json.loads((src_dir / 'config.json').read_text())
        mismatch = [k for k in GENERATION_KEYS if fingerprint(src)[k] != fingerprint(cfg)[k]]
        if mismatch and not a.force:
            raise SystemExit(f'--generations-from {a.generations_from}: generation config differs ({mismatch})')
        cfg['generations_from'] = {'run_id': a.generations_from, 'git': src.get('git'),
                                   'copied': seed_generations(src_dir, run_dir, pairs)}
    atomic_json(run_dir / 'config.json', cfg)
    print(f'HLE run {a.run_id}: {len(chosen)} items x {a.repeats} | model {model_ep.served_name} ({a.model}) | '
          f'judge {judge_ep.served_name} | {run_dir}', flush=True)

    model = ChatClient(a.model, timeout=a.timeout, max_attempts=a.max_attempts) if a.stage != 'grade' else None
    judge = ChatClient(a.judge, timeout=1800, max_attempts=a.max_attempts) if a.stage != 'generate' else None
    runner = Runner(a, chosen, model, judge, gen, jset, run_dir)
    rows = asyncio.run(runner.run(pairs))

    with open(run_dir / '.items.jsonl.tmp', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    (run_dir / '.items.jsonl.tmp').replace(run_dir / 'items.jsonl')
    summary = scoring.summarize(rows, n_items=len(chosen), n_repeats=a.repeats, dataset=a.dataset,
                                aa_slug=cfg['aa_slug'],
                                extra={'run_id': a.run_id, 'model': model_ep.served_name, 'judge': judge_ep.served_name,
                                       'subset': a.limit is not None or ids is not None,
                                       'task_set': a.task_set,
                                       'n_cutoff': sum(r['finish_reason'] == 'cutoff' for r in rows),
                                       'n_reasoning_unclosed': sum(bool(r.get('reasoning_unclosed')) for r in rows),
                                       'judge_arm': a.judge_arm,
                                       'n_over_window': None if not (context or {}).get('window') else sum(
                                           (r.get('prompt_tokens') or 0) + (r.get('completion_tokens') or 0)
                                           > context['window'] for r in rows),
                                       **protocol.summary_fields(model=a.model, served_name=model_ep.served_name,
                                           body=gen, context=context, judge_arm=a.judge_arm,
                                           judge_settings_key=runner.judge_key, code_sha=cfg['git'].get('sha'),
                                           repeats=a.repeats, image_inputs=not a.text_only),
                                       'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    summary = fold_summary(run_dir, summary)   # n_flagged: the M3.4 trace judge (#177)
    summary['aai_open_version'] = cfg['aai_open']['version']
    atomic_json(run_dir / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('score', 'ci95', 'n_items', 'n_scored_items', 'n_infra_errors',
                                              'n_model_errors', 'valid', 'aa_reference')}), flush=True)
    if a.stage == 'generate':
        return 0 if summary['generation_complete'] and not summary['n_infra_errors'] else 2
    return 0 if summary['valid'] else 2


if __name__ == '__main__':
    sys.exit(main())

"""SciCode / SciCode-Verified on our cluster: official-style cumulative multi-step generation, sandboxed grading.

    # the AAI-open stand-in (FrontierCode slot): Verified + our patch list, no background, 5 repeats
    python -m aa_evals.scicode.run --model serving:<name> --run-id <id> --dataset verified-aa-v3 [--effort medium] --condor
    # harness check against AA: the original SciCode, with background (AA's setting)
    python -m aa_evals.scicode.run --model serving:<name> --run-id <id> --dataset original --background on --repeats 3 --condor
    # recompute summary.json from items.jsonl; regrade saved code (sandbox: --condor)
    python -m aa_evals.scicode.run --run-id <id> --summarize
    python -m aa_evals.scicode.run --run-id <id> --regrade [--dataset ...] [--condor]

Each (problem, repeat) is a chain: step k's prompt holds steps 1..k-1 with the model's own code, so steps run in
order; chains run in parallel. Each step is graded as soon as it is generated, in the grading SIFs (OR over the
np126/np22 envs). Rerunning the same command resumes: finished steps are kept, infra errors are retried.

Outputs in /data/aai/aa_evals/runs/scicode/<run_id>/: config.json, items.jsonl (one row per step x repeat; the last
row of a key wins), chains/<pid>.r<k>.json (code + metadata per step), transcripts/<pid>.r<k>.jsonl.gz (prompts,
reasoning, responses: transcripts/<pid>.r<k>.gen.jsonl.gz; the trace judge's per-step view with harness events:
transcripts/<pid>.r<k>.jsonl), summary.json (n_flagged from flags.jsonl via aa_evals.tracejudge.fold_summary).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from ..common import aai_open, jsonl, response_budget_env, serving
from . import data, grader, presets, prompts
from ..hle.client import ChatClient, InfraError, message_fields, resolve_now

RUNS = Path(os.environ.get('SCICODE_RUNS') or '/data/aai/aa_evals/runs/scicode')
WORKTREE = Path(__file__).resolve().parents[2]
VENV_PY = '/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python'   # httpx; the grading stack is in the SIFs
# --condor jobs run a frozen `git archive` of these; generate() imports tb_agents' response budget (budget or not)
FROZEN = ('aa_evals', 'tb_agents')
CTX_MARGIN = 256          # tokens kept free between prompt + max_tokens and the context cap
DEFAULT_TIMEOUT = {'original': 300, 'verified': 1800}   # AA's grading timeout / SciCode(-Verified)'s


# ----------------------------------------------------------------------------------------------- small helpers

def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(path)


BAD_LINES = jsonl.Tolerant()   # undecodable lines seen in this run's files: counted in summary.json, which is then invalid


def read_jsonl(path: Path) -> list[dict]:
    """'\n'-separated (aa_evals.common.jsonl); an unterminated last line is a row still being appended; any other
    bad line is recorded in BAD_LINES."""
    return BAD_LINES.read(path)


def git_sha() -> str:
    if os.environ.get('SCICODE_CODE_SHA'):
        return os.environ['SCICODE_CODE_SHA']
    try:
        return subprocess.run(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def git_dirty() -> bool:
    if os.environ.get('SCICODE_CODE_SHA'):
        return False
    r = subprocess.run(['git', '-C', str(WORKTREE), 'status', '--porcelain', '--', *FROZEN], capture_output=True,
                       text=True)
    return bool(r.stdout.strip())


def family(dataset: str) -> str:
    return 'original' if dataset == 'original' else 'verified'


# ----------------------------------------------------------------------------------------------- token budget

class Budget:
    """max_tokens for a request = context cap - prompt tokens - margin. Prompt tokens come from the server's
    /tokenize (vLLM; chat-format, so the template is counted) when it answers, else a conservative estimate
    (bytes / 2.5; code-heavy prompts run ~3.5 bytes per token)."""

    def __init__(self, cap: int, endpoint):
        self.cap, self.endpoint, self.mode = cap, endpoint, None

    async def prompt_tokens(self, http: httpx.AsyncClient, messages, gen) -> int:
        if self.mode != 'estimate':
            ep = self.endpoint
            for url in ep.urls[:1]:
                root = url[:-3] if url.endswith('/v1') else url
                try:
                    body = {'model': ep.served_name, 'messages': messages, 'add_generation_prompt': True}
                    if gen.get('chat_template_kwargs'):
                        body['chat_template_kwargs'] = gen['chat_template_kwargs']
                    r = await http.post(root + '/tokenize', json=body, timeout=60)
                    if r.status_code == 200 and 'count' in r.json():
                        self.mode = 'tokenize'
                        return int(r.json()['count'])
                except (httpx.HTTPError, ValueError):
                    pass
            if self.mode is None:
                self.mode = 'estimate'
                print('/tokenize unavailable: prompt tokens estimated as bytes/2.5', flush=True)
        return math.ceil(len(json.dumps(messages).encode()) / 2.5)

    async def max_tokens(self, http, messages, gen) -> tuple[int, int]:
        n = await self.prompt_tokens(http, messages, gen)
        return self.cap - n - CTX_MARGIN, n


# ----------------------------------------------------------------------------------------------- generation + grading

class Runner:
    def __init__(self, a, run_dir: Path, cfg: dict, ds: data.Dataset):
        self.a, self.run_dir, self.cfg, self.ds = a, run_dir, cfg, ds
        self.items_path = run_dir / 'items.jsonl'
        (run_dir / 'chains').mkdir(parents=True, exist_ok=True)
        (run_dir / 'transcripts').mkdir(parents=True, exist_ok=True)
        self.gen = cfg['generation']['request']
        self.with_bg = cfg['background']
        self.timeout = cfg['grading']['timeout_s']
        self.envs = cfg['grading']['envs']
        self.grade_sem = asyncio.Semaphore(a.grade_jobs)
        self.req_sem = asyncio.Semaphore(a.concurrency)
        # #344 dispatcher (--dispatch DIR): a chain holds one slot of the run's shared in-flight budget while it
        # generates (longest predicted first); --chain-cap-sec bounds a chain's generation wall time
        self.lane, self.pred = None, None
        if getattr(a, 'dispatch', None):
            from aa_evals.common.dispatch.lane import open_dispatch
            self.lane, self.pred, _ = open_dispatch(a.dispatch, 'scicode', run_dir)
        self.chain_cap = getattr(a, 'chain_cap_sec', None)
        self.lock = asyncio.Lock()
        self.done = 0
        # a step whose chain record is graded but whose final items row is missing or not ok (e.g. the append
        # failed after the chain write on a /data fault) gets its transcript and row re-emitted on resume
        self.final = {k: r['status'] for k, r in final_rows(run_dir).items()} if self.items_path.exists() else {}

    def chain_path(self, pid, r):
        return self.run_dir / 'chains' / f'{pid}.r{r}.json'

    def load_chain(self, pid, r) -> dict:
        p = self.chain_path(pid, r)
        return json.loads(p.read_text()) if p.exists() else {'problem_id': pid, 'repeat': r, 'steps': {}}

    async def append_item(self, row: dict):
        async with self.lock:
            with open(self.items_path, 'a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def transcript(self, pid, r, rec: dict):
        """The raw generation record (prompt, reasoning, response) of a step, written when it is generated."""
        with gzip.open(self.run_dir / 'transcripts' / f'{pid}.r{r}.gen.jsonl.gz', 'at') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def gen_record(self, pid, r, sid) -> dict:
        p = self.run_dir / 'transcripts' / f'{pid}.r{r}.gen.jsonl.gz'
        out = {}
        if p.exists():
            with gzip.open(p, 'rt', encoding='utf-8') as f:
                text = f.read()
            bad: list = []
            for x in jsonl.loads(text, where=str(p), bad=bad):
                if x.get('step') == sid:
                    out = x   # the last generation of this step wins
            for where, i, why in bad:
                print(f'jsonl: {where}:{i}: undecodable line skipped ({why})', file=sys.stderr, flush=True)
        return out

    async def judged_transcript(self, pid, r, sid, rec, grade, gen=None):
        """The trace judge's view of one step attempt (aa_evals/tracejudge generic layout): the response plus the
        harness events of its generation and grading. Written once per graded attempt, before its items row."""
        gen = gen or self.gen_record(pid, r, sid)
        ev = [f"generation: finish_reason={rec.get('finish_reason')} max_tokens={rec.get('max_tokens')} "
              f"prompt_tokens_est={rec.get('prompt_tokens_est')} request_attempts={rec.get('n_attempts')} "
              f"latency_s={rec.get('latency_s')} context_cap={self.cfg['generation']['context_window']}",
              f"code extraction: {len(rec.get('code') or '')} chars"
              + (' (no ``` block: whole response used)' if rec.get('no_code_block') else '')
              + (' (empty response content)' if rec.get('empty_content') else '')]
        if rec.get('reason'):
            ev.append(f"generation not attempted: {rec['reason']}")
        for env, v in ((grade or {}).get('per_env') or {}).items():
            ev.append({'source': f'grader:{env}', 'text': f"rc={v['rc']} pass={v['pass']} timeout={v['timeout']} "
                                                          f"secs={v['secs']} (limit {self.timeout}s)"
                                                          + ('' if v['pass'] else f" tail: {v['tail'][-400:]}")})
        row = {'item_id': sid, 'repeat': r, 'key': f'{sid}/r{r}',
               'generation': {'content': gen.get('content'), 'reasoning': gen.get('reasoning'),
                              'finish_reason': rec.get('finish_reason'), 'usage': rec.get('usage'),
                              'max_tokens': rec.get('max_tokens')},
               'events': ev}
        async with self.lock:
            with open(self.run_dir / 'transcripts' / f'{pid}.r{r}.jsonl', 'a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    async def generate(self, client, http, budget, prompt: str) -> dict:
        messages = [{'role': 'user', 'content': prompt}]
        max_tokens, n_prompt = await budget.max_tokens(http, messages, self.gen)
        if max_tokens < 1024:
            return {'status': 'model_error', 'reason': 'prompt_over_context', 'prompt_tokens_est': n_prompt,
                    'content': None, 'reasoning': None, 'finish_reason': None, 'usage': {}}
        body = {'messages': messages, **self.gen, 'max_tokens': max_tokens}
        from tb_agents import response_budget as response_policy
        from tb_agents.response_budget_single import run_async_chat
        policy = response_policy.with_benchmark(response_policy.settings(), 'scicode')   # capped protocol (off unless set)
        async with self.req_sem:
            t0 = time.time()
            if policy:   # per step: each step's response is one budgeted single-turn item
                tries = []

                async def budgeted(b):
                    r, a = await client.chat(b)
                    tries.extend(a)
                    return r
                resp, attempts = await run_async_chat(budgeted, body, policy), tries
            else:
                resp, attempts = await client.chat(body)
        content, reasoning, finish, usage = message_fields(resp)
        return {'status': 'ok', 'content': content, 'reasoning': reasoning, 'finish_reason': finish, 'usage': usage,
                'max_tokens': max_tokens, 'prompt_tokens_est': n_prompt, 'latency_s': round(time.time() - t0, 1),
                'n_attempts': len(attempts), 'response_budget': resp.get('_response_budget')}

    async def grade(self, prob, k, codes) -> dict:
        s = prob['sub_steps'][k]
        sid = s['step_number']
        fn = lambda vendor, h5: grader.build_script(prob, k, codes, len(s['test_cases']), s['test_cases'],
                                                    vendor=vendor, h5=h5)
        async with self.grade_sem:
            return await asyncio.to_thread(grader.grade_step, fn, self.envs, self.ds.h5_for(sid), self.timeout,
                                           scratch=grader.scratch_dir())

    def to_generate(self, prob: dict, r: int) -> bool:
        st = self.load_chain(str(prob['problem_id']), r)
        return any(s['step_number'] not in data.SKIP_STEPS and (st['steps'].get(s['step_number']) is None or
                   st['steps'][s['step_number']].get('gen_status') == 'infra_error') for s in prob['sub_steps'])

    async def chain(self, client, http, budget, prob: dict, r: int):
        await asyncio.gather(*await self.generate_chain(client, http, budget, prob, r))

    async def generate_chain(self, client, http, budget, prob: dict, r: int) -> list:
        """Generate one chain inside its lane slot; return the tasks still grading its steps (they run outside the
        slot: grading is CPU only)."""
        pid = str(prob['problem_id'])
        if self.lane and self.to_generate(prob, r):
            async with self.lane.aslot(f'{pid}.r{r}', pid, self.pred(pid)):
                return await self._chain(client, http, budget, prob, r)
        return await self._chain(client, http, budget, prob, r)

    async def _chain(self, client, http, budget, prob: dict, r: int) -> list:
        pid = str(prob['problem_id'])
        st = self.load_chain(pid, r)
        codes, pending, gen_cache = [], [], {}
        # the cap counts this chain's generation time across resumes (st['chain_gen_sec']), not downtime between them
        deadline = time.time() + max(0.0, self.chain_cap - float(st.get('chain_gen_sec') or 0)) if self.chain_cap else None
        for k, s in enumerate(prob['sub_steps']):
            sid = s['step_number']
            if sid in data.SKIP_STEPS:
                codes.append(data.skip_gold(sid))
                continue
            rec = st['steps'].get(sid)
            if rec is None or rec.get('gen_status') == 'infra_error':
                prompt = prompts.build_prompt(prob, k, codes, self.with_bg)
                t_step = time.time()
                try:
                    if deadline is None:
                        g = await self.generate(client, http, budget, prompt)
                    elif deadline - time.time() <= 0:
                        g = self.capped()
                    else:
                        try:
                            g = await asyncio.wait_for(self.generate(client, http, budget, prompt),
                                                       deadline - time.time())
                        except TimeoutError:
                            g = self.capped()
                except InfraError as e:
                    rec = {'gen_status': 'infra_error', 'error': str(e)[:500]}
                    st['steps'][sid] = rec
                    if self.chain_cap:
                        st['chain_gen_sec'] = round(float(st.get('chain_gen_sec') or 0) + time.time() - t_step, 1)
                    async with self.lock:
                        atomic_json(self.chain_path(pid, r), st)
                    await self.append_item(self.row(pid, r, sid, rec, None))
                    return pending   # later steps need this one's code: the chain resumes here next time
                code = prompts.extract_code(g['content'])
                rec = {'gen_status': g['status'], 'reason': g.get('reason'), 'finish_reason': g['finish_reason'],
                       'cutoff': g['finish_reason'] == 'length', 'code': code,
                       'no_code_block': bool(g['content']) and '```' not in g['content'],
                       'empty_content': not g['content'], 'usage': g['usage'], 'max_tokens': g.get('max_tokens'),
                       'prompt_tokens_est': g.get('prompt_tokens_est'), 'prompt_sha': prompts.sha(prompt),
                       'latency_s': g.get('latency_s'), 'generated_at': time.time()}
                rec['n_attempts'] = g.get('n_attempts')
                genrec = {'step': sid, 'prompt': prompt, 'content': g['content'], 'reasoning': g['reasoning'],
                          'finish_reason': g['finish_reason'], 'usage': g['usage']}
                self.transcript(pid, r, genrec)
                gen_cache[sid] = genrec
                st['steps'][sid] = rec
                if self.chain_cap:
                    st['chain_gen_sec'] = round(float(st.get('chain_gen_sec') or 0) + time.time() - t_step, 1)
                async with self.lock:
                    atomic_json(self.chain_path(pid, r), st)
            codes.append(rec.get('code') or '')
            if sid not in self.ds.excluded_steps and rec.get('grade') is not None \
                    and self.final.get((sid, r)) in (None, 'infra_error'):
                row = self.row(pid, r, sid, rec, rec['grade'])
                await self.judged_transcript(pid, r, sid, rec, rec['grade'])
                await self.append_item(row)
                self.final[(sid, r)] = row['status']
            if sid not in self.ds.excluded_steps and rec.get('grade') is None:
                # graded in the background: the next step's generation does not wait for this step's tests
                pending.append(asyncio.create_task(self.grade_and_record(st, pid, r, prob, k, list(codes), rec,
                                                                         gen_cache.pop(sid, None))))
        return pending

    def capped(self) -> dict:
        """#344 chain cap: a step not generated within the chain's wall-clock cap is the model's (no code, scored 0,
        kept on resume); the steps before it are graded on what they produced."""
        return {'status': 'model_error', 'reason': 'chain_cap', 'content': None, 'reasoning': None,
                'finish_reason': 'chain_cap', 'usage': {}}

    async def grade_and_record(self, st, pid, r, prob, k, codes, rec, gen=None):
        sid = prob['sub_steps'][k]['step_number']
        if rec['gen_status'] != 'ok':
            rec['grade'] = {'pass': False, 'per_env': {}, 'note': rec.get('reason')}
        else:
            try:
                rec['grade'] = await self.grade(prob, k, codes)
            except grader.GradeInfraError as e:
                await self.append_item(self.row(pid, r, sid, {**rec, 'gen_status': 'grade_infra_error',
                                                              'error': str(e)[:500]}, None))
                return   # the code is kept; a resume regrades it
        async with self.lock:
            atomic_json(self.chain_path(pid, r), st)
        await self.judged_transcript(pid, r, sid, rec, rec['grade'], gen)
        await self.append_item(self.row(pid, r, sid, rec, rec['grade']))
        self.done += 1

    def row(self, pid, r, sid, rec, grade):
        if rec.get('gen_status') in ('infra_error', 'grade_infra_error'):
            status, score = 'infra_error', None
        elif rec.get('gen_status') == 'model_error':
            status, score = 'model_error', 0
        else:
            status, score = 'ok', int(bool(grade and grade['pass']))
        u = rec.get('usage') or {}
        return {'item_id': sid, 'problem_id': pid, 'repeat': r, 'score': score, 'status': status,
                'cutoff': bool(rec.get('cutoff')), 'finish_reason': rec.get('finish_reason'),
                'no_code_block': rec.get('no_code_block'), 'empty_content': rec.get('empty_content'),
                'completion_tokens': u.get('completion_tokens'), 'prompt_tokens': u.get('prompt_tokens'),
                'per_env': {k: v['pass'] for k, v in ((grade or {}).get('per_env') or {}).items()},
                'timeouts': [k for k, v in ((grade or {}).get('per_env') or {}).items() if v.get('timeout')],
                'fail_tail': next((v['tail'][-300:] for v in ((grade or {}).get('per_env') or {}).values()
                                   if not v['pass']), None) if grade and not grade['pass'] else None,
                'error': rec.get('error'),
                'transcript': f'transcripts/{pid}.r{r}.jsonl#{sid}/r{r}', 'chain': f'chains/{pid}.r{r}.json',
                # provenance: the dataset version and code this row is scored under, and, for a step carried over
                # by --upgrade-from, where its generation (and grade, if kept) came from
                'dataset': self.ds.name, 'code_sha': self.cfg['code_sha'], 'carried': rec.get('carried'),
                'at': time.strftime('%Y-%m-%dT%H:%M:%S')}

    async def run(self, probs: list[dict]):
        client = ChatClient(self.cfg['model']['spec'], timeout=self.a.request_timeout)
        work = [(p, r) for r in range(self.a.repeats) for p in probs]
        random.Random(0).shuffle(work)   # mix long and short problems across the run
        gen = [(p, r) for p, r in work if self.to_generate(p, r)]
        # a resume that only grades (the lane closed and the server was released while the last tests ran, #531)
        # needs no server: resolve the endpoint only when some chain still generates
        budget = Budget(self.cfg['generation']['context_window'],
                        resolve_now(self.cfg['model']['spec'], wait=600)) if gen else None
        if self.lane:   # the lane starts chains longest predicted first
            self.lane.expect([(f"{p['problem_id']}.r{r}", str(p['problem_id']), self.pred(str(p['problem_id'])))
                              for p, r in gen])
        t0 = time.time()

        async def progress():
            while True:
                await asyncio.sleep(300)
                print(f'[{time.strftime("%H:%M:%S")}] steps done {self.done}, {time.time() - t0:.0f}s', flush=True)

        pt = asyncio.create_task(progress())
        async with httpx.AsyncClient(trust_env=False) as http:
            grading = await asyncio.gather(*(self.generate_chain(client, http, budget, p, r) for p, r in work))
        try:
            if self.lane:
                # every model request is done: close the lane now, not after the CPU tests, so the dispatcher can
                # release the server (#531 B1: a step hung in its 1800 s test timeout x 2 envs held 2 idle B200s ~1 h,
                # 11-18). While the last tests run the lane no longer counts as owned: don't start a second client on
                # this run dir until this one exits.
                self.lane.close()
        finally:   # a failed lane write (Lustre) must not abandon the grading still in progress
            await asyncio.gather(*(t for ts in grading for t in ts))
        pt.cancel()
        await client.close()
        if budget is not None:
            self.cfg['generation']['prompt_token_count'] = budget.mode
        atomic_json(self.run_dir / 'config.json', self.cfg)


# ----------------------------------------------------------------------------------------------- scoring

def final_rows(run_dir: Path) -> dict:
    out = {}
    for row in read_jsonl(run_dir / 'items.jsonl'):
        out[(row['item_id'], row['repeat'])] = row
    return out


def bootstrap_ci(per_problem: dict[str, list[float]], weights: dict[str, int] | None, n=4000, seed=0):
    """95% CI of a weighted mean over problems, resampling problems (steps of a problem are not independent)."""
    keys = sorted(per_problem)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    vals = [sum(per_problem[k]) / len(per_problem[k]) for k in keys]
    w = [(weights or {}).get(k, 1) for k in keys]
    stats = []
    for _ in range(n):
        idx = [rng.randrange(len(keys)) for _ in keys]
        tw = sum(w[i] for i in idx)
        stats.append(sum(vals[i] * w[i] for i in idx) / tw)
    stats.sort()
    return [round(100 * stats[int(0.025 * n)], 2), round(100 * stats[int(0.975 * n) - 1], 2)]


def summarize(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / 'config.json').read_text())
    ds = data.load(cfg['dataset']['dataset'], check_h5=False)
    probs = [p for p in ds.problems if str(p['problem_id']) in set(cfg['problems'])]
    rows = final_rows(run_dir)
    R = cfg['repeats']
    step_scores, main_scores, steps_per_prob = {}, {}, {}
    n_infra = n_cutoff = n_scored = n_missing = 0
    env_pass = {lab: 0 for lab in cfg['grading']['envs']}
    main_by_rep = []
    for r in range(R):
        ok_main = n_main = 0
        for p in probs:
            pid = str(p['problem_id'])
            sids = [s['step_number'] for s in p['sub_steps']
                    if s['step_number'] not in data.SKIP_STEPS and s['step_number'] not in ds.excluded_steps]
            steps_per_prob[pid] = len(sids)
            res, infra = [], False
            for sid in sids:
                row = rows.get((sid, r))
                if row is None:
                    n_missing += 1
                    infra = True
                    continue
                if row['status'] == 'infra_error':
                    n_infra += 1
                    infra = True
                    continue
                n_scored += 1
                n_cutoff += bool(row.get('cutoff'))
                for lab, v in (row.get('per_env') or {}).items():
                    env_pass[lab] = env_pass.get(lab, 0) + bool(v)
                res.append(row['score'])
                step_scores.setdefault(pid, []).append(row['score'])
            if not infra and res:
                c = int(all(res))
                main_scores.setdefault(pid, []).append(c)
                ok_main += c
                n_main += 1
        main_by_rep.append(round(100 * ok_main / n_main, 2) if n_main else None)
    n_total = sum(steps_per_prob.values()) * R
    sub = 100 * sum(sum(v) for v in step_scores.values()) / max(1, n_scored)
    main = 100 * sum(sum(v) for v in main_scores.values()) / max(1, sum(len(v) for v in main_scores.values()))
    gen = cfg['generation']
    aa = presets.aa_reference(cfg['model'].get('served_name'), presets.effective_effort(gen['request']),
                              cfg['model'].get('aa_slug'))
    infra_rate = (n_infra + n_missing) / max(1, n_total)
    summary = {
        'benchmark': 'scicode' if cfg['dataset']['dataset'] == 'original' else 'scicode-verified',
        'dataset': cfg['dataset'],
        'metric': 'main-problem accuracy (headline) and sub-step accuracy, pass@1 averaged over repeats',
        'score': round(main if family(cfg['dataset']['dataset']) == 'verified' else sub, 2),
        'headline': 'main' if family(cfg['dataset']['dataset']) == 'verified' else 'sub',
        'main_problem_accuracy': round(main, 2),
        'main_ci95': bootstrap_ci(main_scores, None),
        'sub_step_accuracy': round(sub, 2),
        'sub_ci95': bootstrap_ci(step_scores, steps_per_prob),
        'ci95_over': 'problems (cluster bootstrap over problems; repeats pooled within a problem)',
        'main_by_repeat': main_by_rep,
        'sub_step_accuracy_by_env': {lab: round(100 * v / max(1, n_scored), 2) for lab, v in env_pass.items()},
        'n_problems': len(probs), 'n_steps': sum(steps_per_prob.values()), 'n_repeats': R,
        'n_step_attempts_total': n_total, 'n_scored': n_scored,
        'n_infra': n_infra, 'n_missing': n_missing, 'n_infra_errors': n_infra + n_missing,
        'infra_rate': round(infra_rate, 4), 'n_bad_jsonl_lines': BAD_LINES.count(run_dir),
        'valid': infra_rate <= 0.02 and not BAD_LINES.count(run_dir),
        'n_cutoff': n_cutoff, 'n_flagged': None,
        'complete': n_missing == 0,   # the trace judge's watch stops once this is true and every attempt is judged
        'background': cfg['background'],
        'model': cfg['model'], 'effort': presets.effective_effort(gen['request']),
        'context_cap': gen['context_window'], 'judge': None,
        'grading': {k: cfg['grading'][k] for k in ('envs', 'rule', 'timeout_s')},
        'code_sha': cfg['code_sha'], 'aa_reference': aa,
        'aa_comparable': cfg['dataset']['dataset'] == 'original' and cfg['background'],
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'aai_open_version': aai_open.of_config(cfg),   # #481
    }
    up = run_dir / 'upgrade.json'
    if up.exists():   # rows carried over from an earlier dataset version (--upgrade-from)
        info = json.loads(up.read_text())
        rows = final_rows(run_dir).values()
        summary['upgrade'] = {'from': info['from'], 'problems_touched': info['problems_touched'],
                              'seeded_steps': info['steps'],
                              'rows': {'carried_with_grade': sum(bool((r.get('carried') or {}).get('grade')) for r in rows),
                                       'carried_regraded': sum((r.get('carried') or {}).get('grade') is False for r in rows),
                                       'generated_here': sum(not r.get('carried') for r in rows)}}
    try:
        from ..tracejudge import fold_summary
        summary = fold_summary(run_dir, summary)   # n_flagged from the trace judge's flags.jsonl (None if not run)
    except ImportError:
        pass
    atomic_json(run_dir / 'summary.json', summary)
    return summary


# ----------------------------------------------------------------------------------------------- config

def select(ds: data.Dataset, a) -> list[dict]:
    probs = ds.problems
    if a.problems:
        want = a.problems.split(',')
        probs = [p for p in probs if str(p['problem_id']) in want]
        if len(probs) != len(want):
            raise SystemExit(f'unknown problem ids in --problems: {a.problems}')
    elif a.limit:
        probs = sorted(random.Random(20261104).sample(probs, a.limit), key=lambda p: int(p['problem_id']))
    return probs


def make_config(a, run_dir: Path, ds: data.Dataset, probs: list[dict]) -> dict:
    ep = resolve_now(a.model, wait=a.wait_server)
    gen = presets.request_settings(ep, effort=a.effort, temperature=a.temperature, top_p=a.top_p,
                                   extra_body=a.extra_body)
    fam = family(a.dataset)
    with_bg = {'on': True, 'off': False}[a.background or ('on' if fam == 'original' else 'off')]
    envs = dict(x.split('=', 1) for x in a.envs.split(',')) if a.envs else dict(grader.DEFAULT_ENVS)
    cfg = {
        'benchmark': 'scicode', 'run_id': a.run_id, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'code_sha': git_sha(), 'code_dirty': git_dirty(),
        'model': {**ep.describe(), 'aa_slug': a.aa_slug},
        'generation': {'request': gen, 'context_window': a.context_window,
                       'max_tokens': f'context_window - prompt tokens - {CTX_MARGIN}',
                       'request_timeout_s': a.request_timeout, 'messages': 'one user message (the template), no system prompt'},
        'dataset': ds.pins, 'problems': [str(p['problem_id']) for p in probs], 'limit': a.limit,
        'background': with_bg,
        'prompt': {'template': 'multistep_template.txt' if with_bg else 'background_comment_template.txt',
                   'template_sha': prompts.sha(prompts.TEMPLATE_BG if with_bg else prompts.TEMPLATE_NOBG),
                   'previous_steps': 'full extracted model code; skip-step gold in full'},
        'grading': {'envs': envs, 'rule': 'pass if any env passes (OR)', 'timeout_s': a.timeout or DEFAULT_TIMEOUT[fam],
                    'mem_limit_kb': grader.MEM_KB, 'threads': grader.THREADS,
                    'sif_sha256': {lab: sif_sha(env) for lab, env in envs.items()}},
        'repeats': a.repeats, 'note': a.note,
        'dispatch': getattr(a, 'dispatch', None) and str(Path(a.dispatch).resolve()),
    }
    if getattr(a, 'chain_cap_sec', None):   # only when set, so earlier runs resume unchanged
        cfg['generation']['chain_cap_sec'] = a.chain_cap_sec
    # #481: the AAI-open version (the launcher's label + what this run shows; a resume keeps its first label)
    dev = (aai_open.client_deviations(context_window=a.context_window,
                                      overrides={'temperature': a.temperature, 'top_p': a.top_p})
           + (['background on (v0.1: off)'] if with_bg else [])
           + ([f'dataset {a.dataset} (v0.1: verified-aa-v3)'] if a.dataset != 'verified-aa-v3' else [])
           + (['extra request body (v0.1: none)'] if a.extra_body else []))
    cfg['aai_open'] = aai_open.record(dev)
    old = run_dir / 'config.json'
    if old.exists():
        prev = json.loads(old.read_text())
        cfg['aai_open'] = aai_open.record(dev, source_cfg=prev)   # keeps the first label (+ any new deviation)
        for k in ('generation', 'dataset', 'problems', 'background', 'grading', 'repeats', 'code_sha'):
            was, now = prev.get(k), cfg.get(k)
            if k == 'generation':
                was = {x: y for x, y in (was or {}).items() if x != 'prompt_token_count'}
            if was != now and not a.force_config:
                raise SystemExit(f'{old}: "{k}" differs (finals never mix commits or settings; new --run-id, or '
                                 f'--force-config):\n  was {json.dumps(was)[:400]}\n  now {json.dumps(now)[:400]}')
        cfg['created_at'] = prev.get('created_at', cfg['created_at'])
    atomic_json(old, cfg)
    return cfg


def sif_sha(env: str):
    if not env.startswith('sif:'):
        return None
    p = Path(env[4:]).with_suffix('.sha256')
    return p.read_text().split()[0] if p.exists() else None


# ----------------------------------------------------------------------------------------------- regrade

async def regrade(a, run_dir: Path):
    """Grade the saved code of a run again (current dataset tests/targets), writing items to a new file
    items.regrade.<dataset>.jsonl so the run itself is untouched. For patch analysis, not for finals."""
    cfg = json.loads((run_dir / 'config.json').read_text())
    ds = data.load(a.dataset or cfg['dataset']['dataset'])
    envs = cfg['grading']['envs']
    out = run_dir / f'items.regrade.{ds.name}.jsonl'
    sem = asyncio.Semaphore(a.grade_jobs)
    probs = {str(p['problem_id']): p for p in ds.problems}
    tasks = []
    for path in sorted((run_dir / 'chains').glob('*.json')):
        st = json.loads(path.read_text())
        prob = probs.get(st['problem_id'])
        if prob is None:
            continue
        codes = []
        for k, s in enumerate(prob['sub_steps']):
            sid = s['step_number']
            if sid in data.SKIP_STEPS:
                codes.append(data.skip_gold(sid))
                continue
            rec = st['steps'].get(sid)
            if rec is None or rec.get('gen_status') == 'infra_error':
                break
            codes.append(rec.get('code') or '')
            if sid in ds.excluded_steps:
                continue
            tasks.append((st, prob, k, list(codes), rec))

    async def one(st, prob, k, codes, rec):
        s = prob['sub_steps'][k]
        if rec['gen_status'] != 'ok':
            g = {'pass': False, 'per_env': {}}
        else:
            fn = lambda vendor, h5: grader.build_script(prob, k, codes, len(s['test_cases']), s['test_cases'],
                                                        vendor=vendor, h5=h5)
            async with sem:
                g = await asyncio.to_thread(grader.grade_step, fn, envs, ds.h5_for(s['step_number']),
                                            cfg['grading']['timeout_s'], scratch=grader.scratch_dir())
        with open(out, 'a') as f:
            f.write(json.dumps({'item_id': s['step_number'], 'problem_id': st['problem_id'], 'repeat': st['repeat'],
                                'score': int(g['pass']), 'per_env': {x: v['pass'] for x, v in g['per_env'].items()},
                                'was': (rec.get('grade') or {}).get('pass')}) + '\n')

    await asyncio.gather(*(one(*t) for t in tasks))
    print(f'regraded {len(tasks)} steps -> {out}')


def seed_upgrade(src_id: str, run_dir: Path, ds: data.Dataset, cfg: dict) -> dict:
    """Seed a new run on a newer dataset version from a finished run, rerunning only what the patches touch
    (the #184 rerun policy: patches to specific problems rerun only the affected chains, from the first
    affected step on). Per chain, in step order, with the run's own earlier-step code:
      - prompt unchanged and tests/targets unchanged  -> generation and grade carried over;
      - prompt unchanged, tests or targets changed    -> generation carried over, regraded;
      - prompt changed (a text patch, or an earlier step regenerated) -> this step and every later step of the
        chain are regenerated.
    Excluded steps follow the new dataset. Carried steps record their origin (`carried` on the chain record and
    the items row). The normal driver then does the rest: generation, grading and the rows."""
    src = RUNS / src_id
    old_cfg = json.loads((src / 'config.json').read_text())
    for k in ('background', 'repeats', 'problems'):
        if old_cfg.get(k) != cfg.get(k):
            raise SystemExit(f'--upgrade-from {src_id}: "{k}" differs ({old_cfg.get(k)!r} vs {cfg.get(k)!r})')
    strip = lambda g: {x: y for x, y in g.items() if x != 'prompt_token_count'}
    if strip(old_cfg['generation']) != strip(cfg['generation']):
        raise SystemExit(f'--upgrade-from {src_id}: generation settings differ (an upgrade changes the dataset only)')
    for k in ('envs', 'timeout_s', 'sif_sha256'):
        if old_cfg['grading'].get(k) != cfg['grading'].get(k):
            raise SystemExit(f'--upgrade-from {src_id}: grading "{k}" differs')
    old_ds = data.load(old_cfg['dataset']['dataset'], check_h5=False)
    oldp = {str(p['problem_id']): p for p in old_ds.problems}
    newp = {str(p['problem_id']): p for p in ds.problems}
    origin = {'run_id': src_id, 'code_sha': old_cfg['code_sha'], 'dataset': old_ds.name}
    (run_dir / 'chains').mkdir(parents=True, exist_ok=True)
    (run_dir / 'transcripts').mkdir(parents=True, exist_ok=True)
    stats, per_problem = Counter(), defaultdict(Counter)
    for cp in sorted((src / 'chains').glob('*.json')):
        st = json.loads(cp.read_text())
        pid, r = str(st['problem_id']), st['repeat']
        if pid not in newp:
            continue
        prob, op = newp[pid], oldp[pid]
        old_steps = {s['step_number']: s for s in op['sub_steps']}
        codes, keep, cut = [], {}, False
        for k, s in enumerate(prob['sub_steps']):
            sid = s['step_number']
            if sid in data.SKIP_STEPS:
                codes.append(data.skip_gold(sid))
                continue
            rec = st['steps'].get(sid)
            if not cut and (rec is None or rec.get('gen_status') == 'infra_error'):
                cut = True   # unfinished in the source run: generated here as usual
            if not cut and prompts.sha(prompts.build_prompt(prob, k, codes, cfg['background'])) != rec.get('prompt_sha'):
                cut = True
            if cut:
                if sid not in ds.excluded_steps:
                    stats['regenerate'] += 1
                    per_problem[pid]['regenerate'] += 1
                continue
            rec = dict(rec)
            o = old_steps.get(sid) or {}
            same_tests = o.get('test_cases') == s['test_cases'] and old_ds.target_key(sid) == ds.target_key(sid)
            if not same_tests or rec.get('grade') is None:
                rec.pop('grade', None)
                rec['carried'] = {**origin, 'grade': False}
                kind = 'regrade'
            else:
                rec['carried'] = {**origin, 'grade': True}
                kind = 'carry'
            if sid not in ds.excluded_steps:
                stats[kind] += 1
                per_problem[pid][kind] += 1
            keep[sid] = rec
            codes.append(rec.get('code') or '')
        atomic_json(run_dir / 'chains' / cp.name, {**st, 'steps': keep})
        g = src / 'transcripts' / f'{pid}.r{r}.gen.jsonl.gz'
        if g.exists():   # the raw generations of the carried steps only (the judge view is rewritten per row)
            with gzip.open(g, 'rt', encoding='utf-8') as fi, \
                    gzip.open(run_dir / 'transcripts' / g.name, 'wt', encoding='utf-8') as fo:
                for line in jsonl.lines(fi.read()):
                    if not line.strip():
                        continue
                    try:
                        if json.loads(line).get('step') in keep:
                            fo.write(line + '\n')
                    except ValueError as e:
                        print(f'jsonl: {g}: undecodable generation line not carried ({e})', file=sys.stderr, flush=True)
    info = {'from': origin, 'to': {'dataset': ds.name, 'code_sha': cfg['code_sha']}, 'steps': dict(stats),
            'problems_touched': sorted((p for p, c in per_problem.items() if c['regenerate'] or c['regrade']),
                                       key=lambda x: int(x)),
            'per_problem': {p: dict(c) for p, c in per_problem.items() if c['regenerate'] or c['regrade']},
            'seeded_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    atomic_json(run_dir / 'upgrade.json', info)
    return info


# ----------------------------------------------------------------------------------------------- condor

def submit_condor(a, argv: list[str]) -> None:
    if git_dirty():
        raise SystemExit('commit aa_evals and tb_agents first (jobs run a frozen snapshot of HEAD)')
    sha = git_sha()
    code = RUNS / '_code' / sha[:12]
    if not code.exists():
        tmp = code.with_name(code.name + '.partial')
        tmp.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(['git', '-C', str(WORKTREE), 'archive', sha, *FROZEN], capture_output=True, check=True)
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=arc.stdout, check=True)
        tmp.rename(code)
    run_dir = RUNS / a.run_id
    (run_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    argv_file = run_dir / 'jobs' / f'argv.{stamp}.json'
    argv_file.write_text(json.dumps([x for x in argv if x != '--condor']))
    from aa_evals.common.condor_cost import bad_nodes   # the shared list, read now (tbshared; at submission)
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in bad_nodes())
    budget = response_budget_env.forwarded(VENV_PY, code)   # PI_RESPONSE_BUDGET (capped protocol), only when set here
    extra_env = (' ' + response_budget_env.condor_env(budget)) if budget else ''
    extra_env += serving.condor_env()   # TASKSYNTH_SERVING_ROOT, only when set here
    extra_env += aai_open.condor_env()  # AAI_OPEN_VERSION (#481), only when set here
    sub = f'''universe = vanilla
executable = {VENV_PY}
arguments = "-m aa_evals.scicode.run --argv-file {argv_file}"
initialdir = {code}
output = {run_dir}/jobs/run.$(Cluster).out
error = {run_dir}/jobs/run.$(Cluster).err
log = {run_dir}/jobs/condor.log
request_gpus = 0
request_cpus = {a.cpus}
request_memory = {a.mem_gb * 1024}
request_disk = 40G
+BypassLXCfs = "true"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin SCICODE_CODE_SHA={sha} SCICODE_RUNS={RUNS} PYTHONUNBUFFERED=1 APPTAINER_CACHEDIR=/data/aai/apptainer-cache{extra_env}"
requirements = {req} && (TARGET.GPUs == 0 || TARGET.GPUs =?= undefined)
queue
'''
    path = run_dir / 'jobs' / f'run.{stamp}.sub'
    path.write_text(sub)
    from aa_evals.common import condor_cost
    condor_cost.submit(path, getattr(a, 'cpu_bid', None))   # default: the CPU bid ladder (#378)


# ----------------------------------------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--argv-file']:
        argv = json.loads(Path(argv[1]).read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='serving:<name> or url:<base_url>@<served_name>')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--dataset', default=None, help='original | verified-v2 | verified-aa-<version> '
                                                     '(default: verified-aa-v3)')
    ap.add_argument('--background', choices=['on', 'off'], default=None,
                    help='default: on for original (AA), off for Verified (#184)')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--limit', type=int, default=None, help='a fixed random subset of N problems (smoke tests)')
    ap.add_argument('--problems', default='', help='comma-separated problem ids')
    ap.add_argument('--effort', default=None, help='reasoning effort (sets it wherever the registry pins one)')
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--extra-body', default=None, help='JSON merged into every request')
    ap.add_argument('--aa-slug', default=None, help='AA row to compare with (default: from served name + effort)')
    ap.add_argument('--context-window', type=int, default=131072, help='prompt + output cap (M3.0: 128k)')
    ap.add_argument('--timeout', type=int, default=None, help='grading timeout per step script (default: 300 '
                                                              'original / 1800 Verified)')
    ap.add_argument('--envs', default=None, help='grading envs label=sif:<path>|local:<python>,... (default: the two SIFs)')
    ap.add_argument('--concurrency', type=int, default=512, help='max requests in flight (the server gate decides)')
    ap.add_argument('--grade-jobs', type=int, default=None, help='parallel grading scripts (default: cpus/2)')
    ap.add_argument('--request-timeout', type=int, default=4 * 3600)
    ap.add_argument('--dispatch', metavar='DIR', help='#344: each chain takes one slot of this dispatcher\'s shared '
                    'in-flight budget while it generates (aa_evals.common.dispatch), longest predicted first')
    ap.add_argument('--chain-cap-sec', type=float, default=None, help='#344: wall cap on a chain\'s generation (2 h: '
                    '7200); steps not generated by then are model errors (chain_cap, scored 0)')
    ap.add_argument('--wait-server', type=int, default=1800, help='seconds to wait for a fresh registry')
    ap.add_argument('--note', default=None)
    ap.add_argument('--force-config', action='store_true')
    ap.add_argument('--upgrade-from', default=None, help='seed this new run (on a newer --dataset) from a finished '
                                                         'run: only the steps the patches touch are regenerated or '
                                                         'regraded; the rest is carried over (upgrade.json)')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--regrade', action='store_true')
    ap.add_argument('--condor', action='store_true', help='run as a CPU condor job (grading needs a compute node)')
    # c30 (#378): the 32-CPU client averaged 3.2 cores; grading runs beside generation (2 threads per grade job)
    ap.add_argument('--cpus', type=int, default=8, help='condor job CPUs (grade jobs = CPUs / 2)')
    ap.add_argument('--mem-gb', type=int, default=32)
    ap.add_argument('--cpu-bid', type=int, default=None, help='fixed CPU bid; default: the CPU bid ladder 26 -> 36 -> 56 (aa_evals/common/condor_cost.py)')
    a = ap.parse_args(argv)
    a.dataset = a.dataset or (json.loads((RUNS / a.run_id / 'config.json').read_text())['dataset']['dataset']
                              if (a.summarize or a.regrade) and (RUNS / a.run_id / 'config.json').exists()
                              else 'verified-aa-v3')
    a.grade_jobs = a.grade_jobs or max(2, (int(os.environ.get('OMP_NUM_THREADS_CONDOR', 0)) or a.cpus) // 2)
    run_dir = RUNS / a.run_id
    if a.summarize:
        print(json.dumps(summarize(run_dir), indent=1))
        return
    if a.condor:
        submit_condor(a, argv)
        return
    if a.regrade:
        cfg = json.loads((run_dir / 'config.json').read_text())
        grader.selftest(cfg['grading']['envs'], grader.scratch_dir())
        asyncio.run(regrade(a, run_dir))
        return
    if a.upgrade_from:   # the source run fixes the model, settings and problem set; --dataset is the new version
        src = json.loads((RUNS / a.upgrade_from / 'config.json').read_text())
        if a.model and a.model != src['model']['spec']:
            raise SystemExit('--upgrade-from: --model differs from the source run')
        a.model = src['model']['spec']
        a.problems = a.problems or ','.join(src['problems'])
        a.repeats, a.context_window = src['repeats'], src['generation']['context_window']
        a.background = 'on' if src['background'] else 'off'
        a.aa_slug = a.aa_slug or src['model'].get('aa_slug')
        a.timeout = a.timeout or src['grading']['timeout_s']
        a.envs = a.envs or ','.join(f'{k}={v}' for k, v in src['grading']['envs'].items())
        a.effort = a.effort or presets.effective_effort(src['generation']['request'])
    if not a.model:
        raise SystemExit('--model is required')
    ds = data.load(a.dataset)
    probs = select(ds, a)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(a, run_dir, ds, probs)
    if a.upgrade_from:
        cfg['upgraded_from'] = a.upgrade_from
        atomic_json(run_dir / 'config.json', cfg)
        if not (run_dir / 'upgrade.json').exists():
            info = seed_upgrade(a.upgrade_from, run_dir, ds, cfg)
            print(f'upgrade from {a.upgrade_from}: {json.dumps(info["steps"])}; problems touched '
                  f'{",".join(info["problems_touched"]) or "none"}', flush=True)
    print('grading envs:', grader.selftest(cfg['grading']['envs'], grader.scratch_dir()), flush=True)
    print(f'{a.run_id}: {len(probs)} problems x {a.repeats} repeats, dataset {ds.name}, background '
          f'{cfg["background"]}, model {cfg["model"]["served_name"]}, request {json.dumps(cfg["generation"]["request"])}',
          flush=True)
    asyncio.run(Runner(a, run_dir, cfg, ds).run(probs))
    s = summarize(run_dir)
    print(json.dumps({k: s[k] for k in ('score', 'main_problem_accuracy', 'sub_step_accuracy', 'n_infra_errors',
                                        'n_cutoff', 'valid')}, indent=1))


if __name__ == '__main__':
    main()

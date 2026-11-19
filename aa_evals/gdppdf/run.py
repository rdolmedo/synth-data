"""GDP.pdf as AA runs it: generate (one turn, no tools) -> grade (per-criterion judge) -> summarize.

  python -m aa_evals.gdppdf.run --model serving:aa-super --run-id smoke --preset super-vl \
      --judge serving:aa-super --limit 5 --repeats 2

Heavy runs go through condor: `python -m aa_evals.gdppdf.submit run --run-id ID -- <these args>`.

Outputs in /data/aai/aa_evals/runs/gdppdf/<run_id>/:
  config.json        code SHA, model spec, generation settings, dataset, LiteParse prep, judge, limits
  generations.jsonl  one line per attempt (append-only, last line per key wins): answer, reasoning, usage,
                     input plan (DPI, composites, coverage), status, error kind, retries   <- the transcripts
  grades.jsonl       one line per graded attempt: every criterion's verdict and rationale
  items.jsonl        one row per item x repeat: item_id, repeat, score (all-pass 0/1), mean_pass, status,
                     transcript (generations.jsonl key)
  summary.json       All-pass on AA's 0-100 scale, Mean Pass, domain cuts, ci95, infra errors, AA reference

Statuses: ok (graded); model_error (scores 0: empty or truncated-before-answer output, input too long even as
text); infra_error (excluded and counted: endpoint/transport failures after retries, a criterion with no
verdict). Every stage resumes: rerunning the same command redoes only missing or infra-failed work.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common import aai_open, protocol
from aa_evals.gdppdf import data, inputs, metrics, prepare
from aa_evals.judges import pins
from aa_evals.tracejudge import fold_summary
from aa_evals.gdppdf.client import Client, ContextRejected, InfraError
from aa_evals.gdppdf.judge import Judge

# AA_GDPPDF_RUNS overrides the runs root (keep it on /data; never /home).
RUNS = Path(os.environ.get('AA_GDPPDF_RUNS') or '/data/aai/aa_evals/runs/gdppdf')
AA_LEADERBOARD = Path('/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json')
DEFAULT_JUDGE = 'serving:aa-judge'   # the shared open judge endpoint (user 2026-11-02, #34: no closed API judges)
BENCH = 'gdppdf'
# Under the M3.0 128k cap (#173) the output reserve is the floor on max_tokens that input planning keeps free;
# max_tokens itself is the window minus the planned prompt (an overestimate of the real one), so short documents
# leave the model more. 32k: Qwen3.8 medium's max completion was 34k (3/500 above 32k), xhigh's p90 24k (#34 r3).
# With 64k, 20 of the 100 documents' text alone would not fit the remaining 64k (scored 0); with 32k, 9 (#34).
CAPPED_OUTPUT_RESERVE = 32768

# Generation presets: request body + the AA leaderboard entry we compare against. A serving: model without a preset
# here (and no --preset) uses its registry's request.overrides (the lab's sampling, thinking on; M3.0 protocol).
PRESETS = {
    'super-vl': {
        'aa_slug': 'nemotron-3-5-super-vl',
        'body': {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 'rest',
                 'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}},
        'output_reserve': 65536,
        'note': "AA lists one Super VL variant (reasoning). Temperature 0.6 = AA's reasoning default and "
                "NVIDIA's vision recommendation (top_p 0.95). NVIDIA discloses no output cap: the output gets the "
                "rest of the context window, as for every model (M3.2; before: a fixed 65,536).",
    },
    # AA's "Qwen3.8 27B (Xhigh)". Sampling = the lab's recommendation as pinned in the aa-qwen registry (AA's rule).
    # max_tokens 'rest': no cap is sent, so vLLM allows the rest of the context (the creator's limit, as in hle);
    # images are planned so that at least `output_reserve` tokens remain for the answer.
    'qwen38-xhigh': {
        'aa_slug': 'qwen3-8-27b',
        'body': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
                 'repetition_penalty': 1.0, 'reasoning_effort': 'xhigh', 'max_tokens': 'rest',
                 'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'}},
        'output_reserve': 65536,
        'note': "Qwen's recommended sampling (T 1.0, top_p 0.95, top_k 20) at reasoning effort xhigh. Output: the "
                "rest of the 262,144-token context; page images are reduced so that >= 65,536 tokens remain.",
    },
    # AA's "Qwen3.8 27B (Medium)": the same, at reasoning effort medium.
    'qwen38-medium': {
        'aa_slug': 'qwen3-8-27b-medium',
        'body': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
                 'repetition_penalty': 1.0, 'reasoning_effort': 'medium', 'max_tokens': 'rest',
                 'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}},
        'output_reserve': 65536,
        'note': "As qwen38-xhigh, at reasoning effort medium (AA's Qwen3.8 27B (Medium)).",
    },
}
# Served name -> preset when --preset is not given: only where GDP.pdf deliberately departs from the registry.
DEFAULT_PRESET = {'nemotron-3.5-super-vl': 'super-vl'}   # NVIDIA's vision sampling (T 0.6), not the registry's 1.0


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def git_sha():
    frozen = Path(__file__).resolve().parents[2] / 'GIT_SHA'
    if frozen.exists():
        return frozen.read_text().strip()
    r = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def read_jsonl(path):
    """Last record per (task_id, repeat). A final line without its newline is a record still being appended by a
    concurrent job (generation runs while grading reads): it is skipped and picked up next time."""
    out = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip() and line.endswith('\n'):
                    r = json.loads(line)
                    out[(r['task_id'], r['repeat'])] = r
    return out


class Appender:
    def __init__(self, path):
        self.f = open(path, 'a')

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def aa_reference(slug):
    if not slug or not AA_LEADERBOARD.exists():
        return None
    d = json.loads(AA_LEADERBOARD.read_text())
    for m in d.get('models', []):
        if m.get('slug') == slug and 'gdp-pdf' in (m.get('scores') or {}):
            return {'slug': slug, 'name': m.get('name'), 'all_pass': round(100 * m['scores']['gdp-pdf'], 1),
                    'source': f'{AA_LEADERBOARD.name} ({d.get("fetched", "")})'}
    return None


# ---------------------------------------------------------------------------------------------------------
# Generation

class Gen:
    def __init__(self, a, run_dir, tasks, texts, client, limits, body, cap_window=None):
        self.a, self.run_dir, self.tasks, self.texts = a, run_dir, tasks, texts
        self.client, self.limits, self.body = client, limits, body
        # set when the endpoint's own context exceeds the run's window and no max_tokens is fixed: each request
        # then gets max_tokens = window - its planned prompt (an overestimate), so prompt + output <= window
        self.cap_window = cap_window
        self.out = Appender(run_dir / 'generations.jsonl')
        self.renderer = inputs.Renderer(workers=a.render_workers)
        self.task_sem = asyncio.Semaphore(a.task_concurrency)
        self.done = 0
        # #344 dispatcher (--dispatch DIR): a task takes one slot per repeat of the run's shared in-flight budget,
        # longest predicted first, instead of --task-concurrency
        self.lane, self.pred = None, None
        if getattr(a, 'dispatch', None):
            from aa_evals.common.dispatch.lane import open_dispatch
            self.lane, self.pred, _ = open_dispatch(a.dispatch, 'gdppdf', run_dir)
        self.request_cap = getattr(a, 'request_cap_sec', None)

    def record(self, task, repeat, status, plan=None, out=None, error_kind=None, error=None, t0=None):
        rec = {'task_id': task.task_id, 'repeat': repeat, 'domain': task.domain, 'status': status,
               'error_kind': error_kind, 'error': error, 'at': now(),
               'seconds': round(time.time() - t0, 1) if t0 else None,
               'plan': plan.describe() if plan else None}
        if out:
            rec.update({'answer': out['content'], 'reasoning': out['reasoning'], 'finish_reason': out['finish_reason'],
                        'usage': out['usage'], 'retries': out.get('attempts'), 'max_tokens': out.get('max_tokens'),
                        'response_budget': out.get('response_budget')})
        self.out.write(rec)
        self.done += 1
        u = (out or {}).get('usage') or {}
        print(f'[gen] {self.done} {task.task_id[:8]} r{repeat} {status}{"/" + error_kind if error_kind else ""} '
              f'mode={plan.mode if plan else "-"} dpi={plan.dpi if plan else "-"} k={plan.k if plan else "-"} '
              f'prompt={u.get("prompt_tokens")} (est {plan.est_prompt_tokens if plan else "-"}) '
              f'completion={u.get("completion_tokens")} finish={(out or {}).get("finish_reason")}', flush=True)

    async def attempt(self, task, repeat, plan, body, first_token=None):
        t0 = time.time()
        try:
            from tb_agents import response_budget as response_policy
            from tb_agents.response_budget_single import run_async as budgeted_call
            call = lambda b: self.client.chat(b, stream=True, first_token=first_token)
            request = budgeted_call(call, body, response_policy.settings()) if response_policy.settings() else call(body)
            if self.request_cap:   # #344 2 h cap: the whole model call (all its retries); at the cap, no answer
                out = await asyncio.wait_for(request, self.request_cap)
            else:
                out = await request
            out['max_tokens'] = body.get('max_tokens')   # None: no cap sent (the rest of the context)
        except InfraError as e:
            self.record(task, repeat, 'infra_error', plan, error_kind='endpoint', error=str(e)[:500], t0=t0)
            return
        except TimeoutError:
            if not self.request_cap:
                raise
            self.record(task, repeat, 'model_error', plan, error_kind='request_cap',
                        error=f'no answer within the {self.request_cap:g} s request cap', t0=t0)
            return
        if out['content'].strip():
            self.record(task, repeat, 'ok', plan, out, t0=t0)
        else:
            kind = 'truncated_before_answer' if out['finish_reason'] == 'length' else 'empty_answer'
            self.record(task, repeat, 'model_error', plan, out, error_kind=kind, t0=t0)

    async def task(self, task, repeats):
        slot = (self.lane.aslot(task.task_id, task.task_id, self.pred(task.task_id), weight=len(repeats))
                if self.lane else self.task_sem)
        async with slot:
            pages_text = [p['text'] for p in self.texts[task.task_id]['pages']]
            base_text = inputs.render_text(task.prompt, pages_text, task.pdf_name, None)
            try:
                base_tok = await self.client.count_tokens(base_text)
            except InfraError as e:
                for r in repeats:
                    self.record(task, r, 'infra_error', error_kind='tokenize', error=str(e)[:300])
                return
            limits = copy.copy(self.limits)
            start_dpi = None
            rejections = []
            while True:
                try:
                    plan, imgs, text = await asyncio.to_thread(inputs.plan_and_build, task, pages_text, limits,
                                                               base_tok, self.renderer, None, start_dpi)
                except Exception as e:   # rendering / probing failed: our side, not the model's
                    for r in repeats:
                        self.record(task, r, 'infra_error', error_kind='input_build', error=f'{type(e).__name__}: {e}'[:500])
                    return
                plan.rejections = rejections
                if plan.reason == 'input_too_long':
                    for r in repeats:
                        self.record(task, r, 'model_error', plan, error_kind='input_too_long',
                                    error=f'protocol limit, not infra: text alone is ~{plan.est_prompt_tokens} tokens, which leaves less '
                                          f'than the {limits.max_tokens}-token output floor in the {limits.max_model_len}-token '
                                          'context (README deviation 11)')
                    return
                content = await asyncio.to_thread(inputs.message_content, imgs, text)
                del imgs
                body = dict(self.body, messages=[{'role': 'user', 'content': content}])
                if self.cap_window and 'max_tokens' not in self.body:
                    body['max_tokens'] = self.cap_window - plan.est_prompt_tokens
                first = asyncio.Event()
                lead = asyncio.create_task(self._guarded(task, repeats[0], plan, body, first))
                waiter = asyncio.create_task(first.wait())
                await asyncio.wait({lead, waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                if lead.done() and isinstance(lead.result(), ContextRejected):
                    e = lead.result()
                    rejections.append({'mode': plan.mode, 'dpi': plan.dpi, 'k': plan.k, 'error': str(e)[:300]})
                    if e.max_images is not None and e.max_images < (limits.max_images or 10 ** 9):
                        limits.max_images = e.max_images
                    elif e.max_decoded_bytes and e.max_decoded_bytes != limits.max_decoded_bytes:
                        limits.max_decoded_bytes = e.max_decoded_bytes   # the server told us its cap: re-plan
                    elif plan.mode == 'images' and plan.dpi > inputs.DPI_LADDER[-1]:
                        start_dpi = max(d for d in inputs.DPI_LADDER if d < plan.dpi)
                    elif plan.mode == 'images':
                        limits.max_images = 0      # images rejected even at 72 DPI: text alone
                    else:
                        for r in repeats:
                            self.record(task, r, 'model_error', plan, error_kind='input_too_long', error=str(e)[:300])
                        return
                    continue
                rest = [asyncio.create_task(self._guarded(task, r, plan, body)) for r in repeats[1:]]
                results = await asyncio.gather(lead, *rest)
                late = [r for r, res in zip(repeats, results) if isinstance(res, ContextRejected)]
                if late:   # rejected although the lead attempt fit: record, never silently drop
                    for r in late:
                        self.record(task, r, 'infra_error', plan, error_kind='late_context_rejection')
                return

    async def _guarded(self, task, repeat, plan, body, first=None):
        try:
            await self.attempt(task, repeat, plan, body, first)
        except ContextRejected as e:
            return e
        return None

    async def run(self, todo):
        print(f'[gen] {sum(len(r) for _, r in todo)} attempts over {len(todo)} tasks; limits {self.limits}', flush=True)
        if self.lane:
            self.lane.expect([(t.task_id, t.task_id, self.pred(t.task_id)) for t, _ in todo])
        try:
            await asyncio.gather(*(self.task(t, r) for t, r in todo))
        finally:
            self.out.close()
            self.renderer.close()
            if self.lane:
                self.lane.close()


# ---------------------------------------------------------------------------------------------------------

async def stage_generate(a, run_dir, tasks, cfg):
    texts = prepare.load_texts()
    missing = [t.task_id for t in tasks if t.task_id not in texts]
    if missing:
        sys.exit(f'{len(missing)} tasks have no LiteParse text in {prepare.TEXTS}; run prepare first')
    prev = read_jsonl(run_dir / 'generations.jsonl')
    todo = []
    for t in tasks:
        reps = [r for r in range(a.repeats) if prev.get((t.task_id, r), {}).get('status') not in ('ok', 'model_error')]
        if reps:
            todo.append((t, reps))
    if not todo:
        print('[gen] nothing to do', flush=True)
        return
    # under a dispatcher the lane bounds the requests; the client's own semaphore must not
    client = Client(a.model, concurrency=4096 if a.dispatch else a.task_concurrency * a.repeats + 4,
                    timeout=a.request_timeout)
    try:
        lim = await client.limits()
        ep_ctx = a.max_model_len or lim['max_model_len']
        win, why = protocol.window(ep_ctx, a.context_window)
        cfg['context'] = {'window': win, 'rule': why, 'endpoint_max_model_len': ep_ctx, 'requested': a.context_window,
                          'output_reserve': cfg['generation'].get('output_reserve')}
        images, images_why = protocol.image_inputs(a.model)
        vision = (cfg.get('inputs') or {}).get('vision', 'auto')
        if a.max_images is not None:
            max_images = a.max_images
        elif images is False or vision == 'off':   # AA's text-alone path (the full LiteParse text, no images)
            max_images = 0
        else:
            max_images = lim['max_images'] if lim['max_images'] is not None else 256
        cfg['image_inputs'] = {'endpoint': images, 'why': images_why, 'vision': vision, 'sent': max_images > 0}
        limits = inputs.Limits(max_model_len=win,
                               max_tokens=cfg['generation'].get('output_reserve') or cfg['generation']['max_tokens'],
                               max_images=max_images,
                               max_payload_mb=a.max_payload_mb,
                               max_decoded_bytes=(a.max_decoded_bytes if a.max_decoded_bytes is not None else
                                                  lim['max_decoded_bytes'] or inputs.VLLM_DECODE_LIMIT),
                               image_rule=a.image_rule or inputs.image_rule_for(client.ep.served_name, client.ep.kind))
        if limits.image_rule == 'probe':
            from aa_evals.gdppdf.client import root_url
            limits.image_tokens = inputs.ProbeRule(root_url(client.ep.urls[0]), client.ep.served_name, client._headers())
        if not limits.max_model_len:
            sys.exit('cannot discover the endpoint context; pass --max-model-len')
        cfg['limits'] = {k: v for k, v in limits.__dict__.items() if k != 'image_tokens'}
        cfg['endpoint'] = client.ep.describe()
        write_config(run_dir, cfg)
        cap = win if win and ep_ctx and ep_ctx > win else None
        await Gen(a, run_dir, tasks, texts, client, limits, cfg['generation']['body'], cap_window=cap).run(todo)
    finally:
        await client.close()


def judge_key(spec, settings):
    """A judge's identity: spec + a hash of its request settings. Grades made under other settings are stale."""
    h = hashlib.sha1(json.dumps(settings or {}, sort_keys=True).encode()).hexdigest()[:10]
    return f'{spec}#{h}'


def current_judge_key(a, cfg):
    j = cfg.get('judge') or {}
    return judge_key(a.judge, j['settings']) if j.get('spec') == a.judge and 'settings' in j else None


async def stage_grade(a, run_dir, tasks, cfg):
    gens = read_jsonl(run_dir / 'generations.jsonl')
    prev = read_jsonl(run_dir / 'grades.jsonl')
    by_id = {t.task_id: t for t in tasks}
    # --judge names the judge (its identity in grade keys); --judge-endpoint, if given, is where its requests go (the
    # same weights served elsewhere, e.g. to offload a busy shared judge). Settings come from the endpoint's registry.
    judge = Judge(a.judge_endpoint or a.judge, concurrency=a.judge_concurrency,
                  extra=json.loads(a.judge_extra_body) if a.judge_extra_body else None)
    info = await judge.setup()
    info['endpoint'] = info['spec']
    info['spec'] = a.judge
    info['arm'] = a.judge_arm
    cfg['judge'] = info
    write_config(run_dir, cfg)
    key = judge_key(a.judge, cfg['judge']['settings'])
    todo = [g for k, g in gens.items() if g['status'] == 'ok' and k[0] in by_id
            and not (prev.get(k, {}).get('judge_key') == key and prev[k].get('n_unscored') == 0
                     and prev[k].get('answer_at') == g['at'])]
    if not todo:
        print('[grade] nothing to do', flush=True)
        await judge.close()
        return
    print(f'[grade] {len(todo)} attempts, judge {key}: {json.dumps(cfg["judge"]["settings"])}', flush=True)
    out = Appender(run_dir / 'grades.jsonl')
    try:
        sem = asyncio.Semaphore(max(1, a.judge_concurrency // 8))
        n = 0

        async def one(g):
            nonlocal n
            async with sem:
                t0 = time.time()
                res = await judge.grade(by_id[g['task_id']], g['answer'])
                rec = {'task_id': g['task_id'], 'repeat': g['repeat'], 'judge': a.judge, 'judge_key': key, 'answer_at': g['at'],
                       'at': now(), 'seconds': round(time.time() - t0, 1), **res}
                out.write(rec)
                n += 1
                print(f'[grade] {n}/{len(todo)} {g["task_id"][:8]} r{g["repeat"]} passed {res["n_passed"]}/'
                      f'{res["n_criteria"]} unscored {res["n_unscored"]} all_pass={res["all_pass"]}', flush=True)

        await asyncio.gather(*(one(g) for g in todo))
    finally:
        out.close()
        await judge.close()


def stage_summarize(a, run_dir, tasks, cfg):
    gens = read_jsonl(run_dir / 'generations.jsonl')
    grades = read_jsonl(run_dir / 'grades.jsonl')
    jkey = current_judge_key(a, cfg)   # only grades made by this judge with its recorded settings count
    items = []
    for t in tasks:
        for r in range(a.repeats):
            g = gens.get((t.task_id, r))
            if g is None:
                if a.cutoff:   # no answer at the run deadline: counts as not correct (overnight CUTOFF rule)
                    items.append({'item_id': t.task_id, 'repeat': r, 'domain': t.domain, 'status': 'cutoff',
                                  'error_kind': 'deadline', 'score': 0.0, 'mean_pass': 0.0, 'plan': {}})
                continue
            row = {'item_id': t.task_id, 'repeat': r, 'domain': t.domain, 'transcript':
                   f'{run_dir}/generations.jsonl#{t.task_id}/{r}', 'error_kind': g.get('error_kind'),
                   'plan': {k: (g.get('plan') or {}).get(k) for k in ('mode', 'dpi', 'k', 'covered', 'n_images', 'jpeg')},
                   'prompt_tokens': (g.get('usage') or {}).get('prompt_tokens'),
                   'completion_tokens': (g.get('usage') or {}).get('completion_tokens'),
                   'finish_reason': g.get('finish_reason')}
            if g['status'] == 'model_error':
                row.update(status='model_error', score=0.0, mean_pass=0.0)
                u = g.get('usage') or {}
                rt = (u.get('completion_tokens_details') or {}).get('reasoning_tokens')
                # the turn ended inside or right at the end of the reasoning, with no answer: EOS before </think>
                # (hle's definition), or '</think>' [whitespace] '<|im_end|>' straight after a reasoning cut
                # mid-sentence (Qwen3.8: 2-3 closing tokens seen); at most 4 non-reasoning tokens
                if (g.get('finish_reason') == 'stop' and not (g.get('answer') or '').strip() and rt is not None
                        and rt >= (u.get('completion_tokens') or 0) - 4):
                    row['reasoning_unclosed'] = True
                    row['error_kind'] = 'reasoning_unclosed'
            elif g['status'] == 'infra_error':
                row.update(status='infra_error', score=None, mean_pass=None)
            else:
                gr = grades.get((t.task_id, r))
                matching_grade = gr and gr.get('judge_key') == jkey and gr.get('answer_at') == g['at']
                if matching_grade and gr['n_unscored'] == 0:
                    row.update(status='ok', score=gr['all_pass'], mean_pass=gr['mean_pass'],
                               n_passed=gr['n_passed'], n_criteria=gr['n_criteria'])
                elif a.cutoff:   # an answer without a complete grade at the deadline counts as not correct
                    row.update(status='ungraded', score=0.0, mean_pass=0.0,
                               error_kind='ungraded' if not gr else 'judge_unscored')
                elif matching_grade:
                    row.update(status='infra_error', score=None, mean_pass=None,
                               error_kind='judge_unscored')
                else:
                    row.update(status='pending', score=None, mean_pass=None, error_kind=None)
            items.append(row)
    with open(run_dir / 'items.jsonl', 'w') as f:
        for row in items:
            f.write(json.dumps(row) + '\n')
    # Task sets (#348): the headline is the run's own set (runs before #348: 'all'); an 'all' run also reports the
    # text-only set, scored from the same rows.
    headline = run_task_set(cfg)
    sets = {}
    for name in ([headline] + [n for n in data.TASK_SETS if n != headline] if headline == 'all' else [headline]):
        keep = {t.task_id for t in data.task_set(tasks, name)}
        sets[name] = metrics.summarize([r for r in items if r['item_id'] in keep], len(keep), a.repeats)
    s = sets[headline]
    kinds = {}
    for row in items:
        if row['status'] not in ('ok', 'pending'):
            kinds[f'{row["status"]}/{row["error_kind"]}'] = kinds.get(f'{row["status"]}/{row["error_kind"]}', 0) + 1
    modes = {}
    for row in items:
        p = row['plan']
        key = 'text' if p.get('mode') == 'text' else f'images k={p.get("k")} dpi={p.get("dpi")}' + (' jpeg' if p.get('jpeg') else '') if p.get('mode') else 'none'
        modes[key] = modes.get(key, 0) + 1
    tok = [row['completion_tokens'] for row in items if row.get('completion_tokens')]
    ptok = [row['prompt_tokens'] for row in items if row.get('prompt_tokens')]
    jin = jout = jcalls = 0
    for gr in grades.values():
        if gr.get('judge_key') != jkey:
            continue
        for c in gr['grades']:
            for u in c.get('usage') or []:
                jcalls += 1
                jin += u.get('prompt_tokens') or 0
                jout += u.get('completion_tokens') or 0
    summary = {'benchmark': 'GDP.pdf', 'metric': 'All-pass', **s, 'n_items': len(tasks), 'n_repeats': a.repeats,
               'grading_code_sha': (cfg.get('stage_git') or {}).get('grade'),
               'error_kinds': kinds, 'input_modes': modes,
               'mean_prompt_tokens': round(sum(ptok) / len(ptok)) if ptok else None,
               'mean_completion_tokens': round(sum(tok) / len(tok)) if tok else None,
               'truncated_with_answer': sum(1 for row in items if row.get('finish_reason') == 'length' and row['status'] == 'ok'),
               'n_reasoning_unclosed': sum(bool(row.get('reasoning_unclosed')) for row in items),
               'k_repeats': a.repeats, 'deadline_mode': bool(a.cutoff),
               'ci95_method': 'normal approximation over task means (attempts clustered by task), 95%, '
                              f'n_tasks={len(tasks)}, k={a.repeats} repeats per task',
               'judge_usage': {'calls': jcalls, 'prompt_tokens': jin, 'completion_tokens': jout},
               'task_set': headline,
               'task_sets': {name: {k: v[k] for k in ('score', 'ci95', 'mean_pass', 'mean_pass_ci95', 'n_attempts_expected',
                                                      'n_tasks_scored', 'valid') if k in v} | {
                                 'n_items': len(data.task_set(tasks, name)),
                                 'status': 'headline' if name == headline else 'reported'}
                             for name, v in sets.items()},
               'n_over_window': None if not (cfg.get('context') or {}).get('window') else sum(
                   (row.get('prompt_tokens') or 0) + (row.get('completion_tokens') or 0) > cfg['context']['window']
                   for row in items),
               'judge_arm': (cfg.get('judge') or {}).get('arm'),
               **protocol.summary_fields(model=a.model, served_name=(cfg.get('endpoint') or {}).get('served_name'),
                   body=cfg['generation']['body'], context=cfg.get('context'),
                   judge_arm=(cfg.get('judge') or {}).get('arm'), judge_settings_key=jkey,
                   code_sha=cfg.get('generation_git_sha') or cfg.get('git_sha'), repeats=a.repeats,
                   image_inputs=(cfg.get('image_inputs') or {}).get('sent', (cfg.get('image_inputs') or {}).get('endpoint'))),
               'model': a.model, 'served_name': (cfg.get('endpoint') or {}).get('served_name'),
               'judge': a.judge, 'judge_key': jkey, 'judge_settings': (cfg.get('judge') or {}).get('settings'), 'aa_reference': aa_reference(cfg.get('aa_slug')), 'run_id': a.run_id,
               'subset': cfg['selection'], 'updated_at': now(), 'aai_open_version': aai_open.of_config(cfg)}
    summary = fold_summary(run_dir, summary)   # n_flagged: the M3.4 trace judge (#177)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def run_task_set(cfg):
    """The task set a run was drawn from ('all' for runs before #348)."""
    return (cfg.get('selection') or {}).get('task_set', 'all')


def write_config(run_dir, cfg):
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=1, default=str) + '\n')


def registry_request(spec):
    """(served name, request block) from a serving: spec's registry, read without a liveness check."""
    kind, _, name = spec.partition(':')
    if kind == 'serving':
        try:
            from aa_evals.common import serving
            reg = json.loads((serving.registry_root(protocol.SERVING_ROOT) / name / 'endpoints.json').read_text())
            return reg.get('served_name'), reg.get('request') or {}
        except (OSError, ValueError):
            pass
    if kind == 'url':
        return spec.rpartition('@')[2], {}
    return None, {}


def build_config(a, run_dir, tasks):
    old = run_dir / 'config.json'
    if old.exists() and a.stage in ('grade', 'summarize'):
        prev = json.loads(old.read_text())
        if a.model != prev['model'] or a.repeats != prev['repeats']:
            sys.exit(f'{old}: grading must retain the stored model and repeats; use a new run ID for a source copy')
        cfg = copy.deepcopy(prev)
        cfg['generation_git_sha'] = prev.get('generation_git_sha') or prev.get('git_sha')
        cfg['git_sha'] = git_sha()
        cfg.setdefault('stage_git', {})[a.stage] = cfg['git_sha']
        cfg['argv'] = sys.argv
        cfg['selection'] = {'limit': a.limit, 'task_ids': [t.task_id for t in tasks],
                            **({'task_set': a.task_set} if a.task_set != 'all' else {})}
        if (prev.get('judge') or {}).get('spec') != a.judge:
            cfg['judge'] = {'spec': a.judge, 'arm': a.judge_arm}
        return cfg
    served, request = registry_request(a.model)
    if not a.preset and served in DEFAULT_PRESET:
        a.preset = DEFAULT_PRESET[served]
    preset = PRESETS.get(a.preset, {}) if a.preset else {}
    if preset:
        body = copy.deepcopy(preset.get('body', {}))
    else:   # the M3.0 protocol: the registry's lab sampling, thinking and default effort; output = rest of window
        body = {**copy.deepcopy(request.get('overrides') or {}), 'max_tokens': 'rest'}
        if not request.get('overrides'):
            body.setdefault('temperature', 0.6)   # AA's default for reasoning models when the lab gives none
    if a.reasoning_effort is not None:
        try:
            body = protocol.apply_effort(body, a.reasoning_effort)
        except ValueError as e:
            sys.exit(f'--reasoning-effort {a.reasoning_effort}: {e}')
    if a.temperature is not None:
        body['temperature'] = a.temperature
    if a.top_p is not None:
        body['top_p'] = a.top_p
    if a.max_tokens is not None:
        body['max_tokens'] = a.max_tokens if a.max_tokens == 'rest' else int(a.max_tokens)
    if a.extra_body:
        body.update(json.loads(a.extra_body))
    body.setdefault('max_tokens', 65536)
    if body['max_tokens'] == 'rest':     # no cap sent: the rest of the context; plan inputs with a reserve
        body.pop('max_tokens')
        reserve = a.output_reserve or (CAPPED_OUTPUT_RESERVE if a.context_window else
                                       preset.get('output_reserve') or 65536)
        max_tokens = None
    else:
        max_tokens = reserve = int(body['max_tokens'])
        body['max_tokens'] = max_tokens
    prep_manifest = prepare.PREP_DIR / 'manifest.json'
    cfg = {'benchmark': 'GDP.pdf', 'run_id': a.run_id, 'git_sha': git_sha(), 'model': a.model,
           'preset': a.preset, 'served_name': served, 'effort': protocol.effort_of(body),
           'aa_slug': a.aa_slug or protocol.aa_model(served, body)[1] or preset.get('aa_slug'),
           'preset_note': preset.get('note'),
           'generation': {'body': body, 'max_tokens': max_tokens, 'output_reserve': reserve, 'stream': True, 'turns': 1, 'tools': None,
                          'system_prompt': None},
           'dataset': data.dataset_info(),
           'prep': json.loads(prep_manifest.read_text()) if prep_manifest.exists() else {'prep_tag': prepare.PREP_TAG},
           'inputs': {'vision': a.vision, 'dpi_ladder': inputs.DPI_LADDER, 'composites': inputs.COMPOSITE_STEPS,
                      'jpeg_quality': inputs.JPEG_QUALITY, 'max_payload_mb': a.max_payload_mb,
                      'renderer': 'LiteParse 2.5.0 screenshot (pdfium), RGBA flattened on white, PNG'},
           'judge': {'spec': a.judge, 'arm': a.judge_arm}, 'repeats': a.repeats,
           # the set is recorded only when it is not 'all', so configs from before #348 still match
           'selection': {'limit': a.limit, 'task_ids': [t.task_id for t in tasks],
                         **({'task_set': a.task_set} if a.task_set != 'all' else {})},
           'harness': 'aa_evals/gdppdf (AA-style: LiteParse text' + (' + page images' if a.vision == 'auto' else ' alone')
                      + ', one turn, per-criterion judge)',
           'argv': sys.argv, 'started_at': now(),
           'dispatch': a.dispatch and str(Path(a.dispatch).resolve()), 'request_cap_sec': a.request_cap_sec}
    if old.exists():
        prev = json.loads(old.read_text())
        for k in ('model', 'repeats'):
            if prev.get(k) != cfg[k] and not a.force:
                sys.exit(f'{old}: {k} was {prev.get(k)!r}, now {cfg[k]!r}; use a new --run-id (or --force)')
        if prev.get('generation', {}).get('body') != body and not a.force:
            sys.exit(f'{old}: generation settings changed; use a new --run-id (or --force)')
        if run_task_set(prev) != a.task_set and not a.force:
            sys.exit(f'{old}: started on --task-set {run_task_set(prev)}; pass --task-set {run_task_set(prev)} to '
                     'resume it (or use a new --run-id)')
        prev_vision = (prev.get('inputs') or {}).get('vision', 'auto')   # runs before #348 sent page images
        if prev_vision != a.vision and not a.force:
            sys.exit(f'{old}: started with --vision {prev_vision}; pass --vision {prev_vision} to resume it '
                     '(or use a new --run-id)')
        for k in ('limits', 'endpoint'):
            if k in prev:
                cfg[k] = prev[k]
        if (prev.get('judge') or {}).get('spec') == a.judge:   # keep the recorded settings (summarize-only runs)
            cfg['judge'] = prev['judge']
        cfg['started_at'] = prev.get('started_at', cfg['started_at'])
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=None, help='serving:<name> | url:<base>@<served_name> | openai:<model>')
    ap.add_argument('--generations-from', default=None, help='re-grade another run\'s stored generations (a run id or '
                    'dir): copies its generations.jsonl here, takes its model, tasks and repeats, and runs grade + '
                    'summarize only (judge studies, #41)')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int, default=None, help='first N tasks of a domain-stratified order')
    ap.add_argument('--task-ids', default=None, help='comma-separated task ids (overrides --limit)')
    ap.add_argument('--repeats', type=int, default=None, help='default 5')
    ap.add_argument('--preset', choices=sorted(PRESETS), default=None)
    ap.add_argument('--aa-slug', default=None, help='AA leaderboard slug for aa_reference')
    ap.add_argument('--reasoning-effort', default=None, help='set wherever the model request carries an effort (top '
                    'level and chat_template_kwargs); refused for models without effort levels')
    ap.add_argument('--context-window', type=int, default=protocol.CONTEXT_WINDOW,
                    help='total tokens per request, prompt + output (M3.0: 131072 for regular evals; 0 = the '
                         "endpoint's full context, for Qwen3.8 calibration runs only)")
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--top-p', type=float, default=None)
    ap.add_argument('--max-tokens', default=None, help="an int, or 'rest' (no cap: the rest of the context)")
    ap.add_argument('--output-reserve', type=int, default=None, help="with max_tokens 'rest': tokens kept free "
                    f"for the output when planning the inputs (default {CAPPED_OUTPUT_RESERVE} under a context cap; "
                    "else the preset's, else 65536)")
    ap.add_argument('--extra-body', default=None, help='JSON merged into the request body')
    ap.add_argument('--judge', default=DEFAULT_JUDGE, help='an open self-hosted judge (serving:/url:); closed API '
                    'judges are not approved')
    ap.add_argument('--judge-endpoint', default=None, help='send judge requests here instead of --judge (same '
                    'weights served elsewhere); grades keep the identity of --judge, the endpoint is recorded')
    ap.add_argument('--judge-extra-body', default=None, help='JSON merged into every judge request')
    ap.add_argument('--judge-arm', default=None, help='a judge arm from aa_evals/judges/arms.py (e.g. '
                    'qwen38-27b-medium-pin): sets --judge to its server and its settings as the extra body; '
                    f'recorded as judge.arm. Default: the pinned arm for GDP.pdf, {pins.default_arm(BENCH)}, '
                    'unless --judge names another judge')
    ap.add_argument('--allow-external-api', action='store_true', help='required for openai: specs (closed external API; not approved for judging)')
    ap.add_argument('--stage', choices=['all', 'generate', 'grade', 'summarize'], default='all')
    ap.add_argument('--max-model-len', type=int, default=None, help='override the discovered context')
    ap.add_argument('--max-images', type=int, default=None, help='images per request (0 = text only)')
    ap.add_argument('--task-set', choices=data.TASK_SETS, default=data.DEFAULT_TASK_SET,
                    help="text-only-v1 (default, #348): drops the 25 tasks whose answers need a figure, map, plan, "
                         "photo or handwriting; 'all': the 100 (runs before #348). --limit/--task-ids select within it")
    ap.add_argument('--vision', choices=['off', 'auto'], default='off',
                    help="off (default, #348): every model gets the LiteParse text alone, AA's text-only path; auto: "
                         'page images when the serving launch takes images (runs before #348)')
    ap.add_argument('--max-payload-mb', type=float, default=100.0)
    ap.add_argument('--max-decoded-bytes', type=int, default=None,
                    help='server cap on decoded image bytes per request (W*H*4 summed); default: the serving launch '
                         'env, else vLLM\'s default 715827880; 0 = none')
    ap.add_argument('--image-rule', choices=sorted(inputs.IMAGE_RULES), default=None)
    ap.add_argument('--task-concurrency', type=int, default=4)
    ap.add_argument('--judge-concurrency', type=int, default=64)
    ap.add_argument('--render-workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--request-timeout', type=float, default=10800)
    ap.add_argument('--dispatch', metavar='DIR', help='#344: one slot per repeat of this dispatcher\'s shared '
                    'in-flight budget per task (aa_evals.common.dispatch), longest predicted first; replaces '
                    '--task-concurrency for generation')
    ap.add_argument('--request-cap-sec', type=float, help='#344: wall cap on one model call including retries; at '
                    'the cap the attempt has no answer (model_error request_cap)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--cutoff', action='store_true', help='deadline mode: attempts with no generation count as 0 '
                    '(n_cutoff) and answers without a complete grade count as 0 (n_ungraded), instead of being '
                    'missing or infra errors')
    a = ap.parse_args(argv)
    if a.judge_arm is None and a.judge == DEFAULT_JUDGE:
        a.judge_arm = pins.default_arm(BENCH)   # the benchmark's pinned judge (#41 audit 1, #175)
    if a.judge_arm:
        from aa_evals.judges.arms import apply_arm
        a.judge, a.judge_extra_body = apply_arm(a.judge_arm, a.judge, a.judge_extra_body, DEFAULT_JUDGE)

    src_cfg = None
    run_dir = RUNS / a.run_id
    if a.generations_from:
        src = Path(a.generations_from) if '/' in a.generations_from else RUNS / a.generations_from
        src_cfg = json.loads((src / 'config.json').read_text())
        if a.stage in ('all', 'generate'):
            a.stage = 'grade'
        a.model = a.model or src_cfg['model']
        a.repeats = a.repeats or src_cfg['repeats']
        a.preset = a.preset or src_cfg.get('preset')
        a.aa_slug = a.aa_slug or src_cfg.get('aa_slug')
        if not a.task_ids and not a.limit:
            a.task_ids = ','.join(src_cfg['selection']['task_ids'])
        a.task_set = run_task_set(src_cfg)   # re-grade the source's own selection
        run_dir.mkdir(parents=True, exist_ok=True)
        dst = run_dir / 'generations.jsonl'
        src_bytes = (src / 'generations.jsonl').read_bytes()
        src_bytes = src_bytes[:src_bytes.rfind(b'\n') + 1]   # complete lines only (the source may be mid-append)
        if not dst.exists():
            dst.write_bytes(src_bytes)
        elif dst.read_bytes() != src_bytes:
            # the source is append-only: a copy that is a prefix of it is just older (new repeats), so refresh it;
            # grades are keyed by answer time, so existing ones stay valid. Anything else needs --force.
            if src_bytes.startswith(dst.read_bytes()) or a.force:
                dst.write_bytes(src_bytes)
            else:
                sys.exit(f'{dst} differs from {src}/generations.jsonl; use a new --run-id (or --force)')
    elif a.stage in ('grade', 'summarize') and (run_dir / 'config.json').exists():
        stored = json.loads((run_dir / 'config.json').read_text())
        a.model = a.model or stored['model']
        a.repeats = a.repeats or stored['repeats']
        if a.task_ids is None and a.limit is None:
            a.task_ids = ','.join(stored['selection']['task_ids'])
        a.task_set = run_task_set(stored)
    if not a.model:
        sys.exit('--model is required (or --generations-from)')
    if a.vision == 'off' and a.max_images:
        sys.exit('--max-images > 0 sends page images: pass --vision auto with it')
    a.repeats = a.repeats or 5
    for spec in (a.model, a.judge, a.judge_endpoint or ''):
        if spec.startswith('openai:') and not a.allow_external_api:
            sys.exit(f'{spec} is an external API (spend needs the user\'s approval): pass --allow-external-api')
    run_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = data.load_tasks()
    if a.task_ids and (out := [i for i in a.task_ids.split(',') if i in data.TEXT_ONLY_EXCLUDED]) \
            and a.task_set != 'all':
        sys.exit(f'--task-ids {out} are excluded by --task-set {a.task_set} (#348); pass --task-set all')
    tasks = data.select(data.task_set(all_tasks, a.task_set), a.limit, a.task_ids.split(',') if a.task_ids else None)
    try:   # the stored config this run continues (a truncated one is overwritten below, as before)
        prev = json.loads((run_dir / 'config.json').read_text())
    except (OSError, ValueError):
        prev = None
    # #481: the AAI-open version (the launcher's label; a resume or a grade of stored generations keeps the label of
    # the run it continues). Checked before build_config, which fills the preset of a served name (Super VL's T 0.6
    # is part of the protocol); an explicit --preset is a deviation.
    dev = [] if (src_cfg or prev) else aai_open.client_deviations(
        context_window=a.context_window, overrides={'temperature': a.temperature, 'top_p': a.top_p},
        flags={'--extra-body': a.extra_body, '--max-tokens': a.max_tokens, '--preset': a.preset,
               '--vision': a.vision if a.vision != 'off' else None})
    dev += aai_open.judge_deviations(a.judge_arm) if a.stage in ('all', 'grade') else []
    cfg = build_config(a, run_dir, tasks)
    cfg['aai_open'] = aai_open.record(dev, source_cfg=src_cfg or prev)
    if src_cfg:
        cfg['generations_from'] = str(src)
        for k in ('generation', 'endpoint', 'limits', 'context', 'image_inputs'):
            if k in src_cfg:
                cfg[k] = src_cfg[k]
        cfg['inputs']['vision'] = (src_cfg.get('inputs') or {}).get('vision', 'auto')   # as the generations were made
        if 'git_sha' in src_cfg:
            cfg['generation_git_sha'] = src_cfg.get('generation_git_sha') or src_cfg['git_sha']
    write_config(run_dir, cfg)
    print(f'[run] {a.run_id}: {len(tasks)} tasks x {a.repeats} repeats, model {a.model}, judge {a.judge}', flush=True)
    if a.stage in ('all', 'generate'):
        asyncio.run(stage_generate(a, run_dir, tasks, cfg))
    if a.stage in ('all', 'grade'):
        asyncio.run(stage_grade(a, run_dir, tasks, cfg))
    if a.stage in ('all', 'summarize', 'grade', 'generate'):
        cfg = json.loads((run_dir / 'config.json').read_text())
        stage_summarize(a, run_dir, tasks, cfg)


if __name__ == '__main__':
    main()

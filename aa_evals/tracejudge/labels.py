"""The labeled set (#66 runs) and the judge comparison.

    python -m aa_evals.tracejudge.labels build --out labels.jsonl
    python -m aa_evals.tracejudge.labels run --labels labels.jsonl --judge serving:X --mode blind --out res.jsonl
    python -m aa_evals.tracejudge eval --labels labels.jsonl --flags qwen=res_qwen.jsonl gemma=res_gemma.jsonl

Label rows: {id, run_dir, key, label: infra|clean|ambiguous, category, group, why, inject?}.
Groups:
  real_silent   infra the pipeline scored as a model outcome (#145's memcap kill of the agent, a verifier that
                wrote no reward before the verifier_no_reward rule existed);
  real_known    infra the pipeline already caught (infra_error rows, retried physical attempts, the run cutoff);
  injected      a clean #66 trace with one infra fault written in (types #66 didn't happen to hit);
  hard_negative model outcomes that look like infra (the model killing its own agent, memory-cap kills of task
                processes, agent timeouts, context-window ends, EOS inside reasoning, max-token stops);
  clean         a stratified sample of passes and fails;
  ambiguous     reported, not scored (proxy 503s on sandbox downloads, verifier timeouts, CritPt's empty summaries).
Modes:
  blind   the LLM alone, with the pipeline's infra fields and the pre-filter hits hidden: can the model find the
          failure from the trace? (the candidate comparison);
  prod    what `judge` does: rules first, the LLM on failed or suspicious attempts (policy `fails`).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import glob
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from aa_evals.tracejudge import digest as dg
from aa_evals.tracejudge.core import build_row, needs_llm
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.rules import decide, run_rules
from aa_evals.tracejudge.trace import Event, Turn

FB = '/data/aai/aa_evals/runs_fallback'
FAST = '/data/aai/aa_evals/runs'
SHARED_LOAD = '/data/aai/aa_evals/runs/judges/s3/shared_load.txt'

TB_RUNS = [f'{FB}/tb21/r3-medium', f'{FB}/tb21/r3-xhigh', f'{FB}/tb21/r4-262k-medium', f'{FB}/tb21/r4-262k-xhigh',
           f'{FB}/tbsci/r3-qwen-medium', f'{FB}/tbsci/r3-qwen-xhigh', f'{FB}/tbsci/r4-262k-medium', f'{FB}/tbsci/r4-262k-xhigh',
           f'{FB}/tbsci_lite/r3-qwen-medium-v2', f'{FB}/tbsci_lite/r3-qwen-xhigh', f'{FB}/tbsci_lite/r4-262k-medium',
           f'{FB}/tbsci_lite/r4-262k-xhigh']
SINGLE_RUNS = {'hle': f'{FB}/hle/qwen-diamond-medium', 'omniscience': f'{FAST}/omniscience/qwen-medium-r3-300',
               'gdppdf': f'{FB}/gdppdf/r3-qwen38-medium', 'critpt': f'{FB}/critpt/qwen-full-medium',
               'automationbench': f'{FB}/automationbench/r3-qwen-medium-600'}


def _jl(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def _outcomes(run):
    for f in sorted(glob.glob(f'{run}/attempts/*/outcome.json')):
        yield Path(f).parent.name, json.load(open(f))


def build(out: str, seed: int = 0, scan: str | None = None):
    rng = random.Random(seed)
    rows = []

    def add(run, key, label, group, why, category=None, inject=None):
        rows.append({'id': f'{Path(run).parent.name}/{Path(run).name}/{key}' + (f'#{inject["type"]}' if inject else ''),
                     'run_dir': run, 'key': key, 'label': label, 'category': category, 'group': group, 'why': why,
                     **({'inject': inject} if inject else {})})

    # ---- real infra the pipeline missed
    add(f'{FB}/tbsci_lite/r4-262k-medium', 'apo-holo-docking__a0__r0', 'infra', 'real_silent',
        'harbor_ns memcap killed the agent (pi) itself; scored 0 (PR #145)', 'sandbox_kill')
    add(f'{FB}/tbsci/r1-qwen-medium', 'tamp-skill-planning__a0__r0', 'infra', 'real_silent',
        'RewardFileNotFoundError: the verifier wrote no reward; scored 0 before the verifier_no_reward rule', 'grader_crash')
    # ---- real infra the pipeline caught
    add(f'{FB}/tb21/r4-262k-medium', 'extract-moves-from-video__a2__r0', 'infra', 'real_known', 'stopped at the run cutoff',
        'harness_cutoff')
    by_reason = defaultdict(list)
    for run in TB_RUNS:
        for k, o in _outcomes(run):
            if o.get('infrastructure_failure'):
                by_reason[o.get('infrastructure_reason')].append((run, k))
    cat = {'pi_backend_unreachable': 'network', 'verifier_no_reward': 'grader_crash', 'worker_stopped': 'harness_cutoff'}
    for reason, n in (('pi_backend_unreachable', 10), ('verifier_no_reward', 8), ('worker_stopped', 6)):
        for run, k in rng.sample(by_reason[reason], min(n, len(by_reason[reason]))):
            add(run, k, 'infra', 'real_known', f'pipeline infra_reason {reason}', cat[reason])
    # (Omniscience's two infra_error rows in #66 were superseded by successful retries: not labels.)
    for r in _jl(f'{FB}/critpt/ref-ds41-max/items.jsonl'):
        if r['status'] == 'infra_error':
            add(f'{FB}/critpt/ref-ds41-max', f"{r['item_id']}/r{r['repeat']}/a{r.get('attempt')}", 'infra', 'real_known',
                r['error'][:120], 'harness_error')
    # ---- hard negatives
    for run, k, why in [
            (f'{FB}/tbsci_lite/r4-262k-medium', 'survival-band-screening__a0__r0', "the model's pkill -f matched the agent's own command line"),
            (f'{FB}/tb21/r4-262k-medium', 'rstan-to-pystan__a0__r0', "the model ran kill 81 on a process of its agent"),
            (f'{FB}/tb21/r3-medium', 'qemu-alpine-ssh__a1__r0', "the model kill -9'd a list of pids including its agent's")]:
        add(run, k, 'clean', 'hard_negative', why)
    scanned = {}
    if scan:
        for r in _jl(scan):
            if 'error' not in r:
                scanned[(r['run'], r['key'])] = r
    memcap_fail, timeouts, ctx = [], [], []
    for run in TB_RUNS:
        for k, o in _outcomes(run):
            st = o.get('stats') or {}
            if o.get('infrastructure_failure'):
                continue
            s = scanned.get((run, k))
            if s and 'memcap_killed_task_process' in [h['rule'] for h in s['hits']] and not o.get('score'):
                memcap_fail.append((run, k))
            if o.get('exception_type') == 'AgentTimeoutError' or st.get('end_state') in ('env_budget_spent', 'wall_clock_backstop'):
                timeouts.append((run, k))
            if st.get('context_limit') and not o.get('score'):
                ctx.append((run, k))
    for lst, n, why in ((memcap_fail, 8, 'the memory cap killed a task process the model started'),
                        (timeouts, 6, 'agent time limit (protocol)'), (ctx, 6, 'the context window ran out (protocol)')):
        for run, k in rng.sample(lst, min(n, len(lst))):
            add(run, k, 'clean', 'hard_negative', why)
    hle = _jl(f"{SINGLE_RUNS['hle']}/items.jsonl")
    for kind, n in (('reasoning_unclosed', 6), ('length', 4)):
        pool = [r for r in hle if r.get('model_error_kind') == kind or (kind == 'length' and r.get('finish_reason') == 'length')]
        for r in rng.sample(pool, min(n, len(pool))):
            add(SINGLE_RUNS['hle'], f"{r['item_id']}.r{r['repeat']}", 'clean', 'hard_negative',
                f'HLE {kind}: model behavior (EOS in reasoning / max tokens at the protocol cap)')
    cp = [r for r in _jl(f"{SINGLE_RUNS['critpt']}/items.jsonl") if 'output_limit' in (r.get('error') or '')]
    for r in rng.sample(cp, min(3, len(cp))):
        add(SINGLE_RUNS['critpt'], f"{r['item_id']}/r{r['repeat']}/a{r.get('attempt')}", 'clean', 'hard_negative', 'CritPt output_limit (model)')
    gd = [g for g in _jl(f"{SINGLE_RUNS['gdppdf']}/generations.jsonl") if g.get('error_kind') == 'reasoning_unclosed']
    for g in gd[:2]:
        add(SINGLE_RUNS['gdppdf'], f"{g['task_id']}/{g['repeat']}", 'clean', 'hard_negative', 'GDP.pdf reasoning_unclosed (model)')
    ab = [r for r in _jl(f"{SINGLE_RUNS['automationbench']}/items.jsonl") if r.get('stop_condition') == 'max_turns_reached']
    for r in rng.sample(ab, min(3, len(ab))):
        add(SINGLE_RUNS['automationbench'], f"{r['item_id']}/r{r['repeat']}", 'clean', 'hard_negative', 'AutomationBench turn cap (protocol)')
    # ---- ambiguous (reported only)
    for run, k, why in [
            (f'{FB}/tb21/r3-medium', 'compile-compcert__a0__r2', 'proxy 503 on a sandbox download (passed anyway)'),
            (f'{FB}/tb21/r3-xhigh', 'mteb-leaderboard__a2__r0', 'proxy 503 on a sandbox request (passed anyway)'),
            (f'{FB}/tb21/r3-xhigh', 'mcmc-sampling-stan__a1__r0', 'agent exit -15 after a model kill, unclear which (passed)'),
            (f'{FB}/tbsci_lite/r4-262k-medium', 'asd-brainspan-enrichment__a0__r0', 'verifier timeout at 180 s'),
            (f'{FB}/tb21/r3-medium', 'query-optimize__a1__r1', "verifier timeout on the model's query")]:
        add(run, k, 'ambiguous', 'ambiguous', why)
    for r in _jl(f"{SINGLE_RUNS['critpt'].replace('medium', 'xhigh')}/items.jsonl"):
        if 'summary_failed' in (r.get('error') or ''):
            add(SINGLE_RUNS['critpt'].replace('medium', 'xhigh'), f"{r['item_id']}/r{r['repeat']}/a{r.get('attempt')}",
                'ambiguous', 'ambiguous', 'CritPt: the model returned an empty compaction summary')
    # ---- clean stratified sample (passes and fails) + injected faults on clean traces
    taken = {(r['run_dir'], r['key']) for r in rows}
    clean = []
    tb_by = defaultdict(list)
    for run in TB_RUNS:
        for k, o in _outcomes(run):
            s = scanned.get((run, k))
            if o.get('infrastructure_failure') or (s and s['verdict'] != 'clean') or (run, k) in taken:
                continue
            tb_by[(Path(run).parent.name, bool(o.get('score')))].append((run, k))
    for (b, ok), lst in sorted(tb_by.items()):
        n = {'tb21': 6, 'tbsci': 3, 'tbsci_lite': 4}[b] if not ok else {'tb21': 4, 'tbsci': 1, 'tbsci_lite': 2}[b]
        clean += [(run, k, f'{b} {"pass" if ok else "fail"}') for run, k in rng.sample(lst, min(n, len(lst)))]
    for b, run in SINGLE_RUNS.items():
        L = open_run(run)
        ks = L.finished()
        rng.shuffle(ks)
        n_pass = n_fail = 0
        for k in ks:
            if (run, k) in taken:
                continue
            t = L.load(k)
            if t.recorded.get('status') not in ('ok', None):
                continue
            sc = t.recorded.get('score')
            ok = bool(sc) and float(sc) >= 1.0
            if ok and n_pass < 3:
                n_pass += 1
                clean.append((run, k, f'{b} pass'))
            elif not ok and n_fail < 5:
                n_fail += 1
                clean.append((run, k, f'{b} fail'))
            if n_pass >= 3 and n_fail >= 5:
                break
    rng.shuffle(clean)
    for run, k, why in clean:
        add(run, k, 'clean', 'clean', why)
    # injected: a fresh draw of clean traces (not the ones above)
    used = {(r['run_dir'], r['key']) for r in rows}
    agentic_pool = [x for lst in tb_by.values() for x in lst if x not in used]
    rng.shuffle(agentic_pool)
    a_types = ['proxy_outage', 'fork_failure', 'disk_full', 'tool_parse', 'empty_responses', 'sandbox_oom']
    for i, typ in enumerate(a_types * 2):
        run, k = agentic_pool[i]
        add(run, k, 'infra', 'injected', f'injected {typ}', INJECT_CATEGORY[typ], {'type': typ, 'at': round(rng.uniform(0.4, 0.8), 2)})
    s_types = ['empty_zero_tokens', 'low_cap_cutoff', 'gateway_errors', 'template_garbage']
    for b in ('hle', 'omniscience', 'gdppdf'):
        L = open_run(SINGLE_RUNS[b])
        ks = [k for k in L.finished() if (SINGLE_RUNS[b], k) not in used]
        rng.shuffle(ks)
        for typ, k in zip(s_types, ks):
            add(SINGLE_RUNS[b], k, 'infra', 'injected', f'injected {typ}', INJECT_CATEGORY[typ], {'type': typ})
    Path(out).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} labels -> {out}:', dict(Counter((r['group'], r['label']) for r in rows)))


# ------------------------------------------------------------------------------------------------- injection
INJECT_CATEGORY = {'proxy_outage': 'network', 'fork_failure': 'sandbox_kill', 'disk_full': 'sandbox_kill',
                   'tool_parse': 'harness_error', 'empty_responses': 'silent_zero', 'sandbox_oom': 'sandbox_kill',
                   'empty_zero_tokens': 'silent_zero', 'low_cap_cutoff': 'harness_cutoff', 'gateway_errors': 'network',
                   'template_garbage': 'harness_error'}


def inject(t, spec):
    """Write one infra fault into a clean trace (the attempt then ends there, scored 0 like the real ones)."""
    t = copy.deepcopy(t)
    typ = spec['type']
    r = t.recorded
    if t.kind == 'agentic':
        cut = max(4, int(len(t.turns) * spec.get('at', 0.6)))
        t.turns = t.turns[:cut]
        n = len(t.turns)
        ref = f'turn {n + 1}'
        r.update(score=0.0, reward=0.0)
        if typ == 'proxy_outage':
            t.turns.append(Turn(n, 'assistant', 'error', 'Connection error. (fetch failed: connect ECONNREFUSED '
                                '10.0.2.2:40417; 4 retries exhausted)', ref))
            r.update(end_state='error', exception_type=None)
            t.events.append(Event('proxy.log', 'upstream http://cpu302.internal.cluster.internal:32034/v1/chat/completions '
                                  'failed: ClientConnectorError: Cannot connect to host (Connection refused)', 'pi-cont16k/proxy.log:3'))
        elif typ == 'fork_failure':
            for j in range(3):
                t.turns.append(Turn(n + 2 * j, 'assistant', 'tool_call', 'bash: ls /app && python3 -c "print(1)"', f'turn {n + 1 + 2 * j}'))
                t.turns.append(Turn(n + 2 * j + 1, 'tool', 'tool_result', 'bash: fork: retry: Resource temporarily unavailable\n'
                                    'bash: fork: retry: Resource temporarily unavailable\nbash: fork: Resource temporarily '
                                    'unavailable', f'turn {n + 2 + 2 * j}', {'is_error': True}))
            t.turns.append(Turn(n + 6, 'assistant', 'text', 'The shell cannot start any process; I cannot continue.', f'turn {n + 7}'))
            r.update(end_state='finished')
        elif typ == 'disk_full':
            t.turns.append(Turn(n, 'assistant', 'tool_call', "bash: echo 'x = 1' > /app/solution.py && cat /app/solution.py", ref))
            t.turns.append(Turn(n + 1, 'tool', 'tool_result', "bash: echo: write error: No space left on device\n"
                                "df: /app: 100% used (0 bytes free of 10G)", f'turn {n + 2}', {'is_error': True}))
            t.turns.append(Turn(n + 2, 'assistant', 'text', 'The disk is full and I did not create large files; stopping.', f'turn {n + 3}'))
            r.update(end_state='finished')
        elif typ == 'tool_parse':
            t.turns.append(Turn(n, 'assistant', 'text', 'Now I will run the tests.\n<tool_call>\n{"name": "bash", '
                                '"arguments": {"command": "cd /app && python -m pytest -q"}}\n</tool_call>', ref,
                                {'stop_reason': 'stop'}))
            r.update(end_state='finished')
        elif typ == 'empty_responses':
            t.turns.append(Turn(n, 'assistant', 'text', '', ref, {'stop_reason': 'stop', 'completion_tokens': 0}))
            r.update(end_state='finished')
        elif typ == 'sandbox_oom':
            t.events.append(Event('harbor.log', 'harbor_ns: sandbox cgroup oom-kill: Out of memory: Killed process 4021 (node) '
                                  'total-vm:2104552kB, anon-rss:1032112kB', 'harbor.log:9'))
            r.update(end_state='error', exception_type='NonZeroAgentExitCodeError', returncode=137)
            t.events.append(Event('outcome', 'NonZeroAgentExitCodeError: Command failed (exit 137): pi --print --mode json ...',
                                  'outcome.json:exception_message'))
        return t
    final = [x for x in t.turns if x.kind == 'text'][-1]
    r.update(score=0, status='ok')
    if typ == 'empty_zero_tokens':
        t.turns = [x for x in t.turns if x.kind == 'task']
        t.turns.append(Turn(len(t.turns), 'assistant', 'text', '', 'response',
                            {'finish_reason': 'stop', 'completion_tokens': 0, 'prompt_tokens': final.meta.get('prompt_tokens')}))
    elif typ == 'low_cap_cutoff':
        th = [x for x in t.turns if x.kind == 'thinking']
        t.turns = [x for x in t.turns if x.kind == 'task']
        if th:
            t.turns.append(Turn(len(t.turns), 'assistant', 'thinking', th[0].text[:3000], 'response'))
        t.turns.append(Turn(len(t.turns), 'assistant', 'text', '', 'response', {'finish_reason': 'length', 'completion_tokens': 1024}))
        r['max_tokens'] = None   # the run config says 64k elsewhere; the judge must notice 1024 is not a model limit
        t.events.append(Event('client', 'request max_tokens=1024 (run config generation.max_tokens=65536)', 'client:request'))
    elif typ == 'gateway_errors':
        t.turns = [x for x in t.turns if x.kind == 'task']
        t.turns.append(Turn(len(t.turns), 'assistant', 'text', '', 'response', {'finish_reason': None, 'completion_tokens': None}))
        for j in range(1, 4):
            t.events.append(Event('client', json.dumps({'attempt': j, 'status': 502, 'error': '<html><head><title>502 Bad Gateway'
                                                         '</title></head></html>'}), f'client:attempt {j}'))
    elif typ == 'template_garbage':
        final.text = '<|im_start|>assistant\n<|im_start|>assistant\n<|im_end|><|endoftext|><|im_start|>'
        final.meta['finish_reason'] = 'stop'
    return t


def redact(t):
    """Blind mode: hide what the pipeline concluded about infra."""
    t = copy.deepcopy(t)
    r = t.recorded
    for k in ('infrastructure_failure', 'infra_reason', 'cutoff', 'pi_error', 'error_stage', 'model_error_kind'):
        r.pop(k, None)
    if r.get('status') in ('infra_error', 'model_error'):
        r['status'] = 'ok'
    for k in ('error', 'error_kind'):
        r.pop(k, None)
    return t


async def run(labels: str, judge_spec: str, mode: str, out: str, concurrency: int = 16, effort=None, thinking=None,
              limit=None):
    from aa_evals.tracejudge.judge import LLMJudge
    judge = LLMJudge(judge_spec, effort=effort, thinking=thinking)
    rows = _jl(labels)[:limit]
    done = {json.loads(x)['id'] for x in open(out)} if Path(out).exists() else set()
    rows = [r for r in rows if r['id'] not in done]
    sem = asyncio.Semaphore(concurrency)
    layouts = {}
    fh = open(out, 'a')
    t0 = time.time()

    async def one(lab):
        async with sem:
            L = layouts.setdefault(lab['run_dir'], open_run(lab['run_dir']))
            t = await asyncio.to_thread(L.load, lab['key'])
            if lab.get('inject'):
                t = inject(t, lab['inject'])
            if mode == 'blind':
                t = redact(t)
                hits, verdict = [], 'blind'
                text = dg.render(t, [])
                llm = await judge.judge(text)
            else:
                hits = run_rules(t)
                verdict = decide(hits)
                llm, text = None, ''
                if needs_llm('fails', verdict, t):
                    text = dg.render(t, hits)
                    llm = await judge.judge(text)
            row = build_row(t, hits, verdict, llm, judge.describe(), len(text) if text else None)
            fh.write(json.dumps({'id': lab['id'], 'mode': mode, **{k: lab[k] for k in ('label', 'category', 'group')}, **row}) + '\n')
            fh.flush()

    try:
        await asyncio.gather(*(one(r) for r in rows))
    finally:
        fh.close()
        await judge.close()
        if judge_spec.startswith('serving:aa-judge-'):
            # aa-judge-s3 bills its servers' GPU-hours by wall time: record our load windows (coordinator, 11-04)
            iso = lambda x: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(x))
            with open(SHARED_LOAD, 'a') as f:
                f.write(f"{judge_spec.split(':', 1)[1]} {iso(t0)} {iso(time.time())} aa-tracejudge {concurrency}\n")
    print(f'{len(rows)} judged in {time.time() - t0:.0f}s -> {out}')


def evaluate(labels: str, arms: dict) -> dict:
    lab = {r['id']: r for r in _jl(labels)}
    res = {}
    for name, path in arms.items():
        rows = {r['id']: r for r in _jl(path)}
        g = defaultdict(lambda: Counter())
        misses, false, amb = [], [], []
        tok_p, tok_c, lat, n_llm, n_err = [], [], [], 0, 0
        for i, r in rows.items():
            L = lab.get(i)
            if not L:
                continue
            if r.get('llm'):
                n_llm += 1
                if not r['llm'].get('ok'):
                    n_err += 1
                tok_p.append(r['llm'].get('prompt_tokens') or 0)
                tok_c.append(r['llm'].get('completion_tokens') or 0)
                lat.append(r['llm'].get('latency_s') or 0)
            grp = L['group']
            g[grp]['n'] += 1
            if r['flagged']:
                g[grp]['flagged'] += 1
                if L['label'] == 'infra' and L.get('category') in r.get('categories', []):
                    g[grp]['category_match'] += 1
            if L['label'] == 'infra' and not r['flagged']:
                misses.append(f"{i} ({L['why'][:60]})")
            if L['label'] == 'clean' and r['flagged']:
                false.append(f"{i}: {','.join(r.get('categories', []))} {(r.get('flags') or [{}])[0].get('evidence', '')[:100]!r}")
            if L['label'] == 'ambiguous':
                amb.append(f"{i}: {'FLAG ' + ','.join(r.get('categories', [])) if r['flagged'] else 'no flag'}")
        inf = [i for i in rows if lab.get(i, {}).get('label') == 'infra']
        cln = [i for i in rows if lab.get(i, {}).get('label') == 'clean']
        tp = sum(1 for i in inf if rows[i]['flagged'])
        fp = sum(1 for i in cln if rows[i]['flagged'])
        res[name] = {
            'recall_infra': f'{tp}/{len(inf)}', 'false_flags_clean': f'{fp}/{len(cln)}',
            'by_group': {k: dict(v) for k, v in sorted(g.items())},
            'misses': misses, 'false_flags': false, 'ambiguous': amb,
            'llm_calls': n_llm, 'llm_errors': n_err,
            'mean_prompt_tokens': round(sum(tok_p) / len(tok_p)) if tok_p else 0,
            'mean_completion_tokens': round(sum(tok_c) / len(tok_c)) if tok_c else 0,
            'mean_latency_s': round(sum(lat) / len(lat), 1) if lat else 0,
        }
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m aa_evals.tracejudge.labels')
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--scan', default=None, help='a rules-only scan of the TB runs (to pick clean / memcap rows)')
    b.add_argument('--seed', type=int, default=0)
    r = sub.add_parser('run')
    r.add_argument('--labels', required=True)
    r.add_argument('--judge', required=True)
    r.add_argument('--mode', choices=['blind', 'prod'], required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--concurrency', type=int, default=16)
    r.add_argument('--effort', default=None)
    r.add_argument('--thinking', choices=['on', 'off'], default=None)
    r.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    if a.cmd == 'build':
        build(a.out, a.seed, a.scan)
    else:
        import signal
        import sys
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # so a kill still records the shared-load window
        asyncio.run(run(a.labels, a.judge, a.mode, a.out, a.concurrency, a.effort, a.thinking, a.limit))


if __name__ == '__main__':
    main()

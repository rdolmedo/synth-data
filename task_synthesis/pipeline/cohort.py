"""Generator cohorts: tasks written by another model for the same specs as a round, checked and judged exactly like
the round's first attempts. Research only (user, 2026-10-30: a standalone cohort to study how generator capability
affects task quality); kept apart from the round and campaign bundles.

  <ROOT>/cohorts/<name>/specs.jsonl, prompts/<spec>.{system,user}.txt   (the exact generator prompts of the round)
  <ROOT>/cohorts/<name>/replies/<spec>.reply.txt                         (one reply per spec, no repair turns)

  python -m task_synthesis.pipeline.cohort submit --cohort sonnet55 --round r5 --judge gen-ds2   (CPU job)
  python -m task_synthesis.pipeline.cohort report --cohort sonnet55 --round r5

Each reply is parsed, materialized, built and checked (check.py) and, when valid, judged (judge.py): the same path as
a round's attempt 0. The paired arm is the round's own attempt 0 for the same spec (its saved check, and its judge
verdict when attempt 0 was valid without repairs).
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from synth.common import atomic

from ..common import ROOT
from .package import PackageError, materialize, parse
from .round import ROUNDS, infra_attempt, submit as submit_job

COHORTS = ROOT / 'cohorts'


async def run(a):
    from .check import check, make_store
    from .judge import judge
    from .llm import Endpoint, wait_healthy
    d = COHORTS / a.cohort
    specs = [json.loads(l) for l in (d / 'specs.jsonl').read_text().splitlines()]
    await wait_healthy(a.judge)
    llm = Endpoint(a.judge, max_in_flight=a.concurrency)
    store = make_store()
    builds = asyncio.Semaphore(a.build_parallel)
    (d / 'records').mkdir(exist_ok=True)

    async def one(spec):
        sid = spec['spec_id']
        out = d / 'records' / f'{sid}.json'
        reply = d / 'replies' / f'{sid}.reply.txt'
        if out.exists() or not reply.exists():
            return
        rec = {'spec_id': sid, 'cohort': a.cohort}
        try:
            pkg = parse(reply.read_text())
        except PackageError as e:
            rec.update(status='format_fail', errors=[str(e)])
            atomic(out, rec)
            return
        task_dir = materialize(pkg, spec, {'model': a.cohort, 'cohort': a.cohort}, d / 'tasks' / sid)
        for i in range(3):
            async with builds:
                try:
                    chk = await check(store, task_dir)
                except Exception as e:  # noqa: BLE001
                    chk = {'ok': False, 'errors': [], 'infra': f'{type(e).__name__}: {e}'}
            if not chk.get('infra'):
                break
            await asyncio.sleep(60 * (i + 1))
        atomic(d / 'records' / f'{sid}.check.json', chk)
        rec.update(status='valid' if chk['ok'] else ('infra_fail' if chk.get('infra') else 'env_fail'),
                   errors=chk['errors'], meta=pkg['meta'], instruction=pkg['instruction'],
                   reply_chars=len(reply.read_text()))
        if chk['ok']:
            rec['judge'] = await judge(llm, spec, pkg, chk)
        atomic(out, rec)
        print(f'{time.strftime("%H:%M:%S")} {sid} {rec["status"]} judge={rec.get("judge", {}).get("pass")}', flush=True)

    await asyncio.gather(*(one(s) for s in specs))
    report(a.cohort, a.round)


def harvest(d, sid, res):
    """A finished `claude -p` result -> replies/<spec>.reply.txt and cli/<spec>.meta.json; returns the model that
    wrote most of the output."""
    if res.get('is_error') or not res.get('result'):
        return None
    usage = res.get('modelUsage') or {}
    main = max(usage, key=lambda m: usage[m].get('outputTokens', 0)) if usage else None
    (d / 'replies' / f'{sid}.reply.txt').write_text(res['result'])
    atomic(d / 'cli' / f'{sid}.meta.json', {'model_main': main, 'models': {m: u.get('outputTokens') for m, u in usage.items()},
                                           'output_tokens': (res.get('usage') or {}).get('output_tokens'),
                                           'safety_stops': res.get('safety_stops'), 'duration_ms': res.get('duration_ms')})
    return main


def generate_cli(a):
    """Write cohort replies with the headless Claude Code CLI (`claude -p`, the user's subscription; user, 2026-10-30:
    Sonnet 5.5 at medium effort, as many as the quota allows). One plain model call per spec: the round's system
    prompt, its user message, no tools. cli/<spec>.json keeps the CLI's result (usage, the models that answered);
    a reply mostly written by another model than --model (fallback after safety stops) is recorded as such."""
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    d = COHORTS / a.cohort
    (d / 'cli').mkdir(exist_ok=True)
    specs = [json.loads(l)['spec_id'] for l in (d / 'specs.jsonl').read_text().splitlines()]
    for f in (d / 'cli').glob('s*.json'):   # results of calls whose generator was stopped
        if f.name.endswith('.meta.json') or (d / 'replies' / f'{f.stem}.reply.txt').exists():
            continue
        try:
            harvest(d, f.stem, json.loads(f.read_text()))
        except ValueError:
            pass   # still running (or died): its spec is skipped below while the file exists
    started = {f.stem for f in (d / 'cli').glob('s*.json') if not f.name.endswith('.meta.json')}
    todo = [s for s in specs if s >= a.start and s not in started
            and not (d / 'replies' / f'{s}.reply.txt').exists()][:a.limit]
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE')}
    env['DISABLE_AUTOUPDATER'] = '1'
    stop = []

    def one(sid):
        if stop or (d / 'replies' / f'{sid}.reply.txt').exists():   # another generator got there first
            return
        out = d / 'cli' / f'{sid}.json'
        with open(d / 'prompts' / f'{sid}.user.txt') as stdin, open(out, 'w') as f:
            r = subprocess.run(['claude', '-p', '--model', a.model, '--effort', a.effort,
                                '--system-prompt-file', str(d / 'prompts' / f'{sid}.system.txt'), '--tools', '',
                                '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json'],
                               stdin=stdin, stdout=f, stderr=subprocess.PIPE, text=True, env=env, timeout=3600)
        try:
            res = json.loads(out.read_text())
        except ValueError:
            res = {'is_error': True, 'result': r.stderr[-500:]}
        if res.get('is_error') or not res.get('result'):
            print(f'{sid} ERROR {str(res.get("result"))[:200]}', flush=True)
            if 'limit' in str(res.get('result', '')).lower():
                stop.append(sid)
            return
        main = harvest(d, sid, res)
        print(f'{time.strftime("%H:%M:%S")} {sid} ok main={main} out={(res.get("usage") or {}).get("output_tokens")}',
              flush=True)

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(one, todo))
    print(f'stopped on quota at {stop[0]}' if stop else 'done', flush=True)


def paired(round_name, sid):
    """The round's attempt 0 for this spec: status and, when it was valid with no repairs, its judge verdict."""
    from .roundpack import RoundFiles   # packed or loose (#245)
    files = RoundFiles(ROUNDS / round_name)
    rec = files.record(sid)
    a0 = rec['attempts'][0]
    if a0.get('stage') == 'format':
        return {'status': 'format_fail'}
    if infra_attempt(a0):
        return {'status': 'infra_fail'}
    if not a0.get('ok'):
        return {'status': 'env_fail', 'errors': a0.get('errors')}
    return {'status': 'valid', 'judge': rec.get('judge'), 'reply_chars': len(files.text(sid, 'attempt-0.reply.txt'))}


def report(cohort, round_name):
    d = COHORTS / cohort
    rows = []
    for p in sorted((d / 'records').glob('s*.json')):
        if p.name.endswith('.check.json'):
            continue
        r = json.loads(p.read_text())
        rows.append((r, paired(round_name, r['spec_id'])))
    n = len(rows)

    def arm(i, label):
        recs = [x[i] for x in rows]
        valid = [r for r in recs if r['status'] == 'valid']
        judged = [r for r in valid if (r.get('judge') or {}).get('scores')]
        passed = [r for r in judged if r['judge']['pass']]
        scores = {k: statistics.mean(r['judge']['scores'][k] for r in judged) for k in judged[0]['judge']['scores']} \
            if judged else {}
        chars = [r['reply_chars'] for r in valid if r.get('reply_chars')]
        return (f'| {label} | {len(valid)}/{n} | {len(passed)}/{len(judged)} | '
                + ' / '.join(f'{scores[k]:.2f}' for k in scores) + f' | {statistics.median(chars) if chars else 0:.0f} |')

    both = sum(1 for c, p in rows if c['status'] == 'valid' and p['status'] == 'valid')
    lines = [f'# Generator cohort {cohort} vs {round_name} attempt 0', '', f'{n} paired specs (same prompts, no repairs).',
             '', '| Arm | First-try valid | Judge pass of judged | Scores cl/co/so/di/re/or | Median reply chars |',
             '| --- | --- | --- | --- | --- |', arm(0, cohort), arm(1, f'{round_name} attempt 0 (DeepSeek)'), '',
             f'Both valid: {both}. Judge: the round\'s judge model (DeepSeek); round-0 verdicts exist only where attempt 0 '
             'was valid without repairs.', '', '| Spec | ' + cohort + ' | DeepSeek a0 |', '| --- | --- | --- |']
    for c, p in rows:
        f = lambda r: r['status'] + (('/pass' if r['judge']['pass'] else '/fail') if (r.get('judge') or {}).get('scores') else '')
        lines.append(f'| {c["spec_id"]} | {f(c)} | {f(p)} |')
    (d / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print((d / 'REPORT.md').read_text())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    for op in ('run', 'submit'):
        p = sub.add_parser(op)
        p.add_argument('--cohort', required=True)
        p.add_argument('--round', required=True)
        p.add_argument('--judge', default='gen-ds')
        p.add_argument('--concurrency', type=int, default=64)
        p.add_argument('--build-parallel', type=int, default=8)
        if op == 'submit':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=16)
            p.add_argument('--memory-gib', type=int, default=64)
    p = sub.add_parser('report')
    p.add_argument('--cohort', required=True)
    p.add_argument('--round', required=True)
    p = sub.add_parser('generate-cli')
    p.add_argument('--cohort', required=True)
    p.add_argument('--model', default='sonnet')
    p.add_argument('--effort', default='medium')
    p.add_argument('--parallel', type=int, default=8)
    p.add_argument('--start', default='', help='first spec id to generate')
    p.add_argument('--limit', type=int, default=100000)
    a = ap.parse_args()
    if a.op == 'generate-cli':
        return generate_cli(a)
    if a.op == 'run':
        asyncio.run(run(a))
    elif a.op == 'submit':
        # two jobs on one cohort rewrite the same task dirs (2026-10-31: empty task.toml files); one at a time
        prev = COHORTS / a.cohort / 'job.json'
        if prev.exists():
            import subprocess
            job = json.loads(prev.read_text())['job_id']
            if subprocess.run(['condor_q', job, '-af', 'JobStatus'], capture_output=True, text=True).stdout.strip():
                raise SystemExit(f'{a.cohort}: job {job} is still in the queue')
        submit_job(a, module='cohort', d=COHORTS / a.cohort, name=f'cohort-{a.cohort}')
    else:
        report(a.cohort, a.round)


if __name__ == '__main__':
    main()

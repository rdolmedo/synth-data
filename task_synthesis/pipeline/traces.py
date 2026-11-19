"""Trajectories of a synth campaign on tasksynth tasks: per-trace facts, offline package misses, and an LLM judge of
how far each trace got (the tasks ship no verifier).

  python -m task_synthesis.pipeline.traces run --campaign /data/aai/tbench/synth/<id> --bundle <stream dir> --judge gen-vocab+gen-ds
  python -m task_synthesis.pipeline.traces report --name <id>
  python -m task_synthesis.pipeline.traces show --name <id> <task> [--full]

Audit tool, not part of production since 2026-11-03 (#57: student 1 trained without the judge's task-defect filter was
not worse); run it to audit a campaign, e.g. after a generator or prompt change.

`run` streams by default: start it with the campaign and it judges each attempt as it finishes, then exits once the
campaign is complete and every attempt is judged (on a finished campaign: one pass). --once: one pass over what has
finished so far. A periodic-pass judge falls behind: each pass waits on its slowest verdict (1-2 h at full load).

State in <ROOT>/traces/<campaign id>/: records.jsonl (one line per attempt) and REPORT.md. The judge sees the
instruction, the author's private solution sketch, the condensed trace (assistant text, tool calls, heads and tails of
tool results; no reasoning) and the final state test.sh recorded (declared outputs, file tree).
"""
import argparse
import asyncio
import collections
import json
import re
import statistics
import time
import zipfile
from pathlib import Path

from synth import archive
from synth.common import atomic, read

from ..common import ROOT
from .campaign import CAMPAIGNS
from .round import ROUNDS

TRACES = ROOT / 'traces'
INSTALL = re.compile(r'\b(?:pip3?|python3? -m pip|uv pip|uv add|apt(?:-get)?|npm|yarn|pnpm|cargo|go|gem|conda|'
                     r'cpanm|R -e)\s+(?:install|add|get|i)\b|\b(?:curl|wget)\s+[^|;&]*https?://|\bgit clone\b')
OFFLINE = re.compile(r'Temporary failure in name resolution|Could not resolve|Network is unreachable|'
                     r'Failed to establish a new connection|No matching distribution found|Unable to locate package|'
                     r'Could not connect|ENOTFOUND|EAI_AGAIN|getaddrinfo|Name or service not known|'
                     r'Cannot initiate the connection|unable to access \'https?://|Failed to fetch')
STEP_CHARS, RESULT_CHARS, TRACE_CHARS = 2000, 1200, 140_000

JUDGE_SYSTEM = """You grade one attempt of an AI agent at a terminal task, to decide whether the task and the attempt are fit for training data. The task has no automatic verifier. You receive the instruction the agent saw, the task author's private solution sketch (the agent never saw it), the agent's trace (its messages, tool calls and the heads and tails of tool results; its reasoning is omitted), and the final state recorded after the agent stopped (each declared output: existence, size, sha256, head; and a file tree).

The sandbox has no internet: packages that are not already installed cannot be installed. That is by design; the task should be solvable without them.

Judge from evidence in the trace and the final state, not from the agent's own claims. Then return only a JSON object:
{
 "outcome": "solved" | "partial" | "failed",
 "requirements_met": <fraction 0-1 of the instruction's requirements the evidence shows met>,
 "cause": null if solved, else one of "agent_mistake", "stopped_early", "ran_out_of_time_or_context", "offline_package", "environment_defect", "task_defect",
 "task_defect": null, or one sentence on what is wrong with the TASK itself (not the agent): contradictory or impossible requirements, missing or broken inputs the instruction promises, data that cannot support the requested result, a needed tool absent with no workaround, an answer readable in the environment,
 "too_easy": true if a competent engineer would finish it in a few commands with no real problem solving,
 "summary": "<two sentences: what the agent did and where it ended>"
}"""

JUDGE_USER = """# Instruction (what the agent saw)
{instruction}

# Author's private solution sketch
{sketch}

# Trace
{trace}

# Final state after the agent stopped
{final_state}
"""


def session_messages(rec):
    for s in rec.get('sessions') or []:
        for line in s['text'].splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get('type') == 'message' and d.get('message'):
                yield d['message']


def clip(text, n):
    text = text or ''
    return text if len(text) <= n else f'{text[:n // 2]}\n[... {len(text) - n} characters ...]\n{text[-n // 2:]}'


def facts(rec):
    """Mechanical facts of one archived attempt."""
    tools, installs, steps, calls = collections.Counter(), [], [], {}
    final_text, assistant_turns = '', 0
    for m in session_messages(rec):
        role, content = m.get('role'), m.get('content')
        items = content if isinstance(content, list) else [{'type': 'text', 'text': content or ''}]
        if role == 'assistant':
            assistant_turns += 1
            text = '\n'.join(i.get('text', '') for i in items if i.get('type') == 'text').strip()
            if text:
                final_text = text
                steps.append(('assistant', clip(text, STEP_CHARS)))
            for i in items:
                if i.get('type') == 'toolCall':
                    tools[i.get('name')] += 1
                    args = i.get('arguments') or {}
                    cmd = args.get('command') if isinstance(args, dict) else None
                    body = cmd if cmd is not None else json.dumps(args)
                    calls[i.get('id')] = cmd or ''
                    steps.append((f'call {i.get("name")}', clip(body, STEP_CHARS)))
        elif role == 'toolResult':
            text = '\n'.join(i.get('text', '') for i in items if i.get('type') == 'text')
            steps.append((f'result{" (error)" if m.get("isError") else ""}', clip(text, RESULT_CHARS)))
            cmd = calls.get(m.get('toolCallId'), '')
            if INSTALL.search(cmd):
                miss = OFFLINE.search(text)
                installs.append({'command': cmd[:300], 'offline_failure': bool(miss),
                                 'marker': miss.group(0) if miss else None})
        elif role == 'user' and not steps:
            steps.append(('user', clip('\n'.join(i.get('text', '') for i in items if i.get('type') == 'text'), 600)))
    wr = rec.get('worker_result') or {}
    r = rec.get('record') or {}
    return {'id': rec['id'], 'task': rec['task'], 'tokens': r.get('tokens'), 'usable': r.get('usable'),
            'truncated': rec.get('truncated'), 'interrupted': wr.get('interrupted'),
            'termination': wr.get('termination_reason'),
            'minutes': (wr['finished_at'] - wr['started_at']) / 60 if wr.get('finished_at') and wr.get('started_at') else None,
            'assistant_turns': assistant_turns, 'tools': dict(tools), 'installs': installs,
            'offline_miss': any(i['offline_failure'] for i in installs), 'final_text': clip(final_text, 3000),
            'steps': steps}


def render_trace(steps):
    lines = [f'[{k}] {v}' for k, v in steps]
    total = sum(len(x) for x in lines)
    if total <= TRACE_CHARS:
        return '\n'.join(lines)
    head, tail, size = [], [], 0
    for x in lines:              # first 30% of the budget from the start, the rest from the end
        if size + len(x) > TRACE_CHARS * 0.3:
            break
        head.append(x)
        size += len(x)
    for x in reversed(lines[len(head):]):
        if size + len(x) > TRACE_CHARS:
            break
        tail.append(x)
        size += len(x)
    return '\n'.join(head + [f'[... {len(lines) - len(head) - len(tail)} steps omitted ...]'] + tail[::-1])


_ZIPS, _INDEX = {}, {}


def read_raw(root, uid, relpath, kind='train'):
    """synth.archive.read_raw with the shard index and zip handles kept open: a judge pass reads two files of every
    attempt, and reopening a shard zip (the central directory of ~1,000 attempts) cost ~0.17 s per read."""
    path = Path(root) / kind / 'raw' / uid / relpath
    try:   # the directory first, then the index: an attempt packed and deleted in between is in the index (#236)
        return path.read_bytes()
    except FileNotFoundError:
        pass
    key = (str(root), kind)
    index = _INDEX.get(key)
    if index is None or uid not in index['attempts']:   # an attempt packed since the last read: reload the index
        index = _INDEX[key] = archive.load_index(root, kind)
    if uid not in index['attempts']:
        raise FileNotFoundError(path)
    zpath = archive.folder(root, kind) / index['shards'][str(index['attempts'][uid][0])]['zip']
    z = _ZIPS.get(zpath)
    if z is None:
        z = _ZIPS[zpath] = zipfile.ZipFile(zpath)   # reads are thread-safe (zipfile locks the shared file)
    return z.read(f'{uid}/{relpath}')


def trial_file(root, rec, relpath):
    sessions = (rec.get('record') or {}).get('native_sessions') or [s['path'] for s in rec.get('sessions') or []]
    if not sessions:
        return None
    trial = Path(sessions[0]).parts[1]
    try:
        return read_raw(root, rec['id'], f'harbor/{trial}/{relpath}')
    except (KeyError, FileNotFoundError):
        return None


def final_state(root, rec):
    raw = trial_file(root, rec, 'verifier/final_state.json')
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def package_requests(root, rec):
    """synth.package_proxy log of the attempt: allowed/blocked request counts and blocked hosts (None: no proxy)."""
    raw = trial_file(root, rec, 'agent/package-proxy/requests.jsonl')
    if raw is None:
        return None
    rows = [json.loads(l) for l in raw.decode(errors='replace').splitlines() if l.strip()]
    return {'allowed': sum(r['allowed'] for r in rows), 'blocked': sum(not r['allowed'] for r in rows),
            'blocked_hosts': sorted({r['host'] for r in rows if not r['allowed'] and r.get('host')})[:10],
            'failed': sum(1 for r in rows if r['allowed'] and r.get('status') not in (None, 200))}


def task_source(manifest, task):
    m = manifest[task]
    if m.get('dir'):
        d = Path(m['dir'])
        return (d / 'instruction.md').read_text(), (d / 'solution' / 'sketch.md').read_text()
    from .roundpack import RoundFiles   # a round's task, loose or packed (#245)
    files = RoundFiles(ROUNDS / m['round'])
    return files.text(m['spec_id'], 'instruction.md', 'tasks'), files.text(m['spec_id'], 'solution/sketch.md', 'tasks')


def parse_verdict(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object')
    j = json.loads(m.group(0))
    if j.get('outcome') not in ('solved', 'partial', 'failed'):
        raise ValueError(f'outcome {j.get("outcome")!r}')
    return {k: j.get(k) for k in ('outcome', 'requirements_met', 'cause', 'task_defect', 'too_easy', 'summary')}


def new_attempts(root, known, kind='train'):
    """archive.read_attempts without what is `known`: a shard whose attempts are all known is not opened and only
    unknown raw directories are summarized, so a --follow rescan stays cheap. The read is consistent while the
    campaign is packed (#236); a usable, finalized attempt that does not come back is logged and read again on the
    next rescan."""
    return archive.read_attempts(root, kind, known=known, check='warn', log=lambda m: print(m, flush=True))


async def run(a):
    from .llm import Endpoint, wait_healthy
    root = Path(a.campaign)
    name = a.name or root.name
    out = TRACES / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CAMPAIGNS / a.bundle / 'manifest.json').read_text())
    done = {}
    if (out / 'records.jsonl').exists():
        text = (out / 'records.jsonl').read_text()
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except ValueError:   # a judge killed mid-append leaves a partial line: that attempt is judged again
                print(f'{name}: skipping a partial line in records.jsonl ({len(line)} chars)', flush=True)
                continue
            if r.get('verdict') or not a.judge:
                done[r['id']] = r
        if text and not text.endswith('\n'):   # start appends on a fresh line, not glued to a partial one
            with open(out / 'records.jsonl', 'a') as fh:
                fh.write('\n')
    await wait_healthy(a.judge)
    llm = Endpoint(a.judge, max_in_flight=a.concurrency)
    slots = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    records = list(done.values())

    def prepare(rec):
        f = facts(rec)
        f['final_state'] = final_state(root, rec)
        f['package_requests'] = package_requests(root, rec)
        instruction, sketch = task_source(manifest, rec['task'])
        fs = json.dumps(f['final_state'], indent=1)[:40_000] if f['final_state'] else '(not recorded)'
        msgs = [{'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user', 'content': JUDGE_USER.format(instruction=instruction, sketch=sketch,
                                                              trace=render_trace(f['steps']), final_state=fs)}]
        return f, msgs

    async def one(rec):
        async with slots:   # prepare inside the slot (only in-flight traces are held prepared), in a thread so the
            f, msgs = await asyncio.to_thread(prepare, rec)   # loop keeps sending and receiving meanwhile
            for _ in range(3):
                content, _, usage = await llm.chat(msgs)   # no cap: the server context (132k) bounds the reply
                if usage.get('finish') == 'length':   # filled the context (a runaway, ~30 min): not again in this
                    f['verdict_error'] = 'reply filled the context'   # run; a later --once pass may retry it
                    tried[rec['id']] = 2
                    break
                try:
                    f['verdict'] = {**parse_verdict(content), 'usage': usage}
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    f['verdict_error'] = f'{e}: {content[-300:]}'
        f.pop('steps')
        async with lock:
            records.append(f)
            if f.get('verdict'):
                done[f['id']] = f
            with open(out / 'records.jsonl', 'a') as fh:
                fh.write(json.dumps(f) + '\n')

    # finished attempts only (a running campaign's raw/ also holds live attempts). Streaming (default): rescan every
    # --interval seconds while the campaign runs and judge new attempts as they finish (no batch boundary: a slow verdict holds
    # only itself); exit once the campaign is complete and every attempt is judged or tried twice.
    inflight, tried, tasks = {}, collections.Counter(), set()
    t0, scheduled = time.time(), 0

    async def guarded(rec):
        try:
            await one(rec)
        except Exception as e:  # noqa: BLE001  (one trace must not stop the others)
            print(f'{rec.get("id")}: {type(e).__name__}: {e}', flush=True)
        finally:
            inflight.pop(rec['id'], None)

    while True:
        known = set(done) | set(inflight) | {i for i, n in tried.items() if n >= 2}
        try:
            progress = read(root / 'ops' / 'progress.json') if a.follow else None
            phase = (progress or {}).get('phase', 'complete')   # no progress.json: not a live synth campaign, one pass
            manifest.update(json.loads((CAMPAIGNS / a.bundle / 'manifest.json').read_text()))
            todo = await asyncio.to_thread(   # a rescan reads shards for ~1 min: off the loop, so replies keep flowing
                lambda: [rec for rec in new_attempts(root, known) if (rec.get('worker_result') or {}).get('finished_at')
                         and rec.get('task') in manifest])   # an attempt not exported yet has no task
        except Exception as e:  # noqa: BLE001  (a failed rescan must not drop the requests in flight)
            if not a.follow:
                raise
            print(f'{name}: rescan failed ({type(e).__name__}: {e}); retrying in {a.interval}s', flush=True)
            await asyncio.sleep(a.interval)
            continue
        if a.per_task:   # judge only the first attempt(s) of each task (defect flags per task, as with one trace per task)
            todo.sort(key=lambda r: (r['task'], r.get('attempt', 0), r.get('retry', 0)))
            seen = collections.Counter([r['task'] for r in records] + list(inflight.values()))
            kept = []
            for r in todo:
                if seen[r['task']] < a.per_task:
                    seen[r['task']] += 1
                    kept.append(r)
            todo = kept
        if todo or not a.follow:
            print(f'{name}: judging {len(todo)} attempts ({len(done)} done, {len(inflight)} in flight)', flush=True)
        for rec in todo:
            inflight[rec['id']] = rec['task']
            tried[rec['id']] += 1
            tasks.add(asyncio.create_task(guarded(rec)))
        scheduled += len(todo)
        tasks = {t for t in tasks if not t.done()}
        if not a.follow or (phase == 'complete' and not todo and not tasks):
            break
        await asyncio.sleep(a.interval)
    await asyncio.gather(*tasks)
    # rewrite without duplicates (a rerun appends a retried attempt again)
    latest = {r['id']: r for r in records}
    (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in latest.values()))
    print(f'{name}: {scheduled} judged in {time.time() - t0:.0f}s', flush=True)
    report(name, manifest)


def bundle_source(root):
    """task -> (instruction, solution text) from a campaign's train bundle and stream chunks (verifier-scored
    campaigns)."""
    import tarfile
    from synth.stream import campaign_tars
    out = {}
    for tar in campaign_tars(root):
        with tarfile.open(tar) as t:
            for m in t:
                parts = m.name.split('/')
                if not m.isfile() or len(parts) < 3:
                    continue
                if parts[2] == 'instruction.md':
                    out.setdefault(parts[1], ['', ''])[0] = t.extractfile(m).read().decode(errors='replace')
                elif parts[2] == 'solution' and m.size < 200_000:
                    out.setdefault(parts[1], ['', ''])[1] += f'## {"/".join(parts[3:])}\n' + t.extractfile(m).read().decode(errors='replace')[:6000]
    return out


async def calibrate(a):
    """Judge Super traces of verifier-scored campaigns and compare with their rewards: how lenient is 'solved'
    when the reward is known? No final state is recorded there, so the judge works from the trace (and the
    reference solution where the bundle has one)."""
    import random
    from .llm import Endpoint, wait_healthy
    out = TRACES / 'calibration'
    out.mkdir(parents=True, exist_ok=True)
    await wait_healthy(a.judge)
    llm = Endpoint(a.judge, max_in_flight=a.concurrency)
    slots = asyncio.Semaphore(a.concurrency)
    rows = []

    async def one(root, rec, src):
        f = facts(rec)
        instruction, solution = src
        msgs = [{'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user', 'content': JUDGE_USER.format(
                    instruction=instruction, sketch=solution or '(no reference solution available)',
                    trace=render_trace(f['steps']),
                    final_state='(not recorded: judge from the trace and its tool outputs)')}]
        async with slots:
            for _ in range(3):
                content, _, usage = await llm.chat(msgs)
                try:
                    v = parse_verdict(content)
                    break
                except (ValueError, json.JSONDecodeError):
                    v = None
        rows.append({'campaign': Path(root).name, 'task': rec['task'], 'id': rec['id'], 'reward': rec['reward'],
                     'has_solution': bool(solution), 'tokens': (rec.get('record') or {}).get('tokens'), 'verdict': v})

    jobs = []
    for root in a.campaigns:
        src = bundle_source(root)
        pool = []
        for rec in archive.iter_attempts(root, 'train'):
            if rec.get('reward') is not None and not rec.get('verifier_infra_failure') and rec['task'] in src \
                    and rec.get('sessions'):
                pool.append(rec)
            if len(pool) >= 20 * a.n:
                break
        for rec in random.Random(a.seed).sample(pool, min(a.n, len(pool))):
            jobs.append(one(root, rec, src[rec['task']]))
    print(f'calibration: judging {len(jobs)} traces', flush=True)
    await asyncio.gather(*jobs)
    (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    lines = ['# Trace judge calibration', '', '| campaign | n | reward=1 | judge solved | solved & r=1 | solved & r=0 | '
             'not solved & r=1 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for c in sorted({r['campaign'] for r in rows}):
        rs = [r for r in rows if r['campaign'] == c and r['verdict']]
        s = lambda r: r['verdict']['outcome'] == 'solved'
        lines.append(f'| {c} | {len(rs)} | {sum(r["reward"] >= 1 for r in rs)} | {sum(map(s, rs))} | '
                     f'{sum(s(r) and r["reward"] >= 1 for r in rs)} | {sum(s(r) and r["reward"] < 1 for r in rs)} | '
                     f'{sum(not s(r) and r["reward"] >= 1 for r in rs)} |')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def report(name, manifest=None):
    out = TRACES / name
    recs = [json.loads(l) for l in (out / 'records.jsonl').read_text().splitlines()]
    if manifest is None:
        bundle = json.loads((out / 'run.json').read_text())['bundle'] if (out / 'run.json').exists() else None
        manifest = json.loads((CAMPAIGNS / bundle / 'manifest.json').read_text()) if bundle else {}
    n = len(recs)
    pct = lambda k: f'{k}/{n} ({100 * k / max(n, 1):.0f}%)'
    tok = [r['tokens'] for r in recs if r.get('tokens')]
    lines = [f'# Traces: {name}', '', f'{n} attempts on {len({r["task"] for r in recs})} tasks.', '',
             '## Mechanics', '',
             f'- Usable: {pct(sum(bool(r.get("usable")) for r in recs))}; truncated {sum(bool(r.get("truncated")) for r in recs)}, '
             f'interrupted {sum(bool(r.get("interrupted")) for r in recs)}',
             f'- Termination: {dict(collections.Counter(r.get("termination") for r in recs))}',
             f'- Saved tokens per trace: mean {statistics.mean(tok):.0f}, median {q(tok, .5)}, p90 {q(tok, .9)}, max {max(tok)}' if tok else '- no tokens',
             f'- Assistant turns: median {q([r["assistant_turns"] for r in recs], .5)}, p90 {q([r["assistant_turns"] for r in recs], .9)}',
             f'- Minutes: median {q([r["minutes"] for r in recs], .5):.1f}, p90 {q([r["minutes"] for r in recs], .9):.1f}' if any(r.get('minutes') for r in recs) else '',
             f'- Final state recorded: {pct(sum(r.get("final_state") is not None for r in recs))}', '',
             '## Offline package misses', '',
             f'- Traces with an install attempt: {pct(sum(bool(r["installs"]) for r in recs))}',
             f'- Traces where an install failed offline: {pct(sum(r["offline_miss"] for r in recs))}']
    missed = collections.Counter()
    for r in recs:
        for i in r['installs']:
            if i['offline_failure']:
                missed[re.sub(r'\s+', ' ', i['command'])[:90]] += 1
    lines += [f'  - `{c}` x{k}' for c, k in missed.most_common(15)]
    pk = [r['package_requests'] for r in recs if r.get('package_requests') is not None]
    if pk:
        hosts = collections.Counter(h for x in pk for h in x['blocked_hosts'])
        lines += [f'- Package proxy: {len(pk)} traces logged; {sum(x["allowed"] > 0 for x in pk)} fetched from a registry '
                  f'({sum(x["allowed"] for x in pk)} requests, {sum(x["failed"] for x in pk)} failed upstream); '
                  f'{sum(x["blocked"] > 0 for x in pk)} had blocked requests: {dict(hosts.most_common(10))}']
    v = [r['verdict'] for r in recs if r.get('verdict')]
    lines += ['', '## Judge', '', f'- Verdicts: {len(v)}/{n}']
    if v:
        lines += [f'- Outcome: {dict(collections.Counter(x["outcome"] for x in v))}',
                  f'- Requirements met: mean {statistics.mean(float(x.get("requirements_met") or 0) for x in v):.2f}',
                  f'- Cause when not solved: {dict(collections.Counter(x["cause"] for x in v if x["outcome"] != "solved"))}',
                  f'- Task defects reported: {sum(bool(x.get("task_defect")) for x in v)}',
                  f'- Too easy: {sum(bool(x.get("too_easy")) for x in v)}', '', '### Task defects', '']
        lines += [f'- {r["task"]} ({manifest.get(r["task"], {}).get("slug")}): {r["verdict"]["task_defect"]}'
                  for r in recs if r.get('verdict') and r['verdict'].get('task_defect')]
        by = collections.defaultdict(list)
        for r in recs:
            if r.get('verdict'):
                m = manifest.get(r['task'], {})
                by[f'minutes<=60' if (m.get('minutes') or 0) <= 60 else 'minutes>60'].append(r['verdict']['outcome'])
                by['services' if m.get('services') else 'no services'].append(r['verdict']['outcome'])
        lines += ['', '### Slices (solved/partial/failed)', '']
        for k, xs in sorted(by.items()):
            c = collections.Counter(xs)
            lines.append(f'- {k}: {c["solved"]}/{c["partial"]}/{c["failed"]}')
    (out / 'REPORT.md').write_text('\n'.join(l for l in lines if l is not None) + '\n')
    print((out / 'REPORT.md').read_text())


def show(name, task, full=False):
    for line in (TRACES / name / 'records.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['task'] == task:
            r = {k: v for k, v in r.items() if full or k != 'final_state'}
            print(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('run')
    p.add_argument('--campaign', required=True, help='synth campaign root')
    p.add_argument('--bundle', required=True, help='task_synthesis.pipeline.campaign bundle name, or a task stream directory')
    p.add_argument('--name')
    p.add_argument('--judge', default='gen-ds')
    p.add_argument('--concurrency', type=int, default=128)
    p.add_argument('--per-task', type=int, default=0, help='judge at most this many attempts per task (0: all)')
    p.add_argument('--once', dest='follow', action='store_false',
                   help='one pass over the attempts finished so far (default: stream until the campaign is complete)')
    p.add_argument('--follow', action='store_true', default=True, help=argparse.SUPPRESS)   # the default; kept for old callers
    p.add_argument('--interval', type=int, default=120, help='seconds between rescans while streaming')
    p = sub.add_parser('calibrate')
    p.add_argument('--campaigns', nargs='+', required=True)
    p.add_argument('--n', type=int, default=60, help='traces per campaign')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--judge', default='gen-ds')
    p.add_argument('--concurrency', type=int, default=200)
    p = sub.add_parser('report')
    p.add_argument('--name', required=True)
    p = sub.add_parser('show')
    p.add_argument('--name', required=True)
    p.add_argument('task')
    p.add_argument('--full', action='store_true')
    a = ap.parse_args()
    if a.op == 'run':
        out = TRACES / (a.name or Path(a.campaign).name)
        out.mkdir(parents=True, exist_ok=True)
        atomic(out / 'run.json', {'campaign': a.campaign, 'bundle': a.bundle, 'judge': a.judge, 'time': time.time()})
        asyncio.run(run(a))
    elif a.op == 'calibrate':
        asyncio.run(calibrate(a))
    elif a.op == 'report':
        report(a.name)
    else:
        show(a.name, a.task, a.full)


if __name__ == '__main__':
    main()

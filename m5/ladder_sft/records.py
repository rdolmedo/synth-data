"""M5 ladder SFT records: chat records {id, group, arm, split, messages, tools, weights, cut_spans, meta} for every new
trajectory of the mixture (user 2026-11-15: M=2, finishing at the train cap, k=4, non-finishers trained too).

Loss rule (as the W-v2 cap SFT, build_data.py):
  * an assistant message cut by the response budget (at the warning, by the final allowance, or by the hard total)
    trains its REAL tokens only; the template's synthetic closure ('\n</think>\n\n' for a reasoning cut, the EOT)
    is masked (cut_spans: message indices; render.py applies the span mask);
  * a natural stop / tool call trains in full; CUT + notice user messages and tool results are context;
  * as native Pi SFT (prepare._native_record): a turn capped by the provider BEFORE the warning (per-response max
    tokens, v4 context clamp) stays zero-loss context, zero-output turns are context, trailing zero-loss messages are
    cut;
  * Pi records come from the response-budget proxy's wire log (response-budget-wire.jsonl): the history exactly as the
    model saw it (CUT marker, budget + v4 notices joined to the trailing message, live tool results); retained
    assistant messages are matched to wire responses in order (Pi may discard a response and retry).

Sources (kind):
  t2_flat    t2/<suite>/flat/*.jsonl (gen_flat.py)                    -> records/t2_flat/<suite>.jsonl
  t2_pi      t2/runs/m5l_<suite>_t2/*/attempts/*/trial.tar.gz          -> records/t2_pi/<suite>.jsonl
  fin_flat   finish/results/<suite>/*.json + finish/inputs/<suite>/*.json -> records/fin_flat/<suite>.jsonl
  fin_pi     finish/runs/m5l_<suite>_finish/r*/... (first natural round, else the last) -> records/fin_pi/<suite>.jsonl
  fin_tb     tb/finish/runs/m5l_tb_finish/r*/...                       -> records/fin_pi/tb.jsonl

  python records.py <kind> <suite> [--workers 16] [--limit N]
"""
import argparse
import collections
import concurrent.futures
import copy
import glob
import json
from pathlib import Path
import re
import sys
import tarfile

L = Path('/data/aai/m5/ladder')
CODE = Path('/data/aai/m5/validation/worktrees/W-v2-code')
sys.path[:0] = [str(CODE)]
from tb_agents import response_budget as B   # noqa: E402

OUT = L / 'sft' / 'records'
EOT = '<|eot_id|>'
CLOSE = '\n</think>\n\n'
PI = {'automationbench', 'gdpval', 'critpt', 'tbsci_lite', 'tb'}
# Harbor scrubs the agent's sensitive env values from the trial dir ("[REDACTED]", harbor/trial/trial.py
# _scrub_jobs_dir), but pi-cont16k/ (the wire log) sits outside it. pi_eval's only sensitive agent env value is the
# local endpoint's dummy OPENAI_API_KEY; mirror the scrub, and drop a record if anything credential-like remains.
AGENT_SECRETS = ('sk-local-dummy',)
CREDENTIAL = re.compile(r'(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}'
                        r'|sk-ant-[A-Za-z0-9_-]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{40,}|AKIA[0-9A-Z]{16}'
                        r'|xox[bpas]-[A-Za-z0-9-]{20,})')


def scrub(o):
    if isinstance(o, str):
        for x in AGENT_SECRETS:
            o = o.replace(x, '[REDACTED]')
        return o
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    return o


def real_text(m):
    """Body text (after the assistant header + '<think>\\n') of the tokens the model really generated in a cut message:
    the template's closing '</think>' of a reasoning cut and the EOT are synthetic."""
    b = B.body(m)
    assert b.endswith(EOT), b[-40:]
    b = b[:-len(EOT)]
    if not m.get('content') and not m.get('tool_calls') and b.endswith(CLOSE):
        b = b[:-len(CLOSE)]
    return b


def has_output(m):
    return bool((m.get('reasoning_content') or '').strip() or (m.get('content') or '').strip() or m.get('tool_calls'))


def trim(rec):
    """Cut trailing zero-loss messages (native Pi SFT); None if nothing trains."""
    w = rec['weights']
    if not any(w):
        return None
    last = max(i for i, x in enumerate(w) if x)
    rec['messages'], rec['weights'] = rec['messages'][:last + 1], w[:last + 1]
    rec['cut_spans'] = [i for i in rec['cut_spans'] if i <= last]
    return rec


# ---------------------------------------------------------------- single-turn
def chain(prefix, first_cut, attempt_msgs, turn_reasons):
    """messages/weights/cut_spans for prefix (+ cut assistant) + [user CUT+notice, assistant]* of one attempt."""
    msgs, weights, cuts = list(prefix), [0] * len(prefix), []
    if first_cut is not None:
        msgs.append(first_cut)
        weights.append(1)
        cuts.append(len(msgs) - 1)
    reasons = iter(turn_reasons)
    for m in attempt_msgs:
        msgs.append(m)
        if m['role'] != 'assistant':
            weights.append(0)
            continue
        fr = next(reasons)
        if not has_output(m):
            weights.append(0)
        elif fr == 'stop':
            weights.append(1)
        else:   # 'length' (allowance / hard total) or an aborted call: real tokens only
            weights.append(1)
            cuts.append(len(msgs) - 1)
    return msgs, weights, cuts


def t2_flat(suite, limit=0):
    n = 0
    for f in sorted((L / 't2' / suite / 'flat').glob('shard-*.jsonl')):
        for line in f.open():
            n += 1
            if limit and n > limit:
                return
            try:
                r = json.loads(line)
            except ValueError:   # a writer killed mid-line (HLE shard-00-of-01 before the 3-way reshard, which redid it)
                print(f'{f.name}: skipping a truncated line ({len(line)} bytes)', flush=True)
                continue
            p1 = r['phase1']
            base = {'id': r['id'], 'group': r['group'], 'arm': suite, 'split': 'train', 'tools': None, 'kind': 't2_flat'}
            if not r['cut']:
                if p1['finish_reason'] != 'stop' or not has_output(p1['message']):
                    yield {**base, 'drop': f"phase1:{p1['finish_reason']}"}
                    continue
                yield {**base, 'messages': r['prompt'] + [p1['message']], 'weights': [0] * len(r['prompt']) + [1],
                       'cut_spans': [], 'meta': {'cut': False, 'tokens': p1['tokens'], 'budget': r['budget']}}
                continue
            a = r['finishing'][r['chosen']]
            msgs, w, cuts = chain(r['prompt'], p1['message'], a['messages'], [t['finish_reason'] for t in a['turns']])
            yield {**base, 'messages': msgs, 'weights': w, 'cut_spans': cuts,
                   'meta': {'cut': True, 'natural': a['natural'], 'attempts': len(r['finishing']), 'chosen': r['chosen'],
                            'phase1_tokens': p1['tokens'], 'tokens_after_warning': a['tokens_after_warning'],
                            'budget': r['budget'], 'near_window_bound': r['near_window_bound']}}


def fin_flat(suite, limit=0):
    for k, p in enumerate(sorted((L / 'finish' / 'results' / suite).glob('*.json'))):
        if limit and k >= limit:
            return
        r = json.loads(p.read_text())
        src = json.loads((L / 'finish' / 'inputs' / suite / p.name).read_text())
        a = r['attempts'][r['chosen']]
        prefix, cut = src['messages'][:-1], src['messages'][-1]
        assert cut['role'] == 'assistant'
        msgs, w, cuts = chain(prefix, cut, a['messages'], [c['finish_reason'] for c in a['calls']])
        # the prefix keeps the record's own weights (system/user context 0)
        w[:len(prefix)] = src['weights'][:len(prefix)]
        yield {'id': src['id'], 'group': src['group'], 'arm': suite, 'split': 'train', 'tools': src.get('tools'),
               'kind': 'fin_flat', 'replaces': src['id'], 'messages': msgs, 'weights': w, 'cut_spans': cuts,
               'meta': {'natural': r['natural'], 'attempts': len(r['attempts']), 'chosen': r['chosen'],
                        'threshold': r['threshold'], 'allowance': r['allowance'], 'cut': src['cut'],
                        'tokens_after_warning': a['tokens_after_warning']}}


# ---------------------------------------------------------------- Pi (wire)
def read_wire(tar):
    with tarfile.open(tar) as tf:
        names = [x for x in tf.getmembers() if x.name.endswith('pi-cont16k/response-budget-wire.jsonl')]
        if len(names) != 1:
            return None
        W, last_msgs, last_tools = [], None, None
        lines = tf.extractfile(names[0]).read().splitlines()
        for i, line in enumerate(lines):
            try:
                x = json.loads(line)
            except ValueError:
                if i == len(lines) - 1:   # last line cut by a killed session: the wire up to it is complete
                    break
                # torn write: a cut fragment, then the complete record on the same line (8 TB-Sci-lite attempts, 11-15)
                j = line.rfind(b'{"time": ')
                if j <= 0:
                    raise
                x = json.loads(line[j:])
            if 'turn' in x and 'response' in x:
                last_msgs, last_tools = x.pop('messages'), x.pop('tools', None)   # keep only the last history
            W.append(x)
    return W, last_msgs, last_tools


def pi_record(tar, src=None):
    """One Pi trajectory from its wire log. src: the finishing input record (replayed turns keep its weights)."""
    got = read_wire(tar)
    if got is None:
        return {'drop': 'no_wire'}
    W, hist, tools = got
    T = [w for w in W if 'turn' in w and 'response' in w]
    if not T:
        return {'drop': 'no_turns'}
    warn = int(T[0]['cap'])   # output cap at used 0 = the warning threshold
    final = T[-1]['response']['choices'][0]['message']
    msgs = [B.normalize(m) for m in hist + [final]]
    mi = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
    try:
        bodies = [B.body(t['response']['choices'][0]['message']) for t in T]
        hb = {i: B.body(msgs[i]) for i in mi}
    except (ValueError, TypeError, KeyError) as e:
        return {'drop': 'malformed_message', 'error': str(e)[:200]}
    retained, cursor = [], 0
    for i in mi:
        j = next((k for k in range(cursor, len(T)) if bodies[k] == hb[i]), None)
        if j is None:
            return {'drop': 'history_not_on_wire', 'message_index': i}
        retained.append((i, T[j]))
        cursor = j + 1
    # Pi re-sends history with the reasoning inlined as '<think>\n...</think>' content (the template splits it back, so
    # the render is identical), which hides where a cut fell: use each retained turn's structured wire response.
    for i, t in retained:
        r = B.normalize(t['response']['choices'][0]['message'])
        msgs[i] = {k: r[k] for k in ('role', 'reasoning_content', 'content', 'tool_calls') if r.get(k) is not None}
        msgs[i].setdefault('content', '')
        assert B.body(msgs[i]) == hb[i], 'structured response renders differently from the history'
    src_w = [w for m, w in zip(src['messages'], src['weights']) if m['role'] == 'assistant'] if src else []
    src_cut = len(src['replay']) - 1 if src and src['cut']['kind'] in ('reasoning', 'text') else None
    weights, cuts = [0] * len(msgs), []
    for i, t in retained:
        m = msgs[i]
        fr = t['response']['choices'][0]['finish_reason']
        if t['replayed']:
            k = t['turn']
            if k == src_cut:
                weights[i] = 1
                cuts.append(i)
            else:
                weights[i] = src_w[k] if k < len(src_w) else 0
            continue
        if not has_output(m):
            continue
        if fr == 'length' or t.get('clipped'):
            if int(t['used_after']) >= warn:   # cut by the response budget: real tokens train
                weights[i] = 1
                cuts.append(i)
            continue   # capped by the provider before the warning: zero-loss context (native rule)
        weights[i] = 1
    last = T[-1]['response']['choices'][0]
    natural = (not W[-1].get('synthetic')) and last.get('finish_reason') == 'stop' and not (last.get('message') or {}).get('tool_calls')
    msgs, tools = scrub(msgs), scrub(tools)
    hit = CREDENTIAL.search(json.dumps([msgs, tools], ensure_ascii=False))
    if hit:
        return {'drop': 'credential_like_text', 'pattern': hit.group(0)[:6]}
    # cumulative budget after each retained assistant message (Pi's discarded retries are charged to the next retained
    # turn), so a resample can replay the trajectory to its warning point with the original charges
    charges = [[i, int(t['used_after']), bool(t['replayed']), t['response']['choices'][0]['finish_reason']] for i, t in retained]
    rec = {'messages': msgs, 'tools': tools, 'weights': weights, 'cut_spans': cuts,
           'meta': {'natural': natural, 'warn': warn, 'used': int(T[-1]['used_after']), 'wire_turns': len(T), 'charges': charges,
                    'discarded_wire_responses': len(T) - len(retained), 'warned': int(T[-1]['used_after']) >= warn,
                    'replay_mismatches': sum(1 for w in W if 'replay_tool' in w and not w.get('matched_up_to_volatile'))}}
    return rec


def _t2_pi_one(job):
    try:
        return _t2_pi_one_(job)
    except Exception as e:   # one unreadable attempt (archive being packed by a live run, corrupt wire) never sinks a build
        att = Path(job[0]).parent
        return {'id': f'{job[1]}:t2/{att.parent.parent.name}/{att.name}', 'group': f"{job[1]}:{att.name.rsplit('__a', 1)[0]}",
                'arm': job[1], 'split': 'train', 'kind': 't2_pi', 'drop': f'unreadable: {type(e).__name__}'}


def _t2_pi_one_(job):
    tar, suite = job
    att = Path(tar).parent
    outcome = json.loads((att / 'outcome.json').read_text()) if (att / 'outcome.json').exists() else {}
    task = att.name.rsplit('__a', 1)[0]
    base = {'id': f'{suite}:t2/{att.parent.parent.name}/{att.name}', 'group': f'{suite}:{task}', 'arm': suite,
            'split': 'train', 'kind': 't2_pi'}
    if outcome.get('infrastructure_failure'):
        return {**base, 'drop': 'infrastructure_failure'}
    r = pi_record(tar)
    r = {**base, **r}
    if 'drop' not in r:
        r['meta']['end'] = (outcome.get('stats') or {}).get('end_state')
        r['meta']['context_limit'] = bool((outcome.get('stats') or {}).get('context_limit'))
        r = trim(r) or {**base, 'drop': 'no_loss_turns'}
    return r


def t2_pi(suite, workers, limit=0):
    tars = sorted(glob.glob(str(L / 't2' / 'runs' / f'm5l_{suite}_t2' / '*' / 'attempts' / '*' / 'trial.tar.gz')))
    tars = tars[:limit] if limit else tars
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        yield from pool.map(_t2_pi_one, [(t, suite) for t in tars], chunksize=8)


def _fin_pi_one(job):
    try:
        return _fin_pi_one_(job)
    except Exception as e:   # as _t2_pi_one: one unreadable attempt is a counted drop, not a failed build
        att = Path(job[0]).parent
        return {'id': f'{job[1]}:{job[3]}/{att.parent.parent.name}/{att.name}', 'arm': job[1], 'kind': job[3],
                'drop': f'unreadable: {type(e).__name__}'}


def _fin_pi_one_(job):
    tar, suite, index_path, kind = job
    att = Path(tar).parent
    task = att.name.rsplit('__a', 1)[0]
    index = json.loads(Path(index_path).read_text())
    src = json.loads(Path(index[task]).read_text())
    outcome = json.loads((att / 'outcome.json').read_text()) if (att / 'outcome.json').exists() else {}
    base = {'id': src['id'] + ('#resample' if kind == 'fin_t2' else ''), 'group': src['group'], 'arm': suite,
            'split': 'train', 'kind': kind, 'replaces': src['id'], 'round': att.parent.parent.name}
    if outcome.get('infrastructure_failure'):
        return {**base, 'drop': 'infrastructure_failure'}
    r = {**base, **pi_record(tar, src)}
    if 'drop' not in r:
        r['meta'].update(cut=src['cut'], threshold=src['threshold'], end=(outcome.get('stats') or {}).get('end_state'))
        r = trim(r) or {**base, 'drop': 'no_loss_turns'}
    return r


def fin_pi(suite, workers, limit=0, kind='fin_pi'):
    """One record per finishing input: the first round whose trajectory finished naturally, else the last round run.
    kind fin_t2: the resamples (rounds 2..4) of warned, unfinished Pi second traces; they replace their t2 record."""
    if kind == 'fin_t2':
        runs, index_path = L / 't2/resample/runs' / f'm5l_{suite}_t2fin', L / 't2/resample' / f'{suite}_replay_index.json'
    else:
        runs = L / ('tb/finish/runs' if suite == 'tb' else 'finish/runs') / f'm5l_{suite}_finish'
        index_path = L / ('tb/finish/replay_index.json' if suite == 'tb' else f'finish/{suite}_replay_index.json')
    # round order (r2b = a split of r2, same round and seeds); retries in key order within a round
    rnd = lambda t: (int(re.match(r'r(\d+)', Path(t).parent.parent.parent.name).group(1)), Path(t).parent.name)
    tars = sorted(glob.glob(str(runs / 'r*' / 'attempts' / '*' / 'trial.tar.gz')), key=rnd)
    tars = tars[:limit] if limit else tars
    best = {}
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_fin_pi_one, [(t, suite, str(index_path), kind) for t in tars], chunksize=4):
            prev = best.get(r['id'])
            if prev is not None and 'drop' in r and 'drop' not in prev:
                continue   # a later round's failed attempt never replaces a real record (11-15 fix: it did)
            if prev is None or not (prev.get('meta') or {}).get('natural'):
                best[r['id']] = r   # rounds in order: a later round replaces an unfinished one
    yield from best.values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['t2_flat', 't2_pi', 'fin_flat', 'fin_pi', 'fin_t2'])
    ap.add_argument('suite')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    gen = {'t2_flat': lambda: t2_flat(a.suite, a.limit), 'fin_flat': lambda: fin_flat(a.suite, a.limit),
           't2_pi': lambda: t2_pi(a.suite, a.workers, a.limit), 'fin_pi': lambda: fin_pi(a.suite, a.workers, a.limit),
           'fin_t2': lambda: fin_pi(a.suite, a.workers, a.limit, kind='fin_t2')}[a.kind]
    out = Path(a.out) if a.out else OUT / a.kind / f'{a.suite}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    tmp = out.with_suffix('.tmp')
    with tmp.open('w') as fh:
        for r in gen():
            if 'drop' in r:
                stats['drop:' + r['drop']] += 1
                continue
            stats['kept'] += 1
            m = r.get('meta') or {}
            stats['cut_spans'] += len(r['cut_spans'])
            if m.get('cut') or m.get('warned'):
                stats['warned'] += 1
                stats['warned_natural'] += bool(m.get('natural'))
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(out)
    s = {'kind': a.kind, 'suite': a.suite, 'out': str(out), **stats}
    out.with_suffix('.stats.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)


if __name__ == '__main__':
    main()

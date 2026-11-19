"""SFT records from pi_eval attempts and finishing rounds: the reusable parts of the M5 ladder's record builder
(/data/aai/m5/ladder/sft/records.py, 11-15/16), with its campaign fixes (#451 workstream 2).

  read_wire(path)            an attempt's response-budget wire log (trial.tar.gz or the .jsonl): torn lines recovered
  pi_record(wire, src=None)  one Pi trajectory: messages as the model saw them, loss weights, cut spans, meta
  attempt_record(tar, base)  pi_record for one attempt dir; an unreadable attempt is a counted drop, never a crash
  keep_best(records)         one record per id over rounds in round order: the first natural finish, else the last
                             real record; a later round's drop never replaces a real record
  round_key(trial_tar)       the round order of an attempt (runs r1, r2, r2b = a split of r2; retries numerically)
  build_rounds(tars, ...)    attempt_record over every round in round order, then keep_best (base/src errors = drops)
  dedupe_lowest_id(records)  one record per task after a split's fence overshoot: the lowest id of its real
                             records. EVERY consumer
                             (SFT assembly, resample inputs, finishing inputs) applies this rule, never an assert.

Loss rule (as M5 / the W-v2 cap SFT): an assistant message cut by the response budget trains its real tokens only
(the template's synthetic closure is masked: cut_spans); a natural stop or tool call trains in full; a turn capped by
the provider BEFORE the warning stays zero-loss context, as do zero-output turns; trailing zero-loss messages are cut.
Records come from the response-budget proxy's wire log (tb_agents/response_budget_proxy.py): the history exactly as the
model saw it; retained assistant messages are matched to wire responses in order (Pi may discard a response and retry).

Failures these reproduce (m5/test_records.py):
  - torn wire writes (11-15): 8 TB-Sci-lite second traces had a line = cut fragment + whole record; the HLE and
    TB-Sci-lite resample builds each died once on an unreadable attempt;
  - keep-best (11-16 08:55): 'a later round replaces an unfinished one' let a later round's infra-failure DROP replace
    an earlier round's real record: 1 AB finishing and 18 AB resample records lost in the first AB build;
  - dedupe (11-16): a split's fence overshoot left two second traces for a few tasks; the CritPt resample-input build
    asserted one per task and crashed (~10 min), while SFT assembly kept the lowest id.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tarfile

from tb_agents import jsonlog
from tb_agents import response_budget as B

WIRE = 'pi-cont16k/response-budget-wire.jsonl'
# Harbor scrubs the agent's sensitive env values from the trial dir, but pi-cont16k/ (the wire log) sits outside it.
# pi_eval's only sensitive agent env value is the local endpoint's dummy key; mirror the scrub, and drop a record if
# anything credential-like remains.
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


class Wire:
    """An attempt's wire records (each turn's 'messages'/'tools' dropped but the last turn's: the history)."""

    def __init__(self, data: bytes):
        lines = jsonlog.parse(data)
        self.cut_tail, self.recovered, self.cut, self.bad = lines.cut_tail, lines.recovered, lines.cut, lines.bad
        self.records, self.messages, self.tools = [], None, None
        for x in lines.records:
            if 'turn' in x and 'response' in x:
                self.messages, self.tools = x.pop('messages'), x.pop('tools', None)
            self.records.append(x)


def read_wire(path) -> Wire | None:
    """The response-budget wire log of an attempt: `path` is its trial.tar.gz or the .jsonl itself. None if the
    archive holds no (or more than one) wire log."""
    path = Path(path)
    if path.name.endswith('.jsonl'):
        return Wire(path.read_bytes())
    with tarfile.open(path) as tf:
        names = [x for x in tf.getmembers() if x.name.endswith(WIRE)]
        if len(names) != 1:
            return None
        return Wire(tf.extractfile(names[0]).read())


def pi_record(wire: Wire | None, src: dict | None = None) -> dict:
    """One Pi trajectory from its wire log, or {'drop': reason}. src: the finishing input record (replayed turns keep
    its weights; its cut turn trains as a cut)."""
    if wire is None:
        return {'drop': 'no_wire'}
    if wire.bad:   # an unreadable line not from this writer: the history may not match what the model saw
        return {'drop': 'wire_unreadable_line', 'lines': wire.bad[:5]}
    W, hist, tools = wire.records, wire.messages, wire.tools
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
        if t.get('replayed'):
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
    charges = [[i, int(t['used_after']), bool(t.get('replayed')), t['response']['choices'][0]['finish_reason']]
               for i, t in retained]
    return {'messages': msgs, 'tools': tools, 'weights': weights, 'cut_spans': cuts,
            'meta': {'natural': natural, 'warn': warn, 'used': int(T[-1]['used_after']), 'wire_turns': len(T),
                     'charges': charges, 'discarded_wire_responses': len(T) - len(retained),
                     'warned': int(T[-1]['used_after']) >= warn, 'wire_cut_tail': wire.cut_tail,
                     'wire_recovered_lines': wire.recovered, 'wire_cut_lines': wire.cut,
                     'replay_mismatches': sum(1 for w in W if 'replay_tool' in w and not w.get('matched_up_to_volatile'))}}


def attempt_record(tar, base: dict, src: dict | None = None) -> dict:
    """The record of one attempt (its dir holds trial.tar.gz and outcome.json), merged into `base` (id, group, ...).
    Infra failures and unreadable attempts (an archive being packed by a live run, a corrupt wire) are counted drops:
    one bad attempt never sinks a build."""
    try:
        att = Path(tar).parent
        outcome = json.loads((att / 'outcome.json').read_text()) if (att / 'outcome.json').exists() else {}
        if outcome.get('infrastructure_failure'):
            return {**base, 'drop': 'infrastructure_failure'}
        r = {**base, **pi_record(read_wire(tar), src)}
        if 'drop' not in r:
            stats = outcome.get('stats') or {}
            r['meta']['end'] = stats.get('end_state')
            r['meta']['context_limit'] = bool(stats.get('context_limit'))
            if src is not None:
                r['meta'].update(cut=src.get('cut'), threshold=src.get('threshold'))
            r = trim(r) or {**base, 'drop': 'no_loss_turns'}
        return r
    except Exception as e:   # noqa: BLE001
        return {**base, 'drop': f'unreadable: {type(e).__name__}'}


def round_key(trial_tar) -> tuple:
    """(round, run dir, task, attempt, retry) of <runs>/r<k>[suffix]/attempts/<task>__a<i>__r<j>/trial.tar.gz: rounds
    in numeric order (r2b, a split of r2, is round 2, after r2's own run), then retries in numeric order (r10 > r9)."""
    p = Path(trial_tar)
    run = p.parent.parent.parent.name
    m = re.match(r'r(\d+)', run)
    if not m:
        raise ValueError(f'not a round run: {p}')
    a = re.match(r'(.*)__a(\d+)(?:__r(\d+))?$', p.parent.name)
    task, attempt, retry = (a.group(1), int(a.group(2)), int(a.group(3) or 0)) if a else (p.parent.name, 0, 0)
    return int(m.group(1)), run, task, attempt, retry


def natural(r: dict) -> bool:
    return 'drop' not in r and bool((r.get('meta') or {}).get('natural'))


def keep_best(records) -> dict:
    """{id: record} from records IN ROUND ORDER (round_key; retries in key order within a round): the first natural
    finish; else the latest real record; a drop only when no round produced a real one. Parallel rounds make a drop in
    a later round likely for every task (each round runs every unfinished task): it must never win."""
    best = {}
    for r in records:
        prev = best.get(r['id'])
        if prev is None:
            best[r['id']] = r
        elif natural(prev):
            continue                      # the first natural finish in round order is final
        elif 'drop' in r and 'drop' not in prev:
            continue                      # a later round's failure never replaces a real record
        else:
            best[r['id']] = r             # a later real record replaces an unfinished one (or a drop)
    return best


def dedupe_lowest_id(records, group=lambda r: r.get('group') or r['id']):
    """(kept, dropped): one record per task (group) -- the lowest id among its real records (a drop only if the task
    has nothing else) -- in input order. A split pi_eval run's fence can overshoot (a boundary task claimed by both
    runs); SFT assembly kept the lowest id (run A's), so every other consumer must keep the same one (a resample built
    from the other copy would replace a record that is not kept)."""
    records = list(records)
    rank = lambda r: ('drop' in r, r['id'])   # real records first, then the lowest id
    low = {}
    for r in records:
        g = group(r)
        if g not in low or rank(r) < low[g]:
            low[g] = rank(r)
    seen, out = set(), []
    for r in records:
        g = group(r)
        if rank(r) == low[g] and g not in seen:   # identical ids twice (one record read twice) collapse too
            seen.add(g)
            out.append(r)
    return out, len(records) - len(out)


def build_rounds(tars, base_of, src_of=None, map_fn=map) -> dict:
    """{id: record} over every round's attempts: attempt_record per trial.tar.gz in round order (round_key), then
    keep_best. base_of(tar) -> the record's base fields (id, group, ...); src_of(tar) -> its finishing input or None.
    A failure in either is that attempt's drop (id = its path), never a crashed build (M5: a missing index entry).
    map_fn: e.g. a process pool's map (with picklable base_of/src_of)."""
    return keep_best(map_fn(_round_one, [(t, base_of, src_of) for t in sorted(map(str, tars), key=_round_order)]))


def _round_order(tar):
    try:
        return 0, round_key(tar)
    except ValueError:   # not under a round run: after every round, by path (its record is built all the same)
        return 1, (tar,)


def _round_one(job):
    tar, base_of, src_of = job
    try:
        base = base_of(tar)
        src = src_of(tar) if src_of else None
    except Exception as e:   # noqa: BLE001
        return {'id': str(tar), 'group': str(tar), 'drop': f'unreadable: {type(e).__name__}'}
    return attempt_record(tar, base, src)

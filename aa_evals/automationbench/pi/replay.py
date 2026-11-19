"""Replay stored upstream rollouts through the Pi port's world service and grader (#348 equivalence check).

  python -m aa_evals.automationbench.pi.replay RUN_DIR [--limit N] [--tasks a,b] [--out FILE]

For each rollout in an upstream-harness run (`aa_evals.automationbench.run`: items.jsonl + transcripts.jsonl), it
starts `image/world_server.World` from the bundle's world spec (`tasks.world_spec`), sends every stored tool call
in order exactly as the `ab` CLI would (the JSON-decoded arguments; a call whose arguments did not decode is
skipped, as verifiers returned the parse error without touching the world; so is a call with no stored tool
message: the rollout ended, at the turn cap, before upstream executed it), and compares:
- each tool result with the stored tool message (`result_match`: identical strings once the random ids and
  timestamps the simulated apps generate are aligned, `align`; each stored id is mapped to the replay's, and
  later arguments are rewritten with that map so they refer to the same objects);
- `pi.run.grade` on the world written to `/state/world.json` with the stored grading (`final_state_score`,
  objectives, guardrails, upstream partial credit and strict pass).
It shows that the world service plus grading give the upstream path's result for the same tool calls. It does not
run a model. CPU only, a few seconds per rollout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from aa_evals.common import jsonl

HERE = Path(__file__).resolve().parent
KEYS = ('final_state_score', 'n_objectives', 'n_objectives_met', 'n_guardrails', 'n_guardrails_broken',
        'task_completed', 'upstream_partial_credit', 'upstream_strict_pass')


# Values the simulated apps draw at random or from the clock (uuid hex, Slack channel ids, Jira keys, Twilio sids,
# Buffer post ids, epoch seconds, wall-clock ISO dates/datetimes, ...): a token that differs between the replay and
# the stored result is accepted as one when both are ISO dates/datetimes, both are numbers of 6+ digits, or both have
# the same length and it has a
# digit (4+ chars) or is long (10+ chars).
DATETIME = r'\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?'
TOKEN = re.compile(DATETIME + r'(?![0-9A-Za-z])|[A-Za-z0-9]+')
_DT = re.compile(DATETIME + '$')


def _volatile(a: str, b: str) -> bool:
    if _DT.match(a) and _DT.match(b):
        return True
    if a.isdigit() and b.isdigit() and min(len(a), len(b)) >= 6:      # random numeric ids (Zoom meeting ids)
        return True
    return len(a) == len(b) and (len(a) >= 10 or (len(a) >= 4 and any(c.isdigit() for c in a + b)))


def align(got: str, stored: str, idmap: dict) -> bool:
    """True when `got` equals `stored` up to volatile tokens (same separators, each differing token pair
    volatile); adds the stored -> got token pairs to `idmap`."""
    if TOKEN.split(got) != TOKEN.split(stored):
        return False
    pairs = [(a, b) for a, b in zip(TOKEN.findall(stored), TOKEN.findall(got)) if a != b]
    if not all(_volatile(a, b) for a, b in pairs):
        return False
    idmap.update(pairs)
    return True


def remap(raw: str, idmap: dict) -> str:
    if not idmap:
        return raw
    return TOKEN.sub(lambda m: idmap.get(m.group(0), m.group(0)), raw)


def world_module():
    spec = importlib.util.spec_from_file_location('ab_world_server', HERE / 'image' / 'world_server.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tool_calls(messages: list[dict]) -> list[tuple[dict, str | None]]:
    """(tool call, the stored tool message content answering it) in order."""
    answers = {m.get('tool_call_id'): m.get('content') for m in messages if m.get('role') == 'tool'}
    out = []
    for m in messages:
        if m.get('role') == 'assistant':
            for tc in m.get('tool_calls') or []:
                out.append((tc, answers.get(tc.get('id'))))
    return out


def call_parts(tc: dict) -> tuple[str, str]:
    f = tc.get('function') or tc
    return f.get('name'), f.get('arguments')


def replay_one(W, row: dict, item: dict, transcript: dict, tmp: Path) -> dict:
    from aa_evals.automationbench.pi import tasks as PT
    from aa_evals.automationbench.pi.run import grade
    d = tmp / item['item_id']
    d.mkdir(parents=True)
    (d / 'task.json').write_text(json.dumps(PT.world_spec(row)))
    world = W.World(d / 'task.json', d / 'state')
    n = n_match = n_skipped = n_unanswered = 0
    mismatches = []
    idmap: dict = {}
    for tc, stored in tool_calls(transcript['messages']):
        name, raw = call_parts(tc)
        if stored is None:
            n_unanswered += 1
            continue
        if isinstance(raw, str):
            raw = remap(raw, idmap)
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            n_skipped += 1
            continue
        if not isinstance(args, dict):
            n_skipped += 1
            continue
        if name not in W.TOOLS:            # upstream: KeyError from tool_map, world untouched
            n_skipped += 1
            continue
        out, _ok = world.call(name, args)
        n += 1
        if out == stored or align(out, stored, idmap):
            n_match += 1
        elif len(mismatches) < 3:
            mismatches.append({'tool': name, 'got': out[:300], 'stored': (stored or '')[:300]})
    g = grade(row, (d / 'state' / 'world.json').read_text(), json.loads((d / 'state' / 'hidden.json').read_text()))
    diff = {k: (item.get(k), g.get(k)) for k in KEYS if item.get(k) != g.get(k)}
    return {'item_id': item['item_id'], 'repeat': item.get('repeat'), 'status': item.get('status'),
            'n_calls': n, 'n_result_match': n_match, 'n_skipped': n_skipped, 'n_unanswered': n_unanswered, 'grade_diff': diff,
            'stored_score': item.get('final_state_score'), 'replay_score': g['final_state_score'],
            'grading_exceptions': g.get('grading_exceptions'), 'mismatches': mismatches}


def main(argv=None):
    from aa_evals.automationbench.pi.tasks import ensure_venv
    ensure_venv('aa_evals.automationbench.pi.replay')
    from aa_evals.automationbench import run as AB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tasks', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    items = jsonl.read(run / 'items.jsonl', required=True)
    last = {}
    for r in items:
        last[(r['item_id'], r['repeat'])] = r
    items = [r for r in last.values() if r.get('status') in ('ok', 'model_error') and r.get('transcript')]
    if a.tasks:
        want = set(a.tasks.split(','))
        items = [r for r in items if r['item_id'] in want]
    items.sort(key=lambda r: (r['item_id'], r['repeat']))
    if a.limit:
        items = items[:a.limit]
    rows = {r['_task_name']: r for r in AB.load_tasks(sorted({i['item_id'].split('.')[0] for i in items}))}
    W = world_module()
    out = open(a.out, 'w') if a.out else None
    tot = {'n': 0, 'calls': 0, 'match': 0, 'grade_equal': 0}
    with tempfile.TemporaryDirectory() as tmp:
        for it in items:
            with open(it['transcript']) as f:
                f.seek(it['transcript_offset'])
                tr = json.loads(f.readline())
            res = replay_one(W, rows[it['item_id']], it, tr, Path(tmp))
            tot['n'] += 1
            tot['calls'] += res['n_calls']
            tot['match'] += res['n_result_match']
            tot['grade_equal'] += not res['grade_diff']
            line = json.dumps(res, default=str)
            if out:
                out.write(line + '\n')
            if res['grade_diff'] or res['n_result_match'] != res['n_calls']:
                print(line[:1500], flush=True)
    print(json.dumps(tot))
    return tot


if __name__ == '__main__':
    main()

"""The real-asset A/B against student 1 (issue #20): which of student 1's tasks are swapped, what their traces weighed,
and the exclusion lists for the treatment's SFT data.

  python -m task_synthesis.assets.ab swapset --bundle <swap bundle> --name <ab>
  python -m task_synthesis.assets.ab select --name <ab> --source <manifest dir>=<campaign root> ... [--seed 0]

`select` (2026-10-31, traces streamed while the rounds ran, every real-asset task traced): the token match is a seeded
random subset. Every student-1 task with a real-asset version leaves the data (that is the treatment: real asset
whenever one fits; a version the trace judge calls a task defect is dropped, as student 1 dropped its own). The
budget is the student-1 train tokens of those tasks; real-asset tasks with a usable, defect-free trace enter in
seeded random order until their tokens (Super saved x the student-1 Qwen/saved ratio) reach it, 1% over, so the SFT
prep's --max_tokens cap at student 1's train total trims the rest at random.

A student-1 task is swapped when its spec has a real-asset task in the swap bundle (valid, judge-passed, not
decon-flagged, not a near-duplicate). Its student-1 traces leave the data; the real-asset task's traces come in.
Everything else (the none-bucket tasks, unbound or failed specs) keeps student 1's traces.

Writes <ROOT>/ab/<name>/:
  swap.json                 new task -> student-1 task
  exclude-tasks.txt         student 1's exclusions + every swapped student-1 task
  exclude-attempts.txt      student 1's attempt exclusions (S2 truncations)
  budget.json               what the swapped tasks weighed in student 1: SFT train tokens (Qwen) and the Super saved
                            tokens behind them (the campaign's tokens_budget unit); the new traces' token target
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ..common import ROOT

S1_DATA = Path('/data/aai/tbenchtrain/sft_data_q35/tasksynth-ds-330m')
S1_EXCLUDE_TASKS = ROOT / 'sft' / 'student1-exclude-tasks.txt'
S1_EXCLUDE_ATTEMPTS = ROOT / 'sft' / 'student1-exclude-attempts.txt'
S1_CAMPAIGNS = [Path('/data/aai/tbench/synth') / c / 'train' for c in (
    '20261030-nemotron35-super-tasksynth-r5', '20261030-nemotron35-super-tasksynth-s1',
    '20261030-nemotron35-super-tasksynth-s2', '20261031-nemotron35-super-tasksynth-s2b')]


def student1_tokens():
    """task -> (train tokens, val tokens, sequences) in student 1's SFT data."""
    out = defaultdict(lambda: [0, 0, 0])
    with open(S1_DATA / 'sequences.csv') as f:
        for r in csv.DictReader(f):
            t = out[r['group']]
            t[0 if r['split'] == 'train' else 1] += int(r['n_tokens'])
            t[2] += 1
    return out


def student1_saved():
    """task -> Super saved tokens of its attempts that student 1's SFT data kept (usable, not excluded)."""
    excl = set(S1_EXCLUDE_ATTEMPTS.read_text().split())
    out = defaultdict(int)
    for root in S1_CAMPAIGNS:
        for r in json.loads((root / 'manifest.json').read_text())['records']:
            if r.get('usable') and r['id'] not in excl:
                out[r['task']] += r['tokens']
    return out


def origins(manifest):
    """real-asset task -> student-1 task it replaces (its round spec's orig_spec_id)."""
    specs, out = {}, {}
    for task, m in manifest.items():
        if m['round'] not in specs:
            specs[m['round']] = {json.loads(l)['spec_id']: json.loads(l)
                                 for l in (ROOT / 'rounds' / m['round'] / 'specs.jsonl').read_text().splitlines()}
        spec = specs[m['round']].get(m['spec_id'])
        if not spec or not spec.get('orig_spec_id'):
            raise SystemExit(f'{task}: no orig_spec_id in its round spec')
        out[task] = f'tasksynth-{spec["orig_spec_id"]}'
    return out


def select(a):
    import random
    toks = student1_tokens()
    s1_excl = set(S1_EXCLUDE_TASKS.read_text().split())
    orig, traces, defects, missing_verdict = {}, defaultdict(int), set(), 0
    for src in a.source:
        mdir, root = (Path(x) for x in src.split('='))
        manifest = json.loads((mdir / 'manifest.json').read_text())
        orig.update(origins(manifest))
        for r in json.loads((root / 'train' / 'manifest.json').read_text())['records']:
            if r.get('usable') and r['task'] in manifest:
                traces[r['task']] += r['tokens']
        verdicts = {}
        rec_path = ROOT / 'traces' / root.name / 'records.jsonl'
        for l in rec_path.read_text().splitlines():
            v = json.loads(l)
            if v.get('verdict'):
                verdicts.setdefault(v['task'], []).append(v['verdict'])
        for t in manifest:
            if t in traces and t not in verdicts:
                missing_verdict += 1
            if any(v.get('task_defect') for v in verdicts.get(t, [])):
                defects.add(t)
    if missing_verdict and not a.allow_unjudged:
        raise SystemExit(f'{missing_verdict} traced tasks have no trace-judge verdict (run traces first)')
    swapped = sorted(set(orig.values()))
    budget = sum(toks[t][0] for t in swapped)
    ratio = 321688447 / 325959657   # student 1: Qwen train tokens per Super saved token of its train tasks
    cands = sorted(t for t in traces if t not in defects)
    random.Random(a.seed).shuffle(cands)
    chosen, total = [], 0
    for t in cands:
        if total >= budget * 1.01:
            break
        chosen.append(t)
        total += traces[t] * ratio
    left_out = set(orig) - set(chosen)
    out = ROOT / 'ab' / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'swap.json').write_text(json.dumps(orig, indent=1))
    (out / 'selected.txt').write_text('\n'.join(sorted(chosen)) + '\n')
    (out / 'exclude-tasks.txt').write_text('\n'.join(sorted(s1_excl | set(swapped) | left_out)) + '\n')
    (out / 'exclude-attempts.txt').write_text(S1_EXCLUDE_ATTEMPTS.read_text())
    info = {'sources': a.source, 'seed': a.seed, 'real_asset_tasks': len(orig), 'with_usable_trace': len(traces),
            'trace_judge_defects': len(defects), 'candidates': len(cands), 'selected': len(chosen),
            'student1_tasks_swapped_out': len(swapped),
            'swapped_out_in_student1_train': sum(1 for t in swapped if toks[t][0]),
            'student1_train_tokens_of_swapped': budget,
            'selected_qwen_tokens_est': round(total),
            'candidate_qwen_tokens_est': round(sum(traces[t] for t in cands) * ratio),
            'student1_train_tokens_total': sum(v[0] for v in toks.values()),
            'sft_max_tokens': 321688447}
    (out / 'selection.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def swapset(a):
    bundle = ROOT / 'campaigns' / a.bundle
    manifest = json.loads((bundle / 'manifest.json').read_text())
    orig = {}
    for task, m in manifest.items():
        spec = None
        for line in (ROOT / 'rounds' / m['round'] / 'specs.jsonl').read_text().splitlines():
            s = json.loads(line)
            if s['spec_id'] == m['spec_id']:
                spec = s
                break
        if not spec or not spec.get('orig_spec_id'):
            raise SystemExit(f'{task}: no orig_spec_id in its round spec')
        orig[task] = f'tasksynth-{spec["orig_spec_id"]}'
    toks = student1_tokens()
    s1_excl = set(S1_EXCLUDE_TASKS.read_text().split())
    swapped = sorted(set(orig.values()))
    train = sum(toks[t][0] for t in swapped)
    saved = student1_saved()
    in_train = [t for t in swapped if toks[t][0]]
    out = ROOT / 'ab' / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'swap.json').write_text(json.dumps(orig, indent=1))
    (out / 'exclude-tasks.txt').write_text('\n'.join(sorted(s1_excl | set(swapped))) + '\n')
    (out / 'exclude-attempts.txt').write_text(S1_EXCLUDE_ATTEMPTS.read_text())
    budget = {'bundle': a.bundle, 'swapped_tasks': len(swapped),
              'swapped_in_student1_train': sum(1 for t in swapped if toks[t][0]),
              'swapped_already_excluded_in_student1': sum(1 for t in swapped if t in s1_excl),
              'student1_train_tokens_of_swapped': train,
              'student1_val_tokens_of_swapped': sum(toks[t][1] for t in swapped),
              'student1_train_tokens_total': sum(v[0] for v in toks.values()),
              # Super saved tokens behind the swapped tasks' student-1 train sequences (tokens_budget unit)
              'student1_saved_tokens_of_swapped_in_train': sum(saved[t] for t in in_train),
              'qwen_train_per_saved_token': round(train / max(1, sum(saved[t] for t in in_train)), 4)}
    (out / 'budget.json').write_text(json.dumps(budget, indent=1))
    print(json.dumps(budget, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('swapset')
    p.add_argument('--bundle', required=True)
    p.add_argument('--name', required=True)
    p = sub.add_parser('select')
    p.add_argument('--name', required=True)
    p.add_argument('--source', action='append', required=True,
                   help='<dir with manifest.json (bundle or stream)>=<synth campaign root>; repeatable')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow-unjudged', action='store_true')
    a = ap.parse_args()
    swapset(a) if a.op == 'swapset' else select(a)


if __name__ == '__main__':
    main()

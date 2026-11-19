"""Pilot / production metrics for TB-Sci-lite synthesis (#353; the M5 checklist).

  python -m m5.tbsci_lite.report --rounds p1 [--traces pilot] [--production-n 6000] [--out FILE.json]

Generation: funnel, generator/judge tokens, instruction words, files, data size, judge scores.
Taxonomy: accepted tasks per domain / field / archetype / modality / source against the vocabulary's targets.
Repeats: TF-IDF cosine >= 0.6 near-duplicate pairs among accepted instructions, and the share of tasks in one,
  projected linearly to the production count (repeats grow ~ N, #25 lesson 2).
Copying: word 8-gram containment of each instruction in the material the generator saw (system prompt + its user
  prompt, which holds the spec and, for real data, the dataset rendering).
Traces (export of traces.py): endings (submitted / context_limit / env_budget / stopped), tokens per trace
  (largest request context = the full sequence), requests, wall time.
"""
import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from .common import ROOT

CLOSING_RE = re.compile(r'\n*You have \d+ seconds to complete this task\..*$', re.S)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {'n': len(xs), 'mean': round(statistics.mean(xs), 1), 'p10': q(xs, 0.1), 'median': q(xs, 0.5),
            'p90': q(xs, 0.9), 'max': max(xs)}


def load_round(r):
    d = ROOT / 'rounds' / r
    specs = [json.loads(l) for l in (d / 'specs.jsonl').read_text().splitlines()]
    recs = {}
    for s in specs:
        p = d / 'work' / s['spec_id'] / 'record.json'
        if p.exists():
            recs[s['spec_id']] = json.loads(p.read_text())
    return d, specs, recs


def main(argv=None):
    import sys
    from task_synthesis.pipeline import textsim
    from . import round_sci, specs as S
    round_sci.patch()
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', nargs='+', required=True)
    ap.add_argument('--traces')
    ap.add_argument('--production-n', type=int, default=6000)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    out = {'rounds': a.rounds}
    accepted, all_specs, all_recs, dirs = [], [], {}, {}
    for r in a.rounds:
        d, specs, recs = load_round(r)
        all_specs += specs
        all_recs.update(recs)
        for s in specs:
            dirs[s['spec_id']] = d
    status = collections.Counter(rec['status'] for rec in all_recs.values())
    jpass = [sid for sid, rec in all_recs.items() if rec['status'] == 'valid' and (rec.get('judge') or {}).get('pass')]
    out['funnel'] = {'specs': len(all_specs), 'recorded': len(all_recs), 'status': dict(status),
                     'judge_pass': len(jpass),
                     'judge_pass_of_valid': round(len(jpass) / max(1, status.get('valid', 0)), 3),
                     'accepted_of_recorded': round(len(jpass) / max(1, len(all_recs)), 3)}
    gen_tok = [sum((x.get('usage') or {}).get('completion') or 0 for x in rec.get('attempts') or []) for rec in all_recs.values()]
    gen_prompt = [sum((x.get('usage') or {}).get('prompt') or 0 for x in rec.get('attempts') or []) for rec in all_recs.values()]
    judge_tok = [((rec.get('judge') or {}).get('usage') or {}).get('completion') or 0 for rec in all_recs.values()]
    out['generator_tokens'] = {'completion_per_spec': summary(gen_tok), 'prompt_per_spec': summary(gen_prompt),
                               'judge_completion_per_valid': summary([x for x in judge_tok if x]),
                               'total_completion': sum(gen_tok) + sum(judge_tok),
                               'repairs': dict(collections.Counter(rec.get('repairs', 0) for rec in all_recs.values()))}
    fails = collections.Counter()
    for rec in all_recs.values():
        if rec['status'] != 'valid':
            for e in (rec['attempts'][-1].get('errors') or [])[:1] if rec.get('attempts') else []:
                fails[re.sub(r'[0-9/][^ ]*', '#', e)[:80]] += 1
        elif not (rec.get('judge') or {}).get('pass'):
            j = rec.get('judge') or {}
            low = [k for k, v in (j.get('scores') or {}).items() if v is not None and v < (4 if k in (
                'clarity', 'consistency', 'solvability', 'asset_use') else 3)]
            fl = [k for k, v in (j.get('flags') or {}).items() if v and k != 'spec_mismatch']
            fails['judge: ' + ','.join(low + fl)] += 1
    out['failures'] = fails.most_common(15)
    # instructions, files, data
    words, files, data, est = [], [], [], []
    texts = {}
    for sid in jpass:
        td = dirs[sid] / 'tasks' / sid
        try:
            ins = (td / 'instruction.md').read_text()
        except OSError:
            continue
        body = CLOSING_RE.sub('', ins)
        texts[sid] = body
        words.append(len(body.split()))
        fe = td / 'environment' / 'files'
        files.append(sum(1 for p in fe.rglob('*') if p.is_file()) if fe.exists() else 0)
        rec = all_recs[sid]
        chk = dirs[sid] / 'work' / sid / f'attempt-{len(rec["attempts"]) - 1}.check.json'
        try:
            c = json.loads(chk.read_text())
            data.append(round(c.get('data_bytes', 0) / 2**20, 2) if 'data_bytes' in c else None)
            n_inputs = sum(1 for line in (c.get('snapshot') or {}).get('tree', []))
            files[-1] = max(files[-1], n_inputs)
        except (OSError, ValueError):
            data.append(None)
        est.append((rec.get('judge') or {}).get('estimated_minutes'))
    out['instructions'] = {'words': summary(words), 'env_files_in_snapshot': summary(files),
                           'data_mb': summary(data), 'judge_estimated_minutes': summary(est),
                           'lite_reference': {'words_median': 576, 'env_files_median': 10}}
    scores = collections.defaultdict(list)
    for sid in jpass:
        for k, v in ((all_recs[sid].get('judge') or {}).get('scores') or {}).items():
            scores[k].append(v)
    out['judge_scores_mean'] = {k: round(statistics.mean(v), 2) for k, v in scores.items()}
    # taxonomy spread
    v = S.vocab()
    acc_specs = [all_recs[sid]['spec'] for sid in jpass]
    target_dom = {d: x['weight'] for d, x in v['domains'].items()}
    target_field = {f: x['weight'] * w for d, x in v['domains'].items() for f, w in x['fields'].items()}

    def spread(key, target):
        c = collections.Counter(s.get(key) for s in acc_specs)
        n = max(1, len(acc_specs))
        return {k: {'target': round(t, 3), 'got': round(c.get(k, 0) / n, 3), 'n': c.get(k, 0)} for k, t in
                sorted(target.items(), key=lambda kv: -kv[1])}
    arch_t = collections.Counter()
    mod_t = collections.Counter()
    for d, x in v['domains'].items():
        for k, p in v['archetype_by_domain'][d].items():
            arch_t[k] += x['weight'] * p
    planned = collections.Counter(s['modality'] for s in all_specs)
    out['taxonomy'] = {'domain': spread('sci_domain', target_dom), 'field': spread('domain', target_field),
                       'archetype': spread('agent_action', dict(arch_t)),
                       'modality_planned_vs_accepted': {k: {'planned': planned[k], 'accepted': sum(
                           1 for s in acc_specs if s['modality'] == k)} for k in planned},
                       'source': dict(collections.Counter(s.get('source') for s in acc_specs)),
                       'open_subtopic': sum(1 for s in acc_specs if not s.get('subtopic')),
                       'distinct_subtopics': len({(s['domain'], (s.get('subtopic') or {}).get('name')) for s in acc_specs})}
    # repeats
    ids = list(texts)
    pairs, nearest = textsim.near_duplicates(texts, 0.6) if len(ids) > 1 else ([], {})
    in_pair = {x for p in pairs for x in p[:2]}
    rate = len(in_pair) / max(1, len(ids))
    out['repeats'] = {'pairs_cos_ge_0.6': len(pairs), 'tasks_in_a_pair': len(in_pair), 'share': round(rate, 4),
                      'projected_share_at_production': round(min(1.0, rate * a.production_n / max(1, len(ids))), 4),
                      'production_n': a.production_n, 'nearest_cosine': summary(list(nearest.values())),
                      'examples': [list(p) for p in sorted(pairs, key=lambda p: -p[2])[:5]]}
    # 8-gram copying of the generator's material
    sysp = round_sci.system_prompt(180)
    cont = []
    for sid in ids:
        spec = all_recs[sid]['spec']
        asset = None
        if spec.get('asset'):
            try:
                asset = (Path(spec['asset']['dir']) / 'gen.txt').read_text()
            except OSError:
                asset = ''
        user = S.render(spec, asset)
        cont.append(textsim.containment(texts[sid], sysp + '\n' + user, 8))
    out['copying_8gram'] = {'containment_in_prompt_material': summary([round(c, 3) for c in cont]),
                            'share_above_0.1': round(sum(1 for c in cont if c > 0.1) / max(1, len(cont)), 3)}
    # traces
    if a.traces:
        recs = json.loads((ROOT / 'traces' / a.traces / 'export' / 'records.json').read_text())
        metas = [r['meta'] for r in recs.values()]
        ends = collections.Counter(m['end'] for m in metas)
        n = max(1, len(metas))
        toks = [m.get('max_context_tokens') for m in metas]
        out['traces'] = {'n': len(metas), 'ends': dict(ends),
                         'submitted_share': round(ends.get('submitted', 0) / n, 3),
                         'context_limit_share': round(ends.get('context_limit', 0) / n, 3),
                         'tokens_per_trace': summary(toks), 'tokens_total': sum(t or 0 for t in toks),
                         'tokens_submitted': summary([m.get('max_context_tokens') for m in metas if m['end'] == 'submitted']),
                         'model_requests': summary([m.get('model_requests') for m in metas]),
                         'wall_min': summary([round((m.get('wall_sec') or 0) / 60, 1) for m in metas]),
                         'model_hours_sum': round(sum((m.get('model_sec') or 0) for m in metas) / 3600, 1)}
        by_arch = collections.defaultdict(list)
        spec_of = {s['spec_id']: s for s in all_specs}
        for r in recs.values():
            s = spec_of.get(r['task'])
            if s:
                by_arch[s['agent_action']].append(r['meta']['end'] == 'submitted')
                by_arch['source:' + s['source']].append(r['meta']['end'] == 'submitted')
        out['traces']['submitted_by_slice'] = {k: [round(sum(v) / len(v), 2), len(v)] for k, v in sorted(by_arch.items())}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()

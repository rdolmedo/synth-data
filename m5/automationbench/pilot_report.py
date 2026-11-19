"""Pilot metrics (#359 checklist) from the accepted tasks, the collected traces and the server snapshots.

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.pilot_report --accepted F --rejected F \
      --traces ROOT --snaps SNAPS.jsonl --fit fit.json --n-prod 14000 --out REPORT.json

- trace endings: submit share (the agent finished by itself), context-limit and time-limit shares, infra;
- tokens per trace: the largest request context of the session (prompt + completion as the server counted them
  with Super's tokenizer: the full sequence for a linear session), and output tokens;
- GPU-hours by token share between the first and last snapshot of each engine: GPUs x hours x our generation
  tokens / the engine's generation tokens (DeepSeek: completion tokens of our generator calls);
- repeat meter: share of tasks whose TF-IDF nearest neighbour (request + data skeleton) is >= 0.8, projected
  linearly to --n-prod (pairs grow quadratically, so the per-task rate grows ~ N);
- copying: request 8-gram containment against the 200 simple tasks (the generator's format examples) and against the
  task's own leaf text, and reuse of the simple tasks' names/IDs in the states;
- taxonomy spread: realized domain / workflow / trap / app shares of accepted tasks vs. the sampler's targets, and
  acceptance per cell.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics
from pathlib import Path


def grams(t, n=8):
    w = re.findall(r'\w+', t.lower())
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def gpu_hours(snaps, names, gpus, our_gen_tokens):
    """names: the engines the work ran on (each `gpus` GPUs); hours = the snapshot window per engine."""
    xs = sorted((s for s in snaps if s['name'] in names), key=lambda s: s['t'])
    if len(xs) < 2:
        return None
    by_url = collections.defaultdict(list)
    for s in xs:
        by_url[s['url']].append(s)
    by_url = {u: v for u, v in by_url.items() if len(v) >= 2}
    tot_gen = sum(v[-1]['generation_tokens'] - v[0]['generation_tokens'] for v in by_url.values())
    hours = sum(v[-1]['t'] - v[0]['t'] for v in by_url.values()) / 3600
    if tot_gen <= 0:
        return None
    share = min(1.0, our_gen_tokens / tot_gen)
    hit = None
    pq = sum(v[-1]['prefix_queries'] - v[0]['prefix_queries'] for v in by_url.values())
    if pq > 0:
        hit = sum(v[-1]['prefix_hits'] - v[0]['prefix_hits'] for v in by_url.values()) / pq
    return {'engines': sorted(names), 'engine_hours': round(hours, 3), 'gpus_per_engine': gpus, 'token_share': round(share, 4),
            'gpu_hours': round(gpus * hours * share, 3), 'engine_prefix_hit': hit}


def main():
    from m5.automationbench.decon import skeleton, strings_of
    from m5.automationbench.spirit import repeat_meter
    from m5.automationbench.taxonomy import strip_suffixes
    from m5.automationbench.upstream import load
    ap = argparse.ArgumentParser()
    ap.add_argument('--accepted', required=True)
    ap.add_argument('--rejected', default=None)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--snaps', required=True)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--n-prod', type=int, default=14000)
    ap.add_argument('--ds-gpus', type=int, default=4)
    ap.add_argument('--runs-root', default='/data/aai/m5/automationbench/pi_runs/m5_automationbench')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    acc = [json.loads(x) for x in open(a.accepted) if x.strip()]
    rej = [json.loads(x) for x in open(a.rejected) if x.strip()] if a.rejected and Path(a.rejected).exists() else []
    R = {'n_specs': len(acc) + len(rej), 'n_accepted': len(acc), 'n_rejected': len(rej),
         'accept_rate': round(len(acc) / max(1, len(acc) + len(rej)), 3),
         'rounds': dict(collections.Counter(r['rounds'] for r in acc)),
         'reject_reasons': dict(collections.Counter((r.get('errors') or ['?'])[0][:60] for r in rej).most_common(10))}
    gen_tok = sum(u['completion'] for r in acc + rej for u in r.get('usage') or [])
    R['generator'] = {'completion_tokens': gen_tok,
                      'completion_per_accepted': round(gen_tok / max(1, len(acc))),
                      'prompt_tokens': sum(u['prompt'] for r in acc + rej for u in r.get('usage') or []),
                      'sec_median': statistics.median([r['sec'] for r in acc]) if acc else None}
    # traces
    idx = json.loads((Path(a.traces) / 'attempts' / 'index.json').read_text())
    lines = []
    for n in sorted(idx['shards'], key=int):
        with gzip.open(Path(a.traces) / 'attempts' / idx['shards'][n]['lines'], 'rt') as f:
            lines += [json.loads(x) for x in f]
    ends = collections.Counter(x['m5']['end'] for x in lines)
    ok = [x for x in lines if x['record']['usable']]
    ctx = [int(x['m5']['stats'].get('max_context_tokens') or 0) for x in ok]
    outs = []
    for x in ok:
        o = 0
        for line in x['sessions'][0]['text'].splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = e.get('message') or {}
            if m.get('role') == 'assistant':
                o += int((m.get('usage') or {}).get('output') or 0)
        outs.append(o)
    n = max(1, len(lines))
    R['traces'] = {'n': len(lines), 'usable': len(ok), 'ends': dict(ends),
                   'submit_share': round(ends.get('finish', 0) / n, 3),
                   'context_limit_share': round(ends.get('context', 0) / n, 3),
                   'time_limit_share': round(ends.get('time_limit', 0) / n, 3),
                   'tokens_per_trace': {'mean': round(statistics.mean(ctx)) if ctx else None,
                                        'median': statistics.median(ctx) if ctx else None,
                                        'p90': sorted(ctx)[int(0.9 * len(ctx))] if ctx else None,
                                        'sum': sum(ctx)},
                   'output_tokens_per_trace': {'mean': round(statistics.mean(outs)) if outs else None},
                   'turns_median': statistics.median([x['m5']['stats'].get('model_requests') or 0 for x in ok]) if ok else None,
                   'tool_calls_median': statistics.median([x['m5']['n_tool_calls'] for x in ok]) if ok else None,
                   'tool_error_share': round(sum(x['m5']['n_tool_errors'] for x in ok) / max(1, sum(x['m5']['n_tool_calls'] for x in ok)), 3),
                   'wall_sec_median': statistics.median([x['m5']['stats'].get('wall_sec') or 0 for x in ok]) if ok else None}
    if ctx:
        R['traces']['projected_tasks_for_330M'] = round(330e6 / statistics.mean(ctx))
    snaps = [json.loads(x) for x in open(a.snaps)]
    engines = sorted({x['m5']['run'] for x in lines})
    runs_engine = {}
    for rn in engines:
        try:
            cfg = json.loads((Path(a.runs_root) / rn / 'config.json').read_text())
            runs_engine[rn] = cfg.get('model', cfg.get('model_spec', '')).split(':')[-1] if isinstance(cfg.get('model'), str) else None
        except (OSError, ValueError):
            runs_engine[rn] = None
    names = sorted({e for e in runs_engine.values() if e}) or ['m5-super-1']
    R['gpu_hours'] = {'super': gpu_hours(snaps, names, 1, sum(outs)),
                      'deepseek': gpu_hours(snaps, ['m5-ds'], a.ds_gpus, gen_tok)}
    # repeat meter
    items = [{'trigger': r['task']['prompt'][1]['content'], 'own_state': r['own_state']} for r in acc]
    rm = repeat_meter(items)
    if rm:
        rm['projected_share_at_n_prod'] = round(rm['share_nn_over'] * a.n_prod / max(1, len(items)), 4)
        rm['n_prod'] = a.n_prod
    R['repeat_meter'] = rm
    # copying vs examples and leaves
    simple = load('simple', noise=False)
    ex_grams = set().union(*(grams(r['trigger']) for r in simple))
    ex_strings = set().union(*(strings_of(r['info']['initial_state']) for r in simple))
    # specific strings only (as decon.py): a digit, an email or 3+ words; generic titles and field values excluded
    ex_strings = {s for s in ex_strings if (re.search(r'\d', s) and len(s) >= 6) or '@' in s or len(s.split()) >= 3}
    common = {s for s in ex_strings if re.fullmatch(r'[\w.+-]*@?(company|example)\.example\.com|\d+', s)}
    cont_ex, cont_leaf, reuse = [], [], []
    for r in acc:
        body = strip_suffixes(r['task']['prompt'][1]['content'])[0]
        g = grams(body)
        if g:
            cont_ex.append(len(g & ex_grams) / len(g))
            lf = r['spec'].get('leaf')
            if lf:
                cont_leaf.append(len(g & grams(lf['scenario'] + ' ' + lf.get('rule_hooks', ''))) / len(g))
        reuse.append(len((strings_of(r['own_state']) & ex_strings) - common))
    R['copying'] = {'trigger_8gram_containment_examples': {'mean': round(statistics.mean(cont_ex), 4) if cont_ex else None,
                                                         'share_any': round(sum(c > 0 for c in cont_ex) / max(1, len(cont_ex)), 3)},
                    'trigger_8gram_containment_leaf': {'mean': round(statistics.mean(cont_leaf), 4) if cont_leaf else None,
                                                     'share_any': round(sum(c > 0 for c in cont_leaf) / max(1, len(cont_leaf)), 3)},
                    'example_strings_reused': {'share_any': round(sum(x > 0 for x in reuse) / max(1, len(reuse)), 3),
                                               'mean': round(statistics.mean(reuse), 3) if reuse else None}}
    # taxonomy spread
    F = json.loads(Path(a.fit).read_text())
    spec_all = [r['spec'] for r in acc + rej]

    def shares(xs):
        c = collections.Counter(xs)
        t = sum(c.values()) or 1
        return {k: round(v / t, 3) for k, v in c.most_common()}
    tgt_wf = collections.Counter()
    for d, D in F['domains'].items():
        tot = sum(D['workflow'].get(w, 0) + 0.5 for w in F['kept']['workflow'])
        for w in F['kept']['workflow']:
            tgt_wf[w] += (D['workflow'].get(w, 0) + 0.5) / tot / len(F['domains'])
    R['spread'] = {'domain_accepted': shares(r['spec']['domain'] for r in acc),
                   'workflow_target': {k: round(v, 3) for k, v in tgt_wf.most_common()},
                   'workflow_accepted': shares(r['spec']['workflow'] for r in acc),
                   'traps_accepted_per_task': round(sum(len(r['spec']['traps']) for r in acc) / max(1, len(acc)), 2),
                   'traps_accepted': shares(t for r in acc for t in r['spec']['traps']),
                   'apps_accepted_top': shares(a_ for r in acc for a_ in r['spec']['apps']),
                   'open_draw_share': round(sum(r['spec']['leaf'] is None for r in acc) / max(1, len(acc)), 3),
                   'acceptance_by_domain': {d: round(sum(1 for r in acc if r['spec']['domain'] == d) /
                                                     max(1, sum(1 for s in spec_all if s['domain'] == d)), 3)
                                            for d in F['domains']}}
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1)[:6000])


if __name__ == '__main__':
    main()

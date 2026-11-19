"""Leniency vs self-preference: judges {Qwen, Super} x answer sources {Qwen, Super} (#41, user's design 2026-11-02).

A gap between two judges on one model's answers is not evidence of self-preference: one judge may be more lenient
everywhere. With both judges grading both models' answers:

  gap(src)        = pass(QwenJudge, src) - pass(SuperJudge, src)       on the items both judges graded
  leniency        = (gap(qwen answers) + gap(super answers)) / 2         Qwen judge minus Super judge, overall
  self-preference = gap(qwen answers) - gap(super answers)              the interaction

A positive interaction means the judges favour their own family's answers (Qwen's judge on Qwen's answers, Super's on
Super's, or both: the 2x2 cannot tell which). CIs come from a paired bootstrap over items: an item resample is applied
to all four cells, so each gap stays paired. When both sources answered the same items (by default the analysis keeps
only those), the two sources share each resample too. GDP.pdf resamples tasks, since a task's criteria are correlated.

On Omniscience the judge also decides the non-hallucination metric (INCORRECT vs PARTIAL vs NOT_ATTEMPTED), so a
second 2x2 uses "graded INCORRECT" as the binary (`omni-incorrect`).

Like-for-like: each judge must grade both answer sources with the same settings (temperature, top_p, top_k, effort,
thinking, max_tokens), read from each run's config.json. A mismatch stops the analysis unless --allow-mixed-settings.

Ground truth, where a rule can decide (deterministic.py): each judge's agreement, false-accept and false-reject rates
against the deterministic verdicts, per cell.

  python3 -m aa_evals.judges.selfpref --name r1 \\
      --cell omni:super:qwen=j41-s1-qwen38-27b-medium --cell omni:super:super=j41-s1-super-medium \\
      --cell omni:qwen:qwen=qwen-medium-r1-60 --cell omni:qwen:super=qwen-medium-r1-60-jsuper

A cell is <bench>:<answer source>:<judge>=<run id> with bench omni | hle | gdp, source and judge qwen | super. The run
id is looked up under the benchmark's runs root; an absolute run dir is taken as is. Writes <STUDY_ROOT>/selfpref/<name>.json and .md (ids and aggregates
only; no question text).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import report as rp
from . import stats
from .sweep import STUDY_ROOT, bench_dir

SIDES = ('qwen', 'super')
BENCH_LOADER = {'omni': rp.load_omni, 'hle': rp.load_hle, 'gdp': rp.load_gdp}


def identity(bench: str, key: tuple) -> tuple:
    """The item a key refers to, regardless of repeat: (item_id,) or, for GDP.pdf, (task_id, criterion_id)."""
    return (key[0], key[2]) if bench == 'gdp' else (key[0],)


def cluster(bench: str, key: tuple):
    """The bootstrap unit: the item, or for GDP.pdf the task."""
    return key[0]


def expectations(bench: str, judged: dict) -> dict:
    if bench == 'omni':
        return rp.omni_expect(judged)
    if bench == 'hle':
        return rp.hle_expect(judged)
    return {}


MEASURES = {'pass': lambda v: v['bin'], 'incorrect': lambda v: v['label'] == 'INCORRECT'}


def judge_settings(run_dir: Path) -> dict:
    """The judge request settings that matter, normalized across the three benchmarks' config.json layouts."""
    j = json.loads((run_dir / 'config.json').read_text()).get('judge') or {}
    g = j.get('generation') or j.get('settings') or {}
    ctk = g.get('chat_template_kwargs') or {}
    return {'server': j.get('spec'), 'temperature': g.get('temperature'), 'top_p': g.get('top_p'),
            'top_k': g.get('top_k'), 'max_tokens': g.get('max_tokens'),
            'thinking': ctk.get('enable_thinking', ctk.get('thinking')),
            'effort': ctk.get('reasoning_effort') or g.get('reasoning_effort'),
            'medium_effort': bool(ctk.get('medium_effort'))}


def _rate(xs: list) -> float | None:
    return 100 * sum(xs) / len(xs) if xs else None


def analyse(bench: str, cells: dict, *, intersect: bool = True, n_boot: int = 2000, seed: int = 0,
            measure: str = 'pass') -> dict:
    """cells: (source, judge) -> judged dict (a report loader's output). Returns the 2x2 measures for one bench.
    measure: 'pass' (CORRECT / judged correct / criterion passed) or 'incorrect' (Omniscience: graded INCORRECT)."""
    B = MEASURES[measure]
    sources = [s for s in SIDES if all((s, j) in cells for j in SIDES)]
    # per source: keys both judges graded ok
    paired = {s: sorted(set(k for k, v in cells[(s, 'qwen')].items() if v['ok'])
                        & set(k for k, v in cells[(s, 'super')].items() if v['ok'])) for s in sources}
    shared = None
    if intersect and len(sources) == 2:
        ids = [set(identity(bench, k) for k in paired[s]) for s in sources]
        shared = ids[0] & ids[1]
        if shared:
            paired = {s: [k for k in paired[s] if identity(bench, k) in shared] for s in sources}
    out = {'bench': bench, 'measure': measure, 'sources': sources, 'intersected': bool(shared), 'cells': {}, 'gaps': {}}
    for s in sources:
        for j in SIDES:
            d = cells[(s, j)]
            ks = paired[s]
            exp = expectations(bench, d)
            exp = {k: e for k, e in exp.items() if k in set(ks)}
            out['cells'][f'{s}-answers|{j}-judge'] = {
                'n': len(ks), 'pass_rate': round(_rate([B(d[k]) for k in ks]), 2) if ks else None,
                'pass_ci': stats.rate_ci(sum(B(d[k]) for k in ks), len(ks)),
                'n_judge_infra': sum(v.get('infra_judge', False) for v in d.values()),
                'det': rp.det_measures(bench, d, exp) if exp else None}
        ks = paired[s]
        gq = [B(cells[(s, 'qwen')][k]) for k in ks]
        gs = [B(cells[(s, 'super')][k]) for k in ks]
        out['gaps'][s] = {'gap_pp': round(_rate(gq) - _rate(gs), 2) if ks else None,
                          'kappa': (round(stats.cohen_kappa(gq, gs), 3) if ks and stats.cohen_kappa(gq, gs) is not None
                                    else None)}
    if len(sources) < 2 or not all(paired[s] for s in sources):
        return out

    # point estimates and the paired bootstrap
    def diff_by_cluster(s):
        by = {}
        for k in paired[s]:
            c = cluster(bench, k)
            q, su = B(cells[(s, 'qwen')][k]), B(cells[(s, 'super')][k])
            a = by.setdefault(c, [0, 0, 0])
            a[0] += q
            a[1] += su
            a[2] += 1
        return by  # cluster -> [qwen passes, super passes, n]

    by = {s: diff_by_cluster(s) for s in sources}

    def gap(s, clusters):
        q = su = n = 0
        for c in clusters:
            a = by[s].get(c)
            if a:
                q, su, n = q + a[0], su + a[1], n + a[2]
        return 100 * (q - su) / n if n else None

    g = {s: gap(s, list(by[s])) for s in sources}
    inter = g['qwen'] - g['super']
    len_ = (g['qwen'] + g['super']) / 2
    rng = random.Random(seed)
    shared_clusters = sorted(set(by['qwen']) & set(by['super'])) if shared else None
    boots_i, boots_l = [], []
    for _ in range(n_boot):
        if shared_clusters:
            draw = [rng.choice(shared_clusters) for _ in shared_clusters]
            dq, ds = draw, draw
        else:
            cq, cs = list(by['qwen']), list(by['super'])
            dq = [rng.choice(cq) for _ in cq]
            ds = [rng.choice(cs) for _ in cs]
        a, b = gap('qwen', dq), gap('super', ds)
        if a is None or b is None:
            continue
        boots_i.append(a - b)
        boots_l.append((a + b) / 2)

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * (len(v) - 1))], 2), round(v[int(0.975 * (len(v) - 1))], 2)] if v else None

    out['leniency_pp'] = {'estimate': round(len_, 2), 'ci95': ci(boots_l),
                          'meaning': 'Qwen judge minus Super judge, averaged over both answer sources'}
    out['self_preference_pp'] = {'estimate': round(inter, 2), 'ci95': ci(boots_i),
                                 'meaning': '(Qwen-Super judge gap on Qwen answers) - (the gap on Super answers); for measure "incorrect" '
                   'a positive value is harshness toward the own family, not self-preference'}
    out['bootstrap'] = {'n_boot': n_boot, 'unit': 'task' if bench == 'gdp' else 'item',
                        'joint_over_sources': bool(shared_clusters)}
    return out


def to_markdown(name: str, res: dict) -> str:
    L = [f'# Leniency vs self-preference `{name}`', '',
         'Pass rate = CORRECT (Omniscience), judged correct (HLE), criterion passed (GDP.pdf). Leniency = Qwen judge '
         'minus Super judge, averaged over both answer sources. Self-preference = the interaction (positive: the '
         'judges favour their own family). 95% CIs: paired bootstrap over items (GDP.pdf: tasks). `omni-incorrect` '
         'uses "graded INCORRECT" as the binary (it drives non-hallucination). The Qwen judge is the same checkpoint '
         'as the Qwen model under test (Qwen3.8 27B FP8): the strongest form of self-judging. A CI that includes 0 '
         'means "not detected at this n", not "absent".', '']
    for bench, r in res.items():
        L += [f'## {bench}' + ('  (both sources on the same items)' if r.get('intersected') else ''), '',
              '| answers \\ judge | Qwen judge | Super judge | gap pp | κ |', '|---|---|---|---|---|']
        for s in r['sources']:
            cq, cs = r['cells'][f'{s}-answers|qwen-judge'], r['cells'][f'{s}-answers|super-judge']
            L.append(f'| {s} answers (n={cq["n"]}) | {cq["pass_rate"]} {cq["pass_ci"]} | {cs["pass_rate"]} '
                     f'{cs["pass_ci"]} | {r["gaps"][s]["gap_pp"]} | {r["gaps"][s]["kappa"]} |')
        if 'leniency_pp' in r:
            harsh = r.get('measure') == 'incorrect'  # a higher rate is harsher: the signs read the other way
            L += ['', f'- {"Harshness" if harsh else "Leniency"} (Qwen − Super judge): '
                      f'**{r["leniency_pp"]["estimate"]} pp** {r["leniency_pp"]["ci95"]}',
                  f'- Interaction: **{r["self_preference_pp"]["estimate"]} pp** {r["self_preference_pp"]["ci95"]} '
                  + ('(positive: each judge calls its own family INCORRECT more often, i.e. harsher on itself; '
                     'negative would be self-preference)' if harsh else
                     '(positive: each judge passes its own family more often, i.e. self-preference)')]
        else:
            L += ['', '- Only one answer source so far: leniency and self-preference need both.']
        if r.get('judge_settings'):
            L += ['', '| judge | settings (both answer sources) |', '|---|---|']
            for j, st in r['judge_settings'].items():
                L.append(f'| {j} | {st} |')
        dets = [(c, v['det']) for c, v in r['cells'].items() if v.get('det')]
        if dets:
            L += ['', '| cell | det. agree % [CI] (n) | false accept % (n neg) | false reject % (n pos) |',
                  '|---|---|---|---|']
            for c, d in dets:
                L.append(f'| {c} | {d["agreement"]} {d["agreement_ci"]} ({d["n"]}) | {d["false_accept"]} '
                         f'({d["n_neg"]}) | {d["false_reject"]} ({d["n_pos"]}) |')
        L.append('')
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--cell', action='append', required=True, help='<bench>:<source>:<judge>=<run id or absolute run dir>')
    ap.add_argument('--no-intersect', action='store_true', help='keep each source\'s own items')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--allow-mixed-settings', action='store_true',
                    help='run even if a judge graded the two answer sources with different settings')
    a = ap.parse_args(argv)
    per_bench: dict = {}
    settings: dict = {}
    for spec in a.cell:
        lhs, run_id = spec.split('=', 1)
        bench, src, judge = lhs.split(':')
        assert bench in BENCH_LOADER and src in SIDES and judge in SIDES, spec
        step = {'omni': 'omni', 'hle': 'hle', 'gdp': 'gdp'}[bench]
        run_dir = Path(run_id) if Path(run_id).is_absolute() else Path(bench_dir(step)) / run_id
        if not run_dir.exists():
            raise SystemExit(f'{spec}: {run_dir} not found')
        per_bench.setdefault(bench, {})[(src, judge)] = BENCH_LOADER[bench](run_dir)
        settings.setdefault(bench, {}).setdefault(judge, {})[src] = judge_settings(run_dir)
    mixed = []
    for bench, by_judge in settings.items():
        for judge, by_src in by_judge.items():
            if len(by_src) == 2 and by_src['qwen'] != by_src['super']:
                mixed.append(f'{bench} {judge} judge: qwen answers {by_src["qwen"]} vs super answers {by_src["super"]}')
    if mixed and not a.allow_mixed_settings:
        raise SystemExit('not like-for-like (pass --allow-mixed-settings to run anyway):\n' + '\n'.join(mixed))
    res = {}
    for b, cells in per_bench.items():
        for measure in (('pass', 'incorrect') if b == 'omni' else ('pass',)):
            r = analyse(b, cells, intersect=not a.no_intersect, n_boot=a.n_boot, measure=measure)
            r['judge_settings'] = {j: (v['qwen'] if len(v) == 1 or v.get('qwen') == v.get('super') else v)
                                   for j, v in settings[b].items()}
            r['mixed_settings'] = [m for m in mixed if m.startswith(b + ' ')]
            res[b if measure == 'pass' else f'{b}-{measure}'] = r
    res_meta = {'cells': a.cell, 'results': res}
    out = STUDY_ROOT / 'selfpref'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{a.name}.json').write_text(json.dumps(res_meta, indent=1, default=str))
    md = to_markdown(a.name, res)
    (out / f'{a.name}.md').write_text(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Judge study s3 (#41): four judges re-grade the same stored generations; who agrees with whom, and where.

  python3 -m aa_evals.judges.s3 report             # writes /data/aai/aa_evals/runs/judges/s3/report.{json,md}
  python3 -m aa_evals.judges.s3 contested --bench hle --n 20   # contested item ids + short excerpts (local use only)

Judges (arms.py): Qwen3.8-27B (`qwen38-27b-medium-pin`), Super VL (`super-full-pin`, a voter), Gemma 4 26B-A4B-it
(`gemma4-26b-a4b-pin`) and Gemma 4 31B-it FP8 (`gemma4-31b-fp8-pin`). Qwen's and Super's verdicts on Qwen's final
generations already exist (the r3 runs and their `-jsuper` siblings); the Gemma verdicts come from the s3 sweeps
(j41-s3q{m,x}-<arm>). Secondary: Super's s1 generations (j41-s1-*, s2pin for GDP.pdf, j41-s3s-<arm>).

Every measure is on the NON-deterministic items only (deterministic items are matched by every judge by construction,
audit 1): pairwise disagreements and κ (Omniscience also 4-class), each judge's agreement with the majority of the
other three, split patterns, leniency against that majority (paired bootstrap; GDP.pdf resamples tasks), the
deterministic false accepts/rejects, Qwen's scores under each judge, reliability, and cost. Adjudications (my reading
of contested items, ids and verdicts only) live in <study dir>/adjudication.json: {"<bench>|<source>|<key>": label}.
Only ids and aggregates are written. HLE questions are never written.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from . import stats
from .report import LOADERS, det_measures, hle_active_wall, hle_expect, omni_expect
from .sweep import RUNS, STUDY_ROOT, read_steps

FB = Path('/data/aai/aa_evals/runs_fallback')
OMNI, HLE, GDP = RUNS / 'omniscience', RUNS / 'hle', RUNS / 'gdppdf'
JUDGES = ['qwen', 'super', 'g26', 'g31']
ARM = {'qwen': 'qwen38-27b-medium-pin', 'super': 'super-full-pin', 'g26': 'gemma4-26b-a4b-pin',
       'g31': 'gemma4-31b-fp8-pin'}
FULL_JUDGMENTS = {'omni': 300, 'hle': 1000, 'gdp': 500 * 12.75}   # one full run (GDP.pdf: k = 5, ~12.75 criteria)


def gem_dir(root: Path, study: str, judge: str, rt: bool = False) -> Path:
    return root / (f'j41-{study}-{ARM[judge]}' + ('-rt' if rt else ''))


def sources() -> dict:
    """bench -> source -> {'role', judge -> run dir, 'rt': judge -> retest run dir}"""
    q = lambda study, j, root, rt=False: gem_dir(root, study, j, rt)
    S = {'omni': {}, 'hle': {}, 'gdp': {}}
    for tag, st in (('medium', 's3qm'), ('xhigh', 's3qx')):
        S['omni'][f'qwen-{tag}'] = {'role': 'primary', 'qwen': OMNI / f'qwen-{tag}-r3-300',
                                    'super': OMNI / f'qwen-{tag}-r3-300-jsuper',
                                    'g26': q(st, 'g26', OMNI), 'g31': q(st, 'g31', OMNI),
                                    'rt': {j: q(st, j, OMNI, True) for j in ('g26', 'g31')}}
    S['hle']['qwen-medium'] = {'role': 'primary', 'qwen': FB / 'hle/qwen-diamond-medium',
                               'super': FB / 'hle/qwen-diamond-medium-jsuper',
                               'g26': q('s3qm', 'g26', HLE), 'g31': q('s3qm', 'g31', HLE)}
    S['hle']['qwen-xhigh'] = {'role': 'primary', 'qwen': FB / 'hle/qwen-diamond-xhigh-strat300',
                              'super': FB / 'hle/qwen-diamond-xhigh-strat300-jsuper',
                              'g26': q('s3qx', 'g26', HLE), 'g31': q('s3qx', 'g31', HLE)}
    for tag, st in (('medium', 's3qm'), ('xhigh', 's3qx')):
        S['gdp'][f'qwen-{tag}'] = {'role': 'primary', 'qwen': FB / f'gdppdf/r3-qwen38-{tag}',
                                   'super': FB / f'gdppdf/r3-qwen38-{tag}-jsuper',
                                   'g26': q(st, 'g26', GDP), 'g31': q(st, 'g31', GDP)}
    # secondary: Super's s1 generations (Qwen at the s1 medium arm = the pinned settings at the registry's 32k;
    # Super at s1 super-full; GDP.pdf at the s2pin pinned arms)
    S['omni']['super-s1'] = {'role': 'secondary', 'qwen': OMNI / 'j41-s1-qwen38-27b-medium',
                             'super': OMNI / 'j41-s1-super-full', 'g26': q('s3s', 'g26', OMNI),
                             'g31': q('s3s', 'g31', OMNI),
                             'rt': {'qwen': OMNI / 'j41-s1-qwen38-27b-medium-rt', 'super': OMNI / 'j41-s1-super-full-rt',
                                    'g26': q('s3s', 'g26', OMNI, True), 'g31': q('s3s', 'g31', OMNI, True)}}
    S['hle']['super-s1'] = {'role': 'secondary', 'qwen': HLE / 'j41-s1-qwen38-27b-medium',
                            'super': HLE / 'j41-s1-super-full', 'g26': q('s3s', 'g26', HLE),
                            'g31': q('s3s', 'g31', HLE)}
    S['gdp']['super-s1'] = {'role': 'secondary', 'qwen': GDP / 'j41-s2pin-qwen38-27b-medium-pin',
                            'super': GDP / 'j41-s2pin-super-full-pin', 'g26': q('s3s', 'g26', GDP),
                            'g31': q('s3s', 'g31', GDP)}
    return S


def binary(bench: str, v: dict, which: str = 'pass') -> bool:
    if bench == 'omni' and which == 'incorrect':
        return v['label'] == 'INCORRECT'
    return bool(v['bin'])


def cluster_of(bench: str, key: tuple):
    return key[:2] if bench == 'gdp' else key        # (source, task_id) for GDP.pdf


def boot_ci(diffs: dict, n_boot: int = 2000, seed: int = 0) -> list | None:
    """Paired bootstrap over clusters of the mean of per-key values; diffs: cluster -> list of values."""
    cl = list(diffs)
    if len(cl) < 2:
        return None
    rng = random.Random(seed)
    tot = [sum(diffs[c]) for c in cl]
    cnt = [len(diffs[c]) for c in cl]
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(cl)) for _ in cl]
        n = sum(cnt[i] for i in idx)
        if n:
            vals.append(sum(tot[i] for i in idx) / n)
    vals.sort()
    return [round(100 * vals[int(0.025 * (len(vals) - 1))], 2), round(100 * vals[int(0.975 * (len(vals) - 1))], 2)]


def load_bench(bench: str, srcs: dict, role: str) -> tuple[dict, dict, dict]:
    """(judge -> key -> row), expect (key -> expectation), retest (judge -> key -> row); keys are (source, *key)."""
    data = {j: {} for j in JUDGES}
    expect, rt = {}, defaultdict(dict)
    for name, s in srcs.items():
        if s['role'] != role:
            continue
        runs = {}
        for j in JUDGES:
            p = s[j]
            if not (p / 'config.json').exists():
                print(f'[s3] {bench}/{name}: no {j} run at {p}')
                continue
            runs[j] = LOADERS[bench](p)
            for k, v in runs[j].items():
                data[j][(name, *k)] = v
        if not runs:
            continue
        first = runs.get('qwen') or next(iter(runs.values()))
        e = omni_expect(first) if bench == 'omni' else hle_expect(first) if bench == 'hle' else {}
        expect.update({(name, *k): x for k, x in e.items()})
        for j, p in (s.get('rt') or {}).items():
            if (p / 'config.json').exists():
                for k, v in LOADERS[bench](p).items():
                    rt[j][(name, *k)] = v
    return data, expect, rt


def analyse(bench: str, role: str, adjud: dict) -> dict:
    data, expect, rt = load_bench(bench, sources()[bench], role)
    js = [j for j in JUDGES if data[j]]
    common = sorted(set.intersection(*[{k for k, v in data[j].items() if v['ok']} for j in js])) if js else []
    nd = [k for k in common if k not in expect]
    out = {'judges': js, 'n_common': len(common), 'n_det': sum(k in expect for k in common), 'n_nondet': len(nd),
           'per_source_nondet': dict(Counter(k[0] for k in nd)), 'pairs': {}, 'judge': {}, 'splits': {}}
    lab = (lambda j, k: data[j][k]['label']) if bench == 'omni' else (lambda j, k: data[j][k]['bin'])
    # pairwise on non-deterministic items
    for x, y in combinations(js, 2):
        a, b = [data[x][k]['bin'] for k in nd], [data[y][k]['bin'] for k in nd]
        rec = {'n': len(nd), 'n_disagree': sum(p != q for p, q in zip(a, b)),
               'kappa': _r(stats.cohen_kappa(a, b))}
        if bench == 'omni':
            a4, b4 = [lab(x, k) for k in nd], [lab(y, k) for k in nd]
            rec.update(n_disagree_4class=sum(p != q for p, q in zip(a4, b4)), kappa_4class=_r(stats.cohen_kappa(a4, b4)))
        out['pairs'][f'{x}|{y}'] = rec
    # split patterns across all four judges
    if len(js) == 4:
        pat, lone, pair22 = Counter(), Counter(), Counter()
        for k in nd:
            votes = {j: lab(j, k) for j in js}
            c = Counter(votes.values())
            shape = '-'.join(str(n) for n in sorted(c.values(), reverse=True))
            pat[shape] += 1
            if shape == '3-1':
                odd = next(v for v, n in c.items() if n == 1)
                lone[next(j for j in js if votes[j] == odd)] += 1
            if shape == '2-2':   # which two side together (named by the side holding the first judge)
                grp = sorted((j for j in js if votes[j] == votes[js[0]]), key=js.index)
                pair22['+'.join(grp)] += 1
        out['splits'] = {'patterns': dict(pat), 'lone_dissenter_3_1': dict(lone), 'pairs_2_2': dict(pair22)}
        if bench == 'omni':  # the binary (CORRECT) splits too
            pb, lb = Counter(), Counter()
            for k in nd:
                votes = {j: data[j][k]['bin'] for j in js}
                c = Counter(votes.values())
                shape = '-'.join(str(n) for n in sorted(c.values(), reverse=True))
                pb[shape] += 1
                if shape == '3-1':
                    lb[next(j for j in js if c[votes[j]] == 1)] += 1
            out['splits']['binary_patterns'] = dict(pb)
            out['splits']['binary_lone_dissenter_3_1'] = dict(lb)
    # per judge: agreement with the majority of the other three, leniency, det errors, scores, reliability, cost
    for j in js:
        others = [o for o in js if o != j]
        row = {}
        for which in (('pass', 'incorrect') if bench == 'omni' else ('pass',)):
            agree = n = 0
            dif = defaultdict(list)
            for k in nd:
                maj = stats.majority([binary(bench, data[o][k], which) for o in others])
                if maj is None:
                    continue
                mine = binary(bench, data[j][k], which)
                n += 1
                agree += mine == maj
                dif[cluster_of(bench, k)].append(int(mine) - int(maj))
            tot = sum(sum(v) for v in dif.values())
            row[which] = {'n': n, 'agree': agree, 'agree_pct': _p(agree, n), 'agree_ci': stats.rate_ci(agree, n),
                          'leniency_pp': round(100 * tot / n, 2) if n else None, 'leniency_ci': boot_ci(dif)}
        if bench == 'omni':  # 4-class agreement with the plurality of the other three (ties excluded)
            agree = n = 0
            for k in nd:
                maj = stats.majority([lab(o, k) for o in others])
                if maj is not None:
                    n += 1
                    agree += lab(j, k) == maj
            row['4class'] = {'n': n, 'agree': agree, 'agree_pct': _p(agree, n)}
        row['det'] = det_measures(bench, data[j], expect) if expect else None
        allrows = list(data[j].values())
        ok = [v for v in allrows if v['ok']]
        row['n_judged'], row['n_ok'] = len(allrows), len(ok)
        row['n_judge_infra'] = sum(v['infra_judge'] for v in allrows)
        row['format_retries'] = sum(v.get('fmt_retries', 0) for v in allrows)
        cts = [v['ct'] for v in ok if v.get('ct')]
        pts = [v['pt'] for v in ok if v.get('pt')]
        row['completion_tokens_mean'] = round(sum(cts) / len(cts)) if cts else None
        row['completion_tokens_total'] = sum(cts)
        row['prompt_tokens_mean'] = round(sum(pts) / len(pts)) if pts else None
        if bench == 'omni' and rt.get(j):
            keys = [k for k in data[j] if k in rt[j] and data[j][k]['ok'] and rt[j][k]['ok']]
            row['retest_kappa_4class'] = _r(stats.cohen_kappa([data[j][k]['label'] for k in keys],
                                                              [rt[j][k]['label'] for k in keys]))
            row['retest_n'] = len(keys)
        adj = [(key, v) for key, v in adjud.items() if key.startswith(bench + '|')]
        if adj:
            right = n = 0
            for key, truth in adj:
                _, src, rest = key.split('|', 2)
                k = (src, *json.loads(rest))
                if k in data[j] and data[j][k]['ok']:
                    n += 1
                    right += (lab(j, k) == truth) if bench == 'omni' and isinstance(truth, str) \
                        else (data[j][k]['bin'] == truth)
            row['adjudicated'] = {'n': n, 'right': right}
        out['judge'][j] = row
    # lone-dissenter categories (binary): "judge j alone says v, every other judge says not v". Adjudicated items in
    # a category estimate how often the dissenter is right there; each judge's errors on these items are then
    # sum over categories of n_c * P(judge wrong | c) (the dissenter is wrong w.p. 1 - a_c, the others w.p. a_c).
    lone = {}
    adj_bin = {}
    for key, truth in adjud.items():
        if key.startswith(bench + '|'):
            _, src, rest = key.split('|', 2)
            adj_bin[(src, *json.loads(rest))] = (truth == 'CORRECT') if isinstance(truth, str) else truth
    for k in nd:
        votes = {j: data[j][k]['bin'] for j in js}
        for j in js:
            if all(votes[o] != votes[j] for o in js if o != j):
                c = lone.setdefault(f'{j}:{"pass" if votes[j] else "fail"}', {'n': 0, 'adj_n': 0, 'adj_dissenter_right': 0})
                c['n'] += 1
                if k in adj_bin:
                    c['adj_n'] += 1
                    c['adj_dissenter_right'] += adj_bin[k] == votes[j]
    est = {j: 0.0 for j in js}
    covered = 0
    for cat, c in lone.items():
        if not c['adj_n']:
            continue
        covered += c['n']
        a = c['adj_dissenter_right'] / c['adj_n']
        dj = cat.split(':')[0]
        for j in js:
            est[j] += c['n'] * ((1 - a) if j == dj else a)
    out['lone'] = lone
    out['lone_est_wrong'] = {j: round(v, 1) for j, v in est.items()}
    out['lone_covered'] = covered

    # pairwise gaps in agreement with the majority of the other three: the "who is closer to consensus" contrast
    # (each judge's own vote is excluded from its reference). Paired bootstrap over the same items/clusters.
    out['gaps'] = {}
    for x, y in combinations(js, 2):
        dif = defaultdict(list)
        for k in nd:
            mx = stats.majority([data[o][k]['bin'] for o in js if o != x])
            my = stats.majority([data[o][k]['bin'] for o in js if o != y])
            if mx is None or my is None:
                continue
            dif[cluster_of(bench, k)].append(int(data[x][k]['bin'] == mx) - int(data[y][k]['bin'] == my))
        n = sum(len(v) for v in dif.values())
        out['gaps'][f'{x}-{y}'] = {'n': n, 'n_differ': sum(v != 0 for vs in dif.values() for v in vs),
                                   'gap_pp': round(100 * sum(map(sum, dif.values())) / n, 2) if n else None,
                                   'ci': boot_ci(dif)}
    return out


def scores(bench: str, role: str) -> dict:
    out = {}
    for name, s in sources()[bench].items():
        if s['role'] != role:
            continue
        for j in JUDGES:
            p = s[j] / 'summary.json'
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if bench == 'omni':
                m = d.get('metrics') or {}
                out[f'{name}|{j}'] = {'accuracy': (m.get('accuracy') or {}).get('score'),
                                      'non_hallucination': (m.get('non_hallucination_rate') or {}).get('score')}
            else:
                out[f'{name}|{j}'] = {'score': d.get('score'), 'mean_pass': d.get('mean_pass')}
    return out


def shared_load() -> list[tuple[str, float, float, str]]:
    """Other sessions' load on our judge servers: lines `<server> <start ISO> <stop ISO> <who> [concurrency]`."""
    import datetime as dt
    p = STUDY_ROOT / 's3' / 'shared_load.txt'
    out = []
    for line in (p.read_text().splitlines() if p.exists() else []):
        f = line.split()
        if len(f) >= 4:
            ts = [dt.datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp() for x in f[1:3]]
            out.append((f[0], ts[0], ts[1], ' '.join(f[3:])))
    return out


def hle_window(run_dir: Path) -> tuple[float, float] | None:
    """[first judgment start, last judgment end] of an HLE re-grade (transcript mtime = when its judgment landed)."""
    spans = []
    for t in (run_dir / 'transcripts').glob('*.json'):
        lat = (json.loads(t.read_text()).get('judgment') or {}).get('latency_s')
        if lat:
            e = t.stat().st_mtime
            spans.append((e - lat, e))
    return (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None


def cost() -> dict:
    """Gemma: judge-busy wall time from the s3 sweeps (1xH100 each). A step that overlaps another session's logged
    load on the same server (shared_load.txt) is approximate; rates come from the clean steps when there are any."""
    loads = shared_load()
    steps = defaultdict(list)
    for study in ('s3qm', 's3qx', 's3s'):
        for r in read_steps(study):
            if r.get('kind') != 'step' or r.get('rc') not in (0, 2):
                continue
            j = next(k for k, v in ARM.items() if v == r['arm'])
            run = Path(r['bench_dir']) / r['run_id']
            wall = (hle_active_wall(run) if r['step'] == 'hle' else None) or r['wall_s']
            rows = LOADERS[r['step']](run)
            t0, t1 = r['started'], r['started'] + r['wall_s']
            if r['step'] == 'hle':   # judging starts after the transcript copy: test overlap on the judge-busy window
                t0, t1 = hle_window(run) or (t0, t1)
            hits = [f'{w}' for srv, a, b, w in loads if srv == r['server'] and a < t1 and b > t0]
            steps[f'{j}|{r["step"]}'].append({'run_id': r['run_id'], 'wall_s': wall, 'judgments': len(rows),
                                             'completion_tokens': sum(v['ct'] or 0 for v in rows.values() if v['ok']),
                                             'shared_with': hits})
    out = {}
    for k, recs in steps.items():
        b = k.split('|')[1].replace('-rt', '')
        clean = [x for x in recs if not x['shared_with']]
        use = clean or recs
        w, n, ct = (sum(x[f] for x in use) for f in ('wall_s', 'judgments', 'completion_tokens'))
        out[k] = {'steps': recs, 'approx': not clean, 'wall_s': w, 'judgments': n, 'completion_tokens': ct,
                  'gen_tok_s': round(ct / w, 1) if w else None,
                  'gpu_h_full_run': round(w / 3600 * FULL_JUDGMENTS[b] / n, 3) if n else None,
                  'n_steps_used': len(use), 'n_steps': len(recs)}
    return out


def _r(x):
    return round(x, 3) if x is not None else None


def _p(a, n):
    return round(100 * a / n, 1) if n else None


def contested(bench: str, n: int, role: str = 'primary') -> None:
    """Print contested non-deterministic items (local reading; never post question text)."""
    data, expect, _ = load_bench(bench, sources()[bench], role)
    js = [j for j in JUDGES if data[j]]
    common = sorted(set.intersection(*[{k for k, v in data[j].items() if v['ok']} for j in js]))
    lab = (lambda j, k: data[j][k]['label']) if bench == 'omni' else (lambda j, k: data[j][k]['bin'])
    rows = [k for k in common if k not in expect and len({lab(j, k) for j in js}) > 1]
    print(f'{len(rows)} contested')
    for k in rows[:n]:
        print(json.dumps({'key': k, **{j: lab(j, k) for j in js}}))


NAMES = {'qwen': 'Qwen3.8-27B', 'super': 'Super VL', 'g26': 'Gemma4-26B-A4B', 'g31': 'Gemma4-31B-FP8'}
BENCH = {'omni': 'Omniscience', 'hle': 'HLE-Diamond', 'gdp': 'GDP.pdf'}


def _ci(c):
    return f'[{c[0]}, {c[1]}]' if c else ''


def to_markdown(rep: dict) -> str:
    L = ['# Judge study s3', '']
    for key, B in rep['benches'].items():
        bench, role = key.split('|')
        if not B['judges']:
            continue
        L += [f'## {BENCH[bench]} ({role}): {B["n_nondet"]} non-deterministic items of {B["n_common"]} '
              f'({B["n_det"]} deterministic); per source {B["per_source_nondet"]}', '',
              '| judge | agrees with the other three (binary) % [CI] (n) | lone dissenter in 3-1 | leniency pp [CI] '
              + ('| INCORRECT-rate gap pp [CI] | 4-class agreement % (n) ' if bench == 'omni' else '')
              + '| det. false accept / reject % | adjudicated right / n | judge infra (format retries) '
              + ('| retest κ (4-class) ' if bench == 'omni' else '') + '| completion tok / judgment |',
              '|---|---|---|---|' + ('---|---|' if bench == 'omni' else '') + '---|---|---|'
              + ('---|' if bench == 'omni' else '') + '---|']
        lone = (B.get('splits') or {}).get('binary_lone_dissenter_3_1' if bench == 'omni' else 'lone_dissenter_3_1', {})
        for j, r in B['judge'].items():
            p, d, adj = r['pass'], r.get('det') or {}, r.get('adjudicated') or {}
            row = (f'| {NAMES[j]} | {p["agree_pct"]} {_ci(p["agree_ci"])} ({p["n"]}) | {lone.get(j, 0)} | '
                   f'{p["leniency_pp"]:+} {_ci(p["leniency_ci"])} |')
            if bench == 'omni':
                inc = r['incorrect']
                row += f' {inc["leniency_pp"]:+} {_ci(inc["leniency_ci"])} | {r["4class"]["agree_pct"]} ({r["4class"]["n"]}) |'
            row += (f' {d.get("false_accept")} / {d.get("false_reject")} (n {d.get("n_neg")}/{d.get("n_pos")}) | '
                    f'{adj.get("right", "")}{" / " + str(adj["n"]) if adj else ""} | {r["n_judge_infra"]} ({r["format_retries"]}) |')
            if bench == 'omni':
                row += f' {r.get("retest_kappa_4class", "")} |'
            row += f' {r["completion_tokens_mean"]} |'
            L.append(row)
        L += ['', '| pair | disagreements (binary) | κ binary' + (' | disagreements 4-class | κ 4-class' if bench == 'omni' else '') + ' |',
              '|---|---|---' + ('|---|---' if bench == 'omni' else '') + '|']
        for pr, v in B['pairs'].items():
            a, b = pr.split('|')
            L.append(f'| {NAMES[a]} / {NAMES[b]} | {v["n_disagree"]} | {v["kappa"]}'
                     + (f' | {v["n_disagree_4class"]} | {v["kappa_4class"]}' if bench == 'omni' else '') + ' |')
        sp = B.get('splits') or {}
        if sp:
            L += ['', f'Split patterns ({"4-class" if bench == "omni" else "binary"}): {sp.get("patterns")}; '
                  f'2-2 pairings: {sp.get("pairs_2_2")}'
                  + (f'; binary patterns: {sp.get("binary_patterns")}' if bench == 'omni' else '')]
        L += ['', 'Gap in agreement with the other three (row minus column, pp [paired bootstrap CI], items where the two differ):', '']
        L += ['| | ' + ' | '.join(NAMES[j] for j in B['judges']) + ' |', '|---' * (len(B['judges']) + 1) + '|']
        for x in B['judges']:
            cells = []
            for y in B['judges']:
                g = B['gaps'].get(f'{x}-{y}')
                gi = B['gaps'].get(f'{y}-{x}')
                if x == y:
                    cells.append('')
                elif g:
                    cells.append(f'{g["gap_pp"]:+} {_ci(g["ci"])} ({g["n_differ"]})')
                elif gi:
                    ci = [-gi['ci'][1], -gi['ci'][0]] if gi['ci'] else None
                    cells.append(f'{-gi["gap_pp"]:+} {_ci(ci)} ({gi["n_differ"]})')
            L.append(f'| {NAMES[x]} | ' + ' | '.join(cells) + ' |')
        if B.get('lone'):
            L += ['', f'Lone-dissenter categories (count; adjudicated: dissenter right / read). Estimated errors on the '
                  f'{B["lone_covered"]} lone-dissent items in adjudicated categories: '
                  + ', '.join(f'{NAMES[j]} {v}' for j, v in B['lone_est_wrong'].items()), '',
                  '| dissenter says | items | dissenter right / read |', '|---|---|---|']
            for cat, c in sorted(B['lone'].items(), key=lambda x: -x[1]['n']):
                j, v = cat.split(':')
                L.append(f'| {NAMES[j]} alone: {v} | {c["n"]} | {c["adj_dissenter_right"]} / {c["adj_n"]} |')
        sc = rep['scores'].get(key) or {}
        if sc:
            L += ['', 'Scores under each judge: ' + '; '.join(f'{k.replace("|", " / ")}: {v}' for k, v in sc.items())]
        L.append('')
    L += ['## Cost (Gemma, 1xH100 each; rates from the steps without logged outside load)', '',
          '| judge / step | steps used | judgments | completion tokens (accepted answers) | judge-busy s | tok/s (accepted) | GPU-h per full run |',
          '|---|---|---|---|---|---|---|']
    for k, v in sorted(rep['cost'].items()):
        j, b = k.split('|')
        L.append(f'| {NAMES[j]} / {b} | {v["n_steps_used"]}/{v["n_steps"]} | {v["judgments"]} | {v["completion_tokens"]} | {round(v["wall_s"])}{"*" if v.get("approx") else ""} | '
                 f'{v["gen_tok_s"]} | {v["gpu_h_full_run"]} |')
    L += ['', '`*` overlaps another session\'s logged load on the same server (shared_load.txt): approximate.']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['report', 'contested'])
    ap.add_argument('--bench', default='hle')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--role', default='primary')
    a = ap.parse_args(argv)
    out = STUDY_ROOT / 's3'
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'contested':
        contested(a.bench, a.n, a.role)
        return 0
    adj_path = out / 'adjudication.json'
    adjud = json.loads(adj_path.read_text()) if adj_path.exists() else {}
    rep = {'arms': ARM, 'benches': {}, 'scores': {}, 'cost': cost()}
    for bench in ('omni', 'hle', 'gdp'):
        for role in ('primary', 'secondary'):
            rep['benches'][f'{bench}|{role}'] = analyse(bench, role, adjud)
            rep['scores'][f'{bench}|{role}'] = scores(bench, role)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    md = to_markdown(rep)
    (out / 'report.md').write_text(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

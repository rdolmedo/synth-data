"""Judge study report: arm × measure tables from the re-grade runs (#41).

  python3 -m aa_evals.judges.report --study s1

It reads /data/aai/aa_evals/runs/judges/<study>/steps.jsonl, which the sweep writes, and each step's bench run.
It writes report.json and report.md next to steps.jsonl. Only ids and aggregates are written, never question text.
The measures (protocol: #41):
  det   agreement with deterministic verdicts (HLE MC, HLE exact sure-sets, Omniscience sure-sets)
  κ     pairwise Cohen's kappa; κ with the leave-one-out majority of the other servers' primary arms
  AA    Super's score under each arm vs AA's published number
  cost  judge tokens per judgment, wall time of the step, GPU-hours extrapolated to a full run
  rel   judge infra errors; Omniscience test-retest κ
  bias  leniency = the arm's pass rate minus the leave-one-out majority's on the same items (self-preference
        when the generations come from the arm's own family; today all generations are Super's)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import arms as arms_mod
from . import deterministic as det
from . import stats
from .sweep import RUNS, STUDY_ROOT, read_steps
from aa_evals.common import jsonl

FULL_JUDGMENTS = {'omni': 300, 'hle': 1000, 'gdp': 500 * 12.75}  # one full run of each benchmark
AA_SUPER = {'omni': {'accuracy': 21.2, 'non_hallucination_rate': 69.5}, 'hle': 8.8, 'gdp': 8.0}


def _jsonl(path: Path) -> list[dict]:
    return jsonl.read(path)


def _last(rows: list[dict], key) -> dict:
    out = {}
    for r in rows:
        out[key(r)] = r
    return out


# ---------- loaders: key -> {'ok', 'bin', 'label', 'pt', 'ct'} ----------

def load_omni(run_dir: Path) -> dict:
    rows = _last(_jsonl(run_dir / 'items.jsonl'), lambda r: (r['item_id'], r['repeat']))
    out = {}
    for k, r in rows.items():
        if r.get('grade_source') in ('rule_empty', 'cutoff') or (r['status'] == 'infra_error' and r.get('stage') == 'model'):
            continue  # no judge call was made (empty answer, or no answer in the source run yet)
        ok = r['status'] in ('ok', 'model_error')
        out[k] = {'ok': ok, 'infra_judge': r['status'] == 'infra_error' and r.get('stage') == 'judge',
                  'label': r.get('grade') if ok else None, 'bin': (r.get('grade') == 'CORRECT') if ok else None,
                  'pt': r.get('judge_prompt_tokens'), 'ct': r.get('judge_completion_tokens'),
                  'response': r.get('response'), 'gold': r.get('gold')}
    return out


def load_hle(run_dir: Path) -> dict:
    rows = _last(_jsonl(run_dir / 'items.jsonl'), lambda r: (r['item_id'], r['repeat']))
    out = {}
    for k, r in rows.items():
        if r['status'] == 'model_error' or (r['status'] == 'infra_error' and r.get('error_stage') != 'judge'):
            continue  # no final answer (or none yet in the source run): never judged
        ok = r['status'] == 'ok'
        pt = ct = None
        fmt_retries = 0
        t = Path(r['transcript']) if r.get('transcript') else None
        if t and t.exists():
            j = json.loads(t.read_text()).get('judgment') or {}
            u = j.get('usage') or {}  # the accepted attempt only
            pt, ct = u.get('prompt_tokens'), u.get('completion_tokens')
            fmt_retries = sum('format_error' in at for at in j.get('attempts') or [])
        out[k] = {'ok': ok, 'infra_judge': r['status'] == 'infra_error' and r.get('error_stage') == 'judge',
                  'label': r.get('judge_correct') if ok else None, 'bin': (r.get('judge_correct') == 'yes') if ok else None,
                  'pt': pt, 'ct': ct, 'answer_type': r.get('answer_type'), 'transcript': r.get('transcript'),
                  'fmt_retries': fmt_retries}
    return out


def hle_active_wall(run_dir: Path) -> float | None:
    """Seconds the judge was busy: the union of [finish - latency, finish] over judged transcripts.

    An HLE re-grade first copies the source transcripts, which can take minutes on a busy Lustre, so the step's wall
    time overstates the judge's. Each judged transcript is rewritten when its judgment lands, so its mtime is the
    finish time, and the judgment's latency_s spans all of its attempts."""
    spans = []
    for t in (run_dir / 'transcripts').glob('*.json'):
        j = json.loads(t.read_text()).get('judgment') or {}
        lat = j.get('latency_s')
        if lat:
            end = t.stat().st_mtime
            spans.append((end - lat, end))
    if not spans:
        return None
    spans.sort()
    total, (s0, e0) = 0.0, spans[0]
    for s, e in spans[1:]:
        if s > e0:
            total, s0, e0 = total + e0 - s0, s, e
        else:
            e0 = max(e0, e)
    return total + e0 - s0


def load_gdp(run_dir: Path) -> dict:
    out = {}
    for rec in _last(_jsonl(run_dir / 'grades.jsonl'), lambda r: (r['task_id'], r['repeat'])).values():
        for g in rec.get('grades') or []:
            ok = g.get('result') in ('pass', 'fail')
            us = g.get('usage') or []
            us = [us] if isinstance(us, dict) else us  # one entry per try
            out[(rec['task_id'], rec['repeat'], g.get('id'))] = {
                'ok': ok, 'infra_judge': not ok, 'label': g.get('result') if ok else None,
                'bin': (g.get('result') == 'pass') if ok else None,
                'pt': sum((u or {}).get('prompt_tokens') or 0 for u in us) or None,
                'ct': sum((u or {}).get('completion_tokens') or 0 for u in us) or None}
    return out


LOADERS = {'omni': load_omni, 'omni-rt': load_omni, 'hle': load_hle, 'gdp': load_gdp}


# ---------- deterministic expectations ----------

def omni_expect(judged: dict) -> dict:
    out = {}
    for k, v in judged.items():
        e = det.omni_expected(v.get('response') or '', v.get('gold') or '')
        if e:
            out[k] = e
    return out


def hle_expect(any_run: dict, dataset: str = 'cais/hle-diamond') -> dict:
    """item key -> (kind, expected bool). Needs the gated dataset (gold answers) and the source transcripts."""
    try:
        from aa_evals.hle import data as hle_data
        gold = {it['id']: it for it in hle_data.load(dataset)}
    except Exception as e:  # dataset not available here
        print(f'[report] HLE deterministic checks skipped: {e}')
        return {}
    out = {}
    for k, v in any_run.items():
        it = gold.get(k[0])
        t = Path(v['transcript']) if v.get('transcript') else None
        if it is None or t is None or not t.exists():
            continue
        content = ((json.loads(t.read_text()).get('generation') or {}).get('content')) or ''
        if it['answer_type'] == 'multipleChoice':
            e = det.hle_mc(content, it['answer'])
            if e is not None:
                out[k] = ('mc', e)
        else:
            e = det.hle_exact(content, it['answer'])
            if e is not None:
                out[k] = ('exact-' + ('correct' if e else 'wrong'), e)
    return out


# ---------- measures ----------

def det_measures(bench: str, judged: dict, expect: dict) -> dict:
    groups = defaultdict(lambda: [0, 0])  # kind -> [agree, n]
    fa = fr = n_pos = n_neg = 0
    for k, e in expect.items():
        v = judged.get(k)
        if not v or not v['ok']:
            continue
        if bench.startswith('omni'):
            kind, agree = f'omni-{e.lower()}', v['label'] == e
            pos, judged_pos = e == 'CORRECT', v['label'] == 'CORRECT'
        else:
            kind, want = e
            agree, pos, judged_pos = v['bin'] == want, want, v['bin']
        groups[kind][0] += agree
        groups[kind][1] += 1
        if pos:
            n_pos += 1
            fr += not judged_pos
        else:
            n_neg += 1
            fa += judged_pos
    agree = sum(g[0] for g in groups.values())
    n = sum(g[1] for g in groups.values())
    return {'n': n, 'agreement': round(100 * agree / n, 1) if n else None, 'agreement_ci': stats.rate_ci(agree, n),
            'false_accept': round(100 * fa / n_neg, 1) if n_neg else None, 'n_neg': n_neg,
            'false_reject': round(100 * fr / n_pos, 1) if n_pos else None, 'n_pos': n_pos,
            'by_kind': {k: {'agree': a, 'n': m} for k, (a, m) in sorted(groups.items())}}


def bench_score(step_rec: dict) -> dict | None:
    s = Path(step_rec['bench_dir']) / step_rec['run_id'] / 'summary.json'
    if not s.exists():
        return None
    s = json.loads(s.read_text())
    if step_rec['step'].startswith('omni'):
        m = s.get('metrics') or {}
        return {'accuracy': (m.get('accuracy') or {}).get('score'), 'accuracy_ci': (m.get('accuracy') or {}).get('ci95'),
                'non_hallucination_rate': (m.get('non_hallucination_rate') or {}).get('score'),
                'non_hallucination_ci': (m.get('non_hallucination_rate') or {}).get('ci95')}
    out = {'score': s.get('score'), 'ci95': s.get('ci95'), 'valid': s.get('valid')}
    if 'mean_pass' in s:  # GDP.pdf: all-pass is the headline; mean pass moves on small samples
        out.update(mean_pass=s['mean_pass'], mean_pass_ci95=s.get('mean_pass_ci95'))
    return out


def build(study: str) -> dict:
    steps = [r for r in read_steps(study) if r.get('kind') == 'step']
    # {'kind': 'note', 'arm', 'step', 'wall_approx': true, 'why'}: the server was busy with other work during the step
    shared_during = {(r['arm'], r['step']): r.get('why') for r in read_steps(study)
                     if r.get('kind') == 'note' and r.get('wall_approx')}
    latest = _last(steps, lambda r: (r['arm'], r['step']))
    # A step may run several times (resumes, stragglers): its cost is the sum of its invocations' wall times.
    wall = defaultdict(float)
    for r in steps:
        if r.get('rc') in (0, 2):
            wall[(r['arm'], r['step'])] += r['wall_s']
    runs = {}  # (arm, step) -> judged dict
    for (arm, step), rec in latest.items():
        runs[(arm, step)] = LOADERS[step](Path(rec['bench_dir']) / rec['run_id'])
    arms_present = sorted({a for a, _ in runs}, key=list(arms_mod.ARMS).index)
    primary = {arms_mod.ARMS[a].server: a for a in arms_present if arms_mod.PRIMARY.get(arms_mod.ARMS[a].server) == a}
    report = {'study': study, 'arms': {a: vars(arms_mod.ARMS[a]) for a in arms_present}, 'benches': {}}
    for bench in ('omni', 'hle', 'gdp'):
        have = {a: runs[(a, bench)] for a in arms_present if (a, bench) in runs}
        if not have:
            continue
        if bench == 'omni':
            expect = omni_expect(next(iter(have.values())))
        elif bench == 'hle':
            expect = hle_expect(next(iter(have.values())))
        else:
            expect = {}
        common = set.intersection(*[{k for k, v in d.items() if v['ok']} for d in have.values()])
        common = sorted(common)
        B = {'n_common': len(common), 'n_expect': len(expect), 'arms': {}, 'pairwise_kappa': {},
             'pairwise_nondet': {}}
        for a, d in have.items():
            rec = latest[(a, bench)]
            ok = [v for v in d.values() if v['ok']]
            pts = [v['pt'] for v in ok if v.get('pt') is not None]
            cts = [v['ct'] for v in ok if v.get('ct') is not None]
            n_judged = len(d)
            arm = arms_mod.ARMS[a]
            w = wall[(a, bench)]
            if bench == 'hle':  # judge-busy time, without the transcript copy that precedes it
                w = hle_active_wall(Path(rec['bench_dir']) / rec['run_id']) or w
            gpu_h = w * arm.gpus / 3600
            row = {
                'n_judged': n_judged, 'n_ok': len(ok), 'n_judge_infra': sum(v['infra_judge'] for v in d.values()),
                'pass_rate': round(100 * sum(v['bin'] for v in ok) / len(ok), 1) if ok else None,
                'score': bench_score(rec), 'det': det_measures(bench, d, expect) if expect else None,
                'prompt_tokens_mean': round(sum(pts) / len(pts)) if pts else None,
                'completion_tokens_mean': round(sum(cts) / len(cts)) if cts else None,
                'wall_s': round(w, 1), 'step_wall_s': round(wall[(a, bench)], 1),
                'format_retries': sum(v.get('fmt_retries', 0) for v in d.values()), 'concurrency': rec.get('concurrency'), 'gpu_h': round(gpu_h, 3),
                'gpu_h_full_run': round(gpu_h * FULL_JUDGMENTS[bench] / n_judged, 3) if n_judged else None,
                'wall_approx': arm.shared_server or (a, bench) in shared_during,
                'wall_note': shared_during.get((a, bench)),
            }
            # leave-one-out majority of the other servers' primary arms
            others = [p for s, p in primary.items() if s != arm.server and p in have]
            if others:
                maj = {k: stats.majority([have[o][k]['bin'] for o in others]) for k in common}
                keys = [k for k in common if maj[k] is not None]
                mine = [have[a][k]['bin'] for k in keys]
                theirs = [maj[k] for k in keys]
                row['kappa_loo_majority'] = round(stats.cohen_kappa(mine, theirs), 3) if keys else None
                row['kappa_loo_ci'] = stats.kappa_ci(mine, theirs) if keys else None
                row['leniency_pp'] = round(100 * (sum(mine) - sum(theirs)) / len(keys), 1) if keys else None
                row['loo_voters'] = others
                row['loo_label'] = ('κ vs the other server\'s primary arm' if len(others) == 1
                                    else 'κ vs the leave-one-out majority')
            if bench == 'omni' and (a, 'omni-rt') in runs:
                rt = runs[(a, 'omni-rt')]
                keys = [k for k in common if k in rt and rt[k]['ok']]
                k4 = stats.cohen_kappa([have[a][k]['label'] for k in keys], [rt[k]['label'] for k in keys])
                row['retest_kappa_4class'] = round(k4, 3) if k4 is not None else None
            B['arms'][a] = row
        names = list(have)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                lab = 'label' if bench == 'omni' else 'bin'
                k = stats.cohen_kappa([have[x][c][lab] for c in common], [have[y][c][lab] for c in common])
                B['pairwise_kappa'][f'{x} | {y}'] = round(k, 3) if k is not None else None
                # Deterministic items can't separate judges (every arm matches them by construction), so agreement
                # where it matters is measured on the rest (audit 1, #41).
                nd = [c for c in common if c not in expect]
                kn = stats.cohen_kappa([have[x][c][lab] for c in nd], [have[y][c][lab] for c in nd]) if nd else None
                dis = [c for c in common if have[x][c][lab] != have[y][c][lab]]
                B['pairwise_nondet'][f'{x} | {y}'] = {
                    'kappa': round(kn, 3) if kn is not None else None, 'n': len(nd), 'n_disagree': len(dis),
                    'n_disagree_on_det_items': sum(c in expect for c in dis)}
        report['benches'][bench] = B
    return report


def to_markdown(rep: dict) -> str:
    L = [f'# Judge study `{rep["study"]}`', '']
    for bench, B in rep['benches'].items():
        L += [f'## {bench}  (items judged ok by every arm: {B["n_common"]}; deterministic: {B["n_expect"]})', '',
              '| arm | ok / judged | judge infra (format retries) | det. agree % [CI] (n) | false accept / reject % '
              '| κ vs other server (LOO) | leniency pp '
              '| retest κ | Super score | tok in / out | wall s | GPU-h full run |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for a, r in B['arms'].items():
            d = r.get('det') or {}
            sc = r.get('score') or {}
            score = (f'{sc.get("accuracy")} / {sc.get("non_hallucination_rate")}' if bench == 'omni'
                     else f'{sc.get("score")} (mean pass {sc.get("mean_pass")})' if bench == 'gdp'
                     else f'{sc.get("score")}')
            L.append(f'| {a} | {r["n_ok"]} / {r["n_judged"]} | {r["n_judge_infra"]} ({r.get("format_retries", 0)}) | '
                     f'{d.get("agreement")} {d.get("agreement_ci")} ({d.get("n")}) | '
                     f'{d.get("false_accept")} / {d.get("false_reject")} | {r.get("kappa_loo_majority")} | '
                     f'{r.get("leniency_pp")} | {r.get("retest_kappa_4class", "")} | {score} | '
                     f'{r["prompt_tokens_mean"]} / {r["completion_tokens_mean"]} | {r["wall_s"]}{"*" if r["wall_approx"] else ""} | '
                     f'{r["gpu_h_full_run"]} |')
        L += ['', 'Pairwise κ: ' + ', '.join(f'{k} = {v}' for k, v in B['pairwise_kappa'].items()), '']
        if B.get('pairwise_nondet'):
            L += ['| pair | κ on non-deterministic items (n) | disagreements (on deterministic items) |',
                  '|---|---|---|']
            for k, v in B['pairwise_nondet'].items():
                L.append(f'| {k} | {v["kappa"]} ({v["n"]}) | {v["n_disagree"]} ({v["n_disagree_on_det_items"]}) |')
            L.append('')
    L += ['`*` wall time on a shared server: approximate. Omniscience κ is 4-class; HLE and GDP are binary.',
          'Deterministic agreement is a sanity check, not a quality ranking: rules decide only items where no '
          'reasonable grader could disagree, so every competent arm scores ~100% there. Judges differ only on the '
          'non-deterministic items; which judge is right on those needs adjudication. With two servers, "κ vs other '
          'server (LOO)" is κ with the one other primary arm, and the two primaries\' leniencies mirror each other.']
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', required=True)
    a = ap.parse_args(argv)
    rep = build(a.study)
    out = STUDY_ROOT / a.study
    (out / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    md = to_markdown(rep)
    (out / 'report.md').write_text(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

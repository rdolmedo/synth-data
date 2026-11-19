#!/usr/bin/env python3
"""TB-Science 0.1 under our Pi protocol (user decision 2026-11-02 ~23:40, #66): the headline harness.

  python -m aa_evals.tbsci.pi --model serving:aa-qwen-any --reasoning-effort medium --run-id r1-medium \
      --include a,b --aa-slug qwen3-8-27b-medium
  python -m aa_evals.tbsci.pi --agent oracle --run-id oracle-pi-v1
  python -m aa_evals.tbsci.pi --run-id ID --status | --score | --stop | --resubmit N
  python -m aa_evals.tbsci.pi ... --run-id r4-262k-xhigh --context-window 262144   # protocol override, new runs only

Defaults: the 70 pinned tasks (DATA/tasks.tar, patches.json applied) with a 2 h agent cap (user decision; environment
time, model time excluded) that the instruction states (cap.py: TASKS_2H; round 1 used DATA/tasks.tar, whose
instructions say 28800 s), seed namespace aa-tbsci-v1, 1 attempt (AA's repeats for the full run), at most 16 in
flight, harbor_ns multi-uid, NsSidecarEnvironment (sidecar artifacts, compose env), workers of 32 CPUs / 256 GB with 4 slots
(tasks declare up to 4 CPUs; harbor_ns does not enforce memory_mb, and a round-2 worker with 8 slots was held at
261 GB, killing all 8 attempts).  Every option of aa_evals.common.pi_eval.run applies.  The mini-swe-agent runner
(aa_evals.tbsci.run) stays as the AA-faithful option.
"""
import json
import math
import os
import re
import sys
import tarfile
import tomllib
from collections import defaultdict
from pathlib import Path

from aa_evals.common.pi_eval.run import RUNS_ROOT, main
from aa_evals.common import jsonl

DATA = Path('/data/aai/aa_evals/data/tbsci/v0.1.0')
ENV_CLASS = 'aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment'


def toml_field(tar: Path, fn) -> dict:
    out = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = Path(m.name).parts
            if len(p) == 3 and p[0] == 'tasks' and p[2] == 'task.toml':
                out[p[1]] = fn(tomllib.loads(tf.extractfile(m).read().decode()))
    return out


def domains(_bundle) -> dict:
    """task -> domain (life / physical / mathematical / engineering / earth), from the pinned manifest."""
    man = json.loads((DATA / 'manifest.json').read_text())['tasks']
    return {t: m['path'].split('/')[1] for t, m in man.items()}


# DATA/tasks.tar with the 2 h cap in task.toml and in the instruction's "You have N seconds" sentence (cap.py)
TASKS_2H = '/data/aai/aa_evals/runs_fallback/data/tbsci/tasks_2h.tar'
DEFAULTS = {'bench': 'tbsci', 'tasks_tar': TASKS_2H, 'seed_namespace': 'aa-tbsci-v1', 'attempts': 1,
            'max_in_flight': 16, 'agent_cap_sec': 7200.0, 'per_task_cap': True, 'multiuid': True, 'harbor_env': {'HARBOR_NS_MEMCAP': '1.5', 'HARBOR_NS_CPU_VIEW': '1'}, 'env_class': ENV_CLASS,   # memcap: PR #82; CPU view (multi-uid): harbor_ns/MULTIUID.md
            'aa_metric_key': 'terminal-bench-science', 'categories_fn': domains,
            # oracle-failing tasks for score_excl_oracle_failures: the Pi-path oracle over all tasks (same sandbox, tars, cap)
            'oracle_run': '/data/aai/aa_evals/runs_fallback/tbsci/oracle-pi-v1',
            # tasks that fail the same way on every same-seed retry in every Pi run (audit 2): deterministic
            # environment defects, reported as excluded tasks rather than infra noise (summary.json tbsci.excluded_tasks)
            'excluded_tasks': {'protein-active-learning': 'verifier_no_reward: the sidecar service `submission` never '
                               'becomes healthy (all 7 attempts of every run; its Pi-path oracle fails too)'},
            'notes': 'Pi protocol (user decision 2026-11-02 ~23:40, #66), not AA\'s mini-swe-agent; 2 h agent cap '
                     '(environment time); TB-Science v0.1.0 pinned (aa_evals/tbsci/README.md)'}


VALIDATION = [DATA / 'validation.json', DATA / 'validation_single_uid.json']   # validate.py: reference solve times
CAP_SEC = 7200.0
POOL = {'n_tasks': 70, 'name': 'TB-Science v0.1.0'}


def clopper_pearson(k: float, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact binomial 95% interval in % (the bootstrap collapses to [0, 0] at 0 passes)."""
    if n == 0:
        return None
    k = round(k)

    def cdf(x, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) > alpha / 2)
    return [round(100 * lower, 2), round(100 * upper, 2)]


def oracle_verdicts(oracle_run: str) -> dict[str, str]:
    """task -> pass / fail / infra / cutoff from the Pi-path oracle runs (comma-separated; the last run decides).
    A missing items.jsonl is an error: the run was never finalized (--score)."""
    out = {}
    for od in [x for x in str(oracle_run).split(',') if x]:
        f = Path(od) / 'items.jsonl'
        if not f.exists():
            raise SystemExit(f'oracle run {od} has no items.jsonl: finalize it (--score) first')
        per = defaultdict(list)
        for r in jsonl.read(f):
            per[r['item_id']].append(r)
        for t, rs in per.items():
            ok = [r for r in rs if r['status'] == 'ok']
            out[t] = (('pass' if all(r.get('score') == 1.0 for r in ok) else 'fail') if ok
                      else 'cutoff' if any(r['status'] == 'pending' for r in rs) else 'infra')
    return out


def over_cap(validation: list[Path], cap_sec: float) -> dict[str, float]:
    """task -> fastest reference solve (min) when even that exceeds the agent cap: infeasible under our protocol."""
    best = {}
    for f in validation:
        if f.exists():
            for t, v in json.loads(f.read_text()).items():
                if v.get('solve_min') is not None and (v.get('oracle') or {}).get('pass') and v['solve_min'] < 1e6:
                    best[t] = min(best.get(t, 1e9), v['solve_min'])
    return {t: round(m, 1) for t, m in sorted(best.items()) if m * 60 > cap_sec}


TOKENS = re.compile(rb'"totalTokens":\s*(\d+)')


def peak_context(transcript) -> int | None:
    """Largest prompt + completion of one episode's requests: max ``usage.totalTokens`` over the Pi session log
    (agent/pi/sessions/*.jsonl) in the packed trial or its directory.  Token counts only; no transcript content."""
    if not transcript:
        return None
    p, best = Path(transcript), None
    try:
        if p.is_dir():
            blobs = (f.read_bytes() for f in p.glob('**/agent/pi/sessions/*.jsonl'))
        else:
            tf = tarfile.open(p)
            blobs = (tf.extractfile(m).read() for m in tf.getmembers()
                     if m.isfile() and '/agent/pi/sessions/' in m.name and m.name.endswith('.jsonl'))
        for b in blobs:
            for x in TOKENS.findall(b):
                best = max(best or 0, int(x))
    except (OSError, tarfile.TarError):
        return None
    return best


def context_use(items: list[dict], window: int, mark: int = 132000) -> dict:
    """Scored episodes at the context window (ended by the submit-at-limit guard) and past ``mark`` tokens, with their
    pass rates (the 262k rerun's report, #29 / #68)."""
    ok = [r for r in items if r['status'] == 'ok']
    peaks = {id(r): peak_context(r.get('transcript')) for r in ok}

    def part(rs):
        n = sum(1 for r in rs if r.get('score') is not None)
        k = sum(float(r['score']) for r in rs if r.get('score') is not None)
        return {'n': n, 'share': round(n / len(ok), 4) if ok else None, 'passes': k,
                'pass_rate': round(k / n, 4) if n else None}
    known = [r for r in ok if peaks[id(r)] is not None]
    return {'window': window, 'n_scored': len(ok), 'n_peak_known': len(known),
            'at_window': part([r for r in ok if r.get('context_limit')]),
            f'past_{mark // 1000}k': part([r for r in known if peaks[id(r)] > mark]),
            'peak_tokens_median': sorted(peaks[id(r)] for r in known)[len(known) // 2] if known else None}


def extend(run: Path, defaults: dict) -> dict | None:
    """TB-Science fields of summary.json, on top of pi_eval's scorer (audit 1, #29 / #68):
      n_cutoff                        attempts unresolved at the deadline (scored 0 in `score_all`)
      score_all                       pass@1 over every task in the run, cutoffs = 0 (the headline)
      oracle_verdicts                 Pi-path oracle per task (pass / fail / infra / cutoff); every non-pass is excluded
      oracle_fail_under_cap           reference solve > the agent cap (validate.py solve_min): infeasible here
      score_excl_oracle_failures      score_all without the oracle's non-pass tasks
      score_excl_oracle_and_cap       ... and without oracle_fail_under_cap
      ci95_exact                      Clopper-Pearson for each of the three
      excluded_tasks                  deterministic environment defects (DEFAULTS excluded_tasks) unresolved here
      infra_tasks_other               unresolved infra tasks that are not on that list (infra noise)
      score_with_excluded_as_0        score_all with the excluded tasks counted 0 (score_all is over the remaining)
    It also rewrites two top-level fields (audit 2), keeping pi_eval's values under the names noted:
      score_excl_oracle_failures, *_resolved_only, n_items_excl_oracle_failures, oracle_failing_tasks
                                      one definition, the tbsci block's: every non-pass oracle verdict (fail, infra,
                                      cutoff) is excluded; pi_eval's fail-only values -> pi_eval_oracle_fail_only
      ci95                            with one repeat, the exact interval of score_all; the bootstrap -> ci95_bootstrap
      task_set                        the tasks run vs the benchmark's pool
      context                         scored episodes at the context window / past 132k tokens, with pass rates"""
    sf, itf = run / 'summary.json', run / 'items.jsonl'
    if not sf.exists() or not itf.exists():
        return None
    s = json.loads(sf.read_text())
    cfg = json.loads((run / 'config.json').read_text())
    items = jsonl.read(itf)
    per, pend = defaultdict(list), set()
    for r in items:
        if r['status'] == 'pending':
            pend.add(r['item_id'])
        per[r['item_id']].append(float(r['score']) if r['status'] == 'ok' and r.get('score') is not None else
                                 0.0 if r['status'] in ('ok', 'pending') else None)
    means = {t: sum(v) / len(v) for t, v in per.items() if None not in v}
    infra = sorted(t for t, v in per.items() if None in v)
    verd = oracle_verdicts(cfg['oracle_run']) if cfg.get('oracle_run') and cfg.get('agent') != 'oracle' else {}
    bad = {t for t, v in verd.items() if v != 'pass'}
    cap = over_cap(defaults.get('validation', VALIDATION), float(cfg.get('agent_cap_sec') or CAP_SEC))

    def sc(ts):
        v = [means[t] for t in ts]
        return {'score': round(100 * sum(v) / len(v), 2) if v else None, 'n_tasks': len(v),
                'passes': round(sum(v), 2), 'ci95_exact': clopper_pearson(sum(v), len(v))}
    pool = defaults.get('pool', POOL)
    excl = {t: why for t, why in defaults.get('excluded_tasks', {}).items() if t in infra}
    all_ = sc(list(means))
    k0 = sum(means.values())
    s['tbsci'] = {
        'n_cutoff': sum(1 for r in items if r['status'] == 'pending'),
        'infra_tasks': infra,
        'score_all': all_,
        'excluded_tasks': excl,
        'infra_tasks_other': [t for t in infra if t not in excl],
        'score_with_excluded_as_0': {
            'score': round(100 * k0 / (len(means) + len(excl)), 2) if means or excl else None,
            'n_tasks': len(means) + len(excl), 'passes': round(k0, 2),
            'ci95_exact': clopper_pearson(k0, len(means) + len(excl))},
        'score_excl_oracle_failures': sc([t for t in means if t not in bad]),
        'score_excl_oracle_and_cap': sc([t for t in means if t not in bad and t not in cap]),
        'oracle_verdicts': {t: v for t, v in sorted(verd.items()) if v != 'pass'},
        'oracle_fail_under_cap': cap,
        'task_set': {'pool': pool['name'], 'n_pool': pool['n_tasks'], 'n_run': len(per),
                     'subset': len(per) < pool['n_tasks'], 'label': (
                         f"{pool['name']}, all {pool['n_tasks']} tasks" if len(per) >= pool['n_tasks'] else
                         f"{pool['name']}: {len(per)}-task subset of {pool['n_tasks']}")},
        'context': context_use(items, int(cfg['protocol']['context_window'])),
    }
    if verd:   # one oracle-exclusion definition for the whole summary (audit 2: 67 vs 66 on TB-Science)
        if 'pi_eval_oracle_fail_only' not in s:
            s['pi_eval_oracle_fail_only'] = {k: s.get(k) for k in (
                'oracle_failing_tasks', 'score_excl_oracle_failures', 'score_excl_oracle_failures_resolved_only',
                'n_items_excl_oracle_failures')}
        kept = [t for t in means if t not in bad]
        kept_res = [t for t in kept if t not in pend]
        s.update(oracle_failing_tasks=sorted(bad), oracle_exclusion='every non-pass Pi-path oracle verdict '
                 '(fail, infra, cutoff), as tbsci.score_excl_oracle_failures; see tbsci.oracle_verdicts',
                 score_excl_oracle_failures=s['tbsci']['score_excl_oracle_failures']['score'],
                 score_excl_oracle_failures_resolved_only=round(100 * sum(means[t] for t in kept_res) / len(kept_res), 2)
                 if kept_res else None,
                 n_items_excl_oracle_failures=len(kept))
    if s.get('n_repeats') == 1 and all_['n_tasks'] == s.get('n_items'):   # audit 2: bootstrap is wrong at small k
        s.setdefault('ci95_bootstrap', s.get('ci95'))
        s.update(ci95=all_['ci95_exact'], ci95_method='exact Clopper-Pearson (1 repeat); ci95_bootstrap: '
                 '10,000-resample task bootstrap')
    tmp = sf.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=1))
    os.replace(tmp, sf)
    return s


def cli(argv, defaults):
    """pi_eval's CLI with TB-Science's defaults. The context window is pi_eval's --context-window (M3.0: 131072 by
    default; 262144 for the Qwen3.8 calibration runs), recorded in config.json protocol / protocol_overrides."""
    argv = list(argv)
    if '--slots' not in argv:
        argv += ['--slots', '4']
    main(argv, defaults)
    if '--run-id' in argv and ('--score' in argv or '--status' in argv):
        run = RUNS_ROOT / defaults['bench'] / argv[argv.index('--run-id') + 1]
        s = extend(run, defaults)
        if s is not None:
            print(json.dumps({'tbsci': s['tbsci']}, indent=1))


if __name__ == '__main__':
    cli(sys.argv[1:], DEFAULTS)

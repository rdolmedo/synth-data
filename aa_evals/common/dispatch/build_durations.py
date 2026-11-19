#!/usr/bin/env python3
"""Build the per-item duration tables of the #344 dispatcher from stored runs. Read-only on runs.

  python -m aa_evals.common.dispatch.build_durations v1 OUT.json [--replay-dir DIR]
  python -m aa_evals.common.dispatch.build_durations extract-c30 ATTEMPTS.jsonl [--audit ATTEMPTS_C30.jsonl]
  python -m aa_evals.common.dispatch.build_durations v2 OUT.json --attempts ATTEMPTS.jsonl [--fallback durations_v1.json]
         [--exclude-model M ...] [--own-weight W]

v1 (durations_v1.json, #344): the replay's per-attempt records (`replay.py records()`, M3.6 seven models +
GLM-5.3-Flash), GDPval and CritPt re-timed for no compaction (`pi_est.json`). Agentic suites: recorded wall time;
single-request suites: completion tokens / the model's per-request decode speed + 1 s; SciCode: one chain. Value: mean
seconds over that model's attempts.

extract-c30 (#378): one record per finished c30 attempt from the c30 efficiency audit's attempts.jsonl (its load.py:
Pi queues, single-turn lane events, dispatcher events), plus the single-turn runs' own records:
- kept: status ok, full runs. Dropped: infra:* (stale_claim and worker_stopped of the 11-11 memory holds and worker
  kills, endpoint_moved, verifier_*, proxy_outage, no_harbor_result), killed single-turn clients, open, Pi end state
  error, and canaries (an idle server: 0.2-0.5x the full run's time on the same item; separate single-turn run dirs).
- sec_raw: Pi = done wall_sec (claim to verifier end); single-turn = lane end - start (a lost end event: t1 - t0).
- gate_sec (single-turn): the server admission-gate wait the client recorded: Omniscience X-Gate-Queue-Ms (time held
  in the gate's queue; its 503 back-off is not stored); HLE 503 back-off (`gate_wait_s` in the transcript's attempts;
  HLE does not record the held-queue header, so its held time stays in); GDP.pdf and SciCode record none.
  sec = sec_raw - gate_sec.
- censored: cut at the 2 h cap, the true duration is longer than sec. Pi: end state wall_clock_backstop; SciCode: a
  step with finish_reason chain_cap; HLE: request_cap; any single-turn sec_raw >= 7140.

v2 (durations_v2.json, #378): per suite, log seconds = model-item median + attempt noise (sigma); model-item medians =
model effect + item effect + interaction (tau). Fitted by EM with censored attempts as left-truncated normals (Tobit);
the stored value of a model-item is its posterior median (its own attempts, shrunk toward the additive fit by
sigma^2/n vs tau^2), so a model-item whose attempts all hit the cap gets an estimate above the cap, not 7200 s.
mean_factor: c30's capped seconds / the fit's E[min(T, cap)] over the suite (1.0 for a lognormal suite; heavier tails,
e.g. Omniscience, > 1), applied by `predict`.
Fallback from v1 (only where c30 has nothing): an item c30 did not run, or a model-item c30 lacks (GLM's unfinished
items), takes exp(alpha + beta log v1) with (alpha, beta) fitted on the model's items shared by both tables in that
suite (suite-wide when fewer than 10 are shared; beta in [0.2, 1.5]): v1's error grows with the item's length, so one
scale would over-predict long items. v1 values <= 1 s (zero-token placeholders) are skipped; values are floored at 1 s.
A suite c30 lacks keeps v1 unscaled. `--exclude-model` drops a model from c30 and v1 (leave-one-model-out validation).
No eval content: item ids, model keys and seconds only.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from aa_evals.common.dispatch.durations import capped_mean

DEFAULT_REPLAY = '/data/aai/aa_evals/audits/m310_runtime_20261110'
DEFAULT_AUDIT = '/data/aai/aa_evals/audits/c30_efficiency_20261113/dispatch/attempts.jsonl'
RUNS = Path('/data/aai/aa_evals/runs')
HERE = Path(__file__).resolve().parent
CAP_SEC = 7200.0
CAP_ST = 7140.0          # a single-turn lane attempt this long hit the cap
PI_SUITES = {'tb21', 'tbsci_lite', 'gdpval', 'critpt', 'automationbench'}
MIN_SHARED = 10          # model-suite v1 -> v2 scale needs this many shared items, else the suite-wide scale
ALIASES = {'ds-max-m35': 'ds-max'}   # campaign m36_key -> table model key
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ------------------------------------------------------------------------------------------- v1

def build_v1(replay_dir: str) -> dict:
    pi = Path(replay_dir) / 'pi_est.json'
    os.environ['PI_EST'] = str(pi)
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, replay_dir)
    import replay as rp   # noqa: E402  (reads PI_EST at import)
    speed = {m: rp.speed(m) for m in rp.MODELS}
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in rp.MODELS:
        for b in rp.SUITES:
            for item, _rep, sec, _score in rp.records(m, b, speed[m]):
                acc[b][str(item)][m].append(float(sec))
            print(m, b, sum(len(v.get(m, ())) for v in acc[b].values()), flush=True)
    suites = {b: {i: {m: round(st.mean(v), 1) for m, v in sorted(per.items())} for i, per in sorted(items.items())}
              for b, items in sorted(acc.items())}
    return {'name': 'durations_v1', 'created': now(),
            'unit': 'seconds per attempt (scicode: per chain)', 'models': rp.MODELS,
            'speed_tok_s': {m: round(v, 1) for m, v in speed.items()},
            'sources': {'replay_py': {'path': str(Path(replay_dir) / 'replay.py'), 'sha256': sha(Path(replay_dir) / 'replay.py')},
                        'pi_est_json': {'path': str(pi), 'sha256': sha(pi)}},
            'notes': 'GDPval/CritPt: no-compaction (Pi) estimates from Stirrup traces (#344 pi_est.py); '
                     'single-request suites timed from tokens at each model\'s per-request speed (#344 replay).',
            'suites': suites}


# ------------------------------------------------------------------------------------------- c30 extraction

def _jl(p):
    out = []
    try:
        for ln in open(p, errors='replace'):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    return out


def _single_turn_side(model: str) -> dict:
    """(suite, lane key) -> {'gate': seconds, 'cap': bool} from the single-turn runs' own records."""
    side = {}
    run = RUNS / 'omniscience' / f'c30-{model}-omniscience'
    for r in _jl(run / 'items.jsonl'):
        side[('omniscience', f"{r['item_id']}.r{r['repeat']}")] = {
            'gate': float(r.get('gate_queue_ms') or 0) / 1000, 'cap': False}
    for r in _jl(RUNS / 'hle' / f'c30-{model}-hle' / 'items.jsonl'):
        g, cap = 0.0, r.get('finish_reason') == 'request_cap'
        t = Path(str(r.get('transcript') or ''))
        if t.is_file():
            try:
                gen = json.loads(t.read_text()).get('generation') or {}
            except ValueError:
                gen = {}
            g = sum(float(a.get('gate_wait_s') or 0) for a in gen.get('attempts') or [] if isinstance(a, dict))
            cap = cap or gen.get('finish_reason') == 'request_cap'
        side[('hle', f"{r['item_id']}.r{r['repeat']}")] = {'gate': g, 'cap': cap}
    capped = defaultdict(bool)
    for r in _jl(RUNS / 'scicode' / f'c30-{model}-scicode' / 'items.jsonl'):
        capped[f"{r['problem_id']}.r{r['repeat']}"] |= r.get('finish_reason') == 'chain_cap'
    for k, v in capped.items():
        side[('scicode', k)] = {'gate': 0.0, 'cap': v}
    return side


def extract_c30(audit: str) -> tuple[list[dict], dict]:
    rs = _jl(audit)
    meta = {'audit': str(audit), 'audit_sha256': sha(audit), 'n_in': len(rs), 'dropped': collections.Counter(),
            'kept': collections.Counter(), 'censored': collections.Counter(), 'gate_h': collections.Counter(),
            'sec_from_t': 0}
    sides, out = {}, []
    for r in rs:
        if r['status'] != 'ok':
            meta['dropped'][r['status']] += 1
            continue
        if r['stage'] != 'full':      # canaries ran on an idle server, and their single-turn runs are separate dirs
            meta['dropped']['canary'] += 1
            continue
        if r['kind'] == 'pi' and r.get('end_state') == 'error':
            meta['dropped']['pi_end_state_error'] += 1
            continue
        m, s = r['model'], r['suite']
        sec = r.get('sec')
        if sec is None and r.get('t0') and r.get('t1'):
            sec = r['t1'] - r['t0']
            meta['sec_from_t'] += 1
        if sec is None or sec <= 0:
            meta['dropped']['ok_without_sec'] += 1
            continue
        gate, cens = 0.0, False
        if r['kind'] == 'pi':
            cens = r.get('end_state') == 'wall_clock_backstop'
        else:
            if m not in sides:
                sides[m] = _single_turn_side(m)
            sd = sides[m].get((s, r['key']), {})
            gate = min(float(sd.get('gate') or 0), 0.95 * sec)
            cens = bool(sd.get('cap')) or sec >= CAP_ST
        out.append({'model': m, 'stage': r['stage'], 'suite': s, 'kind': r['kind'], 'item': str(r['item']),
                    'key': r['key'], 'rep': r['rep'], 'weight': r['weight'], 't0': r['t0'],
                    'sec_raw': round(sec, 1), 'gate_sec': round(gate, 1), 'sec': round(sec - gate, 1),
                    'censored': cens, 'pred_v1_table': r.get('pred_table'), 'pred_v1': r.get('pred')})
        meta['kept'][f'{m}:{s}'] += 1
        meta['gate_h'][f'{m}:{s}'] += gate / 3600
        if cens:
            meta['censored'][f'{m}:{s}'] += 1
    meta['gate_h'] = {k: round(v, 2) for k, v in meta['gate_h'].items() if v > 0.005}
    meta['n_out'] = len(out)
    return out, meta


# ------------------------------------------------------------------------------------------- v2 fit

def _lam(a: float) -> float:
    """Inverse Mills ratio phi(a) / (1 - Phi(a)), stable for large a."""
    tail = 0.5 * math.erfc(a / math.sqrt(2))
    if tail < 1e-12:
        return a + 1 / a
    return math.exp(-a * a / 2) / math.sqrt(2 * math.pi) / tail


def _additive(z: dict, iters: int = 30):
    """z[(m, i)] -> (a[m], b[i]) least squares of z = a_m + b_i, mean a = 0."""
    ms = sorted({m for m, _ in z})
    its = sorted({i for _, i in z})
    a = {m: 0.0 for m in ms}
    b = {i: 0.0 for i in its}
    by_i, by_m = defaultdict(list), defaultdict(list)
    for (m, i), v in z.items():
        by_i[i].append((m, v))
        by_m[m].append((i, v))
    for _ in range(iters):
        b = {i: st.fmean(v - a[m] for m, v in xs) for i, xs in by_i.items()}
        a = {m: st.fmean(v - b[i] for i, v in xs) for m, xs in by_m.items()}
        c = st.fmean(a.values())
        a = {m: x - c for m, x in a.items()}
        b = {i: x + c for i, x in b.items()}
    return a, b


def fit_suite(obs: dict, iters: int = 80) -> dict:
    """obs[(model, item)] = [(log seconds, censored)]. Returns cell medians (log), sigma, tau, model effects."""
    z = {c: st.fmean(y for y, _ in v) for c, v in obs.items()}
    unc = [(c, y) for c, v in obs.items() for y, k in v if not k]
    within = [y - st.fmean(yy for yy, kk in obs[c] if not kk) for c, y in unc if sum(1 for _, kk in obs[c] if not kk) > 1]
    s2 = max(0.01, st.fmean(x * x for x in within)) if within else 0.25
    t2 = 0.25
    n_obs = sum(len(v) for v in obs.values())
    for _ in range(iters):
        a, b = _additive(z)
        p = {c: a[c[0]] + b[c[1]] for c in obs}
        s = math.sqrt(s2)
        ss, tt, znew = 0.0, 0.0, {}
        for c, v in obs.items():
            mu = z[c]
            e1, e2 = [], []
            for y, k in v:
                if not k:
                    e1.append(y)
                    e2.append(y * y)
                else:
                    al = (y - mu) / s
                    lam = _lam(al)
                    m1 = mu + s * lam
                    var = s2 * max(1e-6, 1 + al * lam - lam * lam)
                    e1.append(m1)
                    e2.append(var + m1 * m1)
            n = len(v)
            prec = n / s2 + 1 / t2
            zc = (sum(e1) / s2 + p[c] / t2) / prec
            vc = 1 / prec
            znew[c] = zc
            ss += sum(q2 - 2 * q1 * zc + zc * zc + vc for q1, q2 in zip(e1, e2))
            tt += (zc - p[c]) ** 2 + vc
        z = znew
        s2 = max(1e-4, ss / n_obs)
        t2 = max(1e-4, tt / len(obs))
    a, _ = _additive(z)
    return {'z': z, 'sigma': math.sqrt(s2), 'tau': math.sqrt(t2), 'a': a}


def _linmap(pairs):
    """[(log v1, log v2)] -> (alpha, beta) of log v2 = alpha + beta log v1 (beta clamped to [0.2, 1.5]), or None."""
    if len(pairs) < MIN_SHARED:
        return None
    mx, my = st.fmean(x for x, _ in pairs), st.fmean(y for _, y in pairs)
    vx = st.fmean((x - mx) ** 2 for x, _ in pairs)
    beta = 1.0 if vx < 1e-6 else min(1.5, max(0.2, st.fmean((x - mx) * (y - my) for x, y in pairs) / vx))
    return my - beta * mx, beta


def build_v2(recs: list[dict], fallback: dict | None, exclude=(), own_weight: float = 0.5, sources=None) -> dict:
    exclude = set(exclude)
    recs = [r for r in recs if r['model'] not in exclude and r.get('stage', 'full') == 'full']
    by_suite = defaultdict(lambda: defaultdict(list))
    cens_sec = defaultdict(list)
    for r in recs:
        y = math.log(max(1.0, float(r['sec'])))
        by_suite[r['suite']][(r['model'], str(r['item']))].append((y, bool(r['censored'])))
        if r['censored']:
            cens_sec[r['suite']].append(float(r['sec_raw']))
    c30_models = sorted({r['model'] for r in recs})
    fb = fallback or {'suites': {}, 'models': []}
    fb_models = [m for m in fb.get('models', []) if m not in exclude]
    models = c30_models + [m for m in fb_models if m not in c30_models]
    pi_cens = [x for s, v in cens_sec.items() if s in PI_SUITES for x in v]
    suites, prov = {}, {}
    lg = {'sigma': {}, 'tau': {}, 'cap_sec': {}, 'mean_factor': {}, 'model_log_effect': {}}
    for s in sorted(set(by_suite) | set(fb['suites'])):
        v1 = fb['suites'].get(s) or {}
        if s not in by_suite:      # a suite c30 lacks: v1 as is
            suites[s] = {i: {m: x for m, x in per.items() if m not in exclude} for i, per in v1.items()}
            suites[s] = {i: per for i, per in suites[s].items() if per}
            prov[s] = {'source': 'v1', 'items': len(suites[s])}
            continue
        f = fit_suite(by_suite[s])
        z = f['z']
        cs = cens_sec.get(s) or []
        cap = st.median(cs) if len(cs) >= 3 else (st.median(pi_cens) if s in PI_SUITES and pi_cens else CAP_SEC)
        cap = round(max(CAP_SEC, cap) if s in PI_SUITES else CAP_SEC, 1)
        # mean calibration: the lognormal's E[min(T, cap)] vs c30's capped seconds over the suite (heavier tails)
        num = sum(math.exp(y) for v in by_suite[s].values() for y, _ in v)
        den = sum(len(v) * capped_mean(z[c], f['sigma'], cap) for c, v in by_suite[s].items())
        mf = min(2.0, max(0.5, num / den)) if den else 1.0
        # v1 -> c30 map per model (else suite-wide): log v2 = alpha + beta log v1 over model-items both have; v1
        # values <= 1 s are v1's no-token placeholders, not durations
        shared = defaultdict(list)
        for (m, i), zz in z.items():
            x = (v1.get(i) or {}).get(m)
            if x and x > 1.0:
                shared[m].append((math.log(x), zz))
        suite_map = _linmap([p for v in shared.values() for p in v]) or (0.0, 1.0)
        mp = {m: (_linmap(shared[m]) or suite_map) for m in models}
        table, n_fb_items, n_fb_cells = {}, 0, 0
        c30_items = {i for _, i in z}
        for (m, i), zz in z.items():
            table.setdefault(i, {})[m] = round(max(1.0, math.exp(zz)), 1)
        for i, per in v1.items():
            new_item = i not in c30_items
            for m, x in per.items():
                if m in exclude or not x or x <= 1.0 or m not in mp:
                    continue
                if new_item or m not in table[i]:
                    al, be = mp[m]
                    table.setdefault(i, {})[m] = round(max(1.0, math.exp(al + be * math.log(x))), 1)
                    n_fb_cells += 1
            n_fb_items += new_item and i in table
        suites[s] = {i: dict(sorted(per.items())) for i, per in sorted(table.items())}
        lg['sigma'][s] = round(f['sigma'], 4)
        lg['tau'][s] = round(f['tau'], 4)
        lg['cap_sec'][s] = cap
        lg['mean_factor'][s] = round(mf, 4)
        lg['model_log_effect'][s] = {m: round(x, 3) for m, x in sorted(f['a'].items())}
        prov[s] = {'source': 'c30', 'c30_items': len(c30_items), 'c30_attempts': sum(len(v) for v in by_suite[s].values()),
                   'c30_censored': sum(1 for v in by_suite[s].values() for _, c in v if c),
                   'v1_fallback_items': n_fb_items, 'v1_fallback_cells': n_fb_cells,
                   'v1_map': {m: [round(a_, 6), round(b_, 6)] for m, (a_, b_) in sorted(mp.items())},
                   'v1_shared_cells': {m: len(v) for m, v in sorted(shared.items())}}
    lg['sigma_default'] = round(st.median(lg['sigma'].values()), 4) if lg['sigma'] else 0.0
    lg['tau_default'] = round(st.median(lg['tau'].values()), 4) if lg['tau'] else 0.0
    lg['cap_default'] = CAP_SEC
    return {'name': 'durations_v2', 'created': now(),
            'unit': 'latent median seconds per attempt (scicode: per chain; gdppdf: per task of its repeats)',
            'models': models, 'aliases': {k_: v for k_, v in ALIASES.items() if v in models},
            'own_weight': own_weight, 'lognormal': lg, 'sources': sources or {},
            'excluded_models': sorted(exclude), 'provenance': prov,
            'notes': 'c30 (Pi harness, no compaction, text-only, 2 h cap) full-run finished non-infra attempts; '
                     'censored lognormal fit (build_durations.py); single-turn values exclude the recorded gate wait; '
                     'v1 mapped per model and suite (log-linear) where c30 has no value.',
            'suites': suites}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('v1')
    p1.add_argument('out')
    p1.add_argument('--replay-dir', default=DEFAULT_REPLAY)
    pe = sub.add_parser('extract-c30')
    pe.add_argument('out')
    pe.add_argument('--audit', default=DEFAULT_AUDIT)
    p2 = sub.add_parser('v2')
    p2.add_argument('out')
    p2.add_argument('--attempts', required=True)
    p2.add_argument('--fallback', default=str(HERE / 'durations_v1.json'))
    p2.add_argument('--exclude-model', action='append', default=[])
    p2.add_argument('--own-weight', type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == 'v1':
        out = build_v1(a.replay_dir)
    elif a.cmd == 'extract-c30':
        recs, meta = extract_c30(a.audit)
        with open(a.out, 'w') as f:
            for r in recs:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')
        Path(a.out).with_suffix('.meta.json').write_text(json.dumps(meta, indent=1, default=dict) + '\n')
        print(len(recs), 'records;', dict(meta['dropped']), 'censored', sum(meta['censored'].values()))
        return
    else:
        recs = _jl(a.attempts)
        fb = json.loads(Path(a.fallback).read_text()) if a.fallback else None
        meta_p = Path(a.attempts).with_suffix('.meta.json')
        src = {'attempts': {'path': str(a.attempts), 'sha256': sha(a.attempts)}}
        if meta_p.exists():
            src['audit'] = {k: json.loads(meta_p.read_text()).get(k) for k in ('audit', 'audit_sha256')}
        if a.fallback:
            src['fallback'] = {'path': str(a.fallback), 'sha256': sha(a.fallback)}
        out = build_v2(recs, fb, a.exclude_model, a.own_weight, src)
    Path(a.out).write_text(json.dumps(out, separators=(',', ':'), sort_keys=False) + '\n')
    print({b: len(v) for b, v in out['suites'].items()}, Path(a.out).stat().st_size, 'bytes')


if __name__ == '__main__':
    main()

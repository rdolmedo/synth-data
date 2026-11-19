"""Answer equivalence and a consensus-reference proxy for CritPt (grading without the authors' server).

There is no public answer key for the 70 challenges. The proxy: sample answers from the strongest models we can
serve, cluster equivalent answers per problem, keep the problems whose majority answer is shared widely enough
(and by more than one model), and score a model by its agreement with that majority. See README "Consensus proxy"
for its biases.

Equivalence follows what the CritPt paper (arXiv 2509.26574, sec. on auto-grading) says about the authors' grader,
as far as it is public: three answer types (numbers, SymPy expressions, Python functions), composite answers compared
element-wise and correct only if every component matches, numbers within a tolerance (the authors use expert,
per-problem tolerances that are not public; exact results default to 12 significant digits), SymPy expressions by
equivalence. We canonicalise each answer by running it in the sandbox:
  - numbers (Python, NumPy, SymPy without free symbols) -> complex value;
  - SymPy expressions -> values at fixed points per symbol name (so different forms of one expression agree);
  - lists / tuples / arrays / matrices -> element-wise; sets -> sorted; dicts -> by key;
  - bool / str -> exact;
  - functions answer(x, ...) whose arguments the template does not define -> values on fixed probe inputs when the
    call succeeds, else "needs_args" (not comparable; such problems are dropped from the proxy).

  python -m aa_evals.critpt.consensus validate                  # the checker on the example's golden answers
  python -m aa_evals.critpt.consensus reference --runs ref-ds41-max --name ds41-max [--min-share 0.6]
  python -m aa_evals.critpt.consensus score --run-id X [--reference ds41-max]
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

RTOL = 1e-4        # between two models' numbers when the problem states no precision (the authors' are not public)
PROBE_VERSION = 3  # 2: arrays as nested lists, intervals, finite sets, physical units, structural fallback
                   # 3: bra-ket inner products evaluated on fixed state vectors (|x*> = conjugate of |x>), #185
# No absolute floor by default: answers in SI units are often 1e-20 or smaller, where any absolute tolerance accepts
# unrelated values (1.04e-21 vs 9.28e-23). An absolute tolerance comes only from a stated number of decimal places.
ATOL = 0.0
EXPR_POINTS = 6

PROBE_PROG = r'''
import cmath, hashlib, inspect, json, math, re, sys
code = sys.stdin.read()
res = {"unfilled": bool(re.search(r"=\s*\.\.\.\s*(#|$)", code, re.M)), "executes": False, "called": False}
NPTS = %(npts)d

def point(name, i):
    h = int(hashlib.sha256(f"{name}/{i}".encode()).hexdigest()[:8], 16)
    return 0.13 + 0.83 * (h / 0xFFFFFFFF)          # in (0.13, 0.96): positive, away from 0 and 1

def num(z):
    z = complex(z)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        return None
    return [z.real, z.imag]

def si_value(v):
    """sympy.physics.units quantities -> their SI value, with the base units set to 1."""
    try:
        from sympy.physics import units as u
        if not v.atoms(u.Quantity):
            return v
        base = [u.meter, u.kilogram, u.second, u.ampere, u.kelvin, u.mole, u.candela]
        # one quantity at a time: convert_to on the whole expression fails inside symbolic powers (c**(2*beta))
        vals = {q: u.convert_to(q, base).subs({b: 1 for b in base}) for q in v.atoms(u.Quantity)}
        return v.xreplace(vals)
    except Exception:
        return v

def state_vec(label):
    """A fixed complex 3-vector per state label; a label ending in '*' is the complex conjugate of its base state
    (CritPt's notation for |phi*>, e.g. Challenge_30)."""
    if label.endswith("*"):
        return [z.conjugate() for z in state_vec(label[:-1])]
    out = []
    for i in range(3):
        h = hashlib.sha256(f"ket/{label}/{i}".encode()).hexdigest()
        out.append(complex(-1 + 2 * int(h[:8], 16) / 0xFFFFFFFF, -1 + 2 * int(h[8:16], 16) / 0xFFFFFFFF))
    return out

def quantum_numeric(v):
    """Inner products <a|b> of sympy.physics.quantum states -> numbers on fixed state vectors, so algebraically equal
    bra-ket answers (a factor distributed over a sum, |<a|b>|**2 vs <a|b><b|a>) get the same values. Only labels are
    used: Bra(Ket(x)) has label |x>, not x, and stays a different state. Anything still holding a state or an
    operator afterwards is returned unchanged (structural fallback)."""
    try:
        import sympy as sp
        from sympy.physics.quantum import InnerProduct
        from sympy.physics.quantum.state import StateBase
        from sympy.physics.quantum.operator import Operator
    except Exception:
        return v
    ips = v.atoms(InnerProduct)
    if not ips:
        return v
    def lab(st):
        return ",".join(str(x) for x in st.label)
    rep = {}
    for ip in ips:
        z = sum(a.conjugate() * b for a, b in zip(state_vec(lab(ip.bra)), state_vec(lab(ip.ket))))
        rep[ip] = sp.Float(z.real, 30) + sp.I * sp.Float(z.imag, 30)
    w = v.xreplace(rep)
    if w.atoms(StateBase) or w.atoms(Operator) or w.atoms(InnerProduct):
        return v
    return w

def structural(v):
    """Fallback for answers with no numeric value: the simplified expression tree."""
    import sympy as sp
    try:
        w = sp.simplify(v)
    except Exception:
        w = v
    try:
        return {"t": "sym", "v": sp.srepr(w)[:4000]}
    except Exception as e:
        return {"t": "expr_err", "err": str(e)[:100]}

def canon(v, depth=0):
    if depth > 6:
        return {"t": "deep"}
    try:
        import sympy as sp
    except Exception:
        sp = None
    try:
        import numpy as np
    except Exception:
        np = None
    if v is None or v is Ellipsis:
        return {"t": "placeholder"}
    if isinstance(v, bool) or (np is not None and isinstance(v, np.bool_)):
        return {"t": "bool", "v": bool(v)}
    if isinstance(v, str):
        return {"t": "str", "v": v.strip().lower()}
    if isinstance(v, (int, float, complex)) or (np is not None and isinstance(v, np.number)):
        return {"t": "num", "v": num(v)}
    if np is not None and isinstance(v, np.ndarray):
        return canon(v.tolist(), depth + 1)            # the same fingerprint as the equivalent nested list
    if sp is not None:
        if isinstance(v, sp.MatrixBase):
            return canon(v.tolist(), depth + 1)
        if isinstance(v, sp.Equality):
            v = v.lhs - v.rhs
        if isinstance(v, sp.Interval):
            return {"t": "interval", "lo": bool(v.left_open), "ro": bool(v.right_open),
                    "v": [canon(v.start, depth + 1), canon(v.end, depth + 1)]}
        if isinstance(v, sp.FiniteSet):
            return canon(set(v.args), depth + 1)
        if isinstance(v, sp.Basic):
            v = si_value(v)
            v = quantum_numeric(v)
            free = sorted(v.free_symbols, key=lambda s: str(s))
            if not free:
                try:
                    return {"t": "num", "v": num(sp.N(v, 30))}
                except Exception:
                    return structural(v)
            vals = []
            for i in range(NPTS):
                try:
                    vals.append(num(sp.N(v.subs({s: point(str(s), i) for s in free}), 30)))
                except Exception:
                    vals.append(None)
            if sum(x is not None for x in vals) < 2:
                return structural(v)                  # no numeric values (operators, traces, bra-kets)
            return {"t": "expr", "syms": [str(s) for s in free], "v": vals, "s": str(v)[:300]}
    if isinstance(v, (set, frozenset)):
        items = [canon(x, depth + 1) for x in v]
        return {"t": "set", "v": sorted(items, key=lambda c: json.dumps(c, sort_keys=True))}
    if isinstance(v, (list, tuple)):
        return {"t": "seq", "v": [canon(x, depth + 1) for x in v]}
    if isinstance(v, dict):
        return {"t": "map", "v": {str(k): canon(x, depth + 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}}
    if callable(v):
        return {"t": "callable"}
    try:
        return {"t": "num", "v": num(v)}
    except Exception:
        return {"t": "other", "s": repr(v)[:200]}

try:
    ns = {"__name__": "answer_module"}
    exec(compile(code, "answer", "exec"), ns)
    res["executes"] = True
    fn = ns.get("answer")
    res["has_answer"] = callable(fn)
    if callable(fn):
        params = list(inspect.signature(fn).parameters)
        missing = [q for q in params if q not in ns]
        if not missing:
            out = fn(*[ns[q] for q in params])
            res["called"] = True
            res["result_type"] = type(out).__name__
            res["fp"] = canon(out)
        else:
            # a function of test-case inputs: probe it on fixed numeric inputs where it accepts them
            res["needs_args"] = missing
            outs = []
            for i in range(3):
                try:
                    kw = {q: (ns[q] if q in ns else point(q, i)) for q in params}
                    outs.append(canon(fn(**kw)))
                except Exception:
                    outs = None
                    break
            res["fp"] = {"t": "probed", "v": outs} if outs else {"t": "needs_args"}
            res["called"] = bool(outs)
        fp = res.get("fp") or {}
        res["result_is_placeholder"] = fp.get("t") == "placeholder" or any(
            isinstance(x, dict) and x.get("t") == "placeholder" for x in fp.get("v") or [] if isinstance(fp.get("v"), list))
except BaseException as e:
    res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
print(json.dumps(res))
''' % {'npts': EXPR_POINTS}


def probe(answer_code: str, run_python, *, timeout: int = 120) -> dict:
    """Run one answer in the sandbox: format fields (placeholder left, executes, defines answer, returns a
    non-placeholder) plus its canonical fingerprint `fp`. Says nothing about correctness by itself."""
    try:
        r = run_python(PROBE_PROG, timeout, stdin=answer_code)
    except subprocess.TimeoutExpired:
        return {'executes': None, 'error': 'timeout'}
    try:
        return json.loads((r.stdout or '').strip().split('\n')[-1])
    except (ValueError, IndexError):
        return {'executes': False, 'error': (r.stderr or '').strip()[-300:]}


# ----------------------------------------------------------------------------------------------- equivalence

def _close(a, b, rtol: float, atol: float) -> bool:
    if a is None or b is None:
        return False
    za, zb = complex(*a), complex(*b)
    return abs(za - zb) <= atol + rtol * max(abs(za), abs(zb))


def comparable(fp: dict | None) -> bool:
    """A fingerprint that can enter a consensus (an actual answer value)."""
    if not fp:
        return False
    t = fp.get('t')
    if t in ('placeholder', 'needs_args', 'callable', 'expr_err', 'other', 'deep'):
        return False
    if t == 'num':
        return fp.get('v') is not None
    if t == 'expr':
        return sum(v is not None for v in fp['v']) >= 2
    if t in ('seq', 'set', 'probed'):
        return bool(fp.get('v')) and all(comparable(x) for x in fp['v'])
    if t == 'interval':
        return all(comparable(x) for x in fp['v'])
    if t == 'map':
        return bool(fp.get('v')) and all(comparable(x) for x in fp['v'].values())
    return t in ('bool', 'str', 'sym')


def _negate(fp: dict) -> dict:
    """The fingerprint of minus the answer (numbers and expressions, recursively); others unchanged."""
    t = fp.get('t')
    if t == 'num' and fp['v'] is not None:
        return {**fp, 'v': [-fp['v'][0], -fp['v'][1]]}
    if t == 'expr':
        return {**fp, 'v': [None if x is None else [-x[0], -x[1]] for x in fp['v']]}
    if t in ('seq', 'probed'):
        return {**fp, 'v': [_negate(x) for x in fp['v']]}
    return fp


def _shifted(v: list) -> list:
    """Expression values relative to the first defined point: invariant under an additive constant."""
    base = next((x for x in v if x is not None), None)
    return [None if x is None or base is None else [x[0] - base[0], x[1] - base[1]] for x in v]


def equivalent(a: dict, b: dict, rtol: float = RTOL, atol: float = ATOL, *, sign_free: bool = False,
               shift_free: bool = False) -> bool:
    """Same answer up to tolerance. Composite answers must match element-wise (the authors' rule).

    sign_free: a vector (a sequence of numbers) also matches its negation ("up to an overall sign").
    shift_free: expressions match up to an additive constant ("omit constants")."""
    kw = {'sign_free': sign_free, 'shift_free': shift_free}
    ta, tb = a.get('t'), b.get('t')
    # a number and a constant expression, or a 1-element list and a scalar, are not merged: the template fixes the type
    if ta != tb:
        return False
    if ta == 'num':
        return _close(a['v'], b['v'], rtol, atol)
    if ta == 'expr':
        va, vb = (_shifted(a['v']), _shifted(b['v'])) if shift_free else (a['v'], b['v'])
        pts = [(x, y) for x, y in zip(va, vb) if x is not None and y is not None]
        # symbols missing from one side are constants there; values are per symbol name, so points line up.
        # Points where both sides evaluate to exactly 0 (e.g. an exp term underflowing) carry no information: at least
        # two informative points are required.
        informative = [(x, y) for x, y in pts if any(x) or any(y)]
        return len(informative) >= 2 and all(_close(x, y, max(rtol, 1e-9), atol) for x, y in pts)
    if ta in ('bool', 'str', 'sym'):
        return a['v'] == b['v']
    if ta == 'interval':
        return (a['lo'], a['ro']) == (b['lo'], b['ro']) and all(
            equivalent(x, y, rtol, atol, **kw) for x, y in zip(a['v'], b['v']))
    if ta in ('seq', 'set', 'probed'):
        if a.get('shape') != b.get('shape') or len(a['v']) != len(b['v']):
            return False
        if ta == 'set':          # sorted by JSON, so near-equal floats may sort differently: match greedily
            rest = list(b['v'])
            for x in a['v']:
                j = next((j for j, y in enumerate(rest) if equivalent(x, y, rtol, atol, **kw)), None)
                if j is None:
                    return False
                rest.pop(j)
            return True
        if all(equivalent(x, y, rtol, atol, **kw) for x, y in zip(a['v'], b['v'])):
            return True
        if sign_free and ta == 'seq' and all(x.get('t') == 'num' for x in a['v']):
            return all(equivalent(x, _negate(y), rtol, atol, **kw) for x, y in zip(a['v'], b['v']))
        return False
    if ta == 'map':
        return a['v'].keys() == b['v'].keys() and all(
            equivalent(a['v'][k], b['v'][k], rtol, atol, **kw) for k in a['v'])
    return False


_WORDNUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
_N = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
_DECIMALS = re.compile(_N + r'[ -](?:decimal places|decimal precision|digits after the decimal point|decimal digits)', re.I)
_SIGFIG = re.compile(_N + r' significant (?:figures|digits|decimal places)', re.I)


def problem_rules(problem_text: str) -> dict:
    """Comparison rules a problem states, as the authors' grader honours them (their exact per-problem tolerances
    are not public): N decimal places -> absolute tolerance 10^-N (two independently rounded values may differ by one
    unit in the last place); N significant figures -> relative tolerance 10^(1-N); "up to an overall sign" ->
    sign_free; "omit constants" -> shift_free. With several stated precisions (one per output), the loosest applies to
    the whole answer."""
    def n(x):
        return int(x) if x.isdigit() else _WORDNUM[x.lower()]
    dec = [n(m.group(1)) for m in _DECIMALS.finditer(problem_text)]
    sig = [n(m.group(1)) for m in _SIGFIG.finditer(problem_text)]
    rules = {'rtol': RTOL, 'atol': ATOL, 'sign_free': False, 'shift_free': False, 'stated': []}
    if dec:
        rules['atol'] = 10.0 ** -min(dec)
        rules['stated'].append(f'{min(dec)} decimal places')
    if sig:
        rules['rtol'] = max(RTOL, 10.0 ** (1 - min(sig)))
        rules['stated'].append(f'{min(sig)} significant figures')
    if re.search(r'up to an? (?:overall )?sign', problem_text, re.I) and not re.search(r'operator up to a sign', problem_text, re.I):
        rules['sign_free'] = True
        rules['stated'].append('up to an overall sign')
    if re.search(r'omit(?:ting)? (?:the )?(?:additive )?constants?', problem_text, re.I):
        rules['shift_free'] = True
        rules['stated'].append('additive constants omitted')
    return rules


def all_problem_rules() -> dict[str, dict]:
    """problem_rules for the 70 challenges and the example items (description + template)."""
    from . import data
    probs = data.load_problems() + data.load_examples()
    return {p.problem_id: problem_rules(p.description + '\n' + p.code_template) for p in probs}


def _eq(a: dict, b: dict, rules: dict | None, rtol: float) -> bool:
    if not rules:
        return equivalent(a, b, rtol)
    return equivalent(a, b, max(rtol, rules['rtol']), rules['atol'], sign_free=rules['sign_free'],
                      shift_free=rules['shift_free'])


def cluster(fps: list[dict], rtol: float = RTOL, rules: dict | None = None) -> list[int]:
    """Greedy clustering into answer classes (label per input; -1 = not comparable). The first member of a class
    is its representative."""
    reps: list[dict] = []
    labels = []
    for fp in fps:
        if not comparable(fp):
            labels.append(-1)
            continue
        j = next((j for j, r in enumerate(reps) if _eq(fp, r, rules, rtol)), None)
        if j is None:
            reps.append(fp)
            j = len(reps) - 1
        labels.append(j)
    return labels


# ----------------------------------------------------------------------------------------------- reference / score

RUNS = Path(os.environ.get('CRITPT_RUNS', '/data/aai/aa_evals/runs/critpt'))   # same as run.RUNS
REFS = RUNS / '_references'
# The default reference (user decision 2026-11-02): DeepSeek V4.1 Flash 1017 (Max), 5 samples per problem, built from
# run DEFAULT_REFERENCE_RUN with `consensus reference --runs ref-ds41-max --name ds41-max`.
DEFAULT_REFERENCE = 'ds41-max'
DEFAULT_REFERENCE_RUN = 'ref-ds41-max'


def run_answers(run_id: str) -> tuple[dict, dict[str, list[dict]]]:
    """config and, per problem, the final rows of every repeat (with their fingerprints)."""
    from .run import final_states, read_jsonl
    run_dir = RUNS / run_id
    cfg = json.loads((run_dir / 'config.json').read_text())
    out: dict[str, list[dict]] = {}
    for (pid, _k), st in sorted(final_states(read_jsonl(run_dir / 'items.jsonl')).items()):
        out.setdefault(pid, []).append(st)
    return cfg, out


def _fp(row: dict) -> dict | None:
    if row['status'] != 'ok':
        return None
    return (row.get('probe') or {}).get('fp')


def build_reference(runs: list[str], *, min_share: float, min_models: int, rtol: float, min_samples: int = 3) -> dict:
    min_models = max(1, min(min_models, len(runs)))
    # (rows probed by an older PROBE_VERSION still work, with the older fingerprints; the reference records them)
    per_model: dict[str, dict[str, list[dict]]] = {}
    for rid in runs:
        cfg, ans = run_answers(rid)
        per_model[rid] = ans
        missing = [r for rows in ans.values() for r in rows if r['status'] == 'ok' and 'probe' not in r]
        if missing:
            raise SystemExit(f'{rid}: {len(missing)} ok rows have no probe; rerun the run command to backfill')
    pids = sorted({p for ans in per_model.values() for p in ans})
    rules = all_problem_rules()
    problems = {}
    for pid in pids:
        samples = [(rid, r) for rid in runs for r in per_model[rid].get(pid, []) if r['status'] != 'infra_error']
        fps = [_fp(r) for _, r in samples]
        labels = cluster(fps, rtol, rules.get(pid))
        n = len(samples)
        counts = Counter(lab for lab in labels if lab >= 0)
        if not counts:
            problems[pid] = {'n': n, 'kept': False, 'why': 'no comparable answers'}
            continue
        top, size = counts.most_common(1)[0]
        models_top = sorted({rid for (rid, _), lab in zip(samples, labels) if lab == top})
        # every model whose own majority is the top class
        model_major = {}
        for rid in runs:
            labs = [lab for (r_, _), lab in zip(samples, labels) if r_ == rid and lab >= 0]
            if labs:
                model_major[rid] = Counter(labs).most_common(1)[0][0]
        agree_models = sorted(r for r, lab in model_major.items() if lab == top)
        share = size / n if n else 0.0
        second = counts.most_common(2)[1][1] / n if len(counts) > 1 else 0.0
        kept = share >= min_share and len(agree_models) >= min_models and n >= min_samples
        rep = fps[labels.index(top)]
        reps = {lab: fps[labels.index(lab)] for lab in counts}
        problems[pid] = {'n': n, 'n_classes': len(counts), 'share': round(share, 3), 'second_share': round(second, 3),
                         'models_in_top': models_top, 'models_majority_top': agree_models, 'kept': kept,
                         'reference_fp': rep, 'top_label': top,
                         'class_fps': {str(k): v for k, v in reps.items()},
                         'members': [{'run': rid, 'repeat': r['repeat'], 'label': lab}
                                     for (rid, r), lab in zip(samples, labels)],
                         'reference_answer': next(r['answer_code'] for (_, r), lab in zip(samples, labels) if lab == top)}
    test_ids = [p for p in pids if p.startswith('Challenge_')]
    return {'name': None, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'runs': runs,
            'params': {'min_share': min_share, 'min_models': min_models, 'rtol': rtol, 'min_samples': min_samples,
                       'probe_version': PROBE_VERSION,
                       # the versions actually in the data: rows probed before PROBE_VERSION need `run --reprobe`
                       'probe_versions_in_data': dict(Counter(
                           str(r.get('probe_version', 1 if 'probe' in r else 0)) for ans in per_model.values()
                           for rows in ans.values() for r in rows if r['status'] == 'ok'))},
            'rules': {p: rules[p] for p in pids if p in rules},
            'n_problems': len(test_ids), 'n_kept': sum(problems[p]['kept'] for p in test_ids),
            'kept': [p for p in test_ids if problems[p]['kept']], 'problems': problems}


def score_run(run_id: str, ref: dict, *, rtol: float, cutoff: bool = False) -> dict:
    """Agreement with the reference on its kept problems: per problem, the share of the run's samples whose answer is
    in the reference's majority class; mean over problems (pass@1-style).

    Leave-one-out for runs that are part of the reference (the reference model itself): each of its samples is
    compared with the majority class of all *other* reference samples of that problem, never with a class it helped
    to form. Problems stay those kept by the full reference.

    Repeats with no final row are pending while the run is live. With `cutoff` (the run deadline passed) they count as
    0 and are reported as n_cutoff. Final infra errors are excluded (never scored as model errors)."""
    cfg, ans = run_answers(run_id)
    in_ref = run_id in ref['runs']
    run_pids = set(cfg['dataset']['problems'])
    repeats = range(cfg['repeats'])
    per_problem, vals, n_cutoff, n_pending = {}, [], 0, 0
    for pid in ref['kept']:
        if pid not in run_pids:
            continue
        have = {r['repeat'] for r in ans.get(pid, [])}
        missing = [k for k in repeats if k not in have]
        rows = [r for r in ans.get(pid, []) if r['status'] != 'infra_error']
        if cutoff:
            n_cutoff += len(missing)
        else:
            n_pending += len(missing)
        if not rows and not (cutoff and missing):
            continue
        P = ref['problems'][pid]
        hits = []
        for r in rows:
            fp = _fp(r)
            if not in_ref:
                hits.append(int(fp is not None and comparable(fp)
                                and _eq(fp, P['reference_fp'], (ref.get('rules') or {}).get(pid), rtol)))
                continue
            mine = [m for m in P['members'] if m['run'] == run_id and m['repeat'] == r['repeat']]
            own = mine[0]['label'] if mine else -1
            others = Counter(m['label'] for m in P['members']
                             if m['label'] >= 0 and not (m['run'] == run_id and m['repeat'] == r['repeat']))
            top = others.most_common(1)[0][0] if others else None
            hits.append(int(own >= 0 and own == top))
        if cutoff:
            hits.extend(0 for _ in missing)
        per_problem[pid] = {'agree': sum(hits), 'n': len(hits), 'n_cutoff': len(missing) if cutoff else 0}
        vals.append(sum(hits) / len(hits))
    score = 100 * sum(vals) / len(vals) if vals else None
    # Every challenge of the run, kept or not (the headline since #185): agreement with the reference's most common
    # answer (its plurality class, whatever its share). A problem whose reference has no comparable answer, and
    # (with cutoff) every missing repeat, counts 0. For a run inside the reference, leave-one-out: each sample against
    # the plurality of the other reference samples.
    all_vals, all_per_problem = [], {}
    for pid in sorted((p for p in run_pids if p.startswith('Challenge_')), key=lambda p: int(p.split('_')[1])):
        P = ref['problems'].get(pid) or {}
        have = {r['repeat'] for r in ans.get(pid, [])}
        missing = [k for k in repeats if k not in have]
        rows = [r for r in ans.get(pid, []) if r['status'] != 'infra_error']
        if not rows and not (cutoff and missing):
            continue
        if not in_ref:
            hits = [int(P.get('reference_fp') is not None and _fp(r) is not None and comparable(_fp(r))
                        and _eq(_fp(r), P['reference_fp'], (ref.get('rules') or {}).get(pid), rtol)) for r in rows]
        else:
            hits = []
            for r in rows:
                members = P.get('members') or []
                mine = [m for m in members if m['run'] == run_id and m['repeat'] == r['repeat']]
                own = mine[0]['label'] if mine else -1
                others = Counter(m['label'] for m in members
                                 if m['label'] >= 0 and not (m['run'] == run_id and m['repeat'] == r['repeat']))
                top = others.most_common(1)[0][0] if others else None
                hits.append(int(own >= 0 and own == top))
        by_repeat = {str(r['repeat']): h for r, h in zip(rows, hits)}
        if cutoff:
            hits.extend(0 for _ in missing)
            by_repeat.update({str(k): 0 for k in missing})
        all_per_problem[pid] = {'agree': sum(hits), 'n': len(hits), 'n_cutoff': len(missing) if cutoff else 0,
                                'kept': bool(P.get('kept')), 'by_repeat': by_repeat}
        all_vals.append(sum(hits) / len(hits))
    ci = None
    if len(vals) >= 2:
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        ci = round(196 * sd / math.sqrt(len(vals)), 2)        # over problems
    return {'run_id': run_id, 'model': cfg['model']['served_name'], 'reference': ref.get('name'),
            'reference_created_at': ref.get('created_at'),
            'reference_runs': ref['runs'], 'leave_one_out': in_ref, 'n_kept': len(ref['kept']),
            'n_scored': len(vals), 'n_cutoff': n_cutoff, 'n_pending': n_pending, 'cutoff_applied': cutoff,
            'agreement': None if score is None else round(score, 2), 'ci95': ci,
            'all_problems': None if not all_vals else {
                'n': len(all_vals), 'leave_one_out': in_ref,
                'agreement_with_plurality': round(100 * sum(all_vals) / len(all_vals), 2),
                'ci95': (round(196 * math.sqrt(sum((v - sum(all_vals) / len(all_vals)) ** 2 for v in all_vals)
                                               / (len(all_vals) - 1)) / math.sqrt(len(all_vals)), 2)
                         if len(all_vals) >= 2 else None),
                'note': 'every challenge of the run, kept or not: agreement with the reference model\'s most common '
                        'answer; no comparable reference answer or a cutoff counts 0'},
            'per_problem': per_problem, 'per_problem_all': all_per_problem, 'params': ref['params']}


def proxy_summary(run_id: str, reference: str = DEFAULT_REFERENCE, *, cutoff: bool = False) -> dict:
    """The proxy block of summary.json (or why it is missing)."""
    path = REFS / f'{reference}.json'
    if not path.exists():
        return {'reference': reference, 'status': 'reference not built yet',
                'how': f'run {DEFAULT_REFERENCE_RUN} (DeepSeek V4.1 Flash 1017 Max, 5 repeats), then '
                       f'consensus reference --runs {DEFAULT_REFERENCE_RUN} --name {reference}'}
    ref = json.loads(path.read_text())
    try:
        res = score_run(run_id, ref, rtol=ref['params']['rtol'], cutoff=cutoff)
    except SystemExit as e:
        return {'reference': reference, 'status': f'not scored: {e}'}
    (RUNS / run_id / f'proxy_{reference}.json').write_text(json.dumps(res, indent=1))   # per-problem results
    res.pop('per_problem')
    res.pop('per_problem_all')
    res['status'] = 'ok'
    return res


# ----------------------------------------------------------------------------------------------- validation

def _variants(golden_code: str) -> list[tuple[str, str, bool]]:
    """(label, answer expression code, should_match) variants of a golden SymPy assignment."""
    import ast
    name = ast.parse(golden_code).body[0].targets[0].id
    expr = golden_code.split('=', 1)[1].strip()
    return [
        ('golden', expr, True),
        ('simplified', f'sp.simplify({expr})', True),
        ('together', f'sp.together({expr})', True),
        ('factored', f'sp.factor(sp.together({expr}))', True),
        ('float-coefficients', f'sp.nsimplify(0) + sp.N({expr}, 15)', True),
        ('1%-perturbed', f'({expr}) * sp.Rational(101, 100)', False),
        ('first-order-only', f'sp.series({expr}, p, 0, 2).removeO()', False),
        ('one-minus', f'1 - ({expr})', False),
    ], name


def validate() -> int:
    """Check the equivalence on the example challenge and its checkpoints: equivalent rewritings of each golden
    answer must match it, perturbed ones must not, and the golden answers must not match each other."""
    from . import data
    exs = data.load_examples()
    run_py = lambda prog, timeout, stdin=None: subprocess.run(  # noqa: E731 - trusted (authors') code only
        [sys.executable, '-I', '-c', prog], capture_output=True, text=True, timeout=timeout, input=stdin)
    ok = True
    goldens = {}
    for e in exs:
        variants, name = _variants(e.golden_code)
        fps = {}
        for label, expr, _ in variants:
            code = e.code_template.replace(f'{name} = ...', f'{name} = {expr}')
            r = probe(code, run_py)
            fps[label] = r.get('fp')
        gold = fps['golden']
        goldens[e.problem_id] = gold
        for label, _, should in variants:
            got = bool(fps[label]) and comparable(fps[label]) and equivalent(fps[label], gold)
            mark = 'ok ' if got == should else 'BAD'
            ok &= got == should
            print(f'{mark} {e.problem_id:34s} {label:22s} match={got} expected={should}')
    ids = list(goldens)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            same = equivalent(goldens[ids[i]], goldens[ids[j]])
            print(f'{"ok " if not same else "BAD"} {ids[i]} vs {ids[j]}: match={same} expected=False')
            ok &= not same
    # numbers, lists, sets: tolerance and element-wise rules
    num = lambda x: {'t': 'num', 'v': [x, 0.0]}  # noqa: E731
    cases = [('1.0 vs 1.00001', num(1.0), num(1.00001), True), ('1.0 vs 1.001', num(1.0), num(1.001), False),
             ('seq equal', {'t': 'seq', 'v': [num(1), num(2)]}, {'t': 'seq', 'v': [num(1), num(2)]}, True),
             ('seq one wrong', {'t': 'seq', 'v': [num(1), num(2)]}, {'t': 'seq', 'v': [num(1), num(3)]}, False),
             ('seq order', {'t': 'seq', 'v': [num(1), num(2)]}, {'t': 'seq', 'v': [num(2), num(1)]}, False),
             ('set order', {'t': 'set', 'v': [num(1), num(2)]}, {'t': 'set', 'v': [num(2), num(1)]}, True),
             ('scalar vs 1-list', num(1), {'t': 'seq', 'v': [num(1)]}, False)]
    for label, a, b, should in cases:
        got = equivalent(a, b)
        ok &= got == should
        print(f'{"ok " if got == should else "BAD"} {label:22s} match={got} expected={should}')
    # Stated rules (problem_rules), on the problems' own wording.
    vec = lambda *xs: {'t': 'seq', 'v': [num(x) for x in xs]}
    ex = lambda off: {'t': 'expr', 'v': [[x + off, 0.0] for x in (1.0, 2.5, 3.7)]}
    iv = lambda lo, hi, ro: {'t': 'interval', 'lo': True, 'ro': ro, 'v': [num(lo), num(hi)]}
    ruled = [('3 decimals, rounded', 'to three decimal places', num(0.658), num(0.6584789), True),
             ('3 decimals, off', 'to three decimal places', num(0.658), num(0.6605), False),
             ('3 sig figs', 'to three significant figures', num(-2.71e-11), num(-2.7149e-11), True),
             ('overall sign', 'up to an overall sign', {'t': 'seq', 'v': [vec(1, -1, 0), num(1.19)]},
              {'t': 'seq', 'v': [vec(-1, 1, 0), num(1.19)]}, True),
             ('sign: scalar fixed', 'up to an overall sign', {'t': 'seq', 'v': [vec(1, -1, 0), num(1.19)]},
              {'t': 'seq', 'v': [vec(-1, 1, 0), num(-1.19)]}, False),
             ('omit constants', 'omit constants', ex(0.0), ex(0.3), True),
             ('no rule: constant', 'nothing stated', ex(0.0), ex(0.3), False),
             ('interval open/closed', 'nothing stated', iv(0, 1.2, True), iv(0, 1.2, False), False),
             ('interval equal', 'nothing stated', iv(0, 1.2, True), iv(0, 1.2, True), True)]
    for label, text, a, b, should in ruled:
        got = _eq(a, b, problem_rules(text), RTOL)
        ok &= got == should
        print(f'{"ok " if got == should else "BAD"} {label:22s} match={got} expected={should}')
    print('VALIDATION', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('validate')
    r = sub.add_parser('reference')
    r.add_argument('--runs', required=True, help='comma-separated run ids (the reference models\' runs)')
    r.add_argument('--name', required=True)
    r.add_argument('--min-share', type=float, default=0.6, help='majority class share of all samples to keep a problem')
    r.add_argument('--min-samples', type=int, default=3, help='samples (ok or model_error) a problem needs to be kept')
    r.add_argument('--min-models', type=int, default=1, help='models whose own majority is that class (capped at the '
                   'number of reference runs; use 2 with two reference models)')
    r.add_argument('--rtol', type=float, default=RTOL)
    s = sub.add_parser('score')
    s.add_argument('--run-id', required=True)
    s.add_argument('--reference', default=DEFAULT_REFERENCE)
    s.add_argument('--rtol', type=float, default=RTOL)
    s.add_argument('--cutoff', action='store_true', help='the run deadline passed: missing repeats count as 0')
    a = ap.parse_args()
    if a.cmd == 'validate':
        sys.exit(validate())
    if a.cmd == 'reference':
        ref = build_reference(a.runs.split(','), min_share=a.min_share, min_models=a.min_models, rtol=a.rtol,
                              min_samples=a.min_samples)
        ref['name'] = a.name
        REFS.mkdir(parents=True, exist_ok=True)
        (REFS / f'{a.name}.json').write_text(json.dumps(ref, indent=1))
        print(json.dumps({k: v for k, v in ref.items() if k != 'problems'}, indent=1))
        return
    ref = json.loads((REFS / f'{a.reference}.json').read_text())
    res = score_run(a.run_id, ref, rtol=a.rtol, cutoff=a.cutoff)
    (RUNS / a.run_id / f'proxy_{a.reference}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_problem'}, indent=1))


if __name__ == '__main__':
    main()

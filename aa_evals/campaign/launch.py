#!/usr/bin/env python3
"""Campaign launcher: one eval campaign (every model x every suite of a plan) from a campaign config (#378).

Generalizes c30's launcher (aa-evals-coord/c30/ops/c30_launch.py) and builds in the c30 audit's checklist
(aa_evals/README.md "Cost and storage rules"):
- servers are submitted largest first (4-GPU, then 2-GPU, then 1-GPU), and the smaller ones only after the larger
  ones run or `--stagger-min` passes, so multi-GPU servers find whole nodes (c30: DS and GLM waited 5+ h);
- serving RAM comes from the aa-serving spec (no override), budgets from the config (each server's knee);
- each model's dispatcher grows its pi workers (pi_eval launches lanes with 25% of the budget) and releases the model's
  own server `release_server_after` s after all its lanes are done;
- Pi workers get pi_eval's lean shapes and every CPU job the bid ladder (no shape or bid flags are passed);
- `promote` starts a model's full run as soon as its own canary is clean and its decode speed matches the reference
  (c30: canaries idled 5.7 GPU-h waiting; the 4B server ran 4.5 h at half speed), and `watch` does it for all models.

  python -m aa_evals.campaign.launch CONFIG show    [MODEL ...] [--stage canary|full]   # dry run: every command
  python -m aa_evals.campaign.launch CONFIG servers [MODEL ...] [--stagger-min 10] [--dry-run]
  python -m aa_evals.campaign.launch CONFIG launch  MODEL [--stage canary|full] [--only SUITES]
  python -m aa_evals.campaign.launch CONFIG check   MODEL          # canary done/infra + decode speed vs reference
  python -m aa_evals.campaign.launch CONFIG promote MODEL [--force]  # check, then launch the full stage
  python -m aa_evals.campaign.launch CONFIG watch   [MODEL ...] [--every 300]   # promote each model once it passes
  python -m aa_evals.campaign.launch CONFIG shadow  MODEL --stage full   # tbsched shadow next to a launched run

CONFIG: JSON like example_c30.json. `{campaign}` in strings is the campaign name. Runs: <name>-<model>-<suite>
[-canary]; dispatch dirs <dispatch_root>/<name>-<model>[-canary] (never reused); launch logs in
<dispatch_root>/<name>-launch/. Everything it writes is on /data.

Capped protocol (off by default): `"capped": "<capped_protocol.json>"` in the config, or per model (the model's wins;
`"capped": null` opts a model out), gives every suite the response budget of that file (`capped_budgets`): Pi suites
get `--harbor-env PI_RESPONSE_BUDGET=<json>`, the single-turn submitters PI_RESPONSE_BUDGET in their environment,
which they forward into their condor jobs (aa_evals/common/response_budget_env.py). With it, the model-time caps
(`suites.wall_cap_sec`: Pi wall clock, SciCode chain cap, HLE/GDP.pdf request caps, Omniscience timeout) default to
12 h, a hang backstop, instead of `cap_sec` (#531 B6); the Pi environment clock stays `cap_sec`. Unset: every
command unchanged.

Sampling (off by default): `"request_overrides": {"temperature": 0.6}` (keys: REQUEST_FLAGS, temperature and top_p),
at campaign level or per model (the model's wins; `null` opts a model out), replaces those sampling values in every suite's requests, over the registry's [request] overrides: Pi suites get
pi_eval `--request-overrides <json>` (recorded in the run config as request.overrides), single-turn suites their
client's `--temperature` / `--top-p` (recorded in the run config's generation settings). Unset: every command unchanged.
The default sampling stays T 1.0 / top_p 0.95 (spec generation config + protocol). T=0.6 was tested on the Snowball
10.73B SFT (11-19, #480 comments 6996073899 and 7015700314) and was not better: GDPval −68 Elo [−121, −15], the other
suites within noise. Keep T=1.0 for SFT evals unless a new test says otherwise; the 0.6 above is only a syntax example.

AAI-open version (#481, docs/aai_open/VERSIONS.md): `version()` checks each model's commands against the current
version's pins (aa_evals/common/aai_open.py: item plan, context window, caps, TB2.1 bundle + offline verifier
toolchain, runtime caches, SciCode dataset, capped-protocol file, no request_overrides). Every suite command runs
with $AAI_OPEN_VERSION = 'v0.1', or 'v0.1+custom' when anything differs; each client records it in its run config
(`aai_open`); `show`, `build()` and the launch log's suite events carry the label and the campaign-level
deviations (the run config records the label). Nothing else changes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from aa_evals.common.dispatch.durations import Durations
from aa_evals.common import aai_open, serving   # serving: the registry reader (tbshared/serving.py, ALIASES copy)
from tb_agents import response_budget

_DUR = Durations()

HARBOR = '/opt/aai/venvs/harbor/bin/python'
STIRRUP = '/data/aai/aa_evals/venvs/critpt-stirrup021/bin/python'
PDFPY = '/data/aai/aa_evals/envs/gdppdf/bin/python'
REG = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry (serve_teacher too)
SUITES = ['tb21', 'tbsci_lite', 'gdpval', 'critpt', 'automationbench', 'scicode', 'hle', 'omniscience', 'gdppdf']
PI_SUITES = {'tb21', 'tbsci_lite', 'gdpval', 'critpt', 'automationbench'}
# Named suite sets: a model's `suites` is a list of SUITES or one of these names. "lean" = the six suites the M5
# Snowball SFT evals kept (sb2-*, 11-17: CritPt, HLE and TB-Sci-lite dropped, the most expensive suites without signal
# at that scale; uval/ucab 11-18/19 the same); "full" = every suite of the plan (c30 cheap_full_v1).
SUITE_SETS = {'lean': ['tb21', 'automationbench', 'gdpval', 'scicode', 'omniscience', 'gdppdf'], 'full': list(SUITES)}
DURATION_KEYS = set(_DUR.table['models']) | set(_DUR.table.get('aliases') or {})   # the default table's model keys
BUDGET_ENV = 'PI_RESPONSE_BUDGET'
# suite -> the benchmark name tb_agents.response_budget.with_benchmark takes (its NOTICE_EXTRA keys; the single-turn
# runners pass the same names: omniscience, hle, scicode)
BENCHMARK = {s: s for s in SUITES}
DECODE_MIN_RATIO = 0.5     # canary decode below this share of the reference: do not promote (c30 4B misconfig: ~0.45)
INFRA_MAX_SHARE = 0.05     # canary infra outcomes above this share: do not promote
# request_overrides key -> the single-turn clients' flag (scicode, hle, omniscience, gdppdf all take both); pi_eval
# takes any JSON, but a key applies to every suite only if every client takes it
REQUEST_FLAGS = {'temperature': '--temperature', 'top_p': '--top-p'}


# ------------------------------------------------------------------------------------------------ config
class Campaign:
    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_text()
        cfg = json.loads(raw)
        self.name = cfg['name']
        cfg = json.loads(raw.replace('{campaign}', self.name))
        self.cfg = cfg
        self.models = cfg['models']
        self.s = cfg.get('suites') or {}
        self.plan = Path(cfg['plan'])
        self.code = Path(cfg.get('code') or '.')
        self.dispatch_root = Path(cfg.get('dispatch_root') or '/data/aai/aa_evals/dispatch')
        self.logdir = self.dispatch_root / f'{self.name}-launch'

    def run_id(self, model, suite, stage):
        return f'{self.name}-{model}-{suite}' + ('-canary' if stage == 'canary' else '')

    def dispatch_dir(self, model, stage):
        return self.dispatch_root / (f'{self.name}-{model}' + ('-canary' if stage == 'canary' else ''))

    def log(self, **kw):
        self.logdir.mkdir(parents=True, exist_ok=True)
        with (self.logdir / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                                **kw}) + '\n')


def model_suites(C: Campaign, model: str) -> list[str]:
    """The suites MODEL runs, in SUITES order: all of them, its `suites` list (e.g. a re-run of some suites under
    a new model key, with new run ids and its own dispatcher) or a SUITE_SETS name ("lean", "full")."""
    x = C.models[model].get('suites')
    if isinstance(x, str):
        if x not in SUITE_SETS:
            raise SystemExit(f'model {model}: suites {x!r} must be a list or one of {sorted(SUITE_SETS)}')
        x = SUITE_SETS[x]
    if x is None:
        return list(SUITES)
    bad = [y for y in x if y not in SUITES]
    if bad or not x:
        raise SystemExit(f'model {model}: suites {x!r} must be a non-empty subset of {SUITES} (unknown: {bad})')
    return [y for y in SUITES if y in x]


def plan_suites(C: Campaign, stage: str) -> dict:
    p = json.loads(C.plan.read_text())
    canary = (C.s.get('canary_items') or {})
    out = {}
    for s in SUITES:
        x = p['suites'][s]
        ids = [i['id'] for i in x['items']]
        reps = {i['repeats'] for i in x['items']}
        assert len(reps) == 1, (s, reps)
        rep = reps.pop()
        if stage == 'canary':
            n = canary.get(s, 3)
            ids, rep = (ids[:n] if n else ids), 1
        out[s] = {'ids': ids, 'repeats': rep}
    return out


def ids_file(C: Campaign, suite: str, ids: list[str], stage: str) -> tuple[Path, str]:
    body = ''.join(i + '\n' for i in ids)
    f = C.dispatch_root / f'{C.name}-ids' / f'{suite}.{stage}.{hashlib.sha256(body.encode()).hexdigest()[:12]}.txt'
    return f, body


# ------------------------------------------------------------------------------------------------ capped protocol
MAX_WALL_CAP_SEC = 43200      # pi_eval's limit for --agent-wall-cap-sec (2 x the harness's six hours)
CAPPED_WALL_CAP_SEC = 43200   # model-time caps under the capped protocol when suites.wall_cap_sec is unset (#531 B6)


def capped_file(C: Campaign, model: str) -> Path | None:
    """The model's capped protocol file: its own `capped` key when present (null: off), else the campaign's."""
    m = C.models[model]
    p = m['capped'] if 'capped' in m else C.cfg.get('capped')
    return Path(p) if p else None


def capped_budgets(path: Path, suites=SUITES) -> dict[str, dict]:
    """suite -> PI_RESPONSE_BUDGET from a capped_protocol.json (`suites`: {warn, allowance, kind: pi|single,
    near_window}, Super-native tokens): warn + allowance, counted in Super tokens, near_window for single-turn suites
    that set it, the benchmark's notice sentence (with_benchmark). Every suite needs an entry with a positive integer
    warn and allowance and the suite's own kind; each budget must pass response_budget.settings."""
    rows = json.loads(Path(path).read_text()).get('suites') or {}
    out = {}
    for s in suites:
        x = rows.get(s)
        if not isinstance(x, dict):
            raise SystemExit(f'capped protocol {path}: no entry for suite {s}')
        warn, allow = x.get('warn'), x.get('allowance')
        if not all(type(v) is int and v > 0 for v in (warn, allow)):
            raise SystemExit(f'capped protocol {path}: {s} needs positive integer warn and allowance '
                             f'(got warn={warn!r}, allowance={allow!r})')
        kind = 'pi' if s in PI_SUITES else 'single'
        if x.get('kind') != kind:
            raise SystemExit(f'capped protocol {path}: {s} is a {kind} suite, the file says kind={x.get("kind")!r}')
        b = {'warn_after_response_tokens': warn, 'finish_response_tokens': allow, 'total_response_tokens': warn + allow,
             'token_counting': 'super'}
        if kind == 'single' and x.get('near_window'):
            b['near_window'] = True
        b = response_budget.with_benchmark(b, BENCHMARK[s])
        try:
            response_budget.settings({BUDGET_ENV: budget_json(b)})
        except (AssertionError, KeyError, ValueError) as e:
            raise SystemExit(f'capped protocol {path}: {s}: response_budget.settings rejects {b}: {e!r}')
        out[s] = b
    return out


def budget_json(b: dict) -> str:
    return json.dumps(b, separators=(',', ':'))


def capped(C: Campaign, model: str) -> dict | None:
    """{file, sha256, budgets} of the model's capped protocol; None when it is off."""
    p = capped_file(C, model)
    if p is None:
        return None
    return {'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'budgets': capped_budgets(p)}


def request_overrides(C: Campaign, model: str) -> dict:
    """The model's `request_overrides` (sampling for every suite's requests), checked; {} when unset. As `capped`:
    the model's key wins over the campaign's, and `"request_overrides": null` on a model opts it out."""
    m = C.models[model]
    ro = m['request_overrides'] if 'request_overrides' in m else C.cfg.get('request_overrides')
    if ro is None:
        return {}
    if not isinstance(ro, dict):
        raise SystemExit(f'model {model}: request_overrides must be an object, got {ro!r}')
    bad = sorted(k for k in ro if k not in REQUEST_FLAGS)
    if bad:
        raise SystemExit(f'model {model}: request_overrides {bad} not supported: every suite takes only '
                         f'{sorted(REQUEST_FLAGS)}')
    for k, v in ro.items():
        num = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
        if not num or (k == 'temperature' and v < 0) or (k == 'top_p' and not 0 < v <= 1):
            raise SystemExit(f'model {model}: request_overrides.{k} must be a finite number '
                             f'({"0 < top_p <= 1" if k == "top_p" else "temperature >= 0"}), got {v!r}')
    return dict(ro)


# ------------------------------------------------------------------------------------------------ AAI-open version
def _pin(v) -> str | None:
    """The sha256 of a [path, sha256] suites pin (None when malformed or unset)."""
    return v[1] if isinstance(v, list) and len(v) == 2 else None


def version(C: Campaign, model: str, stage: str) -> dict:
    """{version, deviations}: the model's commands against the current AAI-open version's pins (aai_open.py,
    docs/aai_open/VERSIONS.md). Any deviation makes the label '<version>+custom'. Per-model settings that are part of
    a model's own row (registry, effort, checkpoint, budget, worker_profile) are not deviations; a lean model `suites`
    list is not either (the aggregate reports the missing slots). A canary is labelled like its full run."""
    A, s, dev = aai_open, C.s, []
    sha = hashlib.sha256(C.plan.read_bytes()).hexdigest()
    if sha != A.PLAN_SHA256:
        dev.append(f'plan {C.plan.name} sha256 {sha[:12]} (v0.1: cheap_full_v1 {A.PLAN_SHA256[:12]})')
    ctx, cap = int(s.get('context_window', 131072)), int(s.get('cap_sec', 7200))
    if ctx != A.CONTEXT_WINDOW:
        dev.append(f'suites.context_window {ctx} (v0.1: {A.CONTEXT_WINDOW})')
    if cap != A.CAP_SEC:
        dev.append(f'suites.cap_sec {cap} (v0.1: {A.CAP_SEC})')
    cp = capped(C, model)
    if cp and cp['sha256'] != A.CAPPED_SHA256:
        dev.append(f"capped protocol {cp['file']} sha256 {cp['sha256'][:12]} (v0.1: {A.CAPPED_SHA256[:12]})")
    w, want = s.get('wall_cap_sec'), (A.CAPPED_WALL_CAP_SEC if cp else A.CAP_SEC)
    if w is not None and int(w) != want:
        dev.append(f"suites.wall_cap_sec {w} (v0.1: {want} {'capped' if cp else 'uncapped'})")
    suites = set(model_suites(C, model))
    for key, suite in (('tb_bundle', 'tb21'), ('tb_cache', 'tb21'), ('tb_toolchain_bundle', 'tb21'),
                       ('lite_cache', 'tbsci_lite')):
        if suite not in suites:
            continue
        got, pin = s.get(key), A.SUITE_PINS[key]
        if (got != pin) if key == 'tb_toolchain_bundle' else (_pin(got) != pin):
            dev.append(f'suites.{key} {str(got)[:80]} (v0.1: {str(pin)[:80]})')
    for key, suite in (('scicode_dataset', 'scicode'), ('lite_pool', 'tbsci_lite')):
        if suite in suites and s.get(key, A.SUITE_DEFAULTS[key]) != A.SUITE_DEFAULTS[key]:
            dev.append(f'suites.{key} {s.get(key)} (v0.1: {A.SUITE_DEFAULTS[key]})')
    ro = request_overrides(C, model)
    if ro:
        dev.append(f'request_overrides {json.dumps(ro, separators=(",", ":"))} (v0.1: the registry sampling)')
    g = C.cfg.get('grading') or {}   # grade.py's judge and GDPval references (its DEFAULTS when unset)
    arm = (g.get('judge') or {}).get('arm', A.JUDGE_ARM)
    if arm != A.JUDGE_ARM:
        dev.append(f'grading.judge.arm {arm} (v0.1: {A.JUDGE_ARM})')
    refs = g.get('gdpval_refs', A.GDPVAL_REFS)
    if 'gdpval' in suites and list(refs or []) != A.GDPVAL_REFS:
        dev.append(f'grading.gdpval_refs {refs} (v0.1: {A.GDPVAL_REFS})')
    return {'version': A.label(None, dev), 'deviations': dev}


# ------------------------------------------------------------------------------------------------ commands
# per-model worker_profile (#531 A/B): "slow" = a slow-served model (e.g. a Snowball SFT on the eval mix; gate: a prior
# TB2.1 run had tool share env_sec/(env_sec+model_sec) <= 0.25, vs ~0.5 at Super speed) runs these suites on the
# pipeline.json shape named here; only suites measured so far are listed (TB2.1: 6 vs 10 CPUs at 32 slots, pass within
# noise, tool time per request +7% [-11%, +30%], worker cost -40%). Applies to new runs (an existing run keeps its
# stored shape). Unset: every command as before.
SLOW_SHAPES = {'tb21': 'tb21_slow'}
WORKER_PROFILES = (None, 'slow')


def slow_shape(C: Campaign, model: str, suite: str) -> list[str]:
    """pi_eval --shape-key for the model's worker_profile (recorded in the run's config worker.shape_check)."""
    m = C.models[model]
    profile = m['worker_profile'] if 'worker_profile' in m else C.cfg.get('worker_profile')   # null opts a model out
    if profile not in WORKER_PROFILES:
        raise SystemExit(f'model {model}: worker_profile must be one of {WORKER_PROFILES}, got {profile!r}')
    return ['--shape-key', SLOW_SHAPES[suite]] if profile == 'slow' and suite in SLOW_SHAPES else []


def tb_toolchain(s: dict) -> list[str]:
    """suites.tb_toolchain_bundle = [stem, sha256 of the tb_bundle it was validated with] -> tb21's
    --verifier-toolchain-bundle. The pin keeps a validated alias from silently applying to another bundle (the staging
    sha check covers test.sh only, not environments or other test files)."""
    v = s.get('tb_toolchain_bundle')
    if not v:
        return []
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, str) and x for x in v)):
        raise SystemExit('suites.tb_toolchain_bundle: [toolchain bundle stem, sha256 of the tb_bundle it was validated with]')
    if v[1] != s['tb_bundle'][1]:
        raise SystemExit(f'suites.tb_toolchain_bundle {v[0]} was validated for tb_bundle sha256 {v[1][:12]}, '
                         f'not {s["tb_bundle"][1][:12]}: re-run the paired oracle gate (#531 B5) before pinning it')
    return ['--verifier-toolchain-bundle', v[0]]


def suite_cmd(C: Campaign, model: str, suite: str, sel: dict, stage: str, rb: dict | None = None) -> dict:
    """The suite's command. `rb`, a response budget (capped protocol): Pi suites get it as pi_eval --harbor-env,
    single-turn suites as PI_RESPONSE_BUDGET in the submitter's environment (`env`), forwarded into its condor job.
    The model's request_overrides: Pi suites get pi_eval --request-overrides, single-turn suites --temperature /
    --top-p (each client's run.py part of the command)."""
    m = C.models[model]
    s = C.s
    ctx, cap = int(s.get('context_window', 131072)), int(s.get('cap_sec', 7200))
    # `cap_sec` is the Pi environment clock (model time excluded, as in the Pi protocol) and, unless `wall_cap_sec`
    # is set, every model-time cap too (#344's 2 h). `wall_cap_sec` sets those apart: the Pi agent's wall-clock cap,
    # SciCode's chain cap, HLE's request cap and per-request timeout, Omniscience's request timeout and GDP.pdf's
    # request cap. A model-time cap scales with the server's decode speed per sequence, so on a slow server it ends
    # attempts before the capped protocol's token budget can (m5l 3B, 11-16: 34% of CritPt at 7200 s).
    # Default per protocol (#531 B6): under the capped protocol the token budgets end attempts and a model-time cap is
    # a hang backstop only (#406 design), so CAPPED_WALL_CAP_SEC (m5l's 12 h, after the 2-h cap cut 1B/3B attempts at
    # 72-90% of the warning and forced ~23 B200-h of re-runs); without it (the uncapped c30 leaderboard config) the
    # 2-h cap stays part of the protocol (#344, -49% GPU-h). An explicit suites.wall_cap_sec wins either way.
    w = s.get('wall_cap_sec')   # null = the default, as elsewhere in this config
    wall = int(w if w is not None else (CAPPED_WALL_CAP_SEC if rb is not None else cap))
    if wall < cap:
        raise SystemExit(f'suites.wall_cap_sec {wall} is below cap_sec {cap}')
    if wall > MAX_WALL_CAP_SEC:   # pi_eval refuses it at launch (--agent-wall-cap-sec <= 2 x the six-hour limit)
        raise SystemExit(f'suites.wall_cap_sec {wall} is above {MAX_WALL_CAP_SEC} (pi_eval: 2 x the six-hour limit)')
    run_id = C.run_id(model, suite, stage)
    spec = f"serving:{m['registry']}"
    disp = str(C.dispatch_dir(model, stage))
    eff = m.get('effort')
    effort = ['--reasoning-effort', eff] if eff else []
    ids, rep, budget = sel['ids'], sel['repeats'], int(m['budget'])
    n_att = len(ids) * rep
    files = {}
    # Pi lanes: no worker count, shape or bid flags. pi_eval launches 25% of the budget under a growing dispatcher,
    # with its lean per-benchmark shapes and the CPU bid ladder; --max-in-flight only caps a lane at the budget.
    pi_tail = ['--context-window', str(ctx), '--model-images', 'off', '--dispatch', disp, '--agent-wall-cap-sec',
               str(wall), '--agent-cap-sec', str(cap), '--no-agent-internet', '--max-in-flight', str(min(n_att, budget))]
    if rb is not None and suite in PI_SUITES:
        pi_tail += ['--harbor-env', f'{BUDGET_ENV}={budget_json(rb)}']
    ro = request_overrides(C, model)
    if ro and suite in PI_SUITES:
        pi_tail += ['--request-overrides', json.dumps(ro, separators=(',', ':'))]
    if suite == 'tb21':
        argv = [HARBOR, '-m', 'aa_evals.tb21.run', '--model', spec, '--run-id', run_id, *effort,
                '--attempts', str(rep), '--include', ','.join(ids), *pi_tail,
                '--runtime-cache-manifest', s['tb_cache'][0], '--runtime-cache-manifest-sha256', s['tb_cache'][1],
                '--required-output-capture-root', s['capture_root'], '--bundle', s['tb_bundle'][0],
                # #531 B5: the offline verifier toolchain's records for a derived bundle (bundle-aptfix.tar ->
                # tb21_6h; paired oracle gate). Unset: the bundle's own stem, as before
                *tb_toolchain(s), *slow_shape(C, model, suite)]
    elif suite == 'tbsci_lite':
        argv = [HARBOR, '-m', 'aa_evals.tbsci_lite.pi', '--pool', s.get('lite_pool', 'v2.1'), '--model', spec,
                '--run-id', run_id, *effort, '--attempts', str(rep), '--include', ','.join(ids), *pi_tail,
                '--runtime-cache-manifest', s['lite_cache'][0], '--runtime-cache-manifest-sha256', s['lite_cache'][1]]
    elif suite == 'gdpval':
        argv = [HARBOR, '-m', 'aa_evals.gdpval.pi.run', 'init', '--run-id', run_id, '--model', spec, *effort,
                '--repeats', str(rep), '--task-set', 'text-only-v1', '--task-ids', ','.join(ids), '--', *pi_tail]
    elif suite == 'critpt':
        argv = [HARBOR, '-m', 'aa_evals.critpt.pi.run', 'init', '--run-id', run_id, '--model', spec, *effort,
                '--repeats', str(rep), '--problems', ','.join(ids), '--', *pi_tail]
    elif suite == 'automationbench':
        f, body = ids_file(C, 'automationbench', ids, stage)
        files[str(f)] = body
        argv = [HARBOR, '-m', 'aa_evals.automationbench.pi.run', 'init', '--run-id', run_id, '--model', spec, *effort,
                '--repeats', str(rep), '--tasks', '@' + str(f), '--', *pi_tail]
    elif suite == 'scicode':
        argv = [STIRRUP, '-m', 'aa_evals.scicode.run', '--model', spec, '--run-id', run_id, '--context-window', str(ctx),
                '--dataset', s.get('scicode_dataset', 'verified-aa-v3'), '--background', 'off',
                *(['--effort', eff] if eff else []), '--repeats', str(rep), '--problems', ','.join(ids),
                '--concurrency', str(budget), '--dispatch', disp, '--chain-cap-sec', str(wall), '--condor']
    elif suite == 'hle':
        f, body = ids_file(C, 'hle', ids, stage)
        files[str(f)] = body
        argv = [STIRRUP, '-m', 'aa_evals.hle.submit', '--run-id', run_id, '--', '--stage', 'generate', '--model', spec,
                *effort, '--context-window', str(ctx), '--task-set', 'text-only-v1', '--ids', str(f),
                '--repeats', str(rep), '--concurrency', str(budget), '--dispatch', disp, '--request-cap-sec', str(wall),
                *(['--timeout', str(wall)] if wall != cap else [])]
    elif suite == 'omniscience':
        argv = ['bash', 'aa_evals/omniscience/submit.sh', run_id, '--stage', 'generate', '--model', spec,
                *effort, '--context-window', str(ctx), '--dataset', 'v1.1', '--repeats', str(rep),
                *(['--limit', '9'] if stage == 'canary' else []), '--concurrency', str(budget), '--dispatch', disp,
                '--timeout', str(wall)]
    elif suite == 'gdppdf':
        argv = [PDFPY, '-m', 'aa_evals.gdppdf.submit', 'run', '--run-id', run_id, '--', '--stage', 'generate',
                '--model', spec, *effort, '--context-window', str(ctx), '--task-set', 'text-only-v1',
                '--task-ids', ','.join(ids), '--repeats', str(rep), '--vision', 'off',
                '--task-concurrency', str(min(budget, n_att)), '--dispatch', disp, '--request-cap-sec', str(wall)]
    else:
        raise KeyError(suite)
    if ro and suite not in PI_SUITES:   # every single-turn command ends in its run.py's own arguments
        argv += [x for k, v in ro.items() for x in (REQUEST_FLAGS[k], json.dumps(v))]
    canary_omni = suite == 'omniscience' and stage == 'canary'
    out = {'suite': suite, 'run_id': run_id, 'n_items': 9 if canary_omni else len(ids), 'repeats': rep,
           'attempts': 9 * rep if canary_omni else n_att, 'dispatch_dir': disp, 'argv': argv, 'files': files}
    if ro:
        out['request_overrides'] = ro
    if rb is not None:
        out['response_budget'] = rb
        if suite not in PI_SUITES:
            out['env'] = {BUDGET_ENV: budget_json(rb)}
    return out


def build(C: Campaign, model: str, stage: str) -> dict:
    m = C.models[model]
    slow_shape(C, model, 'tb21')   # validates worker_profile even when the model runs no tb21
    sel = plan_suites(C, stage)
    cap = capped(C, model)
    cmds = [suite_cmd(C, model, s, sel[s], stage, cap['budgets'][s] if cap else None) for s in model_suites(C, model)]
    ver = version(C, model, stage)
    for c in cmds:   # $AAI_OPEN_VERSION of the suite's process (launch()); the clients record it
        c['aai_open'] = ver['version']
    disp = cmds[0]['dispatch_dir']
    release = int(m.get('release_server_after', C.cfg.get('release_server_after', 600)) or 0)
    init = [HARBOR, '-m', 'aa_evals.common.dispatch.dispatcher', 'init', disp, '--budget', str(m['budget']),
            *(['--model', m['m36_key']] if m.get('m36_key') in DURATION_KEYS else []),
            '--expect-lanes', str(len(cmds)), '--registry', m['registry']]
    # A fixed budget by default; `auto_budget: true` (model, else campaign) opts into #378's hill-climb. It follows a
    # tok/s signal that swings -68%..+102% between windows; on a compute-bound server it raised m5l's 3B from 256 to
    # 499 on 2 GPUs (same tok/s per GPU, per-request decode 26 -> 13-20 tok/s, more attempts at the model-time caps);
    # user 11-16: fixed 128 agents per GPU; #451 decision 3: autobudget retired (#531 B7)
    if not m.get('auto_budget', C.cfg.get('auto_budget', False)):
        init.append('--no-auto-budget')
    # the canary's dispatcher never releases the server: the full run follows on it
    if release and stage == 'full' and not m.get('shared_server'):
        init += ['--release-server-after', str(release)]
    out = {'model': model, 'stage': stage, 'plan': str(C.plan), 'plan_sha256': hashlib.sha256(C.plan.read_bytes()).hexdigest(),
           'registry': m['registry'], 'server': server_argv(C, model), 'dispatcher_init': init,
           'dispatcher_run': [HARBOR, '-m', 'aa_evals.common.dispatch.dispatcher', 'run', disp],
           'suites': cmds, 'attempts': sum(c['attempts'] for c in cmds), 'aai_open': ver}
    if cap:
        out['capped'] = cap
    if request_overrides(C, model):
        out['request_overrides'] = request_overrides(C, model)
    return out


def server_argv(C: Campaign, model: str) -> list[str]:
    """serve_teacher submit for the model. Optional model keys: `checkpoint` (a fine-tuned checkpoint served with the
    teacher spec, e.g. aa-serving snowball-eval for SFT students) and `serve_extra` (extra submit arguments, verbatim)."""
    m = C.models[model]
    return [HARBOR, '-m', 'task_synthesis.serve.serve_teacher', 'submit', '--teacher', m['teacher'], '--name', m['registry'],
            '--gpus', str(m['gpus']), '--cuda-cap', '10.0' if m['gpu'] == 'B200' else '9.0', '--bid', str(m['bid']),
            '--hours', str(m.get('hours', 16)), '--protocol', json.dumps(m['protocol'], separators=(',', ':'))] + \
           (['--tunables', json.dumps(m['tunables'], separators=(',', ':'))] if m.get('tunables') else []) + \
           (['--checkpoint', str(m['checkpoint'])] if m.get('checkpoint') else []) + [str(x) for x in m.get('serve_extra') or []]


# ------------------------------------------------------------------------------------------------ servers
def server_order(C: Campaign, models: list[str]) -> list[list[str]]:
    """Waves of models by GPUs per server, largest first: multi-GPU servers get whole nodes before 1-GPU ones."""
    by = {}
    for x in models:
        by.setdefault(int(C.models[x]['gpus']), []).append(x)
    return [by[g] for g in sorted(by, reverse=True)]


def condor_job_status(job: str, runner=subprocess.run) -> str | None:
    out = runner(['condor_q', job, '-af', 'JobStatus'], capture_output=True, text=True, timeout=120).stdout.split()
    return out[0] if out else None


def live_server(registry: str, runner=subprocess.run) -> list[str]:
    out = runner(['condor_q', '-af:r', 'ClusterId', 'JobStatus', 'JobBatchName', '-constraint',
                  f'JobBatchName == "tasksynth-serve-{registry}" && (JobStatus == 1 || JobStatus == 2)'],
                 capture_output=True, text=True, timeout=120).stdout
    return [l for l in out.splitlines() if l.strip()]


def submit_server(C: Campaign, model: str, dry: bool, runner=subprocess.run) -> str | None:
    m = C.models[model]
    if live_server(m['registry'], runner):
        print(f"{model}: {m['registry']} already has a live job; skipped", flush=True)
        return None
    argv = server_argv(C, model)
    if dry:
        print(f'{model}: ', shlex.join(argv), flush=True)
        return None
    p = runner(argv, cwd=C.cfg['serving_workdir'], text=True, capture_output=True)
    job = (re.findall(r'(\d+)\.0', p.stdout or '') or [None])[-1]
    C.log(event='server_submit', model=model, registry=m['registry'], rc=p.returncode, job=job, argv=argv)
    print(f"{model}: rc {p.returncode} job {job}  {(p.stderr or '').strip()[-200:]}", flush=True)
    return job


def servers(C: Campaign, models: list[str], stagger_min: float, dry: bool, runner=subprocess.run, sleep=time.sleep):
    waves = server_order(C, models)
    for i, wave in enumerate(waves):
        jobs = [j for j in (submit_server(C, x, dry, runner) for x in wave) if j]
        if dry or i == len(waves) - 1 or not jobs:
            continue
        t0 = time.time()      # the next, smaller wave waits until this one runs, at most stagger_min
        while time.time() - t0 < stagger_min * 60:
            if all(condor_job_status(j, runner) == '2' for j in jobs):
                break
            sleep(30)
        print(f'wave of {int(C.models[wave[0]]["gpus"])}-GPU servers: '
              f'{"running" if all(condor_job_status(j, runner) == "2" for j in jobs) else "still waiting"}; next wave',
              flush=True)


def healthy(registry: str) -> dict:
    """endpoints.json of the registry's current job, fresh (< 120 s), and the job live in condor; else SystemExit."""
    e = serving.record(registry, REG)
    j = json.loads((serving.registry_root(REG) / registry / 'job.json').read_text())
    if e.get('job_id') != j.get('job_id') or serving.age(e) > serving.STALE_SEC:
        raise SystemExit(f'{registry}: endpoints.json not fresh for job {j.get("job_id")}')
    if not live_server(registry):
        raise SystemExit(f'{registry}: no live condor job')
    return e


# ------------------------------------------------------------------------------------------------ launch
def launch(C: Campaign, model: str, stage: str, only: list[str] | None = None):
    """Dispatcher init (+ --metrics from the live registry) and loop (nohup, this node), then every suite's command
    from the campaign's checkout; each rc and output tail goes to <dispatch_root>/<name>-launch/events.jsonl."""
    m = C.models[model]
    e = healthy(m['registry'])
    b = build(C, model, stage)
    b['dispatcher_init'] += ['--metrics', e['engine_urls'][0].removesuffix('/v1')]
    disp = Path(b['suites'][0]['dispatch_dir'])
    if not only:
        if disp.exists():
            raise SystemExit(f'{disp} exists: a dispatch dir is never reused')
        for f, body in (x for c in b['suites'] for x in c['files'].items()):
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            if not Path(f).exists():
                Path(f).write_text(body)
            assert Path(f).read_text() == body, f
        p = subprocess.run(b['dispatcher_init'], cwd=C.code, text=True, capture_output=True)
        C.log(event='dispatcher_init', model=model, stage=stage, rc=p.returncode, out=(p.stdout + p.stderr)[-600:])
        if p.returncode:
            raise SystemExit(f'dispatcher init failed: {p.stderr[-400:]}')
        out = open(disp / 'dispatcher.out', 'a')
        proc = subprocess.Popen(b['dispatcher_run'], cwd=C.code, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        (disp / 'dispatcher.pid').write_text(f'{proc.pid} {os.uname().nodename}\n')
        C.log(event='dispatcher_run', model=model, stage=stage, pid=proc.pid, host=os.uname().nodename)
        time.sleep(5)
        if proc.poll() is not None:
            raise SystemExit(f'dispatcher exited rc {proc.returncode}: see {disp}/dispatcher.out')
    else:
        b = {**b, 'suites': [c for c in b['suites'] if c['suite'] in only]}
        assert disp.exists(), disp
    # PI_RESPONSE_BUDGET only from the campaign's capped protocol (suite env) and AAI_OPEN_VERSION only from
    # version(), never from this shell
    base_env = {k: v for k, v in os.environ.items() if k not in (BUDGET_ENV, aai_open.ENV)}
    for c in b['suites']:
        if only and c['suite'] == 'hle' and '--force' not in c['argv']:   # resubmit: the wrapper's receipt exists
            j = c['argv'].index('--')
            c['argv'] = c['argv'][:j] + ['--force'] + c['argv'][j:]
        p = subprocess.run(c['argv'], cwd=C.code, text=True, capture_output=True,
                           env={**base_env, 'PYTHONDONTWRITEBYTECODE': '1', aai_open.ENV: c['aai_open'],
                                **(c.get('env') or {})})
        tail = (p.stdout + p.stderr).strip()[-800:]
        C.log(event='suite', model=model, stage=stage, suite=c['suite'], run_id=c['run_id'], rc=p.returncode, out=tail,
              aai_open=c['aai_open'], **({'aai_open_deviations': b['aai_open']['deviations']}
                                         if b['aai_open']['deviations'] else {}),
              **({'response_budget': c['response_budget']} if c.get('response_budget') else {}),
              **({'request_overrides': c['request_overrides']} if c.get('request_overrides') else {}))
        print(f"{model} {c['suite']:16} rc {p.returncode}  {tail.splitlines()[-1][:150] if tail else ''}", flush=True)
    if stage == 'full' and not only and shadow_on(C, model):
        try:   # optional and read-only: never fails a launch (or watch()'s promotion of the other models)
            start_shadow(C, model, stage)
        except Exception as e:   # noqa: BLE001
            C.log(event='tbsched_shadow_failed', model=model, stage=stage, error=repr(e)[:300])
            print(f'{model}: tbsched shadow not started: {e!r}', flush=True)


# ------------------------------------------------------------------------------------------------ tbsched shadow
# `tbsched_shadow: true` (model or campaign, the model's wins; off by default): after a full stage's suites are
# launched, a tbsched loop in shadow mode (python -m tbsched.loop, #451 workstream 1) runs next to the campaign: it
# scrapes the server and the model's Pi runs and records what it would decide under <dispatch_root>/<name>-<model>-sched/
# (sched/events.jsonl, engines/, groups/), and acts on nothing (mode "shadow", no run "hooked"). It is the evidence for
# the tbsched "on" canary (#451 open item); scores and scheduling are unchanged. `launch.py CONFIG shadow MODEL
# --stage full` starts it by hand for a run already launched. It stops when the dispatcher has released the server
# (state.json `released`), else after the server's hours. Pi lanes that register after SHADOW_WAIT_SEC are left out
# (start it again by hand with the shadow op: the new loop takes over the lease).
SHADOW_WAIT_SEC = 180   # Pi lanes register within ~1 min of their launch (uval 11-18: all six lanes in 60 s)
# the loop in the background; every 60 s: exit (SIGTERM, the loop releases its lease) once the dispatcher released
SHADOW_SH = ('"$1" -B -m tbsched.loop "$3" & p=$!\n'
             'while kill -0 $p 2>/dev/null; do\n'
             '  "$1" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(\\"released\\") else 1)" '
             '"$2/state.json" 2>/dev/null && { kill $p; break; }\n'
             '  sleep 60\ndone\nwait $p\n')


def shadow_on(C: Campaign, model: str) -> bool:
    m = C.models[model]
    return bool(m['tbsched_shadow'] if 'tbsched_shadow' in m else C.cfg.get('tbsched_shadow', False))


def shadow_dir(C: Campaign, model: str, stage: str) -> Path:
    return Path(str(C.dispatch_dir(model, stage)) + '-sched')


def sched_config(C: Campaign, model: str, stage: str) -> dict:
    """The shadow loop's CAMPAIGN/sched/config.json: the model's server as one group ("mixed": its single-turn lanes
    share the engines, so their hit rate is no thrash evidence) and every Pi lane registered in its dispatch dir as an
    unhooked agentic run (slots from the run's config)."""
    m = C.models[model]
    runs = {}
    for f in sorted((C.dispatch_dir(model, stage) / 'lanes').glob('*/registration.json')):
        r = json.loads(f.read_text())
        if r.get('kind') != 'pi' or not r.get('run'):
            continue
        rc = Path(r['run']) / 'config.json'
        slots = (json.loads(rc.read_text()).get('worker') or {}).get('slots') if rc.exists() else None
        runs[r.get('suite') or f.parent.name] = {'dir': r['run'], 'kind': 'agentic', 'groups': {m['registry']: 1.0},
                                                 'slots': int(slots or 16), 'model': m['teacher'], 'hooked': False}
    group = {'registry': str(serving.registry_root(REG) / m['registry']), 'model': m['teacher'], 'mixed': True}
    return {'mode': 'shadow', 'tick_sec': 60, 'scrape_sec': 15, 'groups': {m['registry']: group}, 'runs': runs}


def start_shadow(C: Campaign, model: str, stage: str, wait_sec: float = SHADOW_WAIT_SEC, sleep=time.sleep,
                 popen=subprocess.Popen) -> Path | None:
    """Write the shadow config once the model's Pi lanes have registered (at most wait_sec) and start the loop on this
    node, detached; it ends when the dispatcher releases the server, at the latest after the server's hours."""
    want = sum(1 for s in model_suites(C, model) if s in PI_SUITES)
    t0 = time.time()
    cfg = sched_config(C, model, stage)
    while len(cfg['runs']) < want and time.time() - t0 < wait_sec:
        sleep(15)
        cfg = sched_config(C, model, stage)
    if not cfg['runs']:
        print(f'{model}: no Pi lane registered in {C.dispatch_dir(model, stage)}; tbsched shadow not started', flush=True)
        return None
    d = shadow_dir(C, model, stage)
    (d / 'sched').mkdir(parents=True, exist_ok=True)
    (d / 'sched' / 'config.json').write_text(json.dumps(cfg, indent=1) + '\n')
    argv = ['timeout', str(int(float(C.models[model].get('hours', 16)) * 3600)), 'bash', '-c', SHADOW_SH,
            'tbsched-shadow', sys.executable, str(C.dispatch_dir(model, stage)), str(d)]
    proc = popen(argv, cwd=C.code, stdout=open(d / 'loop.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    (d / 'loop.pid').write_text(f'{proc.pid} {os.uname().nodename}\n')
    C.log(event='tbsched_shadow', model=model, stage=stage, dir=str(d), runs=sorted(cfg['runs']), pid=proc.pid)
    print(f'{model}: tbsched shadow (read-only) in {d}, runs {sorted(cfg["runs"])}', flush=True)
    return d


# ------------------------------------------------------------------------------------------------ canary gate
STATS = re.compile(r'Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs')
BINS = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64), (65, 96), (97, 128), (129, 192), (193, 256), (257, 384),
        (385, 512), (513, 768)]


def bin_of(running: int) -> str | None:
    for lo, hi in BINS:
        if lo <= running <= hi:
            return f'{lo}-{hi}'
    return None


def decode_by_bin(lines) -> dict[str, float]:
    """Median generated tok/s by running-requests bin over vLLM's 10 s stats lines. Each line is one engine (one
    GPU in our TP1 / DP layouts), the same unit as the reference (`ref_decode_per_gpu`, from c30's logs)."""
    acc: dict[str, list[float]] = {}
    for line in lines:
        mm = STATS.search(line)
        if not mm:
            continue
        b = bin_of(int(mm.group(2)))
        if b:
            acc.setdefault(b, []).append(float(mm.group(1)))
    out = {}
    for b, v in acc.items():
        v.sort()
        out[b] = v[len(v) // 2]
    return out


def decode_ratio(measured: dict[str, float], ref: dict[str, float], min_lines_bins=None) -> float | None:
    """Median over bins present in both of measured / reference; None without overlap. Bins 1-8 are skipped (prompt
    bursts dominate them)."""
    r = sorted(measured[b] / ref[b] for b in measured if b in ref and b != '1-8' and ref[b] > 0)
    return r[len(r) // 2] if r else None


def canary_state(C: Campaign, model: str) -> dict:
    """From the canary dispatcher: every expected lane registered and done, and the share of infra outcomes."""
    d = C.dispatch_dir(model, 'canary')
    st = json.loads((d / 'state.json').read_text()) if (d / 'state.json').exists() else {}
    lanes = st.get('lanes') or {}
    ends = [json.loads(l) for l in open(d / 'events.jsonl')] if (d / 'events.jsonl').exists() else []
    ends = [e for e in ends if e.get('ev') == 'end']
    infra = sum(1 for e in ends if str(e.get('status', '')).startswith('infra'))
    return {'lanes': len(lanes), 'expected': len(model_suites(C, model)), 'done': bool(lanes) and all(x.get('done') for x in lanes.values()),
            'in_flight': st.get('in_flight'), 'ended': len(ends), 'infra': infra,
            'infra_share': infra / len(ends) if ends else 0.0}


def check(C: Campaign, model: str) -> dict:
    m = C.models[model]
    cs = canary_state(C, model)
    reg = serving.registry_root(REG) / m['registry']
    logs = sorted(reg.glob('vllm*.log'))
    lines = []
    for f in logs:
        lines += f.read_text(errors='replace').splitlines()
    meas = decode_by_bin(lines)
    ratio = decode_ratio(meas, m.get('ref_decode_per_gpu') or {})
    reasons = []
    if not (cs['done'] and cs['lanes'] >= cs['expected']):
        reasons.append(f"canary not done ({cs['lanes']}/{cs['expected']} lanes registered, done={cs['done']})")
    if cs['infra_share'] > INFRA_MAX_SHARE:
        reasons.append(f"canary infra {cs['infra']}/{cs['ended']}")
    if ratio is not None and ratio < DECODE_MIN_RATIO:
        reasons.append(f'decode {ratio:.2f}x the reference (< {DECODE_MIN_RATIO}): check the serving config')
    return {'model': model, 'ok': not reasons, 'reasons': reasons, 'canary': cs, 'decode_ratio': ratio,
            'decode_by_bin': {b: round(v) for b, v in meas.items()}, 'no_reference': not m.get('ref_decode_per_gpu')}


def promote(C: Campaign, model: str, force: bool = False) -> bool:
    r = check(C, model)
    print(json.dumps(r, indent=1), flush=True)
    C.log(event='check', **{k: v for k, v in r.items() if k != 'decode_by_bin'})
    if not (r['ok'] or force):
        return False
    if C.dispatch_dir(model, 'full').exists():
        print(f'{model}: full run already launched', flush=True)
        return True
    launch(C, model, 'full')
    C.log(event='promoted', model=model, forced=force and not r['ok'])
    return True


def watch(C: Campaign, models: list[str], every: float):
    left = [x for x in models if not C.dispatch_dir(x, 'full').exists()]
    while left:
        for x in list(left):
            try:
                if C.dispatch_dir(x, 'canary').exists() and promote(C, x):
                    left.remove(x)
            except SystemExit as e:
                print(f'{x}: {e}', flush=True)
        if left:
            time.sleep(every)


# ------------------------------------------------------------------------------------------------ show
def show(C: Campaign, models: list[str], stage: str):
    tot = 0
    for wave in server_order(C, models):
        print('server wave:', ', '.join(f"{x} ({C.models[x]['gpus']}x{C.models[x]['gpu']})" for x in wave))
    for model in models:
        b = build(C, model, stage)
        m = C.models[model]
        tot += b['attempts']
        sset = m.get('suites') if isinstance(m.get('suites'), str) else ('custom' if m.get('suites') else 'full')
        print(f"=== {model}  ({m['name']}; {m['gpus']}x{m['gpu']} bid {m['bid']}; registry {m['registry']}; "
              f"budget {m['budget']}; effort {m.get('effort') or 'registry default'}; suites {sset}; "
              f"{b['attempts']} attempts)")
        print(f"  aai-open: {b['aai_open']['version']}" + ''.join(f'\n            deviates: {d}' for d in b['aai_open']['deviations']))
        if shadow_on(C, model) and stage == 'full':
            print(f"  tbsched: shadow (read-only) in {shadow_dir(C, model, stage)} after launch")
        if b.get('capped'):
            print(f"  capped:   {b['capped']['file']} (sha256 {b['capped']['sha256'][:12]})")
        if b.get('request_overrides'):
            print(f"  sampling: {json.dumps(b['request_overrides'])} (every suite's requests)")
        print('  server:  ', shlex.join(b['server']))
        print('  dispatch:', shlex.join(b['dispatcher_init']))
        for c in b['suites']:
            print(f"  [{c['suite']:15}] {c['n_items']:>3} x {c['repeats']} = {c['attempts']:>4}  {c['run_id']}")
            short = [a if len(a) < 120 or a.startswith(BUDGET_ENV + '=') else a[:60] + f'...<{a.count(",") + 1} ids>'
                     for a in c['argv']]
            print('      ', shlex.join(short))
            for k, v in (c.get('env') or {}).items():
                print('       env:', shlex.join([f'{k}={v}']))
    print(f'total attempts: {tot}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('op', choices=['show', 'servers', 'launch', 'check', 'promote', 'watch', 'shadow'])
    ap.add_argument('models', nargs='*')
    ap.add_argument('--stage', choices=['full', 'canary'], default='canary')
    ap.add_argument('--only', default=None)
    ap.add_argument('--stagger-min', type=float, default=10.0)
    ap.add_argument('--every', type=float, default=300.0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    C = Campaign(a.config)
    models = a.models or list(C.models)
    for x in models:
        if x not in C.models:
            raise SystemExit(f'unknown model {x}')
    if a.op == 'show':
        show(C, models, a.stage)
    elif a.op == 'servers':
        servers(C, models, a.stagger_min, a.dry_run)
    elif a.op == 'launch':
        assert len(models) == 1, 'launch takes one model'
        launch(C, models[0], a.stage, a.only.split(',') if a.only else None)
    elif a.op == 'check':
        for x in models:
            print(json.dumps(check(C, x), indent=1))
    elif a.op == 'promote':
        assert len(models) == 1, 'promote takes one model'
        promote(C, models[0], a.force)
    elif a.op == 'watch':
        watch(C, models, a.every)
    elif a.op == 'shadow':
        assert len(models) == 1, 'shadow takes one model'
        start_shadow(C, models[0], a.stage)


if __name__ == '__main__':
    main()

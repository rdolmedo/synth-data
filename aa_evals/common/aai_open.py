"""AAI-open version of a run (#481). The versions are specified in docs/aai_open/VERSIONS.md; this module holds the
current version's pins, the label every new run records, and the aggregate formula.

Label: CURRENT ('v0.1' on main) when the run follows the version's spec, CURRENT + '+custom' when it deviates in a
way that can be checked (another item plan, context window, caps, verifier bundle, capped-protocol file, sampling
overrides, finish notice, image inputs, ...). The campaign launcher (aa_evals/campaign/launch.py) checks the campaign
config against the spec and passes its label to every suite command in $AAI_OPEN_VERSION; the single-turn submitters
forward it into their condor jobs (`forwarded`). Each client then writes `aai_open` into its run config (`record`),
adding the deviations it can see itself (e.g. --temperature, --request-overrides, --context-window). A run started
by hand has no launcher label: it records CURRENT (source 'client') unless the client sees a deviation; the item
plan is only checked by the launcher.

Recording changes nothing a run does: it is one more key in config.json.

Public helpers for readers and reports (stable): `of_run(run_dir) -> str` and `of_config(cfg) -> str`, the recorded
label of a run ('unrecorded' for runs from before #481; a collected GDPval/CritPt/AutomationBench dir answers for
its Pi run).

  python -m aa_evals.common.aai_open show RUN_DIR [...]   # the recorded version of each run ('unrecorded' before v0.1)
  python -m aa_evals.common.aai_open condor-env           # " AAI_OPEN_VERSION='...'" or nothing (shell submitters)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

CURRENT = 'v0.1'
CUSTOM = '+custom'
ENV = 'AAI_OPEN_VERSION'
LABEL_RE = re.compile(r'v\d+(\.\d+)*(\+custom)?')   # used with fullmatch (no trailing newline)

# ---- v0.1 pins (docs/aai_open/VERSIONS.md, section v0.1); the campaign launcher checks a config against them
PLAN_SHA256 = 'b3f0a1fc86cc7f16b2afcd1a38eebe24708ca01eae890336da48c1bc774ec943'   # c30 cheap_full_v1.json
CONTEXT_WINDOW = 131072
CAP_SEC = 7200                    # Pi environment clock; also every model-time cap without the capped protocol (#344)
CAPPED_WALL_CAP_SEC = 43200       # model-time caps under the capped protocol (hang backstop, #531 B6)
CAPPED_SHA256 = '403ad91eb3582e718556a9d0070be42044971aaf6857e10dbff2c9645f399f96'   # capped-sft-eval-v1
SUITE_PINS = {   # campaign `suites` key -> the value v0.1 needs ([path, sha256] pins compare the sha only)
    'tb_bundle': '9b0b4dbf5b1ac92cb0469d2eecf91a402afe1b1ca34f0f77905a28f4c7781ddc',      # bundle-aptfix.tar
    'tb_toolchain_bundle': ['tb21_6h', '9b0b4dbf5b1ac92cb0469d2eecf91a402afe1b1ca34f0f77905a28f4c7781ddc'],   # #543
    'tb_cache': 'd9bcbf4231b1b2e449dd011840d4a58d4211f57b808543b1c1fc6b2b8394e5e4',       # Pi runtime cache v6
    'lite_cache': 'd7fcf56be3d5c3d79d7d50e8b059772d4fcc2df11a22f39ef6f67737145fc317',     # Pi runtime cache v5
}
SUITE_DEFAULTS = {'scicode_dataset': 'verified-aa-v3', 'lite_pool': 'v2.1'}   # launcher defaults when unset
NOTICE = {'policy': 'submit-at-context-limit-v4-sticky-notice', 'warn_at': 40000, 'reserve': 16384,
          'text_sha256': 'e8147af470746c79df7a707ff1ed7dc01cbde38d13f9e31a2c2dfe44e067fd15'}   # Pi finish notice v4

# ---- aggregate (unchanged from v0): AA v5.0 slot weights, raw % per slot, GDPval Elo through AA's Briefcase clamp
SLOTS = [('tb21', 10), ('tbsci_lite', 15), ('automationbench', 5), ('omniscience', 10), ('omni_nhr', 5),
         ('critpt', 10), ('hle', 10), ('gdppdf_allpass', 5), ('scicode_main', 15), ('gdpval', 15)]
# c30 leaderboard rows (#375, marin#11563's open-model table): each suite's score on the plan's items times
# retained / universe (the items no list-builder ever solved count 0). The SFT ladder rows used the raw scores.
UNIVERSE = {'tb21': (84, 89), 'tbsci_lite': (32, 135), 'automationbench': (553, 600), 'critpt': (70, 70),
            'scicode_main': (60, 64), 'hle': (728, 728), 'omniscience': (300, 300), 'gdppdf_allpass': (37, 75)}


def briefcase(elo: float) -> float:
    """GDPval Elo -> the Briefcase slot: 100 * clamp((Elo - 500) / 2000, 0, 1)."""
    return 100.0 * min(1.0, max(0.0, (float(elo) - 500.0) / 2000.0))


def slot_value(key: str, scores: dict, scale: str = 'raw') -> float:
    v = briefcase(scores[key]) if key == 'gdpval' else float(scores[key])
    if scale == 'universe' and key in UNIVERSE:
        r, u = UNIVERSE[key]
        v *= r / u
    return v


def aggregate(scores: dict, scale: str = 'raw', fill: dict | None = None) -> dict:
    """The AAI-open score of one model from its slot scores ({slot key: %, 'gdpval': Elo}).
    scale 'raw' (the M5 SFT ladder, v0) or 'universe' (c30 leaderboard rows). A missing slot takes `fill`'s value
    when given (v0 lean evals: the 10.73B SFT's TB-Sci-lite / CritPt / HLE), else the slot is left out and the result
    is out of the weights present ('points': 65 for the lean set)."""
    if scale not in ('raw', 'universe'):
        raise ValueError(scale)
    tot, pts, filled, missing = 0.0, 0, [], []
    for k, w in SLOTS:
        src = scores if scores.get(k) is not None else (fill or {})
        if src.get(k) is None:
            missing.append(k)
            continue
        if src is not scores:
            filled.append(k)
        tot += w / 100 * slot_value(k, src, scale)
        pts += w
    return {'score': tot, 'points': pts, 'filled': filled, 'missing': missing, 'scale': scale}


# ---- labels
def valid(v) -> bool:
    return isinstance(v, str) and LABEL_RE.fullmatch(v) is not None


def label(base: str | None = None, deviations=()) -> str:
    """`base` (default CURRENT) with '+custom' added when there are deviations. ValueError for a malformed base
    (callers on production paths pass only validated labels: launcher_value never returns a malformed one)."""
    base = base or CURRENT
    if not valid(base):
        raise ValueError(f'{ENV}={base!r}: expected v<N>[.<M>][+custom]')
    return base + CUSTOM if deviations and not base.endswith(CUSTOM) else base


def env_label(env=None) -> tuple[str | None, list[str]]:
    """(label, deviations) from $AAI_OPEN_VERSION. Unset: (None, []). A value that is not a label (a future format
    seen by older code, a typo) never fails a run: it becomes CURRENT+custom with a deviation naming it."""
    env = os.environ if env is None else env
    v = env.get(ENV)
    if not v:
        return None, []
    if valid(v):
        return v, []
    return CURRENT + CUSTOM, [f'unrecognized {ENV} {v[:40]!r}']


def launcher_value(env=None) -> str | None:
    """The launcher's label from $AAI_OPEN_VERSION (CURRENT+custom for a malformed value); None when unset."""
    return env_label(env)[0]


def forwarded(env=None) -> dict:
    """{AAI_OPEN_VERSION: label} for a condor job's environment when the submitter has it set; {} otherwise. A
    malformed value is forwarded as CURRENT+custom, so nothing unvalidated reaches a submit file."""
    v = launcher_value(env)
    return {ENV: v} if v else {}


def condor_env(fwd=None) -> str:
    """`fwd` (default: forwarded() of this process) as text to append to a condor `environment` line:
    " AAI_OPEN_VERSION='v0.1'", or '' when unset, so the submit file is unchanged (a label never needs escaping)."""
    fwd = forwarded() if fwd is None else fwd
    return ''.join(f" {k}='{v}'" for k, v in fwd.items())


def record(deviations=(), env=None, source_cfg: dict | None = None) -> dict:
    """The run config's `aai_open` block: {'version', 'deviations', 'source'}. `deviations` are what the client sees
    itself; the launcher's label (from $AAI_OPEN_VERSION) already carries the campaign-level ones.
    `source_cfg`: the config of the run this one continues (a resume, or the generation run of a grade stage): its
    record is kept, plus any new deviation; a run started before the field existed stays 'unrecorded'."""
    dev = [str(d) for d in deviations if d]
    if source_cfg is not None:
        prev = source_cfg.get('aai_open')
        if not prev:
            return {'version': 'unrecorded', 'deviations': dev, 'source': 'continues a run from before #481'}
        old = list(prev.get('deviations') or [])
        new = [d for d in dev if d not in old]
        v = prev.get('version') or 'unrecorded'
        # a continued 'unrecorded' run (a re-grade of a re-grade of an old run) stays unrecorded
        return {'version': label(v, new) if valid(v) else v, 'deviations': old + new,
                'source': prev.get('source')}
    base, bad = env_label(env)
    dev = bad + dev
    # source 'client': no launcher label, so the item plan and the campaign-level pins were not checked
    return {'version': label(base, dev), 'deviations': dev, 'source': 'launcher' if base else 'client'}


JUDGE_ARM = 'qwen38-27b-medium-pin'   # HLE, Omniscience, GDP.pdf graders and the GDPval pairwise judge (v0, v0.1)
GDPVAL_REFS = ['m36-qwen38-xhigh-gdpval-full', 'm36-lightning-gdpval-full']   # GDPval reference runs (#183)


def judge_deviations(arm) -> list[str]:
    """A grade stage on another judge arm than the version's (None: a judge given without an arm). Call it only for
    stages that grade."""
    return [] if arm == JUDGE_ARM else [f'judge arm {arm} (v0.1: {JUDGE_ARM})']


def client_deviations(*, context_window=None, overrides=None, notice='unchecked', images=None,
                      flags=None) -> list[str]:
    """What a client can check from its own settings: context window, explicit sampling overrides, the Pi finish
    notice (its recorded protocol.finish_notice; pass 'unchecked' for single-turn clients), image inputs, and request
    flags given by hand (`flags`: {'--extra-body': value, ...}; set ones are deviations: the launcher passes none)."""
    out = [f'{k} {str(v)[:60]} (v0.1: unset)' for k, v in (flags or {}).items() if v not in (None, False, '')]
    if context_window is not None and int(context_window) != CONTEXT_WINDOW:
        out.append(f'context_window {context_window} (v0.1: {CONTEXT_WINDOW})')
    for k, v in (overrides or {}).items():
        if v is not None:
            out.append(f'sampling override {k}={v}')
    if notice != 'unchecked':
        got = {k: (notice or {}).get(k) for k in NOTICE}
        if got != NOTICE:
            out.append(f'finish notice {got if notice else "off"} (v0.1: v4 {NOTICE["warn_at"]}:{NOTICE["reserve"]})')
    if images:
        out.append(f'image inputs {images} (v0.1: text only)')
    return out


def of_config(cfg: dict) -> str:
    """The recorded label of a run config (a collector-made GDPval/CritPt/AutomationBench dir points at its Pi run)."""
    v = (cfg.get('aai_open') or {}).get('version')
    if v:
        return v
    pi_run = (cfg.get('harness') or {}).get('pi_run') if isinstance(cfg.get('harness'), dict) else None
    if pi_run:
        try:
            return of_config(json.loads((Path(pi_run) / 'config.json').read_text()))
        except (OSError, ValueError):
            pass
    return 'unrecorded'


def of_run(run_dir) -> str:
    try:
        return of_config(json.loads((Path(run_dir) / 'config.json').read_text()))
    except (OSError, ValueError):
        return 'unrecorded'


if __name__ == '__main__':
    if sys.argv[1:] == ['condor-env']:
        print(condor_env(), end='')
    elif sys.argv[1:2] == ['show'] and len(sys.argv) > 2:
        for d in sys.argv[2:]:
            print(f'{of_run(d):14s} {d}')
    else:
        raise SystemExit('\n'.join(__doc__.strip().splitlines()[-2:]))

"""Per-sandbox disk cap of the AA harnesses (#405): every harbor_ns run of pi_eval (TB 2.1, TB-Science(-lite) Pi,
GDPval, CritPt, AutomationBench) and harbor_mswea (TB 4.0, TB-Science(-lite)) caps each sandbox's disk use.

Why: all sandboxes of a worker write their overlay upper dirs to one HTCondor scratch, and harbor_ns has no per-box
filesystem. One runaway writer fills it and every other attempt on the worker fails as a model failure (#371:
TB2.1-easy agents on qemu-alpine-ssh ran a 1 TiB `dd`). Mechanism: harbor_ns/diskcap.py (RLIMIT_FSIZE per file plus a
watchdog that kills the box's top writer above the cap; multi-uid boxes included).

Same default and opt-out as eval/eval_q35 (PR #372, #425):
- run.py (pi_eval and harbor_mswea) writes "diskcap" to every new run's config.json: `--diskcap auto|FACTOR[:MIN_MB]|off`,
  default auto, as eval_q35's submit.py: a bench that names its family in its run defaults ("diskcap_family") gets
  eval_q35's measured per-task caps (common.DISKCAP_FAMILIES via diskcap_for; TB 2.1: rstan-to-pystan 90 GiB and one
  attempt at a time per worker, sam-cell-seg 40 GiB, others 32 GiB), every other bench DEFAULT: cap =
  max(storage_mb x 3, 30720) MB, 30 GiB for tasks that declare no storage_mb;
- "diskcap": null / false / 0 / "off" turns it off (explicit HARBOR_NS_DISKCAP=0, so a value inherited from the
  worker's environment cannot turn it on);
- a dict sets another cap, in eval_q35's format (eval_q35/common.py diskcap_config: {"factor": F, "min_mb": M,
  "tasks": {task: MB}, "one_per_worker": [...]}; one_per_worker is honored by pi_eval, whose worker is eval_q35's).
Unlike eval_q35, a config WITHOUT the key is off: every such AA run predates #405 and ran uncapped (pi_eval:
PiEvalWorker.DISKCAP_WITHOUT_KEY = False). The parsing itself is eval_q35's, so both pipelines read a "diskcap" alike.
Each attempt records its effective cap (pi_eval: attempt.json "diskcap"; harbor_mswea: the outcome's "diskcap"), the
box's limit, peak, largest file and kills in diskcap.json (a separate verifier environment: diskcap.verifier.json), and
outcome fields from it (`outcome_fields`). Runs execute their own code snapshot, so runs created before this keep their
behavior.
"""
from __future__ import annotations

import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

DEFAULT = {'factor': 3.0, 'min_mb': 30720}
# run.py --diskcap default, as eval_q35's submit.py: the benchmark family's measured per-task caps
# (eval_q35/common.py DISKCAP_FAMILIES via diskcap_for; a bench names its family in its run defaults, "diskcap_family"),
# DEFAULT for a bench without one
DEFAULT_ARG = 'auto'
# HARBOR_NS_DISKCAP_SPARE is popped too, so a value in the worker's environment cannot change who is spared
ENV_KEYS = ('HARBOR_NS_DISKCAP', 'HARBOR_NS_DISKCAP_MIN_MB', 'HARBOR_NS_DISKCAP_LOG', 'HARBOR_NS_DISKCAP_SPARE')
OFF = ('', 'off', 'false', 'no', '0', 'null', 'none')
ON = ('on', 'true', 'yes')


@lru_cache(maxsize=1)
def _q35():
    """eval_q35/common.py (constants and pure functions; not a package, so loaded by path)."""
    spec = importlib.util.spec_from_file_location('aa_diskcap_eval_q35_common',
                                                  Path(__file__).resolve().parents[2] / 'eval_q35' / 'common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DISKCAP_DEFAULT == DEFAULT, 'AA and eval_q35 share the default cap (#372, #405)'
    return mod


def auto(family: str | None, names=()) -> dict:
    """The "diskcap" of a new run of these tasks: eval_q35's per-task caps of the benchmark family (diskcap_for: {"factor":
    1, "min_mb": M, "tasks": {task: MB}, "one_per_worker": [...]}), DEFAULT without a family or measurements."""
    if not family:
        return dict(DEFAULT)
    return _q35().diskcap_for(family, list(names))


def arg(s: str, family: str | None = None, names=()) -> dict | None:
    """--diskcap auto|FACTOR[:MIN_MB]|off -> the run config's "diskcap": auto = the family's caps (auto); None (off,
    recorded as null) for off / 0 / factor <= 0; DEFAULT for on."""
    s = str(s).strip().lower()
    if s == 'auto':
        return auto(family, names)
    if s in OFF:
        return None
    if s in ON:
        return dict(DEFAULT)
    parts = s.split(':')
    if len(parts) > 2:
        raise ValueError(f'--diskcap {s!r}: expected FACTOR[:MIN_MB]')
    f, m = float(parts[0]), float(parts[1] or 0) if len(parts) > 1 else 0.0
    if not (math.isfinite(f) and math.isfinite(m)) or m < 0:
        raise ValueError(f'--diskcap {s!r}: FACTOR and MIN_MB must be finite, MIN_MB >= 0')
    if f <= 0:
        return None
    return {'factor': f, 'min_mb': int(m)}


def config(cfg: dict, task: str | None = None) -> dict | None:
    """The run's effective disk cap for a task, {"factor": F, "min_mb": M} (the task's "tasks" entry as its min_mb), or
    None (off). Off without a "diskcap" key (AA runs before #405); otherwise eval_q35's diskcap_config (ValueError for a
    malformed one), so a pi_eval worker (eval_q35's env) and this read a config alike."""
    if 'diskcap' not in cfg:
        return None
    return _q35().diskcap_config(cfg, task)


def env(e: dict, cfg: dict, log_path: Path | str, task: str | None = None) -> dict:
    """Set the harbor_ns disk-cap environment of one attempt in e (in place; returned): the run's cap for the task, its
    summary to log_path. Off is explicit (HARBOR_NS_DISKCAP=0)."""
    cap = config(cfg, task)
    for k in ENV_KEYS:
        e.pop(k, None)
    if not cap:
        e['HARBOR_NS_DISKCAP'] = '0'
        return e
    e['HARBOR_NS_DISKCAP'] = f"{cap['factor']:g}"
    e['HARBOR_NS_DISKCAP_MIN_MB'] = str(cap['min_mb'])
    e['HARBOR_NS_DISKCAP_LOG'] = str(log_path)
    return e


def outcome_fields(log_path: Path | str) -> dict:
    """Outcome fields from an attempt's diskcap.json (and diskcap.verifier.json of a separate verifier): the cap,
    the box's peak and largest file, kills, files at the per-file limit. diskcap_limit_mb None: cap off or no box."""
    p = Path(log_path)
    out = {'diskcap_limit_mb': None}
    try:
        box = json.loads(p.read_text())
    except (OSError, ValueError):
        return out
    out.update(diskcap_limit_mb=box.get('limit_mb'), sandbox_disk_peak_mb=box.get('peak_mb'),
               sandbox_largest_file_mb=box.get('largest_file_mb'), diskcap_kills=len(box.get('kills') or []),
               diskcap_at_fsize=len(box.get('at_fsize') or []))
    try:
        ver = json.loads(p.with_name(f'{p.stem}.verifier{p.suffix}').read_text())
    except (OSError, ValueError):
        return out
    out.update(verifier_disk_peak_mb=ver.get('peak_mb'), diskcap_kills=out['diskcap_kills'] + len(ver.get('kills') or []))
    return out


def check_harbor_env(harbor_env: dict) -> None:
    """The cap is the run config's "diskcap" (--diskcap), not a free-form harbor env entry (nor an ns_diskcap
    environment kwarg inside one, e.g. HARBOR_EXTRA_ARGS)."""
    bad = sorted(k for k, v in harbor_env.items() if k.startswith('HARBOR_NS_DISKCAP') or 'ns_diskcap' in str(v))
    if bad:
        raise SystemExit(f'{", ".join(bad)}: set the disk cap with --diskcap FACTOR[:MIN_MB]|off, not --harbor-env')


def check_stored(cfg: dict, flag: str | None, family: str | None = None, names=()) -> None:
    """An explicit --diskcap on an existing run must match its stored setting (runs keep their policy; a run without
    the key is off). auto is resolved for the run's own tasks (names)."""
    if flag is None:
        return
    try:
        stored = cfg['diskcap'] if isinstance(cfg.get('diskcap'), dict) else config(cfg)
        new = arg(flag, family, names)
        differs = json.loads(json.dumps(new)) != stored   # as stored in config.json (lists, not tuples)
    except ValueError as e:
        raise SystemExit(f'--diskcap {flag} / stored "diskcap" {cfg.get("diskcap")!r}: {e}') from None
    if differs:
        raise SystemExit(f'--diskcap {flag} differs from this stored run ({cfg.get("diskcap", "off: no key")}); '
                         'use a new run ID')

"""Sandbox limits of a synth campaign's trace generation: memory, stack and disk (#378 train/test audit, fix 4).

Trace generation stays a bit stricter than the checkpoint eval (eval_q35 on the eval branch), as the user allowed
("eval can be a bit looser than trace generation"); the coordinator set the values on 2026-11-14:

  memory  the synth PSS guard (harbor_ns.sandbox, PI_CAMPAIGN_RESOURCE_GUARD) at max(1.5 x memory_mb, 24 GiB), the
          agent (`pi`) killed only when nothing else is left, as eval's harbor_ns/memcap.py spares it. Eval:
          max(2.0 x memory_mb, 32 GiB), counting Pss_Anon + Pss_Shmem; the synth guard counts all of Pss (file-backed
          mappings too), so it stays the stricter of the two at the same budget.
  stack   unlimited, as eval (was an 8 MiB soft RLIMIT_STACK). Not 64 MiB: glibc sizes a new thread's default stack
          from the soft limit (2 MiB when unlimited), so 64 MiB threads would make tasks with RLIMIT_AS / ulimit -v
          caps behave differently from the eval. A runaway recursion now grows until the PSS guard (stack pages are
          anonymous memory) or synth.memory_guard stops it, as in eval.
  disk    eval's per-sandbox disk cap (harbor_ns/diskcap.py, eval PR #425), with eval_q35's rules copied below: a
          family's per-task caps for TB tasks (harbor.diskcap_family = "tb2.1" | "tblite"), else eval's default rule
          max(3 x storage_mb, 30 GiB) (DISKCAP_DEFAULT). Synth tasks are not TB tasks and have no measurements, so they
          get the default: task_synthesis/pipeline/package.py declares storage_mb = 10240, i.e. 30 GiB.
          A family's "one_per_worker" list is recorded but not enforced: synth workers do not schedule by task
          (eval's worker runs those tasks one at a time). Each attempt's box peaks and kills go to raw/<uid>/diskcap.json.
  network unchanged (offline sandboxes with the package allowlist proxy, synth.package_proxy).

synth.config.build writes these values into every new campaign's frozen [harbor] (NEW_CAMPAIGN_HARBOR, under whatever
the campaign TOML sets). A frozen campaign without the keys keeps its old behaviour: stack 8 MiB, guard at 1.0 x
memory_mb (4096 MB undeclared) killing the largest process whatever it is, no disk cap. A partial [harbor.diskcap] or
[harbor.sandbox_memory] table in a new campaign's TOML is merged into these defaults (e.g. {min_mb = 40960} keeps
factor 3; {factor = 2.0} keeps min_mb 24576 and spare ["pi"]).
"""
import copy

STACK_KB_LEGACY = 8192
STACK_UNLIMITED = 'unlimited (as eval)'
SANDBOX_MEMORY = {'factor': 1.5, 'min_mb': 24576, 'spare': ['pi']}

# eval_q35/common.py (origin/eval 84e37296, #425), copied verbatim: DISKCAP_DEFAULT, DISKCAP_WORKER_DISK_MB,
# DISKCAP_FAMILIES, check_diskcap. Keep them equal to eval's.
DISKCAP_DEFAULT = {'factor': 3.0, 'min_mb': 30720}
DISKCAP_WORKER_DISK_MB = 200 * 1024
DISKCAP_FAMILIES = {
    'tb2.1': {'min_mb': 32768, 'tasks': {'rstan-to-pystan': 92160, 'sam-cell-seg': 40960},
              'one_per_worker': ('rstan-to-pystan',)},
    'tblite': {'min_mb': 20480, 'tasks': {}},
}

NEW_CAMPAIGN_HARBOR = {'stack': STACK_UNLIMITED, 'sandbox_memory': SANDBOX_MEMORY, 'diskcap': DISKCAP_DEFAULT}


def check_diskcap(cap):
    """A dict "diskcap" config, validated (ValueError): "tasks" entries are caps in MB > 0 (0 is not "off" for one
    task; turn the run's cap off instead), "one_per_worker" a list of task names. "min_mb" may be 0 only for the bare
    factor form (cap = storage_mb x factor; tasks without storage_mb then have none). Returns cap."""
    tasks = cap.get('tasks') or {}
    if not isinstance(tasks, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                                              for v in tasks.values()):
        raise ValueError(f'diskcap "tasks" must map task names to caps in MB > 0: {tasks!r}')
    single = cap.get('one_per_worker') or []
    if not isinstance(single, (list, tuple)) or not all(isinstance(t, str) for t in single):
        raise ValueError(f'diskcap "one_per_worker" must be a list of task names: {single!r}')
    if 'min_mb' in cap and (isinstance(cap['min_mb'], bool) or not isinstance(cap['min_mb'], (int, float))
                            or cap['min_mb'] < 0):
        raise ValueError(f'diskcap "min_mb" must be a number of MB >= 0: {cap["min_mb"]!r}')
    if (tasks or single) and not cap.get('min_mb'):
        raise ValueError('a per-task diskcap needs "min_mb" > 0 (the cap of the tasks not listed)')
    return cap


def family_diskcap(family):
    """The frozen "diskcap" of a campaign of one TB family's tasks: eval_q35.common.diskcap_for's result for a run of
    that family ({"factor": 1, "min_mb": the family's, "tasks": its per-task caps, "one_per_worker": ...})."""
    if family not in DISKCAP_FAMILIES:
        raise ValueError(f'harbor.diskcap_family must be one of {sorted(DISKCAP_FAMILIES)}: {family!r}')
    f = DISKCAP_FAMILIES[family]
    single = sorted(t for t, mb in f['tasks'].items() if t in f.get('one_per_worker', ()) or mb > DISKCAP_WORKER_DISK_MB / 4)
    return check_diskcap({'factor': 1.0, 'min_mb': f['min_mb'], 'tasks': dict(sorted(f['tasks'].items())),
                          **({'one_per_worker': single} if single else {})})


def new_campaign_harbor(harbor):
    """[harbor] of a NEW campaign: NEW_CAMPAIGN_HARBOR under the campaign's own keys; harbor.diskcap_family picks a TB
    family's per-task disk caps; a harbor.diskcap / harbor.sandbox_memory table is merged into its default (false / off
turns the disk cap off). Validated; the result is what gets frozen."""
    harbor = copy.deepcopy(harbor or {})
    out = {**copy.deepcopy(NEW_CAMPAIGN_HARBOR), **harbor}
    if 'stack_kb' in harbor:            # the campaign's explicit soft limit wins over the unlimited default
        out.pop('stack', None)
    elif out.get('stack') != STACK_UNLIMITED:
        raise ValueError(f'harbor.stack must be {STACK_UNLIMITED!r} (or set harbor.stack_kb): {out.get("stack")!r}')
    # a partial table is merged into the default (a whole replacement would drop its factor, its spared agent, ...)
    family = out.get('diskcap_family')
    base = family_diskcap(family) if family is not None else copy.deepcopy(DISKCAP_DEFAULT)
    if isinstance(harbor.get('diskcap'), dict):
        out['diskcap'] = {**base, **harbor['diskcap']}
    elif 'diskcap' not in harbor:
        out['diskcap'] = base
    if isinstance(harbor.get('sandbox_memory'), dict):
        out['sandbox_memory'] = {**copy.deepcopy(SANDBOX_MEMORY), **harbor['sandbox_memory']}
    diskcap_config(out)                 # validates
    sandbox_memory(out)
    return out


def diskcap_config(harbor, task=None):
    """A campaign's effective disk cap for a task ({"factor", "min_mb"}) or None (off), by eval_q35's diskcap_config
    rules, except that a [harbor] without "diskcap" is off (campaigns frozen before 2026-11-14 had none)."""
    if 'diskcap' not in (harbor or {}):
        return None
    cap = harbor['diskcap']
    if cap is True:   # "on": the default cap
        return dict(DISKCAP_DEFAULT)
    if cap is False or not cap or (isinstance(cap, str) and cap.strip().lower() in ('off', 'false', 'no', '0')):
        return None
    if not isinstance(cap, dict):
        cap = {'factor': float(cap), 'min_mb': 0}
    if not cap.get('factor') or float(cap['factor']) <= 0:
        return None
    check_diskcap(cap)
    per_task = (cap.get('tasks') or {}).get(task)
    return {'factor': float(cap['factor']), 'min_mb': int(per_task if per_task is not None else cap.get('min_mb') or 0)}


def sandbox_memory(harbor):
    """The campaign's synth PSS guard settings ({"factor", "min_mb", "spare"}), or None for the legacy rule."""
    mem = (harbor or {}).get('sandbox_memory')
    if not mem:
        return None
    factor, floor, spare = float(mem.get('factor') or 1.0), mem.get('min_mb') or 0, mem.get('spare') or []
    if factor <= 0 or isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor < 0 \
            or not isinstance(spare, (list, tuple)) or not all(isinstance(c, str) and c for c in spare):
        raise ValueError(f'harbor.sandbox_memory must be {{factor > 0, min_mb >= 0, spare: [comm, ...]}}: {mem!r}')
    return {'factor': factor, 'min_mb': int(floor), 'spare': list(spare)}


def stack_kb(harbor):
    """The soft RLIMIT_STACK of a campaign's worker in KiB: harbor.stack_kb (0: unchanged); None for unlimited
    (harbor.stack = STACK_UNLIMITED, new campaigns); STACK_KB_LEGACY for a campaign frozen with neither."""
    harbor = harbor or {}
    if 'stack_kb' in harbor:
        return int(harbor['stack_kb'])
    if harbor.get('stack') == STACK_UNLIMITED:
        return None
    return STACK_KB_LEGACY


def attempt_env(env, harbor, task, attempt_dir):
    """Set an attempt's sandbox-limit environment from the campaign's [harbor] (in place; returns env). The disk cap's
    off is explicit (HARBOR_NS_DISKCAP=0), so a value in the worker's own environment cannot turn it on."""
    from pathlib import Path
    for k in ('PI_CAMPAIGN_MEMORY_FACTOR', 'PI_CAMPAIGN_MEMORY_MIN_MB', 'PI_CAMPAIGN_MEMORY_SPARE',
              'HARBOR_NS_DISKCAP_MIN_MB', 'HARBOR_NS_DISKCAP_LOG', 'HARBOR_NS_MEMCAP_MIN_MB', 'HARBOR_NS_MEMCAP_LOG'):
        env.pop(k, None)
    # eval's memcap (harbor_ns/memcap.py) never runs alongside synth's own PSS guard
    env['HARBOR_NS_MEMCAP'] = '0'
    mem = sandbox_memory(harbor)
    if mem:
        env['PI_CAMPAIGN_MEMORY_FACTOR'] = f"{mem['factor']:g}"
        env['PI_CAMPAIGN_MEMORY_MIN_MB'] = str(mem['min_mb'])
        env['PI_CAMPAIGN_MEMORY_SPARE'] = ','.join(mem['spare'])
    cap = diskcap_config(harbor, task)
    if not cap:
        env['HARBOR_NS_DISKCAP'] = '0'
        return env
    env['HARBOR_NS_DISKCAP'] = f"{cap['factor']:g}"
    env['HARBOR_NS_DISKCAP_MIN_MB'] = str(cap['min_mb'])
    env['HARBOR_NS_DISKCAP_LOG'] = str(Path(attempt_dir) / 'diskcap.json')
    return env

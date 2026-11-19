"""Campaign + teacher configuration.

A campaign TOML names one teacher (synth/teachers/<slug>.toml) and fixes the
protocol, task lists, targets and resources. `prepare` freezes the merged
result to <root>/ops/campaign.json; every job reads only that frozen copy, so a
running campaign never changes when the repo does.
"""
import copy
import fnmatch
import json
import os
from pathlib import Path
import tomllib

from .common import REPO, Log, atomic, read, read_task_list, sha

TEACHERS = REPO / 'synth' / 'teachers'

# New campaigns follow the eval protocol (#378 train/test audit, fix 1; user 2026-11-13: SFT traces keep the Pi finish
# notice "to avoid train/test gaps"; 2026-11-14: the default is the eval protocol, the v4 notice at a 131,072 context).
# Today's reference for it is aa-evals' pi_eval and the c30 launcher (aa_evals/campaign/launch.py:111, context_window
# 131072); eval_q35 moves to v4 at 131,072 for new runs (eval PR pending, branch eval-harness-from-data).
# `finish_notice` is frozen as the full v4 record
# (tb_agents.pi_context_limit.finish_notice_record, the same record pi_eval freezes), or None for "off". A campaign
# frozen before this change has no `finish_notice` and runs without it (v2), at its recorded context.
# Per-benchmark overrides, set by the user in the campaign TOML: [protocol] context_tokens / finish_notice for the
# whole campaign, and [[protocol.overrides]] (tasks = [glob, ...], context_tokens and/or finish_notice) for a task
# source within it (the first matching entry wins; context_tokens at most the campaign's, which the engines serve).
# Differences from eval that come from these overrides are deliberate.
FINISH_NOTICE_DEFAULT = (40000, 16384)   # WARN_AT:RESERVE, aa_evals pi_eval DEFAULT_FINISH_NOTICE
PROTOCOL_DEFAULTS = {
    'context_tokens': 131072,
    'max_output_tokens': 16384,
    'agent_timeout_sec': 21600,
    'pi_version': '1.2.0',
    'pi_resamples': 0,
    'compaction': False,
    'context_limit_action': 'submit',
    'finish_notice': 'v4',   # replaced by its frozen record below (finish_notice_config)
}
FINISH_OFF = ('off', 'none', '0', '0:0', 'false', 'no')

ADMISSION_DEFAULTS = {
    'initial_limit_per_backend': 64,
    'maximum_limit_per_backend': 160,
    'limit_step': 16,
    'minimum_limit_hold_seconds': 120,
    'emergency_kv_high': 0.85,
    'resume_kv': 0.68,
    'increase_below_kv': 0.45,
    'forecast_seconds': 180,
    'forecast_kv': 0.90,
    'max_waiting': 32,
    'burst': 8,                 # new attempts per backend per period (burst_per_backend)
    'burst_per_backend': True,
    'period_seconds': 15,
    'eval_share': 0.25,
}


def finish_notice_config(value):
    """A campaign's finish_notice setting -> the frozen record (v4 sticky) or None (off). Accepts 'v4' / 'on' / true
    (the default thresholds), 'WARN_AT:RESERVE', {warn_at, reserve} (a frozen record passes through unchanged when
    it is a v4 one), or 'off' / false / none. A non-sticky (v3) policy is refused: its traces do not record where the
    model saw the notices, so SFT cannot use them (aa_evals finish_map.run_policy)."""
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tb_agents.pi_context_limit import STICKY_POLICY, finish_notice_record
    if value is None or value is False or (isinstance(value, str) and value.strip().lower() in FINISH_OFF):
        return None
    if value is True or (isinstance(value, str) and value.strip().lower() in ('v4', 'on', 'true', 'default')):
        return finish_notice_record(*FINISH_NOTICE_DEFAULT)
    if isinstance(value, str):
        try:
            warn, reserve = (int(x) for x in value.split(':'))
        except ValueError:
            raise ValueError(f'finish_notice expects v4, WARN_AT:RESERVE or off, got {value!r}') from None
        return finish_notice_record(warn, reserve)
    if isinstance(value, dict):
        if value.get('policy', STICKY_POLICY) != STICKY_POLICY or value.get('sticky', True) is not True:
            raise ValueError(f'finish_notice must be the v4 sticky policy ({STICKY_POLICY}): {value!r}')
        record = finish_notice_record(value.get('warn_at', FINISH_NOTICE_DEFAULT[0]),
                                      value.get('reserve', FINISH_NOTICE_DEFAULT[1]))
        if 'policy' in value and value != record:
            raise ValueError(f'finish_notice record differs from this checkout\'s v4 record: {value!r} != {record!r}')
        return record
    raise ValueError(f'finish_notice expects v4, WARN_AT:RESERVE, a table or off, got {value!r}')


PROTOCOL_DEFAULTS['finish_notice'] = finish_notice_config(PROTOCOL_DEFAULTS['finish_notice'])


def check_protocol(protocol):
    """Normalize and validate a NEW campaign's protocol in place: finish_notice and every override's -> frozen records;
    override context_tokens at most the campaign's (the engines' max_model_len) and above the notice reserve."""
    def check_context(where, context, fn):
        if isinstance(context, bool) or not isinstance(context, int) or context < 1:
            raise ValueError(f'{where}: context_tokens must be a positive integer: {context!r}')
        if fn and context <= fn['final_at'] + 4096:
            raise ValueError(f'{where}: context_tokens {context} leaves no room before the finish notice '
                             f'(final turn at {fn["final_at"]} tokens left)')
    protocol['finish_notice'] = finish_notice_config(protocol.get('finish_notice'))
    check_context('[protocol]', protocol['context_tokens'], protocol['finish_notice'])
    overrides = protocol.get('overrides') or []
    if not isinstance(overrides, list):
        raise ValueError('[[protocol.overrides]] must be an array of tables')
    for i, o in enumerate(overrides):
        where = f'[[protocol.overrides]] #{i}'
        if not isinstance(o, dict):
            raise ValueError(f'{where}: each override must be a table, got {o!r}')
        unknown = set(o) - {'tasks', 'context_tokens', 'finish_notice', 'note'}
        if unknown:
            raise ValueError(f'{where}: unknown keys {sorted(unknown)} (an override sets context_tokens/finish_notice)')
        # a list of strings only: fnmatch over a bare string would match its characters, i.e. almost every task
        if not isinstance(o.get('tasks'), list) or not o['tasks'] or not all(isinstance(g, str) and g for g in o['tasks']):
            raise ValueError(f'{where}: tasks must be a non-empty list of task-name globs')
        if 'finish_notice' in o:
            o['finish_notice'] = finish_notice_config(o['finish_notice'])
        context = o.get('context_tokens', protocol['context_tokens'])
        if isinstance(context, int) and not isinstance(context, bool) and context > protocol['context_tokens']:
            raise ValueError(f'{where}: context_tokens {context} > the campaign\'s {protocol["context_tokens"]} '
                             f'(the engines serve that many)')
        check_context(where, context, o.get('finish_notice', protocol['finish_notice']))
    if overrides:
        protocol['overrides'] = overrides
    return protocol


def effective_protocol(protocol, task):
    """The context and finish notice one task's attempts run with: the first [[protocol.overrides]] entry whose globs
    match the task, else the campaign's. A frozen protocol without finish_notice (campaigns before 2026-11-14) -> None
    (no notice). -> {context_tokens, finish_notice, override (its index or None)}."""
    out = {'context_tokens': protocol['context_tokens'], 'finish_notice': protocol.get('finish_notice'), 'override': None}
    for i, o in enumerate(protocol.get('overrides') or []):
        if any(fnmatch.fnmatchcase(task, g) for g in o['tasks']):
            out.update({k: o[k] for k in ('context_tokens', 'finish_notice') if k in o}, override=i)
            break
    return out


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_teacher(slug, overrides=None):
    spec = tomllib.loads((TEACHERS / f'{slug}.toml').read_text())
    if spec.get('slug') != slug:
        raise ValueError(f'Teacher file {slug}.toml declares slug {spec.get("slug")!r}')
    spec = deep_merge(spec, overrides)
    checkpoint = spec['checkpoint']
    def expand(value):
        if isinstance(value, str):
            return value.replace('{checkpoint}', checkpoint).replace('{repo}', str(REPO))
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value
    return expand(spec)


def hung_watchdog(value):
    """harbor.hung_watchdog of a NEW campaign (#531; tb_agents/hung.py, the rule pi_eval runs): on by default with the
    most lenient idle limit of pi_eval's measured table (synthesis has no per-suite gap table) and Pi's stall limit;
    false turns it off; a table {idle_sec, stall_sec} sets the limits. Frozen campaigns without the key run without it."""
    from tb_agents.hung import FLOOR_SEC, resolve
    if value is False or value is None:
        return None
    if value is True:
        # stall_sec None: synthesis writes no wire-responses/ (PI_PROXY_CAPTURE_FILES=0), so the stall rule has no
        # evidence there; off rather than scanning for it every tick
        rec = resolve(None)
        return {**rec, 'stall_sec': None, 'source': f"{rec['source']}; synthesis default"}
    if (isinstance(value, dict) and set(value) <= {'idle_sec', 'stall_sec'}
            and float(value.get('idle_sec', 0)) >= FLOOR_SEC and float(value.get('stall_sec') or 0) >= 0):
        return {'idle_sec': float(value['idle_sec']),
                'stall_sec': float(value['stall_sec']) if value.get('stall_sec') else None,
                'p99_gap_sec': None, 'source': 'campaign harbor.hung_watchdog'}
    raise ValueError(f'harbor.hung_watchdog: true, false or {{idle_sec >= {FLOOR_SEC:g}, stall_sec >= 0}}: {value!r}')


def build(path):
    """Merge a campaign TOML with its teacher spec into the frozen form."""
    raw = tomllib.loads(Path(path).read_text())
    teacher = load_teacher(raw['teacher'], raw.get('teacher_overrides'))
    config = {k: v for k, v in raw.items() if k != 'teacher_overrides'}
    config['teacher'] = teacher
    config['protocol'] = check_protocol({**PROTOCOL_DEFAULTS, **teacher.get('protocol', {}), **raw.get('protocol', {})})
    # sandbox limits of new campaigns (synth.sandbox_limits: memory, stack and disk), under the campaign's own [harbor]
    from .sandbox_limits import new_campaign_harbor
    config['harbor'] = new_campaign_harbor(raw.get('harbor'))
    config['harbor']['hung_watchdog'] = hung_watchdog((raw.get('harbor') or {}).get('hung_watchdog', True))
    admission = {**teacher.get('admission', {}), **raw.get('admission', {})}
    if admission.get('mode') == 'kv_fit':
        # The limit controller's keys share names (emergency_kv_high, resume_kv, max_waiting) but not values.
        from .admission import DEFAULTS as KV_FIT_DEFAULTS
        limits_only = ('initial_limit_per_backend', 'maximum_limit_per_backend')
        config['admission'] = {**{k: ADMISSION_DEFAULTS[k] for k in ('period_seconds', 'eval_share')},
                               **KV_FIT_DEFAULTS, **{k: v for k, v in admission.items() if k not in limits_only}}
    else:
        config['admission'] = {**ADMISSION_DEFAULTS, **admission}
    serving = deep_merge(teacher.get('serving_resources', {}), raw.get('serving', {}))
    config['serving'] = serving
    config['workers'] = deep_merge(teacher.get('worker_resources', {}), raw.get('workers', {}))
    config['source'] = {'path': str(Path(path).resolve()), 'sha256': sha(path)}
    for kind in ('train', 'eval'):
        section = config.get(kind)
        if not section or not section.get('enabled', True):
            continue
        if kind == 'train' and section.get('stream'):
            if not Path(section['stream'].get('dir', '')).is_absolute():
                raise ValueError('[train.stream] dir must be an absolute path')
            if 'tasks' not in section:  # every task comes from the stream (synth/stream.py)
                section['task_list'] = []
                continue
        tasks = read_task_list(section['tasks'])
        if len(set(tasks)) != len(tasks):
            raise ValueError(f'Duplicate tasks in {section["tasks"]}')
        if section.get('tasks_sha256') and sha(section['tasks']) != section['tasks_sha256']:
            raise ValueError(f'{kind} task list checksum changed: {section["tasks"]}')
        section['task_list'] = tasks
        section['tasks_sha256'] = sha(section['tasks'])
    return config


class Campaign:
    """Paths and settings of one frozen campaign."""

    def __init__(self, config):
        self.config = config
        self.id = config['id']
        self.teacher = config['teacher']
        self.slug = self.teacher['slug']
        self.root = Path(config['root']) / self.id
        self.ops = self.root / 'ops'
        self.train = self.root / 'train'
        self.eval = self.root / 'eval'
        self.smoke = self.root / 'smoke'
        self.protocol = config['protocol']
        self.log = Log(self.ops / 'status.log')

    @classmethod
    def load(cls, path=None):
        path = Path(path or os.environ['SYNTH_CAMPAIGN'])
        return cls(json.loads(path.read_text()))

    @property
    def frozen_path(self):
        return self.ops / 'campaign.json'

    def freeze(self):
        existing = read(self.frozen_path)
        if existing and existing != self.config:
            raise ValueError(f'{self.frozen_path} exists with different settings; use a new campaign id')
        atomic(self.frozen_path, self.config)
        return self.frozen_path

    def root_for(self, kind):
        return {'train': self.train, 'eval': self.eval, 'smoke': self.smoke}[kind]

    def tasks(self, kind):
        section = self.config.get(kind) or {}
        return section.get('task_list', []) if section.get('enabled', True) else []

    def enabled(self, kind):
        section = self.config.get(kind)
        return bool(section) and section.get('enabled', True)

    @property
    def eval_attempts(self):
        return int((self.config.get('eval') or {}).get('attempts_per_task', 3))

    @property
    def eval_expected(self):
        return len(self.tasks('eval')) * self.eval_attempts if self.enabled('eval') else 0

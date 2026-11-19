"""Frozen, task-scoped Pi runtime dispatch. Legacy configs retain their original class."""
from tb_agents.pi_musl_compat import TASK

STANDARD = 'PiCont16kEnvClock'
MUSL = 'PiCont16kEnvClockMuslCompat'
ALLOWED = {TASK: MUSL}
FIELD = 'task_agent_classes'
MARKER = 'PI_TASK_AGENT_CLASS_TASK'


def selected(declared, names):
    if not isinstance(declared, dict) or any(ALLOWED.get(task) != cls for task, cls in declared.items()):
        raise ValueError('unsupported task-specific Pi agent class declaration')
    return {task: cls for task, cls in declared.items() if task in names}


def frozen(config):
    policy = config.get(FIELD)
    protocol = config.get('protocol') or {}
    if policy is None:
        if FIELD in protocol:
            raise ValueError('Pi task class protocol lacks its recorded dispatch')
        return None
    if selected(policy, config['task_names']) != policy:
        raise ValueError('Pi task class dispatch contains an unselected task')
    if protocol.get(FIELD) != policy:
        raise ValueError('Pi config and protocol task class dispatch disagree')
    if not config.get('envclock'):
        raise ValueError('Pi task class dispatch requires the recorded environment clock protocol')
    return policy


def apply_env(config, task, env):
    policy = frozen(config)
    if policy is None:
        return  # no automatic runtime upgrade for historical stored runs
    if task not in config['task_names']:
        raise ValueError('Pi task class dispatch received an unselected task')
    expected = policy.get(task, STANDARD)
    current = env.get('PI_CONT16K_CLASS', STANDARD)
    if current not in (STANDARD, expected):
        raise ValueError('PI_CONT16K_CLASS conflicts with frozen task-specific Pi dispatch')
    marker = task if expected != STANDARD else None
    if MARKER in env and env[MARKER] != marker:
        raise ValueError('reserved task class marker conflicts with frozen Pi dispatch')
    env['PI_CONT16K_CLASS'] = expected
    if marker:
        env[MARKER] = marker

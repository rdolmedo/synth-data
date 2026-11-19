"""Versioned task-scoped evaluator compatibility overrides; never migrate stored configs."""
import hashlib
import json


FLAG = 'HARBOR_NS_SKIP_PIP_PROXY_SEED'


def digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def new_policy(benchmark, agent, agent_internet, tasks):
    if benchmark != 'tb21' or agent != 'pi' or agent_internet is not False:
        return None
    # This task explicitly requires pip --index-url localhost. An evaluator-added
    # pip global proxy bypasses NO_PROXY and redirects that local installation.
    return {'schema_version': 1, 'overrides': {'pypi-server': {FLAG: '1'}} if 'pypi-server' in tasks else {}}


def validate(config):
    protocol = config.get('protocol') or {}
    if not isinstance(protocol, dict):
        raise ValueError('task environment policy needs a protocol object')
    names = ('task_environment_policy', 'task_environment_policy_sha256')
    present = [name in source for source in (config, protocol) for name in names]
    if not any(present):
        return None   # legacy stored configurations keep their original environment
    if not all(present):
        raise ValueError('task environment policy/digest must be present in config and protocol')
    policy = config['task_environment_policy']
    names = config.get('task_names')
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names) or len(names) != len(set(names)):
        raise ValueError('task environment policy needs the actual selected task list')
    expected = new_policy(config.get('benchmark'), config.get('agent'), config.get('agent_internet'), names)
    if (not isinstance(policy, dict) or type(policy.get('schema_version')) is not int or policy != expected
            or config.get('task_environment_policy_sha256') != digest(policy)
            or protocol.get('task_environment_policy') != policy
            or digest(protocol.get('task_environment_policy')) != digest(policy)
            or protocol.get('task_environment_policy_sha256') != digest(policy)):
        raise ValueError('unreviewed or inconsistent task environment policy')
    return policy


def overrides_for(config, task):
    policy = validate(config)
    if policy is None:
        return {}
    if task not in config['task_names']:
        raise ValueError('task environment dispatch escaped the selected task list')
    return dict(policy['overrides'].get(task, {}))

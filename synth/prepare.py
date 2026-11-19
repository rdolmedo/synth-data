"""Freeze a campaign, build its task bundles and protocol record, and start its controller.

  python -m synth.prepare synth/campaigns/<name>.toml [--submit] [--dry-run]
  python -m synth.prepare <root>/<id>/ops/campaign.json --submit-only   (from a frozen run checkout)

Bundles keep only the selected tasks and change only agent limits: [agent] and
every [steps.agent] timeout become protocol.agent_timeout_sec; verifier and
build limits are the originals.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import tomllib

from . import condor
from .common import REPO, atomic, read, sha
from .config import Campaign, build

AGENT_TIMEOUT = re.compile(rb'(?ms)(^\[(?:agent|steps\.agent)\]\s*\n(?:(?!^\[).)*?^timeout_sec\s*=\s*)[0-9.]+')


def rewrite_agent_timeouts(before, timeout):
    after, count = AGENT_TIMEOUT.subn(rb'\g<1>' + str(timeout).encode(), before)
    old, new = tomllib.loads(before.decode()), tomllib.loads(after.decode())
    expected = copy.deepcopy(old)
    if 'agent' in expected:
        expected['agent']['timeout_sec'] = timeout
    for step in expected.get('steps', []):
        if 'agent' in step and 'timeout_sec' in step['agent']:
            step['agent']['timeout_sec'] = timeout
    if new != expected:
        raise ValueError('Agent timeout rewrite changed more than agent limits')
    if 'timeout_sec' not in old.get('agent', {}):
        raise ValueError('Task has no [agent].timeout_sec')
    return after, count, old


def bundle(campaign, kind):
    section = campaign.config[kind]
    selected = set(campaign.tasks(kind))
    target = campaign.ops / f'{kind}_tasks.tar'
    audit_path = campaign.ops / f'{kind}_bundle_audit.json'
    if target.exists():
        return read(audit_path)
    audit = rewrite_bundle(Path(section['bundle_source']), selected, target, campaign.protocol['agent_timeout_sec'], kind)
    audit.update(path=str(target), sha256=sha(target), count=len(selected))
    atomic(audit_path, audit)
    return audit


def rewrite_bundle(source, selected, target, timeout, kind='train'):
    """The `selected` tasks of the source tar, agent limits rewritten to `timeout`, into target. -> the audit (the
    caller adds path, sha256 and count). Also used for stream chunks (synth.stream.admit)."""
    audit = {'source': str(source), 'source_sha256': sha(source), 'agent_timeout_sec': timeout,
             'verifier_limits_preserved': True, 'tasks': {}}
    tmp = target.with_suffix('.partial')
    with tarfile.open(source) as src, tarfile.open(tmp, 'w') as dst:
        for member in src:
            parts = Path(member.name).parts
            if len(parts) < 2 or parts[0] != 'tasks' or parts[1] not in selected:
                continue
            data = src.extractfile(member) if member.isfile() else None
            if member.name == f'tasks/{parts[1]}/task.toml':
                before = data.read()
                after, count, old = rewrite_agent_timeouts(before, timeout)
                audit['tasks'][parts[1]] = {'source_config_sha256': sha_bytes(before), 'config_sha256': sha_bytes(after),
                    'agent_timeouts_rewritten': count, 'original_agent': old.get('agent'),
                    'original_step_agents': [s.get('agent') for s in old.get('steps', [])],
                    'verifier': old.get('verifier'), 'environment': old.get('environment')}
                member = copy.copy(member)
                member.size = len(after)
                data = io.BytesIO(after)
            dst.addfile(member, data)
    missing = selected - set(audit['tasks'])
    if missing:
        raise ValueError(f'{kind} tasks missing from {source}: {sorted(missing)}')
    tmp.replace(target)
    return audit


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def runtime_packages(teacher):
    python = Path(teacher['runtime']['venv']) / 'bin' / 'python'
    out = subprocess.check_output([str(python), '-c', 'import importlib.metadata,json; print(json.dumps('
                                   '{d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}))'],
                                  text=True)
    return json.loads(out)


def protocol_record(campaign):
    sys.path.insert(0, str(REPO))
    from tb_agents.pi_context_limit import finish_policy
    from tb_agents.pi_continuation import CACHE, bundle_provenance, verify_bundle
    teacher, protocol = campaign.teacher, campaign.protocol
    fn = protocol.get('finish_notice')
    pi_cache = Path(protocol.get('pi_cache') or CACHE)
    pi_sha = verify_bundle(pi_cache)
    if protocol.get('pi_bundle_sha256') and pi_sha != protocol['pi_bundle_sha256']:
        raise ValueError(f'Pi bundle {pi_sha} != pinned {protocol["pi_bundle_sha256"]}')
    packages = runtime_packages(teacher)
    expected = teacher['runtime'].get('vllm_version')
    if expected and packages.get('vllm') != expected:
        raise ValueError(f'vLLM {packages.get("vllm")} in {teacher["runtime"]["venv"]} != pinned {expected}')
    atomic(campaign.ops / 'runtime-packages.json', packages)
    checkpoint = Path(teacher['checkpoint'])
    files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in checkpoint.iterdir()
             if not p.name.endswith('.safetensors') and p.is_file() and p.stat().st_size < 64 << 20}
    record = {'campaign': campaign.id, 'teacher': campaign.slug, 'model': teacher['model'], 'revision': teacher['revision'],
              'checkpoint': str(checkpoint), 'checkpoint_files': files, 'served_name': teacher['served_name'],
              'runtime_packages': {'path': str(campaign.ops / 'runtime-packages.json'),
                                   'sha256': sha(campaign.ops / 'runtime-packages.json'), 'vllm': packages.get('vllm')},
              'serving': teacher['serving'], 'request': teacher.get('request'), 'counting': teacher['counting'],
              'pi': {'version': protocol['pi_version'], 'bundle_sha256': pi_sha, 'resamples': protocol['pi_resamples'],
                     'bash_default_timeout_sec': bundle_provenance(pi_cache, pi_sha)['bash_default_timeout_sec'],
                     'proxy_continuations': 0, 'context_window': protocol['context_tokens'],
                     'max_output_tokens': protocol['max_output_tokens'], 'compaction_enabled': protocol['compaction'],
                     'context_limit_action': protocol['context_limit_action'],
                     'context_policy': finish_policy(fn['warn_at'], fn['reserve'], fn.get('sticky')) if fn
                     else finish_policy(0, 0, False),
                     'finish_notice': fn, 'protocol_overrides': protocol.get('overrides') or [],
                     'turn_limit': None},
              'sandbox': {k: (campaign.config.get('harbor') or {}).get(k) for k in ('stack', 'stack_kb', 'sandbox_memory', 'diskcap')
                          if k in (campaign.config.get('harbor') or {})},
              'agent_timeout_sec': protocol['agent_timeout_sec'], 'verifier_limits': 'unchanged from each original bundle',
              'seeds': 'SHA256(campaign/kind/task/attempt) mod 2^31; infrastructure retries reuse the attempt seed',
              'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              'created_at': time.time()}
    atomic(campaign.ops / 'protocol.json', record)
    return record


def check_smokes(campaign):
    """Each smoke runs from its kind's task bundle, which holds only that kind's task list: a smoke task outside it
    fails in Harbor ("Either datasets or tasks must be provided") at the first tick (2026-10-25). With a task stream a
    train smoke task may come in a chunk instead; the controller waits for it."""
    stream = (campaign.config.get('train') or {}).get('stream')
    for spec in (campaign.config.get('smoke') or {}).get('attempts', []):
        kind = spec.get('kind', 'train')
        if spec['task'] in set(campaign.tasks(kind)):
            continue
        if kind == 'train' and stream:
            print(f'smoke task {spec["task"]} is not in the frozen train list; the smoke waits for its stream chunk')
            continue
        raise SystemExit(f'smoke task {spec["task"]} is not in the {kind} task list, so not in its bundle')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign', help='synth/campaigns/<name>.toml')
    ap.add_argument('--submit', action='store_true', help='submit the controller job after preparing')
    ap.add_argument('--dry-run', action='store_true', help='print the frozen configuration only')
    ap.add_argument('--submit-only', action='store_true', help='submit the controller of an already frozen campaign')
    ap.add_argument('--generation', type=int, help='controller generation (default: latest receipt + 1)')
    a = ap.parse_args()
    if a.submit_only:
        campaign = Campaign.load(a.campaign)
        if a.generation is None:
            latest = condor.latest(campaign, 'controller')
            a.generation = latest['generation'] + 1 if latest else 0
        print(condor.submit_controller(campaign, a.generation))
        return
    campaign = Campaign(build(a.campaign))
    check_smokes(campaign)
    if a.dry_run:
        print(json.dumps(campaign.config, indent=2))
        return
    campaign.ops.mkdir(parents=True, exist_ok=True)
    campaign.freeze()
    stream = (campaign.config.get('train') or {}).get('stream')
    for kind in ('train', 'eval'):
        if campaign.enabled(kind) and (campaign.tasks(kind) or not (kind == 'train' and stream)):
            audit = bundle(campaign, kind)
            campaign.log(f'{kind} bundle: {audit["count"]} tasks, sha256 {audit["sha256"]}')
    if stream:
        campaign.log(f'train task stream: {stream["dir"]} (chunks are admitted while the campaign runs)')
    protocol_record(campaign)
    if 'control' in campaign.config.get('serving', {}) and not (campaign.ops / 'serving' / 'control.json').exists():
        atomic(campaign.ops / 'serving' / 'control.json', {'generation': 0, **campaign.teacher['serving'].get('tunables', {}),
                                                            **campaign.config['serving']['control']})
    elif not (campaign.ops / 'serving' / 'control.json').exists():
        atomic(campaign.ops / 'serving' / 'control.json', {'generation': 0, **campaign.teacher['serving'].get('tunables', {})})
    campaign.log(f'Prepared campaign {campaign.id} for {campaign.slug} at {campaign.root}')
    if a.submit:
        latest = condor.latest(campaign, 'controller')
        print(condor.submit_controller(campaign, latest['generation'] + 1 if latest else 0))


if __name__ == '__main__':
    main()

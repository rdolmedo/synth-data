"""Generic dependency-cache integrity and isolation; no real task text, solution or network."""
import asyncio
import copy
import json
import os
from pathlib import Path
import pytest
from aa_evals.gdpval import runtime_cache as C
from aa_evals.gdpval.tasks import Task


def fixture(tmp_path):
    tasks = [Task(t, 'sector', 'occupation', 'neutral fixture instruction') for t in sorted(C.REQUIRED_TASKS)]
    root = tmp_path / 'payload'
    for rel in ('runtime/node.tar.xz', 'npm/_cacache/content-v2/sha512/object'):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'generic public artifact fixture')
    sif = tmp_path / 'container.sif'
    sif.write_bytes(b'neutral pinned container fixture')
    entries = {t.task_id: {'requested': ['react@18.3.1'], 'payload_root': str(root),
               'instruction_sha256': C.instruction_sha(t), 'task_prompt_sha256': C.prompt_sha(t),
               'sif_sha256': C.sha(sif), 'classification': 'explicit-required-generic-dependencies',
               'files': C.inventory(root)} for t in tasks}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'cache_version': 'neutral-fixture', 'tasks': entries}))
    cfg = {'sandbox': {'sif_sha256': C.sha(sif), 'sif_local': str(sif)}}
    cfg['sandbox']['runtime_packages'] = C.freeze(manifest, tasks, C.sha(sif))
    return tasks, root, manifest, cfg


def test_local_install_seed_preserves_generic_cache_and_host_environment(tmp_path):
    tasks, root, manifest, cfg = fixture(tmp_path)
    before = C.inventory(root)
    for path in root.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    environment = dict(os.environ)
    home = tmp_path / 'home'
    home.mkdir()
    result = C.prepare(cfg, tasks[0], home)
    assert result['environment']['NPM_CONFIG_OFFLINE'] == 'true'
    assert result['environment']['NPM_CONFIG_CACHE'] == '/home/user/.cache/aa-npm'
    local = home / '.cache/aa-npm/_cacache/content-v2/sha512/object'
    local.write_bytes(b'agent-local mutation')
    assert C.inventory(root) == before
    assert dict(os.environ) == environment
    from aa_evals.gdpval.sandbox import ApptainerCodeExecToolProvider
    provider = ApptainerCodeExecToolProvider(sif=cfg['sandbox']['sif_local'], home=home,
                                            tmp=tmp_path / 'tmp', runtime_cache=result)
    argv = provider.argv('npm install --offline react@18.3.1')
    assert argv[:5] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net', '--']
    assert str(root) + ':/opt/aa-runtime-cache:ro' in argv
    assert 'NPM_CONFIG_OFFLINE=true' in argv
    assert not any('proxy=' in x.lower() for x in argv)
    assert dict(os.environ) == environment


@pytest.mark.parametrize('mutation', ['prompt', 'sif', 'payload', 'extra_file', 'symlink', 'manifest'])
def test_changed_frozen_inputs_refused_before_install(tmp_path, mutation):
    tasks, root, manifest, cfg = fixture(tmp_path)
    task = tasks[0]
    if mutation == 'prompt': task.prompt += ' changed'
    elif mutation == 'sif': Path(cfg['sandbox']['sif_local']).write_bytes(b'changed actual SIF')
    elif mutation == 'payload': (root / 'runtime/node.tar.xz').write_bytes(b'changed runtime')
    elif mutation == 'extra_file': (root / 'extra').write_bytes(b'unreviewed')
    elif mutation == 'symlink': (root / 'escape').symlink_to(tmp_path / 'container.sif')
    else: manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError): C.prepare(cfg, task, tmp_path / 'home')
    assert not (tmp_path / 'home/.cache/aa-npm').exists()


def test_required_task_without_manifest_fails_closed_but_business_task_is_unchanged(tmp_path):
    tasks, root, manifest, cfg = fixture(tmp_path)
    cfg['sandbox'].pop('runtime_packages')
    with pytest.raises(ValueError, match='Required dependency cache missing'):
        C.prepare(cfg, tasks[0], tmp_path / 'home')
    task = Task('neutral-business-task', 'sector', 'occupation', 'neutral fixture')
    assert C.prepare(cfg, task, tmp_path / 'home') is None
    assert C.freeze(None, [task], cfg['sandbox']['sif_sha256'])['mode'] == 'none-required'


def test_unsafe_inventory_path_is_rejected(tmp_path):
    tasks, root, manifest, cfg = fixture(tmp_path)
    obj = json.loads(manifest.read_text())
    obj['tasks'][tasks[0].task_id]['files']['../escape'] = 'a' * 64
    with pytest.raises(ValueError, match='unsafe'):
        C.checked_entry(obj, tasks[0], cfg['sandbox']['sif_sha256'])


def test_setup_integrity_failure_is_infra_without_model_client(tmp_path, monkeypatch):
    from aa_evals.gdpval import agent
    from aa_evals.gdpval.network import POLICY
    tasks, root, manifest, cfg = fixture(tmp_path)
    cfg['agent_network'] = dict(POLICY)
    (root / 'runtime/node.tar.xz').write_bytes(b'corrupt artifact')
    def no_client(**kwargs):
        raise AssertionError('model client must never be constructed after invalid setup')
    monkeypatch.setattr(agent, 'CappedVLLMClient', no_client)
    result = asyncio.run(agent.run_attempt(tasks[0], 0, cfg=cfg, work=tmp_path / 'work', out_dir=tmp_path / 'out'))
    assert result['status'] == 'infra_error' and result['end'] == 'infra'
    assert result['turns'] == 0 and result['n_submitted'] == 0
    assert 'runtime cache setup' in result['error']
    assert (tmp_path / 'out/transcript.json.gz').exists()

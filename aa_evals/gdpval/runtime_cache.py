"""Generic evaluator-cached dependencies; agents install them locally without egress.

No task implementation, reference answer, data, credential or live service is cached.
The manifest and each payload are immutable inputs to a new generation protocol.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

DEFAULT_MANIFEST = Path('/data/aai/aa_evals/runtime_packages/gdpval/v1/manifest.json')
REQUIRED_TASKS = frozenset({
    '7de33b48-5163-4f50-b5f3-8deea8185e57',
    '0e386e32-df20-4d1f-b536-7159bc409ad5',
    '4122f866-01fa-400b-904d-fa171cdab7c7',
})
VERSION = 1
MOUNT = '/opt/aa-runtime-cache'
_SIF_VERIFIED = set()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def instruction_sha(task) -> str:
    return hashlib.sha256(task.prompt.encode()).hexdigest()


def prompt_sha(task) -> str:
    from aa_evals.gdpval.tasks import task_prompt
    return hashlib.sha256(task_prompt(task).encode()).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError('runtime cache payload root must be a real directory')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('runtime cache contains a symlink')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
        elif not path.is_dir():
            raise ValueError('runtime cache contains a special file')
    return result


def load_manifest(path: Path, expected_sha: str) -> dict:
    if path.is_symlink() or sha(path) != expected_sha:
        raise ValueError('runtime cache manifest hash mismatch or symlink')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != VERSION or set(manifest.get('tasks', {})) != REQUIRED_TASKS:
        raise ValueError('runtime cache manifest schema/task coverage mismatch')
    return manifest


def checked_entry(manifest: dict, task, sif_sha256: str) -> dict | None:
    entry = manifest['tasks'].get(task.task_id)
    if entry is None:
        return None
    if (entry.get('instruction_sha256') != instruction_sha(task) or
            entry.get('task_prompt_sha256') != prompt_sha(task) or entry.get('sif_sha256') != sif_sha256):
        raise ValueError('runtime cache task instruction/prompt/SIF identity mismatch')
    if entry.get('classification') != 'explicit-required-generic-dependencies':
        raise ValueError('runtime cache classification mismatch')
    files = entry.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('runtime cache requires an exact nonempty file inventory')
    for rel, digest in files.items():
        p = PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or str(p) != rel or len(digest) != 64:
            raise ValueError('unsafe runtime cache inventory path/hash')
    root = Path(entry['payload_root'])
    if not root.is_absolute() or inventory(root) != files:
        raise ValueError('runtime cache payload file inventory/hash mismatch')
    if 'runtime/node.tar.xz' not in files or not any(p.startswith('npm/_cacache/') for p in files):
        raise ValueError('runtime cache lacks Node archive/npm dependency closure')
    return entry


def freeze(path, tasks, sif_sha256: str) -> dict:
    """Called during native init, before workers/model clients are started."""
    needs = {t.task_id for t in tasks} & REQUIRED_TASKS
    if not path and not needs:
        return {'version': VERSION, 'mode': 'none-required', 'tasks': []}
    path = Path(path or DEFAULT_MANIFEST).absolute()
    if not path.exists():
        raise ValueError(f'Explicit required dependencies need evaluator setup first: {path}; '
                         'pass --runtime-cache-manifest with a reviewed immutable manifest')
    digest = sha(path)
    manifest = load_manifest(path, digest)
    for task in tasks:
        checked_entry(manifest, task, sif_sha256)
    return {'version': VERSION, 'mode': 'offline-attempt-local-install', 'manifest': str(path),
            'manifest_sha256': digest, 'cache_version': manifest['cache_version'],
            'tasks': sorted(REQUIRED_TASKS)}


def prepare(cfg: dict, task, home: Path) -> dict | None:
    """Verify frozen inputs and copy only generic npm cache into the writable attempt home."""
    frozen = cfg['sandbox'].get('runtime_packages')
    if not frozen or frozen.get('mode') == 'none-required':
        if task.task_id in REQUIRED_TASKS:
            raise ValueError('Required dependency cache missing; use a new initialized run')
        return None
    if frozen.get('version') != VERSION or frozen.get('mode') != 'offline-attempt-local-install':
        raise ValueError('Unknown runtime cache generation protocol')
    manifest = load_manifest(Path(frozen['manifest']), frozen['manifest_sha256'])
    entry = checked_entry(manifest, task, cfg['sandbox']['sif_sha256'])
    if entry is None:
        return None
    sif = Path(cfg['sandbox']['sif_local'])
    stat = sif.stat()
    stamp = (str(sif.resolve()), stat.st_size, stat.st_mtime_ns, entry['sif_sha256'])
    if stamp not in _SIF_VERIFIED:
        if sha(sif) != entry['sif_sha256']:
            raise ValueError('runtime cache actual staged SIF hash mismatch')
        _SIF_VERIFIED.add(stamp)
    root = Path(entry['payload_root'])
    npm = home / '.cache/aa-npm'
    npm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / 'npm', npm)
    expected_npm = {p[len('npm/'):]: h for p, h in entry['files'].items() if p.startswith('npm/')}
    if inventory(npm) != expected_npm:
        raise ValueError('runtime npm cache changed during attempt-local staging')
    for path in npm.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    npm.chmod(0o700)
    # Installation writes belong only to this attempt. No package-manager setting reaches the host/grader.
    env = {'PATH': '/home/user/.local/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'NPM_CONFIG_CACHE': '/home/user/.cache/aa-npm', 'NPM_CONFIG_OFFLINE': 'true',
           'NPM_CONFIG_AUDIT': 'false', 'NPM_CONFIG_FUND': 'false', 'NPM_CONFIG_UPDATE_NOTIFIER': 'false',
           'PIP_NO_INDEX': '1'}
    if any(p.startswith('terraform-providers/') for p in entry['files']):
        rc = home / '.config/aa-runtime-cache/terraform.rc'
        rc.parent.mkdir(parents=True, exist_ok=True)
        rc.write_text('provider_installation {\n  filesystem_mirror {\n    path = "' + MOUNT +
                      '/terraform-providers"\n  }\n}\ndisable_checkpoint = true\n')
        env['TF_CLI_CONFIG_FILE'] = '/home/user/.config/aa-runtime-cache/terraform.rc'
        env['CHECKPOINT_DISABLE'] = '1'
    receipt = {'version': VERSION, 'manifest_sha256': frozen['manifest_sha256'],
               'cache_version': frozen['cache_version'], 'task_id': task.task_id,
               'instruction_sha256': entry['instruction_sha256'], 'sif_sha256': entry['sif_sha256'],
               'files': len(entry['files']), 'requested': entry['requested'], 'mode': 'offline-attempt-local-install'}
    notice = ('\n\nEvaluator setup cached only generic dependencies required by this task. Internet remains disabled. '
              f'Read-only artifacts are at {MOUNT}. To install the supplied Node/npm runtime locally, create '
              '/home/user/.local and unpack /opt/aa-runtime-cache/runtime/node.tar.xz there with one leading '
              'archive directory removed. Python standard-library tarfile/lzma can unpack this archive; the external xz '
              'command is absent. Ignore archive ownership and preserve executable modes when extracting: upstream '
              'numeric owners are not mapped in this namespace. /home/user/.local/bin is on PATH. npm uses an attempt-local seeded '
              'cache and offline mode; install needed cached packages into your own project. '
              'Available pinned package selectors: ' + ', '.join(entry['requested']) + '. '
              'Other supplied runtime archives, if any, are under /opt/aa-runtime-cache/runtime. '
              'No task implementation, external data or credentials are supplied.')
    return {'payload_root': str(root), 'environment': env, 'receipt': receipt, 'notice': notice}

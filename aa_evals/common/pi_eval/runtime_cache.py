"""Public package and raw input caches staged by evaluator setup.

Payloads and reviewed per-task manifests live outside git. Config records the manifest
SHA256; every staged file is checked before it enters the task's private filesystem.
Packages stay uninstalled. The agent's existing network isolation is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex

REMOTE = '/opt/aa-runtime-cache'
# Immutable reviewed public-package release; validation receipts are outside git.
DEFAULT_MANIFEST = '/data/aai/aa_evals/runtime_packages/pi/v5/manifest.json'
DEFAULT_MANIFEST_SHA256 = 'd7fcf56be3d5c3d79d7d50e8b059772d4fcc2df11a22f39ef6f67737145fc317'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(path, expected_sha256):
    path = Path(path)
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or '') or sha256(path) != expected_sha256:
        raise ValueError('runtime package manifest SHA256 mismatch')
    data = json.loads(path.read_text())
    version = data.get('schema_version')
    if type(version) is not int or version not in (1, 2) or not isinstance(data.get('tasks'), dict):
        raise ValueError('unsupported runtime package manifest')
    if version == 2 and not isinstance(data.get('public_inputs'), dict):
        raise ValueError('public input manifest needs a task map')
    if version == 1 and 'public_inputs' in data:
        raise ValueError('public inputs require manifest schema version 2')
    return data


def validate_payload(entry):
    root = Path(entry['payload_root'])
    if root.is_symlink() or not root.is_dir():
        raise ValueError('runtime cache root must be a regular directory')
    files = entry['files']
    if 'directories' in entry:
        actual_dirs = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_dir() and not p.is_symlink()}
        if actual_dirs != set(entry['directories']):
            raise ValueError('runtime cache directory inventory differs from pinned manifest')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if not files or actual != set(files):
        raise ValueError('runtime cache file inventory differs from pinned manifest')
    for rel, expected in files.items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('unsafe runtime cache path')
        p = root / rel
        if p.is_symlink() or not p.is_file() or sha256(p) != expected:
            raise ValueError(f'runtime cache payload SHA256 mismatch: {rel}')
    return root


async def prepare(environment, manifest_path, manifest_sha256):
    """Evaluator-only setup. Returns a factual notice; does not install requested packages."""
    if not manifest_path:
        return None
    manifest = load_manifest(manifest_path, manifest_sha256)
    # Validate both selected payloads before uploading either. Legacy v1 calls
    # keep their existing package-only behavior and return value.
    from . import public_inputs
    inputs = await public_inputs.validate(environment, manifest)
    packages = await _prepare_packages(environment, manifest, manifest_sha256)
    if inputs is None:
        return packages
    receipt = await public_inputs.stage(environment, inputs)
    if packages is None:
        packages = {'task': receipt['task'], 'manifest_sha256': manifest_sha256,
                    'managers': [], 'requested': [], 'agent_env': {},
                    'native_rootfs_key': receipt['native_rootfs_key'], 'notice': ''}
    packages['public_inputs'] = receipt
    packages['notice'] = '\n'.join(x for x in (packages['notice'], receipt['notice']) if x)
    return packages


async def _prepare_packages(environment, manifest, manifest_sha256):
    task = environment.environment_dir.parent.name
    entry = manifest['tasks'].get(task)
    if entry is None:
        return None
    instruction = environment.environment_dir.parent / 'instruction.md'
    if sha256(instruction) != entry['instruction_sha256']:
        raise ValueError(f'{task}: runtime cache was reviewed for a different task instruction')
    root = validate_payload(entry)
    managers = entry.get('managers') or [entry['manager']]
    if not managers or not set(managers) <= {'apt', 'pip', 'cran', 'source'}:
        raise ValueError(f'unsupported runtime cache managers {managers}')
    required = set()
    if 'apt' in managers:
        required.update({'apt/apt.conf', 'apt/sources.list', 'apt/Packages', 'apt/Release'})
    if 'cran' in managers:
        required.update({'cran/Rprofile', 'cran/src/contrib/PACKAGES'})
    if not required <= entry['files'].keys():
        raise ValueError('runtime cache is missing package-manager metadata')
    if 'pip' in managers and not any(x.startswith('wheels/') for x in entry['files']):
        raise ValueError('runtime Python package cache is empty')
    # Bind cache to the actual native image distribution and package ABI.
    actual_rootfs = getattr(getattr(environment, '_main', None), 'rootfs', None)
    if actual_rootfs is None or Path(actual_rootfs).parent.name != entry.get('native_rootfs_key'):
        raise ValueError(f'{task}: runtime package cache native image key mismatch')
    probe = await environment.exec('cat /etc/os-release; printf "\\nAA_DPKG_STATUS_SHA256="; '
                                   'if test -f /var/lib/dpkg/status; then sha256sum /var/lib/dpkg/status | cut -d" " -f1; else echo none; fi', user='root')
    if probe.return_code or entry.get('image_probe') != probe.stdout:
        raise ValueError(f'{task}: runtime package cache image/installed-package inventory mismatch')
    await environment.upload_dir(root, REMOTE)
    # Package-manager configuration is agent-process-local. Evaluator grading must
    # retain its original repositories/network rather than inherit this cache policy.
    result = await environment.exec('chmod -R a+rX ' + shlex.quote(REMOTE), user='root')
    if result.return_code:
        raise RuntimeError('could not expose runtime cache: ' + result.stderr)
    agent_env = {}
    if 'apt' in managers:
        agent_env['APT_CONFIG'] = REMOTE + '/apt/apt.conf'
    if 'pip' in managers:
        agent_env.update(PIP_NO_INDEX='1', PIP_FIND_LINKS=REMOTE + '/wheels')
    if 'cran' in managers:
        agent_env['R_PROFILE_USER'] = REMOTE + '/cran/Rprofile'
    git_sources = [s for s in entry.get('upstream_sources', []) if s.get('kind') == 'git']
    if git_sources:
        settings = []
        for source in git_sources:
            filename = source['filename']
            if not re.fullmatch(r'[A-Za-z0-9_.+-]+', filename) or not source['url'].startswith('https://'):
                raise ValueError('unsafe public Git cache descriptor')
            mirror = root / 'sources' / filename
            if not all((mirror / x).is_file() for x in ('HEAD', 'config')) or not all((mirror / x).is_dir() for x in ('objects', 'refs')):
                raise ValueError('public Git mirror is incomplete')
            local = REMOTE + '/sources/' + source['filename']
            for url in dict.fromkeys([source['url'], source['url'].removesuffix('.git')]):
                settings.append(('url.file://' + local + '.insteadOf', url))
        agent_env['GIT_CONFIG_COUNT'] = str(len(settings))
        for i, (key, value) in enumerate(settings):
            agent_env[f'GIT_CONFIG_KEY_{i}'] = key
            agent_env[f'GIT_CONFIG_VALUE_{i}'] = value
    # CRAN/source trees are exposed as files; the agent performs installation/build.
    packages = ', '.join(entry['requested'])
    notice = f'Runtime package cache: {REMOTE}. Public cached packages: {packages}. Package versions, upstream sources and file hashes are listed in {REMOTE}/public-packages.json. '
    if 'apt' in managers:
        notice += 'APT is configured for the local cache. Install packages with apt-get update and apt-get install; no internet is needed. '
    if 'pip' in managers:
        notice += 'pip is configured for local package files only. Install the requested packages with pip; no internet is needed. '
    if 'cran' in managers:
        notice += f'Public R package sources and metadata are under {REMOTE}/cran; install from the local repository. '
    if 'source' in managers:
        notice += f'Unmodified public upstream archives are under {REMOTE}/sources; configure, build and install them yourself. '
    if git_sources:
        notice += 'The explicitly required upstream Git URLs resolve to their local source mirrors.'
    return {'task': task, 'manifest_sha256': manifest_sha256, 'managers': managers,
            'requested': entry['requested'], 'native_rootfs_key': entry['native_rootfs_key'], 'notice': notice, 'agent_env': agent_env}

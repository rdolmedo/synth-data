"""Stage reviewed raw public task inputs; never compute or install task outputs."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
import stat

from .runtime_cache import sha256

REMOTE = '/opt/aa-public-inputs'
KINDS = {'model', 'tokenizer', 'dataset', 'archive', 'api_snapshot', 'historical_results'}
SHA = re.compile(r'[0-9a-f]{64}')


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or str(PurePosixPath(value)) != value or PurePosixPath(value).is_absolute()
            or '..' in PurePosixPath(value).parts or value == '.'):
        raise ValueError('unsafe public input path')
    return value


def _inventory(entry):
    root = Path(entry['payload_root'])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('public input root must be an absolute regular directory')
    files, directories = entry.get('files'), entry.get('directories')
    if not isinstance(files, dict) or not files or not isinstance(directories, list):
        raise ValueError('public input inventory is missing')
    if (len(directories) != len(set(directories))
            or any(_relative(x) != x for x in directories)):
        raise ValueError('invalid public input directory inventory')
    actual_files, actual_dirs = set(), set()
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        rel = str(path.relative_to(root))
        if stat.S_ISDIR(mode):
            actual_dirs.add(rel)
        elif stat.S_ISREG(mode):
            actual_files.add(rel)
        else:
            raise ValueError('public input inventory contains a link or special file')
    if actual_files != set(files) or actual_dirs != set(directories):
        raise ValueError('public input inventory differs from the pinned manifest')
    for rel, expected in files.items():
        _relative(rel)
        if not isinstance(expected, str) or not SHA.fullmatch(expected) or sha256(root / rel) != expected:
            raise ValueError(f'public input SHA256 mismatch: {rel}')
    return root


async def validate(environment, manifest):
    """Check selected task, original instruction, native image, and exact raw bytes."""
    if manifest['schema_version'] == 1:
        return None
    task = environment.environment_dir.parent.name
    entry = manifest['public_inputs'].get(task)
    if entry is None:
        return None
    if not isinstance(entry, dict) or sha256(environment.environment_dir.parent / 'instruction.md') != entry.get('instruction_sha256'):
        raise ValueError('public inputs were reviewed for a different task instruction')
    root = _inventory(entry)
    index = _relative(entry.get('source_index'))
    if index not in entry['files']:
        raise ValueError('public input source index is not a pinned payload file')
    items = entry.get('items')
    if not isinstance(items, list) or not items:
        raise ValueError('public input descriptors are missing')
    described = []
    for item in items:
        if not isinstance(item, dict) or item.get('kind') not in KINDS:
            raise ValueError('unsupported public input descriptor')
        rel = _relative(item.get('path'))
        if rel not in entry['files'] and rel not in entry['directories']:
            raise ValueError('public input descriptor is outside the pinned inventory')
        label = item.get('label')
        if (not isinstance(label, str) or not label.strip() or len(label) > 300
                or any(ord(c) < 32 or ord(c) == 127 for c in label)):
            raise ValueError('invalid public input label')
        described.append({'kind': item['kind'], 'path': REMOTE + '/' + rel, 'label': label})
    native = getattr(getattr(environment, '_main', None), 'rootfs', None)
    if native is None or Path(native).parent.name != entry.get('native_rootfs_key'):
        raise ValueError('public input native image key mismatch')
    probe = await environment.exec('cat /etc/os-release; printf "\\nAA_DPKG_STATUS_SHA256="; '
                                   'if test -f /var/lib/dpkg/status; then sha256sum /var/lib/dpkg/status | cut -d" " -f1; else echo none; fi', user='root')
    if probe.return_code or not entry.get('image_probe') or probe.stdout != entry['image_probe']:
        raise ValueError('public input native image inventory mismatch')
    return {'task': task, 'root': root, 'entry': entry, 'items': described}


async def stage(environment, validated):
    """Copy only this task's input tree. No code execution, answers or network mapping."""
    root, entry = validated['root'], validated['entry']
    # Revalidate after package setup, before exposing the reviewed public bytes.
    if _inventory(entry) != root:
        raise ValueError('public input root changed before staging')
    absent = await environment.exec('test ! -e ' + REMOTE + ' && test ! -L ' + REMOTE, user='root')
    if absent.return_code:
        raise ValueError('public input destination already exists')
    await environment.upload_dir(root, REMOTE)
    # Native upload_dir merges and copies asynchronously. The host tree can be
    # valid before/after a copy while the copied bytes differ, so authenticate
    # the actual remote bytes and reject every extra path, link or special file.
    checks = ''.join(f'{expected} *{rel}\n' for rel, expected in sorted(entry['files'].items()))
    command = ('cd ' + REMOTE + ' && printf %s ' + shlex.quote(checks)
               + ' | sha256sum --check --status && find . -mindepth 1 -printf ' + shlex.quote(r'%y %P\0'))
    verified = await environment.exec(command, user='root')
    expected_paths = {'f ' + rel for rel in entry['files']} | {'d ' + rel for rel in entry['directories']}
    actual = verified.stdout.split('\0')
    if (verified.return_code or actual[-1:] != ['']
            or len(actual[:-1]) != len(expected_paths) or set(actual[:-1]) != expected_paths):
        raise ValueError('staged public input inventory or SHA256 mismatch')
    result = await environment.exec('chmod -R a+rX ' + REMOTE, user='root')
    if result.return_code:
        raise RuntimeError('could not expose public inputs: ' + result.stderr)
    index = REMOTE + '/' + entry['source_index']
    notice = (f'Public task inputs are cached locally at {REMOTE}. Source URLs, revisions, '
              f'retrieval dates and file hashes are listed in {index}. Use these local files '
              'for the public resources required by the task. ')
    notice += ' '.join(f'{item["label"]}: {item["path"]}.' for item in validated['items'])
    return {'task': validated['task'], 'root': REMOTE, 'source_index': index,
            'source_index_sha256': entry['files'][entry['source_index']],
            'instruction_sha256': entry['instruction_sha256'],
            'native_rootfs_key': entry['native_rootfs_key'], 'items': validated['items'],
            'notice': notice}

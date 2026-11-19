"""Opt-in, post-agent evidence for fixed required outputs; never a grader.

The source is read through a sandbox-root dirfd. No sandbox command, submitted
binary, directory snapshot, or dynamic dependency discovery is executed.
"""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import threading
import time

VERSION = 1
MIB = 1024 * 1024
DIRECT_ALIGNMENT = 4096
DIRECT_TIMEOUT = 20
# Caffe limits are capture budgets, NOT additional grading requirements.
SPECS = {
    'train-fasttext': {'/app/model.bin': 150 * MIB - 1},
    'caffe-cifar-10': {
        '/app/caffe/examples/cifar10/cifar10_quick_train_test.prototxt': MIB,
        '/app/caffe/examples/cifar10/cifar10_quick_solver.prototxt': MIB,
        '/app/caffe/examples/cifar10/cifar10_quick_iter_500.caffemodel': 256 * MIB,
        '/app/caffe/training_output.txt': 16 * MIB,
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def _write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(Path(path).parent)


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_fixed(source_fd, output_fd, size):
    """Copy exactly the authenticated size, using direct large-file writes.

    /proc/self/fd refers only to inherited, already-open HOST descriptors. No
    sandbox pathname is reopened and no agent program or shell is executed.
    """
    prefix = size - size % DIRECT_ALIGNMENT if size > MIB else 0
    if prefix:
        subprocess.run(['/usr/bin/dd', f'if=/proc/self/fd/{source_fd}',
            f'of=/proc/self/fd/{output_fd}', 'bs=1M', f'count={prefix}',
            'iflag=fullblock,count_bytes', 'oflag=direct',
            'conv=notrunc', 'status=none'], pass_fds=(source_fd, output_fd),
            check=True, capture_output=True, timeout=DIRECT_TIMEOUT)
        if os.fstat(output_fd).st_size != prefix:
            raise EOFError('short direct-write prefix')
    position = prefix
    while position < size:
        chunk = os.pread(source_fd, min(MIB, size - position), position)
        if not chunk:
            raise EOFError('short source read')
        written = 0
        while written < len(chunk):
            count = os.pwrite(output_fd, chunk[written:], position + written)
            if count <= 0:
                raise OSError(errno.EIO, 'short output write')
            written += count
        position += len(chunk)
    os.fsync(output_fd)
    if os.fstat(output_fd).st_size != size:
        raise EOFError('captured size mismatch')
    # Hash the actual destination bytes, rather than assuming a successful
    # source read or subprocess exit proves the evidence's contents.
    checksum = hashlib.sha256()
    position = 0
    while position < size:
        chunk = os.pread(output_fd, min(MIB, size - position), position)
        if not chunk:
            raise EOFError('short captured read')
        checksum.update(chunk)
        position += len(chunk)
    return checksum.hexdigest(), 'direct_prefix_bounded_tail' if prefix else 'buffered_small'


def _independent_storage(raw_root, destination):
    return os.stat(raw_root).st_dev != os.stat(destination).st_dev


def _validate_caffe_artifacts(task, task_bytes):
    if task != 'caffe-cifar-10':
        return
    import tomllib
    entries = tomllib.loads(task_bytes.decode()).get('artifacts', [])
    if (not isinstance(entries, list) or len(entries) != len(SPECS[task])
            or any(not isinstance(entry, dict) or set(entry) != {'source'} for entry in entries)
            or {entry['source'] for entry in entries} != set(SPECS[task])):
        raise ValueError('Caffe capture requires the separately reviewed fixed-artifact overlay')


def plan_for(config, key, task_dir, raw_root, config_sha256, config_path=None):
    """Create and bind an append-only plan BEFORE Harbor/agent startup."""
    setting = config.get('required_output_capture')
    task = key.split('__a', 1)[0]
    if not setting or task not in setting['tasks']:
        return None
    if type(setting.get('schema_version')) is not int or setting['schema_version'] != VERSION:
        raise ValueError('invalid required-output capture setting')
    if task not in SPECS or not re.fullmatch(re.escape(task) + r'__a\d+__r\d+', key):
        raise ValueError('invalid required-output task/key')
    commit = config.get('code_commit')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('capture requires a frozen execution SHA')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config['run_id']):
        raise ValueError('invalid capture run ID')
    task_dir = Path(task_dir).resolve()
    task_bytes = (task_dir / 'task.toml').read_bytes()
    _validate_caffe_artifacts(task, task_bytes)
    directory = Path(setting['root']).resolve() / config['run_id'] / key
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    if not _independent_storage(raw_root, directory):
        raise ValueError('capture requires an independent filesystem')
    plan = {
        'schema_version': VERSION, 'key': key, 'task': task,
        'run_id': config['run_id'], 'execution_sha': commit,
        'collector_sha256': digest(Path(__file__).read_bytes()),
        'config_sha256': config_sha256,
        'config_path': str(Path(config_path).resolve()) if config_path is not None else None,
        'raw_root': str(Path(raw_root).resolve()),
        'task_dir': str(task_dir), 'destination': str(directory),
        'instruction_sha256': digest((task_dir / 'instruction.md').read_bytes()),
        'task_toml_sha256': digest(task_bytes),
        'sources': SPECS[task],
        'replay_complete': False,
        'replay_limit': 'Captured final model still needs the original data/runtime/grader; '
            'capture is not a replay or grade verdict.' if task == 'train-fasttext' else
            'Fixed submitted files only. Compiled binary/shared libraries, external '
            'means, custom layers and input-representation evidence are not captured.',
    }
    data = encoded(plan)
    _write_new(directory / 'plan.json', data)
    return plan, data


def harbor_args(plan_path, data):
    plan = json.loads(data)
    args = []
    for source in plan['sources']:
        args += ['--artifact', source]
    return args + ['--ek', f'required_output_capture_plan={plan_path}',
                   '--ek', f'required_output_capture_sha256={digest(data)}']


def load_plan(path, expected_sha, environment_dir, trial_dir, *, code_root=None):
    """Authenticate the exact plan against the actual task/trial/imported code."""
    raw = Path(path).read_bytes()
    if digest(raw) != expected_sha:
        raise ValueError('required-output plan hash mismatch')
    plan = json.loads(raw)
    task = plan.get('task')
    if (type(plan.get('schema_version')) is not int or plan['schema_version'] != VERSION
            or task not in SPECS or plan.get('sources') != SPECS[task]
            or any(type(value) is not int for value in plan['sources'].values())
            or plan.get('collector_sha256') != digest(Path(__file__).read_bytes())):
        raise ValueError('required-output plan contract mismatch')
    if (not plan.get('config_path')
            or digest(Path(plan['config_path']).read_bytes()) != plan['config_sha256']):
        raise ValueError('required-output actual run config mismatch')
    actual_task = Path(environment_dir).parent.resolve()
    _validate_caffe_artifacts(task, (actual_task / 'task.toml').read_bytes())
    if (actual_task != Path(plan['task_dir']) or actual_task.name != task
            or digest((actual_task / 'instruction.md').read_bytes()) != plan['instruction_sha256']
            or digest((actual_task / 'task.toml').read_bytes()) != plan['task_toml_sha256']
            or not Path(trial_dir).resolve().is_relative_to(Path(plan['raw_root']))):
        raise ValueError('required-output actual task/trial mismatch')
    root = code_root or Path(__file__).resolve().parents[3]
    complete = json.loads((Path(root) / '.complete').read_text())
    if complete.get('git_sha') != plan['execution_sha']:
        raise ValueError('required-output actual execution SHA mismatch')
    if Path(plan['destination']).resolve() == Path(plan['raw_root']).resolve() or (
            Path(plan['destination']).resolve().is_relative_to(Path(plan['raw_root']).resolve())):
        raise ValueError('capture destination must be independent of raw trial storage')
    if not _independent_storage(plan['raw_root'], plan['destination']):
        raise ValueError('capture destination shares the raw trial filesystem')
    if (Path(plan['destination']) / 'plan.json').read_bytes() != raw:
        raise ValueError('independent capture plan mismatch')
    return plan


def _identity(info):
    return {name: getattr(info, name) for name in (
        'st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _open_beneath(rootfd, source, flags):
    parts = PurePosixPath(source).parts
    if not parts or parts[0] != '/' or any(p in ('.', '..') for p in parts):
        raise ValueError('invalid fixed source path')
    current = os.dup(rootfd)
    try:
        for component in parts[1:-1]:
            child = os.open(component, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=current)
            os.close(current)
            current = child
        return os.open(parts[-1], flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
    finally:
        os.close(current)


def _proc_text(procfd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=procfd)
    try:
        data = os.read(fd, 65537)
        if len(data) > 65536:
            raise ValueError('oversized process identity record')
        return data.decode()
    finally:
        os.close(fd)


def _proc_identity(procfd, pid):
    """Host /proc identity, read relative to an already-pinned PID directory."""
    raw = _proc_text(procfd, 'stat')
    fields = raw.rsplit(')', 1)[1].split()
    if int(raw.split(' ', 1)[0]) != pid or fields[0] in ('Z', 'X', 'x'):
        raise ValueError('sandbox process is gone or has wrong identity')
    status = dict(line.split(':', 1) for line in _proc_text(procfd, 'status').splitlines() if ':' in line)
    return {'pid': pid, 'parent_pid': int(fields[1]), 'start_ticks': int(fields[19]),
            'host_uids': [int(value) for value in status['Uid'].split()],
            'namespace_pids': [int(value) for value in status['NSpid'].split()],
            'pid_namespace': os.readlink('ns/pid', dir_fd=procfd),
            'mount_namespace': os.readlink('ns/mnt', dir_fd=procfd)}


def _owned_sandbox_root(root, sandbox):
    """Follow ONLY the root magic link of the provider's owned live init.

    Sandbox.start creates _unshare as an evaluator child, waits for ready and
    sets init_pid to its direct child. That init pivots root and remains PID 1.
    No identity comes from task data, a plan, or an arbitrary context PID.
    """
    from harbor_ns.sandbox import Sandbox
    if not isinstance(sandbox, Sandbox):
        raise ValueError('proc-root binding requires the native Sandbox handle')
    launch = sandbox._unshare
    pid = sandbox.init_pid
    if (type(pid) is not int or pid <= 0 or not isinstance(launch, asyncio.subprocess.Process)
            or type(launch.pid) is not int or launch.pid <= 0 or launch.returncode is not None
            or str(root) != f'/proc/{pid}/root'):
        raise ValueError('proc-root binding does not match the live owned Sandbox')
    procfd = os.open('/proc', os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
    parentfd = initfd = rootfd = None
    try:
        parentfd = os.open(str(launch.pid), os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=procfd)
        initfd = os.open(str(pid), os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=procfd)
        parent = _proc_identity(parentfd, launch.pid)
        child = _proc_identity(initfd, pid)
        # UIDs are observed in the evaluator's host /proc, not guessed from
        # namespace UID 0. Native multiuid also maps init's root to this owner.
        uid = os.geteuid()
        if (parent['parent_pid'] != os.getpid() or child['parent_pid'] != launch.pid
                or parent['host_uids'] != [uid] * 4 or child['host_uids'] != [uid] * 4
                or len(child['namespace_pids']) < 2 or child['namespace_pids'][-1] != 1
                or child['pid_namespace'] == os.readlink('/proc/self/ns/pid')
                or child['mount_namespace'] == os.readlink('/proc/self/ns/mnt')):
            raise ValueError('sandbox init ownership or namespace identity mismatch')
        # The PID directory and its kernel-produced root entry are trusted;
        # this is the ONLY intentional magic-link follow. Submitted child
        # paths continue through _open_beneath with no-follow at every step.
        rootfd = os.open('root', os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=initfd)
        if (launch.returncode is not None or sandbox._unshare is not launch or sandbox.init_pid != pid
                or parent != _proc_identity(parentfd, launch.pid)
                or child != _proc_identity(initfd, pid)):
            raise ValueError('sandbox process identity changed while pinning root')
        result, rootfd = rootfd, None
        return result, {'launcher': parent, 'init': child,
                        'identity_observer': 'evaluator host /proc; before agent',
                        'root_magic_link_followed_once': True}
    finally:
        for fd in (rootfd, initfd, parentfd, procfd):
            if fd is not None:
                os.close(fd)


class Capture:
    def __init__(self, plan):
        self.plan = plan
        self.records = {}
        self.materialized = {}
        self.rootfd = None
        self._root_lock = threading.Lock()
        self._root_bound = False
        self._root_closed = False

    def bind_root(self, root, context=None, *, sandbox=None):
        """Pin the actual initialized sandbox root BEFORE agent execution."""
        with self._root_lock:
            if self._root_bound or self._root_closed:
                raise RuntimeError('required-output sandbox root already bound or closed')
            proof = None
            if sandbox is None:
                fd = os.open(root, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
            else:
                fd, proof = _owned_sandbox_root(root, sandbox)
            try:
                _write_new(Path(self.plan['destination']) / 'root-binding.json', encoded({
                    'root_handle': str(root), 'root_identity': _identity(os.fstat(fd)),
                    'key': self.plan['key'], 'execution_sha': self.plan['execution_sha'],
                    'plan_sha256': digest(encoded(self.plan)), 'at_epoch': time.time(),
                    'actual_environment': context,
                    'sandbox_process_identity': proof,
                    'image_identity_limit': 'Native cached rootfs key, not a hash of the live mounted filesystem.'}))
            except BaseException:
                os.close(fd)
                raise
            self.rootfd = fd
            self._root_bound = True

    def close(self):
        with self._root_lock:
            self._root_closed = True
            if self.rootfd is not None:
                os.close(self.rootfd)
                self.rootfd = None

    def _rootfd(self, root):
        # root is deliberately unused: teardown must never reopen a mutable
        # pathname. A thread holding an earlier dup may finish on that pin.
        with self._root_lock:
            if self.rootfd is None or self._root_closed:
                raise OSError(errno.EBADF, 'required-output sandbox root is unbound or closed')
            return os.dup(self.rootfd)

    def selected(self, source, service=None):
        return service in (None, 'main') and source in self.plan['sources']

    def is_dir(self, root, source):
        """Host metadata only: avoid executing sandbox test -d for selected paths."""
        rootfd = self._rootfd(root)
        try:
            try:
                fd = _open_beneath(rootfd, source, os.O_PATH)
            except OSError:
                return False
            try:
                return stat.S_ISDIR(os.fstat(fd).st_mode)
            finally:
                os.close(fd)
        finally:
            os.close(rootfd)

    def capture(self, root, source):
        if source in self.records:
            return self.records[source]
        limit = self.plan['sources'][source]
        index = list(self.plan['sources']).index(source)
        directory = Path(self.plan['destination'])
        target = directory / f'{index:02d}-{PurePosixPath(source).name}'
        partial = target.with_name(target.name + '.partial')
        record = {'source': source, 'max_bytes': limit, 'started_at_epoch': time.time(),
                  'plan_sha256': digest(encoded(self.plan)), 'key': self.plan['key'],
                  'task': self.plan['task'], 'execution_sha': self.plan['execution_sha']}
        _write_new(directory / f'{index:02d}.started.json', encoded(record))
        rootfd = fd = None
        try:
            rootfd = self._rootfd(root)
            record['sandbox_root_identity'] = _identity(os.fstat(rootfd))
            probe = _open_beneath(rootfd, source, os.O_PATH)
            try:
                before = os.fstat(probe)
                record['before'] = _identity(before)
                if stat.S_ISLNK(before.st_mode):
                    record['status'] = 'symlink_rejected'
                    return self._finish(source, record, index)
                if not stat.S_ISREG(before.st_mode):
                    record['status'] = 'nonregular'
                    return self._finish(source, record, index)
                if before.st_size > limit:
                    record['status'] = 'oversized'
                    return self._finish(source, record, index)
            finally:
                os.close(probe)
            fd = _open_beneath(rootfd, source, os.O_RDONLY | os.O_NONBLOCK)
            if _identity(os.fstat(fd)) != record['before']:
                record['status'] = 'unstable'
                return self._finish(source, record, index)
            output = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                checksum, method = _copy_fixed(fd, output, before.st_size)
            finally:
                os.close(output)
            record['after'] = _identity(os.fstat(fd))
            check = _open_beneath(rootfd, source, os.O_PATH)
            try:
                record['path_after'] = _identity(os.fstat(check))
            finally:
                os.close(check)
            if (record['before'] != record['after']
                    or record['before'] != record['path_after']):
                record['status'] = 'unstable'
            else:
                os.link(partial, target, follow_symlinks=False)
                partial.unlink()
                _sync_directory(directory)
                record.update(status='captured', bytes=before.st_size, sha256=checksum,
                              copy_method=method, data_fsynced=True,
                              captured_path=str(target))
        except EOFError as error:
            record.update(status='unstable', error=str(error))
        except subprocess.TimeoutExpired as error:
            record.update(status='capture_timeout', error=str(error))
        except subprocess.CalledProcessError as error:
            record.update(status='capture_error', error=str(error),
                          direct_copy_stderr=(error.stderr or b'').decode(errors='replace')[:4096])
        except FileNotFoundError as error:
            record.update(status='missing' if 'before' not in record else 'unstable', error=str(error))
        except PermissionError as error:
            record.update(status='unreadable', error=str(error))
        except OSError as error:
            record.update(status='symlink_or_parent_rejected' if error.errno in
                          (errno.ELOOP, errno.ENOTDIR) else 'capture_error', error=str(error))
        finally:
            if fd is not None:
                os.close(fd)
            if rootfd is not None:
                os.close(rootfd)
            partial.unlink(missing_ok=True)
        return self._finish(source, record, index)

    def _finish(self, source, record, index):
        record['finished_at_epoch'] = time.time()
        record['record_durability'] = 'file_and_parent_directory_fsync'
        _write_new(Path(self.plan['destination']) / f'{index:02d}.record.json', encoded(record))
        self.records[source] = record
        return record

    def materialize(self, root, source, target):
        previous = self.materialized.get((source, str(target)))
        if previous is not None:
            if previous['status'] != 'ok':
                raise OSError(previous.get('errno'), previous.get('error'))
            return
        record = self.capture(root, source)
        if record['status'] != 'captured':
            raise OSError(f"required output evidence: {record['status']}")
        # This ordinary Harbor copy may fail (including ENOSPC); independent
        # evidence and its typed record have already been committed elsewhere.
        ordinary = {'source': source, 'target': str(target), 'at_epoch': time.time()}
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record['captured_path'], target)
            ordinary['status'] = 'ok'
        except OSError as error:
            ordinary.update(status='failed', errno=error.errno, error=str(error))
            raise
        finally:
            self.materialized[(source, str(target))] = ordinary
            index = list(self.plan['sources']).index(source)
            _write_new(Path(self.plan['destination']) / f'{index:02d}.ordinary-copy.json', encoded(ordinary))

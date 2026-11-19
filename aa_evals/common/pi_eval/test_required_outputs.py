import asyncio
import errno
import json
import logging
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import tempfile
import threading
import sys
import unittest
from unittest import mock

from . import required_outputs as capture


class RequiredOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.task = self.base / 'train-fasttext'
        (self.task / 'environment').mkdir(parents=True)
        (self.task / 'instruction.md').write_text('Original instruction\n')
        (self.task / 'task.toml').write_text('artifacts = []\n')
        self.raw = self.base / 'raw'
        (self.raw / 'trial').mkdir(parents=True)
        self.config = {'run_id': 'capture-unit', 'code_commit': 'a' * 40,
                       'required_output_capture': {'schema_version': 1,
                           'root': str(self.base / 'independent'), 'tasks': ['train-fasttext']}}
        self.config_path = self.base / 'run-config.json'
        self.config_path.write_text(json.dumps(self.config))
        # Unit fixtures emulate distinct devices. Production requires actual
        # different st_dev values; no real filesystem is filled in these tests.
        with mock.patch.object(capture, '_independent_storage', return_value=True):
            self.plan, self.data = capture.plan_for(self.config, 'train-fasttext__a0__r0',
                self.task, self.raw, capture.digest(self.config_path.read_bytes()), self.config_path)
        self.plan_path = self.base / 'plan.json'
        self.plan_path.write_bytes(self.data)
        self.code = self.base / 'code'
        self.code.mkdir()
        (self.code / '.complete').write_text(json.dumps({'git_sha': 'a' * 40}))
        self.root = self.base / 'sandbox'
        (self.root / 'app').mkdir(parents=True)
        self.model = self.root / 'app/model.bin'
        self.collector = capture.Capture(self.plan)
        self.collector.bind_root(self.root)
        self.addCleanup(self.collector.close)

    def load(self):
        with mock.patch.object(capture, '_independent_storage', return_value=True):
            return capture.load_plan(self.plan_path, capture.digest(self.data),
                                     self.task / 'environment', self.raw / 'trial', code_root=self.code)

    def record(self):
        return json.loads((Path(self.plan['destination']) / '00.record.json').read_text())

    def test_captured_bytes_hash_and_stability(self):
        self.model.write_bytes(b'neutral model bytes')
        self.collector.materialize(self.root, '/app/model.bin', self.raw / 'artifact.bin')
        record = self.record()
        self.assertEqual(record['status'], 'captured')
        self.assertEqual(record['sha256'], capture.digest(b'neutral model bytes'))
        self.assertEqual(record['before'], record['after'])
        self.assertEqual(record['before'], record['path_after'])
        self.assertEqual(Path(record['captured_path']).read_bytes(), b'neutral model bytes')
        self.assertEqual((self.raw / 'artifact.bin').read_bytes(), b'neutral model bytes')
        # Duplicate task/trial selectors must not overwrite the evidence.
        self.collector.materialize(self.root, '/app/model.bin', self.raw / 'artifact.bin')

    def test_ordinary_copy_enospc_preserves_independent_evidence(self):
        self.model.write_bytes(b'submission')
        with mock.patch.object(capture.shutil, 'copyfile', side_effect=OSError(errno.ENOSPC, 'full raw')):
            with self.assertRaises(OSError):
                self.collector.materialize(self.root, '/app/model.bin', self.raw / 'artifact.bin')
        record = self.record()
        self.assertEqual(record['status'], 'captured')
        self.assertEqual(Path(record['captured_path']).read_bytes(), b'submission')
        ordinary = json.loads((Path(self.plan['destination']) / '00.ordinary-copy.json').read_text())
        self.assertEqual(ordinary['errno'], errno.ENOSPC)
        self.assertEqual(ordinary['status'], 'failed')

    def test_missing_is_positive_metadata_not_a_grade(self):
        record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'missing')
        self.assertNotIn('score', record)
        self.assertNotIn('captured_path', record)

    def test_original_fasttext_strict_size_contract(self):
        maximum = capture.SPECS['train-fasttext']['/app/model.bin']
        self.assertEqual(maximum, 150 * 1024 * 1024 - 1)
        self.assertLessEqual(110513478, maximum)  # actual valid a3 output size
        with self.model.open('wb') as stream:
            stream.truncate(maximum + 1)  # sparse fixture; no bulk payload written
        with mock.patch.object(capture.os, 'read', side_effect=AssertionError('oversize bytes read')):
            self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'oversized')

    def test_final_symlink_is_not_followed(self):
        private = self.base / 'outside'
        private.write_bytes(b'neutral outside fixture')
        self.model.symlink_to(private)
        self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'symlink_rejected')

    def test_parent_symlink_is_not_followed(self):
        self.model.parent.rmdir()
        outside = self.base / 'outside-dir'
        outside.mkdir()
        (outside / 'model.bin').write_bytes(b'neutral outside fixture')
        (self.root / 'app').symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'],
                         'symlink_or_parent_rejected')

    def test_directory_is_metadata_only(self):
        self.model.mkdir()
        (self.model / 'junk').write_bytes(b'must not recurse')
        with mock.patch.object(capture.os, 'read', side_effect=AssertionError('directory bytes read')):
            self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'nonregular')

    @unittest.skipIf(os.geteuid() == 0, 'DAC unreadability requires a non-root host fixture')
    def test_unreadable_never_runs_a_sandbox_copy(self):
        self.model.write_bytes(b'unreadable neutral fixture')
        self.model.chmod(0)
        self.addCleanup(self.model.chmod, 0o600)
        self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'unreadable')

    def test_mutated_during_read_is_not_committed(self):
        self.model.write_bytes(b'original')
        original_read = os.pread
        changed = False
        def read(fd, count, offset):
            nonlocal changed
            data = original_read(fd, count, offset)
            if data and not changed:
                changed = True
                with self.model.open('ab') as stream:
                    stream.write(b'change')
            return data
        with mock.patch.object(capture.os, 'pread', side_effect=read):
            self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'unstable')
        self.assertFalse(list(Path(self.plan['destination']).glob('*.partial')))

    def test_replaced_path_is_not_committed(self):
        self.model.write_bytes(b'original')
        original_read = os.pread
        changed = False
        def read(fd, count, offset):
            nonlocal changed
            data = original_read(fd, count, offset)
            if data and not changed:
                changed = True
                replacement = self.model.with_name('replacement')
                replacement.write_bytes(b'original')
                replacement.replace(self.model)
            return data
        with mock.patch.object(capture.os, 'pread', side_effect=read):
            self.assertEqual(self.collector.capture(self.root, '/app/model.bin')['status'], 'unstable')

    def test_root_is_pinned_before_agent(self):
        self.model.write_bytes(b'actual initialized sandbox')
        self.root.rename(self.base / 'old-sandbox')
        (self.root / 'app').mkdir(parents=True)
        self.model.write_bytes(b'replacement root')
        record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(Path(record['captured_path']).read_bytes(), b'actual initialized sandbox')

    def test_unbound_and_closed_root_never_reopen_replacement(self):
        other = capture.Capture(self.plan)
        with mock.patch.object(capture.os, 'open', side_effect=AssertionError('root reopened')):
            with self.assertRaises(OSError):
                other._rootfd(self.root)
        self.collector.close()
        self.root.rename(self.base / 'old-sandbox')
        (self.root / 'app').mkdir(parents=True)
        self.model.write_bytes(b'wrong replacement')
        with mock.patch.object(capture.os, 'open', side_effect=AssertionError('root reopened')):
            with self.assertRaises(OSError):
                self.collector._rootfd(self.root)
        with self.assertRaises(RuntimeError):
            self.collector.bind_root(self.root)
        record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'capture_error')
        self.assertNotIn('captured_path', record)

    def test_close_waits_for_atomic_dup_and_existing_pin_survives(self):
        self.model.write_bytes(b'original pin')
        entered, proceed = threading.Event(), threading.Event()
        duplicate = []
        original_dup = os.dup
        def dup(fd):
            entered.set()
            self.assertTrue(proceed.wait(2))
            return original_dup(fd)
        with mock.patch.object(capture.os, 'dup', side_effect=dup):
            worker = threading.Thread(target=lambda: duplicate.append(self.collector._rootfd(self.root)))
            worker.start()
            self.assertTrue(entered.wait(2))
            closer = threading.Thread(target=self.collector.close)
            closer.start()
            proceed.set()
            worker.join(2)
            closer.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        try:
            self.root.rename(self.base / 'old-sandbox')
            (self.root / 'app').mkdir(parents=True)
            self.model.write_bytes(b'replacement')
            fd = capture._open_beneath(duplicate[0], '/app/model.bin', os.O_RDONLY)
            try:
                self.assertEqual(os.read(fd, 100), b'original pin')
            finally:
                os.close(fd)
        finally:
            os.close(duplicate[0])
        with self.assertRaises(OSError):
            self.collector._rootfd(self.root)

    def test_large_direct_copy_exact_prefix_tail_and_destination_hash(self):
        payload = b'neutral' * (capture.MIB // 7 + 2)
        self.model.write_bytes(payload)
        original_run = capture.subprocess.run
        with mock.patch.object(capture.subprocess, 'run', wraps=original_run) as run:
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'captured')
        self.assertEqual(record['bytes'], len(payload))
        self.assertEqual(record['sha256'], capture.digest(payload))
        self.assertEqual(Path(record['captured_path']).read_bytes(), payload)
        self.assertEqual(record['copy_method'], 'direct_prefix_bounded_tail')
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], '/usr/bin/dd')
        self.assertIn('oflag=direct', argv)
        self.assertIn('conv=notrunc', argv)
        self.assertTrue(all(arg.startswith(('if=/proc/self/fd/', 'of=/proc/self/fd/'))
                            for arg in argv if arg.startswith(('if=', 'of='))))
        self.assertEqual(len(run.call_args.kwargs['pass_fds']), 2)
        self.assertEqual(run.call_args.kwargs['timeout'], capture.DIRECT_TIMEOUT)

    def test_large_short_read_is_unstable_not_full_capture(self):
        self.model.write_bytes(b'x' * (capture.MIB + 19))
        with mock.patch.object(capture.subprocess, 'run', return_value=None):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'unstable')
        self.assertNotIn('captured_path', record)
        self.assertFalse(list(Path(self.plan['destination']).glob('*.partial')))

    def test_direct_errors_and_timeout_do_not_fall_back_to_buffered_copy(self):
        self.model.write_bytes(b'x' * (capture.MIB + 19))
        error = capture.subprocess.CalledProcessError(1, ['/usr/bin/dd'], stderr=b'no direct I/O')
        with mock.patch.object(capture.subprocess, 'run', side_effect=error), \
                mock.patch.object(capture.os, 'pread', side_effect=AssertionError('buffered fallback')):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'capture_error')
        self.assertNotIn('captured_path', record)

    def test_direct_timeout_is_typed_and_partial_removed(self):
        self.model.write_bytes(b'x' * (capture.MIB + 19))
        error = capture.subprocess.TimeoutExpired(['/usr/bin/dd'], capture.DIRECT_TIMEOUT)
        with mock.patch.object(capture.subprocess, 'run', side_effect=error):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'capture_timeout')
        self.assertNotIn('captured_path', record)
        self.assertFalse(list(Path(self.plan['destination']).glob('*.partial')))

    def test_large_source_growth_is_not_read_or_claimed_as_full_capture(self):
        payload = b'x' * (capture.MIB + 19)
        self.model.write_bytes(payload)
        original_run, original_read = capture.subprocess.run, os.pread
        def run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            with self.model.open('ab') as stream:
                stream.write(b'new bytes beyond before.size')
            return result
        reads = []
        def read(fd, count, offset):
            reads.append((count, offset))
            return original_read(fd, count, offset)
        with mock.patch.object(capture.subprocess, 'run', side_effect=run), \
                mock.patch.object(capture.os, 'pread', side_effect=read):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'unstable')
        self.assertTrue(all(count + offset <= len(payload) for count, offset in reads))
        self.assertNotIn('captured_path', record)

    def test_read_bound_uses_before_size_and_hashes_destination(self):
        self.model.write_bytes(b'fixed bytes')
        original_pread = os.pread
        reads = []
        def pread(fd, count, offset):
            reads.append((count, offset))
            return original_pread(fd, count, offset)
        with mock.patch.object(capture.os, 'pread', side_effect=pread):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'captured')
        self.assertTrue(all(count + offset <= len(b'fixed bytes') for count, offset in reads))
        self.assertEqual(len(reads), 2)  # source read and independent destination hash

    def test_data_record_and_directory_fsync_precede_success(self):
        self.model.write_bytes(b'neutral')
        synced = []
        original = os.fsync
        def fsync(fd):
            synced.append(os.fstat(fd).st_mode)
            return original(fd)
        with mock.patch.object(capture.os, 'fsync', side_effect=fsync):
            record = self.collector.capture(self.root, '/app/model.bin')
        import stat
        self.assertEqual(record['status'], 'captured')
        self.assertGreaterEqual(sum(stat.S_ISREG(mode) for mode in synced), 3)
        self.assertGreaterEqual(sum(stat.S_ISDIR(mode) for mode in synced), 3)

    def test_failed_data_fsync_never_claims_capture(self):
        self.model.write_bytes(b'neutral')
        original = os.fsync
        def fsync(fd):
            if os.fstat(fd).st_size == len(b'neutral'):
                raise OSError(errno.EIO, 'neutral data sync failure')
            return original(fd)
        with mock.patch.object(capture.os, 'fsync', side_effect=fsync):
            record = self.collector.capture(self.root, '/app/model.bin')
        self.assertEqual(record['status'], 'capture_error')
        self.assertNotIn('captured_path', record)

    def test_plan_binds_actual_task_trial_code_and_source_bytes(self):
        self.assertEqual(self.load()['key'], 'train-fasttext__a0__r0')
        (self.task / 'instruction.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'task/trial'):
            self.load()

    def test_plan_hash_and_execution_mismatch_rejected(self):
        self.plan_path.write_bytes(self.data + b' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load()
        self.plan_path.write_bytes(self.data)
        (self.code / '.complete').write_text(json.dumps({'git_sha': 'c' * 40}))
        with self.assertRaisesRegex(ValueError, 'execution SHA'):
            self.load()

    def test_run_config_change_rejected_before_agent(self):
        self.config_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'actual run config'):
            self.load()

    def test_forged_source_or_float_limit_rejected_even_with_new_hash(self):
        for sources in ({'/etc/passwd': 100}, {'/app/model.bin': float(150 * capture.MIB - 1)}):
            with self.subTest(sources=sources):
                plan = dict(self.plan, sources=sources)
                data = capture.encoded(plan)
                self.plan_path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                    capture.load_plan(self.plan_path, capture.digest(data),
                        self.task / 'environment', self.raw / 'trial', code_root=self.code)

    def test_same_filesystem_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, 'shares the raw'):
            capture.load_plan(self.plan_path, capture.digest(self.data),
                              self.task / 'environment', self.raw / 'trial', code_root=self.code)

    def test_disabled_and_unselected_leave_default_argv_unchanged(self):
        config = dict(self.config)
        config.pop('required_output_capture')
        self.assertIsNone(capture.plan_for(config, 'train-fasttext__a1__r0', self.task, self.raw, 'b' * 64))
        self.assertIsNone(capture.plan_for(self.config, 'other-task__a0__r0', self.task, self.raw, 'b' * 64))
        original = ['bash', 'run_harbor.sh', 'task', 'raw', '-k', '1']
        combined = original + capture.harbor_args(self.base / 'plan with spaces.json', self.data)
        self.assertEqual(combined[:len(original)], original)
        self.assertIn('--artifact', combined)
        self.assertTrue(any('plan with spaces.json' in arg for arg in combined))
        self.assertFalse(any('HARBOR_EXTRA_ARGS' in arg for arg in combined))
        from harbor.cli.utils import parse_kwargs
        parsed = parse_kwargs([combined[-3], combined[-1]])
        self.assertEqual(parsed['required_output_capture_plan'], str(self.base / 'plan with spaces.json'))
        self.assertEqual(parsed['required_output_capture_sha256'], capture.digest(self.data))

    def test_default_provider_download_delegates_unchanged(self):
        from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment
        env = object.__new__(NsSidecarEnvironment)
        env._required_outputs = None
        env.download_file = mock.AsyncMock()
        asyncio.run(env.service_download_file('/app/model.bin', self.raw / 'same-target'))
        env.download_file.assert_awaited_once_with('/app/model.bin', self.raw / 'same-target')

    def test_caffe_scope_is_fixed_and_dependency_limit_honest(self):
        sources = capture.SPECS['caffe-cifar-10']
        self.assertEqual(len(sources), 4)
        self.assertTrue(all(Path(path).suffix in ('.prototxt', '.caffemodel', '.txt') for path in sources))
        self.assertFalse(any('lmdb' in path or 'leveldb' in path for path in sources))
        with self.assertRaisesRegex(ValueError, 'fixed-artifact overlay'):
            capture._validate_caffe_artifacts('caffe-cifar-10', b'artifacts = []')
        values = [{'source': path} for path in sources]
        capture._validate_caffe_artifacts('caffe-cifar-10',
            ('artifacts = ' + json.dumps(values).replace(': ', ' = ')).encode())

    def test_native_no_submit_records_opt_in_and_never_upgrades_stored_config(self):
        from . import run
        bundle = self.base / 'bundle.tar'
        bundle.write_bytes(b'neutral metadata-only bundle placeholder')
        endpoint = {'served': 'fake-model', 'urls': ['http://127.0.0.1:9/v1'],
                    'request': {'pi_thinking': 'max', 'overrides': {'temperature': 1.0}}}
        common = SimpleNamespace(PROTOCOLS={run.HARNESS: {
            'context_window': 131072, 'agent_timeout_sec': 21600}})
        args = ['--bench', 'tb21', '--run-id', 'capture-init', '--model', 'fake-model',
                '--seed-namespace', 'unit', '--bundle', str(bundle), '--attempts', '1',
                '--runtime-cache-manifest', '', '--runtime-cache-manifest-sha256', '', '--no-submit']
        with mock.patch.object(run, 'RUNS_ROOT', self.base / 'runs'), \
                mock.patch.object(run.endpoint, 'alive', return_value=(False, endpoint, 'fake only')), \
                mock.patch.object(run, 'snapshot_code', return_value=str(self.code)), \
                mock.patch.object(run, 'q35_common', return_value=common), \
                mock.patch.object(run, 'bundle_tasks', return_value=['train-fasttext', 'unrelated']), \
                mock.patch.object(run, 'submit_workers') as submit:
            run.main(args)
            path = self.base / 'runs/tb21/capture-init/config.json'
            original = path.read_bytes()
            self.assertNotIn('required_output_capture', json.loads(original))
            with self.assertRaisesRegex(SystemExit, 'capture policy differs'):
                run.main(args + ['--required-output-capture-root', '/shared/future-capture-unit'])
            self.assertEqual(path.read_bytes(), original)
            args[args.index('capture-init')] = 'capture-enabled'
            run.main(args + ['--required-output-capture-root', '/shared/future-capture-unit'])
            setting = json.loads((path.parent.parent / 'capture-enabled/config.json').read_text())['required_output_capture']
            self.assertEqual(setting['tasks'], ['train-fasttext'])
            self.assertEqual(setting['root'], '/shared/future-capture-unit')
            submit.assert_not_called()

    def test_real_harbor_collection_precedes_failed_tests_upload(self):
        from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment
        from harbor.models.task.config import TaskConfig
        from harbor.trial.artifact_handler import ArtifactHandler
        from harbor.trial.single_step import SingleStepTrial
        from harbor.verifier.verifier import Verifier
        self.model.write_bytes(b'neutral submitted bytes')
        events = []
        root = self.root
        class Environment:
            capabilities = SimpleNamespace(mounted=True)
            _required_outputs = self.collector
            _root = lambda instance: root
            service_is_dir = NsSidecarEnvironment.service_is_dir
            service_download_file = NsSidecarEnvironment.service_download_file
            service_download_dir = NsSidecarEnvironment.service_download_dir
        env = Environment()
        handler = ArtifactHandler(artifacts=['/app/model.bin'], logger=logging.getLogger('capture-test'))
        async def agent(): events.append('agent-finished')
        async def logs(): events.append('agent-logs')
        async def collect(**kwargs):
            await handler.download_artifacts(env, self.raw / 'artifacts',
                                            source_artifacts_dir=PurePosixPath('/logs/artifacts'))
            self.assertEqual(self.record()['status'], 'captured')
            events.append('captured')
        async def upload(**kwargs):
            events.append('tests-upload')
            raise OSError(errno.ENOSPC, 'neutral verifier upload failure')
        tests = self.base / 'tests'
        tests.mkdir()
        (tests / 'test.sh').write_text('true\n')
        verifier = SimpleNamespace(environment=SimpleNamespace(os='linux', upload_dir=upload),
            _skip_tests_upload=False, _resolve_tests=lambda: ([tests], tests, tests / 'test.sh'),
            task=SimpleNamespace(paths=SimpleNamespace(validate_input_tree=lambda path: None)))
        async def verify():
            try:
                await Verifier.verify(verifier)
            except Exception as error:
                self.assertEqual(error.__cause__.errno, errno.ENOSPC)
        async def stop(): events.append('stopped')
        trial = SimpleNamespace(task=SimpleNamespace(config=TaskConfig()), _run_agent=agent,
            _upload_agent_logs=logs, _collect_artifacts=collect, _run_verifier=verify,
            _stop_agent_environment=stop)
        with mock.patch.object(capture.shutil, 'copyfile', side_effect=OSError(errno.ENOSPC, 'full raw')):
            asyncio.run(SingleStepTrial._run(trial))
        self.assertEqual(events, ['agent-finished', 'agent-logs', 'captured', 'tests-upload', 'stopped'])
        manifest = json.loads((self.raw / 'artifacts/manifest.json').read_text())
        self.assertEqual(next(row['status'] for row in manifest if row['source'] == '/app/model.bin'), 'failed')
        self.assertEqual(self.record()['status'], 'captured')


class ProcRootBindingTests(unittest.IsolatedAsyncioTestCase):
    """Real host /proc links/owned process tree, without launching namespaces.

    The positive link-opening test substitutes ONLY namespace metadata because
    these CPU tests intentionally do not unshare. The separate neutral job
    must validate the full native Sandbox.start identity without substitutions.
    """
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name)
        self.collector = capture.Capture({'destination': str(self.destination),
            'key': 'neutral-proc-root__a0__r0', 'execution_sha': 'a' * 40})
        self.addCleanup(self.collector.close)
        self.launch = await asyncio.create_subprocess_exec(sys.executable, '-c',
            'import os,time\npid=os.fork()\n'
            'if pid == 0:\n time.sleep(60)\nelse:\n print(pid,flush=True)\n os.waitpid(pid,0)\n',
            stdout=asyncio.subprocess.PIPE)
        self.pid = int(await self.launch.stdout.readline())
        from harbor_ns.sandbox import Sandbox
        self.box = object.__new__(Sandbox)
        self.box._unshare, self.box.init_pid = self.launch, self.pid
        self.root = Path(f'/proc/{self.pid}/root')

    async def asyncTearDown(self):
        try:
            os.kill(self.pid, 9)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.launch.wait(), 2)
        except asyncio.TimeoutError:
            self.launch.kill()
            await self.launch.wait()

    def namespace_metadata(self):
        original = capture._proc_identity
        def identity(fd, pid):
            value = original(fd, pid)
            if pid == self.pid:
                value.update(namespace_pids=[pid, 1], pid_namespace='pid:[neutral-test-substitution]',
                             mount_namespace='mnt:[neutral-test-substitution]')
            return value
        return identity

    def test_real_proc_magic_link_opens_once_with_owned_process_identity(self):
        marker = self.destination / 'marker'
        marker.write_bytes(b'neutral real proc-root file')
        with mock.patch.object(capture, '_proc_identity', side_effect=self.namespace_metadata()):
            self.collector.bind_root(self.root, sandbox=self.box)
        binding = json.loads((self.destination / 'root-binding.json').read_text())
        proof = binding['sandbox_process_identity']
        self.assertEqual(proof['launcher']['parent_pid'], os.getpid())
        self.assertEqual(proof['init']['parent_pid'], self.launch.pid)
        self.assertEqual(proof['init']['host_uids'], [os.geteuid()] * 4)
        self.assertGreater(proof['init']['start_ticks'], 0)
        rootfd = self.collector._rootfd(self.root)
        try:
            fd = capture._open_beneath(rootfd, str(marker), os.O_RDONLY)
            try:
                self.assertEqual(os.read(fd, 100), b'neutral real proc-root file')
            finally:
                os.close(fd)
        finally:
            os.close(rootfd)
        # It remains a pin, even after close and with an alternate root argument.
        self.collector.close()
        with self.assertRaises(OSError):
            self.collector._rootfd(Path('/proc/self/root'))

    def test_real_proc_link_without_native_sandbox_is_refused(self):
        with self.assertRaises(OSError):
            self.collector.bind_root(self.root)
        self.assertIsNone(self.collector.rootfd)

    def test_non_namespace_child_is_refused_without_metadata_substitution(self):
        with self.assertRaisesRegex(ValueError, 'namespace identity'):
            self.collector.bind_root(self.root, sandbox=self.box)
        self.assertIsNone(self.collector.rootfd)

    def test_arbitrary_context_pid_and_foreign_path_cannot_enable_follow(self):
        with self.assertRaisesRegex(ValueError, 'native Sandbox'):
            self.collector.bind_root(self.root, sandbox=SimpleNamespace(init_pid=self.pid, _unshare=self.launch))
        with self.assertRaisesRegex(ValueError, 'live owned Sandbox'):
            self.collector.bind_root('/proc/self/root', sandbox=self.box)
        with self.assertRaisesRegex(ValueError, 'live owned Sandbox'):
            self.collector.bind_root(self.destination, sandbox=self.box)

    def test_ordinary_directory_symlink_is_refused(self):
        alias = self.destination / 'ordinary-symlink'
        alias.symlink_to(self.destination, target_is_directory=True)
        with self.assertRaises(OSError):
            self.collector.bind_root(alias)
        self.assertIsNone(self.collector.rootfd)

    def test_process_start_identity_drift_after_follow_is_refused(self):
        original = self.namespace_metadata()
        reads = 0
        def identity(fd, pid):
            nonlocal reads
            value = original(fd, pid)
            if pid == self.pid:
                reads += 1
                if reads > 1:
                    value['start_ticks'] += 1
            return value
        with mock.patch.object(capture, '_proc_identity', side_effect=identity):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.collector.bind_root(self.root, sandbox=self.box)
        self.assertIsNone(self.collector.rootfd)
        self.assertFalse((self.destination / 'root-binding.json').exists())

    def test_host_uid_mismatch_is_refused(self):
        original = self.namespace_metadata()
        def identity(fd, pid):
            value = original(fd, pid)
            if pid == self.pid:
                value['host_uids'] = [os.geteuid() + 1] * 4
            return value
        with mock.patch.object(capture, '_proc_identity', side_effect=identity):
            with self.assertRaisesRegex(ValueError, 'ownership'):
                self.collector.bind_root(self.root, sandbox=self.box)


if __name__ == '__main__':
    unittest.main()

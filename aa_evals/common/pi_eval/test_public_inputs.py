from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from . import runtime_cache as cache


class PublicInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task = self.root / 'hf-model-inference'
        (self.task / 'environment').mkdir(parents=True)
        (self.task / 'instruction.md').write_text('Download the public sentiment model; author a service.\n')
        self.payload = self.root / 'public'
        (self.payload / 'snapshot').mkdir(parents=True)
        (self.payload / 'snapshot/config.json').write_text('{"fake":"neutral public input"}')
        (self.payload / 'sources.json').write_text('{"url":"https://example.invalid/public"}')
        self.entry = {'instruction_sha256': cache.sha256(self.task / 'instruction.md'),
                      'payload_root': str(self.payload), 'native_rootfs_key': 'native-key',
                      'image_probe': 'neutral native image', 'source_index': 'sources.json',
                      'items': [{'kind': 'model', 'label': 'Unmodified public model', 'path': 'snapshot'}]}
        self.refresh_inventory()
        self.manifest = {'schema_version': 2, 'tasks': {}, 'public_inputs': {'hf-model-inference': self.entry}}
        self.file = self.root / 'manifest.json'
        self.calls, self.uploads = [], []
        parent = self

        class Environment:
            environment_dir = parent.task / 'environment'
            _main = SimpleNamespace(rootfs=Path('/cache/native-key/rootfs'))

            async def exec(self, command, **kwargs):
                parent.calls.append((command, kwargs))
                if 'sha256sum --check' in command:
                    paths = ['f ' + p for p in parent.entry['files']] + ['d ' + p for p in parent.entry['directories']]
                    return SimpleNamespace(return_code=0, stdout='\0'.join(paths) + '\0', stderr='')
                return SimpleNamespace(return_code=0, stdout='neutral native image', stderr='')

            async def upload_dir(self, source, destination):
                parent.uploads.append((source, destination))

        self.env = Environment()

    def tearDown(self):
        self.tmp.cleanup()

    def refresh_inventory(self):
        self.entry['files'] = {str(p.relative_to(self.payload)): cache.sha256(p)
                               for p in self.payload.rglob('*') if p.is_file()}
        self.entry['directories'] = sorted(str(p.relative_to(self.payload))
                                           for p in self.payload.rglob('*') if p.is_dir())

    async def prepare(self):
        self.file.write_text(json.dumps(self.manifest))
        return await cache.prepare(self.env, str(self.file), cache.sha256(self.file))

    async def test_selected_public_inputs_stage_without_install_or_global_policy(self):
        result = await self.prepare()
        self.assertEqual(self.uploads, [(self.payload, '/opt/aa-public-inputs')])
        self.assertEqual(result['agent_env'], {})
        self.assertIn('/opt/aa-public-inputs/snapshot', result['notice'])
        self.assertEqual(result['public_inputs']['source_index_sha256'], self.entry['files']['sources.json'])
        self.assertEqual(self.calls[-1][0], 'chmod -R a+rX /opt/aa-public-inputs')
        self.assertIn('test ! -e /opt/aa-public-inputs && test ! -L /opt/aa-public-inputs', self.calls[1][0])
        self.assertFalse((self.task / 'model_cache').exists())

    async def test_unselected_task_is_noop(self):
        self.env.environment_dir = self.root / 'other-task/environment'
        self.assertIsNone(await self.prepare())
        self.assertEqual((self.calls, self.uploads), ([], []))

    async def test_changed_instruction_is_rejected_before_exposure(self):
        (self.task / 'instruction.md').write_text('different task')
        with self.assertRaisesRegex(ValueError, 'instruction'):
            await self.prepare()
        self.assertFalse(self.uploads)

    async def test_existing_destination_is_rejected_before_copy(self):
        original = self.env.exec

        async def existing(command, **kwargs):
            if command.startswith('test ! -e'):
                return SimpleNamespace(return_code=1, stdout='', stderr='')
            return await original(command, **kwargs)

        self.env.exec = existing
        with self.assertRaisesRegex(ValueError, 'destination already exists'):
            await self.prepare()
        self.assertFalse(self.uploads)

    async def test_remote_copy_hash_or_inventory_mismatch_cannot_succeed(self):
        original = self.env.exec
        for return_code, inventory in [(1, ''), (0, 'f unexpected\0'), (0, 'l sources.json\0')]:
            async def invalid_copy(command, **kwargs):
                if 'sha256sum --check' in command:
                    return SimpleNamespace(return_code=return_code, stdout=inventory, stderr='')
                return await original(command, **kwargs)

            self.env.exec = invalid_copy
            with self.assertRaisesRegex(ValueError, 'staged public input'):
                await self.prepare()
        self.assertFalse(any(cmd.startswith('chmod') for cmd, _ in self.calls))

    async def test_changed_image_key_and_probe_are_rejected(self):
        self.env._main.rootfs = Path('/cache/other/rootfs')
        with self.assertRaisesRegex(ValueError, 'image key'):
            await self.prepare()
        self.env._main.rootfs = Path('/cache/native-key/rootfs')
        self.entry['image_probe'] = 'different installed packages'
        with self.assertRaisesRegex(ValueError, 'image inventory'):
            await self.prepare()
        self.assertFalse(self.uploads)

    async def test_changed_or_unlisted_public_bytes_are_rejected(self):
        (self.payload / 'snapshot/config.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            await self.prepare()
        self.refresh_inventory()
        (self.payload / 'unexpected').write_text('unreviewed')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            await self.prepare()
        self.assertFalse(self.uploads)

    async def test_links_and_empty_directory_changes_are_rejected(self):
        (self.payload / 'snapshot/link').symlink_to(self.payload / 'sources.json')
        with self.assertRaisesRegex(ValueError, 'link or special'):
            await self.prepare()
        (self.payload / 'snapshot/link').unlink()
        (self.payload / 'unlisted-empty').mkdir()
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            await self.prepare()
        self.assertFalse(self.uploads)

    async def test_descriptor_and_source_index_cannot_escape_inventory(self):
        for field, value in [('source_index', '../sources.json'), ('source_index', 'missing.json')]:
            self.entry[field] = value
            with self.assertRaises(ValueError):
                await self.prepare()
        self.entry['source_index'] = 'sources.json'
        for path in ['/absolute', '../escape', 'snapshot//config.json', 'not-present', 'snapshot/line\nname', 'snapshot/tab\tname']:
            self.entry['items'][0]['path'] = path
            with self.assertRaises(ValueError):
                await self.prepare()
        self.assertFalse(self.uploads)

    async def test_input_tree_rechecked_after_package_staging(self):
        # Package setup awaits I/O before input upload; a mutation during that
        # interval must not publish changed input bytes under an old manifest.
        from unittest.mock import patch

        async def mutate(*args):
            (self.payload / 'sources.json').write_text('changed during setup')

        with patch.object(cache, '_prepare_packages', mutate):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                await self.prepare()
        self.assertFalse(self.uploads)

    async def test_public_inputs_require_explicit_new_schema(self):
        self.manifest['schema_version'] = 1
        with self.assertRaisesRegex(ValueError, 'schema version 2'):
            await self.prepare()
        self.manifest['schema_version'] = True
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            await self.prepare()
        self.assertEqual((self.calls, self.uploads), ([], []))

    async def test_package_and_input_notices_merge_without_changing_package_env(self):
        from unittest.mock import patch, AsyncMock
        packages = {'task': 'hf-model-inference', 'notice': 'Install from local wheels.',
                    'agent_env': {'PIP_NO_INDEX': '1'}, 'requested': ['neutral'], 'managers': ['pip']}
        with patch.object(cache, '_prepare_packages', AsyncMock(return_value=packages)):
            result = await self.prepare()
        self.assertEqual(result['agent_env'], {'PIP_NO_INDEX': '1'})
        self.assertTrue(result['notice'].startswith('Install from local wheels.\n'))
        self.assertEqual(result['requested'], ['neutral'])


if __name__ == '__main__':
    unittest.main()

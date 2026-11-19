import io
import json
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest

from . import bundle
from aa_evals.common.pi_eval.required_outputs import SPECS


class CaptureOverlayTests(unittest.TestCase):
    def test_only_our_artifact_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'old-v4.tar', root / 'new-proposal.tar'
            config = ('artifacts = ' + json.dumps(bundle.ARTIFACTS).replace(': ', ' = ')
                      + '\n[agent]\ntimeout_sec = 21600\n').encode()
            files = {'tasks/caffe-cifar-10/task.toml': config,
                     'tasks/caffe-cifar-10/instruction.md': b'unchanged instruction',
                     'tasks/caffe-cifar-10/tests/test_outputs.py': b'unchanged official assertions',
                     'tasks/caffe-cifar-10/tests/official-assets/test_batch.bin': b'neutral private fixture',
                     'tasks/unrelated/environment/Dockerfile': b'unchanged other task'}
            with tarfile.open(source, 'w') as archive:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(data), 0o600
                    archive.addfile(info, io.BytesIO(data))
                link = tarfile.TarInfo('tasks/unrelated/link')
                link.type, link.linkname = tarfile.SYMTYPE, 'environment/Dockerfile'
                archive.addfile(link)
            result = bundle.narrow_artifacts(source, output)
            self.assertEqual(result['state'], 'FUTURE_PROPOSAL_NOT_ADOPTED')
            self.assertFalse(result['replay_complete'])
            self.assertEqual({x['source'] for x in bundle.FIXED_ARTIFACTS}, set(SPECS['caffe-cifar-10']))
            with tarfile.open(output) as archive:
                for name, data in files.items():
                    actual = archive.extractfile(name).read()
                    if name.endswith('task.toml'):
                        after, before = tomllib.loads(actual.decode()), tomllib.loads(data.decode())
                        self.assertEqual(after.pop('artifacts'), bundle.FIXED_ARTIFACTS)
                        before.pop('artifacts')
                        self.assertEqual(after, before)
                    else:
                        self.assertEqual(actual, data)
                self.assertEqual(archive.getmember('tasks/unrelated/link').linkname, 'environment/Dockerfile')
            with self.assertRaises(FileExistsError):
                bundle.narrow_artifacts(source, output)


if __name__ == '__main__':
    unittest.main()

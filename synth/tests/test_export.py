"""Exporter passes: finished exports reused without Lustre checks, a process pool, index checkpoints."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from synth import config as config_module
from synth import export
from synth.config import Campaign, load_teacher
from synth.serializers import serializer_for
from synth.wire import WireLog

SUPER_CHECKPOINT = Path('/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4')
TOOLS = [{'type': 'function', 'function': {'name': 'bash', 'description': 'b', 'parameters': {'type': 'object'}}}]


def make_attempt(raw, uid, finished=True, answer='done'):
    folder = raw / uid
    sessions = folder / 'harbor' / 't__abc' / 'agent' / 'pi' / 'sessions'
    sessions.mkdir(parents=True)
    (folder / 'attempt.json').write_text(json.dumps({'id': uid, 'kind': 'train', 'task': 't', 'attempt': 0}))
    if finished:
        (folder / 'worker-result.json').write_text(json.dumps({'interrupted': False}))
    events = [{'type': 'session', 'id': 's1'},
              {'type': 'message', 'id': 'u', 'parentId': None, 'message': {'role': 'user', 'content': 'Do it'}},
              {'type': 'message', 'id': 'a1', 'parentId': 'u', 'message': {'role': 'assistant', 'stopReason': 'stop',
                  'content': [{'type': 'thinking', 'thinking': 'plan'}, {'type': 'text', 'text': answer}]}}]
    (sessions / 's.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    log = WireLog(str(folder / 'harbor' / 'pi-cont16k' / 'wire.jsonl'))
    seq = log.request(json.dumps({'messages': [{'role': 'system', 'content': 'You are Pi.'},
                                               {'role': 'user', 'content': 'Do it'}], 'tools': TOOLS}).encode(), 'b', '/v1')
    log.end(seq, 200)
    log.close()
    return folder


@unittest.skipUnless(SUPER_CHECKPOINT.exists(), 'needs the Nemotron Super tokenizer')
class Passes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.campaign = Campaign({'id': 'unit', 'root': self.tmp.name, 'teacher': load_teacher('nemotron35-super'),
                                  'protocol': {**config_module.PROTOCOL_DEFAULTS},
                                  'train': {'task_list': ['t'], 'mode': 'per_task'}})
        self.serializer = serializer_for(self.campaign.teacher)
        self.raw = self.campaign.train / 'raw'
        self.index_path = self.campaign.train / 'export-index.json'

    def test_pool_matches_serial_and_skips_unfinished(self):
        for i in range(12):
            make_attempt(self.raw, f'x{i:02d}', finished=i != 5, answer=f'answer {i}')
        (self.raw / 'x99').mkdir()  # no attempt.json yet
        serial = export.run_once(self.campaign, 'train', self.serializer)
        records = {k: json.loads(gzip.open(v['path'], 'rt').read()) for k, v in serial.items()}
        self.index_path.unlink()
        pooled = export.run_once(self.campaign, 'train', self.serializer, workers=3)
        self.assertEqual(sorted(serial), [f'x{i:02d}' for i in range(12) if i != 5])
        strip = lambda r: {k: v for k, v in r.items() if k != 'updated_at'}
        for key, entry in pooled.items():
            self.assertTrue(entry['usable'])
            self.assertEqual({k: v for k, v in entry.items() if k != 'sha256'},
                             {k: v for k, v in serial[key].items() if k != 'sha256'})
            data = Path(entry['path']).read_bytes()
            self.assertEqual(entry['sha256'], export.hashlib.sha256(data).hexdigest())
            self.assertEqual(strip(json.loads(gzip.decompress(data))), strip(records[key]))
        self.assertEqual(json.loads(self.index_path.read_text()), pooled)
        self.assertEqual(list(json.loads(self.index_path.read_text())), sorted(pooled))

    def test_finished_usable_exports_are_reused_without_touching_the_folder(self):
        for i in range(3):
            make_attempt(self.raw, f'x{i}')
        first = export.run_once(self.campaign, 'train', self.serializer)
        make_attempt(self.raw, 'x3')
        touched = []
        real = export.export_folder
        def spy(campaign, serializer, kind, live, folder, old):
            touched.append(folder.name)
            return real(campaign, serializer, kind, live, folder, old)
        with mock.patch.object(export, 'export_folder', spy):
            second = export.run_once(self.campaign, 'train', self.serializer)
        self.assertEqual(touched, ['x3'])
        self.assertEqual({k: second[k] for k in first}, first)
        self.assertTrue(second['x3']['usable'])

    def test_unusable_exports_are_retried(self):
        folder = make_attempt(self.raw, 'x0')
        session = folder / 'harbor' / 't__abc' / 'agent' / 'pi' / 'sessions' / 's.jsonl'
        events = session.read_text()
        session.write_text('')
        first = export.run_once(self.campaign, 'train', self.serializer)
        self.assertFalse(first['x0']['usable'])
        self.assertTrue((self.campaign.train / 'quarantine' / 'x0.json').exists())
        session.write_text(events)  # e.g. a reviewed recovery made it exportable
        second = export.run_once(self.campaign, 'train', self.serializer)
        self.assertTrue(second['x0']['usable'])

    def test_a_stopped_pass_checkpoints_what_it_has(self):
        for i in range(6):
            make_attempt(self.raw, f'x{i}')
        # A previous index for x4 (exported by an older exporter version): kept until this pass re-exports it.
        stale = {'x4': {'id': 'x4', 'usable': True, 'tokens': 1, 'exporter_version': 'old', 'finalized': True}}
        export.atomic(self.index_path, stale)
        real = export.export_folder
        def stop_at_x3(campaign, serializer, kind, live, folder, old):
            if folder.name == 'x3':
                raise export.Stopped()
            return real(campaign, serializer, kind, live, folder, old)
        with mock.patch.object(export, 'export_folder', stop_at_x3), self.assertRaises(export.Stopped):
            export.run_once(self.campaign, 'train', self.serializer)
        saved = json.loads(self.index_path.read_text())
        self.assertEqual(sorted(saved), ['x0', 'x1', 'x2', 'x4'])
        self.assertEqual(saved['x4'], stale['x4'])
        self.assertTrue(all(saved[k]['usable'] and saved[k]['finalized'] for k in ('x0', 'x1', 'x2')))
        touched = []
        def spy(campaign, serializer, kind, live, folder, old):
            touched.append(folder.name)
            return real(campaign, serializer, kind, live, folder, old)
        with mock.patch.object(export, 'export_folder', spy):
            resumed = export.run_once(self.campaign, 'train', self.serializer)
        self.assertEqual(touched, ['x3', 'x4', 'x5'])
        self.assertEqual(sorted(resumed), [f'x{i}' for i in range(6)])
        self.assertEqual(resumed['x4']['exporter_version'], export.EXPORTER_VERSION)

    def test_periodic_checkpoint(self):
        for i in range(3):
            make_attempt(self.raw, f'x{i}')
        writes = []
        real = export.atomic
        def record(path, value, *args, **kwargs):
            if Path(path) == self.index_path:
                writes.append(sorted(value))
            return real(path, value, *args, **kwargs)
        with mock.patch.object(export, 'atomic', record):
            export.run_once(self.campaign, 'train', self.serializer, checkpoint_sec=0)
        self.assertEqual(writes, [['x0'], ['x0', 'x1'], ['x0', 'x1', 'x2'], ['x0', 'x1', 'x2']])


if __name__ == '__main__':
    unittest.main()

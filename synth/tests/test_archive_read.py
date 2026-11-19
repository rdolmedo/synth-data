"""synth.archive.read_attempts, the bank deposit and traces.new_attempts read a campaign consistently while
synth/archive.py packs it (#236, the #166 race on task-synthesis). Real packs (archive.pack with delete: shard, then
index.json, then rmtree) are injected between the index read and the directory reads, in the middle of one
directory read, and between the index read and the directory listing."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from synth import archive, bank
from synth.tests.test_archive import campaign

QUIET = dict(shard_size=10, workers=1, delete=True, log=lambda m: None)


def pack(root, ids):
    archive.pack(root, only=set(ids), **QUIET)


def strip(line):
    return {k: v for k, v in line.items() if k != 'record_from_manifest'}


class ConsistentRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = campaign(self.tmp)
        self.reference = {x['id']: x for x in archive.read_attempts(self.root)}   # nothing packed yet
        self.assertEqual(len(self.reference), 5)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def read(self, **kw):
        return {x['id']: x for x in archive.read_attempts(self.root, workers=1, chunk=2, **kw)}

    def test_pack_between_index_read_and_directory_reads(self):
        real, calls = archive.summarize, []

        def summarize(raw, *a, **kw):
            if not calls:   # the first directory read starts after t1..t3 were packed and deleted
                pack(self.root, ['c1-t1-a0', 'c1-t2-a0', 'c1-t3-a0'])
            calls.append(Path(raw).name)
            return real(raw, *a, **kw)
        with mock.patch.object(archive, 'summarize', summarize):
            got = self.read()
        self.assertFalse((self.root / 'train' / 'raw' / 'c1-t2-a0').exists())
        self.assertEqual(got, self.reference)

    def test_pack_in_the_middle_of_a_directory_read(self):
        """t2's sessions are read, then its directory is packed and deleted before its wire log is read."""
        real = archive.first_request

        def first_request(raw):
            if Path(raw).name == 'c1-t2-a0' and Path(raw).exists():
                pack(self.root, ['c1-t2-a0'])
            return real(raw)
        with mock.patch.object(archive, 'first_request', first_request):
            got = self.read()
        self.assertEqual(got, self.reference)

    def test_pack_between_index_read_and_directory_listing(self):
        """Packed and deleted before the directories are listed: in neither the first index nor on disk."""
        real, calls = archive.load_index, []

        def load_index(root, kind='train'):
            index = real(root, kind)
            if not calls:
                calls.append(1)
                with mock.patch.object(archive, 'load_index', real):   # the packer's own index reads
                    pack(self.root, ['c1-t0-a0', 'c1-t3-a0'])
            return index
        with mock.patch.object(archive, 'load_index', load_index):
            got = self.read()
        self.assertEqual(got, self.reference)

    def test_known_and_tasks_filters(self):
        pack(self.root, ['c1-t0-a0', 'c1-t1-a0'])
        self.assertEqual(sorted(self.read(known={'c1-t0-a0', 'c1-t2-a0'})), ['c1-t1-a0', 'c1-t3-a0', 'c1-t4-a0'])
        self.assertEqual(sorted(self.read(tasks={'t1', 't3'})), ['c1-t1-a0', 'c1-t3-a0'])

    def test_a_line_packed_before_its_export_was_finalized_takes_the_manifest_record(self):
        """m1-hero: 1,526 shard lines with "record": null; the manifest marks them usable and finalized later."""
        path = self.root / 'train' / 'manifest.json'
        manifest = json.loads(path.read_text())
        late = [r for r in manifest['records'] if r['id'] == 'c1-t3-a0']
        path.write_text(json.dumps({**manifest, 'records': [r for r in manifest['records'] if r not in late]}))
        pack(self.root, ['c1-t3-a0'])   # packed with no record; its export file stays
        path.write_text(json.dumps(manifest))
        line = self.read()['c1-t3-a0']
        self.assertTrue(line['record_from_manifest'])
        self.assertEqual(strip(line), self.reference['c1-t3-a0'])
        member = bank.deposit(Path(self.tmp) / 'bank', self.root)
        self.assertEqual(member['trajectories'], 4)

    def test_a_missing_attempt_fails_loudly(self):
        shutil.rmtree(self.root / 'train' / 'raw' / 'c1-t1-a0')   # deleted without a pack
        with self.assertRaises(archive.Incomplete) as e:
            self.read()
        self.assertEqual(e.exception.missing, {'c1-t1-a0': 'in neither a shard nor its directory'})
        logs = []
        got = {x['id'] for x in archive.read_attempts(self.root, check='warn', log=logs.append)}
        self.assertNotIn('c1-t1-a0', got)
        self.assertTrue(any('c1-t1-a0' in m for m in logs))
        with self.assertRaises(archive.Incomplete):
            bank.deposit(Path(self.tmp) / 'bank', self.root)

    def test_an_unreadable_session_fails_loudly(self):
        (self.root / 'train' / 'raw' / 'c1-t0-a0' / 'harbor' / 't0__x' / 'agent' / 'pi' / 'sessions' / 's.jsonl').unlink()
        with self.assertRaises(archive.Incomplete) as e:
            self.read()
        self.assertIn('s.jsonl', e.exception.missing['c1-t0-a0'])
        self.assertEqual(list(e.exception.missing), ['c1-t0-a0'])

    def test_bank_deposit_with_a_pack_during_its_directory_reads(self):
        loose = json.loads(json.dumps(bank.deposit(Path(self.tmp) / 'loose', self.root)))
        real, calls, lock = archive.verdict, [], threading.Lock()

        def verdict(raw):   # the deposit reads directories in threads: the first read packs, the others wait
            with lock:
                if not calls:
                    pack(self.root, ['c1-t0-a0', 'c1-t1-a0', 'c1-t2-a0'])
                calls.append(1)
            return real(raw)
        with mock.patch.object(archive, 'verdict', verdict):
            summary = bank.deposit(Path(self.tmp) / 'bank', self.root)
        self.assertEqual({k: summary[k] for k in ('trajectories', 'reward_1', 'verifier_infra_failures', 'truncated')},
                         {k: loose[k] for k in ('trajectories', 'reward_1', 'verifier_infra_failures', 'truncated')})
        member = json.loads((Path(self.tmp) / 'bank' / 'deepseek' / 'tmax' / 'members' / 'c1.json').read_text())
        by = {r['id']: r for r in member['records']}
        self.assertIn('archive', by['c1-t1-a0'])
        self.assertEqual((by['c1-t0-a0']['reward'], by['c1-t1-a0']['verifier_infra_failure'], by['c1-t2-a0']['truncated']),
                         (1.0, True, True))

    def test_traces_new_attempts_skips_known_and_warns(self):
        from task_synthesis.pipeline import traces
        pack(self.root, ['c1-t0-a0'])
        shutil.rmtree(self.root / 'train' / 'raw' / 'c1-t3-a0')
        with mock.patch('builtins.print') as out:
            got = [x['id'] for x in traces.new_attempts(self.root, {'c1-t1-a0'})]
        self.assertEqual(got, ['c1-t0-a0', 'c1-t2-a0', 'c1-t4-a0'])
        self.assertTrue(any('c1-t3-a0' in str(c) for c in out.call_args_list))

    def test_read_raw_after_a_pack(self):
        rel = 'harbor/t0__x/verifier/reward.txt'
        before = archive.read_raw(self.root, 'c1-t0-a0', rel)
        pack(self.root, ['c1-t0-a0'])
        self.assertEqual(archive.read_raw(self.root, 'c1-t0-a0', rel), before)
        with self.assertRaises(FileNotFoundError):
            archive.read_raw(self.root, 'c1-t1-a0', 'nope.txt')


if __name__ == '__main__':
    unittest.main()

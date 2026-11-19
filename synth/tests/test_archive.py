"""synth.archive: finished campaigns pack losslessly into JSONL shards + zips that training and the bank read."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from synth import archive, bank


def campaign(root, n=5):
    """A finished campaign: n attempt dirs (one interrupted, one with a proxy-broken verifier, one never exported)."""
    root = Path(root) / 'c1'
    (root / 'ops').mkdir(parents=True)
    (root / 'ops' / 'campaign.json').write_text(json.dumps({
        'id': 'c1', 'teacher': {'slug': 'deepseek', 'serving': {}},
        'train': {'bundle_source': '/x/datasets/tmax/t.tar'}, 'admission': {}}))
    (root / 'ops' / 'completion.json').write_text('{"status": "complete"}')
    records = []
    for i in range(n):
        uid = f'c1-t{i}-a0'
        raw = root / 'train' / 'raw' / uid
        trial = raw / 'harbor' / f't{i}__x'
        (trial / 'verifier').mkdir(parents=True)
        (trial / 'verifier' / 'reward.txt').write_text('1' if i % 2 == 0 else '0')
        (trial / 'verifier' / 'test-stdout.txt').write_text('Failed to connect to 10.22.0.107 port 8080' if i == 1 else 'ok')
        (trial / 'agent' / 'pi' / 'sessions').mkdir(parents=True)
        (trial / 'agent' / 'pi' / 'sessions' / 's.jsonl').write_text(f'{{"type": "message", "n": {i}}}\n')
        (raw / 'harbor' / 'pi-cont16k').mkdir(parents=True)
        body = {'messages': [{'role': 'system', 'content': 'sys'}], 'tools': [{'name': 'bash'}]}
        (raw / 'harbor' / 'pi-cont16k' / 'wire.jsonl').write_text(json.dumps({'t': 'req', 'body': body}) + '\n')
        (raw / 'harbor' / 'empty').mkdir()
        (raw / 'worker-result.json').write_text(json.dumps({'interrupted': i == 2}))
        (raw / 'attempt.json').write_text(json.dumps({'task': f't{i}', 'attempt': 0, 'retry': 0}))
        if i == 4:
            continue                                        # finished but never exported: no record
        export = root / 'train' / 'exports' / f'{uid}.json.gz'
        export.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(export, 'wt') as f:
            json.dump({'tokens': 100 + i}, f)
        records.append({'id': uid, 'task': f't{i}', 'usable': True, 'finalized': True, 'tokens': 100 + i,
                        'sha256': f's{i}', 'raw': str(raw), 'path': str(export),
                        'native_sessions': [f'harbor/t{i}__x/agent/pi/sessions/s.jsonl']})
    (root / 'train' / 'manifest.json').write_text(json.dumps({'trajectories_per_task': 8, 'records': records}))
    return root


class Archive(unittest.TestCase):
    def test_pack_is_lossless_resumable_and_readable(self):
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d)
            before = {x['id']: x for x in archive.iter_attempts(root)}
            files = {p.relative_to(root / 'train' / 'raw').as_posix(): p.read_bytes()
                     for p in (root / 'train' / 'raw').rglob('*') if p.is_file()}
            archive.pack(root, shard_size=2, workers=1, delete=False, log=lambda m: None)
            index = archive.pack(root, shard_size=2, workers=1, delete=True, log=lambda m: None)  # resumes, deletes
            self.assertEqual((len(index['shards']), len(index['attempts'])), (3, 5))
            self.assertEqual(list((root / 'train' / 'raw').iterdir()), [])
            self.assertEqual(list((root / 'train' / 'exports').iterdir()), [])
            after = {x['id']: x for x in archive.iter_attempts(root)}
            self.assertEqual(after, before, 'training lines unchanged by packing')
            x = after['c1-t0-a0']
            self.assertEqual((x['reward'], x['first_request']['tools'], x['sessions'][0]['text'], json.loads(x['export'])),
                             (1.0, [{'name': 'bash'}], '{"type": "message", "n": 0}\n', {'tokens': 100}))
            self.assertEqual((after['c1-t1-a0']['verifier_infra_failure'], after['c1-t2-a0']['truncated'],
                              after['c1-t4-a0']['record']), (True, True, None))
            for rel, data in files.items():                 # every file is still readable, byte for byte
                uid, rest = rel.split('/', 1)
                self.assertEqual(archive.read_raw(root, uid, rest), data)
            names = set()
            for z in (root / 'train' / 'attempts').glob('raw-*.zip'):
                names |= set(zipfile.ZipFile(z).namelist())
            self.assertIn('c1-t0-a0/harbor/empty/', names, 'empty directories are kept')
            self.assertEqual(archive.verify(root), (3, 5))

    def test_bank_deposit_reads_a_packed_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d)
            loose = bank.deposit(Path(d) / 'bank', root)
            archive.pack(root, shard_size=3, workers=1, delete=True, log=lambda m: None)
            packed = bank.deposit(Path(d) / 'bank', root)
            keys = ('trajectories', 'reward_1', 'verifier_infra_failures', 'truncated', 'tasks_at_8')
            self.assertEqual({k: packed[k] for k in keys}, {k: loose[k] for k in keys})
            member = json.loads((Path(d) / 'bank' / 'deepseek' / 'tmax' / 'members' / 'c1.json').read_text())
            ref = member['records'][0]['archive']
            with gzip.open(ref['lines'], 'rt') as f:
                self.assertEqual(json.loads(f.readlines()[ref['line']])['id'], member['records'][0]['id'])

    def test_one_pool_packs_several_campaigns(self):
        with tempfile.TemporaryDirectory() as d:
            roots = [campaign(Path(d) / 'a', n=3), campaign(Path(d) / 'b', n=4)]
            out = archive.pack_many(roots, shard_size=2, workers=2, delete=True, log=lambda m: None)
            self.assertEqual(sorted(len(i['attempts']) for i in out.values()), [3, 4])
            for root in roots:
                self.assertEqual(list((root / 'train' / 'raw').iterdir()), [])
                self.assertEqual(archive.verify(root)[1], len(list(archive.iter_attempts(root))))

    def test_published_but_unindexed_shards_are_adopted_not_repacked(self):
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d)
            before = {x['id']: x for x in archive.iter_attempts(root)}
            archive.pack(root, shard_size=2, workers=1, delete=False, log=lambda m: None)
            (root / 'train' / 'attempts' / 'index.json').unlink()        # killed before indexing
            logs = []
            index = archive.pack(root, shard_size=2, workers=1, delete=True, log=logs.append)
            self.assertIn('3 published shards adopted', logs[0])
            self.assertIn('0 to pack', logs[0])
            self.assertEqual(sorted(index['shards']), ['0', '1', '2'])
            self.assertEqual(list((root / 'train' / 'raw').iterdir()), [])
            self.assertEqual({x['id']: x for x in archive.iter_attempts(root)}, before)

    def test_symlinks_are_archived_as_links_and_removed_without_following(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d, n=2)
            keep = Path(d) / 'outside'
            keep.mkdir()
            (keep / 'precious').write_text('x')
            art = root / 'train' / 'raw' / 'c1-t0-a0' / 'harbor' / 't0__x' / 'artifacts'
            art.mkdir(parents=True)
            os.symlink('/nonexistent/bazel-out', art / 'bazel-app')        # dangling, as in a Bazel workspace
            os.symlink(str(keep), art / 'linked-dir')                       # to a directory outside the attempt
            archive.pack(root, shard_size=1, workers=1, delete=True, log=lambda m: None)
            self.assertEqual(archive.read_raw(root, 'c1-t0-a0', 'harbor/t0__x/artifacts/bazel-app'), b'/nonexistent/bazel-out')
            self.assertEqual(archive.read_raw(root, 'c1-t0-a0', 'harbor/t0__x/artifacts/linked-dir'), str(keep).encode())
            self.assertEqual(list((root / 'train' / 'raw').iterdir()), [])
            self.assertTrue((keep / 'precious').exists(), 'deleting the attempt never follows its symlinks')

    def test_a_failing_shard_does_not_stop_the_others(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d, n=4)
            os.mkfifo(root / 'train' / 'raw' / 'c1-t0-a0' / 'pipe')
            logs = []
            with self.assertRaisesRegex(RuntimeError, '1 shards failed'):
                archive.pack(root, shard_size=1, workers=1, delete=True, log=logs.append)
            self.assertTrue(any('FAILED shard' in m and 'not a regular file' in m for m in logs))
            self.assertEqual(sorted(p.name for p in (root / 'train' / 'raw').iterdir()), ['c1-t0-a0'])

    def test_owner_access_is_restored_before_packing(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d, n=2)
            agent = root / 'train' / 'raw' / 'c1-t0-a0' / 'harbor' / 't0__x' / 'agent'
            agent.mkdir(parents=True, exist_ok=True)
            (agent / 'pi.txt').write_text('log')
            os.chmod(agent / 'pi.txt', 0)
            os.chmod(agent, 0o644)                  # what an agent's sandbox left behind: no execute bit
            archive.pack(root, shard_size=1, workers=1, delete=True, log=lambda m: None)
            self.assertEqual(archive.read_raw(root, 'c1-t0-a0', 'harbor/t0__x/agent/pi.txt'), b'log')
            self.assertEqual(list((root / 'train' / 'raw').iterdir()), [])

    def test_refuses_a_running_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d)
            (root / 'ops' / 'completion.json').unlink()
            with self.assertRaises(SystemExit):
                archive.pack(root, workers=1, log=lambda m: None)


if __name__ == '__main__':
    unittest.main()


def full_config(d, root, **extra):
    """A complete campaign config for the fixture campaign (Campaign/Controller need teacher and protocol)."""
    from synth.config import PROTOCOL_DEFAULTS, ADMISSION_DEFAULTS, load_teacher
    config = {'id': 'c1', 'root': str(d), 'teacher': load_teacher('nemotron35-super'), 'protocol': dict(PROTOCOL_DEFAULTS),
              'admission': dict(ADMISSION_DEFAULTS), 'eval': {'enabled': False}, 'smoke': {'attempts': []},
              'train': {'mode': 'per_task', 'trajectories_per_task': 8, 'bundle_source': '/x/datasets/tmax/t.tar',
                        'task_list': [f't{i}' for i in range(5)]},
              'serving': {'jobs': 1, 'gpus_per_job': 4}, 'workers': {'count': 1, 'slots': 4}, **extra}
    (root / 'ops' / 'campaign.json').write_text(json.dumps(config))
    return config


class Pipeline(unittest.TestCase):
    """Packing is the default: the exporter keeps packed entries, the controller packs while running and at the end."""

    def test_exporter_keeps_entries_of_packed_attempts(self):
        from unittest import mock
        from synth import export
        from synth.config import Campaign
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d, n=2)
            c = Campaign(full_config(d, root))
            entry = {'id': 'c1-t0-a0', 'usable': True, 'finalized': True, 'exporter_version': export.EXPORTER_VERSION}
            (root / 'train' / 'export-index.json').write_text(json.dumps({'c1-t0-a0': entry}))
            archive.pack(root, shard_size=1, workers=1, delete=True, log=lambda m: None)   # both attempts packed
            (root / 'train' / 'raw' / 'c1-new-a0').mkdir(parents=True)                     # a new attempt
            with mock.patch.object(export, 'export_folder', lambda *a: {'id': 'c1-new-a0', 'usable': False}):
                index = export.run_once(c, 'train', serializer=None)
            self.assertEqual(sorted(index), ['c1-new-a0', 'c1-t0-a0'], 'the packed entry survives the pass')
            self.assertEqual(json.loads((root / 'train' / 'export-index.json').read_text())['c1-t0-a0'], entry)

    def test_controller_packs_finished_attempts_while_running(self):
        import time as _time
        from unittest import mock
        from synth import controller as controller_module
        from synth.config import Campaign
        with tempfile.TemporaryDirectory() as d:
            root = campaign(d, n=4)
            (root / 'ops' / 'completion.json').unlink()                  # still running
            config = full_config(d, root, archive={'shard_size': 2, 'workers': 1, 'min_age_sec': 600, 'interval_sec': 0})
            c = controller_module.Controller(Campaign(config))
            old = _time.time() - 3600
            c.s['items'] = {f'c1-t{i}-a0': {'id': f'c1-t{i}-a0', 'kind': 'train', 'status': 'finished', 'task': f't{i}',
                                            'finished_at': old if i < 3 else _time.time()} for i in range(4)}
            c.indices['train'] = {f'c1-t{i}-a0': {'usable': i != 1, 'finalized': True} for i in range(4)}
            with mock.patch.object(controller_module.subprocess, 'Popen') as popen:
                c.archive_tick()
            args = popen.call_args[0][0]
            only = json.loads(Path(args[args.index('--only') + 1]).read_text())
            self.assertEqual(only, ['c1-t0-a0', 'c1-t2-a0'], 'usable, final and older than min_age only')
            self.assertIn('--delete', args)

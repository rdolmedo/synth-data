"""roundpack (#245): packing a finished round is lossless, deletes a loose file only when the zip holds it as it is on
disk, and every round reader (RoundFiles and the readers routed through it) reads packed, partly packed and unpacked
rounds alike, consistently while a pack runs. Synthetic rounds only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import types
import unittest
from unittest import mock

from task_synthesis.pipeline import campaign, production, round as round_mod, roundpack
from task_synthesis.pipeline.roundpack import RoundFiles


def make_round(base, n=7, finished=True):
    """A round: specs s-00000.. with work/ (record or error, reply, check, .claim/owner, an empty dir) and tasks/
    (task.toml, instruction, an executable setup.sh, a symlink, an empty dir); s-00003 has an error only."""
    rd = Path(base) / 'rounds' / 'r1'
    specs = []
    for i in range(n):
        sid = f's-{i:05d}'
        specs.append({'spec_id': sid})
        wd, td = rd / 'work' / sid, rd / 'tasks' / sid
        (wd / '.claim').mkdir(parents=True)
        (wd / '.claim' / 'owner').write_text('job 1')
        (wd / 'attempt-0.reply.txt').write_text(f'reply {i}\n' * 50)
        (wd / 'attempt-0.check.json').write_text(json.dumps({'errors': [], 'snapshot': {'tree': ['a']}}))
        (wd / 'empty').mkdir()
        if i == 3:
            (wd / 'error.json').write_text(json.dumps({'spec_id': sid, 'status': 'error', 'error': 'boom'}))
        elif finished or i < n - 1:
            (wd / 'record.json').write_text(json.dumps({
                'spec_id': sid, 'status': 'valid' if i % 2 == 0 else 'invalid', 'instruction': f'do task {i}',
                'judge': {'pass': i % 4 == 0}, 'repairs': 0, 'attempts': [{'k': 0, 'ok': True}],
                'spec': {'minutes': 30}, 'meta': {'slug': f'slug-{i}'}}))
        if i == 3:
            continue
        (td / 'environment' / 'tasksynth').mkdir(parents=True)
        (td / 'task.toml').write_text(f'[metadata]\nname = "{sid}"\n')
        (td / 'instruction.md').write_text(f'do task {i}\n')
        (td / 'solution').mkdir()
        (td / 'solution' / 'sketch.md').write_text('sketch\n')
        setup = td / 'environment' / 'tasksynth' / 'setup.sh'
        setup.write_text('#!/bin/sh\necho hi\n')
        setup.chmod(0o755)
        os.symlink('../instruction.md', td / 'environment' / 'link.md')
        (td / 'tests').mkdir()
    (rd / 'specs.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in specs))
    if finished:
        (rd / 'STOP').write_text('budget\n')
    return rd


def snapshot(rd):
    """Every path under work/ and tasks/: file bytes and mode, symlink target, or 'dir' for a directory."""
    out = {}
    for area in ('work', 'tasks'):
        for p in sorted((rd / area).rglob('*')):
            rel = p.relative_to(rd).as_posix()
            if p.is_symlink():
                out[rel] = ('link', os.readlink(p))
            elif p.is_dir():
                out[rel] = 'dir'
            else:
                out[rel] = (p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
    return out


def loose_files(rd):
    return [p for area in ('work', 'tasks') for p in (rd / area).rglob('*') if not p.is_dir() or p.is_symlink()]


QUIET = dict(shard_size=3, workers=1, log=lambda m: None)


class RoundPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rd = make_round(self.tmp)
        patches = [mock.patch.object(roundpack, 'ROOT', Path(self.tmp)),
                   mock.patch.object(roundpack, 'ROUNDS', Path(self.tmp) / 'rounds'),
                   mock.patch.object(round_mod, 'ROUNDS', Path(self.tmp) / 'rounds'),
                   mock.patch.object(campaign, 'ROUNDS', Path(self.tmp) / 'rounds')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.before = snapshot(self.rd)
        self.collected = round_mod.collect('r1')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pack_and_delete_is_lossless_and_readable(self):
        out = roundpack.pack(self.rd, delete=True, **QUIET)
        self.assertEqual((out['shards'], out['specs'], out['kept']), (3, 7, 0))
        self.assertEqual(loose_files(self.rd), [])
        self.assertEqual(sorted(os.listdir(self.rd / 'work')), [], 'emptied spec directories are removed')
        self.assertEqual(roundpack.verify(self.rd), (3, 7))
        files = RoundFiles(self.rd)
        self.assertEqual(round_mod.collect('r1'), self.collected)
        for rel, value in self.before.items():
            area, sid, *rest = rel.split('/')
            if value == 'dir' or value[0] == 'link':
                continue
            self.assertEqual(files.read(sid, '/'.join(rest), area), value[0], rel)
        with self.assertRaises(FileNotFoundError):
            files.read('s-00000', 'nope.txt')
        self.assertTrue(files.exists('s-00001', 'task.toml', 'tasks'))
        self.assertFalse(files.exists('s-00003', 'task.toml', 'tasks'))
        dest = Path(self.tmp) / 'copy'
        for sid in ('s-00000', 's-00005'):
            files.copy_area(sid, 'tasks', dest / 'tasks' / sid)
            files.copy_area(sid, 'work', dest / 'work' / sid)
        copied = snapshot(dest)
        self.assertEqual(copied, {k: v for k, v in self.before.items() if k.split('/')[1] in ('s-00000', 's-00005')},
                         'files, modes, symlinks and empty directories come back')
        self.assertTrue(campaign.round_done('r1'))
        self.assertTrue(production.round_finished(self.rd))
        self.assertEqual([r['spec_id'] for r in campaign.judge_passed('r1')], ['s-00000', 's-00004'])
        driver = types.SimpleNamespace(rd=self.rd, records={})
        self.assertEqual(production.Driver.accepted(driver), 2)
        with self.assertRaises(SystemExit):   # a packed round never runs again
            round_mod.Round(argparse.Namespace(round='r1'))

    def test_partly_packed_round_reads_both(self):
        index, jobs, _ = roundpack.plan(self.rd, 3)
        with mock.patch.object(roundpack, 'plan', lambda rd, size: (index, jobs[:1], roundpack.folder(rd))):
            roundpack.pack(self.rd, delete=True, **QUIET)
        self.assertEqual(sorted(RoundFiles(self.rd).index()['specs']), ['s-00000', 's-00001', 's-00002'])
        self.assertTrue((self.rd / 'work' / 's-00004' / 'record.json').exists())
        self.assertEqual(round_mod.collect('r1'), self.collected)
        self.assertEqual(RoundFiles(self.rd).specs(), [f's-{i:05d}' for i in range(7)])

    def test_a_file_changed_or_added_after_packing_is_not_deleted(self):
        roundpack.pack(self.rd, delete=False, **QUIET)
        self.assertEqual(snapshot(self.rd), self.before, 'without --delete nothing changes on disk')
        changed = self.rd / 'work' / 's-00001' / 'attempt-0.reply.txt'
        changed.write_text('rewritten after the pack\n')
        added = self.rd / 'tasks' / 's-00002' / 'late.txt'
        added.write_text('new')
        roundpack.pack(self.rd, delete=True, **QUIET)   # resumes: deletes the shards packed earlier, file by file
        self.assertEqual(sorted(p.relative_to(self.rd).as_posix() for p in loose_files(self.rd)),
                         ['tasks/s-00002/late.txt', 'work/s-00001/attempt-0.reply.txt'])
        self.assertEqual(RoundFiles(self.rd).text('s-00001', 'attempt-0.reply.txt'), 'rewritten after the pack\n',
                         'a loose file is read first')
        index = roundpack.load_index(self.rd)
        self.assertEqual(sorted(sum((s.get('kept_files', []) for s in index['shards'].values()), [])),
                         ['tasks/s-00002/late.txt', 'work/s-00001/attempt-0.reply.txt'])

    def test_readiness(self):
        os.unlink(self.rd / 'STOP')
        shutil.rmtree(self.rd / 'work' / 's-00006')
        self.assertTrue(any('not finished' in w for w in roundpack.ready(self.rd)))
        with self.assertRaises(SystemExit):
            roundpack.pack(self.rd, **QUIET)
        with mock.patch.object(roundpack, 'live_round_job', lambda rd: '123.0'):
            with self.assertRaises(SystemExit):   # never while a round job is queued or running, even on demand
                roundpack.pack(self.rd, force=True, **QUIET)
        (self.rd / 'STOP').write_text('x')
        run = Path(self.tmp) / 'production' / 'p1'
        run.mkdir(parents=True)
        stream = Path(self.tmp) / 'streams' / 'p1'
        stream.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps({'round': 'r1', 'stream': str(stream)}))
        self.assertEqual(len(roundpack.ready(self.rd)), 2, roundpack.ready(self.rd))   # stream END, decontam list
        (stream / 'END.json').write_text('{}')
        (Path(self.tmp) / 'decon' / 'p1').mkdir(parents=True)
        (Path(self.tmp) / 'decon' / 'p1' / 'decontamination-variants.json').write_text('{}')
        self.assertEqual(roundpack.ready(self.rd), [])
        plan = roundpack.survey(self.rd, 3, 2)
        self.assertEqual((plan['specs_to_pack'], plan['shards'], plan['loose_files_to_pack']),
                         (7, 3, len(loose_files(self.rd)) + 0))

    def test_outcomes_while_a_pack_runs(self):
        """Packs injected between the index read and the loose reads, and between the index read and the listing."""
        real = roundpack.archive.consistent_reads

        def packed_first(paths, read_one, packed_now, *a, **kw):
            roundpack.pack(self.rd, delete=True, **QUIET)   # every spec packed and deleted before the reads
            return real(paths, read_one, packed_now, *a, **kw)
        with mock.patch.object(roundpack.archive, 'consistent_reads', packed_first):
            self.assertEqual(round_mod.collect('r1'), self.collected)
        self.tearDown()
        self.setUp()
        real_index, calls = RoundFiles.index, []

        def index(files, reload=False):
            out = real_index(files, reload)
            if not calls:
                calls.append(1)
                with mock.patch.object(RoundFiles, 'index', real_index):
                    roundpack.pack(self.rd, delete=True, **QUIET)
            return out
        with mock.patch.object(RoundFiles, 'index', index):
            self.assertEqual(round_mod.collect('r1'), self.collected)
        self.assertEqual(loose_files(self.rd), [])

    def test_read_falls_back_to_the_zip_when_the_loose_file_goes(self):
        files = RoundFiles(self.rd)
        self.assertIsNone(files.index()['specs'].get('s-00000'))   # index cached before the pack
        roundpack.pack(self.rd, delete=True, **QUIET)
        self.assertEqual(files.record('s-00000')['instruction'], 'do task 0')
        self.assertEqual(files.text('s-00000', 'instruction.md', 'tasks'), 'do task 0\n')


if __name__ == '__main__':
    unittest.main()

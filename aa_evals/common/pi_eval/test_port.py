"""Task inputs, SIF base meta and the import helpers of the Pi ports (#348). CPU only, no sandbox."""
from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from aa_evals.common.pi_eval import port, task_inputs


class TaskInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.task = self.tmp / 'task'
        (self.task / 'inputs/home/user/outputs').mkdir(parents=True)
        (self.task / 'inputs/home/user/a.txt').write_text('a')

    def test_roundtrip(self):
        m = task_inputs.write_manifest(self.task, {'note': 1})
        self.assertEqual(sorted(m['files']), ['home/user/a.txt'])
        self.assertIn('home/user/outputs', m['dirs'])   # empty output dir is staged too
        self.assertEqual(task_inputs.load(self.task)['note'], 1)

    def test_no_manifest_is_noop(self):
        self.assertIsNone(task_inputs.load(self.tmp / 'other'))

    def test_changed_inputs_refused(self):
        task_inputs.write_manifest(self.task)
        (self.task / 'inputs/home/user/a.txt').write_text('b')
        with self.assertRaises(ValueError):
            task_inputs.load(self.task)
        task_inputs.write_manifest(self.task)
        (self.task / 'inputs/home/user/extra').write_text('x')
        with self.assertRaises(ValueError):
            task_inputs.load(self.task)

    def test_links_refused(self):
        os.symlink('/etc/passwd', self.task / 'inputs/home/user/link')
        with self.assertRaises(ValueError):
            task_inputs.write_manifest(self.task)

    def test_unsafe_manifest_path(self):
        task_inputs.write_manifest(self.task)
        m = json.loads((self.task / 'inputs.json').read_text())
        m['dirs'].append('../escape')
        (self.task / 'inputs.json').write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            task_inputs.load(self.task)


def _trial_tar(path: Path, trial: str, files: dict[str, bytes], links: dict[str, str] = None):
    with tarfile.open(path, 'w:gz') as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(f'{trial}/{name}')
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        for name, target in (links or {}).items():
            ti = tarfile.TarInfo(f'{trial}/{name}')
            ti.type, ti.linkname = tarfile.SYMTYPE, target
            tf.addfile(ti)


class Extract(unittest.TestCase):
    def test_artifacts_session_and_receipt(self):
        tmp = Path(tempfile.mkdtemp())
        run = tmp / 'run'
        key = 't__a0__r0'
        (run / 'attempts' / key).mkdir(parents=True)
        trial = f'{key}/{key}/t__abc'
        session = [{'type': 'session'}, {'type': 'message', 'message': {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'x'}, {'type': 'text', 'text': 'Done: wrote report.docx'}]}}]
        _trial_tar(run / 'attempts' / key / 'trial.tar.gz', trial, {
            'artifacts/home/user/outputs/report.docx': b'docx',
            'artifacts/home/user/outputs/sub/b.csv': b'1,2',
            'artifacts/manifest.json': json.dumps([{'source': '/home/user/outputs', 'status': 'ok'}]).encode(),
            'agent/pi/sessions/s.jsonl': '\n'.join(json.dumps(e) for e in session).encode(),
            'task_inputs.json': b'{"n_files": 3}',
            'verifier/reward.txt': b'1\n',
            'agent/pi.txt': b'not wanted'},
            links={'artifacts/home/user/outputs/evil': '/etc/passwd'})
        (run / 'attempts' / key / 'outcome.json').write_text(json.dumps({
            'trial_archive': str(run / 'attempts' / key / 'trial.tar.gz'), 'trial_dir_in_archive': trial,
            'wall_sec': 12.5}))
        dest = tmp / 'dest'
        got = port.extract(run, {'key': key, 'transcript': ''}, dest, ['/home/user/outputs'])
        out = dest / 'artifacts/home/user/outputs'
        self.assertEqual(sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()),
                         ['report.docx', 'sub/b.csv'])
        self.assertFalse((out / 'evil').exists())   # links are never extracted, but they are reported
        self.assertEqual(got['links'], ['/home/user/outputs/evil'])
        self.assertTrue(got['found'])
        self.assertEqual(got['artifact_status'], {'/home/user/outputs': 'ok'})
        self.assertEqual(port.verifier_reward(got), 1.0)
        self.assertEqual(got['inputs_receipt'], {'n_files': 3})
        self.assertEqual(got['verifier']['reward.txt'].strip(), '1')
        self.assertEqual(got['wall_sec'], 12.5)
        self.assertEqual(port.final_text(got['session']), 'Done: wrote report.docx')

    def test_session_with_unicode_line_separators(self):
        """Pi (JSON.stringify) leaves U+0085 / U+2028 / U+2029 raw inside strings: they are data, not line breaks
        (str.splitlines() cut those records in two and the attempt never imported)."""
        tmp = Path(tempfile.mkdtemp())
        run = tmp / 'run'
        key = 't__a0__r0'
        (run / 'attempts' / key).mkdir(parents=True)
        trial = f'{key}/{key}/t__abc'
        text = 'Q1 rev\u0085up 4%\u2028next\u2029para \x0b\x0c\x1c'
        session = [{'type': 'session'}, {'type': 'message', 'message': {'role': 'assistant', 'content': [
            {'type': 'text', 'text': text}]}}]
        raw = '\n'.join(json.dumps(e, ensure_ascii=False) for e in session) + '\n'
        self.assertIn('\u2028', raw)
        self.assertNotEqual(len(raw.splitlines()), 2)   # the old reader's split
        _trial_tar(run / 'attempts' / key / 'trial.tar.gz', trial, {
            'artifacts/manifest.json': json.dumps([{'source': '/home/user/outputs', 'status': 'ok'}]).encode(),
            'agent/pi/sessions/s.jsonl': raw.encode()})
        (run / 'attempts' / key / 'outcome.json').write_text(json.dumps({
            'trial_archive': str(run / 'attempts' / key / 'trial.tar.gz'), 'trial_dir_in_archive': trial}))
        got = port.extract(run, {'key': key, 'transcript': ''}, tmp / 'dest', ['/home/user/outputs'])
        self.assertTrue(got['found'])
        self.assertEqual(len(got['session']), 2)
        self.assertEqual(port.final_text(got['session']), text)

    def test_session_line_overwritten_by_the_agent_is_recorded_not_fatal(self):
        """An agent overwrote its session file's header with its own non-JSON text (marin-lr1, 11-18): the attempt
        still imports (its score comes from the extracted state) and the bad line is recorded."""
        tmp = Path(tempfile.mkdtemp())
        run = tmp / 'run'
        key = 't__a0__r0'
        (run / 'attempts' / key).mkdir(parents=True)
        trial = f'{key}/{key}/t__abc'
        good = {'type': 'message', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Done'}]}}
        raw = '{record: "SUMMARY", data: {ok: true}}\n' + json.dumps(good) + '\n'
        _trial_tar(run / 'attempts' / key / 'trial.tar.gz', trial, {
            'artifacts/manifest.json': json.dumps([{'source': '/home/user/outputs', 'status': 'ok'}]).encode(),
            'agent/pi/sessions/s.jsonl': raw.encode()})
        (run / 'attempts' / key / 'outcome.json').write_text(json.dumps({
            'trial_archive': str(run / 'attempts' / key / 'trial.tar.gz'), 'trial_dir_in_archive': trial}))
        got = port.extract(run, {'key': key, 'transcript': ''}, tmp / 'dest', ['/home/user/outputs'])
        self.assertTrue(got['found'])
        self.assertEqual(got['session'], [good])
        self.assertEqual([b['line'] for b in got['session_bad_lines']], [1])
        self.assertEqual(port.final_text(got['session']), 'Done')

    def test_missing_archive_or_manifest_is_not_found(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / 'attempts/k').mkdir(parents=True)
        (tmp / 'attempts/k/outcome.json').write_text(json.dumps({'trial_dir_in_archive': 'k/k/t',
                                                                 'trial_archive': str(tmp / 'gone.tar.gz')}))
        got = port.extract(tmp, {'key': 'k', 'transcript': ''}, tmp / 'd', ['/x'])
        self.assertFalse(got['found'])
        self.assertIn('missing', got['why'])
        self.assertFalse(got['permanent'])   # a fresh record: the archive may not be visible yet
        _trial_tar(tmp / 'attempts/k/trial.tar.gz', 'k/k/t', {'agent/env_clock.json': b'{}'})
        (tmp / 'attempts/k/outcome.json').write_text(json.dumps({'trial_dir_in_archive': 'k/k/t',
                                                                 'trial_archive': str(tmp / 'attempts/k/trial.tar.gz')}))
        got = port.extract(tmp, {'key': 'k', 'transcript': ''}, tmp / 'd', ['/x'])
        self.assertFalse(got['found'])   # no artifact manifest: collection never ran
        self.assertIn('manifest', got['why'])
        self.assertTrue(got['permanent'])


class Ends(unittest.TestCase):
    def test_end_of(self):
        self.assertEqual(port.end_of({'end_state': 'env_budget_spent'}), 'time_limit')
        self.assertEqual(port.end_of({'exception_type': 'AgentTimeoutError'}), 'time_limit')
        self.assertEqual(port.end_of({'end_state': 'finished', 'context_limit': True}), 'context')
        self.assertEqual(port.end_of({'end_state': 'finished', 'context_limit': False}), 'finish')

    def test_base_args(self):
        a = port.base_args('b', 'r', 'serving:x', Path('/b.tar'), ['t1', 't2'], 2, 'ns', 'medium')
        self.assertEqual(a[a.index('--agent-cap-sec') + 1], '7200')
        self.assertIn('--no-multiuid', a)
        self.assertEqual(a[a.index('--env-class') + 1], port.ENVIRONMENT_CLASS)
        self.assertEqual(a[a.index('--include') + 1], 't1,t2')
        self.assertEqual(a[-2:], ['--reasoning-effort', 'medium'])
        self.assertEqual(a[a.index('--model-images') + 1], 'off')   # text-only: pi_eval refuses an image launch
        with self.assertRaises(SystemExit):
            port.base_args('b', 'r', 'serving:x', Path('/b.tar'), [], 2, 'ns')


if __name__ == '__main__':
    unittest.main()

"""GDPval under the Pi harness (#348): prompt adaptation, task build, import into the GDPval run layout. CPU only."""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from aa_evals.common.pi_eval import task_inputs
from aa_evals.gdpval import run as G
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.pi import run as PR
from aa_evals.gdpval.pi import tasks as PT


def fake_task(refs=()):
    return T.Task('00000000-test', 'Sector', 'Occupation', 'Write a memo about X.', list(refs), ['deliv/x.docx'])


class Prompt(unittest.TestCase):
    def test_no_finish_tool_or_step_budget(self):
        text = PT.pi_instruction(fake_task())
        for gone in ('`finish`', 'finish tool', 'abandon_task_finish', '250 steps', 'code_exec', 'terminated after 10 minutes',
                     'UID 1000'):
            self.assertNotIn(gone, text)
        self.assertIn(PT.OUTPUT_DIR, text)
        self.assertIn('Write a memo about X.', text)
        self.assertNotIn('## Reference Files Location', text)   # no references: section dropped, as on Stirrup
        self.assertTrue(text.rstrip().endswith('Please begin working on the task now.'))

    def test_reference_paths_listed(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(T, 'DATA', Path(d)):
            text = PT.pi_instruction(fake_task(['reference_files/abc/data.xlsx']))
        self.assertIn('- /home/user/data.xlsx', text)


class BuildTask(unittest.TestCase):
    def test_layout(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / 'data'
            (data / 'reference_files/abc').mkdir(parents=True)
            (data / 'reference_files/abc/data.csv').write_text('a,b\n')
            out = Path(d) / 'tasks'
            out.mkdir()
            with mock.patch.object(T, 'DATA', data):
                meta = PT.build_task(fake_task(['reference_files/abc/data.csv']), out, None)
            t = out / '00000000-test'
            cfg = tomllib.loads((t / 'task.toml').read_text())
            self.assertEqual(cfg['environment']['docker_image'], PT.BASE_REF)
            self.assertEqual(cfg['agent']['timeout_sec'], 21600.0)   # pi_eval's six-hour bundle form
            self.assertEqual(cfg['artifacts'], [PT.OUTPUT_DIR])
            m = task_inputs.load(t)
            self.assertEqual(sorted(m['files']), ['home/user/data.csv'])
            self.assertIn('home/user/outputs', m['dirs'])
            self.assertEqual(meta['n_input_files'], 1)
            self.assertTrue((t / 'tests/test.sh').stat().st_mode & 0o100)


def _attempt(pi_run: Path, key: str, files: dict[str, bytes], outputs_status='ok', reward='1', tsv=None,
             manifest=True):
    trial = f'{key}/{key}/trial'
    d = pi_run / 'attempts' / key
    d.mkdir(parents=True)
    files = dict(files)
    if manifest:
        files['artifacts/manifest.json'] = json.dumps([{'source': '/logs/artifacts', 'status': 'empty'},
                                                       {'source': PT.OUTPUT_DIR, 'status': outputs_status}]).encode()
    if reward is not None:
        files['verifier/reward.txt'] = (reward + '\n').encode()
    if tsv is None:
        tsv = ''.join(f'1\t{n.split("outputs/", 1)[1]}\n' for n in files if n.startswith('artifacts/home/user/outputs/'))
    files['verifier/outputs.tsv'] = tsv.encode()
    with tarfile.open(d / 'trial.tar.gz', 'w:gz') as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(f'{trial}/{name}')
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    (d / 'outcome.json').write_text(json.dumps({'trial_archive': str(d / 'trial.tar.gz'),
                                                'trial_dir_in_archive': trial, 'wall_sec': 99.0}))


def _row(key, **kw):
    item, a = key.split('__a')[0], int(key.split('__a')[1].split('__')[0])
    return {'item_id': item, 'repeat': a, 'status': 'ok', 'key': key, 'transcript': '', **kw}


class Init(unittest.TestCase):
    def test_unknown_or_empty_selection_refused(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / 'b.tar'
            b.with_suffix('.json').write_text('{}')
            a = argparse.Namespace(run_id='x-test-never-created', bundle=str(b), task_ids='nope,00000000-test',
                                   task_set='text-only-v1', limit=None, model='serving:x', repeats=1,
                                   reasoning_effort=None, pi_args=[], sif=None, aa_slug=None)
            with mock.patch.object(G, 'RUNS', Path(d)), mock.patch.object(G, 'select', return_value=[fake_task()]), \
                    mock.patch.object(PR.port, 'launch') as launch:
                with self.assertRaises(SystemExit):
                    PR.init(a)
                a.task_ids = None
                with mock.patch.object(G, 'select', return_value=[]), self.assertRaises(SystemExit):
                    PR.init(a)
                launch.assert_not_called()


class Import(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.pi_run, self.run_dir = d / 'pi', d / 'g'
        (self.run_dir / 'attempts').mkdir(parents=True)
        (self.run_dir / 'matches').mkdir()
        self.cfg = {'harness': {'pi_run': str(self.pi_run)}, 'agent_network': {'mode': 'offline'}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_ok_and_infra(self):
        _attempt(self.pi_run, 'tid__a0__r0', {
            'artifacts/home/user/outputs/memo.docx': b'x',
            'agent/pi/sessions/s.jsonl': json.dumps({'type': 'message', 'message': {
                'role': 'assistant', 'content': [{'type': 'text', 'text': 'summary'}]}}).encode()})
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0', end_state='finished', context_limit=True,
                                                           model_requests=40, output_cap_hits=2))
        self.assertEqual((o['status'], o['end'], o['submitted_files']), ('ok', 'context', ['memo.docx']))
        self.assertEqual(o['client']['output_cap_hits'], 2)
        self.assertEqual(G.attempt_state(self.run_dir, 'tid__r0'), 'render')   # deliverables await render.py
        tr = json.loads(gzip.open(self.run_dir / 'attempts/tid__r0/transcript.json.gz').read())
        self.assertEqual(tr['final_message'], 'summary')
        self.assertFalse((self.run_dir / 'attempts/tid__r0/artifacts').exists())
        bad = PR.import_attempt(self.run_dir, self.cfg, {'item_id': 'tid', 'repeat': 1, 'status': 'infra_error',
                                                         'infrastructure_reason': 'environment_start', 'key': 'x'})
        self.assertEqual(bad['status'], 'infra_error')
        self.assertEqual(G.attempt_state(self.run_dir, 'tid__r1'), 'done')   # pi_eval already spent the retries

    def test_session_line_the_agent_overwrote_is_recorded(self):
        """port.extract keeps a session whose header the agent overwrote (marin-lr1, 11-18); the outcome records the
        bad line instead of dropping it silently, and the attempt still imports."""
        msg = json.dumps({'type': 'message', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 's'}]}})
        _attempt(self.pi_run, 'tid__a0__r0', {'artifacts/home/user/outputs/memo.docx': b'x',
                                              'agent/pi/sessions/s.jsonl': ('{not json\n' + msg + '\n').encode()})
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0', end_state='finished'))
        self.assertEqual((o['status'], [b['line'] for b in o['session_bad_lines']]), ('ok', [1]))
        _attempt(self.pi_run, 'tid__a1__r0', {'artifacts/home/user/outputs/memo.docx': b'x',
                                              'agent/pi/sessions/s.jsonl': (msg + '\n').encode()})
        self.assertNotIn('session_bad_lines', PR.import_attempt(self.run_dir, self.cfg, _row('tid__a1__r0')))

    def test_no_files_is_a_model_outcome_only_when_the_verifier_agrees(self):
        _attempt(self.pi_run, 'tid__a0__r0', {}, outputs_status='empty', reward='0')
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0', end_state='env_budget_spent'))
        self.assertEqual((o['status'], o['end'], o['n_submitted']), ('ok', 'time_limit', 0))
        self.assertFalse((self.run_dir / 'attempts/tid__r0/deliverables').exists())   # judged as "no files"
        _attempt(self.pi_run, 'tid__a1__r0', {}, outputs_status='failed', reward='1', tsv='5\treport.pdf\n')
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a1__r0'))
        self.assertEqual((o['status'], o['end']), ('infra_error', 'infra'))   # collection lost the file

    def test_partial_collection_is_infra(self):
        _attempt(self.pi_run, 'tid__a0__r0', {'artifacts/home/user/outputs/a.txt': b'a'},
                 tsv='1\ta.txt\n1\tb.txt\n')
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0'))
        self.assertEqual(o['status'], 'infra_error')
        self.assertIn('incomplete', o['error'])

    def test_unreadable_trial(self):
        import os
        import time
        # an archive not visible yet: retried
        (self.pi_run / 'attempts/tid__a0__r0').mkdir(parents=True)
        rec = self.pi_run / 'attempts/tid__a0__r0/outcome.json'
        rec.write_text(json.dumps({'trial_archive': str(rec.parent / 'trial.tar.gz'), 'trial_dir_in_archive': 't'}))
        self.assertIsNone(PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0')))
        self.assertFalse((self.run_dir / 'attempts/tid__r0/outcome.json').exists())
        # ...still missing an hour later: lost, an infrastructure failure
        old = time.time() - 2 * 3600
        os.utime(rec, (old, old))
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a0__r0'))
        self.assertEqual((o['status'], o['end']), ('infra_error', 'infra'))
        self.assertIn('archive missing', o['error'])
        self.assertEqual(G.attempt_state(self.run_dir, 'tid__r0'), 'done')   # collect --follow terminates
        # pack error: immediately infra
        (self.pi_run / 'attempts/tid__a1__r0').mkdir(parents=True)
        (self.pi_run / 'attempts/tid__a1__r0/outcome.json').write_text(json.dumps({'pack_error': "OSError('quota')"}))
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a1__r0'))
        self.assertIn('pack_error', o['error'])
        # an archive without the artifact manifest (collection never ran): immediately infra
        _attempt(self.pi_run, 'tid__a2__r0', {}, manifest=False)
        o = PR.import_attempt(self.run_dir, self.cfg, _row('tid__a2__r0'))
        self.assertEqual(o['status'], 'infra_error')
        self.assertIn('manifest', o['error'])

    def test_supersede_drops_render_and_matches(self):
        d = self.run_dir / 'attempts/tid__r0'
        (d / 'render').mkdir(parents=True)
        (d / 'render/manifest.json').write_text('{}')
        m = self.run_dir / 'matches/arm.jsonl'
        m.write_text(json.dumps({'key': 'tid__r0', 'order': 'MA', 'winner': 'A'}) + '\n'
                     + json.dumps({'key': 'other__r0', 'order': 'MA', 'winner': 'B'}) + '\n')
        marker = self.run_dir / 'matches' / (G.JUDGE_MARKER + '123-host')
        marker.write_text('{}')
        self.assertTrue(G.judge_active(self.run_dir))
        self.assertFalse(PR._supersede(self.run_dir, 'tid__r0'))   # a judge appends to the file: wait
        self.assertTrue((d / 'render').exists())
        marker.unlink()
        self.assertTrue(PR._supersede(self.run_dir, 'tid__r0'))
        self.assertFalse((d / 'render').exists())
        self.assertEqual([json.loads(x)['key'] for x in m.read_text().splitlines()], ['other__r0'])
        self.assertIn('tid__r0', (self.run_dir / 'matches/superseded/arm.jsonl').read_text())
        self.assertEqual(sorted(p.name for p in (self.run_dir / 'matches').glob('*.jsonl')), ['arm.jsonl'])


if __name__ == '__main__':
    unittest.main()

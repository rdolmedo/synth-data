"""CritPt under the Pi harness (#348): prompt adaptation, answer-file import. CPU only, no sandbox."""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from aa_evals.critpt import data, run as C
from aa_evals.critpt.pi import run as PR
from aa_evals.critpt.pi import tasks as PT

TEMPLATE = 'def answer():\n    x = ...\n    return x\n'


def problem():
    return data.Problem('Challenge_99_main', 'Compute the thing.', TEMPLATE)


class Prompt(unittest.TestCase):
    def test_answer_file_replaces_finish_and_turns(self):
        t = PT.user_text(problem())
        for gone in ('`finish`', 'code_exec', 'turns', '0 seconds'):
            self.assertNotIn(gone, t)
        self.assertIn(PT.ANSWER, t)
        self.assertIn(TEMPLATE.strip(), t)
        self.assertIn('Compute the thing.', t)

    def test_system_prompt_first(self):
        self.assertTrue(PT.pi_instruction(problem()).startswith('You are a physics research assistant'))


class Answer(unittest.TestCase):
    def test_answer_of(self):
        self.assertEqual(PR.answer_of('```python\ndef answer():\n    return 1\n```\n')[0],
                         'def answer():\n    return 1')
        self.assertIsNone(PR.answer_of(None)[0])
        self.assertIsNone(PR.answer_of('  \n')[0])
        code, why = PR.answer_of('def answer(:\n')
        self.assertIsNone(code)
        self.assertIn('not valid Python', why)


def _attempt(pi_run: Path, key: str, files: dict[str, bytes], status='ok', reward='1', manifest=True):
    trial = f'{key}/{key}/trial'
    d = pi_run / 'attempts' / key
    d.mkdir(parents=True)
    files = dict(files)
    if manifest:
        files['artifacts/manifest.json'] = json.dumps([{'source': PT.ANSWER, 'status': status}]).encode()
    if reward is not None:
        files['verifier/reward.txt'] = (reward + '\n').encode()
    with tarfile.open(d / 'trial.tar.gz', 'w:gz') as tf:
        for name, b in files.items():
            ti = tarfile.TarInfo(f'{trial}/{name}')
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    (d / 'outcome.json').write_text(json.dumps({'trial_archive': str(d / 'trial.tar.gz'),
                                                'trial_dir_in_archive': trial, 'wall_sec': 5.0}))


class Import(unittest.TestCase):
    def test_rows(self):
        with tempfile.TemporaryDirectory() as d:
            pi_run, run_dir, scratch = Path(d) / 'pi', Path(d) / 'c', Path(d) / 's'
            run_dir.mkdir()
            cfg = {'harness': {'pi_run': str(pi_run)}}
            _attempt(pi_run, 'P__a0__r0', {'artifacts/work/answer.py': b'def answer():\n    return 2\n'})
            _attempt(pi_run, 'P__a1__r0', {'agent/env_clock.json': b'{}'}, status='failed', reward='0')
            _attempt(pi_run, 'P__a3__r0', {}, status='failed', reward='1')
            _attempt(pi_run, 'P__a4__r0', {}, manifest=False)
            row = lambda k, **kw: {'item_id': 'P', 'repeat': int(k.split('__a')[1][0]), 'status': 'ok', 'key': k, 'transcript': '', **kw}
            ok = PR.import_row(run_dir, cfg, row('P__a0__r0', model_requests=3), scratch)
            self.assertEqual((ok['status'], ok['score'], ok['turns']), ('ok', None, 3))
            self.assertIn('return 2', ok['answer_code'])
            miss = PR.import_row(run_dir, cfg, row('P__a1__r0', context_limit=True), scratch)
            self.assertEqual((miss['status'], miss['score']), ('model_error', 0))
            self.assertTrue(miss['error'].startswith('no_finish'))   # counted with Stirrup's no-finish errors
            self.assertIn('end=context', miss['error'])
            lost = PR.import_row(run_dir, cfg, row('P__a3__r0'), scratch)   # verifier saw an answer, collection lost it
            self.assertEqual((lost['status'], lost['score']), ('infra_error', None))
            gone = PR.import_row(run_dir, cfg, row('P__a4__r0'), scratch)   # no artifact manifest: infra, final
            self.assertEqual(gone['status'], 'infra_error')
            (pi_run / 'attempts/P__a5__r0').mkdir(parents=True)
            rec = pi_run / 'attempts/P__a5__r0/outcome.json'
            rec.write_text(json.dumps({'trial_archive': str(rec.parent / 'x.tar.gz'), 'trial_dir_in_archive': 't'}))
            self.assertIsNone(PR.import_row(run_dir, cfg, row('P__a5__r0'), scratch))   # not visible yet: retried
            infra = PR.import_row(run_dir, cfg, {'item_id': 'P', 'repeat': 2, 'status': 'infra_error',
                                                 'key': 'P__a2__r6'}, scratch)
            self.assertEqual(infra['status'], 'infra_error')
            for r in (ok, miss, infra):
                C.append_jsonl(run_dir / 'items.jsonl', r)
            st = C.final_states(C.read_jsonl(run_dir / 'items.jsonl'))
            self.assertEqual(st[('P', 0)]['attempts'], 1)


class SessionBadLines(unittest.TestCase):
    def test_overwritten_session_line_is_recorded_and_the_attempt_scores(self):
        """port.extract keeps a session whose header the agent overwrote (marin-lr1, 11-18); CritPt scores from the
        session when there is no answer file, so the row records the bad line instead of dropping it silently."""
        with tempfile.TemporaryDirectory() as d:
            pi_run, run_dir, scratch = Path(d) / 'pi', Path(d) / 'c', Path(d) / 's'
            run_dir.mkdir()
            cfg = {'harness': {'pi_run': str(pi_run)}}
            msg = json.dumps({'type': 'message', 'message': {'role': 'assistant', 'content': [
                {'type': 'text', 'text': '```python\ndef answer():\n    return 2\n```'}]}})
            _attempt(pi_run, 'P__a0__r0', {'agent/pi/sessions/s.jsonl': ('{not json\n' + msg + '\n').encode()},
                     status='failed', reward='0')
            r = PR.import_row(run_dir, cfg, {'item_id': 'P', 'repeat': 0, 'status': 'ok', 'key': 'P__a0__r0',
                                            'transcript': '', 'end_state': 'finished'}, scratch)
            self.assertEqual([b['line'] for b in r['session_bad_lines']], [1])
            self.assertIn('return 2', r['answer_code'])
            _attempt(pi_run, 'P__a1__r0', {'agent/pi/sessions/s.jsonl': (msg + '\n').encode()}, status='failed', reward='0')
            r = PR.import_row(run_dir, cfg, {'item_id': 'P', 'repeat': 1, 'status': 'ok', 'key': 'P__a1__r0',
                                            'transcript': '', 'end_state': 'finished'}, scratch)
            self.assertNotIn('session_bad_lines', r)


def _session(text):
    return json.dumps({'type': 'message', 'message': {'role': 'assistant', 'content': [
        {'type': 'thinking', 'thinking': 'x'}, {'type': 'text', 'text': text}]}}).encode()


class FinalMessageFallback(unittest.TestCase):
    MSG = 'Final Answer: 2\n\n```python\ndef answer():\n    return 1\n```\nand the filled template:\n```python\ndef answer():\n    return 2\n```'

    def test_only_for_finish_without_file(self):
        with tempfile.TemporaryDirectory() as d:
            pi_run, run_dir, scratch = Path(d) / 'pi', Path(d) / 'c', Path(d) / 's'
            run_dir.mkdir()
            cfg = {'harness': {'pi_run': str(pi_run)}}
            sess = {'agent/pi/sessions/s.jsonl': _session(self.MSG)}
            _attempt(pi_run, 'P__a0__r0', sess, status='failed', reward='0')
            _attempt(pi_run, 'P__a1__r0', sess, status='failed', reward='0')
            _attempt(pi_run, 'P__a2__r0', {**sess, 'artifacts/work/answer.py': b'def answer():\n    return 9\n'})
            _attempt(pi_run, 'P__a3__r0', {'agent/pi/sessions/s.jsonl': _session('no code here')},
                     status='failed', reward='0')
            row = lambda k, **kw: {'item_id': 'P', 'repeat': int(k.split('__a')[1][0]), 'status': 'ok', 'key': k,
                                   'transcript': '', **kw}
            fm = PR.import_row(run_dir, cfg, row('P__a0__r0', end_state='finished'), scratch)
            self.assertEqual((fm['status'], fm['answer_source']), ('ok', 'final_message'))
            self.assertIn('return 2', fm['answer_code'])   # the last block
            ctx = PR.import_row(run_dir, cfg, row('P__a1__r0', context_limit=True), scratch)
            self.assertEqual((ctx['status'], ctx['answer_source']), ('model_error', None))   # mid-thought: no fallback
            f = PR.import_row(run_dir, cfg, row('P__a2__r0'), scratch)
            self.assertEqual((f['answer_source'], 'return 9' in f['answer_code']), ('file', True))
            none = PR.import_row(run_dir, cfg, row('P__a3__r0'), scratch)
            self.assertEqual(none['status'], 'model_error')


if __name__ == '__main__':
    unittest.main()

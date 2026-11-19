"""AutomationBench under the Pi harness (#348): world service, `ab` CLI, grading, bundle task and run arguments.

  /opt/aai/venvs/automationbench/bin/python -m unittest aa_evals.automationbench.test_pi

CPU only, no sandbox or endpoint. Reads the public task data from the pinned upstream clone (never copied here).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from aa_evals.automationbench.pi import replay as RP
from aa_evals.automationbench.pi import run as PR
from aa_evals.automationbench.pi import tasks as PT

HERE = Path(__file__).resolve().parent
W = RP.world_module()
TASK = 'finance.capex_approval'
# The update Qwen3.8 Xhigh made in M3.6 (rows 2-4 of the Pending sheet): it breaks the row-3 guardrail, which
# upstream's rubric sees only through google_sheets._updated_row_keys (not a pydantic field).
UPDATE = {'method': 'PUT', 'url': 'https://sheets.googleapis.com/v4/spreadsheets/ss_capex/values/Pending!I2:I4',
          'params': '{"valueInputOption": "USER_ENTERED"}',
          'body': '{"range": "Pending!I2:I4", "values": [["Approved"], ["Denied"], ["Denied"]]}'}

_ROWS: dict = {}


def row(name: str = TASK) -> dict:
    if not _ROWS:
        from aa_evals.automationbench.run import load_tasks
        _ROWS.update({r['_task_name']: r for r in load_tasks(PT.DOMAINS)})
    return _ROWS[name]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / 'task.json').write_text(json.dumps(PT.world_spec(row())))
        self.world = W.World(self.tmp / 'task.json', self.tmp / 'state')

    def state(self, name):
        return (self.tmp / 'state' / name).read_text()


class WorldService(Base):
    def test_start_state_matches_upstream_setup(self):
        from automationbench.runner import compute_allowed_services, strip_none_values
        info = json.loads(row()['info']) if isinstance(row()['info'], str) else row()['info']
        init = strip_none_values(info['initial_state'])
        self.assertEqual(self.world.world.meta.allowed_services,
                         compute_allowed_services(init, [strip_none_values(a) for a in info['assertions']],
                                                  info.get('zapier_tools', [])))
        self.assertEqual(json.loads(self.state('hidden.json')), {})
        self.assertEqual(self.state('calls.jsonl'), '')

    def test_call_persists_and_hidden_state_saved(self):
        out, ok = self.world.call('api_fetch', UPDATE)
        self.assertTrue(ok)
        self.assertIn('"updatedRows": 3', out)
        hidden = json.loads(self.state('hidden.json'))
        self.assertEqual(sorted(hidden['google_sheets']['_updated_row_keys']['__set__']),
                         sorted(f'ss_capex:ws_pending_capex:{i}' for i in (2, 3, 4)))
        calls = [json.loads(x) for x in self.state('calls.jsonl').splitlines()]
        self.assertEqual([c['tool'] for c in calls], ['api_fetch'])

    def test_upstream_argument_handling(self):
        # {} means "not given" (update_tool_args)
        out, ok = self.world.call('api_search', {'query': 'slack post message', 'top_k': {}})
        self.assertTrue(ok)
        self.assertEqual(len(json.loads(out)['results']), 5)          # top_k's default
        out, ok = self.world.call('nope', {})
        self.assertFalse(ok)
        self.assertIn('unknown tool', out)
        out, ok = self.world.call('api_search', ['q'])
        self.assertFalse(ok)
        out, ok = self.world.call('api_search', {'bogus': 1})          # str(exception), as verifiers returns
        self.assertFalse(ok)
        self.assertIn('bogus', out)
        # `world` is injected only into tools that take it (upstream's skipped args); elsewhere it is an error there too
        self.assertFalse(self.world.call('api_search', {'query': 'x', 'world': 'x'})[1])
        out, ok = self.world.call('base64_encode', {'text': 'hi'})
        self.assertEqual((out, ok), ('aGk=', True))


class Grading(Base):
    def test_hidden_state_decides_the_guardrail(self):
        self.world.call('api_fetch', UPDATE)
        hidden = json.loads(self.state('hidden.json'))
        g = PR.grade(row(), self.state('world.json'), hidden)
        self.assertEqual(g['n_guardrails_broken'], 1)
        self.assertEqual(g['final_state_score'], 0.0)
        # without it the dropped tracking would hide the broken guardrail
        self.assertEqual(PR.grade(row(), self.state('world.json'))['n_guardrails_broken'], 0)

    def test_same_as_upstream_rubric_on_the_live_world(self):
        from automationbench.rubric import partial_credit
        from automationbench.runner import strip_none_values
        self.world.call('api_fetch', UPDATE)
        info = json.loads(row()['info']) if isinstance(row()['info'], str) else row()['info']
        live = partial_credit({'info': {**info, 'assertions': [strip_none_values(a) for a in info['assertions']]},
                               'world': self.world.world,
                               'initial_state': strip_none_values(info['initial_state'])})
        g = PR.grade(row(), self.state('world.json'), json.loads(self.state('hidden.json')))
        self.assertEqual(g['upstream_partial_credit'], live)

    def test_initial_world_grades_like_doing_nothing(self):
        g = PR.grade(row(), self.state('world.json'), json.loads(self.state('hidden.json')))
        self.assertEqual(g['n_objectives_met'], 0)
        self.assertEqual(g['n_guardrails_broken'], 0)


class Replay(Base):
    def test_align_and_remap(self):
        idmap = {}
        self.assertTrue(RP.align('{"id": "0977734d4d67418b", "ts": "1794423581.419499"}',
                                 '{"id": "7094444ca7cb492f", "ts": "1794076856.879306"}', idmap))
        self.assertEqual(RP.remap('{"messageId": "7094444ca7cb492f"}', idmap), '{"messageId": "0977734d4d67418b"}')
        self.assertFalse(RP.align('{"id": "0977734d4d67418b", "x": 1}', '{"id": "7094444ca7cb492f", "x": 2}', {}))
        idmap = {}
        self.assertTrue(RP.align('{"id": 995007110, "at": "2026-11-11T20:10:56.578403"}',
                                 '{"id": 2614908053, "at": "2026-11-07T17:26:09.439177"}', idmap))
        self.assertEqual(RP.remap('/meetings/2614908053/registrants', idmap), '/meetings/995007110/registrants')
        self.assertFalse(RP.align('{"count": 12}', '{"count": 13}', {}))         # small numbers are content

    def test_replay_one_skips_unanswered_and_unparseable(self):
        msgs = [{'role': 'assistant', 'tool_calls': [
                    {'id': 'a', 'name': 'api_fetch', 'arguments': json.dumps(UPDATE)},
                    {'id': 'b', 'name': 'api_fetch', 'arguments': '{not json'},
                    {'id': 'c', 'name': 'api_search', 'arguments': '{"query": "x"}'}]},
                {'role': 'tool', 'tool_call_id': 'a', 'content': None},
                {'role': 'tool', 'tool_call_id': 'b', 'content': 'Expecting property name'}]
        tmp = Path(tempfile.mkdtemp())
        out, _ = W.World(self.tmp / 'task.json', tmp / 'scratch').call('api_fetch', UPDATE)
        msgs[1]['content'] = out
        w2 = W.World(self.tmp / 'task.json', tmp / 'expected')
        w2.call('api_fetch', UPDATE)
        expected = PR.grade(row(), (tmp / 'expected/world.json').read_text(),
                            json.loads((tmp / 'expected/hidden.json').read_text()))
        item = {'item_id': TASK, 'repeat': 0, 'status': 'ok', **{k: expected[k] for k in RP.KEYS}}
        res = RP.replay_one(W, row(), item, {'messages': msgs}, tmp)
        self.assertEqual((res['n_calls'], res['n_result_match'], res['n_skipped'], res['n_unanswered']), (1, 1, 1, 1))
        self.assertEqual(res['grade_diff'], {})


class Cli(Base):
    def setUp(self):
        super().setUp()
        self.srv = W.make_server(self.world, '127.0.0.1', 0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        tools = self.tmp / 'tools.json'
        tools.write_text(json.dumps(PT.tool_schemas()))
        self.env = dict(os.environ, AB_WORLD_URL=f'http://127.0.0.1:{self.srv.server_address[1]}',
                        AB_TOOLS_FILE=str(tools))

    def tearDown(self):
        self.srv.shutdown()

    def ab(self, *args, stdin=None, env=None):
        return subprocess.run([sys.executable, str(HERE / 'pi' / 'image' / 'ab'), *args], input=stdin, text=True,
                              capture_output=True, env=env or self.env, timeout=60)

    def test_calls(self):
        r = self.ab('api_search', '{"query": "google sheets values update", "top_k": 1}')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(json.loads(r.stdout)['results']), 1)
        r = self.ab('api_fetch', '-', stdin=json.dumps(UPDATE))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.world.n, 2)
        self.assertIn('ss_capex:ws_pending_capex:3', self.state('hidden.json'))

    def test_errors(self):
        self.assertEqual(self.ab('api_fetch', 'not json').returncode, 2)
        self.assertEqual(self.ab('api_fetch', '{}', 'extra').returncode, 2)
        r = self.ab('nope', '{}')                      # checked locally against the tool list
        self.assertEqual(r.returncode, 2)
        self.assertIn('api_search, api_fetch, base64_encode', r.stderr)
        self.assertEqual(self.world.n, 0)               # nothing ran: bad JSON never left the CLI; no such tool
        self.assertIn('api_search', self.ab('tools').stdout)
        self.assertIn('ab <tool>', self.ab('--help').stdout)
        dead = dict(self.env, AB_WORLD_URL='http://127.0.0.1:9')
        self.assertEqual(self.ab('api_search', '{"query": "x"}', env=dead).returncode, 3)


class Bundle(unittest.TestCase):
    def test_instruction(self):
        tools = PT.tool_schemas()
        self.assertEqual([t['function']['name'] for t in tools], ['api_search', 'api_fetch', 'base64_encode'])
        text = PT.instruction(row(), tools)
        user = [m['content'] for m in row()['prompt'] if m['role'] == 'user'][0]
        system = [m['content'] for m in row()['prompt'] if m['role'] == 'system'][0]
        self.assertTrue(text.endswith('## Request\n\n' + user))
        self.assertTrue(text.startswith(system.split(PT.BUDGET)[0]))
        self.assertNotIn('50 tool-using turns', text)
        self.assertIn("ab <tool> '<JSON arguments>'", text)
        self.assertIn('### ab api_fetch', text)
        self.assertIn('the local filesystem has nothing', text)

    def test_build_task(self):
        d = Path(tempfile.mkdtemp())
        PT.build_task(row(), PT.tool_schemas(), d, oracle=True)
        t = d / TASK
        self.assertEqual(json.loads((t / 'environment/world/task.json').read_text()), PT.world_spec(row()))
        self.assertEqual((t / 'environment/world/world_server.py').read_bytes(),
                         (HERE / 'pi/image/world_server.py').read_bytes())
        self.assertIn('/task/world_server.py', (t / 'environment/docker-compose.yaml').read_text())
        toml = (t / 'task.toml').read_text()
        self.assertIn('artifacts = [{ source = "/state", service = "world" }]', toml)
        self.assertIn('allow_internet = false', toml)
        self.assertTrue(os.access(t / 'inputs/usr/local/bin/ab', os.X_OK))
        self.assertTrue((t / 'solution/solve.sh').exists())


class RunArgs(unittest.TestCase):
    def test_task_ids(self):
        self.assertEqual(PR.task_ids('a.b, c.d'), ['a.b', 'c.d'])
        f = Path(tempfile.mkdtemp()) / 'ids.txt'
        f.write_text('# plan\nsales.x\nhr.y,  finance.z\n\n')
        self.assertEqual(PR.task_ids('@' + str(f)), ['sales.x', 'hr.y', 'finance.z'])
        self.assertEqual(PR.task_ids(str(f)), ['sales.x', 'hr.y', 'finance.z'])
        with self.assertRaises(SystemExit):
            PR.task_ids('a.b,a.b')

    def test_dispatch_suite(self):
        self.assertEqual(PR.pi_eval_args(['--max-in-flight', '8']), ['--max-in-flight', '8'])
        self.assertEqual(PR.pi_eval_args(['--dispatch', 'D']), ['--dispatch', 'D', '--dispatch-suite', 'automationbench'])
        self.assertEqual(PR.pi_eval_args(['--dispatch=D', '--dispatch-suite', 'x']), ['--dispatch=D', '--dispatch-suite', 'x'])


class Collect(unittest.TestCase):
    def test_world_failing_validation_is_ungradeable_not_a_crash(self):
        # #375, Qwen3.6: the world service took mailchimp.subscribers[].notes[] = {'text': ...} (a list[str] field)
        from unittest import mock
        import pydantic
        from aa_evals.common.pi_eval import port
        tmp = Path(tempfile.mkdtemp())

        def extract(pi_run, r, dest, artifacts):
            (dest / 'artifacts' / 'state').mkdir(parents=True)
            (dest / 'artifacts' / 'state' / 'world.json').write_text('{}')
            return {'session': [], 'wall_sec': 1, 'archive': 'a.tar'}

        class M(pydantic.BaseModel):
            notes: list[str]
        try:
            M(notes=[{'text': 'x'}])
        except pydantic.ValidationError as e:
            err = e

        def bad_grade(*a, **k):
            raise err
        r = {'item_id': TASK, 'repeat': 0, 'key': 'k0', 'status': 'ok'}
        with mock.patch.object(port, 'extract', extract), mock.patch.object(PR, 'grade', bad_grade), \
                mock.patch.object(port, 'end_of', lambda row: 'finish'), mock.patch.object(port, 'stats', lambda row: {}):
            rec, tr = PR.import_row({'harness': {'pi_run': str(tmp)}}, row(), r, tmp / 'scratch')
        self.assertEqual((rec['status'], rec['score'], rec['ungradeable']), ('infra_error', None, True))
        self.assertIn('WorldState validation', rec['error'])
        self.assertEqual(tr['status'], 'infra_error')


if __name__ == '__main__':
    unittest.main()

"""#531: synthesis runs the hung-session watchdog pi_eval runs (tb_agents/hung.py): new campaigns get
harbor.hung_watchdog by default, and the worker stops a hung attempt as an infrastructure failure."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from synth.config import build, hung_watchdog
from tb_agents import hung

HARBOR = """#!/bin/bash
# fake gen/run_harbor.sh: the attempt's Pi proxy log dir with its address, a session log, the guard's log with an
# accounting failure (the 11-15 hang cause, when a campaign runs the guard), then a wait the watchdog has to cut
raw="$2"
mkdir -p "$raw/.pi-cont16k/x" "$raw/job/trial/agent/pi/sessions"
echo '{"proxy_api": "http://127.0.0.1:9/v1"}' > "$raw/.pi-cont16k/x/protocol.json"
echo '{"type": "message"}' > "$raw/job/trial/agent/pi/sessions/s.jsonl"
if [ -n "$ACCOUNTING" ]; then echo '{"time": 1, "accounting_error": "TimeoutError: ", "upstream": {}}' > "$raw/.pi-cont16k/x/response-budget-wire.jsonl"; fi
exec sleep 40
"""


def campaign_toml(d, harbor=''):
    p = Path(d) / 'c.toml'
    p.write_text(f'id = "x"\nteacher = "nemotron35-super"\nroot = "{d}"\n{harbor}[train]\nmode = "per_task"\n'
                 f'trajectories_per_task = 1\n[train.stream]\ndir = "{Path(d) / "stream"}"\n')
    return p


class CampaignDefault(unittest.TestCase):
    def test_new_campaigns_get_it_unless_turned_off(self):
        with tempfile.TemporaryDirectory() as d:
            on = build(campaign_toml(d))['harbor']['hung_watchdog']
            self.assertEqual(on['idle_sec'], hung.resolve(None)['idle_sec'])
            self.assertIsNone(on['stall_sec'])   # synthesis writes no wire-responses/: the stall rule has no evidence
            self.assertIsNone(build(campaign_toml(d, '[harbor]\nhung_watchdog = false\n'))['harbor']['hung_watchdog'])
            own = build(campaign_toml(d, '[harbor]\nhung_watchdog = {idle_sec = 900, stall_sec = 3660}\n'))
            self.assertEqual((own['harbor']['hung_watchdog']['idle_sec'], own['harbor']['hung_watchdog']['stall_sec']),
                             (900.0, 3660.0))
        for bad in ({'idle_sec': 0}, {'idle_sec': 60}, {'idle_sec': 900, 'stall_sec': -1}, {'idle_sec': 900, 'stal_sec': 9},
                    'yes'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'hung_watchdog'):
                hung_watchdog(bad)


class AgentEnd(unittest.TestCase):
    def test_trajectory_json_disarms_the_watchdog(self):
        """Synthesis' agent writes no env_clock.json; Harbor's trajectory.json (written before the verifier) is the
        agent's end, so a long verifier is never judged idle."""
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            (raw / 'job' / 'trial' / 'agent' / 'pi' / 'sessions').mkdir(parents=True)
            (raw / 'job' / 'trial' / 'agent' / 'pi' / 'sessions' / 's.jsonl').write_text('{"type": "message"}\n')
            (raw / '.pi-cont16k' / 'x').mkdir(parents=True)
            (raw / '.pi-cont16k' / 'x' / 'protocol.json').write_text('{"proxy_api": "http://127.0.0.1:9/v1"}')
            clock = {'in_flight': 0, 'requests': 5, 'model_sec': 9.0}
            t = [1000.0]
            wd = hung.Watchdog(600, None, read=lambda api: clock, clock=lambda: t[0])
            self.assertIsNone(wd.tick(raw))          # armed: requests made, nothing in flight
            t[0] += 700
            self.assertEqual(wd.tick(raw)[0], 'idle')
            wd2 = hung.Watchdog(600, None, read=lambda api: clock, clock=lambda: t[0])
            self.assertIsNone(wd2.tick(raw))
            (raw / 'job' / 'trial' / 'agent' / 'trajectory.json').write_text('{}')   # the agent stopped
            t[0] += 700
            self.assertTrue(hung.agent_ended(raw))
            self.assertIsNone(wd2.tick(raw))         # the verifier may run as long as it needs


class WorkerStopsAHungAttempt(unittest.TestCase):
    def attempt(self, accounting=False, watchdog=None):
        from synth import worker as W
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            d = Path(d)
            (d / 'repo' / 'gen').mkdir(parents=True)
            (d / 'repo' / 'gen' / 'run_harbor.sh').write_text(HARBOR)
            config = build(campaign_toml(d))
            (d / 'campaign.json').write_text(json.dumps(config))
            scratch = d / 'scratch'
            scratch.mkdir()
            stack.enter_context(mock.patch.dict(os.environ, {'SYNTH_CAMPAIGN': str(d / 'campaign.json'),
                                                             '_CONDOR_SCRATCH_DIR': str(scratch),
                                                             'ACCOUNTING': '1' if accounting else ''}))
            stack.enter_context(mock.patch.object(W, 'REPO', d / 'repo'))
            stack.enter_context(mock.patch.object(W, 'HUNG_TICK_SEC', 1.0))
            if watchdog is not None:
                stack.enter_context(mock.patch.object(W.hung, 'Watchdog', watchdog))
            w = W.Worker(0, 1)
            item = {'id': 'u1', 'kind': 'train', 'task': 't1', 'endpoint': 'http://127.0.0.1:9/v1', 'backend': 'b',
                    'seed': 1}
            w.active = {'u1': {}}
            t0 = time.time()
            asyncio.run(w.run(item))
            wall = time.time() - t0
            res = json.loads((w.campaign.root_for('train') / 'raw' / 'u1' / 'worker-result.json').read_text())
        return res, wall, config

    def test_guard_accounting_failure_ends_the_attempt_as_infrastructure(self):
        res, wall, _ = self.attempt(accounting=True)
        self.assertEqual(res['infrastructure_failure'], 'ResponseBudgetAccounting')
        self.assertEqual((res['hung_kind'], res['termination_reason']), ('budget_accounting_failure', 'hung_session'))
        self.assertLess(wall, 35)   # cut by the watchdog, not the 40 s session

    def test_idle_session_ends_as_hung_session(self):
        """The rule synthesis can reach: Pi idle past idle_sec (stubbed watchdog: its timing is tested in pi_eval)."""
        class Idle:
            def __init__(self, idle_sec, stall_sec):
                self.idle_sec = idle_sec

            def tick(self, raw, ok=True):
                return 'idle', self.idle_sec + 100
        res, wall, config = self.attempt(watchdog=Idle)
        limit = config['harbor']['hung_watchdog']['idle_sec']
        self.assertEqual(res['infrastructure_failure'], 'HungSession')
        self.assertEqual((res['hung_kind'], res['hung_limit_sec'], res['hung_sec']), ('idle', limit, limit + 100))
        self.assertLess(wall, 35)


if __name__ == '__main__':
    unittest.main()

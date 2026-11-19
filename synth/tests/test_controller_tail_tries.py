"""Training must not close on tries that are still running (uni-canary 11-18, #451 workstream 2)."""
import tempfile
import unittest

from synth import controller as controller_module
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher

TASKS = [f't{i:02d}' for i in range(10)]


def item(task, status='active', **extra):
    return {'kind': 'train', 'task': task, 'status': status, 'id': f'x-train-{task}-a0-r0', **extra}


class LastTryStillRunning(unittest.TestCase):
    def controller(self, d, extra=0):
        config = {'id': 'tail', 'root': d, 'teacher': load_teacher('nemotron35-super'),
                  'protocol': dict(PROTOCOL_DEFAULTS), 'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0},
                  'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'max_extra_attempts_per_task': extra,
                            'task_list': list(TASKS)},
                  'eval': {'enabled': False}, 'smoke': {'attempts': []},
                  'serving': {'jobs': 1, 'gpus_per_job': 1}, 'workers': {'count': 1, 'slots': 12}}
        campaign = Campaign(config)
        campaign.ops.mkdir(parents=True)
        c = controller_module.Controller(campaign)
        c.s['stage'] = 'running'
        return c

    def test_first_attempts_running_with_no_extra_tries_is_not_the_target(self):
        # the canary: max_extra_attempts_per_task = 0, all 10 first attempts live -> "Training target reached on live
        # counts", every attempt interrupted 14 s in (TrainingTargetReached counts as no try), then re-dispatched
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.s['items'] = {f'{t}-a0': item(t) for t in TASKS}
            self.assertTrue(all(c.tried()[t] >= c.attempt_caps()[t] for t in TASKS), 'dispatch: no further tries')
            self.assertFalse(c.train_target_reached())

    def test_finished_tries_at_the_cap_close_training(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.s['items'] = {f'{t}-a0': item(t, 'finished') for t in TASKS}   # finished, not (yet) exported
            self.assertTrue(c.train_target_reached())

    def test_interrupted_and_infrastructure_tries_keep_their_rules(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.s['items'] = {f'{t}-a0': item(t, 'finished') for t in TASKS[1:]}
            c.s['items']['t00-a0'] = item('t00', 'finished', infrastructure_failure='TrainingTargetReached',
                                          controller_interruption=True)
            self.assertFalse(c.train_target_reached(), "the controller's interruption is no try")
            c.s['items']['t00-a1'] = item('t00', 'active')
            self.assertFalse(c.train_target_reached(), 'its replacement is running')
            c.s['items']['t00-a1']['status'] = 'finished'
            self.assertTrue(c.train_target_reached())

    def test_infrastructure_failures_count_toward_their_own_cap_when_finished(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.s['items'] = {f'{t}-a0': item(t, 'finished') for t in TASKS[1:]}
            cap = c.infrastructure_cap()
            for k in range(cap):   # pre-agent failures: no try, but the task stops after `cap` of them
                c.s['items'][f't00-i{k}'] = item('t00', 'finished', infrastructure_failure='PreAgentFailure/BuildError')
            self.assertTrue(c.train_target_reached(), 'an infra-capped task is done')
            c.s['items'][f't00-i{cap - 1}']['status'] = 'active'   # its last one still running: not yet
            self.assertFalse(c.train_target_reached())


if __name__ == '__main__':
    unittest.main()

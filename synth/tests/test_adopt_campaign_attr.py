"""An adopted serving job stays in the adopter's queue view (uni-canary 11-18, #451 workstream 2)."""
import tempfile
import time
import unittest
from unittest import mock

from synth import condor, controller as controller_module
from synth.common import atomic
from synth.config import ADMISSION_DEFAULTS, PROTOCOL_DEFAULTS, Campaign, load_teacher


class Schedd:
    """condor_q as the schedd answers it: a job matches by cluster id, or by the +SynthCampaign it was submitted with."""
    def __init__(self):
        self.jobs = {}   # job_id -> SynthCampaign attribute (or None)

    def query(self, job_ids, attr=None):
        clusters = {j.split('.')[0] for j in job_ids}
        return {j: {'ClusterId': int(j.split('.')[0]), 'ProcId': 0, 'JobStatus': 2} for j, a in self.jobs.items()
                if j.split('.')[0] in clusters or (attr is not None and a == attr)}


class AdoptedJobInQueueView(unittest.TestCase):
    def test_source_job_submitted_with_its_campaign_attribute(self):
        with tempfile.TemporaryDirectory() as d:
            base = {'teacher': load_teacher('nemotron35-super'), 'protocol': dict(PROTOCOL_DEFAULTS),
                    'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0}, 'eval': {'enabled': False},
                    'smoke': {'attempts': []}, 'workers': {'count': 1, 'slots': 4}}
            source = Campaign({**base, 'id': 'old', 'root': d, 'serving': {'jobs': 1, 'gpus_per_job': 4},
                               'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['x']}})
            source.ops.mkdir(parents=True)
            source.freeze()
            atomic(source.ops / 'serving' / 'RETAIN.json', {'reason': 'test'})
            # since #235 condor.submit receipts carry campaign_attr: the job has +SynthCampaign = "old"
            atomic(source.ops / 'jobs' / 'serving-0-g0-submission.json',
                   {'job_id': '900.0', 'role': 'serving', 'index': 0, 'generation': 0, 'time': time.time() - 600,
                    'campaign_attr': True})
            campaign = Campaign({**base, 'id': 'new', 'root': d,
                                 'serving': {'jobs': 1, 'gpus_per_job': 4, 'adopt_ops': str(source.ops)},
                                 'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']}})
            campaign.ops.mkdir(parents=True)
            # the adopter's own workers were submitted with +SynthCampaign = "new"
            atomic(campaign.ops / 'jobs' / 'worker-0-g0-submission.json',
                   {'job_id': '950.0', 'role': 'worker', 'index': 0, 'generation': 0, 'time': time.time() - 600,
                    'campaign_attr': True})
            schedd = Schedd()
            schedd.jobs.update({'900.0': 'old', '950.0': 'new'})
            condor._RECEIPTS.clear()
            condor._LISTINGS.clear()
            with mock.patch.object(condor, 'query', schedd.query):
                c = controller_module.Controller(campaign)
                c.adopt_serving(condor.queue(campaign))
                adopted = condor.latest(campaign, 'serving', 0)
                self.assertEqual(adopted['job_id'], '900.0')
                self.assertFalse(adopted.get('campaign_attr'), 'the adopted job carries the source campaign attribute')
                condor._LISTINGS.clear()
                view = condor.queue(campaign)
                self.assertIn('900.0', view, 'adopted serving must stay in the queue view (else: lost, resubmitted)')
                self.assertIn('950.0', view)


    def test_a_receipt_adopted_before_the_fix_is_readopted_clean(self):
        # a controller restarted on a campaign whose adopted receipt carries campaign_attr (written by the old code)
        with tempfile.TemporaryDirectory() as d:
            base = {'teacher': load_teacher('nemotron35-super'), 'protocol': dict(PROTOCOL_DEFAULTS),
                    'admission': {**ADMISSION_DEFAULTS, 'period_seconds': 0}, 'eval': {'enabled': False},
                    'smoke': {'attempts': []}, 'workers': {'count': 1, 'slots': 4}}
            source = Campaign({**base, 'id': 'old', 'root': d, 'serving': {'jobs': 1, 'gpus_per_job': 4},
                               'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['x']}})
            source.ops.mkdir(parents=True)
            source.freeze()
            atomic(source.ops / 'serving' / 'RETAIN.json', {'reason': 'test'})
            atomic(source.ops / 'jobs' / 'serving-0-g0-submission.json',
                   {'job_id': '900.0', 'role': 'serving', 'index': 0, 'generation': 0, 'time': time.time() - 600,
                    'campaign_attr': True})
            campaign = Campaign({**base, 'id': 'new', 'root': d,
                                 'serving': {'jobs': 1, 'gpus_per_job': 4, 'adopt_ops': str(source.ops)},
                                 'train': {'mode': 'per_task', 'trajectories_per_task': 1, 'task_list': ['a']}})
            campaign.ops.mkdir(parents=True)
            atomic(campaign.ops / 'jobs' / 'serving-0-g0-submission.json',   # the old code's adopted receipt
                   {'job_id': '900.0', 'role': 'serving', 'index': 0, 'generation': 0, 'time': time.time() - 600,
                    'campaign_attr': True, 'serving_ops': str(source.ops), 'adopted_at': time.time() - 600})
            schedd = Schedd()
            schedd.jobs.update({'900.0': 'old'})
            condor._RECEIPTS.clear()
            condor._LISTINGS.clear()
            with mock.patch.object(condor, 'query', schedd.query):
                self.assertNotIn('900.0', condor.queue(campaign), 'the stale receipt hides the job (the bug)')
                c = controller_module.Controller(campaign)
                c.adopt_serving(condor.queue(campaign))
                adopted = condor.latest(campaign, 'serving', 0)
                self.assertEqual((adopted['job_id'], adopted['generation']), ('900.0', 1))
                self.assertFalse(adopted.get('campaign_attr'))
                condor._LISTINGS.clear()
                self.assertIn('900.0', condor.queue(campaign))
                c.adopt_serving(condor.queue(campaign))   # visible now: no further re-adoption
                self.assertEqual(condor.latest(campaign, 'serving', 0)['generation'], 1)


if __name__ == '__main__':
    unittest.main()

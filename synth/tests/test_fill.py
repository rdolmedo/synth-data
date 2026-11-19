"""tbshared 1.5 fill in synth's submitters: GPU roles within the B200 class, CPU roles at their fixed bid, workers
once per tick for the batch (condor.worker_fill), with the decision in ops/jobs/fill.jsonl."""
import json
import tempfile
from unittest import mock

from synth import condor, controller as controller_module
from synth.common import read
from synth.tests.test_controller_control import campaign_in
from tbshared import condor_cost

GB = 1024
# condor_status_slots lines: name floor Cpus Memory(MB) Gpus TotalGpus TotalCpus TotalMemory CUDACapability
NODES = '\n'.join([
    f'b01\t55\t384\t{3022 * GB}\t8\t8\t384\t{3022 * GB}\t10.0',          # B200: 48 CPUs, 377 GB per GPU
    f'c01\t15\t128\t{1007 * GB}\t0\t0\t128\t{1007 * GB}\tundefined',     # cheap CPU node: 7.9 GB per CPU free
    f'c02\t35\t192\t{3023 * GB}\t0\t0\t192\t{3023 * GB}\tundefined',
])


def live(monkeypatch, text=NODES):
    monkeypatch.setattr(condor_cost, 'condor_status_slots', lambda *a, **k: text)


def submitted(monkeypatch):
    run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.'))
    monkeypatch.setattr(condor.subprocess, 'run', run)
    return run


def test_controller_and_serving_are_filled_and_recorded(monkeypatch):
    live(monkeypatch)
    submitted(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        campaign = campaign_in(d)
        condor.submit(campaign, 'controller', cpus=8, memory_gib=48, disk_gib=64, bid=56)
        sub = (campaign.ops / 'jobs' / 'controller-0-g0.sub').read_text()
        assert 'request_cpus = 8\n' in sub and 'request_memory = 131072\n' in sub              # 8c/48G -> 8c/128G
        assert read(campaign.ops / 'jobs' / 'controller-0-g0-submission.json')['memory_gib'] == 128
        # DeepSeek-style 4xB200 at 32c/768G: 48 CPUs (under 4 x 48), RAM as asked
        condor.submit(campaign, 'serving', cpus=32, memory_gib=768, disk_gib=1, bid=63, gpus=4)
        sub = (campaign.ops / 'jobs' / 'serving-0-g0.sub').read_text()
        assert 'request_cpus = 48\n' in sub and 'request_memory = 786432\n' in sub and 'request_gpus = 4' in sub
        recs = [json.loads(x) for x in (campaign.ops / 'jobs' / 'fill.jsonl').read_text().splitlines()]
        assert [r['decision'] for r in recs] == ['filled', 'filled'] and recs[1]['gpus'] == 4.0


def test_worker_role_is_not_filled_at_submit(monkeypatch):
    live(monkeypatch)
    submitted(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        campaign = campaign_in(d)
        condor.submit(campaign, 'worker', cpus=12, memory_gib=96, disk_gib=1, bid=25, fill=False)
        assert 'request_memory = 98304\n' in (campaign.ops / 'jobs' / 'worker-0-g0.sub').read_text()


def test_worker_fill_once_for_the_batch(monkeypatch):
    live(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        campaign = campaign_in(d, workers={'cpus': 12, 'memory_gib': 96, 'bid': 40, 'cheap_bid': 25})
        # 4 workers of 12c/192G still fit the cheap node (128 CPUs, 1007 GB at floor 15): filled
        assert condor.worker_fill(campaign, 4) == {'cpus': 12, 'memory_gib': 192.0}
        # 10: 10 x 96 G fit, 10 x 192 G do not; capped at the cheap node's 7.9 GB per CPU = 94 G < 96: as configured
        assert condor.worker_fill(campaign, 10) == {}
        recs = [json.loads(x) for x in (campaign.ops / 'jobs' / 'fill.jsonl').read_text().splitlines()]
        assert [r['decision'] for r in recs] == ['filled', 'unfilled'] and recs[0]['n'] == 4
        campaign.config['workers']['fill'] = False
        assert condor.worker_fill(campaign, 4) == {}


def test_worker_fill_without_condor_status_keeps_the_configured_shape():
    with tempfile.TemporaryDirectory() as d:                     # conftest: condor_status unavailable
        campaign = campaign_in(d, workers={'cpus': 12, 'memory_gib': 96, 'bid': 40, 'cheap_bid': 25})
        assert condor.worker_fill(campaign, 4) == {}


def test_ensure_workers_uses_the_ticks_filled_shape_and_keeps_overrides(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        c = object.__new__(controller_module.Controller)
        c.c = mock.Mock(config={'workers': {'bid': 40, 'cheap_bid': 25, 'cpus': 12, 'memory_gib': 96}}, id='t')
        c.ops = condor.Path(d)
        c.s = {'items': {}, 'retired_workers': []}
        c._heartbeats = {}
        (c.ops / 'worker-resource-overrides.json').write_text(json.dumps({'2': {'cpus': 32}}))
        calls, rooms = [], []
        monkeypatch.setattr(condor, 'latest', lambda campaign, role, i: None)
        monkeypatch.setattr(condor, 'receipts', lambda campaign: [])
        monkeypatch.setattr(condor, 'worker_fill', lambda campaign, n, exclude=(): calls.append(n) or
                            {'cpus': 12, 'memory_gib': 192.0})
        monkeypatch.setattr(condor, 'cheap_capacity', lambda cpus, mem, **k: rooms.append((cpus, mem)) or 2)
        shapes = []
        monkeypatch.setattr(condor, 'submit_worker', lambda campaign, index, gen, shape, cheap=True, avoid=():
                            shapes.append((index, shape, cheap)))
        c.ensure_workers({}, 3)
    assert calls == [3] and rooms == [(12, 192.0)]                         # one decision, for the 3 workers
    assert shapes == [(0, {'cpus': 12, 'memory_gib': 192.0}, True), (1, {'cpus': 12, 'memory_gib': 192.0}, True),
                      (2, {'cpus': 32}, False)]                            # an operator override is kept as given


def test_a_failed_fill_decision_keeps_the_configured_shape(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        c = object.__new__(controller_module.Controller)
        c.c = mock.Mock(config={'workers': {'bid': 40, 'cpus': 12, 'memory_gib': 96}}, id='t')
        c.ops = condor.Path(d)
        c.s = {'items': {}, 'retired_workers': []}
        c._heartbeats = {}
        monkeypatch.setattr(condor, 'latest', lambda campaign, role, i: None)
        monkeypatch.setattr(condor, 'receipts', lambda campaign: [])

        def boom(*a, **k):
            raise ValueError('bad config')
        monkeypatch.setattr(condor, 'worker_fill', boom)
        shapes = []
        monkeypatch.setattr(condor, 'submit_worker', lambda campaign, index, gen, shape, avoid=(): shapes.append(shape))
        c.ensure_workers({}, 2)
    assert shapes == [{}, {}]

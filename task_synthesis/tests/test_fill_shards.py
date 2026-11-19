"""tbshared 1.5 fill in task-synthesis's own submitters: round shards decided once for all jobs on their ladder (the
cheap split then counts the filled shape, and the shards get the memory net with AUTO_RELEASE); generator servers
within the B200 class. condor_status is faked here (conftest makes the live one unavailable)."""
import json
from argparse import Namespace
from pathlib import Path
from unittest import mock

from task_synthesis.pipeline import round as R
from task_synthesis.serve import serve_teacher as S
from tbshared import condor_cost

GB = 1024
NODES = '\n'.join([   # name floor Cpus Memory(MB) Gpus TotalGpus TotalCpus TotalMemory CUDACapability
    f'b01\t55\t384\t{3022 * GB}\t8\t8\t384\t{3022 * GB}\t10.0',
    f'c01\t15\t128\t{1007 * GB}\t0\t0\t128\t{1007 * GB}\tundefined',
    f'c02\t35\t192\t{3023 * GB}\t0\t0\t192\t{3023 * GB}\tundefined',
])
BODY = 'request_cpus = {c}\nrequest_memory = {m}\n+JobBatchName = "tasksynth-round-x"\n'


def server_args(**kw):
    """serve_teacher submit's arguments as production.submit_server passes them (--pipeline task-synthesis, #480)."""
    return Namespace(pipeline='task-synthesis', fill=None, tunables='', protocol='', checkpoint='', dry_run=False,
                     bypass_lxcfs=False, exclude_nodes='', cpus=0, memory_gib=0, cuda_cap='', **kw)


def live(monkeypatch, text=NODES):
    monkeypatch.setattr(condor_cost, 'condor_status_slots', lambda *a, **k: text)


def test_fill_shards_for_the_batch_on_its_ladder(monkeypatch, tmp_path):
    live(monkeypatch)
    sub = tmp_path / 'job.sub'
    a = Namespace(cpus=32, memory_gib=96, bid=56, cheap_bid=25)
    body, shape = R.fill_shards(BODY.format(c=32, m=96 * 1024), 1, a, sub)
    assert shape == (32, 512.0) and 'request_memory = 524288\n' in body          # 1 shard fits the cheap node
    body, shape = R.fill_shards(BODY.format(c=32, m=96 * 1024), 4, a, sub)
    assert shape == (32, 251.0)         # 4 x 512 G do not fit at 25; capped at the cheap node's 7.9 GB/CPU, 4 do
    a.fill = False
    assert R.fill_shards(BODY.format(c=32, m=96 * 1024), 1, a, sub)[1] == (32, 96.0)
    recs = [json.loads(x)['decision'] for x in (tmp_path / 'fill.jsonl').read_text().splitlines()]
    assert recs == ['filled', 'capped', 'off']


def test_production_shards_are_already_at_the_binding_level(monkeypatch, tmp_path):
    live(monkeypatch)
    a = Namespace(cpus=24, memory_gib=384, bid=56, cheap_bid=25)      # task_synthesis/production/m1.toml [round]
    body, shape = R.fill_shards(BODY.format(c=24, m=384 * 1024), 12, a, tmp_path / 'job.sub')
    assert shape == (24, 384.0)


def test_cheap_split_counts_the_submitted_shape(monkeypatch):
    seen = []
    import synth.condor
    monkeypatch.setattr(synth.condor, 'cheap_capacity', lambda c, m, **k: seen.append((c, m)) or 3)
    a = Namespace(cpus=32, memory_gib=96, bid=56, cheap_bid=25)
    assert R.cheap_split(5, a, 32, 512.0) == 3 and seen == [(32, 512.0)]
    assert R.cheap_split(5, a) == 3 and seen[-1] == (32, 96)


def test_round_submit_writes_the_filled_shape_with_the_memory_net(monkeypatch, tmp_path):
    live(monkeypatch)
    monkeypatch.setattr(R, 'ROUNDS', tmp_path)
    monkeypatch.setattr(R, 'frozen_checkout', lambda: tmp_path / 'code')
    monkeypatch.setattr(R, 'cheap_split', lambda n, a, *shape: 0)
    run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 77.\n'))
    monkeypatch.setattr(R.subprocess, 'run', run)
    a = Namespace(round='r1', cpus=32, memory_gib=96, bid=56, cheap_bid=0, jobs=1, disk_gib=200)
    R.submit(a)
    text = (tmp_path / 'r1' / 'job.sub').read_text()
    assert f'request_memory = 524288 + 131072 * {condor_cost.MEMORY_HOLDS}\n' in text and 'request_cpus = 32\n' in text
    assert condor_cost.AUTO_RELEASE in text and text.endswith('queue 1\n')
    assert json.loads((tmp_path / 'r1' / 'job.json').read_text())['shape'] == {'cpus': 32, 'memory_gib': 512.0}
    assert '--fill' not in text                                                    # not forwarded to the job


def test_generator_server_fill_within_the_b200_class(monkeypatch, tmp_path):
    live(monkeypatch)
    monkeypatch.setattr(S, 'ROOT', tmp_path)
    monkeypatch.setattr(S, 'frozen_checkout', lambda: tmp_path / 'code')
    run = mock.Mock(return_value=mock.Mock(stdout='1 job(s) submitted to cluster 88.\n'))
    monkeypatch.setattr(S.subprocess, 'run', run)
    S.submit(server_args(teacher='deepseek-v41-flash-1017', name='ds1', gpus=4, bid=62, hours=24, force=False))
    job = json.loads((tmp_path / 'ds1' / 'job.json').read_text())
    assert (job['cpus'], job['memory_gib']) == (20, 320.0)          # the profile's 20c/320G: already at 16 GB/CPU
    S.submit(server_args(teacher='glm53-flash', name='glm', gpus=4, bid=63, hours=24, force=False))
    job = json.loads((tmp_path / 'glm' / 'job.json').read_text())
    assert (job['cpus'], job['memory_gib']) == (96, 1536.0)         # 32c/1536G -> 96c (cap 4 x 48 = 192)
    assert 'request_cpus = 96\n' in (tmp_path / 'glm' / 'job.sub').read_text()

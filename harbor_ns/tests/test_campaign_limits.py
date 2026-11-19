"""Synth campaigns' sandbox limits (synth/sandbox_limits.py, #378 fix 4): the PSS guard's budget and spared agent, and
the disk cap's per-file limit on every process of a box.

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q harbor_ns/tests
"""
import asyncio
from pathlib import Path

import pytest

from harbor_ns import sandbox
from harbor_ns.environment import campaign_memory_limit_mb, campaign_memory_spare
from harbor_ns.sandbox import RuntimeConfig, Sandbox, memory_victim


@pytest.mark.parametrize('memory_mb,env,limit', [
    (2048, {}, 2048),                                   # campaigns frozen before 2026-11-14: 1.0 x memory_mb
    (None, {}, 4096),                                   # ... and 4 GiB without memory_mb
    (2048, {'PI_CAMPAIGN_MEMORY_FACTOR': '1.5', 'PI_CAMPAIGN_MEMORY_MIN_MB': '24576'}, 24576),
    (32768, {'PI_CAMPAIGN_MEMORY_FACTOR': '1.5', 'PI_CAMPAIGN_MEMORY_MIN_MB': '24576'}, 49152),
    (None, {'PI_CAMPAIGN_MEMORY_FACTOR': '1.5', 'PI_CAMPAIGN_MEMORY_MIN_MB': '24576'}, 24576),
    (4096, {'PI_CAMPAIGN_MEMORY_FACTOR': '2'}, 8192),
])
def test_campaign_memory_limit(memory_mb, env, limit):
    assert campaign_memory_limit_mb(memory_mb, env) == limit


def test_campaign_memory_spare():
    assert campaign_memory_spare({}) == ()
    assert campaign_memory_spare({'PI_CAMPAIGN_MEMORY_SPARE': 'pi,node'}) == ('pi', 'node')
    with pytest.raises(ValueError):
        campaign_memory_limit_mb(1024, {'PI_CAMPAIGN_MEMORY_FACTOR': '-1'})


def test_memory_victim_spares_the_agent_unless_alone():
    rows = [{'pid': 2, 'pss_kib': 900, 'comm': 'pi'}, {'pid': 3, 'pss_kib': 500, 'comm': 'python3'}]
    assert memory_victim(rows)['pid'] == 2                       # legacy: the largest, whatever it is
    assert memory_victim(rows, ('pi',))['pid'] == 3              # the agent is spared ...
    assert memory_victim(rows[:1], ('pi',))['pid'] == 2          # ... unless nothing else is left


def test_guard_kills_the_largest_non_agent_inside_the_box(tmp_path):
    box = Sandbox(cfg=RuntimeConfig(nsinit=Path('/x'), work_root=tmp_path), lower=[], init_pid=123,
                  memory_limit_mb=1, memory_spare=('pi',), memory_poll_s=0, memory_event_log=tmp_path / 'events.jsonl')
    box._dir = tmp_path
    rows = [{'pid': 1, 'pss_kib': 10, 'comm': 'nsinit'}, {'pid': 7, 'pss_kib': 4000, 'comm': 'pi'},
            {'pid': 9, 'pss_kib': 1000, 'comm': 'python3'}]
    killed = []
    box._memory_snapshot = lambda: rows

    async def exec_(cmd, **kw):
        killed.append(cmd)
        box.init_pid = None                                       # one pass
    box.exec = exec_
    asyncio.run(box._watch_memory())
    assert killed == ['kill -KILL 9'] and box.memory_kills[0]['comm'] == 'python3'
    assert 'spare pi' in (tmp_path / 'events.jsonl').read_text()


def test_disk_cap_limits_box_launch_and_execs():
    cfg = RuntimeConfig(nsinit=Path('/x'), work_root=Path('/tmp'))
    box = Sandbox(cfg=cfg, lower=[], init_pid=123, fsize=30720 << 20)
    if not sandbox.fsize_supported():
        pytest.skip('prlimit not installed')
    argv = box._nsenter()
    assert argv[0].endswith('prlimit') and argv[1] == f'--fsize={30720 << 20}:{30720 << 20}'
    assert 'prlimit' not in ' '.join(Sandbox(cfg=cfg, lower=[], init_pid=123)._nsenter())

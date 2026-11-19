"""#375 finish notice: default for new Pi runs, `off`, non-Pi agents, and resume of runs that predate it."""
import argparse

import pytest

from aa_evals.common.pi_eval.run import DEFAULT_FINISH_NOTICE, finish_notice, finish_notice_conflict


def ns(**kw):
    return argparse.Namespace(**{'agent': 'pi', 'finish_notice': None, **kw})


def test_default_is_on_for_new_pi_runs():
    fn = finish_notice(ns())
    assert DEFAULT_FINISH_NOTICE == '40000:16384'
    assert (fn['warn_at'], fn['reserve']) == (40000, 16384)
    assert fn['policy'] == 'submit-at-context-limit-v4-sticky-notice' and fn['sticky'] is True   # #375 sftnotice
    assert fn['final_at'] == 16384 + 8192


@pytest.mark.parametrize('off', ['off', 'OFF', 'none', '0', '0:0'])
def test_off(off):
    assert finish_notice(ns(finish_notice=off)) is None


def test_explicit_values_and_validation():
    assert finish_notice(ns(finish_notice='30000:8192'))['reserve'] == 8192
    with pytest.raises(SystemExit):
        finish_notice(ns(finish_notice='30000'))
    with pytest.raises(SystemExit):
        finish_notice(ns(finish_notice='30000:100'))   # reserve below 4096


def test_non_pi_agents_default_off_and_reject_explicit():
    assert finish_notice(ns(agent='oracle')) is None
    with pytest.raises(SystemExit):
        finish_notice(ns(agent='oracle', finish_notice='40000:16384'))


def test_resume_keeps_stored_policy():
    old = {'protocol': {}}                                   # a run from before the default (c30, GLM held workers)
    new = {'protocol': {'finish_notice': finish_notice(ns())}}
    assert not finish_notice_conflict(old, ns())             # no flag: keep the stored (absent) policy
    assert not finish_notice_conflict(new, ns())
    assert finish_notice_conflict(old, ns(finish_notice='40000:16384'))   # explicit and different: refuse
    assert not finish_notice_conflict(old, ns(finish_notice='off'))
    assert finish_notice_conflict(new, ns(finish_notice='off'))


def test_resume_compares_values_not_policy_version():
    """#375 sftnotice review NIT-1: a v3 run resumed with an explicit, matching --finish-notice keeps going (and keeps
    its stored v3 policy); different values are still refused."""
    v3 = {'protocol': {'finish_notice': {'warn_at': 40000, 'reserve': 16384,
                                         'policy': 'submit-at-context-limit-v3-finish-notice'}}}
    assert not finish_notice_conflict(v3, ns(finish_notice='40000:16384'))
    assert finish_notice_conflict(v3, ns(finish_notice='30000:16384'))
    assert finish_notice_conflict(v3, ns(finish_notice='off'))

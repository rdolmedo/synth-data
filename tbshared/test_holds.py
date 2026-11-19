"""tbshared.holds: the memory net (optim-servers' serve_teacher net, unchanged) and the replace rule."""
import types

from tbshared import holds as H


def test_memory_net_matches_the_aa_serving_reference():
    assert H.MEMORY_STEP == 0.25 and H.MEMORY_RELEASES == 2
    assert H.memory_request(64) == f'65536 + 16384 * {H.MEMORY_HOLDS}'
    assert 'HoldReasonCode =?= 34' in H.MEMORY_RELEASE and 'NumHoldsByReason.JobOutOfResources' in H.MEMORY_RELEASE
    assert H.submit_lines(1) == f'request_memory = 1024 + 256 * {H.MEMORY_HOLDS}\nperiodic_release = {H.MEMORY_RELEASE}\n'


def test_held_action():
    assert H.held_action('34', '1') == 'release' and H.held_action(34, 2) == 'release'
    assert H.held_action('34', '3') == 'replace'                 # past the bound
    assert H.held_action('26', '0') == 'replace'                 # any other hold
    assert H.held_action(None, None) == 'wait' and H.held_action('undefined', '0') == 'wait'


def test_node_of():
    assert H.node_of('slot1_2@cpu302.internal.cluster.internal') == 'cpu302'
    assert H.node_of('undefined') is None and H.node_of('') is None


def _runner(q_out, rm_rc=0, calls=None):
    def run(argv, **kw):
        if calls is not None:
            calls.append(argv)
        if argv[0] == 'condor_q':
            return types.SimpleNamespace(returncode=0, stdout=q_out, stderr='')
        return types.SimpleNamespace(returncode=rm_rc, stdout='', stderr='')
    return run


def test_remove_held():
    calls = []
    assert H.remove_held('5.0', _runner('34 1 slot1@gpu101.x', calls=calls)) == ('release', None, '34')
    assert not any(c[0] == 'condor_rm' for c in calls)
    assert H.remove_held('5.0', _runner('34 3 slot1@gpu101.x', calls=calls)) == ('replace', 'gpu101', '34')
    assert ['condor_rm', '-constraint', 'ClusterId == 5 && ProcId == 0 && JobStatus == 5'] in calls   # still held only
    assert H.remove_held('5.0', _runner('21 0 slot1_4@cpu305.internal')) == ('replace', 'cpu305', '21')
    assert H.remove_held('5.0', _runner('21 0 slot1@cpu305.x', rm_rc=1)) == ('wait', None, '21')   # retried next pass
    assert H.remove_held('5.0', _runner('')) == ('wait', None, None)
    assert H.remove_held('5.0', _runner('1 0 slot1@gpu101.x')) == ('wait', None, '1')     # a user's hold: left alone
    import pytest
    with pytest.raises(ValueError):
        H.remove_held('5', _runner('21 0 x'))                                            # never a whole cluster


def test_replacement_gib_and_cap():
    assert H.replacement_gib(100, '34') == 150 and H.replacement_gib(100, 21) == 100 and H.replacement_gib(100, None) == 100
    assert H.MAX_REPLACEMENTS == 3


def test_jobs_without_the_net_replace_memory_holds_too():
    assert H.held_action('34', '1', has_net=False) == 'replace' and H.held_action('1', '0', has_net=False) == 'wait'
    assert H.remove_held('5.0', _runner('34 1 slot1@gpu101.x'), has_net=False) == ('replace', 'gpu101', '34')

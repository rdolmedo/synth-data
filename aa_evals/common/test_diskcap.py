"""aa_evals.common.diskcap: the AA harnesses' per-sandbox disk cap settings (#405)."""
import json

import pytest

from aa_evals.common import diskcap


def test_arg():
    assert diskcap.arg('3:30720') == {'factor': 3.0, 'min_mb': 30720} == diskcap.DEFAULT
    assert diskcap.arg('2') == {'factor': 2.0, 'min_mb': 0}
    for off in ('off', 'OFF', '0', '', '0:30720', '-1:5'):
        assert diskcap.arg(off) is None, off
    with pytest.raises(ValueError):
        diskcap.arg('lots')


def test_config_default_and_opt_out():
    assert diskcap.config({}) is None                                      # no key: an AA run before #405, uncapped
    assert diskcap.config({'diskcap': diskcap.DEFAULT}) == diskcap.DEFAULT == diskcap.arg(diskcap.DEFAULT_ARG)   # auto, no family
    assert diskcap.config({'diskcap': True}) == diskcap.DEFAULT
    for off in (None, False, 0, 'off', {'factor': 0, 'min_mb': 30720}):
        assert diskcap.config({'diskcap': off}) is None, off
    assert diskcap.config({'diskcap': {'factor': 1.5, 'min_mb': 20480}}) == {'factor': 1.5, 'min_mb': 20480}
    assert diskcap.config({'diskcap': 2}) == {'factor': 2.0, 'min_mb': 0}
    cfg = {'diskcap': diskcap.DEFAULT}
    diskcap.config(cfg)['factor'] = 9                                     # a copy, not the run config
    assert cfg['diskcap']['factor'] == 3.0 and diskcap.DEFAULT['factor'] == 3.0


def test_env(tmp_path):
    e = diskcap.env({'HARBOR_NS_DISKCAP': '7', 'HARBOR_NS_DISKCAP_MIN_MB': '1', 'X': 'y'}, {'diskcap': diskcap.DEFAULT},
                    tmp_path / 'diskcap.json')
    assert e == {'X': 'y', 'HARBOR_NS_DISKCAP': '3', 'HARBOR_NS_DISKCAP_MIN_MB': '30720',
                 'HARBOR_NS_DISKCAP_LOG': str(tmp_path / 'diskcap.json')}
    off = diskcap.env({'HARBOR_NS_DISKCAP': '3', 'HARBOR_NS_DISKCAP_LOG': '/x'}, {'diskcap': None}, tmp_path / 'd.json')
    assert off == {'HARBOR_NS_DISKCAP': '0'}                              # explicit: an inherited value cannot turn it on
    assert diskcap.env({}, {'diskcap': {'factor': 1.5, 'min_mb': 0}}, 'p')['HARBOR_NS_DISKCAP'] == '1.5'


def test_outcome_fields(tmp_path):
    p = tmp_path / 'diskcap.json'
    assert diskcap.outcome_fields(p) == {'diskcap_limit_mb': None}        # cap off, or the box never started
    p.write_text(json.dumps({'limit_mb': 30720.0, 'peak_mb': 812.4, 'largest_file_mb': 300.0,
                             'kills': [{'pid': 1, 'why': 'writer'}], 'at_fsize': ['main:/app/disk.qcow2']}))
    assert diskcap.outcome_fields(p) == {'diskcap_limit_mb': 30720.0, 'sandbox_disk_peak_mb': 812.4,
                                         'sandbox_largest_file_mb': 300.0, 'diskcap_kills': 1, 'diskcap_at_fsize': 1}
    (tmp_path / 'diskcap.verifier.json').write_text(json.dumps({'limit_mb': 30720.0, 'peak_mb': 5.0, 'kills': [{}]}))
    out = diskcap.outcome_fields(p)
    assert out['verifier_disk_peak_mb'] == 5.0 and out['diskcap_kills'] == 2


def test_check_harbor_env():
    diskcap.check_harbor_env({'HARBOR_NS_MEMCAP': '1.5'})
    with pytest.raises(SystemExit):
        diskcap.check_harbor_env({'HARBOR_NS_DISKCAP': '0'})


def test_on_and_default_arg():
    assert diskcap.DEFAULT_ARG == 'auto' and diskcap.arg('3:30720') == diskcap.DEFAULT
    assert diskcap.arg('on') == diskcap.DEFAULT


def test_spare_does_not_leak_from_the_worker_env():
    for cfg in ({}, {'diskcap': diskcap.DEFAULT}):
        assert 'HARBOR_NS_DISKCAP_SPARE' not in diskcap.env({'HARBOR_NS_DISKCAP_SPARE': 'python3'}, cfg, 'p')


def test_harbor_env_cannot_smuggle_the_cap():
    with pytest.raises(SystemExit):
        diskcap.check_harbor_env({'HARBOR_EXTRA_ARGS': '--ek ns_diskcap=0'})


def test_check_stored():
    diskcap.check_stored({'diskcap': None}, None)              # no flag: nothing to check
    diskcap.check_stored({'diskcap': None}, 'off')
    diskcap.check_stored({'diskcap': diskcap.DEFAULT}, '3:30720')
    diskcap.check_stored({}, 'off')                             # a config without the key = off (before #405)
    for cfg in ({'diskcap': None}, {}):
        with pytest.raises(SystemExit):
            diskcap.check_stored(cfg, '3:30720')


def test_same_parsing_as_eval_q35():
    """eval_q35's per-task form: a pi_eval worker (eval_q35's env) and harbor_mswea read it alike."""
    cfg = {'diskcap': {'factor': 1, 'min_mb': 32768, 'tasks': {'rstan-to-pystan': 92160},
                       'one_per_worker': ['rstan-to-pystan']}}
    assert diskcap.config(cfg, 'rstan-to-pystan') == {'factor': 1.0, 'min_mb': 92160}
    assert diskcap.config(cfg, 'other') == {'factor': 1.0, 'min_mb': 32768}
    assert diskcap.env({}, cfg, 'p', 'rstan-to-pystan')['HARBOR_NS_DISKCAP_MIN_MB'] == '92160'
    with pytest.raises(ValueError):
        diskcap.config({'diskcap': {'factor': 1, 'tasks': {'x': 0}}})
    assert diskcap._q35().DISKCAP_DEFAULT == diskcap.DEFAULT


def test_arg_rejects_what_harbor_ns_cannot_run():
    for bad in ('3:-5', 'nan', 'inf', '3:inf', '3:30720:9'):
        with pytest.raises(ValueError):
            diskcap.arg(bad)
    with pytest.raises(SystemExit):
        diskcap.check_stored({'diskcap': diskcap.DEFAULT}, 'lots')     # a clean exit, not a traceback


def test_auto_is_eval_q35s_per_task_caps():
    """#405 (coordinator, 2026-11-18): TB 2.1 in AA uses eval's DISKCAP_FAMILIES['tb2.1'] (aa_evals/tb21/run.py
    "diskcap_family"); a bench without a family gets DEFAULT."""
    names = ['rstan-to-pystan', 'sam-cell-seg', 'fix-git']
    cap = diskcap.arg('auto', 'tb2.1', names)
    assert cap == {'factor': 1.0, 'min_mb': 32768, 'tasks': {'rstan-to-pystan': 92160, 'sam-cell-seg': 40960},
                   'one_per_worker': ['rstan-to-pystan']}
    cfg = {'diskcap': json.loads(json.dumps(cap))}                       # as stored in config.json
    assert [diskcap.config(cfg, t)['min_mb'] for t in names] == [92160, 40960, 32768]
    assert diskcap.arg('auto', None, names) == diskcap.arg('auto', 'no-such-family', []) == diskcap.DEFAULT
    diskcap.check_stored(cfg, 'auto', 'tb2.1', names)                    # same policy
    with pytest.raises(SystemExit):
        diskcap.check_stored(cfg, '3:30720', 'tb2.1', names)
    diskcap.check_stored({'diskcap': diskcap.DEFAULT}, 'auto', None, names)


def test_tb21_defaults_name_the_family():
    pytest.importorskip('harbor')
    from aa_evals.tb21.run import DEFAULTS
    assert DEFAULTS['diskcap_family'] == 'tb2.1' and 'diskcap' not in DEFAULTS   # --diskcap auto resolves it

"""TB 2.1 run defaults (#378): the per-sandbox memory cap is on for new runs."""
import pytest

pytest.importorskip('harbor')


def test_tb21_caps_sandbox_memory_like_tbsci():
    from aa_evals.tb21.run import DEFAULTS
    from aa_evals.tbsci import pi as tbsci
    assert DEFAULTS['harbor_env']['HARBOR_NS_MEMCAP'] == '1.5'
    lite = next(v for k, v in vars(tbsci).items() if isinstance(v, dict) and 'harbor_env' in v)
    assert DEFAULTS['harbor_env']['HARBOR_NS_MEMCAP'] == lite['harbor_env']['HARBOR_NS_MEMCAP']

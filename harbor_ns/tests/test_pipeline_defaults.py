"""Defaults that differ between pipelines on main (#480): aa-evals/eval keep their own unless the synth variables are
set (synth/worker.py).

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q harbor_ns/tests/test_pipeline_defaults.py
"""
import pytest

from harbor_ns.environment import service_proxy_env


# Image eviction is one rule for every pipeline since #451 lesson 7: tests/test_scratch_recovery.py.


def _rootfs(tmp_path):
    r = tmp_path / 'rootfs'
    (r / 'etc').mkdir(parents=True)
    (r / 'etc' / 'hosts').write_text('127.0.0.10 api.lab.internal\n10.1.2.3 db.example\n')
    return r


@pytest.fixture
def proxy(monkeypatch):
    from harbor_ns import sandbox
    for k in ('HARBOR_NS_OFFLINE_PROXY_HOSTS', 'HARBOR_NS_NO_PROXY'):   # read from the process env, not `env`
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sandbox, '_PROXY_URL', 'http://proxy.internal:8080', raising=False)
    monkeypatch.setattr(sandbox, 'proxy_url', lambda: 'http://proxy.internal:8080')


def test_offline_service_keeps_online_no_proxy_by_default(tmp_path, proxy):
    """aa-evals/eval: a service started without internet gets the same no_proxy as an online one."""
    r = _rootfs(tmp_path)
    off = service_proxy_env({'main': '10.0.0.2'}, r, internet=False, env={})
    on = service_proxy_env({'main': '10.0.0.2'}, r, internet=True, env={})
    assert off == on and 'api.lab.internal' not in off['no_proxy'] and '.internal' not in off['no_proxy']


def test_offline_service_bypass_with_synth_flag(tmp_path, proxy):
    r = _rootfs(tmp_path)
    px = service_proxy_env({'main': '10.0.0.2'}, r, internet=False, env={'HARBOR_NS_OFFLINE_BYPASS': '1'})
    no = px['no_proxy'].split(',')
    assert 'api.lab.internal' in no and 'db.example' in no and '.internal' in no and '10.0.0.0/8' in no
    # online services never get the offline additions
    on = service_proxy_env({'main': '10.0.0.2'}, r, internet=True, env={'HARBOR_NS_OFFLINE_BYPASS': '1'})
    assert '.internal' not in on['no_proxy'].split(',')


def test_offline_package_proxy_and_bare(tmp_path, proxy):
    r = _rootfs(tmp_path)
    px = service_proxy_env({}, r, internet=False, env={'HARBOR_NS_OFFLINE_PROXY_URL': 'http://127.0.0.1:3128',
                                                       'HARBOR_NS_OFFLINE_PIP_INDEX': 'http://127.0.0.1:3128/pypi/simple/'})
    assert px['http_proxy'] == 'http://127.0.0.1:3128' and 'api.lab.internal' in px['no_proxy']
    assert px['PIP_INDEX_URL'].endswith('/pypi/simple/') and px['PIP_TRUSTED_HOST'] == '127.0.0.1'
    assert service_proxy_env({}, r, internet=False, env={'HARBOR_NS_OFFLINE_NO_PROXY': '1'}) == {}

"""CPU tests for harbor_ns build-time proxy handling (issue #7).

  PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
    /opt/aai/venvs/harbor/bin/python -m pytest -q harbor_ns/tests

The network scenario (a real build through a relay that refuses connections) is
harbor_ns/tests/net_build_check.py; it needs user namespaces and the cluster proxy.
"""
import asyncio
import json
from pathlib import Path

import pytest

from harbor_ns import builder
from harbor_ns.builder import BuildError, ImageMeta, ImageStore, proxy_failures
from harbor_ns.sandbox import RuntimeConfig

# Lines from /data/aai/tbench/harbor_ns/logs/*.failed.log (2026-10-23).
APT_REFUSED = """\
Err:1 http://archive.ubuntu.com/ubuntu jammy InRelease
  Could not connect to 10.22.0.107:8080 (10.22.0.107). - connect (104: Connection reset by peer) [IP: 10.22.0.107 8080]
W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease  Could not connect to 10.22.0.107:8080 (10.22.0.107). - connect (104: Connection reset by peer) [IP: 10.22.0.107 8080]
W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy-updates/InRelease  Unable to connect to 10.22.0.107:8080: [IP: 10.22.0.107 8080]
W: Some index files failed to download. They have been ignored, or old ones used instead.
E: Failed to fetch http://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/python3.10_3.10.12-1~22.04.11_amd64.deb  Connection failed [IP: 10.22.0.107 8080]
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
/tmp/post_install.sh: line 2: pip3: command not found
"""
PIP_PROXY = """\
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProxyError('Cannot connect to proxy.', ConnectionResetError(104, 'Connection reset by peer'))': /simple/pytest/
ERROR: Could not find a version that satisfies the requirement pytest (from versions: none)
ERROR: No matching distribution found for pytest
"""
NOT_PROXY = """\
E: Failed to fetch http://archive.ubuntu.com/ubuntu/pool/main/x/x.deb  404  Not Found [IP: 10.22.0.107 8080]
E: Unable to locate package python3-nonexistent
ERROR: No matching distribution found for nonexistent-package
curl: (7) Failed to connect to localhost port 5000 after 0 ms: Connection refused
"""


def test_proxy_failures_counts_final_apt_and_pip_failures():
    assert proxy_failures(APT_REFUSED) == 3  # two index files + one archive; Err:/Some index lines are not double counted
    assert proxy_failures(PIP_PROXY) == 2  # pip's final errors after proxy retries
    assert proxy_failures(APT_REFUSED + PIP_PROXY) == 5


def test_proxy_failures_ignores_genuine_failures_and_localhost():
    assert proxy_failures(NOT_PROXY, 8080) == 0


def test_proxy_failures_curl_and_wget_only_for_the_proxy_port():
    text = ("curl: (7) Failed to connect to 10.22.0.107 port 8080 after 3 ms: Connection refused\n"
            "Connecting to 10.22.0.107:8080... failed: Connection refused.\n")
    assert proxy_failures(text, 8080) == 2
    assert proxy_failures(text, None) == 0


def store(tmp_path, **kw):
    rt = RuntimeConfig(nsinit=tmp_path / 'nsinit', work_root=tmp_path / 'boxes', nsfake=tmp_path / 'nsfake.so')
    return ImageStore(local_root=tmp_path / 'local', shared_root=tmp_path / 'shared', rt=rt, nsfake=rt.nsfake,
                      network_backoff=(0.0,), **kw)


def context(tmp_path):
    ctx = tmp_path / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text('FROM ubuntu:22.04\nRUN true\n')
    return ctx


def fake_builds(outcomes):
    """Stand-in for ImageStore._build: each call appends a log and fails or succeeds."""
    calls = []

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        text, fail = outcomes[len(calls)]
        calls.append(sorted(p.name for p in d.iterdir()))
        (d / 'rootfs').mkdir()
        (d / 'rootfs' / 'marker').write_text(str(len(calls)))
        with open(log, 'a') as f:
            f.write(text)
        if fail:
            raise BuildError('RUN (line 7) failed with exit code 127')
        return ImageMeta()
    return _build, calls


def test_build_retries_after_proxy_failure_then_succeeds(tmp_path, monkeypatch):
    _build, calls = fake_builds([(APT_REFUSED, True), ('Setting up jq\n', False)])
    monkeypatch.setattr(ImageStore, '_build', _build)
    s = store(tmp_path)
    key, rootfs, meta = asyncio.run(s.build(context(tmp_path), label='t'))
    assert len(calls) == 2
    assert calls[1] == ['build.log']  # the failed attempt's rootfs was removed, its log kept
    assert (rootfs / 'marker').read_text() == '2'
    md = json.loads((s.image_dir(key) / 'meta.json').read_text())
    assert md['build_attempts'] == 2 and md['proxy_failures_last_attempt'] == 0
    assert 'rebuilding from scratch' in (s.image_dir(key) / 'build.log').read_text()
    assert not list((tmp_path / 'shared' / 'logs').glob('*.failed.log'))


def test_build_with_silent_proxy_failure_is_rebuilt(tmp_path, monkeypatch):
    # post_install.sh without `set -e`: apt failed but the build "succeeded".
    _build, calls = fake_builds([(APT_REFUSED.replace('/tmp/post_install.sh: line 2: pip3: command not found\n', ''), False),
                                 ('Setting up jq\n', False)])
    monkeypatch.setattr(ImageStore, '_build', _build)
    s = store(tmp_path)
    key, rootfs, _ = asyncio.run(s.build(context(tmp_path), label='t'))
    assert len(calls) == 2 and (rootfs / 'marker').read_text() == '2'


def test_build_gives_up_after_network_rebuilds_and_keeps_log(tmp_path, monkeypatch):
    _build, calls = fake_builds([(APT_REFUSED, True)] * 3)
    monkeypatch.setattr(ImageStore, '_build', _build)
    s = store(tmp_path, network_rebuilds=2)
    with pytest.raises(BuildError):
        asyncio.run(s.build(context(tmp_path), label='t'))
    assert len(calls) == 3
    failed = list((tmp_path / 'shared' / 'logs').glob('t.*.failed.log'))
    assert len(failed) == 1 and failed[0].read_text().count('rebuilding from scratch') == 2


def test_non_network_failure_is_not_rebuilt(tmp_path, monkeypatch):
    _build, calls = fake_builds([('E: Unable to locate package nonexistent\n', True)])
    monkeypatch.setattr(ImageStore, '_build', _build)
    with pytest.raises(BuildError):
        asyncio.run(store(tmp_path).build(context(tmp_path), label='t'))
    assert len(calls) == 1


def test_network_rebuilds_zero_disables(tmp_path, monkeypatch):
    _build, calls = fake_builds([(APT_REFUSED, True)])
    monkeypatch.setattr(ImageStore, '_build', _build)
    with pytest.raises(BuildError):
        asyncio.run(store(tmp_path, network_rebuilds=0).build(context(tmp_path), label='t'))
    assert len(calls) == 1


def test_shims_write_build_only_apt_config(tmp_path):
    root = tmp_path / 'rootfs'
    (root / 'etc/apt/apt.conf.d').mkdir(parents=True)
    store(tmp_path).install_shims(root, 'musl')  # libc branch not needed for the apt files
    conf = root / builder.APT_BUILD_CONF_PATH.lstrip('/')
    assert 'Acquire::Retries "10";' in conf.read_text()
    # Runtime apt config is unchanged: only the sandbox-user line, no retries.
    assert (root / 'etc/apt/apt.conf.d/00harbor-ns').read_text() == 'APT::Sandbox::User "root";\n'


def test_run_step_sets_apt_config_only_when_present(tmp_path, monkeypatch):
    seen = []

    async def fake_build_run(cfg, rootfs, argv, env, cwd, lf, timeout=None, binds=None):
        seen.append(env.get('APT_CONFIG'))
        return 0
    monkeypatch.setattr(builder, 'build_run', fake_build_run)
    from harbor_ns import dockerfile as dfp
    stages, _ = dfp.parse('FROM ubuntu:22.04\nRUN apt-get update\n')
    ins = stages[0].instructions[0]
    s = store(tmp_path)
    root = tmp_path / 'r'
    (root / 'tmp').mkdir(parents=True)
    with open(tmp_path / 'log', 'wb') as lf:
        asyncio.run(s._run_step(ins, root, {}, {}, '/', ['/bin/sh', '-c'], lf))
        (root / 'etc/apt/apt.conf.d').mkdir(parents=True)
        s.install_shims(root, 'musl')
        asyncio.run(s._run_step(ins, root, {}, {}, '/', ['/bin/sh', '-c'], lf))
        asyncio.run(s._run_step(ins, root, {'APT_CONFIG': '/etc/own.conf'}, {}, '/', ['/bin/sh', '-c'], lf))
    assert seen == [None, builder.APT_BUILD_CONF_PATH, '/etc/own.conf']


NPM_PROXY = ("npm ERR! code FETCH_ERROR\nnpm ERR! errno FETCH_ERROR\nnpm ERR! request to https://registry.npmjs.org/"
             "@types%2fnode failed, reason: Proxy connection ended before receiving CONNECT response\n")
GIT_PROXY = ("fatal: unable to access 'https://github.com/x/y.git/': Received HTTP code 503 from proxy after CONNECT\n"
             "fatal: unable to access 'https://github.com/x/z.git/': Failed to connect to 10.22.0.107 port 8080 after "
             "3 ms: Connection refused\n")
NPM_GIT_LOCAL = ("npm ERR! request to http://127.0.0.1:4873/left-pad failed, reason: connect ECONNREFUSED 127.0.0.1:4873\n"
                 "fatal: unable to access 'http://localhost:3000/repo.git/': Failed to connect to localhost port 3000\n"
                 "fatal: repository 'https://github.com/x/missing.git/' not found\n")


def test_proxy_failures_npm_and_git_through_the_proxy_only():
    assert proxy_failures(NPM_PROXY, 8080) == 1
    assert proxy_failures(GIT_PROXY, 8080) == 2
    assert proxy_failures(NPM_GIT_LOCAL, 8080) == 0


def cache_tag_file(tmp_path, monkeypatch):
    """A package cache store whose per-build log the fake build writes into (the cache answered 503)."""
    root = tmp_path / 'pkgcache'
    (root / 'tags').mkdir(parents=True)
    monkeypatch.setattr(builder, 'package_cache', lambda: {'root': str(root), 'url': 'http://127.0.0.1:1', 'port': 1})

    def answer_503(url):
        tag = builder._BUILD_CACHE.get()[0]
        import time
        with open(root / 'tags' / f'{tag}.jsonl', 'a') as f:
            f.write(json.dumps({'t': time.time(), 'tag': tag, 'kind': 'pypi-page', 'url': url, 'status': 503,
                                'cache': 'error', 'error': 'UpstreamError: upstream breaker open'}) + '\n')
    return answer_503


PIP_NONE = ("ERROR: Could not find a version that satisfies the requirement pyarrow (from versions: none)\n"
            "ERROR: No matching distribution found for pyarrow\n")


def test_cache_503_counts_as_a_network_failure_and_is_rebuilt(tmp_path, monkeypatch):
    """pip at default verbosity says only "from versions: none" when the cache's index answered 503 (upstream down,
    page older than --stale-max): the cache's own log for the build marks it as a network failure."""
    answer_503 = cache_tag_file(tmp_path, monkeypatch)
    calls = []

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        calls.append(1)
        (d / 'rootfs').mkdir(exist_ok=True)
        if len(calls) == 1:
            answer_503('https://pypi.org/simple/pyarrow/')
            with open(log, 'a') as f:
                f.write(PIP_NONE)
            raise BuildError('RUN (line 3) failed with exit code 1')
        return ImageMeta()
    monkeypatch.setattr(ImageStore, '_build', _build)
    s = store(tmp_path)
    key, _, _ = asyncio.run(s.build(context(tmp_path), label='t'))
    assert len(calls) == 2
    text = (s.image_dir(key) / 'build.log').read_text()
    assert 'package cache: 1 requests of this attempt failed upstream' in text and 'rebuilding from scratch' in text


def test_cache_503_in_the_last_attempt_marks_the_failed_log(tmp_path, monkeypatch):
    answer_503 = cache_tag_file(tmp_path, monkeypatch)

    async def _build(self, ctx, dockerfile, d, log, ensure_tmux=False):
        answer_503('https://pypi.org/simple/pyarrow/')
        with open(log, 'a') as f:
            f.write(PIP_NONE)
        raise BuildError('RUN (line 3) failed with exit code 1')
    monkeypatch.setattr(ImageStore, '_build', _build)
    with pytest.raises(BuildError):
        asyncio.run(store(tmp_path, network_rebuilds=1).build(context(tmp_path), label='t'))
    failed = next((tmp_path / 'shared' / 'logs').glob('t.*.failed.log')).read_text()
    assert proxy_failures(failed.rsplit('rebuilding from scratch', 1)[-1], 8080) >= 1   # check(): infra, not the task
    assert proxy_failures(PIP_NONE, 8080) == 0    # without the cache's record it reads like a missing package

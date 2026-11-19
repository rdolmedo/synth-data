"""Skip the tests listed in CLUSTER_ONLY_TESTS (they need the original cluster: HTCondor, shared filesystems,
GPUs or prebuilt container images). One pytest node id or file path per line; '#' starts a comment."""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    p = os.path.join(_HERE, "CLUSTER_ONLY_TESTS")
    out = set()
    if os.path.exists(p):
        for line in open(p):
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


_SKIP = _load()


def pytest_ignore_collect(collection_path, config):
    rel = os.path.relpath(str(collection_path), _HERE)
    return rel in _SKIP or None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in _SKIP:
            item.add_marker(pytest.mark.skip(reason="needs the original cluster (CLUSTER_ONLY_TESTS)"))

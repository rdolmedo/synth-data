"""Tests for harbor_ns.memcap (python -m pytest harbor_ns/test_memcap.py)."""
import asyncio
import os
import shutil
import subprocess
import sys
import time

import pytest

from harbor_ns import memcap

HOG = "import sys, time; b = bytearray(int(sys.argv[1]) << 20); b[::4096] = b'x' * len(b[::4096]); print('ok', flush=True); time.sleep(120)"


def hog(mb: int) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", HOG, str(mb)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ok"   # memory touched
    return p


def test_parse():
    assert memcap.parse(None) is None
    for off in ("", "0", "false", "off", "no", "-1"):
        assert memcap.parse(off) is None
    for on in ("1", "true", "on", "yes", "TRUE"):
        assert memcap.parse(on) == 1.0
    assert memcap.parse("1.5") == 1.5
    assert memcap.parse(2) == 2.0


def test_usage_counts_anonymous_memory():
    p = hog(200)
    try:
        assert 190 * 1024 < memcap.usage_kb(p.pid) < 260 * 1024
    finally:
        p.kill(); p.wait()
    assert memcap.usage_kb(p.pid) == 0   # gone


def test_kills_largest_only_above_limit():
    big, small = hog(300), hog(50)
    try:
        logs = []
        cap = memcap.MemCap(None, 1000, members=lambda: [big.pid, small.pid], log=logs.append)
        assert cap.check() is None and big.poll() is None   # 350 MB < 1000 MB
        cap = memcap.MemCap(None, 200, members=lambda: [big.pid, small.pid], log=logs.append)
        assert cap.check() == big.pid
        big.wait(timeout=10)
        assert big.returncode == -9 and small.poll() is None
        assert cap.kills[0]["pid"] == big.pid and "killed pid" in logs[-1]
        assert cap.check() is None   # 50 MB left, under the limit
    finally:
        for p in (big, small):
            p.kill(); p.wait()


def test_spares_the_agent_unless_it_is_all_that_is_left():
    agent, worker = hog(300), hog(50)
    try:
        names = {agent.pid: "pi", worker.pid: "python3"}
        cap = memcap.MemCap(None, 200, members=lambda: [p for p in names if memcap.usage_kb(p)],
                            comm=names.__getitem__, log=lambda m: None)
        assert cap.check() == worker.pid   # the smaller task process goes first, not the agent
        worker.wait(timeout=10)
        assert worker.returncode == -9 and agent.poll() is None
        assert cap.check() == agent.pid    # still over the limit with only the agent left
        agent.wait(timeout=10)
        assert agent.returncode == -9
    finally:
        for p in (agent, worker):
            p.kill(); p.wait()


def test_run_loop_kills_and_survives_errors():
    p = hog(150)
    calls = {"n": 0}

    def members():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")   # logged, the loop goes on
        return [p.pid]

    async def go():
        cap = memcap.MemCap(None, 100, interval=0.05, members=members, log=lambda m: None)
        t = asyncio.create_task(cap.run())
        for _ in range(100):
            if p.poll() is not None:
                break
            await asyncio.sleep(0.05)
        t.cancel()
        return cap

    try:
        cap = asyncio.run(go())
        assert p.wait(timeout=10) == -9 and len(cap.kills) == 1
    finally:
        if p.poll() is None:
            p.kill(); p.wait()


@pytest.mark.skipif(shutil.which("unshare") is None, reason="needs unshare")
def test_pidns_members_finds_processes_in_the_namespace():
    """A real pid namespace (as a Sandbox makes): the box's processes are found, init is not, the host's are not."""
    outer = subprocess.Popen(["unshare", "-Upf", "--kill-child", "sh", "-c", "sleep 60 & sleep 60 & wait"],
                             stderr=subprocess.PIPE)
    try:
        init = None
        for _ in range(100):
            kids = subprocess.run(["pgrep", "-P", str(outer.pid)], capture_output=True, text=True).stdout.split()
            if kids:
                init = int(kids[0])
                if len(subprocess.run(["pgrep", "-P", str(init)], capture_output=True, text=True).stdout.split()) == 2:
                    break
            time.sleep(0.05)
        if init is None:
            pytest.skip("unprivileged user+pid namespaces unavailable")
        found = memcap.pidns_members(init)
        assert len(found) == 2 and init not in found and os.getpid() not in found
    finally:
        outer.kill(); outer.wait()


def test_limit_mb(monkeypatch):
    monkeypatch.delenv("HARBOR_NS_MEMCAP_MIN_MB", raising=False)
    assert memcap.limit_mb(2048, None) is None               # cap off
    assert memcap.limit_mb(2048, 2.0) == 4096
    assert memcap.limit_mb(None, 2.0) is None                # undeclared, no floor: uncapped (aa-evals behaviour)
    assert memcap.limit_mb(2048, 2.0, min_mb=16384) == 16384  # floor raises small caps
    assert memcap.limit_mb(16384, 2.0, min_mb=8192) == 32768
    assert memcap.limit_mb(None, 2.0, min_mb=16384) == 16384  # undeclared gets the floor
    monkeypatch.setenv("HARBOR_NS_MEMCAP_MIN_MB", "12288")
    assert memcap.limit_mb(None, 1.0) == 12288
    assert memcap.limit_mb(None, None) is None


def test_summary():
    m = memcap.MemCap(None, 100, members=lambda: [], log=lambda s: None)
    m.peak_kb = 50 * 1024
    assert m.summary() == {"limit_mb": 100, "peak_mb": 50, "kills": []}

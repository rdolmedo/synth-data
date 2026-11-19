"""Tests for harbor_ns.diskcap (python -m pytest harbor_ns/test_diskcap.py)."""
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harbor_ns import diskcap, sandbox

# Rewrites a 1 MiB block of `path` forever (fsync'ed): write_bytes grows by ~MB/s while the file stays 1 MiB.
WRITER = ("import os, sys, time\n"
          "fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT)\n"
          "b = b'x' * (1 << 20)\n"
          "print('ok', flush=True)\n"
          "while True:\n"
          "    os.pwrite(fd, b, 0); os.fsync(fd); time.sleep(0.02)\n")


def writer(path) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", WRITER, str(path)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ok"
    return p


def idle() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "120"])


def fill(path: Path, mb: int) -> None:
    with open(path, "wb") as f:
        f.write(os.urandom(1 << 20) * mb)


def test_parse_and_limit_mb(monkeypatch):
    monkeypatch.delenv("HARBOR_NS_DISKCAP_MIN_MB", raising=False)
    assert diskcap.parse(None) is None and diskcap.parse("off") is None and diskcap.parse("0") is None
    assert diskcap.parse("2") == 2.0 and diskcap.parse("on") == 1.0
    assert diskcap.limit_mb(10240, None) is None                  # cap off
    assert diskcap.limit_mb(10240, 2.0) == 20480                  # TB2.1: storage_mb 10240
    assert diskcap.limit_mb(None, 2.0) is None                    # undeclared, no floor: uncapped
    assert diskcap.limit_mb(1024, 2.0, min_mb=20480) == 20480     # the floor raises small caps
    assert diskcap.limit_mb(None, 2.0, min_mb=20480) == 20480     # undeclared tasks get the floor
    monkeypatch.setenv("HARBOR_NS_DISKCAP_MIN_MB", "4096")
    assert diskcap.limit_mb(None, 1.0) == 4096
    assert diskcap.limit_mb(10240, 1.0) == 10240


def test_fsize_bytes():
    assert diskcap.fsize_bytes(None) is None and diskcap.fsize_bytes(0) is None
    assert diskcap.fsize_bytes(20480) == 20480 << 20


def test_tree_usage_counts_allocated_blocks(tmp_path):
    up = tmp_path / "upper"
    (up / "app" / "deep").mkdir(parents=True)
    fill(up / "app" / "big.bin", 3)
    fill(up / "app" / "deep" / "small.bin", 1)
    os.link(up / "app" / "big.bin", up / "app" / "big.link")      # a hard link is counted once
    with open(up / "app" / "sparse.img", "wb") as f:               # 1 GiB apparent, nothing allocated
        f.truncate(1 << 30)
    os.symlink("/nowhere", up / "app" / "dangling")
    u = diskcap.tree_usage([("main", up)], fsize=512 << 20)
    assert 4 << 20 <= u["bytes"] < 5 << 20
    assert u["files"] == 4   # big, small, sparse, the symlink (the hard link is not counted again)
    assert u["largest"] == (3 << 20, "main:/app/big.bin") or u["largest"][1] == "main:/app/big.link"
    assert u["largest_size"] == (1 << 30, "main:/app/sparse.img")
    assert u["at_fsize"] == ["main:/app/sparse.img"]               # at/over the per-file limit
    assert diskcap.tree_usage([("main", tmp_path / "gone")])["bytes"] == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any dir")
def test_tree_usage_reports_unreadable_dirs(tmp_path):
    d = tmp_path / "upper" / "locked"
    d.mkdir(parents=True)
    fill(d / "f", 1)
    d.chmod(0)
    try:
        u = diskcap.tree_usage([("main", tmp_path / "upper")])
        assert u["unreadable"] == 1 and u["files"] == 0
    finally:
        d.chmod(0o755)


def test_write_bytes_of_a_writer(tmp_path):
    p = writer(tmp_path / "w")
    try:
        a = diskcap.write_bytes(p.pid)
        time.sleep(0.5)
        b = diskcap.write_bytes(p.pid)
        assert a is not None and b - a >= 4 << 20
    finally:
        p.kill(); p.wait()
    assert diskcap.write_bytes(p.pid) is None   # gone


def test_write_bytes_leaves_out_reaped_children(tmp_path):
    """/proc/<pid>/io adds a reaped child's writes to its parent: a shell would look like a writer once its dd exits
    (2026-11-12 repro: the cap killed the agent's solve.sh at '78391 MB/s' right after its dd)."""
    sh = subprocess.Popen(["bash", "-c", f"dd if=/dev/zero of={tmp_path / 'f'} bs=1M count=16 conv=fsync status=none; "
                                         "echo done; sleep 30"], stdout=subprocess.PIPE, text=True)
    try:
        assert sh.stdout.readline().strip() == "done"
        whole = int(next(ln for ln in open(f"/proc/{sh.pid}/io") if ln.startswith("write_bytes:")).split()[1])
        assert whole >= 16 << 20                       # the kernel's per-process total includes the dd
        assert diskcap.write_bytes(sh.pid) < 1 << 20   # its own threads wrote nothing
    finally:
        sh.kill(); sh.wait()


@pytest.mark.skipif(shutil.which("fallocate") is None, reason="needs util-linux fallocate")
def test_tree_usage_leaves_out_blocks_past_eof(tmp_path):
    """XFS preallocates up to 8 GiB past EOF of a growing file (trimmed at close): 2026-11-12 repro, a 20 GiB file
    held 24 GiB of blocks.  Blocks past EOF (here fallocate --keep-size) do not count."""
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "growing", 1)
    if subprocess.run(["fallocate", "-n", "-o", str(1 << 20), "-l", str(8 << 20), str(up / "growing")]).returncode:
        pytest.skip("filesystem without fallocate")
    assert os.stat(up / "growing").st_blocks * 512 >= 9 << 20 and os.stat(up / "growing").st_size == 1 << 20
    u = diskcap.tree_usage([("main", up)])
    assert u["bytes"] < 2 << 20 and u["largest"] == (1 << 20, "main:/growing")


def capped(root, limit_mb, procs, **kw):
    return diskcap.DiskCap(lambda: [("main", root)], limit_mb, members_fn=lambda: [p.pid for p in procs if p.poll() is None],
                           log=kw.pop("log", lambda m: None), **kw)


def test_below_the_cap_nothing_is_killed(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)
    w = writer(tmp_path / "elsewhere")
    try:
        cap = capped(up, 100, [w])
        for _ in range(3):
            assert cap.check() is None
            time.sleep(0.3)
        assert w.poll() is None and not cap.kills and 2 <= cap.peak / 2**20 < 3
    finally:
        w.kill(); w.wait()


def test_above_the_cap_kills_the_writer_not_the_idle(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)                       # 2 MB > 1.5 MB: over the cap (not the 2.25 MB ceiling)
    w, z = writer(tmp_path / "elsewhere"), idle()
    try:
        logs = []
        cap = capped(up, 1.5, [w, z], log=logs.append)
        assert cap.check() == w.pid         # the first pass above the cap samples twice (PROBE) and kills
        w.wait(timeout=10)
        assert w.returncode == -9 and z.poll() is None
        assert cap.kills[0]["pid"] == w.pid and cap.kills[0]["mb_per_s"] >= 2 and "killed pid" in logs[-1]
        time.sleep(0.5)
        assert cap.check() is None and z.poll() is None   # still over the cap, but nothing writes: left alone
    finally:
        for p in (w, z):
            p.kill(); p.wait()


def test_spares_the_agent_unless_it_is_the_only_writer(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)
    agent, task = writer(tmp_path / "a"), writer(tmp_path / "b")
    names = {agent.pid: "pi", task.pid: "python3"}
    try:
        cap = diskcap.DiskCap(lambda: [("main", up)], 1.5, members_fn=lambda: [p for p in names if diskcap.write_bytes(p)],
                              comm=names.__getitem__, log=lambda m: None)
        assert cap.check() == task.pid      # the task's writer goes, not the agent
        task.wait(timeout=10)
        time.sleep(0.5)
        assert cap.check() == agent.pid     # the agent is the only writer left above the cap
        agent.wait(timeout=10)
        assert agent.returncode == -9
    finally:
        for p in (agent, task):
            p.kill(); p.wait()


def test_ceiling_kills_everything_once_then_writers(tmp_path):
    """A loop of short writers is not stopped one writer at a time: at CEILING x the cap all but init goes, agent too."""
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)                       # 2 MB > 1.5 x 1 MB
    agent, z = idle(), idle()
    names = {agent.pid: "pi", z.pid: "bash"}
    try:
        logs = []
        cap = diskcap.DiskCap(lambda: [("main", up)], 1, members_fn=lambda: [p for p in names if diskcap.write_bytes(p)
                                                                            is not None],
                              comm=names.__getitem__, log=logs.append)
        assert cap.check() in names
        for p in (agent, z):
            assert p.wait(timeout=10) == -9
        assert [k["why"] for k in cap.kills] == ["ceiling", "ceiling"] and "runaway" in logs[-1]
        w = writer(tmp_path / "elsewhere")  # e.g. the verifier, started after: only killed if it writes
        names[w.pid] = "python3"
        z2 = idle()
        names[z2.pid] = "bash"
        assert cap.check() == w.pid and z2.poll() is None
        z2.kill(); z2.wait(); w.wait()
    finally:
        for p in (agent, z):
            p.kill(); p.wait()


def test_deleting_brings_the_box_back_under(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)
    t = [0.0]
    cap = diskcap.DiskCap(lambda: [("main", up)], 1, members_fn=lambda: [], fs_used=lambda: 0, clock=lambda: t[0],
                          log=lambda m: None)
    cap.check()
    assert cap.over and cap.over_since == 0.0
    (up / "f").unlink()
    t[0] = 1.0
    cap.check()
    assert cap.over                         # not re-walked yet (OVER_GAP)
    t[0] = diskcap.OVER_GAP
    cap.check()
    assert not cap.over and cap.over_s == diskcap.OVER_GAP


def test_walk_schedule():
    t, used = [0.0], [0]
    cap = diskcap.DiskCap(lambda: [], 1000, fs_used=lambda: used[0], clock=lambda: t[0], members_fn=lambda: [])
    assert cap.due(0.0, 0)                                     # first pass
    cap.walk(0.0, 0)
    assert not cap.due(diskcap.MIN_GAP / 2, 10**12)            # MIN_GAP
    assert not cap.due(10.0, 400 << 20)                        # grew 400 MB < half of the 1000 MB headroom
    assert cap.due(10.0, 600 << 20)                            # grew more than half the headroom
    assert cap.due(diskcap.MAX_GAP, 0)                         # at least every MAX_GAP
    assert cap.due(10.0, None)                                 # no statvfs: time-based only
    cap.total = 990 << 20                                      # 10 MB of headroom: TRIGGER bounds the walk rate
    assert not cap.due(10.0, 100 << 20) and cap.due(10.0, diskcap.TRIGGER)
    cap.walk_s = 1.0                                           # a 1 s walk: at most DUTY of a core
    assert not cap.due(1.0 / diskcap.DUTY - 1, diskcap.TRIGGER) and cap.due(1.0 / diskcap.DUTY, diskcap.TRIGGER)
    # ... unless the scratch grew by more than the whole headroom (at least FORCE): the box may be past its cap
    assert not cap.due(1.0, diskcap.FORCE - 1) and cap.due(1.0, diskcap.FORCE)
    assert not cap.due(diskcap.MIN_GAP / 2, 10**12)
    cap.walk_s, cap.total = 0.0, 1200 << 20                    # over the cap: walk towards the 1500 MB ceiling
    assert not cap.due(2.0, 100 << 20) and cap.due(2.0, diskcap.TRIGGER) and cap.due(diskcap.OVER_GAP, 0)
    cap.walk_s = 1.0                                           # over the cap, a slow walk: FORCE past the ceiling
    assert not cap.due(2.0, diskcap.FORCE - 1) and cap.due(2.0, diskcap.FORCE)
    cap.walk_s, cap.total = 0.0, 2000 << 20                    # past the ceiling: re-walk every OVER_GAP
    assert not cap.due(diskcap.OVER_GAP - 1, 10**12) and cap.due(diskcap.OVER_GAP, 0)


def test_run_loop_kills_and_survives_errors(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 2)
    w = writer(tmp_path / "elsewhere")
    calls = {"n": 0}

    def members():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")   # logged, the loop goes on
        return [w.pid]

    async def go():
        cap = diskcap.DiskCap(lambda: [("main", up)], 1, members_fn=members, log=lambda m: None)
        task = asyncio.create_task(cap.run())
        for _ in range(100):
            if w.poll() is not None:
                break
            await asyncio.sleep(0.1)
        task.cancel()
        return cap

    try:
        cap = asyncio.run(go())
        assert w.wait(timeout=10) == -9 and len(cap.kills) == 1
    finally:
        if w.poll() is None:
            w.kill(); w.wait()


def test_summary(tmp_path):
    up = tmp_path / "upper"
    up.mkdir()
    fill(up / "f", 3)
    cap = diskcap.DiskCap(lambda: [("main", up)], 20480, fsize=20480 << 20, members_fn=lambda: [], log=lambda m: None)
    cap.check()
    cap.final()
    s = cap.summary()
    assert s["limit_mb"] == 20480 and s["fsize_mb"] == 20480 and s["kills"] == [] and s["over_s"] == 0
    assert s["ceiling_mb"] == 30720
    assert 3 <= s["peak_mb"] < 3.1 and s["final_mb"] == s["peak_mb"] and s["largest_file"] == "main:/f"
    assert s["walks"] == 2 and s["largest_file_mb"] == 3.0


# ---- the per-file limit (RLIMIT_FSIZE via prlimit, sandbox.py)

needs_prlimit = pytest.mark.skipif(shutil.which("prlimit") is None, reason="needs util-linux prlimit")


@needs_prlimit
def test_fsize_argv_stops_dd_at_the_limit(tmp_path):
    out = tmp_path / "alpine-disk.qcow2"
    r = subprocess.run([*sandbox._fsize_argv(4 << 20), "dd", "if=/dev/zero", f"of={out}", "bs=1M", "count=64"],
                       capture_output=True, text=True)
    assert r.returncode == -25 or r.returncode == 128 + 25   # SIGXFSZ ("File size limit exceeded")
    assert out.stat().st_size == 4 << 20


@needs_prlimit
def test_fsize_python_gets_efbig_and_children_inherit(tmp_path):
    code = ("import errno, resource, subprocess, sys\n"
            "f = open(sys.argv[1], 'wb')\n"
            "try:\n"
            "    f.write(b'x' * (8 << 20)); f.flush()\n"
            "except OSError as e:\n"
            "    print('EFBIG' if e.errno == errno.EFBIG else repr(e))\n"
            "try:\n"
            "    resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY,) * 2)\n"
            "    print('raised')\n"
            "except (OSError, ValueError):\n"
            "    print('EPERM')\n"
            "print(subprocess.run([sys.executable, '-c', 'import resource; print(*resource.getrlimit(resource.RLIMIT_FSIZE))'],"
            " capture_output=True, text=True).stdout.strip())\n")
    r = subprocess.run([*sandbox._fsize_argv(1 << 20), sys.executable, "-c", code, str(tmp_path / "f")],
                       capture_output=True, text=True)
    assert r.stdout.split() == ["EFBIG", "EPERM", str(1 << 20), str(1 << 20)], r.stdout + r.stderr
    assert (tmp_path / "f").stat().st_size == 1 << 20   # Python ignores SIGXFSZ: the write fails like a quota error


def test_sandbox_argv_has_the_limit_only_when_capped():
    cfg = sandbox.RuntimeConfig(nsinit=Path("/x"), work_root=Path("/tmp"))
    off = sandbox.Sandbox(cfg=cfg, lower=[], init_pid=123)
    assert "prlimit" not in " ".join(off._nsenter())
    on = sandbox.Sandbox(cfg=cfg, lower=[], init_pid=123, fsize=20480 << 20)
    argv = on._nsenter()
    if sandbox.fsize_supported():
        assert argv[0].endswith("prlimit") and argv[1:3] == [f"--fsize={20480 << 20}:{20480 << 20}", "--"]
        assert "nsenter" in argv[3:]


@pytest.mark.skipif(shutil.which("unshare") is None, reason="needs unshare")
def test_members_of_two_namespaces():
    """Real pid namespaces (as a pod's service boxes): their processes are found, the inits and the host's are not."""
    outs = [subprocess.Popen(["unshare", "-Upf", "--kill-child", "sh", "-c", "sleep 60 & sleep 60 & wait"],
                             stderr=subprocess.PIPE) for _ in range(2)]
    try:
        inits = []
        for o in outs:
            for _ in range(100):
                kids = subprocess.run(["pgrep", "-P", str(o.pid)], capture_output=True, text=True).stdout.split()
                if kids and len(subprocess.run(["pgrep", "-P", kids[0]], capture_output=True,
                                               text=True).stdout.split()) == 2:
                    inits.append(int(kids[0]))
                    break
                time.sleep(0.05)
        if len(inits) != 2:
            pytest.skip("unprivileged user+pid namespaces unavailable")
        found = diskcap.members(inits)
        assert len(found) == 4 and not set(inits) & set(found) and os.getpid() not in found
        assert diskcap.members([]) == []
    finally:
        for o in outs:
            o.kill(); o.wait()


def test_tree_usage_opens_dirs_without_following_symlinks(tmp_path, monkeypatch):
    """Directories are opened O_NOFOLLOW (and must stay on the root's filesystem), so a box that swaps a listed dir
    for a symlink between the listing and the open does not send the walk into the host."""
    up = tmp_path / "upper"
    (up / "d").mkdir(parents=True)
    fill(up / "d" / "f", 1)
    os.symlink("/usr", up / "link")                       # a symlink is a file entry, never walked
    opened = []
    real_open = os.open
    monkeypatch.setattr(diskcap.os, "open", lambda p, fl, *a, **k: opened.append((p, fl)) or real_open(p, fl, *a, **k))
    u = diskcap.tree_usage([("main", up)])
    assert u["files"] == 2 and u["bytes"] < 2 << 20 and u["unreadable"] == 0
    assert len(opened) == 2 and all(fl & os.O_NOFOLLOW for _, fl in opened)
    # the swap itself: the queued dir is now a symlink -> ELOOP, counted unreadable, not followed
    monkeypatch.setattr(diskcap.os, "open", real_open)
    os.rename(up / "d", up / "d.real")
    os.symlink("/usr", up / "d")
    with pytest.raises(OSError):
        os.open(up / "d", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

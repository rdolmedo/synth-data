"""Multi-uid checks of the disk cap (#405), on real namespaces (python -m pytest harbor_ns/test_diskcap_multiuid.py).

A multi-uid Pod (MULTIUID.md: newuidmap/newgidmap over the caller's /etc/subuid range) with one Sandbox of a cached
python:3.12-slim-bookworm image (HARBOR_NS_TEST_IMAGE: a harbor_ns base .sqfs with bash, setpriv, su, dd, python3),
unpacked to local scratch.  Checks:

1. RLIMIT_FSIZE still applies after a privilege drop: dd run as a box user through setpriv, su and Harbor's
   exec(user=...) (nsenter -S/-G) stops at the cap with SIGXFSZ; Python gets EFBIG.
2. The watchdog walks upper dirs holding directories that subordinate ids made private (mode 0700) and kills processes
   running under those ids.
3. The known gap: nsdev.so clamps only RLIMIT_CORE, so `ulimit -f unlimited` in a box fails with EPERM.

Skipped where the node lacks a subordinate id range, newuidmap, unprivileged user namespaces or the image (cluster
compute nodes have all of them).  The overlay upper dir must be on a local filesystem: HARBOR_NS_TEST_SCRATCH, else
$_CONDOR_SCRATCH_DIR, else /tmp.
"""
import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from harbor_ns import diskcap, sandbox
from harbor_ns.sandbox import Pod, RuntimeConfig, Sandbox

CAP = 8 << 20   # per-file limit of the box (bytes)
UID = 1000      # a box user inside the mapped bands (MULTIUID_BANDS: 1..11000)
IMAGE = Path(os.environ.get("HARBOR_NS_TEST_IMAGE",
                            "/data/aai/tbench/harbor_ns/bases/python_3.12-slim-bookworm.sqfs"))


def _have_subids() -> bool:
    try:
        sandbox._subid_range("/etc/subuid")
        sandbox._subid_range("/etc/subgid")
    except Exception:  # noqa: BLE001
        return False
    return IMAGE.is_file() and all(shutil.which(t) for t in ("newuidmap", "newgidmap", "unshare", "nsenter", "prlimit",
                                                             "unsquashfs", "gcc"))


pytestmark = pytest.mark.skipif(not _have_subids(), reason="needs a subordinate id range, newuidmap/newgidmap, "
                                                           "unshare/nsenter, prlimit, unsquashfs, gcc and the image")


@pytest.fixture(scope="module")
def box():
    """(sandbox, pod, scratch, loop) of a multi-uid pod."""
    scratch = Path(tempfile.mkdtemp(prefix="diskcap-mu-", dir=os.environ.get("HARBOR_NS_TEST_SCRATCH")
                                    or os.environ.get("_CONDOR_SCRATCH_DIR") or "/tmp"))
    loop = asyncio.new_event_loop()
    work = scratch / "work"
    work.mkdir()
    cache = scratch / "bin"
    cfg = RuntimeConfig(nsinit=sandbox.find_runtime_bins(cache), work_root=work)
    rootfs = scratch / "rootfs"
    subprocess.run(["unsquashfs", "-no-xattrs", "-q", "-n", "-d", str(rootfs), str(IMAGE)], check=True,
                   stdout=subprocess.DEVNULL)
    shim = scratch / "shim"   # nsdev.so, bind-mounted into the box (a base image has no /usr/lib/harbor-ns)
    shim.mkdir()
    shutil.copy(sandbox.build_nsdev(cache), shim / "nsdev.so")
    pod = Pod(multiuid=True)
    try:
        loop.run_until_complete(pod.start())
    except Exception as e:  # noqa: BLE001
        loop.close()
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.skip(f"multi-uid pod unavailable here: {e}")
    b = Sandbox(cfg=cfg, lower=[rootfs], hostname="mu", pod=pod, internet=False, fsize=CAP,
                binds=[(str(shim), "/opt/shim")])
    loop.run_until_complete(b.start(timeout=60))
    try:
        yield b, pod, scratch, loop
    finally:
        loop.run_until_complete(b.stop(delete=True))
        loop.run_until_complete(pod.stop())
        loop.close()
        shutil.rmtree(scratch, ignore_errors=True)


def run(box, cmd, ids=None, timeout=60):
    b, _, _, loop = box
    return loop.run_until_complete(b.exec(cmd, ids=ids, timeout=timeout))


def test_box_is_multiuid(box):
    r = run(box, f"setpriv --reuid {UID} --regid {UID} --clear-groups id -u; cat /proc/self/uid_map")
    assert r.return_code == 0, r.stderr
    lines = r.stdout.split("\n")
    assert lines[0] == str(UID) and len([ln for ln in lines[1:] if ln.strip()]) > 1   # subordinate bands mapped


# ---- 1. RLIMIT_FSIZE after a privilege drop

@pytest.mark.parametrize("how", ["setpriv", "su", "exec_user"])
def test_dd_as_a_box_user_stops_at_the_cap(box, how):
    f = f"/tmp/dd-{how}"
    dd = f"dd if=/dev/zero of={f} bs=1M count=64 status=none"
    run(box, f"rm -f {f}")
    if how == "setpriv":
        r = run(box, f"setpriv --reuid {UID} --regid {UID} --clear-groups {dd}; echo rc=$?")
    elif how == "su":
        if run(box, "command -v su").return_code:
            pytest.skip("no su on this host")
        r = run(box, f"su -s /bin/sh nobody -c '{dd}'; echo rc=$?")
        if "rc=1\n" in r.stdout and "dd" not in r.stderr:   # PAM refused the switch (host config), not the cap
            pytest.skip(f"su unusable in the box: {r.stderr.strip()[-200:]}")
    else:   # Harbor's exec(user=...): nsenter -S/-G, the same prlimit prefix
        r = run(box, f"{dd}; echo rc=$?", ids=(UID, UID))
    assert "rc=153" in r.stdout, r.stdout + r.stderr   # 128 + SIGXFSZ
    st = run(box, f"stat -c '%s %u' {f}").stdout.split()
    assert int(st[0]) == CAP and st[1] != "0"           # written by the box user, stopped at the cap


def test_python_as_a_box_user_gets_efbig_and_cannot_raise_the_limit(box):
    code = ("import errno, resource\n"
            "f = open('/tmp/py-efbig', 'wb')\n"
            "try:\n"
            "    f.write(b'x' * (16 << 20)); f.flush(); print('wrote')\n"
            "except OSError as e:\n"
            "    print('EFBIG' if e.errno == errno.EFBIG else repr(e))\n"
            "try:\n"
            "    resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY,) * 2); print('raised')\n"
            "except (OSError, ValueError):\n"
            "    print('EPERM')\n"
            "print(*resource.getrlimit(resource.RLIMIT_FSIZE))\n")
    r = run(box, f"setpriv --reuid {UID} --regid {UID} --clear-groups python3 -c \"{code}\"")
    assert r.stdout.split() == ["EFBIG", "EPERM", str(CAP), str(CAP)], r.stdout + r.stderr


# ---- 3. the documented gap: nsdev clamps RLIMIT_CORE only

def test_ulimit_f_unlimited_fails_with_eperm_under_nsdev(box):
    """`ulimit -c unlimited` is clamped by nsdev (succeeds); `ulimit -f unlimited` is not (EPERM): the hard limit is
    the cap and nobody in the box can raise it.  No current task raises it (MULTIUID.md, diskcap.py)."""
    r = run(box, "LD_PRELOAD=/opt/shim/nsdev.so bash -c 'ulimit -c unlimited; echo core=$?; "
                 "ulimit -f unlimited; echo fsize=$?; ulimit -f'")
    assert "core=0" in r.stdout, r.stdout + r.stderr
    assert "fsize=1" in r.stdout and "not permitted" in r.stderr.lower(), r.stdout + r.stderr
    assert r.stdout.split()[-1] == str(CAP // 1024)   # bash reports 1024-byte blocks: unchanged


# ---- 2. the watchdog: private subordinate-id dirs, subordinate-id writers

def test_walk_counts_private_subordinate_id_dirs_and_kills_their_writer(box):
    b, pod, _, loop = box
    upper = b.dir / "upper"
    run(box, "rm -f /tmp/dd-* /tmp/py-efbig")                # the other tests' files
    # a box user's private dir (0700) with 6 MiB in it, owned by a subordinate id on the host
    r = run(box, f"mkdir -p /srv/private && chown {UID}:{UID} /srv/private && "
                 f"setpriv --reuid {UID} --regid {UID} --clear-groups sh -c "
                 "'chmod 700 /srv/private && mkdir /srv/private/deep && chmod 700 /srv/private/deep && "
                 "dd if=/dev/urandom of=/srv/private/deep/blob bs=1M count=6 status=none'")
    assert r.return_code == 0, r.stderr
    host = upper / "srv/private"
    assert host.stat().st_uid != os.getuid()
    with pytest.raises(PermissionError):
        os.listdir(host)                                    # the host user cannot read it ...
    as_host = diskcap.tree_usage([("main", upper)])
    assert as_host["unreadable"] >= 1 and as_host["bytes"] < 6 << 20
    ns = diskcap.NsView(pod.enter_argv)                     # ... namespace root can
    u = ns.tree_usage([("main", upper)], CAP)
    assert "fallback" not in u and u["unreadable"] == 0 and u["bytes"] >= 6 << 20
    assert u["largest"] == (6 << 20, "main:/srv/private/deep/blob")

    # a writer running as the box user, rewriting one block of its file in its private dir
    writer = ("import os, time\n"
              "fd = os.open('/srv/private/w', os.O_WRONLY | os.O_CREAT)\n"
              "b = b'x' * (1 << 20)\n"
              "while True:\n"
              "    os.pwrite(fd, b, 0); os.fsync(fd); time.sleep(0.02)\n")
    bg = loop.run_until_complete(b.spawn_background(["python3", "-c", writer], ids=(UID, UID)))
    try:
        pid = None
        for _ in range(100):
            ms = diskcap.members([b.init_pid])
            pids = [p for p in ms if open(f"/proc/{p}/cmdline", "rb").read().startswith(b"python3")]
            if pids and (ns.write_bytes(pids).get(pids[0]) or 0) > 0:
                pid = pids[0]
                break
            time.sleep(0.1)
        assert pid is not None, "writer not found in the box's pid namespace"
        uid = int(next(ln for ln in open(f"/proc/{pid}/status") if ln.startswith("Uid:")).split()[1])
        assert uid != os.getuid() and uid != 0                # a subordinate id on the host
        assert diskcap.write_bytes(pid) is None               # its /proc io is not the host user's to read
        logs = []
        # ~7 MiB in the box: above a 5.5 MiB cap, below its 8.25 MiB ceiling, so the writer goes, not everything
        cap = diskcap.DiskCap(lambda: [("main", upper)], 5.5, inits=lambda: [b.init_pid], fsize=CAP,
                              log=logs.append, ns=ns)
        killed = cap.check()
        assert killed == pid, logs
        for _ in range(100):
            if not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("the subordinate-id writer survived SIGKILL")
        assert cap.kills[0]["why"] == "writer" and ns.fallbacks == 0 and cap.summary()["ns_helper_fallbacks"] == 0
        assert b.init_pid and os.path.exists(f"/proc/{b.init_pid}")   # init untouched
    finally:
        if bg.returncode is None:
            try:
                os.killpg(bg.pid, 9)
            except ProcessLookupError:
                pass
        loop.run_until_complete(bg.wait())


def test_helper_falls_back_to_the_host_view_when_the_namespace_is_gone(tmp_path):
    (tmp_path / "f").write_bytes(b"x" * 8192)
    logs = []
    ns = diskcap.NsView(lambda: [], log=logs.append)
    u = ns.tree_usage([("main", tmp_path)])
    assert u["fallback"] == "RuntimeError" and u["bytes"] >= 8192 and len(logs) == 1
    ns.tree_usage([("main", tmp_path)])
    assert len(logs) == 1 and ns.fallbacks == 2             # logged once per kind of failure, every one counted
    assert ns.write_bytes([os.getpid()])[os.getpid()] >= 0 and len(logs) == 2
    bad = diskcap.NsView(lambda: ["false"], log=logs.append)
    assert bad.tree_usage([("main", tmp_path)])["fallback"] == "RuntimeError"


def test_helper_in_a_plain_user_namespace(tmp_path):
    """The helper path itself (no pod): `unshare -U -r` as the prefix, same answers as in-process."""
    if subprocess.run(["unshare", "-U", "-r", "true"]).returncode:
        pytest.skip("no unprivileged user namespaces")
    (tmp_path / "f").write_bytes(b"x" * (1 << 20))
    ns = diskcap.NsView(lambda: ["unshare", "-U", "-r", "--"])
    u = ns.tree_usage([("main", tmp_path)], 4096)
    assert "fallback" not in u and u["bytes"] == diskcap.tree_usage([("main", tmp_path)])["bytes"]
    assert u["largest"] == (1 << 20, "main:/f") and u["at_fsize"] == ["main:/f"]
    assert set(ns.write_bytes([os.getpid(), 2**22 + 7])) <= {os.getpid()} and ns.fallbacks == 0


def test_helper_timeout_raises_instead_of_walking_in_process(tmp_path, monkeypatch):
    """A helper walk that times out is not repeated in this process (it would hold the cap's lock in stop())."""
    monkeypatch.setattr(diskcap, "HELPER_TIMEOUT", 0.5)
    ns = diskcap.NsView(lambda: ["sh", "-c", "sleep 5", "x"], log=lambda m: None)
    with pytest.raises(subprocess.TimeoutExpired):
        ns.tree_usage([("main", tmp_path)])
    assert ns.fallbacks == 1


def test_no_rate_across_a_change_of_view(tmp_path):
    """write_bytes samples from the host view (helper fallback) and the namespace view are not compared: a
    subordinate-id pid missing from one would otherwise rate at its whole total and be killed as a writer."""
    up = tmp_path / "upper"
    up.mkdir()
    (up / "f").write_bytes(b"x" * (2 << 20))
    views = iter(["host", "ns", "ns"])
    totals = iter([{}, {4242: 10 << 30}, {4242: 10 << 30}])   # an idle daemon: 10 GiB written over its lifetime

    class Stub:
        fallbacks, io_view = 0, "ns"

        def tree_usage(self, roots, fsize=None):
            return diskcap.tree_usage(roots, fsize)

        def write_bytes(self, pids):
            self.io_view = next(views)
            return next(totals)

    killed = []
    cap = diskcap.DiskCap(lambda: [("main", up)], 1.5, members_fn=lambda: [4242], kill=killed.append,
                          comm=lambda p: "daemon", log=lambda m: None, ns=Stub())
    assert cap.check() is None   # over the cap (2 MB > 1.5 MB, below the ceiling); host then ns sample: no rate
    time.sleep(0.05)
    assert cap.check() is None and not killed   # same view, no growth: not a writer

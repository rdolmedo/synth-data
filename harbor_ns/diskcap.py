"""Per-sandbox disk cap: ``--ek ns_diskcap=<factor>`` or ``HARBOR_NS_DISKCAP=<factor>`` (off here unless set).  Who
sets it: eval_q35 by default, per benchmark and task (eval_q35/common.DISKCAP_FAMILIES, #425); eval/eval_job.sh factor 3,
HARBOR_NS_DISKCAP_MIN_MB 30720 (#372); synth campaigns from their frozen [harbor] (synth/sandbox_limits.py, #378); the
AA harnesses (aa_evals pi_eval and harbor_mswea) per run config ``"diskcap"`` once #405's aa_evals/common/diskcap.py
lands.  Multi-uid support: #405.

A sandbox writes into its overlay upper dir on the node-local scratch that every sandbox of a worker shares: harbor_ns
has no per-box filesystem (unprivileged: no loop mounts, no project quotas; a tmpfs would count against memory).  So one
runaway writer fills the scratch and breaks every other trial on the worker.  TB2.1-easy, 2026-11: agents on
qemu-alpine-ssh ran ``dd if=/dev/zero of=/app/alpine-disk.qcow2 bs=1M count=1048576``, a worker's 200 GB scratch filled
in minutes and the other sandboxes' agents crashed (~40% NonZeroAgentExitCodeError), recorded as model failures.

The cap is ``storage_mb x factor`` MB (task.toml [environment]; every TB2.1 task declares 10240), enforced twice:

1. Per file, by the kernel.  Every process of the box runs with ``RLIMIT_FSIZE`` = the cap, soft = hard (root in a user
   namespace cannot raise a hard limit): ``prlimit`` wraps the box's ``unshare`` and every ``nsenter`` (sandbox.py), so
   the agent, its commands, services and the verifier all inherit it.  A write past the cap fails at once with EFBIG and
   the writer gets SIGXFSZ ("File size limit exceeded"; Python ignores the signal and raises OSError), the closest
   unprivileged analogue of a quota error.  This alone stops the dd case at the cap, with no polling.
2. In total, by this watchdog: the data of the box's upper dir(s) plus the trial's compose volumes, per file its
   allocated size (``st_blocks``) but at most its apparent size (sparse files count what they hold; XFS's speculative
   preallocation past EOF of a growing file, up to 8 GiB and trimmed when it is closed, does not count).  A full walk
   costs ~5 us per entry (0.47 s for 100k files), so it is not run on a timer alone: every ``POLL`` s a ``statvfs``
   of the scratch filesystem (one syscall) tells how much the whole filesystem grew since the last walk, and the box
   is walked when that growth reaches half of its remaining headroom to the cap (above the cap: to the ceiling below;
   at least ``TRIGGER`` bytes), at least every ``MAX_GAP`` s, and never more often than ``DUTY`` of one core unless
   the filesystem grew by more than the box's whole headroom (at least ``FORCE`` bytes: it may be past it).  Above
   the cap it SIGKILLs the process writing the most (write_bytes of its threads, ``/proc/<pid>/task/*/io``, at least
   ``MIN_RATE`` bytes/s since the last sample; ``/proc/<pid>/io`` also counts reaped children, so a shell would look
   like the writer once its dd exits), never the box's init, as the memory cap kills the largest process (memcap.py).
   The agent (``comm`` in ``SPARE``, default ``pi``; ``HARBOR_NS_DISKCAP_SPARE`` = comma list) is killed only when it
   is the only writer.  Killing frees nothing: the files stay until something in the box deletes them, as on a full
   disk, and a box above the cap that writes nothing is left alone.  A loop that keeps starting short writers (``for
   i in ...; do dd ... of=part$i; done``) is not stopped by killing them one at a time, so a box that reaches
   ``CEILING`` x the cap is runaway: every process in it but init is killed once, the agent too (the trial then ends
   as the memory cap ends one whose agent is the last process left), and from then on writers again.  So no box takes
   much more than ``CEILING`` x the cap of the scratch.

The value is a factor as for memcap: ``1``/``true``/``on`` = 1.0, ``2`` = twice ``storage_mb``, ``0``/``off``/empty
disables it.  Tasks that declare no ``storage_mb`` are uncapped unless ``HARBOR_NS_DISKCAP_MIN_MB`` is set: then every
cap is at least that many MB, and undeclared tasks get exactly it.  ``HARBOR_NS_DISKCAP_LOG``: a file that gets the
box's limit, peak, largest file and kills as JSON when the sandbox stops (a directory: one ``<session_id>.json`` per
sandbox, for runs of many trials in one Harbor process).  The per-file limit applies to the logical size, so a sparse
file (``truncate -s``) larger than the cap fails too.

Multi-uid pods (MULTIUID.md): box processes and files may belong to subordinate ids.  The per-file limit is unchanged
(rlimits survive ``setuid``/``setpriv``/``su``, and nobody in the box can raise a hard limit).  The host user cannot
read a directory a subordinate id made private (mode 0700), nor ``/proc/<pid>/task/*/io`` of a process running as one
(mode 0400, owned by that id), so walks and write_bytes samples run in a helper process as root of the pod's user
namespace (``NsView``: ``nsenter -U`` into the pod, as ``Pod.purge`` does), which reads every mapped id's files; a
failed helper falls back to the host user's view.  Finding the box's processes (``/proc/<pid>/ns/pid``) and killing
them need no helper: the host user owns the pod's user namespace, so it holds every capability over its processes.
Known gap, as in single-uid boxes: nsfake/nsdev clamp only ``RLIMIT_CORE``, so ``ulimit -f unlimited`` in a box fails
with EPERM (no current task raises it).
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .memcap import _comm, parse  # noqa: F401  (parse: the same factor syntax as ns_memcap)

POLL = 0.25              # statvfs period (s)
OVER_POLL = 0.5          # above the cap: writer sampling period (s)
MIN_GAP = 0.5            # never walk more often than this (s) ...
DUTY = 0.05              # ... or than this share of one core (gap >= walk time / DUTY)
MAX_GAP = 60.0           # walk at least this often (s): the statvfs trigger misses growth hidden by deletions elsewhere
OVER_GAP = 5.0           # above the cap: re-walk this often (s), to see deletions bring the box back under
PROBE = 0.25             # first pass above the cap: write_bytes twice this far apart, so the writer goes at once
TRIGGER = 256 << 20      # filesystem growth (bytes) that always allows a walk, however close the box is to its cap
FORCE = 1 << 30          # growth past the box's whole headroom (and at least this) walks it whatever DUTY says
MIN_RATE = 2 << 20       # bytes/s of write_bytes that make a process a writer
CEILING = 1.5            # x the cap: a runaway box, everything in it but init is killed once
SPARE = tuple(c for c in os.environ.get("HARBOR_NS_DISKCAP_SPARE", "pi").split(",") if c)


def limit_mb(storage_mb: float | None, factor: float | None, min_mb: float | None = None) -> float | None:
    """The cap of a box: storage_mb x factor, raised to min_mb (env HARBOR_NS_DISKCAP_MIN_MB when None); min_mb alone for
    a task without storage_mb; None (no cap) when the cap is off or there is nothing to cap at."""
    if not factor:
        return None
    if min_mb is None:
        try:
            min_mb = float(os.environ.get("HARBOR_NS_DISKCAP_MIN_MB") or 0)
        except ValueError:
            min_mb = 0.0
    if storage_mb:
        return max(storage_mb * factor, min_mb or 0)
    return min_mb or None


def fsize_bytes(mb: float | None) -> int | None:
    """RLIMIT_FSIZE for a cap of mb MB, clamped to this process's own hard limit (prlimit cannot raise it)."""
    if not mb:
        return None
    n = int(mb * 2**20)
    hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]
    return n if hard == resource.RLIM_INFINITY else min(n, hard)


def tree_usage(roots: Iterable[tuple[str, Path]], fsize: int | None = None) -> dict:
    """Bytes of data under each (label, dir): per file min(allocated, apparent size rounded up to 4 KiB), hard links
    counted once, directories by allocated size; the largest file by that and by apparent size (as "<label>:<path in the
    box>"), files at the per-file limit, and directories that could not be read.  Directories are opened with
    O_NOFOLLOW and must be on the root's filesystem, so a directory the box swaps for a symlink between the listing and
    the open is not followed out of the scratch (the multi-uid helper walks in the host mount namespace)."""
    total = files = unreadable = 0
    big, size = (0, None), (0, None)
    at_fsize: list[str] = []
    seen: set[tuple[int, int]] = set()
    for label, root in roots:
        root = str(root)
        try:
            dev = os.stat(root).st_dev
        except FileNotFoundError:
            continue
        except OSError:
            unreadable += 1
            continue
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            except FileNotFoundError:
                continue
            except OSError:   # e.g. a dir the box chmod'ed 000, or one swapped for a symlink (ELOOP / ENOTDIR)
                unreadable += 1
                continue
            try:
                if os.fstat(fd).st_dev != dev:   # a parent swapped for a symlink out of the scratch
                    unreadable += 1
                    continue
                with os.scandir(fd) as it:
                    for e in it:
                        path = f"{d}/{e.name}"
                        try:
                            st = e.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if stat.S_ISDIR(st.st_mode):
                            total += st.st_blocks * 512
                            stack.append(path)
                            continue
                        if st.st_nlink > 1:
                            k = (st.st_dev, st.st_ino)
                            if k in seen:
                                continue
                            seen.add(k)
                        b = min(st.st_blocks * 512, -(-st.st_size // 4096) * 4096)   # no post-EOF preallocation
                        total += b
                        files += 1
                        if b > big[0]:
                            big = (b, f"{label}:{path[len(root):]}")
                        if st.st_size > size[0]:
                            size = (st.st_size, f"{label}:{path[len(root):]}")
                        if fsize and st.st_size >= fsize and len(at_fsize) < 5:
                            at_fsize.append(f"{label}:{path[len(root):]}")
            except OSError:
                unreadable += 1
            finally:
                os.close(fd)
    return {"bytes": total, "files": files, "largest": big, "largest_size": size, "at_fsize": at_fsize,
            "unreadable": unreadable}


HELPER_TIMEOUT = 600.0   # s: a namespace helper (multi-uid) that takes longer is abandoned for the host user's view
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
_HELPER_CODE = ("import json, sys; sys.path.insert(0, sys.argv[1]); from harbor_ns import diskcap; "
                "print(json.dumps(getattr(diskcap, sys.argv[2])(*json.loads(sys.argv[3]))))")


class NsView:
    """The watchdog's reads done by a helper process started under the argv prefix enter() (multi-uid: ``nsenter -t
    <pod holder> -U --preserve-credentials --``, root of the pod's user namespace).  The host user cannot read a
    directory a subordinate id made private, nor ``/proc/<pid>/task/*/io`` of a process running as one (mode 0400,
    owned by that id); namespace root can.  A failed helper (no prefix, nsenter error, timeout, bad output) falls back
    to the host user's view, logged once per kind of failure, and is counted in ``fallbacks``."""

    def __init__(self, enter: Callable[[], list[str]], log: Callable[[str], object] = print):
        self.enter = enter
        self.log = log
        self.fallbacks = 0
        self.io_view = "ns"   # where the last write_bytes sample came from: "ns" or "host" (a fallback)
        self._warned: set[tuple[str, str]] = set()

    def _run(self, fn: str, *args):
        prefix = self.enter()
        if not prefix:
            raise RuntimeError("no namespace to enter")
        r = subprocess.run([*prefix, sys.executable, "-I", "-c", _HELPER_CODE, _PKG_PARENT, fn, json.dumps(args)],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=HELPER_TIMEOUT)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[-300:]}")
        return json.loads(r.stdout)

    def _failed(self, what: str, e: Exception) -> str:
        self.fallbacks += 1
        kind = type(e).__name__
        if (what, kind) not in self._warned:
            self._warned.add((what, kind))
            self.log(f"harbor_ns diskcap: {what} in the pod's user namespace failed ({e!r})"
                     + ("" if isinstance(e, subprocess.TimeoutExpired) else "; as the host user instead"))
        return kind

    def tree_usage(self, roots: Iterable[tuple[str, Path]], fsize: int | None = None) -> dict:
        roots = [(str(label), str(r)) for label, r in roots]
        try:
            u = self._run("tree_usage", roots, fsize)
            u["largest"], u["largest_size"] = tuple(u["largest"]), tuple(u["largest_size"])
            return u
        except subprocess.TimeoutExpired as e:   # a tree too big or too slow: walking it here would hang stop()
            self._failed("walk", e)
            raise
        except Exception as e:  # noqa: BLE001  (never lose the walk)
            return {**tree_usage(roots, fsize), "fallback": self._failed("walk", e)}

    def write_bytes(self, pids: Iterable[int]) -> dict[int, int]:
        pids = list(pids)
        if not pids:
            return {}
        try:
            out = {int(p): b for p, b in self._run("write_bytes_of", pids).items()}
            self.io_view = "ns"
            return out
        except Exception as e:  # noqa: BLE001
            self._failed("write_bytes", e)
            self.io_view = "host"
            return write_bytes_of(pids)


def write_bytes(pid: int) -> int | None:
    """Bytes the live threads of the process have caused to be written to storage (/proc/<pid>/task/*/io), None when
    gone or unreadable.  Not /proc/<pid>/io: it adds the counts of reaped children (and dead threads)."""
    total, seen = 0, False
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return None
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/io") as f:
                for ln in f:
                    if ln.startswith("write_bytes:"):
                        total += int(ln.split()[1])
                        seen = True
                        break
        except (OSError, ValueError, IndexError):
            continue
    return total if seen else None


def write_bytes_of(pids: Iterable[int]) -> dict[int, int]:
    """write_bytes of each pid that is still there and readable."""
    out = {}
    for p in pids:
        b = write_bytes(p)
        if b is not None:
            out[p] = b
    return out


def members(init_pids: Iterable[int]) -> list[int]:
    """Host pids of the processes in the pid namespaces of init_pids, the inits excluded (one scan of /proc)."""
    inits = set(init_pids)
    ns = set()
    for p in inits:
        try:
            ns.add(os.readlink(f"/proc/{p}/ns/pid"))
        except OSError:
            pass
    if not ns:
        return []
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in inits:
            continue
        try:
            if os.readlink(f"/proc/{d}/ns/pid") in ns:
                out.append(int(d))
        except OSError:   # gone, or not ours to inspect
            continue
    return out


def _mb(b: float) -> float:
    return round(b / 2**20, 1)


class DiskCap:
    """Watch one trial's sandbox(es); while their upper dirs are above limit_mb, kill the process writing the most."""

    def __init__(self, roots: Callable[[], Iterable[tuple[str, Path]]], limit_mb: float, *,
                 inits: Callable[[], Iterable[int]] = lambda: (), fsize: int | None = None,
                 log: Callable[[str], object] = print, kill: Callable[[int], object] | None = None,
                 spare: Iterable[str] = SPARE, comm: Callable[[int], str] | None = None,
                 members_fn: Callable[[], Iterable[int]] | None = None,
                 fs_used: Callable[[], int | None] | None = None, clock: Callable[[], float] = time.monotonic,
                 ns: NsView | None = None):
        self.limit = int(limit_mb * 2**20)
        self.ns = ns   # multi-uid: walks and write_bytes as root of the pod's user namespace
        self.fsize = fsize
        self.roots = roots
        self.log = log
        self.spare = frozenset(spare)
        self._kill = kill or (lambda pid: os.kill(pid, signal.SIGKILL))
        self._comm = comm or _comm
        self._members = members_fn or (lambda: members(inits()))
        self._fs_used = fs_used or self._statvfs_used
        self._clock = clock
        self._lock = threading.Lock()
        self.total = self.peak = self.peak_files = 0
        self.largest: tuple[int, str | None] = (0, None)
        self.largest_size: tuple[int, str | None] = (0, None)
        self.at_fsize: list[str] = []
        self.unreadable = 0
        self.walks = 0
        self.walk_s = self.walk_s_max = self.walk_s_total = 0.0
        self.last_walk: float | None = None
        self.used_at_walk: int | None = None
        self.over_since: float | None = None
        self.over_s = 0.0
        self.final_bytes: int | None = None
        self.kills: list[dict] = []
        self._io: tuple[float, dict[int, int], str] | None = None   # (time, write_bytes, view)
        self._ceiling_hit = False

    @property
    def over(self) -> bool:
        return self.total > self.limit

    def _statvfs_used(self) -> int | None:
        for _, r in self.roots():
            try:
                s = os.statvfs(r)
                return (s.f_blocks - s.f_bfree) * s.f_frsize
            except OSError:
                continue
        return None

    def due(self, now: float, used: int | None) -> bool:
        """Walk now? (see the module docstring)"""
        if self.last_walk is None:
            return True
        gap = now - self.last_walk
        if gap < MIN_GAP:
            return False
        target = self.limit * CEILING if self.over else self.limit   # above the cap, watch for the ceiling
        grown = None if used is None or self.used_at_walk is None else used - self.used_at_walk
        # The scratch grew by more than the box's whole headroom, so the box may be past it: walk despite DUTY (a 2.5 s
        # walk would otherwise wait 50 s, long enough for a loop of writers to put ~100 GB past the cap).
        if grown is not None and self.total <= target and grown >= max(target - self.total, FORCE):
            return True
        if gap < self.walk_s / DUTY:
            return False
        if gap >= MAX_GAP or grown is None or (self.over and gap >= OVER_GAP):
            return True
        if self.total > target:
            return False
        return grown >= max((target - self.total) / 2, TRIGGER)

    def walk(self, now: float | None = None, used: int | None = None) -> dict:
        t0 = time.perf_counter()
        u = (self.ns.tree_usage if self.ns else tree_usage)(self.roots(), self.fsize)
        dt = time.perf_counter() - t0
        self.walks += 1
        self.walk_s, self.walk_s_max, self.walk_s_total = dt, max(self.walk_s_max, dt), self.walk_s_total + dt
        self.last_walk = self._clock() if now is None else now
        self.used_at_walk = used
        self.total = u["bytes"]
        if u["bytes"] > self.peak:
            self.peak, self.peak_files = u["bytes"], u["files"]
        if u["largest"][0] > self.largest[0]:
            self.largest = u["largest"]
        if u["largest_size"][0] > self.largest_size[0]:
            self.largest_size = u["largest_size"]
        self.at_fsize = sorted(set(self.at_fsize) | set(u["at_fsize"]))[:5]
        self.unreadable = max(self.unreadable, u["unreadable"])
        return u

    def check(self) -> int | None:
        """One pass: the pid killed, or None."""
        with self._lock:
            now = self._clock()
            used = self._fs_used()
            if self.due(now, used):
                self.walk(now, used)
            if not self.over:
                if self.over_since is not None:
                    self.over_s += now - self.over_since
                    self.over_since = None
                self._io = None
                self._ceiling_hit = False
                return None
            if self.over_since is None:
                self.over_since = now
                self.log(f"harbor_ns diskcap: sandbox at {_mb(self.total):.0f} MB > limit {_mb(self.limit):.0f} MB "
                         f"(largest file {self.largest[1]}, {_mb(self.largest[0]):.0f} MB)")
            if self.total > self.limit * CEILING and not self._ceiling_hit:
                self._ceiling_hit = True
                return self._kill_all()
            return self._kill_writer(self._clock())   # after the walk: rates over the sampling interval only

    def _kill_all(self) -> int | None:
        killed = []
        for p in self._members():
            comm = self._comm(p)
            try:
                self._kill(p)
            except ProcessLookupError:
                continue
            killed.append(p)
            self.kills.append({"t": time.time(), "pid": p, "comm": comm, "why": "ceiling", "total_mb": _mb(self.total)})
        self._io = None
        self.log(f"harbor_ns diskcap: sandbox at {_mb(self.total):.0f} MB > {CEILING:g} x limit {_mb(self.limit):.0f} MB "
                 f"(runaway); killed all {len(killed)} processes but init")
        return killed[0] if killed else None

    def _sample(self) -> tuple[dict[int, int], str]:
        """write_bytes of the box's processes and the view they came from ("ns" or "host"): a sample taken in another
        view than the previous one cannot be compared with it (subordinate-id pids missing from the host view would
        rate at their whole total)."""
        if self.ns is None:
            return write_bytes_of(self._members()), "host"
        cur = self.ns.write_bytes(self._members())
        return cur, self.ns.io_view

    def _kill_writer(self, now: float) -> int | None:
        if self._io is None:   # the first pass above the cap: rates need two samples
            self._io = (now, *self._sample())
            time.sleep(PROBE)
            now = self._clock()
        prev = self._io
        cur, view = self._sample()
        self._io = (now, cur, view)
        if view != prev[2]:   # the samples came from different views (a helper fallback): rate from the next pair
            return None
        dt = max(now - prev[0], 1e-3)
        writers = {p: r for p, r in ((p, (b - prev[1].get(p, 0)) / dt) for p, b in cur.items()) if r >= MIN_RATE}
        if not writers:
            return None
        comms = {p: self._comm(p) for p in writers}
        others = [p for p in writers if comms[p] not in self.spare]
        pid = max(others or writers, key=writers.__getitem__)   # the agent only when it is the only writer
        try:
            self._kill(pid)
        except ProcessLookupError:
            return None
        cur.pop(pid, None)
        self.kills.append({"t": time.time(), "pid": pid, "comm": comms[pid], "why": "writer", "mb_per_s": _mb(writers[pid]),
                           "total_mb": _mb(self.total)})
        self.log(f"harbor_ns diskcap: sandbox at {_mb(self.total):.0f} MB > limit {_mb(self.limit):.0f} MB; "
                 f"killed pid {pid} ({comms[pid]}, writing {_mb(writers[pid]):.0f} MB/s)")
        return pid

    def final(self) -> None:
        """A last walk when the sandbox stops (its end state, after the verifier)."""
        with self._lock:
            self.walk()
            self.final_bytes = self.total
            if self.over_since is not None:
                self.over_s += self._clock() - self.over_since
                self.over_since = None

    def summary(self) -> dict:
        return {"limit_mb": _mb(self.limit), "fsize_mb": _mb(self.fsize) if self.fsize else None,
                "ceiling_mb": _mb(self.limit * CEILING),
                "peak_mb": _mb(self.peak), "peak_files": self.peak_files,
                "final_mb": _mb(self.final_bytes) if self.final_bytes is not None else None,
                "largest_file_mb": _mb(self.largest[0]), "largest_file": self.largest[1],
                "largest_size_mb": _mb(self.largest_size[0]), "largest_size_file": self.largest_size[1],
                "at_fsize": self.at_fsize, "over_s": round(self.over_s, 1), "walks": self.walks,
                "walk_s_max": round(self.walk_s_max, 3), "walk_s_total": round(self.walk_s_total, 2),
                "unreadable_dirs": self.unreadable, "kills": self.kills,
                **({"ns_helper_fallbacks": self.ns.fallbacks} if self.ns else {})}

    async def run(self) -> None:
        while True:
            try:
                killed = await asyncio.to_thread(self.check)
            except Exception as e:  # noqa: BLE001  (never take the trial down)
                self.log(f"harbor_ns diskcap: check failed: {e!r}")
                killed = None
            await asyncio.sleep(0.2 if killed else OVER_POLL if self.over else POLL)

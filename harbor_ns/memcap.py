"""Opt-in per-sandbox memory cap: ``--ek ns_memcap=<factor>`` or ``HARBOR_NS_MEMCAP=<factor>`` (default off).

harbor_ns creates no cgroup of its own: every sandbox of a worker shares the batch job's cgroup, so a task's
``memory_mb`` is not enforced.  One runaway process can push the whole job over its ``request_memory``; HTCondor
then holds the job and every trial on it is lost (TB-Science round 2: an 8-slot worker held at 261 GB while its
tasks declared 55 GB in total).

This watchdog stands in for a container's ``memory.max``.  Every ``interval`` seconds it sums the unreclaimable
memory of the processes in the main sandbox's pid namespace: ``Pss_Anon + Pss_Shmem`` from
``/proc/<pid>/smaps_rollup``, or ``RssAnon + RssShmem`` from ``/proc/<pid>/status`` if that is unreadable.  Page
cache is left out, as a cgroup would reclaim it first.  Above ``memory_mb * factor`` it SIGKILLs the largest
process, never the box's init, as the kernel OOM killer does inside a container.  Then it checks again.

The agent's own process (``comm`` in ``SPARE``, default ``pi``; ``HARBOR_NS_MEMCAP_SPARE`` = comma list) is killed
only when nothing else is left to kill.  The agent runs inside the sandbox here, but in AA's harness (Terminus 2 in
E2B) it runs outside, so a task's memory overrun can never end the episode there.  Pss also splits shared pages
across forked workers, so with many workers the small, unshared agent can be the largest by Pss (TB-Science-lite
r4 medium, apo-holo-docking: 96 forked workers put the box at 12469 MB and the cap killed ``pi`` at 123 MB).

The value is a factor: ``1``/``true``/``on`` means 1.0 (Docker's ``mem_limit = memory_mb``); ``1.5`` gives 50%
headroom; ``0``/``false``/``off``/empty disables it.  Tasks that declare no ``memory_mb`` are not capped, unless
``HARBOR_NS_MEMCAP_MIN_MB`` is set: then every cap is at least that many MB, and undeclared tasks get exactly it
(eval_q35: half of TB-Lite declares nothing).  ``HARBOR_NS_MEMCAP_LOG``: a file that gets the box's limit, peak and
kills as JSON when the sandbox stops (per-attempt peaks for sizing worker requests).  A process that allocates faster than ``interval`` can still overshoot briefly, so keep slack in ``request_memory``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Callable, Iterable

INTERVAL = 1.0
SPARE = tuple(c for c in os.environ.get("HARBOR_NS_MEMCAP_SPARE", "pi").split(",") if c)


def parse(value: object) -> float | None:
    """--ek / env value -> factor, or None when off."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "0", "false", "no", "off"):
        return None
    if s in ("1", "true", "yes", "on"):
        return 1.0
    f = float(s)
    if f <= 0:
        return None
    return f


def limit_mb(memory_mb: float | None, factor: float | None, min_mb: float | None = None) -> float | None:
    """The cap of a box: memory_mb x factor, raised to min_mb (env HARBOR_NS_MEMCAP_MIN_MB when None); min_mb alone for
    a task without memory_mb; None (no cap) when the cap is off or there is nothing to cap at."""
    if not factor:
        return None
    if min_mb is None:
        try:
            min_mb = float(os.environ.get("HARBOR_NS_MEMCAP_MIN_MB") or 0)
        except ValueError:
            min_mb = 0.0
    if memory_mb:
        return max(memory_mb * factor, min_mb or 0)
    return min_mb or None


def pidns_members(init_pid: int) -> list[int]:
    """Host pids of the processes in init_pid's pid namespace, init excluded (nsenter'ed execs included)."""
    try:
        ns = os.readlink(f"/proc/{init_pid}/ns/pid")
    except OSError:
        return []
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == init_pid:
            continue
        try:
            if os.readlink(f"/proc/{d}/ns/pid") == ns:
                out.append(int(d))
        except OSError:  # gone, or not ours to inspect
            continue
    return out


def usage_kb(pid: int) -> int:
    """Unreclaimable memory of one process in kB (0 when it is gone or unreadable)."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            v = {k: int(r.split()[0]) for k, _, r in (ln.partition(":") for ln in f) if k in ("Pss_Anon", "Pss_Shmem")}
        if "Pss_Anon" in v:
            return v["Pss_Anon"] + v.get("Pss_Shmem", 0)
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            v = {k: int(r.split()[0]) for k, _, r in (ln.partition(":") for ln in f) if k in ("RssAnon", "RssShmem")}
        return v.get("RssAnon", 0) + v.get("RssShmem", 0)
    except (OSError, ValueError, IndexError):
        return 0


def _comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


class MemCap:
    """Watch one sandbox; kill its largest process while the sum is above limit_mb."""

    def __init__(self, init_pid: int | None, limit_mb: float, *, interval: float = INTERVAL,
                 members: Callable[[], Iterable[int]] | None = None, log: Callable[[str], object] = print,
                 kill: Callable[[int], object] | None = None, spare: Iterable[str] = SPARE,
                 comm: Callable[[int], str] | None = None):
        self.limit_kb = int(limit_mb * 1024)
        self.spare = frozenset(spare)
        self._comm = comm or _comm
        self.interval = interval
        self.members = members or (lambda: pidns_members(init_pid) if init_pid else [])
        self.log = log
        self._kill = kill or (lambda pid: os.kill(pid, signal.SIGKILL))
        self.kills: list[dict] = []
        self.peak_kb = 0

    def check(self) -> int | None:
        """One pass: the pid killed, or None."""
        use = {p: usage_kb(p) for p in self.members()}
        total = sum(use.values())
        self.peak_kb = max(self.peak_kb, total)
        if total <= self.limit_kb or not use:
            return None
        comms = {p: self._comm(p) for p in use}
        others = [p for p in use if comms[p] not in self.spare]
        pid = max(others or use, key=use.__getitem__)   # the agent only when nothing else is left
        comm = comms[pid]
        try:
            self._kill(pid)
        except ProcessLookupError:
            return None
        self.kills.append({"t": time.time(), "pid": pid, "comm": comm, "kb": use[pid], "total_kb": total})
        self.log(f"harbor_ns memcap: sandbox at {total // 1024} MB > limit {self.limit_kb // 1024} MB; "
                 f"killed pid {pid} ({comm}, {use[pid] // 1024} MB)")
        return pid

    def summary(self) -> dict:
        return {"limit_mb": self.limit_kb // 1024, "peak_mb": self.peak_kb // 1024, "kills": self.kills}

    async def run(self) -> None:
        while True:
            try:
                killed = await asyncio.to_thread(self.check)
            except Exception as e:  # noqa: BLE001  (never take the trial down)
                self.log(f"harbor_ns memcap: check failed: {e!r}")
                killed = None
            await asyncio.sleep(0.2 if killed else self.interval)
